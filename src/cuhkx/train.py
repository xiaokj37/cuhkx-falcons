"""Four-GPU DDP training with subject folds and cached teacher supervision."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .data import ClipDataset, prepare_vision, read_meta, seed_worker, to_device, warm_cache
from .models import build_model, classification_loss, classification_correct, distillation_loss
from .tracking import RunTracker, atomic_json, capture_rng, restore_rng
from .weights import state_to_cpu


def distributed_setup():
    world, rank, local_rank = (int(os.environ.get(k, d)) for k, d in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training configuration")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    return rank, world, device


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def barrier():
    if dist.is_initialized():
        dist.barrier()


def save_checkpoint(value, path):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def model_state(model):
    return state_to_cpu(unwrap(model).state_dict())


def split_indices(meta, fold, smoke=False):
    if meta["split"] != "train" and not (smoke and meta["split"] == "synthetic"):
        raise ValueError("Training accepts only labeled training cache; never test.csv predictions")
    records = meta["records"]
    if fold == 0:
        if {r["label"] for r in records} != set(range(40)) and not smoke:
            raise ValueError("Full training must cover all 40 classes")
        return list(range(len(records))), [], sorted({r["subject"] for r in records}), []
    train = [i for i, r in enumerate(records) if r["fold"] != fold]
    val = [i for i, r in enumerate(records) if r["fold"] == fold]
    if not train or not val:
        raise ValueError("Empty training or validation split")
    train_subjects = sorted({records[i]["subject"] for i in train})
    val_subjects = sorted({records[i]["subject"] for i in val})
    if set(train_subjects) & set(val_subjects):
        raise ValueError("Subject leakage detected")
    if not smoke:
        for indices in (train, val):
            if {records[i]["label"] for i in indices} != set(range(40)):
                raise ValueError("Every fold must cover all 40 classes")
    return train, val, train_subjects, val_subjects


def load_teacher(path, meta, fold, train_indices, train_subjects, val_subjects):
    path = Path(path)
    info = json.loads(path.with_suffix(".json").read_text())
    if (info["cache_signature"] != meta["signature"] or info["fold"] != fold
            or info["train_subjects"] != train_subjects or info["validation_subjects"] != val_subjects):
        raise ValueError("Teacher provenance does not match this subject fold")
    with np.load(path, allow_pickle=False) as values:
        if set(values["indices"].tolist()) != set(train_indices):
            raise ValueError("Teacher targets must cover exactly the training portion of this fold")
        result = {name: np.zeros((len(meta["records"]), values[name].shape[1]), dtype=np.float32) for name in ("logits", "features")}
        for name in result:
            if not np.isfinite(values[name]).all():
                raise ValueError("Teacher targets contain nonfinite values")
            result[name][values["indices"]] = values[name]
    return result


def make_loader(dataset, batch, workers, sampler=None, shuffle=False):
    options = dict(dataset=dataset, batch_size=batch, num_workers=workers, pin_memory=True,
                   sampler=sampler, shuffle=shuffle and sampler is None, worker_init_fn=seed_worker)
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**options)


@torch.inference_mode()
def evaluate(model, loader, config, device, rank, world):
    model = unwrap(model)
    model.eval()
    arrays = {k: [] for k in ("indices", "labels", "logits", "branch_logits", "weights", "availability")}
    if model.residual_fusion is not None:
        arrays.update(late_logits=[], interaction_logits=[])
    for batch in loader:
        batch = to_device(batch, device)
        vision = prepare_vision(batch, config["image_size"], False, config.get("visual_configs"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch, vision)
        values = dict(indices=batch["index"], labels=batch["label"], **{k: output[k] for k in ("logits", "branch_logits", "weights", "availability")})
        if model.residual_fusion is not None:
            values.update(late_logits=output["late_logits"], interaction_logits=output["interaction_logits"])
        for key, value in values.items():
            arrays[key].append(value.detach().float().cpu().numpy() if value.is_floating_point() else value.cpu().numpy())
    local = {k: np.concatenate(v) if v else None for k, v in arrays.items()}
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if rank != 0:
        return None, None
    combined = {k: np.concatenate([g[k] for g in gathered if g[k] is not None]) for k in arrays}
    order = np.argsort(combined["indices"])
    combined = {k: v[order] for k, v in combined.items()}
    if len(np.unique(combined["indices"])) != len(combined["indices"]):
        raise ValueError("Validation padding/duplicate rows detected")
    labels, logits = combined["labels"], combined["logits"]
    predicted = logits.argmax(-1)
    confusion = np.bincount(labels * 40 + predicted, minlength=1600).reshape(40, 40)
    tp = confusion.diagonal()
    f1 = 2 * tp / np.maximum(confusion.sum(0) + confusion.sum(1), 1)
    metrics = dict(accuracy=float(np.mean(predicted == labels)), macro_f1=float(f1.mean()),
                   loss=float(torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels).long())),
                   clips=len(labels), confusion_matrix=confusion.tolist(), branch_accuracy={}, fusion_weights={})
    if "late_logits" in combined:
        metrics["late_path_accuracy"] = float(np.mean(combined["late_logits"].argmax(-1) == labels))
        metrics["interaction_logit_rms"] = float(np.sqrt(np.mean(combined["interaction_logits"] ** 2)))
    for k, name in enumerate(model.active):
        valid = combined["availability"][:, k] > 0
        metrics["branch_accuracy"][name] = dict(clips=int(valid.sum()),
            accuracy=float(np.mean(combined["branch_logits"][valid, k].argmax(-1) == labels[valid])) if valid.any() else None)
        metrics["fusion_weights"][name] = float(combined["weights"][:, k].mean())
    return metrics, combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--fold", type=int, default=1, choices=(0, 1, 2, 3), help="0 = final full-data fit without validation")
    parser.add_argument("--out", required=True)
    parser.add_argument("--teacher-targets")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, help="Pause after saving this epoch; --resume continues the same schedule")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    dg_options = config.get("domain_generalization")
    prediction_options = config.get("masked_prediction")
    if prediction_options and (config.get("fusion") != "residual_bottleneck"
            or max(config.get("feature_mixup", 0.), config.get("input_mixup", 0.)) > 0):
        raise ValueError("Masked prediction requires unmixed residual bottleneck features")
    if dg_options and max(config.get("feature_mixup", 0.), config.get("input_mixup", 0.)) > 0:
        raise ValueError("Cross-user losses require unmixed source actions and identities")
    if args.teacher_targets and max(config.get("feature_mixup", 0.), config.get("input_mixup", 0.)) > 0:
        raise ValueError("MixUp requires mixed teacher targets; this supervised configuration excludes KD")
    rank, world, device = distributed_setup()
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    seed = config.get("seed", 20260906) + args.fold * 100
    torch.manual_seed(seed)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    meta = read_meta(args.cache)
    from .data import validate_preprocessing
    preprocessing = validate_preprocessing(meta, config)
    train_indices, val_indices, train_subjects, val_subjects = split_indices(meta, args.fold, args.smoke)
    from .sensor_initialization import source_sensor_identity, initialize_source_sensor, restore_source_sensor_audit
    initialization = source_sensor_identity(config, meta,
        np.load(Path(args.cache) / "availability.npy", mmap_mode="r"), args.fold, args.smoke)
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        import fcntl
        training_lock = (out / ".training.lock").open("w")
        fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "run.json").exists() and not args.resume:
            raise ValueError("Experiment already exists. Choose a new output or use --resume")
    barrier()
    provenance = dict(config=config, cache=str(Path(args.cache).resolve()), cache_signature=meta["signature"], fold=args.fold,
                      train_subjects=train_subjects, validation_subjects=val_subjects, train_clips=len(train_indices),
                      validation_clips=len(val_indices), world_size=world, synthetic_smoke=args.smoke,
                      teacher_targets=args.teacher_targets, preprocessing=preprocessing)
    import subprocess
    try:
        provenance["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        provenance["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        provenance["git_commit"], provenance["git_dirty"] = None, None
    teacher = load_teacher(args.teacher_targets, meta, args.fold, train_indices, train_subjects, val_subjects) if args.teacher_targets else None
    if rank == 0:
        if config.get("warm_cache", True):
            print("Warming shared page cache:", warm_cache(args.cache), "bytes", flush=True)
    barrier()
    training = ClipDataset(args.cache, train_indices, config["frames"], training=True, teacher=teacher)
    validation = ClipDataset(args.cache, val_indices[rank::world], config["frames"], training=False)
    sampler = DistributedSampler(training, num_replicas=world, rank=rank, seed=seed, shuffle=True) if world > 1 else None
    workers, batch_size = config.get("workers", 2), config.get("batch_size", 4)
    train_loader = make_loader(training, batch_size, workers, sampler, shuffle=True)
    val_loader = make_loader(validation, config.get("val_batch_size", batch_size), workers)
    model = build_model(config, pretrained=not args.resume).to(device)
    if initialization is not None and not args.resume:
        provenance["source_sensor_initialization"] = initialize_source_sensor(model, initialization)
        if rank == 0:
            print(json.dumps(dict(event="source_sensor_initialized", **provenance["source_sensor_initialization"])), flush=True)
    generalization = None
    if dg_options:
        from .domain_generalization import SourceUserGeneralization, STAT_NAMES, state_sha256, alignment_inputs
        availability = np.load(Path(args.cache) / "availability.npy", mmap_mode="r")
        generalization = SourceUserGeneralization(meta, train_indices, availability, list(model.active) + ["fusion"],
                        config.get("dim", 256), dg_options, seed=seed + 900).to(device)
        provenance["domain_generalization"] = generalization.audit
        if world > 1 and any(True for _ in generalization.parameters()):
            generalization = DDP(generalization, device_ids=[device.index], broadcast_buffers=False)
    masked_prediction = None
    if prediction_options:
        from .residual_fusion import MaskedModalityPrediction, PREDICTION_STATS, mask_one_visible
        options = config.get("fusion_config", {})
        masked_prediction = MaskedModalityPrediction(len(model.active), config.get("dim", 256),
            hidden=options.get("hidden", 128), heads=options.get("heads", 4), **prediction_options).to(device)
        if world > 1:
            masked_prediction = DDP(masked_prediction, device_ids=[device.index], broadcast_buffers=False)
    if world > 1:
        model = DDP(model, device_ids=[device.index], broadcast_buffers=False)
    from .optim import make_optimizer
    optimizer = make_optimizer(model, config)
    if generalization is not None:
        auxiliary_parameters = list(generalization.parameters())
        if auxiliary_parameters:
            optimizer.add_param_group(dict(params=auxiliary_parameters, lr=config.get("lr", .0006),
                                           weight_decay=config.get("weight_decay", .01)))
    if masked_prediction is not None:
        optimizer.add_param_group(dict(params=list(masked_prediction.parameters()), lr=config.get("lr", .0006),
                                       weight_decay=config.get("weight_decay", .05)))
    optimization_parameters = [p for group in optimizer.param_groups for p in group["params"]]
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    epochs, accumulation = config.get("epochs", 40), config.get("accumulation", 4)
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = config.get("schedule_epochs", epochs) * steps_per_epoch
    warm_steps = config.get("warmup_epochs", 3) * steps_per_epoch
    start_epoch, global_step, best = 0, 0, -1.
    if args.resume:
        previous = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        for field in ("config", "cache_signature", "fold", "train_subjects", "validation_subjects", "synthetic_smoke", "teacher_targets"):
            if previous["provenance"][field] != provenance[field]:
                raise ValueError(f"Resume provenance differs in {field}")
        if previous["provenance"].get("domain_generalization") != provenance.get("domain_generalization"):
            raise ValueError("Resume source-user support differs")
        initialization_audit = restore_source_sensor_audit(previous["provenance"].get("source_sensor_initialization"), initialization)
        if initialization_audit is not None:
            provenance["source_sensor_initialization"] = initialization_audit
        unwrap(model).load_state_dict(previous["model"])
        if generalization is not None:
            unwrap(generalization).load_state_dict(previous["training_auxiliary_state"]["domain_generalization"], strict=True)
            auxiliary_resume_sha = state_sha256(unwrap(generalization).state_dict())
        if masked_prediction is not None:
            from .domain_generalization import state_sha256
            unwrap(masked_prediction).load_state_dict(previous["training_auxiliary_state"]["masked_prediction"], strict=True)
            prediction_resume_sha = state_sha256(unwrap(masked_prediction).state_dict())
        optimizer.load_state_dict(previous["optimizer"])
        start_epoch, global_step, best = previous["epoch"] + 1, previous["global_step"], previous["best_accuracy"]
        if "scheduler" in previous:
            schedule = previous["scheduler"]
            if schedule["total_steps"] != total_steps or schedule["warmup_steps"] != warm_steps or schedule["base_lrs"] != base_lrs:
                raise ValueError("Resume learning-rate schedule differs from last.pt")
        rng_states = previous.get("rng_states")
        if rng_states:
            if len(rng_states) != world:
                raise ValueError("Resume requires the checkpoint's original GPU count")
            restore_rng(rng_states[rank], device)
        provenance["resume"] = dict(checkpoint=str((out / "last.pt").resolve()), completed_epoch=start_epoch,
                                    global_step=global_step, prior_git_commit=previous["provenance"].get("git_commit"),
                                    optimizer_state_entries=len(optimizer.state), restored_rng=bool(rng_states))
        if generalization is not None:
            provenance["resume"].update(restored_domain_generalization=True, domain_state_sha256=auxiliary_resume_sha)
        if masked_prediction is not None:
            provenance["resume"].update(restored_masked_prediction=True,
                                         masked_prediction_state_sha256=prediction_resume_sha)
        if rank == 0:
            print(json.dumps(dict(event="resume", **provenance["resume"])), flush=True)
            directory = out / "resume_history"
            directory.mkdir(exist_ok=True)
            atomic_json(directory / f"{time.time_ns()}.json", provenance)
        del previous
    tracker = None
    if rank == 0:
        atomic_json(out / "run.json", provenance)
        tracker = RunTracker(out, config, provenance, start_epoch, global_step, steps_per_epoch, args.resume)
    barrier()
    paused = False
    # Device-local memory metric excludes previous unrelated jobs' allocations.
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, epochs):
        started = time.monotonic()
        model.train()
        if sampler:
            sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        loss_sum, examples, correct, classification_sum = 0., 0, 0, 0.
        accumulated = torch.zeros(4, device=device)
        if generalization is not None:
            generalization.train()
            dg_accumulated = torch.zeros(len(STAT_NAMES), device=device)
            dg_epoch = torch.zeros_like(dg_accumulated)
        if masked_prediction is not None:
            masked_prediction.train()
            mp_accumulated = torch.zeros(len(PREDICTION_STATS), device=device)
            mp_epoch = torch.zeros_like(mp_accumulated)
        for step, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            vision = prepare_vision(batch, config["image_size"], True, config.get("visual_configs"))
            sync = (step + 1) % accumulation == 0 or step + 1 == len(train_loader)
            remaining = len(train_loader) - (step // accumulation) * accumulation
            divisor = min(accumulation, remaining)
            context = model.no_sync() if isinstance(model, DDP) and not sync else nullcontext()
            dg_context = generalization.no_sync() if isinstance(generalization, DDP) and not sync else nullcontext()
            mp_context = masked_prediction.no_sync() if isinstance(masked_prediction, DDP) and not sync else nullcontext()
            with context, dg_context, mp_context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(batch, vision)
                    loss = classification_loss(output, batch["label"], config.get("label_smoothing", .05), config.get("auxiliary_weight", .3))
                    if teacher is not None:
                        loss = loss + distillation_loss(output, batch, config.get("temperature", 3.), config.get("kd_weight", .7), config.get("feature_weight", .02))
                task_loss = loss.detach()
                if masked_prediction is not None:
                    visible, hidden = mask_one_visible(output["fusion_mask"])
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        # Recompute context without the target; reusing full-input
                        # latent tokens would leak the very feature being predicted.
                        latent = unwrap(model).residual_fusion.encode(output["branch_features"], visible)
                        prediction_loss, mp_statistics = masked_prediction(latent, output["branch_features"],
                            hidden, epoch + step / len(train_loader))
                    loss = loss + prediction_loss
                    mp_accumulated += mp_statistics * len(batch["label"])
                    mp_epoch += mp_statistics * len(batch["label"])
                if generalization is not None:
                    aligned_features, aligned_available = alignment_inputs(output)
                    auxiliary_loss, dg_statistics = generalization(aligned_features, batch["label"],
                        batch["index"], aligned_available, epoch + step / len(train_loader))
                    loss = loss + auxiliary_loss
                    dg_accumulated += dg_statistics * len(batch["label"])
                    dg_epoch += dg_statistics * len(batch["label"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                (loss / divisor).backward()
            loss_sum += float(loss.detach()) * len(batch["label"])
            classification_sum += float(task_loss) * len(batch["label"])
            examples += len(batch["label"])
            batch_correct = classification_correct(output, batch["label"])
            correct += float(batch_correct)
            accumulated += torch.stack((loss.detach().float() * len(batch["label"]), batch_correct.float(),
                                         loss.new_tensor(float(len(batch["label"]))), task_loss.float() * len(batch["label"])))
            if sync:
                factor = (global_step + 1) / max(warm_steps, 1) if global_step < warm_steps else .5 * (1 + math.cos(math.pi * (global_step - warm_steps) / max(total_steps - warm_steps, 1)))
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr * factor
                grad_norm = torch.nn.utils.clip_grad_norm_(optimization_parameters, config.get("clip_grad", 5.), error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if world > 1:
                    dist.all_reduce(accumulated)
                    if masked_prediction is not None:
                        dist.all_reduce(mp_accumulated)
                if tracker:
                    tracker.step(epoch + 1, step + 1, len(train_loader), global_step,
                                 float(accumulated[0] / accumulated[2]), float(accumulated[1] / accumulated[2]),
                                 [g["lr"] for g in optimizer.param_groups], float(grad_norm))
                    if generalization is not None:
                        tracker.scalar("train/classification_loss_step", accumulated[3] / accumulated[2], global_step)
                        dg_values = (dg_accumulated * world / accumulated[2]).tolist()
                        for name, value in zip(STAT_NAMES, dg_values):
                            tracker.scalar("domain_generalization/" + name, value, global_step)
                    if masked_prediction is not None:
                        for name, value in zip(PREDICTION_STATS, (mp_accumulated / accumulated[2]).tolist()):
                            tracker.scalar("masked_prediction/" + name, value, global_step)
                accumulated.zero_()
                if generalization is not None:
                    dg_accumulated.zero_()
                if masked_prediction is not None:
                    mp_accumulated.zero_()
            if rank == 0 and (step + 1) % config.get("log_every", 20) == 0:
                print(json.dumps(dict(event="train", epoch=epoch + 1, step=step + 1, steps=len(train_loader), loss=loss_sum / examples)), flush=True)
        # Keep normalization buffers identical for validation/export on every rank.
        if world > 1:
            for buffer in unwrap(model).buffers():
                if buffer.is_floating_point():
                    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
                    buffer /= world
        if val_indices:
            metrics, predictions = evaluate(model, val_loader, config, device, rank, world)
        else:
            metrics, predictions = dict(accuracy=None, macro_f1=None, clips=0, branch_accuracy={}), None
        stats = torch.tensor([loss_sum, examples, correct, classification_sum], dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(stats)
            if masked_prediction is not None:
                dist.all_reduce(mp_epoch)
        rng = capture_rng(device)
        rng_states = [None] * world
        if world > 1:
            dist.all_gather_object(rng_states, rng)
        else:
            rng_states = [rng]
        if rank == 0:
            improved = (epoch + 1 == epochs) if not val_indices else metrics["accuracy"] > best
            best = max(best, metrics["accuracy"]) if val_indices else -1.
            row = dict(epoch=epoch + 1, train_loss=float(stats[0] / stats[1]), seconds=time.monotonic() - started,
                       train_accuracy=float(stats[2] / stats[1]), global_step=global_step, best_accuracy=best,
                       peak_gpu_gib=torch.cuda.max_memory_allocated(device) / 1024 ** 3, **metrics)
            if max(config.get("feature_mixup", 0.), config.get("input_mixup", 0.)) > 0:
                row["train_accuracy_definition"] = "accuracy weighted by the two mixed training labels"
            if generalization is not None:
                row["train_classification_loss"] = float(stats[3] / stats[1])
                row["domain_generalization"] = dict(zip(STAT_NAMES, (dg_epoch * world / stats[1]).tolist()))
            if masked_prediction is not None:
                row["masked_prediction"] = dict(zip(PREDICTION_STATS, (mp_epoch / stats[1]).tolist()))
            saving_started = time.monotonic()
            checkpoint = dict(model=model_state(model), config=config, provenance=provenance,
                              epoch=epoch, best_accuracy=best, metrics=metrics, global_step=global_step,
                              checkpoint_version=2)
            if improved:
                save_checkpoint(checkpoint, out / "best.pt")
                if predictions is not None:
                    np.savez_compressed(out / "validation_predictions.npz", **predictions)
            checkpoint.update(optimizer=optimizer.state_dict(), rng_states=rng_states,
                              scheduler=dict(type="warmup_cosine", total_steps=total_steps, warmup_steps=warm_steps,
                                             base_lrs=base_lrs, global_step=global_step))
            if generalization is not None:
                checkpoint["training_auxiliary_state"] = dict(domain_generalization=model_state(generalization))
            if masked_prediction is not None:
                checkpoint.setdefault("training_auxiliary_state", {})["masked_prediction"] = model_state(masked_prediction)
            save_checkpoint(checkpoint, out / "last.pt")
            row.update(checkpoint_seconds=time.monotonic() - saving_started, wall_seconds=time.monotonic() - started)
            with (out / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "confusion_matrix"}), flush=True)
            tracker.epoch(row, global_step)
        barrier()
        stop = torch.tensor(int((out / "STOP_AFTER_EPOCH").exists() or
                                 (args.stop_after_epoch is not None and epoch + 1 >= args.stop_after_epoch)), device=device)
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item() and epoch + 1 < epochs:
            paused = True
            break
    if rank == 0:
        status = dict(status="paused" if paused else "complete", best_accuracy=best,
                      epochs=epochs, completed_epoch=epoch + 1 if start_epoch < epochs else start_epoch,
                      global_step=global_step, synthetic_smoke=args.smoke)
        atomic_json(out / ("paused.json" if paused else "complete.json"), status)
        atomic_json(out / "progress.json", status)
        if not paused:
            (out / "paused.json").unlink(missing_ok=True)
        tracker.close()
    barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
