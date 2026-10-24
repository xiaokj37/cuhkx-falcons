"""Cache teacher targets, export a single small checkpoint, or predict test clips."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from .data import ClipDataset, prepare_vision, read_meta, to_device
from .models import build_model, inference_config
from .weights import pack_state, unpack_state
from .train import barrier, distributed_setup, evaluate, make_loader, save_checkpoint, split_indices


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("targets", "export", "test", "validate"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--storage", choices=("fp16", "int8", "int5"), default="fp16", help="Single-file inference weight storage")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config, provenance = checkpoint["config"], checkpoint["provenance"]
    if provenance.get("synthetic_smoke") and not args.smoke:
        raise ValueError("Synthetic smoke weights cannot be used as competition weights")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "export":
        if int(__import__("os").environ.get("WORLD_SIZE", 1)) != 1:
            raise ValueError("Export once, outside torchrun")
        formats = {"fp16":"cuhkx-single-checkpoint-v1", "int8":"cuhkx-single-checkpoint-v2-int8",
                   "int5":"cuhkx-single-checkpoint-v3-int5"}
        compact = dict(format=formats[args.storage],
                       config=inference_config(config), provenance=provenance,
                       model=pack_state(checkpoint["model"], args.storage),
                       source_sha256=file_hash(args.checkpoint))
        if config.get("input_crop") == "person_roi":
            from .roi import pack_detector
            asset = checkpoint.get("detector") or pack_detector(config["detector"])
            if asset["state_sha256"] != provenance["preprocessing"]["detector_state_sha256"]:
                raise ValueError("Export detector differs from the one used to prepare training images")
            compact["detector"] = asset
            compact["config"]["detector"] = None
        # Save to a new temporary filename; an oversized artifact is not advertised as deployable.
        candidate = out.with_name(out.name + ".candidate")
        save_checkpoint(compact, candidate)
        size = candidate.stat().st_size
        if size >= 100_000_000:
            raise ValueError(f"Checkpoint exceeds competition limit: {size} bytes; retained as {candidate.name}")
        verify = torch.load(candidate, map_location="cpu", weights_only=True)
        model = build_model(verify["config"], pretrained=False)
        decoded = unpack_state(verify["model"])
        model.load_state_dict(decoded, strict=True)
        if any(not torch.isfinite(v).all() for v in decoded.values() if v.is_floating_point()):
            raise ValueError("Nonfinite values after storage conversion")
        if "detector" in verify:
            from .roi import detector_from_asset
            detector_from_asset(verify["detector"])
        candidate.replace(out)
        report = dict(bytes=size, limit_bytes=100_000_000, parameters=sum(p.numel() for p in model.parameters()),
                      source_sha256=compact["source_sha256"], sha256=file_hash(out), synthetic_smoke=provenance.get("synthetic_smoke", False),
                      includes_detector="detector" in compact, storage=args.storage)
        out.with_suffix(".json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return
    if not args.cache:
        raise ValueError("--cache is required")
    rank, world, device = distributed_setup()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    meta = read_meta(args.cache)
    from .data import validate_preprocessing
    validate_preprocessing(meta, config, provenance)
    if args.mode in ("targets", "validate"):
        if meta["signature"] != provenance["cache_signature"]:
            raise ValueError("Teacher checkpoint and cache do not match")
        indices, validation_indices, train_subjects, val_subjects = split_indices(meta, provenance["fold"], args.smoke)
        if train_subjects != provenance["train_subjects"] or val_subjects != provenance["validation_subjects"]:
            raise ValueError("Teacher subject provenance mismatch")
        if args.mode == "validate":
            if not validation_indices:
                raise ValueError("A full-data fit has no held-out validation fold")
            indices = validation_indices
    else:
        if meta["split"] != "test" and not (args.smoke and meta["split"] == "synthetic"):
            raise ValueError("Submission inference requires the unlabeled test cache")
        if meta["split"] == "test" and len(meta["records"]) != 405:
            raise ValueError("Official test cache must have exactly 405 clips")
        indices = list(range(len(meta["records"])))
    dataset = ClipDataset(args.cache, indices[rank::world], config["frames"], training=False)
    loader = make_loader(dataset, args.batch_size, args.workers)
    model = build_model(config, pretrained=False).to(device).eval()
    model.load_state_dict(unpack_state(checkpoint["model"]), strict=True)
    if args.mode == "validate":
        metrics, predictions = evaluate(model, loader, config, device, rank, world)
        if rank == 0:
            metrics.update(checkpoint_sha256=file_hash(args.checkpoint), cache_signature=meta["signature"],
                           synthetic_smoke=args.smoke)
            out.write_text(json.dumps(metrics, indent=2))
            np.savez_compressed(out.with_suffix(".npz"), **predictions)
            print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}), flush=True)
        barrier()
        if world > 1:
            dist.destroy_process_group()
        return
    arrays = {k: [] for k in ("indices", "logits", "features", "weights")}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        batch = to_device(batch, device)
        vision = prepare_vision(batch, config["image_size"], False, config.get("visual_configs"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(batch, vision)
        arrays["indices"].append(batch["index"].cpu().numpy())
        for key in ("logits", "features", "weights"):
            arrays[key].append(prediction[key].float().cpu().numpy())
    local = {k: np.concatenate(v) if v else None for k, v in arrays.items()}
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if rank == 0:
        result = {k: np.concatenate([g[k] for g in gathered if g[k] is not None]) for k in arrays}
        order = np.argsort(result["indices"])
        result = {k: v[order] for k, v in result.items()}
        if result["indices"].tolist() != sorted(indices):
            raise ValueError("Incomplete or duplicate inference rows")
        if not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError("Nonfinite inference result")
        if args.mode == "targets":
            np.savez_compressed(out, **result)
            info = dict(cache_signature=meta["signature"], fold=provenance["fold"],
                        train_subjects=train_subjects, validation_subjects=val_subjects,
                        teacher_checkpoint_sha256=file_hash(args.checkpoint), clips=len(indices))
        else:
            with out.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("path", "prediction"))
                for index, label in zip(result["indices"], result["logits"].argmax(-1)):
                    writer.writerow((meta["records"][int(index)]["path"], int(label)))
            np.savez_compressed(out.with_suffix(".npz"), **result)
            info = dict(clips=len(indices), checkpoint_sha256=file_hash(args.checkpoint),
                        cache_signature=meta["signature"], synthetic_smoke=args.smoke,
                        elapsed_seconds=time.monotonic() - started,
                        peak_gpu_gib=torch.cuda.max_memory_allocated(device) / 1024 ** 3)
        out.with_suffix(".json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info), flush=True)
    barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
