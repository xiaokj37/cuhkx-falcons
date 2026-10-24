"""Rank-zero TensorBoard logging and restart-safe epoch history."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def capture_rng(device):
    import random
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(state[0], torch.tensor(state[1].astype(np.int64)),
                int(state[2]), int(state[3]), float(state[4])), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def restore_rng(value, device):
    import random
    random.setstate(value["python"])
    state = value["numpy"]
    np.random.set_state((state[0], state[1].cpu().numpy().astype(np.uint32), state[2], state[3], state[4]))
    torch.set_rng_state(value["torch"].cpu())
    if device.type == "cuda" and value["cuda"] is not None:
        torch.cuda.set_rng_state(value["cuda"].cpu(), device)


def committed_history(out, completed_epoch):
    """Remove journal entries beyond last.pt, retaining the original for audit."""
    path = Path(out) / "metrics.jsonl"
    if not path.exists():
        return []
    rows, discarded = {}, False
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            epoch = int(row["epoch"])
            if epoch > completed_epoch:
                discarded = True
            else:
                discarded |= epoch in rows
                rows[epoch] = row
        except (ValueError, KeyError):
            discarded = True
    result = [rows[e] for e in sorted(rows)]
    if discarded:
        backup = path.with_name(f"metrics.before_resume.{time.time_ns()}.jsonl")
        backup.write_bytes(path.read_bytes())
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text("".join(json.dumps(r) + "\n" for r in result))
        temporary.replace(path)
    return result


class RunTracker:
    def __init__(self, out, config, provenance, start_epoch=0, global_step=0, steps_per_epoch=1, resume=False):
        from torch.utils.tensorboard import SummaryWriter
        self.out, self.config = Path(out), config
        self.steps_per_epoch = steps_per_epoch
        self.path = self.out / "tensorboard"
        existing = list(self.path.glob("events.out.tfevents.*"))
        self.writer = SummaryWriter(str(self.path), flush_secs=5, max_queue=10,
                                    purge_step=global_step + 1 if resume and existing else None)
        self.writer.add_text("run/config", "```json\n" + json.dumps(config, indent=2) + "\n```", global_step)
        self.writer.add_text("run/provenance", "```json\n" + json.dumps(provenance, indent=2) + "\n```", global_step)
        rows = committed_history(self.out, start_epoch) if resume else []
        if not existing:
            for row in rows:
                self.epoch(row, int(row.get("global_step", row["epoch"] * steps_per_epoch)))
            if rows:
                self.writer.add_text("run/history_import", "Historical epoch metrics imported from metrics.jsonl. "
                                     "Their original wall-clock timestamps are unavailable; step/epoch values are preserved.", global_step)
        self.writer.flush()

    def scalar(self, tag, value, step):
        if value is not None and math.isfinite(float(value)):
            self.writer.add_scalar(tag, float(value), int(step))

    def step(self, epoch, step, total_steps, global_step, loss, accuracy, lrs, grad_norm):
        self.scalar("train/loss_step", loss, global_step)
        self.scalar("train/accuracy_step", accuracy, global_step)
        if len(lrs) == 2:
            self.scalar("optimizer/lr_backbone", lrs[0], global_step)
            self.scalar("optimizer/lr_other", lrs[1], global_step)
        else:
            self.scalar("optimizer/lr_min", min(lrs), global_step)
            self.scalar("optimizer/lr_max", max(lrs), global_step)
        self.scalar("optimizer/grad_norm", grad_norm, global_step)
        self.scalar("progress/epoch", epoch - 1 + step / total_steps, global_step)
        self.scalar("progress/percent", 100 * (epoch - 1 + step / total_steps) / self.config["epochs"], global_step)
        atomic_json(self.out / "progress.json", dict(status="training", epoch=epoch, epochs=self.config["epochs"],
                    step=step, steps=total_steps, global_step=global_step, loss=loss, accuracy=accuracy,
                    updated_at=datetime.now(timezone.utc).isoformat()))

    def epoch(self, row, global_step):
        for key, tag in (("train_loss", "train/loss_epoch"), ("train_accuracy", "train/accuracy_epoch"),
                         ("loss", "val/loss"), ("accuracy", "val/accuracy"), ("macro_f1", "val/macro_f1"),
                         ("seconds", "timing/compute_validation_seconds"), ("checkpoint_seconds", "timing/checkpoint_seconds"),
                         ("wall_seconds", "timing/epoch_wall_seconds"), ("peak_gpu_gib", "gpu/peak_allocated_gib")):
            self.scalar(tag, row.get(key), global_step)
        self.scalar("progress/completed_epochs", row["epoch"], global_step)
        self.scalar("checkpoint/last_saved_epoch", row["epoch"], global_step)
        self.scalar("checkpoint/best_accuracy", row.get("best_accuracy"), global_step)
        self.scalar("train/classification_loss_epoch", row.get("train_classification_loss"), global_step)
        for modality, value in row.get("branch_accuracy", {}).items():
            self.scalar(f"val_modalities/{modality}", value.get("accuracy"), global_step)
        for modality, weight in row.get("fusion_weights", {}).items():
            self.scalar(f"fusion/{modality}", weight, global_step)
        for name, value in row.get("domain_generalization", {}).items():
            self.scalar("domain_generalization_epoch/" + name, value, global_step)
        for name, value in row.get("masked_prediction", {}).items():
            self.scalar("masked_prediction_epoch/" + name, value, global_step)
        self.scalar("val/late_path_accuracy", row.get("late_path_accuracy"), global_step)
        self.scalar("fusion/interaction_logit_rms", row.get("interaction_logit_rms"), global_step)
        self.writer.flush()

    def close(self):
        self.writer.flush()
        self.writer.close()
