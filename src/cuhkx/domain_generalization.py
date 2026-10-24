"""Source-user-only representation losses; never part of the inference model."""
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from . import MODALITIES


STAT_NAMES = ("contrastive_loss", "contrastive_positive_fraction", "domain_loss",
              "domain_accuracy", "domain_chance", "domain_eligible_fraction",
              "contrastive_coefficient", "domain_coefficient")


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, coefficient):
        ctx.coefficient = coefficient
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


def world_size(distributed=True):
    return dist.get_world_size() if distributed and dist.is_initialized() else 1


def global_sum(value, distributed=True):
    result = value.detach().clone()
    if world_size(distributed) > 1:
        dist.all_reduce(result)
    return result


def modality_mean(sums, counts, distributed=True):
    """DDP averages parameter gradients; undo that average for local sums."""
    denominator = global_sum(counts, distributed)
    active = denominator > 0
    scale = active / denominator.clamp_min(1)
    active_count = active.sum().clamp_min(1)
    loss = (sums * scale).sum() * world_size(distributed) / active_count
    reported = (global_sum(sums, distributed) * scale).sum() / active_count
    return loss, reported, denominator


class GatherFeatures(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.rank, ctx.batch = dist.get_rank(), len(value)
        values = [torch.empty_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(values, value.contiguous())
        return torch.cat(values, 0)

    @staticmethod
    def backward(ctx, gradient):
        # Sum every rank's candidate contribution, then return the owner's slice.
        # Clone because the incoming gradient may alias another autograd branch.
        total = gradient.contiguous().clone()
        dist.all_reduce(total)
        begin = ctx.rank * ctx.batch
        return total[begin:begin + ctx.batch]


def gather_candidates(features, metadata, distributed=True):
    if world_size(distributed) == 1:
        return features, metadata
    # All ranks, including ranks with no eligible anchors, retain this graph.
    features = GatherFeatures.apply(features)
    parts = [torch.empty_like(metadata) for _ in range(world_size(distributed))]
    dist.all_gather(parts, metadata.contiguous())
    return features, torch.cat(parts, 0)


def cross_user_contrastive(features, labels, users, available, indices, temperature, distributed=True):
    metadata = torch.cat((labels[:, None], users[:, None], indices[:, None], available.long()), 1)
    candidates, other = gather_candidates(features, metadata, distributed)
    # Training sampler padding must not create extra copies of a candidate.
    duplicate = other[:, 2, None] == other[None, :, 2]
    unique = ~duplicate.tril(-1).any(1)
    same_action = labels[:, None] == other[None, :, 0]
    different_user = users[:, None] != other[None, :, 1]
    valid = available.T[:, :, None] & other[:, 3:].T.bool()[:, None, :] & unique[None, None, :]
    positive = valid & (same_action & different_user)[None]
    # Same-action/same-user pairs are neither positives nor negatives.
    allowed = valid & ((~same_action) | different_user)[None]
    logits = torch.einsum("bmd,nmd->mbn", features, candidates) / temperature
    log_probability = logits - logits.masked_fill(~allowed, -1e4).logsumexp(-1, keepdim=True)
    positive_count = positive.sum(-1)
    anchors = positive_count > 0
    per_anchor = -log_probability.masked_fill(~positive, 0).sum(-1) / positive_count.clamp_min(1)
    loss, reported, count = modality_mean((per_anchor * anchors).sum(-1), anchors.sum(-1), distributed)
    all_valid = global_sum(available.sum(0), distributed).sum().clamp_min(1)
    return loss, reported, count.sum() / all_valid


def alignment_inputs(output):
    features = torch.cat((output["branch_features"], output["features"][:, None]), 1)
    available = output["availability"]
    return features, torch.cat((available, available.any(1, keepdim=True).to(available.dtype)), 1)


class SourceUserGeneralization(nn.Module):
    """Per-modality same-action alignment across the training users only."""
    def __init__(self, meta, train_indices, availability, modalities, dim, options, seed=0):
        super().__init__()
        allowed = {"contrastive_weight", "temperature", "domain_max_coefficient", "start_epoch", "ramp_epochs"}
        if set(options) - allowed:
            raise ValueError("Unknown cross-user generalization option")
        self.options = {"contrastive_weight": 0., "temperature": .2, "domain_max_coefficient": 0.,
                        "start_epoch": 5., "ramp_epochs": 10., **options}
        if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x)
               for x in self.options.values()):
            raise ValueError("Cross-user hyperparameters must be finite numbers")
        if (not 0 <= self.options["contrastive_weight"] <= 1
                or not 0 <= self.options["domain_max_coefficient"] <= 1
                or not 0 < self.options["temperature"] <= 1
                or self.options["start_epoch"] < 0 or self.options["ramp_epochs"] <= 0
                or max(self.options["contrastive_weight"], self.options["domain_max_coefficient"]) == 0):
            raise ValueError("Invalid cross-user loss weights or schedule")
        if meta["split"] not in ("train", "synthetic"):
            raise ValueError("Generalization losses accept only source training data")
        records = meta["records"]
        selected = np.asarray(train_indices, dtype=np.int64)
        if (not len(selected) or len(np.unique(selected)) != len(selected)
                or selected.min() < 0 or selected.max() >= len(records)):
            raise ValueError("Choose distinct in-range source training indices")
        if (len(set(modalities)) != len(modalities) or not modalities
                or set(modalities) - (set(MODALITIES) | {"fusion"})):
            raise ValueError("Choose distinct supported modalities")
        physical = [MODALITIES.index(m) for m in modalities if m != "fusion"]
        if not physical:
            raise ValueError("Fusion alignment also needs its physical modalities")
        if tuple(availability.shape) != (len(records), len(MODALITIES)):
            raise ValueError("Source availability shape differs from the manifest")
        source_available = np.asarray(availability[selected]) > 0
        valid = np.stack([source_available[:, physical].any(1) if m == "fusion"
                          else source_available[:, MODALITIES.index(m)] for m in modalities], 1)
        subjects = sorted({records[int(i)]["subject"] for i in selected})
        if len(subjects) < 2:
            raise ValueError("Cross-user training needs at least two source users")
        users = np.array([subjects.index(records[int(i)]["subject"]) for i in selected])
        labels = np.array([records[int(i)]["label"] for i in selected])
        if labels.min() < 0 or labels.max() >= 40:
            raise ValueError("Source actions must be labels 0 through 39")
        counts = np.zeros((len(modalities), 40, len(subjects)), dtype=np.int64)
        for m in range(len(modalities)):
            np.add.at(counts[m], (labels[valid[:, m]], users[valid[:, m]]), 1)
        user_lookup = torch.full((len(records),), -1, dtype=torch.long)
        label_lookup = torch.full_like(user_lookup, -1)
        user_lookup[selected] = torch.from_numpy(users)
        label_lookup[selected] = torch.from_numpy(labels)
        self.register_buffer("user_lookup", user_lookup)
        self.register_buffer("label_lookup", label_lookup)
        self.register_buffer("counts", torch.from_numpy(counts).float())
        self.subjects, self.modalities = subjects, list(modalities)
        if self.options["domain_max_coefficient"] > 0:
            # A separate conditional linear classifier for each camera/sensor and action.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            self.domain_weight = nn.Parameter(torch.randn(len(modalities), 40, dim, len(subjects), generator=generator) * .01)
            self.domain_bias = nn.Parameter(torch.zeros(len(modalities), 40, len(subjects)))
        self.audit = dict(source_subjects=subjects, source_clips=len(selected), modalities=list(modalities),
                          source_indices_sha256=hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
                          source_counts_sha256=hashlib.sha256(counts.astype("<i8").tobytes()).hexdigest(),
                          multi_user_actions_per_modality=((counts > 0).sum(-1) >= 2).sum(-1).tolist(),
                          identity_used_only_by_training_loss=True, inference_parameters_added=0,
                          options=self.options, initialization_seed=seed)

    def coefficients(self, epoch_position):
        progress = min(1., max(0., (epoch_position - self.options["start_epoch"]) / self.options["ramp_epochs"]))
        return self.options["contrastive_weight"] * progress, self.options["domain_max_coefficient"] * progress

    def forward(self, features, labels, indices, available, epoch_position, distributed=True):
        if (features.ndim != 3 or features.shape[:2] != available.shape
                or features.shape[1] != len(self.modalities) or labels.shape != indices.shape
                or indices.shape != features.shape[:1]):
            raise ValueError("Cross-user feature, mask and label dimensions differ")
        if (indices < 0).any() or (indices >= len(self.user_lookup)).any():
            raise ValueError("Training index is out of range")
        users = self.user_lookup[indices]
        if (users < 0).any() or not torch.equal(labels, self.label_lookup[indices]):
            raise ValueError("Cross-user loss cannot use held-out users, indices or substituted labels")
        available = available > 0
        with torch.autocast(features.device.type, enabled=False):
            z = F.normalize(features.float().masked_fill(~available[..., None], 0), dim=-1, eps=1e-6)
            contrastive_coefficient, domain_coefficient = self.coefficients(epoch_position)
            loss = z.sum() * 0
            statistics = z.new_zeros(len(STAT_NAMES))
            if self.options["contrastive_weight"] > 0:
                contrastive, report, fraction = cross_user_contrastive(
                    z, labels, users, available, indices, self.options["temperature"], distributed)
                loss = loss + contrastive_coefficient * contrastive
                statistics[0], statistics[1] = report, fraction
            if self.options["domain_max_coefficient"] > 0:
                reversed_features = GradientReverse.apply(z, domain_coefficient)
                logits = torch.einsum("bmd,mbdu->bmu", reversed_features, self.domain_weight[:, labels])
                logits = logits + self.domain_bias[:, labels].permute(1, 0, 2)
                support = self.counts[:, labels].permute(1, 0, 2)
                domain_count = (support > 0).sum(-1)
                eligible = available & (domain_count >= 2)
                logits = logits.masked_fill(support <= 0, -1e4)
                target = users[:, None].expand_as(available)
                own_count = support.gather(-1, target[..., None]).squeeze(-1)
                if (available & (own_count <= 0)).any():
                    raise ValueError("Batch availability contradicts source support")
                weight = support.sum(-1) / (domain_count * own_count).clamp_min(1)
                weight = weight * eligible
                cross_entropy = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="none").view_as(weight)
                domain_loss, report, count = modality_mean((cross_entropy * weight).sum(0), weight.sum(0), distributed)
                loss = loss + domain_loss
                statistics[2] = report
                correct = ((logits.argmax(-1) == target) * weight).sum(0)
                chance = (weight / domain_count.clamp_min(1)).sum(0)
                denominator = count.sum().clamp_min(1)
                statistics[3] = global_sum(correct, distributed).sum() / denominator
                statistics[4] = global_sum(chance, distributed).sum() / denominator
                numerator = global_sum(eligible.sum(), distributed)
                statistics[5] = numerator / global_sum(available.sum(), distributed).clamp_min(1)
            statistics[6], statistics[7] = contrastive_coefficient, domain_coefficient
            return loss, statistics.detach()


def state_sha256(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
