"""Deterministic IR intensity transforms computed independently for each clip."""
import math

import torch


def normalize_ir_clip(pixels, options):
    """Map uint8 B,T,1,H,W to [0,1], with one monotone mapping per clip.

    Histogram clipping is global within a clip, not spatial CLAHE. Pooling the
    histogram over time keeps the mapping identical for all frames. No batch
    statistics, fitted data statistics, random state, or learned weights are used.
    """
    if pixels.dtype != torch.uint8 or pixels.ndim != 5 or pixels.shape[2] != 1:
        raise ValueError("IR intensity expects uint8 B,T,1,H,W")
    mode = options.get("mode", "none")
    allowed = {"none": {"mode"}, "gamma": {"mode", "gamma"},
               "clip_histogram": {"mode", "clip_limit"}}
    if mode not in allowed or set(options) - allowed[mode]:
        raise ValueError("Unknown IR intensity mode or option")
    value = pixels.float() / 255
    if mode == "none":
        return value
    if mode == "gamma":
        gamma = options.get("gamma", 1.)
        if not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("IR gamma must be finite and positive")
        return value.pow(gamma)
    limit = options.get("clip_limit", 2.)
    if not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit < 1:
        raise ValueError("IR histogram clip limit must be finite and at least one")
    flat = pixels.flatten(1).long()
    hist = torch.zeros((len(pixels), 256), device=pixels.device, dtype=torch.float32)
    hist.scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.float32))
    clipped = hist.clamp_max(max(1., limit * flat.shape[1] / 256))
    # Redistribute excess uniformly. Each clip retains its original total count.
    clipped = clipped + (hist - clipped).sum(1, keepdim=True) / 256
    cumulative = clipped.cumsum(1)
    table = (cumulative - cumulative[:, :1]) / (cumulative[:, -1:] - cumulative[:, :1]).clamp_min(1)
    result = table.gather(1, flat).reshape_as(value)
    # Preserve constant clips, including all-zero missing-camera inputs.
    constant = (flat.amax(1) == flat.amin(1)).view(-1, 1, 1, 1, 1)
    return torch.where(constant, value, result)
