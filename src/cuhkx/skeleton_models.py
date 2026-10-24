"""Public MotionBERT-Lite initialization for the official DSTformer skeleton family."""
from functools import partial
import hashlib
from pathlib import Path

import torch
from torch import nn

from .vendor.dstformer import DSTformer


def load_motionbert_state(path, sha256):
    if not sha256 or hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha256:
        raise ValueError("MotionBERT checkpoint SHA256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if set(checkpoint) != {"model_pos"}:
        raise ValueError("Expected the original released MotionBERT model_pos checkpoint")
    state = checkpoint["model_pos"]
    if not state or not all(k.startswith("module.") and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Unexpected MotionBERT state dictionary")
    return {k.removeprefix("module."): v for k, v in state.items()}


def orthographic_skeleton_views(x, valid):
    """World XYZ -> frontal X/-Z and lateral Y/-Z, with frame confidence.

    One clip-wide isotropic scale retains relative motion and axis magnitudes.
    Both views use the existing H36M joint order. Zero coordinates are valid;
    only the explicit frame mask determines which points set the bounds.
    """
    present = valid > 0
    mask = present[:, :, None, None]
    safe = x.float().masked_fill(~mask, 0)
    lower = safe.masked_fill(~mask, float("inf")).amin((1, 2))
    upper = safe.masked_fill(~mask, -float("inf")).amax((1, 2))
    any_valid = present.any(1, keepdim=True)
    lower = torch.where(any_valid, lower, 0.)
    upper = torch.where(any_valid, upper, 0.)
    center = (lower + upper) * .5
    scale = (upper - lower).amax(-1, keepdim=True).clamp_min(1e-6)
    normalized = (2 * (safe - center[:, None, None]) / scale[:, None, None]).clamp(-1, 1)
    confidence = valid.float().clamp(0, 1)[:, :, None].expand(-1, -1, x.shape[2])
    views = [torch.stack((normalized[..., axis], -normalized[..., 2], confidence), -1)
             for axis in (0, 1)]
    return torch.stack(views, 1).masked_fill(~mask[:, None], 0)


class MotionBERTLiteEncoder(nn.Module):
    def __init__(self, dim=256, pretrained=None, pretrained_sha256=None, drop_path=.1):
        super().__init__()
        # These are the exact public MB_lite.yaml and load_backbone settings.
        self.backbone = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                                  depth=5, num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                  drop_path_rate=drop_path, att_fuse=True)
        if pretrained:
            self.backbone.load_state_dict(load_motionbert_state(pretrained, pretrained_sha256), strict=True)
        # Load the original 3D pose head strictly, then remove only this task head.
        self.backbone.head = nn.Identity()
        self.projection = nn.Sequential(nn.Dropout(.3), nn.Linear(2 * 17 * 512, dim),
                                        nn.LayerNorm(dim), nn.GELU())

    def forward(self, x, valid):
        b, t, joints, channels = x.shape
        if joints != 17 or channels != 3 or t > 243:
            raise ValueError("MotionBERT-Lite expects H36M 17-joint XYZ clips with at most 243 frames")
        views = orthographic_skeleton_views(x, valid)
        features = self.backbone.get_representation(views.reshape(b * 2, t, 17, 3))
        features = features.reshape(b, 2, t, 17, 512)
        weight = valid.to(features.dtype).clamp(0, 1)[:, None, :, None, None]
        pooled = (features * weight).sum(2) / weight.sum(2).clamp_min(1)
        return self.projection(pooled.flatten(1))
