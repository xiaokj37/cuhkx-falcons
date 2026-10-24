"""VideoMAE encoders initialized from original public video releases."""
import hashlib
from pathlib import Path

import torch
from torch import nn

from .vendor.videomae import vit_small_patch16_224, vit_base_patch16_224


def load_video_mae_state(path, expected_sha256):
    if not expected_sha256:
        raise ValueError("Pin the public VideoMAE checkpoint SHA256")
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("VideoMAE pretraining checkpoint SHA256 mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("model", state.get("module", state))
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected the original VideoMAE model state dictionary")
    return state


class VideoMAEEncoder(nn.Module):
    def __init__(self, channels=3, frames=16, dim=256, image_size=224,
                 pretrained=None, pretrained_sha256=None, drop_path=.1,
                 activation_checkpointing=False, pretraining_classes=400, *, variant="small"):
        super().__init__()
        factories = {"small": (vit_small_patch16_224, 384), "base": (vit_base_patch16_224, 768)}
        if variant not in factories:
            raise ValueError("Choose the small or base VideoMAE architecture")
        factory, width = factories[variant]
        self.variant = variant
        if channels not in (1, 3) or frames < 2 or frames % 2:
            raise ValueError("VideoMAE requires one or three channels and an even clip length")
        if image_size < 16 or image_size % 16 or not 0 <= drop_path < 1:
            raise ValueError("VideoMAE requires a multiple-of-16 image size and drop path in [0,1)")
        if pretraining_classes not in (174, 400, 710):
            raise ValueError("Select a public VideoMAE class count: 174, 400, or 710")
        self.frames, self.image_size, self.channels = frames, image_size, channels
        net = factory(
            img_size=image_size, all_frames=frames, num_classes=pretraining_classes,
            drop_path_rate=drop_path, use_checkpoint=activation_checkpointing)
        if pretrained:
            net.load_state_dict(load_video_mae_state(pretrained, pretrained_sha256), strict=True)
        if channels == 1:
            original = net.patch_embed.proj
            adapted = nn.Conv3d(1, original.out_channels, original.kernel_size,
                                original.stride, original.padding, bias=True)
            with torch.no_grad():
                adapted.weight.copy_(original.weight.sum(1, keepdim=True))
                adapted.bias.copy_(original.bias)
            net.patch_embed.proj = adapted
        net.head = nn.Identity()
        self.backbone = net
        self.projection = nn.Sequential(nn.Dropout(.3), nn.Linear(width, dim),
                                        nn.LayerNorm(dim), nn.GELU())

    def forward(self, x):
        if x.ndim != 5 or tuple(x.shape[1:]) != (
                self.frames, self.channels, self.image_size, self.image_size):
            raise ValueError("Expected B,T,C,H,W matching the configured VideoMAE clip")
        return self.projection(self.backbone(x.permute(0, 2, 1, 3, 4).contiguous()))


class VideoMAESmallEncoder(VideoMAEEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, variant="small", **kwargs)


class VideoMAEBaseEncoder(VideoMAEEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, variant="base", **kwargs)
