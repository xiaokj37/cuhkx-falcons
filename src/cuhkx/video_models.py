"""Video encoders with public R(2+1)D and tiny Swin initialization.

The Caffe2-compatible layer widths follow moabitcoin/ig65m-pytorch
fc749e2ee354c3e4ddbb144cf511bb868b008f61. See third_party/NOTICE.md.
"""
import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models.video.resnet import BasicBlock, Conv2Plus1D, R2Plus1dStem, VideoResNet


class R2Plus1DEncoder(nn.Module):
    def __init__(self, channels=3, frames=16, dim=256, pretrained=None,
                 pretrained_sha256=None, partial_bn=True, mixstyle=0.):
        super().__init__()
        if not 0 <= mixstyle <= 1:
            raise ValueError("MixStyle probability must be between zero and one")
        self.mixstyle = mixstyle
        if channels not in (1, 3):
            raise ValueError("R(2+1)D accepts one camera with 1 or 3 channels")
        self.frames, self.partial_bn = frames, partial_bn
        net = VideoResNet(BasicBlock, [Conv2Plus1D] * 4, [3, 4, 6, 3], R2Plus1dStem,
                          num_classes=400)
        # These transition widths differ from torchvision's default construction.
        for stage, width in ((net.layer2, 128), (net.layer3, 256), (net.layer4, 512)):
            stage[0].conv2[0] = Conv2Plus1D(width, width, width * 9 // 4)
        for module in net.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.eps, module.momentum = 1e-3, .9
        if pretrained:
            if not pretrained_sha256:
                raise ValueError("Pin the public R(2+1)D checkpoint SHA256")
            path = Path(pretrained)
            if hashlib.sha256(path.read_bytes()).hexdigest() != pretrained_sha256:
                raise ValueError("R(2+1)D pretraining checkpoint SHA256 mismatch")
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        if channels == 1:
            original = net.stem[0]
            adapted = nn.Conv3d(1, original.out_channels, original.kernel_size,
                                original.stride, original.padding, bias=False)
            with torch.no_grad():
                adapted.weight.copy_(original.weight.sum(1, keepdim=True))
            net.stem[0] = adapted
        feature_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.projection = nn.Sequential(nn.Dropout(.3), nn.Linear(feature_dim, dim),
                                        nn.LayerNorm(dim), nn.GELU())

    def train(self, mode=True):
        super().train(mode)
        if mode and self.partial_bn:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm3d) and module is not self.backbone.stem[1]:
                    module.eval()
        return self

    def mix_style(self, x, valid=None):
        """Mix per-camera clip statistics over time and space, using valid clips only."""
        if not self.training or self.mixstyle <= 0:
            return x
        if valid is not None and valid.shape != (len(x),):
            raise ValueError("Style validity must contain one entry per clip")
        selected = torch.arange(len(x), device=x.device) if valid is None else torch.nonzero(valid > 0).flatten()
        if len(selected) < 2 or torch.rand((), device=x.device) >= self.mixstyle:
            return x
        values = x[selected].float()
        mean = values.mean((2,3,4), keepdim=True).detach()
        std = (values.var((2,3,4), keepdim=True, unbiased=False) + 1e-6).sqrt().detach()
        coefficient = torch.distributions.Beta(.1,.1).sample((len(selected),)).to(x.device).view(-1,1,1,1,1)
        order = torch.randperm(len(selected), device=x.device)
        mixed_mean = coefficient * mean + (1 - coefficient) * mean[order]
        mixed_std = coefficient * std + (1 - coefficient) * std[order]
        result = x.clone()
        result[selected] = (((values - mean) / std) * mixed_std + mixed_mean).to(x.dtype)
        return result

    def forward(self, x, style_valid=None):
        if x.ndim != 5 or x.shape[1] != self.frames:
            raise ValueError("Expected B,T,C,H,W with the configured clip length")
        values = x.permute(0, 2, 1, 3, 4).contiguous()
        if self.mixstyle > 0:
            net = self.backbone
            values = net.stem(values)
            values = self.mix_style(net.layer1(values), style_valid)
            values = self.mix_style(net.layer2(values), style_valid)
            values = net.layer4(net.layer3(values))
            features = net.avgpool(values).flatten(1)
        else:
            features = self.backbone(values)
        return self.projection(features)


def load_swin_state(path, expected_sha256):
    if not expected_sha256:
        raise ValueError("Pin the public Video Swin checkpoint SHA256")
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Video Swin pretraining checkpoint SHA256 mismatch")
    return torch.load(path, map_location="cpu", weights_only=True)


class SwinTinyEncoder(nn.Module):
    """TorchVision Video Swin-T with independent camera stems and pooled features."""
    def __init__(self, channels=3, frames=16, dim=256, pretrained=None,
                 pretrained_sha256=None, drop_path=.1):
        super().__init__()
        from torchvision.models.video import swin3d_t
        from torchvision.models.video.swin_transformer import ShiftedWindowAttention3d
        from torchvision.ops import StochasticDepth
        if channels not in (1, 3) or frames < 1 or not 0 <= drop_path < 1:
            raise ValueError("Invalid Video Swin channels, clip length, or drop path")
        self.frames, self.channels = frames, channels
        net = swin3d_t(weights=None)
        # The public tiny factory fixes its constructor keyword. Adjust the
        # existing per-block probabilities with the same linear depth schedule.
        depth_modules = [module for module in net.modules() if isinstance(module, StochasticDepth)]
        for index, module in enumerate(depth_modules):
            module.p = drop_path * index / max(len(depth_modules) - 1, 1)
        if pretrained:
            # These integer tensors describe window geometry, not learned values.
            # Verify the release agrees with constructor geometry before omitting
            # their repeated copies from checkpoints.
            geometry = {name:value.clone() for name, value in net.named_buffers()}
            net.load_state_dict(load_swin_state(pretrained, pretrained_sha256), strict=True)
            for name, value in net.named_buffers():
                if not torch.equal(value, geometry[name]):
                    raise ValueError(f"Pretrained window geometry differs from constructor: {name}")
        for module in net.modules():
            if isinstance(module, ShiftedWindowAttention3d):
                module.register_buffer("relative_position_index", module.relative_position_index, persistent=False)
        if channels == 1:
            original = net.patch_embed.proj
            adapted = nn.Conv3d(1, original.out_channels, original.kernel_size,
                                original.stride, original.padding, bias=True)
            with torch.no_grad():
                adapted.weight.copy_(original.weight.sum(1, keepdim=True))
                adapted.bias.copy_(original.bias)
            net.patch_embed.proj = adapted
        feature_dim = net.head.in_features
        net.head = nn.Identity()
        self.backbone = net
        self.projection = nn.Sequential(nn.Dropout(.3), nn.Linear(feature_dim, dim),
                                        nn.LayerNorm(dim), nn.GELU())

    def forward(self, x):
        if x.ndim != 5 or tuple(x.shape[1:3]) != (self.frames, self.channels):
            raise ValueError("Expected B,T,C,H,W matching the configured Video Swin clip")
        return self.projection(self.backbone(x.permute(0, 2, 1, 3, 4).contiguous()))
