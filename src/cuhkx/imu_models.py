"""Public LIMU-BERT-X transfer to the five synchronized CUHK-X IMU devices."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .vendor.limu_bert import LIMUBertModel4Pretrain


def limu_config():
    return SimpleNamespace(feature_num=6, hidden=72, hidden_ff=144, n_layers=4,
                           n_heads=4, seq_len=20, emb_norm=True)


def load_limu_state(path, expected_sha256):
    path = Path(path)
    if not expected_sha256 or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("LIMU-BERT-X public checkpoint SHA256 mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected the original LIMU-BERT-X tensor state dictionary")
    return state


def imu_windows(values, valid, window=20):
    """Undo cache scaling, retain device identities, and cover the whole sequence."""
    if values.ndim != 4 or tuple(values.shape[2:]) != (5, 6) or valid.shape != (len(values), 5):
        raise ValueError("Expected B,T,5,6 IMU values with B,5 device validity")
    if values.shape[1] < window:
        raise ValueError("LIMU-BERT-X needs at least 20 cached time samples")
    # Cache contains acc[g]/4 and gyro[rad/s]/5. The upstream normalization
    # expects acceleration in g and leaves angular velocity in rad/s unchanged.
    physical = values.float() * values.new_tensor([4., 4., 4., 5., 5., 5.])
    physical = physical * valid[:, None, :, None]
    starts = list(range(0, values.shape[1] - window + 1, window))
    if starts[-1] != values.shape[1] - window:
        starts.append(values.shape[1] - window)
    indices = torch.tensor(starts, device=values.device)[:, None] + torch.arange(window, device=values.device)
    # B,D,W,T,C: each device uses the same encoder, without flattening device axes.
    return physical[:, indices].permute(0, 3, 1, 2, 4).contiguous()


class LIMUBertIMUEncoder(nn.Module):
    def __init__(self, dim=256, pretrained=None, pretrained_sha256=None):
        super().__init__()
        public = LIMUBertModel4Pretrain(limu_config(), output_embed=True)
        if pretrained:
            public.load_state_dict(load_limu_state(pretrained, pretrained_sha256), strict=True)
        # Reconstruction and unused template heads are not deployed or optimized.
        self.backbone = public.transformer
        self.temporal = nn.GRU(5 * 144 + 5, 64, bidirectional=True, batch_first=True)
        self.projection = nn.Sequential(nn.Dropout(.3), nn.Linear(256, dim), nn.LayerNorm(dim), nn.GELU())

    def forward(self, values, valid):
        windows = imu_windows(values, valid)
        b, devices, count, length, channels = windows.shape
        # Preserve the public custom LayerNorm epsilon and recurrence in FP32;
        # the backbone is small, and recurrent CUDA kernels require FP32 here.
        with torch.autocast(values.device.type, enabled=False):
            feature = self.backbone(windows.reshape(-1, length, channels).float())
            feature = torch.cat((feature.mean(1), feature.amax(1)), -1).reshape(b, devices, count, 144)
            feature = feature * valid[:, :, None, None]
            sequence = feature.permute(0, 2, 1, 3).flatten(2)
            sequence = torch.cat((sequence, valid[:, None].expand(-1, count, -1)), -1)
            temporal, _ = self.temporal(sequence)
            pooled = torch.cat((temporal.mean(1), temporal.amax(1)), -1)
        return self.projection(pooled) * (valid.sum(1, keepdim=True) > 0)
