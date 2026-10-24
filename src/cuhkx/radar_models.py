"""Transfer the public X-Fi radar backbone to CUHK-X point-cloud sequences."""
import hashlib
from pathlib import Path

import torch
from torch import nn

from .official_models import masked_pool
from .vendor.xfi_point_transformer import mmwave_PointTransformerReg


def load_xfi_state(path, expected_sha256):
    path = Path(path)
    if not expected_sha256 or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("X-Fi public radar checkpoint SHA256 mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected the original X-Fi radar tensor state dictionary")
    return state


def radar_public_inputs(values, valid):
    """Restore units, map sensor axes to ROS, and fill slots using valid points.

    Extended cache: [x/5, y/5, z/5, velocity/3, native_snr].
    Public binary input: [y, -x, z, native_snr/10, velocity].
    Repeating valid points to a fixed count avoids batch-dependent zero padding;
    missing frames are excluded from the backbone by the caller.
    """
    if values.ndim != 4 or values.shape[-1] != 5 or valid.shape != values.shape[:-1]:
        raise ValueError("X-Fi requires B,T,N,5 radar values and B,T,N validity")
    if values.shape[2] < 1 or values.shape[1] < 1:
        raise ValueError("Radar time and point dimensions must be positive")
    mask = valid > 0
    # Mask before arithmetic: arbitrary invalid values, including NaNs, must
    # neither become inputs nor affect gradients through unselected padding.
    x = values.float().masked_fill(~mask[..., None], 0)
    physical = torch.stack((x[..., 1] * 5, -x[..., 0] * 5, x[..., 2] * 5,
                            x[..., 4] * .1, x[..., 3] * 3), -1)
    count = mask.sum(-1)
    n = values.shape[2]
    order = (~mask).long().argsort(dim=-1, stable=True)
    positions = torch.arange(n, device=values.device) % count.clamp_min(1)[..., None]
    indices = order.gather(-1, positions)
    filled = physical.gather(2, indices[..., None].expand(-1, -1, -1, 5))
    return filled, count > 0


class XFiRadarEncoder(nn.Module):
    def __init__(self, dim=256, pretrained=None, pretrained_sha256=None, freeze_backbone_bn=True):
        super().__init__()
        public = mmwave_PointTransformerReg()
        if pretrained:
            public.load_state_dict(load_xfi_state(pretrained, pretrained_sha256), strict=True)
        # The external 27-class classifier is checked when loading, then removed.
        self.backbone = public.backbone
        self.freeze_backbone_bn = freeze_backbone_bn
        self.temporal = nn.GRU(512, 128, bidirectional=True, batch_first=True)
        self.projection = nn.Sequential(nn.Dropout(.3), nn.Linear(512, dim), nn.LayerNorm(dim), nn.GELU())
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone_bn:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, values, valid):
        points, frame_valid = radar_public_inputs(values, valid)
        b, t, n, _ = points.shape
        selected = frame_valid.flatten().nonzero(as_tuple=False).flatten()
        flat = points.reshape(b * t, n, 5)
        # FP32 preserves the public point attention, distances and recurrent math.
        with torch.autocast(values.device.type, enabled=False):
            if len(selected):
                encoded, _ = self.backbone(flat.index_select(0, selected))
                frames = flat.new_zeros(b * t, 512).index_copy(0, selected, encoded.mean(1))
            else:
                # All DDP ranks must traverse every parameter, even when this
                # rank has no radar. Do not update BN statistics on a dummy cloud.
                modes = [(m, m.training) for m in self.backbone.modules()
                         if isinstance(m, nn.modules.batchnorm._BatchNorm)]
                for module, _ in modes:
                    module.eval()
                try:
                    encoded, _ = self.backbone(flat[:1])
                finally:
                    for module, mode in modes:
                        module.train(mode)
                frames = flat.new_zeros(b * t, 512) + encoded.sum() * 0
            temporal, _ = self.temporal(frames.reshape(b, t, 512))
            pooled = masked_pool(temporal, frame_valid.float())
        return self.projection(pooled) * frame_valid.any(1, keepdim=True)
