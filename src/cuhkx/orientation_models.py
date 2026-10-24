"""Body-relative IMU orientation with a real pretrained inertial encoder."""
import torch
from torch import nn
from .imu_models import LIMUBertIMUEncoder
from .official_models import masked_pool


def quaternion_matrix(q):
    q = nn.functional.normalize(q.float(), dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def body_orientation_features(quaternion, valid):
    """Gravity and chest-relative rotation, invariant to a common world yaw."""
    if quaternion.ndim != 4 or quaternion.shape[2:] != (5, 4) or valid.shape != quaternion.shape[:-1]:
        raise ValueError('Expected B,T,5,4 quaternions and B,T,5 validity')
    mask = valid > 0
    q = quaternion.float().masked_fill(~mask[..., None], 0)
    if not torch.isfinite(q).all():
        raise ValueError('Nonfinite valid quaternion')
    rotation = quaternion_matrix(q)
    gravity = rotation[..., 2, :] * mask[..., None]
    chest = rotation[:, :, 2]
    relative = chest.transpose(-1, -2)[:, :, None] @ rotation
    pair = mask & mask[:, :, 2:3]
    six = relative[..., :2].flatten(-2) * pair[..., None]
    return torch.cat((gravity, six, mask[..., None].float(), pair[..., None].float()), -1)


class TemporalPoseBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(64, 64, 5, padding=2*dilation, dilation=dilation, groups=64),
                                 nn.GroupNorm(8, 64), nn.GELU(), nn.Conv1d(64, 64, 1), nn.Dropout(.2))

    def forward(self, x):
        return x + self.net(x)


class PoseIMUEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.device_encoder = nn.Sequential(nn.Linear(11, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.temporal = nn.Sequential(nn.Conv1d(5*32, 64, 5, padding=2), nn.GroupNorm(8, 64), nn.GELU(),
                                      TemporalPoseBlock(1), TemporalPoseBlock(2))
        self.projection = nn.Sequential(nn.Dropout(.3), nn.Linear(128, dim), nn.LayerNorm(dim), nn.GELU())

    def forward(self, values, valid):
        if values.shape[-1] != 10 or valid.shape != (*values.shape[:-1], 2):
            raise ValueError('Pose diagnostics require six old channels, four quaternions, and two validity fields')
        mask = valid[..., 1] > 0
        feature = self.device_encoder(body_orientation_features(values[..., 6:10], mask))
        feature = feature * mask[..., None]
        temporal = self.temporal(feature.flatten(2).transpose(1, 2)).transpose(1, 2)
        pooled = masked_pool(temporal, mask.any(2))
        return self.projection(pooled) * mask.any((1, 2))[:, None]


class LIMUPoseIMUEncoder(LIMUBertIMUEncoder):
    def __init__(self, dim=256, pretrained=None, pretrained_sha256=None):
        super().__init__(dim=dim, pretrained=pretrained, pretrained_sha256=pretrained_sha256)
        self.pose = PoseIMUEncoder(dim=128)
        self.combination = nn.Sequential(nn.Linear(dim+128, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(.2))

    def forward(self, values, valid):
        inertial_valid = valid[:, 0, :, 0]
        inertial = super().forward(values[..., :6], inertial_valid)
        pose = self.pose(values, valid)
        return self.combination(torch.cat((inertial, pose), -1)) * (inertial_valid.sum(1, keepdim=True) > 0)
