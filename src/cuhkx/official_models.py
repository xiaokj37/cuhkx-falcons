"""Sequence/missing-data adapters for the official CUHK-X model components.

These are architecture adaptations, not pretrained CUHK-X weights. The release
has no checkpoints. Source lineage and changes are recorded in third_party/.
"""
import torch
from torch import nn

from .vendor.dstformer import DSTformer
from .vendor.cnn_transformer import CNNTransformer
from .vendor.radar_pointnet import RadarPointNet


def masked_pool(x, valid):
    """Mean/max over time; all-missing branches produce zero, never NaN."""
    weight = valid.to(x.dtype).clamp(0, 1)
    mean = (x * weight[..., None]).sum(1) / weight.sum(1, keepdim=True).clamp_min(1)
    maximum = x.masked_fill(weight[..., None] <= 0, -1e4).amax(1)
    maximum = maximum * (weight.sum(1, keepdim=True) > 0)
    return torch.cat((mean, maximum), -1)


class OfficialSkeletonEncoder(nn.Module):
    def __init__(self, dim=256, width=64, depth=1, heads=2, maxlen=64):
        super().__init__()
        self.backbone = DSTformer(dim_in=3, dim_out=0, dim_feat=width, dim_rep=width,
                                  depth=depth, num_heads=heads, mlp_ratio=2,
                                  num_joints=17, maxlen=maxlen, att_fuse=True)
        # Keep joint identities, as in the official ActionHeadClassification.
        self.projection = nn.Sequential(nn.Linear(17 * width, dim), nn.LayerNorm(dim), nn.GELU())

    def forward(self, x, valid):
        root = x[:, :, :1]
        torso = (x[:, :, 8] - x[:, :, 0]).norm(dim=-1)
        scale = torso.masked_fill(valid <= 0, float("nan")).nanmedian(1).values
        scale = torch.nan_to_num(scale, nan=1.).clamp_min(.05).view(-1, 1, 1, 1)
        # Zero z is valid. Invalid frames are zeroed before spatial/temporal attention.
        joints = ((x - root) / scale).clamp(-10, 10) * valid[:, :, None, None]
        feature = self.backbone.get_representation(joints)
        pooled = (feature * valid[:, :, None, None]).sum(1) / valid.sum(1)[:, None, None].clamp_min(1)
        return self.projection(pooled.flatten(1))


class OfficialIMUEncoder(nn.Module):
    def __init__(self, dim=256, maxlen=128):
        super().__init__()
        # Six acc/gyro channels per device plus five device-validity indicators.
        self.encoder = CNNTransformer(35, maxlen, out_channels=[32, 64, 128],
                                       num_head=2, num_attn_layers=2)
        # Upstream keeps an unused template layer as a trainable module. Remove
        # it: cloned transformer_encoder layers are the ones used in forward.
        del self.encoder.encoder_layer
        self.projection = nn.Sequential(nn.Linear(256, dim), nn.LayerNorm(dim), nn.GELU())

    def forward(self, x, valid):
        x = x * valid[:, None, :, None]
        values = torch.cat((x.flatten(2), valid[:, None].expand(-1, x.shape[1], -1)), -1)
        feature = self.encoder(values.transpose(1, 2)).transpose(1, 2)
        return self.projection(torch.cat((feature.mean(1), feature.amax(1)), -1))


class MaskedRadarPointNet(RadarPointNet):
    def forward(self, x, valid):
        b, t, n, _ = x.shape
        mask = valid.reshape(b * t, n) > 0
        points = x.reshape(b * t, n, 4).masked_fill(~mask[..., None], 0)
        # Mask both TNet pooling and feature pooling, so padding cannot win max.
        tn = self.input_tnet
        hidden = torch.relu(tn.conv1(points.transpose(1, 2)))
        hidden = torch.relu(tn.conv2(hidden))
        hidden = tn.conv3(hidden).masked_fill(~mask[:, None], -1e4).amax(-1)
        hidden = hidden * mask.any(-1, keepdim=True)
        transform = tn.fc(hidden).reshape(-1, 4, 4) + torch.eye(4, device=x.device, dtype=hidden.dtype)
        points = torch.bmm(points, transform)
        feature = self.mlp(points.transpose(1, 2)).masked_fill(~mask[:, None], -1e4).amax(-1)
        feature = feature * mask.any(-1, keepdim=True)
        return feature.reshape(b, t, 256)


class OfficialRadarEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.pointnet = MaskedRadarPointNet()
        # The release passes B,T,C to a time-first LSTM. Explicit batch_first
        # fixes that bug: predictions must not depend on other clips in a batch.
        self.lstm = nn.LSTM(256, 128, num_layers=2, bidirectional=True, dropout=.3, batch_first=True)
        self.projection = nn.Sequential(nn.Linear(512, dim), nn.LayerNorm(dim), nn.GELU())

    def forward(self, x, valid):
        point_features = self.pointnet(x, valid)
        # cuDNN LSTM on this stack does not accept bf16 outside an autocast region
        # consistently; the small recurrent branch runs explicitly in fp32.
        with torch.autocast(x.device.type, enabled=False):
            feature, _ = self.lstm(point_features.float())
        return self.projection(masked_pool(feature, (valid.sum(-1) > 0).float()))


class ModalityAttention(nn.Module):
    """Attention between six clip embeddings, with explicit padding masks."""
    def __init__(self, count, dim=256, layers=2, heads=4):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(1, count, dim))
        nn.init.normal_(self.embedding, std=.02)
        block = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=.1,
                                            activation="gelu", batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, features, mask):
        valid = mask[..., None].to(features.dtype)
        tokens = (features + self.embedding) * valid
        output = self.layers(tokens, src_key_padding_mask=mask <= 0)
        return self.norm(output) * valid
