"""CNN temporal branches plus compact sensor encoders and masked fusion."""
from pathlib import Path
import hashlib
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50

from . import MODALITIES
from .video_models import R2Plus1DEncoder, SwinTinyEncoder

PARENTS = (0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)


def sample_clip_mix(availability, alpha):
    """One pairing/target coefficient for every modality, with matched availability."""
    count = len(availability)
    order = torch.arange(count, device=availability.device)
    _, groups = torch.unique(availability > 0, dim=0, return_inverse=True)
    for group in groups.unique():
        members = (groups == group).nonzero(as_tuple=True)[0]
        if len(members) > 1:
            shuffled = members[torch.randperm(len(members), device=availability.device)]
            order[shuffled] = shuffled.roll(1)
    coefficient = torch.distributions.Beta(alpha, alpha).sample((count,)).to(availability.device)
    coefficient = coefficient.maximum(1 - coefficient)
    coefficient = torch.where(order == torch.arange(count, device=availability.device), 1., coefficient)
    return order, coefficient


def mix_values(value, order, coefficient):
    weight = coefficient.view(len(value), *([1] * (value.ndim - 1)))
    return (weight * value.float() + (1 - weight) * value[order].float()).to(value.dtype)


def mix_clip_features(features, branches, quality, availability, alpha):
    """v11 feature-only interpolation; encoders first process individual clips."""
    order, coefficient = sample_clip_mix(availability, alpha)
    return (mix_values(features, order, coefficient), mix_values(branches, order, coefficient),
            mix_values(quality, order, coefficient), order, coefficient)


class TemporalShift(nn.Module):
    """Zero-padded bidirectional TSM before a residual block's first convolution."""
    def __init__(self, net, frames, division=8):
        super().__init__()
        self.net, self.frames, self.division = net, frames, division

    def forward(self, x):
        bt, c, h, w = x.shape
        if bt % self.frames:
            raise ValueError("Clip boundaries do not match temporal segment count")
        v = x.reshape(-1, self.frames, c, h, w)
        k = c // self.division
        shifted = torch.zeros_like(v)
        shifted[:, :-1, :k] = v[:, 1:, :k]
        shifted[:, 1:, k:2 * k] = v[:, :-1, k:2 * k]
        shifted[:, :, 2 * k:] = v[:, :, 2 * k:]
        return self.net(shifted.reshape(bt, c, h, w))


class VisualEncoder(nn.Module):
    def __init__(self, backbone="resnet18", channels=3, frames=16, dim=256, pretrained=None,
                 pretrained_sha256=None, mixstyle=0., partial_bn=False):
        super().__init__()
        self.frames, self.mixstyle, self.partial_bn = frames, mixstyle, partial_bn
        factories = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}
        net = factories[backbone](weights=None)
        if pretrained:
            if pretrained_sha256:
                digest = hashlib.sha256(Path(pretrained).read_bytes()).hexdigest()
                if digest != pretrained_sha256:
                    raise ValueError("Video pretraining checkpoint SHA256 mismatch")
            state = torch.load(Path(pretrained), map_location="cpu", weights_only=True)
            if "state_dict" in state:
                # Original MIT TSM release: shift wrappers do not add parameters.
                source = state["state_dict"]
                if backbone != "resnet50" or not all(k.startswith("module.base_model.") or k.startswith("module.new_fc.") for k in source):
                    raise ValueError("Expected the pinned original TSM ResNet50 checkpoint")
                mapped = {k.removeprefix("module.base_model.").replace(".conv1.net.", ".conv1."):v
                          for k,v in source.items() if k.startswith("module.base_model.")}
                missing, extra = net.load_state_dict(mapped, strict=False)
                if set(missing) != {"fc.weight", "fc.bias"} or extra:
                    raise ValueError(f"Incomplete video-pretrained backbone: {missing}, {extra}")
            else:
                net.load_state_dict(state, strict=True)
        if channels == 1:
            old = net.conv1
            net.conv1 = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
            with torch.no_grad():
                net.conv1.weight.copy_(old.weight.sum(1, keepdim=True))
        for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
            for block in layer:
                block.conv1 = TemporalShift(block.conv1, frames)
        self.feature_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.attention = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.projection = nn.Sequential(nn.Linear(self.feature_dim * 2, dim), nn.LayerNorm(dim), nn.GELU())

    def train(self, mode=True):
        super().train(mode)
        if mode and self.partial_bn:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d) and module is not self.backbone.bn1:
                    module.eval()
        return self

    def mix_style(self, x):
        if not self.training or self.mixstyle <= 0 or torch.rand((),device=x.device) >= self.mixstyle:
            return x
        bt,c,h,w = x.shape
        values = x.reshape(-1,self.frames,c,h,w)
        # One style per clip, shared by every frame. Validation never contributes.
        mean = values.float().mean((1,3,4),keepdim=True).detach()
        std = (values.float().var((1,3,4),keepdim=True,unbiased=False)+1e-6).sqrt().detach()
        coefficient = torch.distributions.Beta(.1,.1).sample((len(values),)).to(x.device).view(-1,1,1,1,1)
        order = torch.randperm(len(values),device=x.device)
        mixed_mean = coefficient*mean + (1-coefficient)*mean[order]
        mixed_std = coefficient*std + (1-coefficient)*std[order]
        return (((values.float()-mean)/std)*mixed_std+mixed_mean).to(x.dtype).reshape(bt,c,h,w)

    def forward(self, x):
        b, t, c, h, w = x.shape
        if self.mixstyle > 0:
            net = self.backbone
            values = net.maxpool(net.relu(net.bn1(net.conv1(x.reshape(b*t,c,h,w)))))
            values = self.mix_style(net.layer1(values))
            values = self.mix_style(net.layer2(values))
            values = net.layer4(net.layer3(values))
            features = net.avgpool(values).flatten(1).view(b,t,self.feature_dim)
        else:
            features = self.backbone(x.reshape(b * t, c, h, w)).view(b, t, self.feature_dim)
        weights = self.attention(features).softmax(dim=1)
        pooled = torch.cat(((features * weights).sum(1), features.amax(1)), dim=-1)
        return self.projection(pooled)


class TemporalEncoder(nn.Module):
    def __init__(self, channels, dim=256, width=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels, width, 5, padding=2), nn.GroupNorm(8, width), nn.GELU(),
                                 nn.Conv1d(width, width, 5, padding=4, dilation=2), nn.GroupNorm(8, width), nn.GELU(),
                                 nn.Conv1d(width, dim, 5, padding=8, dilation=4), nn.GroupNorm(8, dim), nn.GELU())
        self.attention = nn.Conv1d(dim, 1, 1)
        self.projection = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU())

    def forward(self, x, valid=None):
        features = self.net(x.transpose(1, 2))
        scores = self.attention(features).float()
        if valid is not None:
            # At least one finite entry even for a wholly absent modality; outer gate masks it.
            scores = scores.masked_fill(valid[:, None] <= 0, -1e4)
        weights = scores.softmax(-1).to(features.dtype)
        pooled = torch.cat(((features * weights).sum(-1), features.amax(-1)), dim=-1)
        return self.projection(pooled)


class SkeletonEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.encoder = TemporalEncoder(17 * 3 * 3 + 4, dim)
        self.register_buffer("parents", torch.tensor(PARENTS), persistent=False)

    def forward(self, x, valid):
        # H36M joints; z=0 is a legitimate coordinate, never a missing-value flag.
        root = x[:, :, :1]
        torso = (x[:, :, 8] - x[:, :, 0]).norm(dim=-1)
        scale = torso.masked_fill(valid <= 0, float("nan")).nanmedian(dim=1).values
        scale = torch.nan_to_num(scale, nan=1.).clamp_min(.05).view(-1, 1, 1, 1)
        joints = ((x - root) / scale).clamp(-10, 10)
        bones = joints - joints[:, :, self.parents]
        velocity = torch.cat((torch.zeros_like(joints[:, :1]), joints[:, 1:] - joints[:, :-1]), dim=1)
        root_delta = torch.cat((torch.zeros_like(root[:, :1]), root[:, 1:] - root[:, :-1]), dim=1) / scale
        values = torch.cat((joints.flatten(2), bones.flatten(2), velocity.flatten(2), root_delta.flatten(2), valid[..., None]), dim=-1)
        return self.encoder(values, valid)


class IMUEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.encoder = TemporalEncoder(35, dim, width=96)

    def forward(self, x, valid):
        values = torch.cat((x.flatten(2), valid[:, None].expand(-1, x.shape[1], -1)), -1)
        return self.encoder(values)


class RadarEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.points = nn.Sequential(nn.Linear(4, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 96), nn.GELU())
        self.encoder = TemporalEncoder(96, dim, width=96)

    def forward(self, x, valid):
        features = self.points(x).masked_fill(valid[..., None] <= 0, -1e4)
        features = features.amax(2)
        frames_valid = (valid.sum(-1) > 0).float()
        features = features * frames_valid[..., None]
        return self.encoder(features, frames_valid)


class MultiModalModel(nn.Module):
    def __init__(self, backbone="resnet18", frames=16, dim=256, pretrained=None, modalities=None, modality_dropout=.15,
                 sensor_family="legacy", fusion="gated", skeleton_config=None, pretrained_sha256=None,
                 mixstyle=0., partial_bn=False, visual_configs=None, shared_video_groups=None, feature_mixup=0., input_mixup=0., imu_config=None, radar_config=None,
                 shared_video_blocks=None, fusion_config=None):
        super().__init__()
        self.fusion = fusion
        if radar_config and sensor_family != "official":
            raise ValueError("Radar transfer configuration requires the official sensor adapter family")
        self.active = tuple(modalities or MODALITIES)
        if not self.active or set(self.active) - set(MODALITIES):
            raise ValueError("Invalid modality selection")
        self.modality_dropout = modality_dropout
        if not math.isfinite(feature_mixup) or feature_mixup < 0:
            raise ValueError("Feature MixUp alpha must be finite and nonnegative")
        self.feature_mixup = feature_mixup
        if not math.isfinite(input_mixup) or input_mixup < 0 or (input_mixup > 0 and feature_mixup > 0):
            raise ValueError("Use one finite nonnegative MixUp alpha: input or feature, not both")
        self.input_mixup = input_mixup
        visual_configs = visual_configs or {}
        if set(visual_configs) - {"thermal", "depth", "ir"}:
            raise ValueError("Visual configuration names must identify a camera")
        shared_video_groups = [] if shared_video_groups is None else shared_video_groups
        if not isinstance(shared_video_groups, (list, tuple)):
            raise ValueError("Shared video groups must be a list of camera groups")
        grouped = set()
        for group in shared_video_groups:
            if (not isinstance(group, (list, tuple)) or len(group) < 2
                    or any(not isinstance(name, str) for name in group)
                    or len(set(group)) != len(group) or set(group) - set(self.active)
                    or set(group) - {"thermal", "depth", "ir"} or set(group) & grouped):
                raise ValueError("Shared video groups require distinct active cameras, each in at most one group")
            grouped.update(group)
        self.shared_video_groups = tuple(tuple(group) for group in shared_video_groups)
        if shared_video_blocks is not None:
            if (not isinstance(shared_video_blocks, (list, tuple)) or not shared_video_blocks
                    or any(type(index) is not int or not 0 <= index < 12 for index in shared_video_blocks)
                    or list(shared_video_blocks) != sorted(set(shared_video_blocks))):
                raise ValueError("Shared VideoMAE block indices must be distinct, ordered integers in [0,11]")
            if not self.shared_video_groups or any(
                    visual_configs.get(name, {}).get("backbone", backbone) not in ("videomae_small", "videomae_base")
                    for group in self.shared_video_groups for name in group):
                raise ValueError("Explicit block sharing requires VideoMAE camera groups")
        self.shared_video_blocks = None if shared_video_blocks is None else tuple(shared_video_blocks)
        self.encoders = nn.ModuleDict()
        sensors = {"skeleton": SkeletonEncoder, "imu": IMUEncoder, "radar": RadarEncoder}
        if sensor_family == "official":
            from .official_models import OfficialSkeletonEncoder, OfficialIMUEncoder, OfficialRadarEncoder
            sensors = {"skeleton": OfficialSkeletonEncoder, "imu": OfficialIMUEncoder, "radar": OfficialRadarEncoder}
        elif sensor_family != "legacy":
            raise ValueError("Unknown sensor model family")
        for name in self.active:
            if name in ("thermal", "depth", "ir"):
                options = dict(backbone=backbone, pretrained=pretrained, pretrained_sha256=pretrained_sha256,
                               mixstyle=mixstyle, partial_bn=partial_bn)
                specification = visual_configs.get(name, {})
                allowed = set(options) | {"image_size", "normalization", "drop_path", "activation_checkpointing", "pretraining_classes", "augmentation", "intensity"}
                if set(specification) - allowed:
                    raise ValueError(f"Unknown visual settings for {name}: {set(specification) - allowed}")
                options.update({k:v for k,v in specification.items() if k in options})
                kind = options.pop("backbone")
                if kind in ("videomae_small", "videomae_base"):
                    from .mae_models import VideoMAESmallEncoder, VideoMAEBaseEncoder
                    if options.pop("mixstyle") or options.pop("partial_bn"):
                        raise ValueError("VideoMAE does not use CNN MixStyle or partial BatchNorm")
                    factory = VideoMAESmallEncoder if kind == "videomae_small" else VideoMAEBaseEncoder
                    self.encoders[name] = factory(
                        1 if name == "ir" else 3, frames, dim,
                        image_size=specification.get("image_size", 224),
                        drop_path=specification.get("drop_path", .1),
                        pretraining_classes=specification.get("pretraining_classes", 400),
                        activation_checkpointing=specification.get("activation_checkpointing", False), **options)
                elif kind == "swin3d_t":
                    if options.pop("mixstyle") or options.pop("partial_bn"):
                        raise ValueError("Video Swin does not use CNN MixStyle or partial BatchNorm")
                    if {"activation_checkpointing", "pretraining_classes"} & set(specification):
                        raise ValueError("VideoMAE-specific settings do not apply to Video Swin")
                    self.encoders[name] = SwinTinyEncoder(
                        1 if name == "ir" else 3, frames, dim,
                        drop_path=specification.get("drop_path", .1), **options)
                elif {"drop_path", "activation_checkpointing", "pretraining_classes"} & set(specification):
                    raise ValueError("Transformer video options require a VideoMAE encoder")
                elif kind == "r2plus1d34":
                    self.encoders[name] = R2Plus1DEncoder(1 if name == "ir" else 3, frames, dim, **options)
                else:
                    self.encoders[name] = VisualEncoder(kind, 1 if name == "ir" else 3, frames, dim, **options)
            else:
                sensor_options = skeleton_config if name == "skeleton" else imu_config if name == "imu" else radar_config
                options = dict(sensor_options or {}) if sensor_family == "official" else {}
                kind = options.pop("backbone", "official_small")
                if kind == "motionbert_lite":
                    from .skeleton_models import MotionBERTLiteEncoder
                    self.encoders[name] = MotionBERTLiteEncoder(dim, **options)
                elif kind == "limu_bert_x" and name == "imu":
                    from .imu_models import LIMUBertIMUEncoder
                    self.encoders[name] = LIMUBertIMUEncoder(dim=dim, **options)
                elif kind == "xfi_point_transformer" and name == "radar":
                    from .radar_models import XFiRadarEncoder
                    self.encoders[name] = XFiRadarEncoder(dim=dim, **options)
                elif kind == "official_small":
                    self.encoders[name] = sensors[name](dim, **options)
                else:
                    raise ValueError("Unknown skeleton architecture")
        for group in self.shared_video_groups:
            members = [self.encoders[name] for name in group]
            from .mae_models import VideoMAEEncoder
            if all(isinstance(encoder, VideoMAEEncoder) for encoder in members):
                signatures = {(encoder.variant, encoder.frames, encoder.image_size,
                               visual_configs.get(name, {}).get("drop_path", .1),
                               visual_configs.get(name, {}).get("activation_checkpointing", False),
                               visual_configs.get(name, {}).get("pretrained_sha256", pretrained_sha256))
                              for name, encoder in zip(group, members)}
                if len(signatures) != 1:
                    raise ValueError("Shared VideoMAE blocks require matching geometry and initialization")
                # Unselected blocks remain separate even when their pretrained
                # values match. Keep the original full-sharing module aliases.
                selected = self.shared_video_blocks
                for encoder in members[1:]:
                    if selected is None or len(selected) == len(members[0].backbone.blocks):
                        encoder.backbone.blocks = members[0].backbone.blocks
                    else:
                        for index in selected:
                            encoder.backbone.blocks[index] = members[0].backbone.blocks[index]
                continue
            if not all(isinstance(encoder, R2Plus1DEncoder) for encoder in members):
                raise ValueError("Share one matching R(2+1)D or VideoMAE encoder family")
            if len({encoder.partial_bn for encoder in members}) != 1:
                raise ValueError("Shared video groups require consistent partial BN settings")
            # Each camera retains its own input stem, normalization and projection.
            # Shared stages appear under multiple state keys; serialization preserves
            # those aliases rather than charging the file budget multiple times.
            for encoder in members[1:]:
                for stage in ("layer1", "layer2", "layer3", "layer4"):
                    setattr(encoder.backbone, stage, getattr(members[0].backbone, stage))
        self.cross_modal = None
        if fusion == "attention":
            from .official_models import ModalityAttention
            self.cross_modal = ModalityAttention(len(self.active), dim)
        elif fusion not in ("gated", "late", "residual_bottleneck"):
            raise ValueError("Unknown fusion architecture")
        self.branch_heads = nn.ModuleDict({name: nn.Linear(dim, 40) for name in self.active})
        self.gates = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim + 1, 64), nn.GELU(), nn.Linear(64, 1)) for name in self.active})
        if fusion not in ("late", "residual_bottleneck"):
            self.interaction = nn.Sequential(nn.Linear(dim * len(self.active), dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(.25))
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, 40)
        self.residual_fusion = None
        if fusion == "residual_bottleneck":
            from .residual_fusion import ResidualBottleneckFusion
            self.residual_fusion = ResidualBottleneckFusion(len(self.active), dim, **(fusion_config or {}))
        elif fusion_config:
            raise ValueError("Fusion configuration requires residual_bottleneck")
        self.register_buffer("modality_indices", torch.tensor([MODALITIES.index(name) for name in self.active]), persistent=False)

    def forward(self, batch, vision):
        features, logits, scores = [], [], []
        available = batch["availability"][:, self.modality_indices]
        quality = batch["quality"][:, self.modality_indices]
        if (available.sum(1) == 0).any():
            raise ValueError("Selected modalities leave a clip with no input")
        mask = available.clone()
        if self.training and self.modality_dropout > 0:
            keep = torch.rand_like(mask) >= self.modality_dropout
            mask = mask * keep
            empty = mask.sum(1) == 0
            mask[empty] = available[empty]
        mix = {}
        if self.training and self.input_mixup > 0:
            order, coefficient = sample_clip_mix(available, self.input_mixup)
            quality = mix_values(quality, order, coefficient)
            mix = dict(mixup_order=order, mixup_coefficient=coefficient)
        for k, name in enumerate(self.active):
            if name in vision:
                encoder = self.encoders[name]
                pixels = mix_values(vision[name], order, coefficient) if mix else vision[name]
                feature = (encoder(pixels, style_valid=available[:, k])
                           if isinstance(encoder, R2Plus1DEncoder) else encoder(pixels))
            else:
                feature = self.encoders[name](batch[name], batch[name + "_valid"])
                if mix:
                    # Raw skeleton/IMU/radar streams are encoded independently;
                    # their representations use the same mix as the video pixels.
                    feature = mix_values(feature, order, coefficient)
            features.append(feature)
            logits.append(self.branch_heads[name](feature))
            scores.append(self.gates[name](torch.cat((feature, quality[:, k:k + 1]), -1)))
        stacked = torch.stack(features, 1)
        branches = torch.stack(logits, 1)
        if self.training and self.feature_mixup > 0:
            stacked, branches, quality, order, coefficient = mix_clip_features(
                stacked, branches, quality, available, self.feature_mixup)
            mix = dict(mixup_order=order, mixup_coefficient=coefficient)
        if self.cross_modal is not None:
            stacked = self.cross_modal(stacked, mask)
        if self.cross_modal is not None or mix:
            scores = [self.gates[name](torch.cat((stacked[:, k], quality[:, k:k + 1]), -1))
                      for k, name in enumerate(self.active)]
        scores = torch.cat(scores, -1).float().masked_fill(mask <= 0, -1e4)
        weights = scores.softmax(-1).to(stacked.dtype)
        fused = (stacked * weights[..., None]).sum(1)
        residual_output = {}
        if self.fusion in ("late", "residual_bottleneck"):
            prediction = (branches * weights[...,None]).sum(1)
            if self.residual_fusion is not None:
                residual = self.residual_fusion(stacked, mask)
                residual_output = dict(late_logits=prediction, interaction_logits=residual)
                prediction = prediction + residual
        else:
            fused = self.norm(fused + self.interaction((stacked * mask[..., None]).flatten(1)))
            prediction = self.head(fused)
        return dict(logits=prediction, features=fused, branch_features=stacked, branch_logits=branches,
                    weights=weights, availability=available, fusion_mask=mask, **residual_output, **mix)


def inference_config(config):
    """Retain the architecture and transforms while removing all external assets."""
    result = copy.deepcopy(config)
    result.pop("domain_generalization", None)
    result.pop("source_sensor_initialization", None)
    result.pop("masked_prediction", None)
    result["pretrained"] = None
    for options in result.get("visual_configs", {}).values():
        if "pretrained" in options:
            options["pretrained"] = None
    if "pretrained" in result.get("skeleton_config", {}):
        result["skeleton_config"]["pretrained"] = None
    if "pretrained" in result.get("imu_config", {}):
        result["imu_config"]["pretrained"] = None
    if "pretrained" in result.get("radar_config", {}):
        result["radar_config"]["pretrained"] = None
    if result.get("input_crop") == "person_roi":
        result["detector"] = None
    return result


def build_model(config, pretrained=True):
    if not pretrained:
        config = inference_config(config)
    return MultiModalModel(backbone=config["backbone"], frames=config["frames"], dim=config.get("dim", 256),
                           pretrained=config.get("pretrained") if pretrained else None,
                           modalities=config.get("modalities"), modality_dropout=config.get("modality_dropout", .15),
                           sensor_family=config.get("sensor_family", "legacy"), fusion=config.get("fusion", "gated"),
                           skeleton_config=config.get("skeleton_config"), pretrained_sha256=config.get("pretrained_sha256"),
                           mixstyle=config.get("mixstyle",0.), partial_bn=config.get("partial_bn",False),
                           visual_configs=config.get("visual_configs"),
                           shared_video_groups=config.get("shared_video_groups"),
                           shared_video_blocks=config.get("shared_video_blocks"),
                           feature_mixup=config.get("feature_mixup", 0.), input_mixup=config.get("input_mixup", 0.),
                           imu_config=config.get("imu_config"), radar_config=config.get("radar_config"),
                           fusion_config=config.get("fusion_config"))


def classification_loss(output, labels, smoothing=.05, auxiliary=.3):
    mixed = "mixup_order" in output
    loss = F.cross_entropy(output["logits"].float(), labels, label_smoothing=smoothing,
                           reduction="none" if mixed else "mean")
    b, m, c = output["branch_logits"].shape
    branch = F.cross_entropy(output["branch_logits"].float().reshape(b * m, c), labels[:, None].expand(-1, m).reshape(-1),
                             label_smoothing=smoothing, reduction="none").view(b, m)
    branch = (branch * output["availability"]).sum(1) / output["availability"].sum(1).clamp_min(1)
    if mixed:
        paired = labels[output["mixup_order"]]
        second = F.cross_entropy(output["logits"].float(), paired, label_smoothing=smoothing, reduction="none")
        second_branch = F.cross_entropy(output["branch_logits"].float().reshape(b * m, c),
                paired[:, None].expand(-1, m).reshape(-1), label_smoothing=smoothing, reduction="none").view(b, m)
        second_branch = (second_branch * output["availability"]).sum(1) / output["availability"].sum(1).clamp_min(1)
        coefficient = output["mixup_coefficient"]
        return (coefficient * (loss + auxiliary * branch)
                + (1 - coefficient) * (second + auxiliary * second_branch)).mean()
    return loss + auxiliary * branch.mean()


def classification_correct(output, labels):
    """Training accuracy uses the same soft targets as feature MixUp's loss."""
    prediction = output["logits"].argmax(-1)
    correct = (prediction == labels).float()
    if "mixup_order" in output:
        coefficient = output["mixup_coefficient"]
        correct = coefficient * correct + (1 - coefficient) * (prediction == labels[output["mixup_order"]]).float()
    return correct.sum()


def distillation_loss(output, batch, temperature=3., weight=.7, feature_weight=.02):
    target = F.softmax(batch["teacher_logits"].float() / temperature, dim=-1)
    log_probs = F.log_softmax(output["logits"].float() / temperature, dim=-1)
    kl = F.kl_div(log_probs, target, reduction="batchmean") * temperature ** 2
    alignment = (1 - F.cosine_similarity(output["features"].float(), batch["teacher_features"].float(), dim=-1)).mean()
    return weight * kl + feature_weight * alignment
