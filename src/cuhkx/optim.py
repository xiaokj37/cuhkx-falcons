"""Preserve existing CNN optimization; optionally decay pretrained ViT learning rates by depth."""
import torch


def make_optimizer(model, config):
    lr = config.get("lr", .0006)
    ratio = config.get("backbone_lr_ratio", .2)
    decay = config.get("video_layer_decay")
    skeleton_ratio = config.get("skeleton_backbone_lr_ratio")
    imu_ratio = config.get("imu_backbone_lr_ratio")
    imu_encoder_ratio = config.get("imu_encoder_lr_ratio")
    radar_ratio = config.get("radar_backbone_lr_ratio")
    if radar_ratio is not None and not 0 < radar_ratio <= 1:
        raise ValueError("Radar backbone learning-rate ratio must be in (0,1]")
    if imu_ratio is not None and not 0 < imu_ratio <= 1:
        raise ValueError("IMU backbone learning-rate ratio must be in (0,1]")
    if imu_encoder_ratio is not None and not 0 < imu_encoder_ratio <= 1:
        raise ValueError("IMU encoder learning-rate ratio must be in (0,1]")
    if skeleton_ratio is not None and not 0 < skeleton_ratio <= 1:
        raise ValueError("Skeleton backbone learning-rate ratio must be in (0,1]")
    weight_decay = config.get("weight_decay", .01)
    if decay is None and skeleton_ratio is None and imu_ratio is None and imu_encoder_ratio is None and radar_ratio is None:
        backbone, other = [], []
        for name, parameter in model.named_parameters():
            visual = any(f"encoders.{m}.backbone." in name for m in ("thermal", "depth", "ir"))
            (backbone if visual else other).append(parameter)
        return torch.optim.AdamW([dict(params=backbone, lr=lr * ratio),
                                  dict(params=other, lr=lr)], weight_decay=weight_decay)
    decay = 1. if decay is None else decay
    if not 0 < decay <= 1:
        raise ValueError("Video layer decay must be in (0,1]")
    unwrapped = model.module if hasattr(model, "module") else model
    grouped = {}
    for full_name, parameter in model.named_parameters():
        name = full_name.removeprefix("module.")
        scale = 1.
        for camera in ("thermal", "depth", "ir"):
            prefix = f"encoders.{camera}.backbone."
            if name.startswith(prefix):
                scale = ratio
                specification = config.get("visual_configs", {}).get(camera, {})
                if specification.get("backbone", config.get("backbone")) in ("videomae_small", "videomae_base"):
                    suffix = name.removeprefix(prefix)
                    last_layer = len(unwrapped.encoders[camera].backbone.blocks) + 1
                    layer = last_layer
                    if suffix.startswith("patch_embed."):
                        layer = 0
                    elif suffix.startswith("blocks."):
                        layer = int(suffix.split(".")[1]) + 1
                    scale *= decay ** (last_layer - layer)
                break
        no_decay = parameter.ndim <= 1 or name.endswith("relative_position_bias_table")
        if skeleton_ratio is not None and name.startswith("encoders.skeleton.backbone."):
            scale = skeleton_ratio
            no_decay |= name.endswith(("pos_embed", "temp_embed"))
        if imu_encoder_ratio is not None and name.startswith("encoders.imu."):
            scale = imu_encoder_ratio
        if name.startswith("encoders.imu.backbone."):
            # An explicitly configured public-backbone ratio overrides the whole encoder ratio.
            if imu_ratio is not None:
                scale = imu_ratio
            no_decay |= name.endswith("pos_embed.weight")
        if radar_ratio is not None and name.startswith("encoders.radar.backbone."):
            scale = radar_ratio
        wd = 0. if config.get("bias_norm_no_decay", True) and no_decay else weight_decay
        grouped.setdefault((scale, wd), []).append(parameter)
    return torch.optim.AdamW([dict(params=values, lr=lr * scale, weight_decay=wd)
                              for (scale, wd), values in grouped.items()])
