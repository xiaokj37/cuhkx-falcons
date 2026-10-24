"""Memory-mapped datasets; raw archives are never opened by an epoch."""
import json
from pathlib import Path
import random
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import sample_indices

ARRAY_NAMES = ("vision", "skeleton", "skeleton_valid", "imu", "imu_valid", "radar", "radar_valid", "quality", "availability")


def read_meta(cache):
    path = Path(cache) / "cache.json"
    if not path.exists():
        raise ValueError(f"Cache is incomplete: {path}")
    return json.loads(path.read_text())


def validate_preprocessing(meta, config, provenance=None):
    actual = meta.get("preprocessing", {"mode": "full"})
    if actual["mode"] != config.get("input_crop", "full"):
        raise ValueError("Model configuration and cache crop mode differ")
    if "radar" in (config.get("modalities") or ["radar"]):
        from .radar_cache import RADAR_CONTRACT
        expected_radar = (RADAR_CONTRACT if config.get("radar_config", {}).get("backbone")
                          == "xfi_point_transformer" else None)
        if actual.get("radar") != expected_radar:
            raise ValueError("Radar cache channels/units differ from the model input contract")
    if provenance is not None and actual != provenance.get("preprocessing", {"mode": "full"}):
        raise ValueError("Training and inference preprocessing contracts differ")
    return actual


def warm_cache(cache):
    """One sequential read populates kernel page cache shared by all ranks/workers."""
    total = 0
    for name in ARRAY_NAMES:
        with (Path(cache) / f"{name}.npy").open("rb", buffering=0) as handle:
            while chunk := handle.read(64 * 1024 * 1024):
                total += len(chunk)
    return total


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


class ClipDataset(Dataset):
    def __init__(self, cache, indices=None, frames=16, training=False, teacher=None):
        self.path = Path(cache)
        self.meta = read_meta(cache)
        self.indices = np.arange(len(self.meta["records"])) if indices is None else np.asarray(indices)
        self.frames, self.training = frames, training
        self.arrays = None
        self.teacher = teacher
        if frames > self.meta["frames"]:
            raise ValueError("Requested more frames than the cache contains")

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["arrays"] = None
        return state

    def __getitem__(self, position):
        if self.arrays is None:
            self.arrays = {n: np.load(self.path / f"{n}.npy", mmap_mode="r") for n in ARRAY_NAMES}
        index = int(self.indices[position])
        length = self.meta["frames"]
        if self.training:
            edges = np.linspace(0, length, self.frames + 1).astype(int)
            times = np.array([np.random.randint(edges[i], max(edges[i + 1], edges[i] + 1)) for i in range(self.frames)])
        else:
            times = sample_indices(length, self.frames)
        item = {name: torch.from_numpy(np.array(a[index, times] if name == "vision" else a[index], copy=True))
                for name, a in self.arrays.items()}
        item["index"] = torch.tensor(index, dtype=torch.long)
        item["label"] = torch.tensor(self.meta["records"][index]["label"], dtype=torch.long)
        if self.teacher is not None:
            item["teacher_logits"] = torch.from_numpy(self.teacher["logits"][index].copy())
            item["teacher_features"] = torch.from_numpy(self.teacher["features"][index].copy())
        return item


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def erase_clip_regions(value, probability):
    """Mean-fill one rectangle consistently across frames and channels of a clip."""
    if not 0 <= probability <= 1:
        raise ValueError("Clip erasing probability must be in [0,1]")
    if probability == 0:
        return value
    b, _, _, h, w = value.shape
    device = value.device
    area = torch.empty(b, device=device).uniform_(.02, .2)
    aspect = torch.empty(b, device=device).uniform_(math.log(.5), math.log(2.)).exp()
    height = (torch.sqrt(area * aspect) * h).round().long().clamp(1, h)
    width = (torch.sqrt(area / aspect) * w).round().long().clamp(1, w)
    top = (torch.rand(b, device=device) * (h - height + 1)).long()
    left = (torch.rand(b, device=device) * (w - width + 1)).long()
    yy = torch.arange(h, device=device)[None, :, None]
    xx = torch.arange(w, device=device)[None, None, :]
    region = ((yy >= top[:, None, None]) & (yy < (top + height)[:, None, None])
              & (xx >= left[:, None, None]) & (xx < (left + width)[:, None, None]))
    region &= (torch.rand(b, device=device) < probability)[:, None, None]
    # Inputs are normalized, so zero is the corresponding camera-channel mean.
    return value.masked_fill(region[:, None, None], 0.)


def augmentation_range(options, name, default, maximum=None):
    values = options.get(name, default)
    if (not isinstance(values, (list, tuple)) or len(values) != 2
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
            or not 0 < values[0] <= values[1] or (maximum is not None and values[1] > maximum)):
        raise ValueError(f"Invalid {name} augmentation range")
    return values


def prepare_vision(batch, size, training, visual_configs=None):
    """Clip-consistent spatial augmentation on GPU, retaining uint8 in the cache."""
    import torch.nn.functional as F
    x = batch["vision"]
    b, t, _, h, w = x.shape
    result = {}
    for name, start, channels in (("thermal", 0, 3), ("depth", 3, 3), ("ir", 6, 1)):
        options = (visual_configs or {}).get(name, {})
        target_size = options.get("image_size", size)
        normalization = options.get("normalization", "imagenet")
        if normalization not in ("imagenet", "kinetics"):
            raise ValueError(f"Unknown camera normalization: {normalization}")
        v = x[:, :, start:start + channels].reshape(b * t, channels, h, w).float() / 255
        if options.get("intensity"):
            if name != "ir":
                raise ValueError("Intensity transforms are currently defined only for grayscale IR")
            from .intensity import normalize_ir_clip
            v = normalize_ir_clip(x[:, :, start:start + channels], options["intensity"]).reshape(b * t, channels, h, w)
        if training:
            augmentation = options.get("augmentation", {})
            if set(augmentation) - {"crop_scale", "gain", "contrast", "erase_prob"}:
                raise ValueError("Unknown video augmentation setting")
            # Independent cameras have independent crops; each crop stays fixed in time.
            scales = torch.empty(b, device=x.device).uniform_(*augmentation_range(augmentation, "crop_scale", (.85, 1.), 1.))
            shifts = (torch.rand(b, 2, device=x.device) * 2 - 1) * (1 - scales[:, None])
            theta = torch.zeros(b, 2, 3, device=x.device)
            theta[:, 0, 0] = theta[:, 1, 1] = scales
            theta[:, :, 2] = shifts
            theta = theta.repeat_interleave(t, dim=0)
            grid = F.affine_grid(theta, (b * t, channels, target_size, target_size), align_corners=False)
            v = F.grid_sample(v, grid, mode="bilinear", padding_mode="border", align_corners=False)
            # No unvalidated left/right label assumption or inconsistent sensor flip.
            gain = torch.empty(b, device=x.device).uniform_(*augmentation_range(augmentation, "gain", (.92, 1.08))).repeat_interleave(t)
            v = (v * gain[:, None, None, None]).clamp(0, 1)
            if "contrast" in augmentation:
                contrast = torch.empty(b, device=x.device).uniform_(*augmentation_range(augmentation, "contrast", (1., 1.)))
                mean = v.view(b, t, channels, target_size, target_size).mean((1, 2, 3, 4)).repeat_interleave(t)
                v = (mean[:, None, None, None] + contrast.repeat_interleave(t)[:, None, None, None]
                     * (v - mean[:, None, None, None])).clamp(0, 1)
        elif (h, w) != (target_size, target_size):
            v = F.interpolate(v, size=(target_size, target_size), mode="bilinear", align_corners=False, antialias=True)
        if normalization == "kinetics":
            means, stds = [.43216, .394666, .37645], [.22803, .22145, .216989]
            if channels == 1:
                means, stds = [sum(means) / 3], [sum(stds) / 3]
        else:
            means = [.485, .456, .406] if channels == 3 else [.449]
            stds = [.229, .224, .225] if channels == 3 else [.226]
        mean = v.new_tensor(means).view(1, channels, 1, 1)
        std = v.new_tensor(stds).view(1, channels, 1, 1)
        result[name] = ((v - mean) / std).view(b, t, channels, target_size, target_size)
        if training:
            result[name] = erase_clip_regions(result[name], augmentation.get("erase_prob", 0.))
    return result
