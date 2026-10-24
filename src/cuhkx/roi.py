"""Deterministic clip-level person crops, computed from original images once."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

VISION_MODALITIES = ("thermal", "depth", "ir")
ROI_DEFAULTS = dict(probes=8, confidence=.25, margin=1.4, min_side=.35, min_hits=2, imgsz=640)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def pack_detector(weights):
    """Convert the trusted official asset to tensors/config, without pickle objects."""
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO, __version__
    yolo = YOLO(str(weights), task="detect")
    state = {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
             for k, v in yolo.model.state_dict().items()}
    return dict(format="ultralytics-state-v1", yaml=yolo.model.yaml, state=state,
                source_sha256=sha256(weights), state_sha256=state_digest(state),
                ultralytics_version=str(__version__))


def detector_from_asset(asset):
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO, __version__
    from ultralytics.nn.tasks import DetectionModel
    if str(__version__) != asset["ultralytics_version"]:
        raise ValueError("Detector runtime version differs from the exported preprocessing contract")
    if state_digest(asset["state"]) != asset["state_sha256"]:
        raise ValueError("Detector asset tensor digest mismatch")
    # Construct from configuration; no network access or external weights needed.
    yolo = YOLO("yolo11n.yaml", task="detect")
    yolo.model = DetectionModel(asset["yaml"], ch=3, nc=80, verbose=False)
    yolo.model.load_state_dict(asset["state"], strict=True)
    yolo.model.eval()
    return yolo


def crop_pixels(image, box):
    if box is None:
        return image
    h, w = image.shape[:2]
    x0, y0 = np.floor(np.asarray(box[:2]) * [w, h]).astype(int)
    x1, y1 = np.ceil(np.asarray(box[2:]) * [w, h]).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Empty person crop")
    return image[y0:y1, x0:x1]


def stable_box(boxes, width, height, margin=1.4, min_side=.35):
    """Union in original pixel coordinates, expanded to retain hands/objects."""
    if not boxes:
        return None
    points = np.asarray(boxes, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError("Malformed detector boxes")
    x0, y0 = points[:, :2].min(0) * [width, height]
    x1, y1 = points[:, 2:].max(0) * [width, height]
    side = max(max(x1 - x0, y1 - y0) * margin, min_side * max(width, height))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return np.clip([(cx - side / 2) / width, (cy - side / 2) / height,
                    (cx + side / 2) / width, (cy + side / 2) / height], 0, 1).tolist()


def contract(asset, settings):
    return dict(mode="person_roi", implementation="independent-camera-union-v1",
                detector_state_sha256=asset["state_sha256"],
                ultralytics_version=asset["ultralytics_version"], settings=settings)


def validate_manifest(value, source, records, manifest_path):
    if value["source_fingerprint"] != source.fingerprint:
        raise ValueError("ROI manifest belongs to a different source archive")
    if value["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("ROI clip manifest mismatch")
    if set(value["clips"]) != {r["key"] for r in records}:
        raise ValueError("ROI manifest is incomplete or contains foreign clips")
    for modalities in value["clips"].values():
        for modality in VISION_MODALITIES:
            box = modalities[modality]["box"]
            if box is not None and (len(box) != 4 or not np.isfinite(box).all() or
                    not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
                raise ValueError("Invalid normalized ROI coordinates")


def build_rois(source, records, manifest_path, asset, out, device="0", settings=None):
    from .cache import sample_indices
    from tqdm import tqdm
    settings = dict(ROI_DEFAULTS if settings is None else settings)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with out.with_suffix(out.suffix + ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = dict(source_fingerprint=source.fingerprint, manifest_sha256=sha256(manifest_path),
                    preprocessing=contract(asset, settings))
        partial = out.with_suffix(out.suffix + ".partial")
        value = json.loads(out.read_text()) if out.exists() else json.loads(partial.read_text()) if partial.exists() else {**spec, "clips": {}}
        if any(value[k] != v for k, v in spec.items()):
            raise ValueError("ROI settings/source changed; use a new output")
        if out.exists():
            validate_manifest(value, source, records, manifest_path)
            return value
        detector = detector_from_asset(asset)
        for record in tqdm(records, desc="person ROI", unit="clip"):
            if record["key"] in value["clips"]:
                continue
            modalities = {}
            for name in VISION_MODALITIES:
                prefix = record["prefixes"][name]
                names = [n for n in source.files(prefix) if Path(n).suffix.lower() in (".png", ".jpg", ".jpeg")
                         and "/visualizations/" not in n] if prefix else []
                images = []
                for index in np.unique(sample_indices(len(names), settings["probes"])) if names else []:
                    image = cv2.imdecode(np.frombuffer(source.read(names[index]), np.uint8), cv2.IMREAD_COLOR)
                    if image is not None and image.any():
                        images.append(image)
                boxes, confidences = [], []
                if images:
                    # Different cameras never share boxes. BGR arrays match Ultralytics input.
                    results = detector.predict(images, classes=[0], conf=settings["confidence"],
                                               imgsz=settings["imgsz"], device=device, verbose=False,
                                               half=False, augment=False, batch=len(images))
                    for result in results:
                        if len(result.boxes):
                            selected = int(result.boxes.conf.argmax())
                            h, w = result.orig_shape
                            boxes.append((result.boxes.xyxy[selected].cpu().numpy() / [w, h, w, h]).tolist())
                            confidences.append(float(result.boxes.conf[selected]))
                enough = len(boxes) >= min(settings["min_hits"], len(images)) and bool(images)
                h, w = images[0].shape[:2] if images else (1, 1)
                box = stable_box(boxes, w, h, settings["margin"], settings["min_side"]) if enough else None
                modalities[name] = dict(box=box, probes=len(images), hits=len(boxes),
                                         mean_confidence=float(np.mean(confidences)) if confidences else 0.,
                                         fallback="insufficient_detections" if box is None else None)
            value["clips"][record["key"]] = modalities
            if len(value["clips"]) % 10 == 0:
                temporary = partial.with_suffix(".tmp")
                temporary.write_text(json.dumps(value))
                temporary.replace(partial)
        validate_manifest(value, source, records, manifest_path)
        value["coverage"] = {m: dict(cropped=sum(v[m]["box"] is not None for v in value["clips"].values()),
                                    total=len(records)) for m in VISION_MODALITIES}
        partial.write_text(json.dumps(value, indent=2))
        partial.replace(out)
        print(json.dumps(value["coverage"]), flush=True)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--detector", default="pretrained/yolo11n.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    from .cache import Source, records_from_manifest
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    build_rois(Source(args.source), records_from_manifest(args.manifest, args.split),
               args.manifest, pack_detector(args.detector), args.out, args.device)


if __name__ == "__main__":
    main()
