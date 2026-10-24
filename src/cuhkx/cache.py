"""Decode the official data once into contiguous, shared-page-cache arrays."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import threading
import time
import zipfile

import cv2
import numpy as np
from tqdm import tqdm

from . import CACHE_VERSION, MODALITIES

DIRECTORIES = dict(zip(MODALITIES, ("Thermal", "Depth_Color", "IR", "Skeleton", "IMU", "Radar")))
DEVICES = ("LA", "RA", "C", "LL", "RL")
TRAIN_SUBJECTS = set(range(1, 10)) | set(range(16, 25))


def natural_key(value):
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", value))


def clean_member(name):
    return not any(p.startswith(".") or p == "__MACOSX" for p in Path(name).parts)


def sample_indices(length, count):
    if length < 1:
        return np.zeros(count, dtype=np.int64)
    return np.minimum(((np.arange(count) + .5) * length / count).astype(np.int64), length - 1)


def resample(sequence, count, times=None):
    """Linear interpolation in time, preserving chronological order."""
    x = np.asarray(sequence, dtype=np.float32)
    if not len(x):
        raise ValueError("Cannot resample an empty sequence")
    if len(x) == 1:
        return np.repeat(x, count, axis=0)
    times = np.arange(len(x), dtype=np.float64) if times is None else np.asarray(times, dtype=np.float64)
    order = np.argsort(times, kind="stable")
    times, x = times[order], x[order]
    times, unique = np.unique(times, return_index=True)
    x = x[unique]
    grid = np.linspace(times[0], times[-1], count)
    flat = x.reshape(len(x), -1)
    result = np.stack([np.interp(grid, times, flat[:, k]) for k in range(flat.shape[1])], axis=1)
    return result.reshape((count,) + x.shape[1:]).astype(np.float32)


def timestamp(value):
    # Absolute dates never become model inputs. Only within-clip intervals are used.
    value = value.strip()
    try:
        return float(np.datetime64(value, "ms").astype(np.int64)) / 1000
    except ValueError:
        # The June 17 recordings use unpadded month/day/seconds.
        pattern = "%Y-%m-%d %H:%M:%S.%f" if "." in value else "%Y-%m-%d %H:%M:%S"
        return (datetime.strptime(value, pattern) - datetime(1970, 1, 1)).total_seconds()


def read_csv_bytes(blob):
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return list(csv.reader(io.StringIO(blob.decode(encoding))))
        except UnicodeDecodeError:
            pass
    raise ValueError("Unrecognized sensor CSV encoding")


class Source:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.is_zip = self.path.is_file()
        self.local = threading.local()
        if self.is_zip:
            with zipfile.ZipFile(self.path) as archive:
                self.entries = {x.filename: (x.file_size, x.CRC) for x in archive.infolist()
                                if not x.is_dir() and clean_member(x.filename)}
        else:
            self.entries = {p.relative_to(self.path).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
                            for p in self.path.rglob("*") if p.is_file() and clean_member(p.relative_to(self.path))}
        if not self.entries:
            raise ValueError("Source has no usable entries")
        self.prefixes = {}
        for name in self.entries:
            parent = str(Path(name).parent)
            while parent != ".":
                self.prefixes.setdefault(parent, []).append(name)
                parent = str(Path(parent).parent)
        digest = hashlib.sha256()
        for name, stat in sorted(self.entries.items()):
            digest.update(f"{name}\t{stat}\n".encode())
        self.fingerprint = digest.hexdigest()

    def files(self, prefix):
        return sorted(self.prefixes.get(prefix.rstrip("/"), []), key=natural_key)

    def read(self, name):
        if not self.is_zip:
            return (self.path / name).read_bytes()
        if not hasattr(self.local, "archive"):
            self.local.archive = zipfile.ZipFile(self.path)
        # ZipFile verifies each decoded entry's CRC.
        return self.local.archive.read(name)


def records_from_manifest(path, split):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8-sig", newline="")))
    records = []
    for row in rows:
        if split == "train":
            subject = int(row["subject"].removeprefix("user"))
            label, fold = int(row["action_id"]), int(row["validation_fold"])
            if subject not in TRAIN_SUBJECTS or not 0 <= label < 40 or fold not in (1, 2, 3):
                raise ValueError(f"Invalid training identity/label/fold: {row}")
            key = f'{row["action"]}/{row["subject"]}/{row["trial"]}'
            prefixes = {m: row[f"{d}_path"] for m, d in DIRECTORIES.items()}
            records.append(dict(key=key, path=key, label=label, subject=subject, fold=fold, prefixes=prefixes))
        else:
            original = row["path"]
            clip = Path(original.rstrip("/")).name
            prefix = "small_model_track_test/" + clip
            records.append(dict(key=clip, path=original, label=-1, subject=-1, fold=-1,
                                prefixes={m: f"{prefix}/{d}" for m, d in DIRECTORIES.items()}))
    if len({r["key"] for r in records}) != len(records):
        raise ValueError("Duplicate clip keys in manifest")
    if split == "train":
        by_subject = {}
        for r in records:
            by_subject.setdefault(r["subject"], set()).add(r["fold"])
        if any(len(x) != 1 for x in by_subject.values()):
            raise ValueError("A subject occurs in multiple validation folds")
    return records


def decode_clip(source, record, frames, size, steps, radar_points, rois=None):
    images = np.zeros((frames, 7, size, size), dtype=np.uint8)
    skeleton = np.zeros((steps, 17, 3), dtype=np.float32)
    skeleton_valid = np.zeros(steps, dtype=np.float32)
    imu = np.zeros((steps * 2, 5, 6), dtype=np.float32)
    imu_valid = np.zeros(5, dtype=np.float32)
    radar = np.zeros((steps, radar_points, 4), dtype=np.float32)
    radar_valid = np.zeros((steps, radar_points), dtype=np.float32)
    quality = np.zeros(6, dtype=np.float32)
    counts = np.zeros(6, dtype=np.int32)
    errors = []
    files = {m: source.files(p) if p else [] for m, p in record["prefixes"].items()}
    for index, (modality, channel, channels) in enumerate((("thermal", 0, 3), ("depth", 3, 3), ("ir", 6, 1))):
        names = [n for n in files[modality] if Path(n).suffix.lower() in (".png", ".jpg", ".jpeg")
                 and "/visualizations/" not in n]
        counts[index] = len(names)
        if not names:
            continue
        decoded = {}
        for k in np.unique(sample_indices(len(names), frames)):
            n = names[k]
            try:
                flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
                im = cv2.imdecode(np.frombuffer(source.read(n), dtype=np.uint8), flag)
                if im is None:
                    raise ValueError("Image decoder returned no pixels")
                if rois is not None:
                    from .roi import crop_pixels
                    im = crop_pixels(im, rois[modality]["box"])
                if channels == 3:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                else:
                    im = im[..., None]
                im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
                if im.ndim == 2:
                    im = im[..., None]
                decoded[int(k)] = im.transpose(2, 0, 1)
            except (ValueError, OSError) as exc:
                errors.append(dict(file=n, error=str(exc)))
        if decoded:
            good = np.array(sorted(decoded))
            for t, k in enumerate(sample_indices(len(names), frames)):
                nearest = int(good[np.abs(good - k).argmin()])
                images[t, channel:channel + channels] = decoded[nearest]
            quality[index] = len(decoded) / len(np.unique(sample_indices(len(names), frames)))

    poses, valid = [], []
    for n in files["skeleton"]:
        if not n.endswith(".json") or "/predictions/" not in n:
            continue
        try:
            people = json.loads(source.read(n))
            candidates = [p for p in people if isinstance(p, dict) and "keypoints" in p] if isinstance(people, list) else []
            candidates.sort(key=lambda p: float(np.mean(p.get("keypoint_scores", p.get("scores", [1])))), reverse=True)
            p = np.asarray(candidates[0]["keypoints"], dtype=np.float32) if candidates else None
            ok = p is not None and p.shape == (17, 3) and np.isfinite(p).all()
            poses.append(p if ok else np.zeros((17, 3), dtype=np.float32))
            valid.append(float(ok))
        except (ValueError, TypeError, OSError) as exc:
            errors.append(dict(file=n, error=str(exc)))
    if any(valid):
        good = np.flatnonzero(valid)
        target = np.linspace(0, len(poses) - 1, steps)
        poses = np.asarray(poses)[good]
        flat = poses.reshape(len(good), -1)
        skeleton = np.stack([np.interp(target, good, flat[:, k]) for k in range(51)], 1).reshape(steps, 17, 3).astype(np.float32)
        skeleton_valid = np.interp(target, np.arange(len(valid)), valid).astype(np.float32)
        quality[3], counts[3] = float(np.mean(valid)), int(sum(valid))

    device_rows = {name: [] for name in DEVICES}
    for n in files["imu"]:
        if not n.lower().endswith(".csv"):
            continue
        for row in read_csv_bytes(source.read(n))[1:]:
            if len(row) < 8:
                continue
            match = re.match(r"WT(LA|RA|LL|RL|C)(?:\(|$)", row[1].strip())
            if not match:
                continue
            try:
                values = np.asarray([float(x) for x in row[2:8]], dtype=np.float32)
                if np.isfinite(values).all():
                    device_rows[match.group(1)].append((timestamp(row[0]), values))
            except ValueError:
                continue
    all_times = [t for rows in device_rows.values() for t, _ in rows]
    if all_times:
        grid = np.linspace(min(all_times), max(all_times), steps * 2)
        for k, device in enumerate(DEVICES):
            rows = sorted(device_rows[device], key=lambda x: x[0])
            if not rows:
                continue
            times, first = np.unique([r[0] for r in rows], return_index=True)
            values = np.stack([rows[i][1] for i in first])
            imu[:, k] = np.stack([np.interp(grid, times, values[:, j]) for j in range(6)], 1)
            imu_valid[k] = 1
        imu[..., :3] /= 4.0
        imu[..., 3:] *= np.pi / 180 / 5.0
        np.clip(imu, -20, 20, out=imu)
        quality[4], counts[4] = imu_valid.mean(), len(all_times)

    radar_frames = {}
    for n in files["radar"]:
        if not n.lower().endswith(".csv"):
            continue
        for row in read_csv_bytes(source.read(n))[1:]:
            if len(row) < 8:
                continue
            try:
                values = np.asarray([float(x) for x in row[3:8]], dtype=np.float32)
                if np.isfinite(values).all():
                    radar_frames.setdefault((timestamp(row[0]), int(row[1])), []).append(values)
            except ValueError:
                continue
    if radar_frames:
        keys = sorted(radar_frames)
        for t, k in enumerate(sample_indices(len(keys), steps)):
            points = np.asarray(radar_frames[keys[k]])
            order = np.argsort(-points[:, 4], kind="stable")[:radar_points]
            points = points[order, :4]
            radar[t, :len(points)] = points
            radar_valid[t, :len(points)] = 1
        # Fixed physical scaling; no test-set statistics or absolute timestamps.
        radar[..., :3] /= 5.0
        radar[..., 3] /= 3.0
        np.clip(radar, -20, 20, out=radar)
        counts[5] = sum(len(v) for v in radar_frames.values())
        quality[5] = min(len(keys) / steps, 1.0)
    return dict(vision=images, skeleton=skeleton, skeleton_valid=skeleton_valid, imu=imu,
                imu_valid=imu_valid, radar=radar, radar_valid=radar_valid, quality=quality,
                availability=(quality > 0).astype(np.float32), counts=counts), errors


def cache_shapes(n, frames, size, steps, radar_points):
    return {"vision": ((n, frames, 7, size, size), np.uint8),
            "skeleton": ((n, steps, 17, 3), np.float32), "skeleton_valid": ((n, steps), np.float32),
            "imu": ((n, steps * 2, 5, 6), np.float32), "imu_valid": ((n, 5), np.float32),
            "radar": ((n, steps, radar_points, 4), np.float32), "radar_valid": ((n, steps, radar_points), np.float32),
            "quality": ((n, 6), np.float32), "availability": ((n, 6), np.float32), "counts": ((n, 6), np.int32)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Merged ZIP, or directory containing HAR / small_model_track_test")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True, choices=("train", "test"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--radar-points", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--roi-manifest", help="Verified independent-camera crops; omitted = full image")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Do not allow concurrent writers, including against a half-finished cache.
    import fcntl
    lock = (out / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = Source(args.source)
    records = records_from_manifest(args.manifest, args.split)
    roi = None
    if args.roi_manifest:
        from .roi import validate_manifest
        roi = json.loads(Path(args.roi_manifest).read_text())
        validate_manifest(roi, source, records, args.manifest)
    spec = dict(version=CACHE_VERSION, split=args.split, frames=args.frames, size=args.size,
                steps=args.steps, radar_points=args.radar_points, modalities=list(MODALITIES),
                source_fingerprint=source.fingerprint,
                decoder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest())
    spec["preprocessing"] = roi["preprocessing"] if roi else {"mode": "full"}
    if roi:
        spec["roi_manifest_sha256"] = hashlib.sha256(Path(args.roi_manifest).read_bytes()).hexdigest()
        spec["roi_decoder_sha256"] = hashlib.sha256(Path(__file__).with_name("roi.py").read_bytes()).hexdigest()
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    spec["signature"] = signature
    spec_path = out / "build.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("Cache settings/source changed; choose a new output directory")
    spec_path.write_text(json.dumps(spec, indent=2))
    if (out / "cache.json").exists():
        print("Cache already complete:", out, flush=True)
        return
    arrays = {}
    for name, (shape, dtype) in cache_shapes(len(records), args.frames, args.size, args.steps, args.radar_points).items():
        path = out / f"{name}.npy"
        arrays[name] = np.load(path, mmap_mode="r+") if path.exists() else np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"Invalid partial cache array {name}")
    progress_path = out / "done.npy"
    done = np.load(progress_path, mmap_mode="r+") if progress_path.exists() else np.lib.format.open_memmap(progress_path, mode="w+", dtype=np.uint8, shape=(len(records),))
    errors_path = out / "decode_errors.jsonl"
    pending = np.flatnonzero(done == 0).tolist()

    def work(index):
        values, errors = decode_clip(source, records[index], args.frames, args.size, args.steps, args.radar_points,
                                     roi["clips"][records[index]["key"]] if roi else None)
        for name, value in values.items():
            arrays[name][index] = value
        return index, errors

    # Bounded futures prevent retaining thousands of decoded clips in RAM.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor, errors_path.open("a") as log, tqdm(total=len(pending), unit="clip") as bar:
        iterator = iter(pending)
        futures = {executor.submit(work, i) for i in list(next(iterator, None) for _ in range(args.workers * 2)) if i is not None}
        completed_batch = []
        while futures:
            ready, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in ready:
                index, errors = future.result()
                if errors:
                    log.write(json.dumps(dict(index=index, key=records[index]["key"], errors=errors)) + "\n")
                    log.flush()
                completed_batch.append(index)
                bar.update()
                next_index = next(iterator, None)
                if next_index is not None:
                    futures.add(executor.submit(work, next_index))
            if len(completed_batch) >= 32 or not futures:
                # Flush payload before marking indices complete for safe resume.
                for array in arrays.values():
                    array.flush()
                done[completed_batch] = 1
                done.flush()
                completed_batch.clear()
    empty = np.flatnonzero(arrays["availability"].sum(1) == 0).tolist()
    if empty:
        raise ValueError(f"Clips have no readable modality: {empty}; inspect source before training")
    meta = dict(**spec, records=records, source=str(source.path),
                availability_counts=dict(zip(MODALITIES, arrays["availability"].sum(0).astype(int).tolist())),
                elapsed_seconds=time.time() - start, bytes=sum(p.stat().st_size for p in out.glob("*.npy")))
    tmp = out / "cache.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    os.replace(tmp, out / "cache.json")
    print(json.dumps({k: v for k, v in meta.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
