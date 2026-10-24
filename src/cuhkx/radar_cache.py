"""Append native SNR to an existing cache without decoding its videos again."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .cache import Source, read_csv_bytes, sample_indices, timestamp


RADAR_CONTRACT = dict(format="native_xyz_velocity_snr_v1", xyz_divisor=5.,
                      velocity_divisor=3., snr_divisor=1.,
                      point_selection="same_frames_and_descending_snr_as_legacy_cache")


def decode_radar_with_snr(source, record, steps, point_count):
    radar = np.zeros((steps, point_count, 5), dtype=np.float32)
    valid = np.zeros((steps, point_count), dtype=np.float32)
    frames = {}
    for name in source.files(record["prefixes"]["radar"]):
        if not name.lower().endswith(".csv"):
            continue
        rows = read_csv_bytes(source.read(name))
        if rows and rows[0][3:8] != ["x", "y", "z", "v", "snr"]:
            raise ValueError("Unexpected radar CSV field order")
        for row in rows[1:]:
            if len(row) < 8:
                continue
            try:
                values = np.asarray([float(v) for v in row[3:8]], dtype=np.float32)
                if np.isfinite(values).all():
                    frames.setdefault((timestamp(row[0]), int(row[1])), []).append(values)
            except ValueError:
                continue
    if frames:
        keys = sorted(frames)
        for t, index in enumerate(sample_indices(len(keys), steps)):
            points = np.asarray(frames[keys[index]])
            points = points[np.argsort(-points[:, 4], kind="stable")[:point_count]]
            radar[t, :len(points)] = points
            valid[t, :len(points)] = 1
        radar[..., :3] /= 5.
        radar[..., 3] /= 3.
        np.clip(radar[..., :4], -20, 20, out=radar[..., :4])
    return radar, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base, out = Path(args.base_cache).resolve(), Path(args.out).resolve()
    if base == out:
        raise ValueError("The radar extension needs a distinct output cache")
    parent_bytes = (base / "cache.json").read_bytes()
    parent = json.loads(parent_bytes)
    if parent["preprocessing"].get("radar"):
        raise ValueError("Expected the original four-channel radar cache")
    old_radar = np.load(base / "radar.npy", mmap_mode="r")
    old_valid = np.load(base / "radar_valid.npy", mmap_mode="r")
    expected = (len(parent["records"]), parent["steps"], parent["radar_points"], 4)
    if old_radar.shape != expected or old_radar.dtype != np.float32 or old_valid.shape != expected[:-1]:
        raise ValueError("Unexpected base radar geometry")
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = dict(parent_signature=parent["signature"],
                    parent_metadata_sha256=hashlib.sha256(parent_bytes).hexdigest(),
                    radar_decoder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    radar=RADAR_CONTRACT)
        signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        spec["signature"] = signature
        build = out / "build.json"
        if build.exists() and json.loads(build.read_text()) != spec:
            raise ValueError("Radar source or settings changed; choose a new output")
        build.write_text(json.dumps(spec, indent=2))
        if (out / "cache.json").exists():
            complete = json.loads((out / "cache.json").read_text())
            if complete["signature"] != signature:
                raise ValueError("Radar cache completion identity differs")
            print("Radar cache already complete:", out, flush=True)
            return
        # Read-only links reuse the unchanged arrays and their kernel page cache.
        for path in base.glob("*.npy"):
            if path.name in ("radar.npy", "done.npy"):
                continue
            target = out / path.name
            if target.exists() or target.is_symlink():
                if not target.is_symlink() or target.resolve() != path.resolve():
                    raise ValueError("Existing cache array is not the expected parent link")
            else:
                target.symlink_to(path)
        shape = expected[:-1] + (5,)
        target = out / "radar.npy"
        if target.is_symlink():
            raise ValueError("The extended radar array must not overwrite a parent link")
        radar = (np.load(target, mmap_mode="r+") if target.exists() else
                 np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=shape))
        done_path = out / "done.npy"
        if done_path.is_symlink():
            raise ValueError("Radar completion flags must be owned by the new cache")
        done = (np.load(done_path, mmap_mode="r+") if done_path.exists() else
                np.lib.format.open_memmap(done_path, mode="w+", dtype=np.uint8, shape=(shape[0],)))
        if radar.shape != shape or radar.dtype != np.float32 or done.shape != (shape[0],) or done.dtype != np.uint8:
            raise ValueError("Unexpected partial radar cache geometry")
        source = Source(parent["source"])
        if source.fingerprint != parent["source_fingerprint"]:
            raise ValueError("Raw data no longer match the base cache")
        completed = []
        for index in tqdm(np.flatnonzero(done == 0), unit="clip"):
            values, valid = decode_radar_with_snr(source, parent["records"][index], shape[1], shape[2])
            if not np.array_equal(values[..., :4], old_radar[index]) or not np.array_equal(valid, old_valid[index]):
                raise ValueError(f"Radar point/time alignment differs from the original cache at index {index}")
            radar[index] = values
            completed.append(index)
            if len(completed) >= 32:
                radar.flush()
                done[completed] = 1
                done.flush()
                completed.clear()
        radar.flush()
        done[completed] = 1
        done.flush()
        if not np.all(done == 1):
            raise ValueError("Radar cache is incomplete")
        meta = copy.deepcopy(parent)
        meta.update(signature=signature, radar_channels=5, radar_extension=spec,
                    radar_alignment_verified_clips=shape[0],
                    parent_cache=str(base), bytes=sum(p.stat().st_size for p in out.glob("*.npy")))
        meta["preprocessing"]["radar"] = RADAR_CONTRACT
        temporary = out / "cache.json.tmp"
        temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        os.replace(temporary, out / "cache.json")
        print(json.dumps(dict(cache=str(out), signature=signature, clips=shape[0],
                              reused_video_cache=True, original_radar_alignment_verified=True)), flush=True)


if __name__ == "__main__":
    main()
