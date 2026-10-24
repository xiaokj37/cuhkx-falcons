"""Add physical IMU orientation; preserve all existing cached inputs exactly."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .cache import Source, DEVICES, read_csv_bytes, timestamp
from .data import ARRAY_NAMES
from .tracking import atomic_json

ORIENTATION_CONTRACT = dict(version=1, quaternion_order='wxyz', devices=list(DEVICES),
    grid='Same 128 sample times as the original six-channel IMU cache',
    interpolation='Shortest-path spherical interpolation of normalized quaternions',
    maximum_gap_seconds=.5, boundary_hold_seconds=.1, norm_range=[.5, 1.5],
    missing='Zero quaternion plus a separate per-time per-device validity mask',
    unused_columns=['magnetic_field', 'temperature', 'firmware_version', 'battery', 'device_address', 'absolute_time'])


def interpolate_orientation(times, quaternions, grid):
    """Use physical time; do not bridge long missing intervals or angle wraps."""
    times, q, grid = np.asarray(times, float), np.asarray(quaternions, float), np.asarray(grid, float)
    if len(times) != len(q) or q.shape != (len(times), 4):
        raise ValueError('Quaternion and time shapes disagree')
    out, valid = np.zeros((len(grid), 4), np.float32), np.zeros(len(grid), np.float32)
    if not len(times):
        return out, valid
    if not np.isfinite(times).all() or not np.isfinite(q).all() or not np.isfinite(grid).all():
        raise ValueError('Nonfinite orientation input')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Orientation samples must have strictly increasing time')
    norm = np.linalg.norm(q, axis=1)
    if np.any((norm < .5) | (norm > 1.5)):
        raise ValueError('Quaternion norm is inconsistent with the sensor contract')
    q = q / norm[:, None]
    age = ORIENTATION_CONTRACT['boundary_hold_seconds']
    valid = ((grid >= times[0] - age) & (grid <= times[-1] + age)).astype(np.float32)
    if len(times) == 1:
        out[:] = q[0]
    else:
        bounded = grid.clip(times[0], times[-1])
        rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]])
        out[:] = Slerp(times, rotations)(bounded).as_quat()[:, [3, 0, 1, 2]]
        upper = np.searchsorted(times, bounded, side='right').clip(1, len(times) - 1)
        lower = upper - 1
        gap = times[upper] - times[lower]
        exact = (np.minimum(np.abs(bounded - times[lower]), np.abs(bounded - times[upper])) < 1e-7)
        valid *= (gap <= ORIENTATION_CONTRACT['maximum_gap_seconds']) | exact
    return out * valid[:, None], valid


def decode_orientation(source, record, steps=128):
    streams = {device: [] for device in DEVICES}
    all_times, agreement, stationary_agreement = [], [], []
    for member in source.files(record['prefixes']['imu']):
        if not member.lower().endswith('.csv'):
            continue
        rows = read_csv_bytes(source.read(member))
        if rows and rows[0][14:18] != [f'四元数{i}()' for i in range(4)]:
            raise ValueError('Unexpected quaternion column order')
        for row in rows[1:]:
            if len(row) < 18:
                continue
            match = re.match(r'WT(LA|RA|LL|RL|C)(?:\(|$)', row[1].strip())
            if not match:
                continue
            try:
                inertial = np.asarray(row[2:8], dtype=np.float64)
                if not np.isfinite(inertial).all():
                    continue
                t = timestamp(row[0])
            except ValueError:
                continue
            all_times.append(t)
            try:
                q = np.asarray(row[14:18], dtype=np.float64)
                norm = np.linalg.norm(q)
                if not np.isfinite(q).all() or not .5 <= norm <= 1.5:
                    continue
            except ValueError:
                continue
            streams[match.group(1)].append((t, q))
            if record['fold'] != 1:  # A format audit uses only original fold-1 source users.
                try:
                    angles = np.asarray(row[8:11], dtype=np.float64)
                    if np.isfinite(angles).all():
                        euler = Rotation.from_euler('ZYX', angles[[2, 1, 0]], degrees=True)
                        measured = Rotation.from_quat(q[[1, 2, 3, 0]])
                        error = float(np.rad2deg((euler.inv() * measured).magnitude()))
                        agreement.append(error)
                        if np.linalg.norm(inertial[3:]) < 1.:
                            stationary_agreement.append(error)
                except ValueError:
                    pass
    out, valid = np.zeros((steps, 5, 4), np.float32), np.zeros((steps, 5), np.float32)
    if all_times:
        origin = min(all_times)
        grid = np.linspace(0., max(all_times) - origin, steps)
        for device, name in enumerate(DEVICES):
            rows = sorted(streams[name], key=lambda row: row[0])
            if rows:
                t, first = np.unique([r[0] for r in rows], return_index=True)
                q = np.stack([rows[i][1] for i in first])
                out[:, device], valid[:, device] = interpolate_orientation(t - origin, q, grid)
    return out, valid, np.asarray(agreement, np.float32), np.asarray(stationary_agreement, np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-cache', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    base, out = args.base_cache.resolve(), args.out.resolve()
    assert base != out
    meta = json.loads((base / 'cache.json').read_text())
    assert meta['split'] == 'train' and len(meta['records']) == 3036
    assert not meta['preprocessing'].get('imu_orientation')
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = Source(meta['source'])
        assert source.fingerprint == meta['source_fingerprint'], 'Raw source differs from the verified parent cache'
        identity = dict(parent_signature=meta['signature'], parent_metadata_sha256=hashlib.sha256((base / 'cache.json').read_bytes()).hexdigest(),
                        source_fingerprint=source.fingerprint, orientation=ORIENTATION_CONTRACT,
                        decoder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        identity['signature'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if (out / 'build.json').exists() and json.loads((out / 'build.json').read_text()) != identity:
            raise ValueError('Orientation cache identity changed')
        atomic_json(out / 'build.json', identity)
        if (out / 'cache.json').exists():
            assert json.loads((out / 'cache.json').read_text())['signature'] == identity['signature']
            print('Orientation cache already complete', flush=True)
            return
        for name in ARRAY_NAMES:
            link = out / (name + '.npy')
            if not link.exists():
                link.symlink_to(base / link.name)
            assert link.resolve() == (base / link.name).resolve()
        parts = out / 'orientation_parts'
        parts.mkdir(exist_ok=True)
        def build(index):
            path = parts / (str(index).zfill(5) + '.npz')
            if not path.exists():
                q, valid, errors, stationary = decode_orientation(source, meta['records'][index], meta['steps'] * 2)
                temporary = path.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, quaternion=q, valid=valid, source_euler_errors=errors, source_stationary_errors=stationary)
                temporary.replace(path)
            return index
        with ThreadPoolExecutor(max_workers=4) as pool:
            for count, _ in enumerate(pool.map(build, range(3036)), 1):
                if count % 200 == 0 or count == 3036:
                    atomic_json(out / 'progress.json', dict(completed=count, total=3036))
                    print(f'Orientation {count}/3036', flush=True)
        q = np.lib.format.open_memmap(out / 'imu_quaternion.npy', mode='w+', dtype=np.float32, shape=(3036, meta['steps'] * 2, 5, 4))
        valid = np.lib.format.open_memmap(out / 'imu_quaternion_valid.npy', mode='w+', dtype=np.float32, shape=q.shape[:-1])
        errors, stationary = [], []
        for index in range(3036):
            with np.load(parts / (str(index).zfill(5) + '.npz')) as data:
                q[index], valid[index] = data['quaternion'], data['valid']
                errors.append(data['source_euler_errors']); stationary.append(data['source_stationary_errors'])
        assert np.isfinite(q).all() and np.isin(valid, [0, 1]).all()
        assert not q[valid == 0].any()
        q.flush(); valid.flush()
        errors, stationary = np.concatenate(errors), np.concatenate(stationary)
        audit = dict(source_fold=1, uses_outer_validation_or_test=False, euler_quaternion_comparisons=len(errors),
                     stationary_comparisons=len(stationary), quantile_levels=[0, .5, .9, .99, 1],
                     error_degrees=np.quantile(errors, [0, .5, .9, .99, 1]).tolist(),
                     stationary_error_degrees=np.quantile(stationary, [0, .5, .9, .99, 1]).tolist())
        atomic_json(out / 'orientation_format_audit.json', audit)
        result = copy.deepcopy(meta)
        result['preprocessing']['imu_orientation'] = ORIENTATION_CONTRACT
        result.update(signature=identity['signature'], orientation_build=identity)
        atomic_json(out / 'cache.json', result)
        print(json.dumps(dict(complete=True, quaternion_available_clips=int(valid.any((1, 2)).sum()), format_audit=audit)), flush=True)


if __name__ == '__main__':
    main()
