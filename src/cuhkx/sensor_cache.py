"""Cache native sensor samples and relative time once; never load videos here."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .cache import Source, DEVICES, read_csv_bytes, records_from_manifest, timestamp
from .tracking import atomic_json

CONTRACT = dict(version=1, devices=list(DEVICES),
                imu_channels=['acc_x_g', 'acc_y_g', 'acc_z_g', 'gyro_x_deg_s', 'gyro_y_deg_s', 'gyro_z_deg_s'],
                radar_channels=['x_m', 'y_m', 'z_m', 'velocity_m_s', 'native_snr', 'native_noise'],
                timestamps='seconds relative to the earliest sample of this modality within this clip',
                duplicates='stable first sample per IMU device and timestamp',
                normalization='none; no learned statistics', invalid_rows='discard nonfinite or malformed sensor rows')


def decode_native(source, record):
    streams = {device: [] for device in DEVICES}
    for name in source.files(record['prefixes']['imu']):
        if not name.lower().endswith('.csv'):
            continue
        for row in read_csv_bytes(source.read(name))[1:]:
            if len(row) < 8:
                continue
            match = re.match(r'WT(LA|RA|LL|RL|C)(?:\(|$)', row[1].strip())
            if not match:
                continue
            try:
                values = np.asarray([float(v) for v in row[2:8]], dtype=np.float32)
                if np.isfinite(values).all():
                    streams[match.group(1)].append((timestamp(row[0]), values))
            except ValueError:
                pass
    imu_origin = min((t for stream in streams.values() for t, _ in stream), default=0.)
    values, times, offsets = [], [], [0]
    for stream in streams.values():
        stream.sort(key=lambda row: row[0])
        if stream:
            t, indices = np.unique([row[0] for row in stream], return_index=True)
            values.extend(stream[index][1] for index in indices)
            times.extend(t - imu_origin)
        offsets.append(len(values))
    radar_frames = {}
    for name in source.files(record['prefixes']['radar']):
        if not name.lower().endswith('.csv'):
            continue
        rows = read_csv_bytes(source.read(name))
        if rows and rows[0][3:9] != ['x', 'y', 'z', 'v', 'snr', 'noise']:
            raise ValueError('Unexpected native radar fields')
        for row in rows[1:]:
            if len(row) < 9:
                continue
            try:
                point = np.asarray([float(v) for v in row[3:9]], dtype=np.float32)
                if np.isfinite(point).all():
                    radar_frames.setdefault((timestamp(row[0]), int(row[1])), []).append(point)
            except ValueError:
                pass
    keys = sorted(radar_frames)
    radar_origin = keys[0][0] if keys else 0.
    points, frame_offsets = [], [0]
    for key in keys:
        points.extend(radar_frames[key])
        frame_offsets.append(len(points))
    return dict(imu_values=np.asarray(values, dtype=np.float32).reshape(-1, 6),
                imu_times=np.asarray(times, dtype=np.float32), imu_offsets=np.asarray(offsets, dtype=np.int64),
                radar_points=np.asarray(points, dtype=np.float32).reshape(-1, 6),
                radar_times=np.asarray([key[0] - radar_origin for key in keys], dtype=np.float32),
                radar_offsets=np.asarray(frame_offsets, dtype=np.int64))


def imagebind_windows(values, times, offsets, order='gyro_acc'):
    """Fixed physical 5-second windows, linearly interpolated onto 2000 samples.

    Short clips use centered, edge-held windows. This does not invent measured
    high-frequency motion. Units are SI. Both channel orders can be predeclared
    as transfer controls: the public release does not provide an IMU CSV loader.
    No absolute time, user, label, or path enters the encoder.
    """
    if order not in ('gyro_acc', 'acc_gyro'):
        raise ValueError('Unknown IMU channel order')
    if tuple(offsets.shape) != (6,) or offsets[0] != 0 or offsets[-1] != len(values):
        raise ValueError('Expected five device streams')
    if np.any(np.diff(offsets) < 0) or values.shape != (len(times), 6):
        raise ValueError('Invalid native IMU array geometry')
    duration = float(times.max()) if len(times) else 0.
    count = max(1, int(np.ceil(duration / 5.)))
    centers = np.linspace(2.5, duration - 2.5, count) if duration > 5 else np.array([duration / 2])
    windows = np.zeros((count, 5, 6, 2000), dtype=np.float32)
    valid = np.zeros((count, 5), dtype=np.float32)
    coverage = np.zeros_like(valid)
    for device in range(5):
        start, stop = offsets[device:device + 2]
        if stop == start:
            continue
        t, x = times[start:stop], values[start:stop].copy()
        if np.any(np.diff(t) <= 0) or not np.isfinite(x).all():
            raise ValueError('Native samples must have strictly increasing times and finite values')
        x[:, :3] *= 9.80665
        x[:, 3:] *= np.pi / 180.
        if order == 'gyro_acc':
            x = x[:, [3, 4, 5, 0, 1, 2]]
        for window, center in enumerate(centers):
            grid = center + (np.arange(2000, dtype=np.float64) + .5) / 400. - 2.5
            if grid[-1] < t[0] or grid[0] > t[-1]:
                continue
            windows[window, device] = np.stack([np.interp(grid, t, x[:, channel]) for channel in range(6)])
            valid[window, device] = 1
            coverage[window, device] = ((grid >= t[0]) & (grid <= t[-1])).mean()
    return windows, valid, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--split', choices=['train', 'test'], required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    records = records_from_manifest(args.manifest, args.split)
    args.out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.out / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = Source(args.source)
        identity = dict(contract=CONTRACT, split=args.split, clips=len(records), source_fingerprint=source.fingerprint,
                        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                        decoder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        identity['signature'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        build = args.out / 'build.json'
        if build.exists() and json.loads(build.read_text()) != identity:
            raise ValueError('Native sensor cache identity changed; choose a new output')
        atomic_json(build, identity)
        if (args.out / 'cache.json').exists():
            assert json.loads((args.out / 'cache.json').read_text())['signature'] == identity['signature']
            print('Native sensor cache already complete', flush=True)
            return
        def build_one(index):
            path = args.out / f'{index:05d}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    assert saved['imu_offsets'].shape == (6,)
                    assert saved['imu_values'].shape == (len(saved['imu_times']), 6)
                    assert saved['radar_points'].shape[-1] == 6
                return index
            value = decode_native(source, records[index])
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **value)
            temporary.replace(path)
            return index
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for count, _ in enumerate(pool.map(build_one, range(len(records))), 1):
                if count % 200 == 0 or count == len(records):
                    atomic_json(args.out / 'progress.json', dict(completed=count, total=len(records)))
                    print(f'Native sensors {count}/{len(records)}', flush=True)
        atomic_json(args.out / 'cache.json', dict(**identity, records=records))


if __name__ == '__main__':
    main()
