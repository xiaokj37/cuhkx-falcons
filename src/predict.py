"""Raw-input single/dual-view inference; no fitting or submitted-answer lookup.

The default checkpoint is pinned to the frozen competition weights. An explicit
checkpoint can be used for separately exported research models.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SHA = 'aafbc72913a5664701dec3815d6126ac1cf0846cb8250114567c9de4547ed55e'


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def event(stage, **details):
    print(json.dumps(dict(stage=stage, **details)), flush=True)


def locate_input(directory):
    """Accept the published Testing root (or a track root containing Testing)."""
    base = Path(directory).expanduser().resolve(strict=True)
    if not base.is_dir():
        raise ValueError('The first argument must be the official Testing directory')
    if not (base / 'test.csv').is_file() and (base / 'Testing/test.csv').is_file():
        base = base / 'Testing'
    manifest = base / 'test.csv'
    if not manifest.is_file():
        raise ValueError(f'Missing label-free manifest: {manifest}')
    # No sample_submission.csv, original submitted answers or training labels are read.
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if 'path' not in (reader.fieldnames or []):
            raise ValueError('test.csv must have a path column')
        paths = [row['path'] for row in reader]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Empty or duplicate paths in test.csv')
    for path in paths:
        parts = Path(path.rstrip('/')).parts
        if len(parts) != 2 or parts[0] != 'small_model_track_test' or '..' in parts:
            raise ValueError(f'Expected published relative path small_model_track_test/<clip>/: {path}')
    archive = base / 'data/small_model_track_test.zip'
    if archive.is_file():
        source = archive
    elif (base / 'data/small_model_track_test').is_dir():
        source = base / 'data'
    elif (base / 'small_model_track_test').is_dir():
        source = base
    else:
        raise ValueError('Missing data/small_model_track_test.zip or extracted small_model_track_test directory')
    return source, manifest


def select_config(mode):
    aliases = {'dual': 'dual', 'single': 'single', 'v61': 'dual', 'v47': 'single',
               '56156565': 'dual', '56125089': 'single'}
    mode = aliases.get(mode or 'dual')
    if mode is None:
        raise ValueError('Choose dual or single temporal-view inference')
    path = ROOT / ('configs/inference_' + mode + '.json')
    config = json.loads(path.read_text())
    if config['views'] not in (['odd', 'even'], ['odd']):
        raise ValueError('Unsupported temporal views')
    if config['batch_size'] != 8 or config['virtual_reference_ranks'] != 4:
        raise ValueError('Frozen inference requires batch eight and four logical shards')
    return config


def resolve_checkpoint(args, chosen):
    weights_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else ROOT / chosen['checkpoint']
    actual_sha = sha256(weights_path)
    expected_sha = args.expected_sha256 or (WEIGHT_SHA if args.checkpoint is None else None)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError('Checkpoint SHA-256 mismatch')
    if weights_path.stat().st_size >= 100_000_000:
        raise ValueError('Exported checkpoint must be smaller than 100,000,000 bytes')
    return weights_path, actual_sha


def run(args, work):
    # Keep argument/help/configuration handling usable without the ML stack.
    import torch
    with torch.inference_mode():
        return _run(args, work)


def _run(args, work):
    import cv2
    import numpy as np
    import torch
    from cuhkx.cache import Source, records_from_manifest
    from cuhkx.data import ClipDataset, prepare_vision, read_meta, to_device, validate_preprocessing
    from cuhkx.models import build_model
    from cuhkx.roi import build_rois
    from cuhkx.temporal_views import fixed_times, temporal_batch, probabilities
    from cuhkx.train import make_loader
    from cuhkx.weights import unpack_state

    started = time.monotonic()
    chosen = select_config(args.submission_id)
    source_path, manifest = locate_input(args.data_dir)
    output_path = Path(args.output_csv).expanduser().resolve()
    if output_path == manifest:
        raise ValueError('Output must not overwrite the input manifest')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path, actual_sha = resolve_checkpoint(args, chosen)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU with native BF16 support is required (Ampere or newer)')
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    device = torch.device('cuda', 0)
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True, mmap=True)
    if checkpoint['format'] != 'cuhkx-single-checkpoint-v2-int8':
        raise ValueError('Unexpected checkpoint format')
    config = checkpoint['config']
    source = Source(source_path)
    records = records_from_manifest(manifest, 'test')
    if any(not any(source.files(prefix) for prefix in item['prefixes'].values()) for item in records):
        raise ValueError('One or more manifest clips have no files at their published paths')
    event('person_roi', clips=len(records), work_directory=str(work))
    roi_file = work / 'person_roi.json'
    build_rois(source, records, manifest, checkpoint['detector'], roi_file, device='0',
               settings=checkpoint['provenance']['preprocessing']['settings'])
    roi_seconds = time.monotonic() - started
    del source
    gc.collect()
    torch.cuda.empty_cache()
    cache = work / 'decoded'
    event('decode', clips=len(records))
    decode_started = time.monotonic()
    subprocess.run([sys.executable, '-m', 'cuhkx.cache', '--source', str(source_path),
                    '--manifest', str(manifest), '--split', 'test', '--out', str(cache),
                    '--frames', '32', '--size', '224', '--steps', '64', '--radar-points', '32',
                    '--workers', '8', '--roi-manifest', str(roi_file)], check=True)
    decode_seconds = time.monotonic() - decode_started
    meta = read_meta(cache)
    validate_preprocessing(meta, config, checkpoint['provenance'])
    if [item['path'] for item in meta['records']] != [item['path'] for item in records]:
        raise ValueError('Preprocessing changed manifest row order')
    model = build_model(config, pretrained=False)
    model.load_state_dict(unpack_state(checkpoint['model']), strict=True)
    model = model.to(device).eval()
    del checkpoint
    indices = list(range(len(records)))
    view_logits = {view: np.empty((len(records), 40), np.float32) for view in chosen['views']}
    seen = np.zeros(len(records), dtype=np.int32)
    times = fixed_times()
    inference_started = time.monotonic()
    for rank in range(4):
        subset = indices[rank::4]
        if not subset:
            continue
        dataset = ClipDataset(cache, subset, frames=32, training=False)
        loader = make_loader(dataset, 8, 2)
        for batch in loader:
            batch = to_device(batch, device)
            batch_indices = batch['index'].cpu().numpy()
            seen[batch_indices] += 1
            for view in chosen['views']:
                current = temporal_batch(batch, times[view])
                vision = prepare_vision(current, config['image_size'], False, config.get('visual_configs'))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model(current, vision)
                view_logits[view][batch_indices] = prediction['logits'].float().cpu().numpy()
                del prediction, vision, current
            event('inference', completed=int(seen.sum()), total=len(records), views=chosen['views'])
        del loader, dataset
    if not np.all(seen == 1) or any(not np.isfinite(value).all() for value in view_logits.values()):
        raise ValueError('Incomplete, duplicate or nonfinite model output')
    ps = [probabilities(view_logits[view]) for view in chosen['views']]
    probability = ps[0] if len(ps) == 1 else (ps[0] + ps[1]) * .5
    labels = probability.argmax(-1)
    # csv.writer's CRLF convention preserves the exact submitted CSV bytes.
    temporary = output_path.with_name(output_path.name + '.writing')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('path', 'prediction'))
        writer.writerows((record['path'], int(label)) for record, label in zip(records, labels))
    temporary.replace(output_path)
    report = dict(configuration=chosen['name'], clips=len(records), views=chosen['views'],
                  checkpoint_sha256=actual_sha, output_sha256=sha256(output_path),
                  roi_seconds=roi_seconds, decode_seconds=decode_seconds,
                  inference_seconds=time.monotonic() - inference_started,
                  elapsed_seconds=time.monotonic() - started,
                  peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                  gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                  cuda_version=torch.version.cuda, network_required=False,
                  reads_original_predictions=False)
    output_path.with_name(output_path.name + '.run.json').write_text(json.dumps(report, indent=2) + '\n')
    event('completed', **report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir')
    parser.add_argument('output_csv')
    parser.add_argument('submission_id', nargs='?', default='dual', help='dual or single')
    parser.add_argument('--checkpoint', help='Explicit research checkpoint; otherwise use frozen models/model.pt')
    parser.add_argument('--expected-sha256', help='Optional integrity pin for an explicit checkpoint')
    args = parser.parse_args()
    # Scratch is rebuilt from the supplied raw input. No precomputed challenge
    # cache or submitted CSV is consulted. It is deleted after successful/failed exit.
    with tempfile.TemporaryDirectory(prefix='cuhkx-reproduce-', dir=os.environ.get('CUHKX_SCRATCH_ROOT')) as temporary:
        run(args, Path(temporary))


if __name__ == '__main__':
    main()
