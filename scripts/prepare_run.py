"""Create portable recipes while retaining source-only IMU leakage checks."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_IMU_SHA = '65697e7017020f36429a221382253f1fc370fdfd493cfb71667f95846c18516c'


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def imu_recipe(fold):
    config = json.loads((ROOT / 'configs/imu_source.json').read_text())
    config.update(full_source_fit=fold == 0, epochs=43 if fold == 0 else 50)
    config['pretrained'] = str(ROOT / config['pretrained'])
    return config


def joint_recipe(fold, cache, imu_run):
    meta = json.loads((Path(cache) / 'cache.json').read_text())
    if meta['split'] != 'train' or len(meta['records']) != 3036:
        raise ValueError('Use the original training population, never test data')
    source = json.loads((Path(imu_run) / 'run.json').read_text())
    checkpoint = Path(imu_run).resolve() / 'best.pt'
    if not (Path(imu_run) / 'complete.json').is_file() or not checkpoint.is_file():
        raise ValueError('IMU source run must be complete')
    if source['fold'] != fold or source['cache_signature'] != meta['signature']:
        raise ValueError('IMU source fold/cache differs from the joint run')
    if source.get('synthetic_smoke') or source.get('uses_test_data') is not False:
        raise ValueError('Only genuine source-training weights may transfer')
    if source.get('git_dirty') is not False or not source.get('git_commit'):
        raise ValueError('Commit the public source before fitting the IMU encoder')
    train_users = sorted({r['subject'] for r in meta['records'] if fold == 0 or r['fold'] != fold})
    val_users = sorted({r['subject'] for r in meta['records'] if fold and r['fold'] == fold})
    if source['train_subjects'] != train_users or source['validation_subjects'] != val_users:
        raise ValueError('Source weights contain the wrong training/validation users')
    if source['config'].get('pretrained_sha256') != PUBLIC_IMU_SHA:
        raise ValueError('IMU public initialization differs')
    if bool(source.get('full_source_fit')) != (fold == 0):
        raise ValueError('Full-source IMU weights cannot initialize a validation fold')
    config = json.loads((ROOT / 'configs' / ('train_full.json' if fold == 0 else 'train_cv.json')).read_text())
    for item in [config] + list(config['visual_configs'].values()):
        if item.get('pretrained'):
            item['pretrained'] = str(ROOT / item['pretrained'])
    config['detector'] = str(ROOT / config['detector'])
    config['source_sensor_initialization'] = dict(
        modality='imu', encoder='limu_bert', source_git_commit=source['git_commit'],
        public_pretrained_sha256=PUBLIC_IMU_SHA, cache_signature=meta['signature'],
        folds={str(fold): dict(checkpoint=str(checkpoint), sha256=file_hash(checkpoint))})
    if fold == 0:
        config['source_sensor_initialization']['full_source_fit'] = True
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['imu', 'joint'])
    parser.add_argument('--fold', required=True, type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--imu-run', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Configuration exists; preserve it for resume')
    if args.stage == 'joint' and (args.cache is None or args.imu_run is None):
        parser.error('joint needs --cache and --imu-run')
    config = imu_recipe(args.fold) if args.stage == 'imu' else joint_recipe(args.fold, args.cache, args.imu_run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps(dict(stage=args.stage, fold=args.fold, config=str(args.out))))


if __name__ == '__main__':
    main()
