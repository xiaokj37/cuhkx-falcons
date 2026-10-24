"""Train a sensor independently on its available clips from original user folds."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import read_meta
from .official_models import OfficialIMUEncoder, OfficialRadarEncoder
from .imu_models import LIMUBertIMUEncoder
from .radar_models import XFiRadarEncoder
from .tracking import RunTracker, atomic_json, capture_rng, restore_rng
from .train import save_checkpoint, split_indices


class SensorDataset(Dataset):
    """Only sensor arrays are opened; keep them in RAM across all epochs."""
    def __init__(self, cache, modality, indices, with_orientation=False):
        cache = Path(cache)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.values = np.load(cache / f'{modality}.npy')[self.indices].copy()
        self.valid = np.load(cache / f'{modality}_valid.npy')[self.indices].copy()
        meta = read_meta(cache)
        if with_orientation:
            from .orientation_cache import ORIENTATION_CONTRACT
            if modality != 'imu' or meta['preprocessing'].get('imu_orientation') != ORIENTATION_CONTRACT:
                raise ValueError('Explicit orientation input contract required')
            quaternion = np.load(cache / 'imu_quaternion.npy')[self.indices]
            pose_valid = np.load(cache / 'imu_quaternion_valid.npy')[self.indices]
            self.values = np.concatenate((self.values, quaternion), -1)
            self.valid = np.stack((np.broadcast_to(self.valid[:, None], pose_valid.shape), pose_valid), -1)
        self.labels = np.array([meta['records'][int(i)]['label'] for i in self.indices], dtype=np.int64)
        if not np.isfinite(self.values).all() or not np.isfinite(self.valid).all():
            raise ValueError('Nonfinite sensor inputs')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return (torch.from_numpy(self.values[index]), torch.from_numpy(self.valid[index]),
                int(self.labels[index]), int(self.indices[index]))


def build_sensor(config):
    name = config['encoder']
    if name == 'official_imu':
        encoder = OfficialIMUEncoder()
    elif name == 'limu_bert':
        encoder = LIMUBertIMUEncoder(pretrained=config['pretrained'], pretrained_sha256=config['pretrained_sha256'])
    elif name == 'official_radar':
        encoder = OfficialRadarEncoder()
    elif name == 'xfi_radar':
        encoder = XFiRadarEncoder(pretrained=config['pretrained'], pretrained_sha256=config['pretrained_sha256'])
    elif name in ['pose_imu', 'limu_pose']:
        from .orientation_models import PoseIMUEncoder, LIMUPoseIMUEncoder
        encoder = (PoseIMUEncoder() if name == 'pose_imu' else
                   LIMUPoseIMUEncoder(pretrained=config['pretrained'], pretrained_sha256=config['pretrained_sha256']))
    else:
        raise ValueError('Unknown standalone sensor encoder')
    return SensorClassifier(encoder)


class SensorClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(.3), nn.Linear(256, 40))

    def forward(self, values, valid):
        return self.head(self.encoder(values, valid))


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    parts = {key: [] for key in ['indices', 'labels', 'logits']}
    loss_sum = 0.
    for values, valid, labels, indices in loader:
        logits = model(values.to(device), valid.to(device)).float()
        loss_sum += float(nn.functional.cross_entropy(logits, labels.to(device), reduction='sum'))
        parts['indices'].append(indices.numpy())
        parts['labels'].append(labels.numpy())
        parts['logits'].append(logits.cpu().numpy())
    data = {key: np.concatenate(value) for key, value in parts.items()}
    if not np.isfinite(data['logits']).all():
        raise ValueError('Nonfinite standalone validation logits')
    prediction = data['logits'].argmax(-1)
    confusion = np.bincount(data['labels'] * 40 + prediction, minlength=1600).reshape(40, 40)
    denominator = confusion.sum(0) + confusion.sum(1)
    macro = np.divide(2 * np.diag(confusion), denominator, out=np.zeros(40, dtype=float), where=denominator > 0).mean()
    return dict(accuracy=float((prediction == data['labels']).mean()), macro_f1=float(macro),
                loss=loss_sum / len(prediction), clips=len(prediction)), data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--fold', required=True, type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    full_fit = args.fold == 0
    if full_fit and (config.get('full_source_fit') is not True or config.get('encoder') != 'limu_bert'):
        raise ValueError('Full source training requires the explicit fixed-epoch LIMU recipe')
    if not full_fit and config.get('full_source_fit'):
        raise ValueError('A full-source recipe cannot be used for a validation fold')
    if not 0 < config['epochs'] <= config.get('schedule_epochs', config['epochs']):
        raise ValueError('Fixed training length must not exceed the original learning-rate schedule')
    torch.set_num_threads(config.get('cpu_threads', 2))
    device = torch.device(args.device)
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = config.get('allow_tf32', True)
    meta = read_meta(args.cache)
    if not ((not args.smoke and meta['split'] == 'train' and len(meta['records']) == 3036)
            or (args.smoke and meta['split'] == 'synthetic')):
        raise ValueError('Standalone diagnostics use only the original training cache')
    modality = config['modality']
    if modality not in ['imu', 'radar']:
        raise ValueError('Select an inertial or radar branch')
    if (modality == 'imu') != (config['encoder'] in ['official_imu', 'limu_bert', 'pose_imu', 'limu_pose']):
        raise ValueError('Modality and encoder disagree')
    if config['encoder'] == 'xfi_radar':
        from .radar_cache import RADAR_CONTRACT
        if meta['preprocessing'].get('radar') != RADAR_CONTRACT:
            raise ValueError('X-Fi expects the existing verified five-channel cache')
    elif meta['preprocessing'].get('radar') is not None and modality == 'radar':
        raise ValueError('The official radar baseline expects four channels')
    train, validation, train_users, validation_users = split_indices(meta, args.fold, args.smoke)
    available = np.load(args.cache / 'availability.npy', mmap_mode='r')[:, 4 if modality == 'imu' else 5] > 0
    train = np.asarray(train, dtype=np.int64)[available[train]]
    validation = np.asarray(validation, dtype=np.int64)[available[validation]]
    assert not set(train_users) & set(validation_users) and len(train) and (full_fit or len(validation))
    provenance = dict(purpose=('Fixed-epoch full-source sensor initialization; no validation accuracy'
                              if full_fit else 'Standalone sensor diagnosis; not a full multimodal competition result'),
                      git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                      config=config, fold=args.fold, train_subjects=train_users, validation_subjects=validation_users,
                      train_clips=len(train), validation_clips=len(validation), cache_signature=meta['signature'],
                      train_indices_sha256=hashlib.sha256(train.astype('<i8').tobytes()).hexdigest(),
                      validation_indices_sha256=hashlib.sha256(validation.astype('<i8').tobytes()).hexdigest(),
                      uses_test_data=False, metadata_used_as_features=False, synthetic_smoke=args.smoke)
    if full_fit:
        provenance['full_source_fit'] = True
        provenance['checkpoint_selection'] = 'fixed_epoch_no_heldout_validation'
    if provenance['git_dirty'] and not args.smoke:
        raise ValueError('Commit sensor code before formal training')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'complete.json').exists():
        raise ValueError('This sensor run is complete; inspect it instead of overwriting')
    if (args.out / 'run.json').exists() and json.loads((args.out / 'run.json').read_text()) != provenance:
        raise ValueError('Sensor configuration, source, cache, or split changed')
    if (args.out / 'last.pt').exists() and not args.resume:
        raise ValueError('Use --resume to preserve the existing sensor checkpoint')
    atomic_json(args.out / 'run.json', provenance)
    model = build_sensor(dict(config, pretrained=None) if args.resume else config).to(device)
    groups = {}
    for name, parameter in model.named_parameters():
        ratio = config.get('backbone_lr_ratio', 1.) if name.startswith('encoder.backbone.') else 1.
        decay = 0. if parameter.ndim <= 1 or 'pos_embed' in name else config['weight_decay']
        groups.setdefault((ratio, decay), []).append(parameter)
    optimizer = torch.optim.AdamW([dict(params=value, lr=config['lr'] * ratio, weight_decay=decay)
                                    for (ratio, decay), value in groups.items()])
    with_orientation = config['encoder'] in ['pose_imu', 'limu_pose']
    training = DataLoader(SensorDataset(args.cache, modality, train, with_orientation), batch_size=config['batch_size'], shuffle=True,
                          num_workers=0, pin_memory=device.type == 'cuda')
    testing = DataLoader(SensorDataset(args.cache, modality, validation, with_orientation), batch_size=config['batch_size'],
                         num_workers=0, pin_memory=device.type == 'cuda')
    start, global_step, best = 0, 0, -1.
    base_lrs = [group['lr'] for group in optimizer.param_groups]
    total_steps = len(training) * config.get('schedule_epochs', config['epochs'])
    warmup = len(training) * config['warmup_epochs']
    if args.resume:
        saved = torch.load(args.out / 'last.pt', map_location=device, weights_only=True)
        if saved['provenance'] != provenance:
            raise ValueError('Standalone resume provenance differs')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        start, global_step, best = saved['epoch'] + 1, saved['global_step'], saved['best_accuracy']
        assert saved['scheduler'] == dict(total_steps=total_steps, warmup_steps=warmup, base_lrs=base_lrs)
        restore_rng(saved['rng_state'], device)
    tracker = RunTracker(args.out, config, provenance, start_epoch=start, global_step=global_step,
                         steps_per_epoch=len(training), resume=args.resume)
    try:
        for epoch in range(start, config['epochs']):
            started = time.monotonic()
            model.train()
            loss_sum = correct = examples = 0
            for step, (values, valid, labels, _) in enumerate(training, 1):
                factor = ((global_step + 1) / warmup if global_step < warmup else
                          .5 * (1 + math.cos(math.pi * (global_step - warmup) / max(1, total_steps - warmup))))
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group['lr'] = base_lr * factor
                optimizer.zero_grad(set_to_none=True)
                labels = labels.to(device)
                logits = model(values.to(device), valid.to(device))
                loss = nn.functional.cross_entropy(logits, labels, label_smoothing=config['label_smoothing'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite sensor training loss')
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                global_step += 1
                loss_sum += float(loss) * len(labels)
                hits = int((logits.argmax(-1) == labels).sum())
                correct += hits
                examples += len(labels)
                if step % 20 == 0 or step == len(training):
                    tracker.step(epoch + 1, step, len(training), global_step, float(loss), hits / len(labels),
                                 [group['lr'] for group in optimizer.param_groups], float(norm))
            metrics, predictions = (evaluate(model, testing, device) if not full_fit else
                (dict(accuracy=None, macro_f1=None, loss=None, clips=0), None))
            improved = epoch + 1 == config['epochs'] if full_fit else metrics['accuracy'] > best
            best = -1. if full_fit else max(best, metrics['accuracy'])
            checkpoint = dict(model={key: value.detach().cpu() for key, value in model.state_dict().items()},
                              config=config, provenance=provenance, epoch=epoch, best_accuracy=best,
                              global_step=global_step, metrics=metrics)
            if improved:
                save_checkpoint(checkpoint, args.out / 'best.pt')
                if predictions is not None:
                    np.savez_compressed(args.out / 'validation_predictions.npz', **predictions)
            checkpoint.update(optimizer=optimizer.state_dict(), rng_state=capture_rng(device),
                              scheduler=dict(total_steps=total_steps, warmup_steps=warmup, base_lrs=base_lrs))
            save_checkpoint(checkpoint, args.out / 'last.pt')
            row = dict(epoch=epoch + 1, global_step=global_step, train_loss=loss_sum / examples,
                       train_accuracy=correct / examples, best_accuracy=best, **metrics,
                       wall_seconds=time.monotonic() - started,
                       peak_gpu_gib=torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0.)
            with (args.out / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            tracker.epoch(row, global_step)
            print(json.dumps(row), flush=True)
            if epoch + 1 < config['epochs'] and ((args.out / 'STOP_AFTER_EPOCH').exists() or
                                                args.stop_after_epoch is not None and epoch + 1 >= args.stop_after_epoch):
                atomic_json(args.out / 'paused.json', row)
                return
        atomic_json(args.out / 'complete.json', dict(best_accuracy=best, epochs=config['epochs'], global_step=global_step))
        (args.out / 'paused.json').unlink(missing_ok=True)
    finally:
        tracker.close()


if __name__ == '__main__':
    main()
