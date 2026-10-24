# Training and evaluation

Run commands from the repository root after installing the environment and making a clean Git commit. Generated recipes, raw data, caches and runs are ignored by Git. The IMU stage records the current clean commit and the source population; joint training verifies that provenance before transferring an encoder.

## 1. Prepare authorized data

Obtain the data through the organizers; this repository does not contain or grant access to it. The original training archive was supplied as nine split ZIP volumes. If using those volumes, merge them with Info-ZIP rather than concatenating bytes:

```bash
cd /path/to/Training/data
zip -s 0 HAR.zip --out HAR_full.zip
```

Verify downloaded files against the provider's checksums before merging. Preserve the `HAR/data/<modality>/<action>/<user>/<trial>/...` structure. A directory containing an extracted `HAR/` is also supported.

```bash
python scripts/prepare_manifest.py \
  --source /path/to/HAR_full.zip --out data/train_clips.csv
python scripts/fetch_pretrained.py
```

The manifest derives labels from official action directory names and assigns the recorded subject folds. It must contain 3,036 clips, with held populations of 1,011 / 1,012 / 1,013. It does not use model outputs or test data to create the split. Directory/ZIP enumeration order can differ from the historical source; fresh retraining is therefore not claimed to recreate the original model bytes.

## 2. Compute person crops and cache data once

```bash
python -m cuhkx.roi --source /path/to/HAR_full.zip \
  --manifest data/train_clips.csv --split train \
  --detector pretrained/yolo11n.pt --out cache/train_roi.json --device 0

python -m cuhkx.cache --source /path/to/HAR_full.zip \
  --manifest data/train_clips.csv --split train --out cache/train \
  --frames 32 --size 224 --steps 64 --radar-points 32 --workers 8 \
  --roi-manifest cache/train_roi.json
```

The cache contains memory-mapped arrays, availability and validity masks, and a preprocessing/signature record. Epochs reuse the decoded data and do not repeat detection or ZIP decoding. The configured frame count is 16 per model forward, even though the preprocessing cache contains 32 frames.

## 3. Subject-disjoint validation

Run the following sequence separately for folds 1, 2 and 3. `fold=1` is shown. The fold assignment is in `configs/subject_folds.json`.

```bash
python scripts/prepare_run.py imu --fold 1 --out runs/fold1/imu_config.json
CUDA_VISIBLE_DEVICES=0 python -m cuhkx.sensor_train \
  --config runs/fold1/imu_config.json --cache cache/train \
  --fold 1 --out runs/fold1/imu

python scripts/prepare_run.py joint --fold 1 --cache cache/train \
  --imu-run runs/fold1/imu --out runs/fold1/joint_config.json
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m cuhkx.train --config runs/fold1/joint_config.json \
  --cache cache/train --fold 1 --out runs/fold1/joint

python -m cuhkx.predict export --checkpoint runs/fold1/joint/best.pt \
  --out models/fold1.pt --storage int8
```

The standalone IMU stage uses the public LIMU-BERT-X initializer and only source users for that fold. It drops unavailable IMU clips. The joint stage transfers only the matching IMU encoder, GRU and projection; the standalone activity classifier is discarded. Source commit, public-weight hash, source-checkpoint hash, cache identity and user/index populations are checked. Do not substitute weights trained using any of the outer fold's validation users.

The three visual streams share the same 12 pretrained Transformer blocks. Their input/output adapters, skeleton/radar encoders and task heads are initialized as in the selected recipe. The six modality encoders, task heads, gate and small cross-attention module are then optimized jointly. YOLO is fixed preprocessing. The masked-feature decoder is trained jointly but excluded from the inference export.

CV trains for up to 50 epochs and retains `best.pt` from held-out single-view Top-1, plus `last.pt` for resume. That selection uses development data; it is not a separate blind test. For a new model, evaluate its own exported single/dual-view predictions on held-out users before drawing conclusions. The original competition scores in README are recorded historical results, not results of these fresh runs.

## 4. Full-data fitting

`fold=0` has no validation population. It uses a distinct IMU initializer fitted on all available training IMU clips, then a fixed-length joint fit on all 3,036 training clips.

```bash
python scripts/prepare_run.py imu --fold 0 --out runs/full/imu_config.json
CUDA_VISIBLE_DEVICES=0 python -m cuhkx.sensor_train \
  --config runs/full/imu_config.json --cache cache/train --fold 0 --out runs/full/imu

python scripts/prepare_run.py joint --fold 0 --cache cache/train \
  --imu-run runs/full/imu --out runs/full/joint_config.json
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m cuhkx.train --config runs/full/joint_config.json \
  --cache cache/train --fold 0 --out runs/full/joint

python -m cuhkx.predict export --checkpoint runs/full/joint/best.pt \
  --out models/retrained.pt --storage int8
```

The selected model's full-data recipe used 43 standalone IMU epochs and 20 joint epochs, each retaining its original 50-epoch learning-rate schedule. Joint AdamW settings are learning rate 0.0006, visual ratio 0.2, full IMU encoder ratio 0.1, visual layer decay 0.85, weight decay 0.05, five warmup epochs, smoothing 0.1, auxiliary branch loss 0.8, modality dropout 0.15, contrastive weight 0.05 and masked-prediction weight 0.1. The 20-epoch stop was chosen from the original development folds before full fit. In a new full-data run, `best.pt` denotes the fixed final weights, not a test-selected checkpoint.

The source-only IMU transfer does not permit using an all-source initializer in CV. No test labels, label corrections, test pseudo-labels, LLM-generated activity labels or test-time training are needed.

## 5. Resume and monitor

Repeat the same command with `--resume` and the same output, cache, recipes and GPU count. Resume reads `last.pt`, restoring the optimizer, learning-rate progress, RNG states and training-only modules. `best.pt` is a compact selection checkpoint; use `last.pt` for continuation. `--stop-after-epoch N` pauses after saving an epoch boundary. Remove a `STOP_AFTER_EPOCH` marker before resuming if you created one.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m cuhkx.train --config runs/fold1/joint_config.json \
  --cache cache/train --fold 1 --out runs/fold1/joint --resume

tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

The original run used four GPUs with batch eight per GPU and accumulation two (effective batch 64). A smaller GPU count changes effective batch and the rank-local contrastive pool unless the recipe is explicitly adapted. Such a change is an experiment, not a bitwise reproduction claim.

## 6. Exported research models

The export routine embeds YOLO11n, checks the preprocessing identity and enforces the decimal 100 MB limit. It reconstructs the saved tensors to check their shapes and finiteness. Supply a newly exported checkpoint explicitly to inference:

```bash
CUDA_VISIBLE_DEVICES=0 bash inference.sh /path/to/Testing outputs/retrained.csv dual \
  --checkpoint models/retrained.pt
```

No original submitted answer CSV is needed or shipped. Training code is provided for research reproduction; training data and historical optimizer checkpoints are not part of the public code archive.
