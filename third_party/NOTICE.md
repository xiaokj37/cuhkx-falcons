# Model sources and adaptations

## CUHK-X and MotionBERT

The official small-model source is [CUHK-X Zenodo record 19144427](https://zenodo.org/records/19144427). Its archive SHA-256 and per-file hashes are retained in `sources.json`; its MIT license is in `licenses/CUHK-X-LICENSE.txt`.

The skeleton DSTformer and DropPath utility originate in MotionBERT, under the retained Apache-2.0 license in `licenses/`. The selected adapter uses width 64, depth one and two heads, root-relative normalization, 64 chronological frames and masked pooling. It does not load MotionBERT pretrained weights. Invalid frames are zeroed before attention; this is not an attention padding mask inside DSTformer. The DropPath utility retains Copyright 2020 Ross Wightman and its original attribution.

The radar adapter extracts TNet and RadarPointNet without the upstream script's import-time training effects. It uses chronological sampling, padded-point masks and a two-layer bidirectional LSTM with `batch_first=True`. The historical optional IMU CNN/Transformer implementation is retained but is not selected by the published LIMU-BERT-X recipe.

## VideoMAEv2 public adapter

`src/cuhkx/vendor/videomae.py` is adapted from the author's MIT implementation:

- Repository: https://github.com/OpenGVLab/VideoMAEv2
- Revision: `29eab1e8a588d1b3ec0cdec7b03a86cca491b74b`
- File: `models/modeling_finetune.py`
- Source SHA-256: `d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb`
- Copyright: 2023 OpenGVLab; full MIT text in `VIDEOMAEV2_LICENSE.txt`.

The upstream source credits BEiT, timm, DINO and DeiT; their source links remain in the code header. Our adaptations remove timm registry imports and unsupported large factories, use the retained DropPath helper, add SDPA, register fixed positions as a nonpersistent buffer, and use non-reentrant checkpointing only while training with gradients. The competition adapter's `fc_drop_rate` and `use_checkpoint` API, state keys, initialization RNG sequence and mean-pooling operations are retained.

The frozen competition adapter was derived from the earlier CC-BY-NC VideoMAE implementation. This public release instead starts from the separately MIT-published VideoMAEv2 author source. It does not change the frozen submitted checkpoint or relabel the old source. Bounded synthetic CPU compatibility checks and their limits are documented in `docs/RELEASE_VALIDATION.md`.

The selected K710 ViT-B student weights are supplied by OpenGVLab/VideoMAE2, revision `706cc172d65ebd4dedbee3f9c0183a93df9fa125`. The model card declares Apache-2.0. Complete 710-class weights load strictly before removing the original head. Thermal, depth and IR share the twelve Transformer blocks, while patch embeddings, output norms and 256-D projections are separate. IR sums RGB input channels and preserves the patch bias. The giant teacher is not loaded or trained by this solution.

## LIMU-BERT-X

The encoder and masked reconstruction model come from [WANDS-HKUST/LIMU-BERT_Experience](https://github.com/WANDS-HKUST/LIMU-BERT_Experience), revision `bdfc3d0f734befaeb9375f5f7b8a7342f0ddfdcb`, under the MIT license in `LIMU_BERT_X_LICENSE.txt`. Imports and two original shape helpers are combined into a self-contained module. Original shared-layer recurrence and normalization are preserved. Our adapter supplies sensor units, windows, missing-device handling, bidirectional GRU and pooling. Public weights load strictly before reconstruction-only heads are removed.

## YOLO11n and other dependencies

YOLO11n is the COCO detector from [Ultralytics assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0), SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`. Runtime is pinned to Ultralytics 8.3.199. Per-camera probe detections produce an expanded union crop, with a full-frame fallback. Detector weights are included in the separately retained checkpoint. Ultralytics retains its own AGPL-3.0 terms.

The optional X-Fi radar Point Transformer originates in [NTUMARS/X-Fi](https://github.com/NTUMARS/X-Fi), revision `9ba74cc98beb2d75a8eadbe74b1b151dc3d39614`, under Apache-2.0. Redundant CUDA allocator flushes and an overwritten random seed were removed. It is not selected by the published model. TorchVision video/ResNet implementations and timm helpers retain their upstream notices in `dependencies/`.

Masked gates, residual cross-attention, auxiliary/contrastive/masked-feature losses, temporal-view inference and INT8 storage serialization are Team Falcons contributions. INT8 storage restores floating-point tensors and does not imply INT8 arithmetic or checkpoint-sized runtime memory. The root `NOTICE` covers original contributions under Apache-2.0.
