# Third-party attribution and licensing

Original Team Falcons contributions are licensed under **Apache-2.0**; see `LICENSE` and `NOTICE`. This grant does not replace upstream licenses or grant rights to the dataset or separately supplied model weights.

| Component | Included code or use | Recorded source and terms |
|---|---|---|
| CUHK-X official small-model code | Small skeleton, radar and optional sensor components | [Zenodo 19144427](https://zenodo.org/records/19144427); code MIT; `third_party/licenses/CUHK-X-LICENSE.txt` and `third_party/sources.json` |
| DSTformer / MotionBERT | `src/cuhkx/vendor/dstformer.py`, drop utility | [Walter0807/MotionBERT](https://github.com/Walter0807/MotionBERT/tree/705d3a95354db8bdb696b3492e47a3b5537174ff); Apache-2.0; no pretrained skeleton weights used |
| VideoMAEv2 implementation | `src/cuhkx/vendor/videomae.py`, adapted factories, SDPA and checkpointing | [OpenGVLab/VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2/tree/29eab1e8a588d1b3ec0cdec7b03a86cca491b74b); MIT, Copyright 2023 OpenGVLab; `third_party/VIDEOMAEV2_LICENSE.txt` |
| VideoMAEv2 public student weights | Training initialization; no weights bundled | [OpenGVLab/VideoMAE2 pinned model release](https://huggingface.co/OpenGVLab/VideoMAE2/tree/706cc172d65ebd4dedbee3f9c0183a93df9fa125); K710 ViT-B distilled student; model card declares Apache-2.0 |
| LIMU-BERT-X | `src/cuhkx/vendor/limu_bert.py`, public initializer | [WANDS-HKUST/LIMU-BERT_Experience](https://github.com/WANDS-HKUST/LIMU-BERT_Experience/tree/bdfc3d0f734befaeb9375f5f7b8a7342f0ddfdcb); MIT, `third_party/LIMU_BERT_X_LICENSE.txt` |
| Ultralytics YOLO11n | External dependency for person detection and detector serialization | [Ultralytics](https://github.com/ultralytics/ultralytics); **AGPL-3.0**, retained at `third_party/dependencies/ultralytics-LICENSE`; not relicensed by this repository |
| X-Fi point transformer | Optional historical radar source, not selected by the published recipe | [NTUMARS/X-Fi](https://github.com/NTUMARS/X-Fi/tree/9ba74cc98beb2d75a8eadbe74b1b151dc3d39614); Apache-2.0, `third_party/X_FI_LICENSE.txt` |
| PyTorch, TorchVision, timm | External dependencies and attributed helpers | Retained notices under `third_party/dependencies/`; upstream terms apply |

The public video adapter is derived from the author's **MIT VideoMAEv2 source** at the pinned revision above. The frozen competition package used an earlier CC-BY-NC VideoMAE adapter. That source has not been relabeled: it is excluded from this public snapshot and replaced with the MIT author implementation, adapted to preserve the selected model's API, state keys, initialization sequence and SDPA operations. Exact public changes are recorded in `docs/SOURCE_MAP.json` and `third_party/NOTICE.md`. Unused historical ImageBind code is also excluded.

Dataset access remains subject to the separate CUHK-X non-commercial research data-use agreement.

Public initializer URLs and hashes are pinned in `scripts/fetch_pretrained.py`. The selected trained checkpoint is separate from the code release. Its identity is recorded in `models/README.md`; this code license does not assert a blanket license for the combined model artifact.
