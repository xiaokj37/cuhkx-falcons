# Model card

- **Task:** six-modality human activity recognition; one primary activity label among 40 classes per clip.
- **Intended use:** non-commercial academic research under CUHK-X and upstream component terms.
- **Developer:** Team Falcons, Zhejiang University.
- **Inputs:** thermal, depth colour and IR video; 17-joint skeleton sequences; five-device IMU; radar point sequences, with missing-modality and validity masks.
- **Outputs:** class probabilities internally; `path,prediction` CSV externally, with labels 0–39.
- **Training population:** 3,036 labelled clips from 18 training users. The full-source IMU stage uses 2,863 clips with available IMU. No test labels or test pseudo-labels.
- **Local evaluation:** three subject-disjoint folds, 12 training and six validation users each. Dual-view Top-1: 2,427 / 3,036, 79.9407%. Public Kaggle: 0.87064. The scores describe the retained competition checkpoint, not arbitrary retraining.
- **Architecture:** public distilled VideoMAEv2 ViT-B student shared across three video modalities; public LIMU-BERT-X IMU encoder; random-initialized small skeleton/radar branches; masked late fusion plus 4×128-D latent cross-attention.
- **Storage:** 96,857,179-byte complete checkpoint, including the detector. INT8 weight storage with BF16/FP32 execution; parameter count of the classifier is 90,690,160, with detector parameters additional.
- **Temporal inference:** 32 cached frames, 16 per view, two fixed interleaved views and equal probability averaging. This is neither a 32-frame Transformer forward nor an ensemble of separately trained checkpoints.
- **Hardware evidence:** original isolated, network-disabled inference on RTX 4090, PyTorch 2.8.0/CUDA 12.8, Ultralytics 8.3.199. Measured peak reserved GPU memory 1.3301 GiB; 300.04 seconds for the original 405-clip dual-view raw-input verification. Cross-hardware bitwise equality is not established.
- **Limitations:** local cross-user performance is below 90%; similar co-occurring actions and brief discriminative actions can remain confused. Fixed temporal sampling can miss evidence. Missing modalities, crop failures, different sensors/users/environments and deployment hardware can change results. The masked feature loss encourages sensor relationships but does not provide annotated primary-activity localization.
- **Model access:** request the trained checkpoint from [xiaokj@zju.edu.cn](mailto:xiaokj@zju.edu.cn); its size and SHA-256 are listed in `models/README.md`.

Upstream model weights, vendored implementations, and dataset access each have their own terms. See the license inventory rather than assuming one license covers all components.
