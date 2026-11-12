# Frozen model artifact

The trained competition model is a single `model.pt`, 96,857,179 bytes. To request the weights, please contact [xiaokj@zju.edu.cn](mailto:xiaokj@zju.edu.cn).

Verify the received file against this checksum:

```text
SHA-256: aafbc72913a5664701dec3815d6126ac1cf0846cb8250114567c9de4547ed55e
```

Save the received file as `models/model.pt`. The default inference command verifies that exact identity.

The artifact includes the six modality encoders, heads, gate, residual cross-attention and YOLO11n. Both reported inference modes use this same file. Public pretraining initializers are not needed at inference and are fetched only for new training.

An explicit `--checkpoint` accepts a separately exported research artifact in the same INT8-storage format. Use `--expected-sha256` to pin its actual hash. A new training run is not expected to match the frozen competition file's hash or scores.

Such a checkpoint must retain the published six-modality input and 16-frame,
224×224 per-view preprocessing contract. The raw-input wrapper is not a generic
entry point for arbitrary frame counts or architectures.
