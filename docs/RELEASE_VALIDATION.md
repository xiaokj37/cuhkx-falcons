# Release validation

The original frozen verification package passed raw-input, network-disabled inference for both selected modes, reproducing the original 405-row CSV bytes. That historical result is distinct from the public-source portability checks below.

## Public snapshot and limits

`SOURCE_MAP.json` records 37 retained original Python files: 34 are byte-identical. The public entry wrapper and extracted temporal helper differ for portability. The visual adapter is rebuilt from the official MIT VideoMAEv2 author source, preserving the competition API and SDPA operations. Two unused historical ImageBind modules are excluded. Fusion, losses, joint training, LIMU-BERT-X, skeleton, radar and storage serialization remain unchanged.

On 24 October 2026, the source scan and dependency-free portability suite passed: 17 tests with no skips. These cover help/config handling without ML imports, single/dual selection without personal metadata, checkpoint integrity, synthetic input populations, subject folds, source-only IMU handoff and inference-mode dispatch. They do not run a real classifier or measure accuracy. The prior snapshot also passed the same 17 standard-library tests on Linux/Python 3.10. The suite can be run locally with the commands in the README.

The new visual adapter additionally passed a bounded synthetic CPU check with PyTorch 2.8.0. A two-block, width-48, three-head model, batch two, four frames and 32x32 pixels matched the frozen adapter exactly for initialization, all 32 state keys, fixed positions, strict state loading, outputs and gradients. Evaluation, ordinary training and activation-checkpointed training each had maximum output and gradient errors of zero, with stochastic masks controlled by identical RNG states. See `visual_compatibility.json`. This is a component compatibility check on a tiny architecture, not a full-size checkpoint, GPU or challenge-dataset rerun.

Full-dataset raw-input reproduction applies to the original frozen verification package. The public-source checks above establish portability and component compatibility; they are not a separate full-dataset evaluation of the public snapshot. Both inference modes use the unchanged competition checkpoint.
