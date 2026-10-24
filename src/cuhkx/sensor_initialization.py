"""Transfer a source-user sensor encoder without importing its action classifier.

The public backbone and the task-trained temporal layers are both transferred.
Every outer fold must have its own initializer. Validation chooses the source
checkpoint epoch, but no held-out user's examples may train its parameters.
"""
import hashlib
import io
from pathlib import Path
import re

import numpy as np
import torch


def source_sensor_identity(config, meta, availability, fold, smoke=False):
    spec = config.get("source_sensor_initialization")
    if spec is None:
        return None
    if (spec.get("modality"), spec.get("encoder")) != ("imu", "limu_bert"):
        raise ValueError("Only the audited standalone LIMU IMU initializer is supported")
    if ("imu" not in config["modalities"] or config.get("dim", 256) != 256
            or config.get("imu_config", {}).get("backbone") != "limu_bert_x"):
        raise ValueError("Source sensor initialization requires the matching LIMU architecture")
    full_fit = fold == 0 and spec.get("full_source_fit") is True and "0" in spec["folds"]
    if fold not in (1, 2, 3) and not full_fit:
        raise ValueError("Full fit needs a separately trained full-source initializer")
    if fold != 0 and spec.get("full_source_fit"):
        raise ValueError("Full-source weights cannot initialize a validation fold")
    if ((smoke and meta["split"] != "synthetic")
            or (not smoke and (meta["split"] != "train" or len(meta["records"]) != 3036))):
        raise ValueError("Source sensor initialization requires the original training cache")
    if meta["signature"] != spec["cache_signature"]:
        raise ValueError("Source sensor cache signature differs")
    item = spec["folds"][str(fold)]
    for value in (item["sha256"], spec["public_pretrained_sha256"]):
        if re.fullmatch(r"[a-f0-9]{64}", value) is None:
            raise ValueError("Pin source and public pretrained weights with SHA-256")
    if re.fullmatch(r"[a-f0-9]{40}", spec["source_git_commit"]) is None:
        raise ValueError("Pin the source training Git revision")
    if availability.shape != (len(meta["records"]), 6) or not np.isin(availability, [0, 1]).all():
        raise ValueError("Source initialization availability differs from the six-modality cache")
    from .train import split_indices
    train, validation, train_users, validation_users = split_indices(meta, fold, smoke)
    available = availability[:, 4] > 0
    train = np.asarray(train, dtype="<i8")[available[train]]
    validation = np.asarray(validation, dtype="<i8")[available[validation]]
    if not len(train) or not full_fit and not len(validation):
        raise ValueError("Source sensor split has no eligible clips")
    return dict(schema=1, modality="imu", encoder="limu_bert", fold=fold,
                checkpoint=item["checkpoint"], checkpoint_sha256=item["sha256"],
                source_git_commit=spec["source_git_commit"],
                public_pretrained_sha256=spec["public_pretrained_sha256"],
                cache_signature=meta["signature"], train_subjects=train_users,
                validation_subjects=validation_users, train_clips=len(train), validation_clips=len(validation),
                train_indices_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
                validation_indices_sha256=hashlib.sha256(validation.tobytes()).hexdigest(),
                uses_test_data=False, metadata_used_as_features=False, synthetic_smoke=smoke,
                transfer_classifier=False)


def initialize_source_sensor(model, identity):
    if identity is None:
        return None
    payload = Path(identity["checkpoint"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != identity["checkpoint_sha256"]:
        raise ValueError("Source sensor checkpoint SHA-256 mismatch")
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    provenance = saved["provenance"]
    for key in ("fold", "cache_signature", "train_subjects", "validation_subjects", "train_clips",
                "validation_clips", "train_indices_sha256", "validation_indices_sha256", "uses_test_data",
                "metadata_used_as_features", "synthetic_smoke"):
        if provenance.get(key) != identity[key]:
            raise ValueError(f"Source sensor provenance differs in {key}")
    if provenance.get("git_commit") != identity["source_git_commit"] or provenance.get("git_dirty") is not False:
        raise ValueError("Source sensor training code identity differs")
    source_config = saved["config"]
    if (source_config != provenance["config"] or source_config.get("encoder") != identity["encoder"]
            or source_config.get("modality") != identity["modality"]
            or source_config.get("pretrained_sha256") != identity["public_pretrained_sha256"]):
        raise ValueError("Source sensor public pretraining or architecture differs")
    full_fit = identity["fold"] == 0
    if full_fit:
        if (provenance.get("full_source_fit") is not True or source_config.get("full_source_fit") is not True
                or provenance.get("checkpoint_selection") != "fixed_epoch_no_heldout_validation"
                or saved["epoch"] + 1 != source_config["epochs"] or identity["validation_clips"] != 0):
            raise ValueError("Full-source initializer must be the fixed final epoch without held-out validation")
    elif provenance.get("full_source_fit") or source_config.get("full_source_fit"):
        raise ValueError("Full-source weights cannot initialize a validation fold")
    encoder = model.encoders[identity["modality"]]
    from .imu_models import LIMUBertIMUEncoder
    if type(encoder) is not LIMUBertIMUEncoder:
        raise ValueError("Destination must be the plain LIMU encoder")
    reference = encoder.state_dict()
    state = saved["model"]
    expected_keys = {"encoder." + key for key in reference} | {"head.1.weight", "head.1.bias"}
    if set(state) != expected_keys:
        raise ValueError("Source sensor checkpoint has missing or unexpected tensors")
    shapes = {"encoder." + key: value.shape for key, value in reference.items()}
    shapes.update({"head.1.weight": (40, 256), "head.1.bias": (40,)})
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != shapes[key] or not torch.isfinite(tensor).all():
            raise ValueError(f"Source sensor tensor is invalid: {key}")
        if tensor.dtype != torch.float32:
            raise ValueError("Source initialization expects audited FP32 training weights")
    transferred = {key: state["encoder." + key] for key in reference}
    # No operation below touches a branch classifier, fusion gate, or other encoder.
    encoder.load_state_dict(transferred, strict=True)
    from .domain_generalization import state_sha256
    return dict(**identity, loaded_tensors=len(transferred), loaded_parameters=sum(p.numel() for p in encoder.parameters()),
                encoder_state_sha256=state_sha256(transferred), source_best_epoch=saved["epoch"] + 1,
                source_global_step=saved["global_step"],
                checkpoint_selection=("fixed final epoch on all original training users" if full_fit else
                                      "best validation epoch within the same outer fold"))


def restore_source_sensor_audit(previous, identity):
    """Resume only from last.pt; the original initializer need not remain on disk."""
    if identity is None:
        if previous is not None:
            raise ValueError("Resume unexpectedly contains source sensor initialization")
        return None
    if not isinstance(previous, dict) or any(previous.get(key) != value for key, value in identity.items()):
        raise ValueError("Resume source sensor initialization identity differs")
    if (previous.get("loaded_tensors") != 33 or previous.get("loaded_parameters") != 414400
            or re.fullmatch(r"[a-f0-9]{64}", previous.get("encoder_state_sha256", "")) is None):
        raise ValueError("Resume source sensor initialization audit is incomplete")
    return dict(previous)
