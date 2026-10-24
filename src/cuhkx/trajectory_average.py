"""Fixed two-checkpoint parameter averaging; preserve the selected model's buffers.

This is a trajectory diagnostic, not a reproduction of SWA or SWAD.
"""
import torch

from .weights import _tensor_key


def parameter_midpoint(model, selected, final):
    reference = model.state_dict()
    if set(selected) != set(reference) or set(final) != set(reference):
        raise ValueError('Checkpoint keys do not match the same model architecture')
    for name, expected in reference.items():
        a, b = selected[name], final[name]
        if a.shape != expected.shape or b.shape != expected.shape or a.dtype != b.dtype:
            raise ValueError('Checkpoint shape/dtype mismatch: ' + name)
        if a.is_floating_point() and (not torch.isfinite(a).all() or not torch.isfinite(b).all()):
            raise ValueError('Nonfinite checkpoint tensor: ' + name)
    owners = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        owners.setdefault(id(parameter), []).append(name)
    result = {}
    for names in owners.values():
        first = names[0]
        if not selected[first].is_floating_point():
            raise ValueError('Only floating-point parameters are supported')
        for name in names[1:]:
            if not torch.equal(selected[first], selected[name]) or not torch.equal(final[first], final[name]):
                raise ValueError('Shared parameter aliases disagree in checkpoint: ' + name)
        value = (selected[first].float() * .5 + final[first].float() * .5).to(selected[first].dtype)
        for name in names:
            result[name] = value
    buffers = {}
    for name in reference:
        if name not in result:
            key = _tensor_key(selected[name])
            if key not in buffers:
                buffers[key] = selected[name].clone()
            result[name] = buffers[key]
    return result
