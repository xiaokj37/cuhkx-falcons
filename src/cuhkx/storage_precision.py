"""Change selected storage precision using the original unquantized weights."""
import torch

from .weights import _tensor_key, unpack_state


def preserve_full_precision(packed, source, prefixes):
    """Keep unselected encoded objects intact; reject aliases across the boundary."""
    if set(packed) != set(source):
        raise ValueError('Packed and source state keys differ')
    if not prefixes or any(not p or not any(n.startswith(p) for n in source) for p in prefixes):
        raise ValueError('Each selection prefix must match a nonempty group')
    selected = {n for n in source if n.startswith(tuple(prefixes))}
    groups = {}
    for name, value in source.items():
        groups.setdefault(_tensor_key(value), []).append(name)
    result = dict(packed)
    for names in groups.values():
        chosen = [n for n in names if n in selected]
        if not chosen:
            continue
        if len(chosen) != len(names):
            raise ValueError('A shared tensor crosses the selected precision boundary')
        if len({id(packed[n]) for n in names}) != 1:
            raise ValueError('Packed and source tensor aliases disagree')
        value = source[names[0]]
        decoded = unpack_state({names[0]: packed[names[0]]})[names[0]]
        if decoded.shape != value.shape or decoded.is_floating_point() != value.is_floating_point():
            raise ValueError('Packed and source tensor shapes or types differ')
        if value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError('Selected source weights must be finite FP32 tensors')
        replacement = value.detach().cpu().clone()
        for name in names:
            result[name] = replacement
    return result


def assert_encoded_equal(before, after):
    """Compare serialized tensor contents, including dtype, without dequantizing."""
    if isinstance(before, torch.Tensor):
        if not isinstance(after, torch.Tensor) or before.dtype != after.dtype or before.shape != after.shape:
            raise ValueError('Encoded tensor shape or dtype changed')
        if not torch.equal(before, after):
            raise ValueError('Encoded tensor contents changed')
    elif isinstance(before, dict):
        if not isinstance(after, dict) or before.keys() != after.keys():
            raise ValueError('Encoded mapping changed')
        for name in before:
            assert_encoded_equal(before[name], after[name])
    elif isinstance(before, (list, tuple)):
        if type(before) is not type(after) or len(before) != len(after):
            raise ValueError('Encoded sequence changed')
        for a, b in zip(before, after):
            assert_encoded_equal(a, b)
    elif type(before) is not type(after) or before != after:
        raise ValueError('Encoded metadata changed')
