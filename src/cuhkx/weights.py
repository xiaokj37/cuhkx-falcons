"""Per-output-channel compact storage; inference reconstructs ordinary tensors."""
import math
import torch


def _tensor_key(value):
    return (value.device, value.dtype, value.untyped_storage().data_ptr(),
            value.storage_offset(), tuple(value.shape), tuple(value.stride()))


def state_to_cpu(state):
    """Copy each shared tensor once, preserving checkpoint storage aliases."""
    result, copied = {}, {}
    for name, value in state.items():
        key = _tensor_key(value)
        if key not in copied:
            copied[key] = value.detach().cpu()
        result[name] = copied[key]
    return result


def pack_int5(quantized):
    """Eight offset-coded signed values occupy five little-endian bytes."""
    if quantized.dtype != torch.int8:
        raise ValueError("INT5 packing requires signed integer values")
    values = quantized.detach().cpu().flatten().to(torch.int64)
    if values.numel() and ((values < -16).any() or (values > 15).any()):
        raise ValueError("Values exceed signed INT5 range")
    values = values + 16
    padding = (-len(values)) % 8
    if padding:
        values = torch.cat((values, torch.full((padding,), 16, dtype=torch.int64)))
    values = values.reshape(-1, 8)
    words = torch.zeros(len(values), dtype=torch.int64)
    for index in range(8):
        words |= values[:, index] << (5 * index)
    return ((words[:, None] >> (torch.arange(5) * 8)) & 255).to(torch.uint8).flatten()


def unpack_int5(packed, count):
    if (not isinstance(count, int) or count < 0 or packed.dtype != torch.uint8 or packed.ndim != 1
            or packed.numel() != ((count + 7) // 8) * 5):
        raise ValueError("Invalid packed INT5 length or dtype")
    data = packed.cpu().reshape(-1, 5).to(torch.int64)
    words = torch.zeros(len(data), dtype=torch.int64)
    for index in range(5):
        words |= data[:, index] << (8 * index)
    values = ((words[:, None] >> (torch.arange(8) * 5)) & 31).flatten()[:count]
    return (values - 16).to(torch.int8)


def pack_state(state, storage="fp16"):
    if storage not in ("fp16", "int8", "int5"):
        raise ValueError(f"Unknown storage format: {storage}")
    result, packed = {}, {}
    for name, value in state.items():
        key = _tensor_key(value)
        if key in packed:
            result[name] = packed[key]
            continue
        value = value.detach().cpu()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
        # Small task heads, gates and narrow convolutions are cheap to keep in
        # FP16; protect their calibration when using the more aggressive format.
        minimum = 65536 if storage == "int5" else 256
        if storage in ("int8", "int5") and value.is_floating_point() and value.ndim >= 2 and value.numel() >= minimum:
            limit = 127 if storage == "int8" else 15
            dims = tuple(range(1, value.ndim))
            scale = value.float().abs().amax(dims, keepdim=True).div(limit).clamp_min(1e-12)
            quantized = (value.float() / scale).round().clamp(-limit,limit).to(torch.int8)
            result[name] = ({"quantized":quantized, "scale":scale} if storage == "int8" else
                            {"packed":pack_int5(quantized), "shape":list(value.shape), "scale":scale, "bits":5})
        else:
            result[name] = value.half() if value.is_floating_point() else value
        packed[key] = result[name]
    return result


def unpack_state(state):
    result, decoded = {}, {}
    for name, value in state.items():
        identity = id(value)
        if identity in decoded:
            result[name] = decoded[identity]
            continue
        if isinstance(value, dict):
            if "packed" in value:
                if (set(value) != {"packed", "shape", "scale", "bits"} or value["bits"] != 5
                        or not isinstance(value["shape"], list) or len(value["shape"]) < 2
                        or any(type(n) is not int or n <= 0 for n in value["shape"])):
                    raise ValueError(f"Invalid INT5 tensor metadata: {name}")
                quantized = unpack_int5(value["packed"], math.prod(value["shape"])).reshape(value["shape"])
                scale = value["scale"]
            else:
                if set(value) != {"quantized", "scale"} or value["quantized"].dtype != torch.int8:
                    raise ValueError(f"Invalid quantized tensor: {name}")
                quantized, scale = value["quantized"], value["scale"]
            expected = (quantized.shape[0],) + (1,) * (quantized.ndim-1)
            if scale.shape != expected or not torch.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError(f"Invalid quantization scale: {name}")
            result[name] = quantized.float() * scale.float()
        else:
            result[name] = value
        decoded[identity] = result[name]
    return result
