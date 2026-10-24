"""Fixed camera views; sensor streams are identical across views."""
import numpy as np
import torch
from .cache import sample_indices

def require(condition, message):
    if not condition:
        raise ValueError(message)

def fixed_times(cache_frames=32, model_frames=16):
    require((cache_frames, model_frames) == (32, 16), 'V61 is fixed to the original 32-to-16 frame contract')
    odd, even = np.arange(1, 32, 2), np.arange(0, 32, 2)
    np.testing.assert_array_equal(odd, sample_indices(32, 16))
    return {'odd': odd, 'even': even}

def temporal_batch(batch, times):
    """Only cameras change; sensors, validity and natural absence are identical."""
    times = torch.as_tensor(times, device=batch['vision'].device, dtype=torch.long)
    return dict(batch, vision=batch['vision'].index_select(1, times))

def probabilities(logits):
    x = np.asarray(logits, dtype=np.float64)
    require(x.shape[-1] == 40 and np.isfinite(x).all(), 'Invalid class logits')
    p = np.exp(x - x.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)

def aggregate(odd_logits, even_logits):
    require(odd_logits.shape == even_logits.shape, 'View dimensions differ')
    return (probabilities(odd_logits) + probabilities(even_logits)) * .5
