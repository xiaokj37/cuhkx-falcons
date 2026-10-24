"""Small residual cross-modal interaction and training-only masked prediction.

Only visible modality features enter attention. The existing late classifier
remains the main path; a zero-initialized head learns a residual correction.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBottleneckFusion(nn.Module):
    def __init__(self, modalities, dim, hidden=128, latent_tokens=4, heads=4, dropout=.1):
        super().__init__()
        if (any(type(v) is not int or v < 1 for v in (modalities, dim, hidden, latent_tokens, heads))
                or hidden % heads or not 0 <= dropout < 1):
            raise ValueError("Invalid residual bottleneck dimensions or dropout")
        self.modalities, self.dim, self.hidden = modalities, dim, hidden
        self.input_norm = nn.LayerNorm(dim)
        self.input_projection = nn.Linear(dim, hidden)
        self.modality_embedding = nn.Parameter(torch.zeros(1, modalities, hidden))
        self.latents = nn.Parameter(torch.empty(1, latent_tokens, hidden))
        nn.init.normal_(self.latents, std=.02)
        self.query_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(hidden * 2, hidden), nn.Dropout(dropout))
        self.output_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 40)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def encode(self, features, visible):
        if features.ndim != 3 or tuple(features.shape[1:]) != (self.modalities, self.dim):
            raise ValueError("Expected B,M,D modality features")
        if visible.shape != features.shape[:2] or not ((visible == 0) | (visible == 1)).all():
            raise ValueError("Visibility must be a binary B,M mask")
        valid = visible.bool()
        if not valid.any(1).all():
            raise ValueError("At least one modality must remain visible")
        # Sanitize before normalization/projection, including NaNs in hidden input.
        clean = features.masked_fill(~valid[..., None], 0)
        tokens = self.input_projection(self.input_norm(clean)) + self.modality_embedding
        z = self.latents.expand(len(features), -1, -1)
        update, _ = self.attention(self.query_norm(z), tokens, tokens,
                                   key_padding_mask=~valid, need_weights=False)
        z = z + update
        return z + self.ff(self.ff_norm(z))

    def forward(self, features, visible):
        return self.head(self.output_norm(self.encode(features, visible).mean(1)))


def mask_one_visible(visible):
    """Hide one currently visible modality when another can remain as context."""
    if visible.ndim != 2 or not ((visible == 0) | (visible == 1)).all():
        raise ValueError("Visibility must be a binary B,M mask")
    valid = visible.bool()
    if not valid.any(1).all():
        raise ValueError("Natural/dropout masks must retain an input")
    eligible = valid.sum(1) >= 2
    draws = torch.rand(visible.shape, device=visible.device).masked_fill(~valid, float("inf"))
    hidden = torch.zeros_like(valid)
    hidden.scatter_(1, draws.argmin(1, keepdim=True), eligible[:, None])
    return valid & ~hidden, hidden


PREDICTION_STATS = ("coefficient", "raw_loss", "weighted_loss", "eligible_fraction",
                    "target_norm", "target_coordinate_std")


class MaskedModalityPrediction(nn.Module):
    """Training-only decoder; no target features or labels enter its queries/KV."""
    def __init__(self, modalities, dim, hidden=128, heads=4, weight=.1,
                 start_epoch=5., ramp_epochs=10.):
        super().__init__()
        if (not all(math.isfinite(v) for v in (weight, start_epoch, ramp_epochs))
                or weight <= 0 or start_epoch < 0 or ramp_epochs <= 0):
            raise ValueError("Use a finite positive prediction weight/ramp and nonnegative start")
        self.weight, self.start, self.ramp = weight, start_epoch, ramp_epochs
        self.queries = nn.Parameter(torch.empty(1, modalities, hidden))
        nn.init.normal_(self.queries, std=.02)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.heads = nn.ModuleList(nn.Linear(hidden, dim) for _ in range(modalities))

    def forward(self, context, targets, hidden, epoch_position):
        if (targets.ndim != 3 or hidden.shape != targets.shape[:2]
                or hidden.shape[1] != len(self.heads) or hidden.sum(1).max() > 1):
            raise ValueError("Predict at most one hidden modality per source clip")
        if not math.isfinite(epoch_position):
            raise ValueError("Epoch position must be finite")
        queries = self.queries.expand(len(context), -1, -1)
        decoded, _ = self.attention(queries, context, context, need_weights=False)
        decoded = self.norm(queries + decoded)
        predicted = torch.stack([head(decoded[:, i]) for i, head in enumerate(self.heads)], 1)
        # Ignore non-target values before computing a loss; naturally missing
        # targets may contain arbitrary placeholders and never create supervision.
        target = targets.detach().float().masked_fill(~hidden.bool()[..., None], 0)
        errors = 1 - F.cosine_similarity(predicted.float(), target, dim=-1, eps=1e-6)
        per_clip = (errors * hidden).sum(1)
        raw = per_clip.mean()  # All source clips: zero for an ineligible row/rank.
        coefficient = self.weight * min(max((epoch_position - self.start) / self.ramp, 0.), 1.)
        loss = coefficient * raw
        norms = (target.norm(dim=-1) * hidden).sum(1).mean()
        spread = (target.std(dim=-1, unbiased=False) * hidden).sum(1).mean()
        stats = torch.stack((raw.new_tensor(coefficient), raw.detach(), loss.detach(),
                             hidden.any(1).float().mean(), norms.detach(), spread.detach()))
        return loss, stats
