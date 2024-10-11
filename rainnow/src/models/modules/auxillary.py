"""Code taken from: https://github.com/Rose-STL-Lab/dyffusion/blob/main/src/models/modules/misc.py."""

import math
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import nn

from rainnow.src.utilities.utils import default


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization.
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def Upsample(dim, dim_out=None, activation=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(default(dim_out, dim)),
        activation if activation is not None else nn.Identity(),
    )


def Downsample(dim, dim_out=None, activation=None):
    return nn.Sequential(
        nn.Conv2d(dim, default(dim_out, dim), kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(num_features=default(dim_out, dim)),
        activation if activation is not None else nn.Identity(),
    )


class LayerNorm(nn.Module):
    def __init__(self, dim):
        """initialisation."""
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm=LayerNorm):
        super().__init__()
        self.fn = fn
        self.norm = norm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def get_time_embedder(
    time_dim: int,
    dim: int,
    learned_sinusoidal_cond: bool = False,
    learned_sinusoidal_dim: int = 16,
):
    if learned_sinusoidal_cond:
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1
    else:
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

    time_emb_mlp = nn.Sequential(
        sinu_pos_emb,
        nn.Linear(fourier_dim, time_dim),
        nn.GELU(),
        nn.Linear(time_dim, time_dim),
    )
    return time_emb_mlp
