"""Basic UNet implementation. 

Code taken from: https://github.com/Rose-STL-Lab/dyffusion/blob/main/src/models/unet_simple.py."""

from typing import Callable, Optional

import torch
import torch.nn as nn
from einops import rearrange
from hydra.utils import instantiate
from torch import Tensor, nn

from rainnow.src.models._base_model import BaseModel
from rainnow.src.models.modules.auxillary import get_time_embedder
from rainnow.src.utilities.utils import exists

RELU_LEAK = 0.2


class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_chans,
        dim_out,
        time_emb_dim=None,
        transposed=False,
        bn=True,
        relu=True,
        size=4,
        pad=1,
        dropout=0.0,
    ):
        """initialisation."""
        super().__init__()
        batch_norm = bn
        relu_leak = None if relu else RELU_LEAK
        kern_size = size
        self.time_mlp = (
            # SilU for the time embedding activation. This is currently fixed.
            nn.Sequential(nn.SiLU(), nn.Linear(in_features=time_emb_dim, out_features=dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        ops = []
        if not transposed:
            # regular conv.
            ops.append(
                torch.nn.Conv2d(
                    in_channels=in_chans,
                    out_channels=dim_out,
                    kernel_size=kern_size,
                    stride=2,
                    padding=pad,
                    bias=True,
                )
            )
        else:
            # upsample and trans conv.
            ops.append(torch.nn.Upsample(scale_factor=2, mode="bilinear"))
            ops.append(
                torch.nn.Conv2d(
                    in_channels=in_chans,
                    out_channels=dim_out,
                    kernel_size=(kern_size - 1),
                    stride=1,
                    padding=pad,
                    bias=True,
                )
            )

        # option for batchnorm. Defaults to Groupnorm.
        if batch_norm:
            ops.append(torch.nn.BatchNorm2d(num_features=dim_out))
        else:
            ops.append(nn.GroupNorm(num_groups=8, num_channels=dim_out))

        self.ops = torch.nn.Sequential(*ops)

        # defaults to ReLU() else will use LeakyReLU()
        if relu_leak is None or relu_leak == 0:
            self.act = torch.nn.ReLU()
        else:
            self.act = torch.nn.LeakyReLU(negative_slope=relu_leak)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, time_emb=None):
        x = self.ops(x)
        if exists(self.time_mlp):
            assert exists(time_emb), "Time embedding must be provided if time_mlp is not None"
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        x = self.dropout(x)

        return x


class UNet(BaseModel):
    def __init__(
        self,
        dim: int,
        with_time_emb: bool = False,
        outer_sample_mode: str = "bilinear",
        upsample_dims: tuple = (256, 256),
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        input_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        output_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ):
        """initialisation."""
        super().__init__(**kwargs)
        self.outer_sample_mode = outer_sample_mode
        if upsample_dims is None:
            self.upsampler = nn.Identity()
        else:
            self.upsampler = torch.nn.Upsample(size=tuple(upsample_dims), mode=self.outer_sample_mode)
        in_channels = self.num_input_channels + self.num_conditional_channels

        if with_time_emb:
            # time embeddings.
            self.time_dim = dim * 2
            self.time_emb_mlp = get_time_embedder(
                time_dim=self.time_dim, dim=dim, learned_sinusoidal_cond=False
            )
        else:
            self.time_dim = None
            self.time_emb_mlp = None

        # activations.
        self.input_activation = instantiate(
            None if not input_activation.get("_target_") else input_activation
        )
        self.output_activation = instantiate(
            None if not output_activation.get("_target_") else output_activation
        )

        # input conv layer.
        self.init_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=dim),
            self.input_activation if self.input_activation is not None else nn.Identity(),
        )

        self.dropout_input = nn.Dropout(p=input_dropout)
        block_kwargs = dict(time_emb_dim=self.time_dim, dropout=dropout)

        # encoding layers.
        # fmt: off
        self.downs = torch.nn.ModuleList(
            [
                UNetBlock(dim, dim * 2, transposed=False, bn=True, relu=False, **block_kwargs),
                UNetBlock(dim * 2, dim * 2, transposed=False, bn=True, relu=False, **block_kwargs),
                UNetBlock(dim * 2, dim * 4, transposed=False, bn=True, relu=False, **block_kwargs),
                UNetBlock(dim * 4, dim * 8, transposed=False, bn=True, relu=False, size=4, **block_kwargs),
                UNetBlock(dim * 8, dim * 8, transposed=False, bn=True, relu=False, size=2, pad=0, **block_kwargs),
                UNetBlock(dim * 8, dim * 8, transposed=False, bn=False, relu=False, size=2, pad=0, **block_kwargs),
            ]
        )
        # decoder layers.
        self.ups = torch.nn.ModuleList(
            [
                UNetBlock(dim * 8, dim * 8, transposed=True, bn=True, relu=True, size=2, pad=0, **block_kwargs),
                UNetBlock(dim * 8 * 2, dim * 8, transposed=True, bn=True, relu=True, size=2, pad=0, **block_kwargs),
                UNetBlock(dim * 8 * 2, dim * 4, transposed=True, bn=True, relu=True, **block_kwargs),
                UNetBlock(dim * 4 * 2, dim * 2, transposed=True, bn=True, relu=True, **block_kwargs),
                UNetBlock(dim * 2 * 2, dim * 2, transposed=True, bn=True, relu=True, **block_kwargs),
                UNetBlock(dim * 2 * 2, dim, transposed=True, bn=True, relu=True, **block_kwargs),
            ]
        )
        # fmt: on

        # final conv layer.
        self.final_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=dim,
                out_channels=self.num_output_channels,
                kernel_size=4,
                stride=2,  # this needs to be 2 to avoid collapsing to 0.
                padding=1,
                bias=True,
            ),
            self.output_activation if self.output_activation is not None else nn.Identity(),
        )

        # initialise weights.
        self.apply(self.__init_weights)

    @staticmethod
    def __init_weights(
        module,
    ):  # taken from: https://github.com/pytorch/examples/blob/main/dcgan/main.py.
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            module.weight.data.normal_(0.0, 0.02)
        elif isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def _apply_ops(self, x: Tensor, time: Tensor = None):
        # apply downsampling and upsampling operations.
        skip_connections = []
        x = self.init_conv(x)
        x = self.dropout_input(x)
        for op in self.downs:
            x = op(x, time)
            skip_connections.append(x)
        x = skip_connections.pop()

        for op in self.ups:
            x = op(x, time)
            if skip_connections:
                x = torch.cat([x, skip_connections.pop()], dim=1)
        x = self.final_conv(x)
        return x

    def forward(self, inputs, time=None, condition=None, return_time_emb: bool = False, **kwargs):
        # preprocess inputs for shape.
        if self.num_conditional_channels > 0:
            x = torch.cat([inputs, condition], dim=1)
        else:
            x = inputs
            assert condition is None

        t = self.time_emb_mlp(time) if exists(self.time_emb_mlp) else None

        # apply operations.
        orig_x_shape = x.shape[-2:]
        x = self.upsampler(x)
        x = self._apply_ops(x, t)

        # interpolate to get original image shape.
        x = torch.nn.functional.interpolate(x, size=orig_x_shape, mode=self.outer_sample_mode)

        if return_time_emb:
            return x, t

        return x
