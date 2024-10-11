"""UNet Resnet (with attention mechanisms) implementation.

Code adapted from: https://github.com/Rose-STL-Lab/dyffusion/blob/main/src/models/unet.py."""

from functools import partial
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from torch import nn

from rainnow.src.models._base_model import BaseModel
from rainnow.src.models.modules.attention import Attention, LinearAttention
from rainnow.src.models.modules.auxillary import (
    Downsample,
    LayerNorm,
    PreNorm,
    Residual,
    Upsample,
    WeightStandardizedConv2d,
    get_time_embedder,
)
from rainnow.src.utilities.utils import default, exists


class Block(nn.Module):
    """Conv --> Normalize (Group or Batch) --> Activation --> Dropout."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        activation: nn.Module,
        norm_groups: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """initialisation."""
        super().__init__()
        if norm_groups is None:
            # normal Conv2d w/ Batchnorm.
            self.proj = nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=3, padding=1)
            self.norm = nn.BatchNorm2d(num_features=dim_out)
        else:
            # WeightStandardizedConv2d + GroupNorm.
            self.proj = WeightStandardizedConv2d(
                in_channels=dim, out_channels=dim_out, kernel_size=3, padding=1
            )
            self.norm = nn.GroupNorm(num_groups=norm_groups, num_channels=dim_out)
        self.act = activation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResnetBlock(nn.Module):
    """(Conv --> Normalize --> Activation --> Dropout) x 2 with optional time embedding and residual connection."""

    def __init__(
        self,
        dim,
        dim_out,
        *,
        norm_groups: Optional[int] = None,
        time_emb_dim=None,
        double_conv_layer: bool = True,
        dropout1: float = 0.0,
        dropout2: float = 0.0,
        activation: Optional[nn.Module] = None,
    ):
        """initialisation."""
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(in_features=time_emb_dim, out_features=dim_out * 2))
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(
            dim=dim, dim_out=dim_out, norm_groups=norm_groups, activation=activation, dropout=dropout1
        )
        self.block2 = (
            Block(
                dim=dim_out,
                dim_out=dim_out,
                norm_groups=norm_groups,
                activation=activation,
                dropout=dropout2,
            )
            if double_conv_layer
            else nn.Identity()
        )
        self.residual_conv = (
            nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=1)
            if dim != dim_out
            else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.residual_conv(x)


class UNet(BaseModel):
    def __init__(
        self,
        dim,
        init_dim=None,
        dim_mults=(1, 2, 4, 8),
        num_conditions: int = 0,
        with_time_emb: bool = False,
        block_dropout: float = 0.0,  # for second block in resnet block
        block_dropout1: float = 0.0,  # for first block in resnet block
        block_num_groups: Optional[int] = None,
        attn_dropout: float = 0.0,
        input_dropout: float = 0.0,
        apply_batchnorm: Optional[bool] = True,
        apply_layernorm_after_attn: Optional[bool] = True,
        double_conv_layer: bool = True,
        learned_variance: bool = False,
        learned_sinusoidal_cond: bool = False,
        learned_sinusoidal_dim: int = 16,
        outer_sample_mode: str = None,  # bilinear or nearest
        upsample_dims: tuple = None,  # (256, 256) or (128, 128)
        keep_spatial_dims: bool = False,
        init_kernel_size: int = 7,
        init_padding: int = 3,
        init_stride: int = 1,
        final_kernel_size: Optional[int] = None,
        layer_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        block_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        output_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ):
        """initialisation."""
        super().__init__(**kwargs)
        # determine dimensions.
        input_channels = self.num_input_channels + self.num_conditional_channels
        output_channels = self.num_output_channels or input_channels
        self.save_hyperparameters()

        if num_conditions >= 1:
            assert (
                self.num_conditional_channels > 0
            ), f"num_conditions is {num_conditions} but num_conditional_channels is {self.num_conditional_channels}"

        # input channel dims (and for the final resnet block).
        self.init_dim = default(init_dim, dim)

        assert (upsample_dims is None and outer_sample_mode is None) or (
            upsample_dims is not None and outer_sample_mode is not None
        ), "upsample_dims and outer_sample_mode must be both None or both not None"
        if outer_sample_mode is not None:
            self.upsampler = torch.nn.Upsample(size=tuple(upsample_dims), mode=self.outer_sample_mode)
        else:
            self.upsampler = None

        # activations.
        self.layer_activation = instantiate(
            None if not layer_activation.get("_target_") else layer_activation
        )
        self.block_activation = instantiate(
            None if not block_activation.get("_target_") else block_activation
        )
        self.output_activation = instantiate(
            None if not output_activation.get("_target_") else output_activation
        )

        # handle norms + final conv kernel size.
        # defaults to 3 and None (batch norm) for kernel size and norm groups respectively.
        self.final_kernel_size = 4 if final_kernel_size is None else final_kernel_size
        self.block_norm_groups = block_num_groups
        self.apply_batchnorm = False if apply_batchnorm is None else apply_batchnorm
        self.attention_norm = LayerNorm if apply_layernorm_after_attn else nn.BatchNorm2d

        # input conv layer.
        self.init_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=self.init_dim,
                kernel_size=init_kernel_size,
                stride=init_stride,
                padding=init_padding,
            ),
            nn.BatchNorm2d(num_features=self.init_dim) if self.apply_batchnorm else nn.Identity(),
            self.layer_activation if self.layer_activation is not None else nn.Identity(),
        )

        # dropouts.
        self.dropout_input = nn.Dropout(p=input_dropout)
        self.dropout_input_for_residual = nn.Dropout(p=input_dropout)

        # time embeddings.
        if with_time_emb:
            self.time_dim = dim * 2
            self.time_emb_mlp = get_time_embedder(
                time_dim=self.time_dim,
                dim=dim,
                learned_sinusoidal_cond=learned_sinusoidal_cond,
                learned_sinusoidal_dim=learned_sinusoidal_dim,
            )
        else:
            self.time_dim = None
            self.time_emb_mlp = None

        dims = [self.init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # wrapper for ResnetBlock(), only need to pass dim_in, dim_out.
        block_klass = partial(
            ResnetBlock,
            norm_groups=self.block_norm_groups,
            dropout2=block_dropout,
            dropout1=block_dropout1,
            double_conv_layer=double_conv_layer,
            time_emb_dim=self.time_dim,
            activation=self.block_activation,
        )

        # layers.
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # encoding layers.
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            do_downsample = not is_last and not keep_spatial_dims
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim=dim_in, dim_out=dim_in),
                        block_klass(dim=dim_in, dim_out=dim_in),
                        Residual(
                            PreNorm(
                                dim=dim_in,
                                fn=LinearAttention(dim=dim_in, rescale="qkv", dropout=attn_dropout),
                                norm=self.attention_norm,
                            )
                        ),
                        (
                            Downsample(
                                dim=dim_in,
                                dim_out=dim_out,
                                activation=(
                                    self.layer_activation
                                    if self.layer_activation is not None
                                    else nn.Identity()
                                ),
                            )
                            if do_downsample
                            else nn.Sequential(
                                nn.Conv2d(
                                    in_channels=dim_in,
                                    out_channels=dim_out,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                ),
                                (
                                    nn.BatchNorm2d(num_features=dim_out)
                                    if self.apply_batchnorm
                                    else nn.Identity()
                                ),
                                (
                                    self.layer_activation
                                    if self.layer_activation is not None
                                    else nn.Identity()
                                ),
                            )
                        ),
                    ]
                )
            )

        # middle blocks.
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(dim=mid_dim, dim_out=mid_dim)
        self.mid_attn = Residual(
            PreNorm(dim=mid_dim, fn=Attention(mid_dim, dropout=attn_dropout), norm=self.attention_norm)
        )
        self.mid_block2 = block_klass(dim=mid_dim, dim_out=mid_dim)

        # decoding layers.
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            do_upsample = not is_last and not keep_spatial_dims
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim=dim_out + dim_in, dim_out=dim_out),
                        block_klass(dim=dim_out + dim_in, dim_out=dim_out),
                        Residual(
                            PreNorm(
                                dim=dim_out,
                                fn=LinearAttention(dim=dim_out, rescale="qkv", dropout=attn_dropout),
                                norm=self.attention_norm,
                            )
                        ),
                        (
                            Upsample(
                                dim=dim_out,
                                dim_out=dim_in,
                                activation=(
                                    self.layer_activation
                                    if self.layer_activation is not None
                                    else nn.Identity()
                                ),
                            )
                            if do_upsample
                            else nn.Sequential(
                                nn.Conv2d(
                                    in_channels=dim_out,
                                    out_channels=dim_in,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                ),
                                (
                                    nn.BatchNorm2d(num_features=dim_in)
                                    if self.apply_batchnorm
                                    else nn.Identity()
                                ),
                                (
                                    self.layer_activation
                                    if self.layer_activation is not None
                                    else nn.Identity()
                                ),
                            )
                        ),
                    ]
                )
            )

        default_out_dim = input_channels * (1 if not learned_variance else 2)
        self.out_dim = default(output_channels, default_out_dim)
        # self.final_res_block = block_klass(dim=dim * 2, dim_out=dim)
        self.final_res_block = block_klass(dim=self.init_dim * 2, dim_out=self.init_dim)
        self.final_conv = self.get_head()

        if hasattr(self, "spatial_shape") and self.spatial_shape is not None:
            b, s1, s2 = 1, *self.spatial_shape
            self.example_input_array = [
                torch.rand(b, self.num_input_channels, s1, s2),
                torch.rand(b) if with_time_emb else None,
                (
                    torch.rand(b, self.num_conditional_channels, s1, s2)
                    if self.num_conditional_channels > 0
                    else None
                ),
            ]

        # initialise weights.
        self.apply(self.__init_weights)

    @staticmethod  # taken from: https://github.com/pytorch/examples/blob/main/dcgan/main.py.
    def __init_weights(module):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            module.weight.data.normal_(0.0, 0.02)
        elif isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def get_head(self):
        """Get network 'head' / final convolutional layer."""
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.init_dim,
                out_channels=self.out_dim,
                kernel_size=self.final_kernel_size,  # this can't be 1. Also > 5 leads to mode collapse to 0.
                stride=2,  # this needs to be 2 to avoid collapsing to 0.
                padding=1,
                bias=True,
            ),
            self.output_activation if self.output_activation is not None else nn.Identity(),
        )

    def set_head_to_identity(self):
        self.final_conv = nn.Identity()

    def get_block(self, dim_in, dim_out, norm_groups, dropout: Optional[float] = None):
        return ResnetBlock(
            dim_in,
            dim_out,
            norm_groups,
            dropout1=dropout or self.hparams.block_dropout1,
            dropout2=dropout or self.hparams.block_dropout,
            time_emb_dim=self.time_dim,
        )

    def get_extra_last_block(self, dropout: Optional[float] = None):
        return self.get_block(self.hparams.dim, self.hparams.dim, dropout=dropout)

    def forward(self, x, time=None, condition=None, return_time_emb: bool = False):
        """Forward pass."""
        if self.num_conditional_channels > 0:
            x = torch.cat((condition, x), dim=1)
        else:
            assert condition is None, "condition is not None but num_conditional_channels is 0"
        orig_x_shape = x.shape[-2:]
        x = self.upsampler(x) if exists(self.upsampler) else x
        x = self.init_conv(x)
        r = self.dropout_input_for_residual(x) if self.hparams.input_dropout > 0 else x.clone()
        x = self.dropout_input(x)

        t = self.time_emb_mlp(time) if exists(self.time_emb_mlp) else None
        h = []
        # encoding (downsampling) operations.
        for i, (block1, block2, attn, downsample) in enumerate(self.downs):
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # decoding (upsampling) operations.
        for _, (block1, block2, attn, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        # interpolate to get original image shape.
        x = F.interpolate(x, size=orig_x_shape, mode="bilinear")
        if exists(self.upsampler):
            x = F.interpolate(x, size=orig_x_shape, mode=self.hparams.outer_sample_mode)

        if return_time_emb:
            return x, t

        return x
