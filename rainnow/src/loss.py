"""Auxillary loss module containing loss functions to use during training ML models."""

from functools import wraps
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchvision import models, transforms

from rainnow.src.normalise import PreProcess


def capture_init_kwargs(cls):
    """Decorator to allow easy access to the init kwargs of a class."""
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        self._init_kwargs = kwargs.copy()
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls


def get_pixelwise_loss(name: str, reduction="mean", **kwargs):
    """
    Returns the (pixel-wise distance-based) loss function with the given name.

    Parameters
    ----------
    name : str
        The name of the loss function to return. Supported options include:
        'l1', 'mae', 'mean_absolute_error', 'l2', 'mse', 'mean_squared_error',
        'smoothl1', 'smooth', 'masked_l2_loss', 'ml2', 'masked_l1_loss', 'ml1',
        'masked_huber_loss', 'mh', 'cb_loss', 'cb'.
    reduction : str, optional
        Specifies the reduction to apply to the output. Default is "mean".

    Returns
    -------
    nn.Module
        The requested loss function.

    Raises
    ------
    ValueError
        If an unknown loss function name is provided.
    """
    name = name.lower().strip().replace("-", "_")
    if name in ["l1", "mae", "mean_absolute_error"]:
        loss = nn.L1Loss(reduction=reduction)
    elif name in ["l2", "mse", "mean_squared_error"]:
        loss = nn.MSELoss(reduction=reduction)
    elif name in ["smoothl1", "smooth"]:
        loss = nn.SmoothL1Loss(reduction=reduction)
    elif name in ["masked_l2_loss", "ml2"]:
        loss = MaskedMSELoss(
            masked_value=0.0,
            masking_type="equal",
            mask_weights=[0.1, 1],
            pixelwise_loss="l2",
            reduction=reduction,
        )
    elif name in ["masked_l1_loss", "ml1"]:
        loss = MaskedMSELoss(
            masked_value=0.0,
            masking_type="equal",
            mask_weights=[0.1, 1],
            pixelwise_loss="l1",
            reduction=reduction,
        )
    elif name in ["masked_huber_loss", "mh"]:
        # kwargs need to contain delta.
        loss = MaskedMSELoss(
            masked_value=0.0,
            masking_type="equal",
            mask_weights=[0.1, 1],
            pixelwise_loss="huber",
            reduction=reduction,
            **kwargs,
        )
    elif name in ["cb_loss", "cb"]:
        loss = CBLoss(reduction=reduction, **kwargs)
    else:
        raise ValueError(f"Unknown loss function {name}")
    return loss


@capture_init_kwargs
class MaskedMSELoss(nn.Module):
    """
    Masked MSE loss with weights for certain pixel values to help give influence to other pixels.

    Parameters
    ----------
    masked_value : float, optional
        The value to be masked in the input. Default is 0.0.
    masking_type : str, optional
        The type of masking to apply. Default is "equal".
    mask_weights : list of float, optional
        The weights to apply for different mask values. Default is [0.1, 1].
    pixelwise_loss : str, optional
        The type of pixelwise loss to use. Default is "l2".
    reduction : str, optional
        Specifies the reduction to apply to the output. Default is "mean".
    device : str, optional
        The device to use for computations. Default is "cpu".
    delta : float, optional
        A parameter for certain loss functions. Default is 1.0.
    """

    def __init__(
        self,
        masked_value: float = 0.0,
        masking_type="equal",
        mask_weights=[0.1, 1],
        pixelwise_loss: str = "l2",
        reduction: str = "mean",
        device: str = "cpu",
        delta: float = 1.0,
    ):
        """Initialisation."""
        super().__init__()
        # masking info.
        self.masked_value = masked_value
        self.masking_type = masking_type
        self.mask_weights = mask_weights
        # loss info.
        self.pixelwise_loss = pixelwise_loss
        self.reduction = reduction
        self.delta = delta

        self.device = device

    @staticmethod
    def reduce(loss, reduction):
        # handle reductions.
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            # defaults to reduction=None
            return loss

    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        """Forward pass."""
        if self.masking_type == "equal":
            weights = torch.where(
                target == self.masked_value,
                torch.tensor(self.mask_weights[0]).to(self.device),
                torch.tensor(self.mask_weights[1]).to(self.device),
            )
        else:
            # <= is only option after 0.0.
            weights = torch.where(
                target <= self.masked_value,
                torch.tensor(self.mask_weights[0]).to(self.device),
                torch.tensor(self.mask_weights[1]).to(self.device),
            )

        if self.pixelwise_loss == "l2":
            loss = weights * (predicted - target) ** 2
        elif self.pixelwise_loss == "l1":
            loss = weights * torch.abs(predicted - target)
        elif self.pixelwise_loss == "smoothl1":
            loss = weights * F.smooth_l1_loss(predicted, target, beta=1.0, reduction="none")
        elif self.pixelwise_loss == "huber":
            loss = weights * F.huber_loss(predicted, target, delta=self.delta, reduction="none")

        loss = self.reduce(loss, reduction=self.reduction)
        return loss


# TODO: test the weight matrix creation in this.
@capture_init_kwargs
class CBLoss(nn.Module):
    """
    A combined loss function using Balanced Mean Squared Error (BMSE) and Balanced Mean Absolute Error (BMAE),
    balanced by a tunable parameter, beta.

    The loss is computed as:

    BMSE + beta * BMAE / 2

    Where:

    BMSE = (1/N) * Σ[n=1 to N] Σ[i=1 to 128] Σ[j=1 to 128] w[n,i,j] * (x[n,i,j] - x̂[n,i,j])^2

    BMAE = (1/N) * Σ[n=1 to N] Σ[i=1 to 128] Σ[j=1 to 128] w[n,i,j] * |x[n,i,j] - x̂[n,i,j]|

    Parameters
    ----------
    N : int
        Number of samples.
    w : array_like
        Weight for each pixel, shape (N, 128, 128).
    x : array_like
        True values, shape (N, 128, 128).
    x_hat : array_like
        Predicted values, shape (N, 128, 128).
    beta : float
        Weighting factor for BMAE.
    """

    nodes = [0.5, 2, 6, 10, 18, 30]  # 1 less than weights.
    weights = [1, 2, 5, 10, 20, 30, 50]

    def __init__(
        self,
        beta: float = 0.1,
        reduction: str = "mean",
        data_preprocessor_obj: Optional[PreProcess] = NotImplemented,
    ):
        """Initialisation."""
        super().__init__()
        self.beta = beta  # .1 default is taken from: https://www.sciencedirect.com/science/article/pii/S0098300424000128
        self.reduction = reduction
        self.pprocessor = data_preprocessor_obj
        if self.pprocessor is not None:
            # need to also preprocess the nodes to align with the preprocessed data.
            self.nnodes = self.pprocessor.apply_preprocessing(np.array(self.nodes))
        else:
            self.nnodes = self.nodes

    @staticmethod
    def reduce(loss, reduction):
        # handle reductions.
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            # defaults to reduction=None
            return loss

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # get mse and mae losses.
        l2_loss = (predicted - target) ** 2
        l1_loss = torch.abs(predicted - target)

        # create weight matrix using target.
        assert len(self.nnodes) == len(self.weights) - 1
        W = torch.full_like(target, self.weights[-1])  # initialise with highest weight.
        for i, node in enumerate(self.nnodes[::-1]):  # reverse order.
            W = torch.where(target < node, self.weights[:-1][::-1][i], W)

        # compute the combined loss and reduce it. beta can help balance l1 and l2.
        loss = ((W * l2_loss) + (self.beta * W * l1_loss)) / 2
        loss = self.reduce(loss, reduction=self.reduction)

        return loss


@capture_init_kwargs
class LPIPSLoss(nn.Module):
    """
    Wrapper around torchmetrics.LPIPS to handle single channel images.

    LPIPS weights and biases are frozen by default by torchmetrics.

    Parameters
    ----------
    model_name : str, optional
        The name of the model to use for LPIPS. Default is "alex".
    reduction : str, optional
        Specifies the reduction to apply to the output. Default is "mean".
    normalize : bool, optional
        If True, assumes input is in range [0, 1]. If False, assumes [-1, 1]. Default is False.
    """

    def __init__(self, model_name: str = "alex", reduction: str = "mean", normalize: bool = False):
        """initialisation."""
        super().__init__()
        self.model_name = model_name
        self.reduction = reduction
        self.normalize = normalize  # False assumes [-1, 1], True assumes [0, 1].
        self.required_channels = 3
        self.lpips = LPIPS(net_type=self.model_name, reduction=self.reduction, normalize=self.normalize)

    @staticmethod
    def expand_1channel_to_nchannel(input, desired_channels):
        """Assumes input is of shape [N, 1, H, W] and expands it to [N, n, H, W]."""
        assert input.size(1) == 1
        return input.expand(-1, desired_channels, -1, -1)

    def forward(self, predicted, target):
        """Forward pass."""
        if predicted.size(1) != self.required_channels:
            assert predicted.shape == target.shape
            predicted = self.expand_1channel_to_nchannel(
                input=predicted, desired_channels=self.required_channels
            )
            target = self.expand_1channel_to_nchannel(
                input=target, desired_channels=self.required_channels
            )
        loss = self.lpips(predicted, target)

        return loss


@capture_init_kwargs
class LPIPSMSELoss(nn.Module):
    """
    Implementation of a combined LPIPS-perceptual and pixel-wise (i.e. MSE) loss. The weighting is controlled via alpha.

    A gamma scaling term can be applied to the LPIPS loss.

    Parameters
    ----------
    alpha : float
        Weight balancing factor between LPIPS and pixel-wise loss.
    model_name : str
        Name of the LPIPs model to use for perceptual loss.
    reduction : str, optional
        Specifies the reduction to apply to the output. Default is "mean".
    gamma : float, optional
        Scaling factor for the LPIPS loss. Default is 1e-3.
    device : str, optional
        Device to use for computations. Default is "cpu".
    mse_type : str, optional
        Type of pixel-wise loss to use. Default is "l2".
    """

    def __init__(
        self,
        alpha,
        model_name: str,
        reduction: str = "mean",
        normalize: bool = False,
        gamma: float = 1e-3,
        mse_type: str = "l2",
        **kwargs,
    ):
        """Initialisation."""
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.model_name = model_name
        self.normalize = normalize
        self.gamma = gamma
        self.mse_type = mse_type
        self.lpips_loss = LPIPSLoss(
            model_name=self.model_name,
            reduction=self.reduction,
            normalize=self.normalize,
        )
        self.ploss = get_pixelwise_loss(name=self.mse_type, reduction=self.reduction, **kwargs)
        if self.ploss.__class__.__name__ == "CBLoss":
            # need to modify the (already processed) nodes of CB loss to [-1, 1] range if using LPIPS.
            # the default range of CB weights is [0, 1] from the data preprocessing.
            # LPIPs uses Tanh() as output activation ([-1, 1]).
            self.ploss.nnodes = 2 * self.ploss.nnodes - 1

    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        """Forward pass."""
        lpips = self.lpips_loss(predicted, target)
        p = self.ploss(predicted, target)
        return ((1 - self.alpha) * self.gamma * lpips) + (self.alpha * p)


@capture_init_kwargs
class VGGLoss(nn.Module):
    """
    Implements a VGG-based loss by comparing the activations of a
    pretrained VGG model at a specified layer.

    Code adapted from: https://github.com/crowsonkb/vgg_loss/blob/master/vgg_loss.py

    Attributes:
        reduction (str): Specifies the reduction to apply to the output:
        'mean', 'sum', or 'none'.
        normalize (transforms.Normalize): Normalization applied to
        input tensors.
        model (nn.Module): Pretrained VGG model truncated to
        the specified layer.

    Args:
        model (str): Identifier for the VGG model to use ('vgg16' or 'vgg19').
        layer (int): Layer of the VGG model at which to compute the loss.
        reduction (str): Reduction method for the loss calculation.
    """

    vgg_models = {"vgg16": models.vgg16, "vgg19": models.vgg19}

    def __init__(
        self,
        model_name: str = "vgg16",
        layer: int = 8,
        reduction: str = "mean",
        device: str = "cuda",
    ):
        """initialisation."""
        super().__init__()
        self.reduction = reduction
        self.device = device
        self.model_name = model_name
        self.layer = layer
        # adjust the normalization for a single channel. (currently R-channel).
        self.normalize = transforms.Normalize(mean=[0.485], std=[0.229])
        self.model = self.vgg_models[self.model_name](weights=True).features[: self.layer + 1]
        # set evaluation mode.
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self.device)

    def get_features(self, input_x: torch.Tensor) -> torch.Tensor:
        """Get the VGG conv features from the input image."""
        # expand the single channel input to three channels
        if input_x.shape[1] != 3:
            input_x = input_x.repeat(1, 3, 1, 1)

        return self.model(self.normalize(input_x))

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch = torch.cat([predicted, target])
        features = self.get_features(batch)
        predicted_features, target_features = torch.chunk(features, chunks=2, dim=0)

        return F.mse_loss(predicted_features, target_features, reduction=self.reduction)


@capture_init_kwargs
class VGGMSELoss(nn.Module):
    """
    Implementation of a combined VGG-perceptual and pixel-wise (i.e. MSE) loss. The weighting is controlled via alpha.

    A gamma scaling term can be applied to the VGG loss.

    Parameters
    ----------
    alpha : float
        Weight balancing factor between VGG and pixel-wise loss.
    model_name : str
        Name of the VGG model to use for perceptual loss.
    reduction : str, optional
        Specifies the reduction to apply to the output. Default is "mean".
    gamma : float, optional
        Scaling factor for the VGG loss. Default is 1e-3.
    device : str, optional
        Device to use for computations. Default is "cpu".
    mse_type : str, optional
        Type of pixel-wise loss to use. Default is "l2".
    """

    def __init__(
        self,
        alpha,
        model_name: str,
        reduction: str = "mean",
        gamma: float = 1e-3,
        device: str = "cpu",
        mse_type: str = "l2",
        **kwargs,
    ):
        """Initialisation."""
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.vgg_model = model_name
        self.device = device
        self.gamma = gamma  # scale VGG to align with ploss.
        self.vgg_loss = VGGLoss(model_name=self.vgg_model, reduction=self.reduction, device=self.device)
        self.mse_type = mse_type
        self.ploss = get_pixelwise_loss(self.mse_type, **kwargs)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        vgg = self.vgg_loss(predicted, target)
        p = self.ploss(predicted, target)
        return ((1 - self.alpha) * self.gamma * vgg) + (self.alpha * p)
