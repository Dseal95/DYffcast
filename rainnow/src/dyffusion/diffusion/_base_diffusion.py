"""Code adapted from: https://github.com/Rose-STL-Lab/dyffusion/blob/main/src/diffusion/_base_diffusion.py."""

from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn
from torch import Tensor

from rainnow.src.models._base_model import BaseModel


class BaseDiffusion(BaseModel):
    def __init__(
        self,
        model: BaseModel,
        timesteps: int,
        sampling_timesteps: int = None,
        sampling_schedule=None,
        **kwargs,
    ):
        """Initialisation."""
        signature = inspect.signature(BaseModel.__init__).parameters
        base_kwargs = {k: model.hparams.get(k) for k in signature if k in model.hparams}
        base_kwargs.update(kwargs)  # override base_kwargs with kwargs
        super().__init__(**kwargs)
        if model is None:
            raise ValueError(
                "Arg ``model`` is missing..."
                " Please provide a backbone model for the diffusion model (e.g. a Unet)"
            )
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.spatial_shape = model.spatial_shape
        self.num_input_channels = model.num_input_channels
        self.num_output_channels = model.num_output_channels
        self.num_conditional_channels = model.num_conditional_channels

    @property
    def short_description(self) -> str:
        name = super().short_description
        name += f" (timesteps={self.num_timesteps})"
        return name

    def sample(self, condition=None, num_samples=1, **kwargs):
        raise NotImplementedError()

    def predict_forward(self, inputs, condition=None, metadata: Any = None, **kwargs):
        """
        Perform a forward prediction using the diffusion model.

        Parameters
        ----------
        inputs : Tensor
            The input data.
        condition : optional
            The condition for the prediction.
        metadata : Any, optional
            Additional metadata for the prediction.

        Returns
        -------
        Any
            The forward prediction results.
        """
        channel_dim = 1
        p_losses_args = inspect.signature(self.p_losses).parameters.keys()
        if inputs is not None and condition is not None:
            if "static_condition" in p_losses_args:
                kwargs["static_condition"] = condition
                inital_condition = inputs
            else:
                # Concatenate the "inputs" and the condition along the channel dimension as conditioning
                try:
                    inital_condition = torch.cat([inputs, condition], dim=channel_dim)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Could not concatenate the inputs (shape={inputs.shape}) and the condition "
                        f"(shape={condition.shape}) along the channel dimension (dim={channel_dim})"
                        f" due to the following error:\n{e}"
                    )
        else:  # if inputs is not None:
            inital_condition = inputs
        return self.sample(inital_condition, **kwargs)

    @abstractmethod
    def p_losses(self, targets: Tensor, condition: Tensor = None, t: Tensor = None, **kwargs):
        """
        Compute the loss for the given targets and condition.

        Parameters
        ----------
        targets : Tensor
            Target data tensor of shape (B, C_out, *).
        condition : Tensor, optional
            Condition data tensor of shape (B, C_in, *).
        t : Tensor, optional
            Timestep of shape (B,).
        """
        raise NotImplementedError()

    def forward(
        self,
        inputs: Tensor,
        criterion: nn.Module,
        targets: Tensor = None,
        condition: Tensor = None,
        time: Tensor = None,
    ):
        """Forward pass."""
        b, c, h, w = targets.shape if targets is not None else inputs.shape
        if time is not None:
            t = time
        else:
            t = torch.randint(0, self.num_timesteps, (b,), device=self.device, dtype=torch.long)

        p_losses_args = inspect.signature(self.p_losses).parameters.keys()
        kwargs = {}

        if "static_condition" in p_losses_args:
            # method handles it internally
            kwargs["static_condition"] = condition
            kwargs["condition"] = inputs
        elif condition is None:
            kwargs["condition"] = inputs
        else:
            channel_dim = 1
            kwargs["condition"] = torch.cat([inputs, condition], dim=channel_dim)

        assert condition is None
        assert torch.allclose(kwargs["condition"], inputs)

        return self.p_losses(targets, t=t, criterion=criterion, **kwargs)

    def get_loss(
        self, inputs: Tensor, targets: Tensor, criterion: nn.Module, metadata: Any = None, **kwargs
    ):
        """
        Get the loss for the given inputs and targets.

        Parameters
        ----------
        inputs : Tensor
            Input data tensor of shape (B, C_in, *).
        targets : Tensor
            Target data tensor of shape (B, C_out, *).
        criterion : nn.Module
            The loss criterion used to calculate the loss.
        metadata : Any, optional
            Optional metadata.

        Returns
        -------
        Any
            The computed loss results.
        """
        results = self(inputs=inputs, targets=targets, criterion=criterion, **kwargs)
        return results
