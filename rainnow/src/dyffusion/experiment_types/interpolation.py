"""Code adapted from: https://github.com/Rose-STL-Lab/dyffusion/blob/main/src/experiment_types/interpolation.py."""

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from torch import Tensor

from rainnow.src.dyffusion.experiment_types._base_experiment import BaseExperiment
from rainnow.src.loss import LPIPSMSELoss
from rainnow.src.normalise import PreProcess


class InterpolationExperiment(BaseExperiment):
    r"""Base class for all interpolation experiments."""

    def __init__(self, stack_window_to_channel_dim: bool = True, **kwargs):
        """Initialisation."""
        super().__init__(**kwargs)
        # save all the args that are passed to the constructor to self.hparams.
        self.save_hyperparameters(ignore=["model"])
        assert self.horizon >= 2, "horizon must be >=2 for interpolation experiments"

        # data normalisation.
        self.data_normalization = OmegaConf.to_container(self.datamodule_config["normalization"])
        self.preprocessor_obj = PreProcess(
            percentiles=self.data_normalization.get("percentiles"),
            minmax=self.data_normalization.get("min_max"),
        )

        # TODO: make this more abstract.
        # get loss criterion.
        # self.criterion = nn.L1Loss(reduction="mean")
        self.criterion = LPIPSMSELoss(
            alpha=0.6,
            model_name="alex",  # trains better with 'alex' - https://github.com/richzhang/PerceptualSimilarity.
            reduction="mean",
            gamma=1.0,
            mse_type="cb",
            **{"beta": 1, "data_preprocessor_obj": self.preprocessor_obj},
        )

    @property
    def horizon_range(self) -> List[int]:
        """Return the horizon range."""
        # We use timesteps  w-l+1, ..., w-1, w+h to predict timesteps w, ..., w+h-1
        # interpolate between step t=0 and t=horizon
        return list(np.arange(1, self.horizon))

    @property
    def true_horizon(self) -> int:
        """Return the self.horizon attr."""
        return self.horizon

    @property
    def horizon_name(self) -> str:
        """Return str '<self.horizon>h' as a str."""
        s = f"{self.true_horizon}h"
        return s

    @property
    def criterion_name(self) -> str:
        """Return the str name of self.criterion attr."""
        criterion_name = self.criterion.__class__.__name__.lower()
        return criterion_name

    @property
    def _model(self) -> nn.Module:
        """Return the underlying model of the experiment."""
        return self.model

    @property
    def model_final_layer_activation(self) -> nn.Module:
        """Return the final conv later activation of the experiment's underlying (interpolation) model."""
        return self._model.final_conv[-1]

    @property
    def experiment_type(self) -> str:
        """Return a str experiment type i.e. 'interpolation'."""
        return self.hparams.experiment_type

    @property
    def default_monitor_metric(self) -> str:
        """Return the metric used for checkpointing / any callbacks i.e. schedulers."""
        return f"val/{self.horizon_name}_avg/{self.experiment_type}/{self.criterion_name}"

    @property
    def short_description(self) -> str:
        name = super().short_description
        name += f" (h={self.horizon_name})"
        return name

    def actual_num_input_channels(self, num_input_channels: int) -> int:
        """
        Calculate the actual number of input channels based on the model configuration.

        For the simple case, returns 2*num input channels to replicate the initial
        condition and horizon used as interpolator inputs I(x0, xh).

        Parameters
        ----------
        num_input_channels : int
            The original number of input channels.

        Returns
        -------
        int
            The actual number of input channels after applying the model's configuration.
        """

        if self.hparams.stack_window_to_channel_dim:
            return num_input_channels * self.window + num_input_channels
        return 2 * num_input_channels  # inputs and targets are concatenated

    def get_metrics(self, split: str, split_name: str, **kwargs) -> torch.nn.ModuleDict:
        """
        Get the metrics dictionary used for evaluation for the specified split.

        This function overrides get_metrics in BaseExperiment. The default metric
        for DYffusion is MSE. The training loss is added on top of the MSE.
        For the average metrics, '_avg' is used as the identifier string.

        Parameters
        ----------
        split : str
            The split type (e.g., 'train', 'val', 'test').
        split_name : str
            The name of the split to be used in metric keys.

        Returns
        -------
        nn.ModuleDict
            A ModuleDict containing the metrics for the specified split.
        """
        metrics = {
            f"{split_name}/{self.horizon_name}_avg/{self.experiment_type}/mse": torch.nn.MSELoss(),
            f"{split_name}/{self.horizon_name}_avg/{self.experiment_type}/{self.criterion_name}": self.criterion,
        }
        for h in self.horizon_range:
            metrics[f"{split_name}/t{h}/{self.experiment_type}/mse"] = torch.nn.MSELoss()
            metrics[f"{split_name}/t{h}/{self.experiment_type}/{self.criterion_name}"] = self.criterion

        return torch.nn.ModuleDict(metrics)

    @torch.no_grad()
    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        dataloader_idx: int = None,
        return_only_preds_and_targets: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform an evaluation step for the Interpolation experiment on a batch of data.

        In each eval step, an ensemble (=num_predictions) of predictions will be used to calcualte the metrics
        including per t metrics.

        1. get inputs and get metrics (from the .get_metrics() class method).
        2. interpolate the intermediate values sequentially.
        3. calculate metrics for both per timestep and mean ensemble predictions.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A dictionary containing the batch data, with 'dynamics' as a key.
        batch_idx : int
            The index of the current batch.
        split : str
            The current data split (e.g., 'train', 'val', 'test').
        dataloader_idx : int, optional
            The index of the dataloader, by default None.
        return_only_preds_and_targets : bool, optional
            If True, return only predictions and targets, by default False.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the evaluation results, including predictions
            and targets for each time step, and potentially other computed values.
        """
        dynamics = batch["dynamics"]  # dynamics is a (b, t, c, h, w) tensor.
        # if output activation is Tanh need to need to normalise targets to is [-1, 1].
        # only need to modify != x0 or xh.
        if isinstance(self.model_final_layer_activation, nn.Tanh):
            dynamics[:, self.horizon_range, ...] = 2 * dynamics[:, self.horizon_range, ...] - 1
        # get inputs.
        inputs = self.get_evaluation_inputs(dynamics, split=split)
        extra_kwargs = {}
        for k, v in batch.items():
            if k != "dynamics":
                extra_kwargs[k] = self.get_ensemble_inputs(v, split=split, add_noise=False)

        # get all metrics.
        compute_metrics = split != "predict"
        split_metrics = getattr(self, f"{split}_metrics") if compute_metrics else None
        avg_metric_keys = [key for key in split_metrics.keys() if "_avg" in key]
        # create a log-able metric tracker (start from 0).
        metric_tracker = {k: 0 for k, _ in split_metrics.items()}
        return_dict = dict()  # store results (preds and targets).
        for t_step in self.horizon_range:
            # dynamics[, self.window] is already the first target frame (t_step=1)
            targets = dynamics[:, self.window + t_step - 1, ...]  # (b, c, h, w)
            time = torch.full((inputs.shape[0],), t_step, device=self.device, dtype=torch.long)
            results = self.predict(inputs, time=time, **extra_kwargs)
            results["targets"] = targets
            preds = results["preds"]
            results = {f"t{t_step}_{k}": v for k, v in results.items()}

            if return_only_preds_and_targets:
                return_dict[f"t{t_step}_preds"] = preds
                return_dict[f"t{t_step}_targets"] = targets
            else:
                return_dict = {**return_dict, **results}
            if not compute_metrics:
                continue
            if self.use_ensemble_predictions(split):
                preds = preds.mean(dim=0)  # average over ensemble

            # compute t_step losses and track avg loss.
            metric_names = [key for key in split_metrics.keys() if f"t{t_step}" in key]
            for e, metric_name in enumerate(metric_names):
                metric = split_metrics[metric_name]
                loss = metric(preds, targets)
                metric_tracker[metric_name] = loss
                metric_tracker[avg_metric_keys[e]] += loss

        # divide avg metric tracker by num t_steps.
        metric_tracker = {
            k: (v / len(self.horizon_range) if k in avg_metric_keys else v)
            for k, v in metric_tracker.items()
        }

        if compute_metrics:
            self.log_dict(metric_tracker, on_step=False, on_epoch=True, sync_dist=True)
        return return_dict

    def get_inputs_from_dynamics(self, dynamics: Tensor, split: str, **kwargs) -> Tensor:
        """
        Get the inputs from the dynamics tensor.

        This function extracts the input frames for the model from the dynamics tensor.
        Since we are doing interpolation, this consists of the first window frames plus the last frame.

        Parameters
        ----------
        dynamics : Tensor
            The input dynamics tensor with shape (b, t, c, h, w), where:
            b: batch size, t: time steps, c: channels, h: height, w: width.
        split : str
            The current data split (e.g., 'train', 'val', 'test').

        Returns
        -------
        Tensor
            The processed inputs for the model. Shape depends on the stack_window_to_channel_dim setting:
            - If True: (b, window*c + c, lat, lon)
            - If False: (b, window + 1, c, lat, lon)

        Raises
        ------
        AssertionError
            If the dynamics tensor doesn't have the expected shape.
        """
        assert (
            dynamics.shape[1] == self.window + self.horizon
        ), "dynamics must have shape (b, t, c, h, w)"
        past_steps = dynamics[:, : self.window, ...]  # (b, window, c, lat, lon) at time 0
        last_step = dynamics[:, -1, ...]  # (b, c, lat, lon) at time t=window+horizon
        if self.hparams.stack_window_to_channel_dim:
            past_steps = rearrange(past_steps, "b window c lat lon -> b (window c) lat lon")
        else:
            last_step = last_step.unsqueeze(1)  # (b, 1, c, lat, lon)
        inputs = torch.cat([past_steps, last_step], dim=1)  # (b, window*c + c, lat, lon)
        return inputs

    def get_evaluation_inputs(self, dynamics: Tensor, split: str, **kwargs) -> Tensor:
        """
        Wrapper function to get inputs for evaluation.
        The output of this function are used in the class method ._evaluation_step().

        Parameters
        ----------
        dynamics : Tensor
            The input dynamics tensor.
        split : str
            The current data split (e.g., 'train', 'val', 'test').

        Returns
        -------
        Tensor
            The processed inputs for evaluation.
        """
        inputs = self.get_inputs_from_dynamics(dynamics, split)
        inputs = self.get_ensemble_inputs(inputs, split)
        return inputs

    def get_loss(self, batch: Any) -> Tensor:
        """
        Compute the loss for the given batch.

        This function replicates training_step(). It takes in a batch, gets the prediction and calculates loss.

        This is a required method of pl.Trainer object.

        Parameters
        ----------
        batch : Dict[str, Tensor]
            A dictionary containing the batch data, with 'dynamics' as a key.

        Returns
        -------
        Tensor
            The computed loss for the batch.
        """
        dynamics = batch["dynamics"]  # dynamics is a (b, t, c, h, w) tensor
        split = "train" if self.training else "val"
        inputs = self.get_inputs_from_dynamics(dynamics, split=split)  # (b, c, h, w) at time 0
        b = dynamics.shape[0]

        # take random choice of possible time.
        possible_times = torch.tensor(self.horizon_range, device=self.device, dtype=torch.long)  # (h,)
        t = possible_times[
            torch.randint(len(possible_times), (b,), device=self.device, dtype=torch.long)
        ]  # (b,)
        targets = dynamics[torch.arange(b), self.window + t - 1, ...]  # (b, c, h, w).
        # We use timesteps  w-l+1, ..., w-1, w+h to predict timesteps w, ..., w+h-1
        # so t=0 corresponds to interpolating w, t=1 to w+1, ..., t=h-1 to w+h-1

        # if output activation is Tanh need to need to normalise targets to is [-1, 1].
        if isinstance(self.model_final_layer_activation, nn.Tanh):
            targets = 2 * targets - 1

        # make prediction and get loss.
        _kwargs = {k: v for k, v in batch.items() if k != "dynamics"}
        predictions = self.model(inputs, time=t, **_kwargs)
        loss = self.criterion(predictions, targets)

        return loss
