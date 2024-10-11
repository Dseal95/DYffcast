"""Code adapted from: https://github.com/Rose-STL-Lab/dyffusion/blob/main/src/experiment_types/forecasting_multi_horizon.py."""

from __future__ import annotations

import math
from abc import ABC
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from torch import Tensor
from tqdm.auto import tqdm

from rainnow.src.dyffusion.diffusion.dyffusion import BaseDYffusion
from rainnow.src.dyffusion.experiment_types._base_experiment import BaseExperiment
from rainnow.src.loss import LPIPSMSELoss
from rainnow.src.normalise import PreProcess
from rainnow.src.utilities.utils import (
    rrearrange,
    torch_select,
    torch_to_numpy,
    transform_0_1_to_minus1_1,
)


class AbstractMultiHorizonForecastingExperiment(BaseExperiment, ABC):
    def __init__(
        self,
        autoregressive_steps: int = 0,
        prediction_timesteps: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        """Initialisation."""
        assert (
            autoregressive_steps >= 0
        ), f"Autoregressive steps must be >= 0, but is {autoregressive_steps}"
        if autoregressive_steps > 0:
            assert (
                self.prediction_horizon is None
            ), "Cannot use ``prediction_horizon`` with autoregressive_steps > 0"
        self.stack_window_to_channel_dim = True
        super().__init__(**kwargs)
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.autoregressive_steps
        self.save_hyperparameters(ignore=["model"])
        self.USE_TIME_AS_EXTRA_INPUT = False
        self._test_metrics_aggregate = defaultdict(list)
        self._prediction_timesteps = prediction_timesteps
        self.hparams.pop("prediction_timesteps", None)
        if prediction_timesteps is not None:
            self.log_text.info(f" Using prediction timesteps {prediction_timesteps}")

        if self.num_autoregressive_steps > 0:
            ar_steps = self.num_autoregressive_steps
            max_horizon = self.true_horizon * (ar_steps + 1)
            self.log_text.info(
                f" Inference with {ar_steps} autoregressive steps for max. horizon={max_horizon}."
            )

        # data normalisation.
        self.data_normalization = OmegaConf.to_container(self.datamodule_config["normalization"])
        self.preprocessor_obj = PreProcess(
            percentiles=self.data_normalization.get("percentiles"),
            minmax=self.data_normalization.get("min_max"),
        )

        # TODO: abstract this out into a loss class.
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
        return list(np.arange(1, self.horizon + 1))

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
        return self.model.model

    @property
    def model_final_layer_activation(self) -> nn.Module:
        """Return the final conv later activation of the experiment's underlying (forecasting) model."""
        return self._model.final_conv[-1]

    @property
    def experiment_type(self) -> str:
        """Return a str experiment type i.e. 'dyffusion'."""
        return self.hparams.experiment_type

    @property
    def default_monitor_metric(self) -> str:
        """Return the metric used for checkpointing / any callbacks i.e. schedulers."""
        return f"val/{self.horizon_name}_avg/{self.experiment_type}/{self.criterion_name}"

    @property
    def prediction_timesteps(self) -> List[float]:
        """By default, we predict the timesteps in the horizon range (i.e. at data resolution)"""
        return self._prediction_timesteps or self.horizon_range

    @property
    def num_autoregressive_steps(self) -> int:
        """Return the number of auto-regressive rollout steps used during the forecast."""
        n_autoregressive_steps = self.hparams.autoregressive_steps
        if n_autoregressive_steps == 0 and self.prediction_horizon is not None:
            n_autoregressive_steps = max(1, math.ceil(self.prediction_horizon / self.true_horizon)) - 1
        return n_autoregressive_steps

    @property
    def short_description(self) -> str:
        name = super().short_description
        name += f" (h={self.horizon_name})"
        return name

    @property
    def prediction_horizon(self) -> int:
        """Return the prediction horizon."""
        if (
            hasattr(self.datamodule_config, "prediction_horizon")
            and self.datamodule_config.prediction_horizon
        ):
            return self.datamodule_config.prediction_horizon
        return self.horizon * (self.hparams.autoregressive_steps + 1)

    @prediction_timesteps.setter
    def prediction_timesteps(self, value: List[float]):
        assert (
            max(value) <= self.horizon_range[-1]
        ), f"Prediction range {value} exceeds horizon range {self.horizon_range}"
        self._prediction_timesteps = value

    def actual_num_input_channels(self, num_input_channels: int) -> int:
        # if we use the inputs as conditioning, and use an output-shaped input (e.g. for DDPM),
        # we need to use the output channels here!
        is_standard_diffusion = (
            self.is_diffusion_model and "dyffusion" not in self.diffusion_config._target_.lower()
        )
        is_dyffusion = self.is_diffusion_model and "dyffusion" in self.diffusion_config._target_.lower()
        if is_standard_diffusion:
            return self.actual_num_output_channels(self.dims["output"])
        elif is_dyffusion:
            return num_input_channels  # window is used as conditioning
        if self.stack_window_to_channel_dim:
            return num_input_channels * self.window
        return num_input_channels

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
        return_outputs: (
            bool | str
        ) = True,  # "auto",  #  True = -> "preds" + "targets". False: None "all": all outputs
        boundary_conditions: Callable = None,
        t0: float = 0.0,
        dt: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Emulate an evaluate step and calculate per t and mean ensemble metrics.

        1. get inputs and get metrics (from the .get_metrics() class method).
        2. generate a sequence, length = h with auto-regressive roll-out.
        3. calculate metrics for both per timestep and mean ensemble predictions.

        Parameters
        ----------
        batch : Any
            The input batch data.
        batch_idx : int
            The index of the current batch.
        split : str
            The data split ('val', 'test', or 'predict').
        dataloader_idx : int, optional
            The index of the dataloader.
        return_outputs : Union[bool, str], optional
            Whether to return outputs and what type (default is True).
        boundary_conditions : Callable, optional
            Function to apply boundary conditions.
        t0 : float, optional
            Initial time (default is 0.0).
        dt : float, optional
            Time step (default is 1.0).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the evaluation results.
        """
        # get inputs.
        # compute how many autoregressive steps to complete:
        dynamics = batch["dynamics"].clone()
        if split == "val" and dataloader_idx in [0, None]:
            n_outer_loops = 1  # Simple evaluation without autoregressive steps
        else:
            assert split in ["val", "test", "predict"]
            n_outer_loops = self.num_autoregressive_steps + 1
            if dynamics.shape[1] < self.prediction_horizon:
                raise ValueError(
                    f"Prediction horizon {self.prediction_horizon} is larger than {dynamics.shape}[1]"
                )

        # get metrics.
        compute_metrics = split not in ["test", "predict"]
        split_metrics = getattr(self, f"{split}_metrics") if compute_metrics else None
        avg_metric_keys = [key for key in split_metrics.keys() if "_avg" in key]

        # create a log-able metric tracker (start from 0).
        metric_tracker = {k: 0 for k, _ in split_metrics.items()}
        return_dict = dict()  # store results (preds and targets).
        if return_outputs == "auto":
            return_outputs = True if split == "predict" else False

        log_dict = {
            "num_predictions": self.hparams.num_predictions,
            "noise_level": self.inputs_noise,
        }

        # initialize autoregressive loop.
        autoregressive_inputs = None
        total_t = t0
        predicted_range_last = [0.0] + self.prediction_timesteps[:-1]
        ar_window_steps_t = self.horizon_range[-self.window :]  # autoregressive window steps,
        for ar_step in tqdm(
            range(n_outer_loops),
            desc="Autoregressive Step",
            position=0,
            leave=True,
            disable=not self.verbose or n_outer_loops <= 1,
        ):
            ar_window_steps = []
            for t_step_last, t_step in zip(predicted_range_last, self.prediction_timesteps):
                total_horizon = ar_step * self.true_horizon + t_step
                if total_horizon > self.prediction_horizon:
                    # may happen if we have a prediction horizon that is not a multiple of the true horizon.
                    break

                pr_kwargs = {} if autoregressive_inputs is None else {"num_predictions": 1}
                results = self.get_preds_at_t_for_batch(
                    batch,
                    t_step,
                    split,
                    autoregressive_inputs,
                    ensemble=True,
                    **pr_kwargs,
                )
                total_t += dt * (t_step - t_step_last)  # update time, by default this is == dt
                total_horizon = ar_step * self.true_horizon + t_step

                # get target(s).
                if float(total_horizon).is_integer():
                    # get the target at t.
                    target_time = self.window + int(total_horizon) - 1
                    targets = dynamics[:, target_time, ...]
                    if isinstance(self.model_final_layer_activation, nn.Tanh):
                        targets = transform_0_1_to_minus1_1(targets)
                else:
                    targets = None

                # returning the predictions, handling BCs if neccessary.
                if boundary_conditions is not None:
                    results[f"t{t_step}_preds"] = boundary_conditions(
                        preds=results[f"t{t_step}_preds"],
                        targets=targets,
                        metadata=batch.get("metadata", None),
                        time=total_t,
                    )

                preds = results.pop(f"t{t_step}_preds")
                if return_outputs in [True, "all"]:
                    return_dict[f"t{total_horizon}_targets"] = torch_to_numpy(targets)
                    return_dict[f"t{total_horizon}_preds"] = torch_to_numpy(preds)

                if return_outputs == "all":
                    return_dict.update(
                        {
                            k.replace(f"t{t_step}", f"t{total_horizon}"): torch_to_numpy(v)
                            for k, v in results.items()
                        }
                    )  # update keys to total horizon (instead of relative horizon of autoregressive step).

                if t_step in ar_window_steps_t:
                    # if predicted_range == self.horizon_range and window == 1, then this is just the last step.
                    # need to keep the last window steps that are INTEGER steps, handles synthetic steps.
                    ar_window_steps += [preds.reshape(-1, *preds.shape[-3:]).unsqueeze(1)]  # (t,c,h,w).

                if not compute_metrics or not float(total_horizon).is_integer():
                    continue

                if self.use_ensemble_predictions(split):
                    preds = preds.mean(dim=0)  # average over ensemble.
                    assert preds.shape == targets.shape, (
                        f"After averaging over ensemble dim: "
                        f"preds.shape={preds.shape} != targets.shape={targets.shape}"
                    )

                # compute t_step losses and track avg loss.
                metric_names = [key for key in split_metrics.keys() if f"t{t_step}" in key]
                for e, metric_name in enumerate(metric_names):
                    metric = split_metrics[metric_name]
                    loss = metric(preds, targets)
                    metric_tracker[metric_name] = loss
                    metric_tracker[avg_metric_keys[e]] += loss

            # handle averages.
            metric_tracker = {
                k: (v / len(self.horizon_range) if k in avg_metric_keys else v)
                for k, v in metric_tracker.items()
            }

            if ar_step < n_outer_loops - 1:  # if not last step, then update dynamics.
                autoregressive_inputs = torch.cat(ar_window_steps, dim=1)  # (b, window, c, h, w).
                autoregressive_inputs = self.transform_inputs(
                    autoregressive_inputs, split=split, ensemble=False
                )
                batch["dynamics"] *= 1e6  # become completely dummy after first multistep prediction.

        if compute_metrics:
            log_dict = {**log_dict, **metric_tracker}  # combine metrics with additonal log info.
            self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)

        return return_dict

    def get_preds_at_t_for_batch(
        self,
        batch: Dict[str, Tensor],
        horizon: int | float,
        split: str,
        autoregressive_inputs: Optional[Tensor] = None,
        ensemble: bool = False,
        prepare_inputs: bool = True,
        **kwargs,
    ) -> Dict[str, Tensor]:
        batch_shape = batch["dynamics"].shape
        b, t = batch_shape[0:2]
        assert 0 < horizon <= self.true_horizon, f"horizon={horizon} must be in [1, {self.true_horizon}]"
        isi1 = isinstance(self, MultiHorizonForecastingDYffusion)
        isi2 = isinstance(self, SimultaneousMultiHorizonForecasting)
        isi3 = isinstance(self, MultiHorizonForecastingTimeConditioned)
        cache_preds = isi1 or isi2
        if not cache_preds or horizon == self.prediction_timesteps[0]:
            if self.prediction_timesteps != self.horizon_range:
                if isi1:
                    self.model.hparams.prediction_timesteps = [p_h for p_h in self.prediction_timesteps]
            # create time tensor full of t_step, with batch size shape.
            t_tensor = torch.full((b,), horizon, device=self.device, dtype=torch.float) if isi3 else None
            if prepare_inputs:
                inputs, extra_kwargs = self.get_inputs_and_extra_kwargs(
                    batch,
                    time=t_tensor,
                    split=split,
                    autoregressive_inputs=autoregressive_inputs,
                    ensemble=ensemble,
                )
            else:
                inputs = batch.pop("dynamics")
                extra_kwargs = batch
                if isi3:
                    extra_kwargs["time"] = t_tensor

            with torch.no_grad():
                self._current_preds = self.predict(inputs, **extra_kwargs, **kwargs)

        if cache_preds:
            # for this model, we can cache the multi-horizon predictions.
            preds_key = f"t{horizon}_preds"
            results = {
                k: self._current_preds.pop(k) for k in list(self._current_preds.keys()) if preds_key in k
            }
            if horizon == self.horizon_range[-1]:
                assert all(["preds" not in k for k in self._current_preds.keys()]), (
                    f'preds_key={preds_key} must be the only key containing "preds" in last prediction. '
                    f"Got: {list(self._current_preds.keys())}"
                )
                results = {**results, **self._current_preds}
                del self._current_preds
        else:
            results = {f"t{horizon}_{k}": v for k, v in self._current_preds.items()}
        return results

    def get_inputs_from_dynamics(
        self, dynamics: Tensor | Dict[str, Tensor]
    ) -> Tensor | Dict[str, Tensor]:
        return dynamics[:, : self.window, ...]  # (b, window, c, lat, lon) at time 0

    def transform_inputs(
        self, inputs: Tensor, time: Tensor = None, ensemble: bool = True, **kwargs
    ) -> Tensor:
        if self.stack_window_to_channel_dim and len(inputs.shape) == 5:
            inputs = rrearrange(inputs, "b window c lat lon -> b (window c) lat lon")
        if ensemble:
            inputs = self.get_ensemble_inputs(inputs, **kwargs)
        return inputs

    def get_extra_model_kwargs(
        self,
        batch: Dict[str, Tensor],
        split: str,
        time: Tensor,
        ensemble: bool,
        is_autoregressive: bool = False,
    ) -> Dict[str, Any]:
        dynamics_shape = batch["dynamics"].shape  # b, dyn_len, c, h, w = dynamics.shape
        extra_kwargs = {}
        ensemble_k = ensemble and not is_autoregressive
        if self.USE_TIME_AS_EXTRA_INPUT:
            batch["time"] = time
        for k, v in batch.items():
            if k == "dynamics":
                continue
            elif k == "metadata":
                extra_kwargs[k] = (
                    self.get_ensemble_inputs(v, split=split, add_noise=False) if ensemble_k else v
                )
                continue

            v_shape_no_channel = v.shape[1 : self.CHANNEL_DIM] + v.shape[self.CHANNEL_DIM + 1 :]
            time_varying_feature = (
                dynamics_shape[1 : self.CHANNEL_DIM] + dynamics_shape[self.CHANNEL_DIM + 1 :]
            )

            if v_shape_no_channel == time_varying_feature:
                # if same shape as dynamics (except for batch size/#channels), then assume it is a time-varying feature
                extra_kwargs[k] = self.get_inputs_from_dynamics(v)
                extra_kwargs[k] = self.transform_inputs(
                    extra_kwargs[k],
                    split=split,
                    time=time,
                    ensemble=ensemble,
                    add_noise=False,
                )
            else:
                # Static features
                extra_kwargs[k] = (
                    self.get_ensemble_inputs(v, split=split, add_noise=False) if ensemble else v
                )
        return extra_kwargs

    def get_inputs_and_extra_kwargs(
        self,
        batch: Dict[str, Tensor],
        time: Tensor = None,
        split: str = None,
        ensemble: bool = True,
        autoregressive_inputs: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        is_autoregressive = autoregressive_inputs is not None
        if is_autoregressive:
            inputs = autoregressive_inputs
        else:
            inputs = self.get_inputs_from_dynamics(batch["dynamics"])
            inputs = self.transform_inputs(inputs, split=split, ensemble=True)
        extra_kwargs = self.get_extra_model_kwargs(
            batch,
            split=split,
            time=time,
            ensemble=ensemble,
            is_autoregressive=is_autoregressive,
        )
        return inputs, extra_kwargs


class MultiHorizonForecastingDYffusion(AbstractMultiHorizonForecastingExperiment):
    model: BaseDYffusion

    def __init__(self, *args, **kwargs):
        """Initialisation."""
        super().__init__(*args, **kwargs)
        assert (
            self.diffusion_config is not None
        ), "diffusion config must be set. Use ``diffusion=<dyffusion>``!"
        assert (
            self.diffusion_config.timesteps == self.horizon
        ), "diffusion timesteps must be equal to horizon"
        if hasattr(self.model, "interpolator"):
            self.log_text.info(
                f"------------------- Setting num_predictions={self.hparams.num_predictions}"
            )
            self.model.interpolator.hparams.num_predictions = self.hparams.num_predictions

    def use_stacked_ensemble_inputs(self, split) -> bool:
        return True  # always use stacked ensemble inputs

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        split = "train" if self.training else "val"
        dynamics = batch["dynamics"]
        x_last = dynamics[:, -1, ...]
        inputs, extra_kwargs = self.get_inputs_and_extra_kwargs(batch, split=split, ensemble=False)
        loss = self.model.get_loss(
            inputs=inputs, targets=x_last, criterion=self.criterion, **extra_kwargs
        )
        return loss


# ** the below classes are not used and were taken
# directly from https://github.com/Rose-STL-Lab/dyffusion/blob/main/src/experiment_types/forecasting_multi_horizon.py **
class MultiHorizonForecastingTimeConditioned(AbstractMultiHorizonForecastingExperiment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.hidden_dims
        self.save_hyperparameters(ignore=["model"])
        self.USE_TIME_AS_EXTRA_INPUT = True

    def get_forward_kwargs(self, batch: Dict[str, Tensor]) -> dict:
        split = "train" if self.training else "val"
        dynamics = batch["dynamics"]
        b, t, c, h, w = dynamics.shape
        time = torch.randint(1, self.horizon + 1, (b,), device=self.device, dtype=torch.long)
        assert (time >= 1).all() and (
            time <= self.true_horizon
        ).all(), f"Train time must be in [1, {self.true_horizon}], but got {time}"
        # Don't ensemble for validation of forward function losses
        # t0_data = self.get_inputs_from_dynamics(dynamics, split=split, ensemble=False)  # (b, c, h, w) at time 0
        shifted_t = (
            self.window + time - 1
        )  # window = past timesteps we use as input, so we shift by that
        targets = dynamics[torch.arange(b), shifted_t.long(), ...]

        inputs, extra_kwargs = self.get_inputs_and_extra_kwargs(
            batch, time=time, ensemble=False, split=split
        )

        kwargs = {**extra_kwargs, "inputs": inputs, "targets": targets}
        return kwargs

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        forward_kwargs = self.get_forward_kwargs(batch)
        loss, predictions = self.model.get_loss(return_predictions=True, **forward_kwargs)
        return loss


class SimultaneousMultiHorizonForecasting(AbstractMultiHorizonForecastingExperiment):
    def __init__(self, timestep_loss_weights: Sequence[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model", "timestep_loss_weights"])

        if timestep_loss_weights is None:
            timestep_loss_weights = [1.0 / self.horizon for _ in range(self.horizon)]
        self.timestep_loss_weights = timestep_loss_weights

    def actual_num_output_channels(self, num_output_channels: int) -> int:
        num_output_channels = super().actual_num_output_channels(num_output_channels)
        if self.stack_window_to_channel_dim:
            return num_output_channels * self.horizon
        return num_output_channels

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        dynamics = batch["dynamics"]
        split = "train" if self.training else "val"
        inputs, extra_kwargs = self.get_inputs_and_extra_kwargs(batch, split=split, ensemble=False)
        targets = dynamics[:, self.window :, ...]
        if self.stack_window_to_channel_dim:
            targets = rearrange(targets, "b t c h w -> b (t c) h w")  # (t_step = 1, ..., horizon)

        loss = self.model.get_loss(inputs=inputs, targets=targets, **extra_kwargs)
        return loss

    def reshape_predictions(
        self,
        results: Dict[str, Tensor],
        reshape_ensemble_dim: bool = True,
        add_ensemble_dim_in_inputs: bool = False,
    ) -> Dict[str, Tensor]:
        """Reshape and unpack the predictions from the model. This modifies the input dictionary in-place.
        Args:
           results (Dict[str, Tensor]): The model outputs. Access the predictions via results['preds'].
           reshape_ensemble_dim (bool, optional): Whether to reshape the ensemble dimension into the first dimension.
           add_ensemble_dim_in_inputs (bool, optional): Whether the ensemble dimension was added in the inputs.
        """
        # reshape predictions to (b, t, c, h, w), where t = num_time_steps predicted
        # ``b`` corresponds to the batch dimension and potentially the ensemble dimension
        results = {
            k: rrearrange(v, "b (t c) h w -> b t c h w", t=self.horizon) for k, v in results.items()
        }
        return super().reshape_predictions(results, reshape_ensemble_dim, add_ensemble_dim_in_inputs)

    def unpack_predictions(self, results: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Unpack the predictions from the model. This modifies the input dictionary in-place.
        Args:
           results (Dict[str, Tensor]): The model outputs. Access the predictions via results['preds'].
        """
        horizon_dim = self.CHANNEL_DIM - 1  # == -4
        preds = results.pop("preds")
        assert preds.shape[horizon_dim] == self.horizon
        for h in self.horizon_range:
            results[f"t{h}_preds"] = torch_select(preds, dim=horizon_dim, index=h - 1)
        return super().unpack_predictions(results)
