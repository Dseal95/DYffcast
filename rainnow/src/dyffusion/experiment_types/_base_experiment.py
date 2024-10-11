"""Code adapted from: https://github.com/Rose-STL-Lab/dyffusion/blob/main/src/experiment_types/_base_experiment.py."""

from __future__ import annotations

import inspect
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Union

import hydra
import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from tensordict import TensorDict
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from rainnow.src.dyffusion.datamodules.abstract_datamodule import BaseDataModule
from rainnow.src.dyffusion.datamodules.dataset_dimensions import get_dims_of_dataset
from rainnow.src.dyffusion.utilities.evaluation import evaluate_ensemble_prediction
from rainnow.src.models._base_model import BaseModel
from rainnow.src.models.modules.ema import LitEma
from rainnow.src.utilities.utils import get_logger, rrearrange, to_DictConfig, torch_to_numpy


class BaseExperiment(LightningModule):
    r"""This is a template base class, that should be inherited by any stand-alone ML model.
    Methods that need to be implemented by your concrete ML model (just as if you would define a :class:`torch.nn.Module`):
        - :func:`__init__`
        - :func:`forward`

    The other methods may be overridden as needed.
    It is recommended to define the attribute
        >>> self.example_input_array = torch.randn(<YourModelInputShape>)  # batch dimension can be anything, e.g. 7


    .. note::
        Please use the function :func:`predict` at inference time for a given input tensor, as it postprocesses the
        raw predictions from the function :func:`raw_predict` (or model.forward or model())!

    Args:
        optimizer: DictConfig with the optimizer configuration (e.g. for AdamW)
        scheduler: DictConfig with the scheduler configuration (e.g. for CosineAnnealingLR)
        monitor (str): The name of the metric to monitor, e.g. 'val/mse'
        mode (str): The mode of the monitor. Default: 'min' (lower is better)
        use_ema (bool): Whether to use an exponential moving average (EMA) of the model weights during inference.
        ema_decay (float): The decay of the EMA. Default: 0.9999 (only used if use_ema=True)
        enable_inference_dropout (bool): Whether to enable dropout during inference. Default: False
        num_predictions (int): The number of predictions to make for each input sample
        prediction_inputs_noise (float): The amount of noise to add to the inputs before predicting
        verbose (bool): Whether to print/log or not

    Read the docs regarding LightningModule for more information:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    CHANNEL_DIM = -3

    def __init__(
        self,
        model_config: DictConfig,
        datamodule_config: DictConfig,
        diffusion_config: Optional[DictConfig] = None,
        optimizer: Optional[DictConfig] = None,
        scheduler: Optional[DictConfig] = None,
        monitor: Optional[str] = None,
        mode: str = "min",
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        enable_inference_dropout: bool = False,
        num_predictions: int = 1,
        logging_infix: str = "",
        prediction_inputs_noise: float = 0.0,
        seed: int = None,
        verbose: bool = True,
        experiment_type: str = None,
    ):
        super().__init__()
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.monitor
        self.save_hyperparameters(
            ignore=["model_config", "datamodule_config", "diffusion_config", "verbose"]
        )
        # Get a logger
        self.log_text = get_logger(name=self.__class__.__name__)
        self.verbose = verbose
        self.logging_infix = logging_infix
        if not self.verbose:  # turn off info level logging
            self.log_text.setLevel(logging.WARN)

        self._datamodule = None
        self.model_config = model_config
        self.datamodule_config = datamodule_config
        self.diffusion_config = diffusion_config
        self.is_diffusion_model = (
            diffusion_config is not None and diffusion_config.get("_target_", None) is not None
        )
        self.dims = get_dims_of_dataset(self.datamodule_config)
        self.model = self.instantiate_model()

        # Initialize the EMA model, if needed
        self.use_ema = use_ema
        self.update_ema = use_ema or (
            self.is_diffusion_model and diffusion_config.get("consistency_strategy") == "ema"
        )
        if self.update_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay)
            self.log_text.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        self.model.ema_scope = self.ema_scope

        if enable_inference_dropout:
            self.log_text.info(" Enabling dropout during inference!")

        # Timing variables to track the training/epoch/validation time
        self._validation_step_outputs, self._predict_step_outputs = [], []
        self._test_step_outputs = defaultdict(list)

        # Metrics (will be initialized in the test/predict method if needed)
        self._val_metrics = self._test_metrics = self._predict_metrics = None

        # Check that the args/hparams are valid
        self._check_args()

        if self.use_ensemble_predictions("val"):
            self.log_text.info(f" Using a {num_predictions}-member ensemble for validation.")

        # Example input array, if set
        if hasattr(self.model, "example_input_array"):
            self.example_input_array = self.model.example_input_array

    @property
    def num_conditional_channels(self) -> int:
        """The number of channels that are used for conditioning as auxiliary inputs."""
        nc = self.dims.get("conditional", 0)
        if self.is_diffusion_model:
            d_class = self.diffusion_config.get("_target_").lower()
            is_standard_diffusion = "dyffusion" not in d_class
            if is_standard_diffusion:
                nc += (
                    self.window * self.dims["input"]
                )  # we use the data from the past window frames as conditioning

            fwd_cond = self.diffusion_config.get("forward_conditioning", "").lower()
            if fwd_cond == "":
                pass  # no forward conditioning, i.e. don't add anything
            elif fwd_cond == "data|noise":
                nc += 2 * self.window * self.dims["input"]
            elif fwd_cond in ["none", None]:
                pass
            else:
                nc += (
                    self.window * self.dims["input"]
                )  # we use the data from the past window frames as conditioning
        return nc

    @property
    def window(self) -> int:
        return self.datamodule_config.get("window", 1)

    @property
    def horizon(self) -> int:
        return self.datamodule_config.horizon

    @property
    def datamodule(self) -> BaseDataModule:
        if self._datamodule is None:  # alt: set in ``on_fit_start``  method
            self._datamodule = self.trainer.datamodule
        return self._datamodule

    @property
    def short_description(self) -> str:
        return self.__class__.__name__

    @property
    def test_set_name(self) -> str:
        return (
            self.trainer.datamodule.test_set_name
            if hasattr(self.trainer.datamodule, "test_set_name")
            else "test"
        )

    @property
    def prediction_set_name(self) -> str:
        return (
            self.trainer.datamodule.prediction_set_name
            if hasattr(self.trainer.datamodule, "prediction_set_name")
            else "predict"
        )

    @property
    def val_metrics(self):
        if self._val_metrics is None:
            self._val_metrics = self.get_metrics(split="val", split_name="val").to(self.device)
        return self._val_metrics

    @property
    def test_metrics(self):
        if self._test_metrics is None:
            self._test_metrics = self.get_metrics(split="test", split_name=self.test_set_name).to(
                self.device
            )
        return self._test_metrics

    @property
    def predict_metrics(self):
        if self._predict_metrics is None:
            self._predict_metrics = self.get_metrics(
                split="predict", split_name=self.prediction_set_name
            ).to(self.device)
        return self._predict_metrics

    @property
    def inputs_noise(self):
        # internally_probabilistic = isinstance(self.model, (GaussianDiffusion, DDPM))
        # return 0 if internally_probabilistic else self.hparams.prediction_inputs_noise
        return self.hparams.prediction_inputs_noise

    def actual_num_input_channels(self, num_input_channels: int) -> int:
        return num_input_channels

    def actual_num_output_channels(self, num_output_channels: int) -> int:
        return num_output_channels

    def instantiate_model(self, *args, **kwargs) -> BaseModel:
        r"""Instantiate the model, e.g. by calling the constructor of the class :class:`BaseModel` or a subclass thereof."""
        in_channels = self.actual_num_input_channels(self.dims["input"])
        out_channels = self.actual_num_output_channels(self.dims["output"])
        cond_channels = self.num_conditional_channels
        kwargs["datamodule_config"] = self.datamodule_config

        model = hydra.utils.instantiate(
            self.model_config,
            num_input_channels=in_channels,
            num_output_channels=out_channels,
            num_conditional_channels=cond_channels,
            spatial_shape=self.dims["spatial"],
            _recursive_=False,
            **kwargs,
        )
        self.log_text.info(
            f"Instantiated model: {model.__class__.__name__}, with"
            f" # input/output/conditional channels: {in_channels}, {out_channels}, {cond_channels}"
        )
        if self.is_diffusion_model:
            model = hydra.utils.instantiate(
                self.diffusion_config, model=model, _recursive_=False, **kwargs
            )
            self.log_text.info(
                f"Instantiated diffusion model: {model.__class__.__name__}, with"
                f" #diffusion steps={model.num_timesteps}"
            )
        return model

    def forward(self, *args, **kwargs) -> Any:
        y = self.model(*args, **kwargs)
        return y

    def get_metrics(self, split: str, split_name: str, **kwargs) -> torch.nn.ModuleDict:
        raise NotImplementedError(
            f"Please implement the method 'get_metrics' in your concrete class {self.__class__.__name__}"
        )

    def _check_args(self):
        """Check if the arguments are valid."""
        pass

    @contextmanager
    def ema_scope(self, context=None, force_non_ema: bool = False, condition: bool = None):
        """Context manager to switch to EMA weights."""
        condition = self.use_ema if condition is None else condition
        if condition and not force_non_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if condition and not force_non_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @contextmanager
    def inference_dropout_scope(self, condition: bool = None, context=None):
        """Context manager to switch to inference dropout mode.
        Args:
            condition (bool, optional): If True, switch to inference dropout mode. If False, switch to training mode.
                If None, use the value of self.hparams.enable_inference_dropout.
                Important: If not None, self.hparams.enable_inference_dropout is ignored!
            context (str, optional): If not None, print this string when switching to inference dropout mode.
        """
        condition = self.hparams.enable_inference_dropout if condition is None else condition
        if condition:
            self.model.enable_inference_dropout()
            if context is not None:
                print(f"{context}: Switched to enabled inference dropout")
        try:
            yield None
        finally:
            if condition:
                self.model.disable_inference_dropout()
                if context is not None:
                    print(f"{context}: Switched to disabled inference dropout")

    @contextmanager
    def timing_scope(self, context="", no_op=True, precision=2):
        """Context manager to measure the time of the code inside the context. (By default, does nothing.)
        Args:
            context (str, optional): If not None, print time elapsed in this context.
        """
        start_time = time.time() if not no_op else None
        try:
            yield None
        finally:
            if not no_op:
                context = f"``{context}``:" if context else ""
                print(f"Elapsed time {context} {time.time() - start_time:.{precision}f}s")

    def predict(
        self,
        inputs: Tensor,
        num_predictions: Optional[int] = None,
        reshape_ensemble_dim: bool = True,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        This should be the main method to use for making predictions/doing inference.

        Args:
            inputs (Tensor): Input data tensor of shape :math:`(B, *, C_{in})`.
                This is the same tensor one would use in :func:`forward`.
            num_predictions (int, optional): Number of predictions to make. If None, use the default value.
            reshape_ensemble_dim (bool, optional): Whether to reshape the ensemble dimension into the first dimension.
            **kwargs: Additional keyword arguments

        Returns:
            Dict[str, Tensor]: The model predictions (in a post-processed format), i.e. a dictionary output_var -> output_var_prediction,
                where each output_var_prediction is a Tensor of shape :math:`(B, *)` in original-scale (e.g.
                in Kelvin for temperature), and non-negativity has been enforced for variables such as precipitation.

        Shapes:
            - Input: :math:`(B, *, C_{in})`
            - Output: Dict :math:`k_i` -> :math:`v_i`, and each :math:`v_i` has shape :math:`(B, *)` for :math:`i=1,..,C_{out}`,

            where :math:`B` is the batch size, :math:`*` is the spatial dimension(s) of the data,
            and :math:`C_{out}` is the number of output features.
        """
        base_num_predictions = self.hparams.num_predictions
        self.hparams.num_predictions = num_predictions or base_num_predictions
        if (
            hasattr(self.model, "sample_loop")
            and "num_predictions" in inspect.signature(self.model.sample_loop).parameters
        ):
            kwargs["num_predictions"] = self.hparams.num_predictions

        results = self.model.predict_forward(inputs, **kwargs)

        if torch.is_tensor(results):
            results = {"preds": results}

        self.hparams.num_predictions = base_num_predictions
        results = self.reshape_predictions(results, reshape_ensemble_dim)
        results = self.unpack_predictions(results)
        return results

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
        ensemble_size = self.hparams.num_predictions
        pred_keys = [k for k in results.keys() if "preds" in k]
        preds_shape = results[pred_keys[0]].shape
        if reshape_ensemble_dim and preds_shape[0] > 1:
            if add_ensemble_dim_in_inputs or (ensemble_size > 1 and preds_shape[0] % ensemble_size == 0):
                results = self._reshape_ensemble_preds(results, "predict")
                preds_shape = results[pred_keys[0]].shape

            if 1 < ensemble_size == preds_shape[0] and len(preds_shape) <= 4:
                # unsqueeze batch dimension if it was removed (data is only (ensemble, c, h, w)
                for k in pred_keys:
                    results[k] = results[k].unsqueeze(1)
        return results

    def unpack_predictions(self, results: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Unpack the predictions from the model. This modifies the input dictionary in-place.
        Args:
           results (Dict[str, Tensor]): The model outputs. Access the predictions via results['preds'].
        """
        #  As of now, only keys with ``preds`` in them are unpacked.
        if self._trainer is None:
            return results
        return results

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch"""
        raise NotImplementedError(f"Please implement the get_loss method for {self.__class__.__name__}")

    def on_train_batch_start(self, *args, **kwargs):
        pass

    def training_step(self, batch: Any, batch_idx: int):
        r"""One step of training (backpropagation is done on the loss returned at the end of this function)."""
        loss_output = self.get_loss(batch)  # either a scalar or a dict with key 'loss'
        if isinstance(loss_output, dict):
            loss = loss_output.pop("loss")
        else:
            loss = loss_output

        # log train metrics.
        self.log(
            name="train/loss",
            value=float(loss),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {"loss": loss}

    def on_train_batch_end(self, *args, **kwargs):
        if self.update_ema:
            self.model_ema(self.model)  # update the model EMA

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        dataloader_idx: int = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        One step of evaluation (forward pass, potentially metrics computation, logging, and return of results)
        Returns:
            results_dict: Dict[str, Tensor], where for each semantically different result, a separate prefix key is used
                Then, for each prefix key <p>, results_dict must contain <p>_preds and <p>_targets.
        """
        raise NotImplementedError(
            f"Please implement the _evaluation_step method for {self.__class__.__name__}"
        )

    def evaluation_step(self, batch: Any, batch_idx: int, split: str, **kwargs) -> Dict[str, Tensor]:
        # handle boundary conditions.
        # unless you have BCs, this defaults to returning preds w/o any modifications.
        if "boundary_conditions" in inspect.signature(self._evaluation_step).parameters.keys():
            kwargs["boundary_conditions"] = self.datamodule.boundary_conditions
            kwargs.update(self.datamodule.get_boundary_condition_kwargs(batch, batch_idx, split))

        with self.ema_scope():  # use the EMA parameters for the validation step (if using EMA)
            with self.inference_dropout_scope():  # Enable dropout during inference
                return self._evaluation_step(batch, batch_idx, split, **kwargs)

    def evaluation_results_to_xarray(
        self, results: Dict[str, np.ndarray], **kwargs
    ) -> Dict[str, xr.DataArray]:
        return self.model.evaluation_results_to_xarray(results, **kwargs)

    def use_ensemble_predictions(self, split: str) -> bool:
        """Determine if use ensemble validation metrics."""
        return self.hparams.num_predictions > 1 and split in ["val", "test", "predict"]

    def use_stacked_ensemble_inputs(self, split: str) -> bool:
        return True

    def get_ensemble_inputs(
        self,
        inputs_raw: Optional[Tensor],
        split: str,
        add_noise: bool = True,
        flatten_into_batch_dim: bool = True,
    ) -> Optional[Tensor]:
        """Get the inputs for the ensemble predictions."""
        if inputs_raw is None:
            return None
        elif not self.use_stacked_ensemble_inputs(split):
            return inputs_raw  # we can sample from the Gaussian distribution directly after the forward pass
        elif self.use_ensemble_predictions(split):
            # create a batch of inputs for the ensemble predictions
            num_predictions = self.hparams.num_predictions
            if isinstance(inputs_raw, dict):
                inputs = {
                    k: self.get_ensemble_inputs(v, split, add_noise, flatten_into_batch_dim)
                    for k, v in inputs_raw.items()
                }
            else:
                if isinstance(inputs_raw, Sequence):
                    inputs = np.array([inputs_raw] * num_predictions)
                elif add_noise:
                    inputs = torch.stack(
                        [
                            inputs_raw + self.inputs_noise * torch.randn_like(inputs_raw)
                            for _ in range(num_predictions)
                        ],
                        dim=0,
                    )
                else:
                    inputs = torch.stack([inputs_raw for _ in range(num_predictions)], dim=0)

                if flatten_into_batch_dim:
                    # flatten num_predictions and batch dimensions
                    inputs = rrearrange(inputs, "N B ... -> (N B) ...")
        else:
            inputs = inputs_raw
        return inputs

    def _reshape_ensemble_preds(self, results: Dict[str, Tensor], split: str) -> Dict[str, Tensor]:
        r"""
        Reshape the predictions of an ensemble to the correct shape,
        where the first dimension is the ensemble dimension, N.

         Args:
                results: The model predictions (in a post-processed format),
                            i.e. a dictionary output_var -> output_var_prediction (plus potentially 'targets')
                split: The split for which the predictions are made (e.g. 'test' or 'predict')

        Returns:
            The reshaped predictions (i.e. each output_var_prediction has shape (N, B, *)). 'targets' is not reshaped.
        """
        num_predictions = self.hparams.num_predictions
        if self.use_ensemble_predictions(split):
            if not self.use_stacked_ensemble_inputs(split):
                return results

            # reshape preds into (num_predictions, batch_size, ...)
            for key in results:
                if (
                    "targets" not in key and "true" not in key
                ):  # and results[key].shape[0] != num_predictions:
                    b = results[key].shape[0]
                    assert (
                        b % num_predictions == 0
                    ), f"key={key}: b % #ens_mems = {b} % {num_predictions} != 0 ...Did you forget to create the input ensemble?"
                    batch_size = max(1, b // num_predictions)
                    results[key] = results[key].reshape(
                        num_predictions, batch_size, *results[key].shape[1:]
                    )
        return results

    def _evaluation_get_preds(
        self, outputs: List[Any]
    ) -> Dict[str, Union[torch.distributions.Normal, np.ndarray]]:
        num_predictions = self.hparams.num_predictions
        if isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], list):
            outputs = outputs[0]
        outputs_keys, results = outputs[0].keys(), dict()
        for key in outputs_keys:
            first_output = outputs[0][key]
            is_tensor_dict = isinstance(first_output, TensorDict)
            is_normal_dict = isinstance(first_output, dict)
            if is_normal_dict:
                s1, s2 = first_output[list(first_output.keys())[0]].shape[:2]
            else:
                s1, s2 = first_output.shape[:2]
            batch_axis = (
                1 if (s1 == num_predictions and "targets" not in key and "true" not in key) else 0
            )
            if is_normal_dict:
                results[key] = {
                    k: np.concatenate([out[key][k] for out in outputs], axis=batch_axis)
                    for k in first_output.keys()
                }
            elif is_tensor_dict:
                results[key] = torch.cat([out[key] for out in outputs], dim=batch_axis)
            else:
                try:
                    results[key] = np.concatenate([out[key] for out in outputs], axis=batch_axis)
                except ValueError as e:
                    raise ValueError(
                        f"Error when concatenating {key}: {e}.\n"
                        f"Shape 0: {first_output.shape}, -1: {outputs[-1][key].shape}"
                    )

        return results

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None, **kwargs):
        results = self.evaluation_step(
            batch, batch_idx, split="val", dataloader_idx=dataloader_idx, **kwargs
        )
        results = torch_to_numpy(results)
        self._validation_step_outputs.append(results)
        return results

    def ensemble_logging_infix(self, split: str) -> str:
        """
        Choose what prefix you want the validation/ensemble logs to have.
        Make sure that no '/' in front of the infix! But '/' at the end!
        """
        s = "" if self.logging_infix == "" else f"{self.logging_infix}/".replace("//", "/")
        if self.inputs_noise > 0.0 and split != "val":
            s += f"{self.inputs_noise}eps/"
        s += f"{self.hparams.num_predictions}ens_mems/"
        return s

    def _eval_ensemble_predictions(self, outputs: List[Any], split: str):
        """Validation ensemble metrics. go through all predictions and compute metrics (i.e. over preds for each time step).
        CRPS, L2 (MSE) and SSR scores are mandatory. The loss used for training will also be computed.
        """
        # get preds and targets for eval.
        numpy_results = self._evaluation_get_preds(outputs)  # keys <p>_preds, <p>_targets
        all_preds_metrics = defaultdict(list)
        preds_keys = [k for k in numpy_results.keys() if k.endswith("preds")]

        if not self.use_ensemble_predictions(split):
            # add a 1st dim to simulate ensemble when num_predictions = 1.
            numpy_results.update({key: np.expand_dims(numpy_results[key], axis=0) for key in preds_keys})
        for preds_key in preds_keys:
            prefix = preds_key.split("_")[0] if preds_key != "preds" else ""
            targets_key = f"{prefix}_targets" if prefix else "targets"
            metrics = evaluate_ensemble_prediction(
                predictions=numpy_results[preds_key],
                targets=numpy_results[targets_key],
                training_criterion=self.criterion,
                device=self.device,
            )
            preds_key_metrics = dict()  # reset for each t.
            for m, v in metrics.items():
                preds_key_metrics[
                    f"{split}/{self.ensemble_logging_infix(split)}{self.experiment_type}/{prefix}/{m}"
                ] = v
                all_preds_metrics[
                    f"{split}/{self.ensemble_logging_infix(split)}{self.experiment_type}/avg/{m}"
                ].append(v)
            self.log_dict(
                preds_key_metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False
            )

        # compute average metrics over all predictions.
        avg_metrics = {k: np.mean(v) for k, v in all_preds_metrics.items()}
        self.log_dict(avg_metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """do a the end of each validation epoch."""
        self._eval_ensemble_predictions(self._validation_step_outputs, split="val")
        self._validation_step_outputs = []  # reset the _validation_step_outputs.

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None, **kwargs):
        split = "test" if dataloader_idx is None else self.test_set_names[dataloader_idx]
        results = self.evaluation_step(batch, batch_idx, split="test", **kwargs)
        results = torch_to_numpy(results)  # self._reshape_ensemble_preds(results, split='test')
        self._test_step_outputs[split].append(results)
        return results

    def on_test_epoch_end(self, calc_ensemble_metrics: bool = True):
        """do a the end of each test epoch."""
        for test_split in self._test_step_outputs.keys():
            if calc_ensemble_metrics:
                self._eval_ensemble_predictions(self._test_step_outputs[test_split], split=test_split)
        self._test_step_outputs = defaultdict(list)

    def on_predict_start(self) -> None:
        for pdl in self.trainer.predict_dataloaders:
            assert (
                pdl.dataset.dataset_id == "predict"
            ), f"dataset_id is not 'predict', but {pdl.dataset.dataset_id}"

        n_preds = self.hparams.num_predictions
        if n_preds > 1:
            self.log_text.info(
                f"Generating {n_preds} predictions per input with noise level {self.inputs_noise}"
            )
        if "autoregressive_steps" in self.hparams:
            print(f"Autoregressive steps: {self.hparams.autoregressive_steps}")

    def on_predict_epoch_start(self) -> None:
        if self.inputs_noise > 0:
            self.log_text.info(f"Adding noise to inputs with level {self.inputs_noise}")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None, **kwargs):
        results = self.evaluation_step(batch, batch_idx, split="predict", **kwargs)
        results = torch_to_numpy(results)  # self._reshape_ensemble_preds(results, split='predict')
        self._predict_step_outputs.append(results)

    def on_predict_epoch_end(self):
        numpy_results = self._evaluation_get_preds(self._predict_step_outputs)
        self._predict_step_outputs = []
        return numpy_results

    def state_dict(self):
        """Override .state_dict() method to avoid adding in any weights and biases associated with self.criterion.

        As self.criterion is a class attr, any of it's w&b will be saved to the model ckpt without this override.
        """
        return self._model.state_dict()

    def _get_optim(self, optim_name: str, **kwargs):
        """
        Method that returns the torch.optim optimizer object.
        May be overridden in subclasses to provide custom optimizers.
        """
        if optim_name.lower() == "fusedadam":
            from apex import optimizers

            optimizer = optimizers.FusedAdam
        elif optim_name.lower() == "adamw":
            optimizer = torch.optim.AdamW
        else:
            raise ValueError(f"Unknown optimizer type: {optim_name}")
        self.log_text.info(f"{optim_name} optim with kwargs: " + str(kwargs))
        return optimizer(filter(lambda p: p.requires_grad, self.model.get_parameters()), **kwargs)

    def configure_optimizers(self):
        """A required method of pl.Trainer object. Configure optimizers (and schedulers).
        Lightning `Trainer` expects as minimum a `training_step()`, `train_dataloader()` and `configure_optimizers()` to be defined.
        """
        # optimizer, defaults to adamw if none passed.
        if "name" not in to_DictConfig(self.hparams.optimizer).keys():
            self.log_text.info(" No optimizer was specified, defaulting to AdamW.")
            self.hparams.optimizer.name = "adamw"
        optim_kwargs = {k: v for k, v in self.hparams.optimizer.items() if k not in ["name", "_target_"]}
        optimizer = self._get_optim(self.hparams.optimizer.name, **optim_kwargs)

        return optimizer
