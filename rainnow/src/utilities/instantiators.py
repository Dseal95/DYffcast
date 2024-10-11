"""Code adapted from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/instantiators.py"""

import os
from pathlib import Path
from typing import Any, Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger

from rainnow.src.dyffusion.experiment_types.forecasting_multi_horizon import (
    MultiHorizonForecastingDYffusion,
)
from rainnow.src.utilities.utils import get_logger

log = get_logger(name=__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def update_callback_monitors(callbacks, new_monitor, callback_names_to_update=None) -> None:
    """
    Inplace update of the 'monitor' attribute for specified callbacks.

    This function is relates to pl.Lightning callbacks.

    This function updates the 'monitor' attribute of callbacks in the provided list
    if their class name is in the callback_names_to_update list.

    Parameters
    ----------
    callbacks : list
        List of callback objects to potentially update.
    new_monitor : str
        The new value to set for the 'monitor' attribute.
    callback_names_to_update : list of str, optional
        List of callback class names to update. If None, defaults to
        ['ModelCheckpoint', 'EarlyStopping', 'ReduceLROnPlateauCallback'].

    Returns
    -------
    None
    """
    if callback_names_to_update is None:
        callback_names_to_update = ["ModelCheckpoint", "EarlyStopping", "ReduceLROnPlateauCallback"]
    for callback in callbacks:
        if hasattr(callback, "monitor"):
            callback_name = callback.__class__.__name__
            if callback_name in callback_names_to_update:
                callback.monitor = new_monitor
                log.info(f"Updated {callback_name} monitor metric to: {new_monitor}")


# TODO: move this to a more suitable module.
def instantiate_multi_horizon_dyffusion_model(
    ckpt_id: str,
    ckpt_base_path: str,
    ckpt_config_name: str = "hparams.yaml",
    diffusion_config_overrides: Dict[str, Any] = {},
) -> MultiHorizonForecastingDYffusion:
    """
    Instantiate a MultiHorizonForecastingDYffusion model from a checkpoint configuration.

    Parameters
    ----------
    ckpt_id : str
        The identifier for the checkpoint.
    ckpt_base_path : str
        The base path where checkpoint directories are stored.
    ckpt_config_name : str, optional
        The name of the configuration file within the checkpoint directory.
        Default is "hparams.yaml".
    diffusion_config_overrides : Dict[str, Any], optional
        A dictionary of key-value pairs to override in the diffusion configuration.
        Default is an empty dictionary.
    """
    # load in model config from ckpt hparams.yaml.
    model_ckpt_cfg = OmegaConf.load(
        Path(os.path.join(ckpt_base_path, ckpt_id, "logs", ckpt_config_name))
    )

    if diffusion_config_overrides:
        model_ckpt_cfg.diffusion_config.update(**diffusion_config_overrides)

    # instantiate a DYffusion model.
    model = MultiHorizonForecastingDYffusion(
        model_config=model_ckpt_cfg.model_config,
        datamodule_config=model_ckpt_cfg.datamodule_config,
        diffusion_config=model_ckpt_cfg.diffusion_config,
    )
    return model
