"""Module with helper functions for loading in model checkpoints."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from rainnow.src.dyffusion.datamodules.imerg_precipitation import IMERGPrecipitationDataModule
from rainnow.src.dyffusion.experiment_types.interpolation import InterpolationExperiment
from rainnow.src.utilities.utils import get_trainable_parameter_names


def get_model_ckpt_path(
    ckpt_id: str, ckpt_base_path: str, ckpt_dir: str = "checkpoints", get_last: bool = True
) -> Path:
    """
    Construct and return a model's Path object.

    Parameters
    ----------
    ckpt_id : str
        The checkpoint ID.
    ckpt_base_path : str
        The base path for checkpoints.
    ckpt_dir : str, optional
        The directory name for checkpoints (default is "checkpoints").
    get_last : bool, optional
        Whether to get the last checkpoint (default is True).

    Returns
    -------
    Path
        The path to the selected checkpoint file.
    """
    ckpt_dir = Path(os.path.join(ckpt_base_path, ckpt_id, ckpt_dir))
    if get_last:
        ckpt_name = "last.ckpt"
    else:
        # use other model in the folder. If not there, resort to last.
        other_ckpts = [
            f.name
            for f in ckpt_dir.iterdir()
            if f.is_file() and f.name != "last.ckpt" and f.name.split(".")[-1] == "ckpt"
        ]
        if len(other_ckpts) == 0:
            ckpt_name = "last.ckpt"
        else:
            ckpt_name = other_ckpts[0]
    ckpt_path = Path(os.path.join(ckpt_dir, ckpt_name))

    return ckpt_path


def load_model_state_dict(ckpt_path: Union[str, Path], device: str) -> Dict[str, torch.Tensor]:
    """
    Load in a model's state dict.

    Parameters
    ----------
    ckpt_path : Union[str, Path]
        The path to the checkpoint file.
    device : str
        The device to map the loaded state dict to.

    Returns
    -------
    Dict[str, torch.Tensor]
        The loaded state dict of the model.
    """
    loaded_state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
    return loaded_state_dict


def reset_model_params(model) -> None:
    """
    Inplace reset a model's params (weights and biases).

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to reset.


    Returns
    -------
    None
    """
    for layer in model.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def check_params(loaded_params: List[str], model_params: List[str]) -> bool:
    """
    Check for a mismatch between loaded and model params.

    Untrainable params in the loaded model are handled naively.

    Parameters
    ----------
    loaded_params : List[str]
        List of parameter names from the loaded state dict.
    model_params : List[str]
        List of parameter names from the current model.

    Returns
    -------
    bool
        True if there's no mismatch, False otherwise.
    """
    # update this list accordingly.
    untrainable_params = ["running_mean", "running_var", "num_batches_tracked"]

    filtered_loaded = [
        param for param in loaded_params if not any(u in param for u in untrainable_params)
    ]
    filtered_model = [param for param in model_params if not any(u in param for u in untrainable_params)]

    in_loaded_not_model = [i for i in filtered_loaded if i not in filtered_model]
    in_model_not_loaded = [i for i in filtered_model if i not in filtered_loaded]

    if in_loaded_not_model or in_model_not_loaded:
        for _param in in_loaded_not_model:
            print(f"{_param} in loaded not in model.")
        for _param in in_model_not_loaded:
            print(f"{_param} in model not in loaded.")
        return False
    return True


def update_state_dict_for_eval(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Helper function to remove keys associated with loss function and remove 'model.' prefix from a loaded state dict.

    1. ignore criterion keys, 2. replace 'model.' prefix with ''.

    Parameters
    ----------
    state_dict : Dict[str, torch.Tensor]
        The original state dictionary of the model.

    Returns
    -------
    Dict[str, torch.Tensor]
        The updated state dictionary with removed criterion keys and 'model.' prefix.
    """
    updated_state_dict = {}
    for k, v in state_dict.items():
        if "criterion" not in k:  # don't need criterion for inference.
            if k.startswith("model."):
                k = k[len("model.") :]
            updated_state_dict[k] = v
        else:
            # uncomment print statement to see what w&b ignoring.
            # print(f"** not including: {k} **")
            pass
    state_dict = updated_state_dict

    return state_dict


def load_model_checkpoint_for_eval(
    model: nn.Module,
    ckpt_id: str,
    ckpt_base_path: str,
    ckpt_dir: str = "checkpoints",
    get_last: bool = False,
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """
    Wrapper function to load in a model for evaluation.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to load the checkpoint into.
    ckpt_id : str
        The checkpoint ID.
    ckpt_base_path : str
        The base path for checkpoints.
    ckpt_dir : str, optional
        The directory name for checkpoints (default is "checkpoints").
    get_last : bool, optional
        Whether to get the last checkpoint (default is False).
    device : Union[str, torch.device], optional
        The device to load the model onto (default is "cpu").

    Returns
    -------
    nn.Module
        The model with loaded checkpoint, ready for evaluation.
    """
    model.criterion = None  # set to None, don't need for inference.
    ckpt_path = get_model_ckpt_path(ckpt_id, ckpt_base_path, ckpt_dir, get_last)
    print(f"loading model checkpoint from {ckpt_path}.")
    state_dict = load_model_state_dict(ckpt_path, device)
    state_dict = update_state_dict_for_eval(state_dict)
    assert check_params(
        loaded_params=[k for k, _ in state_dict.items()],
        model_params=get_trainable_parameter_names(model),
    )
    reset_model_params(model)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_imerg_datamodule_from_config(
    cfg_base_path: str, cfg_name: str = "imerg_precipitation.yaml", overrides: Dict[str, Any] = {}
) -> IMERGPrecipitationDataModule:
    """
    Instantiate a IMERGPrecipitationDataModule() obj using a imerg_precipitation.yaml file and any specified overrides.

    Parameters
    ----------
    cfg_base_path : str
        The base path for the configuration file.
    cfg_name : str, optional
        The name of the configuration file (default is "imerg_precipitation.yaml").
    overrides : Dict[str, Any], optional
        Dictionary of configuration parameters to override (default is an empty dictionary).

    Returns
    -------
    IMERGPrecipitationDataModule
        An instance of IMERGPrecipitationDataModule initialized with the specified configuration.
    """
    _path = os.path.join(cfg_base_path, "datamodule", cfg_name)
    assert os.path.exists(_path), f"{_path} doesn't exist."
    datamodule_cfg = OmegaConf.load(Path(_path))
    datamodule_cfg.pop("_target_")  # drop the target key.
    # override any config params.
    if overrides:
        datamodule_cfg.update(**overrides)
    datamodule = IMERGPrecipitationDataModule(**datamodule_cfg)

    return datamodule


def load_model_checkpoint_config(ckpt_id: str, model_ckpt_cfg_base_path: str) -> DictConfig:
    """
    Load in a model checkpoint config file using OmegaConf.load().

    Parameters
    ----------
    ckpt_id : str
        The identifier for the checkpoint.
    model_ckpt_cfg_base_path : str
        The base path for the model checkpoint configuration.

    Returns
    -------
    DictConfig
        The loaded model checkpoint configuration.
    """
    _path = Path(os.path.join(model_ckpt_cfg_base_path, ckpt_id, "logs", "hparams.yaml"))
    assert os.path.exists(_path)
    model_ckpt_cfg = OmegaConf.load(_path)
    return model_ckpt_cfg


def instantiate_interpolator_experiment(
    model_ckpt_cfg: DictConfig, datamodule_cfg: DictConfig
) -> InterpolationExperiment:
    """
    Instantiate an interpolator experiment using model and datamodule configs.

    Parameters
    ----------
    model_ckpt_cfg : DictConfig
        The model checkpoint configuration.
    datamodule_cfg : DictConfig
        The datamodule configuration.

    Returns
    -------
    InterpolationExperiment
        An instance of InterpolationExperiment initialized with the specified configurations.
    """
    interpolation_exp = InterpolationExperiment(
        model_config=model_ckpt_cfg,
        datamodule_config=datamodule_cfg,
        diffusion_config=None,
    )
    return interpolation_exp


def instantiate_interpolator_model_from_base_config(
    cfg_base_path: str,
    datamodule_cfg_name: str = "imerg_precipitation.yaml",
    model_cfg_name: str = "unet_resnet.yaml",
    model_overrides: Optional[Dict] = None,
) -> Any:
    """
    Wrapper function to load an Interpolator Experiment from the base model config with OmegaConf.load()
    and return the underlying interpolator model.

    Parameters
    ----------
    cfg_base_path : str
        The base path for configuration files.
    datamodule_cfg_name : str, optional
        The name of the datamodule configuration file (default is "imerg_precipitation.yaml").
    model_cfg_name : str, optional
        The name of the model configuration file (default is "unet_resnet.yaml").
    model_overrides : Optional[Dict], optional
        Dictionary of model configuration overrides (default is None).

    Returns
    -------
    Any
        The instantiated model from the interpolator experiment.
    """
    # load in desired model config.
    model_cfg = OmegaConf.load(Path(os.path.join(cfg_base_path, "model", model_cfg_name)))

    # override model config.
    if model_overrides:
        model_cfg["model"].update({k: v for k, v in model_overrides.items() if k in model_cfg["model"]})

    # load in desired datamodule config.
    datamodule_cfg = OmegaConf.load(Path(os.path.join(cfg_base_path, "datamodule", datamodule_cfg_name)))

    # instantiate interpolation experiment.
    interpolation_exp = InterpolationExperiment(
        model_config=model_cfg["model"],
        datamodule_config=datamodule_cfg,
        diffusion_config=None,
    )
    model = interpolation_exp.model

    return model


def instantiate_interpolator_model_ckpt_from_config(
    ckpt_id: str,
    model_ckpt_cfg_base_path: str,
    cfg_base_path: str,
    model_cfg_name: str = "hparams.yaml",
    datamodule_cfg_name: str = "imerg_precipitation.yaml",
) -> Any:
    """
    Wrapper function to load an Interpolator Experiment from a supplied model config with OmegaConf.load()
    and return the underlying interpolator model.


    Parameters
    ----------
    ckpt_id : str
        The checkpoint ID.
    model_ckpt_cfg_base_path : str
        The base path for the model checkpoint configuration.
    cfg_base_path : str
        The base path for configuration files.
    model_cfg_name : str, optional
        The name of the model configuration file (default is "hparams.yaml").
    datamodule_cfg_name : str, optional
        The name of the datamodule configuration file (default is "imerg_precipitation.yaml").

    Returns
    -------
    Any
        The instantiated model from the interpolator experiment.
    """
    # load in configs.
    model_ckpt_cfg = OmegaConf.load(
        Path(os.path.join(model_ckpt_cfg_base_path, ckpt_id, "logs", model_cfg_name))
    )
    datamodule_cfg = OmegaConf.load(Path(os.path.join(cfg_base_path, "datamodule", datamodule_cfg_name)))
    # instantiate model.
    interpolation_exp = InterpolationExperiment(
        model_config=model_ckpt_cfg.model_config,
        datamodule_config=datamodule_cfg,
        diffusion_config=None,
    )
    model = interpolation_exp.model

    return model


def instantiate_interpolator_experiment_from_config(
    ckpt_id: str,
    model_ckpt_cfg_base_path: str,
    cfg_base_path: str,
    model_cfg_name: str = "hparams.yaml",
    datamodule_cfg_name: str = "imerg_precipitation.yaml",
) -> InterpolationExperiment:
    """
    Wrapper function to instantiate and return a InterpolationExperiment from a checkpoint.

    Parameters
    ----------
    ckpt_id : str
        The checkpoint ID.
    model_ckpt_cfg_base_path : str
        The base path for the model checkpoint configuration.
    cfg_base_path : str
        The base path for configuration files.
    model_cfg_name : str, optional
        The name of the model configuration file (default is "hparams.yaml").
    datamodule_cfg_name : str, optional
        The name of the datamodule configuration file (default is "imerg_precipitation.yaml").

    Returns
    -------
    InterpolationExperiment
        The instantiated interpolator experiment.
    """
    # load in configs.
    model_ckpt_cfg = OmegaConf.load(
        Path(os.path.join(model_ckpt_cfg_base_path, ckpt_id, "logs", model_cfg_name))
    )
    datamodule_cfg = OmegaConf.load(Path(os.path.join(cfg_base_path, "datamodule", datamodule_cfg_name)))
    # instantiate model.
    interpolation_exp = InterpolationExperiment(
        model_config=model_ckpt_cfg.model_config,
        datamodule_config=datamodule_cfg,
        diffusion_config=None,
    )

    return interpolation_exp
