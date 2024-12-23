"""Utility module containing utility functions used throughout package."""

import argparse
import datetime
import logging
import math
import re
import secrets
import string
from inspect import isfunction
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from tensordict import TensorDict
from torch import Tensor

distribution_params_to_edit = ["loc", "scale"]


def get_device() -> str:
    """
    Return the device (GPU or CPU) that the user's environment is running on/in.

    Returns
    -------
    str
        A string indicating the device:
        - 'cuda' if a CUDA-capable GPU is available
        - 'cpu' if no GPU is available or CUDA is not installed
    """
    device = "cpu"
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        device = "cuda"
        print(f"Cuda installed! Running on GPU! (device = {device})")
    else:
        print(f"No GPU available! (device = {device})")
    return device


def str_input_to_bool(input: str) -> bool:
    """
    Convert a string input to a boolean type.

    This is a helper function for argparser when running .py files from the terminal.

    The string input list is NOT exhaustive.

    Parameters
    ----------
    input : str
        The input string to convert to boolean.

    Returns
    -------
    bool
        The boolean value corresponding to the input string.

    Raises
    ------
    argparse.ArgumentTypeError
        If the input string cannot be converted to a boolean value.
    """
    if isinstance(input, bool):
        return input
    if input.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif input.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected.")


# TODO: unit test this.
def calc_avg_value_2d_arr(arr) -> float:
    """
    Calculate the average value of a 2D numpy array.

    Parameters
    ----------
    arr : np.ndarray
        The 2D numpy array to calculate the average from.

    Returns
    -------
    float
        The average value of the input array.
    """
    return np.mean(arr)


def get_logger(log_file_name: str = None, name=__name__, level=logging.INFO) -> logging.Logger:
    """
    Initialize a Python logger.

    Parameters
    ----------
    log_file_name : str, optional
        The name of the log file. If provided, a FileHandler will be added to the logger.
    name : str, optional
        The name of the logger. Default is the name of the current module.
    level : int, optional
        The logging level. Default is logging.INFO.

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # set desired log formats.
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] --> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # handlers.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # if log_file_name is provided, add file handler.
    if log_file_name is not None:
        file_handler = logging.FileHandler(f"{log_file_name}.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# TODO: unit test this function.
def get_year_month_day_hour_min_from_datetime(
    dt: datetime.datetime, zero_padding: int
) -> Tuple[str, str, str, str, str]:
    """
    Extract year, month, day, hour, and minute from a datetime object as strings.

    The zero_padding is added for suitability for IMERG data.

    Parameters
    ----------
    dt : datetime.datetime
        The datetime object to extract information from.
    zero_padding : int
        The number of digits to zero-pad the month, day, hour, and minute.

    Returns
    -------
    Tuple[str, str, str, str, str]
        A tuple containing the year, month, day, hour, and minute as strings.
        The month, day, hour, and minute are zero-padded according to the zero_padding parameter.
    """
    year = str(dt.year)
    month = str(dt.month).zfill(zero_padding)
    day = str(dt.day).zfill(zero_padding)
    hour = str(dt.hour).zfill(zero_padding)
    min = str(dt.minute).zfill(zero_padding)

    return year, month, day, hour, min


def calculate_required_1d_padding(X: int, Y: int, frac: int) -> int:
    """
    Calculate the padding needed to X to fit Y into X, (X/Y) rounded to 1/frac times.

    For example, if X = 10, Y = 3 and frac = 2, (X/Y) is 3.3333 which is 3.5 to the nearest (1/2)
    Therfore, to get 3.5*3 (= 10.5) into X we need to pad X by 1.

    Parameters
    ----------
    X : int
        The total dimension of the dataset along one axis.
    Y : int
        The desired patch size along the same axis.
    frac : int
        The denominator to determine the fraction for rounding.

    Returns
    -------
    int
        The additional padding needed to make the dimension X divisible by Y to the nearest 1/frac.
    """
    frac = 1 if frac <= 1 else frac  # handle 0 division.
    quotient = X / Y
    rounded_quotient = math.ceil(quotient * frac) / frac
    padding_needed = (rounded_quotient - quotient) * Y

    return int(np.ceil(padding_needed))


# TODO: unit test this function.
def extract_imerg_hdf5_hhmm_ref_sort_key(file_name: str) -> int:
    """
    Extracts the time, hh:mm reference from the IMERG HDF5 file name to be used as a sort key.

    Parameters
    ----------
    file_name : str
        The name of the file from which to extract the hhmm reference.

    Returns
    -------
    int
        The hhmm reference as an integer if matched, otherwise 0.
    """
    match = re.search(r"\.(\d{4})\.V\d{2}[A-Z]\.HDF5$", file_name)
    if match:
        return int(match.group(1))
    return 0


# TODO: unit test this function.
def generate_alphanumeric_id(length=8) -> str:
    """
    Generate a random alphanumeric ID of specified length.

    Parameters
    ----------
    length : int, optional
        The length of the ID to generate. Default is 8.

    Returns
    -------
    str
        A random alphanumeric ID.
    """
    characters = string.ascii_letters + string.digits
    return "".join(secrets.choice(characters) for _ in range(length))


# TODO: unit test this.
def transform_0_1_to_minus1_1(X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """transform / scale data X from [0, 1] to [-1, 1]."""
    return 2 * X - 1


# TODO: unit test this.
def transform_minus1_1_to_0_1(X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform/scale data X from [-1, 1] to [0, 1].

    Parameters
    ----------
    X : Union[np.ndarray, torch.Tensor]
        Input data in the range [-1, 1].

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Transformed data in the range [0, 1].
    """
    return (X + 1) / 2


def get_trainable_parameter_names(model: Any) -> list[torch.Tensor]:
    """
    Get the names of trainable parameters in the model.

    Parameters
    ----------
    model : Any
        The model to inspect for trainable parameters.

    Returns
    -------
    list[torch.Tensor]
        A list of names of trainable parameters.
    """
    return [name for name, param in model.named_parameters() if param.requires_grad]


def freeze_model(model: nn.Module) -> nn.Module:
    """
    Freeze a torch.nn.Module's parameters.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to freeze.

    Returns
    -------
    nn.Module
        The frozen model.
    """
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  # set to eval mode.
    return model


def generate_data_driven_gaussian_noise(
    batch: torch.Tensor, lam: float = 0.5, std_factor: float = 0.15, data_lims: List[float] = [0, 1]
) -> torch.Tensor:
    """
    Generate data-driven Gaussian noise using batch statistics.
    This function creates noise based on the mean and standard deviation of the input batch,
    allowing for a weighted combination of batch-wise and element-wise statistics.
    Parameters
    ----------
    batch : torch.Tensor
        Input tensor for which to generate noise.
    lam : float, optional
        Weighting factor for combining batch data points with batch mean (default is 0.5).
        Higher values give more weight to individual data points.
    std_factor : float, optional
        Scaling factor controlling the overall magnitude of the standard deviation (default is 0.15).
    data_lims : List[float], optional
        Lower and upper limits for clamping the generated noise (default is [0, 1]).
    Returns
    -------
    torch.Tensor
        Generated noise tensor of the same shape as the input batch,
        with values clamped to the specified data limits.
    """
    batch_mean = batch.mean()
    batch_std = batch.std()
    adjusted_mean = lam * batch + (1 - lam) * batch_mean
    adjusted_std = std_factor * (batch * batch_std + (1 - batch) * (1 - batch_mean))
    noise = torch.clamp(
        torch.normal(mean=adjusted_mean, std=adjusted_std), min=data_lims[0], max=data_lims[1]
    )
    return noise


def get_datamodule(config: DictConfig):
    """
    Instantiate a PyTorch Lightning datamodule with a config.

    Parameters
    ----------
    config : DictConfig
        Configuration object containing datamodule and model settings.

    Returns
    -------
    pl.LightningDataModule
        An instantiated PyTorch Lightning datamodule.
    """
    datamodule = hydra.utils.instantiate(
        config.datamodule,
        _recursive_=False,
        model_config=config.model,  # kwargs
    )
    return datamodule


def get_module(config: DictConfig, **kwargs):
    """
    Instantiate a PyTorch Lightning module with a config.

    Parameters
    ----------
    config : DictConfig
        Configuration object containing module, model, datamodule, and diffusion settings.
    **kwargs : dict
        Additional keyword arguments to pass to the module instantiation.

    Returns
    -------
    pl.LightningModule
        An instantiated PyTorch Lightning module.
    """
    model = hydra.utils.instantiate(
        config.module,
        model_config=config.model,
        datamodule_config=config.datamodule,
        # TODO: find a way to avoid the nested 'diffusion' key on instantiation.
        diffusion_config=(
            config.get("diffusion", default_value=None)["diffusion"]
            if config.get("diffusion", default_value=None)
            else None
        ),
        _recursive_=False,
        **kwargs,
    )
    return model



# ** all the functions below are taken directly from: https://github.com/Rose-STL-Lab/dyffusion/tree/main/src/utilities **
def raise_error_if_invalid_value(value: Any, possible_values: Sequence[Any], name: str = None):
    """Raises an error if the given value (optionally named by `name`) is not one of the possible values."""
    if value not in possible_values:
        name = name or (value.__name__ if hasattr(value, "__name__") else "value")
        raise ValueError(f"{name} must be one of {possible_values}, but was {value}")
    return value


def raise_error_if_has_attr_with_invalid_value(obj: Any, attr: str, possible_values: Sequence[Any]):
    if hasattr(obj, attr):
        raise_error_if_invalid_value(
            getattr(obj, attr), possible_values, name=f"{obj.__class__.__name__}.{attr}"
        )


def raise_error_if_invalid_type(value: Any, possible_types: Sequence[Any], name: str = None):
    """Raises an error if the given value (optionally named by `name`) is not one of the possible types."""
    if all([not isinstance(value, t) for t in possible_types]):
        name = name or (value.__name__ if hasattr(value, "__name__") else "value")
        raise ValueError(
            f"{name} must be an instance of either of {possible_types}, but was {type(value)}"
        )
    return value


def raise_if_invalid_shape(
    value: Union[np.ndarray, torch.Tensor],
    expected_shape: Sequence[int] | int,
    axis: int = None,
    name: str = None,
):
    if isinstance(expected_shape, int):
        if value.shape[axis] != expected_shape:
            name = name or (value.__name__ if hasattr(value, "__name__") else "value")
            raise ValueError(
                f"{name} must have shape {expected_shape} along axis {axis}, but shape={value.shape}"
            )
    else:
        if value.shape != expected_shape:
            name = name or (value.__name__ if hasattr(value, "__name__") else "value")
            raise ValueError(f"{name} must have shape {expected_shape}, but was {value.shape}")


def exists(x):
    return x is not None


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def enable_inference_dropout(model: nn.Module):
    """Set all dropout layers to training mode"""
    # find all dropout layers
    dropout_layers = [
        m for m in model.modules() if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d)
    ]
    for layer in dropout_layers:
        layer.train()
    # assert all([layer.training for layer in [m for m in model.modules() if isinstance(m, nn.Dropout)]])


def disable_inference_dropout(model: nn.Module):
    """Set all dropout layers to eval mode"""
    # find all dropout layers
    dropout_layers = [
        m for m in model.modules() if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d)
    ]
    for layer in dropout_layers:
        layer.eval()


def rrearrange(
    data: Union[Tensor, torch.distributions.Distribution, TensorDict],
    pattern: str,
    **axes_lengths,
):
    """Extend einops.rearrange to work with distributions."""
    if torch.is_tensor(data) or isinstance(data, np.ndarray):
        return rearrange(data, pattern, **axes_lengths)
    elif isinstance(data, torch.distributions.Distribution):
        dist_params = {
            k: rearrange(getattr(data, k), pattern, **axes_lengths)
            for k in distribution_params_to_edit
            if hasattr(data, k)
        }
        return type(data)(**dist_params)
    elif isinstance(data, TensorDict):
        new_data = {k: rrearrange(v, pattern, **axes_lengths) for k, v in data.items()}
        return TensorDict(new_data, batch_size=new_data[list(new_data.keys())[0]].shape)
    elif isinstance(data, dict):
        return {k: rrearrange(v, pattern, **axes_lengths) for k, v in data.items()}
    else:
        raise ValueError(f"Cannot rearrange {type(data)}")


def to_DictConfig(obj: Optional[Union[List, Dict]]):
    """Tries to convert the given object to a DictConfig."""
    if isinstance(obj, DictConfig):
        return obj

    if isinstance(obj, list):
        try:
            dict_config = OmegaConf.from_dotlist(obj)
        except ValueError:
            dict_config = OmegaConf.create(obj)

    elif isinstance(obj, dict):
        dict_config = OmegaConf.create(obj)

    else:
        dict_config = OmegaConf.create()  # empty

    return dict_config


def torch_to_numpy(x: Union[Tensor, Dict[str, Tensor]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, TensorDict):
        return {k: v.detach().cpu().numpy() for k, v in x.items()}
        # return x.detach().cpu()   # numpy() not implemented for TensorDict
    elif isinstance(x, dict):
        return {k: torch_to_numpy(v) for k, v in x.items()}
    elif isinstance(x, torch.distributions.Distribution):
        # only move the parameters to cpu
        for k in distribution_params_to_edit:
            if hasattr(x, k):
                setattr(x, k, getattr(x, k).detach().cpu())
        return x
    else:
        return x


def torch_select(input: Tensor, dim: int, index: int):
    """Extends torch.select to work with distributions."""
    if isinstance(input, torch.distributions.Distribution):
        dist_params = {
            k: torch.select(getattr(input, k), dim, index)
            for k in distribution_params_to_edit
            if hasattr(input, k)
        }
        return type(input)(**dist_params)
    else:
        return torch.select(input, dim, index)


@rank_zero_only
def print_config(
    config,
    fields: Union[str, Sequence[str]] = (
        "model",
        "diffusion",
        "datamodule",
        "module",
        "trainer",
        "callbacks",
        "seed",
        "work_dir",
    ),
    resolve: bool = True,
    rich_style: str = "magenta",
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure (if installed: ``pip install rich``).

    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Args:
        config (ConfigDict): Configuration
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        rich_style (str, optional): Style of Rich library to use for printing. E.g "magenta", "bold", "italic", etc.
    """
    import importlib

    if not importlib.util.find_spec("rich") or not importlib.util.find_spec("omegaconf"):
        # no pretty printing
        print(OmegaConf.to_yaml(config, resolve=resolve))
        return
    import rich.syntax  # IMPORTANT to have, otherwise errors are thrown
    import rich.tree

    tree = rich.tree.Tree(":gear: CONFIG", style=rich_style, guide_style=rich_style)
    if isinstance(fields, str):
        if fields.lower() == "all":
            fields = config.keys()
        else:
            fields = [fields]

    for field in fields:
        branch = tree.add(field, style=rich_style, guide_style=rich_style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)
