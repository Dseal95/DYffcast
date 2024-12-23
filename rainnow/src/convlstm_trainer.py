"""A train/eval utilities module for the training and inference of a ConvLSTM."""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rainnow.src.dyffusion.datamodules.imerg_precipitation import IMERGPrecipitationDataModule
from rainnow.src.models.conv_lstm import ConvLSTMModel
from rainnow.src.utilities.utils import transform_0_1_to_minus1_1, transform_minus1_1_to_0_1


def train(
    model: nn.Module,
    device: str,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    tepoch: tqdm,
    curr_epoch: int,
    n_epochs: int,
    log_training: bool = True,
) -> float:
    """Train the (ConvLSTM) model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    device : str
        The device to run the model on, either 'cpu' or 'cuda'.
    data_loader : DataLoader
        The DataLoader for providing the training data.
    optimizer : optim.Optimizer
        The optimizer used for updating the weights.
    criterion: nn.Module
        The loss criterion used for training.
    tepoch : tqdm
        The tqdm object for progress display.
    curr_epoch : int
        The current epoch number.
    n_epochs : int
        The total number of epochs to train.
    log_training : bool, optional
        Flag to control the display of training progress, by default True.

    Returns
    -------
    float
        The average training loss for the epoch.
    """
    model.train()
    train_loss = 0.0
    for batch_idx, (X, y) in enumerate(data_loader, start=1):
        X, y = X.to(device), y.to(device)

        # transform y to X range.
        if isinstance(model.output_activation, nn.Tanh):
            y = transform_0_1_to_minus1_1(y)

        optimizer.zero_grad()
        pred = model(X)

        if criterion.__class__.__name__ == "LPIPSMSELoss":
            # for LPIPS, need to get it in form N, C, H, W.
            loss = criterion(pred.reshape(-1, 1, 128, 128), y.reshape(-1, 1, 128, 128))
        else:
            loss = criterion(pred, y)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if log_training:
            tepoch.set_description(
                f"Epoch: {curr_epoch}/{n_epochs} | Batch: {batch_idx}/{len(data_loader)}"
            )
            tepoch.set_postfix(loss=loss.item() / X.size(0), refresh=False)
            tepoch.update()

    return train_loss / len(data_loader.dataset)


def validate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    """Validate the (ConvLSTM) model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to validate.
    data_loader : DataLoader
        The DataLoader for providing the training data.
    criterion: nn.Module
        The loss criterion used for training.
    device : str
        The device to run the model on, either 'cpu' or 'cuda'.

    Returns
    -------
    float
        The average validation loss for the epoch.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            # transform y to X range.
            if isinstance(model.output_activation, nn.Tanh):
                y = transform_0_1_to_minus1_1(y)

            pred = model(X)

            if criterion.__class__.__name__ == "LPIPSMSELoss":
                # for LPIPS, need to get it in form N, C, H, W.
                loss = criterion(pred.reshape(-1, 1, 128, 128), y.reshape(-1, 1, 128, 128))
            else:
                loss = criterion(pred, y)

            val_loss += loss.item()

    return val_loss / len(data_loader.dataset)


def save_checkpoint(
    model_save_path: str,
    model_weights: dict,
    optimizer_info: dict,
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> None:
    """
    Save the model checkpoint.

    Parameters
    ----------
    model_save_path : str
        The file path to save the model checkpoint.
    model_weights : dict
        The state dictionary of the model.
    optimizer_info : dict
        The state dictionary of the optimizer.
    epoch : int
        The current epoch number.
    train_loss : float
        The training loss at the current epoch.
    val_loss : float
        The validation loss at the current epoch.

    Returns
    -------
    None
    """
    torch.save(
        {
            "model_state_dict": model_weights,
            "optimizer_state_dict": optimizer_info,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        model_save_path,
    )


def create_eval_loader(
    data_loader: DataLoader,
    horizon: int = 8,
    input_sequence_length: int = 4,
    img_dims: Tuple[int, int] = (128, 128),
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Create a custom dataloader containing input:target samples with an overlap of input_sequence_length.

    Using the input_sequence_length = 4 and horizon = 8 as an example:
       - input=4 -  ------ target=8 ------
    1. [x][x][x][x]|[y][y][y][y][y][y][y][y]
    2               [x][x][x][x]|[y][y][y][y][y][y][y][y]
    3.                           [x][x][x][x]|[y][y][y][y][y][y][y][y]

    Parameters
    ----------
    data_loader : DataLoader
        The original DataLoader containing the dataset.
    horizon : int, optional
        The number of future time steps to predict (default is 8).
    input_sequence_length : int, optional
        The number of input time steps used for each prediction (default is 4).

    Returns
    -------
    tuple
        A tuple containing two lists:
        - eval_loader : list
            List of tuples (input, target) where each input has shape (input_sequence_length, 1, 128, 128)
            and each target has shape (horizon, 1, 128, 128).
        - uneligible : list
            List of tuples (input, target) that didn't meet the horizon length requirement.
    """
    # exhaustively get all samples (including input and target) from the dataloader.
    samples = []
    iter_loader = iter(data_loader)  # Assuming 'loader' is your DataLoader
    while True:
        try:
            X_next, target_next = next(iter_loader)
            samples.append(torch.cat([X_next, target_next], dim=1))
        except StopIteration:
            break  # Exit the loop when the iterator is exhausted.

    # this contains all HxW images in the dataloader.
    unravel_loader = torch.cat([i.view(-1, 1, *img_dims) for i in samples], dim=0)

    # create input_sequence_length:horizon, input:target pairs, keeping only the complete pairs (eligible).
    eval_loader, uneligible = [], []
    for i in range((unravel_loader.size(0) - input_sequence_length) // input_sequence_length):
        seq = unravel_loader[
            i * input_sequence_length : (i * input_sequence_length) + horizon + input_sequence_length,
            ...,
        ]
        _input, _target = seq[:input_sequence_length, ...], seq[input_sequence_length:, ...]
        if _target.size(0) == horizon:
            # only keep complete sequences.
            eval_loader.append((_input, _target))
        else:
            uneligible.append((_input, _target))

    # check that the loader was made properly.
    # the first input_sequence_length of images in the target should
    # equal the input images. The samples are made with no input
    # sequence overlap.
    prev_target = None
    for X, target in eval_loader:
        if prev_target is not None:
            assert torch.allclose(prev_target[:input_sequence_length, ...], X)
        prev_target = target

    dims = list(set([torch.cat([a, b], dim=0).size() for a, b in eval_loader]))[0]
    assert dims[0] == horizon + input_sequence_length

    # debug info.
    print("** eval loader (INFO) **")
    print(f"Num samples = {len(eval_loader)} w/ dims: {dims}\n")

    return eval_loader, uneligible


def generate_sequence_conv_lstm(
    model: ConvLSTMModel, inputs: torch.Tensor, device: str, horizon: int
) -> Dict[str, torch.Tensor]:
    """
    Generate a sequence using a ConvLSTM model.

    This function performs a roll-out prediction using the provided ConvLSTM model
    for the specified horizon length.

    Parameters
    ----------
    model : ConvLSTMModel
        The ConvLSTM model to use for prediction.
    inputs : torch.Tensor
        The input tensor to start the prediction from.
    device : str
        The device to run the model on (e.g., 'cuda' or 'cpu').
    horizon : int
        The number of time steps to predict into the future.

    Dict[str, torch.Tensor]
        A dictionary containing the predictions for each time step. The keys are in the format
        't{step}_preds' and the values are the corresponding prediction tensors.
    """
    predictions = {}
    model, inputs = model.to(device), inputs.to(device)
    # roll-out.
    for t in range(horizon):
        prediction = model(inputs)
        if isinstance(model.output_activation, nn.Tanh):
            prediction = transform_minus1_1_to_0_1(prediction)
        predictions[f"t{t+1}_preds"] = prediction.squeeze(0)
        # update the inputs with the last pred.
        inputs = torch.concat([inputs[:, 1:, ...], prediction], dim=1)
    return predictions
