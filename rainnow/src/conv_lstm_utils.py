"""A utilities module for the training and inference of a ConvLSTM.

For simplicity, all of the functions required to train and validate a ConvLSTM are in here.
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rainnow.src.dyffusion.datamodules.imerg_precipitation import IMERGPrecipitationDataModule
from rainnow.src.models.conv_lstm import ConvLSTMModel
from rainnow.src.utilities.utils import transform_0_1_to_minus1_1, transform_minus1_1_to_0_1


class IMERGDataset(Dataset):
    """
    A wrapper around the IMERGPrecipitationDataModule, extending it to torch.Dataset.

    The IMERGPrecipitationDataModule needs to have run it's class method .setup() to ensure
    that the _data_<split> methods contain the data.
    The data will inherit any normalisation from IMERGPrecipitationDataModule().

    Parameters
    ----------
    imerg_datamodule : IMERGPrecipitationDataModule
        The data module containing IMERG precipitation data.
    split : str
        The dataset split to use. Must be one of 'train', 'validate', 'test', or 'predict'.
    sequence_length : int
        The length of the input sequence.
    target_length : int
        The length of the target sequence.
    """

    def __init__(
        self,
        imerg_datamodule: IMERGPrecipitationDataModule,
        split: str,
        sequence_length: int,
        target_length: int,
        reverse_probability: float = 0.0,
    ):
        """Initialisation."""
        # inputs.
        self.datamodule = imerg_datamodule
        self.sequence_length = sequence_length
        self.target_length = target_length
        assert self.sequence_length >= self.target_length

        # data augmentation.
        self.reverse_probability = reverse_probability  # [0, 1].

        # handle splits.
        if split == "train":
            self.data = self.datamodule._data_train
        elif split == "validate":
            self.data = self.datamodule._data_val
        elif split == "test":
            self.data = self.datamodule._data_test
        elif split == "predict":
            self.data = self.datamodule._data_predict
        else:
            raise ValueError()

    @staticmethod
    def _reverse_sequence(sequence: Tensor, dim_to_flip: int):
        """Reverse a tensor sequence on a certain dimension to help regularise the model."""
        return torch.flip(sequence, dims=[dim_to_flip])

    # TODO: unit test this.
    def __getitem__(self, idx):
        sequence = self.data.__getitem__(idx)["dynamics"]

        if torch.rand(1) < self.reverse_probability:
            sequence = self._reverse_sequence(sequence=sequence, dim_to_flip=0)

        inputs = sequence[-(self.sequence_length + self.target_length) : -self.target_length, ...]
        target = sequence[-self.target_length :, ...]

        return inputs, target

    def __len__(self):
        return self.data.__len__()


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


def plot_training_val_loss(
    train_losses: List[float],
    val_losses: List[float],
    criterion_name: str,
    figsize: Dict[str, Any] = (8, 6),
) -> plt.Figure:
    """
    Plot the training and validation loss curves.

    Parameters
    ----------
    train_losses : list or array-like
        The training loss values for each epoch.
    val_losses : list or array-like
        The validation loss values for each epoch.
    criterion_name : str
        The name of the loss criterion (e.g., 'MSE', 'Cross Entropy').
    figsize : tuple, optional
        The figure size in inches (width, height). Default is (8, 6).

    Returns
    -------
    fig : plt.Figure
        The generated figure object containing the plot.
    """

    num_epochs = [i for i in range(len(train_losses))]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(num_epochs, train_losses, color="C0", label="train loss")
    ax.plot(num_epochs, val_losses, color="C1", label="val loss")
    ax.set_ylabel(f"{criterion_name}")
    ax.set_xlabel("epochs")
    ax.legend(loc="best")

    return fig


def plot_predicted_sequence(
    X: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    batch_num: int,
    plot_params: Dict[str, Any],
    figsize: Tuple[float, float],
) -> None:
    """
    Plot the input sequence, target sequence, and predicted sequence.

    This function creates a visualization of the input sequence, the target sequence,
    and the predicted sequence for a specific batch.

    Parameters
    ----------
    X : torch.Tensor
        The input sequence tensor of shape (batch_size, input_sequence_len, channels, height, width).
    target : torch.Tensor
        The target sequence tensor of shape (batch_size, target_sequence_len, channels, height, width).
    pred : torch.Tensor
        The predicted sequence tensor of shape (batch_size, target_sequence_len, channels, height, width).
    batch_num : int
        The index of the batch to plot.
    plot_params : Dict[str, Any]
        A dictionary of parameters to pass to the imshow function.
    figsize : Tuple[float, float]
        The figure size in inches (width, height).

    Returns
    -------
    None
    """
    input_sequence_len = X.size(1)
    target_sequence_len = target.size(1)
    nrows = 2
    fig, axs = plt.subplots(nrows=nrows, ncols=input_sequence_len + target_sequence_len, figsize=figsize)
    for i in range(input_sequence_len + target_sequence_len):
        if i < input_sequence_len:
            axs[0, i].imshow(X[batch_num, i, 0, :, :], **plot_params)
        else:
            axs[0, i].imshow(target[batch_num, i - input_sequence_len, 0, :, :], **plot_params)
            axs[1, i].imshow(pred[batch_num, i - input_sequence_len, 0, :, :], **plot_params)

    for ax in axs.flatten():
        ax.axis("off")

    plt.tight_layout()


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
            break  # Exit the loop when the iterator is exhausted
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
