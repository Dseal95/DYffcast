"""Torch Datasets."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from rainnow.src.dyffusion.datamodules.imerg_precipitation import IMERGPrecipitationDataModule


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