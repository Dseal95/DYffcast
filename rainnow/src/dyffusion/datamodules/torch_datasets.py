"""Code adapted from: https://github.com/Rose-STL-Lab/dyffusion/blob/main/src/datamodules/torch_datasets.py."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from rainnow.src.normalise import PreProcess
from rainnow.src.utilities.utils import get_logger

log = get_logger(name=__name__)


class MyTensorDataset(Dataset[Dict[str, Tensor]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    tensors: Dict[str, Tensor]

    @staticmethod
    def convert_to_tensor(item):
        """Single job function to allow multi-threading."""
        key, tensor = item
        if isinstance(tensor, np.ndarray):
            return key, torch.from_numpy(tensor).float()
        return key, tensor

    @staticmethod
    def _validate_tensors(tensors, dataset_size):
        """Check that the number of tensors == dataset size."""
        for k, value in tensors.items():
            if torch.is_tensor(value):
                assert value.size(0) == dataset_size, f"Size mismatch for tensor {k}"
            elif isinstance(value, Sequence):
                assert (
                    len(value) == dataset_size
                ), f"Size mismatch between list ``{k}`` of length {len(value)} and tensors {dataset_size}"
            else:
                raise TypeError(f"Invalid type for tensor {k}: {type(value)}")

    def __init__(
        self,
        tensors: Dict[str, Tensor] | Dict[str, np.ndarray],
        dataset_id: str = "",
        **kwargs,
    ):
        """initialisation."""
        log.info(f"creating {dataset_id.upper()} tensor dataset.")

        self.dataset_id = dataset_id

        # pre-processing / normalisation info.
        self.normalize = kwargs.get("norm", False)
        self.train_percentiles = kwargs.get("percentiles", None)
        self.minmax = kwargs.get("min_max", None)
        self.pprocessor = None

        # training data augmentation.
        if self.dataset_id == "train":
            # defaults to 0.0, reverse no sequences.
            self.reverse_probability = (
                kwargs.get("reverse_sequences_prob", 0.0) or 0.0
            )  # handle possible None value.
            if self.reverse_probability > 0.0:
                log.info(
                    f"Data Augmentation: Reversing {self.reverse_probability*100}% random sequences."
                )
        else:
            # only reverse training data.
            self.reverse_probability = 0.0

        with ThreadPoolExecutor() as executor:
            # create a {key: tensor} from {key: numpy} dict.
            tensors = dict(executor.map(self.convert_to_tensor, tensors.items()))

        any_tensor = next(iter(tensors.values()))
        self.dataset_size = any_tensor.size(0)
        self._validate_tensors(tensors, self.dataset_size)

        self.tensors = tensors
        self.dataset_id = dataset_id

        if self.normalize:
            self.pprocessor = PreProcess(percentiles=self.train_percentiles, minmax=self.minmax)
            self._normalize_tensors()

    @staticmethod
    def _reverse_sequence(sequence: Tensor, dim_to_flip: int):
        """Reverse a tensor sequence on a certain dimension to help regularise the model."""
        return torch.flip(sequence, dims=[dim_to_flip])

    def _normalize_tensors(self):
        """Normalise the data using the self.pprocessor object."""
        # TODO: maybe do this on the fly for memory.
        for key, tensor in self.tensors.items():
            self.tensors[key] = self.pprocessor.apply_preprocessing(tensor)

    def __getitem__(self, index):
        sample = {key: tensor[index] for key, tensor in self.tensors.items()}

        # TODO: this is harcoded to only work for the 'dynamics' key.
        # data augmentation (only applies to training data if set).
        if torch.rand(1) < self.reverse_probability:
            sample["dynamics"] = self._reverse_sequence(sequence=sample["dynamics"], dim_to_flip=0)

        return sample

    def __len__(self):
        return self.dataset_size


def get_tensor_dataset_from_numpy(*ndarrays, dataset_id="", dataset_class=MyTensorDataset, **kwargs):
    tensors = [torch.from_numpy(ndarray.copy()).float() for ndarray in ndarrays]
    return dataset_class(*tensors, dataset_id=dataset_id, **kwargs)
