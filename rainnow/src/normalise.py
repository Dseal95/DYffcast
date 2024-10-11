"""Module containing preprocessing object."""

from typing import List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from rainnow.src.utilities.utils import get_logger

log = get_logger(log_file_name=None, name=__name__)

DataType = Union[Tensor, np.ndarray]


# TODO: unit test this class.
class PreProcess:
    """
    A class for preprocessing data through linear transformation, log scaling, and normalization.
    All of the preprocessing is reversible via class methods.

    This class implements a three-step preprocessing pipeline:
    1. Outlier removal via linear transformation based on data percentiles.
    2. Logarithmic transformation to widen data distribution.
    3. Min-max normalization to scale data to the [0, 1] range.

    The class supports both PyTorch Tensors and NumPy arrays as input.

    Parameters
    ----------
    percentiles : List[float]
        A list of two floats representing the lower and upper percentiles
        for the linear transformation step. Typically [1.0, 99.0].
    minmax : List[float]
        A list of two floats representing the minimum and maximum values
        for the final normalization step.
    """

    def __init__(self, percentiles: Optional[List[float]] = None, minmax: Optional[List[float]] = None):
        """Initialisation."""
        self.percentiles: Optional[List[float]] = percentiles
        self.minmax: Optional[List[float]] = minmax
        self.checked = False
        self.check_params()

        if self.checked:
            log.info(
                f"pprocessing w/ percentiles (1st, 99th): {self.percentiles},  (min, max): {self.minmax}"
            )

    def check_params(self) -> None:
        """Check the inputs. They all need to be lists of 2 floating point elements.
        Those elements may also need to be sequential.
        """

        def _check(data: List[float], _list_sequential: bool = False) -> None:
            assert isinstance(data, list)
            assert len(data) == 2
            assert all(isinstance(p, float) for p in data)
            if _list_sequential:
                assert data[1] > data[0]

        if self.percentiles:
            _check(self.percentiles, True)
        if self.minmax:
            _check(self.minmax, True)
        self.checked = True

    @staticmethod
    def log_linear_norm(data: DataType, c: float, d: float, a: float = 0, b: float = 1) -> DataType:
        """
        Apply logarithmic transformation (log +1) to the linear norm of a tensor/array. The linear normalisation is:

        data_lin_norm = (data_in - c) * (b-a) / (d-c) + a

        where a and b are the lower and upper limits of the
        resulting range and c and d are the lower and upper
        values of the input range. By default, the resulting range of
        values [a, b] is chosen to be between 0 and 1.

        arr_log_lin_norm = np.log(arr_lin_norm + 1)

        Parameters
        ----------
        data : DataType
            Input data, either a NumPy array or a PyTorch tensor.
        c : float
            The lower bound of the input range.
        d : float
            The upper bound of the input range.
        a : float, optional
            The lower bound of the output range. Defaults to 0.
        b : float, optional
            The upper bound of the output range. Defaults to 1.

        Returns
        -------
        DataType
            The transformed data with a logarithmic scale applied after linear normalization.
        """
        scale = (b - a) / (d - c)
        if isinstance(data, Tensor):
            return torch.log1p(((data - c) * scale) + a)
        return np.log1p(((data - c) * scale) + a)

    @staticmethod
    def min_max_norm(data: DataType, _min: float, _max: float) -> DataType:
        """Apply min-max normalization."""
        return (data - _min) / (_max - _min)

    @staticmethod
    def reverse_log_linear_norm(
        data: DataType, c: float, d: float, a: float = 0, b: float = 1
    ) -> DataType:
        """
        Reverse the log_linear_normalisation.
        arr_in = (exp(arr_out - 1) - a).(d-c / b-a) + c)
        """
        scale = (d - c) / (b - a)
        if isinstance(data, Tensor):
            return ((torch.expm1(data) - a) * scale) + c
        return ((np.expm1(data) - a) * scale) + c

    @staticmethod
    def reverse_min_max_norm(data: DataType, _min: float, _max: float) -> DataType:
        """Reverse min-max normalization."""
        return (data * (_max - _min)) + _min

    def apply_preprocessing(self, data: DataType) -> DataType:
        """Wrapper function to apply the preprocessing pipeline."""
        assert self.percentiles is not None and self.minmax is not None
        processed_data = self.min_max_norm(
            data=self.log_linear_norm(data=data, c=self.percentiles[0], d=self.percentiles[1], a=0, b=1),
            _min=self.minmax[0],
            _max=self.minmax[1],
        )
        return processed_data

    def reverse_processing(self, data: DataType) -> DataType:
        """Wrapper function to reverse the preprocessing pipeline."""
        assert self.percentiles is not None and self.minmax is not None
        reversed_data = self.reverse_log_linear_norm(
            data=self.reverse_min_max_norm(data=data, _min=self.minmax[0], _max=self.minmax[1]),
            c=self.percentiles[0],
            d=self.percentiles[1],
            a=0,
            b=1,
        )
        return reversed_data
