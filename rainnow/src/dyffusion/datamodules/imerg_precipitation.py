"""Datamodule for IMERG Precipitation data.

This datamodule follows the same structure the datamodules in DYffusion.
See: https://github.com/Rose-STL-Lab/dyffusion/tree/main/src/datamodules.
"""

from __future__ import annotations

import datetime
import os
from os.path import join
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import dask
import numpy as np
import xarray as xr
from omegaconf import OmegaConf

from rainnow.src.dyffusion.configs.imerg_data_config import IMERGEarlyRunDataConfig
from rainnow.src.dyffusion.datamodules.abstract_datamodule import BaseDataModule
from rainnow.src.dyffusion.datamodules.torch_datasets import MyTensorDataset
from rainnow.src.utilities.utils import (
    get_logger,
    raise_error_if_invalid_type,
    raise_error_if_invalid_value,
)

log = get_logger(log_file_name=None, name=__name__)


class IMERGPrecipitationDataModule(BaseDataModule):
    """
    Data module for IMERG precipitation data.

    Parameters
    ----------
    data_dir : str
        Directory containing the IMERG precipitation data.
    boxes : Union[List, str], optional
        Specifies which boxes to use. Can be a list of box indices or "all". Default is "all".
    box_size : int, optional
        Size of each box. Default is 128.
    sequence_dt : int, optional
        Time step between sequences. Default is 1.
    window : int, optional
        Number of time steps to use as input. Default is 1.
    horizon : int, optional
        Number of time steps to predict. Default is 1.
    prediction_horizon : int, optional
        Specific prediction horizon. If None, uses the value of horizon. Default is None.
    multi_horizon : bool, optional
        If True, predicts multiple horizons. Default is False.
    normalization : Dict[str, Union[bool, List[float]]], optional
        Normalization parameters for the data. Default is None.
    data_splits : Dict[str, List[Optional[str]]], optional
        Specifies how to split the data into train, validation, and test sets. Default is an empty dictionary.
    """

    def __init__(
        self,
        data_dir: str,
        boxes: Union[List, str] = "all",
        box_size: int = 128,
        sequence_dt: int = 1,
        window: int = 1,
        horizon: int = 1,
        prediction_horizon: int = None,  # None means use horizon.
        multi_horizon: bool = False,
        normalization: Dict[str, Union[bool, List[float]]] = None,
        data_augmentation: Dict[str, Union[bool, List[float]]] = None,
        data_splits: Dict[str, List[Optional[str]]] = {},
        **kwargs,
    ):
        """initialisation."""
        raise_error_if_invalid_type(data_dir, possible_types=[str], name="data_dir")
        raise_error_if_invalid_value(box_size, [128], name="box_size")

        if "imerg-precipitation" not in data_dir:
            for name in ["imerg-precipitation"]:
                if os.path.isdir(join(data_dir, name)):
                    data_dir = join(data_dir, name)
                    break

        super().__init__(data_dir=data_dir, **kwargs)
        self.save_hyperparameters()

        # data pre-processing info.
        self.normalization_hparams = OmegaConf.to_container(normalization) if normalization else {}
        self.data_augmentation_hparams = (
            OmegaConf.to_container(data_augmentation) if data_augmentation else {}
        )
        self.dataset_kwargs = {**self.normalization_hparams, **self.data_augmentation_hparams}

        # 'train', 'val', 'test' and 'predict' data splits.
        self.train_lims = data_splits.get("train")
        self.val_lims = data_splits.get("val")
        self.test_lims = data_splits.get("test")
        self.predict_lims = data_splits.get("predict", [])  # prediction slice can be none.

        # error handling.
        s = "dates must be specified in imerg_precipitation.yaml"
        assert self.train_lims is not None, f"training {s}"
        assert self.val_lims is not None, f"val {s}"
        assert self.test_lims is not None, f"test {s}"

        # convert lims to index key (specific to IMERG).
        self.train_index = [self.convert_str_dt_to_imerg_int_dt(t) for t in self.train_lims]
        self.val_index = [self.convert_str_dt_to_imerg_int_dt(t) for t in self.val_lims]
        self.test_index = [self.convert_str_dt_to_imerg_int_dt(t) for t in self.test_lims]
        self.predict_index = [self.convert_str_dt_to_imerg_int_dt(t) for t in self.predict_lims]

        # set the temporal slices for the train, val, test and predict sets.
        self.train_slice = slice(*self.train_index)
        self.val_slice = slice(*self.val_index)
        self.test_slice = slice(*self.test_index)
        self.predict_slice = slice(*self.predict_index) if self.predict_index else None

        self.stage_to_slice = {
            "train": self.train_slice,
            "validate": self.val_slice,
            "test": self.test_slice,
            "predict": self.predict_slice,
            None: None,
        }

    # TODO: unit test this.
    @staticmethod
    def convert_str_to_datetime(str_dt: str, dt_format: str = "%Y-%m-%d %H:%M:%S") -> datetime.datetime:
        """
        Convert a string representation of a date to a datetime object with format: <dt_format>.

        Parameters
        ----------
        str_dt : str
            The string representation of the date.
        dt_format : str, optional
            The format of the date string. Default is "%Y-%m-%d %H:%M:%S".

        Returns
        -------
        datetime.datetime
            The datetime object corresponding to the input string.
        """
        return datetime.datetime.strptime(str_dt, dt_format)

    # TODO: unit test this.
    @staticmethod
    def convert_str_dt_to_imerg_int_dt(str_dt: Optional[str]) -> str:
        """
        Convert a string date to an IMERG integer date format.

        This function is specific to the way IMERG data is stored in the NASA PPS server.

        Parameters
        ----------
        str_dt : Optional[str]
            The string representation of the date, or None.

        Returns
        -------
        Optional[int]
            The IMERG integer representation of the date, or None if input is None.
        """
        if str_dt is not None:
            datetime_dt = IMERGPrecipitationDataModule.convert_str_to_datetime(str_dt)
            y, m, d, hh, mm = IMERGPrecipitationDataModule.get_year_month_day_hour_min_from_datetime(
                datetime_dt, zero_padding=2
            )
            hhmm_in_dt_s = (
                IMERGEarlyRunDataConfig().download_file_path_info["hhmm_to_dt_secs"].get(f"{hh}{mm}")
            )
            str_dt = int(f"{y}{m}{d}{hhmm_in_dt_s}")
        return str_dt

    # TODO: unit test this.
    @staticmethod
    def get_year_month_day_hour_min_from_datetime(
        dt: datetime.datetime, zero_padding: int
    ) -> Tuple[str, str, str, str, str]:
        """
        Extract year, month, day, hour, and minute from a datetime object.

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
        """
        _year = str(dt.year)
        _month = str(dt.month).zfill(zero_padding)
        _day = str(dt.day).zfill(zero_padding)
        _hour = str(dt.hour).zfill(zero_padding)
        _min = str(dt.minute).zfill(zero_padding)

        return _year, _month, _day, _hour, _min

    def get_horizon(self, split: str) -> str:
        """Return the data horizon. Emulate @property but handles different splits."""
        if split in ["predict", "test"]:
            return self.hparams.prediction_horizon or self.hparams.horizon
        else:
            return self.hparams.horizon

    def _check_args(self):
        """Check the initialisation params used."""
        boxes = self.hparams.boxes
        h = self.hparams.horizon
        w = self.hparams.window
        assert isinstance(h, list) or h > 0, f"horizon must be > 0 or a list, but is {h}"
        assert w > 0, f"window must be > 0, but is {w}"
        assert self.hparams.box_size > 0, f"box_size must be > 0, but is {self.hparams.box_size}"
        assert isinstance(boxes, Sequence) or boxes in [
            "all"
        ], f"boxes must be a list or 'all', but is {self.hparams.boxes}"

    # TODO: unit test this.
    def _create_time_index_mask(self, index_to_slice, _slice: slice) -> np.ndarray:
        """
        create a mask for the t_index in the imerg .nc files.
        """
        if (_slice.start is None) and (_slice.stop is not None):
            mask = index_to_slice < _slice.stop
        elif (_slice.stop is None) and (_slice.start is not None):
            mask = index_to_slice >= _slice.start
        elif (_slice.stop is not None) and (_slice.start is not None):
            mask = (index_to_slice >= _slice.start) & (index_to_slice < _slice.stop)
        else:
            mask = None
        return mask

    # TODO: unit test this.
    def get_glob_pattern(self, boxes: Union[List, str] = "all") -> Union[List[Path], str]:
        """
        Get the glob pattern or list of file paths for the IMERG data files.

        The file structure is hard-coded specifically for IMERG data.

        Parameters
        ----------
        boxes : Union[List, str], optional
            Specifies which boxes to use. Can be a list of box indices or "all". Default is "all".

        Returns
        -------
        Union[List[Path], str]
            If boxes is a list, returns a list of Path objects for each specified box.
            If boxes is "all", returns a string glob pattern for all boxes.
        """
        imerg_file_scaffolding = f"imerg.box.*.*.{self.hparams['box_size']}.dt{self.hparams['sequence_dt']}.s{self.hparams['horizon']+self.hparams['window']}.sequenced.nc"
        ddir = Path(self.hparams.data_dir)
        if isinstance(boxes, Sequence) and boxes != "all":
            self.n_boxes = len(boxes)
            log.info(f"training, validation & test using {self.n_boxes} (i, j) boxes: {boxes}.")
            return [
                ddir
                / f"imerg.box.{b.split(',')[0]}.{b.split(',')[1]}.{self.hparams['box_size']}.dt{self.hparams['sequence_dt']}.s{self.hparams['horizon']+self.hparams['window']}.sequenced.nc"
                for b in boxes
            ]
        elif boxes == "all":
            log.info(f"training, validation & test using 'all' (i, j) boxes: {boxes}.")
            # deduce the number of boxes.
            self.n_boxes = len(list(ddir.glob(imerg_file_scaffolding)))
            return str(ddir / imerg_file_scaffolding)
        else:
            raise ValueError(f"Unknown value for boxes: {boxes}")

    # TODO: unit test this.
    def get_ds_xarray(self, split: str, time_slice) -> Union[xr.DataArray, Dict[str, np.ndarray]]:
        """
        Retrieve and process xarray dataset for a given split and time slice.

        Parameters
        ----------
        split : str
            The data split to retrieve ('train', 'val', 'test', or 'predict').
        time_slice : slice
            The time slice to select data from.

        Returns
        -------
        Union[xr.DataArray, Dict[str, np.ndarray]]
            The processed data as an xarray DataArray or a dictionary of numpy arrays.

        Raises
        ------
        ValueError
            If the data files cannot be opened or found in the specified directory.
        """
        glob_pattern = self.get_glob_pattern(self.hparams.boxes)
        log.info(f"{split} data split: [{time_slice.start}, {time_slice.stop}]")
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            try:
                data = xr.open_mfdataset(
                    paths=glob_pattern,
                    combine="nested",
                    concat_dim="time",
                )
                data.coords["time"] = data.coords["time"].astype(int)  # convert to int to allow < > ops.
                mask = self._create_time_index_mask(index_to_slice=data.time, _slice=time_slice)
                masked_data = data.where(mask, drop=True)

                # handle self.predict_slice. ensure predict slice doesn't contaminate 'train' or 'val'.
                # ignore if it is already in test data.
                if self.predict_slice:
                    if split not in ["test", "predict"]:
                        non_predic_slice_mask = (masked_data.time < self.predict_index[0]) | (
                            masked_data.time > self.predict_index[1]
                        )
                        masked_data = masked_data.where(non_predic_slice_mask, drop=True)
                    else:
                        if split == "test":
                            # if predict not in test, add it.
                            predict_mask = self._create_time_index_mask(
                                index_to_slice=data.time, _slice=self.predict_slice
                            )
                            predict_data = data.where(predict_mask, drop=True)
                            masked_data = xr.concat(
                                objs=[masked_data, predict_data], dim="time"
                            ).drop_duplicates(dim="time")

            except OSError as e:
                raise ValueError(
                    f"Could not open imerg-precipitation data files from {glob_pattern}. "
                    f"Check that the data directory is correct: {self.hparams.data_dir}"
                ) from e

        return masked_data.__xarray_dataarray_variable__

    # TODO: unit test this.
    def create_and_set_dataset(self, split: str, dataset: xr.DataArray) -> Dict[str, np.ndarray]:
        """
        Create a torch dataset from the given xarray DataArray.

        For the IMERG data we have to expand the dims.

        Parameters
        ----------
        split : str
            The data split ('train', 'val', 'test', or 'predict').
        dataset : xr.DataArray
            The input xarray DataArray containing the data.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary with a single key 'dynamics' mapping to a numpy array
            of shape (N, S, C, H, W), where:
            N is the number of samples,
            S is window + horizon,
            C is 1 (number of channels),
            H and W are the box size.
        """
        window, horizon = self.hparams.window, self.get_horizon(split)
        X = dataset.to_numpy()  # dims: (N, S, H, W)
        X = np.expand_dims(X, 2)  # dims: (N, S, C, H, W)
        assert X.shape == (
            dataset.shape[0],
            window + horizon,
            1,
            self.hparams["box_size"],
            self.hparams["box_size"],
        )
        return {"dynamics": X}

    def setup(self, stage: Optional[str] = None):
        """
        Load data and set internal variables: self._data_train, self._data_val, self._data_test, self._data_predict.

        This method is automatically called when the PyTorch Lightning Trainer module is initialized.

        Parameters
        ----------
        stage : Optional[str], optional
            Defines the stage of the training process. Can be 'fit', 'validate', 'test', or 'predict'.
            If None, sets up data for all stages. Default is None.
        """
        # get xarray data for each slice.
        ds_train = (
            self.get_ds_xarray("train", self.stage_to_slice["train"])
            if stage in ["fit", "train", None]
            else None
        )
        ds_val = (
            self.get_ds_xarray("validate", self.stage_to_slice["validate"])
            if stage in ["fit", "validate", None]
            else None
        )
        ds_test = (
            self.get_ds_xarray("test", self.stage_to_slice["test"]) if stage in ["test", None] else None
        )
        ds_predict = (
            self.get_ds_xarray("predict", self.stage_to_slice["predict"]) if stage == "predict" else None
        )
        ds_splits = {
            "train": ds_train,
            "val": ds_val,
            "test": ds_test,
            "predict": ds_predict,
        }
        # create the dataset for each split.
        for split, split_ds in ds_splits.items():
            if split_ds is None:
                continue
            if isinstance(split_ds, xr.DataArray):
                # TODO: this is memory intensive. Maybe do it on the fly.
                # create the numpy arrays from the xarray dataset.
                numpy_tensors = self.create_and_set_dataset(split, split_ds)
            else:
                # Alternatively, load the numpy arrays from disk (if requested).
                numpy_tensors = split_ds

            tensor_ds = MyTensorDataset(numpy_tensors, dataset_id=split, **self.dataset_kwargs)
            # save the tensor dataset to self._data_{split}
            setattr(self, f"_data_{split}", tensor_ds)
            assert getattr(self, f"_data_{split}") is not None, f"Could not create {split} dataset"

            # clear space after assignment.
            del numpy_tensors
            del tensor_ds

        # print sizes of the datasets (how many examples).
        self.print_data_sizes(stage)
