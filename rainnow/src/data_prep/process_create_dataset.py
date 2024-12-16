"""A module to create the IMERG dataset using the sequence.csv files."""

import argparse
import logging
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from rainnow.src.configs.config import IMERGEarlyRunConfig
from rainnow.src.configs.data_processing_configs import sequencing_config
from rainnow.src.data_prep.data_processing import (
    create_file_path,
    create_imerg_file_path_mapping,
    open_file_with_h5py_and_get_data,
)
from rainnow.src.utilities.utils import get_logger


def create_sequence_stack(df_sequences, i, j, config, file_mapping):
    """
    Function used for parallelising main().

    Create a stack of sequences from dataframe rows.

    This function creates a 4D numpy array (stack) of sequences and an array of indices
    based on the input dataframe of sequences and configuration parameters.

    Parameters
    ----------
    df_sequences : pandas.DataFrame
        DataFrame containing sequences of file identifiers.
    i : int
        First index for selecting patch from the loaded data.
    j : int
        Second index for selecting patch from the loaded data.
    config : dict
        Configuration dictionary containing keys:
        - 'sequence_length': int
        - 'horizon': int
        - 'patch_size': int
        - 'download_dir': str
        - 'file_suffix': str
        - 'file_type': str
    file_mapping : dict
        Mapping of file identifiers to actual file names.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        stack : np.ndarray
            4D numpy array of shape (n_sequences, sequence_length + horizon, patch_size, patch_size)
            containing the stacked sequences.
        index : np.ndarray
            1D numpy array of objects containing the first file identifier of each sequence.
    """
    stack = np.zeros(
        (
            len(df_sequences),
            config["sequence_length"] + config["horizon"],
            config["patch_size"],
            config["patch_size"],
        )
    )
    index = np.empty(len(df_sequences), dtype=object)

    # loop through all sequences, open and stack the elements and add them to the stack.
    for s in range(len(df_sequences)):
        df_sequence_tmp = df_sequences.iloc[s]
        seq = df_sequence_tmp.astype(str).values.tolist()

        stack[s] = np.stack(
            [
                open_file_with_h5py_and_get_data(file_path=fp)[i, j, :, :]
                for fp in [
                    create_file_path(
                        data_dir=config["download_dir"],
                        file_name=file_mapping[e],
                        file_suffix=config["file_suffix"],
                        file_type=config["file_type"],
                    )
                    for e in seq
                ]
            ]
        )
        index[s] = str(seq[0])

    return stack, index


def main(start_time, end_time, dataset_save_dir, sequence_files, config, imerg_config):
    """
    examples:
    save in curr dir: python process_create_dataset.py "2020-01-01 00:00:00" "2020-02-01 00:00:00" ""
    save in chosen dir: python process_create_dataset.py "2020-01-01 00:00:00" "2020-02-01 00:00:00" "/Volumes/external_disk_seal/data/imerg_datasets"
    """
    log = get_logger(log_file_name=f"logs/{__name__}.log")
    start = time.time()
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    # debug info.
    log.info("running create_datasets.py")
    log.info(f"time window: {start_time} to {end_time}")

    file_mapping = create_imerg_file_path_mapping(
        start_time=start_time, end_time=end_time, imerg_config=imerg_config
    )

    for file_path in sequence_files:
        _start = time.time()
        log.info(f"processing file: {file_path}.")
        pattern = r"_i(\d+)_j(\d+)"  # identify the i, j in the file names.
        matches = re.search(pattern, file_path)
        i = matches.group(1)
        j = matches.group(2)
        df_tmp = pd.read_csv(file_path, header=None)
        seq_start_date, seq_end_date = (
            df_tmp.iloc[:, 0].values.min(),
            df_tmp.iloc[:, 0].values.max(),
        )

        file_save_path = os.path.join(
            f"imerg.box.{i}.{j}.{config['patch_size']}.dt{config['dt']}.s{config['sequence_length']+config['horizon']}.sequenced.{str(seq_start_date)}.{str(seq_end_date)}.nc",
        )

        log.info(f"box (i, j) = ({i}, {j}).")
        log.info(f"number of sequences = {len(df_tmp)}.")

        stack, index = create_sequence_stack(
            df_sequences=df_tmp,
            i=int(i),
            j=int(j),
            config=config,
            file_mapping=file_mapping,
        )

        # check dims.
        assert stack.shape == (
            len(df_tmp),
            config["sequence_length"] + config["horizon"],
            config["patch_size"],
            config["patch_size"],
        )

        if os.path.exists(file_save_path):
            log.warning(
                f"{file_save_path} already exists. Not overwriting, please delete the file manually and run again."
            )
            pass
        else:
            data_array = xr.DataArray(data=stack, dims=["time", "s", "h", "w"], coords={"time": index})

            data_array.to_netcdf(path=file_save_path, engine="h5netcdf")

            log.info(f"{file_save_path} created.")
            log.info(f"time taken ({i},{j}) = {time.time() - _start}s\n")
    log.info(f"(total) time taken = {time.time() - start}s\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start_time", type=str)
    parser.add_argument("end_time", type=str)
    parser.add_argument("dataset_save_dir", type=str)
    args = parser.parse_args()

    # TODO: make this more dynamic.
    # hardcoded for simplicity.
    sequence_files = [
        "sequences/2020-01-01_2024-01-01_i0_j0_s4_h1_dt1_str5_thr0.1_PS_128/eligible_2020-01-01_2024-01-01_i0_j0_s4_h1_dt1_str5_thr0.1_PS_128.csv",
        "sequences/2020-01-01_2024-01-01_i1_j0_s4_h1_dt1_str5_thr0.1_PS_128/eligible_2020-01-01_2024-01-01_i1_j0_s4_h1_dt1_str5_thr0.1_PS_128.csv",
        "sequences/2020-01-01_2024-01-01_i2_j0_s4_h1_dt1_str5_thr0.1_PS_128/eligible_2020-01-01_2024-01-01_i2_j0_s4_h1_dt1_str5_thr0.1_PS_128.csv",
        "sequences/2020-01-01_2024-01-01_i2_j1_s4_h1_dt1_str5_thr0.1_PS_128/eligible_2020-01-01_2024-01-01_i2_j1_s4_h1_dt1_str5_thr0.1_PS_128.csv",
    ]

    config = IMERGEarlyRunConfig()
    main(
        args.start_time,
        args.end_time,
        args.dataset_save_dir,
        sequence_files,
        sequencing_config,
        imerg_config=config,
    )
