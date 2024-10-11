"""A wrapper module to sequence the cropped and patched IMERG data. 

This file runs on the output of process_raw_imerg_data.py"""

import argparse
import csv
import datetime
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from rainnow.src.configs.data_processing_configs import sequencing_config
from rainnow.src.data_prep.data_processing import (
    create_file_path,
    create_imerg_file_path_mapping,
    generate_list_of_sequences_from_file_mapping,
    get_available_sequences_from_file_mapping,
    open_file_with_h5py_and_get_data,
)
from rainnow.src.utilities.utils import calc_avg_value_2d_arr, get_logger


def sequence_element_check_wrapper(file_path: str, i: int, j: int) -> float:
    """
    Process a single sequence element file w/ dims: (i, j, H, W) and calculate its average value.

    This function serves as a wrapper for parallel processing of sequence elements.
    It opens an HDF5 file containing sequence element data, extracts a 2D slice,
    and calculates the average value of that slice.

    Args:
        file_path (str): Path to the HDF5 file containing sequence element data.
        i (int): First index for selecting the 2D slice from the 4D array.
        j (int): Second index for selecting the 2D slice from the 4D array.

    Returns:
        float: The calculated average value of the 2D slice.
    """

    data = open_file_with_h5py_and_get_data(file_path)
    value = calc_avg_value_2d_arr(arr=data[i, j, :, :])
    return value


def process_sequence_wrapper(
    seq_idx: int,
    sequence: List[str],
    i: int,
    j: int,
    file_mapping: Dict[str, str],
    config: Dict[str, str],
) -> Tuple[bool, int]:
    """
    Process an entire sequence of elements in parallel and check against a threshold.

    This function serves as a wrapper for parallel processing of multiple sequence
    elements. It creates file paths for each element, processes them in parallel
    using ThreadPoolExecutor, and checks if any processed value exceeds a specified
    threshold.

    Args:
        seq_idx (int): Index of the current sequence being processed.
        sequence (list): List of sequence elements to process.
        i (int): First index for selecting the 2D slice from each 4D array.
        j (int): Second index for selecting the 2D slice from each 4D array.
        file_mapping (dict): Mapping of sequence elements to their corresponding file names.
        config (dict): Configuration dictionary containing processing parameters.

    Returns:
        tuple: A tuple containing:
            - bool: True if any processed value exceeds the threshold, False otherwise.
            - int: The input seq_idx.
    """
    file_paths = [
        create_file_path(
            data_dir=config["download_dir"],
            file_name=file_mapping[seq_elem],
            file_suffix=config["file_suffix"],
            file_type=config["file_type"],
        )
        for seq_elem in sequence
    ]
    sequence_chks = []
    with ThreadPoolExecutor(max_workers=len(sequence)) as tpool:  # thread per sequence.
        futures = [
            tpool.submit(
                sequence_element_check_wrapper,
                file_path,
                i,
                j,
            )
            for file_path in file_paths
        ]
        for future in as_completed(futures):
            sequence_chks.append(future.result())

    if max(sequence_chks) >= config["pixel_threshold"]:
        return True, seq_idx
    else:
        return False, seq_idx


def main(start_time: str, end_time: str, i: str, j: str, config: dict):
    """
    Example python: python process_data_sequences.py '2020-01-01 00:00:00' '2024-01-01 00:00:00'

    Run this via scripts/process_data_sequences.sh via cmd:
    ./scripts/process_data_sequences.sh '2020-01-01 00:00:00' '2024-01-01 00:00:00' scripts/sequence_grid_ij_config.txt
    """
    log = get_logger(log_file_name=config["logs"], name=__name__)
    start = time.time()
    start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    i, j = int(i), int(j)

    # debug info.
    log.info("running generate_data_sequences.py")
    log.info(f"time window: {start_time} to {end_time}")
    log.info(f"(i, j) = ({i}, {j})")
    log.info(
        f"sequence config:\n\n***** SEQUENCING CONFIG *****\n{json.dumps(config, indent=2)}\n*****************************"
    )

    # check if need to run.
    config_suffix = (
        f"{start_time.date()}_{end_time.date()}_"
        f"i{i}_j{j}_"
        f"s{config['sequence_length']}_"
        f"h{config['horizon']}_"
        f"dt{config['dt']}_"
        f"str{config['stride']}_"
        f"thr{config['pixel_threshold']}_"
        f"PS_{config['patch_size']}"
    )

    output_dir = Path(f"sequences/{config_suffix}")
    if not os.path.exists(output_dir):
        # .csv files already exists.
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # assume already have folder with files.
        return

    file_mapping = create_imerg_file_path_mapping(
        start_time=start_time, end_time=end_time, imerg_config=config
    )
    list_sequences = generate_list_of_sequences_from_file_mapping(
        file_mapping=file_mapping,
        sequence_length=config["sequence_length"],
        horizon=config["horizon"],
        dt=config["dt"],  # undersampling if > 1.
        stride=config["stride"],
    )
    available, unavailable = get_available_sequences_from_file_mapping(
        list_sequences=list_sequences, file_mapping=file_mapping, config=config
    )

    # importance sampling (** multi-processed & multi-threaded **).
    eligible_idxs, not_eligible_idxs = [], []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ppool:
        futures = [
            ppool.submit(process_sequence_wrapper, e, sequence, i, j, file_mapping, config)
            for e, sequence in enumerate(available)
        ]
        # handle each output concurrently (unordered).
        for future in as_completed(futures):
            if future.result()[0]:
                eligible_idxs.append(future.result()[1])
            else:
                not_eligible_idxs.append(future.result()[1])

    eligible = np.array(available)[sorted(eligible_idxs)]
    not_eligible = np.array(available)[sorted(not_eligible_idxs)]

    # write all lists to .csv files.
    files_to_write = [
        (list_sequences, f"raw_sequences_{config_suffix}.csv"),
        (available, f"available_{config_suffix}.csv"),
        (unavailable, f"unavailable_{config_suffix}.csv"),
        (eligible, f"eligible_{config_suffix}.csv"),
        (not_eligible, f"not_eligible_{config_suffix}.csv"),
    ]
    for data, file_name in files_to_write:
        with open(output_dir / file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)

    # debug info.
    log.info(f"number of raw sequences: {len(list_sequences)}")
    log.info(f"number of available sequences: {len(available)}/{len(list_sequences)}")
    log.info(f"number of eligible sequences: {len(eligible)}/{len(list_sequences)}")
    log.info(f"time taken = {time.time() - start}s\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start_time", type=str)
    parser.add_argument("end_time", type=str)
    parser.add_argument("i", type=str)
    parser.add_argument("j", type=str)

    args = parser.parse_args()

    main(
        start_time=args.start_time,
        end_time=args.end_time,
        i=args.i,
        j=args.j,
        config=sequencing_config,
    )
