"""A wrapper module to crop and patch raw IMERG data. The module also checks raw data for corruption."""

import argparse
import datetime
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Union

from rainnow.src.configs.data_processing_configs import data_processing_config
from rainnow.src.data_prep.data_processing import (
    create_file_path,
    create_imerg_file_path_mapping,
    crop_xarr_using_lat_and_lon,
    load_imerg_h5_file_into_xarr,
    patch_2d_arr_into_nxn_squares,
    save_arr_data_with_h5py,
)
from rainnow.src.utilities.utils import calculate_required_1d_padding, get_logger, str_input_to_bool


def data_prep_wrapper(
    imerg_config,
    file_path: str,
    save_file_path: str,
) -> Union[None, str]:
    """
    Data preparation wrapper to allow parallelization.

    This function performs the following steps:
    1. Opens and crops the .RT-H5 file using xarray and slicing with lat/lon limits.
    2. Patches the cropped array to create patch_size x patch_size squares.
    3. Saves the file as save_file_path.

    Parameters
    ----------
    imerg_config : dict
        Configuration dictionary containing crop, method, patch_size, and overlap parameters.
    file_path : str
        Path to the input .RT-H5 file.
    save_file_path : str
        Path where the processed file will be saved.

    Returns
    -------
    Union[None, str]
        None if the processing is successful, or the file_path as a string if an exception occurs
        (assumed to be due to a corrupt file).
    """
    try:
        # open, crop, patch + save raw IMERG tile.
        raw_arr = load_imerg_h5_file_into_xarr(file_path=file_path)
        cropped_arr = crop_xarr_using_lat_and_lon(
            xarr=raw_arr,
            lat_lims=imerg_config["crop"]["latitude"],
            lon_lims=imerg_config["crop"]["longitude"],
        )
        y_max, x_max = cropped_arr.shape  # opposite due to (lat, lon).
        frac = 0
        if imerg_config["method"] == "overlapping_patches":
            frac = int(1 / imerg_config["overlap"])
        x_pad = calculate_required_1d_padding(X=x_max, Y=imerg_config["patch_size"], frac=frac)
        y_pad = calculate_required_1d_padding(X=y_max, Y=imerg_config["patch_size"], frac=frac)
        patched_arr = patch_2d_arr_into_nxn_squares(
            arr2d=cropped_arr,
            n=imerg_config["patch_size"],
            x_pad=x_pad,
            y_pad=y_pad,
            flip_pixels=True,
            overlap=imerg_config["overlap"],
        )
        save_arr_data_with_h5py(arr_data=patched_arr, file_save_path=save_file_path)
        return None
    except:
        # assume corrupt file.
        return file_path


def main(
    start_time: str,
    end_time: str,
    overwrite: bool,
    config: dict,
):
    """
    wrapper function to execute data pipeline for raw IMERG Early Run .RT-H5 files.
    The files are processed and saved back to disk.
    >>> python process_raw_imerg_data.py "2022-01-01 00:00:00" "2022-01-01 01:00:00" False
    """
    log = get_logger(log_file_name=config["logs"], name=__name__)
    start = time.time()

    # check to see if the download dir is available
    if not os.path.exists(config["download_dir"]):
        log.warning(f"The specified data directory: {config['download_dir']} does not exist.")
        log.warning("If the data is on an external disk, please make sure it is plugged in.")
        log.warning(
            "If the data is stored on an online server, please make sure that you are connected."
        )
        return

    # debug info.
    log.info(f"running process_raw_imerg_data.py\n\n")
    log.info(
        f"data processing config:\n\n***** DATA CONFIG *****\n{json.dumps(config, indent=2)}\n***********************\n"
    )
    log.info(f"OVERWRITE is set to {overwrite}")

    start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    file_mapping = create_imerg_file_path_mapping(
        start_time=start_time,
        end_time=end_time,
        imerg_config=config,
    )

    # modify the file save suffix if using overlapping patches.
    if config["method"] == "overlapping_patches":
        save_file_suffix = config["save_file_suffix"] + "_olap"
    else:
        save_file_suffix = config["save_file_suffix"]

    if not overwrite:
        # check to see if any files already exist to subset the file_mapping.
        file_mapping = {
            k: v
            for k, v in file_mapping.items()
            if not os.path.exists(
                create_file_path(
                    data_dir=config["download_dir"],
                    file_name=v,
                    file_suffix=save_file_suffix,
                    file_type=config["save_file_type"],
                )
            )
        }

    # data processing: crop, patch and save. (** multi-processed **).
    futures = []
    corrupted_files = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for _, v in file_mapping.items():
            file_path = create_file_path(
                data_dir=config["download_dir"],
                file_name=v,
                file_suffix="",
                file_type=config["raw_file_type"],
            )
            save_file_path = create_file_path(
                data_dir=config["download_dir"],
                file_name=v,
                file_suffix=save_file_suffix,
                file_type=config["save_file_type"],
            )
            futures.append(
                executor.submit(
                    data_prep_wrapper,
                    config,
                    file_path,
                    save_file_path,
                )
            )
        for future in as_completed(futures):
            result = future.result()
            if result:
                corrupted_files.append(result)

    with open(config["corrupted_files_log_filename"], "w") as f:
        for file in corrupted_files:
            f.write(f"{file}\n")

    # debug info.
    log.info(f"time window: {start_time} to {end_time}")
    log.info(f"number of files to process: {len(file_mapping)}")
    log.info(f"number of parallel tasks to be executed: {len(futures)}")
    log.info(f"number of corrupted files: {len(corrupted_files)}")
    log.info(f"time taken = {time.time() - start}s\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start_time", type=str)
    parser.add_argument("end_time", type=str)
    parser.add_argument("overwrite", type=str_input_to_bool)
    args = parser.parse_args()

    main(
        start_time=args.start_time,
        end_time=args.end_time,
        overwrite=args.overwrite,
        config=data_processing_config,
    )
