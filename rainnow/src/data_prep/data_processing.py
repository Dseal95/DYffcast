"""data preparation module containing functionality to process raw data."""

import datetime
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import xarray as xr

from rainnow.src.configs.config import CROP_DIMS, IMERGEarlyRunConfig
from rainnow.src.utilities.utils import (
    calculate_required_1d_padding,
    get_year_month_day_hour_min_from_datetime,
)


# TODO: unit test this.
def create_file_path(data_dir: str, file_name: str, file_suffix: str, file_type: str) -> str:
    """
    Generate a file path using a data DIR and then file = filename + filesuffix + file_type.

    Parameters
    ----------
    data_dir : str
        The directory path.
    file_name : str
        The base name of the file.
    file_suffix : str
        The suffix to be added to the file name.
    file_type : str
        The file extension.

    Returns
    -------
    str
        The complete file path.
    """
    file_path = os.path.join(
        data_dir,
        f"{file_name}{file_suffix}{file_type}",
    )

    return file_path


# TODO: unit test this.
def generate_list_of_sequences_from_file_mapping(
    file_mapping: Dict[int, str], sequence_length: int, horizon: int, dt: int, stride: int
) -> List[List[int]]:
    """
    Generate an exhaustive list of sequences based on a pre-built file_mapping.

    Parameters
    ----------
    file_mapping : Dict[int, str]
        A dictionary mapping timestamps to file paths.
    sequence_length : int
        The length of each sequence.
    horizon : int
        The forecast horizon.
    dt : int
        The time step between consecutive elements in a sequence.
    stride : int
        The number of time steps to move between sequences.

    Returns
    -------
    List[List[int]]
        A list of sequences, where each sequence is a list of timestamps.
    """
    sorted_file_mapping = sorted(file_mapping.keys())
    sequences = []
    for i in range(0, len(sorted_file_mapping) - (sequence_length + horizon) * dt, dt * stride):
        sequences.append(sorted_file_mapping[i : (i + ((sequence_length + horizon)) * dt) : dt])

    return sequences


# TODO: unit test this.
def get_available_sequences_from_file_mapping(
    list_sequences: List[List[str]], file_mapping: Dict[str, str], config: Dict[str, str]
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Check which sequences have all files available in the specified directory.

    This function verifies the existence of files for each sequence in the input list,
    using the provided file mapping and configuration.

    Parameters
    ----------
    list_sequences : List[List[str]]
        Sequences of file keys to check. Each inner list represents a sequence.
    file_mapping : Dict[str, str]
        Maps file keys to file names.
    config : Dict[str, str]
        IMERG data configuration. Must contain keys:
        - 'download_dir': directory where files are stored
        - 'file_suffix': suffix of the files
        - 'file_type': type of the files

    Returns
    -------
    Tuple[List[List[str]], List[List[str]]]
        A tuple containing two lists:
        - available: sequences where all files exist
        - unavailable: sequences where at least one file is missing
    """
    available, unavailable = [], []
    for sequence in list_sequences:
        chk = []
        for file_key in sequence:
            file_path = create_file_path(
                data_dir=config["download_dir"],
                file_name=file_mapping[file_key],
                file_suffix=config["file_suffix"],
                file_type=config["file_type"],
            )
            chk.append(os.path.exists(file_path))
        if np.all(chk):
            available.append(sequence)
        else:
            unavailable.append(sequence)

    return available, unavailable


# TODO: unit test this.
def load_imerg_h5_file_into_xarr(file_path) -> xr.DataArray:
    """
    Load a raw IMERG .h5 file into an xarray.DataArray.

    This function reads latitude, longitude, and precipitation data from the loaded data.

    Parameters
    ----------
    file_path : str
        Path to the IMERG .h5 file.

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing the IMERG precipitation data.
        The DataArray has dimensions 'latitude' and 'longitude'.
    """
    # get IMERG data config.
    imerg_config = IMERGEarlyRunConfig()
    with h5py.File(file_path, "r") as h5file:
        lat = h5file[imerg_config.lat_data_attr][:]
        lon = h5file[imerg_config.lon_data_attr][:]
        # need to transpose to gets it into lat, lon dims.
        data = h5file[imerg_config.raw_data_attr[imerg_config.version]][0, :, :].T
        # load into xarray obj with lat, lon dims.
        xarr = xr.DataArray(
            data,
            dims=["latitude", "longitude"],
            coords={"latitude": lat, "longitude": lon},
        )
        return xarr


# TODO: unit test this.
def crop_xarr_using_lat_and_lon(xarr, lat_lims, lon_lims) -> xr.DataArray:
    """
    Crop an xarray DataArray using latitude and longitude limits.

    This function assumes the input DataArray has labeled dimensions
    'latitude' and 'longitude'.

    Parameters
    ----------
    xarr : xr.DataArray
        The input xarray DataArray to be cropped.
    lat_lims : Optional[Tuple[float, float]]
        The latitude limits for cropping, given as (min_lat, max_lat).
        If None, no latitude cropping is performed.
    lon_lims : Optional[Tuple[float, float]]
        The longitude limits for cropping, given as (min_lon, max_lon).
        If None, no longitude cropping is performed.

    Returns
    -------
    xr.DataArray
        A new xarray DataArray that has been cropped according to the provided limits.
    """
    # slice the array using latitude and longitude.
    slice_params = {}
    if lat_lims is not None:
        slice_params["latitude"] = slice(*lat_lims)
    if lon_lims is not None:
        slice_params["longitude"] = slice(*lon_lims)

    return xarr.sel(**slice_params)


# TODO: unit test this.
def open_file_with_h5py_and_get_data(file_path, data_attr: str = "data") -> np.ndarray:
    """
    Open an HDF5 file and retrieve data from a specified attribute (explicitly .h5.).

    This function opens an HDF5 (.h5) file at the given file path and extracts
    data from the specified attribute.

    Parameters
    ----------
    file_path : str
        The path to the HDF5 file to be opened.
    data_attr : str, optional
        The name of the attribute in the HDF5 file from which to retrieve data.
        Default is "data".

    Returns
    -------
    np.ndarray
        The data retrieved from the specified attribute in the HDF5 file.
    """
    with h5py.File(f"{file_path}", "r") as f:
        data = f[data_attr][:]
    return data


def save_arr_data_with_h5py(arr_data, file_save_path, chunk_size=None, data_attr: str = "data") -> None:
    """
    Save array data to an HDF5 file using h5py (explicitly .h5.).

    Parameters
    ----------
    arr_data : np.ndarray
        The array data to be saved.
    file_save_path : str
        The path where the HDF5 file will be saved. Should end with '.h5'.
    chunk_size : Optional[Tuple[int, ...]], optional
        The chunk size for the HDF5 dataset. If None, h5py will use its default chunking.
        The tuple should have the same number of elements as the dimensions of arr_data.
    data_attr : str, optional
        The name of the dataset within the HDF5 file. Default is "data".

    Returns
    -------
    None
    """
    with h5py.File(f"{file_save_path}", "w") as f:
        f.create_dataset(
            name=data_attr,
            data=arr_data,
            compression="gzip",
            chunks=chunk_size,
            dtype=arr_data.dtype,
        )


def patch_2d_arr_into_nxn_squares(
    arr2d, n, x_pad, y_pad, flip_pixels: bool = False, overlap: float = 0
) -> np.ndarray:
    """
    Patch a 2D array into n x n squares with optional overlap and padding.

    This function divides a 2D array into patches of size n x n. It can handle
    cases where the array dimensions are not exact multiples of n by using
    padding. It also supports overlapping patches.

    Parameters
    ----------
    arr2d : np.ndarray
        The input 2D array to be patched.
    n : int
        The size of each square patch.
    x_pad : int
        The amount of padding to use in the x-direction when necessary.
    y_pad : int
        The amount of padding to use in the y-direction when necessary.
    flip_pixels : bool, optional
        If True, the input array is flipped vertically before patching.
        Default is False.
    overlap : float, optional
        The fraction of overlap between adjacent patches. Must be between 0 and 1.
        Default is 0 (no overlap).

    Returns
    -------
    np.ndarray
        A 4D array of patches with shape (max_rows, max_cols, n, n).
    """

    # error handling.
    assert len(arr2d.shape) == 2  # 2d.
    y_max, x_max = arr2d.shape
    assert (y_max >= n) and (x_max >= n)  # sufficient input arr dims.

    if flip_pixels:
        # flip pixels so that (0, 0) is the 0th pixel top-left.
        arr2d = np.flipud(arr2d)

    stride = int(n * (1 - overlap))
    max_rows = int(np.ceil((y_max - n) / stride) + 1)
    max_cols = int(np.ceil((x_max - n) / stride) + 1)
    patches = []
    for r in range(max_rows):
        for c in range(max_cols):
            row_start = r * stride
            col_start = c * stride
            if (row_start + n <= y_max) and (col_start + n <= x_max):
                # no overlap.
                patch = arr2d[
                    row_start : row_start + n,
                    col_start : col_start + n,
                ]
            elif (row_start + n <= y_max) and (col_start + n > x_max):
                # only longitude overlap.
                patch = arr2d[
                    row_start : row_start + n,
                    col_start - x_pad : col_start + n,
                ]
            elif (row_start + n > y_max) and (col_start + n <= x_max):
                # only latitude overlap.
                patch = arr2d[
                    row_start - y_pad : row_start + n,
                    col_start : col_start + n,
                ]
            else:
                # overlap in lat and lon.
                patch = arr2d[
                    row_start - y_pad : row_start + n,
                    col_start - x_pad : col_start + n,
                ]
            patches.append(patch)

    arr_patches = np.array(patches).reshape((max_rows, max_cols, n, n))

    return arr_patches


def create_square_patches_from_2d_arr_using_mirror_padding(
    arr2d: np.ndarray,
    patch_size: int = 256,
    overlap: int = 1,
    flip_pixels: bool = False,
    _print: bool = False,
) -> np.ndarray:
    """
    Create 2D patches (from the top-left down) from an IMERG data tile with optional overlap and mirror padding.

    A padding number of pixels are taken from the latitude and longitude max and padded to the end of the
    image before the patches are created. This will create a mirror of pixels in the outer _| patches of the grid.

    For example, consider a 3x3 grid, the patches created at [0,2], [1,2] will have a padded number of
    pixels from [0,1] and [1,1] longitude only (x-axis) mirrored and attached to the end of their pixels to
    force patch_size in x_axis. Similarly, [2,0] and [2,1] will have the a padded number of latitude pixels (y-axis)
    from y_max of [1,0] and [1,1] mirrored and attached to their y_max. Lastly, the [2,2] will have both latitude
    and longitude pixels attached to its x and y axis.

               |
     [0,0][0,1]|[0,2]
     [1,0][1,1]|[1,2]
     ------------------
     [2,0][2,1]|[2,2]
               |

    Parameters
    ----------
    arr2d : np.ndarray
        The 2D numpy array representing the IMERG data tile.
    patch_size : int
        The size of each square patch.
    overlap : int
        The denominator to determine the fractional overlap (1/overlap).
    flip_pixels : bool
        Flag to enable or disable flipping the image pixels to ensure (0,0) is top-left.
    _print : bool, optional
        Flag to enable or disable printing of debug information, by default True.

    Returns
    -------
    np.ndarray
        A 4D numpy array where the first two dimensions index the patch position
        and the last two dimensions are the patch data.
    """
    if flip_pixels:
        # flip pixels so that (0, 0) is the 0th pixel top-left.
        arr2d = np.flipud(arr2d)

    # calculate the padding needed to make the tile divisible by patch_size with approximately half-patch overlaps.
    lat_pad = calculate_required_1d_padding(X=arr2d.shape[0], Y=patch_size, frac=overlap)
    lon_pad = calculate_required_1d_padding(X=arr2d.shape[1], Y=patch_size, frac=overlap)

    # add padding along the latitude (y-axis) and longitude (x-axis).
    lat_padding = arr2d[-lat_pad:, :]
    imerg_padded_lat = np.concatenate([arr2d, lat_padding], axis=0)
    lon_padding = imerg_padded_lat[:, -lon_pad:]
    imerg_padded_lat_lon = np.concatenate((imerg_padded_lat, lon_padding), axis=1)

    # calculate the stride for overlapping patches.
    frac = 0.0 if overlap <= 1 else (1 / overlap)
    x_stride = int(patch_size - (frac * patch_size))
    y_stride = int(patch_size - (frac * patch_size))
    y_coord, x_coord = imerg_padded_lat_lon.shape
    row_cnt, col_cnt = 0, 0
    patches = []
    for r in range(0, y_coord - patch_size + 1, y_stride):
        for c in range(0, x_coord - patch_size + 1, x_stride):
            # flip the data so that the 0th pixel at (0, 0) to top left.
            patch = imerg_padded_lat_lon[r : r + patch_size, c : c + patch_size]
            patches.append(patch)
            col_cnt += 1
        row_cnt += 1

    arr_patches = np.array(patches).reshape(row_cnt, int(col_cnt / row_cnt), patch_size, patch_size)

    # debug info.
    if _print:
        print("** padding info **")
        print(f"latitude padding = {lat_pad}")
        print(f"longitude padding = {lon_pad}\n")
        print(f"created {len(patches)}, {patch_size}x{patch_size} patches: ({arr_patches.shape})")

    return arr_patches


# TODO: unit test this.
def create_imerg_file_path_mapping(
    start_time: datetime.datetime, end_time: datetime.datetime, imerg_config: dict
) -> Dict[str, str]:
    """
    Create a mapping of IMERG file paths for a given time range.

    This function generates a dictionary, mapping time keys to IMERG file paths
    for all files within the specified time range.

    Parameters
    ----------
    start_time : datetime.datetime
        The start time of the desired IMERG data range.
    end_time : datetime.datetime
        The end time of the desired IMERG data range.
    imerg_config : dict
        A dictionary containing IMERG configuration parameters.

    Returns
    -------
    Dict[str, str]
        A dictionary where keys are time strings in the format 'YYYYMMDDHHMM',
        and values are the corresponding IMERG file paths.
    """
    imerg_config = IMERGEarlyRunConfig()

    # get imerg information from config.
    download_file_prefix = imerg_config.file_prefix
    version = imerg_config.version_mapping[imerg_config.version]
    t_pad = imerg_config.time_var_padding
    month_day_mapping = imerg_config.month_days_mapping
    s_to_hhmm_mapping = imerg_config.secs_to_hhmm
    hhmm_in_s = imerg_config.hhmm_secs
    dt = imerg_config.time_interval

    s_year, s_month, s_day, s_hour, s_min = get_year_month_day_hour_min_from_datetime(
        dt=start_time, zero_padding=t_pad
    )
    e_year, e_month, e_day, e_hour, e_min = get_year_month_day_hour_min_from_datetime(
        dt=end_time, zero_padding=t_pad
    )
    # generate exhaustive file mapping for start and end (yyyy-mm-dd)
    yy = [y for y in range(int(s_year), int(e_year) + 1)]
    mths = [str(month).zfill(2) for month in range(1, 12 + 1)]
    yrs_mnths = [f"{y}{m}" for y in yy for m in mths] + [
        f"{e_year}{m}" for m in mths[: mths.index(e_month) + 1]
    ]
    yrs_mnths_days = (
        [
            f"{yymm}{dd}"
            for yymm in yrs_mnths[:1]
            for dd in [
                str(day).zfill(t_pad) for day in range(int(s_day), month_day_mapping[yymm[-2:]] + 1)
            ]
        ]
        + [
            f"{yymm}{dd}"
            for yymm in yrs_mnths[1:-1]
            for dd in [str(day).zfill(t_pad) for day in range(1, month_day_mapping[yymm[-2:]] + 1)]
        ]
        + [
            f"{yymm}{dd}"
            for yymm in yrs_mnths[-1:]
            for dd in [str(day).zfill(t_pad) for day in range(1, int(e_day) + 1)]
        ]
    )
    yyyymmddhhmm = [f"{yyyymmdd}{hhmm}" for yyyymmdd in yrs_mnths_days for hhmm in hhmm_in_s]

    file_mapping = {
        k: v
        for k in yyyymmddhhmm
        for v in [
            f"{k[:4]}/{k[4:6]}/{k[6:8]}/{download_file_prefix}.{k[:8]}-{s_to_hhmm_mapping[k[8:]]}.{k[8:]}.{version}"
        ]
    }

    # subset the file_mapping with start and end times (hh:mm).
    start_key = s_year + s_month + s_day + str(((int(s_hour) * 2) + int(s_min) // dt) * dt).zfill(4)
    end_key = e_year + e_month + e_day + str(((int(e_hour) * 2) + int(e_min) // dt) * dt).zfill(4)

    subset_file_mapping = {
        k: v for k, v in file_mapping.items() if int(start_key) <= int(k) <= int(end_key)
    }

    return subset_file_mapping
