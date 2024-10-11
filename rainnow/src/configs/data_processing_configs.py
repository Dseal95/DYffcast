"""Module containing the data-level and sequencing-level IMERG config files."""

import os

from rainnow.src.configs.config import (
    DOWNLOAD_DIR,
    GPM_API_DOWNLOAD_DIR_STRUCTURE,
    IMERG_CROP,
    PATCH_SIZE,
    IMERGEarlyRunConfig,
)

imerg_config = IMERGEarlyRunConfig()

data_processing_config = {
    "method": "non_overlapping_patches",
    "overlap": 0,
    # "method": "overlapping_patches",
    # "overlap": 0.5,
    "patch_size": PATCH_SIZE,
    "crop": IMERG_CROP["inner"],
    # "crop": IMERG_CROP["outer"],
    "download_dir": os.path.join(DOWNLOAD_DIR, GPM_API_DOWNLOAD_DIR_STRUCTURE),
    "raw_file_type": imerg_config.file_type,
    "save_file_type": ".h5",
    "save_file_data_attr": "data",
    "save_file_suffix": f"_cropped_{PATCH_SIZE}_patched",  # to know what the patch sizes are.
    "corrupted_files_log_filename": "logs/corrupt_files.txt",  # keep hardcoded for simplicity.
    "logs": "logs/data_prep",
}

sequencing_config = {
    "sequence_length": 15,  # 8,  # 4,  # the *total sequence length* = sequence_length + h
    "horizon": 1,
    "dt": 1,  # intervals between consecutive imerg tiles.
    "stride": 16,  # no overlap if stride = h + seq_length (mimicing for DYffusion).
    "pixel_threshold": 0.1,
    "file_suffix": f"_cropped_{PATCH_SIZE}_patched",
    "file_type": ".h5",
    "logs": "logs/data_sequencing",
    "download_dir": os.path.join(DOWNLOAD_DIR, GPM_API_DOWNLOAD_DIR_STRUCTURE),
    "patch_size": PATCH_SIZE,
}
