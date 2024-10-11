"""Wrapper file to re-download any files that are corrupt."""

import argparse
import logging
import os
import re
from datetime import datetime

import gpm

from rainnow.src.configs.config import DOWNLOAD_DIR, IMERGEarlyRunConfig
from rainnow.src.utilities.utils import get_logger, str_input_to_bool


def main(file_path: str, delete_files: bool):
    """
    >>> python process_corrupted_files.py logs/corrupt_files.txt True
    """
    log = get_logger(log_file_name="logs/del_and_download_logs", name=__name__)
    log.info(f"running process_corrupted_files.py")
    log.info(f"DELETE_FILES is set to {delete_files}")

    files_to_delete = []
    if delete_files:
        with open(file_path, "r") as file:
            corrupted_files = [line.strip() for line in file.readlines()]

        # handle no corrupted files.
        if len(corrupted_files) == 0:
            log.info(f"No corrupted files in {file_path}")
            return

        # get min and max dates for re-download.
        date_pattern = re.compile(r"/(\d{4}/\d{2}/\d{2})/")
        dates = []
        for i in corrupted_files:
            yyyy_mm_dd = date_pattern.search(i).group(1)
            date = datetime.strptime(f"{yyyy_mm_dd} 00:00:00", "%Y/%m/%d %H:%M:%S")
            dates.append(date)
        min_date = min(dates)
        max_date = max(dates)

        for file_to_delete in corrupted_files:
            if os.path.exists(file_to_delete):
                files_to_delete.append(file_to_delete)
                os.remove(file_to_delete)

        log.info(f"deleted {len(corrupted_files)} raw IMERG files")
        log.info(f"re-downloading files between {min_date} and {max_date}")

        # define gpm download config & re-download the data.
        imerg_config = IMERGEarlyRunConfig()
        gpm.define_configs(
            username_pps=imerg_config.username,
            password_pps=imerg_config.password,
            base_dir=DOWNLOAD_DIR,
        )
        gpm.download(
            product=imerg_config.product,
            product_type=imerg_config.product_type,
            storage=imerg_config.download_from,
            version=6,  # gpm-api doesn't currently handle 7 (imerg_config.version).
            start_time=min_date,
            end_time=max_date,
            force_download=False,  # fixed this to False to only download deleted files.
            verbose=True,
            progress_bar=True,
            check_integrity=False,
        )
        # check that deleted files were downloaded.
        if not all([os.path.exists(file_path) for file_path in files_to_delete]):
            log.warning("ALL deleted files not re-downloaded!!!")


if __name__ == "__main__":
    # handle input args.
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("delete_files", type=str_input_to_bool)
    args = parser.parse_args()

    main(file_path=args.file_path, delete_files=args.delete_files)
