"""A wrapper module to download IMERG data usin the gpm_api library."""

import argparse
import datetime
import logging

import gpm

from rainnow.src.configs.config import DOWNLOAD_DIR, IMERGEarlyRunConfig
from rainnow.src.utilities.utils import get_logger, str_input_to_bool


def main(start_time: str, end_time: str, force_download: bool):
    """
    >>> python download_imerg_data.py "2022-01-01 00:00:00" "2022-01-01 01:00:00" False
    """
    get_logger(log_file_name="logs/downloading_logs", name=__name__)

    # debug info.
    logging.info(f"running download_imerg_data.py")
    logging.info(f"downloading time window: {start_time} to {end_time}")

    start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    imerg_config = IMERGEarlyRunConfig()

    # re-download IMERG data using gpm-api
    gpm.define_configs(
        username_pps=imerg_config.username,
        password_pps=imerg_config.password,
        base_dir=DOWNLOAD_DIR,
    )
    gpm.download(
        product=imerg_config.product,
        product_type=imerg_config.product_type,
        storage=imerg_config.download_from,
        version=None,  # None will get the latest available version (V07B). gpm-api doesn't currently handle 7 (imerg_config.version)
        start_time=start_time,
        end_time=end_time,
        n_threads=12,
        force_download=force_download,  # defaults to False to only download non-existing files.
        verbose=True,
        progress_bar=True,
        check_integrity=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start_time", type=str)
    parser.add_argument("end_time", type=str)
    parser.add_argument("force_download", type=str_input_to_bool, nargs="?", default=False)
    args = parser.parse_args()

    main(args.start_time, args.end_time, args.force_download)
