"""Config file for the rainnow package."""

import os
from dataclasses import dataclass, field
from typing import Dict, List

import yaml

# FILEPATHS:
DOWNLOAD_DIR = "/Volumes/external_disk_seal/data"  # place to download IMERG data.
GPM_API_DOWNLOAD_DIR_STRUCTURE = "GPM/NRT/IMERG/IMERG-ER"
CREDENTIALS_DIR = "credentials.json"  # credentials will always be in the same place as the config.

# IMERG USER-DEFINED SETTINGS:
IMERG_CROP = {
    "outer": {"latitude": (-66.25, 23.25), "longitude": (-89.95, -26.05)},
    "inner": {"latitude": (-59.85, 16.85), "longitude": (-83.55, -32.45)},
}
IMERG_PIXEL_RES = 0.1
CROP_DIMS = (512, 768)  # x, y (derived from above IMERG_CROP dict)
# PATCH_SIZE = 256
PATCH_SIZE = 128


# TODO: move this back into utils.py.
def _load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        The path to the YAML configuration file.

    Returns
    -------
    dict
        A dictionary containing the configuration data.
    """
    abs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    with open(abs_dir, "r") as file:
        return yaml.safe_load(file)


# IMERG GPM-API DOWNLOAD SETTINGS:
@dataclass
class IMERGEarlyRunConfig:
    """
    PPS.
    """

    # get username and pwd for IMERG PPS server.
    config = _load_config(CREDENTIALS_DIR)
    username = config.get("username")
    password = config.get("password")
    download_from = "PPS"  # what storage to download from. PPS or GES DISK.
    product = "IMERG-ER"  # early-run.
    product_type = "NRT"  # near-real time.
    version = "7"  # TODO: update gpm-api with version 7 applicability.
    time_interval: int = 30  # in minutes.
    file_prefix: str = "3B-HHR-E.MS.MRG.3IMERG"
    file_type: str = ".RT-H5"
    time_var_padding: int = 2  # PPS file structure
    version_mapping: Dict[str, str] = field(default_factory=lambda: {"6": "V06B", "7": "V07B"})
    raw_data_attr: Dict[str, str] = field(
        default_factory=lambda: {
            "6": "Grid/precipitationCal",
            "7": "Grid/precipitation",
        }
    )
    lat_data_attr: str = "Grid/lat"
    lon_data_attr: str = "Grid/lon"
    month_days_mapping: Dict[str, int] = field(
        default_factory=lambda: {
            "01": 31,
            "02": 28,
            "03": 31,
            "04": 30,
            "05": 31,
            "06": 30,
            "07": 31,
            "08": 31,
            "09": 30,
            "10": 31,
            "11": 30,
            "12": 31,
        }
    )
    hhmm_secs: List[str] = field(
        default_factory=lambda: [
            "0000",
            "0030",
            "0060",
            "0090",
            "0120",
            "0150",
            "0180",
            "0210",
            "0240",
            "0270",
            "0300",
            "0330",
            "0360",
            "0390",
            "0420",
            "0450",
            "0480",
            "0510",
            "0540",
            "0570",
            "0600",
            "0630",
            "0660",
            "0690",
            "0720",
            "0750",
            "0780",
            "0810",
            "0840",
            "0870",
            "0900",
            "0930",
            "0960",
            "0990",
            "1020",
            "1050",
            "1080",
            "1110",
            "1140",
            "1170",
            "1200",
            "1230",
            "1260",
            "1290",
            "1320",
            "1350",
            "1380",
            "1410",
        ]
    )
    secs_to_hhmm: Dict[str, str] = field(
        default_factory=lambda: {
            "0000": "S000000-E002959",
            "0030": "S003000-E005959",
            "0060": "S010000-E012959",
            "0090": "S013000-E015959",
            "0120": "S020000-E022959",
            "0150": "S023000-E025959",
            "0180": "S030000-E032959",
            "0210": "S033000-E035959",
            "0240": "S040000-E042959",
            "0270": "S043000-E045959",
            "0300": "S050000-E052959",
            "0330": "S053000-E055959",
            "0360": "S060000-E062959",
            "0390": "S063000-E065959",
            "0420": "S070000-E072959",
            "0450": "S073000-E075959",
            "0480": "S080000-E082959",
            "0510": "S083000-E085959",
            "0540": "S090000-E092959",
            "0570": "S093000-E095959",
            "0600": "S100000-E102959",
            "0630": "S103000-E105959",
            "0660": "S110000-E112959",
            "0690": "S113000-E115959",
            "0720": "S120000-E122959",
            "0750": "S123000-E125959",
            "0780": "S130000-E132959",
            "0810": "S133000-E135959",
            "0840": "S140000-E142959",
            "0870": "S143000-E145959",
            "0900": "S150000-E152959",
            "0930": "S153000-E155959",
            "0960": "S160000-E162959",
            "0990": "S163000-E165959",
            "1020": "S170000-E172959",
            "1050": "S173000-E175959",
            "1080": "S180000-E182959",
            "1110": "S183000-E185959",
            "1140": "S190000-E192959",
            "1170": "S193000-E195959",
            "1200": "S200000-E202959",
            "1230": "S203000-E205959",
            "1260": "S210000-E212959",
            "1290": "S213000-E215959",
            "1320": "S220000-E222959",
            "1350": "S223000-E225959",
            "1380": "S230000-E232959",
            "1410": "S233000-E235959",
        }
    )
