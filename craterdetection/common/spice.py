import os
import urllib
from pathlib import Path
from typing import List

import spiceypy as spice

import craterdetection.common.constants as const


def download_kernel(file_path, base_url=const.SPICE_BASE_URL, base_folder=const.KERNEL_ROOT):
    if isinstance(file_path, str):
        file_path = Path(file_path)

    local_path = base_folder / file_path
    url = base_url + file_path.as_posix()

    # Create necessary sub-directories in the DL_PATH direction
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # If the file is not present in the download directory -> download it
        if not os.path.isfile(local_path):
            print(f"Downloading {url}", end="  ")
            # Download the file with the urllib  package
            urllib.request.urlretrieve(str(url), str(local_path))
            print("Done.")
        else:
            print(f"{base_folder / file_path} already exists!")
    except urllib.error.HTTPError as e:
        print(f"Error: \n{url} could not be found: ", e)


def setup_spice(kernels: List[str]):
    kernels = list(map(Path, kernels))

    for k in kernels:
        download_kernel(k)

    spice.furnsh(list(map(lambda x: str(_KERNEL_ROOT / x), kernels)))


def get_sun_pos(date):
    et = spice.str2et(str(date))
    return spice.spkpos('sun', et, 'iau_moon', 'lt+s', 'moon')


def get_earth_pos(date):
    et = spice.str2et(str(date))
    return spice.spkpos('earth', et, 'iau_moon', 'lt+s', 'moon')
