import os
import urllib
from pathlib import Path

import numpy as np
import spiceypy as spice

import src.common.constants as const

KERNELS = tuple(map(Path, (
    'generic_kernels/lsk/naif0012.tls',
    'generic_kernels/pck/pck00010.tpc',
    'pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/spk/de421.bsp',
    'pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/pck/moon_pa_de421_1900_2050.bpc',
    'pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/fk/moon_assoc_pa.tf',
    'pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/fk/moon_080317.tf'
)))


def download_kernel(file_path, base_url=const.SPICE_BASE_URL, base_folder=const.KERNEL_ROOT, verbose=False):
    if isinstance(file_path, str):
        file_path = Path(file_path)

    local_path = base_folder / file_path
    url = base_url + file_path.as_posix()

    # Create necessary sub-directories in the DL_PATH direction
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # If the file is not present in the download directory -> download it
        if not os.path.isfile(local_path):
            if verbose:
                print(f"Downloading {url}")
            # Download the file with the urllib  package
            urllib.request.urlretrieve(str(url), str(local_path))
        else:
            if verbose:
                print(f"{base_folder / file_path} already exists!")
    except urllib.error.HTTPError as e:
        print(f"Error: \n{url} could not be found: ", e)


def setup_spice(kernels=None, base_url=const.SPICE_BASE_URL, base_folder=const.KERNEL_ROOT):
    if kernels is None:
        kernels = KERNELS
    kernels = tuple(map(Path, kernels))

    for k in kernels:
        download_kernel(k, base_url, base_folder)

    spice.furnsh(list(map(lambda x: str((const.KERNEL_ROOT / x).resolve()), kernels)))


def get_sun_pos(date):
    et = spice.str2et(str(date))
    out, _ = spice.spkpos('sun', et, 'iau_moon', 'lt+s', 'moon')
    return out


def get_earth_pos(date):
    et = spice.str2et(str(date))
    out, _ = spice.spkpos('earth', et, 'iau_moon', 'lt+s', 'moon')
    return out


def get_sol_incidence(date, r):
    et = spice.str2et(str(date))
    _, _, _, sol_incidence, _ = spice.ilumin('ellipsoid', 'moon', et, 'iau_moon', 'lt+s', 'sun', r)
    return np.degrees(sol_incidence)
