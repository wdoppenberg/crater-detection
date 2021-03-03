from pathlib import Path

"""
Project constants
"""

TRIAD_RADIUS = 200  # Create triangles with every crater within this TRIAD_RADIUS [km]
RMOON = 1737.1  # Body radius (moon) [km]
DIAMLIMS = [4, 30]  # Limit dataset to craters with diameter between 4 and 30 km
MAX_ELLIPTICITY = 1.1  # Limit dataset to craters with an ellipticity <= 1.1]
CAMERA_FOV = 45  # Camera field-of-view (degrees)
CAMERA_RESOLUTION = [256, 256]
DB_CAM_ALTITUDE = 300


SPICE_BASE_URL = 'https://naif.jpl.nasa.gov/pub/naif/'
KERNEL_ROOT = Path('data/spice_kernels')
