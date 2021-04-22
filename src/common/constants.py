from pathlib import Path

"""
Project constants
"""
# Physical
RMOON = 1737.1  # Body radius (moon) [km]


# Camera
CAMERA_FOV = 45  # Camera field-of-view (degrees)
CAMERA_RESOLUTION = (256, 256)


# Dataset generation
DIAMLIMS = (4, 100)  # Limit dataset to craters with diameter between 4 and 30 km
MAX_ELLIPTICITY = 1.1  # Limit dataset to craters with an ellipticity <= 1.1]
ARC_LIMS = 0.3
AXIS_THRESHOLD = (5, 100)


# Database generation
TRIAD_RADIUS = 100  # Create triangles with every crater within this TRIAD_RADIUS [km]
DB_CAM_ALTITUDE = 300


# SPICE
SPICE_BASE_URL = 'https://naif.jpl.nasa.gov/pub/naif/'
KERNEL_ROOT = Path('data/spice_kernels')
