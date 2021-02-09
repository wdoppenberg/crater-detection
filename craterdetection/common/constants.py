"""
Project constants
"""

TRIAD_RADIUS = 200  # Create triangles with every crater within this TRIAD_RADIUS [km]
RBODY = 1737.1  # Body radius (moon) [km]
DIAMLIMS = [4, 30]  # Limit dataset to craters with diameter between 4 and 30 km
MAX_ELLIPTICITY = 1.1  # Limit dataset to craters with an ellipticity <= 1.1]
CAMERA_FOV = 45  # Camera field-of-view (degrees)
CAMERA_RESOLUTION = [1000, 1000]
DB_CAM_ALTITUDE = 300