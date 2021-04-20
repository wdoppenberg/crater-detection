from abc import ABC
from collections import Iterable
from typing import Union

import numpy as np
import datetime as dt
from scipy.spatial.transform import Rotation
from surrender.geometry import vec4, gaussian, quat, vec3
from surrender.surrender_client import surrender_client

import constants as const
from .camera import Renderer
from .spice import get_sun_pos, get_earth_pos

SUN_RADIUS = 696342e3
EARTH_RADIUS = 6371e3
UA2KM = 149597870.700
UA = UA2KM * 1e3


def setup_renderer(
            fov=const.CAMERA_FOV,
            raytracing=False,
            preview_mode=True,
            resolution=const.CAMERA_RESOLUTION
        ):
    # Image setup:
    raytracing = raytracing
    rays = 64

    # set PSF
    surech_PSF = 5
    sigma = 1
    wPSF = 5
    PSF = gaussian(wPSF * surech_PSF, sigma * surech_PSF)

    # Initializing SurRender
    s = surrender_client()
    s.setVerbosityLevel(2)
    s.connectToServer('127.0.0.1')
    s.setCompressionLevel(0)
    s.closeViewer()
    s.setTimeOut(86400)
    s.setShadowMapSize(4096)
    s.setCubeMapSize(4096)
    s.enableMultilateralFiltering(True)
    s.enablePreviewMode(preview_mode)
    s.enableDoublePrecisionMode(True)
    s.enableRaytracing(raytracing)
    s.setConventions(s.XYZ_SCALAR_CONVENTION, s.Z_FRONTWARD)
    s.setPSF(PSF, wPSF, wPSF)

    if raytracing:
        s.enableFastPSFMode(False)
        s.enableRaytracing(True)
        s.enableIrradianceMode(False)
        s.setNbSamplesPerPixel(rays)  # Raytracing
        s.enableRegularPSFSampling(True)
        s.enablePathTracing(False)
    else:
        s.enableFastPSFMode(True)
        s.enableRaytracing(False)
        s.enableIrradianceMode(False)
        s.enablePathTracing(True)

    if not isinstance(fov, Iterable):
        fov = (fov, fov)

    s.setCameraFOVDeg(*fov)
    s.setImageSize(*resolution)

    s.setSunPower(UA * UA * np.pi * vec4(1, 1, 1, 1))

    return s


class SurRenderer(Renderer):
    def __init__(self,
                 r,
                 T=None,
                 fov=const.CAMERA_FOV,
                 resolution=const.CAMERA_RESOLUTION,
                 DEM_filename="FullMoon.dem",
                 texture_filename="lroc_color_poles.tiff",
                 datetime=dt.datetime(2021, 1, 1)
                 ):
        super().__init__(r=r, T=T, fov=fov, resolution=resolution)
        self.backend = setup_renderer(fov=fov, resolution=resolution)

        self.backend.createBRDF('sun', 'sun.brdf', {})
        self.backend.createShape('sun', 'sphere.shp', {'radius': SUN_RADIUS})
        self.backend.createBody('sun', 'sun', 'sun', [])

        self.backend.createBRDF("mate", "mate.brdf", {})
        self.backend.createShape("earth_shape", "sphere.shp", {'radius': EARTH_RADIUS})
        self.backend.createBody("earth", "earth_shape", "mate", ["earth.jpg"])

        self.backend.createBRDF('hapke', 'hapke.brdf', {})
        self.backend.createSphericalDEM('moon', DEM_filename, 'hapke', texture_filename)
        self.backend.setObjectElementBRDF('moon', 'moon', 'hapke')
        self.backend.setObjectAttitude('moon', np.array([0, 0, 0, 1]))

        self.backend.setObjectPosition('moon', (0, 0, 0))
        self.backend.setObjectAttitude('moon', quat(vec3(1, 0, 0), 0))
        self.backend.setObjectAttitude('moon', Rotation.from_euler('z', np.pi, degrees=False).as_quat())

        self.datetime = datetime

        self.sun_pos, _ = get_sun_pos(self.datetime)
        self.earth_pos, _ = get_earth_pos(self.datetime)

        self.backend.setObjectPosition('earth', self.earth_pos * 1e3)
        self.backend.setObjectPosition('sun', self.sun_pos * 1e3)

    def set_position(self, r: np.ndarray):
        super(SurRenderer, self).set_position(r)

        self.backend.setObjectPosition('camera', self._r.ravel() * 1e3)

    def set_orientation(self, T: Union[np.ndarray, Rotation]):
        super(SurRenderer, self).set_orientation(T)

        self.backend.setObjectAttitude('camera', Rotation.from_matrix(self._T).as_quat())

    def set_resolution(self, resolution):
        super(SurRenderer, self).set_resolution(resolution)

        self.backend.setImageSize(*self._resolution)

    def set_fov(self, fov):
        super(SurRenderer, self).set_fov(fov)

        self.backend.setCameraFOVRad(*self._fov)

    def set_datetime(self, datetime):
        self.datetime = datetime

        self.sun_pos, _ = get_sun_pos(self.datetime)
        self.earth_pos, _ = get_earth_pos(self.datetime)

        self.backend.setObjectPosition('earth', self.earth_pos * 1e3)
        self.backend.setObjectPosition('sun', self.sun_pos * 1e3)

    def get_image(self):
        self.backend.render()
        return self.backend.getImageGray32F()
