import datetime as dt
from collections import Iterable

import numpy as np
from scipy.spatial.transform import Rotation
from surrender.geometry import vec4, gaussian, quat, vec3
from surrender.surrender_client import surrender_client

import src.common.constants as const
from .camera import Camera
from .coordinates import suborbital_coords
from .spice import get_sun_pos, get_earth_pos, setup_spice, get_sol_incidence

SUN_RADIUS = 696342e3
EARTH_RADIUS = 6371e3
UA2KM = 149597870.700
UA = UA2KM * 1e3


def setup_renderer(
            fov=const.CAMERA_FOV,
            raytracing=False,
            preview_mode=False,
            resolution=const.CAMERA_RESOLUTION,
            hostname='127.0.0.1',
            port=5151
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
    s.connectToServer(hostname, port=port)
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


class SurRenderer(Camera):
    """
    SurRender software wrapper for Lunar scene generation with accurately generated Sun & Earth positions.
    """

    def __init__(self,
                 DEM_filename="FullMoon.dem",
                 texture_filename="lroc_color_poles.tiff",
                 scene_time=dt.datetime(2021, 1, 1),
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.scene_time = scene_time
        setup_spice()
        self.__setup_backend(DEM_filename=DEM_filename, texture_filename=texture_filename)
        self.__sync_backend()

    def __setup_backend(self,
                        DEM_filename="FullMoon.dem",
                        texture_filename="lroc_color_poles.tiff"):
        self.backend = setup_renderer(fov=self.fov, resolution=self.resolution)

        self.backend.createBRDF('sun', 'sun.brdf', {})
        self.backend.createShape('sun', 'sphere.shp', {'radius': SUN_RADIUS})
        self.backend.createBody('sun', 'sun', 'sun', [])
        self.backend.setObjectPosition('sun', self.sun_pos * 1e3)

        self.backend.createBRDF("mate", "mate.brdf", {})
        self.backend.createShape("earth_shape", "sphere.shp", {'radius': EARTH_RADIUS})
        self.backend.createBody("earth", "earth_shape", "mate", ["earth.jpg"])
        self.backend.setObjectPosition('earth', self.earth_pos * 1e3)

        self.backend.createBRDF('hapke', 'hapke.brdf', {})
        self.backend.createSphericalDEM('moon', DEM_filename, 'hapke', texture_filename)
        self.backend.setObjectElementBRDF('moon', 'moon', 'hapke')
        self.backend.setObjectAttitude('moon', np.array([0, 0, 0, 1]))

        self.backend.setObjectPosition('moon', (0, 0, 0))
        self.backend.setObjectAttitude('moon', quat(vec3(1, 0, 0), 0))
        self.backend.setObjectAttitude('moon', Rotation.from_euler('z', np.pi, degrees=False).as_quat())

    def __sync_backend(self):
        """
        Synchronise Renderer state with backend.

        Currently, a change in the resolution attribute is not propagated back to the backend after initialization
        as it breaks the renderer.
        """
        # TODO: Fix resolution adjustment.

        self.backend.setCameraFOVDeg(*self.fov)
        # self.backend.setImageSize(*self.resolution)
        self.backend.setObjectPosition('camera', self.position.ravel() * 1e3)
        self.backend.setObjectAttitude('camera', Rotation.from_matrix(self.attitude).as_quat())
        self.backend.setObjectPosition('earth', self.earth_pos * 1e3)
        self.backend.setObjectPosition('sun', self.sun_pos * 1e3)

    @property
    def scene_time(self):
        return self.__scene_time

    @scene_time.setter
    def scene_time(self, scene_time):
        self.__scene_time = scene_time

    @property
    def sun_pos(self):
        return get_sun_pos(self.scene_time)

    @property
    def earth_pos(self):
        return get_earth_pos(self.scene_time)

    @property
    def solar_incidence_angle(self):
        return get_sol_incidence(self.scene_time, self.suborbital_position().squeeze())

    def generate_image(self, stretch=True) -> np.ndarray:
        self.__sync_backend()

        self.backend.render()
        out = self.backend.getImageGray32F()

        max_pixel = out.max()

        if stretch and max_pixel > 0:
            return out / max_pixel
        else:
            return out

    def __del__(self):
        del self.backend
