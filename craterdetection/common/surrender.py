from collections import Iterable

import numpy as np
from surrender.geometry import vec4, gaussian
from surrender.surrender_client import surrender_client

import craterdetection.common.constants as const


def setup_renderer(fov=const.CAMERA_FOV, raytracing=False, preview_mode=True, resolution=const.CAMERA_RESOLUTION):
    # Constants:
    sun_radius = 696342000
    earth_radius = 6371e3
    ua2km = 149597870.700
    ua = ua2km * 1e3

    # Image setup:
    raytracing = raytracing
    N = resolution
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
    s.setImageSize(*N)

    s.setSunPower(ua * ua * np.pi * vec4(1, 1, 1, 1))

    return s
