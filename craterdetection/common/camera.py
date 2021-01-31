from collections.abc import Iterable

import numpy as np
import numpy.linalg as LA
from astropy.coordinates import spherical_to_cartesian

import craterdetection.common.constants as const


def camera_matrix(fov=const.CAMERA_FOV, offset=0, alpha=0):
    """Returns camera matrix [1] from Field-of-View, skew, and offset.

    Parameters
    ----------
    fov : float, Iterable
        Field-of-View angle (degrees), if type is Iterable it will be interpreted as (fov_x, fov_y)
    offset : int, Iterable
        Offset in pixel-frame, if type is Iterable it will be interpreted as (offset_x, offset_y)
    alpha : float
        Camera skew angle.

    Returns
    -------
    np.ndarray
        3x3 camera matrix

    References
    ----------
    .. [1] http://www.cs.ucf.edu/~mtappen/cap5415/lecs/lec19.pdf
    """

    if isinstance(offset, Iterable):
        x_0, y_0 = offset
    else:
        x_0 = offset
        y_0 = offset

    if isinstance(fov, Iterable):
        f_x, f_y = map(lambda fov_, offset_: offset_ / np.tan(np.radians(fov_) / 2), fov, (x_0, y_0))
    else:
        f_x, f_y = map(lambda offset_: offset_ / np.tan(np.radians(fov) / 2), (x_0, y_0))

    return np.array([[f_x, alpha, x_0],
                     [0, f_y, y_0],
                     [0, 0, 1]])


def projection_matrix(K, T, r):
    """Return Projection matrix [1] according to:

    .. math:: ^x\mathbf{P}_C = \mathbf{K} [ ^x\mathbf{T}_C & -r_C]

    Parameters
    ----------
    K : np.ndarray
        3x3 camera matrix
    T : np.ndarray
        3x3 attitude transformation matrix into camera frame.
    r : np.ndarray
        3x1 camera position in world reference frame

    Returns
    -------
    np.ndarray
        3x4 projection matrix

    References
    ----------
    .. [1] http://www.cs.ucf.edu/~mtappen/cap5415/lecs/lec19.pdf

    See Also
    --------
    camera_matrix

    """
    r_C = T @ r
    return K @ np.concatenate((T, -r_C), axis=1)


class Camera:
    """
    Camera data class with associated projection methods.
    """

    def __init__(self,
                 r,
                 T=None,
                 fov=const.CAMERA_FOV,
                 resolution=const.CAMERA_RESOLUTION,
                 alpha=0
                 ):
        self.r = r
        self.fov = fov
        self.resolution = resolution
        self.alpha = alpha

        if T is None:
            k = np.array([0., 0., 1.])[:, None]
            Z_ax_cam = -(r / LA.norm(r))

            X_ax_cam = np.cross(k, r, axis=0)
            X_ax_cam /= LA.norm(X_ax_cam, ord=2)

            Y_ax_cam = np.cross(Z_ax_cam, X_ax_cam, axis=0)
            Y_ax_cam /= LA.norm(Y_ax_cam, ord=2)

            self.T = np.concatenate((X_ax_cam, Y_ax_cam, Z_ax_cam), axis=-1)
            if LA.matrix_rank(self.T) != 3:
                raise ValueError("Invalid camera attitude matrix!:\n", self.T)
        else:
            self.T = T

    @classmethod
    def from_coordinates(cls,
                         lat,
                         long,
                         altitude,
                         fov,
                         resolution,
                         T=None,
                         Rbody=const.RBODY,
                         alpha=0,
                         convert_to_radians=True
                         ):
        if convert_to_radians:
            lat, long = map(np.radians, (lat, long))

        r_M = np.array(spherical_to_cartesian(Rbody + altitude, lat, long))[:, None]
        return cls(r_M, T, fov, resolution, alpha)

    def K(self):
        if isinstance(self.resolution, Iterable):
            offset = map(lambda x: x / 2, self.resolution)
        else:
            offset = self.resolution / 2
        return camera_matrix(self.fov, offset, self.alpha)

    def P(self):
        return projection_matrix(self.K(), self.T, self.r)
