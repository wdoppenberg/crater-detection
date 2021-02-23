from collections.abc import Iterable

import numpy as np
import numpy.linalg as LA
from astropy.coordinates import spherical_to_cartesian

import craterdetection.common.constants as const
from craterdetection.common.coordinates import ENU_system, nadir_attitude


def camera_matrix(fov=const.CAMERA_FOV, resolution=const.CAMERA_RESOLUTION, alpha=0):
    """Returns camera matrix [1] from Field-of-View, skew, and offset.

    Parameters
    ----------
    fov : float, Iterable
        Field-of-View angle (degrees), if type is Iterable it will be interpreted as (fov_x, fov_y)
    resolution : float, Iterable
        X- and Y-resolution of the image in pixels
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

    if isinstance(resolution, Iterable):
        x_0, y_0 = map(lambda x: x/2, resolution)
    else:
        x_0 = resolution/2
        y_0 = resolution/2

    if isinstance(fov, Iterable):
        f_x, f_y = map(lambda x, fov_: x / np.tan(np.radians(fov_/2)), (x_0, y_0), fov)
    else:
        f_x, f_y = map(lambda x: x / np.tan(np.radians(fov/2)), (x_0, y_0))

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


def crater_camera_homography(p, P_MC):
    """Calculate homography between crater-plane and camera reference frame.

    .. math:: \mathbf{H}_{C_i} =  ^\mathcal{M}\mathbf{P}_\mathcal{C} [[H_{M_i}], [k^T]]

    Parameters
    ----------
    p : np.ndarray
        (Nx)3x1 position vector of craters.
    P_MC : np.ndarray
        (Nx)3x4 projection matrix from selenographic frame to camera pixel frame.

    Returns
    -------
        (Nx)3x3 homography matrix
    """
    S = np.concatenate((np.identity(2), np.zeros((1, 2))), axis=0)
    k = np.array([0, 0, 1])[:, None]

    H_Mi = np.concatenate((np.concatenate(ENU_system(p), axis=-1) @ S, p), axis=-1)

    return P_MC @ np.concatenate((H_Mi, np.tile(k.T[None, ...], (len(H_Mi), 1, 1))), axis=1)


def project_crater_conics(C, r_craters, fov, resolution, T_CM, r_M):
    """Project crater conics into digital pixel frame. See pages 17 - 25 from [1] for methodology.

    Parameters
    ----------
    C : np.ndarray
        Nx3x3 array of crater conics
    r_craters : np.ndarray
        Nx3x1 position vector of craters.
    fov : float, Iterable
        Field-of-View angle (radians), if type is Iterable it will be interpreted as (fov_x, fov_y)
    resolution : int, Iterable
        Image resolution, if type is Iterable it will be interpreted as (res_x, res_y)
    T_CM : np.ndarray
        3x3 matrix representing camera attitude in world reference frame
    r_M : np.ndarray
        3x1 position vector of camera

    Returns
    -------
    np.ndarray
        Nx3x3 Homography matrix H_Ci

    References
    ----------
    .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. http://arxiv.org/abs/2009.01228
    """

    T_MC = LA.inv(T_CM)
    K = camera_matrix(fov, resolution)
    P_MC = projection_matrix(K, T_MC, r_M)
    H_Ci = crater_camera_homography(r_craters, P_MC)
    return LA.inv(H_Ci).transpose((0, 2, 1)) @ C @ LA.inv(H_Ci)


def project_crater_centers(r_craters, fov, resolution, T_CM, r_M):
    """Project crater centers into digital pixel frame.

    Parameters
    ----------
    r_craters : np.ndarray
        Nx3x1 position vector of craters.
    fov : float, Iterable
        Field-of-View angle (radians), if type is Iterable it will be interpreted as (fov_x, fov_y)
    resolution : int, Iterable
        Image resolution, if type is Iterable it will be interpreted as (res_x, res_y)
    T_CM : np.ndarray
        3x3 matrix representing camera attitude in world reference frame
    r_M : np.ndarray
        3x1 position vector of camera

    Returns
    -------
    np.ndarray
        Nx2x1 2D positions of craters in pixel frame
    """

    T_MC = LA.inv(T_CM)
    K = camera_matrix(fov, resolution)
    P_MC = projection_matrix(K, T_MC, r_M)
    H_Ci = crater_camera_homography(r_craters, P_MC)
    return (H_Ci @ np.array([0, 0, 1]) / (H_Ci @ np.array([0, 0, 1]))[:, -1][:, None])[:, :2]


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
        if len(r.shape) == 1:
            r = r[:, None]
        elif len(r.shape) > 2:
            raise ValueError("Position vector must be 1 or 2-dimensional (3x1)!")

        self.r = r
        self.fov = fov
        self.resolution = resolution
        self.alpha = alpha

        if T is None:
            self.T = np.concatenate(nadir_attitude(r), axis=-1)

            if LA.matrix_rank(self.T) != 3:
                raise ValueError("Invalid camera attitude matrix!:\n", self.T)
        else:
            self.T = T

    @classmethod
    def from_coordinates(cls,
                         lat,
                         long,
                         altitude,
                         fov=const.CAMERA_FOV,
                         resolution=const.CAMERA_RESOLUTION,
                         T=None,
                         Rbody=const.RMOON,
                         alpha=0,
                         convert_to_radians=False
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

    def project_crater_conics(self, C, r_craters):
        return project_crater_conics(C, r_craters, self.fov, self.resolution, self.T, self.r)

    def project_crater_centers(self, r_craters):
        return project_crater_centers(r_craters, self.fov, self.resolution, self.T, self.r)
