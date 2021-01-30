from collections.abc import Iterable

import cv2
import numpy as np
import numpy.linalg as LA

from .coordinates import ENU_system
from .camera import camera_matrix, projection_matrix


def crater_camera_homography(p, P_MC):
    """Calculate homography necessary to project crater rim into camera reference frame.

    .. math:: \mathbf{H}_{C_i} =  ^\mathcal{M}\mathbf{P}_\mathcal{C} [[H_{M_i}], [k^T]]

    Parameters
    ----------
    p : np.ndarray
        (Nx)3x1 position vector of craters.
    P_MC : np.ndarray
        3x4 projection matrix from selenographic frame to camera pixel frame.

    Returns
    -------
        (Nx)3x3 homography matrix
    """
    S = np.concatenate((np.identity(2), np.zeros((1, 2))), axis=0)
    k = np.array([0, 0, 1])[:, None]

    e_i, n_i, u_i = ENU_system(p)

    T_EM = np.concatenate((e_i, n_i, u_i), axis=-1)

    H_Mi = np.concatenate((T_EM @ S, p), axis=-1)
    return P_MC @ np.concatenate((H_Mi, np.tile(k.T[None, ...], (len(H_Mi), 1, 1))), axis=1)


def project_craters(C, p, fov, resolution, T_MC, r_M):
    """Project crater conics into digital pixel frame. See pages 17 - 25 from [1] for methodology.

    Parameters
    ----------
    C : np.ndarray
        Nx3x3 array of crater conics
    p : np.ndarray
        Nx3x1 position vector of craters.
    fov : float, Iterable
        Field-of-View angle (radians), if type is Iterable it will be interpreted as (fov_x, fov_y)
    resolution : int, Iterable
        Image resolution, if type is Iterable it will be interpreted as (res_x, res_y)
    T_MC : np.ndarray
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

    if isinstance(resolution, Iterable):
        offset = map(lambda x: x/2, resolution)
    else:
        offset = resolution/2

    K = camera_matrix(fov, offset)
    P_MC = projection_matrix(K, T_MC, r_M)
    H_Ci = crater_camera_homography(p, P_MC)
    return LA.inv(H_Ci).transpose(0, 2, 1) @ C @ LA.inv(H_Ci)
