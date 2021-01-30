import cv2
import numpy as np

from .coordinates import ENU_system


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
