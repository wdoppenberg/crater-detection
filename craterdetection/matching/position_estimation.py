import numpy as np
import numpy.linalg as LA

from craterdetection.common.coordinates import ENU_system


def vec(arr):
    """Returns column vector of input array. Works for 2D & 3D arrays.

    Parameters
    ----------
    arr: np.ndarray
        (Mx)NxK array

    Returns
    -------
    np.ndarray
    """
    if len(arr.shape) == 3:
        return arr.reshape(-1, np.multiply(*arr.shape[-2:]), 1)
    elif len(arr.shape) == 2:
        return arr.reshape(-1, np.multiply(*arr.shape))
    else:
        raise ValueError("Input shape must be 2- or 3-dimensional array")


def derive_position(A_craters, r_craters, C_craters, T_CM, K, use_scale=False):
    """Derive position using detected craters in the image plane with matched catalogue information. The spacecraft
    attitude is assumed known (some error margin is possible). Uses least-squares (with QR-decomposition) to find
    'optimal' S/C position for which the projection of the catalogue craters matches the detections. For methodology
    see p. 69 from [1].

    Parameters
    ----------
    A_craters : np.ndarray
        Nx3x3 array of detected crater conics in image plane in matrix representation
    r_craters : np.ndarray
        Nx3x1 array of vectors defining matched crater location in selenographic frame
    C_craters : np.ndarray
        Nx3x3 array of catalogue crater conics in their respective ENU plane in matrix representation
    T_CM : np.ndarray
        3x3 array describing the current spacecraft attitude
    K : np.ndarray
        3x3 camera calibration matrix
    use_scale : bool
        Set to True if catalogue craters are not generated with their center being in the origin of their respective
        ENU plane. This option is present for performance benefits.

    Returns
    -------
    est_pos : np.ndarray
        3x1 position of spacecraft in selenographic frame.

    References
    ----------
    .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. http://arxiv.org/abs/2009.01228
    """
    k = np.array([0., 0., 1.])[:, None]
    S = np.concatenate((np.identity(2), np.zeros((1, 2))), axis=0)

    B_craters = T_CM @ K.T @ A_craters @ K @ LA.inv(T_CM)
    T_EM = np.concatenate(ENU_system(r_craters), axis=-1)
    T_ME = LA.inv(T_EM)

    A = (S.T @ T_ME @ B_craters).reshape(-1, 3)

    if use_scale:
        scale_i = (vec(S.T @ C_craters @ S).transpose(0, 2, 1) @ vec(S.T @ T_ME @ B_craters @ T_EM @ S)) \
                  / (vec(S.T @ C_craters @ S).transpose(0, 2, 1) @ vec(S.T @ C_craters @ S))
        b = (S.T @ T_ME @ B_craters @ r_craters - scale_i * S.T @ C_craters @ k).reshape(-1, 1)
    else:
        b = (S.T @ T_ME @ B_craters @ r_craters).reshape(-1, 1)

    Q, R = LA.qr(A)
    Qb = np.dot(Q.T, b)
    est_pos = LA.solve(R, Qb)

    # TODO: Implement check to resolve the case when the estimated position is below the Moon's surface.

    return est_pos
