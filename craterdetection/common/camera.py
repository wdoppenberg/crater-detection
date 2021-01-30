from collections.abc import Iterable

import numpy as np


def camera_matrix(fov, offset=0, alpha=0):
    """Returns camera matrix [1] from Field-of-View, skew, and offset.

    Parameters
    ----------
    fov : float, Iterable
        Field-of-View angle (radians), if type is Iterable it will be interpreted as (fov_x, fov_y)
    offset : float, Iterable
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
        f_x, f_y = map(lambda fov_, offset_: offset_ / np.tan(np.radians(fov_) / 2), zip(fov, (x_0, y_0)))
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
