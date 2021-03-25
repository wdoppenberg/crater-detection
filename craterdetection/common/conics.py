import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection
from numpy import linalg as LA

from numba import njit

import craterdetection.common.constants as const


def matrix_adjugate(matrix):
    """Return adjugate matrix [1].

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix

    Returns
    -------
    np.ndarray
        Adjugate of input matrix

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Adjugate_matrix
    """

    cofactor = LA.inv(matrix).T * LA.det(matrix)
    return cofactor.T


def scale_det(A):
    """Rescale matrix such that det(A) = 1.

    Parameters
    ----------
    A: np.ndarray
        Matrix input
    Returns
    -------
    np.ndarray
        Normalised matrix.
    """

    return np.cbrt((1. / LA.det(A)))[..., None, None] * A


def crater_representation(a, b, psi, x=0, y=0):
    """Returns matrix representation for crater derived from ellipse parameters

    Parameters
    ----------
    x
        X-position in 2D cartesian coordinate system (coplanar)
    y
        Y-position in 2D cartesian coordinate system (coplanar)
    a
        Semi-major ellipse axis
    b
        Semi-minor ellipse axis
    psi
        Ellipse angle (radians)

    Returns
    -------
    np.ndarray
        Array of ellipse matrices
    """
    out = np.empty((len(a), 3, 3))

    A = (a ** 2) * np.sin(psi) ** 2 + (b ** 2) * np.cos(psi) ** 2
    B = 2 * ((b ** 2) - (a ** 2)) * np.cos(psi) * np.sin(psi)
    C = (a ** 2) * np.cos(psi) ** 2 + b ** 2 * np.sin(psi) ** 2
    D = -2 * A * x - B * y
    F = -B * x - 2 * C * y
    G = A * (x ** 2) + B * x * y + C * (y ** 2) - (a ** 2) * (b ** 2)

    out[:, 0, 0] = A
    out[:, 1, 1] = C
    out[:, 2, 2] = G

    out[:, 1, 0] = B/2
    out[:, 0, 1] = B/2

    out[:, 2, 0] = D/2
    out[:, 0, 2] = D/2

    out[:, 2, 1] = F/2
    out[:, 1, 2] = F/2

    return scale_det(out)

@njit
def conic_center_numba(A):
    a = LA.inv(A[:2, :2])
    b = np.expand_dims(-A[:2, 2], axis=-1)
    return a @ b


def conic_center(A):
    return (LA.inv(A[..., :2, :2]) @ -A[..., :2, 2][..., None])[..., 0]


def ellipse_axes(A):
    lambdas = LA.eigvalsh(A[..., :2, :2]) / (-LA.det(A) / LA.det(A[..., :2, :2]))[..., None]
    axes = np.sqrt(1 / lambdas)
    return axes[..., 1], axes[..., 0]


def ellipse_angle(A):
    return np.arctan2(2 * A[..., 1, 0], (A[..., 0, 0] - A[..., 1, 1])) / 2


def plot_conics(A_craters,
                resolution=const.CAMERA_RESOLUTION,
                figsize=(15, 15),
                plot_centers=False,
                ax=None,
                rim_color='r'):
    a_proj, b_proj = ellipse_axes(A_craters)
    psi_proj = ellipse_angle(A_craters)
    r_pix_proj = conic_center(A_craters)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'aspect': 'equal'})

    # Set axes according to camera pixel space convention
    ax.set_xlim(0, resolution[0])
    ax.set_ylim(resolution[1], 0)

    ec = EllipseCollection(a_proj, b_proj, np.degrees(psi_proj), units='xy', offsets=r_pix_proj,
                           transOffset=ax.transData, facecolors="None", edgecolors=rim_color)
    ax.add_collection(ec)

    if plot_centers:
        crater_centers = conic_center(A_craters)
        for k, c_i in enumerate(crater_centers):
            x, y = c_i[0], c_i[1]
            if 0 <= x <= resolution[0] and 0 <= y <= resolution[1]:
                ax.text(x, y, str(k))


def generate_mask(A_craters,
                  resolution=const.CAMERA_RESOLUTION,
                  filled=False,
                  instancing=False
                  ):
    a_proj, b_proj = map(lambda x: x / 2, ellipse_axes(A_craters))
    psi_proj = np.degrees(ellipse_angle(A_craters))
    r_pix_proj = conic_center(A_craters)

    a_proj, b_proj, psi_proj, r_pix_proj = map(lambda i: np.round(i).astype(np.int),
                                               (a_proj, b_proj, psi_proj, r_pix_proj))

    mask = np.zeros(resolution)

    if filled:
        thickness = -1
    else:
        thickness = 1

    if instancing:
        for i, (a, b, x, y, psi) in enumerate(zip(a_proj, b_proj, *r_pix_proj.T, psi_proj)):
            if a >= 1 and b >= 1:
                mask = cv2.ellipse(mask,
                                   (x, y),
                                   (a, b),
                                   psi,
                                   0,
                                   360,
                                   i,
                                   thickness)
    else:
        for a, b, x, y, psi in zip(a_proj, b_proj, *r_pix_proj.T, psi_proj):
            if a >= 1 and b >= 1:
                mask = cv2.ellipse(mask,
                                   (x, y),
                                   (a, b),
                                   psi,
                                   0,
                                   360,
                                   1,
                                   thickness)

    return mask
