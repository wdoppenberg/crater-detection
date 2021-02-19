import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

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

    if len(A.shape) == 2:
        return np.cbrt(1. / LA.det(A)) * A
    elif len(A.shape) == 3:
        return np.cbrt((1. / LA.det(A)).reshape(np.shape(A)[0], 1, 1)) * A
    else:
        raise ValueError("Input must be nxn matrix of kxnxn array of matrices.")


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

    A = (a ** 2) * np.sin(psi) ** 2 + (b ** 2) * np.cos(psi) ** 2
    B = 2 * ((b ** 2) - (a ** 2)) * np.cos(psi) * np.sin(psi)
    C = (a ** 2) * np.cos(psi) ** 2 + b ** 2 * np.sin(psi) ** 2
    D = -2 * A * x - B * y
    F = -B * x - 2 * C * y
    G = A * (x ** 2) + B * x * y + C * (y ** 2) - (a ** 2) * (b ** 2)

    return scale_det(np.array([
        [A, B / 2, D / 2],
        [B / 2, C, F / 2],
        [D / 2, F / 2, G]
    ]).T)


def conic_center(A):
    if len(A.shape) == 3:
        return (LA.inv(A[:, :2, :2]) @ -A[:, :2, 2][..., None])[..., 0]
    elif len(A.shape) == 2:
        return (LA.inv(A[:2, :2]) @ -A[:2, 2][..., None])[..., 0]
    else:
        raise ValueError("Conic (array) must be of shape (Nx)3x3!")


def ellipse_axes(A):
    if len(A.shape) == 3:
        lambdas = LA.eigvalsh(A[:, :2, :2]) / (-LA.det(A) / LA.det(A[:, :2, :2]))[:, None]
        axes = np.sqrt(1 / lambdas)
        return axes[:, 1], axes[:, 0]

    elif len(A.shape) == 2:
        lambdas = LA.eigvalsh(A[:2, :2]) / (-LA.det(A) / LA.det(A[:2, :2]))[:, None]
        axes = np.sqrt(1 / lambdas)
        return axes[1], axes[0]

    else:
        raise ValueError("Conic (array) must be of shape (Nx)3x3!")


def ellipse_angle(A):
    if len(A.shape) == 3:
        return np.pi + np.arctan(2 * A[:, 1, 0] / (A[:, 0, 0] - A[:, 1, 1])) / 2

    elif len(A.shape) == 2:
        return np.pi + np.arctan(2 * A[1, 0] / (A[0, 0] - A[1, 1])) / 2

    else:
        raise ValueError("Conic (array) must be of shape (Nx)3x3!")


def plot_conics(A,
                gridsize=250,
                resolution=const.CAMERA_RESOLUTION,
                figsize=(15, 15),
                plot_centers=False,
                ax=None,
                rim_color='r'):
    x_plot = np.linspace(0, resolution[0], gridsize)
    y_plot = np.linspace(0, resolution[1], gridsize)
    x_plot, y_plot = np.meshgrid(x_plot, y_plot)

    xy_homogeneous = np.concatenate(
        (
            x_plot.ravel()[None, :],
            y_plot.ravel()[None, :],
            np.ones_like(x_plot.ravel()[None, :])
        ),
        axis=0
    ).T[..., None]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'aspect': 'equal'})

    # Set axes according to camera pixel space convention
    ax.set_xlim(0, resolution[0])
    ax.set_ylim(resolution[1], 0)

    for a_i in A:
        c = xy_homogeneous.transpose(0, 2, 1) @ a_i @ xy_homogeneous
        ax.contour(x_plot, y_plot, c.reshape(x_plot.shape), [0], colors=rim_color)

    if plot_centers:
        crater_centers = conic_center(A)
        for k, c_i in enumerate(crater_centers):
            x, y = c_i[0], c_i[1]
            if x <= resolution[0] and y <=resolution[1]:
                ax.text(x, y, str(k))
