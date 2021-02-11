import numpy as np
from numpy import linalg as LA


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
