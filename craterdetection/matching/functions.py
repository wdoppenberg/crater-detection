import numpy as np
import numpy.linalg as LA


def cyclic_permutations(it):
    yield it
    for k in range(1, len(it)):
        p = it[k:] + it[:k]
        if p == it:
            break
        yield p


def matrix_adjugate(matrix, normalised=True):
    if normalised:
        return LA.inv(matrix)
    else:
        return (LA.inv(matrix).T * LA.det(matrix)).T


def scale_det(A, n=3):
    # rescale matrix such that det(A) = 1
    # np.cbrt: cube root
    if len(A.shape) == 2:
        return np.cbrt(1. / LA.det(A)) * A
    elif len(A.shape) == 3:
        return np.cbrt((1. / LA.det(A)).reshape(np.shape(A)[0], 1, 1)) * A
    else:
        raise ValueError("Input must be nxn matrix of kxnxn array of matrices.")


def crater_representation(x, y, a, b, psi):
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
