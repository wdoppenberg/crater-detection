import numpy as np
import numpy.linalg as LA


def ENU_system(p):
    """Return local East-North-Up (ENU) coordinate system for point defined by p.

    Parameters
    ----------
    p : np.ndarray
        (Nx)3x1 vector that defines point.

    Returns
    -------
    e_i, n_i, u_i : np.ndarray
        Normalized i, j, k components of coordinate system.

    """
    k = np.array([0, 0, 1])[:, None]

    if len(p.shape) > 1:
        u_i = p / LA.norm(p, ord=2, axis=(1, 2))[:, None, None]

        e_i = np.cross(k[None, ...], u_i, axis=1)
        e_i /= LA.norm(e_i, ord=2, axis=(1, 2))[:, None, None]

        n_i = np.cross(u_i, e_i, axis=1)
        n_i /= LA.norm(n_i, ord=2, axis=(1, 2))[:, None, None]
    else:
        u_i = p / LA.norm(p, ord=2)

        e_i = np.cross(k, u_i, axis=0)
        e_i /= LA.norm(e_i, ord=2)

        n_i = np.cross(u_i, e_i, axis=0)
        n_i /= LA.norm(n_i, ord=2)

    return e_i, n_i, u_i
