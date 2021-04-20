import numpy as np
import numpy.linalg as LA

import src.common.constants as const


def ENU_system(r):
    """Return local East-North-Up (ENU) coordinate system for point defined by p.

    Using the coordinate system defined using:

    .. math::
        \mathbf{u}_i = \mathbf{p}^{(c)}_{M_i}/||\mathbf{p}^{(c)}_{M_i}||

        \mathbf{e}_i = cross(\mathbf{k}, \mathbf{u}_i )/|| cross(\mathbf{k}, \mathbf{u}_i) ||

        \mathbf{n}_i = cross(\mathbf{u}_i, \mathbf{e}_i)/|| cross(\mathbf{u}_i, \mathbf{e}_i) ||

    with

    .. math::
        \mathbf{k} = [0 & 0 & 1]^T

    and :math:`p_{Mi}` is the selenographic 3D cartesian coordinate derived from latitude & longitude.

    Parameters
    ----------
    r : np.ndarray
        (Nx)3x1 vector that defines origin.

    Returns
    -------
    e_i, n_i, u_i : np.ndarray
        Normalized i, j, k components of coordinate system.

    """
    k = np.array([0, 0, 1])[:, None]

    if len(r.shape) == 4:
        u_i = r / LA.norm(r, ord=2, axis=(-2, -1))[:, None, :, None]

        e_i = np.cross(k[None, ...], r, axis=-2)
        e_i /= LA.norm(e_i, ord=2, axis=(-2, -1))[:, None, :, None]

        n_i = np.cross(r, e_i, axis=-2)
        n_i /= LA.norm(n_i, ord=2, axis=(-2, -1))[:, None, :, None]

    elif len(r.shape) == 3:
        u_i = r / LA.norm(r, ord=2, axis=(1, 2))[:, None, None]

        e_i = np.cross(k[None, ...], r, axis=1)
        e_i /= LA.norm(e_i, ord=2, axis=(1, 2))[:, None, None]

        n_i = np.cross(r, e_i, axis=1)
        n_i /= LA.norm(n_i, ord=2, axis=(1, 2))[:, None, None]
    else:
        u_i = r / LA.norm(r, ord=2)

        e_i = np.cross(k, r, axis=0)
        e_i /= LA.norm(e_i, ord=2)

        n_i = np.cross(r, e_i, axis=0)
        n_i /= LA.norm(n_i, ord=2)

    return e_i, n_i, u_i


def nadir_attitude(r):
    """Return nadir-pointing (z-axis) coordinate system for point defined by r in world reference frame. X- and
    Y-components are defined by East and South respectively.

    Parameters
    ----------
    r : np.ndarray
        (Nx)3x1 vector that defines origin.

    Returns
    -------
    e_i, n_i, d_i : np.ndarray
        Normalized i, j, k components of coordinate system.

    """
    k = np.array([0, 0, 1])[:, None]

    if len(r.shape) == 3:
        d_i = -(r / LA.norm(r, ord=2, axis=(1, 2))[:, None, None])

        e_i = np.cross(k[None, ...], -d_i, axis=1)
        e_i /= LA.norm(e_i, ord=2, axis=(1, 2))[:, None, None]

        s_i = np.cross(d_i, e_i, axis=1)
        s_i /= LA.norm(s_i, ord=2, axis=(1, 2))[:, None, None]
    elif len(r.shape) == 2:
        d_i = -(r / LA.norm(r))

        e_i = np.cross(k, r, axis=0)
        e_i /= LA.norm(e_i, ord=2)

        s_i = np.cross(d_i, e_i, axis=0)
        s_i /= LA.norm(s_i, ord=2)
    else:
        raise ValueError(f"Input shape is invalid! -> {r.shape}")

    return e_i, s_i, d_i


def suborbital_coords(r, R_body=const.RMOON):
    """Return coordinates directly below orbital position.

    Parameters
    ----------
    r : np.ndarray
        Position above body (e.g. Moon)
    R_body : np.ndarray
        Radius of body in km, defaults to const.RMOON

    Returns
    -------
    np.ndarray
        Suborbital coordinates
    """
    return (r / LA.norm(r)) * R_body
