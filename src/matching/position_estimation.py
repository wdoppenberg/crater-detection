from functools import partial

import numpy as np
import numpy.linalg as LA
from numba import njit

import src.common.constants as const
from src.common.conics import scale_det, conic_center, ellipse_axes
from src.common.coordinates import ENU_system
from src.matching import CoplanarInvariants


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


@njit
def pos_lsq_broadcast(A, b):
    out = np.empty((A.shape[0], 3, 1))

    for ii in range(A.shape[0]):
        Q, R = LA.qr(A[ii])
        Qb = np.dot(Q.T, b[ii])
        out[ii] = LA.solve(R, Qb)
    return out


def calculate_position(A_detections,
                       db,
                       T,
                       K,
                       batch_size=1000,
                       top_n=3,
                       sigma_pix=1,
                       max_matched_triads=30,
                       max_alt=500,
                       return_all_positions=False,
                       filter_outliers=False
                       ):
    # top_n = [top_n] if top_n == 1 else top_n

    # Generate matchable features to query the database index with
    crater_triads, key = next(CoplanarInvariants.match_generator(
        A_craters=A_detections,
        max_iter=1,
        batch_size=batch_size
    ))

    if len(crater_triads) < batch_size:
        batch_size = len(crater_triads)

    # Get top-k matches w.r.t. index
    min_n = db.query(key, k=top_n)

    A_match = A_detections[crater_triads]
    r_match, C_match = map(partial(np.moveaxis, source=1, destination=0), db[min_n])

    S = np.concatenate((np.identity(2), np.zeros((1, 2))), axis=0)
    k = np.array([0., 0., 1.])[:, None]

    B_craters = T @ K.T @ A_match @ K @ LA.inv(T)

    confirmations = np.full((top_n, batch_size), False)
    A_projected_store = np.empty((top_n, batch_size, 3, 3, 3))

    position_store = np.empty((top_n, batch_size, 3, 1))

    for i, (r, C) in enumerate(zip(r_match, C_match)):
        T_EM = np.concatenate(ENU_system(r), axis=-1)
        T_ME = LA.inv(T_EM)

        A = (S.T @ T_ME @ B_craters).reshape(-1, 6, 3)
        b = (S.T @ T_ME @ B_craters @ r).reshape(-1, 6, 1)
        match_est_pos = pos_lsq_broadcast(A, b)

        position_store[i] = match_est_pos

        H_Mi = np.concatenate((np.concatenate(ENU_system(r), axis=-1) @ S, r), axis=-1)
        P_MC = K @ LA.inv(T) @ np.concatenate((np.tile(np.identity(3), (len(match_est_pos), 1, 1)), -match_est_pos),
                                              axis=2)
        H_C = P_MC[:, None, ...] @ np.concatenate((H_Mi, np.tile(k.T[None, ...], (len(H_Mi), 3, 1, 1))), axis=-2)
        A_projected = LA.inv(H_C.transpose(0, 1, 3, 2)) @ C @ LA.inv(H_C)

        A_projected_store[i] = A_projected

        Y_i = -scale_det(A_projected)[..., :2, :2]
        Y_j = -scale_det(A_match)[..., :2, :2]

        y_i = np.expand_dims(conic_center(A_projected), axis=-1)
        y_j = np.expand_dims(conic_center(A_match), axis=-1)

        d = np.arccos(
            (4 * np.sqrt(LA.det(Y_i) * LA.det(Y_j)) / (LA.det(Y_i + Y_j))) \
            * np.exp(-0.5 * (y_i - y_j).transpose(0, 1, 3, 2) @ Y_i @ LA.inv(Y_i + Y_j) @ Y_j @ (y_i - y_j)).squeeze()
        )

        a_i, b_i = ellipse_axes(A_projected)

        sigma = (0.85 / np.sqrt(a_i * b_i)) * sigma_pix

        mask = np.logical_and(
            np.logical_and.reduce(((d / sigma) ** 2) <= 13.276, axis=1),
            LA.norm(match_est_pos, axis=(-2, -1)) < const.RMOON + max_alt,
            LA.norm(match_est_pos, axis=(-2, -1)) > const.RMOON
        )

        confirmations[i] = mask

    n_idx, b_idx = np.where(confirmations)

    if len(n_idx) == 0:
        return np.full((3, 1), -1)

    est_r = derive_position(A_projected_store[n_idx, b_idx].reshape(-1, 3, 3),
                            r_match[n_idx, b_idx].reshape(-1, 3, 1),
                            C_match[n_idx, b_idx].reshape(-1, 3, 3),
                            T,
                            K)

    if filter_outliers:
        inliers = np.logical_and.reduce(
            (position_store[n_idx, b_idx] - est_r) < 1.5 * np.std(position_store[n_idx, b_idx], axis=0),
            axis=1
        ).squeeze()

        if np.sum(inliers) == 0:
            return np.full((3, 1), -1)

        est_r = derive_position(A_projected_store[n_idx, b_idx][inliers].reshape(-1, 3, 3),
                                r_match[n_idx, b_idx][inliers].reshape(-1, 3, 1),
                                C_match[n_idx, b_idx][inliers].reshape(-1, 3, 3),
                                T,
                                K)

    if return_all_positions:
        return est_r, position_store[n_idx, b_idx]
    else:
        return est_r
