from functools import partial
from typing import Iterable

import numpy as np
import numpy.linalg as LA
from numba import njit
from sklearn.linear_model import RANSACRegressor

import src.common.constants as const
from common.conics import ConicProjector, ellipse_axes, scale_det, conic_center
from detection.metrics import gaussian_angle_distance
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


def model_validator_(model, *args, min_alt=10, max_alt=1000, primary_body_radius=const.RMOON):
    return (primary_body_radius + min_alt) < LA.norm(model.coef_) < (primary_body_radius + max_alt)


def derive_position_ransac(A_craters,
                           r_craters,
                           T,
                           K,
                           return_inlier_mask=False,
                           **ransac_kwargs
                           ):
    """Derive position using detected craters in the image plane with matched catalogue information. The spacecraft
    attitude is assumed known (some error margin is possible). Uses RANSAC to find 'optimal' S/C position for which the
    projection of the catalogue craters matches the detections. For methodology see p. 69 from [1].

    Parameters
    ----------
    A_craters : np.ndarray
        Nx3x3 array of detected crater conics in image plane in matrix representation
    r_craters : np.ndarray
        Nx3x1 array of vectors defining matched crater center location in selenographic frame
    T : np.ndarray
        3x3 array describing the current spacecraft attitude
    K : np.ndarray
        3x3 camera calibration matrix
    return_inlier_mask : bool
        Whether to return inlier mask for (A_craters, r_craters)
    ransac_kwargs:
        Keyword arguments for RANSACRegressor initialisation

    Returns
    -------
    est_pos : np.ndarray
        3x1 position of spacecraft in selenographic frame.

    References
    ----------
    .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. https://arxiv.org/abs/2009.01228
    """
    kwargs = dict(residual_threshold=1.,
                  stop_n_inliers=30,
                  max_trials=10000,
                  loss='squared_loss',
                  is_model_valid=model_validator_)

    kwargs.update(ransac_kwargs)

    if len(A_craters) < kwargs['stop_n_inliers']:
        if return_inlier_mask:
            return None, np.zeros(0, dtype=np.bool)
        else:
            return None

    S = np.concatenate((np.identity(2), np.zeros((1, 2))), axis=0)

    B_craters = T @ K.T @ A_craters @ K @ LA.inv(T)
    T_EM = np.concatenate(ENU_system(r_craters), axis=-1)
    T_ME = LA.inv(T_EM)

    A_ = S.T @ T_ME @ B_craters
    A = A_.reshape(-1, 3)
    b_ = S.T @ T_ME @ B_craters @ r_craters
    b = b_.reshape(-1, 1)

    ransac = RANSACRegressor(**kwargs)
    ransac.fit(A, b)

    inlier_mask = np.logical_and.reduce(ransac.inlier_mask_.reshape(b_.shape), axis=(-1, -2))
    num_inliers = inlier_mask.sum()
    est_pos = ransac.estimator_.coef_.T

    if num_inliers >= ransac.stop_n_inliers:
        if return_inlier_mask:
            return est_pos, inlier_mask
        else:
            return est_pos
    else:
        if return_inlier_mask:
            return None, inlier_mask
        else:
            return None


def query_position_ransac(A_detections: np.ndarray,
                          database,
                          T: np.ndarray,
                          K: np.ndarray,
                          batch_size: int = 1000,
                          top_n: int = 1,
                          sigma_pix: float = 3.,
                          return_num_matches: bool = False,
                          max_distance: float = 0.1,
                          use_reprojection: bool = True,
                          stop_n_inliers_fraction: float = 0.25,
                          **ransac_kwargs
                          ):
    if len(A_detections) < 4:
        if return_num_matches:
            return None, 0
        else:
            return None

    # Generate matchable features to query the database index with
    crater_triads, key = next(CoplanarInvariants.match_generator(
        A_craters=A_detections,
        max_iter=1,
        batch_size=batch_size
    ))

    crater_triads = np.repeat(crater_triads[:, None, :], top_n, axis=1)
    # A_db_ = np.repeat(A_detections[crater_triads][:, None, ...], top_n, axis=1).reshape(-1, 3, 3)

    match_idxs = database.query(key, k=top_n)

    dist = np.abs((key[:, None, :] - database._features[match_idxs]) / key[:, None, :]).mean(-1)
    dist_filter = dist < max_distance
    r_query, C_query = database[match_idxs[dist_filter]]
    A_query = A_detections[crater_triads[dist_filter]]

    A_query, r_query, C_query = A_query.reshape(-1, 3, 3), r_query.reshape(-1, 3, 1), C_query.reshape(-1, 3, 3)

    if stop_n_inliers_fraction is not None:
        ransac_kwargs.update({"stop_n_inliers": max(int(len(A_query) * stop_n_inliers_fraction), 10)})

    est_pos, inlier_mask = derive_position_ransac(A_craters=A_query, r_craters=r_query, T=T, K=K, return_inlier_mask=True,
                                                  **ransac_kwargs)

    if est_pos is None:
        if return_num_matches:
            return None, 0
        else:
            return None

    if use_reprojection:

        C_inlier, r_inlier = C_query.reshape(-1, 3, 3)[inlier_mask], r_query.reshape(-1, 3, 1)[inlier_mask]

        projector = ConicProjector(position=est_pos, attitude=T)

        A_projected = projector.project_crater_conics(C_inlier, r_inlier)
        A_matched = A_query[inlier_mask]

        divergence = gaussian_angle_distance(A_projected, A_matched)

        a_i, b_i = ellipse_axes(A_projected)

        sigma = (0.85 / np.sqrt(a_i * b_i)) * sigma_pix

        reprojection_mask = ((divergence / sigma) ** 2) <= 13.276

        if reprojection_mask.sum() < 3:
            if return_num_matches:
                return None, 0
            else:
                return None

        est_pos, inlier_mask = derive_position_ransac(A_craters=A_matched[reprojection_mask],
                                                      r_craters=r_inlier[reprojection_mask], T=T, K=K,
                                                      **ransac_kwargs
                                                      )
    if return_num_matches:
        return est_pos, inlier_mask.sum()
    else:
        return est_pos


def derive_position_lsq(A_craters, r_craters, C_craters, T_CM, K, use_scale=False):
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
    .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. https://arxiv.org/abs/2009.01228
    """
    k = np.array([0., 0., 1.])[:, None]
    S = np.concatenate((np.identity(2), np.zeros((1, 2))), axis=0)

    B_craters = T_CM @ K.T @ A_craters @ K @ LA.inv(T_CM)
    T_EM = np.concatenate(ENU_system(r_craters), axis=-1)
    T_ME = LA.inv(T_EM)

    A = (S.T @ T_ME @ B_craters).reshape(-1, 3)

    if use_scale:
        scale_i = (vec(S.T @ C_craters @ S).transpose((0, 2, 1)) @ vec(S.T @ T_ME @ B_craters @ T_EM @ S)) \
                  / (vec(S.T @ C_craters @ S).transpose((0, 2, 1)) @ vec(S.T @ C_craters @ S))
        b = (S.T @ T_ME @ B_craters @ r_craters - scale_i * S.T @ C_craters @ k).reshape(-1, 1)
    else:
        b = (S.T @ T_ME @ B_craters @ r_craters).reshape(-1, 1)

    Q, R = LA.qr(A)
    Qb = np.dot(Q.T, b)
    est_pos = LA.solve(R, Qb)

    # TODO: Implement check to resolve the case when the estimated position is below the Moon's surface.

    return est_pos


@njit
def pos_lsq_broadcast(A: np.ndarray, b: np.ndarray):
    out = np.empty((A.shape[0], 3, 1))

    for ii in range(A.shape[0]):
        Q, R = LA.qr(A[ii])
        Qb = np.dot(Q.T, b[ii])
        out[ii] = LA.solve(R, Qb)
    return out


def remove_outliers(estimations: np.ndarray, max_deviation: float = 3):
    if len(estimations) <= 3:
        return None
    idxs = np.arange(len(estimations))
    np.random.shuffle(idxs)
    for idx in idxs:
        close = LA.norm(estimations[idx] - estimations, axis=(1, 2)) < max_deviation
        close = LA.norm(estimations[close].mean(0) - estimations, axis=(1, 2)) < max_deviation
        if close.sum() > np.round(0.3*len(estimations)):
            return np.where(close)
    return None


def query_position_lsq(A_detections: np.ndarray,
                       database,
                       T: np.ndarray,
                       K: np.ndarray,
                       batch_size: int = 1000,
                       top_n: int = 3,
                       sigma_pix: float = 3,
                       max_deviation: float = 3,
                       max_alt: float = 500,
                       primary_body_radius: float = const.RMOON,
                       return_num_matches: bool = False
                       ):

    if len(A_detections) < 4:
        if return_num_matches:
            return None, 0
        else:
            return None

    # Generate matchable features to query the database index with
    crater_triads, key = next(CoplanarInvariants.match_generator(
        A_craters=A_detections,
        max_iter=1,
        batch_size=batch_size
    ))

    if len(crater_triads) < batch_size:
        batch_size = len(crater_triads)

    # Get top-k matches w.r.t. index
    min_n = database.query(key, k=top_n)

    A_match = A_detections[crater_triads]
    r_match, C_match = map(partial(np.moveaxis, source=1, destination=0), database[min_n])

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

        norms = LA.norm(match_est_pos, axis=(-2, -1))

        mask = np.logical_and.reduce(((d / sigma) ** 2) <= 13.276, axis=1) & \
               (primary_body_radius < norms) & \
               (norms < (primary_body_radius + max_alt))

        confirmations[i] = mask

    n_idx, b_idx = np.where(confirmations)

    if len(n_idx) == 0:
        if return_num_matches:
            return None, 0
        else:
            return None

    ransac_idx = remove_outliers(position_store[n_idx, b_idx], max_deviation=max_deviation)

    if ransac_idx is None:
        if return_num_matches:
            return None, 0
        else:
            return None

    # est_r = derive_position_lsq(A_projected_store[n_idx, b_idx][ransac_idx].reshape(-1, 3, 3),
    #                         r_match[n_idx, b_idx][ransac_idx].reshape(-1, 3, 1),
    #                         C_match[n_idx, b_idx][ransac_idx].reshape(-1, 3, 3),
    #                         T,
    #                         K)

    if return_num_matches:
        return position_store[n_idx, b_idx][ransac_idx].mean(0), len(position_store[n_idx, b_idx][ransac_idx])
    else:
        return position_store[n_idx, b_idx][ransac_idx].mean(0)
