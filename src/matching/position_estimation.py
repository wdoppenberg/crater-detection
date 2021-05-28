from functools import partial
from typing import Iterable

import numpy as np
import numpy.linalg as LA
from numba import njit
from sklearn.linear_model import RANSACRegressor

import src.common.constants as const
from common.conics import ConicProjector, ellipse_axes
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


def derive_position(A_craters,
                    r_craters,
                    T,
                    K,
                    residual_threshold=1.,
                    min_inliers=30,
                    max_trials=10000,
                    model_validator_fn=model_validator_,
                    altitude_range=None,
                    return_inlier_mask=False
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
    residual_threshold : float
        Residual threshold for RANSACRegressor
    min_inliers : int
        Minimum number of inliers before match is considered correct for RANSACRegressor
    max_trials : int
        Maximum number of RANSAC trials to perform for RANSACRegressor
    model_validator_fn : callable
        Callable with arguments (model, X, y) which validates the model fit for RANSACRegressor
    altitude_range : tuple, list
        Altitude range to consider for solution (live model validation)
    return_inlier_mask : bool
        Whether to return the mask for the inliers

    Returns
    -------
    est_pos : np.ndarray
        3x1 position of spacecraft in selenographic frame.

    References
    ----------
    .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. https://arxiv.org/abs/2009.01228
    """
    S = np.concatenate((np.identity(2), np.zeros((1, 2))), axis=0)

    B_craters = T @ K.T @ A_craters @ K @ LA.inv(T)
    T_EM = np.concatenate(ENU_system(r_craters), axis=-1)
    T_ME = LA.inv(T_EM)

    A_ = S.T @ T_ME @ B_craters
    A = A_.reshape(-1, 3)
    b_ = S.T @ T_ME @ B_craters @ r_craters
    b = b_.reshape(-1, 1)

    if altitude_range is not None:
        model_validator_fn = partial(model_validator_fn, min_alt=min(altitude_range), max_alt=max(altitude_range))

    ransac = RANSACRegressor(residual_threshold=residual_threshold,
                             stop_n_inliers=min_inliers,
                             max_trials=max_trials,
                             loss='squared_loss',
                             is_model_valid=model_validator_fn
                             )
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


def calculate_position(A_detections: np.ndarray,
                       database,
                       T: np.ndarray,
                       K: np.ndarray,
                       batch_size: int = 10000,
                       top_n: int = 1,
                       min_inliers: int = None,
                       min_inliers_fraction: float = 0.25,
                       sigma_pix: float = 3.,
                       max_trials: int = 10000,
                       residual_threshold: float = 1.,
                       altitude_range: Iterable = None,
                       return_num_matches: bool = False,
                       max_distance: float = 500,
                       use_reprojection: bool = True
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

    A_db_ = np.repeat(A_detections[crater_triads][:, None, ...], top_n, axis=1).reshape(-1, 3, 3)

    match_idxs, dist = map(lambda x: x.reshape(-1), database.query(key, k=top_n, return_distance=True))

    r_query_, C_query_ = database[match_idxs]
    r_query_, C_query_ = r_query_.reshape(-1, 3, 1), C_query_.reshape(-1, 3, 3)
    _, unique_mapping = np.unique(match_idxs, return_index=True)

    if max_distance is None:
        max_distance = 0.2*LA.norm(np.repeat(key, top_n, axis=0), axis=1)[unique_mapping]
    dist_filter = dist.ravel()[unique_mapping] < max_distance

    A_db, r_query, C_query = map(lambda x: x[unique_mapping][dist_filter], (A_db_, r_query_, C_query_))

    if min_inliers is None:
        min_inliers = max(int(len(A_db) * min_inliers_fraction), 10)

    if len(A_db) < min_inliers:
        if return_num_matches:
            return None, 0
        else:
            return None

    est_pos, inlier_mask = derive_position(A_craters=A_db, r_craters=r_query, T=T, K=K, min_inliers=min_inliers,
                                                  max_trials=max_trials, residual_threshold=residual_threshold,
                                                  return_inlier_mask=True, altitude_range=altitude_range)
    if est_pos is None:
        if return_num_matches:
            return None, 0
        else:
            return None

    if use_reprojection:

        C_inlier, r_inlier = C_query.reshape(-1, 3, 3)[inlier_mask], r_query.reshape(-1, 3, 1)[inlier_mask]

        projector = ConicProjector(position=est_pos, attitude=T)

        A_projected = projector.project_crater_conics(C_inlier, r_inlier)
        A_matched = A_db[inlier_mask]

        divergence = gaussian_angle_distance(A_projected, A_matched)

        a_i, b_i = ellipse_axes(A_projected)

        sigma = (0.85 / np.sqrt(a_i * b_i)) * sigma_pix

        reprojection_mask = ((divergence / sigma) ** 2) <= 13.276

        if reprojection_mask.sum() < 3:
            if return_num_matches:
                return None, 0
            else:
                return None

        est_pos, inlier_mask = derive_position(A_craters=A_matched[reprojection_mask],
                                                       r_craters=r_inlier[reprojection_mask], T=T, K=K,
                                                       min_inliers=3, max_trials=1000,
                                                       return_inlier_mask=True, altitude_range=altitude_range
                                                       )
    if return_num_matches:
        return est_pos, inlier_mask.sum()
    else:
        return est_pos
