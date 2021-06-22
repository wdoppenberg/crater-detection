import warnings

import numpy as np
import numpy.linalg as LA
from numba import njit
from scipy.optimize import least_squares
from sklearn.linear_model import RANSACRegressor

import src.common.constants as const
from src.common.conics import ConicProjector, ellipse_axes
from src.common.coordinates import ENU_system
from src.detection.metrics import gaussian_angle_distance


def _model_validator(min_alt=30, max_alt=500, primary_body_radius=const.RMOON):
    return lambda model, *args: (primary_body_radius + min_alt) < LA.norm(model.coef_) < (primary_body_radius + max_alt)


class PositionRegressor:
    def __init__(self, sigma_pix=6, **ransac_kwargs):
        kwargs = dict(residual_threshold=0.01,
                      max_trials=1000,
                      is_model_valid=_model_validator()
                      )

        kwargs.update(ransac_kwargs)

        self.ransac = RANSACRegressor(**kwargs)
        self.sigma_pix = sigma_pix
        self.inlier_mask = None
        self.est_pos_ransac = None
        self._projector = ConicProjector()
        self.reprojection_mask = np.empty(0, dtype=bool)
        self.est_pos_verified = None
        self.optimize_result = None

    def fit(self, A_query, C_query, r_query, attitude, camera_matrix, reprojection=True):

        if len(A_query) > 4:
            S = np.concatenate((np.identity(2), np.zeros((1, 2))), axis=0)

            B_craters = attitude @ camera_matrix.T @ A_query @ camera_matrix @ LA.inv(attitude)
            T_EM = np.concatenate(ENU_system(r_query), axis=-1)
            T_ME = LA.inv(T_EM)

            X_ = S.T @ T_ME @ B_craters
            X = X_.reshape(-1, 3)
            y_ = S.T @ T_ME @ B_craters @ r_query
            y = y_.reshape(-1, 1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.ransac.fit(X / np.mean(y), y / np.mean(y))

            self.est_pos_ransac = self.ransac.estimator_.coef_.T

            self.inlier_mask = np.logical_and.reduce(self.ransac.inlier_mask_.reshape(y_.shape), axis=(-1, -2))

            self.est_pos_verified = None
            if self.num_inliers > 2 and reprojection and self.ransac.is_model_valid(self.ransac.estimator_):
                C_inlier, r_inlier = C_query[self.inlier_mask], r_query[self.inlier_mask]

                self._projector.attitude = attitude
                self._projector.position = self.est_pos_ransac

                A_projected = self._projector.project_crater_conics(C_inlier, r_inlier)
                A_matched = A_query[self.inlier_mask]

                divergence = gaussian_angle_distance(A_projected, A_matched)
                a_i, b_i = ellipse_axes(A_projected)
                sigma = (0.85 / np.sqrt(a_i * b_i)) * self.sigma_pix
                self.reprojection_mask = ((divergence / sigma) ** 2) <= 13.276

                if self.reprojection_mask.sum() > 2:
                    X_verified = A_matched[self.reprojection_mask]
                    r_verified = r_query[self.inlier_mask][self.reprojection_mask]

                    B_verified = attitude @ camera_matrix.T @ X_verified @ camera_matrix @ LA.inv(attitude)
                    T_EM = np.concatenate(ENU_system(r_verified), axis=-1)
                    T_ME = LA.inv(T_EM)

                    X_verified = (S.T @ T_ME @ B_verified).reshape(-1, 3)
                    y_verified = (S.T @ T_ME @ B_verified @ r_verified).reshape(-1, 1)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        self.optimize_result = least_squares(
                            lambda x: ((y_verified / np.mean(y_verified)) -
                                       (X_verified / np.mean(y_verified)) @ x[:, None]).squeeze(),
                            x0=self.est_pos_ransac.squeeze()
                        )
                        self.est_pos_verified = self.optimize_result.x[:, None]

    @property
    def num_inliers(self):
        if self.inlier_mask is None:
            return 0
        else:
            return self.inlier_mask.sum()

    @property
    def num_verified(self):
        if self.reprojection_mask is None:
            return 0
        else:
            return self.reprojection_mask.sum()

    def ransac_match(self):
        return self.num_inliers > 2 and self.est_pos_ransac is not None

    def reprojection_match(self):
        return self.reprojection_mask.sum() > 2 and self.est_pos_verified is not None


# https://www.researchgate.net/publication/259143595_Simultaneous_spacecraft_orbit_estimation_and_control_based_on_GPS_measurements_via_extended_Kalman_filter
@njit
def systems_dynamics_matrix(x_state: np.ndarray) -> np.ndarray:
    mu_moon = 0.00490e6  # km^3 / s^2
    A = np.zeros((6, 6))
    A[:3, 3:] = np.identity(3)

    X: float = x_state[0, 0]
    Y: float = x_state[1, 0]
    Z: float = x_state[2, 0]
    r = LA.norm(x_state[:3], ord=2)

    f1_rtx = mu_moon * (3 * X ** 2 - r ** 2) / (r ** 5)
    f1_rty = mu_moon * (3 * X * Y) / (r ** 5)
    f1_rtz = mu_moon * (3 * X * Z) / (r ** 5)

    f2_rtx = f1_rty
    f2_rty = mu_moon * (3 * Y ** 2 - r ** 2) / (r ** 5)
    f2_rtz = mu_moon * (3 * Y * Z) / (r ** 5)

    f3_rtx = f1_rtz
    f3_rty = f2_rtz
    f3_rtz = mu_moon * (3 * Z ** 2 - r ** 2) / (r ** 5)

    A[3, :3] = np.array([f1_rtx, f1_rty, f1_rtz])
    A[4, :3] = np.array([f2_rtx, f2_rty, f2_rtz])
    A[5, :3] = np.array([f3_rtx, f3_rty, f3_rtz])

    return A


def Hx(x_state: np.ndarray) -> np.ndarray:
    return HJacobian(x_state) @ x_state


def HJacobian(*args) -> np.ndarray:
    measurement_matrix = np.zeros((3, 6))
    measurement_matrix[:3, :3] = np.identity(3)
    return measurement_matrix
