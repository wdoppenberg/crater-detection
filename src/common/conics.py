from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import EllipseCollection
from numba import njit
from numpy import linalg as LA

import src.common.constants as const
from common.camera import camera_matrix, projection_matrix
from common.coordinates import ENU_system
from src.common.camera import Camera


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


def scale_det(matrix):
    """Rescale matrix such that det(A) = 1.

    Parameters
    ----------
    matrix: np.ndarray
        Matrix input
    Returns
    -------
    np.ndarray
        Normalised matrix.
    """

    return np.cbrt((1. / LA.det(matrix)))[..., None, None] * matrix


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
    if isinstance(a, (int, float)):
        out = np.empty((3, 3))
    else:
        out = np.empty((len(a), 3, 3))

    A = (a ** 2) * np.sin(psi) ** 2 + (b ** 2) * np.cos(psi) ** 2
    B = 2 * ((b ** 2) - (a ** 2)) * np.cos(psi) * np.sin(psi)
    C = (a ** 2) * np.cos(psi) ** 2 + b ** 2 * np.sin(psi) ** 2
    D = -2 * A * x - B * y
    F = -B * x - 2 * C * y
    G = A * (x ** 2) + B * x * y + C * (y ** 2) - (a ** 2) * (b ** 2)

    out[..., 0, 0] = A
    out[..., 1, 1] = C
    out[..., 2, 2] = G

    out[..., 1, 0] = B / 2
    out[..., 0, 1] = B / 2

    out[..., 2, 0] = D / 2
    out[..., 0, 2] = D / 2

    out[..., 2, 1] = F / 2
    out[..., 1, 2] = F / 2

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


def plot_conics(A_craters: Union[np.ndarray, torch.Tensor],
                resolution=const.CAMERA_RESOLUTION,
                figsize=(15, 15),
                plot_centers=False,
                ax=None,
                rim_color='r'):
    if isinstance(A_craters, torch.Tensor):
        A_craters = A_craters.numpy()

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
                  instancing=False,
                  thickness=1
                  ):
    a_proj, b_proj = map(lambda x: x / 2, ellipse_axes(A_craters))
    psi_proj = np.degrees(ellipse_angle(A_craters))
    r_pix_proj = conic_center(A_craters)

    a_proj, b_proj, psi_proj, r_pix_proj = map(lambda i: np.round(i).astype(int),
                                               (a_proj, b_proj, psi_proj, r_pix_proj))

    mask = np.zeros(resolution)

    if filled:
        thickness = -1

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


def crater_camera_homography(r_craters, P_MC):
    """Calculate homography between crater-plane and camera reference frame.

    .. math:: \mathbf{H}_{C_i} =  ^\mathcal{M}\mathbf{P}_\mathcal{C_{craters}} [[H_{M_i}], [k^T]]

    Parameters
    ----------
    r_craters : np.ndarray
        (Nx)3x1 position vector of craters.
    P_MC : np.ndarray
        (Nx)3x4 projection matrix from selenographic frame to camera pixel frame.

    Returns
    -------
        (Nx)3x3 homography matrix
    """
    S = np.concatenate((np.identity(2), np.zeros((1, 2))), axis=0)
    k = np.array([0, 0, 1])[:, None]

    H_Mi = np.concatenate((np.concatenate(ENU_system(r_craters), axis=-1) @ S, r_craters), axis=-1)

    return P_MC @ np.concatenate((H_Mi, np.tile(k.T[None, ...], (len(H_Mi), 1, 1))), axis=1)


def project_crater_conics(C_craters, r_craters, fov, resolution, T_CM, r_M):
    """Project crater conics into digital pixel frame. See pages 17 - 25 from [1] for methodology.

    Parameters
    ----------
    C_craters : np.ndarray
        Nx3x3 array of crater conics
    r_craters : np.ndarray
        Nx3x1 position vector of craters.
    fov : float, Iterable
        Field-of-View angle (radians), if type is Iterable it will be interpreted as (fov_x, fov_y)
    resolution : int, Iterable
        Image resolution, if type is Iterable it will be interpreted as (res_x, res_y)
    T_CM : np.ndarray
        3x3 matrix representing camera attitude in world reference frame
    r_M : np.ndarray
        3x1 position vector of camera

    Returns
    -------
    np.ndarray
        Nx3x3 Homography matrix H_Ci

    References
    ----------
    .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. https://arxiv.org/abs/2009.01228
    """

    K = camera_matrix(fov, resolution)
    P_MC = projection_matrix(K, T_CM, r_M)
    H_Ci = crater_camera_homography(r_craters, P_MC)
    return LA.inv(H_Ci).transpose((0, 2, 1)) @ C_craters @ LA.inv(H_Ci)


def project_crater_centers(r_craters, fov, resolution, T_CM, r_M):
    """Project crater centers into digital pixel frame.

    Parameters
    ----------
    r_craters : np.ndarray
        Nx3x1 position vector of craters.
    fov : int, float, Iterable
        Field-of-View angle (radians), if type is Iterable it will be interpreted as (fov_x, fov_y)
    resolution : int, Iterable
        Image resolution, if type is Iterable it will be interpreted as (res_x, res_y)
    T_CM : np.ndarray
        3x3 matrix representing camera attitude in world reference frame
    r_M : np.ndarray
        3x1 position vector of camera

    Returns
    -------
    np.ndarray
        Nx2x1 2D positions of craters in pixel frame
    """

    K = camera_matrix(fov, resolution)
    P_MC = projection_matrix(K, T_CM, r_M)
    H_Ci = crater_camera_homography(r_craters, P_MC)
    return (H_Ci @ np.array([0, 0, 1]) / (H_Ci @ np.array([0, 0, 1]))[:, -1][:, None])[:, :2]


class ConicProjector(Camera):
    def project_crater_conics(self, C_craters, r_craters):
        H_Ci = crater_camera_homography(r_craters, self.projection_matrix)
        return LA.inv(H_Ci).transpose((0, 2, 1)) @ C_craters @ LA.inv(H_Ci)

    def project_crater_centers(self, r_craters):
        H_Ci = crater_camera_homography(r_craters, self.projection_matrix)
        return (H_Ci @ np.array([0, 0, 1]) / (H_Ci @ np.array([0, 0, 1]))[:, -1][:, None])[:, :2]

    def generate_mask(self,
                      A_craters=None,
                      C_craters=None,
                      r_craters=None,
                      filled=False,
                      instancing=True,
                      thickness=1
                      ):

        if A_craters is None:
            if C_craters is None or r_craters is None:
                raise ValueError("Must provide either crater data in respective ENU-frame (C_craters & r_craters) "
                                 "or in image-frame (A_craters)!")

            A_craters = self.project_crater_conics(C_craters, r_craters)

        return generate_mask(A_craters=A_craters,
                             resolution=self.resolution,
                             filled=filled,
                             instancing=instancing,
                             thickness=thickness
                             )
