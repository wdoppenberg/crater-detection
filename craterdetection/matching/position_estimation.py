import numpy as np
import numpy.linalg as LA

from craterdetection.common.coordinates import ENU_system


def vectorize(arr):
    return arr.reshape(-1, np.multiply(*arr.shape[-2:]), 1)


def derive_position(A_craters, r_craters, C_craters, T_CM, K, use_scale=False):
    k = np.array([0., 0., 1.])[:, None]
    S = np.concatenate((np.identity(2), np.zeros((1, 2))), axis=0)

    B_craters = T_CM @ K.T @ A_craters @ K @ LA.inv(T_CM)
    T_EM = np.concatenate(ENU_system(r_craters), axis=-1)
    T_ME = LA.inv(T_EM)

    A = (S.T @ T_ME @ B_craters).reshape(-1, 3)
    if use_scale:
        scale_i = (vectorize(S.T @ C_craters @ S).transpose(0, 2, 1) @ vectorize(S.T @ T_ME @ B_craters @ T_EM @ S)) \
                  / (vectorize(S.T @ C_craters @ S).transpose(0, 2, 1) @ vectorize(S.T @ C_craters @ S))
        b = (S.T @ T_ME @ B_craters @ r_craters - scale_i * S.T @ C_craters @ k).reshape(-1, 1)
    else:
        b = (S.T @ T_ME @ B_craters @ r_craters).reshape(-1, 1)

    Q, R = LA.qr(A)
    Qb = np.dot(Q.T, b)
    return LA.solve(R, Qb)
