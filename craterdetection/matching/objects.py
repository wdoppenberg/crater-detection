from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.linalg as LA

from .functions import matrix_adjugate, scale_det


class PermutationInvariant:
    @staticmethod
    def F1(x, y, z):
        return x + y + z

    @staticmethod
    def F2(x, y, z):
        return (2 * (x ** 3 + y ** 3 + z ** 3) + 12 * x * y * z - 3 * (
                    (x ** 2) * y + (y ** 2) * x + (y ** 2) * z + (z ** 2) * y + (z ** 2) * x + (x ** 2) * z)) / \
               (x ** 2 + y ** 2 + z ** 2 - (x * y + y * z + z * x))

    @staticmethod
    def F3(x, y, z):
        return (-3 * np.sqrt(3) * (x - y) * (y - z) * (z - x)) / (x ** 2 + y ** 2 + z ** 2 - (x * y + y * z + z * x))

    @classmethod
    def F(cls, x, y, z):
        return np.array((cls.F1(x, y, z), cls.F2(x, y, z), cls.F3(x, y, z)))

    @staticmethod
    def G1(x1, y1, z1, x2, y2, z2):
        return 1.5 * (x1 * x2 + y1 * y2 + z1 * z2) - 0.5 * (x1 + y1 + z1) * (x2 + y2 + z2)

    @staticmethod
    def G2(x1, y1, z1, x2, y2, z2):
        return (np.sqrt(3) / 2.) * ((x1 * z2 + y1 * x2 + z1 * y2) - (x1 * y2 + y1 * z2 + z1 * x2))

    @classmethod
    def G(cls, x1, y1, z1, x2, y2, z2):
        return np.array((cls.G1(x1, y1, z1, x2, y2, z2), cls.G2(x1, y1, z1, x2, y2, z2)))

    @classmethod
    def G_tilde(cls, x1, y1, z1, x2, y2, z2):
        return cls.G(x1, y1, z1, x2, y2, z2) / np.sqrt(np.sqrt(
            (((x1 - y1) ** 2 + (y1 - z1) ** 2 + (z1 - x1) ** 2) * ((x2 - y2) ** 2 + (y2 - z2) ** 2 + (z2 - x2) ** 2))))


@dataclass
class CraterTriad:
    I_ij: float
    I_ji: float
    I_ik: float
    I_ki: float
    I_jk: float
    I_kj: float
    I_ijk: float
    indices: tuple

    def __sub__(self, other):
        if not isinstance(other, CraterTriad):
            raise TypeError("'-' only works on other CraterTriad instance.")
        return np.array((
            self.I_ij - other.I_ij,
            self.I_ji - other.I_ji,
            self.I_ik - other.I_ik,
            self.I_ki - other.I_ki,
            self.I_jk - other.I_jk,
            self.I_kj - other.I_kj,
            self.I_ijk - other.I_ijk
        ))


class CoplanarInvariants:
    def __init__(self, craters, normalize_det=False):
        self.crater_triads = np.array(list(combinations(range(len(craters)), 3)))

        A_i = craters[self.crater_triads[:, 0]]
        A_j = craters[self.crater_triads[:, 1]]
        A_k = craters[self.crater_triads[:, 2]]

        overlapped_craters = ~np.logical_and.reduce((
            np.isfinite(LA.cond(A_i - A_j)),
            np.isfinite(LA.cond(A_i - A_k)),
            np.isfinite(LA.cond(A_j - A_k))
        ))
        A_i, A_j, A_k = map(lambda craters_: craters_[~overlapped_craters], (A_i, A_j, A_k))
        self.crater_triads = self.crater_triads[~overlapped_craters]

        if normalize_det:
            A_i, A_j, A_k = map(scale_det, (A_i, A_j, A_k))

        self.I_ij, self.I_ji = np.trace(LA.inv(A_i) @ A_j, axis1=-1, axis2=-2), np.trace(LA.inv(A_j) @ A_i, axis1=-1,
                                                                                         axis2=-2)
        self.I_ik, self.I_ki = np.trace(LA.inv(A_i) @ A_k, axis1=-1, axis2=-2), np.trace(LA.inv(A_k) @ A_i, axis1=-1,
                                                                                         axis2=-2)
        self.I_jk, self.I_kj = np.trace(LA.inv(A_j) @ A_k, axis1=-1, axis2=-2), np.trace(LA.inv(A_k) @ A_j, axis1=-1,
                                                                                         axis2=-2)
        self.I_ijk = np.trace((matrix_adjugate(A_j + A_k) - matrix_adjugate(A_j - A_k)) @ A_i, axis1=-1, axis2=-2)

        self.pattern = self._get_pattern()

    def __getitem__(self, item):
        return CraterTriad(
            self.I_ij[item],
            self.I_ji[item],
            self.I_ik[item],
            self.I_ki[item],
            self.I_jk[item],
            self.I_kj[item],
            self.I_ijk[item],
            tuple(self.crater_triads[item])
        )

    def _get_pattern(self):
        return np.column_stack((
            PermutationInvariant.F(self.I_ij, self.I_jk, self.I_ki).T,
            PermutationInvariant.F1(self.I_ji, self.I_kj, self.I_ik),
            PermutationInvariant.G_tilde(self.I_ij, self.I_jk, self.I_ki, self.I_ji, self.I_kj, self.I_ik).T,
            self.I_ijk
        )
        ).T

    def __len__(self):
        return len(self.I_ij)
