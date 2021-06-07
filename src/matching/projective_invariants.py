import numpy as np
import numpy.linalg as LA

from src.common.conics import matrix_adjugate, scale_det, conic_matrix, conic_center
from src.matching.utils import np_swap_columns, is_colinear, is_clockwise, enhanced_pattern_shifting, \
    shift_nd


class PermutationInvariant:
    """
    Namespace for permutation invariants functions
    """

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
        """Three-pair cyclic permutation invariant function.

        Parameters
        ----------
        x, y, z : int or float or np.ndarray
            Values to generate cyclic permutation invariant features for

        Returns
        -------
        np.ndarray
            Array containing cyclic permutation invariants F1, F2, F3

        References
        ----------
        .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. https://arxiv.org/abs/2009.01228
        """
        return np.array((cls.F1(x, y, z), cls.F2(x, y, z), cls.F3(x, y, z)))

    @staticmethod
    def G1(x1, y1, z1, x2, y2, z2):
        return 1.5 * (x1 * x2 + y1 * y2 + z1 * z2) - 0.5 * (x1 + y1 + z1) * (x2 + y2 + z2)

    @staticmethod
    def G2(x1, y1, z1, x2, y2, z2):
        return (np.sqrt(3) / 2.) * ((x1 * z2 + y1 * x2 + z1 * y2) - (x1 * y2 + y1 * z2 + z1 * x2))

    # TODO: Fix cyclic permutation invariant G
    @classmethod
    def G(cls, x1, y1, z1, x2, y2, z2):
        return np.array((cls.G1(x1, y1, z1, x2, y2, z2), cls.G2(x1, y1, z1, x2, y2, z2)))

    @classmethod
    def G_tilde(cls, x1, y1, z1, x2, y2, z2):
        """

        Parameters
        ----------
        x1, y1, z1 : int or float or np.ndarray
            First set of values to generate cyclic permutation invariant features for
        x2, y2, z2 : int or float or np.ndarray
            Second set of values to generate cyclic permutation invariant features for

        Returns
        -------
        np.ndarray
            Array containing cyclic permutation invariants G1, G2

        References
        ----------
        .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. https://arxiv.org/abs/2009.01228

        """
        return cls.G(x1, y1, z1, x2, y2, z2) / np.sqrt(np.sqrt(
            (((x1 - y1) ** 2 + (y1 - z1) ** 2 + (z1 - x1) ** 2) * ((x2 - y2) ** 2 + (y2 - z2) ** 2 + (z2 - x2) ** 2))))


# TODO: Refactor
class CoplanarInvariants:
    def __init__(self, crater_triads, A_i, A_j, A_k, normalize_det=True):
        """Generates projective invariants [1] assuming craters are coplanar. Input is an array of crater matrices
        such as those generated using L{conic_matrix}.

        Parameters
        ----------
        crater_triads : np.ndarray
            Crater triad indices (nx3) for slicing arrays
        A_i : np.ndarray
            Crater representation first crater in triad
        A_j : np.ndarray
            Crater representation second crater in triad
        A_k : np.ndarray
            Crater representation third crater in triad
        normalize_det : bool
            Set to True to normalize matrices to achieve det(A) = 1

        References
        ----------
        .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. https://arxiv.org/abs/2009.01228
        """

        self.crater_triads = crater_triads
        overlapped_craters = ~np.logical_and.reduce((
            np.isfinite(LA.cond(A_i - A_j)),
            np.isfinite(LA.cond(A_i - A_k)),
            np.isfinite(LA.cond(A_j - A_k))
        ))
        A_i, A_j, A_k = map(lambda craters_: craters_[~overlapped_craters], (A_i, A_j, A_k))
        self.crater_triads = self.crater_triads[~overlapped_craters]

        if normalize_det:
            A_i, A_j, A_k = map(scale_det, (A_i, A_j, A_k))

        self.I_ij, self.I_ji = np.trace(LA.inv(A_i) @ A_j, axis1=-2, axis2=-1), np.trace(LA.inv(A_j) @ A_i, axis1=-2,
                                                                                         axis2=-1)
        self.I_ik, self.I_ki = np.trace(LA.inv(A_i) @ A_k, axis1=-2, axis2=-1), np.trace(LA.inv(A_k) @ A_i, axis1=-2,
                                                                                         axis2=-1)
        self.I_jk, self.I_kj = np.trace(LA.inv(A_j) @ A_k, axis1=-2, axis2=-1), np.trace(LA.inv(A_k) @ A_j, axis1=-2,
                                                                                         axis2=-1)
        self.I_ijk = np.trace((matrix_adjugate(A_j + A_k) - matrix_adjugate(A_j - A_k)) @ A_i, axis1=-2, axis2=-1)

    @classmethod
    def from_detection_conics(cls,
                              A_craters,
                              crater_triads=None
                              ):

        if crater_triads is None:
            n_det = len(A_craters)
            n_comb = int((n_det * (n_det - 1) * (n_det - 2)) // 6)

            crater_triads = np.zeros((n_comb, 3), int)
            for it, (i, j, k) in enumerate(enhanced_pattern_shifting(n_det)):
                crater_triads[it] = np.array([i, j, k])

        r_pix = conic_center(A_craters)
        x_pix = r_pix[:, 0]
        y_pix = r_pix[:, 1]

        x_triads, y_triads = x_pix[crater_triads].T, y_pix[crater_triads].T
        clockwise = is_clockwise(x_triads, y_triads)

        crater_triads_cw = crater_triads.copy()
        crater_triads_cw[~clockwise] = np_swap_columns(crater_triads[~clockwise])
        x_triads[:, ~clockwise] = np_swap_columns(x_triads.T[~clockwise]).T
        y_triads[:, ~clockwise] = np_swap_columns(y_triads.T[~clockwise]).T

        crater_triads_cw = crater_triads_cw[~is_colinear(x_triads, y_triads)]

        A_i, A_j, A_k = np.array(list(map(lambda vertex: A_craters[vertex], crater_triads_cw.T)))

        return cls(crater_triads_cw, A_i, A_j, A_k)

    @classmethod
    def match_generator(cls,
                        A_craters=None,
                        x_pix=None,
                        y_pix=None,
                        a_pix=None,
                        b_pix=None,
                        psi_pix=None,
                        batch_size=1,
                        convert_to_radians=True,
                        max_iter=10000,
                        sort_ij=True
                        ):
        """Generator function that yields crater triad and its associated projective invariants [1]. Triads are formed
        using Enhanced Pattern Shifting method [2]. Input craters can either be parsed as  parameterized ellipses
        (x_pix, y_pix, a_pix, b_pix, psi_pix) or as matrix representation of conic.

        Parameters
        ----------
        A_craters : np.ndarray
            Crater detections in conic representation.
        x_pix, y_pix : np.ndarray
            Crater center positions in image plane
        a_pix, b_pix : np.ndarray
            Crater ellipse axis parameters in image plane
        psi_pix : np.ndarray
            Crater ellipse angle w.r.t. x-direction in image plane
        batch_size : int
            Return single detection feature, or create a batch for array-values
        convert_to_radians : bool
            Whether to convert psi to radians inside method (default: True)
        max_iter : int
            Maximum iterations (default: 10000)
        sort_ij : bool
            Whether to sort triad features with I_ij being the lowest absolute value

        Yields
        ------
        crater_triad : np.ndarray
            Triad indices (1x3)
        CoplanarInvariants
             Associated CoplanarInvariants instance

        References
        ----------
        .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. https://arxiv.org/abs/2009.01228
        .. [2] Arnas, D., Fialho, M. A. A., & Mortari, D. (2017). Fast and robust kernel generators for star trackers. Acta Astronautica, 134 (August 2016), 291–302. https://doi.org/10.1016/j.actaastro.2017.02.016
        """

        if A_craters is not None:
            pass
        elif all(x is not None for x in (x_pix, y_pix, a_pix, b_pix, psi_pix)):
            if convert_to_radians:
                psi_pix = np.radians(psi_pix)

            A_craters = conic_matrix(a_pix, b_pix, psi_pix, x_pix, y_pix)
        else:
            raise ValueError("No detections provided! Use either parameterized ellipse values or conics as input.")

        n_det = len(A_craters)
        if batch_size == 1:
            for it, (i, j, k) in enumerate(enhanced_pattern_shifting(n_det)):

                crater_triad = np.array([i, j, k])
                r_pix = conic_center(A_craters[crater_triad])
                x_pix = r_pix[:, 0]
                y_pix = r_pix[:, 1]

                if not is_clockwise(x_pix, y_pix):
                    crater_triad[[0, 1]] = crater_triad[[1, 0]]

                A_i, A_j, A_k = np.array(list(map(lambda vertex: A_craters[vertex], crater_triad)))
                out = cls(crater_triad[None, :], A_i, A_j, A_k)
                key = out.get_pattern()

                if sort_ij:
                    ij_idx = np.abs(key[:3]).argmin()
                    order = np.roll(np.arange(3), -ij_idx)
                    order_full = np.append(np.concatenate((order, order + 3)), -1)

                    yield crater_triad[order], key[order_full]
                else:
                    yield crater_triad, key

                if it >= max_iter:
                    break

        elif batch_size > 1:
            n_comb = int((n_det * (n_det - 1) * (n_det - 2)) // 6)

            if batch_size > n_comb:
                batch_size = n_comb

            crater_triads = np.zeros((batch_size, 3), int)
            eps_generator = enhanced_pattern_shifting(n_det)
            for it in range(n_comb // batch_size):
                for index, (i, j, k) in enumerate(eps_generator):
                    crater_triads[index] = np.array([i, j, k])
                    if index >= batch_size - 1:
                        break

                r_pix = conic_center(A_craters)
                x_pix = r_pix[:, 0]
                y_pix = r_pix[:, 1]

                x_triads, y_triads = x_pix[crater_triads].T, y_pix[crater_triads].T
                clockwise = is_clockwise(x_triads, y_triads)

                crater_triads_cw = crater_triads.copy()
                crater_triads_cw[~clockwise] = np_swap_columns(crater_triads[~clockwise])
                x_triads[:, ~clockwise] = np_swap_columns(x_triads.T[~clockwise]).T
                y_triads[:, ~clockwise] = np_swap_columns(y_triads.T[~clockwise]).T

                crater_triads_cw = crater_triads_cw[~is_colinear(x_triads, y_triads)]

                A_i, A_j, A_k = np.array(list(map(lambda vertex: A_craters[vertex], crater_triads_cw.T)))
                out = cls(crater_triads_cw, A_i, A_j, A_k)
                key = out.get_pattern()

                if sort_ij:
                    ij_idx = np.abs(key[..., :3]).argmin(1)
                    key = np.concatenate((
                        shift_nd(key[..., :3], -ij_idx),
                        shift_nd(key[..., 3:6], -ij_idx),
                        key[:, [-1]]
                    ),
                        axis=-1
                    )
                    crater_triads_sorted = shift_nd(out.crater_triads, -ij_idx)

                    yield crater_triads_sorted, key

                else:
                    yield out.crater_triads, key

                if it >= max_iter:
                    break
        else:
            raise ValueError("batch_size must be 1 or more!")

    def get_pattern(self, permutation_invariant=False):
        """Get matching pattern using either permutation invariant features (eq. 134 from [1]) or raw projective
        invariants (p. 61 from [1]).

        Parameters
        ----------
        permutation_invariant : bool
            Set this to True if permutation_invariants are needed.

        Returns
        -------
        np.ndarray
            Array of features linked to the crater triads generated during initialisation.

        References
        ----------
        .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. https://arxiv.org/abs/2009.01228
        """

        if permutation_invariant:
            out = np.column_stack((
                PermutationInvariant.F(self.I_ij, self.I_jk, self.I_ki).T,
                PermutationInvariant.F1(self.I_ji, self.I_kj, self.I_ik),
                PermutationInvariant.G_tilde(self.I_ij, self.I_jk, self.I_ki, self.I_ji, self.I_kj, self.I_ik).T,
                self.I_ijk
            )).T

        else:
            out = np.column_stack((
                self.I_ij,
                self.I_jk,
                self.I_ki,
                self.I_ji,
                self.I_kj,
                self.I_ik,
                self.I_ijk,
            ))

        if len(self) == 1:
            return out.squeeze()
        else:
            return out

    def __len__(self):
        return len(self.I_ij)
