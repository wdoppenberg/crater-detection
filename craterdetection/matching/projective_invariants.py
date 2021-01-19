import numpy as np
import numpy.linalg as LA

from craterdetection.matching.utils import np_swap_columns, is_colinear, is_clockwise, all_clockwise, \
    cyclic_permutations, enhanced_pattern_shifting


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

    cofactor = np.linalg.inv(matrix).T * np.linalg.det(matrix)
    return cofactor.T


def scale_det(A):
    """Rescale matrix such that det(A) = 1.

    Parameters
    ----------
    A: np.ndarray
        Matrix input
    Returns
    -------
    np.ndarray
        Normalised matrix.
    """

    if len(A.shape) == 2:
        return np.cbrt(1. / LA.det(A)) * A
    elif len(A.shape) == 3:
        return np.cbrt((1. / LA.det(A)).reshape(np.shape(A)[0], 1, 1)) * A
    else:
        raise ValueError("Input must be nxn matrix of kxnxn array of matrices.")


def crater_representation(x, y, a, b, psi):
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

    A = (a ** 2) * np.sin(psi) ** 2 + (b ** 2) * np.cos(psi) ** 2
    B = 2 * ((b ** 2) - (a ** 2)) * np.cos(psi) * np.sin(psi)
    C = (a ** 2) * np.cos(psi) ** 2 + b ** 2 * np.sin(psi) ** 2
    D = -2 * A * x - B * y
    F = -B * x - 2 * C * y
    G = A * (x ** 2) + B * x * y + C * (y ** 2) - (a ** 2) * (b ** 2)

    return scale_det(np.array([
        [A, B / 2, D / 2],
        [B / 2, C, F / 2],
        [D / 2, F / 2, G]
    ]).T)


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
        .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. http://arxiv.org/abs/2009.01228
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
        .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. http://arxiv.org/abs/2009.01228

        """
        return cls.G(x1, y1, z1, x2, y2, z2) / np.sqrt(np.sqrt(
            (((x1 - y1) ** 2 + (y1 - z1) ** 2 + (z1 - x1) ** 2) * ((x2 - y2) ** 2 + (y2 - z2) ** 2 + (z2 - x2) ** 2))))


class CoplanarInvariants:
    def __init__(self, crater_triads, A_i, A_j, A_k, normalize_det=True):
        """Generates projective invariants [1] assuming craters are coplanar. Input is an array of crater matrices
        such as those generated using L{crater_representation}.

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
        .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. http://arxiv.org/abs/2009.01228
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

        self.I_ij, self.I_ji = np.trace(LA.inv(A_i) @ A_j, axis1=-1, axis2=-2), np.trace(LA.inv(A_j) @ A_i, axis1=-1,
                                                                                         axis2=-2)
        self.I_ik, self.I_ki = np.trace(LA.inv(A_i) @ A_k, axis1=-1, axis2=-2), np.trace(LA.inv(A_k) @ A_i, axis1=-1,
                                                                                         axis2=-2)
        self.I_jk, self.I_kj = np.trace(LA.inv(A_j) @ A_k, axis1=-1, axis2=-2), np.trace(LA.inv(A_k) @ A_j, axis1=-1,
                                                                                         axis2=-2)
        self.I_ijk = np.trace((matrix_adjugate(A_j + A_k) - matrix_adjugate(A_j - A_k)) @ A_i, axis1=-1, axis2=-2)

    @classmethod
    def from_detection(cls,
                       x_pix,
                       y_pix,
                       a_pix,
                       b_pix,
                       psi_pix,
                       convert_to_radians=True,
                       crater_triads=None
                       ):

        if convert_to_radians:
            psi_pix = np.radians(psi_pix)

        if crater_triads is None:
            n_det = len(x_pix)
            n_comb = int((n_det * (n_det - 1) * (n_det - 2)) / 6)

            crater_triads = np.zeros((n_comb, 3), np.int)
            for it, (i, j, k) in enumerate(enhanced_pattern_shifting(n_det)):
                crater_triads[it] = np.array([i, j, k])

        x_pix_triad, y_pix_triad = x_pix[crater_triads].T, y_pix[crater_triads].T
        avg_triad_x, avg_triad_y = map(lambda c: np.sum(c, axis=0) / 3., (x_pix_triad, y_pix_triad))

        x_triads = x_pix_triad - np.tile(avg_triad_x, (3, 1))
        y_triads = y_pix_triad - np.tile(avg_triad_y, (3, 1))

        clockwise = is_clockwise(x_triads, y_triads)

        crater_triads_cw = crater_triads.copy()
        crater_triads_cw[~clockwise] = np_swap_columns(crater_triads[~clockwise])
        x_triads[:, ~clockwise] = np_swap_columns(x_triads.T[~clockwise]).T
        y_triads[:, ~clockwise] = np_swap_columns(y_triads.T[~clockwise]).T

        if not all_clockwise(x_triads, y_triads):
            line = is_colinear(x_triads, y_triads)
            x_triads = x_triads[:, ~line]
            y_triads = y_triads[:, ~line]
            crater_triads_cw = crater_triads_cw[~line]

            if not all_clockwise(x_triads, y_triads):
                raise Warning("Failed to order triads in clockwise order.")

        A_i, A_j, A_k = map(crater_representation, x_triads, y_triads, a_pix[crater_triads_cw].T,
                            b_pix[crater_triads_cw].T,
                            psi_pix[crater_triads_cw].T)

        return cls(crater_triads_cw, A_i, A_j, A_k)

    @classmethod
    def match_generator(cls,
                        x_pix,
                        y_pix,
                        a_pix,
                        b_pix,
                        psi_pix,
                        convert_to_radians=True
                        ):
        n_det = len(x_pix)
        for i, j, k in enhanced_pattern_shifting(n_det):
            crater_triads = np.array([i, j, k])[None, :]
            yield crater_triads.squeeze(), cls.from_detection(x_pix,
                                                              y_pix,
                                                              a_pix,
                                                              b_pix,
                                                              psi_pix,
                                                              convert_to_radians,
                                                              crater_triads
                                                              )

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
        .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. http://arxiv.org/abs/2009.01228
        """

        if permutation_invariant:
            return np.column_stack((
                PermutationInvariant.F(self.I_ij, self.I_jk, self.I_ki).T,
                PermutationInvariant.F1(self.I_ji, self.I_kj, self.I_ik),
                PermutationInvariant.G_tilde(self.I_ij, self.I_jk, self.I_ki, self.I_ji, self.I_kj, self.I_ik).T,
                self.I_ijk
            )).T

        else:
            return np.column_stack((
                self.I_ij,
                self.I_ji,
                self.I_ik,
                self.I_ki,
                self.I_jk,
                self.I_kj,
                self.I_ijk,
            ))

    def __len__(self):
        return len(self.I_ij)


if __name__ == "__main__":
    xyz = (-1, -2, 3)
    xyz_perm = cyclic_permutations(xyz)
    for xyz_ in xyz_perm:
        print(f"F{xyz_} = {PermutationInvariant.F(*xyz_)}")

    pqr = (5, -10, 100)
    for (xyz_, pqr_) in zip(cyclic_permutations(xyz), cyclic_permutations(pqr)):
        print(f"G_tilde({xyz_}, {pqr_}) = {PermutationInvariant.G_tilde(*xyz, *pqr_)}")
