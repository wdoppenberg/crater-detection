from functools import partial
from itertools import repeat

import networkx as nx
import numpy as np
import numpy.linalg as LA
import pandas as pd
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
from sklearn.neighbors import radius_neighbors_graph

import craterdetection.common.constants as const
from craterdetection.common.camera import crater_camera_homography, camera_matrix
from craterdetection.common.conics import crater_representation, conic_center
from craterdetection.common.coordinates import ENU_system, nadir_attitude
from craterdetection.matching.projective_invariants import CoplanarInvariants
from craterdetection.matching.utils import triad_splice, get_cliques_by_length, cyclic_permutations


def load_craters(path="../../data/lunar_crater_database_robbins_2018.csv",
                 latlims=None,
                 longlims=None,
                 diamlims=const.DIAMLIMS,
                 ellipse_limit=const.MAX_ELLIPTICITY,
                 arc_lims=0.8
                 ):
    df_craters = pd.read_csv(path)
    df_craters.query("ARC_IMG > @arc_lims", inplace=True)

    if latlims:
        lat0, lat1 = latlims
        df_craters.query('(LAT_ELLI_IMG > @lat0) & (LAT_ELLI_IMG < @lat1)', inplace=True)
    if longlims:
        long0, long1 = longlims
        df_craters.query('(LON_ELLI_IMG > @long0) & (LON_ELLI_IMG < @long1)', inplace=True)
    if diamlims:
        diam0, diam1 = diamlims
        df_craters.query('(DIAM_CIRC_IMG >= @diam0) & (DIAM_CIRC_IMG <= @diam1)', inplace=True)

    df_craters.dropna(inplace=True)
    df_craters.query('(DIAM_ELLI_MAJOR_IMG/DIAM_ELLI_MINOR_IMG) <= @ellipse_limit', inplace=True)

    return df_craters


def extract_robbins_dataset(df=None, column_keys=None, radians=True):
    if df is None:
        df = load_craters()

    if column_keys is None:
        column_keys = dict(lat='LAT_ELLI_IMG', long='LON_ELLI_IMG', major='DIAM_ELLI_MAJOR_IMG',
                           minor='DIAM_ELLI_MINOR_IMG', angle='DIAM_ELLI_ANGLE_IMG', id='CRATER_ID')

    lat, long = df[[column_keys['lat'], column_keys['long']]].to_numpy().T
    major, minor = df[[column_keys['major'], column_keys['minor']]].to_numpy().T
    psi = df[column_keys['angle']].to_numpy()
    if radians:
        lat, long = map(np.radians, (lat, long))  # ALWAYS CONVERT TO RADIANS
        psi = np.radians(psi)
    crater_id = df[column_keys['id']].to_numpy()

    return lat, long, major, minor, psi, crater_id


# Deprecated
def _gen_local_cartesian_coords(lat, long, x, y, z, crater_triads, Rbody=const.RMOON):
    avg_triad_x, avg_triad_y, avg_triad_z = map(lambda c: np.sum(triad_splice(c, crater_triads), axis=0) / 3.,
                                                (x, y, z))
    _, avg_triad_lat, avg_triad_long = cartesian_to_spherical(avg_triad_x, avg_triad_y, avg_triad_z)
    avg_triad_lat, avg_triad_long = map(np.array, (avg_triad_lat, avg_triad_long))

    dlat = np.array(triad_splice(lat, crater_triads)) - np.tile(avg_triad_lat, (3, 1))
    dlong = np.array(triad_splice(long, crater_triads)) - np.tile(avg_triad_long, (3, 1))

    """
    Use Haversine great circle function decomposed for lat & long to generate approximations for cartesian coordinates
    in a 2D ENU-reference frame.
    """
    x_triads = np.array(
        [2 * Rbody * np.arcsin(np.cos(np.radians(avg_triad_lat)) * np.sin(dlong_i / 2)) for dlong_i in dlong]
    )
    y_triads = np.array([Rbody * dlat_i for dlat_i in dlat])

    return x_triads, y_triads


def gen_ENU_coordinates(lat, long, crater_triads, Rbody=const.RMOON):
    """Generate local 2D coordinates for crater triads by constructing a plane normal to the centroid. This is an
    approximation that is only valid for craters that, for practical reasons, can be considered coplanar.

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
    lat : np.ndarray
        Crater latitude [radians]
    long : np.ndarray
        Crater longitude [radians]
    crater_triads : np.ndarray
        Crater triad indices (nx3) for slicing arrays
    Rbody : float, optional
        Body radius, defaults to RMOON [km]
    Returns
    -------
    x_triad, y_triad : np.ndarray
        3xN array containing x or y coordinate in a per-triad ENU 2D (East, North) coordinate system.
    """
    x, y, z = map(np.array, spherical_to_cartesian(Rbody, lat[crater_triads], long[crater_triads]))
    avg_x, avg_y, avg_z = map(partial(np.mean, axis=-1), (x, y, z))

    p_centroid = np.array([avg_x, avg_y, avg_z])[:, None].transpose(2, 0, 1)

    e_i, n_i, u_i = ENU_system(p_centroid)

    T_ME = LA.inv(np.concatenate((e_i, n_i, u_i), axis=-1))

    dx, dy, dz = x - avg_x[:, None], y - avg_y[:, None], z - avg_z[:, None]
    delta_pos = np.concatenate((dx[None, :], dy[None, :], dz[None, :]), axis=0).T[..., None]
    ENU_pos = T_ME @ delta_pos

    if np.mean(np.abs(ENU_pos[..., 2, 0])) / np.mean(np.abs(ENU_pos[..., 0, 0])) > 0.05 or \
            np.mean(np.abs(ENU_pos[..., 2, 0])) / np.mean(np.abs(ENU_pos[..., 1, 0])) > 0.05:
        raise Warning("Average absolute Z-component in ENU coordinate system exceeds 5%!")

    # Z-component is negligible, as is intended.
    x_triad = ENU_pos[:, :, 0, 0]
    y_triad = ENU_pos[:, :, 1, 0]

    return x_triad, y_triad


class CraterDatabase:
    def __init__(self,
                 lat,
                 long,
                 major_axis,
                 minor_axis,
                 psi,
                 crater_id=None,
                 Rbody=const.RMOON,
                 radius=const.TRIAD_RADIUS,
                 vcam_alt=const.DB_CAM_ALTITUDE
                 ):
        """Crater database abstraction keyed by crater triads that generate projective invariants using information
        about their elliptical shape and relative positions [1]. Input is a crater dataset [2] that has positional
        and geometrical (ellipse parameters) information; output is an array of 7 features per crater triad.

        Parameters
        ----------
        lat : np.ndarray
            Crater latitude [radians]
        long : np.ndarray
            Crater longitude [radians]
        major_axis : np.ndarray
            Crater major axis [km]
        minor_axis : np.ndarray
            Crater minor axis [km]
        psi : np.ndarray
            Crater ellipse tilt angle, major axis w.r.t. East-West direction (0, pi) [radians]
        crater_id : np.ndarray, optional
            Crater identifier, defaults to enumerated array over len(lat)
        Rbody : float, optional
            Body radius, defaults to RMOON [km]
        radius : float, int
            Maximum radius to consider two craters connected, defaults to TRIAD_RADIUS [km]
        vcam_alt : float, int
            Altitude of virtual per-triad camera

        References
        ----------
        .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. http://arxiv.org/abs/2009.01228
        .. [2] Robbins, S. J. (2019). A New Global Database of Lunar Impact Craters &gt;1–2 km: 1. Crater Locations and Sizes, Comparisons With Published Databases, and Global Analysis. Journal of Geophysical Research: Planets, 124(4), 871–892. https://doi.org/10.1029/2018JE005592
        """

        if crater_id is None:
            self.crater_id = np.arange(len(lat))
        else:
            self.crater_id = crater_id

        self.lat = lat
        self.long = long
        self.C_cat = crater_representation(major_axis, minor_axis, psi)

        x, y, z = map(np.array, spherical_to_cartesian(Rbody, self.lat, self.long))

        self.r_craters = np.array((x, y, z)).T[..., None]

        """
        Construct adjacency matrix and generate Graph instance
        """
        self._adjacency_matrix = radius_neighbors_graph(np.array([x, y, z]).T, radius,
                                                        mode='distance',
                                                        metric='euclidean', n_jobs=-1)

        self._graph = nx.from_scipy_sparse_matrix(self._adjacency_matrix)

        """
        Get all crater triads using cycle basis with length = 3
        https://en.wikipedia.org/wiki/Cycle_basis
        
        The following returns a nx3 array containing the indices of crater triads
        """
        crater_triads = np.array(get_cliques_by_length(self._graph, 3))

        """
        Project crater triads into virtual image plane using homography
        """
        r_M_ijk = np.moveaxis(
            np.concatenate(
                (x[crater_triads].T[None, ...],
                 y[crater_triads].T[None, ...],
                 z[crater_triads].T[None, ...]),
                axis=0
            ),
            0, 2)[..., None]
        r_centroid = np.mean(r_M_ijk, axis=0)
        r_vcam = r_centroid + (r_centroid / LA.norm(r_centroid, axis=1)[..., None]) * vcam_alt

        T_CM = np.concatenate(nadir_attitude(r_vcam), axis=-1)
        if (LA.matrix_rank(T_CM) != 3).any():
            raise Warning("Invalid camera attitude matrices present!:\n", T_CM)

        K = camera_matrix()
        P_MC = K @ LA.inv(T_CM) @ np.concatenate((np.tile(np.identity(3), (len(r_vcam), 1, 1)), -r_vcam), axis=2)

        H_C_triads = np.array(list(map(crater_camera_homography, r_M_ijk, repeat(P_MC))))

        """
        Ensure all crater triads are clockwise
        """
        C_triads = np.array(list(map(lambda vertex: self.C_cat[vertex], crater_triads.T)))

        A_i, A_j, A_k = map(lambda T, C: LA.inv(T).transpose((0, 2, 1)) @ C @ LA.inv(T), H_C_triads, C_triads)
        r_i, r_j, r_k = map(conic_center, (A_i, A_j, A_k))

        cw_value = LA.det(np.moveaxis(np.array([[r_i[:, 0], r_i[:, 1], np.ones_like(r_i[:, 0])],
                                                [r_j[:, 0], r_j[:, 1], np.ones_like(r_i[:, 0])],
                                                [r_k[:, 0], r_k[:, 1], np.ones_like(r_i[:, 0])]]), -1, 0))
        clockwise = cw_value < 0
        line = cw_value == 0

        clockwise = clockwise[~line]
        crater_triads = crater_triads[~line]
        H_C_triads = H_C_triads[:, ~line]

        crater_triads[np.argwhere(~clockwise), [0, 1]] = crater_triads[np.argwhere(~clockwise), [1, 0]]
        H_C_triads[[0, 1], np.argwhere(~clockwise)] = H_C_triads[[1, 0], np.argwhere(~clockwise)]

        C_triads = np.array(list(map(lambda vertex: self.C_cat[vertex], crater_triads.T)))
        A_i, A_j, A_k = map(lambda T, C: LA.inv(T).transpose((0, 2, 1)) @ C @ LA.inv(T), H_C_triads, C_triads)

        invariants = CoplanarInvariants(crater_triads, A_i, A_j, A_k, normalize_det=True)

        self.features = invariants.get_pattern()
        self.crater_triads = invariants.crater_triads

    @classmethod
    def from_df(cls,
                df,
                column_keys=None,
                Rbody=const.RMOON,
                radius=const.TRIAD_RADIUS
                ):
        """
        Class method for constructing from pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Crater dataset
        column_keys : dict
            Mapping for extracting lat, long, major, minor, angle, id from DataFrame columns
        Rbody : float, optional
            Body radius, defaults to RMOON [km]
        radius :
            Maximum radius to consider two craters connected, defaults to TRIAD_RADIUS [km]

        Returns
        -------
        CraterDatabase
        """

        if column_keys is None:
            column_keys = dict(lat='LAT_ELLI_IMG', long='LON_ELLI_IMG', major='DIAM_ELLI_MAJOR_IMG',
                               minor='DIAM_ELLI_MINOR_IMG', angle='DIAM_ELLI_ANGLE_IMG', id='CRATER_ID')

        lat, long = df[[column_keys['lat'], column_keys['long']]].to_numpy().T
        lat, long = map(np.radians, (lat, long))  # ALWAYS CONVERT TO RADIANS
        major, minor = df[[column_keys['major'], column_keys['minor']]].to_numpy().T
        psi = np.radians(df[column_keys['angle']].to_numpy())
        crater_id = df[column_keys['id']].to_numpy()

        return cls(lat, long, major, minor, psi, crater_id, Rbody, radius)

    @classmethod
    def from_file(cls,
                  path=None,
                  latlims=None,
                  longlims=None,
                  diamlims=const.DIAMLIMS,
                  ellipse_limit=const.MAX_ELLIPTICITY,
                  column_keys=None,
                  Rbody=const.RMOON,
                  radius=const.TRIAD_RADIUS
                  ):
        """

        Parameters
        ----------
        path
            Path to crater dataset CSV
        latlims, longlims : list
            Limits for latitude & longitude (format: [min, max])
        diamlims : list
            Limits for crater diameter (format: [min, max]), defaults to _diamlims
        ellipse_limit : float
            Limit dataset to craters with b/a <= MAX_ELLIPTICITY
        column_keys : dict
            Mapping for extracting lat, long, major, minor, angle, id from DataFrame columns
        Rbody : float, optional
            Body radius, defaults to RMOON [km]
        radius :
            Maximum radius to consider two craters connected, defaults to TRIAD_RADIUS [km]

        Returns
        -------
        CraterDatabase
        """

        if path is None:
            if __name__ == "__main__":
                path = "../../data/lunar_crater_database_robbins_2018.csv"
            else:
                path = "data/lunar_crater_database_robbins_2018.csv"

        if column_keys is None:
            column_keys = dict(lat='LAT_ELLI_IMG', long='LON_ELLI_IMG', major='DIAM_ELLI_MAJOR_IMG',
                               minor='DIAM_ELLI_MINOR_IMG', angle='DIAM_ELLI_ANGLE_IMG', id='CRATER_ID')

        if diamlims is None:
            diamlims = const.DIAMLIMS

        df_craters = load_craters(path, latlims, longlims, diamlims, ellipse_limit)

        return cls.from_df(df_craters, column_keys, Rbody, radius)

    def get_features(self):
        id_i, id_j, id_k = triad_splice(self.crater_id, self.crater_triads)
        return id_i, id_j, id_k, self.features

    def get_position(self, index=None):
        if index is None:
            return triad_splice(self.lat, self.crater_triads), triad_splice(self.long, self.crater_triads)

        return triad_splice(self.lat, self.crater_triads[index]), triad_splice(self.long, self.crater_triads[index])

    def match_detections(self, A_detections, threshold=0.05, max_iter=50, unique_matches=4, top_n_matches=10):
        matches = {detection_key: [] for detection_key, _ in enumerate(A_detections)}

        for i, (crater_triad, features) in enumerate(CoplanarInvariants.match_generator(A_craters=A_detections)):
            for order in cyclic_permutations(np.arange(3)):
                order_full = np.append(np.concatenate((order, order + 3)), -1)
                diff = np.mean(np.abs(((self.features - features[order_full]) / features[order_full])), axis=1)

                if np.min(diff) < threshold:
                    min_n = np.argpartition(diff, 5)[:top_n_matches]
                    for min_idx in min_n:
                        for detection_idx, db_idx in zip(crater_triad[order], self.crater_triads[min_idx]):
                            matches[detection_idx] += [db_idx]
                    break

            if i >= max_iter:
                break

        matches_val = dict()

        for k, v in matches.items():
            if len(v) >= 2 * top_n_matches:
                match_idx, counts = np.unique(np.array(v), return_counts=True)
                count_order = np.argsort(counts)
                if counts[count_order][-1] > unique_matches:
                    matches_val[k] = match_idx[count_order][-1].item()

        A_craters_det = A_detections[list(matches_val.keys())]
        C_craters_det = self.C_cat[list(matches_val.values())]
        r_craters_det = self.r_craters[list(matches_val.values())]

        # TODO: Implement 'no-matches' handling

        return A_craters_det, r_craters_det, C_craters_det

    def __len__(self):
        return len(self.features)


if __name__ == "__main__":
    db = CraterDatabase.from_file()
    print(db.get_features())
