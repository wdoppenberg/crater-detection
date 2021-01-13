from itertools import repeat

import networkx as nx
import numpy as np
import numpy.linalg as LA
import pandas as pd
import sklearn.neighbors
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian

from craterdetection.matching.projective_invariants import crater_representation, CoplanarInvariants

_radius = 200  # Create triangles with every crater within this radius [km]
_Rbody = 1737.1  # Body radius (moon) [km]
_diamlims = [4, 30]
_ellipse_limit = 1.1


# https://stackoverflow.com/questions/1705824/finding-cycle-of-3-nodes-or-triangles-in-a-graph
def get_cliques_by_length(G, length_clique):
    """ Return the list of all cliques in an undirected graph G with length
    equal to length_clique. """
    cliques = []
    for c in nx.enumerate_all_cliques(G):
        if len(c) <= length_clique:
            if len(c) == length_clique:
                cliques.append(c)
        else:
            return cliques
    # return empty list if nothing is found
    return cliques


def load_craters(path="../../data/lunar_crater_database_robbins_2018.csv", latlims=None, longlims=None, diamlims=None,
                 ellipse_limit=_ellipse_limit):
    df_craters = pd.read_csv(path)

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


def latlong2cartesian(lat, long, alt=0, rad=1737.1):
    """
    Calculate Cartesian coordinates from latitude + longitude information
    """
    f = 1. / 825.  # flattening (Moon)
    ls = np.arctan((1 - f) ** 2 * np.tan(lat))  # lambda

    x = rad * np.cos(ls) * np.cos(long) + alt * np.cos(lat) * np.cos(long)
    y = rad * np.cos(ls) * np.sin(long) + alt * np.cos(lat) * np.sin(long)
    z = rad * np.sin(ls) + alt * np.sin(lat)

    return x, y, z


def _triad_splice(arr, triangles_):
    return np.array((arr[triangles_[:, 0]], arr[triangles_[:, 1]], arr[triangles_[:, 2]]))


def _np_swap_columns(arr):
    arr[:, 0], arr[:, 1] = arr[:, 1], arr[:, 0].copy()
    return arr


def _gen_local_cartesian_coords(lat, long, x, y, z, crater_triads, Rbody=1737.1):
    avg_triad_x, avg_triad_y, avg_triad_z = map(lambda c: np.sum(_triad_splice(c, crater_triads), axis=0) / 3.,
                                                (x, y, z))
    _, avg_triad_lat, avg_triad_long = cartesian_to_spherical(avg_triad_x, avg_triad_y, avg_triad_z)
    avg_triad_lat, avg_triad_long = map(np.array, (avg_triad_lat, avg_triad_long))

    dlat = np.array(_triad_splice(lat, crater_triads)) - np.tile(avg_triad_lat, (3, 1))
    dlong = np.array(_triad_splice(long, crater_triads)) - np.tile(avg_triad_long, (3, 1))

    """
    Use Haversine great circle function decomposed for lat & long to generate approximations for cartesian coordinates
    in a 2D ENU-reference frame.
    """
    x_triads = np.array(
        [2 * Rbody * np.arcsin(np.cos(np.radians(avg_triad_lat)) * np.sin(dlong_i / 2)) for dlong_i in dlong]
    )
    y_triads = np.array([Rbody * dlat_i for dlat_i in dlat])

    return x_triads, y_triads


def _cw_or_ccw(x_triads_, y_triads_):
    return LA.det(np.moveaxis(np.array([[x_triads_[0], y_triads_[0], np.ones_like(x_triads_[0])],
                                        [x_triads_[1], y_triads_[1], np.ones_like(x_triads_[0])],
                                        [x_triads_[2], y_triads_[2], np.ones_like(x_triads_[0])]]), -1, 0))


def _is_colinear(x_triads_, y_triads_):
    return _cw_or_ccw(x_triads_, y_triads_) == 0


def _is_clockwise(x_triads, y_triads):
    """Returns boolean array which tells whether the three points in 2D plane given by x_triads & y_triads are
    oriented clockwise. https://en.wikipedia.org/wiki/Curve_orientation#Orientation_of_a_simple_polygon

    Parameters
    ----------
    x_triads, y_triads : np.ndarray
        Array of 2D coordinates for triangles in a plane

    Returns
    -------
    np.ndarray
    """
    return _cw_or_ccw(x_triads, y_triads) < 0


def _all_clockwise(x_triads_, y_triads_):
    return np.logical_and.reduce(_is_clockwise(x_triads_, y_triads_))


class CraterDatabase:
    def __init__(self,
                 lat,
                 long,
                 major_axis,
                 minor_axis,
                 psi,
                 crater_id=None,
                 Rbody=_Rbody,
                 radius=_radius
                 ):
        """Crater database abstraction keyed by crater triads that generate projective invariants using information
        about their elliptical shape and relative positions [1]. Input is a crater dataset [2] that has positional
        and geometrical (ellipse parameters) information, and generates an array of 7 features per
        crater triad

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
            Crater ellipse angle [radians]
        crater_id : np.ndarray, optional
            Crater identifier, defaults to enumerated array over len(lat)
        Rbody : float, optional
            Body radius, defaults to _radius [km]
        radius :
            Maximum radius to consider two craters connected [km]

        References
        ----------
        .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. http://arxiv.org/abs/2009.01228
        .. [2] Robbins, S. J. (2019). A New Global Database of Lunar Impact Craters &gt;1–2 km: 1. Crater Locations and Sizes, Comparisons With Published Databases, and Global Analysis. Journal of Geophysical Research: Planets, 124(4), 871–892. https://doi.org/10.1029/2018JE005592
        """

        if crater_id is None:
            crater_id = np.arange(len(lat))
        else:
            crater_id = crater_id

        x, y, z = map(np.array, spherical_to_cartesian(Rbody, lat, long))

        """
        Construct adjacency matrix to generate Graph instance
        """
        self._adjacency_matrix = sklearn.neighbors.radius_neighbors_graph(np.array([x, y, z]).T, radius,
                                                                          mode='connectivity',
                                                                          metric='euclidean', n_jobs=-1)

        self._graph = nx.from_scipy_sparse_matrix(self._adjacency_matrix)

        """
        Get all crater triads using cycle basis with length = 3
        https://en.wikipedia.org/wiki/Cycle_basis
        
        The following returns a nx3 array containing the indices of crater triads
        """
        crater_triads = np.array([c for c in nx.cycle_basis(self._graph) if len(c) == 3])

        """
        Ensure all crater triads are clockwise
        """
        clockwise = _is_clockwise(*_gen_local_cartesian_coords(lat, long, x, y, z, crater_triads, Rbody))

        crater_triads_cw = crater_triads.copy()
        crater_triads_cw[~clockwise] = _np_swap_columns(crater_triads[~clockwise])

        # TODO: Implement swap here for better performance
        x_triads, y_triads = _gen_local_cartesian_coords(lat, long, x, y, z, crater_triads_cw, Rbody)

        if not _all_clockwise(x_triads, y_triads):
            line = _is_colinear(x_triads, y_triads)
            x_triads = x_triads[:, ~line]
            y_triads = y_triads[:, ~line]
            crater_triads_cw = crater_triads_cw[~line]

            if not _all_clockwise(x_triads, y_triads):
                raise RuntimeError("Failed to order triads in clockwise order.")

        """
        Generate crater matrix representation using per-triad coordinate system along with major- and minor-axis, as 
        well as angle. 
        """
        a_triads, b_triads, psi_triads = map(_triad_splice, (major_axis, minor_axis, psi),
                                             repeat(crater_triads_cw))

        crater_triads_matrices = []
        for args in zip(x_triads, y_triads, a_triads, b_triads, psi_triads):
            crater_triads_matrices.append(crater_representation(*args))

        A_i, A_j, A_k = crater_triads_matrices
        invariants = CoplanarInvariants(crater_triads_cw, A_i, A_j, A_k, normalize_det=True)

        self.features = invariants.get_pattern()
        self.crater_triads = invariants.crater_triads

        self.crater_triads_id = _triad_splice(crater_id, self.crater_triads)

    @classmethod
    def from_df(cls,
                df,
                column_keys=None,
                Rbody=_Rbody,
                radius=_radius
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
            Body radius, defaults to _radius [km]
        radius :
            Maximum radius to consider two craters connected [km]

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
        psi = np.radians(df['DIAM_ELLI_ANGLE_IMG'].to_numpy())
        crater_id = df[column_keys['id']].to_numpy()

        return cls(lat, long, major, minor, psi, crater_id, Rbody, radius)

    @classmethod
    def from_file(cls,
                  path=None,
                  latlims=None,
                  longlims=None,
                  diamlims=None,
                  ellipse_limit=_ellipse_limit,
                  column_keys=None,
                  Rbody=_Rbody,
                  radius=_radius
                  ):
        """

        Parameters
        ----------
        path
        latlims
        longlims
        diamlims
        ellipse_limit
        column_keys : dict
            Mapping for extracting lat, long, major, minor, angle, id from DataFrame columns
        Rbody : float, optional
            Body radius, defaults to _radius [km]
        radius :
            Maximum radius to consider two craters connected [km]

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
            diamlims = _diamlims

        df_craters = load_craters(path, latlims, longlims, diamlims, ellipse_limit)

        return cls.from_df(df_craters, column_keys, Rbody, radius)

    def get_features(self):
        id_i, id_j, id_k = self.crater_triads_id
        return id_i, id_j, id_k, self.features

    def I_ij(self):
        return self.features[0]

    def I_ji(self):
        return self.features[1]

    def I_ik(self):
        return self.features[2]

    def I_ki(self):
        return self.features[3]

    def I_jk(self):
        return self.features[4]

    def I_kj(self):
        return self.features[5]

    def I_ijk(self):
        return self.features[6]


if __name__ == "__main__":
    db = CraterDatabase.from_file()
    print(db.get_features())
