from itertools import repeat

import networkx as nx
import numpy as np
import pandas as pd
import sklearn.neighbors
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian

from craterdetection.matching.projective_invariants import crater_representation, CoplanarInvariants
from craterdetection.matching.utils import triad_splice, np_swap_columns, is_colinear, is_clockwise, all_clockwise

_radius = 200  # Create triangles with every crater within this radius [km]
_Rbody = 1737.1  # Body radius (moon) [km]
_diamlims = [4, 30]  # Limit dataset to craters with diameter between 4 and 30 km
_ellipse_limit = 1.1  # Limit dataset to craters with an ellipticity <= 1.1


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


def _gen_local_cartesian_coords(lat, long, x, y, z, crater_triads, Rbody=1737.1):
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
            Crater ellipse tilt angle, major axis w.r.t. East-West direction (0, pi) [radians]
        crater_id : np.ndarray, optional
            Crater identifier, defaults to enumerated array over len(lat)
        Rbody : float, optional
            Body radius, defaults to _Rbody [km]
        radius :
            Maximum radius to consider two craters connected, defaults to _radius [km]

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

        x, y, z = map(np.array, spherical_to_cartesian(Rbody, self.lat, self.long))

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
        x_triads, y_triads = _gen_local_cartesian_coords(self.lat, self.long, x, y, z, crater_triads, Rbody)

        """
        Ensure all crater triads are clockwise
        """
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
                raise RuntimeError("Failed to order triads in clockwise order.")

        """
        Generate crater matrix representation using per-triad coordinate system along with major- and minor-axis, as 
        well as angle. 
        """
        a_triads, b_triads, psi_triads = map(triad_splice, (major_axis, minor_axis, psi),
                                             repeat(crater_triads_cw))

        A_i, A_j, A_k = map(crater_representation, x_triads, y_triads, a_triads, b_triads, psi_triads)
        invariants = CoplanarInvariants(crater_triads_cw, A_i, A_j, A_k, normalize_det=True)

        self.features = invariants.get_pattern()
        self.crater_triads = invariants.crater_triads

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
            Body radius, defaults to _Rbody [km]
        radius :
            Maximum radius to consider two craters connected, defaults to _radius [km]

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
            Path to crater dataset CSV
        latlims, longlims : list
            Limits for latitude & longitude (format: [min, max])
        diamlims : list
            Limits for crater diameter (format: [min, max]), defaults to _diamlims
        ellipse_limit : float
            Limit dataset to craters with b/a <= ellipse_limit
        column_keys : dict
            Mapping for extracting lat, long, major, minor, angle, id from DataFrame columns
        Rbody : float, optional
            Body radius, defaults to _Rbody [km]
        radius :
            Maximum radius to consider two craters connected, defaults to _radius [km]

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
        id_i, id_j, id_k = triad_splice(self.crater_id, self.crater_triads)
        return id_i, id_j, id_k, self.features

    def get_position(self, index=None):
        if index is None:
            return triad_splice(self.lat, self.crater_triads), triad_splice(self.long, self.crater_triads)

        return triad_splice(self.lat, self.crater_triads[index]), triad_splice(self.long, self.crater_triads[index])

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

    def __len__(self):
        return len(self.I_ij())


if __name__ == "__main__":
    db = CraterDatabase.from_file()
    print(db.get_features())
