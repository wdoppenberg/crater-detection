from itertools import repeat

import networkx as nx
import numpy as np
import numpy.linalg as LA
import pandas as pd
import torch
from astropy.coordinates import spherical_to_cartesian
from scipy.spatial import KDTree
from sklearn.neighbors import radius_neighbors_graph

import src.common.constants as const
from src.common.conics import crater_camera_homography
from src.common.robbins import load_craters
from src.common.camera import camera_matrix
from src.common.conics import conic_matrix, conic_center
from src.common.coordinates import nadir_attitude
from src.matching.position_estimation import PositionRegressor
from src.matching.projective_invariants import CoplanarInvariants
from src.matching.utils import get_cliques_by_length, shift_nd


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
                 vcam_alt=const.DB_CAM_ALTITUDE,
                 sort_ij=True
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
        sort_ij : bool
            Whether to sort triad features with I_ij being the lowest absolute value

        References
        ----------
        .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. https://arxiv.org/abs/2009.01228
        .. [2] Robbins, S. J. (2019). A New Global Database of Lunar Impact Craters &gt;1–2 km: 1. Crater Locations and Sizes, Comparisons With Published Databases, and Global Analysis. Journal of Geophysical Research: Planets, 124(4), 871–892. https://doi.org/10.1029/2018JE005592
        """

        if crater_id is None:
            self.crater_id = np.arange(len(lat))
        else:
            self.crater_id = crater_id

        self._lat = lat
        self._long = long
        self._C_cat = conic_matrix(major_axis, minor_axis, psi)

        x, y, z = map(np.array, spherical_to_cartesian(Rbody, self._lat, self._long))

        self._r_craters = np.array((x, y, z)).T[..., None]

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
        C_triads = np.array(list(map(lambda vertex: self._C_cat[vertex], crater_triads.T)))

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

        C_triads = np.array(list(map(lambda vertex: self._C_cat[vertex], crater_triads.T)))
        A_i, A_j, A_k = map(lambda T, C: LA.inv(T).transpose((0, 2, 1)) @ C @ LA.inv(T), H_C_triads, C_triads)

        invariants = CoplanarInvariants(crater_triads, A_i, A_j, A_k, normalize_det=True)

        self._features = invariants.get_pattern()
        self._crater_triads = invariants.crater_triads

        if sort_ij:
            ij_idx = np.abs(self._features[:, :3]).argmin(1)
            self._features = np.concatenate((
                shift_nd(self._features[:, :3], -ij_idx),
                shift_nd(self._features[:, 3:6], -ij_idx),
                self._features[:, [-1]]
            ),
                axis=-1
            )
            self._crater_triads = shift_nd(self._crater_triads, -ij_idx)

            too_close = np.logical_or.reduce(
                (
                    np.abs((self._features[:, 0] - self._features[:, 2]) / self._features[:, 0]) < 0.1,
                    np.abs((self._features[:, 0] - self._features[:, 1]) / self._features[:, 0]) < 0.1
                )
            )

            self._features = np.concatenate(
                (
                    self._features,
                    np.concatenate((
                        np.roll(self._features[too_close, :3], 1),
                        np.roll(self._features[too_close, :3], 1),
                        self._features[too_close, -1:]
                    ),
                        axis=-1
                    )
                ),
                axis=0
            )

            self._crater_triads = np.concatenate(
                (self._crater_triads, np.roll(self._crater_triads[too_close], 1))
            )

        self._kdtree = KDTree(self._features)

    @classmethod
    def from_df(cls,
                df,
                column_keys=None,
                Rbody=const.RMOON,
                radius=const.TRIAD_RADIUS,
                **kwargs
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

        return cls(lat, long, major, minor, psi, crater_id, Rbody, radius, **kwargs)

    @classmethod
    def from_file(cls,
                  path=None,
                  latlims=None,
                  longlims=None,
                  diamlims=const.DIAMLIMS,
                  ellipse_limit=const.MAX_ELLIPTICITY,
                  column_keys=None,
                  Rbody=const.RMOON,
                  radius=const.TRIAD_RADIUS,
                  **kwargs
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

    def query(self,
              key,
              k=1,
              return_distance=False,
              max_distance=0.1,
              batch_size=100
              ):

        if k == 1:
            k = [k]

        if key.shape[-1] == 3:
            if len(key) < 3:
                raise ValueError("Must give at least 3 conics for matching!")

            crater_triads, features = next(CoplanarInvariants.match_generator(
                A_craters=key,
                max_iter=1,
                batch_size=batch_size
            ))

            crater_triads = np.repeat(crater_triads[:, None, :], k, axis=1)

            _, match_idxs = self._kdtree.query(features, k=k, p=2, workers=-1)

            dist = np.abs((features[:, None, :] - self._features[match_idxs]) / features[:, None, :]).mean(-1)
            dist_filter = dist < max_distance
            r_query, C_query = self[match_idxs[dist_filter]]
            A_query = key[crater_triads[dist_filter]]
            A_query, r_query, C_query = A_query.reshape(-1, 3, 3), r_query.reshape(-1, 3, 1), C_query.reshape(-1, 3, 3)

            if return_distance:
                return A_query, r_query, C_query, dist
            else:
                return A_query, r_query, C_query

        elif key.shape[-1] == 7:
            dist, entries = self._kdtree.query(key, k=k, p=2, workers=-1)

            if return_distance:
                return entries, dist
            else:
                return entries

        else:
            raise ValueError("Key has invalid shape! Must be [...")

    def query_position(self,
                       A_detections,
                       T,
                       K,
                       sigma_pix=5,
                       k=30,
                       max_distance=0.043,
                       batch_size=500,
                       residual_threshold=0.011,
                       max_trials=1250,
                       **kwargs
                       ):
        A_query, r_query, C_query = self.query(A_detections, k=k, max_distance=max_distance, batch_size=batch_size)
        estimator = PositionRegressor(sigma_pix=sigma_pix,
                                      residual_threshold=residual_threshold,
                                      max_trials=max_trials,
                                      **kwargs)
        estimator.fit(A_query=A_query, C_query=C_query, r_query=r_query, attitude=T, camera_matrix=K)

        return estimator

    def __getitem__(self, item):
        ct = self._crater_triads[item]

        return self._r_craters[ct], self._C_cat[ct]

    def __len__(self):
        return len(self._features)
