from functools import reduce
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn.neighbors
import networkx as nx


def load_craters(latlims=None, longlims=None, diamlims=None):
    df_craters = pd.read_csv("../../data/lunar_crater_database_robbins_2018.csv")

    if latlims:
        lat0, lat1 = latlims
        df_craters.query('(LAT_CIRC_IMG > @lat0) & (LAT_CIRC_IMG < @lat1)', inplace=True)
    if longlims:
        long0, long1 = longlims
        df_craters.query('(LON_CIRC_IMG > @long0) & (LON_CIRC_IMG < @long1)', inplace=True)
    if diamlims:
        diam0, diam1 = diamlims
        df_craters.query('(DIAM_CIRC_IMG >= @diam0) & (DIAM_CIRC_IMG <= @diam1)', inplace=True)

    df_craters.dropna(inplace=True)

    return df_craters


def latlong2cartesian(lat, long, alt=0, rad=1737.1):
    """
    Calculate Cartesian coordinates from latitude + longitude information
    """
    lat, long = np.deg2rad(lat), np.deg2rad(long)
    f = 1. / 825.  # flattening (Moon)
    ls = np.arctan((1 - f) ** 2 * np.tan(lat))  # lambda

    x = rad * np.cos(ls) * np.cos(long) + alt * np.cos(lat) * np.cos(long)
    y = rad * np.cos(ls) * np.sin(long) + alt * np.cos(lat) * np.sin(long)
    z = rad * np.sin(ls) + alt * np.sin(lat)

    return x, y, z

@dataclass
class DatabaseEntry:
    source_id: int
    coords_source: tuple
    dest_id: int
    coords_dest: tuple
    dx: float
    dy: float


class CraterDatabase:
    def __init__(self,
                 crater_id: np.ndarray,
                 coords: np.ndarray,
                 dest_id: np.ndarray,
                 dx: np.ndarray,
                 dy: np.ndarray
                 ):
        self.crater_id = crater_id
        self.coords = coords
        self.dest_id = dest_id
        self.dx = dx
        self.dy = dy

        if not reduce(lambda x, y: True == (len(x) == len(y)), self.__dict__.values()):
            raise RuntimeError("Database columns are not equally sized!")

    @classmethod
    def from_df(cls, df, radius=300, Rplanet=1737.1):
        lat, long = df[['LAT_ELLI_IMG', 'LON_ELLI_IMG']].to_numpy().T
        major, minor = df[['DIAM_ELLI_MAJOR_IMG', 'DIAM_ELLI_MINOR_IMG']].to_numpy().T
        x, y, z = latlong2cartesian(lat, long)

        A = sklearn.neighbors.radius_neighbors_graph(np.array([x, y, z]).T, radius, mode='connectivity',
                                                     metric='euclidean', n_jobs=-1)
        source_id, dest_id = A.nonzero()
        lat_source, long_source, lat_dest, long_dest = map(np.radians, (
            lat[source_id], long[source_id], lat[dest_id], long[dest_id])
        )
        coords_source = np.column_stack(lat_source, long_source)
        coords_dest = np.column_stack(lat_dest, long_dest)

        dlat = lat_dest - lat_source
        dlong = long_dest - long_source

        dx = 2 * Rplanet * np.arcsin(np.cos(lat_source) * np.sin(dlong / 2))
        dy = Rplanet * dlat

        return cls(source_id, coords_source, dest_id, coords_dest, dx, dy)

    def __getitem__(self, item):
        return DatabaseEntry(
            self.source_id[item],
            tuple(self.coords_source[item]),
            self.dest_id[item],
            tuple(self.coords_dest[item]),
            self.dx[item],
            self.dy[item],
        )

    def __repr__(self):
        return self.__class__.__name__ + f"( ({len(np.unique(self.source_id))}x{len(np.unique(self.dest_id))}) <-> {len(self.dist)})"


if __name__ == "__main__":
    df_craters = load_craters(diamlims=[4, 30])
    lat, long = df_craters[['LAT_ELLI_IMG', 'LON_ELLI_IMG']].to_numpy().T
    major, minor = df_craters[['DIAM_ELLI_MAJOR_IMG', 'DIAM_ELLI_MINOR_IMG']].to_numpy().T
    x, y, z = latlong2cartesian(lat, long)

    A = sklearn.neighbors.radius_neighbors_graph(np.array([x, y, z]).T, 300, mode='connectivity',
                                                 metric='euclidean', n_jobs=-1)

    # G = nx.convert_matrix.from_scipy_sparse_matrix(A)
