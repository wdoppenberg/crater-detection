from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import sklearn.neighbors


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
                 ellipse_limit=1.1):
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


class CraterDatabase:
    def __init__(self,
                 connectivity_matrix,
                 crater_id: np.ndarray,
                 lat: np.ndarray,
                 long: np.ndarray,
                 a: np.ndarray,
                 b: np.ndarray,
                 connectivity_index=True
                 ):
        self._connectivity_matrix = connectivity_matrix
        self._connectivity_index = connectivity_index

        self.crater_id = crater_id
        self.lat = lat  # [rad]
        self.long = long  # [rad]
        self.a = a
        self.b = b

    @classmethod
    def from_df(cls, df, radius=150, Rplanet=1737.1):
        lat, long = df[['LAT_ELLI_IMG', 'LON_ELLI_IMG']].to_numpy().T
        lat, long = map(np.radians, (lat, long))  # ALWAYS CONVERT TO RADIANS

        major, minor = df[['DIAM_ELLI_MAJOR_IMG', 'DIAM_ELLI_MINOR_IMG']].to_numpy().T
        x, y, z = latlong2cartesian(lat, long)

        connectivity_matrix = sklearn.neighbors.radius_neighbors_graph(np.array([x, y, z]).T, radius,
                                                                       mode='connectivity',
                                                                       metric='euclidean', n_jobs=-1)
        source_index, _ = connectivity_matrix.nonzero()

        lat, long = lat[source_index], long[source_index]
        crater_id = df['CRATER_ID'].to_numpy()

        # dlat = lat_dest - lat_source
        # dlong = long_dest - long_source
        #
        # dx = 2 * Rplanet * np.arcsin(np.cos(lat_source) * np.sin(dlong / 2))
        # dy = Rplanet * dlat

        return cls(connectivity_matrix, crater_id, lat, long, major, minor)

    def __getitem__(self, item):
        return (self.crater_id[item],
                self.lat[item],
                self.long[item],
                self.a[item],
                self.b[item])

    def __len__(self):
        return len(self.crater_id)

    def __repr__(self):
        return self.__class__.__name__ + f"( ({len(np.unique(self.crater_id))}) <-> " \
                                         f"{len(self._connectivity_matrix.nonzero()[0])})"


if __name__ == "__main__":
    df_craters = load_craters(diamlims=[4, 30])
    print(df_craters)
    print(df_craters.columns)
    db = CraterDatabase.from_df(df_craters)
    print(np.array(list(combinations(range(len(db)), 3))))

    # https://github.com/michelp/pygraphblas
