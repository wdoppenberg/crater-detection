import numpy as np
import pandas as pd
import sklearn
from scipy.spatial.distance import cdist
import cv2

from .functions import latlong2cartesian, haversine_np


class CraterDatabase:
    def __init__(self,
                 source_id: np.ndarray,
                 dest_id: np.ndarray,
                 dist: np.ndarray,
                 source_diam: np.ndarray,
                 dest_diam: np.ndarray,
                 normalised: bool
                 ):
        self.source_id = source_id
        self.dest_id = dest_id
        self.dist = dist
        self.source_diam = source_diam
        self.dest_diam = dest_diam
        self._normalised = normalised

    def is_normalised(self):
        return self._normalised

    @classmethod
    def from_df(cls, df, n_neighbors=300, normalise=True, keys={'lat': 'lat', 'long': 'long', 'diam': 'diam'}):
        lat, long = df[[keys['lat'], keys['long']]].to_numpy().T
        x, y, z = latlong2cartesian(lat, long)
        diam = df[keys['diam']].to_numpy()

        A = sklearn.neighbors.kneighbors_graph(np.array([x, y, z]).T, n_neighbors, mode='connectivity', n_jobs=-1)
        source_id, dest_id = A.nonzero()
        dist = haversine_np(lat[source_id], long[source_id], lat[dest_id], long[dest_id])

        if normalise:
            dist /= diam[source_id]

        return cls(source_id, dest_id, dist, diam[source_id], diam[dest_id], normalised=normalise)

    def __repr__(self):
        n = 'normalised' if self._normalised else 'absolute'
        return self.__class__.__name__+f"( ({len(np.unique(self.source_id))}x{len(np.unique(self.dest_id))}) <-> {len(self.dist)} [{n}] )"


class CraterSet:
    """
    Set of craters in pixel space
    """
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 diam: np.ndarray,
                 id: np.ndarray):
        self.x = x
        self.y = y
        self.diam = diam
        self.id = id

    @classmethod
    def from_df(cls, df, keys={'x': 'x', 'y': 'y', 'diam': 'diam'}):
        x = df[keys['x']].to_numpy()
        y = df[keys['y']].to_numpy()
        diam = df[keys['diam']].to_numpy()
        id = np.array(df.index)
        return cls(x, y, diam, id)

    def to_array(self):
        return np.array((self.x, self.y, self.diam))

    def to_df(self, keys={'x': 'x', 'y': 'y', 'diam': 'diam'}):
        return pd.DataFrame(
            {
                keys['x']: self.x,
                keys['y']: self.y,
                keys['diam']: self.diam
            },
            index=self.id
        )

    def xy(self):
        return self.to_array()[:2]

    def slice(self, i):
        return CraterSet(self.x[i], self.y[i], self.diam[i], self.id[i])

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.diam[i]

    def __len__(self):
        return len(self.x)

    def __repr__(self):
        return self.__class__.__name__+f"({len(self)})"

    def __iter__(self):
        for x, y, diam, id in zip(self.x, self.y, self.diam, self.id):
            yield x, y, diam, id


class AdjacentCraters:
    def __init__(self,
                 craters: CraterSet,
                 dist: np.ndarray,
                 normalised: bool
                 ):
        self.craters = craters
        self.dist = dist
        self._normalised = normalised

    def is_normalised(self):
        return self._normalised

    def __repr__(self):
        n = 'normalised' if self._normalised else 'absolute'
        return self.__class__.__name__+f"( craters={repr(self.craters)}, dist=ndarray{self.dist.shape} [{n}] )"


class PoleCrater:
    def __init__(self,
                 x: np.float64,
                 y: np.float64,
                 diam: np.float64,
                 index: int,
                 adj: AdjacentCraters,
                 ):
        self.x = x
        self.y = y
        self.diam = diam
        self.index = index
        self.adj = adj

    def is_normalised(self):
        return self.adj.is_normalised()

    @classmethod
    def from_craterset(cls,
                       craterset,
                       offset=0,
                       pixel_space=np.array([[256, 256]]),
                       normalise=True
                       ):
        if offset > (len(craterset) - 1):
            raise KeyError(f"Offset ({offset}) out of bounds! Cannot be higher than {(len(craterset) - 1)}.")

        center = pixel_space // 2

        index = np.argsort(cdist(center, craterset.xy().T, 'euclidean')).squeeze()[offset]
        x, y, diam = craterset[index]

        adj_craters = craterset.slice(craterset.id != index)
        adj_dist = cdist(np.array([[x, y]]), adj_craters.xy().T, 'euclidean').squeeze()

        if normalise:
            adj_dist /= diam

        adj = AdjacentCraters(adj_craters, adj_dist, normalised=normalise)

        return cls(x, y, diam, index, adj)

    def to_img(self, img=None, shape=(256, 256)):
        if not img:
            img = np.zeros(shape)

        for x_adj, y_adj, _, _ in self.adj.craters:
            cv2.line(img, (round(self.x), round(self.y)), (round(x_adj), round(y_adj)), (255, 1, 0), 1)

        return img

    def __repr__(self):
        return self.__class__.__name__+f"( x={self.x}, y={self.y}, diam={self.diam}, index={self.index}, adj={repr(self.adj)} )"
