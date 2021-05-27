import numpy as np
import pandas as pd

import src.common.constants as const


def load_craters(path="../../data/lunar_crater_database_robbins_2018.csv",
                 latlims=None,
                 longlims=None,
                 diamlims=const.DIAMLIMS,
                 ellipse_limit=const.MAX_ELLIPTICITY,
                 arc_lims=0.0
                 ):
    df_craters = pd.read_csv(path)
    df_craters.query("ARC_IMG > @arc_lims", inplace=True)

    if latlims:
        lat0, lat1 = latlims
        df_craters.query('(LAT_ELLI_IMG >= @lat0) & (LAT_ELLI_IMG <= @lat1)', inplace=True)
    if longlims:
        long0, long1 = map(lambda x: x + 360 if x < 0 else x, longlims)
        long0, long1 = (long1, long0) if long0 > long1 else (long0, long1)
        df_craters.query('(LON_ELLI_IMG >= @long0) & (LON_ELLI_IMG <= @long1)', inplace=True)
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