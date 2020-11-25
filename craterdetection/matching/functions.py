def latlong2cartesian(lat, long, alt=0, rad=1737.1):
    """
    Calculate Cartesian coordinates from latitude + longitude information
    """
    lat, long = np.deg2rad(lat), np.deg2rad(long)
    f  = 1./825.                                # flattening (Moon)
    ls = np.arctan((1 - f)**2 * np.tan(lat))    # lambda

    x = rad * np.cos(ls) * np.cos(long) + alt * np.cos(lat) * np.cos(long)
    y = rad * np.cos(ls) * np.sin(long) + alt * np.cos(lat) * np.sin(long)
    z = rad * np.sin(ls) + alt * np.sin(lat)

    return x, y, z

def haversine_np(lat1, long1, lat2, long2):
    """
    Calculate the great circle distance between two points
    on the Moon (specified in decimal degrees)

    All args must be of equal length.

    """
    long1, lat1, long2, lat2 = map(np.radians, [long1, lat1, long2, lat2])

    dlong = long2 - long1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 1737.1 * c
    return km