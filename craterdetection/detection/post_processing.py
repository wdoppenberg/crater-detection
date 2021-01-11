import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


# Works, but reduces the amount of detected craters; May be useful in future when assessing crater matching performance.
def skeletonise(img_threshold):
    img = img_threshold.copy()
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel


def ellipticity(a, b):
    return (a - b) / a if a >= b else (b - a) / b


def crater_detection(mask, threshold_lower=80, ellipticity_threshold=0.01):
    mask = np.uint8(mask * 255)
    ret, thresh = cv2.threshold(mask, threshold_lower, 255, 0)
    # thresh = skeletonise(thresh)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if (len(cnt) >= 5)]

    convex_hulls = [cv2.convexHull(cnt) for cnt in contours]
    ellipses = [cv2.fitEllipse(cnt) for cnt in convex_hulls if len(cnt) >= 5]

    ellipses_arr = np.array([[e[0][0], e[0][1], e[1][0] / 2, e[1][1] / 2, e[2]] for e in ellipses])
    df_detections = pd.DataFrame(ellipses_arr, columns=['x_pix', 'y_pix', 'a_pix', 'b_pix', 'angle_pix'])

    # Remove detections based on ellipticity

    df_detections['e'] = df_detections.apply(lambda row: ellipticity(row['a_pix'], row['b_pix']), axis=1)
    df_detections.query('e <= @ellipticity_threshold', inplace=True)
    df_detections.reset_index(inplace=True, drop=True)

    # Remove duplicate craters (based on position)
    xy = df_detections[['x_pix', 'y_pix']].to_numpy()
    dist = cdist(xy, xy, 'euclidean')

    duplicates = np.array(np.where(np.bitwise_and(dist > 0, dist < 2))).T[::2]
    df_detections.drop(index=duplicates[:, 0], inplace=True)
    df_detections.reset_index(inplace=True, drop=True)

    df_detections.eval('r_pix = (a_pix + b_pix)/2', inplace=True)
    df_detections.eval('diam_pix = 2*r_pix', inplace=True)

    # TODO: (Optional) add another filter based on crater size

    return df_detections


def draw_detections(df, shape=(256, 256)):
    img_ellipses = np.zeros(shape)
    for i, r in df.iterrows():
        center_coordinates = (round(r['x_pix']), round(r['y_pix']))
        axes_length = (round(r['a_pix']), round(r['b_pix']))
        angle = round(r['angle_pix'])
        img_ellipses = cv2.ellipse(img_ellipses, center_coordinates, axes_length,
                                   angle, 0, 360, (255, 255, 255), 1)
        img_ellipses = cv2.circle(img_ellipses, center_coordinates, 0, (255, 255, 255), 1)

    return img_ellipses
