import cv2
import numpy as np
from numba import njit


# https://automaticaddison.com/how-to-do-histogram-matching-using-opencv/


def calculate_cdf(histogram: np.ndarray) -> np.ndarray:
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()

    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())

    return normalized_cdf


@njit
def calculate_lookup(src_cdf: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table


def match_histograms(src_image: np.ndarray, ref_image: np.ndarray = None, ref_cdf: np.ndarray = None) -> np.ndarray:
    dtype = src_image.dtype
    src_hist, bin_0 = np.histogram(src_image.flatten(), 256, [0, 256])
    src_cdf = calculate_cdf(src_hist)

    if ref_image is not None:
        ref_hist, _ = np.histogram(ref_image.flatten(), 256, [0, 256])
        ref_cdf = calculate_cdf(ref_hist)

    if ref_cdf is None:
        raise ValueError("Either ref_image or lookup_table must be given")

    # Make a separate lookup table for each color
    lookup_table = calculate_lookup(src_cdf, ref_cdf)

    # Use the lookup function to transform the colors of the original
    # source image
    image_after_matching = cv2.LUT(src_image.astype(np.uint8), lookup_table)

    image_after_matching = cv2.convertScaleAbs(image_after_matching)

    return image_after_matching.astype(dtype)
