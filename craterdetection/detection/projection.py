import cv2
import numpy as np



def camera_calibration_matrix(f_x, f_y, x_0=0, y_0=0, alpha=0):
    return np.array([[f_x, alpha, x_0],
                     [0,   f_y,   y_0],
                     [0,   0,     1  ]])
