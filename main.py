import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

from craterdetection.detection import DeepMoon
from craterdetection.detection.post_processing import crater_detection, draw_detections
from craterdetection.matching import CoplanarInvariants, CraterDatabase


if __name__ == "__main__":
    CraterDatabase.from_file()
