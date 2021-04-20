import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

from src.detection import DeepMoon
from src.detection.post_processing import crater_detection
from src.detection.visualisation import draw_detections
from src.matching import CoplanarInvariants, CraterDatabase


if __name__ == "__main__":
    CraterDatabase.from_file()
