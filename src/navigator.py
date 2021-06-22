from typing import Union

import numpy as np
import torch
from filterpy.kalman import EKF

from common.coordinates import OrbitingBodyBase
from src.detection import CraterDetector
from src.matching import CraterDatabase
from src.common.camera import Camera


class LunarNavigator(CraterDetector, Camera):
    def __init__(self,
                 database: CraterDatabase,
                 ekf: EKF,
                 **kwargs):
        super().__init__(**kwargs)
        self._database = database
        self._ekf = ekf

    @torch.no_grad()
    def derive_position(self, image: Union[np.ndarray, torch.Tensor], attitude, confidence=0.75):
        self.eval()

        if len(image.shape) == 2:
            image = image[None, None, ...]
        elif len(image.shape) == 3:
            image = image[None, ...]

        device = next(self.parameters()).device
        image = torch.as_tensor(image).to(device)

        A_detections = self.get_conics(image, min_score=confidence)[0]

        return self.database.query_position(A_detections=A_detections,
                                            T=attitude,
                                            K=self.camera_matrix
                                            )

    def __call__(self, image: Union[np.ndarray, torch.Tensor], *args, **kwargs):
        return self.derive_position(image)
