from typing import Union

import numpy as np
import torch

from src.detection import CraterDetector
from src.matching import CraterDatabase


class LunarNavigator(CraterDetector):
    def __init__(self, camera_matrix, latlims, longlims, diamlims, **kwargs):
        super().__init__(**kwargs)
        self.camera_matrix = camera_matrix
        self.database = CraterDatabase.from_file(latlims=latlims, longlims=longlims, diamlims=diamlims)

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
