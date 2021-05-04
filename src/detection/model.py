import cv2
import mlflow
import numpy as np
import torch
from torch.nn import Conv2d
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from common.conics import crater_representation


def create_detection_model(backbone_name='resnet18', image_size=256):
    backbone = resnet_fpn_backbone(backbone_name, pretrained=True, trainable_layers=5)

    # Input image is grayscale -> in_channels = 1 instead of 3 (COCO)
    backbone.body.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    model = MaskRCNN(
        backbone=backbone,
        num_classes=2,
        min_size=image_size,
        max_size=image_size,
        image_mean=[0.156],
        image_std=[0.272]
    )

    return model


class CraterDetector(MaskRCNN):
    def __init__(self,
                 backbone_name='resnet18',
                 image_size=256,
                 image_mean=0.156,
                 image_std=0.272,
                 **maskrcnn_kwargs
                 ):
        backbone = resnet_fpn_backbone(backbone_name, pretrained=True, trainable_layers=5)

        # Input image is grayscale -> in_channels = 1 instead of 3 (COCO)
        backbone.body.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        super().__init__(backbone,
                         num_classes=2,
                         min_size=image_size,
                         max_size=image_size,
                         image_mean=[image_mean],
                         image_std=[image_std],
                         **maskrcnn_kwargs
                         )

    def from_run_id(self, run_id):
        checkpoint = mlflow.pytorch.load_state_dict(f'runs:/{run_id}/artifacts/checkpoint')
        self.load_state_dict(checkpoint['model_state_dict'])

    @torch.no_grad()
    def get_conics(self, image, min_score=0.98):
        if self.training:
            raise RuntimeError("Conic fitting not available when in training mode.")

        if len(image) > 1:
            raise ValueError("Ellipse fitting works for single image batches only.")

        out = self(image)[0]

        masks = out['masks']
        scores = out['scores']
        masks = masks[scores > min_score]

        n_det = len(masks)
        A = np.zeros((n_det, 3, 3))

        for i in range(n_det):
            cnt = np.array(np.where(masks[i, 0].numpy() > 0.0)).T[:, None, :]
            cnt[..., [0, 1]] = cnt[..., [1, 0]]
            (x, y), (a, b), psi = cv2.fitEllipse(cnt)
            psi = np.radians(psi)
            A[i] = crater_representation(a, b, psi, x, y)

        return A