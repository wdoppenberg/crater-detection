from torch.nn import Conv2d
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def create_detection_model(backbone_name='resnet18'):
    backbone = resnet_fpn_backbone(backbone_name, pretrained=True, trainable_layers=5)

    # Input image is grayscale -> in_channels = 1 instead of 3 (COCO)
    backbone.body.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    model = MaskRCNN(
        backbone=backbone,
        num_classes=2,
        min_size=256,
        max_size=256,
        image_mean=[0.156],
        image_std=[0.272]
    )

    return model
