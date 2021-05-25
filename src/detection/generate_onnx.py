import torch
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn

from detection.training import CraterEllipseDataset, collate_fn
from src import CraterDetector

if __name__ == "__main__":
    model = CraterDetector()
    model.load_state_dict(torch.load("blobs/CraterRCNN.pth"))
    model.eval()
    device = torch.device('cpu')
    model.to(device)

    ds = CraterEllipseDataset(file_path="data/dataset_crater_detection_80k.h5", group="test")
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    images, targets = next(iter(loader))

    model(images)

    with torch.no_grad():
        torch.onnx.export(
            model,
            images[0][None, ...],
            "blobs/CraterRCNN.onnx",
            # do_constant_folding=True,
            verbose=True,
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=11,
            input_names=['image_in'],  # the model's input names
            output_names=['output']
        )
