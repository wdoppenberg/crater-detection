import torch
from craterdetection.model import CraterUNet, export2onnx

checkpoint_path = 'blobs/CraterUNet.pth'
print(f"Loading {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path)

net = CraterUNet(1, 1)
net.load_state_dict(checkpoint)

export2onnx(net)
