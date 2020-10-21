import torch
from craterdetection.deepmoon import DeepMoon

checkpoint_path = 'blobs/DeepMoon.pth'
print(f"Loading {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path)

net = DeepMoon()
net.load_state_dict(checkpoint)

net.export2onnx()
