import cv2
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms

from craterdetection import CraterUNet, export2onnx

checkpoint = torch.load('blobs/CraterUNet.pth')
net = CraterUNet(1, 1)
net.load_state_dict(checkpoint)
net.eval()
img = cv2.imread('sample_data/test_input.png', cv2.IMREAD_GRAYSCALE)

export2onnx(net)

batch = transforms.ToTensor()(img).unsqueeze_(0)

with torch.no_grad():
    out = net(batch)
    out = nn.Sigmoid()(out)
    out.squeeze_()

plt.imshow(out.numpy(), cmap='Greys_r')
plt.show()
