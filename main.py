import torch
from torch import nn
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

from craterdetection import CraterUNet


checkpoint = torch.load('blobs/CraterUNet.pth')
net = CraterUNet(1, 1)
net.load_state_dict(checkpoint)
net.eval()
img = cv2.imread('data/test_input.png', cv2.IMREAD_GRAYSCALE)
batch = transforms.ToTensor()(img).unsqueeze_(0)

with torch.no_grad():
    out = net(batch)
    out = nn.Sigmoid()(out)
    out.squeeze_()

plt.imshow(out.numpy(), cmap='Greys_r')
plt.show()
