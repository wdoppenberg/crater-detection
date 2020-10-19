from craterdetection import CraterUNet
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms

net = CraterUNet(1, 1)
net.eval()

img = Image.open(r'data/test_input.png').convert('L')

images = transforms.ToTensor()(img).unsqueeze_(0)

with torch.no_grad():
    out = net(images)

plt.imshow(out.squeeze(), cmap='Greys_r')
plt.show()
