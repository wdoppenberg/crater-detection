import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(2,1)


from keras.models import load_model
deepmoon_model_path = '/Users/doppenberg/Documents/Workspaces/DeepMoon/zenodo_downloads/model_keras2.h5'

deepmoon = load_model(deepmoon_model_path)

img = cv2.imread('data/test_input.png', cv2.IMREAD_GRAYSCALE)
img = img.reshape(256, 256, 1)

out = deepmoon.predict(img[np.newaxis, :])

axes[0].imshow(out[0], cmap='Greys_r')

from craterdetection import CraterUNet
import torch
import torch.nn as nn
from torchvision import transforms

img = cv2.imread('data/test_input.png', cv2.IMREAD_GRAYSCALE)

unet = CraterUNet(1, 1)
for (name, param), w in zip(unet.named_parameters(), deepmoon.weights):
    if len(param.shape) == 4:
        param.data = nn.Parameter(torch.Tensor(w.numpy().transpose(3, 2, 1, 0)))
    else:
        param.data = nn.Parameter(torch.Tensor(w.numpy()))
unet.eval()

with torch.no_grad():
    out = unet(transforms.ToTensor()(img).unsqueeze_(0))

axes[1].imshow(nn.Sigmoid()(out).squeeze().numpy(), cmap='Greys_r')

torch.save(unet.state_dict(), 'CraterUNet.pth')

plt.show()