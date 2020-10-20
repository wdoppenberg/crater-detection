import cv2
from torchvision import transforms
import matplotlib.pyplot as plt

from VPU import NCSInferenceHandler

exp = NCSInferenceHandler('DeepMoon')

img = cv2.imread('sample_data/test_input.png', cv2.IMREAD_GRAYSCALE)
batch = transforms.ToTensor()(img).unsqueeze_(0)

out = exp.infer(batch)

plt.imshow(out[0][0], cmap='Greys_r')
plt.show()
