import cv2
from torchvision import transforms
import matplotlib.pyplot as plt

from VPU import NCSInferenceHandler
from craterdetection import template_match_t

exp = NCSInferenceHandler('DeepMoon')

img = cv2.imread('sample_data/test_input.png', cv2.IMREAD_GRAYSCALE)
batch = transforms.ToTensor()(img).unsqueeze_(0)

out = exp.infer(batch)
extracted_rings = template_match_t(out[0,0].copy(), minrad=3)

fig, axes = plt.subplots(1,2, figsize=(15,15))

for x, y, r in extracted_rings:
    circle = plt.Circle((x, y), r, color='blue', fill=False, linewidth=2, alpha=0.5)
    axes[0].add_artist(circle)

axes[0].imshow(img, cmap='Greys_r')
axes[1].imshow(out[0,0], cmap='Greys_r')
plt.show()
