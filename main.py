import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

from VPU import OpenVINOHandler
from craterdetection.deepmoon import template_match_t


def main():
    exp = OpenVINOHandler('DeepMoon', device='CPU')

    img = cv2.imread('sample_data/test_input.png', cv2.IMREAD_GRAYSCALE)
    batch = transforms.ToTensor()(img).unsqueeze_(0)

    out = exp.infer(batch)
    extracted_rings = template_match_t(out[0, 0].copy(), minrad=3)

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    for x, y, r in extracted_rings:
        circle = plt.Circle((x, y), r, color='blue', fill=False, linewidth=2, alpha=0.5)
        axes[0].add_artist(circle)

    axes[0].imshow(img, cmap='Greys_r')
    axes[1].imshow(out[0, 0], cmap='Greys_r')
    plt.show()


if __name__ == "__main__":
    main()
