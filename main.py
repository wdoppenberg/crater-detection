import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from VPU import OpenVINOHandler
from craterdetection.deepmoon import template_match_t


def main():
    exp = OpenVINOHandler('DeepMoon', device='CPU')

    img = cv2.imread('sample_data/test_input.png', cv2.IMREAD_GRAYSCALE)
    batch = transforms.ToTensor()(img).unsqueeze_(0)

    out = exp.infer(batch)
    # extracted_rings = template_match_t(out[0, 0].copy(), minrad=3)

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    circles = cv2.HoughCircles(np.uint8(out[0, 0].copy()*255), cv2.HOUGH_GRADIENT, 1, 20, None, 160, 25, 1, 18)
    if circles is not None:
        # Get the (x, y, r) as integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the circles
        for x, y, r in circles:
            circle = plt.Circle((x, y), r, color='blue', fill=False, linewidth=2, alpha=0.5)
            axes[0].add_artist(circle)
    else:
        print("No circles found!")

    axes[0].imshow(img, cmap='Greys_r')
    axes[1].imshow(out[0, 0], cmap='Greys_r')
    plt.show()


if __name__ == "__main__":
    main()
