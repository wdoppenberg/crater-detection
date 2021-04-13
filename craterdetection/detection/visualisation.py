from typing import Union, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt, patches


def draw_patches(
        img: Union[np.ndarray, torch.TensorType],
        bboxes: Union[np.ndarray, torch.TensorType],
        labels: Union[np.ndarray, torch.TensorType],
        scores: Union[np.ndarray, torch.TensorType],
        masks: Union[np.ndarray, torch.TensorType] = None,
        min_score: float = 0.,
        ax = None,
        return_fig: bool = False,
        figsize: Tuple[int, int] = (10, 10)
):
    img, bboxes, labels, scores, masks = map(lambda arr: arr.numpy() if isinstance(arr, torch.TensorType) else arr,
                                             (img, bboxes, labels, scores, masks))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.imshow(img[0], cmap='gray')
    for (xmin, ymin, xmax, ymax), s in zip(bboxes, scores):
        if s < min_score: continue
        cx, cy, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
        ax.add_patch(patches.Rectangle((cx - 0.5 * w, cy - 0.5 * h),
                                       w, h, fill=False, color="r"))
        bbox_props = dict(boxstyle="round", fc="y", ec="0.5", alpha=0.3)
        ax.text(cx - 0.5 * w, cy - 0.5 * h, f"{s:.0%}", ha="center", va="center", size=10, bbox=bbox_props)