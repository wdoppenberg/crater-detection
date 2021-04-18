from typing import Union

import numpy as np
import torch
from matplotlib import pyplot as plt, patches


def draw_patches(
        img: Union[np.ndarray, torch.Tensor],
        bboxes: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        scores: Union[np.ndarray, torch.Tensor],
        masks: Union[np.ndarray, torch.Tensor] = None,
        min_score: float = 0.,
        ax=None,
        return_fig: bool = False,
        figsize=(10, 10)
):
    img, bboxes, labels, scores, masks = map(lambda arr: arr.numpy() if isinstance(arr, torch.Tensor) else arr,
                                             (img, bboxes, labels, scores, masks))

    return_fig_check = False

    if ax is None:
        return_fig_check = True
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.imshow(img[0], cmap='gray')
    for (xmin, ymin, xmax, ymax), s in zip(bboxes, scores):
        if s < min_score: continue
        cx, cy, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
        patch = patches.Rectangle((cx - 0.5 * w, cy - 0.5 * h),
                                  w, h, fill=False, color="r")
        ax.add_patch(patch)
        bbox_props = dict(boxstyle="round", fc="cyan", ec="0.5", alpha=0.5)
        if w < 15:
            ax.text(cx - 1.2 * w, cy, f"{s:.0%}", ha="center", va="center", size=8, bbox=bbox_props)
        else:
            ax.text(cx, cy, f"{s:.0%}", ha="center", va="center", size=8, bbox=bbox_props)

    if return_fig and return_fig_check:
        return fig
