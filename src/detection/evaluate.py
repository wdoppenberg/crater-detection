import time

import numpy as np
import torch
from astropy.coordinates import cartesian_to_spherical
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm as tq

from common import constants as const
from common.conics import plot_conics
from detection.metrics import detection_metrics
from detection.training import CraterEllipseDataset, collate_fn
from src import CraterDetector


class Evaluator:
    def __init__(self, model=None, device="cpu", dataset_path="data/dataset_crater_detection.h5",
                 batch_size=32):
        if model is None:
            self._model = CraterDetector()
            self._model.load_state_dict(torch.load("blobs/CraterRCNN.pth"))
        else:
            self._model = model

        self._model.eval()

        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise ValueError("Type for device is incorrect, must be str or torch.device.")

        self._model.to(self.device)

        self.ds = CraterEllipseDataset(file_path=dataset_path, group="test")
        self.loader = DataLoader(self.ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    @torch.no_grad()
    def make_grid(self, n_rows=3, n_cols=4, min_score=0.6):
        i = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

        loader = DataLoader(self.ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

        for row in range(n_rows):
            for col in range(n_cols):
                images, target = next(iter(loader))
                images = list(image.to(self.device) for image in images)
                A_craters_pred = self._model.get_conics(images, min_score=min_score)

                A_craters_target = target[0]["ellipse_matrices"]
                position = target[0]["position"]
                r, lat, long = cartesian_to_spherical(*position.numpy())

                textstr = '\n'.join((
                    rf'$\varphi={np.degrees(lat.value)[0]:.1f}^o$',
                    rf'$\lambda={np.degrees(long.value)[0]:.1f}^o$',
                    rf'$h={r.value[0] - const.RMOON:.0f}$ km',
                ))

                axes[row, col].imshow(images[0][0].cpu().numpy(), cmap='gray')
                axes[row, col].axis("off")
                axes[row, col].set_title(i)

                plot_conics(A_craters_target, ax=axes[row, col], rim_color='cyan')
                plot_conics(A_craters_pred.cpu(), ax=axes[row, col])

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                axes[row, col].text(0.05, 0.95, textstr, transform=axes[row, col].transAxes, fontsize=14,
                                    verticalalignment='top', bbox=props)

                i += 1
        fig.tight_layout()

        return fig

    @torch.no_grad()
    def get_scores(self, iou_threshold=0.5, min_class_score=0.75):
        bar = tq(self.loader, desc=f"Testing",
                 postfix={
                     "IoU": 0.,
                     "GA_distance": 0.,
                     "precision": 0.,
                     "recall": 0.,
                     "f1_score": 0.
                 })

        precision_list = torch.zeros(len(self.loader), device=self.device)
        recall_list = torch.zeros(len(self.loader), device=self.device)
        f1_list = torch.zeros(len(self.loader), device=self.device)
        iou_list = torch.zeros(len(self.loader), device=self.device)
        dist_list = torch.zeros(len(self.loader), device=self.device)

        for batch, (images, targets_all) in enumerate(bar):
            images = list(image.to(self.device) for image in images)
            targets_all = [{k: v.to(self.device) for k, v in t.items()} for t in targets_all]

            pred_all = self._model(images)
            batch_iou, batch_dist, batch_precision, batch_recall, batch_f1 = torch.zeros(len(pred_all)), torch.zeros(
                len(pred_all)), torch.zeros(len(pred_all)), torch.zeros(len(pred_all)), torch.zeros(len(pred_all))

            for i, (pred, targets) in enumerate(zip(pred_all, targets_all)):
                batch_precision[i], batch_recall[i], batch_f1[i], batch_iou[i], batch_dist[i] = detection_metrics(pred,
                                                                                                                  targets,
                                                                                                                  iou_threshold=iou_threshold)

            batch_iou, batch_dist, batch_precision, batch_recall, batch_f1 = map(lambda x: x[x != 0].mean().item(),
                                                                                 (batch_iou, batch_dist,
                                                                                  batch_precision, batch_recall,
                                                                                  batch_f1
                                                                                  ))

            postfix = dict(
                IoU=batch_iou,
                GA_distance=batch_dist,
                precision=batch_precision,
                recall=batch_recall,
                f1_score=batch_f1
            )
            bar.set_postfix(ordered_dict=postfix)

            iou_list[batch] = batch_iou
            dist_list[batch] = batch_dist
            precision_list[batch] = batch_precision
            recall_list[batch] = batch_recall
            f1_list[batch] = batch_f1

        del images, targets_all

        return iou_list, dist_list, precision_list, recall_list, f1_list

    def iou_sweep(self, iou_range=(0.5, 0.95), steps=10):
        iou, dist, precision, recall, f1, iou_thresholds = torch.zeros(steps, device=self.device), \
                                                           torch.zeros(steps, device=self.device), \
                                                           torch.zeros(steps, device=self.device), \
                                                           torch.zeros(steps, device=self.device), \
                                                           torch.zeros(steps, device=self.device), \
                                                           torch.zeros(steps, device=self.device)

        for i, iou_threshold in enumerate(torch.linspace(0.5, 0.95, 10)):
            iou_thresholds[i] = iou_threshold
            iou_list, dist_list, precision_list, recall_list, f1_list = self.get_scores(iou_threshold=iou_threshold)
            iou[i], dist[i], precision[i], recall[i], f1[i] = iou_list.mean(), dist_list.mean(), \
                                                              precision_list.mean(), recall_list.mean(), f1_list.mean()
            time.sleep(1)
            print(f"\n[IoU threshold: {iou_threshold:.2f}]\n\tIoU: {iou[i]:.3f}, GA distance: {dist[i]:.3f}, "
                  f"AP: {precision[i]:.1%}, Recall: {recall[i]:.1%}, F1 score: {f1[i]:.3f}\n")

        return iou, dist, precision, recall, f1, iou_thresholds


if __name__ == "__main__":
    ev = Evaluator(device="cuda")

    ev.make_grid()
    plt.show()

    ev.iou_sweep()
