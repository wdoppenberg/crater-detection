from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torchvision.models.detection.roi_heads import fastrcnn_loss, RoIHeads

from src.common.conics import crater_representation, conic_center, scale_det


class EllipseRegressor(nn.Module):
    def __init__(self, in_channels=1024, hidden_size=512, out_features=3):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, hidden_size)
        self.fc7 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = torch.sigmoid(self.fc6(x))
        x = torch.sigmoid(self.fc7(x))

        return x


def postprocess_ellipse_predictor(d_a: torch.Tensor, d_b: torch.Tensor, d_angle: torch.Tensor, boxes: torch.Tensor):
    box_diag = torch.sqrt((boxes[:, 2] - boxes[:, 0]) ** 2 + (boxes[:, 2] - boxes[:, 0]) ** 2)
    cx = boxes[:, 0] + ((boxes[:, 2] - boxes[:, 0]) / 2)
    cy = boxes[:, 1] + ((boxes[:, 3] - boxes[:, 1]) / 2)

    a, b = ((torch.exp(param) * box_diag / 2).T for param in (d_a, d_b))
    theta = d_angle * np.pi
    ang_cond1 = torch.cos(theta) >= 0
    ang_cond2 = ~ang_cond1

    theta[ang_cond1] = torch.atan2(torch.sin(theta[ang_cond1]), torch.cos(theta[ang_cond1]))
    theta[ang_cond2] = torch.atan2(-torch.sin(theta[ang_cond2]), -torch.cos(theta[ang_cond2]))

    return a, b, theta, cx, cy


def mv_kullback_leibler_divergence(A1: torch.Tensor, A2: torch.Tensor, shape_only: bool = False):
    cov1, cov2 = map(lambda arr: arr[..., :2, :2], (A1, A2))
    m1, m2 = map(lambda arr: torch.vstack(tuple(conic_center(arr).T)).T[..., None], (A1, A2))

    trace_term = (torch.inverse(cov1) @ cov2).diagonal(dim2=-2, dim1=-1).sum(1)
    log_term = torch.log(torch.det(cov1) / torch.det(cov2))

    if shape_only:
        displacement_term = 0
    else:
        displacement_term = ((m1 - m2).transpose(-1, -2) @ cov1.inverse() @ (m1 - m2)).squeeze()

    return 0.5 * (trace_term + displacement_term - 2 + log_term)


def ellipse_loss_KLD(d_pred: torch.Tensor, ellipse_matrix_targets: List[torch.Tensor],
                     pos_matched_idxs: List[torch.Tensor], boxes: List[torch.Tensor], multiplier: float = 1.):
    A_target = torch.cat([o[idxs] for o, idxs in zip(ellipse_matrix_targets, pos_matched_idxs)], dim=0)
    boxes = torch.cat(boxes, dim=0)

    if A_target.numel() == 0:
        return d_pred.sum() * 0

    d_a = d_pred[:, 0]
    d_b = d_pred[:, 1]
    d_angle = d_pred[:, 2]

    A_pred = crater_representation(*postprocess_ellipse_predictor(d_a, d_b, d_angle, boxes))

    A_pred, A_target = map(scale_det, (A_pred, A_target))

    loss1 = mv_kullback_leibler_divergence(A_pred, A_target, shape_only=True)
    loss2 = mv_kullback_leibler_divergence(A_target, A_pred, shape_only=True)

    return multiplier * (loss1 + loss2).mean()


class EllipseRoIHeads(RoIHeads):
    def __init__(self, box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh, batch_size_per_image,
                 positive_fraction, bbox_reg_weights, score_thresh, nms_thresh, detections_per_img,
                 ellipse_roi_pool, ellipse_head, ellipse_predictor, ellipse_loss=ellipse_loss_KLD, min_class_score=0.5):

        super().__init__(box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh, batch_size_per_image,
                         positive_fraction, bbox_reg_weights, score_thresh, nms_thresh, detections_per_img)

        self.ellipse_roi_pool = ellipse_roi_pool
        self.ellipse_head = ellipse_head
        self.ellipse_predictor = ellipse_predictor
        self.ellipse_loss = ellipse_loss
        self.min_class_score = min_class_score

    def has_ellipse_reg(self):
        if self.ellipse_roi_pool is None:
            return False
        if self.ellipse_head is None:
            return False
        if self.ellipse_predictor is None:
            return False
        return True

    def forward(self,
                features,  # type: Dict[str, torch.Tensor]
                proposals,  # type: List[torch.Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None  # type: Optional[List[Dict[str, torch.Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["ellipse_matrices"].dtype in floating_point_types, 'target ellipse_offsets must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_ellipse_reg():
            ellipse_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                ellipse_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > self.min_class_score)[0]
                    ellipse_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.ellipse_roi_pool is not None:
                ellipse_features = self.ellipse_roi_pool(features, ellipse_proposals, image_shapes)
                ellipse_features = self.ellipse_head(ellipse_features)
                ellipse_shapes_normalised = self.ellipse_predictor(ellipse_features)
            else:
                raise Exception("Expected ellipse_roi_pool to be not None")

            loss_ellipse_offsets = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert ellipse_shapes_normalised is not None

                ellipse_matrix_targets = [t["ellipse_matrices"] for t in targets]
                rcnn_loss_ellipse = self.ellipse_loss(
                    ellipse_shapes_normalised, ellipse_matrix_targets, pos_matched_idxs, ellipse_proposals, multiplier=5.
                )
                loss_ellipse_offsets = {
                    "loss_ellipse_similarity": rcnn_loss_ellipse
                }
            else:
                ellipses_per_image = [l.shape[0] for l in labels]
                for e_l, r, box in zip(ellipse_shapes_normalised.split(ellipses_per_image, dim=0), result, ellipse_proposals):
                    d_a = e_l[:, 0]
                    d_b = e_l[:, 1]
                    d_angle = e_l[:, 2]
                    r["ellipse_matrices"] = crater_representation(*postprocess_ellipse_predictor(d_a, d_b, d_angle, box))

            losses.update(loss_ellipse_offsets)

        return result, losses
