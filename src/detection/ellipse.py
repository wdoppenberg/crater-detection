from typing import List, Dict, Optional, Tuple

import torch
from torch import nn
from torchvision.models.detection.roi_heads import fastrcnn_loss, RoIHeads


class EllipseRegressor(nn.Module):
    def __init__(self, in_channels=1024, hidden_size=512):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, hidden_size)
        self.fc7 = nn.Linear(hidden_size, 5)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))

        return x


def ellipsercnn_loss(ellipse_logits: torch.Tensor, gt_ellipse_offsets: List[torch.Tensor],
                     pos_matched_idxs: List[torch.Tensor], loss_func=None) -> torch.Tensor:

    offset_targets = torch.cat([o[idxs] for o, idxs in zip(gt_ellipse_offsets, pos_matched_idxs)], dim=0)

    if offset_targets.numel() == 0:
        return ellipse_logits.sum() * 0

    if loss_func is None:
        loss_func = nn.SmoothL1Loss()

    mask_loss = loss_func(
        ellipse_logits, offset_targets
    )

    return mask_loss


class EllipseRoIHeads(RoIHeads):
    def __init__(self, box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh, batch_size_per_image,
                 positive_fraction, bbox_reg_weights, score_thresh, nms_thresh, detections_per_img,
                 ellipse_roi_pool, ellipse_head, ellipse_predictor):

        super().__init__(box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh, batch_size_per_image,
                         positive_fraction, bbox_reg_weights, score_thresh, nms_thresh, detections_per_img)

        self.ellipse_roi_pool = ellipse_roi_pool
        self.ellipse_head = ellipse_head
        self.ellipse_predictor = ellipse_predictor

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
                    pos = torch.where(labels[img_id] > 0)[0]
                    ellipse_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.ellipse_roi_pool is not None:
                ellipse_features = self.ellipse_roi_pool(features, ellipse_proposals, image_shapes)
                ellipse_features = self.ellipse_head(ellipse_features)
                ellipse_logits = self.ellipse_predictor(ellipse_features)
            else:
                raise Exception("Expected ellipse_roi_pool to be not None")

            loss_ellipse_offsets = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert ellipse_logits is not None

                gt_ellipse_offsets = [t["ellipse_offsets"] for t in targets]
                rcnn_loss_ellipse_reg = ellipsercnn_loss(
                    ellipse_logits, gt_ellipse_offsets, pos_matched_idxs
                )
                loss_ellipse_offsets = {
                    "loss_ellipse_offsets": rcnn_loss_ellipse_reg
                }
            else:
                ellipses_per_image = [l.shape[0] for l in labels]
                for e_l, r in zip(ellipse_logits.split(ellipses_per_image, dim=0), result):
                    r["ellipse_offsets"] = e_l

            losses.update(loss_ellipse_offsets)

        return result, losses
