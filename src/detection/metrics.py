from typing import Tuple

import torch

from src.common.conics import conic_center, ellipse_axes, scale_det


def accuracy_values(A_pred: torch.Tensor, A_target: torch.Tensor, axis_margin: float = 0.4, dist_margin: float = 0.8) \
        -> Tuple[int, int, int]:
    m_target = conic_center(A_target)
    m_pred = conic_center(A_pred)

    matched_center_idxs = torch.cdist(m_target, m_pred).argmin(0)
    A_matched = A_target[matched_center_idxs]

    dist = gaussian_angle_distance(A_pred, A_matched)
    dist_mask = dist < dist_margin

    a_pred, b_pred = ellipse_axes(A_matched)
    a_target, b_target = ellipse_axes(A_matched)
    axis_mask = (((a_pred - a_target) / a_target).abs() < axis_margin) & (
            ((b_pred - b_target) / b_target).abs() < axis_margin)

    matches = dist_mask & axis_mask

    TP = matches.sum().item()
    FP = len(matches) - TP
    FN = len(A_target) - TP

    return TP, FP, FN


def precision_recall(A_pred: torch.Tensor, A_target: torch.Tensor, **accuracy_kwargs) -> Tuple[float, float]:
    TP, FP, FN = accuracy_values(A_pred, A_target, **accuracy_kwargs)
    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    return precision, recall


def f1_score(A_pred: torch.Tensor, A_target: torch.Tensor, **accuracy_kwargs) -> float:
    precision, recall = precision_recall(A_pred, A_target, **accuracy_kwargs)
    return (precision * recall) / ((precision + recall) / 2)


def mv_kullback_leibler_divergence(A1: torch.Tensor, A2: torch.Tensor, shape_only: bool = True) -> torch.Tensor:
    A1, A2 = map(scale_det, (A1, A2))
    cov1, cov2 = map(lambda arr: -arr[..., :2, :2], (A1, A2))
    m1, m2 = map(lambda arr: torch.vstack(tuple(conic_center(arr).T)).T[..., None], (A1, A2))

    trace_term = (torch.inverse(cov1) @ cov2).diagonal(dim2=-2, dim1=-1).sum(1)
    log_term = torch.log(torch.det(cov1) / torch.det(cov2))

    if shape_only:
        displacement_term = 0
    else:
        displacement_term = ((m1 - m2).transpose(-1, -2) @ cov1.inverse() @ (m1 - m2)).squeeze()

    return 0.5 * (trace_term + displacement_term - 2 + log_term)


def norm_mv_kullback_leibler_divergence(A1: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
    return 1 - torch.exp(-mv_kullback_leibler_divergence(A1, A2))


def gaussian_angle_distance(A1: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
    A1, A2 = map(scale_det, (A1, A2))
    cov1, cov2 = map(lambda arr: -arr[..., :2, :2], (A1, A2))
    m1, m2 = map(lambda arr: torch.vstack(tuple(conic_center(arr).T)).T[..., None], (A1, A2))

    frac_term = (4 * torch.sqrt(cov1.det() * cov2.det())) / (cov1 + cov2).det()
    exp_term = torch.exp(
        -0.5 * (m1 - m2).transpose(-1, -2) @ cov1 @ (cov1 + cov2).inverse() @ cov2 @ (m1 - m2)
    ).squeeze()

    return (frac_term * exp_term).arccos()
