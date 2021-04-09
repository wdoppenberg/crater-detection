from collections import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = F.threshold(pr, threshold, 0.)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


def dice_coefficient(pred, target, eps=1e-7):
    numerator = 2 * torch.sum(pred * target)
    denominator = torch.sum(pred + target)
    return (numerator + eps) / (denominator + eps)


class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-7, sigmoid=True):
        super(SoftDiceLoss, self).__init__()
        self.eps = eps
        self.sigmoid = sigmoid

    def forward(self, preds, targets):
        if self.sigmoid:
            preds = preds.sigmoid()

        score = dice_coefficient(preds, targets, self.eps)
        score = 1 - score.sum()
        return score


class BCEDiceLoss(nn.Module):
    def __init__(self,
                 lambda_bce=1.0,
                 lambda_dice=1.0,
                 eps=1e-7
                 ):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(self.eps)

    def forward(self, logits, targets):
        return (self.bce(logits, targets) * self.lambda_bce) + \
               (self.dice(logits, targets) * self.lambda_dice)