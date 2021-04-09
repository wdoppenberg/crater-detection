import torch.nn as nn


class TverskyLoss(nn.Module):
    def __init__(self,
                 alpha=0.5,
                 beta=0.5,
                 eps=1.0
                 ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, inputs, targets):
        inputs = nn.Sigmoid()(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        loss = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)

        return 1 - loss
