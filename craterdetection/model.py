""" Full assembly of the parts to form the complete network """
from .components import *
import torch

class CraterUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, drop=0.0):
        super(CraterUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 112)
        self.down1 = Down(112, 224)
        self.down2 = Down(224, 448)
        self.down3 = Down(448, 448)
        self.up1 = Up(896, 224, bilinear, drop)
        self.up2 = Up(448, 112, bilinear, drop)
        self.up3 = Up(224, 112, bilinear, drop)
        self.outc = OutConv(112, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


def export2onnx(model, path='blobs/CraterUNet.onnx'):
    dummy_input = torch.ones(1, 1, 256, 256)
    input_names = [n for n, _ in model.named_parameters()]
    output_name = ['output1']

    torch.onnx.export(model, dummy_input, path, verbose=True, input_names=input_names, output_names=output_name, opset_version=10)


def deepmoon2torch(deepmoon, model=None):
    if model is None:
        model = CraterUNet(1, 1)

    for param, w in zip(model.parameters(), deepmoon.weights):
        if len(param.shape) == 4:
            param.data = nn.Parameter(torch.Tensor(w.numpy().transpose(3, 2, 1, 0)))
        else:
            param.data = nn.Parameter(torch.Tensor(w.numpy()))

    return model