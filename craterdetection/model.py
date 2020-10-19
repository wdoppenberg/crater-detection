""" Full assembly of the parts to form the complete network """
from .components import *

class CraterDetection(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, drop=0.0):
        super(CraterDetection, self).__init__()
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