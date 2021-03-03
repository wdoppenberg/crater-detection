""" Full assembly of the parts to form the complete network """
from .components import *


class DeepMoon(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, upsample=True, drop=0.0, sigmoid=True):
        super(DeepMoon, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 112)
        self.down1 = Down(112, 224)
        self.down2 = Down(224, 448)
        self.down3 = Down(448, 448)
        self.up1 = Up(896, 224, upsample, drop)
        self.up2 = Up(448, 112, upsample, drop)
        self.up3 = Up(224, 112, upsample, drop)
        self.outc = OutConv(112, n_classes, sigmoid)

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

    def export2onnx(self, path='blobs/DeepMoon.onnx'):
        dummy_input = torch.ones(1, 1, 256, 256)
        input_names = [n for n, _ in self.named_parameters()]
        output_name = ['output1']

        torch.onnx.export(self, dummy_input, path, verbose=False, input_names=input_names, output_names=output_name,
                          opset_version=11)

    @classmethod
    def from_keras(cls, deepmoon):
        model = cls()

        for param, w in zip(model.parameters(), deepmoon.weights):
            if len(param.shape) == 4:
                param.data = nn.Parameter(torch.Tensor(w.numpy().transpose(3, 2, 1, 0)))
            else:
                param.data = nn.Parameter(torch.Tensor(w.numpy()))

        return model
