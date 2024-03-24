import torch.nn as nn
from hubmap.models.linknet.basicblock import BasicBlock

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Encoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.basic_block = BasicBlock(out_channels, out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.basic_block(out)

        return out