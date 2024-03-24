"""
Sources:
 - https://github.com/nikhilroxtomar/TransResUNet
 - https://arxiv.org/pdf/2206.08985.pdf

Changes were made to the original code to work with our task at hand.
"""
import torch
import torch.nn as nn
from hubmap.models.trans_res_u_net.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
    wide_resnet50_2,
    wide_resnet101_2,
)
import numpy as np
import cv2


def save_feats_mean(x):
    b, c, h, w = x.shape
    if h == 256:
        with torch.no_grad():
            x = x.detach().cpu().numpy()
            x = np.transpose(x[0], (1, 2, 0))
            x = np.mean(x, axis=-1)
            x = x / np.max(x)
            x = x * 255.0
            x = x.astype(np.uint8)
            x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
            x = np.array(x, dtype=np.uint8)
            return x


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0), 
            nn.BatchNorm2d(out_c)
        )

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.r1 = ResidualBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.r1(inputs)
        p = self.pool(x)
        return x, p


class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, dim, num_layers=2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.tblock = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        b, c, h, w = x.shape
        x = x.reshape((b, c, h * w))
        x = self.tblock(x)
        x = x.reshape((b, c, h, w))
        x = self.conv2(x)
        return x


class DilatedConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=9, dilation=9),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

        self.c5 = nn.Sequential(
            nn.Conv2d(out_c * 4, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.c5(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c[0] + in_c[1], out_c)
        self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r1(x)
        x = self.r2(x)
        return x


backbone_map = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnext50_32x4d": resnext50_32x4d,
    "resnext101_32x8d": resnext101_32x8d,
    "wide_resnet50_2": wide_resnet50_2,
    "wide_resnet101_2": wide_resnet101_2,
}


class TResUnet(nn.Module):
    def __init__(
        self, num_classes: int, backbone: str = "resnet50", pretrained: bool = True, verbose: bool = False
    ):
        super().__init__()
        self.verbose = verbose

        """ ResNet X """
        backbone = backbone_map[backbone](pretrained=pretrained)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        """ Bridge blocks """
        self.b1 = Bottleneck(1024, 256, 256, num_layers=2)
        self.b2 = DilatedConv(1024, 256)

        """ Decoder """
        self.d1 = DecoderBlock([512, 512], 256)
        self.d2 = DecoderBlock([256, 256], 128)
        self.d3 = DecoderBlock([128, 64], 64)
        self.d4 = DecoderBlock([64, 3], 32)

        self.output = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x, heatmap=None):
        s0 = x
        print(s0.size()) if self.verbose else None
        s1 = self.layer0(s0)  ## [-1, 64, h/2, w/2]
        print(s1.size()) if self.verbose else None
        s2 = self.layer1(s1)  ## [-1, 256, h/4, w/4]
        print(s2.size()) if self.verbose else None
        s3 = self.layer2(s2)  ## [-1, 512, h/8, w/8]
        print(s3.size()) if self.verbose else None
        s4 = self.layer3(s3)  ## [-1, 1024, h/16, w/16]
        print(s4.size()) if self.verbose else None

        b1 = self.b1(s4)
        print(b1.size()) if self.verbose else None
        b2 = self.b2(s4)
        print(b2.size()) if self.verbose else None
        b3 = torch.cat([b1, b2], axis=1)
        print(b3.size()) if self.verbose else None

        d1 = self.d1(b3, s3)
        print(d1.size()) if self.verbose else None
        d2 = self.d2(d1, s2)
        print(d2.size()) if self.verbose else None
        d3 = self.d3(d2, s1)
        print(d3.size()) if self.verbose else None
        d4 = self.d4(d3, s0)
        print(d4.size()) if self.verbose else None

        y = self.output(d4)
        print(y.size()) if self.verbose else None

        if heatmap != None:
            hmap = save_feats_mean(d4)
            return hmap, y
        else:
            return y


class TResUnet512(nn.Module):
    """
    Adaption of the `TResUnet` model by us, to work with 512x512 images.
    """
    def __init__(
        self, num_classes: int, backbone: str = "resnet50", pretrained: bool = True, verbose: bool = False
    ):
        super().__init__()
        self.verbose = verbose

        """ ResNet X """
        backbone = backbone_map[backbone](pretrained=pretrained)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        """ Bridge blocks """
        self.b1 = Bottleneck(2048, 512, 256, num_layers=2)
        self.b2 = DilatedConv(2048, 512)

        """ Decoder """
        self.d1 = DecoderBlock([1024, 1024], 512)
        self.d2 = DecoderBlock([512, 512], 256)
        self.d3 = DecoderBlock([256, 256], 128)
        self.d4 = DecoderBlock([128, 64], 64)
        self.d5 = DecoderBlock([64, 3], 32)

        self.output = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x, heatmap=None):
        s0 = x
        print(s0.size()) if self.verbose else None
        s1 = self.layer0(s0)  ## [-1, 64, h/2, w/2]
        print(s1.size()) if self.verbose else None
        s2 = self.layer1(s1)  ## [-1, 256, h/4, w/4]
        print(s2.size()) if self.verbose else None
        s3 = self.layer2(s2)  ## [-1, 512, h/8, w/8]
        print(s3.size()) if self.verbose else None
        s4 = self.layer3(s3)  ## [-1, 1024, h/16, w/16]
        print(s4.size()) if self.verbose else None
        s5 = self.layer4(s4)  ## [-1, 2048, h/32, w/32]
        print(s5.size()) if self.verbose else None

        b1 = self.b1(s5)
        print(b1.size()) if self.verbose else None
        b2 = self.b2(s5)
        print(b2.size()) if self.verbose else None
        b3 = torch.cat([b1, b2], axis=1)
        print(b3.size()) if self.verbose else None

        d1 = self.d1(b3, s4)
        print(d1.size()) if self.verbose else None
        d2 = self.d2(d1, s3)
        print(d2.size()) if self.verbose else None
        d3 = self.d3(d2, s2)
        print(d3.size()) if self.verbose else None
        d4 = self.d4(d3, s1)
        print(d4.size()) if self.verbose else None
        d5 = self.d5(d4, s0)
        print(d5.size()) if self.verbose else None

        y = self.output(d5)
        print(y.size()) if self.verbose else None

        if heatmap != None:
            hmap = save_feats_mean(d4)
            return hmap, y
        else:
            return y


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = TResUnet(num_classes=4)
    y = model(x)
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    x = torch.randn((1, 3, 512, 512))
    model = TResUnet512(num_classes=4)
    y = model(x)

