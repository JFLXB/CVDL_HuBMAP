import torch.nn as nn
import torch
from torchvision.models import resnet18
import torch.nn.functional as F
from hubmap.models.linknet.decoder import Decoder
from hubmap.models.linknet.encoder import Encoder

<<<<<<< HEAD
=======
"""
Source: https://github.com/e-lab/pytorch-linknet
"""
>>>>>>> main

class LinkNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(LinkNet, self).__init__()

        self.num_classes = num_classes

        # Load the pretrained ResNet18 model
        resnet = resnet18(pretrained=pretrained)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder2 = Decoder(128, 64)
        self.decoder1 = Decoder(64, 64)

        self.final_deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.final_bn1 = nn.BatchNorm2d(32)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Decoder
        dec4 = self.decoder4(enc4)
        dec4 = torch.add(dec4, enc3)  # Use torch.add() instead of +=
        dec3 = self.decoder3(dec4)
        dec3 = torch.add(dec3, enc2)  # Use torch.add() instead of +=
        dec2 = self.decoder2(dec3)
        dec2 = torch.add(dec2, enc1)  # Use torch.add() instead of +=
        dec1 = self.decoder1(dec2)

        # Final Convolution
        x = self.final_deconv1(dec1)
        x = self.final_bn1(x)
        x = self.final_relu1(x)
        x = self.final_conv2(x)

        # Upsample to the original input size
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        return x[:, :self.num_classes, :, :]