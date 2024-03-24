import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights
import torch.nn.functional as F
from hubmap.models.ducknet.convblock2d import ConvBlock2D


##Pretrained DUCKNet model with resnet18 as a backbone
class DUCKNetPretrained(nn.Module):
    def __init__(self, input_channels, out_classes):
        super(DUCKNetPretrained, self).__init__()

        print('Starting DUCK-Net with pretrained resnet18')

        # Load the pretrained ResNet18
        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Remove the last layers (avgpool and fc) since we won't be needing them.
        del self.resnet18.avgpool
        del self.resnet18.fc

        # Additional conv layers for p3, p4 and p5
        self.p3 = nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0)
        self.p4 = nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=0)  # 256 channels from resnet18.layer2
        self.p5 = nn.Conv2d(512, 1024, kernel_size=2, stride=2, padding=0)
        
        # Adjust starting_filters to match ResNet18's layer2 output channels
        self.starting_filters = 256

        # Adjusting t-blocks
        self.t0 = ConvBlock2D(input_channels, 64, block_type='duckv2', repeat=1)
        self.l1i = nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0)  # Adjusted for the output size of ResNet's conv1
        self.t1 = ConvBlock2D(64, 64, 'duckv2', repeat=1)

        self.l2i = nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0)
        self.t2 = ConvBlock2D(128, 128, 'duckv2', repeat=1)

        self.l3i = nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0)
        self.t3 = ConvBlock2D(256, 256, 'duckv2', repeat=1)

        self.l4i = nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=0)
        self.t4 = ConvBlock2D(512, 512, 'duckv2', repeat=1)

        self.l5i = nn.Conv2d(512, 1024, kernel_size=2, stride=2, padding=0)
        self.t51 = ConvBlock2D(1024, 1024, 'resnet', repeat=2)
        self.t53 = ConvBlock2D(1024, 512, 'resnet', repeat=2)


        # Upsampling for l4o, l3o, l2o, and l1o
        self.l5o = nn.Upsample(scale_factor=2, mode='nearest')
        self.l4o = nn.Upsample(scale_factor=2, mode='nearest')
        self.l3o = nn.Upsample(scale_factor=2, mode='nearest')
        self.l2o = nn.Upsample(scale_factor=2, mode='nearest')
        self.l1o = nn.Upsample(scale_factor=2, mode='nearest')

        # Adjusting q-blocks
        self.q4 = ConvBlock2D(512, 256, 'duckv2', repeat=1)
        self.q3 = ConvBlock2D(256, 128, 'duckv2', repeat=1)
        self.q2 = ConvBlock2D(128, 64, 'duckv2', repeat=1)
        self.q1 = ConvBlock2D(64, 64, 'duckv2', repeat=1)
        self.z1 = ConvBlock2D(64, 64, 'duckv2', repeat=1)

        # Last Convolutional layer to get the final output
        self.final_conv = nn.Conv2d(64, out_classes, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        # Extracting features using ResNet18
        x_conv = self.resnet18.conv1(x)
        x_conv = self.resnet18.bn1(x_conv)
        p1 = self.resnet18.relu(x_conv)
        p1_new = self.resnet18.layer1(p1)

        p2 = self.resnet18.layer2(p1_new)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        t0_out = self.t0(x)

        l1i_out = self.l1i(t0_out)
        s1_out = l1i_out + p1 # Updated to p2
        t1_out = self.t1(s1_out)

        l2i_out = self.l2i(t1_out)
        s2_out = l2i_out + p2  # Updated to p3
        t2_out = self.t2(s2_out)

        l3i_out = self.l3i(t2_out)
        s3_out = l3i_out + p3  # Updated to p4
        t3_out = self.t3(s3_out)

        l4i_out = self.l4i(t3_out)
        s4_out = l4i_out + p4  # Updated to p5
        t4_out = self.t4(s4_out)


        l5i_out = self.l5i(t4_out)
        s5_out = l5i_out + p5
        t51_out = self.t51(s5_out)
        t53_out = self.t53(t51_out)

        #SECOND PART
        
        l5o_out = self.l5o(t53_out)
        c4_out = l5o_out + t4_out
        q4_out = self.q4(c4_out)

        l4o_out = self.l4o(q4_out)
        c3_out = l4o_out + t3_out
        q3_out = self.q3(c3_out)

        l3o_out = self.l3o(q3_out)
        c2_out = l3o_out + t2_out
        q2_out = self.q2(c2_out)

        l2o_out = self.l2o(q2_out)
        c1_out = l2o_out + t1_out
        q1_out = self.q1(c1_out)

        l1o_out = self.l1o(q1_out)
        c0_out = l1o_out + t0_out
        z1_out = self.z1(c0_out)

        output = self.final_conv(z1_out)
        output = self.final_activation(output)

        return output
    


import torch
import torch.nn as nn
from torchvision.models import resnet34
from hubmap.models.ducknet.convblock2d import ConvBlock2D

class DUCKNetPretrained34(nn.Module):
    def __init__(self, input_channels, out_classes):
        super(DUCKNetPretrained34, self).__init__()

        print('Starting DUCK-Net with pretrained resnet34')

        # Load the pretrained ResNet34
        self.resnet34 = resnet34(weights=ResNet34_Weights.DEFAULT)
        
        # Remove the last layers (avgpool and fc) since we won't be needing them.
        del self.resnet34.avgpool
        del self.resnet34.fc

        # Additional conv layers for p3, p4 and p5
        # self.p3 = nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0)
        self.p4 = nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=0)  
        self.p5 = nn.Conv2d(512, 1024, kernel_size=2, stride=2, padding=0)

        # Adjusting t-blocks
        self.t0 = ConvBlock2D(input_channels, 64, block_type='duckv2', repeat=1)
        self.l1i = nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0)  
        self.t1 = ConvBlock2D(64, 64, 'duckv2', repeat=1)

        self.l2i = nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0)
        self.t2 = ConvBlock2D(128, 128, 'duckv2', repeat=1)

        self.l3i = nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0)
        self.t3 = ConvBlock2D(256, 256, 'duckv2', repeat=1)

        self.l4i = nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=0)
        self.t4 = ConvBlock2D(512, 512, 'duckv2', repeat=1)

        self.l5i = nn.Conv2d(512, 1024, kernel_size=2, stride=2, padding=0)
        self.t51 = ConvBlock2D(1024, 1024, 'resnet', repeat=2)
        self.t53 = ConvBlock2D(1024, 512, 'resnet', repeat=2)

        # Upsampling
        self.l5o = nn.Upsample(scale_factor=2, mode='nearest')
        self.l4o = nn.Upsample(scale_factor=2, mode='nearest')
        self.l3o = nn.Upsample(scale_factor=2, mode='nearest')
        self.l2o = nn.Upsample(scale_factor=2, mode='nearest')
        self.l1o = nn.Upsample(scale_factor=2, mode='nearest')

        # Adjusting q-blocks
        self.q4 = ConvBlock2D(512, 256, 'duckv2', repeat=1)
        self.q3 = ConvBlock2D(256, 128, 'duckv2', repeat=1)
        self.q2 = ConvBlock2D(128, 64, 'duckv2', repeat=1)
        self.q1 = ConvBlock2D(64, 64, 'duckv2', repeat=1)
        self.z1 = ConvBlock2D(64, 64, 'duckv2', repeat=1)

        # Last Convolutional layer
        self.final_conv = nn.Conv2d(64, out_classes, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        # Extracting features using ResNet18
        x_conv = self.resnet34.conv1(x)
        x_conv = self.resnet34.bn1(x_conv)
        p1 = self.resnet34.relu(x_conv)
        p1_new = self.resnet34.layer1(p1)

        p2 = self.resnet34.layer2(p1_new)
        p3 = self.resnet34.layer3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        t0_out = self.t0(x)

        l1i_out = self.l1i(t0_out)
        s1_out = l1i_out + p1 # Updated to p2
        t1_out = self.t1(s1_out)

        l2i_out = self.l2i(t1_out)
        s2_out = l2i_out + p2  # Updated to p3
        t2_out = self.t2(s2_out)

        l3i_out = self.l3i(t2_out)
        s3_out = l3i_out + p3  # Updated to p4
        t3_out = self.t3(s3_out)

        l4i_out = self.l4i(t3_out)
        s4_out = l4i_out + p4  # Updated to p5
        t4_out = self.t4(s4_out)


        l5i_out = self.l5i(t4_out)
        s5_out = l5i_out + p5
        t51_out = self.t51(s5_out)
        t53_out = self.t53(t51_out)

        #SECOND PART
        
        l5o_out = self.l5o(t53_out)
        c4_out = l5o_out + t4_out
        q4_out = self.q4(c4_out)

        l4o_out = self.l4o(q4_out)
        c3_out = l4o_out + t3_out
        q3_out = self.q3(c3_out)

        l3o_out = self.l3o(q3_out)
        c2_out = l3o_out + t2_out
        q2_out = self.q2(c2_out)

        l2o_out = self.l2o(q2_out)
        c1_out = l2o_out + t1_out
        q1_out = self.q1(c1_out)

        l1o_out = self.l1o(q1_out)
        c0_out = l1o_out + t0_out
        z1_out = self.z1(c0_out)

        output = self.final_conv(z1_out)
        output = self.final_activation(output)

        return output

