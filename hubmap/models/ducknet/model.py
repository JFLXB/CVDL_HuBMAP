import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F
from hubmap.models.ducknet.convblock2d import ConvBlock2D

"""
Adapted from: https://github.com/RazvanDu/DUCK-Net
"""


class DUCKNet(nn.Module):
    def __init__(self, input_channels, out_classes, starting_filters):
        super(DUCKNet, self).__init__()

        print('Starting DUCK-Net')

        # Initial layers
        self.p1 = nn.Conv2d(input_channels, starting_filters * 2, kernel_size=2, stride=2, padding=0)
        self.p2 = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding=0)
        self.p3 = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding=0)
        self.p4 = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding=0)
        self.p5 = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding=0)

        # First t block
        self.t0 = ConvBlock2D(in_channels=input_channels, filters=starting_filters, block_type='duckv2', repeat=1)
        self.l1i = nn.Conv2d(starting_filters, starting_filters * 2, kernel_size=2, stride=2, padding=0)
        self.t1 = ConvBlock2D(starting_filters * 2, starting_filters * 2, 'duckv2', repeat=1)

        #second t block
        self.l2i = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding=0)
        self.t2 = ConvBlock2D(starting_filters * 4, starting_filters * 4, 'duckv2', repeat=1)

        #third t block
        self.l3i = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding=0)
        self.t3 = ConvBlock2D(starting_filters * 8, starting_filters * 8, 'duckv2', repeat=1)

        #fourth t block
        self.l4i = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding=0)
        self.t4 = ConvBlock2D(starting_filters * 16, starting_filters * 16, 'duckv2', repeat=1)

        #final t block
        self.l5i = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding=0)
        self.t51 = ConvBlock2D(starting_filters * 32, starting_filters * 32, 'resnet', repeat=2)
        self.t53 = ConvBlock2D(starting_filters * 32, starting_filters * 16, 'resnet', repeat=2)

        # Upsampling for l4o, l3o, l2o, and l1o
        self.l5o = nn.Upsample(scale_factor=2, mode='nearest')
        self.l4o = nn.Upsample(scale_factor=2, mode='nearest')
        self.l3o = nn.Upsample(scale_factor=2, mode='nearest')
        self.l2o = nn.Upsample(scale_factor=2, mode='nearest')
        self.l1o = nn.Upsample(scale_factor=2, mode='nearest')


        # Conv blocks q3, q2, q1, z1
        self.q4 = ConvBlock2D(starting_filters * 16, starting_filters * 8, 'duckv2', repeat=1)
        self.q3 = ConvBlock2D(starting_filters * 8, starting_filters * 4, 'duckv2', repeat=1)
        self.q2 = ConvBlock2D(starting_filters * 4, starting_filters * 2, 'duckv2', repeat=1)
        self.q1 = ConvBlock2D(starting_filters * 2, starting_filters, 'duckv2', repeat=1)
        self.z1 = ConvBlock2D(starting_filters, starting_filters, 'duckv2', repeat=1)

        # Last Convolutional layer to get the final output
        self.final_conv = nn.Conv2d(starting_filters, out_classes, kernel_size=1)
        self.final_activation = nn.Sigmoid()


    def forward(self, x):
        # Initial pyramid features extraction
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        t0_out = self.t0(x)
        l1i_out = self.l1i(t0_out)
        s1_out = l1i_out + p1
        t1_out = self.t1(s1_out)

        l2i_out = self.l2i(t1_out)
        s2_out = l2i_out + p2
        t2_out = self.t2(s2_out)

        l3i_out = self.l3i(t2_out)
        s3_out = l3i_out + p3
        t3_out = self.t3(s3_out)

        l4i_out = self.l4i(t3_out)
        s4_out = l4i_out + p4
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
