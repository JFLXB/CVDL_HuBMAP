"""
Sources: 
 - https://github.com/Thanos-DB/FullyConvolutionalTransformer
 - https://openaccess.thecvf.com/content/WACV2023/papers/Tragakis_The_Fully_Convolutional_Transformer_for_Medical_Image_Segmentation_WACV_2023_paper.pdf
"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hubmap.models.fct.blocks import Block_encoder_bottleneck
from hubmap.models.fct.blocks import Block_decoder
from hubmap.models.fct.blocks import DS_out


class FCT(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, verbose: bool = False):
        # Added additional parameters to work with our task at hand.
        # Changed the following code and dependencies accordingly.
        super().__init__()
        self._verbose = verbose

        # attention heads and filters per block
        # Changes according to the specification given in the paper
        att_heads = [2, 4, 8, 16, 32, 16, 8, 4, 2]
        filters = [16, 32, 64, 128, 384, 128, 64, 32, 16]

        # number of blocks used in the model
        blocks = len(filters)

        stochastic_depth_rate = 0.0

        # probability for each block
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]

        self.drp_out = 0.3

        # Multi-scale input
        self.scale_img = nn.AvgPool2d(2, 2)

        # model
        self.block_1 = Block_encoder_bottleneck(
            "first", in_channels, filters[0], att_heads[0], dpr[0]
        )
        self.block_2 = Block_encoder_bottleneck(
            "second", filters[0], filters[1], att_heads[1], dpr[1], in_channels
        )
        self.block_3 = Block_encoder_bottleneck(
            "third", filters[1], filters[2], att_heads[2], dpr[2], in_channels
        )
        self.block_4 = Block_encoder_bottleneck(
            "fourth", filters[2], filters[3], att_heads[3], dpr[3], in_channels
        )
        self.block_5 = Block_encoder_bottleneck(
            "bottleneck", filters[3], filters[4], att_heads[4], dpr[4]
        )
        self.block_6 = Block_decoder(filters[4], filters[5], att_heads[5], dpr[5])
        self.block_7 = Block_decoder(filters[5], filters[6], att_heads[6], dpr[6])
        self.block_8 = Block_decoder(filters[6], filters[7], att_heads[7], dpr[7])
        self.block_9 = Block_decoder(filters[7], filters[8], att_heads[8], dpr[8])

        self.ds7 = DS_out(filters[6], num_classes)
        self.ds8 = DS_out(filters[7], num_classes)
        self.ds9 = DS_out(filters[8], num_classes)

    def forward(self, x):
        # Multi-scale input
        scale_img_2 = self.scale_img(x)
        scale_img_3 = self.scale_img(scale_img_2)
        scale_img_4 = self.scale_img(scale_img_3)

        x = self.block_1(x)
        print(f"Block 1 out -> {list(x.size())}") if self._verbose else None
        skip1 = x
        x = self.block_2(x, scale_img_2)
        print(f"Block 2 out -> {list(x.size())}") if self._verbose else None
        skip2 = x
        x = self.block_3(x, scale_img_3)
        print(f"Block 3 out -> {list(x.size())}") if self._verbose else None
        skip3 = x
        x = self.block_4(x, scale_img_4)
        print(f"Block 4 out -> {list(x.size())}") if self._verbose else None
        skip4 = x
        x = self.block_5(x)
        print(f"Block 5 out -> {list(x.size())}") if self._verbose else None
        x = self.block_6(x, skip4)
        print(f"Block 6 out -> {list(x.size())}") if self._verbose else None
        x = self.block_7(x, skip3)
        print(f"Block 7 out -> {list(x.size())}") if self._verbose else None
        skip7 = x
        x = self.block_8(x, skip2)
        print(f"Block 8 out -> {list(x.size())}") if self._verbose else None
        skip8 = x
        x = self.block_9(x, skip1)
        print(f"Block 9 out -> {list(x.size())}") if self._verbose else None
        skip9 = x

        out7 = self.ds7(skip7)
        print(f"DS 7 out -> {list(out7.size())}") if self._verbose else None
        out8 = self.ds8(skip8)
        print(f"DS 8 out -> {list(out8.size())}") if self._verbose else None
        out9 = self.ds9(skip9)
        print(f"DS 9 out -> {list(out9.size())}") if self._verbose else None

        return out7, out8, out9
