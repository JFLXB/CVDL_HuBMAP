"""
Sources: 
 - https://github.com/Thanos-DB/FullyConvolutionalTransformer
 - https://openaccess.thecvf.com/content/WACV2023/papers/Tragakis_The_Fully_Convolutional_Transformer_for_Medical_Image_Segmentation_WACV_2023_paper.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        proj_drop=0.0,
        kernel_size=3,
        stride_kv=1,
        stride_q=1,
        padding_kv="same",
        padding_q="same",
        attention_bias=True,
    ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop

        self.conv_q = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride_q,
            padding_q,
            bias=attention_bias,
            groups=channels,
        )
        self.layernorm_q = nn.LayerNorm(channels, eps=1e-5)
        self.conv_k = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride_kv,
            stride_kv,
            bias=attention_bias,
            groups=channels,
        )
        self.layernorm_k = nn.LayerNorm(channels, eps=1e-5)
        self.conv_v = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride_kv,
            stride_kv,
            bias=attention_bias,
            groups=channels,
        )
        self.layernorm_v = nn.LayerNorm(channels, eps=1e-5)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            bias=attention_bias,
            batch_first=True,
            # dropout = 0.0,
            num_heads=1,
        )  # num_heads=self.num_heads)

    def _build_projection(self, x, qkv):
        if qkv == "q":
            x1 = F.relu(self.conv_q(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "k":
            x1 = F.relu(self.conv_k(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "v":
            x1 = F.relu(self.conv_v(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v(x1)
            proj = x1.permute(0, 3, 1, 2)

        return proj

    def forward_conv(self, x):
        q = self._build_projection(x, "q")
        k = self._build_projection(x, "k")
        v = self._build_projection(x, "v")

        return q, k, v

    def forward(self, x):
        q, k, v = self.forward_conv(x)
        q = q.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        k = k.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        v = v.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)

        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(
            x1.shape[0],
            x1.shape[1],
            np.sqrt(x1.shape[2]).astype(int),
            np.sqrt(x1.shape[2]).astype(int),
        )
        x1 = F.dropout(x1, self.proj_drop)

        return x1


class Transformer(nn.Module):
    def __init__(
        self,
        out_channels,
        num_heads,
        dpr,
        proj_drop=0.0,
        attention_bias=True,
        padding_q="same",
        padding_kv="same",
        stride_kv=1,
        stride_q=1,
    ):
        super().__init__()

        self.attention_output = Attention(
            channels=out_channels,
            num_heads=num_heads,
            proj_drop=proj_drop,
            padding_q=padding_q,
            padding_kv=padding_kv,
            stride_kv=stride_kv,
            stride_q=stride_q,
            attention_bias=attention_bias,
        )

        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.layernorm = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.wide_focus = Wide_Focus(out_channels, out_channels)

    def forward(self, x):
        x1 = self.attention_output(x)
        x1 = self.conv1(x1)
        x2 = torch.add(x1, x)
        x3 = x2.permute(0, 2, 3, 1)
        x3 = self.layernorm(x3)
        x3 = x3.permute(0, 3, 1, 2)
        x3 = self.wide_focus(x3)
        x3 = torch.add(x2, x3)
        return x3


class Wide_Focus(nn.Module):
    """
    Wide-Focus module.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, 3, 1, padding="same", dilation=2
        )
        self.conv3 = nn.Conv2d(
            in_channels, out_channels, 3, 1, padding="same", dilation=3
        )
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, 0.1)
        x2 = self.conv2(x)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, 0.1)
        x3 = self.conv3(x)
        x3 = F.gelu(x3)
        x3 = F.dropout(x3, 0.1)
        added = torch.add(x1, x2)
        added = torch.add(added, x3)
        x_out = self.conv4(added)
        x_out = F.gelu(x_out)
        x_out = F.dropout(x_out, 0.1)
        return x_out


class Block_encoder_bottleneck(nn.Module):
    def __init__(
        self, blk, in_channels, out_channels, att_heads, dpr, image_in_channels=1
    ):
        super().__init__()
        self.blk = blk
        if (self.blk == "first") or (self.blk == "bottleneck"):
            self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.trans = Transformer(out_channels, att_heads, dpr)
        elif (self.blk == "second") or (self.blk == "third") or (self.blk == "fourth"):
            self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
            self.conv1 = nn.Conv2d(image_in_channels, in_channels, 3, 1, padding="same")
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.trans = Transformer(out_channels, att_heads, dpr)

    def forward(self, x, scale_img="none"):
        if (self.blk == "first") or (self.blk == "bottleneck"):
            x1 = x.permute(0, 2, 3, 1)
            x1 = self.layernorm(x1)
            x1 = x1.permute(0, 3, 1, 2)
            x1 = F.relu(self.conv1(x1))
            x1 = F.relu(self.conv2(x1))
            x1 = F.dropout(x1, 0.3)
            x1 = F.max_pool2d(x1, (2, 2))
            out = self.trans(x1)
            # without skip
        elif (self.blk == "second") or (self.blk == "third") or (self.blk == "fourth"):
            x1 = x.permute(0, 2, 3, 1)
            x1 = self.layernorm(x1)
            x1 = x1.permute(0, 3, 1, 2)
            x1 = torch.cat((F.relu(self.conv1(scale_img)), x1), axis=1)
            x1 = F.relu(self.conv2(x1))
            x1 = F.relu(self.conv3(x1))
            x1 = F.dropout(x1, 0.3)
            x1 = F.max_pool2d(x1, (2, 2))
            out = self.trans(x1)
            # with skip
        return out


class Block_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.trans = Transformer(out_channels, att_heads, dpr)

    def forward(self, x, skip):
        x1 = x.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.upsample(x1)
        x1 = F.relu(self.conv1(x1))
        x1 = torch.cat((skip, x1), axis=1)
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = F.dropout(x1, 0.3)
        out = self.trans(x1)
        return out


class DS_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        x1 = self.conv3(x1)
        # Above was changed from the original implementation:
        #   out = torch.sigmoid(self.conv3(x1))
        return x1
