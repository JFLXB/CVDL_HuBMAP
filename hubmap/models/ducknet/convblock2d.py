import torch
import torch.nn as nn
import torch.nn.functional as F

####### GPT 4 VERSION ##########

    
class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, filters, block_type, repeat=1, dilation_rate=1, size=3, padding=1):
        super(ConvBlock2D, self).__init__()
        self.blocks = nn.ModuleList()

        for _ in range(repeat):
            if block_type == 'separated':
                self.blocks.append(SeparatedConv2DBlock(in_channels, filters, size, padding))
            elif block_type == 'duckv2':
                self.blocks.append(DuckV2Conv2DBlock(in_channels, filters, size))
            elif block_type == 'midscope':
                self.blocks.append(MidScopeConv2DBlock(in_channels, filters))
            elif block_type == 'widescope':
                self.blocks.append(WideScopeConv2DBlock(in_channels, filters))
            elif block_type == 'resnet':
                self.blocks.append(ResNetConv2DBlock(in_channels, filters, dilation_rate))
            elif block_type == 'conv':
                self.blocks.extend([
                    nn.Conv2d(in_channels, filters, kernel_size=size, padding=padding),
                    nn.ReLU()
                ])
            elif block_type == 'double_convolution':
                self.blocks.append(DoubleConv2DWithBatchNorm(in_channels, filters, dilation_rate))
            else:
                raise ValueError(f"Unsupported block_type: {block_type}")

            in_channels = filters

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x



class DuckV2Conv2DBlock(nn.Module):
    # ... (The implementation remains largely the same, using PyTorch constructs)
    def __init__(self, in_channels, filters, size=3, padding=1):
        super(DuckV2Conv2DBlock, self).__init__()
        
        # Initially set self.norm1 to None
        self.bn_initial = None
        self.initialized = False

        self.widescope = WideScopeConv2DBlock(in_channels, filters)
        self.midscope = MidScopeConv2DBlock(in_channels, filters)
        self.resnet1 = ResNetConv2DBlock(in_channels, filters)
        self.resnet2 = ResNetConv2DBlock(in_channels, filters)
        self.resnet3 = ResNetConv2DBlock(in_channels, filters)
        self.separated = SeparatedConv2DBlock(in_channels, filters, size=6, padding=1)

        self.bn_final = nn.BatchNorm2d(filters)

    def forward(self, x):

        # Initialize self.norm1 on the first forward pass
        if not self.initialized:
            self.bn_initial = nn.BatchNorm2d(x.size(1)).to(x.device)
            self.initialized = True

        x = self.bn_initial(x)

        x1 = self.widescope(x)
        x2 = self.midscope(x)
        x3 = self.resnet1(x)
        x4 = self.resnet2(x)
        x5 = self.resnet3(x)
        x6 = self.separated(x)
        x = x1 + x2 + x3 + x4 + x5 + x6
        x = self.bn_final(x)
        return x
    

class SeparatedConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size=3, padding=0):
        super(SeparatedConv2DBlock, self).__init__()

        #asymmetric padding: 
        self.pad = nn.ConstantPad2d((0, 3, 0, 3), 0)

        # First convolution (vertical) + batch normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, size), 
                               padding=(0, padding), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution (horizontal) + batch normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(size, 1),   #is out_channel correct?
                               padding=(padding, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)

        return x


class MidScopeConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(MidScopeConv2DBlock, self).__init__()

        # First convolution + batch normalization with dilation rate 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                               padding=padding, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution + batch normalization with dilation rate 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, #is out_channel correct?
                               padding=2, dilation=2, bias=False)  # Note: adjusted padding to account for dilation
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)

        return x


class WideScopeConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(WideScopeConv2DBlock, self).__init__()

        # First convolution + batch normalization with dilation rate 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               padding=padding, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution + batch normalization with dilation rate 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, #is out_channel correct?
                               padding=2, dilation=2, bias=False)  # Note: adjusted padding to account for dilation
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Third convolution + batch normalization with dilation rate 3
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, #is out_channel correct?
                               padding=3, dilation=3, bias=False)  # Note: adjusted padding to account for dilation
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)

        return x


class ResNetConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1, kernel_size=3, padding=1):
        super(ResNetConv2DBlock, self).__init__()

        # Shortcut connection (1x1 convolution without batch normalization)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                  padding=0, dilation=dilation_rate, bias=True)
        self.shortcut_activation = nn.ReLU()

        # First convolution + batch normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               padding=padding, dilation=dilation_rate, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution + batch normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,   #Is this correct?
                               padding=padding, dilation=dilation_rate, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Final batch normalization after the addition of the shortcut connection
        self.final_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.shortcut(x)
        x1 = self.shortcut_activation(x1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)

        # Adding the shortcut connection to the result
        x_final = x + x1

        x_final = self.final_bn(x_final)

        return x_final


class DoubleConv2DWithBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1, kernel_size=3, padding=1):
        super(DoubleConv2DWithBatchNorm, self).__init__()
        
        # First convolution + batch normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                               dilation=dilation_rate, bias=False) # Note: bias is set to False since BatchNorm will handle it.
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Second convolution + batch normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding,
                               dilation=dilation_rate, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        
        return x