"""
Code is based on `DiceBCELoss` implementation of
https://github.com/nikhilroxtomar/TransResUNet
"""
import torch
import torch.nn.functional as F
import torch.nn as nn


class ChannelWeightedDiceBCELoss(nn.Module):
    def __init__(self, weights=torch.tensor([1, 1, 1, 1])):
        super(ChannelWeightedDiceBCELoss, self).__init__()
        self.weights = weights

    def forward(self, inputs, targets, smooth=1):
        w = self.weights.unsqueeze(0).repeat(inputs.size(0), 1)
        w = w.to(inputs.device)        
        inputs = torch.sigmoid(inputs)

        intersection = (inputs * targets).sum((-2, -1))
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            inputs.sum((-2, -1)) + targets.sum((-2, -1)) + smooth
        )
        dice_loss = (dice_loss * w).mean()
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction="none")
        BCE_per_channel = BCE.mean(dim=(-2, -1))
        BCE = (BCE_per_channel * w).mean()
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
