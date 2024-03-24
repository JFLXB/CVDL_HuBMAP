"""
Sources: 
 - https://github.com/Thanos-DB/FullyConvolutionalTransformer
 - https://openaccess.thecvf.com/content/WACV2023/papers/Tragakis_The_Fully_Convolutional_Transformer_for_Medical_Image_Segmentation_WACV_2023_paper.pdf
"""
import torch
import torch.nn as nn


def init_weights(m):
    """
    Initialize the weights
    """
    if isinstance(m, nn.Conv2d):
        # Changed the initialization method to kaiming_normal_.
        # Accoding to the changes suggested by the PyTorch Framework.
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
