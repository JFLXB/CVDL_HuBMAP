from matplotlib import colors
import torch


def mask_to_rgb(mask: torch.Tensor, color_map: dict, bg_channel: int = -1):
    if mask.size(-1) != mask.size(-2):
        raise ValueError(
            "Mask must be square, and in the last to dimensions. Expected: (B, C, H, W) or (C, H, W)"
        )
    if len(mask.size()) != 3:
        raise ValueError("Expecting a 3D tensor. Got: {}".format(mask.size()))

    channel_dim = 0
    rgb_mask = torch.zeros((3, mask.size(1), mask.size(2)))

    for channel in range(mask.size(channel_dim)):
        if bg_channel == channel:
            continue

        c = colors.to_rgb(color_map[channel])
        rgb_mask = rgb_mask + mask[channel, :, :] * torch.tensor(c).view(3, 1, 1)
    return rgb_mask


def mask_to_rgba(
    mask: torch.Tensor, color_map: dict, bg_channel: int = -1, alpha: float = 0.5
):
    if mask.size(-1) != mask.size(-2):
        raise ValueError(
            "Mask must be square, and in the last to dimensions. Expected: (B, C, H, W) or (C, H, W)"
        )
    if len(mask.size()) != 3:
        raise ValueError(f"Expecting a 3D tensor. Got: {mask.size()}")

    channel_dim = 0
    rgb_mask = torch.zeros((4, mask.size(1), mask.size(2)))

    for channel in range(mask.size(channel_dim)):
        if bg_channel == channel:
            alpha_t = 0.0
        else:
            alpha_t = alpha

        c = colors.to_rgba(color_map[channel], alpha=alpha_t)
        rgb_mask = rgb_mask + mask[channel, :, :] * torch.tensor(c).view(4, 1, 1)
    return rgb_mask
