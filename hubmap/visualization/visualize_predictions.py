import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from hubmap.dataset import label2id, label2title
from hubmap.visualization.visualize_mask import mask_to_rgb, mask_to_rgba
from hubmap.losses import DiceLoss, DiceBCELoss, ChannelWeightedDiceBCELoss


def visualize_detailed_results(model, image, target, device, checkpoint_name):
    plt.style.use(["science"])

    image = image.to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(image.unsqueeze(0))

    probs = F.sigmoid(prediction)
    classes = torch.argmax(probs, dim=1, keepdims=True)
    classes_per_channel = torch.zeros_like(prediction)
    classes_per_channel.scatter_(1, classes, 1)
    classes_per_channel = classes_per_channel.squeeze(0)
    classes = classes.squeeze(0).cpu()

    image = image.cpu()
    classes_per_channel = classes_per_channel.cpu()

    colors = {
        "blood_vessel": "tomato",
        "glomerulus": "dodgerblue",
        "unsure": "palegreen",
        "background": "black",
    }
    colors = colors
    cmap = {label2id[l]: colors[l] for l in colors.keys()}

    image_np = image.permute(1, 2, 0).squeeze().numpy()

    target_mask_rgb = mask_to_rgb(target, color_map=cmap, bg_channel=-1)
    pred_mask_rgb = mask_to_rgb(classes_per_channel, color_map=cmap, bg_channel=-1)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6, 2.5))
    axs[0].imshow(image_np)
    axs[0].set_title(f"Image")
    axs[1].imshow(target_mask_rgb.permute(1, 2, 0))
    axs[1].set_title(f"Ground Truth")
    axs[2].imshow(pred_mask_rgb.permute(1, 2, 0))
    axs[2].set_title(f"Prediction")

    blood_vessel_patch = mpatches.Patch(
        facecolor=colors["blood_vessel"],
        label=f"{label2title['blood_vessel']}",
        edgecolor="black",
    )
    glomerulus_patch = mpatches.Patch(
        facecolor=colors["glomerulus"],
        label=f"{label2title['glomerulus']}",
        edgecolor="black",
    )
    unsure_patch = mpatches.Patch(
        facecolor=colors["unsure"],
        label=f"{label2title['unsure']}",
        edgecolor="black",
    )
    background_patch = mpatches.Patch(
        facecolor=colors["background"],
        label=f"{label2title['background']}",
        edgecolor="black",
    )
    handles = [blood_vessel_patch, glomerulus_patch, unsure_patch, background_patch]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=4)

    fig.suptitle(f"{checkpoint_name}")
    fig.tight_layout()
    return fig


def visualize_detailed_results_overlay(model, image, target, device, checkpoint_name):
    plt.style.use(["science"])

    image = image.to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(image.unsqueeze(0))

    probs = F.sigmoid(prediction)
    classes = torch.argmax(probs, dim=1, keepdims=True)
    classes_per_channel = torch.zeros_like(prediction)
    classes_per_channel.scatter_(1, classes, 1)
    classes_per_channel = classes_per_channel.squeeze(0)
    classes = classes.squeeze(0).cpu()

    image = image.cpu()
    classes_per_channel = classes_per_channel.cpu()

    colors = {
        "blood_vessel": "tomato",
        "glomerulus": "dodgerblue",
        "unsure": "palegreen",
        "background": "black",
    }
    colors = colors
    cmap = {label2id[l]: colors[l] for l in colors.keys()}

    image_np = image.permute(1, 2, 0).squeeze().numpy()

    target_mask_rgba = mask_to_rgba(target, color_map=cmap, bg_channel=3, alpha=1.0)
    pred_mask_rgba = mask_to_rgba(
        classes_per_channel, color_map=cmap, bg_channel=3, alpha=1.0
    )

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4.5, 2.5))
    axs[0].imshow(image_np)
    axs[0].imshow(target_mask_rgba.permute(1, 2, 0))
    axs[0].set_title(f"Ground Truth")
    axs[1].imshow(image_np)
    axs[1].imshow(pred_mask_rgba.permute(1, 2, 0))
    axs[1].set_title(f"Prediction")

    blood_vessel_patch = mpatches.Patch(
        facecolor=colors["blood_vessel"],
        label=f"{label2title['blood_vessel']}",
        edgecolor="black",
    )
    glomerulus_patch = mpatches.Patch(
        facecolor=colors["glomerulus"],
        label=f"{label2title['glomerulus']}",
        edgecolor="black",
    )
    unsure_patch = mpatches.Patch(
        facecolor=colors["unsure"],
        label=f"{label2title['unsure']}",
        edgecolor="black",
    )
    background_patch = mpatches.Patch(
        facecolor=colors["background"],
        label=f"{label2title['background']}",
        edgecolor="black",
    )
    handles = [blood_vessel_patch, glomerulus_patch, unsure_patch, background_patch]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=4)

    fig.suptitle(f"{checkpoint_name}")
    fig.tight_layout()
    return fig
