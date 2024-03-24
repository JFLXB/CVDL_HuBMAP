import os
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scienceplots as _
from checkpoints import CHECKPOINT_DIR
from hubmap.data import DATA_DIR
from hubmap.dataset import TrainDataset, ValDataset
from hubmap.dataset import label2id, label2title
from hubmap.metrics.jac import Jac as IoU
from hubmap.training import LRScheduler
from hubmap.training import EarlyStopping
from hubmap.visualization.visualize_mask import mask_to_rgb, mask_to_rgba
from hubmap.visualization import visualize_result


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs = torch.softmax(inputs, dim=1)
        inputs = torch.sigmoid(inputs)

        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, **kwargs):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth
        )
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
    
    
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
    
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = nn.functional.cross_entropy(input, target, reduction='none')  # Shape: (batch_size, H, W)
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha[target.view(-1)].view(target.size())  # Reshape alpha to match target shape
        loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def train(model, loader, optimizer, criterion, device):
    training_losses = []
    training_accuracies = []

    model.train()
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        # probs = F.sigmoid(predictions)
        # probs = F.softmax(predictions, dim=1)
        probs = predictions
        classes = torch.argmax(probs, dim=1, keepdims=True)
        classes_per_channel = torch.zeros_like(predictions)
        classes_per_channel.scatter_(1, classes, 1)

        targets_classes = torch.argmax(targets, dim=1, keepdims=True)
        mask = targets_classes == 0
        correct = (classes[mask] == targets_classes[mask]).sum()
        total = mask.sum()
        accuracy = correct / total

        training_losses.append(loss.item())
        training_accuracies.append(accuracy.item())

    return training_losses, training_accuracies


def validate(model, loader, criterion, device):
    validation_losses = []
    validation_accuracies = []

    model.eval()
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            predictions = model(images)
            loss = criterion(predictions, targets)

            # probs = F.softmax(predictions, dim=1)
            probs = predictions
            classes = torch.argmax(probs, dim=1, keepdims=True)
            classes_per_channel = torch.zeros_like(predictions)
            classes_per_channel.scatter_(1, classes, 1)

            targets_classes = torch.argmax(targets, dim=1, keepdims=True)
            mask = targets_classes == 0
            correct = (classes[mask] == targets_classes[mask]).sum()
            total = mask.sum()
            acc = correct / total

        validation_losses.append(loss.item())
        validation_accuracies.append(acc.item())

    return validation_losses, validation_accuracies


def run(
    num_epochs,
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    early_stopping,
    lr_scheduler,
    checkpoint_name,
    continue_training=False,
    from_checkpoint=None,
):
    start_epoch = 1

    training_loss_history = []
    training_acc_history = []

    validation_loss_history = []
    validation_acc_history = []
    
    current_best_accuracy = 0.0

    if continue_training:
        # Load checkpoint.
        print(f"Loading checkpoint: '{from_checkpoint}'")
        checkpoint = torch.load(Path(CHECKPOINT_DIR / from_checkpoint))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        training_loss_history = checkpoint["training_loss_history"]
        training_acc_history = checkpoint["training_acc_history"]
        validation_loss_history = checkpoint["validation_loss_history"]
        validation_acc_history = checkpoint["validation_acc_history"]

    for epoch in range(start_epoch, num_epochs + 1):
        train_losses, train_accs = train(
            model, train_loader, optimizer, criterion, device
        )
        val_losses, val_accs = validate(model, val_loader, criterion, device)
        training_loss_history.append(train_losses)
        training_acc_history.append(train_accs)

        validation_loss_history.append(val_losses)
        validation_acc_history.append(val_accs)

        log = f"Epoch {epoch}/{num_epochs} - Summary: "
        log += f"Train Loss: {np.mean(train_losses):.4f} - "
        # log += f"Acc: {np.mean(train_accs):.4f} --- "
        log += f"Val Loss: {np.mean(val_losses):.4f} - "
        # log += f"Acc: {np.mean(val_accs):.4f}"
        print(log)


        data_to_save = {
            "early_stopping": False,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_loss_history": training_loss_history,
            "training_acc_history": training_acc_history,
            "validation_loss_history": validation_loss_history,
            "validation_acc_history": validation_acc_history,
        }

        # NOW DO THE ADJUSTMENTS USING THE LEARNING RATE SCHEDULER.
        if lr_scheduler:
            lr_scheduler(np.mean(val_losses))
        # NOW DO THE ADJUSTMENTS USING THE EARLY STOPPING.
        if early_stopping:
            early_stopping(np.mean(val_losses))
            # MODIFY THE DATA TO SAVE ACCORDING TO THE EARLY STOPPING RESULT.
            data_to_save["early_stopping"] = early_stopping.early_stop

        # SAVE THE DATA.
        os.makedirs(
            Path(CHECKPOINT_DIR / checkpoint_name).parent.resolve(), exist_ok=True
        )
        torch.save(data_to_save, Path(CHECKPOINT_DIR / checkpoint_name))
        
        if np.mean(val_accs) >= current_best_accuracy:
            current_best_accuracy = np.mean(val_accs)
            parent = checkpoint_name.parent
            best_checkpoint_name = parent / f"{checkpoint_name.stem}_best{checkpoint_name.suffix}"
            best_path = Path(CHECKPOINT_DIR / best_checkpoint_name)
            torch.save(data_to_save, best_path)

        # DO THE EARLY STOPPING IF NECESSARY.
        if early_stopping and early_stopping.early_stop:
            break

    result = {
        "epoch": epoch,
        "training": {
            "loss": training_loss_history,
            "acc": training_acc_history,
        },
        "validation": {
            "loss": validation_loss_history,
            "acc": validation_acc_history,
        },
    }
    return result


def accuracy(target, prediction, cls_idx):
    target_classes = torch.argmax(target, dim=0, keepdim=True)
    mask = target_classes == cls_idx
    correct = (prediction[mask] == target_classes[mask]).sum().item()
    total = mask.sum().item()
    if total == 0:
        return 1.0
    return correct / total


def visualize_detailed_results(model, image, target, device, checkpoint_name):
    plt.style.use(["science"])

    image = image.to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(image.unsqueeze(0))

    # probs = F.softmax(prediction, dim=1)
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
        # \nIoU: {iou_blood_vessel_score * 100:.2f} / Acc: {acc_bv * 100:.2f}",
        edgecolor="black",
    )
    glomerulus_patch = mpatches.Patch(
        facecolor=colors["glomerulus"],
        label=f"{label2title['glomerulus']}",
        # \nIoU: {iou_glomerulus_score * 100:.2f} / Acc: {acc_gl * 100:.2f}",
        edgecolor="black",
    )
    unsure_patch = mpatches.Patch(
        facecolor=colors["unsure"],
        label=f"{label2title['unsure']}",
        # \nIoU: {iou_unsure_score * 100:.2f} / Acc: {acc_un * 100:.2f}",
        edgecolor="black",
    )
    background_patch = mpatches.Patch(
        facecolor=colors["background"],
        label=f"{label2title['background']}",
        # \nIoU: {iou_background_score * 100:.2f} / Acc: {acc_bg * 100:.2f}",
        edgecolor="black",
    )
    handles = [blood_vessel_patch, glomerulus_patch, unsure_patch, background_patch]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=4)

    fig.suptitle(f"{checkpoint_name}")
                #  / IoU: {(iou_score * 100):.2f}")
    fig.tight_layout()
    return fig


def visualize_detailed_results_overlay(model, image, target, device, checkpoint_name):
    plt.style.use(["science"])

    image = image.to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(image.unsqueeze(0))

    # probs = F.softmax(prediction, dim=1)
    probs = F.sigmoid(prediction)
    classes = torch.argmax(probs, dim=1, keepdims=True)
    classes_per_channel = torch.zeros_like(prediction)
    classes_per_channel.scatter_(1, classes, 1)
    classes_per_channel = classes_per_channel.squeeze(0)
    classes = classes.squeeze(0).cpu()

    # iou = IoU()
    # iou_score = iou(classes_per_channel, target).item()
    # iou_blood_vessel = IoU(class_index=label2id["blood_vessel"])
    # iou_blood_vessel_score = iou_blood_vessel(classes_per_channel, target).item()
    # iou_glomerulus = IoU(class_index=label2id["glomerulus"])
    # iou_glomerulus_score = iou_glomerulus(classes_per_channel, target).item()
    # iou_unsure = IoU(class_index=label2id["unsure"])
    # iou_unsure_score = iou_unsure(classes_per_channel, target).item()
    # iou_background = IoU(class_index=label2id["background"])
    # iou_background_score = iou_background(classes_per_channel, target).item()

    # acc_bv = accuracy(target, classes, 0)
    # acc_gl = accuracy(target, classes, 1)
    # acc_un = accuracy(target, classes, 2)
    # acc_bg = accuracy(target, classes, 3)

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
        # \nIoU: {iou_blood_vessel_score * 100:.2f} / Acc: {acc_bv * 100:.2f}",
        edgecolor="black",
    )
    glomerulus_patch = mpatches.Patch(
        facecolor=colors["glomerulus"],
        label=f"{label2title['glomerulus']}",
        # \nIoU: {iou_glomerulus_score * 100:.2f} / Acc: {acc_gl * 100:.2f}",
        edgecolor="black",
    )
    unsure_patch = mpatches.Patch(
        facecolor=colors["unsure"],
        label=f"{label2title['unsure']}",
        # \nIoU: {iou_unsure_score * 100:.2f} / Acc: {acc_un * 100:.2f}",
        edgecolor="black",
    )
    background_patch = mpatches.Patch(
        facecolor=colors["background"],
        label=f"{label2title['background']}",
        # \nIoU: {iou_background_score * 100:.2f} / Acc: {acc_bg * 100:.2f}",
        edgecolor="black",
    )
    handles = [blood_vessel_patch, glomerulus_patch, unsure_patch, background_patch]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=4)

    fig.suptitle(f"{checkpoint_name}")
    # / IoU: {(iou_score * 100):.2f}")
    fig.tight_layout()
    return fig
