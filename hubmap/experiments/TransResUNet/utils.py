import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import scienceplots as _
from checkpoints import CHECKPOINT_DIR
from hubmap.metrics import Acc


def train(model, loader, optimizer, criterion, device, acc):
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

        probs = F.sigmoid(predictions)
        classes = torch.argmax(probs, dim=1, keepdims=True)
        classes_per_channel = torch.zeros_like(predictions)
        classes_per_channel.scatter_(1, classes, 1)
        accuracy = acc(classes_per_channel, targets).mean()

        training_losses.append(loss.item())
        training_accuracies.append(accuracy.item())

    return training_losses, training_accuracies


def validate(model, loader, criterion, device, acc):
    validation_losses = []
    validation_accuracies = []

    model.eval()
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            predictions = model(images)
            loss = criterion(predictions, targets)

            probs = F.sigmoid(predictions)
            classes = torch.argmax(probs, dim=1, keepdims=True)
            classes_per_channel = torch.zeros_like(predictions)
            classes_per_channel.scatter_(1, classes, 1)

            accuracy = acc(classes_per_channel, targets).mean()

        validation_losses.append(loss.item())
        validation_accuracies.append(accuracy.item())

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
    acc = Acc()

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
            model, train_loader, optimizer, criterion, device, acc
        )
        val_losses, val_accs = validate(model, val_loader, criterion, device, acc)

        training_loss_history.append(train_losses)
        training_acc_history.append(train_accs)

        validation_loss_history.append(val_losses)
        validation_acc_history.append(val_accs)

        log = f"Epoch {epoch}/{num_epochs} - Summary: "
        log += f"Train Loss: {np.mean(train_losses):.4f} - "
        log += f"Val Loss: {np.mean(val_losses):.4f}"
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



