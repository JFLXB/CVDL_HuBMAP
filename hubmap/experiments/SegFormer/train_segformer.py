import sys
sys.path.append("../dataset/")
from datasets import TrainDataset, ValDataset
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
import torch
import torch.nn as nn
from sklearn.metrics import jaccard_score
import os
import pandas as pd
import transforms as tran
from skimage.color import label2rgb
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import random




def calculate_iou(predicted, labels, mask):
    labels_masked = labels[mask].cpu().numpy()
    predicted_masked = predicted[mask].cpu().numpy()

    intersection = ((predicted_masked == 0) & (labels_masked == 0)).sum()
    union = ((predicted_masked == 0) | (labels_masked == 0)).sum()

    return intersection / union if union != 0 else 0


def train(name, model, lr, num_epochs, batch_size, tran_train, tran_val):
    result_dir = f'results/{name}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    logging.basicConfig(filename=f'{result_dir}/train.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    train = TrainDataset('../../data/', transform=tran_train, with_background=True, as_id_mask=True)
    val = ValDataset('../../data/', transform=tran_val, with_background=True, as_id_mask=True)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device: {device}')
    print(f'Device: {device}')

    assert torch.cuda.is_available()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss = []
    train_accuracy = []
    all_train_accuracy = []
    train_iou = []
    test_loss = []
    test_accuracy = []
    all_test_accuracy = []
    test_iou = []
    logging.info(f'Starting training of Model: {name}')
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        correct_all = 0
        total_all = 0
        total = 0
        iou_score = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = labels.squeeze(1).long().to(device)
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            upsampled_logits = nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            mask = (labels == 0)
            correct += (predicted[mask] == labels[mask]).sum().item()
            correct_all += (predicted == labels).sum().item()
            total_all += torch.numel(labels)

            total += mask.sum().item()

            running_loss += loss.item()
            iou_score += calculate_iou(predicted, labels, mask)

        all_train_accuracy.append(100* correct_all/ total_all)
        train_accuracy.append(100 * correct / total if total != 0 else 0)
        train_loss.append(running_loss / i)
        train_iou.append(iou_score / (i + 1))

        # Testing
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        total_all = 0
        iou_score = 0.0
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            labels = labels.squeeze(1).long().to(device)
            inputs = inputs.to(device)
            outputs = model(pixel_values=inputs, labels=labels)
            loss = outputs.loss

            upsampled_logits = nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            mask = (labels == 0)
            correct += (predicted[mask] == labels[mask]).sum().item()
            correct_all += (predicted == labels).sum().item()
            total_all += torch.numel(labels)
            total += mask.sum().item()

            running_loss += loss.item()
            iou_score += calculate_iou(predicted, labels, mask)

        all_test_accuracy.append(100* correct_all / total_all)
        test_accuracy.append(100 * correct / total if total != 0 else 0)
        test_loss.append(running_loss / i)
        test_iou.append(iou_score / (i + 1))
        str = f"Epoch {epoch+1}, Train Loss: {train_loss[-1]}, Train Acc: {train_accuracy[-1]}, Train IoU: {train_iou[-1]}, Test Loss: {test_loss[-1]}, Test Acc: {test_accuracy[-1]}, Test IoU: {test_iou[-1]}"
        print(str)
        logging.info(str)
        if (epoch + 1) % 4 == 0 or epoch == num_epochs - 1:
            checkpoint_name = f'{result_dir}/model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_name)
            logging.info(f'Saved model checkpoint at epoch {epoch+1} as {checkpoint_name}')

    end_time = time.time()
    with open(f'{result_dir}/train.time', 'w') as file:
        file.write(f'{end_time-start_time}')
    logging.info(f'Finished training of Model: {name}')

    stats = pd.DataFrame({
    'Epoch': range(1, num_epochs+1),
    'Train_Loss': train_loss,
    'Train_Accuracy': train_accuracy,
    'Train_Accuracy_all' : all_train_accuracy,
    'Train_IoU': train_iou,
    'Test_Loss': test_loss,
    'Test_Accuracy': test_accuracy,
    'Test_Accuracy_all' : all_test_accuracy,
    'Test_IoU': test_iou,
    })

    stats.to_csv(os.path.join(result_dir, 'training_stats.csv'), index=False)
    image_dir = f'{result_dir}/sample_images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    random_numbers = random.sample(range(0, len(val)), 10)

    for i, num in enumerate(random_numbers, 0):
        fig = visualize_example(num, val, model)
        fig.savefig(f'{image_dir}/{i}.png')

    

def visualize_example(idx, data, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image, target = data.get(idx, transform=tran.ToTensor(mask_as_integer=True))
    input, mask = data.get(idx, transform=data.transform)
    input = input.unsqueeze(0).to(device)
    mask = mask.to(device).long()

    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=input, labels=mask.long())
        
    upsampled_logits = nn.functional.interpolate(outputs.logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    predicted = upsampled_logits.argmax(dim=1)



    toPILImage = T.ToPILImage()
    imagePIL = toPILImage(image)
    image_np = np.array(imagePIL)

    targetPIL = toPILImage(target.byte())
    target_np = np.array(targetPIL)

    predictedPIL = toPILImage(predicted.byte())
    predicted_np = np.array(predictedPIL)

    image_label_overlay = label2rgb(
        target_np,
        image=image_np,
        bg_label=3,
        colors=["red", "green", "blue"],
        kind="overlay",
        saturation=1.0,
    )


    image_predicted_overlay = label2rgb(
        predicted_np,
        image=image_np,
        bg_label=3,
        colors=["red", "green", "blue"],
        kind="overlay",
        saturation=1.0,
    )

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(image_label_overlay)
    ax[0].set_title('Target Labels')

    ax[1].imshow(image_predicted_overlay)
    ax[1].set_title('Predicted Labels')

    plt.show()
    return fig


