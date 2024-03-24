from typing import Any
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

BLOOD_VESSEL_CLASS_INDEX = 0


class Precision:
    @property
    def name(self):
        return self._name
    
    def __init__(self, name="Precision"):
        self._name = name
    
    def __call__(self, prediction, target):
        intersection = (target * prediction).sum((-2, -1))
        result = (intersection + 1e-15) / (prediction.sum((-2, -1)) + 1e-15)
        return result[:, BLOOD_VESSEL_CLASS_INDEX]

# def precision(y_true, y_pred):
#     intersection = (y_true * y_pred).sum()
#     return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

class Recall:
    @property
    def name(self):
        return self._name
    
    def __init__(self, name="Recall"):
        self._name = name
    
    def __call__(self, prediction, target):
        intersection = (target * prediction).sum((-2, -1))
        result = (intersection + 1e-15) / (target.sum((-2, -1)) + 1e-15)
        return result[:, BLOOD_VESSEL_CLASS_INDEX]

# def recall(y_true, y_pred):
#     intersection = (y_true * y_pred).sum()
#     return (intersection + 1e-15) / (y_true.sum() + 1e-15)

class F2:
    @property
    def name(self):
        return self._name
    
    def __init__(self, name="F2", beta=2):
        self._name = name
        self._beta = beta
    
    def __call__(self, prediction, target):
        p = Precision()(prediction, target)
        r = Recall()(prediction, target)
        return (1+self._beta**2.) *(p*r) / float(self._beta**2*p + r + 1e-15)

class F1:
    @property
    def name(self):
        return self._name
    
    def __init__(self, name="F1", beta=2):
        self._name = name
        self._beta = beta
    
    def __call__(self, prediction, target):
        p = Precision()(prediction, target)
        r = Recall()(prediction, target)
        return (2 * p * r) / float((p + r))


# def F2(y_true, y_pred, beta=2):
#     p = precision(y_true,y_pred)
#     r = recall(y_true, y_pred)
#     return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

class DiceScore:
    @property
    def name(self):
        return self._name
    
    def __init__(self, name="DiceScore"):
        self._name = name
    
    def __call__(self, prediction, target):
        result = (2 * (target * prediction).sum((-2, -1)) + 1e-15) / (target.sum((-2, -1)) + prediction.sum((-2, -1)) + 1e-15)
        return result[:, BLOOD_VESSEL_CLASS_INDEX]

# def dice_score(y_true, y_pred):
#     return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

class Jac:
    
    @property
    def name(self):
        return self._name
    
    def __init__(self, name="Jac"):
        self._name = name
    
    def __call__(self, prediction, target):
        # Isolate the specific class if multi-class segmentation
        prediction_class = prediction[:, BLOOD_VESSEL_CLASS_INDEX]
        target_class = target[:, BLOOD_VESSEL_CLASS_INDEX]
        
        intersection = (target_class * prediction_class).sum((-2, -1))
        union = target_class.sum((-2, -1)) + prediction_class.sum((-2, -1)) - intersection
        jac = (intersection + 1e-15) / (union + 1e-15)
        return jac


class Acc:
    
    @property
    def name(self):
        return self._name
    
    def __init__(self, name="Acc"):
        self._name = name
        
    def __call__(self, prediction, target):
        prediction_bv_mask = prediction[:, BLOOD_VESSEL_CLASS_INDEX, :, :]
        target_bv_mask = target[:, BLOOD_VESSEL_CLASS_INDEX, :, :]
        
        correct = (prediction_bv_mask == target_bv_mask).sum((-2, -1))
        total = target.size(-2) * target.size(-1)
        accuracy = correct / total
        return accuracy
    
    
class Confidence:
    @property
    def name(self):
        return self._name
    
    def __init__(self, name="Confidence"):
        self._name = name
        
    def __call__(self, probs):
        # Get the maximum probability over all classes
        max_probs, _ = probs.max(dim=1)
        # Select the blood vessel probabilities
        blood_vessel_probs = max_probs[:, BLOOD_VESSEL_CLASS_INDEX]
        # Calculate the mean probability over all pixels
        mean_prob = blood_vessel_probs.mean()
        return mean_prob

def calculate_statistics(model, device, val_set, val_loader, metrics):
    best_model_results = torch.zeros(len(val_set), len(metrics))
    model.eval()
    with torch.no_grad():
        for i, (image, mask) in enumerate(val_loader):
            image = image.to(device)
            mask = mask.to(device)
            
            prediction = model(pixel_values=image)
            upsampled_logits = nn.functional.interpolate(prediction.logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)


            probs = F.sigmoid(upsampled_logits)

            classes = torch.argmax(probs, dim=1, keepdims=True)
            classes_per_channel = torch.zeros_like(probs)
            classes_per_channel.scatter_(1, classes, 1)

            
            for j, metric in enumerate(metrics):
                if isinstance(metric, Confidence):
                    best_model_results[i, j] = metric(probs)
                else:
                    best_model_results[i, j] = metric(classes_per_channel, mask)
    return best_model_results


def print_statistics(results, metrics, title):
    mean_results = results.mean(dim=0).numpy()
    var_results = results.var(dim=0).numpy()
    std_results = results.std(dim=0).numpy()

    metric_names = [metric.name for metric in metrics]
    print("-----------------------------------")
    print(title)

    for i, metric_name in enumerate(metric_names):
        print(f"\t{metric_name}: {mean_results[i]:.4f}% | "
            f"± {std_results[i]:.4f} (std) | "
            f"± {var_results[i]:.4f} (var)"
            )
    print("-----------------------------------")



# val_transforms = T.Compose(
#     [
#         T.ToTensor(),
#         T.Resize((256, 256)),
#     ]
# )
# val_set = ValDataset(DATA_DIR, transform=val_transforms, with_background=True)
# val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=16)

# models = []
# for file in Path(CONFIG_DIR, "TransResUNet").glob('*'):
#     model_name = file.stem
#     print(f"Loading model: {model_name}")
    
#     data = torch.load(file)
#     checkpoint_path = Path(CHECKPOINT_DIR, data["checkpoint_name"])
    
#     model_checkpoint = torch.load(checkpoint_path)
#     backbone = data["backbone"]
#     pretrained = data["pretrained"]
    
#     model = TResUnet(num_classes=4, backbone=backbone, pretrained=pretrained)
#     model.load_state_dict(model_checkpoint["model_state_dict"])
#     models.append((model_name, model))


# device = "cuda" if torch.cuda.is_available() else "cpu"

# # metrics = [Acc(name="Accuracy"), Jac("Jaccard", class_index=0)]
# metrics = [Precision(), Recall(), F2(), DiceScore(), Jac(), Acc(), Confidence()]

# results = torch.zeros(len(models), len(val_set), len(metrics))

# for i, (name, model) in enumerate(models):
#     print(f"Evaluating model: {name}")
#     model = model.to(device)
#     model.eval()
    
#     torch.set_grad_enabled(False)
    
#     for j, (image, mask) in enumerate(val_loader):
#         image = image.to(device)
#         mask = mask.to(device)
        
#         prediction = model(image)
    
#         probs = F.sigmoid(prediction)
#         # probs = F.softmax(prediction, dim=1)
#         classes = torch.argmax(probs, dim=1, keepdims=True)
#         classes_per_channel = torch.zeros_like(prediction)
#         classes_per_channel.scatter_(1, classes, 1)
        
#         for l, metric in enumerate(metrics):
#             if isinstance(metric, Confidence):
#                 results[i, j, l] = metric(probs)
#             else:
#                 results[i, j, l] = metric(classes_per_channel, mask)


# mean_results = results.mean(dim=1).numpy()
# var_results = results.var(dim=1).numpy()
# std_results = results.std(dim=1).numpy()

# metric_names = [metric.name for metric in metrics]
# print("-----------------------------------")
# for i, (name, _) in enumerate(models):
#     print(name)
#     for j, metric_name in enumerate(metric_names):
#         print(f"\t{metric_name}: {mean_results[i, j]*100:.2f}% | ± {std_results[i, j]*100:.2f} (std) | ± {var_results[i, j]*100:.2f} (var)")
#     print("-----------------------------------")


"""
-----------------------------------
pretrained_resnext101_32x8d_trial_0
        Precision: 68.75% | ± 21.75 (std) | ± 4.73 (var)
        Recall: 54.30% | ± 21.24 (std) | ± 4.51 (var)
        F2: 54.82% | ± 21.00 (std) | ± 4.41 (var)
        DiceScore: 57.46% | ± 20.95 (std) | ± 4.39 (var)
        Jac: 43.07% | ± 19.15 (std) | ± 3.67 (var)
        Acc: 54.30% | ± 21.24 (std) | ± 4.51 (var)
-----------------------------------
pretrained_resnet50_trial_0
        Precision: 66.13% | ± 22.28 (std) | ± 4.97 (var)
        Recall: 54.26% | ± 20.35 (std) | ± 4.14 (var)
        F2: 54.49% | ± 20.11 (std) | ± 4.04 (var)
        DiceScore: 56.66% | ± 20.20 (std) | ± 4.08 (var)
        Jac: 42.03% | ± 18.11 (std) | ± 3.28 (var)
        Acc: 54.26% | ± 20.35 (std) | ± 4.14 (var)
-----------------------------------
pretrained_resnet101_trial_0
        Precision: 68.85% | ± 21.95 (std) | ± 4.82 (var)
        Recall: 53.45% | ± 20.35 (std) | ± 4.14 (var)
        F2: 54.28% | ± 19.99 (std) | ± 3.99 (var)
        DiceScore: 57.16% | ± 19.97 (std) | ± 3.99 (var)
        Jac: 42.47% | ± 17.86 (std) | ± 3.19 (var)
        Acc: 53.45% | ± 20.35 (std) | ± 4.14 (var)
-----------------------------------
pretrained_resnext50_32x4d_trial_0
        Precision: 69.22% | ± 20.60 (std) | ± 4.25 (var)
        Recall: 54.57% | ± 20.20 (std) | ± 4.08 (var)
        F2: 54.94% | ± 19.47 (std) | ± 3.79 (var)
        DiceScore: 57.51% | ± 19.32 (std) | ± 3.73 (var)
        Jac: 42.67% | ± 17.41 (std) | ± 3.03 (var)
        Acc: 54.57% | ± 20.20 (std) | ± 4.08 (var)
-----------------------------------
pretrained_wide_resnet50_2_trial_0
        Precision: 67.91% | ± 21.84 (std) | ± 4.77 (var)
        Recall: 55.44% | ± 20.02 (std) | ± 4.01 (var)
        F2: 55.95% | ± 19.73 (std) | ± 3.89 (var)
        DiceScore: 58.20% | ± 20.02 (std) | ± 4.01 (var)
        Jac: 43.56% | ± 18.21 (std) | ± 3.32 (var)
        Acc: 55.44% | ± 20.02 (std) | ± 4.01 (var)
-----------------------------------
pretrained_wide_resnet101_2_trial_0
        Precision: 69.96% | ± 20.53 (std) | ± 4.22 (var)
        Recall: 55.25% | ± 20.24 (std) | ± 4.10 (var)
        F2: 55.83% | ± 19.75 (std) | ± 3.90 (var)
        DiceScore: 58.48% | ± 19.52 (std) | ± 3.81 (var)
        Jac: 43.71% | ± 17.69 (std) | ± 3.13 (var)
        Acc: 55.25% | ± 20.24 (std) | ± 4.10 (var)
-----------------------------------
pretrained_resnet152_trial_0
        Precision: 68.12% | ± 21.27 (std) | ± 4.53 (var)
        Recall: 54.78% | ± 20.76 (std) | ± 4.31 (var)
        F2: 55.34% | ± 20.06 (std) | ± 4.02 (var)
        DiceScore: 57.69% | ± 19.71 (std) | ± 3.89 (var)
        Jac: 42.95% | ± 17.80 (std) | ± 3.17 (var)
        Acc: 54.78% | ± 20.76 (std) | ± 4.31 (var)
-----------------------------------
"""