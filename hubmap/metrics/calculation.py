import torch
import torch.nn.functional as F
from hubmap.metrics.confidence import Confidence

BLOOD_VESSEL_CLASS_INDEX = 0


def calculate_statistics(model, device, val_set, val_loader, metrics):
    best_model_results = torch.zeros(len(val_set), len(metrics))
    model.eval()
    with torch.no_grad():
        for i, (image, mask) in enumerate(val_loader):
            image = image.to(device)
            mask = mask.to(device)
            
            prediction = model(image)
        
            probs = F.sigmoid(prediction)
            classes = torch.argmax(probs, dim=1, keepdims=True)
            classes_per_channel = torch.zeros_like(prediction)
            classes_per_channel.scatter_(1, classes, 1)
            
            for j, metric in enumerate(metrics):
                if isinstance(metric, Confidence):
                    best_model_results[i, j] = metric(probs)
                else:
                    best_model_results[i, j] = metric(classes_per_channel, mask)
    return best_model_results


def print_statistics(results, metrics, title):
    mean_results = results.mean(dim=0).numpy()

    assert len(mean_results) == len(metrics)

    metric_names = [metric.name for metric in metrics]
    print("-----------------------------------")
    print(title)
    for i, metric_name in enumerate(metric_names):
        print(f"\t{metric_name}: {mean_results[i]:.4f}")
    print("-----------------------------------")