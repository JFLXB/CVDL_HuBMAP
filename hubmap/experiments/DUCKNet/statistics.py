import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from checkpoints import CHECKPOINT_DIR
from configs import CONFIG_DIR
from hubmap.data import DATA_DIR
from hubmap.dataset import transforms as T
from hubmap.dataset import ValDataset, TrainDataset, TestDataset
from hubmap.models import DUCKNet, DUCKNetPretrained, DUCKNetPretrained34
from hubmap.experiments.DUCKNet.utils import train, DiceLoss

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
        intersection = (target * prediction).sum((-2, -1))
        union = target.sum((-2, -1)) + prediction.sum((-2, -1)) - intersection
        jac = (intersection + 1e-15) / (union + 1e-15)
        return jac[:, BLOOD_VESSEL_CLASS_INDEX]
        
        # if self._class_index is not None:
        #     return jac[:, self._class_index]
        # else:
        #     return jac.mean()


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
    var_results = results.var(dim=0).numpy()
    std_results = results.std(dim=0).numpy()

    metric_names = [metric.name for metric in metrics]
    print("-----------------------------------")
    print(title)
    for i, metric_name in enumerate(metric_names):
        print(f"\t{metric_name}: {mean_results[i]:.4f}")
    print("-----------------------------------")


if __name__ == "__main__":
    for file in Path(CONFIG_DIR, "DUCKNet_final_test").glob('*'):
        print(f"Loading model in file: {file.stem}")

        data = torch.load(file)
        model_name = data["model"]
        # backbone = data["backbone"]
        # pretrained = data["pretrained"]
        image_size = data["image_size"]
        

        
        if model_name == "DUCKNet":
            model = DUCKNet(input_channels=3, out_classes=4, starting_filters=32)
        elif model_name == "DUCKNetPretrained":
            model = DUCKNetPretrained(input_channels=3, out_classes=4)
        elif model_name == "DUCKNetPretrained34":
            model = DUCKNetPretrained34(input_channels=3, out_classes=4)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        #also cover ducknet with resnet34 backbone when model name is not saved
        if 'pretrained34' in str(data['checkpoint_name']):
            model = DUCKNetPretrained34(input_channels=3, out_classes=4)

        print("checkpoint name: ", data["checkpoint_name"])
        model_ckpt = torch.load(Path(CHECKPOINT_DIR / data["checkpoint_name"]))
        state_dict = model_ckpt["model_state_dict"]

        #---------------Addition-----------------#
        #Run one training loop because otherwise the model is missing some keys
        train_transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize((64, 64)),
            ]
        )
        val_transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize((64, 64)),
            ]
        )
        train_set = TrainDataset(DATA_DIR, transform=train_transforms, with_background=True)
        val_set = ValDataset(DATA_DIR, transform=val_transforms, with_background=True)

        train_loader = DataLoader(
            train_set, batch_size=32, shuffle=False
        )
        val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        train_losses, train_accs = train(
            model, train_loader, 
            optimizer=optim.Adam(model.parameters(), lr=0.001), 
            criterion=DiceLoss(), device=device
        )
        #---------------END OF Addition-----------------#


        model.load_state_dict(state_dict)
        
        #model.to(device)
        
        
        val_transforms = T.Compose([
                T.ToTensor(),
                T.Resize((image_size, image_size)),
        ])
        # val_set = ValDataset(DATA_DIR, transform=val_transforms, with_background=True)
        # val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        val_set = TestDataset(DATA_DIR, transform=val_transforms, with_background=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        metrics = [Precision(), Recall(), F1(), F2(), DiceScore(), Jac(), Acc(), Confidence()]
        
        results = calculate_statistics(model, device, val_set, val_loader, metrics)
        print_statistics(results, metrics, f"MODEL FOR FILE: {file.stem}")
