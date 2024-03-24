import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from checkpoints import CHECKPOINT_DIR
from configs import CONFIG_DIR
from hubmap.data import DATA_DIR
from hubmap.dataset import transforms as T
from hubmap.dataset import ValDataset, TestDataset
from hubmap.models.trans_res_u_net.model import TResUnet, TResUnet512
from hubmap.metrics import Precision, Recall, F2, Jac, Acc
from hubmap.metrics import calculate_statistics, print_statistics


if __name__ == "__main__":
    for file in Path(CONFIG_DIR, "TransResUNet").glob('*'):
        print(f"Loading model in file: {file.stem}")

        data = torch.load(file)
        model_name = data["model"]
        
        backbone = data["backbone"]
        pretrained = data["pretrained"]
        image_size = data["image_size"]
        
        print("model: ", data["model"])
        print("backbone: ", data["backbone"])
        print("pretrained: ",data["pretrained"])
        
        if model_name == "TransResUNet":
            model = TResUnet(num_classes=4, backbone=backbone, pretrained=pretrained)
        elif model_name == "TransResUNet512":
            model = TResUnet512(num_classes=4, backbone=backbone, pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        model_ckpt = torch.load(Path(CHECKPOINT_DIR / data["checkpoint_name"]))
        state_dict = model_ckpt["model_state_dict"]
        model.load_state_dict(state_dict)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model.to(device)
        
        val_transforms = T.Compose([
                T.ToTensor(),
                T.Resize((image_size, image_size)),
        ])
        val_set = ValDataset(DATA_DIR, transform=val_transforms, with_background=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=16)
        
        metrics = [Precision(), Recall(), F2(), Jac(), Acc()]
        
        results = calculate_statistics(model, device, val_set, val_loader, metrics)
        print_statistics(results, metrics, f"MODEL FOR FILE: {file.stem}")
