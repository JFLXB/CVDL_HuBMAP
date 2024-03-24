import os
import argparse
import time
from pathlib import Path
from configs import CONFIG_DIR
from figures import FIGURES_DIR

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from hubmap.data import DATA_DIR
from hubmap.dataset import transforms as T
from hubmap.dataset import TrainDataset, ValDataset

from hubmap.experiments.DUCKNet.utils import run
from hubmap.experiments.DUCKNet.utils import DiceLoss, DiceBCELoss, ChannelWeightedDiceBCELoss

from hubmap.training import LRScheduler
from hubmap.training import EarlyStopping

from hubmap.visualization import visualize_result

from hubmap.models.ducknet import DUCKNet, DUCKNetPretrained, DUCKNetPretrained34


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--epochs", type=int, required=True)
# parser.add_argument("--backbone", type=str, required=True) backbone not required
parser.add_argument("--img-size", type=int, required=True)
# parser.add_argument("--pretrained", action='store_true')
parser.add_argument("--model", type=str, required=True)

# Optional
# LRScheduler
parser.add_argument("--use-lr-scheduler", action='store_true')
parser.add_argument("--lrs-patience", type=int, required=False, default=None)
# EarlyStopping
parser.add_argument("--use-early-stopping", action='store_true')
parser.add_argument("--es-patience", type=int, required=False, default=None)
# Loss
parser.add_argument("--loss", type=str, required=False, default="DiceBCELoss")
parser.add_argument("--weights", nargs="+", type=int, required=False, default=None)
# Continue Training
parser.add_argument("--continue-training", action='store_true')
parser.add_argument("--from-checkpoint", type=str, required=False, default=None)
args = parser.parse_args()

print(args)

#Model, batch size, img size
MODEL = None
BATCH_SIZE = None
IMG_SIZE = None
STARTING_FILTERS = None
if args.img_size < 128:
    BATCH_SIZE = 16
elif args.img_size < 256:
    BATCH_SIZE = 8
else:
    BATCH_SIZE = 4

if args.model == "DUCKNet":
    MODEL = DUCKNet
    IMG_SIZE = args.img_size
    PRETRAINED = False
    STARTING_FILTERS = 32
elif args.model == "DUCKNetPretrained":
    MODEL = DUCKNetPretrained
    IMG_SIZE = args.img_size
    PRETRAINED = True
elif args.model == "DUCKNetPretrained34":
    MODEL = DUCKNetPretrained34
    IMG_SIZE = args.img_size
    PRETRAINED = True
else:
    raise ValueError(f"Unknown model {args.model}")

#Loss and weights if necessary
LOSS = None
WEIGHT = None
if args.loss == "DiceBCELoss":
    LOSS = DiceBCELoss
elif args.loss == "DiceLoss":
    LOSS = DiceLoss
elif args.loss == "ChannelWeightedDiceBCELoss":
    LOSS = ChannelWeightedDiceBCELoss
    WEIGHT = torch.tensor([1.2753886168323578, 1.2737381309347808, 1.3285854321630461, 0.12228782006981492])
elif args.loss == "CrossEntropyLoss":
    LOSS = nn.CrossEntropyLoss
# elif args.loss == "FocalLoss":
#     LOSS = FocalLoss
#     WEIGHT = torch.tensor(args.weights)
# elif args.loss == "IoULoss":
#     LOSS = IoULoss

#learning rate scheduler
LR_SCHEDULER = None
LRS_PATIENCE = None
if args.use_lr_scheduler:
    LRS_PATIENCE = args.lrs_patience
    LR_SCHEDULER = LRScheduler

EARLY_STOPPING = None
ES_PATIENCE = None
if args.use_early_stopping:
    ES_PATIENCE = args.es_patience
    EARLY_STOPPING = EarlyStopping

CHECKPOINT = args.name
NUM_EPOCHS = args.epochs
# BACKBONE = args.backbone 
# PRETRAINED = args.pretrained

LR = 1e-4
CONTINUE_TRAINING = args.continue_training
FROM_CHECKPOINT = args.from_checkpoint

FIGURES_CHECKPOINT_PATH = Path(FIGURES_DIR, "DUCKNet", f"{CHECKPOINT}")
os.makedirs(FIGURES_CHECKPOINT_PATH, exist_ok=True)

CHECKPOINT_FILE_NAME = f"{CHECKPOINT}.pt"
CHECKPOINT_NAME = Path("DUCKNet", CHECKPOINT_FILE_NAME)
config = {
    "model": args.model,
    "image_size": IMG_SIZE,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "checkpoint_name": CHECKPOINT_NAME,
    "use_lr_scheduler": LR_SCHEDULER is not None,
    "lr_scheduler_patience": LRS_PATIENCE,
    "use_early_stopping": EARLY_STOPPING is not None,
    "early_stopping_patience": ES_PATIENCE,
    "loss": args.loss,
    "lr": LR,
    # "backbone": BACKBONE,
    "starting_filters": STARTING_FILTERS,
    "pretrained": PRETRAINED,
    "figures_directory": FIGURES_CHECKPOINT_PATH,
    "weight": WEIGHT
}
os.makedirs(Path(CONFIG_DIR / CHECKPOINT_NAME).parent.resolve(), exist_ok=True)
torch.save(config, Path(CONFIG_DIR / CHECKPOINT_NAME))

train_transforms = T.Compose(
    [
        T.ToTensor(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomCrop((IMG_SIZE, IMG_SIZE)),
    ]
)

val_transforms = T.Compose(
    [
        T.ToTensor(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
    ]
)

train_set = TrainDataset(DATA_DIR, transform=train_transforms, with_background=True)
val_set = ValDataset(DATA_DIR, transform=val_transforms, with_background=True)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=False
)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.model == "DUCKNet":
    model = DUCKNet(input_channels=3, out_classes=4, starting_filters=STARTING_FILTERS).to(device)
elif args.model == "DUCKNetPretrained":
    model = DUCKNetPretrained(input_channels=3, out_classes=4).to(device)
elif args.model == "DUCKNetPretrained34":
    model = DUCKNetPretrained34(input_channels=3, out_classes=4).to(device)
else:
    raise ValueError(f"Unknown model {args.model}")


optimizer = optim.Adam(model.parameters(), lr=LR)

if args.weights:
    criterion = LOSS(weights=WEIGHT)
else:
    criterion = LOSS()
lr_scheduler = LR_SCHEDULER(optimizer, patience=LRS_PATIENCE) if LR_SCHEDULER is not None else None
early_stopping = EARLY_STOPPING(patience=ES_PATIENCE) if EARLY_STOPPING is not None else None

start = time.time()
result = run(
    num_epochs=NUM_EPOCHS,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    early_stopping=early_stopping,
    lr_scheduler=lr_scheduler,
    checkpoint_name=CHECKPOINT_NAME,
    continue_training=CONTINUE_TRAINING,
    from_checkpoint=FROM_CHECKPOINT
)
total = time.time() - start

print(f"TRAINING TOOK A TOTAL OF '{total}' FOR '{NUM_EPOCHS}' EPOCHS => PER EPOCH TIME: '{total / NUM_EPOCHS}'")

loss_fig, benchmark_fig = visualize_result(result)
loss_fig.savefig(Path(FIGURES_CHECKPOINT_PATH, "paper_results_loss.svg"))
benchmark_fig.savefig(Path(FIGURES_CHECKPOINT_PATH, "paper_results_accuracy.svg"))
