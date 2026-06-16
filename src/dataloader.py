from torch.utils.data import DataLoader

from dataset import OtitisDataset
from transform import (
    train_transform, 
    val_transform
)

train_dataset = OtitisDataset(
    "data/splits/train/train.csv",
    transform=train_transform
)

val_dataset = OtitisDataset(
    "data/splits/val/val.csv",
    transform=val_transform
)

test_dataset = OtitisDataset(
    "data/splits/test/test.csv",
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)