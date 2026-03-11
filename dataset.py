"""
Dataset class for NIH ChestX-ray14.

Download: https://nihcc.app.box.com/v/ChestXray-NIHCC
Official split file: https://nihcc.app.box.com/v/ChestXray-NIHCC  (train_val_list.txt / test_list.txt)

Expected directory layout:
    data/
    ├── images/           # all 112,120 PNG images
    ├── Data_Entry_2017.csv
    ├── train_val_list.txt
    └── test_list.txt
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]


def get_transforms(split: str, img_size: int = 224):
    """Return augmentation pipeline for train / val / test splits."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


class ChestXrayDataset(Dataset):
    """
    Multi-label dataset for NIH ChestX-ray14.

    Args:
        root      (str): Path to data/ folder.
        split     (str): One of 'train', 'val', 'test'.
        img_size  (int): Image will be resized to (img_size × img_size).
        val_frac  (float): Fraction of train_val list to use as validation.
    """

    def __init__(self, root: str, split: str = "train",
                 img_size: int = 224, val_frac: float = 0.1):
        super().__init__()
        self.root      = root
        self.split     = split
        self.transform = get_transforms(split, img_size)

        # Load metadata
        csv_path = os.path.join(root, "Data_Entry_2017.csv")
        df = pd.read_csv(csv_path)

        # Build label matrix
        for cls in CLASSES:
            df[cls] = df["Finding Labels"].apply(lambda x: int(cls in x))

        # Split file filtering
        if split in ("train", "val"):
            list_file = os.path.join(root, "train_val_list.txt")
            with open(list_file) as f:
                file_list = [l.strip() for l in f.readlines()]
            df = df[df["Image Index"].isin(file_list)].reset_index(drop=True)

            # 90/10 train-val split (deterministic)
            np.random.seed(42)
            idx = np.random.permutation(len(df))
            cut = int(len(df) * (1 - val_frac))
            df = df.iloc[idx[:cut]] if split == "train" else df.iloc[idx[cut:]]

        else:  # test
            list_file = os.path.join(root, "test_list.txt")
            with open(list_file) as f:
                file_list = [l.strip() for l in f.readlines()]
            df = df[df["Image Index"].isin(file_list)].reset_index(drop=True)

        self.df     = df.reset_index(drop=True)
        self.labels = df[CLASSES].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "Image Index"]
        img_path = os.path.join(self.root, "images", img_name)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label
