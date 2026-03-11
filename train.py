"""
Chest X-Ray Multi-Label Disease Classification
Using EfficientNetV2-S with Transfer Learning

Dataset : NIH ChestX-ray14
Model   : EfficientNetV2-S (pretrained on ImageNet)
Task    : Multi-label classification (14 disease classes)
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import ChestXrayDataset
from model import ChestXrayModel
from utils import save_checkpoint, load_checkpoint, plot_training_curves

# ── Disease labels ─────────────────────────────────────────────────────────────
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Chest X-Ray Classifier")
    parser.add_argument("--data_dir",    type=str, default="data/",          help="Path to dataset root")
    parser.add_argument("--epochs",      type=int, default=30,               help="Number of training epochs")
    parser.add_argument("--batch_size",  type=int, default=32,               help="Batch size")
    parser.add_argument("--lr",          type=float, default=1e-4,           help="Learning rate")
    parser.add_argument("--img_size",    type=int, default=224,              help="Input image size")
    parser.add_argument("--num_workers", type=int, default=4,                help="DataLoader workers")
    parser.add_argument("--resume",      type=str, default=None,             help="Resume from checkpoint")
    parser.add_argument("--output_dir",  type=str, default="results/",       help="Output directory")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        all_preds.append(outputs.sigmoid().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Per-class AUC
    aucs = []
    for i, cls in enumerate(CLASSES):
        if all_labels[:, i].sum() > 0:
            aucs.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))

    mean_auc = np.mean(aucs)
    val_loss = running_loss / len(loader.dataset)
    return val_loss, mean_auc, aucs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Chest X-Ray Classification — EfficientNetV2-S")
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs}  |  Batch : {args.batch_size}  |  LR : {args.lr}")
    print(f"{'='*55}\n")

    # ── Datasets & Loaders ─────────────────────────────────────────────────────
    train_ds = ChestXrayDataset(args.data_dir, split="train", img_size=args.img_size)
    val_ds   = ChestXrayDataset(args.data_dir, split="val",   img_size=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"  Train samples : {len(train_ds):,}")
    print(f"  Val   samples : {len(val_ds):,}\n")

    # ── Model, Loss, Optimizer ─────────────────────────────────────────────────
    model     = ChestXrayModel(num_classes=len(CLASSES), pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    best_auc    = 0.0
    history     = {"train_loss": [], "val_loss": [], "val_auc": []}

    if args.resume:
        start_epoch, best_auc = load_checkpoint(args.resume, model, optimizer)
        print(f"  Resumed from epoch {start_epoch}, best AUC = {best_auc:.4f}\n")

    # ── Training Loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch [{epoch+1:02d}/{args.epochs}]")
        train_loss           = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, mean_auc, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(mean_auc)

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  mean_AUC={mean_auc:.4f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_auc": best_auc,
            }, filepath=os.path.join(args.output_dir, "best_model.pth"))
            print(f"  ✅ Saved best model  (AUC={best_auc:.4f})")

    # ── Final Evaluation ───────────────────────────────────────────────────────
    print("\n── Final Evaluation on Validation Set ──")
    load_checkpoint(os.path.join(args.output_dir, "best_model.pth"), model)
    _, mean_auc, per_class_aucs = evaluate(model, val_loader, criterion, device)

    print(f"\n  Mean AUC : {mean_auc:.4f}\n")
    for cls, auc in zip(CLASSES, per_class_aucs):
        print(f"  {cls:<22} AUC = {auc:.4f}")

    # Save results
    results_df = pd.DataFrame({"Disease": CLASSES, "AUC": per_class_aucs})
    results_df.to_csv(os.path.join(args.output_dir, "per_class_auc.csv"), index=False)
    plot_training_curves(history, save_path=os.path.join(args.output_dir, "training_curves.png"))
    print(f"\n  Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
