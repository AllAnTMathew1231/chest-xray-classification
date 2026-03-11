"""
Utility functions — checkpointing, Grad-CAM, plotting.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# ── Checkpointing ──────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model, optimizer=None):
    checkpoint  = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    epoch    = checkpoint.get("epoch", 0)
    best_auc = checkpoint.get("best_auc", 0.0)
    return epoch, best_auc


# ── Training Curves ────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str = "results/training_curves.png"):
    """Plot and save loss + AUC curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#0891B2")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#DC2626")
    axes[0].set_title("Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_auc"], label="Val Mean AUC", color="#059669", linewidth=2)
    axes[1].set_title("Validation AUC-ROC", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean AUC")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved training curves → {save_path}")


# ── Grad-CAM ───────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for EfficientNetV2.

    Usage:
        cam   = GradCAM(model, target_layer=model.features[-1])
        heatmap = cam(image_tensor, class_idx=6)  # 6 = Pneumonia
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        x = x.unsqueeze(0).requires_grad_(True)

        logits = self.model(x)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = F.relu(cam)
        cam     = cam.cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def visualize_gradcam(model, image_path: str, class_idx: int, class_name: str,
                      save_path: str = "results/gradcam.png", img_size: int = 224):
    """Overlay Grad-CAM heatmap on original image and save."""
    # Preprocess
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    orig  = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    inp   = preprocess(orig)

    # Grad-CAM
    cam_extractor = GradCAM(model, target_layer=model.features[-1])
    heatmap       = cam_extractor(inp, class_idx)

    # Overlay
    heatmap_img = Image.fromarray(np.uint8(plt.cm.jet(heatmap) * 255)).resize((img_size, img_size))
    overlay     = Image.blend(orig, heatmap_img.convert("RGB"), alpha=0.45)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig);           axes[0].set_title("Original X-Ray");      axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
    axes[2].imshow(overlay);        axes[2].set_title(f"Overlay — {class_name}"); axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved Grad-CAM → {save_path}")
