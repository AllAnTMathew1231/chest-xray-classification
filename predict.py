"""
Run inference on a single X-ray image or a folder of images.

Usage:
    python src/predict.py --image data/images/00000001_000.png \
                          --checkpoint results/best_model.pth

    python src/predict.py --folder data/images/ \
                          --checkpoint results/best_model.pth \
                          --gradcam
"""

import os
import argparse
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

from model import ChestXrayModel
from utils import load_checkpoint, visualize_gradcam

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

THRESHOLD = 0.5  # Prediction threshold (tune per class if needed)


def preprocess(image_path: str, img_size: int = 224) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img)


@torch.no_grad()
def predict(model, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    inp  = image_tensor.unsqueeze(0).to(device)
    probs = torch.sigmoid(model(inp)).squeeze(0).cpu().numpy()
    return probs


def print_predictions(probs: np.ndarray, threshold: float = THRESHOLD):
    print(f"\n{'Disease':<25} {'Probability':>12}  {'Predicted':>10}")
    print("─" * 52)
    for cls, p in zip(CLASSES, probs):
        flag = "✅ YES" if p >= threshold else "   no"
        bar  = "█" * int(p * 20)
        print(f"  {cls:<23} {p:>10.4f}   {flag}  {bar}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      type=str, required=False, help="Path to single image")
    parser.add_argument("--folder",     type=str, required=False, help="Path to folder of images")
    parser.add_argument("--checkpoint", type=str, required=True,  help="Model checkpoint path")
    parser.add_argument("--img_size",   type=int, default=224)
    parser.add_argument("--gradcam",    action="store_true",       help="Generate Grad-CAM for top prediction")
    parser.add_argument("--output_dir", type=str, default="results/gradcam/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ChestXrayModel(num_classes=14, pretrained=False).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    print(f"  Model loaded from {args.checkpoint}")

    images = []
    if args.image:
        images = [args.image]
    elif args.folder:
        images = [os.path.join(args.folder, f) for f in os.listdir(args.folder)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    else:
        parser.error("Provide --image or --folder")

    os.makedirs(args.output_dir, exist_ok=True)

    for img_path in images:
        print(f"\n📸 Image: {os.path.basename(img_path)}")
        tensor = preprocess(img_path, args.img_size)
        probs  = predict(model, tensor, device)
        print_predictions(probs)

        if args.gradcam:
            top_class = int(np.argmax(probs))
            save_path = os.path.join(args.output_dir,
                                     os.path.splitext(os.path.basename(img_path))[0] + "_gradcam.png")
            visualize_gradcam(model, img_path, top_class, CLASSES[top_class],
                              save_path=save_path, img_size=args.img_size)


if __name__ == "__main__":
    main()
