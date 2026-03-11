"""
Model Definition — EfficientNetV2-S with custom classification head
for multi-label chest X-ray disease classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ChestXrayModel(nn.Module):
    """
    EfficientNetV2-S backbone with a multi-label classification head.

    Architecture:
        - EfficientNetV2-S pretrained on ImageNet (feature extractor)
        - Custom head: Dropout → Linear(1280 → num_classes)
        - Raw logits output (BCEWithLogitsLoss handles sigmoid internally)

    Args:
        num_classes (int): Number of disease labels (14 for NIH ChestX-ray14).
        pretrained  (bool): Load ImageNet pretrained weights.
        freeze_base (bool): Freeze backbone during initial training phase.
    """

    def __init__(self, num_classes: int = 14, pretrained: bool = True, freeze_base: bool = False):
        super().__init__()

        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_v2_s(weights=weights)

        # Feature extractor (everything except the final classifier)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        # Get the in_features from the original classifier
        in_features = backbone.classifier[1].in_features  # 1280

        # Custom multi-label head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        if freeze_base:
            self._freeze_backbone()

        self._init_classifier()

    def _freeze_backbone(self):
        """Freeze all backbone parameters (useful for Stage 1 training)."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for Stage 2 fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True

    def _init_classifier(self):
        """Xavier initialization for the linear head."""
        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x  # raw logits

    def get_num_params(self) -> dict:
        total  = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    model = ChestXrayModel(num_classes=14, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out   = model(dummy)
    stats = model.get_num_params()

    print(f"Output shape  : {out.shape}")          # (2, 14)
    print(f"Total params  : {stats['total']:,}")
    print(f"Trainable     : {stats['trainable']:,}")
