"""
models/cnn.py
-------------
ResNet-50 backbone fine-tuned for NCT-CRC-HE 9-class histology classification.

Two-stage training strategy:
  Stage 1 — frozen backbone, train head only (fast, stable)
  Stage 2 — unfreeze all layers, low LR fine-tune (squeezes extra accuracy)

Usage:
    from models.cnn import build_cnn
    model = build_cnn(num_classes=9, pretrained=True)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def build_cnn(
    num_classes: int = 9,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Returns a ResNet-50 with a custom classification head.

    Architecture:
        ResNet-50 backbone (ImageNet pretrained)
        └── AdaptiveAvgPool2d (built-in)
        └── Dropout(0.3)
        └── Linear(2048 → 512)
        └── ReLU
        └── Dropout(0.3)
        └── Linear(512 → num_classes)

    The two-layer head gives the model capacity to adapt ImageNet features
    to H&E histology without overfitting on 5K training samples.

    Args:
        num_classes: number of tissue classes (9 for NCT-CRC-HE)
        pretrained:  load ImageNet weights
        dropout:     dropout rate in classification head

    Returns:
        nn.Module with .backbone and .head attributes for easy layer access
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = models.resnet50(weights=weights)

    # Replace the default single-layer fc with a richer head
    in_features = backbone.fc.in_features   # 2048 for ResNet-50
    backbone.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )

    # Convenience attributes for freeze/unfreeze helpers
    backbone.head = backbone.fc

    return backbone


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all layers except the classification head.
    Use for Stage 1 training — fast convergence, head only.
    """
    for name, param in model.named_parameters():
        if "fc" not in name and "head" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [CNN] Backbone frozen. Trainable params: {trainable:,}")


def unfreeze_backbone(model: nn.Module, lr_scale: float = 0.1) -> None:
    """
    Unfreeze all layers for Stage 2 fine-tuning.
    lr_scale is a reminder — apply lower LR to backbone in optimizer.
    """
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [CNN] Backbone unfrozen. Trainable params: {trainable:,}")
    print(f"  [CNN] Use backbone LR = base_lr × {lr_scale} in optimizer.")


def get_optimizer(model: nn.Module, stage: int = 1, base_lr: float = 1e-3):
    """
    Returns optimizer with differential learning rates.

    Stage 1: only head params, single LR
    Stage 2: backbone at 0.1× LR, head at 1× LR
    """
    if stage == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=base_lr, weight_decay=1e-4)

    # Stage 2: differential LR
    head_params     = list(model.fc.parameters())
    head_param_ids  = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    return torch.optim.AdamW([
        {"params": backbone_params, "lr": base_lr * 0.1},
        {"params": head_params,     "lr": base_lr},
    ], weight_decay=1e-4)


if __name__ == "__main__":
    model = build_cnn(num_classes=9, pretrained=False)
    total  = sum(p.numel() for p in model.parameters())
    freeze_backbone(model)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape : {out.shape}")     # [2, 9]
    print(f"Total params : {total:,}")       # ~25M