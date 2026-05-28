"""
models/hybrid.py
----------------
ConvNeXt-Base — a "hybrid" CNN that incorporates transformer design principles
(depthwise convolutions, inverted bottlenecks, LayerNorm, GELU) while keeping
the convolutional inductive bias that works well on small datasets.

Why "hybrid" for this project:
    Pure CNN (ResNet) → strong spatial inductive bias, data-efficient
    Pure ViT           → powerful global attention, needs more data
    ConvNeXt           → CNN structure + transformer design choices
                         Best of both — consistently outperforms both on
                         small-to-medium pathology datasets.

Reference: "A ConvNet for the 2020s" — Liu et al. 2022

Usage:
    from models.hybrid import build_hybrid
    model = build_hybrid(num_classes=9, pretrained=True)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Base_Weights


def build_hybrid(
    num_classes: int = 9,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Returns a ConvNeXt-Base with a custom classification head.

    Architecture:
        ConvNeXt-Base backbone (ImageNet-1K pretrained)
        └── AdaptiveAvgPool + LayerNorm (built-in)
        └── Flatten
        └── Dropout(0.2)
        └── Linear(1024 → 256)
        └── GELU
        └── Dropout(0.2)
        └── Linear(256 → num_classes)

    ConvNeXt-Base output features: 1024-dim
    Smaller head than ResNet-50 (1024→256 vs 2048→512) because
    ConvNeXt features are already more discriminative.

    Args:
        num_classes: number of tissue classes
        pretrained:  load ImageNet weights
        dropout:     dropout rate in head

    Returns:
        nn.Module with .classifier attribute for head access
    """
    weights  = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.convnext_base(weights=weights)

    # ConvNeXt's built-in head: LayerNorm → Linear(1024 → 1000)
    in_features = backbone.classifier[-1].in_features   # 1024

    backbone.classifier[-1] = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )

    # Convenience alias
    backbone.head = backbone.classifier

    return backbone


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all ConvNeXt stages, train classifier only.
    ConvNeXt stages: features.0 through features.7
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Hybrid] Backbone frozen. Trainable params: {trainable:,}")


def unfreeze_last_stage(model: nn.Module) -> None:
    """
    Unfreeze the last ConvNeXt stage (features.6 + features.7) + classifier.
    ConvNeXt has 4 stages, each with 2 sub-modules (downsample + blocks).
    Unfreezing the last stage (highest-level features) is usually enough.
    """
    for param in model.parameters():
        param.requires_grad = False   # freeze all first

    for name, param in model.named_parameters():
        if any(s in name for s in ["features.6", "features.7", "classifier"]):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Hybrid] Last stage unfrozen. Trainable params: {trainable:,}")


def get_optimizer(model: nn.Module, stage: int = 1, base_lr: float = 5e-4):
    """
    Stage 1: classifier only
    Stage 2: last stage at 0.1× LR + classifier at 1× LR
    """
    if stage == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=base_lr, weight_decay=1e-4)

    clf_params    = list(model.classifier.parameters())
    clf_param_ids = {id(p) for p in clf_params}
    other_params  = [p for p in model.parameters()
                     if p.requires_grad and id(p) not in clf_param_ids]

    return torch.optim.AdamW([
        {"params": other_params, "lr": base_lr * 0.1},
        {"params": clf_params,   "lr": base_lr},
    ], weight_decay=1e-4)


if __name__ == "__main__":
    model = build_hybrid(num_classes=9, pretrained=False)
    total = sum(p.numel() for p in model.parameters())
    freeze_backbone(model)
    x   = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape : {out.shape}")    # [2, 9]
    print(f"Total params : {total:,}")      # ~89M