"""
models/vit.py
-------------
Vision Transformer (ViT-B/16) fine-tuned for NCT-CRC-HE 9-class classification.

ViT splits the 224×224 image into 16×16 patches (196 patches total),
embeds them, and processes with self-attention. Unlike CNNs it has no
inductive bias for locality — it learns global tissue structure from scratch,
which is why it often outperforms CNNs on pathology given enough data.

Usage:
    from models.vit import build_vit
    model = build_vit(num_classes=9, pretrained=True)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights


def build_vit(
    num_classes: int = 9,
    pretrained: bool = True,
    dropout: float = 0.1,
    img_size: int = 224,
) -> nn.Module:
    """
    Returns a ViT-B/16 with a custom classification head.

    Architecture:
        ViT-B/16 backbone (ImageNet-21k pretrained via torchvision)
        └── [CLS] token representation (768-dim)
        └── Dropout(0.1)
        └── Linear(768 → 256)
        └── GELU
        └── Dropout(0.1)
        └── Linear(256 → num_classes)

    Notes:
        - img_size must be 224 (ViT-B/16 patch grid = 14×14 = 196 patches)
        - ViT is more sensitive to LR than ResNet — use smaller base_lr (3e-5)
        - Needs more epochs than CNN to converge on small datasets

    Args:
        num_classes: number of tissue classes
        pretrained:  load ImageNet weights (strongly recommended)
        dropout:     dropout in classification head
        img_size:    must be 224 for ViT-B/16

    Returns:
        nn.Module — ViT backbone with custom head
    """
    assert img_size == 224, "ViT-B/16 requires img_size=224 (14×14 patch grid)"

    weights  = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1 if pretrained else None
    backbone = models.vit_b_16(weights=weights)

    # ViT-B/16 hidden dim is 768
    in_features = backbone.heads.head.in_features   # 768

    backbone.heads = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )

    # Convenience alias
    backbone.head = backbone.heads

    return backbone


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze encoder blocks, keep heads trainable.
    For ViT, Stage 1 freezes the patch embedding + all transformer blocks.
    """
    for name, param in model.named_parameters():
        if "heads" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [ViT] Backbone frozen. Trainable params: {trainable:,}")


def unfreeze_last_n_blocks(model: nn.Module, n: int = 4) -> None:
    """
    Unfreeze the last N transformer encoder blocks + heads.
    Recommended for Stage 2 — unfreezing all 12 blocks on 5K samples risks
    catastrophic forgetting. Last 4 blocks is a good balance.
    """
    # First unfreeze everything to reset, then re-freeze selectively
    for param in model.parameters():
        param.requires_grad = True

    # Freeze patch embedding + positional embedding + first (12-n) blocks
    blocks_to_freeze = 12 - n
    for name, param in model.named_parameters():
        if "encoder.layers" in name:
            # layer name: encoder.layers.encoder_layer_N.*
            parts = name.split(".")
            try:
                layer_idx = int(parts[parts.index("encoder_layer_0"[:14]) + 1]
                                if False else
                                [p for p in parts if p.startswith("encoder_layer_")][0]
                                .replace("encoder_layer_", ""))
                if layer_idx < blocks_to_freeze:
                    param.requires_grad = False
            except (ValueError, IndexError):
                pass
        elif "conv_proj" in name or "class_token" in name or "pos_embedding" in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [ViT] Last {n} blocks unfrozen. Trainable params: {trainable:,}")


def get_optimizer(model: nn.Module, stage: int = 1, base_lr: float = 3e-5):
    """
    ViT needs a smaller LR than CNN — transformers are sensitive to LR.

    Stage 1: head only, lr=3e-5
    Stage 2: last N blocks + head, differential LR
    """
    if stage == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=base_lr, weight_decay=0.01)

    head_params     = list(model.heads.parameters())
    head_param_ids  = {id(p) for p in head_params}
    other_params    = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in head_param_ids]

    return torch.optim.AdamW([
        {"params": other_params, "lr": base_lr * 0.1},
        {"params": head_params,  "lr": base_lr},
    ], weight_decay=0.01)


if __name__ == "__main__":
    model  = build_vit(num_classes=9, pretrained=False)
    total  = sum(p.numel() for p in model.parameters())
    freeze_backbone(model)
    x   = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape : {out.shape}")    # [2, 9]
    print(f"Total params : {total:,}")      # ~86M