from .cnn    import build_cnn,    freeze_backbone as freeze_cnn,    get_optimizer as cnn_optimizer
from .vit    import build_vit,    freeze_backbone as freeze_vit,    get_optimizer as vit_optimizer
from .hybrid import build_hybrid, freeze_backbone as freeze_hybrid, get_optimizer as hybrid_optimizer

MODEL_REGISTRY = {
    "cnn":    build_cnn,
    "vit":    build_vit,
    "hybrid": build_hybrid,
}

def build_model(name: str, num_classes: int = 9, pretrained: bool = True):
    """Factory function — build any model by name string."""
    assert name in MODEL_REGISTRY, f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}"
    return MODEL_REGISTRY[name](num_classes=num_classes, pretrained=pretrained)