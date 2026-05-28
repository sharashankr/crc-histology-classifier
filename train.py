"""
train.py
--------
MLflow-instrumented training loop for CNN / ViT / Hybrid benchmarking
on NCT-CRC-HE histology dataset with optional diffusion augmentation.

Two-stage training per model:
  Stage 1 — frozen backbone, train head only (5 epochs)
  Stage 2 — unfreeze top layers, fine-tune end-to-end (10 epochs)

Ablation flag:
  --use_synthetic   injects 200 synthetic DEB+STR patches into training set
  (omit flag)       trains on real data only → baseline run

Usage:
    cd "Trial Calibrated Synthetic Data/Image Analysis"

    # Baseline run (real data only)
    python train.py --model cnn
    python train.py --model vit
    python train.py --model hybrid

    # Ablation run (real + synthetic)
    python train.py --model cnn    --use_synthetic
    python train.py --model vit    --use_synthetic
    python train.py --model hybrid --use_synthetic

MLflow UI:
    mlflow ui --port 5000
    open http://localhost:5000
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.loader import (
    CLASS_NAMES,
    NUM_CLASSES,
    get_dataloaders,
)
from models import build_model
from models.cnn    import freeze_backbone as freeze_cnn,    get_optimizer as cnn_opt,    unfreeze_backbone
from models.vit    import freeze_backbone as freeze_vit,    get_optimizer as vit_opt,    unfreeze_last_n_blocks
from models.hybrid import freeze_backbone as freeze_hybrid, get_optimizer as hybrid_opt, unfreeze_last_stage


# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent
SYNTH_DIR     = PROJECT_ROOT / "outputs" / "synthetic"
CHECKPT_DIR   = PROJECT_ROOT / "outputs" / "checkpoints"
MLFLOW_DIR    = PROJECT_ROOT / "outputs" / "mlruns"
CHECKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Per-model training config ─────────────────────────────────────────────────

MODEL_CONFIG = {
    "cnn": {
        "stage1_epochs": 5,
        "stage2_epochs": 10,
        "stage1_lr":     1e-3,
        "stage2_lr":     1e-4,
        "batch_size":    32,
        "freeze_fn":     freeze_cnn,
        "unfreeze_fn":   lambda m: unfreeze_backbone(m),
        "opt_fn":        cnn_opt,
        "label":         "ResNet-50",
    },
    "vit": {
        "stage1_epochs": 5,
        "stage2_epochs": 15,   # ViT needs more epochs to converge
        "stage1_lr":     3e-5,
        "stage2_lr":     1e-5,
        "batch_size":    16,   # ViT uses more memory
        "freeze_fn":     freeze_vit,
        "unfreeze_fn":   lambda m: unfreeze_last_n_blocks(m, n=4),
        "opt_fn":        vit_opt,
        "label":         "ViT-B/16",
    },
    "hybrid": {
        "stage1_epochs": 5,
        "stage2_epochs": 10,
        "stage1_lr":     5e-4,
        "stage2_lr":     5e-5,
        "batch_size":    32,
        "freeze_fn":     freeze_hybrid,
        "unfreeze_fn":   lambda m: unfreeze_last_stage(m),
        "opt_fn":        hybrid_opt,
        "label":         "ConvNeXt-Base",
    },
}


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Training utilities ────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    scaler=None,
) -> tuple:
    """One training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.type):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> tuple:
    """One eval epoch. Returns (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  eval ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def run_stage(
    stage: int,
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    device: torch.device,
    run_name: str,
) -> nn.Module:
    """
    Run one training stage (frozen head-only or full fine-tune).
    Logs all metrics to the active MLflow run.
    Returns the model with best val accuracy weights loaded.
    """
    epochs = config[f"stage{stage}_epochs"]
    lr     = config[f"stage{stage}_lr"]

    optimizer  = config["opt_fn"](model, stage=stage, base_lr=lr)
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)

    # MPS doesn't support GradScaler — use it only on CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_acc  = 0.0
    best_ckpt     = CHECKPT_DIR / f"{run_name}_stage{stage}_best.pt"

    print(f"\n── Stage {stage} | {epochs} epochs | lr={lr} ──────────────────")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc, preds, labels = eval_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        f1  = f1_score(labels, preds, average="macro", zero_division=0)
        elapsed = time.time() - t0

        # ── MLflow logging ───────────────────────────────────────────────────
        step = (stage - 1) * 20 + epoch   # global step across both stages
        mlflow.log_metrics({
            f"s{stage}_train_loss": round(train_loss, 4),
            f"s{stage}_train_acc":  round(train_acc,  4),
            f"s{stage}_val_loss":   round(val_loss,   4),
            f"s{stage}_val_acc":    round(val_acc,    4),
            f"s{stage}_val_f1":     round(f1,         4),
        }, step=step)

        print(
            f"  Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f} f1={f1:.3f} | "
            f"{elapsed:.1f}s"
        )

        # ── Save best checkpoint ─────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)

    # Load best weights before returning
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"  Best val acc (stage {stage}): {best_val_acc:.4f}")
    mlflow.log_metric(f"best_s{stage}_val_acc", round(best_val_acc, 4))
    return model


# ── Synthetic sample loader ───────────────────────────────────────────────────

def load_synthetic_samples(class_names: list) -> list:
    """
    Loads pre-generated synthetic images from outputs/synthetic/.
    Returns list of (PIL.Image, int) pairs.
    """
    from PIL import Image as PILImage

    samples = []
    for class_idx, name in enumerate(class_names):
        class_dir = SYNTH_DIR / name
        if not class_dir.exists():
            continue
        imgs = list(class_dir.glob("*.png"))
        for img_path in imgs:
            img = PILImage.open(img_path).convert("RGB")
            samples.append((img, class_idx))

    print(f"  Loaded {len(samples)} synthetic samples from disk.")
    per_class = {}
    for _, lbl in samples:
        per_class[class_names[lbl]] = per_class.get(class_names[lbl], 0) + 1
    for name, count in per_class.items():
        print(f"    {name}: {count}")
    return samples


# ── Main training function ────────────────────────────────────────────────────

def train(args):
    device = get_device()
    config = MODEL_CONFIG[args.model]
    print(f"\nModel  : {config['label']}")
    print(f"Device : {device}")
    print(f"Synth  : {args.use_synthetic}")

    # ── Load synthetic samples if requested ──────────────────────────────────
    synthetic_samples = None
    if args.use_synthetic:
        print("\nLoading synthetic samples...")
        synthetic_samples = load_synthetic_samples(CLASS_NAMES)
        if not synthetic_samples:
            print("  WARNING: no synthetic samples found in outputs/synthetic/")
            print("  Run: python - << ... generate_synthetic_samples() first")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    print("\nBuilding dataloaders...")
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        batch_size=config["batch_size"],
        num_workers=0,   # 0 for MPS stability
        synthetic_samples=synthetic_samples,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # ── MLflow setup ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("crc-histology-benchmark")

    run_name = (
        f"{args.model}"
        f"{'_synthetic' if args.use_synthetic else '_baseline'}"
        f"_seed{args.seed}"
    )

    with mlflow.start_run(run_name=run_name):

        # ── Log hyperparameters ──────────────────────────────────────────────
        mlflow.log_params({
            "model":            args.model,
            "model_label":      config["label"],
            "use_synthetic":    args.use_synthetic,
            "n_synthetic":      meta["n_synthetic"],
            "n_train":          meta["n_train"],
            "n_val":            meta["n_val"],
            "n_test":           meta["n_test"],
            "stage1_epochs":    config["stage1_epochs"],
            "stage2_epochs":    config["stage2_epochs"],
            "stage1_lr":        config["stage1_lr"],
            "stage2_lr":        config["stage2_lr"],
            "batch_size":       config["batch_size"],
            "seed":             args.seed,
            "device":           str(device),
            "minority_classes": str([CLASS_NAMES[i] for i in meta["minority_classes"]]),
        })

        # ── Stage 1: frozen backbone ─────────────────────────────────────────
        config["freeze_fn"](model)
        model = run_stage(
            stage=1,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            run_name=run_name,
        )

        # ── Stage 2: partial unfreeze ────────────────────────────────────────
        config["unfreeze_fn"](model)
        model = run_stage(
            stage=2,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            run_name=run_name,
        )

        # ── Final test evaluation ────────────────────────────────────────────
        print("\n── Final test evaluation ───────────────────────────────────")
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, preds, labels = eval_epoch(
            model, test_loader, criterion, device
        )
        test_f1 = f1_score(labels, preds, average="macro", zero_division=0)

        # Per-class accuracy — key for ablation analysis
        per_class_acc = {}
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            mask = np.array(labels) == cls_idx
            if mask.sum() > 0:
                cls_acc = accuracy_score(
                    np.array(labels)[mask],
                    np.array(preds)[mask]
                )
                per_class_acc[cls_name] = round(cls_acc, 4)

        print(f"  Test accuracy : {test_acc:.4f}")
        print(f"  Test macro F1 : {test_f1:.4f}")
        print("\n  Per-class accuracy:")
        for name, acc in per_class_acc.items():
            flag = " ← minority" if name in ["DEB", "STR"] else ""
            print(f"    {name:<5}  {acc:.4f}{flag}")

        print("\n  Classification report:")
        print(classification_report(
            labels, preds,
            target_names=CLASS_NAMES,
            digits=3,
            zero_division=0,
        ))

        # ── Log final metrics to MLflow ──────────────────────────────────────
        mlflow.log_metrics({
            "test_loss":     round(test_loss, 4),
            "test_accuracy": round(test_acc,  4),
            "test_macro_f1": round(test_f1,   4),
            **{f"test_acc_{k}": v for k, v in per_class_acc.items()},
        })

        # ── Save per-class results as artifact ───────────────────────────────
        results = {
            "model":          args.model,
            "use_synthetic":  args.use_synthetic,
            "test_accuracy":  round(test_acc, 4),
            "test_macro_f1":  round(test_f1,  4),
            "per_class_acc":  per_class_acc,
        }
        results_path = CHECKPT_DIR / f"{run_name}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact(str(results_path))

        # ── Register best model ──────────────────────────────────────────────
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"crc_{args.model}",
        )

        print(f"\nRun '{run_name}' complete.")
        print(f"MLflow UI: mlflow ui --backend-store-uri file://{MLFLOW_DIR}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRC histology classifier")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "vit", "hybrid"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Inject synthetic DEB+STR samples into training set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)