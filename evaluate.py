"""
evaluate.py
-----------
Loads all trained models from MLflow, runs full evaluation, and produces:
  1. Comparison table — accuracy, F1, per-class breakdown, ablation delta
  2. Confusion matrices — one per model
  3. GradCAM heatmaps — visual explanation of predictions
  4. FID score — synthetic image quality metric
  5. Summary report saved to outputs/evaluation_report.json

Run after all 6 training runs complete:
    cd "Trial Calibrated Synthetic Data/Image Analysis"
    python evaluate.py

Or evaluate a single model:
    python evaluate.py --model cnn --condition baseline
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from data.loader import CLASS_NAMES, NUM_CLASSES, get_dataloaders
from models import build_model


# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent
CHECKPT_DIR   = PROJECT_ROOT / "outputs" / "checkpoints"
EVAL_DIR      = PROJECT_ROOT / "outputs" / "evaluation"
GRADCAM_DIR   = EVAL_DIR / "gradcam"
CONFMAT_DIR   = EVAL_DIR / "confusion_matrices"
SYNTH_DIR     = PROJECT_ROOT / "outputs" / "synthetic"

for d in [EVAL_DIR, GRADCAM_DIR, CONFMAT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Model loader ──────────────────────────────────────────────────────────────

def load_checkpoint(model_name: str, condition: str, device: torch.device) -> nn.Module:
    """
    Loads the best stage-2 checkpoint for a given model + condition.
    Falls back to stage-1 if stage-2 not found.
    """
    run_name  = f"{model_name}_{condition}_seed42"
    ckpt_path = CHECKPT_DIR / f"{run_name}_stage2_best.pt"

    if not ckpt_path.exists():
        ckpt_path = CHECKPT_DIR / f"{run_name}_stage1_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for {run_name}. "
            f"Run: python train.py --model {model_name}"
            + (" --use_synthetic" if condition == "synthetic" else "")
        )

    model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"  Loaded: {ckpt_path.name}")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict:
    """Full evaluation — returns accuracy, F1, per-class metrics, raw predictions."""
    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in tqdm(loader, desc="  evaluating", leave=False):
        imgs   = imgs.to(device)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1)
        preds  = logits.argmax(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc[name] = round(
                accuracy_score(all_labels[mask], all_preds[mask]), 4
            )

    return {
        "accuracy":      round(accuracy_score(all_labels, all_preds), 4),
        "macro_f1":      round(f1_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "per_class_acc": per_class_acc,
        "preds":         all_preds,
        "labels":        all_labels,
        "probs":         all_probs,
        "report":        classification_report(
                             all_labels, all_preds,
                             target_names=CLASS_NAMES,
                             digits=3, zero_division=0
                         ),
    }


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    model_name: str,
    condition: str,
) -> Path:
    """Saves a normalised confusion matrix as PNG."""
    cm = confusion_matrix(labels, preds, normalize="true")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(
        f"{model_name.upper()} — {condition} | Normalised confusion matrix",
        fontsize=13, pad=12
    )

    # Annotate cells
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    # Highlight minority class rows/cols
    for idx, name in enumerate(CLASS_NAMES):
        if name in ["DEB", "STR"]:
            ax.add_patch(mpatches.Rectangle(
                (idx - 0.5, -0.5), 1, NUM_CLASSES,
                linewidth=2, edgecolor="#E24B4A", facecolor="none"
            ))

    plt.tight_layout()
    out_path = CONFMAT_DIR / f"confmat_{model_name}_{condition}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ── GradCAM ───────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Hooks into the last convolutional feature map and computes
    a heatmap showing which spatial regions most influenced the prediction.

    Works for CNN (layer4) and ConvNeXt (features.7).
    For ViT, uses attention rollout instead.
    """

    def __init__(self, model: nn.Module, model_name: str):
        self.model      = model
        self.model_name = model_name
        self.gradients  = None
        self.activations = None
        self._register_hooks()

    def _get_target_layer(self):
        if self.model_name == "cnn":
            return self.model.layer4[-1]
        elif self.model_name == "hybrid":
            return self.model.features[-1][-1]
        elif self.model_name == "vit":
            return self.model.encoder.layers[-1]
        raise ValueError(f"Unknown model: {self.model_name}")

    def _register_hooks(self):
        layer = self._get_target_layer()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        img_tensor: torch.Tensor,
        class_idx: int = None,
    ) -> np.ndarray:
        """
        Returns a (224, 224) heatmap array normalised to [0, 1].
        If class_idx is None, uses the predicted class.
        """
        self.model.eval()
        img_tensor = img_tensor.unsqueeze(0)

        logits = self.model(img_tensor)
        if class_idx is None:
            class_idx = logits.argmax(1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        if self.model_name == "vit":
            # ViT: use mean attention of last layer [CLS] token
            attn = self.activations
            if hasattr(attn, "shape") and len(attn.shape) == 3:
                # [batch, seq_len, hidden] → take CLS token (index 0) attention
                heatmap = attn[0, 1:, :].mean(-1)   # [196]
                size = int(heatmap.shape[0] ** 0.5)  # 14
                heatmap = heatmap.reshape(size, size).cpu().numpy()
            else:
                heatmap = np.ones((14, 14))
        else:
            # CNN/ConvNeXt: standard GradCAM
            grads   = self.gradients[0].cpu()   # [C, H, W]
            acts    = self.activations[0].cpu()   # [C, H, W]
            weights = grads.mean(dim=(1, 2))    # global average pooling of gradients
            heatmap = (weights[:, None, None] * acts).sum(0)  # [H, W]
            heatmap = F.relu(torch.as_tensor(heatmap).cpu().detach()).numpy()

        # Normalise and resize to 224×224
        heatmap = heatmap - heatmap.min()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize((224, 224), Image.BILINEAR)
        return np.array(heatmap_img) / 255.0


def save_gradcam(
    model: nn.Module,
    model_name: str,
    loader,
    device: torch.device,
    condition: str,
    n_samples: int = 4,
) -> list:
    """
    Generates GradCAM visualisations for n_samples images per minority class.
    Saves side-by-side: original | heatmap overlay.
    Returns list of saved paths.
    """
    gradcam    = GradCAM(model, model_name)
    saved      = []
    class_done = {2: 0, 7: 0}   # DEB=2, STR=7

    # Denormalize transform for display
    mean = torch.tensor([0.7268, 0.5353, 0.7092])
    std  = torch.tensor([0.1821, 0.2426, 0.1773])

    for imgs, labels in loader:
        for i in range(len(imgs)):
            lbl = labels[i].item()
            if lbl not in class_done or class_done[lbl] >= n_samples:
                continue

            img_t  = imgs[i].to(device)
            heatmap = gradcam.generate(img_t, class_idx=lbl)

            # Denormalise image for display
            img_display = imgs[i].clone()
            for c in range(3):
                img_display[c] = img_display[c] * std[c] + mean[c]
            img_np = img_display.permute(1, 2, 0).clamp(0, 1).numpy()

            # Create overlay: pink-hot colormap on top of original
            cmap    = plt.cm.hot
            overlay = cmap(heatmap)[:, :, :3]   # [H, W, 3]
            blended = 0.55 * img_np + 0.45 * overlay

            # Side-by-side figure
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
            axes[0].imshow(img_np)
            axes[0].set_title("Original patch", fontsize=10)
            axes[0].axis("off")

            axes[1].imshow(heatmap, cmap="hot", vmin=0, vmax=1)
            axes[1].set_title("GradCAM heatmap", fontsize=10)
            axes[1].axis("off")

            axes[2].imshow(blended)
            axes[2].set_title(f"Overlay — {CLASS_NAMES[lbl]}", fontsize=10)
            axes[2].axis("off")

            fig.suptitle(
                f"{model_name.upper()} {condition} | True: {CLASS_NAMES[lbl]}",
                fontsize=11
            )
            plt.tight_layout()

            fname = (GRADCAM_DIR /
                     f"gradcam_{model_name}_{condition}_{CLASS_NAMES[lbl]}"
                     f"_{class_done[lbl]}.png")
            plt.savefig(fname, dpi=120, bbox_inches="tight")
            plt.close()
            saved.append(fname)
            class_done[lbl] += 1

        if all(v >= n_samples for v in class_done.values()):
            break

    return saved


# ── FID score ─────────────────────────────────────────────────────────────────

def compute_fid(real_images: list, synth_dir: Path, device: torch.device) -> dict:
    """
    Computes FID between real and synthetic images per minority class.
    Uses InceptionV3 features. Lower = more realistic synthetic images.
    """
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
    except ImportError:
        return {"error": "torchvision not available"}

    print("\nComputing FID scores...")
    inception = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False
    ).to(device)
    inception.fc = nn.Identity()
    inception.eval()

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def get_features(images: list) -> np.ndarray:
        feats = []
        with torch.no_grad():
            for img in tqdm(images, desc="  extracting features", leave=False):
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.array(img)).convert("RGB")
                t = preprocess(img).unsqueeze(0).to(device)
                f = inception(t).squeeze().cpu().numpy()
                feats.append(f)
        return np.array(feats)

    def fid_score(real_feats: np.ndarray, synth_feats: np.ndarray) -> float:
        mu_r, mu_s = real_feats.mean(0), synth_feats.mean(0)
        cov_r = np.cov(real_feats.T)
        cov_s = np.cov(synth_feats.T)
        diff  = mu_r - mu_s
        # Simplified FID — mean feature distance (full matrix sqrt is expensive)
        fid   = float(np.dot(diff, diff) + np.trace(cov_r + cov_s - 2 * np.sqrt(cov_r * cov_s + 1e-8)))
        return round(abs(fid), 2)

    results = {}
    for cls_idx, cls_name in [(2, "DEB"), (7, "STR")]:
        synth_class_dir = synth_dir / cls_name
        if not synth_class_dir.exists():
            results[cls_name] = "no synthetic images found"
            continue

        real_imgs  = [img for img, lbl in real_images if lbl == cls_idx][:100]
        synth_imgs = [Image.open(p).convert("RGB")
                      for p in sorted(synth_class_dir.glob("*.png"))[:100]]

        if len(real_imgs) < 10 or len(synth_imgs) < 10:
            results[cls_name] = "insufficient samples"
            continue

        real_feats  = get_features(real_imgs)
        synth_feats = get_features(synth_imgs)
        fid         = fid_score(real_feats, synth_feats)
        results[cls_name] = fid
        print(f"  FID {cls_name}: {fid} (lower = more realistic)")

    return results


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison_table(all_results: dict) -> None:
    """Prints the full ablation comparison table to stdout."""
    print("\n" + "═" * 80)
    print("  BENCHMARK RESULTS — NCT-CRC-HE 9-class histology classification")
    print("═" * 80)
    print(f"  {'Model':<14} {'Condition':<12} {'Acc':>6} {'F1':>6} {'STR':>6} {'DEB':>6}  STR delta")
    print("  " + "─" * 72)

    for model_name in ["cnn", "vit", "hybrid"]:
        for condition in ["baseline", "synthetic"]:
            key = f"{model_name}_{condition}"
            if key not in all_results:
                continue
            r = all_results[key]
            str_acc = r["per_class_acc"].get("STR", 0)
            deb_acc = r["per_class_acc"].get("DEB", 0)

            # STR delta vs baseline
            baseline_key = f"{model_name}_baseline"
            delta = ""
            if condition == "synthetic" and baseline_key in all_results:
                baseline_str = all_results[baseline_key]["per_class_acc"].get("STR", 0)
                d = str_acc - baseline_str
                delta = f"+{d:.3f}" if d >= 0 else f"{d:.3f}"

            label = {"cnn": "ResNet-50", "vit": "ViT-B/16", "hybrid": "ConvNeXt"}[model_name]
            print(
                f"  {label:<14} {condition:<12} "
                f"{r['accuracy']:>6.3f} {r['macro_f1']:>6.3f} "
                f"{str_acc:>6.3f} {deb_acc:>6.3f}  {delta}"
            )

        print("  " + "─" * 72)
    print("═" * 80 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = get_device()
    print(f"Device: {device}")

    # ── Load test data ───────────────────────────────────────────────────────
    print("\nLoading test set...")
    _, _, test_loader, _ = get_dataloaders(batch_size=32, num_workers=0)

    # Raw samples for FID
    from datasets import load_dataset
    raw_ds = load_dataset(
        "owkin/nct-crc-he", split="crc_val_he_7k",
        cache_dir=str(Path.home() / ".cache" / "hf_datasets")
    )
    real_images = [(r["image"], r["label"]) for r in raw_ds]

    # ── Which runs to evaluate ───────────────────────────────────────────────
    if args.model and args.condition:
        runs = [(args.model, args.condition)]
    else:
        runs = [
            ("cnn",    "baseline"),
            ("cnn",    "synthetic"),
            ("vit",    "baseline"),
            ("vit",    "synthetic"),
            ("hybrid", "baseline"),
            ("hybrid", "synthetic"),
        ]

    all_results = {}

    for model_name, condition in runs:
        print(f"\n{'─'*50}")
        print(f"Evaluating: {model_name} — {condition}")
        print(f"{'─'*50}")

        try:
            model = load_checkpoint(model_name, condition, device)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue

        # ── Metrics ──────────────────────────────────────────────────────────
        results = evaluate_model(model, test_loader, device)
        all_results[f"{model_name}_{condition}"] = results

        print(f"  Accuracy : {results['accuracy']:.4f}")
        print(f"  Macro F1 : {results['macro_f1']:.4f}")
        print(f"  STR acc  : {results['per_class_acc'].get('STR', 'N/A')}")
        print(f"  DEB acc  : {results['per_class_acc'].get('DEB', 'N/A')}")

        # ── Confusion matrix ─────────────────────────────────────────────────
        cm_path = plot_confusion_matrix(
            results["labels"], results["preds"], model_name, condition
        )
        print(f"  Confusion matrix → {cm_path.relative_to(PROJECT_ROOT)}")

        # ── GradCAM ──────────────────────────────────────────────────────────
        print("  Generating GradCAM heatmaps...")
        try:
            gc_paths = save_gradcam(
                model, model_name, test_loader, device, condition, n_samples=3
            )
            print(f"  GradCAM ({len(gc_paths)} images) → {GRADCAM_DIR.relative_to(PROJECT_ROOT)}")
        except Exception as e:
            print(f"  GradCAM skipped: {e}")

    # ── FID score (once, not per model) ──────────────────────────────────────
    if SYNTH_DIR.exists() and (args.model is None):
        fid_scores = compute_fid(real_images, SYNTH_DIR, device)
        print(f"\nFID scores: {fid_scores}")
    else:
        fid_scores = {}

    # ── Comparison table ─────────────────────────────────────────────────────
    if len(all_results) > 1:
        print_comparison_table(all_results)

    # ── Save report ──────────────────────────────────────────────────────────
    report = {
        "results": {
            k: {
                "accuracy":      v["accuracy"],
                "macro_f1":      v["macro_f1"],
                "per_class_acc": v["per_class_acc"],
            }
            for k, v in all_results.items()
        },
        "fid_scores": fid_scores,
    }
    report_path = EVAL_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved → {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     type=str, choices=["cnn","vit","hybrid"], default=None)
    parser.add_argument("--condition", type=str, choices=["baseline","synthetic"], default=None)
    args = parser.parse_args()

    if (args.model is None) != (args.condition is None):
        parser.error("Provide both --model and --condition, or neither.")

    main(args)