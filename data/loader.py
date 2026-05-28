"""
data/loader.py
--------------
Loads the NCT-CRC-HE-7K colorectal cancer histology dataset from Hugging Face,
applies stratified train/val/test splits, and returns PyTorch DataLoaders.

Run from project root:
    cd "Trial Calibrated Synthetic Data/Image Analysis"
    python -m data.loader

Dataset : https://huggingface.co/datasets/1aurent/NCT-CRC-HE  (~715 MB)
Classes (9): ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM
Note    : 7,180 patches from 50 CRC patients — no overlap with the 100K training set.
          Smaller size means fast iteration; class imbalance and clinical relevance
          are preserved, making it ideal for the diffusion augmentation ablation.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
from sklearn.model_selection import train_test_split


# ── Paths ────────────────────────────────────────────────────────────────────

# Project root = "Trial Calibrated Synthetic Data/Image Analysis/"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CACHE_DIR    = Path.home() / ".cache" / "hf_datasets"


# ── Class metadata ────────────────────────────────────────────────────────────

CLASS_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CLASS_DESCRIPTIONS = {
    "ADI":  "Adipose tissue",
    "BACK": "Background",
    "DEB":  "Debris",
    "LYM":  "Lymphocytes",
    "MUC":  "Mucus",
    "MUS":  "Smooth muscle",
    "NORM": "Normal colon mucosa",
    "STR":  "Cancer-associated stroma",
    "TUM":  "Colorectal adenocarcinoma epithelium",
}
NUM_CLASSES = len(CLASS_NAMES)

# Default: ImageNet stats. After running data/class_stats.py, replace with
# the H&E-specific values from data/dataset_stats.json for better accuracy.
IMAGENET_MEAN = [0.7268, 0.5353, 0.7092]
IMAGENET_STD  = [0.1821, 0.2426, 0.1773]


# ── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(split: str, img_size: int = 224) -> transforms.Compose:
    """
    Returns a transform pipeline for the given split.
    Augmentation on train only — all architectures see identical
    preprocessing so benchmarks are fair.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),          # valid: tissue has no orientation
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.1, hue=0.05,             # subtle: H&E staining varies
            ),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class HistologyDataset(Dataset):
    """
    Wraps a list of (image, label) pairs with a transform.
    Accepts PIL images (from HF datasets) or file paths
    (for synthetic images written to disk by augmentation/diffusion_aug.py).
    """

    def __init__(self, samples: list, transform: Optional[transforms.Compose] = None):
        self.samples   = samples    # list of (PIL.Image | Path | str, int)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.samples[idx]

        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("RGB")
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# ── Main loader ───────────────────────────────────────────────────────────────

def get_dataloaders(
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0,
    synthetic_samples: Optional[list] = None,
    use_weighted_sampler: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Downloads (or loads cached) NCT-CRC-HE-100K, splits stratified by label,
    optionally injects synthetic samples into the training set, and returns
    (train_loader, val_loader, test_loader, meta).

    Args:
        split_ratios:         (train, val, test) — must sum to 1.
        batch_size:           Samples per batch.
        img_size:             Resize target. 224 for ResNet/ViT-B, 384 for ViT-L.
        num_workers:          Set 0 on MPS (Apple Silicon) to avoid fork issues.
        synthetic_samples:    List of (PIL.Image, int) from diffusion_aug.py.
                              Appended to train split only — ablation hook.
        use_weighted_sampler: Oversample minority classes during training.
        seed:                 Reproducibility seed.

    Returns:
        train_loader, val_loader, test_loader, meta dict
    """
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios must sum to 1."
    train_frac, val_frac, _ = split_ratios

    print("Loading NCT-CRC-HE (7K) — downloading CRC_VAL_HE_7K split only (~715 MB)...")
    # Explicit split name prevents HF from downloading the full 100K alongside it
    all_data = load_dataset(
        "owkin/nct-crc-he",
        split="crc_val_he_7k",
        cache_dir=str(CACHE_DIR),
    )
    images = [r["image"] for r in all_data]
    labels = [r["label"] for r in all_data]
    print(f"  Loaded {len(images):,} samples across {len(set(labels))} classes.")

    # ── Stratified splits ────────────────────────────────────────────────────
    indices = np.arange(len(labels))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1.0 - train_frac),
        stratify=labels,
        random_state=seed,
    )
    val_relative = val_frac / (1.0 - train_frac)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_relative),
        stratify=[labels[i] for i in temp_idx],
        random_state=seed,
    )

    def _build(idx_list):
        return [(images[i], labels[i]) for i in idx_list]

    train_samples = _build(train_idx)
    val_samples   = _build(val_idx)
    test_samples  = _build(test_idx)

    # ── Inject synthetic samples (train only) ────────────────────────────────
    n_synthetic = 0
    if synthetic_samples:
        train_samples = train_samples + synthetic_samples
        n_synthetic   = len(synthetic_samples)
        print(f"  + {n_synthetic:,} synthetic samples injected into training set.")

    # ── Class counts & minority detection ───────────────────────────────────
    train_labels  = [s[1] for s in train_samples]
    class_counts  = np.bincount(train_labels, minlength=NUM_CLASSES)
    minority_idxs = _get_minority_classes(class_counts)

    # ── Weighted sampler ─────────────────────────────────────────────────────
    sampler = None
    if use_weighted_sampler:
        class_weights  = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[lbl] for lbl in train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float),
            num_samples=len(train_samples),
            replacement=True,
        )

    # ── Build datasets ───────────────────────────────────────────────────────
    train_ds = HistologyDataset(train_samples, get_transforms("train", img_size))
    val_ds   = HistologyDataset(val_samples,   get_transforms("val",   img_size))
    test_ds  = HistologyDataset(test_samples,  get_transforms("test",  img_size))

    # pin_memory=False — Metal (MPS) does not support pinned memory
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)

    train_loader = DataLoader(train_ds, shuffle=(sampler is None), sampler=sampler, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    meta = {
        "class_names":      CLASS_NAMES,
        "class_counts":     class_counts.tolist(),
        "n_train":          len(train_samples),
        "n_val":            len(val_samples),
        "n_test":           len(test_samples),
        "n_synthetic":      n_synthetic,
        "minority_classes": minority_idxs,
    }
    _print_summary(meta)
    return train_loader, val_loader, test_loader, meta


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_minority_classes(class_counts: np.ndarray, threshold_factor: float = 0.75) -> list:
    """Classes whose count < threshold_factor × median — targeted by diffusion aug."""
    median = np.median(class_counts)
    return [i for i, c in enumerate(class_counts) if c < threshold_factor * median]


def _print_summary(meta: dict) -> None:
    print("\n── Dataset split summary ──────────────────────────────")
    print(f"  Train : {meta['n_train']:>8,}  (incl. {meta['n_synthetic']:,} synthetic)")
    print(f"  Val   : {meta['n_val']:>8,}")
    print(f"  Test  : {meta['n_test']:>8,}")
    print("\n  Class distribution (train):")
    for i, (name, count) in enumerate(zip(meta["class_names"], meta["class_counts"])):
        flag = "  ← minority" if i in meta["minority_classes"] else ""
        bar  = "█" * (count // 1000)
        print(f"    {name:<5}  {count:>8,}  {bar}{flag}")
    print("───────────────────────────────────────────────────────\n")


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        batch_size=32,
        num_workers=0,
    )
    imgs, lbls = next(iter(train_loader))
    print(f"Batch shape  : {imgs.shape}")
    print(f"Label dtype  : {lbls.dtype}")
    print(f"Pixel range  : [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"Minority cls : {[meta['class_names'][i] for i in meta['minority_classes']]}")