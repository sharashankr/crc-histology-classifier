"""
data/class_stats.py
-------------------
Run once after downloading the dataset to:
  1. Visualise class distribution and flag minority classes
  2. Compute dataset-specific mean/std for H&E normalization
  3. Save results to data/dataset_stats.json

Run from project root:
    cd "Trial Calibrated Synthetic Data/Image Analysis"
    python -m data.class_stats
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # Image Analysis/
DATA_DIR     = PROJECT_ROOT / "data"
STATS_PATH   = DATA_DIR / "dataset_stats.json"
CACHE_DIR    = Path.home() / ".cache" / "hf_datasets"


def compute_mean_std(hf_dataset, img_size: int = 224, max_samples: int = 10_000, seed: int = 42) -> dict:
    """
    Estimate per-channel mean and std over a random subset.
    Uses Welford's online algorithm — no large arrays in memory.
    """
    print(f"\nComputing mean/std over up to {max_samples:,} samples...")
    to_tensor = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    rng     = np.random.default_rng(seed)
    n_total = len(hf_dataset)
    indices = rng.choice(n_total, size=min(max_samples, n_total), replace=False)

    count, mean, M2 = 0, np.zeros(3), np.zeros(3)
    for idx in tqdm(indices, desc="  scanning"):
        img = hf_dataset[int(idx)]["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img)).convert("RGB")
        pixels = to_tensor(img).numpy().reshape(3, -1).T   # [H*W, 3]
        for pixel in pixels:
            count += 1
            delta  = pixel - mean
            mean  += delta / count
            M2    += delta * (pixel - mean)

    std   = np.sqrt(M2 / (count - 1))
    stats = {"mean": mean.tolist(), "std": std.tolist(), "n_pixels_sampled": count}
    print(f"  Mean : {[f'{v:.4f}' for v in stats['mean']]}")
    print(f"  Std  : {[f'{v:.4f}' for v in stats['std']]}")
    return stats


def analyse_distribution(labels: list) -> dict:
    counts    = Counter(labels)
    n_classes = max(counts.keys()) + 1
    count_arr = [counts.get(i, 0) for i in range(n_classes)]
    median    = float(np.median(count_arr))
    return {
        "counts":             count_arr,
        "imbalance_ratio":    round(max(count_arr) / max(min(count_arr), 1), 2),
        "median_count":       int(median),
        "minority_threshold": round(0.75 * median),
    }


def print_distribution(class_names: list, dist: dict) -> None:
    counts    = dist["counts"]
    threshold = dist["minority_threshold"]
    max_count = max(counts)
    bar_width = 40
    print("\n── Class distribution ──────────────────────────────────────────────")
    print(f"  {'Class':<6} {'Count':>8}   {'Bar':40}   Note")
    print("  " + "─" * 72)
    for name, count in zip(class_names, counts):
        filled = int(bar_width * count / max_count)
        bar    = "█" * filled + "░" * (bar_width - filled)
        flag   = "← MINORITY" if count < threshold else ""
        print(f"  {name:<6} {count:>8,}   {bar}   {flag}")
    print("  " + "─" * 72)
    print(f"  Imbalance ratio   : {dist['imbalance_ratio']}x")
    print(f"  Minority threshold: < {int(threshold):,} samples\n")


def main():
    from data.loader import CLASS_NAMES

    print("Loading NCT-CRC-HE (7K)...")
    # Explicit split name — only downloads the 7K split (~715 MB)
    all_data = load_dataset(
        "owkin/nct-crc-he",
        split="crc_val_he_7k",
        cache_dir=str(CACHE_DIR),
    )

    dist       = analyse_distribution(all_data["label"])
    norm_stats = compute_mean_std(all_data)
    print_distribution(CLASS_NAMES, dist)

    output = {
        "class_names":   CLASS_NAMES,
        "distribution":  dist,
        "normalization": norm_stats,
        "note": (
            "Replace IMAGENET_MEAN/STD in data/loader.py with "
            "normalization.mean and normalization.std for H&E-specific accuracy."
        ),
    }
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {STATS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()