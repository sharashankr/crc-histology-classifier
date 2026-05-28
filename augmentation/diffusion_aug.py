"""
augmentation/diffusion_aug.py
------------------------------
Generates synthetic H&E histology patches for minority classes using
Stable Diffusion img2img on Apple Silicon (MPS). Filters outputs by
CLIP similarity to ensure generated images are on-distribution.

Pipeline:
  1. Sample real patches from minority classes (DEB, STR)
  2. Run SD img2img with class-specific prompts at low noise strength
     (preserves tissue structure, adds plausible variation)
  3. Filter by CLIP cosine similarity against the class prompt
  4. Return list of (PIL.Image, int) ready for get_dataloaders()

Run from project root:
    cd "Trial Calibrated Synthetic Data/Image Analysis"
    python -m augmentation.diffusion_aug

Usage in train.py:
    from augmentation.diffusion_aug import generate_synthetic_samples
    synthetic = generate_synthetic_samples(
        dataset=all_data,
        minority_classes=meta["minority_classes"],
        class_names=meta["class_names"],
        n_per_class=100,
    )
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        synthetic_samples=synthetic
    )
"""

import gc
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATS_PATH   = PROJECT_ROOT / "data" / "dataset_stats.json"
SYNTH_DIR    = PROJECT_ROOT / "outputs" / "synthetic"

# ── Class-specific prompts ────────────────────────────────────────────────────
# Carefully worded to stay within H&E histology domain.
# Strength controls how much SD changes the source image:
#   0.3 = subtle variation (safe, structure-preserving)
#   0.5 = moderate variation (more diversity, slightly riskier)

CLASS_PROMPTS = {
    "DEB": (
        "hematoxylin and eosin stained histology, debris and necrotic tissue, "
        "colorectal cancer, cellular fragments, eosinophilic material, "
        "224x224 microscopy patch, high quality pathology slide",
        0.35,   # low strength — debris is visually noisy, preserve that
    ),
    "STR": (
        "hematoxylin and eosin stained histology, cancer-associated stroma, "
        "colorectal adenocarcinoma, spindle cells, fibrous connective tissue, "
        "desmoplastic stroma, 224x224 microscopy patch, high quality pathology slide",
        0.40,   # slightly higher — stroma has more regular structure
    ),
    "LYM": (
        "hematoxylin and eosin stained histology, lymphocytes, immune cells, "
        "small dark round nuclei, colorectal cancer tissue, "
        "224x224 microscopy patch, high quality pathology slide",
        0.35,
    ),
    "MUS": (
        "hematoxylin and eosin stained histology, smooth muscle tissue, "
        "elongated muscle cells, eosinophilic cytoplasm, colorectal tissue, "
        "224x224 microscopy patch, high quality pathology slide",
        0.35,
    ),
}

# Negative prompt applied to all classes — steers away from artefacts
NEGATIVE_PROMPT = (
    "blurry, low quality, oversaturated, cartoon, drawing, text, "
    "out of focus, jpeg artefacts, watermark"
)


# ── Device setup ─────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Returns MPS on Apple Silicon, CUDA if available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── SD pipeline loader ────────────────────────────────────────────────────────

def load_sd_pipeline(device: torch.device):
    """
    Loads Stable Diffusion v1.5 img2img pipeline in float16.
    First run downloads ~4GB to ~/.cache/huggingface/hub/.
    Subsequent runs load from cache instantly.
    """
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
    except ImportError:
        raise ImportError(
            "diffusers not installed. Run: pip install diffusers accelerate"
        )

    print("Loading Stable Diffusion v1.5 img2img pipeline...")
    print("  (first run downloads ~4 GB — cached after)")

    dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None,       # disable — histology is not flagged content
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    # MPS-specific: disable attention slicing (causes issues on some M-series)
    if device.type == "mps":
        pipe.enable_attention_slicing(1)

    pipe.set_progress_bar_config(disable=True)
    print(f"  Pipeline loaded on {device}")
    return pipe


# ── CLIP filter ───────────────────────────────────────────────────────────────

def load_clip(device: torch.device):
    """Loads OpenCLIP ViT-B/32 for quality filtering."""
    try:
        import open_clip
    except ImportError:
        raise ImportError(
            "open_clip not installed. Run: pip install open-clip-torch"
        )
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def clip_similarity(
    images: list,
    prompt: str,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    device: torch.device,
) -> list:
    """
    Returns cosine similarity scores between each image and the text prompt.
    Higher = more similar to the class description = better quality synthetic.
    """
    text_tokens = clip_tokenizer([prompt]).to(device)
    text_feat   = clip_model.encode_text(text_tokens)
    text_feat   = text_feat / text_feat.norm(dim=-1, keepdim=True)

    scores = []
    for img in images:
        img_t    = clip_preprocess(img).unsqueeze(0).to(device)
        img_feat = clip_model.encode_image(img_t)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        score    = (img_feat @ text_feat.T).item()
        scores.append(score)
    return scores


# ── Core generation function ──────────────────────────────────────────────────

def generate_for_class(
    class_name: str,
    class_idx: int,
    source_images: list,
    n_target: int,
    pipe,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    device: torch.device,
    clip_threshold: float = 0.22,
    batch_size: int = 4,
    seed: int = 42,
) -> list:
    """
    Generates synthetic patches for a single class.

    Args:
        class_name:      e.g. "DEB"
        class_idx:       integer label
        source_images:   real PIL images from this class to use as init images
        n_target:        how many synthetic samples to produce
        pipe:            SD img2img pipeline
        clip_*:          CLIP model components for quality filtering
        device:          torch device
        clip_threshold:  minimum CLIP similarity to keep a generated image
        batch_size:      images per SD inference call (lower = less VRAM)
        seed:            for reproducibility

    Returns:
        List of (PIL.Image, int) pairs — synthetic image + class label
    """
    if class_name not in CLASS_PROMPTS:
        print(f"  No prompt defined for {class_name}, skipping.")
        return []

    prompt, strength = CLASS_PROMPTS[class_name]
    generator = torch.Generator(device=device).manual_seed(seed)
    rng       = random.Random(seed)

    accepted  = []
    attempts  = 0
    max_attempts = n_target * 4   # generate up to 4x target, filter down

    print(f"\n  Generating for {class_name} (target: {n_target}, "
          f"strength: {strength}, CLIP threshold: {clip_threshold})")

    with tqdm(total=n_target, desc=f"  {class_name}", unit="img") as pbar:
        while len(accepted) < n_target and attempts < max_attempts:
            # Sample source images (with replacement)
            batch_sources = [
                rng.choice(source_images).resize((512, 512), Image.LANCZOS)
                for _ in range(min(batch_size, n_target - len(accepted)))
            ]

            try:
                # MPS requires single images — iterate one at a time
                outputs = []
                for src in batch_sources:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        image=src,
                        strength=strength,
                        guidance_scale=7.5,
                        num_inference_steps=30,
                        generator=generator,
                    ).images[0]
                    outputs.append(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                attempts += len(batch_sources)
                continue

            # Resize back to 224x224 (model input size)
            outputs_resized = [img.resize((224, 224), Image.LANCZOS) for img in outputs]

            # CLIP quality filter
            scores = clip_similarity(
                outputs_resized, prompt,
                clip_model, clip_preprocess, clip_tokenizer, device
            )

            for img, score in zip(outputs_resized, scores):
                if score >= clip_threshold:
                    accepted.append((img, class_idx))
                    pbar.update(1)
                    if len(accepted) >= n_target:
                        break

            attempts += len(batch_sources)

        # Free MPS memory between classes
        if device.type == "mps":
            gc.collect()
            torch.mps.empty_cache()

    kept_pct = len(accepted) / max(attempts, 1) * 100
    print(f"  {class_name}: kept {len(accepted)}/{attempts} "
          f"({kept_pct:.0f}%) above CLIP threshold {clip_threshold}")
    return accepted


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_synthetic_samples(
    dataset,
    minority_classes: list,
    class_names: list,
    n_per_class: int = 100,
    clip_threshold: float = 0.22,
    batch_size: int = 2,
    seed: int = 42,
    save_to_disk: bool = True,
) -> list:
    """
    Full pipeline: load SD + CLIP, generate synthetic patches for each
    minority class, filter by quality, optionally save to disk.

    Args:
        dataset:          HF dataset or list of dicts with "image"/"label" keys
        minority_classes: list of class indices to augment (from loader meta)
        class_names:      list of all class name strings
        n_per_class:      synthetic samples to generate per minority class
        clip_threshold:   minimum CLIP similarity score (0–1) to accept image
        batch_size:       SD batch size — keep at 2 for MPS 16GB, 4 for 24GB+
        seed:             reproducibility seed
        save_to_disk:     save accepted images to outputs/synthetic/

    Returns:
        List of (PIL.Image, int) — ready for get_dataloaders(synthetic_samples=)
    """
    device   = get_device()
    print(f"Device: {device}")

    # ── Group real images by class ───────────────────────────────────────────
    print("Indexing source images by class...")
    class_images: dict = {i: [] for i in minority_classes}
    for record in tqdm(dataset, desc="  indexing", leave=False):
        lbl = record["label"]
        if lbl in class_images:
            img = record["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img)).convert("RGB")
            class_images[lbl].append(img)

    for idx in minority_classes:
        print(f"  {class_names[idx]}: {len(class_images[idx])} source images")

    # ── Load models ──────────────────────────────────────────────────────────
    pipe = load_sd_pipeline(device)
    clip_model, clip_preprocess, clip_tokenizer = load_clip(device)

    # ── Generate ─────────────────────────────────────────────────────────────
    all_synthetic = []
    for class_idx in minority_classes:
        name    = class_names[class_idx]
        sources = class_images[class_idx]
        if not sources:
            print(f"  No source images found for {name}, skipping.")
            continue

        samples = generate_for_class(
            class_name=name,
            class_idx=class_idx,
            source_images=sources,
            n_target=n_per_class,
            pipe=pipe,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            clip_tokenizer=clip_tokenizer,
            device=device,
            clip_threshold=clip_threshold,
            batch_size=batch_size,
            seed=seed,
        )
        all_synthetic.extend(samples)

        # ── Optionally save to disk ──────────────────────────────────────────
        if save_to_disk and samples:
            out_dir = SYNTH_DIR / name
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, (img, _) in enumerate(samples):
                img.save(out_dir / f"{name}_synth_{i:04d}.png")
            print(f"  Saved {len(samples)} images to {out_dir.relative_to(PROJECT_ROOT)}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n── Synthetic generation summary ─────────────────────")
    for class_idx in minority_classes:
        name  = class_names[class_idx]
        count = sum(1 for _, lbl in all_synthetic if lbl == class_idx)
        print(f"  {name}: {count} synthetic samples generated")
    print(f"  Total: {len(all_synthetic)} samples")
    print(f"─────────────────────────────────────────────────────\n")

    # ── Persist metadata ─────────────────────────────────────────────────────
    if save_to_disk:
        meta_path = SYNTH_DIR / "generation_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "n_per_class":     n_per_class,
            "clip_threshold":  clip_threshold,
            "batch_size":      batch_size,
            "seed":            seed,
            "device":          str(device),
            "classes_augmented": {
                class_names[i]: sum(1 for _, lbl in all_synthetic if lbl == i)
                for i in minority_classes
            },
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Metadata saved to {meta_path.relative_to(PROJECT_ROOT)}")

    return all_synthetic


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datasets import load_dataset
    from data.loader import CLASS_NAMES

    print("Loading dataset for smoke test...")
    ds = load_dataset(
        "owkin/nct-crc-he",
        split="crc_val_he_7k",
        cache_dir=str(Path.home() / ".cache" / "hf_datasets"),
    )

    # Quick test: generate just 4 samples per minority class
    synthetic = generate_synthetic_samples(
        dataset=ds,
        minority_classes=[2, 7],       # DEB=2, STR=7
        class_names=CLASS_NAMES,
        n_per_class=4,                 # small number for smoke test
        clip_threshold=0.20,           # lower threshold for quick test
        batch_size=2,
        save_to_disk=True,
    )

    print(f"\nSmoke test passed — {len(synthetic)} synthetic samples generated")
    if synthetic:
        print(f"Sample: label={synthetic[0][1]}, "
              f"image size={synthetic[0][0].size}, "
              f"mode={synthetic[0][0].mode}")
    else:
        print("No samples passed the CLIP filter — try lowering clip_threshold.")