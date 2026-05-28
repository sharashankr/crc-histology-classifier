"""
app/api.py
----------
FastAPI inference server for CRC histology classification.
Loads the best model from MLflow registry and serves predictions
with GradCAM heatmaps.

Endpoints:
    GET  /health          — liveness check
    GET  /models          — list available models
    POST /predict         — classify an uploaded patch
    POST /predict/gradcam — classify + return GradCAM heatmap

Run locally:
    cd "Trial Calibrated Synthetic Data/Image Analysis"
    uvicorn app.api:app --reload --port 8000

Run via Docker:
    docker-compose up
"""

import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import CLASS_NAMES, CLASS_DESCRIPTIONS, NUM_CLASSES
from models import build_model


# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPT_DIR  = PROJECT_ROOT / "outputs" / "checkpoints"
RESULTS_DIR  = PROJECT_ROOT / "outputs" / "checkpoints"


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CRC Histology Classifier",
    description="CNN / ViT / ConvNeXt tissue classification on NCT-CRC-HE patches",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()


# ── Preprocessing ─────────────────────────────────────────────────────────────

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.7268, 0.5353, 0.7092],
        std=[0.1821, 0.2426, 0.1773],
    ),
])

DENORM_MEAN = torch.tensor([0.7268, 0.5353, 0.7092])
DENORM_STD  = torch.tensor([0.1821, 0.2426, 0.1773])


# ── Model registry ────────────────────────────────────────────────────────────

_model_cache: dict = {}


def get_best_model_info() -> dict:
    """
    Reads result JSONs from checkpoints to find the best model by test accuracy.
    Returns {model_name, condition, accuracy}.
    """
    best = {"model": "cnn", "condition": "baseline", "accuracy": 0.0}
    for results_file in RESULTS_DIR.glob("*_results.json"):
        try:
            with open(results_file) as f:
                r = json.load(f)
            if r.get("test_accuracy", 0) > best["accuracy"]:
                best = {
                    "model":     r["model"],
                    "condition": "synthetic" if r.get("use_synthetic") else "baseline",
                    "accuracy":  r["test_accuracy"],
                }
        except Exception:
            continue
    return best


def load_model(model_name: str, condition: str) -> torch.nn.Module:
    """Loads and caches a model from checkpoint."""
    key = f"{model_name}_{condition}"
    if key in _model_cache:
        return _model_cache[key]

    run_name  = f"{model_name}_{condition}_seed42"
    ckpt_path = CHECKPT_DIR / f"{run_name}_stage2_best.pt"
    if not ckpt_path.exists():
        ckpt_path = CHECKPT_DIR / f"{run_name}_stage1_best.pt"
    if not ckpt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No checkpoint for {model_name} ({condition}). Train it first."
        )

    model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    _model_cache[key] = model
    return model


def get_available_models() -> list:
    """Returns list of trained model checkpoints."""
    available = []
    for ckpt in sorted(CHECKPT_DIR.glob("*_stage2_best.pt")):
        parts = ckpt.stem.replace("_stage2_best", "").split("_")
        if len(parts) >= 3:
            model_name = parts[0]
            condition  = "_".join(parts[1:-1])
            available.append({
                "model":     model_name,
                "condition": condition,
                "checkpoint": ckpt.name,
            })
    return available


# ── GradCAM ───────────────────────────────────────────────────────────────────

class GradCAMExtractor:
    def __init__(self, model: torch.nn.Module, model_name: str):
        self.model      = model
        self.model_name = model_name
        self.gradients  = None
        self.activations = None
        self._register()

    def _get_layer(self):
        if self.model_name == "cnn":
            return self.model.layer4[-1]
        elif self.model_name == "hybrid":
            return self.model.features[-1][-1]
        elif self.model_name == "vit":
            return self.model.encoder.layers[-1]
        raise ValueError(f"Unknown model: {self.model_name}")

    def _register(self):
        layer = self._get_layer()
        layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach())
        )
        layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach())
        )

    def compute(self, img_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        logits = self.model(img_tensor)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        if self.model_name == "vit":
            acts = self.activations
            if acts is not None and len(acts.shape) == 3:
                heatmap = acts[0, 1:, :].mean(-1)
                size    = int(heatmap.shape[0] ** 0.5)
                heatmap = heatmap.reshape(size, size).cpu().numpy()
            else:
                heatmap = np.ones((14, 14))
        else:
            grads   = self.gradients[0]
            acts    = self.activations[0]
            weights = grads.mean(dim=(1, 2))
            heatmap = (weights[:, None, None] * acts).sum(0)
            heatmap = F.relu(torch.as_tensor(heatmap).cpu().detach()).numpy()

        heatmap = heatmap - heatmap.min()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize((224, 224), Image.BILINEAR)
        return np.array(heatmap_img) / 255.0


def heatmap_to_base64(
    original_img: Image.Image,
    heatmap: np.ndarray,
) -> str:
    """Blends original image with heatmap, returns base64 PNG string."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orig_np  = np.array(original_img.resize((224, 224))) / 255.0
    cmap     = plt.cm.hot
    overlay  = cmap(heatmap)[:, :, :3]
    blended  = (0.55 * orig_np + 0.45 * overlay)
    blended  = np.clip(blended * 255, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Monte Carlo Dropout uncertainty ──────────────────────────────────────────

def predict_with_uncertainty(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    n_passes: int = 20,
    uncertainty_threshold: float = 0.08,
) -> dict:
    """
    Monte Carlo Dropout uncertainty estimation.

    Runs n_passes forward passes with dropout ACTIVE (model.train() mode).
    The variance across passes estimates epistemic uncertainty —
    how uncertain the model is about this specific input.

    High uncertainty = model has not seen enough examples like this
    during training = flag for expert review.

    Args:
        model:                 trained model with dropout layers
        img_tensor:            preprocessed [3, 224, 224] tensor
        n_passes:              number of stochastic forward passes (20 is standard)
        uncertainty_threshold: std threshold above which to flag for review

    Returns:
        dict with mean probs, std probs, uncertainty score, flag
    """
    model.train()   # activates dropout — key step
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(img_tensor)
            probs  = F.softmax(logits, dim=1)[0]
            all_probs.append(probs.cpu().numpy())

    model.eval()    # restore eval mode
    all_probs = np.array(all_probs)   # [n_passes, n_classes]

    mean_probs = all_probs.mean(0)    # [n_classes]
    std_probs  = all_probs.std(0)     # [n_classes]

    pred_idx    = mean_probs.argmax()
    uncertainty = float(std_probs.max())   # max std across classes

    return {
        "pred_idx":          int(pred_idx),
        "predicted_class":   CLASS_NAMES[pred_idx],
        "confidence":        round(float(mean_probs[pred_idx]), 4),
        "uncertainty":       round(uncertainty, 4),
        "flag_for_review":   uncertainty > uncertainty_threshold,
        "mean_probs":        {CLASS_NAMES[i]: round(float(mean_probs[i]), 4) for i in range(NUM_CLASSES)},
        "std_probs":         {CLASS_NAMES[i]: round(float(std_probs[i]),  4) for i in range(NUM_CLASSES)},
    }


# ── ViT attention extractor ───────────────────────────────────────────────────

class ViTAttentionExtractor:
    """
    Extracts real attention weights from torchvision ViT-B/16 by monkey-patching
    each encoder layer's MultiheadAttention to force need_weights=True.

    Returns attention rollout (mean CLS attention across heads projected to
    14x14 spatial map) and per-head entropy (focused vs diffuse).
    """

    def __init__(self, model: torch.nn.Module):
        self.model      = model
        self.attn_maps  = []
        self._patch_attention()

    def _patch_attention(self):
        for layer in self.model.encoder.layers:
            mha = layer.self_attention
            orig_forward = mha.forward

            def make_patched(orig, extractor):
                def patched_forward(query, key, value, **kwargs):
                    kwargs["need_weights"]         = True
                    kwargs["average_attn_weights"] = False
                    out, attn = orig(query, key, value, **kwargs)
                    if attn is not None:
                        extractor.attn_maps.append(attn.detach().cpu())
                    return out, attn
                return patched_forward

            mha.forward = make_patched(orig_forward, self)

    def extract(self, img_tensor: torch.Tensor) -> dict:
        self.attn_maps = []
        self.model.eval()

        with torch.no_grad():
            logits = self.model(img_tensor.unsqueeze(0).to(DEVICE))
            probs  = F.softmax(logits, dim=1)[0]

        pred_idx = probs.argmax().item()

        if not self.attn_maps:
            return {
                "pred_idx":     pred_idx,
                "rollout":      np.ones((14, 14)),
                "head_entropy": [1.0] * 12,
            }

        # Last layer: [1, n_heads, seq_len, seq_len]
        last_attn = self.attn_maps[-1][0]        # [n_heads, 197, 197]
        n_heads   = last_attn.shape[0]

        # CLS token (index 0) attention to all patch tokens (1:)
        cls_attn  = last_attn[:, 0, 1:].numpy()  # [n_heads, 196]

        # Attention rollout: mean across heads → 14×14
        rollout   = cls_attn.mean(0).reshape(14, 14)
        rollout   = (rollout - rollout.min()) / (rollout.max() + 1e-8)

        # Per-head entropy
        head_entropy = []
        for h in range(n_heads):
            attn_h  = cls_attn[h] + 1e-8
            attn_h  = attn_h / attn_h.sum()
            entropy = float(-np.sum(attn_h * np.log(attn_h)))
            head_entropy.append(round(entropy, 3))

        return {
            "pred_idx":     pred_idx,
            "rollout":      rollout,
            "head_entropy": head_entropy,
        }


def attention_to_base64(
    original_img: Image.Image,
    rollout: np.ndarray,
) -> str:
    """Overlays attention rollout on original image, returns base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orig_np  = np.array(original_img.resize((224, 224))) / 255.0
    attn_img = Image.fromarray((rollout * 255).astype(np.uint8))
    attn_224 = np.array(attn_img.resize((224, 224), Image.BILINEAR)) / 255.0
    cmap     = plt.cm.viridis
    overlay  = cmap(attn_224)[:, :, :3]
    blended  = np.clip(0.5 * orig_np + 0.5 * overlay, 0, 1)
    blended  = (blended * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Response models ───────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    predicted_class:    str
    predicted_label:    int
    description:        str
    confidence:         float
    inference_ms:       float
    model_used:         str
    condition:          str
    all_probabilities:  dict


class GradCAMResponse(BaseModel):
    prediction:         PredictionResponse
    gradcam_base64:     str
    original_base64:    str


class UncertaintyResponse(BaseModel):
    predicted_class:    str
    description:        str
    confidence:         float
    uncertainty:        float          # std across MC passes — higher = less certain
    flag_for_review:    bool           # True if uncertainty > threshold
    all_mean_probs:     dict           # mean probability per class
    all_std_probs:      dict           # std probability per class (uncertainty per class)
    n_passes:           int
    inference_ms:       float
    model_used:         str
    condition:          str


class ViTAttentionResponse(BaseModel):
    predicted_class:    str
    confidence:         float
    attention_base64:   str            # attention rollout heatmap as base64 PNG
    head_entropy:       list           # entropy per attention head (12 heads)
    inference_ms:       float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "device":  str(DEVICE),
        "models_available": len(get_available_models()),
    }


@app.get("/models")
def list_models():
    available = get_available_models()
    best      = get_best_model_info()
    return {
        "available": available,
        "best_model": best,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file:      UploadFile = File(...),
    model_name: str = "cnn",
    condition:  str = "baseline",
):
    """
    Classify a histology patch.

    Args:
        file:       PNG/JPG image upload (224×224 recommended)
        model_name: cnn | vit | hybrid
        condition:  baseline | synthetic
    """
    if model_name not in ["cnn", "vit", "hybrid"]:
        raise HTTPException(400, "model_name must be cnn, vit, or hybrid")
    if condition not in ["baseline", "synthetic"]:
        raise HTTPException(400, "condition must be baseline or synthetic")

    # Read image
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    # Preprocess
    img_tensor = PREPROCESS(img).to(DEVICE)

    # Inference
    model = load_model(model_name, condition)
    t0    = time.time()
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0))
        probs  = F.softmax(logits, dim=1)[0]
    inference_ms = (time.time() - t0) * 1000

    pred_idx    = probs.argmax().item()
    pred_class  = CLASS_NAMES[pred_idx]
    confidence  = round(probs[pred_idx].item(), 4)
    all_probs   = {
        CLASS_NAMES[i]: round(probs[i].item(), 4)
        for i in range(NUM_CLASSES)
    }

    return PredictionResponse(
        predicted_class   = pred_class,
        predicted_label   = pred_idx,
        description       = CLASS_DESCRIPTIONS.get(pred_class, ""),
        confidence        = confidence,
        inference_ms      = round(inference_ms, 1),
        model_used        = model_name,
        condition         = condition,
        all_probabilities = all_probs,
    )


@app.post("/predict/gradcam", response_model=GradCAMResponse)
async def predict_gradcam(
    file:       UploadFile = File(...),
    model_name: str = "cnn",
    condition:  str = "baseline",
):
    """
    Classify a patch and return GradCAM heatmap overlay as base64 PNG.
    """
    if model_name not in ["cnn", "vit", "hybrid"]:
        raise HTTPException(400, "model_name must be cnn, vit, or hybrid")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    img_tensor = PREPROCESS(img).to(DEVICE)
    model      = load_model(model_name, condition)

    # Forward pass for prediction
    t0 = time.time()
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0))
        probs  = F.softmax(logits, dim=1)[0]
    inference_ms = (time.time() - t0) * 1000

    pred_idx   = probs.argmax().item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = round(probs[pred_idx].item(), 4)
    all_probs  = {
        CLASS_NAMES[i]: round(probs[i].item(), 4)
        for i in range(NUM_CLASSES)
    }

    # GradCAM — needs gradient so separate forward pass
    extractor = GradCAMExtractor(model, model_name)
    heatmap   = extractor.compute(img_tensor.clone(), pred_idx)

    # Encode images as base64
    gradcam_b64  = heatmap_to_base64(img, heatmap)
    orig_buf     = io.BytesIO()
    img.resize((224, 224)).save(orig_buf, format="PNG")
    original_b64 = base64.b64encode(orig_buf.getvalue()).decode("utf-8")

    prediction = PredictionResponse(
        predicted_class   = pred_class,
        predicted_label   = pred_idx,
        description       = CLASS_DESCRIPTIONS.get(pred_class, ""),
        confidence        = confidence,
        inference_ms      = round(inference_ms, 1),
        model_used        = model_name,
        condition         = condition,
        all_probabilities = all_probs,
    )

    return GradCAMResponse(
        prediction      = prediction,
        gradcam_base64  = gradcam_b64,
        original_base64 = original_b64,
    )


# ── Uncertainty endpoint ──────────────────────────────────────────────────────

@app.post("/predict/uncertain", response_model=UncertaintyResponse)
async def predict_uncertain(
    file:                  UploadFile = File(...),
    model_name:            str = "cnn",
    condition:             str = "baseline",
    n_passes:              int = 20,
    uncertainty_threshold: float = 0.08,
):
    """
    Monte Carlo Dropout uncertainty estimation.
    Runs n_passes stochastic forward passes with dropout active.
    flag_for_review=True means a pathologist should inspect this patch.
    """
    if model_name not in ["cnn", "vit", "hybrid"]:
        raise HTTPException(400, "model_name must be cnn, vit, or hybrid")
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")
    img_tensor   = PREPROCESS(img)
    model        = load_model(model_name, condition)
    t0           = time.time()
    results      = predict_with_uncertainty(model, img_tensor, n_passes=n_passes,
                       uncertainty_threshold=uncertainty_threshold)
    inference_ms = (time.time() - t0) * 1000
    return UncertaintyResponse(
        predicted_class = results["predicted_class"],
        description     = CLASS_DESCRIPTIONS.get(results["predicted_class"], ""),
        confidence      = results["confidence"],
        uncertainty     = results["uncertainty"],
        flag_for_review = results["flag_for_review"],
        all_mean_probs  = results["mean_probs"],
        all_std_probs   = results["std_probs"],
        n_passes        = n_passes,
        inference_ms    = round(inference_ms, 1),
        model_used      = model_name,
        condition       = condition,
    )


# ── ViT attention endpoint ────────────────────────────────────────────────────

@app.post("/predict/attention", response_model=ViTAttentionResponse)
async def predict_attention(
    file:      UploadFile = File(...),
    condition: str = "baseline",
):
    """
    ViT-only: classify + return attention rollout heatmap and head entropy scores.
    head_entropy: 12 values — low = focused head, high = diffuse head.
    """
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")
    img_tensor   = PREPROCESS(img)
    model        = load_model("vit", condition)
    t0           = time.time()
    extractor    = ViTAttentionExtractor(model)
    attn_result  = extractor.extract(img_tensor)
    inference_ms = (time.time() - t0) * 1000
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(DEVICE))
        probs  = F.softmax(logits, dim=1)[0]
    pred_idx   = attn_result["pred_idx"]
    confidence = round(probs[pred_idx].item(), 4)
    attn_b64   = attention_to_base64(img, attn_result["rollout"])
    return ViTAttentionResponse(
        predicted_class  = CLASS_NAMES[pred_idx],
        confidence       = confidence,
        attention_base64 = attn_b64,
        head_entropy     = attn_result["head_entropy"],
        inference_ms     = round(inference_ms, 1),
    )


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)