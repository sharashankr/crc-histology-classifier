[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/Sharashankr/crc-histology-classifier)

# CRC Histology Classifier

End-to-end deep learning pipeline for colorectal cancer tissue classification on the [NCT-CRC-HE](https://huggingface.co/datasets/owkin/nct-crc-he) dataset, with Stable Diffusion img2img synthetic augmentation for minority classes and a production web app with GradCAM explainability and Monte Carlo Dropout uncertainty quantification.

## Results

| Model | Condition | Test Acc | Macro F1 | STR Acc | DEB Acc | STR Delta |
|-------|-----------|----------|----------|---------|---------|-----------|
| ResNet-50 | baseline | 98.6% | 98.5% | 92.1% | 98.0% | — |
| ResNet-50 | +synthetic | **98.9%** | **98.7%** | **95.2%** | 98.0% | **+3.1%** |
| ViT-B/16 | baseline | 98.8% | 98.5% | 95.2% | 98.0% | — |
| ViT-B/16 | +synthetic | 98.9% | 98.6% | 95.2% | 98.0% | 0.0% |
| ConvNeXt | baseline | 98.6% | 98.4% | 92.1% | 98.0% | — |
| ConvNeXt | +synthetic | 98.1% | 97.5% | 90.5% | 94.1% | -1.6% |

**Key finding:** Diffusion augmentation is architecture-dependent. It helps ResNet-50 (+3.1% STR), makes no difference for ViT-B/16 (already solved by global attention), and hurts ConvNeXt (distribution-sensitive hybrid design).

## Overview

This project benchmarks three vision architectures on 9-class H&E histology patch classification and investigates whether Stable Diffusion img2img augmentation improves accuracy on minority tissue classes (DEB, STR).

**Research questions:**
1. Which architecture best handles minority class imbalance on a small clinical dataset?
2. Does diffusion-based augmentation improve minority class accuracy, and is this effect architecture-dependent?

## Dataset

**NCT-CRC-HE** — 7,180 H&E stained colorectal cancer tissue patches (224x224 px, 0.5 MPP) from 50 patients. Macenko color-normalized. No patient overlap with the 100K training set.

| Class | Description | Train Samples |
|-------|-------------|---------------|
| TUM | Colorectal adenocarcinoma epithelium | 863 |
| STR | Cancer-associated stroma | 295 minority |
| LYM | Lymphocytes | 444 |
| MUS | Smooth muscle | 414 |
| NORM | Normal colon mucosa | 519 |
| ADI | Adipose tissue | 936 |
| MUC | Mucus | 724 |
| DEB | Debris / necrosis | 237 minority |
| BACK | Background | 593 |

## Project Structure

```
Image Analysis/
├── data/
│   ├── loader.py           # Dataset loading, stratified splits, DataLoaders
│   └── class_stats.py      # Class distribution analysis, H&E normalization stats
├── augmentation/
│   └── diffusion_aug.py    # Stable Diffusion img2img + CLIP quality filter
├── models/
│   ├── cnn.py              # ResNet-50 with custom head + two-stage training
│   ├── vit.py              # ViT-B/16 with custom head + two-stage training
│   └── hybrid.py           # ConvNeXt-Base with custom head + two-stage training
├── app/
│   ├── api.py              # FastAPI: /predict, /predict/gradcam, /predict/uncertain, /predict/attention
│   └── ui.py               # Streamlit web interface
├── train.py                # MLflow-instrumented two-stage training with ablation flags
├── evaluate.py             # Confusion matrices, GradCAM, FID scores, comparison table
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/sharashankr/crc-histology-classifier.git
cd crc-histology-classifier
pip install -r requirements.txt
```

## Usage

### 1. Analyse dataset
```bash
python -m data.class_stats
```

### 2. Generate synthetic augmentation
```bash
python -m augmentation.diffusion_aug
```
Generates 100 synthetic DEB + 100 synthetic STR patches using Stable Diffusion v1.5 img2img, filtered by CLIP similarity >= 0.22. ~35 min on Apple M4 MPS.

### 3. Train models

```bash
# Baseline (real data only)
python train.py --model cnn
python train.py --model vit
python train.py --model hybrid

# Ablation (real + synthetic)
python train.py --model cnn    --use_synthetic
python train.py --model vit    --use_synthetic
python train.py --model hybrid --use_synthetic
```

### 4. Track experiments
```bash
mlflow ui --backend-store-uri "file://$(pwd)/outputs/mlruns" --port 5000
```

### 5. Evaluate all models
```bash
python evaluate.py
```

### 6. Run the web app

```bash
# Terminal 1
uvicorn app.api:app --port 8000 --reload

# Terminal 2
streamlit run app/ui.py
```
Open http://localhost:8501

Or with Docker:
```bash
docker-compose up
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| GET /health | Liveness check |
| GET /models | List available checkpoints |
| POST /predict | Classify patch |
| POST /predict/gradcam | Classify + GradCAM heatmap |
| POST /predict/uncertain | MC Dropout uncertainty |
| POST /predict/attention | ViT attention rollout |

## Architecture Notes

**Why ViT outperforms CNN on STR:** Cancer-associated stroma is diagnosed by the overall tissue arrangement across the whole patch. ViT global self-attention captures this distributed signal; CNN local 3x3 filters cannot.

**Why synthetic augmentation hurts ConvNeXt:** FID scores of 423/467 confirm a domain gap. ConvNeXt LayerNorm and depthwise convolutions are sensitive to this shift; ResNet local filters are not.

## Hardware

Apple M4 MacBook Pro (MPS backend). Training times: ResNet-50 ~30 min, ViT-B/16 ~45 min, ConvNeXt ~35 min, synthetic generation ~35 min (one-time).

## References

- Kather et al. (2019) — NCT-CRC-HE-100K dataset
- Dosovitskiy et al. (2020) — Vision Transformer
- Liu et al. (2022) — ConvNeXt
- Rombach et al. (2022) — Stable Diffusion
- Selvaraju et al. (2017) — GradCAM
- Gal & Ghahramani (2016) — MC Dropout
