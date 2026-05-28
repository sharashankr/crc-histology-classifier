"""
app/ui.py
---------
Streamlit web app for CRC histology patch classification.
Connects to the FastAPI backend running on localhost:8000.

Run:
    cd "Trial Calibrated Synthetic Data/Image Analysis"
    streamlit run app/ui.py
"""

import base64
import io
import json

import requests
import streamlit as st
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"

CLASS_COLORS = {
    "TUM":  "#E24B4A",
    "STR":  "#D85A30",
    "LYM":  "#378ADD",
    "MUS":  "#639922",
    "NORM": "#1D9E75",
    "ADI":  "#EF9F27",
    "MUC":  "#7F77DD",
    "DEB":  "#888780",
    "BACK": "#B4B2A9",
}

CLASS_DESCRIPTIONS = {
    "TUM":  "Colorectal adenocarcinoma epithelium",
    "STR":  "Cancer-associated stroma",
    "LYM":  "Lymphocytes",
    "MUS":  "Smooth muscle",
    "NORM": "Normal colon mucosa",
    "ADI":  "Adipose tissue",
    "MUC":  "Mucus",
    "DEB":  "Debris / necrosis",
    "BACK": "Background",
}

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CRC Histology Classifier",
    page_icon="🔬",
    layout="wide",
)

st.title("CRC Tissue Classifier")
st.caption("NCT-CRC-HE — 9-class colorectal cancer histology classification")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Model settings")

    # Check API health
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"API online — {health['models_available']} models loaded")
        api_ok = True
    except Exception:
        st.error("API offline. Run: uvicorn app.api:app --port 8000")
        api_ok = False

    model_name = st.selectbox(
        "Architecture",
        ["cnn", "vit", "hybrid"],
        format_func=lambda x: {
            "cnn":    "ResNet-50",
            "vit":    "ViT-B/16",
            "hybrid": "ConvNeXt-Base",
        }[x],
    )

    condition = st.selectbox(
        "Training condition",
        ["baseline", "synthetic"],
        format_func=lambda x: {
            "baseline":  "Baseline (real data only)",
            "synthetic": "+ Synthetic augmentation",
        }[x],
    )

    show_gradcam     = st.toggle("Show GradCAM heatmap", value=True)
    show_uncertainty = st.toggle("Show uncertainty (MC Dropout)", value=True)
    show_attention   = st.toggle("Show ViT attention map", value=model_name == "vit")
    n_mc_passes      = st.slider("MC dropout passes", 10, 50, 20, 5,
                                  help="More passes = better uncertainty estimate, slower inference")

    st.divider()
    st.caption("Model benchmark (test set)")

    # Load benchmark results if available
    try:
        with open("outputs/evaluation/evaluation_report.json") as f:
            report = json.load(f)
        results = report.get("results", {})
        key = f"{model_name}_{condition}"
        if key in results:
            r = results[key]
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{r['accuracy']*100:.1f}%")
            col2.metric("Macro F1", f"{r['macro_f1']*100:.1f}%")

            st.caption("Per-class accuracy")
            for cls, acc in r.get("per_class_acc", {}).items():
                color = CLASS_COLORS.get(cls, "#888")
                flag  = " ←" if cls in ["DEB", "STR"] else ""
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:12px;margin:2px 0">'
                    f'<span style="color:{color};font-weight:500">{cls}{flag}</span>'
                    f'<span>{acc*100:.1f}%</span></div>',
                    unsafe_allow_html=True
                )
    except Exception:
        st.caption("Run evaluate.py to see benchmark results here")

# ── Main panel ────────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload a histology patch",
    type=["png", "jpg", "jpeg"],
    help="224×224 px H&E stained tissue patch. Any size accepted — resized automatically.",
)

if uploaded and api_ok:
    img = Image.open(uploaded).convert("RGB")
    img_bytes = uploaded.getvalue()

    endpoint = "/predict/gradcam" if show_gradcam else "/predict"

    with st.spinner("Running inference..."):
        try:
            response = requests.post(
                f"{API_URL}{endpoint}",
                files={"file": (uploaded.name, img_bytes, "image/png")},
                params={"model_name": model_name, "condition": condition},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Start it with: uvicorn app.api:app --port 8000")
            st.stop()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # Extract prediction
    pred = data["prediction"] if show_gradcam else data
    predicted_class = pred["predicted_class"]
    confidence      = pred["confidence"]
    inference_ms    = pred["inference_ms"]
    all_probs       = pred["all_probabilities"]
    description     = pred["description"]

    # ── Results layout ───────────────────────────────────────────────────────
    if show_gradcam:
        col1, col2, col3 = st.columns(3)
    else:
        col1, col2 = st.columns([1, 2])

    # Original image
    with col1:
        st.subheader("Uploaded patch")
        st.image(img.resize((224, 224)), width=224)
        st.caption(f"224 × 224 px · H&E stained")

    # GradCAM
    if show_gradcam and "gradcam_base64" in data:
        with col2:
            st.subheader("GradCAM heatmap")
            gradcam_bytes = base64.b64decode(data["gradcam_base64"])
            gradcam_img   = Image.open(io.BytesIO(gradcam_bytes))
            st.image(gradcam_img, width=224)
            st.caption("Red = where the model focused most")

        result_col = col3
    else:
        result_col = col2

    # Prediction result
    with result_col:
        st.subheader("Prediction")
        color = CLASS_COLORS.get(predicted_class, "#888")

        st.markdown(
            f'<div style="background:{color}18;border:2px solid {color};'
            f'border-radius:12px;padding:16px;margin-bottom:12px">'
            f'<p style="font-size:32px;font-weight:700;color:{color};margin:0">'
            f'{predicted_class}</p>'
            f'<p style="font-size:14px;color:{color};margin:4px 0 0">'
            f'{description}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        col_a.metric("Confidence", f"{confidence*100:.1f}%")
        col_b.metric("Inference", f"{inference_ms:.0f} ms")

        st.caption("All class probabilities")
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_probs:
            color_bar = CLASS_COLORS.get(cls, "#888")
            is_pred   = cls == predicted_class
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
                f'<span style="font-size:12px;font-weight:{"600" if is_pred else "400"};'
                f'min-width:40px;color:{color_bar}">{cls}</span>'
                f'<div style="flex:1;background:#f0f0f0;border-radius:4px;height:12px">'
                f'<div style="width:{prob*100:.1f}%;background:{color_bar};'
                f'height:100%;border-radius:4px"></div></div>'
                f'<span style="font-size:12px;min-width:42px;text-align:right">'
                f'{prob*100:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Uncertainty panel ─────────────────────────────────────────────────────
    if show_uncertainty:
        st.divider()
        st.subheader("Uncertainty estimation (MC Dropout)")
        with st.spinner(f"Running {n_mc_passes} stochastic passes..."):
            try:
                unc_resp = requests.post(
                    f"{API_URL}/predict/uncertain",
                    files={"file": (uploaded.name, img_bytes, "image/png")},
                    params={"model_name": model_name, "condition": condition,
                            "n_passes": n_mc_passes, "uncertainty_threshold": 0.08},
                    timeout=60,
                )
                unc_resp.raise_for_status()
                unc_data = unc_resp.json()

                uc1, uc2, uc3 = st.columns(3)
                uc1.metric("Mean confidence", f"{unc_data['confidence']*100:.1f}%")
                uc2.metric("Uncertainty (std)", f"{unc_data['uncertainty']*100:.2f}%")
                uc3.metric("MC passes", unc_data["n_passes"])

                if unc_data["flag_for_review"]:
                    st.warning(
                        "⚠️ **High uncertainty** — this patch should be reviewed "
                        "by a pathologist. The model is not confident in its prediction.",
                        icon="⚠️"
                    )
                else:
                    st.success("✓ Low uncertainty — model is confident in this prediction.")

                st.caption("Per-class uncertainty (std across MC passes)")
                std_probs = unc_data["all_std_probs"]
                for cls, std in sorted(std_probs.items(), key=lambda x: x[1], reverse=True):
                    color = CLASS_COLORS.get(cls, "#888")
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
                        f'<span style="font-size:12px;min-width:40px;color:{color}">{cls}</span>'
                        f'<div style="flex:1;background:#f0f0f0;border-radius:4px;height:8px">'
                        f'<div style="width:{std*500:.1f}%;max-width:100%;background:{color};'
                        f'height:100%;border-radius:4px"></div></div>'
                        f'<span style="font-size:11px;min-width:40px;text-align:right">'
                        f'±{std*100:.2f}%</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Uncertainty estimation failed: {e}")

    # ── ViT attention panel ────────────────────────────────────────────────────
    if show_attention and model_name == "vit":
        st.divider()
        st.subheader("ViT attention map")
        with st.spinner("Extracting attention rollout..."):
            try:
                attn_resp = requests.post(
                    f"{API_URL}/predict/attention",
                    files={"file": (uploaded.name, img_bytes, "image/png")},
                    params={"condition": condition},
                    timeout=30,
                )
                attn_resp.raise_for_status()
                attn_data = attn_resp.json()

                acol1, acol2 = st.columns([1, 2])
                with acol1:
                    attn_bytes = base64.b64decode(attn_data["attention_base64"])
                    attn_img   = Image.open(io.BytesIO(attn_bytes))
                    st.image(attn_img, width=224,
                             caption="Attention rollout — where ViT looked")

                with acol2:
                    st.caption("Head entropy (12 attention heads)")
                    st.caption("Low = focused on specific regions · High = diffuse attention")
                    entropies = attn_data["head_entropy"]
                    for i, ent in enumerate(entropies):
                        max_ent  = max(entropies) + 1e-8
                        pct      = ent / max_ent * 100
                        focus    = "focused" if ent < 4.0 else "diffuse"
                        color    = "#1D9E75" if ent < 4.0 else "#B4B2A9"
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
                            f'<span style="font-size:11px;min-width:52px">Head {i+1:02d}</span>'
                            f'<div style="flex:1;background:#f0f0f0;border-radius:4px;height:8px">'
                            f'<div style="width:{pct:.0f}%;background:{color};'
                            f'height:100%;border-radius:4px"></div></div>'
                            f'<span style="font-size:11px;min-width:52px">{ent:.2f} ({focus})</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            except Exception as e:
                st.error(f"Attention extraction failed: {e}")
    elif show_attention and model_name != "vit":
        st.info("Attention maps are only available for ViT-B/16. Switch model to ViT to see them.")

    # ── Model info footer ─────────────────────────────────────────────────────
    st.divider()
    model_labels = {"cnn": "ResNet-50", "vit": "ViT-B/16", "hybrid": "ConvNeXt-Base"}
    st.caption(
        f"Model: {model_labels[model_name]} · "
        f"Condition: {condition} · "
        f"Device: {health.get('device', 'unknown')}"
    )

elif not api_ok:
    st.info("Start the API server first, then upload an image.")
else:
    # Landing state
    st.info("Upload a 224×224 H&E histology patch to classify it.")

    st.subheader("About this classifier")
    st.markdown("""
    This app classifies colorectal cancer tissue patches into 9 tissue types
    using deep learning models trained on the NCT-CRC-HE dataset.

    **Tissue classes:**
    """)

    cols = st.columns(3)
    for i, (cls, desc) in enumerate(CLASS_DESCRIPTIONS.items()):
        color = CLASS_COLORS[cls]
        cols[i % 3].markdown(
            f'<div style="padding:8px;margin:4px 0;border-left:3px solid {color}">'
            f'<strong style="color:{color}">{cls}</strong><br>'
            f'<span style="font-size:12px">{desc}</span></div>',
            unsafe_allow_html=True,
        )