"""
ai-laundry-sorter – Streamlit Demo (2 models)

Model A: multi-task ConvNeXt for color group, fabric group, wash program
Model B: single-task ConvNeXt for garment type (garment_label)

Both models are downloaded from Google Drive and used together.
"""

import os
import io
import datetime

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import timm
import streamlit as st
import gdown


# --------------------------------------------------
# 0) Paths
# --------------------------------------------------
HERE = os.path.dirname(__file__)

# Existing labels for color / fabric / wash
LABELS_WASH_CSV = os.path.join(HERE, "wash_labels.csv")

# Labels that include garment_label / garment_type
LABELS_GARMENT_CSV = os.path.join(HERE, "wash_labels_with_garment.csv")

HEADER_IMG = os.path.join(HERE, "ai.jpg")
DEMO_LOG   = os.path.join(HERE, "demo_usage_log.csv")

# Existing multi-task wash model (already working)
CKPT_WASH   = os.path.join(HERE, "best_model_wash.pt")
MODEL_URL_WASH = "https://drive.google.com/uc?id=1Siu8S9OVwfu7v13M0ip8ucf4_CRuGUS9"

# NEW: garment-only model
CKPT_GARMENT   = os.path.join(HERE, "best_model_garment.pt")
# TODO: replace THIS ID with your garment model file ID
MODEL_URL_GARMENT = "https://drive.google.com/uc?id=YOUR_GARMENT_MODEL_ID_HERE"


def download_if_missing(url: str, dest_path: str, label: str):
    """Download a file from Google Drive if it does not exist."""
    if os.path.exists(dest_path):
        return
    st.write(f"Downloading {label} model from Google Drive (first run only)...")
    gdown.download(url, dest_path, quiet=False)


# --------------------------------------------------
# 1) Label metadata
# --------------------------------------------------
df_wash = pd.read_csv(LABELS_WASH_CSV)
df_wash["color_label"]      = df_wash["color_label"].astype(int)
df_wash["fabric_label"]     = df_wash["fabric_label"].astype(int)
df_wash["wash_cycle_label"] = df_wash["wash_cycle_label"].astype(int)

num_color_classes  = df_wash["color_label"].nunique()
num_fabric_classes = df_wash["fabric_label"].nunique()
num_wash_classes   = df_wash["wash_cycle_label"].nunique()

color_map  = dict(zip(df_wash["color_label"],  df_wash["color_group"]))
fabric_map = dict(zip(df_wash["fabric_label"], df_wash["fabric_group"]))
wash_map   = dict(zip(df_wash["wash_cycle_label"], df_wash["wash_cycle"]))

# Garment labels
df_garment = pd.read_csv(LABELS_GARMENT_CSV)
df_garment["garment_label"] = df_garment["garment_label"].astype(int)

num_garment_classes = df_garment["garment_label"].nunique()
garment_map = dict(zip(df_garment["garment_label"], df_garment["garment_type"]))

wash_description = {
    "delicate": "Delicate – gentle wash, cold water, low spin (silk, lace, sensitive fabrics).",
    "normal":   "Normal – standard wash, warm water, medium spin (daily cotton & mixed clothes).",
    "heavy":    "Heavy – deep-clean cycle, hot water, high spin (jeans, towels, sportswear).",
    "quick":    "Quick – rapid wash, cool water, medium spin (lightly soiled garments).",
    "wool":     "Wool – gentle wool cycle, cool water, very low spin (prevents shrinkage).",
}


# --------------------------------------------------
# 2) Shared transforms
# --------------------------------------------------
IMG_SIZE = 256
demo_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

BACKBONE = "convnext_tiny"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 3) Model definitions
# --------------------------------------------------
class WashMultiTaskConvNeXt(nn.Module):
    """Multi-task ConvNeXt for color, fabric, wash."""
    def __init__(self, n_color, n_fabric, n_wash):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.head_color      = nn.Linear(feat_dim, n_color)
        self.head_fabric     = nn.Linear(feat_dim, n_fabric)
        self.head_wash_cycle = nn.Linear(feat_dim, n_wash)

    def forward(self, x):
        feat = self.backbone(x)
        return (
            self.head_color(feat),
            self.head_fabric(feat),
            self.head_wash_cycle(feat),
        )


class GarmentConvNeXt(nn.Module):
    """Single-task ConvNeXt for garment type.

    This should match the architecture used when you trained `model`
    in the notebook that produced your confusion-matrix code.
    """
    def __init__(self, n_garment):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.head_garment = nn.Linear(feat_dim, n_garment)

    def forward(self, x):
        feat = self.backbone(x)
        logits_garment = self.head_garment(feat)
        return logits_garment


# --------------------------------------------------
# 4) Load models
# --------------------------------------------------
# download weight files if missing
download_if_missing(MODEL_URL_WASH,    CKPT_WASH,    "wash (color/fabric/wash)")
download_if_missing(MODEL_URL_GARMENT, CKPT_GARMENT, "garment-type")

# Multi-task wash model
wash_model = WashMultiTaskConvNeXt(
    n_color=num_color_classes,
    n_fabric=num_fabric_classes,
    n_wash=num_wash_classes,
).to(device)
state_wash = torch.load(CKPT_WASH, map_location=device)
wash_model.load_state_dict(state_wash)
wash_model.eval()

# Garment-only model
garment_model = GarmentConvNeXt(
    n_garment=num_garment_classes,
).to(device)
state_garment = torch.load(CKPT_GARMENT, map_location=device)
garment_model.load_state_dict(state_garment)
garment_model.eval()


# --------------------------------------------------
# 5) Prediction helpers
# --------------------------------------------------
def predict_all(pil_img: Image.Image):
    """Run both models on a single image and return all predictions."""
    x = demo_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # wash model
        logits_c, logits_f, logits_w = wash_model(x)
        c_idx = logits_c.argmax(1).item()
        f_idx = logits_f.argmax(1).item()
        w_idx = logits_w.argmax(1).item()

        # garment model
        logits_g = garment_model(x)
        g_idx = logits_g.argmax(1).item()

    wash_key  = wash_map[w_idx]
    wash_text = wash_description.get(wash_key, wash_key)

    return {
        "garment": garment_map[g_idx],
        "color":   color_map[c_idx],
        "fabric":  fabric_map[f_idx],
        "wash":    wash_text,
    }


# --------------------------------------------------
# 6) Streamlit UI
# --------------------------------------------------
def main():
    st.set_page_config(
        page_title="ai-laundry-sorter",
        page_icon="🧺",
        layout="centered",
    )

    if os.path.exists(HEADER_IMG):
        st.image(HEADER_IMG, use_container_width=True)

    st.title("ai-laundry-sorter")
    st.caption("Upload a garment image to get garment type and washing program.")

    uploaded = st.file_uploader(
        "Upload an image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded:
        pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(pil, use_column_width=True)

        result = predict_all(pil)

        with col2:
            st.subheader("Prediction")
            st.markdown(f"**Garment Type:** {result['garment']}")
            st.markdown(f"**Color Group:** {result['color']}")
            st.markdown(f"**Fabric Group:** {result['fabric']}")
            st.markdown(f"**Wash Program:** {result['wash']}")

        # optional logging
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        row = pd.DataFrame([{
            "timestamp": ts,
            "file": uploaded.name,
            "garment": result["garment"],
            "color": result["color"],
            "fabric": result["fabric"],
            "wash": result["wash"],
        }])
        try:
            if os.path.exists(DEMO_LOG):
                old = pd.read_csv(DEMO_LOG)
                pd.concat([old, row], ignore_index=True).to_csv(DEMO_LOG, index=False)
            else:
                row.to_csv(DEMO_LOG, index=False)
        except Exception:
            pass

    else:
        st.info("Upload a clothing image above to see garment type and washing suggestion.")


if __name__ == "__main__":
    main()
