import os
import io
import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import timm
import gdown

import streamlit as st

# --------------------------------------------------
# 0) Paths and Google Drive model download
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LABELS_CSV = os.path.join(BASE_DIR, "wash_labels.csv")
HEADER_IMG = os.path.join(BASE_DIR, "ai.jpg")
MODEL_PATH = os.path.join(BASE_DIR, "best_model_wash.pt")

# Google Drive ID for best_model_wash.pt
GDRIVE_FILE_ID = "1OQdbqnSoMVL-t-RcQ015l52bgHFyRM5H"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

def ensure_model_exists():
    """Download best_model_wash.pt from Google Drive if it does not exist."""
    if os.path.exists(MODEL_PATH):
        return
    st.write("Downloading model weights from Google Drive… This is done once.")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# --------------------------------------------------
# 1) Load label metadata from wash_labels.csv
# --------------------------------------------------
if not os.path.exists(LABELS_CSV):
    raise FileNotFoundError(f"wash_labels.csv not found at: {LABELS_CSV}")

df_all = pd.read_csv(LABELS_CSV)

df_all["color_label"]      = df_all["color_label"].astype(int)
df_all["fabric_label"]     = df_all["fabric_label"].astype(int)
df_all["wash_cycle_label"] = df_all["wash_cycle_label"].astype(int)

num_color_classes  = df_all["color_label"].nunique()
num_fabric_classes = df_all["fabric_label"].nunique()
num_wash_classes   = df_all["wash_cycle_label"].nunique()

# Basic label → name maps
color_map_df  = df_all[["color_label", "color_group"]].drop_duplicates()
fabric_map_df = df_all[["fabric_label", "fabric_group"]].drop_duplicates()
wash_map_df   = df_all[["wash_cycle_label", "wash_cycle"]].drop_duplicates()

idx2color  = dict(zip(color_map_df["color_label"],  color_map_df["color_group"]))
idx2fabric = dict(zip(fabric_map_df["fabric_label"], fabric_map_df["fabric_group"]))

# For wash-cycle, build a richer description using temperature / spin / notes
if {"wash_temp_c", "spin_rpm", "wash_notes"}.issubset(df_all.columns):
    wash_meta_df = (
        df_all[["wash_cycle_label", "wash_cycle", "wash_temp_c", "spin_rpm", "wash_notes"]]
        .drop_duplicates(subset=["wash_cycle_label"])
        .reset_index(drop=True)
    )
    idx2wash_name = dict(zip(wash_meta_df["wash_cycle_label"], wash_meta_df["wash_cycle"]))

    idx2wash_desc = {}
    for _, row in wash_meta_df.iterrows():
        lbl = int(row["wash_cycle_label"])
        name = str(row["wash_cycle"])
        temp = row["wash_temp_c"]
        rpm  = row["spin_rpm"]
        notes = str(row["wash_notes"])
        desc = f"{name} — {temp:.0f}°C, {rpm:.0f} rpm. {notes}"
        idx2wash_desc[lbl] = desc
else:
    # Fallback: only name
    wash_map_df = df_all[["wash_cycle_label", "wash_cycle"]].drop_duplicates()
    idx2wash_name = dict(zip(wash_map_df["wash_cycle_label"], wash_map_df["wash_cycle"]))
    idx2wash_desc = {k: v for k, v in idx2wash_name.items()}

# --------------------------------------------------
# 2) Model definition + loading
# --------------------------------------------------
IMG_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

demo_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

BACKBONE_NAME = "convnext_tiny"

class WashMultiTaskConvNeXt(nn.Module):
    def __init__(self,
                 backbone_name=BACKBONE_NAME,
                 num_color=4,
                 num_fabric=5,
                 num_wash=5):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.head_color      = nn.Linear(feat_dim, num_color)
        self.head_fabric     = nn.Linear(feat_dim, num_fabric)
        self.head_wash_cycle = nn.Linear(feat_dim, num_wash)

    def forward(self, x):
        feat = self.backbone(x)
        logits_color      = self.head_color(feat)
        logits_fabric     = self.head_fabric(feat)
        logits_wash_cycle = self.head_wash_cycle(feat)
        return logits_color, logits_fabric, logits_wash_cycle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make sure model file is present
ensure_model_exists()

model = WashMultiTaskConvNeXt(
    backbone_name=BACKBONE_NAME,
    num_color=num_color_classes,
    num_fabric=num_fabric_classes,
    num_wash=num_wash_classes,
).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# --------------------------------------------------
# 3) Helper: single image prediction
# --------------------------------------------------
def predict_single_image(pil_img: Image.Image):
    x = demo_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_c, logits_f, logits_w = model(x)
        pc = logits_c.argmax(dim=1).item()
        pf = logits_f.argmax(dim=1).item()
        pw = logits_w.argmax(dim=1).item()

    return {
        "color_label":  pc,
        "fabric_label": pf,
        "wash_label":   pw,
        "color_name":   idx2color.get(pc,  f"color_{pc}"),
        "fabric_name":  idx2fabric.get(pf, f"fabric_{pf}"),
        "wash_name":    idx2wash_name.get(pw, f"wash_{pw}"),
        "wash_desc":    idx2wash_desc.get(pw,  f"Program {pw}"),
    }

# Optional log file inside the app folder
DEMO_LOG = os.path.join(BASE_DIR, "demo_usage_log.csv")

def log_demo_call(image_name: str, result: dict):
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    row = {
        "timestamp": ts,
        "image_name": image_name,
        "pred_color_label":  result["color_label"],
        "pred_color_group":  result["color_name"],
        "pred_fabric_label": result["fabric_label"],
        "pred_fabric_group": result["fabric_name"],
        "pred_wash_label":   result["wash_label"],
        "pred_wash_cycle":   result["wash_name"],
    }
    if os.path.exists(DEMO_LOG):
        log_df = pd.read_csv(DEMO_LOG)
        log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
    else:
        log_df = pd.DataFrame([row])
    log_df.to_csv(DEMO_LOG, index=False)

# --------------------------------------------------
# 4) Streamlit UI
# --------------------------------------------------
def main():
    st.set_page_config(
        page_title="AI Laundry Sorter",
        page_icon="🧺",
        layout="centered",
    )

    # Header image (project design)
    if os.path.exists(HEADER_IMG):
        st.image(HEADER_IMG, use_container_width=True)

    # Title and subtitle
    st.title("AI Laundry Sorter")
    st.caption("Multi-task ConvNeXt model for automatic laundry sorting and wash program recommendation.")

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a clothing image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Input garment")
            st.image(pil_img, use_container_width=True)

        # Run prediction
        result = predict_single_image(pil_img)

        with col2:
            st.subheader("Predicted laundry settings")

            st.write(f"**Color group:** {result['color_name']}")
            st.write(f"**Fabric group:** {result['fabric_name']}")

            st.markdown("**Recommended wash program:**")
            st.success(result["wash_desc"])

            st.info(
                "These settings are generated by a ConvNeXt-based multi-task deep learning model "
                "trained on clothing images, jointly predicting color, fabric, and washing program."
            )

        # Log usage (optional, stays in app folder)
        log_demo_call(uploaded_file.name, result)
        st.caption("Prediction logged to demo_usage_log.csv (local to this app).")

    else:
        st.info("Please upload an image of a single garment to receive an AI-generated washing recommendation.")

if __name__ == "__main__":
    main()
