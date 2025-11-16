"""
AI Laundry Sorter – Streamlit Demo (fixed)
- Compatible with training checkpoint (head_wash_cycle)
- No numeric labels in UI
- Full descriptive wash program names
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

# --------------------------------------------------
# 0) Paths
# --------------------------------------------------
WASH_ROOT   = "/content/drive/MyDrive/wash_ai_project"
LABELS_CSV  = os.path.join(WASH_ROOT, "wash_labels.csv")
CKPT_PATH   = os.path.join(WASH_ROOT, "outputs_multitask_wash", "best_model_wash.pt")
HEADER_IMG  = os.path.join(WASH_ROOT, "ai.jpg")
DEMO_LOG    = os.path.join(WASH_ROOT, "demo_usage_log.csv")

# --------------------------------------------------
# 1) Label metadata
# --------------------------------------------------
df_all = pd.read_csv(LABELS_CSV)

df_all["color_label"]      = df_all["color_label"].astype(int)
df_all["fabric_label"]     = df_all["fabric_label"].astype(int)
df_all["wash_cycle_label"] = df_all["wash_cycle_label"].astype(int)

num_color_classes  = df_all["color_label"].nunique()
num_fabric_classes = df_all["fabric_label"].nunique()
num_wash_classes   = df_all["wash_cycle_label"].nunique()

color_map  = dict(zip(df_all["color_label"],  df_all["color_group"]))
fabric_map = dict(zip(df_all["fabric_label"], df_all["fabric_group"]))
wash_map   = dict(zip(df_all["wash_cycle_label"], df_all["wash_cycle"]))

# Full descriptive wash-cycle names
wash_full_description = {
    "delicate": "Delicate – Gentle wash, cold water, low spin (ideal for silk, lace, sensitive fabrics)",
    "normal":   "Normal – Standard wash, 40°C warm water, medium spin (suitable for cotton & daily wear)",
    "heavy":    "Heavy – Deep clean, 60°C hot water, high spin (good for jeans, towels, sportswear)",
    "quick":    "Quick – Rapid wash, 30°C cool water, medium spin (for lightly-soiled clothes)",
    "wool":     "Wool – Special wool cycle, 30°C cold, ultra-low spin (prevents shrinkage)",
}

# --------------------------------------------------
# 2) Model + transforms
# --------------------------------------------------
IMG_SIZE = 256
demo_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

BACKBONE_NAME = "convnext_tiny"


class WashMultiTaskConvNeXt(nn.Module):
    """
    MUST match the architecture used during training.
    Note: the third head is called `head_wash_cycle`
    because the checkpoint was saved with this name.
    """
    def __init__(self, num_color, num_fabric, num_wash):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features

        self.head_color      = nn.Linear(feat_dim, num_color)
        self.head_fabric     = nn.Linear(feat_dim, num_fabric)
        self.head_wash_cycle = nn.Linear(feat_dim, num_wash)  # <-- IMPORTANT

    def forward(self, x):
        feat = self.backbone(x)
        logits_color      = self.head_color(feat)
        logits_fabric     = self.head_fabric(feat)
        logits_wash_cycle = self.head_wash_cycle(feat)
        return logits_color, logits_fabric, logits_wash_cycle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WashMultiTaskConvNeXt(
    num_color=num_color_classes,
    num_fabric=num_fabric_classes,
    num_wash=num_wash_classes,
).to(device)

state_dict = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state_dict)   # now keys match
model.eval()

# --------------------------------------------------
# 3) Prediction helper
# --------------------------------------------------
def predict_single_image(pil_img: Image.Image):
    x = demo_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_c, logits_f, logits_w = model(x)
        pc = logits_c.argmax(1).item()
        pf = logits_f.argmax(1).item()
        pw = logits_w.argmax(1).item()

    wash_key = wash_map[pw]
    full_wash_text = wash_full_description.get(wash_key, wash_key)

    return {
        "color":  color_map[pc],
        "fabric": fabric_map[pf],
        "wash":   full_wash_text,
    }

# --------------------------------------------------
# 4) Streamlit app
# --------------------------------------------------
def main():
    st.set_page_config(
        page_title="AI Laundry Sorter",
        page_icon="🧺",
        layout="centered",
    )

    if os.path.exists(HEADER_IMG):
        st.image(HEADER_IMG, use_container_width=True)

    st.title("AI Laundry Sorter")
    st.caption("Deep Learning–powered automatic laundry wash-program recommender")

    uploaded_file = st.file_uploader(
        "Upload a clothing image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file:
        pil_img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(pil_img, use_container_width=True)

        result = predict_single_image(pil_img)

        with col2:
            st.subheader("Recommended Settings")
            st.markdown(f"**Color Group:** {result['color']}")
            st.markdown(f"**Fabric Group:** {result['fabric']}")
            st.markdown(f"**Wash Program:** {result['wash']}")

        # Log usage
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        row = pd.DataFrame([{
            "timestamp": ts,
            "image": uploaded_file.name,
            "color": result["color"],
            "fabric": result["fabric"],
            "wash": result["wash"],
        }])
        if os.path.exists(DEMO_LOG):
            old = pd.read_csv(DEMO_LOG)
            pd.concat([old, row], ignore_index=True).to_csv(DEMO_LOG, index=False)
        else:
            row.to_csv(DEMO_LOG, index=False)

        st.success("Prediction logged!")
    else:
        st.info("Upload a garment image to receive washing instructions.")

if __name__ == "__main__":
    main()
