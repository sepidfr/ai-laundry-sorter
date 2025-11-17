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
# Root = folder where app.py lives (works both locally and on Streamlit Cloud)
WASH_ROOT = os.path.dirname(os.path.abspath(__file__))

LABELS_CSV = os.path.join(WASH_ROOT, "wash_labels.csv")
HEADER_IMG = os.path.join(WASH_ROOT, "ai.jpg")
MODEL_PATH = os.path.join(WASH_ROOT, "best_model_wash.pt")
DEMO_LOG   = os.path.join(WASH_ROOT, "demo_usage_log.csv")

# Google Drive model file (your link)
MODEL_FILE_ID = "1OQdbqnSoMVL-t-RcQ015l52bgHFyRM5H"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

# Download model from Drive only if it is not present locally
if not os.path.exists(MODEL_PATH):
    os.makedirs(WASH_ROOT, exist_ok=True)
    st.write("Downloading model weights from Google Drive... (first run only)")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# --------------------------------------------------
# 1) Load label metadata from wash_labels.csv
# --------------------------------------------------
df_all = pd.read_csv(LABELS_CSV)

df_all["color_label"]      = df_all["color_label"].astype(int)
df_all["fabric_label"]     = df_all["fabric_label"].astype(int)
df_all["wash_cycle_label"] = df_all["wash_cycle_label"].astype(int)

num_color_classes  = df_all["color_label"].nunique()
num_fabric_classes = df_all["fabric_label"].nunique()
num_wash_classes   = df_all["wash_cycle_label"].nunique()

color_map_df  = df_all[["color_label", "color_group"]].drop_duplicates()
fabric_map_df = df_all[["fabric_label", "fabric_group"]].drop_duplicates()
wash_map_df   = df_all[["wash_cycle_label", "wash_cycle"]].drop_duplicates()

idx2color  = dict(zip(color_map_df["color_label"],  color_map_df["color_group"]))
idx2fabric = dict(zip(fabric_map_df["fabric_label"], fabric_map_df["fabric_group"]))
idx2wash   = dict(zip(wash_map_df["wash_cycle_label"], wash_map_df["wash_cycle"]))

# Optional: if some wash_cycle text is very short, you can expand here manually
WASH_PROGRAM_OVERRIDE = {
    # example:
    # "normal": "Normal — Standard wash, 40°C warm water, medium spin (suitable for cotton & daily wear)"
}
def verbose_wash_text(raw_name: str) -> str:
    """Return a nicely formatted, human-readable wash program description."""
    return WASH_PROGRAM_OVERRIDE.get(raw_name, raw_name)

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
                 backbone_name: str = BACKBONE_NAME,
                 num_color: int = 4,
                 num_fabric: int = 5,
                 num_wash: int = 5):
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
# 3) Prediction + logging helpers
# --------------------------------------------------
def predict_single_image(pil_img: Image.Image):
    """Run the multi-task model on one garment image."""
    x = demo_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_c, logits_f, logits_w = model(x)
        pc = logits_c.argmax(dim=1).item()
        pf = logits_f.argmax(dim=1).item()
        pw = logits_w.argmax(dim=1).item()

    color_name  = idx2color.get(pc,  f"color_{pc}")
    fabric_name = idx2fabric.get(pf, f"fabric_{pf}")
    wash_name   = idx2wash.get(pw,  f"wash_{pw}")
    wash_text   = verbose_wash_text(wash_name)

    return {
        "color_label":  pc,
        "fabric_label": pf,
        "wash_label":   pw,
        "color_name":   color_name,
        "fabric_name":  fabric_name,
        "wash_name":    wash_text,
    }

def log_demo_call(image_name: str, result: dict):
    """Append each demo call to demo_usage_log.csv (for your own analysis)."""
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

    # Project header image
    if os.path.exists(HEADER_IMG):
        st.image(HEADER_IMG, use_container_width=True)

    # Title and subtitle
    st.title("AI Laundry Sorter")
    st.caption("Deep Learning-powered automatic laundry wash-program recommender")

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
            st.subheader("Input")
            st.image(pil_img, use_container_width=True)

        # Run prediction
        result = predict_single_image(pil_img)

        with col2:
            st.subheader("Recommended Settings")
            st.markdown(f"**Color Group:** {result['color_name']}")
            st.markdown(f"**Fabric Group:** {result['fabric_name']}")
            st.markdown(f"**Wash Program:** {result['wash_name']}")

            st.info(
                "Use these settings on your smart washing machine. "
                "They are generated by a ConvNeXt-based multi-task deep learning model "
                "trained on labeled clothing images."
            )

        # Log the call (optional analytics for you)
        try:
            log_demo_call(uploaded_file.name, result)
            st.success("Prediction logged to demo_usage_log.csv")
        except Exception as e:
            st.warning(f"Could not log this demo call: {e}")

    else:
        st.info("Please upload an image of a single garment to receive a wash recommendation.")

if __name__ == "__main__":
    main()
