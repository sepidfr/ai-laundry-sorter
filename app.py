import os
import io
import datetime

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import timm
import gdown

import streamlit as st


# ===============================================================
# 0) Paths & Google Drive model link
# ===============================================================
BASE_DIR   = os.path.dirname(__file__)
LABELS_CSV = os.path.join(BASE_DIR, "wash_labels.csv")
HEADER_IMG = os.path.join(BASE_DIR, "ai.jpg")
MODEL_PATH = os.path.join(BASE_DIR, "best_model_wash.pt")
DEMO_LOG   = os.path.join(BASE_DIR, "demo_usage_log.csv")

# Your shared model link:
GDRIVE_URL = "https://drive.google.com/uc?id=1OQdbqnSoMVL-t-RcQ015l52bgHFyRM5H"


# ===============================================================
# 1) Load label metadata
# ===============================================================
@st.cache_data(show_spinner=True)
def load_label_metadata():
    df = pd.read_csv(LABELS_CSV)

    df["color_label"]      = df["color_label"].astype(int)
    df["fabric_label"]     = df["fabric_label"].astype(int)
    df["wash_cycle_label"] = df["wash_cycle_label"].astype(int)

    idx2color  = dict(df[["color_label", "color_group"]].drop_duplicates().values)
    idx2fabric = dict(df[["fabric_label", "fabric_group"]].drop_duplicates().values)
    idx2wash   = dict(df[["wash_cycle_label", "wash_cycle"]].drop_duplicates().values)

    return (
        df["color_label"].nunique(),
        df["fabric_label"].nunique(),
        df["wash_cycle_label"].nunique(),
        idx2color,
        idx2fabric,
        idx2wash,
    )


(
    NUM_COLORS,
    NUM_FABRICS,
    NUM_WASH,
    idx2color,
    idx2fabric,
    idx2wash,
) = load_label_metadata()


# ===============================================================
# 2) Model definition
# ===============================================================
BACKBONE_NAME = "convnext_tiny"

class WashModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features

        self.head_color  = nn.Linear(feat_dim, NUM_COLORS)
        self.head_fabric = nn.Linear(feat_dim, NUM_FABRICS)
        self.head_wash   = nn.Linear(feat_dim, NUM_WASH)

    def forward(self, x):
        feat = self.backbone(x)
        return (
            self.head_color(feat),
            self.head_fabric(feat),
            self.head_wash(feat),
        )


# ===============================================================
# 3) Download model (1 time) + load it
# ===============================================================
@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading trained model from Google Drive…")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WashModel().to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, device


model, device = load_model()


# ===============================================================
# 4) Preprocessing transform
# ===============================================================
IMG_SIZE = 256

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ===============================================================
# 5) Wash program descriptions
# ===============================================================
def wash_description(name: str) -> str:
    key = str(name).lower()

    mapping = {
        "normal": "Normal — standard wash, 40°C warm water, medium spin. Suitable for cotton & daily wear.",
        "delicate": "Delicate — gentle wash for silk/viscose/light fabrics.",
        "quick": "Quick — short cycle for lightly used clothes.",
        "hand wash": "Hand Wash — very gentle, cold water, for wool or delicate knitwear.",
        "heavy": "Heavy — longer cycle for towels, jeans and bedding.",
        "cotton": "Cotton — strong wash for cotton fabrics.",
        "synthetics": "Synthetics — moderate temperature & spin for polyester blends.",
    }

    for k in mapping:
        if key.startswith(k):
            return mapping[k]

    return f"{name} — standard program."


# ===============================================================
# 6) Prediction
# ===============================================================
def predict(pil_img):
    x = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_c, logits_f, logits_w = model(x)
        pc = logits_c.argmax(1).item()
        pf = logits_f.argmax(1).item()
        pw = logits_w.argmax(1).item()

    return {
        "color": idx2color.get(pc, f"color_{pc}"),
        "fabric": idx2fabric.get(pf, f"fabric_{pf}"),
        "wash": idx2wash.get(pw, f"wash_{pw}"),
        "wash_long": wash_description(idx2wash.get(pw, f"wash_{pw}")),
    }


# ===============================================================
# 7) Logging
# ===============================================================
def log_event(img_name, res):
    row = {
        "timestamp": datetime.datetime.now().isoformat(),
        "image_name": img_name,
        "color": res["color"],
        "fabric": res["fabric"],
        "wash": res["wash"],
    }

    if os.path.exists(DEMO_LOG):
        df = pd.read_csv(DEMO_LOG)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(DEMO_LOG, index=False)


# ===============================================================
# 8) Streamlit UI
# ===============================================================
def main():
    st.set_page_config(
        page_title="AI Laundry Sorter",
        page_icon="🧺",
        layout="centered",
    )

    if os.path.exists(HEADER_IMG):
        st.image(HEADER_IMG, use_container_width=True)

    st.title("AI Laundry Sorter")
    st.caption("Deep Learning–powered laundry wash-program recommender")

    st.markdown("---")

    f = st.file_uploader("Upload a clothing image", ["jpg", "jpeg", "png"])

    if f:
        pil_img = Image.open(io.BytesIO(f.read())).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input")
            st.image(pil_img, use_container_width=True)

        res = predict(pil_img)

        with col2:
            st.subheader("Recommended Settings")
            st.write(f"**Color Group:** {res['color']}")
            st.write(f"**Fabric Group:** {res['fabric']}")
            st.write(f"**Wash Program:**")
            st.info(res["wash_long"])

        log_event(f.name, res)
        st.success("Saved to usage log.")
    else:
        st.info("Upload a garment image to begin.")


if __name__ == "__main__":
    main()
