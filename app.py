import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
from PIL import Image
import cv2
import numpy as np

import os

def load_image(file_or_path):
    is_tiff = False
    if isinstance(file_or_path, str):
        if file_or_path.lower().endswith(('.tif', '.tiff')):
            is_tiff = True
    elif hasattr(file_or_path, 'name'):
        if file_or_path.name.lower().endswith(('.tif', '.tiff')):
            is_tiff = True
            
    if is_tiff:
        import tifffile
        try:
            img = tifffile.imread(file_or_path)
            if len(img.shape) == 3 and img.shape[2] == 13:
                r = img[:, :, 3]
                g = img[:, :, 2]
                b = img[:, :, 1]
                rgb = np.stack([r, g, b], axis=-1)
                rgb = np.clip(rgb / 4000.0, 0, 1) * 255
                return Image.fromarray(rgb.astype(np.uint8))
            elif len(img.shape) == 3 and img.shape[2] == 3:
                return Image.fromarray(img)
            else:
                return Image.open(file_or_path).convert("RGB")
        except Exception:
            return Image.open(file_or_path).convert("RGB")
    else:
        return Image.open(file_or_path).convert("RGB")

st.set_page_config(
    page_title="Satellite Land-Use Classification",
    layout="wide"
)

st.title("🛰️ Satellite Land-Use Classification & Change Detection")

classes = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))

model_loaded = False
if os.path.exists("saved_model.pth"):
    try:
        model.load_state_dict(
            torch.load("saved_model.pth", map_location=device)
        )
        model = model.to(device)
        model.eval()
        model_loaded = True
    except Exception as e:
        st.sidebar.error(f"Error loading saved_model.pth: {e}")
else:
    st.sidebar.warning("saved_model.pth not found. Please train the classification model.")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

st.header("Land-Use Classification")

uploaded_image = st.file_uploader(
    "Upload a Satellite Image",
    type=["jpg", "png", "jpeg", "tif", "tiff"]
)

if uploaded_image is not None:
    if not model_loaded:
        st.error("Model file `saved_model.pth` not found. Please train the model first by running `python train.py`.")
    else:
        image = load_image(uploaded_image)

        st.image(image, caption="Uploaded Image", width=300)

        tensor = transform(image)
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():

            output = model(tensor)

            probabilities = torch.softmax(output, dim=1)

            confidence, prediction = torch.max(probabilities,1)

        st.success(
            f"Prediction : {classes[prediction.item()]}"
        )

        st.info(
            f"Confidence : {confidence.item()*100:.2f}%"
        )

st.header("Temporal Change Detection")

old_image = st.file_uploader(
    "Upload Old Image",
    type=["jpg", "png", "jpeg", "tif", "tiff"],
    key="old"
)

new_image = st.file_uploader(
    "Upload New Image",
    type=["jpg", "png", "jpeg", "tif", "tiff"],
    key="new"
)

feature_model = resnet18(weights=ResNet18_Weights.DEFAULT)
feature_model.fc = nn.Identity()

feature_model = feature_model.to(device)
feature_model.eval()

if old_image and new_image:

    old = load_image(old_image)
    new = load_image(new_image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(old, caption="Old Image")

    with col2:
        st.image(new, caption="New Image")

    old_tensor = transform(old).unsqueeze(0).to(device)
    new_tensor = transform(new).unsqueeze(0).to(device)

    with torch.no_grad():

        old_feature = feature_model(old_tensor)
        new_feature = feature_model(new_tensor)

        similarity = F.cosine_similarity(
            old_feature,
            new_feature
        )

    score = similarity.item()

    st.subheader(f"Similarity : {score:.4f}")

    threshold = 0.90

    if score > threshold:
        st.success("No Significant Change")
    else:
        st.error("Change Detected")

        old_np = np.array(old.resize((224,224)))
        new_np = np.array(new.resize((224,224)))

        difference = cv2.absdiff(old_np, new_np)

        gray = cv2.cvtColor(
            difference,
            cv2.COLOR_RGB2GRAY
        )

        heatmap = cv2.applyColorMap(
            gray,
            cv2.COLORMAP_JET
        )

        st.image(
            heatmap,
            caption="Change Heatmap"
        )


