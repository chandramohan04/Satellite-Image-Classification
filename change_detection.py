import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

model = resnet18(weights=ResNet18_Weights.DEFAULT)

model.fc = nn.Identity()

model = model.to(device)
model.eval()

def extract_features(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        features = model(image)

    return features

old_features = extract_features("sample_images/old.jpg")

new_features = extract_features("sample_images/new.jpg")

similarity = F.cosine_similarity(
    old_features,
    new_features
)

score = similarity.item()

print("Cosine Similarity:", score)

threshold = 0.90

if score > threshold:
    print("No Significant Change")
else:
    print("Change Detected")

    import os
    os.makedirs("outputs", exist_ok=True)

    old = cv2.imread("sample_images/old.jpg")
    new = cv2.imread("sample_images/new.jpg")

    old = cv2.resize(old, (224,224))
    new = cv2.resize(new, (224,224))

    difference = cv2.absdiff(old, new)

    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(old, cv2.COLOR_BGR2RGB))
    plt.title("Old Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(cv2.cvtColor(new, cv2.COLOR_BGR2RGB))
    plt.title("New Image")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.title("Change Heatmap")
    plt.axis("off")

    plt.savefig("outputs/change_detection_plot.png")
    print("Saved plot to outputs/change_detection_plot.png")
    # plt.show()

