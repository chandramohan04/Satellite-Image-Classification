import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn

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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

model = resnet18(weights=ResNet18_Weights.DEFAULT)

model.fc = nn.Linear(model.fc.in_features, len(classes))

model.load_state_dict(torch.load("saved_model.pth", map_location=device))

model = model.to(device)
model.eval()

print("Model Loaded Successfully!")

sample_path = "sample_images/test.jpg"
if not os.path.exists(sample_path):
    # fallback: pick the first image in sample_images if test.jpg is missing
    found = False
    for fname in os.listdir("sample_images"):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            sample_path = os.path.join("sample_images", fname)
            found = True
            break
    if not found:
        raise FileNotFoundError("No sample images found in sample_images/. Add at least one image.")

image = load_image(sample_path)
image = transform(image)
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

print("Predicted Class:", classes[predicted.item()])
print(f"Confidence: {confidence.item() * 100:.2f}%")