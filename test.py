import torch
import numpy as np
import matplotlib.pyplot as plt
import os


from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from torch import nn

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

def tiff_rgb_loader(path):
    import tifffile
    from PIL import Image
    try:
        img = tifffile.imread(path)
        if len(img.shape) == 3 and img.shape[2] == 13:
            r = img[:, :, 3]
            g = img[:, :, 2]
            b = img[:, :, 1]
            rgb = np.stack([r, g, b], axis=-1)
            rgb = np.clip(rgb / 4000.0, 0, 1) * 255
            return Image.fromarray(rgb.astype(np.uint8))
        else:
            return Image.open(path).convert("RGB")
    except Exception:
        return Image.open(path).convert("RGB")

dataset = datasets.ImageFolder(
    root="dataset",
    loader=tiff_rgb_loader,
    transform=transform
)

print(dataset.classes)

train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=generator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

model = resnet18(weights=ResNet18_Weights.DEFAULT)

model.fc = nn.Linear(
    model.fc.in_features,
    len(dataset.classes)
)

model.load_state_dict(torch.load("saved_model.pth", map_location=device))

model = model.to(device)

model.eval()

print("Model Loaded Successfully!")

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs,1)

        all_labels.extend(labels.numpy())

        all_predictions.extend(predicted.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted"
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted"
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted"
)

print()

print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=dataset.classes
)

plt.figure(figsize=(10,10))

disp.plot(
    xticks_rotation=45,
    cmap="Blues"
)

plt.title("Confusion Matrix")

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/confusion_matrix.png")

# plt.show()



