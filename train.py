import os
import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision
import numpy as np

from torchvision import datasets
from torchvision import transforms

from torch import nn
from torch import optim

from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train Satellite Land-Use Classifier")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--quick", action="store_true", help="Train on a small subset for testing")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
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
classes = dataset.classes

if args.quick:
    print("Quick mode enabled: Using a subset of 200 random images.")
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=g)[:200]
    dataset = torch.utils.data.Subset(dataset, indices)
    # Monkeypatch classes to Subset
    dataset.classes = classes

print(classes)
print("Total Images:", len(dataset))

train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=generator
)

print("Training Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))
print("Testing Images:", len(test_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False
)

print("DataLoaders Created Successfully!")


model = resnet18(weights=ResNet18_Weights.DEFAULT)

print("ResNet18 Loaded Successfully!")

num_classes = len(classes)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(device)

print("Model Modified for", num_classes, "Classes")

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

print("Optimizer Ready!")

epochs = args.epochs

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"Loss: {running_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print("-" * 40)

    torch.save(model.state_dict(), "saved_model.pth")

print("Model Saved Successfully!")



