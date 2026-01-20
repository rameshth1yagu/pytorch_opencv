"""
MobileNetV3-Small Fine-Tuning Pipeline
======================================
This script demonstrates how to fine-tune a lightweight model (MobileNetV3-Small)
on the Caltech-256 subset.

FIXES APPLIED:
1. Replaced .double() with .float() to prevent MPS (Mac) crashes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import zipfile
import requests
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Global Settings
plt.style.use('ggplot')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def setup_data(dataset_name='caltech256_subset'):
    """Downloads and extracts the dataset if not present."""
    url = 'https://www.dropbox.com/s/0ltu2bsja3sb2j4/caltech256_subset.zip?dl=1'
    file_name = f"{dataset_name}.zip"

    if not os.path.exists(dataset_name):
        print(f"Downloading {dataset_name}...")
        if not os.path.exists(file_name):
            r = requests.get(url)
            with open(file_name, 'wb') as f:
                f.write(r.content)

        print("Extracting...")
        with zipfile.ZipFile(file_name, 'r') as z:
            z.extractall()
        print("Done!")
    else:
        print(f"Dataset {dataset_name} already exists.")
    return dataset_name

def get_transforms():
    """Standard ImageNet transformations."""
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
    }

def create_dataloaders(data_dir, batch_size=32):
    """Creates DataLoaders for train, valid, and test."""
    transforms_map = get_transforms()
    dirs = {x: os.path.join(data_dir, x) for x in ['train', 'valid', 'test']}
    datasets_map = {x: datasets.ImageFolder(dirs[x], transform=transforms_map[x]) for x in ['train', 'valid', 'test']}

    dataloaders = {
        'train': DataLoader(datasets_map['train'], batch_size=batch_size, shuffle=True),
        'valid': DataLoader(datasets_map['valid'], batch_size=batch_size, shuffle=False),
        'test': DataLoader(datasets_map['test'], batch_size=batch_size, shuffle=False)
    }

    sizes = {x: len(datasets_map[x]) for x in ['train', 'valid', 'test']}
    class_names = datasets_map['train'].classes
    idx_to_class = {v: k for k, v in datasets_map['train'].class_to_idx.items()}

    return dataloaders, sizes, idx_to_class, len(class_names)

def build_mobilenet_model(num_classes):
    """Builds a MobileNetV3-Small model for Fine-Tuning."""
    print("Loading MobileNetV3-Small...")
    model = models.mobilenet_v3_small(weights='DEFAULT')

    # 1. Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # 2. UNFREEZE the last feature block.
    for param in model.features[-1].parameters():
        param.requires_grad = True

    # 3. Replace the Head (Classifier)
    num_inputs = model.classifier[0].in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_inputs, 256),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)
    )

    return model.to(DEVICE)

def train_model(model, dataloaders, dataset_sizes, epochs=15):
    """Training loop with Differential Learning Rates."""
    criterion = nn.NLLLoss()

    optimizer = optim.SGD([
        {'params': model.features[-1].parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(),   'lr': 1e-2}
    ], momentum=0.9)

    best_loss = float('inf')
    history = []

    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"Epoch: {epoch+1}/{epochs}")

        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels.data)

        # --- VALIDATION ---
        model.eval()
        valid_loss = 0.0
        valid_acc = 0.0

        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                valid_acc += torch.sum(preds == labels.data)

        # --- FIX: USE .float() INSTEAD OF .double() ---
        # MPS (Mac) does not support double (float64), only float (float32)
        avg_train_loss = train_loss / dataset_sizes['train']
        avg_train_acc = train_acc.float() / dataset_sizes['train'] # <--- Changed here

        avg_valid_loss = valid_loss / dataset_sizes['valid']
        avg_valid_acc = valid_acc.float() / dataset_sizes['valid'] # <--- Changed here

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc.item(), avg_valid_acc.item()])

        print(f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f}")
        print(f"Valid Loss: {avg_valid_loss:.4f} Acc: {avg_valid_acc:.4f}")

        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            if not os.path.exists('model/m6'):
                os.makedirs('model/m6')
            torch.save(model.state_dict(), 'model/m6/best_mobilenet.pth')
            print("Saved Best Model 💾")

    return model, history

def predict(model, image_path, idx_to_class):
    """Simple inference function."""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    transform = get_transforms()['test']
    img = Image.open(image_path)

    img_t = transform(img).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        out = model(img_t)
        probs = torch.exp(out)
        top_p, top_class = probs.topk(3, dim=1)

        print(f"\nPrediction for {os.path.basename(image_path)}:")
        for i in range(3):
            label = idx_to_class[top_class.cpu().numpy()[0][i]]
            score = top_p.cpu().numpy()[0][i]
            print(f"{i+1}. {label}: {score*100:.2f}%")

if __name__ == "__main__":
    # 1. Setup
    data_dir = setup_data()
    dataloaders, sizes, idx_to_class, num_classes = create_dataloaders(data_dir)
    print(f"Classes: {num_classes}, Device: {DEVICE}")

    # 2. Build MobileNet Model
    model = build_mobilenet_model(num_classes)

    # 3. Train
    model, history = train_model(model, dataloaders, sizes, epochs=15)

    # 4. Load & Predict (Example)
    model_path = 'model/m6/best_mobilenet.pth'
    if os.path.exists(model_path):
        print("\nLoading fine-tuned weights...")
        model = build_mobilenet_model(num_classes)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))

        # Test Prediction
        test_images = ['src/m6/skunk.jpg', 'src/m6/zebra.jpg', 'src/m6/llama-scaled.jpg']
        for img_path in test_images:
            predict(model, img_path, idx_to_class)
    else:
        print("Model file not found.")