"""
ResNet50 Transfer Learning Pipeline
===================================
This script demonstrates how to fine-tune a massive pre-trained model (ResNet50)
on a smaller dataset (Caltech-256 subset).

Concepts Covered:
1. Data Augmentation: Randomly rotating/cropping images to prevent overfitting.
2. Transfer Learning: Freezing the 'body' and replacing the 'head' (classifier).
3. Training Loop: The standard Forward -> Loss -> Backward -> Optimizer cycle.
4. Inference: Running the model on new, unseen images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import requests
from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Global Settings
plt.style.use('ggplot')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def setup_data(dataset_name='caltech256_subset'):
    """
    Downloads and extracts the dataset if not present.
    Returns the root directory of the dataset.
    """
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
    """
    Returns a dictionary of transformations for train, validation, and test sets.

    - Train: Includes Augmentation (Rotation, Flip) to make the model robust.
    - Valid/Test: Only Resize and Normalize (We don't augment test data).
    """
    # ImageNet statistics (Mean & Std Dev) required for pre-trained models
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224), # Final input size must be 224x224
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
    }

def create_dataloaders(data_dir, batch_size=32):
    """
    Creates PyTorch DataLoaders.

    A DataLoader wraps the dataset and handles:
    1. Batching (Loading 32 images at a time).
    2. Shuffling (Randomizing order for training).
    3. Multiprocessing (Loading data in parallel).
    """
    transforms_map = get_transforms()
    dirs = {x: os.path.join(data_dir, x) for x in ['train', 'valid', 'test']}

    # Load ImageFolder datasets
    datasets_map = {x: datasets.ImageFolder(dirs[x], transform=transforms_map[x]) for x in ['train', 'valid', 'test']}

    # Create DataLoaders
    # Note: We shuffle 'train' but NOT 'valid' or 'test'.
    dataloaders = {
        'train': DataLoader(datasets_map['train'], batch_size=batch_size, shuffle=True),
        'valid': DataLoader(datasets_map['valid'], batch_size=batch_size, shuffle=False),
        'test': DataLoader(datasets_map['test'], batch_size=batch_size, shuffle=False)
    }

    dataset_sizes = {x: len(datasets_map[x]) for x in ['train', 'valid', 'test']}
    class_names = datasets_map['train'].classes

    # Create index to class mapping (e.g., 0 -> "bear", 1 -> "chimp")
    idx_to_class = {v: k for k, v in datasets_map['train'].class_to_idx.items()}

    return dataloaders, dataset_sizes, idx_to_class, len(class_names)

def build_model(num_classes):
    """
    Loads ResNet50, freezes the body, and replaces the head.
    """
    print("Loading Pre-trained ResNet50...")
    model = models.resnet50(weights='DEFAULT') # Load ImageNet weights

    # FREEZE PARAMETERS:
    # We tell PyTorch not to calculate gradients for the convolutional layers.
    # This keeps the pre-trained "knowledge" intact and speeds up training.
    for param in model.parameters():
        param.requires_grad = False

    # MODIFY CLASSIFIER:
    # ResNet's original classifier (fc) outputs 1000 classes (ImageNet).
    # We replace it with a new sequential block for OUR specific classes.
    num_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_inputs, 256), # Dense layer
        nn.ReLU(),                  # Activation
        nn.Dropout(0.4),            # Dropout: Randomly zeroes neurons to prevent overfitting
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)        # Output: Log Probabilities
    )

    return model.to(DEVICE)

def train_model(model, dataloaders, dataset_sizes, epochs=25):
    """
    The Main Training Loop.
    """
    criterion = nn.NLLLoss() # Negative Log Likelihood Loss (Standard for LogSoftmax)
    # Optimizer: Only optimize parameters that require gradients (the new head)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    best_loss = float('inf')
    history = []

    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"Epoch: {epoch+1}/{epochs}")

        # --- TRAINING PHASE ---
        model.train() # Enable Dropout/BatchNorm
        train_loss = 0.0
        train_acc = 0.0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()           # Reset gradients
            outputs = model(inputs)         # Forward pass
            loss = criterion(outputs, labels) # Calculate error
            loss.backward()                 # Backward pass (Calculate gradients)
            optimizer.step()                # Update weights

            # Tracking
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels.data)

        # --- VALIDATION PHASE ---
        model.eval() # Disable Dropout/BatchNorm
        valid_loss = 0.0
        valid_acc = 0.0

        with torch.no_grad(): # Turn off gradient engine (saves memory)
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                valid_acc += torch.sum(preds == labels.data)

        # Calculate Averages
        avg_train_loss = train_loss / dataset_sizes['train']
        avg_train_acc = train_acc.double() / dataset_sizes['train']
        avg_valid_loss = valid_loss / dataset_sizes['valid']
        avg_valid_acc = valid_acc.double() / dataset_sizes['valid']

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc.item(), avg_valid_acc.item()])

        print(f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f}")
        print(f"Valid Loss: {avg_valid_loss:.4f} Acc: {avg_valid_acc:.4f}")

        # Save Best Model
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            # Create model directory if not exists
            if not os.path.exists('model/m6'):
                os.makedirs('model/m6')
            print("Saving new best model...")
            torch.save(model, 'model/m6/best_transfer_learning.pt')

    return model, history

def compute_test_accuracy(model, dataloader, dataset_size):
    """
    Evaluates the model on the Test Set.
    """
    criterion = nn.NLLLoss()
    model.eval()
    test_loss = 0.0
    test_acc = 0.0

    print(f"\nEvaluating on Test Set ({dataset_size} images)...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            test_acc += torch.sum(preds == labels.data)

    avg_acc = test_acc.double() / dataset_size
    print(f"Test Accuracy: {avg_acc:.4f}")

def predict_image(model, image_path, idx_to_class):
    """
    Runs inference on a single custom image.
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # 1. Prepare Image
    transform = get_transforms()['test']
    img = Image.open(image_path)

    # Visualization
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # 2. Transform & Move to Device
    img_t = transform(img).unsqueeze(0) # Add batch dimension [1, 3, 224, 224]
    img_t = img_t.to(DEVICE)

    # 3. Predict
    model.eval()
    with torch.no_grad():
        outputs = model(img_t)
        # Convert LogProbabilities to Probabilities (0-1 range)
        probs = torch.exp(outputs)

        # Get Top 3 Predictions
        top_p, top_class = probs.topk(3, dim=1)

        # Convert to CPU for printing
        top_p = top_p.cpu().numpy()[0]
        top_class = top_class.cpu().numpy()[0]

        print(f"\nPredictions for {os.path.basename(image_path)}:")
        for i in range(3):
            label = idx_to_class[top_class[i]]
            score = top_p[i] * 100
            print(f"{i+1}. {label}: {score:.2f}%")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Setup
    data_dir = setup_data()
    dataloaders, sizes, idx_to_class, num_classes = create_dataloaders(data_dir)
    print(f"Classes: {num_classes}, Device: {DEVICE}")

    # 2. Build Model
    model = build_model(num_classes)

    # 3. Train (Uncomment to retrain)
    # model, history = train_model(model, dataloaders, sizes, epochs=5)

    # 4. Load Pre-trained (Safe Loading Logic)
    model_path = 'model/m6/best_transfer_learning.pt'
    if os.path.exists(model_path):
        print("\nLoading saved model...")
        # weights_only=False is required because we saved the full model object previously
        model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    else:
        print("\nModel not found. Please uncomment training line above.")

    # 5. Evaluate
    compute_test_accuracy(model, dataloaders['test'], sizes['test'])

    # 6. Predict Custom Images
    custom_images = ['src/m6/skunk.jpg', 'src/m6/zebra.jpg', 'src/m6/llama-scaled.jpg']
    for img_path in custom_images:
        predict_image(model, img_path, idx_to_class)