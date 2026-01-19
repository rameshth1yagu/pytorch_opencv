"""
m4_train.py

Training script for the MonkeyClassifier. This file provides:
- train_one_epoch: run a single epoch of training
- validate: evaluate the model on the validation set
- a __main__ block that wires data, model, optimizer and runs training

Configuration is intentionally minimal; change NUM_EPOCHS, LEARNING_RATE
or SAVE_PATH near the top of the file.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

# Import our custom modules (File 1 and File 2)
import m4_data_setup as ds
import m4_cnn_model as model_def

# --- CONFIGURATION ---
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4 # 0.0001
DEVICE = model_def.DEVICE
SAVE_PATH = "model/m4/best_monkey_model.pth"

def train_one_epoch(model, loader, optimizer, criterion):
    """Runs one pass of training (Gradient Descent).

    Inputs:
    - model: nn.Module to train
    - loader: DataLoader providing (images, labels)
    - optimizer: optimizer instance (e.g., Adam)
    - criterion: loss function (e.g., CrossEntropyLoss)

    Returns:
    - avg_loss (float): average loss over the loader
    - acc (float): classification accuracy (percentage)
    """
    model.train() # Enable Dropout and BatchNorm
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()           # 1. Reset Gradients (Clean whiteboard)
        outputs = model(images)         # 2. Forward Pass (Make guess)
        loss = criterion(outputs, labels) # 3. Calculate Loss (How bad was the guess?)
        loss.backward()                 # 4. Backward Pass (Calculate corrections)
        optimizer.step()                # 5. Update Weights (Apply corrections)

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        # (predicted == labels) is a BoolTensor; use .sum().item() to count matches
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc

def validate(model, loader, criterion):
    """Runs validation (No Gradients, just checking).

    Same return values as train_one_epoch.
    """
    model.eval() # Disable Dropout
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # Disable math engine to save memory
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc

if __name__ == "__main__":
    # 1. Setup Data
    ds.set_seed(42)
    ds.download_and_unzip()
    train_loader, val_loader, class_names = ds.get_data_loaders()
    print(f"Data ready. Classes: {class_names}")

    # 2. Setup Model
    model = model_def.MonkeyClassifier(num_classes=len(class_names)).to(DEVICE)

    # 3. Setup Tools
    criterion = nn.CrossEntropyLoss() # Standard loss for classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter("runs/monkey_experiment_1") # TensorBoard logger

    # 4. Training Loop
    best_val_acc = 0.0
    print(f"Starting training on {DEVICE}...")

    for epoch in range(NUM_EPOCHS):
        # Run loops
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)

        # Log to Console
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)

        # Save Best Model Only
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  --> New Best Model Saved! ({val_acc:.2f}%)")

    print("Training Complete.")
    writer.close()