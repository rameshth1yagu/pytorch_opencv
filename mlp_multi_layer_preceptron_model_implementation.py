import torch  # PyTorch for tensor operations
from torch import nn  # Neural network modules
import torch.nn.functional as F  # Functional API for layers and activations
import torch.optim as optim  # Optimizers
from torchinfo import summary  # Model summary utility
import download_transform_loader as dtl  # Custom data loader module

import torchvision  # PyTorch vision library
from torchvision import datasets, transforms  # Datasets and transforms

import matplotlib.pyplot as plt  # Plotting
import numpy as np  # Array operations
import random  # Random utilities
import time  # Timing utilities

# Define a Multi-Layer Perceptron (MLP) model for image classification
class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc0 = nn.Linear(28 * 28, 512)  # First fully connected layer
        self.bn0 = nn.BatchNorm1d(512)  # Batch normalization
        self.fc1 = nn.Linear(512, 256)  # Second fully connected layer
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)  # Third fully connected layer
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)  # Output layer
        self.bn3 = nn.BatchNorm1d(num_classes)
        self.dropout = nn.Dropout(0.3)  # Dropout for regularization

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn0(self.fc0(x)))  # FC + BN + ReLU
        x = self.dropout(x)  # Dropout
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # Output layer
        x = F.log_softmax(x, dim=1)  # Log softmax for classification
        return x

# Utility to save model weights to disk
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("Model weights saved to disk!")

# Setup loss, optimizer, and device
def setup():
    # 1. Define the Loss Function
    # Since your model ends with F.log_softmax, we MUST use NLLLoss (Negative Log Likelihood)
    criterion = nn.NLLLoss()
    # 2. Define the Optimizer
    # We use Adam (it's generally faster/smarter than SGD).
    # lr=0.001 is the "Learning Rate" (how big of a step to take)
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    # 3. Move model to GPU (if available) - Critical for speed!
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mlp.to(device)
    return criterion, optimizer, device

# Training loop for the MLP model
def train_the_model(model, train_loader, criterion, optimizer, device, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        # --- THE CONNECTION HAPPENS HERE ---
        # The 'train_loader' automatically gives us a Batch of 64 Images and 64 Labels
        for i, (images, labels) in enumerate(train_loader):
            # A. Move data to the same device as the model (GPU/MPS)
            images, labels = images.to(device), labels.to(device)
            # B. Zero the Gradients
            # (Reset the mechanic's tools from the previous step)
            optimizer.zero_grad()
            # C. Forward Pass (The Guess)
            # We pass the batch of images into the model
            outputs = mlp(images)
            # D. Calculate Loss (The Score)
            # Compare model's 'outputs' vs the actual 'labels'
            loss = criterion(outputs, labels)
            # E. Backward Pass (The Blame Game)
            # Calculate how much each weight contributed to the error
            loss.backward()
            # F. Optimization Step (The Fix)
            # Update the weights to reduce error next time
            optimizer.step()
            # (Optional) Print stats every 100 batches so we know it's working
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0
    print("Training Finished!")

# Create an instance of the MLP model
mlp = MLP(num_classes=10)  # 10 classes for FashionMNIST
# Print the model summary
print(summary(mlp, input_size=(1, 1, 28, 28), row_settings = ["var_names"]))
criterion, optimizer, device = setup()  # Setup training utilities
train_loader, test_loader = dtl.data_loader()  # Get data loaders
train_the_model(mlp, train_loader, criterion, optimizer, device, epochs=50)  # Train model
save_model(mlp, "fashion_mnist_mlp.pth")  # Save trained model
