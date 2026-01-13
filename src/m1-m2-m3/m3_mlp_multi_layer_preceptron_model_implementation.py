import torch  # PyTorch for tensor operations
from torch import nn  # Neural network modules
import torch.nn.functional as F  # Functional API for layers and activations
import torch.optim as optim  # Optimizers
from torchinfo import summary  # Model summary utility
import download_transform_loader as dtl  # Custom data loader module
import matplotlib.pyplot as plt  # Plotting

# Define a Multi-Layer Perceptron (MLP) model for image classification
class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc0 = nn.Linear(28 * 28, 512)  # First fully connected layer
        self.bn0 = nn.BatchNorm1d(512)      # Batch normalization
        self.fc1 = nn.Linear(512, 256)      # Second fully connected layer
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)      # Third fully connected layer
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)  # Output layer
        self.bn3 = nn.BatchNorm1d(num_classes)
        self.dropout = nn.Dropout(0.3)      # Dropout for regularization

    def forward(self, x):
        # Flatten the input tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn0(self.fc0(x)))   # FC + BN + ReLU
        x = self.dropout(x)                 # Dropout
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)                     # Output layer
        x = F.log_softmax(x, dim=1)         # Log softmax for classification
        return x

# Utility to save model weights to disk
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to disk at {path}!")

# Setup loss, optimizer, and device
def training_configuration():
    # Define the Loss Function (NLLLoss for log_softmax output)
    criterion = nn.NLLLoss()
    # Define the Optimizer (Adam is generally faster/smarter than SGD)
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    # Move model to GPU (if available) for speed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mlp.to(device)
    print(f"Training on device: {device}")
    return criterion, optimizer, device

# Training loop for the MLP model
def train(model, train_loader, criterion, optimizer, device, epochs=1):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    print("\nStarting training...")
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Reset gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item()
        predicted = torch.max(outputs.data, 1)[1]  # Get predicted class indices
        total_samples += labels.size(0)
        correct_predictions += predicted.eq(labels).float().sum().item()  # Count correct predictions
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(train_loader)} - Current Loss: {loss.item():.4f}")
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct_predictions / total_samples
    print(f"Training complete. Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def validation(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    print("\nStarting validation...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = torch.max(outputs.data, 1)[1]  # Get predicted class indices
            total_samples += labels.size(0)
            correct_predictions += predicted.eq(labels).float().sum().item()  # Count correct predictions
            if (i + 1) % 100 == 0:
                print(f"  Batch {i+1}/{len(val_loader)} - Current Loss: {loss.item():.4f}")
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct_predictions / total_samples
    print(f"Validation complete. Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def main(model, train_loader, val_loader, criterion, optimizer, device, epochs=5):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validation(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1:0>2}/{epochs} Summary: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% | Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    # Plotting loss and accuracy
    print("\nPlotting training and validation metrics...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    print("Training and validation complete.")
    return train_losses, train_accuracies, val_losses, val_accuracies

# Create an instance of the MLP model
mlp = MLP(num_classes=10)  # 10 classes for FashionMNIST
print("\nModel architecture summary:")
print(summary(mlp, input_size=(1, 1, 28, 28), row_settings = ["var_names"]))
criterion, optimizer, device = training_configuration()  # Setup training utilities
train_loader, test_loader = dtl.data_loader()  # Get data loaders
main(mlp, train_loader, test_loader, criterion, optimizer, device, epochs=40)  # Train and validate model
save_model(mlp, "fashion_mnist_mlp.pth")  # Save trained model
