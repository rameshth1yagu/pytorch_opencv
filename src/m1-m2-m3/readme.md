# PyTorch & Deep Learning Project Overview

## 1. Tensors: The Core Object

A **Tensor** is a multi-dimensional matrix, similar to a NumPy array, but with two key advantages:

| Feature         | NumPy Array | PyTorch Tensor                |
|-----------------|-------------|-------------------------------|
| Hardware        | CPU Only    | CPU + GPU (Parallel speed)    |
| Intelligence    | Just data   | Tracks math for calculus      |
| Usage           | Data Analysis | Deep Learning / Gradients   |

### Hardware Acceleration

Move tensors between CPU and GPU for faster computation.

```python
import torch

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
x_gpu = x.to(device)      # Move to GPU
x_cpu = x_gpu.cpu()       # Move back to CPU
```

**Note:** You cannot mix devices (CPU tensors cannot interact with GPU tensors).

---

## 2. Shaping Data

Change tensor dimensions without altering the total number of elements.

- `.view(rows, cols)`: Fast, requires contiguous memory.
- `.reshape(rows, cols)`: Safer, copies data if needed.
- Magic `-1`: `x.view(4, -1)` lets PyTorch infer the correct dimension.

---

## 3. Autograd & Backpropagation

PyTorch tracks operations for automatic differentiation.

- `requires_grad=True`: Start recording operations.
- `loss.backward()`: Computes gradients.
- `w.grad`: Accesses the gradient.
- `w.grad.zero_()`: Clears accumulated gradients.

**Example:**
```python
import torch

w = torch.tensor([1.0], requires_grad=True)
target = torch.tensor([4.0])

for i in range(10):
    loss = (w - target) ** 2
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 0.1
        w.grad.zero_()
print(f"Learned w: {w.item()}")
```

---

## 4. Image Basics & OpenCV

Images are arrays of numbers. Use OpenCV to read and preprocess images.

- OpenCV reads images in BGR; PyTorch expects RGB.
- Grayscale images are 2D matrices.

**Example:**
```python
import cv2

img = cv2.imread("digit_0.jpg", cv2.IMREAD_GRAYSCALE)
# Pixel values: 0 (Black) to 255 (White)
```

---

## 5. Data Pipeline

Prepare data for training using datasets, transforms, and loaders.

- **Dataset:** Downloads and manages data.
- **Transforms:** Preprocesses data (e.g., normalization).
- **DataLoader:** Batches and shuffles data.

**Example:**
```python
from torchvision import transforms
import torch

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
```

---

## 6. Neural Network Architecture

A deep neural network for image classification.

- **Linear Layers:** Perform `y = wx + b`.
- **ReLU:** Non-linear activation.
- **Output:** 10 neurons for 10 classes, uses `log_softmax`.

**Example:**
```python
import torch.nn as nn
import torch.nn.functional as F

class FashionNet(nn.Module):
    def forward(self, x):
        x = x.view(-1, 784)      # Flatten
        x = F.relu(self.fc1(x))  # Linear + ReLU
        x = self.output(x)       # Final prediction
        return F.log_softmax(x, dim=1)
```

---

## 7. Autograd Example

PyTorch computes derivatives automatically.

**Example:**
```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)
loss = x * y + y ** 2
loss.backward()
print(x.grad) # Output: 3.0
print(y.grad) # Output: 8.0
```

---

## 8. The Training Loop

The standard training cycle:

1. Forward: Pass data through the model.
2. Loss: Compute error.
3. Zero Grad: Clear gradients.
4. Backward: Compute new gradients.
5. Step: Update weights.

---

## Project Files Overview

### Python Files

#### `hello.py`
- Demonstrates basic tensor operations, device movement, and matrix multiplication.

#### `train_mlp_multi_layer_perceptron.py`
- Implements the data pipeline, neural network architecture, and training loop for FashionMNIST classification.

#### `autograd_backpropagation.py`
- Shows how to use PyTorch's autograd to compute gradients for simple mathematical expressions.

#### `mlp_multi_layer_preceptron_model_implementation.py`
- Implements a deep learning pipeline using PyTorch for classifying FashionMNIST images with a multi-layer perceptron (MLP) architecture.

#### `predict_model.py`
- Provides utilities for loading a trained model, preprocessing images, making predictions, and plotting confusion matrices.

### Image Files

#### `digit_0.jpg` / `digit_0.png`
- Example grayscale image of a handwritten digit, used for image loading and preprocessing demonstrations.

---

# Multi-Layer Perceptron (MLP) Model Implementation for FashionMNIST

This file implements a deep learning pipeline using PyTorch for classifying FashionMNIST images with a multi-layer perceptron (MLP) architecture. Below is a summary and code snippets from the implementation.

---

## Key Components

### 1. Imports
- PyTorch core and neural network modules
- Optimizers and model summary utilities
- Custom data loader for FashionMNIST
- Torchvision for datasets and transforms
- Matplotlib, NumPy, random, and time for visualization and utilities

### 2. Model Architecture

```python
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
        x = x.view(x.size(0), -1)           # Flatten input
        x = F.relu(self.bn0(self.fc0(x)))   # FC + BN + ReLU
        x = self.dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)                     # Output layer
        x = F.log_softmax(x, dim=1)         # Log softmax for classification
        return x
```

### 3. Training Utilities
- **save_model(model, path):** Saves model weights to disk.
- **setup():** Configures loss function (NLLLoss), optimizer (Adam), and device (MPS or CPU).
- **train_the_model(...):** Standard training loop with forward, backward, and optimization steps. Prints loss every 100 batches.

### 4. Training Workflow

```python
mlp = MLP(num_classes=10)  # 10 classes for FashionMNIST
print(summary(mlp, input_size=(1, 1, 28, 28), row_settings=["var_names"]))
criterion, optimizer, device = setup()  # Setup training utilities
train_loader, test_loader = dtl.data_loader()  # Get data loaders
train_the_model(mlp, train_loader, criterion, optimizer, device, epochs=5)  # Train model
save_model(mlp, "fashion_mnist_mlp.pth")  # Save trained model
```

---

## Highlights
- Uses batch normalization and dropout for improved generalization.
- Trains on GPU (Apple MPS or CPU fallback).
- Prints model summary and training progress.
- Modular design for easy extension and experimentation.

---

## Advanced Concepts and Insights

### 1. Advanced Architecture (The "Professional" MLP)

We moved from a basic network to a robust, production-grade architecture.

**The "Sandwich" Pattern:** We established the standard order of operations for a layer:

    Linear → Batch Norm → ReLU → Dropout

- **Batch Normalization (bn):** The "Stabilizer." It re-centers data inside the network (Mean=0, Std=1) so the optimizer doesn't get confused by wild swings in values.
- **Dropout:** The "Drill Sergeant." It randomly kills neurons (e.g., 30%) during training to force the model to be redundant and prevent Overfitting (memorization).
- **The Output Layer:** We learned not to use ReLU or BatchNorm on the final layer. We use log_softmax to get probabilities, which requires NLLLoss (Negative Log Likelihood) as the loss function.

### 2. Understanding Model Internals

- **Parameter Counting:** We analyzed the "Summary Table."
- **Weights:** Connections between neurons (Input×Output).
- **Biases:** One per output neuron.
- **Total Params:** The sum of all learnable numbers (the model's "brain size").
- **The "Crash Test Dummy":** The input_size=(1, 1, 28, 28) in the summary function is a fake input used to run a "dry pass" so PyTorch can calculate the shape of every layer dynamically.

### 3. The Training Loop (The "Gym Workout")

We broke down the manual engine of learning:

- `model.train()`: Crucial command to tell Dropout and BatchNorm to "wake up" and behave dynamically.
- `zero_grad()`: Cleaning the whiteboard before the next calculation.
- `backward()`: The "Blame Game" (calculating gradients).
- `step()`: The actual update of weights.
- **Epochs:** We learned we must loop multiple times because the Learning Rate forces us to take tiny steps. We can't jump to the solution in one go.

### 4. Validation & Inference (The "Exam")

- `model.eval()`: The command to freeze Dropout and lock BatchNorm statistics. Without this, predictions are random and inconsistent.
- `torch.no_grad()`: Turning off the calculus engine to save memory and speed up testing.
- **Saving/Loading:** We learned that a .pth file is just a dictionary of numbers (state_dict). To use it, you must first build the exact same Code Structure (Class Definition) and then pour the numbers into it.

### 5. Evaluation Metrics (The "Detective Work")

- **Confusion Matrix:** A grid showing Truth vs. Prediction.
    - **Diagonal:** Correct guesses.
    - **Off-Diagonal:** Mistakes.
- **Insight:** Your model confused Shirts with T-shirts and Pullovers because they are geometrically similar (long sleeves, upper body), but it perfectly identified Boots and Bags.
- **The MLP Flaw:** We concluded that MLPs are fundamentally limited for vision because Flattening (28×28→784) destroys spatial relationships (neighbors).

---

## Main Training and Validation Functions

### main Function
The `main` function orchestrates the full training and validation process for the MLP model. It:
- Iterates over the specified number of epochs.
- Calls the `train` function for each epoch to train the model and collect training loss and accuracy.
- Calls the `validation` function for each epoch to evaluate the model on the validation set and collect validation loss and accuracy.
- Prints a summary for each epoch and plots the loss and accuracy curves at the end.

**Example:**
```python
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
```

### validation Function
The `validation` function evaluates the model on the validation (or test) set. It:
- Sets the model to evaluation mode.
- Iterates through the validation data, computes loss and accuracy.
- Prints progress every 100 batches and a summary at the end.

**Example:**
```python
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
```

---

# Model Prediction and Evaluation Script (`predict_model.py`)

This script provides utilities for loading a trained MLP model, preprocessing images, making predictions, visualizing results, and plotting a confusion matrix for the FashionMNIST dataset.

## Key Functions

### load_model
Loads the trained model weights and prepares the model for inference.

### preprocess_image
Reads an image, converts it to grayscale, resizes to 28x28, inverts if needed, and normalizes it to match training conditions.

### predict
Runs the model on a preprocessed image tensor and returns the top prediction and top-3 probabilities.

### visualize_prediction
Displays the input image and a bar chart of the top-3 predicted classes and their probabilities.

### plot_confusion_matrix
Generates and displays a confusion matrix for the model's predictions on a dataset, with progress print statements for each step.

**Example:**
```python
def plot_confusion_matrix(model, data_loader, device):
    print("\n[INFO] Generating predictions for confusion matrix...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(data_loader):
                print(f"  Processed batch {batch_idx+1}/{len(data_loader)}")
    print("[INFO] Computing confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    print("[INFO] Plotting confusion matrix...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    print("[INFO] Confusion matrix plot complete.")
```

---

## Example Usage

```python
# Load the trained model
model = load_model()

# Preprocess an input image
img_tensor, original_img = preprocess_image('path/to/image.png')

# Make a prediction
pred_idx, confidence, top3_idx, top3_prob = predict(model, img_tensor)

# Visualize the prediction
visualize_prediction(original_img, pred_idx, confidence, top3_idx, top3_prob)

# Plot the confusion matrix on the test set
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_, test_loader = data_loader()
plot_confusion_matrix(model, test_loader, device)
```

---

For full implementation details, see `predict_model.py` in this repository.
