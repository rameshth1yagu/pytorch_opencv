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

For full implementation details, see `mlp_multi_layer_preceptron_model_implementation.py` in this repository.
