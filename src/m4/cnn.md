
---

# Convolutional Neural Networks (CNN): Concepts & Implementation

## 1. Understanding CNNs (The "Pattern Finder")

A **Convolutional Neural Network (CNN)** is a specialized neural network designed to process data with a grid-like topology, such as images. Unlike a standard Multi-Layer Perceptron (MLP) which flattens an image and treats every pixel as independent, a CNN preserves the spatial structure of the image.

### Key Concepts

#### A. The Convolution (`Conv2d`)

Think of this as a **"Scanner"**. Instead of connecting every input pixel to every neuron, a CNN uses a small filter (kernel) that slides across the image.

* **Filter/Kernel:** A small matrix (e.g., ) that detects specific features like edges, curves, or textures.
* **Feature Map:** The output of the scan. It tells us *where* a feature exists in the image.

#### B. The Pooling (`MaxPool2d`)

Think of this as the **"Summarizer"**. After scanning, we often have too much detail. Pooling shrinks the image to reduce computation and make the model robust to small shifts.

* **Max Pooling:** Looks at a small patch (e.g., ) and keeps only the highest value.
* **Result:** The image gets smaller (spatial compression), but the important features are kept.

#### C. The Feature Explosion

As the network goes deeper, the image size () gets **smaller**, but the number of filters (Channels) gets **larger** (e.g., ).

* **Early Layers:** See simple edges.
* **Deep Layers:** See complex objects (eyes, ears, wheels).

---

## 2. Project Implementation: Monkey Species Classification

This project classifies 10 different monkey species using a custom CNN architecture in PyTorch. The code is modularized into four main parts.

### Step 1: Data Setup (`m4_data_setup.py`)

This module handles downloading the dataset, preparing the transformations (augmentations), and creating the DataLoaders.

**Key Features:**

* **Reproducibility:** A `set_seed()` function ensures we get the same random split every time.
* **Augmentation:** We use `RandomHorizontalFlip` and `RandomErasing` to create "fake" training data, which prevents the model from memorizing the images.
* **Normalization:** All images are normalized using the specific mean and standard deviation of this dataset.

```python
# snippet from m4_data_setup.py
def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5), # Augmentation
        transforms.Normalize(mean=[0.4368, ...], std=[0.2457, ...])
    ])
    return train_transforms, val_transforms

```

### Step 2: The Model Architecture (`m4_cnn_model.py`)

We built a custom class `MonkeyClassifier`. It addresses a specific issue with Apple Silicon (MPS) by moving the pooling operation to the CPU.

**Architecture Breakdown:**

1. **Feature Extractor:** A stack of `Conv2d` -> `BatchNorm` -> `ReLU` -> `MaxPool` blocks.
2. **Adaptive Pooling:** Forces the output to a fixed  size, regardless of input dimensions.
3. **Classifier:** A Flatten layer followed by Linear layers to predict the 10 classes.

```python
# snippet from m4_cnn_model.py
class MonkeyClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # ... more layers ...
        )
        # The Mac/MPS Fix Layer
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.cpu() # Move to CPU for safe pooling on Mac
        x = self.avgpool(x)
        x = x.to(DEVICE) # Move back to GPU
        x = self.classifier(x)
        return x

```

### Step 3: Training the Model (`m4_train.py`)

This script orchestrates the learning process. It runs for a specified number of epochs, tracks the loss/accuracy, and saves the best version of the model.

**Process:**

1. **Forward Pass:** The model guesses the monkey species.
2. **Loss Calculation:** We compare the guess to the actual label using `CrossEntropyLoss`.
3. **Backward Pass:** Gradients are calculated to understand how to fix the error.
4. **Optimizer Step:** The `Adam` optimizer updates the weights.
5. **Validation:** We check the model's performance on unseen data to ensure it's not overfitting.

```python
# snippet from m4_train.py
for epoch in range(NUM_EPOCHS):
    # Train
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    # Save Best Model
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"--> New Best Model Saved! ({val_acc:.2f}%)")

```

### Step 4: Prediction (`m4_predict.py`)

This script loads the saved `best_monkey_model.pth` and tests it on random images from the validation set.

**Key Logic:**

* **Smart Loading:** It checks if the model file exists and handles mapping GPU weights to CPU if necessary.
* **Visualization:** It displays a grid of images with their predicted labels and confidence scores. Green titles indicate correct predictions; red indicates errors.

```python
# snippet from m4_predict.py
def load_smart_model():
    model = MonkeyClassifier(num_classes=10)
    # Load weights safely
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval() # Freeze Dropout/BatchNorm
    return model

```

---

## Summary of Workflow

1. **Run `m4_train.py**`: Downloads data, trains the CNN, and saves `best_monkey_model.pth`.
2. **Run `m4_predict.py**`: Loads the best model and visualizes predictions on random test images.
