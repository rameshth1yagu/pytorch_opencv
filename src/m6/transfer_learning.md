

# Transfer Learning: Fine-Tuning ResNet50

## 1. The Core Concept

**Transfer Learning** is the process of taking a neural network that has been trained on a massive dataset (like **ImageNet**, which has 1.2 million images and 1000 categories) and adapting it to solve a new, specific problem with a smaller dataset.

Instead of teaching a model how to "see" (detect edges, textures, shapes) from scratch, we reuse the "visual cortex" of a powerful pre-existing model.

### The Strategy: "Body" vs. "Head"

A Deep CNN can be thought of in two parts:

1. **The Body (Feature Extractor):** A stack of Convolutional layers that turns raw pixels into meaningful features. We **Freeze** this part to keep its pre-trained knowledge.
2. **The Head (Classifier):** The final Linear layers that make the decision. We **Replace** this part to match our new number of classes.

---

## 2. Implementation Walkthrough

### Step 1: Configuration & Device Detection

We write code that works everywhere. It automatically detects if you are using an NVIDIA GPU (`cuda`), a Mac with Apple Silicon (`mps`), or a CPU.

```python
import torch
# Automatically select the fastest available hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

```

### Step 2: Data Augmentation

We treat training and testing data differently.

* **Training:** We "mess up" the images (random crops, rotations, flips). This forces the model to learn robust features (e.g., recognizing a bear even if it's upside down) rather than memorizing exact pixels.
* **Validation/Test:** We only resize and normalize. We want to test the model on "real" clean images.

**Critical Detail:** We must normalize using the **ImageNet Mean & Std** (`[0.485, ...]`) because the pre-trained ResNet expects inputs with this specific mathematical distribution.

```python
def get_transforms():
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)), # Augmentation
            transforms.RandomRotation(15),                       # Augmentation
            transforms.RandomHorizontalFlip(),                   # Augmentation
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

```

### Step 3: Building the Transfer Model

This is the most important part of the code. We perform "Brain Surgery" on ResNet50.

1. **Load:** Download ResNet50 with `weights='DEFAULT'`.
2. **Freeze:** Loop through parameters and set `requires_grad = False`. This stops the optimizer from changing the Convolutional weights.
3. **Swap:** Replace `model.fc` (the final layer) with a new block that outputs `num_classes`.

```python
def build_model(num_classes):
    # 1. Load Pre-trained Brain
    model = models.resnet50(weights='DEFAULT') 
    
    # 2. Freeze the Body
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Replace the Head
    # in_features is 2048 for ResNet50
    num_inputs = model.fc.in_features 
    
    model.fc = nn.Sequential(
        nn.Linear(num_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),            # Dropout prevents overfitting
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)        # Output format
    )
    
    return model.to(DEVICE)

```

### Step 4: The Training Loop

We use **Negative Log Likelihood Loss (`NLLLoss`)** because our model outputs `LogSoftmax`.

* **Optimizer:** Notice we pass `model.parameters()`. Since we froze the body, the optimizer *only* sees the parameters in the new head. This makes training extremely fast.

```python
def train_epoch(model, dataloader, optimizer, criterion):
    model.train() # Enable Dropout/BatchNorm
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()           # 1. Clear gradients
        outputs = model(inputs)         # 2. Forward pass
        loss = criterion(outputs, labels) # 3. Calculate error
        loss.backward()                 # 4. Backward pass
        optimizer.step()                # 5. Update weights

```

### Step 5: Safe Model Loading

In PyTorch 2.6+, loading a full model object requires explicitly allowing it for security reasons.

```python
# weights_only=False is required if you used torch.save(model)
# map_location ensures we can load a GPU-trained model on a CPU/Mac
model = torch.load('best_transfer_learning.pt', map_location=DEVICE, weights_only=False)

```

---

## 3. Why This Works Better Than "From Scratch"

1. **Data Efficiency:** You can get 95%+ accuracy with only 50 images per class, whereas training from scratch might require 1000+ images per class.
2. **Speed:** We are only training the final 2 layers, not the deep 50 layers. An epoch takes seconds, not minutes.
3. **Feature Richness:** The model already knows complex concepts (shapes, lighting, perspective) from ImageNet, so it doesn't have to relearn basic vision.