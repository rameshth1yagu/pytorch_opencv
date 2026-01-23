# Semantic Segmentation with FCN-ResNet101

## 1. What is Semantic Segmentation?

**Semantic Segmentation** is a Computer Vision task where the goal is to classify **every single pixel** in an image.

Unlike other tasks:

* **Image Classification:** Tells you "There is a bird in this image" (one label for the whole image).
* **Object Detection:** Draws a box around the bird (localization).
* **Semantic Segmentation:** "Paint" the pixels that belong to the bird in one color, and the pixels that belong to the sky in another.

### How is it different from "Instance Segmentation"?

* **Semantic Segmentation:** All "birds" are colored the same (e.g., all birds are blue).
* **Instance Segmentation:** Each individual bird gets a unique color (e.g., Bird 1 is Blue, Bird 2 is Green).

---

## 2. The Model Architecture: FCN-ResNet101

We use a **Fully Convolutional Network (FCN)** based on the ResNet101 architecture. This is a standard, powerful model provided by Torchvision.

### The "Backbone" (ResNet101)

This is the feature extractor (the "Body"). It takes the image and compresses it into high-level features (finding edges, textures, and shapes).

### The "Head" (FCN)

Standard CNNs end with a Flatten layer (destroying spatial info) to make a single prediction.
An **FCN (Fully Convolutional Network)** replaces the Flatten layer with more Convolutional layers that **upsample** (enlarge) the features back to the original image size. This allows the model to output a grid of predictions instead of a single number.

* **Input Shape:** `[Batch, 3, Height, Width]` (RGB Image)
* **Output Shape:** `[Batch, 21, Height, Width]` (21 Class Scores per pixel)

---

## 3. Implementation Step-by-Step

### Step 1: Configuration & Model Loading

We load `fcn_resnet101` with `DEFAULT` weights. These weights were trained on the **PASCAL VOC** dataset, which knows 21 specific classes (Aeroplane, Bicycle, Bird, Boat, etc.).

```python
import torch
import torchvision
from torchvision import models

# Detect hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Load Pre-trained Model
fcn = models.segmentation.fcn_resnet101(
    weights=torchvision.models.segmentation.FCN_ResNet101_Weights.DEFAULT
).eval() # Set to eval mode!

```

### Step 2: The "Translator" (Visualization)

The model outputs numbers (0, 1, 2...). We need a function to convert those numbers into colors so humans can see the result. This is the `decode_segmap` function.

**Concept:**
We create a "Lookup Table" of colors.

* Class 0 (Background)  Black `(0, 0, 0)`
* Class 3 (Bird)  Olive `(128, 128, 0)`

```python
import numpy as np

def decode_segmap(image, nc=21):
    # 1. Define Color Palette (PASCAL VOC Standard)
    label_colors = np.array([
        (0, 0, 0),       # 0=background
        (128, 0, 0),     # 1=aeroplane
        (0, 128, 0),     # 2=bicycle
        (128, 128, 0),   # 3=bird
        # ... (full list in main code)
    ])

    # 2. Create Empty RGB Layers
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    # 3. Paint Pixels
    for l in range(0, nc):
        idx = image == l        # Find all pixels of class 'l'
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    # 4. Stack Layers
    return np.stack([r, g, b], axis=2)

```

### Step 3: The Inference Pipeline

This function (`segment_image`) combines everything: Preprocessing  Inference  Post-processing.

**Key Logic:**

1. **Transform:** Resize to a standard size (e.g., 640px) and Normalize using ImageNet stats.
2. **Argmax:** The model gives 21 scores for every pixel. `argmax(dim=0)` collapses these 21 scores into a single "Winner" class ID.

```python
import torchvision.transforms as T
from PIL import Image

def segment_image(model, image_path):
    # 1. Load Image
    img = Image.open(image_path).convert('RGB')
    
    # 2. Preprocess
    transform = T.Compose([
        T.Resize(640),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # 3. Inference
    model.to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)['out']
    
    # 4. Post-Process (The Decision)
    # [1, 21, H, W] -> [21, H, W] -> [H, W]
    seg_map = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    
    # 5. Colorize
    rgb_mask = decode_segmap(seg_map)
    
    return rgb_mask

```

### Step 4: Smart Downloading

We use `requests` to safely download test images only if they don't already exist.

```python
import os
import requests

def download_file(url, save_name):
    if os.path.exists(save_name):
        print(f"Skipping {save_name}, already exists.")
        return

    print(f"Downloading {save_name}...")
    r = requests.get(url)
    with open(save_name, 'wb') as f:
        f.write(r.content)

```

---

## 4. Summary of Logic Flow

1. **Input:** A raw JPEG image (e.g., a bird in the sky).
2. **Preprocessing:** Image is resized to 640px and normalized (Math ready).
3. **Forward Pass:** The AI scans the image and outputs a `[21, 640, 640]` tensor. It is basically saying, "For pixel (0,0), I am 90% sure it's sky and 10% sure it's a bird."
4. **Argmax:** We force the AI to choose. "Pick the highest percentage." The tensor becomes `[640, 640]` integers.
5. **Decoding:** We map those integers (0, 3, 15) to colors (Black, Olive, Tan) so we can visualize the result.