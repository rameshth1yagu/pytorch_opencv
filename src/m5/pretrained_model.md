# Transfer Learning: The "Superpower" of Deep Learning

## 1. What is Transfer Learning?

In the previous module (CNNs), you built a "brain" from scratch. You defined the layers, initialized random weights, and taught it what an "edge" and a "curve" were by showing it thousands of monkey images. This is effective but slow and requires massive datasets.

**Transfer Learning** is the technique of taking a model that has *already* been trained on a massive dataset (like **ImageNet**) and reusing its knowledge for your own task.

* **The "Body" (Feature Extractor):** Already knows how to see edges, textures, eyes, wheels, and fur because it has seen 1.2 million images.
* **The "Head" (Classifier):** We can either use the original head (to classify general objects) or swap it for a new one (to classify specific things like your monkeys).

In this module, we are doing **Inference with a Pre-trained Model**. We are using a standard **ResNet18** model to classify everyday objects without any training.

---

## 2. Key Components of the Code

### A. The Model: ResNet18

We use **ResNet18** (Residual Network with 18 layers). It is a famous architecture that solved the "Vanishing Gradient" problem using "Skip Connections," allowing for very deep networks.

```python
# Loading the pre-trained brain
model = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

```

* **`weights=...IMAGENET1K_V1`**: This downloads the "knowledge." Without this, the model is just an empty shell of random numbers.
* **`model.eval()`**: Crucial! It freezes dynamic layers like **Dropout** and **BatchNorm**. If you forget this, the model might give different results every time you run it, even on the same image.

### B. The Preprocessing Rules

Models are picky. ResNet was trained on images that were processed in a very specific way. To get correct predictions, we must treat our new images exactly the same way.

```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), # The standard input size for ResNet
    transforms.ToTensor(),      # Convert pixels (0-255) to Tensors (0-1)
    transforms.Normalize(       # Mathematical alignment
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

```

* **Why those specific numbers?** `[0.485, 0.456, 0.406]` are the average Red, Green, and Blue values of the millions of images in the ImageNet dataset. Subtracting them centers the data, making the math inside the neural network stable.

### C. The Inference Step

Running a prediction involves manipulating the data shape.

```python
# 1. Add Batch Dimension
# Input: [3, 224, 224] -> Model expects: [Batch_Size, 3, 224, 224]
batch_t = torch.unsqueeze(img_t, 0)

# 2. Forward Pass (No Gradients needed for inference)
with torch.no_grad():
    out = model(batch_t)

# 3. Interpret Output
# The model outputs "Logits" (raw scores). We use Softmax to get Percentages.
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

```

---

## 3. Complete Python Script

Save this code as `m5_transfer_learning.py`. It automates the entire process: downloading assets, loading the model, and visualizing predictions.

```python
"""
ResNet18 Image Classification Pipeline
======================================
Key Steps:
1. Downloads necessary assets (test images and class labels).
2. Sets up the specific image transformations required by ResNet.
3. Loads the pre-trained model.
4. Runs prediction on images and visualizes the results.
"""

import os
import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from urllib.request import urlretrieve
from zipfile import ZipFile

# --- CONFIGURATION ---
DATA_DIR = "data"
ASSET_ZIP_URL = "https://www.dropbox.com/s/8srx6xdjt9me3do/TF-Keras-Bootcamp-NB07-assets.zip?dl=1"
CLASS_FILE_PATH = os.path.join(DATA_DIR, "imagenet_classes.txt")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

def download_and_setup_data():
    """Checks if data exists. If not, downloads and unzips the assets."""
    zip_path = os.path.join(os.getcwd(), DATA_DIR, "PyTorch-Bootcamp-NB07-assets.zip")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(zip_path) and not os.path.exists(CLASS_FILE_PATH):
        print(f"Downloading assets from {ASSET_ZIP_URL}...")
        try:
            urlretrieve(ASSET_ZIP_URL, zip_path)
            print("Extracting assets...")
            with ZipFile(zip_path, 'r') as z:
                z.extractall(DATA_DIR)
            print("Setup Complete.")
        except Exception as e:
            print(f"\nError downloading files: {e}")
    else:
        print("Assets already downloaded.")

def get_resnet_transforms():
    """Returns the specific transformation pipeline required by ResNet."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_model():
    """Downloads and loads the ResNet18 model with pre-trained ImageNet weights."""
    print("Loading ResNet18 Model...")
    model = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval() # Freeze layers for consistent inference
    return model

def load_class_names(file_path):
    """Reads the text file containing the 1000 class names."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Class file not found at {file_path}")
    with open(file_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def predict_single_image(image_path, model, transform, class_names):
    """Runs inference on a single image file."""
    if not os.path.exists(image_path):
        return None, None, None

    # 1. Load & Transform
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0) # Add batch dimension [1, 3, 224, 224]

    # 2. Predict
    with torch.no_grad():
        logits = model(img_tensor)
    
    # 3. Interpret
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0] * 100
    _, indices = torch.sort(logits, descending=True)
    top_idx = indices[0][0].item()
    
    return img, class_names[top_idx], probabilities[top_idx].item()

def visualize_result(img, class_name, conf):
    """Draws the prediction text on the image."""
    if img is None: return

    # Convert RGB (PIL) to BGR (OpenCV)
    bgr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_h, img_w = bgr_img.shape[:2]

    # Dynamic font scaling
    font_scale = max(0.003 * img_h, 0.5)
    thickness = max(1, int(img_h / 200))
    text = f"{class_name.split(',')[0]}: {conf:.1f}%"

    # Center Text
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = (img_w - text_w) // 2
    text_y = (img_h + text_h) // 10 + 20

    cv2.putText(bgr_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    # Convert back to RGB for Matplotlib
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.title(f"Prediction: {text}")
    plt.show()

if __name__ == "__main__":
    download_and_setup_data()
    model = load_model()
    transform = get_resnet_transforms()
    class_names = load_class_names(CLASS_FILE_PATH)
    
    if os.path.exists(IMAGES_DIR):
        print(f"\nProcessing images from: {IMAGES_DIR}")
        for img_file in os.listdir(IMAGES_DIR):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(IMAGES_DIR, img_file)
                print(f"Predicting: {img_file}...")
                img, label, conf = predict_single_image(full_path, model, transform, class_names)
                visualize_result(img, label, conf)

```