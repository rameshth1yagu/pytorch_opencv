"""
Semantic Segmentation with FCN-ResNet101
========================================
This script performs Semantic Segmentation (pixel-level classification)
using a pre-trained FCN-ResNet101 model.

It includes a safe download utility that only downloads test images
if they are not already present locally.
"""

import torch
import torchvision
from torchvision import models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import requests

# --- CONFIGURATION ---
plt.style.use('ggplot')
# Save images in a dedicated folder to keep things clean
INFERENCE_DIR = 'data/m7_inference'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def setup_directories():
    """Creates the folder for storing downloaded images."""
    if not os.path.exists(INFERENCE_DIR):
        os.makedirs(INFERENCE_DIR)

def download_file(url, save_name):
    """
    Downloads a file ONLY if it does not already exist.
    """
    # 1. THE CHECK: If file exists, skip everything.
    if os.path.exists(save_name):
        print(f"✅ Found {save_name}, skipping download.")
        return

    # 2. If we get here, the file is missing. Download it.
    print(f"⬇️  Downloading {save_name}...")
    try:
        r = requests.get(url)
        with open(save_name, 'wb') as f:
            f.write(r.content)
        print("   Download complete.")
    except Exception as e:
        print(f"   ❌ Error downloading: {e}")

def decode_segmap(image, nc=21):
    """
    Maps the 2D output (indices 0-20) to a 3D RGB image.

    Args:
    - image: 2D NumPy array of shape (H, W). Each value is a class ID (0-20).
    - nc: Number of classes (21 for PASCAL VOC).
    """

    # ---------------------------------------------------------
    # 1. THE PALETTE (LOOKUP TABLE)
    # ---------------------------------------------------------
    # This array defines the color for every class ID.
    # Index 0 is Background (Black), Index 1 is Aeroplane (Red), etc.
    label_colors = np.array([
        (0, 0, 0),       # 0=background
        (128, 0, 0),     # 1=aeroplane
        (0, 128, 0),     # 2=bicycle
        (128, 128, 0),   # 3=bird
        (0, 0, 128),     # 4=boat
        # ... (remaining colors for all 21 classes)
        (0, 64, 128)     # 20=tv/monitor
    ])

    # ---------------------------------------------------------
    # 2. CREATE EMPTY CANVAS
    # ---------------------------------------------------------
    # Create 3 empty 2D arrays (Red, Green, Blue) of the same size as the input.
    # We use 'uint8' because images use 0-255 integers.
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    # ---------------------------------------------------------
    # 3. PAINT BY NUMBERS (THE LOOP)
    # ---------------------------------------------------------
    # Iterate through every possible class (0 to 20).
    for l in range(0, nc):

        # BOOLEAN MASKING:
        # Create a True/False grid finding where the current class 'l' exists.
        # Example: If l=3 (Bird), 'idx' is True only where pixels are birds.
        idx = image == l

        # ASSIGN COLORS:
        # Look up the color in our palette: label_colors[l] -> (R, G, B)
        # Apply that color ONLY to the pixels found in 'idx'.
        # Python applies this to thousands of pixels instantly (Vectorization).
        r[idx] = label_colors[l, 0] # Red component
        g[idx] = label_colors[l, 1] # Green component
        b[idx] = label_colors[l, 2] # Blue component

    # ---------------------------------------------------------
    # 4. STACK LAYERS
    # ---------------------------------------------------------
    # Combine the 3 separate color layers back into one RGB image.
    # Shape: (H, W) + (H, W) + (H, W) -> (H, W, 3)
    rgb = np.stack([r, g, b], axis=2)

    return rgb

def segment_image(model, image_path):
    """
    Loads an image, runs inference, and displays the original vs. segmentation map.
    """
    # Safety Check: Don't crash if the file is missing
    if not os.path.exists(image_path):
        print(f"❌ Error: Image {image_path} not found.")
        return

    # ---------------------------------------------------------
    # STEP 1: LOAD & PREPROCESS
    # ---------------------------------------------------------
    # Load the image using PIL (Python Imaging Library).
    # .convert('RGB') ensures we have 3 channels (Red, Green, Blue)
    # even if the input is grayscale or has transparency (RGBA).
    img = Image.open(image_path).convert('RGB')

    # Define the Transformation Pipeline.
    # Models are trained on very specific data shapes. We must match that.
    transform = T.Compose([
        T.Resize(640), # Resize to 640px to prevent GPU memory overflow on large images.
        T.ToTensor(),  # Convert PIL Image (0-255) to PyTorch Tensor (0.0-1.0).
        # Normalize using ImageNet statistics. The model "expects" these
        # mean/std values. Without this, colors will look wrong to the AI.
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transforms and add the Batch Dimension.
    # shape: [3, H, W] -> [1, 3, H, W]
    # The model expects a batch of images, even if it's just one.
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # ---------------------------------------------------------
    # STEP 2: INFERENCE (THE AI PART)
    # ---------------------------------------------------------
    model.to(DEVICE) # Ensure model is on the GPU/MPS
    model.eval()     # Set to Evaluation mode (freezes BatchNorm/Dropout)

    with torch.no_grad(): # Disable gradient calculation (saves memory/speed)
        # Forward Pass. The model returns a dictionary.
        # We access ['out'] to get the raw predictions.
        # output shape: [1, 21, H, W] (Batch, Classes, Height, Width)
        output = model(input_tensor)['out']

    # ---------------------------------------------------------
    # STEP 3: PROCESS OUTPUT (ARGMAX)
    # ---------------------------------------------------------
    # The 'output' tensor contains 21 probability scores for EACH pixel.
    # We want to know: "Which of the 21 classes has the HIGHEST score?"
    # .squeeze() removes batch dim -> [21, H, W]
    # .argmax(dim=0) collapses 21 channels into 1 channel -> [H, W]
    # Result: A 2D grid where every pixel is an Integer ID (e.g., 3 for Bird).
    seg_map = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

    # ---------------------------------------------------------
    # STEP 4: VISUALIZATION
    # ---------------------------------------------------------
    # Convert the grid of IDs (0, 3, 15...) into colors (Black, Green, Red...)
    rgb_mask = decode_segmap(seg_map)

    # Plotting using Matplotlib
    plt.figure(figsize=(12, 6))

    # Left: Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Original Image")

    # Right: The Colored Segmentation Map
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_mask)
    plt.axis('off')
    plt.title("Segmentation Map")

    plt.show()

if __name__ == "__main__":
    setup_directories()
    print(f"Running on device: {DEVICE}")

    # 1. Load Model
    print("Loading FCN-ResNet101 Model...")
    fcn = models.segmentation.fcn_resnet101(
        weights=torchvision.models.segmentation.FCN_ResNet101_Weights.DEFAULT
    ).eval()

    # 2. Define Test Images
    images = {
        'bird.jpg': 'https://learnopencv.com/wp-content/uploads/2022/10/bird.jpg',
        'horse.jpg': 'https://www.learnopencv.com/wp-content/uploads/2021/01/horse-segmentation.jpeg',
        'person.jpeg': ''
    }

    # 3. Run Pipeline
    for filename, url in images.items():
        local_path = os.path.join(INFERENCE_DIR, filename)

        # Download (Safe Check Included)
        download_file(url, local_path)

        # Segment
        print(f"Segmenting {filename}...")
        segment_image(fcn, local_path)