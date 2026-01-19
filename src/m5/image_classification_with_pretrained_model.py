"""
ResNet18 Image Classification Pipeline
======================================

This script demonstrates how to use Transfer Learning for inference.
It uses a pre-trained ResNet18 model (trained on ImageNet) to classify
images into one of 1000 categories.

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
    """
    Checks if data exists. If not, downloads and unzips the assets.
    This ensures the script can run on any machine without manual setup.
    """
    zip_path = os.path.join(os.getcwd(), DATA_DIR, "PyTorch-Bootcamp-NB07-assets.zip")

    # Create data directory if it doesn't exist
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
    """
    Returns the specific transformation pipeline required by ResNet.

    Deep Learning models expect input to be in a very specific format.
    For ResNet trained on ImageNet, we must:
    1. Resize to 256x256.
    2. Crop the center 224x224 (The Standard Input Size).
    3. Convert pixel values (0-255) to Tensors (0-1).
    4. Normalize using the exact Mean and Std Dev of the ImageNet dataset.
    """
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
    """
    Downloads and loads the ResNet18 model with pre-trained ImageNet weights.

    Returns:
        model: The PyTorch model in evaluation mode.
    """
    print("Loading ResNet18 Model...")
    # 'IMAGENET1K_V1' means we use the best available weights from the 1st version of ImageNet training.
    model = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

    # CRITICAL: Switch to 'eval' mode.
    # This freezes layers like Dropout and BatchNorm so they behave consistently during inference.
    model.eval()
    return model

def load_class_names(file_path):
    """
    Reads the text file containing the 1000 class names (e.g., 'goldfish', 'baseball').

    Returns:
        list: A list of strings where index 0 matches the model's output 0.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Class file not found at {file_path}. Run download first.")

    with open(file_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def predict_single_image(image_path, model, transform, class_names):
    """
    Runs inference on a single image file.

    Steps:
    1. Load Image.
    2. Transform (Resize, Normalize).
    3. Batch (Add dimension).
    4. Predict (Forward pass).
    5. Interpret (Softmax -> Probabilities).

    Returns:
        original_img: The PIL image (for visualization).
        pred_label: The string name of the top class.
        confidence: The confidence score (0-100).
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None, None, None

    # 1. Load Image
    img = Image.open(image_path).convert('RGB')

    # 2. Transform & 3. Batch
    # unsqueeze(0) turns shape [3, 224, 224] into [1, 3, 224, 224]
    img_tensor = transform(img).unsqueeze(0)

    # 4. Predict
    # We use no_grad() because we don't need to calculate gradients for backprop (saves memory)
    with torch.no_grad():
        logits = model(img_tensor)

    # 5. Interpret
    # Softmax converts raw logits into probabilities (percentages)
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0] * 100

    # Get the index of the highest score
    _, indices = torch.sort(logits, descending=True)
    top_idx = indices[0][0].item()

    pred_label = class_names[top_idx]
    confidence = probabilities[top_idx].item()

    return img, pred_label, confidence

def visualize_result(img, class_name, conf):
    """
    Draws the prediction text on the image and displays it using Matplotlib.
    """
    if img is None: return

    # Convert PIL Image to Numpy Array (OpenCV uses Numpy)
    # Convert RGB to BGR because OpenCV expects BGR
    bgr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_h, img_w = bgr_img.shape[:2]

    # Dynamic font scaling based on image size
    font_scale = max(0.003 * img_h, 0.5)
    thickness = max(1, int(img_h / 200))

    text = f"{class_name.split(',')[0]}: {conf:.1f}%"

    # Calculate text position to center it near the top
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = (img_w - text_w) // 2
    text_y = (img_h + text_h) // 10 + 20

    # Draw Text
    cv2.putText(
        img=bgr_img,
        org=(text_x, text_y),
        text=text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        color=(0, 0, 255), # Red color
        fontScale=font_scale,
        thickness=thickness,
        lineType=cv2.LINE_AA
    )

    # Convert back to RGB for Matplotlib display
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.title(f"Prediction: {text}")
    plt.show()

def main():
    """
    Main execution function. Orchestrates the flow.
    """
    # 1. Setup Data
    download_and_setup_data()

    # 2. Setup Resources
    model = load_model()
    transform = get_resnet_transforms()
    class_names = load_class_names(CLASS_FILE_PATH)

    print(f"\nProcessing images from: {IMAGES_DIR}")

    # 3. Loop through all images in the folder
    if os.path.exists(IMAGES_DIR):
        image_files = os.listdir(IMAGES_DIR)

        for img_file in image_files:
            # Skip non-image files
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            full_path = os.path.join(IMAGES_DIR, img_file)
            print(f"Predicting: {img_file}...")

            # Run Prediction
            img, label, conf = predict_single_image(full_path, model, transform, class_names)

            # Show Result
            visualize_result(img, label, conf)
    else:
        print(f"Error: Image directory '{IMAGES_DIR}' not found.")

if __name__ == "__main__":
    main()