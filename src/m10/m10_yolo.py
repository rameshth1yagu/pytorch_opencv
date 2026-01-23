"""
YOLOv11 Object Detection Inference
==================================
This script demonstrates how to run inference using the Ultralytics YOLOv11 model.

Key Concepts:
1. Model Loading: Downloading the pre-trained weights.
2. Inference: Running the model on an image.
3. Visualization: Using .plot() to render results in-memory (Avoiding disk I/O errors).
4. BGR vs RGB: Handling OpenCV's color format correctly.
"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import requests
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
# Use the Nano model ('yolo11n.pt') for speed, or Large ('yolo11l.pt') for accuracy.
MODEL_TYPE = "yolo11n.pt"
IMG_NAME = "football.jpg"
IMG_URL = "https://learnopencv.com/wp-content/uploads/2024/08/soccer-scaled.jpg"

def download_file(url, save_name):
    """
    Downloads a file safely.
    Checks if the file exists AND if it is a valid image.
    Deletes corrupt files to allow re-downloading.
    """
    # 1. Validation Check
    if os.path.exists(save_name):
        try:
            # Try opening the image to prove it's not corrupt
            with Image.open(save_name) as img:
                img.verify()
            print(f"✅ Found valid image: {save_name}, skipping download.")
            return
        except Exception:
            print(f"⚠️  File {save_name} is corrupt. Deleting and re-downloading...")
            os.remove(save_name)

    # 2. Download
    print(f"⬇️  Downloading {save_name}...")
    try:
        # User-Agent header tricks servers into thinking we are a browser, avoiding 403 errors
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        r.raise_for_status() # Crash immediately if URL is bad (404)

        with open(save_name, 'wb') as f:
            f.write(r.content)
        print("   Download complete.")
    except Exception as e:
        print(f"   ❌ Error downloading: {e}")

def prediction(model_type, img_path, display_result=False, task="Object Detection"):
    """
    Runs YOLO inference and displays the result using Matplotlib.
    """
    # 1. Path Safety Check
    if not os.path.exists(img_path):
        print(f"❌ Error: File not found at {os.path.abspath(img_path)}")
        return

    # 2. Load Model
    # The first time this runs, it will download the weights file (e.g. yolo11n.pt)
    print(f"Loading {model_type}...")
    model = YOLO(model_type)

    # 3. Run Inference
    # We do NOT use 'save=True'. That causes the numpy/path errors you saw.
    # We just want the data in memory.
    print("Running inference...")
    results = model(img_path, conf=0.5)

    # 4. Process Results
    # YOLO returns a LIST of result objects (one per image).
    for r in results:

        # --- A. Rendering ---
        # r.plot() draws the bounding boxes and labels onto the image
        # and returns a NumPy array. This happens entirely in RAM.
        im_array = r.plot()

        # --- B. Color Correction ---
        # OpenCV/YOLO uses BGR (Blue-Green-Red).
        # Matplotlib uses RGB (Red-Green-Blue).
        # We must convert it, otherwise blue things look red.
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

        # --- C. Display ---
        plt.figure(figsize=(10, 8))
        plt.imshow(im_rgb)
        plt.axis('off') # Hide x/y pixel rulers
        plt.title(f"YOLO11 - {task}")
        plt.show()

        # --- D. Print Data (Optional) ---
        if display_result:
            print("\n--- Detections ---")
            # r.boxes.data format: [x1, y1, x2, y2, confidence, class_id]
            for box in r.boxes:
                class_id = int(box.cls[0])
                name = r.names[class_id]
                conf = float(box.conf[0])
                print(f"Found {name} ({conf:.2f} confidence)")

if __name__ == "__main__":
    # 1. Setup Data
    download_file(IMG_URL, IMG_NAME)

    # 2. Run Pipeline
    prediction(MODEL_TYPE, IMG_NAME, display_result=True)