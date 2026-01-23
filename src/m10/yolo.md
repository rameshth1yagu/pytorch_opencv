# Object Detection with YOLOv11

## 1. Core Concepts: What is YOLO?

**YOLO (You Only Look Once)** is a state-of-the-art object detection architecture. It revolutionized computer vision by changing the approach from "scanning" an image to "looking" at it once.

### The Key Difference

* **Old Way (R-CNN):** Treat detection as a classification problem. Crop thousands of regions, feed them to a CNN, and classify them. (Slow, accurate).
* **YOLO Way:** Treat detection as a **single regression problem**. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. (Fast, Real-time).

### How It Works (The Grid System)

1. **Grid Division:** YOLO divides the input image into an  grid.
2. **Responsibility:** If the **center** of an object falls into a grid cell, that cell is responsible for detecting that object.
3. **Prediction:** Each cell predicts:
* **B Bounding Boxes:** Coordinates .
* **Confidence Score:** How likely is it that a box contains an object?
* **Class Probabilities:** If there is an object, is it a Dog? A Car?



---

## 2. Setting Up YOLOv11

We use the `ultralytics` library, which provides the easiest interface for modern YOLO models.

### Installation

```bash
pip install ultralytics
# Note: Requires PyTorch to be installed first

```

### Model Types

* `yolo11n.pt`: **Nano**. Fastest, lowest accuracy. Good for mobile/video.
* `yolo11s.pt`: **Small**.
* `yolo11m.pt`: **Medium**.
* `yolo11l.pt`: **Large**. High accuracy, requires strong GPU.
* `yolo11x.pt`: **Extra Large**. SOTA accuracy, slowest.

---

## 3. The Output Structure: "The List"

A common source of confusion is the output format.

```python
results = model('image.jpg')

```

YOLO is built for batch processing. Even if you input **one** image, it returns a **list** of result objects.

**Hierarchy:**

* `results` (List)
* `results[0]` -> **`Results` Object** (Container for Image 1)
* `.boxes` -> The Detections (Coordinates, Confidence, Class ID).
* `.masks` -> Segmentation Masks (if using a `-seg` model).
* `.orig_img` -> The original raw image (numpy array).
* `.names` -> Dictionary mapping IDs to names `{0: 'person', ...}`.
* `.plot()` -> Helper method to visualize the result.





---

## 4. Robust Implementation (Code Snippet)

This implementation handles the common pitfalls:

1. **File Corruption:** It checks if the image is valid before processing.
2. **Numpy/Path Errors:** It avoids `save=True` and renders in-memory to prevent `ValueError: need at least one array to stack`.
3. **Color Space:** It converts OpenCV's BGR to Matplotlib's RGB.

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import requests
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
MODEL_TYPE = "yolo11n.pt" 
IMG_PATH = "football.jpg"
IMG_URL = "https://learnopencv.com/wp-content/uploads/2024/08/soccer-scaled.jpg"

def download_safe(url, save_name):
    """
    Downloads file only if it doesn't exist or is corrupt.
    """
    if os.path.exists(save_name):
        try:
            with Image.open(save_name) as img:
                img.verify() # Verify image integrity
            print(f"✅ Found valid image: {save_name}")
            return
        except:
            print(f"⚠️ Corrupt file found. Deleting: {save_name}")
            os.remove(save_name)

    print(f"⬇️ Downloading {save_name}...")
    # User-Agent prevents 403 Forbidden errors
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    with open(save_name, 'wb') as f:
        f.write(r.content)

def run_inference():
    # 1. Load Model
    print(f"Loading {MODEL_TYPE}...")
    model = YOLO(MODEL_TYPE)
    
    # 2. Run Inference
    # conf=0.5: Ignore weak detections (below 50% confidence)
    results = model(IMG_PATH, conf=0.5)

    # 3. Process Results
    for r in results:
        # A. Render in Memory (Returns BGR Numpy Array)
        im_bgr = r.plot() 
        
        # B. Convert BGR -> RGB for Matplotlib
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        
        # C. Display
        plt.figure(figsize=(10, 8))
        plt.imshow(im_rgb)
        plt.axis('off')
        plt.title(f"YOLO Detection Results")
        plt.show()

        # D. Print Text Data
        print("\n--- Detections ---")
        for box in r.boxes:
            # box.cls gives a tensor (e.g., tensor([0.])), we cast to int
            cls_id = int(box.cls[0])
            name = r.names[cls_id]
            conf = float(box.conf[0])
            print(f"Found {name} ({conf:.2f})")

if __name__ == "__main__":
    download_safe(IMG_URL, IMG_PATH)
    run_inference()

```

## 5. Common Pitfalls Checklist

| Error / Issue | Cause | Solution |
| --- | --- | --- |
| `ValueError: need at least one array to stack` | Using `save=True` and trying to read the file immediately, or bad file paths. | Remove `save=True`. Use `r.plot()` to get the image in memory. |
| **Blue/Weird Colors** | Displaying an OpenCV image (BGR) directly in Matplotlib (RGB). | Use `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`. |
| **"File Not Found"** | Assuming `save=True` always saves to the same path (e.g. `predict/`). YOLO increments folders (`predict2/`). | Don't rely on hardcoded output paths. Use the returned `results` object. |
| **Empty Downloads** | `requests.get` blocked by server security. | Add `headers={'User-Agent': 'Mozilla/5.0'}` to your request. |