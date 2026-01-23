"""
Instance Segmentation with Mask R-CNN
=====================================
This script performs Instance Segmentation: identifying objects, drawing bounding boxes,
AND finding the precise pixel-wise mask for each object.

Model: Mask R-CNN (ResNet50 Backbone + FPN).
Weights: Pre-trained on COCO Dataset (80 classes).
"""

import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image
import os
import requests

# --- CONFIGURATION ---
# Seed for reproducibility of colors
np.random.seed(42)

# COCO Class Names (Standard list for pre-trained models)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Generate random colors for visualization (one for each class)
COLORS = np.random.uniform(0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))

def load_model():
    """
    Loads the pre-trained Mask R-CNN model.
    """
    print("Loading Mask R-CNN Model...")
    # Load model with DEFAULT weights (COCO pre-trained)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    # Set to evaluation mode (crucial for inference consistency)
    model.eval()
    return model

def random_color_masks(mask, color):
    """
    Applies a specific color to the binary mask.

    Args:
        mask (numpy array): Binary mask where 1=Object, 0=Background
        color (list): RGB color values [R, G, B]

    Returns:
        colored_mask: An RGB image where the object pixels are colored.
    """
    # Create 3 empty channels (R, G, B) matching the image size
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)

    # Fill the channels with the color ONLY where the mask is 1 (True)
    r[mask == 1], g[mask == 1], b[mask == 1] = color

    # Stack channels to create a standard RGB image
    colored_mask = np.stack([r, g, b], axis=2)
    return colored_mask

def get_prediction(model, img_path, threshold):
    """
    Runs inference on the image to get masks and boxes.

    Returns:
        masks: Binary masks for detected objects
        boxes: Bounding box coordinates
        pred_cls: Class names
    """
    # 1. Load and Transform
    img = Image.open(img_path).convert('RGB')
    transform = T.Compose([T.ToTensor()]) # Convert PIL -> Tensor (0-1 range)
    img_tensor = transform(img)

    # 2. Inference
    # .cpu() ensures everything runs on CPU if GPU is not available (simplifies logic)
    # Ideally, move model to GPU if available.
    with torch.no_grad():
        pred = model([img_tensor])

    # 3. Process Outputs
    pred_data = pred[0]

    # Extract raw data
    pred_scores = pred_data['scores'].detach().cpu().numpy()
    pred_labels = pred_data['labels'].detach().cpu().numpy()
    pred_boxes = pred_data['boxes'].detach().cpu().numpy()

    # Process Masks:
    # The model outputs "soft" masks (probabilities 0.0 to 1.0).
    # We threshold at 0.5 to make them "hard" binary masks (0 or 1).
    # .squeeze() removes the channel dimension: [N, 1, H, W] -> [N, H, W]
    pred_masks = (pred_data['masks'] > 0.5).squeeze().detach().cpu().numpy()

    # 4. Filter by Confidence Threshold
    # We only keep predictions where the model is confident (score > threshold)
    valid_indices = pred_scores >= threshold

    # Return filtered lists
    masks = pred_masks[valid_indices]
    boxes = pred_boxes[valid_indices]
    # Map Integer IDs to Strings
    pred_cls = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in pred_labels[valid_indices]]

    return masks, boxes, pred_cls

def detect_and_visualize(model, img_path, threshold=0.5):
    """
    Main visualization pipeline.
    Combines original image + Bounding Boxes + Colored Masks.
    """
    if not os.path.exists(img_path):
        print(f"❌ Error: Image {img_path} not found.")
        return

    print(f"Segmenting {os.path.basename(img_path)} (Threshold: {threshold})...")

    # 1. Get Predictions
    masks, boxes, pred_cls = get_prediction(model, img_path, threshold)
    print(f"   Found {len(masks)} objects.")

    # 2. Prepare Image for Drawing (OpenCV)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB for Matplotlib

    # 3. Dynamic Styling
    rect_th = max(round(sum(img.shape) / 2 * 0.003), 2)
    text_th = max(rect_th - 1, 1)

    # 4. Loop through every detected object
    for i in range(len(masks)):

        # --- A. DRAW THE MASK ---
        # Get a random color for this object
        color = COLORS[COCO_INSTANCE_CATEGORY_NAMES.index(pred_cls[i])]

        # Create a colored mask image
        rgb_mask = random_color_masks(masks[i], color)

        # Blend the colored mask with the original image.
        # cv2.addWeighted calculates: (img * alpha) + (mask * beta) + gamma
        # 0.5 beta means the mask is 50% transparent.
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)

        # --- B. DRAW THE BOX ---
        # Coordinates
        p1 = (int(boxes[i][0]), int(boxes[i][1]))
        p2 = (int(boxes[i][2]), int(boxes[i][3]))

        # Draw Rectangle
        cv2.rectangle(img, p1, p2, color=color, thickness=rect_th)

        # --- C. DRAW THE LABEL ---
        class_name = pred_cls[i]
        label_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, rect_th/3, text_th)
        w, h = label_size

        # Handle label position (Inside vs Outside box)
        outside = p1[1] - h >= 3
        if outside:
            text_p2 = (p1[0] + w, p1[1] - h - 3)
            text_origin = (p1[0], p1[1] - 5)
        else:
            text_p2 = (p1[0] + w, p1[1] + h + 3)
            text_origin = (p1[0], p1[1] + h + 2)

        # Background for text
        cv2.rectangle(img, p1, text_p2, color=color, thickness=-1)
        # Text itself
        cv2.putText(img, class_name, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                    rect_th/3, (255, 255, 255), thickness=text_th, lineType=cv2.LINE_AA)

    # 5. Display Final Result
    plt.figure(figsize=(20, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Instance Segmentation (Threshold: {threshold})")
    plt.show()

def download_file(url, save_name):
    """Safe download utility."""
    if os.path.exists(save_name):
        print(f"✅ Found {save_name}, skipping download.")
        return
    print(f"⬇️  Downloading {save_name}...")
    try:
        r = requests.get(url)
        with open(save_name, 'wb') as f:
            f.write(r.content)
        print("   Download complete.")
    except Exception as e:
        print(f"   ❌ Error downloading: {e}")

if __name__ == "__main__":
    # 1. Setup Data
    inference_dir = 'inference_data'
    os.makedirs(inference_dir, exist_ok=True)

    # 2. Download Image
    img_name = 'mrcnn_cars.jpg'
    img_url = 'https://learnopencv.com/wp-content/uploads/2022/10/mrcnn_cars-scaled.jpg'
    local_path = os.path.join(inference_dir, img_name)
    download_file(img_url, local_path)

    # 3. Load Model
    model = load_model()

    # 4. Run Instance Segmentation
    # Threshold 0.9 means we only want SUPER confident predictions.
    detect_and_visualize(model, local_path, threshold=0.9)