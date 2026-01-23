"""
Object Detection with Faster R-CNN
==================================
This script demonstrates Object Detection: identifying WHAT objects are in an image
and WHERE they are (using Bounding Boxes).

Model: Faster R-CNN with ResNet50 Backbone + FPN (Feature Pyramid Network).
Weights: Pre-trained on COCO Dataset (80 classes: Person, Car, Dog, etc.).
"""

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import requests

# --- CONFIGURATION ---
# Fix random seed for consistent colors in visualization
np.random.seed(20)

# Standard COCO Dataset Classes (The "Dictionary" for the model)
# The model outputs an integer (e.g., 1). This list maps 1 -> 'person'.
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

# Generate random unique colors for each class ID so we can distinguish them visually.
# Format: Array of shape (91, 3) with values 0-255
COLORS = np.random.uniform(0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))

def load_model():
    """
    Loads the Faster R-CNN model with ResNet50-FPN backbone.
    """
    print("Loading Faster R-CNN Model...")
    # 'DEFAULT' weights = best available pre-trained weights on COCO
    # This automatically downloads the ResNet50 backbone + FPN layers.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    # Set to Evaluation Mode. Crucial!
    # Disables Dropout and freezes BatchNorm stats to ensure deterministic results.
    model.eval()
    return model

def get_prediction(model, img_path, threshold):
    """
    Runs the model on a single image and filters weak predictions.

    Args:
        model: The loaded Faster R-CNN model.
        img_path (str): Path to image file.
        threshold (float): Minimum confidence score (0.0 to 1.0).
                           Predictions below this are discarded.

    Returns:
        pred_boxes (list): List of coordinates [(x1, y1), (x2, y2)]
        pred_class (list): List of class names ["person", "car", ...]
    """
    # 1. Load Image
    img = Image.open(img_path).convert('RGB')

    # 2. Transform to Tensor
    # Faster R-CNN expects a Tensor [C, H, W] in range 0-1.
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img)

    # 3. Inference
    # The model expects a LIST of images. We pass a list with one image.
    # We do not need gradients since we are only predicting, not training.
    with torch.no_grad():
        pred = model([img_tensor])

    # 4. Process Output
    # The output 'pred' is a list of dictionaries (one per input image).
    # We grab the first (and only) dictionary.
    pred_data = pred[0]

    # Extract tensors and move to CPU/Numpy for processing
    pred_scores = pred_data['scores'].detach().cpu().numpy()
    pred_boxes_raw = pred_data['boxes'].detach().cpu().numpy()
    pred_labels_raw = pred_data['labels'].detach().cpu().numpy()

    # 5. Filter by Threshold
    # Create a boolean mask where score > threshold
    valid_indices = pred_scores >= threshold

    # Filter the arrays using the mask
    # We convert boxes to tuples of tuples: ((x1, y1), (x2, y2)) for easier drawing later
    pred_boxes = [((int(b[0]), int(b[1])), (int(b[2]), int(b[3])))
                  for b in pred_boxes_raw[valid_indices]]

    # Map Integer IDs (e.g., 1) to Strings (e.g., "person")
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in pred_labels_raw[valid_indices]]

    return pred_boxes, pred_class

def detect_and_visualize(model, img_path, threshold=0.5):
    """
    Main pipeline: Predict -> Draw -> Show.
    Uses OpenCV for drawing boxes and Matplotlib for display.
    """
    if not os.path.exists(img_path):
        print(f"❌ Error: Image {img_path} not found.")
        return

    print(f"Detecting objects in {os.path.basename(img_path)} (Threshold: {threshold})...")

    # 1. Get Predictions
    boxes, pred_cls = get_prediction(model, img_path, threshold)
    print(f"   Found {len(boxes)} objects.")

    # 2. Prepare Image for Drawing (OpenCV)
    # OpenCV loads images as BGR (Blue-Green-Red). 
    # Matplotlib needs RGB. We convert it to ensure colors are correct.
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Dynamic Styling
    # Calculate line thickness based on image size so boxes look good on 4K AND 240p.
    # Formula: Thickness is approx 0.3% of the image average dimension.
    rect_th = max(round(sum(img.shape) / 2 * 0.003), 2)
    text_th = max(rect_th - 1, 1)

    # 4. Draw Boxes Loop
    for i in range(len(boxes)):
        # Box coordinates
        p1 = boxes[i][0] # Top-Left (x1, y1)
        p2 = boxes[i][1] # Bottom-Right (x2, y2)

        # Get Class Name and corresponding random Color
        class_name = pred_cls[i]
        color = COLORS[COCO_INSTANCE_CATEGORY_NAMES.index(class_name)]

        # Draw Rectangle
        cv2.rectangle(img, p1, p2, color=color, thickness=rect_th)

        # 5. Draw Text Label
        # Calculate text size to create a background box for readability
        label_text = class_name
        label_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX,
                                               fontScale=rect_th/3, thickness=text_th)
        w, h = label_size

        # Smart Text Positioning:
        # Check if there is space ABOVE the box to put the text. 
        # If not (e.g., object is at top edge), put text INSIDE the box.
        outside = p1[1] - h >= 3
        if outside:
            text_p2 = (p1[0] + w, p1[1] - h - 3)
            text_origin = (p1[0], p1[1] - 5)
        else:
            text_p2 = (p1[0] + w, p1[1] + h + 3)
            text_origin = (p1[0], p1[1] + h + 2)

        # Draw filled rectangle behind text for better contrast
        cv2.rectangle(img, p1, text_p2, color=color, thickness=-1) # -1 means filled

        # Write Text (White Color)
        cv2.putText(img, label_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=rect_th/3, color=(255, 255, 255), thickness=text_th, lineType=cv2.LINE_AA)

    # 6. Display Result
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Detection Results (Threshold: {threshold})")
    plt.show()

def download_file(url, save_name):
    """
    Downloads a file ONLY if it does not already exist.
    Prevents repeated downloads of large image files.
    """
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
    # 1. Setup Data Directory
    inference_dir = 'inference_data'
    os.makedirs(inference_dir, exist_ok=True)

    # 2. Download Test Image (People walking)
    img_name = 'people.jpg'
    img_url = 'https://learnopencv.com/wp-content/uploads/2022/10/people.jpg'
    local_path = os.path.join(inference_dir, img_name)
    download_file(img_url, local_path)

    # 3. Load Model
    model = load_model()

    # 4. Run Detection
    # Threshold Note:
    # 0.1 = Finds everything (lots of mistakes)
    # 0.8 = Finds only very clear objects
    detect_and_visualize(model, local_path, threshold=0.8)