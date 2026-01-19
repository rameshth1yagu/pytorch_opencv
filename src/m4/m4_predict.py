"""
m4_predict.py

Simple script to load a trained MonkeyClassifier and run predictions on a
small random sample of images from the validation set.

The script performs:
- Safe model loading (handles missing files)
- Image preprocessing that matches validation transforms used during training
- Batch-less prediction (one image at a time) and plotting of results
"""

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import glob

# Import model class so we can load the structure
from m4_cnn_model import MonkeyClassifier, DEVICE

# --- CONFIGURATION ---
MODEL_PATH = "model/m4/best_monkey_model.pth" # Or "best.pt" if you renamed it
VALIDATION_PATH = "data/10_Monkey_Species/validation/validation"

# Class names must match the folder indices (n0 -> index 0, n1 -> index 1)
CLASS_NAMES = [
    "mantled_howler", "patas_monkey", "bald_uakari", "japanese_macaque",
    "pygmy_marmoset", "white_headed_capuchin", "silvery_marmoset",
    "common_squirrel_monkey", "black_headed_night_monkey", "nilgiri_langur"
]

def load_smart_model():
    """Checks for model file and loads it safely.

    Returns:
    - model (nn.Module) or None if loading failed
    """
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found.")
        return None

    print(f"✅ Found model: {MODEL_PATH}")
    model = MonkeyClassifier(num_classes=len(CLASS_NAMES))

    # Load weights (map_location handles GPU->CPU translation if needed)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval() # CRITICAL: Freezes Dropout and BatchNorm
    return model

def process_image(image_path):
    """Prepares a single image for the model.

    Returns:
    - img_tensor: Tensor ready to be sent to the model (1, C, H, W)
    - img: original PIL.Image for visualization
    """
    img = Image.open(image_path).convert('RGB')

    # Same transforms as validation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4368, 0.4336, 0.3294],
                             std=[0.2457, 0.2413, 0.2447])
    ])

    img_tensor = transform(img).unsqueeze(0) # Add batch dimension
    return img_tensor, img

def get_true_label(file_path):
    """
    Extracts the true label from the folder name.
    Example path: .../validation/n3/n3015.jpg
    'n3' means index 3 -> 'japanese_macaque'
    """
    parent_folder = os.path.basename(os.path.dirname(file_path)) # e.g., "n3"
    try:
        class_idx = int(parent_folder[1:]) # remove 'n', keep '3'
        return CLASS_NAMES[class_idx]
    except:
        return "Unknown"

if __name__ == "__main__":
    # 1. Load Model
    model = load_smart_model()

    if model:
        # 2. Find all images
        print(f"Scanning {VALIDATION_PATH}...")
        # Recursively find all .jpg files
        all_images = glob.glob(os.path.join(VALIDATION_PATH, "*", "*.jpg"))

        if len(all_images) < 10:
            print(f"Not enough images found! Found {len(all_images)}, need 10.")
        else:
            # 3. Pick 10 Random Images
            selected_images = random.sample(all_images, 10)

            # Setup Plotting Grid (2 rows, 5 columns)
            fig, axes = plt.subplots(2, 5, figsize=(15, 7))
            fig.suptitle('Random Batch Predictions', fontsize=16)
            axes = axes.flatten()

            for i, img_path in enumerate(selected_images):
                # Process
                tensor, original_img = process_image(img_path)
                true_label = get_true_label(img_path)

                # Predict
                tensor = tensor.to(DEVICE)
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred_idx = torch.max(probs, 1)

                pred_label = CLASS_NAMES[pred_idx.item()]
                confidence_score = conf.item() * 100

                # Visualize
                ax = axes[i]
                ax.imshow(original_img)
                ax.axis('off')

                # Color code title: Green if Correct, Red if Wrong
                color = 'green' if pred_label == true_label else 'red'

                title_text = f"True: {true_label}\nPred: {pred_label}\n({confidence_score:.1f}%)"
                ax.set_title(title_text, color=color, fontsize=9)

            plt.tight_layout()
            plt.show()
            print("Done! Check the popup window for results.")