import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
import os
from download_transform_loader import data_loader

# --- 1. DEFINE THE ARCHITECTURE (MUST MATCH TRAINING EXACTLY) ---
class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc0 = nn.Linear(28 * 28, 512)
        self.bn0 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        # Note: bn3 is defined here to match saved weights, even if unused in forward
        self.fc3 = nn.Linear(128, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn0(self.fc0(x)))
        x = self.dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # No BN/ReLU on output
        x = F.log_softmax(x, dim=1)
        return x

# --- 2. CONFIGURATION ---
MODEL_PATH = "fashion_mnist_mlp.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# FashionMNIST Mapping (Index -> Label)
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def load_model():
    """Recreates the model structure and loads the saved weights."""
    print(f"Loading model from {MODEL_PATH}...")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Train the model first!")
        sys.exit()

    model = MLP(num_classes=10)

    # Load state dictionary
    # map_location ensures we can load a GPU model onto CPU if needed
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()  # CRITICAL: Switches off Dropout and locks BatchNorm
    print("Model loaded successfully!")
    return model

def preprocess_image(image_path):
    """
    Reads an image, converts to grayscale, resizes to 28x28,
    and applies the same Normalization as training.
    """
    print(f"Processing image: {image_path}")

    # 1. Read Image using OpenCV
    # IMREAD_GRAYSCALE ensures 1 channel
    img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img_cv is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None

    # 2. Resize to 28x28 (Model expects this exact size)
    img_resized = cv2.resize(img_cv, (28, 28))

    # 3. Inversion Logic (Optional but often needed)
    # FashionMNIST items are White on Black background.
    # Real photos are often Dark on Light background.
    # If the corner pixel is bright (>127), assume background is white and invert.
    if img_resized[0, 0] > 127:
        print("Detected light background. Inverting image...")
        img_resized = 255 - img_resized

    # 4. Convert to Tensor and Normalize
    # We used these exact stats in training
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(), # Scales [0, 255] -> [0.0, 1.0]
        transforms.Normalize((0.2860,), (0.3530,)) # Mean and Std from training
    ])

    img_tensor = transform(img_resized)

    # 5. Add Batch Dimension: [1, 28, 28] -> [1, 1, 28, 28]
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, img_resized

def predict(model, image_tensor):
    """Performs the prediction."""
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad(): # Disable gradient calculation
        output = model(image_tensor)

        # Convert Log-Probabilities to actual Probabilities (0-100%)
        probabilities = torch.exp(output)

        # Get the top 1 prediction
        top_p, top_class = probabilities.topk(1, dim=1)

        # Get top 3 for detailed output
        top3_p, top3_class = probabilities.topk(3, dim=1)

    return top_class.item(), top_p.item(), top3_class.cpu().numpy()[0], top3_p.cpu().numpy()[0]

def visualize_prediction(original_img, prediction_idx, confidence, top3_indices, top3_probs):
    """Shows the image and the prediction results."""
    label = CLASS_NAMES[prediction_idx]

    plt.figure(figsize=(8, 4))

    # Plot Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title(f"Pred: {label}\nConf: {confidence*100:.2f}%")
    plt.axis('off')

    # Plot Top 3 Probabilities
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(top3_indices))
    top3_labels = [CLASS_NAMES[i] for i in top3_indices]

    plt.barh(y_pos, top3_probs, align='center', color='skyblue')
    plt.yticks(y_pos, top3_labels)
    plt.xlabel('Probability')
    plt.title('Top 3 Guesses')
    plt.gca().invert_yaxis() # Highest probability on top

    plt.tight_layout()
    #plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(model, data_loader, device):
    print("\n[INFO] Generating predictions for confusion matrix...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(data_loader):
                print(f"  Processed batch {batch_idx+1}/{len(data_loader)}")

    print("[INFO] Computing confusion matrix...")
    # Generate the matrix
    cm = confusion_matrix(all_labels, all_preds)

    print("[INFO] Plotting confusion matrix...")
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    print("[INFO] Confusion matrix plot complete.")

# Usage (assuming you have the loader and model ready)
# plot_confusion_matrix(mlp, test_loader, device)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Model
    model = load_model()

    # 2. Define Image Path (Change this to your image file!)
    # You can upload a file named 'test_shoe.jpg' or similar
    TEST_IMAGE = "references/predict.png" # Using the one from previous context as placeholder

    # 3. Preprocess
    img_tensor, original_img = preprocess_image(TEST_IMAGE)

    if img_tensor is not None:
        # 4. Predict
        pred_idx, confidence, top3_idx, top3_prob = predict(model, img_tensor)

        print(f"\n--- PREDICTION RESULT ---")
        print(f"Winner: {CLASS_NAMES[pred_idx]} ({confidence*100:.2f}%)")
        print(f"Top 3 Candidates:")
        for i in range(3):
            print(f"  {i+1}. {CLASS_NAMES[top3_idx[i]]}: {top3_prob[i]*100:.2f}%")

        # 5. Show
        visualize_prediction(original_img, pred_idx, confidence, top3_idx, top3_prob)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        _, test_loader = data_loader()
        plot_confusion_matrix(model, test_loader, device)