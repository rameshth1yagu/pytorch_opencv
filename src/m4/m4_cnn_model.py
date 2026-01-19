"""
m4_cnn_model.py

Convolutional Neural Network definition used to classify the 10 Monkey
Species dataset.

This file exposes:
- DEVICE: chosen computation device (CUDA, MPS, or CPU)
- MonkeyClassifier: nn.Module implementing the convolutional body, pooling,
  and classifier head used by training and inference scripts.

Notes:
- The forward pass contains a small workaround for macOS MPS where
  AdaptiveAvgPool2d may be unreliable for some tensor shapes: the code
  moves tensors to CPU for pooling and then back to the selected device.
"""

import torch
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation

# Define Device inside the model file so the model knows where to live
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class MonkeyClassifier(nn.Module):
    """
    Simple CNN classifier with three main parts:
    1) features: convolutional feature extractor (stacked conv, BN, ReLU, pools)
    2) avgpool: adaptive pooling to a fixed spatial size
    3) classifier: fully-connected head that maps features to class logits

    The architecture matches the saved model weights used elsewhere in
    this project (no extra dropout or activation between the final
    linear layers, so saved-state compatibility is preserved).
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # ------------------------------------------------------------
        # PART 1: The Feature Extractor (Convolutional Body)
        # Goal: Turn raw pixels into meaningful spatial features.
        # ------------------------------------------------------------
        self.features = nn.Sequential(
            # Block 1: Edges & Simple Shapes
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5), # 5x5 kernel looks at broad shapes
            nn.BatchNorm2d(32),        # Stabilizes math
            nn.ReLU(inplace=True),     # Adds non-linearity

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # Shrinks image size by half

            # Block 2: Textures & Parts (LazyConv2d figures out input size automatically)
            nn.LazyConv2d(out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.LazyConv2d(out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: Complex Objects (Ears, Eyes, Noses)
            # Conv2dNormActivation is a shortcut for Conv+Batch+ReLU
            Conv2dNormActivation(in_channels=128, out_channels=256, kernel_size=3),
            Conv2dNormActivation(in_channels=256, out_channels=256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),

            Conv2dNormActivation(in_channels=256, out_channels=512, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
        )

        # ------------------------------------------------------------
        # PART 2: The Pooling Layer (The "Mac Fix" Zone)
        # Goal: Reduce feature map size to a fixed 3x3 grid
        # ------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(3, 3))

        # ------------------------------------------------------------
        # PART 3: The Classifier (The Head)
        # Goal: Make a decision based on the extracted features.
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # PART 3: The Classifier (MATCHING YOUR SAVED FILE)
        # ------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 * 3 * 3, out_features=256),
            # nn.ReLU(),         <-- REMOVE OR COMMENT OUT
            # nn.Dropout(p=0.5), <-- REMOVE OR COMMENT OUT
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Steps:
        1) Extract features with the convolutional body on the selected device.
        2) Apply adaptive average pooling. On macOS/MPS this operation can be
           unreliable for some shapes, so we move tensors temporarily to CPU
           for pooling and then move them back to the target DEVICE.
        3) Run the classifier head to produce raw logits (un-normalized scores)
           for each class.

        Inputs:
        - x: float Tensor of shape (N, 3, H, W)

        Returns:
        - logits: Tensor of shape (N, num_classes)
        """
        # 1. Run the features on the GPU/MPS
        x = self.features(x)

        # 2. THE MAC FIX: Move to CPU for the pooling operation
        # AdaptiveAvgPool2d is broken on MPS (Mac) for some sizes.
        x = x.cpu()
        x = self.avgpool(x)
        x = x.to(DEVICE)  # Move back to GPU/MPS

        # 3. Run classification on the GPU/MPS
        x = self.classifier(x)
        return x

