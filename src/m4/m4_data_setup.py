"""
m4_data_setup.py

Utilities for preparing the 10-Monkey-Species dataset and creating
PyTorch DataLoaders used by the CNN training pipeline.

This module provides:
- Configuration constants (data root, batch size)
- A deterministic seed helper for reproducibility
- A simple downloader/unzip utility for the dataset
- Transform pipelines for training and validation
- A pair of DataLoaders (train and validation)

Notes:
- NUM_WORKERS is set to 0 by default to avoid multiprocessing spawn errors on
  macOS and Windows when running from some environments (e.g., notebooks).
- The transforms use precomputed mean/std for this dataset; normalization
  must be applied after converting images to tensors.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import urllib.request
import zipfile
import random
import numpy as np

# --- 1. CONFIGURATION ---
# Where the dataset will be stored and basic DataLoader settings.
DATA_ROOT = "./data"
BATCH_SIZE = 32
NUM_WORKERS = 0  # Set to 0 to be safe on Mac/Windows (avoids the spawn error)

# --- 2. REPRODUCIBILITY ---
def set_seed(seed=42):
    """
    Set random seeds for Python, NumPy, and PyTorch to help produce
    repeatable runs.

    Note: For full determinism with CUDA one would normally set
    torch.backends.cudnn.deterministic = True and torch.backends.cudnn.benchmark = False.
    This code leaves `benchmark` True when CUDA is available to retain
    performance; remove that line for absolute determinism.

    Inputs:
    - seed (int): integer seed to use for all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

# --- 3. DOWNLOAD UTILITY ---
def download_and_unzip():
    """
    Downloads the 10_Monkey_Species dataset (from a hosted URL) and
    extracts it under DATA_ROOT if it is not already present.

    This is a convenience helper so the training script can ensure data
    exists before creating DataLoaders.
    """
    url = 'https://www.dropbox.com/s/45jdd8padeyjq6t/10_Monkey_Species.zip?dl=1'
    zip_path = os.path.join(DATA_ROOT, '10_Monkey_Species.zip')
    extract_path = os.path.join(DATA_ROOT, '10_Monkey_Species')

    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)

    if not os.path.exists(extract_path):
        print(f"Downloading dataset to {zip_path}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete. Unzipping...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_ROOT)
            print("Unzip complete.")
        except Exception as e:
            print(f"Error downloading data: {e}")
    else:
        print("Dataset already found.")

# --- 4. TRANSFORMS (PREPROCESSING) ---
def get_transforms():
    """
    Build and return train and validation torchvision transforms.

    - train_transforms: includes augmentation (horizontal flip, random
      erasing, occasional affine transforms) to help generalization.
    - val_transforms: deterministic resize + normalization.

    Returns:
    - (train_transforms, val_transforms)
    """
    # Stats calculated for this specific dataset (Mean & Std Deviation)
    mean = [0.4368, 0.4336, 0.3294]
    std = [0.2457, 0.2413, 0.2447]
    img_size = (224, 224) # Standard input size for CNNs

    # Base steps needed for ALL images
    base_transform = [
        transforms.Resize(img_size, antialias=True),
        transforms.ToTensor(),
    ]

    # Normalization (must happen after ToTensor)
    norm = transforms.Normalize(mean=mean, std=std)

    # 1. Validation Pipeline (Clean)
    val_transforms = transforms.Compose(base_transform + [norm])

    # 2. Training Pipeline (Augmented)
    # We create "fake" new images by flipping and erasing parts of real ones.
    # This forces the model to learn features, not just memorize pixels.
    train_transforms = transforms.Compose(base_transform + [
        transforms.RandomHorizontalFlip(p=0.5), # Mirror image
        transforms.RandomErasing(p=0.4),        # Cut out a random square
        transforms.RandomApply([
            transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
        ], p=0.1),
        norm
    ])

    return train_transforms, val_transforms

# --- 5. DATA LOADERS ---
def get_data_loaders():
    """
    Create and return training and validation DataLoaders based on the
    established DATA_ROOT structure produced by the download helper.

    Returns:
    - train_loader (DataLoader)
    - val_loader (DataLoader)
    - class_names (list[str]): ordered class names matching folder indices
    """
    train_dir = os.path.join(DATA_ROOT, "10_Monkey_Species", "training", "training")
    val_dir = os.path.join(DATA_ROOT, "10_Monkey_Species", "validation", "validation")

    train_tf, val_tf = get_transforms()

    # ImageFolder automatically labels images based on the folder name
    train_data = datasets.ImageFolder(root=train_dir, transform=train_tf)
    val_data = datasets.ImageFolder(root=val_dir, transform=val_tf)

    # DataLoader handles batching (grouping images) and shuffling
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,       # Shuffle training data to break order bias
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,      # Never shuffle validation data (we want consistent tests)
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, train_data.classes

# Helper to run this file alone
if __name__ == "__main__":
    download_and_unzip()
    tr, val, classes = get_data_loaders()
    print(f"Classes found: {classes}")
    print(f"Training batches: {len(tr)}")