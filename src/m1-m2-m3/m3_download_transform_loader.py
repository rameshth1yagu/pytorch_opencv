import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary

import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import random
import time

SEED_VALUE = 42
#class to idx mapping
f_mnist_class_mapping = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"  }

# This ensures that if you change your code and the accuracy drops,
# you know your code caused it, not just bad luck with the random number generator
def seed_example():
    torch.manual_seed(SEED_VALUE)
    for i in range(5):
        print(torch.rand(2))

def set_seeds():
    # set random seed value

    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    # Fix seed to make training deterministic.
    if torch.cuda.is_available():
        print(f"Setting SEED for CUDA")
        torch.cuda.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed_all(SEED_VALUE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        print(f"Setting SEED for MPS")
        torch.mps.manual_seed(SEED_VALUE)
        #torch.mps.manual_seed_all(SEED_VALUE)
        torch.backends.mps.deterministic = True
        torch.backends.mps.benchmark = True

def raw_compose_transform_normalize():
    raw_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    raw_train_set = datasets.FashionMNIST(
        root="F_MNIST_data",
        train=True,
        download=True,
        transform=raw_transform
    )
    all_pixels = torch.cat([img.view(-1) for img, _ in raw_train_set], dim=0)
    mean = torch.mean(all_pixels).item()
    std = torch.std(all_pixels).item()
    print(f"Calculated mean: {mean}, std: {std}")
    return mean, std

def compose_transform_normalize(mean, std):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transform

def download_transform_data(transform):
    train_set = datasets.FashionMNIST(
        root="F_MNIST_data",
        train=True,
        download=True,
        transform=transform
    )
    test_set = datasets.FashionMNIST(
        root="F_MNIST_data",
        train=False,
        download=True,
        transform=transform
    )
    print(f"Train set: {len(train_set)}")
    print(f"Test set: {len(test_set)}")
    return train_set, test_set

def data_loader():
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    return train_loader, test_loader

def visualize_images(train_loader, num_images=20):
    fig = plt.figure(figsize=(10, 10))

    # Iterate over the first batch
    images, labels = next(iter(train_loader))

    #To calculate the number of rows and columns for subplots
    num_rows = 4
    num_cols = int(np.ceil(num_images / num_rows))

    for idx in range(min(num_images, len(images))):
        image, label = images[idx], labels[idx]

        ax = fig.add_subplot(num_rows, num_cols, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(image), cmap="gray")
        ax.set_title(f"{label.item()}:{f_mnist_class_mapping[label.item()]}")

    fig.tight_layout()
    plt.show()

def example_relu():
    x = torch.tensor([-1, -99, 1, 2, -6])
    y = F.relu(x)
    print(f"Input tensor: {x}")
    print(f"ReLU output tensor: {y}")

set_seeds()
mean, std = raw_compose_transform_normalize()
transformer = compose_transform_normalize(mean, std)
train_set, test_set = download_transform_data(transformer)
train_loader, test_loader = data_loader()
#visualize_images(train_loader, num_images=16)
example_relu()
