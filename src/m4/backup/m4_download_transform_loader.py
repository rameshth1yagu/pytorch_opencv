import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torchvision.ops import Conv2dNormActivation

from dataclasses import dataclass
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sn

import matplotlib.pyplot as plt
import time
import numpy as np
import random
import warnings
import os
from tqdm import tqdm

import pandas as pd
import urllib.request
import zipfile

from m4_cnn_model import MyModel

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
class_mapping = {
    0: "mantled_howler",
    1: "patas_monkey",
    2: "bald_uakari",
    3: "japanese_macaque",
    4: "pygmy_marmoset",
    5: "white_headed_capuchin",
    6: "silvery_marmoset",
    7: "common_squirrel_monkey",
    8: "black_headed_night_monkey",
    9: "nilgiri_langur"
}

@dataclass(frozen=True)
class TrainingConfig:
    ''' Configuration for Training '''
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4

    log_interval: int = 1
    test_interval: int = 1
    data_root: int = "./data"
    num_workers: int = 5
    device: str = DEVICE


#Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

# Download and store the dataset in 'data' folder
def download_data():
    data_url = 'https://www.dropbox.com/s/45jdd8padeyjq6t/10_Monkey_Species.zip?dl=1'
    data_path = '/data/10_Monkey_Species'

    if not os.path.exists('data'):
        os.makedirs('data')

    zip_path = 'data/10_Monkey_Species.zip'
    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(data_url, zip_path)
        print("Download complete.")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data')

def train_test_data():
    train_root = os.path.join("data", "10_Monkey_Species", "training", "training")
    val_root = os.path.join(train_config.data_root, "10_Monkey_Species", "validation", "validation")

    return train_root, val_root

def display_labels():
    df = pd.read_csv(os.path.join("data","10_Monkey_Species","monkey_labels.txt"), sep=",", header=None)
    df.columns = ["Label", "Latin Name", "Common Name", "Train Images", "Validation Images"]
    df['Latin Name'] = df['Latin Name'].str.replace("\t", " ")
    print(df)

def process_data():
    mean = [0.4368, 0.4336, 0.3294]  #mean and std of this Monkey Species dataset
    std = [0.2457, 0.2413, 0.2447]
    img_size = (224,224)
    preprocess = transforms.Compose(
        [
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor()
        ]
    )
    common_transforms = transforms.Compose(
        [
            preprocess,
            transforms.Normalize(mean=mean,std=std)
        ]
    )
    train_transforms = transforms.Compose(
        [
            preprocess,
            transforms.RandomHorizontalFlip(),

            transforms.RandomErasing(p=0.4),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            ], p =0.1),

            transforms.Normalize(mean = mean,std = std)
        ]
    )
    return train_transforms, common_transforms

def data_loaders(train_root, val_root, train_transforms, common_transforms):
    #Apply augmentations to the training dataset
    train_data = datasets.ImageFolder(root = train_root, transform = train_transforms)

    # The validation dataset should have only common transforms like Resize, ToTensor and Normalize.
    val_data = datasets.ImageFolder(root=val_root, transform = common_transforms)
    train_loader = DataLoader(
        train_data,
        shuffle = True,
        batch_size = train_config.batch_size,
        num_workers = train_config.num_workers
    )
    val_loader = DataLoader(
        val_data,
        shuffle = False,
        batch_size = train_config.batch_size,
        num_workers = train_config.num_workers
    )
    print(f"Length of Training dataset: {len(next(iter(train_loader)))}")
    print(f"size of Training dataset: {len(train_data)}")
    print(f"Length of Validation dataset: {len(next(iter(val_loader)))}")
    print(f"size of Validation dataset: {len(val_data)}")

    return train_loader, val_loader

def visualize_images(dataloader, num_images = 20):
    fig = plt.figure(figsize=(10,10))

    #Iterate over the first batch
    images, labels = next(iter(dataloader))
    # print(images.shape)

    num_rows = 4
    num_cols = int(np.ceil((num_images / num_rows)))

    for idx in range(min(num_images, len(images))):
        image, label = images[idx], labels[idx]


        ax = fig.add_subplot(num_rows, num_cols, idx+1, xticks = [], yticks = [])

        image = image.permute(1,2,0)

        #Normalize the image to [0,1] to display

        image = (image - image.min()) / (image.max() - image.min())
        ax.imshow(image, cmap="gray")  # remove the batch dimension
        ax.set_title(f"{label.item()}: {class_mapping[label.item()]}")

    fig.tight_layout()
    plt.show()

def train(model, train_loader):
    model.train()
    model.to(DEVICE)

    running_loss = 0
    correct_predictions = 0
    total_train_samples = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        total_train_samples += labels.shape[0]
        correct_predictions += (predicted == labels).sum().item()

    train_avg_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_predictions / total_train_samples
    return train_avg_loss, train_accuracy

def validation(model, val_loader):
    model.eval()
    model.to(DEVICE)

    running_loss = 0
    correct_predictions = 0
    total_val_samples = 0

    for images, labels in tqdm(val_loader, desc="Validation"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with torch.no_grad():
            outputs = model(images)

        loss = F.cross_entropy(outputs, labels)
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        total_val_samples += labels.shape[0]
        correct_predictions += (predicted == labels).sum().item()

    val_avg_loss = running_loss / len(val_loader)
    val_accuracy = 100 * correct_predictions / total_val_samples
    return val_avg_loss, val_accuracy

# --- CHANGE IS HERE ---
if __name__ == "__main__":
    train_config = TrainingConfig()
    print("Available Device: ", DEVICE)
    set_seed(42)
    download_data()
    train_root, val_root = train_test_data()
    display_labels()
    train_transforms, common_transforms = process_data()
    # This call is what triggered the crash. Now it's safe.
    train_loader, val_loader = data_loaders(train_root, val_root, train_transforms, common_transforms)
    #visualize_images(train_loader, num_images = 16)

    model = MyModel()
    optimizer  = Adam(model.parameters(), lr = train_config.learning_rate)

    logdir = "runs/80epochs-3.3M_param_dropout"
    writer = SummaryWriter(logdir)
    dummy_input = (1,3,224,224)
    print(summary(model, dummy_input, row_settings = ["var_names"], device=DEVICE))
    train_avg_loss, train_accuracy = train(model, train_loader)
    val_avg_loss, val_accuracy = validation(model, val_loader)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(train_config.num_epochs):
        train_loss, train_accuracy = train(model, train_loader)
        val_loss, val_accuracy = validation(model, val_loader)


        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1:0>2}/{train_config.num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Logging metrics to tensorboard
        writer.add_scalar('Loss/train', train_loss)
        writer.add_scalar('Loss/val', val_loss)
        writer.add_scalar('Accuracy/train', train_accuracy)
        writer.add_scalar('Accuracy/val', val_accuracy)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_weights =  model.state_dict()
            print(f"Saving best model...💾")
            torch.save(best_weights, "best.pt")

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(range(1,train_config.num_epochs + 1), train_losses, label = "Train Loss")
    plt.plot(range(1, train_config.num_epochs + 1), val_losses, label = "Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1,train_config.num_epochs + 1), train_accuracies, label = "Train Accuracy")
    plt.plot(range(1, train_config.num_epochs + 1), val_accuracies, label = "Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    model.load_state_dict(torch.load("best.pt"))
    model.eval()


