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
from m4_download_transform_loader import DEVICE, train_config

def init_model():
    model = MyModel()
    optimizer  = Adam(model.parameters(), lr = train_config.learning_rate)

    logdir = "runs/80epochs-3.3M_param_dropout"
    writer = SummaryWriter(logdir)
    dummy_input = (1,3,224,224)
    print(summary(model, dummy_input, row_settings = ["var_names"],device="cpu"))

init_model()