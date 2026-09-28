import glob
import json
import os
import re
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
 
# ---- config ----
DATA_DIR = "/kaggle/input/ultrasound-nerve-segmentation"
IMG_H, IMG_W = 96, 128     # raw frames are 420 x 580, this keeps the aspect ratio (~1.38)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 0
DUP_CORR = 0.99            # two frames correlated above this (at low resolution) count as near-duplicates
DROP_CONFLICTING = False   # True = remove near-duplicate frames whose masks disagree (nerve vs no nerve)
 
torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

