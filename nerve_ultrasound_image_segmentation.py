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


##### each training file is named as.tif. aand the subject number is the patient
 
all_tifs = glob.glob(os.path.join(DATA_DIR, "**", "*.tif"), recursive=True)
print(f"Found {len(all_tifs)} .tif file(s) under {DATA_DIR}")
 
assert len(all_tifs) > 0, (
    f"No .tif file found under {DATA_DIR}. "
    f"Contents of that folder: {os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else 'PATH DOES NOT EXIST'}"
)
 
rows = []
for p in all_tifs:
    m = re.match(r"^(\d+)_(\d+)\.tif$", os.path.basename(p))
    if m is None:
        continue                      # skips masks and unlabeled test frames
    mask_p = p[:-4] + "_mask.tif"
    if os.path.exists(mask_p):
        rows.append({"subject": int(m.group(1)), "frame": int(m.group(2)),
                     "image_path": p, "mask_path": mask_p})
 
meta = pd.DataFrame(rows).sort_values(["subject", "frame"]).reset_index(drop=True)
print(f"{len(meta)} labeled frames from {meta['subject'].nunique()} subjects")
 
first = Image.open(meta.loc[0, "image_path"])
print("raw frame size (W x H):", first.size, " mode:", first.mode)




