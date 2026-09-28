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



##### 2. Load and resize every frame
#### Image: bilinear resize, because intensities are continuous (echo amplitude).
#### Mask: nearest-neighbour resize, same reason as the wafer maps. A bilinear resize would invent
#### values between 0 and 1 at the border, which are not a real label.
#### Presence and area are read from the full-resolution mask, before resizing can shrink anything.##### 2. Load and resize every frame
#### Image: bilinear resize, because intensities are continuous (echo amplitude).
#### Mask: nearest-neighbour resize, same reason as the wafer maps. A bilinear resize would invent
#### values between 0 and 1 at the border, which are not a real label.
#### Presence and area are read from the full-resolution mask, before resizing can shrink anything.



def load_pair(image_path, mask_path):
    img = Image.open(image_path).convert("L")
    msk_full = np.asarray(Image.open(mask_path).convert("L")) > 127
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    msk = Image.fromarray(msk_full.astype(np.uint8) * 255).resize((IMG_W, IMG_H), Image.NEAREST)
    return (np.asarray(img, dtype=np.float32) / 255.0,
            (np.asarray(msk) > 127).astype(np.float32),
            int(msk_full.sum()))
 
X, M, full_area = [], [], []
for i, r in meta.iterrows():
    x, m, a = load_pair(r["image_path"], r["mask_path"])
    X.append(x); M.append(m); full_area.append(a)
    if (i + 1) % 1000 == 0:
        print(f"  loaded {i + 1}/{len(meta)}")
 
X = np.stack(X)                       # (N, H, W), values in [0, 1]
M = np.stack(M)                       # (N, H, W), values in {0, 1}
meta["mask_area_full"] = full_area
meta["nerve_present"] = (meta["mask_area_full"] > 0).astype(int)
present = meta["nerve_present"].to_numpy()
print("X shape:", X.shape, " M shape:", M.shape)
 def load_pair(image_path, mask_path):
    img = Image.open(image_path).convert("L")
    msk_full = np.asarray(Image.open(mask_path).convert("L")) > 127
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    msk = Image.fromarray(msk_full.astype(np.uint8) * 255).resize((IMG_W, IMG_H), Image.NEAREST)
    return (np.asarray(img, dtype=np.float32) / 255.0,
            (np.asarray(msk) > 127).astype(np.float32),
            int(msk_full.sum()))
 
X, M, full_area = [], [], []
for i, r in meta.iterrows():
    x, m, a = load_pair(r["image_path"], r["mask_path"])
    X.append(x); M.append(m); full_area.append(a)
    if (i + 1) % 1000 == 0:
        print(f"  loaded {i + 1}/{len(meta)}")
 
X = np.stack(X)                       # (N, H, W), values in [0, 1]
M = np.stack(M)                       # (N, H, W), values in {0, 1}
meta["mask_area_full"] = full_area
meta["nerve_present"] = (meta["mask_area_full"] > 0).astype(int)
present = meta["nerve_present"].to_numpy()
print("X shape:", X.shape, " M shape:", M.shape)


##### 3. Inspect labels
#### Two levels of imbalance: per frame (nerve present or not) and per pixel (nerve vs background).
 
print("\nFrames with nerve:", present.sum(), f"({present.mean():.1%})")
print("Frames without nerve:", (1 - present).sum(), f"({1 - present.mean():.1%})")
print("Nerve pixels among all pixels:", f"{M.mean():.2%}")
print("Nerve pixels inside frames that have a nerve:", f"{M[present == 1].mean():.2%}")
 
per_subject = meta.groupby("subject").agg(frames=("frame", "size"), present_frac=("nerve_present", "mean"))
print("\nFrames per subject:", per_subject["frames"].describe()[["min", "mean", "max"]].round(1).to_dict())
 
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
per_subject["frames"].plot(kind="bar", ax=axes[0])
axes[0].set_title("Frames per subject"); axes[0].set_xlabel("subject"); axes[0].set_ylabel("frames")
axes[0].set_xticks([])
per_subject["present_frac"].plot(kind="bar", ax=axes[1], color="tab:orange")
axes[1].set_title("Fraction of frames with nerve, per subject"); axes[1].set_xlabel("subject")
axes[1].set_xticks([])
plt.tight_layout()
plt.savefig("01_subjects_and_balance.png", dpi=150)
plt.show()
 
# %% Examples with the mask outline drawn on top (3 with nerve, 3 without)
ex_idx = list(np.where(present == 1)[0][:3]) + list(np.where(present == 0)[0][:3])
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, i in zip(axes.flat, ex_idx):
    ax.imshow(X[i], cmap="gray")
    if M[i].any():
        ax.contour(M[i], levels=[0.5], colors="lime", linewidths=1)
    ax.set_title(f"subject {meta.loc[i, 'subject']}, frame {meta.loc[i, 'frame']}\n"
                 f"{'nerve' if present[i] else 'no nerve'}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("02_examples_with_masks.png", dpi=150)
plt.show()
 


