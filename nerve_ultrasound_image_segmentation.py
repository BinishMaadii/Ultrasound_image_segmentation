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




##### 4. Diagnostics: patient leakage and conflicting labels
#### This is the equivalent of the lot check in the wafer pipeline, but the effect is much stronger here.
#### Frames from one subject come from the same scan session, so many are almost identical.
#### If frames are split at random, the test set is full of near-copies of training frames.
####
#### The same comparison also finds label noise: near-identical frames where one mask has a nerve
#### and the other is empty. No model can get both right, so this sets a ceiling on the score.
 
small = X[:, ::4, ::4].reshape(len(X), -1)                      # 24 x 32 thumbnails
small = (small - small.mean(1, keepdims=True)) / (small.std(1, keepdims=True) + 1e-6)
 
dup_pairs = []
for subj, idx in meta.groupby("subject").indices.items():
    A = small[idx]
    C = A @ A.T / A.shape[1]                                     # Pearson correlation between frames
    a, b = np.where(np.triu(C > DUP_CORR, k=1))
    dup_pairs += [(idx[i], idx[j]) for i, j in zip(a, b)]
dup_pairs = np.array(dup_pairs, dtype=int).reshape(-1, 2)
 
conflict = dup_pairs[present[dup_pairs[:, 0]] != present[dup_pairs[:, 1]]]
conflict_frames = np.unique(conflict)
print(f"\nNear-duplicate pairs (corr > {DUP_CORR}): {len(dup_pairs)}")
print(f"Pairs where one mask has a nerve and the other does not: {len(conflict)} "
      f"(touching {len(conflict_frames)} frames, {len(conflict_frames) / len(X):.1%} of data)")
 
# What a random frame-level 80/20 split would do
perm = np.random.permutation(len(X))
rand_train = np.zeros(len(X), bool); rand_train[perm[: int(0.8 * len(X))]] = True
has_dup_in_train = np.zeros(len(X), bool)
for i, j in dup_pairs:
    if rand_train[i] and not rand_train[j]:
        has_dup_in_train[j] = True
    if rand_train[j] and not rand_train[i]:
        has_dup_in_train[i] = True
rand_test = ~rand_train
print(f"Random frame split: {has_dup_in_train[rand_test].mean():.1%} of test frames have a "
      f"near-duplicate in train. A split by subject brings this to 0% by construction.")
 
if len(conflict) > 0:
    i, j = conflict[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, k in zip(axes, (i, j)):
        ax.imshow(X[k], cmap="gray")
        if M[k].any():
            ax.contour(M[k], levels=[0.5], colors="lime", linewidths=1)
        ax.set_title(f"subject {meta.loc[k, 'subject']}, frame {meta.loc[k, 'frame']}: "
                     f"{'nerve' if present[k] else 'empty'}")
        ax.axis("off")
    plt.suptitle("Near-identical frames with contradicting masks")
    plt.tight_layout()
    plt.savefig("03_conflicting_labels.png", dpi=150)
    plt.show()
 
if DROP_CONFLICTING:
    keep = np.setdiff1d(np.arange(len(X)), conflict_frames)
    X, M, meta = X[keep], M[keep], meta.iloc[keep].reset_index(drop=True)
    present = meta["nerve_present"].to_numpy()
    print(f"Dropped {len(conflict_frames)} conflicting frames, {len(X)} remain")




##### 5. Split by patient
#### GroupShuffleSplit keeps every frame of a subject in exactly one of train / val / test.
#### This answers the question a customer actually asks: how does it work on a new patient?
#### Stratification is no longer exact, so the nerve fraction per split is printed to check it.
#### Split: 70% of subjects train, 15% val, 15% test.
 
groups = meta["subject"].to_numpy()
idx = np.arange(len(X))
 
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
idx_train, idx_temp = next(gss.split(idx, present, groups))
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
v, t = next(gss2.split(idx_temp, present[idx_temp], groups[idx_temp]))
idx_val, idx_test = idx_temp[v], idx_temp[t]
 
s_train, s_val, s_test = set(groups[idx_train]), set(groups[idx_val]), set(groups[idx_test])
assert not (s_train & s_val) and not (s_train & s_test) and not (s_val & s_test), "subject leaked across splits"
 
for name, ids in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
    print(f"{name:5s}: {len(ids):5d} frames, {len(set(groups[ids])):3d} subjects, "
          f"nerve present in {present[ids].mean():.1%}")
 
 
##### 6. Imbalance: what it means for segmentation
#### Frame level: a model that always predicts an empty mask already scores well on mean Dice,
#### because an empty prediction on an empty frame counts as Dice = 1. That is the "always none"
#### baseline from the wafer pipeline, and it is printed here so the model has something to beat.
####
#### Pixel level: nerve pixels are a few percent of the image, so plain BCE is dominated by background.
#### Instead of inverse-frequency weights, the loss adds a soft Dice term. Dice only looks at the
#### overlap between prediction and nerve, so the number of background pixels does not dilute it.
 
baseline_val = (1 - present[idx_val]).mean()
print(f"\n'Always empty' baseline, mean Dice on val: {baseline_val:.3f}")
print(f"Nerve pixel fraction in train: {M[idx_train].mean():.2%}")
 
def to_loader(ids, shuffle):
    ds = TensorDataset(torch.from_numpy(X[ids]).unsqueeze(1), torch.from_numpy(M[ids]).unsqueeze(1))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)
 
train_loader = to_loader(idx_train, shuffle=True)
val_loader = to_loader(idx_val, shuffle=False)
test_loader = to_loader(idx_test, shuffle=False)




##### 7. Model: small U-Net
#### Encoder downsamples 3 times (96x128 -> 12x16), decoder upsamples back and concatenates the
#### encoder features at each level, so fine boundary detail survives. About 0.5 M parameters.
 
def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )
 
class SmallUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = conv_block(1, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.bottleneck = conv_block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)
 
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)            # logits, shape (N, 1, H, W)
 
model = SmallUNet().to(device)
print("parameters:", sum(p.numel() for p in model.parameters()))
 
bce = nn.BCEWithLogitsLoss()
 
def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()
 
#### Augmentation that matches how ultrasound varies between devices and operators:
#### overall gain (brightness) and a gamma change (contrast curve). No flips, because the
#### anatomy has a fixed orientation in these scans and a flipped neck is not a realistic input.
def augment(x):
    n = x.size(0)
    gain = torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
    gamma = torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.8, 1.25)
    return (x.clamp(1e-6, 1) ** gamma * gain).clamp(0, 1)
 
def dice_per_image(pred, target):
    # pred, target: binary numpy arrays (N, H, W). Empty prediction on empty target counts as 1.
    inter = (pred * target).sum(axis=(1, 2))
    total = pred.sum(axis=(1, 2)) + target.sum(axis=(1, 2))
    return np.where(total == 0, 1.0, 2 * inter / np.maximum(total, 1))
 
@torch.no_grad()
def predict_probs(loader):
    model.eval()
    out = [torch.sigmoid(model(xb.to(device))).cpu().numpy()[:, 0] for xb, _ in loader]
    return np.concatenate(out)
 
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
 
history = {"train_loss": [], "val_dice": []}
best_dice, best_state = -1.0, None
 
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for xb, mb in train_loader:
        xb, mb = augment(xb.to(device)), mb.to(device)
        logits = model(xb)
        loss = bce(logits, mb) + soft_dice_loss(logits, mb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
 
    val_probs = predict_probs(val_loader)
    val_dice = dice_per_image((val_probs > 0.5).astype(np.float32), M[idx_val]).mean()
    scheduler.step(val_dice)
    history["train_loss"].append(running / len(idx_train))
    history["val_dice"].append(val_dice)
 
    if val_dice > best_dice:           # keep the weights from the best validation epoch
        best_dice = val_dice
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"epoch {epoch:2d}  train loss {history['train_loss'][-1]:.4f}  val Dice {val_dice:.4f}")
 
model.load_state_dict(best_state)
 
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history["train_loss"]); axes[0].set_title("train loss (BCE + Dice)"); axes[0].set_xlabel("epoch")
axes[1].plot(history["val_dice"], label="model")
axes[1].axhline(baseline_val, color="gray", ls="--", label="always empty")
axes[1].set_title("val mean Dice"); axes[1].set_xlabel("epoch"); axes[1].legend()
plt.tight_layout()
plt.savefig("04_training_curves.png", dpi=150)
plt.show()
 


##### 8. Post-processing, tuned on validation only
#### Two decisions turn probabilities into a final mask:
####   pixel threshold: which pixels count as nerve
####   minimum area:    if fewer pixels than this survive, call the frame empty
#### The minimum area is what fixes the frame-level imbalance at inference time: small false-positive
#### blobs on empty frames cost a full Dice point each, so removing them matters a lot.
#### Both are chosen on val, then frozen. The test set is not touched until step 9.
 
def postprocess(probs, pix_thresh, min_area):
    masks = (probs > pix_thresh).astype(np.float32)
    small_masks = masks.sum(axis=(1, 2)) < min_area
    masks[small_masks] = 0
    return masks
 
val_probs = predict_probs(val_loader)
grid = [(t, a) for t in (0.3, 0.4, 0.5, 0.6, 0.7) for a in (0, 25, 50, 100, 150, 200, 300)]
scores = {(t, a): dice_per_image(postprocess(val_probs, t, a), M[idx_val]).mean() for t, a in grid}
PIX_THRESH, MIN_AREA = max(scores, key=scores.get)
print(f"\nchosen on val: pixel threshold {PIX_THRESH}, min area {MIN_AREA} px "
      f"(val Dice {scores[(PIX_THRESH, MIN_AREA)]:.4f}, without post-processing {scores[(0.5, 0)]:.4f})")
if postprocess(val_probs, PIX_THRESH, MIN_AREA).sum() == 0:
    print("WARNING: the chosen settings predict an empty mask on every val frame. "
          "The model is not beating the always-empty baseline yet (train longer or check the data).")
 





##### 9. Test evaluation on unseen patients
#### Three views of the same predictions:
####   mean Dice (the segmentation score, compared with the always-empty baseline)
####   nerve present / absent as a classification (same report as the wafer pipeline)
####   Dice per subject, because a mean can hide one patient where the model fails completely
 
test_probs = predict_probs(test_loader)
test_masks = postprocess(test_probs, PIX_THRESH, MIN_AREA)
test_dice = dice_per_image(test_masks, M[idx_test])
baseline_test = (1 - present[idx_test]).mean()
 
has_nerve = present[idx_test] == 1
print(f"\nTEST mean Dice:                  {test_dice.mean():.4f}")
print(f"TEST 'always empty' baseline:    {baseline_test:.4f}")
print(f"TEST Dice on frames with nerve:  {test_dice[has_nerve].mean():.4f}")
 
pred_present = (test_masks.sum(axis=(1, 2)) > 0).astype(int)
true_present = present[idx_test]
print("\nNerve present / absent:")
print(classification_report(true_present, pred_present, target_names=["no nerve", "nerve"], digits=3, zero_division=0))
print("macro F1:", round(f1_score(true_present, pred_present, average="macro"), 3))
print("confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(true_present, pred_present))
 
subj_dice = pd.Series(test_dice, index=groups[idx_test]).groupby(level=0).mean().sort_values()
print("\nDice per test subject:")
print(subj_dice.round(3).to_string())
 
fig, ax = plt.subplots(figsize=(8, 4))
subj_dice.plot(kind="bar", ax=ax)
ax.axhline(test_dice.mean(), color="gray", ls="--", label="mean")
ax.set_title("Test Dice per subject"); ax.set_xlabel("subject"); ax.set_ylabel("mean Dice"); ax.legend()
plt.tight_layout()
plt.savefig("05_dice_per_subject.png", dpi=150)
plt.show()
 
# %% Best and worst test frames that contain a nerve (green = truth, red = prediction)
nerve_ids = np.where(has_nerve)[0]
order = nerve_ids[np.argsort(test_dice[nerve_ids])]
show = list(order[-3:][::-1]) + list(order[:3])
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, k in zip(axes.flat, show):
    ax.imshow(X[idx_test][k], cmap="gray")
    ax.contour(M[idx_test][k], levels=[0.5], colors="lime", linewidths=1)
    if test_masks[k].any():
        ax.contour(test_masks[k], levels=[0.5], colors="red", linewidths=1)
    ax.set_title(f"subject {groups[idx_test][k]}  Dice {test_dice[k]:.2f}")
    ax.axis("off")
plt.suptitle("Top row: best 3   Bottom row: worst 3   (green = truth, red = prediction)")
plt.tight_layout()
plt.savefig("06_best_and_worst_predictions.png", dpi=150)
plt.show()
 
 
##### 10. Export for integration
#### This makes the research model runnable on the customer side in the following steps
#### TorchScript produces a single file that libtorch can load from C++ without any Python code.
#### The post-processing values go into a small JSON next to it, because they are part of the model:
#### The same weights with a different threshold give different results.
#### The reload check confirms the exported file gives the same output as the Python model.
 
model.eval().cpu()
example = torch.from_numpy(X[idx_test[:1]]).unsqueeze(1)
scripted = torch.jit.trace(model, example)
scripted.save("nerve_unet_torchscript.pt")
 
reloaded = torch.jit.load("nerve_unet_torchscript.pt")
with torch.no_grad():
    max_diff = (reloaded(example) - model(example)).abs().max().item()
print(f"\nexported nerve_unet_torchscript.pt, max difference after reload: {max_diff:.2e}")
assert max_diff < 1e-4
 
with open("nerve_unet_config.json", "w") as f:
    json.dump({
        "input_shape": [1, 1, IMG_H, IMG_W],
        "input_scaling": "grayscale / 255, bilinear resize to input_shape",
        "output": "logits, apply sigmoid",
        "pixel_threshold": float(PIX_THRESH),
        "min_area_px": int(MIN_AREA),
        "split": "by subject (GroupShuffleSplit), seed %d" % SEED,
        "test_mean_dice": round(float(test_dice.mean()), 4),
        "test_baseline_always_empty": round(float(baseline_test), 4),
    }, f, indent=2)
print("saved nerve_unet_config.json")


