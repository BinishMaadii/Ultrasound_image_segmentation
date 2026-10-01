# Ultrasound Nerve Segmentation with PyTorch & Small U-Net

An end-to-end deep learning pipeline for segmenting the brachial plexus nerve in ultrasound images using PyTorch. Designed for high reliability in medical imaging, this workflow prevents data leakage through patient-grouped cross-validation, cleans noise via near-duplicate pair analysis, and exports deployment-ready TorchScript models with sidecar configurations.

---

## Key Features

* **Patient-Grouped Splitting (`GroupShuffleSplit`):** Prevents patient data leakage across training, validation, and test splits ($0\%$ patient overlap).
* **Near-Duplicate & Conflict Detection:** Uses Pearson correlation ($r > 0.99$) on image embeddings to identify near-duplicate frames and resolve conflicting labels.
* **Hybrid Loss Function:** Combines **Binary Cross-Entropy (BCE)** and **Soft Dice Loss** to handle extreme pixel-level class imbalance.
* **Domain-Specific Augmentation:** Applies gain (brightness) and gamma (contrast) variations to simulate ultrasound machine calibration variances without spatial flipping.
* **Val-Tuned Post-Processing:** Optimizes pixel-probability and minimum-connected-area thresholds to reduce false positives on empty background scans.
* **Production Deployment:** Exports a serialized **TorchScript (`.pt`)** model and sidecar JSON configuration for standalone inference in C++/libtorch.

---

## Directory Structure

```text
.
├── 01_subjects_and_balance.png       # Dataset distribution per subject
├── 02_examples_with_masks.png        # Sample ultrasound scans with mask overlays
├── 03_conflicting_labels.png        # Near-identical scan diagnostics
├── 04_training_curves.png           # Training loss and validation Dice curves
├── 05_dice_per_subject.png          # Test set evaluation per patient
├── 06_best_and_worst_predictions.png# Model prediction visual error analysis
├── nerve_unet_torchscript.pt        # Exported TorchScript model
├── nerve_unet_config.json           # Post-processing & normalization metadata
└── train_and_eval.py                 # Core pipeline implementation
