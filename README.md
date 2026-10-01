#Ultrasound Nerve Segmentation Pipeline

An end-to-end PyTorch deep learning pipeline designed to segment the brachial plexus nerve in ultrasound images using a lightweight U-Net architecture. This project incorporates rigorous medical data diagnostic steps—such as duplicate detection, conflicting label filtering, and patient-level cross-validation—to ensure zero data leakage and high real-world performance on unseen clinical subjects.


Key FeaturesPatient Leakage Prevention: Uses group-based splitting (GroupShuffleSplit) on patient IDs to guarantee that frames from the same subject never cross into validation or test sets.Label Conflict Diagnostics: Identifies near-duplicate frames using downsampled correlation analysis ($r > 0.99$) and flags contradictory annotations (where one frame marks a nerve and an identical frame marks background).Custom Small U-Net: A memory-efficient U-Net variant (~500k parameters) optimized for rapid training and edge deployment.Hybrid Loss Function: Combines Binary Cross-Entropy (BCE) with Soft Dice Loss to effectively address pixel-level class imbalance.Inference Post-Processing: Optimizes pixel decision thresholds and minimum connected area thresholds to remove small false-positive blobs on negative frames.Production-Ready Export: Saves model weights as a standalone TorchScript module (.pt) and writes metadata/post-processing thresholds to a json configuration file for deployment in C++ or Python environments.

Dataset & PreprocessingThe pipeline expects data structured from the Kaggle Ultrasound Nerve Segmentation dataset:Image Dimensions: Resized from $420 \times 580$ to $96 \times 128$ while preserving aspect ratio (~1.38).Interpolation Strategy:Images: Resized with Bilinear Interpolation to maintain continuous ultrasound echo intensity gradients.Masks: Resized with Nearest-Neighbor Interpolation to prevent soft boundary artifacts and preserve exact binary labels.

Architecture Overview

Input Image (1x96x128)
       │
   [ConvBlock] ─────────────── Skip 1 ──────────────┐ (base = 16)
       │                                            │
   [MaxPool]                                        │
   [ConvBlock] ─────────────── Skip 2 ──────────┐   │
       │                                        │   │
   [MaxPool]                                    │   │
   [ConvBlock] ─────────────── Skip 3 ──────┐   │   │
       │                                    │   │   │
   [MaxPool]                                │   │   │
  [Bottleneck] (base * 8)                   │   │   │
       │                                    │   │   │
[ConvTranspose + Cat] ◄─────────────────────┘   │   │
   [ConvBlock]                                  │   │
       │                                        │   │
[ConvTranspose + Cat] ◄─────────────────────────┘   │
   [ConvBlock]                                      │
       │                                            │
[ConvTranspose + Cat] ◄─────────────────────────────┘
   [ConvBlock]
       │
  [Conv2d (1x1)] ──► Logits Output (1x96x128)
