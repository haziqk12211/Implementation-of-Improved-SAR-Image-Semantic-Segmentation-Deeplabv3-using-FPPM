# Improved DeepLabV3+ for Semantic Segmentation

An implementation of the paper **"An Improved SAR Image Semantic Segmentation DeepLabV3+ Network Based on the Feature Post-Processing Module"** (Li & Kong, *Remote Sensing*, 2023) — adapted and trained on the **LoveDA** remote sensing dataset.

---

## Overview

This project reimplements the core architectural improvements proposed in the paper using PyTorch and evaluates them on the publicly available LoveDA urban/rural segmentation benchmark. The original paper targeted polarimetric SAR images; our implementation extends the same principles to high-resolution optical remote sensing imagery.

---

## Paper vs. Implementation — Key Differences

| Aspect | Paper | This Implementation |
|---|---|---|
| **Dataset** | Custom SAR dataset (Sentinel-1, Nanjing region) — 1,800 train / 200 val images, 5 classes | [LoveDA](https://www.kaggle.com/datasets/mohammedjaveed/loveda-dataset) — large-scale public dataset, **7 classes** |
| **Classes** | Background, River, Forest, Building, Road | Background, Building, Road, Water, Barren, Forest, Agriculture |
| **Input modality** | Polarimetric SAR + paired optical | RGB optical satellite imagery only |
| **Backbone** | MobileNet-V2 | MobileNet-V2 (same) |
| **Attention module** | Coordinate Attention (CA) — applied after backbone | CA — applied after backbone (same) |
| **Loss function** | Focal Loss (α=0.25, γ=2) | Focal Loss (α=0.25, γ=2, ignore\_index=255) — with proper ignore masking for void labels |
| **ASPP improvement** | 3×3 atrous convolutions decomposed into 3×1 → 1×3 | Same 2D decomposition (maintained dilation rates 6, 12, 18) |
| **FPPM branches** | 3 branches (determined via MVG image quality scoring) | 3 branches (adopted paper's finding directly; MVG scoring not re-run) |
| **FPPM attention** | ECA (Efficient Channel Attention) on combined branches | Same ECA module |
| **Non-local blocks** | Custom non-local blocks per pyramid branch | Same architecture |
| **Training epochs** | Not specified | 40 epochs |
| **Optimizer** | Not specified | Adam, lr=1e-4 |
| **Augmentation** | Random rotation | Random crop (256×256), horizontal flip, vertical flip, random 90° rotate, ImageNet normalization |
| **Metrics** | DICE, mIoU, per-class IoU, Global Accuracy | Same metrics (mIoU, per-class IoU, Global Accuracy) |
| **Evaluation** | Validation set | Full validation split |

### Notable Implementation Choices

**Focal Loss ignore masking** — LoveDA uses pixel value `255` to mark unlabeled/void regions. The paper's original Focal Loss formulation does not account for this (the SAR dataset had no void pixels). We added `ignore_index=255` to prevent void pixels from contributing to the loss or metric calculations.

**Dataset difference** — The paper's private SAR dataset cannot be reproduced. LoveDA was selected as the closest publicly available remote sensing segmentation benchmark, enabling full reproducibility.

**FPPM branch count** — The paper uses a Multivariate Gaussian (MVG) image quality algorithm to empirically determine that 3 branches is optimal. We adopted this finding directly rather than re-running the quality evaluation, since it is dataset-agnostic as a structural conclusion.

---

## Architecture Summary

```
Input (RGB, 256×256)
    │
    ├── MobileNet-V2 Backbone
    │       ├── Low-level features  [stride 4]
    │       └── High-level features [stride 16]
    │
    ├── Coordinate Attention (CA) — after backbone
    │
    ├── Improved ASPP
    │       ├── 1×1 Conv
    │       ├── 3×1 → 1×3 Conv (rate 6)   ← 33% fewer params vs. standard 3×3
    │       ├── 3×1 → 1×3 Conv (rate 12)
    │       ├── 3×1 → 1×3 Conv (rate 18)
    │       └── Global Average Pooling
    │
    ├── FPPM (Feature Post-Processing Module)
    │       ├── Branch 1: Non-local on full feature map (H×W)
    │       ├── Branch 2: Non-local on (H/2)×(W/2) patches
    │       ├── Branch 3: Non-local on (H/4)×(W/4) patches
    │       └── ECA module on concatenated branches
    │
    └── Decoder
            ├── Upsample ×4 + concat with low-level features
            ├── 3×3 Conv → 3×3 Conv
            └── Upsample ×4 → Output (7 classes, 256×256)
```

---

## Results

Training converged at **Epoch 28** (best checkpoint saved).

### Per-Class IoU (Validation Set of Our Implementation)

| Class | IoU with Baseline DeepLabV3+ | IoU with Full model|
|---|---|---|
| Background | 41.73% | 45.90% | 
| Building | 39.18% | 42.79% |
| Road | 28.45% |29.90% |
| Water | 47.18% | 54.99% |
| Barren | 19.95% | 15.63% |
| Forest | 26.06% | 25.66% |
| Agriculture | 42.72% | 44.87% |

### Final Results of Our Implementation

| Model | mIoU | Global Accuracy |
|---|---|---|
| Baseline DeepLabV3+ | 35.04% | 61.19% |
| + All (Full model) | **37.11%** | **64.65%** |

---

>GLobal accuracy increase is around 3%, a very similar increase to paper result with SAR dataset with 5 classes

## Project Structure

```
├── AI_Project_improvedModel.ipynb   # Main notebook (all phases)
│
│   Phase 1  — Dataset loading & augmentation (LoveDA)
│   Phase 2  — Baseline DeepLabV3+ (MobileNet-V2 + standard ASPP)
│   Phase 3-6 — Improved model (CA + Focal Loss + Improved ASPP + FPPM)
│   Phase 7  — Training loop (Adam, 40 epochs, best-model checkpointing)
│   Phase 8  — Inference visualization
│   Phase 9  — Full quantitative evaluation
│
└── README.md
```

---

## Setup & Usage

### Requirements

```bash
pip install torch torchvision albumentations kagglehub opencv-python matplotlib numpy
```

### Running in Google Colab

1. Mount Google Drive (for checkpoint saving).
2. Set your `KAGGLE_API_TOKEN` in Colab Secrets.
3. Run all cells in order — the dataset will be auto-downloaded via `kagglehub`.
4. Trained weights are saved to `Google Drive/best_deeplabv3plus_model.pth` whenever validation mIoU improves.

### Inference on saved weights

Load the `ImprovedDeepLabV3Plus` class and point `weights_path` to your saved `.pth` file, then run the Phase 8 / Phase 9 cells.

---

## Reference

> Li, Q.; Kong, Y. An Improved SAR Image Semantic Segmentation DeepLabV3+ Network Based on the Feature Post-Processing Module. *Remote Sens.* **2023**, *15*, 2153. https://doi.org/10.3390/rs15082153
