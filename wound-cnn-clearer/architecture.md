# CNN Wound Clearer: Training Architecture Spec

**Goal:** Semantic segmentation of autofluorescence max stack images to remove artifacts.
**Current State:** 256x256 image and mask patches are generated and split into train/val sets in `data/patches_img/` and `data/patches_mask/`. 
**Framework:** PyTorch, segmentation_models_pytorch (SMP), Albumentations.

**Model Details:**
- **Architecture:** U-Net
- **Encoder:** ResNet34
- **Weights:** ImageNet (crucial for our small dataset)
- **Input Channels:** 3 (RGB, as data is mapped to the red channel)
- **Output Classes:** 1 (Binary mask: 1 for artifact/keep, 0 for background)

**File Structure to Build:**
1. `src/dataset.py`: PyTorch Dataset using Albumentations for heavy augmentation.
2. `src/train.py`: Training loop with Dice Loss + BCE Loss, validation tracking, and model checkpointing.