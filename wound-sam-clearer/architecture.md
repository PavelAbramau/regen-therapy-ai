# Architecture: Interactive SAM Wound Clearer

## 🎯 Objective
To provide a zero-shot, interactive pre-processing module for clinical tissue imagery. This tool replaces autonomous U-Net segmentation, allowing human-in-the-loop precision to mask out high-variance surgical artifacts (tools, glares, external blood vessels) prior to downstream spatial analysis.

## 🧠 Core Engine
* **Model:** Meta Segment Anything Model (SAM)
* **Checkpoint:** `vit_b` (Base Vision Transformer) chosen to balance memory constraints on local MPS/CUDA hardware while maintaining state-of-the-art boundary calculation.
* **Mechanism:** Interactive Point Prompting.

## ⚙️ System Components
The module is divided into three functional layers:

1. **The Interface Layer (OpenCV)**
   * Renders the target raw image.
   * Captures user-defined spatial coordinates `(X, Y)` via mouse click events.
   * Passes coordinates to the inference engine as positive foreground prompts.

2. **The Inference Layer (SAM)**
   * Ingests the RGB image tensor and the coordinate array.
   * Calculates the topological boundaries of the targeted artifact without requiring pre-defined class labels.
   * Outputs a highly accurate Boolean mask (`True` for artifact, `False` for background).

3. **The Processing Layer (NumPy)**
   * Inverts the inference logic: target pixels are converted to pure black `[0, 0, 0]`, preserving the native RGB values of the surrounding usable wound bed.
   * Saves the cleaned artifact to `data/predictions/` for seamless ingestion into the Nextflow `distance-tool` pipeline.

## 📂 Directory Structure

wound-sam-clearer/
├── src/
│   └── sam_clearer.py     # Main interactive script
├── weights/
│   └── sam_vit_b_01ec64.pth # Meta SAM weights
├── data/
│   ├── raw_test_images/   # Input directory
│   └── predictions/       # Cleaned output directory
├── .venv/                 # Isolated Python environment
├── architecture.md        # Technical specification
└── requirements.txt       # Dependency lock 
