# 🔬 CNN Wound Clearer (Pre-Processing Module)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

A deep learning-based image denoising and artifact removal module for the **Regen Platform**. This microservice is designed to clean raw clinical wound imagery, extracting purely semantic tissue data before passing the output to downstream spatial and distance-measuring pipelines.
---------------------------------------------------------------------------------
## 📖 Overview
Clinical imagery is notoriously noisy, often containing surgical tools, lighting reflections, and non-target biological matter. The CNN Wound Clearer acts as an automated, intelligent filter. By isolating the wound bed from background artifacts, it significantly improves the accuracy of downstream YOLO segmentation models.
---------------------------------------------------------------------------------
## 🧠 Model Architecture
This module utilizes a **U-Net** architecture with a **ResNet34** encoder, chosen for its strong feature extraction capabilities and efficiency in biomedical semantic segmentation. 

* **Framework:** PyTorch & Segmentation Models PyTorch (SMP)
* **Input:** Raw clinical tiff imagery (dynamically padded to multiples of 32 for U-Net compatibility).
* **Output:** Binary spatial mask isolating the target wound bed.
---------------------------------------------------------------------------------
### Loss Function
The model was trained using a composite loss function to handle class imbalance and ensure sharp boundary predictions. It combines Binary Cross Entropy (BCE) for pixel-wise classification and Dice Loss for spatial overlap:

$$\mathcal{L}_{total} = \mathcal{L}_{BCE}(y, \hat{y}) + \mathcal{L}_{Dice}(y, \hat{y})$$
---------------------------------------------------------------------------------
## 📂 Directory Structure
```text
cnn-wound-clearer/
├── src/
│   ├── dataset.py         # PyTorch Dataset and Albumentations pipeline
│   ├── train.py           # Training loop and loss definitions
│   └── predict.py         # Dynamic padding and inference script
├── weights/
│   └── best_model.pth     # Model weights (Local only, ignored by Git)
├── data/
│   ├── raw_test_images/   # Drop input images here
│   └── predictions/       # Generated masks will appear here
├── Dockerfile             # Container specification
└── requirements.txt       # Strict version dependencies
---------------------------------------------------------------------------------
🚀 Execution & Usage
---------------------------------------------------------------------------------
Option 1: Containerized (Nextflow / Production Ready)
This module is fully containerized to ensure cross-platform compatibility without dependency conflicts.

# Build the inference container
docker build -t cnn-clearer:latest .

# Run the container (maps local data folders to the container)
docker run -v $(pwd)/data:/app/data cnn-clearer:latest
---------------------------------------------------------------------------------
Option 2: Local Virtual Environment (Development)
For local testing and model fine-tuning:


# Set up the isolated environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run inference on images in data/raw_test_images/
python src/predict.py
---------------------------------------------------------------------------------
⚙️ Data Augmentation Pipeline
To ensure robustness against varied lighting and angles in clinical settings, the training pipeline utilizes aggressive albumentations, including:

ShiftScaleRotate (simulating camera angle variations)

GridDistortion (simulating curved anatomical structures)

Random Flips and Rotations
---------------------------------------------------------------------------------
🤝 Integration
This module is designed to be the first step in the regen-platform DAG (Directed Acyclic Graph). Output masks generated in data/predictions/ are formatted to be instantly ingested by the distance-tool Nextflow pipeline.