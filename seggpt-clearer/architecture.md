Architecture: SegGPT Contextual Wound Clearer
🎯 Objective
To provide an automated, zero-training pre-processing module for clinical tissue imagery. This microservice utilizes Few-Shot Visual Prompting to dynamically identify and mask high-variance biological artifacts (autofluorescence, cartilage structures) based on paired reference images, feeding clean phenotypic data to downstream predictive models.

🧠 Core Engine
Model: Meta SegGPT (Segment Everything in Context) / Painter

Architecture: Vision Transformer (ViT-Base, Patch 16)

Mechanism: In-context learning via spatial attention matrices.

⚙️ System Components
The Prompt Ingestion Layer:

Loads highly curated (Image, Mask) pairs defining specific artifact topologies (e.g., prompt_cartilage_img.png and prompt_cartilage_mask.png).

The Inference Engine:

Concatenates the prompt pair with the unseen target image.

SegGPT extrapolates the visual texture from the prompt and generates a continuous semantic mask over the target image.

The Masking Logic (NumPy):

Converts the predicted artifact locations to pure black [0, 0, 0], isolating the usable wound bed/blood vessels for predictive analysis.

📂 Directory Structure
Plaintext
wound-seggpt-clearer/
├── src/
│   └── seggpt_inference.py # Execution script
├── weights/
│   └── seggpt_vit_base_patch16_input896.pth
├── data/
│   ├── prompts/           # ImageJ Swatches (Image + Mask pairs)
│   ├── raw_targets/       # Unseen clinical images
│   └── predictions/       # Cleaned output
├── .venv/                 
└── architecture.md
