# 🧬 RegenOS: The Operating System for Human Regeneration

A unified, multiscale computational drug discovery platform designed to identify, optimize, and translate epigenetic small molecules for localized tissue regeneration. 

## 🎯 Vision
RegenOS moves regenerative medicine from observational biology to predictive engineering. By bridging single-cell multi-omics, molecular screening, and macroscopic pharmacokinetics, this platform designs localized therapies that erase "fibrotic memory" and induce true morphogenetic repair in chronic wounds (e.g., Diabetic Foot Ulcers) without systemic toxicity.

## 🏗️ Architecture & Core Modules

The platform operates as a modular ecosystem, utilizing machine learning inference, graph databases, and molecular modeling to simulate the entire drug discovery lifecycle.

### 🟢 Active Core Pipelines

#### 1. [epi-screener](./epi-screener/) (The Discovery Engine)
The high-throughput virtual screening module. Instead of searching blindly, EpiScreener uses a highly constrained, multi-phase funnel to discover and optimize regenerative therapies, specifically aiming to discover **"Small Molecule Yamanaka Cocktails"**—synergistic drug combinations that mimic the reprogramming power of transcription factors (OSKM) without the oncogenic risk or delivery nightmare of gene therapy.
* **Phase 1: Target & Pathway Anchoring:** We map the specific metabolic pathways and epigenetic targets (e.g., DNMT1, HDACs) responsible for locking cells in a fibrotic state.
* **Phase 2: Epigenetic Compound Filtering:** We scan massive compound libraries (10M+ SMILES) using MPNNs and ChemBERTa to isolate molecules with high binding affinity to our anchored targets.
* **Phase 3: Cocktail Optimization & Extended Search:** We utilize synergy-prediction algorithms to identify combinations of molecules (The Eraser + The Driver) that safely push cells into a transient, embryonic-like progenitor state. 

#### 2. [computear-pipeline](./computear-pipeline/) (The Translation Engine)
The multiscale inference engine that bridges computational predictions with tangible biological ground truth. 
* Anchored by a proprietary repository of *in vivo* transcriptomic signatures from successful mammalian appendage regeneration (e.g., Zebularine PoC).
* Uses fine-tuned foundation models (like scGPT) to simulate how top-ranked EpiScreener compounds will rewire single-cell Gene Regulatory Networks (GRNs).

#### 3. [kg](./kg/) (Knowledge Graph)
The relational memory of RegenOS. 
* Maps interconnected nodes of screened small molecules, protein targets, biological pathways, and ADMET profiles.
* Integrated with PostgreSQL/Supabase for persistent storage and NetworkX/PyVis for interactive topological visualization of the compound "Leaderboard".

### 🟡 Deprecated / Pivoting

* **[seggpt-clearer](./seggpt-clearer/) & [wound-cnn-clearer](./wound-cnn-clearer/):** * *Status:* Development paused.
  * *Reasoning:* Attempts to use zero-shot segmenters (SAM, SegGPT) directly on Masson's Trichrome histology failed due to contrast limitations between native vs. fibrotic collagen. 
  * *Future:* Pivoting to a heavily fine-tuned, medical-specific U-Net trained exclusively on manually annotated appendage regeneration datasets to quantify *de novo* growth.

## 🛠️ Core Tech Stack
* **Machine Learning:** PyTorch, scGPT, ChemBERTa, Message Passing Neural Networks (MPNN)
* **Data Engineering & DBs:** PostgreSQL (Supabase), NetworkX, PyVis, Pandas, R
* **Bioinformatics:** Single-cell RNA-seq trajectory analysis, ADMET/Hydrogel stability modeling
* **Web/Frontend:** Streamlit / React

## 🤝 Collaboration & Usage
RegenOS is built to bridge the gap between computational power and wet-lab biological reality. We are actively looking for collaborators. Whether you are a computational biologist, an AI engineer, or a TechBio founder interested in epigenetic reprogramming and wound care, feel free to reach out, explore the architecture, or open a pull request!
