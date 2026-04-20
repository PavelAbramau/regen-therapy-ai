# Computear Pipeline: Automated Clinical Image Analysis

A scalable, containerized machine learning pipeline for automated wound segmentation and spatial analysis in regenerative medicine. 

## 🧬 Overview
Analyzing clinical tissue images at scale is a massive bottleneck in biological research. This project automates the extraction of spatial metrics from clinical imagery. Designed initially to process a dataset of 18,000+ images, this pipeline utilizes a dual-model computer vision architecture (YOLO) wrapped in a highly parallelized Nextflow scatter-gather workflow.

**Key Features:**
* **Dual-Model Inference:** Utilizes fine-tuned YOLO segmentation and pose-estimation models via PyTorch to identify wound boundaries and calculate critical distance metrics.
* **Massively Parallel:** Built with Nextflow to dynamically distribute workloads. Processes images in parallel, enabling seamless scaling from a local MacBook to a cloud supercomputer cluster.
* **Fully Containerized:** Zero dependency hell. The entire PyTorch and OpenCV environment is containerized via Docker for instant reproducibility.
* **Automated Data Aggregation:** Automatically merges thousands of individual inference outputs into a single, clean master CSV for downstream statistical analysis.

## 🛠️ Architecture & Tech Stack
* **Machine Learning:** PyTorch, Ultralytics (YOLOv11), OpenCV (Headless)
* **Orchestration:** Nextflow (DSL2)
* **Infrastructure:** Docker, Linux
* **Hardware Profile:** Agnostic (Tested on Apple Silicon MPS, Nvidia CUDA, and CPU)

## 📂 Project Structure
```text
computear-pipeline/
├── analyze_wound.py       # Core inference and processing script
├── weights/               # Trained .pt models (segmentation & pose)
├── Dockerfile             # Defines the isolated execution environment
├── main.nf                # Nextflow orchestration script
├── requirements.txt       # Python dependencies
└── README.md              # Documentation

Quick Start Guide
Prerequisites
You must have the following installed on your system:

Docker

Nextflow

1. Build the Environment
Clone this repository and build the Docker image. This will download the necessary PyTorch and OpenCV libraries.

Bash
git clone [https://github.com/PavelAbramau/computear-pipeline.git](https://github.com/PavelAbramau/computear-pipeline.git)
cd computear-pipeline
docker build -t distance-tool:latest .



2. Run the Pipeline
Execute the pipeline using Nextflow. Point the --input_dir to your folder of raw images (supports nested sub-folders and .jpg, .jpeg, .png formats).

Bash
nextflow run main.nf -with-docker distance-tool:latest --input_dir "data_test" --output_dir "results"
3. Pipeline Execution Flow (Scatter-Gather)
Scatter (ANALYZE_IMAGES): Nextflow recursively scans the input directory and spins up isolated Docker containers for each image in parallel.

Process: analyze_wound.py generates an annotated image mask and a unique micro-CSV containing the calculated spatial metrics.

Gather (MERGE_RESULTS): Nextflow catches all individual outputs and safely merges them into a single master_results.csv in the output directory.

📊 Output
Upon completion, your --output_dir will contain:

master_results.csv: A complete dataset containing the image IDs and their calculated distances/metrics.

Annotated image files with overlaid segmentation masks and distance markers for visual validation.