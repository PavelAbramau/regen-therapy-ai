# Regenerative Therapy AI - Wound Distance Pipeline

An automated, 2-model computer vision pipeline for processing 18,000+ murine ear wound images. The system handles instance segmentation, keypoint detection for anatomical landmarks, and automates millimeter-scale distance measurements for clinical trials.

## Architecture
1. **Segmentation Model (YOLOv11-Seg):** Isolates the ear tissue and the internal wound area.
2. **Pose/Keypoint Model (YOLOv11-Pose):** Identifies the proximal base and distal tip of the ear.
3. **Geometric Engine:** Calculates the shortest orthogonal distance from the wound edge to the proximal/distal regions using Shapely and Scipy, calibrated dynamically against an in-frame surgical ruler.

## Prerequisites (Windows/Mac/Linux)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* [Nextflow](https://www.nextflow.io/docs/latest/getstarted.html) installed.

## Quick Start

**1. Build the Docker Image**
Open your terminal in this directory and run:
\`\`\`bash
docker build -t regen-pipeline:latest .
\`\`\`

**2. Run the Pipeline**
Ensure your images are in `data/raw_images` and your weights are in the `weights/` folder.
\`\`\`bash
nextflow run main.nf -with-docker regen-pipeline:latest
\`\`\`

**3. View Results**
Check the `data/results/` folder for the annotated images and the `master_results.csv` containing the measurements.