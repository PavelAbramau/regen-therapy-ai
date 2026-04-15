# wound_segm_pipeline_with_dist_v2


# Mouse Ear Wound Segmentation & Anatomical Measurement

An automated YOLOv11 pipeline that segments surgical wounds on mouse ears and calculates the true millimeter distance to the proximal skull base and distal ear tip. 

## Architecture
This pipeline uses a Dual-Model approach:
1. **Model A (Segmentation):** Extracts the precise wound polygon.
2. **Model B (Pose):** Detects 3 anatomical keypoints (Skull Base A, Distal Tip, Skull Base B) to build an "Equator" map.
The perimeter of the ear is mathematically sliced using the Equator to calculate shortest-path distances to true tissue edges.

##  Installation
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Open your terminal/command prompt and install the requirements:
   ```bash
   pip install -r requirements.txt






1)Install Miniconda
2)Run conda env create -f environment.yml
3)Place your images in "data" folder
4)Run python analyze_wound.py  
            --input_dir ./data \
            --output_dir ./results \
            --weights_seg ./best_seg.pt \
            --weights_pose ./best_pose.pt



Outputs
master_results.csv: Contains exact distances (px and mm), wound area, and the source folder path.

results_test/: Mirrors your input folder structure, containing annotated images with segmentation masks and exact measurement lin