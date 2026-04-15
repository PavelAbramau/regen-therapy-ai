# V2.3 Pipeline Architecture: Diagnostic Visual Reference

## 1. Goal
Visually verify the YOLO-Pose keypoint order and establish a clear anatomical border for proximal and distal regions. Halt complex distance math until visuals are confirmed.

## 2. Models
* **Model A (Segmentation):** Finds the `wound` polygon.
* **Model B (Pose):** Finds 3 keypoints. We need to visually determine their order.

## 3. The Logic (Diagnostic Mode)
1. Convert `wound` mask to a Shapely `Polygon`.
2. Extract the 3 keypoints from Model B.
3. Pass the keypoints to the drawing block.
4. Draw a complete Triangle connecting all 3 points to visually identify the anatomical regions.
5. Draw a distinct "Border Line" between Point 0 and Point 1 to see if that represents the true Skull Base. 
6. Output the image. Do not calculate Shapely shortest-distances at this stage.