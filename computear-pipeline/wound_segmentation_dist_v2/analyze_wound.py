import os
import argparse
import glob
import cv2
import numpy as np
import pandas as pd
from scipy import stats
from ultralytics import YOLO
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points, split, linemerge


# --- CONFIGURATION ---
KNOWN_RULER_MM = 10.0  

def calibrate_ruler(image_rgb):
    """Calculates mm/px ratio using the surgical ruler tick marks."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)
    edges = cv2.Canny(gray_enhanced, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=15, maxLineGap=5)
    
    if lines is None: return 0.0092 
        
    line_data = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle < 0: angle += 180
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        line_data.append({'angle': angle, 'cx': cx, 'cy': cy})
        
    angles = [ld['angle'] for ld in line_data]
    rounded_angles = np.round(np.array(angles) / 5.0) * 5.0
    mode_result = stats.mode(rounded_angles, keepdims=True)
    dominant_angle = mode_result.mode[0]
    
    valid_ticks = [ld for ld in line_data if abs(ld['angle'] - dominant_angle) <= 5]
    if len(valid_ticks) < 2: return 0.0092

    is_mostly_horizontal = 45 <= dominant_angle <= 135
    if is_mostly_horizontal:
        valid_ticks.sort(key=lambda item: item['cx'])
    else:
        valid_ticks.sort(key=lambda item: item['cy'])

    distances = []
    for i in range(len(valid_ticks) - 1):
        pt1, pt2 = valid_ticks[i], valid_ticks[i+1]
        dist = np.sqrt((pt2['cx'] - pt1['cx'])**2 + (pt2['cy'] - pt1['cy'])**2)
        distances.append(dist)
        
    distances = np.array(distances)
    valid_distances = distances[(distances > 60) & (distances < 150)]
    
    if len(valid_distances) == 0: return 0.0092
    return 1.0 / np.median(valid_distances)


def calculate_regions_and_distances(keypoints, wound_poly, ear_poly):
    """
    Slices the ear perimeter using the Equator to find true tissue distances.
    """
    if len(keypoints) < 3 or ear_poly is None:
        return None

    # --- REGION MATH (NumPy) ---
    pt0_coords = np.array([keypoints[0][0], keypoints[0][1]])
    pt1_coords = np.array([keypoints[1][0], keypoints[1][1]])  # Tip
    pt2_coords = np.array([keypoints[2][0], keypoints[2][1]])

    base_mid = (pt0_coords + pt2_coords) / 2.0
    axis_vec = pt1_coords - base_mid
    axis_norm = np.linalg.norm(axis_vec)
    if axis_norm < 1e-9:
        return None

    ear_mid = base_mid + (axis_vec / 2.0)

    perp_vec = np.array([-axis_vec[1], axis_vec[0]])
    perp_vec = perp_vec / np.linalg.norm(perp_vec)

    line_length = 2000
    eq_p1 = ear_mid + (perp_vec * line_length)
    eq_p2 = ear_mid - (perp_vec * line_length)

    equator_line = LineString(
        [(float(eq_p1[0]), float(eq_p1[1])), (float(eq_p2[0]), float(eq_p2[1]))]
    )

    # --- SHAPE CUTTING MATH (Shapely) ---
    pt1_shape = Point(float(pt1_coords[0]), float(pt1_coords[1]))
    base_mid_shape = Point(float(base_mid[0]), float(base_mid[1]))

    # Split the ear perimeter using the Equator (LinearRing is not supported by split())
    ear_perimeter = LineString(ear_poly.exterior.coords)
    perimeter_segments = split(ear_perimeter, equator_line)

    # Fallback if split fails (equator doesn't intersect ear perfectly)
    if len(perimeter_segments.geoms) < 2:
        proximal_boundary = LineString(
            [
                (float(pt0_coords[0]), float(pt0_coords[1])),
                (float(pt2_coords[0]), float(pt2_coords[1])),
            ]
        )
        distal_boundary = pt1_shape
    else:
        distal_lines = []
        proximal_lines = []

        # Sort the cut perimeter pieces into Proximal vs Distal
        for segment in perimeter_segments.geoms:
            seg_mid = segment.interpolate(0.5, normalized=True)
            # If the segment's middle is closer to the tip, it's Distal tissue
            if seg_mid.distance(pt1_shape) < seg_mid.distance(base_mid_shape):
                distal_lines.append(segment)
            else:
                proximal_lines.append(segment)

        if not distal_lines or not proximal_lines:
            proximal_boundary = LineString(
                [
                    (float(pt0_coords[0]), float(pt0_coords[1])),
                    (float(pt2_coords[0]), float(pt2_coords[1])),
                ]
            )
            distal_boundary = pt1_shape
        else:
            # Merge the broken lines back together into unified boundaries
            distal_boundary = (
                linemerge(distal_lines) if len(distal_lines) > 1 else distal_lines[0]
            )
            proximal_boundary = (
                linemerge(proximal_lines) if len(proximal_lines) > 1 else proximal_lines[0]
            )

    # --- DISTANCE MATH (Shapely) ---
    pt_wound_prox, pt_prox_edge = nearest_points(wound_poly, proximal_boundary)
    dist_proximal = pt_wound_prox.distance(pt_prox_edge)

    pt_wound_distal, pt_distal_edge = nearest_points(wound_poly, distal_boundary)
    dist_distal = pt_wound_distal.distance(pt_distal_edge)

    return {
        'pt0': (int(pt0_coords[0]), int(pt0_coords[1])),
        'pt1': (int(pt1_coords[0]), int(pt1_coords[1])),
        'pt2': (int(pt2_coords[0]), int(pt2_coords[1])),
        'base_mid': (int(base_mid[0]), int(base_mid[1])),
        'eq_p1': (int(eq_p1[0]), int(eq_p1[1])),
        'eq_p2': (int(eq_p2[0]), int(eq_p2[1])),
        # New Tissue Edge Points
        'wound_prox_pt': (int(pt_wound_prox.x), int(pt_wound_prox.y)),
        'skull_closest_pt': (int(pt_prox_edge.x), int(pt_prox_edge.y)),
        'wound_distal_pt': (int(pt_wound_distal.x), int(pt_wound_distal.y)),
        'tip_closest_pt': (int(pt_distal_edge.x), int(pt_distal_edge.y)),
        'dist_proximal_px': dist_proximal,
        'dist_distal_px': dist_distal,
    }


def _draw_dashed_line_bgr(img_bgr, p0, p1, color, thickness=2, dash_len=12, gap_len=8):
    """Draw a dashed segment from p0 to p1 (image coordinates as ints)."""
    a = np.array([float(p0[0]), float(p0[1])], dtype=np.float64)
    b = np.array([float(p1[0]), float(p1[1])], dtype=np.float64)
    vec = b - a
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return
    u = vec / length
    t = 0.0
    draw = True
    while t < length:
        if draw:
            t_end = min(t + dash_len, length)
            q0 = tuple(np.round(a + u * t).astype(np.int32))
            q1 = tuple(np.round(a + u * t_end).astype(np.int32))
            cv2.line(img_bgr, q0, q1, color, thickness)
            t = t_end
        else:
            t = min(t + gap_len, length)
        draw = not draw


def process_image(img_path, model_seg, model_pose, out_dir, rel_dir):
    img_name = os.path.basename(img_path)
    
    # Load correctly
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Run model A (segmentation): wound mask
    results_seg = model_seg(img_bgr, conf=0.05, verbose=False)[0]
    # Run model B (pose): ear keypoints
    results_pose = model_pose(img_bgr, conf=0.25, verbose=False)[0]

    if results_seg.masks is None:
        print(f"[{img_name}] No segmentation masks detected.")
        return None

    classes = results_seg.boxes.cls.cpu().numpy()
    names = results_seg.names
    masks_xy = results_seg.masks.xy

    polygons = {}
    for cls_id, mask_coords in zip(classes, masks_xy):
        class_name = names[int(cls_id)].lower()
        if len(mask_coords) >= 3: 
            poly = Polygon(mask_coords)
            if class_name not in polygons or poly.area > polygons[class_name].area:
                polygons[class_name] = poly

    if 'wound' not in polygons:
        print(f"[{img_name}] No wound detected.")
        return None

    wound_poly = polygons['wound']
    wound_area_px = wound_poly.area
    
    # Ruler Call
    px_to_mm_ratio = calibrate_ruler(img_rgb)

    # Geometry Call
    geom_data = None
    if results_pose.keypoints is not None and len(results_pose.keypoints.xy) > 0:
        geom_data = calculate_regions_and_distances(
            results_pose.keypoints.xy[0].cpu().numpy(),
            wound_poly,
            polygons['ear'],
        )

    # --- DRAWING VISUALIZATIONS ---
    pts_wound = np.array(wound_poly.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
    overlay = img_bgr.copy()
    cv2.fillPoly(overlay, [pts_wound], (255, 0, 0))
    img_bgr = cv2.addWeighted(overlay, 0.5, img_bgr, 0.5, 0)
    cv2.polylines(img_bgr, [pts_wound], isClosed=True, color=(255, 0, 0), thickness=5)

    # Segmented ear region visualization: semi-transparent green overlay
    if 'ear' in polygons:
        pts_ear = np.array(polygons['ear'].exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
        ear_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        cv2.fillPoly(ear_mask, [pts_ear], 255)

        green_overlay = np.full(img_bgr.shape, (0, 255, 0), dtype=np.uint8)
        blended_green = cv2.addWeighted(img_bgr, 0.6, green_overlay, 0.4, 0)
        img_bgr = np.where(ear_mask[..., None] == 0, img_bgr, blended_green)
        cv2.polylines(img_bgr, [pts_ear], isClosed=True, color=(0, 255, 0), thickness=3)

    if geom_data:
        pt0 = geom_data['pt0']
        pt1 = geom_data['pt1']
        pt2 = geom_data['pt2']
        base_mid = geom_data['base_mid']
        eq_p1 = geom_data['eq_p1']
        eq_p2 = geom_data['eq_p2']
        wound_prox_pt = geom_data['wound_prox_pt']
        skull_closest_pt = geom_data['skull_closest_pt']
        wound_distal_pt = geom_data['wound_distal_pt']
        tip_closest_pt = geom_data['tip_closest_pt']

        # Triangle (all edges)
        cv2.line(img_bgr, pt0, pt1, (0, 255, 0), 2)
        cv2.line(img_bgr, pt1, pt2, (0, 255, 0), 2)
        cv2.line(img_bgr, pt2, pt0, (0, 255, 0), 2)

        # Base midpoint + major axis (base_mid -> tip)
        cv2.circle(img_bgr, base_mid, 6, (255, 255, 0), -1)
        _draw_dashed_line_bgr(img_bgr, base_mid, pt1, (255, 255, 0), thickness=2)

        # Equator between proximal and distal regions
        cv2.line(img_bgr, eq_p1, eq_p2, (255, 0, 255), 4)

        cv2.putText(
            img_bgr,
            "PROXIMAL REGION",
            (base_mid[0], base_mid[1] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img_bgr,
            "DISTAL REGION",
            (pt1[0], pt1[1] + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            2,
        )

        # Proximal shortest distance (wound -> skull base line)
        cv2.line(img_bgr, wound_prox_pt, skull_closest_pt, (0, 0, 255), 4)
        cv2.circle(img_bgr, wound_prox_pt, 6, (0, 0, 255), -1)
        cv2.circle(img_bgr, skull_closest_pt, 6, (0, 0, 255), -1)

        # Distal shortest distance (wound -> tip)
        cv2.line(img_bgr, wound_distal_pt, tip_closest_pt, (0, 255, 255), 4)
        cv2.circle(img_bgr, wound_distal_pt, 6, (0, 255, 255), -1)
        cv2.circle(img_bgr, tip_closest_pt, 6, (0, 255, 255), -1)

        # Keypoint index labels
        cv2.putText(img_bgr, "0", (pt0[0] + 8, pt0[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(img_bgr, "1", (pt1[0] + 8, pt1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(img_bgr, "2", (pt2[0] + 8, pt2[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    # --- ONLY ONE SAVING BLOCK ---
    save_dir = os.path.join(out_dir, rel_dir)
    os.makedirs(save_dir, exist_ok=True)
    out_img_path = os.path.join(save_dir, f"{os.path.splitext(img_name)[0]}_annotated.jpg")
    cv2.imwrite(out_img_path, img_bgr)

    def safe_round(val, digits):
        return round(val, digits) if val is not None else None

    # Compile Math Data
    dist_proximal_px = geom_data['dist_proximal_px'] if geom_data else None
    dist_distal_px = geom_data['dist_distal_px'] if geom_data else None

    wound_area_mm2 = (
        wound_area_px * (px_to_mm_ratio**2)
        if px_to_mm_ratio is not None
        else None
    )
    dist_proximal_mm = (
        dist_proximal_px * px_to_mm_ratio
        if dist_proximal_px is not None and px_to_mm_ratio is not None
        else None
    )
    dist_distal_mm = (
        dist_distal_px * px_to_mm_ratio
        if dist_distal_px is not None and px_to_mm_ratio is not None
        else None
    )

    result_dict = {
        'filename': img_name,
        'folder_path': rel_dir,
        'px_to_mm_ratio': safe_round(px_to_mm_ratio, 4),
        'wound_area_px': safe_round(wound_area_px, 3),
        'wound_area_mm2': safe_round(wound_area_mm2, 3),
        'dist_proximal_px': safe_round(dist_proximal_px, 3),
        'dist_distal_px': safe_round(dist_distal_px, 3),
        'dist_proximal_mm': safe_round(dist_proximal_mm, 3),
        'dist_distal_mm': safe_round(dist_distal_mm, 3),
    }
    
    return result_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--weights_seg', type=str, required=True)
    parser.add_argument('--weights_pose', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_seg = YOLO(args.weights_seg)
    model_pose = YOLO(args.weights_pose)
    
    image_paths = []
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    for ext in extensions:
        search_pattern = os.path.join(args.input_dir, '**', ext)
        image_paths.extend(glob.glob(search_pattern, recursive=True))

    print(f"Found {len(image_paths)} images. Starting pipeline...")
    
    all_results = []
    for img_path in image_paths:
        rel_dir = os.path.relpath(os.path.dirname(img_path), args.input_dir)
        res = process_image(img_path, model_seg, model_pose, args.output_dir, rel_dir)
        if res:
            all_results.append(res)
            
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(args.output_dir, "master_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"SUCCESS! Processed {len(all_results)} images.")
    else:
        print("No valid results were generated.")

if __name__ == "__main__":
    main()