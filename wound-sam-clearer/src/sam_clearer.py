import cv2
import numpy as np
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

# Global list to store the X, Y coordinates of your clicks
input_points = []

def click_event(event, x, y, flags, param):
    """Captures mouse clicks and draws a red dot on the image."""
    global input_points
    if event == cv2.EVENT_LBUTTONDOWN:
        input_points.append([x, y])
        print(f"Artifact marked at: ({x}, {y})")
        # Draw a temporary dot so you can see what you clicked
        cv2.circle(param['display_img'], (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Click Artifacts (Press ENTER when done)", param['display_img'])

def main():
    project_root = Path(__file__).resolve().parents[1]
    weights_path = project_root / "weights" / "sam_vit_b_01ec64.pth"
    
    # Define directories
    input_dir = project_root / "data" / "raw_test_images"
    output_dir = project_root / "data" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    # Find the first image in the directory
    valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    images_found = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_extensions]
    
    if not images_found:
        print(f"No images found in {input_dir}. Please drop a test image there!")
        return
        
    img_path = images_found[0] # Just grab the first one to test
    print(f"Loading image: {img_path.name}")

    # 1. Setup Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Loading SAM on {device}...")

    # 2. Load SAM
    sam = sam_model_registry["vit_b"](checkpoint=str(weights_path))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 3. Load Image
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 4. Generate Interactive UI
    print("\n--- INSTRUCTIONS ---")
    print("1. Click directly on the artifacts/blood vessels you want to remove.")
    print("2. Press the 'ENTER' key when you are finished clicking.")
    print("3. Press 'q' to quit without saving.\n")

    display_img = image_bgr.copy()
    cv2.imshow("Click Artifacts (Press ENTER when done)", display_img)
    cv2.setMouseCallback("Click Artifacts (Press ENTER when done)", click_event, {'display_img': display_img})

    # Wait for the user to press Enter (Key code 13)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            break
        elif key == ord('q'): # Quit
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    # 5. Run SAM if points were clicked
    if not input_points:
        print("No points clicked. Exiting.")
        return

    print("Processing with SAM... This may take a few seconds.")
    predictor.set_image(image_rgb)
    
    input_point_array = np.array(input_points)
    input_label_array = np.ones(len(input_points)) 

    masks, scores, logits = predictor.predict(
        point_coords=input_point_array,
        point_labels=input_label_array,
        multimask_output=False,
    )
    
    # 6. Apply the Mask
    artifact_mask = masks[0] 
    clean_image = image_bgr.copy()
    clean_image[artifact_mask == True] = [0, 0, 0] 

    # Save
    out_name = f"SAM_cleaned_{img_path.name}"
    cv2.imwrite(str(output_dir / out_name), clean_image)
    print(f"Success! Cleaned image saved to {output_dir / out_name}")

if __name__ == "__main__":
    main()