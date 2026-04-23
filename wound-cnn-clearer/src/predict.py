import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path
from albumentations.pytorch import ToTensorV2

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def pad_to_multiple(image: np.ndarray, multiple: int = 32):
    """Pads the image to ensure dimensions are divisible by 32 (required by U-Net)."""
    h, w = image.shape[:2]
    new_h = ((h - 1) // multiple + 1) * multiple
    new_w = ((w - 1) // multiple + 1) * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return image, h, w

def main():
    # 1. Setup paths
    project_root = Path(__file__).resolve().parents[1]
    weights_path = project_root / "weights" / "best_model.pth"
    input_dir = project_root / "data" / "raw_test_images"  # Create this folder
    output_dir = project_root / "data" / "predictions"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    # 2. Load the Exact Architecture from train.py
    print("Loading model architecture...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # No need to download imagenet weights for inference
        in_channels=3,
        classes=1,
    )

    # 3. Inject your trained weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval() # CRITICAL: Turns off dropout and batch normalization

    # 4. Transform (Matches dataset.py exactly)
    transform = ToTensorV2()

    # 5. Process the images
    valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    images_found = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_extensions]
    
    if not images_found:
        print(f"No images found in {input_dir}. Please add some to test!")
        return

    print(f"Found {len(images_found)} images. Starting inference...")
    
    for img_path in images_found:
        # Read and convert to RGB
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Pad to multiple of 32
        padded_img, orig_h, orig_w = pad_to_multiple(image_rgb, 32)
        
        # Transform (No division by 255, keeping it identical to dataset.py logic)
        input_tensor = transform(image=padded_img)["image"].float().unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(input_tensor)
            # The model outputs raw logits (because you used BCEWithLogitsLoss). We need Sigmoid.
            prob_mask = torch.sigmoid(logits).squeeze().cpu().numpy()
            
        # Crop the mask back to the original image dimensions
        prob_mask = prob_mask[:orig_h, :orig_w]
        
        # Convert probabilities to a hard binary mask (0 or 255)
        binary_mask = (prob_mask > 0.5).astype(np.uint8) * 255
        
        # Save output
        out_name = f"cleaned_{img_path.name}"
        cv2.imwrite(str(output_dir / out_name), binary_mask)
        print(f"Saved: {out_name}")

if __name__ == "__main__":
    main()