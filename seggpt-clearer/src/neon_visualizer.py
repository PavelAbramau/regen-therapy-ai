import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import Meta's architecture
import sys
sys.path.append(str(Path(__file__).parent))
from models_seggpt import seggpt_vit_large_patch16_input896x448

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

TILE_SIZE = 1024 # Set this to match your prompt patch size (512 or 1024)

def preprocess_image(image_bgr, target_size=(448, 896)):
    """Resizes and normalizes image tiles for SegGPT ViT-Base."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_normalized = (img_resized / 255.0 - imagenet_mean) / imagenet_std
    img_tensor = torch.tensor(img_normalized).permute(2, 0, 1).float()
    return img_tensor.unsqueeze(0)

def preprocess_mask(mask_gray, target_size=(448, 896)):
    """SegGPT requires 3-channel masks for the prompt."""
    mask_resized = cv2.resize(mask_gray, target_size, interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.tensor(mask_resized).float() / 255.0
    mask_tensor = mask_tensor.unsqueeze(0).repeat(3, 1, 1)
    return mask_tensor.unsqueeze(0)

def get_prompt_pairs(prompt_dir):
    pairs = []
    for img_path in prompt_dir.glob("*_img.png"):
        mask_path = prompt_dir / img_path.name.replace("_img.png", "_mask.png")
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs

def validate_weights_file(weights_path: Path):
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing weights file: {weights_path}\n"
            "Download the official checkpoint and place it here, e.g.:\n"
            "  weights/seggpt_vit_large.pth"
        )

    # A real SegGPT checkpoint is large (GB-scale). Tiny files are usually error pages/text.
    min_expected_bytes = 100 * 1024 * 1024  # 100 MB safety floor
    size_bytes = weights_path.stat().st_size
    if size_bytes < min_expected_bytes:
        try:
            preview = weights_path.read_bytes()[:80].decode("utf-8", errors="replace")
        except OSError:
            preview = "<unreadable>"
        raise ValueError(
            f"Invalid checkpoint file: {weights_path} ({size_bytes} bytes).\n"
            f"File preview: {preview!r}\n"
            "Expected a real SegGPT .pth checkpoint (typically ~1.5 GB), not a placeholder/error file."
        )

def main():
    project_root = Path(__file__).resolve().parents[1]
    weights_path = project_root / "weights" / "seggpt_vit_large.pth"
    prompt_dir = project_root / "data" / "prompts"
    target_dir = project_root / "data" / "raw_targets"
    output_dir = project_root / "data" / "predictions"

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    print(f"📦 Loading Meta SegGPT weights on {device.type.upper()}...")
    validate_weights_file(weights_path)
    model = seggpt_vit_large_patch16_input896x448()
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()

    prompt_pairs = get_prompt_pairs(prompt_dir)
    if not prompt_pairs:
        print("❌ Error: No valid prompt pairs found.")
        return

    # Pre-process flashcards to save compute time
    processed_prompts = []
    for p_img, p_mask in prompt_pairs:
        img_bgr = cv2.imread(str(p_img))
        mask_gray = cv2.imread(str(p_mask), cv2.IMREAD_GRAYSCALE)
        pr_img_tensor = preprocess_image(img_bgr).to(device)
        pr_mask_tensor = preprocess_mask(mask_gray).to(device)
        processed_prompts.append((pr_img_tensor, pr_mask_tensor, p_img.name))

    targets = [p for p in target_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")]

    for target_path in targets:
        print(f"\n🔍 Analyzing unseen target: {target_path.name}...")
        target_img = cv2.imread(str(target_path))
        orig_h, orig_w = target_img.shape[:2]

        # 1. Padding Geometry
        # Calculate how much padding is needed to make dimensions perfectly divisible by TILE_SIZE
        pad_h = (TILE_SIZE - orig_h % TILE_SIZE) % TILE_SIZE
        pad_w = (TILE_SIZE - orig_w % TILE_SIZE) % TILE_SIZE

        # Use BORDER_REFLECT so the AI doesn't get confused by hard black borders on the edges
        padded_img = cv2.copyMakeBorder(target_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        padded_mask = np.zeros(padded_img.shape[:2], dtype=np.uint8)

        print(f"   📐 Original size: {orig_w}x{orig_h}. Padded for tiling: {padded_img.shape[1]}x{padded_img.shape[0]}")

        # 2. Sliding Window Inference
        # Chop the padded image into a grid and run SegGPT on each tile
        with torch.no_grad():
            for y in tqdm(range(0, padded_img.shape[0], TILE_SIZE), desc="Processing Grid"):
                for x in range(0, padded_img.shape[1], TILE_SIZE):

                    # Extract the tile
                    tile_bgr = padded_img[y:y+TILE_SIZE, x:x+TILE_SIZE]
                    tile_tensor = preprocess_image(tile_bgr).to(device)

                    tile_master_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

                    # Test this tile against all flashcards
                    for pr_img_tensor, pr_mask_tensor, _ in processed_prompts:
                        img_batch = torch.cat((pr_img_tensor, tile_tensor), dim=0)
                        seg_batch = torch.cat((pr_mask_tensor, torch.zeros_like(pr_mask_tensor)), dim=0)
                        num_patches = (tile_tensor.shape[-2] // model.patch_size) * (tile_tensor.shape[-1] // model.patch_size)
                        bool_masked_pos = torch.zeros((img_batch.shape[0], num_patches), device=device, dtype=torch.bool)

                        seg_type = torch.zeros(img_batch.shape[0], device=device, dtype=torch.long)
                        valid = torch.ones_like(seg_batch)
                        _, pred_patches, _ = model(
                            img_batch,
                            seg_batch,
                            bool_masked_pos=bool_masked_pos,
                            valid=valid,
                            seg_type=seg_type,
                        )
                        output = model.unpatchify(pred_patches) # Stitch ViT patches

                        pred_tensor = output[1].detach().cpu()
                        pred_img = pred_tensor.permute(1, 2, 0).numpy()

                        pred_gray = pred_img[:, :, 0]
                        pred_gray_resized = cv2.resize(pred_gray, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR)

                        pred_scaled = np.clip(pred_gray_resized * 255, 0, 255).astype(np.uint8)
                        binary_mask = (pred_scaled > 127).astype(np.uint8) * 255
                        tile_master_mask = cv2.bitwise_or(tile_master_mask, binary_mask)

                    # Place the predicted tile back into the master padded mask
                    padded_mask[y:y+TILE_SIZE, x:x+TILE_SIZE] = tile_master_mask

        # 3. Crop back to original dimensions
        master_mask = padded_mask[:orig_h, :orig_w]

        # Save the DEBUG MASK to verify the AI's "brainwaves"
        debug_name = f"DEBUG_MASK_{target_path.name}"
        cv2.imwrite(str(output_dir / debug_name), master_mask)

        # --- NEW NEON OVERLAY LOGIC ---
        print("   🔴 Generating neon diagnostic overlay...")
        overlay = target_img.copy()
        # Turn all flagged pixels bright Red (OpenCV uses BGR, so [0, 0, 255])
        overlay[master_mask > 127] = [0, 0, 255]

        # Blend the red mask with the original image at 50% opacity
        blended_diagnostic = cv2.addWeighted(target_img, 0.5, overlay, 0.5, 0)

        diagnostic_name = f"DIAGNOSTIC_{target_path.name}"
        cv2.imwrite(str(output_dir / diagnostic_name), blended_diagnostic)
        print(f"   👁️ Saved visual diagnostic: {diagnostic_name}")

        # 4. Apply erase logic
        print("   🧠 Tiling complete. Applying final topological erase...")
        cleaned_image = target_img.copy()
        cleaned_image[master_mask > 127] = [0, 0, 0]

        out_name = f"SegGPT_cleaned_{target_path.name}"
        cv2.imwrite(str(output_dir / out_name), cleaned_image)
        print(f"   💾 Cleaned image saved: {out_name}")

if __name__ == "__main__":
    main()
