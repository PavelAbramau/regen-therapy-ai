
import os
import cv2
import torch
import numpy as np
from pathlib import Path
import sys

# Import Meta's architecture
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
from models_seggpt import seggpt_vit_large_patch16_input896x448

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
TILE_SIZE = 512 
MODEL_INPUT_H = 896
MODEL_INPUT_W = 448

def preprocess_image(image_bgr):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (MODEL_INPUT_W, MODEL_INPUT_H))
    img_normalized = (img_resized / 255.0 - imagenet_mean) / imagenet_std
    return torch.tensor(img_normalized).permute(2, 0, 1).float().unsqueeze(0)

def preprocess_mask(mask_gray):
    mask_resized = cv2.resize(mask_gray, (MODEL_INPUT_W, MODEL_INPUT_H), interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.tensor(mask_resized).float() / 255.0
    return mask_tensor.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0) 

def main():
    project_root = PROJECT_ROOT
    exp_module_root = Path(__file__).resolve().parent
    weights_path = project_root / "weights" / "seggpt_vit_large.pth"
    prompt_dir = exp_module_root / "data" / "experiment_prompts"
    target_dir = exp_module_root / "data" / "experiment_target"
    output_dir = exp_module_root / "data" / "experiment_output"
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"🧪 Starting EXPERIMENT on {device.type.upper()}...")
    
    model = seggpt_vit_large_patch16_input896x448()
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'], strict=False)
    model.to(device)
    model.eval()

    # Load EXACTLY one prompt pair for the experiment
    prompts = list(prompt_dir.glob("*_img.png"))
    if not prompts:
        print("❌ Error: Drop exactly one *_img.png and *_mask.png pair into data/experiment_prompts/")
        return
    
    p_img_path = prompts[0]
    p_mask_path = prompt_dir / p_img_path.name.replace("_img.png", "_mask.png")
    
    pr_img_tensor = preprocess_image(cv2.imread(str(p_img_path))).to(device)
    pr_mask_tensor = preprocess_mask(cv2.imread(str(p_mask_path), cv2.IMREAD_GRAYSCALE)).to(device)
    
    targets = list(target_dir.glob("*.*"))
    if not targets:
        print("❌ Error: Drop one 2048x2048 image into data/experiment_target/")
        return
        
    target_path = targets[0]
    target_img = cv2.imread(str(target_path))
    orig_h, orig_w = target_img.shape[:2]
    
    pad_h = (TILE_SIZE - orig_h % TILE_SIZE) % TILE_SIZE
    pad_w = (TILE_SIZE - orig_w % TILE_SIZE) % TILE_SIZE
    padded_img = cv2.copyMakeBorder(target_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    
    # We use float32 to capture exact AI confidence percentages, not just 0 or 255
    master_confidence_map = np.zeros(padded_img.shape[:2], dtype=np.float32)

    print(f"🔍 Processing {target_path.name} ({orig_w}x{orig_h}) in tiles...")
    with torch.no_grad():
        for y in range(0, padded_img.shape[0], TILE_SIZE):
            for x in range(0, padded_img.shape[1], TILE_SIZE):
                tile_tensor = preprocess_image(padded_img[y:y+TILE_SIZE, x:x+TILE_SIZE]).to(device)
                
                # 1. Create the batches FIRST
                img_batch = torch.cat((pr_img_tensor, tile_tensor), dim=0)
                seg_batch = torch.cat((pr_mask_tensor, torch.zeros_like(pr_mask_tensor)), dim=0)
                
                # 2. THEN create the masked pos tensor
                bool_masked_pos = torch.zeros(img_batch.shape[0], model.patch_embed.num_patches, dtype=torch.bool).to(device)
                
                # 3. Create dummy "valid" mask to prevent the loss calculation crash
                valid = torch.ones_like(seg_batch).to(device)
                
                # 4. Forward Pass (model returns loss, prediction, and mask)
                loss, pred, mask = model(img_batch, seg_batch, bool_masked_pos, valid)
                
                # 5. Unpatchify the raw prediction
                output_unpatched = model.unpatchify(pred)
                
                pred_gray = output_unpatched[1].detach().cpu().permute(1, 2, 0).numpy()[:, :, 0]
                pred_resized = cv2.resize(pred_gray, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR)
                
                # Store the raw probability (0.0 to 1.0)
                master_confidence_map[y:y+TILE_SIZE, x:x+TILE_SIZE] = pred_resized

    master_confidence_map = master_confidence_map[:orig_h, :orig_w]

    # --- 1. THE HEATMAP OUTPUT ---
    # Convert probabilities to a 0-255 scale and apply Jet colormap
    # Red = 100% sure it's honeycomb. Blue = 0% sure. 
    heatmap_norm = np.clip(master_confidence_map * 255, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / f"HEATMAP_{target_path.name}"), heatmap_color)
    print(f"🔥 Saved AI Confidence Heatmap.")

    # --- 2. THE THRESHOLD OVERLAY ---
    # Only highlight pixels where AI is > 80% confident (0.8), avoiding noise saturation
    confident_mask = (master_confidence_map > 0.8).astype(np.uint8) * 255
    overlay = target_img.copy()
    overlay[confident_mask == 255] = [0, 0, 255] 
    blended = cv2.addWeighted(target_img, 0.6, overlay, 0.4, 0)
    cv2.imwrite(str(output_dir / f"OVERLAY_{target_path.name}"), blended)
    print(f"🎯 Saved 80% Confidence Overlay.")

if __name__ == "__main__":
    main()