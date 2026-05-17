from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    weights_path = project_root / "weights" / "best_model.pth"
    img_path = project_root / "data" / "raw_test_images" / "MAX_A1L_1-12_cropped.tif"
    output_dir = project_root / "data" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path}")
    if not img_path.exists():
        raise FileNotFoundError(f"Test image not found at {img_path}")

    device = get_device()

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    rgb_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (512, 512))
    rgb_img_float = np.float32(rgb_img) / 255.0
    input_tensor = ToTensorV2()(image=rgb_img)["image"].unsqueeze(0).float().to(device)

    # 1. Get the actual prediction (Binary)
    with torch.no_grad():
        output = model(input_tensor)
        prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        binary_mask = (prob_mask > 0.5).astype(np.float32)

    # 2. Setup Grad-CAM for 1 channel
    target_layers = [model.decoder.blocks[-1]]

    # Custom target for binary (channel 0)
    class BinarySegmentationTarget:
        def __init__(self, mask: np.ndarray) -> None:
            self.mask = torch.from_numpy(mask).to(device)

        def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
            return (model_output[0, :, :] * self.mask).sum()

    targets = [BinarySegmentationTarget(mask=binary_mask)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

    output_path = output_dir / "heatmap_artifacts.jpg"
    cv2.imwrite(str(output_path), cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    print(f"Heatmap saved at {output_path}")


if __name__ == "__main__":
    main()
