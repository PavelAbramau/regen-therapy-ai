from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


VALID_IMAGE_EXTENSIONS: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def get_train_transforms(resize_to: Optional[Tuple[int, int]] = None) -> A.Compose:
    transforms: List[A.BasicTransform] = []
    if resize_to is not None:
        transforms.append(A.Resize(height=resize_to[0], width=resize_to[1]))

    transforms.extend(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.12,
                rotate_limit=45,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.6,
            ),
            # Grid distortion is important for curved/irregular nerve fiber structures.
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def get_val_transforms(resize_to: Optional[Tuple[int, int]] = None) -> A.Compose:
    transforms: List[A.BasicTransform] = []
    if resize_to is not None:
        transforms.append(A.Resize(height=resize_to[0], width=resize_to[1]))
    transforms.append(ToTensorV2())
    return A.Compose(transforms)


class WoundPatchDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        train: bool = True,
        resize_to: Optional[Tuple[int, int]] = None,
        transforms: Optional[A.Compose] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory does not exist: {self.mask_dir}")

        self.samples = self._build_pairs()
        if not self.samples:
            raise ValueError(
                f"No valid image/mask pairs found in {self.image_dir} and {self.mask_dir}"
            )

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = (
                get_train_transforms(resize_to=resize_to)
                if train
                else get_val_transforms(resize_to=resize_to)
            )

    def _build_pairs(self) -> List[Tuple[Path, Path]]:
        image_paths = sorted(
            p
            for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        mask_lookup = {
            p.stem: p
            for p in self.mask_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        }

        pairs: List[Tuple[Path, Path]] = []
        missing_masks: List[str] = []
        for image_path in image_paths:
            mask_path = mask_lookup.get(image_path.stem)
            if mask_path is None:
                missing_masks.append(image_path.name)
                continue
            pairs.append((image_path, mask_path))

        if missing_masks:
            print(
                f"Warning: {len(missing_masks)} images were skipped due to missing masks "
                f"in {self.mask_dir}."
            )

        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[idx]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        mask = mask.astype(np.float32) / 255.0

        transformed = self.transforms(image=image, mask=mask)
        image_tensor = transformed["image"].float()
        mask_tensor = transformed["mask"].float()

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        return image_tensor, mask_tensor


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    image_dir = root_dir / "data" / "patches_img"
    mask_dir = root_dir / "data" / "patches_mask"

    dataset = WoundPatchDataset(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        train=True,
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    images, masks = next(iter(dataloader))
    print(f"Images shape: {images.shape}")
    print(f"Masks shape:  {masks.shape}")
