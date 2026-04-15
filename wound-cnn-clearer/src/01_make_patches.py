from pathlib import Path
import random
import re
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


PATCH_SIZE = 256
VAL_SPLIT = 0.2
RANDOM_SEED = 42

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def ensure_dirs(paths: List[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for file_path in path.iterdir():
        if file_path.is_file():
            file_path.unlink()


def list_image_files(directory: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def extract_sample_key(filename_stem: str) -> Optional[Tuple[str, str, str]]:
    lowered = filename_stem.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered)
    sample_match = re.search(r"\d+\s+\d+", normalized)
    bn_match = re.search(r"bn\d+", normalized)
    area_match = re.search(r"[a-z]\d+[a-z]", normalized)
    if sample_match is None or bn_match is None or area_match is None:
        return None
    sample_token = sample_match.group(0).replace(" ", "-")
    return (sample_token, bn_match.group(0), area_match.group(0))


def select_primary_raw_files(raw_files: List[Path]) -> List[Path]:
    grouped: Dict[Tuple[str, str, str], List[Path]] = {}
    passthrough: List[Path] = []

    for raw_path in raw_files:
        key = extract_sample_key(raw_path.stem)
        if key is None:
            passthrough.append(raw_path)
            continue
        grouped.setdefault(key, []).append(raw_path)

    selected: List[Path] = []
    for key in sorted(grouped.keys()):
        candidates = grouped[key]
        preferred = next(
            (path for path in candidates if path.name.upper().startswith("RED_")),
            candidates[0],
        )
        selected.append(preferred)

    selected.extend(passthrough)
    return sorted(selected)


def build_mask_lookup(mask_files: List[Path]) -> Dict[Tuple[str, str, str], Path]:
    lookup: Dict[Tuple[str, str, str], Path] = {}
    for mask_path in mask_files:
        key = extract_sample_key(mask_path.stem)
        if key is None:
            continue
        lookup[key] = mask_path
    return lookup


def subtract_background(gray_image: np.ndarray, value: int = 20) -> np.ndarray:
    shifted = gray_image.astype(np.int16) - value
    shifted = np.clip(shifted, 0, 255)
    return shifted.astype(np.uint8)


def iter_patches(image: np.ndarray, mask: np.ndarray, patch_size: int) -> List[Tuple[np.ndarray, np.ndarray, int, int]]:
    patches: List[Tuple[np.ndarray, np.ndarray, int, int]] = []
    height, width = image.shape
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            image_patch = image[y : y + patch_size, x : x + patch_size]
            mask_patch = mask[y : y + patch_size, x : x + patch_size]
            patches.append((image_patch, mask_patch, x, y))
    return patches


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "data" / "gt_raw"
    mask_dir = project_root / "data" / "gt_masks"
    train_img_dir = project_root / "data" / "patches_img"
    train_mask_dir = project_root / "data" / "patches_mask"
    val_img_dir = project_root / "data" / "val_patches_img"
    val_mask_dir = project_root / "data" / "val_patches_mask"

    ensure_dirs([train_img_dir, train_mask_dir, val_img_dir, val_mask_dir])
    # Keep outputs deterministic and avoid mixing with previous runs.
    clear_directory(train_img_dir)
    clear_directory(train_mask_dir)
    clear_directory(val_img_dir)
    clear_directory(val_mask_dir)

    raw_files = list_image_files(raw_dir)
    if len(raw_files) == 0:
        raise ValueError(f"No raw images found in {raw_dir}")
    selected_raw_files = select_primary_raw_files(raw_files)

    mask_files = list_image_files(mask_dir)
    if len(mask_files) == 0:
        raise ValueError(f"No mask images found in {mask_dir}")
    mask_lookup = build_mask_lookup(mask_files)

    total_saved = 0
    skipped_black = 0
    pair_names: List[str] = []

    for raw_path in selected_raw_files:
        sample_key = extract_sample_key(raw_path.stem)
        if sample_key is None:
            print(f"Skipping {raw_path.name}: unable to parse sample key from filename.")
            continue

        mask_path = mask_lookup.get(sample_key)
        if mask_path is None:
            print(f"Skipping {raw_path.name}: matching mask not found for key {sample_key}.")
            continue

        raw_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_bgr is None:
            print(f"Skipping {raw_path.name}: failed to load raw image.")
            continue
        if mask_gray is None:
            print(f"Skipping {mask_path.name}: failed to load mask.")
            continue

        raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        raw_gray = subtract_background(raw_gray, value=20)

        # Ensure binary mask (0/255) even if source has soft values.
        _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

        if raw_gray.shape != mask_binary.shape:
            raise ValueError(
                f"Shape mismatch for {raw_path.name}: raw={raw_gray.shape}, mask={mask_binary.shape}"
            )

        for image_patch, mask_patch, x, y in iter_patches(raw_gray, mask_binary, PATCH_SIZE):
            if np.count_nonzero(mask_patch) == 0:
                skipped_black += 1
                continue

            patch_id = f"{raw_path.stem}_x{x}_y{y}"
            image_out = train_img_dir / f"{patch_id}.png"
            mask_out = train_mask_dir / f"{patch_id}.png"

            ok_img = cv2.imwrite(str(image_out), image_patch)
            ok_mask = cv2.imwrite(str(mask_out), mask_patch)
            if not (ok_img and ok_mask):
                raise IOError(f"Failed writing patch pair: {patch_id}")

            pair_names.append(patch_id)
            total_saved += 1

    if total_saved == 0:
        raise ValueError("No valid patches were generated after filtering black masks.")

    random.seed(RANDOM_SEED)
    random.shuffle(pair_names)
    val_count = int(len(pair_names) * VAL_SPLIT)
    val_names = pair_names[:val_count]

    for patch_id in val_names:
        src_img = train_img_dir / f"{patch_id}.png"
        src_mask = train_mask_dir / f"{patch_id}.png"
        dst_img = val_img_dir / f"{patch_id}.png"
        dst_mask = val_mask_dir / f"{patch_id}.png"
        shutil.move(str(src_img), str(dst_img))
        shutil.move(str(src_mask), str(dst_mask))

    print(f"Raw files discovered: {len(raw_files)}")
    print(f"Raw files selected (deduplicated pairs): {len(selected_raw_files)}")
    print(f"Total valid patches generated: {total_saved}")
    print(f"Patches skipped (100% black mask): {skipped_black}")
    print(f"Train patches: {len(pair_names) - val_count}")
    print(f"Validation patches: {val_count}")


if __name__ == "__main__":
    main()
