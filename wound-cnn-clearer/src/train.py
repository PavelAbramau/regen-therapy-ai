from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import WoundPatchDataset


EPOCHS = 25
LEARNING_RATE = 1e-4
BATCH_SIZE = 4  # Increase to 8 if memory allows.


class CombinedLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass")
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        targets = targets.long()
        return self.dice(logits, targets) + self.ce(logits, targets)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_idx: int,
) -> float:
    model.train()
    running_loss = 0.0

    progress = tqdm(dataloader, desc=f"Epoch {epoch_idx} [Train]", leave=False)
    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(1, len(dataloader))


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch_idx: int,
) -> float:
    model.eval()
    running_loss = 0.0

    progress = tqdm(dataloader, desc=f"Epoch {epoch_idx} [Val]", leave=False)
    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(1, len(dataloader))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    weights_dir = project_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weights_dir / "best_model.pth"

    device = get_device()
    print(f"Using device: {device}")

    train_dataset = WoundPatchDataset(
        image_dir=str(project_root / "data" / "patches_img"),
        mask_dir=str(project_root / "data" / "patches_mask"),
        train=True,
    )
    val_dataset = WoundPatchDataset(
        image_dir=str(project_root / "data" / "val_patches_img"),
        mask_dir=str(project_root / "data" / "val_patches_mask"),
        train=False,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
    ).to(device)

    criterion = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch_idx=epoch,
        )
        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch_idx=epoch,
        )

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best model to {checkpoint_path} (val_loss={val_loss:.4f})")

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
