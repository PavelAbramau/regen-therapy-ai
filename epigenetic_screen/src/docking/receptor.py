from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ReceptorSpec:
    target_name: str
    receptor_path: Path
    box_center: tuple[float, float, float]
    box_size: tuple[float, float, float]
    exhaustiveness: int = 12
    n_poses: int = 9


def load_receptor_spec(path: Path, receptor_root: Path | None = None) -> ReceptorSpec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key in ["target_name", "receptor_path", "box_center", "box_size"]:
        if key not in raw:
            raise ValueError(f"Missing {key} in target config: {path}")

    receptor_path = Path(raw["receptor_path"])
    if not receptor_path.is_absolute() and receptor_root is not None:
        receptor_path = (receptor_root / receptor_path.name).resolve()

    box_center = tuple(float(v) for v in raw["box_center"])
    box_size = tuple(float(v) for v in raw["box_size"])
    if len(box_center) != 3 or len(box_size) != 3:
        raise ValueError(f"box_center and box_size must be length-3 vectors: {path}")

    return ReceptorSpec(
        target_name=str(raw["target_name"]),
        receptor_path=receptor_path,
        box_center=box_center,  # type: ignore[arg-type]
        box_size=box_size,  # type: ignore[arg-type]
        exhaustiveness=int(raw.get("exhaustiveness", 12)),
        n_poses=int(raw.get("n_poses", 9)),
    )

