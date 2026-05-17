from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    input_path: Path
    output_root: Path
    receptor_dir: Path
    target_configs_dir: Path


@dataclass
class AdmetConfig:
    featurizer: str = "molgraphconv"
    endpoints: list[str] = field(default_factory=list)
    use_mock_models: bool = True


@dataclass
class DockingConfig:
    use_mock_if_unavailable: bool = True
    random_seed: int = 42


@dataclass
class LipinskiConfig:
    soft_violation_max: int = 1
    drop_failures: bool = False


@dataclass
class RankingConfig:
    descriptor_terms: dict[str, float] = field(default_factory=dict)
    admet_terms: dict[str, float] = field(default_factory=dict)
    docking_weight: float = 1.0
    shortlist_top_n: int = 100


@dataclass
class PipelineConfig:
    random_seed: int
    n_jobs: int
    targets: list[str]
    paths: PathsConfig
    admet: AdmetConfig
    docking: DockingConfig
    lipinski: LipinskiConfig
    ranking: RankingConfig


def _to_path(base_dir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p).resolve()


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    cfg_path = Path(path).resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    base_dir = cfg_path.parent.parent

    paths_raw = raw.get("paths", {})
    paths = PathsConfig(
        input_path=_to_path(base_dir, paths_raw.get("input_path", "data/raw/compounds.smi")),
        output_root=_to_path(base_dir, paths_raw.get("output_root", "output")),
        receptor_dir=_to_path(base_dir, paths_raw.get("receptor_dir", "data/receptors")),
        target_configs_dir=_to_path(base_dir, paths_raw.get("target_configs_dir", "configs/targets")),
    )

    return PipelineConfig(
        random_seed=int(raw.get("random_seed", 42)),
        n_jobs=int(raw.get("n_jobs", 1)),
        targets=list(raw.get("targets", [])),
        paths=paths,
        admet=AdmetConfig(**raw.get("admet", {})),
        docking=DockingConfig(**raw.get("docking", {})),
        lipinski=LipinskiConfig(**raw.get("lipinski", {})),
        ranking=RankingConfig(**raw.get("ranking", {})),
    )


def load_target_config(target_yaml_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(target_yaml_path.read_text(encoding="utf-8")) or {}
    required = ["target_name", "receptor_path", "box_center", "box_size"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Target config missing required keys {missing}: {target_yaml_path}")
    return config

