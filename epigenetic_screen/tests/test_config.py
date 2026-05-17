from pathlib import Path

from src.config import load_pipeline_config


def test_load_pipeline_config() -> None:
    cfg = load_pipeline_config(Path("configs/pipeline.yaml"))
    assert cfg.random_seed == 42
    assert "hdac2" in cfg.targets
    assert cfg.paths.input_path.name == "compounds.smi"

