from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Running `python src/main.py …` sets __package__ to None and breaks `from src.*` imports unless
# the project root (`epigenetic_screen/`) is on sys.path. `python -m src.main` does not need this.
if __package__ is None:  # pragma: no cover
    _project_root = Path(__file__).resolve().parent.parent
    _root_s = str(_project_root)
    if _root_s not in sys.path:
        sys.path.insert(0, _root_s)

import pandas as pd

from src.admet.calibrate import calibrate_admet_scores
from src.admet.featurize import featurize_compounds
from src.admet.predict import build_model_registry, predict_admet
from src.chem.conformers import generate_best_conformers
from src.chem.descriptors import compute_descriptors
from src.chem.lipinski import annotate_lipinski, maybe_filter_lipinski
from src.chem.pdbqt import PDBQTConversionError, convert_sdf_to_pdbqt
from src.chem.standardize import parse_and_standardize
from src.config import PipelineConfig, load_pipeline_config
from src.docking.parse_results import normalize_docking_rows
from src.docking.receptor import load_receptor_spec
from src.docking.vina_runner import dock_ligands
from src.io.readers import load_compound_table
from src.io.writers import ensure_dirs, write_failure_log, write_stage_dataframe, write_stage_manifest
from src.ranking.aggregate import aggregate_and_rank
from src.ranking.filters import shortlist
from src.utils.logging import get_logger

STAGES = [
    "ingest",
    "standardize",
    "descriptors",
    "lipinski",
    "admet_featurize",
    "admet_predict",
    "conformers",
    "ligand_pdbqt",
    "receptor_prep",
    "docking",
    "parse_docking",
    "ranking",
]

STAGE_ALIASES = {
    "qc": "descriptors",
    "admet": "admet_predict",
    "docking": "parse_docking",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Epigenetic drug screening pipeline",
        epilog=(
            "Examples (from epigenetic_screen/):\n"
            "  %(prog)s run --config configs/pipeline.yaml\n"
            "  %(prog)s run --config configs/pipeline.yaml --stage lipinski\n"
            "  python -m src.main run --config configs/pipeline.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    run = sub.add_parser("run", help="Run pipeline")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--stage", default="all", choices=["all"] + STAGES + sorted(STAGE_ALIASES.keys()))
    run.add_argument("--force", action="store_true", help="Recompute stage outputs even if existing")
    return parser.parse_args()


def _stage_csv_paths(cfg: PipelineConfig) -> dict[str, Path]:
    o = cfg.paths.output_root
    return {
        "ingest": o / "01_qc" / "01_ingest.csv",
        "standardize": o / "01_qc" / "02_standardized.csv",
        "descriptors": o / "01_qc" / "03_descriptors.csv",
        "lipinski": o / "02_lipinski" / "04_lipinski.csv",
        "admet_featurize": o / "03_admet" / "05_admet_featurized.csv",
        "admet_predict": o / "03_admet" / "06_admet_predictions.csv",
        "conformers": o / "04_ligands_3d" / "07_conformers.csv",
        "ligand_pdbqt": o / "05_docking" / "08_ligands_pdbqt.csv",
        "receptor_prep": o / "05_docking" / "09_receptors.csv",
        "docking": o / "05_docking" / "10_docking_raw.csv",
        "parse_docking": o / "05_docking" / "11_docking_parsed.csv",
        "ranking": o / "06_ranked" / "12_ranked.csv",
    }


def _manifest_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".manifest.json")


def _failure_path(cfg: PipelineConfig, stage: str) -> Path:
    return cfg.paths.output_root / "logs" / f"{stage}_failures.csv"


def _should_run(stage: str, requested: str) -> bool:
    if requested == "all":
        return True
    requested = STAGE_ALIASES.get(requested, requested)
    return STAGES.index(stage) <= STAGES.index(requested)


def _maybe_load_stage(path: Path, force: bool) -> pd.DataFrame | None:
    if path.exists() and not force:
        return pd.read_csv(path)
    return None


def _write_stage(cfg: PipelineConfig, stage: str, df: pd.DataFrame, failures: list[dict[str, Any]], source: list[str]) -> None:
    csv_path = _stage_csv_paths(cfg)[stage]
    write_stage_dataframe(df, csv_path)
    write_failure_log(failures, _failure_path(cfg, stage))
    write_stage_manifest(_manifest_path(csv_path), stage=stage, row_count=len(df), input_sources=source)


def run(cfg: PipelineConfig, requested_stage: str, force: bool) -> None:
    stage_paths = _stage_csv_paths(cfg)
    ensure_dirs([p.parent for p in stage_paths.values()] + [cfg.paths.output_root / "logs"])
    logger = get_logger(cfg.paths.output_root / "logs" / "pipeline.log")
    logger.info("Starting run stage=%s force=%s", requested_stage, force)

    # 1 Input ingest
    ingest_df = _maybe_load_stage(stage_paths["ingest"], force)
    if ingest_df is None and _should_run("ingest", requested_stage):
        ingest_df = load_compound_table(cfg.paths.input_path)
        _write_stage(cfg, "ingest", ingest_df, [], [str(cfg.paths.input_path)])
    elif ingest_df is None:
        ingest_df = pd.read_csv(stage_paths["ingest"])

    # 2 Parse/standardize
    standard_df = _maybe_load_stage(stage_paths["standardize"], force)
    if standard_df is None and _should_run("standardize", requested_stage):
        standard_df, failures = parse_and_standardize(ingest_df)
        _write_stage(cfg, "standardize", standard_df, failures, [str(stage_paths["ingest"])])
    elif standard_df is None:
        standard_df = pd.read_csv(stage_paths["standardize"])

    # 3 Descriptors
    desc_df = _maybe_load_stage(stage_paths["descriptors"], force)
    if desc_df is None and _should_run("descriptors", requested_stage):
        desc_df, failures = compute_descriptors(standard_df)
        _write_stage(cfg, "descriptors", desc_df, failures, [str(stage_paths["standardize"])])
    elif desc_df is None:
        desc_df = pd.read_csv(stage_paths["descriptors"])

    # 4 Lipinski
    lip_df = _maybe_load_stage(stage_paths["lipinski"], force)
    if lip_df is None and _should_run("lipinski", requested_stage):
        lip_df = annotate_lipinski(desc_df, soft_violation_max=cfg.lipinski.soft_violation_max)
        lip_df = maybe_filter_lipinski(lip_df, drop_failures=cfg.lipinski.drop_failures)
        _write_stage(cfg, "lipinski", lip_df, [], [str(stage_paths["descriptors"])])
    elif lip_df is None:
        lip_df = pd.read_csv(stage_paths["lipinski"])

    # 5 ADMET featurization
    feat_df = _maybe_load_stage(stage_paths["admet_featurize"], force)
    if feat_df is None and _should_run("admet_featurize", requested_stage):
        feat = featurize_compounds(lip_df, cfg.admet.featurizer)
        feat_df = feat.dataframe
        _write_stage(cfg, "admet_featurize", feat_df, feat.failures, [str(stage_paths["lipinski"])])
    elif feat_df is None:
        feat_df = pd.read_csv(stage_paths["admet_featurize"])

    # 6 ADMET prediction
    admet_df = _maybe_load_stage(stage_paths["admet_predict"], force)
    if admet_df is None and _should_run("admet_predict", requested_stage):
        registry = build_model_registry(cfg.admet.endpoints, use_mock_models=cfg.admet.use_mock_models)
        admet_df = predict_admet(feat_df, registry, cfg.admet.endpoints)
        admet_df = calibrate_admet_scores(admet_df)
        _write_stage(cfg, "admet_predict", admet_df, [], [str(stage_paths["admet_featurize"])])
    elif admet_df is None:
        admet_df = pd.read_csv(stage_paths["admet_predict"])

    # 7 Conformers
    conf_df = _maybe_load_stage(stage_paths["conformers"], force)
    if conf_df is None and _should_run("conformers", requested_stage):
        conf_df, failures = generate_best_conformers(
            admet_df,
            cfg.paths.output_root / "04_ligands_3d" / "sdf",
            seed=cfg.random_seed,
        )
        _write_stage(cfg, "conformers", conf_df, failures, [str(stage_paths["admet_predict"])])
    elif conf_df is None:
        conf_df = pd.read_csv(stage_paths["conformers"])

    # 8 Ligand docking prep (PDBQT)
    ligprep_df = _maybe_load_stage(stage_paths["ligand_pdbqt"], force)
    if ligprep_df is None and _should_run("ligand_pdbqt", requested_stage):
        rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        pdbqt_dir = cfg.paths.output_root / "05_docking" / "ligands_pdbqt"
        for _, row in conf_df.iterrows():
            cid = str(row["compound_id"])
            sdf_path = Path(str(row["ligand_sdf_path"]))
            rec = row.to_dict()
            try:
                pdbqt_path = convert_sdf_to_pdbqt(sdf_path, pdbqt_dir)
                rec["ligand_pdbqt_path"] = str(pdbqt_path)
            except PDBQTConversionError as e:
                rec["ligand_pdbqt_path"] = str((pdbqt_dir / f"{cid}.pdbqt"))
                failures.append({"compound_id": cid, "stage": "ligand_pdbqt", "reason": str(e)})
            rows.append(rec)
        ligprep_df = pd.DataFrame(rows)
        _write_stage(cfg, "ligand_pdbqt", ligprep_df, failures, [str(stage_paths["conformers"])])
    elif ligprep_df is None:
        ligprep_df = pd.read_csv(stage_paths["ligand_pdbqt"])

    # 9 Receptor prep
    receptor_df = _maybe_load_stage(stage_paths["receptor_prep"], force)
    receptor_specs = []
    if receptor_df is None and _should_run("receptor_prep", requested_stage):
        failures = []
        rows = []
        for target_key in cfg.targets:
            target_cfg_path = cfg.paths.target_configs_dir / f"{target_key}.yaml"
            try:
                spec = load_receptor_spec(target_cfg_path, receptor_root=cfg.paths.receptor_dir)
                receptor_specs.append(spec)
                rows.append(
                    {
                        "target_name": spec.target_name,
                        "receptor_path": str(spec.receptor_path),
                        "box_center": list(spec.box_center),
                        "box_size": list(spec.box_size),
                        "exhaustiveness": spec.exhaustiveness,
                        "n_poses": spec.n_poses,
                    }
                )
            except Exception as e:
                failures.append({"compound_id": "", "stage": "receptor_prep", "reason": f"{target_key}: {e}"})
        receptor_df = pd.DataFrame(rows)
        _write_stage(cfg, "receptor_prep", receptor_df, failures, [str(cfg.paths.target_configs_dir)])
    elif receptor_df is None:
        receptor_df = pd.read_csv(stage_paths["receptor_prep"])
    if not receptor_specs:
        for target_key in cfg.targets:
            target_cfg_path = cfg.paths.target_configs_dir / f"{target_key}.yaml"
            if target_cfg_path.exists():
                receptor_specs.append(load_receptor_spec(target_cfg_path, receptor_root=cfg.paths.receptor_dir))

    # 10 Docking
    docking_raw_df = _maybe_load_stage(stage_paths["docking"], force)
    if docking_raw_df is None and _should_run("docking", requested_stage):
        all_rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        ligand_paths = {
            str(row["compound_id"]): Path(str(row["ligand_pdbqt_path"])) for _, row in ligprep_df.iterrows()
        }
        for spec in receptor_specs:
            rows, errs = dock_ligands(
                spec,
                ligand_paths,
                cfg.paths.output_root / "05_docking" / spec.target_name,
                use_mock_if_unavailable=cfg.docking.use_mock_if_unavailable,
            )
            all_rows.extend(rows)
            failures.extend(errs)
        docking_raw_df = pd.DataFrame(all_rows)
        _write_stage(cfg, "docking", docking_raw_df, failures, [str(stage_paths["ligand_pdbqt"]), str(stage_paths["receptor_prep"])])
    elif docking_raw_df is None:
        docking_raw_df = pd.read_csv(stage_paths["docking"])

    # 11 Parse docking results
    docking_df = _maybe_load_stage(stage_paths["parse_docking"], force)
    if docking_df is None and _should_run("parse_docking", requested_stage):
        docking_df = normalize_docking_rows(docking_raw_df.to_dict(orient="records"))
        _write_stage(cfg, "parse_docking", docking_df, [], [str(stage_paths["docking"])])
    elif docking_df is None:
        docking_df = pd.read_csv(stage_paths["parse_docking"])

    # 12 Ranking
    if _should_run("ranking", requested_stage):
        ranked = aggregate_and_rank(lip_df, admet_df, docking_df, cfg.ranking)
        _write_stage(cfg, "ranking", ranked, [], [str(stage_paths["lipinski"]), str(stage_paths["admet_predict"]), str(stage_paths["parse_docking"])])
        shortlist_df = shortlist(ranked, cfg.ranking.shortlist_top_n)
        write_stage_dataframe(shortlist_df, cfg.paths.output_root / "06_ranked" / "shortlist.csv")

    logger.info("Pipeline completed")


def main() -> None:
    if len(sys.argv) == 1:
        print(
            "No command given. Use:\n"
            "  python src/main.py run --config configs/pipeline.yaml\n"
            "Or: epigenetic-screen run --config configs/pipeline.yaml\n"
            "Try: python src/main.py --help",
            file=sys.stderr,
        )
        raise SystemExit(2)
    args = parse_args()
    if args.command == "run":
        cfg = load_pipeline_config(args.config)
        run(cfg, requested_stage=args.stage, force=bool(args.force))


if __name__ == "__main__":
    main()

