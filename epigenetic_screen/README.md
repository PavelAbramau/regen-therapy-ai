# Epigenetic Screening Pipeline

Production-oriented, restartable Python pipeline for small-molecule epigenetic screening across chemistry QC, ADMET, docking, and ranking.

## Purpose

This project screens a compound library (for example ~1,300 SMILES) through:

1. Chemistry ingest, parsing, and standardization (RDKit)
2. Descriptor and Lipinski/developability annotation (RDKit)
3. ADMET featurization and prediction (DeepChem-ready, mock-safe fallback)
4. 3D ligand generation and docking preparation (RDKit + external PDBQT adapter)
5. Multi-target docking with AutoDock Vina (real API when available, mock-safe fallback)
6. Composite ranking and shortlist export

The design preserves compound provenance (`compound_id`) and writes stage outputs to disk for auditability and resumability.

## Architecture Overview

`src/main.py` orchestrates 12 explicit stages:

1. input ingest
2. parse + standardize
3. descriptor calculation
4. Lipinski annotation/filtering
5. ADMET featurization
6. ADMET prediction
7. 3D conformers
8. ligand PDBQT prep
9. receptor prep
10. docking
11. docking parse/normalize
12. composite ranking + shortlist

### Modules

- `src/io`: readers/writers and manifest/failure-log output
- `src/chem`: standardization, descriptors, Lipinski, conformers, PDBQT adapter
- `src/admet`: featurization + endpoint model registry + calibration hook
- `src/docking`: receptor config loader, Vina runner, result parser
- `src/ranking`: merge + scoring + shortlist helpers
- `src/config.py`: YAML config loading into dataclasses

## Data Flow

Input: `data/raw/compounds.smi` (or CSV/TSV via config)  
Output root: `output/`

Each stage writes:

- stage CSV output
- stage manifest JSON (`*.manifest.json`)
- stage failure CSV (`output/logs/<stage>_failures.csv`)

Final outputs:

- `output/06_ranked/12_ranked.csv`
- `output/06_ranked/shortlist.csv`

Expected key columns include:

- `compound_id`, `input_smiles`, `canonical_smiles`
- `mw`, `logp`, `hbd`, `hba`, `tpsa`, `rot_bonds`
- `lipinski_violations`, `lipinski_strict_pass`, `lipinski_soft_pass`
- `admet_*` endpoint columns
- `vina_predock_score`, `vina_optimized_score`, `vina_best_score`
- `target_name`, `composite_score`, `rank_global`, `rank_per_target`

## Installation

From `epigenetic_screen/` (dedicated environment in **this folder**):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -U pip setuptools wheel
pip install -e ".[dev]"
```

Python **3.11+** is recommended for stacks like RDKit/DeepChem; the project metadata allows **3.9+** so a default `python3 -m venv` still installs on older system Pythons.

### Dependency Notes

- **RDKit**: required for chemistry stages (`standardize`, `descriptors`, `conformers`)
- **DeepChem**: optional in v1; featurization auto-falls back to deterministic mock vectors
- **AutoDock Vina Python API** (`vina`): optional in v1; docking auto-falls back to deterministic mock docking if enabled
- **PDBQT conversion**: environment-dependent. `src/chem/pdbqt.py` currently expects `obabel` if real conversion is needed.

## CLI Usage

Always run commands from inside the **`epigenetic_screen/`** directory (so `configs/` resolves).

Run full pipeline (recommended):

```bash
python -m src.main run --config configs/pipeline.yaml
```

Equivalent:

```bash
python -m src run --config configs/pipeline.yaml
```

Direct script (also supported after patching `sys.path`):

```bash
python src/main.py run --config configs/pipeline.yaml
```

After editable install (`pip install -e .`):

```bash
epigenetic-screen run --config configs/pipeline.yaml
```

Run through a specific stage:

```bash
python -m src.main run --config configs/pipeline.yaml --stage admet_predict
python -m src.main run --config configs/pipeline.yaml --stage docking
```

Force recompute:

```bash
python -m src.main run --config configs/pipeline.yaml --force
```

## Restartable Stage Behavior

- Existing stage outputs are reused automatically.
- If a stage output CSV exists and `--force` is not set, it is loaded instead of recomputed.
- Failures are persisted per stage, so partial runs are auditable.

## Input Formats

Supported input formats:

- `.smi`/`.smiles`/`.txt`: whitespace-delimited `<smiles> <compound_id>`
- `.csv` and `.tsv`: must contain a SMILES column (`smiles`, `SMILES`, etc.)

`compound_id` is always normalized and preserved for downstream joins.

## Extending Targets and ADMET Models

### Add new targets

1. Add a YAML in `configs/targets/`
2. Add target key to `configs/pipeline.yaml -> targets`
3. Place receptor PDBQT in `data/receptors/` (or update path in target YAML)

### Add real ADMET models

Replace `MockAdmetModel` registration in `src/admet/predict.py` with real model wrappers per endpoint.

## Scientific and Operational Notes

- No scientific efficacy claims are made by this codebase.
- Mock ADMET and mock docking are clearly marked in output (`admet_featurization_status`, `docking_engine`).
- Protonation state, tautomer selection, and docking-format assumptions are intentionally isolated in adapter layers (`standardize.py`, `pdbqt.py`) for future hardening.

## Tests

Run:

```bash
pytest -q
```

Current tests cover:

- config loading
- SMILES ingest parsing
- descriptor computation
- Lipinski annotation
- ranking logic

