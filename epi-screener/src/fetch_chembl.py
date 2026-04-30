from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from chembl_webresource_client.new_client import new_client


OUTPUT_COLUMNS = [
    "Molecule_ChEMBL_ID",
    "Target_ChEMBL_ID",
    "IC50_Value",
    "IC50_Unit",
    "Canonical_SMILES",
]

UNIT_TO_MICROMOLAR = {
    "um": 1.0,
    "µm": 1.0,
    "nm": 0.001,
    "mm": 1000.0,
    "pm": 0.000001,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch ChEMBL small-molecule activities for human targets and export "
            "IC50 < 10 micromolar records to CSV."
        )
    )
    parser.add_argument(
        "target_ids",
        nargs="+",
        help=(
            "One or more human ChEMBL target IDs. "
            "Example: CHEMBL1991 CHEMBL301"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("01_data_ingestion/results/chembl_ic50_lt10um.csv"),
        help="Output CSV path (default: 01_data_ingestion/results/chembl_ic50_lt10um.csv).",
    )
    return parser.parse_args()


def normalize_target_ids(raw_ids: Iterable[str]) -> list[str]:
    ids: list[str] = []
    for raw_id in raw_ids:
        for token in raw_id.split(","):
            cleaned = token.strip().upper()
            if cleaned:
                ids.append(cleaned)
    return sorted(set(ids))


def to_micromolar(value: float, unit: str) -> float | None:
    factor = UNIT_TO_MICROMOLAR.get((unit or "").strip().lower())
    if factor is None:
        return None
    return value * factor


def get_small_molecule_smiles(molecule_ids: Iterable[str]) -> dict[str, str]:
    molecule = new_client.molecule
    smiles_map: dict[str, str] = {}
    for chembl_id in sorted(set(molecule_ids)):
        if not chembl_id:
            continue
        details = molecule.get(chembl_id)
        if not details:
            continue
        if (details.get("molecule_type") or "").lower() != "small molecule":
            continue
        structures = details.get("molecule_structures") or {}
        canonical_smiles = structures.get("canonical_smiles")
        if canonical_smiles:
            smiles_map[chembl_id] = canonical_smiles
    return smiles_map


def fetch_filtered_rows(target_ids: list[str]) -> list[dict[str, str | float]]:
    activity = new_client.activity
    rows: list[dict[str, str | float]] = []

    for target_id in target_ids:
        activities = activity.filter(
            target_chembl_id=target_id,
            standard_type="IC50",
            standard_relation__in=["<", "<=", "="],
            target_organism="Homo sapiens",
        ).only(
            "molecule_chembl_id",
            "target_chembl_id",
            "standard_value",
            "standard_units",
        )

        candidate_records = []
        candidate_ids: set[str] = set()

        for act in activities:
            value_raw = act.get("standard_value")
            unit_raw = act.get("standard_units")
            mol_id = act.get("molecule_chembl_id")

            if value_raw in (None, "", "None") or not mol_id or not unit_raw:
                continue

            try:
                value = float(value_raw)
            except (TypeError, ValueError):
                continue

            value_um = to_micromolar(value, str(unit_raw))
            if value_um is None or value_um >= 10.0:
                continue

            candidate_records.append(
                {
                    "Molecule_ChEMBL_ID": mol_id,
                    "Target_ChEMBL_ID": act.get("target_chembl_id") or target_id,
                    "IC50_Value": value,
                    "IC50_Unit": unit_raw,
                }
            )
            candidate_ids.add(mol_id)

        smiles_map = get_small_molecule_smiles(candidate_ids)
        for record in candidate_records:
            smiles = smiles_map.get(record["Molecule_ChEMBL_ID"])  # type: ignore[index]
            if not smiles:
                continue
            rows.append({**record, "Canonical_SMILES": smiles})

    return rows


def write_csv(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    target_ids = normalize_target_ids(args.target_ids)
    if not target_ids:
        raise ValueError("No valid target IDs provided.")

    rows = fetch_filtered_rows(target_ids)
    write_csv(rows, args.output)

    print(f"Targets queried: {', '.join(target_ids)}")
    print("Filter applied: IC50 < 10 micromolar, human targets, small molecules only")
    print(f"Rows written: {len(rows)}")
    print(f"Output CSV: {args.output.resolve()}")


if __name__ == "__main__":
    main()
