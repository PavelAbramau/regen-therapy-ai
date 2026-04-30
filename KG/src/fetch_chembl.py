from __future__ import annotations

from pathlib import Path

import pandas as pd
from chembl_webresource_client.new_client import new_client


TARGETS = {
    "CHEMBL1978": "DNMT1",
    "CHEMBL3004": "HDAC6",
}
MAX_RESULTS_PER_TARGET = 50
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "chembl_seed.csv"


def fetch_smiles_map(molecule_ids: set[str]) -> dict[str, str]:
    molecule_client = new_client.molecule
    smiles_map: dict[str, str] = {}

    for molecule_id in molecule_ids:
        try:
            molecule_data = molecule_client.get(molecule_id)
        except Exception as exc:  # pragma: no cover - API/network dependent
            print(f"Failed to fetch molecule {molecule_id}: {exc}")
            continue

        if not molecule_data:
            continue

        structures = molecule_data.get("molecule_structures") or {}
        smiles = structures.get("canonical_smiles")
        if smiles:
            smiles_map[molecule_id] = smiles

    return smiles_map


def fetch_target_rows(target_id: str) -> list[dict[str, str | float]]:
    activity_client = new_client.activity
    rows: list[dict[str, str | float]] = []
    candidate_molecules: set[str] = set()

    try:
        activities = activity_client.filter(
            target_chembl_id=target_id,
            target_organism="Homo sapiens",
            standard_type="IC50",
            standard_units="nM",
            standard_relation__in=["<", "<=", "="],
            standard_value__lt=10000,
        ).only("molecule_chembl_id", "target_chembl_id", "standard_value")
    except Exception as exc:  # pragma: no cover - API/network dependent
        print(f"Failed to query activities for {target_id}: {exc}")
        return rows

    for activity in activities:
        if len(rows) >= MAX_RESULTS_PER_TARGET:
            break

        molecule_id = activity.get("molecule_chembl_id")
        ic50_raw = activity.get("standard_value")
        if not molecule_id or ic50_raw in (None, ""):
            continue

        try:
            ic50_value = float(ic50_raw)
        except (TypeError, ValueError):
            continue

        rows.append(
            {
                "Molecule_ChEMBL_ID": molecule_id,
                "Target_ChEMBL_ID": activity.get("target_chembl_id") or target_id,
                "IC50_Value": ic50_value,
            }
        )
        candidate_molecules.add(molecule_id)

    smiles_map = fetch_smiles_map(candidate_molecules)
    completed_rows = []
    for row in rows:
        smiles = smiles_map.get(str(row["Molecule_ChEMBL_ID"]))
        if not smiles:
            continue
        completed_rows.append({**row, "Canonical_SMILES": smiles})

    return completed_rows


def main() -> None:
    all_rows: list[dict[str, str | float]] = []

    for target_id, target_name in TARGETS.items():
        print(f"Fetching {target_name} inhibitors ({target_id})...")
        rows = fetch_target_rows(target_id)
        print(f"Collected {len(rows)} rows for {target_id}")
        all_rows.extend(rows)

    output_df = pd.DataFrame(
        all_rows,
        columns=[
            "Molecule_ChEMBL_ID",
            "Canonical_SMILES",
            "Target_ChEMBL_ID",
            "IC50_Value",
        ],
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(output_df)} total rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
