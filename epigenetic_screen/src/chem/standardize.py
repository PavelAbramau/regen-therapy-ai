from __future__ import annotations

from typing import Any

import pandas as pd


def parse_and_standardize(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Parse SMILES, sanitize molecules, and create canonical isomeric SMILES."""
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RDKit is required for stage 'standardize'. Install rdkit.") from exc

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        cid = str(row["compound_id"])
        input_smiles = str(row["smiles"])
        mol = Chem.MolFromSmiles(input_smiles, sanitize=False)
        if mol is None:
            failures.append({"compound_id": cid, "stage": "standardize", "reason": "MolFromSmiles returned None"})
            continue
        try:
            Chem.SanitizeMol(mol)
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception as e:  # pragma: no cover
            failures.append({"compound_id": cid, "stage": "standardize", "reason": f"sanitize failed: {e}"})
            continue

        out = row.to_dict()
        out["input_smiles"] = input_smiles
        out["canonical_smiles"] = canonical
        rows.append(out)

    return pd.DataFrame(rows), failures

