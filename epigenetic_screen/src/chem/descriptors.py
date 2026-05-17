from __future__ import annotations

from typing import Any

import pandas as pd


def compute_descriptors(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Compute developability descriptors from canonical SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RDKit is required for stage 'descriptors'. Install rdkit.") from exc

    out_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        cid = str(row["compound_id"])
        smiles = str(row["canonical_smiles"])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failures.append({"compound_id": cid, "stage": "descriptors", "reason": "Invalid canonical_smiles"})
            continue

        d = row.to_dict()
        d["mw"] = float(Descriptors.MolWt(mol))
        d["logp"] = float(Descriptors.MolLogP(mol))
        d["hbd"] = int(Lipinski.NumHDonors(mol))
        d["hba"] = int(Lipinski.NumHAcceptors(mol))
        d["rot_bonds"] = int(Lipinski.NumRotatableBonds(mol))
        d["tpsa"] = float(rdMolDescriptors.CalcTPSA(mol))
        d["ring_count"] = int(rdMolDescriptors.CalcNumRings(mol))
        d["heavy_atom_count"] = int(rdMolDescriptors.CalcNumHeavyAtoms(mol))
        d["fraction_csp3"] = float(rdMolDescriptors.CalcFractionCSP3(mol))
        out_rows.append(d)

    return pd.DataFrame(out_rows), failures

