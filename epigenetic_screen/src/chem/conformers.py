from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def generate_best_conformers(
    df: pd.DataFrame,
    ligands_dir: Path,
    *,
    seed: int = 42,
    minimize_mmff: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Generate 3D conformers with ETKDGv3 and keep one optimized conformer per ligand.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RDKit is required for stage 'conformers'. Install rdkit.") from exc

    ligands_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        cid = str(row["compound_id"])
        smiles = str(row["canonical_smiles"])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failures.append({"compound_id": cid, "stage": "conformers", "reason": "Invalid canonical_smiles"})
            continue

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=5, params=params)
        if not conf_ids:
            failures.append({"compound_id": cid, "stage": "conformers", "reason": "No conformer generated"})
            continue

        best_conf = int(conf_ids[0])
        best_energy = None
        if minimize_mmff:
            mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
            for conf_id in conf_ids:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=int(conf_id))
                if ff is None:
                    continue
                ff.Minimize(maxIts=200)
                e = float(ff.CalcEnergy())
                if best_energy is None or e < best_energy:
                    best_energy = e
                    best_conf = int(conf_id)

        out_sdf = ligands_dir / f"{cid}.sdf"
        writer = Chem.SDWriter(str(out_sdf))
        writer.write(mol, confId=best_conf)
        writer.close()

        rec = row.to_dict()
        rec["ligand_sdf_path"] = str(out_sdf)
        rec["conformer_energy"] = best_energy if best_energy is not None else float("nan")
        rows.append(rec)

    return pd.DataFrame(rows), failures

