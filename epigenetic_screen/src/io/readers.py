from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_compound_table(path: str | Path) -> pd.DataFrame:
    """
    Load compounds from SMI/CSV/TSV and return a normalized dataframe.

    Required output columns:
    - compound_id
    - smiles
    """
    input_path = Path(path)
    ext = input_path.suffix.lower()

    if ext in {".smi", ".smiles", ".txt"}:
        df = pd.read_csv(
            input_path,
            sep=r"\s+",
            header=None,
            names=["smiles", "compound_id"],
            engine="python",
        )
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext in {".tsv", ".tab"}:
        df = pd.read_csv(input_path, sep="\t")
    else:
        raise ValueError(f"Unsupported input format: {ext}")

    if "smiles" not in df.columns:
        for candidate in ["SMILES", "canonical_smiles", "Canonical_SMILES"]:
            if candidate in df.columns:
                df["smiles"] = df[candidate]
                break
    if "smiles" not in df.columns:
        raise ValueError("Input must contain a SMILES-like column.")

    if "compound_id" not in df.columns:
        for candidate in ["Compound_ID", "id", "ID", "name", "Name"]:
            if candidate in df.columns:
                df["compound_id"] = df[candidate]
                break
    if "compound_id" not in df.columns:
        df["compound_id"] = [f"CMPD_{i+1:05d}" for i in range(len(df))]

    df["compound_id"] = df["compound_id"].astype(str).str.strip()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[df["compound_id"] != ""].copy()
    df = df[df["smiles"] != ""].copy()
    df = df.drop_duplicates(subset=["compound_id"], keep="first")
    return df.reset_index(drop=True)

