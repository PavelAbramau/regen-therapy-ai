from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class FeaturizationResult:
    dataframe: pd.DataFrame
    features: dict[str, Any]
    failures: list[dict[str, Any]]


def featurize_compounds(df: pd.DataFrame, featurizer_name: str) -> FeaturizationResult:
    """
    Pluggable DeepChem featurization wrapper.
    Supports MolGraphConvFeaturizer and CircularFingerprint.
    """
    featurizer_key = featurizer_name.strip().lower()
    smiles = df["canonical_smiles"].astype(str).tolist()
    failures: list[dict[str, Any]] = []
    features: dict[str, Any] = {}
    status = "mock"

    try:
        import deepchem as dc

        if featurizer_key in {"molgraphconv", "molgraphconvfeaturizer"}:
            featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        elif featurizer_key in {"circular", "circularfingerprint", "ecfp"}:
            featurizer = dc.feat.CircularFingerprint(size=2048, radius=2)
        else:
            raise ValueError(f"Unsupported featurizer {featurizer_name}")

        feats = featurizer.featurize(smiles)
        features["X"] = feats
        status = "deepchem"
    except Exception:
        # Keep pipeline runnable without DeepChem runtime.
        features["X"] = [[len(s)] for s in smiles]
        status = "mock"

    out = df.copy()
    out["admet_featurizer"] = featurizer_name
    out["admet_featurization_status"] = status
    return FeaturizationResult(dataframe=out, features=features, failures=failures)

