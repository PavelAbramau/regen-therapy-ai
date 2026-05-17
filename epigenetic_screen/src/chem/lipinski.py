from __future__ import annotations

import pandas as pd


def annotate_lipinski(df: pd.DataFrame, soft_violation_max: int = 1) -> pd.DataFrame:
    """Annotate Lipinski strict/soft/fail classes without hard dropping by default."""
    out = df.copy()
    out["lipinski_violations"] = (
        (out["mw"] > 500).astype(int)
        + (out["logp"] > 5).astype(int)
        + (out["hbd"] > 5).astype(int)
        + (out["hba"] > 10).astype(int)
    )

    out["lipinski_strict_pass"] = out["lipinski_violations"] == 0
    out["lipinski_soft_pass"] = out["lipinski_violations"] <= int(soft_violation_max)
    out["lipinski_fail"] = ~out["lipinski_soft_pass"]
    return out


def maybe_filter_lipinski(df: pd.DataFrame, drop_failures: bool) -> pd.DataFrame:
    if not drop_failures:
        return df
    return df[~df["lipinski_fail"]].copy()

