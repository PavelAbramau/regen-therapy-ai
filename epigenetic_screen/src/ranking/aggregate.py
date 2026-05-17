from __future__ import annotations

import pandas as pd

from src.config import RankingConfig


def _safe_col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def aggregate_and_rank(
    chemistry_df: pd.DataFrame,
    admet_df: pd.DataFrame,
    docking_df: pd.DataFrame,
    cfg: RankingConfig,
) -> pd.DataFrame:
    merged = chemistry_df.merge(admet_df, on="compound_id", how="left", suffixes=("", "_admet"))
    merged = merged.merge(docking_df, on="compound_id", how="left")

    score = pd.Series([0.0] * len(merged), index=merged.index, dtype=float)

    for col, w in cfg.descriptor_terms.items():
        score += _safe_col(merged, col) * float(w)
    for endpoint, w in cfg.admet_terms.items():
        score += _safe_col(merged, f"admet_{endpoint}") * float(w)

    # Lower (more negative) vina_best_score is generally better => subtract weighted score.
    score -= _safe_col(merged, "vina_best_score") * float(cfg.docking_weight)

    merged["composite_score"] = score
    merged["rank_global"] = merged["composite_score"].rank(ascending=False, method="dense").astype(int)
    if "target_name" in merged.columns:
        merged["rank_per_target"] = (
            merged.groupby("target_name")["composite_score"].rank(ascending=False, method="dense").astype(int)
        )
    else:
        merged["target_name"] = "unknown"
        merged["rank_per_target"] = merged["rank_global"]

    return merged.sort_values(["rank_global", "compound_id"]).reset_index(drop=True)

