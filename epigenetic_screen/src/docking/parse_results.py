from __future__ import annotations

import pandas as pd


def normalize_docking_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "compound_id",
                "target_name",
                "vina_predock_score",
                "vina_optimized_score",
                "vina_best_score",
                "pose_path",
                "docking_engine",
            ]
        )
    return pd.DataFrame(rows)

