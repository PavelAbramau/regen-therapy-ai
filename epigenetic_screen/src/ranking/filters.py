from __future__ import annotations

import pandas as pd


def shortlist(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    return df.nsmallest(top_n, "rank_global").copy()

