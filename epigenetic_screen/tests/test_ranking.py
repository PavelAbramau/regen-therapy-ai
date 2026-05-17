import pandas as pd

from src.config import RankingConfig
from src.ranking.aggregate import aggregate_and_rank


def test_aggregate_ranking() -> None:
    chem = pd.DataFrame(
        [
            {"compound_id": "A", "mw": 300, "logp": 2.0, "tpsa": 70},
            {"compound_id": "B", "mw": 450, "logp": 3.5, "tpsa": 95},
        ]
    )
    admet = pd.DataFrame(
        [
            {"compound_id": "A", "admet_solubility": 0.8},
            {"compound_id": "B", "admet_solubility": 0.2},
        ]
    )
    docking = pd.DataFrame(
        [
            {"compound_id": "A", "target_name": "HDAC2", "vina_best_score": -9.0},
            {"compound_id": "B", "target_name": "HDAC2", "vina_best_score": -6.0},
        ]
    )
    cfg = RankingConfig(
        descriptor_terms={"mw": -0.001},
        admet_terms={"solubility": 1.0},
        docking_weight=1.0,
        shortlist_top_n=10,
    )
    out = aggregate_and_rank(chem, admet, docking, cfg)
    assert out.iloc[0]["compound_id"] == "A"
    assert "rank_global" in out.columns

