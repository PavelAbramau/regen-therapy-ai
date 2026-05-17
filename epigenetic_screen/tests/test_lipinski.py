import pandas as pd

from src.chem.lipinski import annotate_lipinski


def test_lipinski_annotation_flags() -> None:
    df = pd.DataFrame(
        [
            {"compound_id": "ok", "mw": 250, "logp": 2.1, "hbd": 1, "hba": 3},
            {"compound_id": "bad", "mw": 800, "logp": 9.0, "hbd": 7, "hba": 12},
        ]
    )
    out = annotate_lipinski(df, soft_violation_max=1)
    assert out.loc[out["compound_id"] == "ok", "lipinski_strict_pass"].item() is True
    assert out.loc[out["compound_id"] == "bad", "lipinski_fail"].item() is True

