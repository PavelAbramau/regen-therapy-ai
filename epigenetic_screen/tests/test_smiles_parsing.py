from pathlib import Path

from src.io.readers import load_compound_table


def test_load_smi_input() -> None:
    df = load_compound_table(Path("data/raw/compounds.smi"))
    assert "compound_id" in df.columns
    assert "smiles" in df.columns
    assert len(df) >= 1

