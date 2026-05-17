import pandas as pd
import pytest

pytest.importorskip("rdkit")

from src.chem.descriptors import compute_descriptors
from src.chem.standardize import parse_and_standardize


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_descriptor_generation() -> None:
    base = pd.DataFrame(
        [
            {"compound_id": "A", "smiles": "CCO"},
            {"compound_id": "B", "smiles": "c1ccccc1"},
        ]
    )
    std_df, std_fail = parse_and_standardize(base)
    assert not std_fail
    desc_df, failures = compute_descriptors(std_df)
    assert not failures
    assert {"mw", "logp", "hbd", "hba", "tpsa", "rot_bonds"}.issubset(desc_df.columns)

