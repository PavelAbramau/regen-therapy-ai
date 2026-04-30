from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski


def count_lipinski_violations(mol: Chem.Mol) -> tuple[int, dict]:
    """Return number of Lipinski rule violations and raw property values."""
    mw = Descriptors.MolWt(mol)
    h_donors = Lipinski.NumHDonors(mol)
    h_acceptors = Lipinski.NumHAcceptors(mol)
    logp = Descriptors.MolLogP(mol)

    violations = 0
    violations += int(mw >= 500.0)
    violations += int(h_donors > 5)
    violations += int(h_acceptors > 10)
    violations += int(logp >= 5.0)

    values = {
        "Molecular_Weight": mw,
        "Num_H_Donors": h_donors,
        "Num_H_Acceptors": h_acceptors,
        "LogP": logp,
        "Lipinski_Violations": violations,
    }
    return violations, values


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "input_library.csv"
    output_path = project_root / "data" / "passed_lipinski.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    required_columns = {"SMILES", "Compound_ID"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    records = []
    total = len(df)
    invalid_smiles = 0

    for _, row in df.iterrows():
        smiles = row["SMILES"]
        compound_id = row["Compound_ID"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles += 1
            continue

        violations, values = count_lipinski_violations(mol)
        if violations <= 1:
            records.append(
                {
                    "Compound_ID": compound_id,
                    "SMILES": smiles,
                    **values,
                }
            )

    passed_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    passed_df.to_csv(output_path, index=False)

    passed_count = len(passed_df)
    failed_count = total - passed_count
    print(f"Total compounds: {total}")
    print(f"Invalid SMILES skipped: {invalid_smiles}")
    print(f"Compounds passed Lipinski filter: {passed_count}")
    print(f"Compounds filtered out: {failed_count}")
    print(f"Saved passing compounds to: {output_path}")


if __name__ == "__main__":
    main()
