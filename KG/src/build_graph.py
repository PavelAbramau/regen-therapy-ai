from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import pandas as pd


TARGETS = {
    "CHEMBL1978": "DNMT1",
    "CHEMBL3004": "HDAC6",
}

BASE_DIR = Path(__file__).resolve().parents[1]
SEED_CSV = BASE_DIR / "data" / "raw" / "chembl_seed.csv"
GRAPH_PATH = BASE_DIR / "data" / "graph_db" / "regen_graph.gpickle"


def main() -> None:
    if not SEED_CSV.exists():
        raise FileNotFoundError(f"Seed CSV not found: {SEED_CSV}")

    graph = nx.DiGraph()

    for target_id, target_name in TARGETS.items():
        graph.add_node(target_id, type="Target", name=target_name)

    df = pd.read_csv(SEED_CSV)
    required_columns = {
        "Molecule_ChEMBL_ID",
        "Canonical_SMILES",
        "Target_ChEMBL_ID",
        "IC50_Value",
    }
    missing_cols = required_columns.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in seed CSV: {sorted(missing_cols)}")

    for _, row in df.iterrows():
        molecule_id = str(row["Molecule_ChEMBL_ID"]).strip()
        target_id = str(row["Target_ChEMBL_ID"]).strip()
        smiles = str(row["Canonical_SMILES"]).strip()

        if not molecule_id or not target_id or not smiles:
            continue

        try:
            ic50_value = float(row["IC50_Value"])
        except (TypeError, ValueError):
            continue

        graph.add_node(molecule_id, type="Molecule", smiles=smiles)
        graph.add_edge(molecule_id, target_id, type="INHIBITS", ic50=ic50_value)

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GRAPH_PATH.open("wb") as f:
        pickle.dump(graph, f)

    print(f"Graph saved: {GRAPH_PATH}")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")


if __name__ == "__main__":
    main()
