from __future__ import annotations

import pickle
import re
from pathlib import Path

import networkx as nx
import pandas as pd


# Map ChEMBL target IDs to preferred human-readable gene/product names for node labels.
TARGET_NAME_MAP: dict[str, str] = {
    "CHEMBL1978": "DNMT1",
    "CHEMBL3004": "HDAC6",
    # Extend as you add seed targets — epigenetics / regeneration-relevant examples:
    "CHEMBL1991": "DNMT3A",
    "CHEMBL1936": "HDAC1",
    "CHEMBL325": "HDAC2",
    "CHEMBL2446": "KDM6B",
    "CHEMBL5122": "KDM6A",
    "CHEMBL1163125": "BRD4",
}

# Canonical pathway nodes: stable graph id -> display label & hover description.
PATHWAYS: dict[str, dict[str, str]] = {
    "PATHWAY:Tissue_Regeneration": {
        "label": "Tissue Regeneration",
        "title": (
            "Tissue regeneration: restoration of damaged tissue architecture and function; "
            "relevant epigenetic programs after injury."
        ),
    },
    "PATHWAY:Fibroblast_Reprogramming": {
        "label": "Fibroblast Reprogramming",
        "title": (
            "Fibroblast reprogramming: transitions in fibroblast cell state and ECM "
            "remodeling; epigenetic regulators modulate plasticity versus fibrosis."
        ),
    },
    "PATHWAY:Wound_Healing": {
        "label": "Wound Healing",
        "title": (
            "Wound healing phases: inflammatory, proliferative, and remodeling programs; "
            "targets herein may regulate repair versus chronic wound phenotypes."
        ),
    },
}

# Which targets link to which pathways (REGULATES). Keys must be ChEMBL target IDs present in CSV.
DEFAULT_PATHWAY_IDS = tuple(PATHWAYS.keys())

TARGET_PATHWAY_IDS: dict[str, tuple[str, ...]] = {
    # DNMT1: methylation-dependent gene programs tied to differentiation and repair
    "CHEMBL1978": DEFAULT_PATHWAY_IDS,
    # HDAC6: chromatin/signaling modulation in stromal and repair contexts
    "CHEMBL3004": DEFAULT_PATHWAY_IDS,
}

BASE_DIR = Path(__file__).resolve().parents[1]
SEED_CSV = BASE_DIR / "data" / "raw" / "chembl_seed.csv"
GRAPH_PATH = BASE_DIR / "data" / "graph_db" / "regen_graph.gpickle"

_SMILES_TRUNCATE_LEN = 48
_CHEMBL_NUM_SUFFIX = re.compile(r"CHEMBL(\d+)", re.I)


def _target_label(target_id: str) -> str:
    tid = target_id.strip()
    return TARGET_NAME_MAP.get(tid, tid)


def _target_hover_title(target_id: str, display_label: str) -> str:
    return (
        f"<b>{display_label}</b><br>"
        f"ChEMBL target ID: <code>{target_id}</code><br>"
        f"Class: Protein target<br>"
        f"Bio: Epigenetic / chromatin-associated target in regeneration-relevant assays."
    )


def _truncated_smiles(smiles: str, max_len: int = _SMILES_TRUNCATE_LEN) -> str:
    s = smiles.strip()
    if len(s) <= max_len:
        return s
    return f"{s[: max_len - 1]}…"


def _molecule_fallback_label(molecule_id: str) -> str:
    """When pref_name is missing, use Analog-<numeric> from ChEMBL id if possible."""
    m = _CHEMBL_NUM_SUFFIX.match(molecule_id.strip())
    if m:
        return f"Analog-{m.group(1)}"
    return molecule_id.strip() or "Unknown_molecule"


def _molecule_label_and_title(
    *,
    molecule_id: str,
    smiles: str,
    pref_raw: object,
) -> tuple[str, str]:
    smiles = smiles.strip()

    pref: str | None = None
    if pref_raw is not None and not (isinstance(pref_raw, float) and pd.isna(pref_raw)):
        p = str(pref_raw).strip()
        pref = p if p else None

    if pref:
        mol_label = pref
        title_head = mol_label
    elif smiles:
        mol_label = _truncated_smiles(smiles)
        title_head = mol_label
    else:
        mol_label = _molecule_fallback_label(molecule_id)
        title_head = mol_label

    title = (
        f"<b>{title_head}</b><br>"
        f"ChEMBL: <code>{molecule_id}</code><br>"
        f"Type: Small molecule (from assay seed)<br>"
    )
    if smiles:
        title += f"<br>Canonical SMILES:<br>{smiles}"
    if pref and mol_label != pref:
        title += f"<br><br>Preferred name (ChEMBL): {pref}"

    return mol_label, title


def _pathway_hover(pathway_graph_id: str, meta: dict[str, str]) -> str:
    label = meta["label"]
    return f"<b>{label}</b><br>Type: Pathway<br><br>{meta['title']}"


def _ensure_targets_and_pathways(graph: nx.DiGraph, target_ids_seen: set[str]) -> None:
    for target_id in sorted(target_ids_seen):
        lbl = _target_label(target_id)
        graph.add_node(
            target_id,
            label=lbl,
            type="Target",
            title=_target_hover_title(target_id, lbl),
        )

        pw_ids = TARGET_PATHWAY_IDS.get(target_id)
        if not pw_ids:
            pw_ids = DEFAULT_PATHWAY_IDS

        for pw_id in pw_ids:
            if pw_id not in PATHWAYS:
                continue
            if not graph.has_node(pw_id):
                meta = PATHWAYS[pw_id]
                graph.add_node(
                    pw_id,
                    label=meta["label"],
                    type="Pathway",
                    title=_pathway_hover(pw_id, meta),
                )
            graph.add_edge(
                target_id,
                pw_id,
                type="REGULATES",
                title=(
                    f"{lbl} regulates or contributes to modulation of pathway "
                    f"“{PATHWAYS[pw_id]['label']}”. (Literature-informed association.)"
                ),
            )


def build_core_graph(csv_path: Path | None = None) -> nx.DiGraph:
    csv_path = csv_path or SEED_CSV

    graph = nx.DiGraph()

    if not csv_path.exists():
        raise FileNotFoundError(f"Seed CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {
        "Molecule_ChEMBL_ID",
        "Canonical_SMILES",
        "Target_ChEMBL_ID",
        "IC50_Value",
    }
    missing_cols = required_columns.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in seed CSV: {sorted(missing_cols)}")

    pref_col_present = "pref_name" in df.columns

    target_ids_seen: set[str] = set()

    for _, row in df.iterrows():
        molecule_id = str(row["Molecule_ChEMBL_ID"]).strip()
        target_id = str(row["Target_ChEMBL_ID"]).strip()
        smiles_raw = row["Canonical_SMILES"]
        smiles = "" if smiles_raw is None or (isinstance(smiles_raw, float) and pd.isna(smiles_raw)) else str(smiles_raw).strip()
        if smiles.lower() == "nan":
            smiles = ""

        pref_raw = row["pref_name"] if pref_col_present else None

        if not molecule_id or not target_id:
            continue

        try:
            ic50_value = float(row["IC50_Value"])
        except (TypeError, ValueError):
            continue

        pref_clean: str | None = None
        if pref_raw is not None and not (
            isinstance(pref_raw, float) and pd.isna(pref_raw)
        ):
            p = str(pref_raw).strip()
            pref_clean = p if p else None

        if not smiles:
            mol_label = _molecule_fallback_label(molecule_id)
            mol_title = (
                f"<b>{mol_label}</b><br>"
                f"ChEMBL: <code>{molecule_id}</code><br>"
                f"Type: Small molecule<br>"
                f"<i>No SMILES in seed row — using formatted label.</i>"
            )
            graph.add_node(
                molecule_id,
                label=mol_label,
                type="Molecule",
                title=mol_title,
                smiles=None,
                pref_name=pref_clean,
            )
        else:
            mol_label, mol_title = _molecule_label_and_title(
                molecule_id=molecule_id,
                smiles=smiles,
                pref_raw=pref_raw,
            )
            graph.add_node(
                molecule_id,
                label=mol_label,
                type="Molecule",
                title=mol_title,
                smiles=smiles,
                pref_name=pref_clean,
            )

        target_ids_seen.add(target_id)

        graph.add_edge(
            molecule_id,
            target_id,
            type="INHIBITS",
            ic50=ic50_value,
            title=(
                f"Assay-derived inhibition / binding (IC<sub>50</sub> proxy): "
                f"{ic50_value} nM (from seed)."
            ),
        )

    _ensure_targets_and_pathways(graph, target_ids_seen)
    return graph


def validate_node_attributes(graph: nx.DiGraph) -> None:
    for node_id, attrs in graph.nodes(data=True):
        missing = []
        if not attrs.get("label"):
            missing.append("label")
        if not attrs.get("type"):
            missing.append("type")
        if not attrs.get("title"):
            missing.append("title")
        if missing:
            raise ValueError(
                f"Node {node_id!r} missing attributes: {missing}. Full attrs: {attrs!r}"
            )


def main() -> None:
    graph = build_core_graph()
    validate_node_attributes(graph)

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GRAPH_PATH.open("wb") as f:
        pickle.dump(graph, f)

    print(f"Graph saved: {GRAPH_PATH}")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")


if __name__ == "__main__":
    main()
