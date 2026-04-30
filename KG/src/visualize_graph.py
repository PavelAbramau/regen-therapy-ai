from __future__ import annotations

import pickle
from pathlib import Path

from pyvis.network import Network


BASE_DIR = Path(__file__).resolve().parents[1]
GRAPH_PATH = BASE_DIR / "data" / "graph_db" / "regen_graph.gpickle"
OUTPUT_HTML = BASE_DIR / "output" / "knowledge_graph.html"


def node_style(node_type: str) -> dict[str, str]:
    if node_type == "Target":
        return {"color": "#d62728", "shape": "square", "size": 32}
    return {"color": "#1f77b4", "shape": "dot", "size": 14}


def main() -> None:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph file not found: {GRAPH_PATH}")

    with GRAPH_PATH.open("rb") as f:
        graph = pickle.load(f)
    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.force_atlas_2based()

    for node_id, attrs in graph.nodes(data=True):
        n_type = attrs.get("type", "Unknown")
        style = node_style(str(n_type))
        label = attrs.get("name", node_id)
        title_lines = [f"ID: {node_id}", f"Type: {n_type}"]
        smiles = attrs.get("smiles")
        if smiles:
            title_lines.append(f"SMILES: {smiles}")

        net.add_node(
            n_id=node_id,
            label=str(label),
            title="<br>".join(title_lines),
            color=style["color"],
            shape=style["shape"],
            size=style["size"],
        )

    for source, target, attrs in graph.edges(data=True):
        rel_type = attrs.get("type", "RELATED_TO")
        ic50 = attrs.get("ic50")
        title = f"Relationship: {rel_type}"
        if ic50 is not None:
            title += f"<br>IC50: {ic50} nM"

        net.add_edge(source, target, title=title, label=str(rel_type))

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    net.show(str(OUTPUT_HTML), notebook=False)
    print(f"Visualization saved: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
