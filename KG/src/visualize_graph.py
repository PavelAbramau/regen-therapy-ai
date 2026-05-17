from __future__ import annotations

import hashlib
import math
import pickle
from pathlib import Path
from typing import Any

from pyvis.network import Network


BASE_DIR = Path(__file__).resolve().parents[1]
GRAPH_PATH = BASE_DIR / "data" / "graph_db" / "regen_graph.gpickle"
OUTPUT_HTML = BASE_DIR / "output" / "knowledge_graph.html"

# Geometry for seed positions (helps physics settle with pathways/targets central, molecules fanning out).
_PATHWAY_CLUSTER_RADIUS = 55.0
_TARGET_RING_RADIUS = 190.0
_MOLECULE_RADIUS_BASE = 95.0
_MOLECULE_RADIUS_SPREAD = 110.0


def _node_visuals(node_type: str) -> dict[str, Any]:
    """Color, shape, size, mass, font for readable on-canvas labels (vis-network format)."""
    if node_type == "Pathway":
        return {
            "color": {
                "background": "#22c55e",
                "border": "#065f46",
                "highlight": {"background": "#4ade80", "border": "#064e3b"},
            },
            "shape": "star",
            "size": 46,
            "mass": 12,
            "font": {"size": 16, "face": "Tahoma", "color": "#0f172a"},
        }
    if node_type == "Target":
        return {
            "color": {
                "background": "#ef4444",
                "border": "#7f1d1d",
                "highlight": {"background": "#f87171", "border": "#450a0a"},
            },
            "shape": "square",
            "size": 30,
            "mass": 8,
            "font": {"size": 14, "face": "Tahoma", "color": "#fafafa"},
        }
    # Molecule and unknown defaults
    return {
        "color": {
            "background": "#3b82f6",
            "border": "#1e3a8a",
            "highlight": {"background": "#60a5fa", "border": "#172554"},
        },
        "shape": "dot",
        "size": 8,
        "mass": 1,
        "font": {"size": 10, "face": "Tahoma", "color": "#f8fafc"},
    }


def _stable_offset(node_id: str, span: float) -> float:
    digest = hashlib.sha256(node_id.encode("utf-8")).digest()
    u = int.from_bytes(digest[:4], "big") / float(2**32) - 0.5
    return u * 2 * span


def compute_seed_xy(graph_nx: Any) -> dict[str, tuple[float, float]]:
    """Initial x,y hints: pathways at center cluster, targets on inner ring, molecules outside targets."""
    pathway_ids = [
        nid for nid, attr in graph_nx.nodes(data=True) if attr.get("type") == "Pathway"
    ]
    target_ids = [
        nid for nid, attr in graph_nx.nodes(data=True) if attr.get("type") == "Target"
    ]
    molecule_ids = [
        nid for nid, attr in graph_nx.nodes(data=True) if attr.get("type") == "Molecule"
    ]

    pos: dict[str, tuple[float, float]] = {}
    sorted_targets = sorted(target_ids)

    n_pw = max(len(pathway_ids), 1)
    for i, pw in enumerate(sorted(pathway_ids)):
        angle = (2 * math.pi * i) / n_pw
        pos[pw] = (
            _PATHWAY_CLUSTER_RADIUS * math.cos(angle),
            _PATHWAY_CLUSTER_RADIUS * math.sin(angle),
        )

    n_t = max(len(sorted_targets), 1)
    for i, tid in enumerate(sorted_targets):
        angle = (2 * math.pi * i) / n_t - math.pi / 2
        pos[tid] = (
            _TARGET_RING_RADIUS * math.cos(angle),
            _TARGET_RING_RADIUS * math.sin(angle),
        )

    # Molecules: place beyond their inhibited target(s); edge direction is molecule -> target (INHIBITS).
    for mid in molecule_ids:
        targets_for_mol = [t for t in graph_nx.successors(mid) if t in pos]
        if not targets_for_mol:
            pos[mid] = (
                _stable_offset(mid, _TARGET_RING_RADIUS + 260),
                _stable_offset(mid[::-1], _TARGET_RING_RADIUS + 260),
            )
            continue

        cx = sum(pos[t][0] for t in targets_for_mol) / len(targets_for_mol)
        cy = sum(pos[t][1] for t in targets_for_mol) / len(targets_for_mol)
        hyp = math.hypot(cx, cy) or 1.0
        ux, uy = cx / hyp, cy / hyp
        jitter = (_stable_offset(mid, 0.35), _stable_offset(mid + "|y", 0.35))
        jx = math.cos(jitter[0]) * ux - math.sin(jitter[0]) * uy
        jy = math.sin(jitter[0]) * ux + math.cos(jitter[0]) * uy
        radial = _MOLECULE_RADIUS_BASE + _molecule_spread(mid)
        jitter2 = jitter[1] * 25.0
        pos[mid] = (cx + jx * radial + jitter2 * (-uy), cy + jy * radial + jitter2 * ux)

    return pos


def _molecule_spread(mid: str) -> float:
    h = int(hashlib.sha1(mid.encode("utf-8")).hexdigest()[:8], 16)
    return _MOLECULE_RADIUS_SPREAD * (0.65 + (h % 129) / 256.0)


def _inject_stabilize_then_freeze(html_path: Path) -> None:
    """
    PyVis's bundled template keeps physics enabled after stabilization, which causes
    endless drift/rotation. After the configured stabilization iteration budget
    completes, disable physics entirely in the browser.
    """
    text = html_path.read_text(encoding="utf-8")
    marker_js = "// regen-platform: freeze physics after stabilization"
    if marker_js in text:
        return

    needle = "network = new vis.Network(container, data, options);"
    if needle not in text:
        raise RuntimeError(
            f"Could not find vis.Network construction in {html_path}; "
            "PyVis template may have changed."
        )

    injected = (
        needle
        + "\n\n                  "
        + marker_js
        + "\n                  "
        + 'network.once("stabilizationIterationsDone", function () {\n'
        + "                      network.setOptions({ physics: false });\n"
        + "                  });\n"
        + '                  network.on("stabilizationFailed", function () {\n'
        + "                      network.setOptions({ physics: false });\n"
        + "                  });\n"
    )
    html_path.write_text(text.replace(needle, injected, 1), encoding="utf-8")


def main() -> None:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph file not found: {GRAPH_PATH}")

    with GRAPH_PATH.open("rb") as f:
        graph = pickle.load(f)

    net = Network(
        height="920px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#f8fafc",
        font_color=False,
    )

    net.show_buttons(filter_=["physics"])

    # Barnes–Hut: strong repulsive field + damping; stabilize for N iterations then freeze (see HTML inject).
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.25,
        spring_length=140,
        spring_strength=0.04,
        damping=0.09,
        overlap=0.65,
    )
    setattr(net.options.physics, "solver", "barnesHut")

    stab = net.options.physics.stabilization
    stab.enabled = True
    stab.iterations = 500
    stab.updateInterval = 25
    stab.fit = True

    net.set_edge_smooth("continuous")

    seed_xy = compute_seed_xy(graph)

    for node_id, attrs in graph.nodes(data=True):
        n_type = str(attrs.get("type", "Unknown"))
        vis = _node_visuals(n_type)
        hr_label = attrs.get("label") or attrs.get("name") or node_id

        tooltip = attrs.get("title")
        if not tooltip:
            title_lines = [
                f"ID: {node_id}",
                f"Type: {n_type}",
            ]
            smiles = attrs.get("smiles")
            if smiles:
                title_lines.append(f"SMILES: {smiles}")
            tooltip = "<br>".join(title_lines)

        xy = seed_xy.get(node_id, (0.0, 0.0))

        net.add_node(
            node_id,
            label=str(hr_label),
            title=str(tooltip),
            color=vis["color"],
            shape=vis["shape"],
            size=vis["size"],
            mass=vis["mass"],
            font=vis["font"],
            borderWidth=2,
            x=float(xy[0]),
            y=float(xy[1]),
            physics=True,
        )

    for source, target, attrs in graph.edges(data=True):
        rel_type = attrs.get("type", "RELATED_TO")
        ic50 = attrs.get("ic50")
        hover = attrs.get("title")
        if hover:
            title = str(hover)
        else:
            title = f"Relationship: {rel_type}"
            if ic50 is not None:
                title += f"<br>IC50: {ic50} nM"

        if rel_type == "INHIBITS":
            edge_color = {
                "inherit": False,
                "color": "#94a3b8",
                "highlight": "#64748b",
                "opacity": 0.5,
            }
            width = 0.42
            physics_enabled = True
        else:
            edge_color = {
                "inherit": False,
                "color": "#0d9488",
                "highlight": "#14b8a6",
                "opacity": 0.95,
            }
            width = 1.9
            physics_enabled = True

        net.add_edge(
            source,
            target,
            title=title,
            label="",
            color=edge_color,
            width=width,
            physics=physics_enabled,
            arrows="to",
        )

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    net.show(str(OUTPUT_HTML), notebook=False)
    _inject_stabilize_then_freeze(OUTPUT_HTML)
    print(f"Visualization saved: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()