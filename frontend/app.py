from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Regen Platform Frontend", layout="wide")

ROOT_DIR = Path(__file__).resolve().parents[1]
KG_HTML_PATH = ROOT_DIR / "KG" / "output" / "knowledge_graph.html"
SCREENER_INPUT_DIR = ROOT_DIR / "epi-screener" / "data" / "input"


def render_knowledge_graph() -> None:
    st.title("Knowledge Graph Viewer")
    st.caption("Rendering `KG/output/knowledge_graph.html`.")

    if not KG_HTML_PATH.exists():
        st.error(f"Knowledge graph file not found: {KG_HTML_PATH}")
        return

    html_content = KG_HTML_PATH.read_text(encoding="utf-8")
    components.html(html_content, height=900, scrolling=True)


def sidebar_csv_uploader() -> None:
    st.sidebar.header("Epigenetic Screener")
    st.sidebar.write("Upload one or more CSV files for `epi-screener/data/input/`.")

    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        return

    SCREENER_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []

    for uploaded_file in uploaded_files:
        target_path = SCREENER_INPUT_DIR / uploaded_file.name
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_files.append(str(target_path))

    st.sidebar.success(f"Saved {len(saved_files)} file(s).")
    for path in saved_files:
        st.sidebar.code(path)


def main() -> None:
    sidebar_csv_uploader()
    render_knowledge_graph()


if __name__ == "__main__":
    main()
