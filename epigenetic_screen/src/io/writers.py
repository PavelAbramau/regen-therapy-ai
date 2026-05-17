from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dirs(paths: list[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def write_stage_dataframe(df: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def write_stage_manifest(
    manifest_path: Path,
    *,
    stage: str,
    row_count: int,
    input_sources: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(row_count),
        "input_sources": input_sources,
    }
    if extra:
        payload.update(extra)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_failure_log(failures: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not failures:
        pd.DataFrame(columns=["compound_id", "stage", "reason"]).to_csv(output_csv, index=False)
        return
    pd.DataFrame(failures).to_csv(output_csv, index=False)

