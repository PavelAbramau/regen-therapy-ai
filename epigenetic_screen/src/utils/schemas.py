from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StagePaths:
    csv: str
    manifest: str
    failures: str

