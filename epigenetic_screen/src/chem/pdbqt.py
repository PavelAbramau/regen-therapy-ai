from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class PDBQTConversionError(RuntimeError):
    """Raised when ligand conversion to PDBQT fails."""


def convert_sdf_to_pdbqt(sdf_path: Path, out_dir: Path) -> Path:
    """
    Convert ligand SDF to PDBQT through external tools.

    Expected external tools:
    - Open Babel (`obabel`) OR
    - MGLTools/AutoDockTools `prepare_ligand4.py`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sdf_path.stem}.pdbqt"

    obabel = shutil.which("obabel")
    if obabel:
        cmd = [obabel, str(sdf_path), "-O", str(out_path), "--partialcharge", "gasteiger"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise PDBQTConversionError(proc.stderr.strip() or "obabel failed")
        return out_path

    raise PDBQTConversionError(
        "No supported PDBQT converter found. Install Open Babel (`obabel`) or provide a custom adapter."
    )

