from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .receptor import ReceptorSpec


def _mock_score(compound_id: str, target_name: str) -> float:
    h = hashlib.sha256(f"{compound_id}|{target_name}".encode("utf-8")).hexdigest()
    # Deterministic docking-like score in [-12, -4]
    return -12.0 + (int(h[:8], 16) % 8000) / 1000.0


def dock_ligands(
    receptor: ReceptorSpec,
    ligand_pdbqt_paths: dict[str, Path],
    output_dir: Path,
    *,
    use_mock_if_unavailable: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    vina_cls = None
    if not use_mock_if_unavailable:
        try:
            from vina import Vina  # type: ignore

            vina_cls = Vina
        except Exception as exc:
            raise RuntimeError("Vina requested but Python package is unavailable.") from exc
    else:
        try:
            from vina import Vina  # type: ignore

            vina_cls = Vina
        except Exception:
            vina_cls = None

    if vina_cls is not None and receptor.receptor_path.exists():
        # Real Vina path
        vina = vina_cls(sf_name="vina")
        vina.set_receptor(str(receptor.receptor_path))
        vina.compute_vina_maps(center=receptor.box_center, box_size=receptor.box_size)
        for compound_id, ligand_pdbqt in ligand_pdbqt_paths.items():
            if not ligand_pdbqt.exists():
                failures.append(
                    {"compound_id": compound_id, "stage": "docking", "reason": f"Missing ligand file {ligand_pdbqt}"}
                )
                continue
            try:
                vina.set_ligand_from_file(str(ligand_pdbqt))
                pre_score = float(vina.score()[0])
                opt_score = float(vina.optimize()[0])
                vina.dock(exhaustiveness=receptor.exhaustiveness, n_poses=receptor.n_poses)
                best_score = float(vina.energies(n_poses=1)[0][0])
                out_pose = output_dir / f"{compound_id}_{receptor.target_name}_poses.pdbqt"
                vina.write_poses(str(out_pose), n_poses=receptor.n_poses, overwrite=True)
                rows.append(
                    {
                        "compound_id": compound_id,
                        "target_name": receptor.target_name,
                        "vina_predock_score": pre_score,
                        "vina_optimized_score": opt_score,
                        "vina_best_score": best_score,
                        "pose_path": str(out_pose),
                        "docking_engine": "vina",
                    }
                )
            except Exception as e:  # pragma: no cover
                failures.append({"compound_id": compound_id, "stage": "docking", "reason": str(e)})
        return rows, failures

    # Mock path for startup dev environments where vina/pdbqt tools are missing.
    for compound_id, _ligand_pdbqt in ligand_pdbqt_paths.items():
        best = _mock_score(compound_id, receptor.target_name)
        rows.append(
            {
                "compound_id": compound_id,
                "target_name": receptor.target_name,
                "vina_predock_score": round(best + 0.7, 3),
                "vina_optimized_score": round(best + 0.2, 3),
                "vina_best_score": round(best, 3),
                "pose_path": "",
                "docking_engine": "mock_vina",
            }
        )

    if not receptor.receptor_path.exists():
        failures.append(
            {
                "compound_id": "",
                "stage": "docking",
                "reason": f"Receptor file missing for {receptor.target_name}: {receptor.receptor_path}. Used mock docking.",
            }
        )

    return rows, failures

