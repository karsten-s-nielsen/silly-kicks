"""ADR-105 Task 3 (VKS-PLAN-05): `defensive_credit` and `gk_decision/_reconstruct.py` are PITCH-CONTROL
FREE, so the ADR-105 cycle does NOT route them through the batch (dead edits / a vacuous guard). This
pins the fact by reading IMPORTS (code, not prose -- ADR-056): a future edit that adds a pitch-control
symbol there fails CI, prompting a routing decision rather than silently reintroducing per-sample PC."""

from __future__ import annotations

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2] / "silly_kicks"
_PC_NAMES = {
    "compute_pitch_control",
    "compute_pitch_control_batch",
    "compute_threat_pc",
    "compute_threat_pc_batch",
    "compute_gk_influence",
    "PitchControlCache",
    "compute_spearman",
    "compute_spearman_batch",
}
_PC_MODULE_HINTS = ("pitch_control", "_cover_shadows", "_gk_influence")


def _pc_import_hits(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if any(h in mod for h in _PC_MODULE_HINTS):
                hits.append(f"{path.name}: from {mod}")
            hits += [f"{path.name}: {a.name}" for a in node.names if a.name in _PC_NAMES]
        elif isinstance(node, ast.Import):
            hits += [f"{path.name}: import {a.name}" for a in node.names if any(h in a.name for h in _PC_MODULE_HINTS)]
    return hits


def test_defensive_credit_is_pitch_control_free():
    hits = [h for py in (_ROOT / "tracking" / "defensive_credit").rglob("*.py") for h in _pc_import_hits(py)]
    assert not hits, (
        f"defensive_credit imports pitch-control symbol(s) {hits}: no longer PC-free -- "
        "route it or update this audit (VKS-PLAN-05)."
    )


def test_gk_decision_reconstruct_is_pitch_control_free():
    hits = _pc_import_hits(_ROOT / "gk_decision" / "_reconstruct.py")
    assert not hits, (
        f"gk_decision/_reconstruct.py imports pitch-control symbol(s) {hits}: no longer PC-free (VKS-PLAN-05)."
    )
