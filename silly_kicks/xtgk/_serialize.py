"""Pickle-free artifact I/O for the fitted possession-value surfaces (ADR-036 §4/G4).

Mirrors the house convention (ghost-GK / xShot / GkCompletionModel): npz for arrays, JSON for
metadata, SHA256SUMS for integrity. No pickle. A fitted grid, not an ADR-011 weights lifecycle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

_ARRAYS = "surfaces.npz"
_META = "metadata.json"
_SUMS = "SHA256SUMS"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_surface(directory, *, surfaces: dict, support: dict, metadata: dict) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for p in (1, 2, 3):
        arrays[f"surface_{p}"] = surfaces[p]
        arrays[f"support_{p}"] = support[p]
    np.savez(d / _ARRAYS, **arrays)
    (d / _META).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    lines = [f"{_sha256(d / f)}  {f}\n" for f in (_ARRAYS, _META)]
    (d / _SUMS).write_text("".join(lines))


def load_surface(directory):
    d = Path(directory)
    expected = {}
    for line in (d / _SUMS).read_text().splitlines():
        h, f = line.split("  ", 1)
        expected[f.strip()] = h.strip()
    for f in (_ARRAYS, _META):
        if _sha256(d / f) != expected.get(f):
            raise ValueError(f"checksum mismatch for {f} — artifact tampered or corrupt")
    npz = np.load(d / _ARRAYS)
    surfaces = {p: npz[f"surface_{p}"] for p in (1, 2, 3)}
    support = {p: npz[f"support_{p}"] for p in (1, 2, 3)}
    metadata = json.loads((d / _META).read_text())
    return surfaces, support, metadata
