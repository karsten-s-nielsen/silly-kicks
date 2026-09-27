"""Task 1 gates for the DAS golden fixture (the accessible-space 2.0.15 parity oracle).

The checksum + non-vacuity tests always run (the fixture is committed). The regeneration test runs
only where the dev-only ``das-reference`` extra (accessible-space 2.0.15) is importable.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
import sys

import numpy as np
import pytest

from tests.tracking._das_golden import GOLDEN_DIR, GOLDEN_FILES, NORMAL_SCENES, load_golden


def test_golden_checksums_match():
    lines = (GOLDEN_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert lines, "SHA256SUMS is empty"
    for line in lines:
        digest, name = line.split(maxsplit=1)
        actual = hashlib.sha256((GOLDEN_DIR / name).read_bytes()).hexdigest()
        assert actual == digest, f"checksum drift for {name}"


def _asp_2_0_15_available() -> bool:
    try:
        return importlib.metadata.version("accessible_space") == "2.0.15"
    except importlib.metadata.PackageNotFoundError:
        return False


@pytest.mark.skipif(not _asp_2_0_15_available(), reason="requires the das-reference extra (accessible-space==2.0.15)")
def test_golden_regenerates_byte_for_byte(tmp_path):
    subprocess.run(  # noqa: S603 -- fixed argv, sys.executable
        [sys.executable, str(GOLDEN_DIR / "_generate.py"), "--out", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    for name in GOLDEN_FILES:
        assert (tmp_path / name).read_bytes() == (GOLDEN_DIR / name).read_bytes(), f"{name} not reproduced"


def test_golden_is_non_vacuous():
    g = load_golden()
    # Normal scenes carry many real finite values (not a crash-only fixture).
    assert np.isfinite(g.das_team["DAS"].to_numpy(dtype=float)).sum() >= 60
    assert np.isfinite(g.das_player["DAS"].to_numpy(dtype=float)).sum() >= 900
    # The NaN-ball defect (D-BALLNAN) is recorded as AS/DAS == 0.0, not NaN.
    vb = g.das_team[g.das_team["scene_id"] == "V-BALLNAN"]
    assert len(vb) and (vb["DAS"].to_numpy(dtype=float) == 0.0).all()
    # The library raises on an absent pass frame / team and on a double ball row -- recorded, not swallowed.
    assert g.errors["V-XC-FRAME"]["type"] == "ValueError"
    assert g.errors["V-XC-TEAM"]["type"] == "ValueError"
    assert "V-MULTIBALL" in g.errors


def test_every_normal_scene_has_reference_rows():
    g = load_golden()
    present = set(g.das_team["scene_id"]) | set(g.das_player["scene_id"])
    missing = [s for s in NORMAL_SCENES if s not in present]
    assert not missing, f"normal scenes missing a reference: {missing}"


def test_frames_for_applies_id_dtype_axis():
    g = load_golden()
    assert str(g.frames_for("S12a")["player_id"].dtype) == "Int64"
    assert str(g.frames_for("S12b")["player_id"].dtype) == "object"
    assert str(g.frames_for("S12c")["player_id"].dtype) == "category"
    # coords are float32 storage on every scene
    assert str(g.frames_for("S01")["x"].dtype) == "float32"
