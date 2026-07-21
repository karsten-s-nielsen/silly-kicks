"""Bundled weights artifacts carry learned parameters only (spec 2026-07-20 §5).

Allowlist, not denylist: the ghost npz may contain ONLY the parameter arrays; any unrecognized
name FAILS (fail-closed against a rename or a new per-sample array). Only ghost ships an npz today
(the other six bundled dirs are model.json boosters/logistic — no arrays to inspect), so the gate
is scoped to the ghost npz by enumeration, with a non-vacuity meta-test.

A bare size cap (max(shape) <= N) was considered and rejected: fail-open for a small-subsample
artifact (per-sample arrays under N pass) AND false-positive for a legitimately larger tree (a
max_leaf_nodes bump pushes tree_nodes_* above N). The name allowlist has neither failure mode.
"""

import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2] / "silly_kicks"
_GHOST_NPZ = _ROOT / "tracking" / "_ghost_gk_weights" / "default" / "rfcde_weights.npz"

# The ONLY array names a parameters-only ghost artifact may contain. Anything else fails.
_ALLOWED_EXACT = {"n_trees", "n_trees_y", "baseline_x", "baseline_y"}
_ALLOWED_PREFIX = ("tree_nodes_", "tree_dtype_")  # covers both x and y ensembles (…_y_ starts with these)


def _is_allowed(name: str) -> bool:
    return name in _ALLOWED_EXACT or name.startswith(_ALLOWED_PREFIX)


def test_the_ghost_npz_exists_and_is_enumerated():
    """Non-vacuity: the gate must actually be pointed at a real bundled artifact."""
    assert _GHOST_NPZ.exists(), f"ghost artifact not found at {_GHOST_NPZ}"


def test_ghost_npz_contains_only_allowed_parameter_arrays():
    with np.load(_GHOST_NPZ, allow_pickle=True) as z:
        unexpected = sorted(n for n in z.files if not _is_allowed(n))
    assert not unexpected, f"ghost npz carries non-parameter arrays {unexpected} — parameters-only violated"


def test_shipped_ghost_default_is_parameters_only_v130():
    """C3: lock the SHIPPED artifact contract, not just the re-save round-trip.

    The round-trip contract test re-saves a copy and checks that; it always passes regardless of
    the committed artifact. This asserts the file that actually ships."""
    meta = json.loads((_GHOST_NPZ.parent / "metadata.json").read_text())
    assert meta["version"] == "1.3.0"
    assert meta["stores_training_data"] is False


def test_allowlist_is_not_vacuous(tmp_path):
    """A synthetic artifact WITH an unrecognized array must fail the allowlist rule."""
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, gk_positions=np.zeros((3, 2)), n_trees=np.array([1]))
    with np.load(bad, allow_pickle=True) as z:
        unexpected = sorted(n for n in z.files if not _is_allowed(n))
    assert unexpected == ["gk_positions"]  # a rename is caught — this is the fail-closed property
