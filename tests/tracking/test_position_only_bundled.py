"""Bundled position-only variants + the re-fit ghost default (Phase B / commit 2, ADR-067).

The Phase-A tests exercise the resolver/seam with mocks and the UNBUNDLED fallback. This file
exercises the REAL bundled artifacts:

* the three ``position_only`` variants LOAD clean (SHA-256 + chirality + declared-constant
  contract all pass -- ``load()`` RAISES on any of those, so a plain load is the assertion);
* the ``add_*`` / serve path produces a VALUE + provenance ``"position_only"`` on a declared
  velocity-less frame (the SB360 unlock -- it was honest-NaN in 4.90.0);
* the RESTRICTED ghost bundle carries a machine-checkable reproducibility caveat (M4);
* the ghost ``default`` bundle is the native-SkillCorner re-fit (``training_commit=a0fc9f9``).

The bundled artifacts were fingerprinted on the DGX (pandas 3), so on a pandas-2 runtime their
``probe_sha256`` mismatches and ``load()`` emits a TOLERATED ``UnverifiableFeatureContractWarning``
(the fingerprint check is skipped; declared constants are still enforced) -- exactly as the existing
pandas-2-fingerprinted defaults warn on a pandas-3 runtime. The warning is not escalated
(``pyproject.toml``), so it is filtered here; a hard failure would be a SHA / chirality /
declared-constant mismatch, which raises.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import silly_kicks.tracking._ghost_gk as _gg
import silly_kicks.tracking._xcross_attempt as _xc
import silly_kicks.tracking._xshot_occurrence as _xs
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE

# The pandas-major probe mismatch is expected on one leg and tolerated (see the module docstring).
pytestmark = pytest.mark.filterwarnings("ignore::silly_kicks.tracking.UnverifiableFeatureContractWarning")

_WEIGHTS_ROOT = pathlib.Path(_gg.__file__).parent
_C1 = "0ce2c2187e09212440063f15494915f0f4a5f2ba"  # Phase-A commit (position-only weights' training_commit)
_C2 = "a0fc9f9ab2d1f40b5a44f0b5131ea14e374e0c1a"  # guardrails commit (faithful-ghost re-fit's training_commit)


def _declare_unavailable(frames: pd.DataFrame) -> pd.DataFrame:
    """SB360 freeze-frame shape: drop vx/vy, mark every row speed_source=unavailable."""
    out = frames.drop(columns=[c for c in ("vx", "vy") if c in frames.columns]).copy()
    out["speed"] = np.nan
    out["speed_source"] = SPEED_SOURCE_UNAVAILABLE
    return out


def _clear_variant_caches() -> None:
    for mod in (_xs, _xc, _gg):
        if hasattr(mod, "_VARIANT_CACHE"):
            mod._VARIANT_CACHE.clear()


# -- Load guards: the real bundled position_only artifacts load clean ------------------------------

_PO_CASES = [
    ("xshot", _xs.XShotOccurrenceModel, "_xshot_weights", 26),
    ("xcross", _xc.XCrossAttemptModel, "_xcross_weights", 15),
    ("ghost", _gg.GhostGkModel, "_ghost_gk_weights", 21),
]


@pytest.mark.parametrize("name,cls,root,n_feat", _PO_CASES, ids=[c[0] for c in _PO_CASES])
def test_bundled_position_only_loads_clean(name, cls, root, n_feat):
    # A plain load IS the assertion: load() raises IntegrityError on a SHA / chirality /
    # declared-constant mismatch. The only non-raising divergence is the pandas-major probe mismatch,
    # which warns (filtered above) and still enforces the declared constants.
    _clear_variant_caches()
    m = cls.from_variant("position_only")
    assert m is not None
    assert m.feature_set == "position_only"
    meta = json.loads((_WEIGHTS_ROOT / root / "position_only" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["feature_set"] == "position_only"
    assert len(meta["feature_names"]) == n_feat


def test_bundled_position_only_carry_training_provenance():
    # Bundled weights must be traceable to a commit (test_artifact_provenance_output enforces this
    # over the whole surface; pinned here to the position_only trio for a focused, in-place check).
    for root in ("_xshot_weights", "_xcross_weights", "_ghost_gk_weights"):
        meta = json.loads((_WEIGHTS_ROOT / root / "position_only" / "metadata.json").read_text(encoding="utf-8"))
        assert meta.get("training_commit") == _C1, f"{root}/position_only training_commit"


def test_bundled_ghost_default_is_the_native_refit():
    # scope item (b): the bundled ghost `default` was re-fit on the native SkillCorner corpus at the
    # guardrails commit, replacing the kloppy-contaminated weights. A velocity-bearing retrain trigger
    # (the velocity-path golden was re-captured; see test_ghost_gk_velocity_path_unchanged).
    _clear_variant_caches()
    m = _gg.GhostGkModel.from_variant("default")
    assert m.feature_set == "faithful"
    meta = json.loads((_WEIGHTS_ROOT / "_ghost_gk_weights" / "default" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["training_commit"] == _C2
    assert meta["feature_set"] == "faithful"
    assert len(meta["feature_names"]) == 26
    assert "skillcorner" in meta["corpus_provenance"]["providers"]


# -- M4: the restricted ghost bundle carries a machine-checkable reproducibility caveat ------------

_M4_CASES = [("_xshot_weights", "public"), ("_xcross_weights", "public"), ("_ghost_gk_weights", "restricted")]


@pytest.mark.parametrize("root,expected", _M4_CASES, ids=[c[0] for c in _M4_CASES])
def test_position_only_reproducibility_caveat(root, expected):
    metrics = json.loads((_WEIGHTS_ROOT / root / "position_only" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics.get("reproducibility") == expected, f"{root}/position_only reproducibility"
    if expected == "restricted":
        note = metrics.get("reproducibility_note")
        assert isinstance(note, str) and len(note) > 20, (
            "a restricted bundle must DOCUMENT why it is not public-reproducible (M4), machine-checkably"
        )


# -- Behavioral: the SB360 unlock on the real bundled weights -------------------------------------


def test_bundled_ghost_serves_a_coordinate_on_a_declared_freeze_frame():
    # The SB360 unlock: a declared-velocity-less (freeze-frame) input now yields a REAL ghost
    # coordinate via the bundled position_only variant -- it was honest-NaN under the ADR-054 refusal.
    from silly_kicks.tracking import add_ghost_gk  # add_ghost_gk lives in features.py, re-exported here
    from tests.sb360._fixture import build_leg_a

    _clear_variant_caches()
    actions, frames, _links = build_leg_a()
    out = add_ghost_gk(actions, frames, home_team_id=1)
    assert out["ghost_gk_x"].notna().any(), "the bundled position_only ghost must serve a coordinate on SB360"
    assert (out["ghost_gk_variant"] == "position_only").all()
    served = out[out["ghost_gk_x"].notna()]
    assert np.isfinite(served["ghost_gk_x"]).all() and np.isfinite(served["ghost_gk_y"]).all()


def test_bundled_xshot_serves_via_position_only_on_declared_frames():
    from tests.tracking.test_xshot_occurrence import _actions_and_frames_for_add

    _clear_variant_caches()
    actions, frames = _actions_and_frames_for_add()
    out = _xs.add_xshot_occurrence(actions, _declare_unavailable(frames), home_team_id=1)
    assert (out["xshot_occurrence_variant"] == "position_only").all()
    assert out["xshot_occurrence"].notna().any(), "the bundled position_only xShot must serve a value on SB360"


def test_bundled_xcross_serves_via_position_only_on_declared_frames():
    from tests.tracking.test_xcross_attempt_velocity_contract import _freeze_frame

    _clear_variant_caches()
    frames, actions = _freeze_frame(declare=SPEED_SOURCE_UNAVAILABLE, with_velocity=False)
    out = _xc.add_xcross_attempt(actions, frames, home_team_id="A")
    assert (out["xcross_attempt_variant"] == "position_only").all()
    assert out["xcross_attempt"].notna().any(), "the bundled position_only xCross must serve a value on SB360"
