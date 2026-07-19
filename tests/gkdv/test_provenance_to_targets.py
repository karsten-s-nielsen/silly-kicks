"""The provenance -> targets adapter contract (spec §4.6, review S7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.gkdv._engine as engine
from silly_kicks.gkdv import build_ghost_frames, provenance_to_targets
from silly_kicks.tracking._model_eval import _TARGET_COLUMNS, _targets_deltas, _validate_targets
from tests.gkdv._fixtures import in_domain_frames, multi_frame_in_domain
from tests.tracking.test_ghost_gk import _fitted_model


def _prov():
    frames = in_domain_frames()
    _cf, prov, _r = build_ghost_frames(frames, model=_fitted_model()[0], home_team_id=1)
    return frames, prov


# ---------------------------------------------------------------------------
# Cross-package contract
# ---------------------------------------------------------------------------


def test_gkdv_declares_the_SAME_target_contract_the_probe_validates():
    """gkdv consumes tracking PUBLIC seams only, so it DECLARES the contract rather than
    importing the probe's private tuple. This test is what keeps the two from drifting."""
    assert engine._TARGET_COLUMNS == _TARGET_COLUMNS


def test_emits_exactly_the_probe_contract_columns():
    frames, prov = _prov()
    t = provenance_to_targets(prov, frames=frames, home_team_id=1)
    assert set(_TARGET_COLUMNS) <= set(t.columns)
    assert list(t.columns) == list(_TARGET_COLUMNS)


def test_passes_the_shipped_validator():
    frames, prov = _prov()
    t = provenance_to_targets(prov, frames=frames, home_team_id=1)
    assert len(t) > 0, "non-vacuity: _validate_targets accepts an empty frame"
    _validate_targets(t)  # must not raise


def test_target_coords_finite_on_every_row():
    frames, prov = _prov()
    t = provenance_to_targets(prov, frames=frames, home_team_id=1)
    assert len(t) > 0
    assert np.isfinite(t["target_x"]).all() and np.isfinite(t["target_y"]).all()


def test_flags_are_real_non_null_booleans():
    """bool(NaN) is True, which would silently WIDEN the probe's trusted stratum."""
    frames, prov = _prov()
    t = provenance_to_targets(prov, frames=frames, home_team_id=1)
    for col in ("ghost_clamped", "ghost_out_of_box"):
        assert t[col].dtype == bool
        assert not t[col].isna().any()


# ---------------------------------------------------------------------------
# Keying: exactly one row per frame, and it must be the DEFENDING keeper
# ---------------------------------------------------------------------------


def test_exactly_one_row_per_frame_triple():
    frames, prov = _prov()
    t = provenance_to_targets(prov, frames=frames, home_team_id=1)
    assert len(t) > 0
    assert not t.duplicated(subset=["game_id", "period_id", "frame_id"]).any()


def test_selects_the_DEFENDING_keeper_not_both_and_not_the_wrong_one():
    """The dangerous mistake is not only duplication -- it is silently choosing the ATTACKING
    team's keeper, which reads as a valid single row. Pin the identity, not just the count."""
    frames, prov = _prov()
    scored = prov[prov["drop_reason"].isna()]
    assert len(scored) == 2, "premise: the serving seam writes BOTH teams' keepers"

    t = provenance_to_targets(prov, frames=frames, home_team_id=1)
    assert len(t) == 1
    defending = scored[scored["gk_team_id"] == scored["defending_team_id"]]
    attacking = scored[scored["gk_team_id"] != scored["defending_team_id"]]
    assert float(t["target_x"].iloc[0]) == pytest.approx(float(defending["ghost_x"].iloc[0]))
    assert float(t["target_x"].iloc[0]) != pytest.approx(float(attacking["ghost_x"].iloc[0]))


def test_dropped_frames_are_excluded_from_a_MIXED_provenance():
    """`_validate_targets` requires finite coordinates on EVERY row, so a dropped frame in
    the targets frame is a hard error rather than a skipped row.

    The provenance must contain BOTH scored and dropped rows: on an all-scored provenance
    the exclusion is a no-op and the test cannot distinguish a live filter from a removed
    one, and on an all-dropped one it never exercises the surviving rows.
    """
    frames = multi_frame_in_domain(6)  # stride 5 -> 2 scored frames, 4 dropped
    _cf, prov, report = build_ghost_frames(frames, model=_fitted_model()[0], home_team_id=1)
    assert report.n_frames_scored == 2 and int(prov["drop_reason"].notna().sum()) == 4, (
        "premise: the provenance must be MIXED for this test to discriminate"
    )
    t = provenance_to_targets(prov, frames=frames, home_team_id=1)
    assert len(t) == 2
    assert np.isfinite(t["target_x"]).all() and np.isfinite(t["target_y"]).all()
    _validate_targets(t)


def test_a_dropped_row_that_CARRIES_a_keeper_is_still_excluded():
    """The drop filter must be load-bearing on its own, not incidentally covered by the
    defending-keeper selection.

    Today a dropped row has a NaN ``gk_team_id``, so the keeper selection happens to exclude
    it too and removing the drop filter changes nothing. That coincidence is not the
    contract: a future drop reason (``no_ghost_served`` already knows the keeper) could
    record one, and the row would then sail through with NaN coordinates and blow up inside
    the probe. This pins the filter directly.
    """
    frames = multi_frame_in_domain(6)
    _cf, prov, _r = build_ghost_frames(frames, model=_fitted_model()[0], home_team_id=1)
    dropped = prov["drop_reason"].notna()
    assert int(dropped.sum()) == 4, "premise: mixed provenance"
    prov = prov.copy()
    prov.loc[dropped, "gk_team_id"] = prov.loc[dropped, "defending_team_id"]

    t = provenance_to_targets(prov, frames=frames, home_team_id=1)
    assert len(t) == 2, "dropped rows leaked into the targets frame"
    assert np.isfinite(t["target_x"]).all()
    _validate_targets(t)


def test_an_all_dropped_provenance_yields_an_empty_but_VALID_targets_frame():
    frames = in_domain_frames()
    frames.loc[frames["is_ball"].astype(bool), "x"] = 60.0
    frames.loc[frames["player_id"] == "a13", "x"] = 60.2
    _cf, prov, report = build_ghost_frames(frames, model=_fitted_model()[0], home_team_id=1)
    assert report.n_frames_scored == 0 and len(prov) == 1, "premise: this frame is dropped"
    t = provenance_to_targets(prov, frames=frames, home_team_id=1)
    assert len(t) == 0
    _validate_targets(t)


# ---------------------------------------------------------------------------
# Red-first: the naive both-teams pass-through
# ---------------------------------------------------------------------------


def test_naive_both_teams_passthrough_RAISES_in_the_probe():
    """RED-FIRST: the dangerous mistake must fail loudly, not silently pick a keeper.

    NOTE the guard's real home: `_validate_targets` checks columns / finiteness / non-null
    flags ONLY -- the uniqueness check lives in `_targets_deltas`, so that is what this
    exercises. (The plan asserted the raise against `_validate_targets`, which would never
    have fired.)
    """
    frames, prov = _prov()
    scored = prov[prov["drop_reason"].isna()]
    naive = scored.rename(columns={"ghost_x": "target_x", "ghost_y": "target_y"})
    naive = naive[list(_TARGET_COLUMNS)].astype({"ghost_clamped": bool, "ghost_out_of_box": bool})
    assert naive.duplicated(subset=["game_id", "period_id", "frame_id"]).any(), (
        "non-vacuity: the pass-through must actually duplicate the frame here"
    )
    _validate_targets(naive)  # deliberately passes -- uniqueness is NOT its job
    with pytest.raises(ValueError, match="exactly one row per"):
        _targets_deltas(None, frames, arm="xs", targets=naive, n_placebo_replicates=1, seed=0, advance_m=2.0)


def test_the_adapter_raises_if_its_own_selection_ever_fails(monkeypatch):
    """The adapter carries its OWN uniqueness guard so a selection regression fails here
    rather than deep inside the probe. Simulate the naive selection by neutering it."""
    frames, prov = _prov()
    monkeypatch.setattr(engine, "_select_defending_keeper", lambda scored, **_kw: scored)
    with pytest.raises(ValueError, match="Do NOT pass both teams"):
        provenance_to_targets(prov, frames=frames, home_team_id=1)


def test_provenance_without_the_pinned_defending_team_is_rejected():
    """The defending keeper is PINNED by the engine (spec §4.2). Re-deriving it here would
    fork the goal map, so a provenance frame lacking the pin must fail rather than guess."""
    frames, prov = _prov()
    with pytest.raises(ValueError, match="defending_team_id"):
        provenance_to_targets(prov.drop(columns=["defending_team_id"]), frames=frames, home_team_id=1)


# ---------------------------------------------------------------------------
# ADR-019 id safety
# ---------------------------------------------------------------------------


def test_dtype_mismatched_home_team_id_gives_IDENTICAL_output():
    frames, prov = _prov()
    a = provenance_to_targets(prov, frames=frames, home_team_id=1)
    b = provenance_to_targets(prov, frames=frames, home_team_id="1")
    pd.testing.assert_frame_equal(a, b)


def test_home_team_id_from_a_DIFFERENT_match_raises():
    """The same scalar was threaded into the ghost feature extractor. If it matches no team
    in these frames, those features were built against a bogus home team -- fail here."""
    frames, prov = _prov()
    with pytest.raises(ValueError, match="matches no team"):
        provenance_to_targets(prov, frames=frames, home_team_id=99)
