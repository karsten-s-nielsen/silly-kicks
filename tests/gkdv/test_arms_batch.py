"""Tests for the batched GKDV arms (``delta_das_batch`` / ``delta_threat_suppression_batch``).

Plan: docs/superpowers/plans/2026-08-27-gkdv-arms-batching.md. Real-scoring tests
``importorskip`` accessible-space; the structural tests (mechanism, call-count) stub the
``_das_port`` seam and run on every leg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

_KEY = ["game_id", "period_id", "frame_id"]
_CARRIER = "ball_carrier_player_id"

#: The ghost keeper, displaced far enough to flip an UNPINNED single-frame direction inference.
_GHOST_GK_X = 100.0


def _good_frame(fid: int, gk_x: float = 10.0) -> pd.DataFrame:
    rows = [
        dict(player_id="gk1", team_id="1", is_ball=False, is_goalkeeper=True, x=gk_x, y=34.0, vx=0.0, vy=0.0),
        dict(player_id="d1", team_id="1", is_ball=False, is_goalkeeper=False, x=20.0, y=30.0, vx=0.3, vy=0.1),
        dict(player_id="a1", team_id="2", is_ball=False, is_goalkeeper=False, x=30.0, y=34.0, vx=1.0, vy=0.0),
        dict(player_id="a2", team_id="2", is_ball=False, is_goalkeeper=False, x=40.0, y=38.0, vx=1.0, vy=0.2),
        dict(player_id="ball", team_id=None, is_ball=True, is_goalkeeper=False, x=40.0, y=34.0, vx=0.0, vy=0.0),
    ]
    for r in rows:
        r.update(game_id=1, period_id=1, frame_id=fid, team_in_possession="2")
    df = pd.DataFrame(rows)
    df[_CARRIER] = pd.Series(["a2"] * len(df), dtype="string", index=df.index)
    return df


def _unit(n: int) -> pd.DataFrame:
    return pd.concat([_good_frame(fid) for fid in range(1, n + 1)], ignore_index=True)


# ---------------------------------------------------------------------------
# Task 2 -- the batch reduce seam
# ---------------------------------------------------------------------------


def test_team_das_by_frame_reduces_per_frame_over_attacking_team():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import _das_port

    unit = _unit(3)
    unit["attacking_direction"] = _das_port.pin_direction(unit).to_numpy()
    out = _das_port.team_das_by_frame(unit, "2", direction_col="attacking_direction")

    assert isinstance(out, pd.Series)
    assert list(out.index.names) == _KEY
    assert len(out) == 3
    # NON-VACUITY (round-4 defect 2): every frame is scoreable, so the reduce must be all-finite.
    # Without this, an all-NaN result (e.g. a tuple-dtype miss in the per-row `MultiIndex.map`)
    # makes `out.dropna()` empty and `(empty > 0).all()` trivially True -- a guard that cannot fail.
    assert out.notna().all(), "every scored frame must reduce to a finite DAS (not silently all-NaN)"
    assert (out > 0.0).all(), "attacking team has positive dangerous space on every scored frame"


def test_team_das_by_frame_series_is_looked_up_per_frame_and_missing_key_raises():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import _das_port

    unit = _unit(2)
    unit["attacking_direction"] = _das_port.pin_direction(unit).to_numpy()

    # Complete Series: OK.
    att = pd.Series({(1, 1, 1): "2", (1, 1, 2): "2"})
    att.index.names = _KEY
    out = _das_port.team_das_by_frame(unit, att, direction_col="attacking_direction")
    assert len(out) == 2

    # Missing key for frame 2: fail-loud, NOT a silent NaN.
    partial = pd.Series({(1, 1, 1): "2"})
    partial.index.names = _KEY
    with pytest.raises((KeyError, ValueError)):
        _das_port.team_das_by_frame(unit, partial, direction_col="attacking_direction")


def test_team_das_by_frame_survives_a_noncontiguous_index():
    """Regression (SB360 Leg B): a filtered frame slice carries a NON-CONTIGUOUS index. ``ids_equal``
    returns a POSITIONAL fresh-RangeIndex mask (ADR-019), so combining it with ``~is_ball`` by LABEL
    silently dropped every attacking player -> an all-NaN reduce. The Task-2 tests above used a
    contiguous ``pd.concat(ignore_index=True)`` index and could not see it."""
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import _das_port

    two = _unit(2)  # contiguous 0..N-1
    one = two[two["frame_id"] == 2].copy()  # FILTERED -> non-contiguous index (the trigger)
    assert not one.index.equals(pd.RangeIndex(len(one))), "fixture must present a non-contiguous index"
    one["attacking_direction"] = _das_port.pin_direction(one).to_numpy()

    out = _das_port.team_das_by_frame(one, "2", direction_col="attacking_direction")
    assert out.notna().all() and (out > 0.0).all(), (
        "attacking-team DAS must survive a non-contiguous index (positional mask, not label-aligned)"
    )


# ---------------------------------------------------------------------------
# Task 3 -- delta_das_batch
# ---------------------------------------------------------------------------


def _looped_reference(actual, ghost, *, attacking_team_id, direction):
    """Amortization reference: the SAME once-per-unit direction, but get_individual_das called
    PER FRAME. Isolates batching (batch vs loop of identical math) from the direction change."""
    from silly_kicks.gkdv import _das_port

    a = actual.copy()
    a["attacking_direction"] = direction.to_numpy()
    g = ghost.copy()
    g["attacking_direction"] = direction.to_numpy()
    out = {}
    for (ka, a_sub), (kg, g_sub) in zip(a.groupby(_KEY), g.groupby(_KEY), strict=True):
        assert ka == kg
        a_das = _das_port.team_das(a_sub, attacking_team_id=attacking_team_id, direction_col="attacking_direction")
        g_das = _das_port.team_das(g_sub, attacking_team_id=attacking_team_id, direction_col="attacking_direction")
        out[ka] = a_das - g_das
    s = pd.Series(out)
    s.index.names = _KEY
    return s


def test_delta_das_batch_is_bit_exact_amortization_of_the_per_frame_loop():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import _das_port, delta_das_batch

    actual = _unit(4)
    ghost = _unit(4)  # scoreable both legs; the ORACLE tests amortization, not deterrence
    ghost.loc[ghost["player_id"] == "gk1", "x"] = 12.0  # a small keeper move so legs are not identical

    direction = _das_port.pin_direction(actual)
    ref = _looped_reference(actual, ghost, attacking_team_id="2", direction=direction)
    got = delta_das_batch(actual, ghost, attacking_team_id_by_frame="2")

    # Bit-exact if accessible-space's per-frame result is call-count-invariant; else pin a
    # measured, version-noted atol here and document it. accessible_space==2.0.15.
    pd.testing.assert_series_equal(got, ref, check_names=False, rtol=0, atol=0)


def test_delta_das_batch_nans_unscoreable_frame_not_zero():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import delta_das_batch

    good_a, good_g = _good_frame(1), _good_frame(1, gk_x=12.0)
    bad_a = _good_frame(2)[lambda d: ~d["is_ball"].astype(bool)].reset_index(drop=True)  # no ball
    bad_g = bad_a.copy()
    actual = pd.concat([good_a, bad_a], ignore_index=True)
    ghost = pd.concat([good_g, bad_g], ignore_index=True)

    out = delta_das_batch(actual, ghost, attacking_team_id_by_frame="2")
    assert np.isfinite(out.loc[(1, 1, 1)]), "the scoreable frame must have a finite delta"
    assert pd.isna(out.loc[(1, 1, 2)]), "the unscoreable frame must be NaN, never a fabricated 0.0"


def test_delta_das_batch_raises_on_misaligned_legs():
    from silly_kicks.gkdv import delta_das_batch

    actual = _unit(2)
    ghost = _unit(2).iloc[::-1].reset_index(drop=True)  # reversed row order, same index 0..n-1
    with pytest.raises(ValueError, match="align"):
        delta_das_batch(actual, ghost, attacking_team_id_by_frame="2")


def test_delta_das_batch_whole_batch_unscoreable_returns_all_nan_over_keys():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import delta_das_batch

    actual = _unit(2).copy()
    actual["team_in_possession"] = pd.NA  # dead-ball whole batch
    ghost = actual.copy()
    out = delta_das_batch(actual, ghost, attacking_team_id_by_frame="2")
    assert len(out) == 2 and out.isna().all(), "a wholly-unscoreable unit is all-NaN over its frame keys"


def test_once_per_unit_pin_is_stable_where_a_single_frame_would_flip():
    """Spec §5.2 CONSEQUENCE: pinning direction ONCE over the unit gives the flip frame the SAME
    (stable) direction as the majority, whereas pinning that frame ALONE (the OLD per-frame
    behaviour) flips it. `pin_direction` uses accessible-space's `infer_playing_direction`, so this
    is a real-scoring test (skipped without [das]). The LOAD-BEARING assertion is
    `d_flip_unit != d_flip_alone`; `d_flip_unit == d_normal` is a near-by-construction sanity check
    (infer_playing_direction is constant per (period, team_in_possession))."""
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import _das_port

    # 3 frames with team-1 keeper low (x=10) + 1 flip frame with it at _GHOST_GK_X (=100). The unit
    # mean keeps team-1 the argmin (26.25 < team-2's 35), so the flip frame stays stable under the
    # per-unit pin; ALONE it crosses team-2's mean and flips.
    unit = pd.concat(
        [_good_frame(1), _good_frame(2), _good_frame(3), _good_frame(4, gk_x=_GHOST_GK_X)],
        ignore_index=True,
    )
    per_unit = _das_port.pin_direction(unit)
    d_flip_unit = per_unit[unit["frame_id"] == 4].iloc[0]
    d_normal = per_unit[unit["frame_id"] == 1].iloc[0]
    d_flip_alone = _das_port.pin_direction(_good_frame(4, gk_x=_GHOST_GK_X)).iloc[0]

    assert d_flip_unit == d_normal, "once-per-unit pin is stable on the flip frame (majority-dominated)"
    assert d_flip_unit != d_flip_alone, "the OLD per-frame pin flips this frame -- exactly what the batch changes"


def test_delta_das_batch_pins_ONE_direction_over_the_unit(monkeypatch):
    """MECHANISM: delta_das_batch calls pin_direction ONCE, on the FULL factual stack, feeding both
    legs. STRUCTURAL -- stubs pin_direction (synthetic) AND team_das_by_frame, so it runs on every
    leg with no accessible-space (round-4 defect 1 sibling)."""
    import silly_kicks.gkdv._das_port as _das_port  # patch the module directly (delta_das_batch imports it locally)
    from silly_kicks.gkdv import delta_das_batch

    seen = {"pin_frames": []}

    def spy_pin(frames):
        # SYNTHETIC direction -- do NOT call the real pin_direction (it uses accessible-space).
        seen["pin_frames"].append(frames.copy())
        return pd.Series(1.0, index=frames.index)

    def stub_team_das_by_frame(frames, attacking_team_id_by_frame, *, direction_col):
        s = frames.groupby(_KEY)[direction_col].mean()
        s.index.names = _KEY
        return s

    monkeypatch.setattr(_das_port, "pin_direction", spy_pin)
    monkeypatch.setattr(_das_port, "team_das_by_frame", stub_team_das_by_frame)

    actual = _unit(3)
    ghost = _unit(3)
    delta_das_batch(actual, ghost, attacking_team_id_by_frame="2")

    assert len(seen["pin_frames"]) == 1, "pin_direction must be called exactly ONCE"
    assert len(seen["pin_frames"][0]) == len(actual), "pin_direction must see the FULL factual stack"


# ---------------------------------------------------------------------------
# Task 4 -- delta_threat_suppression_batch
# ---------------------------------------------------------------------------


def _threat_unit(n: int) -> pd.DataFrame:
    """A threat-scoreable factual unit: the working single-frame threat fixture
    (`test_compute_threat_pc._frame`, LTR-normalized, GK per team) stacked to n frame_ids."""
    from tests.tracking.test_compute_threat_pc import GK_ON_LINE, _frame

    parts = []
    for fid in range(1, n + 1):
        f = _frame(GK_ON_LINE).copy()
        f["frame_id"] = fid
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def test_delta_threat_suppression_batch_equals_looping_the_single_frame_arm():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import delta_threat_suppression, delta_threat_suppression_batch
    from tests.tracking.test_compute_threat_pc import HOME_GOAL_MAP, _fitted_xt

    actual = _threat_unit(3)
    ghost = actual.copy()
    ghost.loc[ghost["is_goalkeeper"].astype(bool), "x"] += 2.0
    xt = _fitted_xt()
    goal_map = HOME_GOAL_MAP

    batched = delta_threat_suppression_batch(actual, ghost, attacking_team_id_by_frame=2, xt=xt, goal_map=goal_map)
    batched_by_key = batched.to_dict()
    for (k, a_sub), (_, g_sub) in zip(actual.groupby(_KEY), ghost.groupby(_KEY), strict=True):
        one = delta_threat_suppression(a_sub, g_sub, attacking_team_id=2, xt=xt, goal_map=goal_map)
        assert batched_by_key[k] == pytest.approx(one, rel=0, abs=0), f"frame {k} batched != looped"


def test_delta_threat_suppression_batch_scores_a_dead_ball_unit_without_crashing():
    """Round-3 finding 3: the threat arm is possession-INDEPENDENT (compute_threat_pc takes
    attacking_team_id explicitly and reads NO team_in_possession), so a dead-ball unit SCORES
    rather than raising -- the inherent, correct asymmetry with the DAS arm's
    DasUnscoreableError->NaN."""
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import delta_threat_suppression_batch
    from tests.tracking.test_compute_threat_pc import HOME_GOAL_MAP, _fitted_xt

    actual = _threat_unit(3)
    ghost = actual.copy()
    ghost.loc[ghost["is_goalkeeper"].astype(bool), "x"] += 2.0
    actual["team_in_possession"] = pd.NA  # dead ball everywhere
    ghost["team_in_possession"] = pd.NA

    out = delta_threat_suppression_batch(
        actual, ghost, attacking_team_id_by_frame=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP
    )
    assert out.notna().all(), "the threat arm scores a dead-ball unit (no DasUnscoreableError equivalent)"


# ---------------------------------------------------------------------------
# Task 5 -- single-frame wrappers delegate to the batch
# ---------------------------------------------------------------------------


def test_single_frame_delta_das_equals_one_frame_batch():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import delta_das, delta_das_batch

    actual, ghost = _good_frame(1), _good_frame(1, gk_x=100.0)
    scalar = delta_das(actual, ghost, attacking_team_id="2")
    batched = delta_das_batch(actual, ghost, attacking_team_id_by_frame="2")
    assert batched.iloc[0] == pytest.approx(scalar, rel=0, abs=0)
    assert np.isfinite(scalar) and scalar != 0.0


# ---------------------------------------------------------------------------
# Task 6 -- purity + structural call-count
# ---------------------------------------------------------------------------


def test_delta_das_batch_does_not_mutate_inputs():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import delta_das_batch

    actual, ghost = _unit(2), _unit(2)
    ghost.loc[ghost["player_id"] == "gk1", "x"] = 12.0
    a_before, g_before = actual.copy(), ghost.copy()
    delta_das_batch(actual, ghost, attacking_team_id_by_frame="2")
    pd.testing.assert_frame_equal(actual, a_before)
    pd.testing.assert_frame_equal(ghost, g_before)


def test_delta_das_batch_calls_accessible_space_once_per_leg_regardless_of_frame_count():
    """The amortization, proven structurally (no wall-clock): 2 legs -> exactly 2 reduce calls
    (each = one get_individual_das) whether the unit has 2 frames or 20."""
    import unittest.mock as mock

    import silly_kicks.gkdv._das_port as _das_port
    from silly_kicks.gkdv import delta_das_batch

    calls = {"n": 0}

    def counting_team_das_by_frame(frames, attacking_team_id_by_frame, *, direction_col):
        calls["n"] += 1
        s = frames.groupby(_KEY)[direction_col].mean()
        s.index.names = _KEY
        return s

    with (
        mock.patch.object(_das_port, "team_das_by_frame", counting_team_das_by_frame),
        mock.patch.object(_das_port, "pin_direction", lambda f: pd.Series(1.0, index=f.index)),
    ):
        for n in (2, 20):
            calls["n"] = 0
            delta_das_batch(_unit(n), _unit(n), attacking_team_id_by_frame="2")
            assert calls["n"] == 2, f"expected 2 reduce calls (one per leg) for n={n}, got {calls['n']}"
