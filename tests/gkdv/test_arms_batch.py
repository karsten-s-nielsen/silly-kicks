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


def _good_frame(fid: int, gk_x: float = 10.0, gk2_x: float = 100.0) -> pd.DataFrame:
    # BOTH teams carry a keeper: native DAS resolves the attacking direction from the GoalMap
    # (ADR-055, keeper geometry), which is UNRESOLVABLE with only one keeper. Team-1 keeper near
    # x=0, team-2 keeper near x=105 -> team 1 defends 0, team 2 defends 105, so team 2 (in
    # possession) attacks 0. `gk_x`/`gk2_x` let a test relocate either keeper.
    rows = [
        dict(player_id="gk1", team_id="1", is_ball=False, is_goalkeeper=True, x=gk_x, y=34.0, vx=0.0, vy=0.0),
        dict(player_id="d1", team_id="1", is_ball=False, is_goalkeeper=False, x=20.0, y=30.0, vx=0.3, vy=0.1),
        dict(player_id="gk2", team_id="2", is_ball=False, is_goalkeeper=True, x=gk2_x, y=34.0, vx=0.0, vy=0.0),
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
    gm = _das_port.pin_direction(unit)
    out = _das_port.team_das_by_frame(unit, "2", goal_map=gm)

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
    gm = _das_port.pin_direction(unit)

    # Complete Series: OK.
    att = pd.Series({(1, 1, 1): "2", (1, 1, 2): "2"})
    att.index.names = _KEY
    out = _das_port.team_das_by_frame(unit, att, goal_map=gm)
    assert len(out) == 2

    # Missing key for frame 2: fail-loud, NOT a silent NaN.
    partial = pd.Series({(1, 1, 1): "2"})
    partial.index.names = _KEY
    with pytest.raises((KeyError, ValueError)):
        _das_port.team_das_by_frame(unit, partial, goal_map=gm)


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
    gm = _das_port.pin_direction(one)

    out = _das_port.team_das_by_frame(one, "2", goal_map=gm)
    assert out.notna().all() and (out > 0.0).all(), (
        "attacking-team DAS must survive a non-contiguous index (positional mask, not label-aligned)"
    )


# ---------------------------------------------------------------------------
# Task 3 -- delta_das_batch
# ---------------------------------------------------------------------------


def _looped_reference(actual, ghost, *, attacking_team_id, goal_map):
    """Amortization reference: the SAME once-per-unit GoalMap, but get_individual_das called
    PER FRAME. Isolates batching (paired batch vs loop of identical math) from direction."""
    from silly_kicks.gkdv import _das_port

    out = {}
    for (ka, a_sub), (kg, g_sub) in zip(actual.groupby(_KEY), ghost.groupby(_KEY), strict=True):
        assert ka == kg
        a_das = _das_port.team_das(a_sub, attacking_team_id=attacking_team_id, goal_map=goal_map)
        g_das = _das_port.team_das(g_sub, attacking_team_id=attacking_team_id, goal_map=goal_map)
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

    goal_map = _das_port.pin_direction(actual)
    ref = _looped_reference(actual, ghost, attacking_team_id="2", goal_map=goal_map)
    got = delta_das_batch(actual, ghost, attacking_team_id_by_frame="2")

    # Bit-exact: the paired seam is a two-call amortization (F2), and per-frame DAS is a
    # snapshot (chunk-invariant), so the paired batch equals the per-frame loop exactly.
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
    """Spec §5.2 CONSEQUENCE: resolving the GoalMap ONCE over the unit (what delta_das_batch does,
    on the FULL factual stack) gives the flip frame the majority's attacked goal, whereas resolving
    that frame ALONE flips it. `pin_direction` returns a GoalMap via resolve_defended_goals (keeper
    geometry, ADR-055), so this runs library-free.

    3 normal frames (team-1 keeper deep at x=10, team-2 keeper deep at x=100) + 1 flip frame with
    the keepers SWAPPED to opposite ends. Over the unit the majority keeps team 1 defending x=0 (so
    team 2 attacks x=0); the flip frame ALONE has team 1 defending x=105 (so team 2 attacks x=105)."""
    from silly_kicks.gkdv import _das_port

    flip = _good_frame(4, gk_x=100.0, gk2_x=5.0)  # keepers swapped to the opposite ends
    unit = pd.concat([_good_frame(1), _good_frame(2), _good_frame(3), flip], ignore_index=True)

    att_unit = _das_port.pin_direction(unit).attacked_goal(1, 1, "2", allow_guess=True)
    att_alone = _das_port.pin_direction(flip).attacked_goal(1, 1, "2", allow_guess=True)

    assert att_unit == 0.0, "team 2 attacks x=0 under the majority (once-per-unit) pin"
    assert att_alone == 105.0, "resolving the flip frame ALONE flips team 2 to attack x=105"
    assert att_unit != att_alone, "resolving the flip frame alone flips it -- what the once-per-unit pin prevents"


def test_delta_das_batch_pins_ONE_direction_over_the_unit(monkeypatch):
    """MECHANISM: delta_das_batch resolves the GoalMap ONCE, on the FULL factual stack, and threads
    that SAME map into the paired seam that scores both legs. STRUCTURAL -- stubs pin_direction AND
    paired_team_das_by_frame, so it runs on every leg with no accessible-space."""
    import silly_kicks.gkdv._das_port as _das_port  # patch the module directly (delta_das_batch imports it locally)
    from silly_kicks.gkdv import delta_das_batch

    sentinel = object()  # a stand-in GoalMap: identity is what the assertions track
    seen = {"pin_frames": [], "paired": []}

    def spy_pin(frames):
        seen["pin_frames"].append(frames.copy())
        return sentinel

    def spy_paired(actual, counterfactual, attacking_team_id_by_frame, *, goal_map):
        seen["paired"].append({"n_actual": len(actual), "n_ghost": len(counterfactual), "goal_map": goal_map})
        ka = pd.MultiIndex.from_frame(actual[_KEY].drop_duplicates())
        kg = pd.MultiIndex.from_frame(counterfactual[_KEY].drop_duplicates())
        return pd.Series(1.0, index=ka), pd.Series(0.5, index=kg)

    monkeypatch.setattr(_das_port, "pin_direction", spy_pin)
    monkeypatch.setattr(_das_port, "paired_team_das_by_frame", spy_paired)

    actual = _unit(3)
    ghost = _unit(3)
    delta_das_batch(actual, ghost, attacking_team_id_by_frame="2")

    assert len(seen["pin_frames"]) == 1, "pin_direction must be called exactly ONCE"
    assert len(seen["pin_frames"][0]) == len(actual), "pin_direction must see the FULL factual stack"
    assert len(seen["paired"]) == 1, "the paired seam scores both legs in exactly ONE call"
    assert seen["paired"][0]["goal_map"] is sentinel, "the ONE pinned GoalMap must feed the paired seam"
    assert seen["paired"][0]["n_actual"] == len(actual) and seen["paired"][0]["n_ghost"] == len(ghost)


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


def test_delta_das_batch_scores_both_legs_in_ONE_paired_pass_regardless_of_frame_count():
    """The amortization, proven structurally (no wall-clock): both legs are scored in exactly ONE
    paired DAS pass (individual_das_paired handles factual + ghost together) whether the unit has
    2 frames or 20."""
    import unittest.mock as mock

    import silly_kicks.gkdv._das_port as _das_port
    from silly_kicks.gkdv import delta_das_batch

    calls = {"n": 0}

    def counting_paired(actual, counterfactual, attacking_team_id_by_frame, *, goal_map):
        calls["n"] += 1
        ka = pd.MultiIndex.from_frame(actual[_KEY].drop_duplicates())
        kg = pd.MultiIndex.from_frame(counterfactual[_KEY].drop_duplicates())
        return pd.Series(1.0, index=ka), pd.Series(0.0, index=kg)

    with (
        mock.patch.object(_das_port, "paired_team_das_by_frame", counting_paired),
        mock.patch.object(_das_port, "pin_direction", lambda f: object()),
    ):
        for n in (2, 20):
            calls["n"] = 0
            delta_das_batch(_unit(n), _unit(n), attacking_team_id_by_frame="2")
            assert calls["n"] == 1, f"expected 1 paired pass (both legs) for n={n}, got {calls['n']}"
