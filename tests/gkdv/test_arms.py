"""Physics arms: polarity, silent-zero guards, method pin, direction pin (spec §5).

FIXTURE PROVENANCE. The threat-arm frames come from
``tests/tracking/test_compute_threat_pc.py``, which owns the keeper-sensitive layout and
documents why the geometry is load-bearing (on a naive layout the threat integral does not
move AT ALL when the defending keeper moves, so every polarity assertion would pass
vacuously). The plan pointed at ``tests.tracking.test_cover_shadows._frame`` / ``_fitted_xt``;
neither exists on this tree.

The DAS fixtures are local: the plan pointed at ``tests.tracking.test_das._frames``, which
is a **method on a test class**, not an importable module-level helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.gkdv import GkdvParams, delta_das, delta_threat_suppression
from tests.tracking.test_compute_threat_pc import (
    GK_ON_LINE,
    GK_OUT_OF_POSITION,
    _fitted_xt,
    _frame,
)

# ---------------------------------------------------------------------------
# Delta-GK-threat-suppression
# ---------------------------------------------------------------------------


def test_deterrent_keeper_gives_a_NEGATIVE_delta():
    """Attacker-value units: actual - ghost. A deterrent actual keeper suppresses threat.

    POLARITY NOTE (a plan assumption corrected by measurement). The plan planted the ghost
    10 m DEEPER than the actual keeper and called that "less suppression". It is the
    opposite: the attacked goal is the high-xT end, so a deeper keeper covers MORE threat,
    and that fixture yields a POSITIVE delta. What actually makes a keeper deterrent here is
    covering its own goalmouth rather than being stranded upfield, so that is what is
    planted -- measured on this tree at roughly -6 threat units, stable for any ghost
    x >= 13 m.
    """
    delta = delta_threat_suppression(
        _frame(GK_ON_LINE),
        _frame(GK_OUT_OF_POSITION),
        attacking_team_id=2,
        xt=_fitted_xt(),
        home_team_id=1,
    )
    assert delta < 0, "a better-positioned actual keeper must score negative (= deterrent)"


def test_polarity_is_antisymmetric_in_the_two_legs():
    """Swapping the legs must flip the sign -- proves the sign tracks the ARGUMENTS.

    Without this, a hard-coded ``-abs(...)`` would satisfy the polarity test above.
    """
    kwargs = {"attacking_team_id": 2, "xt": _fitted_xt(), "home_team_id": 1}
    forward = delta_threat_suppression(_frame(GK_ON_LINE), _frame(GK_OUT_OF_POSITION), **kwargs)
    reverse = delta_threat_suppression(_frame(GK_OUT_OF_POSITION), _frame(GK_ON_LINE), **kwargs)
    assert forward == pytest.approx(-reverse)
    assert forward != 0.0, "vacuous: the fixture produced no threat difference at all"


def test_identical_frames_give_exactly_zero():
    frame = _frame()
    delta = delta_threat_suppression(frame, frame.copy(), attacking_team_id=2, xt=_fitted_xt(), home_team_id=1)
    assert delta == 0.0


def test_threat_arm_pins_the_configured_method_into_BOTH_legs(monkeypatch):
    """``lambda_gk`` lives ONLY on SpearmanParams, so a leg that silently used another
    method would measure nothing about the keeper. Asserted on the CALLS, so it detects a
    dropped or divergent ``method=`` kwarg rather than trusting the default."""
    import silly_kicks.tracking as tracking

    seen: list[dict] = []

    def _spy(frame, **kwargs):
        seen.append(kwargs)
        return float(len(seen))

    monkeypatch.setattr(tracking, "compute_threat_pc", _spy)
    delta_threat_suppression(
        _frame(GK_ON_LINE), _frame(GK_OUT_OF_POSITION), attacking_team_id=2, xt=None, home_team_id=1
    )

    assert len(seen) == 2, f"expected exactly two threat legs, saw {len(seen)}"
    assert [call.get("method") for call in seen] == ["spearman", "spearman"]
    assert seen[0]["attacking_team_id"] == seen[1]["attacking_team_id"] == 2
    assert seen[0]["home_team_id"] == seen[1]["home_team_id"] == 1


def test_threat_arm_FORWARDS_lambda_gk_into_the_pitch_control_params(monkeypatch):
    """``GkdvParams.lambda_gk`` must REACH the surface, not merely be documented.

    ``lambda_gk`` exists only on ``SpearmanParams`` and is the single term through which this
    arm sees the keeper at all -- it is what ``pitch_control_method``'s construction guard is
    defending. Left unforwarded, ``compute_threat_pc`` falls back to the pitch-control default
    and a caller raising ``lambda_gk`` silently gets the default gain, while ``GkdvReport``
    echoes the raised value as though it had been used. Asserted on the CALLS (both legs),
    because the default happens to equal ``SpearmanParams.lambda_gk`` -- so an
    output-comparison test would pass on the unforwarded code.
    """
    import silly_kicks.tracking as tracking

    seen: list[dict] = []

    def _spy(frame, **kwargs):
        seen.append(kwargs)
        return float(len(seen))

    monkeypatch.setattr(tracking, "compute_threat_pc", _spy)
    delta_threat_suppression(
        _frame(GK_ON_LINE),
        _frame(GK_OUT_OF_POSITION),
        attacking_team_id=2,
        xt=None,
        home_team_id=1,
        params=GkdvParams(lambda_gk=7.5),
    )

    assert len(seen) == 2, f"expected exactly two threat legs, saw {len(seen)}"
    for call in seen:
        assert call.get("params") is not None, (
            "the arm passed no pitch-control params, so lambda_gk never reached the surface"
        )
        assert call["params"].lambda_gk == 7.5
    assert seen[0]["params"] == seen[1]["params"], "the two legs must share one parameterization"

    # Non-vacuity: the assertion must be able to FAIL. 7.5 is not the default it would have
    # silently fallen back to, so seeing 7.5 is evidence of forwarding, not of the default.
    assert GkdvParams().lambda_gk != 7.5


def test_threat_arm_is_id_dtype_safe(monkeypatch):
    """ADR-019: a value-equal scalar of a different dtype must give an IDENTICAL result.

    ``home_team_id`` is a caller-supplied scalar of uncontrolled dtype, and a raw ``==``
    against an id COLUMN is the single most damaging bug shape in this codebase (measured
    live: an object-string ``team_id`` vs an int scalar makes ``!=`` True for EVERY row).
    """
    # ONE fitted model, passed by KEYWORD to both legs. Deliberately not splatted from a
    # dict: a `**kwargs` of uniform value type lets the checker match `ExpectedThreat`
    # against every remaining keyword-only parameter -- including `params: GkdvParams` --
    # so the splat both hides the real target and manufactures a type error.
    xt = _fitted_xt()
    numeric = delta_threat_suppression(
        _frame(GK_ON_LINE), _frame(GK_OUT_OF_POSITION), attacking_team_id=2, home_team_id=1, xt=xt
    )
    stringy = delta_threat_suppression(
        _frame(GK_ON_LINE), _frame(GK_OUT_OF_POSITION), attacking_team_id="2", home_team_id="1", xt=xt
    )
    assert numeric == stringy
    assert numeric != 0.0, "vacuous: both dtypes agreed only because the value is zero"


def test_threat_arm_does_not_mutate_its_inputs():
    actual, ghost = _frame(GK_ON_LINE), _frame(GK_OUT_OF_POSITION)
    before_actual, before_ghost = actual.copy(deep=True), ghost.copy(deep=True)
    delta_threat_suppression(actual, ghost, attacking_team_id=2, xt=_fitted_xt(), home_team_id=1)
    pd.testing.assert_frame_equal(actual, before_actual)
    pd.testing.assert_frame_equal(ghost, before_ghost)


def test_non_spearman_method_is_rejected_AT_CONSTRUCTION():
    """A GK-blind method must be unrepresentable, not merely rejected at call time.

    ``lambda_gk`` exists ONLY on ``SpearmanParams``, so any other method silently yields an
    arm that measures nothing about the keeper. ``GkdvParams.__post_init__`` raises, so the
    bad configuration cannot be built and then passed around -- which is why the arm itself
    deliberately carries NO duplicate check that could drift from that allowlist.
    """
    with pytest.raises(ValueError, match="GK-BLIND"):
        GkdvParams(pitch_control_method="voronoi")
    with pytest.raises(ValueError, match="GK-BLIND"):
        GkdvParams(pitch_control_method="fernandez_bornn")


@pytest.mark.parametrize("arm", [delta_threat_suppression, delta_das], ids=lambda f: f.__name__)
def test_arms_refuse_a_pitch_control_cache(arm):
    """The cache key excludes player positions -> a shared cache would silently return
    Delta == 0, because the ghost frame carries the same frame identity as its twin."""
    import inspect

    assert "pitch_control_cache" not in inspect.signature(arm).parameters


# ---------------------------------------------------------------------------
# Delta-DAS
# ---------------------------------------------------------------------------


def _port_frames() -> pd.DataFrame:
    """Minimal frames for the STUBBED path. Content is irrelevant -- the port is stubbed --
    so this is deliberately local rather than imported from a sibling test module."""
    return pd.DataFrame(
        {
            "game_id": ["100"] * 3,
            "period_id": [1] * 3,
            "frame_id": [1] * 3,
            "player_id": ["p1", "p2", None],
            "team_id": [1, 2, None],
            "x": [50.0, 55.0, 52.0],
            "y": [34.0, 30.0, 34.0],
            "is_ball": [False, False, True],
            "is_goalkeeper": [True, False, False],
        }
    )


def _das_frames(gk_x: float = 10.0) -> pd.DataFrame:
    """A DELIBERATELY EXTREME 2-v-2 roster for the direction-inference discriminator.

    accessible-space infers direction from ``groupby(team)[x].mean().idxmin()``. A 4 m
    keeper move shifts an 11-player mean by only ~0.36 m, so a realistic roster can never
    flip the argmin and could not discriminate a pinned implementation from an unpinned one.
    Here team 1 has two players, so relocating its keeper from x=10 to x=100 moves the team
    mean from 15 to 60 and crosses team 2's mean of 35 -- the argmin genuinely flips.
    """
    rows = [
        dict(player_id="gk1", team_id="1", is_ball=False, is_goalkeeper=True, x=gk_x, y=34.0, vx=0.0, vy=0.0),
        dict(player_id="d1", team_id="1", is_ball=False, is_goalkeeper=False, x=20.0, y=30.0, vx=0.0, vy=0.0),
        dict(player_id="a1", team_id="2", is_ball=False, is_goalkeeper=False, x=30.0, y=34.0, vx=1.0, vy=0.0),
        dict(player_id="a2", team_id="2", is_ball=False, is_goalkeeper=False, x=40.0, y=38.0, vx=1.0, vy=0.0),
        dict(player_id="ball", team_id=None, is_ball=True, is_goalkeeper=False, x=40.0, y=34.0, vx=0.0, vy=0.0),
    ]
    for row in rows:
        row.update(game_id=1, period_id=1, frame_id=1, team_in_possession="2")
    return pd.DataFrame(rows)


#: The ghost keeper, displaced far enough to flip an UNPINNED direction inference.
_GHOST_GK_X = 100.0


def test_das_arm_passes_ONE_pinned_direction_to_BOTH_legs(monkeypatch):
    """STRUCTURAL primary -- and it runs on EVERY CI leg, with NO accessible-space.

    Every asserted fact here is about the CALLS, not the returns, so the stub returns a
    synthetic scalar instead of delegating to the real library. Delegation was the ONLY
    reason this guard would have needed ``importorskip``, and a guard for a declared live
    hazard that skips everywhere is not a guard.

    The property -- "one direction, computed on the FACTUAL frames, passed identically to
    both legs" -- is gkdv's own code, so testing it must not require the optional extra.
    Detects a revert to ``get_das(infer_attacking_direction=True)``.
    """
    import silly_kicks.gkdv._das_port as port

    # The stub's return is DERIVED FROM THE FRAMES IT IS GIVEN, and every call is recorded.
    # A constant stub cannot see two plausible reverts: pinning on the GHOST leg instead of
    # the factual one, and pinning per-leg (both of which return the same constant vector
    # and would sail through an equality assertion). Keyed on mean x, the ghost's 6 m
    # displacement makes those two implementations produce visibly different vectors.
    pin_calls: list[pd.DataFrame] = []

    def _stub_pin_direction(frames):
        pin_calls.append(frames)
        return pd.Series([round(float(frames["x"].mean()), 6)] * len(frames))

    monkeypatch.setattr(port, "pin_direction", _stub_pin_direction)

    calls: list[dict] = []

    def _stub_team_das(frames, *, attacking_team_id, direction_col):
        calls.append(
            {
                "col": direction_col,
                "values": tuple(frames[direction_col]) if direction_col in frames else None,
            }
        )
        return float(len(calls))  # synthetic -- NO delegation, NO library

    monkeypatch.setattr(port, "team_das", _stub_team_das)

    actual = _port_frames()
    ghost = actual.copy()
    gk = ghost["is_goalkeeper"].astype(bool) & ~ghost["is_ball"].astype(bool)
    ghost.loc[gk, "x"] = ghost.loc[gk, "x"] - 6.0
    delta_das(actual, ghost, attacking_team_id=2)

    assert len(calls) == 2, f"expected exactly two DAS legs, saw {len(calls)}"
    assert all(call["col"] == "attacking_direction" for call in calls), (
        "a leg ran WITHOUT a pinned direction column -- accessible-space would re-infer "
        "direction from team mean-x, which the ghost displacement perturbs"
    )
    assert calls[0]["values"] == calls[1]["values"], (
        "the two legs used DIFFERENT direction vectors -- the delta is not a counterfactual"
    )

    # ONE inference, and it must be the FACTUAL one. Both are separately mutable:
    # pinning per-leg calls twice; pinning on the ghost calls once with the wrong frames.
    assert len(pin_calls) == 1, (
        f"direction was inferred {len(pin_calls)} times -- it must be pinned ONCE, on the factual frames, and reused"
    )
    expected = round(float(actual["x"].mean()), 6)
    assert pin_calls[0]["x"].mean() == pytest.approx(actual["x"].mean()), (
        "direction was pinned on the GHOST frames -- the counterfactual leg must inherit "
        "the factual direction, not define it"
    )
    assert calls[0]["values"] == (expected,) * len(actual), "the pinned FACTUAL direction was not used"
    assert round(float(ghost["x"].mean()), 6) != expected, (
        "vacuous: the ghost displacement did not move the stub's keying statistic, so the "
        "factual-vs-ghost assertions above could not fail"
    )


def test_das_arm_does_not_mutate_its_inputs(monkeypatch):
    """Runs library-free: the arm writes ``attacking_direction`` onto both legs, and doing
    that in place would corrupt the caller's frames."""
    import silly_kicks.gkdv._das_port as port

    monkeypatch.setattr(port, "pin_direction", lambda frames: pd.Series([1.0] * len(frames)))
    monkeypatch.setattr(port, "team_das", lambda frames, **kwargs: 1.0)

    actual, ghost = _port_frames(), _port_frames()
    before_actual, before_ghost = actual.copy(deep=True), ghost.copy(deep=True)
    delta_das(actual, ghost, attacking_team_id=2)
    pd.testing.assert_frame_equal(actual, before_actual)
    pd.testing.assert_frame_equal(ghost, before_ghost)


def test_das_arm_rejects_row_misaligned_legs(monkeypatch):
    """The pin is applied POSITIONALLY, so a misaligned ghost would be scored against
    another row's direction -- a per-row sign flip invisible in the returned scalar."""
    import silly_kicks.gkdv._das_port as port

    monkeypatch.setattr(port, "pin_direction", lambda frames: pd.Series([1.0] * len(frames)))
    monkeypatch.setattr(port, "team_das", lambda frames, **kwargs: 1.0)

    actual = _port_frames()
    misaligned = actual.iloc[::-1]  # same rows, same length, DIFFERENT order
    with pytest.raises(ValueError, match="row-aligned"):
        delta_das(actual, misaligned, attacking_team_id=2)


def _infer_direction(frames: pd.DataFrame) -> tuple:
    """What an UNPINNED accessible-space would infer for these frames."""
    from accessible_space.interface import infer_playing_direction

    masked = frames.copy()
    # Mirrors _pin_attacking_direction's own ball-masking step. House idiom rather than
    # `== True`: equivalent on a plain bool column, safe on a nullable one.
    masked.loc[masked["is_ball"].astype(bool), "team_id"] = None
    return tuple(
        infer_playing_direction(
            masked,
            team_col="team_id",
            period_col="period_id",
            team_in_possession_col="team_in_possession",
            x_col="x",
            ball_team=None,
            frame_col="frame_id",
        ).to_numpy()
    )


def test_unpinned_implementation_would_measurably_differ():
    """VALUE discriminator: prove the pin is not a no-op on this fixture.

    If this ever goes green with the pin removed, the fixture has stopped discriminating
    and must be made more extreme.
    """
    pytest.importorskip("accessible_space")

    assert _infer_direction(_das_frames()) != _infer_direction(_das_frames(_GHOST_GK_X)), (
        "fixture no longer discriminates: an unpinned implementation infers the SAME "
        "direction for both legs here, so the pinning guard above proves nothing"
    )


def test_pinned_delta_differs_from_the_unpinned_delta():
    """The strongest form: the pin changes the ANSWER, not merely an intermediate column.

    Computes the same difference the arm computes, but letting each leg infer its own
    direction -- exactly what a revert to ``get_das(infer_attacking_direction=True)`` would
    do -- and asserts the two disagree.
    """
    pytest.importorskip("accessible_space")
    from silly_kicks.tracking import get_individual_das

    actual, ghost = _das_frames(), _das_frames(_GHOST_GK_X)
    pinned = delta_das(actual, ghost, attacking_team_id="2")

    def _unpinned(frames):
        out = get_individual_das(frames)
        rows = out[~out["is_ball"].astype(bool) & (out["team_id"] == "2")]
        return float(rows["DAS"].dropna().sum())

    unpinned = _unpinned(actual) - _unpinned(ghost)
    assert pinned != unpinned, (
        "pinned and unpinned deltas agree -- either the pin is not being applied, or the fixture stopped discriminating"
    )


def test_das_arm_identical_frames_give_exactly_zero():
    pytest.importorskip("accessible_space")

    frames = _das_frames()
    assert delta_das(frames, frames.copy(), attacking_team_id="2") == 0.0


def test_das_arm_is_id_dtype_safe():
    """ADR-019: ``attacking_team_id`` is compared against ``team_id`` inside the port, so a
    value-equal scalar of a different dtype must give an IDENTICAL result."""
    pytest.importorskip("accessible_space")

    actual, ghost = _das_frames(), _das_frames(_GHOST_GK_X)
    stringy = delta_das(actual, ghost, attacking_team_id="2")
    numeric = delta_das(actual, ghost, attacking_team_id=2)
    assert stringy == numeric
    assert stringy != 0.0, "vacuous: both dtypes agreed only because the value is zero"


# ---------------------------------------------------------------------------
# Delta-DAS -- LIVE through real accessible-space
# ---------------------------------------------------------------------------

#: accessible-space's carrier/offside column, resolved by NAME inside ``_das``.
_CARRIER_COL = "ball_carrier_player_id"


def _das_frames_with_carrier(gk_x: float = 10.0) -> pd.DataFrame:
    """``_das_frames`` plus the ball-carrier column real frames actually carry.

    The sibling DAS fixtures above carry NO carrier column, so
    ``_resolve_player_in_possession_col`` returns ``None`` for them and accessible-space's
    offside path -- which 2-D-indexes the carrier as PASSERS -- never runs. Production
    frames DO carry it (``derive_team_in_possession`` preserves it), so without this the
    arm's live behaviour is untested exactly where it is exercised.

    The dtype is pinned EXPLICITLY, not left to inference: pandas 2 infers ``object`` (where
    the 2-D indexing is harmless) and pandas 3 infers ``StringDtype`` (where it raises).
    Relying on inference would make this guard silently interpreter-dependent -- which is
    precisely how the all-NaN defect shipped.
    """
    frames = _das_frames(gk_x)
    # "a2" is a real team-2 player sitting on the ball in this layout.
    frames[_CARRIER_COL] = pd.Series(["a2"] * len(frames), dtype="string", index=frames.index)
    return frames


def test_das_arm_returns_a_LIVE_FINITE_delta_through_real_accessible_space():
    """The arm's headline number must be a real one -- NOT a silent all-NaN collapse.

    Every other assertion about the DAS arm either stubs ``_das_port`` or runs on frames
    with no carrier column, so nothing noticed when accessible-space scored ZERO frames on
    pandas 3: ``team_das`` sums ``DAS.dropna()``, and the sum of an empty selection is
    ``0.0``, so BOTH legs degraded to ``0.0`` and the delta came back a tidy, finite,
    completely fictional zero. Measured: reintroducing the defect leaves the rest of
    ``tests/gkdv/`` green, all 163 of it.

    So finiteness alone is NOT sufficient here, and the two non-vacuity assertions below
    are the actual guard:

    * the underlying per-player DAS values must be finite -- the frames really scored;
    * the delta must be non-zero -- the all-NaN collapse produces exactly ``0.0``.
    """
    pytest.importorskip("accessible_space")
    from silly_kicks.tracking import get_individual_das

    actual, ghost = _das_frames_with_carrier(), _das_frames_with_carrier(_GHOST_GK_X)

    # Fixture non-vacuity: this must really present the dtype that exercises the 2-D
    # indexing path. Were it to arrive as object, the guard would pass without testing.
    assert isinstance(actual[_CARRIER_COL].dtype, pd.StringDtype), (
        "fixture must carry a StringDtype carrier column, else the offside path is untested"
    )

    # (1) The frames genuinely scored -- not an all-NaN degrade that sums to zero.
    scored = get_individual_das(actual)
    finite = scored.loc[~scored["is_ball"].astype(bool), "DAS"]
    assert finite.notna().any(), (
        "accessible-space scored NO frame: every underlying DAS is NaN, so any delta built "
        "from these legs is fictional regardless of how finite it looks"
    )
    assert float(finite.dropna().sum()) > 0.0, "a real accessible-space area is strictly positive"

    # (2) The arm's own output, through the real library end to end.
    delta = delta_das(actual, ghost, attacking_team_id="2")
    assert np.isfinite(delta), f"delta_das returned a non-finite value: {delta!r}"
    assert delta != 0.0, (
        "delta is exactly zero, which is the signature of BOTH legs degrading to all-NaN "
        "(team_das sums DAS.dropna(), and an empty sum is 0.0) -- not of a keeper who "
        "changed nothing while relocated 90 m upfield"
    )
