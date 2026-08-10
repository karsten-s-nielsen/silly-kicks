"""CONSUMER behaviour under the goal-map seam (ADR-055).

Two jobs, and they are different questions:

1. **Map dependence for consumers the mirror registry cannot reach.** Gate C
   (``test_mirror_registry.py``) varies the map for registered ``add_*`` aggregators. Two public
   paths are re-keyed by ADR-055 and are NOT ``add_*``, so no registry gate covers them:
   ``gk_closing_time_min_s`` / ``gk_closing_time_mean_s``, which route through
   ``_closing_time_per_series``.

   This is not a hypothetical hole. Executing the defect -- patching
   ``_closing_time_per_series`` back onto a self-built map and running Gate C on
   ``add_gk_influence`` -- leaves Gate C **GREEN**, because ``add_gk_influence`` never calls that
   function; its closing-time columns come from ``_gk_influence_at_actions``. The plan for this
   cycle asserted the opposite ("a 1-column result means ``_closing_time_per_series`` was
   missed"), which was a claim about a call graph nobody had run.

2. **Correctness, which Gate C explicitly does not prove.** ``get`` and ``attacked_goal`` both
   move when the map is swapped, so a moved column shows the map is CONSULTED, never that the own
   end and the attacked end were not transposed. The degenerate-shape tests below assert the
   documented outcome per consumer instead of merely "differs from before".
"""

from __future__ import annotations

import warnings
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from silly_kicks.id_compat import ids_match
from silly_kicks.tracking import GoalMap, resolve_defended_goals
from tests.tracking._mirror_registry import FIELD_LENGTH, canonical_scene


def _flipped(gm: GoalMap) -> GoalMap:
    """Both ends swapped -- a COHERENT rival map, not a corrupted one.

    The two teams still defend opposite ends, so ``attacked_goal`` resolves and the degeneracy
    guard does not fire. A consumer that reads the map must therefore answer differently.
    """

    def _swap(pool):
        return MappingProxyType({k: (FIELD_LENGTH if v == 0.0 else 0.0) for k, v in pool.items()})

    return GoalMap(_swap(gm.resolved), _swap(gm.guessed), gm.unresolved)


def _max_delta(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    both = np.isfinite(x) & np.isfinite(y)
    if not both.any():
        return float("nan")
    return float(np.abs(x[both] - y[both]).max())


# ---------------------------------------------------------------------------
# 1. The path Gate C is blind to
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn_name",
    ["gk_closing_time_min_s", "gk_closing_time_mean_s"],
)
def test_closing_time_helpers_read_the_injected_goal_map(fn_name):
    """``_closing_time_per_series`` must consume the caller's map, not build its own.

    Registered ``add_*`` aggregators get this from Gate C. These two do not: they are public
    exports but not aggregators, so the mirror registry never sees them, and ``add_gk_influence``
    reaches its closing-time columns by a different route entirely.
    """
    from silly_kicks.tracking import features as F

    fn = getattr(F, fn_name)
    actions, frames = canonical_scene()
    true_map = resolve_defended_goals(frames)

    ref = fn(actions.copy(), frames.copy(), goal_map=true_map)
    alt = fn(actions.copy(), frames.copy(), goal_map=_flipped(true_map))

    assert ref.notna().any(), f"{fn_name} produced no finite values -- the comparison is vacuous"
    delta = _max_delta(ref, alt)
    assert delta > 1e-12, (
        f"{fn_name}: swapping the goal map moved NOTHING (max delta {delta}). The helper is "
        "ignoring the injected map and building its own -- a partial re-key."
    )


def test_closing_time_helpers_default_to_a_map_built_from_frames():
    """``goal_map=None`` is not a degradation: it means "build it from these frames".

    Correct by construction here -- the function HAS the full frames -- and it is what keeps the
    parameter optional at the aggregator layer while staying REQUIRED on the per-frame functions,
    where a default would re-admit per-frame construction.
    """
    from silly_kicks.tracking import features as F

    actions, frames = canonical_scene()
    explicit = F.gk_closing_time_min_s(actions.copy(), frames.copy(), goal_map=resolve_defended_goals(frames))
    implied = F.gk_closing_time_min_s(actions.copy(), frames.copy())

    assert implied.notna().any(), "the default path produced nothing -- this proves no equality"
    pd.testing.assert_series_equal(explicit, implied)


def test_the_removed_home_team_id_is_not_silently_accepted():
    """Hyrum: the removal must FAIL LOUD, not be swallowed by a **kwargs or an unread parameter.

    A signature that still accepts ``home_team_id`` and ignores it is strictly worse than one
    that raises -- the caller keeps passing an identity that no longer steers anything.
    """
    from silly_kicks.tracking import features as F

    actions, frames = canonical_scene()
    for fn in (
        F.gk_closing_time_min_s,
        F.gk_closing_time_mean_s,
    ):
        with pytest.raises(TypeError, match="home_team_id"):
            # type: ignore is REQUIRED and is the assertion's whole point -- the checker
            # agreeing that this call is invalid is the same fact the raise verifies at runtime.
            fn(actions.copy(), frames.copy(), home_team_id=1)  # type: ignore[call-arg]

    for fn in (F.gk_pitch_control_share_weighted, F.gk_reachable_area_m2):
        with pytest.raises(TypeError, match="home_team_id"):
            fn(actions.copy(), frames.copy(), None, home_team_id=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 3. The ghost-GK provenance token (ADR-055)
# ---------------------------------------------------------------------------


def _unresolvable_away_end():
    """``(actions, frames)`` where the DEFENDING keeper is present but its end is unresolvable.

    Every away row (keeper included) carries NaN coordinates, so ``resolve_defended_goals`` puts
    that team in ``unresolved`` -- in NEITHER mapping. The keeper ROW still exists, which is the
    whole point: this is the state that ``no_keeper`` describes wrongly.
    """
    actions, frames = canonical_scene()
    f = frames.copy()
    # `ids_match`, NOT `astype(str) == "2"`: the ball row makes `team_id` float64, so
    # `astype(str)` renders the away team as "2.0" and the mask matches NOTHING -- the
    # str()-comparison trap this repo has already shipped twice (ADR-043).
    away = ~f["is_ball"].astype(bool) & ids_match(f["team_id"], 2)
    assert away.sum() > 0, "the away mask matched no rows -- the fixture builds nothing"
    f.loc[away, ["x", "y"]] = np.nan
    return actions.copy(), f


def test_ghost_gk_source_names_the_unresolvable_end_not_a_missing_keeper():
    """The token must be earned by the ROW, and must not be `no_keeper`.

    Both halves matter. Asserting only "the token appears somewhere" would pass on an
    implementation that emitted it unconditionally; asserting only "not no_keeper" would pass on
    one that emitted `unlinked`. So: the token IS present, `no_keeper` is NOT, and the keeper row
    the verdict is about genuinely exists in the frames.
    """
    from silly_kicks.tracking import GHOST_GK_GOAL_END_UNRESOLVED, GHOST_GK_NO_KEEPER
    from silly_kicks.tracking.features import add_ghost_gk

    actions, frames = _unresolvable_away_end()

    gm = resolve_defended_goals(frames)
    assert gm.get(1, 1, 2, allow_guess=True) is None, "fixture no longer degenerate -- test is vacuous"
    keeper_rows = frames[frames["is_goalkeeper"].astype(bool) & ids_match(frames["team_id"], 2)]
    assert len(keeper_rows) > 0, "the away KEEPER ROW must exist, or this is just a missing keeper"

    out = add_ghost_gk(actions, frames, home_team_id=1)
    home_rows = out[ids_match(out["team_id"], 1)]
    sources = set(home_rows["ghost_gk_source"])

    assert GHOST_GK_GOAL_END_UNRESOLVED in sources, (
        f"home actions defend against the degenerate away keeper, so at least one row must be "
        f"tagged {GHOST_GK_GOAL_END_UNRESOLVED!r}; got {sorted(sources)}"
    )
    assert GHOST_GK_NO_KEEPER not in sources, (
        "the away keeper ROW is present, so `no_keeper` states something the frames refute"
    )
    assert home_rows.loc[home_rows["ghost_gk_source"] == GHOST_GK_GOAL_END_UNRESOLVED, "ghost_gk_x"].isna().all()


def test_a_resolvable_scene_does_NOT_get_the_token():
    """Non-vacuity for the test above: the token must be earned, not unconditional."""
    from silly_kicks.tracking import GHOST_GK_GOAL_END_UNRESOLVED
    from silly_kicks.tracking.features import add_ghost_gk

    actions, frames = canonical_scene()
    out = add_ghost_gk(actions.copy(), frames.copy(), home_team_id=1)
    assert GHOST_GK_GOAL_END_UNRESOLVED not in set(out["ghost_gk_source"]), (
        "a fully-resolvable scene emitted the unresolved token -- it is not conditional on anything"
    )


def test_serve_ghost_gk_positions_returns_NO_ROW_for_an_unresolvable_end():
    """The documented asymmetry (ADR-054 D2): the aggregator NaNs, the serve seam OMITS.

    `gkdv/_engine.py` RAISES on a non-finite served ghost, so `serve_ghost_gk_positions` must
    drop the row rather than hand back a NaN one. Asserted against the aggregator's own count on
    the same frames, so the two seams are compared rather than each judged in isolation.
    """
    from silly_kicks.tracking import serve_ghost_gk_positions

    actions, frames = _unresolvable_away_end()
    served = serve_ghost_gk_positions(frames, home_team_id=1, actions=actions)

    away_served = served[ids_match(served["gk_team_id"], 2)] if len(served) else served
    assert len(away_served) == 0, (
        f"the serve seam returned {len(away_served)} row(s) for a keeper whose end does not "
        "resolve; gkdv raises on a non-finite ghost, so the row must be absent, not NaN"
    )


# ---------------------------------------------------------------------------
# 4. CONSUMER x degenerate-shape characterization
# ---------------------------------------------------------------------------
#
# Every outcome below was MEASURED on the current tree and transcribed, in the SB360 registry's
# idiom: the point is to make a future change to any of these paths VISIBLE, not to assert what
# the behaviour ought to be. Two findings the matrix produced, both recorded rather than fixed
# (neither is this cycle's doing, and repairing either is a scope decision):
#
# 1. `nullable_boolean_na` RAISES for ALL SIX consumers, identically --
#    ValueError: cannot convert to bool-dtype NumPy array with missing values. A pd.NA in a
#    nullable-boolean `is_goalkeeper` reaches `.astype(bool)` unguarded everywhere. It is uniform,
#    so it is a property of the shared idiom rather than of any one aggregator.
#
# 2. `string_is_ball` splits the surface. Four consumers go all-NaN (every row reads as the ball,
#    because `pd.Series(["false"]).astype(bool)` is True, so no players remain); `add_cover_shadows`
#    and `add_shot_goalmouth` stay live, i.e. they do NOT share that idiom. The all-NaN four
#    degrade honestly; the split itself is the finding.
#
# `add_cover_shadows` staying LIVE on the three keeper-degenerate shapes is NOT in tension with
# the SB360 `gk_absent` collapse: there both teams' outfield means landed past the midline and the
# guessed map was DEGENERATE, while `canonical_scene`'s outfield means straddle it and resolve to
# opposite ends. Same code, different fixture -- which is exactly why the SB360 verdict names the
# measured means rather than saying "no keeper".

_SHAPES = (
    "baseline",
    "gk_absent",
    "na_team_gk",
    "all_nan_x_gk",
    "string_is_ball",
    "nullable_boolean_na",
)

#: (consumer, shape) -> "live" | "all_nan" | "raises"
_CONSUMER_MATRIX: dict[tuple[str, str], str] = {}
for _fn, _row in {
    "add_gk_influence": ("live", "all_nan", "all_nan", "all_nan", "all_nan", "raises"),
    "add_cover_shadows": ("live", "live", "live", "live", "live", "raises"),
    "add_ghost_gk": ("live", "all_nan", "all_nan", "live", "all_nan", "raises"),
    "add_xcross_attempt": ("live", "live", "live", "live", "all_nan", "raises"),
    "add_xshot_occurrence": ("live", "live", "live", "live", "all_nan", "raises"),
    "add_shot_goalmouth": ("live", "live", "live", "live", "live", "raises"),
}.items():
    for _shape, _outcome in zip(_SHAPES, _row, strict=True):
        _CONSUMER_MATRIX[(_fn, _shape)] = _outcome

#: Columns whose liveness decides the verdict. Provenance/linkage columns are excluded on
#: purpose: `frame_id` is populated whenever the action linked, so including it would make every
#: row read "live" and the matrix would assert nothing.
_VERDICT_COLUMNS: dict[str, tuple[str, ...]] = {
    "add_gk_influence": (
        "gk_pitch_control_share_weighted",
        "gk_reachable_area_m2",
        "gk_closing_time_min_s__six_yard_box",
        "gk_closing_time_mean_s__six_yard_box",
    ),
    "add_cover_shadows": (
        "n_blocked_receivers",
        "n_potential_receivers",
        "blocking_score",
        "blocked_threat_fraction",
        "max_single_defender_blocking_score",
    ),
    "add_ghost_gk": ("ghost_gk_x", "ghost_gk_y"),
    "add_xcross_attempt": ("xcross_attempt",),
    "add_xshot_occurrence": ("xshot_occurrence",),
    "add_shot_goalmouth": ("shot_crossing_source", "shot_crossing_confidence"),
}


def _degenerate(shape: str):
    """``(actions, frames)`` for one named degeneracy, all built off ``canonical_scene``."""
    actions, frames = canonical_scene()
    actions, frames = actions.copy(), frames.copy()
    gk = frames["is_goalkeeper"].astype(bool) & ~frames["is_ball"].astype(bool)
    assert gk.any(), "canonical_scene has no keeper -- every shape below would be vacuous"

    if shape == "baseline":
        return actions, frames
    if shape == "gk_absent":
        return actions, frames[~gk].copy()
    if shape == "na_team_gk":
        frames.loc[gk, "team_id"] = pd.NA
    elif shape == "all_nan_x_gk":
        frames.loc[gk, ["x", "y"]] = np.nan
    elif shape == "string_is_ball":
        frames["is_ball"] = frames["is_ball"].map({True: "true", False: "false"})
    elif shape == "nullable_boolean_na":
        frames["is_goalkeeper"] = pd.array(frames["is_goalkeeper"].tolist(), dtype="boolean")
        frames.loc[gk.idxmax(), "is_goalkeeper"] = pd.NA
    else:  # pragma: no cover - guarded by the parametrization
        raise AssertionError(f"unknown shape {shape!r}")
    return actions, frames


def _invoke(name: str, actions, frames):
    import silly_kicks.tracking as T
    from tests.tracking._mirror_registry import gate_xt

    fn = getattr(T, name)
    if name in ("add_gk_influence", "add_cover_shadows"):
        return fn(actions, frames, gate_xt())
    if name in ("add_ghost_gk", "add_xcross_attempt", "add_xshot_occurrence"):
        return fn(actions, frames, home_team_id=1)
    return fn(actions, frames)


@pytest.mark.parametrize(("consumer", "shape"), sorted(_CONSUMER_MATRIX))
def test_consumer_behaviour_on_a_degenerate_frame_shape(consumer, shape):
    """Characterization: each consumer's MEASURED outcome per degeneracy.

    Asserts the documented outcome rather than "differs from before", so a regression names
    which consumer and which shape moved.
    """
    expected = _CONSUMER_MATRIX[(consumer, shape)]
    actions, frames = _degenerate(shape)

    if expected == "raises":
        with pytest.raises(ValueError, match="bool"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _invoke(consumer, actions, frames)
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _invoke(consumer, actions, frames)

    cols = [c for c in _VERDICT_COLUMNS[consumer] if c in out.columns]
    assert cols, f"{consumer}: none of its verdict columns were emitted -- the matrix is vacuous"
    live = {c: int(out[c].notna().sum()) for c in cols}
    if expected == "all_nan":
        assert all(v == 0 for v in live.values()), (
            f"{consumer} on {shape}: expected every verdict column NaN, got {live}. A value here "
            "means the aggregator produced a number from a frame that cannot support one."
        )
    else:
        assert any(v > 0 for v in live.values()), f"{consumer} on {shape}: expected live values, got {live}"


def test_the_baseline_row_is_live_for_every_consumer():
    """Non-vacuity for the whole matrix: on a HEALTHY scene nothing is all-NaN.

    Without this, an ``all_nan`` expectation would be satisfiable by a consumer that never
    produces anything at all, and the matrix would record a broken fixture as a contract.
    """
    for consumer, shape in _CONSUMER_MATRIX:
        if shape == "baseline":
            assert _CONSUMER_MATRIX[(consumer, shape)] == "live", (
                f"{consumer} is not live on the baseline scene -- every degenerate expectation "
                "below it is then unfalsifiable"
            )


# ---------------------------------------------------------------------------
# 5. The estimator claim itself (ADR-055)
# ---------------------------------------------------------------------------


def test_a_per_frame_map_really_is_a_different_estimator_on_sparse_detection():
    """The cycle's central design decision, re-derived rather than cited.

    "Build the map ONCE per match" is the rule every per-frame signature in this cycle exists to
    enforce, and it rests on a number. The spec's headline was 78.8% wrong; measured here it is
    **7.1%** on skillcorner and **0.0%** on dense tracking, so the ADR cites the reproduced
    figures instead -- see ADR-055 "The per-frame cost, measured".

    What this test pins is the SHAPE of that finding, which is what the design depends on:

    * on SPARSE broadcast detection a per-frame map is materially different -- it disagrees on
      some team-frames and, far more often, cannot answer at all;
    * on DENSE tracking it is identical, so the rule costs nothing there.

    Both halves matter. Without the first the threading is unmotivated; without the second a
    reader would think the seam was fixing something on every provider.
    """
    from silly_kicks.tracking import resolve_defended_goals
    from tests.tracking._provider_inputs import load_provider_frames

    def _rates(provider):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frames = load_provider_frames(provider)
        per_match = resolve_defended_goals(frames)
        disagree = unresolvable = total = 0
        for (gid, pid, _fid), grp in frames.groupby(["game_id", "period_id", "frame_id"]):
            per_frame = resolve_defended_goals(grp)
            for team in grp["team_id"].dropna().unique():
                truth = per_match.get(gid, pid, team, allow_guess=True)
                if truth is None:
                    continue
                total += 1
                if per_frame.get(gid, pid, team, allow_guess=True) != truth:
                    disagree += 1
                if per_frame.attacked_goal(gid, pid, team, allow_guess=True) is None:
                    unresolvable += 1
        assert total > 0, f"{provider}: no comparable team-frames -- this measurement is vacuous"
        return disagree / total, unresolvable / total

    sparse_disagree, sparse_unresolvable = _rates("skillcorner")
    dense_disagree, dense_unresolvable = _rates("sportec")

    # SPARSE: both effects present and large enough to decide a design.
    assert sparse_disagree > 0.02, (
        f"skillcorner per-frame disagreement fell to {sparse_disagree:.3f}; the threading rule's "
        "motivation is measured, so a collapse here means the rule needs re-justifying"
    )
    assert sparse_unresolvable > 0.20, (
        f"skillcorner per-frame attacked_goal unresolvable fell to {sparse_unresolvable:.3f} -- "
        "this is the LARGER half of the finding and the one the old code answered with 105.0"
    )

    # DENSE: the rule is free. Asserting this is what stops the ADR overclaiming.
    assert dense_disagree == 0.0 and dense_unresolvable == 0.0, (
        f"sportec is no longer exactly reproducible per-frame ({dense_disagree:.3f} / "
        f"{dense_unresolvable:.3f}); the 'provider-dependent' claim in ADR-055 assumes it is"
    )
    assert sparse_unresolvable > dense_unresolvable, "the sparse/dense contrast is the finding"
