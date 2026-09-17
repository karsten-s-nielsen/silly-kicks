"""method="counterfactual" joint valuation q*c*xT -- the §5.7 goldens + membership/frame pins (Task 9).

The counterfactual prevented-valuation is the GSAA-analog ``(P_complete - outcome) * xT(target)`` summed
over opponent passes aimed into the defender's trimmed hull. A completed pass is valued at its observed
end; a failed pass is valued over its death-direction cone's target distribution ``q`` (renormalized
transition mass restricted to the cone-and-hull zones), each hypothesized target weighted by a fitted
completion ``c`` and the injected xT value. Frame reconciliation (ADR-028): origin/death/zone-centres are
in the OPPONENT frame (where the cone + q + xT(z) live); only hull MEMBERSHIP reflects the zone centres
(and a completed pass end) into the DEFENDER frame ``(fl-x, fw-y)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.territory import CounterfactualParams, compute_territorial_dominance
from silly_kicks.territory._hull import build_trimmed_hull
from silly_kicks.xthreat import ExpectedThreat, GridSpec, destination_profiles

_PASS = spadlconfig.actiontype_id["pass"]
_TACKLE = spadlconfig.actiontype_id["tackle"]
_SHOT = spadlconfig.actiontype_id["shot"]
_OK = spadlconfig.result_id["success"]
_FAIL = spadlconfig.result_id["fail"]


class _ToyUniformXt:
    """Duck-typed fitted xT with a UNIFORM grid: ``values_at_points == 0.1`` for ANY point
    (inversion-invariant), and a uniform positive transition_matrix so every zone is supported. Makes
    ``Σ q·xT = 0.1·(Σq=1) = 0.1`` exactly -- independent of the reflection/cone details -- so the test
    pins the completion multiply + renormalize-to-1 to the last digit (spec §5.7 uniform check)."""

    def __init__(self, value: float = 0.1, nx: int = 8, ny: int = 6) -> None:
        self.l, self.w = nx, ny  # xthreat reads model.l / model.w (grid dims)
        self.grid = GridSpec(n_zones_x=nx, n_zones_y=ny)
        n = nx * ny
        self.xT = np.full((ny, nx), float(value))
        self.transition_matrix = np.full((n, n), 1.0 / n)
        self.method = "singh_counts"


class _ConstCompletion:
    def predict_completion(self, ox, oy, tx, ty):
        return np.full(np.asarray(tx, float).shape, 0.6)  # c == 0.6 (spec section 5.7)


# --- scene builders -------------------------------------------------------------------------------------
# Defender team 1 (player 7) attacks x=105 in action-LTR; its 3 own-half tackles at (2,20),(2,48),(14,34)
# form (with trim_fraction=0.70) a triangle hull containing the reflected completed-pass end (10,34) but
# NOT the reflected failed-pass death (15,34). Team-2 opponent passes are in the opponent frame (high x);
# the completed end (95,34) reflects to (10,34) INSIDE the hull; the failed pass (80,34)->(90,34) has a
# +x death direction whose ±45° cone selects the two reflected hull zones (verified geometry), yet its
# death (15,34) lies OUTSIDE the hull (aimed-in-died-short). See the module docstring for the frame rule.
_TACKLES = [(2.0, 20.0), (2.0, 48.0), (14.0, 34.0)]


def _rows(defs, passes):
    out = []
    for x, y in defs:
        out.append(
            {
                "game_id": None,
                "period_id": 1,
                "team_id": 1,
                "player_id": 7,
                "type_id": _TACKLE,
                "result_id": _OK,
                "start_x": x,
                "start_y": y,
                "end_x": x,
                "end_y": y,
            }
        )
    for sx, sy, ex, ey, res in passes:
        out.append(
            {
                "game_id": None,
                "period_id": 1,
                "team_id": 2,
                "player_id": 21,
                "type_id": _PASS,
                "result_id": res,
                "start_x": sx,
                "start_y": sy,
                "end_x": ex,
                "end_y": ey,
            }
        )
    return out


def _finish(rows, game_id):
    df = pd.DataFrame(rows)
    df["game_id"] = game_id
    df["action_id"] = range(len(df))
    return df


def _scene(game_id: int = 1) -> pd.DataFrame:
    passes = [
        (88.0, 34.0, 95.0, 34.0, _OK),  # completed: reflected end (10,34) INSIDE hull -> conceded
        (80.0, 34.0, 90.0, 34.0, _FAIL),  # failed: +x cone selects hull zones; death (15,34) OUTSIDE
    ]
    return _finish(_rows(_TACKLES, passes), game_id)


def _scene_completed_only(game_id: int = 1) -> pd.DataFrame:
    return _finish(_rows(_TACKLES, [(88.0, 34.0, 95.0, 34.0, _OK)]), game_id)


def _scene_failed_points_away(game_id: int = 1) -> pd.DataFrame:
    passes = [
        (88.0, 34.0, 95.0, 34.0, _OK),  # normal completed conceded
        (80.0, 34.0, 70.0, 34.0, _FAIL),  # death direction -x points AWAY -> cone∩hull empty -> unresolved
    ]
    return _finish(_rows(_TACKLES, passes), game_id)


# --- M2 (whole-branch review): a hull off-centre from y=34, so the y-reflection term is DISCRIMINATING.
# Every scene above straddles y=34 with passes AT y=34 (fw-y = 68-34 = 34 == y), so dropping the `fw-`
# term entirely would be indistinguishable from identity there. This triangle's y-range [5, 25] sits
# nowhere near the midline.
_ASYM_TACKLES = [(2.0, 5.0), (2.0, 25.0), (20.0, 15.0)]


def _asym_scene(game_id: int = 1) -> pd.DataFrame:
    """A completed pass whose raw end (95, 53) reflects via ``(fl-x, fw-y)`` to (10, 15) -- INSIDE the
    off-centre hull -- but whose RAW (un-reflected) y=53 would test containment of (10, 53), which is
    outside the hull's y-range entirely. See the test docstring for the containment arithmetic.
    """
    passes = [(88.0, 50.0, 95.0, 53.0, _OK)]
    return _finish(_rows(_ASYM_TACKLES, passes), game_id)


def _mirror_scene(scene: pd.DataFrame) -> pd.DataFrame:
    """Same physical scene from the opposite (either-team) perspective: swap the team labels 1<->2 so
    the DEFENDER becomes team 2 (its own-half tackles unchanged) and the opponent becomes team 1. Each
    team's actions already live in its own action-LTR frame, so relabelling -- not a global coordinate
    reflection -- is the scoreable either-perspective transform (a literal ``(105-x, 68-y)`` reflection
    moves the defender's tackles out of its own half, which the ``start_x < own_half_max_x`` filter then
    drops, yielding an empty mirrored output). Pins team-identity invariance (ADR-028 / ADR-051-D3): the
    counterfactual numbers must not depend on which integer labels the defender/opponent teams carry.
    """
    m = scene.copy(deep=True)
    m["team_id"] = m["team_id"].map({1: 2, 2: 1}).astype(m["team_id"].dtype)
    return m


def _disjoint_corpus() -> pd.DataFrame:
    """A small synthetic SPADL corpus for a NON-degenerate, NON-uniform ``ExpectedThreat`` fit: a dense
    forward-pass grid populating every move cell + explicit successful passes from ~(80,34) to a spread
    of forward-right destinations (so the failed pass's origin cell has transition mass on the hull-region
    zones), plus shots/goals near the goal so scoring-prob (hence xT) is non-uniform.
    """
    rows: list[dict] = []

    def add(type_id, res, sx, sy, ex, ey):
        rows.append(
            {
                "game_id": 99,
                "period_id": 1,
                "team_id": 1,
                "player_id": 1,
                "type_id": type_id,
                "result_id": res,
                "start_x": float(sx),
                "start_y": float(sy),
                "end_x": float(ex),
                "end_y": float(ey),
            }
        )

    for sx in range(4, 100, 7):
        for sy in range(4, 65, 7):
            add(_PASS, _OK, sx, sy, min(104.0, sx + 11.0), float(np.clip(sy + 2.0, 1, 67)))
    # populate the (80,34) origin cell's transition row toward the hull-region destinations (varying
    # frequencies -> a non-uniform q that the renormalization must handle correctly).
    for tx, ty, cnt in [(95, 25, 3), (95, 34, 5), (95, 43, 2), (101, 30, 1), (101, 38, 4), (104, 34, 2)]:
        for _ in range(cnt):
            add(_PASS, _OK, 80, 34, tx, ty)
    # shots near goal, some goals -> non-zero, non-uniform scoring probability.
    for i, (sx, sy) in enumerate([(100, 34), (98, 30), (99, 38), (102, 34), (101, 28), (97, 40)]):
        add(_SHOT, _OK if i % 2 == 0 else _FAIL, sx, sy, 105, 34)
    df = pd.DataFrame(rows)
    df["action_id"] = range(len(df))
    return df


def _mixed_pass_corpus() -> pd.DataFrame:
    """A SPADL corpus with BOTH completed and failed passes -- a real ``PassCompletionModel`` needs two
    outcome classes to fit -- reusing ``_disjoint_corpus`` (dense forward-pass grid + shots for a
    non-degenerate xT) but flipping every 3rd pass to a failure."""
    df = _disjoint_corpus().copy()
    pass_idx = df.index[df["type_id"] == _PASS]
    df.loc[pass_idx[::3], "result_id"] = _FAIL
    return df


def _within_cone(dvec, zc, origin, cone_deg):
    dnorm = float(np.hypot(dvec[0], dvec[1]))
    if dnorm == 0.0:
        return np.zeros(len(zc), dtype=bool)
    zvec = zc - np.asarray(origin, float)
    znorm = np.hypot(zvec[:, 0], zvec[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (zvec @ np.asarray(dvec, float)) / (znorm * dnorm)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return (ang <= cone_deg) & (znorm > 0)


def _inline_reference_prevented(scene, xt, *, c, cone_deg, fl=105.0, fw=68.0, trim=0.70):
    """Re-derive prevented from the PUBLIC seam + an explicit cone/reflection/renorm loop -- independent
    of ``_counterfactual.py``'s helpers -- so a wrong cone/renorm/reflection in production diverges."""
    defs = scene[scene["type_id"] == _TACKLE]
    hull = build_trimmed_hull(defs[["start_x", "start_y"]].to_numpy(float), trim_fraction=trim)
    assert hull is not None
    failed = scene[(scene["type_id"] == _PASS) & (scene["result_id"] == _FAIL)]
    total = 0.0
    for _, p in failed.iterrows():
        ox, oy, dx, dy = p["start_x"], p["start_y"], p["end_x"], p["end_y"]
        prof = destination_profiles(xt, np.array([ox]), np.array([oy]))
        centres, values, probs = prof.zone_centres, prof.zone_values, prof.probabilities[0]
        refl = np.column_stack([fl - centres[:, 0], fw - centres[:, 1]])
        in_hull = hull.contains(refl)
        in_cone = _within_cone((dx - ox, dy - oy), centres, (ox, oy), cone_deg)
        sel = in_cone & in_hull
        support = float(probs[sel].sum())
        if support < CounterfactualParams.default().min_transition_support:
            continue
        q = probs[sel] / support
        total += float((q * c * values[sel]).sum())
    return total


# --- tests ----------------------------------------------------------------------------------------------
def test_uniform_xt_exact_golden():
    out, rep = compute_territorial_dominance(
        _scene(),
        xt=_ToyUniformXt(),  # type: ignore[arg-type]
        method="counterfactual",
        completion_model=_ConstCompletion(),  # type: ignore[arg-type]
        params=CounterfactualParams.default(),
    )
    row = out.iloc[0]
    # section 5.7 uniform check: conceded = xT(end) = 0.1; prevented = c*0.1*(sum q=1) = 0.06;
    # expected_faced = 0.06(completed) + 0.06(failed) = 0.12; above_expectation = 0.12 - 0.10 = 0.02.
    assert row["territory_xt_conceded"] == pytest.approx(0.10, abs=1e-12)
    assert row["territory_xt_prevented"] == pytest.approx(0.06, abs=1e-12)
    assert row["territory_expected_threat_faced"] == pytest.approx(0.12, abs=1e-12)
    assert row["territory_xt_prevented_above_expectation"] == pytest.approx(0.02, abs=1e-12)
    assert row["territory_mean_completion_faced"] == pytest.approx(0.6, abs=1e-12)
    # SPEC-04 count distinction: passes_into_hull = v1 observed-end-in-hull (completed only = 1);
    # passes_aimed_into_hull = completed + failed-aimed-in = 2.
    assert int(row["territory_passes_into_hull"]) == 1
    assert int(row["territory_passes_aimed_into_hull"]) == 2
    assert row["territory_xt_prevented_rate"] == pytest.approx(0.06 / 2, abs=1e-12)  # denom = aimed-in
    # census conserves the failed-pass target resolution (ADR-042).
    assert rep.n_target_modeled == 1
    assert rep.n_target_unresolved == 0


def test_varying_xt_matches_inline_reference():
    xt = ExpectedThreat(l=16, w=12).fit(_disjoint_corpus())
    out, _ = compute_territorial_dominance(
        _scene(),
        xt=xt,
        method="counterfactual",
        completion_model=_ConstCompletion(),  # type: ignore[arg-type]
        params=CounterfactualParams.default(),
    )
    expected = _inline_reference_prevented(_scene(), xt, c=0.6, cone_deg=45.0)
    assert expected > 0.0  # non-vacuous: the fit corpus gives the origin cell real transition mass
    assert out.iloc[0]["territory_xt_prevented"] == pytest.approx(expected, abs=1e-9)


def test_completed_leg_above_expectation_is_c_minus_one_times_end_value():
    out, _ = compute_territorial_dominance(
        _scene_completed_only(),
        xt=_ToyUniformXt(),  # type: ignore[arg-type]
        method="counterfactual",
        completion_model=_ConstCompletion(),  # type: ignore[arg-type]
        params=CounterfactualParams.default(),
    )
    # (c-1)*xT(end) = (0.6-1)*0.1 = -0.04
    assert out.iloc[0]["territory_xt_prevented_above_expectation"] == pytest.approx(-0.04, abs=1e-12)
    assert out.iloc[0]["territory_xt_prevented"] == pytest.approx(0.0, abs=1e-12)


def test_completed_pass_membership_pins_the_y_reflection_term():
    """Every other cf VALUE test in this module uses a hull symmetric about y=34 with passes at y=34, so
    a bug dropping the ``fw-`` half of the point reflection ``(fl-x, fw-y)`` in ``_counterfactual.py``'s
    hull-membership test (``refl_end``, ~line 172) would be silently indistinguishable from identity and
    would pass every existing golden. This scene's hull is asymmetric -- triangle (2,5)-(2,25)-(20,15),
    y-range [5,25], nowhere near y=34 -- and the completed pass's raw end (95,53) reflects to (10,15).

    Containment check: fl-95=10 sits within the hull's x-span [2,20] either way (only the y term is in
    question here). Reflected y = fw-53 = 15; at x=10 the triangle's two slanted edges
    A(2,5)-C(20,15) and B(2,25)-C(20,15) bound y in [5 + (10/18)*(15-5), 25 - (10/18)*(25-15)] =
    [10*10/18+5, 25-10*10/18] = [9.44, 20.56] -- 15 is inside. Using the RAW (un-reflected) y=53 instead
    -- i.e. testing containment of (10, 53) -- is far outside the hull's whole y-range [5,25], so a
    dropped ``fw-`` term would collapse ``territory_xt_conceded``/``territory_passes_into_hull`` to 0.
    """
    out, _ = compute_territorial_dominance(
        _asym_scene(),
        xt=_ToyUniformXt(),  # type: ignore[arg-type]
        method="counterfactual",
        completion_model=_ConstCompletion(),  # type: ignore[arg-type]
        params=CounterfactualParams.default(),
    )
    row = out.iloc[0]
    assert row["territory_xt_conceded"] == pytest.approx(0.10, abs=1e-12)
    assert int(row["territory_passes_into_hull"]) == 1


def test_real_completion_model_through_failed_pass_multizone_path():
    """IMPL-01 regression: a REAL ``PassCompletionModel`` (NOT the geometry-blind ``_ConstCompletion``
    toy every other cf test uses) must not crash on the FAILED-pass path, where ONE origin is scored
    against its k>1 selected zone centres. The toy ignores its args, so it never called
    ``pass_completion_features`` and never exercised the scalar-origin vs length-k-target broadcast --
    the whole owner-run validation pass crashed on the first multi-zone failed pass before this."""
    from silly_kicks.expected_passing import PassCompletionModel

    xt = ExpectedThreat(l=16, w=12).fit(_disjoint_corpus())
    model = PassCompletionModel().fit(_mixed_pass_corpus())
    out, rep = compute_territorial_dominance(
        _scene(),
        xt=xt,
        method="counterfactual",
        completion_model=model,
        params=CounterfactualParams.default(),
    )
    prevented = out.iloc[0]["territory_xt_prevented"]
    assert np.isfinite(prevented) and prevented > 0.0  # the failed pass into the hull WAS valued (k>1 zones)
    assert int(rep.n_target_modeled) >= 1  # modeled, not crashed/dropped


def test_counterfactual_requires_a_completion_model():
    with pytest.raises(Exception):  # noqa: B017 -- any raise: a completion model is required
        compute_territorial_dominance(_scene(), xt=_ToyUniformXt(), method="counterfactual")  # type: ignore[arg-type]


def test_unresolvable_target_is_dropped_and_counted():
    out, rep = compute_territorial_dominance(
        _scene_failed_points_away(),
        xt=_ToyUniformXt(),  # type: ignore[arg-type]
        method="counterfactual",
        completion_model=_ConstCompletion(),  # type: ignore[arg-type]
        params=CounterfactualParams.default(),
    )
    assert rep.n_target_unresolved >= 1
    assert (out["territory_target_source"] == "unresolved").any()
    # dropped, never fabricated: the away pass contributes nothing to prevented.
    assert out.iloc[0]["territory_xt_prevented"] == pytest.approx(0.0, abs=1e-12)


def test_reflection_invariance_either_perspective():
    base, _ = compute_territorial_dominance(
        _scene(),
        xt=_ToyUniformXt(),  # type: ignore[arg-type]
        method="counterfactual",
        completion_model=_ConstCompletion(),  # type: ignore[arg-type]
    )
    mirrored, _ = compute_territorial_dominance(
        _mirror_scene(_scene()),
        xt=_ToyUniformXt(),  # type: ignore[arg-type]
        method="counterfactual",
        completion_model=_ConstCompletion(),  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(
        base["territory_xt_prevented"].to_numpy(),
        mirrored["territory_xt_prevented"].to_numpy(),
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        base["territory_xt_conceded"].to_numpy(),
        mirrored["territory_xt_conceded"].to_numpy(),
        rtol=0,
        atol=1e-9,
    )


def test_window_pools_sums_and_support_weighted_completion():
    two = pd.concat([_scene(game_id=1), _scene(game_id=2)], ignore_index=True)
    two["action_id"] = range(len(two))
    out, _ = compute_territorial_dominance(
        two,
        xt=_ToyUniformXt(),  # type: ignore[arg-type]
        method="counterfactual",
        completion_model=_ConstCompletion(),  # type: ignore[arg-type]
        window=[1, 2],
    )
    row = out.iloc[0]  # one pooled row per player
    assert int(row["territory_passes_aimed_into_hull"]) == 4  # 2 games x 2
    assert row["territory_xt_prevented"] == pytest.approx(0.12, abs=1e-12)  # 2 x 0.06
    assert row["territory_xt_conceded"] == pytest.approx(0.20, abs=1e-12)  # 2 x 0.10
    assert row["territory_mean_completion_faced"] == pytest.approx(0.6, abs=1e-12)  # sum c / sum passes
    assert row["territory_xt_prevented_rate"] == pytest.approx(0.12 / 4, abs=1e-12)
