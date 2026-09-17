"""Mechanism tests for ``_counterfactual`` -- purity + failed-pass target-recovery (TF-54b, Task 8).

Two behavioural tests that the §5.7 goldens in ``test_counterfactual_compute.py`` do NOT cover:

* ``test_counterfactual_rows_does_not_mutate_inputs`` -- ``counterfactual_rows`` is documented PURE
  (its docstring: "never mutates ``defs_grouped`` / ``passes_by_game`` / the caller's ``actions``").
  territory ships no ``add_*`` so it is outside ``tests/test_add_star_purity.py``'s registry; this is
  its no-mutation gate. Snapshot every input DataFrame/ndarray, call, assert byte-for-byte unchanged
  AND assert the call did real work (a modeled target, ADR-042 -- a no-op would pass purity vacuously).

* ``test_cone_estimator_recovers_hidden_end_from_both_sides`` -- the failed-pass target estimator (the
  cone-restricted, renormalized transition distribution ``q`` from ``destination_profiles``, spec §5.2)
  recovers a hidden true end better than the naive baselines, AND (the from-both-sides rule) a
  DEGENERATE cone does NOT. We take a COMPLETED pass with a known true end, synthesize a "death" with
  ``perturb_interception`` (Task 7) so the estimator sees only origin+death, then re-derive the
  cone-conditioned estimate from the SAME public seams ``_counterfactual`` uses (``destination_profiles``
  + ``_within_cone``), independent of the production per-defender loop.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.id_compat import canonical_id_series
from silly_kicks.territory import CounterfactualParams
from silly_kicks.territory._counterfactual import _within_cone, counterfactual_rows
from silly_kicks.territory._hull import build_trimmed_hull
from silly_kicks.xthreat import ExpectedThreat, destination_profiles, values_at_points
from silly_kicks.xthreat._grid import _get_flat_indexes

# perturb_interception lives in scripts/ (not importable as a package); load it by path.
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from _synthetic_interception import perturb_interception  # noqa: E402

_PASS = spadlconfig.actiontype_id["pass"]
_SHOT = spadlconfig.actiontype_id["shot"]
_OK = spadlconfig.result_id["success"]
_FAIL = spadlconfig.result_id["fail"]


def _nonuniform_xt(*, origin: tuple[float, float], seeded_targets: list[tuple[float, float, int]]) -> ExpectedThreat:
    """A NON-degenerate, NON-uniform fitted xT: a dense forward-pass grid (so every move cell has mass)
    + a spread of successful passes FROM ``origin`` toward ``seeded_targets`` (varying counts -> a
    non-uniform transition row on the origin cell) + shots/goals near goal (non-uniform xT). A uniform
    xT would make the cone-weighted centroid degenerate to the cone's geometric centre, defeating the
    recovery test -- the whole point is that ``q`` prefers the true target's direction."""
    rows: list[dict] = []

    def add(type_id: int, res: int, sx: float, sy: float, ex: float, ey: float) -> None:
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
    for tx, ty, cnt in seeded_targets:
        for _ in range(cnt):
            add(_PASS, _OK, origin[0], origin[1], tx, ty)
    for i, (sx, sy) in enumerate([(100, 34), (98, 30), (99, 38), (102, 34), (101, 28), (97, 40)]):
        add(_SHOT, _OK if i % 2 == 0 else _FAIL, sx, sy, 105, 34)
    df = pd.DataFrame(rows)
    df["action_id"] = range(len(df))
    return ExpectedThreat(l=16, w=12).fit(df)


class _ConstCompletion:
    """Duck-typed completion port: geometry-blind ``c == 0.6`` (matches the §5.7 toy)."""

    def predict_completion(self, ox, oy, tx, ty):
        return np.full(np.asarray(tx, dtype=float).shape, 0.6)


# --- Test 1: counterfactual_rows purity ------------------------------------------------------------------
def test_counterfactual_rows_does_not_mutate_inputs():
    """``counterfactual_rows`` is documented PURE. Build a real (hull, opponent-passes) scene, snapshot
    every input DataFrame/ndarray, call, and assert byte-for-byte equality after -- AND assert the call
    actually modeled a target (n_target_modeled >= 1), so purity is proven over a path that did work,
    not vacuously over a no-op."""
    fl = float(spadlconfig.field_length)
    fw = float(spadlconfig.field_width)

    xt = _nonuniform_xt(
        origin=(80.0, 34.0),
        seeded_targets=[(95, 25, 3), (95, 34, 5), (95, 43, 2), (101, 30, 1), (101, 38, 4), (104, 34, 2)],
    )

    # Defender team 1 (player 7): 3 own-half tackles forming a triangle hull. The reflected completed
    # end lands inside; the failed pass's +x death-cone selects hull zones -> a MODELED target.
    tackles = np.array([(2.0, 20.0), (2.0, 48.0), (14.0, 34.0)], dtype=float)
    hull = build_trimmed_hull(tackles, trim_fraction=0.70)
    assert hull is not None

    # Opponent (team 2) passes, prepared with the derived columns counterfactual_rows reads (mirrors the
    # `_compute._counterfactual_dispatch` contract): _completed / _forward / _xt_end / _g / _t.
    raw = [
        (88.0, 34.0, 95.0, 34.0, _OK),  # completed: reflects into the hull -> conceded
        (80.0, 34.0, 90.0, 34.0, _FAIL),  # failed: +x cone selects hull zones -> modeled target
    ]
    passes = pd.DataFrame(
        [
            {"game_id": 7, "team_id": 2, "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey, "result_id": res}
            for sx, sy, ex, ey, res in raw
        ]
    )
    passes["_xt_end"] = values_at_points(xt, passes["end_x"].to_numpy(), passes["end_y"].to_numpy())
    passes["_completed"] = (passes["result_id"] == _OK).to_numpy()
    passes["_forward"] = (passes["end_x"] - passes["start_x"]).to_numpy() > 0.0
    passes["_g"] = canonical_id_series(passes["game_id"])
    passes["_t"] = canonical_id_series(passes["team_id"])

    game_key = canonical_id_series(pd.Series([7]))[0]
    team_canon = canonical_id_series(pd.Series([1]))[0]
    passes_by_game: dict[object, pd.DataFrame] = {game_key: passes}
    def_xy = tackles.copy()
    groups = [(7, 7, team_canon, hull, def_xy, [game_key])]

    # Deep snapshots of every mutable input BEFORE the call: the opponent-pass frames, the def_xy array,
    # AND the hull object (its area/centroid + the underlying Delaunay vertex array) that the group carries.
    passes_snapshot = {k: v.copy(deep=True) for k, v in passes_by_game.items()}
    def_xy_snapshot = def_xy.copy()
    hull_snapshot = copy.deepcopy(hull)
    hull_area_snapshot = hull.area
    hull_centroid_snapshot = tuple(hull.centroid)
    hull_points_snapshot = np.array(hull._delaunay.points, copy=True)

    result = counterfactual_rows(
        groups,
        passes_by_game,
        xt=xt,  # type: ignore[arg-type]
        completion_model=_ConstCompletion(),  # type: ignore[arg-type]
        params=CounterfactualParams.default(),
        fl=fl,
        fw=fw,
        window=None,
    )

    # Return contract: a (rows, census) tuple.
    assert isinstance(result, tuple) and len(result) == 2
    rows, census = result
    assert isinstance(rows, list) and isinstance(census, dict)

    # Non-vacuity: the call did real work (a target was modeled), so purity is not proven over a no-op.
    assert census["n_target_modeled"] >= 1
    assert census["n_scored"] == 1

    # Purity: every input frame / ndarray / hull is byte-for-byte unchanged after the call.
    for k in passes_by_game:
        pd.testing.assert_frame_equal(passes_by_game[k], passes_snapshot[k])
    np.testing.assert_array_equal(def_xy, def_xy_snapshot)
    assert hull.area == hull_area_snapshot
    assert tuple(hull.centroid) == hull_centroid_snapshot
    np.testing.assert_array_equal(hull._delaunay.points, hull_points_snapshot)
    assert hull_snapshot.area == hull_area_snapshot  # the deep copy is an independent reference witness


# --- Test 2: cone-conditioned target recovery (from BOTH sides) ------------------------------------------
def _origin_zone_centre(xt: ExpectedThreat, origin: tuple[float, float]) -> np.ndarray:
    """The physical centre of ``origin``'s own transition cell -- the 'stay put' naive baseline."""
    centres = destination_profiles(xt, np.array([origin[0]]), np.array([origin[1]])).zone_centres
    cell = _get_flat_indexes(pd.Series([origin[0]]), pd.Series([origin[1]]), xt.l, xt.w).to_numpy()[0]
    return centres[cell]


def _cone_conditioned_estimate(
    xt: ExpectedThreat,
    origin: tuple[float, float],
    death: tuple[float, float],
    cone_deg: float,
) -> np.ndarray:
    """Re-derive the failed-pass target estimate from the SAME public seams ``_counterfactual`` uses
    (``destination_profiles`` + ``_within_cone``): select the transition zones inside the death-direction
    cone, renormalize their mass to ``q`` (spec §5.2), and return the q-weighted centroid. If the cone
    selects nothing (a degenerate cone collapsing to the death ray barely selects any zone centre), fall
    back to the death itself -- the honest 'no cone information available' answer."""
    prof = destination_profiles(xt, np.array([origin[0]]), np.array([origin[1]]))
    centres, probs = prof.zone_centres, prof.probabilities[0]
    in_cone = _within_cone((death[0] - origin[0], death[1] - origin[1]), centres, origin, cone_deg)
    sel = in_cone & (probs > 0)
    if not sel.any():
        return np.asarray(death, dtype=float)  # nothing selected -> death is the best available guess
    q = probs[sel] / probs[sel].sum()
    return (q[:, None] * centres[sel]).sum(axis=0)


def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def test_cone_estimator_recovers_hidden_end_from_both_sides():
    """The cone-conditioned target estimator recovers a hidden true end better than BOTH naive baselines
    -- (a) 'death = the synthesized intercept' and (b) the origin-zone centroid -- AND, from the other
    side, a DEGENERATE cone (``direction_cone_degrees ~= 1``, which barely selects anything / collapses
    to the death ray) does NOT beat the baseline it collapses onto."""
    rng = np.random.default_rng(20260917)  # seeded (unused directly, but pins any future stochastic knob)
    assert rng is not None

    origin = (60.0, 34.0)
    true_end = (75.0, 50.0)  # a seeded destination -> the origin cell has real transition mass there
    xt = _nonuniform_xt(
        origin=origin,
        seeded_targets=[(75, 25, 3), (75, 34, 5), (75, 43, 2), (82, 30, 1), (82, 40, 4), (88, 34, 2), (75, 50, 4)],
    )

    # Synthesize a death that HIDES the true end: intercepted at flight-fraction 0.5 AND rotated 20 deg
    # off the origin->end ray (so both the intended distance and direction are corrupted, Task 7).
    dx, dy = perturb_interception(origin, true_end, fraction=0.5, angle_offset_rad=np.radians(20.0))
    death = (float(dx), float(dy))

    # Non-vacuity: the perturbation actually moved the death away from the true end (it hides something).
    assert _dist(death, true_end) > 1.0

    baseline_death = _dist(death, true_end)
    baseline_origin_zone = _dist(_origin_zone_centre(xt, origin), true_end)

    # --- side 1: the real cone recovers the hidden end better than BOTH baselines --------------------
    est = _cone_conditioned_estimate(xt, origin, death, cone_deg=45.0)
    dist_est = _dist(est, true_end)
    tol = 1e-6  # a strict "closer than" margin; the effect is many metres, far above float noise
    assert dist_est + tol < baseline_death, (dist_est, baseline_death)
    assert dist_est + tol < baseline_origin_zone, (dist_est, baseline_origin_zone)

    # --- side 2 (from-both-sides): a degenerate cone does NOT beat the baseline it collapses onto -----
    # At ~1 deg the cone barely selects any zone centre, so the estimator falls back to the death ray;
    # its recovery is therefore NO BETTER than the naive "death = intercept" baseline (they coincide).
    est_degenerate = _cone_conditioned_estimate(xt, origin, death, cone_deg=1.0)
    dist_degenerate = _dist(est_degenerate, true_end)
    assert not (dist_degenerate + tol < baseline_death), (dist_degenerate, baseline_death)
    # And the degenerate estimate is strictly worse than the real 45 deg cone (the mechanism did matter).
    assert dist_degenerate > dist_est + tol, (dist_degenerate, dist_est)

    # --- side 2 (from-both-sides): a WRONG-DIRECTION cone does NOT beat the baselines either ----------
    # Negate the death vector so the 45 deg cone points OPPOSITE to the actual death; the mutation that
    # SHOULD break the mechanism must produce an estimate no better than the naive death baseline (this
    # proves side 1's positive assertion measures the cone DIRECTION, not an incidental property).
    wrong_death = (2.0 * origin[0] - death[0], 2.0 * origin[1] - death[1])  # origin - (death - origin)
    est_wrongdir = _cone_conditioned_estimate(xt, origin, wrong_death, cone_deg=45.0)
    dist_wrongdir = _dist(est_wrongdir, true_end)
    assert not (dist_wrongdir + tol < baseline_death), (dist_wrongdir, baseline_death)
