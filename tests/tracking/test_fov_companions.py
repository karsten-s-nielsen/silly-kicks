"""Both-sides FOV-observability companions for ``add_pressure_on_actor`` + ``add_packing`` (ADR-077).

Task 3 wires two more aggregators into the ``_fov_registry`` engine (Task 2). Each aggregator gains
an OPT-IN ``visible_area`` kwarg that appends ``<col>_observed_fraction`` / ``_observed_source``
companions for its region-based metric(s), reusing the ONE companion engine.

Every assertion is BOTH-sided (CLAUDE.md): the PRIMARY columns are byte-identical with and without
``visible_area`` (the companion path touches no metric value), and the COMPANION fraction MOVES with
the crop -- a full-pitch polygon observes the whole region (fraction ~1.0) while a partial crop that
cuts through the region yields a fraction strictly inside (0, 1) (the non-vacuity guard). The
region is the metric's ACTUAL ROI: the Andrienko directional OVAL (oriented toward the goal, NOT an
axis-aligned ellipse) and the packing full-height passer->receiver x-BAND.

Fixtures come from the committed SB360 paired fixture's Leg A builder (``tests/tracking/
_fov_fixtures.py`` -> ``tests/sb360/_fixture.py``), so ``actions`` / ``frames`` are produced by the
REAL ``snapshot_to_tracking_frames`` and link with the REAL linker -- freeze-frames are action-LTR,
which is exactly the frame the regions and the ``visible_area`` polygons share.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._fov_registry import _OBSERVABILITY_EXEMPT, _PACKING_REGION_COUNT_COLUMNS, RegionCtx
from silly_kicks.tracking._xt_gk import XtGkParams
from silly_kicks.tracking.features import (
    add_defensive_line,
    add_packing,
    add_player_influence,
    add_pressure_on_actor,
    add_team_shape,
    add_xt_gk,
)
from tests.tracking._fov_fixtures import tiny_actions, tiny_frames, tiny_visible_area

# Action 0 in the fixture: pass, actor at (52.5, 34.0), end (70.0, 40.0).
#   * Andrienko oval (default d_back=3, d_front=9, goal at x=105) spans x in [49.5, 61.5].
#   * Packing band spans x in [52.5, 70.0].
# A crop at x <= 55 cuts the oval; a crop at x <= 60 cuts the packing band -- both partial, not empty.
_A0 = 0

# The fixed-zone builders ignore ``(i, ctx)`` (fixed geometry), so tests pass ``None`` deliberately.
# ``cast`` is a no-op at runtime (still ``None``); it only satisfies the ``RegionCtx`` parameter type.
_NO_CTX = cast(RegionCtx, None)

_PRESSURE_COL = "pressure_on_actor__andrienko_oval"
_PRESSURE_FRAC = f"{_PRESSURE_COL}_observed_fraction"
_PRESSURE_SRC = f"{_PRESSURE_COL}_observed_source"

_PACKING_COLS = ("packing_made", "packing_net", "packing_goal_threat")


def _va(action_ids, polygon) -> pd.DataFrame:
    return pd.DataFrame({"action_id": list(action_ids), "polygon": [list(polygon)] * len(action_ids)})


def _full_pitch(action_ids) -> pd.DataFrame:
    return _va(action_ids, [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)])


def _left_crop(action_ids, right_x: float) -> pd.DataFrame:
    return _va(action_ids, [(0.0, 0.0), (right_x, 0.0), (right_x, 68.0), (0.0, 68.0)])


# ---------------------------------------------------------------------------
# add_pressure_on_actor
# ---------------------------------------------------------------------------
def test_pressure_companion_absent_without_visible_area():
    """The companion is conditional: omitting ``visible_area`` produces NO companion columns
    (the RED state -- the test below would KeyError on the un-wired aggregator)."""
    a, f = tiny_actions(), tiny_frames()
    base = add_pressure_on_actor(a, f)
    assert _PRESSURE_FRAC not in base.columns
    assert _PRESSURE_SRC not in base.columns


def test_pressure_companion_half_crop():
    a, f = tiny_actions(), tiny_frames()
    ids = list(a["action_id"])
    va_full = _full_pitch(ids)
    va_crop = _left_crop(ids, right_x=55.0)  # cuts action-0 oval (spans to x=61.5)

    full = add_pressure_on_actor(a, f, visible_area=va_full)
    left = add_pressure_on_actor(a, f, visible_area=va_crop)
    base = add_pressure_on_actor(a, f)

    # PRIMARY byte-identical: the companion path never touches the metric value.
    np.testing.assert_array_equal(full[_PRESSURE_COL].to_numpy(), base[_PRESSURE_COL].to_numpy())
    np.testing.assert_array_equal(left[_PRESSURE_COL].to_numpy(), base[_PRESSURE_COL].to_numpy())

    # COMPANION present, action 0 observed on both, and the region has area (source 'observed').
    assert full[_PRESSURE_SRC].iloc[_A0] == "observed"
    assert left[_PRESSURE_SRC].iloc[_A0] == "observed"

    # Full pitch observes the whole oval; the crop observes strictly less.
    full_frac = float(full[_PRESSURE_FRAC].iloc[_A0])
    left_frac = float(left[_PRESSURE_FRAC].iloc[_A0])
    assert full_frac == pytest.approx(1.0, abs=1e-9)
    assert left_frac < 1.0
    # Non-vacuity: the crop genuinely straddles the oval (a strict interior fraction).
    assert 0.0 < left_frac < 1.0


# ---------------------------------------------------------------------------
# add_packing
# ---------------------------------------------------------------------------
def test_packing_companion_absent_without_visible_area():
    a, f = tiny_actions(), tiny_frames()
    base = add_packing(a, f)
    for col in _PACKING_COLS:
        assert f"{col}_observed_fraction" not in base.columns
        assert f"{col}_observed_source" not in base.columns


def test_packing_companion_half_crop():
    a, f = tiny_actions(), tiny_frames()
    ids = list(a["action_id"])
    va_full = _full_pitch(ids)
    va_crop = _left_crop(ids, right_x=60.0)  # cuts action-0 band (spans x in [52.5, 70.0])

    full = add_packing(a, f, visible_area=va_full)
    left = add_packing(a, f, visible_area=va_crop)
    base = add_packing(a, f)

    # PRIMARY byte-identical for every region-count column (Int64/float; compare via to_numpy).
    for col in _PACKING_COLS:
        pd.testing.assert_series_equal(full[col], base[col], check_names=True)
        pd.testing.assert_series_equal(left[col], base[col], check_names=True)

    # All three counts SHARE one region (the passer->receiver x-band), so all three move together.
    for col in _PACKING_COLS:
        frac_col = f"{col}_observed_fraction"
        src_col = f"{col}_observed_source"
        assert full[src_col].iloc[_A0] == "observed"
        assert left[src_col].iloc[_A0] == "observed"
        full_frac = float(full[frac_col].iloc[_A0])
        left_frac = float(left[frac_col].iloc[_A0])
        assert full_frac == pytest.approx(1.0, abs=1e-9)
        assert left_frac < 1.0
        assert 0.0 < left_frac < 1.0

    # The band is x in [52.5, 70] cropped at x=60 -> exactly (60-52.5)/(70-52.5) observed.
    expected = (60.0 - 52.5) / (70.0 - 52.5)
    assert float(left[f"{_PACKING_COLS[0]}_observed_fraction"].iloc[_A0]) == pytest.approx(expected, abs=1e-9)


def test_packing_region_count_columns_are_real_add_packing_outputs():
    """A2 drift guard (anti-rot; partial by necessity): every name in
    ``_PACKING_REGION_COUNT_COLUMNS`` must be an ACTUAL emitted column of ``add_packing``, so a
    rename in ``add_packing`` that leaves the registry constant stale is caught. This catches only
    the rename direction -- a NEWLY-ADDED region-count column cannot be auto-detected (no structural
    is-region-count signal); a contributor must register it in the constant + ``_AGGREGATE_FOV_SENSITIVE``
    by hand (documented at the constant)."""
    out = add_packing(tiny_actions(), tiny_frames())
    assert set(_PACKING_REGION_COUNT_COLUMNS) <= set(out.columns), (
        f"stale packing region-count columns: {sorted(set(_PACKING_REGION_COUNT_COLUMNS) - set(out.columns))}"
    )
    # Non-vacuity: the constant is non-empty, so the subset assertion is not trivially true.
    assert _PACKING_REGION_COUNT_COLUMNS


# ---------------------------------------------------------------------------
# Task 4: aggregate-position companions on FIXED action-LTR pitch zones (ADR-077).
#
# Unlike the pressure oval / packing band (which are drawn from the action's own start/end), these
# zones are CONSTANT bands keyed only on the column's ROLE -- the acting team attacks x=105, so the
# defended end is fixed, and no per-action goal_map / team-id is consulted. The `visible_area`
# polygon shares that action-LTR frame (the only supplier, SB360, is action-LTR), which is exactly
# why a fixed zone is correct where a goal_map-keyed one would mis-orient away-possession actions.
# ---------------------------------------------------------------------------

_DL_COL = "defensive_line_x"  # DEFENDING team's line -> defended third [70, 105]
_DL_FRAC = f"{_DL_COL}_observed_fraction"
_DL_SRC = f"{_DL_COL}_observed_source"

# add_team_shape emits FOUR centroid columns -> TWO role companions (there is NO bare
# `team_shape_centroid` column). Attacking = acting team's OWN half [0, 52.5]; defending =
# opponent's OWN half [52.5, 105] -- opposite ends.
_TS_ATT = "team_shape_centroid_attacking"
_TS_DEF = "team_shape_centroid_defending"

_PI_COL = "off_ball_xt_team"  # ATTACKING/possession team -> attacking half [52.5, 105]
_PI_FRAC = f"{_PI_COL}_observed_fraction"
_PI_SRC = f"{_PI_COL}_observed_source"

_PI_PRIMARY_COLS = (
    "actor_reachable_area_m2",
    "off_ball_xt_team",
    "off_ball_xt_opponent",
    "off_ball_xt_diff",
    "reachable_area_team",
    "reachable_area_opponent",
    "reachable_area_diff",
)


def _fresh_xt():
    """A tiny fitted xT (deterministic) for add_player_influence."""
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


def _shift_outfield_x(frames: pd.DataFrame, dx: float = 20.0) -> pd.DataFrame:
    """Move every OUTFIELD player (non-GK, non-ball) by +dx in x (clamped to the pitch), keeping
    GK and ball rows fixed. A player-DERIVED zone (a bbox around the observed outfielders) would
    move under this shift; the FIXED geometric zone must not -- the N2 frame-independence guard."""
    f = frames.copy()
    outfield = ~f["is_ball"].astype(bool) & ~f["is_goalkeeper"].astype(bool)
    f.loc[outfield, "x"] = (f.loc[outfield, "x"] + dx).clip(0.0, 105.0)
    return f


# ---------------------------------------------------------------------------
# End-correctness: each fixed builder sits on the CORRECT action-LTR end.
# ---------------------------------------------------------------------------
def test_zones_sit_on_the_correct_action_ltr_ends():
    """The fixed builders return the right pitch band (no goal_map -- the acting team attacks
    x=105, so the ends are determined). Replaces the goal_map `test_zone_sits_on_the_correct_end`
    / `test_unresolved_end_yields_no_region` of the pre-correction brief -- the fixed design has no
    unresolved-end case."""
    from silly_kicks.tracking._fov_registry import (
        attacking_half_region,
        attacking_own_half_region,
        defended_third_region,
        defending_own_half_region,
    )

    # The builders ignore (i, ctx) entirely -- fixed geometry, never player-derived.
    z = defended_third_region(0, _NO_CTX)  # DEFENDING team's defended third
    assert z[:, 0].min() == 70.0 and z[:, 0].max() == 105.0
    z = attacking_own_half_region(0, _NO_CTX)  # acting team's own half (LOW end)
    assert z[:, 0].min() == 0.0 and z[:, 0].max() == 52.5
    z = defending_own_half_region(0, _NO_CTX)  # opponent's own half (HIGH end)
    assert z[:, 0].min() == 52.5 and z[:, 0].max() == 105.0
    z = attacking_half_region(0, _NO_CTX)  # acting team's attacking half (HIGH end)
    assert z[:, 0].min() == 52.5 and z[:, 0].max() == 105.0
    # Full pitch height on every band, and convex (Sutherland-Hodgman clips a convex ROI only).
    for builder in (defended_third_region, attacking_own_half_region, attacking_half_region):
        band = builder(0, _NO_CTX)
        assert band[:, 1].min() == 0.0 and band[:, 1].max() == 68.0


# ---------------------------------------------------------------------------
# add_defensive_line
# ---------------------------------------------------------------------------
def test_defensive_line_companion_absent_without_visible_area():
    a, f = tiny_actions(), tiny_frames()
    base = add_defensive_line(a, f)
    assert _DL_FRAC not in base.columns
    assert _DL_SRC not in base.columns


def test_defensive_line_companion_half_crop():
    a, f = tiny_actions(), tiny_frames()
    ids = list(a["action_id"])
    va_full = _full_pitch(ids)
    va_crop = _left_crop(ids, right_x=85.0)  # cuts the defended-third band [70, 105]

    full = add_defensive_line(a, f, visible_area=va_full)
    left = add_defensive_line(a, f, visible_area=va_crop)
    base = add_defensive_line(a, f)

    # PRIMARY byte-identical: the companion path never touches the six metric columns.
    for col in ("defensive_line_x", "back_line_high_x", "compactness_x", "lateral_width", "max_lateral_gap"):
        np.testing.assert_array_equal(full[col].to_numpy(), base[col].to_numpy())
        np.testing.assert_array_equal(left[col].to_numpy(), base[col].to_numpy())
    pd.testing.assert_series_equal(full["back_n_count"], base["back_n_count"])
    pd.testing.assert_series_equal(left["back_n_count"], base["back_n_count"])

    # COMPANION present + observed on both.
    assert full[_DL_SRC].iloc[_A0] == "observed"
    assert left[_DL_SRC].iloc[_A0] == "observed"

    # Full pitch observes the whole defended third; the crop observes strictly less.
    full_frac = float(full[_DL_FRAC].iloc[_A0])
    left_frac = float(left[_DL_FRAC].iloc[_A0])
    assert full_frac == pytest.approx(1.0, abs=1e-9)
    assert left_frac < 1.0
    # Non-vacuity: the crop genuinely straddles the [70, 105] band (a strict interior fraction).
    assert 0.0 < left_frac < 1.0
    # The band is x in [70, 105] cropped at x=85 -> exactly (85-70)/(105-70) observed.
    assert left_frac == pytest.approx((85.0 - 70.0) / (105.0 - 70.0), abs=1e-9)


def test_defensive_line_companion_invariant_to_outfield_shift():
    """N2 frame-independence (P4): the companion fraction is a FIXED geometric zone, so moving the
    observed outfield players cannot change it -- proving the zone is NOT drawn around the observed
    outfielders (the S1 regression the tautological `pitch_zone()-twice` test would miss). Keeps GK
    + ball rows fixed."""
    a = tiny_actions()
    f = tiny_frames()
    f2 = _shift_outfield_x(f, dx=20.0)
    ids = list(a["action_id"])
    va = _left_crop(ids, right_x=85.0)

    base = add_defensive_line(a, f, visible_area=va)
    shifted = add_defensive_line(a, f2, visible_area=va)

    # Non-vacuity: the shift genuinely moved outfield positions.
    assert not f["x"].equals(f2["x"])
    # Counterfactual non-vacuity (CLAUDE.md): the shift genuinely moved the PRIMARY metric on a
    # finite row -- so the companion being UNCHANGED is a real frame-independence result (zone is
    # geometry, not player-derived), not a computation that silently produced nothing. Compare only
    # rows finite on BOTH sides, since NaN != NaN would make a bare `.any()` vacuously true.
    b = base["defensive_line_x"].to_numpy()
    s = shifted["defensive_line_x"].to_numpy()
    finite = np.isfinite(b) & np.isfinite(s)
    assert finite.any() and (b[finite] != s[finite]).any(), (
        "the +20 outfield shift did not move defensive_line_x -- the frame-independence assertion is vacuous"
    )
    # The fixed zone is unaffected -- companion fraction UNCHANGED (NaNs compare equal here).
    np.testing.assert_array_equal(base[_DL_FRAC].to_numpy(), shifted[_DL_FRAC].to_numpy())
    assert list(base[_DL_SRC]) == list(shifted[_DL_SRC])


# ---------------------------------------------------------------------------
# add_team_shape (R2: four real columns -> two role-keyed companions)
# ---------------------------------------------------------------------------
def test_team_shape_companion_absent_without_visible_area():
    a, f = tiny_actions(), tiny_frames()
    base = add_team_shape(a, f)
    for role in (_TS_ATT, _TS_DEF):
        assert f"{role}_observed_fraction" not in base.columns
        assert f"{role}_observed_source" not in base.columns


def test_team_shape_two_role_companions():
    a, f = tiny_actions(), tiny_frames()
    ids = list(a["action_id"])
    va_crop = _left_crop(ids, right_x=60.0)  # one-sided: fully covers [0,52.5], clips [52.5,105]

    base = add_team_shape(a, f)
    out = add_team_shape(a, f, visible_area=va_crop)

    # PRIMARY byte-identical for all 20 team-shape columns.
    ts_cols = [c for c in base.columns if c.startswith("team_shape_")]
    assert len(ts_cols) == 20
    for col in ts_cols:
        np.testing.assert_array_equal(out[col].to_numpy(), base[col].to_numpy())

    # Two role companions present; the nonexistent bare-column companion is ABSENT.
    assert f"{_TS_ATT}_observed_fraction" in out.columns  # annotates x/y_attacking
    assert f"{_TS_DEF}_observed_fraction" in out.columns  # annotates x/y_defending
    assert "team_shape_centroid_observed_fraction" not in out.columns  # the nonexistent-column bug

    # The two roles' own halves are on OPPOSITE ends, so a one-sided crop observes them differently.
    att_frac = float(out[f"{_TS_ATT}_observed_fraction"].iloc[_A0])
    def_frac = float(out[f"{_TS_DEF}_observed_fraction"].iloc[_A0])
    assert out[f"{_TS_ATT}_observed_source"].iloc[_A0] == "observed"
    assert out[f"{_TS_DEF}_observed_source"].iloc[_A0] == "observed"
    assert att_frac != def_frac
    # attacking own half [0, 52.5] fully inside the x<=60 crop -> 1.0; defending own half
    # [52.5, 105] clipped -> (60-52.5)/(105-52.5).
    assert att_frac == pytest.approx(1.0, abs=1e-9)
    assert def_frac == pytest.approx((60.0 - 52.5) / (105.0 - 52.5), abs=1e-9)


# ---------------------------------------------------------------------------
# add_player_influence
# ---------------------------------------------------------------------------
def test_player_influence_companion_absent_without_visible_area():
    a, f = tiny_actions(), tiny_frames()
    base = add_player_influence(a, f, _fresh_xt())
    assert _PI_FRAC not in base.columns
    assert _PI_SRC not in base.columns


def test_player_influence_companion_half_crop():
    a, f = tiny_actions(), tiny_frames()
    ids = list(a["action_id"])
    va_full = _full_pitch(ids)
    va_crop = _left_crop(ids, right_x=85.0)  # cuts the attacking-half band [52.5, 105]
    xt = _fresh_xt()

    full = add_player_influence(a, f, xt, visible_area=va_full)
    left = add_player_influence(a, f, xt, visible_area=va_crop)
    base = add_player_influence(a, f, xt)

    # PRIMARY byte-identical for every player-influence column.
    for col in _PI_PRIMARY_COLS:
        np.testing.assert_array_equal(full[col].to_numpy(), base[col].to_numpy())
        np.testing.assert_array_equal(left[col].to_numpy(), base[col].to_numpy())

    # Only off_ball_xt_team is companioned (the other columns are not region-based counts).
    assert "off_ball_xt_opponent_observed_fraction" not in full.columns
    assert "reachable_area_team_observed_fraction" not in full.columns

    assert full[_PI_SRC].iloc[_A0] == "observed"
    assert left[_PI_SRC].iloc[_A0] == "observed"

    full_frac = float(full[_PI_FRAC].iloc[_A0])
    left_frac = float(left[_PI_FRAC].iloc[_A0])
    assert full_frac == pytest.approx(1.0, abs=1e-9)
    assert left_frac < 1.0
    assert 0.0 < left_frac < 1.0
    # The band is x in [52.5, 105] cropped at x=85 -> exactly (85-52.5)/(105-52.5) observed.
    assert left_frac == pytest.approx((85.0 - 52.5) / (105.0 - 52.5), abs=1e-9)


# ---------------------------------------------------------------------------
# Task 4: provenance-token coverage at the three aggregate-position seams (ADR-077).
#
# These fixed-zone builders always return a region, so the only non-``observed`` tokens the shared
# engine can emit here are ``unlinked`` (the action links to no frame) and ``no_polygon`` (the action
# is linked but its ``visible_area`` row is absent). Both yield a NaN fraction. The three seams
# (add_defensive_line / add_team_shape / add_player_influence) run through the shared engine but were
# untested for these tokens.
# ---------------------------------------------------------------------------
_NO_POLYGON_AID = 3  # action 3 is deliberately OMITTED from tiny_visible_area() -> no_polygon
_UNLINK_AID = 0  # forced unlinked via a NaN frame_id in the passed links


def _links_one_unlinked(actions: pd.DataFrame, frames: pd.DataFrame, unlink_aid: int) -> pd.DataFrame:
    """Real link pointers with ONE action forced unlinked (its ``frame_id`` -> NA)."""
    from silly_kicks.tracking import link_actions_to_frames

    links = link_actions_to_frames(actions, frames)[0].copy()
    links["frame_id"] = links["frame_id"].astype("Int64")
    links.loc[links["action_id"] == unlink_aid, "frame_id"] = pd.NA
    return links


def _row(out: pd.DataFrame, aid: int) -> int:
    return int(np.flatnonzero(out["action_id"].to_numpy() == aid)[0])


def test_defensive_line_companion_unlinked_and_no_polygon():
    a, f = tiny_actions(), tiny_frames()
    ids = list(a["action_id"])

    # UNLINKED: the forced-unlinked action -> 'unlinked', fraction NaN (full-pitch polygon present).
    unl = add_defensive_line(a, f, links=_links_one_unlinked(a, f, _UNLINK_AID), visible_area=_full_pitch(ids))
    r = _row(unl, _UNLINK_AID)
    assert unl[_DL_SRC].iloc[r] == "unlinked"
    assert np.isnan(unl[_DL_FRAC].iloc[r])

    # NO_POLYGON: action 3 is linked but omitted from tiny_visible_area() -> 'no_polygon', fraction NaN.
    nop = add_defensive_line(a, f, visible_area=tiny_visible_area())
    r3 = _row(nop, _NO_POLYGON_AID)
    assert nop[_DL_SRC].iloc[r3] == "no_polygon"
    assert np.isnan(nop[_DL_FRAC].iloc[r3])


def test_team_shape_companion_unlinked_and_no_polygon():
    a, f = tiny_actions(), tiny_frames()
    ids = list(a["action_id"])

    unl = add_team_shape(a, f, links=_links_one_unlinked(a, f, _UNLINK_AID), visible_area=_full_pitch(ids))
    r = _row(unl, _UNLINK_AID)
    for role in (_TS_ATT, _TS_DEF):  # both role companions carry the same link-level verdict
        assert unl[f"{role}_observed_source"].iloc[r] == "unlinked"
        assert np.isnan(unl[f"{role}_observed_fraction"].iloc[r])

    nop = add_team_shape(a, f, visible_area=tiny_visible_area())
    r3 = _row(nop, _NO_POLYGON_AID)
    for role in (_TS_ATT, _TS_DEF):
        assert nop[f"{role}_observed_source"].iloc[r3] == "no_polygon"
        assert np.isnan(nop[f"{role}_observed_fraction"].iloc[r3])


def test_player_influence_companion_unlinked_and_no_polygon():
    a, f = tiny_actions(), tiny_frames()
    ids = list(a["action_id"])
    xt = _fresh_xt()

    unl = add_player_influence(a, f, xt, links=_links_one_unlinked(a, f, _UNLINK_AID), visible_area=_full_pitch(ids))
    r = _row(unl, _UNLINK_AID)
    assert unl[_PI_SRC].iloc[r] == "unlinked"
    assert np.isnan(unl[_PI_FRAC].iloc[r])

    nop = add_player_influence(a, f, xt, visible_area=tiny_visible_area())
    r3 = _row(nop, _NO_POLYGON_AID)
    assert nop[_PI_SRC].iloc[r3] == "no_polygon"
    assert np.isnan(nop[_PI_FRAC].iloc[r3])


# ---------------------------------------------------------------------------
# Task 5: add_xt_gk method-dispatched pressure-ROI companions (ADR-077, T2/N3/M1).
#
# Unlike the aggregators above, the pressure ROI is NOT drawn from the action's raw start/end: it
# is centred on the RESOLVED GK origin (``compute_xt_gk`` feeds ``pressure_on_actor``
# ``sub_for_pressure["start_x"] = origin_x``), and the SHAPE is method-dispatched -- the Andrienko
# directional oval for ``pressure_method == "andrienko_oval"``, the Link effective-support disk for
# ``"link_zones"``, and NO region (degenerate_region) for the velocity-derived ``"bekkers_pi"``.
# Both pressure-bearing columns (``xt_gk_pressure`` = rho and ``xt_gk_pev``) share the region; the
# composite ``xt_gk`` is EXEMPT (M1). The fixture's GK-distribution actions are action 0 (a pass by
# the HOME keeper, player 10) and action 3 (a goalkick); action 3 is the primary scored row here.
# ---------------------------------------------------------------------------

_XT_GK_PRESS_FRAC = "xt_gk_pressure_observed_fraction"
_XT_GK_PRESS_SRC = "xt_gk_pressure_observed_source"
_XT_GK_PEV_FRAC = "xt_gk_pev_observed_fraction"
_XT_GK_PEV_SRC = "xt_gk_pev_observed_source"
_XT_GK_OUTPUT_COLS = ("xt_gk_base", "xt_gk_pev", "xt_gk_rav", "xt_gk_dzv", "xt_gk_pressure", "xt_gk")
_GOALKICK_AID = 3  # the fixture's goalkick action (a GK distribution -> xt_gk scores it)


def _fitted_xt_gk_grid():
    """A tiny non-zero fitted xT (deterministic) -- ``compute_xt_gk`` requires a fitted grid."""
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


def _gk_row(out: pd.DataFrame) -> int:
    """Positional index of the goalkick action (``action_id == 3``)."""
    return int(np.flatnonzero(out["action_id"].to_numpy() == _GOALKICK_AID)[0])


def test_xt_gk_companion_absent_without_visible_area():
    """Conditional: omitting ``visible_area`` produces NO xt_gk companion columns."""
    a, f = tiny_actions(), tiny_frames()
    base = add_xt_gk(a, f, _fitted_xt_gk_grid())
    for col in (_XT_GK_PRESS_FRAC, _XT_GK_PRESS_SRC, _XT_GK_PEV_FRAC, _XT_GK_PEV_SRC):
        assert col not in base.columns


def test_xt_gk_primary_byte_identical():
    """The companion path never touches the six xt_gk value columns (ADR-009, no VAEP retrain)."""
    a, f = tiny_actions(), tiny_frames()
    ids = list(a["action_id"])
    base = add_xt_gk(a, f, _fitted_xt_gk_grid())
    withva = add_xt_gk(a, f, _fitted_xt_gk_grid(), visible_area=_full_pitch(ids))
    for col in _XT_GK_OUTPUT_COLS:
        # assert_array_equal treats NaN == NaN as equal (out-of-scope rows are NaN).
        np.testing.assert_array_equal(withva[col].to_numpy(), base[col].to_numpy())


def test_xt_gk_andrienko_companion_present_and_populated():
    """Default (andrienko_oval): both pressure-bearing companions present + populated; a crop that
    straddles the directional oval yields a strict-interior fraction."""
    a, f = tiny_actions(), tiny_frames()
    xt = _fitted_xt_gk_grid()
    ids = list(a["action_id"])

    full = add_xt_gk(a, f, xt, visible_area=_full_pitch(ids))
    for col in (_XT_GK_PRESS_FRAC, _XT_GK_PRESS_SRC, _XT_GK_PEV_FRAC, _XT_GK_PEV_SRC):
        assert col in full.columns

    r = _gk_row(full)
    # The goalkick is scored (finite resolved origin) -> the oval is a real region, fully observed
    # by the full-pitch polygon.
    assert full[_XT_GK_PRESS_SRC].iloc[r] == "observed"
    assert full[_XT_GK_PEV_SRC].iloc[r] == "observed"
    assert float(full[_XT_GK_PRESS_FRAC].iloc[r]) == pytest.approx(1.0, abs=1e-9)
    assert float(full[_XT_GK_PEV_FRAC].iloc[r]) == pytest.approx(1.0, abs=1e-9)

    # Crop through the oval: it is centred on the RESOLVED GK origin and extends d_front=9 toward the
    # goal (+x), so a crop at origin_x + 4 keeps the back half and cuts the front.
    ox = float(full["xt_gk_origin_x"].iloc[r])
    left = add_xt_gk(a, f, xt, visible_area=_left_crop(ids, right_x=ox + 4.0))
    lf = float(left[_XT_GK_PRESS_FRAC].iloc[r])
    assert left[_XT_GK_PRESS_SRC].iloc[r] == "observed"
    assert 0.0 < lf < 1.0  # non-vacuity: the crop genuinely straddles the oval
    # xt_gk_pev shares the SAME pressure region -> identical fraction.
    assert float(left[_XT_GK_PEV_FRAC].iloc[r]) == pytest.approx(lf, abs=1e-12)


def test_xt_gk_link_zones_companion_present_and_populated():
    """N3: on ``pressure_method='link_zones'`` the companion is present-and-populated (a real
    fraction < 1 under crop), NOT silently absent -- the Link effective-support disk is a genuine
    convex, non-degenerate region."""
    a, f = tiny_actions(), tiny_frames()
    xt = _fitted_xt_gk_grid()
    ids = list(a["action_id"])
    params = XtGkParams(pressure_method="link_zones")

    full = add_xt_gk(a, f, xt, params=params, visible_area=_full_pitch(ids))
    r = _gk_row(full)
    assert full[_XT_GK_PRESS_SRC].iloc[r] == "observed"
    assert float(full[_XT_GK_PRESS_FRAC].iloc[r]) == pytest.approx(1.0, abs=1e-9)

    # The link support disk has radius max(r_hoz, r_lz, r_hz) = 4 m; a crop at origin_x + 1.5 cuts it.
    ox = float(full["xt_gk_origin_x"].iloc[r])
    left = add_xt_gk(a, f, xt, params=params, visible_area=_left_crop(ids, right_x=ox + 1.5))
    frac = left[_XT_GK_PRESS_FRAC]
    assert frac.notna().any() and (frac.dropna() < 1.0).any()  # present + populated, NOT absent
    assert 0.0 < float(frac.iloc[r]) < 1.0


def test_xt_gk_bekkers_no_companion():
    """``bekkers_pi`` is velocity-derived (honest-NaN on these freeze frames) with no fixed spatial
    ROI -> the companion is ``degenerate_region`` / NaN, never a fabricated fraction."""
    a, f = tiny_actions(), tiny_frames()
    ids = list(a["action_id"])
    out = add_xt_gk(
        a, f, _fitted_xt_gk_grid(), params=XtGkParams(pressure_method="bekkers_pi"), visible_area=_full_pitch(ids)
    )

    r = _gk_row(out)
    # The companion columns still EXIST (the aggregator was asked to companion), but every value is
    # degenerate_region -- there is no region to observe.
    assert out[_XT_GK_PRESS_SRC].iloc[r] == "degenerate_region"
    assert out[_XT_GK_PEV_SRC].iloc[r] == "degenerate_region"
    assert np.isnan(out[_XT_GK_PRESS_FRAC].iloc[r])
    assert np.isnan(out[_XT_GK_PEV_FRAC].iloc[r])
    # No scored row escapes the degenerate verdict for bekkers.
    assert (out[_XT_GK_PRESS_SRC] == "degenerate_region").all()


def test_no_composite_xt_gk_fraction():
    """M1: the composite ``xt_gk`` gets NO ``xt_gk_observed_fraction`` -- it mixes a region-dependent
    term with the GK-geometry base/rav/dzv, so there is no honest single fraction."""
    a, f = tiny_actions(), tiny_frames()
    ids = list(a["action_id"])
    out = add_xt_gk(a, f, _fitted_xt_gk_grid(), visible_area=_full_pitch(ids))
    assert "xt_gk_observed_fraction" not in out.columns
    assert "xt_gk_observed_source" not in out.columns


def test_xt_gk_in_exempt():
    """M1: the composite is registered EXEMPT (so the Task 8 completeness gate accepts it)."""
    assert "xt_gk" in _OBSERVABILITY_EXEMPT
    assert isinstance(_OBSERVABILITY_EXEMPT["xt_gk"], str) and _OBSERVABILITY_EXEMPT["xt_gk"]


# ---------------------------------------------------------------------------
# Task 6: add_defensive_credit mode-aware corridor/disk rollup companion (ADR-077).
#
# ``add_defensive_credit`` emits ONE per-action rollup companion for the WHOLE credit family (a
# CUSTOM path, not the generic per-column engine): ``defensive_credit_observed_fraction`` /
# ``_observed_source``. Each defending credit's region is built BY MODE -- a proximity DISK
# (nearest / all_within / all_within_beyond_nearest / nearest_fallback), the shot->goal CORRIDOR
# (lane), or NO region (anchor_actor, event-resolved) -- and the per-action fraction is the credit-
# MAGNITUDE-weighted mean over the region-bearing OBSERVED credits (P5).
# ---------------------------------------------------------------------------
from silly_kicks._polygon import is_convex  # noqa: E402 -- grouped with the Task 6 block
from silly_kicks.tracking import add_defensive_credit  # noqa: E402
from silly_kicks.tracking._fov_registry import (  # noqa: E402
    _CUSTOM_COMPANION_COVERS,
    _NO_REGION,
    _rollup_credit_observed_fraction,
    defensive_credit_region_for_mode,
)
from silly_kicks.tracking.defensive_credit._params import RESOLUTION_VALUES  # noqa: E402
from tests.tracking._defensive_credit_fixtures import frame_with_defender, one_action  # noqa: E402

_DC_FRAC = "defensive_credit_observed_fraction"
_DC_SRC = "defensive_credit_observed_source"
#: The default ``DefensiveCreditParams`` lane geometry (kept in sync via a drift guard below).
_LANE_KW = {"lane_cone_width_factor": 0.2, "lane_max_t": 0.9, "lane_min_half_width_m": 1.0}


def test_custom_covers_declared_at_module_load():
    """The rollup covers exactly the three SB360 ``region_support`` credit columns (net/minus/n);
    ``defensive_credit_plus`` is ``no_support`` -> NOT covered. Declared at MODULE LOAD so Task 8
    reads a populated set without running the aggregator."""
    assert _CUSTOM_COMPANION_COVERS == {
        "defensive_credit_net",
        "defensive_credit_minus",
        "n_defensive_credits",
    }


def test_fov_registry_mode_tokens_match_resolution_vocabulary():
    """Drift guard: the tokens the neutral ``_fov_registry`` dispatches on ARE the defensive-credit
    resolution vocabulary (it keeps them as literals to avoid importing ``defensive_credit``)."""
    from silly_kicks.tracking import _fov_registry as R

    assert R._DC_MODE_ANCHOR_ACTOR in RESOLUTION_VALUES
    assert R._DC_MODE_LANE in RESOLUTION_VALUES


def test_defensive_credit_region_for_mode():
    # anchor_actor -> _NO_REGION (event-resolved; P2 identity sentinel).
    assert (
        defensive_credit_region_for_mode("anchor_actor", origin_x=95.0, origin_y=34.0, region_radius=3.0, **_LANE_KW)
        is _NO_REGION
    )
    # every proximity mode -> a convex inscribed disk.
    for mode in ("nearest", "all_within", "all_within_beyond_nearest", "nearest_fallback"):
        disk = defensive_credit_region_for_mode(mode, origin_x=95.0, origin_y=34.0, region_radius=3.0, **_LANE_KW)
        assert disk is not _NO_REGION
        assert isinstance(disk, np.ndarray)
        assert is_convex(disk)
    # lane -> a convex 4-corner corridor trapezoid (radius unused / NaN).
    lane = defensive_credit_region_for_mode(
        "lane", origin_x=90.0, origin_y=30.0, region_radius=float("nan"), **_LANE_KW
    )
    assert lane is not _NO_REGION
    assert isinstance(lane, np.ndarray)
    assert is_convex(lane) and len(lane) == 4
    # degenerate: a non-finite origin, or a disk mode with a non-finite radius -> _NO_REGION.
    assert (
        defensive_credit_region_for_mode("nearest", origin_x=float("nan"), origin_y=34.0, region_radius=3.0, **_LANE_KW)
        is _NO_REGION
    )
    assert (
        defensive_credit_region_for_mode(
            "nearest", origin_x=95.0, origin_y=34.0, region_radius=float("nan"), **_LANE_KW
        )
        is _NO_REGION
    )


def test_rollup_is_magnitude_weighted():
    """P5, the pinned formula: two region-bearing credits, magnitude 3.0 @ observed_fraction 0.4 and
    1.0 @ 1.0 -> weighted (3*0.4 + 1*1.0)/(3+1) = 0.55; an anchor_actor credit (region_bearing=False,
    magnitude 5.0) is excluded from BOTH sums -- were it counted, the answer would move."""
    obs = [
        (3.0, 0.4, "observed", True),
        (1.0, 1.0, "observed", True),
        (5.0, float("nan"), "degenerate_region", False),  # anchor_actor -> excluded from both sums
    ]
    frac, src = _rollup_credit_observed_fraction(obs)
    assert round(float(frac), 4) == 0.55
    assert src == "observed"


def test_rollup_all_zero_magnitude_observed_is_degenerate_not_observed():
    """observed => finite-fraction contract: region-bearing credits that were OBSERVED but ALL carry
    magnitude 0.0 have weight-sum 0, so there is no honest weighted mean. The rollup must NOT fall
    through to ``(NaN, 'observed')`` -- an ``observed`` source with a NaN fraction is a silent null --
    but return ``(NaN, 'degenerate_region')``. Landed RED (the old fall-through returned
    ``region_bearing[0][2] == 'observed'``)."""
    obs = [
        (0.0, 0.4, "observed", True),
        (0.0, 1.0, "observed", True),
    ]
    frac, src = _rollup_credit_observed_fraction(obs)
    assert np.isnan(frac)
    assert src == "degenerate_region"


def test_rollup_anchor_only_is_nan_never_one():
    """An action whose region-bearing set is EMPTY (only anchor_actor credits, or none) -> NaN +
    degenerate_region, never a spurious 1.0."""
    frac, src = _rollup_credit_observed_fraction([(5.0, float("nan"), "degenerate_region", False)])
    assert np.isnan(frac) and src == "degenerate_region"
    frac0, src0 = _rollup_credit_observed_fraction([])
    assert np.isnan(frac0) and src0 == "degenerate_region"


def _dc_actions() -> pd.DataFrame:
    """Two actions: a0 = failed pass by team 10 with a nearby opponent (a NEAREST presser) AND an
    opponent recovery (an ANCHOR_ACTOR debit) -> a mode MIX in one action; a1 = failed pass by
    team 20 far from any defender -> NO defending credit."""
    a0 = one_action(
        action_id=0,
        type_name="pass",
        result_name="fail",
        start_x=95.0,
        start_y=34.0,
        team_id=10,
        player_id=5,
        time_seconds=50.0,
    )
    a1 = one_action(
        action_id=1,
        type_name="pass",
        result_name="fail",
        start_x=50.0,
        start_y=34.0,
        team_id=20,
        player_id=900,
        time_seconds=60.0,
    )
    a = pd.concat([a0, a1], ignore_index=True)
    a["shot_blocked"] = pd.array([pd.NA, pd.NA], dtype="boolean")
    a["cross_blocked"] = pd.array([pd.NA, pd.NA], dtype="boolean")
    a["shot_on_target_derived"] = pd.array([pd.NA, pd.NA], dtype="boolean")  # present -> no TF-48 fallback
    a["xg"] = [np.nan, np.nan]
    return a


def _dc_frames() -> pd.DataFrame:
    f0 = frame_with_defender(
        action_time=50.0,
        acting_team_id=10,
        defender_team_id=20,
        defender_x=96.0,
        defender_y=34.0,
        frame_id=500,
        home_team_id=10,
    )
    f1 = frame_with_defender(
        action_time=60.0,
        acting_team_id=20,
        defender_team_id=10,
        defender_x=90.0,
        defender_y=34.0,
        frame_id=501,
        home_team_id=10,
    )
    return pd.concat([f0, f1], ignore_index=True)


def test_defensive_credit_companion_absent_without_visible_area(fitted_xt):
    out = add_defensive_credit(_dc_actions(), _dc_frames(), xg_column="xg", xt=fitted_xt)
    assert _DC_FRAC not in out.columns
    assert _DC_SRC not in out.columns


def test_defensive_credit_primary_byte_identical(fitted_xt):
    a, f = _dc_actions(), _dc_frames()
    ids = list(a["action_id"])
    base = add_defensive_credit(a, f, xg_column="xg", xt=fitted_xt)
    withva = add_defensive_credit(a, f, xg_column="xg", xt=fitted_xt, visible_area=_left_crop(ids, right_x=95.0))
    for col in ("defensive_credit_net", "defensive_credit_plus", "defensive_credit_minus", "n_defensive_credits"):
        pd.testing.assert_series_equal(withva[col], base[col], check_names=True)


def test_defensive_credit_mode_aware_rollup(fitted_xt):
    a, f = _dc_actions(), _dc_frames()
    ids = list(a["action_id"])
    # The crop x<=95 cuts the a0 presser DISK (centre (95, 34), radius 3.0 -> x in [92, 98]).
    out = add_defensive_credit(a, f, xg_column="xg", xt=fitted_xt, visible_area=_left_crop(ids, right_x=95.0))
    frac = out[_DC_FRAC]

    # A region-bearing credit's region is partly outside the FOV -> a strict-interior fraction.
    assert (frac.dropna() < 1.0).any()

    a0 = int(np.flatnonzero(out["action_id"].to_numpy() == 0)[0])
    assert out[_DC_SRC].iloc[a0] == "observed"
    assert 0.0 < float(frac.iloc[a0]) < 1.0
    # a0 carries a MIX: the nearest presser (region-bearing) + a recovery anchor_actor credit that is
    # EXCLUDED -- so n >= 2 while the fraction reflects only the region-bearing presser.
    assert int(out["n_defensive_credits"].iloc[a0]) >= 2

    # a1 has NO region-bearing defending credit -> NaN, never a spurious 1.0.
    a1 = int(np.flatnonzero(out["action_id"].to_numpy() == 1)[0])
    assert np.isnan(float(frac.iloc[a1]))
    assert out[_DC_SRC].iloc[a1] == "degenerate_region"
