"""SB360 verdicts -- frame-consuming entry points OUTSIDE ``tracking.__all__``.

Observations are TRANSCRIBED FROM EXECUTION; only a human writes an adjudication or rationale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import silly_kicks.spadl as spadl
from silly_kicks.gkdv import GkdvParams, build_ghost_frames
from silly_kicks.id_compat import canonical_id, ids_equal, same_id
from silly_kicks.xtgk import DeltaV, PressureLevels, State, compute_xt_gk_v2
from tests.sb360._registry import AxisVerdict, _entry


def _call_restart_coordinates(actions, frames, links, home_team_id):
    """``add_restart_coordinates(actions, *, frames, links)`` -- no ``home_team_id``."""
    return spadl.add_restart_coordinates(actions, frames=frames, links=links)


_WORKS = "works"
_ALL = (
    "enriched_start_x",
    "enriched_start_y",
    "start_coord_source",
    "start_coord_confidence",
    "enriched_end_x",
    "enriched_end_y",
    "end_coord_source",
    "end_coord_confidence",
)

_entry(
    "spadl.add_restart_coordinates",
    _call_restart_coordinates,
    columns=_ALL,
    velocity={c: AxisVerdict("identical", _WORKS) for c in _ALL},
    visibility={
        "gk_absent": {c: AxisVerdict("identical", _WORKS) for c in _ALL},
        "defender_absent": {c: AxisVerdict("identical", _WORKS) for c in _ALL},
        # MEASURED, not assumed by symmetry: all 8 columns observed `identical` on gk_one_end,
        # same as the other two rosters. This entry is hand-maintained -- `_regenerate.py` loops
        # `tracking.__all__` and this is a BOUNDARY_ENTRY_POINT outside it -- so a new roster does
        # NOT reach it automatically, and `test_every_visibility_roster_has_its_own_slot` is what
        # says so (it failed with a KeyError until this block existed).
        "gk_one_end": {c: AxisVerdict("identical", _WORKS) for c in _ALL},
    },
    # ADR-025 imputes restart coordinates from Law-fixed spots and the action's own geometry;
    # nothing here reads another player's position, so both probes correctly move nothing.
    applicability={c: "no_support" for c in _ALL},
    applicability_deltas={c: {"extreme": 0.0, "near": 0.0} for c in _ALL},
    verdict_provenance="structural",
    provenance_rationale=(
        "Reads no velocity-sensitive input -- ADR-025 imputes restart coordinates from the "
        "action's own geometry -- so both legs observe `identical`. A frame-coupling regression "
        "tripwire, not degradation coverage."
    ),
)


def _xt_gk_pressure_levels() -> PressureLevels:
    # Cutpoints (1.0, 2.0) split the synthetic pressure {0.5, 1.5, 2.5} into levels {1, 2, 3}.
    return PressureLevels.from_cutpoints((1.0, 2.0))


class _XtGkPossessionValueDouble:
    """Deterministic, velocity-blind PossessionValue double. Monotone in zone so position/dzv are
    LIVE; reads only (zone, pressure), never a frame, so it cannot mask a velocity dependence."""

    def __init__(self, pressure_levels: PressureLevels) -> None:
        self.pressure_levels = pressure_levels

    def value(self, zone: int, p: int) -> float:
        return 0.001 * (int(zone) + 1) * (1.0 + 0.1 * int(p))

    def surface(self, p: int):  # part of the Protocol; unused by the per-action loop
        return np.zeros((1, 1), dtype=float)

    def delta_v(self, s: State, s_next: State) -> DeltaV:
        pos = self.value(s_next.zone, s.pressure_level) - self.value(s.zone, s.pressure_level)
        return DeltaV(delta=pos, pressure_component=0.0, position_component=pos)  # p'=p -> pev 0


class _XtGkRetentionDouble:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        f = features["feat"].to_numpy(dtype=float)
        return 0.35 + 0.30 * (np.sin(f) * 0.5 + 0.5)  # deterministic, in (0.35, 0.65), non-constant


class _XtGkTurnoverDouble:
    def value(self, zone: int, p: int) -> float:
        return 0.0005 * (int(zone) + 1) * (1.0 + 0.05 * int(p))


def _call_xt_gk_v2(actions, frames, links, home_team_id):
    """Frame-blind: `frames`/`links`/`home_team_id` are ignored (compute_xt_gk_v2 reads only
    `actions` + injected ports). Both legs share identical `actions`, so the output is identical
    by construction -- a frame-coupling tripwire (spec Part 1)."""
    a = actions.copy()
    # Positional index, NOT action_id-cast: dtype-agnostic (ADR-019 consistency with the gkdv
    # sibling; immune if the harness ever grows an id_dtype axis).
    idx = np.arange(len(a), dtype=float)
    a["pressure"] = 0.5 + (idx % 3)  # 0.5/1.5/2.5 -> levels 1/2/3
    pl = _xt_gk_pressure_levels()
    rf = pd.DataFrame({"feat": idx}, index=a.index)
    return compute_xt_gk_v2(
        a,
        possession_value=_XtGkPossessionValueDouble(pl),
        retention=_XtGkRetentionDouble(),
        turnover_cost=_XtGkTurnoverDouble(),
        pressure_levels=pl,
        retention_features=rf,
    )


_XTGK_V2_COLS = (
    "xt_gk_v2_position",
    "xt_gk_v2_pev",
    "xt_gk_v2_retention_loss",
    "xt_gk_v2_dzv",
    "xt_gk_v2",
)

_entry(
    "xtgk.compute_xt_gk_v2",
    _call_xt_gk_v2,
    columns=_XTGK_V2_COLS,
    velocity={c: AxisVerdict("identical", "works") for c in _XTGK_V2_COLS},
    visibility={
        r: {c: AxisVerdict("identical", "works") for c in _XTGK_V2_COLS}
        for r in ("gk_absent", "defender_absent", "gk_one_end")
    },
    applicability={c: "no_support" for c in _XTGK_V2_COLS},
    applicability_deltas={c: {"extreme": 0.0, "near": 0.0} for c in _XTGK_V2_COLS},
    verdict_provenance="structural",
    provenance_rationale=(
        "Frame-blind: compute_xt_gk_v2 reads `actions` + injected ports, never a frame, so both "
        "legs observe `identical`. `works` means it fabricates nothing through a frame it never "
        "reads -- NOT that xt_gk_v2 is velocity-robust or SB360-computable (its velocity-dependence "
        "lives in its inputs: pressure/is_gk_distribution/retention_features, computed upstream from "
        "tracking, unavailable on real SB360). A frame-coupling regression tripwire. ADR-053 Part 4."
    ),
)


_GK_FK = ["game_id", "period_id", "frame_id"]


def _gkdv_scored(frames, home_team_id):
    """Run build_ghost_frames with the fixture's OWN possession (no shared-position change) and
    possession_stride=1 (score every eligible frame so anchor frames are never stride-dropped)."""
    players = frames[~frames["is_ball"].astype(bool)]
    carrier = (
        players[[*_GK_FK, "team_in_possession"]]
        .drop_duplicates(subset=_GK_FK)
        .rename(columns={"team_in_possession": "ball_carrier_team_id"})
        .reset_index(drop=True)
    )
    cf, prov, _ = build_ghost_frames(
        frames, home_team_id=home_team_id, carrier=carrier, params=GkdvParams(possession_stride=1)
    )
    return cf, prov


def _gkdv_per_action(frames, cf, prov, links, actions, arm_fn):
    """Project a per-frame arm to per-action: the arm value at the action's ANCHOR frame if that
    frame was scored, else NaN. Frame ids matched canonically (ADR-019)."""
    scored = prov[prov["drop_reason"].isna()]
    scored_ids = {canonical_id(f) for f in scored["frame_id"]}
    anchor = {canonical_id(aid): fid for aid, fid in zip(links["action_id"], links["frame_id"], strict=True)}
    team = {canonical_id(aid): tid for aid, tid in zip(actions["action_id"], actions["team_id"], strict=True)}
    vals = []
    for aid in actions["action_id"]:
        key = canonical_id(aid)
        fid = anchor.get(key)
        if fid is None or pd.isna(fid) or canonical_id(fid) not in scored_ids:
            vals.append(np.nan)
            continue
        actual = frames[ids_equal(frames["frame_id"], pd.Series(fid, index=frames.index))]
        ghost = cf[ids_equal(cf["frame_id"], pd.Series(fid, index=cf.index))]
        vals.append(float(arm_fn(actual, ghost, team[key])))
    return pd.Series(vals, index=actions.index)


_GKDV_STRUCTURAL_RATIONALE = (
    "Inherited from serve_ghost_gk_positions, which REFUSES velocity-less freeze-frames "
    "(ADR-054): Leg A scores zero frames -> NaN, Leg B scores the in-domain actions. Same "
    "`honest_nan` class as add_ghost_gk. The arms are never reached on Leg A, so their intrinsic "
    "zero-velocity behaviour is out of scope and contingent on that refusal (cf. ADR-063). ADR-053 Part 4."
)


def _call_gkdv_build_ghost_frames(actions, frames, links, home_team_id):
    cf, prov = _gkdv_scored(frames, home_team_id)
    scored = prov[prov["drop_reason"].isna()]
    # `ids_equal` is POSITIONAL (fresh RangeIndex); `scored` carries a non-contiguous index
    # (prov is sorted+concat), so mask via `.to_numpy()` to avoid pandas label alignment --
    # the same idiom the engine's own `_write_back`/`_same_team` uses for this exact filter.
    defending = scored[ids_equal(scored["gk_team_id"], scored["defending_team_id"]).to_numpy()]

    def _arm(actual, ghost, _team):  # arm_fn shape; reads the defending keeper's provenance row
        target = actual["frame_id"].iloc[0]
        row = defending[[bool(same_id(f, target)) for f in defending["frame_id"]]]
        if not len(row):
            return np.nan
        return float(row["displacement_m"].iloc[0])

    disp = _gkdv_per_action(frames, cf, prov, links, actions, _arm)

    # ghost_x/ghost_y follow the same projection.
    def _xy(colname):
        def arm(actual, ghost, _team):
            target = actual["frame_id"].iloc[0]
            row = defending[[bool(same_id(f, target)) for f in defending["frame_id"]]]
            return float(row[colname].iloc[0]) if len(row) else np.nan

        return _gkdv_per_action(frames, cf, prov, links, actions, arm)

    return actions.assign(
        ghost_x=_xy("ghost_x").to_numpy(),
        ghost_y=_xy("ghost_y").to_numpy(),
        displacement_m=disp.to_numpy(),
    )


_GKDV_BGF_COLS = ("ghost_x", "ghost_y", "displacement_m")
_entry(
    "gkdv.build_ghost_frames",
    _call_gkdv_build_ghost_frames,
    columns=_GKDV_BGF_COLS,
    velocity={c: AxisVerdict("all_nan", "honest_nan") for c in _GKDV_BGF_COLS},
    visibility={
        "gk_absent": {
            c: AxisVerdict("no_signal", "not_exercised", rationale=_GKDV_STRUCTURAL_RATIONALE) for c in _GKDV_BGF_COLS
        },
        "defender_absent": {c: AxisVerdict("all_nan", "honest_nan") for c in _GKDV_BGF_COLS},
        "gk_one_end": {c: AxisVerdict("all_nan", "honest_nan") for c in _GKDV_BGF_COLS},
    },
    applicability={c: "no_support" for c in _GKDV_BGF_COLS},
    applicability_deltas={c: {"extreme": 0.0, "near": 0.0} for c in _GKDV_BGF_COLS},
    verdict_provenance="structural",
    provenance_rationale=_GKDV_STRUCTURAL_RATIONALE,
)


def _call_gkdv_delta_das(actions, frames, links, home_team_id):
    from silly_kicks.gkdv import delta_das

    cf, prov = _gkdv_scored(frames, home_team_id)
    vals = _gkdv_per_action(
        frames,
        cf,
        prov,
        links,
        actions,
        lambda actual, ghost, tid: delta_das(actual, ghost, attacking_team_id=tid),
    )
    return actions.assign(delta_das=vals.to_numpy())


def _call_gkdv_delta_threat(actions, frames, links, home_team_id):
    from scripts._sb_battery import audit_xt
    from silly_kicks.gkdv import delta_threat_suppression
    from silly_kicks.tracking import resolve_defended_goals

    cf, prov = _gkdv_scored(frames, home_team_id)
    goal_map = resolve_defended_goals(frames)  # byte-identical to the engine's _pin_defended_goal
    xt = audit_xt()
    vals = _gkdv_per_action(
        frames,
        cf,
        prov,
        links,
        actions,
        lambda actual, ghost, tid: delta_threat_suppression(
            actual, ghost, attacking_team_id=tid, xt=xt, goal_map=goal_map
        ),
    )
    return actions.assign(delta_threat_suppression=vals.to_numpy())


for _name, _fn, _col in (
    ("gkdv.delta_das", _call_gkdv_delta_das, "delta_das"),
    ("gkdv.delta_threat_suppression", _call_gkdv_delta_threat, "delta_threat_suppression"),
):
    _entry(
        _name,
        _fn,
        columns=(_col,),
        velocity={_col: AxisVerdict("all_nan", "honest_nan")},
        visibility={
            "gk_absent": {_col: AxisVerdict("no_signal", "not_exercised", rationale=_GKDV_STRUCTURAL_RATIONALE)},
            "defender_absent": {_col: AxisVerdict("all_nan", "honest_nan")},
            "gk_one_end": {_col: AxisVerdict("all_nan", "honest_nan")},
        },
        applicability={_col: "no_support"},
        applicability_deltas={_col: {"extreme": 0.0, "near": 0.0}},
        verdict_provenance="structural",
        provenance_rationale=_GKDV_STRUCTURAL_RATIONALE,
    )
