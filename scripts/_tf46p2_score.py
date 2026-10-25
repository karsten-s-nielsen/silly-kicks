"""Per-match truth scoring for the TF-46 Arm 1 Part 2 pre-flight pipelines (spec sections 5, 15.2, 16.2).

Each scorer runs the REAL library port on one staged match with a pass-all gate table injected (so it scores the
RULE, never the shipped gate) and compares the rule's output with that match's tier-1 truth. It emits one SCORE
row per scored item (``SCORE_COLUMNS``): the cell key, the match, and either a boolean ``ok`` (a fixed bar) or a
numeric ``err`` (a calibrated tolerance). ``aggregate_scores`` reduces the rows to the committed per-match
``(k, n)`` aggregate in ``AGGREGATE_COLUMNS`` form, with the cell-level calibration, the LOMO gate value, the
GS admission per signal and the subtype baseline margin -- exactly what ``derive_gate_table`` reads (ADR-056).

Tiers (spec section 5): ``reference`` rows score a provider's rule on its OWN undegraded tracking against its own
native truth (profile = that provider's profile). ``transfer`` rows score the rule on tracking degraded to a
TARGET SkillCorner profile (profile = the target; ``base`` = the source provider). No raw per-row provider field
leaves a unit result except through the reduce, which writes counts and derived statistics only (ADR-116).

ASCII-only (the driver ASCII gate). Imports no pining loader (the ADR-128 I1 gate).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from silly_kicks.id_compat import same_id
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._restart_validation import (
    AGGREGATE_COLUMNS,
    RATIFIED,
    DataProfile,
    GateEntry,
    cell_passes,
    noninferior,
)


def _i(v) -> int:
    """pandas Scalar -> int (see :func:`_f`)."""
    return int(v)


def _f(v) -> float:
    """pandas Scalar -> float; the untyped arg sidesteps the Scalar stub friction."""
    return float(v)


# ---------------------------------------------------------------------------
# Score rows
# ---------------------------------------------------------------------------

_PROFILE_COLS = ("provider", "tracking_release", "gi_release", "competition_season")

#: One row per scored item. ``ok`` is the fixed-bar verdict (``<NA>`` for a calibrated signal, whose verdict is
#: decided in the reduce from ``err``); ``pred`` / ``truth`` carry the subtype direction labels for the
#: majority-baseline comparison.
SCORE_COLUMNS: tuple[str, ...] = (
    *_PROFILE_COLS,
    "restart_type",
    "signal",
    "cell_state",
    "base",
    "tier",
    "stratum",
    "parent_cell_state",
    "match_id",
    "ok",
    "err",
    "pred",
    "truth",
    "footprint_inside_share",
    # kick_time transfer rows (run (2), spec 6.5 amended 2026-10-22): each candidate's |t - K| where BOTH were observed
    "err_contact",
    "err_onset",
)


def score_row(
    profile: DataProfile,
    restart_type: str,
    signal: str,
    cell_state: str,
    *,
    base: str,
    tier: str,
    match_id: str,
    ok: bool | None = None,
    err: float = float("nan"),
    stratum: str = "",
    parent_cell_state: str = "",
    pred: Any = None,
    truth: Any = None,
    footprint_inside_share: float = float("nan"),
    err_contact: float = float("nan"),
    err_onset: float = float("nan"),
) -> dict:
    """One ``SCORE_COLUMNS`` row."""
    return {
        "provider": profile.provider,
        "tracking_release": profile.tracking_release,
        "gi_release": profile.gi_release,
        "competition_season": profile.competition_season,
        "restart_type": restart_type,
        "signal": signal,
        "cell_state": cell_state,
        "base": base,
        "tier": tier,
        "stratum": stratum,
        "parent_cell_state": parent_cell_state,
        "match_id": str(match_id),
        "ok": pd.NA if ok is None else bool(ok),
        "err": float(err),
        "pred": pred,
        "truth": truth,
        "footprint_inside_share": float(footprint_inside_share),
        "err_contact": float(err_contact),
        "err_onset": float(err_onset),
    }


def scores_frame(rows: list[dict]) -> pd.DataFrame:
    """A ``SCORE_COLUMNS`` frame from score rows (empty-safe; ``ok`` nullable boolean)."""
    df = pd.DataFrame(rows, columns=list(SCORE_COLUMNS))
    df["ok"] = df["ok"].astype("boolean")
    df["err"] = pd.to_numeric(df["err"], errors="coerce").astype("float64")
    df["footprint_inside_share"] = pd.to_numeric(df["footprint_inside_share"], errors="coerce").astype("float64")
    for c in ("err_contact", "err_onset"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in (*_PROFILE_COLS, "restart_type", "signal", "cell_state", "base", "tier", "stratum"):
        df[c] = df[c].astype("object")
    return df


# ---------------------------------------------------------------------------
# The pass-all scoring gate (scores the RULE regardless of the shipped table)
# ---------------------------------------------------------------------------

_SCORING_ENTRY = GateEntry(
    tier="reference",
    base="native",
    metric=float("nan"),
    wilson_lb=float("nan"),
    bootstrap_lb=float("nan"),
    n=0,
    m=0,
    cell_resolution="cell",
    agreement_winner="tf79_contact",  # scoring only: run (2) scores BOTH candidates' errors independently of it
)


class PassAllGate(Mapping):
    """A gate table that passes EVERY key. Injected through the ports' ``gate_table=`` so a driver measures each
    rule's output; it never reaches production (the shipped table is ``RESTART_GATE_TABLE``)."""

    def __getitem__(self, key):
        return _SCORING_ENTRY

    def get(self, key, default=None):
        return _SCORING_ENTRY

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0


PASS_ALL = PassAllGate()


class _ExceptSignal(Mapping):
    """``PASS_ALL`` except ONE profile-level signal (the z_overhead gate-off arm)."""

    def __init__(self, signal: str):
        self.signal = signal

    def __getitem__(self, key):
        return self.get(key)

    def get(self, key, default=None):
        return default if key[2] == self.signal else _SCORING_ENTRY

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# A staged match (the TF-46 Part 2 staged layout, scripts._driver_sources.PART2_EXTRA_FILES)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Part2Match:
    """One staged match: SPADL actions, frames, the provider-native events, the camera footprint (SkillCorner),
    the data profile and the provider pitch length."""

    provider: str
    match_id: str
    actions: pd.DataFrame
    frames: pd.DataFrame
    native: pd.DataFrame
    footprint: pd.DataFrame
    profile: DataProfile
    pitch_length: float
    is_opendata: bool = False  # opendata ships GI + tracking together: known-paired (C-11, assert_known_paired)
    #: the frames' applied orientation flip per (game_id, period_id) (SkillCorner; the staged
    #: ``report.ltr_flip_by_period``): passed EXPLICITLY to the GI direction check (R.2 2026-10-23)
    flips: Mapping[tuple[Any, int], bool | None] = field(default_factory=dict)
    #: the PROVIDER period range per (game_id, period_id) (SkillCorner; the staged
    #: ``report.provider_period_frame_ranges``): passed EXPLICITLY to the pairing Range test (spec section 6.1,
    #: amended 2026-10-25); empty = not staged (the pairing then records range_basis "retained")
    period_ranges: Mapping[tuple[Any, int], tuple[int, int]] = field(default_factory=dict)
    #: the staged GI period-label repair census of the match (spec section 26.8; SkillCorner), else None
    period_repair: Mapping[str, Any] | None = None


def load_part2_match(ref, root: Path) -> Part2Match:
    """Read one ``staged_part2_refs`` unit through the dataset read gate (``_driver_sources.read_dataset_*``)."""
    from scripts._driver_sources import read_dataset_json, read_dataset_parquet

    frames_f, actions_f, native_f, foot_f, prof_f = (root / f for f in ref.files)
    frames = read_dataset_parquet(frames_f)
    actions = read_dataset_parquet(actions_f) if actions_f.exists() else pd.DataFrame()
    native = read_dataset_parquet(native_f) if native_f.exists() else pd.DataFrame()
    foot = read_dataset_parquet(foot_f) if foot_f.exists() else pd.DataFrame()
    meta = read_dataset_json(prof_f) if prof_f.exists() else {}
    m = part2_match_from(ref.source_provider, ref.game, actions, frames, native, foot, meta)
    assert_staging_current(m)
    return m


class StaleStagingError(RuntimeError):
    """A staged SkillCorner unit predates the GI period-label repair / provider-range staging (spec section 26.8):
    restage at the current sha before R.2."""


def assert_staging_current(m: Part2Match) -> None:
    """Reject a stale SkillCorner staging mechanically (spec section 26.8, DL ruling 2026-10-25, Q6): the staged
    ``_profile`` must carry the period-repair census, the staged actions the ``period_source`` column, and -- when the
    unit has tracking frames -- the provider period range (a unit staged without tracking has no adapter report, so
    no range; restaging could not add one). Other providers pass (no GI)."""
    if m.provider != "skillcorner":
        return
    missing = [
        name
        for name, ok in (
            ("_profile provider_period_frame_ranges", bool(m.period_ranges) or m.frames.empty),
            ("_profile gi_period_repair", m.period_repair is not None),
            ("_actions period_source", "period_source" in m.actions.columns),
        )
        if not ok
    ]
    if missing:
        raise StaleStagingError(
            f"skillcorner match {m.match_id}: stale staging (missing {missing}); restage at the current sha "
            "(spec section 26.8)."
        )


def part2_match_from(provider, match_id, actions, frames, native, footprint, meta: Mapping) -> Part2Match:
    """Assemble a :class:`Part2Match` (the staged loader and the tests share it)."""
    from scripts._tf46p2_common import data_profile_for

    pitch_length = float(meta.get("pitch_length", spadlconfig.field_length))
    pitch_width = float(meta.get("pitch_width", spadlconfig.field_width))
    frames = frames.copy()
    # the converter stamps (C2-IMPL-07, C2 r2 C-25), restored after staging: every Law location scales with them
    frames.attrs["pitch_length"] = pitch_length
    frames.attrs["pitch_width"] = pitch_width
    return Part2Match(
        provider=str(provider),
        match_id=str(match_id),
        actions=actions,
        frames=frames,
        native=native,
        footprint=footprint,
        profile=data_profile_for(str(provider), meta),
        pitch_length=pitch_length,
        is_opendata=bool(meta.get("is_opendata", False)),
        flips={(str(match_id), int(p)): v for p, v in dict(meta.get("ltr_flip_by_period") or {}).items()},
        period_ranges={
            (str(match_id), int(p)): (int(lo), int(hi))
            for p, (lo, hi) in dict(meta.get("provider_period_frame_ranges") or {}).items()
        },
        period_repair=meta.get("gi_period_repair"),
    )


def provider_base(provider: str) -> str:
    """The ``GateEntry.base`` of a source provider (spec section 5.3)."""
    return {"sportec": "idsse", "idsse": "idsse", "gradientsports": "gradientsports"}.get(str(provider), "native")


# ---------------------------------------------------------------------------
# Tier-1 truth (spec sections 5.2, 5.6)
# ---------------------------------------------------------------------------

_RESTART_BASE = {
    "corner_crossed": "corner",
    "corner_short": "corner",
    "freekick_crossed": "free_kick",
    "freekick_short": "free_kick",
    "goalkick": "goal_kick",
    "throw_in": "throw_in",
    "shot_penalty": "penalty",
}


def _type_name(tid) -> str:
    t = int(tid)
    return spadlconfig.actiontypes[t] if 0 <= t < len(spadlconfig.actiontypes) else ""


def native_restarts(actions: pd.DataFrame) -> pd.DataFrame:
    """The WHICH truth: the native restart rows of a reference provider (spec section 5.2) -- one row per SPADL
    restart (the converter types them from the native restart event), with the event time, team, player and the
    event-coded start location."""
    cols = ["restart_id", "game_id", "period_id", "restart_type", "base", "t_event", "team_id", "player_id", "x", "y"]
    if actions.empty or "type_id" not in actions.columns:
        return pd.DataFrame(columns=cols)
    names = actions["type_id"].map(_type_name)
    sel = actions[names.isin(list(_RESTART_BASE)).to_numpy()].copy()
    if sel.empty:
        return pd.DataFrame(columns=cols)
    tn = sel["type_id"].map(_type_name)
    return pd.DataFrame(
        {
            "restart_id": sel["action_id"].to_numpy(),
            "game_id": sel["game_id"].to_numpy(),
            "period_id": sel["period_id"].astype("int64").to_numpy(),
            "restart_type": tn.to_numpy(),
            "base": tn.map(_RESTART_BASE).to_numpy(),
            "t_event": sel["time_seconds"].astype("float64").to_numpy(),
            "team_id": sel["team_id"].to_numpy(),
            "player_id": sel["player_id"].to_numpy() if "player_id" in sel.columns else None,
            "x": sel["start_x"].astype("float64").to_numpy(),
            "y": sel["start_y"].astype("float64").to_numpy(),
        }
    ).reset_index(drop=True)


def _ball_by_period(frames: pd.DataFrame) -> dict[int, tuple[pd.DataFrame, np.ndarray]]:
    """Each period's ball rows sorted by time, built ONCE (ADR-068)."""
    out: dict[int, tuple[pd.DataFrame, np.ndarray]] = {}
    if frames.empty:
        return out
    ball = frames[frames["is_ball"].to_numpy(dtype=bool)]
    for p, g in ball.groupby("period_id", sort=True):
        g = g.sort_values("time_seconds").reset_index(drop=True)
        out[_i(p)] = (g, g["time_seconds"].to_numpy(dtype=float))
    return out


def _player_xy_at(frames_by_period, period: int, player, t: float):
    hit = frames_by_period.get(period)
    if hit is None or player is None or pd.isna(player):
        return None
    rows, times = hit
    j = int(np.searchsorted(times, t))
    lo, hi = max(0, j - 60), min(len(times), j + 60)
    cand = rows.iloc[lo:hi]
    cand = cand[cand["player_id"].map(lambda v: v is not None and same_id(v, player))]
    if cand.empty:
        return None
    k = int(np.argmin(np.abs(cand["time_seconds"].to_numpy(dtype=float) - t)))
    return float(cand["x"].iloc[k]), float(cand["y"].iloc[k])


def _restart_ball_windows(restarts: pd.DataFrame, frames: pd.DataFrame, goal_map):
    """Yield ``(row position, restart, ball window, attacks_rtl, thrower_xy, pitch)`` per native restart: the
    undegraded
    ball track in ``[t_event - a, t_event + b]`` (a = b = the 2.0 s bracket clip), the restarting team's GoalMap
    direction (``None`` = unresolved; section 6.4 fails a corner / goal kick / penalty closed) and, for a throw-in,
    the restart player's tracked position at the event; ``pitch`` = the provider pitch (``resolve_pitch_dims``: the
    frames' attrs stamp, else 105 x 68). ``goal_map`` defaults to ``resolve_defended_goals(frames)``, built once."""
    from silly_kicks.tracking._gk_resolve import resolve_defended_goals
    from silly_kicks.tracking._law_geometry import resolve_pitch_dims
    from silly_kicks.tracking._restart_frame import team_attacks_rtl

    if restarts.empty or frames.empty:
        return
    dims = resolve_pitch_dims(frames)
    if goal_map is None:
        goal_map = resolve_defended_goals(frames)
    a = RATIFIED["bracket_clip_s"]
    balls = _ball_by_period(frames)
    players = frames[~frames["is_ball"].to_numpy(dtype=bool)]
    pl_by_period = {
        _i(p): (g.sort_values("time_seconds").reset_index(drop=True), np.sort(g["time_seconds"].to_numpy(dtype=float)))
        for p, g in players.groupby("period_id", sort=True)
    }
    rtl = team_attacks_rtl(restarts.reset_index(drop=True), goal_map).to_numpy(dtype=object)
    for i, r in enumerate(restarts.itertuples(index=False)):
        hit = balls.get(_i(r.period_id))
        if hit is None:
            continue
        rows, times = hit
        lo_i = int(np.searchsorted(times, _f(r.t_event) - a, side="left"))
        hi_i = int(np.searchsorted(times, _f(r.t_event) + a, side="right"))
        win = rows.iloc[lo_i:hi_i]
        if len(win) < 2:
            continue
        thrower = (
            _player_xy_at(pl_by_period, _i(r.period_id), r.player_id, _f(r.t_event)) if r.base == "throw_in" else None
        )
        flag = None if pd.isna(rtl[i]) else bool(rtl[i])
        yield i, r, win, flag, thrower, dims


def kick_truth(
    restarts: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    thresholds: Mapping[str, float] | None = None,
    goal_map=None,
) -> np.ndarray:
    """The WHEN truth ``K`` per native restart (spec section 5.2): the kinematic release on the UNDEGRADED track
    (``_restart_ball_windows``), with the section 6.4 rest location and direction and the strike rule (spec
    amendment 2026-10-22, plan CL-30a): K is the release of the qualifying rest at or before the FIRST strike; a
    rest after it is post-kick and never anchors K. NaN where the rule does not fire. ``thresholds`` = the run (1)
    +/-50 % sensitivity override."""
    from silly_kicks.tracking._kick_resolver import kinematic_release

    k = np.full(len(restarts), np.nan)
    for i, r, win, flag, thrower, dims in _restart_ball_windows(restarts, frames, goal_map):
        rel, _rest = kinematic_release(
            win, restart_type=str(r.base), thrower_xy=thrower, thresholds=thresholds, attacks_rtl=flag, pitch=dims
        )
        k[i] = rel
    return k


def _blocked_classes(actions: pd.DataFrame, restarts: pd.DataFrame) -> list[str]:
    """Per native restart: ``blocked`` / ``not_blocked`` from the reference events (spec amendment 2026-10-22,
    run (1) outcome-class guard; plan CL-32). The restart row's own native block flag (``shot_blocked`` /
    ``cross_blocked``, ADR-046) decides when it is known; else the next action of the period within the
    ``rstar_window_s`` window is the BLOCK when it is by the other team and starts within ``penalty_enc_disk_m`` +
    ``penalty_e_enc_m`` of the restart origin (the wall stands 9.15 m away). The per-period action times are
    sorted once (ADR-068)."""
    from silly_kicks.id_compat import same_id

    out = ["not_blocked"] * len(restarts)
    if restarts.empty or actions.empty:
        return out
    by_id = {int(a): i for i, a in enumerate(actions["action_id"].to_numpy())}
    per = {}
    for p_, g in actions.groupby("period_id", sort=True):
        g = g.sort_values("time_seconds", kind="mergesort")
        per[_i(p_)] = (g, g["time_seconds"].to_numpy(dtype=np.float64))
    reach = RATIFIED["penalty_enc_disk_m"] + RATIFIED["penalty_e_enc_m"]
    for k, r in enumerate(restarts.itertuples(index=False)):
        row = actions.iloc[by_id[_i(r.restart_id)]] if _i(r.restart_id) in by_id else None
        flags = [row.get(c) for c in ("shot_blocked", "cross_blocked")] if row is not None else []
        known = [bool(f) for f in flags if f is not None and not pd.isna(f)]
        if known:
            out[k] = "blocked" if any(known) else "not_blocked"
            continue
        hit = per.get(_i(r.period_id))
        if hit is None:
            continue
        g, t = hit
        j = int(np.searchsorted(t, _f(r.t_event), side="right"))
        if j >= len(g) or t[j] - _f(r.t_event) > RATIFIED["rstar_window_s"]:
            continue
        nxt = g.iloc[j]
        other = not same_id(nxt["team_id"], r.team_id)
        near = float(np.hypot(_f(nxt["start_x"]) - _f(r.x), _f(nxt["start_y"]) - _f(r.y))) <= reach
        if other and near:
            out[k] = "blocked"
    return out


def kick_episode_rows(
    restarts: pd.DataFrame, frames: pd.DataFrame, *, goal_map=None, actions: pd.DataFrame | None = None
) -> list[dict]:
    """Run (1) episode diagnostic per dead-ball native restart (DL rulings 2026-10-22: C2-IMPL-19 item 6 and the
    strike-rule amendment): the number of qualifying rest episodes, how many have a release and a strike, the
    FIRST and LAST qualifying release, the K verdict ``reason`` (``strike`` / ``single_episode`` /
    ``no_strike_ambiguous`` / ``no_episode``), the subtype, the outcome class (``_blocked_classes``) and the peak
    speed of the first strike of a flagged retake. Reported only (``kick_episode_table``,
    ``kick_strike_guard_table``); never a tuning input."""
    from silly_kicks.tracking._kick_resolver import detect_retake, kick_verdict, rest_episodes
    from silly_kicks.tracking._law_geometry import law_boundary_distance

    blocked = _blocked_classes(actions if actions is not None else pd.DataFrame(), restarts)
    out = []
    for i_row, r, win, flag, _thrower, dims in _restart_ball_windows(restarts, frames, goal_map):
        base = str(r.base)
        if base == "throw_in":
            continue
        all_eps = rest_episodes(win, restart_type=base, attacks_rtl=flag, pitch=dims)
        boundary = _anchor_boundary_distance(all_eps, base, flag, dims, law_boundary_distance)
        eps = [e for e in all_eps if e.qualifies]
        rel = [e.release for e in eps if np.isfinite(e.release)]
        strikes = [e for e in eps if e.strike]
        is_retake, _last = detect_retake(all_eps, base, -np.inf, np.inf)
        out.append(
            {
                "restart_type": base,
                "subtype": str(r.restart_type),
                "outcome_class": blocked[i_row],
                "reason": kick_verdict(all_eps).reason,
                "n_qualifying": len(eps),
                "n_released": len(rel),
                "n_strikes": len(strikes),
                "k_first": rel[0] if rel else None,
                "k_last": rel[-1] if rel else None,
                "retake": bool(is_retake),
                "retake_first_peak_ms": float(strikes[0].peak_speed) if is_retake and strikes else None,
                "boundary_dist_m": boundary,
            }
        )
    return out


def _anchor_boundary_distance(episodes, base: str, flag, dims, law_boundary_distance) -> float | None:
    """The run (1) Law-boundary diagnostic (spec amendment 2026-10-22, C2 r2 C-25 item 6): the signed real-metre
    distance of the ANCHOR rest -- the rest whose release is the first strike, regardless of its location; else the
    last rest -- to the section 6.4 Law location of a corner / goal kick / penalty. ``None`` for a free kick, an
    unresolved direction or a trace without a rest. Diagnostic only, never a tuning input: if legal rests sit
    outside the Law boundary because of tracking noise, the noise margin returns to the owner with this
    distribution."""
    if base not in ("corner", "goal_kick", "penalty") or flag is None or not episodes:
        return None
    strikes = [e for e in episodes if e.strike]
    anchor = strikes[0] if strikes else episodes[-1]
    cx, cy = anchor.centroid
    ax, ay = (spadlconfig.field_length - cx, spadlconfig.field_width - cy) if flag else (cx, cy)
    return float(law_boundary_distance(base, ax, ay, dims))


def kick_boundary_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per data profile x restart type (corner / goal kick / penalty): the n anchor rests, the share outside the
    Law tolerance (corner ``corner_rest_m``, goal kick one ball radius, penalty ``penalty_spot_r_m``) and the
    p50 / p90 / max signed distance to the Law boundary (m). Diagnostic only (C2 r2 C-25 item 6)."""
    cols = ["data_profile", "restart_type", "n", "share_outside", "p50_m", "p90_m", "max_m"]
    if rows.empty or "boundary_dist_m" not in rows.columns:
        return pd.DataFrame(columns=cols)
    tol = {
        "corner": RATIFIED["corner_rest_m"],
        "goal_kick": spadlconfig.ball_radius,
        "penalty": RATIFIED["penalty_spot_r_m"],
    }
    out = []
    for (dp, rt), g in rows.groupby(["data_profile", "restart_type"], sort=True):
        if str(rt) not in tol:
            continue
        d = pd.to_numeric(g["boundary_dist_m"], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(d):
            continue
        out.append(
            {
                "data_profile": dp,
                "restart_type": rt,
                "n": len(d),
                "share_outside": float((d > tol[str(rt)]).mean()),
                "p50_m": float(np.quantile(d, 0.5)),
                "p90_m": float(np.quantile(d, 0.9)),
                "max_m": float(d.max()),
            }
        )
    return pd.DataFrame(out, columns=cols)


def kick_strike_guard_table(rows: pd.DataFrame) -> pd.DataFrame:
    """The pre-registered run (1) strike-rule guards (spec amendment 2026-10-22), per data profile x restart type:
    the no-strike ``none`` rate overall, for ``short`` and ``crossed`` subtypes and for the ``blocked`` /
    ``not_blocked`` outcome classes. ``v_kick_returns_to_owner`` -- the section 6.3 B-41 rule: the short coverage
    (1 - none rate) is below the crossed coverage by more than ``sc_freekick_short_cover_margin`` (0.10).
    ``rule_returns_to_owner`` -- the none rates of the two outcome classes differ by more than 0.10. Each flag is
    NaN when a side has no rows. The flagged retakes' median first-strike peak speed is reported alongside."""
    cols = [
        "data_profile",
        "restart_type",
        "n",
        "none_rate",
        "none_rate_short",
        "none_rate_crossed",
        "v_kick_returns_to_owner",
        "none_rate_blocked",
        "none_rate_not_blocked",
        "rule_returns_to_owner",
        "n_retake",
        "retake_first_peak_ms_median",
    ]
    if rows.empty or "reason" not in rows.columns:
        return pd.DataFrame(columns=cols)
    margin = RATIFIED["sc_freekick_short_cover_margin"]

    def _rate(g: pd.DataFrame) -> float:
        return float((g["reason"] == "no_strike_ambiguous").mean()) if len(g) else float("nan")

    out = []
    for (dp, rt), g in rows.groupby(["data_profile", "restart_type"], sort=True):
        sub = g["subtype"].astype(str)
        r_short, r_cross = _rate(g[sub.str.endswith("_short")]), _rate(g[sub.str.endswith("_crossed")])
        r_blk, r_not = _rate(g[g["outcome_class"] == "blocked"]), _rate(g[g["outcome_class"] == "not_blocked"])
        peaks = pd.to_numeric(g.loc[g["retake"].astype(bool), "retake_first_peak_ms"], errors="coerce").dropna()
        out.append(
            {
                "data_profile": dp,
                "restart_type": rt,
                "n": len(g),
                "none_rate": _rate(g),
                "none_rate_short": r_short,
                "none_rate_crossed": r_cross,
                "v_kick_returns_to_owner": (
                    bool((1.0 - r_short) < (1.0 - r_cross) - margin)
                    if np.isfinite(r_short) and np.isfinite(r_cross)
                    else None
                ),
                "none_rate_blocked": r_blk,
                "none_rate_not_blocked": r_not,
                "rule_returns_to_owner": (
                    bool(abs(r_blk - r_not) > 0.10) if np.isfinite(r_blk) and np.isfinite(r_not) else None
                ),
                "n_retake": int(g["retake"].astype(bool).sum()),
                "retake_first_peak_ms_median": float(peaks.median()) if len(peaks) else float("nan"),
            }
        )
    return pd.DataFrame(out, columns=cols)


def kick_episode_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per data profile x restart type: n traces, the share with >= 2 qualifying episodes, and, on those with >= 2
    released episodes, the median and max ``k_last - k_first`` (s). A diagnostic, never a tuning input."""
    cols = [
        "data_profile",
        "restart_type",
        "n",
        "share_multi_episode",
        "n_multi_release",
        "median_diff_s",
        "max_diff_s",
    ]
    if rows.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for (dp, rt), g in rows.groupby(["data_profile", "restart_type"], sort=True):
        multi = g[g["n_released"].astype(int) >= 2]
        diff = (pd.to_numeric(multi["k_last"]) - pd.to_numeric(multi["k_first"])).to_numpy(dtype=float)
        out.append(
            {
                "data_profile": dp,
                "restart_type": rt,
                "n": len(g),
                "share_multi_episode": float((g["n_qualifying"].astype(int) >= 2).mean()),
                "n_multi_release": len(multi),
                "median_diff_s": float(np.median(diff)) if len(diff) else float("nan"),
                "max_diff_s": float(np.max(diff)) if len(diff) else float("nan"),
            }
        )
    return pd.DataFrame(out, columns=cols)


def dfl_touch_events(native: pd.DataFrame, restarts: pd.DataFrame) -> pd.DataFrame:
    """The ``dfl_first_touch_truth`` input (spec section 5.6) from the DFL native events: per corner restart, the
    native events after the kick in ``(t_event, t_event + 6]`` of the same period, ordered (``seq``)."""
    cols = [
        "restart_id",
        "seq",
        "event_type",
        "player_id",
        "team_id",
        "timestamp_seconds",
        "tackle_winner",
        "tackle_winner_team",
    ]
    if native.empty or restarts.empty or "timestamp_seconds" not in native.columns:
        return pd.DataFrame(columns=cols)
    period_col = "period" if "period" in native.columns else "period_id"
    nat = native.sort_values([period_col, "timestamp_seconds"], kind="stable")
    by_p = {_i(p): g for p, g in nat.groupby(period_col, sort=True)}
    rows = []
    for r in restarts.itertuples(index=False):
        g = by_p.get(_i(r.period_id))
        if g is None:
            continue
        t = g["timestamp_seconds"].to_numpy(dtype=float)
        lo_i = int(np.searchsorted(t, r.t_event, side="right"))
        hi_i = int(np.searchsorted(t, _f(r.t_event) + RATIFIED["rstar_window_s"], side="right"))
        for seq, (_, e) in enumerate(g.iloc[lo_i:hi_i].iterrows()):
            rows.append(
                {
                    "restart_id": r.restart_id,
                    "seq": seq,
                    "event_type": e.get("event_type"),
                    "player_id": e.get("player_id"),
                    "team_id": e.get("team_id_native", e.get("team")),
                    "timestamp_seconds": float(e["timestamp_seconds"]),
                    "tackle_winner": e.get("tackle_winner"),
                    "tackle_winner_team": e.get("tackle_winner_team"),
                }
            )
    return pd.DataFrame(rows, columns=cols)


def first_touch_truth(m: Part2Match, restarts: pd.DataFrame) -> pd.DataFrame:
    """The section 5.6 first-touch truth per corner restart of a reference provider: DFL extraction for IDSSE
    (``dfl_first_touch_truth``); PFF for GS via ``pff_truth`` (``first_contact`` stays IDSSE-only, so GS rows are
    reported, never a base). Columns: ``restart_id``, ``player_id``, ``team_id``, ``timestamp_seconds``,
    ``touch_class``."""
    from scripts._tf46p2_common import dfl_first_touch_truth

    corners = restarts[restarts["base"] == "corner"]
    if provider_base(m.provider) == "idsse":
        return dfl_first_touch_truth(dfl_touch_events(m.native, corners))
    return pd.DataFrame(columns=["restart_id", "player_id", "team_id", "timestamp_seconds", "touch_class"])


#: The DFL location fields run (1) characterises (spec section 5.2, C-30): the field name -> its x / y columns.
DFL_LOCATION_FIELDS: dict[str, tuple[str, str]] = {
    "position": ("x", "y"),
    "source_position": ("x_source_position", "y_source_position"),
    "position_from_tracking": ("x_position_from_tracking", "y_position_from_tracking"),
}


def location_diffs(m: Part2Match) -> list[dict]:
    """Run (1) location-field characterisation input (spec section 5.2, C-30): per DFL native event with an actor,
    per location field, the distance (m) between the event position and the tracked ACTOR at the nearest frame.
    Native positions and the absolute-frame Sportec tracking share the 105 x 68 per-period-absolute frame. The
    reduce calls a field a tracking COPY for an event type when its median distance is <= 0.1 m."""
    if provider_base(m.provider) != "idsse" or m.native.empty or m.frames.empty or "timestamp_seconds" not in m.native:
        return []
    period_col = "period" if "period" in m.native.columns else "period_id"
    players = m.frames[~m.frames["is_ball"].to_numpy(dtype=bool)]
    by_p = {
        _i(p): (g.sort_values("time_seconds").reset_index(drop=True), np.sort(g["time_seconds"].to_numpy(dtype=float)))
        for p, g in players.groupby("period_id", sort=True)
    }
    rows = []
    for _, e in m.native.iterrows():
        pid = e.get("player_id")
        if pid is None or pd.isna(pid):
            continue
        actor = _player_xy_at(by_p, int(e[period_col]), pid, float(e["timestamp_seconds"]))
        if actor is None:
            continue
        for field_name, (cx, cy) in DFL_LOCATION_FIELDS.items():
            if cx not in m.native.columns or cy not in m.native.columns:
                continue
            ex, ey = e.get(cx), e.get(cy)
            if ex is None or ey is None or pd.isna(ex) or pd.isna(ey):
                continue
            rows.append(
                {
                    "data_profile": f"{m.profile.provider}|{m.profile.tracking_release}|{m.profile.gi_release}|"
                    f"{m.profile.competition_season}",
                    "event_type": str(e.get("event_type")),
                    "field": field_name,
                    "diff_m": float(np.hypot(float(ex) - actor[0], float(ey) - actor[1])),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Tier-2 simulation: a reference match seen as a SkillCorner match (spec section 5.3)
# ---------------------------------------------------------------------------


def simulated_skillcorner_actions(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """The reference actions with every native restart (except penalties, which never create a derived row,
    Part-1 section 6.5) re-cast as a SkillCorner DERIVED row (Part-1 shape: NaN start, NaN player,
    ``action_provenance="derived"``, ``type_source="default"``), its GI frame ids bracketing the kick by the
    section 5.3 anchor simulation. The resolver then runs INSIDE that simulated bracket."""
    out = actions.copy()
    if out.empty:
        return out
    for c, v in (("action_provenance", "native"), ("type_source", "native")):
        if c not in out.columns:
            out[c] = v
    names = out["type_id"].map(_type_name)
    restart = names.isin([n for n in _RESTART_BASE if n != "shot_penalty"]).to_numpy()
    out.loc[restart, "action_provenance"] = "derived"
    out.loc[restart, "type_source"] = "default"
    out.loc[restart, "start_x"] = np.nan
    out.loc[restart, "start_y"] = np.nan
    out["player_id"] = out["player_id"].astype("object")
    out.loc[restart, "player_id"] = pd.NA
    return simulated_bracket_ids(out, frames)


def paired_by_construction(actions: pd.DataFrame) -> dict:
    """The GI / tracking pairing of a simulated match: paired by construction (the bracket is built from the same
    native events the tracking is synchronised to). Every verdict is True by the same construction (C2 delta C-29:
    "paired only when all four are True" holds on transfer-run records too)."""
    from silly_kicks.tracking._restart_frame import GiPairing

    return {
        gid: GiPairing(
            gid,
            "paired",
            None,
            1.0,
            0,
            1.0,
            1,
            "simulated",
            0,
            None,
            None,
            False,
            pair_range_ok=True,
            pair_proximity_ok=True,
            pair_lag_ok=True,
            pair_direction_ok=True,
        )
        for gid in actions["game_id"].dropna().unique()
    }


# ---------------------------------------------------------------------------
# Scorers (one per signal); each returns score rows
# ---------------------------------------------------------------------------


def score_k_gate(m: Part2Match, restarts: pd.DataFrame, k: np.ndarray) -> list[dict]:
    """Run (1) / (14) K gate (spec section 5.2): per reference restart, ``err = t_event - K`` (the reduce centres it
    on the per-type median offset and gates the dispersion within ``tau_event``)."""
    base = provider_base(m.provider)
    rows = []
    for r, kk in zip(restarts.itertuples(index=False), k, strict=True):
        if not np.isfinite(kk) or r.base == "penalty":
            continue
        rows.append(
            score_row(
                m.profile,
                str(r.base),
                "kick_time",
                "k_gate",
                base=base,
                tier="reference",
                match_id=m.match_id,
                err=float(r.t_event - kk),
            )
        )
    return rows


def kick_time_cell_state(source: Any, delta: float) -> str:
    """The ``kick_time`` cell (spec section 15.2): resolver state x sign(delta); ``retake`` separately."""
    src = "none" if source is None or pd.isna(source) else str(source)
    sign = "pos" if np.isfinite(delta) and delta > 0 else ("neg" if np.isfinite(delta) and delta < 0 else "zero")
    return f"{src}_{sign}"


def score_z_overhead(m: Part2Match, restarts: pd.DataFrame, truth: pd.DataFrame) -> list[dict]:
    """Spec section 14.1: first-contact team agreement with tier-1 truth, the F3 overhead gate ON vs OFF, on the
    SAME contacts (paired per match). Reference only (the overhead gate needs measured z)."""
    from silly_kicks.corners import add_corner_windows, identify_corners
    from silly_kicks.tracking._ball_contact import detect_ball_contacts

    corners_truth = truth[truth["touch_class"].isin(["touch", "duel"])] if not truth.empty else truth
    if corners_truth.empty or m.frames.empty:
        return []
    corners = identify_corners(m.actions, provider=m.provider)
    if corners.empty:
        return []
    rs = restarts[restarts["base"] == "corner"].reset_index(drop=True)
    tr = corners_truth.set_index("restart_id")
    base = provider_base(m.provider)
    rows = []
    for arm, table in (("gate_on", PASS_ALL), ("gate_off", _ExceptSignal("z_overhead"))):
        contacts = detect_ball_contacts(m.frames, data_profile=m.profile, gate_table=table)
        win = add_corner_windows(corners, m.frames, contacts, data_profile=m.profile, gate_table=PASS_ALL)
        for j in range(min(len(win), len(rs))):
            rid = rs["restart_id"].iloc[j]
            if rid not in tr.index:
                continue
            w, t = win.iloc[j], tr.loc[rid]
            ok = pd.notna(w["winner_team_id"]) and pd.notna(t["team_id"]) and same_id(w["winner_team_id"], t["team_id"])
            rows.append(
                score_row(
                    m.profile, "*", "z_overhead", arm, base=base, tier="reference", match_id=m.match_id, ok=bool(ok)
                )
            )
    return rows


def _crossed_truth_idsse(m: Part2Match, restarts: pd.DataFrame) -> dict:
    """IDSSE subtype truth (spec section 11.2): the event-coded location of the NEXT touch after the delivery,
    inside the penalty area -> ``crossed``, else ``short``. Uses the next SPADL action's start (the DFL
    ``X-Position`` transformed by the converter); a type flagged as a tracking copy by run (1) is excluded by the
    caller (B-29)."""
    acts = m.actions.sort_values(["period_id", "time_seconds"], kind="stable").reset_index(drop=True)
    pos = {aid: i for i, aid in enumerate(acts["action_id"].to_numpy())}
    out = {}
    pa_x = spadlconfig.field_length - 16.5
    for r in restarts.itertuples(index=False):
        i = pos.get(r.restart_id)
        if i is None or i + 1 >= len(acts) or int(acts["period_id"].iloc[i + 1]) != _i(r.period_id):
            continue
        nx, ny = float(acts["start_x"].iloc[i + 1]), float(acts["start_y"].iloc[i + 1])
        if not (np.isfinite(nx) and np.isfinite(ny)):
            continue
        inside = nx >= pa_x and abs(ny - spadlconfig.field_width / 2.0) <= spadlconfig.penalty_area_half_width
        out[r.restart_id] = "crossed" if inside else "short"
    return out


def _crossed_truth_pff(m: Part2Match, restarts: pd.DataFrame) -> dict:
    """GS subtype truth (spec section 5.6): the PFF label ``possessionEventType`` (``CR`` crossed / ``PA`` short)
    on the set-piece event, read from the native events by ``gameEventId`` == the action's original event id."""
    if m.native.empty or "possessionEventType" not in m.native.columns:
        return {}
    key = "gameEventId" if "gameEventId" in m.native.columns else None
    if key is None or "original_event_id" not in m.actions.columns:
        return {}
    lab = dict(zip(m.native[key].map(str), m.native["possessionEventType"], strict=False))
    oe = dict(zip(m.actions["action_id"], m.actions["original_event_id"].map(str), strict=False))
    out = {}
    for r in restarts.itertuples(index=False):
        v = {"CR": "crossed", "PA": "short"}.get(str(lab.get(oe.get(r.restart_id, ""), "")))
        if v is not None:
            out[r.restart_id] = v
    return out


@dataclass(frozen=True)
class PipelineRun:
    """One run of the restart pipeline on one match (``resolve_kicks`` with every cell passing): the profile it is
    scored under, the provider the ports see, the resolved actions, the frames, the contacts and the GoalMap.

    ``reference``: the provider's own undegraded tracking and native rows. ``transfer``: tracking degraded to a
    TARGET SkillCorner profile, the native restarts re-cast as SkillCorner derived rows inside the simulated
    bracket (spec section 5.3), provider ``skillcorner``."""

    tier: str
    profile: DataProfile
    provider: str
    actions: pd.DataFrame
    frames: pd.DataFrame
    contacts: pd.DataFrame
    goal_map: Any


def run_pipeline(
    m: Part2Match,
    *,
    target: DataProfile | None = None,
    degraded: pd.DataFrame | None = None,
    bracket_ab: Mapping[str, tuple[float, float]] | None = None,
    tier: str | None = None,
) -> PipelineRun:
    """The reference run (``target`` None) or the transfer run to ``target`` on ``degraded`` tracking.
    ``bracket_ab`` = the run (1) native-anchor bracket per restart base (C-03 inter-run flow). ``tier`` overrides
    the tier label: the section 5.3 GS / IDSSE ADMISSION arm runs the SkillCorner rules (the tracking taker, the
    TF-79 kick contact) on the provider's OWN undegraded tracking and scores it as ``reference``."""
    from silly_kicks.tracking._ball_contact import detect_ball_contacts
    from silly_kicks.tracking._gk_resolve import resolve_defended_goals
    from silly_kicks.tracking._kick_resolver import resolve_kicks

    tier_override = tier
    if target is None:
        frames, profile, provider, acts, pairing, tier = m.frames, m.profile, m.provider, m.actions, None, "reference"
    else:
        frames = degraded if degraded is not None else m.frames
        profile, provider, tier = target, "skillcorner", "transfer"
        acts = simulated_skillcorner_actions(m.actions, frames)
        pairing = paired_by_construction(acts)
    goal_map = resolve_defended_goals(frames)
    contacts = detect_ball_contacts(frames, data_profile=profile, gate_table=PASS_ALL)
    resolved = resolve_kicks(
        acts,
        frames,
        provider=provider,
        goal_map=goal_map,
        contacts=contacts,
        data_profile=profile,
        gi_pairing=pairing,
        gate_table=PASS_ALL,
        bracket_ab=bracket_ab if target is None else None,  # the run (1) a / b apply to NATIVE anchors only
        flips=m.flips if target is None else None,  # the staged applied flip (transfer runs pair by construction)
        period_ranges=(m.period_ranges or None) if target is None else None,  # the staged provider range (26.8)
    )
    label = tier if tier_override is None else tier_override
    return PipelineRun(label, profile, provider, resolved, frames, contacts, goal_map)


def _by_action(run: PipelineRun) -> dict:
    return {aid: i for i, aid in enumerate(run.actions["action_id"].to_numpy())}


def score_kick_time_and_taker(m: Part2Match, run: PipelineRun, restarts: pd.DataFrame, k: np.ndarray) -> list[dict]:
    """Runs (2)-(5), (11) (spec sections 5.3, 6, 7) on a run of the SkillCorner rules (provider ``skillcorner``):

    * a TRANSFER run (degraded tracking): ``kick_time`` ``err = |t_kick - K|`` per cell (resolver state x
      sign(delta); ``retake``) and ``taker`` agreement with the native restart player per cell (restart type x
      subtype);
    * the reference ADMISSION arm (the rules on the provider's OWN undegraded tracking, section 5.3 GS / IDSSE
      admission): ``kick_time`` via ``tf79_contact`` only (``err = |t_contact - K|``; ``ball_onset`` is never
      scored against K on the same undegraded track, section 5.2) and the ``taker`` (the kick-contact player vs
      the native player).
    A run of the NATIVE rules (passthrough taker) is not scored here."""
    from silly_kicks.tracking._kick_resolver import taker_cell_state

    if run.provider != "skillcorner":
        return []
    by_id = _by_action(run)
    base = provider_base(m.provider)
    rows = []
    for r, kk in zip(restarts.itertuples(index=False), k, strict=True):
        i = by_id.get(r.restart_id)
        if i is None or r.base == "penalty":
            continue
        row = run.actions.iloc[i]
        if np.isfinite(kk):
            if run.tier == "reference":
                tc = float(row["t_contact"])
                state, err = "tf79_contact", (abs(tc - kk) if np.isfinite(tc) else float("inf"))
            else:
                retake = bool(row.get("kick_retake")) if pd.notna(row.get("kick_retake")) else False
                state = (
                    "retake" if retake else kick_time_cell_state(row["kick_time_source"], float(row["kick_time_delta"]))
                )
                tk = float(row["t_kick"])
                err = abs(tk - kk) if np.isfinite(tk) else float("inf")
            tc_, to_ = float(row["t_contact"]), float(row["t_onset"])
            paired = (
                run.tier != "reference" and np.isfinite(tc_) and np.isfinite(to_)
            )  # never ball_onset vs K, same track
            rows.append(
                score_row(
                    run.profile,
                    str(r.base),
                    "kick_time",
                    state,
                    base=base,
                    tier=run.tier,
                    match_id=m.match_id,
                    err=err,
                    err_contact=abs(tc_ - kk) if paired else float("nan"),
                    err_onset=abs(to_ - kk) if paired else float("nan"),
                )
            )
        if r.player_id is not None and not pd.isna(r.player_id):
            pred = row["taker_player_id"]
            ok = pred is not None and same_id(pred, r.player_id)
            cell = taker_cell_state({"type_id": spadlconfig.actiontype_id[str(r.restart_type)]})
            rows.append(
                score_row(run.profile, str(r.base), "taker", cell, base=base, tier=run.tier, match_id=m.match_id, ok=ok)
            )
    return rows


def _corner_windows(run: PipelineRun, provider_for_identify: str):
    """identify_corners on the RESOLVED rows + add_corner_windows (every cell passing), one row per corner in
    chronological order (the order of the native corner restarts)."""
    from silly_kicks.corners import add_corner_windows, identify_corners

    corners = identify_corners(run.actions, provider=provider_for_identify)
    if corners.empty:
        return corners
    return add_corner_windows(corners, run.frames, run.contacts, data_profile=run.profile, gate_table=PASS_ALL)


def score_first_contact(m: Part2Match, run: PipelineRun, restarts: pd.DataFrame, truth: pd.DataFrame) -> list[dict]:
    """Run (2) first contact (spec sections 9.1, 9.2): the R* winner of ``add_corner_windows`` (every cell passing,
    on the resolved rows of this run) vs the section 5.6 first-touch truth. ``err = |t - t_truth|`` when the team
    (and, off duels, the player) agree, else ``inf``; ``foul`` truth is excluded (reported). Cell: contested x
    taker_known from the R* contact's own margin. IDSSE is the only base (spec 5.3)."""
    from silly_kicks.corners._windows import first_contact_cell_state

    scored = truth[truth["touch_class"].isin(["touch", "duel"])] if not truth.empty else truth
    if scored.empty or run.frames.empty:
        return []
    win = _corner_windows(run, run.provider)
    rs = restarts[restarts["base"] == "corner"].sort_values(["period_id", "t_event"], kind="stable")
    rs = rs.reset_index(drop=True)
    if len(win) != len(rs):
        return []  # the corner population must align one-to-one; a mismatch is not scored (reported by count)
    tr = scored.set_index("restart_id")
    base = provider_base(m.provider)
    rows = []
    for j in range(len(rs)):
        rid = rs["restart_id"].iloc[j]
        if rid not in tr.index:
            continue
        t, w = tr.loc[rid], win.iloc[j]
        state = first_contact_cell_state(float(w["first_contact_attribution_margin"]), bool(w["taker_known"]))
        team_ok = (
            pd.notna(w["winner_team_id"]) and pd.notna(t["team_id"]) and same_id(w["winner_team_id"], t["team_id"])
        )
        player_ok = t["touch_class"] == "duel" or (
            pd.notna(w["winner_player_id"]) and same_id(w["winner_player_id"], t["player_id"])
        )
        tfc = float(w["first_contact_time"])
        err = abs(tfc - float(t["timestamp_seconds"])) if (team_ok and player_ok and np.isfinite(tfc)) else float("inf")
        restart_type = "corner_short" if str(w["corner_kind"]) == "short" else "corner_crossed"
        rows.append(
            score_row(
                run.profile,
                restart_type,
                "first_contact",
                state,
                base=base,
                tier=run.tier,
                match_id=m.match_id,
                err=err,
                parent_cell_state=state.rsplit("_taker", 1)[0] + "_taker_known",
            )
        )
    return rows


def score_subtype(m: Part2Match, run: PipelineRun, restarts: pd.DataFrame) -> list[dict]:
    """Run (6) subtype (spec section 11.2): ``refine_restart_types``' tracking rule on default-typed copies of the
    corner / free-kick rows (every cell passing) vs the profile truth (IDSSE next-touch location; GS label). One
    PRECISION row per prediction, ``pred`` / ``truth`` kept for the majority-baseline comparison in the reduce."""
    from silly_kicks.spadl import refine_restart_types

    rs = restarts[restarts["base"].isin(["corner", "free_kick"])]
    truth = _crossed_truth_idsse(m, rs) if provider_base(m.provider) == "idsse" else _crossed_truth_pff(m, rs)
    if not truth or run.frames.empty:
        return []
    acts = m.actions.copy()
    ts = acts["type_source"] if "type_source" in acts.columns else pd.Series("native", index=acts.index)
    acts["type_source"] = np.where(acts["action_id"].isin(list(truth)), "default", ts.astype(object))
    out = refine_restart_types(
        acts, run.frames, provider=run.provider, data_profile=run.profile, gate_table=PASS_ALL, goal_map=run.goal_map
    )
    by_time = {
        (int(p), round(float(t), 6)): tid
        for p, t, tid in zip(out["period_id"], out["time_seconds"], out["type_id"], strict=False)
    }
    base = provider_base(m.provider)
    rows = []
    for r in rs.itertuples(index=False):
        if r.restart_id not in truth:
            continue
        tid = by_time.get((_i(r.period_id), round(_f(r.t_event), 6)))
        if tid is None:
            continue
        pred = "crossed" if _type_name(tid).endswith("_crossed") else "short"
        rows.append(
            score_row(
                run.profile,
                str(r.base),
                "subtype",
                pred,
                base=base,
                tier=run.tier,
                match_id=m.match_id,
                ok=pred == truth[r.restart_id],
                pred=pred,
                truth=truth[r.restart_id],
            )
        )
    return rows


def merge_simulation(actions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run (7) tier-2 merge population (spec section 11.4): from native restarts whose own event is the shot
    (``shot_freekick`` / ``shot_penalty``) build the SkillCorner shape -- a derived free-kick row + the restarting
    team's shot (a POSITIVE pair) -- and from a native free kick followed by a SEPARATE shot keep the two events
    (the derived free kick + a real shot: a NEGATIVE pair). Returns the simulated actions and
    ``{derived_action_id: is_one_restart_shot}``."""
    acts = actions.sort_values(["period_id", "time_seconds"], kind="stable").reset_index(drop=True)
    if acts.empty:
        return acts, {}
    names = acts["type_id"].map(_type_name)
    sim_rows, truth_pairs = [], {}
    next_id = int(acts["action_id"].max()) + 1
    for i in range(len(acts)):
        nm, a = names.iloc[i], acts.iloc[i]
        if nm in ("shot_freekick", "shot_penalty"):
            fk = a.copy()
            fk["action_id"] = next_id
            fk["type_id"] = spadlconfig.actiontype_id["freekick_short"]
            fk["action_provenance"], fk["type_source"] = "derived", "default"
            fk["start_x"], fk["start_y"], fk["player_id"] = np.nan, np.nan, pd.NA
            fk["time_seconds"] = float(a["time_seconds"]) - 0.5
            shot = a.copy()
            shot["type_id"] = spadlconfig.actiontype_id["shot"]
            sim_rows.extend([fk, shot])
            truth_pairs[next_id] = True
            next_id += 1
        elif nm in ("freekick_short", "freekick_crossed") and i + 1 < len(acts) and names.iloc[i + 1] == "shot":
            fk = a.copy()
            fk["action_provenance"], fk["type_source"] = "derived", "default"
            fk["start_x"], fk["start_y"], fk["player_id"] = np.nan, np.nan, pd.NA
            sim_rows.append(fk)
            truth_pairs[int(a["action_id"])] = False
        else:
            sim_rows.append(a)
    return pd.DataFrame(sim_rows).reset_index(drop=True), truth_pairs


def score_merge(m: Part2Match, target: DataProfile, frames: pd.DataFrame, *, tier: str = "transfer") -> list[dict]:
    """Run (7) merge (spec section 11.4): the merge population simulation, then the restart pipeline as production
    runs it -- ``resolve_kicks`` (gated taker) then ``refine_restart_types`` (the gated origin it computes) --
    every cell passing. One row per MERGE performed: correct iff the pair was one restart-shot (precision; the form
    events_first / tracking is the cell)."""
    from silly_kicks.spadl import refine_restart_types
    from silly_kicks.tracking._ball_contact import detect_ball_contacts
    from silly_kicks.tracking._gk_resolve import resolve_defended_goals
    from silly_kicks.tracking._kick_resolver import resolve_kicks

    sim, truth_pairs = merge_simulation(m.actions)
    if not truth_pairs or frames.empty:
        return []
    for c, v in (("action_provenance", "native"), ("type_source", "native")):
        if c not in sim.columns:
            sim[c] = v
    goal_map = resolve_defended_goals(frames)
    contacts = detect_ball_contacts(frames, data_profile=target, gate_table=PASS_ALL)
    sim = simulated_bracket_ids(sim, frames)
    res = resolve_kicks(
        sim,
        frames,
        provider="skillcorner",
        goal_map=goal_map,
        contacts=contacts,
        data_profile=target,
        gi_pairing=paired_by_construction(sim),
        gate_table=PASS_ALL,
    )
    out = refine_restart_types(
        res, frames, provider="skillcorner", data_profile=target, gate_table=PASS_ALL, goal_map=goal_map
    )
    base = provider_base(m.provider)
    rows = []
    for _, mr in out[out["merged_from_derived_action_id"].notna()].iterrows():
        did = int(mr["merged_from_derived_action_id"])
        if did not in truth_pairs:
            continue
        form = "events_first" if _type_name(mr["type_id"]) == "shot_penalty" else "tracking"
        rows.append(
            score_row(
                target,
                "free_kick",
                "merge",
                form,
                base=base,
                tier=tier,
                match_id=m.match_id,
                ok=bool(truth_pairs[did]),
            )
        )
    return rows


def simulated_bracket_ids(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Give every DERIVED restart row the GI frame ids of its section 5.3 simulated bracket (lo = the previous
    event, hi = the next event), mapped to the nearest ball frame."""
    from scripts._tf46p2_common import simulate_skillcorner_anchor

    out = actions.copy()
    out["type_name"] = out["type_id"].map(_type_name)
    anchors = simulate_skillcorner_anchor(out).set_index("action_id")
    out = out.drop(columns=["type_name"])
    ball = frames[frames["is_ball"].to_numpy(dtype=bool)]
    fid = {
        _i(p): (
            g.sort_values("time_seconds")["time_seconds"].to_numpy(float),
            g.sort_values("time_seconds")["frame_id"].to_numpy(),
        )
        for p, g in ball.groupby("period_id", sort=True)
    }

    def _nearest(p, t):
        hit = fid.get(int(p))
        if hit is None or not np.isfinite(t):
            return pd.NA
        times, ids = hit
        return int(ids[int(np.clip(np.searchsorted(times, t), 0, len(times) - 1))])

    lo_ids, hi_ids = [], []
    for aid, prov, p in zip(
        out["action_id"], out.get("action_provenance", pd.Series([None] * len(out))), out["period_id"], strict=False
    ):
        if prov == "derived" and aid in anchors.index:
            lo_ids.append(_nearest(p, _f(anchors.loc[aid, "kick_bracket_lo"])))
            hi_ids.append(_nearest(p, _f(anchors.loc[aid, "kick_bracket_hi"])))
        else:
            lo_ids.append(pd.NA)
            hi_ids.append(pd.NA)
    out["gi_marker_possession_frame_end"] = pd.array(lo_ids, dtype="Int64")
    out["gi_next_possession_frame_start"] = pd.array(hi_ids, dtype="Int64")
    return out


def score_penalty(m: Part2Match, run: PipelineRun) -> list[dict]:
    """Run (12) penalty (spec section 12): ``detect_penalties`` (every cell passing; SkillCorner goal-time semantics
    on a transfer run) vs the native penalties (DFL ``Penalty`` / PFF, SPADL ``shot_penalty`` with the section 25.2
    result). Detection PRECISION: one row per detection, correct iff a native penalty of the same period and
    attacking team lies within ``W`` of the release (``stratum`` in-play / shoot-out). Attribution: per matched
    detection, goal vs missed against the native result. Split: per matched MISSED penalty with a detector split,
    saved vs off-target against the native keeper outcome (a keeper action within ``W``)."""
    from silly_kicks.tracking._penalty_detector import detect_penalties

    if run.frames.empty:
        return []
    det = detect_penalties(
        m.actions,
        run.frames,
        provider=run.provider,
        goal_map=run.goal_map,
        contacts=run.contacts,
        data_profile=run.profile,
        gate_table=PASS_ALL,
        pitch_length=m.pitch_length,
    )
    if det.empty:
        return []
    acts = m.actions
    pens = acts[acts["type_id"] == spadlconfig.actiontype_id["shot_penalty"]]
    saves = acts[acts["type_id"] == spadlconfig.actiontype_id["keeper_save"]]
    w = RATIFIED["penalty_W_s"]
    base = provider_base(m.provider)
    rows = []
    for _, d in det.iterrows():
        cand = pens[
            (pens["period_id"] == d["period_id"]) & ((pens["time_seconds"] - float(d["release_time"])).abs() <= w)
        ]
        team = d["attacking_team_id"]
        cand = cand[cand["team_id"].map(lambda t, team=team: same_id(t, team))]
        hit = not cand.empty
        stratum = str(d["stratum"])
        kw: dict[str, Any] = dict(base=base, tier=run.tier, match_id=m.match_id, stratum=stratum)
        rows.append(score_row(run.profile, "penalty", "penalty", "detection", ok=hit, **kw))
        if not hit:
            continue
        truth_goal = int(cand["result_id"].iloc[0]) == spadlconfig.result_id["success"]
        pred = d["outcome"]
        if pd.notna(pred):
            rows.append(
                score_row(run.profile, "penalty", "penalty", "attribution", ok=(pred == "goal") == truth_goal, **kw)
            )
        if not truth_goal and pred in ("saved", "off_target"):
            t0 = float(cand["time_seconds"].iloc[0])
            near = (saves["period_id"] == d["period_id"]) & ((saves["time_seconds"] - t0).abs() <= w)
            rows.append(
                score_row(run.profile, "penalty", "penalty", "split", ok=(pred == "saved") == bool(near.any()), **kw)
            )
    return rows


def score_origin(m: Part2Match, run: PipelineRun, restarts: pd.DataFrame) -> list[dict]:
    """Runs (10), (13) origin (spec section 10.2): ``add_restart_coordinates`` on the RESOLVED rows of this run
    (every origin cell passing) vs the section 10.2 truth per type -- corner: the 1 m corner arc; throw-in: the
    touchline y and the native x; goal kick: the native location (the goal-area constraint is the tier's own
    clamp); free kick: the native execution spot. GS has truth for GEOMETRY ONLY (the corner arc, the throw-in y;
    C-43). ``err`` = metres; cell = restart type x origin tier. The reduce applies the fixed median <= 2 m AND
    share-within-5 m bar."""
    from silly_kicks.spadl.utils import add_restart_coordinates

    if run.frames.empty or restarts.empty:
        return []
    from silly_kicks.tracking._gk_geometry import gated_origin

    enr = add_restart_coordinates(run.actions, frames=run.frames, data_profile=run.profile, gate_table=PASS_ALL)
    pre = gated_origin(run.actions, run.frames, run.profile, PASS_ALL)  # the tier each row WOULD ship, pre-tripwire
    gated_before = {
        aid: src
        for aid, src, ev in zip(
            run.actions["action_id"], pre["start_coord_source"], pre["origin_evidence"], strict=True
        )
        if ev is not None and not pd.isna(ev)
    }
    by_id = {aid: i for i, aid in enumerate(enr["action_id"].to_numpy())}
    gs_geometry_only = provider_base(m.provider) == "gradientsports"
    base = provider_base(m.provider)
    rows = []
    for r in restarts.itertuples(index=False):
        i = by_id.get(r.restart_id)
        if i is None or r.base == "penalty":
            continue
        e = enr.iloc[i]
        if e.get("start_coord_source") == "tripwire_reverted" and gated_before.get(r.restart_id) is not None:
            # the rule produced a gated origin OUTSIDE its Law region: a failed origin of that tier
            rows.append(
                score_row(
                    run.profile,
                    str(r.base),
                    "origin",
                    gated_before[r.restart_id],
                    base=base,
                    tier=run.tier,
                    match_id=m.match_id,
                    err=float("inf"),
                )
            )
            continue
        if pd.isna(e.get("origin_evidence")):
            continue  # no gated tier produced a candidate for this row
        ox, oy = float(e["enriched_start_x"]), float(e["enriched_start_y"])
        if r.base == "corner":
            tx = spadlconfig.field_length if ox > spadlconfig.field_length / 2 else 0.0
            ty = spadlconfig.field_width if oy > spadlconfig.field_width / 2 else 0.0
            err = max(float(np.hypot(ox - tx, oy - ty)) - 1.0, 0.0)  # distance to the 1 m corner arc
        elif r.base == "throw_in":
            ty = spadlconfig.field_width if oy > spadlconfig.field_width / 2 else 0.0
            err = abs(oy - ty) if gs_geometry_only else float(np.hypot(ox - _f(r.x), oy - ty))
        elif gs_geometry_only or not (np.isfinite(r.x) and np.isfinite(r.y)):
            continue  # GS: no free-kick spot / goal-kick spot truth (C-43)
        else:
            err = float(np.hypot(ox - _f(r.x), oy - _f(r.y)))
        rows.append(
            score_row(
                run.profile,
                str(r.base),
                "origin",
                str(e["start_coord_source"]),
                base=base,
                tier=run.tier,
                match_id=m.match_id,
                err=err,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# The reduce: score rows -> the committed (k, n) aggregate (AGGREGATE_COLUMNS)
# ---------------------------------------------------------------------------

#: Calibrated tolerances (spec section 16.1): ``(quantile, ceiling, centre)``. The tolerance is calibrated on the
#: calibration matches at the section 3 bar quantile and scored held-out (LOMO); the SHIPPED constant is the
#: all-match calibration, which must sit within its ceiling (``calibration_metric`` <= ``calibration_ceiling``).
_CALIBRATED: dict[tuple[str, str | None], tuple[float, float, bool]] = {
    ("kick_time", "k_gate"): (0.90, RATIFIED["tau_event_ceiling_s"], True),
    ("kick_time", None): (0.90, RATIFIED["tau_detector_ceiling_s"], False),
    ("first_contact", None): (0.90, RATIFIED["tau_fc_ceiling_s"], False),
}


def _calibration_spec(signal: str, cell_state: str):
    return _CALIBRATED.get((signal, cell_state)) or _CALIBRATED.get((signal, None))


def _quantile(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else float("inf")


def _ok_calibrated(block: pd.DataFrame, q: float, center: bool) -> tuple[pd.Series, float, float]:
    """LOMO verdicts for a calibrated block: each held-out match is scored with the tolerance (and centre)
    calibrated on the OTHER matches. Returns ``(ok per row, lomo share, all-match calibrated tolerance)``."""
    err = block["err"].to_numpy(dtype=float)
    mids = block["match_id"].to_numpy()
    ok = np.zeros(len(block), dtype=bool)
    for mid in pd.unique(mids):
        held = mids == mid
        others = err[~held]
        if not np.isfinite(others).any():
            continue
        c = float(np.median(others[np.isfinite(others)])) if center else 0.0
        tol = _quantile(np.abs(others - c), q)
        ok[held] = np.abs(err[held] - c) <= tol
    c_all = float(np.median(err[np.isfinite(err)])) if center and np.isfinite(err).any() else 0.0
    calib = _quantile(np.abs(err - c_all), q)
    return pd.Series(ok, index=block.index), float(ok.mean()) if len(ok) else float("nan"), calib


def _gs_admission(verdicts: pd.DataFrame) -> dict[str, bool]:
    """GS admission PER SIGNAL (spec section 5.3): GS is a transfer base for a signal only when its own tier-1
    (reference) rows for that signal pass the section 3 gate on GS truth. ``verdicts`` carries the FINAL per-row
    verdict ``_ok`` (a calibrated signal's LOMO verdict included). ``first_contact`` / ``z_overhead`` are never
    admitted (IDSSE-only / off for GS)."""
    out: dict[str, bool] = {}
    ref = verdicts[(verdicts["tier"] == "reference") & (verdicts["base"] == "gradientsports")]
    for sig, g in ref.groupby("signal", sort=True):
        if sig in ("first_contact", "z_overhead"):
            out[str(sig)] = False
            continue
        out[str(sig)] = bool(cell_passes(pd.DataFrame({"match_id": g["match_id"], "ok": g["_ok"].astype(bool)})))
    return out


def _baseline_margin_lb(block: pd.DataFrame, *, b: int = 2000, seed: int = 46) -> float:
    """Spec section 11.2: the match-cluster bootstrap 2.5th percentile of (rule precision in this direction - the
    per-cell majority-class baseline precision). The baseline predicts the majority truth class of the cell for
    every item, so its precision is that class's share."""
    if block.empty or block["truth"].isna().all():
        return float("nan")
    direction = str(block["cell_state"].iloc[0])
    mids = block["match_id"].to_numpy()
    ok = block["ok"].fillna(False).astype(bool).to_numpy()
    is_dir = (block["truth"].astype(str) == direction).to_numpy()
    uniq = pd.unique(mids)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(b):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([np.flatnonzero(mids == u) for u in pick])
        if not len(sel):
            continue
        prec = ok[sel].mean()
        base_share = max(is_dir[sel].mean(), 1.0 - is_dir[sel].mean())
        diffs.append(prec - base_share)
    return float(np.quantile(diffs, 0.025)) if diffs else float("nan")


def _block_verdicts(sig: str, cs: str, block: pd.DataFrame) -> tuple[pd.Series, float, float, float]:
    """``(ok per row, lomo_metric, calibration_metric, calibration_ceiling)`` for one cell block."""
    if sig == "footprint":
        return pd.Series(True, index=block.index), float(block["footprint_inside_share"].iloc[0]), np.nan, np.nan
    spec = _calibration_spec(sig, cs)
    if spec is not None:
        q, ceiling, center = spec
        ok, lomo, calib = _ok_calibrated(block, q, center)
        return ok, lomo, calib, ceiling
    if sig == "origin":
        err = block["err"].to_numpy(dtype=float)
        ok = pd.Series(err <= RATIFIED["origin_share_m"], index=block.index)
        calib = float(np.median(err)) if len(err) else float("nan")
        return ok, float(ok.mean()), calib, RATIFIED["origin_median_m"]
    ok = block["ok"].fillna(False).astype(bool)
    return ok, float(ok.mean()) if len(ok) else float("nan"), np.nan, np.nan


def _median_or_nan(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna()
    return float(v.median()) if len(v) else float("nan")


def aggregate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Score rows -> the committed aggregate in ``AGGREGATE_COLUMNS`` form: one row per (cell, base, stratum,
    match) with ``(k, n)``, plus the cell-level ``lomo_metric``, ``calibration_metric`` / ``calibration_ceiling``,
    ``admitted`` (GS per signal, from the final verdicts) and ``baseline_margin_lb`` (subtype). Pure; no I/O."""
    if scores.empty:
        return pd.DataFrame(columns=list(AGGREGATE_COLUMNS))
    keys = [*_PROFILE_COLS, "restart_type", "signal", "cell_state", "base", "tier", "stratum", "parent_cell_state"]
    blocks = []
    verdict = pd.Series(False, index=scores.index)
    for kv, block in scores.groupby(keys, sort=True, dropna=False):
        key = dict(zip(keys, kv, strict=True))
        ok, lomo, calib, ceiling = _block_verdicts(str(key["signal"]), str(key["cell_state"]), block)
        verdict.loc[block.index] = ok.to_numpy(dtype=bool)
        blocks.append((key, block, ok, lomo, calib, ceiling))
    admission = _gs_admission(scores.assign(_ok=verdict))
    out_rows: list[dict] = []
    for key, block, ok, lomo, calib, ceiling in blocks:
        sig, cs = str(key["signal"]), str(key["cell_state"])
        baseline = _baseline_margin_lb(block) if sig == "subtype" else float("nan")
        admitted = bool(admission.get(sig, False)) if key["base"] == "gradientsports" else key["base"] == "idsse"
        for mid, g in block.groupby("match_id", sort=True):
            okg = ok.loc[g.index]
            out_rows.append(
                {
                    **{c: key[c] for c in _PROFILE_COLS},
                    "restart_type": key["restart_type"],
                    "signal": sig,
                    "cell_state": cs,
                    "base": key["base"],
                    "admitted": admitted,
                    "stratum": key["stratum"],
                    "parent_cell_state": key["parent_cell_state"],
                    "match_id": mid,
                    "k": int(okg.sum()),
                    "n": len(g),
                    "tier": key["tier"],
                    "lomo_metric": lomo,
                    "calibration_metric": calib,
                    "calibration_ceiling": ceiling,
                    "footprint_inside_share": float(g["footprint_inside_share"].iloc[0]),
                    "baseline_margin_lb": baseline,
                    "agree_err_contact_s": _median_or_nan(g["err_contact"]) if sig == "kick_time" else float("nan"),
                    "agree_err_onset_s": _median_or_nan(g["err_onset"]) if sig == "kick_time" else float("nan"),
                }
            )
    out = pd.DataFrame(out_rows, columns=list(AGGREGATE_COLUMNS))
    out["gi_release"] = out["gi_release"].astype("object")
    return out


def gs_admission_table(scores: pd.DataFrame) -> pd.DataFrame:
    """The run (2) GS admission report per signal (from the final verdicts, calibrated signals included)."""
    if scores.empty:
        return pd.DataFrame(columns=["signal", "gs_admitted"])
    keys = [*_PROFILE_COLS, "restart_type", "signal", "cell_state", "base", "tier", "stratum", "parent_cell_state"]
    verdict = pd.Series(False, index=scores.index)
    for kv, block in scores.groupby(keys, sort=True, dropna=False):
        key = dict(zip(keys, kv, strict=True))
        ok, *_ = _block_verdicts(str(key["signal"]), str(key["cell_state"]), block)
        verdict.loc[block.index] = ok.to_numpy(dtype=bool)
    adm = _gs_admission(scores.assign(_ok=verdict))
    return pd.DataFrame(
        [{"signal": k, "gs_admitted": v} for k, v in sorted(adm.items())], columns=["signal", "gs_admitted"]
    )


def pooled_noninferior(child: pd.DataFrame, parent: pd.DataFrame) -> bool:
    """Re-export of the section 15.3 non-inferiority test for the drivers' combined-base report."""
    return bool(noninferior(child, parent))


def write_json(path: Path, obj: Any) -> None:
    """ASCII JSON writer for the run reports."""
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
