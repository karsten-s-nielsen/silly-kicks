"""TF-35 off-ball run detection + valuation (ADR-042).

Two pure primitives, kept apart so the expensive half is optional:

* :func:`detect_off_ball_runs` -- geometry only. One row per qualifying
  (action, runner) pair, positions emitted in SPADL action-LTR (ADR-028).
* :func:`value_off_ball_runs` -- attaches a pitch-control x threat value to each
  detected run, plus the run's role relative to the pass receiver.

The valuation answers "how much dangerous space did this runner own at the moment
of the pass", which is why it is a MAX over the region the runner controls rather
than an influence-weighted mean -- see :class:`RunValuationParams` for the
consequence (the region floor exists only because of that choice) and ADR-042 for
the recorded v2 fork.

Attribution: Sumpter / Twelve (Soccermatics Pro, module 16.2) as a practitioner
anchor for run valuation via controlled-space threat; Esposito et al. 2026 for the
target/disruptive framing. See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id, canonical_id_series, ids_equal, ids_match

from ..spadl import config as spadlconfig
from ._action_orientation import FIELD_LENGTH, FIELD_WIDTH, acting_team_attacks_rtl
from ._off_ball_runs import _prepare_run_candidates
from ._warnings import RunValueCoverageWarning

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..xthreat import ExpectedThreat
    from .pitch_control import PitchControlCache

__all__ = [
    "RunValuationParams",
    "action_level_context",
    "detect_off_ball_runs",
    "value_off_ball_runs",
]

#: Per-method region floors. ONLY methods with a defensible floor appear here --
#: ``voronoi`` is deliberately absent because its per-player influence is binary
#: {0, 1}, so any floor in (0, 1] selects the same cells and the knob silently
#: stops meaning anything.
_FLOOR_BY_METHOD: dict[str, float] = {"spearman": 0.1}

#: Columns emitted by :func:`detect_off_ball_runs`, in order.
RUN_COLUMNS = [
    "game_id",
    "period_id",
    "action_id",
    "player_id",
    "run_start_x",
    "run_start_y",
    "run_end_x",
    "run_end_y",
    "displacement_m",
    "duration_s",
    "mean_speed_ms",
    "peak_speed_ms",
    "peak_speed_source",
    "toward_goal",
]

#: Columns added by :func:`value_off_ball_runs`.
RUN_VALUE_COLUMNS = ["role", "is_receiver", "run_value", "enabled_pass_credit"]

_PassRole = Literal["target", "disruptive"]


@dataclass(frozen=True)
class RunValuationParams:
    """Parameters for TF-35 off-ball run detection and valuation.

    Parameters
    ----------
    pre_seconds : float, default 1.5
        Length of the pre-action window a run is measured over. Matches TF-4's
        default so the two families describe the same window.
    min_displacement_m : float, default 3.0
        Straight-line first-to-last displacement a candidate must cover.
    min_peak_speed_ms : float, default 5.56
        Peak-speed gate (5.56 m/s == 20 km/h, the usual sprint threshold). Set to
        ``0.0`` to disable the gate and recover TF-4's displacement-only domain.
    region_influence_floor : float or None, default None
        Per-player pitch-control influence above which a cell counts as "owned by
        this runner". ``None`` resolves from ``pitch_control_method`` via
        :meth:`resolved_region_floor`.
    pitch_control_method : str, default "spearman"
        Pitch-control flavour used for the valuation surface.

    Notes
    -----
    **The 0.1 default floor is a spec-time starting value, NOT calibrated.** Its
    calibration is the sensitivity probe recorded in ADR-042 (0.05 / 0.1 / 0.2);
    treat cross-study comparisons of absolute ``run_value`` with that in mind.

    The floor apparatus exists ONLY because ``run_value`` is a max over a
    thresholded region. An influence-weighted mean would delete the knob entirely
    (every cell contributes in proportion to influence, so there is nothing to
    threshold). That trade -- peak opportunity vs average ownership -- is the
    recorded v2 fork in ADR-042, not an oversight.

    Examples
    --------
    Disable the sprint gate to reproduce TF-4's run domain::

        from silly_kicks.tracking import RunValuationParams

        params = RunValuationParams(min_peak_speed_ms=0.0)
    """

    pre_seconds: float = 1.5
    min_displacement_m: float = 3.0
    min_peak_speed_ms: float = 5.56
    region_influence_floor: float | None = None
    pitch_control_method: str = "spearman"

    def __post_init__(self) -> None:
        if not self.pre_seconds > 0:
            raise ValueError(f"RunValuationParams: pre_seconds must be > 0, got {self.pre_seconds!r}")
        if self.min_displacement_m < 0:
            raise ValueError(f"RunValuationParams: min_displacement_m must be >= 0, got {self.min_displacement_m!r}")
        if self.min_peak_speed_ms < 0:
            raise ValueError(f"RunValuationParams: min_peak_speed_ms must be >= 0, got {self.min_peak_speed_ms!r}")
        if self.region_influence_floor is not None and not 0.0 < self.region_influence_floor <= 1.0:
            raise ValueError(
                f"RunValuationParams: region_influence_floor must lie in (0, 1], got {self.region_influence_floor!r}"
            )

    def resolved_region_floor(self) -> float:
        """Return the effective region floor, failing loud when none is calibrated.

        Examples
        --------
        >>> from silly_kicks.tracking import RunValuationParams
        >>> RunValuationParams().resolved_region_floor()
        0.1
        """
        if self.region_influence_floor is not None:
            return float(self.region_influence_floor)
        try:
            return _FLOOR_BY_METHOD[self.pitch_control_method]
        except KeyError:
            raise ValueError(
                f"RunValuationParams: no calibrated floor for pitch_control_method="
                f"{self.pitch_control_method!r} (calibrated: {sorted(_FLOOR_BY_METHOD)}). "
                "Pass region_influence_floor= explicitly. Note that 'voronoi' per-player "
                "influence is binary {0, 1}, so every floor in (0, 1] selects the same "
                "cells -- the knob does not mean what it means for a continuous flavour."
            ) from None


def _empty_runs() -> pd.DataFrame:
    """Empty frame carrying the full detection schema (so consumers can chain)."""
    return pd.DataFrame(
        {
            "game_id": pd.Series(dtype="object"),
            "period_id": pd.Series(dtype="int64"),
            "action_id": pd.Series(dtype="object"),
            "player_id": pd.Series(dtype="object"),
            "run_start_x": pd.Series(dtype="float64"),
            "run_start_y": pd.Series(dtype="float64"),
            "run_end_x": pd.Series(dtype="float64"),
            "run_end_y": pd.Series(dtype="float64"),
            "displacement_m": pd.Series(dtype="float64"),
            "duration_s": pd.Series(dtype="float64"),
            "mean_speed_ms": pd.Series(dtype="float64"),
            "peak_speed_ms": pd.Series(dtype="float64"),
            "peak_speed_source": pd.Series(dtype="object"),
            "toward_goal": pd.Series(dtype="bool"),
        }
    )


def detect_off_ball_runs(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    params: RunValuationParams | None = None,
) -> pd.DataFrame:
    """Detect qualifying off-ball runs in the pre-action window, one row per runner.

    Candidacy is the SHARED TF-4/TF-35 predicate
    (:func:`silly_kicks.tracking._off_ball_runs._prepare_run_candidates`): same team
    as the actor, not the actor, not a goalkeeper, positions present, ball alive. A
    candidate becomes a run when its first-to-last displacement clears
    ``min_displacement_m`` AND its peak speed clears ``min_peak_speed_ms``.

    Emitted positions are re-projected into SPADL **action-LTR** (ADR-028), so
    ``run_end_x > run_start_x`` means "toward the goal the acting team attacks" for
    both teams. Ids pass through at their source dtype (ADR-019).

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions.
    frames : pd.DataFrame
        Long-form tracking frames.
    home_team_id : int or str
        Retained for signature symmetry with the TF-4 family; orientation itself is
        derived from the frames' own ``team_attacking_direction`` (ADR-028), never
        from home/away identity.
    params : RunValuationParams, optional

    Returns
    -------
    pd.DataFrame
        One row per qualifying (action, runner); see ``RUN_COLUMNS``. Empty (with
        the full schema) when nothing qualifies.

    Notes
    -----
    ``peak_speed_source`` records HOW the peak was obtained: ``"measured"`` from the
    frames' ``speed`` column, or ``"displacement_rate"`` when every window sample
    for that runner is NaN and the gate falls back to ``displacement / duration``.
    The fallback systematically UNDER-states peak speed (an average cannot exceed a
    maximum), so a provider with sparse speeds loses runs rather than gaining
    them -- the bias is emitted as data, not buried in a docstring.

    Examples
    --------
    Detect off-ball runs around each action from a linked match's frames::

        from silly_kicks.tracking import detect_off_ball_runs

        runs = detect_off_ball_runs(actions, frames, home_team_id=1)
        runs[["action_id", "player_id", "displacement_m", "peak_speed_ms"]].head()
    """
    from .utils import slice_around_event

    if params is None:
        params = RunValuationParams()
    if len(actions) == 0 or len(frames) == 0:
        return _empty_runs()

    records: list[dict] = []
    for game_id, game_actions in actions.groupby("game_id", sort=False):
        game_frames = frames[ids_equal(frames["game_id"], pd.Series(game_id, index=frames.index))]
        if game_frames.empty:
            continue

        sliced = slice_around_event(game_actions, game_frames, pre_seconds=params.pre_seconds, post_seconds=0.0)
        if sliced.empty:
            continue

        # Dead-ball tagging must read the BALL rows, so it happens before they are dropped
        # (identical rule to the TF-4 kernel: the frame nearest the action clock decides).
        dead_actions: set = set()
        if "ball_state" in game_frames.columns:
            ball_in_window = sliced[sliced["is_ball"].astype(bool)]
            if not ball_in_window.empty:
                closest_idx = ball_in_window["time_offset_seconds"].abs().groupby(ball_in_window["action_id"]).idxmin()
                closest = ball_in_window.loc[closest_idx]
                dead_actions = set(closest.loc[closest["ball_state"] == "dead", "action_id"])

        sliced = sliced[~sliced["is_ball"].astype(bool)].copy()
        actor = game_actions[["action_id", "player_id", "team_id"]].rename(
            columns={"player_id": "actor_player_id", "team_id": "action_team_id"}
        )
        sliced = sliced.merge(actor, on="action_id", how="left")
        teammates = _prepare_run_candidates(sliced)
        if teammates.empty:
            continue

        flip_by_action = dict(
            zip(
                game_actions["action_id"].to_numpy(),
                acting_team_attacks_rtl(game_actions, game_frames).to_numpy(dtype=bool),
                strict=False,
            )
        )
        period_by_action = dict(
            zip(game_actions["action_id"].to_numpy(), game_actions["period_id"].to_numpy(), strict=False)
        )

        for (aid, pid), grp in teammates.groupby(["action_id", "player_id"], sort=False):
            if aid in dead_actions or len(grp) < 2:
                continue
            ordered = grp.sort_values("time_seconds")
            x0, y0 = float(ordered["x"].iloc[0]), float(ordered["y"].iloc[0])
            x1, y1 = float(ordered["x"].iloc[-1]), float(ordered["y"].iloc[-1])
            displacement = float(np.hypot(x1 - x0, y1 - y0))
            if displacement < params.min_displacement_m:
                continue

            duration = float(ordered["time_seconds"].iloc[-1] - ordered["time_seconds"].iloc[0])
            mean_speed = displacement / duration if duration > 0 else np.nan

            speeds = pd.to_numeric(ordered.get("speed", pd.Series(dtype="float64")), errors="coerce")
            finite = speeds[np.isfinite(speeds)] if len(speeds) else speeds
            if len(finite):
                peak_speed = float(finite.max())
                peak_source = "measured"
            else:
                peak_speed = float(mean_speed)
                peak_source = "displacement_rate"
            if not (peak_speed >= params.min_peak_speed_ms):
                continue

            # ADR-028: emit in the ACTING team's LTR frame, never the frame convention.
            if flip_by_action.get(aid, False):
                sx, sy = FIELD_LENGTH - x0, FIELD_WIDTH - y0
                ex, ey = FIELD_LENGTH - x1, FIELD_WIDTH - y1
            else:
                sx, sy, ex, ey = x0, y0, x1, y1

            records.append(
                {
                    "game_id": game_id,
                    "period_id": period_by_action.get(aid),
                    "action_id": aid,
                    "player_id": pid,
                    "run_start_x": sx,
                    "run_start_y": sy,
                    "run_end_x": ex,
                    "run_end_y": ey,
                    "displacement_m": displacement,
                    "duration_s": duration,
                    "mean_speed_ms": mean_speed,
                    "peak_speed_ms": peak_speed,
                    "peak_speed_source": peak_source,
                    "toward_goal": bool(ex > sx),
                }
            )

    if not records:
        return _empty_runs()
    out = pd.DataFrame.from_records(records, columns=RUN_COLUMNS).reset_index(drop=True)
    return _restore_player_id_dtype(out, frames)


def _restore_player_id_dtype(runs: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Emit ``player_id`` as Int64 when the frame column is a NaN-coded integer.

    Tracking frames carry ``player_id`` NaN on ball rows, which forces the whole column
    to float64 on every numeric-id provider -- so a naive passthrough would emit
    ``101.0`` and break every downstream id join. Same rule (and the same lossless,
    NA-safe conversion) as ``resolve_next_touch_receiver``'s dtype contract; genuine
    string ids pass through untouched.
    """
    dtype = frames["player_id"].dtype
    is_plain_int = pd.api.types.is_integer_dtype(dtype) and not isinstance(dtype, pd.Int64Dtype)
    if is_plain_int or pd.api.types.is_float_dtype(dtype):
        out = runs.copy()
        out["player_id"] = out["player_id"].astype("Int64")
        return out
    return runs


def _safe_index_of(player_ids: np.ndarray | None, player_id) -> int | None:
    """Position of ``player_id`` in ``player_ids`` (ADR-019 dtype-safe), or ``None`` if absent.

    A thin index-or-``None`` helper over the shared ``ids_match`` seam. The caller needs the
    *index* to slice ``per_player_influence`` and to degrade an absent runner to a NaN value,
    whereas ``PitchControlSurface.player_surface`` returns the array and RAISES on a miss. An
    NA id (e.g. a ball row) matches nothing and yields ``None`` (``ids_match`` resolves an NA
    scalar to an all-False mask).
    """
    if player_ids is None:
        return None
    idx = np.where(ids_match(player_ids, player_id).to_numpy())[0]
    return int(idx[0]) if len(idx) else None


def action_level_context(actions: pd.DataFrame, xt: ExpectedThreat) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Per-action receiver, on-domain mask and floored pass threat gain.

    ONE authority for the valuation domain, shared by :func:`value_off_ball_runs` and
    the ``add_off_ball_run_values`` aggregator. They must not drift: the aggregator has
    to emit ``0`` for an on-domain action that produced no runs, which the runs frame
    cannot express (it has no row for that action at all), so it necessarily re-derives
    the mask -- and a second copy of the rule is exactly how the two would diverge.

    Domain: a ``pass``/``cross`` with ``result_id == success`` whose next same-team
    touch resolves to a receiver.

    Examples
    --------
    Resolve the per-action receiver, on-domain mask and floored pass-threat gain::

        from silly_kicks.tracking._run_values import action_level_context

        receiver, on_domain, credit = action_level_context(actions, xt)
    """
    from ..spadl.utils import resolve_next_touch_receiver
    from ..xthreat import values_at_points

    pass_ids = {spadlconfig.actiontype_id["pass"], spadlconfig.actiontype_id["cross"]}
    success_id = spadlconfig.result_id["success"]
    receiver = resolve_next_touch_receiver(actions)
    on_domain = (
        actions["type_id"].isin(pass_ids).to_numpy()
        & (actions["result_id"].to_numpy() == success_id)
        & receiver.notna().to_numpy()
    )
    # Floored at 0: a backward pass does not create negative run credit.
    gain = values_at_points(xt, actions["end_x"], actions["end_y"]) - values_at_points(
        xt, actions["start_x"], actions["start_y"]
    )
    return receiver, on_domain, np.maximum(0.0, gain)


def value_off_ball_runs(
    runs: pd.DataFrame,
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    links: pd.DataFrame | None = None,
    pitch_control_cache: PitchControlCache | None = None,
    params: RunValuationParams | None = None,
) -> pd.DataFrame:
    """Attach role + threat-weighted controlled-space value to detected runs.

    Domain is completed passes and crosses whose receiver resolves via
    :func:`silly_kicks.spadl.utils.resolve_next_touch_receiver`; every other action
    is off-domain and its runs come back with NaN values and ``<NA>`` roles.

    For an on-domain action, one decomposed pitch-control surface is computed at the
    linked frame and multiplied by the acting team's threat surface. Each runner's
    ``run_value`` is the MAXIMUM of that product over the cells the runner controls
    (per-player influence at or above the resolved region floor); a runner who
    controls nothing scores an honest ``0.0``.

    Parameters
    ----------
    runs : pd.DataFrame
        Output of :func:`detect_off_ball_runs`.
    actions, frames : pd.DataFrame
    xt : ExpectedThreat
        A FITTED model. Unfitted / ``None`` / a variant string fails loud.
    links : pd.DataFrame, optional
        Pre-computed ``link_actions_to_frames`` pointers (pipeline reuse).
    pitch_control_cache : PitchControlCache, optional
    params : RunValuationParams, optional

    Returns
    -------
    pd.DataFrame
        ``runs`` plus ``role`` (``"target"`` / ``"disruptive"``), ``is_receiver``,
        ``run_value`` and ``enabled_pass_credit``. A NEW frame (ADR-033).

    Warns
    -----
    RunValueCoverageWarning
        Once per call, when some runner was absent from the linked frame's pitch
        control (a tracking-visibility gap). Those rows SURVIVE with ``run_value``
        NaN and their role still assigned -- a visibility gap is not a zero.

    Examples
    --------
    Value each detected off-ball run against a fitted expected-threat model::

        from silly_kicks.tracking import detect_off_ball_runs, value_off_ball_runs

        runs = detect_off_ball_runs(actions, frames, home_team_id=1)
        valued = value_off_ball_runs(runs, actions, frames, xt)
        valued[["player_id", "role", "run_value"]].head()
    """
    from ..xthreat import physical_grid, require_fitted_xt
    from . import _kernels
    from .pitch_control import PitchControlCache as _PitchControlCache

    require_fitted_xt(xt, caller="value_off_ball_runs")
    if params is None:
        params = RunValuationParams()

    out = runs.copy()
    out["role"] = pd.array([pd.NA] * len(out), dtype="string")
    out["is_receiver"] = pd.array([pd.NA] * len(out), dtype="boolean")
    out["run_value"] = np.nan
    out["enabled_pass_credit"] = np.nan
    if len(out) == 0 or len(actions) == 0 or len(frames) == 0:
        return out

    receiver, on_domain, enabled_credit = action_level_context(actions, xt)

    flip_rtl = acting_team_attacks_rtl(actions, frames).to_numpy(dtype=bool)
    fid_by_pos = _kernels.resolve_frame_ids_by_position(actions, frames, links=links)
    cache = pitch_control_cache if pitch_control_cache is not None else _PitchControlCache()
    floor = params.resolved_region_floor()

    # Keys carry game_id: SPADL action_id restarts per game, so an action_id-only lookup
    # writes one game's roles and values onto another game's runs (ADR-042 review finding
    # 2). detect_off_ball_runs already partitions by game; the valuation must match.
    # Group on a CANONICALIZED game key, not the raw column (ADR-019): a raw tuple lookup
    # is dtype-sensitive, so on the documented lakehouse shape (actions int64 vs frames
    # native string) EVERY get_group missed, every run came back NaN, and the coverage
    # warning misattributed it to a tracking-visibility gap. detect_off_ball_runs is
    # dtype-safe on this same column via ids_equal, so detection succeeded and only the
    # valuation failed. Same idiom as _off_ball_runs.py::_line_break_kernel.
    has_game = "game_id" in frames.columns
    if has_game:
        frames = frames.copy()
        frames["_gid_key"] = canonical_id_series(frames["game_id"])
    frame_group_cols = (["_gid_key"] if has_game else []) + ["period_id", "frame_id"]
    frame_groups = frames.groupby(frame_group_cols)

    run_keys = list(zip(out["game_id"].to_numpy(), out["action_id"].to_numpy(), strict=True))
    runs_by_action: dict = {}
    for pos, key in enumerate(run_keys):
        runs_by_action.setdefault(key, []).append(out.index[pos])
    n_unvalued = 0

    for i, (_idx, action_row) in enumerate(actions.iterrows()):
        row_idx = runs_by_action.get((action_row["game_id"], action_row["action_id"]))
        if row_idx is None or not on_domain[i]:
            continue
        row_idx = pd.Index(row_idx)
        recv = receiver.iloc[i]
        is_recv = ids_equal(out.loc[row_idx, "player_id"], pd.Series(recv, index=row_idx)).to_numpy()
        out.loc[row_idx, "is_receiver"] = pd.array(is_recv, dtype="boolean")
        out.loc[row_idx, "role"] = pd.array(np.where(is_recv, "target", "disruptive"), dtype="string")
        out.loc[row_idx, "enabled_pass_credit"] = np.where(is_recv, np.nan, enabled_credit[i])

        if np.isnan(fid_by_pos[i]):
            n_unvalued += len(row_idx)
            continue
        group_key = (int(action_row["period_id"]), int(fid_by_pos[i]))
        if has_game:
            group_key = (canonical_id(action_row["game_id"]), *group_key)
        try:
            frame = frame_groups.get_group(group_key)
        except KeyError:
            n_unvalued += len(row_idx)
            continue

        pc = cache.surface(
            frame,
            action_row["team_id"],
            method=params.pitch_control_method,  # type: ignore[arg-type]
            decompose=True,
        )
        # Pitch control lives in FRAME coordinates; the threat grid is built in action-LTR,
        # so it is POINT-reflected (both axes, ADR-028) for an RTL-attacking acting team.
        threat = physical_grid(xt, pc.grid_x, pc.grid_y)
        if flip_rtl[i]:
            threat = threat[::-1, ::-1]
        weighted = np.asarray(pc.surface) * threat

        for ridx in row_idx:
            pidx = _safe_index_of(pc.player_ids, out.at[ridx, "player_id"])
            if pidx is None or pc.per_player_influence is None:
                n_unvalued += 1
                continue
            region = pc.per_player_influence[pidx] >= floor
            out.at[ridx, "run_value"] = float(weighted[region].max()) if region.any() else 0.0

    if n_unvalued:
        warnings.warn(
            f"value_off_ball_runs: {n_unvalued} detected run(s) could not be valued -- the "
            "runner was absent from the linked frame's pitch control (a tracking-visibility "
            "gap). Those rows keep run_value=NaN rather than a fabricated 0.",
            RunValueCoverageWarning,
            stacklevel=2,
        )
    return out
