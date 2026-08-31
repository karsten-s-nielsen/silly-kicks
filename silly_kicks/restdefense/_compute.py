"""compute_rest_defense / summarize_rest_defense (TF-60, ADR-080).

The orchestrator: build the ``GoalMap`` once (ADR-055), select the action-grid samples (``_windows``),
compute ``compute_defensive_line`` / ``compute_team_shape`` ONCE per match (only for teams whose end
resolves -- ``compute_defensive_line`` raises on an unresolved end, so those teams' samples are
emitted as honest-NaN ``rd_geometry_source="unresolved"`` rows instead), then score each sample once
via a single ``group_rows`` pass (ADR-068; scale-guarded, ADR-073). Pure: caller inputs are never
mutated. Nullable dtypes (counts ``Int64``, metrics ``float64``); NA on unscoreable rows, never a 0.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from silly_kicks._frame_index import RowGroups, group_rows
from silly_kicks.id_compat import align_join_keys, canonical_id, canonical_id_series
from silly_kicks.tracking import (
    GoalEndUnresolvedError,
    compute_defensive_line,
    compute_team_shape,
    resolve_defended_goals,
)

from ._columns import (
    RD_FRAME_KEYS,
    RD_GEOMETRY_SOURCE,
    RD_METRIC_COLUMNS,
    RD_NUM_SUPERIORITY,
    RD_NUM_SUPERIORITY_GK,
    RD_SAMPLE_KEYS,
    RD_SHAPE_STAGGER,
    RD_ZONE_OCCUPANCY,
)
from ._config import RestDefenseParams
from ._danger import layer2_metrics
from ._fov import append_fov_companions
from ._report import RestDefenseReport
from ._structure import SampleContext, layer1_metrics
from ._windows import _GOAL_UNRESOLVED, select_rest_defense_samples

_COUNT_COLS = [RD_NUM_SUPERIORITY, RD_NUM_SUPERIORITY_GK, RD_ZONE_OCCUPANCY]
_FLOAT_METRIC_COLS = [c for c in RD_METRIC_COLUMNS if c not in _COUNT_COLS and c != RD_SHAPE_STAGGER]
_SAMPLE_META = ["possession_id", "is_possession_loss"]
_OUTPUT_COLS = [*RD_SAMPLE_KEYS, *_SAMPLE_META, *RD_METRIC_COLUMNS, RD_GEOMETRY_SOURCE]
# The engine columns (compute_defensive_line / compute_team_shape names) threaded into each
# SampleContext -- NOT the RD_* output names (e.g. rd_compactness_x IS compute_defensive_line's
# compactness_x, but the emitted column is renamed).
_DL_COLS = ["defensive_line_x", "compactness_x", "lateral_width"]
_TS_COLS = ["team_length"]

#: Shared frozen default (immutable, so one singleton is safe; avoids a B008 call-in-default).
_DEFAULT_PARAMS = RestDefenseParams()


def _resolved_team_keys(frames: pd.DataFrame, goal_map) -> set:
    """Canonical ``(game, period, team)`` keys whose defended end resolves (else NaN geometry)."""
    teams = frames.loc[frames["team_id"].notna(), ["game_id", "period_id", "team_id"]].drop_duplicates()
    resolved = set()
    for g, p, t in teams.itertuples(index=False):
        if goal_map.get(g, p, t, allow_guess=True) is not None:
            resolved.add((canonical_id(g), canonical_id(p), canonical_id(t)))
    return resolved


def _engine_tables(frames: pd.DataFrame, goal_map, params: RestDefenseParams):
    """``(defensive_line, team_shape)`` for the resolvable teams only (ADR-055 build-once)."""
    resolved = _resolved_team_keys(frames, goal_map)
    if not resolved:
        return None, None
    gk = canonical_id_series(frames["game_id"])
    pk = canonical_id_series(frames["period_id"])
    tk = canonical_id_series(frames["team_id"])
    keymask = np.array([(g, p, t) in resolved for g, p, t in zip(gk, pk, tk, strict=True)])
    rframes = frames[frames["is_ball"].to_numpy(dtype=bool) | keymask]
    dl = compute_defensive_line(rframes, goal_map=goal_map, n=params.n_rearguard)
    team_ids = rframes.loc[rframes["team_id"].notna(), "team_id"].dropna().unique()
    ts = pd.concat([compute_team_shape(rframes, team_id=t) for t in team_ids], ignore_index=True)
    return dl, ts


def _geometry_source_by_key(keep: pd.DataFrame, goal_map) -> dict:
    """``{(canon game, canon period, canon team): "resolved"|"guessed"|"unresolved"}`` (IMPL-02).

    "resolved" = a GoalMap end from clear GK evidence (``allow_guess=False``); "guessed" = only an
    ``allow_guess=True`` fallback resolves it (an inference that matters on FOV-cropped SB360);
    "unresolved" = no end at all. Only the ``(game, period, team)`` triples present in ``keep`` are
    resolved (a small set)."""
    out: dict = {}
    keys = keep[["game_id", "period_id", "team_id"]].drop_duplicates()
    for g, p, t in keys.itertuples(index=False):
        if goal_map.get(g, p, t, allow_guess=False) is not None:
            src = "resolved"
        elif goal_map.get(g, p, t, allow_guess=True) is not None:
            src = "guessed"
        else:
            src = "unresolved"
        out[(canonical_id(g), canonical_id(p), canonical_id(t))] = src
    return out


def _opponent_map(frames: pd.DataFrame) -> dict:
    """``{(canonical game, canonical team): opponent_id}`` for two-team games (else absent)."""
    out: dict = {}
    by_game = frames.loc[frames["team_id"].notna()].groupby("game_id")["team_id"].unique()
    for game, teamlist in by_game.items():
        teams = list(teamlist)
        if len(teams) == 2:
            out[(canonical_id(game), canonical_id(teams[0]))] = teams[1]
            out[(canonical_id(game), canonical_id(teams[1]))] = teams[0]
    return out


def _merge_engine_cols(keep: pd.DataFrame, dl, ts) -> pd.DataFrame:
    merge_keys = [*RD_FRAME_KEYS, "team_id"]
    out = keep
    for table, cols in ((dl, _DL_COLS), (ts, _TS_COLS)):
        if table is None:
            for c in cols:
                out[c] = np.nan
            continue
        sel = table[[*merge_keys, *cols]]
        left, right = align_join_keys(out, sel, merge_keys)
        out = left.merge(right, on=merge_keys, how="left")
    return out


def _score_samples(
    keep: pd.DataFrame,
    frames: pd.DataFrame,
    opp_map: dict,
    params: RestDefenseParams,
    *,
    groups: RowGroups | None = None,
    geom_src: dict | None = None,
    xt=None,
    goal_map=None,
    pitch_control_cache=None,
) -> pd.DataFrame:
    """Per-sample Layer-1 + Layer-2 metrics (one ``group_rows`` pass; ADR-068/073). NaN row for
    unresolved geometry. Layer 2 reuses the SAME per-sample ``frame_rows`` -- no new looping pass.

    ``geom_src`` (from :func:`_geometry_source_by_key`) labels each scored row "resolved" vs "guessed"
    (IMPL-02); absent -> "resolved" (the historical default; used by the scale guard). ``xt`` /
    ``goal_map`` / ``pitch_control_cache`` feed Layer 2 (all NaN when ``xt`` is None; P2-02)."""
    if groups is None:
        groups = group_rows(frames, tuple(RD_FRAME_KEYS))
    gsrc = geom_src or {}
    rows: list[dict] = []
    for row in keep.itertuples(index=False):
        key = (canonical_id(row.game_id), canonical_id(row.period_id), canonical_id(row.team_id))
        m: dict[str, object]
        if pd.isna(row.gate_drop_reason):  # scored (own goal resolved-or-guessed + committed forward)
            frame_rows = groups.get(row.game_id, row.period_id, row.frame_id)
            ctx = SampleContext(
                team_id=row.team_id,
                opponent_id=opp_map.get((canonical_id(row.game_id), canonical_id(row.team_id)), pd.NA),
                ball_x=_f(row.ball_x),
                own_goal_x=_f(row.own_goal_x),
                attacked_goal_x=_f(row.attacked_goal_x),
                defensive_line_x=_f(row.defensive_line_x),
                compactness_x=_f(row.compactness_x),
                lateral_width=_f(row.lateral_width),
                team_length=_f(row.team_length),
            )
            m = layer1_metrics(frame_rows, ctx, params=params)
            m.update(
                layer2_metrics(
                    frame_rows, ctx, xt=xt, goal_map=goal_map, params=params, pitch_control_cache=pitch_control_cache
                )
            )
            m[RD_GEOMETRY_SOURCE] = gsrc.get(key, "resolved")  # "resolved" or "guessed" (IMPL-02)
        else:  # goal_end_unresolved -> honest-NaN row (ADR-055)
            m = {c: pd.NA for c in RD_METRIC_COLUMNS}
            m[RD_GEOMETRY_SOURCE] = "unresolved"
        rows.append(m)
    return pd.DataFrame(rows, index=keep.index)


def _f(v) -> float:
    """Scalar -> float, mapping NA to NaN (for the SampleContext float fields)."""
    return float("nan") if pd.isna(v) else float(v)


def compute_rest_defense(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    xt=None,
    goal_map=None,
    links: pd.DataFrame | None = None,
    pitch_control_cache=None,
    visible_area: pd.DataFrame | None = None,
    params: RestDefenseParams = _DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, RestDefenseReport]:
    """Per-sample Layer-1 rest-defense structure metrics + a conserving report (spec §5.3 / §14).

    The samples table has one row per in-possession on-ball action that passed the committed-forward
    gate, with ``rd_geometry_source`` in ``{"resolved", "guessed"}`` (a clear-GK end vs a GoalMap
    ``allow_guess`` inference; IMPL-02), plus any that passed but whose goal end did not resolve at all
    (``"unresolved"``, honest-NaN metrics). Every other action is dropped-and-counted in the report.

    When ``visible_area`` (an ``action_id`` -> polygon table) is supplied, each count/region column
    gains an opt-in ``<col>_observed_fraction`` / ``_observed_source`` FOV companion (ADR-077); the
    primary columns are BYTE-IDENTICAL with and without it (additive -- no VAEP change, ADR-062).

    Examples
    --------
    Given a match's SPADL ``actions`` and LTR-normalised tracking ``frames`` (native GK), score the
    Layer-1 rest-defense structure at each in-possession on-ball action and check drop-conservation::

        samples, report = compute_rest_defense(actions, frames)
        samples[["team_id", "action_id", "rd_num_superiority", "rd_gk_to_line_distance"]].head()
        assert report.n_frames_scored + sum(report.drop_reasons.values()) == report.n_frames_in
    """
    if goal_map is None:
        goal_map = resolve_defended_goals(frames)
    windows = select_rest_defense_samples(actions, frames, goal_map=goal_map, params=params, links=links)

    reason = windows["gate_drop_reason"]
    scored_mask = reason.isna()
    keep = windows[scored_mask | reason.eq(_GOAL_UNRESOLVED)].reset_index(drop=True)

    # ADR-055 build-once. _engine_tables pre-filters to resolvable teams because
    # compute_defensive_line raises GoalEndUnresolvedError per-CALL (all teams), not per-sample, so a
    # naive edge-catch would lose EVERY team's line when one is unresolved -- the pre-filter degrades
    # per-team instead. The try/except is the spec-§15 safety net for the residual case (a team the
    # pre-filter admitted as resolvable that compute_defensive_line still refuses): degrade to
    # all-unresolved NaN rows rather than crash (IMPL-03).
    try:
        dl, ts = _engine_tables(frames, goal_map, params)
    except GoalEndUnresolvedError:
        dl, ts = None, None
    keep = _merge_engine_cols(keep, dl, ts)
    geom_src = _geometry_source_by_key(keep, goal_map)
    metrics = _score_samples(
        keep,
        frames,
        _opponent_map(frames),
        params,
        geom_src=geom_src,
        xt=xt,
        goal_map=goal_map,
        pitch_control_cache=pitch_control_cache,
    ).reset_index(drop=True)

    base = keep[[*RD_SAMPLE_KEYS, *_SAMPLE_META]].reset_index(drop=True)
    samples = pd.concat([base, metrics], axis=1)
    for c in _COUNT_COLS:
        samples[c] = samples[c].astype("Int64")
    for c in _FLOAT_METRIC_COLS:
        samples[c] = pd.to_numeric(samples[c], errors="coerce").astype("float64")
    samples[RD_SHAPE_STAGGER] = samples[RD_SHAPE_STAGGER].astype("object")
    samples["is_possession_loss"] = samples["is_possession_loss"].astype(bool)
    samples = samples[_OUTPUT_COLS]

    if visible_area is not None:
        samples = append_fov_companions(samples, keep, visible_area=visible_area, params=params)

    report = RestDefenseReport(
        params=params,
        n_frames_in=len(windows),
        n_frames_scored=int(scored_mask.sum()),
        drop_reasons={str(k): int(v) for k, v in reason.dropna().value_counts().items()},
    )
    return samples, report


def summarize_rest_defense(samples: pd.DataFrame, *, by: Literal["possession", "match"] = "possession") -> pd.DataFrame:
    """Pure per-group reduction of the samples table -- the coach-facing rollup (spec §5.3).

    Every numeric Layer-1 metric is MEANED over the group's RESOLVED samples (matching how the source
    KPIs are reported -- e.g. Forcher's mean numerical superiority; a summed count would scale with
    sample count and be uninterpretable). ``n_samples`` counts the resolved samples; a group with no
    resolved sample is honest-NaN (``min_count=1``), never a fabricated 0. The moment-of-loss snapshot
    is not duplicated here -- it is a row in the samples table (``is_possession_loss``), which keeps the
    rollup a pure reduction the consumer can re-aggregate with its own policy.

    Examples
    --------
    Reduce the per-sample table to the coach-facing per-``(team, game)`` rollup (one row per group)::

        samples, _report = compute_rest_defense(actions, frames)
        per_match = summarize_rest_defense(samples, by="match")
        per_match[["team_id", "rd_num_superiority", "n_samples"]]
    """
    if by == "possession":
        keys = ["game_id", "period_id", "team_id", "possession_id"]
    elif by == "match":
        keys = ["game_id", "team_id"]
    else:  # pragma: no cover -- Literal guards the public surface
        raise ValueError(f"by must be 'possession' or 'match', got {by!r}")

    resolved = samples[samples[RD_GEOMETRY_SOURCE] == "resolved"]
    numeric = [*_COUNT_COLS, *_FLOAT_METRIC_COLS]
    grouped = resolved.groupby(keys, dropna=False, sort=False)
    agg = grouped[numeric].mean(numeric_only=True)
    agg["n_samples"] = grouped.size()
    agg["n_losses"] = grouped["is_possession_loss"].sum(min_count=1)
    return agg.reset_index()
