"""Gradient Sports (formerly PFF FC) tracking DataFrame converter.

Mirrors silly_kicks.tracking.sportec but for Gradient Sports-shaped input.
Reuses the shared ``direction`` helper extracted from
silly_kicks.spadl.gradientsports (PR-S18) for
``home_team_start_left[_extratime]`` direction normalization.

Input contract (EXPECTED_INPUT_COLUMNS):
  Same shape as sportec, except ``player_id`` / ``team_id`` are nullable
  Int64 (Gradient Sports integer identifiers) and ``game_id`` is int64.

Coordinate transformation: ``x = x_centered + 52.5``;
``y = y_centered + 34.0``. Per-period direction flip via the shared
``home_attacks_right_per_period`` helper.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Literal

import pandas as pd

from . import direction
from ._id_compat import ids_match
from .schema import GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS, TrackingConversionReport
from .sportec import _resolve_output_convention

if TYPE_CHECKING:
    from .preprocess import PreprocessConfig

_PROVIDER_NAME = "gradientsports"

EXPECTED_INPUT_COLUMNS: frozenset[str] = frozenset(
    {
        "game_id",
        "period_id",
        "frame_id",
        "time_seconds",
        "frame_rate",
        "player_id",
        "team_id",
        "is_ball",
        "is_goalkeeper",
        "x_centered",
        "y_centered",
        "z",
        "speed_native",
        "ball_state",
    }
)


def convert_to_frames(
    raw_frames: pd.DataFrame,
    home_team_id: int,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
    preserve_native: list[str] | None = None,
    *,
    output_convention: Literal["absolute_frame", "ltr"] | None = None,
    preprocess: PreprocessConfig | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    """Convert Gradient Sports-shaped raw tracking frames to canonical schema.

    Parameters
    ----------
    raw_frames : pd.DataFrame
        Gradient Sports input (see EXPECTED_INPUT_COLUMNS).
        ``raw_frames["time_seconds"]`` must be **period-relative** (seconds since
        the start of each period, resetting to 0 -- NOT absolute match-clock).
        This is silly_kicks' canonical convention (matches the events converters)
        and what :func:`silly_kicks.tracking.utils.link_actions_to_frames`
        requires. See ADR-017.
    home_team_id : int
        homeTeam.id from the metadata JSON.
    home_team_start_left : bool
        From metadata ``homeTeamStartLeft``.
    home_team_start_left_extratime : bool | None
        From metadata ``homeTeamStartLeftExtraTime``; required when
        periods 3/4 are present.
    output_convention : {"absolute_frame", "ltr"} | None, default None
        Coordinate convention of the returned frames. ``"absolute_frame"`` (the
        historical default behaviour) emits frames in absolute-frame-home-right
        convention with per-row ``team_attacking_direction``. ``"ltr"`` applies
        :func:`silly_kicks.tracking.utils.play_left_to_right` internally so the
        output is in canonical SPADL "all teams attack left-to-right"
        convention. Passing ``None`` (the legacy unspecified state) emits a
        ``DeprecationWarning`` and defaults to ``"absolute_frame"`` -- callers
        should pick one explicitly. See ADR-006 (silly-kicks 3.0.0).

    Returns
    -------
    frames : pd.DataFrame
        GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS-shaped output, 105x68 m SPADL coordinates,
        in the convention requested by ``output_convention``.
    report : TrackingConversionReport

    Examples
    --------
    Read Gradient Sports tracking JSONL.bz2, flatten to frames, then convert::

        import bz2, json, pandas as pd
        from silly_kicks.tracking.gradientsports import convert_to_frames
        with bz2.open("10501.jsonl.bz2", "rt") as fh:
            rows = [json.loads(line) for line in fh]
        raw = pd.json_normalize(rows)  # caller-shaped flattening
        frames, report = convert_to_frames(
            raw, home_team_id=366, home_team_start_left=True,
            output_convention="absolute_frame",
        )
    """
    _ = preserve_native  # reserved for future PR
    output_convention = _resolve_output_convention(output_convention, _adapter_name="gradientsports")
    missing = EXPECTED_INPUT_COLUMNS - set(raw_frames.columns)
    if missing:
        raise ValueError(f"gradientsports convert_to_frames missing columns: {sorted(missing)}")

    direction.require_et_direction(
        raw_frames["period_id"], home_team_start_left_extratime, source="gradientsports convert_to_frames"
    )

    out = raw_frames.copy()
    out["x"] = out["x_centered"] + 52.5
    out["y"] = out["y_centered"] + 34.0

    home_attacks_right = direction.home_attacks_right_per_period(
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
    )
    home_rtl_periods = {p for p, attacks_right in home_attacks_right.items() if not attacks_right}
    flip_mask = out["period_id"].isin(home_rtl_periods).to_numpy()
    out.loc[flip_mask, "x"] = 105.0 - out.loc[flip_mask, "x"]
    out.loc[flip_mask, "y"] = 68.0 - out.loc[flip_mask, "y"]

    out["team_attacking_direction"] = None
    is_player = (~out["is_ball"].astype(bool)).to_numpy(dtype=bool)
    # ADR-019: dtype-safe is_home. A raw `==` silently matched zero players when home_team_id was
    # int and the frame team_id was object-string -> every player mislabeled "rtl" -> downstream
    # play_left_to_right double-flip -> mis-oriented frames (2026-06-09 fix).
    is_home = ids_match(out["team_id"], home_team_id).fillna(False).to_numpy(dtype=bool)
    if is_player.any() and not (is_player & is_home).any():
        warnings.warn(
            f"gradientsports.convert_to_frames: home_team_id={home_team_id!r} matched ZERO player "
            "rows (id dtype vs frame team_id mismatch?) -- frame orientation would be wrong.",
            stacklevel=2,
        )
    is_known_period = out["period_id"].isin([1, 2, 3, 4]).to_numpy(dtype=bool)
    out.loc[is_player & is_home & is_known_period, "team_attacking_direction"] = "ltr"
    out.loc[is_player & ~is_home & is_known_period, "team_attacking_direction"] = "rtl"

    out["speed"] = out["speed_native"].astype("float64")
    speed_source: list[object] = ["native" if pd.notna(v) else None for v in out["speed"]]
    out["speed_source"] = pd.Series(speed_source, index=out.index, dtype="object")
    out["confidence"] = None
    out["visibility"] = None
    out["source_provider"] = _PROVIDER_NAME
    out["is_goalkeeper_source"] = "native"

    final = pd.DataFrame({col: out[col] for col in GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS})
    for col, dtype_str in GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS.items():
        if dtype_str == "bool":
            final[col] = final[col].astype("bool")
        elif dtype_str == "Int64":
            final[col] = final[col].astype("Int64")
        elif dtype_str in {"int64", "float64"}:
            final[col] = pd.to_numeric(final[col], errors="coerce").astype(dtype_str)  # type: ignore[arg-type]
        elif dtype_str == "object":
            final[col] = final[col].astype(object)

    n_input_frames = int(raw_frames["frame_id"].nunique())
    n_periods = int(raw_frames["period_id"].nunique())
    cov: dict[int, float] = {}
    ball_out: dict[int, float] = {}
    for p, g in final.groupby("period_id", sort=True):
        expected = int(g["frame_id"].max() - g["frame_id"].min() + 1)
        actual = int(g["frame_id"].nunique())
        cov[int(p)] = float(actual) / max(expected, 1)  # type: ignore[arg-type]
        ball_g = g[g["is_ball"]]
        if len(ball_g):
            dt = 1.0 / float(ball_g["frame_rate"].iloc[0])
            ball_out[int(p)] = float((ball_g["ball_state"] == "dead").sum() * dt)  # type: ignore[arg-type]

    nan_rate = {col: float(final[col].isna().mean()) for col in final.columns}

    report = TrackingConversionReport(
        provider="gradientsports",
        total_input_frames=n_input_frames,
        total_output_rows=len(final),
        n_periods=n_periods,
        frame_coverage_per_period=cov,
        ball_out_seconds_per_period=ball_out,
        nan_rate_per_column=nan_rate,
        derived_speed_rows=int((final["speed_source"] == "derived").sum()),
        unrecognized_player_ids=set(),
        n_teams_gk_derived=0,
        derived_gk_picks={},
    )

    if output_convention == "ltr":
        from .utils import play_left_to_right

        final = play_left_to_right(final, home_team_id)

    if preprocess is not None:
        from .preprocess import derive_velocities, interpolate_frames, smooth_frames
        from .preprocess._resolve import resolve_preprocess

        cfg = resolve_preprocess(preprocess, provider=_PROVIDER_NAME)
        if cfg.interpolation_method is not None:
            final = interpolate_frames(final, config=cfg)
        if cfg.smoothing_method is not None:
            final = smooth_frames(final, config=cfg)
        if cfg.derive_velocity:
            final = derive_velocities(final, config=cfg)

    return final, report


# ---------------------------------------------------------------------------
# Jersey -> roster player-id resolution (TF-24 PR-A)
# ---------------------------------------------------------------------------

_FRAMES_REQUIRED: frozenset[str] = frozenset({"team_side", "jersey_number", "is_ball"})
_ROSTER_REQUIRED: frozenset[str] = frozenset({"team_id", "shirt_number", "player_id"})


@dataclasses.dataclass(frozen=True)
class GradientsportsRosterReport:
    """Audit of a :func:`add_gradientsports_player_ids` resolution.

    Attributes
    ----------
    n_player_rows : int
        Non-ball rows seen.
    n_matched : int
        Player rows whose ``(team_id, jersey_number)`` matched a roster entry.
    n_unmatched : int
        ``n_player_rows - n_matched``.
    unmatched_jerseys : frozenset[tuple[int, str]]
        Distinct ``(team_id, jersey_number)`` keys that did not match.
    roster_size : int
        Roster rows used for the join (after de-duplication).
    n_duplicate_roster_keys : int
        Duplicate ``(team_id, shirt_number)`` roster keys dropped (``keep="first"``).

    Examples
    --------
    >>> _, report = add_gradientsports_player_ids(frames, roster, home_team_id=366, away_team_id=51)
    >>> report.n_matched
    4
    """

    n_player_rows: int
    n_matched: int
    n_unmatched: int
    unmatched_jerseys: frozenset[tuple[int, str]]
    roster_size: int
    n_duplicate_roster_keys: int


def add_gradientsports_player_ids(
    jersey_frames: pd.DataFrame,
    roster: pd.DataFrame,
    *,
    home_team_id: int,
    away_team_id: int,
) -> tuple[pd.DataFrame, GradientsportsRosterReport]:
    """Resolve GS tracking jersey numbers to the events SPADL ``player_id`` space.

    Gradient Sports tracking frames carry only ``jerseyNum`` (+ a home/away split);
    GS events SPADL ``player_id`` is the integer roster ``player.id``. This helper
    joins ``(team_id, jersey_number)`` -> roster ``player_id`` so a tracking carrier
    is joinable to events. Run it BEFORE
    :func:`silly_kicks.tracking.gradientsports.convert_to_frames`.

    Parameters
    ----------
    jersey_frames : pd.DataFrame
        Long-form GS tracking rows. Required columns: ``team_side`` ("home"/"away";
        ``None`` for ball), ``jersey_number`` (object/string; ``None`` for ball),
        ``is_ball`` (bool). Other tracking columns are passed through untouched.
    roster : pd.DataFrame
        Required columns: ``team_id`` (coercible to int), ``shirt_number``
        (object/string), ``player_id`` (int). Optional ``position_group_type``
        (literal ``"GK"`` flags the goalkeeper).
    home_team_id, away_team_id : int
        The events SPADL ``int64`` team ids. ``team_side`` maps to these.

    Returns
    -------
    frames : pd.DataFrame
        Copy of ``jersey_frames`` with ``player_id`` (``Int64``; ``pd.NA`` for
        ball/unmatched), ``team_id`` (``Int64``), ``is_goalkeeper`` (bool) added.
    report : GradientsportsRosterReport

    Examples
    --------
    >>> frames, report = add_gradientsports_player_ids(
    ...     jersey_frames, roster, home_team_id=366, away_team_id=51
    ... )
    >>> report.n_matched >= 0
    True
    """
    miss_f = _FRAMES_REQUIRED - set(jersey_frames.columns)
    if miss_f:
        raise ValueError(f"add_gradientsports_player_ids: jersey_frames missing columns: {sorted(miss_f)}")
    miss_r = _ROSTER_REQUIRED - set(roster.columns)
    if miss_r:
        raise ValueError(f"add_gradientsports_player_ids: roster missing columns: {sorted(miss_r)}")

    out = jersey_frames.copy()
    is_ball = out["is_ball"].astype(bool)
    is_player = ~is_ball

    # team_side -> team_id (Int64; ball / unknown side -> NA)
    side = out["team_side"].astype("string")
    team_id = pd.Series(pd.NA, index=out.index, dtype="Int64")
    team_id = team_id.mask(is_player & (side == "home"), home_team_id)
    team_id = team_id.mask(is_player & (side == "away"), away_team_id)
    out["team_id"] = team_id

    # roster lookup as a "team|shirt" -> value dict.
    has_pos = "position_group_type" in roster.columns
    if not has_pos:
        warnings.warn(
            "gradientsports roster has no 'position_group_type' column; is_goalkeeper will be all-False",
            UserWarning,
            stacklevel=2,
        )
    r = roster.copy()
    r["_team"] = pd.to_numeric(r["team_id"], errors="coerce").astype("Int64")
    r["_shirt"] = r["shirt_number"].astype("string").str.strip()
    r["_pid"] = pd.to_numeric(r["player_id"], errors="coerce").astype("Int64")
    r["_is_gk"] = (r["position_group_type"].astype("string") == "GK") if has_pos else False
    # N1: enforce roster (team, shirt) uniqueness BEFORE building the dicts. A duplicate key
    # would let dict(zip(...)) keep the LAST entry (wrong) -- dedupe keep="first" so the first
    # roster row wins, warn, and record the count. (.map can't explode rows; this guards value
    # correctness + surfaces the anomaly.)
    _dup = r.duplicated(subset=["_team", "_shirt"], keep="first")
    n_duplicate_roster_keys = int(_dup.sum())
    if n_duplicate_roster_keys:
        warnings.warn(
            f"gradientsports roster has {n_duplicate_roster_keys} duplicate "
            "(team_id, shirt_number) key(s); keeping first",
            UserWarning,
            stacklevel=2,
        )
        r = r[~_dup]
    r["_key"] = r["_team"].astype("string").str.cat(r["_shirt"], sep="|")
    # NOTE: dict(zip(...)) keeps the LAST value on a duplicate key; Task 3 dedupes r with
    # keep="first" BEFORE this so the first roster entry wins (and rows can't explode).
    pid_map = dict(zip(r["_key"].to_list(), r["_pid"].to_list(), strict=False))
    gk_map = dict(zip(r["_key"].to_list(), r["_is_gk"].to_list(), strict=False))

    # ORDER-SAFE resolution (C2): elementwise .map on a same-index key Series -- positionally
    # exact by construction (no merge / no index reassignment / no reorder risk). A frame row
    # with NA team or NA jersey (ball rows) yields an NA key -> map miss -> pd.NA.
    frame_key = out["team_id"].astype("string").str.cat(out["jersey_number"].astype("string").str.strip(), sep="|")
    out["player_id"] = frame_key.map(pid_map).astype("Int64")
    # `== True` maps True->True, False->False, NaN(miss)->False in one bool Series (no fillna
    # object-downcast). E712 noqa matches the codebase idiom (e.g. tracking/_das.py).
    out["is_goalkeeper"] = (frame_key.map(gk_map) == True).astype("bool")  # noqa: E712

    matched = is_player & out["player_id"].notna()
    n_player = int(is_player.sum())
    n_matched = int(matched.sum())
    unmatched_mask = is_player & out["player_id"].isna() & out["team_id"].notna() & out["jersey_number"].notna()
    unmatched = {
        (int(t), str(j))
        for t, j in zip(out.loc[unmatched_mask, "team_id"], out.loc[unmatched_mask, "jersey_number"], strict=False)
    }

    # N2: vocabulary-drift guard -- if position groups are present + players matched but ZERO
    # are flagged GK, the "GK" literal likely drifted; announce it instead of a silent GK-less
    # match (which would degrade defending_gk / gk_influence for GS).
    if has_pos and n_player and not bool(out["is_goalkeeper"].any()):
        observed = sorted({str(v) for v in roster["position_group_type"].dropna().unique()})
        warnings.warn(
            f"gradientsports: no GK found (positionGroupType values were {observed}); expected literal 'GK'",
            UserWarning,
            stacklevel=2,
        )

    # M2: a >=50% unmatched (or zero-match) rate is the precise signature of a wrong team-id
    # space / roster mismatch (the silent bug being fixed). Warn loudly; never raise (ADR-003).
    if n_player and (n_matched == 0 or (n_player - n_matched) / n_player >= 0.5):
        warnings.warn(
            f"gradientsports player-id resolution matched {n_matched}/{n_player} player rows "
            "(>=50% unmatched); check team-id space / roster alignment",
            UserWarning,
            stacklevel=2,
        )

    return out, GradientsportsRosterReport(
        n_player_rows=n_player,
        n_matched=n_matched,
        n_unmatched=n_player - n_matched,
        unmatched_jerseys=frozenset(unmatched),
        roster_size=len(r),
        n_duplicate_roster_keys=n_duplicate_roster_keys,
    )
