"""SkillCorner SPADL converter.

Converts SkillCorner ``dynamic_events.csv`` DataFrames to SPADL actions.

SkillCorner events are possession-centric: the primary event type is
``player_possession`` (one row per possession phase). Defensive actions
(interceptions, tackles) are derived from the ``start_type`` column and
cross-referenced with ``on_ball_engagement`` rows. Keeper saves are
inferred from shot-to-GK possession sequences.

Coordinate system: attacking-direction-normalized centered meters
(origin at center spot, positive x toward the goal being attacked).
This is ``POSSESSION_PERSPECTIVE`` -- the same as StatsBomb/Wyscout.
Rescaled to SPADL 0-105 x 0-68 frame using pitch dimensions from
``match_metadata``.

See spec: ``docs/superpowers/specs/2026-05-14-skillcorner-events-converter-design.md``
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from . import config as spadlconfig
from ._skillcorner_inference import infer_defensive_actions, infer_keeper_saves
from .base import _add_dribbles, _derive_end_coordinates
from .orientation import POSSESSION_PERSPECTIVE, to_spadl_ltr
from .schema import SKILLCORNER_SPADL_COLUMNS, ConversionReport
from .utils import _finalize_output


def _transform_coords(
    x: pd.Series,
    y: pd.Series,
    pitch_length: int | float,
    pitch_width: int | float,
) -> tuple[pd.Series, pd.Series]:
    """Rescale centered meters to SPADL 0-based frame.

    Parameters
    ----------
    x, y : pd.Series
        Coordinates in centered meters (origin at center spot).
    pitch_length, pitch_width : int or float
        Actual pitch dimensions from match_metadata.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        ``(x_spadl, y_spadl)`` in SPADL [0, 105] x [0, 68] frame.
    """
    half_length = pitch_length / 2
    half_width = pitch_width / 2
    x_out = (x / half_length) * 52.5 + 52.5
    y_out = (y / half_width) * 34.0 + 34.0
    # Clamp to SPADL pitch boundaries (raw data can slightly exceed pitch dims)
    x_out = x_out.clip(lower=0.0, upper=105.0)
    y_out = y_out.clip(lower=0.0, upper=68.0)
    return x_out, y_out


# SkillCorner's `time_start` is the CONTINUOUS broadcast clock (the 2nd half shows 45:00+,
# extra time 90:00+/105:00+), so each period's nominal start offset must be subtracted to get
# the period-relative seconds the tracking-frame contract requires (ADR-017). The offsets are
# the regulation period boundaries (45/90/105/120 min), matching the Gradient Sports adapter.
_PERIOD_START_SECONDS = {1: 0.0, 2: 45 * 60.0, 3: 90 * 60.0, 4: 105 * 60.0, 5: 120 * 60.0}


def _parse_time_start(time_start: pd.Series) -> pd.Series:
    """Parse ``MM:SS.d`` time strings to float seconds (CONTINUOUS match clock).

    NOTE: SkillCorner's ``time_start`` is the continuous broadcast clock (2nd-half values are
    45:00+), so this returns seconds since MATCH start, NOT since period start. Callers must
    re-base to period-relative via :func:`_to_period_relative` to satisfy the period-relative
    frame contract (ADR-017).

    Parameters
    ----------
    time_start : pd.Series
        String series in ``"MM:SS.d"`` format (e.g. ``"12:34.5"``).

    Returns
    -------
    pd.Series
        Float64 seconds since match start (continuous broadcast clock).
    """
    parts = time_start.str.split(":", expand=True)
    minutes = parts[0].astype("float64")
    seconds = parts[1].astype("float64")
    return minutes * 60 + seconds


def _to_period_relative(time_seconds: pd.Series, period: pd.Series) -> pd.Series:
    """Re-base the continuous match clock to PERIOD-RELATIVE seconds (ADR-017).

    Subtracts each period's nominal start offset (``_PERIOD_START_SECONDS``) so the result aligns
    with the period-relative tracking frames the linker joins against. Unknown period values fall
    back to a 0 offset (no re-base). Fixes the SkillCorner 2nd-half action↔frame linkage failure
    (BUG 1, 2026-06-09)."""
    offset = period.map(_PERIOD_START_SECONDS).fillna(0.0).to_numpy()
    return pd.Series(time_seconds.to_numpy(dtype="float64") - offset, index=time_seconds.index)


def _dispatch_bodypart(
    is_header: pd.Series,
    hand_pass: pd.Series,
) -> np.ndarray:
    """Map SkillCorner body part booleans to SPADL bodypart_id.

    Priority: is_header > hand_pass > default foot.
    """
    return np.select(
        [is_header.eq("True") | is_header.eq(True), hand_pass.eq("True") | hand_pass.eq(True)],
        [spadlconfig.bodypart_id["head"], spadlconfig.bodypart_id["other"]],
        default=spadlconfig.bodypart_id["foot"],
    )


def _is_cross(
    third: pd.Series,
    channel: pd.Series,
    start_x_spadl: pd.Series,
    start_y_spadl: pd.Series,
) -> pd.Series:
    """Detect crosses using native SC columns with spatial fallback.

    A pass is a cross when it originates in the attacking third from a
    wide channel. Uses ``player_targeted_third_pass`` /
    ``player_targeted_channel_pass`` when available (~98%), falling back
    to a coordinate heuristic for NaN rows.
    """
    has_native = third.notna() & channel.notna()
    native_cross = (third == "attacking_third") & channel.isin({"wide_left", "wide_right"})
    spatial_cross = (start_x_spadl > 70.0) & ((start_y_spadl < 15.0) | (start_y_spadl > 53.0))
    return (has_native & native_cross) | (~has_native & spatial_cross)


def convert_to_actions(
    events: pd.DataFrame,
    match_metadata: dict,
    *,
    preserve_native: bool = False,
) -> tuple[pd.DataFrame, ConversionReport]:
    """Convert SkillCorner dynamic_events.csv to SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        Full ``dynamic_events.csv`` as a DataFrame (all 294 columns;
        the converter selects what it needs).
    match_metadata : dict
        Parsed ``match.json`` dict. Required keys: ``pitch_length``,
        ``pitch_width``, ``home_team`` (with ``id`` sub-key).
    preserve_native : bool, default False
        When True, attach ``original_event_id`` (the SC ``event_id``)
        as an extra column.

    Returns
    -------
    tuple[pd.DataFrame, ConversionReport]
        SPADL actions and conversion audit trail.

    Examples
    --------
    Convert a single match::

        import pandas as pd, json
        from silly_kicks.spadl import skillcorner

        events = pd.read_csv("dynamic_events.csv", low_memory=False)
        with open("match.json") as f:
            meta = json.load(f)
        actions, report = skillcorner.convert_to_actions(events, meta)
        assert not report.has_unrecognized

    See NOTICE for full bibliographic citations.
    """
    pitch_length = match_metadata["pitch_length"]
    pitch_width = match_metadata["pitch_width"]
    home_team_id = str(match_metadata["home_team"]["id"])

    # --- Filter to player_possession rows with valid actors ---
    pp = events[events["event_type"] == "player_possession"].copy()
    pp = pp[pp["player_id"].notna() & pp["team_id"].notna()].copy()
    pp["team_id"] = pp["team_id"].astype(str)
    pp["player_id"] = pp["player_id"].astype(str)

    # OBE rows for tackle enrichment
    obe = events[events["event_type"] == "on_ball_engagement"].copy()
    if len(obe) > 0:
        obe["team_id"] = obe["team_id"].astype(str)
        obe["player_id"] = obe["player_id"].astype(str)

    # --- Time parsing --- (re-base the continuous broadcast clock to period-relative; ADR-017)
    pp["time_seconds"] = _to_period_relative(_parse_time_start(pp["time_start"]), pp["period"])
    if len(obe) > 0 and "time_start" in obe.columns and "period" in obe.columns:
        obe["time_seconds"] = _to_period_relative(_parse_time_start(obe["time_start"]), obe["period"])

    total_pp = len(pp)

    # --- Coordinate transform (start) ---
    sx, sy = _transform_coords(
        pp["x_start"].astype("float64"),
        pp["y_start"].astype("float64"),
        pitch_length,
        pitch_width,
    )

    # --- End coordinates: per-action-type strategy ---
    # Default: use x_end/y_end (carrier's end position)
    raw_end_x = pp["x_end"].astype("float64")
    raw_end_y = pp["y_end"].astype("float64")

    # For passes: prefer player_targeted_x_reception
    has_targeted = pp["player_targeted_x_reception"].notna()
    use_targeted = has_targeted & (pp["end_type"] == "pass")

    end_x_raw = raw_end_x.copy()
    end_y_raw = raw_end_y.copy()
    end_x_raw[use_targeted] = pp.loc[use_targeted, "player_targeted_x_reception"].astype("float64")
    end_y_raw[use_targeted] = pp.loc[use_targeted, "player_targeted_y_reception"].astype("float64")

    ex, ey = _transform_coords(end_x_raw, end_y_raw, pitch_length, pitch_width)

    # --- Body part dispatch ---
    bodypart_arr = _dispatch_bodypart(
        pp["is_header"] if "is_header" in pp.columns else pd.Series(False, index=pp.index),
        pp["hand_pass"] if "hand_pass" in pp.columns else pd.Series(False, index=pp.index),
    )

    # --- Action type + result dispatch ---
    gi_before = (
        pp["game_interruption_before"].fillna("")
        if "game_interruption_before" in pp.columns
        else pd.Series("", index=pp.index)
    )
    gi_after = (
        pp["game_interruption_after"].fillna("")
        if "game_interruption_after" in pp.columns
        else pd.Series("", index=pp.index)
    )
    end_type = pp["end_type"].fillna("") if "end_type" in pp.columns else pd.Series("", index=pp.index)

    # Cross detection
    third_col = (
        pp["player_targeted_third_pass"]
        if "player_targeted_third_pass" in pp.columns
        else pd.Series(dtype="object", index=pp.index)
    )
    channel_col = (
        pp["player_targeted_channel_pass"]
        if "player_targeted_channel_pass" in pp.columns
        else pd.Series(dtype="object", index=pp.index)
    )
    cross_mask = _is_cross(third_col, channel_col, sx, sy)

    # Next-possession lookups for result logic
    next_team = pp["team_id"].shift(-1)
    same_team_next = (pp["team_id"] == next_team).fillna(False)

    # Short corner/freekick detection: next action same team within 15m
    next_sx = sx.shift(-1)
    next_sy = sy.shift(-1)
    dist_to_next = np.sqrt((sx - next_sx) ** 2 + (sy - next_sy) ** 2)
    is_short = same_team_next & (dist_to_next < 15.0)

    # --- Vectorized dispatch ---
    # Priority 1: set pieces from game_interruption_before
    # Exclude shots -- a free kick or corner that ends with a shot is a shot, not a set piece pass
    is_goalkick = gi_before == "goal_kick_for"
    is_corner = (gi_before == "corner_for") & (end_type != "shot")
    is_throw_in = gi_before == "throw_in_for"
    is_freekick = (gi_before == "free_kick_for") & (end_type != "shot")

    # Priority 2: end_type
    is_shot = end_type == "shot"
    is_pass = (end_type == "pass") & ~cross_mask
    is_cross_action = (end_type == "pass") & cross_mask
    is_clearance = end_type == "clearance"
    # NOTE: foul is attributed to the fouled player (possession holder), not the
    # fouler. Other providers (StatsBomb, Wyscout, Gradient Sports) attribute to the
    # fouler. SkillCorner's foul_suffered is from the victim's perspective and OBE
    # foul_committed cross-referencing is not implemented. This affects per-player
    # VAEP foul credit in cross-provider analyses.
    is_foul = end_type == "foul_suffered"

    # Priority 3: residuals
    is_possession_loss = end_type == "possession_loss"
    is_unknown = end_type == "unknown"

    type_id_arr = np.select(
        [
            is_goalkick,
            is_corner & is_short,
            is_corner & ~is_short,
            is_throw_in,
            is_freekick & is_short,
            is_freekick & ~is_short,
            is_shot,
            is_cross_action,
            is_pass,
            is_clearance,
            is_foul,
            is_possession_loss,
            is_unknown,
        ],
        [
            spadlconfig.actiontype_id["goalkick"],
            spadlconfig.actiontype_id["corner_short"],
            spadlconfig.actiontype_id["corner_crossed"],
            spadlconfig.actiontype_id["throw_in"],
            spadlconfig.actiontype_id["freekick_short"],
            spadlconfig.actiontype_id["freekick_crossed"],
            spadlconfig.actiontype_id["shot"],
            spadlconfig.actiontype_id["cross"],
            spadlconfig.actiontype_id["pass"],
            spadlconfig.actiontype_id["clearance"],
            spadlconfig.actiontype_id["foul"],
            spadlconfig.actiontype_id["non_action"],
            spadlconfig.actiontype_id["non_action"],
        ],
        default=spadlconfig.actiontype_id["non_action"],
    )

    # Result dispatch
    # NOTE (BUG 2 fix, 2026-06-09): goalkick is NOT hard-wired to success -- it falls through to
    # the same possession-based `same_team_next` test as every other open-play / set-piece pass
    # (a goalkick lost to the opponent is a `fail`). Previously `is_goalkick -> success` zeroed
    # the goalkick label variance, corrupting goalkick-completion / VAEP goalkick labels.
    is_goal = gi_after == "goal_for"
    result_id_arr = np.select(
        [
            is_clearance,
            is_foul,
            is_shot & is_goal,
            is_shot & ~is_goal,
            same_team_next,
            ~same_team_next,
        ],
        [
            spadlconfig.result_id["success"],
            spadlconfig.result_id["success"],
            spadlconfig.result_id["success"],
            spadlconfig.result_id["fail"],
            spadlconfig.result_id["success"],
            spadlconfig.result_id["fail"],
        ],
        default=spadlconfig.result_id["fail"],
    )

    # --- Build native actions DataFrame ---
    game_id = str(match_metadata.get("id", "unknown"))
    actions = pd.DataFrame(
        {
            "game_id": game_id,
            "original_event_id": pp["event_id"].astype("object").values,
            "action_id": np.arange(len(pp), dtype="int64"),
            "period_id": pp["period"].astype("int64").values,
            "time_seconds": pp["time_seconds"].values,
            "team_id": pp["team_id"].values,
            "player_id": pp["player_id"].values,
            "start_x": sx.values,
            "start_y": sy.values,
            "end_x": ex.values,
            "end_y": ey.values,
            "type_id": type_id_arr,
            "result_id": result_id_arr,
            "bodypart_id": bodypart_arr,
            "action_provenance": "native",
        }
    )

    # --- Derived actions ---
    # Prepare OBE for inference
    if len(obe) > 0:
        obe_for_inference = obe.copy()
    else:
        obe_for_inference = pd.DataFrame(
            columns=["period", "time_seconds", "team_id", "player_id", "end_type", "x_start", "y_start"]
        )

    pp_for_inference = pp[
        ["event_id", "period", "time_seconds", "team_id", "player_id", "start_type", "x_start", "y_start"]
    ].copy()

    defensive = infer_defensive_actions(pp_for_inference, obe_for_inference)
    if len(defensive) > 0:
        # Transform defensive action coordinates
        d_sx, d_sy = _transform_coords(
            pd.Series(defensive["start_x"].values, dtype="float64"),
            pd.Series(defensive["start_y"].values, dtype="float64"),
            pitch_length,
            pitch_width,
        )
        defensive["start_x"] = d_sx.values
        defensive["start_y"] = d_sy.values
        defensive["end_x"] = d_sx.values
        defensive["end_y"] = d_sy.values
        defensive["game_id"] = game_id
        defensive["original_event_id"] = defensive["event_id"]
        defensive["action_id"] = 0  # will be re-indexed

    keeper_saves_pp = pp[
        ["event_id", "period", "time_seconds", "team_id", "player_id", "end_type", "x_start", "y_start"]
    ].copy()
    if "game_interruption_after" in pp.columns:
        keeper_saves_pp["game_interruption_after"] = pp["game_interruption_after"].values

    ks = infer_keeper_saves(keeper_saves_pp)
    if len(ks) > 0:
        ks_sx, ks_sy = _transform_coords(
            pd.Series(ks["start_x"].values, dtype="float64"),
            pd.Series(ks["start_y"].values, dtype="float64"),
            pitch_length,
            pitch_width,
        )
        ks["start_x"] = ks_sx.values
        ks["start_y"] = ks_sy.values
        ks["end_x"] = ks_sx.values
        ks["end_y"] = ks_sy.values
        ks["game_id"] = game_id
        ks["original_event_id"] = pd.NA
        ks["action_id"] = 0

    # Merge all actions
    parts = [actions]
    if len(defensive) > 0:
        parts.append(defensive[actions.columns])
    if len(ks) > 0:
        parts.append(ks[actions.columns])

    actions = pd.concat(parts, ignore_index=True)
    actions = actions.sort_values(["period_id", "time_seconds"]).reset_index(drop=True)
    actions["action_id"] = np.arange(len(actions), dtype="int64")

    # --- Post-processors ---
    actions = _derive_end_coordinates(actions)
    actions = _add_dribbles(actions)

    # Mark dribbles as derived
    dribble_mask = actions["type_id"] == spadlconfig.actiontype_id["dribble"]
    actions.loc[dribble_mask, "action_provenance"] = "derived"

    # --- LTR normalization (no-op for possession perspective) ---
    actions = to_spadl_ltr(
        actions,
        input_convention=POSSESSION_PERSPECTIVE,
        home_team_id=home_team_id,
    )

    # --- ConversionReport ---
    # NOTE: on_ball_engagement rows are consumed for tackle enrichment but are not
    # directly mapped to native actions, so they appear in excluded_counts alongside
    # passing_option and off_ball_run. This matches other converters' semantics where
    # "excluded" means "not directly mapped to a SPADL action row".
    excluded_types = events[~events["event_type"].isin({"player_possession"})]["event_type"]
    excluded_counts = dict(Counter(excluded_types))

    mapped_counts: dict[str, int] = {}
    id_to_name = {i: name for i, name in enumerate(spadlconfig.actiontypes)}
    for tid in actions["type_id"].to_numpy():
        name = id_to_name.get(int(tid), "unknown")
        mapped_counts[name] = mapped_counts.get(name, 0) + 1

    # Field names verified against schema.py ConversionReport dataclass:
    # mapped_counts, excluded_counts, unrecognized_counts (not _types)
    report = ConversionReport(
        provider="skillcorner",
        total_events=total_pp,
        total_actions=len(actions),
        mapped_counts=mapped_counts,
        excluded_counts=excluded_counts,
        unrecognized_counts={},
    )

    # --- Finalize output ---
    extra = ["original_event_id"] if preserve_native else None
    actions = _finalize_output(actions, SKILLCORNER_SPADL_COLUMNS, extra_columns=extra)

    return actions, report
