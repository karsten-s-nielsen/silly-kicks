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

import warnings
from collections import Counter

import numpy as np
import pandas as pd

from . import config as spadlconfig
from ._skillcorner_inference import infer_defensive_actions, infer_keeper_saves, infer_penalties, infer_restarts
from .base import (
    _DERIVE_END_TYPE_IDS,
    _add_dribbles,
    _derive_end_coordinates,
    classify_deliveries,
    sort_actions_chronologically,
)
from .orientation import POSSESSION_PERSPECTIVE, to_spadl_ltr
from .schema import SKILLCORNER_SPADL_COLUMNS, ConversionReport
from .utils import _blocked_flag, _finalize_output, _validate_preserve_native


def _scale_to_spadl(
    x: pd.Series,
    y: pd.Series,
    pitch_length: int | float,
    pitch_width: int | float,
) -> tuple[pd.Series, pd.Series]:
    """Affine map from centred metres to the SPADL 105x68 frame. NO clamping.

    This is the single-sourced coordinate truth for SkillCorner: the EVENTS converter calls it
    via ``_transform_coords`` (which clamps afterwards, safe because an action's location is
    on-pitch by construction); the TRACKING builder (``tracking/skillcorner.py``) calls it
    DIRECTLY, because tracking is full of legitimately off-pitch positions -- an out-of-play
    ball, a keeper behind his line, and decisively a ball that has crossed the goal line, which
    is what a goal IS. Clamping tracking would erase goal-vs-save. See spec 3.4.
    """
    half_length = pitch_length / 2
    half_width = pitch_width / 2
    return (x / half_length) * 52.5 + 52.5, (y / half_width) * 34.0 + 34.0


def _transform_coords(
    x: pd.Series,
    y: pd.Series,
    pitch_length: int | float,
    pitch_width: int | float,
) -> tuple[pd.Series, pd.Series]:
    """Rescale centered meters to SPADL 0-based frame, clamped to the pitch (EVENTS only).

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
    x_out, y_out = _scale_to_spadl(x, y, pitch_length, pitch_width)
    # Clamp to SPADL pitch boundaries (raw data can slightly exceed pitch dims).
    # NEVER reuse this for tracking -- see _scale_to_spadl's docstring.
    return x_out.clip(lower=0.0, upper=105.0), y_out.clip(lower=0.0, upper=68.0)


# SkillCorner's `time_start` is the CONTINUOUS broadcast clock (the 2nd half shows 45:00+,
# extra time 90:00+/105:00+), so each period's nominal start offset must be subtracted to get
# the period-relative seconds the tracking-frame contract requires (ADR-017). The offsets are
# the regulation period boundaries (45/90/105/120 min), matching the Gradient Sports adapter.
_PERIOD_START_SECONDS = {1: 0.0, 2: 45 * 60.0, 3: 90 * 60.0, 4: 105 * 60.0, 5: 120 * 60.0}

#: The possession-end fields a native SPADL row is built from (ADR-133).
_REQUIRED_END_COLUMNS = ("time_end", "x_end", "y_end")


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


def _native_completion_result(
    pass_outcome: pd.Series,
    received: pd.Series,
    same_team_next: pd.Series,
    is_passlike: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-construct pass completion for SkillCorner pass/set-piece rows (D-S8, reviews N1/G1).

    SkillCorner carries a NATIVE pass outcome (``pass_outcome``) the converter previously ignored,
    defaulting every pass to the ``same_team_next`` possession proxy (a poor completion label:
    ~0.72-0.79 agreement with the native outcome, and it overstated goal-kick success by ~16pp).
    This routes ``result_id`` through the native construct (SPADL ``success`` = the pass reached a
    teammate), tiered for provenance:

    * **native**  -- ``pass_outcome`` present: ``successful`` -> success; ``unsuccessful`` /
      ``offside`` -> fail. The one construct giving BOTH classes from a single rule.
    * **inferred** -- ``pass_outcome`` absent but ``received==True``: a confirmed completion ->
      success. SUCCESS-ONLY: ``received==False`` is NOT a fail signal (it may be a completion to a
      NON-targeted teammate; N1), so it falls to the residual, never forced to fail.
    * **stopgap**  -- neither native field: the ``same_team_next`` proxy, kept for VAEP ``result_id``
      coverage only.

    Only the **native** tier is clean enough for GK-distribution model training (G1: inferred is
    structurally positive-only -> would bias calibration).
    Non-passlike rows are left at the fail default + ``stopgap`` tag; the converter's ``np.select``
    overrides them with their own explicit (clearance/foul/shot) result.

    Returns ``(result_id_array, result_source_array)``.
    """
    n = len(pass_outcome)
    rid = np.full(n, spadlconfig.result_id["fail"], dtype=int)
    src = np.full(n, "stopgap", dtype=object)
    po = pass_outcome.astype("string").str.lower()
    has_po = po.isin(["successful", "unsuccessful", "offside"]).to_numpy()
    po_succ = (po == "successful").to_numpy()
    rec_true = received.astype("string").str.lower().isin(["true", "1", "1.0"]).to_numpy()
    stn = same_team_next.fillna(False).to_numpy()
    passlike = is_passlike.fillna(False).to_numpy()
    # tier 1: native pass_outcome (both classes)
    rid = np.where(passlike & has_po & po_succ, spadlconfig.result_id["success"], rid)
    rid = np.where(passlike & has_po & ~po_succ, spadlconfig.result_id["fail"], rid)
    src = np.where(passlike & has_po, "native", src)
    # tier 2: received==True -> success-only augmentation (clean success; tagged "inferred", G1
    # training-excluded). received==False is NOT a fail signal (N1).
    rec_aug = passlike & ~has_po & rec_true
    rid = np.where(rec_aug, spadlconfig.result_id["success"], rid)
    src = np.where(rec_aug, "inferred", src)
    # tier 3 (residual): flagged stopgap via same_team_next (best-available VAEP coverage only)
    resid = passlike & ~has_po & ~rec_true
    rid = np.where(resid & stn, spadlconfig.result_id["success"], rid)
    rid = np.where(resid & ~stn, spadlconfig.result_id["fail"], rid)
    src = np.where(resid, "stopgap", src)
    return rid, src


def convert_to_actions(
    events: pd.DataFrame,
    match_metadata: dict,
    *,
    preserve_native: bool | list[str] = False,
) -> tuple[pd.DataFrame, ConversionReport]:
    """Convert SkillCorner dynamic_events.csv to SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        Full ``dynamic_events.csv`` as a DataFrame (all 294 columns;
        the converter selects what it needs).
    match_metadata : dict
        Parsed ``match.json`` dict. Required keys: ``pitch_length``,
        ``pitch_width``, ``home_team`` (with ``id`` sub-key). ``away_team`` (with ``id``) is optional: without it,
        the away team is the one other team id of the events (the team of a derived restart row "against" home).
    preserve_native : bool or list of str, default False
        ``True`` adds no column: ``original_event_id`` is a schema column already (earlier releases raised a
        ``ValueError`` on ``True``). A list carries the named
        event fields (for example ``["team_score", "opponent_team_score"]``, the provider's running score) on the
        native rows as extra columns; derived rows carry ``NaN`` there (action-time repair spec section 4.15).

    Returns
    -------
    tuple[pd.DataFrame, ConversionReport]
        SPADL actions and conversion audit trail.

    Raises
    ------
    ValueError
        If ``time_end``, ``x_end`` or ``y_end`` is missing (ADR-133): a native row is the on-ball action at the
        possession END, so its time and start come from them.

    Warns
    -----
    UserWarning
        Once, with the count, when ``player_possession`` rows have no ``time_end``; their ``time_seconds`` is
        NaN, never filled from ``time_start``.

    Notes
    -----
    A native row (pass, shot, clearance, foul, loss) is the receiver's own action at the possession end:
    ``time_end`` and ``x_end``/``y_end``; it takes its type from its own ``end_type``. Each restart (corner, free
    kick, goal kick, throw-in) and each penalty is its own derived row from the interruption markers, with
    ``player_id`` ``<NA>`` (the data has no taker), a rule-point start, and ``time_source`` ``"placeholder"``
    (a restart with a first possession: the first reception minus 0.02 s, an upper bound of the kick) or
    ``"order_only"`` (no first possession: the time keeps the row order only); see the action-time repair spec
    sections 4.2 and 4.17. Start-type interceptions/recoveries and keeper saves stay at the possession start.
    Deliveries follow :func:`silly_kicks.spadl.base.classify_deliveries` and its event-only limits (ADR-133).
    SkillCorner has no native cross label (``cross_native`` is ``<NA>``), and its events miss defenders'
    touches in the box, so a cross whose first touch is a defender's reads as a pass in events alone.

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
    missing_cols = [c for c in _REQUIRED_END_COLUMNS if c not in events.columns]
    if missing_cols:
        raise ValueError(
            f"skillcorner.convert_to_actions: missing column(s) {missing_cols}. A SPADL row is the on-ball action "
            "at the possession END (time_end, x_end, y_end), not the reception at its start (ADR-133)."
        )

    native_fields: list[str] = list(preserve_native) if isinstance(preserve_native, list) else []
    _validate_preserve_native(events, native_fields, "skillcorner", schema=SKILLCORNER_SPADL_COLUMNS)

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
    # A player_possession row runs from the reception (time_start, x_start) to the on-ball action at its END
    # (time_end, x_end): the pass, shot, clearance, set-piece delivery, foul or loss. The native SPADL row is that
    # action, so it takes the end (ADR-133). `start_seconds` keeps the reception time for the
    # possession order and for the rows that happen at the start (start-type interceptions, keeper saves).
    pp["start_seconds"] = _to_period_relative(_parse_time_start(pp["time_start"]), pp["period"])
    time_end = pp["time_end"].astype("string")
    has_end = time_end.notna() & (time_end.str.len() > 0)
    end_seconds = pd.Series(np.nan, index=pp.index, dtype="float64")
    if has_end.any():
        end_seconds[has_end] = _to_period_relative(
            _parse_time_start(time_end[has_end].astype(str)), pp.loc[has_end, "period"]
        )
    pp["time_seconds"] = end_seconds
    n_missing_end = int((~has_end).sum())
    if n_missing_end:
        warnings.warn(
            f"skillcorner: {n_missing_end} player_possession row(s) without time_end; their time_seconds is NaN "
            "(never filled from time_start).",
            UserWarning,
            stacklevel=2,
        )
    if len(obe) > 0 and "time_start" in obe.columns and "period" in obe.columns:
        obe["time_seconds"] = _to_period_relative(_parse_time_start(obe["time_start"]), obe["period"])

    # Order-insensitivity (chronological-action_id invariant): sort pp chronologically NOW --
    # before the sx/sy transforms and the .shift(-1)-based same_team_next / is_short lookups
    # below, whose results (and thus type_id/result_id and the derived-row times) otherwise
    # depend on raw input order. The possession order is the reception order (`start_seconds`,
    # always known); a possession never ends after the next one starts (measured).
    # mergesort keeps any co-timestamped rows in native order (§3a inter-timestamp scope). pp
    # carries `period` (not `period_id`) at this stage. The final combined sort
    # (period_id, time_seconds) at the end still runs to interleave the defensive/goalkick rows.
    pp = sort_actions_chronologically(pp, by=("period", "start_seconds"))

    total_pp = len(pp)

    # --- Coordinate transform (start = the kick point, x_end / y_end) ---
    sx, sy = _transform_coords(
        pp["x_end"].astype("float64"),
        pp["y_end"].astype("float64"),
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

    # --- Body part dispatch ---
    bodypart_arr = _dispatch_bodypart(
        pp["is_header"] if "is_header" in pp.columns else pd.Series(False, index=pp.index),
        pp["hand_pass"] if "hand_pass" in pp.columns else pd.Series(False, index=pp.index),
    )

    # --- Action type + result dispatch ---
    gi_after = (
        pp["game_interruption_after"].fillna("")
        if "game_interruption_after" in pp.columns
        else pd.Series("", index=pp.index)
    )
    end_type = pp["end_type"].fillna("") if "end_type" in pp.columns else pd.Series("", index=pp.index)

    # Crosses: no zone rule here. Every pass is typed `pass`, and the one geometric delivery rule
    # (classify_deliveries, ADR-133) re-types it after LTR normalization below.

    # Next-possession lookups for result logic
    next_team = pp["team_id"].shift(-1)
    same_team_next = (pp["team_id"] == next_team).fillna(False)

    # --- Vectorized dispatch ---
    # The first possession after a restart takes its type from its own end_type (R30): the restart itself is a
    # derived row (infer_restarts below), never the receiver's possession.
    is_shot = end_type == "shot"
    is_pass = end_type == "pass"
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
            is_shot,
            is_pass,
            is_clearance,
            is_foul,
            is_possession_loss,
            is_unknown,
        ],
        [
            spadlconfig.actiontype_id["shot"],
            spadlconfig.actiontype_id["pass"],
            spadlconfig.actiontype_id["clearance"],
            spadlconfig.actiontype_id["foul"],
            spadlconfig.actiontype_id["non_action"],
            spadlconfig.actiontype_id["non_action"],
        ],
        default=spadlconfig.actiontype_id["non_action"],
    )

    # A pass-class row with no targeted reception ends at the NEXT possession's reception (its x_start/y_start; pp is
    # in reception order), the ball's destination (ADR-133). The next action's start is the
    # next KICK, so _derive_end_coordinates would end the pass at the receiver's kick and drop the receiver's
    # carry. The last possession of a period has no next reception: its end stays the start, and
    # _derive_end_coordinates applies as before.
    by_period = pp.groupby("period", sort=False)
    next_rx = by_period["x_start"].shift(-1).astype("float64")
    next_ry = by_period["y_start"].shift(-1).astype("float64")
    to_next_reception = np.isin(type_id_arr, list(_DERIVE_END_TYPE_IDS)) & ~use_targeted & next_rx.notna()
    end_x_raw[to_next_reception] = next_rx[to_next_reception]
    end_y_raw[to_next_reception] = next_ry[to_next_reception]

    ex, ey = _transform_coords(end_x_raw, end_y_raw, pitch_length, pitch_width)

    # Result dispatch
    # D-S8 (2026-06-09): pass/set-piece result_id comes from the NATIVE pass outcome
    # (`pass_outcome` -> `received` -> residual `same_team_next`) via `_native_completion_result`,
    # NOT the bare `same_team_next` possession proxy (which overstated goal-kick success ~16pp and
    # agreed with the native outcome only ~0.72-0.79). Clearance/foul/shot keep their explicit,
    # deterministic results. `result_source` records the per-row completion-label tier
    # (native/inferred/stopgap) -- only `native` trains the GK-completion model (G1).
    is_goal = gi_after == "goal_for"
    pass_outcome_col = pp["pass_outcome"] if "pass_outcome" in pp.columns else pd.Series(np.nan, index=pp.index)
    received_col = pp["received"] if "received" in pp.columns else pd.Series(np.nan, index=pp.index)
    is_passlike = pd.Series(
        np.isin(
            type_id_arr,
            [
                spadlconfig.actiontype_id["pass"],
                spadlconfig.actiontype_id["cross"],
                spadlconfig.actiontype_id["goalkick"],
                spadlconfig.actiontype_id["throw_in"],
                spadlconfig.actiontype_id["corner_short"],
                spadlconfig.actiontype_id["corner_crossed"],
                spadlconfig.actiontype_id["freekick_short"],
                spadlconfig.actiontype_id["freekick_crossed"],
            ],
        ),
        index=pp.index,
    )
    passlike_rid, passlike_src = _native_completion_result(pass_outcome_col, received_col, same_team_next, is_passlike)
    result_id_arr = np.select(
        [
            is_clearance,
            is_foul,
            is_shot & is_goal,
            is_shot & ~is_goal,
            is_passlike.to_numpy(),
        ],
        [
            spadlconfig.result_id["success"],
            spadlconfig.result_id["success"],
            spadlconfig.result_id["success"],
            spadlconfig.result_id["fail"],
            passlike_rid,
        ],
        default=spadlconfig.result_id["fail"],
    )
    # result_source: native/inferred/stopgap for passlike rows; "native" for the explicit
    # deterministic branches (clearance/foul/shot) -- their result is not a completion estimate.
    result_source_arr = np.where(is_passlike.to_numpy(), passlike_src, "native")

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
            "result_source": result_source_arr,
            "bodypart_id": bodypart_arr,
            "action_provenance": "native",
        }
    )
    for field in native_fields:
        actions[field] = pp[field].to_numpy()

    # --- Derived actions ---
    # Prepare OBE for inference
    if len(obe) > 0:
        obe_for_inference = obe.copy()
    else:
        obe_for_inference = pd.DataFrame(
            columns=["period", "time_seconds", "team_id", "player_id", "end_type", "x_start", "y_start"]
        )

    # The derived rows happen at the possession START (ADR-133), so the inference reads the reception time.
    pp_for_inference = pp[
        ["event_id", "period", "start_seconds", "team_id", "player_id", "start_type", "x_start", "y_start"]
    ].rename(columns={"start_seconds": "time_seconds"})

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
        ["event_id", "period", "start_seconds", "team_id", "player_id", "end_type", "x_start", "y_start"]
    ].rename(columns={"start_seconds": "time_seconds"})
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

    # Derived defensive / keeper-save actions have deterministic (non-completion) results -> tag
    # their result_source "native" so the `[actions.columns]` reindex below finds the column.
    for _derived in (defensive, ks):
        if len(_derived) > 0 and "result_source" not in _derived.columns:
            _derived["result_source"] = "native"

    # Merge all actions. A derived row has no event of its own, so the native fields of a `preserve_native` list
    # (R56: the running score) are NaN on it; any other missing column still raises.
    def _as_actions(derived: pd.DataFrame) -> pd.DataFrame:
        return derived.assign(**{c: np.nan for c in native_fields if c not in derived.columns})[actions.columns]

    parts = [actions]
    if len(defensive) > 0:
        parts.append(_as_actions(defensive))
    if len(ks) > 0:
        parts.append(_as_actions(ks))

    actions = pd.concat(parts, ignore_index=True)
    actions = actions.sort_values(["period_id", "time_seconds"]).reset_index(drop=True)
    actions["action_id"] = np.arange(len(actions), dtype="int64")

    # --- Post-processors ---
    actions = _derive_end_coordinates(actions)

    # --- Derived restart and penalty rows (action-time repair spec sections 4.2, 4.17) ---
    # Added AFTER the end derivation: their ends are defined (a rule point, the first reception, or the start
    # itself), and they must not become the "next row" that derives a native row's end.
    sx_start, sy_start = _transform_coords(
        pp["x_start"].astype("float64"), pp["y_start"].astype("float64"), pitch_length, pitch_width
    )
    pp_restarts = pp[["event_id", "period", "start_seconds", "time_seconds", "team_id"]].copy()
    for col in ("game_interruption_before", "game_interruption_after"):
        pp_restarts[col] = pp[col].to_numpy() if col in pp.columns else None
    for col in ("team_score", "opponent_team_score"):
        if col in pp.columns:
            pp_restarts[col] = pp[col].to_numpy()
    pp_restarts["sx_end"], pp_restarts["sy_end"] = sx.to_numpy(), sy.to_numpy()
    pp_restarts["sx_start"], pp_restarts["sy_start"] = sx_start.to_numpy(), sy_start.to_numpy()
    away = match_metadata.get("away_team", {}).get("id")
    if away is None:  # not a required key: the away team is the events' other team (never the "" team)
        others = {str(t) for t in pp["team_id"].dropna().unique()} - {home_team_id}
        away = next(iter(others)) if len(others) == 1 else ""
    teams = (home_team_id, str(away))
    restart_parts = [
        part
        for part in (
            infer_restarts(pp_restarts, teams),
            infer_penalties(pp_restarts, teams, match_metadata),
        )
        if len(part) > 0
    ]
    if restart_parts:
        restarts = pd.concat(restart_parts, ignore_index=True)
        restarts["game_id"] = game_id
        restarts["original_event_id"] = restarts["event_id"]
        restarts["action_id"] = 0
        actions = pd.concat(
            [actions, restarts[[*actions.columns.intersection(restarts.columns), "time_source"]]], ignore_index=True
        )
        actions = actions.sort_values(["period_id", "time_seconds"], kind="mergesort").reset_index(drop=True)
        actions["action_id"] = np.arange(len(actions), dtype="int64")

    # --- LTR normalization (no-op for possession perspective) ---
    # Runs BEFORE _add_dribbles (ADR-133): the delivery rule needs canonical LTR rows with no
    # synthetic dribbles.
    actions = to_spadl_ltr(
        actions,
        input_convention=POSSESSION_PERSPECTIVE,
        home_team_id=home_team_id,
    )
    # One delivery rule for every provider (ADR-133). SkillCorner has no native cross label, so cross_native
    # stays <NA> (set below). Event-only limit: SkillCorner events miss defenders' in-box touches, so a cross
    # cleared by a defender reads as a pass here.
    actions = classify_deliveries(actions)
    # The receiver's own action is timed on SkillCorner's tracking clock; the derived restart and penalty rows
    # carry their own "placeholder" / "order_only" (action-time repair spec section 4.1).
    if "time_source" in actions.columns:
        actions["time_source"] = actions["time_source"].fillna("tracking")
    else:
        actions["time_source"] = "tracking"
    actions = _add_dribbles(actions)

    # Mark dribbles as derived
    dribble_mask = actions["type_id"] == spadlconfig.actiontype_id["dribble"]
    actions.loc[dribble_mask, "action_provenance"] = "derived"

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
    # Block-detection: SkillCorner records no shot/cross-block signal (real-data verified, both
    # tiers) -> all pd.NA. Set AFTER the derived-actions pd.concat above to avoid the
    # `defensive[actions.columns]` reindex KeyError on frames that lack these columns.
    # ``preserve_native=True`` adds no column: ``original_event_id``, its former extra column, is a schema column
    # now, so naming it again raised the extra-columns overlap ValueError (R62).
    extra: list[str] | None = native_fields or None
    actions["shot_blocked"] = _blocked_flag(len(actions))
    actions["cross_blocked"] = _blocked_flag(len(actions))
    actions["cross_native"] = _blocked_flag(len(actions))  # SkillCorner has no native cross label (ADR-133)
    actions = _finalize_output(actions, SKILLCORNER_SPADL_COLUMNS, extra_columns=extra)

    return actions, report
