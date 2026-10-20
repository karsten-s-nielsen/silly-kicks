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

from ..id_compat import canonical_id
from . import config as spadlconfig
from ._skillcorner_goals import reconcile_goals
from ._skillcorner_inference import infer_defensive_actions, infer_keeper_saves
from ._skillcorner_restarts import derive_restarts
from .base import _add_dribbles, _derive_end_coordinates, sort_actions_chronologically
from .orientation import POSSESSION_PERSPECTIVE, to_spadl_ltr
from .schema import SKILLCORNER_SPADL_COLUMNS, SkillCornerConversionReport
from .utils import _blocked_flag, _ensure_base_signal_columns, _finalize_output


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
    # fillna(False) + bool dtype: the nullable-string `==` yields <NA> for rows without a known outcome,
    # which makes `~po_succ` a per-element Python-bool inversion (deprecated). Those rows are masked out
    # by `has_po &` below, so coercing <NA> to False is behaviour-preserving and keeps a numpy bool array.
    po_succ = (po == "successful").fillna(False).to_numpy(dtype=bool)
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


def _resolve_opponent(pp: pd.DataFrame, match_metadata: dict) -> str | None:
    """Resolve the single opponent team id from the possession team ids (decision C6, spec 5.2).

    The opponent is the distinct event ``team_id`` that is not ``home_team.id``. Ids compare
    through :mod:`silly_kicks.id_compat` (``canonical_id``), never ``str()`` -- a float id would
    stringify to ``"866.0"`` and never equal ``"866"``. Returns the canonical opponent id, or
    ``None`` when the slice carries a single team (no ``_against`` restart is derivable; the caller
    counts it unresolved).

    Raises
    ------
    ValueError
        When more than two distinct team ids appear; when ``home_team.id`` is absent from the event
        team ids; or when ``away_team.id`` is present and the event team set is not
        ``{home_team.id, away_team.id}``. Each message names ``match_metadata["id"]``.
    """
    match_id = match_metadata.get("id")
    home_id = str(canonical_id(match_metadata["home_team"]["id"]))
    team_ids = {str(canonical_id(t)) for t in pp["team_id"].dropna().unique()}
    if len(team_ids) > 2:
        raise ValueError(
            f"SkillCorner match {match_id!r}: more than two distinct team ids in the possessions "
            f"{sorted(team_ids)}; cannot resolve the opponent."
        )
    if home_id not in team_ids:
        raise ValueError(
            f"SkillCorner match {match_id!r}: home_team.id {home_id!r} is not among the event team "
            f"ids {sorted(team_ids)}."
        )
    away = match_metadata.get("away_team") or {}
    away_raw = away.get("id")
    if away_raw is not None:
        away_id = str(canonical_id(away_raw))
        if team_ids != {home_id, away_id}:
            raise ValueError(
                f"SkillCorner match {match_id!r}: event team set {sorted(team_ids)} does not equal "
                f"{{home_team.id, away_team.id}} = {sorted({home_id, away_id})}."
            )
    others = team_ids - {home_id}
    if not others:
        return None
    return next(iter(others))


def convert_to_actions(
    events: pd.DataFrame,
    match_metadata: dict,
    *,
    preserve_native: bool = False,
) -> tuple[pd.DataFrame, SkillCornerConversionReport]:
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
    tuple[pd.DataFrame, SkillCornerConversionReport]
        SPADL actions and the conversion audit trail (a ``ConversionReport`` subclass carrying the
        TF-46 restart + goals diagnostics).

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

    # Order-insensitivity (chronological-action_id invariant): sort pp chronologically NOW --
    # before the sx/sy transforms and the .shift(-1)-based same_team_next / is_short lookups
    # below, whose results (and thus type_id/result_id and the derived-row times) otherwise
    # depend on raw input order. SkillCorner's continuous broadcast clock makes ties rare;
    # mergesort keeps any co-timestamped rows in native order (§3a inter-timestamp scope). pp
    # carries `period` (not `period_id`) at this stage. The final combined sort
    # (period_id, time_seconds) at the end still runs to interleave the defensive/goalkick rows.
    pp = sort_actions_chronologically(pp, by=("period", "time_seconds"))

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

    # --- Vectorized dispatch ---
    # Every possession keeps its own type from end_type (spec section 5.3, F2 fix). The former
    # priority-1 GI relabel (receiver -> set piece) and the native 15 m short rule are deleted: each
    # restart is now a separate DERIVED row (_skillcorner_restarts.derive_restarts), so a receiver's
    # shot or cross stays visible and a defence-won restart is no longer dropped.
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

    # Priority: residuals
    is_possession_loss = end_type == "possession_loss"
    is_unknown = end_type == "unknown"

    type_id_arr = np.select(
        [
            is_shot,
            is_cross_action,
            is_pass,
            is_clearance,
            is_foul,
            is_possession_loss,
            is_unknown,
        ],
        [
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
            # Base signal columns pre-created so the derived-restart concat (which sets real values)
            # survives the `[actions.columns]` reindex; <NA> on native rows.
            "restart_evidence": pd.NA,
            "type_source": pd.NA,
            "repair_kind": pd.NA,
        }
    )

    # --- Derived restarts (marker-keyed; spec section 5) ---
    opponent = _resolve_opponent(pp, match_metadata)
    restart_start_type = (
        pp["start_type"].fillna("").astype(str).to_numpy()
        if "start_type" in pp.columns
        else np.full(len(pp), "", dtype=object)
    )
    pp_restart = pd.DataFrame(
        {
            "event_id": pp["event_id"].astype("object").to_numpy(),
            "period_id": pp["period"].astype("int64").to_numpy(),
            "time_seconds": pp["time_seconds"].to_numpy(),
            "team_id": pp["team_id"].to_numpy(),
            "start_type": restart_start_type,
            "game_interruption_before": gi_before.to_numpy(),
            "game_interruption_after": gi_after.to_numpy(),
            "start_x": sx.to_numpy(),
            "start_y": sy.to_numpy(),
        }
    )
    derived_restarts = derive_restarts(pp_restart, opponent, match_metadata)

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

    # Derived defensive / keeper-save actions have deterministic (non-completion) results -> tag
    # result_source "native"; they are not restarts/repairs -> base signal columns <NA>. All four are
    # set so the `[actions.columns]` reindex below finds every column.
    for _derived in (defensive, ks):
        if len(_derived) > 0:
            for _col, _val in (
                ("result_source", "native"),
                ("restart_evidence", pd.NA),
                ("type_source", pd.NA),
                ("repair_kind", pd.NA),
            ):
                if _col not in _derived.columns:
                    _derived[_col] = _val

    # Merge all actions
    parts = [actions]
    if len(defensive) > 0:
        parts.append(defensive[actions.columns])
    if len(ks) > 0:
        parts.append(ks[actions.columns])
    if len(derived_restarts) > 0:
        parts.append(derived_restarts[actions.columns])

    actions = pd.concat(parts, ignore_index=True)

    # --- Goals reconciliation (spec section 6): repairs shot results, sets repair_kind, and may append
    # own-goal / derived-shot rows. Runs on the combined frame before the sort so appended rows order in.
    pp_goals = pd.DataFrame(
        {
            "event_id": pp["event_id"].astype("object").to_numpy(),
            "period_id": pp["period"].astype("int64").to_numpy(),
            "time_seconds": pp["time_seconds"].to_numpy(),
            "team_id": pp["team_id"].to_numpy(),
            "player_id": pp["player_id"].to_numpy(),
            "end_type": end_type.to_numpy(),
            "start_type": restart_start_type,
            "game_interruption_before": gi_before.to_numpy(),
            "game_interruption_after": gi_after.to_numpy(),
            "team_score": pp["team_score"].to_numpy() if "team_score" in pp.columns else np.nan,
            "opponent_team_score": (
                pp["opponent_team_score"].to_numpy() if "opponent_team_score" in pp.columns else np.nan
            ),
        }
    )
    actions, goals_report = reconcile_goals(actions, pp_goals, match_metadata)

    # ADR-065 order-insensitive combined sort (the placeholder restart times make co-timestamped rows
    # likelier, so the guarded chronological path matters); _assert_chronological_action_id runs in
    # _finalize_output.
    actions = sort_actions_chronologically(actions, by=("period_id", "time_seconds")).reset_index(drop=True)
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

    # --- Restart + goals census (C4): per-type and per-restart_evidence derived-row counts, the
    # repeated-restart diagnostic, and the goals-reconciliation summary. ``derived_restarts`` is the
    # pre-concat frame (its .attrs carry the unresolved-against count).
    restart_counts: dict[str, int] = {}
    restart_evidence_counts: dict[str, int] = {}
    repeated_restart_count = 0
    if len(derived_restarts) > 0:
        for tid in derived_restarts["type_id"].to_numpy():
            name = id_to_name.get(int(tid), "unknown")
            restart_counts[name] = restart_counts.get(name, 0) + 1
        restart_evidence_counts = {
            str(k): int(v) for k, v in derived_restarts["restart_evidence"].value_counts().items()
        }
        repeated_restart_count = int(
            derived_restarts["restart_evidence"].isin(["chain", "chain_ambiguous", "tag_only"]).sum()
        )

    report = SkillCornerConversionReport(
        provider="skillcorner",
        total_events=total_pp,
        total_actions=len(actions),
        mapped_counts=mapped_counts,
        excluded_counts=excluded_counts,
        unrecognized_counts={},
        restart_counts=restart_counts,
        restart_evidence_counts=restart_evidence_counts,
        restart_unresolved=int(derived_restarts.attrs.get("unresolved_against", 0)),
        goals_repaired=dict(goals_report.repaired),
        goals_ambiguous=goals_report.ambiguous,
        goals_unresolved=goals_report.unresolved,
        goals_coverage_gap=goals_report.coverage_gap,
        repeated_restart_count=repeated_restart_count,
        kickoff_unresolved=0,
    )

    # --- Finalize output ---
    # Block-detection: SkillCorner records no shot/cross-block signal (real-data verified, both
    # tiers) -> all pd.NA. Set AFTER the derived-actions pd.concat above to avoid the
    # `defensive[actions.columns]` reindex KeyError on frames that lack these columns.
    extra = ["original_event_id"] if preserve_native else None
    actions["shot_blocked"] = _blocked_flag(len(actions))
    actions["cross_blocked"] = _blocked_flag(len(actions))
    # Base signal columns are pre-created on every part above (native <NA>; derived restarts carry real
    # result_source / restart_evidence / type_source; goals reconciliation (Task 2.4) sets repair_kind).
    # _ensure_base_signal_columns is the contract no-op guaranteeing presence before strict projection.
    actions = _ensure_base_signal_columns(actions)
    actions = _finalize_output(actions, SKILLCORNER_SPADL_COLUMNS, extra_columns=extra)

    return actions, report
