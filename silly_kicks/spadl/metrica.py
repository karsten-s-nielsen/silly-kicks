"""Metrica DataFrame SPADL converter.

Converts already-normalized Metrica event DataFrames (e.g., parsed from
Metrica's open-data CSV / EPTS-JSON formats) to SPADL actions.

Consumers with raw Metrica files should use ``silly_kicks.spadl.kloppy`` after
``kloppy.metrica.load_event(...)``.

Metrica ``type`` / ``subtype`` vocabulary
-----------------------------------------

Recognized event types (case-sensitive ``UPPER``):

==================== =====================================================
``type``             SPADL mapping
==================== =====================================================
``PASS``             ``pass`` (default) / ``cross`` / ``goalkick`` /
                     ``keeper_pick_up + pass`` (when by GK and goalkeeper_ids)
``SHOT``             ``shot`` (with set-piece composition for FREE KICK)
``RECOVERY``         ``interception`` (default) / ``keeper_pick_up`` (when by GK)
``CHALLENGE``        ``tackle`` (when WON) / ``keeper_claim`` (AERIAL-WON by GK) /
                     dropped (LOST / other AERIAL variants)
``BALL LOST``        ``bad_touch`` (fail)
``FAULT``            ``foul`` (with CARD pairing for cards)
``SET PIECE``        ``freekick_short`` / ``corner_short`` / ``throw_in`` / ``goalkick``
==================== =====================================================

Goalkeeper coverage contract
----------------------------

Metrica's source format does NOT natively mark GK actions in any event
subtype — the taxonomy is purely positional/contextual (PASS / CARRY /
CHALLENGE / etc.). To surface ``keeper_*`` SPADL actions, callers must
pass ``goalkeeper_ids`` from match metadata (squad records or
``dim_players`` join on the registered position group).

Without ``goalkeeper_ids``, the output contains zero ``keeper_*`` actions
(preserves 1.9.0 default behaviour).

With ``goalkeeper_ids``, conservative routing applies:

- ``PASS`` (any subtype) by GK → synthesize ``keeper_pick_up + pass``
- ``RECOVERY`` (any subtype) by GK → ``keeper_pick_up``
- ``CHALLENGE`` ``AERIAL-WON`` by GK → ``keeper_claim``
- All other event types unchanged (a GK taking a free kick is still
  ``freekick_short``, not a keeper action — set pieces are positional
  acts, not GK acts in the SPADL vocabulary)

Bug history
-----------

Pre-1.10.0: silly-kicks Metrica converter emitted zero ``keeper_*`` actions
on every match (no source GK markers, no parameter to disambiguate).
``add_gk_role`` and ``add_pre_shot_gk_context`` correctly emitted NULL on
the resulting SPADL — but the upstream gap meant lakehouse production had
zero GK coverage on Metrica. Fixed in 1.10.0 by adding the
``goalkeeper_ids: set[str] | None`` parameter.

See ``docs/superpowers/specs/2026-04-29-gk-converter-coverage-parity-design.md``
for the full design rationale.
"""

import warnings
from collections import Counter
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..tracking import _direction
from . import config as spadlconfig
from .base import _add_dribbles, _fix_clearances
from .orientation import PER_PERIOD_ABSOLUTE, to_spadl_ltr, validate_input_convention
from .schema import KLOPPY_SPADL_COLUMNS, ConversionReport
from .utils import _finalize_output, _validate_input_columns, _validate_preserve_native

EXPECTED_INPUT_COLUMNS: set[str] = {
    "match_id",
    "event_id",
    "type",
    "subtype",
    "period",
    "start_time_s",
    "end_time_s",
    "player",
    "team",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
}

_MAPPED_TYPES: frozenset[str] = frozenset(
    {
        "PASS",
        "SHOT",
        "RECOVERY",
        "CHALLENGE",
        "BALL LOST",
        "FAULT",
        "SET PIECE",
    }
)

_EXCLUDED_TYPES: frozenset[str] = frozenset(
    {
        "BALL OUT",
        "FAULT RECEIVED",
        "CARD",
        "SUBSTITUTION",
    }
)


_MIGRATION_3_0_1_MESSAGE_METRICA = (
    "silly_kicks.spadl.metrica.convert_to_actions requires explicit per-period "
    "direction info as of silly-kicks 3.0.1. Metrica bronze events ship "
    "per-period-absolute (teams switch ends after halftime). Pass "
    "`home_team_start_left=True/False` from Metrica metadata.xml "
    "(Period entity team-direction attribute), OR pass "
    "`home_attacks_right_per_period={1: bool, 2: bool, ...}` directly. "
    "See ADR-006 erratum + CHANGELOG 3.0.1 for the migration snippet."
)


def _resolve_per_period_flips_metrica(
    *,
    events: pd.DataFrame,
    home_team_start_left: bool | None,
    home_team_start_left_extratime: bool | None,
    home_attacks_right_per_period: Mapping[int, bool] | None,
) -> dict[int, bool]:
    """Resolve the bool-pair-OR-mapping kwargs to a concrete per-period flip dict.

    Mutual exclusion + loud failure on missing-input. Mirrors PFF events-side
    and Sportec events-side API exactly.
    """
    if home_attacks_right_per_period is not None and (
        home_team_start_left is not None or home_team_start_left_extratime is not None
    ):
        raise ValueError(
            "metrica.convert_to_actions: pass home_team_start_left[+_extratime] OR "
            "home_attacks_right_per_period, not both."
        )

    if home_attacks_right_per_period is not None:
        return dict(home_attacks_right_per_period)

    if home_team_start_left is None and home_team_start_left_extratime is not None:
        raise ValueError(
            "metrica.convert_to_actions: home_team_start_left_extratime supplied without "
            "home_team_start_left. ET flag is meaningful only as a refinement of the "
            "regular-time flag."
        )

    if home_team_start_left is None:
        raise ValueError(_MIGRATION_3_0_1_MESSAGE_METRICA)

    if "period" in events.columns and events["period"].isin([3, 4]).any() and home_team_start_left_extratime is None:
        raise ValueError(
            "metrica.convert_to_actions: events contain ET periods (period in {3, 4}) "
            "but home_team_start_left_extratime was not provided. Set it explicitly to "
            "match metadata, or filter ET events out before calling."
        )

    return _direction.home_attacks_right_per_period(
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
    )


def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: str,
    *,
    home_team_start_left: bool | None = None,
    home_team_start_left_extratime: bool | None = None,
    home_attacks_right_per_period: Mapping[int, bool] | None = None,
    preserve_native: list[str] | None = None,
    goalkeeper_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, ConversionReport]:
    """Convert normalized Metrica event DataFrame to SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        Normalized Metrica event data.
    home_team_id : str
        Home-team identifier (used to route per-period coord flips).
    home_team_start_left : bool or None, default ``None``
        From Metrica ``metadata.xml`` (Period entity team-direction attribute).
        When True, the home team starts on the left side in period 1.
        silly-kicks 3.0.1 introduced this kwarg to fix the per-period-absolute
        direction-of-play bug; pass exactly one of ``home_team_start_left`` or
        ``home_attacks_right_per_period``.
    home_team_start_left_extratime : bool or None, default ``None``
        Same flag for ET periods 3/4. Required only when the input contains
        period 3 or 4 rows.
    home_attacks_right_per_period : Mapping[int, bool] or None, default ``None``
        Escape hatch: pass the per-period flip mapping directly (e.g.
        ``{1: True, 2: False}``). Pass exactly one of ``home_team_start_left``
        or ``home_attacks_right_per_period``.
    preserve_native : list[str] or None, default ``None``
        Caller-attached columns to preserve through to the output.
    goalkeeper_ids : set[str] or None, default ``None``
        Optional set of player_ids known to be goalkeepers in this match.
        Metrica's source format does not natively mark GK actions in any
        event subtype; without this parameter the output contains zero
        ``keeper_*`` actions (preserved as 1.9.0 default behaviour, no
        breaking change). When provided, applies conservative routing:

        - ``PASS`` (any subtype) by GK → synth ``keeper_pick_up + pass``
          (GK had ball, then distributed).
        - ``RECOVERY`` (any subtype) by GK → ``keeper_pick_up``.
        - ``CHALLENGE`` with subtype ``"AERIAL-WON"`` by GK → ``keeper_claim``.

        Other event types and other CHALLENGE subtypes (including
        ``"AERIAL-LOST"``) are unchanged. An empty set is equivalent to
        ``None``. Pass goalkeeper_ids from match metadata (squad records
        / dim_players join) to enable GK coverage on Metrica data.

    Examples
    --------
    Convert a Metrica bronze events DataFrame to SPADL -- preferred path::

        from silly_kicks.spadl import metrica

        actions, report = metrica.convert_to_actions(
            events,
            home_team_id="Home",
            home_team_start_left=True,    # from Metrica metadata.xml
            goalkeeper_ids={"player_42", "player_99"},  # required for GK coverage
        )
        # Metrica has no native GK markers -- pass goalkeeper_ids to recover
        # keeper_pick_up / keeper_claim / synthesized GK distribution actions.

    Same conversion via the explicit-mapping escape hatch::

        actions, report = metrica.convert_to_actions(
            events,
            home_team_id="Home",
            home_attacks_right_per_period={1: True, 2: False},
        )
    """
    _validate_input_columns(events, EXPECTED_INPUT_COLUMNS, provider="Metrica")
    _validate_preserve_native(events, preserve_native, provider="Metrica", schema=KLOPPY_SPADL_COLUMNS)

    event_type_counts = Counter(events["type"])

    # PR-S23 / silly-kicks 3.0.1: native Metrica bronze events ship
    # PER_PERIOD_ABSOLUTE (teams switch ends after halftime; verified
    # empirically via lakehouse SK3-MIG probe). See ADR-006 erratum.
    flips = _resolve_per_period_flips_metrica(
        events=events,
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
        home_attacks_right_per_period=home_attacks_right_per_period,
    )

    # Convention sanity check: declared PER_PERIOD_ABSOLUTE; validator
    # surfaces upstream loader regressions. Strict mode under
    # SILLY_KICKS_ASSERT_INVARIANTS=1.
    validate_input_convention(
        events.assign(_sk_is_shot=(events["type"] == "SHOT")),
        declared=PER_PERIOD_ABSOLUTE,
        match_col="match_id",
        team_col="team",
        period_col="period",
        is_shot_col="_sk_is_shot",
        x_max=spadlconfig.field_length,
    )

    raw_actions = _build_raw_actions(events, preserve_native, goalkeeper_ids=goalkeeper_ids)

    if len(raw_actions) > 0:
        actions = _fix_clearances(raw_actions)
        actions = to_spadl_ltr(
            actions,
            input_convention=PER_PERIOD_ABSOLUTE,
            home_team_id=home_team_id,
            home_attacks_right_per_period=flips,
        )
        actions["action_id"] = range(len(actions))
        actions = _add_dribbles(actions)

        actions["start_x"] = actions["start_x"].clip(0, spadlconfig.field_length)
        actions["start_y"] = actions["start_y"].clip(0, spadlconfig.field_width)
        actions["end_x"] = actions["end_x"].clip(0, spadlconfig.field_length)
        actions["end_y"] = actions["end_y"].clip(0, spadlconfig.field_width)
    else:
        actions = raw_actions

    extras = list(preserve_native) if preserve_native else None
    actions = _finalize_output(actions, schema=KLOPPY_SPADL_COLUMNS, extra_columns=extras)

    mapped_counts: dict[str, int] = {}
    excluded_counts: dict[str, int] = {}
    unrecognized_counts: dict[str, int] = {}
    for et, count in event_type_counts.items():
        label = str(et)
        if et in _MAPPED_TYPES:
            mapped_counts[label] = count
        elif et in _EXCLUDED_TYPES:
            excluded_counts[label] = count
        else:
            unrecognized_counts[label] = count
    if unrecognized_counts:
        warnings.warn(
            f"Metrica: {sum(unrecognized_counts.values())} unrecognized types dropped: {dict(unrecognized_counts)}",
            stacklevel=2,
        )
    report = ConversionReport(
        provider="Metrica",
        total_events=sum(event_type_counts.values()),
        total_actions=len(actions),
        mapped_counts=mapped_counts,
        excluded_counts=excluded_counts,
        unrecognized_counts=unrecognized_counts,
    )
    return actions, report


def _empty_raw_actions(preserve_native: list[str] | None, events: pd.DataFrame) -> pd.DataFrame:
    cols = {col: pd.Series(dtype=dtype) for col, dtype in KLOPPY_SPADL_COLUMNS.items()}
    if preserve_native:
        for col in preserve_native:
            cols[col] = pd.Series(dtype=events[col].dtype if col in events.columns else "object")
    return pd.DataFrame(cols)


def _apply_card_pairs(actions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Upgrade FAULT-derived foul rows' result_id when paired with a CARD row (≤ 3s, same player)."""
    if "type" not in events.columns:
        return actions
    cards = events[events["type"] == "CARD"]
    if cards.empty:
        return actions

    foul_id = spadlconfig.actiontype_id["foul"]
    foul_actions = actions[actions["type_id"] == foul_id]
    for idx, foul_row in foul_actions.iterrows():
        candidates = cards[
            (cards["period"] == foul_row["period_id"])
            & (cards["player"] == foul_row["player_id"])
            & (cards["start_time_s"] >= foul_row["time_seconds"])
            & (cards["start_time_s"] <= foul_row["time_seconds"] + 3.0)
        ]
        if candidates.empty:
            continue
        sub = str(candidates.iloc[0]["subtype"]).upper()
        if "RED" in sub:
            actions.loc[idx, "result_id"] = spadlconfig.result_id["red_card"]
        elif "YELLOW" in sub:
            actions.loc[idx, "result_id"] = spadlconfig.result_id["yellow_card"]
    return actions


def _build_raw_actions(
    events: pd.DataFrame,
    preserve_native: list[str] | None,
    *,
    goalkeeper_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Build raw SPADL actions from Metrica event rows.

    Implements the §6.3 dispatch + §6.4 set-piece-then-shot composition rule.
    Direction-of-play flipping happens later in ``convert_to_actions``;
    this function emits source-orientation coords.
    """
    mask = events["type"].isin(_MAPPED_TYPES)
    rows = events.loc[mask].copy().reset_index(drop=True)
    n = len(rows)

    if n == 0:
        return _empty_raw_actions(preserve_native, events)

    type_ids = np.zeros(n, dtype=np.int64)
    result_ids = np.full(n, spadlconfig.result_id["success"], dtype=np.int64)
    bodypart_ids = np.full(n, spadlconfig.bodypart_id["foot"], dtype=np.int64)

    typ = rows["type"].to_numpy()
    sub_raw = rows["subtype"].fillna("").astype(str).str.upper().to_numpy()

    # Goalkeeper routing (Bug 3 fix, 1.10.0). Metrica has no native GK
    # markers; this is the only mechanism that surfaces GK actions.
    is_gk_player = np.zeros(n, dtype=bool)
    if goalkeeper_ids:
        gk_str_set = {str(g) for g in goalkeeper_ids}
        is_gk_player = rows["player"].astype(str).isin(gk_str_set).to_numpy()

    # --- PASS ---
    is_pass = typ == "PASS"
    is_pass_cross = is_pass & (sub_raw == "CROSS")
    is_pass_goalkick = is_pass & (sub_raw == "GOAL KICK")
    is_pass_head = is_pass & (sub_raw == "HEAD")
    is_pass_default = is_pass & ~is_pass_cross & ~is_pass_goalkick
    type_ids[is_pass_default] = spadlconfig.actiontype_id["pass"]
    type_ids[is_pass_cross] = spadlconfig.actiontype_id["cross"]
    type_ids[is_pass_goalkick] = spadlconfig.actiontype_id["goalkick"]
    bodypart_ids[is_pass_head] = spadlconfig.bodypart_id["head"]

    # GK distribution (Bug 3 fix): PASS by GK → keeper_pick_up; the
    # synthesized pass second-action is added below in the synthesis block.
    # Force end coords to match start coords for the keeper_pick_up so
    # _add_dribbles doesn't inject a spurious dribble between the pickup
    # and the synthetic pass that follows (the pickup is stationary —
    # the GK gathered the ball at start_x/start_y and held it before
    # distributing).
    is_pass_gk_distribution = is_pass & is_gk_player
    type_ids[is_pass_gk_distribution] = spadlconfig.actiontype_id["keeper_pick_up"]
    bodypart_ids[is_pass_gk_distribution] = spadlconfig.bodypart_id["other"]

    # --- RECOVERY -> interception (or keeper_pick_up if by known GK) ---
    is_recovery = typ == "RECOVERY"
    is_recovery_gk = is_recovery & is_gk_player
    type_ids[is_recovery & ~is_gk_player] = spadlconfig.actiontype_id["interception"]
    type_ids[is_recovery_gk] = spadlconfig.actiontype_id["keeper_pick_up"]
    bodypart_ids[is_recovery_gk] = spadlconfig.bodypart_id["other"]

    # --- CHALLENGE: WON -> tackle; AERIAL-WON-by-GK -> keeper_claim;
    # other LOST/AERIAL-LOST/etc -> drop ---
    is_challenge = typ == "CHALLENGE"
    is_challenge_won = is_challenge & (sub_raw == "WON")
    is_challenge_aerial_won = is_challenge & (sub_raw == "AERIAL-WON")
    is_challenge_aerial_won_gk = is_challenge_aerial_won & is_gk_player
    type_ids[is_challenge_won] = spadlconfig.actiontype_id["tackle"]
    type_ids[is_challenge_aerial_won_gk] = spadlconfig.actiontype_id["keeper_claim"]
    bodypart_ids[is_challenge_aerial_won_gk] = spadlconfig.bodypart_id["other"]
    is_challenge_dropped = is_challenge & ~is_challenge_won & ~is_challenge_aerial_won_gk
    type_ids[is_challenge_dropped] = spadlconfig.actiontype_id["non_action"]

    # --- BALL LOST -> bad_touch fail ---
    is_ball_lost = typ == "BALL LOST"
    type_ids[is_ball_lost] = spadlconfig.actiontype_id["bad_touch"]
    result_ids[is_ball_lost] = spadlconfig.result_id["fail"]

    # --- FAULT -> foul fail (CARD pairing applied below) ---
    is_fault = typ == "FAULT"
    type_ids[is_fault] = spadlconfig.actiontype_id["foul"]
    result_ids[is_fault] = spadlconfig.result_id["fail"]

    # --- SET PIECE dispatch on subtype ---
    is_setpiece = typ == "SET PIECE"
    is_sp_freekick = is_setpiece & (sub_raw == "FREE KICK")
    is_sp_corner = is_setpiece & (sub_raw == "CORNER KICK")
    is_sp_throwin = is_setpiece & (sub_raw == "THROW IN")
    is_sp_goalkick = is_setpiece & (sub_raw == "GOAL KICK")
    is_sp_kickoff = is_setpiece & (sub_raw == "KICK OFF")
    type_ids[is_sp_freekick] = spadlconfig.actiontype_id["freekick_short"]
    type_ids[is_sp_corner] = spadlconfig.actiontype_id["corner_short"]
    type_ids[is_sp_throwin] = spadlconfig.actiontype_id["throw_in"]
    bodypart_ids[is_sp_throwin] = spadlconfig.bodypart_id["other"]
    type_ids[is_sp_goalkick] = spadlconfig.actiontype_id["goalkick"]
    type_ids[is_sp_kickoff] = spadlconfig.actiontype_id["non_action"]

    # --- SHOT (with set-piece composition) ---
    is_shot = typ == "SHOT"
    type_ids[is_shot] = spadlconfig.actiontype_id["shot"]
    is_goal = is_shot & (sub_raw == "GOAL")
    result_ids[is_shot] = spadlconfig.result_id["fail"]
    result_ids[is_goal] = spadlconfig.result_id["success"]

    # §6.4 composition: SET PIECE FREE KICK + SHOT (≤ 5s, same player, same period)
    # → upgrade SHOT to shot_freekick AND drop the SET PIECE row.
    drop_mask = np.zeros(n, dtype=bool)
    if is_sp_freekick.any() and is_shot.any():
        for i in np.where(is_sp_freekick)[0]:
            if i + 1 >= n:
                continue
            cur_row = rows.iloc[i]
            next_row = rows.iloc[i + 1]
            if (
                typ[i + 1] == "SHOT"
                and next_row["period"] == cur_row["period"]
                and next_row["player"] == cur_row["player"]
                and (next_row["start_time_s"] - cur_row["start_time_s"]) <= 5.0
            ):
                type_ids[i + 1] = spadlconfig.actiontype_id["shot_freekick"]
                drop_mask[i] = True

    # Stationary keeper_pick_up: for PASS-by-GK rows, override end coords
    # to match start coords so _add_dribbles does not inject a spurious
    # dribble between the pickup and the synthetic pass that follows
    # (the pickup is semantically stationary — the GK gathered the ball
    # at start_x / start_y and held it before distributing). Apply the
    # override at the numpy-array level BEFORE building the DataFrame to
    # keep the column types narrow for pandas-stubs.
    start_x_arr = rows["start_x"].astype(np.float64).to_numpy()
    start_y_arr = rows["start_y"].astype(np.float64).to_numpy()
    end_x_arr = rows["end_x"].astype(np.float64).to_numpy()
    end_y_arr = rows["end_y"].astype(np.float64).to_numpy()
    if is_pass_gk_distribution.any():
        end_x_arr = np.where(is_pass_gk_distribution, start_x_arr, end_x_arr)
        end_y_arr = np.where(is_pass_gk_distribution, start_y_arr, end_y_arr)

    actions = pd.DataFrame(
        {
            "game_id": rows["match_id"].astype("object"),
            "original_event_id": rows["event_id"].astype("object"),
            "period_id": rows["period"].astype(np.int64),
            "time_seconds": rows["start_time_s"].astype(np.float64),
            "team_id": rows["team"].astype("object"),
            "player_id": rows["player"].astype("object"),
            "start_x": start_x_arr,
            "start_y": start_y_arr,
            "end_x": end_x_arr,
            "end_y": end_y_arr,
            "type_id": type_ids,
            "result_id": result_ids,
            "bodypart_id": bodypart_ids,
        }
    )

    # Carry preserve_native onto main actions BEFORE drop / synthesis
    # interleave so the indices stay aligned.
    if preserve_native:
        for col in preserve_native:
            if col in rows.columns:
                actions[col] = rows[col].to_numpy()
            else:
                actions[col] = np.nan

    # Tag main rows with _row_order = source_index*2 (so synth rows can
    # interleave at source_index*2 + 1 immediately after).
    actions["_row_order"] = np.arange(len(actions), dtype=np.int64) * 2

    # Apply drop_mask (set-piece-then-shot composition: FREE KICK rows
    # composed into a following SHOT). drop_mask only fires on SET PIECE
    # rows — never on PASS-by-GK rows — so removing them BEFORE synthesis
    # cannot orphan synthetic rows.
    actions = actions.loc[~drop_mask].reset_index(drop=True)

    # Synthesize keeper-distribution pass rows (Bug 3 fix). PASS-by-GK
    # rows already have type_id = keeper_pick_up; the synth DataFrame
    # adds a second "pass" action immediately after.
    synth_mask = is_pass & is_gk_player & ~drop_mask
    synth = _synthesize_metrica_gk_pass(rows, synth_mask, preserve_native)
    if len(synth) > 0:
        combined = pd.concat([actions, synth], ignore_index=True, sort=False)
        actions = (
            combined.sort_values("_row_order", kind="mergesort").drop(columns=["_row_order"]).reset_index(drop=True)
        )
    else:
        actions = actions.drop(columns=["_row_order"])

    actions = actions[actions["type_id"] != spadlconfig.actiontype_id["non_action"]].reset_index(drop=True)

    # If post-filter all rows dropped, return canonical empty schema so downstream
    # _finalize_output can assemble action_id and other schema-required columns.
    if len(actions) == 0:
        return _empty_raw_actions(preserve_native, events)

    actions = _apply_card_pairs(actions, events)

    return actions


def _synthesize_metrica_gk_pass(
    rows: pd.DataFrame,
    is_pass_gk_distribution: np.ndarray,
    preserve_native: list[str] | None,
) -> pd.DataFrame:
    """Build synthetic pass rows for PASS-by-GK events (Bug 3 fix, 1.10.0).

    Each source row matching ``is_pass_gk_distribution`` already had its
    type_id set to keeper_pick_up by the caller. This helper emits the
    second half (a synthesized SPADL pass with bodypart=other) immediately
    after, mirroring the sportec throwOut shape.

    Returned DataFrame includes a ``_row_order`` column for interleaving.
    Empty DataFrame returned when no qualifying rows exist.
    """
    if not is_pass_gk_distribution.any():
        return pd.DataFrame()

    src_indices = np.where(is_pass_gk_distribution)[0]
    src = rows.iloc[src_indices].copy().reset_index(drop=True)
    n_synth = len(src)

    synth = pd.DataFrame(
        {
            "game_id": src["match_id"].astype("object"),
            "original_event_id": (src["event_id"].astype(str) + "_synth_pass").astype("object"),
            "period_id": src["period"].astype(np.int64),
            "time_seconds": src["start_time_s"].astype(np.float64),
            "team_id": src["team"].astype("object"),
            "player_id": src["player"].astype("object"),
            "start_x": src["start_x"].astype(np.float64),
            "start_y": src["start_y"].astype(np.float64),
            "end_x": src["end_x"].astype(np.float64),
            "end_y": src["end_y"].astype(np.float64),
            "type_id": np.full(n_synth, spadlconfig.actiontype_id["pass"], dtype=np.int64),
            "result_id": np.full(n_synth, spadlconfig.result_id["success"], dtype=np.int64),
            "bodypart_id": np.full(n_synth, spadlconfig.bodypart_id["other"], dtype=np.int64),
        }
    )
    synth["_row_order"] = src_indices.astype(np.int64) * 2 + 1

    if preserve_native:
        for col in preserve_native:
            if col in src.columns:
                synth[col] = src[col].to_numpy()
            else:
                synth[col] = np.nan

    return synth
