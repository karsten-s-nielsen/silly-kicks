"""DFL / Sportec parse+shape port --- bytes -> bronze -> silly-kicks converter input.

ADR-031 (T3). This module is **upstreamed verbatim** from the luxury-lakehouse
DFL/Sportec parser, pinned at commit ``0efac60`` (see
``tests/datasets/sportec/idsse_slice/SOURCE_SHA``). It single-sources the IDSSE/Sportec
DFL parse layer so the silly-kicks dev harness (``scripts/_loader_pining.py``) and the
lakehouse stop maintaining two divergent parsers. See
``docs/superpowers/specs/2026-06-16-dfl-parse-port-design.md``.

Provenance of the lifted bodies (luxury-lakehouse @ 0efac60):

* Parse layer --- ``src/ingestion/idsse.py``: ``_SECTION_TO_PERIOD``, ``_FRAME_RATE``,
  ``_to_snake_case`` + the event prefix maps, ``_parse_teams``, ``_MatchMetadata`` +
  ``_parse_match_metadata``, ``_parse_float_or_none`` / ``_parse_bool_or_none``,
  ``_parse_positions_xml``, ``_IDSSE_TRACKING_BRONZE_COLS`` (+ dtype overrides),
  ``_build_event_row``, ``_scan_kickoff_times`` / ``_derive_period_from_kickoffs``,
  ``_parse_events_xml``.
* ``idsse_native_match_id`` --- ``src/shared/identifiers.py``.
* ``finalize_bronze_df`` --- ``src/ingestion/utils.py``.
* ``_IDSSE_EVENTS_BRONZE_COLS`` --- **materialised** from the lakehouse's
  ``_compute_idsse_events_bronze_cols()`` (which derives it from
  ``ingestion._dfl_event_schema``) rather than lifting that derivation; pinned as a
  literal here. Re-pin via the SOURCE_SHA file.
* Shape layer --- tracking ``_bronze_idsse_to_sportec_input``
  (``src/analytics/action_context/convert.py``); events
  ``adapt_idsse_events_for_silly_kicks`` + ``derive_idsse_home_team_start_left`` /
  ``...extratime`` (``src/ingestion/spadl_adapter.py``).

The bodies are lifted byte-for-byte; the ONLY adaptations are (a) the lakehouse
``logger`` argument defaults to this module's logger, (b) the two cross-module helpers
above are inlined, (c) the events bronze-column set is materialised, and (d) the
silly-kicks-local hardening helpers marked ``LOCAL HARDENING`` below
(:func:`_normalize_ball_state` / :func:`_resolve_period_column`), which close two
*silent-degradation* paths in the lifted bodies. Those helpers are **output-neutral on
valid input** --- they only add loudness where the lift previously degraded quietly --- so
the golden parity gate stays the authority on the lift itself. A future re-pin to a newer
lakehouse SHA must re-apply them, not drop them. The parse layer
is faithful ``bytes -> RAW bronze``; **data-quality (Savitzky-Golay smoothing, velocity
derivation) stays consumer-side** --- the lakehouse applies its own ``_smooth_tracking``
after this parse, and the silly-kicks harness applies ``_preprocess``. The Phase-2.1
golden parity test (``tests/providers/sportec/test_parse_port_parity.py``) guards the lift.
"""

from __future__ import annotations

import logging
import math
import re
import warnings
import xml.etree.ElementTree as ET  # nosemgrep: use-defused-xml -- trusted local DFL XML files, not untrusted input
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifted helper: idsse_native_match_id (luxury-lakehouse src/shared/identifiers.py)
# ---------------------------------------------------------------------------
_IDSSE_MATCH_ID_PATTERN = re.compile(r"^[A-Z0-9]+$")


def idsse_native_match_id(raw_dfl_match_id: str) -> str:
    """Canonical IDSSE native match id --- bare DFL MatchId (e.g. ``J03WMX``).

    Lifted verbatim from luxury-lakehouse ``src/shared/identifiers.py`` @ 0efac60.
    """
    if not _IDSSE_MATCH_ID_PATTERN.match(raw_dfl_match_id):
        raise ValueError(f"invalid IDSSE match id: {raw_dfl_match_id!r} (expected bare DFL MatchId like 'J03WMX')")
    return raw_dfl_match_id


# ---------------------------------------------------------------------------
# Lifted helper: finalize_bronze_df (luxury-lakehouse src/ingestion/utils.py)
# ---------------------------------------------------------------------------
def finalize_bronze_df(
    df: pd.DataFrame,
    expected_cols: Iterable[str],
    dtype_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Add any missing ``expected_cols`` as typed all-NA columns + cast all-null object
    columns to an explicit nullable dtype.

    Lifted verbatim from luxury-lakehouse ``src/ingestion/utils.py`` @ 0efac60 (the
    pandas->Arrow->Spark NullType-drop guard; pure pandas, no spark).
    """
    overrides = dtype_overrides or {}
    n_rows = len(df)

    for col in expected_cols:
        if col not in df.columns:
            target = overrides.get(col, "string")
            df[col] = pd.array([None] * n_rows, dtype=target)  # type: ignore[call-overload]

    for col in list(df.columns):
        if df[col].dtype == object and df[col].isna().all():
            target = overrides.get(col, "string")
            df[col] = df[col].astype(target)  # type: ignore[call-overload]

    return df


# --- lifted: idsse.py L135-141 (luxury-lakehouse @ 0efac60) ---
_SECTION_TO_PERIOD: dict[str, int] = {
    "firstHalf": 1,
    "secondHalf": 2,
    "extraTimeFirstHalf": 3,
    "extraTimeSecondHalf": 4,
    "penaltyShootout": 5,
}

# --- lifted: idsse.py L144-145 (luxury-lakehouse @ 0efac60) ---
# Frame rate for all IDSSE matches (DFL position data is 25fps)
_FRAME_RATE = 25

# --- lifted: idsse.py L150-156 (luxury-lakehouse @ 0efac60) ---
# Player attribute lookup order per event child tag.
# For most event types, the primary actor is in the ``Player`` attribute.
# TacklingGame uses ``Winner`` as the primary actor.
_PLAYER_ATTR_ORDER: dict[str, list[str]] = {
    "TacklingGame": ["Winner", "Player"],
}
_DEFAULT_PLAYER_ATTRS: list[str] = ["Player"]

# --- lifted: idsse.py L172-185 (luxury-lakehouse @ 0efac60) ---
# Pre-compiled regex for splitting CamelCase / PascalCase at word boundaries.
# Matches (lower→Upper) and (Upper→Upper-before-lower). Shared conceptually
# with src/tests/coverage_utils.to_snake_case — the mirror lives there for
# the coverage-test infrastructure.
_ATTR_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _to_snake_case(name: str) -> str:
    """Normalise a DFL XML attribute name to a bronze column suffix.

    Handles CamelCase (``PlayAngle`` → ``play_angle``) and hyphenated
    (``X-Position`` → ``x_position``) attribute names consistently.
    """
    return _ATTR_CAMEL_BOUNDARY.sub("_", name.replace("-", "_")).lower()


# --- lifted: idsse.py L188-213 (luxury-lakehouse @ 0efac60) ---
# Raw DFL <Event>-level XML attribute names → bronze column names.
# Thirteen DFL attributes map to thirteen bronze columns. ``match_id`` on the
# row (derived: ``idsse_{match_id}``) is DISTINCT from ``match_id_raw`` (the
# DFL-MAT-* identifier captured here) — both land in bronze per
# bronze-completeness.
_EVENT_LEVEL_ATTR_MAP: dict[str, str] = {
    "MatchId": "match_id_raw",
    "EventId": "event_id",
    "EventTime": "event_time",
    "StartFrame": "start_frame",
    "EndFrame": "end_frame",
    "CalculatedFrame": "calculated_frame",
    "CalculatedTimestamp": "calculated_timestamp",
    "X-Position": "x",
    "Y-Position": "y",
    "X-Source-Position": "x_source_position",
    "Y-Source-Position": "y_source_position",
    "X-PositionFromTracking": "x_position_from_tracking",
    "Y-PositionFromTracking": "y_position_from_tracking",
}

# Event-level bronze cols that must be cast float / int (vs. pass-through string).
_EVENT_LEVEL_FLOAT_COLS: frozenset[str] = frozenset(
    {"x", "y", "x_source_position", "y_source_position", "x_position_from_tracking", "y_position_from_tracking"},
)
_EVENT_LEVEL_INT_COLS: frozenset[str] = frozenset({"start_frame", "end_frame", "calculated_frame"})

# --- lifted: idsse.py L215-251 (luxury-lakehouse @ 0efac60) ---
# First-child tag → bronze column prefix. Every attribute on a first-child
# element lands as ``{prefix}_{to_snake_case(attr)}``. Prefixes are short,
# readable, and distinct across all first-child types.
_EVENT_TYPE_PREFIX: dict[str, str] = {
    "BallClaiming": "claim",
    "BallDeflection": "deflection",
    "Caution": "caution",
    "CautionTeamofficial": "caution_official",
    "ChanceWithoutShot": "chance",
    "CornerKick": "corner",
    "Delete": "delete",
    "FairPlay": "fairplay",
    "FinalWhistle": "whistle",
    "Foul": "foul",
    "FreeKick": "freekick",
    "GoalDisallowed": "goaldis",
    "GoalKick": "goalkick",
    "KickOff": "kickoff",
    "Nutmeg": "nutmeg",
    "Offside": "offside",
    "OtherBallAction": "otherball",
    "OtherPlayerAction": "other_action",
    "Penalty": "penalty",
    "PenaltyNotAwarded": "penalty_not",
    "Play": "play",
    "PlayerNotSentOff": "not_sent_off",
    "PossessionLossBeforeGoal": "possloss",
    "RefereeBall": "refball",
    "Run": "run",
    "ShotAtGoal": "shot",
    "SitterPrevented": "sitter_prev",
    "SpectacularPlay": "spectacular",
    "Substitution": "sub",
    "TacklingGame": "tackle",
    "ThrowIn": "throwin",
    "VideoAssistantAction": "var",
}

# --- lifted: idsse.py L253-281 (luxury-lakehouse @ 0efac60) ---
# Nested tag → bronze column prefix. Nested children that reuse a top-level
# event type keep the same prefix (a Play nested inside KickOff writes to
# ``play_*`` — same as a standalone Play). Shot-outcome variants share the
# ``shot_outcome_*`` prefix and use ``shot_outcome_type`` to disambiguate.
_NESTED_PREFIX_MAP: dict[str, str] = {
    "Pass": "pass",
    "Cross": "cross",
    "Play": "play",
    "ShotAtGoal": "shot",
    "FairPlay": "fairplay",
    "FaultExecution": "fault_execution",
    "SuccessfulShot": "shot_outcome",
    "SavedShot": "shot_outcome",
    "ShotWide": "shot_outcome",
    "ShotWoodWork": "shot_outcome",
    "BlockedShot": "shot_outcome",
    "OtherShot": "shot_outcome",
}

# Shot-outcome nested tag name → disambiguator value emitted on
# ``shot_outcome_type`` when a ShotAtGoal event has one of these nested.
_SHOT_OUTCOME_NAMES: dict[str, str] = {
    "SuccessfulShot": "successful",
    "SavedShot": "saved",
    "ShotWide": "wide",
    "ShotWoodWork": "woodwork",
    "BlockedShot": "blocked",
    "OtherShot": "other",
}


# --- lifted: idsse.py L437-482 (luxury-lakehouse @ 0efac60) ---
def _parse_teams(info_path: str) -> tuple[str, str, dict[str, str], set[str]]:
    """Parse match info XML to get home/away team IDs, player-to-team mapping, and GK IDs.

    Args:
        info_path: Path to match info XML file.

    Returns:
        Tuple of (home_team_id, away_team_id, {person_id: "home"|"away"}, gk_player_ids).
        ``gk_player_ids`` contains PersonIds of players with ``PlayingPosition="TW"``
        (DFL standard for Torwart/goalkeeper).

    Note:
        The per-row DFL ``TeamId`` that lands in bronze is NOT sourced from
        this mapping — it is taken directly from the enclosing FrameSet's
        ``TeamId`` attribute during position parsing, which is always
        available and avoids an extra map plumb-through.
    """
    tree = ET.parse(info_path)  # noqa: S314  # nosemgrep: use-defused-xml-parse
    root = tree.getroot()

    home_team_id = ""
    away_team_id = ""
    player_team_map: dict[str, str] = {}
    gk_player_ids: set[str] = set()

    for team_el in root.iter("Team"):
        team_id = team_el.get("TeamId", "")
        role = team_el.get("Role", "")

        if role == "home":
            home_team_id = team_id
            team_label = "home"
        elif role == "guest":
            away_team_id = team_id
            team_label = "away"
        else:
            continue

        for player_el in team_el.iter("Player"):
            person_id = player_el.get("PersonId", "")
            if person_id:
                player_team_map[person_id] = team_label
                if player_el.get("PlayingPosition") == "TW":
                    gk_player_ids.add(person_id)

    return home_team_id, away_team_id, player_team_map, gk_player_ids


# --- lifted: idsse.py L485-581 (luxury-lakehouse @ 0efac60) ---
@dataclass(frozen=True)
class _MatchMetadata:
    """Match-level metadata sourced from the DFL ``<General>`` and
    ``<Environment>`` elements in the matchinformation XML.

    PR-LL2 Path B: previously the SPADL/VAEP pipeline had no way to access
    the per-match competition / season / pitch dimensions because none of
    these landed in ``bronze.idsse_events``. This dataclass + parser
    surface them so the bronze writer can populate the LL2-added columns
    (``competition_native_id``, ``season_native_id``,
    ``home_team_id_native``, ``away_team_id_native``).
    """

    competition_id: str
    """DFL CompetitionId, e.g. ``DFL-COM-000001``. Format ``DFL-COM-XXXXXX``."""

    season_id: str
    """DFL SeasonId, e.g. ``DFL-SEA-0001K6``. Format ``DFL-SEA-XXXXXX``."""

    home_team_id: str
    """DFL HomeTeamId, e.g. ``DFL-CLU-000008``. Format ``DFL-CLU-XXXXXX``."""

    away_team_id: str
    """DFL GuestTeamId (DFL spec calls the away team "guest")."""

    pitch_x: float | None
    """Pitch length in meters (e.g. 105.0). NULL if absent from XML."""

    pitch_y: float | None
    """Pitch width in meters (e.g. 68.0). NULL if absent from XML."""


def _parse_match_metadata(info_path: str) -> _MatchMetadata:
    """Parse the ``<General>`` + ``<Environment>`` elements of a DFL
    matchinformation XML, returning competition / season / pitch metadata.

    DFL spec, ``DFL_02_01_matchinformation_*.xml`` shape::

        <PutDataRequest>
          <MatchInformation>
            <General CompetitionId="DFL-COM-000001"
                     SeasonId="DFL-SEA-0001K6"
                     HomeTeamId="DFL-CLU-000008"
                     GuestTeamId="DFL-CLU-00000G"
                     ... />
            <Environment PitchX="105.00" PitchY="68.00" ... />
            <Teams>...</Teams>
            ...

    Args:
        info_path: Filesystem path (or UC Volume path) to the matchinformation XML.

    Returns:
        Populated ``_MatchMetadata``. Empty strings for any missing IDs
        (callers tolerate empty home_team_id today; we surface that as
        empty string rather than raising so a malformed XML doesn't kill
        the whole batch).
    """
    tree = ET.parse(info_path)  # noqa: S314  # nosemgrep: use-defused-xml-parse
    root = tree.getroot()

    general = root.find(".//General")
    environment = root.find(".//Environment")

    competition_id = general.get("CompetitionId", "") if general is not None else ""
    season_id = general.get("SeasonId", "") if general is not None else ""
    home_team_id = general.get("HomeTeamId", "") if general is not None else ""
    away_team_id = general.get("GuestTeamId", "") if general is not None else ""

    pitch_x: float | None = None
    pitch_y: float | None = None
    if environment is not None:
        pitch_x = _parse_float_or_none(environment.get("PitchX", ""))
        pitch_y = _parse_float_or_none(environment.get("PitchY", ""))

    return _MatchMetadata(
        competition_id=competition_id,
        season_id=season_id,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
    )


# Sentinel used by ``_parse_events_xml`` / ``_build_event_row`` when callers
# don't have a real matchinformation XML available (e.g. unit tests with
# synthetic event XML, no companion matchinfo). Production ingestion always
# passes a populated metadata via ``_parse_match_metadata``.
_EMPTY_MATCH_METADATA: _MatchMetadata = _MatchMetadata(
    competition_id="",
    season_id="",
    home_team_id="",
    away_team_id="",
    pitch_x=None,
    pitch_y=None,
)


# --- lifted: idsse.py L595-617 (luxury-lakehouse @ 0efac60) ---
def _parse_float_or_none(raw: str) -> float | None:
    """Parse an XML attribute into a float, returning None on empty/NaN."""
    if not raw:
        return None
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return None
    if math.isnan(value):
        return None
    return round(value, 4)


def _parse_bool_or_none(raw: str) -> bool | None:
    """Parse ``"true"``/``"false"`` (case-insensitive) into a Python bool."""
    if not raw:
        return None
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


# --- lifted: idsse.py L620-839 (luxury-lakehouse @ 0efac60) ---
def _parse_positions_xml(
    pos_path: str,
    player_team_map: dict[str, str],
    match_id: str,
    logger: logging.Logger,
    gk_player_ids: set[str] | None = None,
    metadata: _MatchMetadata = _EMPTY_MATCH_METADATA,
) -> dict[int, list[dict[str, object]]]:
    """Parse DFL position XML into bronze-complete row dicts, split by period.

    Uses TWO-PASS iterative XML parsing.

    1. **First pass (ball-only)**: Scans ball FrameSets
       (``TeamId="BALL"``) and populates ``ball_by_frame`` keyed on
       ``(period, frame_n)``. Each ball entry captures EVERY DFL
       ``<Frame>`` attribute (X, Y, Z, S, A, D, M, T, BallPossession,
       BallStatus) so downstream joins see the full source schema.
       Also populates ``period_first_frame`` for period-relative timestamps.
    2. **Second pass (players-only)**: Emits one row per player per frame
       with every DFL ``<Frame>`` attribute (X, Y, S, A, D, M, T), the
       player's DFL ``TeamId`` (directly from the enclosing FrameSet),
       and a ``ball_*`` join from pass 1.

    **Bronze-completeness:** every attribute enumerated in
    ``src/tests/fixtures/idsse_dfl_tracking_attr_enumeration.json`` lands
    in a dedicated bronze column. Nothing is dropped silently.

    **Why two-pass (PR 1.7):** real DFL position XMLs have ball FrameSets
    AFTER all player / referee FrameSets in the file. A single-pass parser
    emitting player rows first would see an empty ball lookup dict and
    produce NULL ball coordinates in bronze.

    Returns rows grouped by period so callers can process and release each
    half independently, halving peak DataFrame memory.

    Args:
        pos_path: Path to position XML file.
        player_team_map: Mapping of PersonId to "home"/"away".
        match_id: Raw match identifier (without ``idsse_`` prefix).
        logger: Structured logger instance.
        gk_player_ids: Set of PersonIds identified as goalkeepers. When
            provided, each tracking row carries an ``is_goalkeeper`` bool.

    Returns:
        Mapping of period number → list of row dicts. Each row carries all
        DFL tracking attributes per the bronze-completeness contract.
    """
    rows_by_period: dict[int, list[dict[str, object]]] = {p: [] for p in _SECTION_TO_PERIOD.values()}
    # PR-LL2 Path B close-out (2026-04-29): bronze.idsse_events.match_id /
    # bronze.idsse_tracking.match_id now use the bare DFL MatchId (e.g.
    # 'J03WMX'). Pre-close-out the format was 'idsse_J03WMX' which
    # produced 100% NULL match_key on fct_action_values for IDSSE rows
    # because dim_matches strips the 'idsse_' prefix. ADR-018 + Bug #1.
    canonical_match_id = idsse_native_match_id(match_id)

    # Ball data per (period, frame_n). Each entry is a dict carrying every
    # DFL ball Frame attribute plus the ball-only ones. Populated in pass 1.
    ball_by_frame: dict[tuple[int, int], dict[str, object]] = {}
    ball_miss_count = 0  # Player frames where ball lookup returned None

    # First-seen frame number per period — used to compute period-relative
    # timestamps (see PR 1.6).
    period_first_frame: dict[int, int] = {}

    # ── PASS 1: ball FrameSets → populate ball_by_frame + period_first_frame ──
    for _event, elem in ET.iterparse(pos_path, events=("end",)):  # noqa: S314
        if elem.tag != "FrameSet":
            continue

        team_id_lower = elem.get("TeamId", "").lower()
        if team_id_lower != "ball":
            elem.clear()
            continue

        section = elem.get("GameSection", "")
        period = _SECTION_TO_PERIOD.get(section)
        if period is None:
            logger.warning(
                "Unrecognized GameSection %r in match %s — skipping FrameSet",
                section,
                match_id,
            )
            elem.clear()
            continue

        for frame_el in elem.iter("Frame"):
            n = int(frame_el.get("N", "0"))
            ball_entry: dict[str, object] = {
                "ball_x": _parse_float_or_none(frame_el.get("X", "")),
                "ball_y": _parse_float_or_none(frame_el.get("Y", "")),
                "ball_z": _parse_float_or_none(frame_el.get("Z", "")),
                "ball_s": _parse_float_or_none(frame_el.get("S", "")),
                "ball_a": _parse_float_or_none(frame_el.get("A", "")),
                "ball_d": _parse_float_or_none(frame_el.get("D", "")),
                "ball_m": _parse_bool_or_none(frame_el.get("M", "")),
                "ball_t": frame_el.get("T", "") or None,
                "ball_possession": frame_el.get("BallPossession", "") or None,
                "ball_status": frame_el.get("BallStatus", "") or None,
            }
            ball_by_frame[(period, n)] = ball_entry
            cur = period_first_frame.get(period)
            if cur is None or n < cur:
                period_first_frame[period] = n
        elem.clear()

    # ── PASS 2: player FrameSets → emit per-player tracking rows ──
    for _event, elem in ET.iterparse(pos_path, events=("end",)):  # noqa: S314
        if elem.tag != "FrameSet":
            continue

        team_id_lower = elem.get("TeamId", "").lower()
        # Pass 2 skips ball (already handled) and referee (not tracked).
        if team_id_lower in ("ball", "referee"):
            elem.clear()
            continue

        section = elem.get("GameSection", "")
        period = _SECTION_TO_PERIOD.get(section)
        if period is None:
            logger.warning(
                "Unrecognized GameSection %r in match %s — skipping FrameSet",
                section,
                match_id,
            )
            elem.clear()
            continue

        team_id = elem.get("TeamId", "") or None
        person_id = elem.get("PersonId", "")
        team_label = player_team_map.get(person_id, "unknown")
        period_rows = rows_by_period[period]

        for frame_el in elem.iter("Frame"):
            n = int(frame_el.get("N", "0"))
            x = _parse_float_or_none(frame_el.get("X", ""))
            y = _parse_float_or_none(frame_el.get("Y", ""))

            # Require at least X/Y to emit a player row — a player tracking
            # row without position is meaningless. Other attrs default to
            # None if missing (bronze-completeness tolerates sparse data).
            if x is None or y is None:
                continue

            cur = period_first_frame.get(period)
            if cur is None or n < cur:
                period_first_frame[period] = n
            period_start = period_first_frame[period]
            timestamp = (n - period_start) / _FRAME_RATE

            ball_lookup = ball_by_frame.get((period, n))
            if ball_lookup is None:
                ball_miss_count += 1
            ball_entry: dict[str, object] = (
                ball_lookup
                if ball_lookup is not None
                else {
                    "ball_x": None,
                    "ball_y": None,
                    "ball_z": None,
                    "ball_s": None,
                    "ball_a": None,
                    "ball_d": None,
                    "ball_m": None,
                    "ball_t": None,
                    "ball_possession": None,
                    "ball_status": None,
                }
            )

            row: dict[str, object] = {
                # Existing columns
                "period": period,
                "frame": n,
                "timestamp": round(timestamp, 4),
                "player_id": person_id,
                "team": team_label,
                "x": x,
                "y": y,
                "match_id": canonical_match_id,
                "frame_rate": _FRAME_RATE,
                # New per-player DFL Frame attrs
                "team_id": team_id,
                "t": frame_el.get("T", "") or None,
                "s": _parse_float_or_none(frame_el.get("S", "")),
                "a": _parse_float_or_none(frame_el.get("A", "")),
                "d": _parse_float_or_none(frame_el.get("D", "")),
                "m": _parse_bool_or_none(frame_el.get("M", "")),
                # Per-match metadata (sourced from <General> in matchinformation XML).
                # Same value for every row of a given match — replicated here for
                # parity with bronze.idsse_events (asserted by
                # test_idsse_match_metadata_parity.py). Empty string when the
                # XML lacked the attribute or the caller passed _EMPTY_MATCH_METADATA
                # (test path with no companion matchinfo).
                "competition_native_id": metadata.competition_id,
                "season_native_id": metadata.season_id,
                "home_team_id_native": metadata.home_team_id,
                "away_team_id_native": metadata.away_team_id,
                # Ball-joined cols
                **ball_entry,
            }
            if gk_player_ids is not None:
                row["is_goalkeeper"] = person_id in gk_player_ids
            period_rows.append(row)

        elem.clear()

    logger.info("Parsed %d ball frames for match %s", len(ball_by_frame), match_id)
    if ball_miss_count > 0 and len(ball_by_frame) > 0:
        total_player_frames = sum(len(rows) for rows in rows_by_period.values())
        miss_pct = 100.0 * ball_miss_count / max(total_player_frames, 1)
        log_fn = logger.warning if miss_pct > 5.0 else logger.info
        log_fn(
            "Ball coordinate lookup missed %d of %d player frames for match %s (%.1f%%)",
            ball_miss_count,
            total_player_frames,
            match_id,
            miss_pct,
        )

    return rows_by_period


# --- lifted: idsse.py L842-905 (luxury-lakehouse @ 0efac60) ---
# Bronze-completeness contract for bronze.idsse_tracking.
# Every column here is emitted by _parse_positions_xml; finalize_bronze_df
# guarantees it lands in Delta regardless of which Frame attrs happen to be
# populated for a given match's rows. Asserted by the coverage tests.
_IDSSE_TRACKING_BRONZE_COLS: tuple[str, ...] = (
    # Derived / join keys
    "period",
    "frame",
    "timestamp",
    "player_id",
    "team",
    "team_id",
    "match_id",
    "frame_rate",
    "is_goalkeeper",
    # DFL per-player Frame attrs
    "x",
    "y",
    "t",
    "s",
    "a",
    "d",
    "m",
    # Ball-joined DFL Frame attrs
    "ball_x",
    "ball_y",
    "ball_z",
    "ball_s",
    "ball_a",
    "ball_d",
    "ball_m",
    "ball_t",
    "ball_possession",
    "ball_status",
    # Per-match metadata sourced from matchinformation XML's <General> element.
    # Parity with bronze.idsse_events; see _IDSSE_MATCH_METADATA_BRONZE_COLS.
    "competition_native_id",
    "season_native_id",
    "home_team_id_native",
    "away_team_id_native",
)

# Nullable dtype overrides for tracking columns. Columns not in this map
# default to pd.StringDtype() via finalize_bronze_df.
_IDSSE_TRACKING_DTYPE_OVERRIDES: dict[str, str] = {
    "period": "Int64",
    "frame": "Int64",
    "timestamp": "Float64",
    "frame_rate": "Int64",
    "is_goalkeeper": "boolean",
    "x": "Float64",
    "y": "Float64",
    "s": "Float64",
    "a": "Float64",
    "d": "Float64",
    "m": "boolean",
    "ball_x": "Float64",
    "ball_y": "Float64",
    "ball_z": "Float64",
    "ball_s": "Float64",
    "ball_a": "Float64",
    "ball_d": "Float64",
    "ball_m": "boolean",
}


# --- lifted: idsse.py L1147-1268 (luxury-lakehouse @ 0efac60) ---
def _build_event_row(
    elem: ET.Element,
    first_child: ET.Element,
    event_type: str,
    canonical_match_id: str,
    current_period: int,
    player_team_map: dict[str, str],
    period_start_time: dict[int, datetime],
    metadata: _MatchMetadata = _EMPTY_MATCH_METADATA,
) -> dict[str, object]:
    """Build one bronze row from a DFL <Event> element + its first child.

    Per the bronze-completeness principle, this extracts EVERY XML attribute
    on the Event + first-child + nested-child elements into a prefixed
    bronze column. Type-casts event-level cols that downstream analysis
    treats as numeric (x/y coords, frame numbers). Returns the full row
    dict; callers need not pre-populate any columns.

    Args:
        ...
        metadata: Match-level metadata from ``_parse_match_metadata``.
            PR-LL2 Path B: lifts ``competition_native_id`` /
            ``season_native_id`` / ``home_team_id_native`` /
            ``away_team_id_native`` onto every row. ``team_id_native`` is
            derived per-row from the ``team`` label.
    """
    row: dict[str, object] = {
        "match_id": canonical_match_id,
        "event_type": event_type,
        "period": current_period,
        "player_id": "",
        "team": "unknown",
        # PR-LL2: match-level metadata (same value for every row of a match).
        "competition_native_id": metadata.competition_id,
        "season_native_id": metadata.season_id,
        "home_team_id_native": metadata.home_team_id,
        "away_team_id_native": metadata.away_team_id,
        # team_id_native filled below after `team` label resolves.
    }

    # --- Event-level attrs → bronze cols via EVENT_LEVEL_ATTR_MAP ---
    for dfl_attr, bronze_col in _EVENT_LEVEL_ATTR_MAP.items():
        raw_val = elem.get(dfl_attr)
        if raw_val is None or raw_val == "":
            row[bronze_col] = None
            continue
        if bronze_col in _EVENT_LEVEL_FLOAT_COLS:
            try:
                fv = float(raw_val)
            except (ValueError, TypeError):
                row[bronze_col] = None
                continue
            row[bronze_col] = round(fv, 4) if not math.isnan(fv) else None
        elif bronze_col in _EVENT_LEVEL_INT_COLS:
            try:
                row[bronze_col] = int(raw_val)
            except (ValueError, TypeError):
                row[bronze_col] = None
        else:
            row[bronze_col] = raw_val

    # --- timestamp_seconds: period-relative from EventTime ---
    event_time_str = elem.get("EventTime", "")
    timestamp_seconds: float | None = None
    if event_time_str:
        try:
            event_dt = datetime.fromisoformat(event_time_str)
            if event_dt.tzinfo is not None:
                event_dt = event_dt.astimezone(timezone.utc)
            if current_period not in period_start_time:
                period_start_time[current_period] = event_dt
            delta = event_dt - period_start_time[current_period]
            timestamp_seconds = round(delta.total_seconds(), 4)
        except (ValueError, TypeError):
            pass
    row["timestamp_seconds"] = timestamp_seconds

    # --- Primary player_id + team label (preserving KickOff nested-Play lookup) ---
    search_elem = first_child
    if event_type == "KickOff":
        for ko_child in first_child:
            if ko_child.tag == "Play":
                search_elem = ko_child
                break
    player_attr_names = _PLAYER_ATTR_ORDER.get(event_type, _DEFAULT_PLAYER_ATTRS)
    for attr_name in player_attr_names:
        pid = search_elem.get(attr_name, "")
        if pid:
            row["player_id"] = pid
            break
    pid_val = row["player_id"]
    if isinstance(pid_val, str) and pid_val:
        row["team"] = player_team_map.get(pid_val, "unknown")

    # PR-LL2: derive `team_id_native` (DFL CLU id of the acting team) from
    # the resolved home/away label. NULL when team is unknown.
    team_label = row["team"]
    if team_label == "home":
        row["team_id_native"] = metadata.home_team_id
    elif team_label == "away":
        row["team_id_native"] = metadata.away_team_id
    else:
        row["team_id_native"] = None

    # --- First-child attrs → {prefix}_{snake(attr)} bronze cols ---
    prefix = _EVENT_TYPE_PREFIX.get(event_type)
    if prefix is not None:
        for attr_name, attr_val in first_child.attrib.items():
            row[f"{prefix}_{_to_snake_case(attr_name)}"] = attr_val

    # --- Nested children attrs → {nested_prefix}_{snake(attr)} bronze cols ---
    for nested_child in first_child:
        nested_prefix = _NESTED_PREFIX_MAP.get(nested_child.tag)
        if nested_prefix is None:
            continue
        for attr_name, attr_val in nested_child.attrib.items():
            row[f"{nested_prefix}_{_to_snake_case(attr_name)}"] = attr_val
        # Disambiguator for ShotAtGoal's six mutually-exclusive outcome tags.
        if event_type == "ShotAtGoal" and nested_child.tag in _SHOT_OUTCOME_NAMES:
            row["shot_outcome_type"] = _SHOT_OUTCOME_NAMES[nested_child.tag]

    return row


# --- lifted: idsse.py L1271-1353 (luxury-lakehouse @ 0efac60) ---
def _scan_kickoff_times(event_path: str) -> dict[int, datetime]:
    """Pass 1 of the 2-pass DFL event parser (ADR-018 / Bug #6 fix).

    Scans ONLY KickOff events to build a ``{period: kickoff_event_time}`` map.
    Pass 2 uses this map to derive each event's period by comparing its
    EventTime to kickoff times — NOT by relying on XML stream-order
    ``current_period`` state, which DFL XML's secondary blocks (BallClaiming,
    RefereeBall, etc., emitted after the secondHalf KickOff) violate.

    Returns:
        Mapping period_id → first KickOff EventTime for that period (UTC).
        Includes only periods whose ``<KickOff GameSection=...>`` has a
        recognized GameSection per ``_SECTION_TO_PERIOD``. Empty dict for
        inputs with no KickOffs.

    Memory: O(periods) — typically O(2). Pass cost is O(events) but
    we use ET.iterparse so the parsed tree never lives in memory.
    """
    kickoff_times: dict[int, datetime] = {}

    for _ev, elem in ET.iterparse(event_path, events=("end",)):  # noqa: S314
        if elem.tag != "Event":
            if elem.tag == "PutDataRequest":
                elem.clear()
            continue

        first_child: ET.Element | None = None
        for child in elem:
            first_child = child
            break

        if first_child is None or first_child.tag != "KickOff":
            elem.clear()
            continue

        section = first_child.get("GameSection", "")
        period = _SECTION_TO_PERIOD.get(section)
        if period is None:
            logger.warning(
                "Unrecognized GameSection %r in %s — skipping KickOff event",
                section,
                event_path,
            )
            elem.clear()
            continue

        event_time_str = elem.get("EventTime", "")
        if event_time_str:
            try:
                event_dt = datetime.fromisoformat(event_time_str)
                if event_dt.tzinfo is not None:
                    event_dt = event_dt.astimezone(timezone.utc)
                # First KickOff for a period wins (defensively — DFL XML
                # should have only one per period anyway).
                if period not in kickoff_times:
                    kickoff_times[period] = event_dt
            except (ValueError, TypeError):
                pass

        elem.clear()

    return kickoff_times


def _derive_period_from_kickoffs(
    event_dt: datetime,
    kickoff_times: dict[int, datetime],
) -> tuple[int | None, datetime | None]:
    """Given an event's EventTime, return ``(period, period_kickoff_time)``.

    Period = the largest period whose ``kickoff_time`` ≤ ``event_dt``.
    Returns ``(None, None)`` if event_dt precedes all kickoffs (legitimate
    edge case — pre-match warmup events; downstream skips them).
    """
    if not kickoff_times:
        return None, None
    best_period: int | None = None
    best_start: datetime | None = None
    for p, p_start in kickoff_times.items():
        if event_dt >= p_start and (best_start is None or p_start > best_start):
            best_period = p
            best_start = p_start
    return best_period, best_start


# --- lifted: idsse.py L1356-1469 (luxury-lakehouse @ 0efac60) ---
def _parse_events_xml(
    event_path: str,
    player_team_map: dict[str, str],
    match_id: str,
    logger: logging.Logger,
    metadata: _MatchMetadata = _EMPTY_MATCH_METADATA,
) -> list[dict[str, object]]:
    """Parse DFL event XML (DFL_03_02 series) into bronze-completeness row dicts.

    Two-pass implementation (ADR-018 / Bug #6 fix, 2026-04-29):

    - **Pass 1** (``_scan_kickoff_times``): scan KickOff events to build
      ``{period: kickoff_event_time}`` map.
    - **Pass 2** (this function body): emit per-event rows with period
      derived from ``event_time`` via ``_derive_period_from_kickoffs``.

    Pre-2026-04-29 used a state-machine ``current_period`` updated at each
    KickOff in stream order. DFL XML emits secondary blocks (BallClaiming,
    RefereeBall, etc.) AFTER the secondHalf KickOff in stream order with
    first-half event_times — these were misclassified as period=2 with
    negative period-relative ``timestamp_seconds``. The 2-pass approach
    derives period from event_time, not stream-order.

    Each ``<Event>`` in the DFL XML has exactly one first-child element
    whose tag name determines the event type (``Play``, ``ShotAtGoal``,
    ``TacklingGame``, etc.). This parser extracts:

    - **Event-level attrs (13)**: renamed to bronze cols via
      ``_EVENT_LEVEL_ATTR_MAP``.
    - **First-child attrs**: prefixed per ``_EVENT_TYPE_PREFIX`` + snake_cased.
    - **Nested-child attrs**: prefixed per ``_NESTED_PREFIX_MAP`` + snake_cased.
      Six shot-outcome tags share ``shot_outcome_*`` columns with
      ``shot_outcome_type`` as the disambiguator.
    - **Derived cols**: ``match_id`` (canonical bare DFL MatchId per
      ``shared.identifiers.idsse_native_match_id``), ``event_type``,
      ``period`` (derived from event_time vs Pass-1 kickoff map),
      ``timestamp_seconds`` (period-relative), ``player_id`` (primary actor
      via ``_PLAYER_ATTR_ORDER``), ``team`` (``home``/``away``/``unknown``
      via ``player_team_map``).

    Events whose ``event_time`` precedes all KickOffs (pre-match warmup) are
    skipped — they cannot be period-attributed. Events without an EventTime
    attribute are also skipped (cannot derive period or timestamp).

    Coordinate system: DFL pitch-origin meters (x 0-105, y 0-68). Staging
    transforms to the shared 120x80 system.
    """
    canonical_match_id = idsse_native_match_id(match_id)

    # PASS 1: build {period: kickoff_time} map.
    kickoff_times = _scan_kickoff_times(event_path)
    if not kickoff_times:
        logger.warning("No KickOff events found in %s — skipping match", event_path)
        return []

    # PASS 2: emit per-event rows.
    rows: list[dict[str, object]] = []

    for _ev, elem in ET.iterparse(event_path, events=("end",)):  # noqa: S314
        if elem.tag != "Event":
            if elem.tag == "PutDataRequest":
                elem.clear()
            continue

        first_child: ET.Element | None = None
        for child in elem:
            first_child = child
            break

        if first_child is None:
            elem.clear()
            continue

        event_type = first_child.tag

        # Derive period from event_time using pass-1 map.
        event_time_str = elem.get("EventTime", "")
        period: int | None = None
        period_start: datetime | None = None
        if event_time_str:
            try:
                event_dt = datetime.fromisoformat(event_time_str)
                if event_dt.tzinfo is not None:
                    event_dt = event_dt.astimezone(timezone.utc)
                period, period_start = _derive_period_from_kickoffs(event_dt, kickoff_times)
            except (ValueError, TypeError):
                pass

        # Skip events that predate all kickoffs (pre-match warmup) or that lack
        # a parseable EventTime — neither can be period-attributed.
        if period is None or period_start is None:
            elem.clear()
            continue

        # Seed _build_event_row's period_start_time dict directly with the
        # derived value (it would otherwise compute it from the first event
        # of the period it sees, which on a single call is just this event).
        period_start_time: dict[int, datetime] = {period: period_start}

        row = _build_event_row(
            elem,
            first_child,
            event_type,
            canonical_match_id,
            period,
            player_team_map,
            period_start_time,
            metadata,
        )
        rows.append(row)
        elem.clear()

    logger.info("Parsed %d events for IDSSE match %s", len(rows), match_id)
    return rows


# --- lifted: convert.py L19-38 (luxury-lakehouse @ 0efac60) ---
_IDSSE_CONSUMED_COLS = frozenset(
    {
        "ball_x",
        "timestamp",
        "ball_z",
        "player_id",
        "period",
        "match_id",
        "ball_s",
        "is_goalkeeper",
        "team_id",
        "frame_rate",
        "ball_y",
        "frame",
        "s",
        "y",
        "x",
        "ball_status",
    }
)


# ---------------------------------------------------------------------------
# LOCAL HARDENING (silly-kicks-local, NOT part of the 0efac60 lift) --- ball_state.
# ---------------------------------------------------------------------------
# DFL XML ``BallStatus`` is "0" (dead) / "1" (alive) in IDSSE; legacy feeds carry
# "Alive"/"Dead". ``infer_ball_carrier`` checks ``bs == "dead"``, so map before
# lowercasing so both encodings resolve correctly.
_BALL_STATUS_MAP: dict[str, str] = {"0": "dead", "1": "alive"}

# The schema value set for the produced column. MIRRORS
# ``silly_kicks.tracking.schema.TRACKING_CATEGORICAL_DOMAINS["ball_state"]`` --- duplicated
# rather than imported to keep this parse port free of any ``silly_kicks`` runtime import
# (it is a standalone bytes->bronze layer). The two are pinned equal by
# ``tests/providers/sportec/test_parse_hardening.py::test_ball_state_domain_matches_tracking_schema``.
_BALL_STATE_DOMAIN: frozenset[str] = frozenset({"alive", "dead"})


def _normalize_ball_state(raw: pd.Series, *, context: str) -> pd.Series:
    """Map a raw DFL ``ball_status`` column to the ``ball_state`` value domain.

    Closes two silent-degradation paths that the lifted body had:

    **1. dtype.** The lifted body called ``.str.lower()`` directly, which requires a
    string-like column. The XML parse path always produces one, but this function is
    also fed by **Databricks Delta round-trips**, where ``ball_status`` can come back
    as ``int64`` --- or ``float64`` when the column carries nulls. Then ``.map()``
    against a string-keyed dict yields all-NA and ``.str`` raises a bare
    ``AttributeError`` pointing at pandas rather than at the schema. Numerically-typed
    input is therefore routed through nullable ``Int64`` *first*: a naive
    ``.astype(str)`` on ``float64`` gives ``"0.0"``, which would NOT match the ``"0"``
    key, whereas ``Int64`` lands both ``int64`` and ``float64`` on ``"0"``/``"1"`` with
    NA preserved. Genuinely string-like input (``object`` of ``str``, ``StringDtype``)
    is passed through **untouched** --- that is what keeps the golden parity gate green.

    **2. output domain.** The lifted body's ``.fillna(bs.str.lower())`` passes ANY
    unmapped token straight through as the ``ball_state`` value. The schema value set is
    ``{"alive", "dead"}``, and an out-of-set value silently zeroes the column out of
    downstream domain filters (the 4.48.1 failure class). Produced values are validated
    against :data:`_BALL_STATE_DOMAIN` and violations are surfaced.

    Missing input is legitimate here (a frame with no BallStatus attribute) and yields
    NA, which is NOT treated as a domain violation.

    Parameters
    ----------
    raw : pd.Series
        Raw ``ball_status`` column.
    context : str
        Human-readable row-class label (e.g. ``"player rows"``) for the warning.

    Returns
    -------
    pd.Series
        ``ball_state`` values; NA where the input was NA.
    """
    bs = raw
    if pd.api.types.is_numeric_dtype(bs):  # covers int64 / float64 / Int64 / Float64 / bool
        bs = bs.astype("Int64").astype("string")
    out = bs.map(_BALL_STATUS_MAP).fillna(bs.str.lower()).where(bs.notna(), other=None)  # type: ignore[arg-type]  # None→NA fill is valid at runtime; pandas-stubs over-narrows `other`
    _warn_unexpected_ball_state(out, context=context)
    return out


def _warn_unexpected_ball_state(state: pd.Series, *, context: str) -> None:
    """Surface any produced ``ball_state`` outside the schema value set.

    WARN, not raise --- deliberately, and consistent with the sibling DFL-token allowlist
    idiom ``silly_kicks.spadl.sportec._warn_unexpected_play_eval``: (a) this is an
    ingestion parse layer, so aborting a whole match over one unknown BallStatus token in
    one frame would destroy every other correct row in it; (b) the token is named in the
    message, so an operator can extend :data:`_BALL_STATUS_MAP` --- coercing to NA instead
    would be a *second* silent transformation that hides which token was seen. The bug
    being fixed is the **invisibility**, not the value.

    NA is not a violation: a frame with no BallStatus attribute legitimately yields NA.
    """
    unexpected = sorted({str(v) for v in pd.Series(state).dropna().unique()} - _BALL_STATE_DOMAIN)
    if unexpected:
        warnings.warn(
            f"sportec parse: unexpected ball_status token(s) {unexpected} on {context} passed "
            f"through as ball_state, outside the schema value set {sorted(_BALL_STATE_DOMAIN)}. "
            "Downstream ball_state filters will not match them; verify against the DFL spec and "
            "extend the BallStatus map.",
            stacklevel=2,
        )


# --- lifted: convert.py L154-276 (luxury-lakehouse @ 0efac60) ---
def _bronze_idsse_to_sportec_input(trk_pdf: pd.DataFrame) -> pd.DataFrame:
    """Map bronze ``idsse_tracking`` columns to silly-kicks sportec input schema.

    Bronze ``idsse_tracking`` stores one row per player per frame with ball
    data denormalized as ``ball_x``/``ball_y``/``ball_z``/``ball_status``
    columns on every player row.  ``convert_to_frames`` expects the sportec
    ``EXPECTED_INPUT_COLUMNS`` schema which includes separate ball rows
    (``is_ball=True``, ``player_id=NaN``, ``team_id=NaN``).

    Column mapping (bronze → sportec input):

    +--------------+--------------+--------------------------------------+
    | Bronze       | Sportec      | Notes                                |
    +--------------+--------------+--------------------------------------+
    | match_id     | game_id      | rename                               |
    | period       | period_id    | rename                               |
    | frame        | frame_id     | rename                               |
    | timestamp    | time_seconds | rename                               |
    | x            | x_centered   | already DFL-centered (±52.5)         |
    | y            | y_centered   | already DFL-centered (±34.0)         |
    | s            | speed_native | rename                               |
    | ball_status  | ball_state   | ``0``→``dead``, ``1``→``alive``,     |
    |              |              | legacy ``Alive``/``Dead`` lowercased |
    | frame_rate   | frame_rate   | identity                             |
    | player_id    | player_id    | identity                             |
    | team_id      | team_id      | identity                             |
    | is_goalkeeper| is_goalkeeper| identity                             |
    +--------------+--------------+--------------------------------------+

    Synthetic ball rows are created by deduplicating
    ``(frame, period)`` and pivoting ``ball_x``/``ball_y``/``ball_z``
    into ``x_centered``/``y_centered``/``z``.  Player rows get
    ``z=NaN`` (DFL does not provide z for non-ball objects).
    """
    import pandas as pd

    # Filter to consumed columns — runtime assertion against drift.
    trk_pdf = trk_pdf[list(_IDSSE_CONSUMED_COLS)].copy()

    # ── Player rows ──────────────────────────────────────────────
    players = trk_pdf.rename(
        columns={
            "match_id": "game_id",
            "period": "period_id",
            "frame": "frame_id",
            "timestamp": "time_seconds",
            "x": "x_centered",
            "y": "y_centered",
            "s": "speed_native",
            "ball_status": "ball_state",
        },
    ).copy()
    players["is_ball"] = False
    players["z"] = np.nan

    # ball_state: see _normalize_ball_state (dtype normalisation + value-domain check).
    players["ball_state"] = _normalize_ball_state(players["ball_state"], context="player rows")

    # ── Synthetic ball rows (one per frame) ──────────────────────
    ball_src = trk_pdf[
        [
            "frame",
            "period",
            "timestamp",
            "ball_x",
            "ball_y",
            "ball_z",
            "ball_s",
            "ball_status",
            "match_id",
            "frame_rate",
        ]
    ].copy()
    ball_src = ball_src.drop_duplicates(subset=["frame", "period"])
    ball_src.rename(
        columns={
            "match_id": "game_id",
            "frame": "frame_id",
            "period": "period_id",
            "timestamp": "time_seconds",
            "ball_x": "x_centered",
            "ball_y": "y_centered",
            "ball_z": "z",
            "ball_s": "speed_native",
            "ball_status": "ball_state",
        },
        inplace=True,
    )
    ball_src["ball_state"] = _normalize_ball_state(ball_src["ball_state"], context="synthetic ball rows")
    ball_src["player_id"] = None
    ball_src["team_id"] = None
    ball_src["is_ball"] = True
    ball_src["is_goalkeeper"] = False

    # ── Combine and select only EXPECTED_INPUT_COLUMNS ───────────
    expected_cols = [
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
    ]
    result = pd.concat(
        [players[expected_cols], ball_src[expected_cols]],
        ignore_index=True,
    )
    return result.sort_values(["frame_id", "is_ball"]).reset_index(drop=True)


# --- lifted: spadl_adapter.py L251-349 (luxury-lakehouse @ 0efac60) ---
def adapt_idsse_events_for_silly_kicks(events_pdf: pd.DataFrame) -> pd.DataFrame:
    """Convert bronze ``idsse_events`` rows to silly-kicks 1.7.0 sportec input.

    Near-identity passthrough — bronze already stores the column names
    silly-kicks expects (``match_id, event_id, event_type, period,
    timestamp_seconds, player_id, team, x, y`` + optional qualifier columns
    via the DFL ``_RECOGNIZED_QUALIFIER_COLUMNS`` set). Returns a copy so
    silly-kicks's internal mutations don't leak back to the caller.

    DFL XML set-piece / foul events store team/player attribution in
    event-type-specific qualifier columns (``play_team``, ``throwin_team``,
    ``foul_team_fouler``, ``play_player``, ``foul_fouler``) rather than the
    generic ``team`` / ``player_id`` attributes.  This adapter resolves
    ``team='unknown'`` and empty ``player_id`` from those qualifiers so that
    silly-kicks receives proper values and the downstream SPADL output has
    correct team/player attribution for all event types.

    Args:
        events_pdf: DataFrame read from the ``bronze.idsse_events`` Delta
            table.

    Returns:
        Adapted DataFrame ready for ``silly_kicks.spadl.sportec.
        convert_to_actions(events, home_team_id='home')``.
    """
    # silly-kicks's converter mutates+writes intermediate columns on its
    # input — return a copy to honor the "input not mutated" contract that
    # silly-kicks's own tests assert.
    df = events_pdf.copy()

    _resolve_idsse_team_from_qualifiers(df)
    _resolve_idsse_player_from_qualifiers(df)

    return df


# -- DFL qualifier column priority for team resolution --------------------
# Each tuple: (qualifier_column, contains_dfl_clu_id).
# Columns that carry a DFL CLU id need home/away resolution; columns that
# already carry 'home'/'away' labels do not (none exist today, but the
# structure supports it).
_TEAM_QUALIFIER_PRIORITY: list[str] = [
    "play_team",
    "throwin_team",
    "freekick_team",
    "goalkick_team",
    "corner_team",
    "penalty_team",
    "foul_team_fouler",
]

# -- DFL qualifier column priority for player resolution ------------------
_PLAYER_QUALIFIER_PRIORITY: list[str] = [
    "play_player",
    "foul_fouler",
]


def _resolve_idsse_team_from_qualifiers(df: pd.DataFrame) -> None:
    """Fill ``team`` from qualifier columns where it is ``'unknown'``.

    A DFL XML set-piece with no nested ``<Play>`` -- a *direct* ThrowIn /
    FreeKick / GoalKick / CornerKick / Penalty, plus Foul -- arrives with
    ``team='unknown'`` and stores the acting (executor) team's CLU id in a
    ``{type}_team`` qualifier column.  This function resolves that CLU id to
    ``'home'`` / ``'away'`` by comparing against the match-level
    ``home_team_id_native`` / ``away_team_id_native``.  Filling from the full
    set-piece executor class is inert where a qualifier is absent and correct
    where present; an unlisted set-piece type would leave ``team='unknown'``
    and crash the downstream opponent guards.

    Mutates *df* in place.
    """
    for qual_col in _TEAM_QUALIFIER_PRIORITY:
        if qual_col not in df.columns:
            continue
        still_unknown = (df["team"] == "unknown") & df[qual_col].notna() & (df[qual_col] != "")
        if not still_unknown.any():
            continue
        is_home = still_unknown & (df[qual_col] == df["home_team_id_native"])
        is_away = still_unknown & (df[qual_col] == df["away_team_id_native"])
        df.loc[is_home, "team"] = "home"
        df.loc[is_away, "team"] = "away"


def _resolve_idsse_player_from_qualifiers(df: pd.DataFrame) -> None:
    """Fill ``player_id`` from qualifier columns where it is empty/null.

    DFL XML set-piece events store the acting player's OBJ id in
    qualifier columns (``play_player``, ``foul_fouler``).

    Mutates *df* in place.
    """
    mask = df["player_id"].isna() | (df["player_id"].astype(str) == "")
    if not mask.any():
        return

    for qual_col in _PLAYER_QUALIFIER_PRIORITY:
        if qual_col not in df.columns:
            continue
        still_empty = mask & (df["player_id"].isna() | (df["player_id"].astype(str) == ""))
        if not still_empty.any():
            break
        qual_vals = df.loc[still_empty, qual_col]
        has_qual = still_empty & qual_vals.notna() & (qual_vals != "")
        if not has_qual.any():
            continue
        df.loc[has_qual, "player_id"] = df.loc[has_qual, qual_col]


# ---------------------------------------------------------------------------
# LOCAL HARDENING (silly-kicks-local, NOT part of the 0efac60 lift) --- period column.
# ---------------------------------------------------------------------------
# Period lives under DIFFERENT names depending on the input shape: ``period_id`` on
# tracking frames, ``period`` on the DFL events bronze (see
# ``_IDSSE_EVENTS_DTYPE_OVERRIDES``, and every frame produced by
# ``shape_events_to_native``). Same both-names precedent as
# ``silly_kicks.tracking.utils.filter_extratime_frames``.
_PERIOD_COLUMN_CANDIDATES: tuple[str, ...] = ("period_id", "period")


def _resolve_period_column(events: pd.DataFrame, *, context: str) -> str:
    """Return the period column name, or RAISE if the frame carries none.

    Raising on "no period column at all" is the point of this helper. The lifted body
    inlined ``"period_id" in events.columns and ...``, which silently evaluates to
    ``False`` on the events shape --- whose column is ``period`` --- permanently disabling
    the ET integrity check downstream of it. *"I could not perform the check"* must never
    be indistinguishable from *"the check passed"*.

    Raises
    ------
    RuntimeError
        Neither ``period_id`` nor ``period`` is present.
    """
    for col in _PERIOD_COLUMN_CANDIDATES:
        if col in events.columns:
            return col
    msg = (
        f"IDSSE {context}: events carry no period column --- looked for "
        f"{list(_PERIOD_COLUMN_CANDIDATES)} (``period_id`` for tracking-frame shapes, "
        "``period`` for the DFL events bronze / shape_events_to_native output), found none. "
        "Cannot determine whether this match has extra-time periods, so the ET integrity "
        "check cannot be performed --- refusing to report a silent pass. Pass the adapted "
        "events frame (see shape_events_to_native), not a column subset."
    )
    raise RuntimeError(msg)


# --- lifted: spadl_adapter.py L438-479 (luxury-lakehouse @ 0efac60) ---
def derive_idsse_home_team_start_left(events: pd.DataFrame, home_team_id_native: str) -> bool:
    """Derive ``home_team_start_left`` for an IDSSE / Sportec match from bronze.

    Reads the firstHalf ``KickOff`` event's ``kickoff_team_left`` attribute
    (captured by the IDSSE bronze parser from the DFL XML) and compares it to
    the home team's native id. AUTHORITATIVE — ground truth from the source
    XML, not derived from event positions.

    Parameters
    ----------
    events : pd.DataFrame
        IDSSE adapted DataFrame (post ``adapt_idsse_events_for_silly_kicks``).
        Must contain ``event_type``, ``kickoff_game_section``,
        ``kickoff_team_left`` columns.
    home_team_id_native : str
        Home team's DFL native id (e.g., ``"DFL-CLU-000008"``).

    Returns
    -------
    bool
        True iff the home team is positioned on the LEFT side of the pitch
        in the first half (and thus attacks toward the right goal).

    Raises
    ------
    RuntimeError
        No firstHalf KickOff row found, or its ``kickoff_team_left`` is null.
    """
    first_half_kickoffs = events[
        (events["event_type"] == "KickOff")
        & (events["kickoff_game_section"] == "firstHalf")
        & events["kickoff_team_left"].notna()
    ]
    if first_half_kickoffs.empty:
        msg = (
            "IDSSE: no firstHalf KickOff row with kickoff_team_left found. "
            "Cannot derive home_team_start_left for silly-kicks 3.0.1 "
            "convert_to_actions(...)."
        )
        raise RuntimeError(msg)
    team_left = str(first_half_kickoffs["kickoff_team_left"].iloc[0])
    return team_left == home_team_id_native


# --- lifted: spadl_adapter.py L562-621 (luxury-lakehouse @ 0efac60) ---
def derive_idsse_home_team_start_left_extratime(events: pd.DataFrame, home_team_id_native: str) -> bool | None:
    """Derive ``home_team_start_left_extratime`` for an IDSSE / Sportec match.

    Reads the ``extraTimeFirstHalf`` (or fallback ``extraTimeSecondHalf``)
    KickOff event's ``kickoff_team_left`` attribute. AUTHORITATIVE — ground
    truth from DFL XML, not derived from positions.

    Returns ``None`` when the match has no ET periods (none of the ET
    KickOff sections present); a ``None`` value is safe to pass to silly-kicks
    4.0+ because its guard only raises when ET periods AND flag-is-None
    coincide. Raises if ET periods are recorded but the KickOff metadata is
    missing — that's an ingestion-data-integrity error, not a no-op.

    Parameters
    ----------
    events : pd.DataFrame
        IDSSE adapted DataFrame (post ``adapt_idsse_events_for_silly_kicks``).
        Must contain ``event_type``, ``kickoff_game_section``,
        ``kickoff_team_left``, and a period column --- ``period_id`` (tracking-frame
        shape) or ``period`` (DFL events bronze / ``shape_events_to_native``
        output). A frame carrying NEITHER is an error, not a no-ET match.
    home_team_id_native : str
        Home team's DFL native id.

    Returns
    -------
    bool | None
        True iff the home team is on the LEFT side at the start of ET.
        None when this match has no ET periods.

    Raises
    ------
    RuntimeError
        ET periods recorded in the period column but no ET KickOff row with
        non-null ``kickoff_team_left`` found in ``events``; or ``events``
        carries no period column at all (the check cannot be performed).
    """
    # No-op: match has no ET periods (zero IDSSE matches in lakehouse bronze
    # have ET as of 2026-05-30; this branch is the steady-state today).
    period_col = _resolve_period_column(events, context="derive_idsse_home_team_start_left_extratime")
    has_et_periods = bool(events[period_col].isin([3, 4]).any())

    et_kickoffs = events[
        (events["event_type"] == "KickOff")
        & (events["kickoff_game_section"].isin(("extraTimeFirstHalf", "extraTimeSecondHalf")))
        & events["kickoff_team_left"].notna()
    ]
    if et_kickoffs.empty:
        if has_et_periods:
            msg = (
                "IDSSE: events contain ET periods (period_id in {3, 4}) but no "
                "ET KickOff event (GameSection in {extraTimeFirstHalf, extraTimeSecondHalf}) "
                "with non-null kickoff_team_left found. Cannot derive "
                "home_team_start_left_extratime — ingestion-data-integrity error."
            )
            raise RuntimeError(msg)
        return None

    # Prefer period-3 (extraTimeFirstHalf) KickOff; fall back to period-4.
    p3_kickoffs = et_kickoffs[et_kickoffs["kickoff_game_section"] == "extraTimeFirstHalf"]
    chosen = p3_kickoffs.iloc[0] if not p3_kickoffs.empty else et_kickoffs.iloc[0]
    team_left = str(chosen["kickoff_team_left"])
    return team_left == home_team_id_native


# ---------------------------------------------------------------------------
# Materialised events bronze-column contract (see header).
# ---------------------------------------------------------------------------
_IDSSE_EVENTS_BRONZE_COLS: tuple[str, ...] = (
    "away_team_id_native",
    "calculated_frame",
    "calculated_timestamp",
    "caution_card_color",
    "caution_card_rating",
    "caution_official_card_color",
    "caution_official_person_sent_off",
    "caution_official_team",
    "caution_other_reason",
    "caution_player",
    "caution_reason",
    "caution_ref_decision_evaluation",
    "caution_team",
    "chance_assist_action",
    "chance_chance_assist",
    "chance_chance_assist_type",
    "chance_counter_attack",
    "chance_player",
    "chance_prevention_goalkeeper",
    "chance_setup_origin",
    "chance_sitter",
    "chance_situation",
    "chance_taker_setup",
    "chance_team",
    "claim_ball_possession_phase",
    "claim_player",
    "claim_team",
    "claim_type",
    "competition_native_id",
    "corner_decision_timestamp",
    "corner_placing",
    "corner_post_marking",
    "corner_rotation",
    "corner_side",
    "corner_target_area",
    "corner_team",
    "cross_goal_keeper",
    "cross_goal_keeper_interference",
    "cross_side",
    "deflection_player",
    "deflection_team",
    "deflection_type",
    "delete_reason",
    "end_frame",
    "event_id",
    "event_time",
    "event_type",
    "fairplay_ball_possession_phase",
    "fairplay_player",
    "fairplay_team",
    "fault_execution_ball_possession_phase",
    "fault_execution_player",
    "fault_execution_team",
    "foul_committing_player_action",
    "foul_foul_type",
    "foul_fouled",
    "foul_fouler",
    "foul_team_fouled",
    "foul_team_fouler",
    "freekick_decision_timestamp",
    "freekick_execution_mode",
    "freekick_team",
    "goaldis_player",
    "goaldis_reason",
    "goaldis_ref_decision_evaluation",
    "goaldis_team",
    "goalkick_decision_timestamp",
    "goalkick_team",
    "home_team_id_native",
    "kickoff_game_section",
    "kickoff_team_left",
    "kickoff_team_right",
    "match_id",
    "match_id_raw",
    "not_sent_off_player",
    "not_sent_off_reason",
    "not_sent_off_ref_decision_evaluation",
    "not_sent_off_team",
    "not_sent_off_type",
    "nutmeg_affected_player",
    "nutmeg_affected_team",
    "nutmeg_player",
    "nutmeg_team",
    "offside_player",
    "offside_team",
    "other_action_change_contingent_exhausted",
    "other_action_change_of_captain",
    "other_action_player",
    "other_action_player_becomes_goalkeeper",
    "other_action_team",
    "otherball_ball_possession_phase",
    "otherball_defensive_clearance",
    "otherball_player",
    "otherball_team",
    "pass_direction",
    "pass_free_kick_layup",
    "pass_one_two",
    "penalty_causing_player",
    "penalty_decision_timestamp",
    "penalty_fouled_player",
    "penalty_goalkeeper_behaviour",
    "penalty_goalkeeper_movement",
    "penalty_not_causing_player",
    "penalty_not_player_to_be_awarded",
    "penalty_not_reason",
    "penalty_not_ref_decision_evaluation",
    "penalty_not_team",
    "penalty_players_in_box",
    "penalty_prospective_taker",
    "penalty_ref_decision_evaluation",
    "penalty_retaken_penalty",
    "penalty_team",
    "period",
    "play_ball_possession_phase",
    "play_distance",
    "play_evaluation",
    "play_flat_cross",
    "play_from_open_play",
    "play_goal_keeper_action",
    "play_height",
    "play_penalty_box",
    "play_play_angle",
    "play_play_origin",
    "play_player",
    "play_recipient",
    "play_rotation",
    "play_semi_field",
    "play_team",
    "player_id",
    "possloss_player",
    "possloss_possession_loss_origin",
    "possloss_team",
    "possloss_type_of_possession_loss",
    "run_player",
    "run_team",
    "season_native_id",
    "shot_after_free_kick",
    "shot_amount_of_defenders",
    "shot_angle_to_goal",
    "shot_assist_action",
    "shot_assist_shot_at_goal",
    "shot_assist_type_shot_at_goal",
    "shot_ball_possession_phase",
    "shot_build_up",
    "shot_chance_evaluation",
    "shot_counter_attack",
    "shot_direct_free_kick_intention",
    "shot_distance_to_goal",
    "shot_extended_type_of_shot",
    "shot_goal_distance_goalkeeper",
    "shot_inside_box",
    "shot_outcome_assist",
    "shot_outcome_assist_contribution",
    "shot_outcome_assist_fouled_player",
    "shot_outcome_assist_type",
    "shot_outcome_blocked_by_own_team",
    "shot_outcome_current_result",
    "shot_outcome_deflection_keeper",
    "shot_outcome_deflection_player",
    "shot_outcome_error",
    "shot_outcome_goal_keeper",
    "shot_outcome_goal_prevented",
    "shot_outcome_goal_zone",
    "shot_outcome_location",
    "shot_outcome_pitch_marking",
    "shot_outcome_placing",
    "shot_outcome_player",
    "shot_outcome_ref_decision_evaluation",
    "shot_outcome_save_evaluation",
    "shot_outcome_save_result",
    "shot_outcome_save_type",
    "shot_outcome_solo",
    "shot_outcome_type",
    "shot_penalty_direction",
    "shot_penalty_execution",
    "shot_player",
    "shot_player_speed",
    "shot_pressure",
    "shot_rotation",
    "shot_setup_origin",
    "shot_shot_assist_fouled_player",
    "shot_shot_condition",
    "shot_shot_contribution",
    "shot_shot_origin",
    "shot_significance_evaluation",
    "shot_sitter_contribution",
    "shot_taker_ball_control",
    "shot_taker_setup",
    "shot_team",
    "shot_type_of_shot",
    "shot_x_g",
    "sitter_prev_player",
    "sitter_prev_reason",
    "sitter_prev_ref_decision_evaluation",
    "sitter_prev_team",
    "spectacular_player",
    "spectacular_team",
    "spectacular_type",
    "start_frame",
    "sub_player_in",
    "sub_player_out",
    "sub_playing_position",
    "sub_team",
    "tackle_ball_possession_phase",
    "tackle_dribble_evaluation",
    "tackle_dribbling_side",
    "tackle_dribbling_type",
    "tackle_goal_keeper_involved",
    "tackle_loser",
    "tackle_loser_role",
    "tackle_loser_team",
    "tackle_possession_change",
    "tackle_type",
    "tackle_winner",
    "tackle_winner_action",
    "tackle_winner_result",
    "tackle_winner_role",
    "tackle_winner_team",
    "team",
    "team_id_native",
    "throwin_decision_timestamp",
    "throwin_side",
    "throwin_team",
    "timestamp_seconds",
    "var_final_decision",
    "var_linesman1",
    "var_linesman2",
    "var_opponent_team",
    "var_proofed_event",
    "var_ref_decision",
    "var_ref_decision_evaluation",
    "var_referee",
    "var_refereein_rra",
    "var_team_challenged",
    "var_timestamp_end_action",
    "var_timestamp_start_action",
    "var_video_assistant",
    "whistle_breaking_off",
    "whistle_final_result",
    "whistle_game_section",
    "x",
    "x_position_from_tracking",
    "x_source_position",
    "y",
    "y_position_from_tracking",
    "y_source_position",
)

_IDSSE_EVENTS_DTYPE_OVERRIDES: dict[str, str] = {
    "period": "Int64",
    "start_frame": "Int64",
    "end_frame": "Int64",
    "calculated_frame": "Int64",
    "timestamp_seconds": "Float64",
    "x": "Float64",
    "y": "Float64",
    "x_source_position": "Float64",
    "y_source_position": "Float64",
    "x_position_from_tracking": "Float64",
    "y_position_from_tracking": "Float64",
}


# ---------------------------------------------------------------------------
# Public port surface (ADR-031 N1 --- silly-kicks' own domain names; the bronze
# frames are field-identical to the lakehouse bronze.idsse_* tables today, a
# versioned cross-repo contract).
# ---------------------------------------------------------------------------
# SportecTrackingBronze / SportecEventBronze are pd.DataFrames whose column SET equals
# the *_BRONZE_COLS tuples. They are type aliases (not subclasses) so callers treat them
# as ordinary DataFrames; the parse functions validate the column set on return.
SportecTrackingBronze = pd.DataFrame
SportecEventBronze = pd.DataFrame


@dataclass(frozen=True)
class MatchInfo:
    """Match-level facts parsed from the DFL matchinformation XML.

    ``home_team_start_left`` / ``...extratime`` are NOT here --- DFL sources them from
    the ``<KickOff>`` events, so derive them from the shaped events via
    :func:`derive_idsse_home_team_start_left` / :func:`derive_idsse_home_team_start_left_extratime`.
    """

    home_team_id: str
    away_team_id: str
    player_team_map: dict[str, str]
    gk_player_ids: frozenset[str]
    competition_id: str
    season_id: str
    pitch_x: float | None
    pitch_y: float | None


def _validate_bronze_columns(df: pd.DataFrame, expected: tuple[str, ...], *, kind: str) -> None:
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"sportec {kind} bronze missing columns: {sorted(missing)}")


def parse_dfl_match_info(info_path: str) -> MatchInfo:
    """Parse a DFL matchinformation XML -> :class:`MatchInfo` (teams, roster, pitch).

    Examples
    --------
    >>> mi = parse_dfl_match_info("DFL_02_01_matchinformation_DFL-MAT-XXXX.xml")  # doctest: +SKIP
    >>> mi.home_team_id  # doctest: +SKIP
    'DFL-CLU-000008'
    """
    home_team_id, away_team_id, player_team_map, gk_player_ids = _parse_teams(info_path)
    meta = _parse_match_metadata(info_path)
    return MatchInfo(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        player_team_map=player_team_map,
        gk_player_ids=frozenset(gk_player_ids),
        competition_id=meta.competition_id,
        season_id=meta.season_id,
        pitch_x=meta.pitch_x,
        pitch_y=meta.pitch_y,
    )


def _match_metadata(match_info: MatchInfo) -> _MatchMetadata:
    """Build the lifted ``_MatchMetadata`` (consumed by the parsers for the per-match
    ``*_native_id`` bronze columns) from the public :class:`MatchInfo`."""
    return _MatchMetadata(
        competition_id=match_info.competition_id,
        season_id=match_info.season_id,
        home_team_id=match_info.home_team_id,
        away_team_id=match_info.away_team_id,
        pitch_x=match_info.pitch_x,
        pitch_y=match_info.pitch_y,
    )


def parse_dfl_tracking(
    positions_path: str,
    *,
    match_info: MatchInfo,
    match_id: str,
) -> SportecTrackingBronze:
    """Parse a DFL position XML -> RAW ``bronze.idsse_tracking``-shaped DataFrame.

    Faithful ``bytes -> bronze``; NO smoothing (data-quality is consumer-side --- see the
    module docstring). The result column SET equals ``_IDSSE_TRACKING_BRONZE_COLS``.
    ``match_id`` is the bare DFL MatchId (e.g. ``"J03WMX"``, NOT ``"DFL-MAT-J03WMX"``).

    Examples
    --------
    >>> mi = parse_dfl_match_info("info.xml")  # doctest: +SKIP
    >>> bronze = parse_dfl_tracking("positions.xml", match_info=mi, match_id="J03WMX")  # doctest: +SKIP
    """
    rows_by_period = _parse_positions_xml(
        positions_path,
        match_info.player_team_map,
        match_id,
        logger,
        gk_player_ids=set(match_info.gk_player_ids),
        metadata=_match_metadata(match_info),
    )
    rows = [row for period_rows in rows_by_period.values() for row in period_rows]
    df = pd.DataFrame(rows)
    df = finalize_bronze_df(df, _IDSSE_TRACKING_BRONZE_COLS, _IDSSE_TRACKING_DTYPE_OVERRIDES)
    _validate_bronze_columns(df, _IDSSE_TRACKING_BRONZE_COLS, kind="tracking")
    return df


def parse_dfl_events(
    events_path: str,
    *,
    match_info: MatchInfo,
    match_id: str,
) -> SportecEventBronze:
    """Parse a DFL event XML -> RAW ``bronze.idsse_events``-shaped DataFrame.

    Faithful ``bytes -> bronze``. The result column SET equals ``_IDSSE_EVENTS_BRONZE_COLS``.
    ``match_id`` is the bare DFL MatchId (e.g. ``"J03WMX"``).

    Examples
    --------
    >>> mi = parse_dfl_match_info("info.xml")  # doctest: +SKIP
    >>> bronze = parse_dfl_events("events.xml", match_info=mi, match_id="J03WMX")  # doctest: +SKIP
    """
    rows = _parse_events_xml(
        events_path,
        match_info.player_team_map,
        match_id,
        logger,
        _match_metadata(match_info),
    )
    df = pd.DataFrame(rows)
    df = finalize_bronze_df(df, _IDSSE_EVENTS_BRONZE_COLS, _IDSSE_EVENTS_DTYPE_OVERRIDES)
    _validate_bronze_columns(df, _IDSSE_EVENTS_BRONZE_COLS, kind="events")
    return df


def shape_tracking_to_native(bronze: SportecTrackingBronze) -> pd.DataFrame:
    """Map RAW ``bronze.idsse_tracking`` -> ``silly_kicks.tracking.sportec`` input
    (``EXPECTED_INPUT_COLUMNS``: separate ball rows, ``x_centered``/``y_centered``, ...).

    Examples
    --------
    >>> native = shape_tracking_to_native(bronze)  # doctest: +SKIP
    """
    return _bronze_idsse_to_sportec_input(bronze)


def shape_events_to_native(bronze: SportecEventBronze) -> pd.DataFrame:
    """Map RAW ``bronze.idsse_events`` -> ``silly_kicks.spadl.sportec`` input
    (resolves set-piece/foul team + player from DFL qualifier columns).

    Examples
    --------
    >>> native = shape_events_to_native(bronze)  # doctest: +SKIP
    """
    return adapt_idsse_events_for_silly_kicks(bronze)
