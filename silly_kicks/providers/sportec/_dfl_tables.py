"""The DFL event rule tables that ``parse.py`` and ``_event_derivation.py`` both read.

One module holds them, so the two modules import it at module level and never each other in a cycle (review B
TB-IMPL-83, spec 2026-11-08 tracking-batch W8.1). The tables are lifted from the luxury-lakehouse ``idsse.py``
parser; each block names its source lines. ``parse.py`` re-exports the names it uses.
"""

from __future__ import annotations

import re

# --- lifted: idsse.py L135-141 (luxury-lakehouse @ 0efac60) ---
_SECTION_TO_PERIOD: dict[str, int] = {
    "firstHalf": 1,
    "secondHalf": 2,
    "extraTimeFirstHalf": 3,
    "extraTimeSecondHalf": 4,
    "penaltyShootout": 5,
}

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
