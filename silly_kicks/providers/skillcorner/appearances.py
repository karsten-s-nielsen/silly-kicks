"""SkillCorner keeper-appearance extractor (TF-59 PR1, spec §5.5).

Produces the :func:`~silly_kicks.keeper_identity.validate_keeper_appearances` port -- one row per
keeper on-pitch interval PER PERIOD -- from a parsed SkillCorner ``match.json`` dict (the shape the
public pining feed serves, verified on real A-League data 2026-09-01). One signal drives it:

* **Keepers from ``players[]``.** Each ``players[]`` record carries a ``player_role`` and a
  ``playing_time.by_period[]``. A player is a keeper iff ``player_role.acronym == "GK"`` (equivalently
  ``player_role.name == "Goalkeeper"``). **Investigation (reported).** ``player_role.position_group``
  is NOT a keeper signal -- it is ``"Other"`` for every keeper (starter or substitute), verified on
  the public fixture and on three real peggy44 GK-change matches. But ``acronym == "GK"`` DOES catch a
  SUBSTITUTE keeper: in every GK-change match inspected the incoming keeper is tagged
  ``{acronym: "GK", name: "Goalkeeper", position_group: "Other"}`` -- so ``acronym == "GK"`` covers
  starters AND sub keepers, and there is no sub-keeper *identification* gap.

For each keeper, the ``playing_time.by_period[]`` entries are mapped to period ids via
``match_periods`` (matched on the ``name`` field, e.g. ``"period_1"``), and each entry's
``start_frame`` / ``end_frame`` is converted to PERIOD-RELATIVE seconds against that period's own
``match_periods[p].start_frame`` offset at THAT PERIOD's own fps (ADR-017 -- the port's time base
resets each period, so each period is converted with its own rate; a single feed-wide rate was the D2
gap). ``fps`` is DERIVED PER PERIOD from ``match_periods`` (``duration_frames / (duration_minutes *
60)``) and asserted against the documented SkillCorner rate of 10 fps; a genuine but within-band
inter-period rate difference warns rather than being silently absorbed.
One :class:`~silly_kicks.keeper_identity.KeeperSegment`
is built per keeper spanning their FIRST -> LAST on-pitch by_period (``source="native_intervals"``); a
keeper whose last by_period reaches the match's final period end (played to the whistle) gets an OPEN
end (``+inf``), else the finite ``end_frame`` in period-relative seconds. The per-period decomposition
is the shared :func:`~silly_kicks.keeper_identity.build_keeper_appearances_from_segments`.

**Two folder layouts, one schema.** SkillCorner ships the same ``match.json`` under ``meta/<id>.json``
(24/25) and ``matches/<id>.json`` (25/26); this function takes the already-parsed dict, so it is
layout-agnostic. **The by_period schema is A-League-native (reported).** The peggy44 / owner export
carries a REDUCED ``match.json`` with no ``match_periods`` / ``by_period`` (only clock ``start_time`` /
``end_time``); on that shape the extractor returns a valid EMPTY appearances frame (no by_period
intervals to extract) -- interval resolution then falls back to the coarse keeper map downstream. The
by_period-bearing A-League schema is where intervals are extractable.

SkillCorner ids are STRINGS; the port's id columns are ``object``, so they are stored un-coerced and
every downstream id comparison routes through :mod:`silly_kicks.id_compat` (ADR-019). Imports are
confined to :mod:`silly_kicks.keeper_identity` + pandas/stdlib -- NOT :mod:`silly_kicks.tracking`.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import pandas as pd

from silly_kicks.keeper_identity import (
    KeeperSegment,
    build_keeper_appearances_from_segments,
)

__all__ = ["extract_keeper_appearances"]

#: The documented SkillCorner tracking frame rate. ``fps`` is DERIVED per match from ``match_periods``
#: and asserted against this; the derivation is a sanity check, not a hardcode.
_DOCUMENTED_FPS = 10.0

#: Tolerance for the derived-vs-documented fps assertion. ``duration_minutes`` is a display value
#: rounded to 0.1 min, so the derived fps is within ~0.01 of 10.0; a 0.5 band flags only a gross error
#: (a mis-parsed feed / a different provider), never rounding noise.
_FPS_TOLERANCE = 0.5

#: Inter-period fps spread above which the periods are sampled at genuinely different rates -> WARN.
#: Well above ``duration_minutes``-rounding noise (~0.02) and well below :data:`_FPS_TOLERANCE`, so a
#: normal feed never warns; each period is still converted with its OWN rate regardless.
_FPS_PERIOD_SPREAD_WARN = 0.1

#: ``player_role.acronym`` value for a goalkeeper (starter OR substitute -- see the module docstring).
_GK_ACRONYM = "GK"
#: ``player_role.name`` value for a goalkeeper (the co-present alternative to ``acronym == "GK"``).
_GK_ROLE_NAME = "Goalkeeper"


def _is_keeper(player: dict) -> bool:
    """True iff the player's ``player_role`` marks a goalkeeper (``acronym == "GK"`` or ``name ==
    "Goalkeeper"``). ``position_group`` is deliberately NOT consulted -- it is ``"Other"`` for every
    keeper (see the module docstring's investigation note)."""
    role = player.get("player_role") or {}
    return role.get("acronym") == _GK_ACRONYM or role.get("name") == _GK_ROLE_NAME


def _index_match_periods(
    match_periods: list[dict],
) -> tuple[dict[Any, dict[str, Any]], list[int], int, float, dict[int, float]]:
    """Index ``match_periods`` into ``(by_name, period_ints, final_period_int, final_end_frame, fps_by_period)``.

    ``by_name`` maps a by_period ``name`` (e.g. ``"period_1"``) -> ``{"period", "start_frame"}``;
    ``period_ints`` is the sorted distinct period ids (the decomposition axis); ``final_period_int`` /
    ``final_end_frame`` are the last period's id and ``end_frame`` (the played-to-whistle boundary);
    ``fps_by_period`` is the PER-PERIOD derived-and-asserted frame rate -- each period's frames are
    converted with its OWN rate (ADR-017; the port's time base resets each period). An entry missing a
    usable ``period`` / ``start_frame`` is skipped; a match with no usable periods yields an empty index
    (the caller then returns an empty frame -- ``final_period_int`` is ``-1``, ``final_end_frame`` ``+inf``).
    """
    by_name: dict[Any, dict[str, Any]] = {}
    period_ints: list[int] = []
    fps_by_period: dict[int, float] = {}
    end_frames: dict[int, float] = {}
    for pm in match_periods:
        name = pm.get("name")
        period = pm.get("period")
        start_frame = pm.get("start_frame")
        if name is None or period is None or start_frame is None:
            continue
        p_int = int(period)
        by_name[name] = {"period": p_int, "start_frame": float(start_frame)}
        period_ints.append(p_int)
        end = _opt_float(pm.get("end_frame"))
        if end is not None:
            end_frames[p_int] = end
        dur_frames = pm.get("duration_frames")
        dur_min = pm.get("duration_minutes")
        if dur_frames is not None and dur_min is not None and float(dur_min) > 0:
            fps_by_period[p_int] = float(dur_frames) / (float(dur_min) * 60.0)

    period_ints = sorted(set(period_ints))
    final_period_int = period_ints[-1] if period_ints else -1
    final_end_frame = end_frames.get(final_period_int, math.inf)
    return by_name, period_ints, final_period_int, final_end_frame, _resolve_fps_by_period(fps_by_period, period_ints)


def _resolve_fps_by_period(derived: dict[int, float], period_ints: list[int]) -> dict[int, float]:
    """A PER-PERIOD fps map, each derived rate asserted against the documented 10 fps.

    Each period's frames are converted with its OWN fps (the port's time base resets every period;
    ADR-017), so a per-period rate difference never leaks across the period boundary -- the historical
    D2 gap, where the first period's derived rate was applied to EVERY period. ``duration_minutes`` is a
    rounded display value, so a genuine feed's per-period derivations sit within :data:`_FPS_TOLERANCE`
    of the documented rate; a rate outside that band is a mis-parsed feed and RAISES. A period carrying
    no duration fields falls back to the documented rate. A genuine BUT within-band inter-period spread
    above :data:`_FPS_PERIOD_SPREAD_WARN` WARNS -- the times stay correct (each period uses its own
    rate), but a real cross-period rate difference is worth surfacing rather than silently absorbing.
    """
    for p, fps in derived.items():
        if abs(fps - _DOCUMENTED_FPS) > _FPS_TOLERANCE:
            raise ValueError(
                f"derived skillcorner fps {fps:.4f} for period {p} is not within {_FPS_TOLERANCE} of "
                f"the documented {_DOCUMENTED_FPS} fps (mis-parsed match_periods?)"
            )
    if derived:
        spread = max(derived.values()) - min(derived.values())
        if spread > _FPS_PERIOD_SPREAD_WARN:
            rates = {p: round(v, 4) for p, v in sorted(derived.items())}
            warnings.warn(
                f"skillcorner fps differs across periods beyond rounding noise (spread {spread:.4f}); "
                f"each period's frames are converted with its own derived rate: {rates}",
                stacklevel=2,
            )
    return {p: derived.get(p, _DOCUMENTED_FPS) for p in period_ints}


def _opt_float(value: Any) -> float | None:
    """``float(value)`` or ``None`` for a missing/NaN value (a by_period / match_periods frame cell)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return float(value)


def _keeper_segment(
    player: dict,
    by_name: dict[Any, dict[str, Any]],
    final_period_int: int,
    final_end_frame: float,
    fps_by_period: dict[int, float],
) -> KeeperSegment | None:
    """Build ONE :class:`KeeperSegment` for a keeper from their ``playing_time.by_period[]``.

    The segment spans the keeper's FIRST -> LAST resolvable by_period (``source="native_intervals"``):
    the first entry's ``start_frame`` -> period-relative ``start_time`` (clamped ``>= 0``), the last
    entry's ``end_frame`` -> period-relative ``end_time`` (or ``+inf`` when the last by_period reaches
    the match's final period end -- played to the whistle). Returns ``None`` when the keeper has no
    resolvable by_period entry (an unused bench keeper, or the reduced peggy44 schema).
    """
    by_period = (player.get("playing_time") or {}).get("by_period") or []
    entries: list[tuple[int, float, float, float | None]] = []
    for bp in by_period:
        pm = by_name.get(bp.get("name"))
        if pm is None:
            continue  # a by_period whose name is not in match_periods -> cannot date it
        start_frame = _opt_float(bp.get("start_frame"))
        if start_frame is None:
            continue
        entries.append((pm["period"], pm["start_frame"], start_frame, _opt_float(bp.get("end_frame"))))
    if not entries:
        return None

    entries.sort(key=lambda e: e[0])
    first_period_int, first_period_start, first_start_frame, _ = entries[0]
    last_period_int, last_period_start, _, last_end_frame = entries[-1]
    # Each end of the span is converted with ITS OWN period's fps (ADR-017; the D2 fix -- a per-period
    # rate difference no longer leaks across the boundary).
    fps_first = fps_by_period[first_period_int]
    fps_last = fps_by_period[last_period_int]

    start_time = max(0.0, (first_start_frame - first_period_start) / fps_first)
    # Played to the whistle iff the last by_period IS the final match period AND its end_frame reaches
    # that period's end (within ~1s -- tracking commonly stops a frame or two before the nominal end);
    # a missing last end_frame is treated as open, too. Otherwise the finite period-relative end.
    if last_end_frame is None or (
        last_period_int == final_period_int
        and math.isfinite(final_end_frame)
        and last_end_frame >= final_end_frame - fps_last
    ):
        end_time = math.inf
    else:
        end_time = (last_end_frame - last_period_start) / fps_last

    return KeeperSegment(
        team_id=player.get("team_id"),
        player_id=player.get("id"),
        source="native_intervals",
        start_period=first_period_int,
        start_time=start_time,
        end_period=last_period_int,
        end_time=end_time,
    )


def extract_keeper_appearances(match_json: dict, *, game_id: object = None) -> pd.DataFrame:
    """Extract per-period keeper on-pitch intervals from a SkillCorner ``match.json`` (the TF-59 port, §5.5).

    See the module docstring for the driving signal (keepers from ``players[]`` by
    ``player_role.acronym == "GK"``; per-period frames -> seconds via ``match_periods``). ``game_id``
    defaults to ``match_json["id"]`` when ``None``. Returns a validated
    :func:`~silly_kicks.keeper_identity.validate_keeper_appearances` frame (columns in
    :data:`~silly_kicks.keeper_identity.KEEPER_APPEARANCE_COLUMNS` order); a reduced-schema input with
    no ``match_periods`` / ``by_period`` yields a valid EMPTY frame. PURE -- ``match_json`` is never
    mutated (only read).

    Examples
    --------
    Read a parsed SkillCorner ``match.json`` (from either the ``meta/<id>.json`` or
    ``matches/<id>.json`` layout) into per-period keeper intervals. A real ``match.json`` carries
    ``players[].playing_time.by_period[]`` + ``match_periods`` -- see
    ``tests/providers/skillcorner/test_appearances.py``:

        from silly_kicks.providers.skillcorner.appearances import extract_keeper_appearances

        appearances = extract_keeper_appearances(match_json)

    See NOTICE for full bibliographic citations.
    """
    if game_id is None:
        game_id = match_json.get("id")

    match_periods = match_json.get("match_periods") or []
    by_name, period_ints, final_period_int, final_end_frame, fps_by_period = _index_match_periods(match_periods)
    if not period_ints:
        # Reduced schema (no match_periods) -> no by_period intervals to extract. Return a valid EMPTY
        # appearances frame (the documented gap for the peggy44 / owner export shape).
        return build_keeper_appearances_from_segments([], [], game_id=game_id)

    segments: list[KeeperSegment] = []
    for player in match_json.get("players", []):
        if not _is_keeper(player):
            continue
        seg = _keeper_segment(player, by_name, final_period_int, final_end_frame, fps_by_period)
        if seg is not None:
            segments.append(seg)

    return build_keeper_appearances_from_segments(segments, period_ints, game_id=game_id)
