"""SkillCorner marker-keyed derived restarts (TF-46 Arm 1, spec section 5).

SkillCorner Game Intelligence records the RECEIVER of a restart, never the taker: ``X_reception`` is
the receiving possession, and the restart itself (corner / free kick / goal kick / throw-in) has no
event row. The shipped converter relabelled the receiving possession as the set piece, which dropped
every defence-won restart and credited receiver shots/goal-kicks to the wrong actor.

This module derives one restart row per restart from two signals (spec section 5.1):

(a) an interruption marker ``game_interruption_after`` in {corner, free_kick, goal_kick, throw_in} x
    {_for, _against} on the last possession before the stoppage; and
(b) a first-possession restart tag ``game_interruption_before`` the previous possession's marker does
    not explain (chain second restarts and the restarts with no marker).

The derived rows carry ``player_id`` NaN, a NaN start (``add_restart_coordinates`` sets geometry,
section 9), an end at the next possession's start reflected 180 degrees into the restarting team's
frame when the next team differs, an ordering-only placeholder time, and the base signal columns
``result_source`` / ``restart_evidence`` / ``type_source`` (spec section 5.4). ``action_provenance`` is
``derived``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..id_compat import canonical_id, same_id
from ..reflection import SPADL_REFLECTION_KINDS, reflect
from . import config as spadlconfig

#: Restart base types a marker/tag can name (section 5.1). ``penalty`` and ``goal`` are intentionally
#: absent: a penalty is never typed from the marker (section 6.5), and ``goal_*`` is not a restart.
_RESTART_BASES = ("corner", "free_kick", "goal_kick", "throw_in")

#: Laws-of-the-game penalty-area depth in metres; the attacking box is the far 16.5 m of the pitch.
_PENALTY_AREA_DEPTH = 16.5

#: The output column order a derived row carries (matches the native actions frame plus the base
#: signal columns the converter pre-creates so the concat preserves these values).
DERIVED_COLUMNS = (
    "game_id",
    "original_event_id",
    "action_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "type_id",
    "result_id",
    "result_source",
    "bodypart_id",
    "action_provenance",
    "restart_evidence",
    "type_source",
    "repair_kind",
    # TF-46 Arm 1 Part 2 GI facts (spec section 17.1 / 17.2). The bracket ids are set on every derived row;
    # the possession facts come from the marker/tag possession the row derives from.
    "gi_marker_possession_frame_end",
    "gi_next_possession_frame_start",
    "gi_attacking_side",
    "gi_possession_frame_start",
    "gi_possession_frame_end",
    "gi_event_id",
)


def _cid(value: object) -> str:
    """Canonical id as a plain ``str`` (ids here are non-null by construction, so never ``<NA>``)."""
    return str(canonical_id(value))


def _parse_marker(value: str) -> tuple[str | None, str | None]:
    """``"corner_for"`` -> ``("corner", "for")``; a non-restart value -> ``(None, None)``.

    Splits on the trailing ``_for`` / ``_against`` so the multi-word bases (``free_kick``,
    ``goal_kick``, ``throw_in``) parse. ``goal_for`` -> base ``goal`` (not a restart); ``penalty_for``
    -> base ``penalty`` (excluded, section 5.1).
    """
    if value.endswith("_for"):
        base, side = value[:-4], "for"
    elif value.endswith("_against"):
        base, side = value[:-8], "against"
    else:
        return None, None
    return (base, side) if base in _RESTART_BASES else (None, None)


def _other_team(team: str, home_id: str, opponent: str | None) -> str | None:
    """The other of the two teams relative to ``team`` (for an ``_against`` restart).

    Returns ``None`` when the opponent is unresolved (single-team slice): no ``_against`` restart can
    be derived and the caller counts it unresolved (section 5.2).
    """
    if opponent is None:
        return None
    return home_id if same_id(team, opponent) else opponent


def _in_attacking_box(end_x: float, end_y: float) -> bool:
    """True when ``(end_x, end_y)`` lies in the restarting team's attacking penalty area, m = 0.

    Each SkillCorner possession is already in its own attacking-left-to-right frame, so the attacking
    box is the far 16.5 m (x >= field_length - 16.5) within the penalty-area half-width of centre.
    NaN (a chain, with no resolved end) is not in the box.
    """
    if not np.isfinite(end_x) or not np.isfinite(end_y):
        return False
    half_w = spadlconfig.penalty_area_half_width
    return (end_x >= spadlconfig.field_length - _PENALTY_AREA_DEPTH) and (
        abs(end_y - spadlconfig.field_width / 2.0) <= half_w
    )


def derive_restarts(pp: pd.DataFrame, opponent: str | None, match_metadata: dict) -> pd.DataFrame:
    """Derive the restart rows from the possession markers and tags (spec section 5).

    Parameters
    ----------
    pp : pd.DataFrame
        The chronologically-sorted ``player_possession`` frame, carrying ``period_id``,
        ``time_seconds``, ``team_id`` (string), ``event_id``, ``start_type``,
        ``game_interruption_before`` / ``game_interruption_after``, and the SPADL-frame start
        coordinates ``start_x`` / ``start_y`` (used, shifted, as each restart's end point).
    opponent : str or None
        The single opponent team id (``_resolve_opponent``); ``None`` for a single-team slice, where
        no ``_against`` restart derives.
    match_metadata : dict
        Parsed ``match.json`` (``id`` -> ``game_id``; ``home_team.id`` anchors ``_other_team``).

    Returns
    -------
    pd.DataFrame
        The derived restart rows in :data:`DERIVED_COLUMNS` order (``action_id`` a 0 placeholder; the
        converter re-indexes after the concat). Empty frame (correct columns) when none derive.
    """
    game_id = str(match_metadata.get("id", "unknown"))
    home_id = _cid(match_metadata["home_team"]["id"])

    pp = pp.reset_index(drop=True)
    n = len(pp)
    if n == 0:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in DERIVED_COLUMNS})

    team = [_cid(t) for t in pp["team_id"]]
    period = pp["period_id"].to_numpy()
    time = pp["time_seconds"].to_numpy(dtype="float64")
    gi_after = pp["game_interruption_after"].fillna("").astype(str).tolist()
    gi_before = pp["game_interruption_before"].fillna("").astype(str).tolist()
    start_type = pp["start_type"].fillna("").astype(str).tolist()
    sx = pp["start_x"].to_numpy(dtype="float64")
    sy = pp["start_y"].to_numpy(dtype="float64")
    event_id = pp["event_id"].astype("object").tolist()

    # GI facts (spec section 17.2, CL-1 / CL-13). Optional on the input: a bronze without the GI frame
    # columns leaves every derived row's GI column <NA> (backward compatible).
    frame_start = pp["frame_start"].tolist() if "frame_start" in pp.columns else [pd.NA] * n
    frame_end = pp["frame_end"].tolist() if "frame_end" in pp.columns else [pd.NA] * n
    attacking_side = pp["attacking_side"].tolist() if "attacking_side" in pp.columns else [pd.NA] * n

    def _gi_int(value) -> object:
        """A frame id as a nullable int, or ``pd.NA`` for a missing / NaN value."""
        if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
            return pd.NA
        return int(value)

    def _gi_obj(value: object) -> object:
        """An object GI fact, or ``pd.NA`` for a missing / NaN value."""
        if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
            return pd.NA
        return value

    # A possession's gi_before restart (base, side), and the team the restart is awarded to.
    tag_base: list[str | None] = []
    tag_restarter: list[str | None] = []
    for i in range(n):
        base, side = _parse_marker(gi_before[i])
        tag_base.append(base)
        tag_restarter.append(team[i] if side == "for" else _other_team(team[i], home_id, opponent) if base else None)

    # A possession's gi_after marker (base, side), excluding the last possession of a period
    # (half-time carry-over) and any penalty marker.
    marker_base: list[str | None] = []
    marker_restarter: list[str | None] = []
    for i in range(n):
        base, side = _parse_marker(gi_after[i])
        last_in_period = i == n - 1 or period[i] != period[i + 1]
        if base is None or last_in_period:
            marker_base.append(None)
            marker_restarter.append(None)
            continue
        marker_base.append(base)
        marker_restarter.append(team[i] if side == "for" else _other_team(team[i], home_id, opponent))

    rows: list[dict] = []
    unresolved_against = 0

    def _restart_type_id(base: str, end_x: float, end_y: float, won: bool) -> tuple[int, object]:
        """``(type_id, type_source)`` for a restart base (section 5.4, 5.6, 5.7)."""
        if base == "goal_kick":
            return spadlconfig.actiontype_id["goalkick"], "native"
        if base == "throw_in":
            return spadlconfig.actiontype_id["throw_in"], "native"
        if base == "corner":
            # Every derived corner is corner_crossed (section 5.6); a won corner is type_source
            # "default" (a constant crossed fails the kappa bar for won corners), a lost corner
            # "inferred".
            return spadlconfig.actiontype_id["corner_crossed"], ("default" if won else "inferred")
        # free_kick: crossed when the first-possession point is in the attacking box (m = 0),
        # else short; type_source inferred (section 5.7).
        crossed = _in_attacking_box(end_x, end_y)
        tid = spadlconfig.actiontype_id["freekick_crossed" if crossed else "freekick_short"]
        return tid, "inferred"

    def _chain_result(second_i: int, first_restarter: str) -> tuple[str, int]:
        """The first restart's ``(restart_evidence, result_id)`` from the second restart (section 5.5)."""
        second_base = tag_base[second_i]
        second_restarter = tag_restarter[second_i]
        awarded_to_t = second_restarter is not None and same_id(second_restarter, first_restarter)
        if second_base == "free_kick":  # free kick (a penalty never derives a restart)
            if awarded_to_t:
                return "chain_ambiguous", spadlconfig.result_id["success"]
            return "chain", spadlconfig.result_id["fail"]
        # corner / goal_kick / throw_in
        if awarded_to_t:
            return "chain", spadlconfig.result_id["fail"]
        return "chain_ambiguous", spadlconfig.result_id["fail"]

    def _normal_receiver(marker_i: int) -> bool:
        """True when ``marker_i + 1`` is the normal receiver of ``marker_i``'s restart.

        The normal receiver carries a ``game_interruption_before`` tag naming the SAME restart (same
        base and same awarded team) as the marker. A differing tag means the next possession is a new
        (second) restart, i.e. a chain (section 5.5).
        """
        nxt = marker_i + 1
        return (
            nxt < n
            and marker_base[marker_i] is not None
            and tag_base[nxt] == marker_base[marker_i]
            and marker_restarter[marker_i] is not None
            and tag_restarter[nxt] is not None
            and same_id(tag_restarter[nxt], marker_restarter[marker_i])
        )

    def _emit(i: int, base: str, restarter: str, kind: str) -> None:
        """Build one derived restart row. ``kind`` is ``"marker"`` (a) or ``"tag_only"`` (b)."""
        nxt = i + 1
        has_next = nxt < n
        next_team = team[nxt] if has_next else None
        if kind == "marker":
            normal = _normal_receiver(i)
            chained = has_next and tag_base[nxt] is not None and not normal
        else:
            # tag_only: resolved on its own next possession; a chain only if that next is itself a
            # restart tag (a rare 3-chain), in which case the end is unresolved.
            normal = has_next and tag_base[nxt] is None
            chained = has_next and tag_base[nxt] is not None
        won = has_next and not chained and same_id(next_team, restarter)
        end_x = float(sx[nxt]) if (has_next and not chained) else float("nan")
        end_y = float(sy[nxt]) if (has_next and not chained) else float("nan")
        tid, type_source = _restart_type_id(base, end_x, end_y, won)

        if kind == "marker" and chained:
            evidence, result_id = _chain_result(nxt, restarter)
        else:
            evidence = "tag_only" if kind != "marker" else ("tagged" if normal else "untagged")
            result_id = spadlconfig.result_id["success"] if won else spadlconfig.result_id["fail"]

        # result_source: native only when a real, tagged reception/interception received it.
        if chained or not has_next or start_type[nxt] in ("", "unknown"):
            result_source = "inferred"
        elif start_type[nxt].endswith(("_reception", "_interception")):
            result_source = "native"
        else:
            result_source = "inferred"

        # Ordering-only placeholder time: just before the next possession, never before this one. The
        # spec section 5.4 chain form 0.02*(n-k+1) staggers rows that share ONE bracket; here each
        # derived restart keys a DISTINCT next possession (a marker's receiver, or a tag_only's own
        # next), so a flat -0.02 already yields distinct, in-bracket, correctly-ordered times (A1-IMPL-07).
        if has_next:
            t = float(time[nxt]) - 0.02
            if np.isfinite(time[i]):
                t = max(t, float(time[i]) + 0.001)
        else:
            t = float(time[i]) + 0.02 if np.isfinite(time[i]) else float("nan")

        bodypart = "other" if base == "throw_in" else "foot"
        rows.append(
            {
                "game_id": game_id,
                "original_event_id": event_id[nxt] if (has_next and not chained) else event_id[i],
                "action_id": 0,
                "period_id": int(period[i]),
                "time_seconds": t,
                "team_id": restarter,
                "player_id": pd.NA,
                "start_x": float("nan"),
                "start_y": float("nan"),
                "end_x": end_x,
                "end_y": end_y,
                "type_id": tid,
                "result_id": result_id,
                "result_source": result_source,
                "bodypart_id": spadlconfig.bodypart_id[bodypart],
                "action_provenance": "derived",
                "restart_evidence": evidence,
                "type_source": type_source,
                "repair_kind": pd.NA,
                # GI facts: the bracket ids (marker possession i's frame_end, next possession's frame_start)
                # and the possession i facts for the per-match pairing (spec section 17.1 / 17.2).
                "gi_marker_possession_frame_end": _gi_int(frame_end[i]),
                # <NA> on a chain row: no resolved possession lies between (CL-13, spec section 17.2).
                "gi_next_possession_frame_start": (_gi_int(frame_start[nxt]) if (has_next and not chained) else pd.NA),
                "gi_attacking_side": _gi_obj(attacking_side[i]),
                "gi_possession_frame_start": _gi_int(frame_start[i]),
                "gi_possession_frame_end": _gi_int(frame_end[i]),
                "gi_event_id": event_id[i],
                "_next_team": next_team,  # scratch, dropped before return (drives reflection)
                "_restarter": restarter,
            }
        )

    for i in range(n):
        # (a) marker-derived restart on the possession before the stoppage.
        mbase, mrest = marker_base[i], marker_restarter[i]
        if mbase is not None:
            if mrest is None:
                unresolved_against += 1  # _against with no resolvable opponent (single-team slice)
            else:
                _emit(i, mbase, mrest, "marker")

        # (b) a first-possession restart tag the previous possession's marker does not explain:
        # a chain's second restart, or a restart with no marker. The normal receiver of a non-chain
        # marker is explained and derives no row here.
        tbase, trest = tag_base[i], tag_restarter[i]
        if tbase is None:
            continue
        if i == 0 or period[i] != period[i - 1]:
            continue  # a tag on the first possession of a period never creates a restart
        if _normal_receiver(i - 1):
            continue  # explained: this possession is the previous marker's normal receiver
        if trest is None:
            unresolved_against += 1
            continue
        _emit(i, tbase, trest, "tag_only")

    if not rows:
        out = pd.DataFrame({c: pd.Series(dtype="object") for c in DERIVED_COLUMNS})
        out.attrs["unresolved_against"] = unresolved_against
        return out

    derived = pd.DataFrame(rows)
    # Reflect the end point 180 degrees into the restarting team's frame when the next possession's
    # team differs from the restarting team (section 5.4, ADR-045). Reflection is a no-op on the NaN
    # start and on chain rows (NaN end).
    differs = np.array(
        [
            not (r_next is not None and same_id(r_next, r_rest))
            for r_next, r_rest in zip(derived["_next_team"], derived["_restarter"], strict=True)
        ],
        dtype=bool,
    )
    derived = reflect(derived, pd.Series(differs, index=derived.index), kinds=SPADL_REFLECTION_KINDS)
    derived = derived.drop(columns=["_next_team", "_restarter"])
    derived = derived[list(DERIVED_COLUMNS)]
    derived.attrs["unresolved_against"] = unresolved_against
    return derived
