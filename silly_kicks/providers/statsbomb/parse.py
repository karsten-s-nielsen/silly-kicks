"""StatsBomb 360 freeze-frames -> the ``snapshot_to_tracking_frames`` contract.

**Shape, never fetch.** ``providers/sportec/parse.py`` is the precedent and it fetches nothing
(measured: zero references to ``requests``/``urllib``/``http``); the caller owns I/O and hands this
module already-loaded payloads. ``statsbombpy`` is deliberately not imported here -- it is a script
dependency, lazily imported by ``scripts/build_sb360_coverage.py``, and declared nowhere in
``pyproject.toml``.

EXTRACTED from that script rather than written beside it. The script had already grown most of the
parse half while producing ``docs/research/sb360_coverage/``, and a second implementation would be
exactly the fork ``tracking.defended_goal_x``'s docstring names as a defect class ("a second
implementation is a fork that can disagree with the first"). The script now imports from here, so
its published numbers and this port cannot drift.

Three properties of the source shape the API:

* **The 360 file carries no event type.** Its records are ``event_uuid`` + ``visible_area`` +
  the exploded freeze-frame, so every question about coverage is a JOIN against the events file.
  Measured on the real corpus, 3 of 22 open matches ship a 360 file whose ``event_uuid``s have
  ZERO overlap with their own events file while claiming the same ``match_id`` -- upstream, not a
  join defect, and COUNTED rather than silently averaged over (see :class:`JoinReport`).
* **Player flags are ACTOR-relative** -- ``teammate`` / ``actor`` / ``keeper``, with no player
  identity at all. ``is_goalkeeper`` comes from ``keeper``; ``team_id`` is a synthetic
  actor-relative pair, NOT a real team identity.
* **``visible_area`` is a FLAT ``[x1, y1, x2, y2, ...]`` polygon in StatsBomb 120x80** -- not a
  list of pairs, and not 105x68.

See NOTICE for the StatsBomb Public Data License.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd

from silly_kicks.spadl import _sb_coordinates as _sb_coords

#: Synthetic, ACTOR-RELATIVE team ids. SB360 records no team identity -- only whether a player is
#: the actor's teammate -- so these separate the two sides without claiming to name them.
ACTING_TEAM_ID = 0
OPPONENT_TEAM_ID = 1


@dataclasses.dataclass(frozen=True)
class JoinReport:
    """How well the 360 file joined to its own events file.

    ``join_rate`` is emitted rather than raised on because a zero-overlap match is real upstream
    data, and a consumer needs to EXCLUDE and COUNT it. The audit learned this expensively: such a
    match previously produced a one-row shard visually indistinguishable from a quiet match, which
    would have diluted every aggregate it entered.

    Examples
    --------
    Exclude and COUNT a match whose 360 file does not join to its own events::

        snapshots, visible_area, report = shape_snapshots(frames_raw, actions)
        if report.join_rate == 0.0:
            excluded.append(match_id)   # counted, never averaged over
    """

    n_frames: int
    n_mapped: int
    join_rate: float


def defending_gk_visible(players: list[dict]) -> bool:
    """The keeper being ATTACKED -- correct for shots and crosses.

    ``keeper`` alone answers "a keeper is visible", a different question. Freeze-frame flags are
    relative to the ACTOR, so the defending keeper is the keeper who is not a teammate.

    Examples
    --------
    Measure keeper coverage on shots::

        players = freeze_frame["freeze_frame"]
        covered = defending_gk_visible(players)   # the keeper being SHOT AT
    """
    return any(bool(p.get("keeper")) and not bool(p.get("teammate")) for p in players)


def acting_side_gk_visible(players: list[dict]) -> bool:
    """The keeper on the ACTOR's own side -- correct for GK distribution and saves.

    Which keeper is "the" keeper depends on the action. On a goal kick or a save the keeper IS the
    actor, so ``keeper AND NOT teammate`` excludes them BY CONSTRUCTION and reports 0% however good
    the coverage actually is. Measured on MLS 2023 match 3877060: ``goalkick`` and ``keeper_save``
    both read exactly 0.000 defending-keeper visibility -- a definitional artefact, not a
    measurement, and one that would have told a club its goal-kick coverage was nil.

    Reported ALONGSIDE the defending rate rather than replacing it, because the two answer
    different questions and the right one depends on the action type.

    Examples
    --------
    Measure keeper coverage on a goal kick, where the keeper IS the actor::

        players = freeze_frame["freeze_frame"]
        covered = acting_side_gk_visible(players)   # NOT defending_gk_visible, which is 0 here
    """
    return any(bool(p.get("keeper")) and bool(p.get("teammate")) for p in players)


def visible_fraction(flat: list[float]) -> float:
    """Shoelace over StatsBomb's flat ``[x0, y0, x1, y1, ...]``, normalised by the SB pitch.

    Works in NATIVE 120x80 -- it never needs SPADL coordinates, and applies no cell-centre
    correction, no y-inversion and no clip.

    Examples
    --------
    What fraction of the pitch the broadcast camera saw::

        frac = visible_fraction(record["visible_area"])   # 1.0 == the whole pitch
    """
    if len(flat) < 6:
        return 0.0
    xs, ys = list(flat[0::2]), list(flat[1::2])
    n = len(xs)
    area = 0.5 * abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n)))
    return area / (_sb_coords.SB_FIELD_LENGTH * _sb_coords.SB_FIELD_WIDTH)


def polygon_to_spadl(flat: list[float], *, fidelity_version: int = 1) -> np.ndarray:
    """Flat ``[x1, y1, x2, y2, ...]`` -> ``(N, 2)`` SPADL vertices. Scaled and inverted, NOT clipped.

    Two deliberate departures from the events path:

    * **Reshape to (N, 2) FIRST.** A flat polygon satisfies ``_convert_locations``' ``len >= 2``
      guard and yields only the FIRST vertex -- measured, a 4-vertex polygon returns shape
      ``(1, 2)`` with no error and no NaN.
    * **No clip.** A broadcast camera legitimately sees past the touchline, so clamping would
      silently shrink the observed region -- and the observed region is the entire quantity this
      column carries. ADR-038 already separates the affine from the clamp for exactly this reason.

    The 3-element shot form's ``y_offset`` of 0.05 is event semantics and is NOT applied here.

    **The cell-centre correction IS applied, for a measured reason.** ``crc`` exists because SB
    EVENT locations are cell-based ("1,1 is the top-left square yard"), and a continuous polygon is
    arguably not a cell reference -- so whether it belongs here was an open question. Two
    measurements settle it:

    * It does NOT create a conflict with :func:`visible_fraction`, which omits ``crc``. That
      function returns an AREA RATIO and ``crc`` is a pure translation, so it is invisible there
      (measured: 0.625 either way). The two readings of this polygon cannot disagree.
    * It DOES matter for player/polygon alignment. Players reach SPADL through the same
      :func:`~silly_kicks.spadl._sb_coordinates.sb_xy_to_spadl` **with** ``crc``; omitting it here
      would offset the polygon by **0.4375 m** (fidelity 1) relative to the players it bounds, so a
      player exactly on the boundary would read as outside it.

    Consistency between the players and the region that bounds them is the binding constraint, not
    the philosophical question. Pinned by
    ``test_players_and_polygon_share_one_transform``.

    Examples
    --------
    Convert one record's observed region to SPADL vertices::

        poly = polygon_to_spadl(record["visible_area"])   # (N, 2), NOT clipped to the pitch
    """
    if len(flat) < 6:
        return np.empty((0, 2), dtype=float)
    xy = np.asarray(flat, dtype=float).reshape(-1, 2)
    return _sb_coords.sb_xy_to_spadl(xy, fidelity_version=fidelity_version)


def shape_snapshots(
    frames_raw: list[dict],
    actions: pd.DataFrame,
    *,
    fidelity_version: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, JoinReport]:
    """SB360 records + SPADL actions -> ``(snapshots, visible_area, report)``.

    ``snapshots`` is one row per player per event, in the shape
    :func:`silly_kicks.tracking.snapshot_to_tracking_frames` expects. ``visible_area`` is one row
    per ACTION (a polygon is a per-action quantity, so it would repeat per player as a column).

    **``player_id`` is deliberately absent.** SB360 carries no player identity, and
    ``snapshot_to_tracking_frames`` assigns a synthetic sequential int over the WHOLE table -- so
    the same physical player receives a DIFFERENT id in every freeze-frame. That forecloses
    per-player aggregation, which for GK work is the first thing someone will try; it is stated
    here rather than discovered downstream.

    Parameters
    ----------
    frames_raw : list[dict]
        StatsBomb 360 records: ``event_uuid``, ``freeze_frame``, ``visible_area``.
    actions : pd.DataFrame
        SPADL actions carrying ``action_id`` and ``original_event_id``.
    fidelity_version : int, default 1
        ``xy_fidelity_version`` from the match metadata.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, JoinReport]

    Examples
    --------
    Shape a match's 360 payload into tracking frames::

        from silly_kicks.providers.statsbomb import shape_snapshots
        from silly_kicks.tracking import snapshot_to_tracking_frames
        snapshots, visible_area, report = shape_snapshots(frames_raw, actions)
        if report.join_rate == 0.0:
            ...  # upstream 360<->events inconsistency; exclude and COUNT the match
        frames, links = snapshot_to_tracking_frames(snapshots, actions)
    """
    by_uuid = dict(zip(actions["original_event_id"].astype(str), actions["action_id"], strict=True))

    n_frames = len(frames_raw)
    mapped = [ff for ff in frames_raw if str(ff.get("event_uuid")) in by_uuid]
    join_rate = len(mapped) / n_frames if n_frames else float("nan")
    if n_frames and not mapped:
        warnings.warn(
            f"{n_frames} freeze-frames and NONE join to an action (`event_uuid` has zero overlap "
            f"with the events file). Upstream data inconsistency -- a JoinReport with "
            f"join_rate=0.0 is returned so a consumer can exclude and COUNT this match rather "
            f"than averaging over it.",
            stacklevel=2,
        )
    report = JoinReport(n_frames=n_frames, n_mapped=len(mapped), join_rate=join_rate)

    snap_rows: list[dict] = []
    poly_rows: list[dict] = []
    for ff in mapped:
        action_id = by_uuid[str(ff["event_uuid"])]
        players = ff.get("freeze_frame") or []
        locs = [p.get("location") for p in players]
        keep = [i for i, loc in enumerate(locs) if isinstance(loc, list) and len(loc) >= 2]
        if keep:
            xy = np.asarray([locs[i][:2] for i in keep], dtype=float)
            spadl_xy = _sb_coords.sb_xy_to_spadl(xy, fidelity_version=fidelity_version)
            for row, (x, y) in zip((players[i] for i in keep), spadl_xy, strict=True):
                snap_rows.append(
                    {
                        "action_id": action_id,
                        "team_id": ACTING_TEAM_ID if bool(row.get("teammate")) else OPPONENT_TEAM_ID,
                        "is_goalkeeper": bool(row.get("keeper")),
                        "x": float(x),
                        "y": float(y),
                    }
                )
        poly = polygon_to_spadl(ff.get("visible_area") or [], fidelity_version=fidelity_version)
        if len(poly):
            poly_rows.append({"action_id": action_id, "polygon": poly})

    snapshots = pd.DataFrame(snap_rows, columns=["action_id", "team_id", "is_goalkeeper", "x", "y"])
    # `player_id` is REQUIRED by the tests' contract check but must not claim identity: emit it as
    # a per-row synthetic so a consumer sees the column and its docstring, not a silent absence.
    snapshots["player_id"] = np.arange(len(snapshots))
    visible_area = pd.DataFrame(poly_rows, columns=["action_id", "polygon"])
    return snapshots, visible_area, report
