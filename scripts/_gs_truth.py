"""The Gradient Sports contact truth (TF-79; detector spec sections 2 and 6, scorer spec sections 3 and 4).

PFF's on-ball sequences (the ``OTB`` game events) are the truth: a player and ``[startTime, endTime]`` on the event
clock, made period-relative with the period's first kickoff ``startTime``. The measure is what the cross enricher and
TF-46 read: after a release, the first contact credited to another player. This module is the single source of the
split rule, the sequences, the next-touch measures, the bar and the selection order; the materializer, the harness
and the scorer driver import it. It is visibility-blind (ADR-123 Amendment B): it names no corpus registry.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Hashable
from typing import Any

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id

#: Tolerance (s) of the next-touch classes: the 10th-90th percentile spread of a ball-only change around PFF's edges
#: is about +-0.2 s on the development half (detector spec section 2).
TRUTH_TAU_S = 0.2
#: The S1 types whose endTime is a kick (the release-timing population).
KICK_TYPES = ("PA", "CR", "SH", "CL")
#: The owner-approved bar on the held-out half (detector spec section 6).
BAR_RIGHT = 0.60
BAR_FALSE_FIRST = 0.10
BAR_TIMING_S = 0.10
#: The track-supported pairs (scorer spec r7 section 3): PFF's receiver is within this horizontal distance (m) of the
#: ball at some ball frame within +-tau of the reception. The bar's "right" and "false first" rows read these pairs.
SUPPORT_RADIUS_M = 1.5
_POPULATIONS = ("supported", "all")
_SUP = ("right", "false_first", "wrong_player", "missed")
#: Points within this of the best feasible "right" share are tied and go to the stricter detector.
NEAR_BEST = 0.005
_KICKOFF = {"FIRSTKICKOFF": 1, "SECONDKICKOFF": 2, "THIRDKICKOFF": 3, "FOURTHKICKOFF": 4}


def split_of(game_id) -> str:
    """``"dev"`` or ``"holdout"`` by the pre-declared rule: sha256 of the PFF game id as a decimal string (UTF-8),
    even -> development. A non-numeric id (the staged key ``gradientsports__<id>``) is refused, never hashed."""
    if isinstance(game_id, str):
        if not game_id.strip().isdigit():
            raise ValueError(f"split_of takes the PFF game id, not {game_id!r}")
        key = str(int(game_id))
    else:
        f = float(game_id)
        if not f.is_integer():
            raise ValueError(f"split_of takes an integral PFF game id, not {game_id!r}")
        key = str(int(f))
    return "dev" if int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 2 == 0 else "holdout"


def otb_sequences(events_json: list[dict]) -> pd.DataFrame:
    """PFF's on-ball sequences from the raw GS events JSON: one row per ``OTB`` game event with a player, with
    ``t0``/``t1`` = ``startTime``/``endTime`` minus the period's first kickoff ``startTime`` (period-relative, as the
    GS converter and the staged frames' ``periodElapsedTime``). ``player`` is the canonical id string."""
    anchor: dict[int, float] = {}
    for e in events_json:
        kind = (e.get("gameEvents") or {}).get("gameEventType")
        p = _KICKOFF.get(kind) if isinstance(kind, str) else None
        if p is not None and p not in anchor and e.get("startTime") is not None:
            anchor[p] = float(e["startTime"])
    rows = []
    for e in events_json:
        g = e.get("gameEvents") or {}
        if g.get("gameEventType") != "OTB" or g.get("playerId") is None:
            continue
        p = g.get("period")
        if p not in anchor or e.get("startTime") is None or e.get("endTime") is None:
            continue
        rows.append(
            (
                int(p),
                float(e["startTime"]) - anchor[p],
                float(e["endTime"]) - anchor[p],
                canonical_id(g["playerId"]),
                (e.get("possessionEvents") or {}).get("possessionEventType"),
            )
        )
    s = pd.DataFrame(rows, columns=["period_id", "t0", "t1", "player", "ptype"])
    return s.sort_values(["period_id", "t0"], kind="stable").reset_index(drop=True)


def score_next_touch(seq: pd.DataFrame, contacts: pd.DataFrame, *, tau: float = TRUTH_TAU_S) -> dict:
    """The detector spec section 6 measures of one match.

    For each pair of consecutive sequences S1 -> S2 by different players (S2 starting after S1 ends): the first
    resolved contact credited to a player other than P1 at or after S1's end is ``right`` (P2, within ``tau`` of S2's
    start), ``false_first`` (any player, before S2's start - ``tau``), ``wrong_player`` (another player within
    ``tau``) or ``missed``. ``release_offsets``: per S1 of a kick type, the P1 contact nearest S1's end within +-0.5 s,
    minus the end (NaN when none). ``block_*`` count the pairs whose S2 is a PFF ``RE`` (a block).

    When ``seq`` has the bool column ``supported`` (:func:`track_supported`), the ``sup_*`` counts repeat ``n`` and
    the four classes over the pairs whose S2 row is supported (scorer spec r7 section 3); without it they are absent.
    """
    src = contacts["contact_source"].astype(object)
    res = contacts.loc[(src != "unresolved") & contacts["player_id"].notna()]
    by_period = {
        p: (g["time_seconds"].to_numpy(dtype=float), [str(canonical_id(v)) for v in g["player_id"]])
        for p, g in res.sort_values("time_seconds", kind="stable").groupby("period_id")
    }
    out: dict[str, Any] = dict.fromkeys(
        ("n", "right", "false_first", "wrong_player", "missed", "n_block", "block_right", "block_missed"), 0
    )
    with_support = "supported" in seq.columns
    if with_support:
        out.update(dict.fromkeys(("sup_n", *(f"sup_{c}" for c in _SUP)), 0))
    offsets: list[float] = []
    for period, s in seq.groupby("period_id", sort=True):
        ct, cp = by_period.get(period, (np.array([]), []))
        players = [str(canonical_id(v)) for v in s["player"]]
        t0s = s["t0"].to_numpy(dtype=float)
        t1s = s["t1"].to_numpy(dtype=float)
        ptypes = s["ptype"].tolist()
        sup = s["supported"].to_numpy(dtype=bool) if with_support else None
        for i in range(len(players)):
            p1, t1 = players[i], float(t1s[i])
            if ptypes[i] in KICK_TYPES:
                own = [ct[j] - t1 for j in range(len(ct)) if cp[j] == p1 and abs(ct[j] - t1) <= 0.5]
                offsets.append(min(own, key=abs) if own else float("nan"))
            if i + 1 >= len(players):
                continue
            p2, t2 = players[i + 1], float(t0s[i + 1])
            if p2 == p1 or not t2 > t1:
                continue
            j = int(np.searchsorted(ct, t1, side="left"))
            while j < len(ct) and cp[j] == p1:
                j += 1
            if j >= len(ct) or ct[j] > t2 + tau:
                cls = "missed"
            elif ct[j] < t2 - tau:
                cls = "false_first"
            else:
                cls = "right" if cp[j] == p2 else "wrong_player"
            out["n"] += 1
            out[cls] += 1
            if sup is not None and sup[i + 1]:
                out["sup_n"] += 1
                out[f"sup_{cls}"] += 1
            if ptypes[i + 1] == "RE":
                out["n_block"] += 1
                out["block_right"] += cls == "right"
                out["block_missed"] += cls == "missed"
    out["n_contacts"] = len(contacts)
    out["n_unresolved"] = int((src == "unresolved").sum())
    out["release_offsets"] = offsets
    return out


def shares(counts: dict, offsets: list[float]) -> dict:
    """The section 6 shares from summed ``score_next_touch`` counts and the pooled release offsets. The plain keys are
    the all-pairs shares; with the ``sup_*`` counts present it adds the supported shares and ``supported_share``
    (scorer spec r7 section 3). The timing keys keep the release population."""
    n = max(int(counts["n"]), 1)
    off = np.asarray(offsets, dtype=float)
    found = off[np.isfinite(off)]
    sup: dict[str, Any] = {}
    if "sup_n" in counts:
        ns = max(int(counts["sup_n"]), 1)
        sup = {
            "sup_n": int(counts["sup_n"]),
            **{f"sup_{c}_share": counts[f"sup_{c}"] / ns for c in _SUP},
            "supported_share": int(counts["sup_n"]) / n,
        }
    return {
        **sup,
        "n": int(counts["n"]),
        "right_share": counts["right"] / n,
        "false_first_share": counts["false_first"] / n,
        "wrong_player_share": counts["wrong_player"] / n,
        "missed_share": counts["missed"] / n,
        "timing_median_s": float(np.median(found)) if len(found) else float("nan"),
        "release_missing_share": float(1.0 - len(found) / len(off)) if len(off) else float("nan"),
        "n_block": int(counts["n_block"]),
        "block_right_share": counts["block_right"] / max(int(counts["n_block"]), 1),
        "block_missed_share": counts["block_missed"] / max(int(counts["n_block"]), 1),
        "unresolved_rate": counts["n_unresolved"] / max(int(counts["n_contacts"]), 1),
    }


def _keys(population: str) -> tuple[str, str]:
    """The (right, false first) share keys of ``population``: ``"supported"`` (the bar) or ``"all"``."""
    if population not in _POPULATIONS:
        raise ValueError(f"population must be one of {_POPULATIONS}, not {population!r}")
    prefix = "sup_" if population == "supported" else ""
    return f"{prefix}right_share", f"{prefix}false_first_share"


def meets_bar(m: dict, *, population: str) -> bool:
    """All three bar rows, each inclusive at its edge; "right" and "false first" read ``population`` (scorer spec r7
    section 3), the timing row the release population. ``"supported"`` without the ``sup_*`` keys raises KeyError."""
    right, false_first = _keys(population)
    return (
        m[right] >= BAR_RIGHT
        and m[false_first] <= BAR_FALSE_FIRST
        and np.isfinite(m["timing_median_s"])
        and abs(m["timing_median_s"]) <= BAR_TIMING_S
    )


def select_point(grid: dict, *, prefer: Callable[[Any], Any], population: str) -> Hashable | None:
    """The section 6 selection over ``{point: metrics}``: the feasible points (all three bar rows), those within
    ``NEAR_BEST`` of the best feasible "right" share, then the minimum of ``prefer(point)`` -- the caller's total
    order towards the stricter detector. ``population`` as :func:`meets_bar`. ``None`` when no point is feasible."""
    right, _ = _keys(population)
    feasible = {k: m for k, m in grid.items() if meets_bar(m, population=population)}
    if not feasible:
        return None
    best = max(m[right] for m in feasible.values())
    near = [k for k, m in feasible.items() if m[right] >= best - NEAR_BEST - 1e-12]
    return min(near, key=prefer)


def track_supported(
    seq: pd.DataFrame, frames: pd.DataFrame, *, radius_m: float = SUPPORT_RADIUS_M, tau: float = TRUTH_TAU_S
) -> np.ndarray:
    """Scorer spec r7 section 3: one bool per row of ``seq`` (as passed). Row i is True when sequence i's player has a
    finite tracked position within ``radius_m`` (horizontal) of a finite ball position at some ball frame of the same
    period with ``|t - t0_i| <= tau``. It reads PFF's identity and the tracks only, never a detector output; the caller
    stores it as the column ``supported``, and :func:`score_next_touch` reads it for the S2 row of each pair."""
    out = np.zeros(len(seq), dtype=bool)
    if len(seq) == 0 or len(frames) == 0:
        return out
    is_ball = frames["is_ball"].to_numpy(dtype=bool)
    xy = ["x", "y"]
    ball = frames.loc[is_ball, ["period_id", "frame_id", "time_seconds", *xy]]
    ball = ball[np.isfinite(ball[xy].to_numpy(dtype=float)).all(axis=1)]
    pl = frames.loc[~is_ball & frames["player_id"].notna().to_numpy(), ["period_id", "frame_id", "player_id", *xy]]
    pl = pl[np.isfinite(pl[xy].to_numpy(dtype=float)).all(axis=1)]
    if ball.empty or pl.empty:
        return out
    j = pl.merge(ball, on=["period_id", "frame_id"], suffixes=("", "_ball"))
    j = j.assign(
        pid=[str(canonical_id(v)) for v in j["player_id"]],
        d=np.hypot(
            j["x"].to_numpy(dtype=float) - j["x_ball"].to_numpy(dtype=float),
            j["y"].to_numpy(dtype=float) - j["y_ball"].to_numpy(dtype=float),
        ),
    )
    near = j.loc[j["d"].to_numpy() <= radius_m]
    times = {
        k: np.sort(g["time_seconds"].to_numpy(dtype=float)) for k, g in near.groupby(["period_id", "pid"], sort=False)
    }
    periods = seq["period_id"].to_numpy()
    starts = seq["t0"].to_numpy(dtype=float)
    for i, player in enumerate(seq["player"]):
        t = times.get((periods[i], str(canonical_id(player))))
        if t is None:
            continue
        k = int(np.searchsorted(t, starts[i] - tau - 1e-9, side="left"))
        out[i] = k < len(t) and t[k] <= starts[i] + tau + 1e-9
    return out
