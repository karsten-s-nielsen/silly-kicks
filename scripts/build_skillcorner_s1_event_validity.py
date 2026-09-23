"""Task 0 measurement (spec 2026-09-22 section 8): does a SkillCorner match excluded by the S1
tracking-geometry gate also have defective EVENT coordinates?

S1 measures tracking coordinates and decides whether a match's TRACKING is admissible. It says
nothing about the match's events unless the defect is shown to reach them. This driver measures that
directly, so events-only consumers can admit an S1-excluded match iff its events pass a positive,
purpose-built event-side check (spec section 8; the admission layer that reads this artifact is
scripts/_events_admission.py).

It runs two for_each passes over the SAME ref list:
  * events pass  -- load_match(ref, events_only=True):  the anchored-consistency + boundary event
    statistics for every match whose events load (including matches whose tracking will not).
  * tracking pass -- load_match(ref, events_only=False): S1's player/ball off-pitch rates; an S1
    exclusion is a persisted marker carrying the rates.

The producer NEVER reads its own verdicts.json (spec CDLS-SPEC-27): a first run cannot be blocked by
a missing artifact, a re-run after new uploads measures the new matches, and a previously non-sound
match is re-measured. Verdicts are RECOMPUTED in every reduce from the per-match statistic shards.

The reduce REFUSES while either pass has outstanding failures (spec CDLS-SPEC-28) unless
--allow-failed, which records them. Provenance: require_clean_tree FIRST, ADR-052 shards, an ADR-056
input contract.

ASCII-only source: --help executes main() on parserless scripts elsewhere, and a non-ASCII byte in a
driver breaks --help on a non-UTF-8 Windows console (the driver ASCII gate).
"""

from __future__ import annotations

import argparse
import functools
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._driver import for_each, read_exclusion, shard_path
from scripts._events_admission import listing_digest
from scripts._loader_pining import (
    LoadedMatch,
    MatchRef,
    list_match_refs,
    load_match,
    resolve_cache_dir,
)

# --- constants -----------------------------------------------------------------------------

#: SkillCorner SPADL types whose HALF of the pitch is fixed by the laws of the game (action-LTR:
#: the acting team attacks x = 105).
OWN_HALF_ANCHORED = ("goalkick", "clearance")
ATTACK_HALF_ANCHORED = ("shot", "corner_short", "corner_crossed")
ANCHORED_TYPES = OWN_HALF_ANCHORED + ATTACK_HALF_ANCHORED

#: The halfway line. Own-half-anchored actions are consistent below it; attacking-half above it.
HALFWAY_X = 52.5

#: A cell (team, period) is SCORED only with at least this many anchored actions.
MIN_CELL_N = 5

#: Open-play action starts pile up on the pitch boundary when an origin/scale break is hidden by the
#: events converter's on-pitch clamp. These restart types have law-fixed origins and are excluded.
SET_PIECE_TYPES = frozenset(
    {"goalkick", "corner_crossed", "corner_short", "throw_in", "freekick_crossed", "freekick_short", "shot_freekick"}
)

#: Distance (m) from a pitch edge within which an action start counts as "on the boundary".
BOUNDARY_EPS_M = 0.01

#: The boundary-pileup threshold floor: tau = max(this, 3 * q99.9 over S1-passing matches).
BOUNDARY_TAU_FLOOR = 0.05

#: Per-cell reversal significance, before the Bonferroni division by m (scored cells across the
#: S1-excluded matches).
REVERSAL_ALPHA = 0.05


@functools.cache
def _type_name_by_id() -> dict[int, str]:
    from silly_kicks.spadl import config as spadlconfig

    return {int(v): k for k, v in spadlconfig.actiontype_id.items()}


@functools.cache
def _anchored_type_ids() -> dict[str, int]:
    from silly_kicks.spadl import config as spadlconfig

    return {name: int(spadlconfig.actiontype_id[name]) for name in ANCHORED_TYPES}


# --- Step 1: anchored-consistency statistic ------------------------------------------------


def anchored_consistency(actions: pd.DataFrame) -> pd.DataFrame:
    """One row per ANCHORED action, tagged consistent against its law-fixed half (action-LTR).

    Columns: ``game_id``, ``team_id``, ``period_id``, ``type_name``, ``start_x``, ``consistent``.
    Non-anchored actions (no half-of-the-pitch law) are dropped.
    """
    ids = _anchored_type_ids()
    own = {ids[t] for t in OWN_HALF_ANCHORED}
    keep = actions[actions["type_id"].isin(ids.values())].copy()
    names = _type_name_by_id()
    keep["type_name"] = keep["type_id"].map(names)
    is_own = keep["type_id"].isin(own)
    start_x = keep["start_x"].astype(float)
    keep["consistent"] = np.where(is_own, start_x < HALFWAY_X, start_x > HALFWAY_X)
    return keep[["game_id", "team_id", "period_id", "type_name", "start_x", "consistent"]].reset_index(drop=True)


def cell_counts(anchored: pd.DataFrame) -> pd.DataFrame:
    """Per (game, team, period) cell: ``n`` anchored actions, ``k`` consistent, ``scored`` iff n >= 5."""
    if not len(anchored):
        return pd.DataFrame(columns=["game_id", "team_id", "period_id", "n", "k", "scored"])
    grouped = (
        anchored.groupby(["game_id", "team_id", "period_id"], dropna=False)["consistent"]
        .agg(n="size", k="sum")
        .reset_index()
    )
    grouped["scored"] = grouped["n"] >= MIN_CELL_N
    return grouped


# --- Step 2: reversal + boundary statistics ------------------------------------------------


def cell_null_cdf(cell_anchored: pd.DataFrame, *, p_t: Mapping[str, float]) -> float:
    """``P_null(K <= k)`` for one scored cell under the exact Poisson-binomial null.

    Each anchored action is a Bernoulli whose success probability is its type's reference
    consistency ``p_t[type]``; ``k`` is the observed consistent count. Uses the public
    :func:`silly_kicks.match_outcome.goal_count_pmf` DP (``pmf[j] = P(exactly j)``), so
    ``P(K <= k) = pmf[: k + 1].sum()``.
    """
    from silly_kicks.match_outcome import goal_count_pmf

    probs = [float(p_t[t]) for t in cell_anchored["type_name"]]
    k = int(cell_anchored["consistent"].sum())
    pmf = goal_count_pmf(probs)
    return float(pmf[: k + 1].sum())


def cell_is_reversed(cell_anchored: pd.DataFrame, *, p_t: Mapping[str, float], m: int) -> bool:
    """A scored cell is ``reversed`` iff BOTH conjuncts fire (spec section 8, one-sided mirror test):

    (a) ``k / n < 0.5`` -- the majority of the cell sits on the wrong side; and
    (b) ``P_null(K <= k) < 0.05 / m`` -- Bonferroni over ``m`` scored cells across the S1-excluded
        matches. Conjunct (a) guards against overdispersion making the binomial-type null
        anti-conservative.
    """
    n = len(cell_anchored)
    if n == 0:
        return False
    k = int(cell_anchored["consistent"].sum())
    minority = k / n < 0.5
    significant = cell_null_cdf(cell_anchored, p_t=p_t) < REVERSAL_ALPHA / m
    return bool(minority and significant)


def boundary_counts(actions: pd.DataFrame) -> tuple[int, int]:
    """``(n_open_play, n_on_boundary)`` for a match.

    Open-play = every type EXCEPT the law-fixed-origin restarts (:data:`SET_PIECE_TYPES`). An action
    start is on the boundary when it lies within :data:`BOUNDARY_EPS_M` of ``x in {0, 105}`` or
    ``y in {0, 68}`` -- the events converter's on-pitch clamp turns off-pitch coordinates into
    boundary mass, so an origin/scale break shows up as a pile-up a sound match does not have.
    """
    from silly_kicks.spadl import config as spadlconfig

    set_piece_ids = {int(spadlconfig.actiontype_id[t]) for t in SET_PIECE_TYPES}
    open_play = actions[~actions["type_id"].isin(set_piece_ids)]
    n = len(open_play)
    if n == 0:
        return (0, 0)
    x = open_play["start_x"].astype(float).to_numpy()
    y = open_play["start_y"].astype(float).to_numpy()
    on_x = (np.abs(x - 0.0) <= BOUNDARY_EPS_M) | (np.abs(x - 105.0) <= BOUNDARY_EPS_M)
    on_y = (np.abs(y - 0.0) <= BOUNDARY_EPS_M) | (np.abs(y - 68.0) <= BOUNDARY_EPS_M)
    return (int(n), int((on_x | on_y).sum()))


def boundary_tau(b_passing: list[float] | np.ndarray) -> float:
    """``tau = max(0.05, 3 * q99.9(b over S1-passing matches))`` (spec section 8).

    The floor keeps a sound corpus's near-zero pile-up from producing a hair-trigger threshold.
    """
    arr = np.asarray(list(b_passing), dtype=float)
    q999 = float(np.quantile(arr, 0.999)) if arr.size else 0.0
    return max(BOUNDARY_TAU_FLOOR, 3.0 * q999)


def match_is_boundary_flagged(b: float, tau: float) -> bool:
    """A match is ``boundary_pileup`` iff its open-play boundary fraction exceeds ``tau``."""
    return bool(b > tau)


# --- Step 3: per-match verdict precedence --------------------------------------------------

#: Regulation periods -- an unscored regulation cell is missing evidence (``insufficient``); an
#: unscored extra-time cell is allowed and merely reported.
REGULATION_PERIODS = (1, 2)


def match_event_verdict(
    match_anchored: pd.DataFrame,
    *,
    boundary_fraction: float,
    p_t: Mapping[str, float],
    tau: float,
    m: int,
) -> str:
    """One match's event verdict, in precedence order (spec section 8):

    ``reversed`` (any scored cell reversed) > ``boundary_pileup`` > ``insufficient`` (any regulation
    (team, period) cell unscored) > ``sound``. Unscored extra-time cells are allowed.
    """
    cells = cell_counts(match_anchored)
    scored = cells[cells["scored"]]
    for _, row in scored.iterrows():
        cell_rows = match_anchored[
            (match_anchored["team_id"] == row["team_id"]) & (match_anchored["period_id"] == row["period_id"])
        ]
        if cell_is_reversed(cell_rows, p_t=p_t, m=m):
            return "reversed"
    if match_is_boundary_flagged(boundary_fraction, tau):
        return "boundary_pileup"
    scored_cells = {(r["team_id"], r["period_id"]) for _, r in scored.iterrows()}
    for team in match_anchored["team_id"].dropna().unique():
        for period in REGULATION_PERIODS:
            if (team, period) not in scored_cells:
                return "insufficient"
    return "sound"


# --- Step 4: the two for_each passes -------------------------------------------------------

#: Per-match EVENT shard. ``kind="anchored"`` rows are per (team, period, type) with n/k consistent
#: counts; the single ``kind="open_play"`` row carries the boundary-pileup counts (n open-play
#: starts, k on the boundary), team/period NA. Everything the reduce needs to recompute p_t / m / tau
#: lives here -- the producer never freezes a verdict.
EVENTS_SHARD_COLUMNS = ["game_id", "team_id", "period_id", "kind", "type_name", "n", "k"]

#: Per-match TRACKING shard: the S1 off-pitch rates for an S1-PASSING match. An S1 exclusion writes a
#: marker (with the same rates in ``details``) instead, so it never has a shard.
TRACKING_SHARD_COLUMNS = ["game_id", "player_off_pitch_rate", "ball_off_pitch_rate"]


def events_statistics(actions: pd.DataFrame, *, game_id: str) -> pd.DataFrame:
    """The per-match EVENT shard (:data:`EVENTS_SHARD_COLUMNS`) from a match's SPADL actions."""
    anchored = anchored_consistency(actions)
    rows: list[dict] = []
    if len(anchored):
        grp = (
            anchored.groupby(["team_id", "period_id", "type_name"], dropna=False)["consistent"]
            .agg(n="size", k="sum")
            .reset_index()
        )
        for rec in grp.to_dict("records"):
            rows.append(
                {
                    "game_id": str(game_id),
                    "team_id": rec["team_id"],
                    "period_id": rec["period_id"],
                    "kind": "anchored",
                    "type_name": rec["type_name"],
                    "n": int(rec["n"]),
                    "k": int(rec["k"]),
                }
            )
    n_open, n_boundary = boundary_counts(actions)
    rows.append(
        {
            "game_id": str(game_id),
            "team_id": pd.NA,
            "period_id": pd.NA,
            "kind": "open_play",
            "type_name": "",
            "n": int(n_open),
            "k": int(n_boundary),
        }
    )
    return pd.DataFrame(rows, columns=EVENTS_SHARD_COLUMNS)


def _events_work(loaded: LoadedMatch) -> pd.DataFrame:
    return events_statistics(loaded.actions, game_id=loaded.match_id)


def _tracking_work(loaded: LoadedMatch) -> pd.DataFrame:
    """S1 rates for an S1-PASSING SkillCorner match (an exclusion never reaches here -- it raised)."""
    rep = loaded.report
    pr = float(getattr(rep, "player_off_pitch_rate", float("nan"))) if rep is not None else float("nan")
    br = float(getattr(rep, "ball_off_pitch_rate", float("nan"))) if rep is not None else float("nan")
    return pd.DataFrame(
        [{"game_id": str(loaded.match_id), "player_off_pitch_rate": pr, "ball_off_pitch_rate": br}],
        columns=TRACKING_SHARD_COLUMNS,
    )


def _events_pass(
    refs: list[MatchRef],
    *,
    shard_root,
    cache_dir,
    token_inputs: Mapping[str, object],
    tag: str = "all",
):
    """The events pass. This is a Rule-D allowlisted producer: it is the ONE consumer besides
    ``_events_admission.events_only_loader`` allowed to call ``load_match(events_only=True)`` without
    going through the admission layer (spec section 8 -- the producer measures every match's events)."""
    return for_each(
        refs,
        key=lambda ref: ref.key,
        work=_events_work,
        load=lambda ref: load_match(ref, events_only=True, cache_dir=cache_dir),
        shard_root=shard_root,
        token_inputs=token_inputs,
        tag=tag,
        label="events",
    )


def _tracking_pass(
    refs: list[MatchRef],
    *,
    shard_root,
    cache_dir,
    token_inputs: Mapping[str, object],
    tag: str = "all",
):
    """The tracking pass: S1's off-pitch rates for every match. An S1 exclusion is a persisted marker
    carrying the rates (``load_match`` raises ``MatchExcluded``); a load failure writes no shard, so a
    resume retries it (spec section 8 -- separate passes keep a transient error from freezing)."""
    return for_each(
        refs,
        key=lambda ref: ref.key,
        work=_tracking_work,
        load=lambda ref: load_match(ref, events_only=False, cache_dir=cache_dir),
        shard_root=shard_root,
        token_inputs=token_inputs,
        tag=tag,
        label="tracking",
    )


# --- Step 5: the reduce (recomputes verdicts every run; never reads its own artifact) --------

#: Status display order for findings (tracking-side statuses, spec section 8).
STATUSES_ORDER = ("s1_passed", "s1_excluded", "tracking_unloadable", "events_unloadable")


def _anchored_from_shard(shard: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct per-action anchored rows from a shard's ``kind="anchored"`` (team, period, type)
    n/k rows. The Poisson-binomial null is order-independent, so exact identities do not matter -- k
    of a type's n actions are consistent."""
    anchored = shard[shard["kind"] == "anchored"]
    rows: list[dict] = []
    for rec in anchored.to_dict("records"):
        n, k = int(rec["n"]), int(rec["k"])
        for i in range(n):
            rows.append(
                {
                    "game_id": rec["game_id"],
                    "team_id": rec["team_id"],
                    "period_id": rec["period_id"],
                    "type_name": rec["type_name"],
                    "consistent": i < k,
                }
            )
    return pd.DataFrame(rows, columns=["game_id", "team_id", "period_id", "type_name", "consistent"])


def _boundary_fraction_from_shard(shard: pd.DataFrame) -> float:
    """The match's open-play boundary fraction ``b`` from its single ``kind="open_play"`` row."""
    op = shard[shard["kind"] == "open_play"]
    if not len(op):
        return 0.0
    n = int(op["n"].iloc[0])
    k = int(op["k"].iloc[0])
    return (k / n) if n else 0.0


def _reference_rates(passing_anchored: pd.DataFrame) -> tuple[dict[str, float], float]:
    """``(p_t, global_rate)``: pooled anchored-type consistency over the SCORED cells of the
    S1-passing matches, plus the overall pooled rate (the fallback for a type unseen in a scored
    passing cell). ``p_t`` is completed to every anchored type so a cell null never KeyErrors."""
    if not len(passing_anchored):
        return ({t: 0.5 for t in ANCHORED_TYPES}, 0.5)
    cells = cell_counts(passing_anchored)
    scored = cells[cells["scored"]][["game_id", "team_id", "period_id"]]
    in_scored = passing_anchored.merge(scored, on=["game_id", "team_id", "period_id"], how="inner")
    if not len(in_scored):
        return ({t: 0.5 for t in ANCHORED_TYPES}, 0.5)
    global_rate = float(in_scored["consistent"].mean())
    observed = in_scored.groupby("type_name")["consistent"].mean().to_dict()
    p_t = {t: float(observed.get(t, global_rate)) for t in ANCHORED_TYPES}
    return (p_t, global_rate)


def _match_reason(status: str, verdict: str | None, b: float, error: str | None) -> str:
    if status == "events_unloadable":
        return f"events failed to load: {error}"
    if status == "tracking_unloadable":
        base = f"tracking failed to load: {error}"
        return f"{base}; event_verdict={verdict}" if verdict else base
    if verdict == "reversed":
        return "anchored actions reversed across the halfway line (mirror signature)"
    if verdict == "boundary_pileup":
        return f"open-play starts pile up on the pitch boundary (b={b:.4f})"
    if verdict == "insufficient":
        return "insufficient anchored evidence in a regulation cell"
    if verdict == "sound":
        return "sound"
    return status


def reduce_verdicts(events_res, tracking_res, *, allow_failed: bool = False) -> dict:
    """Assign every listed match a status + (where events loaded) an event verdict, RECOMPUTING
    ``p_t`` / ``tau`` / ``m`` from the shards every run (spec section 8, CDLS-SPEC-27/28).

    Refuses (``RuntimeError``) while either pass has outstanding failures unless ``allow_failed``,
    which records each such match's error and sets ``allowed_failed: true``. The producer NEVER reads
    ``verdicts.json`` -- this function references neither the artifact path nor the admission layer.
    """
    if not allow_failed and (events_res.failures or tracking_res.failures):
        parts = []
        if events_res.failures:
            parts.append(f"events pass: {sorted(events_res.failures)}")
        if tracking_res.failures:
            parts.append(f"tracking pass: {sorted(tracking_res.failures)}")
        raise RuntimeError(
            "refusing to write verdicts with outstanding load failures ("
            + "; ".join(parts)
            + "). Re-run (resumable) to retry them, or pass --allow-failed to record them as "
            "tracking_unloadable / events_unloadable."
        )

    keys = sorted(set(events_res.keys) | set(tracking_res.keys))
    events_by_key = {k: pd.read_parquet(shard_path(events_res.shard_dir, k)) for k in events_res.shard_keys}

    status: dict[str, str] = {}
    errors: dict[str, str] = {}
    rates: dict[str, tuple[float | None, float | None]] = {}
    for k in keys:
        if k in events_res.failures:
            status[k] = "events_unloadable"
            errors[k] = events_res.failures[k]
        elif k in tracking_res.failures:
            status[k] = "tracking_unloadable"
            errors[k] = tracking_res.failures[k]
        elif k in tracking_res.exclusions:
            status[k] = "s1_excluded"
            marker = read_exclusion(tracking_res.shard_dir, k) or {}
            det = marker.get("details") or {}
            rates[k] = (det.get("player_off_pitch_rate"), det.get("ball_off_pitch_rate"))
        else:
            status[k] = "s1_passed"
            trk = pd.read_parquet(shard_path(tracking_res.shard_dir, k))
            rates[k] = (float(trk["player_off_pitch_rate"].iloc[0]), float(trk["ball_off_pitch_rate"].iloc[0]))

    # Corpus reference quantities, recomputed from the shards.
    passing_keys = [k for k in keys if status[k] == "s1_passed" and k in events_by_key]
    passing_anchored = (
        pd.concat([_anchored_from_shard(events_by_key[k]) for k in passing_keys], ignore_index=True)
        if passing_keys
        else pd.DataFrame(columns=["game_id", "team_id", "period_id", "type_name", "consistent"])
    )
    p_t, _global_rate = _reference_rates(passing_anchored)
    b_passing = [_boundary_fraction_from_shard(events_by_key[k]) for k in passing_keys]
    tau = boundary_tau(b_passing)

    excluded_keys = [k for k in keys if status[k] == "s1_excluded" and k in events_by_key]
    m = 0
    for k in excluded_keys:
        cells = cell_counts(_anchored_from_shard(events_by_key[k]))
        m += int(cells["scored"].sum())
    m_eff = max(1, m)

    matches: dict[str, dict] = {}
    for k in keys:
        verdict: str | None = None
        b = 0.0
        cell_stats: list[dict] = []
        if k in events_by_key:
            anchored = _anchored_from_shard(events_by_key[k])
            b = _boundary_fraction_from_shard(events_by_key[k])
            verdict = match_event_verdict(anchored, boundary_fraction=b, p_t=p_t, tau=tau, m=m_eff)
            cells = cell_counts(anchored)
            cell_stats = [
                {
                    "team_id": str(r["team_id"]),
                    "period_id": int(r["period_id"]),
                    "n": int(r["n"]),
                    "k": int(r["k"]),
                    "scored": bool(r["scored"]),
                }
                for r in cells.to_dict("records")
            ]
        pr, br = rates.get(k, (None, None))
        rec = {
            "status": status[k],
            "event_verdict": verdict,
            "reason": _match_reason(status[k], verdict, b, errors.get(k)),
            "player_off_pitch_rate": pr,
            "ball_off_pitch_rate": br,
            "boundary_fraction": b if k in events_by_key else None,
            "cells": cell_stats,
        }
        if k in errors:
            rec["error"] = errors[k]
        matches[k] = rec

    # Empirical calibration (reported): the identical thresholds applied to S1-passing matches.
    calib = {"n": len(passing_keys), "reversed": 0, "boundary_pileup": 0, "insufficient": 0}
    for k in passing_keys:
        v = matches[k]["event_verdict"]
        if v in calib:
            calib[v] += 1

    return {
        "matches": matches,
        "digest": listing_digest(matches),
        "p_t": p_t,
        "tau": tau,
        "m": m,
        "s1_passing_calibration": calib,
        "allowed_failed": bool(allow_failed),
    }


def findings_markdown(artifact: dict) -> str:
    """A human-readable companion to verdicts.json (spec section 8)."""
    matches = artifact["matches"]
    by_status: dict[str, int] = {}
    by_verdict: dict[str, int] = {}
    for rec in matches.values():
        by_status[rec["status"]] = by_status.get(rec["status"], 0) + 1
        v = rec.get("event_verdict")
        if v is not None:
            by_verdict[v] = by_verdict.get(v, 0) + 1
    lines = ["# SkillCorner S1 event-validity findings", ""]
    lines.append(f"Matches measured: {len(matches)}")
    lines.append("")
    lines.append("## Tracking status")
    for s in STATUSES_ORDER:
        if s in by_status:
            lines.append(f"- {s}: {by_status[s]}")
    lines.append("")
    lines.append("## Event verdict")
    for v in ("sound", "insufficient", "boundary_pileup", "reversed"):
        if v in by_verdict:
            lines.append(f"- {v}: {by_verdict[v]}")
    lines.append("")
    lines.append(f"Reference: m={artifact['m']} scored excluded cells, tau={artifact['tau']:.4f}.")
    lines.append(f"S1-passing calibration (false-alarm context): {artifact['s1_passing_calibration']}.")
    lines.append("")
    lines.append("## Admitted for events-only use (status/verdict)")
    for key in sorted(matches):
        rec = matches[key]
        admitted = rec["status"] == "s1_passed" or (
            rec["status"] in ("s1_excluded", "tracking_unloadable") and rec.get("event_verdict") == "sound"
        )
        if admitted:
            lines.append(f"- {key}: {rec['status']} / {rec.get('event_verdict')}")
    lines.append("")
    return "\n".join(lines)


def input_contract() -> dict:
    """Declare WHICH SYMBOLS these numbers depend on (ADR-056). The event-side check is fixed by the
    anchored type sets, the halfway line, the cell-scoring floor and the two significance thresholds;
    the Poisson-binomial null is the public ``goal_count_pmf`` DP."""
    from _input_contract import declare_inputs

    return declare_inputs(
        driver="build_skillcorner_s1_event_validity",
        own_half_anchored=list(OWN_HALF_ANCHORED),
        attack_half_anchored=list(ATTACK_HALF_ANCHORED),
        set_piece_types=sorted(SET_PIECE_TYPES),
        halfway_x=HALFWAY_X,
        min_cell_n=MIN_CELL_N,
        reversal_alpha=REVERSAL_ALPHA,
        boundary_eps_m=BOUNDARY_EPS_M,
        boundary_tau_floor=BOUNDARY_TAU_FLOOR,
        null=("silly_kicks.match_outcome.goal_count_pmf",),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="SkillCorner S1 event-validity measurement (Task 0).")
    ap.add_argument(
        "--out-dir", default=None, help="verdicts.json / findings.md dir (default: the admission artifact dir)"
    )
    ap.add_argument(
        "--shard-root", default=None, help="per-match shard root (default: <out-dir>/shards; keep gitignored)"
    )
    ap.add_argument("--providers", default="skillcorner", help="comma-separated providers (default: skillcorner)")
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="raw-artifact cache root (default: $SILLY_KICKS_CORPUS_CACHE_DIR, else no cache)",
    )
    ap.add_argument(
        "--match-ids-json",
        default=None,
        help='JSON {"skillcorner": ["1", ...]} pinning WHICH matches this process handles (parallel split).',
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact is marked)")
    ap.add_argument(
        "--allow-failed",
        action="store_true",
        help="write verdicts despite outstanding load failures, recording each as tracking/events_unloadable",
    )
    ap.add_argument("--list-matches", action="store_true", help="print the available match ids as JSON and exit")
    args = ap.parse_args()

    from _provenance import git_provenance, require_clean_tree

    if args.list_matches:
        from _partition import list_match_ids

        print(json.dumps(list_match_ids(args.providers.split(",")), indent=2))
        return

    # Provenance FIRST, before any corpus work (ADR-037).
    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)

    from _events_admission import ADMISSION_ARTIFACT
    from _partition import providers_for_slice, worker_tag

    out_dir = Path(args.out_dir) if args.out_dir else ADMISSION_ARTIFACT.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_root = Path(args.shard_root) if args.shard_root else out_dir / "shards"
    cache_dir = resolve_cache_dir(args.cache_dir)
    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None
    tag = worker_tag(args.match_ids_json)

    refs = list_match_refs(
        providers=providers_for_slice(args.providers.split(","), match_ids),
        match_ids=match_ids,
        max_per_provider=args.max_per_provider,
    )
    # Both passes over the SAME ref list, distinct shard roots so their identical keys never collide.
    token = {"providers": sorted(args.providers.split(",")), "max_per_provider": args.max_per_provider}
    events_res = _events_pass(refs, shard_root=shard_root / "events", cache_dir=cache_dir, token_inputs=token, tag=tag)
    tracking_res = _tracking_pass(
        refs, shard_root=shard_root / "tracking", cache_dir=cache_dir, token_inputs=token, tag=tag
    )

    artifact = reduce_verdicts(events_res, tracking_res, allow_failed=args.allow_failed)
    artifact.update(
        run_commit=prov["commit"],
        run_tree_dirty=prov["dirty"],
        run_tree_state=prov.get("tree_state"),
        input_contract=input_contract(),
    )
    (out_dir / "verdicts.json").write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    (out_dir / "findings.md").write_text(findings_markdown(artifact), encoding="utf-8")
    print(json.dumps({k: v for k, v in artifact.items() if k != "matches"}, indent=2, default=str))


if __name__ == "__main__":
    main()
