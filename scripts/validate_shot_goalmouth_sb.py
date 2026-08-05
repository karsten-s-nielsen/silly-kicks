"""TF-48 owner-run validation harness: GS WC2022 tracking-derived goalmouth crossings
vs StatsBomb open-data hand-coded ``end_location`` (spec section 10; ADR-030).

Protocol (spec 2026-06-10-shot-goalmouth-psxg-design.md section 10):
  1.  Load GS matches (pining, owner token) -> add_shot_goalmouth per match.
  2.  Pull SB open data (statsbombpy; competition 43 / season 106 = men's WC2022);
      map GS<->SB matches by team names + kickoff date (pining manifest ``home`` /
      ``away`` / ``date`` fields, probed 2026-06-10).
  3.  OUTCOME-LITERAL RUNTIME ASSERT: the on-target literal set must be a subset of
      the actual statsbombpy vocabulary -- fail loud, never zero-match (spec L-4).
  4.  Shot matching GS<->SB per (match, period): tie-breaker ORDERING is
      (1) same period + same team + game-clock distance (nearest within +-10 s),
      (2) ambiguous (two candidates within 2 s of each other) -> UNMATCHED
      (reported, never best-effort matched).
  5.  HANDEDNESS SETTLEMENT BEFORE FLOORS (spec section 7 / H5; instrument REBUILT at
      round 2, ADR-030): the y-sign of the meters->SB mapping is settled on GK
      GEOMETRY -- per matched shot, the SB freeze-frame DEFENDING GK's y vs the
      GS-tracked GK's y (canonicalized via the engine's goal_x reflection). One
      well-identified object, fit-independent, no goal-mouth ball chaos; ASSERT
      agreement >= 0.8 (see _HANDEDNESS_FLOOR), else abort AFTER writing the report.
      The round-1 ball-tag vote (goal crossings vs SB end_location) is demoted to the
      informational ``handedness_ball_diag`` incl. an in-mouth plausibility split (a
      goal crossing outside the mouth is a self-evident measurement failure).
  6.  STRATIFY GOALS vs SAVES (H4): floors are calibrated/evaluated on GOALS only
      (true plane crossings); saves reported separately -- Delta(saves)-Delta(goals)
      quantifies the lakehouse PSxG train/serve shift for free.
  7.  Sensitivity (--sweep, pilot only): break_residual_m x {0.5, 0.75, 1.0, 1.5} on a
      10 fps-downsampled copy (every 3rd ball frame) of each match, plus the module
      constants invisible to the params surface (_REFINE_SPEED_JUMP_MS x {2, 3, 5} --
      patched on the module, documented).
  8.  Raw-z vs smoothed-z (--z-compare, pilot only): rebuilds ball z from the cached
      GS tracking artifact's ``ballsSmoothed`` records, re-runs the enrichment, and
      reports crossing deltas. Conclusion is relayed cross-session (the lakehouse
      adapter z-source switch is conditional on it -- spec section 2). Requires
      --cache-dir (the artifact must be on disk).
  9.  Per-shot diagnostics (--debug-shots; pilot AND the one-shot holdout -- the
      holdout may not be re-run, so its failure analysis must be capturable in the
      single pass; the tracer is verified byte-identical to the untraced run):
      records, for every MATCHED
      shot, the FULL-resolution ball window series + the kernel's fit internals
      (flight-run anchor, refined t0, segment growth + end reason, flight-core trim,
      z taxonomy) + the SB reference + the post-handedness dy/dz, all in the main
      --out artifact under ``debug_shots`` (sorted worst-|dy| first). Captured by
      WRAPPING the real kernel functions during the enrichment call itself -- zero
      replay drift (a separate replay-diag script went stale mid-loop on the first
      pilot and masked the plateau; consolidated here 2026-06-11).

Pilot subset: STRATIFIED 12 group-stage + 4 knockout GS matches (knockout = manifest
``date`` >= 2022-12-03, the first R16 day; GS match ids do NOT ascend with the
schedule -- probed 2026-06-10, ids span 3812..10517). Within each stratum: ascending
(date, id) order, first N. Deterministic.

Usage (DGX, owner token in PINING_FOR_THE_DATA_TOKEN):
    python scripts/validate_shot_goalmouth_sb.py --matches pilot --sweep --z-compare \
        --cache-dir ~/pining_cache --out pilot.json
    python scripts/validate_shot_goalmouth_sb.py --matches holdout --cache-dir ~/pining_cache \
        --out holdout.json
"""

from __future__ import annotations

import argparse
import bz2
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _loader_pining import _base_url, _list_matches, _resolve_token, load_matches

_SB_COMPETITION_ID = 43  # FIFA World Cup
_SB_SEASON_ID = 106  # 2022
_M_PER_YD = 0.9144
# On-target vocabulary -- ASSERTED against the live statsbombpy vocabulary at runtime
# (step 3); never trusted from spec text. 'Saved to Post' casing is LIVE-VERIFIED
# (2026-06-11 pilot: the spec-text guess 'Saved To Post' tripped the L-4 guard).
_SB_ON_TARGET = {"Goal", "Saved", "Saved to Post"}
_CLOCK_TOL_S = 10.0
_AMBIGUITY_GAP_S = 2.0
# Relaxed 0.9 -> 0.8 at floor registration (ADR-030, 2026-06-11): the gate's two hypotheses are
# separable -- a wrong sign reads ~0.1, frontier hand-tag noise reads 0.8-0.95 (the pilot's one
# dissenter is a measured GS-vs-SB sides-flip on a near-centre crossing, GS-corroborated). 0.8
# still discriminates; 0.9 would abort the one-shot holdout on a predictable false positive.
_HANDEDNESS_FLOOR = 0.8
_KNOCKOUT_FROM_DATE = "2022-12-03"  # first R16 day
_N_PILOT_GROUP, _N_PILOT_KNOCKOUT = 12, 4


def _pilot_holdout_split(manifest: list[dict]) -> tuple[list[str], list[str]]:
    """Stratified deterministic split per the module docstring."""
    rows = sorted(manifest, key=lambda m: (m["date"], int(m["id"])))
    group = [m["id"] for m in rows if m["date"] < _KNOCKOUT_FROM_DATE]
    knockout = [m["id"] for m in rows if m["date"] >= _KNOCKOUT_FROM_DATE]
    pilot = group[:_N_PILOT_GROUP] + knockout[:_N_PILOT_KNOCKOUT]
    holdout = [m["id"] for m in rows if m["id"] not in set(pilot)]
    return pilot, holdout


def _check_outcome_vocabulary(sb_shots: pd.DataFrame, seen_vocab: set) -> None:
    """Per-match: accumulate observed outcomes + HARD-FAIL on a case-insensitive
    near-miss of our literals (wrong casing would silently zero-match -- spec L-4).
    A rare literal (e.g. 'Saved To Post') legitimately won't appear in every match;
    the corpus-level subset check runs once at the end (`_assert_vocab_corpus`)."""
    vocab = set(sb_shots["shot_outcome"].dropna().unique())
    seen_vocab |= vocab
    ours_lower = {s.lower(): s for s in _SB_ON_TARGET}
    for v in vocab:
        ours = ours_lower.get(str(v).lower())
        if ours is not None and ours != v:
            raise AssertionError(
                f"SB outcome literal casing mismatch: live {v!r} vs ours {ours!r} -- "
                "fix _SB_ON_TARGET (spec L-4: a casing drift would silently zero-match)"
            )


def _assert_vocab_corpus(seen_vocab: set) -> None:
    """End-of-run: the workhorse literals MUST have been observed (a 16+-match corpus
    without any Goal/Saved means the join/literals are broken, not the data)."""
    required = {"Goal", "Saved"}
    missing = required - seen_vocab
    if missing:
        raise AssertionError(
            f"corpus-level on-target literals never observed: {sorted(missing)} "
            f"(observed vocabulary: {sorted(seen_vocab)}) -- spec L-4"
        )


def _retry(fn, attempts: int = 4, base_sleep: float = 5.0) -> Any:
    """Ride out transient network blips (DNS, resets) on the ~17 SB open-data calls a
    full run makes -- a 90-minute pilot should not die to one failed resolution."""
    import time

    for k in range(attempts):
        try:
            return fn()
        except Exception:
            if k == attempts - 1:
                raise
            time.sleep(base_sleep * (2**k))
    raise AssertionError("unreachable: the final attempt either returned or raised")


def _sb_shots_for(sb, sb_match_id: int, seen_vocab: set) -> pd.DataFrame:
    evts = _retry(lambda: sb.events(match_id=sb_match_id))
    shots = evts[evts["type"] == "Shot"].copy()
    _check_outcome_vocabulary(shots, seen_vocab)
    # SB minutes are cumulative-regulation per period (45*(p-1) base; ET 15-min blocks
    # continue at 90/105) -- EXACTLY the GS SPADL cumulative time base (loader-probed
    # 2026-06-11: GS P2 shots/frames sit at 2700+rel and matched SB cumulative TO THE
    # SECOND, e.g. 5647<->2947+2700). Use the cumulative clock UNCONVERTED on both
    # sides. The original period-relative conversion silently no_candidate'd ~ALL
    # period-2 shots (681/704 on the first holdout -- every pilot metric was P1-only)
    # and let late-P2 SB stoppage shots spuriously match early-GS-P2 ones.
    shots["_clock_s"] = shots["minute"] * 60 + shots["second"]
    end = shots["shot_end_location"]
    shots["_end_y_sb"] = end.map(lambda v: v[1] if isinstance(v, list) and len(v) >= 2 else np.nan)
    shots["_end_z_sb"] = end.map(lambda v: v[2] if isinstance(v, list) and len(v) >= 3 else np.nan)
    return shots


def _match_shots(gs_shots: pd.DataFrame, sb_shots: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Tie-breaker ordering per module docstring. Returns (matched, unmatched). Unmatched
    rows carry characterization fields (gs_type/gs_result/gs_period + nearest-SB clock
    distance / candidate spread) so the failure mix is auditable without a re-run."""
    matched, unmatched = [], []
    for _, g in gs_shots.iterrows():
        same = sb_shots[(sb_shots["period"] == g["period_id"]) & (sb_shots["_team_gs_id"] == g["team_id"])]
        info = {
            "gs_action_id": int(g["action_id"]),
            "gs_type": str(g.get("type_name", "")),
            "gs_result": str(g.get("result_name", "")),
            "gs_period": int(g["period_id"]),
        }
        cand = same[(same["_clock_s"] - g["time_seconds"]).abs() <= _CLOCK_TOL_S].copy()
        if cand.empty:
            dt_all = (same["_clock_s"] - g["time_seconds"]).abs()
            info["reason"] = "no_candidate"
            info["n_sb_same_team_period"] = len(same)
            info["nearest_sb_dt_s"] = round(float(dt_all.min()), 2) if len(dt_all) else None
            unmatched.append(info)
            continue
        cand["_d"] = (cand["_clock_s"] - g["time_seconds"]).abs()
        cand = cand.sort_values("_d")
        if len(cand) > 1 and float(cand["_d"].iloc[1] - cand["_d"].iloc[0]) < _AMBIGUITY_GAP_S:
            info["reason"] = "ambiguous"
            info["n_candidates"] = len(cand)
            info["cand_dts_s"] = [round(float(v), 2) for v in cand["_d"].to_numpy()[:3]]
            unmatched.append(info)
            continue
        matched.append({"gs": g, "sb": cand.iloc[0]})
    return matched, unmatched


_SIDE_MIN_M = 1.0  # only CLEARLY-sided voters count (near-centre side-signs are noise)
_SIDE_MIN_YD = 1.0
_GK_SIDE_MIN_M = 0.5  # GKs hug the centre; their positions are far less noisy than
_GK_SIDE_MIN_YD = 0.5  # goal-mouth ball tags, so a tighter clearly-sided bar is sound
_GK_TIME_TOL_S = 1.0  # GS GK sample must sit within this of the shot stamp
_MIN_SIDE_VOTES = 3
_GOAL_MOUTH_PLAUSIBLE_M = 3.66 + 0.11  # half-mouth + ball radius


def _settle_votes(votes: list[int], *, strict: bool) -> tuple[int, dict]:
    """Majority sign over flip(+1)/keep(-1) votes; assert >= _HANDEDNESS_FLOOR agreement.
    Returns (+1 for the FLIP mapping -- the expected answer per
    silly_kicks/spadl/statsbomb.py's inversion -- or -1, diagnostics dict). With fewer
    than _MIN_SIDE_VOTES voters, falls back to the in-repo converter-derived sign (+1)
    and RECORDS the fallback (never silent)."""
    diag: dict[str, object] = {
        "n_side_votes": len(votes),
        "votes_flip": votes.count(1),
        "votes_keep": votes.count(-1),
    }
    if len(votes) < _MIN_SIDE_VOTES:
        diag["fallback"] = "in-repo converter sign (+1 flip); too few clearly-sided voters"
        return 1, diag
    sign = 1 if sum(votes) > 0 else -1
    agreement = votes.count(sign) / len(votes)
    diag["agreement"] = round(agreement, 3)
    if agreement < _HANDEDNESS_FLOOR:
        # the FLOOR run (holdout) hard-aborts (spec H5) -- but the abort is RAISED BY
        # THE CALLER **after** the full report (incl. --debug-shots) is written: the
        # first one-shot holdout (2026-06-11) raised here pre-write and destroyed its
        # own failure-analysis data ("abort with a report" was always the contract).
        diag["below_floor"] = True
        diag["strict_abort_pending"] = bool(strict)
    return sign, diag


def _extract_sb_gk_y(sb_row) -> float:
    """SB shot freeze-frame -> the DEFENDING GK's y (SB attacking-frame yards), or NaN."""
    ff = sb_row.get("shot_freeze_frame")
    if not isinstance(ff, list):
        return float("nan")
    for p in ff:
        if not p.get("teammate") and (p.get("position") or {}).get("name") == "Goalkeeper":
            loc = p.get("location")
            if isinstance(loc, list) and len(loc) >= 2:
                return float(loc[1])
    return float("nan")


def _gs_gk_y_canonical(g, gk_frames: pd.DataFrame, goal_map: dict) -> float:
    """GS-tracked DEFENDING-GK y at the shot stamp, canonicalized to the shooter's
    attacked-at-105 frame (the engine's own goal_x reflection), or NaN."""
    key = (g["game_id"], g["period_id"])
    ends = {k[2]: v for k, v in goal_map.items() if (k[0], k[1]) == key}
    opp = [(tid, v) for tid, v in ends.items() if str(tid) != str(g["team_id"])]
    if len(opp) != 1:
        return float("nan")
    tid, goal_x = opp[0]
    sub = gk_frames[(gk_frames["period_id"] == g["period_id"]) & (gk_frames["team_id"].astype(str) == str(tid))]
    if sub.empty:
        return float("nan")
    dt = (sub["time_seconds"] - float(g["time_seconds"])).abs()
    i = dt.idxmin()
    if float(dt.loc[i]) > _GK_TIME_TOL_S:
        return float("nan")
    yv = float(sub.loc[i, "y"])
    return yv if float(goal_x) > 50.0 else 68.0 - yv


def _vote(gs_off: float, sb_off: float, min_m: float, min_yd: float) -> int | None:
    """flip(+1)/keep(-1) when clearly-sided on BOTH sources, else None."""
    if not (np.isfinite(gs_off) and np.isfinite(sb_off)):
        return None
    if abs(gs_off) < min_m or abs(sb_off) < min_yd:
        return None
    return 1 if np.sign(gs_off) != np.sign(sb_off) else -1


def _ball_handedness_diag(goal_rows: pd.DataFrame) -> dict:
    """INFORMATIONAL ball-tag diag (the round-1 gate input, demoted at round 2 --
    ADR-030): near-centre goal-mouth ball tags proved too noisy to settle a transform
    (in-mouth-filtered agreement 0.75-0.79 on BOTH pilot and holdout-r1, with several
    dissenters agreeing with SB exactly under keep). Also splits by PLAUSIBILITY: a
    GOAL whose derived crossing is outside the mouth is a self-evident measurement
    failure (flaggable without SB) and cannot be transform evidence."""
    votes_all: list[int] = []
    votes_plausible: list[int] = []
    for r in goal_rows.to_dict("records"):
        cy = float(r["crossing_y_m"]) if r["crossing_y_m"] is not None else float("nan")
        sy = float(r["sb_end_y"]) if r["sb_end_y"] is not None else float("nan")
        v = _vote(cy - 34.0, sy - 40.0, _SIDE_MIN_M, _SIDE_MIN_YD)
        if v is None:
            continue
        votes_all.append(v)
        if abs(cy - 34.0) <= _GOAL_MOUTH_PLAUSIBLE_M:
            votes_plausible.append(v)
    return {
        "n_votes": len(votes_all),
        "votes_flip": votes_all.count(1),
        "votes_keep": votes_all.count(-1),
        "in_mouth_n": len(votes_plausible),
        "in_mouth_flip": votes_plausible.count(1),
        "in_mouth_keep": votes_plausible.count(-1),
    }


def _delta_stats(sub: pd.DataFrame) -> dict:
    return {
        "n": len(sub),
        "dy_median_m": float(sub["dy"].median()) if len(sub) else None,
        "dy_p90_m": float(sub["dy"].quantile(0.9)) if len(sub) else None,
        "dz_median_m": float(sub["dz"].dropna().median()) if sub["dz"].notna().any() else None,
        "dz_p90_m": float(sub["dz"].dropna().quantile(0.9)) if sub["dz"].notna().any() else None,
    }


# Hand-set kernel module-constant sensitivity grid (spec 10.4 / plan M2; SK-S93). Swept on
# the 10 fps-downsampled copy where frame-rate stress is worst -- a 30 m/s ball moves ~3 m per
# 10 fps frame, so the contact-existence radius is the prime spurious-failure risk. Each is
# monkeypatched + restored; source_counts record the effect (a robust constant moves counts
# little across its row). Includes the v7 contact/flight constants + the SK-S93 additions
# (_FLIGHT_REACH_M earliest-reaching tie-break; the quad chip-curl signal floor).
_SWEEP_CONSTANTS: dict[str, tuple[float, ...]] = {
    "_CONTACT_EXIST_RADIUS_M": (3.0, 5.0, 8.0),
    "_CONTACT_RADIUS_M": (1.5, 2.0, 3.0),
    "_CONTACT_MAX_Z_M": (2.0, 2.6, 3.5),
    "_LOCAL_FIT_WINDOW_S": (0.3, 0.4, 0.6),
    "_FLIGHT_MIN_APPROACH_MS": (5.0, 7.0, 9.0),
    "_FLIGHT_REACH_M": (1.0, 2.0, 3.0),
    "_QUAD_MIN_LIN_RMSE_M": (0.35, 0.5, 0.8),
}


def _downsample_ball_to_10fps(frames: pd.DataFrame) -> pd.DataFrame:
    """SkillCorner-rate simulation: keep every 3rd BALL frame (29.97 -> ~10 fps);
    player rows untouched (only ball samples feed the fit)."""
    is_ball = frames["is_ball"].astype(bool)
    ball = frames[is_ball].sort_values(["period_id", "frame_id"])
    kept = ball.iloc[::3]
    return pd.concat([frames[~is_ball], kept], ignore_index=True)


def _smoothed_ball_z(cache_dir: Path, match_id: str, frames: pd.DataFrame) -> pd.DataFrame | None:
    """Substitute ball z with the raw artifact's ``ballsSmoothed.z`` (probe 2026-06-10:
    a dict {visibility, x, y, z} per frame record). Returns None when the cached
    tracking artifact is absent."""
    tdir = cache_dir / "gradientsports" / str(match_id)
    # cached artifacts are extension-less role files: gradientsports_<id>_tracking
    cands = list(tdir.glob("*tracking*")) if tdir.exists() else []
    if not cands:
        return None
    raw = cands[0].read_bytes()
    text = bz2.decompress(raw).decode("utf-8") if raw[:2] == b"BZ" else raw.decode("utf-8")
    zmap: dict[tuple[int, int], float] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fr = json.loads(line)
        bs = fr.get("ballsSmoothed")
        if isinstance(bs, dict) and bs.get("z") is not None:
            zmap[(int(fr["period"]), int(fr["frameNum"]))] = float(bs["z"])
    out = frames.copy()
    is_ball = out["is_ball"].astype(bool)
    keys = list(zip(out.loc[is_ball, "period_id"], out.loc[is_ball, "frame_id"], strict=True))
    out.loc[is_ball, "z"] = [zmap.get((int(p), int(f)), np.nan) for p, f in keys]
    return out


def _jsonable(v):
    """Kernel result values -> JSON-safe (NaN/inf -> None; numpy scalars -> python)."""
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _summarize_internal(name: str, args: tuple, result) -> dict:
    """One JSON-safe record per traced kernel-helper call (see _KernelTracer)."""
    if name == "_find_flight_run":
        t = np.asarray(args[0], float)
        out: dict = {"run_start": None if result is None else int(result)}
        if result is not None:
            out["t_at_run_start"] = round(float(t[int(result)]), 3)
        return out
    if name == "_refine_contact":
        return {"t0": round(float(result), 3)}
    if name == "_grow_segment":
        end, reason, straddle = result
        return {
            "end": int(end),
            "reason": reason,
            "straddle": None if straddle is None else [int(straddle[0]), int(straddle[1])],
        }
    if name == "_trim_to_flight_core":
        ts_in, ts_out = np.asarray(args[0], float), np.asarray(result[0], float)
        return {
            "n_in": len(ts_in),
            "n_out": len(ts_out),
            "t_span_in": [round(float(ts_in[0]), 3), round(float(ts_in[-1]), 3)] if len(ts_in) else None,
            # NOTE: the trim re-zeroes its output times to the kept head sample
            "t_span_out": [round(float(ts_out[0]), 3), round(float(ts_out[-1]), 3)] if len(ts_out) else None,
        }
    if name == "_classify_z":
        profile, zstart = result
        return {"profile": profile, "zstart": int(zstart)}
    if name == "_collapse_held_samples":
        return {"n_in": len(args[0]), "n_out": len(result[0])}
    if name == "_z_onset":
        return {"onset": int(result)}
    return {"result": repr(result)}


class _KernelTracer:
    """--debug-shots capture: wraps the REAL ``_shot_goalmouth`` kernel functions for
    the duration of the main ``add_shot_goalmouth`` call, so the recorded internals are
    byte-identical to what produced the reported numbers (zero replay drift -- module
    docstring step 9). ``_fit_one_shot`` resolves its helpers via module globals at
    call time, so patching the module attributes intercepts the unmodified kernel.

    One record per ``_fit_one_shot`` call, in engine iteration order (= action order
    of the shot rows whose source is NOT unresolved/no_ball_frames -- the only two
    sources assigned without calling the kernel); the caller asserts that count.
    """

    _INNER = (
        "_collapse_held_samples",
        "_find_flight_run",
        "_refine_contact",
        "_grow_segment",
        "_trim_to_flight_core",
        "_classify_z",
        "_z_onset",
    )

    def __init__(self, sgm_module) -> None:
        self._sgm = sgm_module
        self.records: list[dict] = []
        self._current: dict = {}

    def __enter__(self) -> _KernelTracer:
        m = self._sgm
        self._orig = {n: getattr(m, n) for n in self._INNER}
        self._orig_fit = m._fit_one_shot
        for n in self._INNER:
            setattr(m, n, self._wrap_inner(n))
        m._fit_one_shot = self._wrap_fit()
        return self

    def __exit__(self, *exc) -> bool:
        for n, fn in self._orig.items():
            setattr(self._sgm, n, fn)
        self._sgm._fit_one_shot = self._orig_fit
        return False

    def _wrap_inner(self, name: str):
        fn = self._orig[name]
        cur = self._current

        def wrapped(*a, **k):
            r = fn(*a, **k)
            cur.setdefault(name, []).append(_summarize_internal(name, a, r))
            return r

        return wrapped

    def _wrap_fit(self):
        orig, cur, records, sgm = self._orig_fit, self._current, self.records, self._sgm

        def wrapped(t, x, y, z, *, goal_x, params, window_truncated=False, **kw):
            cur.clear()
            r = orig(t, x, y, z, goal_x=goal_x, params=params, window_truncated=window_truncated, **kw)
            # convenience t0 (derived the same way the kernel derives it; the raw
            # anchor records are alongside in ``internals`` if this ever disagrees)
            ffr = cur.get("_find_flight_run", [])
            rc = cur.get("_refine_contact", [])
            if ffr and ffr[0]["run_start"] is not None:
                t0 = ffr[0]["t_at_run_start"] + (sgm._MIN_VEL_BASELINE_S if ffr[0]["run_start"] > 0 else 0.0)
            elif rc:
                t0 = rc[0]["t0"]
            else:
                t0 = 0.0
            tv, xv, yv, zv = (np.asarray(v, float) for v in (t, x, y, z))
            records.append(
                {
                    "goal_x": float(goal_x),
                    "window_truncated": bool(window_truncated),
                    "contact_xy": [round(float(v), 2) for v in kw["contact_xy"]] if kw.get("contact_xy") else None,
                    "t0": round(float(t0), 3),
                    "series": {
                        "t": [round(float(v), 3) for v in tv],
                        "x": [round(float(v), 2) for v in xv],
                        "y": [round(float(v), 2) for v in yv],
                        "z": [round(float(v), 2) if np.isfinite(v) else None for v in zv],
                    },
                    "internals": dict(cur),
                    "result": {k: _jsonable(v) for k, v in r.items()},
                }
            )
            return r

        return wrapped


def _json_default(v):
    """The ONE conversion a match bundle is allowed to make on its way to a shard.

    Deliberately not ``default=str``, which the final report uses and which is wrong HERE: it
    renders ``pd.NA`` as the string ``"<NA>"``, and the debug block asks
    ``pd.isna(r["on_target_derived"])`` -- True for NA, False for the string -- so a shot with an
    unknown on-target verdict would come back from a shard reported as a MISS. Missing has to
    round-trip as ``None``. Anything else unknown RAISES here rather than being stringified into a
    plausible-looking value, which is the same reason `_settle_votes` records its fallback.
    """
    if isinstance(v, np.generic):
        return v.item()
    if v is pd.NA or v is pd.NaT:
        return None
    raise TypeError(f"{type(v).__name__} is not JSON-native and has no declared shard encoding: {v!r}")


def _shard_root(out_path: str) -> Path:
    """``--out`` is a FILE here (unlike the ``--out DIR`` drivers), so the shards get a sibling."""
    out = Path(out_path)
    return out.parent / f"{out.stem}_shards"


def run(
    matches: str,
    out_path: str,
    cache_dir: str | None,
    tracking_limit: int | None,
    *,
    sweep: bool = False,
    z_compare: bool = False,
    debug_shots: bool = False,
    provenance: dict | None = None,
) -> dict:
    from statsbombpy import sb  # type: ignore[import-not-found]  # importorskip-guarded optional dep

    import silly_kicks.tracking._shot_goalmouth as sgm
    from scripts._driver import for_each, shard_path
    from silly_kicks.spadl import config as spadlconfig
    from silly_kicks.tracking._gk_geometry import _truthy_bool
    from silly_kicks.tracking._gk_resolve import defended_goal_x
    from silly_kicks.tracking._shot_goalmouth import ShotGoalmouthParams, ShotGoalmouthReport
    from silly_kicks.tracking.features import add_shot_goalmouth

    sb_matches = _retry(lambda: sb.matches(competition_id=_SB_COMPETITION_ID, season_id=_SB_SEASON_ID))
    if len(sb_matches) == 0:
        raise AssertionError("statsbombpy returned 0 WC2022 matches -- open data unavailable?")
    sb_matches["_key"] = (
        sb_matches["home_team"].str.lower().str.strip() + "|" + sb_matches["away_team"].str.lower().str.strip()
    )

    tok = _resolve_token(None)
    manifest = _list_matches("gradientsports", tok, _base_url())
    by_id = {m["id"]: m for m in manifest}
    pilot, holdout = _pilot_holdout_split(manifest)
    wanted = {
        "pilot": pilot,
        "holdout": holdout,
        "all": sorted(by_id, key=lambda s: int(s)),
    }[matches]

    shot_ids = [spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick", "shot_penalty")]

    def _work(item) -> pd.DataFrame:
        """One match -> its whole contribution, as a ONE-ROW ``(match_id, payload)`` frame.

        WHY A JSON PAYLOAD RATHER THAN A TIDY TABLE. This match loop produces SIX heterogeneous
        outputs -- matched-shot rows, unmatched rows, a per-match report, sweep rows, a z-compare
        row, and the deeply-nested ``--debug-shots`` kernel capture -- and `for_each`'s contract is
        one tidy frame per item plus SUMMABLE counters. None of the five side outputs is summable
        (they are per-match records, not totals), and the debug capture is not tabular at all.
        Carrying only the shot rows and rebuilding the rest in memory was the alternative, and it
        is the shape that fails silently: a resumed pass would write a report whose
        ``per_match_reports`` / ``unmatched`` / ``sweep`` / ``debug_shots`` covered only the
        matches this pass happened to redo, while looking exactly like a full run. (It would also
        hard-fail on a spurious L-4 violation, since ``seen_vocab`` is filled here.)

        So the shard's tidy unit is the MATCH, and its payload is the bundle. Nothing here is lost
        on resume; the encoding is the one place that has to be exact, which is `_json_default`.
        """
        _provider, match_id, actions, frames, home_id = item
        rows: list[dict] = []
        unmatched_all: list[dict] = []
        sweep_rows: list[dict] = []
        zcmp_rows: list[dict] = []
        debug_store: dict[str, dict] = {}
        seen_vocab: set = set()
        if debug_shots:
            with _KernelTracer(sgm) as tracer:
                enriched = add_shot_goalmouth(actions, frames)
        else:
            enriched = add_shot_goalmouth(actions, frames)
        gs_shots = enriched[enriched["type_id"].isin(shot_ids)]
        report = dict(ShotGoalmouthReport.from_frame(gs_shots).__dict__)

        def _bundle() -> pd.DataFrame:
            """This match's whole contribution, ready to shard. Called at every exit."""
            return pd.DataFrame(
                [
                    {
                        "match_id": match_id,
                        "payload": json.dumps(
                            {
                                "rows": rows,
                                "report": report,
                                "unmatched": unmatched_all,
                                "sweep": sweep_rows,
                                "zcmp": zcmp_rows,
                                "debug": debug_store,
                                # Rides the bundle because `_assert_vocab_corpus` reads it AFTER the
                                # walk: a fully resumed run would otherwise observe an empty
                                # vocabulary and abort on a spurious L-4 violation.
                                "vocab": sorted(seen_vocab),
                            },
                            default=_json_default,
                        ),
                    }
                ]
            )

        if debug_shots:
            called = gs_shots[~gs_shots["shot_crossing_source"].isin(["unresolved", "no_ball_frames"])]
            if len(called) != len(tracer.records):
                raise AssertionError(
                    f"debug tracer mapping mismatch on match {match_id}: {len(called)} kernel-called "
                    f"shots vs {len(tracer.records)} traced calls -- the source-based mapping "
                    "contract in _KernelTracer's docstring no longer holds"
                )
            # Keyed by action id ALONE (as a string): the match is already the shard's identity, and
            # a JSON object cannot take the old `(match_id, action_id)` tuple as a key.
            for aid, rec in zip(called["action_id"], tracer.records, strict=True):
                debug_store[str(int(aid))] = rec

        man = by_id[match_id]
        ht, at = man["home"].lower().strip(), man["away"].lower().strip()
        cand = sb_matches[sb_matches["_key"] == f"{ht}|{at}"]
        if cand.empty:
            unmatched_all.append({"match_id": match_id, "reason": "no_sb_match_mapping", "key": f"{ht}|{at}"})
            return _bundle()
        sb_shots = _sb_shots_for(sb, int(cand.iloc[0]["match_id"]), seen_vocab)
        # team name -> GS team id: home from the loader; away = the other id among the
        # SHOT rows (shooters are always real teams -- the full actions frame can carry a
        # fillna-sentinel team_id=0 on a few rows, measured 10/1400 on match 10502, which
        # broke the old all-actions derivation). Fail loud if still ambiguous.
        shot_team_ids = {v for v in gs_shots["team_id"].dropna().unique()}
        away_ids = [t for t in shot_team_ids if str(t) != str(home_id)]
        if len(away_ids) != 1:
            unmatched_all.append(
                {
                    "match_id": match_id,
                    "reason": "away_id_unresolved",
                    "shot_team_ids": [str(t) for t in sorted(shot_team_ids)],
                }
            )
            return _bundle()
        name_to_gs = {ht: type(gs_shots["team_id"].iloc[0])(home_id), at: away_ids[0]}
        sb_shots["_team_gs_id"] = sb_shots["team"].str.lower().str.strip().map(name_to_gs)

        matched, unmatched = _match_shots(gs_shots, sb_shots)
        unmatched_all.extend({**u, "match_id": match_id} for u in unmatched)
        # GK-geometry handedness inputs (round-2 instrument, ADR-030): the defending
        # GK's tracked y vs the SB freeze-frame GK -- per matched shot, fit-independent
        goal_map = defended_goal_x(frames)
        gk_frames = frames[_truthy_bool(frames["is_goalkeeper"])]
        for m in matched:
            rows.append(
                {
                    "match_id": match_id,
                    "gk_y_m": _gs_gk_y_canonical(m["gs"], gk_frames, goal_map),
                    "sb_gk_y": _extract_sb_gk_y(m["sb"]),
                    "gs_action_id": int(m["gs"]["action_id"]),
                    "sb_outcome": m["sb"]["shot_outcome"],
                    "on_target_sb": m["sb"]["shot_outcome"] in _SB_ON_TARGET,
                    "source": m["gs"]["shot_crossing_source"],
                    "z_profile": m["gs"]["shot_z_profile"],
                    "crossing_y_m": m["gs"]["shot_crossing_y"],
                    "crossing_z_m": m["gs"]["shot_crossing_z"],
                    "on_target_derived": m["gs"]["shot_on_target_derived"],
                    "sb_end_y": m["sb"]["_end_y_sb"],
                    "sb_end_z": m["sb"]["_end_z_sb"],
                    "is_goal": m["sb"]["shot_outcome"] == "Goal",
                    # shooter context for --debug-shots inspection (harmless df extras)
                    "gs_time_seconds": float(m["gs"]["time_seconds"]),
                    "gs_start_x": float(m["gs"]["start_x"]),
                    "gs_start_y": float(m["gs"]["start_y"]),
                    "gs_period": int(m["gs"]["period_id"]),
                    "sb_clock_s": float(m["sb"]["_clock_s"]),
                }
            )

        if sweep:  # pilot-only: per-frame-rate sensitivity (spec section 10.4 / plan M2)
            ds = _downsample_ball_to_10fps(frames)
            for br in (0.5, 0.75, 1.0, 1.5):
                p = ShotGoalmouthParams(break_residual_m=br)
                e_ds = add_shot_goalmouth(actions, ds, params=p)
                rep = ShotGoalmouthReport.from_frame(e_ds[e_ds["type_id"].isin(shot_ids)])
                sweep_rows.append(
                    {"match_id": match_id, "axis": "break_residual_m@10fps", "value": br, **rep.source_counts}
                )
            for jump in (2.0, 3.0, 5.0):  # module constant, patched (documented in spec/plan)
                orig = sgm._REFINE_SPEED_JUMP_MS
                try:
                    sgm._REFINE_SPEED_JUMP_MS = jump
                    e_j = add_shot_goalmouth(actions, frames)
                    rep = ShotGoalmouthReport.from_frame(e_j[e_j["type_id"].isin(shot_ids)])
                finally:
                    sgm._REFINE_SPEED_JUMP_MS = orig
                sweep_rows.append(
                    {"match_id": match_id, "axis": "_REFINE_SPEED_JUMP_MS", "value": jump, **rep.source_counts}
                )
            # v7 + SK-S93 module-constant sensitivity, swept on the 10 fps copy (frame-rate stress)
            for name, values in _SWEEP_CONSTANTS.items():
                orig = getattr(sgm, name)
                for v in values:
                    try:
                        setattr(sgm, name, v)
                        e_c = add_shot_goalmouth(actions, ds, params=ShotGoalmouthParams())
                        rep = ShotGoalmouthReport.from_frame(e_c[e_c["type_id"].isin(shot_ids)])
                    finally:
                        setattr(sgm, name, orig)
                    sweep_rows.append({"match_id": match_id, "axis": f"{name}@10fps", "value": v, **rep.source_counts})

        if z_compare and cache_dir:  # pilot-only: raw-z vs smoothed-z (spec sections 1/10.4)
            sm = _smoothed_ball_z(Path(cache_dir), match_id, frames)
            if sm is not None:
                e2 = add_shot_goalmouth(actions, sm)
                a = gs_shots.set_index("action_id")["shot_crossing_z"]
                b = e2[e2["type_id"].isin(shot_ids)].set_index("action_id")["shot_crossing_z"]
                dz = (a - b).abs().dropna()
                zcmp_rows.append(
                    {
                        "match_id": match_id,
                        "n": len(dz),
                        "dz_median_m": float(dz.median()) if len(dz) else None,
                        "dz_max_m": float(dz.max()) if len(dz) else None,
                    }
                )
        return _bundle()

    res = for_each(
        load_matches(
            providers=["gradientsports"],
            match_ids={"gradientsports": list(wanted)},
            tracking_limit=tracking_limit,
            cache_dir=cache_dir,
        ),
        key=lambda item: (str(item[0]), str(item[1])),
        work=_work,
        shard_root=_shard_root(out_path),
        # What determines a bundle's CONTENT. The three optional passes are declared because each
        # ADDS records to the bundle: without them, a `--sweep` run over a directory built without
        # it would reuse those shards and report an EMPTY sweep -- a missing measurement that reads
        # as a completed one. `--tracking-limit` is declared for the same reason (it truncates the
        # frames the fit sees, so a dev smoke must never be mistaken for a real run), and
        # `z_compare` is declared as the pass that ACTUALLY ran, since it is silently inert
        # without --cache-dir.
        #
        # `--matches` (pilot/holdout/all) is deliberately NOT declared: it selects WHICH matches
        # are walked, and the key already separates them, so a holdout run reuses a pilot match's
        # shard rather than re-fetching it. That is also why the combine below reads this pass's
        # own keys instead of `_driver.reconcile` -- see its precondition.
        token_inputs={
            "enrichment": "add_shot_goalmouth",
            "sb_competition": _SB_COMPETITION_ID,
            "sb_season": _SB_SEASON_ID,
            "clock_tol_s": _CLOCK_TOL_S,
            "ambiguity_gap_s": _AMBIGUITY_GAP_S,
            "sweep": bool(sweep),
            "z_compare": bool(z_compare and cache_dir),
            "debug_shots": bool(debug_shots),
            "tracking_limit": tracking_limit,
        },
        tag="shot_goalmouth_sb",
        label="match",
    )
    if res.failures:
        raise RuntimeError(
            f"{len(res.failures)} match(es) failed: {res.failures}. Re-run to retry only those -- "
            f"the matches that succeeded are already sharded."
        )

    rows, reports, unmatched_all = [], {}, []
    sweep_rows, zcmp_rows = [], []
    debug_store: dict[tuple[str, int], dict] = {}
    seen_vocab: set = set()
    for k in res.keys:
        shard = pd.read_parquet(shard_path(res.shard_dir, k))
        for rec in shard.to_dict("records"):
            bundle = json.loads(rec["payload"])
            mid = rec["match_id"]
            rows.extend(bundle["rows"])
            reports[mid] = bundle["report"]
            unmatched_all.extend(bundle["unmatched"])
            sweep_rows.extend(bundle["sweep"])
            zcmp_rows.extend(bundle["zcmp"])
            seen_vocab |= set(bundle["vocab"])
            for aid, dbg in bundle["debug"].items():
                debug_store[(mid, int(aid))] = dbg

    _assert_vocab_corpus(seen_vocab)
    df = pd.DataFrame(rows)
    result: dict = {
        "matches_arg": matches,
        "sb_vocab_observed": sorted(seen_vocab),
        "n_matched": len(df),
        "n_unmatched": len(unmatched_all),
        "unmatched": unmatched_all,
        "per_match_reports": reports,
        "sweep": sweep_rows,
        "z_compare": zcmp_rows,
    }
    if len(df):
        usable = df[df["source"].isin(["observed", "extrapolated"]) & df["on_target_sb"]].copy()
        result["coverage_on_target"] = float(len(usable) / max((df["on_target_sb"]).sum(), 1))
        # GK-GEOMETRY handedness settlement (round-2 instrument, ADR-030): one vote per
        # matched shot from the SB freeze-frame GK vs the GS-tracked GK -- a single
        # well-identified object, independent of fit success and of goal-mouth ball
        # chaos (the round-1 ball-tag gate could not separate transform sign from
        # near-centre measurement noise: in-mouth-filtered agreement 0.75-0.79 on both
        # pilot and holdout-r1 with sign 16-5/17-5 -- a wrong sign reads ~0.2).
        gk_votes = []
        for r in df.to_dict("records"):
            gy = float(r["gk_y_m"]) if r["gk_y_m"] is not None else float("nan")
            sgy = float(r["sb_gk_y"]) if r["sb_gk_y"] is not None else float("nan")
            v = _vote(gy - 34.0, sgy - 40.0, _GK_SIDE_MIN_M, _GK_SIDE_MIN_YD)
            if v is not None:
                gk_votes.append(v)
        sign, hand_diag = _settle_votes(gk_votes, strict=(matches == "holdout"))
        hand_diag["instrument"] = "gk_freeze_frame"
        result["handedness_sign"] = sign
        result["handedness_diag"] = hand_diag
        goals = usable[usable["is_goal"]]
        if len(goals):
            result["handedness_ball_diag"] = _ball_handedness_diag(goals)

            # per-shot pairs for diagnosis (goals AND saves; action_id enables joining the
            # per-shot trajectory diagnostics for error categorization -- Option B)
            def _pairs(sub: pd.DataFrame):
                return [
                    {
                        "match_id": r["match_id"],
                        "action_id": int(r["gs_action_id"]),
                        "gs_y_m": round(float(r["crossing_y_m"]), 2),
                        "gs_z_m": round(float(r["crossing_z_m"]), 2) if np.isfinite(float(r["crossing_z_m"])) else None,
                        "sb_y": r["sb_end_y"],
                        "sb_z": r["sb_end_z"],
                        "source": r["source"],
                        "z_profile": r["z_profile"],
                    }
                    for r in sub.to_dict("records")
                ]

            result["goal_pairs"] = _pairs(goals)
            result["save_pairs"] = _pairs(usable[~usable["is_goal"]])
        usable["_y_sb"] = 40.0 - sign * (usable["crossing_y_m"] - 34.0) / _M_PER_YD
        usable["_z_sb"] = usable["crossing_z_m"] / _M_PER_YD
        usable["dy"] = (usable["_y_sb"] - usable["sb_end_y"]).abs() * _M_PER_YD
        usable["dz"] = (usable["_z_sb"] - usable["sb_end_z"]).abs() * _M_PER_YD
        result["goals"] = _delta_stats(usable[usable["is_goal"]])
        result["saves"] = _delta_stats(usable[~usable["is_goal"]])
        agree = usable["on_target_derived"].astype("boolean")
        result["on_target_agreement"] = float((agree == True).mean())  # noqa: E712
    if debug_shots and len(df):
        sign = result.get("handedness_sign", 1)
        dbg = []
        for r in df.to_dict("records"):  # dict access: itertuples types every field Scalar

            def _f(v) -> float:  # None-tolerant float (object columns can carry None)
                return float(v) if v is not None else float("nan")

            dy = dz = None
            cy, sy = _f(r["crossing_y_m"]), _f(r["sb_end_y"])
            if r["source"] in ("observed", "extrapolated") and np.isfinite(cy) and np.isfinite(sy):
                y_sb = 40.0 - sign * (cy - 34.0) / _M_PER_YD
                dy = round(abs(y_sb - sy) * _M_PER_YD, 2)
                cz, sz = _f(r["crossing_z_m"]), _f(r["sb_end_z"])
                if np.isfinite(cz) and np.isfinite(sz):
                    dz = round(abs(cz / _M_PER_YD - sz) * _M_PER_YD, 2)
            dbg.append(
                {
                    "match_id": r["match_id"],
                    "action_id": int(r["gs_action_id"]),
                    "sb_outcome": r["sb_outcome"],
                    "is_goal": bool(r["is_goal"]),
                    "on_target_sb": bool(r["on_target_sb"]),
                    "gs_period": int(r["gs_period"]),
                    "gs_time_seconds": round(float(r["gs_time_seconds"]), 2),
                    "sb_clock_s": round(float(r["sb_clock_s"]), 2),
                    "gs_start_xy": [round(float(r["gs_start_x"]), 2), round(float(r["gs_start_y"]), 2)],
                    "crossing_y_m": _jsonable(r["crossing_y_m"]),
                    "crossing_z_m": _jsonable(r["crossing_z_m"]),
                    "sb_end_y": _jsonable(r["sb_end_y"]),
                    "sb_end_z": _jsonable(r["sb_end_z"]),
                    "source": r["source"],
                    "z_profile": r["z_profile"],
                    "on_target_derived": None if pd.isna(r["on_target_derived"]) else bool(r["on_target_derived"]),
                    "dy_m": dy,
                    "dz_m": dz,
                    # full kernel capture; None when the kernel never ran for this shot
                    "kernel": debug_store.get((r["match_id"], int(r["gs_action_id"]))),
                }
            )
        dbg.sort(key=lambda e: (e["dy_m"] is None, -(e["dy_m"] or 0.0)))
        result["debug_shots"] = dbg
    # ADR-037: the CLI REFUSES a dirty tree (see main()); run() RECORDS what it was given. A
    # run() that refused could not be tested without mocking git, which is why the split exists.
    if provenance is not None:
        result["run_commit"] = provenance["commit"]
        result["run_tree_dirty"] = provenance["dirty"]
    Path(out_path).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("unmatched", "per_match_reports", "sweep", "debug_shots")},
            indent=2,
            default=str,
        )
    )
    diag = result.get("handedness_diag", {})
    if diag.get("strict_abort_pending"):
        # spec H5 strict abort -- AFTER the report is on disk (see _settle_handedness)
        raise AssertionError(
            f"handedness agreement {diag.get('agreement')} < {_HANDEDNESS_FLOOR} on "
            f"{diag.get('n_side_votes')} clearly-sided goals -- transform ambiguous; "
            f"aborting before floors (spec H5); full report written to {out_path}"
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="TF-48 GS<->SB WC2022 validation harness (see module docstring)")
    ap.add_argument("--matches", choices=("pilot", "holdout", "all"), default="pilot")
    ap.add_argument("--out", default="shot_goalmouth_sb_report.json")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--sweep", action="store_true", help="pilot-only sensitivity sweep")
    ap.add_argument("--z-compare", action="store_true", help="pilot-only raw-vs-smoothed z (needs --cache-dir)")
    ap.add_argument(
        "--debug-shots",
        action="store_true",
        help="per matched shot: full-resolution ball window + kernel fit internals in --out",
    )
    ap.add_argument("--tracking-limit", type=int, default=None, help="dev-smoke only; NEVER for the real run")
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit a dev run from a modified tree; the artifact still records dirty: true",
    )
    args = ap.parse_args()

    # Refuse BEFORE paying for the StatsBomb pull and the tracking corpus walk.
    from scripts._provenance import git_provenance, require_clean_tree

    provenance = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)

    run(
        args.matches,
        args.out,
        args.cache_dir,
        args.tracking_limit,
        sweep=args.sweep,
        z_compare=args.z_compare,
        debug_shots=args.debug_shots,
        provenance=provenance,
    )


if __name__ == "__main__":
    main()
