"""Pure reducers for the TF-58 coordination artifacts (spec 8.2 / spec 8.5).

``boundary_f1_by_gap`` serves the D1 ``possession_gap_s`` derivation (Pass B). The seven pre-registered
hypothesis reducers (H1-H7), ``evaluate_hypotheses`` and ``gated_pass`` implement spec 8.5; they are consumed by
the D2 confirmation gate (``gated_pass``) and the D3 report. Every reducer is a PURE function over the per-match
family tables (concatenated across the corpus); each returns ``{"pass": bool | None, ...stats}`` and reads its
threshold from :mod:`scripts._coordination_thresholds` -- never an inline literal. A failed hypothesis is a
recorded finding, never a drop (spec 8.5).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as st

from scripts import _coordination_thresholds as thr
from silly_kicks.coordination import (
    CoordinationParams,
    possession_windows_from_actions,
    possession_windows_from_frames,
)
from silly_kicks.id_compat import canonical_id_series, ids_differ, ids_equal
from silly_kicks.spadl.utils import boundary_metrics


def _pval(result: Any) -> float:
    """Extract a scipy test's ``pvalue`` as a float (single-sources the stub-friction cast)."""
    return float(result.pvalue)


def _frame_possession_ids(frames: pd.DataFrame, windows: pd.DataFrame) -> pd.Series:
    """Per unique (game, period, frame) in time order, the covering possession window's index (-1 if none).

    A per-frame possession-id sequence, so two possession segmentations (event vs frames-only) can be compared
    boundary-for-boundary over one shared row order (the TF-52 idiom). ``-1`` (uncovered) is a distinct id, so
    an uncovered stretch is its own "possession" and any transition into or out of it is a real boundary.
    """
    uniq = (
        frames.drop_duplicates(["game_id", "period_id", "frame_id"])
        .sort_values(["game_id", "period_id", "time_seconds"])
        .reset_index(drop=True)
    )
    ids = np.full(len(uniq), -1, dtype=np.int64)
    g = uniq["game_id"].to_numpy(dtype=object)
    p = uniq["period_id"].to_numpy()
    t = uniq["time_seconds"].to_numpy(dtype=np.float64)
    poss = windows[windows["window_kind"] == "possession"].reset_index(drop=True)
    for wi, w in enumerate(poss.itertuples(index=False)):
        mask = (g == w.game_id) & (p == w.period_id) & (t >= w.start_time_s) & (t < w.end_time_s)
        ids[mask] = wi
    return pd.Series(ids)


def boundary_f1_by_gap(frames: pd.DataFrame, actions: pd.DataFrame, gaps_s: Sequence[float]) -> dict[float, float]:
    """Boundary F1 of frames-only possession spells vs event possessions, per candidate gap (spec 8.2, TF-52 idiom).

    Event possessions (``possession_windows_from_actions``) are the ground truth. For each ``g`` in ``gaps_s``,
    frames-only spells are rebuilt with ``possession_gap_s = g``; both segmentations are mapped to a per-frame
    possession id over one shared frame order, and ``boundary_metrics`` scores where their boundaries agree. The
    gap maximising this F1 is the derived ``possession_gap_s``
    (:func:`derive_coordination_params.possession_gap_argmax` breaks ties toward the smaller gap).
    """
    from silly_kicks.tracking import infer_ball_carrier

    # F3 (speed): the ball carrier is gap-invariant (frames-only) -- infer it ONCE and thread it into
    # each gap's possession rebuild, rather than re-inferring (and re-_pre_index_frames, ~79-87% of
    # pass-b) per candidate gap. possession_windows_from_frames(carrier=...) skips the re-inference.
    carrier = infer_ball_carrier(frames)
    event_ids = _frame_possession_ids(frames, possession_windows_from_actions(actions, frames))
    base = CoordinationParams()
    out: dict[float, float] = {}
    for g in gaps_s:
        windows = possession_windows_from_frames(
            frames, carrier=carrier, params=dataclasses.replace(base, possession_gap_s=float(g))
        )
        heuristic = _frame_possession_ids(frames, windows)
        out[float(g)] = float(boundary_metrics(heuristic=heuristic, native=event_ids)["f1"])
    return out


# ============================================================================ pre-registered hypotheses (spec 8.5)
_HALF_KEYS = ["game_id", "period_id", "window_id"]


#: Moura's early third. Phases are 1..n_phases (spec 7.6: phase k holds the samples in ((k-1)/n, k/n]), so the early
#: third is phase 1 -- H3 once read phase 0, which no row has, so it could never be evaluated (round-2 finding).
EARLY_PHASE = 1
#: The keys a family row shares with its window row: the long frame's lead tags (two providers or matches can share
#: a game / window id) plus the window grain.
_WINDOW_JOIN_KEYS = ("provider", "match_id", "variant", "game_id", "period_id", "window_id")


def _window_join(rows: pd.DataFrame, windows: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """``rows`` with the possession windows' ``columns`` joined on every shared join key (left join)."""
    keys = [k for k in _WINDOW_JOIN_KEYS if k in rows.columns and k in windows.columns]
    poss = windows.loc[windows["window_kind"] == "possession", [*keys, *(c for c in columns if c in windows.columns)]]
    return rows.merge(poss, on=keys, how="left")


def _team_key(df: pd.DataFrame, col: str) -> pd.Series:
    """A team's grouping key: the canonical id (ADR-019), qualified by provider when the frame carries one --
    team ids are provider-scoped (review A-11)."""
    team = canonical_id_series(df[col]).astype(object)
    if "provider" in df.columns:
        return df["provider"].astype(str) + "|" + team.astype(str)
    return team.astype(str)


def _has(df: pd.DataFrame, *cols: str) -> bool:
    """True iff ``df`` is non-empty and carries every named column (a guard so a partial/empty table -- a thin
    corpus, or a caller that never populated a family -- degrades to a recorded non-pass, never a ``KeyError``)."""
    return len(df) > 0 and all(c in df.columns for c in cols)


def _sign_test_greater(a: np.ndarray, b: np.ndarray) -> tuple[float, int, float]:
    """One-sided paired sign test that ``a > b``: ``binomtest(#(a>b), n, 0.5, "greater")`` -> (p, n, share).

    A-53: TIES (``a == b``) carry no sign and are **excluded** from the trial count ``n`` (the standard paired
    sign test), so a tie no longer counts as a loss and deflates the share.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b) & (a != b)  # drop non-finite AND tied pairs
    a, b = a[m], b[m]
    n = int(a.size)
    if n == 0:
        return float("nan"), 0, float("nan")
    k = int(np.sum(a > b))
    return _pval(st.binomtest(k, n, 0.5, alternative="greater")), n, k / n


def _tost_equivalent(diffs: np.ndarray, margin: float, alpha: float) -> tuple[bool, float, float]:
    """TOST: paired ``diffs`` are equivalent to 0 within ``+-margin`` iff both one-sided t-tests reject at ``alpha``."""
    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return False, float("nan"), float("nan")
    if np.allclose(d, d[0]):
        # A-53: a zero-variance difference set has no t-statistic, but it is trivially equivalent iff its single
        # value sits strictly inside +-margin (returning False would falsely call a perfect match non-equivalent).
        equiv = bool(abs(float(d[0])) < margin)
        return equiv, 0.0 if equiv else float("nan"), 0.0 if equiv else float("nan")
    p_lower = _pval(st.ttest_1samp(d, -margin, alternative="greater"))  # reject H0: mean <= -margin
    p_upper = _pval(st.ttest_1samp(d, margin, alternative="less"))  # reject H0: mean >= +margin
    return bool(p_lower < alpha and p_upper < alpha), p_lower, p_upper


def _paired_axis(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """Join a table's x-axis and y-axis rows on the match-half keys into (rx, ry) columns."""
    keys = [k for k in (*_HALF_KEYS, "team_id", "team_a_id", "team_b_id") if k in df.columns]
    rx = df[df["axis"] == "x"].set_index(keys)[value]
    ry = df[df["axis"] == "y"].set_index(keys)[value]
    return pd.concat([rx.rename("rx"), ry.rename("ry")], axis=1).dropna()


def h1_centroid_phase_stability(pair: pd.DataFrame) -> dict:
    """H1 (Bourbousson 2010): team-centroid relative phase is more stable longitudinally than laterally AND near
    in-phase -- paired sign test ``R_x > R_y`` over match-halves (p < H1_P) and the pooled longitudinal circular
    mean within ``+-H1_LONGITUDINAL_MEAN_WITHIN_DEG`` of 0."""
    if not _has(pair, "level", "window_kind", "signal_a", "coord_rp_resultant_length", "coord_rp_mean_deg"):
        return {
            "pass": False,
            "p_value": float("nan"),
            "n_halves": 0,
            "share_x_gt_y": float("nan"),
            "longitudinal_circ_mean_deg": float("nan"),
        }
    tt = pair[(pair["level"] == "team_team") & (pair["window_kind"] == "period")]
    keys = [k for k in (*_HALF_KEYS, "team_a_id", "team_b_id") if k in tt.columns]
    rx = tt[tt["signal_a"] == "centroid_x"].set_index(keys)["coord_rp_resultant_length"]
    ry = tt[tt["signal_a"] == "centroid_y"].set_index(keys)["coord_rp_resultant_length"]
    joined = pd.concat([rx.rename("rx"), ry.rename("ry")], axis=1).dropna()
    p, n, share = _sign_test_greater(joined["rx"].to_numpy(), joined["ry"].to_numpy())
    long_deg = tt.loc[tt["signal_a"] == "centroid_x", "coord_rp_mean_deg"].to_numpy(dtype=np.float64)
    long_deg = long_deg[np.isfinite(long_deg)]
    circ_mean = float(st.circmean(long_deg, high=180, low=-180)) if long_deg.size else float("nan")
    within = bool(np.isfinite(circ_mean) and abs(circ_mean) <= thr.H1_LONGITUDINAL_MEAN_WITHIN_DEG)
    return {
        "pass": bool(np.isfinite(p) and p < thr.H1_P and within),
        "p_value": p,
        "n_halves": n,
        "share_x_gt_y": share,
        "longitudinal_circ_mean_deg": circ_mean,
    }


def h2_spread_xcorr(pair: pd.DataFrame) -> dict:
    """H2 (Moura 2016): team-spread cross-correlation is positive with a short lag -- signed ``r`` at max ``|r|``
    positive in >= H2_POSITIVE_SHARE of team-pair halves AND median ``|lag|`` <= H2_MEDIAN_ABS_LAG_S."""
    if not _has(pair, "level", "signal_a", "window_kind", "coord_xc_r_at_max", "coord_xc_lag_s"):
        return {"pass": False, "positive_share": float("nan"), "median_abs_lag_s": float("nan"), "n": 0}
    tt = pair[(pair["level"] == "team_team") & (pair["signal_a"] == "spread") & (pair["window_kind"] == "period")]
    r = tt["coord_xc_r_at_max"].to_numpy(dtype=np.float64)
    lag = tt["coord_xc_lag_s"].to_numpy(dtype=np.float64)
    r, lag = r[np.isfinite(r)], lag[np.isfinite(lag)]
    pos_share = float(np.mean(r > 0.0)) if r.size else float("nan")
    med_lag = float(np.median(np.abs(lag))) if lag.size else float("nan")
    return {
        "pass": bool(
            np.isfinite(pos_share)
            and pos_share >= thr.H2_POSITIVE_SHARE
            and np.isfinite(med_lag)
            and med_lag <= thr.H2_MEDIAN_ABS_LAG_S
        ),
        "positive_share": pos_share,
        "median_abs_lag_s": med_lag,
        "n": int(r.size),
    }


def h3_early_third_by_terminal(pair_phase: pd.DataFrame, windows: pd.DataFrame) -> dict:
    """H3 (Moura 2016): early-third anti-phase AND attacking-team-phase fractions are higher when a possession ends
    in a shot than in a tackle -- one-sided Mann-Whitney (p < H3_P each), event providers only."""
    if not _has(pair_phase, "level", "signal_a", "window_kind", "phase_index", "coord_vc_pct_anti_phase") or not len(
        windows
    ):
        return {"pass": False, "p_anti_phase": float("nan"), "p_attack_phase": float("nan")}
    early = pair_phase[
        (pair_phase["level"] == "team_team")
        & (pair_phase["signal_a"] == "spread")
        & (pair_phase["window_kind"] == "possession")
        & (pair_phase["phase_index"] == EARLY_PHASE)
    ]
    early = _window_join(early, windows, ("terminal_action", "attacking_team_id")).reset_index(drop=True)
    a_attacks = ids_equal(early["team_a_id"], early["attacking_team_id"]).to_numpy()  # id_compat, never to_numeric
    early = early.assign(attack_phase=np.where(a_attacks, early["coord_vc_pct_a_phase"], early["coord_vc_pct_b_phase"]))

    def _mwu(col: str) -> float:
        shot = early.loc[early["terminal_action"] == "shot", col].to_numpy(dtype=np.float64)
        tackle = early.loc[early["terminal_action"] == "tackle", col].to_numpy(dtype=np.float64)
        shot, tackle = shot[np.isfinite(shot)], tackle[np.isfinite(tackle)]
        if shot.size == 0 or tackle.size == 0:
            return float("nan")
        return _pval(st.mannwhitneyu(shot, tackle, alternative="greater"))

    p_anti, p_attack = _mwu("coord_vc_pct_anti_phase"), _mwu("attack_phase")
    return {
        "pass": bool(np.isfinite(p_anti) and np.isfinite(p_attack) and p_anti < thr.H3_P and p_attack < thr.H3_P),
        "p_anti_phase": p_anti,
        "p_attack_phase": p_attack,
    }


def h4_median_frequency(spectral: pd.DataFrame) -> dict:
    """H4 (Moura 2013): median frequency < H4_CEIL_CPM in >= H4_BELOW_CEIL_SHARE of team-halves AND first half >
    second half (paired one-sided Wilcoxon signed-rank, p < H4_P)."""
    empty = {"pass": False, "below_ceiling_share": float("nan"), "p_first_gt_second": float("nan"), "by_signal": {}}
    if not _has(spectral, "window_kind", "coord_median_freq_cpm", "team_id", "signal", "period_id"):
        return empty
    sp = spectral[spectral["window_kind"] == "period"]
    keys = [k for k in ("provider", "match_id", "game_id", "team_id") if k in sp.columns]  # one team-half pair each
    by_signal: dict[str, dict[str, float | bool]] = {}
    for signal in thr.H4_SIGNALS:  # Moura 2013's area and spread, each on its own (review A-13)
        s = sp[sp["signal"] == signal]
        freqs = s["coord_median_freq_cpm"].to_numpy(dtype=np.float64)
        freqs = freqs[np.isfinite(freqs)]
        below_share = float(np.mean(freqs < thr.H4_CEIL_CPM)) if freqs.size else float("nan")
        p_wil = float("nan")
        if len(s):
            piv = s.pivot_table(index=keys, columns="period_id", values="coord_median_freq_cpm", observed=True)
            if {1, 2} <= set(piv.columns):
                both = piv[[1, 2]].dropna()
                if len(both) and not np.allclose(both[1].to_numpy(), both[2].to_numpy()):
                    p_wil = _pval(st.wilcoxon(both[1].to_numpy(), both[2].to_numpy(), alternative="greater"))
        by_signal[signal] = {
            "pass": bool(
                np.isfinite(below_share)
                and below_share >= thr.H4_BELOW_CEIL_SHARE
                and np.isfinite(p_wil)
                and p_wil < thr.H4_P
            ),
            "below_ceiling_share": below_share,
            "p_first_gt_second": p_wil,
        }
    shares = [v["below_ceiling_share"] for v in by_signal.values()]
    pvals = [v["p_first_gt_second"] for v in by_signal.values()]
    return {
        "pass": all(v["pass"] for v in by_signal.values()),
        # the binding values across Moura's signals: the lowest below-ceiling share and the largest p
        "below_ceiling_share": float(np.min(shares)) if np.isfinite(shares).all() else float("nan"),
        "p_first_gt_second": float(np.max(pvals)) if np.isfinite(pvals).all() else float("nan"),
        "by_signal": by_signal,
    }


def h5_rho_group(cluster_team: pd.DataFrame, windows: pd.DataFrame) -> dict:
    """H5 (Duarte 2013): cluster-phase rho_group is higher longitudinally than laterally (paired sign test p <
    H5_P) AND possession has no effect (paired TOST equivalence in vs out of possession within +-H5_TOST_MARGIN)."""
    if not _has(cluster_team, "window_kind", "axis", "coord_rho_group_mean") or not len(windows):
        return {
            "pass": False,
            "p_x_gt_y": float("nan"),
            "n_halves": 0,
            "share_x_gt_y": float("nan"),
            "possession_tost_equivalent": False,
            "tost_p_lower": float("nan"),
            "tost_p_upper": float("nan"),
        }
    ct = cluster_team[cluster_team["window_kind"] == "period"]
    joined = _paired_axis(ct, "coord_rho_group_mean")
    p, n, share = _sign_test_greater(joined["rx"].to_numpy(), joined["ry"].to_numpy())
    poss = cluster_team[cluster_team["window_kind"] == "possession"]
    poss = _window_join(poss, windows, ("attacking_team_id",)).reset_index(drop=True)
    in_poss = ids_equal(poss["team_id"], poss["attacking_team_id"]).to_numpy()  # id_compat (review A-11)
    out_poss = ids_differ(poss["team_id"], poss["attacking_team_id"]).to_numpy()  # an unknown attacker is neither
    team = _team_key(poss, "team_id")
    in_mean = poss["coord_rho_group_mean"][in_poss].groupby(team[in_poss], observed=True).mean()
    out_mean = poss["coord_rho_group_mean"][out_poss].groupby(team[out_poss], observed=True).mean()
    diffs = (in_mean - out_mean).dropna().to_numpy(dtype=np.float64)
    equiv, p_lo, p_hi = _tost_equivalent(diffs, thr.H5_TOST_MARGIN, thr.H5_TOST_ALPHA)
    return {
        "pass": bool(np.isfinite(p) and p < thr.H5_P and equiv),
        "p_x_gt_y": p,
        "n_halves": n,
        "share_x_gt_y": share,
        "possession_tost_equivalent": equiv,
        "tost_p_lower": p_lo,
        "tost_p_upper": p_hi,
    }


def h6_dyad_near_in_phase(pair: pd.DataFrame) -> dict:
    """H6 (Folgado 2014): descriptive -- the dyad near-in-phase distribution per axis (quartiles). ``pass`` is None."""
    if not _has(pair, "level", "axis", "coord_rp_pct_near_in_phase"):
        return {"pass": None, "quartiles_x": [float("nan")] * 3, "quartiles_y": [float("nan")] * 3}
    dyad = pair[pair["level"] == "dyad"]

    def _quartiles(axis: str) -> list[float]:
        v = dyad.loc[dyad["axis"] == axis, "coord_rp_pct_near_in_phase"].to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        return [float(np.quantile(v, q)) for q in (0.25, 0.5, 0.75)] if v.size else [float("nan")] * 3

    return {"pass": None, "quartiles_x": _quartiles("x"), "quartiles_y": _quartiles("y")}


def h7_rsi(
    rsi: pd.DataFrame,
    rsi_switch_times: pd.DataFrame,
    possession_changes: pd.DataFrame,
    *,
    seed: int = thr.H7_SEED,
    n_surrogates: int = thr.H7_N_SURROGATES,
) -> dict:
    """H7 (Bourbousson 2010): RSI is bimodal (BC > H7_BC_THRESHOLD in >= H7_BIMODAL_SHARE of team-halves) AND its
    sign switches follow possession changes -- the switch share within H7_SWITCH_WINDOW_S after a change exceeds the
    time-shift-surrogate H7_SURROGATE_PERCENTILE. ``rsi_switch_times`` / ``possession_changes`` carry ``match_id`` +
    ``time`` columns (produced by the D3 metrics pass)."""
    if not _has(rsi, "coord_rsi_bimodality_coefficient"):
        return {"pass": False, "bc_share": float("nan"), "switch_share": float("nan"), "surrogate_95": float("nan")}
    r = rsi[rsi["window_kind"] == "period"] if "window_kind" in rsi.columns else rsi
    bc = r["coord_rsi_bimodality_coefficient"].to_numpy(dtype=np.float64) if len(r) else np.array([])
    bc = bc[np.isfinite(bc)]
    bc_share = float(np.mean(bc > thr.H7_BC_THRESHOLD)) if bc.size else float("nan")

    obs, surro = _switch_vs_change_surrogate(rsi_switch_times, possession_changes, seed=seed, n_surrogates=n_surrogates)
    surro_pct = float(np.nanpercentile(surro, thr.H7_SURROGATE_PERCENTILE)) if surro.size else float("nan")
    return {
        "pass": bool(
            np.isfinite(bc_share)
            and bc_share >= thr.H7_BIMODAL_SHARE
            and np.isfinite(obs)
            and np.isfinite(surro_pct)
            and obs > surro_pct
        ),
        "bc_share": bc_share,
        "switch_share": obs,
        "surrogate_95": surro_pct,
    }


#: An H7 match-half: the long frame's lead tags plus the game and period. Time is period-relative (ADR-017), so a
#: switch is matched only to its own match-half's possession changes (review A-12).
_H7_HALF_KEYS = ("provider", "match_id", "variant", "game_id", "period_id")


def _switch_count(switch_times: np.ndarray, change_times: np.ndarray, window_s: float) -> int:
    """How many RSI sign-switches fall within ``window_s`` AFTER a possession change."""
    if switch_times.size == 0 or change_times.size == 0:
        return 0
    ch = np.sort(change_times)
    before = np.searchsorted(ch, switch_times, side="right") - 1  # the nearest change at or before
    ok = before >= 0
    delay = np.full(switch_times.shape, np.inf)
    delay[ok] = switch_times[ok] - ch[before[ok]]
    return int(np.sum((delay >= 0) & (delay <= window_s)))


def _switch_vs_change_surrogate(
    switch_times: pd.DataFrame, changes: pd.DataFrame, *, seed: int, n_surrogates: int
) -> tuple[float, np.ndarray]:
    """The share of RSI sign-switches within ``H7_SWITCH_WINDOW_S`` after a possession change, POOLED over every switch
    (spec 8.5: "share of switches"), plus ``n_surrogates`` shares of the same statistic under a time shift.

    The unit is one match-half and axis: a switch is matched only to its own match-half's changes (the clock is
    period-relative -- a second-half switch at 600 s is not "after" a first-half change at 595 s), and each unit's
    switches are circularly shifted independently (review A-12)."""
    if not len(switch_times) or "time" not in switch_times.columns:
        return float("nan"), np.array([])
    half = [k for k in _H7_HALF_KEYS if k in switch_times.columns]
    unit = [*half, "axis"] if "axis" in switch_times.columns else half
    change_by_half: dict[tuple, np.ndarray] = {}
    if len(changes) and half and all(k in changes.columns for k in half):
        for key, g in changes.groupby(half, sort=True, dropna=False, observed=True):
            change_by_half[key if isinstance(key, tuple) else (key,)] = g["time"].to_numpy(dtype=np.float64)
    units: list[tuple[np.ndarray, np.ndarray, float]] = []
    for key, g in switch_times.groupby(unit, sort=True, dropna=False, observed=True) if unit else [((), switch_times)]:
        key = key if isinstance(key, tuple) else (key,)
        sw = g["time"].to_numpy(dtype=np.float64)
        ch = change_by_half.get(tuple(key[: len(half)]), np.empty(0))
        dur = float(max(sw.max(initial=0.0), ch.max(initial=0.0))) + 1.0
        units.append((sw, ch, dur))
    n_switches = sum(sw.size for sw, _, _ in units)
    if n_switches == 0:
        return float("nan"), np.array([])
    window = thr.H7_SWITCH_WINDOW_S
    obs = sum(_switch_count(sw, ch, window) for sw, ch, _ in units) / n_switches
    rng = np.random.default_rng(seed)
    surro = np.empty(n_surrogates, dtype=np.float64)
    for i in range(n_surrogates):
        hits = sum(_switch_count((sw + rng.uniform(0.0, dur)) % dur, ch, window) for sw, ch, dur in units)
        surro[i] = hits / n_switches
    return float(obs), surro


def evaluate_hypotheses(tables: Mapping[str, pd.DataFrame], *, seed: int = thr.H7_SEED) -> dict[str, dict]:
    """Run all seven pre-registered hypotheses over the per-match family tables (spec 8.5)."""
    empty = pd.DataFrame()
    return {
        "H1": h1_centroid_phase_stability(tables.get("pair", empty)),
        "H2": h2_spread_xcorr(tables.get("pair", empty)),
        "H3": h3_early_third_by_terminal(tables.get("pair_phase", empty), tables.get("windows", empty)),
        "H4": h4_median_frequency(tables.get("spectral", empty)),
        "H5": h5_rho_group(tables.get("cluster_team", empty), tables.get("windows", empty)),
        "H6": h6_dyad_near_in_phase(tables.get("pair", empty)),
        "H7": h7_rsi(
            tables.get("rsi", empty),
            tables.get("rsi_switch_times", empty),
            tables.get("possession_changes", empty),
            seed=seed,
        ),
    }


def gated_pass(results: Mapping[str, dict]) -> bool:
    """True iff every gated hypothesis (H1-H5, H7; H6 is descriptive) passed (spec 8.5)."""
    return all(results.get(h, {}).get("pass") is True for h in thr.GATED_HYPOTHESES)
