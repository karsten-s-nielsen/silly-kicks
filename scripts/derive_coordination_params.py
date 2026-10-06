#!/usr/bin/env python
"""D1 (Tier B, spec 8.2): derive per-provider coordination parameters from the ~980-match corpus.

Five shardable passes (ADR-052: expensive passes are their own passes), each run per worker over a disjoint
``--match-ids-json`` slice; every step that needs a corpus-wide input COMBINES all workers' shares first and refuses a
partial corpus (``--corpus-json``, spec 8.3):
  --pass a              raw positions, native rate -> per (provider, match, player, axis, run) residual-analysis cutoff
  --pass b              with the pass-A cutoffs (all workers) -> MAD epsilon, ACF zero crossing, median frequency,
                        possession-gap F1
  --pass occlusion-cal  GS + IDSSE (fully observed) -> the FOV-width calibration histogram
  --pass occlusion      ONE width W from every worker's histogram -> occluded-vs-truth error curves at W
  --pass reduce         combines a / b / occlusion, writes ``<--out>/derivation.json`` and ``<--out>/
                        _provider_params_generated.py`` (``render_generated_params(derivation, None)``) -- never the
                        package: D2/D3 read the artifact (owner ruling 2026-10-02), and commit 2 copies both in.

Shared corpus flags come from ``_coordination_corpus.add_common_args``: ``--out``, ``--token``, ``--max-matches``,
``--cache-dir``, ``--match-ids-json``, ``--providers``, ``--allow-dirty`` (dev-only; the artifact is marked dirty) and
``--list-matches``. The tree is checked with ``require_clean_tree(git_provenance())`` FIRST (ADR-037).

The pooled base is PROVIDER-NEUTRAL (one draw/provider; TF58-PLAN-02/A3): a provider absent from the generated
map is by construction unseen, and Tier-B quantities are properties of a provider's DATA, so the base must not inherit
SkillCorner's signature (909/980 matches). Per-provider values are match-weighted within that provider. One weighting
scheme (`unit_weights`) and one quantile definition (`weighted_quantile`) serve both, so they are directly comparable.

This module's pure reducers (below) are unit-tested in tests/scripts/test_derive_coordination_params.py; the passes
compose them over the corpus tables built by scripts/_coordination_corpus.match_tables.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import pathlib
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from scripts._driver import CorpusPassResult

# Sibling driver modules imported in package form so ``python -m scripts.derive_coordination_params`` resolves them;
# tests import this driver and patch the names bound HERE (``corpus_source`` / ``match_tables`` / ...).
from scripts import _coordination_thresholds as thr
from scripts._coordination_corpus import (
    StageTimer,
    add_common_args,
    build_windows,
    combine_workers,
    corpus_source,
    corpus_visibility_label,
    expected_corpus,
    run_params_token,
    table_pairs,
    visibility_preflight,
    write_worker_partial,
)
from scripts._coordination_hypotheses import boundary_f1_by_gap
from scripts._coordination_occlusion import (
    detection_rates,
    fov_mask,
    player_period_row_groups,
    simulate_broadcast_occlusion,
)
from scripts._coordination_params_codegen import render_generated_params
from scripts._coordination_reliability import CONSTRUCT_KEY_COLS, reliability_scored_columns
from silly_kicks.coordination import build_coordination_signals
from silly_kicks.coordination._columns import COORD_METHOD_FAMILIES, TEAM_SIGNALS, reliability_kind
from silly_kicks.coordination._compute import _result_from_signals
from silly_kicks.coordination._config import CoordinationParams
from silly_kicks.coordination._kernels._circular import wrap_deg
from silly_kicks.coordination._kernels._spectral import (
    median_frequency_cpm,
    min_spectral_samples,
    pooled_median_frequency,
)
from silly_kicks.coordination._kernels._surrogates import key_words
from silly_kicks.coordination._signals import _build_coordination_signals, _split_indices
from silly_kicks.tracking._geometry import GEOMETRY_VERSION
from silly_kicks.tracking._provider_visibility import detected_mask
from silly_kicks.tracking.preprocess._butterworth import residual_analysis_cutoff

#: The residual-analysis grid (spec 8.2): [0.1, 5.0] Hz in 0.05 steps, 5.0 INCLUSIVE (review A-52; it stopped at
#: 4.95). residual_analysis_cutoff clips it to the evaluable sub-grid at each run's native rate (C18), so the same
#: grid serves every provider.
RESIDUAL_GRID = np.round(np.arange(0.1, 5.0 + 0.025, 0.05), 2)
#: The possession-gap search grid (spec 8.2): {0.2, 0.4, ..., 3.0} s.
POSSESSION_GAP_GRID = tuple(round(0.2 * k, 1) for k in range(1, 16))


# --------------------------------------------------------------------------- Pass A reducer
def residual_cutoff_for_run(values: np.ndarray, fs: float) -> float:
    """Winter (2009) residual-analysis cutoff for one player-axis run on the spec 8.2 grid (clipped, C18).

    Thin wrapper over :func:`residual_analysis_cutoff` so the grid is single-sourced; raises the same
    ``ValueError`` when the run is too short or noise-free (the caller drops that run).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(4000) / 25.0
    >>> x = np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.05, t.size)
    >>> bool(0.3 < residual_cutoff_for_run(x, 25.0) < 1.2)
    True
    """
    return residual_analysis_cutoff(np.asarray(values, dtype=np.float64), fs, RESIDUAL_GRID)


# --------------------------------------------------------------------------- weighting + quantiles (single-sourced)
def weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray) -> float:
    """The ONE quantile definition every reducer uses: ``np.quantile(..., method="inverted_cdf")`` (numpy>=2).

    Examples
    --------
    >>> import numpy as np
    >>> v = np.array([1.0, 2.0, 3.0, 4.0])
    >>> weighted_quantile(v, 0.5, np.ones(4))
    2.0
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.quantile(values, q, weights=weights, method="inverted_cdf"))


def unit_weights(provider: np.ndarray, match: np.ndarray, *, provider_neutral: bool) -> np.ndarray:
    """Per-unit weights: ``1 / U_m`` within each match (each match counts once); times ``1 / M_p`` within each
    provider when ``provider_neutral`` (each provider counts once); normalised to sum 1.

    ``provider_neutral=True`` -> each provider carries total weight ``1/P`` (the pooled base, A3/TF58-PLAN-02);
    ``provider_neutral=False`` on one provider's units -> match-weighted within that provider. Matches are keyed by
    ``(provider, match)`` so a match id reused across providers never merges.

    Examples
    --------
    >>> import numpy as np
    >>> p = np.array(["a", "a", "b"])
    >>> m = np.array([1, 1, 9])
    >>> w = unit_weights(p, m, provider_neutral=True)
    >>> [round(float(x), 4) for x in w]  # a: two units in one match -> 1/4 each; b: one unit -> 1/2
    [0.25, 0.25, 0.5]
    """
    df = pd.DataFrame({"p": np.asarray(provider), "m": np.asarray(match)})
    grp = df.groupby(["p", "m"], sort=False, observed=True)
    u_m = grp["m"].transform("size").to_numpy(dtype=np.float64)  # units in each (provider, match)
    w = 1.0 / u_m
    if provider_neutral:
        matches_per_p = df.drop_duplicates(["p", "m"]).groupby("p", sort=False, observed=True).size()  # M_p
        m_p = df["p"].map(matches_per_p).to_numpy(dtype=np.float64)
        w = w / m_p
    total = w.sum()
    return w / total if total > 0 else w


def provider_cutoff(cutoffs: np.ndarray, weights: np.ndarray) -> float:
    """The provider's Butterworth cutoff: the weighted median over its runs (spec 8.2)."""
    return weighted_quantile(cutoffs, 0.5, weights)


# --------------------------------------------------------------------------- Pass B reducers
def vc_epsilon_for_signal(raw_resampled: np.ndarray, filtered: np.ndarray) -> float:
    """One match's vector-coding epsilon: ``1.4826 * MAD`` of the first differences of ``raw - filtered`` (spec 8.2).

    The noise-induced change per sample; combined across matches by weighted median in the reduce step.
    """
    resid = np.asarray(raw_resampled, dtype=np.float64) - np.asarray(filtered, dtype=np.float64)
    d = np.diff(resid)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(d - np.median(d))))


def first_acf_zero_crossing_s(x: np.ndarray, fs: float) -> float:
    """The first lag (seconds) at which the autocorrelation crosses zero, linear-interpolated (spec 8.2).

    On a sinusoid this is a quarter period (ACF is a cosine).

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(4000) / 25.0
    >>> x = np.sin(2 * np.pi * 0.5 * t)  # period 2 s -> quarter period 0.5 s
    >>> bool(abs(first_acf_zero_crossing_s(x, 25.0) - 0.5) < 0.02)
    True
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    acf = np.correlate(x, x, mode="full")[x.size - 1 :]
    if acf[0] == 0:
        return float("nan")
    acf = acf / acf[0]
    neg = np.flatnonzero(acf <= 0.0)
    if neg.size == 0:
        return float((x.size - 1) / fs)  # never crosses on this run -> its full length
    k = int(neg[0])
    if k == 0:
        return 0.0
    a0, a1 = acf[k - 1], acf[k]
    frac = a0 / (a0 - a1)  # a0 > 0 >= a1, so frac in [0, 1]
    return float((k - 1 + frac) / fs)


def min_shift_for_signal(zero_crossings_s: np.ndarray, weights: np.ndarray) -> float:
    """``min_shift_s`` per signal: weighted 95th percentile over match-halves of the ACF zero crossing (spec 8.2)."""
    return weighted_quantile(zero_crossings_s, 0.95, weights)


def band_from_median_frequencies(values_cpm: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """``(band_low_cpm, band_high_cpm)``: weighted 5th and 95th percentiles of team-signal median freqs (spec 8.2)."""
    return weighted_quantile(values_cpm, 0.05, weights), weighted_quantile(values_cpm, 0.95, weights)


def welch_segment_rule(band_low_cpm: float, band_high_cpm: float, half_s: float = 2700.0) -> tuple[float, bool]:
    """``welch_segment_s`` (spec 8.2): the segment giving frequency resolution <= (band width)/4, rounded UP to the
    next 10 s, and whether >= 8 Welch segments per half at 50% overlap also holds.

    Resolution sets the length (``60 / L`` cpm <= band_width / 4  =>  ``L >= 240 / band_width``). If that length
    yields fewer than 8 segments, resolution wins (the length is kept) and the returned flag is ``False`` to record
    the shortfall; when both hold the flag is ``True``.

    Examples
    --------
    >>> welch_segment_rule(0.22, 0.83, 2700.0)  # resolution needs >= 393.4 s -> 400; 12 segments >= 8 holds
    (400.0, True)
    """
    band_width = band_high_cpm - band_low_cpm
    if band_width <= 0:
        raise ValueError(f"welch_segment_rule: band width must be positive, got {band_width}")
    res_min_s = 240.0 / band_width  # 60 / L <= band_width / 4
    segment_s = math.ceil(res_min_s / 10.0) * 10.0  # round UP to the next 10 s
    step = segment_s / 2.0  # 50% overlap
    n_segments = 1 + math.floor((half_s - segment_s) / step) if half_s >= segment_s else 0
    return float(segment_s), bool(n_segments >= 8)


def possession_gap_argmax(f1_by_gap: Mapping[float, float]) -> float:
    """``possession_gap_s`` (spec 8.2): the gap maximising boundary F1; ties resolve to the SMALLER gap."""
    return float(min(f1_by_gap, key=lambda g: (-f1_by_gap[g], g)))


# --------------------------------------------------------------------------- Occlusion reducers
def min_observed_fraction_rule(err_by_bin: Mapping[float, float], between_match_sd: float) -> float:
    """``min_observed_fraction`` per metric family (spec 8.2): the smallest observed-fraction bin whose median absolute
    error under occlusion is at most ``0.5 * between_match_sd`` (the metric's full-observation between-match SD).

    Raises ``ValueError`` when no bin qualifies (the caller records the shortfall).
    """
    threshold = 0.5 * between_match_sd
    for frac in sorted(err_by_bin):
        if err_by_bin[frac] <= threshold:
            return float(frac)
    raise ValueError("min_observed_fraction_rule: no observed-fraction bin meets 0.5 * between-match SD")


def construct_qualifying_share(
    err_by_bin: Mapping[float, float],
    n_by_bin: Mapping[float, int],
    between_match_sd: float,
    *,
    min_matches: int,
) -> dict:
    """A CONSTRUCT's smallest qualifying observed-fraction bin and whether its curve is estimable (A-09 / C.8.6).

    The qualifying share is the smallest bin whose median absolute error is at most ``0.5 * between_match_sd`` (for a
    circular-mean construct the caller passes its circular SD and wrapped angular errors, section 8.5). The curve is
    **estimable** iff every present decile bin carries at least ``min_matches`` matches AND the crossing is finite and
    UNIQUE (once the error drops to/below the bar it stays below for every higher observed-fraction bin -- a single
    clean transition). A non-estimable curve, or one that never clears the bar below 1.0, yields share ``1.0``
    (full observation only) and a reason -- a FINDING, never a silent lowering.

    Returns ``{share, estimable, reason, threshold, crossing_bin}``.
    """
    threshold = 0.5 * float(between_match_sd)
    bins = sorted(float(b) for b in err_by_bin)
    if not bins:
        return {"share": 1.0, "estimable": False, "reason": "no_bins", "threshold": threshold, "crossing_bin": None}
    below = [b for b in bins if float(err_by_bin[b]) <= threshold]
    if not below:
        return {"share": 1.0, "estimable": False, "reason": "no_crossing", "threshold": threshold, "crossing_bin": None}
    crossing = min(below)
    underpowered = any(int(n_by_bin.get(b, 0)) < min_matches for b in bins)
    unique = all(float(err_by_bin[b]) <= threshold for b in bins if b >= crossing)
    if underpowered:
        return {
            "share": 1.0,
            "estimable": False,
            "reason": "underpowered_bins",
            "threshold": threshold,
            "crossing_bin": crossing,
        }
    if not unique:
        return {
            "share": 1.0,
            "estimable": False,
            "reason": "non_unique_crossing",
            "threshold": threshold,
            "crossing_bin": crossing,
        }
    return {
        "share": float(crossing),
        "estimable": True,
        "reason": None,
        "threshold": threshold,
        "crossing_bin": float(crossing),
    }


def family_max_observed_fraction(per_construct: Mapping[str, Mapping[str, Any]]) -> dict:
    """The family's consumed ``min_observed_fraction`` = **MAX** over its constructs' qualifying shares (A-09 / C.8.6,
    owner ruling 2026-10-04 -- fail-closed, since MAX >= every construct's own share, so each is gated at least as
    strictly as its own curve requires). Also records the **over-restriction** diagnostic (how much more strictly the
    MAX gates than each non-binding construct's own share) -- the owner's decision input for the per-construct-
    consumption follow-up -- and flags a FINDING when any construct is non-estimable or forces the family to 1.0.

    Returns ``{threshold, binding_construct, over_restriction, finding}``.
    """
    if not per_construct:
        return {"threshold": 1.0, "binding_construct": None, "over_restriction": {}, "finding": True}
    shares = {c: float(d["share"]) for c, d in per_construct.items()}
    threshold = max(shares.values())
    binding = max(shares, key=lambda c: (shares[c], str(c)))
    over_restriction = {c: float(threshold - s) for c, s in shares.items() if s < threshold}
    finding = any((not d.get("estimable")) or float(d["share"]) >= 1.0 for d in per_construct.values())
    return {
        "threshold": float(threshold),
        "binding_construct": binding,
        "over_restriction": over_restriction,
        "finding": bool(finding),
    }


def max_detection_gap_rule(rmse_by_gap_s: Mapping[float, float], noise_rms: float) -> float:
    """``max_detection_gap_s`` (spec 8.2): the longest gap whose bridged-position RMSE is at most ``2 * noise_rms``.

    Raises ``ValueError`` when no gap qualifies (even the shortest exceeds the bound).
    """
    threshold = 2.0 * noise_rms
    qualifying = [g for g, rmse in rmse_by_gap_s.items() if rmse <= threshold]
    if not qualifying:
        raise ValueError("max_detection_gap_rule: even the shortest gap exceeds 2 * noise RMS")
    return float(max(qualifying))


# --------------------------------------------------------------------------- Thin-provider precision (TF58-PLAN-03)
def provider_bootstrap_se(
    units: pd.DataFrame,
    reducer: Callable[[pd.DataFrame], float],
    *,
    seed_key: tuple,
    n_boot: int = 1000,
) -> float:
    """The match-level standard error of a provider's Tier-B estimate (TF58-PLAN-03).

    Resamples the provider's MATCHES with replacement (seeded via ``SeedSequence(entropy=<input_contract digest>,
    spawn_key=key_words(seed_key))`` so a given ``seed_key`` is reproducible), re-applies ``reducer`` to each
    resample, and returns the std (``ddof=1``) of the ``n_boot`` values. ``units`` must carry a ``match`` column;
    ``reducer`` re-derives its own per-provider ``unit_weights`` from the resampled units.

    A provider whose runs vary only WITHIN matches (identical per-match reducer values) has SE 0 -- match resampling
    cannot move a match-weighted statistic when every match agrees.
    """
    units = units.reset_index(drop=True)
    matches = np.asarray(pd.unique(units["match"]))
    entropy = int(input_contract()["digest"], 16)
    rng = np.random.default_rng(np.random.SeedSequence(entropy=entropy, spawn_key=key_words(seed_key)))
    # F2 (speed): resample per-match ROW-INDEX arrays and take ONE .iloc per draw, rather than
    # pd.concat-ing per-match frames 1000x. Row order within a draw is identical to the concat
    # (drawn-match order, each match's rows in situ) -> byte-identical reducer input (ADR-105).
    pos = {m: np.flatnonzero((units["match"] == m).to_numpy()) for m in matches}
    vals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        drawn = rng.choice(matches, size=matches.shape[0], replace=True)
        rows = np.concatenate([pos[m] for m in drawn])
        vals[i] = reducer(units.iloc[rows].reset_index(drop=True))
    return float(np.std(vals, ddof=1))


def thin_provider_flags(per_provider: Mapping[str, float], se: Mapping[str, float]) -> dict[str, str]:
    """Per Tier-B quantity, flag a provider whose estimate is noisier than the between-provider spread.

    ``"flagged"`` when ``se[p]`` exceeds the ``ddof=1`` SD of the per-provider values (its own uncertainty exceeds
    the spread it contributes to, so it is too noisy to count as one clean draw); ``"ok"`` otherwise; every provider
    ``"not_assessable"`` when fewer than 2 providers carry the quantity. REPORTS only -- never drops or re-weights.
    """
    providers = list(per_provider)
    if len(providers) < 2:
        return {p: "not_assessable" for p in providers}
    spread = float(np.std(np.asarray(list(per_provider.values()), dtype=np.float64), ddof=1))
    return {p: ("flagged" if se[p] > spread else "ok") for p in providers}


# --------------------------------------------------------------------------- input contract (declare_inputs)
def _coordination_params_defaults() -> dict:
    """CoordinationParams() as a JSON-able dict. ``dataclasses.asdict`` cannot be used: the map fields freeze to
    ``MappingProxyType`` in ``__post_init__`` (C16), which asdict's deepcopy refuses to pickle."""
    p = CoordinationParams()
    return {k: (dict(v) if isinstance(v, MappingProxyType) else v) for k, v in vars(p).items() if not k.startswith("_")}


def input_contract() -> dict:
    """The declared symbols D1's numbers depend on (ADR-056); written into ``derivation.json`` beside provenance."""
    from scripts._input_contract import declare_inputs

    return declare_inputs(
        driver="derive_coordination_params",
        grid=RESIDUAL_GRID.tolist(),
        possession_gap_grid=list(POSSESSION_GAP_GRID),
        percentiles={"band": [0.05, 0.95], "min_shift": 0.95},
        geometry_version=GEOMETRY_VERSION,
        params_defaults=_coordination_params_defaults(),
        occlusion_min_matches_per_bin=thr.OCCLUSION_MIN_MATCHES_PER_BIN,  # A-09 / C.8.6 per-construct estimability
    )


# =============================================================================== corpus passes + reduce (Task 20)
#: Bumping this invalidates every shard generation (it is in each pass's ``token_inputs``).
_SHARD_SCHEMA_VERSION = "tf58-d1-1"
#: The occlusion leg runs only over the fully-observed providers (spec 8.2, D9): GS 64 + IDSSE 7.
OCCLUSION_PROVIDERS = ("gradientsports", "idsse")
#: FOV widths (m) swept to calibrate W to SkillCorner's 66.6% outfield rate (sharded histogram, then interp).
_WIDTH_GRID = tuple(float(w) for w in range(10, 106, 5))
#: Detection-gap lengths (s) for the bridged-position RMSE curve -> ``max_detection_gap_s``.
_DETECTION_GAP_GRID = tuple(round(0.5 * k, 1) for k in range(1, 7))
_TARGET_OUTFIELD = 0.666
_SKILLCORNER_GK_RATE = 0.196
#: Bootstrap resamples for the thin-provider SE (TF58-PLAN-03; Efron & Tibshirani's >=200 for a standard error).
#: Module-level so a test can dial it down; the reduce pass reads it at call time.
THIN_PROVIDER_N_BOOT = 1000
#: The generated-module file name the reduce writes beside derivation.json (commit 2 copies it into the package).
GENERATED_ARTIFACT = "_provider_params_generated.py"

#: Each method family's representative metric column + its result table, for the occlusion error (spec 8.2).
_FAMILY_METRIC: Mapping[str, tuple[str, str]] = {
    "relative_phase": ("pair", "coord_rp_resultant_length"),
    "cross_correlation": ("pair", "coord_xc_max_abs_r"),
    "vector_coding": ("pair", "coord_vc_pct_in_phase"),
    "coherence": ("pair", "coord_coh_band_mean"),
    "spectral": ("spectral", "coord_median_freq_cpm"),
    "cluster": ("cluster_team", "coord_rho_group_mean"),
    "team_sync": ("team_sync", "coord_team_sync_pearson_r"),
    "rsi": ("rsi", "coord_rsi_mean_m"),
}
_TABLE_KEYS: Mapping[str, tuple[str, ...]] = {
    "pair": ("game_id", "period_id", "window_kind", "window_id", "level", "signal_a", "signal_b", "axis"),
    "spectral": ("game_id", "period_id", "window_kind", "window_id", "team_id", "signal"),
    "cluster_team": ("game_id", "period_id", "window_kind", "window_id", "team_id", "axis"),
    "team_sync": ("game_id", "period_id", "window_kind", "window_id", "axis"),
    "rsi": ("game_id", "period_id", "window_kind", "window_id", "axis"),
}
_WINDOW_KEYS = ("game_id", "period_id", "window_kind", "window_id")
#: The min_shift_s signals that are not team signals (player + cluster series).
_EXTRA_SHIFT_SIGNALS = ("player_x", "player_y", "cluster_amplitude")

_PASS_A_COLUMNS = ["provider", "match_id", "period_id", "team_id", "player_id", "axis", "run_index", "cutoff_hz"]
_PASS_B_COLUMNS = [
    "provider",
    "match_id",
    "period_id",
    "team_id",
    "player_id",
    "quantity",
    "signal",
    "value",
    "gap_s",
    "duration_s",
]
_OCC_COLUMNS = [
    "provider",
    "match_id",
    "quantity",
    "family",
    "construct",
    "column",
    "kind",
    "obs_bin",
    "gap_s",
    "value",
]

#: Column-prefix -> method family (A-09: the occlusion gate is consumed per family; a construct's column names it).
_FAMILY_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("coord_rp_", "relative_phase"),
    ("coord_xc_", "cross_correlation"),
    ("coord_vc_", "vector_coding"),
    ("coord_coh_", "coherence"),
    ("coord_median_freq", "spectral"),
    ("coord_rho_", "cluster"),
    ("coord_phi_", "cluster"),
    ("coord_team_sync_", "team_sync"),
    ("coord_rsi_", "rsi"),
)


def _column_family(column: str) -> str | None:
    """The method family a metric column belongs to (its ``min_observed_fraction`` gate), or None if ungoverned."""
    for prefix, family in _FAMILY_BY_PREFIX:
        if column.startswith(prefix):
            return family
    return None


def _construct_str(column: str, key_cols: list[str], key_vals) -> str:
    """A stable per-construct identity string: the column plus its C.1 what-keys (level/signal/axis/...)."""
    vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
    parts = [f"{k}={v}" for k, v in zip(key_cols, vals, strict=True)]
    return "|".join([f"column={column}", *parts])


def _circ_mean_deg(values_deg: np.ndarray) -> float:
    """Circular mean of degree values (NaN-safe); NaN when no finite value."""
    v = values_deg[np.isfinite(values_deg)]
    if not v.size:
        return float("nan")
    rad = np.radians(v)
    return float(np.degrees(np.arctan2(np.nanmean(np.sin(rad)), np.nanmean(np.cos(rad)))))


def _as_float(v) -> float:
    """A pandas groupby-key Scalar -> float, untyped arg to sidestep the stub's ConvertibleToFloat friction."""
    return float(v)


# --------------------------------------------------------------------------- Pass A (native-rate residual cutoffs)
def pass_a_match(loaded) -> pd.DataFrame:
    """One shard row per (period, team, player, axis, run) with that native-rate run's residual-analysis cutoff.

    Reads RAW native positions (no resample/filter): detected rows only (``detected_mask``), split into contiguous
    runs at any missing native frame, one :func:`residual_cutoff_for_run` per run (runs too short or noise-free are
    dropped, the reducer's ``ValueError``). Orientation-invariant, so raw ``x``/``y`` are used directly.
    """
    frames = loaded.frames
    if frames is None or not len(frames):
        return pd.DataFrame(columns=_PASS_A_COLUMNS)
    provider, match_id = loaded.provider, str(loaded.match_id)
    native_hz = float(frames["frame_rate"].iloc[0])
    gap_s = 1.5 / native_hz  # split at any dropped native frame -> uniformly-sampled runs
    non_ball = frames[~frames["is_ball"].to_numpy(bool)]
    keep = detected_mask(non_ball["visibility"], provider=provider, assume_observed=False)
    det = non_ball[keep]
    rows: list[dict[str, object]] = []
    for (period, team, pid), pg in det.groupby(["period_id", "team_id", "player_id"], sort=True, observed=True):
        pg = pg.sort_values("time_seconds")
        tp = pg["time_seconds"].to_numpy(dtype=np.float64)
        for axis in ("x", "y"):
            vals = pg[axis].to_numpy(dtype=np.float64)
            for run_index, (s, e) in enumerate(_split_indices(tp, native_hz, gap_s)):
                try:
                    cut = residual_cutoff_for_run(vals[s:e], native_hz)
                except ValueError:
                    continue
                rows.append(
                    {
                        "provider": provider,
                        "match_id": match_id,
                        "period_id": int(period),
                        "team_id": team,
                        "player_id": pid,
                        "axis": axis,
                        "run_index": run_index,
                        "cutoff_hz": cut,
                    }
                )
    return pd.DataFrame(rows, columns=_PASS_A_COLUMNS)


def _pass_a(args, prov) -> CorpusPassResult:
    from scripts._driver import for_each
    from scripts._partition import worker_tag

    dest = pathlib.Path(args.out)
    tag = worker_tag(args.match_ids_json)
    timer = StageTimer()
    refs, load = corpus_source(args)
    visibility_preflight(refs, load)  # ADR-069 Layer 2: before any compute
    with timer("corpus"):
        res = for_each(
            refs,
            key=lambda r: r.key,
            load=load,
            work=pass_a_match,
            shard_root=dest / "a_shards",
            token_inputs={
                "pass": "a",
                "grid": RESIDUAL_GRID.tolist(),
                "geometry_version": GEOMETRY_VERSION,
                "schema": _SHARD_SCHEMA_VERSION,
                "params": run_params_token(args, _interim_params),  # M-4: the values the pass consumes (R2-3)
            },
            tag=tag,
            label="match",
        )
    write_worker_partial(dest, "a", res, prov, timer, tag=tag)
    return res


# --------------------------------------------------------------------------- Pass B (epsilon / ACF / freq / gap F1)
def _provider_cutoffs_from_pass_a(a_df: pd.DataFrame) -> dict[str, float]:
    """Per-provider Butterworth cutoff: the match-weighted median of that provider's per-run cutoffs (spec 8.2)."""
    out: dict[str, float] = {}
    for provider, g in a_df.groupby("provider", sort=True, observed=True):
        w = unit_weights(g["provider"].to_numpy(), g["match_id"].to_numpy(), provider_neutral=False)
        out[str(provider)] = provider_cutoff(g["cutoff_hz"].to_numpy(dtype=np.float64), w)
    return out


def _cluster_amplitude(period_signals, team) -> np.ndarray:
    """The Kuramoto order-parameter magnitude over team ``team``'s player x-phasors (NaN-aware, per grid point)."""
    phasors = [ps.phasor_x for (tm, _pid), ps in period_signals.players.items() if tm == team]
    if not phasors:
        return np.full(period_signals.t.size, np.nan)
    stack = np.vstack(phasors)  # (P, N) complex, NaN where a player has no phasor
    valid = np.isfinite(stack)
    count = valid.sum(axis=0)
    summed = np.where(valid, stack, 0.0 + 0.0j).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        amp = np.where(count > 0, np.abs(summed / np.maximum(count, 1)), np.nan)
    return amp


def _median_freq_over_segments(values: np.ndarray, segments, fs: float, min_n: int) -> tuple[float, float]:
    """Duration-weighted median frequency (cpm) over a team signal's stationary segments (mirrors compute_spectral).

    A segment shorter than ``min_n`` samples is skipped -- the SAME ``min_spectral_samples`` floor the metric applies,
    so a tiny segment cannot inflate ``band_high_cpm`` with a coarse, unresolved median (review A-39)."""
    if segments is None or len(segments) == 0:
        return float("nan"), 0.0
    mfs, durs = [], []
    for lo, hi in segments:
        seg = values[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size < min_n:
            continue
        mfs.append(median_frequency_cpm(seg, fs))
        durs.append(seg.size / fs)
    if not mfs:
        return float("nan"), 0.0
    return pooled_median_frequency(np.array(mfs), np.array(durs)), float(sum(durs))


def _acf_zero_over_segments(values: np.ndarray, segments, fs: float) -> float:
    """Duration-weighted mean first-ACF-zero-crossing (s) over a signal's stationary segments.

    Computed PER segment -- never over the concatenation of non-adjacent runs -- so the decorrelation estimate reads
    the signal the way the coordination families do (within a segment), not spliced across a gap or stoppage (A-39)."""
    if segments is None or len(segments) == 0:
        return float("nan")
    crossings, durs = [], []
    for lo, hi in segments:
        seg = values[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size < 2:
            continue
        c = first_acf_zero_crossing_s(seg, fs)
        if np.isfinite(c):
            crossings.append(c)
            durs.append(seg.size / fs)
    if not crossings:
        return float("nan")
    return float(np.average(crossings, weights=durs))


def pass_b_match(loaded, cutoff_hz: float) -> pd.DataFrame:
    """Per-signal noise (``vc_epsilon``), decorrelation (ACF zero crossing), rhythm (median freq) and, on event
    matches, the possession-gap boundary-F1 curve -- everything the D1 reduce needs beyond the Pass-A cutoff (spec 8.2).

    Signals are prepared ONCE at the provider's Pass-A cutoff (``n_surrogates=0``): the filtered build and the
    raw-resampled build (``apply_filter=False``) are aligned element-wise, so ``vc_epsilon`` is the MAD of their
    first-difference gap. Median frequency uses the filtered team signal over its stationary segments.
    """
    frames, actions = loaded.frames, loaded.actions
    provider, match_id = loaded.provider, str(loaded.match_id)
    params = dataclasses.replace(CoordinationParams(), butterworth_cutoff_hz=float(cutoff_hz), n_surrogates=0)
    windows = build_windows(frames, actions, params)
    filt = build_coordination_signals(frames, windows=windows, params=params)
    raw = _build_coordination_signals(frames, windows=windows, params=params, apply_filter=False)
    fs = filt.fs
    rows: list[dict[str, object]] = []

    def emit(period, team, pid, quantity, signal, value, *, gap_s=None, duration_s=None):
        rows.append(
            {
                "provider": provider,
                "match_id": match_id,
                "period_id": period,
                "team_id": team,
                "player_id": pid,
                "quantity": quantity,
                "signal": signal,
                "value": float(value),
                "gap_s": gap_s,
                "duration_s": duration_s,
            }
        )

    # The metric's own spectral floor: a segment shorter than this cannot resolve the low band, so D1 skips it too
    # (review A-39). Derived from the INTERIM band (pass-B runs on default params, like every other pass-B estimate).
    min_n = min_spectral_samples(fs, params.band_low_cpm)
    for pf, pr in zip(filt.periods, raw.periods, strict=True):
        period = int(pf.period_id)
        for team in pf.team_ids:
            segs = pf.segments.get(team)  # the SAME stationary segments the families read -- never splice across them
            for sig in TEAM_SIGNALS:
                a = pf.team_signal.get((team, sig))
                b = pr.team_signal.get((team, sig))
                if a is None or b is None:
                    continue
                emit(period, team, None, "vc_epsilon", sig, vc_epsilon_for_signal(b, a))
                emit(period, team, None, "acf_zero_s", sig, _acf_zero_over_segments(a, segs, fs))
                mf, dur = _median_freq_over_segments(a, segs, fs, min_n)
                emit(period, team, None, "median_freq_cpm", sig, mf, duration_s=dur)
            amp = _cluster_amplitude(pf, team)
            emit(period, team, None, "acf_zero_s", "cluster_amplitude", _acf_zero_over_segments(amp, segs, fs))
        for (team, pid), ps in pf.players.items():
            emit(period, team, pid, "acf_zero_s", "player_x", _acf_zero_over_segments(ps.x, ps.runs, fs))
            emit(period, team, pid, "acf_zero_s", "player_y", _acf_zero_over_segments(ps.y, ps.runs, fs))
    if actions is not None and len(actions):
        for gap, f1 in boundary_f1_by_gap(frames, actions, POSSESSION_GAP_GRID).items():
            emit(None, None, None, "boundary_f1", None, f1, gap_s=float(gap))
    return pd.DataFrame(rows, columns=_PASS_B_COLUMNS)


def _corpus_providers(args) -> tuple:
    """Every provider the ``--corpus-json`` lists (the full corpus), else the pass's ``--providers``. A corpus-wide
    combine expects ALL of the corpus, not just the worker's provider subset (review B m5); when there is no corpus
    list, :func:`expected_corpus` handles the unpartitioned / refusal cases from ``--providers``."""
    if getattr(args, "corpus_json", None):
        return tuple(json.loads(pathlib.Path(args.corpus_json).read_text(encoding="utf-8")).keys())
    return tuple(args.providers)


def _pass_b(args, prov) -> CorpusPassResult:
    from scripts._driver import for_each
    from scripts._partition import worker_tag

    dest = pathlib.Path(args.out)
    tag = worker_tag(args.match_ids_json)
    # The cutoffs are corpus-wide: EVERY worker's pass-a share, over EVERY provider the corpus lists -- NOT this
    # worker's `--providers` subset. Pass a ran the whole corpus, so a subset-slice pass-b worker must still expect all
    # of it; keying the expectation off `args.providers` fails such a worker closed ("add N"), review B m5.
    a_df, _a = combine_workers(dest, "a", expected=expected_corpus(args, _corpus_providers(args)))
    cutoffs = _provider_cutoffs_from_pass_a(a_df)
    default_cutoff = CoordinationParams().butterworth_cutoff_hz
    timer = StageTimer()
    refs, load = corpus_source(args)
    visibility_preflight(refs, load)  # ADR-069 Layer 2: before any compute

    def work(loaded):
        return pass_b_match(loaded, cutoffs.get(loaded.provider, default_cutoff))

    with timer("corpus"):
        res = for_each(
            refs,
            key=lambda r: r.key,
            load=load,
            work=work,
            shard_root=dest / "b_shards",
            token_inputs={
                "pass": "b",
                "cutoffs": {k: round(v, 6) for k, v in sorted(cutoffs.items())},
                "possession_gap_grid": list(POSSESSION_GAP_GRID),
                "geometry_version": GEOMETRY_VERSION,
                "schema": _SHARD_SCHEMA_VERSION,
                "params": run_params_token(args, _interim_params),  # M-4: the values the pass consumes (R2-3)
            },
            tag=tag,
            label="match",
        )
    write_worker_partial(dest, "b", res, prov, timer, tag=tag, extra={"cutoffs": cutoffs})
    return res


# --------------------------------------------------------------------------- Occlusion leg (spec 8.2, D9)
def occlusion_width_hist(loaded) -> pd.DataFrame:
    """Per candidate FOV width, this match's (detected, total) outfield counts -- the sharded W-calibration input."""
    frames = loaded.frames
    of = (~frames["is_ball"].to_numpy(bool)) & (~frames["is_goalkeeper"].to_numpy(bool))
    n_total = int(of.sum())
    rows = [
        {
            "provider": loaded.provider,
            "match_id": str(loaded.match_id),
            "width_m": w,
            "n_detected": int(fov_mask(frames, width_m=w)[of].sum()),
            "n_total": n_total,
        }
        for w in _WIDTH_GRID
    ]
    return pd.DataFrame(rows, columns=["provider", "match_id", "width_m", "n_detected", "n_total"])


def calibrate_width_from_hist(hist_df: pd.DataFrame) -> float:
    """The FOV width whose pooled outfield rate hits ``_TARGET_OUTFIELD``, linear-interpolated (rate rises with W)."""
    agg = (
        hist_df.groupby("width_m", observed=True)
        .agg(det=("n_detected", "sum"), tot=("n_total", "sum"))
        .reset_index()
        .sort_values("width_m")
    )
    w = agg["width_m"].to_numpy(dtype=np.float64)
    rate = (agg["det"] / agg["tot"].where(agg["tot"] > 0, 1)).to_numpy(dtype=np.float64)
    if rate[0] >= _TARGET_OUTFIELD:
        return float(w[0])
    if rate[-1] <= _TARGET_OUTFIELD:
        return float(w[-1])
    i = int(np.searchsorted(rate, _TARGET_OUTFIELD))
    return float(w[i - 1] + (_TARGET_OUTFIELD - rate[i - 1]) * (w[i] - w[i - 1]) / (rate[i] - rate[i - 1]))


def _obs_bin(x: float) -> float:
    """Observed-fraction decile upper edge in {0.1, ..., 1.0}."""
    return float(np.clip(np.ceil(round(x, 6) * 10.0) / 10.0, 0.1, 1.0))


def _window_obs_fraction(frames: pd.DataFrame, mask: np.ndarray, windows: pd.DataFrame) -> pd.DataFrame:
    """Per window, the mean outfield FOV-detection fraction over its frames -- the occlusion binning axis (spec 8.2)."""
    of = (~frames["is_ball"].to_numpy(bool)) & (~frames["is_goalkeeper"].to_numpy(bool))
    t = frames["time_seconds"].to_numpy(dtype=np.float64)
    g = frames["game_id"].to_numpy(dtype=object)
    p = frames["period_id"].to_numpy()
    rows = []
    for w in windows.itertuples(index=False):
        sel = of & (g == w.game_id) & (p == w.period_id) & (t >= w.start_time_s) & (t < w.end_time_s)
        rows.append(
            {
                "game_id": w.game_id,
                "period_id": w.period_id,
                "window_kind": w.window_kind,
                "window_id": w.window_id,
                "obs_frac": float(mask[sel].mean()) if sel.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _bridge_rmse(frames: pd.DataFrame, mask: np.ndarray, gap_s: float) -> float:
    """RMSE of linearly-bridged vs true positions over masked runs no longer than ``gap_s`` (the D9 gap curve)."""
    native_hz = float(frames["frame_rate"].iloc[0])
    max_run = round(gap_s * native_hz)
    x = frames["x"].to_numpy(dtype=np.float64)
    y = frames["y"].to_numpy(dtype=np.float64)
    sq: list[float] = []
    for _key, rows in player_period_row_groups(frames):  # within (game, period, player): the clock is per period
        if rows.size < 3:
            continue
        det = mask[rows]
        i = 0
        while i < rows.size:
            if det[i]:
                i += 1
                continue
            j = i
            while j < rows.size and not det[j]:
                j += 1
            if 0 < (j - i) <= max_run and i > 0 and j < rows.size:
                lo, hi = i - 1, j
                for k in range(i, j):
                    frac = (k - lo) / (hi - lo)
                    xe = x[rows[lo]] + (x[rows[hi]] - x[rows[lo]]) * frac
                    ye = y[rows[lo]] + (y[rows[hi]] - y[rows[lo]]) * frac
                    sq.append(float((xe - x[rows[k]]) ** 2 + (ye - y[rows[k]]) ** 2))
            i = j
    return float(np.sqrt(np.mean(sq))) if sq else float("nan")


def _position_noise_rms(frames: pd.DataFrame) -> float:
    """The residual-analysis noise floor (spec 8.2): the median over player-axis runs of Winter's noise-floor RMS --
    the intercept of the residual curve's high-frequency tail (``residual_analysis_noise_rms``), NOT the residual at
    the fixed interim 0.4-Hz cutoff (review A-14). A run too short, non-finite, or noise-free (ill-posed) is skipped."""
    from silly_kicks.tracking.preprocess._butterworth import butterworth_min_length, residual_analysis_noise_rms

    native_hz = float(frames["frame_rate"].iloc[0])
    params = CoordinationParams()
    min_len = butterworth_min_length(native_hz, params.butterworth_cutoff_hz, params.butterworth_order)
    floors: list[float] = []
    players = frames[~frames["is_ball"].to_numpy(bool)]
    for _key, pg in players.groupby(["game_id", "period_id", "player_id"], sort=False, observed=True):
        pg = pg.sort_values("time_seconds", kind="mergesort")  # within one period: the clock is period-relative
        for axis in ("x", "y"):
            v = pg[axis].to_numpy(dtype=np.float64)
            if v.size < min_len or not np.isfinite(v).all():
                continue
            try:
                floors.append(residual_analysis_noise_rms(v, native_hz, RESIDUAL_GRID, order=params.butterworth_order))
            except ValueError:
                continue  # noise-free run: no identifiable floor (ill-posed), skip
    return float(np.median(floors)) if floors else float("nan")


def occlusion_metrics(loaded, width_m: float, *, params: CoordinationParams | None = None) -> pd.DataFrame:
    """Full-vs-occluded coordination error per family (binned by observed fraction), the bridged-position RMSE per
    gap, the noise floor and the goalkeeper rate -- everything the reduce needs for ``min_observed_fraction`` /
    ``max_detection_gap_s`` and the 19.6% check (spec 8.2). Occlusion bites through EXTRAPOLATED positions and the
    observed fraction comes from ``fov_mask`` (owner ruling: spec-based, external observed-fraction)."""
    frames, actions = loaded.frames, loaded.actions
    provider, match_id = loaded.provider, str(loaded.match_id)
    # D1 derives with the interim base (Tier B does not exist yet); D3 hands its FINAL params (review A-10)
    params = dataclasses.replace(params if params is not None else CoordinationParams(), n_surrogates=0)
    windows = build_windows(frames, actions, params)
    mask = fov_mask(frames, width_m=width_m)
    occ_frames = simulate_broadcast_occlusion(frames, width_m=width_m)
    full = _result_from_signals(build_coordination_signals(frames, windows=windows, params=params))
    occ = _result_from_signals(build_coordination_signals(occ_frames, windows=windows, params=params))
    obs = _window_obs_fraction(frames, mask, windows)
    rows: list[dict[str, object]] = []
    # A-09 / C.8.6: full-vs-occluded error PER CONSTRUCT (column x the C.1 what-keys) within each family table, so the
    # reduce can take the fail-closed family-MAX over its constructs and record each construct's own curve.
    for table, key_cols in CONSTRUCT_KEY_COLS.items():
        if table not in _TABLE_KEYS:  # occlusion covers the five family tables (they span every governed family)
            continue
        ft, ot = getattr(full, table), getattr(occ, table)
        if not len(ft):
            continue
        full_keys = [k for k in _TABLE_KEYS[table] if k in ft.columns and k in ot.columns]
        ckeys = [k for k in key_cols if k in ft.columns]
        for col in reliability_scored_columns(table):
            if col not in ft.columns or col not in ot.columns:
                continue
            family = _column_family(col)
            if family is None:
                continue
            kind = reliability_kind(col)
            merged = ft[[*full_keys, col]].merge(ot[[*full_keys, col]], on=full_keys, suffixes=("_full", "_occ"))
            merged = merged.merge(obs, on=[k for k in _WINDOW_KEYS if k in full_keys], how="left")
            fv = pd.to_numeric(merged[f"{col}_full"], errors="coerce").to_numpy()
            ov = pd.to_numeric(merged[f"{col}_occ"], errors="coerce").to_numpy()
            merged = merged.assign(
                _full=fv,
                _err=np.abs(wrap_deg(ov - fv)) if kind == "circular" else np.abs(ov - fv),
            )
            groups = merged.groupby(ckeys, sort=True, dropna=False, observed=True) if ckeys else [((), merged)]
            for ckey_vals, grp in groups:
                construct = _construct_str(col, ckeys, ckey_vals)
                gfull = grp["_full"].to_numpy(dtype=np.float64)
                full_mean = _circ_mean_deg(gfull) if kind == "circular" else float(np.nanmean(gfull))
                rows.append(
                    {
                        "provider": provider,
                        "match_id": match_id,
                        "quantity": "occ_full",
                        "family": family,
                        "construct": construct,
                        "column": col,
                        "kind": kind,
                        "obs_bin": None,
                        "gap_s": None,
                        "value": full_mean,
                    }
                )
                for ob, e in zip(grp["obs_frac"].to_numpy(), grp["_err"].to_numpy(), strict=True):
                    if not (np.isfinite(ob) and np.isfinite(e)):
                        continue
                    rows.append(
                        {
                            "provider": provider,
                            "match_id": match_id,
                            "quantity": "occ_err",
                            "family": family,
                            "construct": construct,
                            "column": col,
                            "kind": kind,
                            "obs_bin": _obs_bin(float(ob)),
                            "gap_s": None,
                            "value": float(e),
                        }
                    )
    for gap in _DETECTION_GAP_GRID:
        rows.append(
            {
                "provider": provider,
                "match_id": match_id,
                "quantity": "bridge_rmse",
                "family": None,
                "obs_bin": None,
                "gap_s": float(gap),
                "value": _bridge_rmse(frames, mask, float(gap)),
            }
        )
    rows.append(
        {
            "provider": provider,
            "match_id": match_id,
            "quantity": "noise_rms",
            "family": None,
            "obs_bin": None,
            "gap_s": None,
            "value": _position_noise_rms(frames),
        }
    )
    _, gk_rate = detection_rates(frames, mask)
    rows.append(
        {
            "provider": provider,
            "match_id": match_id,
            "quantity": "gk_rate",
            "family": None,
            "obs_bin": None,
            "gap_s": None,
            "value": gk_rate,
        }
    )
    return pd.DataFrame(rows, columns=_OCC_COLUMNS)


def _occlusion_args(args):
    """A shallow copy of ``args`` restricted to the fully-observed occlusion providers (spec 8.2, D9)."""
    providers = tuple(p for p in args.providers if p in OCCLUSION_PROVIDERS)
    return argparse.Namespace(**{**vars(args), "providers": providers or OCCLUSION_PROVIDERS})


def _pass_occlusion_cal(args, prov) -> CorpusPassResult:
    """Sub-pass 1 of the occlusion leg: this worker's share of the FOV-width calibration histogram."""
    from scripts._driver import for_each
    from scripts._partition import worker_tag

    dest = pathlib.Path(args.out)
    tag = worker_tag(args.match_ids_json)
    timer = StageTimer()
    refs, load = corpus_source(_occlusion_args(args))
    visibility_preflight(refs, load)  # ADR-069 Layer 2: before any compute
    with timer("calibrate"):
        res = for_each(
            refs,
            key=lambda r: r.key,
            load=load,
            work=occlusion_width_hist,
            shard_root=dest / "occlusion_cal_shards",
            token_inputs={
                "pass": "occlusion_cal",
                "widths": list(_WIDTH_GRID),
                "schema": _SHARD_SCHEMA_VERSION,
                "params": run_params_token(args, _interim_params),  # M-4: the values the pass consumes (R2-3)
            },
            tag=tag,
            label="match",
        )
    write_worker_partial(dest, "occlusion_cal", res, prov, timer, tag=tag)
    return res


def _pass_occlusion(args, prov) -> CorpusPassResult:
    """Sub-pass 2: ONE width W interpolated from EVERY worker's histogram (B-1: a per-worker W would split the corpus
    into differently calibrated regimes), then this worker's occluded-vs-truth error curves at W."""
    from scripts._driver import for_each
    from scripts._partition import worker_tag

    dest = pathlib.Path(args.out)
    tag = worker_tag(args.match_ids_json)
    occ_args = _occlusion_args(args)
    hist, _cal = combine_workers(dest, "occlusion_cal", expected=expected_corpus(args, occ_args.providers))
    width_m = calibrate_width_from_hist(hist) if len(hist) else 40.0
    timer = StageTimer()
    refs, load = corpus_source(occ_args)
    visibility_preflight(refs, load)  # ADR-069 Layer 2: before any compute
    with timer("corpus"):
        res = for_each(
            refs,
            key=lambda r: r.key,
            load=load,
            work=lambda loaded: occlusion_metrics(loaded, width_m),
            shard_root=dest / "occlusion_shards",
            token_inputs={
                "pass": "occlusion",
                "width_m": round(width_m, 4),
                "detection_gap_grid": list(_DETECTION_GAP_GRID),
                "geometry_version": GEOMETRY_VERSION,
                "schema": _SHARD_SCHEMA_VERSION,
                "params": run_params_token(args, _interim_params),  # M-4: the values the pass consumes (R2-3)
            },
            tag=tag,
            label="match",
        )
    write_worker_partial(
        dest, "occlusion", res, prov, timer, tag=tag, extra={"width_m": width_m, "n_calibrated": len(hist)}
    )
    return res


def _interim_params(_provider: str) -> CoordinationParams:
    """The params every D1 pass computes with: the interim base (Tier B is what D1 derives, so it cannot use it)."""
    return CoordinationParams()


# --------------------------------------------------------------------------- reduce (per-provider + pooled, A3)
#: The occlusion leg's outputs (spec 8.2). They are derived ONCE, from the fully observed providers (GS + IDSSE)
#: simulating SkillCorner's broadcast camera, for the detection-aware provider -- not per-provider quantities. Every
#: provider block therefore carries the pooled value: a provider without occlusion units (SkillCorner itself) must
#: never fall back to "cannot assess" and override exactly the values the leg exists to derive (review A-05).
OCCLUSION_KEYS = ("max_detection_gap_s", "min_observed_fraction")


def _weighted_by_signal(
    df: pd.DataFrame, q: float, *, provider_neutral: bool, key: str, why: dict[str, str], fallback: bool = True
) -> dict[str, float]:
    """Per ``signal``, the weighted ``q`` quantile of ``value`` under :func:`unit_weights` (one weighting scheme).

    Every signal of the interim base gets a value: one without a finite unit falls back to its interim value with a
    reason under ``why`` (``key.signal``), never a silent NaN (review A-32). ``fallback=False`` (the thin-provider
    report's per-provider estimates) leaves NaN instead."""
    base = dict(getattr(CoordinationParams(), key))
    out: dict[str, float] = {}
    groups = {str(sig): g for sig, g in df.groupby("signal", sort=True, observed=True)} if len(df) else {}
    for sig in sorted(set(base) | set(groups)):
        g = groups.get(sig)
        vals = g["value"].to_numpy(dtype=np.float64) if g is not None else np.empty(0)
        m = np.isfinite(vals)
        if g is None or not m.any():
            if fallback and sig in base:
                out[sig] = float(base[sig])
                why[f"{key}.{sig}"] = "no finite unit for this signal: the interim value stands"
            else:
                out[sig] = float("nan")
            continue
        w = unit_weights(g["provider"].to_numpy()[m], g["match_id"].to_numpy()[m], provider_neutral=provider_neutral)
        out[sig] = weighted_quantile(vals[m], q, w)
    return out


def _reduce_band(
    med_df: pd.DataFrame, *, provider_neutral: bool, why: dict[str, str], fallback: bool = True
) -> tuple[float, float]:
    vals = med_df["value"].to_numpy(dtype=np.float64) if len(med_df) else np.empty(0)
    m = np.isfinite(vals)
    base = CoordinationParams()
    if not m.any():
        if not fallback:
            return float("nan"), float("nan")
        why["band_low_cpm"] = why["band_high_cpm"] = "no team-signal median frequency: the interim band stands"
        return base.band_low_cpm, base.band_high_cpm
    w = unit_weights(
        med_df["provider"].to_numpy()[m], med_df["match_id"].to_numpy()[m], provider_neutral=provider_neutral
    )
    band = band_from_median_frequencies(vals[m], w)
    if not (band[0] < band[1]):
        # A degenerate band (identical median frequencies -> zero width) is unusable for the Welch rule. On the real
        # corpus the team-signal median frequencies span a range, so this bites only a thin or uniform slice.
        if not fallback:
            return float("nan"), float("nan")
        why["band_low_cpm"] = why["band_high_cpm"] = "degenerate band (zero width): the interim band stands"
        return base.band_low_cpm, base.band_high_cpm
    return band


def _reduce_possession_gap(
    f1_df: pd.DataFrame, *, provider_neutral: bool, why: dict[str, str], fallback: bool = True
) -> float:
    curve: dict[float, float] = {}
    for gap, grp in f1_df.groupby("gap_s", observed=True) if len(f1_df) else []:
        vals = grp["value"].to_numpy(dtype=np.float64)
        m = np.isfinite(vals)
        if not m.any():
            continue
        w = unit_weights(
            grp["provider"].to_numpy()[m], grp["match_id"].to_numpy()[m], provider_neutral=provider_neutral
        )
        curve[_as_float(gap)] = float(np.sum(vals[m] * w) / np.sum(w))
    if not curve:
        if not fallback:
            return float("nan")
        why["possession_gap_s"] = "no event+tracking boundary-F1 unit: the interim gap stands"
        return CoordinationParams().possession_gap_s
    return possession_gap_argmax(curve)


def _between_sd(full_vals: np.ndarray, kind: str) -> float:
    """Between-match SD of a construct's full-observation value: circular SD (degrees) for a circular-mean construct,
    else the sample SD. Infinite when the circular resultant is zero (no concentration)."""
    v = full_vals[np.isfinite(full_vals)]
    if v.size < 2:
        return float("nan")
    if kind == "circular":
        rad = np.radians(v)
        r = float(np.hypot(np.mean(np.cos(rad)), np.mean(np.sin(rad))))
        return float(np.degrees(np.sqrt(-2.0 * np.log(r)))) if r > 0 else float("inf")
    return float(np.std(v, ddof=1))


#: Seeded bootstrap for the per-bin occlusion-error CI (reproducible; the D1 reduce is a once-off aggregation).
_OCCLUSION_CI_SEED = 20260926
_OCCLUSION_CI_DRAWS = 400
#: Per-bin CI cost cap. A bin's ``occ_err`` can hold millions of rows; the weighted-median CI from a seeded subsample
#: of this many rows is indistinguishable from the full-bin CI (the median's sampling SD falls as 1/sqrt(n) and is
#: already negligible at this size), while each of the ``_OCCLUSION_CI_DRAWS`` resamples then sorts at most this many
#: rows instead of the whole bin. The POINT estimate (``err_by_bin``) is unaffected -- the caller computes it on the
#: full bin. (ADR-112 follow-up: the old row-grain 400x full-bin bootstrap was the D1-reduce hotspot.)
_OCCLUSION_CI_MAX_ROWS = 20_000


def _bootstrap_weighted_median_ci(values: np.ndarray, weights: np.ndarray, match_ids: np.ndarray) -> list[float]:
    """Seeded percentile 95% CI of the weighted-median occlusion error in one bin (B-R3-02 / C.5 "n/CI per bin").

    CLUSTER bootstrap at MATCH grain: the resampling unit is the match, not the row. ``occ_err`` rows within a match
    are not independent (same broadcast, same possessions), so a row-level bootstrap understates the CI -- and
    resampling the full million-row bin ``_OCCLUSION_CI_DRAWS`` times was the D1-reduce hotspot (ADR-112 follow-up).
    Rows are first subsampled (seeded) to ``_OCCLUSION_CI_MAX_ROWS`` so each draw sorts at most that many; the caller's
    point estimate uses the full bin and is unaffected. Needs >= 2 matches (else the CI is undefined -> NaN).
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(v)
    v, w = v[finite], w[finite]
    codes = pd.factorize(np.asarray(match_ids, dtype=object)[finite])[0]  # match -> int cluster id (fast, order-stable)
    if v.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(_OCCLUSION_CI_SEED)
    order = np.argsort(codes, kind="stable")  # group rows into match clusters (one np.argsort over the full bin)
    cut = np.flatnonzero(codes[order][1:] != codes[order][:-1]) + 1
    clusters = np.split(order, cut)
    if len(clusters) < 2:
        return [float("nan"), float("nan")]
    # Per-match cap so every draw's pooled size is bounded: a cluster bootstrap over UNEVEN clusters could otherwise
    # pool far above the cap by picking a big cluster repeatedly. Balanced clusters -> each draw sorts <= the cap.
    per_match_cap = max(1, _OCCLUSION_CI_MAX_ROWS // len(clusters))
    capped = [
        np.sort(rng.choice(c, size=per_match_cap, replace=False)) if c.size > per_match_cap else c for c in clusters
    ]
    stats = []
    for _ in range(_OCCLUSION_CI_DRAWS):
        pick = rng.integers(0, len(capped), size=len(capped))
        rows = np.concatenate([capped[j] for j in pick])
        stats.append(weighted_quantile(v[rows], 0.5, w[rows]))
    lo, hi = (float(x) for x in np.percentile(stats, [2.5, 97.5]))
    return [lo, hi]


def _per_construct_shares(
    occ_df: pd.DataFrame, *, provider_neutral: bool, min_matches: int
) -> dict[str, dict[str, dict]]:
    """Per family, per construct: the occlusion-error curve (weighted median per bin), the match count per bin, the
    between-match SD and the qualifying share + estimable verdict (A-09 / C.8.6). ``{family: {construct: record}}``."""
    if not len(occ_df) or "construct" not in occ_df.columns:
        return {}
    err = occ_df[occ_df["quantity"] == "occ_err"]
    full = occ_df[occ_df["quantity"] == "occ_full"]
    if not len(err):
        return {}
    out: dict[str, dict[str, dict]] = {}
    for (family, construct), ge in err.groupby(["family", "construct"], observed=True):
        kind = str(ge["kind"].iloc[0])
        gf = full[(full["family"] == family) & (full["construct"] == construct)] if len(full) else full
        full_vals = gf["value"].to_numpy(dtype=np.float64) if len(gf) else np.empty(0)
        if full_vals[np.isfinite(full_vals)].size < 2:
            rec = {
                "share": 1.0,
                "estimable": False,
                "reason": "insufficient_full_obs",
                "threshold": None,
                "crossing_bin": None,
                "curve": {},
                "n_by_bin": {},
                "ci_by_bin": {},
                "kind": kind,
                "between_sd": float("nan"),
            }
        else:
            sd = _between_sd(full_vals, kind)
            err_by_bin: dict[float, float] = {}
            n_by_bin: dict[float, int] = {}
            ci_by_bin: dict[float, list[float]] = {}  # B-R3-02: per-bin CI (spec/C.5 "curve + n/CI per bin")
            for ob, g in ge.groupby("obs_bin", observed=True):
                w = unit_weights(g["provider"].to_numpy(), g["match_id"].to_numpy(), provider_neutral=provider_neutral)
                vals = g["value"].to_numpy(dtype=np.float64)
                err_by_bin[_as_float(ob)] = weighted_quantile(vals, 0.5, w)
                n_by_bin[_as_float(ob)] = len(pd.unique(g["match_id"]))
                ci_by_bin[_as_float(ob)] = _bootstrap_weighted_median_ci(vals, w, g["match_id"].to_numpy())
            share = construct_qualifying_share(err_by_bin, n_by_bin, sd, min_matches=min_matches)
            rec = {
                **share,
                "curve": err_by_bin,
                "n_by_bin": n_by_bin,
                "ci_by_bin": ci_by_bin,
                "kind": kind,
                "between_sd": float(sd),
            }
        out.setdefault(str(family), {})[str(construct)] = rec
    return out


def _reduce_min_observed(
    occ_df: pd.DataFrame, *, provider_neutral: bool, why: dict[str, str], fallback: bool = True
) -> dict[str, float]:
    """Per family, the **consumed** ``min_observed_fraction`` = the fail-closed MAX over the family's constructs'
    qualifying shares (A-09 / C.8.6, owner ruling 2026-10-04). A family with no estimable construct falls back to 1.0
    (require full observation) WITH its reason; ``fallback=False`` leaves NaN (the honest per-provider estimate)."""
    shares = _per_construct_shares(
        occ_df, provider_neutral=provider_neutral, min_matches=thr.OCCLUSION_MIN_MATCHES_PER_BIN
    )
    out: dict[str, float] = {}
    for family in COORD_METHOD_FAMILIES:
        fam_constructs = shares.get(family, {})
        if not fam_constructs:
            out[family] = 1.0 if fallback else float("nan")
            if fallback:
                why[f"min_observed_fraction.{family}"] = "no occlusion constructs: 1.0 (require full observation)"
            continue
        fam = family_max_observed_fraction(fam_constructs)
        if fam["finding"] and not fallback:
            out[family] = float("nan")
        else:
            out[family] = float(fam["threshold"])
        if fallback and fam["finding"]:
            why[f"min_observed_fraction.{family}"] = (
                f"binding construct {fam['binding_construct']} non-estimable or no sub-1.0 crossing: "
                f"{fam['threshold']:.1f} (fail-closed family MAX)"
            )
    return out


def occlusion_per_construct_detail(occ_df: pd.DataFrame, *, provider_neutral: bool) -> dict[str, dict]:
    """The per-construct occlusion analysis recorded in ``derivation.json`` (A-09 / C.8.6): per family, the consumed
    family-MAX threshold, its binding construct, the over-restriction diagnostic (how much more strictly the MAX gates
    than each non-binding construct's own share -- the owner's decision input for the per-construct-consumption
    follow-up), a FINDING flag, and every construct's curve + per-bin match count + per-bin CI + estimable verdict."""
    shares = _per_construct_shares(
        occ_df, provider_neutral=provider_neutral, min_matches=thr.OCCLUSION_MIN_MATCHES_PER_BIN
    )
    detail: dict[str, dict] = {}
    for family in COORD_METHOD_FAMILIES:
        fam_constructs = shares.get(family, {})
        if not fam_constructs:
            detail[family] = {
                "threshold": 1.0,
                "binding_construct": None,
                "over_restriction": {},
                "finding": True,
                "constructs": {},
            }
            continue
        fam = family_max_observed_fraction(fam_constructs)
        detail[family] = {
            "threshold": fam["threshold"],
            "binding_construct": fam["binding_construct"],
            "over_restriction": fam["over_restriction"],
            "finding": fam["finding"],
            "constructs": {
                c: {
                    k: rec[k]
                    for k in (
                        "share",
                        "estimable",
                        "reason",
                        "crossing_bin",
                        "curve",
                        "n_by_bin",
                        "ci_by_bin",
                        "kind",
                        "between_sd",
                    )
                }
                for c, rec in fam_constructs.items()
            },
        }
    return detail


def _reduce_max_gap(
    occ_df: pd.DataFrame, *, provider_neutral: bool, why: dict[str, str], fallback: bool = True
) -> float:
    interim = CoordinationParams().max_detection_gap_s
    rmse = occ_df[occ_df["quantity"] == "bridge_rmse"] if len(occ_df) else occ_df
    noise = occ_df[occ_df["quantity"] == "noise_rms"] if len(occ_df) else occ_df
    reason = None
    if not len(occ_df):
        reason = "no occlusion units"
    elif not len(rmse) or not len(noise):
        reason = "no bridged-RMSE or noise unit"
    if reason is None:
        nv = noise["value"].to_numpy(dtype=np.float64)
        nm = np.isfinite(nv)
        if not nm.any():
            reason = "no finite noise RMS"
    if reason is None:
        nw = unit_weights(
            noise["provider"].to_numpy()[nm], noise["match_id"].to_numpy()[nm], provider_neutral=provider_neutral
        )
        noise_rms = weighted_quantile(nv[nm], 0.5, nw)
        rmse_by_gap: dict[float, float] = {}
        for gap, grp in rmse.groupby("gap_s", observed=True):
            vals = grp["value"].to_numpy(dtype=np.float64)
            m = np.isfinite(vals)
            if not m.any():
                continue
            w = unit_weights(
                grp["provider"].to_numpy()[m], grp["match_id"].to_numpy()[m], provider_neutral=provider_neutral
            )
            rmse_by_gap[_as_float(gap)] = weighted_quantile(vals[m], 0.5, w)
        if not rmse_by_gap:
            reason = "no finite bridged RMSE"
        else:
            try:
                return max_detection_gap_rule(rmse_by_gap, noise_rms)
            except ValueError:
                if fallback:
                    why["max_detection_gap_s"] = "every gap's bridged RMSE exceeds 2 x the noise: the shortest gap"
                return float(min(rmse_by_gap)) if fallback else float("nan")
    if fallback:
        why["max_detection_gap_s"] = f"{reason}: the interim gap stands"
        return interim
    return float("nan")


def _reduce_params(
    a_df,
    b_df,
    occ_df,
    *,
    provider_neutral: bool,
    restrict: str | None,
    why: dict[str, str] | None = None,
    welch_shortfall: dict[str, bool] | None = None,
    fallback: bool = True,
) -> dict[str, object]:
    """One full Tier-B params block (all INTERIM_BASE keys) under one weighting scheme.

    ``why`` collects one reason per value that FELL BACK (``key`` or ``key.signal`` / ``key.family``), so
    ``derivation.json`` never carries a silent default (review A-32); ``welch_shortfall`` receives the spec-8.2
    shortfall flag of the Welch rule (resolution kept, fewer than 8 segments). ``fallback=False`` returns NaN where a
    value cannot be derived -- the thin-provider report's honest per-provider estimates."""
    why = why if why is not None else {}
    if restrict is not None:
        a_df = a_df[a_df["provider"] == restrict] if len(a_df) else a_df
        b_df = b_df[b_df["provider"] == restrict] if len(b_df) else b_df
        occ_df = occ_df[occ_df["provider"] == restrict] if len(occ_df) else occ_df
    if len(a_df):
        cutoff = provider_cutoff(
            a_df["cutoff_hz"].to_numpy(dtype=np.float64),
            unit_weights(a_df["provider"].to_numpy(), a_df["match_id"].to_numpy(), provider_neutral=provider_neutral),
        )
    elif fallback:
        cutoff = CoordinationParams().butterworth_cutoff_hz
        why["butterworth_cutoff_hz"] = "no pass-a run: the interim cutoff stands"
    else:
        cutoff = float("nan")

    def quantity(name: str) -> pd.DataFrame:
        return b_df[b_df["quantity"] == name] if len(b_df) else b_df

    band = _reduce_band(quantity("median_freq_cpm"), provider_neutral=provider_neutral, why=why, fallback=fallback)
    welch, shortfall = welch_segment_rule(band[0], band[1]) if np.isfinite(band).all() else (float("nan"), False)
    if welch_shortfall is not None:
        welch_shortfall["shortfall"] = bool(shortfall)
    return {
        "butterworth_cutoff_hz": cutoff,
        "max_detection_gap_s": _reduce_max_gap(occ_df, provider_neutral=provider_neutral, why=why, fallback=fallback),
        "min_observed_fraction": _reduce_min_observed(
            occ_df, provider_neutral=provider_neutral, why=why, fallback=fallback
        ),
        "vc_epsilon": _weighted_by_signal(
            quantity("vc_epsilon"), 0.5, provider_neutral=provider_neutral, key="vc_epsilon", why=why, fallback=fallback
        ),
        "min_shift_s": _weighted_by_signal(
            quantity("acf_zero_s"),
            0.95,
            provider_neutral=provider_neutral,
            key="min_shift_s",
            why=why,
            fallback=fallback,
        ),
        "band_low_cpm": band[0],
        "band_high_cpm": band[1],
        "welch_segment_s": welch,
        "possession_gap_s": _reduce_possession_gap(
            quantity("boundary_f1"), provider_neutral=provider_neutral, why=why, fallback=fallback
        ),
    }


def _provider_weights(a_df, b_df, occ_df, providers) -> dict[str, dict[str, dict[str, float]]]:
    """Per provider and pass, its total weight in the provider-neutral pooled base (each provider with units carries
    an equal share, TF58-PLAN-02) and its match count -- plan Task 20's diagnostic beside ``pooled``."""
    frames = {"a": a_df, "b": b_df, "occlusion": occ_df}
    counts = {
        name: {p: (int(df.loc[df["provider"] == p, "match_id"].nunique()) if len(df) else 0) for p in providers}
        for name, df in frames.items()
    }
    return {
        p: {
            "weight": {name: (1.0 / sum(1 for q in providers if c[q]) if c[p] else 0.0) for name, c in counts.items()},
            "n_matches": {name: c[p] for name, c in counts.items()},
        }
        for p in providers
    }


def _summary(values: np.ndarray, provider: np.ndarray, match: np.ndarray) -> dict[str, object]:
    finite = np.isfinite(values)
    v = values[finite]
    if not v.size:
        return {"n_units": 0, "n_matches": 0}
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n_units": int(v.size),
        "n_matches": len({(str(a), str(b)) for a, b in zip(provider[finite], match[finite], strict=True)}),
        "mean": float(v.mean()),
        **{name: float(x) for name, x in zip(("p05", "p25", "p50", "p75", "p95"), q, strict=True)},
    }


def _distribution_summaries(a_df, b_df, occ_df) -> dict[str, object]:
    """Every intermediate distribution the reducers consume (plan Task 20): per quantity its unweighted descriptive
    summary, broken down by the reducer's own key (signal, family, gap) where it has one."""

    def summ(df: pd.DataFrame, col: str = "value") -> dict[str, object]:
        if not len(df):
            return {"n_units": 0, "n_matches": 0}
        return _summary(df[col].to_numpy(dtype=np.float64), df["provider"].to_numpy(), df["match_id"].to_numpy())

    def by(df: pd.DataFrame, key: str) -> dict[str, object]:
        return {str(k): summ(g) for k, g in df.groupby(key, sort=True, observed=True)} if len(df) else {}

    out: dict[str, object] = {
        "cutoff_hz": summ(a_df, "cutoff_hz") if "cutoff_hz" in a_df.columns else summ(a_df.iloc[0:0])
    }
    for quantity, key in (
        ("median_freq_cpm", "signal"),
        ("vc_epsilon", "signal"),
        ("acf_zero_s", "signal"),
        ("boundary_f1", "gap_s"),
    ):
        sub = b_df[b_df["quantity"] == quantity] if len(b_df) else b_df
        out[quantity] = {**summ(sub), f"by_{key}": by(sub, key)}
    for quantity, key in (
        ("bridge_rmse", "gap_s"),
        ("occ_err", "family"),
        ("occ_full", "family"),
        ("noise_rms", None),
        ("gk_rate", None),
    ):
        sub = occ_df[occ_df["quantity"] == quantity] if len(occ_df) else occ_df
        out[quantity] = {**summ(sub), **({f"by_{key}": by(sub, key)} if key else {})}
    return out


def _match_weighted_quantile(q: float) -> Callable[[pd.DataFrame], float]:
    """A reducer over a single provider's units: the ``q`` quantile of ``value`` match-weighted within it."""

    def reducer(u: pd.DataFrame) -> float:
        vals = u["value"].to_numpy(dtype=np.float64)
        finite = np.isfinite(vals)
        if not finite.any():
            return float("nan")
        w = unit_weights(u["provider"].to_numpy()[finite], u["match_id"].to_numpy()[finite], provider_neutral=False)
        return weighted_quantile(vals[finite], q, w)

    return reducer


def _thin_providers(a_df, b_df, occ_df, per_provider, *, n_boot: int) -> dict[str, object]:
    """Per Tier-B quantity -- EVERY scalar AND every per-signal/per-family map entry (TF58-PLAN-03) -- each
    provider's estimate, its match-level bootstrap SE (``provider_bootstrap_se``, matches resampled with
    replacement) and the resulting flag. REPORT ONLY: nothing here re-weights or drops a provider; a flag is
    surfaced for the owner (Task 28 Step 1 stops on any). ``not_assessable`` when fewer than two providers carry
    the quantity."""
    providers = sorted(per_provider)

    def _units(df: pd.DataFrame, provider: str, mask=None) -> pd.DataFrame:
        d = df[df["provider"] == provider] if len(df) else df
        if mask is not None and len(d):
            d = d[mask(d)]
        return d.assign(match=d["match_id"]) if len(d) else d

    def _block(values: dict, units_of, reducer, tag: tuple) -> dict[str, object]:
        present = {p: v for p, v in values.items() if isinstance(v, (int, float)) and np.isfinite(v)}
        se = {}
        for p in present:
            u = units_of(p)
            se[p] = provider_bootstrap_se(u, reducer, seed_key=(*tag, p), n_boot=n_boot) if len(u) else float("nan")
        return {"values": present, "se": se, "flags": thin_provider_flags(present, se)}

    def _scalar(quantity: str, units_of, reducer) -> dict[str, object]:
        return _block({p: per_provider[p][quantity] for p in providers}, units_of, reducer, (quantity,))

    a_val = a_df.assign(value=a_df["cutoff_hz"]) if "cutoff_hz" in a_df.columns else a_df

    def _band_reducer(idx: int):
        return lambda u: _reduce_band(u, provider_neutral=False, why={}, fallback=False)[idx]

    def _welch_reducer(u: pd.DataFrame) -> float:
        lo, hi = _reduce_band(u, provider_neutral=False, why={}, fallback=False)
        return welch_segment_rule(lo, hi)[0] if np.isfinite([lo, hi]).all() and lo < hi else float("nan")

    def med(p):
        return _units(b_df, p, lambda d: d["quantity"] == "median_freq_cpm")

    out: dict[str, object] = {
        "butterworth_cutoff_hz": _scalar(
            "butterworth_cutoff_hz", lambda p: _units(a_val, p), _match_weighted_quantile(0.5)
        ),
        "possession_gap_s": _scalar(
            "possession_gap_s",
            lambda p: _units(b_df, p, lambda d: d["quantity"] == "boundary_f1"),
            lambda u: _reduce_possession_gap(u, provider_neutral=False, why={}, fallback=False),
        ),
        "max_detection_gap_s": _scalar(
            "max_detection_gap_s",
            lambda p: _units(occ_df, p, lambda d: d["quantity"].isin(["bridge_rmse", "noise_rms"])),
            lambda u: _reduce_max_gap(u, provider_neutral=False, why={}, fallback=False),
        ),
        "band_low_cpm": _scalar("band_low_cpm", med, _band_reducer(0)),
        "band_high_cpm": _scalar("band_high_cpm", med, _band_reducer(1)),
        "welch_segment_s": _scalar("welch_segment_s", med, _welch_reducer),
    }

    def _signal_map(quantity: str, source_quantity: str, reducer) -> dict[str, object]:
        keys = sorted({k for p in providers for k in per_provider[p].get(quantity, {})})
        result = {}
        for key in keys:
            values = {p: per_provider[p].get(quantity, {}).get(key, float("nan")) for p in providers}

            def units_of(p, key=key):
                return _units(b_df, p, lambda d, key=key: (d["quantity"] == source_quantity) & (d["signal"] == key))

            result[key] = _block(values, units_of, reducer, (quantity, key))
        return result

    out["vc_epsilon"] = _signal_map("vc_epsilon", "vc_epsilon", _match_weighted_quantile(0.5))
    out["min_shift_s"] = _signal_map("min_shift_s", "acf_zero_s", _match_weighted_quantile(0.95))

    fam_keys = sorted({k for p in providers for k in per_provider[p].get("min_observed_fraction", {})})
    fam_out = {}
    for fam in fam_keys:
        values = {p: per_provider[p].get("min_observed_fraction", {}).get(fam, float("nan")) for p in providers}

        def units_of(p, fam=fam):
            return _units(
                occ_df, p, lambda d, fam=fam: (d["quantity"].isin(["occ_err", "occ_full"])) & (d["family"] == fam)
            )

        fam_out[fam] = _block(
            values,
            units_of,
            lambda u, fam=fam: _reduce_min_observed(u, provider_neutral=False, why={}, fallback=False)[fam],
            ("min_observed_fraction", fam),
        )
    out["min_observed_fraction"] = fam_out
    return out


def build_derivation(
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    occ_df: pd.DataFrame,
    prov,
    *,
    n_boot: int = 1000,
    occlusion_width_m: float | None = None,
    occlusion_n_calibrated: int | None = None,
) -> dict:
    """The full D1 derivation: per-provider (match-weighted) + pooled provider-neutral base (A3, TF58-PLAN-02),
    the thin-provider precision block, the occlusion width/gk-rate check, and provenance + the input contract.

    ``occlusion_width_m`` is the calibrated FOV width every occlusion worker used (their manifests agree); it is
    recorded in the occlusion block so the D3 validation driver can read the SAME W without re-calibrating."""
    providers = sorted(
        set(a_df.get("provider", pd.Series(dtype=str))) | set(b_df.get("provider", pd.Series(dtype=str)))
    )
    pooled_why: dict[str, str] = {}
    pooled_welch: dict[str, bool] = {}
    pooled = _reduce_params(
        a_df, b_df, occ_df, provider_neutral=True, restrict=None, why=pooled_why, welch_shortfall=pooled_welch
    )
    per_provider: dict[str, dict[str, object]] = {}
    estimates: dict[str, dict[str, object]] = {}  # honest per-provider estimates (NaN where underivable)
    provider_why: dict[str, dict[str, str]] = {}
    provider_welch: dict[str, dict[str, bool]] = {}
    for p in providers:
        why_p: dict[str, str] = {}
        welch_p: dict[str, bool] = {}
        block = _reduce_params(
            a_df, b_df, occ_df, provider_neutral=False, restrict=p, why=why_p, welch_shortfall=welch_p
        )
        for key in OCCLUSION_KEYS:  # corpus-level: the pooled occlusion-derived value, for every provider (A-05)
            block[key] = copy.deepcopy(pooled[key])
        per_provider[p] = block
        provider_why[p] = {k: v for k, v in why_p.items() if k.split(".")[0] not in OCCLUSION_KEYS}
        provider_welch[p] = welch_p
        estimates[p] = _reduce_params(a_df, b_df, occ_df, provider_neutral=False, restrict=p, fallback=False)
    gk = occ_df[occ_df["quantity"] == "gk_rate"]["value"].to_numpy(dtype=np.float64) if len(occ_df) else np.array([])
    return {
        "pooled": pooled,
        "providers": per_provider,
        # plan Task 20: each provider's weight and match count beside `pooled`, the corpus-representative (match-
        # weighted) value of every quantity, every intermediate distribution summary -- and (review A-32) the reason
        # for every value that fell back, plus the Welch rule's spec-8.2 shortfall flag
        "provider_weights": _provider_weights(a_df, b_df, occ_df, providers),
        "representative": _reduce_params(a_df, b_df, occ_df, provider_neutral=False, restrict=None),
        "summaries": _distribution_summaries(a_df, b_df, occ_df),
        "fallbacks": {"pooled": pooled_why, "providers": provider_why},
        "welch_shortfall": {
            "pooled": pooled_welch.get("shortfall", False),
            "providers": {p: w.get("shortfall", False) for p, w in provider_welch.items()},
        },
        "thin_providers": _thin_providers(a_df, b_df, occ_df, estimates, n_boot=n_boot) if estimates else {},
        "occlusion": {
            "gk_rate": float(np.nanmean(gk)) if gk.size else float("nan"),
            "gk_rate_target": _SKILLCORNER_GK_RATE,
            # A-09 / C.8.6: the per-construct curves + over-restriction diagnostic behind the per-family MAX threshold.
            "per_construct": occlusion_per_construct_detail(occ_df, provider_neutral=True),
            "min_matches_per_bin": thr.OCCLUSION_MIN_MATCHES_PER_BIN,  # C.5: the estimability floor, recorded
            "width_m": float(occlusion_width_m) if occlusion_width_m is not None else None,
            # review B m6: record WHETHER the width was calibrated from the FOV histogram or fell back to the 40 m
            # default (an empty histogram) -- a silent fallback was visible only in a worker manifest's n_calibrated.
            "n_calibrated": int(occlusion_n_calibrated) if occlusion_n_calibrated is not None else None,
            "width_source": (
                None
                if occlusion_n_calibrated is None
                else ("fallback_default" if occlusion_n_calibrated == 0 else "calibrated")
            ),
        },
        "input_contract": input_contract(),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
    }


def _reduce(args, prov) -> None:
    dest = pathlib.Path(args.out)
    # B-1: every worker's share of the three passes, proven complete against the listed corpus (or refused, spec 8.3);
    # the occlusion shares must all have used ONE calibrated width.
    occ_providers = _occlusion_args(args).providers
    a_df, a_sum = combine_workers(dest, "a", expected=expected_corpus(args, args.providers))
    b_df, b_sum = combine_workers(dest, "b", expected=expected_corpus(args, args.providers), consistent=("cutoffs",))
    occ_df, occ_sum = combine_workers(
        dest,
        "occlusion",
        expected=expected_corpus(args, occ_providers),
        consistent=("width_m", "n_calibrated"),
        categorical=True,  # reduce-memory: occ_err is the 465M-row OOM share
    )
    # the FOV-width calibration sub-pass is its own pass; surface its timing + population too (review m8: stage_seconds
    # omitted occlusion-cal). Its table (the histogram) is not an artifact input -- only its summary is used here.
    _occcal_df, occcal_sum = combine_workers(dest, "occlusion_cal", expected=expected_corpus(args, occ_providers))
    derivation = build_derivation(
        a_df,
        b_df,
        occ_df,
        prov,
        n_boot=THIN_PROVIDER_N_BOOT,
        occlusion_width_m=occ_sum["consistent"]["width_m"],
        occlusion_n_calibrated=occ_sum["consistent"].get("n_calibrated"),
    )
    summaries = {"a": a_sum, "b": b_sum, "occlusion-cal": occcal_sum, "occlusion": occ_sum}
    derivation["stage_seconds"] = {name: s["stage_seconds"] for name, s in summaries.items()}  # Ruling C(a) timers
    derivation["population"] = {
        name: {k: v for k, v in s.items() if k != "stage_seconds"} for name, s in summaries.items()
    }
    # ADR-038: the label of the matches the aggregates came from (spec 8.3; NDA-tier aggregates are allowed, labelled)
    pairs = table_pairs(a_df, b_df, occ_df)
    derivation["corpus_visibility"] = corpus_visibility_label(pairs, token=getattr(args, "token", None))
    # M-5 artifact handoff: both outputs land in --out, never the package or docs/research (D2/D3 read the artifact on
    # the clean commit-1 tree; commit 2 copies them in, where the codegen test pins the module to derivation.json).
    (dest / "derivation.json").write_text(json.dumps(derivation, indent=2, default=str), encoding="utf-8")
    (dest / GENERATED_ARTIFACT).write_text(render_generated_params(derivation, None), encoding="utf-8")
    print(json.dumps({"pooled": derivation["pooled"], "occlusion": derivation["occlusion"]}, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pass", dest="which", required=True, choices=["a", "b", "occlusion-cal", "occlusion", "reduce"])
    add_common_args(ap)
    args = ap.parse_args()

    from scripts._provenance import git_provenance, require_clean_tree

    if args.list_matches:
        refs, _ = corpus_source(args)
        print(json.dumps([list(r.key) for r in refs], indent=2))
        return
    if not args.out:
        raise SystemExit("--out is required")
    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    passes = {"a": _pass_a, "b": _pass_b, "occlusion-cal": _pass_occlusion_cal, "occlusion": _pass_occlusion}
    {**passes, "reduce": _reduce}[args.which](args, prov)


if __name__ == "__main__":
    main()
