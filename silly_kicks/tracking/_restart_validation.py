"""Restart-validation truth model: the data profile, the pre-registered constants, the statistics, and the
derived validation gate table (TF-46 Arm 1 Part 2, spec sections 5.3, 15, 16).

This module is the single home of the restart-refinement validation contract:

- ``DataProfile`` — the (provider, tracking release, GI release, competition-season) key a gate cell is scored
  under. It has NO pairing field: GI / tracking pairing is decided per MATCH, measured (spec section 6.1).
- ``RATIFIED`` / ``sensitivity_values`` — every section 16.1 constant, pre-registered before any gate is scored.
  The values are the ONLY source of truth; a test asserts a literal copy (anti-drift).
- ``wilson_lower`` / ``cluster_bootstrap_lower`` / ``cell_passes`` / ``noninferior`` / ``choose_base`` — the pure
  statistics (Wilson two-sided 95 %, match-cluster bootstrap, non-inferiority, base precedence). No statistic
  reads an outcome to decide a base; ``choose_base`` decides on sample adequacy only (spec section 5.3).
- ``RESTART_GATE_TABLE`` / ``gate`` / ``derive_gate_table`` — the gate table. It SHIPS EMPTY in commit 2
  (``MappingProxyType({})``; CL-3): every tracking-derived output is withheld and fail-closed. Commit 3 derives
  the real table from the committed pre-flight aggregate (ADR-056) through ``derive_gate_table``.
- ``FIDELITY_TOLERANCES`` / ``FIDELITY_DEPENDENCIES`` — the section 16.3 tier-2 fidelity contract and the
  dependency map (a failed statistic withholds only the signals that depend on it; the B-33 source-path rule
  puts the contact statistics on ``origin`` and the C-51 split puts them on ``penalty``).
- ``RESTART_REFINEMENT_COLUMNS`` — the section 17.1 additive action columns.

Pure numpy / pandas; no I/O. The bootstrap uses ``numpy.random.default_rng(seed)`` so a run is reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Literal, cast, get_args

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data profile (spec section 5.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataProfile:
    """The key a gate cell is scored under. No pairing field (spec section 6.1, C-11 ruling): GI / tracking
    pairing is per MATCH and measured by ``_restart_frame.GiPairing``, never a profile-level flag."""

    provider: str
    tracking_release: str
    gi_release: str | None
    competition_season: str


#: The missed-penalty signal carries three gates keyed by ``cell_state`` (CL-10, B-03): detection precision,
#: attribution accuracy, and the saved / off-target split.
PENALTY_CELL_STATES: tuple[str, ...] = ("detection", "attribution", "split")

#: Every validated signal. ``z_overhead`` and ``footprint`` are profile-level (restart_type ``"*"``,
#: cell_state ``"-"``); ``footprint`` is the section 16.1 camera-footprint gate (CL-16).
SIGNALS: tuple[str, ...] = (
    "kick_time",
    "taker",
    "first_contact",
    "subtype",
    "merge",
    "penalty",
    "origin",
    "z_overhead",
    "footprint",
)


# ---------------------------------------------------------------------------
# Gate entry and key (spec section 15)
# ---------------------------------------------------------------------------


#: The bases a cell can pass on (spec section 5.3).
GateBase = Literal["idsse", "gradientsports", "combined", "native"]


@dataclass(frozen=True)
class GateEntry:
    """One ratified cell: the passing tier and base, the shipped metric, the Wilson and cluster-bootstrap lower
    bounds, the sample size n, the match count m, and how the cell resolved (``"cell"`` or ``"pooled_<level>"``)."""

    tier: Literal["reference", "transfer"]
    base: GateBase
    metric: float
    wilson_lb: float
    bootstrap_lb: float
    n: int
    m: int
    cell_resolution: str
    #: ``kick_time`` only (spec section 6.5 / 16.1, amended 2026-10-22): the pre-registered agreement winner, the
    #: candidate (``tf79_contact`` / ``ball_onset``) with the smaller median absolute error against K in run (2) for
    #: the data profile x restart type; an exact tie (or no paired error) gives the documented default
    #: ``tf79_contact``. ``None`` on every other signal.
    agreement_winner: str | None = None


#: ``(data_profile, restart_type, signal, cell_state)``. An absent key is withheld and counted.
GateKey = tuple[DataProfile, str, str, str]

#: The shipped table. EMPTY in commit 2 (CL-3): every lookup is withheld, fail-closed. Commit 3 replaces it
#: with ``derive_gate_table(<committed aggregate>)``.
RESTART_GATE_TABLE: Mapping[GateKey, GateEntry] = MappingProxyType({})

#: The gate-table artifact hash. ``"unpinned"`` in commit 2; commit 3 stamps the aggregate sha256.
GATE_TABLE_VERSION: str = "unpinned"

#: A module counter: every withheld (absent-key) ``gate`` lookup increments it. Visibility only; never decisive.
withheld_lookups: int = 0


def gate(
    profile: DataProfile,
    restart_type: str,
    signal: str,
    cell_state: str,
    *,
    table: Mapping[GateKey, GateEntry] | None = None,
) -> GateEntry | None:
    """Look a cell up in ``table`` (the shipped ``RESTART_GATE_TABLE`` by default). Return its ``GateEntry`` or,
    on an absent key, ``None`` after incrementing ``withheld_lookups`` (fail-closed). Profile-level signals
    (``z_overhead``, ``footprint``) use restart_type ``"*"`` and cell_state ``"-"``."""
    global withheld_lookups
    tbl = RESTART_GATE_TABLE if table is None else table
    entry = tbl.get((profile, restart_type, signal, cell_state))
    if entry is None:
        withheld_lookups += 1
    return entry


# ---------------------------------------------------------------------------
# Statistics (spec section 3 conventions, section 16.1)
# ---------------------------------------------------------------------------


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """The Wilson score interval lower bound for ``k`` successes in ``n`` trials. ``n == 0`` returns NaN (no
    evidence is not a pass)."""
    if n <= 0:
        return float("nan")
    p = k / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return float((centre - half) / (1 + z2 / n))


def _match_arrays(hits: pd.DataFrame) -> list[np.ndarray]:
    """One boolean array of row outcomes per match, for the cluster resample."""
    return [grp["ok"].to_numpy(dtype=bool, copy=True) for _, grp in hits.groupby("match_id", sort=False)]


def cluster_bootstrap_lower(hits: pd.DataFrame, *, b: int = 2000, seed: int = 46, q: float = 0.025) -> float:
    """The ``q``-quantile of the bootstrap distribution of row accuracy, resampling MATCHES with replacement
    (``hits`` columns: ``match_id``, ``ok``). The match is the cluster, so a single bad match carrying every
    error pulls the lower bound down. Empty input returns NaN."""
    arrays = _match_arrays(hits)
    if not arrays:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(arrays))
    means = np.empty(b, dtype=float)
    for i in range(b):
        pick = rng.choice(idx, size=len(idx), replace=True)
        means[i] = np.concatenate([arrays[j] for j in pick]).mean()
    return float(np.quantile(means, q))


def cell_passes(hits: pd.DataFrame, *, bar: float = 0.90) -> bool:
    """A cell passes the section 3 gate: Wilson lower bound AND cluster-bootstrap lower bound both at or above
    ``bar``, with n >= 35 rows and m >= 5 matches (``hits`` columns: ``match_id``, ``ok``)."""
    n = len(hits)
    if n < 35:
        return False
    m = int(hits["match_id"].nunique())
    if m < 5:
        return False
    k = int(hits["ok"].to_numpy(dtype=bool).sum())
    return wilson_lower(k, n) >= bar and cluster_bootstrap_lower(hits) >= bar


def noninferior(
    child: pd.DataFrame,
    parent: pd.DataFrame,
    *,
    margin: float = 0.10,
    ci: float = 0.90,
    b: int = 2000,
    seed: int = 46,
) -> bool:
    """Non-inferiority of ``child`` to ``parent`` (both: columns ``match_id``, ``ok``). True when the child has
    n >= 20 rows AND the match-cluster bootstrap ``ci`` CI of (child accuracy - parent accuracy) has its lower
    bound above ``-margin``. A non-significant difference test is NOT used (small n passes it by construction)."""
    if len(child) < 20:
        return False
    ca = _match_arrays(child)
    pa = _match_arrays(parent)
    if not ca or not pa:
        return False
    rng = np.random.default_rng(seed)
    ci_, pi_ = np.arange(len(ca)), np.arange(len(pa))
    diffs = np.empty(b, dtype=float)
    for i in range(b):
        cpick = rng.choice(ci_, size=len(ci_), replace=True)
        ppick = rng.choice(pi_, size=len(pi_), replace=True)
        diffs[i] = np.concatenate([ca[j] for j in cpick]).mean() - np.concatenate([pa[j] for j in ppick]).mean()
    lower = float(np.quantile(diffs, (1 - ci) / 2))
    return lower > -margin


def choose_base(
    adequacy: Mapping[str, tuple[int, int]],
    admitted: frozenset[str],
    *,
    same_meaning: bool,
) -> str | None:
    """Pre-registered base precedence (spec section 5.3, B-30 / B-39), decided on SAMPLE ADEQUACY only — never
    on which base passes the gate (this function takes no outcome input).

    ``adequacy`` maps a base to its ``(n, m)``; ``admitted`` is the set of bases admitted for this signal (IDSSE
    is always admitted). A base is adequate when it is admitted and meets n >= 35 and m >= 5.

    - ``same_meaning=False`` (contact / label signals keep IDSSE first): IDSSE alone if adequate; else GS alone
      if adequate; else the combined base if the pooled (n, m) is adequate and both bases are admitted; else None.
    - ``same_meaning=True`` (``kick_time`` via K, ``penalty``, ``merge``, origin geometry — GS truth has the same
      meaning and independence): the larger-n adequate base first; else combined; else None.
    """

    def adequate(base: str) -> bool:
        if base not in admitted or base not in adequacy:
            return False
        n, m = adequacy[base]
        return n >= 35 and m >= 5

    def combined_base() -> str | None:
        if not ({"idsse", "gradientsports"} <= set(admitted) & set(adequacy)):
            return None
        n = adequacy["idsse"][0] + adequacy["gradientsports"][0]
        m = adequacy["idsse"][1] + adequacy["gradientsports"][1]
        return "combined" if n >= 35 and m >= 5 else None

    if same_meaning:
        ok = [base for base in ("idsse", "gradientsports") if adequate(base)]
        if ok:
            return max(ok, key=lambda base: adequacy[base][0])
        return combined_base()

    if adequate("idsse"):
        return "idsse"
    if adequate("gradientsports"):
        return "gradientsports"
    return combined_base()


# ---------------------------------------------------------------------------
# Ratified constants (spec section 16.1); pre-registered
# ---------------------------------------------------------------------------

#: Every section 16.1 constant. Pre-registered before any gate is scored. A test asserts a literal copy
#: (anti-drift). Frame / count constants are ``int`` (``sensitivity_values`` rounds them); times and distances
#: are ``float``.
RATIFIED: Mapping[str, float] = MappingProxyType(
    {
        # WHEN / WHICH tolerances
        "tau_event_ceiling_s": 0.40,
        "tau_event_decision_flag_s": 1.0,
        "tau_detector_ceiling_s": 0.30,
        "tau_fc_ceiling_s": 0.40,
        # native bracket (a, b)
        "bracket_clip_s": 2.0,
        "bracket_outlier_share": 0.02,
        # origin
        "origin_median_m": 2.0,
        "origin_share_m": 5.0,
        # first-contact contested threshold
        "contested_threshold_m": 0.75,
        "contested_min_m": 0.5,
        "contested_max_m": 1.0,
        # penalty geometry and windows
        "penalty_spot_r_m": 1.0,
        "penalty_rest_s": 1.0,
        "penalty_position_eval_s": 0.5,
        "penalty_d_gk_m": 2.0,
        "penalty_gk_lateral_m": 1.0,
        "penalty_gk_y_tol_m": 4.66,
        "penalty_e_enc_m": 1.0,
        "penalty_enc_disk_m": 9.15,
        "penalty_enc_count_k": 2,
        "penalty_taker_near_m": 3.0,
        "penalty_W_s": 3.0,
        "penalty_W_cross_extra_s": 0.40,
        "penalty_W_rb_s": 8.0,
        "penalty_W_rb_ceiling_s": 10.0,
        "penalty_rb_live_speed_ms": 3.0,
        "penalty_rb_live_min_m": 3.0,
        # retake radii
        "retake_rho_penalty_m": 1.0,
        "retake_rho_freekick_m": 1.0,
        "retake_rho_corner_m": 1.5,
        # merge
        "merge_r_m": 2.0,
        "merge_dt_s": 2.0,
        # minimum match count
        "m_min": 5,
        # K detection (dead-ball kicks and throw-ins)
        "k_rest_m": 0.5,
        "k_rest_s": 0.5,
        "k_release_disp_m": 0.5,
        "k_release_rest_sd_mult": 3.0,
        "k_release_speed_ms": 3.0,
        "k_release_frames": 2,
        "throwin_release_speed_ms": 4.0,
        "throwin_release_dist_m": 0.3,
        "throwin_release_window_s": 0.2,
        # SkillCorner free-kick launch
        "sc_freekick_vkick_ms": 8.0,
        "sc_freekick_ambiguous_share": 0.15,
        "sc_freekick_region_m": 5.0,
        "sc_freekick_short_cover_margin": 0.10,
        # restart regions
        "corner_region_m": 10.0,
        "throwin_region_m": 3.0,
        "freekick_region_m": 3.0,
        "corner_rest_m": 1.5,
        # R* window
        "rstar_window_s": 6.0,
        "rstar_flag_m": 10.0,
        # pooling
        "pool_child_min_n": 20,
        "pool_noninferior_margin": 0.10,
        # GI direction check
        "gi_direction_min_possessions": 5,
        "gi_direction_mismatch_share": 0.01,
        # GI / tracking pairing (C-11). pair_prox_share is DERIVED (OD-6 amendment 2026-10-23, spec sections 6.1 /
        # 16.1): the pair_neg_quantile of the time-shifted negative-control shares (+/-5, 10, 30 s) + pair_prox_margin,
        # adopted as the MAXIMUM per-profile derived bar on the R.2 staging (sc_parquet 2024/25: 0.24407, n = 417),
        # rounded UP to 3 places so the guard holds on that staging (second R.2 stop ruling, 2026-10-23).
        # pair_lag_tol_frames = 2 (symmetric): the sc_possession field leads GI by 1-2 frames (-1 in 760 of 895
        # matches, -2 in 133) while the physical proximity peaks at offset 0 -- a label definition difference.
        "pair_prox_m": 2.0,
        "pair_prox_share": 0.245,
        "pair_prox_margin": 0.10,
        "pair_neg_quantile": 0.99,
        "pair_lag_tol_frames": 2,
        "pair_lag_scan_frames": 20,
        # camera-follow segments (DL ruling 1)
        "cam_seg_s": 10.0,
        "cam_stitch_k": 5,
        "cam_hold_report_s": 2.0,
        # camera-footprint validation (DL ruling C-PLAN-19)
        "footprint_inside_share": 0.99,
        "footprint_tol_m": 1.0,
        # statistics
        "wilson_z": 1.96,
        "bootstrap_b": 2000,
        "bootstrap_q": 0.025,
        "n_min": 35,
    }
)


#: The ratified constants valued as a share / probability in [0, 1]. Their sensitivity moves +/-50 % of the
#: distance to the NEARER bound, ``min(s, 1 - s)`` (DL ruling 2026-10-23), so a share near 1 stays a share.
SHARE_CONSTANTS: frozenset[str] = frozenset(
    {
        "bracket_outlier_share",
        "sc_freekick_ambiguous_share",
        "sc_freekick_short_cover_margin",
        "pool_noninferior_margin",
        "gi_direction_mismatch_share",
        "pair_prox_share",
        "pair_prox_margin",
        "pair_neg_quantile",
        "footprint_inside_share",
        "bootstrap_q",
    }
)


def sensitivity_values(key: str) -> tuple[float, float] | tuple[int, int]:
    """The +/-50 % sensitivity pair for a ratified constant (run (1), run (2), run (9), run (12)). An integer
    constant (a frame or a count) rounds each endpoint to the nearest integer; a float keeps the raw value. A
    :data:`SHARE_CONSTANTS` share ``s`` moves +/-50 % of ``min(s, 1 - s)`` (0.90 -> 0.85 / 0.95; DL ruling
    2026-10-23), never past 0 or 1."""
    v = RATIFIED[key]
    if key in SHARE_CONSTANTS:
        half = 0.5 * min(v, 1.0 - v)
        return (round(v - half, 12), round(v + half, 12))  # 12 places: no float noise in a run (2) label
    lo, hi = v * 0.5, v * 1.5
    if isinstance(v, int):
        return (round(lo), round(hi))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Tier-2 fidelity contract (spec section 16.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FidelityTolerance:
    """One section 16.3 statistic: how it is measured and the fixed pass rule (the margin rule and the LOMO
    held-out scoring live in the run (9) driver, not here)."""

    how: str
    rule: str


#: The ball statistics (do not read a player contact).
_BALL_STATS: frozenset[str] = frozenset(
    {
        "frame_rate",
        "z_source_mix",
        "ball_detection_rate",
        "gap_distribution",
        "kick_window_miss_rate",
        "noise_sd",
        "fill_share",
    }
)

#: The contact statistics (read a player near the ball). The B-33 source-path rule adds them to every signal
#: whose row actually reads a contact (``taker``, ``first_contact``, ``kick_time`` via ``tf79_contact``,
#: ``origin`` read at the launch contact, ``merge``, and the ``penalty`` split per C-51).
_CONTACT_STATS: frozenset[str] = frozenset(
    {
        "player_visibility_near_ball",
        "nearest_player_visible_at_contact",
        "identity_swap_rate",
        "off_camera_position_error",
    }
)

FIDELITY_TOLERANCES: Mapping[str, FidelityTolerance] = MappingProxyType(
    {
        "frame_rate": FidelityTolerance("resample to the target rate", "exact"),
        "z_source_mix": FidelityTolerance("replace z with the estimated z", "exact, by construction"),
        "ball_detection_rate": FidelityTolerance(
            "share of frames with a detected ball, per region", "abs diff <= 0.03"
        ),
        "gap_distribution": FidelityTolerance("gap lengths in seconds, per region", "KS <= 0.10 AND p90 within +/-20%"),
        "kick_window_miss_rate": FidelityTolerance(
            "share of restart kicks with no detected ball within +/-0.3 s of K", "abs diff <= 0.03"
        ),
        "noise_sd": FidelityTolerance("position SD over stationary windows, per region", "ratio within [0.85, 1.15]"),
        "fill_share": FidelityTolerance(
            "share of ball frames filled or extrapolated", "abs diff <= 0.03 overall, <= 0.05 per region"
        ),
        "player_visibility_near_ball": FidelityTolerance(
            "share of player-frames detected within 10 m of the ball", "abs diff <= 0.03 (<= 0.05 all players)"
        ),
        "nearest_player_visible_at_contact": FidelityTolerance(
            "share of contact frames where the nearest player is detected", "abs diff <= 0.03"
        ),
        "identity_swap_rate": FidelityTolerance(
            "same-id jumps implying > 12 m/s per player-hour", "95% CI of rate ratio within [0.67, 1.5]"
        ),
        "off_camera_position_error": FidelityTolerance(
            "re-detection jump per off-camera spell, binned by time off camera (R2)",
            "per bin: KS <= 0.10 AND p90 within +/-20%",
        ),
    }
)

#: A statistic gates only the signals that depend on it (spec section 16.3). The source-path rule (B-33, C-51)
#: puts the contact statistics on ``origin`` and ``penalty`` as well as the obvious contact signals.
FIDELITY_DEPENDENCIES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "kick_time": _BALL_STATS | _CONTACT_STATS,
        "taker": _BALL_STATS | _CONTACT_STATS,
        "first_contact": _BALL_STATS | _CONTACT_STATS,
        "subtype": _BALL_STATS,
        "merge": _BALL_STATS | _CONTACT_STATS,
        "origin": _BALL_STATS | _CONTACT_STATS,
        "penalty": _BALL_STATS | _CONTACT_STATS | frozenset({"player_visibility_all"}),
    }
)


# ---------------------------------------------------------------------------
# Pooling ladder (spec section 15.3)
# ---------------------------------------------------------------------------

#: Per signal, the parent cell_state a child cell_state may pool to (fine -> coarse), and the cell_states that
#: never pool. ``kick_time`` conflict states never pool with non-conflict; ``retake`` and throw-in
#: ``kick_time`` never pool; ``first_contact`` contested never pools with uncontested; ``subtype`` / ``origin``
#: / ``merge`` never pool.
_NEVER_POOLED: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "subtype": frozenset({"*"}),
        "origin": frozenset({"*"}),
        "merge": frozenset({"*"}),
    }
)


def _may_pool(signal: str, restart_type: str, cell_state: str) -> bool:
    """True when a child cell of ``signal`` / ``restart_type`` / ``cell_state`` is allowed to take a parent
    result at all (the ladder of spec section 15.3). The three unconditional conditions (parent passes, child
    n >= 20, non-inferiority) are checked in ``derive_gate_table``; ``first_contact`` additionally requires the
    child and parent to share the contested-ness token (contested never pools with uncontested)."""
    never = _NEVER_POOLED.get(signal)
    if never and ("*" in never or cell_state in never):
        return False
    if signal == "kick_time" and (restart_type == "throw_in" or "conflict" in cell_state or cell_state == "retake"):
        return False
    return True


# ---------------------------------------------------------------------------
# Gate-table derivation (spec section 15.3, consumed in commit 3)
# ---------------------------------------------------------------------------

#: The columns ``derive_gate_table`` reads from the committed pre-flight aggregate. One row per
#: (profile, restart_type, signal, cell_state, base, stratum, match_id), with per-match counts and the
#: cell-level metrics. The run (…) drivers (Task 2.13) write exactly these columns (ADR-056).
AGGREGATE_COLUMNS: tuple[str, ...] = (
    "provider",
    "tracking_release",
    "gi_release",
    "competition_season",
    "restart_type",
    "signal",
    "cell_state",
    "base",
    "admitted",
    "stratum",
    "parent_cell_state",
    "match_id",
    "k",
    "n",
    "tier",
    "lomo_metric",
    "calibration_metric",
    "calibration_ceiling",
    "footprint_inside_share",
    # subtype only (spec section 11.2): the match-cluster bootstrap lower bound of (rule precision - the
    # per-cell majority-class baseline precision); the cell passes only when it is > 0. NaN elsewhere.
    "baseline_margin_lb",
    # kick_time only (spec section 6.5, amended 2026-10-22): the match's median absolute error against K of each
    # candidate, over the rows where both were observed (run (2)); the agreement winner derives from them. NaN
    # elsewhere.
    "agree_err_contact_s",
    "agree_err_onset_s",
)

_PROFILE_COLS = ("provider", "tracking_release", "gi_release", "competition_season")


def _profile_of(row: pd.Series) -> DataProfile:
    gi = row["gi_release"]
    return DataProfile(
        provider=row["provider"],
        tracking_release=row["tracking_release"],
        gi_release=None if pd.isna(gi) else gi,
        competition_season=row["competition_season"],
    )


def _row_hits(block: pd.DataFrame) -> pd.DataFrame:
    """Expand per-match ``(k, n)`` counts into a per-row ``(match_id, ok)`` frame for the statistics."""
    match_ids: list[object] = []
    ok: list[bool] = []
    for _, r in block.iterrows():
        n, k = int(r["n"]), int(r["k"])
        match_ids.extend([r["match_id"]] * n)
        ok.extend([True] * k + [False] * (n - k))
    return pd.DataFrame({"match_id": match_ids, "ok": ok})


def _calibration_ok(block: pd.DataFrame) -> bool:
    """C-03: the all-match calibration constant must be within its ceiling. A ceiling of NaN means the signal
    has no calibrated constant (an accuracy signal), so the check passes."""
    ceiling = block["calibration_ceiling"].iloc[0]
    if pd.isna(ceiling):
        return True
    return float(block["calibration_metric"].iloc[0]) <= float(ceiling)


def _as_gate_base(value) -> GateBase:
    """A base read from the aggregate, checked against :data:`GateBase` (an unknown base is an aggregate defect,
    never a silently shipped cell)."""
    base = str(value)
    if base not in get_args(GateBase):
        raise ValueError(f"unknown gate base {base!r}; expected one of {get_args(GateBase)}")
    return cast(GateBase, base)


def _entry(block: pd.DataFrame, hits: pd.DataFrame, base: str, resolution: str) -> GateEntry:
    k = int(hits["ok"].to_numpy(dtype=bool).sum())
    n = len(hits)
    return GateEntry(
        tier=block["tier"].iloc[0],
        base=_as_gate_base(base),
        metric=float(block["lomo_metric"].iloc[0]),
        wilson_lb=wilson_lower(k, n),
        bootstrap_lb=cluster_bootstrap_lower(hits),
        n=n,
        m=int(hits["match_id"].nunique()),
        cell_resolution=resolution,
    )


def _derive_penalty_cell(block: pd.DataFrame) -> tuple[GateEntry | None, pd.DataFrame | None]:
    """A penalty cell (one ``cell_state``). Strata ``in_play`` / ``shoot_out`` pool into the cell only when each
    stratum passes non-inferiority against the pooled population with n >= 20; otherwise the cell is withheld."""
    if not _calibration_ok(block):
        return None, None
    strata = [s for s in block["stratum"].dropna().unique() if s]
    pooled = _row_hits(block)
    if len(strata) > 1:
        for s in strata:
            s_hits = _row_hits(block[block["stratum"] == s])
            if len(s_hits) < 20 or not noninferior(s_hits, pooled):
                return None, None
        resolution = "pooled_strata"
    else:
        resolution = "cell"
    if not cell_passes(pooled):
        return None, None
    return _entry(block, pooled, block["base"].iloc[0], resolution), pooled


def _derive_standard_cell(block: pd.DataFrame) -> tuple[GateEntry | None, pd.DataFrame | None]:
    """A non-penalty, non-profile cell. Choose the base on adequacy (spec section 5.3), then score that base's
    hits. Returns the entry (or None) and the chosen base's hits (for the pooling ladder)."""
    if not _calibration_ok(block):
        return None, None
    if block["signal"].iloc[0] == "subtype":
        # Spec section 11.2: per-direction precision must ALSO beat the majority-class baseline (CI > 0).
        lb = block["baseline_margin_lb"].iloc[0] if "baseline_margin_lb" in block.columns else float("nan")
        if pd.isna(lb) or float(lb) <= 0.0:
            return None, None
    same_meaning = block["signal"].iloc[0] in ("kick_time", "penalty", "merge", "origin")
    admitted = frozenset(block.loc[block["admitted"].astype(bool), "base"].unique()) | {"idsse"}
    adequacy: dict[str, tuple[int, int]] = {}
    per_base: dict[str, pd.DataFrame] = {}
    for base, grp in block.groupby("base", sort=False):
        hits = _row_hits(grp)
        per_base[str(base)] = hits
        adequacy[str(base)] = (len(hits), int(hits["match_id"].nunique()))
    chosen = choose_base(adequacy, admitted, same_meaning=same_meaning)
    if chosen is None:
        return None, None
    hits = pd.concat(list(per_base.values()), ignore_index=True) if chosen == "combined" else per_base[chosen]
    if not cell_passes(hits):
        return None, None
    return _entry(block, hits, chosen, "cell"), hits


#: The z_overhead aggregate rows carry two cell states per match: first-contact agreement with tier-1 truth
#: with the F3 overhead gate applied (``gate_on``) and without it (``gate_off``), on the same contacts.
Z_OVERHEAD_CELL_STATES: tuple[str, str] = ("gate_on", "gate_off")


def paired_bootstrap_lower(per_match: pd.DataFrame, *, b: int = 2000, seed: int = 46, q: float = 0.025) -> float:
    """The match-cluster bootstrap ``q`` lower bound of the POOLED paired difference
    ``(sum k_on - sum k_off) / sum n`` (spec section 14.1). ``per_match``: one row per match with ``k_on``,
    ``k_off``, ``n``. Resamples MATCHES with replacement (seeded). NaN when there is no contact."""
    if per_match.empty or int(per_match["n"].sum()) == 0:
        return float("nan")
    k_on = per_match["k_on"].to_numpy(dtype=np.float64)
    k_off = per_match["k_off"].to_numpy(dtype=np.float64)
    n = per_match["n"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_match), size=(b, len(per_match)))
    num = (k_on[idx] - k_off[idx]).sum(axis=1)
    den = n[idx].sum(axis=1)
    diffs = np.divide(num, den, out=np.full(b, np.nan), where=den > 0)
    return float(np.nanquantile(diffs, q))


def _derive_z_overhead(block: pd.DataFrame) -> GateEntry | None:
    """The z_overhead entry for ONE profile (spec section 14.1). Passes only when (i) the provider DECLARES
    ``z_source == "measured"`` (ADR-125 ``Z_PROVENANCE``; necessary, not sufficient) and (ii) the paired
    match-cluster bootstrap lower bound of (agreement gate-on - agreement gate-off) is >= 0, i.e. applying
    the overhead gate does not reduce first-contact agreement with tier-1 truth. Otherwise withheld."""
    from .schema import Z_PROVENANCE

    provider = block["provider"].iloc[0]
    if Z_PROVENANCE.get(str(provider)) != "measured":
        return None
    on = block[block["cell_state"] == "gate_on"].groupby("match_id", sort=True)[["k", "n"]].sum()
    off = block[block["cell_state"] == "gate_off"].groupby("match_id", sort=True)[["k", "n"]].sum()
    paired = on.join(off, lsuffix="_on", rsuffix="_off", how="inner")
    if paired.empty or (paired["n_on"] != paired["n_off"]).any():
        return None  # unpaired rows: the comparison is not on the same contacts -> fail closed
    per_match = pd.DataFrame({"k_on": paired["k_on"], "k_off": paired["k_off"], "n": paired["n_on"]})
    lower = paired_bootstrap_lower(per_match)
    if not np.isfinite(lower) or lower < 0.0:
        return None
    n = int(per_match["n"].sum())
    return GateEntry(
        tier=block["tier"].iloc[0],
        base=block["base"].iloc[0],
        metric=float((per_match["k_on"].sum() - per_match["k_off"].sum()) / n),
        wilson_lb=float("nan"),
        bootstrap_lb=lower,
        n=n,
        m=len(per_match),
        cell_resolution="cell",
    )


def derive_gate_table(aggregate: pd.DataFrame) -> dict[GateKey, GateEntry]:
    """Derive the validation gate table from the committed pre-flight aggregate (``AGGREGATE_COLUMNS``). A cell
    ships a ``GateEntry`` only when it passes the section 3 gate under its chosen base (or pools per section
    15.3); every other cell is absent (withheld). Pure: no I/O, no global mutation."""
    out: dict[GateKey, GateEntry] = {}
    cell_keys = [*_PROFILE_COLS, "restart_type", "signal", "cell_state"]

    # Profile-level footprint: emit only when the inside share clears the ratified bar.
    foot = aggregate[aggregate["signal"] == "footprint"]
    for _, grp in foot.groupby(list(_PROFILE_COLS), sort=False, dropna=False):
        row = grp.iloc[0]
        if float(row["footprint_inside_share"]) >= RATIFIED["footprint_inside_share"]:
            out[(_profile_of(row), "*", "footprint", "-")] = GateEntry(
                tier=row["tier"],
                base=row["base"],
                metric=float(row["footprint_inside_share"]),
                wilson_lb=float("nan"),
                bootstrap_lb=float("nan"),
                n=int(row.get("n", 0) or 0),
                m=int(grp["match_id"].nunique()),
                cell_resolution="cell",
            )

    # z_overhead: a profile-level PAIRED gate (spec section 14.1), not an accuracy cell.
    zo = aggregate[aggregate["signal"] == "z_overhead"]
    for _, grp in zo.groupby(list(_PROFILE_COLS), sort=False, dropna=False):
        entry = _derive_z_overhead(grp)
        if entry is not None:
            out[(_profile_of(grp.iloc[0]), "*", "z_overhead", "-")] = entry

    body = aggregate[~aggregate["signal"].isin(("footprint", "z_overhead"))]
    passed: dict[GateKey, pd.DataFrame] = {}  # cell -> hits, for the pooling ladder
    pending: list[tuple[GateKey, pd.DataFrame]] = []
    for keyvals, block in body.groupby(cell_keys, sort=False, dropna=False):
        profile = _profile_of(block.iloc[0])
        restart_type, signal, cell_state = keyvals[4], keyvals[5], keyvals[6]
        key: GateKey = (profile, str(restart_type), str(signal), str(cell_state))
        if signal == "penalty":
            entry, hits = _derive_penalty_cell(block)
        else:
            entry, hits = _derive_standard_cell(block)
        if entry is not None and hits is not None:
            out[key] = entry
            passed[key] = hits
        else:
            pending.append((key, block))

    # Pooling ladder: a withheld child may take a passing parent's result (same signal, same profile) when the
    # child has n >= 20 and is non-inferior to the parent.
    for key, block in pending:
        profile, restart_type, signal, cell_state = key
        if not _may_pool(signal, restart_type, cell_state):
            continue
        child_hits = _row_hits(block[block["base"] == block["base"].iloc[0]]) if len(block) else None
        if child_hits is None or len(child_hits) < 20:
            continue
        parent_state = block["parent_cell_state"].iloc[0]
        if pd.isna(parent_state) or not parent_state:
            continue
        if signal == "first_contact" and cell_state.split("_")[0] != str(parent_state).split("_")[0]:
            continue  # contested never pools with uncontested (spec section 15.3)
        parent_key: GateKey = (profile, restart_type, signal, parent_state)
        parent_hits = passed.get(parent_key)
        if parent_hits is None or not noninferior(child_hits, parent_hits):
            continue
        parent_entry = out[parent_key]
        out[key] = GateEntry(
            tier=parent_entry.tier,
            base=parent_entry.base,
            metric=parent_entry.metric,
            wilson_lb=parent_entry.wilson_lb,
            bootstrap_lb=parent_entry.bootstrap_lb,
            n=parent_entry.n,
            m=parent_entry.m,
            cell_resolution=f"pooled_{parent_state}",
        )
    return _with_agreement_winners(out, aggregate)


#: The documented default when the run (2) errors cannot separate the candidates (spec section 16.1).
AGREEMENT_WINNER_DEFAULT = "tf79_contact"


def agreement_winner(aggregate: pd.DataFrame) -> str:
    """The section 6.5 agreement winner from ``kick_time`` aggregate rows (one data profile x restart type): the
    candidate whose median of the per-match median absolute errors is smaller; an exact tie or no paired error
    gives :data:`AGREEMENT_WINNER_DEFAULT` (a documented default, not a result).

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.tracking._restart_validation import agreement_winner
    >>> agreement_winner(pd.DataFrame({"agree_err_contact_s": [0.12, 0.10], "agree_err_onset_s": [0.06, 0.08]}))
    'ball_onset'
    """
    if aggregate.empty or "agree_err_contact_s" not in aggregate.columns:
        return AGREEMENT_WINNER_DEFAULT
    c = pd.to_numeric(aggregate["agree_err_contact_s"], errors="coerce").dropna()
    o = pd.to_numeric(aggregate["agree_err_onset_s"], errors="coerce").dropna()
    if c.empty or o.empty:
        return AGREEMENT_WINNER_DEFAULT
    return "ball_onset" if float(o.median()) < float(c.median()) else AGREEMENT_WINNER_DEFAULT


def _with_agreement_winners(table: dict[GateKey, GateEntry], aggregate: pd.DataFrame) -> dict[GateKey, GateEntry]:
    """Stamp each ``kick_time`` entry with its data profile x restart type agreement winner (from the run (2)
    paired errors, :func:`agreement_winner`)."""
    from dataclasses import replace

    if aggregate.empty:
        return table
    kt = aggregate[aggregate["signal"] == "kick_time"]
    winners: dict[tuple[DataProfile, str], str] = {}
    for _keyvals, grp in kt.groupby([*_PROFILE_COLS, "restart_type"], sort=False, dropna=False):
        winners[(_profile_of(grp.iloc[0]), str(grp["restart_type"].iloc[0]))] = agreement_winner(grp)
    out = dict(table)
    for key, entry in table.items():
        profile, restart_type, signal, _cs = key
        if signal == "kick_time":
            out[key] = replace(entry, agreement_winner=winners.get((profile, restart_type), AGREEMENT_WINNER_DEFAULT))
    return out


# ---------------------------------------------------------------------------
# Additive action columns (spec section 17.1)
# ---------------------------------------------------------------------------

#: The restart-refinement columns added to actions (additive enrichment, never base ``SPADL_COLUMNS``).
RESTART_REFINEMENT_COLUMNS: dict[str, str] = {
    # resolve_kicks
    "kick_bracket_lo": "float64",
    "kick_bracket_hi": "float64",
    "t_contact": "float64",
    "t_onset": "float64",
    "kick_time_delta": "float64",
    "t_kick": "float64",
    "kick_time_source": "string",
    "kick_time_resolution": "string",
    "kick_time_evidence": "string",
    "kick_time_gate_version": "string",
    "kick_time_cell_resolution": "string",
    "kick_retake": "boolean",
    "kick_rest_x": "float64",
    "kick_rest_y": "float64",
    "gi_pairing": "string",
    "gi_pairing_failed_test": "string",
    "gi_next_possession_time": "float64",
    "taker_source": "string",
    "taker_evidence": "string",
    "taker_gate_version": "string",
    "taker_cell_resolution": "string",
    # refine
    "refine_created": "boolean",
    "type_id_events": "Int64",
    "type_source_events": "string",
    "start_x_events": "float64",
    "start_y_events": "float64",
    "type_tracking_disagrees": "boolean",
    "subtype_evidence": "string",
    "subtype_gate_version": "string",
    "subtype_cell_resolution": "string",
    "merged_from_derived_action_id": "Int64",
    "merged_from_shot_action_id": "Int64",
    "merged_derived_type_id": "Int64",
    "merged_derived_result_id": "Int64",
    "merged_derived_time_seconds": "float64",
    "merged_derived_end_x": "float64",
    "merged_derived_end_y": "float64",
    "merged_derived_gi_marker_possession_frame_end": "Int64",
    "merged_derived_gi_next_possession_frame_start": "Int64",
    "merged_derived_gi_attacking_side": "string",
    "merged_derived_result_source": "string",
    "merged_derived_restart_evidence": "string",
    "merged_derived_type_source": "string",
    "merged_derived_original_event_id": "string",
    "merged_derived_repair_kind": "string",
    "merge_evidence": "string",
    "merge_gate_version": "string",
    "merge_cell_resolution": "string",
    "penalty_evidence": "string",
    "penalty_gate_version": "string",
    "penalty_cell_resolution": "string",
    "penalty_crosscheck": "string",
    "penalty_outcome": "string",
    # add_restart_coordinates
    "origin_evidence": "string",
    "origin_gate_version": "string",
    "origin_cell_resolution": "string",
}

#: The columns ``resolve_kicks`` writes on a restart row (spec sections 6.6, 7): the kick-time set, the rest location,
#: the GI pairing and the taker set. A B8-merged row keeps the derived row's values of these (spec section 11.5).
RESOLVE_KICKS_COLUMNS: tuple[str, ...] = (
    *list(RESTART_REFINEMENT_COLUMNS)[: list(RESTART_REFINEMENT_COLUMNS).index("taker_cell_resolution") + 1],
    "taker_player_id",
)
# ``taker_player_id`` is a provider-id-dtype column (ADR-019), added by resolve_kicks outside this dtype map.

__all__ = [
    "AGGREGATE_COLUMNS",
    "FIDELITY_DEPENDENCIES",
    "FIDELITY_TOLERANCES",
    "GATE_TABLE_VERSION",
    "PENALTY_CELL_STATES",
    "RATIFIED",
    "RESTART_GATE_TABLE",
    "RESTART_REFINEMENT_COLUMNS",
    "SIGNALS",
    "Z_OVERHEAD_CELL_STATES",
    "DataProfile",
    "FidelityTolerance",
    "GateEntry",
    "GateKey",
    "cell_passes",
    "choose_base",
    "cluster_bootstrap_lower",
    "derive_gate_table",
    "gate",
    "noninferior",
    "paired_bootstrap_lower",
    "sensitivity_values",
    "wilson_lower",
]


def _profile_fields() -> tuple[str, ...]:
    """The ``DataProfile`` field names (used by the no-pairing-field anti-rot test)."""
    return tuple(f.name for f in fields(DataProfile))
