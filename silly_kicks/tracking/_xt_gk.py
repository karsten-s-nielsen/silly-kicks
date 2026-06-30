"""xT-GK — Expected Threat for Goalkeepers (Eyestone).

A pure parametric compute feature (NOT a trained model): re-values GK distribution
actions (goal-kicks, keeper passes/throws) by composing the xT grid with GK-specific
terms under a frozen, team-tunable parameter set.

Attribution: Jeffrey Eyestone, *Expected Threat for Goalkeepers (xT-GK)*, winner of
Pitch to the Pros 1 (May 2025). Contributed publicly with attribution by Jeffrey's
explicit permission (email 2026-06-06; formula confirmation 2026-06-08). The functional
forms here are the silly-kicks formulation of Eyestone's xT-GK (the deck gives components
+ parameter ranges, not closed-form equations). See NOTICE for full bibliographic
citations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import gaussian_filter

from silly_kicks.spadl import config as spadlconfig

from ._id_compat import canonical_id_series, ids_equal

if TYPE_CHECKING:
    from silly_kicks.xthreat import ExpectedThreat

    from ._gk_completion import GkCompletionModel

_PressureMethod = Literal["andrienko_oval", "link_zones", "bekkers_pi"]

_GOALKICK = spadlconfig.actiontype_id["goalkick"]  # 22
_PASS = spadlconfig.actiontype_id["pass"]  # 0
_THROW_IN = spadlconfig.actiontype_id["throw_in"]  # 2

_OUTPUT_COLS = ["xt_gk_base", "xt_gk_pev", "xt_gk_rav", "xt_gk_dzv", "xt_gk_pressure", "xt_gk"]
# Resolved-coordinate AUDIT columns (handoff 2026-06-29): the exact origin/destination the grid
# lookups used (for goal-kicks ~67% are imputed, NOT the native start_x/end_x). Emitted per in-scope
# row for external verifiability; deliberately NOT in _OUTPUT_COLS so xt_gk_xfns does not surface them
# as per-slot VAEP features (they are provenance, not a metric).
_COORD_COLS = ["xt_gk_origin_x", "xt_gk_origin_y", "xt_gk_dest_x", "xt_gk_dest_y"]
_PROVENANCE_COLS = [
    "xt_gk_origin_source",
    "xt_gk_dest_source",
    "xt_gk_origin_confidence",
    "xt_gk_completion_variant",
    "xt_gk_completion_source",
    "xt_gk_native_goalkick_out_of_region",  # S4 data-quality flag (CR 2026-06-30); provenance, not a metric
]


@dataclass(frozen=True)
class XtGkReport:
    """Aggregate provenance QA for an xT-GK output frame: counts per origin/destination
    resolution tier + a scored-row tally. Mirrors ConversionReport / LinkReport (ADR-004);
    a convenience over a downstream ``GROUP BY xt_gk_origin_source`` -- not load-bearing.

    By construction ``origin_source_counts`` / ``dest_source_counts`` equal the output
    columns' ``value_counts`` (the §6 acceptance contract).

    ``spans_multiple_variants`` (review m-c) is True when the scored rows span >1
    ``xt_gk_completion_variant`` -- the machine-observable signal that a pooled aggregation
    mixes completion-model variants (the no-pool-without-comparability contract, H1/D-S9).
    ``completion_variant_counts`` mirrors the column's ``value_counts``."""

    n_rows: int
    n_scored: int
    origin_source_counts: dict[str, int]
    dest_source_counts: dict[str, int]
    completion_variant_counts: dict[str, int]
    completion_source_counts: dict[str, int]  # model vs base_rate (per-type serve gate); mirrors value_counts
    spans_multiple_variants: bool
    n_native_goalkick_out_of_region: int = 0  # S4 (CR 2026-06-30): countable data-quality signal

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> XtGkReport:
        """Build the report from a ``compute_xt_gk`` / ``add_xt_gk`` output frame (it carries
        the ``xt_gk_origin_source`` / ``xt_gk_dest_source`` / ``xt_gk_completion_variant`` /
        ``xt_gk`` columns)."""
        osc = df["xt_gk_origin_source"].value_counts(dropna=True)
        dsc = df["xt_gk_dest_source"].value_counts(dropna=True)
        cvc = (
            df["xt_gk_completion_variant"].value_counts(dropna=True)
            if "xt_gk_completion_variant" in df.columns
            else pd.Series(dtype=int)
        )
        csc = (
            df["xt_gk_completion_source"].value_counts(dropna=True)
            if "xt_gk_completion_source" in df.columns
            else pd.Series(dtype=int)
        )
        return cls(
            n_rows=len(df),
            n_scored=int(df["xt_gk"].notna().sum()),
            origin_source_counts={str(k): int(v) for k, v in osc.items()},
            dest_source_counts={str(k): int(v) for k, v in dsc.items()},
            completion_variant_counts={str(k): int(v) for k, v in cvc.items()},
            completion_source_counts={str(k): int(v) for k, v in csc.items()},
            spans_multiple_variants=bool(len(cvc) > 1),
            n_native_goalkick_out_of_region=int(
                df["xt_gk_native_goalkick_out_of_region"].sum()
                if "xt_gk_native_goalkick_out_of_region" in df.columns
                else 0
            ),
        )


# Deck parameter ranges: gamma 0.1-0.4, delta 0.3-0.8, eta 0.8-0.9.
# Point values are PROVISIONAL (in-range), per Jeffrey's "ship presets, whatever is
# easy" delegation + his 2026-06-08 "OK to go with provisional values"; exact table is
# open. dzv_alpha / dzv_beta are CANONICAL (Eyestone 2026-06-27); dzv_d_max /
# defensive_third_boundary (= D_threshold) / pressure_scale are normative intent-set
# constants (never calibrated).


@dataclass(frozen=True)
class XtGkParams:
    # --- interpretive / intent-set (NOT VAEP-calibrated) ---
    gamma: float = 0.25  # PEV pressure-escape sensitivity   (range 0.1-0.4)
    delta: float = 0.55  # RAV risk-aversion                 (range 0.3-0.8)
    phi: float = 1.0  # DZV overall weight (preset-modulated; canonical SHAPE is dzv_alpha/dzv_beta)
    eta: float = 0.85  # temporal-sequence discount        (range 0.8-0.9)
    # --- DZV canonical revaluation phi(z,d) = alpha*(1 - d/D_max)^(-beta) for d < D_threshold else 1 ---
    dzv_alpha: float = 2.1  # CANONICAL (Eyestone 2026-06-27)
    dzv_beta: float = 0.8  # CANONICAL (Eyestone 2026-06-27)
    dzv_d_max: float = 105.0  # provisional; pitch length
    defensive_third_boundary: float = 35.0  # NORMATIVE: D_threshold = own defensive third end (105/3)
    pressure_scale: float = 50.0  # rho squash scale; intent-set
    # --- structural smoothing (hand-set; one-off sensitivity scan) ---
    convolution_sigma: float = 0.8
    # --- method selector ---
    pressure_method: _PressureMethod = "andrienko_oval"

    @classmethod
    def for_philosophy(cls, name: str) -> XtGkParams:
        """Return the deck's five team-philosophy presets (provisional point values
        within the deck ranges; exact values are an open question)."""
        base = cls()
        presets: dict[str, dict[str, float]] = {
            "possession": dict(gamma=0.30, delta=0.45, phi=1.2, eta=0.88),
            "counter": dict(gamma=0.15, delta=0.70, phi=0.8, eta=0.82),
            "direct": dict(gamma=0.20, delta=0.60, phi=0.9, eta=0.80),
            "high_press": dict(gamma=0.35, delta=0.50, phi=1.1, eta=0.86),
            "low_block": dict(gamma=0.12, delta=0.75, phi=1.3, eta=0.90),
        }
        if name not in presets:
            raise ValueError(f"unknown xT-GK philosophy preset: {name!r}")
        return replace(base, **presets[name])


# --------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------
def _convolve_grid(xt_grid: npt.NDArray[np.float64], sigma: float) -> npt.NDArray[np.float64]:
    """Separable Gaussian smoothing of the xT grid (the public-app spatial-convolution
    term). sigma <= 0 returns the raw grid unchanged (xT* == xT)."""
    if sigma <= 0:
        return xt_grid
    return gaussian_filter(xt_grid, sigma=sigma, mode="nearest")


def _phi_of_d(
    d: npt.NDArray[np.float64], alpha: float, beta: float, d_max: float, d_threshold: float
) -> npt.NDArray[np.float64]:
    """Eyestone's defensive-zone revaluation factor phi(z,d) = alpha*(1 - d/D_max)^(-beta)
    for d < D_threshold, else 1.0. d = distance from own goal = LTR origin x (team attacks +x).
    phi >= 1, rising with depth toward the threshold, then cliffing to 1 outside the defensive
    third. See NOTICE for full bibliographic citations (Eyestone xT-GK)."""
    d = np.asarray(d, float)
    active = d < d_threshold
    # (1 - d/D_max) is strictly positive for d < D_threshold < D_max -> no negative base.
    raised = alpha * np.power(1.0 - np.where(active, d, 0.0) / d_max, -beta)
    return np.where(active, raised, 1.0)


def _phi_grid(
    shape: tuple[int, int], alpha: float, beta: float, d_max: float, d_threshold: float
) -> npt.NDArray[np.float64]:
    """Per-cell phi(z,d) grid matching an xT grid's (n_rows, n_cols). phi depends on x only
    (d = column-centre x), so every row is identical. Column c -> x = field_length*(c+0.5)/n_cols
    (cell centre, matching xthreat's cell convention)."""
    n_rows, n_cols = shape
    xc = spadlconfig.field_length * (np.arange(n_cols) + 0.5) / n_cols
    row = _phi_of_d(xc, alpha, beta, d_max, d_threshold)
    return np.tile(row, (n_rows, 1))


def _grid_value(
    grid: npt.NDArray[np.float64], x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Vectorized grid lookup at SPADL coords (LTR-normalized; team attacks +x).

    DRY (review H1): reuse xthreat's frozen cell-indexer (ADR-021) instead of
    reimplementing the (x,y)->cell math -- this is xthreat's port, not xT-GK's to own.
    Apply the same row inversion ExpectedThreat.rate uses: row = (w-1) - yj, col = xi
    (row 0 is the top of the pitch). Pinned to .rate by a cross-check test."""
    from silly_kicks.xthreat._grid import _get_cell_indexes

    n_rows, n_cols = grid.shape
    xa = np.asarray(x, float)
    ya = np.asarray(y, float)
    # NaN-safe: a grid lookup at an unknown (NaN) coordinate is NaN. _get_cell_indexes does
    # .astype("int64"), which RAISES on NaN -- real provider data (e.g. GS goal-kicks with a
    # missing destination) carries NaN coords, so guard them out instead of crashing.
    result = np.full(xa.shape, np.nan, dtype=float)
    valid = np.isfinite(xa) & np.isfinite(ya)
    if valid.any():
        xi, yj = _get_cell_indexes(pd.Series(xa[valid]), pd.Series(ya[valid]), n_cols, n_rows)
        result[valid] = grid[(n_rows - 1) - yj.to_numpy(), xi.to_numpy()]
    return result


def _counter_value(
    grid: npt.NDArray[np.float64], x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Opponent threat from the (intended) loss zone: full 180-degree point-reflection
    xT*(L - x, W - y) (both axes)."""
    return _grid_value(
        grid,
        spadlconfig.field_length - np.asarray(x, dtype=float),
        spadlconfig.field_width - np.asarray(y, dtype=float),
    )


def _normalize_pressure(raw: npt.NDArray[np.float64], scale: float) -> npt.NDArray[np.float64]:
    """Saturating exponential-CDF squash rho = 1 - exp(-max(0, raw)/s) -> [0, 1).
    max(0, .) guards any method returning a negative raw (none currently do)."""
    clamped = np.maximum(0.0, np.asarray(raw, dtype=float))
    return 1.0 - np.exp(-clamped / scale)


def _possession_depth(actions: pd.DataFrame) -> npt.NDArray[np.intp]:
    """k(a): action's positional depth within its possession run. A run breaks when
    team_id changes or period_id changes. GK distributions are possession-starters so
    k ~ 0 (the temporal term is near-inert here by construction)."""
    team = actions["team_id"]
    period = actions["period_id"]
    # same-column self-shift; ids_equal is dtype/NaN-safe (first row -> not-equal -> new run)
    team_changed = ~ids_equal(team, team.shift())
    period_changed = period.ne(period.shift())
    run_id = (np.asarray(team_changed) | np.asarray(period_changed)).cumsum()
    return actions.groupby(run_id, sort=False).cumcount().to_numpy().astype(np.intp)


def _progress(xt_star_dest, xt_star_origin):
    """Forward move value xT*(z') - xT*(z); feeds PEV only (Option B keeps the destination
    out of the composite's base term -- RAV owns z')."""
    return np.asarray(xt_star_dest, float) - np.asarray(xt_star_origin, float)


# --------------------------------------------------------------------------------------
# Components (raw, before parameter weighting). Option B (Jeffrey 2026-06-08): the
# destination value is owned solely by RAV; the composite base is origin-only.
# --------------------------------------------------------------------------------------
def _base(xt_star_origin):
    """Composite base term = - xT*(z): the threat given up by leaving the origin. The
    destination value is owned solely by RAV (no double-count)."""
    return -np.asarray(xt_star_origin, float)


def _pev(rho, progress):
    return np.asarray(rho, float) * np.maximum(0.0, np.asarray(progress, float))


def _rav(p, xt_star_dest, xt_star_counter, delta):
    p = np.asarray(p, float)
    return p * np.asarray(xt_star_dest, float) - delta * (1.0 - p) * np.asarray(xt_star_counter, float)


def _dzv(start_x, vgk_star_origin, vgk_star_max, alpha, beta, d_max, boundary):
    """Eyestone DZV -- defensive-zone revaluation, Option A (the increment over raw credit).
    M(z) = phi(z,d)*(1 - V_GK(z)/max V_GK) is the published multiplier (~2.5); the composite
    adds the revaluation GAIN it confers on the origin possession value, (M-1)*V_GK(z), so base
    (which surrenders raw origin threat, Option B) and DZV stay orthogonal. Gated to the
    defensive third (phi's active region). See NOTICE (Eyestone xT-GK)."""
    start_x = np.asarray(start_x, float)
    vgk = np.asarray(vgk_star_origin, float)
    phi = _phi_of_d(start_x, alpha, beta, d_max, boundary)
    m = phi * (1.0 - vgk / vgk_star_max)
    in_def_third = start_x < boundary
    return np.where(in_def_third, (m - 1.0) * vgk, 0.0)


def _temporal(k, eta):
    return np.power(eta, np.asarray(k, float))


def _composite(t, base, pev, rav, dzv, gamma, phi):
    # T scales the threat-bearing terms only; the corrective DZV is undiscounted.
    return np.asarray(t, float) * (
        np.asarray(base, float) + gamma * np.asarray(pev, float) + np.asarray(rav, float)
    ) + phi * np.asarray(dzv, float)


# --------------------------------------------------------------------------------------
# Domain filter
# --------------------------------------------------------------------------------------
def _gk_distribution_mask(actions: pd.DataFrame, frames: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """True for in-scope GK distributions: any goalkick, OR a pass/throw_in whose actor is
    the acting team's goalkeeper (resolved from frames' is_goalkeeper flag, which
    derived-GK populates for Metrica/SkillCorner). dtype-safe id matching (ADR-019).
    Non-GK-distribution rows -> False (pass through unchanged downstream)."""
    type_id = actions["type_id"].to_numpy()
    is_goalkick = type_id == _GOALKICK
    is_open = np.isin(type_id, (_PASS, _THROW_IN))

    gk = frames[frames["is_goalkeeper"].astype(bool) & (~frames["is_ball"].astype(bool))]
    keyed_by_game = "game_id" in actions.columns and "game_id" in frames.columns

    gk_team = canonical_id_series(gk["team_id"]).to_numpy()
    gk_player = canonical_id_series(gk["player_id"]).to_numpy()
    act_team = canonical_id_series(actions["team_id"]).to_numpy()
    act_player = canonical_id_series(actions["player_id"]).to_numpy()
    if keyed_by_game:
        gk_game = canonical_id_series(gk["game_id"]).to_numpy()
        act_game = canonical_id_series(actions["game_id"]).to_numpy()
        gk_set = set(zip(gk_game, gk_team, gk_player, strict=True))
        actor_is_gk = np.array([(g, t, p) in gk_set for g, t, p in zip(act_game, act_team, act_player, strict=True)])
    else:
        gk_set = set(zip(gk_team, gk_player, strict=True))
        actor_is_gk = np.array([(t, p) in gk_set for t, p in zip(act_team, act_player, strict=True)])

    return is_goalkick | (is_open & actor_is_gk)


# --------------------------------------------------------------------------------------
# Batch compute
# --------------------------------------------------------------------------------------
def _completion_p(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    geom: pd.DataFrame,
    mask: npt.NDArray[np.bool_],
    links: pd.DataFrame | None,
    completion: GkCompletionModel | None,
) -> npt.NDArray[np.float64]:
    """RAV P(success) for in-scope rows via the GK-distribution completion model. Default =
    the bundled GS ``default`` model; a caller may inject a fitted ``GkCompletionModel``.
    Builds features through the SAME shared density producer + extract path used at train
    (train==serve parity, review C1/C3): one ``_gk_completion_density`` producer, one
    ``extract_gk_completion_features``."""
    from ._gk_completion import (
        GkCompletionModel,
        _gk_completion_density,
        extract_gk_completion_features,
    )

    model = completion if isinstance(completion, GkCompletionModel) else GkCompletionModel.from_variant("default")
    sub = actions.loc[mask]
    sub_geom = geom.loc[mask]
    dens = _gk_completion_density(sub, frames, sub_geom, links)  # the one shared producer
    feats = extract_gk_completion_features(sub_geom.assign(type_id=sub["type_id"].to_numpy()), defender_density=dens)
    return model.predict_proba(feats)


def _resolve_single_provider(frames: pd.DataFrame) -> str | None:
    """The single REAL tracking provider for a one-match frame set (``snapshot`` excluded, C3).
    Raises on >1 (one call = one match = one provider). Returns None when no provider tag is present.
    Single-sourced (CR 2026-06-30 L1): used by both the completion-variant resolution AND the
    geometry distrust decision, so the rule lives in one place."""
    provs = []
    if "source_provider" in frames.columns:
        provs = [p for p in pd.unique(frames["source_provider"].dropna()) if str(p).lower() != "snapshot"]
    if len(provs) > 1:
        raise ValueError(
            f"xT-GK: frames span multiple real providers {sorted(map(str, provs))}; one call = one "
            "match = one provider. Pass an explicit completion= model for a mixed/cross-provider stack."
        )
    return str(provs[0]) if provs else None


def _resolve_completion_for_frames(frames: pd.DataFrame, completion: GkCompletionModel | None):
    """Resolve the GK-completion model + its variant key for a ``compute_xt_gk`` call (D-S2).
    Returns ``(model, variant_key)`` -- the key feeds the ``xt_gk_completion_variant`` provenance
    column (Task 8). A caller-supplied ``GkCompletionModel`` wins (override / mismatched-stack escape,
    m-a; key = its ``shipped_variant`` or ``"custom"``). Otherwise auto-select from
    ``frames["source_provider"]``: raise on >1 REAL provider (``snapshot`` excluded, C3); fall back to
    the bundled ``gs`` ``default`` (key ``"gs"``) with a warning if a variant artifact is absent (D-S1
    may not bundle distinct SkillCorner weights)."""
    from ._gk_completion import GkCompletionModel, variant_key_for_provider

    if isinstance(completion, GkCompletionModel):
        return completion, (getattr(completion, "shipped_variant", None) or "custom")
    key = variant_key_for_provider(_resolve_single_provider(frames))
    try:
        return GkCompletionModel.from_variant(key), key
    except FileNotFoundError:
        if key != "gs":
            warnings.warn(
                f"xT-GK: no bundled GK-completion weights for variant {key!r}; falling back to the "
                "'default' (gs) model.",
                stacklevel=2,
            )
        return GkCompletionModel.from_variant("default"), "gs"  # the model that ACTUALLY scored


def compute_xt_gk(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    xt: ExpectedThreat,
    params: XtGkParams | None = None,
    links: pd.DataFrame | None = None,
    completion: GkCompletionModel | None = None,
) -> pd.DataFrame:
    """Batch xT-GK over GK-distribution actions. Returns a DataFrame indexed like
    ``actions`` with the six xt_gk_* value columns + three provenance columns
    (``xt_gk_origin_source``, ``xt_gk_dest_source``, ``xt_gk_origin_confidence``);
    out-of-scope rows are NaN.

    Goal-kick origin/destination coordinates are derived when the SPADL event omits them
    (real GS data: ~67% NaN origin) via ``resolve_gk_geometry`` -- a scoped, provenance-tagged
    resolution that NEVER mutates ``actions``. RAV's pass-completion probability comes from a
    fitted ``GkCompletionModel`` (the bundled GS ``default`` unless ``completion=`` is
    supplied); the open-play accessible-space xC was OOD on goal-kicks (~31% coverage), so the
    [das] extra is no longer required.

    ``xt`` MUST be a pre-fitted ExpectedThreat fitted on a corpus DISJOINT from the scored
    matches (the OBSO/frozen pattern). This function NEVER fits xT internally (no in-sample
    leakage).

    See NOTICE for full bibliographic citations (Eyestone xT-GK)."""
    from ._gk_geometry import (
        flag_native_goalkick_out_of_region,
        native_origin_is_trusted,
        resolve_gk_geometry,
    )
    from .features import pressure_on_actor  # lazy: avoids an import cycle at module load
    from .utils import link_actions_to_frames

    p = params or XtGkParams()
    # M1 leakage/garbage-in guard: xt MUST be fitted (no self-fit). compute reads xt.xT
    # directly and would not otherwise raise (only .rate() raises NotFittedError).
    if not np.asarray(xt.xT).any():
        raise ValueError(
            "xT-GK requires a FITTED ExpectedThreat (xt.xT is all-zero). Fit xt on a corpus "
            "disjoint from the scored matches; xT-GK never self-fits."
        )

    # Provider-aware completion variant (D-S2): when the caller supplies no model, auto-select from
    # the tracking provider (a proxy for the event/label provider -- actions carry no provider tag;
    # exact in the single-stack case, the `completion=` override covers mismatched stacks, m-a). A
    # frame set spanning >1 REAL provider is a linkage/ingestion bug (one call = one match = one
    # provider); `snapshot` is a synthetic frames-only tag, excluded from the uniqueness check (C3).
    completion_model, completion_key = _resolve_completion_for_frames(frames, completion)
    # Provider-aware native-origin trust (CR 2026-06-30 H1/C1): a broadcast provider's native origin
    # is a ball-detection artifact, not the keeper -> distrust + route through the detection-aware
    # ladder. The same single-source provider resolution enforces one-call-one-match uniformly (C1:
    # a >1-provider frame set now raises here too, even with completion= supplied).
    distrust = not native_origin_is_trusted(_resolve_single_provider(frames))

    out = pd.DataFrame(
        {c: np.full(len(actions), np.nan, dtype=float) for c in _OUTPUT_COLS},
        index=actions.index,
    )
    for c in _COORD_COLS:
        out[c] = np.full(len(actions), np.nan, dtype=float)
    out["xt_gk_origin_source"] = np.full(len(actions), None, dtype=object)
    out["xt_gk_dest_source"] = np.full(len(actions), None, dtype=object)
    out["xt_gk_origin_confidence"] = np.full(len(actions), np.nan, dtype=float)
    out["xt_gk_completion_variant"] = np.full(len(actions), None, dtype=object)
    out["xt_gk_completion_source"] = np.full(len(actions), None, dtype=object)
    out["xt_gk_native_goalkick_out_of_region"] = np.zeros(len(actions), dtype=bool)  # S4 (CR 2026-06-30)

    in_scope = _gk_distribution_mask(actions, frames)
    # Link once (reuse caller-supplied pointers): pressure_on_actor + resolve_gk_geometry +
    # the completion density helper all take ``links``; pass the same pointers to each.
    pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]

    # Conditional coordinate derivation (scoped; NEVER mutates ``actions``). A goal-kick with a
    # NaN native origin gets an imputed origin (in-area tracking-GK -> rule point); a NaN native
    # destination resolves to the in-period next-event start. Provenance + confidence emitted
    # for every in-scope row (off-scope rows stay NaN provenance).
    geom = resolve_gk_geometry(
        actions,
        frames=frames,
        links=pointers,
        distrust_native_origin=distrust,
    )
    out.loc[in_scope, "xt_gk_origin_source"] = geom.loc[in_scope, "origin_source"].to_numpy()
    out.loc[in_scope, "xt_gk_dest_source"] = geom.loc[in_scope, "dest_source"].to_numpy()
    out.loc[in_scope, "xt_gk_origin_confidence"] = geom.loc[in_scope, "origin_confidence"].to_numpy()
    # Resolved-coordinate audit columns (handoff 2026-06-29 item 1): surface the EXACT coords the grid
    # lookups use, emitted for every in-scope row BEFORE the not-scoreable early return so an
    # unresolvable-destination goalkick still shows its resolved origin (+ NaN dest) for inspection.
    out.loc[in_scope, "xt_gk_origin_x"] = geom.loc[in_scope, "origin_x"].to_numpy()
    out.loc[in_scope, "xt_gk_origin_y"] = geom.loc[in_scope, "origin_y"].to_numpy()
    out.loc[in_scope, "xt_gk_dest_x"] = geom.loc[in_scope, "dest_x"].to_numpy()
    out.loc[in_scope, "xt_gk_dest_y"] = geom.loc[in_scope, "dest_y"].to_numpy()
    # S4 (CR 2026-06-30): warn + per-row machine-observable flag for an implausible native goal-kick
    # origin (provider feeding ball-location-as-origin). Never reverts; XtGkReport sums it.
    out["xt_gk_native_goalkick_out_of_region"] = flag_native_goalkick_out_of_region(actions, geom)

    # B2 NaN-safety contract (ADR-003) implemented in the BODY (the @nan_safe_enrichment
    # marker confers no behavior). Route in-scope rows with a NaN identifier to the NaN default.
    id_ok = actions["player_id"].notna().to_numpy() & actions["team_id"].notna().to_numpy()
    # The coords gate reads the RESOLVED geometry: a goal-kick always gets a resolved origin
    # (rule-point fallback), so a row fails only when its DESTINATION cannot be resolved (no
    # native end, no in-period next-event) -> honest NaN (no z' => no RAV/xT*(z')), NOT
    # base-rated. So in this RAV path the model's geometry-unscoreable base-rate fallback never
    # fires (that path is exercised only by the standalone compute_gk_completion).
    coords_ok = (
        np.isfinite(geom["origin_x"].to_numpy())
        & np.isfinite(geom["origin_y"].to_numpy())
        & np.isfinite(geom["dest_x"].to_numpy())
        & np.isfinite(geom["dest_y"].to_numpy())
    )
    mask = in_scope & id_ok & coords_ok
    if not mask.any():
        return out

    sub = actions.loc[mask]
    sub_geom = geom.loc[mask]

    xt_star = _convolve_grid(xt.xT, p.convolution_sigma)  # raw -- base + RAV (Option B, unchanged)
    # GK-revalued surface V_GK = xT (.) phi(z,d); convolved like xT*. PEV + DZV read this; base +
    # RAV stay raw (the Eyestone invariant: phi enters value via PEV and DZV ONLY).
    # explicit 2-tuple: numpy's ``.shape`` types as tuple[int, ...] on some stub versions, which
    # fails _phi_grid's tuple[int, int] param under the CI pyright (pre-existing cross-version nit).
    xt_shape = (int(xt.xT.shape[0]), int(xt.xT.shape[1]))
    phi_grid = _phi_grid(xt_shape, p.dzv_alpha, p.dzv_beta, p.dzv_d_max, p.defensive_third_boundary)
    vgk_star = _convolve_grid(np.asarray(xt.xT, float) * phi_grid, p.convolution_sigma)
    vgk_max = float(np.nanmax(vgk_star))

    sx = sub_geom["origin_x"].to_numpy(float)  # RESOLVED origin (derived coords feed compute)
    sy = sub_geom["origin_y"].to_numpy(float)
    ex = sub_geom["dest_x"].to_numpy(float)  # RESOLVED destination
    ey = sub_geom["dest_y"].to_numpy(float)

    dest_star = _grid_value(xt_star, ex, ey)  # raw -- RAV owns the destination (Option B)
    origin_star = _grid_value(xt_star, sx, sy)  # raw -- base (Option B)
    base = _base(origin_star)
    # CHANGE 1 (Eyestone Q1+Q2): PEV's forward gain is measured on the revalued surface -- raw xT
    # flatlines in keeper zones (the measured PEV inertia), so progress is V_GK*(z') - V_GK*(z).
    dest_vgk = _grid_value(vgk_star, ex, ey)
    origin_vgk = _grid_value(vgk_star, sx, sy)
    progress = _progress(dest_vgk, origin_vgk)  # forward move value on the revalued surface (feeds PEV)

    # Pressure on the actor at the (resolved) origin: pressure_on_actor locates the actor from
    # start_x/start_y, so feed it the RESOLVED origin -- a NaN-native-origin goalkick must use
    # its derived origin here too (consistent with base/dzv/rav above), not the NaN native coord.
    sub_for_pressure = sub.copy()
    sub_for_pressure["start_x"] = sx
    sub_for_pressure["start_y"] = sy
    rho_raw = pressure_on_actor(sub_for_pressure, frames, method=p.pressure_method, links=pointers).to_numpy(float)
    rho = _normalize_pressure(rho_raw, p.pressure_scale)
    pev = _pev(rho, progress)

    pc = _completion_p(actions, frames, geom, mask, pointers, completion_model)  # RAV owns z' (Option B)
    # Per-type base-rate serve switch (spec 2026-06-09 §2.3/m3): a type whose held-out AUC can't beat
    # chance with confidence serves the calibrated per-type base rate (tagged "base_rate") instead of
    # the geometric p. Geometry-missing rows are already excluded from `mask` (m2), so the per-type
    # gate is the only base-rate trigger here.
    tids = actions.loc[mask, "type_id"].to_numpy()
    serve_mode = completion_model.serve_mode_for_types(tids)
    is_base = serve_mode == "base_rate"
    if is_base.any():
        pc[is_base] = completion_model.base_rate_for_types(tids[is_base])
    out.loc[mask, "xt_gk_completion_variant"] = completion_key
    out.loc[mask, "xt_gk_completion_source"] = np.where(is_base, "base_rate", "model")
    rav = _rav(pc, dest_star, _counter_value(xt_star, ex, ey), p.delta)

    # CHANGE 2 (Eyestone Q3): DZV = the published revaluation multiplier's increment on the origin
    # possession value (Option A), on the revalued surface; gated to the defensive third.
    dzv = _dzv(sx, origin_vgk, vgk_max, p.dzv_alpha, p.dzv_beta, p.dzv_d_max, p.defensive_third_boundary)

    k = _possession_depth(actions)[mask]
    t = _temporal(k, p.eta)

    composite = _composite(t, base, pev, rav, dzv, p.gamma, p.phi)

    # M2 surface (don't silently swallow) unlinked in-scope rows. Per-row detail is in the
    # provenance link_quality_score column; FIXED message so warnings dedup collapses it
    # (xfns calls this 3x/slot/match).
    if bool(np.isnan(composite).any()):
        warnings.warn(
            "xT-GK: one or more in-scope GK distributions produced NaN xt_gk (pressure could "
            "not link to a frame); see the link_quality_score column.",
            stacklevel=2,
        )

    out.loc[mask, "xt_gk_base"] = base
    out.loc[mask, "xt_gk_pev"] = pev
    out.loc[mask, "xt_gk_rav"] = rav
    out.loc[mask, "xt_gk_dzv"] = dzv
    out.loc[mask, "xt_gk_pressure"] = rho
    out.loc[mask, "xt_gk"] = composite
    return out
