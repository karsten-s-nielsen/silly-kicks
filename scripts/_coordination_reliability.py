"""A-09: per-construct reliability reduce for the TF-58 D3 validation artifact (spec section 8.5, ADR-111).

The single source of the per-construct grain and reduce that D3 reports and D1/D2 share. A **construct** is a
metric column x the keys that define WHAT it measures (C.1): pair families ``level`` + ``signal_a`` + ``signal_b``
+ ``axis`` (+ ``phase_index`` for the phase-row table); spectral ``signal``; cluster / team-sync / RSI ``axis``.
The measured **unit** (C.4) is the team, the player, or the unordered player pair; keys are per-(match, entity),
with the two match-halves as the retest axis -- so the binding reliability is a within-match internal consistency,
an upper bound on true match-to-match reliability (``HONESTY_LINES``).

Binding reliability is linear ICC(1) (``scripts/_reliability.icc1``) for a linear construct and rotation-invariant
circular reliability (``silly_kicks.coordination._kernels._circular.circular_reliability``) for a circular-mean
construct -- a plain ICC on a circular mean is origin-dependent. Every cell carries a 95% bootstrap CI and an
explicit power verdict from the pre-registered thresholds in ``_coordination_thresholds``; an underpowered or
hopelessly imprecise cell is terminal "unmeasurable" and is NEVER pooled up a level to rescue power.

Pure numpy / pandas, no I/O; the D3 driver threads the sample tables in and serialises the returned cells.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats

from scripts import _coordination_thresholds as thr
from scripts._reliability import _COMPARABILITY_ICC_TOL, icc1
from silly_kicks.coordination._columns import (
    COORDINATION_CLUSTER_PLAYER_METRIC_COLUMNS,
    COORDINATION_CLUSTER_TEAM_METRIC_COLUMNS,
    COORDINATION_PAIR_METRIC_COLUMNS,
    COORDINATION_PAIR_PHASE_METRIC_COLUMNS,
    COORDINATION_RSI_METRIC_COLUMNS,
    COORDINATION_SPECTRAL_METRIC_COLUMNS,
    COORDINATION_TEAM_SYNC_METRIC_COLUMNS,
    COVERAGE_COLUMNS,
    reliability_kind,
)
from silly_kicks.coordination._kernels._circular import circular_reliability
from silly_kicks.id_compat import canonical_id

#: The C.1 construct grain per shard table: the keys that define WHAT a metric column measures.
CONSTRUCT_KEY_COLS: Mapping[str, tuple[str, ...]] = {
    "pair": ("level", "signal_a", "signal_b", "axis"),
    "pair_phase": ("level", "signal_a", "signal_b", "axis", "phase_index"),
    "spectral": ("signal",),
    "cluster_team": ("axis",),
    "cluster_player": ("axis",),
    "team_sync": ("axis",),
    "rsi": ("axis",),
}

#: The metric columns each table emits (for the construct sweep); coverage columns are excluded below.
_TABLE_METRIC_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "pair": COORDINATION_PAIR_METRIC_COLUMNS,
    "pair_phase": COORDINATION_PAIR_PHASE_METRIC_COLUMNS,
    "spectral": COORDINATION_SPECTRAL_METRIC_COLUMNS,
    "cluster_team": COORDINATION_CLUSTER_TEAM_METRIC_COLUMNS,
    "cluster_player": COORDINATION_CLUSTER_PLAYER_METRIC_COLUMNS,
    "team_sync": COORDINATION_TEAM_SYNC_METRIC_COLUMNS,
    "rsi": COORDINATION_RSI_METRIC_COLUMNS,
}

#: Tables whose metric is a MUTUAL team-pair quantity: attribute it to BOTH teams it describes (the established
#: ``_metric_samples`` convention) so a symmetric coordination value becomes two team observations.
_MUTUAL_TEAM_TABLES = frozenset({"team_sync", "rsi"})

#: The SkillCorner detection-coverage axis for the decile stratification -- the one coverage column every table
#: carries, so every construct is binned on the SAME observed-fraction axis.
OBS_COLUMN = "coord_detected_share"

#: Seeded group-bootstrap for the reliability CI (reproducible across runs / platforms); the D3 reduce is a
#: once-off aggregation, not a per-match hot path.
RELIABILITY_BOOTSTRAP_SEED = 20260926
RELIABILITY_BOOTSTRAP_DRAWS = 400

#: Nominal regulation-half length (s); the driver overrides it with the measured mean half duration.
NOMINAL_HALF_LENGTH_S = 2700.0

#: The two C.4 report-honesty lines (verbatim in metrics.json provenance and report.md).
HONESTY_LINES: tuple[str, str] = (
    "Across-halves reliability is a within-match split-half (a shared opponent and setup): internal "
    "consistency, and an upper bound on true match-to-match reliability.",
    "Cross-match player reliability is unmeasurable on anonymised corpora (no roster linkage across matches); "
    "stable-roster providers are a future extension.",
)


def reliability_scored_columns(table: str) -> tuple[str, ...]:
    """The metric columns of ``table`` that get a reliability cell: all metrics bar coverage denominators."""
    return tuple(c for c in _TABLE_METRIC_COLUMNS.get(table, ()) if c not in COVERAGE_COLUMNS)


@dataclass(frozen=True)
class Construct:
    """A metric column at the C.1 grain, with its measured unit (C.4) and reliability kind (C.3)."""

    column: str
    table: str
    key: Mapping[str, object] = field(default_factory=dict)
    unit: str = "team"
    kind: str = "linear"


def _canon_pair(a: object, b: object) -> str:
    """A canonical unordered player-pair key: the two ids canonicalised, sorted, joined (``id_compat``)."""
    ids = sorted(str(canonical_id(a)) for a in (a, b))
    return "|".join(ids)


def _attach_entity(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``entity`` column (the measured unit, C.4) and a ``unit`` label; melt mutual team quantities.

    A mutual team quantity (``team_team`` pair, ``team_sync``, ``rsi``) is attributed to BOTH teams, so the frame
    gains two rows per input row. A dyad pair is keyed on the unordered player pair; every other table keeps one
    row keyed on its single team or player.
    """
    if table in _MUTUAL_TEAM_TABLES:
        a = df.assign(entity=df["team_a_id"], unit="team")
        b = df.assign(entity=df["team_b_id"], unit="team")
        return pd.concat([a, b], ignore_index=True)
    if table in ("pair", "pair_phase"):
        parts: list[pd.DataFrame] = []
        tt = df[df["level"] == "team_team"]
        if len(tt):
            parts.append(tt.assign(entity=tt["team_a_id"], unit="team"))
            parts.append(tt.assign(entity=tt["team_b_id"], unit="team"))
        single = df[df["level"].isin(("cross_variable", "intra_team"))]
        if len(single):
            parts.append(single.assign(entity=single["team_a_id"], unit="team"))
        dyad = df[df["level"] == "dyad"]
        if len(dyad):
            ent = [_canon_pair(a, b) for a, b in zip(dyad["player_a_id"], dyad["player_b_id"], strict=True)]
            parts.append(dyad.assign(entity=ent, unit="unordered_pair"))
        return (
            pd.concat(parts, ignore_index=True)
            if parts
            else df.iloc[:0].assign(entity=pd.Series(dtype="object"), unit="team")
        )
    if table == "cluster_player":
        return df.assign(entity=df["player_id"], unit="player")
    # team-keyed tables (spectral, cluster_team)
    return df.assign(entity=df["team_id"], unit="team")


def derive_constructs(tables: Mapping[str, pd.DataFrame]) -> list[tuple[Construct, pd.DataFrame]]:
    """Enumerate every observed construct and its per-(match, entity, half) samples (C.1 grain, C.4 units).

    Only the period windows are kept (one match-half per observation). A construct is emitted only where it has at
    least one finite sample, so the set is data-driven (~2000 cells on the full corpus).
    """
    out: list[tuple[Construct, pd.DataFrame]] = []
    for table, key_cols in CONSTRUCT_KEY_COLS.items():
        df = tables.get(table, pd.DataFrame())
        if not len(df):
            continue
        if "window_kind" in df.columns:
            df = df[df["window_kind"] == "period"]
        if not len(df):
            continue
        ent = _attach_entity(table, df)
        if not len(ent):
            continue
        has_obs = OBS_COLUMN in ent.columns
        for col in reliability_scored_columns(table):
            if col not in ent.columns:
                continue
            cols = ["game_id", "period_id", "entity", "unit", *key_cols, col]
            if has_obs:
                cols.append(OBS_COLUMN)
            sub = ent[cols].rename(columns={col: "value", OBS_COLUMN: "obs_frac"})
            sub = sub.assign(value=pd.to_numeric(sub["value"], errors="coerce")).dropna(subset=["value", "entity"])
            if not len(sub):
                continue
            keep = ["game_id", "entity", "period_id", "value"] + (["obs_frac"] if has_obs else [])
            for keyvals, grp in sub.groupby(list(key_cols), sort=True, dropna=False, observed=True):
                keytuple = keyvals if isinstance(keyvals, tuple) else (keyvals,)
                construct = Construct(
                    column=col,
                    table=table,
                    key=dict(zip(key_cols, keytuple, strict=True)),
                    unit=str(grp["unit"].iloc[0]),
                    kind=reliability_kind(col),
                )
                out.append((construct, grp[keep].reset_index(drop=True)))
    return out


def _group_keys(samples: pd.DataFrame) -> np.ndarray:
    """The per-(match, entity) retest group label as a single string (icc1 / circular_reliability grouping)."""
    return (samples["game_id"].astype(str) + "||" + samples["entity"].astype(str)).to_numpy()


def _n_retest_groups(samples: pd.DataFrame, groups: np.ndarray) -> int:
    """Groups with at least two finite observations -- the retest-capable count for the power verdict."""
    s = pd.Series(np.isfinite(samples["value"].to_numpy(dtype="float64")), index=groups)
    return int((s.groupby(level=0, observed=True).sum() >= 2).sum())


def _point_estimate(values: np.ndarray, groups: np.ndarray, kind: str) -> tuple[float, float]:
    """The binding reliability point estimate and overall concentration (rbar; NaN for a linear construct)."""
    if kind == "circular":
        return circular_reliability(values, groups)
    return icc1(values, groups), float("nan")


def _bootstrap_ci(samples: pd.DataFrame, groups: np.ndarray, kind: str, seed: int, draws: int) -> tuple[float, float]:
    """Percentile 95% CI by resampling the retest GROUPS with replacement (groups kept whole)."""
    uniq = pd.unique(groups)
    if uniq.size < 2:
        return float("nan"), float("nan")
    by_group = {g: samples.index[groups == g].to_numpy() for g in uniq}
    values = samples["value"].to_numpy(dtype="float64")
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(draws):
        picks = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([by_group[g] for g in picks])
        # relabel each resampled group uniquely so a group drawn twice counts as two groups
        labels = np.concatenate([np.full(by_group[g].size, f"{g}#{i}") for i, g in enumerate(picks)])
        val, _ = _point_estimate(values[idx], labels, kind)
        if np.isfinite(val):
            stats.append(val)
    if len(stats) < 2:
        return float("nan"), float("nan")
    lo, hi = (float(x) for x in np.percentile(stats, [2.5, 97.5]))
    return lo, hi


def _power_verdict(
    value: float, ci: tuple[float, float], n_groups: int, kind: str, rbar: float
) -> tuple[str, str | None]:
    """Apply the pre-registered thresholds. Returns ``(power, unmeasurable_reason)``.

    Order: too few groups ("n<min"); then, for a circular mean, concentration below the floor ("Rbar->0");
    then a CI half-width wider than the ceiling ("ci_too_wide"). Otherwise "measured".
    """
    if n_groups < thr.RELIABILITY_MIN_N_GROUPS:
        return "unmeasurable", "n<min"
    if kind == "circular" and (not np.isfinite(rbar) or rbar < thr.CIRCULAR_RELIABILITY_MIN_RBAR):
        return "unmeasurable", "Rbar->0"
    lo, hi = ci
    if not (np.isfinite(value) and np.isfinite(lo) and np.isfinite(hi)):
        return "unmeasurable", "ci_too_wide"
    if (hi - lo) / 2.0 > thr.RELIABILITY_MAX_CI_HALFWIDTH:
        return "unmeasurable", "ci_too_wide"
    return "measured", None


def _circular_diagnostics(samples: pd.DataFrame, groups: np.ndarray) -> dict:
    """cos/sin component ICCs (origin-dependent -- diagnostics only) with the pinned origin (C.3)."""
    rad = np.radians(samples["value"].to_numpy(dtype="float64"))
    return {
        "icc_cos": _finite_or_none(icc1(np.cos(rad), groups)),
        "icc_sin": _finite_or_none(icc1(np.sin(rad), groups)),
        "origin_deg": 0.0,
    }


def _split_half(samples: pd.DataFrame, half_length_s: float) -> dict:
    """Within-match split-half: the first two period values per (match, entity), Pearson + Spearman-Brown (A-53)."""
    groups = _group_keys(samples)
    df = samples.assign(_g=groups).sort_values(["_g", "period_id"])
    firsts, seconds = [], []
    for _, grp in df.groupby("_g", sort=False, observed=True):
        vals = grp["value"].to_numpy(dtype="float64")
        vals = vals[np.isfinite(vals)]
        if vals.size >= 2:
            firsts.append(vals[0])
            seconds.append(vals[1])
    value: float
    if len(firsts) >= 3 and np.std(firsts) > 0 and np.std(seconds) > 0:
        r = float(np.corrcoef(firsts, seconds)[0, 1])
        value = 2.0 * r / (1.0 + r) if (1.0 + r) != 0 else float("nan")
    else:
        value = float("nan")
    return {
        "value": value,
        "spearman_brown": True,
        "half_length_s": float(half_length_s),
        "n_pairs": len(firsts),
    }


def _finite_or_none(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def _deciles(samples: pd.DataFrame, kind: str) -> list[dict]:
    """Each construct's value distribution by SkillCorner observed-fraction (``coord_detected_share``) decile.

    Linear: median + inter-quartile spread. Circular: circular mean + circular SD (degrees), so the wrap is
    respected (a plain median across +/-180 is wrong). Empty when the coverage axis is absent.
    """
    from scripts.derive_coordination_params import _as_float, _obs_bin

    if "obs_frac" not in samples.columns:
        return []
    df = samples.assign(obs_frac=pd.to_numeric(samples["obs_frac"], errors="coerce"))
    df = df.dropna(subset=["value", "obs_frac"])
    if not len(df):
        return []
    out: list[dict] = []
    for binval, grp in df.assign(_d=df["obs_frac"].map(_obs_bin)).groupby("_d", sort=True, observed=True):
        v = grp["value"].to_numpy(dtype="float64")
        if kind == "circular":
            # scipy.stats.circmean/circstd carry no parameter annotations; pyright infers ``low: int``
            # from the default ``low=0`` and would reject the float ``-np.pi`` these (numerically
            # load-bearing) calls pass. Bind through ``Any`` -- the standard pattern for an unannotated
            # third-party callable -- rather than altering the call.
            _circmean: Any = scipy.stats.circmean
            _circstd: Any = scipy.stats.circstd
            rad = np.radians(v)
            center = float(np.degrees(_circmean(rad, high=np.pi, low=-np.pi)))
            spread = float(np.degrees(_circstd(rad, high=np.pi, low=-np.pi)))
        else:
            q25, q75 = (float(x) for x in np.percentile(v, [25, 75]))
            center = float(np.median(v))
            spread = q75 - q25
        out.append({"bin": round(float(_as_float(binval)), 1), "n": int(v.size), "median": center, "spread": spread})
    return out


def reliability_cell(
    construct: Construct,
    samples: pd.DataFrame,
    *,
    seed: int = RELIABILITY_BOOTSTRAP_SEED,
    draws: int = RELIABILITY_BOOTSTRAP_DRAWS,
    half_length_s: float = NOMINAL_HALF_LENGTH_S,
) -> dict:
    """The per-construct metrics.json cell (minus the cross-provider ``poolability`` and the ``deciles``).

    Computes the binding reliability + 95% bootstrap CI + power verdict (C.2), the circular diagnostics (C.3), and
    the split-half Spearman-Brown block (A-53). A terminal "unmeasurable" cell reports ``value=None``.
    """
    values = samples["value"].to_numpy(dtype="float64")
    groups = _group_keys(samples)
    kind = construct.kind
    n_groups = _n_retest_groups(samples, groups)
    n_obs = int(np.isfinite(values).sum())
    point, rbar = _point_estimate(values, groups, kind)
    ci = _bootstrap_ci(samples, groups, kind, seed, draws)
    power, reason = _power_verdict(point, ci, n_groups, kind, rbar)
    estimator = "circular_reliability" if kind == "circular" else "icc1"
    reliability = {
        "value": _finite_or_none(point) if power == "measured" else None,
        "ci": [_finite_or_none(ci[0]), _finite_or_none(ci[1])],
        "estimator": estimator,
        "n_groups": n_groups,
        "n_obs": n_obs,
        "power": power,
        "unmeasurable_reason": reason,
    }
    if kind == "circular":
        reliability["rbar"] = _finite_or_none(rbar)
    return {
        "column": construct.column,
        "construct_key": dict(construct.key),
        "unit": construct.unit,
        "kind": kind,
        "reliability": reliability,
        "diagnostics": _circular_diagnostics(samples, groups) if kind == "circular" else {},
        "split_mode": _split_half(samples, half_length_s),
        "deciles": _deciles(samples, kind),
        "poolability": None,  # attached cross-provider by build_constructs_report
    }


def iter_cells(
    tables: Mapping[str, pd.DataFrame], *, seed: int = RELIABILITY_BOOTSTRAP_SEED, half_length_s: float | None = None
) -> list[dict]:
    """Every observed construct's cell for one provider (cross-provider poolability attached by the assembler)."""
    hl = NOMINAL_HALF_LENGTH_S if half_length_s is None else half_length_s
    return [reliability_cell(c, s, seed=seed, half_length_s=hl) for c, s in derive_constructs(tables)]


def _construct_identity(cell: Mapping) -> tuple:
    """A hashable identity for the same construct across providers: column + sorted key items + unit."""
    key_items = tuple(sorted((str(k), str(v)) for k, v in cell["construct_key"].items()))
    return (cell["column"], key_items, cell["unit"])


def _attach_poolability(cells: list[dict]) -> None:
    """Attach each construct's cross-provider poolability in place (``compare_providers`` tolerance, per construct).

    POOLABLE iff every provider that measured the construct reports a finite, same-sign binding reliability within
    ``_COMPARABILITY_ICC_TOL``. A construct measured by fewer than two providers is not poolable (stated, not NaN).
    """
    by_identity: dict[tuple, list[dict]] = {}
    for cell in cells:
        by_identity.setdefault(_construct_identity(cell), []).append(cell)
    for group in by_identity.values():
        measured = {
            c.get("provider"): c["reliability"]["value"]
            for c in group
            if c["reliability"]["power"] == "measured" and c["reliability"]["value"] is not None
        }
        vals = list(measured.values())
        if len(vals) >= 2:
            spread = float(max(vals) - min(vals))
            poolable = all(np.sign(vals[0]) == np.sign(v) for v in vals) and spread <= _COMPARABILITY_ICC_TOL
            pool = {
                "providers": {str(p): float(v) for p, v in measured.items()},
                "icc_spread": spread,
                "poolable": bool(poolable),
                "n_providers": len(vals),
            }
        else:
            pool = {
                "providers": {str(p): float(v) for p, v in measured.items()},
                "icc_spread": None,
                "poolable": False,
                "n_providers": len(vals),
            }
        for cell in group:
            cell["poolability"] = dict(pool)


def _column_summary(cells: list[dict]) -> dict:
    """Per column, a DESCRIPTIVE summary (median + range of measured reliability across its constructs, and the
    count of unmeasurable cells). NEVER presented as "the column's reliability" (spec section 8.5)."""
    out: dict[str, dict] = {}
    by_col: dict[str, list[dict]] = {}
    for cell in cells:
        by_col.setdefault(cell["column"], []).append(cell)
    for col, group in by_col.items():
        measured = [c["reliability"]["value"] for c in group if c["reliability"]["value"] is not None]
        n_unmeasurable = sum(1 for c in group if c["reliability"]["power"] == "unmeasurable")
        out[col] = {
            "n_constructs": len(group),
            "n_measured": len(measured),
            "n_unmeasurable": n_unmeasurable,
            "reliability_median": float(np.median(measured)) if measured else None,
            "reliability_min": float(min(measured)) if measured else None,
            "reliability_max": float(max(measured)) if measured else None,
        }
    return out


def build_constructs_report(
    tables_by_provider: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    seed: int = RELIABILITY_BOOTSTRAP_SEED,
    half_length_by_provider: Mapping[str, float] | None = None,
) -> dict:
    """The full per-construct reliability section of metrics.json: the flat ``constructs`` list (each tagged with its
    provider, with cross-provider ``poolability`` attached), the two honesty lines, and the descriptive per-column
    summary for report.md (spec section 8.5, C.4/C.7)."""
    halves = half_length_by_provider or {}
    cells: list[dict] = []
    for provider, tables in tables_by_provider.items():
        hl = halves.get(provider, NOMINAL_HALF_LENGTH_S)
        for cell in iter_cells(tables, seed=seed, half_length_s=hl):
            cell["provider"] = str(provider)
            cells.append(cell)
    _attach_poolability(cells)
    return {
        "honesty": list(HONESTY_LINES),
        "constructs": cells,
        "column_summary": _column_summary(cells),
    }
