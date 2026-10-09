#!/usr/bin/env python
"""D3 (spec 8.5): the in-cycle coordination validation artifact -- REPORTED, never gated.

Runs the FINAL per-provider parameters -- built from D1's ``derivation.json`` and D2's ``calibration.json``
(``--derivation``/``--calibration``: the artifact handoff, owner ruling 2026-10-02, so every pass runs on the clean
commit-1 tree; ``--in-package-params`` for a dev run on the committed module) -- on the full TF-58 corpus and writes
``<--out>/{metrics.json,report.md}`` (copied into ``docs/research/tf58_team_coordination/`` at commit 2): the seven
pre-registered hypotheses
(H1-H7, spec 8.5), per-metric/per-provider reliability (ICC(1) / split-half / Type-II slope), cross-provider
poolability, the SkillCorner coverage stratification, the D20 stoppage leg, the occlusion error curves and the
real-data liveness of every metric column. A failed hypothesis is a RECORDED finding for the owner to rule on
(ADR); nothing is auto-dropped, nothing here changes a library default.

Four shardable passes (ADR-052; each is one ``for_each`` with its own generation):
  --pass metrics     final params, ``n_surrogates=D3_N_SURROGATES`` -> all 7 family tables + windows + the H7
                     switch-event blocks (``include_switch_events=True``), per match.
  --pass stoppage    GS + IDSSE (true ``ball_state``, D20): each match's tables under stoppage_evidence in
                     ``STOPPAGE_MODES`` plus the event-vs-ball_state interval agreement (precision/recall/IoU).
  --pass occlusion   GS + IDSSE: the final occlusion error curves at the calibrated FOV width W, read from
                     ``--derivation`` (D1's occlusion block).
  --pass reduce      combines EVERY worker's share of the three passes (refusing a missing worker, a failed match
                     or a partial corpus: ``--corpus-json``, spec 8.3) and writes ``metrics.json`` (with
                     ``input_contract`` and the population it judged) and ``report.md``.

Every corpus pass runs per worker over a disjoint ``--match-ids-json`` slice and writes its own share
(``scripts._coordination_corpus.write_worker_partial``); the reduce combines them (``combine_workers``).

Shared corpus flags come from ``_coordination_corpus.add_common_args`` (``--out``/``--token``/``--max-matches``/
``--cache-dir``/``--match-ids-json``/``--providers``/``--allow-dirty``/``--list-matches``); the tree is checked with
``require_clean_tree(git_provenance())`` FIRST (ADR-037). Hypothesis thresholds live ONLY in
``scripts._coordination_thresholds`` (referenced, never inlined -- ``test_reduce_reports_every_hypothesis_
with_threshold_refs`` AST-guards this driver against a hardcoded threshold value).
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from scripts._driver import CorpusPassResult

from scripts import _coordination_thresholds as thr

# Sibling driver modules imported in package form so ``python -m scripts.validate_team_coordination`` resolves
# them; tests import this driver and patch the names bound HERE (``corpus_source`` / ``match_tables`` / ...).
from scripts._coordination_corpus import (
    PARAMS_PROVENANCE_FIELDS,
    StageTimer,
    add_common_args,
    add_params_args,
    combine_workers,
    corpus_source,
    corpus_visibility_label,
    expected_corpus,
    match_tables,
    params_resolver,
    read_artifact,
    run_params_token,
    split_tables,
    table_pairs,
    visibility_preflight,
    write_worker_partial,
)
from scripts._coordination_hypotheses import evaluate_hypotheses, gated_pass
from scripts._coordination_reliability import CONSTRUCT_KEY_COLS, build_constructs_report

# The representative family metric map and the occlusion kernel are single-sourced in D1 (shared driver
# discipline, ADR-052): D3 reports the SAME representative metric per family and the SAME occlusion error.
from scripts.derive_coordination_params import (
    _FAMILY_METRIC,
    OCCLUSION_PROVIDERS,
    _as_float,
    occlusion_metrics,
)
from silly_kicks.coordination import CoordinationParams
from silly_kicks.coordination._columns import (
    COORDINATION_CLUSTER_PLAYER_METRIC_COLUMNS,
    COORDINATION_CLUSTER_TEAM_METRIC_COLUMNS,
    COORDINATION_PAIR_METRIC_COLUMNS,
    COORDINATION_PAIR_PHASE_METRIC_COLUMNS,
    COORDINATION_RSI_METRIC_COLUMNS,
    COORDINATION_SPECTRAL_METRIC_COLUMNS,
    COORDINATION_TEAM_SYNC_METRIC_COLUMNS,
)
from silly_kicks.coordination._windows import resolve_stoppages
from silly_kicks.tracking._geometry import GEOMETRY_VERSION

#: Bumping this invalidates every D3 shard generation (it is in each pass's ``token_inputs``).
_SHARD_SCHEMA_VERSION = "tf58-d3-1"
#: The metrics.json top-level ``schema_version`` (A-09 per-construct contract); bump on a schema change.
_METRICS_SCHEMA_VERSION = "tf58-metrics-2"
#: The final validation surrogate count (spec 8.5): a full baseline draw, not the D2 layer-a 0. Single-sourced in
#: _coordination_thresholds (referenced, never inlined, so the threshold-literal gate stays green; review A-41/A-56).
D3_N_SURROGATES = thr.D3_N_SURROGATES
#: The H7 switch-surrogate seed (reproducible across runs / platforms; not a hypothesis threshold).
#: The D20 stoppage leg runs only where a true ``ball_state`` exists (GS + IDSSE): the leg measures the
#: event-derived approximation's error against it.
STOPPAGE_PROVIDERS = ("gradientsports", "idsse")
#: The three window-splitting regimes the stoppage leg compares (C12): true dead-ball, event-derived, none.
STOPPAGE_MODES = ("ball_state", "events", "none")

#: Every family's declared metric columns, for the real-data liveness sweep (spec 8.5).
_TABLE_METRIC_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "pair": COORDINATION_PAIR_METRIC_COLUMNS,
    "pair_phase": COORDINATION_PAIR_PHASE_METRIC_COLUMNS,
    "spectral": COORDINATION_SPECTRAL_METRIC_COLUMNS,
    "cluster_team": COORDINATION_CLUSTER_TEAM_METRIC_COLUMNS,
    "cluster_player": COORDINATION_CLUSTER_PLAYER_METRIC_COLUMNS,
    "team_sync": COORDINATION_TEAM_SYNC_METRIC_COLUMNS,
    "rsi": COORDINATION_RSI_METRIC_COLUMNS,
}


# =============================================================================== pure reduce kernels (unit-tested)
def _interval_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two ``[start, end)`` intervals (0 when they do not overlap)."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return float(inter / union) if union > 0 else 0.0


def stoppage_interval_agreement(
    event_intervals: Mapping[tuple[object, object], np.ndarray],
    ball_state_intervals: Mapping[tuple[object, object], np.ndarray],
) -> dict:
    """Agreement of event-derived stoppages against the true ``ball_state`` stoppages (D20, spec 8.5).

    Compared per ``(game, period)`` and pooled: an event interval is a HIT when it overlaps any ball-state
    interval in the same key (precision = hits / event count); a ball-state interval is RECALLED when any event
    interval overlaps it (recall = recalled / ball-state count); ``mean_iou`` is the mean over each event
    interval's best-overlapping ball-state interval (matched pairs only). Both dicts map ``(game, period)`` to
    an ``(S, 2)`` array of ``[start, end)`` rows, as :func:`resolve_stoppages` returns.
    """
    keys = set(event_intervals) | set(ball_state_intervals)
    n_event = n_bs = event_hits = bs_recalled = 0
    ious: list[float] = []
    for key in keys:
        ev = np.asarray(event_intervals.get(key, np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
        bs = np.asarray(ball_state_intervals.get(key, np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
        n_event += len(ev)
        n_bs += len(bs)
        for e in ev:
            overlaps = [_interval_iou(e, s) for s in bs]
            best = max(overlaps) if overlaps else 0.0
            if best > 0.0:
                event_hits += 1
                ious.append(best)
        for s in bs:
            if any(_interval_iou(e, s) > 0.0 for e in ev):
                bs_recalled += 1
    return {
        "precision": float(event_hits / n_event) if n_event else float("nan"),
        "recall": float(bs_recalled / n_bs) if n_bs else float("nan"),
        "mean_iou": float(np.mean(ious)) if ious else float("nan"),
        "n_event": int(n_event),
        "n_ball_state": int(n_bs),
    }


def liveness_block(tables: Mapping[str, pd.DataFrame]) -> dict:
    """Every declared metric column: non-NaN AND non-constant somewhere in the corpus (spec 8.5). REPORTED.

    A column absent from its table, all-NaN, or constant is a ``dead`` finding for the owner -- never a drop.
    """
    columns: dict[str, dict] = {}
    dead: list[str] = []
    for table, cols in _TABLE_METRIC_COLUMNS.items():
        df = tables.get(table, pd.DataFrame())
        for col in cols:
            key = f"{table}.{col}"
            if not len(df) or col not in df.columns:
                columns[key] = {"live": False, "n_finite": 0, "n_unique": 0, "reason": "absent"}
                dead.append(key)
                continue
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            v = v[np.isfinite(v)]
            n_unique = int(np.unique(v).size)
            live = bool(v.size > 0 and n_unique > 1)
            columns[key] = {"live": live, "n_finite": int(v.size), "n_unique": n_unique}
            if not live:
                columns[key]["reason"] = "all_nan" if v.size == 0 else "constant"
                dead.append(key)
    return {"columns": columns, "dead": dead, "n_dead": len(dead)}


def stoppage_leg(stoppage_df: pd.DataFrame) -> dict:
    """Reduce the stoppage shards (D20): mean interval agreement over matches + each representative metric's mean
    under the three splitting regimes (the effect of splitting choice on the numbers, spec 8.5)."""
    agree = (
        stoppage_df[stoppage_df["table"] == "stoppage_agreement"] if "table" in stoppage_df.columns else pd.DataFrame()
    )
    agreement = {
        "precision_mean": _nanmean(agree, "precision"),
        "recall_mean": _nanmean(agree, "recall"),
        "mean_iou": _nanmean(agree, "mean_iou"),
        "n_matches": len(agree),
    }
    by_mode: dict[str, dict] = {}
    if "stoppage_mode" in stoppage_df.columns:
        for family, (table, col) in _FAMILY_METRIC.items():
            fam_rows = (
                stoppage_df[(stoppage_df["table"] == table)] if "table" in stoppage_df.columns else pd.DataFrame()
            )
            if not len(fam_rows) or col not in fam_rows.columns:
                continue
            means = {}
            for mode in STOPPAGE_MODES:
                v = pd.to_numeric(fam_rows.loc[fam_rows["stoppage_mode"] == mode, col], errors="coerce").to_numpy()
                v = v[np.isfinite(v)]
                means[mode] = float(np.mean(v)) if v.size else float("nan")
            finite = [m for m in means.values() if np.isfinite(m)]
            means["spread"] = float(max(finite) - min(finite)) if len(finite) > 1 else float("nan")
            by_mode[family] = means
    return {"interval_agreement": agreement, "metric_by_split_mode": by_mode}


def occlusion_curves(occ_df: pd.DataFrame) -> dict:
    """The final occlusion error curve PER CONSTRUCT (median absolute error by observed-fraction decile) for the
    report (A-09 / C.8.6, spec section 8.5). Keyed ``family -> construct -> {bin: median error}``; a producer that
    predates the per-construct schema (no ``construct`` column) still reports per family."""
    if not len(occ_df):
        return {}
    err = occ_df[occ_df["quantity"] == "occ_err"]
    has_construct = "construct" in err.columns
    group_cols = ["family", "construct"] if has_construct else ["family"]
    out: dict[str, dict] = {}
    for key, g in err.groupby(group_cols, observed=True):
        family = str(key[0] if has_construct else key)
        curve = {}
        for ob, gg in g.groupby("obs_bin", observed=True):
            v = gg["value"].to_numpy(dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size:
                curve[f"{_as_float(ob):.1f}"] = float(np.median(v))
        if has_construct:
            out.setdefault(family, {})[str(key[1])] = curve
        else:
            out[family] = curve
    return out


def _nanmean(df: pd.DataFrame, col: str) -> float:
    if not len(df) or col not in df.columns:
        return float("nan")
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


# =============================================================================== input contract (declare_inputs)
def input_contract() -> dict:
    """The declared symbols D3's numbers depend on (ADR-056); written into ``metrics.json`` beside provenance."""
    from scripts._input_contract import declare_inputs

    return declare_inputs(
        driver="validate_team_coordination",
        n_surrogates=D3_N_SURROGATES,
        h7_seed=thr.H7_SEED,
        stoppage_leg_min_s=thr.STOPPAGE_LEG_MIN_S,
        stoppage_modes=list(STOPPAGE_MODES),
        family_metrics={fam: list(v) for fam, v in _FAMILY_METRIC.items()},  # the stoppage-leg + occlusion families
        # A-09: the per-construct reliability grain (C.1) and its pre-registered power thresholds (C.2).
        construct_key_cols={t: list(v) for t, v in CONSTRUCT_KEY_COLS.items()},
        reliability_min_n_groups=thr.RELIABILITY_MIN_N_GROUPS,
        reliability_max_ci_halfwidth=thr.RELIABILITY_MAX_CI_HALFWIDTH,
        circular_reliability_min_rbar=thr.CIRCULAR_RELIABILITY_MIN_RBAR,
        gated_hypotheses=list(thr.GATED_HYPOTHESES),
        geometry_version=GEOMETRY_VERSION,
    )


# =============================================================================== corpus passes + reduce
def _timed(timer: StageTimer, work):
    """Wrap a ``for_each`` work function so per-match COMPUTE accumulates in the ``compute`` stage (R8), separate
    from the outer ``corpus`` stage (which also carries per-match load/download). ``load ~= corpus - compute``."""

    def wrapped(loaded):
        with timer("compute"):
            return work(loaded)

    return wrapped


def metrics_match(
    loaded,
    *,
    params_for: Callable[[str], CoordinationParams] = CoordinationParams.for_provider,
    timer: StageTimer | None = None,
) -> pd.DataFrame:
    """One match's final family tables + the H7 switch-event blocks (``n_surrogates=D3_N_SURROGATES``).

    ``params_for`` resolves the provider's params (:func:`params_resolver`); ``timer`` receives ``match_tables``'
    per-stage breakdown (R8); the tables never depend on it.
    """
    return match_tables(
        loaded,
        params_for(loaded.provider),
        n_surrogates=D3_N_SURROGATES,
        include_switch_events=True,
        timer=timer,
    )


def _pass_metrics(args, prov) -> CorpusPassResult:
    from scripts._driver import for_each
    from scripts._partition import worker_tag

    dest = pathlib.Path(args.out)
    tag = worker_tag(args.match_ids_json)
    params_for, params_src = params_resolver(args)
    timer = StageTimer()
    refs, load = corpus_source(args)
    visibility_preflight(refs, load)  # ADR-069 Layer 2: before any compute
    with timer("corpus"):
        res = for_each(
            refs,
            key=lambda r: r.key,
            load=load,
            work=_timed(timer, lambda loaded: metrics_match(loaded, params_for=params_for, timer=timer)),
            shard_root=dest / "metrics_shards",
            token_inputs={
                "pass": "metrics",
                "n_surrogates": D3_N_SURROGATES,
                "geometry_version": GEOMETRY_VERSION,
                "schema": _SHARD_SCHEMA_VERSION,
                "params": run_params_token(args, params_for),  # M-4: the values the work consumes
                **params_src,  # M-5: the artifact pair's digests in every token
            },
            tag=tag,
            label="match",
        )
    write_worker_partial(dest, "metrics", res, prov, timer, tag=tag, extra=params_src)
    return res


def stoppage_match(
    loaded,
    *,
    params_for: Callable[[str], CoordinationParams] = CoordinationParams.for_provider,
    timer: StageTimer | None = None,
) -> pd.DataFrame:
    """One match's tables under each of ``STOPPAGE_MODES`` (tagged ``stoppage_mode``) plus the event-vs-ball_state
    interval agreement row (the D20 leg). Runs at ``n_surrogates=0`` -- the leg measures splitting effect, not the
    surrogate baseline. ``timer`` receives ``match_tables``' per-stage breakdown (R8), summed over the modes."""
    params = params_for(loaded.provider)
    parts: list[pd.DataFrame] = []
    for mode in STOPPAGE_MODES:
        tbl = match_tables(loaded, params, n_surrogates=0, stoppage_evidence=mode, timer=timer)
        parts.append(tbl.assign(stoppage_mode=mode))
    ev = resolve_stoppages(
        loaded.frames,
        actions=loaded.actions,
        provider=loaded.provider,
        max_stoppage_s=thr.STOPPAGE_LEG_MIN_S,
        mode="events",
    ).intervals
    bs = resolve_stoppages(
        loaded.frames,
        actions=loaded.actions,
        provider=loaded.provider,
        max_stoppage_s=thr.STOPPAGE_LEG_MIN_S,
        mode="ball_state",
    ).intervals
    ag = stoppage_interval_agreement(ev, bs)
    parts.append(
        pd.DataFrame(
            [
                {
                    "provider": loaded.provider,
                    "match_id": str(loaded.match_id),
                    "table": "stoppage_agreement",
                    "stoppage_mode": "n/a",
                    **ag,
                }
            ]
        )
    )
    return pd.concat(parts, ignore_index=True)


def _stoppage_args(args):
    providers = tuple(p for p in args.providers if p in STOPPAGE_PROVIDERS)
    return argparse.Namespace(**{**vars(args), "providers": providers or STOPPAGE_PROVIDERS})


def _pass_stoppage(args, prov) -> CorpusPassResult:
    from scripts._driver import for_each
    from scripts._partition import worker_tag

    dest = pathlib.Path(args.out)
    tag = worker_tag(args.match_ids_json)
    params_for, params_src = params_resolver(args)
    stoppage_args = _stoppage_args(args)
    timer = StageTimer()
    refs, load = corpus_source(stoppage_args)
    visibility_preflight(refs, load)  # ADR-069 Layer 2: before any compute
    with timer("corpus"):
        res = for_each(
            refs,
            key=lambda r: r.key,
            load=load,
            work=_timed(timer, lambda loaded: stoppage_match(loaded, params_for=params_for, timer=timer)),
            shard_root=dest / "stoppage_shards",
            token_inputs={
                "pass": "stoppage",
                "modes": list(STOPPAGE_MODES),
                "min_s": thr.STOPPAGE_LEG_MIN_S,
                "geometry_version": GEOMETRY_VERSION,
                "schema": _SHARD_SCHEMA_VERSION,
                "params": run_params_token(args, params_for),  # M-4 (every provider, one generation per run)
                **params_src,  # M-5: the artifact pair's digests in every token
            },
            tag=tag,
            label="match",
        )
    write_worker_partial(dest, "stoppage", res, prov, timer, tag=tag, extra=params_src)
    return res


def _calibrated_width(args) -> float:
    """The FOV width W calibrated by D1, read from ``--derivation`` (its ``occlusion.width_m``)."""
    derivation, _sha = read_artifact(args, "derivation")
    width = derivation.get("occlusion", {}).get("width_m")
    if width is None:
        raise SystemExit(f"{args.derivation} has no occlusion.width_m (re-run D1 --pass occlusion + reduce)")
    return float(width)


def _occlusion_args(args):
    """``args`` restricted to the fully-observed occlusion providers (D1's OCCLUSION_PROVIDERS)."""
    providers = tuple(p for p in args.providers if p in OCCLUSION_PROVIDERS) or OCCLUSION_PROVIDERS
    return argparse.Namespace(**{**vars(args), "providers": providers})


def _pass_occlusion(args, prov) -> CorpusPassResult:
    from scripts._driver import for_each
    from scripts._partition import worker_tag

    dest = pathlib.Path(args.out)
    tag = worker_tag(args.match_ids_json)
    width_m = _calibrated_width(args)
    params_for, params_src = params_resolver(args)  # the FINAL params, like every other D3 pass (review A-10)
    occ_args = _occlusion_args(args)
    timer = StageTimer()
    refs, load = corpus_source(occ_args)
    visibility_preflight(refs, load)  # ADR-069 Layer 2: before any compute
    with timer("corpus"):
        res = for_each(
            refs,
            key=lambda r: r.key,
            load=load,
            work=_timed(timer, lambda loaded: occlusion_metrics(loaded, width_m, params=params_for(loaded.provider))),
            shard_root=dest / "occlusion_shards",
            token_inputs={
                "pass": "occlusion",
                "width_m": round(width_m, 4),
                "geometry_version": GEOMETRY_VERSION,
                "schema": _SHARD_SCHEMA_VERSION,
                "params": run_params_token(occ_args, params_for),  # M-4: the values the work consumes
                **params_src,  # M-5: the artifact pair's digests in every token
            },
            tag=tag,
            label="match",
        )
    write_worker_partial(dest, "occlusion", res, prov, timer, tag=tag, extra={"width_m": width_m, **params_src})
    return res


def _stage_timing(summaries: Mapping[str, dict]) -> dict:
    """Per pass, its summed ``stage_seconds`` (every worker) and per-match seconds -- the report's stage-timing summary
    against the spec 7.15 cost model. ``n_matches`` counts ATTEMPTED matches (a resumed skip did no work)."""
    out: dict[str, dict] = {}
    for pass_name, summary in summaries.items():
        stage_seconds = summary.get("stage_seconds", {})
        n = int(summary.get("n_attempted", 0) or 0)
        corpus_s = float(stage_seconds.get("corpus", 0.0))
        compute_s = float(stage_seconds.get("compute", 0.0))
        out[pass_name] = {
            "stage_seconds": stage_seconds,
            "n_matches": n,
            "seconds_per_match": (corpus_s / n) if n else float("nan"),  # incl. load/download
            "compute_seconds_per_match": (compute_s / n) if n else float("nan"),  # the spec 7.15 budget comparand
        }
    return out


def build_report(
    metrics_df: pd.DataFrame, stoppage_df: pd.DataFrame, occ_df: pd.DataFrame, prov, *, stage_timing: dict | None = None
) -> dict:
    """The full D3 report: hypotheses (recorded, never dropped), the per-construct reliability + poolability section
    (A-09), the stoppage leg, the occlusion curves, real-data liveness and the stage-timing summary (spec 8.5)."""
    tables = split_tables(metrics_df)
    hypotheses = evaluate_hypotheses(tables)
    tables_by_provider = (
        {str(p): split_tables(g) for p, g in metrics_df.groupby("provider", observed=True)} if len(metrics_df) else {}
    )
    constructs = build_constructs_report(tables_by_provider)
    return {
        "schema_version": _METRICS_SCHEMA_VERSION,
        "hypotheses": hypotheses,
        "gated_pass": gated_pass(hypotheses),
        # A-09: per-construct reliability (binding def + power verdict), cross-provider poolability per construct,
        # the two honesty lines, and the descriptive per-column summary for report.md.
        "constructs": constructs["constructs"],
        "column_summary": constructs["column_summary"],
        "honesty": constructs["honesty"],
        # A-35: H1-H7 are evaluated on the D2-gated population; the gate decides which params produced these tables.
        "d2_gate_population": (
            "H1-H7 and every construct cell are computed on the D2-gated parameter selection; a different gate "
            "outcome (Tier-B vs calibrated) would move the population these numbers are measured on."
        ),
        # A-35 (P-2): the ruled reconciliation between D2's selection objective and this report's reliability.
        "d2_objective_reconciliation": (
            "D2 selects parameters on held-out TEAM-DISCRIMINATION reliability (its calibration purpose), computed "
            "with the SAME per-construct estimator this report uses (circular reliability for circular-mean "
            "constructs, linear ICC(1) otherwise). Its grain (team discrimination across matches) differs from the "
            "binding reliability reported here (within-match split-half internal consistency); that difference is a "
            "ruled reconciliation, not a definition mismatch. D2's swept parameters affect only linear "
            "magnitude/fraction constructs, so no circular-mean construct enters D2's objective."
        ),
        "stoppage_leg": stoppage_leg(stoppage_df),
        "occlusion": occlusion_curves(occ_df),
        "liveness": liveness_block(tables),
        "stage_timing": stage_timing or {},
        "input_contract": input_contract(),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        "run_tree_hash": prov.get("tree_hash"),  # A-55
    }


def _render_report_md(report: dict) -> str:
    """A terse human-readable ``report.md`` from the metrics report (the owner reads this first)."""
    lines = ["# TF-58 team coordination -- D3 validation", ""]
    lines.append(f"Gated hypotheses pass: **{report['gated_pass']}** (H6 is descriptive).")
    lines.append("")
    lines.append("## Hypotheses (recorded findings; nothing dropped)")
    for h in ("H1", "H2", "H3", "H4", "H5", "H6", "H7"):
        res = report["hypotheses"].get(h, {})
        verdict = "descriptive" if res.get("pass") is None else ("PASS" if res.get("pass") else "FAIL")
        lines.append(f"- **{h}**: {verdict}")
    lines.append("")
    lines.append("## Reliability per column (DESCRIPTIVE: median + range across the column's constructs)")
    lines.append("")
    lines.append("> Not the column's reliability -- each column spans many constructs (A-09). An `unmeasurable`")
    lines.append("> cell is terminal and is never pooled up a level to rescue power.")
    lines.append("")
    summary = report.get("column_summary", {})
    for col in sorted(summary):
        s = summary[col]
        med = s["reliability_median"]
        med_str = f"{med:.2f}" if med is not None else "n/a"
        rng = (
            f"[{s['reliability_min']:.2f}, {s['reliability_max']:.2f}]" if s["reliability_min"] is not None else "[n/a]"
        )
        lines.append(
            f"- `{col}`: median {med_str} {rng} over {s['n_measured']}/{s['n_constructs']} measured constructs"
            f" ({s['n_unmeasurable']} unmeasurable)"
        )
    lines.append("")
    lines.append("## Honesty (bounds on what these reliabilities mean)")
    for line in report.get("honesty", []):
        lines.append(f"- {line}")
    lines.append("")
    lines.append(f"Dead metric columns: {report['liveness']['n_dead']}.")
    lines.append("")
    lines.append(f"Provenance: commit `{report['run_commit']}` (dirty={report['run_tree_dirty']}).")
    return "\n".join(lines) + "\n"


def _reduce(args, prov) -> None:
    dest = pathlib.Path(args.out)
    # B-1: every worker's share of each pass, proven complete against the listed corpus (or refused, spec 8.3); the
    # metrics + stoppage shares must also agree on WHICH params they computed with (one artifact pair, M-5).
    tables, summaries = {}, {}
    for name, providers, consistent in (
        ("metrics", args.providers, PARAMS_PROVENANCE_FIELDS),
        ("stoppage", _stoppage_args(args).providers, PARAMS_PROVENANCE_FIELDS),
        ("occlusion", _occlusion_args(args).providers, ("width_m", *PARAMS_PROVENANCE_FIELDS)),
    ):
        tables[name], summaries[name] = combine_workers(
            dest, name, expected=expected_corpus(args, providers), consistent=consistent, categorical=True
        )
    report = build_report(
        tables["metrics"], tables["stoppage"], tables["occlusion"], prov, stage_timing=_stage_timing(summaries)
    )
    report["population"] = {name: {k: v for k, v in s.items() if k != "stage_seconds"} for name, s in summaries.items()}
    pairs = table_pairs(tables["metrics"], tables["stoppage"], tables["occlusion"])
    report["corpus_visibility"] = corpus_visibility_label(pairs, token=getattr(args, "token", None))  # ADR-038
    (dest / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (dest / "report.md").write_text(_render_report_md(report), encoding="utf-8")
    print(json.dumps({"gated_pass": report["gated_pass"], "n_dead": report["liveness"]["n_dead"]}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pass", dest="which", required=True, choices=["metrics", "stoppage", "occlusion", "reduce"])
    add_params_args(ap)
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
    {"metrics": _pass_metrics, "stoppage": _pass_stoppage, "occlusion": _pass_occlusion, "reduce": _reduce}[args.which](
        args, prov
    )


if __name__ == "__main__":
    main()
