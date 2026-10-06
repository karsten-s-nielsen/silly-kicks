#!/usr/bin/env python
"""D2 (Tier C, spec 8.4): gated sensitivity calibration of the coordination Tier-B parameters.

A pre-registered, one-at-a-time (OAT) grid on ruthless ``GridSearchStrategy`` (0.7.0). Each swept parameter has a
short discrete set of levels RELATIVE to each provider's own Tier-B value (C26). Every layer computes with D1's
``derivation.json`` (``--derivation``, the artifact handoff -- owner ruling M-5, 2026-10-02): never the in-package
module, and the artifact's sha256 is in every shard token, manifest and the calibration artifact. The corpus layers
run per worker over a disjoint ``--match-ids-json`` slice; every step that reads the corpus-wide result COMBINES all
workers' shares first and refuses a partial corpus (``--corpus-json``, spec 8.3):

  --layer a --level baseline | <param>=<mult>
        the heavy corpus pass, one per PREPARATION level (7 + 5 - 1 shared baseline = 11): build signals and score
        every match with ``match_tables`` (n_surrogates=0). The baseline pass also carries every POST-preparation
        level as a reused-signals variant (spec 8.4). Writes this worker's share of the level.
  --layer b
        combines every worker's share of all 11 levels into the per-level tables, then the OAT selection on
        ruthless: ``CoordinationReliabilityObjective`` reads the deviating level's table, returns held-out
        team-discrimination reliability (ICC(1), match-CV), and ``select_recommended_point`` moves a parameter only
        when the gain clears the effect-size floor AND the paired SE (ADR-060). Writes ``<--out>/oat.json`` (the
        selections and the joint point).
  --layer joint
        the joint confirmation point's preparation pass (per worker), when the joint moves >= 2 parameters -- the
        combination is precomputed nowhere else (IMPL-04); a no-op otherwise.
  --layer confirm
        evaluate the JOINT selected point; the gate is ``Selection.moved AND gated_pass(evaluate_hypotheses(...))``.
        Writes ``<--out>/calibration.json`` and ``<--out>/_provider_params_generated.py`` =
        ``render_generated_params(derivation, calibration_moves(calibration))``: the moved selections applied to the
        pooled base AND every provider (A3) when the gate clears, else exactly D1's module (the Tier-B values stand,
        the fallback and its reason are recorded). Never the package: commit 2 copies both in.

Every level, baseline and point is a native Python ``float`` (spec 8.4). Reliability CV uses
``silly_kicks.calibration`` (`match_cv_splits`/`cv_standard_error`) and the ADR-060 selection
(`select_recommended_point`/`exceeds_noise_floor`). Requires ruthless-efficiency >= 0.7.0.

Shared corpus flags come from ``_coordination_corpus.add_common_args``: ``--out``, ``--token``, ``--max-matches``,
``--cache-dir``, ``--match-ids-json``, ``--corpus-json``, ``--providers``, ``--allow-dirty`` (dev-only; the artifact
is marked dirty) and ``--list-matches``. The tree is checked with ``require_clean_tree(git_provenance())`` FIRST
(ADR-037).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

# ruthless 0.7.0 re-exports these dynamically; pyright cannot see them, but they resolve at runtime (tested).
from ruthless import (
    Choice,
    Direction,
    GridConfig,  # pyright: ignore[reportAttributeAccessIssue]
    GridSearchStrategy,  # pyright: ignore[reportAttributeAccessIssue]
    InProcessBackend,
)
from ruthless.config import ParamSpec, StoreConfig
from ruthless.errors import FatalEvaluationError
from ruthless.result import Candidate

from scripts._coordination_corpus import (
    StageTimer,
    add_common_args,
    combine_workers,
    corpus_source,
    corpus_visibility_label,
    expected_corpus,
    match_tables,
    read_artifact,
    run_params_token,
    split_tables,
    table_pairs,
    visibility_preflight,
    write_worker_partial,
    write_worker_partial_by_variant,
)
from scripts._coordination_hypotheses import evaluate_hypotheses, gated_pass
from scripts._coordination_params_codegen import (
    apply_move,
    calibration_moves,
    params_from_artifacts,
    render_generated_params,
)
from scripts._provenance import dirty_suffix, store_path_for
from scripts._reliability import icc1
from silly_kicks.calibration import (
    MIN_EFFECT_SIZE,
    PointScore,
    Selection,
    cv_standard_error,
    match_cv_splits,
    select_recommended_point,
)
from silly_kicks.coordination import CoordinationParams
from silly_kicks.coordination._columns import CIRCULAR_MEAN_COLUMNS, COORD_METHOD_FAMILIES
from silly_kicks.coordination._kernels._circular import circular_reliability
from silly_kicks.tracking._geometry import GEOMETRY_VERSION

#: Swept parameters and their levels RELATIVE to each provider's Tier-B value (C26). Cutoff/gap/welch/vc are
#: multipliers; min_observed_fraction is an additive offset clipped to [0, 1]. Each set includes its baseline.
SWEEP: dict[str, tuple[float, ...]] = {
    "butterworth_cutoff_hz": (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0),
    "max_detection_gap_s": (0.5, 0.75, 1.0, 1.5, 2.0),
    "min_observed_fraction": (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3),
    "welch_segment_s": (0.5, 0.75, 1.0, 1.5, 2.0),
    "vc_epsilon": (0.5, 0.75, 1.0, 1.5, 2.0),
}
BASELINE: dict[str, float] = {
    "butterworth_cutoff_hz": 1.0,
    "max_detection_gap_s": 1.0,
    "min_observed_fraction": 0.0,
    "welch_segment_s": 1.0,
    "vc_epsilon": 1.0,
}
PREPARATION_PARAMS = ("butterworth_cutoff_hz", "max_detection_gap_s")
POST_PREPARATION_PARAMS = ("min_observed_fraction", "welch_segment_s", "vc_epsilon")
AFFECTED_FAMILIES: dict[str, tuple[str, ...]] = {
    "butterworth_cutoff_hz": COORD_METHOD_FAMILIES,
    "max_detection_gap_s": COORD_METHOD_FAMILIES,
    "min_observed_fraction": COORD_METHOD_FAMILIES,
    "welch_segment_s": ("coherence",),
    "vc_epsilon": ("vector_coding",),
}
#: Each family's primary team-discrimination metric columns + the melted ``table`` they live in and the team key.
#: Pairwise families (team_sync, rsi) have no single-team grouping, so they contribute no ICC column.
_FAMILY_COLUMNS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "relative_phase": ("pair", "team_a_id", ("coord_rp_resultant_length", "coord_rp_pct_near_in_phase")),
    "cross_correlation": ("pair", "team_a_id", ("coord_xc_max_abs_r",)),
    "vector_coding": ("pair", "team_a_id", ("coord_vc_pct_in_phase", "coord_vc_pct_anti_phase")),
    "coherence": ("pair", "team_a_id", ("coord_coh_band_mean",)),
    "spectral": ("spectral", "team_id", ("coord_median_freq_cpm",)),
    "cluster": ("cluster_team", "team_id", ("coord_rho_group_mean",)),
    "team_sync": ("team_sync", "", ()),
    "rsi": ("rsi", "", ()),
}
#: Bump when the reliability objective's ARITHMETIC changes in a way the structural inputs (``SWEEP``, ``BASELINE``,
#: ``_FAMILY_COLUMNS``, ``AFFECTED_FAMILIES``) do not capture -- the match-CV fold scheme (``match_cv_splits``) or the
#: provider-weighted per-construct estimator in :func:`_reliability_weighted_over_providers`. It rides in the C27
#: store id and the ``input_contract`` so an objective edit never resumes a store under the old definition (A-33).
OBJECTIVE_VERSION = "2"  # A-34 NaN-kept index-keyed folds + A-35 circular-aware estimator + m7 join-key CV
#: The generated-module file name the confirm writes beside calibration.json (commit 2 copies it into the package).
GENERATED_ARTIFACT = "_provider_params_generated.py"
_SHARD_SCHEMA_VERSION = "tf58-d2-3"  # tf58-d2-3: baseline share split per variant (ADR-112 follow-up, D2 layer-a OOM)
BASELINE_LEVEL = "baseline"
#: The worker-share name of the joint confirmation point's preparation pass.
JOINT_PREP = "joint_prep"


def levels_are_floats() -> bool:
    """True iff every SWEEP level, baseline and derived point is a native ``float`` (spec 8.4 level types)."""
    return all(type(v) is float for levels in SWEEP.values() for v in levels) and all(
        type(v) is float for v in BASELINE.values()
    )


def oat_param_space() -> dict[str, ParamSpec]:
    """The ruthless ``param_space`` for the OAT grid: one ``Choice`` of native-float levels per swept parameter."""
    return {p: Choice(kind="choice", choices=tuple(float(v) for v in levels)) for p, levels in SWEEP.items()}


def oat_config(store_path: str, objective_id: str) -> GridConfig:
    """The one-at-a-time ``GridConfig`` maximising held-out reliability around the Tier-B baseline (spec 8.4)."""
    return GridConfig(
        kind="grid",
        metric="reliability",
        direction=Direction.MAXIMIZE,
        design="one_at_a_time",
        baseline=dict(BASELINE),
        param_space=oat_param_space(),
        # objective_id is a required 0.7.0 StoreConfig field (Amendment A1); the PYRIGHT-specific ignore suppresses only
        # the reportCallIssue from ruthless-efficiency 0.7.0's StoreConfig stub (which omits objective_id) -- a bare
        # `# type: ignore[call-arg]` is a mypy code that pyright reads as suppress-ALL for the line, so it could hide an
        # unrelated error (review B m4; the deterministic dep pin + keep/remove is owner-ruled).
        store=StoreConfig(kind="sqlite", path=store_path, objective_id=objective_id),  # pyright: ignore[reportCallIssue]
    )


def variant_label(param: str, value: float) -> str:
    """The ``match_tables`` variant name for a post-preparation level, e.g. ``"welch_segment_s=1.5"``."""
    return f"{param}={float(value)}"


def deviating_param(params: Mapping[str, float]) -> tuple[str | None, float]:
    """The single parameter whose value differs from the OAT baseline (the deviating one), or (None, nan)."""
    for p, v in params.items():
        if not np.isclose(float(v), BASELINE[p]):
            return p, float(v)
    return None, float("nan")


def apply_level(base: CoordinationParams, param: str, value: float) -> CoordinationParams:
    """Apply one relative sweep level to a provider's Tier-B params (multiplier, or additive+clip for min_obs)."""
    # the codegen's own move function, so a scored level is exactly what the generated module renders (review A-15)
    if param == "min_observed_fraction":  # an additive offset (C26)
        return dataclasses.replace(base, **{param: apply_move(param, base.min_observed_fraction, offset=value)})
    return dataclasses.replace(base, **{param: apply_move(param, getattr(base, param), multiplier=value)})


# --------------------------------------------------------------------------- reliability objective (layer b)
def _affected_columns(families) -> list[tuple[str, str, str]]:
    """(table, team_key, column) triples with a usable team grouping for the affected families."""
    out: list[tuple[str, str, str]] = []
    for fam in families:
        table, team_key, cols = _FAMILY_COLUMNS[fam]
        if not team_key:
            continue
        out.extend((table, team_key, col) for col in cols)
    return out


def _join_keys(frame: pd.DataFrame) -> np.ndarray:
    """The provider-qualified match keys for CV folds (m7): a bare ``match_id`` collides across providers."""
    join = frame["provider"].astype(str) + "||" + frame["match_id"].astype(str)
    return np.array(sorted(pd.unique(join)))


def _reliability_weighted_over_providers(sub: pd.DataFrame, team_key: str, col: str) -> float:
    """Team-discrimination reliability of ``col`` pooled over provider strata, weighted by each stratum's match count.

    A-35: a circular-mean column uses rotation-invariant circular reliability (never a linear ICC on a circular
    mean); every other column uses linear ICC(1). (D2's swept params touch only linear columns today, but the branch
    keeps selection and the D3 report on the SAME per-construct estimator.)"""
    is_circular = col in CIRCULAR_MEAN_COLUMNS
    vals_list, weights = [], []
    for _prov, g in sub.groupby("provider", sort=True, observed=True):
        vals = g[col].to_numpy(dtype=np.float64)
        groups = g[team_key].to_numpy()
        finite = np.isfinite(vals)
        if finite.sum() < 2 or len(np.unique(groups[finite])) < 2:
            continue
        rel = (
            circular_reliability(vals[finite], groups[finite])[0] if is_circular else icc1(vals[finite], groups[finite])
        )
        if np.isfinite(rel):
            vals_list.append(rel)
            weights.append(float(g["match_id"].nunique()))
    if not vals_list:
        return float("nan")
    return float(np.average(vals_list, weights=weights))


def reliability_over_folds(frame: pd.DataFrame, families, splits) -> tuple[float, tuple[float, ...]]:
    """Mean held-out team-discrimination reliability over the affected families' columns, one value per match-CV fold
    (spec 8.4). A-34: the returned per-fold vector is keyed by fold INDEX -- an unscoreable fold contributes NaN
    (kept, never dropped) so the vectors align across candidates for the paired selection; the mean is NaN-aware."""
    cols = _affected_columns(families)
    keys = _join_keys(frame)
    join = frame["provider"].astype(str) + "||" + frame["match_id"].astype(str)
    fold_vals: list[float] = []
    for _train, test in splits:
        test_keys = set(keys[test])
        sub = frame[join.isin(test_keys)]
        col_rels = []
        for table, team_key, col in cols:
            t = sub[sub["table"] == table]
            if col not in t.columns or not len(t):
                continue
            rel = _reliability_weighted_over_providers(t, team_key, col)
            if np.isfinite(rel):
                col_rels.append(rel)
        fold_vals.append(float(np.mean(col_rels)) if col_rels else float("nan"))
    finite = [v for v in fold_vals if np.isfinite(v)]
    mean = float(np.mean(finite)) if finite else float("nan")
    return mean, tuple(fold_vals)


class CoordinationReliabilityObjective:
    """A ruthless ``Objective``: score an OAT candidate by held-out team-discrimination reliability (spec 8.4).

    Reads the combined layer-a table for the candidate's single deviating level (a preparation level is its own
    file; a post-preparation level is a variant inside the baseline file). A missing table raises
    ``FatalEvaluationError`` -- the sweep never runs on a partial corpus (spec 8.4).
    """

    def __init__(self, out: pathlib.Path) -> None:
        self._out = pathlib.Path(out)

    def _frame_for(self, param: str | None, value: float) -> tuple[pd.DataFrame, tuple[str, ...]]:
        if param is None:
            return self._read(level_combined_path(self._out, BASELINE_LEVEL, "base"), "base"), COORD_METHOD_FAMILIES
        if param in PREPARATION_PARAMS:
            return self._read(level_combined_path(self._out, _level_key(param, value)), "base"), AFFECTED_FAMILIES[
                param
            ]
        return self._read(
            level_combined_path(self._out, BASELINE_LEVEL, variant_label(param, value)), variant_label(param, value)
        ), AFFECTED_FAMILIES[param]

    def _read(self, path: pathlib.Path, variant: str) -> pd.DataFrame:
        if not path.is_file():
            raise FatalEvaluationError(f"layer-a shards missing for {path.name}; run --layer a first (spec 8.4)")
        frame = pd.read_parquet(path)
        return frame[frame["variant"] == variant]

    def evaluate(self, candidate: Candidate) -> dict[str, float]:
        param, value = deviating_param(dict(candidate.params))
        frame, families = self._frame_for(param, value)
        keys = _join_keys(frame)  # m7: provider-qualified, so a shared match_id never collides across providers
        if keys.size == 0:
            raise FatalEvaluationError(f"layer-a shards for {param}={value} carry no matches (spec 8.4)")
        mean, folds = reliability_over_folds(frame, families, match_cv_splits(keys))
        finite_folds = [v for v in folds if np.isfinite(v)]  # A-34: NaN folds are kept in the vector, dropped for SE
        se = cv_standard_error(finite_folds) if finite_folds else float("nan")
        metrics = {"reliability": mean, "reliability_se": se}
        metrics.update({f"fold_{i:02d}": v for i, v in enumerate(folds)})
        return metrics


# --------------------------------------------------------------------------- selection (layer b) + confirm
def _point_scores(history) -> dict[tuple[str, float], PointScore]:
    """Each evaluated OAT candidate as a ``PointScore`` keyed by its (deviating param, value); baseline is ("", 0.0)."""
    out: dict[tuple[str, float], PointScore] = {}
    for ev in history:
        params = dict(ev.candidate.params)
        param, value = deviating_param(params)
        folds = tuple(v for k, v in sorted(ev.metrics.items()) if k.startswith("fold_"))
        key = (param or "", value if param else 0.0)
        out[key] = PointScore(
            label=f"{param}={value}" if param else BASELINE_LEVEL,
            params=params,
            per_fold=folds,
            mean=ev.metrics["reliability"],
        )
    return out


def select_per_parameter(history) -> dict[str, Selection]:
    """Per swept parameter, ``select_recommended_point`` of its off-baseline levels against the shared baseline."""
    scores = _point_scores(history)
    baseline = scores[("", 0.0)]
    selections: dict[str, Selection] = {}
    for param, levels in SWEEP.items():
        candidates = [
            scores[(param, float(v))]
            for v in levels
            if not np.isclose(v, BASELINE[param]) and (param, float(v)) in scores
        ]
        if candidates:
            selections[param] = select_recommended_point(
                incumbent=baseline, candidates=candidates, min_effect_size=MIN_EFFECT_SIZE
            )
    return selections


def joint_point(selections: Mapping[str, Selection]) -> dict[str, float]:
    """The baseline with every MOVED parameter set to its selected level (spec 8.4 confirmation point)."""
    point = dict(BASELINE)
    for param, sel in selections.items():
        if sel.moved:
            point[param] = float(sel.selected.params[param])
    return point


def calibration_multipliers(selections: Mapping[str, Selection]) -> dict[str, dict[str, float]]:
    """The MOVED selections as codegen ``{param: {multiplier|offset}}`` (min_observed is an offset; others multiply)."""
    out: dict[str, dict[str, float]] = {}
    for param, sel in selections.items():
        if not sel.moved:
            continue
        value = float(sel.selected.params[param])
        out[param] = {"offset": value} if param == "min_observed_fraction" else {"multiplier": value}
    return out


# --------------------------------------------------------------------------- levels + C27 objective identity
def _level_key(param: str, value: float) -> str:
    return f"{param}={float(value)}"


def preparation_levels() -> tuple[str, ...]:
    """The layer-a passes: the baseline + every off-baseline PREPARATION level (7 + 5 - 1 = 11, plan Task 22)."""
    return (
        BASELINE_LEVEL,
        *(_level_key(p, v) for p in PREPARATION_PARAMS for v in SWEEP[p] if not np.isclose(v, BASELINE[p])),
    )


def _parse_level(level: str) -> tuple[str, float]:
    """``<param>=<value>`` as the off-baseline PREPARATION level it names (the grid's own float), or refuse."""
    param, _, text = level.partition("=")
    if param not in PREPARATION_PARAMS:
        raise SystemExit(
            f"--level {level!r}: layer a runs 'baseline' or a preparation level of {list(PREPARATION_PARAMS)}; "
            "post-preparation levels are variants of the baseline pass (spec 8.4)"
        )
    try:
        value = float(text)
    except ValueError:
        raise SystemExit(f"--level {level!r}: {text!r} is not a number") from None
    on_grid = [v for v in SWEEP[param] if np.isclose(v, value) and not np.isclose(v, BASELINE[param])]
    if not on_grid:
        raise SystemExit(f"--level {level!r}: {value} is not an off-baseline level of {param} {SWEEP[param]}")
    return param, float(on_grid[0])


def canonical_level(level: str) -> str:
    """``--level`` as its canonical key: ``baseline`` or ``<param>=<grid float>`` (so ``0.50`` and ``0.5`` agree)."""
    return level if level == BASELINE_LEVEL else _level_key(*_parse_level(level))


def level_share_name(level_key: str, variant: str | None = None) -> str:
    """The worker-share name of a layer-a level (``baseline``, ``<param>=<value>`` or ``joint``). With ``variant``,
    the per-variant baseline share (ADR-112 follow-up: the baseline pass writes one share per post-prep variant
    instead of one stacked share, to bound the D2 layer-a memory)."""
    name = "layer_a__" + level_key.replace("=", "__").replace(".", "p")
    if variant is not None:
        name += "__v__" + variant.replace("=", "__").replace(".", "p")
    return name


def level_combined_path(out: pathlib.Path, level_key: str, variant: str | None = None) -> pathlib.Path:
    """The per-level (per-variant for the baseline) combined layer-a metric table a reader reads (every worker's
    share, combined). ``variant`` selects the baseline's per-variant file."""
    return pathlib.Path(out) / f"{level_share_name(level_key, variant)}.parquet"


def _objective_fingerprint() -> str:
    """A stable digest of the reliability objective's DEFINITION: which columns it reads (``_FAMILY_COLUMNS``), the
    swept families (``AFFECTED_FAMILIES``), and ``OBJECTIVE_VERSION`` (the fold scheme + ICC weighting, bumped by hand).
    Folded into the C27 store id and the ``input_contract`` so an edit to the objective -- not just to the shards it
    reads -- moves the identity and never resumes stale scores (review A-33)."""
    payload = json.dumps(
        {
            "family_columns": {k: [v[0], v[1], list(v[2])] for k, v in _FAMILY_COLUMNS.items()},
            "affected_families": {k: list(v) for k, v in AFFECTED_FAMILIES.items()},
            "version": OBJECTIVE_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def objective_id_for(generation_tokens, *, prov: Mapping[str, object] | None = None) -> str:
    """C27: the objective's identity is the digest of the generation tokens it reads (spec 8.4) AND of the objective's
    own DEFINITION (:func:`_objective_fingerprint`, review A-33); a regenerated shard set OR an objective edit changes
    it and ruthless refuses to resume from stale scores. On a tree that is not clean the D21 dirty nonce is appended
    (C4: one identity rule for every store), so :func:`store_path_for` opens a fresh store."""
    joined = "|".join([*sorted(generation_tokens), "objective:" + _objective_fingerprint()])
    oid = "calibrate_coordination:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    return oid + (dirty_suffix(prov) if prov is not None else "")


def generation_tokens(summaries: Mapping[str, Mapping]) -> list[str]:
    """The C27 tokens of the combined passes the objective reads: ``<pass>:<shard generation>:<population digest>``
    -- the generation directory its shards came from (M-3: read, not assumed) AND the exact corpus they cover, so a
    same-generation shard set grown by more matches is a different input too."""
    return [f"{name}:{s['generation']}:{s['population_digest']}" for name, s in sorted(summaries.items())]


def _combine_baseline_variants(dest: pathlib.Path, expected) -> dict:
    """Combine the baseline's per-variant shares (ADR-112 follow-up) into per-variant combined tables, and FOLD the
    per-variant combine summaries into the ONE baseline-level summary (D2-SPEC-05).

    The per-variant shares come from one layer-a pass (one shard ``generation``) over the identical match set (one
    ``population_digest``); only the summary's name-derived ``pass`` field differs. So the fold asserts generation +
    population_digest agree across variants (else refuse -- a genuine cross-run corruption), then returns the common
    summary with ``pass`` set to the baseline level's canonical share name. ``generation_tokens`` /
    ``objective_id_for`` / ``_population`` (keyed by the level) are therefore UNCHANGED vs the former single stacked
    combine -> the whole ``calibration.json`` stays byte-identical."""
    from scripts._partition import write_table_atomically

    per_variant: list[dict] = []
    for variant in sorted(_post_preparation_variants(CoordinationParams())):
        table, summary = combine_workers(
            dest,
            level_share_name(BASELINE_LEVEL, variant),
            expected=expected,
            consistent=("derivation_sha256",),
            categorical=True,
        )
        write_table_atomically(table, level_combined_path(dest, BASELINE_LEVEL, variant), tag="combine")
        per_variant.append(summary)
    return _fold_baseline_summary(per_variant)


def _fold_baseline_summary(per_variant: list[dict]) -> dict:
    """Fold the baseline's per-variant combine summaries to the ONE baseline-level summary (D2-SPEC-05). The variants
    come from one layer-a pass over one match set, so they MUST agree on ``generation`` + ``population_digest`` (else
    a genuine cross-run corruption -> refuse); only the name-derived ``pass`` differs. Return the common summary with
    ``pass`` reset to the baseline level's canonical share name, so ``generation_tokens`` / ``objective_id_for`` /
    ``_population`` are byte-identical to the former single stacked combine."""
    if not per_variant:
        raise SystemExit("baseline has no post-preparation variants to combine (spec 8.4)")
    keys = {(s["generation"], s["population_digest"]) for s in per_variant}
    if len(keys) != 1:
        raise SystemExit(
            f"baseline per-variant combines disagree on (generation, population_digest): {sorted(keys)} "
            "-- re-run --layer a baseline in a fresh --out"
        )
    # SUM stage_seconds across variants so the baseline-level timing reflects ALL the per-variant combines, not just
    # the first (IMPL-04; stage_seconds is a volatile stripped from calibration.json, so this does not change the
    # artifact -- generation/population_digest/pass, which DO feed it, are kept from the agreed common value).
    stage_seconds: dict[str, float] = {}
    for summary in per_variant:
        for stage, seconds in (summary.get("stage_seconds") or {}).items():
            stage_seconds[stage] = stage_seconds.get(stage, 0.0) + float(seconds)
    return {**per_variant[0], "pass": level_share_name(BASELINE_LEVEL), "stage_seconds": stage_seconds}


def combine_levels(args) -> tuple[dict[str, dict], str]:
    """Every worker's share of every layer-a level, proven complete against the corpus (B-1, spec 8.3), written as
    the per-level tables the objective reads. Refuses levels computed from different derivations (M-5). Returns
    ``({level: combine summary}, derivation sha256)``."""
    from scripts._partition import write_table_atomically

    dest = pathlib.Path(args.out)
    expected = expected_corpus(args, args.providers)
    summaries: dict[str, dict] = {}
    for level in preparation_levels():
        if level == BASELINE_LEVEL:
            summaries[level] = _combine_baseline_variants(dest, expected)
            continue
        table, summary = combine_workers(
            dest, level_share_name(level), expected=expected, consistent=("derivation_sha256",), categorical=True
        )
        write_table_atomically(table, level_combined_path(dest, level), tag="combine")
        summaries[level] = summary
    shas = {s["consistent"]["derivation_sha256"] for s in summaries.values()}
    if len(shas) != 1:
        raise SystemExit(
            f"the layer-a levels were computed from different derivations {sorted(shas)}: re-run layer a with one "
            "--derivation (the artifact handoff, M-5)"
        )
    return summaries, next(iter(shas))


def _scored_pairs(out: pathlib.Path, joint_summary: Mapping | None) -> set[tuple[str, str]]:
    """The (provider, match) pairs every combined table the confirm read holds -- the calibration's population."""
    keys = [*preparation_levels(), *(["joint"] if joint_summary is not None else [])]
    # the baseline is stored per variant (population is variant-invariant -- read its "base" file); others unchanged.
    paths = [level_combined_path(out, k, "base") if k == BASELINE_LEVEL else level_combined_path(out, k) for k in keys]
    tables = [pd.read_parquet(p, columns=["provider", "match_id"]) for p in paths]
    return table_pairs(*tables)


def _population(summaries: Mapping[str, Mapping]) -> dict[str, dict]:
    return {name: {k: v for k, v in s.items() if k != "stage_seconds"} for name, s in summaries.items()}


# --------------------------------------------------------------------------- layers
def _artifact_base(derivation: Mapping) -> Callable[[str], CoordinationParams]:
    """Each provider's Tier-B params from the D1 artifact -- exactly what commit 2's module will hold (M-5)."""
    return lambda provider: params_from_artifacts(provider, derivation, None)


def _post_preparation_variants(base: CoordinationParams) -> dict[str, CoordinationParams]:
    """Every off-baseline post-preparation level as a reused-signals ``match_tables`` variant (spec 8.4)."""
    variants = {"base": base}
    for param in POST_PREPARATION_PARAMS:
        for value in SWEEP[param]:
            if np.isclose(value, BASELINE[param]):
                continue
            variants[variant_label(param, value)] = apply_level(base, param, float(value))
    return variants


def _layer_a(args, prov):
    from scripts._driver import for_each
    from scripts._partition import worker_tag

    dest = pathlib.Path(args.out)
    level = canonical_level(args.level)
    derivation, derivation_sha = read_artifact(args, "derivation")
    base_for = _artifact_base(derivation)
    tag = worker_tag(args.match_ids_json)
    timer = StageTimer()

    def work(loaded):
        base = base_for(loaded.provider)
        if level == BASELINE_LEVEL:
            variants = _post_preparation_variants(base)
            return match_tables(
                loaded, base, n_surrogates=0, variants=variants, include_switch_events=True, timer=timer
            )
        prep = apply_level(base, *_parse_level(level))
        return match_tables(loaded, prep, n_surrogates=0, include_switch_events=True, timer=timer)

    variants = sorted(_post_preparation_variants(CoordinationParams())) if level == BASELINE_LEVEL else ["base"]
    refs, load = corpus_source(args)
    visibility_preflight(refs, load)  # ADR-069 Layer 2: before any compute
    with timer("corpus"):
        res = for_each(
            refs,
            key=lambda r: r.key,
            load=load,
            work=work,
            shard_root=dest / f"a_shards__{level_share_name(level)}",
            token_inputs={
                "layer": "a",
                "level": level,
                "variants": variants,
                "derivation_sha256": derivation_sha,  # M-5: the artifact the work computes with
                "params": run_params_token(args, base_for),  # M-4: the values it consumes (every provider)
                "input_contract": input_contract()["digest"],
                "geometry_version": GEOMETRY_VERSION,
                "schema": _SHARD_SCHEMA_VERSION,
            },
            tag=tag,
            label="match",
        )
    extra = {"level": level, "derivation_sha256": derivation_sha}
    if level == BASELINE_LEVEL:
        # per-variant shares (ADR-112 follow-up): one share per post-prep variant, materialised one at a time, so the
        # worker never concatenates the ~13x stacked melt (the D2 baseline OOM). `variants` is the melt's variant set.
        write_worker_partial_by_variant(
            dest,
            lambda v: level_share_name(BASELINE_LEVEL, v),
            res,
            prov,
            timer,
            tag=tag,
            variants=variants,
            extra=extra,
        )
    else:
        write_worker_partial(dest, level_share_name(level), res, prov, timer, tag=tag, extra=extra)
    return res


def _run_oat(out: pathlib.Path, summaries: Mapping[str, Mapping], prov) -> tuple[dict[str, Selection], list, str]:
    """Run the OAT grid on ruthless over the combined level tables and select per parameter.

    Returns ``(selections, history, objective_id)``."""
    oid = objective_id_for(generation_tokens(summaries), prov=prov)
    config = oat_config(store_path_for(out / "grid_oat.db", oid), oid)
    result = GridSearchStrategy(config).run(CoordinationReliabilityObjective(out), backend=InProcessBackend())
    return select_per_parameter(result.history), result.history, oid


def needs_joint_pass(joint: Mapping[str, float]) -> bool:
    """True when the joint moves >= 2 parameters (any mix): only a single deviation is precomputed by layer a."""
    return sum(not np.isclose(joint[p], BASELINE[p]) for p in SWEEP) >= 2


def _layer_b(args, prov) -> None:
    dest = pathlib.Path(args.out)
    summaries, derivation_sha = combine_levels(args)
    selections, _history, oat_id = _run_oat(dest, summaries, prov)
    joint = joint_point(selections)
    oat = {
        "objective_id": oat_id,
        "generation_tokens": generation_tokens(summaries),
        "derivation_sha256": derivation_sha,
        "selections": {p: _selection_dict(s) for p, s in selections.items()},
        "joint_point": joint,
        "needs_joint_pass": needs_joint_pass(joint),
        "population": _population(summaries),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
    }
    (dest / "oat.json").write_text(json.dumps(oat, indent=2, default=str), encoding="utf-8")
    print(json.dumps({p: {"moved": s.moved, "reason": s.reason} for p, s in selections.items()}, indent=2))


def _layer_joint(args, prov):
    """This worker's share of the joint point's preparation pass (IMPL-04), from ``oat.json``; ``None`` (nothing to
    run) when the joint moves <= 1 parameter -- that deviation is a layer-a level already."""
    from scripts._driver import for_each
    from scripts._partition import worker_tag

    dest = pathlib.Path(args.out)
    oat_path = dest / "oat.json"
    if not oat_path.is_file():
        raise SystemExit(f"--layer joint reads the OAT selection {oat_path}: run --layer b first")
    oat = json.loads(oat_path.read_text(encoding="utf-8"))
    joint = {k: float(v) for k, v in oat["joint_point"].items()}
    if not needs_joint_pass(joint):
        print(json.dumps({"joint_pass": "not needed: the joint is a precomputed layer-a level", "joint": joint}))
        return None
    derivation, derivation_sha = read_artifact(args, "derivation")
    if derivation_sha != oat["derivation_sha256"]:
        raise SystemExit(
            f"--derivation {args.derivation} is a different derivation from the one the OAT selected the joint "
            f"from ({oat['derivation_sha256'][:12]}...): pass the same artifact (M-5)"
        )
    base_for = _artifact_base(derivation)
    tag = worker_tag(args.match_ids_json)
    timer = StageTimer()

    def work(loaded):
        prep = base_for(loaded.provider)
        for p in PREPARATION_PARAMS:
            if not np.isclose(joint[p], BASELINE[p]):
                prep = apply_level(prep, p, joint[p])
        full = prep
        for p in POST_PREPARATION_PARAMS:
            if not np.isclose(joint[p], BASELINE[p]):
                full = apply_level(full, p, joint[p])
        return match_tables(
            loaded, prep, n_surrogates=0, variants={"joint": full}, include_switch_events=True, timer=timer
        )

    refs, load = corpus_source(args)
    visibility_preflight(refs, load)  # ADR-069 Layer 2: before any compute
    with timer("corpus"):
        res = for_each(
            refs,
            key=lambda r: r.key,
            load=load,
            work=work,
            shard_root=dest / "joint_prep_shards",
            token_inputs={
                "layer": "joint",
                "joint": dict(sorted(joint.items())),
                "derivation_sha256": derivation_sha,
                "params": run_params_token(args, base_for),
                "input_contract": input_contract()["digest"],
                "geometry_version": GEOMETRY_VERSION,
                "schema": _SHARD_SCHEMA_VERSION,
            },
            tag=tag,
            label="match",
        )
    extra = {"joint": joint, "derivation_sha256": derivation_sha}
    write_worker_partial(dest, JOINT_PREP, res, prov, timer, tag=tag, extra=extra)
    return res


def _confirm(args, prov) -> None:
    dest = pathlib.Path(args.out)
    derivation, derivation_sha = read_artifact(args, "derivation")
    summaries, levels_sha = combine_levels(args)
    if levels_sha != derivation_sha:
        raise SystemExit(
            f"--derivation {args.derivation} is a different derivation from the one layer a computed with "
            f"({levels_sha[:12]}...): pass the same artifact (M-5)"
        )
    selections, history, oat_id = _run_oat(dest, summaries, prov)
    joint = joint_point(selections)
    moved = {p: s for p, s in selections.items() if s.moved}
    # Confirm the joint point: it must beat the baseline (moved) AND clear the hypotheses (spec 8.5).
    baseline_score = _point_scores(history)[("", 0.0)]
    joint_frame, _variant, joint_summary = _joint_source(dest, joint, args=args)
    read = {**summaries, **({JOINT_PREP: joint_summary} if joint_summary is not None else {})}
    confirm_id = objective_id_for([*generation_tokens(read), "confirm"], prov=prov)
    joint_moved = False
    if moved and len(joint_frame):
        joint_score = _joint_point_score(joint, joint_frame, dest, confirm_id)
        joint_sel = select_recommended_point(
            incumbent=baseline_score, candidates=[joint_score], min_effect_size=MIN_EFFECT_SIZE
        )
        joint_moved = joint_sel.moved
    hypotheses = evaluate_hypotheses(split_tables(joint_frame))  # the pre-registered H7 seed (A-41)
    gate = _confirm_gate(joint_moved, hypotheses)
    calibration = {
        "selections": {p: _selection_dict(s) for p, s in selections.items()},
        "joint_point": joint,
        "confirmation": {"moved": joint_moved, "hypotheses_pass": gated_pass(hypotheses), "gate_cleared": gate},
        "hypotheses": dict(hypotheses),
        "objective_ids": {"oat": oat_id, "confirm": confirm_id},
        "moved_multipliers": calibration_multipliers(selections),
        "fallback_reason": _fallback_reason(bool(moved), joint_moved, hypotheses, gate),
        "derivation_sha256": derivation_sha,
        "population": _population(read),
        "stage_seconds": {name: s["stage_seconds"] for name, s in read.items()},  # C14 / Ruling C(a) timers
        "corpus_visibility": corpus_visibility_label(_scored_pairs(dest, joint_summary), token=args.token),  # ADR-038
        "input_contract": input_contract(),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
    }
    # M-5 artifact handoff: both outputs land in --out, never the package or docs/research (commit 2 copies them in,
    # where the codegen test pins the module to derivation.json + calibration.json).
    (dest / "calibration.json").write_text(json.dumps(calibration, indent=2, default=str), encoding="utf-8")
    module = render_generated_params(derivation, calibration_moves(calibration))
    (dest / GENERATED_ARTIFACT).write_text(module, encoding="utf-8")
    print(json.dumps({"gate_cleared": gate, "joint_point": joint}, indent=2, default=str))


class _JointObjective:
    """Scores ONE pre-built joint frame (the confirmation point), over every family, for the ruthless points-grid.

    Unlike the OAT objective it does not map a candidate to a level -- the joint may move several parameters at once,
    so its tables are supplied directly (from :func:`_joint_source`)."""

    def __init__(self, joint_frame: pd.DataFrame) -> None:
        self._frame = joint_frame

    def evaluate(self, candidate: Candidate) -> dict[str, float]:
        keys = _join_keys(self._frame) if len(self._frame) else np.array([])  # m7: provider-qualified CV keys
        if keys.size == 0:
            raise FatalEvaluationError("joint confirmation frame carries no matches (spec 8.4)")
        mean, folds = reliability_over_folds(self._frame, COORD_METHOD_FAMILIES, match_cv_splits(keys))
        finite_folds = [v for v in folds if np.isfinite(v)]  # A-34: NaN folds kept in the vector, dropped for SE
        se = cv_standard_error(finite_folds) if finite_folds else float("nan")
        metrics = {"reliability": mean, "reliability_se": se}
        metrics.update({f"fold_{i:02d}": v for i, v in enumerate(folds)})
        return metrics


def _joint_source(out: pathlib.Path, joint: Mapping[str, float], *, args=None) -> tuple[pd.DataFrame, str, dict | None]:
    """The joint confirmation point's combined table, its variant, and the joint pass's combine summary (or None).

    0 moved -> the baseline file's ``base`` variant; 1 moved post-prep -> its baseline-file variant; 1 moved prep ->
    that preparation level's file (``base``); >= 2 moved (any mix) -> every worker's share of the ``--layer joint``
    pass, combined and proven complete (B-1), refused unless every share was computed for THIS joint.
    """
    from scripts._partition import write_table_atomically

    moved_prep = [p for p in PREPARATION_PARAMS if not np.isclose(joint[p], BASELINE[p])]
    moved_post = [p for p in POST_PREPARATION_PARAMS if not np.isclose(joint[p], BASELINE[p])]
    if not needs_joint_pass(joint):
        if moved_prep:
            path, variant = level_combined_path(out, _level_key(moved_prep[0], joint[moved_prep[0]])), "base"
        elif moved_post:
            variant = variant_label(moved_post[0], joint[moved_post[0]])
            path = level_combined_path(out, BASELINE_LEVEL, variant)
        else:
            path, variant = level_combined_path(out, BASELINE_LEVEL, "base"), "base"
        frame = pd.read_parquet(path) if path.is_file() else pd.DataFrame()
        return (frame[frame["variant"] == variant] if len(frame) else frame), variant, None
    if args is None:
        raise SystemExit("a joint that moves >= 2 parameters needs a corpus to prove its joint pass complete")
    table, summary = combine_workers(
        out,
        JOINT_PREP,
        expected=expected_corpus(args, args.providers),
        consistent=("joint", "derivation_sha256"),
        categorical=True,
    )
    computed_for = {k: float(v) for k, v in summary["consistent"]["joint"].items()}
    if computed_for != {k: float(v) for k, v in joint.items()}:
        raise SystemExit(
            f"the joint-prep shares were computed for another joint {computed_for} (this confirm selects {dict(joint)})"
            ": re-run --layer b, then --layer joint, in a fresh --out"
        )
    write_table_atomically(table, level_combined_path(out, "joint"), tag="combine")
    return (table[table["variant"] == "joint"] if len(table) else table), "joint", summary


def _joint_point_score(
    joint: Mapping[str, float], joint_frame: pd.DataFrame, out: pathlib.Path, objective_id: str
) -> PointScore:
    """Score the joint point via a one-point ruthless grid over its own store (spec 8.4), reading the joint frame."""
    config = GridConfig(
        kind="grid",
        metric="reliability",
        direction=Direction.MAXIMIZE,
        design="points",
        param_space=oat_param_space(),
        points=[dict(joint)],
        store=StoreConfig(  # objective_id stub lag (ruthless-efficiency 0.7.0); see the first StoreConfig above
            kind="sqlite",
            path=store_path_for(out / "grid_confirm.db", objective_id),
            objective_id=objective_id,  # pyright: ignore[reportCallIssue]  # B m4: pyright-specific, not suppress-all
        ),
    )
    result = GridSearchStrategy(config).run(_JointObjective(joint_frame), backend=InProcessBackend())
    ev = result.history[0]
    folds = tuple(v for k, v in sorted(ev.metrics.items()) if k.startswith("fold_"))
    return PointScore(label="joint", params=dict(joint), per_fold=folds, mean=ev.metrics["reliability"])


def _confirm_gate(joint_moved: bool, hypotheses: Mapping[str, dict]) -> bool:
    """Spec 8.4 confirmation gate: the joint point beats the baseline (moved) AND clears the hypotheses (spec 8.5)."""
    return bool(joint_moved and gated_pass(hypotheses))


def _fallback_reason(
    any_level_moved: bool, joint_moved: bool, hypotheses: Mapping[str, dict], gate: bool
) -> str | None:
    """Why Tier-B stands when the gate did not clear -- the ACTUAL cause, not a blanket "gate not cleared" (review
    A-56): no OAT level improved at all (nothing to confirm), the joint candidate did not beat the baseline, or the
    hypotheses gate did not pass. ``None`` when the gate cleared."""
    if gate:
        return None
    if not any_level_moved:
        return "no OAT level improved the objective; Tier-B values stand"
    if not joint_moved:
        return "the joint candidate did not beat the baseline; Tier-B values stand"
    return "the hypotheses gate did not pass; Tier-B values stand"


def _selection_dict(sel: Selection) -> dict:
    """Serialise one ADR-060 ``Selection`` for ``calibration.json`` (generic over the swept parameter; the shared
    ``build_selection_artifact`` is xt-bandwidth-specific -- it hardcodes ``beta``/``gamma`` -- so not reused here)."""
    return {
        "moved": bool(sel.moved),
        "reason": sel.reason,
        "effect_size": sel.effect_size,
        "paired_se": sel.paired_se,
        "selected": {"label": sel.selected.label, "params": dict(sel.selected.params), "mean": sel.selected.mean},
        "incumbent": {"label": sel.incumbent.label, "mean": sel.incumbent.mean},
    }


# --------------------------------------------------------------------------- provenance / input contract
def input_contract() -> dict:
    from scripts._input_contract import declare_inputs

    return declare_inputs(
        driver="calibrate_coordination",
        sweep={k: list(v) for k, v in SWEEP.items()},
        baseline=dict(BASELINE),
        affected_families={k: list(v) for k, v in AFFECTED_FAMILIES.items()},
        # A-33: the objective DEFINITION, not only the swept params -- which columns define reliability and the
        # arithmetic version, so an objective edit is a different input contract.
        family_columns={k: [v[0], v[1], list(v[2])] for k, v in _FAMILY_COLUMNS.items()},
        objective_version=OBJECTIVE_VERSION,
        min_effect_size=MIN_EFFECT_SIZE,
        geometry_version=GEOMETRY_VERSION,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", required=True, choices=["a", "b", "joint", "confirm"])
    ap.add_argument("--level", default=BASELINE_LEVEL, help="layer a only: 'baseline' or '<param>=<multiplier>'")
    ap.add_argument(
        "--derivation", default=None, help="D1's derivation.json (layers a, joint, confirm; owner ruling M-5)"
    )
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
    {"a": _layer_a, "b": _layer_b, "joint": _layer_joint, "confirm": _confirm}[args.layer](args, prov)


if __name__ == "__main__":
    main()
