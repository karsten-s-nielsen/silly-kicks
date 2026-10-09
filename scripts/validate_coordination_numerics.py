#!/usr/bin/env python
"""ADR-111 corpus no-flip gate: the production coordination numerics against the as-built reference numerics.

The owner's HARD GATE for the pre-authorised numerics changes (2026-09-28) -- the D2(b) surrogate identities (spec 7.9:
relative phase by one circular cross-correlation per segment, cross-correlation by the circular FFT identity with its
exact edge correction) and the D4 cluster/team-sync arithmetic (explicit real arithmetic, sequential sums, two-pass
Pearson; numba == numpy). On the FULL TF-58 corpus, in the D3 configuration (the FINAL params -- D1's
``derivation.json`` and D2's ``calibration.json`` via ``--derivation`` / ``--calibration``, the configuration that
ships, recorded with both sha256 in every token, manifest and the verdict; ``--in-package-params`` is a recorded dev
run -- and ``n_surrogates`` = ``D3_N_SURROGATES``), no source token may flip, no surrogate percentile may cross a
decision threshold and no value may change between NaN and finite, between the reference numerics
(``SILLY_KICKS_COORDINATION_REFERENCE_NUMERICS=1``: the direct per-draw nulls and the as-built cluster arithmetic) and
production. A violation STOPS the rollout: it is a real
behaviour change for the owner to rule on, never shipped under the pre-authorisation. Every other float column's max
|delta| is reported beside the verdict (the owner's rule: the measured number goes on record, not only the bound).
Owner-run on the DGX, like ``validate_das_native_parity`` (ADR-107).

Two passes (ADR-052):
  --pass map     per match, ``match_tables`` under BOTH numerics, compared in memory. The shard holds AGGREGATES ONLY
                 (per table x column: rows compared, source-token flips, NaN-pattern changes, percentiles changed,
                 threshold crossings, max |delta|) -- never an owner-tier row (ADR-038). The manifest carries the
                 per-leg wall clock (R8).
  --pass reduce  corpus totals per (table, column) and per provider, the gate verdict and the leg timings; writes
                 ``<--out>/numerics_noflip.json`` with its ``input_contract`` (commit 2 copies it into
                 ``docs/research/tf58_team_coordination/``; the DGX run never writes the repo).

Shared corpus flags come from ``_coordination_corpus.add_common_args`` (``--out``/``--token``/``--max-matches``/
``--cache-dir``/``--match-ids-json``/``--providers``/``--allow-dirty``/``--list-matches``); the tree is checked with
``require_clean_tree(git_provenance())`` FIRST (ADR-037).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from scripts._driver import CorpusPassResult

# Sibling driver modules in package form (``python -m scripts.validate_coordination_numerics``); tests patch the names
# bound HERE (``corpus_source`` / ``match_tables``).
from scripts._coordination_corpus import (
    PARAMS_PROVENANCE_FIELDS,
    RESULT_TABLES,
    StageTimer,
    add_common_args,
    add_params_args,
    assert_exclusions_declared,
    combine_workers,
    corpus_source,
    corpus_visibility_label,
    expected_corpus,
    match_tables,
    params_resolver,
    read_declared_exclusions,
    run_params_token,
    table_pairs,
    visibility_preflight,
    write_worker_partial,
)
from scripts.validate_team_coordination import D3_N_SURROGATES
from silly_kicks.coordination import CoordinationParams
from silly_kicks.coordination._compute import REFERENCE_NUMERICS_ENV
from silly_kicks.tracking._geometry import GEOMETRY_VERSION

#: Bumping this invalidates every numerics shard generation (it is in the pass's ``token_inputs``).
_SHARD_SCHEMA_VERSION = "tf58-numerics-1"
#: The surrogate-percentile decision thresholds a percentile may not cross: the two-sided 2.5 %/5 % tails and the
#: median. (A percentile moving WITHOUT crossing one is counted and reported, not gated.)
PERCENTILE_THRESHOLDS = (0.025, 0.05, 0.5, 0.95, 0.975)
#: The compared tables: every metric table ``match_tables`` melts (the window table carries no numerics).
COMPARED_TABLES = tuple(t for t in RESULT_TABLES if t != "windows")

_LEAD = ("provider", "match_id", "variant", "table")
_AGG_COLUMNS = [
    "table",
    "column",
    "n",
    "n_compared",  # rows actually compared: NaN-NaN cells excluded for float columns (review A-55)
    "source_flips",
    "nan_changes",
    "pct_changed",
    "crossings",
    "max_abs_dev",
]


@contextlib.contextmanager
def reference_numerics() -> Iterator[None]:
    """Score under the as-built reference numerics inside the block; the prior environment is restored after."""
    prior = os.environ.get(REFERENCE_NUMERICS_ENV)
    os.environ[REFERENCE_NUMERICS_ENV] = "1"
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(REFERENCE_NUMERICS_ENV, None)
        else:
            os.environ[REFERENCE_NUMERICS_ENV] = prior


def _tokens(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), "<NA>")


def compare_numerics(reference: pd.DataFrame, production: pd.DataFrame) -> pd.DataFrame:
    """Aggregate comparison of one match's ``match_tables`` frames under the two numerics (same rows, same order).

    One row per (table, column) the table fills: rows compared ``n``; for ``*_source`` columns the token flips; for
    float columns the NaN-pattern changes and the max |delta| over rows finite in both; for ``*_percentile`` columns
    also the percentiles changed and the crossings of each of :data:`PERCENTILE_THRESHOLDS`. Any other column (the
    keys, the counts) must be identical -- a difference raises, since the two legs would describe different rows.
    """
    rows: list[dict[str, object]] = []
    for table in COMPARED_TABLES:
        ref = reference[reference["table"] == table].reset_index(drop=True)
        new = production[production["table"] == table].reset_index(drop=True)
        if len(ref) != len(new):
            raise ValueError(f"{table}: {len(ref)} reference rows vs {len(new)} production rows")
        if not len(ref):
            continue
        for col in ref.columns:
            if col in _LEAD or not (ref[col].notna().any() or new[col].notna().any()):
                continue  # a lead column, or another table's column (NaN throughout the melted union)
            a, b = ref[col], new[col]
            entry: dict[str, object] = dict.fromkeys(_AGG_COLUMNS, 0)
            # n = rows in the table; n_compared = rows that actually carry a value for this column (NaN-NaN cells of
            # the melted union are not a comparison). A source/token column compares every row (A-55).
            entry.update({"table": table, "column": col, "n": len(ref), "n_compared": len(ref), "max_abs_dev": 0.0})
            if col.endswith("_source"):
                entry["source_flips"] = int((_tokens(a) != _tokens(b)).sum())
            elif a.dtype == np.float64 and b.dtype == np.float64:
                x, y = a.to_numpy(dtype=np.float64), b.to_numpy(dtype=np.float64)
                entry["nan_changes"] = int((np.isnan(x) != np.isnan(y)).sum())
                entry["n_compared"] = int((~np.isnan(x) | ~np.isnan(y)).sum())  # finite in at least one leg
                both = ~np.isnan(x) & ~np.isnan(y)
                if both.any():
                    entry["max_abs_dev"] = float(np.max(np.abs(x[both] - y[both])))
                if col.endswith("_percentile"):
                    entry["pct_changed"] = int((x[both] != y[both]).sum())
                    entry["crossings"] = sum(
                        int(((x[both] >= thr) != (y[both] >= thr)).sum()) for thr in PERCENTILE_THRESHOLDS
                    )
            elif not _tokens(a).equals(_tokens(b)):
                raise ValueError(f"{table}.{col}: the reference and production legs describe different rows")
            else:
                continue  # an identical key/count column: nothing to aggregate
            rows.append(entry)
    return pd.DataFrame(rows, columns=_AGG_COLUMNS)


def numerics_match(
    loaded,
    *,
    params_for: Callable[[str], CoordinationParams] = CoordinationParams.for_provider,
    timer: StageTimer | None = None,
) -> pd.DataFrame:
    """One match's aggregate comparison: ``match_tables`` in the D3 configuration under the reference numerics, then
    under production. ``params_for`` resolves the provider's params (``params_resolver``: the D1/D2 artifacts);
    ``timer`` receives the two legs' wall clock (``reference`` / ``production``, R8)."""
    params = params_for(loaded.provider)
    stage = timer if timer is not None else (lambda _name: contextlib.nullcontext())
    with stage("reference"), reference_numerics():
        ref = match_tables(loaded, params, n_surrogates=D3_N_SURROGATES)
    with stage("production"):
        prod = match_tables(loaded, params, n_surrogates=D3_N_SURROGATES)
    agg = compare_numerics(ref, prod)
    agg.insert(0, "match_id", str(loaded.match_id))
    agg.insert(0, "provider", loaded.provider)
    return agg


def reduce_numerics(agg: pd.DataFrame) -> dict:
    """Corpus totals per (table, column) and per provider, and the gate verdict (``no_flip``): True only when at least
    one match was compared and there is no source-token flip, no percentile threshold crossing and no NaN-pattern
    change anywhere."""
    n_matches = int(agg[["provider", "match_id"]].drop_duplicates().shape[0]) if len(agg) else 0

    def _totals(frame: pd.DataFrame, by: list[str]) -> list[dict]:
        if not len(frame):
            return []
        grouped = frame.groupby(by, sort=True, observed=True)
        out = grouped[["n", "n_compared", "source_flips", "nan_changes", "pct_changed", "crossings"]].sum()
        out["max_abs_dev"] = grouped["max_abs_dev"].max()
        return [
            {**dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True)), **row}
            for key, row in out.to_dict(orient="index").items()
        ]

    flips = int(agg["source_flips"].sum()) if len(agg) else 0
    crossings = int(agg["crossings"].sum()) if len(agg) else 0
    nan_changes = int(agg["nan_changes"].sum()) if len(agg) else 0
    return {
        "no_flip": bool(n_matches > 0 and flips == 0 and crossings == 0 and nan_changes == 0),
        "n_matches": n_matches,
        "n_values_compared": int(agg["n_compared"].sum()) if len(agg) else 0,  # NaN-NaN cells excluded (A-55)
        "n_cells_total": int(agg["n"].sum()) if len(agg) else 0,
        "source_flips": flips,
        "percentile_threshold_crossings": crossings,
        "nan_pattern_changes": nan_changes,
        "percentiles_changed": int(agg["pct_changed"].sum()) if len(agg) else 0,
        "percentile_thresholds": list(PERCENTILE_THRESHOLDS),
        "by_column": _totals(agg, ["table", "column"]),
        "by_provider": _totals(agg, ["provider"]),
    }


# =============================================================================== input contract (declare_inputs)
def input_contract() -> dict:
    """The declared symbols the gate's numbers depend on (ADR-056); written into ``numerics_noflip.json``."""
    from scripts._input_contract import declare_inputs

    return declare_inputs(
        driver="validate_coordination_numerics",
        n_surrogates=D3_N_SURROGATES,
        percentile_thresholds=list(PERCENTILE_THRESHOLDS),
        compared_tables=list(COMPARED_TABLES),
        reference_env=REFERENCE_NUMERICS_ENV,
        geometry_version=GEOMETRY_VERSION,
    )


# =============================================================================== corpus pass + reduce
def _pass_map(args, prov) -> CorpusPassResult:
    from scripts._driver import for_each
    from scripts._partition import worker_tag

    dest = pathlib.Path(args.out)
    tag = worker_tag(args.match_ids_json)
    params_for, params_src = params_resolver(args)  # the configuration that SHIPS (review A-16 / R2-2)
    timer = StageTimer()
    refs, load = corpus_source(args)
    visibility_preflight(refs, load)  # ADR-069 Layer 2: before any compute
    with timer("corpus"):
        res = for_each(
            refs,
            key=lambda r: r.key,
            load=load,
            work=lambda loaded: numerics_match(loaded, params_for=params_for, timer=timer),
            shard_root=dest / "numerics_shards",
            token_inputs={
                "pass": "numerics",
                "n_surrogates": D3_N_SURROGATES,
                "percentile_thresholds": list(PERCENTILE_THRESHOLDS),
                "geometry_version": GEOMETRY_VERSION,
                "schema": _SHARD_SCHEMA_VERSION,
                "params": run_params_token(args, params_for),  # M-4: the values both legs compute with
                **params_src,  # M-5: the artifact pair's digests in every token
            },
            tag=tag,
            label="match",
        )
    write_worker_partial(dest, "numerics", res, prov, timer, tag=tag, extra=params_src)
    return res


def _reduce(args, prov) -> None:
    from scripts._provenance import git_tree_hash

    dest = pathlib.Path(args.out)
    # B-1: EVERY worker's share, proven to be the whole listed corpus with no failed match -- or a refusal (no
    # verdict is ever written over a partial corpus, spec 8.3).
    agg, population = combine_workers(
        dest, "numerics", expected=expected_corpus(args, args.providers), consistent=PARAMS_PROVENANCE_FIELDS
    )
    # exclusions nit: a PASS must cover (scored union declared_excluded) == corpus. Every excluded key must be DECLARED
    # (an input, not the run's own .excluded.json), else the gate refuses a silent-subset PASS; declared carries reason.
    excluded = assert_exclusions_declared(
        population.get("excluded_keys", []), read_declared_exclusions(getattr(args, "declared_excluded", None))
    )
    if not len(agg):
        agg = pd.DataFrame(columns=["provider", "match_id", *_AGG_COLUMNS])
    report = {
        **reduce_numerics(agg),
        "params": population["consistent"],  # WHICH params were certified (source + both sha256)
        "n_excluded": population.get("n_excluded", 0),
        "excluded": excluded,  # {declared-excluded key: reason} covered by this PASS (exclusions nit)
        "population": {k: v for k, v in population.items() if k != "stage_seconds"},
        "corpus_visibility": corpus_visibility_label(table_pairs(agg), token=getattr(args, "token", None)),  # ADR-038
        "stage_seconds": population["stage_seconds"],
        "input_contract": input_contract(),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        # A-55: the CONTENT hash of the tree the verdict ran on -- stable even when dirty vs HEAD, so a reader can tie
        # these numbers to the exact tree (e.g. 8a219312), not only to a commit + a dirty flag.
        "run_tree_hash": git_tree_hash(),
    }
    (dest / "numerics_noflip.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "no_flip",
                    "n_matches",
                    "n_excluded",
                    "excluded",
                    "source_flips",
                    "percentile_threshold_crossings",
                    "nan_pattern_changes",
                )
            },
            indent=2,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pass", dest="which", required=True, choices=["map", "reduce"])
    ap.add_argument(
        "--declared-excluded",
        dest="declared_excluded",
        default=None,
        help="reduce: JSON {provider__match: reason} of the matches allowed to be excluded (closed reason vocabulary); "
        "a PASS refuses any UNDECLARED exclusion (exclusions nit). Default: none (any exclusion fails).",
    )
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
    {"map": _pass_map, "reduce": _reduce}[args.which](args, prov)


if __name__ == "__main__":
    main()
