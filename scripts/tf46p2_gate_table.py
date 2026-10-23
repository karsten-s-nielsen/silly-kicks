"""TF-46 Arm 1 Part 2: assemble the pre-flight aggregate the gate table derives from (spec sections 15.3, 16.2).

Concatenates the committed per-run aggregates (``truth``, ``kick_cells``, ``subtype_merge``, ``penalty``,
``origin`` -- each written by its stage-contract walker in ``AGGREGATE_COLUMNS`` form) under ``--run-root`` into
ONE ``gate_aggregate.parquet``, and writes ``coverage.parquet`` (``coverage_report``: the share of each cell's
rows the derived table would ship, per data profile x cell x evidence x cell resolution). At the C3 phase
``silly_kicks.tracking._restart_validation.derive_gate_table`` reads ``gate_aggregate.parquet`` and derives the real
``RESTART_GATE_TABLE`` (ADR-056). Every source run must carry the SAME ``run_commit`` (C-03), and the derivation is
run here once as a smoke check so a malformed aggregate fails BEFORE it is committed.

Not a corpus walker (it reads five committed files), so it is not in the ADR-128 compute-contract population; it
refuses a dirty tree (ADR-037). Usage::

    python scripts/tf46p2_gate_table.py --out docs/research/tf46_part2 --run-root docs/research/tf46_part2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `from scripts._X` resolves

from scripts._provenance import git_provenance, require_clean_tree
from scripts._tf46p2_common import RESEARCH_ROOT, coverage_report, write_run_aggregate

_RUN = "gate_aggregate"
SOURCE_RUNS: tuple[str, ...] = ("truth", "kick_cells", "subtype_merge", "penalty", "origin")


def assemble(run_root: Path, *, expect_commit: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Concatenate the present per-run aggregates (``AGGREGATE_COLUMNS``), each tagged with ``source_run``.
    Returns ``(aggregate, provenance)``; ``provenance`` records each run's ``run_commit``. With
    ``expect_commit`` a run built at another commit refuses (C-03). A missing run is recorded, not invented:
    its cells derive to withheld."""
    from silly_kicks.tracking._restart_validation import AGGREGATE_COLUMNS

    parts, prov = [], {}
    for run in SOURCE_RUNS:
        agg, side = run_root / f"{run}.parquet", run_root / f"{run}.json"
        if not agg.exists():
            prov[run] = None
            continue
        meta = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
        if expect_commit is not None and meta.get("run_commit") != expect_commit:
            raise RuntimeError(f"{run}: run_commit {meta.get('run_commit')!r} != {expect_commit!r} (C-03)")
        prov[run] = meta.get("run_commit")
        df = pd.read_parquet(agg)
        missing = set(AGGREGATE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{run}.parquet is not an AGGREGATE_COLUMNS aggregate; missing {sorted(missing)}")
        parts.append(df[list(AGGREGATE_COLUMNS)].assign(source_run=run))
    cols = [*AGGREGATE_COLUMNS, "source_run"]
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)), prov


def coverage(aggregate: pd.DataFrame) -> pd.DataFrame:
    """The section 19 coverage report over the assembled aggregate: per data profile x cell x evidence x cell
    resolution, the share of rows in cells the derived table ships (a withheld cell counts as not valued)."""
    from silly_kicks.tracking._restart_validation import derive_gate_table

    if aggregate.empty:
        return coverage_report(pd.DataFrame())
    table = derive_gate_table(aggregate)
    shipped = {(k[1], k[2], k[3]): v for k, v in table.items()}
    rows = []
    for r in aggregate.itertuples(index=False):
        entry = shipped.get((str(r.restart_type), str(r.signal), str(r.cell_state)))
        rows.append(
            {
                "data_profile": f"{r.provider}|{r.tracking_release}|{r.gi_release}|{r.competition_season}",
                "cell": f"{r.restart_type}|{r.signal}|{r.cell_state}",
                "evidence": r.tier,
                "cell_resolution": entry.cell_resolution if entry is not None else "withheld",
                "value": 1.0 if entry is not None else float("nan"),
            }
        )
    return coverage_report(pd.DataFrame(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=RESEARCH_ROOT, help="Output DIRECTORY for the assembled aggregate.")
    ap.add_argument("--run-root", default=RESEARCH_ROOT, help="Directory holding the per-run aggregates.")
    ap.add_argument("--allow-dirty", action="store_true", help="dev run from a modified tree; recorded dirty: true")
    args = ap.parse_args()

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table, sources = assemble(Path(args.run_root), expect_commit=None if args.allow_dirty else prov["commit"])
    write_run_aggregate(table, out, _RUN, prov)
    coverage(table).to_parquet(out / "coverage.parquet", index=False)
    metrics = {
        "run": _RUN,
        "n_rows": len(table),
        "source_runs": sources,
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
    }
    (out / f"{_RUN}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
