"""Render the licensed SB360 coverage parquet as `coverage.md`.

No provenance guard, deliberately -- same class as `render_sb360_matrix.py`: reads a COMMITTED
artifact (`coverage.parquet` + `manifest_all.json`) and writes a document. It does no corpus work
and consumes no external data, so a guard would add nothing and would make the report unrenderable
during the session that produces it. Provenance travels BY REFERENCE to the manifest it stamps.

Usage::

    python scripts/render_sb360_licensed_coverage.py
    python scripts/render_sb360_licensed_coverage.py --out docs/research/sb360_licensed_coverage/coverage.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

_DEFAULT_DIR = "docs/research/sb360_licensed_coverage"
_CAVEAT = (
    "_Battery numbers are STRUCTURAL coverage (did the aggregator run + fraction populated on real "
    "freeze-frames), NOT tactical values -- they are synthetic-input hybrids; a coverage fraction is "
    "a denominator, never a signal (ADR-042)._"
)


def _frame_coverage(df: pd.DataFrame) -> list[str]:
    fc = df[df["kind"] == "frame_coverage"]
    rows = []
    for subj, g in fc.groupby("subject"):
        actions = float(g["denominator"].sum())
        rate = (g["value"] * g["denominator"]).sum() / actions if actions else float("nan")
        rows.append((subj, int(g["match_id"].nunique()), int(actions), rate))
    rows.sort(key=lambda r: (r[0] != "all", -r[3]))
    out = [
        "## Frame-existence coverage (per GK-domain type)",
        "",
        "| Type | matches | actions | frame-existence |",
        "|---|---|---|---|",
    ]
    for subj, m, a, rate in rows:
        out.append(f"| `{subj}` | {m} | {a} | {rate:.3f} |")
    return [*out, ""]


def _battery(df: pd.DataFrame) -> list[str]:
    bc = df[df["kind"] == "battery_column"]
    means = bc.groupby("subject")["value"].mean()
    full = sorted(means[means == 0.0].index)
    out = [
        "## Battery aggregator coverage",
        "",
        _CAVEAT,
        "",
        f"**{len(full)} of {means.shape[0]}** battery columns are fully-NaN across the corpus "
        "(mean populated fraction 0) -- the velocity-derived, ADR-063 Tier-2-suppressed, "
        "constitutively-tracking, and SB360-anonymity (no persistent freeze-frame player identity, "
        "ADR-054) columns. `add_visible_area_coverage`-style coverage fractions are denominators, "
        "not signals.",
        "",
        "<details><summary>The fully-NaN columns</summary>",
        "",
    ]
    out += [f"- `{c}`" for c in full]
    out += ["", "</details>", ""]
    return out


def _companion(df: pd.DataFrame) -> list[str]:
    cs = df[df["kind"] == "companion_source"]
    cf = df[df["kind"] == "companion_fraction"]  # mean_observed_fraction per feature
    out = [
        "## ADR-062 visibility companions",
        "",
        "Per count feature: the source-token breakdown (row counts) and the mean observed "
        "fraction. _An observed fraction is a coverage denominator, not a signal (ADR-042)._ "
        "Fractions are UNWEIGHTED per-match means; the frame-existence table above is "
        "denominator-weighted -- do not cross-compare them as the same statistic.",
        "",
        "| Feature | source | total rows |",
        "|---|---|---|",
    ]
    for subj, v in cs.groupby("subject")["value"].sum().sort_index().items():
        feature, _, token = str(subj).partition(".")
        out.append(f"| `{feature}` | `{token}` | {int(v)} |")
    out += ["", "| Feature | mean observed fraction |", "|---|---|"]
    for subj, v in cf.groupby("subject")["value"].mean().sort_index().items():
        out.append(f"| `{subj}` | {v:.3f} |")
    return [*out, ""]


def _pitch_roster_raises(df: pd.DataFrame) -> list[str]:
    pc = df[df["kind"] == "pitch_coverage"]["value"]
    pcs = df[df["kind"] == "pitch_coverage_source"]  # source-token counts (observed/no_polygon/...)
    ros = df[df["kind"] == "roster"]["value"]
    raises = df[df["kind"] == "battery_raises"]
    out = [
        "## Pitch coverage, roster, raises",
        "",
        f"- **Observed pitch fraction** (real `visible_area`): mean {pc.mean():.3f}, "
        f"min {pc.min():.3f}, max {pc.max():.3f} over {pc.shape[0]} matches. "
        "_A coverage denominator, not a signal._",
    ]
    if not pcs.empty:
        toks = ", ".join(f"`{t}` {int(v)}" for t, v in pcs.groupby("subject")["value"].sum().sort_index().items())
        out.append(f"- **Pitch-coverage source tokens** (summed rows -- coverage counts, not signals): {toks}.")
    out.append(
        f"- **Roster keeper-resolution rate** (a coverage rate, not a signal): "
        f"mean {ros.mean():.3f} over {ros.shape[0]} matches."
    )
    if not raises.empty:
        by = raises.groupby("subject")["match_id"].nunique().sort_index()
        out.append("- **Aggregators that raised** (an honest refusal, not a defect):")
        for subj, n in by.items():
            out.append(
                f"  - `{subj}`: {int(n)} matches (freeze-frame carried only one team's players near the action)."
            )
    return [*out, ""]


def render(df: pd.DataFrame, meta: dict) -> str:
    lines = [
        "# SB360 licensed-corpus coverage",
        "",
        "What the library produces on the **licensed** StatsBomb 360 corpus (30 matches). "
        "The companion to the open-data `../sb360_coverage/coverage.md`.",
        "",
        "## Provenance",
        "",
        "| | |",
        "|---|---|",
        "| Driver | `scripts/validate_sb360_licensed_corpus.py` |",
        f"| Generation | `{meta['generation']}` |",
        f"| Matches | {meta['n_attempted']} attempted, {meta['n_failed']} failed |",
        f"| Commit | `{meta['run_commit']}` |",
        f"| Tree | {'dirty' if meta.get('run_tree_dirty') else 'clean'} |",
        "",
        "Rendered from the committed `coverage.parquet`; licensed data is never committed.",
        "",
    ]
    lines += _frame_coverage(df)
    lines += _battery(df)
    lines += _companion(df)
    lines += _pitch_roster_raises(df)
    lines += [
        "## The 40 -> 31 fully-NaN lift",
        "",
        "The 4.85.0 velocity-less lift (ADR-063) moved the fully-NaN battery count from **40** "
        "(prior state) to the **31** this parquet records: velocity-requiring pitch-control "
        "aggregators now serve the zero-velocity positional model on declared freeze-frames.",
        "",
        "## Reading limits / reproducing",
        "",
        "- The battery per-column numbers are structural coverage, not tactics (see the caveat above).",
        "- Licensed data is never committed. Refresh the parquet with "
        "`python scripts/validate_sb360_licensed_corpus.py` (owner token required), then re-render with "
        "`python scripts/render_sb360_licensed_coverage.py`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(_DEFAULT_DIR) / "coverage.md",
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.out.parent / "coverage.parquet")
    meta = json.loads((args.out.parent / "manifest_all.json").read_text(encoding="utf-8"))
    args.out.write_text(render(df, meta), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
