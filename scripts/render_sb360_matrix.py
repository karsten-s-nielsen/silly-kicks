"""Render the SB360 behaviour matrix from the verdict registry.

No provenance guard, deliberately. ``scripts/_provenance.require_clean_tree`` exists for
drivers whose numbers come from a CORPUS PASS, where a bare ``git rev-parse HEAD`` would
misattribute expensive results to code that did not produce them (ADR-037). This renderer reads
a COMMITTED registry and writes a document: it does no corpus work, consumes no external data,
and is deterministic given the tree. Adding the guard here would be cargo-cult -- and would make
the report unrenderable during exactly the editing session that produces it.

``scripts/build_sb360_coverage.py`` (Layer B) DOES take the guard, because it measures real
match data.

Usage::

    python scripts/render_sb360_matrix.py --out docs/research/sb360_coverage/behaviour_matrix.md
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tests.sb360._fixture import FIXTURE_VERSION
from tests.sb360._registry import (
    PROVENANCE_COLUMNS,
    SB360_ENTRIES,
    iter_verdicts,
)

#: Reported in the summary so a reader can see which adjudications are findings and which are
#: reassurance, without counting rows.
_ORDER = (
    "silent_degrade",
    "differs_by_design",
    "not_exercised",
    "honest_nan",
    "works",
    "raises",
)


def _tally() -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for entry in SB360_ENTRIES.values():
        for _axis, _roster, _col, v in iter_verdicts(entry):
            c[v.adjudication] += 1
    return c


def _findings() -> list[tuple[str, str, str, str, str]]:
    """Every `silent_degrade`, with its rationale. The actionable half of the audit."""
    rows = []
    for name in sorted(SB360_ENTRIES):
        for axis, roster, col, v in iter_verdicts(SB360_ENTRIES[name]):
            if v.adjudication == "silent_degrade":
                rows.append((name, col, axis, roster, v.rationale or ""))
    return rows


#: GK-domain entry points, ENUMERATED not name-matched.
#:
#: An earlier version selected on the substrings "gk"/"goalmouth"/"ghost". That is the defect
#: class ADR-043 ruled on when it deleted the id-compat AST lint: complete by ENUMERATION where
#: a heuristic is incomplete by construction. The heuristic already omitted
#: `add_xshot_occurrence`, and a future GK feature not named `*gk*` would vanish from a table
#: that goes in front of a club -- silently, because a missing row looks like no row.
#:
#: `add_xshot_occurrence` is INCLUDED: it is shot-domain and its coverage bears on the same
#: question, even though the fixture currently leaves it unexercised.
GK_DOMAIN_ENTRIES: tuple[str, ...] = (
    "add_xt_gk",
    "add_gk_completion",
    "add_gk_influence",
    "add_pre_shot_gk_position",
    "add_pre_shot_gk_angle",
    "add_ghost_gk",
    "add_shot_goalmouth",
    "add_xshot_occurrence",
)


def _gk_summary() -> list[tuple[str, str, str]]:
    """GK-domain entry points: what they do on a freeze-frame, and what happens without the
    keeper. The collaboration's actual question."""
    missing = [n for n in GK_DOMAIN_ENTRIES if n not in SB360_ENTRIES]
    if missing:
        raise KeyError(
            f"GK_DOMAIN_ENTRIES names {missing}, which are not in the registry. A renamed or "
            f"removed aggregator must be reflected here rather than silently dropping a row."
        )
    gk = [n for n in GK_DOMAIN_ENTRIES if n in SB360_ENTRIES]
    out = []
    for name in gk:
        e = SB360_ENTRIES[name]
        vel = collections.Counter(v.adjudication for v in e.velocity.values())
        abs_ = collections.Counter(v.adjudication for v in e.visibility.get("gk_absent", {}).values())
        out.append((name, _fmt_counter(vel), _fmt_counter(abs_)))
    return out


def _fmt_counter(c: collections.Counter) -> str:
    if not c:
        return "-"
    return ", ".join(f"{k} x{n}" for k, n in sorted(c.items(), key=lambda kv: -kv[1]))


def render() -> str:
    tally = _tally()
    total = sum(tally.values())
    lines = [
        "# SB360 behaviour matrix",
        "",
        f"Fixture `{FIXTURE_VERSION}`. **Observations are re-derived and locked on every CI "
        "run; adjudications are human judgements carrying a written rationale.** The lock "
        "covers the machine half only -- what it guarantees is that a stale adjudication "
        "cannot hide, not that the adjudication is right.",
        "",
        "Linkage-provenance columns "
        f"(`{'`, `'.join(sorted(PROVENANCE_COLUMNS))}`) carry no verdict: Leg A's `frame_id` "
        "*is* its `action_id` by construction while Leg B numbers a 10 Hz stream, so they "
        "differ between two legs that agree about everything that matters.",
        "",
        "## Summary",
        "",
        f"{total} verdicts across {len(SB360_ENTRIES)} entry points, on three axes "
        "(velocity; visibility with the keeper removed; visibility with an outfielder removed).",
        "",
        "| Adjudication | Count | Meaning |",
        "|---|---|---|",
    ]
    meaning = {
        "silent_degrade": "**Returns a plausible number with no basis.** The actionable finding.",
        "differs_by_design": "Differs, but coherently -- a weaker model, not an invented value.",
        "not_exercised": "The fixture does not reach this column on this axis.",
        "honest_nan": "Declines cleanly. Absence stays visible downstream.",
        "works": "Identical with or without velocity.",
        "raises": "Fails loud on freeze-frame input.",
    }
    for k in _ORDER:
        if tally.get(k):
            lines.append(f"| `{k}` | {tally[k]} | {meaning[k]} |")

    lines += ["", "## Findings -- every `silent_degrade`", ""]
    findings = _findings()
    if not findings:
        lines.append("None. No column was adjudicated a fabrication.")
    else:
        lines += ["| Function | Column | Axis | Roster | Rationale |", "|---|---|---|---|---|"]
        for name, col, axis, roster, rat in findings:
            lines.append(f"| `{name}` | `{col}` | {axis} | {roster} | {rat} |")

    lines += [
        "",
        "## GK domain",
        "",
        "The collaboration's question, in two columns: what each GK entry point does on a "
        "freeze-frame, and what happens when the keeper is **not in it** -- which for SB360 "
        "means outside the broadcast camera.",
        "",
        "| Entry point | Velocity axis | Keeper absent |",
        "|---|---|---|",
    ]
    for name, vel, absent in _gk_summary():
        lines.append(f"| `{name}` | {vel} | {absent} |")

    lines += [
        "",
        "## Full matrix",
        "",
        "| Function | Column | Axis/roster | Observation | Adjudication | Applicability |",
        "|---|---|---|---|---|---|",
    ]
    for name in sorted(SB360_ENTRIES):
        e = SB360_ENTRIES[name]
        for axis, roster, col, v in iter_verdicts(e):
            scope = "velocity" if axis == "velocity" else roster
            lines.append(
                f"| `{name}` | `{col}` | {scope} | `{v.observation}` | `{v.adjudication}` | "
                f"{e.applicability.get(col, '')} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(), encoding="utf-8")
    print(f"wrote {args.out} ({sum(_tally().values())} verdicts)")


if __name__ == "__main__":
    main()
