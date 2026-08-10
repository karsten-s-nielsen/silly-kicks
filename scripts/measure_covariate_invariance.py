#!/usr/bin/env python
"""Measure which causal covariates move under a geometry change, and ATTRIBUTE the movement.

Why this exists
---------------
ADR-051 PR 5 changed geometry in **two** independent ways, and a spec revision asserted a clean
two-axis decomposition that does not exist on that diff:

* **Axis A** -- the goal-relative transform gained ``to_goal_relative_y``, so the two goal ends became
  a 180-degree ROTATION apart instead of opposite handedness.
* **Axis B** -- ``_dominant_region_area``'s y grid was re-anchored from centre 34.50 to 34.00.

They interact. The NEW grid is closed under the ADR-028 point reflection (``1.0 -> 67.0`` is a grid
centre) and the OLD one is not (``1.5 -> 66.5`` is not), so measuring axis A against the current code
forces ``space_controlled``'s axis-A delta to zero BY CONSTRUCTION -- while the baseline, which
carries the old grid, does move under it. Two arms cannot separate that; three can.

    arm            transform   grid      isolates
    parent         old         old       (the baseline)
    old_grid       new         old       axis A  = |old_grid - parent|
    current        new         new       axis B  = |current - old_grid|
                                         total   = |current - parent|

Isolation is by CHECKOUT, not by monkeypatch
--------------------------------------------
Both extractors bind geometry absolutely (``from silly_kicks.tracking import _geometry as _geo``), so
importing a baseline copy under another package name still resolves ``_geo`` to the CURRENT module.
For PR 5's diff that is inert -- the ``_geometry`` change is purely additive and the baseline
extractors contain zero references to ``to_goal_relative_y`` -- but this driver is billed as the
reusable instrument for PR 6, PR 7 and Cycle B, where a ``_geometry`` function may change BEHAVIOUR
rather than being added. In that case an in-process emulation would silently measure zero, which is
the silent-null shape this repo already catalogues four instances of. **Do not simplify this to
same-process imports on the grounds that today it makes no difference.** The baseline arm also stamps
its own ``GEOMETRY_VERSION``, which Step 5's control asserts differs from the current one.

The ``old_grid`` arm IS an in-process patch, and legitimately so: the grid change is exactly one
function (``_grid_centres``), so replacing that one function is the faithful hybrid.

Usage
-----
    git archive 6e3a132~1 | tar -x -C "$HOME/pr5_scratch/baseline_tree"
    python scripts/measure_covariate_invariance.py \
        --baseline-tree "$HOME/pr5_scratch/baseline_tree" \
        --out "$HOME/pr5_runs/covariate_invariance"

Writes ``metrics.json`` (+ ``report.md``) with ``run_commit`` / ``run_tree_dirty`` / ``status``.
Never a bare ``assert`` on a measurement: the table is written FIRST and a breach sets ``status`` and
exits non-zero, because an invariance that genuinely breaks must leave an artifact behind.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _input_contract import declare_inputs
from _provenance import git_provenance, require_clean_tree

REPO = Path(__file__).resolve().parents[1]
SLIM = REPO / "tests" / "datasets" / "tracking" / "action_context_slim"
GOAL_ENDS = (0.0, 105.0)

#: Frame columns the extractors consume. Mirrors the fixture generators.
_KEEP = {
    "game_id", "period_id", "frame_id", "time_seconds", "frame_rate", "player_id", "team_id",
    "is_ball", "is_goalkeeper", "x", "y", "z", "speed", "speed_source", "ball_state",
    "team_attacking_direction", "confidence", "visibility", "source_provider",
}  # fmt: skip

#: LAYER2 confounders that are per-spell JOINS, not extractor features -- emitted as data rather than
#: omitted, because two of them (`defensive_line_height`, `_compactness`) are PR 6's own mechanism and
#: an instrument billed as reusable for PR 6 must not silently drop them.
_NOT_MEASURABLE = (
    "defensive_line_height",
    "defensive_line_compactness",
    "pressure_on_actor__bekkers_pi",
    "time_remaining_s",
)

#: NOT importable: these exist only as literal `gk_block=` tuples at opportunities.py:139 and :198.
#: `GK_BLOCK` itself is the SIX lowercase xCross names.
_XS_GK_BLOCK = ("GK_r", "GK_theta")


def input_contract() -> dict:
    """Declare WHICH SYMBOLS these numbers depend on (Cycle B).

    This driver MEASURES the geometry transform, so its own dependence on `GEOMETRY_VERSION` is
    the point rather than an incidental input: a bump means the measurement describes a transform
    that is no longer the live one.
    """
    from silly_kicks.tracking import _geometry as _geo

    return declare_inputs(
        driver="measure_covariate_invariance",
        geometry_version=_geo.GEOMETRY_VERSION,
        extractors=(
            "silly_kicks.tracking._xshot_occurrence",
            "silly_kicks.tracking._xcross_attempt",
        ),
    )


def _load_frames() -> list[pd.DataFrame]:
    """One DataFrame per (game, period, frame) group across the committed slim fixtures."""
    groups: list[pd.DataFrame] = []
    for prov in ("sportec", "skillcorner", "metrica"):
        path = SLIM / f"{prov}_slim.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        frames = df[df["__kind"] == "frame"].drop(columns=["__kind"]).reset_index(drop=True)
        frames = frames[[c for c in frames.columns if c in _KEEP]].copy()
        frames["vx"] = frames["speed"].astype(float) if "speed" in frames.columns else 0.0
        frames["vy"] = 0.0
        outfield = frames[~frames["is_ball"].astype(bool)]
        if outfield.empty:
            continue
        def_team = outfield.groupby("team_id")["x"].mean().idxmin()
        for _, g in frames.groupby(["game_id", "period_id", "frame_id"], dropna=False):
            g = g.copy()
            g.attrs["gk_team_id"] = def_team
            groups.append(g)
    return groups


def _extract_arm(*, old_grid: bool) -> dict[str, dict[str, list[float]]]:
    """Extract every covariate at both goal ends, in THIS interpreter.

    ``old_grid`` swaps ``_grid_centres`` for the pre-PR-5 ``arange(res/2, L, res)`` anchor.
    """
    from silly_kicks.tracking import _geometry as _geo
    from silly_kicks.tracking import _xcross_attempt as _xc
    from silly_kicks.tracking import _xshot_occurrence as _xs

    # `_grid_centres` was ADDED by the fix commit, so the BASELINE tree does not have it -- it
    # anchors its grid inline. Probing with getattr is therefore load-bearing, not defensive: an
    # AttributeError here is what the baseline arm legitimately looks like, and treating it as fatal
    # would make the driver unable to measure the very diff it exists for. (It is also the clearest
    # proof the isolation is real: if the subprocess had resolved `silly_kicks` to the CURRENT tree,
    # the attribute WOULD be present and the arms would silently share a grid.)
    original_grid = getattr(_xc, "_grid_centres", None)
    if old_grid:
        _xc._grid_centres = lambda length, res: np.arange(res / 2.0, length, res)
    try:
        out = _extract_all(_xs, _xc)
    finally:
        # RESTORE, always. Without this the patch LEAKS into the next arm: `current` would run on
        # the OLD grid, axis B would measure exactly 0.0, and the artifact would assert that the
        # grid re-anchor moved nothing -- the very claim the three-arm design exists to test. The
        # positive control does NOT catch it, because axis A still moves.
        if original_grid is not None:
            _xc._grid_centres = original_grid
        elif hasattr(_xc, "_grid_centres"):
            del _xc._grid_centres  # never existed here; do not leave one behind
    out["_geometry_version"] = getattr(_geo, "GEOMETRY_VERSION", "<absent>")
    return out


def _extract_all(_xs, _xc) -> dict:
    out: dict[str, dict[str, list[float]]] = {}
    groups = _load_frames()
    for goal_x in GOAL_ENDS:
        key = f"goal_x_{int(goal_x)}"
        rows_xs, rows_xc = [], []
        for g in groups:
            gk = g.attrs["gk_team_id"]
            rows_xs.append(_xs.extract_xshot_features(g, gk_team_id=gk, goal_x=goal_x).iloc[0])
            carrier = g[(~g["is_ball"].astype(bool))]["player_id"].iloc[0]
            rows_xc.append(
                _xc.extract_xcross_features(
                    g,
                    gk_team_id=gk,
                    goal_x=goal_x,
                    carrier_player_id=carrier,
                    score_differential=np.nan,
                ).iloc[0]
            )
        xs_df, xc_df = pd.DataFrame(rows_xs), pd.DataFrame(rows_xc)
        # gk_depth_x is the Layer-2 TREATMENT and is not in any confounder tuple -- it exists only as
        # _COVARIATES["gk_depth_x"] (opportunities.py:395). Asserting on it without deriving it here
        # is what would raise KeyError before the table was ever written.
        xs_df["gk_depth_x"] = xs_df["GK_r"].astype(float) * np.cos(xs_df["GK_theta"].astype(float))
        cols = {c: xs_df[c].astype(float).tolist() for c in xs_df.columns}
        cols.update({c: xc_df[c].astype(float).tolist() for c in xc_df.columns})
        out[key] = cols
    return out


def _run_baseline_arm(baseline_tree: Path) -> dict:
    """Run the parent-commit extractors in a SEPARATE interpreter against ``baseline_tree``."""
    prog = (
        "import json,sys;"
        f"sys.path.insert(0, r'{baseline_tree}');"
        f"sys.path.insert(0, r'{REPO / 'scripts'}');"
        "import measure_covariate_invariance as m;"
        "print('@@JSON@@' + json.dumps(m._extract_arm(old_grid=False)))"
    )
    env = dict(os.environ, PYTHONPATH=str(baseline_tree), PYTHONDONTWRITEBYTECODE="1")
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-c", prog], capture_output=True, text=True, env=env, cwd=str(baseline_tree)
    )
    if res.returncode != 0 or "@@JSON@@" not in res.stdout:
        raise SystemExit(f"baseline arm failed (rc={res.returncode}):\n{res.stderr[-2000:]}")
    return json.loads(res.stdout.split("@@JSON@@", 1)[1].strip())


def _max_abs(a: list[float], b: list[float]) -> float:
    x, y = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.max(np.abs(x[m] - y[m]))) if m.any() else float("nan")


def _arm_membership() -> dict[str, list[str]]:
    from silly_kicks.causal.opportunities import (
        GK_BLOCK,
        LAYER2_BUILD_CONFOUNDERS,
        LAYER2_CONFOUNDERS,
        PAPER_CONFOUNDERS,
        SHOT_ARM_CONFOUNDERS,
    )

    return {
        "shot": sorted(set(SHOT_ARM_CONFOUNDERS) | set(_XS_GK_BLOCK)),
        "cross": sorted(set(PAPER_CONFOUNDERS) | set(GK_BLOCK)),
        "layer2_build": sorted(LAYER2_BUILD_CONFOUNDERS),
        "layer2_analysis": sorted(LAYER2_CONFOUNDERS),
        "treatment": ["gk_depth_x"],
    }


def run(out: Path, baseline_tree: Path) -> dict:
    parent = _run_baseline_arm(baseline_tree)
    old_grid = _extract_arm(old_grid=True)
    current = _extract_arm(old_grid=False)

    membership = _arm_membership()
    in_any = {c for names in membership.values() for c in names}

    rows: list[dict] = []
    for end in (f"goal_x_{int(g)}" for g in GOAL_ENDS):
        shared = set(current[end]) & set(parent.get(end, {})) & set(old_grid[end])
        for col in sorted(shared):
            axis_a = _max_abs(old_grid[end][col], parent[end][col])
            axis_b = _max_abs(current[end][col], old_grid[end][col])
            total = _max_abs(current[end][col], parent[end][col])
            # NaN FIRST. A NaN delta means the covariate could not be compared at all -- here
            # `score_differential` is all-NaN by construction, since the slim fixtures carry no score
            # context and `_extract_arm` passes np.nan. Without this branch every NaN comparison below
            # is False and the covariate falls through to "B", so the artifact would assert that the
            # grid re-anchor moved a confounder it never measured. A wrong axis on a causal covariate
            # is worse than an absent one.
            if not (np.isfinite(axis_a) and np.isfinite(axis_b) and np.isfinite(total)):
                axis = "not-measurable"
            elif max(axis_a, axis_b, total) <= 1e-12:
                axis = "none"
            elif axis_a > 1e-12 and axis_b > 1e-12:
                axis = "A+interaction"
            else:
                axis = "A" if axis_a > 1e-12 else "B"
            rows.append(
                {
                    "covariate": col,
                    "goal_end": end,
                    "arms": [k for k, v in membership.items() if col in v] or ["model-feature-only"],
                    "axis_a_transform": axis_a,
                    "axis_b_grid": axis_b,
                    "total": total,
                    "axis": axis,
                    "is_causal_covariate": col in in_any,
                }
            )

    for col in _NOT_MEASURABLE:
        rows.append(
            {
                "covariate": col,
                "goal_end": "n/a",
                "arms": ["layer2_analysis"],
                "source": "join-time (causal/_confounders.py)",
                "delta": "not-measurable-by-this-driver",
                "note": "per-spell join, not an extractor feature (opportunities.py:152-157)",
            }
        )

    prov = git_provenance()
    metrics = {
        "n_frame_groups": len(_load_frames()),
        "goal_ends": list(GOAL_ENDS),
        "arms": {
            "parent": "old transform, old grid",
            "old_grid": "new transform, old grid",
            "current": "new transform, new grid",
        },
        "baseline_geometry_version": parent.get("_geometry_version"),
        "current_geometry_version": current.get("_geometry_version"),
        "membership": membership,
        "rows": rows,
        "input_contract": input_contract(),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
    }

    # --- controls, evaluated AFTER the table exists -------------------------------------------
    breaches: list[str] = []

    def _delta(col: str, end: str, field: str) -> float:
        for r in rows:
            if r.get("covariate") == col and r.get("goal_end") == end:
                return float(r[field])
        return float("nan")

    # NEGATIVE controls -- structurally exact, and what the decision not to rebuild
    # tf19_signoff_power rests on.
    for col, why in (("gk_depth_x", "cos is even"), ("GK_r", "hypot(a,-b) == hypot(a,b)")):
        d = _delta(col, "goal_x_105", "axis_a_transform")
        if not (d == 0.0):
            breaches.append(f"{col} axis-A@105 must be EXACTLY 0 ({why}); measured {d!r}")

    # POSITIVE control -- an all-zero table is what a FAILED baseline isolation looks like, and two
    # `== 0` assertions would pass on it. This is the assertion that can fail.
    numeric = [r for r in rows if "total" in r]
    if not any(np.isfinite(r["total"]) and abs(float(r["total"])) > 1e-6 for r in numeric):
        breaches.append("all-zero table: the baseline arm did not isolate (nothing moved anywhere)")
    # AXIS-B control. A leaked `_grid_centres` patch makes every axis-B delta 0.0 while axis A still
    # moves, so the "any delta moved" control above passes on a contaminated run. `space_controlled`
    # is the one covariate the grid re-anchor is KNOWN to move; if it did not, the arms are not what
    # they claim to be.
    if not any(
        r.get("covariate") == "space_controlled"
        and np.isfinite(r.get("axis_b_grid", float("nan")))
        and abs(float(r["axis_b_grid"])) > 1e-6
        for r in numeric
    ):
        breaches.append(
            "axis B moved nothing on space_controlled: the old_grid patch likely leaked into the "
            "current arm, so the two arms share a grid"
        )
    if metrics["baseline_geometry_version"] == metrics["current_geometry_version"]:
        breaches.append(
            f"baseline resolved to the CURRENT _geometry "
            f"({metrics['baseline_geometry_version']!r} == {metrics['current_geometry_version']!r})"
        )

    metrics["status"] = (
        "ok"
        if not breaches
        else ("isolation_failed" if any("baseline" in b or "all-zero" in b for b in breaches) else "invariance_breach")
    )
    metrics["breaches"] = breaches

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "report.md").write_text(_render(metrics), encoding="utf-8")
    return metrics


def _render(m: dict) -> str:
    lines = [
        "# Covariate invariance under the ADR-051 geometry change",
        "",
        f"status: **{m['status']}**  |  run_commit `{m['run_commit'][:12]}`  |  "
        f"dirty {m['run_tree_dirty']}  |  {m['n_frame_groups']} frame groups",
        "",
        f"baseline geometry_version `{m['baseline_geometry_version']}` -> current `{m['current_geometry_version']}`",
        "",
        "| covariate | end | arms | axis A (transform) | axis B (grid) | total | axis |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in m["rows"]:
        if "total" not in r:
            continue
        lines.append(
            f"| `{r['covariate']}` | {r['goal_end']} | {', '.join(r['arms'])} | "
            f"{r['axis_a_transform']:.3e} | {r['axis_b_grid']:.3e} | {r['total']:.3e} | {r['axis']} |"
        )
    lines += ["", "## Not measurable by this driver", ""]
    for r in m["rows"]:
        if "total" in r:
            continue
        lines.append(f"- `{r['covariate']}` -- {r['source']}; {r['note']}")
    if m["breaches"]:
        lines += ["", "## Breaches", ""] + [f"- {b}" for b in m["breaches"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--baseline-tree",
        type=Path,
        required=True,
        help="checkout of the pre-change commit (git archive <sha>~1 | tar -x -C <dir>)",
    )
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit a dev run on a dirty tree; the artifact still records run_tree_dirty=true",
    )
    a = ap.parse_args()
    # ADR-037: the CLI refuses, run() records the truth.
    require_clean_tree(git_provenance(), allow_dirty=a.allow_dirty)
    m = run(a.out, a.baseline_tree)
    print(f"status={m['status']}  rows={len([r for r in m['rows'] if 'total' in r])}")
    for b in m["breaches"]:
        print(f"  BREACH: {b}")
    if m["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
