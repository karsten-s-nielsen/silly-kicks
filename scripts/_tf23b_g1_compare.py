"""TF-23b G1 compare: assert the backstop is a byte-identical no-op except for wrong-flag ET (ADR-035).

Compares per-match frames dumped by `_tf23b_g1_dump.py` from the 4.33.0 baseline vs the 4.34.0 tree.
Strict `check_dtype=True` (both build `final` identically; in-memory vs in-memory). Reports the
no-op set, the CHANGED set (with which periods differ + max |Δ| — the enumerated retrain scope), and
which matches carry period-5 (PSO) frames (now left un-oriented by the net for all callers).

    .venv/bin/python scripts/_tf23b_g1_compare.py --base /tmp/g1_base --tree /tmp/g1_434
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--tree", required=True, type=Path)
    args = ap.parse_args()

    base_dir, tree_dir = args.base.expanduser(), args.tree.expanduser()
    base_files = {p.name for p in base_dir.glob("*.parquet")}
    tree_files = {p.name for p in tree_dir.glob("*.parquet")}
    common = sorted(base_files & tree_files)
    only_base, only_tree = sorted(base_files - tree_files), sorted(tree_files - base_files)
    if only_base or only_tree:
        print(f"WARNING: file-set mismatch — only_base={only_base} only_tree={only_tree}")

    noop, changed, period5 = [], [], []
    for name in common:
        b = pd.read_parquet(base_dir / name)
        t = pd.read_parquet(tree_dir / name)
        if 5 in set(b["period_id"].dropna().astype(int).unique()):
            period5.append(name)

        identical = False
        if b.shape == t.shape:
            try:
                pd.testing.assert_frame_equal(b.reset_index(drop=True), t.reset_index(drop=True), check_dtype=True)
                identical = True
            except AssertionError:
                identical = False

        if identical:
            noop.append(name)
            continue

        # Characterise the change: which periods differ + max |Δ| on x/y (row-aligned).
        detail = {"match": name, "shape_base": b.shape, "shape_tree": t.shape}
        if b.shape == t.shape:
            diff_periods = sorted(
                int(p)
                for p in b["period_id"].dropna().unique()
                if not (
                    b.loc[b["period_id"] == p, ["x", "y"]]
                    .reset_index(drop=True)
                    .equals(t.loc[t["period_id"] == p, ["x", "y"]].reset_index(drop=True))
                )
            )
            dx = b["x"].to_numpy() - t["x"].to_numpy()
            dy = b["y"].to_numpy() - t["y"].to_numpy()
            import numpy as np

            detail["diff_periods"] = diff_periods
            detail["max_abs_dx"] = float(np.nanmax(np.abs(dx)))
            detail["max_abs_dy"] = float(np.nanmax(np.abs(dy)))
        changed.append(detail)

    print(f"=== G1 RESULT: {len(common)} matches compared ===")
    print(f"NO-OP (byte-identical, strict dtype): {len(noop)}")
    print(f"CHANGED (retrain scope): {len(changed)}")
    for d in changed:
        print(f"  - {d}")
    print(f"PERIOD-5 (PSO) matches: {len(period5)} -> {period5}")
    # Machine-readable tail for ADR-035.
    print("CHANGED_MATCH_IDS=" + ",".join(sorted(d["match"] for d in changed)))
    print("PERIOD5_MATCH_IDS=" + ",".join(period5))


if __name__ == "__main__":
    main()
