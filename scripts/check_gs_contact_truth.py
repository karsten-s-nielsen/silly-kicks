"""Staged equals served: the ``gs_contact_truth`` frames against the live pining loader (scorer spec section 4).

Before the scorer's development run: for every unit (all 64; parallel workers through the partition wrapper),
the staged frames must equal the frames the loader returns at the run's commit, on every detector input column
and on the row set, exactly. Ids
compare by value (``id_compat``: a stored float team id and a categorical one are the same id). It compares data
only and computes no detector output, so checking the held-out units reveals nothing about the held-out score. A
mismatch exits non-zero and stops the cycle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: The frame columns the detector reads (``_ball_contact._candidate_peaks``), checked exactly.
CHECK_COLUMNS = (
    "period_id",
    "frame_id",
    "time_seconds",
    "is_ball",
    "player_id",
    "team_id",
    "x",
    "y",
    "z",
    "z_source",
    "frame_rate",
)
_IDS = ("player_id", "team_id")


def _keyed(df: pd.DataFrame) -> pd.DataFrame:
    from silly_kicks.id_compat import canonical_id_series

    out = df.copy()
    for col in _IDS:
        if col in out:
            out[col] = canonical_id_series(out[col]).astype(object)
    if "z_source" in out:
        out["z_source"] = out["z_source"].astype(object)
    key = ["period_id", "frame_id", "is_ball", "player_id"]
    return out.sort_values(key, kind="stable", na_position="first").reset_index(drop=True)


def compare_frames(staged: pd.DataFrame, live: pd.DataFrame) -> list[str]:
    """The names of the checked columns that differ (``"rows"`` when the row sets differ); empty when equal."""
    if len(staged) != len(live):
        return ["rows"]
    s, v = _keyed(staged), _keyed(live)
    bad = []
    for col in CHECK_COLUMNS:
        if col not in s or col not in v:
            if col in s or col in v:
                bad.append(col)
            continue
        a, b = s[col], v[col]
        if col in ("x", "y", "z", "time_seconds", "frame_rate"):
            x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            same = np.array_equal(x, y, equal_nan=True)
        else:
            same = bool(((a == b) | (a.isna() & b.isna())).fillna(False).all())
        if not same:
            bad.append(col)
    return bad


def run(data, *, work_dir, provenance=None, cache_dir=None, source=None) -> dict:
    """Compare every unit of ``data`` with the live loader; ``source=(refs, load)`` injects a loader for tests, where
    ``load(ref)`` returns the loaded match. Resume is ``for_each`` over cheap refs (ADR-052)."""
    from scripts._driver import for_each, shard_path
    from scripts._driver_sources import read_dataset_parquet
    from scripts._provenance import git_provenance
    from scripts.hf_jobs.datasets import pergame_name

    data, work_dir = Path(data), Path(work_dir)
    prov = provenance or git_provenance()
    if source is None:
        from _loader_pining import pining_source, resolve_cache_dir

        refs, load = pining_source(["gradientsports"], token=None, cache_dir=resolve_cache_dir(cache_dir))
    else:
        refs, load = source
    staged_units = {p.name for p in data.iterdir() if p.is_dir() and not p.name.startswith(".")}

    def _work(lm):
        unit = pergame_name("gradientsports", str(lm[1]))
        if unit not in staged_units:
            return pd.DataFrame({"unit": [unit], "mismatches": [json.dumps(["unit missing"])]})
        staged = read_dataset_parquet(data / unit / "frames.parquet")
        return pd.DataFrame({"unit": [unit], "mismatches": [json.dumps(compare_frames(staged, lm[3]))]})

    res = for_each(
        refs,
        key=lambda ref: ref.key,
        load=load,
        work=_work,
        counters=lambda _item, _frame: {"n_units": 1},
        shard_root=work_dir / "shards",
        token_inputs={"mode": "check_gs_contact_truth", "columns": list(CHECK_COLUMNS)},
        tag="check_gs_contact_truth",
        label="match",
    )
    if res.failures:
        raise RuntimeError(f"check failures: {res.failures}")
    rows = pd.concat([pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.shard_keys], ignore_index=True)
    bad = {u: json.loads(m) for u, m in zip(rows["unit"], rows["mismatches"], strict=True) if json.loads(m)}
    extra = sorted(staged_units - set(rows["unit"]))
    if extra:
        bad.update({u: ["not listed by the loader"] for u in extra})
    return {
        "run_commit": prov.get("commit"),
        "run_tree_dirty": prov.get("dirty"),
        "n_units": len(rows),
        "mismatches": bad,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged gs_contact_truth frames == the live pining loader.")
    parser.add_argument("--data", type=Path, required=True, help="the materialized gs_contact_truth directory")
    parser.add_argument("--work-dir", type=Path, required=True, help="resume shards, outside the dataset")
    parser.add_argument("--cache-dir", type=Path, default=None, help="the pining cache")
    parser.add_argument("--out", type=Path, required=True, help="the report (JSON)")
    parser.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (provenance stamps it)")
    args = parser.parse_args()

    from scripts._provenance import git_provenance, require_clean_tree

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    report = run(args.data, work_dir=args.work_dir, provenance=prov, cache_dir=args.cache_dir)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    bad = report["mismatches"]
    print(f"staged == served on {report['n_units'] - len(bad)} of {report['n_units']} units")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
