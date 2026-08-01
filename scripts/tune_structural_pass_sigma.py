"""Reproduce the TF-45 SGM sigma choice (frozen default 15.0) on real WC2022 GS data.

Owner-gated (needs PINING_FOR_THE_DATA_TOKEN). Emits the sigma-sweep table used to
justify sigma=15 and the e2e SGM-conditioning ceiling (max|SGM|<=200, p99<=20).

Usage: python scripts/tune_structural_pass_sigma.py [n_matches] [--shard-dir DIR]

The per-match sweep is resumable on the shared `scripts/_driver` seam: one parquet shard per
match, so a crash part-way through an owner-tier GS corpus resumes instead of restarting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._loader_pining import load_matches
from silly_kicks.tracking._structural_pass import _structural_pass_core
from silly_kicks.tracking.utils import link_actions_to_frames

SIGMAS = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0]
FINAL_THIRD_X = 70.0


def _match_records(item) -> pd.DataFrame | None:
    """One match's per-pass sigma sweep, as a tidy frame.

    ``None`` when the match contributes no scoreable pass -- which `write_shard` still records as
    an EMPTY shard ("ran, produced nothing"), so a resume does not re-derive the same verdict.
    """
    _prov, _mid, actions, frames, home = item
    home_id = int(home)  # type: ignore[reportArgumentType]
    passes = actions[(actions["type_id"] == 0) & (actions["result_id"] == 1)].copy()
    if passes.empty:
        return None
    pointers, _ = link_actions_to_frames(passes, frames, on_low_coverage="ignore")
    ptr = pointers.set_index("action_id")["frame_id"]
    outf = frames[(~frames["is_ball"].astype(bool)) & (~frames["is_goalkeeper"].astype(bool))]
    fg = outf.groupby(["period_id", "frame_id"])
    records: list[dict] = []
    for _, row in passes.iterrows():
        fid = ptr.get(row["action_id"])
        if pd.isna(fid):
            continue
        try:
            fr = fg.get_group((int(row["period_id"]), int(fid)))
        except KeyError:
            continue
        opp = fr[fr["team_id"].astype(str) != str(row["team_id"])]
        dx = opp["x"].to_numpy(float)
        dy = opp["y"].to_numpy(float)
        ok = np.isfinite(dx) & np.isfinite(dy)
        dx, dy = dx[ok], dy[ok]
        if dx.size == 0:
            continue
        if int(row["team_id"]) != home_id:
            dx, dy = 105.0 - dx, 68.0 - dy
        d = np.column_stack([dx, dy])
        sx, sy, ex, ey = (float(row[c]) for c in ("start_x", "start_y", "end_x", "end_y"))
        rec: dict[str, object] = {"enters_third": (sx < FINAL_THIRD_X) and (ex >= FINAL_THIRD_X)}
        for sig in SIGMAS:
            _, sgm, _ = _structural_pass_core(d, (sx, sy), (ex, ey), sig)
            rec[f"sgm_{sig}"] = sgm
        records.append(rec)
    return pd.DataFrame(records) if records else None


def report(df: pd.DataFrame) -> None:
    """Print the sweep table. Pure of I/O beyond stdout, so it is callable on any assembled frame."""
    from sklearn.metrics import roc_auc_score

    label = df["enters_third"].to_numpy(bool)
    print(f"passes={len(df)} base_rate_enters_third={label.mean():.3f}")
    print(f"{'sigma':>6} {'sgmAUC':>7} {'p99abs':>9} {'maxabs':>9}")
    for sig in SIGMAS:
        s = df[f"sgm_{sig}"].to_numpy(float)
        m = np.isfinite(s)
        auc = roc_auc_score(label[m], s[m]) if label[m].any() and not label[m].all() else float("nan")
        print(f"{sig:>6.1f} {auc:>7.4f} {np.percentile(np.abs(s[m]), 99):>9.3f} {np.abs(s[m]).max():>9.1f}")


def main(n_matches: int = 3, shard_dir: str | None = None) -> None:
    from scripts._driver import for_each, shard_path

    # STREAMED, not inverted onto `select_match_ids`: the per-item cost here is the linking plus a
    # seven-sigma sweep over every completed pass in the match, so the shard check already skips
    # the expensive half. See `derive_opengoal_range` for the same judgement and its reasoning.
    res = for_each(
        load_matches(providers=["gradientsports"], max_per_provider=n_matches, tracking_limit=None),
        key=lambda item: (str(item[0]), str(item[1])),
        work=_match_records,
        shard_root=Path(shard_dir or "tune_structural_pass_sigma_shards") / "shards",
        # What determines a shard's CONTENT: the sigma grid swept, the final-third boundary that
        # labels each pass, and the kernel that scores it. `--n-matches` only chooses how many
        # matches are walked, and the key already separates one match's shard from another's.
        token_inputs={
            "sigmas": list(SIGMAS),
            "final_third_x": FINAL_THIRD_X,
            "kernel": "_structural_pass_core",
        },
        tag="sigma_sweep",
        label="match",
    )
    if res.failures:
        raise RuntimeError(f"{len(res.failures)} match(es) failed: {res.failures}. Re-run to retry only them.")

    # Combined from THIS PASS'S keys, not `_driver.reconcile`: that helper's whole-generation read
    # requires a partition surface (see its docstring), and this driver has none -- so it would
    # otherwise fold in matches from a wider earlier run over the same --shard-dir.
    frames = [f for f in (pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.keys) if len(f)]
    if not frames:
        raise ValueError("no scoreable passes in the corpus -- refusing to print a sweep over nothing")
    report(pd.concat(frames, ignore_index=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TF-45 SGM sigma sweep over real WC2022 GS matches")
    ap.add_argument("n_matches", nargs="?", type=int, default=3)
    ap.add_argument("--shard-dir", default=None, help="dir for the resumable per-match shards")
    _args = ap.parse_args()
    main(_args.n_matches, _args.shard_dir)
