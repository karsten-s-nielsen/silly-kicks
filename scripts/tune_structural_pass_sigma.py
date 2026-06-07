"""Reproduce the TF-45 SGM sigma choice (frozen default 15.0) on real WC2022 GS data.

Owner-gated (needs PINING_FOR_THE_DATA_TOKEN). Emits the sigma-sweep table used to
justify sigma=15 and the e2e SGM-conditioning ceiling (max|SGM|<=200, p99<=20).

Usage: python scripts/tune_structural_pass_sigma.py [n_matches]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from scripts._loader_pining import load_matches
from silly_kicks.tracking._structural_pass import _structural_pass_core
from silly_kicks.tracking.utils import link_actions_to_frames

SIGMAS = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0]
FINAL_THIRD_X = 70.0


def main(n_matches: int) -> None:
    records: list[dict] = []
    for _prov, _mid, actions, frames, home in load_matches(
        providers=["gradientsports"], max_per_provider=n_matches, tracking_limit=None
    ):
        home_id = int(home)
        passes = actions[(actions["type_id"] == 0) & (actions["result_id"] == 1)].copy()
        if passes.empty:
            continue
        pointers, _ = link_actions_to_frames(passes, frames, on_low_coverage="ignore")
        ptr = pointers.set_index("action_id")["frame_id"]
        outf = frames[(~frames["is_ball"].astype(bool)) & (~frames["is_goalkeeper"].astype(bool))]
        fg = outf.groupby(["period_id", "frame_id"])
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
            rec = {"enters_third": (sx < FINAL_THIRD_X) and (ex >= FINAL_THIRD_X)}
            for sig in SIGMAS:
                _, sgm, _ = _structural_pass_core(d, (sx, sy), (ex, ey), sig)
                rec[f"sgm_{sig}"] = sgm
            records.append(rec)

    df = pd.DataFrame(records)
    label = df["enters_third"].to_numpy(bool)
    from sklearn.metrics import roc_auc_score

    print(f"passes={len(df)} base_rate_enters_third={label.mean():.3f}")
    print(f"{'sigma':>6} {'sgmAUC':>7} {'p99abs':>9} {'maxabs':>9}")
    for sig in SIGMAS:
        s = df[f"sgm_{sig}"].to_numpy(float)
        m = np.isfinite(s)
        auc = roc_auc_score(label[m], s[m]) if label[m].any() and not label[m].all() else float("nan")
        print(f"{sig:>6.1f} {auc:>7.4f} {np.percentile(np.abs(s[m]), 99):>9.3f} {np.abs(s[m]).max():>9.1f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
