"""Read-only (spec r11, the derived-row key).

(1) On a stage (--stage): are the SkillCorner derived rows unique on (game_id, original_event_id, type_id)?
(2) On the cached matches (--cache-dir; 889 on 2026-11-18) (keys on b; a period-end "after" marker keys on a): a gap (a, b) between two possessions holds at most one derived restart or penalty row
per kind; count the "before" markers on the first possession of a period (no a).
Output: stdout.
"""
import argparse
import glob
import os
from multiprocessing import Pool

import pandas as pd

STAGE = CACHE = ""  # set from the arguments (the workers fork after that)
KINDS = ("corner", "free_kick", "goal_kick", "throw_in", "penalty")


def kind(s):
    s = str(s)
    for k in KINDS:
        if s in (k + "_for", k + "_against"):
            return k
    return ""


def one(m):
    f = glob.glob(f"{CACHE}/{m}/*_events.parquet")
    if not f:
        return None
    ev = pd.read_parquet(f[0], columns=["event_type", "frame_start", "period", "game_interruption_before",
                                        "game_interruption_after"])
    pp = ev[ev.event_type == "player_possession"].sort_values("frame_start", kind="mergesort").reset_index(drop=True)
    gib = pp.game_interruption_before.astype("string").fillna("").map(kind)
    gia = pp.game_interruption_after.astype("string").fillna("").map(kind)
    per = pp.period.to_numpy()
    no_a = sum(1 for i in range(len(pp)) if gib[i] and (i == 0 or per[i] != per[i - 1]))
    dup = 0
    for i in range(len(pp)):
        rows = []
        if gia[i]:
            rows.append(gia[i])
        if i + 1 < len(pp) and per[i + 1] == per[i] and gib[i + 1] and gib[i + 1] != gia[i]:
            rows.append(gib[i + 1])
        dup += len(rows) - len(set(rows))
    # b-key edge: an "after" marker with no next possession in the period (a period end) keys on a; it collides only
    # if the gap before a holds a row of the same kind (a "before" marker of that kind on a)
    # the gap before a holds a penalty row of any class: a "before" penalty marker on a (paired or "before"-only), or an
    # "after" penalty marker on the previous possession of the period ("after"-only); each such row takes a's id
    pe = [i for i in range(len(pp)) if gia[i] == "penalty" and (i + 1 >= len(pp) or per[i + 1] != per[i])]
    pe_collide_before = sum(1 for i in pe if gib[i] == "penalty")
    pe_collide_after = sum(1 for i in pe if i > 0 and per[i - 1] == per[i] and gia[i - 1] == "penalty" and gib[i] != "penalty")
    pe_collide = pe_collide_before + pe_collide_after
    pe_pen = sum(1 for i in range(len(pp)) if gia[i] == "penalty" and (i + 1 >= len(pp) or per[i + 1] != per[i]))
    return {"no_a_before": no_a, "dup_kind_in_gap": dup, "pe_collide": pe_collide, "pe_pen": pe_pen,
            "pe_cb": pe_collide_before, "pe_ca": pe_collide_after}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="The key of a derived SkillCorner row.")
    ap.add_argument("--stage", required=True, help="stage root (holds _actions/)")
    ap.add_argument("--cache-dir", required=True, help="raw provider cache root (<provider>/<match>/)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    STAGE, CACHE = os.path.join(args.stage, "_actions"), os.path.join(args.cache_dir, "skillcorner")
    n = d = 0
    for f in sorted(glob.glob(f"{STAGE}/skillcorner__*.parquet")):
        a = pd.read_parquet(f)
        if "action_provenance" not in a or "original_event_id" not in a:
            continue
        x = a[(a.action_provenance != "native") & a.original_event_id.notna()]
        n += len(x)
        d += int(x.duplicated(["game_id", "original_event_id", "type_id"]).sum())
    print("stage derived rows:", n, "| duplicates on (game_id, original_event_id, type_id):", d)
    with Pool(args.workers) as p:
        r = pd.DataFrame([x for x in p.map(one, sorted(os.listdir(CACHE))) if x])
    print("cache matches:", len(r), "| before markers on a period first possession:", int(r.no_a_before.sum()),
          "| gaps with two derived rows of one kind:", int(r.dup_kind_in_gap.sum()))
    print("period-end penalty markers (keyed on a):", int(r.pe_pen.sum()), "| period-end penalty sharing a's id with a penalty row of the gap before a (any class):", int(r.pe_collide.sum()),
          "(paired or before-only:", int(r.pe_cb.sum()), "; after-only:", int(r.pe_ca.sum()), ")")
    print("DONE")
