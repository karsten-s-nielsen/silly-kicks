"""Read-only: the wait from the previous possession's end to the penalty kick, from the parquet file that
sc_penalty_check.py writes (--table; column wait_s; rows with a kick found). Output: stdout."""
import argparse

import pandas as pd

ap = argparse.ArgumentParser(description="The wait before a SkillCorner penalty kick.")
ap.add_argument("--table", required=True, help="the parquet file of sc_penalty_check.py")
d = pd.read_parquet(ap.parse_args().table)
f = d[d["found"] == True]  # noqa: E712
print("penalties with a kick found:", len(f))
print("wait_s min / q10 / median / max:", round(f.wait_s.min(), 1), round(f.wait_s.quantile(0.1), 1),
      round(f.wait_s.median(), 1), round(f.wait_s.max(), 1))
print("DONE")
