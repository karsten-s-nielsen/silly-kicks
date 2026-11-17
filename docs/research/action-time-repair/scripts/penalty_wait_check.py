"""Read-only: the wait from the previous possession's end to the penalty kick, from the supervisor's
~/dl-review-xdef/sc_penalty_check.parquet (column wait_s; rows with a kick found). Output: stdout."""
import pandas as pd

d = pd.read_parquet("/home/karsten/dl-review-xdef/sc_penalty_check.parquet")
f = d[d["found"] == True]  # noqa: E712
print("penalties with a kick found:", len(f))
print("wait_s min / q10 / median / max:", round(f.wait_s.min(), 1), round(f.wait_s.quantile(0.1), 1),
      round(f.wait_s.median(), 1), round(f.wait_s.max(), 1))
print("DONE")
