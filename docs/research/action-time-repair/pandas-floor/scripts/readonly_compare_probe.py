"""Measure which pandas compares a read-only object array with a scalar (the C gate) and what `.to_numpy()` of an
`Int64` column with NA gives (the D helper). Run it once per pandas version in a throwaway environment, e.g.
`pandas==2.1.1 numpy<2` up to `pandas==2.3.3`.

Recorded 2026-11-19 (py3.11, numpy 1.26.4):

    2.1.1 FAIL buffer source array is read-only   Int64-with-NA to_numpy(): object [1, <NA>, 3]
    2.1.4 FAIL buffer source array is read-only   Int64-with-NA to_numpy(): object [1, <NA>, 3]
    2.2.0 FAIL buffer source array is read-only   Int64-with-NA to_numpy(): float64 [1.0, nan, 3.0]
    2.2.1 OK [False, True, False]
    2.2.2 OK [False, True, False]
    2.2.3 OK [False, True, False]                 Int64-with-NA to_numpy(): float64 [1.0, nan, 3.0]
    2.3.3 OK [False, True, False]                 Int64-with-NA to_numpy(): float64 [1.0, nan, 3.0]

So `silly_kicks._pandas_compat.PANDAS_COMPARES_READ_ONLY_OBJECT` is `pandas >= 2.2.1`, and `masked_to_numpy`
reproduces the pandas 2.2 `to_numpy()` result.
"""

import numpy as np
import pandas as pd

a = np.array(["1", "2", "3"], dtype=object)
a.flags.writeable = False
try:
    print(pd.__version__, "OK", (pd.Series(a) == "2").tolist())
except ValueError as exc:
    print(pd.__version__, "FAIL", exc)
m = pd.Series([1, None, 3], dtype="Int64").to_numpy()
print("  Int64-with-NA to_numpy():", m.dtype, m.tolist())
