"""What the library needs to behave the same on its declared pandas floor as on a later pandas (R75, R79).

The declared floor is ``pandas>=2.1.1`` (``pyproject.toml``); the ``pandas-floor`` CI job runs the suite
there. Two pandas changes after 2.1.1 reach library code paths, and this module holds the one answer
to each, so that no call site carries its own version check:

- **Read-only object buffers.** Before pandas 2.2.1, an object-dtype ``Series == scalar`` goes through
  a Cython routine that needs a writeable buffer and raises "buffer source array is read-only".
  :data:`PANDAS_COMPARES_READ_ONLY_OBJECT` is the gate; :mod:`silly_kicks.id_compat` copies such an input
  only when the gate is ``False``.
- **Masked numeric ``to_numpy()``.** From pandas 2.2, ``.to_numpy()`` of a nullable integer column with
  NA gives float64 with NaN; before, an object array of ``pd.NA``, which raises "boolean value of NA is
  ambiguous" inside a ``&``. :func:`masked_to_numpy` gives the pandas 2.2 result on every version.

Neither changes a value or a dtype on pandas 2.2.1 or later (``tests/test_pandas_floor_compat.py``).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

#: ``True`` from pandas 2.2.1, the first release whose object-dtype ``Series == scalar`` accepts a
#: read-only buffer. Measured 2026-11-19 (py3.11, numpy 1.26.4): 2.1.1, 2.1.4 and 2.2.0 raise
#: "buffer source array is read-only"; 2.2.1, 2.2.2, 2.2.3 and 2.3.3 do not. Set once, at import.
PANDAS_COMPARES_READ_ONLY_OBJECT: Final[bool] = np.lib.NumpyVersion(pd.__version__) >= "2.2.1"


def masked_to_numpy(s: pd.Series) -> np.ndarray:
    """``s.to_numpy()`` as pandas 2.2 and later give it, on every supported pandas.

    A nullable integer column gives float64 with NaN when it holds NA, else its numpy integer dtype; a
    nullable float column gives its numpy float dtype with NaN. Every other dtype is returned exactly as
    ``s.to_numpy()`` returns it. On pandas 2.2 and later the result equals ``s.to_numpy()``.

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks._pandas_compat import masked_to_numpy
    >>> masked_to_numpy(pd.Series([1, None, 3], dtype="Int64")).tolist()
    [1.0, nan, 3.0]
    >>> masked_to_numpy(pd.Series([1, 2], dtype="Int64")).dtype
    dtype('int64')
    """
    values = s.array
    if isinstance(values, pd.arrays.IntegerArray):
        if s.hasnans:
            return s.to_numpy(dtype=np.float64, na_value=np.nan)
        return s.to_numpy(dtype=np.dtype(values.dtype.name.lower()))  # Int64 -> int64, UInt8 -> uint8
    if isinstance(values, pd.arrays.FloatingArray):
        return s.to_numpy(dtype=np.dtype(values.dtype.name.lower()), na_value=np.nan)  # Float32 -> float32
    return s.to_numpy()
