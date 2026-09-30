"""Share the F1b design matrix across study workers via a file-backed mmap (spec 6, 5a).

The ~15 F1b HPO studies (3 candidates x k folds) all consume the SAME ``(X, y, groups)``.
Run serially in one process they share one in-memory ``X``; fan them out to N worker
processes and each would reload/reparse a full copy, multiplying RAM by N -- exactly the
pressure that has OOM-killed the DGX. This module persists ``X`` once as an ``.npy`` the
workers open ``mmap_mode="r"``, so the OS page cache backs one physical copy shared across
processes (Task-5 probe: pandas 2.3.3 / numpy 2.x keep ``pd.DataFrame(mm, columns, copy=False)``
memmap-backed, dtypes + columns preserved, and a booster fit on it is byte-identical).

Consumer contract: the study path needs a **pandas DataFrame with the named feature columns**
(``X.iloc[...]``, boolean-mask ``X[mask]``, and ``list(X.columns)`` -> ``booster.feature_names``,
_xshot_occurrence.py:482). A bare numpy array would AttributeError and drop the names, so the
roundtrip preserves both. No consumer mutates ``X`` in place, so the read-only mmap is safe.

The production design matrix is uniform float64 (27 geometric features), which takes the
zero-copy single-``.npy`` path. A mixed-dtype matrix cannot share one ``.npy`` without a
per-column cast (a hidden copy that also corrupts dtypes), so it falls back to an exact
parquet reconstruction -- a copy, so the 5a win there is avoiding the re-BUILD, not RAM
sharing; ``assert_frame_equal(check_exact=True)`` guards both paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_META = "design_matrix.json"


def persist_design_matrix(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    path: str | Path,
) -> Path:
    """Persist ``(X, y, groups)`` under ``path`` for zero-copy mmap sharing.

    ``X`` values go to an mmap-able ``X.npy`` when every column shares one dtype (the
    production case), else to an exact ``X.parquet`` (copy-fallback). Column names and
    per-column dtype strings ride in a ``design_matrix.json`` sidecar. ``y``/``groups``
    are small and stored as plain ``.npy`` (object arrays -- e.g. str game ids -- pickle).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    columns = list(X.columns)
    dtypes = [str(X[c].dtype) for c in columns]
    uniform = len(set(dtypes)) == 1
    meta = {"columns": columns, "dtypes": dtypes, "uniform": uniform}
    if uniform:
        np.save(path / "X.npy", X.to_numpy())  # single dtype -> mmap-able, zero-copy on load
    else:
        X.to_parquet(path / "X.parquet")  # parquet preserves per-column dtypes exactly
    (path / _META).write_text(json.dumps(meta), encoding="utf-8")
    # Pickle-free (house style + trusted self-produced scratch): y is numeric; object (str)
    # groups become fixed-width unicode ('<U'), which np.save writes without pickle.
    np.save(path / "y.npy", np.asarray(y))
    g = np.asarray(groups)
    np.save(path / "groups.npy", g.astype(str) if g.dtype == object else g)
    return path


def load_design_matrix(
    path: str | Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Reconstruct ``(X, y, groups)`` from ``path``, mmap-shared when persisted uniform.

    For the uniform path ``X`` is a DataFrame over a read-only memmap (``X.values`` is
    memmap-backed -> shared across processes); for the mixed path it is read from parquet
    (an in-process copy). Either way columns + dtypes match what was persisted exactly.
    """
    path = Path(path)
    meta = json.loads((path / _META).read_text(encoding="utf-8"))
    if meta["uniform"]:
        mm = np.load(path / "X.npy", mmap_mode="r")
        X = pd.DataFrame(mm, columns=meta["columns"], copy=False)
    else:
        X = pd.read_parquet(path / "X.parquet")
        X = X[meta["columns"]]  # restore column order
    y = np.load(path / "y.npy", allow_pickle=False)
    groups = np.load(path / "groups.npy", allow_pickle=False)
    return X, y, groups
