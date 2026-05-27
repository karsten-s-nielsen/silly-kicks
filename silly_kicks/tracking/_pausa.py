"""PAUSA scoring -- Passing Ability Under Spatiotemporal Awareness.

PAUSA decomposes pass quality into two orthogonal components:
- **Temporal judgment**: Was the pass released at the peak OBSO moment?
  ``temporal = actual_obso / peak_obso``
- **Spatial selection**: Was the target the best available receiver?
  ``spatial = actual_obso / optimal_obso``
- **PAUSA composite**: ``pausa = temporal * spatial``

All values are clamped to [0, 1]. Division by zero (peak=0 or optimal=0)
yields 0 -- if there was no scoring opportunity, the pass cannot be evaluated.

See NOTICE for full bibliographic citations.

References
----------
Lee, I. H. (2026). "Valuing passes in football using PAUSA (Pass Utility
using Space Analysis)." arXiv:2506.09349.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_pausa(
    actual_obso: float,
    peak_obso: float,
    optimal_obso: float,
) -> dict[str, float]:
    """PAUSA decomposition: temporal judgment x spatial selection.

    Parameters
    ----------
    actual_obso : float
        OBSO at the actual pass target at the event frame.
    peak_obso : float
        Maximum OBSO at target across the frame window.
    optimal_obso : float
        Maximum OBSO across all teammate positions at event frame.

    Returns
    -------
    dict[str, float]
        ``pausa_temporal``, ``pausa_spatial``, ``pausa_composite``.

    Examples
    --------
    >>> compute_pausa(0.5, 1.0, 0.8)
    {'pausa_temporal': 0.5, 'pausa_spatial': 0.625, 'pausa_composite': 0.3125}
    """
    temporal = actual_obso / peak_obso if peak_obso > 0 else 0.0
    spatial = actual_obso / optimal_obso if optimal_obso > 0 else 0.0
    temporal = float(np.clip(temporal, 0.0, 1.0))
    spatial = float(np.clip(spatial, 0.0, 1.0))
    return {
        "pausa_temporal": temporal,
        "pausa_spatial": spatial,
        "pausa_composite": temporal * spatial,
    }


def compute_pausa_batch(actions: pd.DataFrame) -> pd.DataFrame:
    """Vectorized PAUSA on DataFrame with obso_actual/peak/optimal columns.

    Parameters
    ----------
    actions : pd.DataFrame
        Must contain ``obso_actual``, ``obso_peak``, ``obso_optimal`` columns.

    Returns
    -------
    pd.DataFrame
        Input augmented with ``pausa_temporal``, ``pausa_spatial``,
        ``pausa_composite`` columns.

    Raises
    ------
    ValueError
        If required OBSO columns are missing.

    Examples
    --------
    >>> enriched = compute_pausa_batch(actions_with_obso)
    >>> enriched[["pausa_temporal", "pausa_spatial", "pausa_composite"]]
    """
    required = {"obso_actual", "obso_peak", "obso_optimal"}
    missing = required - set(actions.columns)
    if missing:
        raise ValueError(
            f"compute_pausa_batch: missing required columns {missing}. Run add_obso() first to produce OBSO columns."
        )

    result = actions.copy()

    if result.empty:
        result["pausa_temporal"] = pd.Series(dtype=np.float64)
        result["pausa_spatial"] = pd.Series(dtype=np.float64)
        result["pausa_composite"] = pd.Series(dtype=np.float64)
        return result

    actual = result["obso_actual"].to_numpy(dtype=np.float64)
    peak = result["obso_peak"].to_numpy(dtype=np.float64)
    optimal = result["obso_optimal"].to_numpy(dtype=np.float64)

    temporal = np.zeros_like(actual, dtype=np.float64)
    np.divide(actual, peak, out=temporal, where=peak > 0)

    spatial = np.zeros_like(actual, dtype=np.float64)
    np.divide(actual, optimal, out=spatial, where=optimal > 0)

    temporal = np.clip(temporal, 0.0, 1.0)
    spatial = np.clip(spatial, 0.0, 1.0)

    result["pausa_temporal"] = temporal
    result["pausa_spatial"] = spatial
    result["pausa_composite"] = temporal * spatial

    return result
