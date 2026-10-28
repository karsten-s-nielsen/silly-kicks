"""Synthetic fit matrices at the models' real feature width for the ADR-128 thread-identity gate (T4).

NEVER owner-tier raw positions (B16; redistribution policy): every value is seeded numpy Gaussian noise shaped
to the model's real feature count, so the sklearn/xgboost parallel fit path engages exactly as it does on real
data while nothing redistributable leaks. The identity test proves byte-identity across thread counts; it does
not need real signal.

Rows default to **12000 (> 10_000)** so the sklearn HistGradientBoostingRegressor takes its production
``early_stopping="auto"`` path (sklearn 1.9.0 ``gradient_boosting.py:533`` enables ES above 10_000 samples); a
sub-10_000 fixture would gate a different code path than production (B43).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Recorded next to any artifact a fixture feeds, so provenance is explicit (never a corpus id list).
PROVENANCE = "synthetic: seeded numpy Gaussian at the model's real feature width; no owner-tier/raw positions"

#: Default fixture rows: above sklearn's 10_000 early-stopping threshold so the gate takes the production path.
DEFAULT_ROWS = 12000


def make_synthetic_ghost_fit(*, n_samples: int = DEFAULT_ROWS, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A synthetic GhostGkModel fit matrix: ``n_samples`` rows x the 26 faithful features, plus (gk_x, gk_y)."""
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES

    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        rng.standard_normal((n_samples, len(GHOST_GK_FEATURE_NAMES))), columns=GHOST_GK_FEATURE_NAMES
    )
    labels = pd.DataFrame({"gk_x": rng.standard_normal(n_samples), "gk_y": rng.standard_normal(n_samples)})
    return features, labels


def make_synthetic_ghost_outfield_table(*, n_samples: int = DEFAULT_ROWS, seed: int = 0) -> pd.DataFrame:
    """A synthetic GhostOutfieldModel extracted table (features + target_x/target_y) for ``_fit_extracted``."""
    from silly_kicks.tracking._ghost_outfield import GHOST_OUTFIELD_FEATURE_NAMES

    rng = np.random.default_rng(seed)
    table = pd.DataFrame(
        rng.standard_normal((n_samples, len(GHOST_OUTFIELD_FEATURE_NAMES))), columns=GHOST_OUTFIELD_FEATURE_NAMES
    )
    table["target_x"] = rng.standard_normal(n_samples)
    table["target_y"] = rng.standard_normal(n_samples)
    return table


def make_synthetic_xshot_fit(*, n_samples: int = DEFAULT_ROWS, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """A synthetic XShotOccurrenceModel fit matrix: ``n_samples`` rows x the 27 faithful features + binary y."""
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL

    return _xgb_fit(XSHOT_FEATURE_NAMES_FAITHFUL, n_samples=n_samples, seed=seed)


def make_synthetic_xcross_fit(*, n_samples: int = DEFAULT_ROWS, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """A synthetic XCrossAttemptModel fit matrix: ``n_samples`` rows x the faithful features + binary y."""
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL

    return _xgb_fit(XCROSS_FEATURE_NAMES_FAITHFUL, n_samples=n_samples, seed=seed)


def _xgb_fit(columns: list[str], *, n_samples: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(rng.standard_normal((n_samples, len(columns))), columns=columns)
    labels = pd.Series((rng.random(n_samples) < 0.3).astype(int))
    return features, labels


def assert_parallel_engages(*, n_samples: int, n_features: int, budget: int) -> None:
    """Non-vacuity precondition (B29, B r4): the thread-identity test is meaningful ONLY if the fixture makes
    the sklearn HistGB parallel loops distribute work across every one of ``budget`` threads. In pinned
    sklearn 1.9.0 the parallelism is TWO axes:

    - FEATURE axis (the histogram SUM + split search): ``prange(n_allowed_features)`` at
      ``histogram.pyx:170,270`` and ``splitting.pyx:519`` -- each thread sums ONE feature in sample order.
      Idle unless ``n_features >= budget``.
    - ROW axis (data movement): the gather ``histogram.pyx:160,164`` over ``n_samples`` and the split-apply
      ``splitting.pyx:353,395`` that partitions ``n_samples`` across threads. Idle unless ``n_samples >= budget``.

    A fixture failing either axis leaves threads idle, so a byte-identity assertion would pass vacuously.
    This is NOT the effective thread count (``_openmp_effective_n_threads() == omp_get_max_threads()`` is OMP
    config, fixture-independent). Raises ``AssertionError`` naming the idle axis.
    """
    assert n_features >= budget, (
        f"feature axis vacuous: n_features={n_features} < budget={budget} "
        "(the feature-parallel loops histogram.pyx:170,270 / splitting.pyx:519 leave threads idle)"
    )
    assert n_samples >= budget, (
        f"row axis vacuous: n_samples={n_samples} < budget={budget} "
        "(the row-parallel loops histogram.pyx:160,164 / splitting.pyx:353,395 leave threads idle)"
    )
