"""Regenerate the ghost-GK KDE query-feature fixture.

As of 4.54.0 / ADR-044 this fixture carries ONLY the deterministic query feature set
(``features`` / ``feature_cols``) — a seeded, version-independent set of 26-dim inputs. The KDE
backend-parity tests (``test_golden_continuous`` / ``_discrete_mode`` / ``_fft_cic_scalars``)
compute their reference at RUNTIME (the closed-form ``vectorized`` backend on a fresh fit), so no
frozen density oracle is stored: a fitted-model oracle was not portable across sklearn/numpy
versions once artifacts became parameters-only (the bundled model can no longer serve density).

Run: .venv/Scripts/python.exe scripts/gen_ghost_gk_kde_golden.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES

OUT = Path(__file__).parent.parent / "tests/tracking/fixtures/ghost_gk_kde_golden.npz"


def _seeded_features(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    X = pd.DataFrame(rng.standard_normal((n, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, n).astype(float)
    return X


def main() -> None:
    X = _seeded_features()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        features=X.values,
        feature_cols=np.array(X.columns, dtype=object),
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.2f} KB) — query features only, no frozen oracle")
    print(f"gen-env: py={sys.version.split()[0]} numpy={np.__version__}")


if __name__ == "__main__":
    main()
