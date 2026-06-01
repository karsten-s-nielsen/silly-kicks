"""Regenerate the ghost-GK KDE golden fixture (Phase 0a).

Frozen against scipy 1.15 / numpy 2.x (pinned). A CI bump past those can shift scipy's
Cholesky / gaussian_kde reduction order enough to fail the 1e-7 golden cross-environment —
regenerate and review the diff when bumping either dependency.

Run: .venv/Scripts/python.exe scripts/gen_ghost_gk_kde_golden.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel

OUT = Path(__file__).parent.parent / "tests/tracking/fixtures/ghost_gk_kde_golden.npz"


def _seeded_features(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    X = pd.DataFrame(rng.standard_normal((n, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, n).astype(float)
    return X


def main() -> None:
    import scipy

    model = GhostGkModel.from_variant("default")
    X = _seeded_features()
    densities = model.predict_density(X, kde_backend="scipy")
    probs = np.stack([d.probabilities for d in densities])  # (n, 60, 64)
    mode_x = np.array([d.mode_x for d in densities])
    mode_y = np.array([d.mode_y for d in densities])
    mean_x = np.array([d.mean_x for d in densities])
    mean_y = np.array([d.mean_y for d in densities])
    spread = np.array([d.spread for d in densities])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        features=X.values,
        feature_cols=np.array(X.columns, dtype=object),
        probs=probs,
        mode_x=mode_x,
        mode_y=mode_y,
        mean_x=mean_x,
        mean_y=mean_y,
        spread=spread,
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
    print(
        f"gen-env: py={sys.version.split()[0]} numpy={np.__version__} scipy={scipy.__version__}"
    )


if __name__ == "__main__":
    main()
