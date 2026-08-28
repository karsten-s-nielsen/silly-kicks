"""KDE default resolution: ``predict_density(kde_backend="auto")`` picks the fastest EXACT backend.

``"auto"`` (the new default, was ``"vectorized"``) resolves to ``cpu-numba`` when the ``[numba]``
extra is installed (exact within 1e-9 -- the golden already runs ``cpu-numba``), else ``vectorized``.
``fft`` stays an explicit opt-in (approximate on the raw grid). Explicit backends are unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel


def _fit(n=400):
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((n, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, n).astype(float)
    m = GhostGkModel(n_estimators=30)
    m.fit(X, pd.DataFrame({"gk_x": rng.uniform(2, 20, n), "gk_y": rng.uniform(25, 45, n)}))
    return m, X


def test_auto_resolves_to_cpu_numba_when_numba_present(monkeypatch):
    pytest.importorskip("numba")
    from silly_kicks.tracking import _ghost_gk as G

    m, X = _fit()
    seen = []
    real = G._kde_density_numba

    def spy(*a, **k):
        seen.append("cpu-numba")
        return real(*a, **k)

    monkeypatch.setattr(G, "_kde_density_numba", spy)
    m.predict_density(X.iloc[:2])  # DEFAULT (no kde_backend) -> must hit cpu-numba
    assert seen == ["cpu-numba", "cpu-numba"]


def test_auto_falls_back_to_vectorized_without_numba(monkeypatch):
    from silly_kicks.tracking import _ghost_gk as G

    m, X = _fit()
    monkeypatch.setattr(G, "_HAS_GHOST_NUMBA", False)  # simulate no [numba]
    seen = []
    real = G._kde_density_vectorized
    monkeypatch.setattr(G, "_kde_density_vectorized", lambda *a, **k: (seen.append("vec"), real(*a, **k))[1])
    m.predict_density(X.iloc[:2])
    assert seen == ["vec", "vec"]


def test_explicit_backends_still_selectable():
    """Every explicit backend stays selectable -- fft/fft-cic remain an opt-in (not the default)."""
    m, X = _fit()
    for backend in ("vectorized", "scipy", "fft", "fft-cic"):
        out = m.predict_density(X.iloc[:1], kde_backend=backend)
        assert out[0].probabilities.sum() == pytest.approx(1.0, abs=1e-9)


def test_cpu_numba_backend_explicitly_selectable():
    pytest.importorskip("numba")
    m, X = _fit()
    out = m.predict_density(X.iloc[:1], kde_backend="cpu-numba")
    assert out[0].probabilities.sum() == pytest.approx(1.0, abs=1e-9)


def test_auto_is_in_the_documented_backend_set():
    """The default value "auto" is listed in the predict_density docstring's value set (spec §5.4)."""
    doc = GhostGkModel.predict_density.__doc__ or ""
    assert '"auto"' in doc and 'default "auto"' in doc
