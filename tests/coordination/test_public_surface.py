"""TF-58 Task 18: the coordination package's public surface is exactly the spec §7.2 list."""

from __future__ import annotations

import importlib

import silly_kicks.coordination as C

# The spec §7.2 public surface, written out independently of the package so the test fails if __all__ drifts.
_EXPECTED = {
    # config
    "CoordinationParams",
    # window builders
    "period_windows",
    "possession_windows_from_actions",
    "possession_windows_from_frames",
    # signals
    "build_coordination_signals",
    "CoordinationSignals",
    # family computes
    "compute_relative_phase",
    "compute_cross_correlation",
    "compute_vector_coding",
    "compute_coherence",
    "compute_spectral",
    "compute_cluster_phase",
    "compute_relative_stretch",
    # raw series + orchestrator
    "compute_coordination_series",
    "compute_team_coordination",
    "CoordinationResult",
    # report
    "CoordinationReport",
    "CoordinationCoverageWarning",
    # pair catalogue
    "PairSpec",
    "DEFAULT_PAIRS",
    "DEFAULT_LEVELS",
    # vocabularies
    "COORD_SOURCE_VALUES",
    "COORD_SURROGATE_SOURCE_VALUES",
    "COORD_LEVELS",
    "COORD_SIGNALS",
    "COORD_WINDOW_COLUMNS",
    # per-family contract constants (7 families x KEYS/METRIC_COLUMNS/COLUMNS)
    *(
        f"COORDINATION_{fam}_{suffix}"
        for fam in ("PAIR", "PAIR_PHASE", "SPECTRAL", "CLUSTER_TEAM", "CLUSTER_PLAYER", "TEAM_SYNC", "RSI")
        for suffix in ("KEYS", "METRIC_COLUMNS", "COLUMNS")
    ),
}


def test_all_is_exact():
    assert set(C.__all__) == _EXPECTED
    assert len(C.__all__) == len(set(C.__all__)), "duplicate name in __all__"


def test_every_public_name_importable():
    for name in C.__all__:
        assert hasattr(C, name), f"{name} in __all__ but not importable from the package"


def test_star_import_matches_all():
    ns: dict[str, object] = {}
    exec("from silly_kicks.coordination import *", ns)  # noqa: S102
    exported = {k for k in ns if not k.startswith("__")}
    assert exported == set(C.__all__)


def test_no_private_leak_in_public_names():
    assert all(not name.startswith("_") for name in C.__all__)


def test_reimport_is_stable():
    mod = importlib.reload(C)
    assert set(mod.__all__) == _EXPECTED
