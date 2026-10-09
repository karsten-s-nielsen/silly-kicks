"""TF-58 Task 18: every public coordination builder/compute is PURE -- it never mutates its inputs.

Frame-consuming entry points are checked against deep copies of ``frames``/``actions``/``windows``/``links``;
the family computes and the raw-series primitive are checked against a pickle fingerprint of the shared
``signals`` object (the orchestrator runs all seven computes over ONE signals, so a mutating compute would
corrupt its neighbours).
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from silly_kicks.coordination import (
    CoordinationParams,
    build_coordination_signals,
    compute_cluster_phase,
    compute_coherence,
    compute_coordination_series,
    compute_cross_correlation,
    compute_relative_phase,
    compute_relative_stretch,
    compute_spectral,
    compute_team_coordination,
    compute_vector_coding,
    period_windows,
    possession_windows_from_actions,
    possession_windows_from_frames,
)
from tests.coordination._fixtures import make_coordination_actions, make_coordination_match

pytestmark = [
    pytest.mark.filterwarnings("ignore:vx/vy columns not found"),
    pytest.mark.filterwarnings("ignore::silly_kicks.coordination.CoordinationCoverageWarning"),
]
_FAST = dataclasses.replace(CoordinationParams(), n_surrogates=5, welch_segment_s=20.0)


def _links_for(actions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"action_id": actions["action_id"].to_numpy(), "frame_id": actions.index.to_numpy()})


def _collect_arrays(obj: object, out: list[bytes]) -> None:
    """Walk a signals object and collect every numpy array's bytes+dtype+shape (mutation fingerprint)."""
    if isinstance(obj, np.ndarray):
        out.append(obj.tobytes())
        out.append(f"{obj.dtype}{obj.shape}".encode())
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for fld in dataclasses.fields(obj):
            _collect_arrays(getattr(obj, fld.name), out)
    elif isinstance(obj, Mapping):
        for v in obj.values():
            _collect_arrays(v, out)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _collect_arrays(x, out)


def _fingerprint(signals) -> str:
    out: list[bytes] = []
    _collect_arrays(signals, out)
    return hashlib.blake2b(b"".join(out)).hexdigest()


def _assert_frames_unmutated(**named: pd.DataFrame):
    """Return a checker that asserts each named frame equals a deep copy taken now."""
    snaps = {k: v.copy(deep=True) for k, v in named.items()}

    def _check():
        for k, before in snaps.items():
            pd.testing.assert_frame_equal(named[k], before, obj=k)

    return _check


def test_window_builders_are_pure():
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec")
    a = make_coordination_actions(f)
    links = _links_for(a)
    check = _assert_frames_unmutated(frames=f, actions=a, links=links)
    period_windows(f)
    possession_windows_from_actions(a, f, links=links)
    possession_windows_from_frames(f)
    check()


def test_build_coordination_signals_is_pure():
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec")
    a = make_coordination_actions(f)
    w = period_windows(f)
    links = _links_for(a)
    check = _assert_frames_unmutated(frames=f, actions=a, windows=w, links=links)
    build_coordination_signals(f, windows=w, params=_FAST, actions=a, links=links)
    check()


def test_orchestrator_is_pure():
    f = make_coordination_match(seconds=150.0, hz=10.0, provider="sportec")
    a = make_coordination_actions(f)
    w = period_windows(f)
    links = _links_for(a)
    check = _assert_frames_unmutated(frames=f, actions=a, windows=w, links=links)
    compute_team_coordination(f, actions=a, windows=w, params=_FAST, links=links)
    check()


def test_family_computes_do_not_mutate_signals():
    f = make_coordination_match(seconds=180.0, hz=10.0, provider="sportec")
    signals = build_coordination_signals(f, windows=period_windows(f), params=_FAST)
    fingerprint = _fingerprint(signals)
    for fn in (
        compute_relative_phase,
        compute_cross_correlation,
        compute_vector_coding,
        compute_coherence,
        compute_spectral,
        compute_cluster_phase,
        compute_relative_stretch,
    ):
        fn(signals)
        assert _fingerprint(signals) == fingerprint, f"{fn.__name__} mutated its signals input"


def test_series_does_not_mutate_signals():
    f = make_coordination_match(seconds=180.0, hz=10.0, provider="sportec")
    signals = build_coordination_signals(f, windows=period_windows(f), params=_FAST)
    fingerprint = _fingerprint(signals)
    for kind in ("relative_phase", "coupling_angle", "cluster_amplitude"):
        compute_coordination_series(signals, kind=kind)
        assert _fingerprint(signals) == fingerprint, f"compute_coordination_series({kind}) mutated its signals input"
