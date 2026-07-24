"""TF-7 xfns cache -- cross-family structural perf guard.

Spy the dominant pitch-control primitive (``compute_pitch_control``, the surface
``cache.surface()`` calls on a MISS -- confirmed by the cross-family reuse test in
``test_pitch_control_cache.py``). With a shared cache, a SECOND family reuses the first
family's canonical surfaces instead of recomputing them.

Deterministic call-count guard (no wall-clock assertion), per the repo's structural perf harness.
"""

from __future__ import annotations

from silly_kicks.tracking.features import obso_xfns, pitch_control_xfns
from silly_kicks.tracking.pitch_control import (
    PitchControlCache,
    _cache,  # where cache.surface() resolves compute_pitch_control
)
from tests._perf_structural import call_counter
from tests.tracking.test_pitch_control_cache import _pc_actions, _pc_frames

_HTID = 1  # _pc_actions() team


def test_second_FAMILY_over_shared_cache_recomputes_nothing(monkeypatch, fitted_xt):
    frames, actions = _pc_frames(), _pc_actions()
    gs = [actions]

    # Spy compute_pitch_control where the cache module RESOLVES it (call_counter's docstring).
    calls = call_counter(monkeypatch, _cache, "compute_pitch_control")

    # (a) FAMILY 2 (obso) with its OWN fresh cache MUST hit the primitive (>0) -- proves it goes
    #     through cache.surface, so the zero-additional assertion below cannot pass vacuously.
    obso_xfns(
        home_team_id=_HTID, pitch_control_method="voronoi", xt=fitted_xt, pitch_control_cache=PitchControlCache()
    )[0](gs, frames)
    assert calls["n"] > 0, "obso must go through compute_pitch_control (else the cache-hit test is vacuous)"

    # (b) FAMILY 1 pre-populates a SHARED cache; FAMILY 2 over the SAME cache computes ZERO additional.
    shared = PitchControlCache()
    pitch_control_xfns("voronoi", pitch_control_cache=shared)[0](gs, frames)
    baseline = calls["n"]
    obso_xfns(home_team_id=_HTID, pitch_control_method="voronoi", xt=fitted_xt, pitch_control_cache=shared)[0](
        gs, frames
    )
    assert calls["n"] == baseline, "family 2 recomputed surfaces instead of hitting the shared cache"


def test_shared_cache_holds_exactly_one_canonical_surface(fitted_xt):
    """Sanity: pitch_control + obso on the SAME frame/team/method share ONE canonical surface."""
    frames, actions = _pc_frames(), _pc_actions()
    gs = [actions]
    shared = PitchControlCache()
    pitch_control_xfns("voronoi", pitch_control_cache=shared)[0](gs, frames)
    obso_xfns(home_team_id=_HTID, pitch_control_method="voronoi", xt=fitted_xt, pitch_control_cache=shared)[0](
        gs, frames
    )
    assert len(shared) == 1
