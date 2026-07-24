"""TF-7 xfns pitch-control cache: value-identity across every PC-consuming family.

A caller-injected :class:`PitchControlCache` threaded through the pitch-control-consuming
``*_xfns`` factories must be **byte-identical** to the ``None`` default (the cache memoises
canonical surfaces; it never changes a value). These tests parametrize over EVERY family so a
threading bug in any single factory is caught, and add the mis-keying failure mode a same-params
test cannot see (two DIFFERENT families with DIVERGENT pitch-control params sharing ONE cache).

Fixtures ``_pc_frames`` / ``_pc_actions`` were extended (velocities, a defending GK, receivers, a
clock) so every family computes NON-vacuously here -- otherwise a family that returns all-NaN
would never reach its ``pitch_control_cache=`` call and the parametrized check would be vacuous.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import features as F
from silly_kicks.tracking.pitch_control import PitchControlCache
from tests.tracking.test_pitch_control_cache import _pc_actions, _pc_frames

# _pc_actions() attributes both actions to team 1, so the families that resolve goal ends /
# attacking direction from home_team_id must use 1 (Minor-5: read the fixture, do not assume 0).
_HTID = 1

# EVERY PC-consuming family, each with its CORRECT kwargs (signatures differ, Minor-5):
#   * obso/space_creation/pausa take `pitch_control_method=` (NOT `method=`) + `home_team_id=`;
#   * cover_shadow/gk_influence/player_influence take `method=` + `home_team_id=`;
#   * off_ball_run_value takes only `home_team_id=`;
#   * pitch_control takes only `method=`.
# `xt` is auto-supplied below to any family whose factory declares it (value-identity holds for
# any fixed xt -- both sides share it).
_PC_FAMILIES = [
    ("pitch_control_xfns", dict(method="voronoi")),
    ("obso_xfns", dict(home_team_id=_HTID, pitch_control_method="voronoi")),
    ("space_creation_xfns", dict(home_team_id=_HTID, pitch_control_method="voronoi")),
    ("pausa_xfns", dict(home_team_id=_HTID, pitch_control_method="voronoi")),
    ("cover_shadow_xfns", dict(home_team_id=_HTID, method="voronoi")),
    ("gk_influence_xfns", dict(home_team_id=_HTID, method="voronoi")),
    ("player_influence_xfns", dict(home_team_id=_HTID, method="voronoi")),
    ("off_ball_run_value_xfns", dict(home_team_id=_HTID)),
]


def _with_xt(factory, kwargs, fitted_xt):
    if "xt" in inspect.signature(factory).parameters:
        return {**kwargs, "xt": fitted_xt}
    return kwargs


@pytest.mark.parametrize("factory_name,kwargs", _PC_FAMILIES)
def test_xfns_shared_cache_byte_identical_to_none(factory_name, kwargs, fitted_xt):
    frames, actions = _pc_frames(), _pc_actions()
    gs = [actions]
    factory = getattr(F, factory_name)
    kwargs = _with_xt(factory, kwargs, fitted_xt)

    none_out = factory(**kwargs)[0](gs, frames)
    cached_out = factory(**{**kwargs, "pitch_control_cache": PitchControlCache()})[0](gs, frames)

    pd.testing.assert_frame_equal(none_out, cached_out)
    # Non-vacuity: this family actually produced values on the fixture, so the byte-identity
    # above genuinely exercised its pitch_control_cache= threading (not an all-NaN no-op).
    assert any(none_out[c].notna().any() for c in none_out.columns), (
        f"{factory_name} produced only NaN -- fixture no longer exercises its PC path"
    )


def test_two_families_divergent_params_share_one_cache_exactly(fitted_xt):
    """The mis-keying mode a same-params test cannot see.

    TWO DIFFERENT families (pitch_control + obso) with DIVERGENT pitch-control params (voronoi vs
    spearman) share ONE cache instance. Each must get its OWN surface; a key that omitted
    method/params would serve a wrong surface to the second consumer and corrupt its output.
    """
    frames, actions = _pc_frames(), _pc_actions()
    gs = [actions]

    cache = PitchControlCache()
    pc_shared = F.pitch_control_xfns("voronoi", pitch_control_cache=cache)[0](gs, frames)
    ob_shared = F.obso_xfns(
        home_team_id=_HTID, pitch_control_method="spearman", xt=fitted_xt, pitch_control_cache=cache
    )[0](gs, frames)

    pc_solo = F.pitch_control_xfns("voronoi")[0](gs, frames)
    ob_solo = F.obso_xfns(home_team_id=_HTID, pitch_control_method="spearman", xt=fitted_xt)[0](gs, frames)

    for shared, solo in ((pc_shared, pc_solo), (ob_shared, ob_solo)):
        for col in shared.columns:
            assert np.array_equal(shared[col].to_numpy(), solo[col].to_numpy(), equal_nan=True), col

    # The cache legitimately holds BOTH surfaces (voronoi + spearman), not one collided entry.
    assert len(cache) == 2


def test_multi_family_xfn_list_one_cache_byte_identical(fitted_xt):
    """One cache across a realistic multi-family xfn list -> every family byte-identical to its
    own-cache baseline. OUTPUT correctness across the whole list; the compute-once perf invariant
    is owned by the cross-family perf-budget test."""
    frames, actions = _pc_frames(), _pc_actions()
    gs = [actions]

    def _list(cache):
        kw = {"pitch_control_cache": cache} if cache is not None else {}
        return [
            F.pitch_control_xfns("voronoi", **kw),
            F.obso_xfns(home_team_id=_HTID, pitch_control_method="voronoi", xt=fitted_xt, **kw),
            F.cover_shadow_xfns(fitted_xt, home_team_id=_HTID, method="voronoi", **kw),
        ]

    solo = [fam[0](gs, frames) for fam in _list(None)]
    shared = [fam[0](gs, frames) for fam in _list(PitchControlCache())]
    for s, b in zip(shared, solo, strict=True):
        pd.testing.assert_frame_equal(s, b)
