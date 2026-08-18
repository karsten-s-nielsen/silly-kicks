"""ADR-063: velocity-less-provider position-only lift.

On frames that DECLARE velocity structurally unavailable (``speed_source == "unavailable"`` on
every row -- the SB360 freeze-frame shape), the four velocity-requiring pitch-control aggregators
now produce the zero-velocity positional model instead of NaN/raise:

* Tier 1 (model-relative) columns are LIFTED -- finite, and responsive to a real change.
* Tier 2 (physical-quantity estimates: reachable area, closing time) are SUPPRESSED to NaN
  (biased at zero velocity; the frame-level ``validate_velocity_regime`` is the signal).
* A frame merely MISSING ``vx``/``vy`` with no marker (a forgotten ``derive_velocities()``)
  fails LOUD -- the caller-bug path.

Fixtures reuse the real ``snapshot_to_tracking_frames`` producer via ``tests/sb360/_fixture.py``,
so the lift is exercised on the exact freeze-frame shape SB360 hits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE
from tests.sb360 import _fixture as F
from tests.sb360._calls import audit_xt

_ROSTER = "gk_one_end"  # one keeper resolvable -> the keeper-dependent aggregators are exercised


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def _leg_a():
    return F.build_leg_a(roster=_ROSTER)


@pytest.fixture
def sb_actions(_leg_a):
    actions, _frames, _links = _leg_a
    return actions.copy()


@pytest.fixture
def links(_leg_a):
    _actions, _frames, links = _leg_a
    return links.copy()


@pytest.fixture
def frames_declared(_leg_a):
    """Declared-velocity-less freeze frames (speed_source == 'unavailable', no vx/vy)."""
    _actions, frames, _links = _leg_a
    frames = frames.copy()
    assert (frames["speed_source"] == SPEED_SOURCE_UNAVAILABLE).all()
    assert "vx" not in frames.columns and "vy" not in frames.columns
    return frames


@pytest.fixture
def frames_forgotten(frames_declared):
    """Same positions, marker STRIPPED -> looks like a forgotten derive_velocities() (a bug)."""
    frames = frames_declared.copy()
    frames["speed_source"] = "native"
    return frames


@pytest.fixture(scope="module")
def xt():
    return audit_xt()


# --------------------------------------------------------------------------- Task 2


def test_pct_at_target_still_works_on_declared_unavailable(sb_actions, frames_declared, links):
    from silly_kicks.tracking.features import pitch_control_at_target

    s = pitch_control_at_target(sb_actions, frames_declared, links=links, method="spearman")
    assert s.notna().any()  # the zero-velocity positional model is served, not NaN


def test_pct_at_target_raises_on_forgotten_velocity(sb_actions, frames_forgotten, links):
    from silly_kicks.tracking.features import pitch_control_at_target

    with pytest.raises(ValueError, match="requires velocity columns"):
        pitch_control_at_target(sb_actions, frames_forgotten, links=links, method="spearman")


# --------------------------------------------------------------------------- Task 3 + 7: gk_influence

_GK_TIER2 = ["gk_reachable_area_m2", "gk_closing_time_min_s__six_yard_box", "gk_closing_time_mean_s__six_yard_box"]


def test_gk_influence_lifts_tier1_suppresses_tier2_on_declared_unavailable(sb_actions, frames_declared, links, xt):
    from silly_kicks.tracking.features import add_gk_influence

    out = add_gk_influence(sb_actions, frames_declared, xt, links=links)
    # Tier 1 (model-relative): the zero-velocity pitch-control share is served, not NaN.
    assert out["gk_pitch_control_share_weighted"].notna().any()
    # Tier 2 (physical-quantity estimates, biased at zero velocity): SUPPRESSED to NaN.
    # The reason is a whole-frame-set fact -> validate_velocity_regime, not a per-row column.
    for col in _GK_TIER2:
        assert not out[col].notna().any(), f"{col} should be suppressed to NaN on declared-unavailable"


def test_gk_influence_tier1_non_vacuous_on_declared_unavailable(sb_actions, frames_declared, links, xt):
    from silly_kicks.tracking.features import add_gk_influence

    base = add_gk_influence(sb_actions, frames_declared, xt, links=links)["gk_pitch_control_share_weighted"]
    moved = frames_declared.copy()
    moved.loc[moved["is_goalkeeper"].astype(bool), "x"] += 20.0
    perturbed = add_gk_influence(sb_actions, moved, xt, links=links)["gk_pitch_control_share_weighted"]
    # A real change to GK position moves the share: the value is a computation, not a constant.
    both = base.notna() & perturbed.notna()
    assert both.any() and (base[both] != perturbed[both]).any()


def test_gk_influence_raises_on_forgotten_velocity(sb_actions, frames_forgotten, links, xt):
    from silly_kicks.tracking.features import add_gk_influence

    with pytest.raises(ValueError, match="requires velocity columns"):
        add_gk_influence(sb_actions, frames_forgotten, xt, links=links)


# --------------------------------------------------------------------------- Task 4: cover_shadows

_COVER_SHADOWS_COLS = [
    "n_blocked_receivers",
    "n_potential_receivers",
    "blocking_score",
    "blocked_threat_fraction",
    "max_single_defender_blocking_score",
]


def test_cover_shadows_lifts_on_declared_unavailable(sb_actions, frames_declared, links, xt):
    from silly_kicks.tracking.features import add_cover_shadows

    out = add_cover_shadows(sb_actions, frames_declared, xt, links=links)
    # All cover-shadow columns are model-relative (Tier 1) -> lifted, not NaN.
    for col in _COVER_SHADOWS_COLS:
        assert out[col].notna().any(), f"{col} should be lifted (finite) on declared-unavailable"


def test_cover_shadows_raises_on_forgotten_velocity(sb_actions, frames_forgotten, links, xt):
    from silly_kicks.tracking.features import add_cover_shadows

    with pytest.raises(ValueError, match="requires velocity columns"):
        add_cover_shadows(sb_actions, frames_forgotten, xt, links=links)


# --------------------------------------------------------------------------- Task 5 + 7: player_influence

_PLAYER_TIER1 = ["off_ball_xt_team", "off_ball_xt_opponent", "off_ball_xt_diff"]
_PLAYER_TIER2 = ["actor_reachable_area_m2", "reachable_area_team", "reachable_area_opponent", "reachable_area_diff"]


def test_player_influence_lifts_tier1_suppresses_tier2_on_declared_unavailable(sb_actions, frames_declared, links, xt):
    from silly_kicks.tracking.features import add_player_influence

    out = add_player_influence(sb_actions, frames_declared, xt, links=links)
    # Tier 1 (model-relative off-ball xT): lifted.
    for col in _PLAYER_TIER1:
        assert out[col].notna().any(), f"{col} should be lifted on declared-unavailable"
    # Tier 2 (reachable area -- biased physical estimate at zero velocity): suppressed to NaN.
    for col in _PLAYER_TIER2:
        assert not out[col].notna().any(), f"{col} should be suppressed to NaN on declared-unavailable"


def test_player_influence_raises_on_forgotten_velocity(sb_actions, frames_forgotten, links, xt):
    from silly_kicks.tracking.features import add_player_influence

    with pytest.raises(ValueError, match="requires velocity columns"):
        add_player_influence(sb_actions, frames_forgotten, xt, links=links)


# --------------------------------------------------------------------------- Task 6: space_creation


def test_space_creation_lifts_on_declared_unavailable(sb_actions, frames_declared, links, xt):
    from silly_kicks.tracking.features import add_space_creation

    out = add_space_creation(sb_actions, frames_declared, links=links, xt=xt)
    # All three space-creation columns are model-relative (Tier 1) -> the zero-velocity model.
    assert out["space_created_m2"].notna().any()
    assert out["space_denied_m2_opponent"].notna().any()
    assert out["obso_epv_source"].notna().any()


def test_space_creation_raises_on_forgotten_velocity(sb_actions, frames_forgotten, links, xt):
    from silly_kicks.tracking.features import add_space_creation

    with pytest.raises(ValueError, match="requires velocity columns"):
        add_space_creation(sb_actions, frames_forgotten, links=links, xt=xt)


# --------------------------------------------------------------------------- obso / pausa (ADR-063 extension)
#
# add_obso/add_pausa were never in the "40 fully-NaN" set (they already zero-filled on declared
# frames), but they kept the SAME loose unconditional zero-fill that silently accepted a forgotten
# derive_velocities(). Extended to fail-fast via the single _precompute_obso_lookup seam; add_pausa
# inherits via add_obso, and the *_xfns via the per-Series / add_pausa delegation.


def test_obso_raises_on_forgotten_velocity(sb_actions, frames_forgotten, xt):
    from silly_kicks.tracking.features import add_obso

    with pytest.raises(ValueError, match="requires velocity columns"):
        add_obso(sb_actions, frames_forgotten, xt=xt)


def test_pausa_raises_on_forgotten_velocity(sb_actions, frames_forgotten, xt):
    from silly_kicks.tracking.features import add_pausa

    # add_pausa computes OBSO via add_obso -> _precompute_obso_lookup, so it inherits the raise.
    with pytest.raises(ValueError, match="requires velocity columns"):
        add_pausa(sb_actions, frames_forgotten, xt=xt)


def test_obso_declared_unavailable_does_not_raise(sb_actions, frames_declared, xt):
    from silly_kicks.tracking.features import add_obso

    # Declared behaviour is UNCHANGED (the seam zero-fills, as the loose block did). obso may be NaN
    # on a single-frame freeze-frame (its pass window is a separate, window-constitutive concern),
    # but it must not RAISE -- only a forgotten (undeclared) frame does.
    out = add_obso(sb_actions, frames_declared, xt=xt)
    assert "obso_actual" in out.columns


# ------------------------------------------------- public lower-level engines (ADR-063 re-review M1/M2)
#
# The ADR promises the lift "for direct callers" of the public compute_* engines. These pin that:
# compute_pass_obso's velocity seam (_ensure_velocity_columns) and compute_threat_pc /
# compute_blocking_score honour declared-velocity-less (lift) vs forgotten (raise).


def test_obso_ensure_velocity_columns_is_marker_aware(frames_declared):
    from silly_kicks.tracking._obso import _ensure_velocity_columns

    out = _ensure_velocity_columns(frames_declared, method="spearman")  # declared -> zero-velocity fill
    assert (out["vx"] == 0.0).all() and (out["vy"] == 0.0).all()

    forgotten = frames_declared.copy()
    forgotten["speed_source"] = "native"  # forgotten -> fail-fast, not a silent zero-fill
    with pytest.raises(ValueError, match="requires velocity columns"):
        _ensure_velocity_columns(forgotten, method="spearman")


def _one_frame(frames):
    key = frames[["period_id", "frame_id"]].iloc[0]
    return frames[(frames["period_id"] == key["period_id"]) & (frames["frame_id"] == key["frame_id"])].copy()


def test_cover_shadow_engines_lift_direct_callers_on_declared_and_raise_on_forgotten(frames_declared, xt):
    from silly_kicks.tracking import resolve_defended_goals
    from silly_kicks.tracking._cover_shadows import compute_blocking_score, compute_threat_pc

    gm = resolve_defended_goals(frames_declared)
    frame = _one_frame(frames_declared)
    tid = F.HOME_TEAM_ID  # has the keeper in gk_one_end -> goal end resolves

    # Declared-velocity-less -> the zero-velocity model, NOT a raise (the ADR "for direct callers"
    # claim: before ADR-063 these bare compute_pitch_control callers raised on a declared frame).
    assert isinstance(compute_threat_pc(frame, attacking_team_id=tid, xt=xt, goal_map=gm), float)
    compute_blocking_score(frame, tid, xt, goal_map=gm)  # no raise

    forgotten = frame.copy()
    forgotten["speed_source"] = "native"
    with pytest.raises(ValueError, match="requires velocity columns"):
        compute_threat_pc(forgotten, attacking_team_id=tid, xt=xt, goal_map=gm)
    with pytest.raises(ValueError, match="requires velocity columns"):
        compute_blocking_score(forgotten, tid, xt, goal_map=gm)


def test_declared_unavailable_aggregator_does_not_mutate_caller_frames(sb_actions, frames_declared, links, xt):
    """ADR-033 purity on the velocity-less path: the edge helper COPIES on the declared branch, so
    the caller's frames must be untouched -- no vx/vy added, positions unchanged. The ADR-033 gate
    exercises velocity-bearing inputs (helper is a no-op there); this covers the copy path."""
    from silly_kicks.tracking.features import add_gk_influence

    before = frames_declared.copy(deep=True)
    add_gk_influence(sb_actions, frames_declared, xt, links=links)
    assert "vx" not in frames_declared.columns and "vy" not in frames_declared.columns
    pd.testing.assert_frame_equal(frames_declared, before)


# --------------------------------------------------------------------------- xfns alignment (opt-in VAEP)
#
# The four *_xfns transformer paths must honour the SAME contract as the add_* battery: Tier-1
# lifted, Tier-2 suppressed, and a forgotten derive_velocities() fails LOUD (rather than the
# per-frame ValueError catch degrading it to warn+NaN).


def _run_xfns(xfns_list, states_actions, frames):
    """Invoke a single-transformer xfns on three identical gamestate slots."""
    transformer = xfns_list[0]
    return transformer([states_actions, states_actions, states_actions], frames)


def test_gk_influence_xfns_lifts_and_suppresses_on_declared(sb_actions, frames_declared, xt):
    from silly_kicks.tracking.features import gk_influence_xfns

    out = _run_xfns(gk_influence_xfns(xt), sb_actions, frames_declared)
    assert out["gk_pitch_control_share_weighted_a0"].notna().any()  # Tier 1 lifted
    assert not out["gk_reachable_area_m2_a0"].notna().any()  # Tier 2 suppressed


def test_cover_shadow_xfns_lifts_on_declared(sb_actions, frames_declared, xt):
    from silly_kicks.tracking.features import cover_shadow_xfns

    out = _run_xfns(cover_shadow_xfns(xt), sb_actions, frames_declared)
    assert out["blocking_score_a0"].notna().any()  # Tier 1 lifted


def test_player_influence_xfns_lifts_and_suppresses_on_declared(sb_actions, frames_declared, xt):
    from silly_kicks.tracking.features import player_influence_xfns

    out = _run_xfns(player_influence_xfns(xt), sb_actions, frames_declared)
    assert out["off_ball_xt_team_a0"].notna().any()  # Tier 1 lifted
    # Tier 2 suppressed across all slots -- including actor_reachable_area_m2, which leaks 0.0
    # for a GK actor unless the transformer's own assembly suppresses it (mirrors add_*).
    for base in ("actor_reachable_area_m2", "reachable_area_team", "reachable_area_opponent", "reachable_area_diff"):
        assert not out[f"{base}_a0"].notna().any(), f"{base}_a0 should be suppressed to NaN"


@pytest.mark.parametrize(
    "factory_name",
    [
        "gk_influence_xfns",
        "cover_shadow_xfns",
        "player_influence_xfns",
        "space_creation_xfns",
        "obso_xfns",
        "pausa_xfns",
    ],
)
def test_xfns_raise_on_forgotten_velocity(factory_name, sb_actions, frames_forgotten, xt):
    import silly_kicks.tracking.features as feats

    factory = getattr(feats, factory_name)
    # gk/cover/player take xt POSITIONALLY; space/obso/pausa take it KEYWORD-only.
    positional = factory_name in ("gk_influence_xfns", "cover_shadow_xfns", "player_influence_xfns")
    xfns_list = factory(xt) if positional else factory(xt=xt)
    with pytest.raises(ValueError, match="requires velocity columns"):
        _run_xfns(xfns_list, sb_actions, frames_forgotten)


# --------------------------------------------------------------------------- Task 8: regression fences


def test_tier3_velocity_constitutive_stays_nan_on_declared_unavailable(sb_actions, frames_declared, links):
    """A velocity-CONSTITUTIVE aggregator NOT in the four (DAS) stays NaN -- guards an over-broad fix.

    DAS is accessible space, which cannot exist without velocity; it degrades honestly on a
    declared-velocity-less frame (``das_source`` -> ``unscoreable_frame``, ADR-043), and the lift
    of the four pitch-control aggregators must not accidentally revive it.
    """
    from scripts._sb_battery import call_aggregator

    out = call_aggregator("add_das", sb_actions, frames_declared, links, F.HOME_TEAM_ID)
    for col in ("das_team", "das_opponent", "das_diff"):
        assert not out[col].notna().any(), f"{col} (velocity-constitutive) must stay NaN"


# --------------------------------------------------------------------------- velocity-bearing fixtures


@pytest.fixture(scope="module")
def _leg_b():
    return F.build_leg_b(roster=_ROSTER)


@pytest.fixture
def frames_velocity(_leg_b):
    _actions, frames, _links = _leg_b
    assert "vx" in frames.columns and "vy" in frames.columns
    return frames.copy()


@pytest.fixture
def sb_actions_b(_leg_b):
    actions, _frames, _links = _leg_b
    return actions.copy()


@pytest.fixture
def links_b(_leg_b):
    _actions, _frames, links = _leg_b
    return links.copy()


def test_helper_is_a_noop_on_velocity_bearing_frames(frames_velocity):
    """The no-retrain proof's linchpin: the edge helper returns the SAME object untouched when
    vx/vy are present, so every aggregator's code path on a velocity-bearing frame is byte-
    identical to before ADR-063."""
    from silly_kicks.tracking._velocity_availability import zero_velocity_if_unavailable

    assert zero_velocity_if_unavailable(frames_velocity, method="spearman") is frames_velocity


def test_velocity_bearing_lifts_both_tiers(sb_actions_b, frames_velocity, links_b, xt):
    """On velocity-BEARING frames both tiers are finite -- Tier-2 is NOT suppressed (velocity is
    present, so ``velocity_unavailable_by_design`` is False). This is what makes the change
    invariant on every existing trained-model input."""
    from silly_kicks.tracking.features import (
        add_gk_influence,
        add_player_influence,
    )

    gk = add_gk_influence(sb_actions_b, frames_velocity, xt, links=links_b)
    assert gk["gk_pitch_control_share_weighted"].notna().any()  # Tier 1
    assert gk["gk_reachable_area_m2"].notna().any()  # Tier 2 NOT suppressed on velocity-bearing

    pi = add_player_influence(sb_actions_b, frames_velocity, xt, links=links_b)
    assert pi["off_ball_xt_team"].notna().any()  # Tier 1
    assert pi["reachable_area_team"].notna().any()  # Tier 2 NOT suppressed on velocity-bearing


def test_d2_boundary_share_invariant_reachable_area_biased(sb_actions_b, frames_velocity, links_b, xt):
    """D2 (measured, not asserted): the model-relative pitch-control share is ~velocity-invariant,
    while the physical-quantity reachable area is substantially velocity-sensitive -- the reason
    the first is lifted (Tier 1) and the second suppressed (Tier 2) on declared-velocity-less
    frames. Encodes the ADR-063 D2 measurement as a durable regression guard (relative ordering,
    not a brittle absolute magnitude)."""
    from silly_kicks.tracking.features import add_gk_influence, add_player_influence

    frames_zero = frames_velocity.copy()
    frames_zero["vx"] = 0.0
    frames_zero["vy"] = 0.0  # zero-velocity but marker still 'derived' -> nothing suppressed

    def _rel(a, b):
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        both = a.notna() & b.notna()
        if not both.any():
            return None
        return float(np.max(np.abs(a[both] - b[both]))) / (float(np.max(np.abs(a[both]))) or 1.0)

    share_v = add_gk_influence(sb_actions_b, frames_velocity, xt, links=links_b)["gk_pitch_control_share_weighted"]
    share_z = add_gk_influence(sb_actions_b, frames_zero, xt, links=links_b)["gk_pitch_control_share_weighted"]
    area_v = add_player_influence(sb_actions_b, frames_velocity, xt, links=links_b)["reachable_area_opponent"]
    area_z = add_player_influence(sb_actions_b, frames_zero, xt, links=links_b)["reachable_area_opponent"]

    share_rel = _rel(share_v, share_z)
    area_rel = _rel(area_v, area_z)
    assert share_rel is not None and area_rel is not None
    # The model-relative share is essentially velocity-invariant...
    assert share_rel < 0.05
    # ...while the physical-quantity reachable area is materially velocity-sensitive (the bias).
    assert area_rel > 0.10
    assert area_rel > share_rel
