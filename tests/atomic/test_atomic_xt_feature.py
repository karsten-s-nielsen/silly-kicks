from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

import silly_kicks.atomic.spadl.config as acfg
import silly_kicks.xthreat as xt
from silly_kicks.atomic.vaep import features as afs
from silly_kicks.atomic.vaep.features import xt_xfns
from tests._xthreat_helpers import _corpus_with_shots


@pytest.fixture(scope="module")
def fitted_xt() -> xt.ExpectedThreat:
    """Fitted on a corpus WITH shots so the xT grid is non-zero (rateable)."""
    return xt.ExpectedThreat().fit(_corpus_with_shots(n_per_zone=40, seed=0))


def _atomic_row(action_id, type_id, x, y, dx, dy, *, game_id=1, period_id=1, team_id=1):
    return dict(
        game_id=game_id,
        period_id=period_id,
        action_id=action_id,
        original_event_id=str(action_id),
        time_seconds=float(action_id),
        team_id=team_id,
        player_id=1,
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        bodypart_id=0,
        bodypart_name="foot",
        type_id=type_id,
        type_name=acfg.actiontypes[type_id],
        result_id=-1,
    )


def test_atomic_fail_closed() -> None:
    with pytest.raises(ValueError):
        xt_xfns(model=None)
    with pytest.raises(NotImplementedError):
        xt_xfns(model="default")
    with pytest.raises(NotFittedError):
        xt_xfns(model=xt.ExpectedThreat())


def test_dribble_is_always_finite(fitted_xt: xt.ExpectedThreat) -> None:
    """A dribble atom (never followed by receival) must still get a finite delta."""
    atomic = pd.DataFrame(
        [
            _atomic_row(0, acfg.actiontype_id["dribble"], 30.0, 34.0, 15.0, 0.0),
            _atomic_row(1, acfg.actiontype_id["pass"], 45.0, 34.0, 20.0, 0.0),
        ]
    )
    out = xt_xfns(model=fitted_xt)[0](afs.gamestates(atomic, 1))
    assert np.isfinite(out["xt__singh_counts_a0"].iloc[0])


def test_pass_success_iff_next_receival(fitted_xt: xt.ExpectedThreat) -> None:
    succ = pd.DataFrame(
        [
            _atomic_row(0, acfg.actiontype_id["pass"], 30.0, 34.0, 20.0, 0.0),
            _atomic_row(1, acfg.actiontype_id["receival"], 50.0, 34.0, 0.0, 0.0),
        ]
    )
    fail = pd.DataFrame(
        [
            _atomic_row(0, acfg.actiontype_id["pass"], 30.0, 34.0, 20.0, 0.0),
            _atomic_row(1, acfg.actiontype_id["interception"], 50.0, 34.0, 0.0, 0.0, team_id=2),
        ]
    )
    out_s = xt_xfns(model=fitted_xt)[0](afs.gamestates(succ, 1))
    out_f = xt_xfns(model=fitted_xt)[0](afs.gamestates(fail, 1))
    assert np.isfinite(out_s["xt__singh_counts_a0"].iloc[0])
    assert np.isnan(out_f["xt__singh_counts_a0"].iloc[0])


def test_non_move_and_period_last_are_nan(fitted_xt: xt.ExpectedThreat) -> None:
    df = pd.DataFrame(
        [
            _atomic_row(0, acfg.actiontype_id["shot"], 100.0, 34.0, 5.0, 0.0),  # non-move
            _atomic_row(1, acfg.actiontype_id["pass"], 30.0, 34.0, 20.0, 0.0),  # last action -> no follow-up
        ]
    )
    out = xt_xfns(model=fitted_xt)[0](afs.gamestates(df, 1))
    assert np.isnan(out["xt__singh_counts_a0"].iloc[0])  # shot
    assert np.isnan(out["xt__singh_counts_a0"].iloc[1])  # period-last pass


def test_column_name_symmetry(fitted_xt: xt.ExpectedThreat) -> None:
    cols = afs.feature_column_names(xt_xfns(model=fitted_xt), nb_prev_actions=3)
    assert cols == ["xt__singh_counts_a0", "xt__singh_counts_a1", "xt__singh_counts_a2"]


def test_atomic_exports() -> None:
    from silly_kicks.atomic.vaep import features as afs2

    assert "xt_xfns" in afs2.__all__
    import silly_kicks.atomic.vaep as av

    assert hasattr(av, "xt_xfns")


def _converted_pair(sb_worldcup_data):
    """One WC2018 game as standard-LTR SPADL and its atomic conversion."""
    from silly_kicks.atomic.spadl.base import convert_to_atomic
    from tests._xthreat_helpers import _worldcup_ltr

    ltr = _worldcup_ltr(sb_worldcup_data)
    one = ltr[ltr.game_id == ltr.game_id.iloc[0]].copy().reset_index(drop=True)
    return one, convert_to_atomic(one)


def _geo_key(type_id, sx, sy, ex, ey):
    """Representation-stable key. The xT delta is a pure function of (start_zone, end_zone),
    and atomic (x, x+dx) == standard (start_x, end_x) by construction, so geometry matches
    across representations even though convert_to_atomic RENUMBERS action_id (so action_id is
    NOT a valid cross-representation key). pass/dribble/cross share ids 0/21/1 across the
    standard and atomic configs, so the raw type_id is also stable in the key."""
    return (int(type_id), round(float(sx), 3), round(float(sy), 3), round(float(ex), 3), round(float(ey), 3))


# pass/dribble/cross ids (identical in standard and atomic configs)
_MOVE_IDS = {acfg.actiontype_id[n] for n in ("pass", "dribble", "cross")}


@pytest.mark.filterwarnings("ignore")
def test_symmetry_oracle_value_agreement(sb_worldcup_data) -> None:
    """For any move action present-and-finite in BOTH representations (matched by GEOMETRY),
    the atomic delta equals the standard rate() delta -- across slots a0/a1/a2. Validates the
    coordinate frame, the y-flip, and the grid lookup. Robust by design: it compares only the
    intersection of finite move-deltas, so inherent success-encoding edges (period-last
    pass/cross, out/offside follow-ups) simply fall out of the intersection rather than
    spuriously failing the build."""
    from silly_kicks.vaep import features as sfs
    from silly_kicks.vaep.features import xt_xfns as std_xt_xfns

    std, atomic = _converted_pair(sb_worldcup_data)
    model = xt.ExpectedThreat().fit(std)

    std_states, atomic_states = sfs.gamestates(std, 3), afs.gamestates(atomic, 3)
    std_out = std_xt_xfns(model=model)[0](std_states)
    atomic_out = xt_xfns(model=model)[0](atomic_states)

    for i in range(3):
        s, a = std_states[i], atomic_states[i]
        s_geo = {
            _geo_key(t, sx, sy, ex, ey): d
            for t, sx, sy, ex, ey, d in zip(
                s.type_id, s.start_x, s.start_y, s.end_x, s.end_y, std_out[f"xt__singh_counts_a{i}"], strict=False
            )
            if int(t) in _MOVE_IDS and np.isfinite(d)
        }
        a_geo = {
            _geo_key(t, x, y, x + dx, y + dy): d
            for t, x, y, dx, dy, d in zip(
                a.type_id, a.x, a.y, a.dx, a.dy, atomic_out[f"xt__singh_counts_a{i}"], strict=False
            )
            if int(t) in _MOVE_IDS and np.isfinite(d)
        }
        common = set(s_geo) & set(a_geo)
        # K>=3 (not just non-empty): a0's intersection is large; a1/a2 are thinner (the atomic
        # "previous atom" is often a filtered receival). On a full WC2018 game all three slots
        # clear this easily -- a future fixture swap that shrinks the game should fail LOUDLY here
        # rather than pass on a one-element intersection.
        assert len(common) >= 3, f"slot a{i}: too few shared finite move geometries ({len(common)})"
        for k in common:
            # atol only -- a bin-flip would give a non-tiny diff, not a 1-ULP one. Theoretical
            # edge: atomic end = x + (end-start) is within ~1 ULP of standard end_x, so a coord
            # sitting EXACTLY on a cell boundary (k*105/16) could bin differently. StatsBomb
            # coords (1-2 decimals) never land on those irrational edges, so this is inert here.
            assert np.isclose(s_geo[k], a_geo[k], rtol=0, atol=1e-9), f"slot a{i} delta mismatch at {k}"


@pytest.mark.filterwarnings("ignore")
def test_symmetry_oracle_dribbles_finite(sb_worldcup_data) -> None:
    """KEYSTONE (round-1 critical invariant on real data): every standard dribble has a finite,
    equal atomic counterpart (geometry-matched). A blanket next-atom-receival predicate would
    NaN all atomic dribbles and fail here. Dribbles have no success-encoding ambiguity (always
    successful both representations), so this is a clean hard gate -- do NOT weaken it."""
    import silly_kicks.spadl.config as scfg
    from silly_kicks.vaep import features as sfs
    from silly_kicks.vaep.features import xt_xfns as std_xt_xfns

    std, atomic = _converted_pair(sb_worldcup_data)
    model = xt.ExpectedThreat().fit(std)
    dribble_std = scfg.actiontype_id["dribble"]
    dribble_atom = acfg.actiontype_id["dribble"]

    std_out = std_xt_xfns(model=model)[0](sfs.gamestates(std, 1))["xt__singh_counts_a0"].to_numpy()
    atomic_out = xt_xfns(model=model)[0](afs.gamestates(atomic, 1))["xt__singh_counts_a0"].to_numpy()
    a_geo = {
        _geo_key(t, x, y, x + dx, y + dy): d
        for t, x, y, dx, dy, d in zip(
            atomic.type_id, atomic.x, atomic.y, atomic.dx, atomic.dy, atomic_out, strict=False
        )
        if int(t) == dribble_atom
    }
    n_checked = 0
    for idx in np.flatnonzero((std.type_id == dribble_std).to_numpy()):
        d_std = std_out[idx]
        if not np.isfinite(d_std):
            continue
        k = _geo_key(
            dribble_std, std.start_x.iloc[idx], std.start_y.iloc[idx], std.end_x.iloc[idx], std.end_y.iloc[idx]
        )
        assert k in a_geo and np.isfinite(a_geo[k]), f"atomic dribble missing/NaN at {k}"
        assert np.isclose(a_geo[k], d_std, rtol=0, atol=1e-9)
        n_checked += 1
    assert n_checked > 0, "fixture had no dribbles to check"


def test_multi_game_composite_key(fitted_xt: xt.ExpectedThreat) -> None:
    """Two games with overlapping action_id ranges must not cross-contaminate. The composite
    key includes game_id; bare-action_id keying would alias game-1 row N with game-2 row N
    (dict last-wins -> game-2 deltas would overwrite game-1). We assert the concatenated result
    equals the per-game results. Coords chosen so the two games' dribble deltas DIFFER (0.25 vs
    ~0.21 on this fixture) -- otherwise a collision would be undetectable."""
    g1 = pd.DataFrame(
        [
            _atomic_row(0, acfg.actiontype_id["dribble"], 75.0, 34.0, 25.0, 0.0, game_id=1),
            _atomic_row(1, acfg.actiontype_id["pass"], 100.0, 34.0, 0.0, 0.0, game_id=1),
        ]
    )
    g2 = pd.DataFrame(
        [
            _atomic_row(0, acfg.actiontype_id["dribble"], 85.0, 34.0, 15.0, 0.0, game_id=2),
            _atomic_row(1, acfg.actiontype_id["pass"], 100.0, 34.0, 0.0, 0.0, game_id=2),
        ]
    )
    both = pd.concat([g1, g2], ignore_index=True)
    fn = xt_xfns(model=fitted_xt)[0]
    out_both = fn(afs.gamestates(both, 1))["xt__singh_counts_a0"].to_numpy()
    out_g1 = fn(afs.gamestates(g1, 1))["xt__singh_counts_a0"].to_numpy()
    out_g2 = fn(afs.gamestates(g2, 1))["xt__singh_counts_a0"].to_numpy()
    # precondition: the two games' row-0 dribble deltas differ, so a key collision WOULD change output
    assert not np.isclose(out_g1[0], out_g2[0])
    # isolation: concatenating games does not corrupt either game's deltas
    np.testing.assert_array_equal(out_both[:2], out_g1)
    np.testing.assert_array_equal(out_both[2:], out_g2)


def test_boundary_a1_is_map_hit_not_nan(fitted_xt: xt.ExpectedThreat) -> None:
    """A boundary a1 row is filled with the first-in-group action (gamestates), so its composite
    key is present -> the atomic loop must emit that action's finite delta, NOT NaN. Guards the
    int/float composite-key dtype handling AND the no-boundary-NaN decision (symmetry vs standard)."""
    df = pd.DataFrame(
        [
            _atomic_row(0, acfg.actiontype_id["dribble"], 25.0, 34.0, 15.0, 0.0),
            _atomic_row(1, acfg.actiontype_id["pass"], 40.0, 34.0, 20.0, 0.0),
            _atomic_row(2, acfg.actiontype_id["receival"], 60.0, 34.0, 0.0, 0.0),
        ]
    )
    out = xt_xfns(model=fitted_xt)[0](afs.gamestates(df, 2))
    # row 0 is a group boundary; a1 is filled with the first-in-group action (the dribble) -> finite
    assert np.isfinite(out["xt__singh_counts_a1"].iloc[0])


@pytest.mark.filterwarnings("ignore")
def test_atomic_vaep_integration(sb_worldcup_data) -> None:
    """AtomicVAEP.compute_features with xt_xfns appended produces the columns and rates."""
    from silly_kicks.atomic.vaep import AtomicVAEP
    from silly_kicks.atomic.vaep import features as afs2

    std, atomic = _converted_pair(sb_worldcup_data)
    model = xt.ExpectedThreat().fit(std)
    game = pd.Series({"game_id": int(atomic.game_id.iloc[0]), "home_team_id": int(atomic.team_id.iloc[0])})
    v = AtomicVAEP(xfns=[afs2.location, afs2.actiontype_onehot, *xt_xfns(model=model)], nb_prev_actions=3)
    X = v.compute_features(game, atomic)
    for c in ("xt__singh_counts_a0", "xt__singh_counts_a1", "xt__singh_counts_a2"):
        assert c in X.columns
        assert X[c].dtype == np.float64
    # prove the pipeline actually RATES something -- guards against an all-NaN integration
    # (e.g. a column-name drift or unexpected LTR mirror through compute_features that the
    # raw-transformer oracle wouldn't catch).
    assert np.isfinite(X["xt__singh_counts_a0"]).any()
