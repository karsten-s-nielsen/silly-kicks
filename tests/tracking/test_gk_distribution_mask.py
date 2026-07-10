"""Public gk_distribution_mask: semantics, native/robust lever, dtype/NaN/index safety."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import gk_distribution_mask

_GOALKICK, _PASS, _THROW_IN, _SHOT = 22, 0, 2, 11  # spadlconfig.actiontype_id


def _frow(pid, team, gk, t, *, is_ball=False, x=50.0, y=34.0):
    return dict(
        game_id=1,
        period_id=1,
        frame_id=round(t * 25),
        time_seconds=t,
        frame_rate=25.0,
        player_id=pid,
        team_id=team,
        is_ball=is_ball,
        is_goalkeeper=gk,
        x=float(x),
        y=float(y),
        z=0.0,
        speed=1.0,
        vx=0.0,
        vy=0.0,
        speed_source="native",
        ball_state="alive",
        team_attacking_direction="ltr",
        confidence=None,
        visibility=None,
        source_provider="gradientsports",
        is_goalkeeper_source="native",
    )


def _frames(rows):
    f = pd.DataFrame(rows)
    f["player_id"] = f["player_id"].astype("Int64")
    f["team_id"] = f["team_id"].astype("Int64")
    return f


def _action(aid, type_id, player_id, team_id, t):
    return dict(
        game_id=1,
        action_id=aid,
        period_id=1,
        time_seconds=t,
        team_id=team_id,
        player_id=player_id,
        type_id=type_id,
        start_x=50.0,
        start_y=34.0,
        end_x=60.0,
        end_y=34.0,
        result_id=1,
    )


# GK = player 1 (team 5) detected across the whole window; outfielder = player 10.
def _one_keeper_frames():
    rows = []
    for t in (9.9, 10.0, 19.9, 20.0):
        rows += [
            _frow(1, 5, True, t),
            _frow(10, 5, False, t, x=60.0),
            _frow(2, 6, True, t, x=100.0),
            _frow(pd.NA, pd.NA, False, t, is_ball=True),
        ]
    return _frames(rows)


def test_frames_none_is_goalkicks_only():
    actions = pd.DataFrame(
        [
            _action(0, _GOALKICK, 1, 5, 10.0),
            _action(1, _PASS, 1, 5, 20.0),  # GK pass, but no frames -> cannot detect
        ]
    )
    out = gk_distribution_mask(actions, None)
    assert isinstance(out, pd.Series) and out.dtype == bool
    assert out.tolist() == [True, False]


def test_full_mask_detects_gk_pass_and_goalkick():
    actions = pd.DataFrame(
        [
            _action(0, _GOALKICK, 10, 5, 10.0),  # goalkick by NON-GK -> True (actor-independent)
            _action(1, _PASS, 1, 5, 20.0),  # GK pass -> True
            _action(2, _PASS, 10, 5, 20.0),  # outfielder pass -> False
        ]
    )
    out = gk_distribution_mask(actions, _one_keeper_frames(), resolve_gk="robust")
    assert out.tolist() == [True, True, False]


def test_throw_in_by_gk_is_true_and_gk_shot_is_false():
    actions = pd.DataFrame(
        [
            _action(0, _THROW_IN, 1, 5, 10.0),  # GK throw-in -> True
            _action(1, _SHOT, 1, 5, 20.0),  # GK shot -> False (not pass/throw_in)
        ]
    )
    out = gk_distribution_mask(actions, _one_keeper_frames(), resolve_gk="robust")
    assert out.tolist() == [True, False]


def test_native_equals_robust_for_single_detected_keeper():
    actions = pd.DataFrame(
        [
            _action(0, _PASS, 1, 5, 10.0),
            _action(1, _PASS, 10, 5, 20.0),
            _action(2, _GOALKICK, 10, 5, 10.0),
        ]
    )
    f = _one_keeper_frames()
    nat = gk_distribution_mask(actions, f, resolve_gk="native")
    rob = gk_distribution_mask(actions, f, resolve_gk="robust")
    assert nat.tolist() == rob.tolist() == [True, False, True]


def test_substitution_native_over_includes_robust_tightens():
    # playerA(1) keeper early (t~10), playerB(9) keeper late (t~50); A off after the sub.
    rows = []
    for t in (9.9, 10.0):
        rows += [_frow(1, 5, True, t), _frow(9, 5, False, t, x=55.0), _frow(2, 6, True, t, x=100.0)]
    for t in (49.9, 50.0):
        rows += [_frow(9, 5, True, t), _frow(1, 5, False, t, x=55.0), _frow(2, 6, True, t, x=100.0)]
    f = _frames(rows)
    # A pass by the SUBSTITUTED-OFF keeper (player 1) AFTER the sub (t=50).
    actions = pd.DataFrame([_action(0, _PASS, 1, 5, 50.0)])
    nat = gk_distribution_mask(actions, f, resolve_gk="native")
    rob = gk_distribution_mask(actions, f, resolve_gk="robust")
    assert nat.tolist() == [True]  # global set still contains player 1 as a keeper
    assert rob.tolist() == [False]  # time-resolved acting keeper at t=50 is player 9
    # robust is the strict subset (tightens, never broadens):
    assert (rob.to_numpy() <= nat.to_numpy()).all()


def test_non_range_index_is_aligned():
    actions = pd.DataFrame(
        [
            _action(0, _GOALKICK, 10, 5, 10.0),
            _action(1, _PASS, 1, 5, 20.0),
        ],
        index=[7, 42],
    )  # non-default index (filtered per-match path)
    out = gk_distribution_mask(actions, _one_keeper_frames(), resolve_gk="robust")
    assert list(out.index) == [7, 42]
    assert out.loc[7] and out.loc[42]


def test_string_vs_numeric_ids_match_native():
    # actions carry string ids; frames Int64 -> the NATIVE path is dtype-safe throughout
    # (canonical_id_series on game/team/player, ADR-019) and must still match.
    # NB: robust's GK *resolution* depends on acting_gk_from_frames's team join (raw ==, same-provider
    # dtypes in practice) -- that's its contract, not gk_distribution_mask's, so cross-dtype is asserted
    # on native here; robust's player match via ids_equal is separately dtype-safe.
    actions = pd.DataFrame([_action(0, _PASS, "1", "5", 20.0)])
    out = gk_distribution_mask(actions, _one_keeper_frames(), resolve_gk="native")
    assert out.tolist() == [True]


def test_nan_player_or_team_not_in_scope():
    actions = pd.DataFrame([_action(0, _PASS, np.nan, 5, 20.0)])
    out = gk_distribution_mask(actions, _one_keeper_frames(), resolve_gk="robust")
    assert out.tolist() == [False]


def test_missing_required_column_raises():
    actions = pd.DataFrame([_action(0, _PASS, 1, 5, 20.0)]).drop(columns=["player_id"])
    with pytest.raises(ValueError, match="player_id"):
        gk_distribution_mask(actions, _one_keeper_frames())


def test_game_id_absent_key_shape_both_modes():
    f = _one_keeper_frames()
    actions = pd.DataFrame([_action(0, _PASS, 1, 5, 20.0)]).drop(columns=["game_id"])
    for mode in ("native", "robust"):
        out = gk_distribution_mask(actions, f, resolve_gk=mode)
        assert out.tolist() == [True], mode


def test_invalid_resolve_gk_raises():
    actions = pd.DataFrame([_action(0, _PASS, 1, 5, 20.0)])
    with pytest.raises(ValueError, match="resolve_gk"):
        gk_distribution_mask(actions, _one_keeper_frames(), resolve_gk="bogus")  # type: ignore[arg-type]


def test_public_export_and_docstring_example():
    import silly_kicks.tracking as T

    assert "gk_distribution_mask" in T.__all__
    assert T.gk_distribution_mask is gk_distribution_mask


def test_v1_shim_byte_identical_golden():
    """The frozen v1 _gk_distribution_mask must equal a pinned golden that INCLUDES a native GK
    open-play pass (exercises is_open & actor_is_gk, not just the goal-kick term)."""
    from silly_kicks.tracking._xt_gk import _gk_distribution_mask

    actions = pd.DataFrame(
        [
            _action(0, _GOALKICK, 10, 5, 10.0),  # goalkick by non-GK -> True
            _action(1, _PASS, 1, 5, 20.0),  # GK open-play pass -> True (the risky branch)
            _action(2, _PASS, 10, 5, 20.0),  # outfielder pass -> False
            _action(3, _SHOT, 1, 5, 20.0),  # GK shot -> False
            _action(4, _THROW_IN, 1, 5, 20.0),  # GK throw-in -> True
        ]
    )
    out = _gk_distribution_mask(actions, _one_keeper_frames())
    golden = np.array([True, True, False, False, True])
    assert isinstance(out, np.ndarray) and out.dtype == np.bool_
    assert out.tolist() == golden.tolist()
    # The golden is non-trivial on the GK-pass branch (rows 1 & 4 are non-goalkick True):
    non_goalkick_true = out[[1, 2, 3, 4]].sum()
    assert non_goalkick_true == 2  # rows 1 (pass) and 4 (throw_in)
    # And equals the public native path exactly:
    assert out.tolist() == gk_distribution_mask(actions, _one_keeper_frames(), resolve_gk="native").tolist()
