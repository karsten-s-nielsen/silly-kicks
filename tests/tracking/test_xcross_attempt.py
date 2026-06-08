"""Unit tests for TF-17 xCrossAttempt (xCross)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _xcross_attempt as xc


def _one_frame():
    """A single (game,period,frame): ball wide-left near the byline, carrier = A1,
    one defender, one teammate, defending GK = B-GK. Attacked goal at x=105."""
    return pd.DataFrame(
        {
            "game_id": ["g"] * 5,
            "period_id": [1] * 5,
            "frame_id": [7] * 5,
            "time_seconds": [40.0] * 5,
            "team_id": ["A", "A", "B", "B", "ball"],
            "player_id": ["A1", "A2", "B1", "Bgk", None],
            "x": [95.0, 88.0, 100.0, 104.0, 95.0],
            "y": [10.0, 30.0, 12.0, 34.0, 10.0],
            "vx": [1.0, 0.0, 0.0, 0.0, 1.0],
            "vy": [0.0, 0.0, 0.0, 0.0, 0.0],
            "is_ball": [False, False, False, False, True],
            "is_goalkeeper": [False, False, False, True, False],
            "ball_state": ["alive"] * 5,
        }
    )


def test_extract_features_faithful_shape():
    feats = xc.extract_xcross_features(_one_frame(), gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    assert list(feats.columns) == xc.XCROSS_FEATURE_NAMES_FAITHFUL
    assert len(feats) == 1
    assert feats.shape[1] == 16  # 3 ball + 7 confounders (#7 dropped) + 6 GK
    assert "crosser_role" not in feats.columns  # H2: dropped (collinear with dist_endline)


def test_extended_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        xc.extract_xcross_features(
            _one_frame(), gk_team_id="B", goal_x=105.0, carrier_player_id="A1", feature_set="extended"
        )


def test_ten_minute_warning_off_early():
    feats = xc.extract_xcross_features(_one_frame(), gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    assert feats["ten_minute_warning"].iloc[0] == 0  # t=40s, not final 10 min


def test_ten_minute_warning_period2_early_is_zero():
    """PA-M1: locks the PER-PERIOD time_seconds contract -- a period-2 frame early in the half
    (t=120s) must be 0. If time_seconds were match-cumulative, t would be ~2820 -> wrongly 1."""
    frame = _one_frame()
    frame["period_id"] = 2
    frame["time_seconds"] = 120.0
    feats = xc.extract_xcross_features(frame, gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    assert feats["ten_minute_warning"].iloc[0] == 0


def test_ten_minute_warning_late_half_is_one():
    frame = _one_frame()
    frame["time_seconds"] = 40 * 60.0  # 40th minute -> within the final 10 of a 45-min half
    feats = xc.extract_xcross_features(frame, gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    assert feats["ten_minute_warning"].iloc[0] == 1


def test_dist_endline_goal_relative():
    feats = xc.extract_xcross_features(_one_frame(), gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    # carrier A1 at x=95, attacked goal at 105 -> 10 m from endline (gr_x = cx)
    assert feats["dist_endline"].iloc[0] == pytest.approx(10.0, abs=1e-6)


def test_box_ratio_counts():
    """Box at the attacked goal in GOAL-RELATIVE coords: gr_x <= 16.5 (gr_x = 105 - x) AND
    |y-34| <= 20.16 (y in [13.84, 54.16]). Build a frame with explicit box occupants -- the
    wide carrier itself sits OUTSIDE the box (that's the point of a cross)."""
    frame = pd.DataFrame(
        {
            "game_id": ["g"] * 6,
            "period_id": [1] * 6,
            "frame_id": [7] * 6,
            "time_seconds": [40.0] * 6,
            "team_id": ["A", "A", "A", "B", "B", "ball"],
            "player_id": ["A1", "Ax1", "Ax2", "Dx1", "Bgk", None],
            "x": [95.0, 100.0, 98.0, 101.0, 104.0, 95.0],  # gr_x = 105-x: A1=10, Ax1=5, Ax2=7, Dx1=4, Bgk=1
            "y": [10.0, 34.0, 40.0, 30.0, 34.0, 10.0],  # A1 wide (out of box); Ax1/Ax2/Dx1 central (in)
            "vx": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "vy": [0.0] * 6,
            "is_ball": [False, False, False, False, False, True],
            "is_goalkeeper": [False, False, False, False, True, False],
            "ball_state": ["alive"] * 6,
        }
    )
    feats = xc.extract_xcross_features(frame, gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    # in box: attackers Ax1(5,34), Ax2(7,40) -> off=2; defender Dx1(4,30) -> def=1; Bgk GK-excluded;
    # A1(10,10) is wide (|10-34|=24 > 20.16) -> out. ratio = 2/1 = 2.0
    assert feats["box_off_def_ratio"].iloc[0] == pytest.approx(2.0, abs=1e-6)


# --- Task 3: GK block ---


def test_gk_block_filled_and_isolatable():
    feats = xc.extract_xcross_features(_one_frame(), gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    # GK block is a contiguous tail block, all non-NaN here (GK row present)
    assert xc.XCROSS_FEATURE_NAMES_FAITHFUL[-6:] == xc.XCROSS_GK_BLOCK
    assert feats[xc.XCROSS_GK_BLOCK].notna().all(axis=None)
    # dropping the block leaves the 10 non-GK features (3 ball + 7 confounders)
    base = [c for c in feats.columns if c not in xc.XCROSS_GK_BLOCK]
    assert len(base) == 10


def test_gk_block_nan_when_no_gk_row():
    frame = _one_frame()
    frame.loc[frame["player_id"] == "Bgk", "is_goalkeeper"] = False  # remove GK identity
    feats = xc.extract_xcross_features(frame, gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    assert feats[xc.XCROSS_GK_BLOCK].isna().all(axis=None)


def test_gk_r_goal_relative():
    feats = xc.extract_xcross_features(_one_frame(), gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    # GK at (104,34), attacked goal x=105 -> gr_x = 1.0, y-34=0 -> gk_r = 1.0
    assert feats["gk_r"].iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_gk_post_distances_goal_relative():
    """C1: posts live at the attacked goal line gr_x=0 (NOT PITCH_LENGTH). GK at gr_x=1, y=34.
    Carrier A1 at y=10 -> near post on the left (post_y = 34 - 3.66 = 30.34), far post right (37.66).
    gk near = hypot(1, 34-30.34) = hypot(1, 3.66); far symmetric (GK central). Both ~3.80, NOT ~104."""
    feats = xc.extract_xcross_features(_one_frame(), gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    assert feats["gk_dist_near_post"].iloc[0] == pytest.approx(math.hypot(1.0, 3.66), abs=1e-6)
    assert feats["gk_dist_far_post"].iloc[0] == pytest.approx(math.hypot(1.0, 3.66), abs=1e-6)
    assert feats["gk_dist_near_post"].iloc[0] < 10.0  # would be ~104 under the inverted convention


def test_faithful_module_never_imports_pitch_control():
    """L-4 (PA-M2): the _xcross_attempt module must NEVER import the pitch_control package -- the
    faithful #3 proxy is pure numpy, locking the TF-19 counterfactual guarantee (a cached canonical
    pitch-control surface must never leak into a counterfactual, player-moved frame). Static
    source-scan for an actual IMPORT statement (the reserved `pitch_control_cache` kwarg is NOT an
    import and is allowed). Non-vacuous: it fails loudly the moment anyone adds the import. (A
    sys.modules check can't work here -- the tracking package __init__ imports pitch_control
    regardless.)"""
    import re

    src = open(xc.__file__, encoding="utf-8").read()
    offenders = [ln for ln in src.splitlines() if re.search(r"^\s*(from|import)\b.*pitch_control", ln)]
    assert offenders == [], f"_xcross_attempt imports pitch_control: {offenders}"


# --- Task 4: build_xcross_labels ---


def _label_frames_and_actions():
    frames_index = pd.DataFrame(  # frames-side team column = team_in_possession (matches prepare)
        {
            "game_id": ["g"] * 4,
            "period_id": [1] * 4,
            "frame_id": [1, 2, 3, 4],
            "time_seconds": [0.0, 0.4, 0.8, 1.2],
            "team_in_possession": ["A"] * 4,
        }
    )
    from silly_kicks.spadl import config as spc

    actions = pd.DataFrame(  # A crosses at t=0.9; a pass (non-cross) at 0.2
        {
            "game_id": ["g", "g"],
            "period_id": [1, 1],
            "team_id": ["A", "A"],
            "time_seconds": [0.9, 0.2],
            "type_id": [spc.actiontype_id["cross"], spc.actiontype_id["pass"]],
        }
    )
    return frames_index, actions


def test_build_xcross_labels_open_play_only():
    fidx, actions = _label_frames_and_actions()
    y = xc.build_xcross_labels(fidx, actions, horizon_seconds=1.0)
    # cross at 0.9 in [t,t+1] for frames at 0.0,0.4,0.8 -> 1; 1.2 -> 0 (0.9 < 1.2)
    assert list(np.asarray(y)) == [1, 1, 1, 0]


def test_build_xcross_labels_set_pieces_togglable():
    from silly_kicks.spadl import config as spc

    fidx, actions = _label_frames_and_actions()
    actions.loc[1, "type_id"] = spc.actiontype_id["corner_crossed"]
    actions.loc[1, "time_seconds"] = 0.1
    y_open = xc.build_xcross_labels(fidx, actions, horizon_seconds=1.0)
    y_all = xc.build_xcross_labels(fidx, actions, horizon_seconds=1.0, cross_types=("cross", "corner_crossed"))
    assert list(np.asarray(y_open)) == [1, 1, 1, 0]  # corner ignored
    assert np.asarray(y_all)[0] == 1  # corner at 0.1 now counts


# --- Task 5: prepare_xcross_training_data ---


def _mini_match():
    rows = []
    for fr, t in enumerate([0.0, 0.4, 0.8, 1.2], start=1):
        rows += [
            dict(
                game_id="g",
                period_id=1,
                frame_id=fr,
                time_seconds=t,
                team_id="A",
                player_id="A1",
                x=95.0,
                y=10.0,
                vx=1.0,
                vy=0.0,
                is_ball=False,
                is_goalkeeper=False,
                ball_state="alive",
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=fr,
                time_seconds=t,
                team_id="B",
                player_id="Bgk",
                x=104.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
                is_ball=False,
                is_goalkeeper=True,
                ball_state="alive",
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=fr,
                time_seconds=t,
                team_id="ball",
                player_id=None,
                x=95.0,
                y=10.0,
                vx=1.0,
                vy=0.0,
                is_ball=True,
                is_goalkeeper=False,
                ball_state="alive",
            ),
        ]
    frames = pd.DataFrame(rows)
    frames["source_provider"] = "test"  # required by link_actions_to_frames
    from silly_kicks.spadl import config as spc

    actions = pd.DataFrame(  # SPADL-shaped: result_id present (required by the score lookup)
        {
            "game_id": ["g"],
            "period_id": [1],
            "team_id": ["A"],
            "time_seconds": [0.9],
            "type_id": [spc.actiontype_id["cross"]],
            "result_id": [spc.result_id["success"]],
        }
    )
    return frames, actions


def test_prepare_returns_features_labels_groups():
    frames, actions = _mini_match()
    X, y, groups = xc.prepare_xcross_training_data(frames, actions, home_team_id="A")
    assert list(X.columns) == xc.XCROSS_FEATURE_NAMES_FAITHFUL
    assert len(X) == len(y) == len(groups)
    assert set(np.unique(y)).issubset({0, 1})
    assert (groups == "g").all()
    assert y.sum() >= 1  # the cross at 0.9 labels some wide-area frames positive


def test_prepare_tolerates_na_team_id_in_frames():
    """Regression (box pilot, real GS match): frames can carry pd.NA in team_id (ball row or an
    unresolved GS jersey). prepare must not raise 'boolean value of NA is ambiguous' at the
    defending-team computation. Inject an NA-team non-ball player into every frame."""
    frames, actions = _mini_match()
    extra = frames[frames["player_id"] == "Bgk"].copy()
    extra["player_id"] = "Bx"
    extra["is_goalkeeper"] = False
    extra["team_id"] = pd.NA  # unresolved-team outfielder
    frames_na = pd.concat([frames, extra], ignore_index=True)
    X, y, _groups = xc.prepare_xcross_training_data(frames_na, actions, home_team_id="A")  # must not raise
    assert list(X.columns) == xc.XCROSS_FEATURE_NAMES_FAITHFUL
    assert len(X) == len(y)
    assert y.sum() >= 1  # still labels the wide-area cross frames (defending team B resolved)


def test_prepare_score_differential_wired_and_signed():
    """PA-H1: confounder #1 must be REALIZED (non-NaN) and signed from the possessing team's
    perspective. Team A (home, possessing) scored 1; B scored 0 -> score_differential = +1."""
    from silly_kicks.spadl import config as spc

    frames, actions = _mini_match()
    goal = pd.DataFrame(
        {
            "game_id": ["g"],
            "period_id": [1],
            "team_id": ["A"],
            "time_seconds": [0.0],
            "type_id": [spc.actiontype_id["shot"]],
            "result_id": [spc.result_id["success"]],
        }
    )
    actions2 = pd.concat([actions.assign(result_id=spc.result_id["success"]), goal], ignore_index=True)
    X, _, _ = xc.prepare_xcross_training_data(frames, actions2, home_team_id="A")
    assert X["score_differential"].notna().all()  # realized for every row (actions supplied)
    assert X["score_differential"].max() == pytest.approx(1.0, abs=1e-6)  # A leads by 1 (positive)
    assert (X["score_differential"].dropna() >= 0.0).all()  # never negative for the leading possessor


# --- Task 6: XCrossAttemptModel ---


def _fit_tiny_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 16)), columns=xc.XCROSS_FEATURE_NAMES_FAITHFUL)
    y = (X["gk_r"] + rng.normal(scale=0.5, size=200) > 0).astype(int).to_numpy()
    m = xc.XCrossAttemptModel().fit(X, pd.Series(y))
    return m, X


def test_model_fit_predict_proba():
    m, X = _fit_tiny_model()
    p = m.predict_proba(X)
    assert p.shape == (200,)
    assert ((p >= 0) & (p <= 1)).all()


def test_model_deterministic():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(150, 16)), columns=xc.XCROSS_FEATURE_NAMES_FAITHFUL)
    y = pd.Series((rng.random(150) > 0.7).astype(int))
    p1 = xc.XCrossAttemptModel().fit(X, y).predict_proba(X)
    p2 = xc.XCrossAttemptModel().fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


def test_model_save_load_roundtrip(tmp_path):
    m, X = _fit_tiny_model()
    d = tmp_path / "xcross_v1"
    m.save(d)
    assert (d / "model.json").exists() and (d / "metadata.json").exists() and (d / "SHA256SUMS").exists()
    m2 = xc.XCrossAttemptModel.load(d)
    np.testing.assert_allclose(m.predict_proba(X), m2.predict_proba(X), rtol=1e-9)


def test_model_sha256_verification(tmp_path):
    m, _ = _fit_tiny_model()
    d = tmp_path / "xcross_v1"
    m.save(d)
    (d / "model.json").write_text((d / "model.json").read_text() + " ")  # tamper
    from silly_kicks.tracking._xshot_occurrence import IntegrityError

    with pytest.raises(IntegrityError):
        xc.XCrossAttemptModel.load(d)


def test_from_variant_unknown_raises_filenotfound():
    """An unbundled variant name raises FileNotFoundError and does NOT cascade to Hub.

    (PR-B reframed the old "default raises until weights" assertion: once PR-B bundles `default`,
    `from_variant("default")` succeeds -- but an unknown variant must always raise. `from_hub` now
    does a real download; its behaviour is covered by the mocked test in the integration suite.)
    """
    xc._VARIANT_CACHE.clear()
    with pytest.raises(FileNotFoundError):
        xc.XCrossAttemptModel.from_variant("does-not-exist")


def test_carrier_params_recorded_and_restored(tmp_path):
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(120, 16)), columns=xc.XCROSS_FEATURE_NAMES_FAITHFUL)
    y = pd.Series((rng.random(120) > 0.6).astype(int))
    cp = {"tolerance_m": 2.0, "beta": 0.1, "gamma": 0.5}
    m = xc.XCrossAttemptModel().fit(X, y, carrier_params=cp)
    d = tmp_path / "m"
    m.save(d)
    assert xc.XCrossAttemptModel.load(d).carrier_params == cp


# --- Task 7: compute_xcross_attempt ---


def _fit_on_mini():
    frames, actions = _mini_match()
    X, y, _ = xc.prepare_xcross_training_data(frames, actions, home_team_id="A")
    return frames, actions, xc.XCrossAttemptModel().fit(X, pd.Series(y))


def test_compute_adds_column_and_uses_metadata_carrier_params(monkeypatch):
    frames, actions = _mini_match()
    X, y, _ = xc.prepare_xcross_training_data(frames, actions, home_team_id="A")
    model = xc.XCrossAttemptModel().fit(
        X, pd.Series(y), carrier_params={"tolerance_m": 2.5, "beta": 0.0, "gamma": 0.25}
    )
    seen: dict = {}
    import silly_kicks.tracking._xcross_attempt as mod

    real = mod.infer_ball_carrier
    monkeypatch.setattr(mod, "infer_ball_carrier", lambda f, **k: seen.update(k) or real(f, **k))
    out = xc.compute_xcross_attempt(frames, model=model, home_team_id="A")
    assert "xcross_attempt" in out.columns
    vals = out["xcross_attempt"].dropna()
    assert ((vals >= 0) & (vals <= 1)).all()
    assert seen["tolerance_m"] == 2.5  # R3: carrier params read from model metadata, not library default


def test_compute_model_none_uses_bundled_default():
    """Post-PR-B (Task 13 bundling): model=None resolves to the bundled 'default' (the production
    path) and produces the column -- pre-weights this raised FileNotFoundError. An unsupported model
    TYPE still raises TypeError (the _resolve_model contract)."""
    frames, _ = _mini_match()
    out = xc.compute_xcross_attempt(frames, model=None, home_team_id="A")
    assert "xcross_attempt" in out.columns
    with pytest.raises(TypeError):
        xc.compute_xcross_attempt(frames, model=123, home_team_id="A")  # unsupported type


def test_compute_actions_populate_score_differential(monkeypatch):
    """PA-H1: at serve, passing actions= must realize score_differential (else NaN by design)."""
    from silly_kicks.spadl import config as spc

    frames, actions, model = _fit_on_mini()
    goal = pd.DataFrame(
        {
            "game_id": ["g"],
            "period_id": [1],
            "team_id": ["A"],
            "time_seconds": [0.0],
            "type_id": [spc.actiontype_id["shot"]],
            "result_id": [spc.result_id["success"]],
        }
    )
    actions2 = pd.concat([actions.assign(result_id=spc.result_id["success"]), goal], ignore_index=True)
    import silly_kicks.tracking._xcross_attempt as mod

    seen_scores: list = []
    real = mod.extract_xcross_features
    monkeypatch.setattr(
        mod,
        "extract_xcross_features",
        lambda *a, **k: seen_scores.append(k.get("score_differential")) or real(*a, **k),
    )
    xc.compute_xcross_attempt(frames, model=model, home_team_id="A", actions=actions2)
    assert any(s is not None and s == s for s in seen_scores)  # at least one non-NaN score reached extract


# --- Task 8: add_xcross_attempt ---


def test_add_xcross_aggregator():
    frames, _actions, model = _fit_on_mini()
    spadl_actions = pd.DataFrame(
        {
            "game_id": ["g"],
            "period_id": [1],
            "team_id": ["A"],
            "time_seconds": [0.4],
            "type_id": [0],
            "action_id": [0],
        }
    )
    out = xc.add_xcross_attempt(spadl_actions, frames, model=model, home_team_id="A")
    assert "xcross_attempt" in out.columns


def test_add_xcross_nan_safe():
    frames, _actions, model = _fit_on_mini()
    spadl_actions = pd.DataFrame(
        {
            "game_id": ["g"],
            "period_id": [1],
            "team_id": [np.nan],
            "time_seconds": [0.4],
            "type_id": [0],
            "action_id": [0],
        }
    )
    out = xc.add_xcross_attempt(spadl_actions, frames, model=model, home_team_id="A")
    assert pd.isna(out["xcross_attempt"].iloc[0])  # NaN id -> NaN out, no crash


def test_add_xcross_has_nan_safe_marker():
    assert getattr(xc.add_xcross_attempt, "_nan_safe", False) is True


# --- Task 9: xcross_attempt_xfns ---


def test_xfns_factory_columns_and_marker():
    _, _, model = _fit_on_mini()
    fns = xc.xcross_attempt_xfns(model=model, home_team_id="A")
    assert len(fns) == 1
    assert getattr(fns[0], "_frame_aware", False) is True


def test_xfns_silent_nan_on_frames_none():
    """M4: ACTUALLY invoke the closure with frames=None and assert the 3-col NaN contract."""
    _, _, model = _fit_on_mini()
    fn = xc.xcross_attempt_xfns(model=model, home_team_id="A")[0]
    states = [pd.DataFrame(index=[0, 1])]  # one gamestate slot; frames=None -> introspection path
    result = fn(states, None)
    assert list(result.columns) == ["xcross_attempt_a0", "xcross_attempt_a1", "xcross_attempt_a2"]
    assert result.isna().all(axis=None)
