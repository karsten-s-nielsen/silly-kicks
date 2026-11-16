"""TF-79 contact scorer (scorer spec S2): the option features, the credit step, the outputs, the parameters and the
artifact (fit / serve / save / load with the ADR-050 feature contract)."""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _ball_contact as bc
from silly_kicks.tracking import _contact_scorer as cs
from tests.tracking.test_ball_contact import _p, _scene

_KICK = [(0.0, 1.0, (60.0, 34.0), (60.0, 34.0)), (1.0, 2.0, (60.0, 34.0), (75.0, 34.0))]  # rest, then 15 m/s along +x


def _touches(t: float) -> float:
    """The time stub: 0.6 at the touch near 1.0 s, 0.8 at the one near 1.2 s, 0.1 at the noise peaks of a
    piecewise-linear scene."""
    return 0.6 if abs(t - 1.0) <= 0.05 else 0.8 if abs(t - 1.2) <= 0.05 else 0.1


class _Stub:
    """A hand-set scorer with the v2 serving surface (``predict_proba``, ``check_params``, ``decode_window_s``): a
    logistic link over named features (a missing value reads 0), or a probability per peak time (``by_time``)."""

    def __init__(self, coef=None, intercept=0.0, *, decode_window_s=0.0, by_time=None):
        self.coef, self.intercept, self.by_time = dict(coef or {}), float(intercept), by_time
        self.decode_window_s = float(decode_window_s)

    def predict_proba(self, features):
        if self.by_time is not None:
            return np.array([self.by_time(float(t)) for t in features["time_seconds"]], dtype=float)
        s = np.full(len(features), self.intercept)
        for name, c in self.coef.items():
            s = s + c * features[name].fillna(0.0).to_numpy(dtype=float)
        return 1.0 / (1.0 + np.exp(-s))

    def check_params(self, params):
        return None


def _scorer(coef: dict[str, float], intercept: float = 0.0, *, decode_window_s: float = 0.0) -> _Stub:
    """A hand-set scorer whose coefficients read directly in feature units; no decoding unless asked."""
    return _Stub(coef, intercept, decode_window_s=decode_window_s)


def _table(frames, **kw):
    return cs.option_table(frames, _p(min_velocity_change_ms=0.0, **kw))


def _at(table, t, tol=0.05):
    return table[(table["time_seconds"] - t).abs() <= tol]


# --------------------------------------------------------------------------- features (S2.2)
def test_kick_from_rest_features():
    t = _at(_table(_scene(_KICK, {7: ("T1", 60.0, 34.6)})), 1.0)
    assert len(t) == 1
    r = t.iloc[0]
    assert r["speed_in"] == pytest.approx(0.0, abs=1e-9)
    assert r["speed_out"] == pytest.approx(15.0, rel=0.05)
    assert r["turn"] == 0.0  # a speed below 1e-9 gives turn 0
    assert r["dv"] == pytest.approx(15.0, rel=0.05)


def test_deflection_turns_a_right_angle():
    legs = [(0.0, 1.0, (50.0, 34.0), (60.0, 34.0)), (1.0, 2.0, (60.0, 34.0), (60.0, 44.0))]
    r = _at(_table(_scene(legs, {7: ("T1", 60.0, 34.5)})), 1.0).iloc[0]
    assert r["turn"] == pytest.approx(math.pi / 2, abs=0.05)


def test_approach_closing_opening_and_absent():
    f = _scene(_KICK, {7: ("T1", 60.0, 34.6), 8: ("T2", 61.0, 35.0)})
    moving = f["player_id"] == 7
    f.loc[moving, "x"] = 58.0 + 2.0 * f.loc[moving, "time_seconds"]  # player 7 runs onto the ball: closing
    opening = f["player_id"] == 8
    f.loc[opening, "y"] = 33.0 + 2.0 * f.loc[opening, "time_seconds"]  # player 8 drifts away: opening
    t = _at(cs.option_table(f, _p(min_velocity_change_ms=0.0)), 1.0)
    by = {int(r["player_id"]): r for _, r in t.iterrows()}
    assert by[7]["approach"] < 0 < by[8]["approach"]
    late = _scene(_KICK, {9: ("T1", 60.0, 34.6)})
    early_9 = late["player_id"].eq(9).fillna(False).to_numpy(dtype=bool) & (late["time_seconds"] < 0.99).to_numpy()
    late = late[~early_9]  # no frame of player 9 before t - span (ball rows untouched)
    assert _at(cs.option_table(late, _p(min_velocity_change_ms=0.0)), 1.0)["approach"].tolist() == [0.0]


@pytest.mark.parametrize(
    ("players", "expected"),
    [
        ({1: ("T1", 60.0, 35.0)}, [(0, 5.0)]),  # one player: no next player -> 5.0
        ({1: ("T1", 60.0, 35.0), 2: ("T2", 60.0, 36.5)}, [(0, 1.5), (1, 5.0)]),
        ({1: ("T1", 60.0, 35.0), 2: ("T2", 60.0, 36.5), 3: ("T1", 60.0, 39.0)}, [(0, 1.5), (1, 2.5), (2, 5.0)]),
        (
            {1: ("T1", 60.0, 35.0), 2: ("T2", 60.0, 36.5), 3: ("T1", 60.0, 39.0), 4: ("T2", 60.0, 41.0)},
            [(0, 1.5), (1, 2.5), (2, 2.0)],  # three options; the 3rd's gap is the distance to the 4th
        ),
        ({1: ("T1", 60.0, 35.0), 2: ("T2", 60.0, 47.0)}, [(0, 5.0), (1, 5.0)]),  # the cap at 5.0
    ],
)
def test_rank_and_gap(players, expected):
    t = _at(_table(_scene(_KICK, players)), 1.0).sort_values("rank")
    assert [(int(r), round(float(g), 6)) for r, g in zip(t["rank"], t["gap"], strict=True)] == expected


@pytest.mark.parametrize(
    ("z", "z_source", "finite"),
    [(0.3, "measured", True), (np.nan, "measured", False), (0.3, "estimated", False), (0.3, "absent", False)],
)
def test_height_reading(z, z_source, finite):
    # Scorer spec r7 2.2: a non-measured or missing height is a missing value (no imputation).
    t = _at(_table(_scene(_KICK, {7: ("T1", 60.0, 34.6)}, z=z, z_source=z_source)), 1.0)
    assert bool(np.isfinite(t["z"].iloc[0])) is finite
    assert finite or pd.isna(t["z"].iloc[0])


# --------------------------------------------------------------------------- control features (r7 S2.2)
_ACROSS = [(0.0, 1.0, (50.0, 34.0), (60.0, 34.0)), (1.0, 2.0, (60.0, 34.0), (60.0, 24.0))]  # 10 m/s, then 90° to -y
_STOP = [(0.0, 1.0, (50.0, 34.0), (60.0, 34.0)), (1.0, 2.0, (60.0, 34.0), (60.0, 34.0))]  # 10 m/s, then at rest


def _ctrl(frames, t=1.0):
    return {int(r["player_id"]): r for _, r in _at(_table(frames), t).iterrows()}


def test_the_control_features_count():
    assert len(cs.FEATURE_NAMES) == 16 and cs.CONTROL_FEATURE_NAMES == (
        "after_ball_speed",
        "after_rel_disp",
        "after_dist_mean",
        "after_dist_end",
        "before_dist_min",
        "before_rel_disp",
        "speed_drop",
    )
    assert cs.CONTROL_WINDOW_S == 0.4


def test_a_receiver_who_stops_the_ball():
    r = _ctrl(_scene(_STOP, {7: ("T1", 60.0, 34.3)}))[7]
    assert r["after_ball_speed"] == pytest.approx(0.0, abs=1e-9)
    assert r["speed_drop"] == pytest.approx(10.0, rel=1e-6)
    assert r["after_rel_disp"] == pytest.approx(0.0, abs=1e-9)
    assert r["after_dist_mean"] == pytest.approx(0.3, abs=1e-9) and r["after_dist_end"] == pytest.approx(0.3, abs=1e-9)


def test_a_ball_that_passes_a_bystander():
    r = _ctrl(_scene(_ACROSS, {8: ("T2", 60.0, 35.0)}))[8]
    assert r["before_dist_min"] == pytest.approx(1.0, abs=1e-9)
    assert r["after_dist_end"] == pytest.approx(5.0, abs=1e-9)  # the sample at exactly t + C is inside
    assert r["speed_drop"] == pytest.approx(0.0, abs=1e-6)
    assert r["after_ball_speed"] == pytest.approx(10.0, rel=1e-6)
    # Review B CD-SB-100 (A7): the mean leaves out k: distances 1 + 0.4 i for i = 1..10 average 3.2 (with k: 3.0).
    assert r["after_dist_mean"] == pytest.approx(3.2, abs=1e-9)


def test_the_relative_rate_uses_the_last_finite_option_sample():
    # Review B CD-SB-100 (A9): the endpoints are the first and the last sample where both are finite, so a missing
    # option row at t + C moves the endpoint to t + C - one sample instead of making the rate missing.
    f = _scene(_ACROSS, {8: ("T2", 60.0, 35.0)})
    last = f["player_id"].eq(8).fillna(False).to_numpy(dtype=bool) & np.isclose(f["time_seconds"].to_numpy(), 1.4)
    r = _ctrl(f[~last])[8]
    assert r["after_rel_disp"] == pytest.approx(10.0, rel=1e-6)
    assert r["after_dist_end"] == pytest.approx(math.hypot(0.0, 1.0 + 0.4 * 9), abs=1e-9)  # the 1.36 s sample


def test_the_relative_rate_needs_a_finite_ball_at_its_endpoints():
    # Review A CD-IMPL-20 (A5): a non-finite ball at the last after sample moves the endpoint, like a missing option.
    f = _scene(_ACROSS, {8: ("T2", 60.0, 35.0)})
    hole = f["is_ball"].to_numpy(dtype=bool) & np.isclose(f["time_seconds"].to_numpy(dtype=float), 1.4)
    f.loc[hole, ["x", "y"]] = np.nan
    r = _ctrl(f)[8]
    assert r["after_rel_disp"] == pytest.approx(10.0, rel=1e-6)
    assert pd.isna(r["after_ball_speed"])  # the path speed, by contrast, is missing with a hole in the set


def test_the_relative_rate_subtracts_the_options_displacement():
    # Review A CD-IMPL-20 (A4): a receiver who carries the ball (ball and player move together after the peak) has a
    # relative rate near 0, although the ball moves at 3 m/s.
    legs = [(0.0, 1.0, (50.0, 34.0), (60.0, 34.0)), (1.0, 2.0, (60.0, 34.0), (63.0, 34.0))]
    f = _scene(legs, {7: ("T1", 60.0, 34.3)})
    mover = f["player_id"].eq(7).fillna(False).to_numpy(dtype=bool)
    t = f["time_seconds"].to_numpy(dtype=float)
    f.loc[mover, "x"] = 60.0 + 3.0 * np.clip(t[mover] - 1.0, 0.0, None)
    r = _ctrl(f)[7]
    assert r["after_ball_speed"] == pytest.approx(3.0, rel=1e-6)
    assert r["after_rel_disp"] == pytest.approx(0.0, abs=1e-6)
    assert r["after_dist_mean"] == pytest.approx(0.3, abs=1e-6)


def test_the_tracks_are_read_per_period():
    # Review A C71 (A8): the frame ids repeat across the periods; each period reads its own player rows.
    p1 = _scene(_STOP, {7: ("T1", 60.0, 34.3)})
    p2 = _scene(_STOP, {7: ("T1", 60.0, 44.0)}).assign(period_id=2)
    t = cs.option_table(pd.concat([p1, p2], ignore_index=True), _p(min_velocity_change_ms=0.0))
    at = t[(t["time_seconds"] - 1.0).abs() <= 0.05].set_index("period_id")
    assert at.loc[1, "after_dist_mean"] == pytest.approx(0.3, abs=1e-9)
    assert at.loc[2, "after_dist_mean"] == pytest.approx(10.0, abs=1e-9)


def test_the_before_window_bound():
    # Ball at (56, 34) at exactly t - C = 0.6 s; at (55.6, 34) at 0.56 s, just beyond the window.
    rows = _ctrl(_scene(_ACROSS, {8: ("T2", 60.0, 35.0), 9: ("T1", 56.0, 34.1), 10: ("T2", 55.6, 34.1)}))
    assert rows[9]["before_dist_min"] == pytest.approx(0.1, abs=1e-9)  # at t - C: inside
    assert rows[10]["before_dist_min"] == pytest.approx(math.hypot(0.4, 0.1), abs=1e-9)  # nearest sample inside


def test_a_player_without_rows_after_the_peak():
    f = _scene(_ACROSS, {8: ("T2", 60.0, 35.0)})
    gone = f["player_id"].eq(8).fillna(False).to_numpy(dtype=bool) & (f["time_seconds"] > 1.0 + 1e-9).to_numpy()
    r = _ctrl(f[~gone])[8]
    assert pd.isna(r["after_dist_mean"]) and pd.isna(r["after_dist_end"]) and pd.isna(r["after_rel_disp"])
    assert r["before_dist_min"] == pytest.approx(1.0, abs=1e-9)


def test_a_non_finite_ball_sample_in_the_after_window():
    f = _scene(_ACROSS, {8: ("T2", 60.0, 35.0)})
    hole = f["is_ball"].to_numpy(dtype=bool) & np.isclose(f["time_seconds"].to_numpy(dtype=float), 1.2)
    f.loc[hole, ["x", "y"]] = np.nan
    r = _ctrl(f)[8]
    assert pd.isna(r["after_ball_speed"]) and pd.isna(r["speed_drop"])
    assert np.isfinite(r["after_dist_mean"]) and r["after_dist_end"] == pytest.approx(5.0, abs=1e-9)


def _held(frames_half_rate):
    """Sample-and-hold upsampling: every 12.5 Hz row emitted twice at 25 Hz (the GS ``balls`` channel)."""
    a = frames_half_rate.copy()
    b = frames_half_rate.copy()
    a["frame_id"] = a["frame_id"] * 2
    b["frame_id"] = b["frame_id"] * 2 + 1
    out = pd.concat([a, b], ignore_index=True)
    out["time_seconds"] = out["frame_id"] / 25.0
    out["frame_rate"] = 25.0
    return out.sort_values(["frame_id", "is_ball"], ascending=[True, False], kind="stable").reset_index(drop=True)


def test_a_held_duplicate_changes_no_control_feature():
    # The control features read the deduped samples, so the sample-and-hold copy gives the same values at every
    # peak the two share. (The v1 velocity window follows frame_rate, so a peak can move by one sample: not compared.)
    plain = _scene(_ACROSS, {8: ("T2", 60.0, 35.0)}, fps=12.5)
    a = _table(plain).set_index(_table(plain)["time_seconds"].round(6))
    b = _table(_held(plain)).set_index(_table(_held(plain))["time_seconds"].round(6))
    shared = a.index.intersection(b.index)
    assert len(shared) >= 4
    for name in cs.CONTROL_FEATURE_NAMES:
        assert a.loc[shared, name].to_numpy() == pytest.approx(b.loc[shared, name].to_numpy(), abs=1e-9), name


def test_the_probe_has_six_peaks_and_no_missing_feature():
    t = cs.option_table(cs.contact_scorer_probe_frames(), bc.BallContactParams(min_velocity_change_ms=0.0))
    assert np.round(np.sort(t["time_seconds"].unique()), 2).tolist() == [1.0, 1.24, 1.36, 1.6, 1.76, 2.16]
    assert np.isfinite(t[list(cs.FEATURE_NAMES)].to_numpy(dtype=float)).all()


# --------------------------------------------------------------------------- credit and outputs (S2.1, S2.5)
def _detect(frames, scorer, *, monkeypatch, threshold=0.5, **kw):
    monkeypatch.setattr(cs, "resolve_scorer", lambda spec: scorer)
    return bc.detect_ball_contacts(
        frames, params=_p(min_velocity_change_ms=0.0, contact_scorer="x", min_contact_probability=threshold, **kw)
    )


def test_highest_probability_option_wins(monkeypatch):
    f = _scene(_KICK, {7: ("T1", 60.0, 34.6), 8: ("T2", 60.0, 37.0)})
    near = _at(_detect(f, _scorer({"dist": -2.0}, 2.0), monkeypatch=monkeypatch), 1.0)
    far = _at(_detect(f, _scorer({"dist": 2.0}, -2.0), monkeypatch=monkeypatch), 1.0)
    assert near["player_id"].tolist() == [7] and far["player_id"].tolist() == [8]


def test_threshold_is_inclusive(monkeypatch):
    f = _scene(_KICK, {7: ("T1", 60.0, 34.6)})
    p = float(_scorer({"dist": -1.0}).predict_proba(_at(_table(f), 1.0))[0])
    at = _at(_detect(f, _scorer({"dist": -1.0}), threshold=p, monkeypatch=monkeypatch), 1.0)
    above = _at(
        _detect(f, _scorer({"dist": -1.0}), threshold=float(np.nextafter(p, 1.0)), monkeypatch=monkeypatch), 1.0
    )
    assert at["contact_source"].tolist() == ["velocity_break"]
    assert above["contact_source"].tolist() == ["unresolved"] and pd.isna(above["confidence"].iloc[0])


def test_scorer_mode_outputs(monkeypatch):
    f = _scene(_KICK, {7: ("T1", 60.0, 34.6)})
    sc = _scorer({"dist": -1.0}, 1.0)
    out = _at(_detect(f, sc, monkeypatch=monkeypatch), 1.0).iloc[0]
    expected = float(sc.predict_proba(_at(_table(f), 1.0))[0])
    assert out["confidence"] == pytest.approx(expected) and 0.0 <= out["confidence"] <= 1.0
    assert out["contact_source"] == "velocity_break"
    assert out["velocity_change_ms"] == pytest.approx(float(_at(_table(f), 1.0)["dv"].iloc[0]))


def test_no_players_is_unresolved(monkeypatch):
    out = _at(_detect(_scene(_KICK, {}), _scorer({}, 5.0), monkeypatch=monkeypatch), 1.0)
    assert out["contact_source"].tolist() == ["unresolved"]


def test_measured_ball_above_reach_is_never_credited(monkeypatch):
    f = _scene(_KICK, {7: ("T1", 60.0, 34.6)}, z=3.0)
    out = _at(_detect(f, _scorer({}, 9.0), monkeypatch=monkeypatch), 1.0)
    assert out["contact_source"].tolist() == ["unresolved"]


def test_merge_runs_before_scoring(monkeypatch):
    legs = [
        (0.0, 1.0, (50.0, 34.0), (60.0, 34.0)),
        (1.0, 1.2, (60.0, 34.0), (62.0, 36.0)),
        (1.2, 2.2, (62.0, 36.0), (72.0, 36.0)),
    ]
    f = _scene(legs, {7: ("T1", 60.0, 34.0), 9: ("T2", 62.0, 36.0)})
    wide = cs.option_table(f, _p(min_velocity_change_ms=0.0, merge_window_s=0.3))
    narrow = cs.option_table(f, _p(min_velocity_change_ms=0.0, merge_window_s=0.1))
    assert _at(wide, 1.1, tol=0.15)["peak_id"].nunique() == 1  # one merged peak, scored once
    assert _at(narrow, 1.1, tol=0.15)["peak_id"].nunique() == 2


# --------------------------------------------------------------------------- params (S2.5)
def test_params_scorer_field_is_a_value():
    a = bc.BallContactParams(contact_scorer="default", min_contact_probability=0.4)
    b = bc.BallContactParams(contact_scorer="default", min_contact_probability=0.4)
    assert a == b and hash(a) == hash(b)
    assert bc.BallContactParams().contact_scorer is None  # commit 4: the reach gate by default; no I/O at import


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_min_contact_probability_check(bad):
    with pytest.raises(ValueError, match="min_contact_probability"):
        bc.BallContactParams(min_contact_probability=bad)
    bc.BallContactParams(min_contact_probability=0.001)
    bc.BallContactParams(min_contact_probability=0.999)


def test_default_scorer_loads_lazily(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_DEFAULT_DIR", tmp_path / "absent")
    cs._load_cached.cache_clear()
    params = bc.BallContactParams(contact_scorer="default")  # constructing it reads nothing
    with pytest.raises(FileNotFoundError, match="absent"):
        bc.detect_ball_contacts(_scene(_KICK), params=params)


def test_a_differing_declared_window_raises(tmp_path):
    sc = _fitted()
    sc.save(tmp_path / "m")
    cs._load_cached.cache_clear()
    with pytest.raises(ValueError, match="kinematic_window_s"):
        bc.detect_ball_contacts(
            _scene(_KICK, {7: ("T1", 60.0, 34.6)}),
            params=_p(contact_scorer=str(tmp_path / "m"), min_velocity_change_ms=0.0, kinematic_window_s=0.1),
        )


# --------------------------------------------------------------------------- model and artifact (r7 S2.3)
def _xy(seed=0, n=400):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({name: rng.normal(size=n) for name in cs.FEATURE_NAMES})
    x.loc[: n // 10, "z"] = np.nan
    x.loc[n // 2 : n // 2 + 20, "after_dist_mean"] = np.nan
    y = ((x["dist"].fillna(0) < 0) ^ (x["speed_drop"].fillna(0) > 1.0)).astype(int).to_numpy()
    return x, y


def _fitted() -> cs.ContactScorer:
    x, y = _xy()
    return cs.ContactScorer().fit(x, y)


def test_numpy_serving_equals_sklearn():
    from sklearn.ensemble import HistGradientBoostingClassifier

    x, y = _xy(seed=1)
    sc = cs.ContactScorer().fit(x, y)
    ref = HistGradientBoostingClassifier(**cs.HGB_SETTINGS).fit(x[list(cs.FEATURE_NAMES)].to_numpy(float), y)
    probe = pd.concat([x, x.head(5).assign(dist=np.nan, gap=np.nan)], ignore_index=True)  # missing values too
    want = ref.predict_proba(probe[list(cs.FEATURE_NAMES)].to_numpy(float))[:, 1]
    assert np.abs(sc.predict_proba(probe) - want).max() <= 1e-12


def test_the_settings_are_the_spec():
    assert cs.HGB_SETTINGS == {
        "max_iter": 200,
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "early_stopping": False,
        "random_state": 0,
    }


def test_two_fits_are_equal():
    assert _fitted().to_dict() == _fitted().to_dict()


def test_the_fit_refuses_an_export_that_does_not_serve_like_sklearn(monkeypatch):
    real = cs._export_trees

    def corrupt(clf):
        trees, baseline = real(clf)
        return trees, baseline + 1e-6

    monkeypatch.setattr(cs, "_export_trees", corrupt)
    x, y = _xy()
    with pytest.raises(cs.ContactScorerIntegrityError, match="parity"):
        cs.ContactScorer().fit(x, y)


def test_the_fit_records_the_sklearn_version():
    import sklearn

    assert _fitted().sklearn_version == sklearn.__version__


def test_round_trip_and_only_parameters(tmp_path):
    sc = _fitted()
    assert cs.ContactScorer.from_dict(sc.to_dict()).to_dict() == sc.to_dict()
    sc.save(tmp_path / "m")
    model = json.loads((tmp_path / "m" / "model.json").read_text(encoding="utf-8"))
    assert set(model) == {"version", "feature_names", "trees", "baseline", "decode_window_s", "feature_contract"}
    assert model["version"] == cs.SCORER_VERSION == "contact-scorer-2"
    assert model["decode_window_s"] == cs.DECODE_WINDOW_S == 0.3
    keys = {"feature", "threshold", "left", "right", "missing_left", "is_leaf", "value"}
    assert len(model["trees"]) == cs.HGB_SETTINGS["max_iter"]
    for tree in model["trees"]:
        assert set(tree) == keys  # no count, no gain: no per-sample statistic
        assert len({len(v) for v in tree.values()}) == 1
    back = cs.ContactScorer.load(tmp_path / "m")
    x = _xy(seed=3, n=20)[0]
    assert np.array_equal(back.predict_proba(x), sc.predict_proba(x))


def test_loader_refuses_a_changed_file(tmp_path):
    _fitted().save(tmp_path / "m")
    p = tmp_path / "m" / "model.json"
    p.write_text(p.read_text(encoding="utf-8").replace('"baseline": ', '"baseline": 1.0 + '), encoding="utf-8")
    with pytest.raises(cs.ContactScorerIntegrityError, match="SHA256SUMS"):
        cs.ContactScorer.load(tmp_path / "m")


@pytest.mark.parametrize("bad", ["missing", float("nan"), -0.1])
def test_loader_refuses_a_bad_decode_window(tmp_path, bad):
    d = tmp_path / "m"
    _fitted().save(d)
    _rewrite(d, lambda m: m.pop("decode_window_s") if bad == "missing" else m.update(decode_window_s=bad))
    with pytest.raises(cs.ContactScorerIntegrityError, match="decode_window_s"):
        cs.ContactScorer.load(d)


@pytest.mark.parametrize(("window", "first_credited"), [(0.0, True), (0.5, False)])
def test_the_detector_decodes_with_the_window_in_model_json(tmp_path, monkeypatch, window, first_credited):
    # Review A CD-IMPL-21: model.json is the single source of the decoding window (scorer spec r7 2.1), not
    # DECODE_WINDOW_S. A non-default window survives save and load, and the detector decodes with it.
    d = tmp_path / "m"
    sc = _fitted()
    sc.decode_window_s = window
    sc.save(d)
    loaded = cs.ContactScorer.load(d)
    assert loaded.decode_window_s == window != cs.DECODE_WINDOW_S
    monkeypatch.setattr(
        loaded, "predict_proba", lambda features: np.array([_touches(float(t)) for t in features["time_seconds"]])
    )
    monkeypatch.setattr(cs, "resolve_scorer", lambda spec: loaded)
    out = bc.detect_ball_contacts(
        _scene(_TWO, {7: ("T1", 60.0, 34.6), 8: ("T2", 60.5, 32.0)}),
        params=_p(min_velocity_change_ms=0.0, contact_scorer="x", min_contact_probability=0.5),
    )
    first = _at(out, 1.0, tol=0.06)["contact_source"].tolist()
    assert first == (["velocity_break"] if first_credited else ["unresolved"])
    assert _at(out, 1.2, tol=0.06)["contact_source"].tolist() == ["velocity_break"]


def _rewrite(d, edit):
    m = json.loads((d / "model.json").read_text(encoding="utf-8"))
    edit(m)
    (d / "model.json").write_text(json.dumps(m, sort_keys=True), encoding="utf-8")
    cs._write_checksums(d)


def test_loader_refuses_a_constant_and_a_fingerprint_mismatch(tmp_path):
    for name, edit in (
        ("const", lambda m: m["feature_contract"]["constants"].update(GAP_CAP_M=4.0)),
        ("fp", lambda m: m["feature_contract"]["fingerprint"].__setitem__(0, 123.0)),
    ):
        d = tmp_path / name
        _fitted().save(d)
        _rewrite(d, edit)
        with pytest.raises(cs.ContactScorerIntegrityError):
            cs.ContactScorer.load(d)


def test_loader_warns_on_a_missing_contract(tmp_path):
    from silly_kicks.tracking import MissingFeatureContractWarning

    d = tmp_path / "m"
    _fitted().save(d)
    _rewrite(d, lambda m: m.pop("feature_contract"))
    with pytest.warns(MissingFeatureContractWarning):
        cs.ContactScorer.load(d)


def test_probe_edit_takes_the_probe_change_path(monkeypatch, tmp_path):
    from silly_kicks.tracking import UnverifiableFeatureContractWarning

    d = tmp_path / "m"
    _fitted().save(d)
    original = cs.contact_scorer_probe_frames
    monkeypatch.setattr(cs, "contact_scorer_probe_frames", lambda: original().assign(x=lambda f: f["x"] + 0.5))
    with pytest.warns(UnverifiableFeatureContractWarning):
        cs.ContactScorer.load(d)  # warns, does not raise


def test_contact_scorer_contract_declares_its_constants(monkeypatch):
    base = cs.feature_contract_for(bc.BallContactParams())
    assert set(base["constants"]) == {
        "K_OPTIONS",
        "GAP_CAP_M",
        "CONTROL_WINDOW_S",
        "kinematic_window_s",
        "min_candidate_speed_ms",
    }
    monkeypatch.setattr(cs, "GAP_CAP_M", 0.5)
    assert cs.feature_contract_for(bc.BallContactParams())["fingerprint"] != base["fingerprint"]
    monkeypatch.setattr(cs, "GAP_CAP_M", 5.0)
    monkeypatch.setattr(cs, "K_OPTIONS", 2)
    assert cs.feature_contract_for(bc.BallContactParams())["fingerprint"] != base["fingerprint"]
    monkeypatch.setattr(cs, "K_OPTIONS", 3)
    monkeypatch.setattr(cs, "CONTROL_WINDOW_S", 0.3)
    assert cs.feature_contract_for(bc.BallContactParams())["fingerprint"] != base["fingerprint"]
    monkeypatch.setattr(cs, "CONTROL_WINDOW_S", 0.4)
    for field, params in (
        ("kinematic_window_s", bc.BallContactParams(kinematic_window_s=0.12)),
        ("min_candidate_speed_ms", bc.BallContactParams(min_candidate_speed_ms=16.0)),
    ):
        assert cs.feature_contract_for(params)["fingerprint"] != base["fingerprint"], field


def test_probe_is_deterministic_and_finite():
    a, b = cs.contact_scorer_probe_frames(), cs.contact_scorer_probe_frames()
    pd.testing.assert_frame_equal(a, b)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        feats = cs.option_table(a, bc.BallContactParams(min_velocity_change_ms=0.0))
    assert len(feats) >= 3 and np.isfinite(feats[list(cs.FEATURE_NAMES)].fillna(0.0).to_numpy()).all()


def test_the_pre_filter_is_not_part_of_the_contract(tmp_path):
    # The driver saves with min_velocity_change_ms=0.0 (threshold-free peaks); load() must still verify. The
    # fingerprint covers the threshold-free extractor, so a pre-filter neither changes nor fails it.
    a = cs.feature_contract_for(bc.BallContactParams(min_velocity_change_ms=0.0))
    b = cs.feature_contract_for(bc.BallContactParams(min_velocity_change_ms=3.0))
    assert a == b
    _fitted().save(tmp_path / "m", params=bc.BallContactParams(min_velocity_change_ms=0.0))
    cs.ContactScorer.load(tmp_path / "m")


# --------------------------------------------------------------------------- review A (scorer impl): one credit gate
def _decide_table(prob, z=np.nan, rank=(0.0, 1.0)):
    return pd.DataFrame(
        {
            "peak_id": [0, 0],
            "period_id": [1, 1],
            "time_seconds": [1.0, 1.0],
            "player_id": [7, 8],
            "rank": list(rank),
            "z": [z, z],
            **{n: 0.0 for n in cs.FEATURE_NAMES if n not in ("rank", "z")},
        }
    ), np.asarray(prob, dtype=float)


def test_decide_threshold_is_inclusive_and_the_height_gate_holds():
    params = bc.BallContactParams(min_contact_probability=0.6)

    def credited(prob, **kw):
        opts, p = _decide_table(prob, **kw)
        return cs.decide_options(opts, p, params, decode_window_s=0.0)["credited"].tolist()

    assert credited([0.6, 0.2]) == [True]  # at the threshold: credited
    assert credited([0.59, 0.2]) == [False]
    assert credited([0.9, 0.2], z=2.7) == [False]  # a measured ball above 2.6 m
    assert credited([0.9, 0.2], z=2.5) == [True]


def test_ties_go_to_the_nearer_rank():
    opts, prob = _decide_table([0.7, 0.7], rank=(1.0, 0.0))  # player 8 is rank 0
    out = cs.decide_options(opts, prob, bc.BallContactParams(), decode_window_s=0.0)
    assert out["player_id"].tolist() == [8]


def test_detector_and_driver_credit_identically(monkeypatch):
    f = _scene(_ACROSS, {7: ("T1", 60.0, 34.6), 8: ("T2", 61.5, 35.0)})
    sc = _Stub(by_time=_touches, decode_window_s=0.3)
    params = _p(min_velocity_change_ms=0.0, contact_scorer="x", min_contact_probability=0.4)
    monkeypatch.setattr(cs, "resolve_scorer", lambda spec: sc)
    det = bc.detect_ball_contacts(f, params=params)
    t = cs.option_table(f, params)
    drv = cs.credit_options(t, sc.predict_proba(t), params, decode_window_s=sc.decode_window_s)
    assert det["contact_source"].astype(str).tolist() == drv["contact_source"].tolist()
    assert det["time_seconds"].tolist() == pytest.approx(drv["time_seconds"].tolist())


def test_the_detector_serves_the_control_features(monkeypatch):
    # Review B CD-SB-98: a scorer that READS the control features must credit identically on the detector path and
    # the driver path; a detector that served missing control features would credit every peak here.
    f = _scene(_STOP, {7: ("T1", 60.0, 34.3), 9: ("T2", 45.0, 30.0)})  # 7 stops the ball at 1.0 s; 9 is far away
    sc = _Stub({"after_dist_mean": -2.0, "before_dist_min": -1.0}, 2.0, decode_window_s=0.3)
    params = _p(min_velocity_change_ms=0.0, contact_scorer="x", min_contact_probability=0.5)
    monkeypatch.setattr(cs, "resolve_scorer", lambda spec: sc)
    det = bc.detect_ball_contacts(f, params=params)
    t = cs.option_table(f, params)
    assert t["after_dist_mean"].notna().all() and (t["after_dist_mean"] > 1.0).any()  # the features bite
    drv = cs.credit_options(t, sc.predict_proba(t), params, decode_window_s=sc.decode_window_s)
    assert det["contact_source"].astype(str).tolist() == drv["contact_source"].tolist()
    assert "unresolved" in drv["contact_source"].tolist() and "velocity_break" in drv["contact_source"].tolist()
    det_players = [None if pd.isna(p) else str(p) for p in det["player_id"]]
    drv_players = [None if p is None else str(p) for p in drv["player_id"]]
    assert det_players == drv_players


# --------------------------------------------------------------------------- the decoding (r7 S2.1 step 7)
def _peaks(rows):
    """rows: (peak_id, time, player, prob, z) -> one option per peak, and its probabilities."""
    opts = pd.DataFrame(
        {
            "peak_id": [r[0] for r in rows],
            "period_id": 1,
            "time_seconds": [r[1] for r in rows],
            "player_id": [r[2] for r in rows],
            "rank": 0.0,
            "z": [r[4] for r in rows],
            **{n: 0.0 for n in cs.FEATURE_NAMES if n not in ("rank", "z")},
        }
    )
    return opts, np.array([r[3] for r in rows], dtype=float)


def _decode(rows, window=0.3, threshold=0.5):
    opts, prob = _peaks(rows)
    out = cs.decide_options(opts, prob, bc.BallContactParams(min_contact_probability=threshold), decode_window_s=window)
    return dict(zip(out["peak_id"].tolist(), out["credited"].tolist(), strict=True))


def test_a_more_probable_neighbour_within_the_window_suppresses():
    assert _decode([(0, 1.0, 7, 0.6, np.nan), (1, 1.3, 8, 0.8, np.nan)]) == {0: False, 1: True}  # 0.30 s: inside
    assert _decode([(0, 1.0, 7, 0.6, np.nan), (1, 1.31, 8, 0.8, np.nan)]) == {0: True, 1: True}  # 0.31 s: outside


def test_an_equal_probability_keeps_only_the_earlier_peak():
    assert _decode([(0, 1.0, 7, 0.7, np.nan), (1, 1.2, 8, 0.7, np.nan)]) == {0: True, 1: False}


def test_an_uncredited_peak_still_suppresses():
    assert _decode([(0, 1.0, 7, 0.6, np.nan), (1, 1.2, 8, 0.8, 2.7)]) == {0: False, 1: False}  # 1 is too high
    assert _decode([(0, 1.0, 7, 0.3, np.nan), (1, 1.2, 8, 0.4, np.nan)], threshold=0.35) == {0: False, 1: True}
    assert _decode([(0, 1.0, 7, 0.6, np.nan), (1, 1.2, 8, 0.8, np.nan)], threshold=0.9) == {0: False, 1: False}


def test_a_window_of_zero_decodes_nothing():
    assert _decode([(0, 1.0, 7, 0.6, np.nan), (1, 1.2, 8, 0.8, np.nan)], window=0.0) == {0: True, 1: True}


def test_the_decoding_reads_the_period():
    opts, prob = _peaks([(0, 1.0, 7, 0.6, np.nan), (1, 1.1, 8, 0.8, np.nan)])
    opts.loc[1, "period_id"] = 2  # same time, another period: no suppression
    out = cs.decide_options(opts, prob, bc.BallContactParams(min_contact_probability=0.5), decode_window_s=0.3)
    assert out["credited"].tolist() == [True, True]


_TWO = [  # two touches 0.2 s apart: 10 m/s along +x, then -y, then +x
    (0.0, 1.0, (50.0, 34.0), (60.0, 34.0)),
    (1.0, 1.2, (60.0, 34.0), (60.0, 32.0)),
    (1.2, 2.2, (60.0, 32.0), (70.0, 32.0)),
]


def test_a_suppressed_peak_is_an_unresolved_row(monkeypatch):
    sc = _Stub(by_time=_touches, decode_window_s=0.3)
    out = _detect(_scene(_TWO, {7: ("T1", 60.0, 34.6), 8: ("T2", 60.5, 32.0)}), sc, monkeypatch=monkeypatch)
    first, second = _at(out, 1.0, tol=0.06).iloc[0], _at(out, 1.2, tol=0.06).iloc[0]
    assert first["contact_source"] == "unresolved" and pd.isna(first["player_id"]) and pd.isna(first["confidence"])
    assert second["contact_source"] == "velocity_break"


def test_a_peak_without_options_takes_no_part(monkeypatch):
    f = _scene(_TWO, {7: ("T1", 60.0, 34.6), 8: ("T2", 60.5, 32.0)})
    t = f["time_seconds"].to_numpy(dtype=float)
    blank = ~f["is_ball"].to_numpy(dtype=bool) & (np.abs(t - 1.2) <= 0.05)  # no player rows at the later peak
    sc = _Stub(by_time=_touches, decode_window_s=0.3)
    out = _detect(f[~blank], sc, monkeypatch=monkeypatch)
    assert _at(out, 1.0, tol=0.06)["contact_source"].tolist() == ["velocity_break"]  # nothing suppresses it
    assert _at(out, 1.2, tol=0.06)["contact_source"].tolist() == ["unresolved"]


def test_checksums_must_cover_the_model(tmp_path):
    d = tmp_path / "m"
    _fitted().save(d)
    (d / "SHA256SUMS").write_text("", encoding="utf-8")
    with pytest.raises(cs.ContactScorerIntegrityError, match=r"does not cover model\.json"):
        cs.ContactScorer.load(d)


def test_the_cache_keys_on_the_resolved_path(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cs.ContactScorer, "load", classmethod(lambda cls, d: calls.append(str(d)) or cls()))
    cs._load_cached.cache_clear()
    (tmp_path / "m").mkdir()
    cs.resolve_scorer(str(tmp_path / "m"))
    cs.resolve_scorer(str(tmp_path / "m" / ".." / "m"))  # the same directory, spelled differently
    cs._load_cached.cache_clear()
    assert len(calls) == 1
