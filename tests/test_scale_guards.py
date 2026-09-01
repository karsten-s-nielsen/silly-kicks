"""Scale-guard adopters (ADR-073): each of the 12 highest-risk primitives carries a growth (or
constant) guard proving it stays sub-quadratic at scale. See tests/_perf_structural.py and the
SCALE_GUARDED registry."""

import numpy as np
import pandas as pd
import pytest

from tests._perf_structural import assert_subquadratic_growth, call_counter, rows_scanned_counter


# ============================ rows_scanned growth adopters ============================
def test__pressure_at_entry_is_subquadratic():
    import silly_kicks.causal._confounders as C
    from tests.causal.test_confounders_perf import _frames_and_spells, _stub_add_pressure

    def measure(n):
        frames, spells = _frames_and_spells(n)
        with rows_scanned_counter() as c:
            C._pressure_at_entry(spells, frames, _stub_add_pressure)
        return c["n"]

    assert_subquadratic_growth(measure, sizes=(128, 256, 512), label="_pressure_at_entry")


def _games_scaling_gk_frames(n_games, n_frames=12):
    # n scales the number of GAMES (2 teams each), so the (game_id, team_id) group count = 2*n_games
    # scales and derive_goalkeepers' per-group rescan would be O(n^2). The earlier `_multi_team_frames(n)`
    # scaled frames within a FIXED 2x2 group set (loop count == 4 forever), so the guard could NOT
    # discriminate the regression (ADR-073 proof: exp 1.0 on the fix vs ~2.0 on the rescan; detection
    # stays intact -- one GK per (game, team)).
    rows = []
    for gi in range(n_games):
        game_id = f"g{gi}"
        for team_id in ("tA", "tB"):
            players = [
                {"player_id": f"{team_id}_gk", "x": 5.0, "y": 34.0},
                {"player_id": f"{team_id}_o1", "x": 50.0, "y": 34.0},
                {"player_id": f"{team_id}_o2", "x": 55.0, "y": 30.0},
            ]
            for _fid in range(n_frames):
                for p in players:
                    rows.append(
                        {
                            "game_id": game_id,
                            "team_id": team_id,
                            "player_id": p["player_id"],
                            "x": p["x"],
                            "y": p["y"],
                            "is_ball": False,
                            "is_goalkeeper": False,
                        }
                    )
    return pd.DataFrame(rows)


def test_derive_goalkeepers_is_subquadratic():
    from silly_kicks.tracking._gk_identification import derive_goalkeepers

    def measure(n):  # n = number of games
        frames = _games_scaling_gk_frames(n)
        with rows_scanned_counter() as c:
            derive_goalkeepers(frames)
        return c["n"]

    assert_subquadratic_growth(measure, sizes=(32, 64, 128), label="derive_goalkeepers")


def test_add_possessions_is_subquadratic():
    import silly_kicks.spadl.utils as spu
    from tests.test_benchmark import _make_spadl_actions

    def measure(n):
        actions = _make_spadl_actions(n)
        with rows_scanned_counter() as c:
            spu.add_possessions(actions)
        return c["n"]

    assert_subquadratic_growth(measure, sizes=(1500, 3000, 6000), label="add_possessions")


# ============================ #9 turnover (counting-array, inner-j) ============================
class _CountingArray:
    """1-D array wrapper counting element reads -- proxy for inner-scan work (spec Section 4.2)."""

    def __init__(self, arr, counts):
        self._a = np.asarray(arr)
        self._c = counts

    def __getitem__(self, i):
        self._c["n"] += 1
        return self._a[i]

    def __len__(self):
        return len(self._a)


def _turnover_fixture(n, *, window):
    from silly_kicks.xtgk._turnover import _equality_codes

    game = np.repeat(np.arange(n // 50 + 1), 50)[:n]
    t = (np.arange(n) % 50).astype(float)  # resets each match -> window binds
    poss = np.arange(n) // 3
    team = np.where((np.arange(n) // 5) % 2 == 0, 0, 1)
    typ = np.zeros(n, dtype=np.int64)
    typ[np.arange(n) % 17 == 0] = 1
    xg = np.zeros(n)
    turn = (np.arange(n) % 10 == 0).astype(bool)
    win = np.inf if window is None else float(window)
    return dict(
        turn=turn,
        game=_equality_codes(pd.Series(game)),
        poss=_equality_codes(pd.Series(poss)),
        team=_equality_codes(pd.Series(team)),
        typ=typ,
        xg=xg,
        t=t,
        shot=1,
        window=win,
    )


def measure_turnover_scan(n):
    from silly_kicks.xtgk._turnover import _opp_first_shot_scan

    f = _turnover_fixture(n, window=5.0)  # FINITE window -> breaks bind -> O(n)
    counts = {"n": 0}
    # Wrap the array read on every inner iteration (game_c[j] != game_c[i] is the first inner check).
    args = (
        f["turn"],
        _CountingArray(f["game"], counts),
        f["poss"],
        f["team"],
        f["typ"],
        f["xg"],
        f["t"],
        f["shot"],
        f["window"],
    )
    _opp_first_shot_scan(*args)  # type: ignore[arg-type]  # _CountingArray is duck-typed
    return counts["n"]


def test__opp_first_shot_scan_is_subquadratic():
    assert_subquadratic_growth(measure_turnover_scan, sizes=(256, 1024, 4096), label="turnover_scan")


def test_turnover_fixture_breaks_actually_bind():
    # S2b: prove the fixture's finite window makes the REAL kernel sub-quadratic (breaks bind), not
    # just that a triangular loop is quadratic -- work at 4096 must be far below the triangular bound.
    n = 4096
    work = measure_turnover_scan(n)
    assert work < 0.1 * n * n, f"breaks not binding: {work} vs triangular {n * n // 2}"


# ============================ #10 possession labels (_LocIndexer) ============================
def measure_possession_labels(n):
    import pandas.core.indexing as _idx

    from silly_kicks.vaep.labels import _scores_possession
    from tests.vaep.test_labels_possession_perf import _single_possession

    mp = pytest.MonkeyPatch()
    calls = call_counter(mp, _idx._LocIndexer, "__getitem__")
    try:
        _scores_possession(_single_possession(n), "xg")
    finally:
        mp.undo()
    return calls["n"]


def test__possession_labels_loc_is_subquadratic():
    # Vectorized path issues ZERO .loc in the hot path -> degenerate-BY-DESIGN. degenerate_ok=True;
    # the mandatory companion below proves the counter distinguishes.
    assert_subquadratic_growth(
        measure_possession_labels,
        sizes=(64, 256, 1024),
        degenerate_ok=True,
        label="possession_labels_loc",
    )


def test_possession_labels_ref_loop_is_superlinear():
    # MANDATORY companion (T1): the verbatim pre-ADR-068 _ref O(k^2) .loc loop must measure
    # super-linear -> proves the _LocIndexer counter distinguishes the vectorized path from the old.
    import pandas.core.indexing as _idx

    from silly_kicks.vaep.labels import _scores_possession  # noqa: F401
    from tests.vaep.test_labels_possession_perf import _ref, _single_possession

    def measure_ref(n):
        mp = pytest.MonkeyPatch()
        calls = call_counter(mp, _idx._LocIndexer, "__getitem__")
        try:
            _ref(_single_possession(n), "xg", col="scores", same_is_goal=True)
        finally:
            mp.undo()
        return calls["n"]

    with pytest.raises(AssertionError):
        assert_subquadratic_growth(measure_ref, sizes=(32, 64, 128), label="possession_ref_loc")


# ============================ #8 databricks (constant-query equality) ============================
def test_load_matches_query_count_is_constant_in_match_count(monkeypatch):
    import scripts._loader_databricks as ld
    from tests.scripts.test_loader_databricks_batch import _FakeConn

    def _count_queries(n_matches):
        seen = []
        all_frames = pd.DataFrame({"match_id": list(range(n_matches)), "frame_id": 0, "x": 1.0})
        all_events = pd.DataFrame({"match_id": list(range(n_matches)), "ev": 0})

        def _fake_query(cur, sql, params=None):
            seen.append(sql)
            return all_frames.copy() if "T_TRACK" in sql else all_events.copy()

        monkeypatch.setattr(ld, "_connect", lambda: _FakeConn())
        monkeypatch.setattr(ld, "_table", lambda p, kind: {"tracking": "T_TRACK", "events": "T_EVT"}[kind])
        monkeypatch.setattr(ld, "_convert", lambda p, e, f: (e, f, "home"))
        monkeypatch.setattr(ld, "_query_param", _fake_query)
        list(
            ld.load_matches(
                providers=["skillcorner"],
                match_ids={"skillcorner": [str(i) for i in range(n_matches)]},
                tracking_limit=None,
            )
        )
        return len(seen)

    assert _count_queries(2) == _count_queries(8) == 2  # one IN-list query per table, always


# ============================ off-ball adopters (#5, #7) ============================
# n scales the number of GAMES, NOT frames-within-one-game. Both kernels `group_rows(frames,
# "game_id")` then loop `for game_id in actions.groupby("game_id")`, so a rescan-in-loop is
# O(games x frames): it only bites -- and the guard can only DISCRIMINATE it -- when the group
# (game) count scales. A single-game fixture (the earlier shape) left the loop count at 1, so the
# regressed raw-`==` code stayed linear and the guard passed on the bug (ADR-073 discrimination
# proof; verified exp 1.0 on the fix vs ~1.9 on the rescan). Multi-game is also the realistic
# batched-corpus axis the lakehouse consumer that first hit ADR-068 actually grows along.
def _multi_game_off_ball(n_games, frames_per_game=6) -> tuple[pd.DataFrame, pd.DataFrame]:
    from tests.tracking.test_off_ball_runs import _make_action_at, _make_multi_frame_fixture

    players = [
        {"player_id": 10, "team_id": 1, "positions": [(50.0, 34.0)] * frames_per_game},
        {"player_id": 11, "team_id": 1, "positions": [(50.0 + (i % 5), 34.0) for i in range(frames_per_game)]},
        {"player_id": 20, "team_id": 2, "positions": [(80.0, 34.0)] * frames_per_game},
        {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102.0, 34.0)] * frames_per_game},
        {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3.0, 34.0)] * frames_per_game},
    ]
    frames_list, acts = [], []
    for g in range(1, n_games + 1):
        frames_list.append(
            _make_multi_frame_fixture(players=players, n_frames=frames_per_game, frame_rate=25.0, game_id=g)
        )
        acts.append(_make_action_at(time_seconds=2.0 / 25.0, player_id=10, team_id=1, game_id=g, action_id=g))
    return pd.concat(acts, ignore_index=True), pd.concat(frames_list, ignore_index=True)


def test__off_ball_runs_kernel_is_subquadratic():
    from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

    def measure(n):  # n = number of games
        actions, frames = _multi_game_off_ball(n)
        with rows_scanned_counter() as c:
            _off_ball_runs_kernel(actions, frames, home_team_id=1)
        return c["n"]

    assert_subquadratic_growth(measure, sizes=(32, 64, 128), label="_off_ball_runs_kernel")


def test_detect_off_ball_runs_is_subquadratic():
    from silly_kicks.tracking._run_values import detect_off_ball_runs

    def measure(n):  # n = number of games
        actions, frames = _multi_game_off_ball(n)
        with rows_scanned_counter() as c:
            detect_off_ball_runs(actions, frames)
        return c["n"]

    assert_subquadratic_growth(measure, sizes=(32, 64, 128), label="detect_off_ball_runs")


# ============================ opportunities (#2) ============================
def test_build_opportunities_is_subquadratic():
    import silly_kicks.causal.opportunities as O
    from tests.causal._fixtures import META, WIDE, actions

    def measure(n):
        from tests.causal._fixtures import frames as _frames

        times = [round(10.0 + i * 0.04, 2) for i in range(n)]
        f = _frames({t: 5 for t in times}, {t: WIDE for t in times})
        with rows_scanned_counter() as c:
            O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
        return c["n"]

    assert_subquadratic_growth(measure, sizes=(128, 256, 512), label="build_opportunities")


# ============================ skillcorner inference (#4) ============================
def test_infer_defensive_actions_is_subquadratic():
    from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

    def measure(n):
        # Spread across n distinct (period, team_id) groups -> each .get() retrieves a BOUNDED
        # group (realistic: a match has few regains per period/team). One giant group would be a
        # fixture artifact making even the O(n) fix look O(n^2). A regression to whole-table rescan
        # per row stays O(n^2) even here, so this fixture is the discriminating one.
        team = [f"t{i}" for i in range(n)]
        pp = pd.DataFrame(
            {
                "event_id": [f"pp_{i}" for i in range(n)],
                "period": [1] * n,
                "time_seconds": [10.0 + i for i in range(n)],
                "team_id": team,
                "player_id": [f"p{i}" for i in range(n)],
                "start_type": ["pass_interception"] * n,
                "x_start": [5.0] * n,
                "y_start": [3.0] * n,
            }
        )
        obe = pd.DataFrame(
            {
                "period": [1] * n,
                "time_seconds": [9.9 + i for i in range(n)],
                "team_id": team,
                "player_id": [f"q{i}" for i in range(n)],
                "end_type": ["direct_regain"] * n,
                "x_start": [4.0] * n,
                "y_start": [2.0] * n,
            }
        )
        with rows_scanned_counter() as c:
            infer_defensive_actions(pp, obe)
        return c["n"]

    assert_subquadratic_growth(measure, sizes=(128, 256, 512), label="infer_defensive_actions")


# ============================ #12 atomic add_possessions (n10) ============================
def test_atomic_add_possessions_is_subquadratic():
    import silly_kicks.atomic.spadl.utils as atomic_spu
    from silly_kicks.atomic.spadl.base import convert_to_atomic
    from tests.test_benchmark import _make_spadl_actions

    def measure(n):
        atomic_actions = convert_to_atomic(_make_spadl_actions(n))
        with rows_scanned_counter() as c:
            atomic_spu.add_possessions(atomic_actions)
        return c["n"]

    assert_subquadratic_growth(measure, sizes=(1500, 3000, 6000), label="atomic_add_possessions")


# ============================ restdefense count_goalside (TF-60) ============================
def test_count_goalside_by_sample_is_subquadratic():
    from silly_kicks.restdefense._counting import count_goalside_by_sample

    def measure(n):  # n = number of frames == number of samples (the loop dimension)
        from tests.restdefense._fixtures import make_scaling_fixture

        frames, samples = make_scaling_fixture(n)
        with rows_scanned_counter() as c:
            count_goalside_by_sample(samples, frames, team_col="team_id", ball_x_col="ball_x", goal_x_col="own_goal_x")
        return c["n"]

    assert_subquadratic_growth(measure, sizes=(256, 1024, 4096), label="count_goalside_by_sample")


def test_score_samples_is_subquadratic():
    from silly_kicks.restdefense._compute import _score_samples
    from silly_kicks.restdefense._config import RestDefenseParams

    def measure(n):  # n = number of samples == number of frames (the loop dimension)
        from tests.restdefense._fixtures import make_score_scaling_fixture

        keep, frames, opp_map = make_score_scaling_fixture(n)
        with rows_scanned_counter() as c:
            _score_samples(keep, frames, opp_map, RestDefenseParams())
        return c["n"]

    assert_subquadratic_growth(measure, sizes=(256, 1024, 4096), label="_score_samples")


# ============================ #3 compute_defensive_credits ============================
def _scaled_defensive_credit_input(n) -> tuple[pd.DataFrame, pd.DataFrame]:
    from tests.tracking._defensive_credit_fixtures import frame_with_defender, one_action

    acts, frames_list = [], []
    for i in range(n):
        t = 50.0 + i * 2.0
        a = one_action(action_id=i, type_name="shot", result_name="fail", start_x=95.0, start_y=34.0, time_seconds=t)
        a["shot_blocked"] = pd.array([False], dtype="boolean")
        a["cross_blocked"] = pd.array([pd.NA], dtype="boolean")
        a["shot_on_target_derived"] = pd.array([False], dtype="boolean")
        a["xg"] = [0.2]
        acts.append(a)
        frames_list.append(frame_with_defender(action_time=t, frame_id=500 + i, defender_x=96.0, defender_y=34.0))
    return pd.concat(acts, ignore_index=True), pd.concat(frames_list, ignore_index=True)


def test_compute_defensive_credits_is_subquadratic(fitted_xt):
    from silly_kicks.tracking.defensive_credit import compute_defensive_credits
    from silly_kicks.tracking.utils import link_actions_to_frames

    def measure(n):
        actions, frames = _scaled_defensive_credit_input(n)
        # Pre-link OUTSIDE the counter (via the public `links=` kwarg) to ISOLATE the group_rows
        # site -- the linking is a pre-existing cost ADR-068 never touched (spec decision 4:
        # count only the suspect op). What remains is the group_rows(frames, "frame_id") lookup.
        links = link_actions_to_frames(actions, frames)[0]
        with rows_scanned_counter() as c:
            compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt, links=links)
        return c["n"]

    assert_subquadratic_growth(measure, sizes=(64, 128, 256), label="compute_defensive_credits")


# ==================== gkdv paired-vector controls (ADR-073, TF-19 A+2) ====================
def _paired_vector_controls_input(n_frames: int):
    """``n_frames`` scales the LOOP dimension (one target per frame), so the group COUNT scales while
    each group's SIZE stays fixed. That is the ADR-073 discrimination proof: ``group_rows`` builds the
    grouping ONCE and does one O(1) ``.get`` per target -> O(T); a full-table filter over ``outfield``
    per target (the regression this guards) rescans T*players rows T times -> O(T^2). Scaling frames
    (not players-per-frame) is what keeps a regressed version quadratic and the fix linear."""
    frame_rows = []
    target_rows = []
    for f in range(n_frames):
        for i in range(3):  # team-1 (defending) outfielders
            frame_rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": f,
                    "team_id": 1,
                    "player_id": 100 + i,
                    "is_ball": False,
                    "is_goalkeeper": False,
                    "x": 20.0 + i,
                    "y": 30.0 + i,
                }
            )
        frame_rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": f,
                "team_id": 1,
                "player_id": 199,
                "is_ball": False,
                "is_goalkeeper": True,
                "x": 4.0,
                "y": 34.0,
            }
        )
        frame_rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": f,
                "team_id": 2,
                "player_id": 200,
                "is_ball": False,
                "is_goalkeeper": False,
                "x": 50.0,
                "y": 34.0,
            }
        )
        frame_rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": f,
                "team_id": None,
                "player_id": None,
                "is_ball": True,
                "is_goalkeeper": False,
                "x": 40.0,
                "y": 34.0,
            }
        )
        target_rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": f,
                "defending_team_id": 1,
                "actual_x": 4.0,
                "actual_y": 34.0,
                "imp_x": 0.0,
                "imp_y": 34.0,
            }
        )
    return pd.DataFrame(frame_rows), pd.DataFrame(target_rows)


def test_paired_vector_controls_is_subquadratic():
    from silly_kicks.gkdv._probe import paired_vector_controls

    def measure(n):
        frames, targets = _paired_vector_controls_input(n)
        with rows_scanned_counter() as c:
            paired_vector_controls(frames, targets, r=1, rng=np.random.default_rng(0))
        return c["n"]

    assert_subquadratic_growth(measure, sizes=(128, 256, 512), label="paired_vector_controls")
