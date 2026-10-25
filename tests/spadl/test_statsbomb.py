"""StatsBomb SPADL converter tests."""

import inspect

import pandas as pd
import pytest

from silly_kicks.spadl import statsbomb
from silly_kicks.spadl.schema import SPADL_COLUMNS, ConversionReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_statsbomb_events() -> pd.DataFrame:
    """Minimal StatsBomb event DataFrame for testing."""
    return pd.DataFrame(
        [
            {
                "game_id": 1,
                "event_id": "abc-123",
                "period_id": 1,
                "timestamp": "00:00:01.000",
                "team_id": 100,
                "player_id": 200,
                "type_name": "Pass",
                "location": [60.0, 40.0],
                "extra": {
                    "pass": {
                        "end_location": [70.0, 40.0],
                        "outcome": {"name": "Complete"},
                        "height": {"name": "Ground Pass"},
                    }
                },
            },
            {
                "game_id": 1,
                "event_id": "abc-456",
                "period_id": 1,
                "timestamp": "00:00:05.000",
                "team_id": 100,
                "player_id": 201,
                "type_name": "Pressure",
                "location": [50.0, 30.0],
                "extra": {},
            },
        ]
    )


# ---------------------------------------------------------------------------
# Tests that use inline data (no external fixtures required)
# ---------------------------------------------------------------------------


def test_statsbomb_no_inplace_fillna() -> None:
    """Bug #946: fillna must not use inplace=True (pandas 3.0 compat)."""
    source = inspect.getsource(statsbomb.convert_to_actions)
    assert "inplace=True" not in source, "inplace=True is deprecated in pandas 2.x"


def test_statsbomb_returns_tuple():
    events = _make_statsbomb_events()
    result = statsbomb.convert_to_actions(events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1)
    assert isinstance(result, tuple)
    assert len(result) == 2
    actions, report = result
    assert isinstance(actions, pd.DataFrame)
    assert isinstance(report, ConversionReport)


def test_statsbomb_conversion_report():
    events = _make_statsbomb_events()
    _actions, report = statsbomb.convert_to_actions(
        events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1
    )
    assert report.provider == "StatsBomb"
    assert report.total_events == 2
    assert "Pass" in report.mapped_counts
    assert "Pressure" in report.excluded_counts
    assert not report.has_unrecognized


def test_statsbomb_unrecognized_event_warning():
    events = _make_statsbomb_events()
    events = pd.concat(
        [
            events,
            pd.DataFrame(
                [
                    {
                        "game_id": 1,
                        "event_id": "abc-789",
                        "period_id": 1,
                        "timestamp": "00:00:10.000",
                        "team_id": 100,
                        "player_id": 202,
                        "type_name": "FutureEventType",
                        "location": [50.0, 30.0],
                        "extra": {},
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    import warnings as w_mod

    with w_mod.catch_warnings(record=True) as w:
        w_mod.simplefilter("always")
        _actions, report = statsbomb.convert_to_actions(
            events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1
        )
    assert report.has_unrecognized
    assert "FutureEventType" in report.unrecognized_counts
    unrecognized_warnings = [x for x in w if "unrecognized" in str(x.message).lower()]
    assert len(unrecognized_warnings) >= 1


def test_statsbomb_output_columns():
    events = _make_statsbomb_events()
    actions, _ = statsbomb.convert_to_actions(events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1)
    assert list(actions.columns) == list(SPADL_COLUMNS.keys())
    for col, dtype in SPADL_COLUMNS.items():
        assert str(actions[col].dtype) == dtype, f"{col}: expected {dtype}, got {actions[col].dtype}"


def test_statsbomb_input_validation():
    df = pd.DataFrame({"game_id": [1]})
    with pytest.raises(ValueError, match="StatsBomb"):
        statsbomb.convert_to_actions(df, home_team_id=100)


def test_statsbomb_goal_keeper_key_fallback():
    """extra["goal_keeper"] (snake_cased) should work the same as extra["goalkeeper"]."""
    events = pd.DataFrame(
        [
            {
                "game_id": 1,
                "event_id": "gk-1",
                "period_id": 1,
                "timestamp": "00:00:30.000",
                "team_id": 100,
                "player_id": 300,
                "type_name": "Goal Keeper",
                "location": [10.0, 34.0],
                "extra": {
                    "goal_keeper": {
                        "type": {"name": "Shot Saved"},
                        "outcome": {"name": "Success"},
                        "body_part": {"name": "Right Hand"},
                    }
                },
            },
        ]
    )
    actions, _report = statsbomb.convert_to_actions(events, home_team_id=100, xy_fidelity_version=1)
    # The keeper action should be produced (not dropped as non_action)
    assert len(actions) >= 1
    from silly_kicks.spadl.config import actiontype_id

    keeper_types = {actiontype_id["keeper_save"], actiontype_id["keeper_claim"], actiontype_id["keeper_punch"]}
    assert any(actions["type_id"].isin(keeper_types)), (
        f"Expected a keeper action, got type_ids: {actions['type_id'].tolist()}"
    )


class TestStatsbombPreserveNative:
    """Tests for the ``preserve_native`` kwarg added in 1.1.0.

    Surfaces provider-native fields (e.g. StatsBomb's top-level ``possession``
    sequence number, ``possession_team``, ``play_pattern``) through conversion
    as extra columns alongside the canonical SPADL output.
    """

    @staticmethod
    def _events_with_possession() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "event_id": "ev-1",
                    "period_id": 1,
                    "timestamp": "00:00:01.000",
                    "team_id": 100,
                    "player_id": 200,
                    "type_name": "Pass",
                    "location": [60.0, 40.0],
                    "extra": {
                        "pass": {
                            "end_location": [70.0, 40.0],
                            "outcome": {"name": "Complete"},
                            "height": {"name": "Ground Pass"},
                        }
                    },
                    "possession": 1,
                    "possession_team": {"id": 100, "name": "Home"},
                },
                {
                    "game_id": 1,
                    "event_id": "ev-2",
                    "period_id": 1,
                    "timestamp": "00:00:05.000",
                    "team_id": 100,
                    "player_id": 201,
                    "type_name": "Pass",
                    "location": [65.0, 42.0],
                    "extra": {
                        "pass": {
                            "end_location": [75.0, 42.0],
                            "outcome": {"name": "Complete"},
                            "height": {"name": "Ground Pass"},
                        }
                    },
                    "possession": 1,
                    "possession_team": {"id": 100, "name": "Home"},
                },
                {
                    "game_id": 1,
                    "event_id": "ev-3",
                    "period_id": 1,
                    "timestamp": "00:00:10.000",
                    "team_id": 200,
                    "player_id": 300,
                    "type_name": "Pass",
                    "location": [60.0, 40.0],
                    "extra": {
                        "pass": {
                            "end_location": [50.0, 40.0],
                            "outcome": {"name": "Complete"},
                            "height": {"name": "Ground Pass"},
                        }
                    },
                    "possession": 2,
                    "possession_team": {"id": 200, "name": "Away"},
                },
            ]
        )

    def test_default_none_unchanged(self):
        events = self._events_with_possession()
        actions, _ = statsbomb.convert_to_actions(
            events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1
        )
        assert "possession" not in actions.columns
        assert list(actions.columns) == list(SPADL_COLUMNS.keys())

    def test_empty_list_unchanged(self):
        events = self._events_with_possession()
        actions, _ = statsbomb.convert_to_actions(
            events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1, preserve_native=[]
        )
        assert list(actions.columns) == list(SPADL_COLUMNS.keys())

    def test_single_field_preserved(self):
        events = self._events_with_possession()
        actions, _ = statsbomb.convert_to_actions(
            events,
            home_team_id=100,
            xy_fidelity_version=1,
            shot_fidelity_version=1,
            preserve_native=["possession"],
        )
        assert "possession" in actions.columns
        # Map original_event_id → preserved possession value (skipping synthetic dribbles which have NaN).
        non_synthetic = actions[actions["original_event_id"].notna() & (actions["original_event_id"] != "")]
        possession_for = dict(zip(non_synthetic["original_event_id"], non_synthetic["possession"], strict=True))
        assert possession_for.get("ev-1") == 1
        assert possession_for.get("ev-2") == 1
        assert possession_for.get("ev-3") == 2

    def test_multiple_fields_preserved(self):
        events = self._events_with_possession()
        actions, _ = statsbomb.convert_to_actions(
            events,
            home_team_id=100,
            xy_fidelity_version=1,
            shot_fidelity_version=1,
            preserve_native=["possession", "possession_team"],
        )
        assert "possession" in actions.columns
        assert "possession_team" in actions.columns

    def test_missing_field_raises(self):
        events = self._events_with_possession()
        with pytest.raises(ValueError, match="preserve_native"):
            statsbomb.convert_to_actions(
                events,
                home_team_id=100,
                xy_fidelity_version=1,
                shot_fidelity_version=1,
                preserve_native=["does_not_exist"],
            )

    def test_overlap_with_schema_raises(self):
        events = self._events_with_possession()
        with pytest.raises(ValueError, match=r"overlap|already"):
            statsbomb.convert_to_actions(
                events,
                home_team_id=100,
                xy_fidelity_version=1,
                shot_fidelity_version=1,
                preserve_native=["team_id"],
            )

    def test_synthetic_dribbles_get_nan(self):
        # Two same-team passes with sufficient gap to trigger _add_dribbles.
        events = pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "event_id": "ev-A",
                    "period_id": 1,
                    "timestamp": "00:00:01.000",
                    "team_id": 100,
                    "player_id": 200,
                    "type_name": "Pass",
                    "location": [10.0, 40.0],
                    "extra": {
                        "pass": {
                            "end_location": [30.0, 40.0],
                            "outcome": {"name": "Complete"},
                            "height": {"name": "Ground Pass"},
                        }
                    },
                    "possession": 1,
                },
                {
                    "game_id": 1,
                    "event_id": "ev-B",
                    "period_id": 1,
                    "timestamp": "00:00:03.000",
                    "team_id": 100,
                    "player_id": 200,
                    "type_name": "Pass",
                    "location": [50.0, 40.0],
                    "extra": {
                        "pass": {
                            "end_location": [70.0, 40.0],
                            "outcome": {"name": "Complete"},
                            "height": {"name": "Ground Pass"},
                        }
                    },
                    "possession": 1,
                },
            ]
        )
        actions, _ = statsbomb.convert_to_actions(
            events,
            home_team_id=100,
            xy_fidelity_version=1,
            shot_fidelity_version=1,
            preserve_native=["possession"],
        )
        # Synthetic dribble has NULL original_event_id (per CHANGELOG).
        synthetic = actions[actions["original_event_id"].isna()]
        if len(synthetic) > 0:
            assert synthetic["possession"].isna().all()


# ---------------------------------------------------------------------------
# goalkeeper_ids — accepted as no-op for cross-provider API symmetry (1.10.0)
# StatsBomb's source events natively mark GK actions, so the parameter has
# no effect on output. Asserting byte-for-byte equivalence catches drift.
# ---------------------------------------------------------------------------


class TestStatsBombGoalkeeperIdsNoOp:
    @staticmethod
    def _events() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": [1, 1],
                "event_id": ["e1", "e2"],
                "period_id": [1, 1],
                "timestamp": ["00:00:01.000", "00:00:02.000"],
                "team_id": [100, 100],
                "player_id": [200, 201],
                "type_name": ["Pass", "Pass"],
                "location": [[60.0, 40.0], [70.0, 40.0]],
                "extra": [{}, {}],
            }
        )

    def test_goalkeeper_ids_parameter_is_accepted(self):
        actions, _ = statsbomb.convert_to_actions(self._events(), home_team_id=100, goalkeeper_ids={200, 300})
        assert isinstance(actions, pd.DataFrame)

    def test_goalkeeper_ids_output_identical_with_and_without(self):
        a_none, _ = statsbomb.convert_to_actions(self._events(), home_team_id=100)
        a_set, _ = statsbomb.convert_to_actions(self._events(), home_team_id=100, goalkeeper_ids={200})
        pd.testing.assert_frame_equal(a_none, a_set, check_dtype=True)

    def test_empty_goalkeeper_ids_set_equivalent_to_none(self):
        a_empty, _ = statsbomb.convert_to_actions(self._events(), home_team_id=100, goalkeeper_ids=set())
        a_none, _ = statsbomb.convert_to_actions(self._events(), home_team_id=100, goalkeeper_ids=None)
        pd.testing.assert_frame_equal(a_empty, a_none, check_dtype=True)


def test_shot_blocked_true_on_real_blocked_shot():
    from silly_kicks.spadl.utils import add_names
    from tests.invariants._loaders import load_statsbomb

    actions, _ = load_statsbomb(7298)  # 12 real blocked shots
    # SPADL output carries type_id, not type_name; add the string names for the filter.
    actions = add_names(actions)
    shots = actions[actions["type_name"] == "shot"]
    assert (shots["shot_blocked"] == True).sum() == 12  # noqa: E712
    # every shot row is True/False (never NA on a provider that encodes it)
    assert shots["shot_blocked"].notna().all()
    _crosses = actions[actions["type_name"] == "cross"]
    assert _crosses["cross_blocked"].notna().all()  # open-play crosses carry True/False
    assert (_crosses["cross_blocked"] == False).all()  # 7298 has no blocked crosses  # noqa: E712
    assert actions[actions["type_name"] != "cross"]["cross_blocked"].isna().all()


def test_open_play_cross_mask_equals_inline_predicate():
    """The extracted helper must equal an independent inline spelling of the open-play-cross rule
    on real events. Deliberately NOT compared against `_vectorized_type_id` (post-refactor it calls
    the helper, so that comparison is tautological); the inline copy here is the invariance anchor.
    """
    import json
    import pathlib

    from scripts._sb_raw import flatten_events
    from silly_kicks.spadl.statsbomb import _flatten_extra, _open_play_cross_mask

    raw = json.loads(pathlib.Path("tests/datasets/statsbomb/raw/events/7584.json").read_text(encoding="utf-8"))
    events = flatten_events(raw, 7584)
    events["extra"] = events["extra"].fillna({})  # mirror convert_to_actions
    events = _flatten_extra(events)

    inline = (
        (events["type_name"] == "Pass")
        & (events["_pass_cross"] == True)  # noqa: E712
        & ~events["_pass_type"].isin(["Free Kick", "Corner", "Goal Kick", "Throw-in"])
    ).to_numpy()
    mask = _open_play_cross_mask(events)
    assert mask.tolist() == inline.tolist()
    assert mask.sum() > 0  # non-vacuous: the fixture actually exercises the True branch


def test_cross_blocked_true_on_real_blocked_cross():
    from silly_kicks.spadl.utils import add_names
    from tests.invariants._loaders import load_statsbomb

    actions, _ = load_statsbomb(7584)
    actions = add_names(actions)
    crosses = actions[actions["type_name"] == "cross"]
    # exactly one genuine blocked cross in 7584 (min 89)
    assert (crosses["cross_blocked"] == True).sum() == 1  # noqa: E712
    blocked = crosses[crosses["cross_blocked"] == True]  # noqa: E712
    assert blocked["original_event_id"].iloc[0] == "e8edd276-8490-456c-b221-240d128f61f1"
    # every open-play cross carries a real True/False, never NA
    assert crosses["cross_blocked"].notna().all()
    # non-open-play rows (regular passes, set-piece crosses, non-passes) stay NA
    assert actions[actions["type_name"] != "cross"]["cross_blocked"].isna().all()


def test_cross_blocked_false_when_no_block_related():
    from silly_kicks.spadl.utils import add_names
    from tests.invariants._loaders import load_statsbomb

    for mid in (7298, 3754058):  # zero blocked crosses in these matches
        actions, _ = load_statsbomb(mid)
        actions = add_names(actions)
        crosses = actions[actions["type_name"] == "cross"]
        assert crosses["cross_blocked"].notna().all()
        assert (crosses["cross_blocked"] == False).all()  # noqa: E712
        assert actions[actions["type_name"] != "cross"]["cross_blocked"].isna().all()


# ---------------------------------------------------------------------------
# K2 — set-piece passes with Unknown outcome + the cross-provider chain convention
# (spec section 8.1/8.2, decisions K2/B4). Synthetic StatsBomb-shaped fixture.
# ---------------------------------------------------------------------------
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from silly_kicks.spadl import config as _spadlcfg  # noqa: E402

_K2_FIXTURE = Path(__file__).parents[1] / "datasets" / "statsbomb" / "k2_synthetic.json"


def _k2_records() -> list[dict]:
    """Synthetic StatsBomb raw events exercising every K2 branch (section 8.1/8.2).

    Chronological, home team 100. Each set-piece-Unknown restart is resolved from its next MEANINGFUL event
    (bookkeeping / Foul Committed-Won are skipped, TF46C-IMPL-21):
      c1 corner (Unknown)   -> next = same-team FREE KICK           -> chain: success + chain_ambiguous;
      c2 free kick (Unknown)-> next = same-team PENALTY             -> chain: success + chain_ambiguous;
      c3 goal kick (Unknown)-> Foul Committed (skipped) -> FREE KICK-> chain THROUGH A FOUL (success + chain_ambiguous);
      c4 throw-in (Unknown) -> next = same-team open pass           -> success (retained), no chain;
      c5 corner (Unknown)   -> next = other-team pass               -> fail (lost), no chain;
      e  non-set-piece pass (Unknown)                               -> non_action (dropped).
    """

    def _pass(evid, t, player, pass_type, outcome, nxt=(80.0, 40.0)):
        extra_pass: dict = {"end_location": list(nxt), "outcome": {"name": outcome}, "height": {"name": "Ground Pass"}}
        if pass_type is not None:
            extra_pass["type"] = {"name": pass_type}
        return {
            "game_id": 1,
            "event_id": evid,
            "period_id": 1,
            "timestamp": t,
            "team_id": player[0],
            "player_id": player[1],
            "type_name": "Pass",
            "location": [60.0, 40.0],
            "extra": {"pass": extra_pass},
        }

    def _pen_shot(evid, t):
        return {
            "game_id": 1,
            "event_id": evid,
            "period_id": 1,
            "timestamp": t,
            "team_id": 100,
            "player_id": 210,
            "type_name": "Shot",
            "location": [108.0, 40.0],
            "extra": {
                "shot": {"type": {"name": "Penalty"}, "outcome": {"name": "Saved"}, "end_location": [120.0, 40.0]}
            },
        }

    def _foul(evid, t, team):
        # Foul Committed is a successor-SKIP type: it sits between a restart and the free kick it wins.
        return {
            "game_id": 1,
            "event_id": evid,
            "period_id": 1,
            "timestamp": t,
            "team_id": team,
            "player_id": 900,
            "type_name": "Foul Committed",
            "location": [50.0, 40.0],
            "extra": {},
        }

    return [
        _pass("c1_corner", "00:00:01.000", (100, 200), "Corner", "Unknown"),  # -> c1_fk (chain, corner->FK)
        _pass("c1_fk", "00:00:02.000", (100, 201), "Free Kick", "Complete"),
        _pass("c2_fk", "00:00:03.000", (100, 202), "Free Kick", "Unknown"),  # -> c2_pen (chain, FK->penalty)
        _pen_shot("c2_pen", "00:00:04.000"),
        _pass("c3_gk", "00:00:05.000", (100, 203), "Goal Kick", "Unknown"),  # -> (foul skipped) -> c3_fk chain
        _foul("c3_foul", "00:00:06.000", 200),
        _pass("c3_fk", "00:00:07.000", (100, 204), "Free Kick", "Complete"),
        _pass("c4_ti", "00:00:08.000", (100, 205), "Throw-in", "Unknown"),  # -> c4_pass (same team, not FK) success
        _pass("c4_pass", "00:00:09.000", (100, 206), None, "Complete"),
        _pass("c5_corner", "00:00:10.000", (100, 207), "Corner", "Unknown"),  # -> c5_pass (other team) fail
        _pass("c5_pass", "00:00:11.000", (200, 300), None, "Complete"),
        _pass("e_nonsp", "00:00:12.000", (100, 208), None, "Unknown"),  # non-set-piece Unknown -> non_action
    ]


def _k2_synthetic_events() -> pd.DataFrame:
    return pd.DataFrame(_k2_records())


def _convert_k2():
    return statsbomb.convert_to_actions(
        _k2_synthetic_events(), home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1
    )


def _k2_by_event(actions: pd.DataFrame) -> pd.DataFrame:
    out = actions.copy()
    out["type_name"] = out["type_id"].map({i: n for i, n in enumerate(_spadlcfg.actiontypes)})
    return out.set_index("original_event_id")


def test_k2_generator_reproduces_committed_fixture():
    # The committed fixture equals the in-code generator byte-for-byte (regen reproducibility).
    committed = json.loads(_K2_FIXTURE.read_text(encoding="utf-8"))
    assert committed == _k2_records()


_K2_KEPT = ("c1_corner", "c2_fk", "c3_gk", "c4_ti", "c5_corner")


def test_k2_setpiece_unknown_keeps_type_and_infers_result():
    actions, _ = _convert_k2()
    by = _k2_by_event(actions)
    # Each set-piece Unknown pass survived (not non_action) and kept its set-piece type.
    assert by.loc["c1_corner", "type_name"] in ("corner_crossed", "corner_short")
    assert by.loc["c2_fk", "type_name"] in ("freekick_crossed", "freekick_short")
    assert by.loc["c3_gk", "type_name"] == "goalkick"
    assert by.loc["c4_ti", "type_name"] == "throw_in"
    assert by.loc["c5_corner", "type_name"] in ("corner_crossed", "corner_short")
    # Every kept set-piece Unknown pass is result_source="inferred".
    for evid in _K2_KEPT:
        assert by.loc[evid, "result_source"] == "inferred", evid
    # Result inferred from the next MEANINGFUL event: c4 retained same team (success), c5 lost (fail).
    assert by.loc["c4_ti", "result_id"] == _spadlcfg.result_id["success"]
    assert by.loc["c5_corner", "result_id"] == _spadlcfg.result_id["fail"]


def test_k2_non_setpiece_unknown_stays_non_action():
    actions, _ = _convert_k2()
    # The non-set-piece Unknown pass is dropped (non_action), so its event_id is absent from the output.
    assert "e_nonsp" not in set(actions["original_event_id"].astype(str))


def test_k2_chain_case_success_and_chain_ambiguous():
    # Chain (spec section 8.2): a set-piece Unknown restart whose next MEANINGFUL event is a FREE KICK or
    # PENALTY by the same team -> result_id="success", restart_evidence="chain_ambiguous". Covers corner->FK
    # (c1), FK->penalty (c2) and goal kick -> (Foul Committed skipped) -> FK through a foul (c3, TF46C-IMPL-21).
    actions, _ = _convert_k2()
    by = _k2_by_event(actions)
    for evid in ("c1_corner", "c2_fk", "c3_gk"):
        assert by.loc[evid, "result_id"] == _spadlcfg.result_id["success"], evid
        assert by.loc[evid, "restart_evidence"] == "chain_ambiguous", evid
        assert by.loc[evid, "result_source"] == "inferred", evid
    # Non-chain set-piece restarts (retained c4, lost c5) leave restart_evidence <NA>.
    assert pd.isna(by.loc["c4_ti", "restart_evidence"])
    assert pd.isna(by.loc["c5_corner", "restart_evidence"])


def test_k2_rows_excluded_by_training_helper():
    # The shared helper (section 11.3) drops every K2 inferred / chain_ambiguous row.
    from silly_kicks.spadl._training_exclusion import training_exclusion_mask

    actions, _ = _convert_k2()
    mask = training_exclusion_mask(actions)
    mask.index = actions["original_event_id"].astype(str)
    for evid in _K2_KEPT:
        assert bool(mask.loc[evid]) is True, evid
