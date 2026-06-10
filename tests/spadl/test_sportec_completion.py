"""sportec.py pass/set-piece completion from native DFL play_evaluation (BUG 2 fix, 2026-06-09).

Previously every Play/set-piece was hard-wired result=success, zeroing failed-pass / failed-
goalkick labels (IDSSE goalkicks showed 100% success vs the real ~71%). DFL Play AND set-piece
events (GoalKick/FreeKick/CornerKick/ThrowIn, via the nested Play) carry an Evaluation
(`successfullyCompleted` / `successful` -> success; `unsuccessful` -> fail; NULL -> success,
conservative). Confirmed on 7 real DFL matches (lakehouse, 10,497 events).
"""

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import sportec as sportec_mod

_FAIL = spadlconfig.result_id["fail"]
_SUCCESS = spadlconfig.result_id["success"]
_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_PASS = spadlconfig.actiontype_id["pass"]


def _ev(rows: list[dict]) -> pd.DataFrame:
    base = dict(match_id="M1", event_type="Play", period=1, player_id="P1", team="DFL-CLU-A", x=50.0, y=34.0)
    out = []
    for i, r in enumerate(rows):
        d = dict(base)
        d.update(r)
        d.setdefault("event_id", f"e{i}")
        d.setdefault("timestamp_seconds", 10.0 + i)
        out.append(d)
    return pd.DataFrame(out)


def _convert(events):
    actions, _ = sportec_mod.convert_to_actions(events, home_team_id="DFL-CLU-A", home_team_start_left=True)
    return actions


@pytest.mark.parametrize(
    ("evaluation", "expected"),
    [
        ("unsuccessful", _FAIL),
        ("successfullyCompleted", _SUCCESS),
        ("successful", _SUCCESS),  # rare second success synonym (lakehouse: 23 occurrences)
        ("", _SUCCESS),  # NULL/missing -> conservative success
    ],
)
def test_pass_result_from_play_evaluation(evaluation, expected):
    actions = _convert(_ev([dict(event_type="Play", play_evaluation=evaluation)]))
    p = actions[actions["type_id"] == _PASS].iloc[0]
    assert p["result_id"] == expected


def test_goalkick_event_unsuccessful_is_fail():
    # formal DFL GoalKick event carries play_evaluation via its nested Play
    actions = _convert(_ev([dict(event_type="GoalKick", play_evaluation="unsuccessful")]))
    gk = actions[actions["type_id"] == _GOALKICK].iloc[0]
    assert gk["result_id"] == _FAIL


def test_goalkick_event_completed_is_success():
    actions = _convert(_ev([dict(event_type="GoalKick", play_evaluation="successfullyCompleted")]))
    gk = actions[actions["type_id"] == _GOALKICK].iloc[0]
    assert gk["result_id"] == _SUCCESS


def test_punt_synth_goalkick_inherits_parent_evaluation():
    # Play + punt qualifier -> keeper_pick_up + synthesized goalkick; the synth goalkick must
    # inherit the parent Play's play_evaluation (not a hard-wired success).
    actions = _convert(_ev([dict(event_type="Play", play_goal_keeper_action="punt", play_evaluation="unsuccessful")]))
    gk = actions[actions["type_id"] == _GOALKICK].iloc[0]
    assert gk["result_id"] == _FAIL


def test_unseen_reason_coded_failure_is_fail():
    # Main path: any non-empty, non-success token (e.g. a reason-coded failure the 4.20.1 exact-match
    # left as success) -> fail. The headline new behavior.
    actions = _convert(_ev([dict(event_type="Play", play_evaluation="unsuccessfulBecauseOfFoul")]))
    assert actions[actions["type_id"] == _PASS].iloc[0]["result_id"] == _FAIL


def test_unexpected_token_warns_and_fails():
    # The warn (observability): an unexpected non-success token is surfaced, not silently classified.
    with pytest.warns(UserWarning, match="unexpected play_evaluation"):
        actions = _convert(_ev([dict(event_type="Play", play_evaluation="weirdNovelToken")]))
    assert actions[actions["type_id"] == _PASS].iloc[0]["result_id"] == _FAIL


def test_synth_goalkick_unseen_token_is_fail():
    # Synth site: the punt-synthesized goalkick inherits the parent Play's eval via the SAME allowlist.
    actions = _convert(
        _ev([dict(event_type="Play", play_goal_keeper_action="punt", play_evaluation="unsuccessfulBecauseOfFoul")])
    )
    assert actions[actions["type_id"] == _GOALKICK].iloc[0]["result_id"] == _FAIL


def test_play_evaluation_column_absent_no_mass_fail():
    # No play_evaluation key at all (column absent) must NOT mass-fail passes (the allowlist trap).
    actions = _convert(_ev([dict(event_type="Play")]))
    assert actions[actions["type_id"] == _PASS].iloc[0]["result_id"] == _SUCCESS


def test_known_tokens_are_warn_silent():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning -> test failure
        for tok in ("successfullyCompleted", "successful", "unsuccessful", ""):
            _convert(_ev([dict(event_type="Play", play_evaluation=tok)]))


@pytest.mark.parametrize(
    "evaluation", ["successfullyCompleted", "successful", "unsuccessful", "unsuccessfulBecauseOfFoul", ""]
)
def test_main_and_synth_paths_agree(evaluation):
    # Single-source guard: the main pass path and the synth-goalkick path must map every token to the
    # same result_id (both route through _play_evaluation_is_fail). A drift would break one side.
    main = _convert(_ev([dict(event_type="Play", play_evaluation=evaluation)]))
    main_res = main[main["type_id"] == _PASS].iloc[0]["result_id"]
    synth = _convert(_ev([dict(event_type="Play", play_goal_keeper_action="punt", play_evaluation=evaluation)]))
    synth_res = synth[synth["type_id"] == _GOALKICK].iloc[0]["result_id"]
    assert main_res == synth_res


def test_observed_distribution_regression_and_single_batch_warn():
    # Lock "robustness hardening, not re-mapping": the full observed DFL vocabulary + one reason-coded
    # token, in one converter pass. Clean tokens map byte-identically to the 4.20.1 exact-match
    # converter; the reason-code -> fail; exactly the one unexpected token is named in a single warn.
    rows = [
        dict(event_type="Play", play_evaluation="successfullyCompleted"),
        dict(event_type="Play", play_evaluation="successful"),
        dict(event_type="Play", play_evaluation="unsuccessful"),
        dict(event_type="Play", play_evaluation=""),
        dict(event_type="Play", play_evaluation="unsuccessfulBecauseOfFoul"),
    ]
    with pytest.warns(UserWarning, match=r"unsuccessfulBecauseOfFoul"):
        actions = _convert(_ev(rows))
    passes = actions[actions["type_id"] == _PASS].reset_index(drop=True)
    assert list(passes["result_id"]) == [_SUCCESS, _SUCCESS, _FAIL, _SUCCESS, _FAIL]
