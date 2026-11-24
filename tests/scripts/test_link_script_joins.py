"""R8: every in-repo ``scripts/`` pointer join reads each action's own game (spec 2026-11-08 tracking-batch §4.9 W9.4).

Plan Task 5 step 4: one test per ``scripts/`` census row, on ``make_two_game_inputs`` (two games that share action
ids, period ids, and frame ids): each action joins only its own game's frames. Each test is red on the
``action_id`` / ``frame_id``-only join it replaces.
"""

from __future__ import annotations

import pandas as pd

from tests._prod_fixtures import make_two_game_inputs
from tests.invariants._link_population import split_by_game


def _two_games():
    """Two games that share action, period, and frame ids. Game B's players stand 5 m further up the pitch than
    game A's, so a join that reads the other game's frame gives a different number, not the same copy."""
    actions, frames = make_two_game_inputs()
    assert actions["action_id"].duplicated().any()  # non-vacuity: the games share action ids
    second = (frames["game_id"] == frames["game_id"].iloc[-1]).to_numpy()
    y = frames["y"].to_numpy(dtype="float64", copy=True)
    y[second] = y[second] + 5.0
    return actions, frames.assign(y=y)


def _per_game(fn):
    """``fn(actions, frames)`` on each game of the two-game input, in game order."""
    actions, frames = _two_games()
    return [fn(a, f) for a, f in split_by_game(actions, frames)]


def _disjoint_links():
    """Two games whose actions link to DIFFERENT frame ids (the early actions of one game, the late actions of the
    other), so a (period_id, frame_id)-only restriction picks frames of the wrong game."""
    actions, frames = _two_games()
    (a, _), (b, _) = split_by_game(actions, frames)
    return pd.concat([a.iloc[:3], b.iloc[-3:]], ignore_index=True), frames


def test_scored_frames_restrict_to_each_actions_own_game():
    """scripts/validate_das_native_parity.py ``_scored_frames``."""
    from scripts.validate_das_native_parity import _scored_frames

    actions, frames = _disjoint_links()
    whole = _scored_frames(actions, frames)
    parts = [_scored_frames(a, f) for a, f in split_by_game(actions, frames)]
    assert not whole.empty  # non-vacuity
    cols = ["game_id", "period_id", "frame_id", "player_id"]
    got = whole[cols].sort_values(cols, kind="stable").reset_index(drop=True)
    want = pd.concat(parts, ignore_index=True)[cols].sort_values(cols, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)


# ---- the per-action loops: each script's result on two games equals its results on each game, in game order ----


def _split_equal(whole, parts) -> None:
    """``whole`` (two games) equals the per-game ``parts`` in game order: a DataFrame or a list per call."""
    from scripts._bitwise import assert_bits_equal

    if isinstance(whole, pd.DataFrame):
        joined = pd.concat([p for p in parts if p is not None], ignore_index=True)
        assert list(whole.columns) == list(joined.columns)
        assert len(whole) == len(joined) > 0  # non-vacuity: the loop produced rows
        lo = 0
        for p in parts:  # slice by slice (a concat would turn an object column's pd.NA into nan)
            assert_bits_equal(whole.iloc[lo : lo + len(p)].reset_index(drop=True), p.reset_index(drop=True))
            lo += len(p)
    else:
        joined = [r for p in parts for r in p]
        assert len(whole) == len(joined) > 0
        assert_bits_equal(pd.DataFrame(whole), pd.DataFrame(joined))


def _home():
    from tests._prod_fixtures import HOME_TEAM_ID

    return HOME_TEAM_ID


def test_structural_pass_sigma_records_read_each_passes_own_game():
    """scripts/tune_structural_pass_sigma.py ``_match_records``."""
    from scripts.tune_structural_pass_sigma import _match_records

    def records(actions, frames):
        return _match_records(("p", "m", actions, frames, _home()))

    actions, frames = _two_games()
    _split_equal(records(actions, frames), _per_game(records))


def test_cover_shadow_agreement_reads_each_actions_own_game():
    """scripts/measure_cover_shadow_argmax_agreement.py ``measure_match``."""
    from scripts.measure_cover_shadow_argmax_agreement import measure_match
    from tests.invariants.link_population_registry import _xt

    def measure(actions, frames):
        return measure_match(actions, frames, _home(), _xt(), match_id="m")

    actions, frames = _two_games()
    _split_equal(measure(actions, frames), _per_game(measure))


def test_f1b_receiver_features_read_each_passes_own_game():
    """scripts/measure_f1b_feature_delta.py ``_feat_receiver``."""
    from scripts.measure_f1b_feature_delta import _feat_receiver

    def features(actions, frames):
        out = _feat_receiver(frames, actions.assign(type_name="pass"), _home())
        return None if out is None else pd.concat(out, axis=1)

    actions, frames = _two_games()
    _split_equal(features(actions, frames), _per_game(features))


def test_receiver_candidate_rows_read_each_passes_own_game():
    """scripts/train_receiver_model.py ``extract_candidate_rows``."""
    from scripts.train_receiver_model import extract_candidate_rows

    actions, frames = _two_games()
    _split_equal(extract_candidate_rows(actions, frames), _per_game(extract_candidate_rows))


def _intercepted(actions):
    """Action 3 of each game becomes a failed pass that the opponent's next touch (action 4) intercepts."""
    third = (actions["action_id"] == 3).to_numpy()
    return actions.assign(type_id=actions["type_id"].mask(third, 0), result_id=actions["result_id"].mask(third, 0))


def _frames_read_by(monkeypatch, module, name, frame_arg: int) -> list[set]:
    """Record the games of the frame that ``module.name`` receives as positional argument ``frame_arg``."""
    seen: list[set] = []
    original = getattr(module, name)

    def spy(*args, **kwargs):
        seen.append(set(args[frame_arg]["game_id"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    return seen


def test_trajectory_weak_labels_read_each_passes_own_game(monkeypatch):
    """scripts/_receiver_validation.py ``trajectory_weak_labels``: the frame of each intercepted pass is the frame
    of its own game (one game per frame read), and the per-game results agree."""
    import scripts._receiver_validation as rv

    actions, frames = _two_games()
    seen = _frames_read_by(monkeypatch, rv, "_teammates", 0)
    whole = rv.trajectory_weak_labels(_intercepted(actions), frames)
    assert len(seen) == 2  # non-vacuity: both games' intercepted passes read a frame
    assert all(len(games) == 1 for games in seen), seen
    parts = [rv.trajectory_weak_labels(_intercepted(a), f) for a, f in split_by_game(actions, frames)]
    _split_equal(whole, parts)


def test_failed_pass_accuracy_reads_each_passes_own_game():
    """scripts/_receiver_validation.py ``receiver_failed_pass_accuracy``: the model ranks each failed pass on the
    frame of its own game."""
    from scripts._receiver_validation import receiver_failed_pass_accuracy

    seen: list[set] = []

    class _Recorder:
        def rank(self, act, fr):
            seen.append(set(fr["game_id"]))
            return pd.Series(dtype="float64")

    actions, frames = _two_games()
    receiver_failed_pass_accuracy(_Recorder(), _intercepted(actions), frames)
    assert seen and all(len(games) == 1 for games in seen), seen


def test_played_passes_read_each_passes_own_game():
    """scripts/_rq_corpus.py ``extract_played_passes``."""
    from scripts._rq_corpus import extract_played_passes

    actions, frames = _two_games()
    _split_equal(extract_played_passes(actions, frames), _per_game(extract_played_passes))


def test_rq_pass_scores_read_each_passes_own_game():
    """scripts/build_rq_pass_scores.py ``score_match``."""
    from scripts.build_rq_pass_scores import score_match

    actions, frames = _two_games()
    _split_equal(score_match(actions, frames), _per_game(score_match))
