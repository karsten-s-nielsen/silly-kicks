"""The licensing control (spec 3.2). A model trained on restricted data must NEVER be labelled
`public`. Today's provider-name rule labels an sc_extended-shaped run "public" -- verified.

`scripts/` is on sys.path via tests/scripts/conftest.py, so `_corpus` / `train_xshot_occurrence`
import by bare name.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest
from _corpus import PUBLIC_CORPUS, artifact_label, is_public_row


def test_absent_visibility_is_restricted():
    """FAIL-CLOSED: unknown provenance is never public."""
    vis = {("skillcorner", "1886347"): "public"}
    got = is_public_row(
        providers=np.array(["skillcorner", "skillcorner"]),
        match_ids=np.array(["1886347", "9999999"]),  # second is absent from the map
        visibility=vis,
    )
    assert list(got) == [True, False]


def test_a_restricted_skillcorner_match_is_not_public():
    vis = {("skillcorner", "1021404"): "private"}
    got = is_public_row(providers=np.array(["skillcorner"]), match_ids=np.array(["1021404"]), visibility=vis)
    assert list(got) == [False]


def test_label_is_never_public_when_the_ship_mask_contains_restricted_rows():
    """The bug that shipped: providers={skillcorner, idsse}, no GS -> old code said "public"."""
    assert artifact_label(providers={"skillcorner", "idsse"}, all_public=False) == "sc_extended"
    assert artifact_label(providers={"skillcorner", "idsse"}, all_public=True) == "public"
    assert artifact_label(providers={"skillcorner", "gradientsports"}, all_public=False) == "full"


def test_public_corpus_is_the_known_17():
    assert len(PUBLIC_CORPUS["skillcorner"]) == 10
    assert len(PUBLIC_CORPUS["idsse"]) == 7


def test_public_corpus_skillcorner_matches_the_sample_loader():
    # The 10 public SkillCorner ids are LICENSING-critical and duplicated in
    # tests/_skillcorner_sample.MATCH_IDS (the sample loader). Guard the two literals against drift:
    # a divergence could load-as-sample a match not registered-as-public, or the reverse.
    from tests._skillcorner_sample import MATCH_IDS

    assert PUBLIC_CORPUS["skillcorner"] == frozenset(MATCH_IDS)


# --- the RED-FIRST slow test (fails against TODAY's LABEL path) ---


def _xshot_frame_rows(fid, t, ball_x):
    """One frame's full-schema rows (mirrors tests/tracking/test_xshot_occurrence_integration.py):
    ball + team-1 (defends x=0) + team-2 (attacks, carrier near the ball). ``ball_x`` drives the
    goal distance `r`, the dominant feature."""
    rows = [
        dict(
            player_id=-1, team_id=-1, is_ball=True, is_goalkeeper=False, x=ball_x, y=34.0, frame_id=fid, time_seconds=t
        ),
        dict(player_id=10, team_id=1, is_ball=False, is_goalkeeper=True, x=2.0, y=34.0, frame_id=fid, time_seconds=t),
        dict(player_id=20, team_id=2, is_ball=False, is_goalkeeper=True, x=103.0, y=34.0, frame_id=fid, time_seconds=t),
        dict(
            player_id=21,
            team_id=2,
            is_ball=False,
            is_goalkeeper=False,
            x=ball_x + 0.3,
            y=34.0,
            frame_id=fid,
            time_seconds=t,  # carrier 0.3 m from ball -> team-2 possession
        ),
    ]
    for k in range(5):  # team-1 outfield defenders
        rows.append(
            dict(
                player_id=11 + k,
                team_id=1,
                is_ball=False,
                is_goalkeeper=False,
                x=6.0 + k,
                y=28.0 + 2 * k,
                frame_id=fid,
                time_seconds=t,
            )
        )
    for k in range(4):  # team-2 outfield attackers
        rows.append(
            dict(
                player_id=22 + k,
                team_id=2,
                is_ball=False,
                is_goalkeeper=False,
                x=ball_x + 2 + k,
                y=30.0 + 2 * k,
                frame_id=fid,
                time_seconds=t,
            )
        )
    return rows


def _game_frames_and_shots(game_id):
    """Learnable game: pre-shot frames put the ball NEAR the attacked goal (small `r`) and quiet
    frames keep it in the attacking third but farther (larger `r`), so `r` cleanly predicts the
    label -> the acceptance gates PASS. 20 episodes x (3 near positive + 3 far negative)."""
    rows, shot_times, fid = [], [], 0
    for ep in range(20):
        base = ep * 7.0
        shot_times.append(base + 0.9)  # one shot covers the 3 near frames (each within 1 s)
        specs = [
            (base, 8.0),
            (base + 0.3, 8.0),
            (base + 0.6, 8.0),  # near goal (r~8) -> positive
            (base + 2.5, 30.0),
            (base + 3.0, 30.0),
            (base + 3.5, 30.0),  # far in the third (r~30) -> negative
        ]
        for t, ball_x in specs:
            rows.extend(_xshot_frame_rows(fid, t, ball_x))
            fid += 1
    frames = pd.DataFrame(rows)
    frames["game_id"] = game_id
    frames["period_id"] = 1
    frames["z"] = 0.0
    frames["frame_rate"] = 10.0
    frames["ball_state"] = "alive"
    shots = pd.DataFrame(
        {
            "game_id": [game_id] * len(shot_times),
            "period_id": [1] * len(shot_times),
            "team_id": [2] * len(shot_times),
            "time_seconds": shot_times,
        }
    )
    home = frames["team_id"].dropna().iloc[0]
    return frames, shots, home


def _synthetic_two_match_corpus():
    """Two skillcorner matches -- id "1886347" (public) + "1021404" (private). game_id (the CV
    grouping key) is distinct from the pining match_id (the visibility key), exactly as in prod."""
    for game_id, match_id in [(0, "1886347"), (1, "1021404")]:
        frames, shots, home = _game_frames_and_shots(game_id)
        yield "skillcorner", match_id, shots, frames, home


@pytest.mark.slow
def test_a_restricted_corpus_NEVER_ships_a_public_label(tmp_path, monkeypatch):
    """RED-FIRST against the pre-Task-9 LABEL path (which FAILS this).

    Pre-Task-9: providers = {skillcorner}, no GS -> `two_candidate` is False -> the else branch
    runs -> `provset <= _PUBLIC_PROVIDERS` is True -> a model trained on RESTRICTED matches ships
    labelled "public". Verified at train_xshot_occurrence.py:313.

    KILL-LINE: restore `if provset <= _PUBLIC_PROVIDERS: shipped = "public"` and this MUST fail.
    """
    sys.path.insert(0, "scripts")
    import train_xshot_occurrence as tr

    monkeypatch.setattr(tr, "_iter_matches_from_pining", lambda *a, **k: iter(_synthetic_two_match_corpus()))
    monkeypatch.setattr(
        "_loader_pining.match_visibility",
        lambda providers, **k: {
            ("skillcorner", "1886347"): "public",
            ("skillcorner", "1021404"): "private",
        },
    )
    out = tmp_path / "run"
    # `--allow-dirty`: ADR-052 enrolled all five weight trainers in the clean-tree guard, and a
    # test run is by definition a dev run. The artifact still records run_tree_dirty=true.
    tr.main(["--providers", "skillcorner", "--output-dir", str(out), "--n-trials", "1", "--allow-dirty"])

    metrics = json.loads((out / "xshot_occurrence_v1" / "metrics.json").read_text())
    assert metrics["shipped_variant"] != "public", (
        "a model trained on restricted data was labelled public -- the licensing landmine"
    )
    assert metrics["shipped_variant"] == "sc_extended"


def test_the_corpus_pin_and_the_visibility_arm_are_orthogonal():
    """Task 13: the --match-ids-json pin says WHICH matches load; visibility says WHICH ARM they
    join. They are orthogonal -- a pinned PRIVATE match is still private (never enters the public
    arm just because it was explicitly selected to load)."""
    vis = {("skillcorner", "1021404"): "private"}
    got = is_public_row(providers=np.array(["skillcorner"]), match_ids=np.array(["1021404"]), visibility=vis)
    assert list(got) == [False]
