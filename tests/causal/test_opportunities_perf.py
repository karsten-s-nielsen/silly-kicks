"""Structural perf guard for build_opportunities (ADR-068): the per-frame lookup is built ONCE
per (game, period), not re-scanned per frame."""

from silly_kicks.causal import opportunities as O
from tests._perf_structural import call_counter
from tests.causal._fixtures import META, WIDE, actions, frames


def test_frame_lookup_built_once_per_period(monkeypatch):
    calls = call_counter(monkeypatch, O, "group_rows")
    # 4 frames, all period 1 -> ONE (game, period) group. A per-frame rescan would build 4x.
    f = frames({10.0: 5, 10.2: 5, 10.4: 5, 10.6: 5}, {t: WIDE for t in (10.0, 10.2, 10.4, 10.6)})
    O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    assert calls["n"] == 1  # once per (game, period), NOT once per frame


def test_mutation_would_raise_the_count(monkeypatch):
    # Mutation proof: if group_rows were (wrongly) rebuilt per frame, the count would exceed 1.
    # We assert the invariant is meaningful by counting frames in the fixture.
    calls = call_counter(monkeypatch, O, "group_rows")
    f = frames({10.0: 5, 10.2: 5, 10.4: 5, 10.6: 5}, {t: WIDE for t in (10.0, 10.2, 10.4, 10.6)})
    O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    n_frames = f.loc[~f["is_ball"], "frame_id"].nunique()
    assert n_frames >= 4  # the fixture really has >=4 frames a per-frame scan would have hit
    assert calls["n"] < n_frames  # built-once is strictly fewer than the per-frame rescan count
