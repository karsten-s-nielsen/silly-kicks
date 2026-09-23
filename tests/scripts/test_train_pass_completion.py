"""Task 9: the pass-completion trainer's ``--all-competitions`` full-corpus re-fit mode.

Smoke-level (``not e2e``): ``all_open_competitions`` + ``load_open_data_matches`` are monkeypatched to
a tiny in-memory fixture, so no network and no real StatsBomb corpus is touched. The single-competition
default path stays byte-identical; the all-competitions path resolves to a DIFFERENT ``for_each``
shard generation (the 4.77.1 stale-shard rule -- a corpus change must invalidate the generation).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import scripts.train_pass_completion as TPC
from silly_kicks.spadl import config as spadlconfig

_ACT_COLS = [
    "action_id",
    "game_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "type_id",
    "result_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
]


def _match(game_id: int, *, complete: bool):
    """One match with a couple of finite-coordinate pass rows -- one completed, one failed -- so the
    pooled fit has both label classes and enough rows to fit the logistic model."""
    P = spadlconfig.actiontype_id["pass"]
    S = spadlconfig.result_id["success"]
    F = spadlconfig.result_id["fail"]
    rows = [
        (1, game_id, 1, 10.0, 1, 9, P, S, 30.0, 34.0, 55.0, 40.0),
        (2, game_id, 1, 11.0, 1, 10, P, S, 55.0, 40.0, 70.0, 30.0),
        (3, game_id, 1, 12.0, 1, 11, P, F, 70.0, 30.0, 90.0, 20.0),
        (4, game_id, 1, 13.0, 2, 20, P, S, 40.0, 50.0, 60.0, 34.0),
    ]
    actions = pd.DataFrame(rows, columns=_ACT_COLS)
    return ("statsbomb", str(game_id), actions, pd.DataFrame(), 1)


def _fake_open_data_source_factory(order=None):
    """An ``open_data_source`` stand-in (Task 8.5): returns ``(refs, load)`` for 2 tiny matches per
    (competition, season). ``order``, if given, records ``"load"`` the first time the loader runs
    (the guard-order test asserts the public-only guard precedes the first match load)."""
    from _fake_corpus import make_ref

    def _source(comps, *, match_ids=None, max_matches=None, preserve_native=(), cache_dir=None):
        refs, by_id = [], {}
        for competition_id, season_id in comps:
            base = competition_id * 1000 + season_id
            for gid, complete in ((base + 1, True), (base + 2, False)):
                refs.append(make_ref("statsbomb", str(gid)))
                by_id[str(gid)] = _match(gid, complete=complete)

        def _load(ref):
            if order is not None and "load" not in order:
                order.append("load")
            return by_id[ref.match_id]

        return refs, _load

    return _source


def test_all_competitions_trains_full_corpus_bundle(tmp_path, monkeypatch):
    """``--all-competitions`` iterates every open competition, shards each's matches, and records
    ``n_competitions`` + the competition list; the card notes the full public open-data corpus."""
    comps = [(43, 106), (11, 90)]
    monkeypatch.setattr("scripts._sb_open_data.all_open_competitions", lambda: list(comps))
    monkeypatch.setattr("scripts._sb_open_data.open_data_source", _fake_open_data_source_factory())
    monkeypatch.setattr("scripts._sb_open_data.assert_statsbomb_open_data_mode", lambda: None)

    out = tmp_path / "out"
    TPC.main(["--all-competitions", "--out", str(out), "--allow-dirty"])

    assert (out / "weights" / "model.json").is_file()
    assert (out / "weights" / "MODEL_CARD.md").is_file()
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_competitions"] == 2
    assert sorted(tuple(c) for c in metrics["competitions"]) == sorted(comps)
    card = (out / "weights" / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "full public open-data corpus" in card
    # the bundle loads back through the ADR-011 integrity path
    from silly_kicks.expected_passing import PassCompletionModel

    PassCompletionModel.load(out / "weights")


def test_all_competitions_calls_the_public_only_guard_before_load(tmp_path, monkeypatch):
    """Fail-closed: ``assert_statsbomb_open_data_mode`` runs before any corpus load."""
    order: list[str] = []

    def _guard():
        order.append("guard")

    def _all():
        order.append("all_open_competitions")
        return [(43, 106)]

    monkeypatch.setattr("scripts._sb_open_data.assert_statsbomb_open_data_mode", _guard)
    monkeypatch.setattr("scripts._sb_open_data.all_open_competitions", _all)
    monkeypatch.setattr("scripts._sb_open_data.open_data_source", _fake_open_data_source_factory(order))

    TPC.main(["--all-competitions", "--out", str(tmp_path / "out"), "--allow-dirty"])
    assert order[0] == "guard"
    assert order.index("guard") < order.index("load")


def test_all_competitions_token_DIFFERS_from_single_competition(tmp_path, monkeypatch):
    """4.77.1 stale-shard rule: a different corpus MUST resolve to a different shard generation, so
    a re-run on the same ``--out`` cannot silently reuse the single-competition (WC2022) shards."""
    comps = [(43, 106), (11, 90)]
    monkeypatch.setattr("scripts._sb_open_data.all_open_competitions", lambda: list(comps))
    monkeypatch.setattr("scripts._sb_open_data.open_data_source", _fake_open_data_source_factory())
    monkeypatch.setattr("scripts._sb_open_data.assert_statsbomb_open_data_mode", lambda: None)

    single_root = tmp_path / "single" / "shards"
    all_root = tmp_path / "all" / "shards"
    TPC.main(["--out", str(tmp_path / "single"), "--allow-dirty"])
    TPC.main(["--all-competitions", "--out", str(tmp_path / "all"), "--allow-dirty"])

    single_gens = {p.name for p in single_root.iterdir() if p.is_dir()}
    all_gens = {p.name for p in all_root.iterdir() if p.is_dir()}
    assert single_gens and all_gens
    assert single_gens.isdisjoint(all_gens), (
        f"all-competitions shard generation {all_gens} collides with the single-competition "
        f"generation {single_gens}: a corpus change did not invalidate the shard generation"
    )


@pytest.mark.parametrize(("flag", "value"), [("--competition-id", "11"), ("--season-id", "106")])
def test_all_competitions_is_mutually_exclusive_with_single_competition_flags(tmp_path, flag, value):
    """``--all-competitions`` with an explicit ``--competition-id`` OR ``--season-id`` is a contradiction
    and must abort -- the guard is ``args.competition_id is not None or args.season_id is not None``, so
    BOTH flags need coverage (a season-only override would silently slip past a competition-only test)."""
    with pytest.raises(SystemExit):
        TPC.main(
            [
                "--all-competitions",
                flag,
                value,
                "--out",
                str(tmp_path / "out"),
                "--allow-dirty",
            ]
        )
