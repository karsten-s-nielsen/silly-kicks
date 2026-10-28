"""The pass-completion trainer as a stage-contract pipeline (HFU R3-10 / D12), and its pining cross-check.

Smoke-level (``not e2e``): the open-data listing and loading are monkeypatched to a tiny in-memory fixture and
the pipeline runs IN PROCESS (``_stage_harness.run_in_process``: the real ``run_unit`` / ``reduce_stage``), so
no network and no real StatsBomb corpus is touched. ``--all-competitions`` resolves to a DIFFERENT extract
generation than the single-competition default (the 4.77.1 stale-shard rule). The owner-tier ``--source pining``
cross-check runs only through ``scripts/_pass_completion_pining.py`` (the I1 severance, ADR-123 Amendment B),
which owns the legacy ``for_each`` flow; its loaders are monkeypatched the same way. The byte identity with the
legacy CLI is ``test_adapter_migration_identity.py`` (the verbatim oracle, open-data toy tree).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import scripts.train_pass_completion as TPC
from scripts._driver_sources import LoadedMatch
from scripts._sb_open_data import OpenDataRef
from scripts._stages import StageIncompleteError
from silly_kicks.spadl import config as spadlconfig
from tests.scripts._stage_harness import export, failure_of, run_in_process

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


#: One synthetic match as `LoadedMatch` takes its first five fields (provider, id, actions, frames, home team).
_Match = tuple[str, str, pd.DataFrame, pd.DataFrame, int]


def _match(game_id: int, *, complete: bool) -> _Match:
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


def _patch_open_data(monkeypatch, *, comps, order=None):
    """Two tiny matches per (competition, season): the listing and the loader the pipeline's ``refs`` and
    ``extract`` stages call. ``order`` records ``"all_open_competitions"``, ``"guard"`` and the first
    ``"load"``."""
    import scripts._sb_open_data as sbod

    by_id: dict[str, _Match] = {}

    def _guard():
        if order is not None:
            order.append("guard")

    def _all():
        if order is not None:
            order.append("all_open_competitions")
        return list(comps)

    def _list(competitions, *, match_ids=None, max_matches=None):
        refs = []
        for competition_id, season_id in competitions:
            base = competition_id * 1000 + season_id
            for gid, complete in ((base + 1, True), (base + 2, False)):
                by_id[str(gid)] = _match(gid, complete=complete)
                refs.append(OpenDataRef(int(competition_id), int(season_id), str(gid), 1, ""))
        return refs if max_matches is None else refs[:max_matches]  # the GLOBAL cap, as the real walk

    def _load(ref, *, preserve_native=(), cache_dir=None):
        if order is not None and "load" not in order:
            order.append("load")
        return LoadedMatch(*by_id[ref.match_id], None, None)

    monkeypatch.setattr(sbod, "assert_statsbomb_open_data_mode", _guard)
    monkeypatch.setattr(sbod, "all_open_competitions", _all)
    monkeypatch.setattr(sbod, "list_open_data_refs", _list)
    monkeypatch.setattr(sbod, "load_open_data_match", _load)


def _run(tmp_path, args, name="s"):
    return run_in_process(TPC.PIPELINE, args, store=tmp_path / name, data_root=tmp_path / "data")


def test_all_competitions_trains_full_corpus_bundle(tmp_path, monkeypatch):
    """``--all-competitions`` lists every open competition, shards each match, and records
    ``n_competitions`` + the competition list; the card notes the full public open-data corpus and names
    the pipeline command (owner ruling 2026-10-30)."""
    comps = [(43, 106), (11, 90)]
    _patch_open_data(monkeypatch, comps=comps)
    out = export(_run(tmp_path, ["--all-competitions"]), tmp_path / "out")

    # one FLAT bundle (adopt refuses a nested path) that ships metrics.json (owner ruling 2026-10-30)
    assert sorted(p.name for p in out.iterdir()) == ["MODEL_CARD.md", "SHA256SUMS", "metrics.json", "model.json"]
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_competitions"] == 2 and metrics["n_matches"] == 4
    assert sorted(tuple(c) for c in metrics["competitions"]) == sorted(comps)
    card = (out / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "full public open-data corpus" in card
    assert "`python -m scripts.train_pass_completion --store <DIR> --run-all --out <DIR>\n--all-competitions`" in card
    from silly_kicks.expected_passing import PassCompletionModel

    PassCompletionModel.load(out)  # the bundle loads back through the ADR-011 integrity path


def test_the_public_only_guard_runs_before_the_listing_and_any_load(tmp_path, monkeypatch):
    """Fail-closed: ``assert_statsbomb_open_data_mode`` runs before the manifest walk and any corpus load."""
    order: list[str] = []
    _patch_open_data(monkeypatch, comps=[(43, 106)], order=order)
    _run(tmp_path, ["--all-competitions"])
    assert order[0] == "guard"
    assert order.index("guard") < order.index("all_open_competitions") < order.index("load")


def test_all_competitions_extract_generation_DIFFERS_from_single_competition(tmp_path, monkeypatch):
    """4.77.1 stale-shard rule: a different corpus MUST resolve to a different extract generation, so a re-run
    on the same store cannot silently reuse the single-competition (WC2022) shards."""
    _patch_open_data(monkeypatch, comps=[(43, 106), (11, 90)])
    single = _run(tmp_path, [], "single")
    every = _run(tmp_path, ["--all-competitions"], "all")
    assert single.generation("extract").rsplit("/", 1)[-1] != every.generation("extract").rsplit("/", 1)[-1]


def test_the_cv_stage_reproduces_the_pooled_out_of_fold_metrics(tmp_path, monkeypatch):
    """The fold units + the reduce give exactly ``cross_val_metrics`` on the same pooled rows (the arithmetic
    the legacy reduce ran in one process)."""
    _patch_open_data(monkeypatch, comps=[(43, 106), (11, 90), (55, 43)])
    ctx = _run(tmp_path, ["--all-competitions"])
    want = TPC.cross_val_metrics(ctx.upstream("extract"))
    assert json.dumps(ctx.upstream("cv"), sort_keys=True) == json.dumps(want, sort_keys=True)
    assert len(TPC.cv_folds(ctx.upstream("extract")) or []) == 5  # 6 matches: the 5-fold path


def test_fewer_than_two_matches_take_the_single_fit_path(tmp_path, monkeypatch):
    """One group: ``cv_folds`` is None and the out-of-fold array is the single in-sample fit, as before."""
    _patch_open_data(monkeypatch, comps=[(43, 106)])
    ctx = _run(tmp_path, ["--max-per-provider", "1"])
    pooled = ctx.upstream("extract")
    assert TPC.cv_folds(pooled) is None
    assert json.dumps(ctx.upstream("cv"), sort_keys=True) == json.dumps(TPC.cross_val_metrics(pooled), sort_keys=True)
    assert np.isfinite(ctx.upstream("cv")["brier"])


def test_source_pining_is_refused_on_the_hf_module(tmp_path, monkeypatch):
    """I1 severance (ADR-123 Amendment B): this module runs on HF and never imports the pining client, so
    ``--source pining`` fails loud in ``refs``, names the owner entry, and loads nothing."""
    order: list[str] = []
    _patch_open_data(monkeypatch, comps=[(43, 106)], order=order)
    with pytest.raises(StageIncompleteError):
        _run(tmp_path, ["--source", "pining"])
    from scripts._stages import LocalStore, make_context
    from tests.scripts._stage_harness import PROV, params_from_args

    ctx = make_context(
        TPC.PIPELINE,
        params_from_args(TPC.PIPELINE, ["--source", "pining"]),
        PROV,
        store=LocalStore(tmp_path / "s"),
        data_root=tmp_path / "data",
    )
    assert "scripts/_pass_completion_pining.py" in failure_of(ctx, "refs")
    assert "load" not in order


@pytest.mark.parametrize(("flag", "value"), [("--competition-id", "11"), ("--season-id", "106")])
def test_all_competitions_is_mutually_exclusive_with_single_competition_flags(tmp_path, monkeypatch, flag, value):
    """``--all-competitions`` with an explicit ``--competition-id`` OR ``--season-id`` is a contradiction and
    must fail -- BOTH flags need coverage (a season-only override would silently slip past)."""
    _patch_open_data(monkeypatch, comps=[(43, 106)])
    with pytest.raises(StageIncompleteError):
        _run(tmp_path, ["--all-competitions", flag, value])


def test_the_pining_entry_trains_through_its_own_flow(tmp_path, monkeypatch):
    """The owner entry owns the legacy ``for_each`` flow over the trainer's pure functions; the IMPL-05
    warning still fires, and the card reproduces through the entry with the bare ``--providers`` list."""
    from _fake_corpus import make_ref

    import scripts._pass_completion_pining as PCP

    by_id = {str(g): _match(g, complete=True) for g in (1, 2)}
    asked: dict = {}

    def _refs(*, providers, match_ids=None, max_per_provider=None):
        asked["providers"] = providers
        return [make_ref("statsbomb", g) for g in by_id]

    def _events_only_loader(refs, *, cache_dir=None):
        return (lambda ref: by_id[ref.match_id]), None

    monkeypatch.setattr("scripts._loader_pining.list_match_refs", _refs)
    monkeypatch.setattr("scripts._events_admission.events_only_loader", _events_only_loader)

    out = tmp_path / "out"
    with pytest.warns(UserWarning, match="MUST NOT be bundled or published"):
        PCP.main(["--source", "pining", "--providers", "statsbomb", "--out", str(out), "--allow-dirty"])

    assert asked["providers"] == ["statsbomb"]
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["providers"] == "pining:statsbomb" and metrics["n_matches"] == 2
    card = (out / "weights" / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "`python scripts/_pass_completion_pining.py --out <DIR>\n--source pining --providers statsbomb`" in card
