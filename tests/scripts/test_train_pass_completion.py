"""The pass-completion pipeline (HFU spec 15.5a) on a tiny in-memory open-data corpus (no network), and the
owner-tier pining CLI (``scripts/_pass_completion_pining.py``, the I1 severance, ADR-123 Amendment B)."""

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


def _patch_open_data(monkeypatch, comps, order=None):
    """The open-data seams the pipeline calls, over 2 tiny matches per release; ``order`` records the guard and
    the first load (the guard-order test)."""
    import scripts._sb_open_data as sod
    from scripts._driver_sources import LoadedMatch

    by_id = {str(c * 1000 + s + k): _match(c * 1000 + s + k, complete=True) for c, s in comps for k in (1, 2)}

    def _guard():
        if order is not None:
            order.append("guard")

    def _load(ref, *, preserve_native=(), cache_dir=None):
        if order is not None and "load" not in order:
            order.append("load")
        return LoadedMatch("statsbomb", ref.match_id, by_id[ref.match_id][2], pd.DataFrame(), 1, None, None)

    def _refs(cs, *, match_ids=None, max_matches=None):
        return [sod.OpenDataRef(c, s, str(c * 1000 + s + k), 1) for c, s in cs for k in (1, 2)]

    monkeypatch.setattr(sod, "assert_statsbomb_open_data_mode", _guard)
    monkeypatch.setattr(sod, "all_open_competitions", lambda: list(comps))
    monkeypatch.setattr(sod, "list_open_data_refs", _refs)
    monkeypatch.setattr(sod, "load_open_data_match", _load)


def _run(tmp_path, args):
    from tests.scripts._stage_harness import export, run_in_process

    ctx = run_in_process(TPC.PIPELINE, args, store=tmp_path / "store", data_root=None)
    return ctx, export(ctx, tmp_path / "out")


def test_all_competitions_trains_full_corpus_bundle(tmp_path, monkeypatch):
    """``--all-competitions`` walks every release and records ``n_competitions`` + the competition list; the
    bundle loads through the ADR-011 integrity path and the card notes the full public open-data corpus."""
    from silly_kicks.expected_passing import PassCompletionModel

    comps = [(43, 106), (11, 90)]
    _patch_open_data(monkeypatch, comps)
    _ctx, out = _run(tmp_path, ["--all-competitions"])
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_competitions"] == 2 and sorted(tuple(c) for c in metrics["competitions"]) == sorted(comps)
    card = (out / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "full public open-data corpus" in card
    assert "`python -m scripts.train_pass_completion --run-all --store <STORE> --out <DIR>\n--all-competitions`" in card
    PassCompletionModel.load(out)


def test_metrics_records_the_open_data_identity(tmp_path, monkeypatch):
    """HFU spec 14.3: metrics.json records the corpus identity of the refs, which verify compares with the lock."""
    from scripts._corpus_identity import open_data_identity

    _patch_open_data(monkeypatch, [(43, 106), (11, 90)])
    _ctx, out = _run(tmp_path, ["--all-competitions"])
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    want = open_data_identity(["43107", "43108", "11091", "11092"])
    assert {k: metrics[k] for k in want} == want


def test_metrics_records_the_corpus_pass_fields(tmp_path, monkeypatch):
    """Decision P4 (ADR-128), as win_probability and the legacy fit: metrics.json records the extract pass."""
    _patch_open_data(monkeypatch, [(43, 106), (11, 90)])
    ctx, out = _run(tmp_path, ["--all-competitions"])
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    want = {"generation": ctx.tokens["extract"], "n_attempted": 4, "n_failed": 0, "n_counters_unrecorded": 0}
    assert {k: metrics[k] for k in (*want, "n_excluded")} == {**want, "n_excluded": 0}


def test_the_public_only_guard_runs_before_any_load(tmp_path, monkeypatch):
    order: list[str] = []
    _patch_open_data(monkeypatch, [(43, 106)], order)
    _run(tmp_path, ["--all-competitions"])
    assert order[0] == "guard" and order.index("guard") < order.index("load")


def test_all_competitions_token_differs_from_single_competition(tmp_path, monkeypatch):
    """A different corpus resolves to a different `refs` token, so its generation can never be reused."""
    from scripts._stages import LocalStore, make_context
    from tests.scripts._stage_harness import PROV, params_from_args

    _patch_open_data(monkeypatch, [(43, 106), (11, 90)])

    def token(args):
        store = LocalStore(tmp_path)
        ctx = make_context(TPC.PIPELINE, params_from_args(TPC.PIPELINE, args), PROV, store=store, data_root=None)
        return ctx.tokens["refs"]

    assert token([]) != token(["--all-competitions"])


def test_the_pining_cli_needs_the_injected_source(tmp_path):
    """I1 severance: the HF module runs the pipeline; its pining CLI cannot run without the injected source."""
    with pytest.raises(TypeError):
        TPC.pining_main(["--out", str(tmp_path / "out"), "--allow-dirty"])  # type: ignore[call-arg]
    assert not (tmp_path / "out").exists()


def test_the_pining_entry_trains_through_the_injected_source(tmp_path, monkeypatch):
    """The owner entry injects the events-only pining source; the fit, metrics and card are the trainer's
    own. The IMPL-05 warning still fires, and the card reproduces through the entry with the bare
    ``--providers`` list (the corpus label carries a ``pining:`` prefix)."""
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


def test_card_states_corpus_identity():
    """HFU spec 14.13: the card states the corpus identity verify checks (count + digest)."""
    from scripts.train_pass_completion import render_model_card

    card = render_model_card({"corpus_match_ids_sha256": "ab" * 32, "corpus_n_matches": 3961})
    assert f"Corpus identity: 3961 matches, sha256 {'ab' * 32}" in card


@pytest.mark.parametrize(
    ("metrics", "corpus"),
    [
        ({"providers": "statsbomb-open (competition 43, season 106)"}, {"competition_id": 43, "season_id": 106}),
        ({"providers": "statsbomb-open (all 2 releases)", "n_competitions": 2}, {"all_competitions": True}),
    ],
)
def test_card_reproduce_command_parses_on_the_pipeline_cli(metrics, corpus):
    """The card's open-data reproduce command is accepted by the real pipeline parser (every required argument)."""
    import re
    import shlex

    from scripts._stages import build_parser
    from scripts.train_pass_completion import render_model_card

    cmd = re.search(r"Reproduce with `([^`]+)`", render_model_card(metrics), re.S)
    assert cmd is not None
    argv = shlex.split(cmd.group(1).replace("<STORE>", "s").replace("<DIR>", "o"))
    assert argv[:3] == ["python", "-m", "scripts.train_pass_completion"]
    ns = vars(build_parser(TPC.PIPELINE).parse_args(argv[3:]))
    assert ns["run_all"] and ns["store"] == "s" and str(ns["out"]) == "o"
    assert {k: ns[k] for k in corpus} == corpus
