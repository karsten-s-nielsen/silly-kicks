"""TF-46 Task 5 (B04): the xcross_probe build must stage the held-out M6 roster from the lock, not an
arbitrary first-N of the Gradient Sports catalog. The wrapper derives the roster from the dataset lock and
forwards it to the materializer as ``--match-ids-json``; without it the probe would stop being held out."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.build_xcross_probe as bxp


def test_probe_roster_is_the_lock_m6_pair():
    # the held-out pair recorded in datasets.lock.json (gradientsports-10502/10503), as native match ids
    assert bxp.probe_roster() == {"gradientsports": ["10502", "10503"]}


def test_materialize_forwards_match_ids_json(monkeypatch):
    # the materializer must RECEIVE the roster; without --match-ids-json it stages the whole catalog
    captured: dict = {}

    def fake_run(cmd, check):
        captured["cmd"] = list(cmd)

    monkeypatch.setattr(bxp.subprocess, "run", fake_run)
    bxp._materialize(Path("cache"), Path("flat"), ["gradientsports"], None, match_ids_json=Path("roster.json"))
    cmd = captured["cmd"]
    assert "--match-ids-json" in cmd
    assert cmd[cmd.index("--match-ids-json") + 1] == str(Path("roster.json"))


def test_main_defaults_to_the_lock_roster(monkeypatch, tmp_path):
    # with no --match-ids-json, main() writes the lock roster and hands it to the materializer
    seen: dict = {}

    def fake_materialize(cache_dir, flat_dir, providers, max_per_provider, match_ids_json=None):
        assert match_ids_json is not None
        seen["roster"] = json.loads(Path(match_ids_json).read_text(encoding="utf-8"))

    monkeypatch.setattr(bxp, "_materialize", fake_materialize)
    monkeypatch.setattr(bxp, "build_probe_pergame", lambda flat, out: 0)
    bxp.main(["--cache-dir", str(tmp_path / "c"), "--out", str(tmp_path / "o")])
    assert seen["roster"] == bxp.probe_roster()  # exactly the lock's held-out pair
