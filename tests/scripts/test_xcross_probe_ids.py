"""HFU spec 18.6 (F35): the xCross probe cohort records canonical match ids, not the staged per-game names."""

from __future__ import annotations

import json

import pandas as pd

import scripts.train_xcross_attempt as X


def test_the_probe_cohort_records_canonical_ids(monkeypatch):
    captured = {}
    monkeypatch.setattr(X, "_load_match", lambda ref, ctx: ("gradientsports", "x", pd.DataFrame(), pd.DataFrame(), 1))
    monkeypatch.setattr(X, "_write_probe_sample", lambda ps, cohort, providers: captured.update(cohort))
    row = pd.DataFrame({"provider": ["gradientsports"], "match_id": ["gradientsports-10502"], "groups": ['["g1"]']})
    refs = {X.join_key(("gradientsports", "gradientsports-10502")): object()}
    X._cohort_files(None, [row], refs, ["gradientsports"])
    assert captured["matches"] == [["gradientsports", "10502"]]
    assert json.loads(json.dumps(captured["match_groups"])) == {"10502": ["g1"]}
