"""ADR-128 Task 20: the ADR-126 integrity inputs travel with the run (extract counters), and every reduced
stage records the inputs the job read (`_inputs.json`, incl. the dataset revision `adopt` checks)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts._integrity_counters import integrity_counters
from scripts._stages import LocalStore, item_counters, make_context, reduce_stage, run_unit, sorted_units
from scripts._toy_corpora import build_xcross_corpus, build_xcross_probe_corpus, build_xshot_corpus, xshot_arms
from silly_kicks.spadl import config as spadlconfig
from tests.scripts._stage_harness import PROV, params_from_args


def test_integrity_counters_count_the_label_domain_frames_and_duration():
    shot, passid = spadlconfig.actiontype_id["shot"], spadlconfig.actiontype_id["pass"]
    events = pd.DataFrame({"type_id": [shot, passid, shot]})
    frames = pd.DataFrame(
        {
            "period_id": [1, 1, 1, 2, 2],
            "frame_id": [0, 0, 1, 7, 8],  # frame 0 has two rows (a player and the ball)
            "time_seconds": [0.0, 0.0, 0.1, 5.0, 5.4],
        }
    )
    assert integrity_counters(frames, events, ("shot",)) == {"n_events": 2, "n_frames": 4, "duration_s": 0.5}


def _extract_only(pipeline, args, tmp_path, data):
    ctx = make_context(
        pipeline, params_from_args(pipeline, args), PROV, store=LocalStore(tmp_path / "s"), data_root=data
    )
    st = pipeline.stage("extract")
    refs = st.ref_map(ctx)
    for key in sorted_units(ctx, st):
        run_unit(ctx, st, key, worker="t", ref=refs.get(key))
    reduce_stage(ctx, "extract")
    return ctx, refs


@pytest.mark.parametrize("family", ["xshot", "xcross"])
def test_the_extract_stage_records_the_integrity_counters_per_match(tmp_path, family):
    data = tmp_path / "data"
    if family == "xshot":
        import scripts.train_xshot_occurrence as mod
        from silly_kicks.tracking._xshot_occurrence import _DEFAULT_SHOT_TYPES as LABEL_TYPES

        build_xshot_corpus(data / "toyxs", heavy_episodes=None)
        args = ["--dataset", "toyxs", "--arms-json", xshot_arms(data / "toyxs", paired=False), "--n-trials", "1"]
    else:
        import scripts.train_xcross_attempt as mod
        from silly_kicks.tracking._xcross_attempt import _DEFAULT_CROSS_TYPES as LABEL_TYPES

        build_xcross_corpus(data / "toyxc", heavy_episodes=None)
        build_xcross_probe_corpus(data / "toyxcprobe")
        args = ["--dataset", "toyxc", "--probe-dataset", "toyxcprobe", "--arms-json",
                xshot_arms(data / "toyxc", paired=False), "--n-trials", "1"]  # fmt: skip
    ctx, refs = _extract_only(mod.PIPELINE, args, tmp_path, data)
    totals, missing = item_counters(ctx, "extract")
    assert missing == 0
    ids = {spadlconfig.actiontype_id[t] for t in LABEL_TYPES}
    root = data / args[1]
    want_events = sum(int(pd.read_parquet(root / r.files[1])["type_id"].isin(ids).sum()) for r in refs.values())
    assert totals["n_events"] == want_events > 0
    assert totals["n_frames"] > 0 and totals["duration_s"] > 0


def test_a_reduced_stage_records_the_inputs_it_read(tmp_path):
    import scripts.train_xshot_occurrence as mod

    data = tmp_path / "data"
    build_xshot_corpus(data / "toyxs", heavy_episodes=None)
    args = ["--dataset", "toyxs", "--arms-json", xshot_arms(data / "toyxs", paired=False), "--n-trials", "1"]
    ctx, _refs = _extract_only(mod.PIPELINE, args, tmp_path, data)
    raw = ctx.store.read_bytes(f"{ctx.generation('extract')}/_inputs.json")
    assert raw is not None
    rec = json.loads(raw)
    assert rec == ctx.token_inputs["extract"]
    assert rec["dataset"]["name"] == "toyxs"  # a local toy corpus: its listing digest stands in for a revision
