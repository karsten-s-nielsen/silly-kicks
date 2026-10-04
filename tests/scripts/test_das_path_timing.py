"""The add_das / das_xfns timing harness (combined-cycle-completion spec 12 D1)."""

import json
import sys

import pandas as pd
import pytest

from scripts import _das_path_timing as pt


def test_time_add_das_and_xfns_uses_precomputed_links_and_warms_up(monkeypatch):
    import silly_kicks.tracking.features as feats
    import silly_kicks.tracking.utils as tu
    import silly_kicks.vaep.feature_framework as ff

    seen = {"add": 0, "xfn": 0}
    monkeypatch.setattr(tu, "link_actions_to_frames", lambda a, f: ("LINKS", None))
    monkeypatch.setattr(ff, "gamestates", lambda a, nb_prev_actions: ["STATES"])

    def fake_add(actions, frames, *, links):
        assert links == "LINKS"
        seen["add"] += 1

    def fake_xfn(states, frames):
        assert states == ["STATES"]
        seen["xfn"] += 1

    monkeypatch.setattr(feats, "add_das", fake_add)
    monkeypatch.setattr(feats, "das_xfns", [fake_xfn])
    out = pt.time_add_das_and_xfns(pd.DataFrame(), pd.DataFrame(), repeat=3, warmup=True)
    assert seen == {"add": 4, "xfn": 4}  # 1 warm-up + 3 timed each
    assert set(out) == {"add_das_s", "das_xfns_s"} and min(out.values()) >= 0.0


def test_main_writes_timing_json_with_version_and_engine_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pt, "time_add_das_and_xfns", lambda f, a, *, repeat, warmup, state: {"add_das_s": 1.0, "das_xfns_s": 2.0}
    )
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "frames.parquet")
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "actions.parquet")
    pt._main(str(tmp_path), str(tmp_path), 3, 100.0)
    t = json.loads((tmp_path / "timing.json").read_text(encoding="utf-8"))
    assert t["add_das_s"] == 1.0 and t["repeat"] == 3 and t["silly_kicks"] and t["native"] is True
    assert t["over_memory"] is False and t["limit_gib"] == 100.0


def test_add_das_time_is_kept_before_the_xfns_phase_starts(monkeypatch):
    # The old path ran out of memory in its das_xfns call on a GS match (combined-cycle Phase B). The add_das
    # time must already be in the shared state when das_xfns starts, so a ceiling trip there keeps it.
    import silly_kicks.tracking.features as feats
    import silly_kicks.tracking.utils as tu
    import silly_kicks.vaep.feature_framework as ff

    state: dict = {}
    seen_at_xfn: list = []
    monkeypatch.setattr(tu, "link_actions_to_frames", lambda a, f: ("LINKS", None))
    monkeypatch.setattr(ff, "gamestates", lambda a, nb_prev_actions: ["STATES"])
    monkeypatch.setattr(feats, "add_das", lambda actions, frames, *, links: None)
    monkeypatch.setattr(feats, "das_xfns", [lambda states, frames: seen_at_xfn.append(dict(state))])
    pt.time_add_das_and_xfns(pd.DataFrame(), pd.DataFrame(), repeat=1, warmup=True, state=state)
    assert seen_at_xfn and all(s["phase"] == "das_xfns" and "add_das_s" in s for s in seen_at_xfn)
    assert set(state) >= {"add_das_s", "das_xfns_s"}


def test_the_ceiling_trips_writes_what_completed_and_exits_clean(tmp_path):
    exits: list = []
    state = {"phase": "das_xfns", "add_das_s": 9.5, "silly_kicks": "4.127.0", "native": False}
    rss = iter([50 * 2**30, 120 * 2**30])
    c = pt._Ceiling(100.0, tmp_path / "timing.json", state, rss=lambda: next(rss), exit_fn=exits.append)
    assert c.check() is False and not (tmp_path / "timing.json").exists()  # under the ceiling: nothing
    assert c.check() is True
    t = json.loads((tmp_path / "timing.json").read_text(encoding="utf-8"))
    assert t["over_memory"] is True and t["phase"] == "das_xfns" and t["limit_gib"] == 100.0
    assert t["peak_gib"] == 120.0 and t["add_das_s"] == 9.5 and "das_xfns_s" not in t
    assert t["native"] is False  # the engine marker survives a trip
    assert exits == [0]  # a clean exit: the driver reads the record, it does not see a crash


def test_rss_reader_never_raises():
    rss = pt._rss_bytes()
    assert rss is None or rss >= 0


def test_rss_is_unmeasured_off_linux(monkeypatch):
    # B r3 m-2: off Linux there is no /proc, so the reading is "not measured" (None), never a measured-looking 0.
    monkeypatch.setattr(pt.sys, "platform", "win32")
    assert pt._rss_bytes() is None


def _write_inputs(d):
    pd.DataFrame({"a": [1]}).to_parquet(d / "frames.parquet")
    pd.DataFrame({"a": [1]}).to_parquet(d / "actions.parquet")


def test_main_arms_the_ceiling(tmp_path, monkeypatch):
    # B r3 m-1: _main itself must start the watchdog -- the guard against the box-killing OOM (F7). Without
    # ceiling.start() in _main this test waits for a trip that never comes and fails.
    import threading

    tripped = threading.Event()
    seen: dict = {}

    def fake_exit(code):
        seen["code"] = code
        seen["record"] = json.loads((tmp_path / "timing.json").read_text(encoding="utf-8"))
        tripped.set()

    def slow_timing(f, a, *, repeat, warmup, state):
        state["phase"] = "das_xfns"
        assert tripped.wait(5), "the watchdog never tripped: _main did not arm the ceiling"
        return {"add_das_s": 1.0, "das_xfns_s": 2.0}

    monkeypatch.setattr(pt, "_rss_bytes", lambda: 8 * 2**30)
    monkeypatch.setattr(pt, "_hard_exit", fake_exit)
    monkeypatch.setattr(pt, "time_add_das_and_xfns", slow_timing)
    _write_inputs(tmp_path)
    pt._main(str(tmp_path), str(tmp_path), 1, 1.0)
    assert seen["code"] == 0 and seen["record"]["over_memory"] is True and seen["record"]["limit_gib"] == 1.0


def test_main_records_an_unmeasured_ceiling_as_such(tmp_path, monkeypatch):
    # B r3 m-2: where memory cannot be read, the record says so instead of reporting a 0.0 peak.
    monkeypatch.setattr(pt, "_rss_bytes", lambda: None)
    monkeypatch.setattr(
        pt, "time_add_das_and_xfns", lambda f, a, *, repeat, warmup, state: {"add_das_s": 1.0, "das_xfns_s": 2.0}
    )
    _write_inputs(tmp_path)
    pt._main(str(tmp_path), str(tmp_path), 1, 100.0)
    t = json.loads((tmp_path / "timing.json").read_text(encoding="utf-8"))
    assert t["memory_measured"] is False and t["peak_gib"] is None and t["over_memory"] is False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the RSS reader is /proc-based (Linux only)")
def test_the_watchdog_stops_a_real_process_and_leaves_the_record(tmp_path):
    # End to end: a real process, a real watchdog thread, a ceiling every process exceeds. It must write the
    # record and exit 0 long before its own 30 s sleep ends -- the mechanism that keeps the box alive.
    import subprocess
    import time

    code = (
        "import sys, time; from pathlib import Path; sys.path.insert(0, 'scripts'); import _das_path_timing as pt; "
        f"c = pt._Ceiling(1e-6, Path(r'{tmp_path / 'timing.json'}'), {{'phase': 'das_xfns', 'add_das_s': 1.5}}, "
        "interval=0.05); c.start(); time.sleep(30)"
    )
    t0 = time.monotonic()
    rc = subprocess.run([sys.executable, "-c", code], check=False, timeout=25).returncode  # noqa: S603
    assert rc == 0 and time.monotonic() - t0 < 20
    t = json.loads((tmp_path / "timing.json").read_text(encoding="utf-8"))
    assert t["over_memory"] is True and t["phase"] == "das_xfns" and t["add_das_s"] == 1.5
