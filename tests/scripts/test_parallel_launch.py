"""Guards for the memory-aware parallel launcher (sizing, backpressure, orchestration)."""

import sys
import threading
import time

import pytest

from scripts import _parallel_launch as pl

GiB = 1024**3


def test_ram_bound_below_cores():
    # 119 GiB total - 20 GiB headroom = 99 usable; 99 // 11 = 9, not the 20 cores.
    assert pl.size_workers(11 * GiB, 119 * GiB, nproc=20, headroom_bytes=20 * GiB) == 9


def test_capped_at_nproc_when_ram_is_ample():
    assert pl.size_workers(1 * GiB, 500 * GiB, nproc=8, headroom_bytes=20 * GiB) == 8


def test_clamped_to_at_least_one_when_tight():
    # One worker (11) does not fit under (30 - 20 = 10) usable, but fits in total (30) -> 1, never 0.
    assert pl.size_workers(11 * GiB, 30 * GiB, nproc=20, headroom_bytes=20 * GiB) == 1


def test_refuses_when_even_one_worker_cannot_fit():
    with pytest.raises(pl.RefusalError):
        pl.size_workers(200 * GiB, 119 * GiB, nproc=20, headroom_bytes=20 * GiB)


def test_backpressure_blocks_when_headroom_gone():
    assert pl.should_launch(mem_available_bytes=8 * GiB, peak_rss_bytes=11 * GiB, margin_bytes=1 * GiB) is False
    assert pl.should_launch(mem_available_bytes=13 * GiB, peak_rss_bytes=11 * GiB, margin_bytes=1 * GiB) is True


def test_relaunches_a_worker_that_exits_nonzero_until_shards_complete(tmp_path):
    # fake driver: writes a .done marker per item; first run of subset containing "b" exits 137.
    driver = tmp_path / "fake_driver.py"
    driver.write_text(
        "import sys,json,pathlib\n"
        "shard=pathlib.Path(sys.argv[sys.argv.index('--shard-root')+1])\n"
        "ids=json.loads(pathlib.Path(sys.argv[sys.argv.index('--subset')+1]).read_text())\n"
        "for i in ids:\n"
        "    if i=='b' and not (shard/'b.attempted').exists():\n"
        "        (shard/'b.attempted').write_text('1'); sys.exit(137)\n"
        "    (shard/(i+'.done')).write_text('ok')\n",
        encoding="utf-8",
    )
    shard_root = tmp_path / "shards"
    shard_root.mkdir()
    res = pl.run_parallel(
        cmd_template=[sys.executable, str(driver), "--shard-root", str(shard_root), "--subset", "{subset}"],
        subsets={"w0": ["a", "b"], "w1": ["c"]},
        cap_bytes=1 << 40,
        backend="none",
        peak_rss_bytes=1,
        headroom_bytes=0,
        done_marker=lambda i: shard_root / (i + ".done"),
        max_relaunch=3,
        shard_root=shard_root,
        poll_interval=0.1,
        mem_available_bytes=10**12,  # fixed budget: no dependency on host /proc/meminfo or psutil
    )
    assert {p.stem for p in shard_root.glob("*.done")} == {"a", "b", "c"}
    assert res.relaunched >= 1  # subset w0 relaunched after the 137 exit
    assert res.completed == 3


def test_run_parallel_uses_explicit_budget_without_reading_host_ram(tmp_path, monkeypatch):
    # A host with neither /proc/meminfo nor psutil: available_ram_bytes() raises. The explicit
    # mem_available_bytes budget must let the launcher run without ever calling it (DPL-IMPL-01).
    def _boom():
        raise RuntimeError("no /proc/meminfo, no psutil")

    monkeypatch.setattr(pl, "available_ram_bytes", _boom)
    root = tmp_path / "s"
    root.mkdir()
    driver = tmp_path / "q.py"
    driver.write_text(
        "import sys,json,pathlib\n"
        "root=pathlib.Path(sys.argv[sys.argv.index('--shard-root')+1])\n"
        "ids=json.loads(pathlib.Path(sys.argv[sys.argv.index('--subset')+1]).read_text())\n"
        "for i in ids:(root/(i+'.done')).write_text('ok')\n",
        encoding="utf-8",
    )
    res = pl.run_parallel(
        cmd_template=[sys.executable, str(driver), "--shard-root", str(root), "--subset", "{subset}"],
        subsets={"w0": ["a"], "w1": ["b"]},
        cap_bytes=1 << 40,
        backend="none",
        peak_rss_bytes=1,
        headroom_bytes=0,
        done_marker=lambda i: root / (i + ".done"),
        shard_root=root,
        poll_interval=0.1,
        mem_available_bytes=10**12,
    )
    assert {p.stem for p in root.glob("*.done")} == {"a", "b"}
    assert res.completed == 2


def test_workers_run_concurrently(tmp_path, monkeypatch):
    root = tmp_path / "s"
    root.mkdir()
    driver = tmp_path / "slow.py"
    driver.write_text(
        "import sys,json,pathlib,time\n"
        "root=pathlib.Path(sys.argv[sys.argv.index('--shard-root')+1])\n"
        "ids=json.loads(pathlib.Path(sys.argv[sys.argv.index('--subset')+1]).read_text())\n"
        "for i in ids:(root/(i+'.started')).write_text('1')\n"
        "time.sleep(3)\n"
        "for i in ids:(root/(i+'.done')).write_text('1')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(pl, "available_ram_bytes", lambda: 10**12)
    peak = [0]

    def watch():
        for _ in range(60):
            live = len(list(root.glob("*.started"))) - len(list(root.glob("*.done")))
            peak[0] = max(peak[0], live)
            time.sleep(0.2)

    t = threading.Thread(target=watch)
    t.start()
    pl.run_parallel(
        cmd_template=[sys.executable, str(driver), "--shard-root", str(root), "--subset", "{subset}"],
        subsets={"w0": ["a"], "w1": ["b"], "w2": ["c"]},
        cap_bytes=1 << 40,
        backend="none",
        peak_rss_bytes=1,
        headroom_bytes=0,
        done_marker=lambda i: root / (i + ".done"),
        shard_root=root,
        nproc=3,
        poll_interval=0.2,
    )
    t.join()
    assert peak[0] >= 2  # >=2 workers alive at once -> real parallelism


def test_backpressure_serializes_when_ram_is_tight(tmp_path, monkeypatch):
    root = tmp_path / "s"
    root.mkdir()
    driver = tmp_path / "quick.py"
    driver.write_text(
        "import sys,json,pathlib,time\n"
        "root=pathlib.Path(sys.argv[sys.argv.index('--shard-root')+1])\n"
        "ids=json.loads(pathlib.Path(sys.argv[sys.argv.index('--subset')+1]).read_text())\n"
        "for i in ids:(root/(i+'.started')).write_text('1')\n"
        "time.sleep(0.5)\n"
        "for i in ids:(root/(i+'.done')).write_text('1')\n",
        encoding="utf-8",
    )
    calls = {"n": 0}

    def fake_ram():
        calls["n"] += 1
        return 10**12 if calls["n"] == 1 else 1  # first call sizes n=3; rest fail the gate

    monkeypatch.setattr(pl, "available_ram_bytes", fake_ram)
    peak = [0]

    def watch():
        for _ in range(60):
            live = len(list(root.glob("*.started"))) - len(list(root.glob("*.done")))
            peak[0] = max(peak[0], live)
            time.sleep(0.05)

    t = threading.Thread(target=watch)
    t.start()
    pl.run_parallel(
        cmd_template=[sys.executable, str(driver), "--shard-root", str(root), "--subset", "{subset}"],
        subsets={"w0": ["a"], "w1": ["b"], "w2": ["c"]},
        cap_bytes=1 << 40,
        backend="none",
        peak_rss_bytes=1,
        headroom_bytes=0,
        margin_bytes=1,
        done_marker=lambda i: root / (i + ".done"),
        shard_root=root,
        nproc=3,
        poll_interval=0.05,
    )
    t.join()
    assert peak[0] == 1  # backpressure held it to one worker at a time
    assert {p.stem for p in root.glob("*.done")} == {"a", "b", "c"}  # still completes


def test_split_round_robin_deals_keys_and_drops_empties():
    assert pl.split_round_robin(["a", "b", "c", "d", "e"], 2) == {"w0": ["a", "c", "e"], "w1": ["b", "d"]}
    assert pl.split_round_robin(["a"], 4) == {"w0": ["a"]}  # empty workers dropped


def test_parser_accepts_the_documented_flags():
    args = pl._build_parser().parse_args(
        [
            "--mode",
            "das",
            "--driver",
            "python x.py --subset {subset}",
            "--corpus-json",
            "c.json",
            "--shard-root",
            "s",
            "--peak-rss-gib",
            "10",
        ]
    )
    assert args.mode == "das" and args.peak_rss_gib == 10.0
