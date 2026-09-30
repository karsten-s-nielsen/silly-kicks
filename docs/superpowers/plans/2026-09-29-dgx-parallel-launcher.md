# DGX memory-aware parallel launcher + shared-corpus study-parallelism — Implementation Plan

> **For agentic workers:** implement task-by-task with superpowers:executing-plans; steps use checkbox (`- [ ]`) tracking. Do **not** adopt a per-task-commit cadence (e.g. subagent-driven-development's default) — this feature is **ONE approval-gated commit** at Task 10 (see Global Constraints).

**Goal:** Give the DGX (and any machine) memory-safe parallelism over the existing resumable drivers, plus a shared-corpus / across-study path for F1b — all output-preserving — so idle cores are used without OOM.

**Architecture:** A driver-agnostic launcher (`scripts/_parallel_launch.py`) wraps the *existing* `--match-ids-json` resumable driver invocations against a shared `shard_root`; it sizes concurrency from measured per-workload peak-RSS (not cores), bounds each worker with a pluggable memory cap (`scripts/_mem_cap.py`: cgroup on Linux / rlimit-or-none elsewhere), pins threads (`scripts/_thread_pin.py`), applies RAM backpressure, and relaunches a worker its cap kills (shards persist → resume). Component B shares the F1b design matrix via a file-backed mmap (`scripts/_corpus_mmap.py`) and runs the ~15 studies concurrently (each serial, fixed-seed → parallel == serial byte-identical). `for_each`'s internals are untouched.

**Tech Stack:** Python 3 stdlib (`subprocess`, `resource`, `os`, `pathlib`, `mmap`/`numpy.memmap`; Linux RAM via `/proc/meminfo`), `psutil` (**optional — lazy-imported only off-Linux**, not required on the DGX), pytest. No new required runtime deps.

**Spec:** `docs/superpowers/specs/2026-09-29-dgx-parallel-launcher-design.md` — read it alongside this plan; section refs (§4 etc.) point into it.

## Global Constraints

- **ONE coherent, approval-gated commit.** No per-task commits, no micro-commits (overrides the writing-plans default). Tasks 1–9 build + test; **Task 10 makes the single commit** only after the full suite is green, ruff+pyright clean, `/final-review` run, the diff shown, and the maintainer gives explicit approval for that commit. `commit`/`push`/`PR` are separate gates. Branch: `feat/combined-provenance-dgx`.
- **Output-preserving.** Every unit runs unchanged/serial-internally; parallel results must be **byte-identical** to serial. On any byte-identity mismatch: block the parallel path, fall back to serial — a non-identical result never ships (spec §9).
- **Per-workload `peak_RSS`.** Never one shared number: size DAS runs from the DAS match-worker's measurement, F1b runs from the F1b study-worker's (spec §4/§9).
- **`N` clamped ≥ 1** — if a single worker's `peak_RSS` exceeds usable RAM, run one worker and warn; refuse only if even one cannot fit (never N=0).
- **Thread-pin var set** (aarch64 DGX numpy uses OpenBLAS): `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`=1 + `VECLIB_MAXIMUM_THREADS`=1 (macOS). One shared helper.
- **No `for_each` internal changes** (ADR-052; every corpus driver depends on that seam). The launcher wraps invocations.
- **Provenance unchanged**: launcher runs the driver from the committed SHA, clean-tree enforced by the wrapped driver, `run_commit` stamped. Launcher writes no artifact → no version bump on its own.
- **Portability**: no module may hard-import a Linux-only module at load time; backend chosen at runtime so every module imports + unit-tests on any OS.
- **Tests**: `tests/scripts/test_*.py` (repo convention); stdlib-only where possible; reuse `tests/scripts/_fake_corpus.py`. Mirror the CI invocation (`python -m pytest tests/ -m "not e2e"`).
- `_`-prefixed scripts are exempt from the driver ASCII/`ARTIFACT_DRIVERS` gates (write no artifact) but still carry an argparse CLI where runnable (never a parser-less script).

## File structure

- `scripts/_thread_pin.py` — `thread_pin_env(base=None) -> dict[str,str]` (the pin var set).
- `scripts/_mem_cap.py` — `detect_backend() -> str`; `MemoryCap` with `wrap(argv, cap_bytes) -> (argv, preexec_fn)`; backends `cgroup`/`rlimit`/`none`.
- `scripts/_parallel_launch.py` — `size_workers(...)`, `should_launch(...)`, `run_parallel(...)`, `main()` CLI.
- `scripts/_corpus_mmap.py` — `persist_design_matrix(X: pd.DataFrame, y, groups, path)`, `load_design_matrix(path) -> (X: pd.DataFrame, y, groups)` (columns/dtypes preserved; zero-copy per the Task-5 probe).
- Modify `scripts/train_xshot_occurrence.py`, `scripts/train_xcross_attempt.py` — a single `(candidate,fold)` study as a runnable unit reading the shared mmap; **+ `assemble_studies(shard_root)` reduce-only entry** (Task 6b).
- Modify `scripts/validate_das_native_parity.py` — **add `--shards-only` + a standalone `reduce_parity_artifact(shard_root, dest)`** (Task 6b).
- Tests: `tests/scripts/test_thread_pin.py`, `test_mem_cap.py`, `test_parallel_launch.py`, `test_corpus_mmap.py`, `test_train_study_parallel_parity.py`, `test_reduce_split.py`.
- **DAS is NOT a no-code-change consumer** (corrects the earlier claim): `run_corpus:737-761` reduces + writes `metrics.json` on every invocation with subset-only population, so N wrapped subset-invocations would race a subset-scoped artifact ≠ serial. It needs the Task-6b split (workers `--shards-only`; launcher calls `reduce_parity_artifact` ONCE over the full shard_root).

---

### Task 1: thread-pin helper

**Files:** Create `scripts/_thread_pin.py`; Test `tests/scripts/test_thread_pin.py`.
**Interfaces:** Produces `thread_pin_env(base: Mapping[str,str] | None = None) -> dict[str,str]` — returns `base` (default `os.environ`) copied with the four pin vars set to `"1"`.

- [ ] **Step 1: Failing test**
```python
# tests/scripts/test_thread_pin.py
from scripts._thread_pin import thread_pin_env

def test_sets_all_four_thread_vars_to_one():
    env = thread_pin_env(base={"PATH": "/x"})
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"   # aarch64 DGX numpy -> OpenBLAS
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["VECLIB_MAXIMUM_THREADS"] == "1"  # macOS Accelerate
    assert env["PATH"] == "/x"                    # base preserved, not mutated in place

def test_does_not_mutate_the_passed_base():
    base = {"PATH": "/x"}
    thread_pin_env(base=base)
    assert "OMP_NUM_THREADS" not in base
```
- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`). `python -m pytest tests/scripts/test_thread_pin.py -v`
- [ ] **Step 3: Implement**
```python
# scripts/_thread_pin.py
"""Single-source the thread-pin env so parallel workers don't oversubscribe cores.

aarch64 DGX numpy links OpenBLAS, so OPENBLAS_NUM_THREADS is load-bearing; macOS
Accelerate reads VECLIB_MAXIMUM_THREADS. train_ghost_gk.py's loky launch (OMP-only,
~line 1048) should adopt this helper in a later cleanup.
"""
from __future__ import annotations
import os
from collections.abc import Mapping

_PIN_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

def thread_pin_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for var in _PIN_VARS:
        env[var] = "1"
    return env
```
- [ ] **Step 4: Run — expect PASS.**

---

### Task 2: pluggable memory-cap backend

**Files:** Create `scripts/_mem_cap.py`; Test `tests/scripts/test_mem_cap.py`.
**Interfaces:** Produces `detect_backend(explicit: str | None = None) -> str` (`"cgroup"|"rlimit"|"none"`); `wrap(backend, argv, cap_bytes) -> tuple[list[str], object | None]` returning the launch argv and an optional `preexec_fn` (rlimit); cgroup wraps argv in `systemd-run --scope -p MemoryMax=<bytes> --`.

- [ ] **Step 1: Failing test** (argv asserted WITHOUT executing → runs on any OS)
```python
# tests/scripts/test_mem_cap.py
import sys, pytest
from scripts import _mem_cap

def test_cgroup_wrap_builds_systemd_run_scope_argv():
    argv, preexec = _mem_cap.wrap("cgroup", ["python", "x.py"], cap_bytes=14 * 1024**3)
    assert argv[:3] == ["systemd-run", "--scope", "-p"]
    assert "MemoryMax=15032385536" in " ".join(argv)  # 14 GiB
    assert argv[-3:] == ["--", "python", "x.py"]
    assert preexec is None

def test_rlimit_wrap_returns_preexec_not_argv_change():
    argv, preexec = _mem_cap.wrap("rlimit", ["python", "x.py"], cap_bytes=14 * 1024**3)
    assert argv == ["python", "x.py"]      # command unchanged
    assert callable(preexec)               # RLIMIT_AS set in the child pre-exec

def test_none_wrap_is_a_passthrough():
    argv, preexec = _mem_cap.wrap("none", ["python", "x.py"], cap_bytes=1)
    assert argv == ["python", "x.py"] and preexec is None

def test_explicit_backend_overrides_detection():
    assert _mem_cap.detect_backend(explicit="none") == "none"

def test_detect_returns_a_valid_backend_on_this_os():
    assert _mem_cap.detect_backend() in {"cgroup", "rlimit", "none"}

@pytest.mark.skipif(sys.platform != "win32", reason="rlimit preexec is POSIX-only")
def test_windows_never_selects_rlimit():
    assert _mem_cap.detect_backend() in {"none"}  # no cgroup, no os.fork/RLIMIT_AS preexec
```
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement**
```python
# scripts/_mem_cap.py
"""Per-worker memory cap, pluggable so the launcher runs anywhere (spec §5).

cgroup (Linux+systemd) is the only HARD cap and bounds the worker AND its children;
rlimit is best-effort POSIX; none relies on the launcher's backpressure. Nothing
Linux-only is imported at module load — the backend is chosen at runtime.
"""
from __future__ import annotations
import os, shutil, subprocess, sys

def detect_backend(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if sys.platform.startswith("linux") and shutil.which("systemd-run"):
        # cgroup v2 delegation check: a dry systemd-run --scope of /bin/true.
        try:
            subprocess.run(["systemd-run", "--user", "--scope", "-q", "true"],
                           check=True, capture_output=True, timeout=10)
            return "cgroup"
        except Exception:
            pass
    if hasattr(os, "fork") and hasattr(__import__("resource"), "RLIMIT_AS"):
        return "rlimit"   # POSIX best-effort (reliable-ish on Linux; weak on macOS)
    return "none"

def wrap(backend: str, argv: list[str], cap_bytes: int):
    if backend == "cgroup":
        return (["systemd-run", "--scope", "-p", f"MemoryMax={int(cap_bytes)}", "--", *argv], None)
    if backend == "rlimit":
        import resource
        def _preexec():  # runs in the child before exec
            resource.setrlimit(resource.RLIMIT_AS, (int(cap_bytes), int(cap_bytes)))
        return (list(argv), _preexec)
    return (list(argv), None)
```
- [ ] **Step 4: Run — expect PASS on the host OS** (cgroup exec-path is Linux-gated by the argv-only assertion — no systemd needed to test).

---

### Task 3: worker sizing + backpressure (pure functions)

**Files:** Create `scripts/_parallel_launch.py` (sizing + backpressure only this task); Test `tests/scripts/test_parallel_launch.py`.
**Interfaces:** Produces `size_workers(peak_rss_bytes, mem_available_bytes, nproc, headroom_bytes) -> int` (clamped ≥1, ≤nproc, `RefusalError` if even one won't fit); `should_launch(running, peak_rss_bytes, headroom_bytes) -> bool` (backpressure).

- [ ] **Step 1: Failing test**
```python
# tests/scripts/test_parallel_launch.py
import pytest
from scripts import _parallel_launch as pl

GiB = 1024**3
def test_ram_bound_below_cores():
    # 99 GiB usable / 11 GiB peak -> 9, not the 20 cores
    assert pl.size_workers(11*GiB, 119*GiB, nproc=20, headroom_bytes=20*GiB) == 9

def test_capped_at_nproc_when_ram_is_ample():
    assert pl.size_workers(1*GiB, 500*GiB, nproc=8, headroom_bytes=20*GiB) == 8

def test_clamped_to_at_least_one_when_tight():
    # one worker (11) fits under (30-20=10)? no -> still >=1 with a warning, never 0
    assert pl.size_workers(11*GiB, 30*GiB, nproc=20, headroom_bytes=20*GiB) == 1

def test_refuses_when_even_one_worker_cannot_fit():
    with pytest.raises(pl.RefusalError):
        pl.size_workers(200*GiB, 119*GiB, nproc=20, headroom_bytes=20*GiB)

def test_backpressure_blocks_when_headroom_gone():
    assert pl.should_launch(mem_available_bytes=8*GiB, peak_rss_bytes=11*GiB, margin_bytes=1*GiB) is False
    assert pl.should_launch(mem_available_bytes=13*GiB, peak_rss_bytes=11*GiB, margin_bytes=1*GiB) is True
```
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** (add to `_parallel_launch.py`)
```python
class RefusalError(RuntimeError):
    """A single worker's peak RSS exceeds usable RAM even alone — refuse rather than thrash."""

def size_workers(peak_rss_bytes, mem_available_bytes, nproc, headroom_bytes):
    usable = mem_available_bytes - headroom_bytes
    if usable < peak_rss_bytes:
        # Can even one worker fit in *total* memory (ignoring headroom)? If not, refuse.
        if mem_available_bytes < peak_rss_bytes:
            raise RefusalError(
                f"one worker needs {peak_rss_bytes} B but only {mem_available_bytes} B available")
        return 1  # tight: one worker, caller warns
    return max(1, min(nproc, usable // peak_rss_bytes))

def should_launch(mem_available_bytes, peak_rss_bytes, margin_bytes):
    return mem_available_bytes > peak_rss_bytes + margin_bytes
```
- [ ] **Step 4: Run — expect PASS.**

---

### Task 4: launcher orchestration + CLI (subset split, run, resume-relaunch)

**Files:** Modify `scripts/_parallel_launch.py` (add `run_parallel` + `main`); Test `tests/scripts/test_parallel_launch.py` (add resume test using a fake driver).
**Interfaces:** Consumes `size_workers`/`should_launch` (Task 3), `_mem_cap.wrap`/`detect_backend` (Task 2), `_thread_pin.thread_pin_env` (Task 1). Produces `run_parallel(cmd_template, subsets, cap_bytes, backend, peak_rss_bytes, headroom_bytes) -> ResultSummary` and a `main()` CLI.

- [ ] **Step 1: Failing test — resume after a simulated cap-kill.** A fake worker script exits non-zero on its first invocation for a subset, writes its "shard" on the second → the launcher must relaunch and complete.
```python
def test_relaunches_a_worker_that_exits_nonzero_until_shards_complete(tmp_path):
    # fake driver: writes a marker per item; first run of subset "b" exits 137 (OOM-like)
    driver = tmp_path / "fake_driver.py"
    driver.write_text(
        "import sys,os,json,pathlib\n"
        "shard=pathlib.Path(sys.argv[sys.argv.index('--shard-root')+1])\n"
        "ids=json.loads(pathlib.Path(sys.argv[sys.argv.index('--subset')+1]).read_text())\n"
        "for i in ids:\n"
        "    p=shard/(i+'.done')\n"
        "    if i=='b' and not (shard/'b.attempted').exists():\n"
        "        (shard/'b.attempted').write_text('1'); sys.exit(137)\n"
        "    p.write_text('ok')\n", encoding="utf-8")
    shard_root = tmp_path / "shards"; shard_root.mkdir()
    from scripts import _parallel_launch as pl
    res = pl.run_parallel(
        cmd_template=["python", str(driver), "--shard-root", str(shard_root), "--subset", "{subset}"],
        subsets={"w0": ["a", "b"], "w1": ["c"]},
        cap_bytes=1<<40, backend="none", peak_rss_bytes=1, headroom_bytes=0,
        done_marker=lambda i: shard_root / (i + ".done"),
        max_relaunch=3, shard_root=shard_root)
    assert {p.stem for p in shard_root.glob("*.done")} == {"a", "b", "c"}
    assert res.relaunched >= 1  # subset w0 was relaunched after the 137 exit
```
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement `run_parallel`** — launch one worker per subset via `_mem_cap.wrap` + `thread_pin_env`, poll `psutil.virtual_memory().available` for backpressure before each launch, wait on workers, and for any non-zero exit whose subset still has un-done items, re-hand the *remaining* (not-done) items and relaunch up to `max_relaunch`. Per-worker consecutive-failure abort. Return `ResultSummary(completed, relaunched, refused)`. (Resume is by the `done_marker` check — the real drivers use `for_each`'s `already_done`; the fake mirrors it.)
```python
import subprocess, time, json
from dataclasses import dataclass
# psutil is NOT in pyproject.toml — never import it at module top (would import-error
# Component A's tests where it is absent, silently disabling the CI guard). RAM is read
# by available_ram_bytes() below (stdlib /proc/meminfo; lazy psutil only off-Linux).
from scripts._mem_cap import wrap as _wrap
from scripts._thread_pin import thread_pin_env

@dataclass
class ResultSummary:
    completed: int; relaunched: int; refused: bool = False

def available_ram_bytes() -> int:
    """Portable available RAM. Linux: /proc/meminfo MemAvailable (stdlib). Else: lazy psutil
    if present. Else: raise, telling the caller to pass an explicit budget."""
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil  # lazy: only where /proc/meminfo is absent (macOS/Windows)
        return int(psutil.virtual_memory().available)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("cannot read available RAM (no /proc/meminfo, no psutil); "
                           "pass --nproc/--peak-rss-gib explicitly") from exc

def run_parallel(*, cmd_template, subsets, cap_bytes, backend, peak_rss_bytes, headroom_bytes,
                 done_marker, shard_root, nproc, max_relaunch=3, margin_bytes=None, reconcile=None):
    """CONCURRENT: keep up to N=size_workers(...) workers alive at once; backpressure each launch
    on available RAM; relaunch a subset with un-done items on non-zero exit; ONE reconcile at end."""
    margin = peak_rss_bytes if margin_bytes is None else margin_bytes
    n = size_workers(peak_rss_bytes, available_ram_bytes(), nproc, headroom_bytes)  # raises RefusalError
    env = thread_pin_env()
    todo = list(subsets.items())            # [(worker_id, [items]), ...]
    attempts = {w: 0 for w, _ in todo}
    running: dict = {}                        # worker_id -> Popen
    relaunched = 0

    def _remaining(items):
        return [i for i in items if not done_marker(i).exists()]

    def _launch(w, items):
        subset_file = shard_root / f"_subset_{w}.json"
        subset_file.write_text(json.dumps(items), encoding="utf-8")
        argv = [a.replace("{subset}", str(subset_file)) for a in cmd_template]
        argv, preexec = _wrap(backend, argv, cap_bytes)
        running[w] = subprocess.Popen(argv, env=env, preexec_fn=preexec)

    while todo or running:
        while todo and len(running) < n and available_ram_bytes() > peak_rss_bytes + margin:
            w, items = todo.pop(0)
            rem = _remaining(items)
            if rem:
                _launch(w, rem)
        time.sleep(2)
        for w, proc in list(running.items()):
            if proc.poll() is None:
                continue
            rc = proc.returncode
            del running[w]
            items = subsets[w]
            if _remaining(items):               # non-zero exit with work left -> relaunch
                attempts[w] += 1; relaunched += 1
                if attempts[w] > max_relaunch:
                    raise RuntimeError(f"worker {w} failed {attempts[w]}x (rc={rc})")
                todo.append((w, items))
    if reconcile is not None:
        reconcile(shard_root)                   # spec §4 Completion: ONE reconcile after all workers
    completed = sum(1 for _, items in subsets.items() for i in items if done_marker(i).exists())
    return ResultSummary(completed=completed, relaunched=relaunched)
```
- [ ] **Step 3b: Concurrency + backpressure tests — parallelism is the point, so prove it** (a sequential loop would pass the resume test; this test fails a sequential implementation).
```python
import threading, time
def test_workers_run_concurrently(tmp_path, monkeypatch):
    from scripts import _parallel_launch as pl
    root = tmp_path / "s"; root.mkdir()
    driver = tmp_path / "slow.py"
    driver.write_text(
        "import sys,json,pathlib,time\n"
        "root=pathlib.Path(sys.argv[sys.argv.index('--shard-root')+1])\n"
        "ids=json.loads(pathlib.Path(sys.argv[sys.argv.index('--subset')+1]).read_text())\n"
        "for i in ids:(root/(i+'.started')).write_text('1')\n"
        "time.sleep(3)\n"
        "for i in ids:(root/(i+'.done')).write_text('1')\n", encoding="utf-8")
    monkeypatch.setattr(pl, "available_ram_bytes", lambda: 10**12)
    peak = [0]
    def watch():
        for _ in range(40):
            live = len(list(root.glob('*.started'))) - len(list(root.glob('*.done')))
            peak[0] = max(peak[0], live); time.sleep(0.2)
    t = threading.Thread(target=watch); t.start()
    pl.run_parallel(cmd_template=["python", str(driver), "--shard-root", str(root), "--subset", "{subset}"],
        subsets={"w0": ["a"], "w1": ["b"], "w2": ["c"]}, cap_bytes=1 << 40, backend="none",
        peak_rss_bytes=1, headroom_bytes=0, done_marker=lambda i: root / (i + ".done"),
        shard_root=root, nproc=3)
    t.join(); assert peak[0] >= 2   # >=2 workers alive at once -> real parallelism
```
Plus a backpressure test: monkeypatch `available_ram_bytes` to a low value -> assert only one worker launches until it "frees".
- [ ] **Step 4: Add `main()`** — argparse: `--driver` (module + args template), `--corpus-json`/`--providers`, `--shard-root`, `--peak-rss-gib`, `--headroom-gib`, `--nproc`, `--mem-backend`, `--cap-gib`. Splits the corpus into `size_workers(...)` subsets (round-robin by key), prints the detected backend (preflight); calls `run_parallel(..., reconcile=<a 1-arg `Callable[[Path], None]` closure bound here over the run's refs/dest/prov — Task 6b: `reduce_parity_artifact` for DAS / `assemble_studies` for F1b>)` — **ONE** reduce over the full `shard_root` after all workers finish (spec §4 Completion). Workers run `--shards-only` and must NOT reduce, else concurrent subset-scoped reduces over one root race (DPL-PLAN-03). Catches `RefusalError` -> prints the reason + `raise SystemExit(2)`. Never a parser-less script.
- [ ] **Step 5: Run — expect PASS** (resume + a concurrency test).

---

### Task 5: shared design-matrix — column/dtype-preserving, zero-copy-probed (5a)

The F1b study path consumes a **pandas DataFrame with named feature columns** (`X: pd.DataFrame`; `X.iloc[...]`; feature-names -> the booster's `feature_names`; `_xshot_occurrence_objective.py:47,85,91,92`). So the shared corpus MUST preserve columns + dtypes, be consumed **without a copy** (a copy defeats 5a's cross-process RAM sharing), and yield a **byte-identical booster** (same values + same `feature_names`). A bare numpy memmap has no `.iloc` and no column names — it would AttributeError and train a column-less, non-identical model. Hence the probe (fixes DPL-PLAN-01).

**Files:** Create `scripts/_corpus_mmap.py`; Test `tests/scripts/test_corpus_mmap.py`.
**Interfaces:** `persist_design_matrix(X: pd.DataFrame, y, groups, path)` — values as a memmap-able `.npy` + columns/dtypes as a `columns.json` sidecar; `load_design_matrix(path) -> (X: pd.DataFrame, y, groups)` — a DataFrame over the read-only memmap (mechanism per Step 0) with the original columns + dtypes.

- [ ] **Step 0 — PROBE (decides the mechanism; per the equivalence discipline, probe a platform-dependent identity before building on it).** A throwaway ~30-line script finds which mechanism is BOTH zero-copy AND column/dtype-preserving AND booster-byte-identical on this pandas/numpy:
  1. `pd.DataFrame(np.load(x, mmap_mode="r"), columns=names, copy=False)` — verify `X.values.base is` the memmap (no copy) AND a small xgboost booster trained on it equals one trained on the original DataFrame (`save_raw("json")` byte-equal; same `feature_names`).
  2. Arrow-memmap (`pyarrow.memory_map` + feather) `to_pandas(zero_copy_only=True)` — only if pyarrow is already a dep.
  3. **Fallback (accept a copy):** reconstruct a plain per-worker DataFrame. Then 5a's win is **avoiding the re-BUILD** (feature extraction from tracking, the expensive part) — NOT cross-process RAM sharing; **record the measured per-worker matrix cost and revise the launcher's RAM budget** (still a real win, just less than mmap-sharing).
  **Probe against the REAL design-matrix dtype profile, not uniform float32** (DPL-PLAN-08): if `X` is mixed-dtype, a single `.npy` forces per-column casts = a copy that silently defeats zero-copy — then persist per-dtype blocks (a `.npy` per dtype group) or take the copy-fallback. `assert_frame_equal(check_exact=True)` guards the dtypes. Record the selected mechanism + numbers in the PR notes; `load_design_matrix` implements it.
- [ ] **Step 1: Failing test — columns + dtypes preserved, byte-identical.**
```python
# tests/scripts/test_corpus_mmap.py
import numpy as np, pandas as pd
from scripts._corpus_mmap import persist_design_matrix, load_design_matrix

def test_roundtrip_preserves_columns_dtypes_and_bytes(tmp_path):
    X = pd.DataFrame(np.random.default_rng(0).standard_normal((500, 3)).astype(np.float32),
                     columns=["a_speed", "b_angle", "c_dist"])
    y = (X["a_speed"] > 0).to_numpy().astype(np.int8); groups = np.arange(500) % 5
    p = tmp_path / "dm"; persist_design_matrix(X, y, groups, p)
    Xr, yr, gr = load_design_matrix(p)
    pd.testing.assert_frame_equal(Xr, X, check_exact=True)   # names + dtypes + bytes
    assert np.array_equal(yr, y) and np.array_equal(gr, groups)
    # For a zero-copy mechanism (probe 1/2), also assert no full copy occurred (probe-validated).
```
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** the probe-selected `load_design_matrix` + `persist_design_matrix` (values `.npy` loaded `mmap_mode="r"`; a `columns.json` sidecar of names + dtype strings; reconstruct the DataFrame per the chosen mechanism; `y`/`groups` are small, plain load).
- [ ] **Step 4: Run — expect PASS.**

---

### Task 6: trainer study-as-unit + shared consumer (5c) + BOOSTER-BYTE parity gate

**Files:** Modify `scripts/train_xshot_occurrence.py` + `scripts/train_xcross_attempt.py` (a `--study <name> --design-matrix <path>` unit reading the shared corpus); Test `tests/scripts/test_train_study_parallel_parity.py`.
**Interfaces:** Consumes `load_design_matrix` (Task 5, returns a **DataFrame**). Produces `run_one_study(design_matrix_path, candidate, fold, seed, out_dir)` yielding the same result as the in-process serial study, and a test seam `_fit_study_for_test(X, y, groups, tag, n_trials, seed) -> (params, booster)`.

- [ ] **Step 1: Failing test — REAL DataFrame column path, assert BOOSTER BYTES** (params-only would false-green a column-less model; fixes DPL-PLAN-01/07).
```python
# tests/scripts/test_train_study_parallel_parity.py  (slow; small synthetic corpus)
import numpy as np, pandas as pd, pytest
from scripts import train_xshot_occurrence as tr
from scripts._corpus_mmap import persist_design_matrix, load_design_matrix

@pytest.mark.slow
def test_study_from_shared_corpus_matches_in_memory_booster_bytes(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((400, 3)).astype(np.float32), columns=["a", "b", "c"])
    y = (X["a"] > 0).to_numpy().astype(np.int8); groups = np.arange(400) % 6
    s_params, s_booster = tr._fit_study_for_test(X, y, groups, "full_f0", n_trials=5, seed=42)
    p = tmp_path / "dm"; persist_design_matrix(X, y, groups, p)
    Xs, ys, gs = load_design_matrix(p)
    p_params, p_booster = tr._fit_study_for_test(Xs, ys, gs, "full_f0", n_trials=5, seed=42)
    assert s_params == p_params
    assert s_booster.save_raw("json") == p_booster.save_raw("json")   # BYTE-identical model
    assert s_booster.feature_names == p_booster.feature_names          # names preserved
```
- [ ] **Step 2: Run — expect FAIL** (no `_fit_study_for_test`/`run_one_study` yet; a bare array would AttributeError on `.iloc`).
- [ ] **Step 3: Implement** — `_fit_study_for_test` (returns params + the fitted booster; test-only seam); `run_one_study(design_matrix_path, candidate, fold, seed, out_dir)` = `load_design_matrix` -> one study; a `--study`/`--design-matrix` CLI branch (the launcher's unit). The real risk is the `.iloc`/columns **contract**, not only in-place writes: confirm the study path accepts the shared DataFrame, preserves feature-names into the booster, and mutates nothing in place (grep the touched fns for `.iloc[...] =` / `X[...] =` **and** that no step relies on a column-less array). Keep the full-run path intact.
- [ ] **Step 4: Run — expect PASS.** If the probe (Task 5) chose the copy-fallback, this parity still holds (same DataFrame, copied); only the Task-5 RAM-budget note differs.

---

### Task 6b: driver reduce/assemble split — ONE combined artifact after the parallel units (fixes DPL-PLAN-03)

Both drivers entangle per-unit work with the reduce that writes the combined artifact, so N parallel subset/study invocations would each race a **subset-scoped** artifact ≠ serial. Split both: workers write shards only; a reduce-only entry the launcher calls **once** assembles the full population. This makes Task 4's `reconcile=` callback concrete.

**Files:** Modify `scripts/validate_das_native_parity.py`, `scripts/train_xshot_occurrence.py`, `scripts/train_xcross_attempt.py`; Test `tests/scripts/test_reduce_split.py`.
**Interfaces:** Produces `validate_das_native_parity.reduce_parity_artifact(shard_root, dest)` + `run_corpus(..., shards_only: bool=False)`; `train_*.assemble_studies(shard_root) -> (verdict, weights)`. Consumed by the launcher's `reconcile=` (Task 4) and by `run_one_study` (Task 6, which writes a per-study shard).

- [ ] **Step 1: DAS failing test — reduce-only == serial `metrics.json`, byte-identical, with an EXCLUDED match across MULTIPLE per-worker manifests** (real `run_corpus` sig `run_corpus(refs, load, dest, *, prov, shard_root=…)`; `metrics.json` population/excluded come from refs + the aggregated manifests, NOT the shards — DPL-PLAN-10/12).
```python
# tests/scripts/test_reduce_split.py
import json
from scripts import validate_das_native_parity as das
# _fake_refs / _fake_loader from tests/scripts/_fake_corpus: loader raises ItemExcluded for an excluded ref
def test_das_reduce_only_equals_serial_with_excluded_and_two_manifests(tmp_path):
    refs = _fake_refs(scoreable=3, excluded=1)          # 1 ref -> ItemExcluded (SB360-like)
    load = _fake_loader; prov = {"commit": "test", "dirty": False}
    par = tmp_path / "par"
    das.run_corpus(refs[:2], load, par, prov=prov, shard_root=par/"shards", shards_only=True)  # worker A -> manifest_A
    das.run_corpus(refs[2:], load, par, prov=prov, shard_root=par/"shards", shards_only=True)  # worker B -> manifest_B
    assert not (par / "metrics.json").exists()                                                  # workers write NO artifact
    das.reduce_parity_artifact(refs, par/"shards", par/"metrics.json", prov=prov)               # ONE reduce: all refs + all manifests
    ser = tmp_path / "ser"
    das.run_corpus(refs, load, ser, prov=prov, shard_root=ser/"shards")                          # serial reference
    assert json.loads((par/"metrics.json").read_text()) == json.loads((ser/"metrics.json").read_text())
```
- [ ] **Step 2: DAS implement** — add `shards_only: bool=False` to `run_corpus`: when True, run `for_each` (writes shards + this worker's `manifest_<tag>.json`) and RETURN, skipping reduce+`_population`+write (`:737-753`). Extract the reduce into `reduce_parity_artifact(refs, shard_root, dest, *, prov)`: `reduce_parity(sorted glob *.parquet)` for the parity VALUES; **`manifest = _partition.aggregate_manifests(shard_root)`** to SUM every per-worker `manifest_<tag>.json` (n_attempted/n_failed/n_excluded) — never re-derived from shards, which cannot see `.excluded.json` markers or SB360's structural exclusion (`:766-771`); `_population(refs, scored, manifest)` with `listed_per_provider` from the FULL `refs` (`:761`); write `metrics.json` exactly as `run_corpus`. The serial `run_corpus` calls `reduce_parity_artifact` too — one reduce implementation, single-sourced. Add `--shards-only` + `--reduce-only` CLI flags. **Persistence detail (r4 heads-up):** each `--shards-only` worker must WRITE a **uniquely-tagged** `manifest_<tag>.json` into the generation dir (`shard_root/<token>/`) — `run_corpus` today keeps `res.manifest()` in-memory and writes none, and a shared/duplicate tag would collide; `reduce_parity_artifact` reads shards + `manifest_*.json` from that same generation dir. The Step-1 byte-identity test pins both. Run green.
- [ ] **Step 3: F1b failing test — assemble over all study shards == serial paired verdict + weights, byte-identical.** `run_one_study` (Task 6) writes a per-study result shard; `assemble_studies(shard_root)` runs the post-study paired comparison + ship decision + weight write (the tail of `_paired_nested`) over ALL study shards. Assert its verdict + `booster.save_raw("json")` == the serial `_paired_nested`.
- [ ] **Step 4: F1b implement** — extract the post-study-loop assembly of `_paired_nested` into `assemble_studies(shard_root) -> (verdict, weights)`; the serial full-run calls it too (single-sourced). Run green.
- [ ] **Step 5: Wire (fixes the arity, DPL-PLAN-11).** `run_parallel`'s `reconcile: Callable[[Path], None]` is called once as `reconcile(shard_root)`. `main()` binds a 1-arg closure over the run's own refs/dest/prov: DAS `reconcile = lambda root: reduce_parity_artifact(refs, root, dest, prov=prov)`; F1b `reconcile = lambda root: assemble_studies(root)` (writes weights+verdict to disk; its return tuple is for the Step-3 test only).

---

### Task 7 (owner-run, DGX): measure per-workload peak-RSS

Not committed code — a measurement that sets the cap (spec §9).
- [ ] Capture the high-watermark RSS (e.g. `/usr/bin/time -v` or a `psutil` sampler) of **one DAS match-worker** (incl. any child) and **one F1b study-worker over an `sc_extended` study**. Record both in the PR description. These set `--peak-rss-gib` per run. Do **not** run against the in-flight F1b job.

### Task 8 (owner-run, DGX): validate the launcher on DAS (the test case)
- [ ] Run through `_parallel_launch.py` — workers `--shards-only`, then the ONE `reduce_parity_artifact` (Task 6b) — on a **small match subset**: confirm no OOM under load, a speedup, and shards **identical** to a serial subset run.
- [ ] Run the full corpus; **assert the parity `metrics.json` is byte-identical to a serial `run_corpus`** (full population, each match deterministic). If it differs — stop, do not ship (spec §9).

### Task 9 (owner-run): F1b parallel==serial parity on a small real set
- [ ] On a small `(candidate × fold)` set (not the in-flight run), confirm parallel-study weights == serial-study weights byte-identical, and shared-mmap `X,y` == per-process-built `X,y`. Mismatch → block parallel, fall back to serial.

---

### Task 10: pre-commit gate + single approval-gated commit

- [ ] **Full suite green** — `python -m pytest tests/ -m "not e2e" -v --tb=short` (+ the `slow` job for the parity test).
- [ ] **Lint/type** — `python -m ruff check silly_kicks/ tests/ scripts/` + `--format --check`; `pyright` on the changed files. (`scripts/_*` new files: confirm the driver ASCII/`ARTIFACT_DRIVERS` gates still pass — `_`-prefix exempts them; the launcher writes no artifact.)
- [ ] **`/final-review`** — regenerates `architecture.html` via Graphviz `dot`. New `scripts/_*` helpers are not C4 containers, so expect no diff; if it diffs, investigate before committing.
- [ ] **Show the diff + file list** to the maintainer. Confirm only the intended files (the 4 new `scripts/_*`, the 2 trainer edits, the 5 new tests, and the spec+plan docs) are staged; `.serena/` not staged.
- [ ] **Commit only on explicit maintainer approval** — ONE coherent commit for the whole feature on `feat/combined-provenance-dgx`. Message: `feat(scripts): memory-aware parallel launcher + shared-corpus study-parallel (DGX)` + the `Co-Authored-By` trailer.
- [ ] **Push / PR** — separate explicit gates.

---

## Self-Review

**Spec coverage (§ → task):**
- §4 launcher (sizing, backpressure, subset, resume, thread-pin) → Tasks 3, 4 (+ Task 1 pin). ✓
- §5 pluggable memory-cap backend (cgroup/rlimit/none, runtime detect, no Linux hard-import) → Task 2. ✓
- §6 shared mmap corpus (5a) + F1b across-study (5c) → Tasks 5, 6. ✓
- §7 data flow (DAS / F1b) → **Task 6b** (the reduce/assemble producing the ONE combined artifact) + Tasks 8, 9 (owner-run validation). ✓
- §8 OOM safety (per-worker cap, backpressure, resume, per-worker failure abort, thread-pin) → Tasks 2,3,4,1. ✓
- §9 testing (launcher units CI; DAS byte-identical; F1b byte-identical; per-workload peak-RSS; mismatch→serial) → Tasks 2-6 (units) + 7,8,9 (owner-run) + the fall-back-to-serial in Tasks 8/9. ✓
- §10 portability (units run any OS; cgroup exec Linux-gated) → Task 2 tests (argv-only + skipif). ✓
- §11 provenance / single commit / `_`-prefix gates → Global Constraints + Task 10. ✓
- §3 items 1–4 + 5a + 5c → Tasks 1–6. 5b explicitly absent (non-goal). ✓

**Placeholder scan:** every code step carries real test + impl. Task 4 is a concrete concurrent Popen pool (not a sequential loop + a note) with a concurrency test that fails a sequential impl; Task 5 Step 0 is a real probe that selects the sharing mechanism (with a copy-fallback + revised RAM math), not a TBD. No "add error handling"/"similar to Task N".

**Type consistency:** `thread_pin_env`; `detect_backend`/`wrap`; `size_workers`/`should_launch`/`available_ram_bytes`/`run_parallel(...,nproc,reconcile)`/`ResultSummary`/`RefusalError`; `persist_design_matrix`/`load_design_matrix` (DataFrame); `_fit_study_for_test`/`run_one_study` — names + signatures match across tasks and tests.

**Round-1 review resolutions** (report `D:\Development\_reviews\2026-09-29-dgx-parallel-launcher-plan.md`):
- **DPL-PLAN-01 (BLOCKING) fixed** — Tasks 5/6 now use a column/dtype-preserving DataFrame (probe-selected zero-copy or copy-fallback) and assert **booster bytes**, not params only.
- **DPL-PLAN-02 fixed** — no top-level `psutil`; `available_ram_bytes()` reads stdlib `/proc/meminfo` (lazy psutil only off-Linux).
- **DPL-PLAN-03 (r2: was PARTIAL — the drivers had NO separable reduce) now FIXED via Task 6b** — DAS `--shards-only` + standalone `reduce_parity_artifact`; F1b per-study shard + `assemble_studies`; the launcher `reconcile=` calls the reduce-only entry ONCE (byte-identity tested vs serial). Corrects "DAS needs no code change".
- **DPL-PLAN-08 (r2)** — Task 5 probe uses the real design-matrix dtype profile (mixed-dtype → per-block `.npy` or copy-fallback; `assert_frame_equal(check_exact=True)` guards it).
- **DPL-PLAN-09 (r2)** — Tech-Stack: psutil marked optional / lazy-off-Linux, not "already a dep".
- **DPL-PLAN-10 (r3, SHOULD FIX) fixed** — `reduce_parity_artifact` no longer "globs shards for population": it aggregates the per-worker manifests (`_partition.aggregate_manifests`) for n_attempted/n_failed/n_excluded and takes `listed` from the FULL refs (matching serial `_population`); the Step-1 test adds an EXCLUDED match + two per-worker manifests so under-reporting is caught.
- **DPL-PLAN-11 (r3) fixed** — `run_parallel`'s `reconcile` is a 1-arg `Callable[[Path], None]`; `main()` binds a closure over refs/dest/prov (DAS) or calls `assemble_studies` (F1b, disk side-effect).
- **DPL-PLAN-12 (r3) fixed** — the Task-6b test uses the real `run_corpus(refs, load, dest, *, prov, shard_root=…)` signature.
- **DPL-PLAN-04 fixed** — `run_parallel` is a concrete concurrent Popen pool + a concurrency test (Step 3b).
- **DPL-PLAN-05 fixed** — `ResultSummary.refused` defaulted; `main()` catches `RefusalError` -> `SystemExit(2)`.
- **DPL-PLAN-06 fixed** — header no longer recommends a per-task-commit cadence.
- **DPL-PLAN-07 fixed** — Task 6 verification targets the `.iloc`/columns contract, not only in-place writes.

**Commit discipline:** no per-task commit anywhere; single approval-gated commit at Task 10 (overrides the writing-plans default), matching the spec §11 + repo rule.
