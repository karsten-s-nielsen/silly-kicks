"""Memory-aware parallel launcher over the existing resumable corpus drivers (spec 4).

The DGX (and any host) has idle cores while a single serial driver runs, but naively
fanning out N workers has driven the box into the OOM killer, which silently kills
sibling workers. This launcher sizes concurrency from measured per-workload peak RSS
(not core count), bounds each worker with a pluggable memory cap (``_mem_cap``), pins
BLAS threads (``_thread_pin``), applies RAM backpressure before each launch, and
relaunches a worker its cap kills (shards persist, so the relaunch resumes). It wraps
the *existing* ``--match-ids-json`` / ``--subset`` driver invocations against a shared
``shard_root``; ``for_each``'s internals are untouched (ADR-052).

This module is import-clean on every OS: no Linux-only module is imported at load time,
and ``psutil`` is only lazy-imported off-Linux where ``/proc/meminfo`` is absent.
"""

from __future__ import annotations


class RefusalError(RuntimeError):
    """A single worker's peak RSS exceeds usable RAM even alone -- refuse rather than thrash."""


def size_workers(
    peak_rss_bytes: int,
    mem_available_bytes: int,
    nproc: int,
    headroom_bytes: int,
) -> int:
    """Number of concurrent workers that fit, clamped to ``[1, nproc]``.

    Sizes from measured peak RSS, reserving ``headroom_bytes`` for the OS and shared
    pages. Returns 1 (with the caller warning) when a single worker fits in total RAM
    but not under the headroom; raises ``RefusalError`` only when even one worker
    cannot fit in total available RAM.
    """
    usable = mem_available_bytes - headroom_bytes
    if usable < peak_rss_bytes:
        if mem_available_bytes < peak_rss_bytes:
            raise RefusalError(f"one worker needs {peak_rss_bytes} B but only {mem_available_bytes} B available")
        return 1  # tight: one worker; the caller warns
    return max(1, min(nproc, usable // peak_rss_bytes))


def should_launch(mem_available_bytes: int, peak_rss_bytes: int, margin_bytes: int) -> bool:
    """Backpressure gate: only launch another worker if its peak RSS plus a margin still fits."""
    return mem_available_bytes > peak_rss_bytes + margin_bytes


# --------------------------------------------------------------------------------------
# Orchestration (Task 4): concurrent Popen pool + resume-relaunch + one reconcile.
# --------------------------------------------------------------------------------------
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Mapping, Sequence  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

from scripts._mem_cap import detect_backend as _detect_backend  # noqa: E402
from scripts._mem_cap import wrap as _wrap  # noqa: E402
from scripts._thread_pin import thread_pin_env  # noqa: E402


@dataclass
class ResultSummary:
    completed: int
    relaunched: int
    refused: bool = False


def available_ram_bytes() -> int:
    """Portable available-RAM reading.

    Linux reads ``/proc/meminfo`` ``MemAvailable`` (stdlib, no dep). Elsewhere it
    lazy-imports ``psutil`` (only where ``/proc/meminfo`` is absent). If neither is
    available it raises, telling the caller to pass an explicit budget -- it never
    guesses, so a wrong reading can't silently oversubscribe the box.
    """
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
    except Exception as exc:
        raise RuntimeError(
            "cannot read available RAM (no /proc/meminfo, no psutil); pass an explicit --nproc / --peak-rss-gib"
        ) from exc


def run_parallel(
    *,
    cmd_template: Sequence[str],
    subsets: Mapping[str, list[str]],
    cap_bytes: int,
    backend: str,
    peak_rss_bytes: int,
    headroom_bytes: int,
    done_marker: Callable[[str], Path],
    shard_root: Path,
    nproc: int | None = None,
    max_relaunch: int = 3,
    margin_bytes: int | None = None,
    reconcile: Callable[[Path], None] | None = None,
    poll_interval: float = 1.0,
    mem_available_bytes: int | None = None,
) -> ResultSummary:
    """Keep up to ``N = size_workers(...)`` workers alive at once over ``subsets``.

    Each subset is one worker: the launcher writes its item list to a JSON file,
    substitutes it into ``cmd_template`` (the ``{subset}`` token), wraps the argv with
    the memory cap, pins threads, and launches. Before each *additional* launch it
    checks RAM backpressure; the sole/first worker always launches (waiting frees
    nothing, and sizing already proved one fits). A worker that exits non-zero with
    un-done items is relaunched with only the remaining items (shards persist ->
    resume), up to ``max_relaunch``. ``reconcile`` runs ONCE after every worker
    finishes (spec 4 Completion) -- workers themselves must not write the combined
    artifact.

    RAM is read via ``available_ram_bytes()`` (Linux ``/proc/meminfo`` + dynamic
    backpressure). ``mem_available_bytes``, when given, is a FIXED budget used instead --
    the escape hatch for a host with neither ``/proc/meminfo`` nor ``psutil`` (backpressure
    then only reflects that static budget, not live free RAM).
    """
    resolved_nproc: int = nproc if nproc is not None else (os.cpu_count() or 1)
    margin = peak_rss_bytes if margin_bytes is None else margin_bytes

    def _ram() -> int:
        return available_ram_bytes() if mem_available_bytes is None else mem_available_bytes

    n = size_workers(peak_rss_bytes, _ram(), resolved_nproc, headroom_bytes)  # raises RefusalError
    env = thread_pin_env()
    shard_root = Path(shard_root)

    todo: list[tuple[str, list[str]]] = [(w, list(items)) for w, items in subsets.items()]
    attempts: dict[str, int] = {w: 0 for w, _ in todo}
    running: dict[str, subprocess.Popen] = {}
    relaunched = 0

    def _remaining(items: Sequence[str]) -> list[str]:
        return [i for i in items if not done_marker(i).exists()]

    def _launch(w: str, items: list[str]) -> None:
        subset_file = shard_root / f"_subset_{w}.json"
        subset_file.write_text(json.dumps(items), encoding="utf-8")
        argv = [a.replace("{subset}", str(subset_file)) for a in cmd_template]
        argv, preexec = _wrap(backend, argv, cap_bytes=cap_bytes)
        # argv is the operator's own --driver template + an internal subset file, not untrusted input.
        running[w] = subprocess.Popen(argv, env=env, preexec_fn=preexec)  # noqa: S603  (preexec None off-POSIX)

    while todo or running:
        while todo and len(running) < n and (not running or should_launch(_ram(), peak_rss_bytes, margin)):
            w, items = todo.pop(0)
            rem = _remaining(items)
            if rem:
                _launch(w, rem)
        time.sleep(poll_interval)
        for w, proc in list(running.items()):
            if proc.poll() is None:
                continue
            rc = proc.returncode
            del running[w]
            items = list(subsets[w])
            # Relaunch ONLY a FAILED worker that still has un-done items (the driver resumes from its
            # shards). A clean exit (rc == 0) is never relaunched -- gating on `_remaining` alone would
            # spin forever whenever the done-marker is imprecise for a driver.
            if rc != 0 and _remaining(items):
                attempts[w] += 1
                relaunched += 1
                if attempts[w] > max_relaunch:
                    raise RuntimeError(f"worker {w} failed {attempts[w]}x (rc={rc})")
                todo.append((w, items))
            elif rc != 0:
                raise RuntimeError(f"worker {w} exited {rc} with no un-done items left to retry")

    if reconcile is not None:
        reconcile(shard_root)  # ONE reduce over the full shard_root after all workers finish
    completed = sum(1 for items in subsets.values() for i in items if done_marker(i).exists())
    return ResultSummary(completed=completed, relaunched=relaunched)


def split_round_robin(keys: Sequence[str], n: int) -> dict[str, list[str]]:
    """Deal ``keys`` round-robin into ``n`` worker subsets ``w0..w{n-1}`` (empty ones dropped)."""
    n = max(1, n)
    buckets: dict[str, list[str]] = {f"w{i}": [] for i in range(n)}
    for idx, key in enumerate(keys):
        buckets[f"w{idx % n}"].append(key)
    return {w: items for w, items in buckets.items() if items}


def _build_parser():
    import argparse

    p = argparse.ArgumentParser(description="Memory-aware parallel launcher over the resumable corpus drivers.")
    p.add_argument(
        "--mode",
        choices=("das", "f1b"),
        required=True,
        help="selects how a done item is detected: das = a <key>.parquet shard, f1b = a <tag>.study.json study shard",
    )
    p.add_argument(
        "--driver", required=True, help="worker command template; use {subset} for the per-worker JSON id list"
    )
    p.add_argument(
        "--corpus-json", required=True, type=Path, help="JSON list of corpus item keys to split across workers"
    )
    p.add_argument("--shard-root", required=True, type=Path)
    p.add_argument(
        "--reduce",
        help="command run ONCE after all workers finish (the driver's own --reduce-only / "
        "--assemble CLI); use {shard_root} for the shard root. Omit to skip the reduce.",
    )
    p.add_argument(
        "--peak-rss-gib", required=True, type=float, help="measured per-worker peak RSS (spec 9) -- sizes concurrency"
    )
    p.add_argument("--headroom-gib", default=20.0, type=float)
    p.add_argument("--cap-gib", type=float, help="per-worker hard cap; defaults to peak*1.3")
    p.add_argument("--nproc", type=int, default=None)
    p.add_argument("--margin-gib", default=None, type=float)
    p.add_argument(
        "--mem-available-gib",
        default=None,
        type=float,
        help="fixed available-RAM budget instead of reading it live -- REQUIRED on a host with neither "
        "/proc/meminfo (Linux) nor psutil (backpressure then reflects only this static budget).",
    )
    p.add_argument(
        "--mem-backend",
        default=None,
        choices=("cgroup", "rlimit", "none"),
        help="override memory-cap backend detection",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    gib = 1024**3
    args = _build_parser().parse_args(argv)
    keys = json.loads(Path(args.corpus_json).read_text(encoding="utf-8"))
    shard_root = Path(args.shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    peak = int(args.peak_rss_gib * gib)
    headroom = int(args.headroom_gib * gib)
    cap = int((args.cap_gib if args.cap_gib is not None else args.peak_rss_gib * 1.3) * gib)
    margin = None if args.margin_gib is None else int(args.margin_gib * gib)
    backend = _detect_backend(explicit=args.mem_backend)
    nproc: int = args.nproc if args.nproc is not None else (os.cpu_count() or 1)
    mem_budget = None if args.mem_available_gib is None else int(args.mem_available_gib * gib)

    try:
        n = size_workers(peak, mem_budget if mem_budget is not None else available_ram_bytes(), nproc, headroom)
    except RefusalError as exc:
        print(f"REFUSED: {exc}")
        return 2
    subsets = split_round_robin(keys, n)
    print(f"backend={backend} workers={len(subsets)} (n<= {n}) items={len(keys)}")

    # The reduce is the driver's OWN --reduce-only / --assemble CLI, run once as a subprocess -- the
    # launcher never imports driver internals or re-runs the corpus loader (it just orchestrates the
    # resumable drivers, per ADR-052).
    reconcile: Callable[[Path], None] | None
    if args.reduce:
        reduce_cmd = _driver_template(args.reduce)

        def _reconcile(root: Path) -> None:
            cmd = [a.replace("{shard_root}", str(root)) for a in reduce_cmd]
            # cmd is the operator's own --reduce template, not untrusted input.
            subprocess.run(cmd, check=True, env=thread_pin_env())  # noqa: S603

        reconcile = _reconcile
    else:
        reconcile = None

    done_marker = _done_marker_for(args.mode, shard_root)
    res = run_parallel(
        cmd_template=_driver_template(args.driver),
        subsets=subsets,
        cap_bytes=cap,
        backend=backend,
        peak_rss_bytes=peak,
        headroom_bytes=headroom,
        done_marker=done_marker,
        shard_root=shard_root,
        nproc=nproc,
        margin_bytes=margin,
        reconcile=reconcile,
        mem_available_bytes=mem_budget,
    )
    print(f"completed={res.completed} relaunched={res.relaunched}")
    return 0


def _driver_template(driver: str) -> list[str]:
    """Split a command template into argv tokens (shell-free, cross-platform)."""
    import shlex

    return shlex.split(driver, posix=(__import__("os").name != "nt"))


def _done_marker_for(mode: str, shard_root: Path) -> Callable[[str], Path]:
    """Map a corpus key to the path whose existence means 'done' for that driver.

    f1b: the study shard ``<tag>.study.json`` (exact). das: the match shard ``<key>.parquet`` under the
    generation dir -- returned via a sentinel-or-hit so ``run_parallel`` can test ``.exists()``; a hit is
    only an optimisation because relaunch is failure-gated (a clean worker is never relaunched).
    """
    if mode == "f1b":
        return lambda i: shard_root / f"{i}.study.json"

    _missing = shard_root / "__never__"

    def _das(i: str) -> Path:
        hits = list(shard_root.rglob(f"{i}.parquet"))
        return hits[0] if hits else _missing

    return _das


if __name__ == "__main__":
    raise SystemExit(main())
