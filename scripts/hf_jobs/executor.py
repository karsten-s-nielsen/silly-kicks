"""Place one stage's units as W memory-capped worker processes (ADR-128, spec 5.2).

Reuses the DGX launcher model (2026-09-29 spec) rather than a `ProcessPoolExecutor`: a pool worker killed
by the OOM killer breaks the whole pool (`BrokenProcessPool`), while a capped SUBPROCESS dies alone and
its relaunch resumes from the persisted units (INV-3). Workers are capped by `_mem_cap` (a cgroup on the
DGX; inside an HF container, which has no systemd, the executor itself watches the largest single RSS in
each worker's process tree and kills one over its cap -- the ``rss`` backend; ``RLIMIT_AS`` capped
address space, not RSS, and failed every unit on real HF), started with `_thread_pin.deterministic_env`
(INV-2), sized by
`_parallel_launch.size_workers` over the cgroup's CPUs and RAM (a container's `os.cpu_count()` and
`/proc/meminfo` are the host's) and gated by `should_launch` backpressure.

Workers take keys from a JOB-LOCAL claim queue (O_EXCL claim files on local disk, never on the bucket
mount), so one slow unit never stalls a static partition. Across jobs the partition is static
(`_driver.shard_select`), so two jobs never run one unit.

A laptop, the DGX and an HF job run this same code (ruling 11): an HF job is only
`python -m <module> --stages S --shard I/N --workers W` inside the pinned image.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import multiprocessing
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from typing import Any

GiB = 1024**3
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
#: Fallback RAM budget where neither /proc/meminfo nor psutil exists (a Windows dev box): the cpu-upgrade size.
_FALLBACK_RAM = 32 * GiB


class StageFailedError(RuntimeError):
    """A stage exceeded its failure tolerance, or a worker kept dying."""


@dataclasses.dataclass(frozen=True)
class ExecutorConfig:
    workers: int = 1
    inner_workers: int = 1
    cap_bytes: int | None = None
    peak_rss_bytes: int | None = None
    headroom_bytes: int = 2 * GiB
    max_relaunch: int = 3
    poll_interval_s: float = 0.2
    profile_units: int | None = None
    mem_backend: str | None = None
    #: Also run units DEFERRED by a memory failure (the controller's W=1, whole-box escalation job).
    escalate: bool = False


@dataclasses.dataclass(frozen=True)
class StageReport:
    stage: str
    n_mine: int
    n_done: int
    n_excluded: int
    n_failed: int
    workers: int
    relaunched: int
    completion_order: tuple[str, ...]


def _inner_init(cap_bytes: int | None, backend: str) -> None:  # pragma: no cover - runs in pool children
    if backend == "rlimit" and cap_bytes:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (int(cap_bytes), int(cap_bytes)))  # type: ignore[attr-defined]


class InnerPool:
    """The pool an ``inner_parallel`` unit maps its inner keys over (spec 5.2: no pool inside a pool).

    Spawn context; children inherit the worker's deterministic environment, so each is single-threaded.
    ``map_keyed(fn, {inner_key: args})`` returns ``{inner_key: value}`` and records completion order;
    callers MUST combine with `_stages.ordered_values` (sorted inner keys), never completion order.
    ``fn`` must be a module-level function (pickled by reference).
    """

    def __init__(self, workers: int, *, cap_bytes: int | None = None, backend: str = "none") -> None:
        self.workers = max(1, int(workers))
        self._cap, self._backend = cap_bytes, backend
        self._ex: concurrent.futures.ProcessPoolExecutor | None = None
        self._log: list[str] = []

    def _executor(self) -> concurrent.futures.ProcessPoolExecutor:
        if self._ex is None:
            self._ex = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_inner_init,
                initargs=(self._cap, self._backend),
            )
        return self._ex

    def map_keyed(self, fn: Callable[..., Any], items: Mapping[str, tuple]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.workers == 1:
            for k in sorted(items):
                out[k] = fn(*items[k])
                self._log.append(k)
            return out
        ex = self._executor()
        futures = {ex.submit(fn, *args): k for k, args in items.items()}
        for fut in concurrent.futures.as_completed(futures):
            k = futures[fut]
            out[k] = fut.result()
            self._log.append(k)
        return out

    def drain_log(self) -> list[str]:
        log, self._log = self._log, []
        return log

    def shutdown(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True, cancel_futures=True)
            self._ex = None


def _claim(claims: pathlib.Path, key: str, wid: str) -> bool:
    try:
        fd = os.open(claims / f"{key}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="ascii") as fh:
        fh.write(wid)
    return True


def _claims_of(claims: pathlib.Path, wid: str) -> list[str]:
    out = []
    for p in claims.glob("*.claim"):
        try:
            if p.read_text(encoding="ascii") == wid:
                out.append(p.name[: -len(".claim")])
        except OSError:
            continue
    return out


def _read_outcomes(outdir: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for p in outdir.glob("*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: r["finished_ns"])


def _first_k_all_failed(st, rows: list[dict]) -> bool:
    """The systematic-bug rule: the first ``abort_first_k`` completed units (by finish time) all failed."""
    k = st.abort_first_k
    first = rows[:k]
    return k > 0 and len(first) == k and all(r["outcome"] == "failed" for r in first)


#: Failed-unit markers inside the job-local outcomes directory. Only failures are listed, so a healthy
#: run reads an empty directory per unit (no rescan-in-loop over the outcome log, ADR-068).
_FAILED_DIR = "failed"


def _worker_should_stop(st, outdir: pathlib.Path, decided: dict) -> bool:
    """The worker-side mirror of `LocalExecutor._check_failure_rule`.

    The parent checks the rule every poll interval; a worker whose units finish faster than one poll would
    otherwise run the whole slice before the parent can stop it. The worker checks before each claim and
    stops claiming (exit 0); the parent's check stays authoritative and raises. ``decided`` caches that
    the first-k window is full and not all failed, after which that rule can never fire again.
    """
    failed_dir = outdir / _FAILED_DIR
    if st.max_failures is not None and failed_dir.is_dir():
        if sum(1 for _ in failed_dir.iterdir()) > st.max_failures:
            return True
    if decided.get("first_k"):
        return False
    rows = _read_outcomes(outdir)
    if _first_k_all_failed(st, rows):
        return True
    if len(rows) >= st.abort_first_k:
        decided["first_k"] = True
    return False


def _charge_death(wid: str, unit_kills: set[str], attempts: dict[str, int]) -> int:
    """Worker ``wid`` died: its deaths charged against ``max_relaunch``. A kill over the memory cap WITH
    its in-progress unit recorded (``unit_kills``) made progress -- the unit escalates -- so it is not
    charged (and is consumed); any other death is."""
    if wid in unit_kills:
        unit_kills.discard(wid)
    else:
        attempts[wid] = attempts.get(wid, 0) + 1
    return attempts.get(wid, 0)


def _terminate(procs: dict[str, subprocess.Popen]) -> None:
    for p in procs.values():
        if p.poll() is None:
            p.terminate()
    for p in procs.values():
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()


class LocalExecutor:
    def __init__(self, cfg: ExecutorConfig, *, rss_of: Callable[[int], tuple[int, int] | None] | None = None) -> None:
        """``rss_of(pid)`` returns ``(pid, bytes)`` of the biggest single process in a worker's tree for
        the ``rss`` backend (default `_mem_cap.largest_rss`; a test injects one)."""
        self.cfg = cfg
        if rss_of is None:
            from scripts._mem_cap import largest_rss

            rss_of = largest_rss
        self._rss_of = rss_of

    def _cpus(self) -> int:
        from scripts._parallel_launch import available_cpus

        return available_cpus()

    def _ram(self) -> int:
        from scripts._parallel_launch import available_ram_bytes

        try:
            return available_ram_bytes()
        except RuntimeError:
            return _FALLBACK_RAM

    # ---- reduce ---------------------------------------------------------------------------------
    def try_reduce(self, ctx, name: str) -> bool:
        """Reduce ``name`` if every unit is done/excluded (failures within tolerance). Idempotent: two
        jobs that both see a complete stage write identical bytes atomically. Returns presence."""
        from scripts._stages import deferred_memory_units, output_present, sorted_units, unit_states

        if output_present(ctx, name, verify=True):
            return True
        st = ctx.pipeline.stage(name)
        keys = sorted_units(ctx, st)
        states = unit_states(ctx, st, keys, verify=False)
        if any(v == "pending" for v in states.values()):
            return False
        if deferred_memory_units(ctx, st, keys):  # waiting for its escalation: a reduce would drop it
            return False
        self._subprocess(ctx, ["reduce", "--stage", name])
        return output_present(ctx, name, verify=True)

    def require_reduced(self, ctx, name: str) -> None:
        from scripts._stages import StageIncompleteError

        if not self.try_reduce(ctx, name):
            raise StageIncompleteError(f"{ctx.pipeline.name}: upstream stage {name!r} is incomplete")

    def _subprocess(self, ctx, args: list[str]) -> None:
        from scripts._stages import context_to_json
        from scripts._thread_pin import deterministic_env

        tmp = pathlib.Path(tempfile.mkdtemp(prefix="sk-reduce-"))
        try:
            (tmp / "ctx.json").write_text(context_to_json(ctx), encoding="utf-8")
            cmd = [sys.executable, "-m", "scripts.hf_jobs.executor", *args, "--ctx", str(tmp / "ctx.json")]
            rc = subprocess.run(cmd, env=deterministic_env(), cwd=REPO_ROOT, check=False).returncode  # noqa: S603
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if rc != 0:
            raise StageFailedError(f"{ctx.pipeline.name}: '{' '.join(args)}' failed (rc={rc})")

    # ---- units ----------------------------------------------------------------------------------
    def run_stage(
        self, ctx, name: str, *, shard: tuple[int, int] = (0, 1), keys: list[str] | None = None
    ) -> StageReport:
        """Run this job's slice of stage ``name``: the round-robin ``shard`` (default), or exactly ``keys``
        (a planner LPT part). Outputs never depend on the slice (INV-1); this is placement only."""
        from scripts._driver import shard_select
        from scripts._stages import deferred_memory_units, sorted_units

        st = ctx.pipeline.stage(name)
        for need in st.needs:
            self.require_reduced(ctx, need)
        units = sorted_units(ctx, st)
        if keys is not None:
            unknown = sorted(set(keys) - set(units))
            if unknown:
                raise ValueError(
                    f"{ctx.pipeline.name}/{name}: --shard-keys lists keys that are not units: {unknown[:5]}"
                )
            mine = sorted(set(keys))
        else:
            mine = shard_select(units, shard[0], shard[1])
        if not self.cfg.escalate:  # a deferred memory unit runs only in its escalation job (W=1, whole box)
            deferred = deferred_memory_units(ctx, st, mine)
            mine = [k for k in mine if k not in deferred]
        if self.cfg.profile_units:
            return self._profile(ctx, st, mine)
        return self._run(ctx, st, mine, workers=self.cfg.workers, inner=self.cfg.inner_workers)

    def _run(self, ctx, st, keys: list[str], *, workers: int, inner: int) -> StageReport:
        from scripts._parallel_launch import size_workers
        from scripts._stages import deferred_memory_units, unit_states

        states = unit_states(ctx, st, keys, verify=True)  # INV-3: a resume re-hashes before it skips
        pending = [k for k in keys if states[k] in ("pending", "failed")]
        relaunched, order, w = 0, (), 0
        if pending:
            w = 1 if st.inner_parallel else max(1, min(workers, len(pending)))
            if self.cfg.peak_rss_bytes and not st.inner_parallel:
                w = min(w, size_workers(self.cfg.peak_rss_bytes, self._ram(), self._cpus(), self.cfg.headroom_bytes))
            relaunched, order = self._dispatch(ctx, st, pending, w, inner if st.inner_parallel else 1)
        final = unit_states(ctx, st, keys, verify=False)
        deferred = deferred_memory_units(ctx, st, keys)  # waiting for escalation: not a stage failure yet
        n_failed = sum(v == "failed" and k not in deferred for k, v in final.items())
        report = StageReport(
            st.name,
            len(keys),
            sum(v == "done" for v in final.values()),
            sum(v == "excluded" for v in final.values()),
            n_failed,
            w,
            relaunched,
            order,
        )
        if st.max_failures is not None and n_failed > st.max_failures:
            raise StageFailedError(
                f"{ctx.pipeline.name}/{st.name}: {n_failed} failed unit(s) > tolerance {st.max_failures}; "
                f"see *.failed.json in {ctx.generation(st.name)}"
            )
        return report

    def _check_failure_rule(self, st, outdir: pathlib.Path, procs) -> None:
        rows = _read_outcomes(outdir)
        failed = [r for r in rows if r["outcome"] == "failed"]
        if st.max_failures is not None and len(failed) > st.max_failures:
            _terminate(procs)
            raise StageFailedError(
                f"{st.name}: {len(failed)} failed unit(s) > tolerance {st.max_failures} (e.g. {failed[0]['key']})"
            )
        if _first_k_all_failed(st, rows):
            _terminate(procs)
            raise StageFailedError(f"{st.name}: the first {st.abort_first_k} completed units all failed (systematic)")

    def _dispatch(self, ctx, st, pending: list[str], workers: int, inner: int) -> tuple[int, tuple[str, ...]]:
        from scripts._mem_cap import detect_backend, kill_tree, preflight, wrap
        from scripts._parallel_launch import should_launch
        from scripts._stages import context_to_json, unit_states
        from scripts._thread_pin import deterministic_env

        tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"sk-{st.name}-"))
        procs: dict[str, subprocess.Popen] = {}
        try:
            (tmp / "ctx.json").write_text(context_to_json(ctx), encoding="utf-8")
            (tmp / "pending.json").write_text(json.dumps(pending), encoding="utf-8")
            claims, outdir = tmp / "claims", tmp / "outcomes"
            claims.mkdir()
            outdir.mkdir()
            backend = detect_backend(self.cfg.mem_backend, watched=True)  # type: ignore[arg-type]
            cap = self.cfg.cap_bytes
            if cap:
                preflight(backend, cap_bytes=cap)
            env = deterministic_env()

            def launch(wid: str) -> None:
                argv = [
                    sys.executable,
                    "-m",
                    "scripts.hf_jobs.executor",
                    "worker",
                    "--ctx",
                    str(tmp / "ctx.json"),
                    "--stage",
                    st.name,
                    "--pending",
                    str(tmp / "pending.json"),
                    "--claims",
                    str(claims),
                    "--outcomes",
                    str(outdir),
                    "--worker-id",
                    wid,
                    "--inner-workers",
                    str(inner),
                    "--mem-backend",
                    backend,
                ]
                if cap:
                    argv += ["--cap-bytes", str(cap)]
                pre = None
                if cap:
                    argv, pre = wrap(backend, argv, cap_bytes=cap)
                procs[wid] = subprocess.Popen(argv, env=env, cwd=REPO_ROOT, preexec_fn=pre)  # noqa: S603 -- own argv

            slots = [f"w{i}" for i in range(workers)]
            launch(slots.pop(0))  # the first worker always starts: sizing already proved one fits
            attempts: dict[str, int] = {}
            over_cap: dict[str, str] = {}  # the last over-cap kill per worker (the relaunch-limit message)
            unit_kills: set[str] = set()  # workers killed over the cap WITH a unit recorded as failed
            relaunched = 0
            while procs or slots:
                time.sleep(self.cfg.poll_interval_s)
                while slots and (
                    not self.cfg.peak_rss_bytes
                    or should_launch(self._ram(), self.cfg.peak_rss_bytes, self.cfg.peak_rss_bytes)
                ):
                    launch(slots.pop(0))
                self._check_failure_rule(st, outdir, procs)
                if backend == "rss" and cap:
                    unit_kills |= self._enforce_rss_cap(ctx, st, procs, claims, cap, over_cap, kill_tree)
                for wid, p in list(procs.items()):
                    rc = p.poll()
                    if rc is None:
                        continue
                    del procs[wid]
                    if rc == 0:
                        continue
                    held = _claims_of(claims, wid)
                    for k, state in unit_states(ctx, st, held, verify=False).items():
                        if state == "pending":
                            (claims / f"{k}.claim").unlink(missing_ok=True)  # the dead worker's unit is free again
                    left = [
                        k
                        for k, v in unit_states(ctx, st, pending, verify=False).items()
                        if v == "pending" and not (claims / f"{k}.claim").exists()
                    ]
                    if not left:
                        continue
                    if _charge_death(wid, unit_kills, attempts) > self.cfg.max_relaunch:
                        _terminate(procs)
                        why = f"; {over_cap[wid]}" if wid in over_cap else ""
                        raise StageFailedError(
                            f"{st.name}: worker {wid} died {attempts[wid]} times (last rc={rc}){why}"
                        )
                    print(f"relaunching worker {wid} (rc={rc}); {len(left)} unit(s) left", file=sys.stderr, flush=True)
                    relaunched += 1
                    launch(wid)
                if not procs and slots:
                    # Backpressure must never deadlock: with nothing running, waiting frees no RAM, and
                    # sizing already proved ONE worker fits. Launch one if unclaimed work is left.
                    states = unit_states(ctx, st, pending, verify=False)
                    if any(v == "pending" and not (claims / f"{k}.claim").exists() for k, v in states.items()):
                        launch(slots.pop(0))
                    else:
                        slots.clear()
            self._check_failure_rule(st, outdir, {})
            order = tuple(r["key"] for r in _read_outcomes(outdir))
            return relaunched, order
        finally:
            _terminate(procs)
            shutil.rmtree(tmp, ignore_errors=True)

    def _enforce_rss_cap(self, ctx, st, procs, claims: pathlib.Path, cap: int, over_cap: dict, kill_tree) -> set[str]:
        """The ``rss`` backend: kill a live worker whose tree holds a process over ``cap`` (the cap is per
        process, `_mem_cap.largest_rss`). Its unit is unclaimed and retried by a relaunch (the dead-worker
        path). The unit in progress is recorded as FAILED for memory (`_stages.record_memory_failure`), so
        the job's other units go on and the controller escalates it to a job of its own with the whole box;
        it is never re-run here. Returns the workers whose in-progress unit was recorded (a kill with no
        unit in progress still counts toward the relaunch limit, whose message names the cap)."""
        from scripts._stages import record_memory_failure, unit_states

        recorded: set[str] = set()
        for wid, p in list(procs.items()):
            if p.poll() is not None:
                continue
            top = self._rss_of(p.pid)
            if top is None or top[1] <= cap:
                continue
            held = _claims_of(claims, wid)
            busy = [k for k, v in unit_states(ctx, st, sorted(held), verify=False).items() if v == "pending"]
            who = "" if top[0] == p.pid else f" in child pid {top[0]}"
            what = f"on unit(s) {busy}" if busy else "with no unit in progress"
            over_cap[wid] = f"over its memory cap (RSS {top[1]} B > cap {cap} B{who}) {what}"
            for k in busy:
                record_memory_failure(ctx.store, ctx.generation(st.name), k, over_cap[wid])
            if busy:
                recorded.add(wid)
            print(f"killing worker {wid}: {over_cap[wid]}", file=sys.stderr, flush=True)
            if not kill_tree(p.pid):
                p.kill()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        return recorded

    # ---- profile (spec 5.5) -----------------------------------------------------------------------
    def _peak_of(self, ctx, st, keys: list[str]) -> int | None:
        peaks = []
        for k in keys:
            raw = ctx.store.read_bytes(f"{ctx.generation(st.name)}/{k}.ledger.json")
            if raw:
                v = json.loads(raw).get("peak_rss_bytes")
                if v:
                    peaks.append(int(v))
        return max(peaks) if peaks else None

    def _profile(self, ctx, st, mine: list[str]) -> StageReport:
        """Run K evenly spaced units: two at W=1 (inner W=1), the rest at the RAM-bounded maximum W
        (inner W for an inner_parallel stage). Writes ``_profile.json``. The units are real: their
        results persist and the campaign never redoes them."""
        from scripts._parallel_launch import size_workers
        from scripts._stages import _canon_json, profile_sample

        sample = profile_sample(mine, self.cfg.profile_units)
        first, rest = sample[:2], sample[2:]
        t0 = time.monotonic()
        self._run(ctx, st, first, workers=1, inner=1)
        wall_1 = time.monotonic() - t0
        peak = self._peak_of(ctx, st, first) or self.cfg.peak_rss_bytes
        if st.inner_parallel:
            wmax, inner = 1, max(1, self.cfg.inner_workers)
        else:
            inner = 1
            wmax = self.cfg.workers
            if peak:
                wmax = min(wmax, size_workers(int(peak * 1.3), self._ram(), self._cpus(), self.cfg.headroom_bytes))
        t1 = time.monotonic()
        report = (
            self._run(ctx, st, rest, workers=wmax, inner=inner) if rest else self._run(ctx, st, [], workers=1, inner=1)
        )
        wall_max = time.monotonic() - t1
        par = inner if st.inner_parallel else wmax
        rate_1 = len(first) / wall_1 if wall_1 > 0 else None
        rate_max = len(rest) / wall_max if (rest and wall_max > 0) else None
        record = {
            "stage": st.name,
            "sample": sample,
            "w1": {"units": first, "wall_s": wall_1},
            "wmax": {"workers": wmax, "inner_workers": inner, "units": rest, "wall_s": wall_max},
            "peak_rss_bytes": peak,
            "efficiency": (rate_max / (par * rate_1)) if (rate_max and rate_1) else None,
            "nproc": self._cpus(),
        }
        ctx.store.write_bytes(f"{ctx.generation(st.name)}/_profile.json", _canon_json(record))
        return report


def worker_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="executor worker")
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--pending", required=True)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--inner-workers", type=int, default=1)
    ap.add_argument("--mem-backend", default="none")
    ap.add_argument("--cap-bytes", type=int, default=None)
    a = ap.parse_args(argv)
    from scripts._stages import ItemStage, context_from_json, run_unit

    ctx = context_from_json(pathlib.Path(a.ctx).read_text(encoding="utf-8"))
    st = ctx.pipeline.stage(a.stage)
    pending = json.loads(pathlib.Path(a.pending).read_text(encoding="utf-8"))
    refs = st.ref_map(ctx) if isinstance(st, ItemStage) else {}
    pool = InnerPool(a.inner_workers, cap_bytes=a.cap_bytes, backend=a.mem_backend) if st.inner_parallel else None
    outdir = pathlib.Path(a.outcomes)
    log = outdir / f"{a.worker_id}.jsonl"
    claims = pathlib.Path(a.claims)
    decided: dict = {}
    try:
        for key in pending:
            if _worker_should_stop(st, outdir, decided):
                break
            if not _claim(claims, key, a.worker_id):
                continue
            outcome = run_unit(ctx, st, key, worker=a.worker_id, pool=pool, ref=refs.get(key))
            if outcome == "failed":
                (outdir / _FAILED_DIR).mkdir(exist_ok=True)
                (outdir / _FAILED_DIR / key).touch()
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "outcome": outcome, "finished_ns": time.time_ns()}) + "\n")
    finally:
        if pool is not None:
            pool.shutdown()
    return 0


def reduce_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="executor reduce")
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--stage", required=True)
    a = ap.parse_args(argv)
    from scripts._stages import StageIncompleteError, context_from_json, reduce_stage

    ctx = context_from_json(pathlib.Path(a.ctx).read_text(encoding="utf-8"))
    try:
        reduce_stage(ctx, a.stage)
    except StageIncompleteError as exc:
        print(f"STAGE FAILED: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    _CMDS = {"worker": worker_main, "reduce": reduce_main}
    if len(sys.argv) < 2 or sys.argv[1] not in _CMDS:
        raise SystemExit(f"usage: python -m scripts.hf_jobs.executor {{{','.join(_CMDS)}}} ...")
    raise SystemExit(_CMDS[sys.argv[1]](sys.argv[2:]))
