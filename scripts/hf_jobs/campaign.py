"""The stateless HF campaign controller (ADR-128, spec 5.5).

A campaign is a COMMITTED TOML file (``scripts/hf_jobs/campaigns/<name>.toml``, INV-7): name, bucket,
wall target, concurrency cap, and per job the trainer or driver, its tier, its dataset and its variants.
Run keys are DERIVED (``{campaign}-{commit7}-{job}-{tier}``, plus ``-r<N>`` for a fresh replica store,
decision P18), so two campaigns can never share a store by accident.

The controller holds NO state of its own. Every step rebuilds the state from two sources only: each
instance's ``--plan`` JSON read against the bucket store (tokens, done/pending units per stage), and the
HF job labels (``list_jobs(labels=...)``: what is running, what ended in ERROR). It can stop at any time;
``resume`` (or a re-run of ``launch``) continues from the store. Each instance advances one stage at a
time (the stage barrier); different instances advance concurrently up to the account concurrency cap.

The pure decision is `next_actions` (tested without HF); `step` applies it through the typed launcher.

Commands: ``python -m scripts.hf_jobs campaign <file> {preflight,profile,plan,launch,resume,status,
retrieve,verify,adopt,publish,cancel,summary}``.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from scripts.hf_jobs.planner import SATURATION_FRACTION, Constraints, UnitCost, irreducible_floor_s, plan_stage

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SUMMARY_DIR = REPO_ROOT / "docs" / "research" / "hf_campaigns"


@dataclasses.dataclass(frozen=True)
class CampaignJob:
    job: str
    tier: str
    dataset: str
    variants: tuple[str, ...]
    probe_dataset: str | None = None


@dataclasses.dataclass(frozen=True)
class Campaign:
    name: str
    bucket: str
    target_minutes: int
    dispatch_cap: int  # jobs in flight at once (the probed HF account budget)
    shard_cap: int  # the planner's per-stage shard ceiling (decoupled; see Constraints / ADR-128 r3 B17)
    attempts: int
    jobs: tuple[CampaignJob, ...]
    path: pathlib.Path | None
    sha256: str

    def constraints(self, *, startup_s: float | None = None) -> Constraints:
        base = Constraints(
            target_s=float(self.target_minutes) * 60.0, dispatch_cap=self.dispatch_cap, shard_cap=self.shard_cap
        )
        return base if startup_s is None else dataclasses.replace(base, startup_s=startup_s)


@dataclasses.dataclass(frozen=True)
class Instance:
    """One (job, tier, variant) of a campaign: the module the jobs run, its args, datasets and store."""

    job: str
    tier: str
    variant: str
    module: str
    args: tuple[str, ...]
    datasets: tuple[Any, ...]  # launch.DatasetMount
    run_key: str

    @property
    def id(self) -> str:
        # spec r4 D18 (c): tiers that read one dataset share the run key, so the id names the tier
        return f"{self.run_key}/{self.tier}/{self.variant}"


@dataclasses.dataclass(frozen=True)
class Action:
    """What one stateless step launches for one instance (spec 5.5)."""

    kind: str  # "profile" | "run" | "reduce" | "halt" (failed units out of attempts: report, launch nothing)
    stages: tuple[str, ...]
    n_jobs: int
    workers: int
    inner_workers: int
    timeout_s: int
    cap_bytes: int | None = None
    #: HFU T11 + reviewer B D1-IMPL-07: the pool CHILD cap of an inner_parallel stage, distinct from ``cap_bytes``
    #: (the unit worker's cap). None for a non-inner stage.
    inner_cap_bytes: int | None = None
    peak_rss_bytes: int | None = None
    partition: tuple[tuple[str, ...], ...] | None = None
    profile: int | None = None
    predicted_job_wall_s: float | None = None
    predicted_cost_usd: float | None = None
    reason: str | None = None
    #: The memory escalation: the job also runs units deferred by a memory failure (`--escalate`).
    escalate: bool = False
    #: HFU plan R3-8 S3 item 5: a WIO stage's run (`--wio`): the executor bounds W by the planned W, not nproc.
    wio: bool = False
    #: HFU plan R3-8 S3 item 6 (MEM): a run under the executor's memory gate; its jobs read the per-unit peaks.
    gated: bool = False
    #: HFU T8: a sub-reduce's shard index (``kind == "subreduce"``). The worker computes its range of DONE units
    #: from ``shard_id`` and ``n_shards`` -- the controller passes no key list (reviewer B D1-IMPL-08).
    shard_id: int | None = None
    #: HFU T8: a sharded stage's shard count. Non-None marks a ``reduce`` action as the FINAL concat over N
    #: partials, a ``subreduce`` action's shard count, and a ``run`` action as belonging to a sharded stage.
    n_shards: int | None = None


# ---- model ---------------------------------------------------------------------------------------


def load_campaign(path: str | pathlib.Path) -> Campaign:
    from scripts._toml import load_toml

    path = pathlib.Path(path)
    raw = load_toml(path)
    c = raw["campaign"]
    jobs = tuple(
        CampaignJob(j["job"], j["tier"], j["dataset"], tuple(j["variants"]), j.get("probe_dataset"))
        for j in raw.get("jobs", [])
    )
    from scripts.hf_jobs.registry import canary_spec

    canaries = sorted({j.job for j in jobs if canary_spec(j.job) is not None})
    if canaries and not str(c["name"]).startswith("canary"):  # HFU rung 5 (C3): the toy halt arm is canary-only
        raise ValueError(
            f"campaign {c['name']!r}: {canaries} are canary-only jobs (HFU rung 5); only a campaign named "
            "canary* may run them"
        )
    # dispatch_cap and shard_cap replace the single concurrency_cap (accepted as a legacy alias for both);
    # shard_cap omitted => equals dispatch_cap so a single-cap campaign is unchanged.
    if "dispatch_cap" in c or "shard_cap" in c:
        dispatch_cap = int(c.get("dispatch_cap", c.get("shard_cap")))
        shard_cap = int(c.get("shard_cap", dispatch_cap))
    else:
        dispatch_cap = shard_cap = int(c["concurrency_cap"])  # legacy single cap
    return Campaign(
        name=c["name"],
        bucket=c["bucket"],
        target_minutes=int(c["target_minutes"]),
        dispatch_cap=dispatch_cap,
        shard_cap=shard_cap,
        attempts=int(c["attempts"]),
        jobs=jobs,
        path=path,
        # The campaign sha256 gates preflight -> launch. Hash the CANONICAL line-ending-normalized bytes so a
        # Windows preflight (CRLF working file, via autocrlf) and a Linux launch (LF) agree -- the gate must not
        # depend on which OS ran preflight vs launch. (The sha256 is used ONLY for this gate, not in run keys or
        # generation tokens, so normalizing changes no store identity.)
        sha256=hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    )


def run_key(c: Campaign, commit: str, job: str, dataset: str, *, replica: int | None = None) -> str:
    """``{campaign}-{commit7}-{job}-{dataset}`` (+ ``-r<N>``: a FRESH store for the INV-1 replica gate, P18). One store
    per (job, dataset): tiers that read one dataset share it and its extraction (spec r4 D18 (c))."""
    key = f"{c.name}-{commit[:7]}-{job}-{dataset}"
    return f"{key}-r{replica}" if replica is not None else key


def _mounts(names: Iterable[str | None]) -> tuple[Any, ...]:
    from scripts.hf_jobs.datasets import declared, locked
    from scripts.hf_jobs.launch import DatasetMount

    out = []
    for name in names:
        if name is None:
            continue
        d = declared()[name]
        out.append(DatasetMount(name, d.repo, locked(name)["revision"], d.access, d.read_concurrency))
    return tuple(out)


def instances(c: Campaign, commit: str, *, replica: int | None = None) -> list[Instance]:
    """Every (job, tier, variant) of the campaign with its module, args and datasets (spec 5.5)."""
    from scripts.hf_jobs import arm_policy
    from scripts.hf_jobs.datasets import declared, locked, staged_units
    from scripts.hf_jobs.registry import DriverSpec, canary_spec, spec_for

    out: list[Instance] = []
    for j in c.jobs:
        rk = run_key(c, commit, j.job, j.dataset, replica=replica)
        canary = canary_spec(j.job)
        if canary is not None:  # HFU rung 5 (C3): the toy halt arm -- its own args, no dataset, no mount
            out.extend(Instance(j.job, j.tier, v, canary.module, canary.base_args, (), rk) for v in j.variants)
            continue
        spec = spec_for(j.job)
        mounts = _mounts([j.dataset, j.probe_dataset])
        if isinstance(spec, DriverSpec):
            for v in j.variants:
                out.append(Instance(j.job, j.tier, v, spec.module, ("--dataset", j.dataset), mounts, rk))
            continue
        builds = {b.variant: b for b in spec.builds}
        for v in j.variants:
            b = builds[v]
            args: tuple[str, ...] = (*spec.base_args, *b.run_args, "--dataset", j.dataset)
            if j.probe_dataset is not None:
                args += ("--probe-dataset", j.probe_dataset)
            if spec.visibility_gated:  # spec r4 D18 (c): only the arms this tier ships, off-HF
                # HFU R4-10: keyed by each locked unit's stem AND source_provider (the in-job key)
                arms = arm_policy.arms_for_tier(arm_policy.compute_arms(staged_units(j.dataset)), j.tier)
                args += ("--arms-json", json.dumps(arms, sort_keys=True))
            if j.job == "receiver_model":  # decision P5: visibility is a parameter, never read in-job
                args += (
                    "--corpus-visibility",
                    declared()[j.dataset].visibility,
                    "--corpus-n-matches",
                    str(locked(j.dataset)["n_units"]),
                )
            out.append(Instance(j.job, j.tier, v, spec.pipeline or spec.module, args, mounts, rk))
    return out


# ---- the pure decision ---------------------------------------------------------------------------


def _group(stages: Sequence[Mapping], start: int, reduced: set[str]) -> tuple[str, ...]:
    """Consecutive single-unit stages that can run in ONE job: each next stage is ready, or becomes ready
    only through stages already in the group; an inner_parallel stage never joins (it owns the pool)."""
    first = stages[start]
    group = [first["name"]]
    if first.get("inner_parallel"):
        return tuple(group)
    for nxt in stages[start + 1 :]:
        needs = set(nxt.get("needs") or ())
        inside = needs & set(group)
        if nxt.get("reduced") or nxt.get("n_units") != 1 or nxt.get("inner_parallel"):
            break
        if not (nxt.get("ready") or (inside and needs <= set(group) | reduced)):
            break
        group.append(nxt["name"])
    return tuple(group)


def _reduce_timeout(reduce_wall, c, *, lower_bound: bool = False):
    """A reduce timeout: timeout_safety x the reduce wall (measured, or the spec r3 D2 estimate) FLOORED at
    timeout_floor_s (a tiny wall must not give a near-zero timeout -- partial review B D1-IMPL-01), else the
    generous profile ceiling. A ``lower_bound`` wall (an estimate whose combine was not measured) floors at that
    ceiling instead: the unmeasured combine keeps its protection, and the read estimate can only raise it."""
    if reduce_wall is None:
        return int(c.profile_timeout_s)
    return int(max(c.profile_timeout_s if lower_bound else c.timeout_floor_s, c.timeout_safety * reduce_wall))


#: The cold per-unit read through the HF bucket mount, measured ON HF (B r4 R4-03): a fresh read-only container read
#: the real 3,961-shard `win_probability` `extract` generation (result + sidecar, verify, decode) at 0.3628 s per
#: key (job ``6ae4884ac78863f9c62e7481``), and fetched a 155.5 MB result in 0.727 s (job
#: ``6ae4a0dfc78863f9c62e806e``, about 214 MB/s). The real reduce reads its units cold, in a fresh container.
COLD_READ_S_PER_UNIT = 0.3628
COLD_READ_BYTES_PER_S = 2.14e8


def _reduce_estimate(entry: Mapping, n_units: int) -> tuple[float, bool] | None:
    """Spec r3 D2 (b): the reduce wall estimated BEFORE any reduce has run, from the profile's measured per-unit
    reduce cost: ``n_units x (read / REDUCE_READ_WORKERS + combine)``. The profile reads back units its OWN container
    just wrote, which a local write cache may serve, so the per-unit read is never taken below the HF-measured cold
    read for the unit's size (``COLD_READ_S_PER_UNIT + bytes / COLD_READ_BYTES_PER_S``; B r4 R4-03). Returns
    ``(seconds, combine_measured)``; a non-shardable reduce's combine is never run on a sample, so its estimate is
    the read alone, a lower bound. None when the profile measured no reduce cost."""
    from scripts._stages import REDUCE_READ_WORKERS

    cost = entry.get("reduce_cost") or {}
    if cost.get("read_s_per_unit") is None:
        return None
    cold = COLD_READ_S_PER_UNIT + float(cost.get("bytes_per_unit") or 0.0) / COLD_READ_BYTES_PER_S
    combine = cost.get("combine_s_per_unit")
    per_unit = max(float(cost["read_s_per_unit"]), cold) / REDUCE_READ_WORKERS + float(combine or 0.0)
    return n_units * per_unit, combine is not None


def _bundle_timeout(grp: Sequence[str], stages: Sequence[Mapping], history: Mapping, c: Constraints) -> int:
    """The tiny/grouped job timeout (spec §D, owner-ratified gold standard 2026-10-29). Size it to the job's
    ACTUAL sequential work -- ``timeout_safety * (sum of n_units * p99 over the bundled stages + their inline
    reduce walls) + startup_s``, floored at ``timeout_floor_s``. If ANY bundled stage has no committed p99 (a
    genuine first run), fall back to the generous ``profile_timeout_s`` ceiling. NEVER a function of ``target_s``
    (R3). This replaces the flat ceiling, which killed a heavy grouped chain whose sequential wall exceeded it
    (reviewer A D-IMPL-07), and the spec's prior ``1.5 * p99`` of ONE unit, which killed a 2-unit tiny job
    (partial review B D1-IMPL-02). It sizes to the real work and does neither."""
    by_name = {s["name"]: s for s in stages}
    total = 0.0
    for name in grp:
        s = by_name.get(name) or {}
        entry = history.get(s.get("history_key") or name) or {}
        cost = entry.get("cost")
        if cost is None:
            return int(c.profile_timeout_s)  # first run: no committed p99 for a bundled stage
        total += int(s.get("n_units") or 1) * cost.p99_s + float(entry.get("reduce_wall") or 0.0)
    return int(max(c.timeout_floor_s, c.timeout_safety * total + c.startup_s))


def _reduce_shards(st: Mapping, reduce_wall: float, budget: float) -> int:
    """HFU T8: the NUMBER of sub-reduce shards a shardable over-budget reduce splits into -- each shard carries
    about one per-job budget of the reduce wall (>= 2, never more than the unit count). The worker computes each
    shard's contiguous range from the stage's DONE units in reduce_order (`reduce_subrange`), so the controller
    passes only this count -- never a plain-sorted key range, which would reorder a custom reduce_order across
    shards (reviewer B D1-IMPL-08)."""
    n = -(-int(reduce_wall) // int(max(1.0, budget)))  # ceil; each shard ~ one per-job budget of reduce wall
    return max(2, min(n, max(2, len(st.get("units") or []))))


def all_settled(rep: Mapping) -> bool:
    """--watch is settled only when NO instance launched a job this step -- in its first reported entry OR any
    concurrent `also` sibling -- and every instance is terminal (done, or failed/halt). A resume-error on one
    stage while a sibling's jobs are live is NOT settled, so --watch never exits with jobs running (partial
    review B D1-IMPL-05 / B-07). A halt whose cancel-API call FAILED is not settled either, so --watch retries
    the cancel and never exits leaving the halted instance's jobs alive (reviewer B D1-IMPL-12), and neither is a
    halt whose cancel waits on another instance's errored plan (spec r4 D17: ``cancel_deferred``). A halt that KEPT
    a job another instance needs settles; that instance is not terminal until the job's work is done."""
    for r in rep.values():
        if r.get("jobs") or any(e.get("jobs") for e in r.get("also", [])):
            return False
        if r.get("cancel_failed"):  # a halt whose cancel-API call failed: not settled, retry it next step (D1-IMPL-12)
            return False
        if r.get("cancel_deferred"):  # spec r4 D17: a cancel waits on another instance's plan; decide it next step
            return False
        if r.get("action") != "done" and not r.get("failed"):
            return False
    return True


def watch_exit_code(rep: Mapping) -> int:
    """The exit code of a SETTLED ``--watch`` (spec r3 D15): non-zero when any instance halted (a ``failed``
    report, in its entry or a concurrent ``also`` sibling), so a wrapper can tell a halted campaign from a
    finished one; 0 only when every instance is done."""
    for r in rep.values():
        if r.get("failed") or any(e.get("failed") for e in r.get("also", [])):
            return 1
    return 0


#: History measurements a live entry may lack while the committed summary holds them: the inline-reduce wall
#: (the reduce has not run this run yet; partial review B D1-IMPL-04), the stage's startup (live history has no
#: HF job start to measure it from) and its upstream load (no unit has loaded yet this run) (spec r3 D11).
_INHERITED_MEASUREMENTS = ("reduce_wall", "startup_s", "upstream_load_s")


def _merge_history(committed: Mapping, live: Mapping) -> dict:
    """Live (this run's) history overrides committed, BUT a measurement the live entry lacks (None or absent;
    `_INHERITED_MEASUREMENTS`) inherits the committed value, so a measured value is not lost after the first wave
    (T8 reads ``reduce_wall``; the D11 overhead reads ``startup_s`` and ``upstream_load_s``)."""
    out = {**committed, **live}
    for hk, entry in live.items():
        prior = committed.get(hk) or {}
        kept = {k: prior[k] for k in _INHERITED_MEASUREMENTS if entry.get(k) is None and prior.get(k) is not None}
        if kept:
            out[hk] = {**entry, **kept}
    return out


def _stage_startup_s(entry: Mapping, c: Constraints) -> float:
    """The stage's own measured startup (HF ``started_at`` to the first unit; spec r3 D11), else the campaign's."""
    startup = entry.get("startup_s")
    return float(c.startup_s if startup is None else startup)


def _stage_overhead_s(entry: Mapping, c: Constraints) -> float:
    """The stage's measured per-job overhead (spec r3 D11): its startup plus its measured upstream load."""
    return _stage_startup_s(entry, c) + float(entry.get("upstream_load_s") or 0.0)


def next_actions(
    plan: Mapping,
    running: set[str],
    history: Mapping[str, Mapping],
    c: Constraints,
    *,
    lpt: bool = True,
    free_slots: int = 0,
    unit_peaks: Mapping[str, Mapping[str, int]] | None = None,
) -> list[Action]:
    """The one action to take for ONE instance now (empty: nothing to do, or waiting).

    Walks the stages in DAG order: a reduced stage is skipped; a stage that is not ready, whose units are
    still unknown, or that is already running stops the walk (the stage barrier). A stage with no pending
    unit gets a reduce job; a tiny stage runs in one job (grouped with the following single-unit stages);
    a stage with no cost history is profiled first; otherwise the planner sizes N jobs x W workers over
    the stage's PENDING units only. An LPT partition is built only when ``lpt`` (an instance with a
    ``download`` dataset fetches by ``--shard i/N`` and takes round-robin shards instead); a pending unit
    with no measured wall is costed at the stage p50. A FAILED unit is scheduled again with the pending
    ones while it has failed fewer than ``c.unit_attempts`` times (a transient MemoryError or EIO must not
    drop a game from a tolerant stage's reduce). Once only failures out of attempts are left, the stage's
    declared ``max_failures`` decides: within it (``None`` = unlimited, the legacy "record failures and fit
    the rest") the stage reduces without them and the action names them; over it the action is ``halt``,
    which names them and launches nothing.

    ``unit_peaks`` (S3 item 6, MEM): per stage name, the per-unit peak RSS of the newest earlier generation
    (`bucket_unit_peaks`). A stage that has them runs under the executor's memory gate.
    """
    stages = list(plan["stages"])
    reduced = {s["name"] for s in stages if s.get("reduced")}
    actions: list[Action] = []
    grouped: set[str] = set()  # stages already covered by a tiny-stage _group action
    for i, st in enumerate(stages):
        # HFU T5: the DAG schedule -- emit an action for EVERY ready stage, not only the first in declaration
        # order. A reduced/grouped/not-ready/running stage is skipped (its `ready` flag already encodes its
        # `needs`), and the walk CONTINUES to later independent stages (no linear barrier).
        if st.get("reduced") or st["name"] in grouped:
            continue
        if not st.get("ready") or st.get("n_units") is None or st["name"] in running:
            continue
        inner_w = int(st.get("inner_width") or 1)
        failed = dict(st.get("failed_units") or {})
        retry = {k for k, n in failed.items() if int(n) < c.unit_attempts}
        memory = retry & set(st.get("memory_units") or ())
        todo = sorted((set(st.get("pending_units") or ()) | retry) - memory)
        # HFU R3 / spec §D: the escalate (memory-alone) job re-runs a memory-FAILED unit, which has no committed
        # wall, so it uses the generous profile ceiling ("else profile_timeout_s"), never 2*target_s. The
        # tiny/grouped run instead sizes its timeout to the job's measured work (`_bundle_timeout` below).
        tiny_timeout = int(c.profile_timeout_s)
        entry = history.get(st.get("history_key") or st["name"]) or {}
        # The reduce wall the decisions below use: the MEASURED one, else (spec r3 D2) the estimate from the
        # profile's per-unit reduce cost, so a heavy reduce can shard and get a sized timeout on its FIRST run.
        reduce_wall = entry.get("reduce_wall")
        lower_bound = False  # True: an estimate whose combine was not measured (a non-shardable reduce)
        if reduce_wall is None:
            est = _reduce_estimate(entry, int(st["n_units"]))
            if est is not None:
                reduce_wall, lower_bound = est[0], not est[1]
        # HFU T8: a shardable reduce whose wall exceeds the per-job budget is range-partitioned. A reduce job's
        # budget charges the stage's own measured startup, as its run jobs do (D11; B r4 R4-09); a reduce job loads
        # no upstream, so the load term is not charged.
        budget = max(1.0, c.target_s - _stage_startup_s(entry, c))
        sharded = st.get("shardable_reduce") is not None and reduce_wall is not None and reduce_wall > budget
        n_sub = _reduce_shards(st, reduce_wall, budget) if sharded and reduce_wall is not None else 0
        if not todo and memory:
            # a unit over its fair-share cap: alone in its job, with the whole box (W=1, cap = usable RAM)
            alone = tuple((k,) for k in sorted(memory))
            actions.append(
                Action(
                    "run",
                    (st["name"],),
                    len(alone) if lpt else 1,
                    1,
                    1,
                    tiny_timeout,
                    cap_bytes=c.usable_bytes,
                    partition=alone if lpt else None,
                    escalate=True,
                )
            )
            continue
        if not todo:
            spent = {k: n for k, n in sorted(failed.items()) if k not in retry}
            if not spent:
                if sharded:  # HFU T8: emit the MISSING sub-reduces, then the final concat once all present
                    present = set(st.get("subreduce_present") or ())
                    missing = [j for j in range(n_sub) if j not in present]
                    if missing:
                        for j in missing:
                            actions.append(
                                Action(
                                    "subreduce",
                                    (st["name"],),
                                    1,
                                    1,
                                    1,
                                    _reduce_timeout(reduce_wall, c, lower_bound=lower_bound),
                                    shard_id=j,
                                    n_shards=n_sub,
                                )
                            )
                        continue
                    actions.append(
                        Action(
                            "reduce",
                            (st["name"],),
                            1,
                            1,
                            1,
                            _reduce_timeout(reduce_wall, c, lower_bound=lower_bound),
                            n_shards=n_sub,
                        )
                    )
                    continue
                actions.append(
                    Action("reduce", (st["name"],), 1, 1, 1, _reduce_timeout(reduce_wall, c, lower_bound=lower_bound))
                )
                continue
            tolerance = st.get("max_failures")
            if tolerance is None or len(spent) <= int(tolerance):
                why = f"{st['name']}: reduces without unit(s) failed {c.unit_attempts} time(s): {spent}"
                actions.append(
                    Action(
                        "reduce",
                        (st["name"],),
                        1,
                        1,
                        1,
                        _reduce_timeout(reduce_wall, c, lower_bound=lower_bound),
                        reason=why,
                    )
                )
                continue
            why = (
                f"{st['name']}: {len(spent)} unit(s) failed {c.unit_attempts} time(s), over the stage's "
                f"max_failures={tolerance}: {spent}"
            )
            actions.append(Action("halt", (st["name"],), 0, 0, 0, 0, reason=why))
            continue
        if st["n_units"] <= 2:
            inner = min(c.vcpu, inner_w) if st.get("inner_parallel") else 1
            grp = _group(stages, i, reduced)
            grouped.update(grp)
            actions.append(Action("run", grp, 1, 1, inner, _bundle_timeout(grp, stages, history, c)))
            continue
        hk = st.get("history_key") or st["name"]
        if hk not in history:
            inner = min(c.vcpu, inner_w) if st.get("inner_parallel") else 1
            workers = 1 if st.get("inner_parallel") else c.vcpu
            sample = min(int(st["n_units"]), 2 * c.vcpu)
            actions.append(
                Action("profile", (st["name"],), 1, workers, inner, int(c.profile_timeout_s), profile=sample)
            )
            continue
        h = history[hk]
        keys = todo
        measured = h.get("per_unit") if lpt else None
        per_unit = {k: float(measured.get(k, h["cost"].p50_s)) for k in keys} if measured else None
        p = plan_stage(
            st["name"],
            keys,
            h["cost"],
            inner_parallel=bool(st.get("inner_parallel")),
            thread_safe=bool(st.get("thread_safe")),
            free_slots=free_slots,
            n_inner=inner_w,
            efficiency=h.get("efficiency"),
            per_unit=per_unit,
            overhead_s=_stage_overhead_s(h, c),
            gated=bool(unit_peaks and unit_peaks.get(st["name"])),  # S3 item 6
            c=c,
        )
        # HFU T4/B-19: the inline reduce runs in whichever run job finishes last; budget it. With a reduce wall
        # (measured, or the spec r3 D2 estimate), add it to the measured job timeout; with none, or with an
        # estimate whose combine was not measured, keep the generous profile ceiling as the floor so the inline
        # reduce is never killed at the bare 1800 floor.
        if reduce_wall is None:
            run_timeout = max(p.timeout_s, int(c.profile_timeout_s))
        elif lower_bound:
            run_timeout = max(p.timeout_s + int(c.timeout_safety * reduce_wall), int(c.profile_timeout_s))
        else:
            run_timeout = p.timeout_s + int(c.timeout_safety * reduce_wall)
        actions.append(
            Action(
                "run",
                (st["name"],),
                p.n_jobs,
                p.workers,
                p.inner_workers,
                run_timeout,
                cap_bytes=p.cap_bytes,
                inner_cap_bytes=p.inner_cap_bytes,
                peak_rss_bytes=p.peak_rss_bytes,
                partition=p.partition,
                predicted_job_wall_s=p.predicted_job_wall_s,
                predicted_cost_usd=p.predicted_cost_usd,
                n_shards=n_sub if sharded else None,  # HFU T8: a sharded stage's run jobs skip the inline reduce
                wio=p.wio,  # S3 item 5
                gated=p.gated,  # S3 item 6
            )
        )
    return actions


# ---- reading state -------------------------------------------------------------------------------


def store_uri(bucket: str, key: str) -> str:
    return f"hf://buckets/{bucket}/{key}"


def read_plan(inst: Instance, *, bucket: str, run=subprocess.run) -> dict:
    """The instance's ``--plan`` JSON against its bucket store (read-only; the module is the authority)."""
    argv = [sys.executable, "-m", inst.module, *inst.args, "--plan", "--store", store_uri(bucket, inst.run_key)]
    p = run(argv, capture_output=True, text=True, check=False, cwd=REPO_ROOT)
    if p.returncode != 0:
        raise RuntimeError(f"{inst.id}: --plan failed (rc={p.returncode}): {(p.stderr or '')[-2000:]}")
    return json.loads(p.stdout)


def committed_history() -> dict[str, dict]:
    """Stage cost history from the committed campaign summaries (the planner learns from them)."""
    from scripts.hf_jobs.ledger import load_history

    out: dict[str, dict] = {}
    for hk, st in load_history(SUMMARY_DIR.glob("*.json")).items():
        cost = st.get("unit_cost")
        if cost:
            out[hk] = {
                "cost": UnitCost(**cost),
                "efficiency": st.get("efficiency"),
                "per_unit": None,
                "reduce_wall": st.get("reduce_wall"),  # HFU T4/B-04: carry the inline-reduce wall to next_actions
                # spec r3 D11: the stage's own measured per-job overhead (HF start to the first unit; upstream load)
                "startup_s": st.get("startup_s"),
                "upstream_load_s": st.get("upstream_load_s"),
            }
    return out


def planning_constraints(c: Campaign, *, summary_dir: pathlib.Path | None = None) -> Constraints:
    """The campaign's constraints with the job startup the committed summaries measured (spec 5.5: the
    planner learns from them); the `Constraints` default applies before any campaign measured one."""
    from scripts.hf_jobs.ledger import median_startup_s

    return c.constraints(startup_s=median_startup_s((summary_dir or SUMMARY_DIR).glob("*.json")))


#: The generation records besides its unit ledgers that `live_history` reads (spec r3 D2, D4; B r4 R4-10).
_HISTORY_RECORDS = frozenset({"_profile.json", "_reduce.json"})


def _history_names(store: Any, gen: str) -> frozenset[str]:
    """The records a generation's history is measured from, as the step's listing holds them now: its unit ledgers,
    `_profile.json` (efficiency, the reduce cost) and `_reduce.json` (the reduce wall). The freshness key of its
    cached history, so a record that lands LATE -- after the ledgers -- refreshes it (B r4 R4-10)."""
    return frozenset(n for n in store.names(gen) if n.endswith(".ledger.json") or n in _HISTORY_RECORDS)


def _history_fresh(cache: Mapping, store: Any, gen: str) -> bool:
    """True when ``cache`` holds a cost for ``gen`` measured over the records ``gen`` holds NOW (spec r3 D4)."""
    hit = cache.get(gen)
    return hit is not None and hit[0] == _history_names(store, gen)


def live_history(plan: Mapping, store: Any, cache: dict) -> dict[str, dict]:
    """Costs measured in THIS store (profile units, finished units), keyed by history key (decision P14).

    ``cache`` maps a generation to ``(its ledger names, its cost)`` and serves an entry only while the
    generation still holds exactly those ledgers (spec r3 D4). The plan counts a unit done when its result and
    `.sha256` exist, but `run_unit` writes the ledger LAST, so a step can read a stage between the two; keying on
    the ledger set re-reads it when the ledger lands, and refreshes a cost when more units finish. A stage
    with no costable ledger yet is NEVER cached (it was frozen as None for the life of a `--watch`, and the
    2474b73 `ghost_outfield` cv re-profiled 354 times)."""
    from scripts.hf_jobs.ledger import compute_wall_s, unit_cost, unit_records, upstream_load_s

    out: dict[str, dict] = {}
    for st in plan["stages"]:
        if not st.get("done"):
            continue
        gen = st["generation"]
        if not _history_fresh(cache, store, gen):
            led = _history_names(store, gen)
            recs = unit_records(store, gen)
            cost = unit_cost(recs)
            if cost is None:
                cache.pop(gen, None)
                continue
            raw_prof = store.read_bytes(f"{gen}/_profile.json")
            prof = json.loads(raw_prof) if raw_prof else {}
            eff = prof.get("efficiency")
            red = store.read_bytes(f"{gen}/_reduce.json")  # HFU T4/B-19: the inline-reduce wall, if measured
            reduce_wall = json.loads(red).get("wall_s") if red else None
            # a D7 repair record is `done` with an unknown wall; it contributes no measured per-unit cost. A
            # unit's wall excludes its upstream load (spec r3 D11 (b)): the load is a per-job overhead.
            per_unit = {
                r["key"]: float(compute_wall_s(r) or 0.0)
                for r in recs
                if r.get("outcome") == "done" and r.get("wall_s") is not None
            }
            cache[gen] = (
                led,
                {
                    "cost": cost,
                    "efficiency": eff,
                    "per_unit": per_unit,
                    "reduce_wall": reduce_wall,
                    "upstream_load_s": upstream_load_s(recs),
                    # spec r3 D2 (b): the profile's per-unit reduce cost, for the first-run reduce estimate
                    "reduce_cost": prof.get("reduce_cost"),
                },
            )
        out[st.get("history_key") or st["name"]] = cache[gen][1]
    return out


def _read_bucket_json(api: Any, bucket: str, paths: Sequence[str]) -> dict[str, Any]:
    """``{path: parsed JSON}`` for ``paths`` in the bucket, in ONE download call (a missing file is absent)."""
    import tempfile

    out: dict[str, Any] = {}
    if not paths:
        return out
    with tempfile.TemporaryDirectory() as tmp:
        dests = {path: pathlib.Path(tmp) / f"{i}.json" for i, path in enumerate(paths)}
        api.download_bucket_files(bucket, [(path, str(d)) for path, d in dests.items()], raise_on_missing_files=False)
        for path, d in dests.items():
            if d.exists():
                out[path] = json.loads(d.read_bytes())
    return out


def bucket_run_keys(api: Any, bucket: str) -> list[str]:
    """Every run key (top-level directory) of the campaign bucket: ONE non-recursive listing."""
    return sorted({e.path.split("/", 1)[0] for e in api.list_bucket_tree(bucket, prefix="", recursive=False)})


#: Concurrent store listings and ``--plan`` subprocesses per step (TF-85: run 1's step ran 42 plans serially).
STEP_WORKERS = 8

#: The sub-key under which a shared ``reads`` cache tracks in-flight single-flight computations (parallel search).
_INFLIGHT = ("__inflight__",)


def _reads_cached(reads: dict, ck: tuple, lock: threading.Lock | None, compute: Callable[[], Any]) -> Any:
    """Return ``reads[ck]``, computing it at most ONCE. Without a ``lock`` this is a plain memo (the serial path).
    With one (the parallel peak search), it is single-flight: the first caller of a key computes while the others
    wait on its event, so each store is still listed and downloaded exactly once across threads (review S3B-12).

    ``compute`` is the network I/O and CAN raise (a transient HF bucket error). An owner that raises clears its
    in-flight entry and wakes its waiters BEFORE propagating, so the error never leaves a waiter blocked forever or
    a poisoned entry in ``reads`` for later ``--watch`` steps (reviews R5-OPT-01 / R5B-12). A woken waiter re-enters
    the loop: it returns the value if the owner succeeded, else it becomes the new owner and recomputes -- so a
    transient failure recovers, and a persistent one makes each caller raise in turn (bounded, fail-loud, no hang)."""
    if lock is None:
        if ck in reads:
            return reads[ck]
        reads[ck] = val = compute()
        return val
    inflight = reads[_INFLIGHT]
    while True:
        with lock:
            if ck in reads:
                return reads[ck]
            ev = inflight.get(ck)
            owner = ev is None
            if owner:
                inflight[ck] = ev = threading.Event()
        if not owner:
            ev.wait()
            continue  # re-check: a value to return, or (the owner failed) become the new owner and recompute
        try:
            val = compute()  # OUTSIDE the lock, so other keys proceed concurrently
        except BaseException:
            with lock:  # release the waiters and clear the entry, then let the error reach this caller
                inflight.pop(ck, None)
                ev.set()
            raise
        with lock:
            reads[ck] = val
            inflight.pop(ck, None)
            ev.set()
        return val


def _entry_time(e: Any) -> float | None:
    """A listing entry's upload time as a POSIX timestamp (the HF API's ``uploaded_at``, else ``mtime``), or None
    when the source carries none (a `StoreSnapshot` lists names only)."""
    for attr in ("uploaded_at", "mtime"):
        v = getattr(e, attr, None)
        if v is not None:
            return v.timestamp() if hasattr(v, "timestamp") else float(v)
    return None


def _stage_generations(
    api: Any, bucket: str, rk: str, base: str, snaps: Mapping[str, Any]
) -> tuple[dict[str, set[str]], dict[str, float]]:
    """``({token: file names}, {token: newest ledger upload time})`` under ``<rk>/<base>``: from the step's snapshot
    of ``rk`` when it has one (no API call, the TF-85 bound; it carries no upload time), else one recursive listing of
    that stage directory only. The per-token time is the max ``uploaded_at`` over the token's ``.ledger.json`` files,
    the newest-first key of `bucket_unit_peaks`; a token with no timed ledger is absent from the time map."""
    if rk in snaps:
        rels = {rel: None for rel in snaps[rk].list_tree(base)}  # a snapshot carries no upload time
    else:
        pre = f"{rk}/{base}/"
        rels = {
            e.path[len(pre) :]: _entry_time(e)
            for e in api.list_bucket_tree(bucket, prefix=pre, recursive=True)
            if getattr(e, "type", "file") == "file"
        }
    out: dict[str, set[str]] = {}
    times: dict[str, float] = {}
    for rel, t in rels.items():
        tok, _, name = rel.partition("/")
        if name and "/" not in name:
            out.setdefault(tok, set()).add(name)
            if name.endswith(".ledger.json") and t is not None:
                times[tok] = t if tok not in times else max(times[tok], t)
    return out, times


def _read_jsons(api: Any, bucket: str, rk: str, rels: Sequence[str], snaps: Mapping[str, Any]) -> dict[str, Any]:
    """``{rel: parsed JSON}`` of files of run key ``rk``: through its step snapshot (batched by the snapshot) when it
    has one, else in one download call."""
    if rk in snaps:
        out = {}
        for rel in rels:
            data = snaps[rk].read_bytes(rel)
            if data is not None:
                out[rel] = json.loads(data)
        return out
    got = _read_bucket_json(api, bucket, [f"{rk}/{rel}" for rel in rels])
    return {rel: got[f"{rk}/{rel}"] for rel in rels if f"{rk}/{rel}" in got}


def _reduced_generations(
    api: Any, bucket: str, rk: str, base: str, snaps: Mapping[str, Any], reads: dict, lock: threading.Lock | None = None
) -> tuple[dict[str, set[str]], dict[str, Any], dict[str, float]]:
    """``({token: file names}, {token: parsed _inputs.json}, {token: newest ledger upload time})`` of the REDUCED
    generations under ``<rk>/<base>`` (both `_inputs.json` and `_summary.json` present). A run key read through the API
    is read once per controller (cached in ``reads``, single-flight under ``lock``); a snapshotted run key is read from
    the step's snapshot (no API call, never cached) and carries no upload time."""

    def compute() -> tuple[dict[str, set[str]], dict[str, Any], dict[str, float]]:
        gens, times = _stage_generations(api, bucket, rk, base, snaps)
        done = {tok: names for tok, names in gens.items() if {"_inputs.json", "_summary.json"} <= names}
        got = _read_jsons(api, bucket, rk, [f"{base}/{tok}/_inputs.json" for tok in sorted(done)], snaps)
        return (
            done,
            {tok: got[f"{base}/{tok}/_inputs.json"] for tok in done if f"{base}/{tok}/_inputs.json" in got},
            {tok: times[tok] for tok in done if tok in times},
        )

    if rk in snaps:
        return compute()
    return _reads_cached(reads, ("reduced", rk, base), lock, compute)


def _generation_peaks(
    api: Any,
    bucket: str,
    rk: str,
    gen: str,
    names: set[str],
    snaps: Mapping[str, Any],
    reads: dict,
    lock: threading.Lock | None = None,
) -> list[dict]:
    """The unit ledgers with a peak RSS of the reduced generation ``<rk>/<gen>`` (never written again, so read once
    per controller for a run key read through the API; single-flight under ``lock``)."""

    def compute() -> list[dict]:
        ledgers = sorted(f"{gen}/{n}" for n in names if n.endswith(".ledger.json"))
        return [r for r in _read_jsons(api, bucket, rk, ledgers, snaps).values() if r.get("peak_rss_bytes")]

    if rk in snaps:
        return compute()
    return _reads_cached(reads, ("ledgers", rk, gen), lock, compute)


def bucket_unit_peaks(
    api: Any,
    bucket: str,
    job: str,
    pipeline: str,
    stage: str,
    history_key: str,
    *,
    exclude: str,
    run_keys: Sequence[str] | None = None,
    snaps: Mapping[str, Any] | None = None,
    reads: dict | None = None,
    lock: threading.Lock | None = None,
) -> dict[str, int] | None:
    """S3 item 6 (MEM; owner ruling 2026-10-31, "Bucket search"): the per-unit peak RSS of the newest REDUCED
    generation of ``<pipeline>/<stage>`` in ANY run key of the campaign bucket that runs this ``job``, whose
    ``_inputs.json`` has the stage's ``history_key`` (the same computation, whatever the commit; review S3B-02: a new
    commit's campaign starts in a fresh store). Only a generation with ``_inputs.json`` and ``_summary.json`` (written
    by its reduce) counts, never ``exclude`` (the current generation, ``<run_key>/<generation>``), so a sibling tier's
    generation (another ``history_key``) or one still running never gates (review S3B-03). Newest = the latest
    finished unit. None when there is none: the planner keeps today's rule (no first-run relaxation).

    A run key names its JOB (`run_key`: ``{campaign}-{commit7}-{job}-{dataset}``), which can differ from its pipeline
    (`build_territory_ranking_census` runs `territory_ranking_census`; review S3B-11). The API cost per controller
    (review S3B-12): the run keys come from ONE root listing (`bucket_run_keys`, passed as ``run_keys``); each run
    key of the job is listed once per stage, with one `_inputs.json` download, and each matching generation's ledgers
    are downloaded once (all cached in ``reads``); a run key the step snapshotted (``snaps``) is read from its
    snapshot at no API cost."""
    from scripts._stages import _history_key

    snaps = snaps or {}
    reads = {} if reads is None else reads
    base = f"{pipeline}/{stage}"
    # A candidate is a matching REDUCED generation in another run key of the job (never ``exclude``), with the upload
    # time of its newest ledger when the listing carries one.
    candidates: list[tuple[str, str, set[str], float | None]] = []  # (run_key, generation, file names, recency)
    for rk in run_keys if run_keys is not None else bucket_run_keys(api, bucket):
        if f"-{job}-" not in rk:
            continue
        done, inputs, times = _reduced_generations(api, bucket, rk, base, snaps, reads, lock)
        for tok in sorted(done):
            gen = f"{base}/{tok}"
            inp = inputs.get(tok)
            if f"{rk}/{gen}" == exclude or inp is None or _history_key(inp) != history_key:
                continue
            candidates.append((rk, gen, done[tok], times.get(tok)))
    if not candidates:
        return None

    def _peaks(rk: str, gen: str, names: set[str]) -> dict[str, int]:
        recs = _generation_peaks(api, bucket, rk, gen, names, snaps, reads, lock)
        return {str(r["key"]): int(r["peak_rss_bytes"]) for r in recs}

    # Newest-first (HFU canary run 2): when every candidate carries a listing upload time, take the one whose latest
    # ledger uploaded last and download ONLY its ledgers -- never the losers' (the 591 s win_probability search kept
    # one of 13 generations). This ASSUMES upload order == finish order, which holds when the candidate generations
    # are distinct earlier campaigns' run keys (the usual case). It can differ for concurrent generations of the
    # same computation -- INV-1 replica run keys (`-r<N>`) share a history_key and run at once (review R5-OPT-02) --
    # where newest-by-upload may pick a different replica than newest-by-finished_ns. Both are valid peak samples:
    # the peaks only size the memory gate (worker count W), and the output stays byte-identical per (W, N) by the
    # ADR-128 R4 identity gates, so either choice is sound (the old finished_ns scan was equally arbitrary there).
    if all(c[3] is not None for c in candidates):
        for rk, gen, names, _t in sorted(candidates, key=lambda c: c[3], reverse=True):  # type: ignore[arg-type,return-value]
            if peaks := _peaks(rk, gen, names):
                return peaks
        return None
    # Fallback: a snapshotted candidate carries no upload time; keep the exact ``finished_ns`` scan (download every
    # candidate, the newest finished unit wins), so the result is unchanged where the metadata is absent.
    best: tuple[int, dict[str, int]] | None = None
    for rk, gen, names, _t in candidates:
        recs = _generation_peaks(api, bucket, rk, gen, names, snaps, reads, lock)
        newest = max((int(r.get("finished_ns") or 0) for r in recs), default=-1)
        if recs and (best is None or newest > best[0]):
            best = (newest, {str(r["key"]): int(r["peak_rss_bytes"]) for r in recs})
    return best[1] if best else None


def _gate_candidate(st: Mapping) -> bool:
    """A stage the memory gate can shape: unreduced, with a generation, not inner_parallel and not thread_safe."""
    return (
        bool(st.get("generation")) and not st.get("reduced") and not (st.get("inner_parallel") or st.get("thread_safe"))
    )


def unit_peak_views(
    insts: Sequence[Instance],
    plans: Mapping[str, Any],
    snaps: Mapping[str, Any],
    *,
    api: Any,
    bucket: str,
    cache: dict,
    workers: int = STEP_WORKERS,
) -> dict[str, dict[str, dict[str, int]]]:
    """``{id: {stage: per-unit peaks}}`` (S3 item 6) for every instance with a plan, from `bucket_unit_peaks`.
    ``cache`` maps ``(run_key, generation)`` to its answer: a reduced earlier generation is never written again, so
    each stage generation's history is read once per controller and served from the cache on every later
    ``--watch`` step. The bucket's run keys are listed once per controller (the history is earlier campaigns'), and
    every earlier store's reads are shared by all instances (``cache["_reads"]``; review S3B-12). The misses are
    searched in parallel (HFU canary run 2: ~40 serial per-stage searches took ~5 min); a single-flight lock on the
    shared ``_reads`` keeps each store listed and downloaded once (`_reads_cached`)."""
    from concurrent.futures import ThreadPoolExecutor

    # Each gate-candidate stage's search key -> (job, pipeline, stage, history_key, exclude); and the per-instance
    # layout that reassembles the result after the parallel search.
    want: dict[tuple[str, str], tuple[str, str, str, str, str]] = {}
    layout: dict[str, list[tuple[str, tuple[str, str]]]] = {}
    for inst in insts:
        plan = plans[inst.id]
        if isinstance(plan, RuntimeError):
            continue
        rows: list[tuple[str, tuple[str, str]]] = []
        for st in plan["stages"]:
            if not _gate_candidate(st):
                continue
            ck = (inst.run_key, st["generation"])
            rows.append((st["name"], ck))
            if ck not in cache and ck not in want:
                excl = f"{inst.run_key}/{st['generation']}"
                want[ck] = (inst.job, plan["pipeline"], st["name"], st["history_key"], excl)
        layout[inst.id] = rows
    if want:
        if "_run_keys" not in cache:
            cache["_run_keys"] = bucket_run_keys(api, bucket)
        reads = cache.setdefault("_reads", {})
        reads.setdefault(_INFLIGHT, {})  # created single-threaded, before the pool shares it
        lock = threading.Lock()

        def one(item: tuple[tuple[str, str], tuple[str, str, str, str, str]]) -> tuple[tuple[str, str], Any]:
            ck, (job, pipeline, stage, history_key, exclude) = item
            return ck, bucket_unit_peaks(
                api,
                bucket,
                job,
                pipeline,
                stage,
                history_key,
                exclude=exclude,
                run_keys=cache["_run_keys"],
                snaps=snaps,
                reads=reads,
                lock=lock,
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for ck, res in pool.map(one, list(want.items())):  # cache written single-threaded, as `read_plans` does
                cache[ck] = res
    return {inst_id: {name: cache[ck] for name, ck in rows if cache.get(ck)} for inst_id, rows in layout.items()}


# ---- bulk step reads (TF-85) ---------------------------------------------------------------------


def _tree_identity() -> str:
    """The code identity of the tree ``--plan`` runs in, computed as `read_plan` computes it: the commit on a
    clean tree, ``dirty-<tree hash>`` on a dirty one, a fresh nonce when unknown (so no plan is reused). It runs
    where `read_plan` runs ``--plan`` (``cwd=REPO_ROOT``), because `_provenance`'s git calls follow the process
    working directory. The plan cache keys on it, so a tree edit during a campaign (D1) re-plans every
    instance."""
    code = "from scripts._provenance import code_identity, git_provenance; print(code_identity(git_provenance()))"
    argv = [sys.executable, "-c", code]
    p = subprocess.run(argv, capture_output=True, text=True, check=True, cwd=REPO_ROOT)  # noqa: S603
    return p.stdout.strip()


def store_snapshots(
    run_keys: Iterable[str], *, bucket: str, api: Any, content: dict, workers: int = STEP_WORKERS
) -> dict[str, Any]:
    """``{run_key: StoreSnapshot}`` for this step: each DISTINCT store listed once, the listings in parallel.
    Variants that share a run key share its snapshot. ``content`` (bytes by xet hash) carries across steps and
    keeps only the hashes these listings still hold, so it never outgrows the stores (``run_keys`` must be every
    store of the step)."""
    from concurrent.futures import ThreadPoolExecutor

    from scripts.hf_jobs.store import BucketStore, StoreSnapshot

    snaps = {rk: StoreSnapshot(BucketStore(bucket, rk, api=api), content) for rk in dict.fromkeys(run_keys)}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda s: s.digest(), snaps.values()))  # forces each listing in parallel
    keep = set().union(*(s.xet_hashes() for s in snaps.values())) if snaps else set()
    for xh in [k for k in content if k not in keep]:
        del content[xh]
    return snaps


def read_plans(
    insts: Sequence[Instance],
    snaps: Mapping[str, Any],
    *,
    bucket: str,
    cache: dict,
    tree: str = "",
    read: Callable[..., dict] = read_plan,
    workers: int = STEP_WORKERS,
) -> dict[str, dict | RuntimeError]:
    """Every instance's ``--plan`` (the module stays the authority), run in parallel, and ONLY for an instance
    whose store listing or working tree changed since its last plan: a plan is a function of the tree (``tree``:
    `_tree_identity`) and of its store (its snapshot digest). A failed plan is returned as its `RuntimeError` and
    is not cached. The step gets a deep copy, because it may derive views from the plan and the cached plan must
    stay as read."""
    from concurrent.futures import ThreadPoolExecutor

    plans: dict[str, tuple[tuple[str, str], dict]] = cache.setdefault("plans", {})  # id -> ((digest, tree), plan)
    key = {i.id: (snaps[i.run_key].digest(), tree) for i in insts}
    todo = [i for i in insts if i.id not in plans or plans[i.id][0] != key[i.id]]

    def one(inst: Instance) -> dict | RuntimeError:
        try:
            return read(inst, bucket=bucket)
        except RuntimeError as exc:
            return exc

    failed: dict[str, RuntimeError] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for inst, got in zip(todo, pool.map(one, todo), strict=True):
            if isinstance(got, RuntimeError):
                failed[inst.id] = got
                plans.pop(inst.id, None)
            else:
                plans[inst.id] = (key[inst.id], got)
    return {i.id: failed[i.id] if i.id in failed else copy.deepcopy(plans[i.id][1]) for i in insts}


def instance_views(
    insts: Sequence[Instance],
    plans: Mapping[str, Any],
    snaps: Mapping[str, Any],
    *,
    cache: dict,
    workers: int = STEP_WORKERS,
) -> dict[str, dict]:
    """``{id: live history}`` for every instance with a successful plan, read in PARALLEL (the old step read the
    instances one after another). ``cache`` is `live_history`'s per-generation cache, NAMESPACED by run key
    (``{run_key: {generation: cost}}``): costs are measured per store, so a replica -- a fresh store with the
    same generation token as its main run -- never reads the main run's measured costs (NF7). Two variants that
    share a run key share its sub-cache, so a generation they both hold is read ONCE, by the first variant (input
    order) that needs it, then served from the cache -- the same values the serial step produced."""
    from concurrent.futures import ThreadPoolExecutor

    rows = [(i, plans[i.id], snaps[i.run_key]) for i in insts if not isinstance(plans[i.id], RuntimeError)]
    first: dict[tuple[str, str], int] = {}  # (run_key, generation) -> the first row (input order) that reads it
    for n, (inst, plan, store) in enumerate(rows):
        for st in plan["stages"]:
            gen = st.get("generation")
            if st.get("done") and gen is not None and not _history_fresh(cache.get(inst.run_key, {}), store, gen):
                first.setdefault((inst.run_key, gen), n)

    def one(n: int) -> tuple[str, dict]:
        inst, plan, store = rows[n]
        mine = [
            st for st in plan["stages"] if st.get("done") and first.get((inst.run_key, st.get("generation", ""))) == n
        ]
        filled: dict = {}
        if mine:  # only the generations this row is the first in its store to need
            live_history({**plan, "stages": mine}, store, filled)
        return inst.run_key, filled

    with ThreadPoolExecutor(max_workers=workers) as pool:
        got = list(pool.map(one, range(len(rows))))
    for run_key, filled in got:  # each (run_key, generation) has exactly one filler
        cache.setdefault(run_key, {}).update(filled)
    # every done generation of every view is cached in its run key's sub-cache now, so this reads nothing
    return {i.id: live_history(plan, store, cache.setdefault(i.run_key, {})) for i, plan, store in rows}


# ---- applying the decision -----------------------------------------------------------------------


def lpt_allowed(inst: Instance) -> bool:
    """An LPT partition names its keys, so every dataset must be mounted: a ``download`` dataset fetches
    only the units of its round-robin ``--shard i/N`` (`_job_specs` refuses a partition there)."""
    return not any(d.access == "download" for d in inst.datasets)


def _error_history(jobs: Iterable[Any]) -> dict[tuple[str, str], list[Any]]:
    """Every job per (run_key, generation), newest first (spec r4 D17): a job counts for EVERY generation it carries,
    whoever launched it, so the instances that share a generation read ONE history (the resume after an ERROR, the
    D5 zero-progress guard, the D6(c) loss note) and reach the same verdict in the same step."""
    from scripts.hf_jobs.launch import gens_of

    out: dict[tuple[str, str], list[Any]] = {}
    for info in jobs:
        lab = dict(getattr(info, "labels", None) or {})
        for g in dict.fromkeys(gens_of(lab, job=str(getattr(info, "id", "?")))):
            out.setdefault((lab.get("run_key", ""), g), []).append(info)
    for v in out.values():
        v.sort(key=lambda j: getattr(j, "created_at", None) or 0, reverse=True)
    return out


def _trailing_errors(history: Sequence[Any]) -> int:
    """How many of the stage's newest jobs ended in ERROR in a row (a COMPLETED job ends the run). A
    never-started timeout is a dispatch overflow, NOT a run error, so it is skipped (exempt from the
    error_resumes bound; the dispatch back-off handles it -- §D)."""
    n = 0
    for info in history:
        if job_never_started_timeout(info):
            continue
        if str(getattr(getattr(info, "status", None), "stage", "")) != "ERROR":
            break
        n += 1
    return n


def _timed_out(info: Any) -> bool:
    return "timeout" in str(getattr(getattr(info, "status", None), "message", "") or "").lower()


# ---- dispatch-concurrency overflow (spec §D) -----------------------------------------------------


def job_never_started(info: Any) -> bool:
    """True when a job's container NEVER STARTED (``started_at`` present and None) -- a HF infra fault BEFORE the
    run (a volume-mount failure, a never-started timeout, a scheduling failure), as opposed to a run that started
    and then failed. A deterministic crash runs first, so its ``started_at`` is set. hf_hub's ``JobInfo`` always
    SETS ``started_at`` (None when the API omits it), so on a real job this reads the value; the attribute-missing
    branch => treated as STARTED (B23 fail-safe) only guards a non-``JobInfo`` object, never never-started on doubt."""
    sentinel = object()
    started = getattr(info, "started_at", sentinel)
    if started is sentinel:  # attribute absent -> fail safe to a started job
        return False
    return started is None


def job_never_started_timeout(info: Any) -> bool:
    """True when a job TIMED OUT and its container never started -- a dispatch overflow (the account's concurrency
    ceiling), exempt from ``error_resumes`` and handled by the dispatch back-off (§D). A non-timeout never-start
    (e.g. a ``Volume mount failed`` infra fault) is NOT a dispatch overflow: it is COUNTED toward ``error_resumes``
    and resumed by ``resume_after_error``, so a transient mount failure retries and a persistent one halts."""
    return _timed_out(info) and job_never_started(info)


def _created_epoch(info: Any) -> float:
    """A null-safe sort key over ``created_at`` (datetime, int, or None -- hf_hub's field is Optional). None
    sorts oldest. Never mixes datetime and int in one comparison (B49: ``created_at or 0`` would TypeError)."""
    c = getattr(info, "created_at", None)
    if c is None:
        return float("-inf")
    return c.timestamp() if hasattr(c, "timestamp") else float(c)


def consecutive_never_started(jobs: Iterable[Any]) -> int:
    """The campaign's LEADING never-started timeouts (newest first); any other terminal state (a COMPLETED,
    a started-timeout, a crash) ends the run. Re-derived from job history each stateless step (no memory)."""
    ordered = sorted(jobs, key=_created_epoch, reverse=True)
    n = 0
    for info in ordered:
        if not job_never_started_timeout(info):
            break
        n += 1
    return n


def effective_dispatch_cap(*, dispatch_cap: int, dispatch_floor: int, never_started: int) -> int:
    """The live dispatch budget after backing off one slot per leading never-started timeout, never below the
    floor (a stage's wave is never throttled below its parallelism; §D, r6 consider 21)."""
    return max(dispatch_floor, dispatch_cap - max(0, never_started))


def dispatch_overflow_halt(*, never_started: int, dispatch_cap: int, dispatch_floor: int, m: int) -> bool:
    """After the back-off reaches the floor, ``m`` MORE consecutive never-started timeouts halt + surface the
    real ceiling (owner ruling B24: an HF ceiling below shard_cap is surfaced, not degraded to slow waves)."""
    return never_started >= (dispatch_cap - dispatch_floor) + m


def observed_max_concurrency(jobs: Iterable[Any]) -> int:
    """The peak number of job containers running at once (``started_at``..``finished_at`` overlap); a
    never-started job never ran, so it adds nothing. Written to the campaign summary (probe-first, §D)."""
    events: list[tuple[float, int]] = []
    for info in jobs:
        started = getattr(info, "started_at", None)
        if started is None:
            continue
        finished = getattr(info, "finished_at", None)
        events.append((started.timestamp(), 1))
        events.append((finished.timestamp() if finished is not None else float("inf"), -1))
    peak = cur = 0
    for _, delta in sorted(events, key=lambda e: (e[0], -e[1])):  # a start at t precedes an end at t
        cur += delta
        peak = max(peak, cur)
    return peak


# ONLY transient concurrency-limit phrases degrade. A quota/billing/rate-limit/ambiguous reject is NOT a
# "slot frees next step" condition -- it would silent-loop (degrade every step, no counter, no halt), so it
# must fail loud instead (B48). HTTP 429 (Too Many Requests) is the transient status signal.
_CAPACITY_PHRASES = ("too many concurrent", "maximum number of jobs", "concurrent job limit")


def _is_capacity_reject(exc: BaseException) -> bool:
    """True ONLY for a RECOGNIZED TRANSIENT concurrency reject (HTTP 429, or a concurrency-limit phrase) -- a
    slot will free as running jobs finish, so the controller degrades (stops launching this step) and retries
    next step. A quota/billing/ambiguous reject returns False so the caller re-raises and fails loud, never
    degrades into a silent loop (B48; B26 g/h; consider-09)."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    return any(p in msg for p in _CAPACITY_PHRASES)


def errors_exhausted(stage: str, history: Sequence[Any], c: Constraints) -> str | None:
    """The halt reason when the stage's latest ``c.error_resumes`` jobs all ended in ERROR, else None. It needs no unit
    record (a timeout or a crash before any unit writes none). Owner ruling 2026-10-31 (HFU rung 5): `step` treats it
    as an INSTANCE halt, named, whose live jobs are cancelled -- like the failed-units halt."""
    errors = _trailing_errors(history)
    return f"{stage}: {errors} job(s) in a row ended in ERROR" if errors >= c.error_resumes else None


def resume_after_error(plan: Mapping, stage: str, history: Sequence[Any], c: Constraints) -> str | None:
    """None when the next action may follow a stage whose latest job ended in ERROR, else the halt reason that
    halts the INSTANCE ("" = the caller's default reason; `step` then cancels the instance's live siblings, HFU
    rung 5). It may when that stage still has a retryable failed unit (a strict
    stage's job ends in ERROR on any failed unit; the retry stays bounded by ``unit_attempts``), when the
    job ran out of time OR its container never started (a ``Volume mount failed`` infra fault) with units pending
    (finished units persist; a transient fault retries, bounded by ``error_resumes``), or when NOTHING is
    pending: every unit is settled, so the job died in the inline reduce and the next action is that reduce
    (spec r3 D3: the 2474b73
    `win_probability` run job timed out in its reduce and the instance halted for 5 h). EVERY such resume is
    also bounded by ``error_resumes`` ERRORs in a row, which needs no unit record: a timeout or a crash before
    any unit writes none, so the attempt count alone would resume forever (Reviewer B CCHC-IMPL-09)."""
    st = next((s for s in plan["stages"] if s["name"] == stage), None)
    if st is None:
        return ""
    if exhausted := errors_exhausted(stage, history, c):
        return exhausted
    if any(int(n) < c.unit_attempts for n in (st.get("failed_units") or {}).values()):
        return None
    if history and (job_never_started(history[0]) or _timed_out(history[0])) and st.get("pending_units"):
        return None
    if not st.get("pending_units"):
        return None
    return ""


#: Seconds a COMPLETED job must have been finished before the zero-progress guard judges it: the records it wrote
#: can reach the bucket listing a step late (spec r3 D5; B plan review PLAN-R3-08).
PROGRESS_SETTLE_S = 600.0
#: The committed records a job can add to a stage (spec r3 D5): a unit result, an exclusion marker, a failure
#: record, the stage output, or a sub-reduce partial. Ledgers, sidecars and stage metadata only accompany them.
_UNIT_RECORD_SUFFIXES = frozenset({".parquet", ".json", ".fileset.json", ".excluded.json", ".failed.json"})
_RESULT_SUFFIXES = frozenset({".parquet", ".json", ".fileset.json"})


def progress_mark_of(names_by_dir: Mapping[str, Iterable[str]], plan: Mapping, stages: Sequence[str]) -> str:
    """The zero-progress guard's measure (spec r3 D5): a fingerprint of the committed records of ``stages`` -- every
    unit result, exclusion marker, failure record WITH its attempt number (from the plan), sub-reduce partial and
    stage output. Any committed work changes it: a new record; a removed one (a retried unit's failure record
    replaced by its result); a failure record rewritten by a later attempt. A record COUNT stands still across the
    last two (B r4 R4-02), so a count would read real work as a stall. ``names_by_dir`` maps a directory (relative
    to the run key) to its entry names, as the step's listing holds them."""
    by_name = {s["name"]: s for s in plan["stages"]}
    items: list[str] = []
    for name in stages:
        st = by_name.get(name) or {}
        gen = st.get("generation")
        if not gen:
            continue
        for n in names_by_dir.get(gen, ()):
            stem, _, rest = n.partition(".")
            if stem and f".{rest}" in _UNIT_RECORD_SUFFIXES and (not stem.startswith("_") or stem == "_output"):
                items.append(f"{gen}/{n}")
        for n in names_by_dir.get(f"{gen}/_subreduce", ()):
            stem, _, rest = n.partition(".")
            if stem.isdigit() and f".{rest}" in _RESULT_SUFFIXES:
                items.append(f"{gen}/_subreduce/{n}")
        for key, attempt in sorted((st.get("failed_units") or {}).items()):
            items.append(f"{gen}/{key}#attempt={int(attempt)}")
    return hashlib.sha256("\n".join(sorted(items)).encode("utf-8")).hexdigest()[:16]


def _progress_mark(store: Any, plan: Mapping, stages: Sequence[str]) -> str:
    """`progress_mark_of` over this step's listing (``store`` is the step's `StoreSnapshot`)."""
    by_name = {s["name"]: s for s in plan["stages"]}
    dirs: dict[str, list[str]] = {}
    for name in stages:
        gen = (by_name.get(name) or {}).get("generation")
        if gen:
            dirs[gen] = list(store.names(gen))
            dirs[f"{gen}/_subreduce"] = list(store.names(f"{gen}/_subreduce"))
    return progress_mark_of(dirs, plan, stages)


def _zero_progress(stage: str, history: Sequence[Any], mark: str, now: Any) -> str | None:
    """The halt reason when a stage's 2 most recent SETTLED completed jobs both launched at the stage's CURRENT
    progress mark (their ``records`` label): neither of them, nor anything after them, committed any work (spec r3
    D5: the 2474b73 `ghost_outfield` cv re-launched a no-op job every step for 6.5 h). A job is settled once it
    finished at least ``PROGRESS_SETTLE_S`` ago, so every record it wrote has landed in the listing; a job inside the
    settle window is IGNORED, never waited for. So a loop that relaunches every step halts about one settle window
    plus two jobs after it starts (B r4 R4-01: requiring the NEWEST job to be settled let a loop run forever), and a
    job whose records land late is never judged before they land. Stateless: the baseline rides the job labels, so a
    restarted controller judges the same evidence the same way. None otherwise."""
    settled = sorted(
        (
            j
            for j in history
            if str(getattr(getattr(j, "status", None), "stage", "") or "") == "COMPLETED"
            and getattr(j, "finished_at", None) is not None
            and (now - j.finished_at).total_seconds() >= PROGRESS_SETTLE_S
        ),
        key=lambda j: j.finished_at,
        reverse=True,
    )
    if len(settled) < 2:
        return None
    newest = settled[:2]
    if any((getattr(j, "labels", None) or {}).get("records") != mark for j in newest):
        return None
    ids = ", ".join(str(j.id) for j in newest)
    return f"{stage}: zero progress -- its 2 most recent settled jobs ({ids}) committed nothing (progress mark {mark})"


def _lost_after_completed(stage: str, history: Sequence[Any], pending: int) -> str | None:
    """Spec r3 D6(c): the note when a stage's NEWEST job is a COMPLETED run job yet units are still pending -- the
    outputs it wrote did not persist (2474b73: one job's mount lost 73 units' files). The units are simply re-run;
    the note makes a storage fault visible instead of silently absorbed. A profile leaves units pending by design,
    so only a run job counts; jobs launched before the ``kind`` label read as run jobs. None otherwise."""
    if not history or pending <= 0:
        return None
    newest = history[0]
    if str(getattr(getattr(newest, "status", None), "stage", "") or "") != "COMPLETED":
        return None
    if (getattr(newest, "labels", None) or {}).get("kind", "run") != "run":
        return None
    return (
        f"{stage}: {pending} unit(s) still pending after completed run job {newest.id}; their outputs did not persist"
    )


def halt_report(a: Action) -> dict:
    """The step report of a ``halt`` action: nothing is launched; the reason names the failed units."""
    return {"action": "halt", "stages": a.stages, "jobs": [], "failed": a.reason}


def _job_specs(
    c: Campaign,
    commit: str,
    inst: Instance,
    plan: Mapping,
    a: Action,
    *,
    store: Any,
    unit_peaks: Mapping[str, Mapping[str, int]] | None = None,
) -> list[Any]:
    from scripts.hf_jobs.launch import JobSpec

    if a.partition is not None and any(d.access == "download" for d in inst.datasets):
        raise ValueError(f"{inst.id}: an LPT partition needs mounted datasets (download access fetches by shard)")
    gens = _plan_gens(plan)
    peaks_path: str | None = None
    if a.gated and a.kind == "run":  # S3 item 6 (MEM): stage the per-unit peaks once; every job of the run reads them
        st0 = next(s for s in plan["stages"] if s["name"] == a.stages[0])
        peaks = (unit_peaks or {}).get(st0["name"])
        if not peaks:
            raise ValueError(f"{inst.id}: the gated run of {st0['name']} has no per-unit peaks to stage")
        rel = f"_unit_peaks/{plan['pipeline']}/{st0['name']}/{st0['token']}.json"
        store.upload_bytes(rel, json.dumps(dict(peaks), sort_keys=True).encode("utf-8"))
        peaks_path = f"/scratch/{inst.run_key}/{rel}"

    def spec(shard: tuple[int, int], shard_keys_path: str | None = None) -> JobSpec:
        return JobSpec(
            campaign=c.name,
            run_key=inst.run_key,
            job=inst.job,
            tier=inst.tier,
            variant=inst.variant,
            stages=a.stages,
            shard=shard,
            module=inst.module,
            args=inst.args,
            commit=commit,
            bucket=c.bucket,
            workers=a.workers,
            inner_workers=a.inner_workers,
            cap_bytes=a.cap_bytes,
            inner_cap_bytes=a.inner_cap_bytes,
            peak_rss_bytes=a.peak_rss_bytes,
            timeout_s=a.timeout_s,
            attempts=c.attempts,
            datasets=inst.datasets,
            extras=(),
            profile=a.profile,
            shard_keys_path=shard_keys_path,
            escalate=a.escalate,
            wio=a.wio,
            unit_peaks_path=peaks_path,  # S3 item 6
            subreduce_shard_id=(a.shard_id if a.kind == "subreduce" else None),  # HFU T8
            subreduce_n_shards=(a.n_shards if a.kind == "subreduce" else None),  # HFU T8: the worker's shard count
            from_partials=(a.n_shards if a.kind == "reduce" else None),
            no_inline_reduce=(a.kind == "run" and a.n_shards is not None),
            # spec r3 D5: the zero-progress baseline, of the action's FIRST generation (spec r4 D17)
            records=_progress_mark(store, plan, a.stages[:1]),
            kind=a.kind,  # spec r3 D6(c): tells a completed run job from a profile
            gens=tuple(gens[s] for s in a.stages),  # spec r4 D17: work is keyed by generation, whoever launched it
        )

    if a.partition is None:
        return [spec((i, a.n_jobs)) for i in range(a.n_jobs)]
    st = next(s for s in plan["stages"] if s["name"] == a.stages[0])
    n = len(a.partition)
    specs = []
    for i, part in enumerate(a.partition):
        rel = f"_partitions/{plan['pipeline']}/{st['name']}/{st['token']}/{i}of{n}.json"
        store.upload_bytes(rel, json.dumps(list(part)).encode("utf-8"))
        specs.append(spec((i, n), f"/scratch/{inst.run_key}/{rel}"))
    return specs


def _live_gens(jobs: Iterable[Any]) -> dict[str, set[str]]:
    """Spec r4 D17: ``{run_key: {generation}}`` that LIVE jobs carry, whoever launched them. A stage is in flight when
    any live job carries its generation, so no instance launches work another instance's job already runs."""
    from scripts.hf_jobs.launch import TERMINAL, gens_of

    live: dict[str, set[str]] = {}
    for info in jobs:
        if str(getattr(getattr(info, "status", None), "stage", "") or "") in TERMINAL:
            continue
        lab = dict(getattr(info, "labels", None) or {})
        live.setdefault(lab.get("run_key", ""), set()).update(gens_of(lab, job=str(getattr(info, "id", "?"))))
    return live


def _gen(st: Mapping) -> str:
    """A plan stage's generation as job labels carry it: its token's prefix (spec r4 D17)."""
    from scripts.hf_jobs.launch import GEN_PREFIX_LEN

    return str(st["token"])[:GEN_PREFIX_LEN]


def _plan_gens(plan: Mapping) -> dict[str, str]:
    """``{stage name: generation}`` of one plan (spec r4 D17)."""
    return {st["name"]: _gen(st) for st in plan["stages"]}


def refuse_prefix_collisions(insts: Sequence[Instance], plans: Mapping[str, Any]) -> None:
    """Spec r4 D17: jobs carry a generation PREFIX, so two generations of one store (run key) that share a prefix would
    alias each other's jobs (in flight, history, cancel). Refuse the step, naming them. A generation that several
    plans share is ONE generation, not a collision."""
    seen: dict[tuple[str, str], set[str]] = {}
    for inst in insts:
        plan = plans[inst.id]
        if isinstance(plan, RuntimeError):
            continue
        for st in plan["stages"]:
            seen.setdefault((inst.run_key, _gen(st)), set()).add(str(st.get("generation") or st["token"]))
    clash = {f"{rk}:{prefix}": sorted(gens) for (rk, prefix), gens in seen.items() if len(gens) > 1}
    if clash:
        raise RuntimeError(
            f"generation prefix collision (spec r4 D17): these generations of one store share a job-label prefix, so "
            f"the controller could not tell their jobs apart; refusing the step: {clash}"
        )


def _active_instances(insts: Sequence[Instance], plans: Mapping[str, Any], live: Mapping[str, set[str]]) -> int:
    """HFU T7 + spec r4 D17: how many instances take a share of the step budget -- those with a ready, unreduced stage
    whose generation no live job carries and no instance before them in input order will launch this step. An
    instance whose ready stages are all in flight waits, so it takes no share."""
    claimed = {rk: set(g) for rk, g in live.items()}
    n = 0
    for inst in insts:
        plan = plans[inst.id]
        if isinstance(plan, RuntimeError):
            continue
        mine = {
            _gen(st)
            for st in plan["stages"]
            if st.get("ready") and not st.get("reduced") and st.get("n_units") is not None
        } - claimed.setdefault(inst.run_key, set())
        if mine:
            n += 1
            claimed[inst.run_key] |= mine
    return n


def _cancel_scope(
    inst: Instance,
    insts: Sequence[Instance],
    plans: Mapping[str, Any],
    halting: set[str],
    last_gens: Mapping[str, frozenset[str]],
) -> tuple[dict[str, list[str]], frozenset[str] | None]:
    """Spec r4 D17: which of a halting instance's jobs it must NOT cancel now. ``keep``: each generation that another
    instance of its run key, not halting, still needs (unreduced in its plan) -> those instance ids. ``defer``: the
    generations an instance whose plan errored this step held in its LAST plan (None: no last plan is known, so any
    job may be shared); those cancels wait for the next step."""
    keep: dict[str, list[str]] = {}
    defer: set[str] | None = set()
    for other in insts:
        if other.run_key != inst.run_key or other.id == inst.id or other.id in halting:
            continue
        plan = plans[other.id]
        if isinstance(plan, RuntimeError):
            held = last_gens.get(other.id)
            defer = None if (defer is None or held is None) else defer | held
            continue
        for st in plan["stages"]:
            if not st.get("reduced"):
                keep.setdefault(_gen(st), []).append(other.id)
    return keep, (None if defer is None else frozenset(defer))


def step(
    c: Campaign,
    commit: str,
    *,
    api: Any,
    token: str,
    replica: int | None = None,
    dry_run: bool = False,
    only: str | None = None,
    cache: dict | None = None,
) -> dict:
    """ONE stateless advance of every instance (spec 5.5). ``only="profile"`` launches profile actions only."""
    from scripts.hf_jobs import launch as launcher
    from scripts.hf_jobs.preflight import require_preflight
    from scripts.hf_jobs.store import BucketStore

    require_preflight(c.name, commit, c.sha256, store=BucketStore(c.bucket, "", api=api))
    tree = _tree_identity()
    if tree.startswith("dirty-"):  # D1: a dirty tree makes every `--plan` token `dirty-<hash>`, so the computed
        raise RuntimeError(  # generations miss the stores written under the committed tokens -- refuse, launch nothing
            f"campaign {c.name}: refusing to run on a dirty tree ({tree}); commit or stash the working tree first"
        )
    cache = {} if cache is None else cache
    all_jobs = list(launcher.jobs_for(api, campaign=c.name, commit=commit))
    # Spec r4 D17: work is keyed by (run_key, generation), whoever launched it -- in flight from the live jobs'
    # `gens` labels, history from every job that carries a generation.
    live = _live_gens(all_jobs)
    by_gen = _error_history(all_jobs)
    hist = committed_history()
    cons = planning_constraints(c)
    insts = instances(c, commit, replica=replica)
    # Dispatch-overflow back-off (§D), re-derived from job history each stateless step: a never-started
    # timeout is an overflow, not a run error -- reduce the live budget toward the floor. Past the floor plus
    # dispatch_halt_after more, surface the real ceiling and halt (owner ruling B24).
    never_started = consecutive_never_started(all_jobs)
    obs_concurrency = observed_max_concurrency(all_jobs)
    if dispatch_overflow_halt(
        never_started=never_started,
        dispatch_cap=c.dispatch_cap,
        dispatch_floor=cons.dispatch_floor,
        m=cons.dispatch_halt_after,
    ):
        reason = (
            f"dispatch overflow: {never_started} consecutive never-started timeouts past the floor "
            f"{cons.dispatch_floor} (the HF ceiling is below shard_cap={c.shard_cap}); observed max "
            f"concurrency {obs_concurrency}"
        )
        return {
            i.id: {
                "action": "halt",
                "stages": (),
                "jobs": [],
                "failed": reason,
                "observed_max_concurrency": obs_concurrency,
            }
            for i in insts
        }
    # Spec §C (owner ruling 2026-10-30): `dispatch_cap` caps total JOBS in flight, so every live job of the campaign
    # holds a slot (2474b73 subtracted one per running stage, however many jobs it ran).
    budget = effective_dispatch_cap(
        dispatch_cap=c.dispatch_cap, dispatch_floor=cons.dispatch_floor, never_started=never_started
    ) - sum(1 for j in all_jobs if str(getattr(getattr(j, "status", None), "stage", "") or "") not in launcher.TERMINAL)
    snaps = store_snapshots(
        [i.run_key for i in insts], bucket=c.bucket, api=api, content=cache.setdefault("content", {})
    )
    plans = read_plans(insts, snaps, bucket=c.bucket, cache=cache, tree=tree, read=read_plan)
    refuse_prefix_collisions(insts, plans)
    # Spec r4 D17: the generations each instance's LAST good plan held, so a halting instance can tell which of its
    # jobs an instance whose plan errored this step may still need (the cancel waits for those).
    last_gens: dict[str, frozenset[str]] = cache.setdefault("last_gens", {})
    for inst in insts:
        if not isinstance(read := plans[inst.id], RuntimeError):
            last_gens[inst.id] = frozenset(_plan_gens(read).values())
    views = instance_views(insts, plans, snaps, cache=cache.setdefault("live_history", {}))
    peak_views = unit_peak_views(
        insts, plans, snaps, api=api, bucket=c.bucket, cache=cache.setdefault("unit_peaks", {})
    )
    report: dict[str, dict] = {}

    # HFU T7: each ACTIVE instance gets a FAIR SLICE of the step's dispatch budget, so one instance's
    # slot-filling cannot take every free slot (B-08). Active (spec r4 D17): the instance has a ready stage that no
    # live job and no instance before it holds; the share is known BEFORE next_actions (no circular dependency),
    # and caps the instance's TOTAL launches.
    share = max(1, budget // max(1, _active_instances(insts, plans, live)))
    halting: list[Instance] = []  # spec r4 D17: cancelled after every instance decided (see `_cancel_scope`)
    for inst in insts:
        plan = plans[inst.id]
        if isinstance(plan, RuntimeError):
            report[inst.id] = {"action": None, "jobs": [], "failed": str(plan)}
            continue
        store = snaps[inst.run_key]
        gens = _plan_gens(plan)
        flying = live.setdefault(inst.run_key, set())  # grows as this step launches (spec r4 D17)
        acts = next_actions(
            plan,
            {name for name, g in gens.items() if g in flying},
            _merge_history(hist, views[inst.id]),
            cons,
            lpt=lpt_allowed(inst),
            free_slots=share,
            unit_peaks=peak_views.get(inst.id),
        )
        if not acts:
            done = all(s.get("reduced") for s in plan["stages"])
            report[inst.id] = {"action": "done" if done else "waiting", "jobs": [], "failed": None}
            continue
        halt = next((a for a in acts if a.kind == "halt"), None)
        if halt is not None:  # a halting stage halts the INSTANCE; its live jobs are cancelled after the loop
            report[inst.id] = halt_report(halt)
            halting.append(inst)
            continue
        # Spec r3 D5: a stage whose last 2 settled jobs added no committed record halts the instance, named,
        # instead of re-launching a no-op job every step. Spec r4 D17: judged on the jobs and the progress mark of
        # the action's FIRST generation, so every instance that shares it reaches the same verdict.
        now = datetime.datetime.now(datetime.timezone.utc)
        stall = next(
            (
                (a, why)
                for a in acts
                if (
                    why := _zero_progress(
                        "-".join(a.stages),
                        by_gen.get((inst.run_key, gens[a.stages[0]]), []),
                        _progress_mark(store, plan, a.stages[:1]),
                        now,
                    )
                )
            ),
            None,
        )
        if stall is not None:
            report[inst.id] = {"action": "halt", "stages": stall[0].stages, "jobs": [], "failed": stall[1]}
            halting.append(inst)
            continue
        # Owner ruling 2026-10-31 (HFU rung 5): `error_resumes` ERRORs in a row halt the INSTANCE, named, and its live
        # jobs are cancelled after the loop, like the failed-units halt above; before, the stage only reported the
        # failure while the instance's other jobs ran on. Judged per action's first generation (spec r4 D17).
        exhausted = next(
            (
                (a, why)
                for a in acts
                if (only is None or a.kind == only)
                and (why := errors_exhausted(a.stages[0], by_gen.get((inst.run_key, gens[a.stages[0]]), []), cons))
            ),
            None,
        )
        if exhausted is not None:
            report[inst.id] = {"action": "halt", "stages": exhausted[0].stages, "jobs": [], "failed": exhausted[1]}
            halting.append(inst)
            continue
        # HFU rung 5 (the canary's halt arm surfaced this): a stage whose latest job ended in ERROR with nothing
        # left to resume (`resume_after_error` returns a halt reason; "" is its default) halts the INSTANCE,
        # named, and its live jobs are cancelled after the loop -- the contract the `errors_exhausted` and
        # `resume_after_error` docstrings state. A RESUMABLE ERROR (a retryable failed unit, or a timeout with
        # pending work) still resumes per stage below (spec r4 D17); only a non-resumable one halts. Before, `step`
        # only reported the ERROR per stage, so a non-resumable crash left a concurrent sibling (the halt arm's
        # `hold`) orphaned -- never cancelled -- and a settled `--watch` exited 1 with it still running.
        error_halt = next(
            (
                (a, why, h[0])
                for a in acts
                if (only is None or a.kind == only)
                and (h := by_gen.get((inst.run_key, gens[a.stages[0]]), []))
                and str(getattr(getattr(h[0], "status", None), "stage", "") or "") == "ERROR"
                and (why := resume_after_error(plan, a.stages[0], h, cons)) is not None
            ),
            None,
        )
        if error_halt is not None:
            a_h, why_h, prev_h = error_halt
            report[inst.id] = {
                "action": "halt",
                "stages": a_h.stages,
                "jobs": [],
                # stage-named so the canary check matches it by structure (HFU rung 5, `_CANARY_HALT_FORMS`)
                "failed": why_h or f"{a_h.stages[0]}: latest job {prev_h.id} ended in ERROR, nothing to resume",
            }
            halting.append(inst)
            continue
        # HFU T5: the DAG emits an action per ready stage; dispatch EVERY one this step (not acts[0]), sharing
        # the step's dispatch budget. Resume is keyed PER STAGE (spec r4 D17: per generation, whoever launched it).
        # The per-instance report keeps the first action's shape (single-action is unchanged) and lists any
        # concurrent siblings under `also`.
        inst_budget = share  # HFU T7: this instance's slice of the step budget, across ALL its ready stages
        sub: list[dict] = []
        for a in acts:
            if only is not None and a.kind != only:
                sub.append({"action": f"skip:{a.kind}", "jobs": [], "failed": None})
                continue
            # A stage whose latest job ERRORed and is still resumable (a retryable failed unit, or a timeout with
            # pending work) falls through to relaunch below; a non-resumable ERROR already halted the instance.
            history = by_gen.get((inst.run_key, gens[a.stages[0]]), [])
            n_planned = a.n_jobs if a.partition is None else len(a.partition)
            if dry_run:
                sub.append(
                    {
                        "action": a.kind,
                        "stages": a.stages,
                        "jobs": [],
                        "planned": dataclasses.asdict(a),
                        "n_jobs": n_planned,
                    }
                )
                flying.update(gens[s] for s in a.stages)  # spec r4 D17: a planned launch holds its generations
                continue
            ids: list[str] = []
            overflow = False
            for spec in _job_specs(c, commit, inst, plan, a, store=store, unit_peaks=peak_views.get(inst.id)):
                if budget <= 0 or inst_budget <= 0:
                    break
                try:
                    ids.append(launcher.launch(spec, api=api, token=token))
                except Exception as exc:  # classify, do NOT crash the step mid-launch (§D, B09/B26)
                    if _is_capacity_reject(exc):
                        overflow = True  # degrade: stop launching this step; the back-off re-derives next step
                        break
                    raise  # a non-capacity or ambiguous reject fails loud (B26 g/h, consider-09)
                budget -= 1
                inst_budget -= 1
            if ids:  # spec r4 D17: in flight at once, so no instance after this one in input order launches it again
                flying.update(gens[s] for s in a.stages)
            entry = {
                "action": a.kind,
                "stages": a.stages,
                "jobs": ids,
                "failed": None,
                "n_jobs": n_planned,
                "observed_max_concurrency": obs_concurrency,
            }
            if overflow:  # the account limit was hit mid-launch; next step's budget self-throttles as slots free
                entry["dispatch_overflow"] = True
            if a.reason:  # a tolerant reduce names the units it drops
                entry["reason"] = a.reason
            pending = sum(int(s.get("pending") or 0) for s in plan["stages"] if s["name"] in a.stages)
            if lost := _lost_after_completed("-".join(a.stages), history, pending):
                entry["lost"] = lost  # spec r3 D6(c): a storage fault is named, never silently absorbed
            sub.append(entry)
        report[inst.id] = {**sub[0], "also": sub[1:]} if len(sub) > 1 else sub[0]
    # Spec r4 D17: a halting instance cancels its live jobs once EVERY instance has decided, except a job that carries
    # a generation another instance (not halting) still needs, and -- while an instance's plan errors -- a job that
    # carries a generation that instance held in its last plan (cancelled next step instead).
    if not dry_run:
        halted = {i.id for i in halting}
        for inst in halting:
            keep, defer = _cancel_scope(inst, insts, plans, halted, last_gens)
            report[inst.id].update(
                _cancel_instance(
                    api,
                    campaign=c.name,
                    commit=commit,
                    run_key=inst.run_key,
                    tier=inst.tier,
                    variant=inst.variant,
                    keep=keep,
                    defer=defer,
                )
            )
    return report


# ---- commands -------------------------------------------------------------------------------------


def _head() -> str:
    argv = ["git", "rev-parse", "HEAD"]  # git from PATH is the house pattern
    return subprocess.run(argv, capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout.strip()  # noqa: S603


def cmd_preflight(c: Campaign, commit: str, *, api: Any) -> dict:
    """All six no-spend checks; uploads the report the launch gate requires (spec 5.4)."""
    from scripts.hf_jobs import preflight
    from scripts.hf_jobs.registry import canary_spec
    from scripts.hf_jobs.store import BucketStore

    insts = instances(c, commit)
    store = BucketStore(c.bucket, "", api=api)
    # HFU rung 5 (review R5B-09): a canary-only job reads no dataset (`none`), so it has none to check
    datasets = [n for j in c.jobs if canary_spec(j.job) is None for n in (j.dataset, j.probe_dataset) if n]
    checks = [
        preflight.check_ref(commit),  # never cached: a force-push can un-green (HFU T9)
        preflight.check_lock(),
        preflight.check_api(api),
        preflight.check_datasets(datasets, api=api),
        # HFU T9: only the ~20-30 min toy train is cached by its input key; the cheap checks above re-observe.
        preflight.cached_check_plans(insts, store=store, campaign_sha256=c.sha256),
    ]
    flags: list[str] = []
    hist = committed_history()
    cons = planning_constraints(c)
    target = cons.target_s
    for inst in insts:
        try:
            plan = read_plan(inst, bucket=c.bucket)
        except RuntimeError as exc:
            flags.append(f"{inst.id}: no plan ({exc})")
            continue
        for a in next_actions(plan, set(), hist, cons, lpt=lpt_allowed(inst)):
            if a.predicted_job_wall_s and a.predicted_job_wall_s > target:
                wall_min, target_min = a.predicted_job_wall_s / 60, target / 60
                flags.append(f"{inst.id}/{'-'.join(a.stages)} predicted {wall_min:.0f} min > {target_min:.0f}")
    checks.append(preflight.Check("estimate", True, f"{len(flags)} stage(s) predicted above the target", tuple(flags)))
    report = preflight.assemble_report(c.name, commit, c.sha256, checks, flags=flags)
    store.upload_bytes(
        preflight.REPORT_REL.format(campaign=c.name, commit7=commit[:7]), json.dumps(report, indent=2).encode("utf-8")
    )
    return report


def cmd_plan(c: Campaign, commit: str, *, replica: int | None = None) -> list[dict]:
    """Each instance's next action and predicted wall/cost, launching nothing."""
    rows = []
    hist = committed_history()
    cons = planning_constraints(c)
    for inst in instances(c, commit, replica=replica):
        try:
            plan = read_plan(inst, bucket=c.bucket)
        except RuntimeError as exc:
            rows.append({"instance": inst.id, "module": inst.module, "action": None, "error": str(exc)[-300:]})
            continue
        acts = next_actions(plan, set(), hist, cons, lpt=lpt_allowed(inst))
        rows.append(
            {
                "instance": inst.id,
                "module": inst.module,
                "action": dataclasses.asdict(acts[0]) if acts else None,  # the first action (back-compat)
                # HFU T5 / reviewer B D1-IMPL-17: the DAG emits an action per ready stage, so report them ALL
                "actions": [dataclasses.asdict(a) for a in acts],
            }
        )
    return rows


def cmd_status(c: Campaign, commit: str, *, api: Any, replica: int | None = None) -> list[dict]:
    from scripts.hf_jobs import launch as launcher

    all_jobs = list(launcher.jobs_for(api, campaign=c.name, commit=commit))
    live, by_gen = _live_gens(all_jobs), _error_history(all_jobs)  # spec r4 D17: by generation, whoever launched it
    rows = []
    for inst in instances(c, commit, replica=replica):
        try:
            plan = read_plan(inst, bucket=c.bucket)
        except RuntimeError as exc:
            rows.append({"instance": inst.id, "error": str(exc)[-300:]})
            continue
        gens = _plan_gens(plan)
        newest = {name: by_gen[(inst.run_key, g)][0] for name, g in gens.items() if (inst.run_key, g) in by_gen}
        rows.append(
            {
                "instance": inst.id,
                "stages": [
                    {
                        "name": s["name"],
                        "done": s["done"],
                        "pending": s["pending"],
                        "failed": s["failed"],
                        "reduced": s["reduced"],
                    }
                    for s in plan["stages"]
                ],
                "running": sorted(name for name, g in gens.items() if g in live.get(inst.run_key, set())),
                "errors": sorted(
                    f"{name}:{getattr(info, 'id', '?')}"
                    for name, info in newest.items()
                    if str(getattr(getattr(info, "status", None), "stage", "")) == "ERROR"
                ),
            }
        )
    return rows


def _cancel_instance(
    api: Any,
    *,
    campaign: str,
    commit: str,
    run_key: str,
    tier: str,
    variant: str,
    keep: Mapping[str, Sequence[str]] | None = None,
    defer: frozenset[str] | None = frozenset(),
) -> dict:
    """HFU T6/B-25/B-30: cancel ONE instance's in-flight jobs (by run_key + tier + variant) when it halts -- the
    same primitive cmd_cancel uses, scoped to the instance so a same-job/variant sibling on another tier (spec r4
    D18 (c): the SAME run key when both tiers read one dataset) is untouched. A cancel-API failure is recorded and
    retried the next stateless step.

    Spec r4 D17: a job that carries a generation in ``keep`` (generation -> the ids of instances that still need it)
    is kept and named under ``kept``; a job that carries a generation in ``defer`` (None: every job) is left for the
    next step and named under ``cancel_deferred``, which keeps the watch from settling until it is decided."""
    from huggingface_hub.errors import HfHubHTTPError

    from scripts.hf_jobs import launch as launcher

    keep = keep or {}
    cancelled: list[str] = []
    failed: list[dict] = []
    kept: list[dict] = []
    deferred: list[str] = []
    for info in launcher.jobs_for(api, campaign=campaign, commit=commit):
        labels = dict(getattr(info, "labels", None) or {})
        if (labels.get("run_key"), labels.get("tier"), labels.get("variant")) != (run_key, tier, variant):
            continue
        if str(getattr(getattr(info, "status", None), "stage", "")) in launcher.TERMINAL:
            continue
        carried = set(launcher.gens_of(labels, job=str(info.id)))
        users = sorted({i for g in carried for i in keep.get(g, ())})
        if users:
            kept.append({"job": info.id, "for": users})
            continue
        if defer is None or carried & defer:
            deferred.append(info.id)
            continue
        try:
            api.cancel_job(job_id=info.id)
            cancelled.append(info.id)
        except HfHubHTTPError as exc:  # ONLY the cancel-API error -- record it and retry next stateless step (B-11,
            failed.append({"job": info.id, "error": str(exc)[:200]})  # D1-IMPL-12); a programming error fails loud
    out: dict[str, Any] = {"cancelled": cancelled, "cancel_failed": failed}
    if kept:
        out["kept"] = kept
    if deferred:
        out["cancel_deferred"] = deferred
    return out


def cmd_cancel(c: Campaign, commit: str, *, api: Any) -> list[str]:
    from scripts.hf_jobs import launch as launcher

    ids = []
    for info in launcher.jobs_for(api, campaign=c.name, commit=commit):
        if str(getattr(getattr(info, "status", None), "stage", "")) not in launcher.TERMINAL:
            print(f"cancel {info.id} {dict(info.labels or {})}", file=sys.stderr)
            api.cancel_job(job_id=info.id)
            ids.append(info.id)
    return ids


#: `SATURATION_FRACTION` (owner ruling 2026-10-30, 0.85) is single-sourced in `planner`, imported above.

#: Owner ruling 2026-10-30 ("too small to pay"; spec §A): a stage whose summed unit CPU plus its reduce CPU is
#: under this many seconds passes R1 by that class, with its measured time; over it, a stage must be shaped.
SMALL_STAGE_CPU_S = 60.0

#: The reviewed serial reasons of the "too small to pay" class start with this (plan R3-8 SERIAL annex). The audit
#: checks the class against the measurement, so a declared "too small" stage that grew past 60 s fails.
TOO_SMALL_PREFIX = "Too small to pay"


def floor_measured_use(row: Mapping) -> bool:
    """Whether a structural floor's thread budget is really used (owner rulings 2026-10-31; spec §A): when the budget
    is above one thread, the stage's D8 job-record cores per process, ``measured_cores / (workers x
    inner_workers)``, must be MORE than half the budget. So a dead lever (one core per process at a budget of 2 or
    8) and a mostly-serial fit (``ghost_outfield`` ``fit`` in S1: 1.75 cores at budget 8) are never a floor. The one
    definition the audit, the plan vocabulary and spec §A name."""
    budget = int(row.get("thread_budget") or 1)
    if budget <= 1:
        return True
    procs = max(1, int(row.get("workers") or 1) * int(row.get("inner_workers") or 1))
    return float(row["measured_cores"]) / procs > budget / 2


def stage_cpu_s(row: Mapping) -> float | None:
    """The stage's measured CPU for the "too small to pay" class: its summed unit CPU plus its reduce CPU. None when
    unknown, including a reduce that ran (``reduce_wall``) without a recorded CPU: the class then fails closed."""
    unit = row.get("unit_cpu_s")
    if unit is None:
        return None
    reduce_cpu = row.get("reduce_cpu_s")
    if reduce_cpu is None and row.get("reduce_wall") is not None:
        return None
    return float(unit) + float(reduce_cpu or 0.0)


def _job_records(store: Any) -> dict[str, list[dict]]:
    """The run key's job records (spec r3 D8), keyed by job label; a retried job keeps one record per attempt."""
    from scripts._stages import JOB_RECORD_DIR

    out: dict[str, list[dict]] = {}
    for name in sorted(store.names(JOB_RECORD_DIR)):
        if not name.endswith(".job.json"):
            continue
        raw = store.read_bytes(f"{JOB_RECORD_DIR}/{name}")
        if raw:
            rec = json.loads(raw)
            out.setdefault(str(rec.get("job_label")), []).append(rec)
    return out


def stage_measured_cores(st_jobs: Sequence[Mapping], job_records: Mapping[str, Sequence[Mapping]]) -> float | None:
    """The MEASURED cores of the jobs that ran a stage (spec r3 D8): their total CPU over their total wall, from the
    job records keyed by job label. R1 is a job property, so a bundled multi-stage job counts, whole, for each stage
    it ran (B review R3-11). None when no job of the stage left a record (unmeasured, never passing)."""
    from scripts.hf_jobs.launch import label_of

    cpu = wall = 0.0
    for j in st_jobs:
        for r in job_records.get(label_of(j["labels"]), ()):
            if r.get("cpu_s") is not None and r.get("wall_s"):
                cpu += float(r["cpu_s"])
                wall += float(r["wall_s"])
    return round(cpu / wall, 3) if wall > 0 else None


def audit_utilization(row: Mapping, vcpu: int) -> dict:
    """HFU T12: the per-stage utilization verdict for the campaign audit. A stage drives
    ``workers x inner_workers x thread_budget`` cores; ``ok`` is True when it SATURATES the vCPU, OR its idle
    cores are a STRUCTURAL floor (spec B) -- an inner_parallel pool AT its declared fold width (``inner_width``:
    no more folds exist, so the idle cores are irreducible), or a wall floor on a LONE unit (a single unit over
    the per-job budget cannot be fanned further). Reviewer B D1-IMPL-15: ``inner_workers > 1`` alone is NOT a
    floor (it blessed a 2-of-5 under-sized pool), and a wall floor on a MULTI-unit stage is NOT a floor (those
    units must fan to saturate). An inner pool held below its fold width by RAM is surfaced (idle cores, not ok)
    so the audit flags it; its planner ``inner-ram-floor`` note explains why. A pinned stage below the vCPU with
    no floor is NOT ok (R1: under-utilized).

    Spec r3 D8: the verdict uses MEASURED cores (``measured_cores``: the stage's jobs' CPU over their wall), never
    the configured ``workers x inner_workers x thread_budget`` (2474b73: configuration passed stages that idled). No
    measurement is UNMEASURED, never passing. A lone unit is a floor only at the FULL thread budget (a final fit);
    a lone single-thread unit is never a floor by itself (``match_outcome_dependence``: 1 core, 7 idle, passed). A
    stage may also declare a reviewed ``serial_reason``."""
    measured = row.get("measured_cores")
    if measured is None:
        return {
            "cores_used": None,
            "idle_cores": None,
            "saturated": False,
            "floor": False,
            "ok": False,
            "unmeasured": True,
        }
    cores = float(measured)
    saturated = cores >= SATURATION_FRACTION * vcpu
    inner = int(row.get("inner_workers") or 1)
    workers = int(row.get("workers") or 1)
    n_units = int(row.get("n_units") or 0)
    budget = int(row.get("thread_budget") or 1)
    # The three structural floors of spec §A (owner rulings 2026-10-30), each only with measured use (2026-10-31).
    at_fold_width = inner > 1 and inner >= int(row.get("inner_width") or inner)
    final_fit = n_units <= 1 and budget >= vcpu
    all_at_once = n_units >= 1 and workers >= n_units and workers * budget >= vcpu
    used = floor_measured_use(row)
    structural = (at_fold_width or final_fit or all_at_once) and used
    # "Too small to pay" (owner ruling 2026-10-30): by measurement, and a declared reason of the class must hold it.
    cpu = stage_cpu_s(row)
    too_small = cpu is not None and cpu < SMALL_STAGE_CPU_S
    reason = row.get("serial_reason")
    declared = bool(reason) and (not str(reason).startswith(TOO_SMALL_PREFIX) or too_small)
    floor = structural or declared or too_small
    return {
        "cores_used": round(cores, 2),
        "idle_cores": round(max(0.0, vcpu - cores), 2),
        "saturated": saturated,
        "floor": floor,
        "measured_use": used,
        "too_small": too_small,
        "stage_cpu_s": None if cpu is None else round(cpu, 3),
        "ok": saturated or floor,
        "unmeasured": False,
    }


def stage_summary(
    instance: str,
    st: Mapping,
    recs: Sequence[Mapping],
    st_jobs: Sequence[Mapping],
    prof: bytes | None,
    cons: Constraints,
    *,
    job_recs: Sequence[Mapping],
    red: bytes | None = None,
    measured_cores: float | None = None,
    shared_with: Sequence[str] = (),
) -> dict:
    """One stage's row of the committed summary (plan Task 21 step 8): unit cost, efficiency, the measured
    workers and job startup, job walls, attempts, resumed units and the irreducible wall floor.

    ``recs`` are this stage's unit records; ``job_recs`` are every stage's records of the instance. A job's
    startup is measured over ``job_recs`` (a grouped job's first unit may be any of its stages) and is
    recorded on the job's FIRST stage only: in a grouped job a later stage starts after the earlier stages'
    walls, which are not startup (review r3, T21-CORR-01 / CCHC-IMPL-01).

    Spec r4 D17: ``shared_with`` names the instances whose jobs ran this stage's generation when this instance
    launched none. The stage is audited once, on the launcher's row; this row carries no verdict (``audit`` None),
    so it never reads as unmeasured, and `ledger.load_history` never takes it for the stage's history."""
    from scripts.hf_jobs.ledger import job_startups, percentile, stage_workers, unit_cost, upstream_load_s

    cost = unit_cost(recs)
    walls = [j["wall_s"] for j in st_jobs if j.get("wall_s") is not None]
    leads = [j for j in st_jobs if j["labels"].get("stage", "").split("-")[0] == st["name"]]
    startups = job_startups(leads, job_recs)
    workers, inner = stage_workers(recs)
    # HFU T12: the applied per-worker thread budget (max over the stage's units)
    thread_budget = max((int(r.get("thread_budget") or 1) for r in recs), default=1)
    row = {
        "instance": instance,
        "stage": st["name"],
        "history_key": st["history_key"],
        "n_units": st["n_units"],
        "n_jobs": len(st_jobs),
        "unit_cost": dataclasses.asdict(cost) if cost else None,
        "efficiency": json.loads(prof).get("efficiency") if prof else None,
        "workers": workers,
        "inner_workers": inner,
        "inner_width": int(st.get("inner_width") or 1),  # HFU T12/D1-IMPL-15: the declared fold width (audit floor)
        "thread_budget": thread_budget,
        "startup_s": percentile(startups, 0.5) if startups else None,
        "upstream_load_s": upstream_load_s(recs),  # spec r3 D11 (b): the per-worker load, part of the job overhead
        "max_job_wall_s": max(walls) if walls else None,
        "p50_job_wall_s": percentile(walls, 0.5) if walls else None,
        "attempts_max": max((int(r.get("attempt", 1)) for r in recs), default=0),
        "resumed_units": sum(1 for r in recs if int(r.get("attempt", 1)) > 1),
        "irreducible_floor_s": irreducible_floor_s(cost, cons) if cost else None,
        "reduce_wall": json.loads(red).get("wall_s") if red else None,  # HFU T4/B-19: the inline-reduce wall
        # S3 item 3: the stage's CPU for the "too small to pay" class (units + reduce; S2B-r2-04)
        "unit_cpu_s": round(sum(float(r.get("cpu_s") or 0.0) for r in recs), 3) if recs else None,
        "reduce_cpu_s": json.loads(red).get("cpu_s") if red else None,
    }
    row["measured_cores"] = measured_cores  # spec r3 D8: from the job records, never from configuration
    row["serial_reason"] = st.get("serial_reason")
    row["shared_with"] = sorted(shared_with)  # spec r4 D17: audited once, under the instance that launched it
    # HFU T12: saturates the vCPU OR records a floor; spec r4 D17: a shared row defers to its launcher's verdict
    row["audit"] = None if shared_with else audit_utilization(row, cons.vcpu)
    return row


def cmd_summary(c: Campaign, commit: str, *, api: Any, replica: int | None = None) -> pathlib.Path:
    """Write ``docs/research/hf_campaigns/<name>.json`` (a registered artifact: clean tree only, ADR-037)."""
    from scripts._provenance import git_provenance, require_clean_tree
    from scripts.hf_jobs.ledger import write_summary

    require_clean_tree(git_provenance(), allow_dirty=False)
    path = SUMMARY_DIR / f"{c.name}.json"
    write_summary(path, summary_data(c, commit, api=api, replica=replica))
    return path


def summary_data(c: Campaign, commit: str, *, api: Any, replica: int | None = None) -> dict:
    """The campaign summary `cmd_summary` commits, unwritten (`canary-check` reads it too). A canary-only instance
    (HFU rung 5, C3: the toy halt arm) and its jobs are left out: they are no campaign computation and no planner
    history."""
    from scripts._provenance import env_provenance
    from scripts.hf_jobs import launch as launcher
    from scripts.hf_jobs.identity import artifact_identity
    from scripts.hf_jobs.ledger import job_record, summarize, unit_records
    from scripts.hf_jobs.registry import canary_spec

    live_jobs = list(launcher.jobs_for(api, campaign=c.name, commit=commit))
    obs_concurrency = observed_max_concurrency(live_jobs)  # §D probe-first: the real HF ceiling this run
    jobs = [job_record(j) for j in live_jobs if canary_spec(str((j.labels or {}).get("job", ""))) is None]
    stages, artifacts = [], []
    staging = REPO_ROOT.parent / "_hf_staging" / c.name
    cons = planning_constraints(c)
    insts = [i for i in instances(c, commit, replica=replica) if canary_spec(i.job) is None]
    # Spec r3 D7: the TF-85 bulk reads -- each store listed once, its record files downloaded in batches, the
    # plans read in parallel. The 2474b73 summary downloaded every ledger with its own call (about 70 min).
    snaps = store_snapshots([i.run_key for i in insts], bucket=c.bucket, api=api, content={})
    plans = read_plans(insts, snaps, bucket=c.bucket, cache={}, read=read_plan)  # resolved now, as `step` does
    launcher_id = {(i.run_key, i.tier, i.variant): i.id for i in insts}
    for inst in insts:
        plan = plans[inst.id]
        if isinstance(plan, RuntimeError):
            raise plan  # a summary missing an instance would mislead; fail loud, as the serial read did
        store = snaps[inst.run_key]
        by_stage = {st["name"]: unit_records(store, st["generation"]) for st in plan["stages"]}
        job_recs = [r for recs in by_stage.values() for r in recs]
        job_records = _job_records(store)  # spec r3 D8: each job's measured CPU and wall, by job label
        for st in plan["stages"]:
            # Spec r3 D7 + r4 D17: a stage's jobs are the jobs of its run key that carry its generation; each is
            # attributed to the instance that launched it (its run_key + tier + variant labels; spec r4 D18 (c)), so
            # a generation that variants or tiers share is audited once, under its launcher, and listed for the
            # others as shared.
            gen = _gen(st)
            carriers = [
                j
                for j in jobs
                if j["labels"].get("run_key") == inst.run_key
                and gen in launcher.gens_of(j["labels"], job=str(j.get("id")))
            ]
            st_jobs = [
                j
                for j in carriers
                if (j["labels"].get("tier"), j["labels"].get("variant")) == (inst.tier, inst.variant)
            ]
            shared_with = (
                ()
                if st_jobs
                else tuple(
                    {
                        launcher_id.get(
                            (j["labels"]["run_key"], j["labels"].get("tier", ""), j["labels"].get("variant", "")),
                            f"{j['labels']['run_key']}/{j['labels'].get('tier', '')}/{j['labels'].get('variant', '')}",
                        )
                        for j in carriers
                    }
                )
            )
            prof = store.read_bytes(f"{st['generation']}/_profile.json")
            red = store.read_bytes(f"{st['generation']}/_reduce.json")
            stages.append(
                stage_summary(
                    inst.id,
                    st,
                    by_stage[st["name"]],
                    st_jobs,
                    prof,
                    cons,
                    job_recs=job_recs,
                    red=red,
                    measured_cores=stage_measured_cores(st_jobs, job_records),
                    shared_with=shared_with,
                )
            )
        out_dir = (
            staging / inst.id
        )  # `adopt.retrieve`'s layout: <staging>/<run_key>/<tier>/<variant>/ (spec r4 D18 (c))
        if out_dir.is_dir():
            artifacts.append({"instance": inst.id, **artifact_identity(out_dir)})
    prov = {"commit": commit, "dirty": False, "env": env_provenance()}
    return summarize(
        campaign=c.name,
        campaign_sha256=c.sha256,
        prov=prov,
        stages=stages,
        jobs=jobs,
        artifacts=artifacts,
        observed_max_concurrency=obs_concurrency,
    )


# ---- the rung-5 canary check (spec `2026-10-31-hfu-rung5-canary-design.md` C4) ---------------------------------------

#: The three controller halt reasons that stop the canary's halt arm on its `strict` stage (C3/C4), matched by
#: structure: the stage prefix and the reason's fixed words, never the literal tail (the spent units' counts or the
#: job id; review R5-SPEC-04).
_CANARY_HALT_FORMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("failed-units", re.compile(r"^strict: \d+ unit\(s\) failed \d+ time\(s\), over the stage's max_failures=0: ")),
    ("errors-in-a-row", re.compile(r"^strict: \d+ job\(s\) in a row ended in ERROR$")),
    ("non-resumable-error", re.compile(r"^strict: latest job \S+ ended in ERROR, nothing to resume$")),
)


def canary_halt_reason(reason: str | None) -> str | None:
    """Which controller halt stopped the arm (``failed-units``, ``errors-in-a-row`` or ``non-resumable-error``), or
    None for any other text."""
    return next((kind for kind, pattern in _CANARY_HALT_FORMS if reason and pattern.match(reason)), None)


def _report_entries(report: Mapping) -> Iterator[tuple[str, Mapping]]:
    """Every (instance id, entry) of one step report, the concurrent `also` siblings included."""
    for inst_id, entry in report.items():
        yield inst_id, entry
        for sibling in entry.get("also") or ():
            yield inst_id, sibling


def _reduced_every_unit(rec: Mapping | None, n_units: int) -> bool:
    """The reduce's `_summary.json` took every planned unit: none failed, done + excluded = the plan's count."""
    return (
        rec is not None
        and int(rec["n_units"]) == n_units
        and int(rec["n_failed"]) == 0
        and int(rec["n_done"]) + int(rec["n_excluded"]) == n_units
    )


def wp_extract_reduces(insts: Sequence[Instance], plans: Mapping, snaps: Mapping) -> dict[str, dict | None]:
    """The `_summary.json` that the reduce of each `win_probability` instance's `extract` wrote, read at the plan's
    generation, or None where no reduce wrote one (C4.5; review R5B-11)."""
    out: dict[str, dict | None] = {}
    for inst in insts:
        if inst.job != "win_probability":
            continue
        plan = plans[inst.id]
        if isinstance(plan, RuntimeError):
            raise plan
        gen = next(st["generation"] for st in plan["stages"] if st["name"] == "extract")
        raw = snaps[inst.run_key].read_bytes(f"{gen}/_summary.json")
        out[inst.id] = json.loads(raw) if raw else None
    return out


def canary_check(
    *,
    reports: Sequence[Mapping],
    summary: Mapping,
    listing: Sequence[str],
    jobs: Sequence[Mapping],
    halt_instance: str,
    wp_reduce: Mapping[str, Mapping | None],
) -> dict:
    """The rung-5 pass check (C4) over the `--reports` record of the watch, the campaign summary (`summary_data`),
    one bucket listing of the canary run keys, the campaign's jobs and the `win_probability` extract's reduce record
    (`wp_extract_reduces`). Each check is a fact the code already single-sources; the R1 verdict is
    `audit_utilization`'s (a shared row defers to its launcher, review R5-SPEC-01)."""
    steps = [r["report"] for r in reports if "report" in r]
    final = steps[-1] if steps else {}
    exit_codes = [r["exit_code"] for r in reports if "exit_code" in r]
    entries = [e for rep in steps for _, e in _report_entries(rep)]
    halt = final.get(halt_instance) or {}
    halt_rk = halt_instance.split("/", 1)[0]
    holds = [
        j
        for j in jobs
        if (j.get("labels") or {}).get("run_key") == halt_rk
        and "hold" in str((j.get("labels") or {}).get("stage", "")).split("-")
    ]
    wp = [s for s in summary.get("stages", []) if "-win_probability-" in s["instance"] and s["stage"] == "extract"]
    # review R5B-10 (a): `n_units` is the planned count; the instance's settled `done` says the extract reduced.
    # Review R5B-11: the extract tolerates failed units (`max_failures=None`), so the reduce's own counts must show
    # that it took all 3,961 units (none failed).
    wp_done = all((final.get(s["instance"]) or {}).get("action") == "done" for s in wp)
    wp_all = all(_reduced_every_unit(wp_reduce.get(s["instance"]), 3961) for s in wp)
    kind = canary_halt_reason(halt.get("failed"))
    checks = {
        "instances_done": bool(final)
        and all(e.get("action") == "done" for i, e in final.items() if i != halt_instance),
        "r1_audit": all((s.get("audit") or {}).get("ok", True) for s in summary.get("stages", [])),
        "no_lost_output": not any(e.get("lost") for e in entries),
        "no_orphan_temp": not any(path.rsplit("/", 1)[-1].startswith(".tmp-") for path in listing),
        "no_zero_progress_halt": not any(": zero progress -- " in str(e.get("failed") or "") for e in entries),
        "win_probability_extract_3961": bool(wp) and wp_done and wp_all and all(int(s["n_units"]) == 3961 for s in wp),
        "halt_arm": halt.get("action") == "halt"
        and kind is not None
        and bool(holds)
        and all(str(j.get("status")) == "CANCELED" for j in holds)
        and bool(exit_codes)
        and exit_codes[-1] != 0,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "halt_reason_kind": kind,
        "halt_reason": halt.get("failed"),
        "wp_reduce": dict(wp_reduce),
    }


def cmd_canary_check(c: Campaign, commit: str, *, api: Any, reports_path: pathlib.Path) -> dict:
    """`canary-check`: gather the inputs of `canary_check` from the bucket and the HF Jobs API, then check."""
    from scripts.hf_jobs import launch as launcher
    from scripts.hf_jobs.registry import canary_spec

    reports = [json.loads(ln) for ln in reports_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    insts = instances(c, commit)
    halts = [i.id for i in insts if canary_spec(i.job) is not None]
    if len(halts) != 1:
        raise SystemExit(f"campaign {c.name}: canary-check needs exactly one halt-arm instance, found {halts}")
    listing = [
        e.path
        for rk in sorted({i.run_key for i in insts})
        for e in api.list_bucket_tree(c.bucket, prefix=f"{rk}/", recursive=True)
    ]
    jobs = [
        {"id": j.id, "labels": dict(j.labels or {}), "status": str(getattr(j.status, "stage", ""))}
        for j in launcher.jobs_for(api, campaign=c.name, commit=commit)
    ]
    summary = summary_data(c, commit, api=api)
    wp = [i for i in insts if i.job == "win_probability"]
    snaps = store_snapshots([i.run_key for i in wp], bucket=c.bucket, api=api, content={})
    plans = read_plans(wp, snaps, bucket=c.bucket, cache={}, read=read_plan)
    return canary_check(
        reports=reports,
        summary=summary,
        listing=listing,
        jobs=jobs,
        halt_instance=halts[0],
        wp_reduce=wp_extract_reduces(wp, plans, snaps),
    )


def _record(path: pathlib.Path | None, row: Mapping) -> None:
    """Append one JSON line to the `--reports` record (C4), when one is kept."""
    if path is None:
        return
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def _inside_repo(path: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.hf_jobs campaign")
    ap.add_argument("file", type=pathlib.Path)
    ap.add_argument(
        "command",
        choices=[
            "preflight",
            "profile",
            "plan",
            "launch",
            "resume",
            "status",
            "retrieve",
            "verify",
            "adopt",
            "publish",
            "cancel",
            "summary",
            "canary-check",
        ],
    )
    ap.add_argument("--replica", type=int, default=None, help="fresh replica store suffix -r<N> (INV-1 gate, P18)")
    ap.add_argument("--watch", action="store_true", help="launch/resume: step every 60 s until done or failed")
    ap.add_argument("--dry-run", action="store_true", help="launch/resume/profile: print the plan, launch nothing")
    ap.add_argument(
        "--reports",
        type=pathlib.Path,
        default=None,
        help="launch/resume --watch: append each step report as a JSON line, the settled exit code last; "
        "canary-check: the record to check (outside the repo)",
    )
    ap.add_argument("--staging", type=pathlib.Path, default=None, help="retrieve/verify/adopt/publish staging dir")
    ap.add_argument("--sandbox", action="store_true", help="publish to private silly-kicks/sandbox-<name> repos")
    ap.add_argument(
        "--i-have-owner-approval", action="append", default=[], metavar="REPO", help="publish: owner-approved repo"
    )
    ap.add_argument(
        "--artifacts", default="", help="publish: comma-separated artifact ids (default: every Hub artifact)"
    )
    a = ap.parse_args(argv)
    c = load_campaign(a.file)
    if c.name.startswith("canary") and a.command in ("retrieve", "verify", "adopt", "publish"):
        raise SystemExit(
            f"campaign {c.name}: a canary campaign proves the controller (HFU rung 5); it never runs `{a.command}`"
        )
    if a.reports is not None and _inside_repo(a.reports):
        raise SystemExit(f"--reports {a.reports}: the record must lie outside the repository")
    if a.reports is not None and a.command == "launch" and a.reports.exists():  # review R5B-10 (c): one run per record
        raise SystemExit(f"--reports {a.reports} already exists: a launch starts a new record (`resume` appends)")
    commit = _head()
    # D1: every command that reads the tree through `--plan` must refuse a dirty tree, or its tokens become
    # `dirty-<hash>` and miss the committed-token stores -- `retrieve` then silently skips instances (reviewer B
    # B04). `step` guards launch/resume/profile already; this covers the rest. `adopt`/`summary` require_clean_tree.
    if a.command in ("plan", "preflight", "status", "retrieve", "verify"):
        tree = _tree_identity()
        if tree.startswith("dirty-"):
            raise SystemExit(
                f"campaign {c.name}: refusing `{a.command}` on a dirty tree ({tree}); commit or stash first"
            )
    if a.command == "plan":
        print(json.dumps(cmd_plan(c, commit, replica=a.replica), indent=2, default=str))
        return 0
    from huggingface_hub import HfApi, get_token

    token = get_token()
    if token is None:
        raise SystemExit("no HF token: run `hf auth login` first")
    api = HfApi()
    if a.command == "preflight":
        rep = cmd_preflight(c, commit, api=api)
        print(json.dumps(rep, indent=2))
        return 0 if rep["ok"] else 1
    if a.command in ("profile", "launch", "resume"):
        cache: dict = {}
        only = "profile" if a.command == "profile" else None
        while True:
            rep = step(c, commit, api=api, token=token, replica=a.replica, dry_run=a.dry_run, only=only, cache=cache)
            print(json.dumps(rep, indent=2, default=str), flush=True)
            _record(a.reports, {"report": rep})
            settled = all_settled(rep)
            if not a.watch or a.dry_run:
                return 0
            if settled:
                code = watch_exit_code(rep)
                _record(a.reports, {"exit_code": code})
                return code
            time.sleep(60)
    if a.command == "status":
        print(json.dumps(cmd_status(c, commit, api=api, replica=a.replica), indent=2))
        return 0
    if a.command == "cancel":
        print(json.dumps(cmd_cancel(c, commit, api=api)))
        return 0
    if a.command == "summary":
        print(cmd_summary(c, commit, api=api, replica=a.replica))
        return 0
    if a.command == "canary-check":
        if not c.name.startswith("canary") or a.reports is None:
            raise SystemExit("canary-check runs on a canary campaign with --reports PATH (the watch record)")
        out = cmd_canary_check(c, commit, api=api, reports_path=a.reports)
        print(json.dumps(out, indent=2, default=str))
        return 0 if out["ok"] else 1
    from scripts.hf_jobs import adopt

    if a.staging is None:
        raise SystemExit(f"{a.command} needs --staging DIR (outside the repo)")
    if a.command == "retrieve":
        print(
            json.dumps(
                {
                    k: str(v)
                    for k, v in adopt.retrieve(c, commit, staging=a.staging, api=api, replica=a.replica).items()
                },
                indent=2,
            )
        )
    elif a.command == "verify":
        print(json.dumps(adopt.verify(c, commit, staging=a.staging, replica=a.replica), indent=2, default=str))
    elif a.command == "adopt":
        print(json.dumps(adopt.adopt(c, commit, staging=a.staging), indent=2, default=str))
    else:
        ids = [x for x in a.artifacts.split(",") if x] or None
        print(
            json.dumps(
                adopt.publish(ids, sandbox=a.sandbox, api=api, staging=a.staging, approved=a.i_have_owner_approval),
                indent=2,
                default=str,
            )
        )
    return 0
