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
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from scripts.hf_jobs.planner import Constraints, UnitCost, irreducible_floor_s, plan_stage

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SUMMARY_DIR = REPO_ROOT / "docs" / "research" / "hf_campaigns"
#: The adapter module that runs an unmigrated trainer's CLI as one unit (spec 5.6, Task 17).
ADAPTER_MODULE = "scripts.hf_jobs.adapter"


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
        return f"{self.run_key}/{self.variant}"


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


def run_key(c: Campaign, commit: str, job: str, tier: str, *, replica: int | None = None) -> str:
    """``{campaign}-{commit7}-{job}-{tier}`` (+ ``-r<N>``: a FRESH store for the INV-1 replica gate, P18)."""
    key = f"{c.name}-{commit[:7]}-{job}-{tier}"
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
    from scripts.hf_jobs.registry import DriverSpec, spec_for

    out: list[Instance] = []
    for j in c.jobs:
        spec = spec_for(j.job)
        rk = run_key(c, commit, j.job, j.tier, replica=replica)
        mounts = _mounts([j.dataset, j.probe_dataset])
        if isinstance(spec, DriverSpec):
            for v in j.variants:
                out.append(Instance(j.job, j.tier, v, spec.module, ("--dataset", j.dataset), mounts, rk))
            continue
        builds = {b.variant: b for b in spec.builds}
        for v in j.variants:
            b = builds[v]
            if spec.adapter:
                args: tuple[str, ...] = (
                    "--target",
                    spec.module,
                    "--target-args",
                    json.dumps([*spec.base_args, *b.run_args]),
                    "--fresh-subdir",
                    b.fresh_subdir,
                    "--dataset",
                    j.dataset,
                )
                out.append(Instance(j.job, j.tier, v, ADAPTER_MODULE, args, mounts, rk))
                continue
            args = (*spec.base_args, *b.run_args, "--dataset", j.dataset)
            if j.probe_dataset is not None:
                args += ("--probe-dataset", j.probe_dataset)
            if spec.visibility_gated:
                stems = [u["stem"] for u in staged_units(j.dataset)]
                args += ("--arms-json", json.dumps(arm_policy.compute_arms(stems), sort_keys=True))
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


def _reduce_timeout(reduce_wall, c):
    """A reduce timeout: timeout_safety x the measured reduce wall FLOORED at timeout_floor_s (a tiny wall
    must not give a near-zero timeout -- partial review B D1-IMPL-01), else the generous profile ceiling."""
    if reduce_wall is None:
        return int(c.profile_timeout_s)
    return int(max(c.timeout_floor_s, c.timeout_safety * reduce_wall))


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
    the cancel and never exits leaving the halted instance's jobs alive (reviewer B D1-IMPL-12)."""
    for r in rep.values():
        if r.get("jobs") or any(e.get("jobs") for e in r.get("also", [])):
            return False
        if r.get("cancel_failed"):  # a halt whose cancel-API call failed: not settled, retry it next step (D1-IMPL-12)
            return False
        if r.get("action") != "done" and not r.get("failed"):
            return False
    return True


def _merge_history(committed: Mapping, live: Mapping) -> dict:
    """Live (this run's) history overrides committed, BUT a live reduce_wall of None (a done stage whose
    reduce has not run this run yet -- live_history caches None) inherits the committed reduce_wall, so a
    measured reduce_wall is not lost after the first wave (partial review B D1-IMPL-04; T8 reads it)."""
    out = {**committed, **live}
    for hk, entry in live.items():
        if entry.get("reduce_wall") is None and (committed.get(hk) or {}).get("reduce_wall") is not None:
            out[hk] = {**entry, "reduce_wall": committed[hk]["reduce_wall"]}
    return out


def next_actions(
    plan: Mapping,
    running: set[str],
    history: Mapping[str, Mapping],
    c: Constraints,
    *,
    lpt: bool = True,
    free_slots: int = 0,
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
        reduce_wall = (history.get(st.get("history_key") or st["name"]) or {}).get("reduce_wall")
        # HFU T8: a shardable reduce whose measured wall exceeds the per-job budget is range-partitioned.
        budget = max(1.0, c.target_s - c.startup_s)
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
                                    _reduce_timeout(reduce_wall, c),
                                    shard_id=j,
                                    n_shards=n_sub,
                                )
                            )
                        continue
                    actions.append(
                        Action("reduce", (st["name"],), 1, 1, 1, _reduce_timeout(reduce_wall, c), n_shards=n_sub)
                    )
                    continue
                actions.append(Action("reduce", (st["name"],), 1, 1, 1, _reduce_timeout(reduce_wall, c)))
                continue
            tolerance = st.get("max_failures")
            if tolerance is None or len(spent) <= int(tolerance):
                why = f"{st['name']}: reduces without unit(s) failed {c.unit_attempts} time(s): {spent}"
                actions.append(Action("reduce", (st["name"],), 1, 1, 1, _reduce_timeout(reduce_wall, c), reason=why))
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
            c=c,
        )
        # HFU T4/B-19: the inline reduce runs in whichever run job finishes last; budget it. With a recorded
        # reduce_wall, add it to the measured job timeout; on a first run (no reduce_wall) use the generous
        # profile ceiling so the inline reduce is never killed at the bare 1800 floor.
        run_timeout = (
            p.timeout_s + int(c.timeout_safety * reduce_wall)
            if reduce_wall is not None
            else max(p.timeout_s, int(c.profile_timeout_s))
        )
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
            }
    return out


def planning_constraints(c: Campaign, *, summary_dir: pathlib.Path | None = None) -> Constraints:
    """The campaign's constraints with the job startup the committed summaries measured (spec 5.5: the
    planner learns from them); the `Constraints` default applies before any campaign measured one."""
    from scripts.hf_jobs.ledger import median_startup_s

    return c.constraints(startup_s=median_startup_s((summary_dir or SUMMARY_DIR).glob("*.json")))


def live_history(plan: Mapping, store: Any, cache: dict) -> dict[str, dict]:
    """Costs measured in THIS store (profile units, finished units), keyed by history key (decision P14)."""
    from scripts.hf_jobs.ledger import unit_cost, unit_records

    out: dict[str, dict] = {}
    for st in plan["stages"]:
        if not st.get("done"):
            continue
        gen = st["generation"]
        if gen not in cache:
            recs = unit_records(store, gen)
            prof = store.read_bytes(f"{gen}/_profile.json")
            eff = json.loads(prof).get("efficiency") if prof else None
            red = store.read_bytes(f"{gen}/_reduce.json")  # HFU T4/B-19: the inline-reduce wall, if measured
            reduce_wall = json.loads(red).get("wall_s") if red else None
            cost = unit_cost(recs)
            # a D7 repair record is `done` with an unknown wall; it contributes no measured per-unit cost
            per_unit = {
                r["key"]: float(r["wall_s"]) for r in recs if r.get("outcome") == "done" and r.get("wall_s") is not None
            }
            cache[gen] = (
                None
                if cost is None
                else {"cost": cost, "efficiency": eff, "per_unit": per_unit, "reduce_wall": reduce_wall}
            )
        if cache[gen] is not None:
            out[st.get("history_key") or st["name"]] = cache[gen]
    return out


# ---- bulk step reads (TF-85) ---------------------------------------------------------------------


#: Concurrent store listings and ``--plan`` subprocesses per step (TF-85: run 1's step ran 42 plans serially).
STEP_WORKERS = 8


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
    for n, (inst, plan, _) in enumerate(rows):
        for st in plan["stages"]:
            gen = st.get("generation")
            if st.get("done") and gen is not None and gen not in cache.get(inst.run_key, {}):
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


def _error_history(jobs: Iterable[Any]) -> dict[tuple[str, str, str], list[Any]]:
    """Every job per (run_key, variant, stage label), newest first."""
    out: dict[tuple[str, str, str], list[Any]] = {}
    for info in jobs:
        lab = dict(getattr(info, "labels", None) or {})
        out.setdefault((lab.get("run_key", ""), lab.get("variant", ""), lab.get("stage", "")), []).append(info)
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


def job_never_started_timeout(info: Any) -> bool:
    """True when a job TIMED OUT and its container NEVER STARTED (``started_at`` present and unset) -- a
    dispatch overflow, exempt from ``error_resumes`` (§D). The attribute MISSING => treated as STARTED (it
    counts), never exempt on doubt (B23 fail-safe); a started or non-timeout job is not an overflow."""
    if not _timed_out(info):
        return False
    sentinel = object()
    started = getattr(info, "started_at", sentinel)
    if started is sentinel:  # attribute absent -> fail safe to a started (counting) running-timeout
        return False
    return started is None


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


def resume_after_error(plan: Mapping, stage: str, history: Sequence[Any], c: Constraints) -> str | None:
    """None when the next action may follow a stage whose latest job ended in ERROR, else the halt
    reason ("" = the caller's default). It may when that stage still has a retryable failed unit (a strict
    stage's job ends in ERROR on any failed unit; the retry stays bounded by ``unit_attempts``), or when the
    job ran out of time with units pending (finished units persist). EVERY such resume is also bounded by
    ``error_resumes`` ERRORs in a row, which needs no unit record: a timeout or a crash before any unit
    writes none, so the attempt count alone would resume forever (Reviewer B CCHC-IMPL-09)."""
    st = next((s for s in plan["stages"] if s["name"] == stage), None)
    if st is None:
        return ""
    errors = _trailing_errors(history)
    if errors >= c.error_resumes:
        return f"{stage}: {errors} job(s) in a row ended in ERROR"
    if any(int(n) < c.unit_attempts for n in (st.get("failed_units") or {}).values()):
        return None
    if history and _timed_out(history[0]) and st.get("pending_units"):
        return None
    return ""


def halt_report(a: Action) -> dict:
    """The step report of a ``halt`` action: nothing is launched; the reason names the failed units."""
    return {"action": "halt", "stages": a.stages, "jobs": [], "failed": a.reason}


def _job_specs(c: Campaign, commit: str, inst: Instance, plan: Mapping, a: Action, *, store: Any) -> list[Any]:
    from scripts.hf_jobs.launch import JobSpec

    if a.partition is not None and any(d.access == "download" for d in inst.datasets):
        raise ValueError(f"{inst.id}: an LPT partition needs mounted datasets (download access fetches by shard)")

    def spec(shard: tuple[int, int], shard_keys_path: str | None = None) -> JobSpec:
        return JobSpec(
            campaign=c.name,
            run_key=inst.run_key,
            job=inst.job,
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
            subreduce_shard_id=(a.shard_id if a.kind == "subreduce" else None),  # HFU T8
            subreduce_n_shards=(a.n_shards if a.kind == "subreduce" else None),  # HFU T8: the worker's shard count
            from_partials=(a.n_shards if a.kind == "reduce" else None),
            no_inline_reduce=(a.kind == "run" and a.n_shards is not None),
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


def _running(jobs: Iterable[Any]) -> tuple[dict[tuple[str, str], set[str]], dict[tuple[str, str, str], Any]]:
    """``{(run_key, variant): {running stage labels}}`` and the LATEST job per (run_key, variant, stage)."""
    from scripts.hf_jobs.launch import TERMINAL

    running: dict[tuple[str, str], set[str]] = {}
    latest: dict[tuple[str, str, str], Any] = {}
    for info in jobs:
        lab = dict(getattr(info, "labels", None) or {})
        key = (lab.get("run_key", ""), lab.get("variant", ""))
        stage = lab.get("stage", "")
        status = str(getattr(getattr(info, "status", None), "stage", "") or "")
        if status not in TERMINAL:
            for s in stage.split("-"):
                running.setdefault(key, set()).add(s)
        k3 = (*key, stage)
        prev = latest.get(k3)
        if prev is None or (getattr(info, "created_at", None) or 0) > (getattr(prev, "created_at", None) or 0):
            latest[k3] = info
    return running, latest


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
    running, latest = _running(all_jobs)
    by_stage = _error_history(all_jobs)
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
    budget = effective_dispatch_cap(
        dispatch_cap=c.dispatch_cap, dispatch_floor=cons.dispatch_floor, never_started=never_started
    ) - sum(1 for v in running.values() for _ in v)
    snaps = store_snapshots(
        [i.run_key for i in insts], bucket=c.bucket, api=api, content=cache.setdefault("content", {})
    )
    plans = read_plans(insts, snaps, bucket=c.bucket, cache=cache, tree=tree, read=read_plan)
    views = instance_views(insts, plans, snaps, cache=cache.setdefault("live_history", {}))
    report: dict[str, dict] = {}

    # HFU T7: each ACTIVE instance gets a FAIR SLICE of the step's dispatch budget, so one instance's
    # slot-filling cannot take every free slot (B-08). Active = not errored and not all-stages-reduced; the
    # share is known BEFORE next_actions (no circular dependency), and caps the instance's TOTAL launches.
    def _active(p: Any) -> bool:
        return not isinstance(p, RuntimeError) and not all(s.get("reduced") for s in p["stages"])

    active = sum(1 for i in insts if _active(plans[i.id]))
    share = max(1, budget // max(1, active))
    for inst in insts:
        plan = plans[inst.id]
        if isinstance(plan, RuntimeError):
            report[inst.id] = {"action": None, "jobs": [], "failed": str(plan)}
            continue
        store = snaps[inst.run_key]
        acts = next_actions(
            plan,
            running.get((inst.run_key, inst.variant), set()),
            _merge_history(hist, views[inst.id]),
            cons,
            lpt=lpt_allowed(inst),
            free_slots=share,
        )
        if not acts:
            done = all(s.get("reduced") for s in plan["stages"])
            report[inst.id] = {"action": "done" if done else "waiting", "jobs": [], "failed": None}
            continue
        halt = next((a for a in acts if a.kind == "halt"), None)
        if halt is not None:  # a halting stage halts the INSTANCE; cancel its live jobs by (run_key, variant)
            rep = halt_report(halt)
            if not dry_run:
                rep.update(
                    _cancel_instance(api, campaign=c.name, commit=commit, run_key=inst.run_key, variant=inst.variant)
                )
            report[inst.id] = rep
            continue
        # HFU T5: the DAG emits an action per ready stage; dispatch EVERY one this step (not acts[0]), sharing
        # the step's dispatch budget. Resume is keyed PER STAGE. The per-instance report keeps the first
        # action's shape (single-action is unchanged) and lists any concurrent siblings under `also`.
        inst_budget = share  # HFU T7: this instance's slice of the step budget, across ALL its ready stages
        sub: list[dict] = []
        for a in acts:
            if only is not None and a.kind != only:
                sub.append({"action": f"skip:{a.kind}", "jobs": [], "failed": None})
                continue
            key = (inst.run_key, inst.variant, "-".join(a.stages))
            prev = latest.get(key)
            prev_status = str(getattr(getattr(prev, "status", None), "stage", "") or "")
            if prev is not None and prev_status == "ERROR":
                why = resume_after_error(plan, a.stages[0], by_stage.get(key, []), cons)
                if why is not None:
                    msg = getattr(getattr(prev, "status", None), "message", None)
                    sub.append(
                        {
                            "action": a.kind,
                            "stages": a.stages,
                            "jobs": [],
                            "failed": why or f"job {prev.id} ERROR: {msg}",
                        }
                    )
                    continue
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
                continue
            ids: list[str] = []
            overflow = False
            for spec in _job_specs(c, commit, inst, plan, a, store=store):
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
            sub.append(entry)
        report[inst.id] = {**sub[0], "also": sub[1:]} if len(sub) > 1 else sub[0]
    return report


# ---- commands -------------------------------------------------------------------------------------


def _head() -> str:
    argv = ["git", "rev-parse", "HEAD"]  # git from PATH is the house pattern
    return subprocess.run(argv, capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout.strip()  # noqa: S603


def cmd_preflight(c: Campaign, commit: str, *, api: Any) -> dict:
    """All six no-spend checks; uploads the report the launch gate requires (spec 5.4)."""
    from scripts.hf_jobs import preflight
    from scripts.hf_jobs.store import BucketStore

    insts = instances(c, commit)
    store = BucketStore(c.bucket, "", api=api)
    checks = [
        preflight.check_ref(commit),  # never cached: a force-push can un-green (HFU T9)
        preflight.check_lock(),
        preflight.check_api(api),
        preflight.check_datasets([n for j in c.jobs for n in (j.dataset, j.probe_dataset) if n], api=api),
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

    running, latest = _running(launcher.jobs_for(api, campaign=c.name, commit=commit))
    rows = []
    for inst in instances(c, commit, replica=replica):
        try:
            plan = read_plan(inst, bucket=c.bucket)
        except RuntimeError as exc:
            rows.append({"instance": inst.id, "error": str(exc)[-300:]})
            continue
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
                "running": sorted(running.get((inst.run_key, inst.variant), set())),
                "errors": sorted(
                    f"{k[2]}:{getattr(v, 'id', '?')}"
                    for k, v in latest.items()
                    if k[:2] == (inst.run_key, inst.variant)
                    and str(getattr(getattr(v, "status", None), "stage", "")) == "ERROR"
                ),
            }
        )
    return rows


def _cancel_instance(api: Any, *, campaign: str, commit: str, run_key: str, variant: str) -> dict:
    """HFU T6/B-25/B-30: cancel ONE instance's in-flight jobs (by run_key + variant) when it halts -- the
    same primitive cmd_cancel uses, scoped to the instance so a same-job/variant sibling on another tier (a
    different run_key) is untouched. A cancel-API failure is recorded and retried the next stateless step."""
    from huggingface_hub.errors import HfHubHTTPError

    from scripts.hf_jobs import launch as launcher

    cancelled: list[str] = []
    failed: list[dict] = []
    for info in launcher.jobs_for(api, campaign=campaign, commit=commit):
        labels = dict(getattr(info, "labels", None) or {})
        if labels.get("run_key") != run_key or labels.get("variant") != variant:
            continue
        if str(getattr(getattr(info, "status", None), "stage", "")) in launcher.TERMINAL:
            continue
        try:
            api.cancel_job(job_id=info.id)
            cancelled.append(info.id)
        except HfHubHTTPError as exc:  # ONLY the cancel-API error -- record it and retry next stateless step (B-11,
            failed.append({"job": info.id, "error": str(exc)[:200]})  # D1-IMPL-12); a programming error fails loud
    return {"cancelled": cancelled, "cancel_failed": failed}


def cmd_cancel(c: Campaign, commit: str, *, api: Any) -> list[str]:
    from scripts.hf_jobs import launch as launcher

    ids = []
    for info in launcher.jobs_for(api, campaign=c.name, commit=commit):
        if str(getattr(getattr(info, "status", None), "stage", "")) not in launcher.TERMINAL:
            print(f"cancel {info.id} {dict(info.labels or {})}", file=sys.stderr)
            api.cancel_job(job_id=info.id)
            ids.append(info.id)
    return ids


def audit_utilization(row: Mapping, vcpu: int) -> dict:
    """HFU T12: the per-stage utilization verdict for the campaign audit. A stage drives
    ``workers x inner_workers x thread_budget`` cores; ``ok`` is True when it SATURATES the vCPU, OR its idle
    cores are a STRUCTURAL floor (spec B) -- an inner_parallel pool AT its declared fold width (``inner_width``:
    no more folds exist, so the idle cores are irreducible), or a wall floor on a LONE unit (a single unit over
    the per-job budget cannot be fanned further). Reviewer B D1-IMPL-15: ``inner_workers > 1`` alone is NOT a
    floor (it blessed a 2-of-5 under-sized pool), and a wall floor on a MULTI-unit stage is NOT a floor (those
    units must fan to saturate). An inner pool held below its fold width by RAM is surfaced (idle cores, not ok)
    so the audit flags it; its planner ``inner-ram-floor`` note explains why. A pinned stage below the vCPU with
    no floor is NOT ok (R1: under-utilized)."""
    cores = int(row.get("workers") or 1) * int(row.get("inner_workers") or 1) * int(row.get("thread_budget") or 1)
    saturated = cores >= vcpu
    inner = int(row.get("inner_workers") or 1)
    at_fold_width = inner > 1 and inner >= int(row.get("inner_width") or inner)
    lone_unit_wall = row.get("irreducible_floor_s") is not None and int(row.get("n_units") or 0) <= 1
    floor = at_fold_width or lone_unit_wall
    return {
        "cores_used": cores,
        "idle_cores": max(0, vcpu - cores),
        "saturated": saturated,
        "floor": floor,
        "ok": saturated or floor,
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
) -> dict:
    """One stage's row of the committed summary (plan Task 21 step 8): unit cost, efficiency, the measured
    workers and job startup, job walls, attempts, resumed units and the irreducible wall floor.

    ``recs`` are this stage's unit records; ``job_recs`` are every stage's records of the instance. A job's
    startup is measured over ``job_recs`` (a grouped job's first unit may be any of its stages) and is
    recorded on the job's FIRST stage only: in a grouped job a later stage starts after the earlier stages'
    walls, which are not startup (review r3, T21-CORR-01 / CCHC-IMPL-01)."""
    from scripts.hf_jobs.ledger import job_startups, percentile, stage_workers, unit_cost

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
        "max_job_wall_s": max(walls) if walls else None,
        "p50_job_wall_s": percentile(walls, 0.5) if walls else None,
        "attempts_max": max((int(r.get("attempt", 1)) for r in recs), default=0),
        "resumed_units": sum(1 for r in recs if int(r.get("attempt", 1)) > 1),
        "irreducible_floor_s": irreducible_floor_s(cost, cons) if cost else None,
        "reduce_wall": json.loads(red).get("wall_s") if red else None,  # HFU T4/B-19: the inline-reduce wall
    }
    row["audit"] = audit_utilization(row, cons.vcpu)  # HFU T12: saturates the vCPU OR records a floor
    return row


def cmd_summary(c: Campaign, commit: str, *, api: Any, replica: int | None = None) -> pathlib.Path:
    """Write ``docs/research/hf_campaigns/<name>.json`` (a registered artifact: clean tree only, ADR-037)."""
    from scripts._provenance import env_provenance, git_provenance, require_clean_tree
    from scripts.hf_jobs import launch as launcher
    from scripts.hf_jobs.identity import artifact_identity
    from scripts.hf_jobs.ledger import job_record, summarize, unit_records, write_summary
    from scripts.hf_jobs.store import BucketStore

    require_clean_tree(git_provenance(), allow_dirty=False)
    live_jobs = list(launcher.jobs_for(api, campaign=c.name, commit=commit))
    obs_concurrency = observed_max_concurrency(live_jobs)  # §D probe-first: the real HF ceiling this run
    jobs = [job_record(j) for j in live_jobs]
    stages, artifacts = [], []
    staging = REPO_ROOT.parent / "_hf_staging" / c.name
    cons = planning_constraints(c)
    for inst in instances(c, commit, replica=replica):
        store = BucketStore(c.bucket, inst.run_key, api=api)
        plan = read_plan(inst, bucket=c.bucket)
        by_stage = {st["name"]: unit_records(store, st["generation"]) for st in plan["stages"]}
        job_recs = [r for recs in by_stage.values() for r in recs]
        for st in plan["stages"]:
            st_jobs = [
                j
                for j in jobs
                if j["labels"].get("run_key") == inst.run_key and st["name"] in j["labels"].get("stage", "").split("-")
            ]
            prof = store.read_bytes(f"{st['generation']}/_profile.json")
            red = store.read_bytes(f"{st['generation']}/_reduce.json")
            stages.append(
                stage_summary(inst.id, st, by_stage[st["name"]], st_jobs, prof, cons, job_recs=job_recs, red=red)
            )
        out_dir = staging / inst.run_key / inst.variant
        if out_dir.is_dir():
            artifacts.append({"instance": inst.id, **artifact_identity(out_dir)})
    prov = {"commit": commit, "dirty": False, "env": env_provenance()}
    path = SUMMARY_DIR / f"{c.name}.json"
    write_summary(
        path,
        summarize(
            campaign=c.name,
            campaign_sha256=c.sha256,
            prov=prov,
            stages=stages,
            jobs=jobs,
            artifacts=artifacts,
            observed_max_concurrency=obs_concurrency,
        ),
    )
    return path


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
        ],
    )
    ap.add_argument("--replica", type=int, default=None, help="fresh replica store suffix -r<N> (INV-1 gate, P18)")
    ap.add_argument("--watch", action="store_true", help="launch/resume: step every 60 s until done or failed")
    ap.add_argument("--dry-run", action="store_true", help="launch/resume/profile: print the plan, launch nothing")
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
            settled = all_settled(rep)
            if not a.watch or a.dry_run or settled:
                return 0
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
