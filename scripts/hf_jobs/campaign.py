"""The stateless HF campaign controller (ADR-128, spec 5.5).

A campaign is a COMMITTED TOML file (``scripts/hf_jobs/campaigns/<name>.toml``, INV-7): name, bucket,
wall target, concurrency cap, and per job the trainer or driver, its tier, its dataset and its variants.
The MAIN run key is CONTENT-ADDRESSED (``{job}-{tier}``, ADR-129): a generation is named by its token, so
two commits or campaigns with equal tokens share one generation and unequal tokens never meet. A replica
(``{campaign}-{commit7}-{job}-{tier}-r<N>``, the INV-1 gate, P18) and a canary (``-canary``) stay FRESH per
campaign and commit.

The controller holds NO state of its own. Every step rebuilds the state from two sources only: each
instance's ``--plan`` JSON read against the bucket store (tokens, done/pending units per stage), and the
HF job labels (``list_jobs(labels=...)``: what is running, what ended in ERROR). It can stop at any time;
``resume`` (or a re-run of ``launch``) continues from the store. The scheduler is DAG-parallel with
action ownership and a critical-path grant order; several stages of an instance can run at once (no
one-stage barrier), bounded by the account concurrency cap.

The pure decision is `next_actions` (tested without HF); `step` applies it through the typed launcher.

Commands: ``python -m scripts.hf_jobs campaign <file> {preflight,profile,plan,launch,resume,status,
retrieve,verify,adopt,publish,cancel,summary}``.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any

from scripts.hf_jobs.planner import (
    Constraints,
    UnitCost,
    ideal_stage_wall,
    irreducible_floor_s,
    plan_reduce,
    plan_stage,
)
from scripts.hf_jobs.store import (  # ADR-129: defined in store.py (visibility-blind), re-exported here
    CanaryReadThroughStore as CanaryReadThroughStore,
)

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
    concurrency_cap: int
    attempts: int
    jobs: tuple[CampaignJob, ...]
    path: pathlib.Path | None
    sha256: str

    def constraints(self, *, startup_s: float | None = None, store_read_s: float | None = None) -> Constraints:
        base = Constraints(target_s=float(self.target_minutes) * 60.0, concurrency_cap=self.concurrency_cap)
        changes: dict[str, float] = {}
        if startup_s is not None:
            changes["startup_s"] = startup_s
        if store_read_s is not None:
            changes["store_read_s"] = store_read_s
        return dataclasses.replace(base, **changes) if changes else base


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
    role: str = "main"  # "main" | "replica" | "canary" (ADR-129, S5.9): the store's addressing role

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
    peak_rss_bytes: int | None = None
    partition: tuple[tuple[str, ...], ...] | None = None
    profile: int | None = None
    predicted_job_wall_s: float | None = None
    predicted_cost_usd: float | None = None
    reason: str | None = None
    #: The memory escalation: the job also runs units deferred by a memory failure (`--escalate`).
    escalate: bool = False
    #: The action's unique id (`make_action_id`), set by `step` before launch; every job of the action
    #: carries it as the `action` label, and its ownership record and partition paths key on it (S5.3).
    action_id: str | None = None
    #: True when, while any of its jobs runs, the action owns EVERY pending unit of its stage(s): a
    #: profile, a grouped/tiny job, a reduce, a round-robin run, a memory escalation of the whole stage.
    owns_all: bool = False


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
    return Campaign(
        name=c["name"],
        bucket=c["bucket"],
        target_minutes=int(c["target_minutes"]),
        concurrency_cap=int(c["concurrency_cap"]),
        attempts=int(c["attempts"]),
        jobs=jobs,
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def run_key(c: Campaign, commit: str, job: str, tier: str, *, replica: int | None = None, canary: bool = False) -> str:
    """The store run key (ADR-129, S5.9). The MAIN store is CONTENT-ADDRESSED: ``{job}-{tier}``, shared across
    commits and campaigns (a generation is named by its token, so equal tokens share one generation and
    unequal tokens never meet). A replica (``-r<N>``, the INV-1 gate, P18) and a canary (``-canary``) stay
    FRESH per campaign and commit, so they never read a main generation."""
    base = f"{job}-{tier}"
    if canary:
        return f"{c.name}-{commit[:7]}-{base}-canary"
    if replica is not None:
        return f"{c.name}-{commit[:7]}-{base}-r{replica}"
    return base


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


def instances(c: Campaign, commit: str, *, replica: int | None = None, canary: bool = False) -> list[Instance]:
    """Every (job, tier, variant) of the campaign with its module, args and datasets (spec 5.5). ``replica`` /
    ``canary`` select the FRESH replica/canary stores (ADR-129); the default is the content-addressed main store."""
    from scripts.hf_jobs import arm_policy
    from scripts.hf_jobs.datasets import declared, locked, staged_units
    from scripts.hf_jobs.registry import DriverSpec, spec_for

    out: list[Instance] = []
    for j in c.jobs:
        spec = spec_for(j.job)
        rk = run_key(c, commit, j.job, j.tier, replica=replica, canary=canary)
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
    role = "canary" if canary else ("replica" if replica is not None else "main")
    return out if role == "main" else [dataclasses.replace(i, role=role) for i in out]


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


@dataclasses.dataclass(frozen=True)
class Ownership:
    """What running actions own in one stage generation: every unit (profile, grouped, round-robin,
    reduce, whole-stage escalation) or the keys of their partitions / escalations (S5.3)."""

    all_units: bool
    keys: frozenset[str]


def make_action_id(instance_id: str, stages: Sequence[str], tokens: Sequence[str], partition, now) -> str:
    """``%Y%m%dT%H%M%SZ_<8 hex>``; the hash names the action itself (instance, stages, tokens, partition), so
    two actions of one instance in one second differ (r8 B R7-02). No ``-``: readers split `stage` on it."""
    digest = hashlib.sha256(
        json.dumps([instance_id, list(stages), list(tokens), partition], sort_keys=True, default=list).encode()
    ).hexdigest()[:8]
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}_{digest}"


def action_rel(pipeline: str, stage: str, token: str, action_id: str) -> str:
    return f"_actions/{pipeline}/{stage}/{token}/{action_id}.json"


def partition_rel(pipeline: str, stage: str, token: str, action_id: str, i: int, n: int) -> str:
    return f"_partitions/{pipeline}/{stage}/{token}/{action_id}/{i}of{n}.json"


def read_ownership_from_records(records: Iterable[Mapping]) -> dict[tuple[str, str], Ownership]:
    """Merge the action records whose jobs still run into ``{(stage, token): Ownership}``."""
    out: dict[tuple[str, str], Ownership] = {}
    for r in records:
        if not r.get("running"):
            continue
        k = (r["stage"], r["token"])
        prev = out.get(k, Ownership(False, frozenset()))
        out[k] = Ownership(prev.all_units or bool(r["owns_all"]), prev.keys | frozenset(r.get("keys") or ()))
    return out


def read_ownership(store: Any, plan: Mapping, jobs: Iterable[Any]) -> dict[tuple[str, str], Ownership]:
    """``{(stage, token): Ownership}`` from the action records in the store (S5.3): list each stage's
    ``_actions/<pipeline>/<stage>/<token>/`` once, read each record, mark it running when a non-terminal job
    on the account carries its action id, and merge. Ownership is read over ALL jobs of the run key (every
    campaign and commit sharing the store), so two concurrent campaigns never run one unit twice."""
    pipeline = plan["pipeline"]
    records: list[dict] = []
    for st in plan["stages"]:
        base = f"_actions/{pipeline}/{st['name']}/{st['token']}"
        for name in sorted(store.names(base)):
            raw = store.read_bytes(f"{base}/{name}")
            if raw:
                records.append(json.loads(raw))
    return read_ownership_from_records(mark_running(records, jobs))


def mark_running(records: Iterable[Mapping], jobs: Iterable[Any]) -> list[dict]:
    """Set each record's ``running`` from the jobs whose ``action`` label (through `parse_stage_label` for the
    R8 fallback) equals its ``action_id`` and whose status is not TERMINAL (S5.3)."""
    from scripts.hf_jobs.launch import TERMINAL, parse_stage_label

    live: set[str] = set()
    for j in jobs:
        lab = dict(getattr(j, "labels", None) or {})
        if str(getattr(getattr(j, "status", None), "stage", "") or "") in TERMINAL:
            continue
        action = lab.get("action")
        if not action:
            _, action = parse_stage_label(lab.get("stage", ""))
        if action:
            live.add(action)
    return [{**r, "running": r.get("action_id") in live} for r in records]


def allocate(items: Sequence[tuple[str, Action]], *, budget: int) -> list[tuple[str, Action, int]]:
    """Grant the launch budget to ``(instance id, action)`` pairs in the given order (S5.5 passes them in
    critical-path order). A round-robin or ``owns_all`` action is all-or-nothing; an LPT action may take a
    partial grant (its unowned remainder waits for the next step). Returns ``(instance id, action, n)``."""
    out: list[tuple[str, Action, int]] = []
    for inst_id, a in items:
        if budget <= 0:
            break
        want = a.n_jobs if a.partition is None else len(a.partition)
        if a.owns_all:  # a profile, a grouped job, a round-robin run or a reduce cannot run partially
            n = want if budget >= want else 0
        else:  # an LPT (or escalation) action takes a partial grant; its remainder waits for the next step
            n = min(budget, want)
        if n > 0:
            out.append((inst_id, a, n))
            budget -= n
    return out


def remaining_critical_path(plan: Mapping, history: Mapping[str, Mapping], c: Constraints) -> float:
    """The longest chain of not-yet-reduced stages for ONE instance, each costed at its planned wall (S5.5),
    by DP over ``needs`` in DAG order. The scheduler grants the budget to the longest remaining chain first."""
    stages = {s["name"]: s for s in plan["stages"]}

    def stage_wall(st: Mapping) -> float:
        if st.get("reduced"):
            return 0.0
        hk = st.get("history_key") or st["name"]
        h = history.get(hk)
        n = int(st.get("n_units") or 0)
        if not h or not n:
            return c.startup_s + c.target_s  # no history: the irreducible target is the best estimate
        # The PLANNED wall of the stage (S5.5, r-impl CCTP-IMPL-02): the makespan the planner would launch
        # with no grant ceiling, so the chain cost matches what the scheduler actually runs (not a fluid
        # single-job estimate). Only affects grant ORDER, never an output.
        keys = [str(k) for k in (st.get("units") or range(n))]
        p = plan_stage(
            st["name"],
            keys,
            h["cost"],
            inner_parallel=bool(st.get("inner_parallel")),
            n_inner=int(st.get("inner_width") or 1),
            efficiency=h.get("efficiency"),
            per_unit=h.get("per_unit"),
            c=c,
        )
        return p.predicted_job_wall_s

    memo: dict[str, float] = {}

    def longest(name: str) -> float:
        if name in memo:
            return memo[name]
        st = stages[name]
        up = max(
            (longest(u) for u in (st.get("needs") or ()) if u in stages and not stages[u].get("reduced")),
            default=0.0,
        )
        memo[name] = stage_wall(st) + up
        return memo[name]

    return max((longest(name) for name in stages), default=0.0)


def grant_order(candidates: Sequence[tuple[str, float, Action]]) -> list[tuple[str, Action]]:
    """``(instance id, remaining critical path, action)`` sorted by the longest chain first, then instance id
    (a pure function of the state, so the order never depends on the input order)."""
    return [(i, a) for i, _, a in sorted(candidates, key=lambda t: (-t[1], t[0]))]


def _stage_action(
    stages: Sequence[Mapping],
    i: int,
    st: Mapping,
    reduced: set[str],
    own: Ownership | None,
    history: Mapping[str, Mapping],
    c: Constraints,
    *,
    lpt: bool,
    grant,
) -> Action | None:
    """The 74d57ba per-stage decision, restricted to the units no running action owns (S5.3): a reduce only
    when NOTHING is pending; an escalation of the unowned memory units; otherwise a tiny/profile/planner run
    over ``todo`` (pending + retryable failures, minus the owned and memory units)."""
    target = int(2 * c.target_s)
    inner_w = int(st.get("inner_width") or 1)
    failed = dict(st.get("failed_units") or {})
    retry = {k for k, n in failed.items() if int(n) < c.unit_attempts}
    memory = retry & set(st.get("memory_units") or ())
    owned_keys = own.keys if own is not None else frozenset()
    pending = set(st.get("pending_units") or ()) | retry
    if pending & owned_keys and not (pending - owned_keys - memory):
        return None  # every pending unit is owned by a running action: WAIT (no reduce, no duplicate)
    todo = sorted(pending - owned_keys - memory)
    unowned_memory = sorted(memory - owned_keys)
    if not todo and unowned_memory:
        # a unit over its fair-share cap: alone in its job, with the whole box (W=1, cap = usable RAM)
        alone = tuple((k,) for k in unowned_memory)
        return Action(
            "run",
            (st["name"],),
            len(alone) if lpt else 1,
            1,
            1,
            target,
            cap_bytes=c.usable_bytes,
            partition=alone if lpt else None,
            escalate=True,
            # A round-robin escalation (no partition, download instance) owns the whole stage, so the next
            # step never relaunches it (CCTP-IMPL-08); an LPT escalation owns the keys it names.
            owns_all=not lpt,
        )
    if not pending:  # nothing pending or owned: reduce / tolerant reduce / halt
        hk = st.get("history_key") or st["name"]
        rp = plan_reduce(int(st.get("done") or 0), c, per_result_s=history.get(hk, {}).get("reduce_per_result_s"))
        spent = {k: n for k, n in sorted(failed.items()) if k not in retry}
        if not spent:
            return Action("reduce", (st["name"],), 1, 1, 1, rp.timeout_s, owns_all=True, predicted_job_wall_s=rp.wall_s)
        tolerance = st.get("max_failures")
        if tolerance is None or len(spent) <= int(tolerance):
            why = f"{st['name']}: reduces without unit(s) failed {c.unit_attempts} time(s): {spent}"
            return Action(
                "reduce",
                (st["name"],),
                1,
                1,
                1,
                rp.timeout_s,
                reason=why,
                owns_all=True,
                predicted_job_wall_s=rp.wall_s,
            )
        why = (
            f"{st['name']}: {len(spent)} unit(s) failed {c.unit_attempts} time(s), over the stage's "
            f"max_failures={tolerance}: {spent}"
        )
        return Action("halt", (st["name"],), 0, 0, 0, 0, reason=why)
    if not todo:
        return None  # remaining pending units are owned or escalating elsewhere: nothing new now
    if st["n_units"] <= 2:
        inner = min(c.vcpu, inner_w) if st.get("inner_parallel") else 1
        return Action("run", _group(stages, i, reduced), 1, 1, inner, target, owns_all=True)
    hk = st.get("history_key") or st["name"]
    if hk not in history:
        inner = min(c.vcpu, inner_w) if st.get("inner_parallel") else 1
        workers = 1 if st.get("inner_parallel") else c.vcpu
        sample = min(int(st["n_units"]), 2 * c.vcpu)
        return Action("profile", (st["name"],), 1, workers, inner, target, profile=sample, owns_all=True)
    h = history[hk]
    measured = h.get("per_unit") if lpt else None
    per_unit = {k: float(measured.get(k, h["cost"].p50_s)) for k in todo} if measured else None

    def _plan(ceiling: int | None):
        return plan_stage(
            st["name"],
            todo,
            h["cost"],
            inner_parallel=bool(st.get("inner_parallel")),
            n_inner=inner_w,
            efficiency=h.get("efficiency"),
            per_unit=per_unit,
            c=c,
            grant=ceiling,
            lpt=lpt,
        )

    p = _plan(None)
    if grant is not None:  # the scheduler's per-action job budget (S5.5): re-plan within the ceiling it grants
        prelim = Action("run", (st["name"],), p.n_jobs, p.workers, p.inner_workers, p.timeout_s, partition=p.partition)
        p = _plan(int(grant(prelim)))
    return Action(
        "run",
        (st["name"],),
        p.n_jobs,
        p.workers,
        p.inner_workers,
        p.timeout_s,
        cap_bytes=p.cap_bytes,
        peak_rss_bytes=p.peak_rss_bytes,
        partition=p.partition,
        predicted_job_wall_s=p.predicted_job_wall_s,
        predicted_cost_usd=p.predicted_cost_usd,
        owns_all=p.partition is None,  # a round-robin run (download) owns the whole stage; an LPT run its keys
    )


def next_actions(
    plan: Mapping,
    owned: Mapping[tuple[str, str], Ownership],
    history: Mapping[str, Mapping],
    c: Constraints,
    *,
    lpt: bool = True,
    grant=None,
) -> list[Action]:
    """EVERY action to take for ONE instance now (S5.3): one per stage whose needs are all reduced and whose
    pending units are not all owned by running actions. A reduced stage, a not-ready stage, or a stage with
    unknown units is skipped (NOT a barrier: a later independent stage may still act). Grouping, retries,
    memory escalation, tolerant reduce, halt and profile keep their 74d57ba rules per stage (`_stage_action`).
    ``grant(action) -> int`` caps an action's jobs (the scheduler's budget, S5.5)."""
    stages = list(plan["stages"])
    reduced = {s["name"] for s in stages if s.get("reduced")}
    grouped: set[str] = set()
    out: list[Action] = []
    for i, st in enumerate(stages):
        if st.get("reduced") or st["name"] in grouped:
            continue
        if not st.get("ready") or st.get("n_units") is None:
            continue  # NOT a return: a later independent stage may be ready
        own = owned.get((st["name"], st["token"]))
        if own is not None and own.all_units:
            continue
        a = _stage_action(stages, i, st, reduced, own, history, c, lpt=lpt, grant=grant)
        if a is None:
            continue
        grouped |= set(a.stages[1:])
        out.append(a)
    return out


# ---- reading state -------------------------------------------------------------------------------


def store_uri(bucket: str, key: str) -> str:
    return f"hf://buckets/{bucket}/{key}"


def read_plan(inst: Instance, *, bucket: str, run=subprocess.run) -> dict:
    """The instance's ``--plan`` JSON against its bucket store (read-only; the module is the authority). The
    subprocess runs under ``env_contract.controller_plan_env`` (output switches stripped): an operator's shell
    switch must not move the ``--plan`` token away from the job's token (CCTP-IMPL-26)."""
    from scripts.hf_jobs.env_contract import controller_plan_env

    argv = [sys.executable, "-m", inst.module, *inst.args, "--plan", "--store", store_uri(bucket, inst.run_key)]
    p = run(argv, capture_output=True, text=True, check=False, cwd=REPO_ROOT, env=controller_plan_env(os.environ))
    if p.returncode != 0:
        raise RuntimeError(f"{inst.id}: --plan failed (rc={p.returncode}): {(p.stderr or '')[-2000:]}")
    return json.loads(p.stdout)


def canary_key(units: Sequence[str], token: str, head: str) -> str:
    """The ONE canary unit for an attested stage (S5.10): deterministic per (token, HEAD), so the controller and
    every reader agree, and it MOVES with HEAD (a new commit re-samples). `min` of ``sha256(token+head+key)``."""
    return min(units, key=lambda k: hashlib.sha256((token + head + k).encode("utf-8")).hexdigest())


def plan_for_role(
    plan: Mapping[str, Any], role: str, attested: set[tuple[str, str]], head: str
) -> tuple[dict, dict[str, list[str]]]:
    """The plan view for a ``main`` / ``replica`` / ``canary`` instance and, for a canary, the unit keys per stage
    (S5.10). An ATTESTED stage (``(name, token) in attested``): the replica SKIPS it (marked reduced, nothing to
    re-materialize), and the canary runs ONE unit (`canary_key`) -- or the whole stage when it has one unit."""
    stages: list[dict] = []
    canary: dict[str, list[str]] = {}
    for st in plan["stages"]:
        is_att = (st["name"], st["token"]) in attested
        view = dict(st)
        if role == "replica" and is_att:
            view["reduced"] = True
        stages.append(view)
        if role == "canary" and is_att:
            units = st["units"]
            canary[st["name"]] = list(units) if st["n_units"] <= 1 else [canary_key(units, st["token"], head)]
    return {**plan, "stages": stages}, canary


def canary_verdict(*, main_bytes: bytes, canary_bytes: bytes, quarantine: Callable[[], object]) -> bool:
    """True when the canary unit reproduced the main store's bytes; on a mismatch it QUARANTINES the generation
    (Task 15) and returns False (unsound reuse -- the summary reports it to the owner, S5.10)."""
    if main_bytes == canary_bytes:
        return True
    quarantine()
    return False


#: S7.2 acceptance bars, DECLARED before the run (INV-9). A bar passes when its value is within the threshold
#: in the direction its formula sets (`compute_bars`).
BAR_THRESHOLDS: dict[str, Any] = {
    "schedule_efficiency": 1.5,  # makespan / lower bound
    "planner_calibration": (0.75, 1.25),  # actual / predicted band, per campaign and per action over 10 min
    "reduce": 1.5,  # reduce wall / plan_reduce prediction, per reduce over 1000 results
    "startup": 1.5,  # measured startup / image.lock startup_s, and NO install phase
    "cost": 1.1,  # main-run job-hours / Cycle 1
    "determinism": 1.0,  # fraction of replica/canary comparisons that matched (1.0 == all equal)
}


def _epoch(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        dt = datetime.datetime.fromisoformat(iso)
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=datetime.timezone.utc)).timestamp()


def compute_bars(
    *,
    jobs: Sequence[Mapping],
    predictions: Sequence[Mapping],
    stages: Sequence[Mapping],
    reduce_records: Sequence[Mapping],
    job_phases: Sequence[Mapping],
    cycle1_main_job_hours: float,
    cap: int,
    declared_startup_s: float,
    comparisons: Sequence[bool],
    cost_exclude_ids: Collection[str] = (),
) -> dict[str, dict]:
    """The six S7.2 acceptance bars from the campaign's records (INV-9; thresholds are `BAR_THRESHOLDS`). Each
    value is ``{"value", "threshold", "pass"}``; bars 2 and 3 also carry ``per_action`` / ``per_reduce`` rows."""
    main_jobs = [j for j in jobs if j.get("role", "main") == "main"]
    starts = [_epoch(j.get("started_at")) for j in main_jobs]
    ends = [_epoch(j.get("finished_at")) for j in main_jobs]
    starts = [s for s in starts if s is not None]
    ends = [e for e in ends if e is not None]
    makespan = (max(ends) - min(starts)) if starts and ends else 0.0

    # bar 1: makespan vs the lower bound (ideal critical path, or total job-hours / cap)
    cons = Constraints(target_s=600.0, concurrency_cap=int(cap), startup_s=float(declared_startup_s))
    reduce_by_stage: dict[str, float] = {}
    reduce_separate: dict[str, bool] = {}
    for r in reduce_records:
        st = r["stage"]
        w = float(r["wall_s"]) if r.get("wall_s") is not None else None
        if w is not None and (st not in reduce_by_stage or w < reduce_by_stage[st]):
            reduce_by_stage[st] = w
            reduce_separate[st] = not bool(r.get("in_job"))
    node: dict[str, float] = {}
    needs: dict[str, list[str]] = {}
    for s in stages:
        name = s["stage"]
        cost = UnitCost(**s["unit_cost"]) if s.get("unit_cost") else None
        ideal = (
            ideal_stage_wall(
                dict(s.get("per_unit") or {}),
                cost,
                cons,
                inner_parallel=bool(s.get("inner_parallel")),
                n_inner=int(s.get("inner_width") or 1),
                efficiency=s.get("efficiency"),
            )
            if cost and s.get("per_unit")
            else 0.0
        )
        add = reduce_by_stage.get(name, 0.0) + (cons.startup_s if reduce_separate.get(name) else 0.0)
        node[name] = ideal + add
        needs[name] = list(s.get("needs") or [])
    memo: dict[str, float] = {}

    def _cp(name: str) -> float:
        if name not in memo:
            up = max((_cp(u) for u in needs.get(name, []) if u in node), default=0.0)
            memo[name] = node[name] + up
        return memo[name]

    critical = max((_cp(n) for n in node), default=0.0)
    total_job_s = sum(float(j["wall_s"]) for j in main_jobs if j.get("wall_s") is not None)
    lb = max(critical, total_job_s / max(1, int(cap)))
    eff = makespan / lb if lb else 0.0
    bars: dict[str, dict] = {
        "schedule_efficiency": {
            "value": eff,
            "threshold": BAR_THRESHOLDS["schedule_efficiency"],
            "pass": eff <= BAR_THRESHOLDS["schedule_efficiency"],
        }
    }

    # bar 2: planner calibration, per campaign and per action over 10 min
    lo, hi = BAR_THRESHOLDS["planner_calibration"]
    pred_wall = {p["action_id"]: float(p["predicted_wall_s"]) for p in predictions if p.get("action_id")}
    actual_wall: dict[str, float] = {}
    for j in main_jobs:
        aid = dict(j.get("labels") or {}).get("action")
        if aid and j.get("wall_s") is not None:
            actual_wall[aid] = actual_wall.get(aid, 0.0) + float(j["wall_s"])
    camp_pred = next((float(p["makespan_s"]) for p in predictions if p.get("kind") == "campaign"), None)
    camp_ratio = makespan / camp_pred if camp_pred else 1.0
    per_action = []
    for aid, pw in sorted(pred_wall.items()):
        aw = actual_wall.get(aid)
        if aw is None or aw <= 600.0:  # only actions over 10 minutes are rated
            continue
        ratio = aw / pw if pw else 1.0
        per_action.append({"action_id": aid, "ratio": ratio, "pass": lo <= ratio <= hi})
    cal_pass = (lo <= camp_ratio <= hi) and all(r["pass"] for r in per_action)
    bars["planner_calibration"] = {
        "value": camp_ratio,
        "threshold": [lo, hi],
        "pass": cal_pass,
        "per_action": per_action,
    }

    # bar 3: reduce wall vs plan_reduce prediction, per reduce over 1000 results
    per_reduce = []
    for r in reduce_records:
        if int(r.get("n_results") or 0) <= 1000 or not r.get("predicted_wall_s"):
            continue
        ratio = float(r["wall_s"]) / float(r["predicted_wall_s"])
        per_reduce.append({"stage": r["stage"], "token": r.get("token"), "ratio": ratio})
    worst = max((r["ratio"] for r in per_reduce), default=0.0)
    bars["reduce"] = {
        "value": worst,
        "threshold": BAR_THRESHOLDS["reduce"],
        "pass": all(r["ratio"] <= BAR_THRESHOLDS["reduce"] for r in per_reduce),
        "per_reduce": per_reduce,
    }

    # bar 4: startup (no install phase; measured startup / declared)
    ratios = []
    has_install = False
    for rec in job_phases:
        ph = rec.get("phases") or {}
        if "install" in ph:
            has_install = True
        if "first_unit" in ph and "script_start" in ph:
            # startup EXCLUDES the dataset fetch window (CCTP-IMPL-04): first_unit - script_start, minus the
            # measured dataset fetch (after_dataset_fetch - after_fetch) when both marks are present.
            startup = float(ph["first_unit"]) - float(ph["script_start"])
            if "after_dataset_fetch" in ph and "after_fetch" in ph:
                startup -= float(ph["after_dataset_fetch"]) - float(ph["after_fetch"])
            ratios.append(startup / declared_startup_s if declared_startup_s else 0.0)
    from scripts.hf_jobs.ledger import percentile as _pct

    startup_val = _pct(ratios, 0.5) if ratios else 0.0
    bars["startup"] = {
        "value": startup_val,
        "threshold": BAR_THRESHOLDS["startup"],
        "pass": (not has_install) and startup_val <= BAR_THRESHOLDS["startup"],
    }

    # bar 5: cost (main-run job-hours / Cycle 1). Excludes replica/canary (role filter above), the profile and
    # Cycle-1 jobs named in ``cost_exclude_ids``, and reused units (which ran no job, so carry no wall here).
    cost_s = sum(
        float(j["wall_s"]) for j in main_jobs if j.get("wall_s") is not None and j.get("id") not in cost_exclude_ids
    )
    cost_ratio = (cost_s / 3600.0) / cycle1_main_job_hours if cycle1_main_job_hours else 0.0
    bars["cost"] = {
        "value": cost_ratio,
        "threshold": BAR_THRESHOLDS["cost"],
        "pass": cost_ratio <= BAR_THRESHOLDS["cost"],
    }

    # bar 6: determinism (every replica/canary comparison equal). NO comparisons is NOT a pass -- determinism was
    # not measured, so the bar cannot claim it held (B#35: an empty list silently passed before).
    frac = (sum(1 for x in comparisons if x) / len(comparisons)) if comparisons else 0.0
    bars["determinism"] = {
        "value": frac,
        "threshold": BAR_THRESHOLDS["determinism"],
        "pass": bool(comparisons) and frac >= BAR_THRESHOLDS["determinism"],
        "n_comparisons": len(comparisons),
    }
    return bars


def canary_allow_rel(campaign: str, commit: str, action_id: str) -> str:
    """Where a canary action records the upstream generations its read-through may serve (S5.10)."""
    return f"_campaign/{campaign}/{commit[:7]}/canary/{action_id}.allow.json"


def attested_stages(store: Any, plan: Mapping, head: str) -> set[tuple[str, str]]:
    """The ``(stage, token)`` pairs of ``plan`` that carry a live attestation on HEAD or an ancestor (S5.10).
    A quarantined generation has no attestation left (`quarantine_generation` moves the ``_attest/`` tree)."""
    from scripts.hf_jobs import adopt

    out: set[tuple[str, str]] = set()
    for st in plan["stages"]:
        atts = adopt.attestations(store, st["name"], st["token"])
        if atts and any(adopt.is_ancestor(a.get("commit", ""), head) for a in atts):
            out.add((st["name"], st["token"]))
    return out


def _schedule_canary(c: Campaign, commit: str, *, api: Any, token: str, budget: int, report: dict, now) -> int:
    """Launch ONE canary unit per attested stage (S5.10): it re-runs the unit over the REUSED upstream (read
    through the main store, gated by the allow file) so `verify` can compare its bytes to main. Returns the
    remaining budget. Nothing runs when no stage is attested."""
    from scripts.hf_jobs import launch as launcher
    from scripts.hf_jobs.launch import BUCKET_MOUNT, JobSpec
    from scripts.hf_jobs.store import BucketStore

    cons = planning_constraints(c)
    for ci in instances(c, commit, canary=True):
        if budget <= 0:
            break
        main_rk = run_key(c, commit, ci.job, ci.tier)
        main_store = BucketStore(c.bucket, main_rk, api=api)
        canary_store = BucketStore(c.bucket, ci.run_key, api=api)
        main_plan = read_plan(dataclasses.replace(ci, run_key=main_rk, role="main"), bucket=c.bucket)
        attested = attested_stages(main_store, main_plan, commit)
        if not attested:
            continue
        _, canary_units = plan_for_role(main_plan, "canary", attested, commit)
        # The reusable upstream for a canary: the generation of every ATTESTED, non-quarantined stage. A canary
        # job for stage X reuses all of these from main EXCEPT X's own (which it recomputes), so its allow set
        # is per-stage, never a whole-instance "stages not in canary_units" (that leaves a chained canary with
        # no upstream and quarantines a sound generation -- B#29/#45-A). ``attested`` is already non-quarantined.
        attested_gen = {s["name"]: s["generation"] for s in main_plan["stages"] if (s["name"], s["token"]) in attested}
        rpt = report.setdefault(ci.id, {"action": "canary", "stages": sorted(canary_units), "jobs": [], "failed": None})
        for st in main_plan["stages"]:
            if budget <= 0:
                break
            keys = canary_units.get(st["name"])
            if not keys:
                continue
            allow = sorted(g for name, g in attested_gen.items() if name != st["name"])
            aid = make_action_id(ci.id, (st["name"],), [st["token"]], (tuple(keys),), now)
            sk_rel = partition_rel(main_plan["pipeline"], st["name"], st["token"], aid, 0, 1)
            canary_store.upload_bytes(sk_rel, json.dumps(list(keys)).encode("utf-8"))
            allow_rel = canary_allow_rel(c.name, commit, aid)
            canary_store.upload_bytes(allow_rel, json.dumps(allow).encode("utf-8"))
            spec = JobSpec(
                campaign=c.name,
                run_key=ci.run_key,
                job=ci.job,
                variant=ci.variant,
                stages=(st["name"],),
                shard=(0, 1),
                module=ci.module,
                args=ci.args,
                commit=commit,
                bucket=c.bucket,
                workers=1,
                inner_workers=1,
                cap_bytes=None,
                peak_rss_bytes=None,
                timeout_s=int(2 * cons.target_s),
                attempts=c.attempts,
                datasets=ci.datasets,
                shard_keys_path=f"{BUCKET_MOUNT}/{ci.run_key}/{sk_rel}",
                read_through=f"{BUCKET_MOUNT}/{main_rk}",
                read_through_allow=f"{BUCKET_MOUNT}/{ci.run_key}/{allow_rel}",
                action_id=aid,
                action_kind="run",
            )
            rpt["jobs"].append(launcher.launch(spec, api=api, token=token))
            budget -= 1
    return budget


#: Cycle 1 profile jobs excluded from the S7.2 bar-5 baseline denominator (exploration, not production; B#35).
CYCLE1_EXCLUDED: tuple[str, ...] = ("6ad89c3b", "6ad8ccde")


def _cycle1_main_job_hours() -> float:
    """Cycle 1's main-run job-hours (S7.2 bar 5 denominator) from its committed summary, EXCLUDING the Cycle 1
    profile jobs (``CYCLE1_EXCLUDED``); 1.0 when absent (the ratio is then meaningless until the Cycle 1
    history is exported, E-5)."""
    p = SUMMARY_DIR / "cycle1.json"
    if not p.is_file():
        return 1.0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 1.0
    total = sum(
        float(j["wall_s"])
        for j in data.get("jobs", [])
        if j.get("wall_s") is not None and not str(j.get("id", "")).startswith(CYCLE1_EXCLUDED)
    )
    return total / 3600.0 if total else 1.0


def _declared_startup_s(cons: Constraints) -> float:
    """The declared job startup for the startup bar: the built image's measured ``startup_s`` (Task 18) when
    the lock carries it, else the planner's measured startup."""
    import importlib

    try:  # dynamic import keeps `image` off campaign's static import graph; fall back before the lock is probed
        image = importlib.import_module("scripts.hf_jobs.image")
        s = image.load_lock().get("startup_s")
    except (ImportError, OSError, ValueError, KeyError, AttributeError):
        s = None
    return float(s) if s else float(cons.startup_s)


def committed_history() -> dict[str, dict]:
    """Stage cost history from the committed campaign summaries (the planner learns from them, S5.7). Merges
    the per-unit cost exports (``*.per_unit.json``, E-5) so LPT uses the measured wall of every unit; a stage
    row's own ``per_unit`` (written by the Cycle 1b `cmd_summary`) wins over the export."""
    from scripts.hf_jobs.ledger import load_history

    summaries = [p for p in SUMMARY_DIR.glob("*.json") if not p.name.endswith(".per_unit.json")]
    out: dict[str, dict] = {}
    for hk, st in load_history(summaries).items():
        cost = st.get("unit_cost")
        if cost:
            out[hk] = {
                "cost": UnitCost(**cost),
                "efficiency": st.get("efficiency"),
                "per_unit": st.get("per_unit"),
                "reduce_per_result_s": st.get("reduce_per_result_s"),
            }
    for p in sorted(SUMMARY_DIR.glob("*.per_unit.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for hk, per_unit in (data.get("per_unit") or {}).items():
            if hk in out and out[hk].get("per_unit") is None:  # a stage row's own per_unit wins
                out[hk]["per_unit"] = per_unit
    return out


def planning_constraints(c: Campaign, *, summary_dir: pathlib.Path | None = None) -> Constraints:
    """The campaign's constraints with the job startup the committed summaries measured (spec 5.5: the
    planner learns from them); the `Constraints` default applies before any campaign measured one."""
    from scripts.hf_jobs.datasets import store_settings
    from scripts.hf_jobs.ledger import median_startup_s

    return c.constraints(
        startup_s=median_startup_s((summary_dir or SUMMARY_DIR).glob("*.json")),
        store_read_s=store_settings().per_result_read_s,
    )


def live_history(plan: Mapping, store: Any, cache: dict) -> dict[str, dict]:
    """Costs measured in THIS store (profile units, finished units), keyed by history key (decision P14).
    Linear (S5.11): reads one per-job cost aggregate per job (`ledger.cost_aggregates`) and a unit record only
    for a done unit that no aggregate carries (a job that crashed before it wrote its aggregate)."""
    from scripts.hf_jobs import ledger

    pipeline = plan["pipeline"]
    out: dict[str, dict] = {}
    for st in plan["stages"]:
        if not st.get("done"):
            continue
        gen = st["generation"]
        if gen not in cache:
            aggs = ledger.cost_aggregates(store, pipeline, st["name"], st["token"])
            done_units = (
                set(st.get("units") or ()) - set(st.get("pending_units") or ()) - set(st.get("failed_units") or ())
            )
            rows: dict[str, dict] = dict(aggs)
            for k in done_units - set(aggs):  # excluded units have no record either; ignore a missing one
                raw = store.read_bytes(f"{gen}/{k}.ledger.json")
                if raw:
                    rows[k] = json.loads(raw)
            prof = store.read_bytes(f"{gen}/_profile.json")
            eff = json.loads(prof).get("efficiency") if prof else None
            cost = ledger.unit_cost(list(rows.values()))
            per_unit = {
                k: float(r["wall_s"])
                for k, r in rows.items()
                if r.get("outcome") == "done" and r.get("wall_s") is not None
            }
            reds = ledger.reduce_records(store, pipeline, st["name"], st["token"])
            per_result = [
                r["wall_s"] / r["n_results"] for r in reds if r.get("n_results") and r.get("wall_s") is not None
            ]
            cache[gen] = (
                None
                if cost is None
                else {
                    "cost": cost,
                    "efficiency": eff,
                    "per_unit": per_unit,
                    "reduce_per_result_s": ledger.percentile(per_result, 0.5) if per_result else None,
                }
            )
        if cache[gen] is not None:
            out[st.get("history_key") or st["name"]] = cache[gen]
    return out


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


def _created_epoch(info: Any) -> float:
    """A job's ``created_at`` as an epoch float (a datetime, an ISO string or a number all compare)."""
    ca = getattr(info, "created_at", None)
    if ca is None:
        return 0.0
    if hasattr(ca, "timestamp"):
        return ca.timestamp()
    try:
        return float(ca)
    except (TypeError, ValueError):
        return 0.0


def _trailing_errors(history: Sequence[Any], operator_after: float | None = None) -> int:
    """How many of the stage's newest jobs ended in ERROR in a row (a COMPLETED job ends the run). With
    ``operator_after`` (the newest operator ``--clear-halt`` record time, S5.8), a job created at or before
    it ends the count, so the operator's reset clears the halt."""
    n = 0
    for info in history:
        if operator_after is not None and _created_epoch(info) <= operator_after:
            break
        if str(getattr(getattr(info, "status", None), "stage", "")) != "ERROR":
            break
        n += 1
    return n


def _job_phase_records(store: Any) -> dict[str, dict]:
    """Every job phase record (`_ledger/_jobs/`) by its in-job label, listed once per step (S5.8, S5.11)."""
    out: dict[str, dict] = {}
    for name in sorted(store.names("_ledger/_jobs")):
        raw = store.read_bytes(f"_ledger/_jobs/{name}")
        if raw:
            rec = json.loads(raw)
            if rec.get("label"):
                out[rec["label"]] = rec
    return out


def _make_infra_checker(store: Any, jobs: Iterable[Any]) -> Callable[[Any], bool]:
    """A predicate: the job ran no unit (no ``first_unit`` in its phase record), so its ERROR is an INFRA
    failure (a mount, an image pull or a crash before the first unit), not a deterministic unit failure."""
    from scripts.hf_jobs.launch import label_of

    recs = _job_phase_records(store)

    def is_infra(job: Any) -> bool:
        label = label_of(dict(getattr(job, "labels", None) or {}))
        rec = recs.get(label)
        # A job that reached a unit (`first_unit`) or ran a separate reduce (`reduce_start`, CCTP-IMPL-11) is
        # not an infra failure; one with no phase record, or that stopped before either, is.
        return rec is None or not ({"first_unit", "reduce_start"} & set(rec.get("phases") or {}))

    return is_infra


def _newest_operator(store: Any, pipeline: str, stage: str, token: str) -> float | None:
    """The newest operator ``--clear-halt`` record time for the stage (S5.8), or None."""
    base = f"_operator/{pipeline}/{stage}/{token}"
    times: list[float] = []
    for name in sorted(store.names(base)):
        raw = store.read_bytes(f"{base}/{name}")
        if raw:
            when = json.loads(raw).get("when")
            if when:
                try:
                    dt = datetime.datetime.fromisoformat(when)
                    times.append((dt if dt.tzinfo else dt.replace(tzinfo=datetime.timezone.utc)).timestamp())
                except ValueError:
                    continue
    return max(times) if times else None


def _error_cause(job: Any, infra: Callable[[Any], bool] | None) -> str:
    """The halt cause class (S5.8): ``infra`` (a mount, an image pull or a crash before the first unit),
    ``timeout`` (the job ran out of time), else ``unit`` (a unit failed deterministically)."""
    if infra is not None and infra(job):
        return "infra"
    if _timed_out(job):
        return "timeout"
    return "unit"


def _timed_out(info: Any) -> bool:
    return "timeout" in str(getattr(getattr(info, "status", None), "message", "") or "").lower()


def resume_after_error(
    plan: Mapping,
    stage: str,
    history: Sequence[Any],
    c: Constraints,
    *,
    infra: Callable[[Any], bool] | None = None,
    operator_after: float | None = None,
) -> str | None:
    """None when the next action may follow a stage whose latest job ended in ERROR, else the halt reason
    ("" = the caller's default). It may when the latest ERROR is an INFRA failure (a mount, an image pull or
    a crash before the first unit; ``infra(job)`` True, S5.8), when the stage still has a retryable failed
    unit (the retry stays bounded by ``unit_attempts``), or when the job ran out of time with units pending.
    EVERY resume is bounded by ``error_resumes`` ERRORs in a row (``operator_after`` resets the count); the
    halt reason then names the cause class."""
    st = next((s for s in plan["stages"] if s["name"] == stage), None)
    if st is None:
        return ""
    errors = _trailing_errors(history, operator_after)
    if errors >= c.error_resumes:
        cause = _error_cause(history[0], infra) if history else "unit"
        return f"{stage}: {errors} job(s) in a row ended in ERROR (cause {cause})"
    if history and operator_after is not None and _created_epoch(history[0]) <= operator_after:
        # An operator --clear-halt record is newer than the latest ERROR: resume the stage (S5.8,
        # CCTP-IMPL-03), so the next step launches it instead of halting on the caller's default.
        return None
    if history and infra is not None and infra(history[0]):
        return None  # a transient infra failure resumes, bounded by error_resumes
    if any(int(n) < c.unit_attempts for n in (st.get("failed_units") or {}).values()):
        return None
    if history and _timed_out(history[0]):
        # A timeout (a unit job OR a separate reduce job) retries with the planned-size timeout, bounded by
        # error_resumes; a single reduce timeout must not dead-halt the stage (CCTP-IMPL-03).
        return None
    return ""


def clear_halt_guard(*, halted: bool) -> None:
    """Refuse ``resume --clear-halt`` unless the stage is halted (S5.8): the operator record only resets a
    real halt, never a running or healthy stage."""
    if not halted:
        raise SystemExit("refuse --clear-halt: the stage is not halted")


def halt_report(a: Action) -> dict:
    """The step report of a ``halt`` action: nothing is launched; the reason names the failed units."""
    return {"action": "halt", "stages": a.stages, "jobs": [], "failed": a.reason}


def _job_specs(c: Campaign, commit: str, inst: Instance, plan: Mapping, a: Action, *, store: Any) -> list[Any]:
    from scripts.hf_jobs.launch import JobSpec

    if a.partition is not None and any(d.access == "download" for d in inst.datasets):
        raise ValueError(f"{inst.id}: an LPT partition needs mounted datasets (download access fetches by shard)")
    action_kind = "reduce" if a.kind == "reduce" else "run"

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
            peak_rss_bytes=a.peak_rss_bytes,
            timeout_s=a.timeout_s,
            attempts=c.attempts,
            datasets=inst.datasets,
            profile=a.profile,
            shard_keys_path=shard_keys_path,
            escalate=a.escalate,
            action_id=a.action_id,
            action_kind=action_kind,
        )

    if a.partition is None:
        return [spec((i, a.n_jobs)) for i in range(a.n_jobs)]
    st = next(s for s in plan["stages"] if s["name"] == a.stages[0])
    n = len(a.partition)
    specs = []
    for i, part in enumerate(a.partition):
        # Under the action id (S5.3): a second grant with the same `n` never overwrites the first partition.
        rel = partition_rel(plan["pipeline"], st["name"], st["token"], a.action_id or "noaction", i, n)
        store.upload_bytes(rel, json.dumps(list(part)).encode("utf-8"))
        specs.append(spec((i, n), f"/scratch/{inst.run_key}/{rel}"))
    return specs


def _running(jobs: Iterable[Any]) -> tuple[dict[tuple[str, str], set[str]], dict[tuple[str, str, str], Any]]:
    """``{(run_key, variant): {running stage labels}}`` and the LATEST job per (run_key, variant, stage)."""
    from scripts.hf_jobs.launch import TERMINAL, parse_stage_label

    running: dict[tuple[str, str], set[str]] = {}
    latest: dict[tuple[str, str, str], Any] = {}
    for info in jobs:
        lab = dict(getattr(info, "labels", None) or {})
        key = (lab.get("run_key", ""), lab.get("variant", ""))
        stage = lab.get("stage", "")
        status = str(getattr(getattr(info, "status", None), "stage", "") or "")
        if status not in TERMINAL:
            for s in parse_stage_label(stage)[0]:
                running.setdefault(key, set()).add(s)
        k3 = (*key, stage)
        prev = latest.get(k3)
        if prev is None or (getattr(info, "created_at", None) or 0) > (getattr(prev, "created_at", None) or 0):
            latest[k3] = info
    return running, latest


def _token_of(plan: Mapping, stage: str) -> str:
    return next(s["token"] for s in plan["stages"] if s["name"] == stage)


def _write_action_record(store: Any, plan: Mapping, a: Action, now: datetime.datetime) -> None:
    """One ownership record per stage of the action (S5.3), written BEFORE its jobs launch so a concurrent
    step never grants the same units. Keyed by pipeline, stage and token under the action id."""
    from scripts._stages import _canon_json

    keys = sorted({k for part in (a.partition or ()) for k in part})
    for stage in a.stages:
        token = _token_of(plan, stage)
        rec = {
            "pipeline": plan["pipeline"],
            "stage": stage,
            "token": token,
            "keys": keys,
            "owns_all": a.owns_all,
            "action_id": a.action_id,
            "kind": a.kind,
            "created_at": now.isoformat(),
        }
        store.upload_bytes(action_rel(plan["pipeline"], stage, token, a.action_id or "noaction"), _canon_json(rec))


def _append_prediction(store: Any, campaign: str, commit: str, a: Action) -> None:
    """Append the launched action to ``_campaign/<campaign>/<commit7>/predictions.jsonl`` (S7.2 bar 2). The
    record is enriched in Task 20; here it carries the action id, stages and predicted walls."""
    rel = f"_campaign/{campaign}/{commit[:7]}/predictions.jsonl"
    rec = {
        "action_id": a.action_id,
        "stages": list(a.stages),
        "stage": a.stages[0] if a.stages else None,
        "kind": a.kind,
        "predicted_job_wall_s": a.predicted_job_wall_s,
        "predicted_wall_s": a.predicted_job_wall_s,  # S7.2 bar 2 reads this per action id
        "predicted_reduce_wall_s": a.predicted_job_wall_s if a.kind == "reduce" else None,
    }
    prev = store.read_bytes(rel) or b""
    store.upload_bytes(rel, prev + (json.dumps(rec, sort_keys=True) + "\n").encode("utf-8"))


def _append_campaign_prediction(store: Any, campaign: str, commit: str, makespan_s: float) -> None:
    """Append the campaign-level makespan prediction (S7.2 bar 2) to ``predictions.jsonl`` once per step."""
    rel = f"_campaign/{campaign}/{commit[:7]}/predictions.jsonl"
    rec = {"kind": "campaign", "makespan_s": makespan_s}
    prev = store.read_bytes(rel) or b""
    store.upload_bytes(rel, prev + (json.dumps(rec, sort_keys=True) + "\n").encode("utf-8"))


def _regrant(plan: Mapping, a: Action, k: int, owned: Mapping, merged: Mapping, cons: Constraints, lpt: bool) -> Action:
    """Re-plan an LPT action within the ``k`` jobs the scheduler granted (S5.5). A round-robin / owns_all
    action (profile, grouped, reduce, download) is all-or-nothing, so it launches as planned."""
    if a.owns_all or a.partition is None:
        return a
    for x in next_actions(plan, owned, merged, cons, lpt=lpt, grant=lambda _a: k):
        if x.stages == a.stages:
            return x
    return a


@dataclasses.dataclass
class _Scheduled:
    inst: Instance
    plan: Mapping
    store: Any
    action: Action
    cp: float
    owned: Mapping
    merged: Mapping


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
    with_replica: int | None = None,
    canary: bool = False,
) -> dict:
    """ONE stateless advance of every instance (S5.5): read ownership from the action records, plan every
    ready stage, then grant the launch budget to the actions in remaining-critical-path order (longest chain
    first). ``only="profile"`` launches profile actions only. ``with_replica=N`` also steps the replica
    instances under the one budget; an attested stage is skipped in the replica (ADR-129). ``canary`` launches
    one canary unit per attested stage over the reused upstream (S5.10)."""
    from scripts.hf_jobs import launch as launcher
    from scripts.hf_jobs.preflight import require_preflight
    from scripts.hf_jobs.store import BucketStore

    require_preflight(c.name, commit, c.sha256, store=BucketStore(c.bucket, "", api=api))
    cache = {} if cache is None else cache
    insts = list(instances(c, commit, replica=replica))
    if with_replica is not None:  # the INV-1 replica runs beside the main run, under the one budget (S5.5)
        insts += list(instances(c, commit, replica=with_replica))
    own_jobs = list(launcher.jobs_for_run_keys(api, [i.run_key for i in insts]))
    error_jobs = list(launcher.jobs_for(api, campaign=c.name, commit=commit))
    _, latest = _running(error_jobs)
    by_stage = _error_history(error_jobs)

    def _is_running(j: Any) -> bool:
        return str(getattr(getattr(j, "status", None), "stage", "") or "") not in launcher.TERMINAL

    running_now = sum(1 for j in own_jobs if _is_running(j))
    # (run_key, variant) with a non-terminal job (CCTP-IMPL-14): an instance still running is NOT settled,
    # even on a step where a sibling stage failed and nothing new launched.
    running_iv = {
        (lab.get("run_key", ""), lab.get("variant", ""))
        for j in own_jobs
        if _is_running(j)
        for lab in [dict(getattr(j, "labels", None) or {})]
    }
    budget = c.concurrency_cap - running_now
    hist = committed_history()
    cons = planning_constraints(c)
    report: dict[str, dict] = {}
    scheduled: list[_Scheduled] = []
    for inst in insts:
        store = BucketStore(c.bucket, inst.run_key, api=api)
        try:
            plan = read_plan(inst, bucket=c.bucket)
        except RuntimeError as exc:
            report[inst.id] = {"action": None, "jobs": [], "failed": str(exc)}
            continue
        if inst.role == "replica":  # ADR-129: an attested stage needs no replica -- mark it reduced (skip)
            main_store = BucketStore(c.bucket, run_key(c, commit, inst.job, inst.tier), api=api)
            plan, _ = plan_for_role(plan, "replica", attested_stages(main_store, plan, commit), commit)
        merged = {**hist, **live_history(plan, store, cache)}
        owned = read_ownership(store, plan, own_jobs)
        acts = next_actions(plan, owned, merged, cons, lpt=lpt_allowed(inst))
        if not acts:
            done = all(s.get("reduced") for s in plan["stages"])
            report[inst.id] = {"action": "done" if done else "waiting", "jobs": [], "failed": None}
            continue
        # EVERY action of the instance is gated INDEPENDENTLY (S5.3, r-impl CCTP-IMPL-01): a halt or an ERROR
        # on one stage never blocks a sibling stage, and the error_resumes bound applies to each stage's own
        # ERROR history. The per-instance report reflects the primary action (`acts[0]`); every non-blocked
        # action is scheduled.
        infra = _make_infra_checker(store, own_jobs)
        cp = remaining_critical_path(plan, merged, cons)
        primary = acts[0]
        report[inst.id] = {
            "action": primary.kind,
            "stages": primary.stages,
            "jobs": [],
            "failed": None,
            "n_jobs": primary.n_jobs if primary.partition is None else len(primary.partition),
        }
        if primary.reason:
            report[inst.id]["reason"] = primary.reason
        for a in acts:
            is_primary = a is primary
            if a.kind == "halt":
                if is_primary:
                    report[inst.id] = halt_report(a)
                continue
            if only is not None and a.kind != only:
                if is_primary:
                    report[inst.id] = {"action": f"skip:{a.kind}", "jobs": [], "failed": None}
                continue
            prev = latest.get((inst.run_key, inst.variant, "-".join(a.stages)))
            prev_status = str(getattr(getattr(prev, "status", None), "stage", "") or "")
            if prev is not None and prev_status == "ERROR":
                why = resume_after_error(
                    plan,
                    a.stages[0],
                    by_stage.get((inst.run_key, inst.variant, "-".join(a.stages)), []),
                    cons,
                    infra=infra,
                    operator_after=_newest_operator(store, plan["pipeline"], a.stages[0], _token_of(plan, a.stages[0])),
                )
                if why is not None:
                    if is_primary:
                        msg = getattr(getattr(prev, "status", None), "message", None)
                        report[inst.id] = {
                            "action": a.kind,
                            "stages": a.stages,
                            "jobs": [],
                            "failed": why or f"job {prev.id} ERROR: {msg}",
                        }
                    continue
            scheduled.append(_Scheduled(inst, plan, store, a, cp, owned, merged))
    if dry_run:
        for s in scheduled:
            report[s.inst.id].setdefault("planned", dataclasses.asdict(s.action))
        return report
    index = {(s.inst.id, i): s for i, s in enumerate(scheduled)}
    ordered = grant_order([(f"{s.inst.id}#{i}", s.cp, s.action) for i, s in enumerate(scheduled)])
    now = datetime.datetime.now(datetime.timezone.utc)
    for tag, a in ordered:
        if budget <= 0:
            break
        inst_id, idx = tag.rsplit("#", 1)
        s = index[(inst_id, int(idx))]
        granted = allocate([(inst_id, a)], budget=budget)
        if not granted:
            continue
        _, _, k = granted[0]
        a_k = _regrant(s.plan, a, k, s.owned, s.merged, cons, lpt_allowed(s.inst))
        aid = make_action_id(s.inst.id, a_k.stages, [_token_of(s.plan, x) for x in a_k.stages], a_k.partition, now)
        a_k = dataclasses.replace(a_k, action_id=aid)
        _write_action_record(s.store, s.plan, a_k, now)
        specs = _job_specs(c, commit, s.inst, s.plan, a_k, store=s.store)
        for spec in specs:
            if budget <= 0:
                break
            report[s.inst.id]["jobs"].append(launcher.launch(spec, api=api, token=token))
            budget -= 1
        _append_prediction(s.store, c.name, commit, a_k)
    if scheduled:  # S7.2 bar 2: the campaign makespan prediction (max remaining critical path over instances)
        _append_campaign_prediction(scheduled[0].store, c.name, commit, max(s.cp for s in scheduled))
    if canary:  # ADR-129 S5.10: one canary unit per attested stage over the reused upstream (read-through)
        budget = _schedule_canary(c, commit, api=api, token=token, budget=budget, report=report, now=now)
    for inst in insts:  # CCTP-IMPL-14: --watch must not treat a still-running instance as settled
        if inst.id in report:
            report[inst.id]["running"] = (inst.run_key, inst.variant) in running_iv
    return report


# ---- commands -------------------------------------------------------------------------------------


def _head() -> str:
    argv = ["git", "rev-parse", "HEAD"]  # git from PATH is the house pattern
    return subprocess.run(argv, capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout.strip()  # noqa: S603


def cmd_preflight(c: Campaign, commit: str, *, api: Any) -> dict:
    """All seven no-spend checks (ADR-129 adds check_image); uploads the report the launch gate requires (spec 5.4)."""
    from scripts.hf_jobs import preflight
    from scripts.hf_jobs.store import BucketStore

    insts = instances(c, commit)
    checks = [
        preflight.check_ref(commit),
        preflight.check_lock(),
        preflight.check_api(api),
        preflight.check_datasets([n for j in c.jobs for n in (j.dataset, j.probe_dataset) if n], api=api),
        preflight.check_plans(insts),
        preflight.check_concurrency(c.concurrency_cap, SUMMARY_DIR.glob("*.json"), api),
        preflight.check_image(),
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
        for a in next_actions(plan, {}, hist, cons, lpt=lpt_allowed(inst)):
            if a.predicted_job_wall_s and a.predicted_job_wall_s > target:
                wall_min, target_min = a.predicted_job_wall_s / 60, target / 60
                flags.append(f"{inst.id}/{'-'.join(a.stages)} predicted {wall_min:.0f} min > {target_min:.0f}")
    checks.append(preflight.Check("estimate", True, f"{len(flags)} stage(s) predicted above the target", tuple(flags)))
    report = preflight.assemble_report(c.name, commit, c.sha256, checks, flags=flags)
    BucketStore(c.bucket, "", api=api).upload_bytes(
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
        acts = next_actions(plan, {}, hist, cons, lpt=lpt_allowed(inst))
        rows.append(
            {"instance": inst.id, "module": inst.module, "action": dataclasses.asdict(acts[0]) if acts else None}
        )
    return rows


def cmd_status(c: Campaign, commit: str, *, api: Any, replica: int | None = None) -> list[dict]:
    from scripts.hf_jobs import launch as launcher
    from scripts.hf_jobs.store import BucketStore

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
                "store_bytes": BucketStore(c.bucket, inst.run_key, api=api).size_bytes(),
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


def cmd_cancel(c: Campaign, commit: str, *, api: Any) -> list[str]:
    from scripts.hf_jobs import launch as launcher

    ids = []
    for info in launcher.jobs_for(api, campaign=c.name, commit=commit):
        if str(getattr(getattr(info, "status", None), "stage", "")) not in launcher.TERMINAL:
            print(f"cancel {info.id} {dict(info.labels or {})}", file=sys.stderr)
            api.cancel_job(job_id=info.id)
            ids.append(info.id)
    return ids


def stage_summary(
    instance: str,
    st: Mapping,
    recs: Sequence[Mapping],
    st_jobs: Sequence[Mapping],
    prof: bytes | None,
    cons: Constraints,
    *,
    job_recs: Sequence[Mapping],
) -> dict:
    """One stage's row of the committed summary (plan Task 21 step 8): unit cost, efficiency, the measured
    workers and job startup, job walls, attempts, resumed units and the irreducible wall floor.

    ``recs`` are this stage's unit records; ``job_recs`` are every stage's records of the instance. A job's
    startup is measured over ``job_recs`` (a grouped job's first unit may be any of its stages) and is
    recorded on the job's FIRST stage only: in a grouped job a later stage starts after the earlier stages'
    walls, which are not startup (review r3, T21-CORR-01 / CCHC-IMPL-01)."""
    from scripts.hf_jobs.launch import parse_stage_label
    from scripts.hf_jobs.ledger import job_startups, percentile, stage_workers, unit_cost

    cost = unit_cost(recs)
    walls = [j["wall_s"] for j in st_jobs if j.get("wall_s") is not None]
    leads = [j for j in st_jobs if parse_stage_label(j["labels"].get("stage", ""))[0][:1] == (st["name"],)]
    startups = job_startups(leads, job_recs)
    workers, inner = stage_workers(recs)
    return {
        "instance": instance,
        "stage": st["name"],
        "history_key": st["history_key"],
        "n_units": st["n_units"],
        "n_jobs": len(st_jobs),
        "unit_cost": dataclasses.asdict(cost) if cost else None,
        "efficiency": json.loads(prof).get("efficiency") if prof else None,
        "workers": workers,
        "inner_workers": inner,
        "startup_s": percentile(startups, 0.5) if startups else None,
        "max_job_wall_s": max(walls) if walls else None,
        "p50_job_wall_s": percentile(walls, 0.5) if walls else None,
        "attempts_max": max((int(r.get("attempt", 1)) for r in recs), default=0),
        "resumed_units": sum(1 for r in recs if int(r.get("attempt", 1)) > 1),
        "irreducible_floor_s": irreducible_floor_s(cost, cons) if cost else None,
        # S5.7 / E-5: the measured wall of every done unit, so the next campaign's LPT uses it (no profile).
        "per_unit": {
            r["key"]: float(r["wall_s"])
            for r in recs
            if r.get("outcome") == "done" and r.get("wall_s") is not None and r.get("key")
        },
    }


def cmd_summary(c: Campaign, commit: str, *, api: Any, replica: int | None = None) -> pathlib.Path:
    """Write ``docs/research/hf_campaigns/<name>.json`` (a registered artifact: clean tree only, ADR-037)."""
    from scripts._provenance import env_provenance, git_provenance, require_clean_tree
    from scripts.hf_jobs import launch as launcher
    from scripts.hf_jobs.identity import artifact_identity
    from scripts.hf_jobs.ledger import job_record, summarize, unit_records, write_summary
    from scripts.hf_jobs.store import BucketStore

    require_clean_tree(git_provenance(), allow_dirty=False)
    jobs = [job_record(j) for j in launcher.jobs_for(api, campaign=c.name, commit=commit)]
    stages, artifacts = [], []
    staging = REPO_ROOT.parent / "_hf_staging" / c.name
    cons = planning_constraints(c)
    bar_stages: list[dict] = []
    reduce_records: list[dict] = []
    predictions: list[dict] = []
    job_phases: list[dict] = []
    from scripts.hf_jobs import adopt as _adopt

    comparisons = _adopt.read_comparisons(c, commit, api=api)  # S7.2 bar 6: THIS campaign's byte-compares (B#35)
    for inst in instances(c, commit, replica=replica):
        store = BucketStore(c.bucket, inst.run_key, api=api)
        plan = read_plan(inst, bucket=c.bucket)
        by_stage = {st["name"]: unit_records(store, st["generation"]) for st in plan["stages"]}
        job_recs = [r for recs in by_stage.values() for r in recs]
        from scripts.hf_jobs import ledger as _ledger
        from scripts.hf_jobs.launch import parse_stage_label

        for st in plan["stages"]:
            st_jobs = [
                j
                for j in jobs
                if j["labels"].get("run_key") == inst.run_key
                and st["name"] in parse_stage_label(j["labels"].get("stage", ""))[0]
            ]
            prof = store.read_bytes(f"{st['generation']}/_profile.json")
            row = stage_summary(inst.id, st, by_stage[st["name"]], st_jobs, prof, cons, job_recs=job_recs)
            reds = _ledger.reduce_records(store, plan["pipeline"], st["name"], st["token"])
            per_result = [r["wall_s"] / r["n_results"] for r in reds if r.get("n_results") and r.get("wall_s")]
            row["reduce_per_result_s"] = _ledger.percentile(per_result, 0.5) if per_result else None
            stages.append(row)
            bar_stages.append({**row, "needs": st.get("needs") or [], "inner_parallel": st.get("inner_parallel")})
            for r in reds:
                pr = plan_reduce(int(r.get("n_results") or 0), cons, per_result_s=row["reduce_per_result_s"])
                reduce_records.append({**r, "stage": st["name"], "token": st["token"], "predicted_wall_s": pr.wall_s})
        job_phases += list(_job_phase_records(store).values())
        raw = store.read_bytes(f"_campaign/{c.name}/{commit[:7]}/predictions.jsonl")
        if raw:
            predictions += [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        out_dir = staging / inst.run_key / inst.variant
        if out_dir.is_dir():
            artifacts.append({"instance": inst.id, **artifact_identity(out_dir)})
    prov = {"commit": commit, "dirty": False, "env": env_provenance()}
    path = SUMMARY_DIR / f"{c.name}.json"
    # The profile jobs (role main, but exploration not production) are excluded from the cost bar; reused units
    # ran no job so carry no wall (CCTP-IMPL-35).
    profile_actions = {p.get("action_id") for p in predictions if p.get("kind") == "profile"}
    cost_exclude_ids = {j["id"] for j in jobs if j["labels"].get("action") in profile_actions}
    bars = compute_bars(
        jobs=jobs,
        predictions=predictions,
        stages=bar_stages,
        reduce_records=reduce_records,
        job_phases=job_phases,
        cycle1_main_job_hours=_cycle1_main_job_hours(),
        cap=c.concurrency_cap,
        declared_startup_s=_declared_startup_s(cons),
        comparisons=comparisons,
        cost_exclude_ids=cost_exclude_ids,
    )
    summary = summarize(
        campaign=c.name, campaign_sha256=c.sha256, prov=prov, stages=stages, jobs=jobs, artifacts=artifacts, bars=bars
    )
    from scripts.hf_jobs.preflight import peak_concurrency

    summary["max_concurrent"] = peak_concurrency(jobs)  # S5.5 check_concurrency reads this next cycle
    write_summary(path, summary)
    return path


def cmd_history_export(c: Campaign, commit: str, *, api: Any) -> list[pathlib.Path]:
    """E-5: the ONE slow read-only pass over the Cycle 1 stores. Writes the stage cost model
    (``<name>.json``) and the per-unit costs (``<name>.per_unit.json``) under ``docs/research/hf_campaigns/``
    so the next campaign plans without a profile job (S5.7). Refuses a dirty tree (ADR-037)."""
    from scripts._provenance import git_provenance, require_clean_tree
    from scripts.hf_jobs import ledger as _ledger
    from scripts.hf_jobs.store import BucketStore

    require_clean_tree(git_provenance(), allow_dirty=False)
    stages: list[dict] = []
    per_unit: dict[str, dict[str, float]] = {}
    for inst in instances(c, commit):
        store = BucketStore(c.bucket, inst.run_key, api=api)
        plan = read_plan(inst, bucket=c.bucket)
        for st in plan["stages"]:
            hk = st.get("history_key") or st["name"]
            recs = _ledger.unit_records(store, st["generation"])
            cost = _ledger.unit_cost(recs)
            prof = store.read_bytes(f"{st['generation']}/_profile.json")
            reds = _ledger.reduce_records(store, plan["pipeline"], st["name"], st["token"])
            per_result = [r["wall_s"] / r["n_results"] for r in reds if r.get("n_results") and r.get("wall_s")]
            if cost:
                stages.append(
                    {
                        "history_key": hk,
                        "instance": inst.id,
                        "stage": st["name"],
                        "unit_cost": dataclasses.asdict(cost),
                        "efficiency": json.loads(prof).get("efficiency") if prof else None,
                        "reduce_per_result_s": _ledger.percentile(per_result, 0.5) if per_result else None,
                    }
                )
            pu = {
                r["key"]: float(r["wall_s"])
                for r in recs
                if r.get("outcome") == "done" and r.get("wall_s") is not None and r.get("key")
            }
            if pu:
                per_unit.setdefault(hk, {}).update(pu)
    base = {"run_commit": commit, "run_tree_dirty": False}
    cost_path = SUMMARY_DIR / f"{c.name}.json"
    unit_path = SUMMARY_DIR / f"{c.name}.per_unit.json"
    cost_path.write_text(json.dumps({**base, "stages": stages}, sort_keys=True, indent=2), encoding="utf-8")
    unit_path.write_text(json.dumps({**base, "per_unit": per_unit}, sort_keys=True, indent=2), encoding="utf-8")
    return [cost_path, unit_path]


def cmd_clear_halt(c: Campaign, commit: str, target: str, reason: str, *, api: Any) -> dict:
    """``resume --clear-halt <instance>/<stage> --reason "<text>"`` (S5.8): refuse unless the stage is
    halted, then write the operator record so the next ``step`` resumes. Launches nothing."""
    import subprocess as _sp

    from scripts.hf_jobs import launch as launcher
    from scripts.hf_jobs.store import BucketStore

    inst_id, _, stage = target.rpartition("/")  # instance id is run_key/variant, so split on the LAST '/'
    insts = {i.id: i for i in instances(c, commit)}
    if inst_id not in insts or not stage:
        raise SystemExit(f"refuse --clear-halt: expected <instance>/<stage>, got {target!r}")
    inst = insts[inst_id]
    store = BucketStore(c.bucket, inst.run_key, api=api)
    plan = read_plan(inst, bucket=c.bucket)
    token = _token_of(plan, stage)
    cons = planning_constraints(c)
    by_stage = _error_history([j for j in launcher.jobs_for(api, campaign=c.name, commit=commit)])
    hist = by_stage.get((inst.run_key, inst.variant, stage), [])
    st_plan = next((s for s in plan["stages"] if s["name"] == stage), None)
    if st_plan is None:
        raise SystemExit(f"refuse --clear-halt: {stage!r} is not a stage of {inst_id}")
    # A UNIT-cause halt (failed units over max_failures) is a deterministic failure (S5.8, CCTP-IMPL-13):
    # the operator record only resets the ERROR-resume count, so clearing it would NOT resume -- a false
    # success. Refuse it with a distinct reason. --clear-halt clears ONLY the error-resume band.
    spent = sum(1 for v in (st_plan.get("failed_units") or {}).values() if int(v) >= cons.unit_attempts)
    tol = st_plan.get("max_failures")
    if tol is not None and spent > int(tol):
        raise SystemExit(
            f"refuse --clear-halt: {stage!r} halted on {spent} unit failure(s) over max_failures={tol}; "
            f"--clear-halt only clears the error-resume bound, not deterministic unit failures"
        )
    clear_halt_guard(halted=_trailing_errors(hist) >= cons.error_resumes)
    git = ["git", "config", "user.email"]  # git from PATH is the house pattern
    who = _sp.run(git, capture_output=True, text=True, check=False).stdout.strip()  # noqa: S603
    utc = datetime.datetime.now(datetime.timezone.utc)
    rel = f"_operator/{plan['pipeline']}/{stage}/{token}/{utc.strftime('%Y%m%dT%H%M%SZ')}.json"
    store.upload_bytes(
        rel, json.dumps({"who": who or "unknown", "when": utc.isoformat(), "reason": reason}).encode("utf-8")
    )
    return {"cleared": target, "record": rel}


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
            "history-export",
        ],
    )
    ap.add_argument("--replica", type=int, default=None, help="fresh replica store suffix -r<N> (INV-1 gate, P18)")
    ap.add_argument(
        "--with-replica",
        type=int,
        default=None,
        help="launch/resume: step the replica instances -r<N> beside the main run, under the one budget (S5.5)",
    )
    ap.add_argument(
        "--canary",
        action="store_true",
        help="launch/resume: also step the canary instances; verify: compare the canary units to main (S5.10)",
    )
    ap.add_argument("--clear-halt", default=None, help="resume: <instance>/<stage> to clear a halted stage (S5.8)")
    ap.add_argument("--reason", default=None, help="resume --clear-halt: the reason recorded with the operator record")
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
    if a.command == "resume" and a.clear_halt:
        print(json.dumps(cmd_clear_halt(c, commit, a.clear_halt, a.reason or "", api=api), indent=2))
        return 0
    if a.command in ("profile", "launch", "resume"):
        cache: dict = {}
        only = "profile" if a.command == "profile" else None
        while True:
            rep = step(
                c,
                commit,
                api=api,
                token=token,
                replica=a.replica,
                dry_run=a.dry_run,
                only=only,
                cache=cache,
                with_replica=a.with_replica,
                canary=a.canary,
            )
            print(json.dumps(rep, indent=2, default=str), flush=True)
            # An instance that LAUNCHED jobs this step is not settled, even if its primary action reported a
            # failure on a sibling stage (CCTP-IMPL-14): those jobs' next stage still needs a later step.
            settled = all(
                (r.get("action") == "done" or r.get("failed")) and not r.get("jobs") and not r.get("running")
                for r in rep.values()
            )
            if not a.watch or a.dry_run or settled:
                return 0
            time.sleep(60)
    if a.command == "status":
        print(json.dumps(cmd_status(c, commit, api=api, replica=a.replica), indent=2))
        return 0
    if a.command == "cancel":
        print(json.dumps(cmd_cancel(c, commit, api=api)))
        return 0
    if a.command == "history-export":
        for p in cmd_history_export(c, commit, api=api):
            print(p)
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
        print(
            json.dumps(
                adopt.verify(
                    c,
                    commit,
                    staging=a.staging,
                    replica=a.replica,
                    api=api,
                    with_replica=a.with_replica,
                    canary=a.canary,
                ),
                indent=2,
                default=str,
            )
        )
    elif a.command == "adopt":
        print(json.dumps(adopt.adopt(c, commit, staging=a.staging, api=api), indent=2, default=str))
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
