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
import dataclasses
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
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
    concurrency_cap: int
    attempts: int
    jobs: tuple[CampaignJob, ...]
    path: pathlib.Path | None
    sha256: str

    def constraints(self, *, startup_s: float | None = None) -> Constraints:
        base = Constraints(target_s=float(self.target_minutes) * 60.0, concurrency_cap=self.concurrency_cap)
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
    peak_rss_bytes: int | None = None
    partition: tuple[tuple[str, ...], ...] | None = None
    profile: int | None = None
    predicted_job_wall_s: float | None = None
    predicted_cost_usd: float | None = None
    reason: str | None = None
    #: The memory escalation: the job also runs units deferred by a memory failure (`--escalate`).
    escalate: bool = False


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


def next_actions(
    plan: Mapping, running: set[str], history: Mapping[str, Mapping], c: Constraints, *, lpt: bool = True
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
    target = int(2 * c.target_s)
    for i, st in enumerate(stages):
        if st.get("reduced"):
            continue
        if not st.get("ready") or st.get("n_units") is None or st["name"] in running:
            return []
        inner_w = int(st.get("inner_width") or 1)
        failed = dict(st.get("failed_units") or {})
        retry = {k for k, n in failed.items() if int(n) < c.unit_attempts}
        memory = retry & set(st.get("memory_units") or ())
        todo = sorted((set(st.get("pending_units") or ()) | retry) - memory)
        if not todo and memory:
            # a unit over its fair-share cap: alone in its job, with the whole box (W=1, cap = usable RAM)
            alone = tuple((k,) for k in sorted(memory))
            return [
                Action(
                    "run",
                    (st["name"],),
                    len(alone) if lpt else 1,
                    1,
                    1,
                    target,
                    cap_bytes=c.usable_bytes,
                    partition=alone if lpt else None,
                    escalate=True,
                )
            ]
        if not todo:
            spent = {k: n for k, n in sorted(failed.items()) if k not in retry}
            if not spent:
                return [Action("reduce", (st["name"],), 1, 1, 1, target)]
            tolerance = st.get("max_failures")
            if tolerance is None or len(spent) <= int(tolerance):
                why = f"{st['name']}: reduces without unit(s) failed {c.unit_attempts} time(s): {spent}"
                return [Action("reduce", (st["name"],), 1, 1, 1, target, reason=why)]
            why = (
                f"{st['name']}: {len(spent)} unit(s) failed {c.unit_attempts} time(s), over the stage's "
                f"max_failures={tolerance}: {spent}"
            )
            return [Action("halt", (st["name"],), 0, 0, 0, 0, reason=why)]
        if st["n_units"] <= 2:
            inner = min(c.vcpu, inner_w) if st.get("inner_parallel") else 1
            return [Action("run", _group(stages, i, reduced), 1, 1, inner, target)]
        hk = st.get("history_key") or st["name"]
        if hk not in history:
            inner = min(c.vcpu, inner_w) if st.get("inner_parallel") else 1
            workers = 1 if st.get("inner_parallel") else c.vcpu
            sample = min(int(st["n_units"]), 2 * c.vcpu)
            return [Action("profile", (st["name"],), 1, workers, inner, target, profile=sample)]
        h = history[hk]
        keys = todo
        measured = h.get("per_unit") if lpt else None
        per_unit = {k: float(measured.get(k, h["cost"].p50_s)) for k in keys} if measured else None
        p = plan_stage(
            st["name"],
            keys,
            h["cost"],
            inner_parallel=bool(st.get("inner_parallel")),
            n_inner=inner_w,
            efficiency=h.get("efficiency"),
            per_unit=per_unit,
            c=c,
        )
        return [
            Action(
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
            )
        ]
    return []


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
            out[hk] = {"cost": UnitCost(**cost), "efficiency": st.get("efficiency"), "per_unit": None}
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
            cost = unit_cost(recs)
            per_unit = {r["key"]: float(r["wall_s"]) for r in recs if r.get("outcome") == "done"}
            cache[gen] = None if cost is None else {"cost": cost, "efficiency": eff, "per_unit": per_unit}
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


def _trailing_errors(history: Sequence[Any]) -> int:
    """How many of the stage's newest jobs ended in ERROR in a row (a COMPLETED job ends the run)."""
    n = 0
    for info in history:
        if str(getattr(getattr(info, "status", None), "stage", "")) != "ERROR":
            break
        n += 1
    return n


def _timed_out(info: Any) -> bool:
    return "timeout" in str(getattr(getattr(info, "status", None), "message", "") or "").lower()


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
            peak_rss_bytes=a.peak_rss_bytes,
            timeout_s=a.timeout_s,
            attempts=c.attempts,
            datasets=inst.datasets,
            extras=(),
            profile=a.profile,
            shard_keys_path=shard_keys_path,
            escalate=a.escalate,
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
    cache = {} if cache is None else cache
    all_jobs = list(launcher.jobs_for(api, campaign=c.name, commit=commit))
    running, latest = _running(all_jobs)
    by_stage = _error_history(all_jobs)
    budget = c.concurrency_cap - sum(1 for v in running.values() for _ in v)
    hist = committed_history()
    cons = planning_constraints(c)
    report: dict[str, dict] = {}
    for inst in instances(c, commit, replica=replica):
        store = BucketStore(c.bucket, inst.run_key, api=api)
        try:
            plan = read_plan(inst, bucket=c.bucket)
        except RuntimeError as exc:
            report[inst.id] = {"action": None, "jobs": [], "failed": str(exc)}
            continue
        acts = next_actions(
            plan,
            running.get((inst.run_key, inst.variant), set()),
            {**hist, **live_history(plan, store, cache)},
            cons,
            lpt=lpt_allowed(inst),
        )
        if not acts:
            done = all(s.get("reduced") for s in plan["stages"])
            report[inst.id] = {"action": "done" if done else "waiting", "jobs": [], "failed": None}
            continue
        a = acts[0]
        if a.kind == "halt":
            report[inst.id] = halt_report(a)
            continue
        if only is not None and a.kind != only:
            report[inst.id] = {"action": f"skip:{a.kind}", "jobs": [], "failed": None}
            continue
        prev = latest.get((inst.run_key, inst.variant, "-".join(a.stages)))
        prev_status = str(getattr(getattr(prev, "status", None), "stage", "") or "")
        if prev is not None and prev_status == "ERROR":
            why = resume_after_error(
                plan, a.stages[0], by_stage.get((inst.run_key, inst.variant, "-".join(a.stages)), []), cons
            )
            if why is not None:
                msg = getattr(getattr(prev, "status", None), "message", None)
                report[inst.id] = {
                    "action": a.kind,
                    "stages": a.stages,
                    "jobs": [],
                    "failed": why or f"job {prev.id} ERROR: {msg}",
                }
                continue
        specs = _job_specs(c, commit, inst, plan, a, store=store) if not dry_run else []
        n_planned = a.n_jobs if a.partition is None else len(a.partition)
        ids: list[str] = []
        if dry_run:
            report[inst.id] = {
                "action": a.kind,
                "stages": a.stages,
                "jobs": [],
                "planned": dataclasses.asdict(a),
                "n_jobs": n_planned,
            }
            continue
        for spec in specs:
            if budget <= 0:
                break
            ids.append(launcher.launch(spec, api=api, token=token))
            budget -= 1
        report[inst.id] = {"action": a.kind, "stages": a.stages, "jobs": ids, "failed": None, "n_jobs": n_planned}
        if a.reason:  # a tolerant reduce names the units it drops
            report[inst.id]["reason"] = a.reason
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
    checks = [
        preflight.check_ref(commit),
        preflight.check_lock(),
        preflight.check_api(api),
        preflight.check_datasets([n for j in c.jobs for n in (j.dataset, j.probe_dataset) if n], api=api),
        preflight.check_plans(insts),
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
        acts = next_actions(plan, set(), hist, cons, lpt=lpt_allowed(inst))
        rows.append(
            {"instance": inst.id, "module": inst.module, "action": dataclasses.asdict(acts[0]) if acts else None}
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
    from scripts.hf_jobs.ledger import job_startups, percentile, stage_workers, unit_cost

    cost = unit_cost(recs)
    walls = [j["wall_s"] for j in st_jobs if j.get("wall_s") is not None]
    leads = [j for j in st_jobs if j["labels"].get("stage", "").split("-")[0] == st["name"]]
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
            stages.append(stage_summary(inst.id, st, by_stage[st["name"]], st_jobs, prof, cons, job_recs=job_recs))
        out_dir = staging / inst.run_key / inst.variant
        if out_dir.is_dir():
            artifacts.append({"instance": inst.id, **artifact_identity(out_dir)})
    prov = {"commit": commit, "dirty": False, "env": env_provenance()}
    path = SUMMARY_DIR / f"{c.name}.json"
    write_summary(
        path,
        summarize(campaign=c.name, campaign_sha256=c.sha256, prov=prov, stages=stages, jobs=jobs, artifacts=artifacts),
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
            settled = all(r.get("action") == "done" or r.get("failed") for r in rep.values())
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
