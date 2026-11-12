"""The minimal HF Jobs campaign driver (ADR-128).

Stateless: every step rebuilds state from two sources only -- each instance's
``--plan`` JSON read against the bucket store (done / pending units per stage)
and the HF job labels (``list_jobs``: what runs, what ended ERROR). No controller
state of its own; ``resume`` continues from the store.

``decide`` is the pure decision (tested without HF): walk each instance's stages
in DAG order, launch the missing units of the first ready un-reduced stage, then
trigger its reduce; stop the whole run on a FATAL job (``failures.classify``).
``step`` applies the decision through the typed launcher.

The reduce is INLINE (the ``_stages`` executor reduces a stage once every unit is
terminal). So the driver never launches a reduce-only job: it launches a run over
an EMPTY key set (``--shard-keys`` of ``[]``), which runs nothing and lets the
executor reduce the stage. One job shape -- main's ``launch.JobSpec`` -- so the
engine (``launch`` / ``executor`` / ``_stages``) is untouched.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import NamedTuple

from scripts._stages import exceeds_tolerance
from scripts.hf_jobs.failures import JobOutcome, classify
from scripts.hf_jobs.hub_client import request_counts
from scripts.hf_jobs.launch import labels_for, shard_text
from scripts.hf_jobs.launched import UNLISTED_STEPS, LaunchedJobs
from scripts.hf_jobs.preflight import PLAN_WORKERS

#: A job's HF limit unless its stage declares ``timeout_s`` (HFU spec 15.7).
DEFAULT_TIMEOUT_S = 7200
#: The cpu-upgrade box: its cgroup `memory.max`, 32,000,000,000 bytes (measured 2026-11-11 in job
#: 6af4d4e79737d9a9e093649c; HFU launch top-up spec §4), with 2 GiB headroom (the cap an escalate job requests).
_BOX_RAM = 32_000_000_000
_HEADROOM = 2 * 1024**3


@dataclass(frozen=True)
class Launch:
    """One job to start: run units (or trigger the reduce) of one stage of one instance."""

    instance_id: str
    stages: tuple[str, ...]
    shard: tuple[int, int]  # (index, count); (0, 1) is the whole stage
    kind: str  # "run" | "reduce" ("reduce" is a run over an empty key set: runs nothing, lets the executor reduce)
    workers: int
    #: The units the job runs. Keyed: its explicit ``--shard-keys``. Unkeyed: the round-robin ``--shard I/N``
    #: slice the executor will run (recorded in the per-job manifest). () for a reduce trigger.
    keys: tuple[str, ...] = ()
    #: False for an instance with a ``download`` dataset: it is fetched by ``--shard``, so the job runs the
    #: round-robin slice instead of explicit keys (as main's controller did).
    keyed: bool = True
    #: A memory escalation job: deferred units alone on the whole box (``--escalate``, W=1).
    escalate: bool = False
    #: The tag of a keyed launch (`launch.keys_tag` of its keys), set here for every keyed launch, a reduce
    #: trigger included, whoever builds it; empty for an unkeyed one (HFU launch top-up spec §3.4).
    tag: str = ""

    def __post_init__(self) -> None:
        if self.keyed and not self.tag:
            from scripts.hf_jobs.launch import keys_tag

            object.__setattr__(self, "tag", keys_tag(self.keys))


@dataclass(frozen=True)
class Decision:
    fatal: tuple[str, str] | None = None  # (reason, detail): stop the whole run, never retry
    launches: tuple[Launch, ...] = ()
    done: bool = False  # every instance's every stage is reduced
    reports: tuple[tuple[str, str], ...] = ()  # (instance_id, reason): a stuck stage, reported not retried


@dataclass(frozen=True)
class StepResult:
    fatal: tuple[str, str] | None
    launched: int
    done: bool
    reports: tuple[tuple[str, str], ...]
    in_flight: int = 0  # jobs RUNNING/SCHEDULING this step (the loop waits while > 0)
    contention: bool = False  # a 429 this step: `run` backs off
    mount_faults: int = 0  # mount-fault jobs seen; a step-over-step rise backs off


def step(
    campaign,
    commit: str,
    *,
    api,
    token: str,
    replica: int | None = None,
    instances: Sequence | None = None,
    read_plan: Callable | None = None,
    live_records: Callable[[], Sequence[Mapping]] | None = None,
    launch_fn: Callable | None = None,
    baseline_error_ids: frozenset[str] = frozenset(),
    gate: Callable[[], object] | None = None,
    reported: set[str] | None = None,
    fetch_logs: Callable[[str], Iterable[str]] | None = None,
    plan_cache: dict[str, Mapping] | None = None,
    launched: LaunchedJobs | None = None,
) -> StepResult:
    """One stateless step: the launch gate (a passing preflight report for this commit and campaign-file hash),
    then rebuild state, decide and launch. A new FATAL stops before any launch; an instance whose ``--plan``
    fails is reported and skipped. The callables are test seams; their defaults do the real HF I/O."""
    do_launch = _resolve_launch(launch_fn)
    launched = launched if launched is not None else LaunchedJobs()  # `run` owns one for the run (spec §3.3)
    if plan_cache is not None:
        read_plan = _cached_reader(_resolve_read_plan(campaign, read_plan), plan_cache)
    t0 = time.monotonic()
    sent0 = request_counts()
    try:
        (gate or _default_gate(campaign, commit, api))()
        state = _gather(
            campaign,
            commit,
            api,
            replica,
            instances,
            read_plan,
            live_records,
            baseline_error_ids,
            fetch_logs,
            launched=launched,
        )
    except Exception as exc:  # a 429 reading the gate, the plans or `list_jobs`: back off, never crash
        if _is_contention(exc):
            total, _ = _hub_requests(sent0)
            _say(f"[driver] step: rate-limited after {time.monotonic() - t0:.1f}s (hub requests {total}); backing off")
            return StepResult(fatal=None, launched=0, done=False, reports=(), in_flight=0, contention=True)
        raise
    state_s = time.monotonic() - t0
    _say_ended_jobs(state.live_jobs, baseline_error_ids, reported if reported is not None else set())
    in_flight = sum(1 for job in state.live_jobs if classify(job).outcome is JobOutcome.IN_FLIGHT)
    foreign = _foreign_identity(state.plans, commit)
    if foreign is not None:  # spec §19.2: the jobs clone `commit`; a plan under another identity reads other tokens
        print(f"FATAL code_identity: {foreign}", file=sys.stderr)
        return StepResult(fatal=("code_identity", foreign), launched=0, done=False, reports=(), in_flight=in_flight)
    decision = _decide(campaign, state, baseline_error_ids)
    reports = (*state.plan_reports, *decision.reports)
    if decision.fatal is not None:
        reason, detail = decision.fatal
        print(f"FATAL {reason}: {detail}", file=sys.stderr)
        return StepResult(fatal=decision.fatal, launched=0, done=False, reports=reports, in_flight=in_flight)

    by_id = {inst.id: inst for inst in state.insts}
    readable = {k: v for k, v in state.plans.items() if v is not None}  # `decide` launches only for a readable plan
    # At most concurrency_cap jobs in flight, and at most launch_batch launches per step.
    budget = max(0, min(campaign.concurrency_cap - in_flight, campaign.launch_batch))
    n_launched, contention = 0, state.cache_unsaved  # an unsaved coverage cache is a 429: back off after launching
    for launch in decision.launches[:budget]:
        inst = by_id[launch.instance_id]
        store = state.stores[inst.id]
        spec = _job_spec(campaign, commit, inst, launch, readable[inst.id], store=store)
        if spec is None:  # its shard keys could not be written (429): nothing runs without them
            contention = True
            break
        try:
            job_id = do_launch(spec, api=api, token=token)
        except Exception as exc:  # a 429 on launch: stop this step and back off
            if _is_contention(exc):
                contention = True
                break
            raise
        # Before the manifest write: the run knows the job's units also when that write gets a 429 (spec §3.3).
        launched.add(str(job_id), labels_for(spec), launch.keys, datetime.now(timezone.utc).isoformat())
        n_launched += 1
        shard = f"{launch.shard[0]}of{launch.shard[1]}" + (f"-{launch.tag}" if launch.tag else "")
        _say(f"[driver] launch {inst.id}/{launch.stages[0]} {shard} units={len(launch.keys)}")
        # The units THIS job holds, keyed by its id; an empty list marks a reduce trigger.
        rec = next(s for s in readable[inst.id]["stages"] if s["name"] == launch.stages[0])
        rel = by_job_rel(readable[inst.id], launch.stages[0], rec["token"], str(job_id))
        if not _guarded_put(store, rel, json.dumps(sorted(launch.keys)).encode("utf-8")):
            _say(f"[driver] job {job_id} manifest not recorded (rate-limited)")  # never charged; looked up each step
            contention = True
            break
    for instance_id, reason in reports:
        print(stuck_line(instance_id, reason), file=sys.stderr)
    n = len(state.insts)
    total, by_plans = _hub_requests(sent0)
    _say(
        f"[driver] step: plans {n} in {state_s:.1f}s ({len(state.plan_reports)} failed); "
        f"hub requests {total} (plans {by_plans}); in flight {in_flight}; "
        f"launched {n_launched}; stuck {len(reports)}; instances done {_instances_done(state.plans)}/{n}"
    )
    return StepResult(
        fatal=None,
        launched=n_launched,
        done=decision.done,
        reports=reports,
        in_flight=in_flight,
        contention=contention,
        mount_faults=sum(state.mount_faults.values()),  # a step-over-step rise signals mount-sidecar contention
    )


def _hub_requests(sent0: tuple[int, int]) -> tuple[int, int]:
    """``(total, plans)``: the Hub requests since the counts ``sent0`` (`hub_client.request_counts`). The total is
    the controller's own requests plus those of its ``--plan`` processes; ``plans`` is the second part. The
    step line prints both, so each launch shows its margin to the HF rate limit (HFU fit-units spec 17). A plan
    that failed is in the count; a plan that timed out was killed and is not, so the total is then a lower bound."""
    own, plans = request_counts()
    return (own - sent0[0]) + (plans - sent0[1]), plans - sent0[1]


class _State(NamedTuple):
    """What ``step`` and ``--dry-run`` both decide on."""

    insts: list
    live_jobs: list
    plans: dict[str, Mapping | None]
    plan_reports: tuple[tuple[str, str], ...]
    stores: dict
    coverage: dict
    reduce_errors: dict
    mount_faults: dict
    timeouts: dict
    cache_unsaved: bool = False  # a 429 kept the coverage cache from being saved this step
    #: The unit list of each in-flight job id of this run's instances; ``None`` when not known (spec §3.3).
    flight_units: dict[str, tuple[str, ...] | None] | None = None


def _decide(campaign, state: _State, baseline_error_ids: frozenset[str]) -> Decision:
    """``decide`` over a gathered state with the campaign's limits: the one call ``step`` and ``--dry-run`` share."""
    return decide(
        instances=state.insts,
        plans=state.plans,
        live_jobs=state.live_jobs,
        attempts=campaign.attempts,
        units_per_job=campaign.units_per_job,
        units_per_job_by_stage=campaign.units_per_job_by_stage,
        max_shards=campaign.max_shards,
        baseline_error_ids=baseline_error_ids,
        coverage=state.coverage,
        reduce_errors=state.reduce_errors,
        mount_faults=state.mount_faults,
        mount_cap_for=campaign.mount_cap_for,
        mount_fault_budget=campaign.mount_fault_budget,
        timeouts=state.timeouts,
        flight_units=state.flight_units or {},
    )


def _output_reduced(plan: Mapping | None) -> bool:
    """True when the plan's output stage is reduced: a reduced output never changes under its run key."""
    return plan is not None and any(s["name"] == plan["output"] and s["reduced"] for s in plan["stages"])


def _instances_done(plans: Mapping[str, Mapping | None]) -> int:
    """The instances whose output stage is reduced (HFU spec 14.8: the step line)."""
    return sum(1 for p in plans.values() if _output_reduced(p))


def _foreign_identity(plans: Mapping[str, Mapping | None], commit: str) -> str | None:
    """The FATAL detail for the first readable plan whose ``code_identity`` is not ``commit`` (a missing value
    counts as another identity), or ``None``. A tracked edit in the controller's tree moves the identity to
    ``dirty-<tree hash>``, while every job clones ``commit``: the two then read different stage tokens."""
    for inst_id, plan in sorted(plans.items()):
        if plan is None:
            continue
        ident = plan.get("code_identity")
        if ident != commit:
            return (
                f"{inst_id} planned under {str(ident)[:18]}, the jobs run {commit[:7]}; the tree changed after "
                "preflight; restore it, then resume"
            )
    return None


def _cached_reader(read: Callable, cache: dict[str, Mapping]) -> Callable:
    """``read`` that reuses the driver run's last plan of a reduced instance (HFU spec 16.4 A)."""

    def cached(inst):
        if inst.id in cache:
            return cache[inst.id]
        plan = read(inst)
        if _output_reduced(plan):
            cache[inst.id] = plan
        return plan

    return cached


def _say_ended_jobs(live_jobs: Sequence[Mapping], baseline_error_ids: frozenset[str], reported: set[str]) -> None:
    """One line per job of this run that newly ended ERROR or CANCELED (HFU spec 13.6, F5: launch 2's failures
    reached the owner only as HF emails). ``reported`` is owned by ``run``, so a job prints once per run."""
    for job in live_jobs:
        jid = str(job.get("id") or "")
        if not jid or jid in baseline_error_ids or jid in reported:
            continue
        verdict = classify(job)
        if verdict.outcome in (JobOutcome.COMPLETED, JobOutcome.IN_FLIGHT):
            continue
        labels = job.get("labels") or {}
        wall = job.get("wall_s")
        after = "-" if wall is None else f"{wall:.0f}"
        _say(
            f"[driver] job {jid} {labels.get('run_key')}/{labels.get('variant')}/{labels.get('stage')} "
            f"{shard_text(labels) if 'shard' in labels else '-'} "
            f"{verdict.reason or verdict.outcome.value} after {after} s"
        )
        reported.add(jid)


def _say(line: str) -> None:
    """One progress line per step, on stderr beside FATAL/STUCK, flushed so a redirected log grows live."""
    print(line, file=sys.stderr, flush=True)


def _default_gate(campaign, commit: str, api) -> Callable[[], object]:
    """The real launch gate: ``preflight.require_preflight`` against the campaign root of the bucket."""

    def gate():
        from scripts.hf_jobs.preflight import require_preflight
        from scripts.hf_jobs.store import BucketStore

        root = BucketStore(campaign.bucket, "", api=api)  # the report lives under _campaign/, not a run key
        return require_preflight(campaign.name, commit, campaign.sha256, store=root)

    return gate


def _gather(
    campaign,
    commit: str,
    api,
    replica,
    instances,
    read_plan,
    live_records,
    baseline_error_ids,
    fetch_logs=None,
    *,
    launched: LaunchedJobs | None = None,
) -> _State:
    """The state ``step`` and ``--dry-run`` both decide on, so a dry run shows what a launch would do: the
    instances, the live job records, each instance's ``--plan`` (a failed one becomes a report and ``None``, never
    an exception that stops the others), the stores, and the attribution from the per-job manifests."""
    from scripts.hf_jobs.store import BucketStore

    insts = _resolve_instances(campaign, commit, replica, instances)
    read = _resolve_read_plan(campaign, read_plan)
    live_jobs = list(_resolve_live_records(api, campaign, commit, live_records)())
    # A job that this run launched and that the listing does not show yet is held in flight (spec §3.3).
    launched = launched if launched is not None else LaunchedJobs()
    held, dropped = launched.reconcile(live_jobs)
    for job_id, label in dropped:
        _say(
            f"[driver] job {job_id} {label} was launched but is not in the listing after {UNLISTED_STEPS} steps; "
            "its units are free again"
        )
    live_jobs += held
    plans: dict[str, Mapping | None] = {}
    plan_reports: list[tuple[str, str]] = []

    def read_one(inst) -> tuple[Mapping | None, str | None]:
        try:
            return read(inst), None
        except RuntimeError as exc:  # `campaign.read_plan`: the instance's `--plan` exited non-zero
            if _plan_rate_limited(exc):  # a 429 inside the subprocess: the step backs off
                raise
            from scripts.hf_jobs.campaign import PlanTimeoutError

            if isinstance(exc, PlanTimeoutError):  # keep the HEAD: the timeout text, then the innermost frames first
                return None, f"--plan failed: {str(exc)[:PLAN_TIMEOUT_REPORT_CHARS]}"
            return None, f"--plan failed: {str(exc)[-300:]}"

    # In parallel; `map` keeps the instance order, so every decision equals a serial read's.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(PLAN_WORKERS, len(insts)))) as pool:
        for inst, (plan, err) in zip(insts, pool.map(read_one, insts), strict=True):
            plans[inst.id] = plan
            if err is not None:
                plan_reports.append((inst.id, err))
    stores = {inst.id: BucketStore(campaign.bucket, inst.run_key, api=api) for inst in insts}

    def store_for(inst):
        return stores[inst.id]

    coverage, reduce_errors, mount_faults, timeouts, unsaved = _attribute(
        insts,
        plans,
        live_jobs,
        baseline_error_ids,
        store_for,
        attempts=campaign.attempts,
        fetch_logs=_resolve_fetch_logs(api, fetch_logs),
    )
    reports = tuple(plan_reports)
    flight = _flight_units(insts, plans, live_jobs, launched, store_for)
    return _State(
        insts, live_jobs, plans, reports, stores, coverage, reduce_errors, mount_faults, timeouts, unsaved, flight
    )


def _flight_units(insts, plans, live_jobs, launched: LaunchedJobs, store_for) -> dict[str, tuple[str, ...] | None]:
    """The unit list of each in-flight job id of this run's instances (HFU launch top-up spec §3.3): from the run's
    record, else from the job's manifest in the store, read once in a run (the answer "none" is kept too)."""
    index = _instance_index(insts, plans)
    out: dict[str, tuple[str, ...] | None] = {}
    for job in live_jobs:
        if classify(job).outcome is not JobOutcome.IN_FLIGHT:
            continue
        labels = job.get("labels") or {}
        job_id = str(job.get("id") or "")
        if not job_id or (labels.get("run_key"), labels.get("variant")) not in index:
            continue
        if not launched.knows(job_id):
            got = _manifest_for(job, index, store_for)
            launched.remember(job_id, None if got is None else got[2])
        out[job_id] = launched.units_of(job_id)
    return out


def _resolve_fetch_logs(api, given: Callable | None) -> Callable[[str], Iterable[str]] | None:
    """The job-log reader of the 2-timeout rule (HFU spec 13.5): the HF API's ``fetch_job_logs`` by default."""
    if given is not None:
        return given
    fetch = getattr(api, "fetch_job_logs", None)
    if fetch is None:
        return None
    return lambda job_id: fetch(job_id=job_id)


def run(
    campaign,
    commit: str,
    *,
    api,
    token: str,
    replica: int | None = None,
    poll_s: float = 30.0,
    max_steps: int = 10000,
    step_fn: Callable[..., StepResult] | None = None,
    sleep_fn: Callable[[float], object] | None = None,
    baseline_fn: Callable[[], frozenset[str]] | None = None,
) -> int:
    """The launch/resume loop: step, then wait while jobs are in flight, until the campaign is done,
    a FATAL stops it, or it stalls (no launch, nothing in flight, not done -> stuck). Returns a process
    exit code: 0 done, 1 stalled/exhausted, 2 FATAL. ``step_fn``/``sleep_fn``/``baseline_fn`` are test seams.
    The jobs already terminal at the start are the baseline: a fault resolved before a resume never blocks."""
    do_step = step_fn or step
    sleep = sleep_fn or time.sleep
    baseline = baseline_fn() if baseline_fn is not None else _seed_baseline(campaign, commit, api=api)
    delay = poll_s  # doubles on contention or rising mount faults, resets on a clean step
    backoff_max = getattr(campaign, "backoff_max_s", 300.0)
    prev_mount_faults = 0
    reported: set[str] = set()  # each ended job prints one line per run
    plan_cache: dict[str, Mapping] = {}  # reduced instances are planned once per run (HFU spec 16.4 A)
    launched = LaunchedJobs()  # the jobs this run launched (HFU launch top-up spec §3.3)
    for _ in range(max_steps):
        result = do_step(
            campaign,
            commit,
            api=api,
            token=token,
            replica=replica,
            baseline_error_ids=baseline,
            reported=reported,
            plan_cache=plan_cache,
            launched=launched,
        )
        if result.fatal is not None:
            return 2
        if result.done:
            return 0
        if result.contention:  # a 429 step: back off; its mount_faults was never read, so keep prev
            delay = min(delay * 2, backoff_max)
            sleep(delay)
            continue
        if result.launched == 0 and result.in_flight == 0:
            return 1  # stalled: nothing launched, nothing in flight, not done, not contention -> stuck
        rising = result.mount_faults > prev_mount_faults  # new mount flakes this step -> sidecar under stress
        prev_mount_faults = result.mount_faults
        delay = min(delay * 2, backoff_max) if rising else poll_s  # back off on a rise, else reset
        sleep(delay)
    return 1  # exhausted max_steps without finishing


#: The waits before the retries of a rate-limited driver store write (HFU spec 15.4).
_PUT_BACKOFF_S: tuple[float, ...] = (2.0, 4.0, 8.0)


def _guarded_put(store, rel: str, data: bytes, *, sleep: Callable[[float], object] = time.sleep) -> bool:
    """``store.upload_bytes`` that retries a 429 after each ``_PUT_BACKOFF_S`` wait; False when they are spent.
    Any other error raises."""
    for wait in (*_PUT_BACKOFF_S, None):
        try:
            store.upload_bytes(rel, data)
            return True
        except Exception as exc:
            if not _is_contention(exc):
                raise
            if wait is None:
                return False
            sleep(wait)
    return False


def _is_contention(exc: BaseException) -> bool:
    """A 429 / rate-limit signal from the HF API, so ``run`` backs off."""
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


def _plan_rate_limited(exc: BaseException) -> bool:
    """A ``--plan`` failure that is an HF rate limit: the reason phrase or "rate limit", never a bare "429" (the
    text is subprocess stderr). A plan timeout is never one, so a permanent failure stays a STUCK report."""
    from scripts.hf_jobs.campaign import PlanTimeoutError

    if isinstance(exc, PlanTimeoutError):
        return False
    text = str(exc).lower()
    return "too many requests" in text or "rate limit" in text


def main(argv: Sequence[str] | None = None, *, load=None, make_api=None, head=None, runner=None) -> int:
    """``python -m scripts.hf_jobs campaign <file> <command> [opts]``. ``load``/``make_api``/``head``/
    ``runner`` are injection seams for tests; their defaults do the real I/O. Dropped vs the old CLI (the
    planner's commands; the driver does not size with the planner): ``profile``, ``plan`` (use ``--dry-run``),
    ``summary``."""
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(prog="python -m scripts.hf_jobs campaign")
    parser.add_argument("file", type=pathlib.Path)
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "launch",
            "resume",
            "status",
            "cancel",
            "retrieve",
            "verify",
            "adopt",
            "publish",
            "compare",
            "prune",
        ],
    )
    parser.add_argument("--replica", type=int, default=None)
    parser.add_argument("--poll-s", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true", help="launch/resume: print the decision, launch nothing")
    parser.add_argument("--staging", type=pathlib.Path, default=None, help="retrieve/verify/adopt/publish staging dir")
    parser.add_argument("--sandbox", action="store_true", help="publish to private silly-kicks/sandbox-<name> repos")
    parser.add_argument(
        "--i-have-owner-approval",
        action="append",
        default=[],
        metavar="REPO_OR_COMMIT7",
        help="publish: a public repo the owner approved; prune --delete: a commit7 the owner approved (once per item)",
    )
    parser.add_argument("--delete", action="store_true", help="prune: delete the run keys of the approved commits")
    parser.add_argument("--artifacts", default="", help="publish: comma-separated artifact ids (default: all)")
    parser.add_argument(
        "--baseline",
        type=pathlib.Path,
        action="append",
        default=None,
        help="compare: a staging dir of what shipped; repeat it, the last that holds an instance is read",
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    campaign = (load or _load_campaign)(args.file)
    if args.command == "compare":  # local files only: no HF API, no commit
        return _compare(campaign, args.staging, args.baseline)
    commit = (head or _git_head)()
    api, token = (make_api or _default_api)()

    if args.command == "preflight":
        from scripts.hf_jobs.campaign import cmd_preflight

        report = cmd_preflight(campaign, commit, api=api)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 1
    if args.command in ("launch", "resume"):
        if args.dry_run:
            return _dry_run(campaign, commit, api=api, replica=args.replica)
        return (runner or run)(campaign, commit, api=api, token=token, replica=args.replica, poll_s=args.poll_s)
    if args.command == "status":
        from scripts.hf_jobs.campaign import cmd_status

        print(json.dumps(cmd_status(campaign, commit, api=api, replica=args.replica), indent=2))
        return 0
    if args.command == "cancel":
        from scripts.hf_jobs.campaign import cmd_cancel

        print(json.dumps(cmd_cancel(campaign, commit, api=api)))
        return 0
    if args.command == "prune":  # lists by default; deletes only the commits the owner approved by name
        from scripts.hf_jobs import prune

        try:
            result = prune.prune(
                campaign,
                head=commit,
                delete=args.delete,
                approved=args.i_have_owner_approval,
                api=api,
                manifest_commits=prune.manifest_commits(),
            )
        except prune.PruneRefusedError as exc:  # a refusal comes before the first delete: one line, exit 1
            raise SystemExit(f"prune refused: {exc}") from None
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1

    from scripts.hf_jobs import adopt

    if args.staging is None:
        raise SystemExit(f"{args.command} needs --staging DIR (outside the repo)")
    if args.command == "retrieve":
        got = adopt.retrieve(campaign, commit, staging=args.staging, api=api, replica=args.replica)
        print(json.dumps({k: str(v) for k, v in got.items()}, indent=2))
    elif args.command == "verify":
        print(
            json.dumps(
                adopt.verify(campaign, commit, staging=args.staging, replica=args.replica), indent=2, default=str
            )
        )
    elif args.command == "adopt":
        print(json.dumps(adopt.adopt(campaign, commit, staging=args.staging), indent=2, default=str))
    else:  # publish
        ids = [x for x in args.artifacts.split(",") if x] or None
        print(
            json.dumps(
                adopt.publish(
                    ids, sandbox=args.sandbox, api=api, staging=args.staging, approved=args.i_have_owner_approval
                ),
                indent=2,
                default=str,
            )
        )
    return 0


#: `adopt.compare_staging` row ``baseline`` -> the words of the `compare` line.
_BASELINE_FORMS = {"adopted": "adopted copy", "retrieved": "retrieved files"}


def _compare(campaign, staging, baselines) -> int:
    """``compare --staging NEW --baseline OLD``: one line per instance, then the JSON; 0 when every artifact is
    identical under `identity.cross_commit_differences` and the census verdict of NEW is ok (HFU fit-units spec 6).
    A trainer line names the baseline form that was read: the adopted copy of OLD, or its retrieved files."""
    from scripts.hf_jobs import adopt

    if staging is None or not baselines:
        raise SystemExit("compare needs --staging DIR and at least one --baseline DIR (retrieved staging folders)")
    result = adopt.compare_staging(campaign, staging=staging, baselines=baselines)
    for key, row in result.items():
        if "census_ok" in row:
            print(f"[compare] {key} census verdict {'ok' if row['census_ok'] else 'NOT ok'}")
            continue
        form = _BASELINE_FORMS.get(row.get("baseline", ""))
        read = f" (baseline: {form} of {row.get('baseline_run_key')})" if form else ""
        if row["identical"]:
            print(f"[compare] {key} IDENTICAL{read}")
        else:
            print(f"[compare] {key} DIFFERENT{read}: {'; '.join(row['differences'])}")
    print(json.dumps(result, indent=2, default=str))
    return 0 if adopt.compare_ok(result) else 1


def _load_campaign(file):
    from scripts.hf_jobs.campaign import load_campaign

    return load_campaign(file)


def _git_head() -> str:
    from scripts.hf_jobs.campaign import _head

    return _head()


def _default_api():
    from huggingface_hub import HfApi, get_token

    token = get_token()
    if token is None:
        raise SystemExit("no HF token: run `hf auth login` first")
    return HfApi(), token


def _dry_run(
    campaign,
    commit: str,
    *,
    api,
    replica: int | None = None,
    instances: Sequence | None = None,
    read_plan: Callable | None = None,
    live_records: Callable[[], Sequence[Mapping]] | None = None,
    baseline_error_ids: frozenset[str] | None = None,
) -> int:
    """Print what ``decide`` would do, launching nothing: the same state as ``step`` and the same baseline as
    ``run``. The keyword arguments are test seams."""
    baseline = (
        baseline_error_ids
        if baseline_error_ids is not None
        else _seed_baseline(campaign, commit, api=api, live_records=live_records)
    )
    state = _gather(campaign, commit, api, replica, instances, read_plan, live_records, baseline)
    decision = _decide(campaign, state, baseline)
    foreign = _foreign_identity(state.plans, commit)  # what `step` would do: no launch (spec §19.2)
    if foreign is not None:
        decision = replace(decision, fatal=("code_identity", foreign), launches=())
    out = {
        "fatal": decision.fatal,
        "done": decision.done,
        "launches": [
            {
                "instance": lx.instance_id,
                "stages": list(lx.stages),
                "shard": list(lx.shard),
                "kind": lx.kind,
                "keys": len(lx.keys),
                "tag": lx.tag,
            }
            for lx in decision.launches
        ],
        "reports": [list(r) for r in (*state.plan_reports, *decision.reports)],
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


def _resolve_instances(campaign, commit: str, replica: int | None, given: Sequence | None) -> list:
    if given is not None:
        return list(given)
    from scripts.hf_jobs.campaign import instances as campaign_instances

    return list(campaign_instances(campaign, commit, replica=replica))


def _resolve_read_plan(campaign, given: Callable | None) -> Callable:
    if given is not None:
        return given
    from scripts.hf_jobs.campaign import read_plan as campaign_read_plan

    def read(inst):
        return campaign_read_plan(inst, bucket=campaign.bucket)

    return read


def _resolve_live_records(api, campaign, commit: str, given: Callable | None) -> Callable:
    if given is not None:
        return given
    from scripts.hf_jobs.launch import jobs_for
    from scripts.hf_jobs.ledger import job_record

    def live():
        return [job_record(j) for j in jobs_for(api, campaign=campaign.name, commit=commit)]

    return live


def _seed_baseline(campaign, commit: str, *, api, live_records: Callable | None = None) -> frozenset[str]:
    """The ids of every job already terminal now. ``decide`` ignores them for the FATAL stop and the coverage, so
    a resume re-attempts; an in-flight job is not seeded and counts. ``live_records`` is a test seam."""
    live = _resolve_live_records(api, campaign, commit, live_records)
    terminal = (JobOutcome.COMPLETED, JobOutcome.CANCELED, JobOutcome.FATAL, JobOutcome.RERUN, JobOutcome.RETRY)
    return frozenset(str(job.get("id") or "") for job in live() if classify(job).outcome in terminal)


def _resolve_launch(given: Callable | None) -> Callable:
    if given is not None:
        return given
    from scripts.hf_jobs.launch import launch as launch_fn

    return launch_fn


def decide(
    *,
    instances: Sequence,
    plans: Mapping[str, Mapping | None],
    live_jobs: Sequence[Mapping],
    attempts: int,
    units_per_job: int,
    max_shards: int,
    workers: int = 8,
    baseline_error_ids: frozenset[str] = frozenset(),
    coverage: Mapping[tuple[str, str, str], int] | None = None,
    reduce_errors: Mapping[tuple[str, str], int] | None = None,
    mount_faults: Mapping[str, int] | None = None,
    mount_cap_for: Callable[[str], int] | None = None,
    mount_fault_budget: int = 10**9,
    timeouts: Mapping[tuple[str, str, str], tuple[str, ...]] | None = None,
    units_per_job_by_stage: Mapping[str, int] | None = None,
    flight_units: Mapping[str, Sequence[str] | None] | None = None,
) -> Decision:
    """The launches, reports and FATAL of one step. A stage with jobs in flight gets jobs for the units that no
    job of its flight holds (HFU launch top-up spec §3.2); ``flight_units`` names the units of each in-flight job
    id (``None``: not known). Without it, a stage with a flight waits, as before the top-up."""
    units_per_job_default = units_per_job
    cover = coverage or {}
    per_stage = units_per_job_by_stage or {}  # "<job>/<stage>" -> units per job (HFU fit-units spec 3.4)
    timed_out = timeouts or {}
    red_err = reduce_errors or {}
    mfaults = mount_faults or {}
    cap_for = mount_cap_for or (lambda _d: 10**9)
    fatal = _fatal_block(instances, plans, live_jobs, baseline_error_ids)
    if fatal is not None:
        return Decision(fatal=fatal)

    index = _instance_index(instances, plans)
    in_flight_mounts = _in_flight_mounts(live_jobs, index)
    chosen_mounts: dict[str, int] = {}
    launches: list[Launch] = []
    reports: list[tuple[str, str]] = []
    # Mount faults over the budget are a mount outage: one report per dataset, never a per-unit STUCK.
    for d, nfaults in sorted(mfaults.items()):
        if nfaults > mount_fault_budget:
            reports.append((f"dataset:{d}", f"mount outage: {nfaults} mount faults > budget {mount_fault_budget}"))

    # Walk each instance's stages in DAG order. Schedule the FIRST ready, un-reduced stage only (the barrier).
    all_reduced = bool(instances)
    for inst in instances:
        plan = plans.get(inst.id)
        if plan is None:
            all_reduced = False
            continue
        for stage in plan["stages"]:
            if stage["reduced"]:
                continue
            all_reduced = False
            if not stage.get("ready", True):
                break  # upstream not reduced yet: wait
            name = stage["name"]
            # A stage's own job size (e.g. 36 memory-heavy fits in 12 jobs of 3), else the campaign's.
            units_per_job = int(per_stage.get(f"{inst.job}/{name}", units_per_job_default))
            # Candidates: pending or recorded-failed units, minus those spent (coverage >= attempts). Memory-
            # deferred units run only in an escalate job.
            pending = list(stage.get("pending_units") or [])
            candidates = list(dict.fromkeys(pending + list(stage.get("failed_units") or {})))
            memory_units = [u for u in (stage.get("memory_units") or []) if cover.get((inst.id, name, u), 0) < attempts]
            to_run = [u for u in candidates if u not in memory_units and cover.get((inst.id, name, u), 0) < attempts]
            # A unit charged by TIMEOUT_LIMIT timed-out jobs is STUCK: reported, never relaunched on any path.
            stuck = [
                u
                for u in dict.fromkeys([*to_run, *memory_units])
                if len(timed_out.get((inst.id, name, u), ())) >= TIMEOUT_LIMIT
            ]
            to_run = [u for u in to_run if u not in stuck]
            memory_units = [u for u in memory_units if u not in stuck]
            stage_launches: list[Launch] = []
            flight = _stage_flight(live_jobs, inst, name)
            if flight:  # a stage with jobs in flight makes no report; it can only be topped up (spec §3.2)
                stage_launches.extend(
                    _top_up(
                        inst, name, flight, flight_units or {}, to_run, memory_units, units_per_job, max_shards, workers
                    )
                )
            else:
                for u in stuck:
                    jobs = ", ".join(timed_out[(inst.id, name, u)])
                    reports.append((inst.id, f"{name}/{u} timed out {TIMEOUT_LIMIT}x (jobs {jobs})"))
                if memory_units:
                    stage_launches.extend(
                        _keyed_launches(inst, name, memory_units, units_per_job, max_shards, workers, escalate=True)
                    )
                if to_run and _keyed(inst):
                    stage_launches.extend(_keyed_launches(inst, name, to_run, units_per_job, max_shards, workers))
                elif to_run:  # a download instance: round-robin --shard I/N slices (its data is fetched by shard)
                    units = list(stage.get("units") or [])
                    stage_launches.extend(
                        _round_robin_launches(
                            inst,
                            name,
                            units,
                            candidates,
                            to_run,
                            units_per_job,
                            max_shards,
                            workers,
                            stuck=stuck,
                            reports=reports,
                        )
                    )
                elif not memory_units and not stuck:  # nothing can run: reduce or report
                    spent_pending = [u for u in pending if cover.get((inst.id, name, u), 0) >= attempts]
                    _reduce_or_report(
                        inst,
                        name,
                        len(candidates),
                        spent_pending,
                        stage.get("max_failures"),
                        red_err.get((inst.id, name), 0),
                        attempts,
                        workers,
                        stage_launches,
                        reports,
                        lost_units=stage.get("lost_units") or (),
                        lost_output=bool(stage.get("lost_output")),
                    )
            # Truncate to the mount headroom (in flight plus chosen this pass) of every mount dataset. A launch
            # that `step` later drops only under-counts, so the cap is never exceeded.
            mds = _mount_datasets(inst)
            if mds and stage_launches:
                headroom = min(cap_for(d) - in_flight_mounts.get(d, 0) - chosen_mounts.get(d, 0) for d in mds)
                if headroom < len(stage_launches):
                    stage_launches = stage_launches[: max(0, headroom)]
                for d in mds:
                    chosen_mounts[d] = chosen_mounts.get(d, 0) + len(stage_launches)
            launches.extend(stage_launches)
            break  # one stage per instance per step

    return Decision(launches=tuple(launches), done=all_reduced, reports=tuple(reports))


#: The operator path of every STUCK report (HFU fit-units spec C5). The driver counts the failed jobs it saw since
#: it started (the baseline rule), so a unit that the operator repairs stays STUCK in that run: a new run, `resume`,
#: has a new baseline and attempts the work again.
OPERATOR_PATH = "stop the driver, then run resume"


def stuck_line(instance_id: str, reason: str) -> str:
    """One ``STUCK <instance>: <reason>`` line that ends with the operator path (a reason that already names
    the path, as the lost-result reports do, is not given a second one)."""
    if reason.endswith(OPERATOR_PATH):
        return f"STUCK {instance_id}: {reason}"
    return f"STUCK {instance_id}: {reason}; fix the cause, {OPERATOR_PATH}"


#: What the operator does after a storage loss (HFU fit-units spec C4): a new driver run has a new baseline.
LOST_RESULT_PATH = f"free storage, {OPERATOR_PATH}"


def _reduce_or_report(
    inst,
    name: str,
    spent: int,
    spent_pending: Sequence[str],
    max_failures,
    reduce_errors: int,
    attempts,
    workers,
    launches,
    reports,
    *,
    lost_units: Sequence[str] = (),
    lost_output: bool = False,
) -> None:
    """No unit can run. A spent unit with no record (a crash that wrote no outcome) cannot be reduced over:
    report it. Else trigger the reduce when the spent recorded failures are within ``max_failures``
    (``exceeds_tolerance``, as ``_stages.reduce_stage``), at most ``attempts`` reduce triggers new to this run;
    else report the stage.

    A spent unit that the plan lists in ``lost_units`` (a `FileSet` result that lost its members), and a stage
    with ``lost_output`` whose reduce triggers are spent, are reported as a LOST result, never as a crash or a
    failed reduce: the jobs ended well and the files are gone (HFU fit-units spec C4)."""
    if spent_pending:
        lost = [u for u in spent_pending if u in set(lost_units)]
        for u in lost:
            reports.append((inst.id, f"{name}/{u}: result lost (members missing); {LOST_RESULT_PATH}"))
        crashed = len(spent_pending) - len(lost)
        if crashed:
            reports.append(
                (inst.id, f"{name}: {crashed} unit(s) crashed with no record (coverage == attempts); cannot reduce")
            )
        return
    if exceeds_tolerance(None if max_failures is None else int(max_failures), spent):
        reports.append((inst.id, f"{name}: {spent} unit(s) failed/crashed over max_failures={max_failures}"))
        return
    if reduce_errors >= attempts:
        if lost_output:
            reports.append((inst.id, f"{name}: output lost (members missing); {LOST_RESULT_PATH}"))
        else:
            reports.append((inst.id, f"{name}: reduce failed {reduce_errors} time(s) (attempts={attempts})"))
        return
    launches.append(Launch(inst.id, (name,), (0, 1), "reduce", workers))


def _job_spec(campaign, commit: str, inst, launch: Launch, plan: Mapping, *, store):
    """The ``launch.JobSpec`` of one Launch. An inner_parallel stage runs one unit with ``inner_width`` threads;
    any other stage runs ``budget`` unit workers. A keyed launch names its units by ``--shard-keys`` (an empty
    list for a reduce trigger); an unkeyed one runs its ``--shard I/N`` slice. Every job gets ONE HF attempt."""
    from scripts.hf_jobs.launch import BUCKET_MOUNT, JobSpec, keys_file_bytes

    by_name = {s["name"]: s for s in plan["stages"]}
    stage_rec = by_name[launch.stages[0]]
    # A per-"job/stage" peak-RSS hint caps each worker; an escalate job runs alone on the whole box.
    peak = campaign.peak_rss_by_stage.get(f"{inst.job}/{stage_rec['name']}")
    if launch.escalate:
        workers, inner_workers = 1, 1
        cap_bytes: int | None = _BOX_RAM - _HEADROOM
        peak_rss_bytes: int | None = None
    else:
        inner = bool(stage_rec.get("inner_parallel"))
        budget = launch.workers
        workers = 1 if inner else budget
        inner_workers = min(budget, int(stage_rec.get("inner_width", 1))) if inner else 1
        cap_bytes = int(peak) if peak else None
        peak_rss_bytes = int(peak) if peak else None

    shard_keys_path: str | None = None
    if launch.keyed:  # an unkeyed (download) shard runs its round-robin --shard I/N slice instead
        index, count = launch.shard
        # Per variant: sibling variants can share a run_key and a stage token. The tag keeps two launches of one
        # stage with the same (index, count) apart (HFU launch top-up spec §3.4).
        rel = (
            f"_partitions/{plan['pipeline']}/{stage_rec['name']}/{stage_rec['token']}/{inst.variant}/"
            f"{index}of{count}-{launch.tag}.json"
        )
        if not _guarded_put(store, rel, keys_file_bytes(launch.keys)):
            return None  # rate-limited: `step` skips this launch and backs off
        shard_keys_path = f"{BUCKET_MOUNT}/{inst.run_key}/{rel}"

    return JobSpec(
        campaign=campaign.name,
        run_key=inst.run_key,
        job=inst.job,
        variant=inst.variant,
        stages=launch.stages,
        shard=launch.shard,
        module=inst.module,
        args=inst.args,
        commit=commit,
        bucket=campaign.bucket,
        workers=workers,
        inner_workers=inner_workers,
        cap_bytes=cap_bytes,
        peak_rss_bytes=peak_rss_bytes,
        timeout_s=int(stage_rec.get("timeout_s") or DEFAULT_TIMEOUT_S),
        attempts=1,  # the driver is the only retry authority (`campaign.attempts` per unit)
        datasets=inst.datasets,
        extras=(),
        shard_keys_path=shard_keys_path,
        escalate=launch.escalate,
        tag=launch.tag,
    )


def _current_work(instances: Sequence, plans: Mapping[str, Mapping | None]) -> set[tuple]:
    """The ``(run_key, variant, stage)`` of every un-reduced stage of every current instance."""
    work: set[tuple] = set()
    for inst in instances:
        plan = plans.get(inst.id)
        if plan is None:
            continue
        for stage in plan["stages"]:
            if stage["reduced"]:
                continue
            work.add((inst.run_key, inst.variant, stage["name"]))
    return work


def _job_key(job: Mapping) -> tuple:
    labels = job.get("labels") or {}
    return (labels.get("run_key"), labels.get("variant"), labels.get("stage"))


def _fatal_block(
    instances: Sequence,
    plans: Mapping[str, Mapping | None],
    live_jobs: Sequence[Mapping],
    baseline_error_ids: frozenset[str],
) -> tuple[str, str] | None:
    """The ``(reason, detail)`` of a FATAL that stops the run, or ``None``: a FATAL new to this run (not in
    ``baseline_error_ids``) on any shard of current work. A resolved FATAL from before a resume does not block."""
    work = _current_work(instances, plans)
    for job in live_jobs:
        if str(job.get("id") or "") in baseline_error_ids:
            continue  # a fault present before this run started: the operator chose to resume past it
        verdict = classify(job)
        if verdict.outcome is JobOutcome.FATAL and _job_key(job) in work:
            return (verdict.reason, verdict.detail)
    return None


def by_job_rel(plan: Mapping, stage: str, token: str, job_id: str) -> str:
    """The per-job manifest path: the units a job was launched to run, keyed by its HF job id (a job record
    never carries them). An empty list marks a reduce trigger."""
    return f"_partitions/{plan['pipeline']}/{stage}/{token}/by_job/{job_id}.json"


def _instance_index(insts: Sequence, plans: Mapping[str, Mapping | None]) -> dict[tuple, tuple]:
    """``{(run_key, variant): (instance, plan)}`` over the instances with a readable plan."""
    return {(inst.run_key, inst.variant): (inst, plans[inst.id]) for inst in insts if plans.get(inst.id) is not None}


def _manifest_for(job: Mapping, index: Mapping[tuple, tuple], store_for) -> tuple | None:
    """``(instance, stage record, [units])`` for a terminal job attributed to its instance by labels and to its
    units by the per-job manifest, or ``None`` when the job is not this run's work or has no manifest (launched
    outside the controller, or a stale generation)."""
    labels = job.get("labels") or {}
    found = index.get((labels.get("run_key"), labels.get("variant")))
    if found is None:
        return None
    inst, plan = found
    stage = labels.get("stage")
    if not stage:
        return None  # no stage label: not a job this controller launched
    srec = next((s for s in plan["stages"] if s["name"] == stage), None)
    if srec is None:
        return None
    raw = store_for(inst).read_bytes(by_job_rel(plan, stage, srec["token"], str(job.get("id") or "")))
    if raw is None:
        return None
    return inst, srec, list(json.loads(raw))


#: The per-``run_key`` read-once coverage cache (under the ``_partitions/`` upload allowlist).
_COVERAGE_STATE = "_partitions/_coverage/state.json"


def _load_cache(store) -> dict:
    """The per-run_key read-once job cache: ``{job_id: {inst_id, stage, created_at, outcome, reason, units}}``.
    A reconstructable CACHE -- ``{}`` on absent or unparseable (a torn/garbage object ⇒ one O(N) rebuild)."""
    raw = store.read_bytes(_COVERAGE_STATE)
    if not raw:
        return {}
    try:
        got = json.loads(raw)
        return got if isinstance(got, dict) else {}
    except (ValueError, TypeError):
        return {}


def _save_cache(store, cache: dict) -> bool:
    """Whole-object PUT (a full replace, no torn object); False when a 429 outlasts the retries."""
    return _guarded_put(store, _COVERAGE_STATE, json.dumps(cache).encode("utf-8"))


#: How much of a plan-timeout message a STUCK report keeps: the timeout text plus the innermost dump frames.
PLAN_TIMEOUT_REPORT_CHARS = 1500
#: A unit charged by this many timed-out jobs is STUCK, not relaunched (HFU spec 13.5).
TIMEOUT_LIMIT = 2
_UNIT_START = re.compile(r"\[unit\] (?P<stage>[^/\s]+)/(?P<key>\S+) start$")
_UNIT_END = re.compile(r"\[unit\] (?P<stage>[^/\s]+)/(?P<key>\S+) (?:done|failed|excluded) in ")


def started_unfinished(log_lines: Iterable[str], stage: str) -> frozenset[str]:
    """The keys of ``stage`` with a ``[unit] <stage>/<key> start`` line and no end line (HFU spec 13.5/13.6)."""
    started: set[str] = set()
    ended: set[str] = set()
    for line in log_lines:
        if (m := _UNIT_START.search(line.rstrip())) and m["stage"] == stage:
            started.add(m["key"])
        elif (m := _UNIT_END.search(line)) and m["stage"] == stage:
            ended.add(m["key"])
    return frozenset(started - ended)


def _attribute(insts, plans, live_jobs, baseline_error_ids, store_for, *, attempts, fetch_logs=None):
    """Fold each new terminal job into its run_key's read-once cache (each manifest read once), save the cache,
    then recompute from the whole cache in ``created_at`` order (equal to a full rescan in any termination
    order). Returns ``(cover, reduce_errors, mount_faults, timeouts, cache_unsaved)``.

    A COMPLETED job credits its units; a crash RETRY charges its not-yet-spent units, recorded failures first; a
    mount fault is per dataset, never per unit; a reduce trigger (empty manifest) is a per-stage reduce failure;
    a timeout (RERUN) spends no coverage and charges ``timeouts`` with the units its log shows started and not
    finished (one log fetch per job; a failed fetch is retried next step)."""
    index = _instance_index(insts, plans)
    inst_by_id = {inst.id: inst for inst in insts}
    stores: dict[str, object] = {}
    for inst in insts:
        if plans.get(inst.id) is not None:
            stores.setdefault(inst.run_key, store_for(inst))

    merged: dict[str, dict] = {}
    unsaved = False
    for rk, store in stores.items():
        cache = _load_cache(store)
        changed = False
        for job in live_jobs:
            jid = str(job.get("id") or "")
            if not jid or jid in baseline_error_ids or jid in cache:
                continue
            if (job.get("labels") or {}).get("run_key") != rk:
                continue
            verdict = classify(job)
            if verdict.outcome not in (JobOutcome.COMPLETED, JobOutcome.RETRY, JobOutcome.RERUN):
                continue  # FATAL/CANCELED/in-flight: not a terminal RUN attempt to attribute
            got = _manifest_for(job, index, store_for)  # the ONLY bucket read, once per new job
            if got is None:
                continue
            inst, srec, units = got
            started: list[str] = []
            if verdict.outcome is JobOutcome.RERUN and fetch_logs is not None:
                try:
                    lines = list(fetch_logs(jid))
                except Exception as exc:  # broad BY DESIGN: a log fetch never fails the step; it retries next step
                    _say(f"[driver] job {jid}: log fetch failed ({type(exc).__name__}: {exc}); timeout not charged yet")
                    continue
                started = sorted(started_unfinished(lines, srec["name"]) & set(units))
            cache[jid] = {
                "inst_id": inst.id,
                "stage": srec["name"],
                "created_at": str(job.get("created_at") or ""),
                "outcome": verdict.outcome.value,
                "reason": verdict.reason,
                "units": list(units),
                "started": started,
            }
            changed = True
        if changed and not _save_cache(store, cache):
            unsaved = True  # the cache is rebuilt next step
        merged.update(cache)

    cover: dict[tuple[str, str, str], int] = {}
    reduce_jobs: dict[tuple[str, str], set[str]] = {}
    mount_jobs: dict[str, set[str]] = {}
    timeouts: dict[tuple[str, str, str], list[str]] = {}
    for jid, rec in sorted(merged.items(), key=lambda kv: (kv[1].get("created_at", ""), kv[0])):
        if jid in baseline_error_ids:
            continue  # a prior run's job: never recounted on a resume
        inst = inst_by_id.get(rec["inst_id"])
        if inst is None:
            continue  # an instance not in this pass (e.g. a replica filter)
        stage, units, reason, outcome = rec["stage"], rec["units"], rec.get("reason", ""), rec["outcome"]
        srec = next((s for s in (plans.get(inst.id) or {}).get("stages", []) if s["name"] == stage), {})
        if srec.get("reduced"):  # a reduced stage is done: its cached jobs no longer count
            continue
        if reason in ("mount_auth_flake", "mount_failed"):  # by-locus: a mount fault is per-DATASET, never per-unit
            for d in _mount_datasets(inst):
                mount_jobs.setdefault(d, set()).add(jid)
            continue
        if not units:  # a reduce trigger ran no unit
            reduce_jobs.setdefault((inst.id, stage), set()).add(jid)
            continue
        if outcome == JobOutcome.RERUN.value:  # a timeout charges only the units that started in it (HFU spec 13.5)
            for u in rec.get("started") or ():
                timeouts.setdefault((inst.id, stage, u), []).append(jid)
            continue
        keys = list(units)
        if outcome == JobOutcome.RETRY.value:
            live = [u for u in keys if cover.get((inst.id, stage, u), 0) < attempts]
            recorded = [u for u in live if u in (srec.get("failed_units") or {})]
            keys = recorded or live
        for u in keys:
            ck = (inst.id, stage, u)
            cover[ck] = cover.get(ck, 0) + 1
    return (
        cover,
        {k: len(v) for k, v in reduce_jobs.items()},
        {k: len(v) for k, v in mount_jobs.items()},
        {k: tuple(v) for k, v in timeouts.items()},
        unsaved,
    )


def _stage_flight(live_jobs: Sequence[Mapping], inst, stage: str) -> list[Mapping]:
    """The jobs of this (instance, stage) that are RUNNING or SCHEDULING: the flight of the stage."""
    want = (inst.run_key, inst.variant, stage)
    flight = []
    for job in live_jobs:
        if classify(job).outcome is not JobOutcome.IN_FLIGHT:
            continue
        labels = job.get("labels") or {}
        if (labels.get("run_key"), labels.get("variant"), labels.get("stage")) == want:
            flight.append(job)
    return flight


def _top_up(
    inst,
    stage: str,
    flight: Sequence[Mapping],
    flight_units: Mapping[str, Sequence[str] | None],
    to_run: Sequence[str],
    memory_units: Sequence[str],
    units_per_job: int,
    max_shards: int,
    workers: int,
) -> list[Launch]:
    """The launches of a stage that has jobs in flight (HFU launch top-up spec §3.2, cases 2 and 3). None when the
    instance is not keyed, or a job of the flight has no known unit list or an empty one (a reduce trigger): the
    stage then waits for its flight. Else the units that no job of the flight holds, in at most
    ``max_shards - len(flight)`` jobs: the escalation launches first, then the run launches in the room left."""
    if not _keyed(inst):
        return []
    held: set[str] = set()
    for job in flight:
        units = flight_units.get(str(job.get("id") or ""))
        if not units:
            return []
        held.update(units)
    rest = [u for u in to_run if u not in held]
    rest_memory = [u for u in memory_units if u not in held]
    room = max_shards - len(flight)
    if room <= 0:
        return []
    out: list[Launch] = []
    if rest_memory:
        out.extend(_keyed_launches(inst, stage, rest_memory, units_per_job, room, workers, escalate=True))
    if rest and room - len(out) >= 1:
        out.extend(_keyed_launches(inst, stage, rest, units_per_job, room - len(out), workers))
    return out


def _keyed(inst) -> bool:
    """False for an instance with a ``download`` dataset: it is fetched by ``--shard I/N``, so it runs round-robin."""
    return not any(getattr(d, "access", None) == "download" for d in inst.datasets)


def _nshards(n_units: int, units_per_job: int, max_shards: int) -> int:
    """``ceil(n_units / units_per_job)`` jobs, at most ``max_shards``: few big jobs."""
    upj = max(1, int(units_per_job))
    return max(1, min(int(max_shards), (int(n_units) + upj - 1) // upj))


def _contiguous(keys: Sequence[str], n: int) -> list[list[str]]:
    keys = sorted(keys)
    return [keys[i * len(keys) // n : (i + 1) * len(keys) // n] for i in range(n)]


def _round_robin_launches(
    inst,
    stage: str,
    units: Sequence[str],
    candidates: Sequence[str],
    to_run: Sequence[str],
    units_per_job: int,
    max_shards: int,
    workers: int,
    *,
    stuck: Sequence[str] = (),
    reports: list | None = None,
) -> list[Launch]:
    """The round-robin ``--shard I/N`` slices of a download instance, each launched only when it holds a unit to
    run; its recorded keys are its slice's candidates. A shard that holds a STUCK unit is held back and reported."""
    from scripts._driver import shard_select  # the slice the executor runs for `--shard I/N`

    n = _nshards(len(units), units_per_job, max_shards)
    cand, want, held = set(candidates), set(to_run), set(stuck)
    out = []
    for i in range(n):
        mine = [u for u in shard_select(list(units), i, n) if u in cand]
        if not any(u in want for u in mine):
            continue
        blocking = [u for u in mine if u in held]
        if blocking:
            if reports is not None:
                reports.append((inst.id, f"{stage} shard {i}of{n} held back: it holds STUCK unit(s) {blocking}"))
            continue
        out.append(Launch(inst.id, (stage,), (i, n), "run", workers, tuple(mine), keyed=False))
    return out


def _keyed_launches(
    inst, stage: str, keys: Sequence[str], units_per_job: int, max_shards: int, workers: int, *, escalate=False
) -> list[Launch]:
    """The explicit keys in ``ceil(len(keys) / units_per_job)`` jobs (at most ``max_shards``), a contiguous
    near-even split of the sorted keys. An escalate job runs memory-deferred units alone on the box (W=1)."""
    n = _nshards(len(keys), units_per_job, max_shards)
    return [
        Launch(inst.id, (stage,), (i, n), "run", 1 if escalate else workers, tuple(group), escalate=escalate)
        for i, group in enumerate(_contiguous(keys, n))
        if group
    ]


def _mount_datasets(inst) -> set[str]:
    """The instance's CSI-mounted datasets (a ``mount`` dataset or an open dataset's snapshot; maybe two)."""
    return {d.name for d in inst.datasets if getattr(d, "mounted", getattr(d, "access", None) == "mount")}


def _in_flight_mounts(live_jobs: Sequence[Mapping], index: Mapping[tuple, tuple]) -> dict[str, int]:
    """``{dataset: in-flight jobs that mount it}``."""
    counts: dict[str, int] = {}
    for job in live_jobs:
        if classify(job).outcome is not JobOutcome.IN_FLIGHT:
            continue
        labels = job.get("labels") or {}
        found = index.get((labels.get("run_key"), labels.get("variant")))
        if found is None:
            continue
        for d in _mount_datasets(found[0]):
            counts[d] = counts.get(d, 0) + 1
    return counts
