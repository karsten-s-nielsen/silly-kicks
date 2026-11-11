"""The HF campaign model and the operator commands that are not the driver loop (ADR-128, spec 5.5).

A campaign is a COMMITTED TOML file (``scripts/hf_jobs/campaigns/<name>.toml``, INV-7): name, bucket,
concurrency cap, attempts, and per job the trainer or driver, its tier, its dataset and its variants.
Run keys are DERIVED (``{campaign}-{commit7}-{job}-{tier}``, plus ``-r<N>`` for a fresh replica store,
decision P18), so two campaigns can never share a store by accident.

The controller is the minimal driver (``scripts/hf_jobs/driver.py``, ADR-128 amendment 2026-11-02): it
holds no state and rebuilds it every step from each instance's ``--plan`` and the HF job labels. This module
supplies its model (``load_campaign``, ``instances``, ``read_plan``) and the ``preflight`` / ``status`` /
``cancel`` commands. ``python -m scripts.hf_jobs campaign <file> <command>`` routes to ``driver.main``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from collections.abc import Iterable, Mapping
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True)
class CampaignJob:
    job: str
    tier: str
    dataset: str
    variants: tuple[str, ...]
    probe_dataset: str | None = None
    #: A second staged dataset the trainer pools in (``--pool-dataset``); mounted and capped like the probe.
    pool_dataset: str | None = None

    @property
    def datasets(self) -> tuple[str, ...]:
        """Every dataset the job mounts: its dataset, then the probe and pool datasets it names."""
        return tuple(n for n in (self.dataset, self.probe_dataset, self.pool_dataset) if n)


@dataclasses.dataclass(frozen=True)
class Campaign:
    name: str
    bucket: str
    #: The in-flight ceiling: at most this many jobs run at once (not the per-stage shard count, which is
    #: `units_per_job`/`max_shards` since the resource-sharing overhaul, 2026-11-03).
    concurrency_cap: int
    #: The per-unit bound: a unit that this many of the run's jobs covered without completing is reported.
    attempts: int
    jobs: tuple[CampaignJob, ...]
    path: pathlib.Path | None
    sha256: str
    #: Job-sizing: a stage of N units launches ceil(N / units_per_job) jobs, capped at max_shards.
    units_per_job: int = 20
    max_shards: int = 12
    #: Per-dataset concurrent-mount cap: default plus overrides keyed by dataset name. Distinct from the
    #: in-job read gate `read_concurrency`.
    mount_cap: int = 25
    mount_cap_by_dataset: Mapping[str, int] = dataclasses.field(default_factory=dict)
    #: a dataset whose mount faults (transient sidecar 401s) exceed this is reported a mount outage.
    mount_fault_budget: int = 12
    #: Launch pacing / backoff.
    launch_batch: int = 8
    backoff_max_s: float = 300.0
    #: Memory sizing hint: peak RSS bytes keyed by "job/stage" for the memory-heavy stages only.
    peak_rss_by_stage: Mapping[str, int] = dataclasses.field(default_factory=dict)
    #: A stage's own job size, keyed by "job/stage": it replaces `units_per_job` for that stage (HFU fit-units
    #: spec 3.4: the 36 xsuccess fits run as 12 jobs of 3, the width that their memory allows).
    units_per_job_by_stage: Mapping[str, int] = dataclasses.field(default_factory=dict)
    #: Preflight check 7 (`preflight.check_storage`): the org's private storage must leave the headroom under
    #: the limit. The limit is 100 GB (owner-verified 2026-11-10, the HF billing page); the headroom is two
    #: campaigns and a margin.
    storage_limit_bytes: int = 100_000_000_000
    storage_headroom_bytes: int = 20_000_000_000

    def mount_cap_for(self, dataset: str) -> int:
        return int(self.mount_cap_by_dataset.get(dataset, self.mount_cap))


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


# ---- model ---------------------------------------------------------------------------------------


def load_campaign(path: str | pathlib.Path) -> Campaign:
    from scripts._toml import load_toml

    path = pathlib.Path(path)
    raw = load_toml(path)
    c = raw["campaign"]
    jobs = tuple(
        CampaignJob(
            j["job"], j["tier"], j["dataset"], tuple(j["variants"]), j.get("probe_dataset"), j.get("pool_dataset")
        )
        for j in raw.get("jobs", [])
    )
    defaults = Campaign.__dataclass_fields__
    return Campaign(
        name=c["name"],
        bucket=c["bucket"],
        concurrency_cap=int(c["concurrency_cap"]),
        attempts=int(c["attempts"]),
        jobs=jobs,
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        units_per_job=int(c.get("units_per_job", defaults["units_per_job"].default)),
        max_shards=int(c.get("max_shards", defaults["max_shards"].default)),
        mount_cap=int(c.get("mount_cap", defaults["mount_cap"].default)),
        mount_cap_by_dataset={str(k): int(v) for k, v in (c.get("mount_cap_by_dataset") or {}).items()},
        mount_fault_budget=int(c.get("mount_fault_budget", defaults["mount_fault_budget"].default)),
        launch_batch=int(c.get("launch_batch", defaults["launch_batch"].default)),
        backoff_max_s=float(c.get("backoff_max_s", defaults["backoff_max_s"].default)),
        peak_rss_by_stage={str(k): int(v) for k, v in (c.get("peak_rss_by_stage") or {}).items()},
        units_per_job_by_stage={str(k): int(v) for k, v in (c.get("units_per_job_by_stage") or {}).items()},
        storage_limit_bytes=int(c.get("storage_limit_bytes", defaults["storage_limit_bytes"].default)),
        storage_headroom_bytes=int(c.get("storage_headroom_bytes", defaults["storage_headroom_bytes"].default)),
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
        d, rec = declared()[name], locked(name)
        snap = rec.get("snapshot")  # an open dataset mounts its staged snapshot once it is recorded
        cap = d.read_concurrency if (d.access == "mount" or snap) else None
        out.append(
            DatasetMount(
                name,
                d.repo,
                rec["revision"],
                d.access,
                cap,
                d.snapshot_repo if snap else None,
                snap["revision"] if snap else None,
            )
        )
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
        mounts = _mounts(j.datasets)
        if isinstance(spec, DriverSpec):
            for v in j.variants:
                args = (*spec.base_args, "--dataset", j.dataset)
                out.append(Instance(j.job, j.tier, v, spec.module, args, mounts, rk))
            continue
        builds = {b.variant: b for b in spec.builds}
        for v in j.variants:
            b = builds[v]
            args = (*spec.base_args, *b.run_args, "--dataset", j.dataset)
            if j.probe_dataset is not None:
                args += ("--probe-dataset", j.probe_dataset)
            if j.pool_dataset is not None:
                args += ("--pool-dataset", j.pool_dataset)
            if spec.visibility_gated:
                args += ("--arms-json", json.dumps(arm_policy.compute_arms(staged_units(j.dataset)), sort_keys=True))
            if j.job == "receiver_model":  # decision P5: visibility is a parameter, never read in-job
                args += (
                    "--corpus-visibility",
                    declared()[j.dataset].visibility,
                    "--corpus-n-matches",
                    str(locked(j.dataset)["n_units"]),
                )
            out.append(Instance(j.job, j.tier, v, spec.pipeline or spec.module, args, mounts, rk))
    return out


def store_uri(bucket: str, key: str) -> str:
    return f"hf://buckets/{bucket}/{key}"


#: A `--plan` read that runs longer is killed and reported (HFU spec 13.4, F3: three plan reads hung 2 h 10 min).
PLAN_TIMEOUT_S = 600
PLAN_DEADLINE_ENV = "SILLY_KICKS_PLAN_DEADLINE_S"


class PlanTimeoutError(RuntimeError):
    """A ``--plan`` read passed its deadline; the message holds the child's dumped stack."""


#: How much of the child's stack dump a plan-timeout report keeps (from the dump's start).
PLAN_DUMP_KEEP_CHARS = 3000


def _plan_timeout_detail(exc: subprocess.TimeoutExpired) -> str:
    """The child's stack dump from its header line (`liveness.stack_dump`: the main thread and the most recent call
    FIRST, so its head holds the frame where the plan hung). Without a dump, the stderr tail.
    ``TimeoutExpired.stderr`` is ``bytes`` on POSIX even with ``text=True`` and ``str`` on Windows."""
    from scripts.hf_jobs.liveness import DUMP_MARK

    err = exc.stderr
    if isinstance(err, bytes):
        err = err.decode("utf-8", errors="replace")
    err = err or ""
    mark = err.rfind(DUMP_MARK)
    if mark < 0:
        return err[-PLAN_DUMP_KEEP_CHARS:]
    return err[err.rfind("\n", 0, mark) + 1 :][:PLAN_DUMP_KEEP_CHARS]


def read_plan(inst: Instance, *, bucket: str, run=subprocess.run) -> dict:
    """The instance's ``--plan`` JSON against its bucket store (read-only; the module is the authority). A read past
    ``PLAN_TIMEOUT_S`` is killed and raises ``PlanTimeoutError`` with the child's dumped stack. The Hub request
    count that the plan process prints is added to `hub_client`'s tally, also for a plan that ended with an error
    (a 429 inside the plan). A plan that timed out was killed and reports none."""
    from scripts.hf_jobs.hub_client import add_plan_requests, parse_plan_requests

    argv = [sys.executable, "-m", inst.module, *inst.args, "--plan", "--store", store_uri(bucket, inst.run_key)]
    env = {**os.environ, PLAN_DEADLINE_ENV: str(PLAN_TIMEOUT_S)}
    try:
        p = run(argv, capture_output=True, text=True, check=False, cwd=REPO_ROOT, timeout=PLAN_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired as exc:
        raise PlanTimeoutError(
            f"{inst.id}: --plan failed (timeout after {PLAN_TIMEOUT_S} s): {_plan_timeout_detail(exc)}"
        ) from None
    sent = parse_plan_requests(p.stderr)
    if sent is not None:
        add_plan_requests(sent)
    if p.returncode != 0:
        raise RuntimeError(f"{inst.id}: --plan failed (rc={p.returncode}): {(p.stderr or '')[-2000:]}")
    return json.loads(p.stdout)


def _head() -> str:
    argv = ["git", "rev-parse", "HEAD"]  # git from PATH is the house pattern
    return subprocess.run(argv, capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout.strip()  # noqa: S603


# ---- operator commands ---------------------------------------------------------------------------


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


def cmd_preflight(c: Campaign, commit: str, *, api: Any) -> dict:
    """The seven no-spend checks; uploads the report the launch gate (``preflight.require_preflight``, called
    by ``driver.step``) requires (spec 5.4): commit pushed and green, lockfile and image, numba, API contract,
    datasets, every instance's ``--plan``, and the org's storage headroom (HFU fit-units spec C3)."""
    from scripts.hf_jobs import preflight
    from scripts.hf_jobs.store import BucketStore

    insts = instances(c, commit)
    names = [n for j in c.jobs for n in j.datasets]
    checks = preflight.run_checks(
        [
            ("check_ref", lambda: preflight.check_ref(commit)),
            ("check_lock", preflight.check_lock),
            ("check_numba", preflight.check_numba),
            ("check_api", lambda: preflight.check_api(api)),
            ("check_datasets", lambda: preflight.check_datasets(names, api=api)),
            ("check_plans", lambda: preflight.check_plans(insts)),
            (
                "check_storage",
                lambda: preflight.check_storage(
                    api,
                    org=c.bucket.split("/")[0],
                    limit_bytes=c.storage_limit_bytes,
                    headroom_bytes=c.storage_headroom_bytes,
                ),
            ),
        ]
    )
    report = preflight.assemble_report(c.name, commit, c.sha256, checks)
    BucketStore(c.bucket, "", api=api).upload_bytes(
        preflight.REPORT_REL.format(campaign=c.name, commit7=commit[:7]), json.dumps(report, indent=2).encode("utf-8")
    )
    return report


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
