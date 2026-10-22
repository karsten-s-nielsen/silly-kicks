"""Launch ADR-128 stage-contract jobs on HF Jobs through the typed `huggingface_hub` API.

One image (digest-pinned, decision P1), one flavor (`cpu-upgrade`, INV-4), one in-job environment built
with `uv sync --frozen` from the committed `uv.lock` (INV-6), and an exact-commit checkout verified with
`git rev-parse` (the clone commit is the campaign commit; preflight proved it is pushed and green). Each
job carries labels (`campaign`, `run_key`, `job`, `variant`, `stage`, `shard`, `commit`) so the stateless
controller rebuilds state from `list_jobs(labels=...)`. Storage: the run's bucket prefix is mounted
read-write at /scratch (the store; persists across `attempts`, so a retry resumes); each staged dataset is
mounted read-only at /data/<name> at its pinned revision (or fetched per shard, `access = "download"`);
a mount's reads are bounded by the job-wide dataset read gate (`read_concurrency`, `scripts._dataset_reads`).
The `hf` CLI stays a human tool; nothing here shells out to it.
"""

from __future__ import annotations

import dataclasses
import re
import shlex
from collections.abc import Iterable, Mapping
from typing import Any, Final

from scripts._dataset_reads import READ_CAP_ENV, STORE_READ_CAP_ENV
from scripts._thread_pin import DETERMINISTIC_VARS

FLAVOR: Final = "cpu-upgrade"
#: The base image (the Dockerfile's `FROM`). The JOB pulls the prebuilt image (`image.image_ref()`); this
#: constant stays as the base (kept equal to `image.BASE_IMAGE` by a test). `IMAGE` is a back-compat alias.
BASE_IMAGE: Final = "python:3.12.14-bookworm@sha256:dbbe4ceb97851e2e5fa83798b239811f871cb743b259ba3563737349f6bcfaa0"
IMAGE: Final = BASE_IMAGE
PYTHON_BIN: Final = "/usr/local/bin/python3.12"
UV_REQUIREMENT: Final = "uv==0.9.28 --hash=sha256:7b8460a2b624d8ab27cb293a2c9f2393f9efc4e36e0fb886a6c2360e23fb48be"
REPO_URL = "https://github.com/karsten-s-nielsen/silly-kicks"
BUCKET_MOUNT = "/scratch"  # the read-write bucket is mounted here inside every job
DATA_MOUNT: Final = "/data"
PHASE_FILE: Final = "/tmp/sk-phases.jsonl"  # noqa: S108 -- the job script's phase timestamps (S7.2); in-job only
TERMINAL = frozenset({"COMPLETED", "CANCELED", "ERROR", "DELETED"})
#: HF Jobs rejects a label value (the job `name` is the `name` label) longer than this, or outside this
#: alphabet. Real-HF 400s on the Cycle 1 launches (2026-10-19): `Too big: expected string to have <=100
#: characters at labels.name`, then `Invalid string: must match pattern /^[a-zA-Z0-9_-]*$/ at labels.name`.
LABEL_MAX: Final = 100
LABEL_PATTERN: Final = re.compile(r"[a-zA-Z0-9_-]*")


@dataclasses.dataclass(frozen=True)
class DatasetMount:
    name: str
    repo: str
    revision: str
    access: str  # "mount" | "download" | "open" (StatsBomb open data, fetched per unit, revision-pinned)
    #: A mount's most concurrent whole-file reads per job (`datasets.toml`; the dataset read gate,
    #: `scripts._dataset_reads`). Required for a mount, refused for any other access.
    read_concurrency: int | None = None

    def __post_init__(self) -> None:
        if self.access == "mount":
            if not isinstance(self.read_concurrency, int) or self.read_concurrency < 1:
                raise ValueError(f"{self.name}: a mounted dataset needs read_concurrency >= 1")
        elif self.read_concurrency is not None:
            raise ValueError(f"{self.name}: read_concurrency applies to a mount only (access={self.access!r})")

    @property
    def path(self) -> str:
        return f"{DATA_MOUNT}/{self.name}"


@dataclasses.dataclass(frozen=True)
class JobSpec:
    campaign: str
    run_key: str
    job: str
    variant: str
    stages: tuple[str, ...]
    shard: tuple[int, int]
    module: str
    args: tuple[str, ...]
    commit: str
    bucket: str
    workers: int
    inner_workers: int
    cap_bytes: int | None
    peak_rss_bytes: int | None
    timeout_s: int
    attempts: int
    datasets: tuple[DatasetMount, ...]
    profile: int | None = None
    shard_keys_path: str | None = None
    extra_env: Mapping[str, str] | None = None
    escalate: bool = False
    action_id: str | None = None
    action_kind: str = "run"
    #: ADR-129 canary: the MAIN store mount to read allowed upstream generations from, and the allow file.
    read_through: str | None = None
    read_through_allow: str | None = None


def parse_stage_label(value: str) -> tuple[tuple[str, ...], str | None]:
    """THE parser of the `stage` label (every reader goes through it): ``a-b`` or, under the R8 fallback,
    ``a-b__<action id>``. Stage names never contain ``-`` or ``__`` (`Pipeline.validate`), and an action id
    never contains ``-``."""
    stages, sep, action = value.partition("__")
    return tuple(s for s in stages.split("-") if s), (action if sep else None)


def label_of(labels: Mapping[str, str]) -> str:
    """The in-job label (`SILLY_KICKS_JOB_LABEL`, stamped on every unit ledger record) rebuilt from a job's
    HF labels; the ledger joins unit records to their job by it. The `action` label (S5.3) is appended, so
    two actions with the same `n`, or two attempts of one retried job, never share a label. A job record with
    no `action` label (every Cycle 1 job) keeps today's value."""
    stages, folded = parse_stage_label(labels["stage"])
    base = f"{labels['job']}.{labels['variant']}.{'-'.join(stages)}.{labels['shard']}"
    action = labels.get("action") or folded
    return f"{base}.{action}" if action else base


def _label(spec: JobSpec) -> str:
    return label_of(labels_for(spec))


def labels_for(spec: JobSpec) -> dict[str, str]:
    """The job labels; a value over the HF limit refuses here, before any job is launched."""
    labels = {
        "campaign": spec.campaign,
        "run_key": spec.run_key,
        "job": spec.job,
        "variant": spec.variant,
        "stage": "-".join(spec.stages),
        "shard": f"{spec.shard[0]}of{spec.shard[1]}",
        "commit": spec.commit,
    }
    if spec.action_id:  # R8 is closed (Phase 0.2): the `action` label is accepted and carries the action id
        labels["action"] = spec.action_id
    _check_label_values(labels)
    return labels


def _check_label_values(labels: Mapping[str, str]) -> None:
    long = {k: len(v) for k, v in labels.items() if len(v) > LABEL_MAX}
    if long:
        raise ValueError(f"HF job label value(s) over {LABEL_MAX} characters: {long}")
    bad = {k: v for k, v in labels.items() if not LABEL_PATTERN.fullmatch(v)}
    if bad:
        raise ValueError(f"HF job label value(s) outside the pattern {LABEL_PATTERN.pattern}: {bad}")


def job_name(spec: JobSpec) -> str:
    """The explicit job name (`run_job(name=)`). Without one, HF derives a name from the command, and the
    launcher's bash command is far over the 100-character limit. The parts join with `--` (HF allows only
    `[a-zA-Z0-9_-]`). Readable when it fits; otherwise a readable prefix plus a digest of the full name, so
    two jobs never share a truncated name."""
    import hashlib

    full = "--".join((spec.run_key, spec.variant, "-".join(spec.stages), f"{spec.shard[0]}of{spec.shard[1]}"))
    if len(full) > LABEL_MAX:
        digest = hashlib.sha256(full.encode("utf-8")).hexdigest()[:16]
        full = f"{full[: LABEL_MAX - len(digest) - 1]}-{digest}"
    _check_label_values({"name": full})
    return full


def job_env(spec: JobSpec) -> dict[str, str]:
    from scripts._provenance import IMAGE_ENV, RUN_KEY_ENV
    from scripts.hf_jobs import image

    env = dict(DETERMINISTIC_VARS)
    env.update(
        {
            IMAGE_ENV: image.image_ref(),
            RUN_KEY_ENV: spec.run_key,
            "SILLY_KICKS_JOB_LABEL": _label(spec),
            "SILLY_KICKS_IN_HF_JOB": "1",  # in-job: an unrecorded dataset is an error, never a local guess
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
    )
    for d in spec.datasets:
        if d.access == "open":
            env["SILLY_KICKS_OPEN_DATA_REVISION"] = d.revision
    caps = [d.read_concurrency for d in spec.datasets if d.access == "mount" and d.read_concurrency]
    if caps:  # ONE job-wide gate over every mount: the smallest declared cap (`scripts._dataset_reads`)
        env[READ_CAP_ENV] = str(min(caps))
    from scripts.hf_jobs.datasets import store_settings
    from scripts.hf_jobs.phases import ACTION_KIND_ENV, PHASE_FILE_ENV

    env[STORE_READ_CAP_ENV] = str(store_settings().read_concurrency)  # the bucket-store read gate (S5.1)
    env[ACTION_KIND_ENV] = spec.action_kind  # "run" | "reduce" (E-4): a separate reduce job marks itself
    env[PHASE_FILE_ENV] = PHASE_FILE
    env.update(spec.extra_env or {})
    # CCTP-IMPL-27: the controller strips every output switch from its `--plan` token (`controller_plan_env`), so
    # the job must NOT set one either -- OPEN_DATA_REVISION aside -- or the two tokens diverge and a generation
    # never matches. Refuse a switch leaked through `extra_env` (the only non-OPEN_DATA source) before launch.
    from scripts.hf_jobs import env_contract
    from scripts.hf_jobs.closures import ClosureViolationError

    leaked = sorted(k for k in env_contract.OUTPUT if k != "SILLY_KICKS_OPEN_DATA_REVISION" and k in env)
    if leaked:
        raise ClosureViolationError(
            f"job_env sets output switch(es) {leaked} the controller cannot mirror (S5.9 / CCTP-IMPL-27): an "
            f"output switch must reach both the controller `--plan` and the job, which no channel does today"
        )
    return env


def _run_argv(spec: JobSpec, *, python: str = ".venv/bin/python") -> list[str]:
    argv = [
        python,
        "-m",
        spec.module,
        *spec.args,
        "--store",
        f"{BUCKET_MOUNT}/{spec.run_key}",
        "--data-root",
        DATA_MOUNT,
    ]
    argv += ["--stages", ",".join(spec.stages)]
    if not spec.shard_keys_path:  # an LPT part names its keys; `--shard` labels it only (Task 5)
        argv += ["--shard", f"{spec.shard[0]}/{spec.shard[1]}"]
    argv += ["--workers", str(spec.workers), "--inner-workers", str(spec.inner_workers)]
    if spec.cap_bytes:
        argv += ["--cap-bytes", str(spec.cap_bytes)]
    if spec.peak_rss_bytes:
        argv += ["--peak-rss-bytes", str(spec.peak_rss_bytes)]
    if spec.profile:
        argv += ["--profile", str(spec.profile)]
    if spec.escalate:  # the memory escalation job also runs units deferred by a memory failure
        argv += ["--escalate"]
    if spec.shard_keys_path:
        argv += ["--shard-keys", spec.shard_keys_path]
    if spec.read_through:  # ADR-129 canary: read allowed upstream generations from the main store mount
        argv += ["--read-through", spec.read_through]
        if spec.read_through_allow:
            argv += ["--read-through-allow", spec.read_through_allow]
    return argv


def job_command(spec: JobSpec) -> list[str]:
    """The in-job script (ADR-129 S5.12): fetch the exact commit into /work, CHECK the prebuilt environment
    (never install it), then run the stage. The environment is the digest-pinned image (`image.image_ref()`);
    `image.CHECK_COMMAND` verifies the fetched `uv.lock` matches it and installs nothing."""
    from scripts.hf_jobs import image

    q = shlex.quote
    env_py = f"{image.VENV}/bin/python"
    lines = [
        "set -euo pipefail",
        # Per-attempt id (S5.3): an HF retry runs a new script, so it gets a new id and a fresh set of per-job
        # files. The script-start epoch is captured here, before anything runs, and marked after the fetch.
        'export SILLY_KICKS_ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(od -An -N2 -tx1 /dev/urandom | tr -d " \\n")"',
        'SK_SCRIPT_START="$(date +%s.%N)"',
        "git init -q /work",
        "cd /work",
        f"git remote add origin {q(REPO_URL)}",
        f"git fetch -q --depth 1 origin {spec.commit}",
        "git checkout -q --detach FETCH_HEAD",
        f'test "$(git rev-parse HEAD)" = "{spec.commit}"',
        f'{env_py} -m scripts.hf_jobs.phases mark script_start "$SK_SCRIPT_START"',
        f"{env_py} -m scripts.hf_jobs.phases mark after_fetch",
        # CHECK the prebuilt environment against the fetched uv.lock; install NOTHING (S5.12).
        image.CHECK_COMMAND,
        f"{env_py} -m scripts.hf_jobs.phases mark after_env_check",
    ]
    run = _run_argv(spec, python=env_py)
    for d in spec.datasets:
        if d.access == "download":
            fetch = [
                env_py,
                "-m",
                "scripts.hf_jobs.datasets",
                "fetch",
                d.name,
                "--dest",
                d.path,
                "--module",
                spec.module,
                "--stages",
                ",".join(spec.stages),
                "--shard",
                f"{spec.shard[0]}/{spec.shard[1]}",
                *(["--profile", str(spec.profile)] if spec.profile else []),
                "--store",
                f"{BUCKET_MOUNT}/{spec.run_key}",
                "--data-root",
                DATA_MOUNT,
                "--",
                *spec.args,
            ]
            lines.append(" ".join(q(a) for a in fetch))
            # Mark the end of the dataset fetch (S7.2): startup excludes this data movement (CCTP-IMPL-04).
            lines.append(f"{env_py} -m scripts.hf_jobs.phases mark after_dataset_fetch")
    lines.append(" ".join(q(a) for a in run))
    return ["bash", "-c", "\n".join(lines)]


def volumes_for(spec: JobSpec) -> list[Any]:
    from huggingface_hub import Volume

    vols = [Volume(type="bucket", source=spec.bucket, mount_path=BUCKET_MOUNT, read_only=False)]
    for d in spec.datasets:
        if d.access == "mount":
            vols.append(Volume(type="dataset", source=d.repo, mount_path=d.path, revision=d.revision, read_only=True))
    return vols


def launch(spec: JobSpec, *, api: Any, token: str) -> str:
    """Start ONE job; returns its id. The token rides `secrets` (never the command line)."""
    from scripts.hf_jobs import image

    info = api.run_job(
        image=image.image_ref(),
        command=job_command(spec),
        env=job_env(spec),
        secrets={"HF_TOKEN": token},
        flavor=FLAVOR,
        timeout=spec.timeout_s,
        attempts=spec.attempts,
        labels=labels_for(spec),
        volumes=volumes_for(spec),
        name=job_name(spec),
    )
    return info.id


def jobs_for(api: Any, *, campaign: str, commit: str) -> list[Any]:
    return list(api.list_jobs(labels={"campaign": campaign, "commit": commit}))


def jobs_for_run_keys(api: Any, run_keys: Iterable[str]) -> list[Any]:
    """Every job on each run key, over ALL campaigns and commits that share the store (S5.3, r4 B R4-04): one
    `list_jobs(labels={"run_key": rk})` call per run key, de-duplicated by job id. Ownership and "running" are
    read from THIS list, so a concurrent campaign on the same store never double-runs an owned unit."""
    seen: dict[str, Any] = {}
    for rk in dict.fromkeys(run_keys):
        for info in api.list_jobs(labels={"run_key": rk}):
            seen.setdefault(info.id, info)
    return list(seen.values())
