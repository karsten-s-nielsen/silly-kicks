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
from collections.abc import Mapping
from typing import Any, Final

from scripts._dataset_reads import READ_CAP_ENV
from scripts._thread_pin import DETERMINISTIC_VARS

FLAVOR: Final = "cpu-upgrade"
IMAGE: Final = "python:3.12.14-bookworm@sha256:dbbe4ceb97851e2e5fa83798b239811f871cb743b259ba3563737349f6bcfaa0"
PYTHON_BIN: Final = "/usr/local/bin/python3.12"
UV_REQUIREMENT: Final = "uv==0.9.28 --hash=sha256:7b8460a2b624d8ab27cb293a2c9f2393f9efc4e36e0fb886a6c2360e23fb48be"
REPO_URL = "https://github.com/karsten-s-nielsen/silly-kicks"
BUCKET_MOUNT = "/scratch"  # the read-write bucket is mounted here inside every job
DATA_MOUNT: Final = "/data"
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
    extras: tuple[str, ...]
    profile: int | None = None
    shard_keys_path: str | None = None
    extra_env: Mapping[str, str] | None = None
    escalate: bool = False
    #: HFU T8 sharded reduce: a sub-reduce's shard index; the final reduce's partial count; a sharded run's
    #: skip of the inline reduce.
    subreduce_shard_id: int | None = None
    subreduce_n_shards: int | None = None
    from_partials: int | None = None
    no_inline_reduce: bool = False
    #: HFU T11 + reviewer B D1-IMPL-07: the inner_parallel pool CHILD cap, distinct from ``cap_bytes`` (the unit
    #: worker's). None for a non-inner stage.
    inner_cap_bytes: int | None = None


def label_of(labels: Mapping[str, str]) -> str:
    """The in-job label (`SILLY_KICKS_JOB_LABEL`, stamped on every unit ledger record) rebuilt from a job's
    HF labels; the ledger joins unit records to their job by it."""
    return f"{labels['job']}.{labels['variant']}.{labels['stage']}.{labels['shard']}"


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

    env = dict(DETERMINISTIC_VARS)
    env.update(
        {
            IMAGE_ENV: IMAGE,
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
    env.update(spec.extra_env or {})
    return env


def _run_argv(spec: JobSpec) -> list[str]:
    argv = [
        ".venv/bin/python",
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
    if spec.inner_cap_bytes:  # HFU T11: the inner_parallel pool child cap (distinct from the unit worker's)
        argv += ["--inner-cap-bytes", str(spec.inner_cap_bytes)]
    if spec.peak_rss_bytes:
        argv += ["--peak-rss-bytes", str(spec.peak_rss_bytes)]
    if spec.profile:
        argv += ["--profile", str(spec.profile)]
    if spec.escalate:  # the memory escalation job also runs units deferred by a memory failure
        argv += ["--escalate"]
    if spec.shard_keys_path:
        argv += ["--shard-keys", spec.shard_keys_path]
    if spec.from_partials is not None:  # HFU T8: the final reduce over N sub-reduce partials
        argv += ["--from-partials", str(spec.from_partials)]
    if spec.subreduce_shard_id is not None:  # HFU T8: a sub-reduce; the worker computes its range from DONE units
        argv += ["--subreduce", "--shard-id", str(spec.subreduce_shard_id), "--n-shards", str(spec.subreduce_n_shards)]
    if spec.no_inline_reduce:  # HFU T8: a sharded stage's run job skips the inline reduce
        argv += ["--no-inline-reduce"]
    return argv


def job_command(spec: JobSpec) -> list[str]:
    q = shlex.quote
    extras = "".join(f" --extra {q(e)}" for e in spec.extras)
    lines = [
        "set -euo pipefail",
        "git init -q /work",
        "cd /work",
        f"git remote add origin {q(REPO_URL)}",
        f"git fetch -q --depth 1 origin {spec.commit}",
        "git checkout -q --detach FETCH_HEAD",
        f'test "$(git rev-parse HEAD)" = "{spec.commit}"',
        f"printf '%s\\n' {q(UV_REQUIREMENT)} > /tmp/uv-req.txt",
        f"{PYTHON_BIN} -m pip install -q --disable-pip-version-check --require-hashes -r /tmp/uv-req.txt",
        f"uv sync --frozen --python {PYTHON_BIN} --extra train{extras} --group hfjobs",
    ]
    run = _run_argv(spec)
    for d in spec.datasets:
        if d.access == "download":
            fetch = [
                ".venv/bin/python",
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
    info = api.run_job(
        image=IMAGE,
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
