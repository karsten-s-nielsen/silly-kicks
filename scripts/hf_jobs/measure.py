"""Read-only HF measurement jobs (spec r4.4; plan R3-8 S0b): a probe script at production shape on unpushed code.

A measurement job runs a probe in the campaign image, on the campaign flavor, in the campaign's `uv sync --frozen`
environment, with the campaign's dataset mounts -- on code that is not pushed yet. It checks out the pushed base
commit and applies the working-tree patch, which travels INSIDE the job request: gzip + base64, split over
environment variables (one ``bash -c`` argument is capped at 128 KiB by Linux), and its sha256 is checked before
``git apply``. Nothing is staged in any bucket.

Every bucket and dataset is mounted READ-ONLY and the job gets no secrets, so a measurement job cannot write a
campaign store or produce an artifact: by mechanism, not by convention. A measurement that must write (rung 3)
uses the canary mode (``--canary``, plan R3-8 S3 item 13): it adds ONE read-write mount, the separate canary bucket
``silly-kicks/hf-refit-canary`` (owner ruling 2026-10-30, created that day) at ``/canary``. `check_mounts` asserts,
where the mounts are built, that no other mount is writable, so a measurement job cannot write a campaign bucket.

    python -m scripts.hf_jobs measure --run-id ID --probe PROBE.py [--base COMMIT] [--patch [--patch-path P ...]]
        [--bucket NAME] [--dataset NAME ...] [--canary] [--timeout SECONDS] [-- PROBE ARGS ...]
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from scripts._thread_pin import DETERMINISTIC_VARS
from scripts.hf_jobs.launch import (
    BUCKET_MOUNT,
    FLAVOR,
    IMAGE,
    PYTHON_BIN,
    REPO_URL,
    UV_REQUIREMENT,
    DatasetMount,
    dataset_env,
    dataset_volumes,
)

#: The environment variables that carry the patch: ``MEASURE_PATCH_0`` ... in order.
PATCH_ENV_PREFIX: Final = "MEASURE_PATCH_"
#: One environment string's payload, well under Linux's 128 KiB cap on a single argument or environment string.
PATCH_CHUNK: Final = 96 * 1024
#: The heredoc delimiter of the probe source; a probe holding it as a line is refused.
PROBE_EOF: Final = "PROBE_EOF"
#: The job container's scratch directory: a path inside the job, not on this machine.
JOB_TMP: Final = "/tmp"  # noqa: S108 -- inside the job container
#: The ONE bucket a measurement job may write (owner ruling 2026-10-30), and where the canary mode mounts it.
CANARY_BUCKET: Final = "silly-kicks/hf-refit-canary"
CANARY_MOUNT: Final = "/canary"
_GIT: Final = shutil.which("git") or "git"


def _git(*args: str, env: Mapping[str, str] | None = None) -> bytes:
    """Run git with a fixed argv (no shell) and return its stdout."""
    return subprocess.run([_GIT, *args], env=env, check=True, capture_output=True).stdout  # noqa: S603 -- fixed argv


def worktree_patch(repo: Path, base: str, *, paths: Sequence[str] = ()) -> bytes:
    """The binary diff from ``base`` to the working tree of ``repo``, untracked (not ignored) files included;
    only under ``paths`` when given (the HF Jobs API refuses a large request, so a job carries what it runs).

    Built in a temporary index, so the repository's own index is untouched.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        _git("-C", str(repo), "read-tree", base, env=env)
        _git("-C", str(repo), "add", "-A", env=env)
        return _git("-C", str(repo), "diff", "--cached", "--binary", base, "--", *paths, env=env)


def patch_env(patch: bytes) -> tuple[dict[str, str], str]:
    """The patch as environment variables (gzip + base64, in chunks of at most `PATCH_CHUNK` characters) and its
    sha256. ``mtime=0`` keeps the encoding deterministic."""
    payload = base64.b64encode(gzip.compress(patch, compresslevel=9, mtime=0)).decode("ascii")
    chunks = [payload[i : i + PATCH_CHUNK] for i in range(0, len(payload), PATCH_CHUNK)] or [""]
    return {f"{PATCH_ENV_PREFIX}{k}": chunk for k, chunk in enumerate(chunks)}, hashlib.sha256(patch).hexdigest()


def _stamp(label: str) -> str:
    return f'echo "PRELUDE {label} $(date -u +%s.%N)"'


def patch_lines(*, n_chunks: int, digest: str, tmp: str = JOB_TMP) -> list[str]:
    """The shell lines that rebuild the patch from its ``n_chunks`` environment variables, refuse it unless its
    sha256 is ``digest`` (``set -e`` stops the job), and apply it to the checkout in the working directory."""
    parts = " ".join(f'"${{{PATCH_ENV_PREFIX}{k}}}"' for k in range(n_chunks))
    return [
        f"printf '%s' {parts} | base64 -d | gunzip > {tmp}/tree.patch",
        f'test "$(sha256sum {tmp}/tree.patch | cut -c1-64)" = {shlex.quote(digest)}',
        f"git apply --binary {tmp}/tree.patch",
        'echo "PATCH_APPLIED $(git status --porcelain | wc -l) paths"',
    ]


def measure_command(
    *, base: str, probe: str, n_chunks: int, digest: str | None, probe_args: Sequence[str] = ()
) -> list[str]:
    """The job command: check out ``base``; when ``digest`` is given, rebuild the patch from its ``n_chunks``
    environment variables, check its sha256 and apply it; build the campaign environment; run the probe."""
    if any(line == PROBE_EOF for line in probe.splitlines()):
        raise ValueError(f"the probe holds the heredoc delimiter {PROBE_EOF!r} as a line")
    q = shlex.quote
    lines = [
        "set -euo pipefail",
        _stamp("container_start"),
        "git init -q /work",
        "cd /work",
        f"git remote add origin {q(REPO_URL)}",
        f"git fetch -q --depth 1 origin {q(base)}",
        "git checkout -q --detach FETCH_HEAD",
        f'test "$(git rev-parse HEAD)" = {q(base)}',
    ]
    if digest is not None:
        lines += patch_lines(n_chunks=n_chunks, digest=digest)
    lines += [
        _stamp("checkout_done"),
        f"printf '%s\\n' {q(UV_REQUIREMENT)} > {JOB_TMP}/uv-req.txt",
        f"{PYTHON_BIN} -m pip install -q --disable-pip-version-check --require-hashes -r {JOB_TMP}/uv-req.txt",
        f"uv sync --frozen --python {PYTHON_BIN} --extra train --group hfjobs",
        _stamp("env_synced"),
        f"cat > {JOB_TMP}/probe.py <<'{PROBE_EOF}'\n{probe}\n{PROBE_EOF}",
        " ".join([".venv/bin/python", f"{JOB_TMP}/probe.py", *(q(a) for a in probe_args)]),
        _stamp("probe_done"),
    ]
    return ["bash", "-c", "\n".join(lines)]


def dataset_mounts(names: Sequence[str]) -> list[DatasetMount]:
    """The named datasets as the campaign declares and locks them (`datasets.toml`, `datasets.lock.json`)."""
    from scripts.hf_jobs.datasets import declared, locked

    out = []
    for name in names:
        d = declared()[name]
        out.append(DatasetMount(name, d.repo, locked(name)["revision"], d.access, d.read_concurrency))
    return out


def measure_volumes(*, bucket: str | None, datasets: Sequence[DatasetMount], canary: bool = False) -> list[Any]:
    """The job's mounts: the campaign bucket at the campaign mount path and every mounted dataset at its pinned
    revision under the campaign data root (`launch.dataset_volumes`, as a campaign job), all READ-ONLY; in the
    canary mode also the canary bucket, read-write, at `CANARY_MOUNT`."""
    from huggingface_hub import Volume

    vols = []
    if bucket is not None:
        vols.append(Volume(type="bucket", source=bucket, mount_path=BUCKET_MOUNT, read_only=True))
    vols += dataset_volumes(datasets)
    if canary:
        vols.append(Volume(type="bucket", source=CANARY_BUCKET, mount_path=CANARY_MOUNT, read_only=False))
    check_mounts(vols)
    return vols


def check_mounts(vols: Sequence[Any]) -> None:
    """The invariant this module exists for, asserted where the mounts are built: a measurement job writes ONLY the
    canary bucket, mounted at `CANARY_MOUNT`; every other mount is read-only."""
    for v in vols:
        if not v.read_only and (v.source != CANARY_BUCKET or v.mount_path != CANARY_MOUNT):
            raise AssertionError(
                f"a measurement job writes only the canary bucket {CANARY_BUCKET} at {CANARY_MOUNT}; refused a "
                f"writable mount of {v.source} at {v.mount_path}"
            )


def measure_env(datasets: Sequence[DatasetMount]) -> dict[str, str]:
    """The job's environment: the campaign's deterministic thread pins, the in-job marker, and what the datasets
    put there (an `open` dataset's pinned revision, the mounts' read gate; `launch.dataset_env`)."""
    env = dict(DETERMINISTIC_VARS)
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["SILLY_KICKS_IN_HF_JOB"] = "1"  # in-job: an unrecorded dataset is an error, never a local guess
    env.update(dataset_env(datasets))
    return env


def launch_measurement(
    api: Any,
    *,
    run_id: str,
    base: str,
    probe: str,
    patch: bytes | None,
    bucket: str | None,
    datasets: Sequence[DatasetMount],
    timeout_s: int,
    probe_args: Sequence[str] = (),
    canary: bool = False,
) -> str:
    """Start ONE measurement job and return its id. No ``secrets``: the job cannot authenticate to write; in the
    canary mode it writes only through the canary bucket's mount."""
    env = measure_env(datasets)
    digest = None
    n_chunks = 0
    if patch is not None:
        chunks, digest = patch_env(patch)
        env.update(chunks)
        n_chunks = len(chunks)
    labels = {"campaign": "measure", "commit": base, "run_id": run_id}
    if digest is not None:
        labels["patch"] = digest[:16]
    if canary:
        labels["canary"] = "1"
    info = api.run_job(
        image=IMAGE,
        command=measure_command(base=base, probe=probe, n_chunks=n_chunks, digest=digest, probe_args=probe_args),
        env=env,
        flavor=FLAVOR,
        timeout=timeout_s,
        labels=labels,
        volumes=measure_volumes(bucket=bucket, datasets=datasets, canary=canary),
        name=f"measure-{run_id}",
    )
    return info.id


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.hf_jobs measure", description="Launch ONE read-only HF measurement job."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--probe", required=True, type=Path, help="the probe script, run as /tmp/probe.py")
    parser.add_argument("--base", default=None, help="the pushed base commit (default: HEAD)")
    parser.add_argument("--patch", action="store_true", help="carry the working-tree patch against --base")
    parser.add_argument("--patch-path", action="append", default=[], help="carry only this path (repeatable)")
    parser.add_argument("--bucket", default=None, help="mount this bucket READ-ONLY at the campaign mount path")
    parser.add_argument("--dataset", action="append", default=[], help="a declared dataset, as a campaign job reads it")
    parser.add_argument(
        "--canary", action="store_true", help=f"also mount {CANARY_BUCKET} READ-WRITE at {CANARY_MOUNT} (rung 3)"
    )
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("probe_args", nargs="*")
    args = parser.parse_args(argv)
    repo = Path.cwd()
    base = args.base or _git("rev-parse", "HEAD").decode("ascii").strip()
    from huggingface_hub import HfApi

    job = launch_measurement(
        HfApi(),
        run_id=args.run_id,
        base=base,
        probe=args.probe.read_text(encoding="utf-8"),
        patch=worktree_patch(repo, base, paths=args.patch_path) if args.patch else None,
        bucket=args.bucket,
        datasets=dataset_mounts(args.dataset),
        timeout_s=args.timeout,
        probe_args=args.probe_args,
        canary=args.canary,
    )
    print("JOB_ID", job)
    return 0
