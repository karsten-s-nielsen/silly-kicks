"""Launch silly-kicks trainers as ephemeral HF Jobs, one job per committed variant.

Pure command builders (golden-tested) + thin orchestration over the ``hf`` CLI (injected ``runner`` so
tests never touch HF). The invocation shapes here are **execution-verified** (2026-10-13/14):

* trainers run as ``python -m scripts.train_<name>`` -- the file form ``python scripts/train_<name>.py``
  raises ``ModuleNotFoundError: No module named 'scripts'`` (no ``scripts/__init__.py``);
* the HF launch shape is ``hf jobs run <hf-opts> python:3.12 -- bash -lc "<cmd>"`` -- the ``--`` marker
  before ``bash`` is mandatory on BOTH CLI v1.7.1 and v2.2.0 (``--dry-run`` confirms v2.2.0 eats ``-lc``
  without it), ``--name`` is NOT passed, the secret rides ``--secrets``;
* **storage is a read-write bucket mount, decoupled from the flavor's fixed ephemeral disk.** Each job
  mounts ``hf://buckets/<ns>/<bucket>`` at ``/scratch`` and writes its output + the corpus-driver shards
  there. ``--attempts N`` makes the platform auto-retry a crashed/timed-out job; because the shard-root
  persists in the bucket, the trainer's ``for_each`` (ADR-052) resumes from the shards it already wrote
  instead of restarting -- the DGX resume behaviour, on HF (HF docs, "Train Models on Jobs").

Each variant is its own job: bounded timeout, per-variant head-to-head floor, and a crash never loses a
finished variant. Outputs live in the bucket; retrieval is a local ``hf cp`` (no in-job upload step, so
no ``huggingface_hub`` import at upload time -- it is still installed for the in-job ``hf download`` of
the staged corpus).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time

from scripts.hf_jobs.registry import TrainerSpec, VariantBuild

REPO_URL = "https://github.com/karsten-s-nielsen/silly-kicks"
BUCKET_MOUNT = "/scratch"  # the read-write bucket is mounted here inside every job


def _out_dir(run_key: str, name: str, variant: str) -> str:
    """This variant's output root inside the mounted bucket (persists across attempts => resume)."""
    return f"{BUCKET_MOUNT}/{run_key}/{name}/{variant}"


def _fresh_dir(out_dir: str, build: VariantBuild) -> str:
    """Where the head-to-head loads this run's fresh weights (``""`` fresh_subdir => the output dir itself)."""
    return f"{out_dir}/{build.fresh_subdir}" if build.fresh_subdir else out_dir


def _corpus_steps(
    spec: TrainerSpec, corpus_repo: str | None, sb360_repo: str | None, out_dir: str
) -> tuple[list[str], str, str]:
    """Return (prep_steps, data_args, headtohead_corpus_arg) for the trainer's source mode.

    The flat TC3 corpus is downloaded to ``/tmp/corpus`` (fits the ephemeral disk); only the trainer
    OUTPUT + shards go to the bucket (``out_dir``), which is what needs to persist for resume. The
    head-to-head owner eval always reads the flat TC3 corpus (``build_eval._owner_shards``).
    """
    if spec.source == "public":
        return [], "", ""
    if corpus_repo is None:
        raise ValueError(f"{spec.name}: owner-tier trainer needs corpus_repo (flat TC3 for the head-to-head eval)")
    steps = [f"hf download {corpus_repo} --repo-type dataset --local-dir /tmp/corpus"]
    corpus_arg = " --corpus-dir /tmp/corpus"
    if spec.source == "owner_flat":
        gen = '"$(ls -d /tmp/corpus/shards/*/ | head -1)"'
        data_args = f"--data-dir {gen} --actions-dir /tmp/corpus/_actions --home-teams /tmp/corpus/home_teams.json"
        return steps, data_args, corpus_arg
    if spec.source == "owner_pergame":
        steps.append("python -m scripts.hf_jobs.repackage /tmp/corpus /tmp/pergame")
        return steps, "--data-dir /tmp/pergame", corpus_arg
    if spec.source == "owner_sb360":
        if sb360_repo is None:
            raise ValueError(f"{spec.name}: owner_sb360 trainer needs sb360_repo (staged SB360 cache + manifest)")
        steps.append(f"hf download {sb360_repo} --repo-type dataset --local-dir /tmp/sb360")
        steps.append("export SILLY_KICKS_PINING_MANIFEST_DIR=/tmp/sb360/manifest")
        # extraction scratch goes to the bucket too, so a retried receiver run resumes its extraction.
        data_args = (
            f"--shard-root {out_dir}/scratch --cache-dir /tmp/sb360/cache --match-ids-json /tmp/sb360/match_ids.json"
        )
        return steps, data_args, corpus_arg
    raise ValueError(f"{spec.name}: unknown source {spec.source!r}")


def _trainer_cmd(spec: TrainerSpec, build: VariantBuild, data_args: str, out_dir: str) -> str:
    args = [*spec.base_args, *build.run_args]
    if data_args:
        args += data_args.split(" ")
    args += [spec.output_flag, out_dir]
    return f"python -m {spec.module} " + " ".join(a for a in args if a)


def build_inner_command(
    spec: TrainerSpec,
    build: VariantBuild,
    *,
    tag: str,
    run_key: str,
    corpus_repo: str | None = None,
    sb360_repo: str | None = None,
) -> str:
    """The ``bash -lc`` payload for ONE variant job. ``tag`` is any git ref; the launcher passes the pushed
    Commit-1 ref so ``scripts/hf_jobs/**`` and the committed weights exist in the clone. Output + shards
    go to the mounted bucket at ``out_dir`` (persist + resume); there is no in-job upload."""
    out_dir = _out_dir(run_key, spec.name, build.variant)
    prep, data_args, corpus_arg = _corpus_steps(spec, corpus_repo, sb360_repo, out_dir)
    committed_dir = f"silly_kicks/{build.committed_path}"
    parts = [
        "set -e",
        "pip install -q uv",
        f"git clone --depth 1 --branch {tag} {REPO_URL} /work",
        "cd /work",
        # huggingface_hub: the in-job `hf download` of the staged corpus needs it (.[train] does not pull it).
        "uv pip install --system -q -e '.[train]'" + "".join(f" {e}" for e in spec.extras) + " huggingface_hub",
        *prep,
        f"mkdir -p {out_dir}",
        _trainer_cmd(spec, build, data_args, out_dir),
        f"python -m scripts.hf_jobs.headtohead {spec.name} --committed-dir {committed_dir} "
        f"--fresh-dir {_fresh_dir(out_dir, build)} --metrics-json {out_dir}/metrics.json{corpus_arg}",
    ]
    return " && ".join(parts)


def build_argv(spec: TrainerSpec, inner: str, *, bucket: str, attempts: int) -> list[str]:
    # `--secrets HF_TOKEN` (no value) forwards the launcher's HF_TOKEN env as a job secret WITHOUT putting
    # the token in the process listing / shell history (the CLI warns against `--secrets KEY=value`).
    # launch_variant_job puts the token in the env first.
    return [
        "hf",
        "jobs",
        "run",
        "--flavor",
        spec.flavor,
        "--secrets",
        "HF_TOKEN",
        "--timeout",
        spec.timeout,
        "--attempts",
        str(attempts),
        "-v",
        f"hf://buckets/{bucket}:{BUCKET_MOUNT}",
        "-d",
        "python:3.12",
        "--",
        "bash",
        "-lc",
        inner,
    ]


def subprocess_runner(argv: list[str]) -> tuple[str, int]:
    p = subprocess.run(argv, capture_output=True, text=True)  # noqa: S603 -- fixed hf CLI argv, never user input
    return (p.stdout + p.stderr), p.returncode


def launch_variant_job(
    spec: TrainerSpec,
    build: VariantBuild,
    *,
    tag: str,
    run_key: str,
    corpus_repo: str | None = None,
    sb360_repo: str | None = None,
    token: str,
    bucket: str,
    attempts: int,
    runner=subprocess_runner,
) -> str:
    inner = build_inner_command(spec, build, tag=tag, run_key=run_key, corpus_repo=corpus_repo, sb360_repo=sb360_repo)
    os.environ["HF_TOKEN"] = token  # forwarded as a job secret via `--secrets HF_TOKEN` (not on the cmdline)
    out, code = runner(build_argv(spec, inner, bucket=bucket, attempts=attempts))
    if code != 0:
        raise RuntimeError(f"{spec.name}/{build.variant}: hf jobs run failed: {out}")
    # v2.2.0 `-d` prints `id=<hex> ... url=.../jobs/<ns>/<hex>`; v1.7.1 printed "Job started with ID: <hex>".
    # Prefer the namespaced `<ns>/<hex>` from the url (what inspect/cancel/logs take), else the bare id.
    m = re.search(r"jobs/([^/\s]+/[0-9a-f]{6,})", out) or re.search(r"(?:ID:\s*|id=)([0-9a-f]{6,})", out)
    if not m:
        raise RuntimeError(f"{spec.name}/{build.variant}: no job id in: {out}")
    return m.group(1)


def launch_all_variants(
    spec: TrainerSpec,
    *,
    tag: str,
    run_key: str,
    corpus_repo: str | None = None,
    sb360_repo: str | None = None,
    token: str,
    bucket: str,
    attempts: int,
    runner=subprocess_runner,
) -> dict[str, str]:
    """Launch one job per committed variant; return ``{variant: job_id}``."""
    return {
        build.variant: launch_variant_job(
            spec,
            build,
            tag=tag,
            run_key=run_key,
            corpus_repo=corpus_repo,
            sb360_repo=sb360_repo,
            token=token,
            bucket=bucket,
            attempts=attempts,
            runner=runner,
        )
        for build in spec.builds
    }


_TERMINAL = {"COMPLETED", "ERROR", "CANCELED"}


def _terminal_stage(out: str) -> str | None:
    """The job's terminal stage from ``hf jobs inspect --format json``, or None if still running.

    Reads ``status.stage`` EXACTLY -- never a substring of the raw output (a failed job's ``status.message``
    contains "Error", and ``command`` echoes the bash payload, so a substring scan would false-positive)."""
    try:
        data = json.loads(out)
    except (json.JSONDecodeError, TypeError):
        return None
    for rec in data if isinstance(data, list) else [data]:
        status = rec.get("status") if isinstance(rec, dict) else None
        stage = status.get("stage") if isinstance(status, dict) else status
        if isinstance(stage, str) and stage.upper() in _TERMINAL:
            return stage.upper()
    return None


def poll_until_terminal(job_id, *, runner=subprocess_runner, sleep=time.sleep, interval=20, max_polls=2000) -> str:
    for _ in range(max_polls):
        out, _ = runner(["hf", "jobs", "inspect", job_id, "--format", "json"])
        stage = _terminal_stage(out)
        if stage is not None:
            return stage
        sleep(interval)
    raise TimeoutError(f"{job_id}: never reached terminal")


def retrieve_variant(bucket, run_key, name, variant, dest, *, runner=subprocess_runner):
    """Copy a finished variant's output (weights + metrics.json) out of the bucket to a local dir."""
    src = f"hf://buckets/{bucket}/{run_key}/{name}/{variant}"
    out, code = runner(["hf", "cp", src, str(dest), "--recursive"])
    if code != 0:
        raise RuntimeError(f"retrieve {src} failed: {out}")
    return dest
