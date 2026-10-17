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

import base64
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


def _corpus_steps(spec: TrainerSpec, corpus_repo: str | None, out_dir: str) -> tuple[list[str], str, str]:
    """Return (prep_steps, data_args, headtohead_corpus_arg) for the trainer's source mode.

    Every owner source reads a STAGED NEUTRAL parquet dataset (``corpus_repo`` -> ``/tmp/corpus``); the
    in-job code is AGNOSTIC of how it was built (ADR-123: no pining, no manifest, no ``--providers``). Only
    the trainer OUTPUT + shards go to the bucket (``out_dir``), which is what persists for resume. The
    head-to-head owner eval always reads the flat corpus (``build_eval._owner_shards``).
    """
    if spec.source == "public":
        return [], "", ""
    if corpus_repo is None:
        raise ValueError(f"{spec.name}: owner-tier trainer needs corpus_repo (staged flat corpus for training + eval)")
    steps = [f"hf download {corpus_repo} --repo-type dataset --local-dir /tmp/corpus"]
    corpus_arg = " --corpus-dir /tmp/corpus"
    if spec.source == "owner_flat":
        gen = '"$(ls -d /tmp/corpus/shards/*/ | head -1)"'
        data_args = f"--data-dir {gen} --actions-dir /tmp/corpus/_actions --home-teams /tmp/corpus/home_teams.json"
        return steps, data_args, corpus_arg
    if spec.source == "owner_pergame":
        # repackage emits {game}/{frames,shots,actions}.parquet (Task 3: actions.parquet is what xcross's
        # _iter_matches_from_dir reads); no in-job symlink workaround needed.
        steps.append("python -m scripts.hf_jobs.repackage /tmp/corpus /tmp/pergame")
        return steps, "--data-dir /tmp/pergame", corpus_arg
    if spec.source == "owner_sb360":
        # ADR-123 AGNOSTIC: receiver reads the STAGED flat per-match frames + _actions under /tmp/corpus via
        # --data-dir (the head-to-head eval shape). --shard-root is the extraction scratch in the bucket (so a
        # retried run resumes). NO pining manifest, NO --cache-dir, NO --match-ids-json -- staging is off-job.
        return steps, f"--data-dir /tmp/corpus --shard-root {out_dir}/scratch", corpus_arg
    raise ValueError(f"{spec.name}: unknown source {spec.source!r}")


def _trainer_cmd(spec: TrainerSpec, build: VariantBuild, data_args: str, out_dir: str, extra: str = "") -> str:
    args = [*spec.base_args, *build.run_args]
    if data_args:
        args += data_args.split(" ")
    if extra:
        args += extra.split(" ")
    args += [spec.output_flag, out_dir]
    return f"python -m {spec.module} " + " ".join(a for a in args if a)


def _phase_flags(
    spec: TrainerSpec, build: VariantBuild, phase: str, out_dir: str, *, shard=None, fold=None, study=None
):
    """The extra trainer flags for one map/reduce phase (ADR-123). ``full`` = the single-job serial run."""
    study_root = f"{out_dir}/{build.fresh_subdir}/studies"  # xshot/xcross persist studies under the art dir
    if phase == "extract":
        return f"--shard-index {shard}"
    if phase == "build_cache":
        return "--build-cache"
    if phase == "fold":
        return f"--fold {fold}"
    if phase == "prep":
        return "--prep-only"
    if phase == "study":
        return f"--study {study} --shard-root {study_root}"
    if phase == "reduce":
        if spec.source == "owner_pergame":  # xshot/xcross: reduce the fanned studies
            return f"--assemble --shard-root {study_root}"
        if spec.name in ("ghost_gk", "ghost_outfield"):
            return "--assemble"
        return ""  # receiver: the reduce is a plain no-shard-index run (reconcile all shards + fit)
    return ""  # full


def build_phase_command(
    spec: TrainerSpec,
    build: VariantBuild,
    *,
    tag: str,
    run_key: str,
    corpus_repo: str | None = None,
    probe_repo: str | None = None,
    phase: str = "full",
    shard=None,
    fold=None,
    study=None,
    cv_folds=None,
    arms_json: str | None = None,
) -> str:
    """The ``bash -lc`` payload for ONE phase of ONE variant (ADR-123 map/reduce). ``phase="full"`` is the
    single-job serial run (train + head-to-head floor). ``extract``/``build_cache``/``fold``/``prep``/``study``
    run only their slice and persist to the bucket; ``reduce`` finishes + runs the floor. ``tag`` is the git
    ref cloned (branch tip). Output + shards persist in the bucket (resume); no in-job upload, no pining.

    ``cv_folds`` (ghost): threaded as ``--cv-folds`` so the trainer's internal CV split has EXACTLY as many
    folds as the orchestrator fans (``n_folds``) -- otherwise ``assemble`` expects ``cv_folds`` fold sidecars
    but only ``n_folds`` were written (a FileNotFoundError in the reduce). The fold count must be coupled."""
    out_dir = _out_dir(run_key, spec.name, build.variant)
    prep, data_args, corpus_arg = _corpus_steps(spec, corpus_repo, out_dir)
    extra = ""
    if spec.probe:
        if probe_repo is None:
            raise ValueError(f"{spec.name}: probe trainer needs probe_repo (staged per-game probe dataset)")
        prep = [*prep, f"hf download {probe_repo} --repo-type dataset --local-dir /tmp/probe"]
        extra = "--probe-data-dir /tmp/probe"
    # ADR-123 Amendment B: a visibility-gated trainer gets the public/owner policy as the neutral --arms
    # parameter, computed OFF-HF by the launcher and embedded here as pre-written data (base64 so it rides
    # the `&&` chain without heredoc line-terminator issues). The job never computes the policy. Threaded
    # only on the phases that run the trainer's whole-corpus extraction+persist (the per-shard/fold/study
    # workers read the already-persisted arms from the shard root).
    if spec.visibility_gated and phase in ("full", "prep"):
        if arms_json is None:
            raise ValueError(f"{spec.name}: visibility-gated trainer needs arms_json (the launcher's --arms policy)")
        arms_b64 = base64.b64encode(arms_json.encode("utf-8")).decode("ascii")
        prep = [*prep, f"echo {arms_b64} | base64 -d > /tmp/arms.json"]
        extra = f"{extra} --arms /tmp/arms.json".strip()
    extra = f"{extra} {_phase_flags(spec, build, phase, out_dir, shard=shard, fold=fold, study=study)}".strip()
    if cv_folds is not None and spec.name in ("ghost_gk", "ghost_outfield"):
        extra = f"{extra} --cv-folds {cv_folds}".strip()
    # match_outcome's for_each (ADR-052) defaults its shard-root to ephemeral /tmp, so a --attempts retry
    # on the 3961-match --all-competitions load re-extracts from zero. Point it at the bucket to RESUME.
    if spec.name == "match_outcome_dependence" and phase == "full":
        extra = f"{extra} --shard-root {out_dir}/shards".strip()
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
        _trainer_cmd(spec, build, data_args, out_dir, extra),
    ]
    if phase in ("full", "reduce"):
        parts.append(
            f"python -m scripts.hf_jobs.headtohead {spec.name} --committed-dir {committed_dir} "
            f"--fresh-dir {_fresh_dir(out_dir, build)} --feature-set {build.feature_set} "
            f"--metrics-json {out_dir}/metrics.json{corpus_arg}"
        )
    return " && ".join(parts)


def build_inner_command(
    spec: TrainerSpec,
    build: VariantBuild,
    *,
    tag: str,
    run_key: str,
    corpus_repo: str | None = None,
    probe_repo: str | None = None,
    arms_json: str | None = None,
) -> str:
    """The single-job (``full`` phase) ``bash -lc`` payload: train + head-to-head floor. Thin wrapper over
    :func:`build_phase_command` kept for the non-fanned path + its golden tests."""
    return build_phase_command(
        spec,
        build,
        tag=tag,
        run_key=run_key,
        corpus_repo=corpus_repo,
        probe_repo=probe_repo,
        phase="full",
        arms_json=arms_json,
    )


def corpus_flat_stems(corpus_repo: str, *, api=None) -> list[str]:
    """OFF-HF: the staged flat corpus's shard filename stems ``{provider}__{game}`` (for ``compute_arms``).

    Lists ``shards/<gen>/<stem>.parquet`` in the dataset repo via the Hub API (local, launcher-side).
    """
    from huggingface_hub import HfApi

    api = api or HfApi()
    files = api.list_repo_files(corpus_repo, repo_type="dataset")
    return sorted(
        {f.split("/")[-1][: -len(".parquet")] for f in files if f.startswith("shards/") and f.endswith(".parquet")}
    )


def arms_payload(corpus_repo: str, *, api=None) -> str:
    """The ``--arms`` JSON string for a staged corpus, computed OFF-HF from the in-repo public/owner registry."""
    from scripts.hf_jobs.arm_policy import compute_arms

    return json.dumps(compute_arms(corpus_flat_stems(corpus_repo, api=api)))


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
        # `:rw` is REQUIRED on hf CLI v2.2: a bucket mounts READ-ONLY by default, so the in-job
        # shard/output writes silently fail without it (the whole resume/persist contract depends on it).
        f"hf://buckets/{bucket}:{BUCKET_MOUNT}:rw",
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
    probe_repo: str | None = None,
    token: str,
    bucket: str,
    attempts: int,
    runner=subprocess_runner,
) -> str:
    # ADR-123 Amendment B: a visibility-gated trainer (xshot/xcross) run as ONE `full` job needs the
    # public/owner policy threaded as --arms, computed OFF-HF from the staged corpus (mirrors run_pipeline).
    # Without it build_phase_command raises; the single-job path otherwise could not re-fit these faithfully.
    arms_json = arms_payload(corpus_repo) if (spec.visibility_gated and corpus_repo) else None
    inner = build_inner_command(
        spec, build, tag=tag, run_key=run_key, corpus_repo=corpus_repo, probe_repo=probe_repo, arms_json=arms_json
    )
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
    probe_repo: str | None = None,
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
            probe_repo=probe_repo,
            token=token,
            bucket=bucket,
            attempts=attempts,
            runner=runner,
        )
        for build in spec.builds
    }


_TERMINAL = {"COMPLETED", "ERROR", "CANCELED"}


_STAGE_RE = re.compile(r"""['"]stage['"]\s*:\s*['"](\w+)['"]""")


def _terminal_stage(out: str) -> str | None:
    """The job's terminal stage from ``hf jobs inspect`` output, or None if still running / absent.

    hf CLI v2.2's ``inspect`` has no ``--format json``; it prints a TSV whose ``status`` column is a Python
    repr like ``{'stage': 'RUNNING'}``. We regex the ``stage`` field SPECIFICALLY -- never a substring of
    ``status.message`` (a failed job's message contains "Error") or ``command`` (echoes the bash payload),
    so a message substring cannot false-positive. JSON (``"stage": "X"``) is matched by the same regex."""
    for m in _STAGE_RE.finditer(out or ""):
        stage = m.group(1).upper()
        if stage in _TERMINAL:
            return stage
    return None


def poll_until_terminal(job_id, *, runner=subprocess_runner, sleep=time.sleep, interval=20, max_polls=2000) -> str:
    for _ in range(max_polls):
        out, _ = runner(["hf", "jobs", "inspect", job_id])  # v2.2: `inspect` has no --format json
        stage = _terminal_stage(out)
        if stage is not None:
            return stage
        sleep(interval)
    raise TimeoutError(f"{job_id}: never reached terminal")


def retrieve_variant(bucket, run_key, name, variant, dest, *, runner=subprocess_runner):
    """Copy a finished variant's output (weights + metrics.json) out of the bucket to a local dir."""
    src = f"hf://buckets/{bucket}/{run_key}/{name}/{variant}"
    # hf CLI v2.2: `hf cp` is single-file; a recursive directory pull is `hf buckets sync`.
    out, code = runner(["hf", "buckets", "sync", src, str(dest)])
    if code != 0:
        raise RuntimeError(f"retrieve {src} failed: {out}")
    return dest


# --- Staged map/reduce orchestrator (ADR-123) ------------------------------------------------------
# A trainer's phases run as separate cpu-upgrade jobs over the shared bucket; each stage's jobs launch in
# parallel, the orchestrator waits for all to reach terminal, then launches the next stage. HF Jobs has no
# native DAG, so the sequencing is here. The injected ``runner`` is the HF CLI (``subprocess_runner``) or a
# local-subprocess runner (``make_local_runner``) -- the SAME pipeline drives the DGX e2e + fallback.
_PHASE_SEQUENCES = {
    "ghost_gk": ["extract", "build_cache", "fold", "reduce"],
    "ghost_outfield": ["extract", "fold", "reduce"],  # no build_cache (no whole-corpus feature cache)
    "xshot_occurrence": ["extract", "prep", "study", "reduce"],
    "xcross_attempt": ["extract", "prep", "study", "reduce"],
    "receiver_model": ["extract", "reduce"],  # reduce = plain no-shard-index run (reconcile all shards + fit)
}


class StagePhaseError(RuntimeError):
    """A map/reduce stage had a terminal ERROR/CANCELED job -- the pipeline stops (fail-loud)."""


def phase_sequence(spec: TrainerSpec) -> list[str]:
    """The ordered map/reduce phases for a trainer (``["full"]`` for the single-job public trainers)."""
    return ["full"] if spec.source == "public" else _PHASE_SEQUENCES[spec.name]


def _parse_job_id(out: str, label: str) -> str:
    m = re.search(r"jobs/([^/\s]+/[0-9a-f]{6,})", out) or re.search(r"(?:ID:\s*|id=)([0-9a-f]{6,})", out)
    if not m:
        raise RuntimeError(f"{label}: no job id in: {out}")
    return m.group(1)


def run_stage(inners: list[str], spec: TrainerSpec, *, bucket: str, attempts: int, runner=subprocess_runner):
    """Launch every inner command in a stage (parallel on HF); return the job ids."""
    ids = []
    for i, inner in enumerate(inners):
        out, code = runner(build_argv(spec, inner, bucket=bucket, attempts=attempts))
        if code != 0:
            raise RuntimeError(f"{spec.name} stage job #{i}: hf jobs run failed: {out}")
        ids.append(_parse_job_id(out, f"{spec.name}#{i}"))
    return ids


def wait_all(job_ids, *, runner=subprocess_runner, poll=poll_until_terminal, sleep=time.sleep) -> dict:
    """Poll every job to a terminal stage; raise ``StagePhaseError`` on the first ERROR/CANCELED."""
    stages = {}
    for jid in job_ids:
        stage = poll(jid, runner=runner, sleep=sleep)
        stages[jid] = stage
        if stage in ("ERROR", "CANCELED"):
            raise StagePhaseError(f"{jid}: terminal {stage}")
    return stages


def job_output(job_id, *, runner=subprocess_runner) -> str:
    """The job's captured stdout/stderr (``hf jobs logs``) -- used to read the prep phase's study list."""
    out, _ = runner(["hf", "jobs", "logs", job_id])
    return out


def _studies_from_prep(prep_jid, *, runner) -> list[str]:
    """xshot/xcross ``--prep-only`` prints ``{"study_root":..., "studies":[...]}``; return the study tags."""
    out = job_output(prep_jid, runner=runner)
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{") and "studies" in line:
            try:
                return list(json.loads(line)["studies"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    raise StagePhaseError(f"{prep_jid}: could not parse the study list from the prep output")


def run_pipeline(
    spec: TrainerSpec,
    build: VariantBuild,
    *,
    tag: str,
    run_key: str,
    corpus_repo: str | None = None,
    probe_repo: str | None = None,
    bucket: str,
    attempts: int,
    n_extract: int = 4,
    n_folds: int = 5,
    arms_json: str | None = None,
    runner=subprocess_runner,
    sleep=time.sleep,
) -> str:
    """Run the full map/reduce DAG for ONE variant and return the reduce (or full) job id. Each stage's
    jobs run in parallel, then the orchestrator waits for all before the next. ``n_extract`` extract shards;
    ``n_folds`` ghost CV folds (match the trainer's ``--cv-folds``); the xshot/xcross study count is
    enumerated from the prep job's output. Fail-loud: any stage ERROR raises ``StagePhaseError``.

    ``arms_json`` (ADR-123 Amendment B): the precomputed visibility-gated --arms policy; if omitted it is
    computed ONCE here, off-HF, from the staged corpus (the job receives it as data, never computes it)."""
    if arms_json is None and spec.visibility_gated and corpus_repo:
        arms_json = arms_payload(corpus_repo)

    def inner(phase, **kw):
        return build_phase_command(
            spec,
            build,
            tag=tag,
            run_key=run_key,
            corpus_repo=corpus_repo,
            probe_repo=probe_repo,
            phase=phase,
            cv_folds=n_folds,
            arms_json=arms_json,
            **kw,
        )

    last, prep_jid = "", None
    for phase in phase_sequence(spec):
        if phase == "extract":
            inners = [inner("extract", shard=f"{i}/{n_extract}") for i in range(n_extract)]
        elif phase == "fold":
            inners = [inner("fold", fold=k) for k in range(n_folds)]
        elif phase == "study":
            inners = [inner("study", study=s) for s in _studies_from_prep(prep_jid, runner=runner)]
        else:  # build_cache / prep / reduce / full -- single job
            inners = [inner(phase)]
        ids = run_stage(inners, spec, bucket=bucket, attempts=attempts, runner=runner)
        wait_all(ids, runner=runner, sleep=sleep)
        if phase == "prep":
            prep_jid = ids[0]
        last = ids[-1]
    return last


def make_local_runner(bash: str = "bash"):
    """A ``runner`` that executes the job inner command as a LOCAL subprocess (the DGX e2e + fallback
    executor) instead of ``hf jobs run``. Handles the ``run``/``inspect``/``logs`` argv the orchestrator
    issues, so ``run_pipeline(..., runner=make_local_runner())`` drives the identical DAG off-HF."""
    store: dict[str, tuple[str, int]] = {}
    seq = {"n": 0}

    def runner(argv):
        head = argv[:3]
        if head == ["hf", "jobs", "run"]:
            inner = argv[-1]  # ... -- bash -lc <inner>
            # `bash -c` (NOT `-lc`): a login shell re-sources the caller's profile, which off-HF (CI
            # windows runners, the DGX) can exit non-zero or reset PATH; the inner command is
            # self-contained and inherits the parent env, so a login shell only adds that hazard.
            p = subprocess.run([bash, "-c", inner], capture_output=True, text=True)  # noqa: S603
            seq["n"] += 1
            jid = f"local/{seq['n']:06x}"
            store[jid] = (p.stdout + p.stderr, p.returncode)
            if p.returncode != 0:
                return (store[jid][0], p.returncode)  # a failed launch -> run_stage raises
            return (f"id={jid.split('/')[-1]} url=https://huggingface.co/jobs/{jid}", 0)
        if head == ["hf", "jobs", "inspect"]:
            jid = argv[3]
            stage = "COMPLETED" if store.get(jid, ("", 1))[1] == 0 else "ERROR"
            return (json.dumps([{"id": jid, "status": {"stage": stage}}]), 0)
        if head == ["hf", "jobs", "logs"]:
            return (store.get(argv[3], ("", 0))[0], 0)
        return ("", 0)

    return runner
