"""One-unit adapter stage (ADR-128 spec 5.6): run an UNMIGRATED trainer's existing CLI unchanged as ONE unit.

Cycle 1 to Cycle 2: the three public trainers (``train_pass_completion``, ``train_xsuccess``,
``train_match_outcome_dependence``) keep their own CLI; the campaign launches them through this adapter, so
they get the stage contract's store, ledger, run key and resume-by-unit without a migration. Their own
``for_each`` shards (``--out`` and, where needed, ``--shard-root``) live on the persistent store, so a killed
attempt resumes through the target's OWN resume. The target runs under the deterministic environment
(INV-2; a `thread_safe` stage may opt into its vCPU share per the ADR-128 amendment). Cycle 2 replaces each
with a real pipeline (spec 12 R7).

Usage::

    python -m scripts.hf_jobs.adapter --target scripts.train_xsuccess \\
        --target-args '["--competition-id","43"]' --fresh-subdir weights --dataset sb_open \\
        --store <DIR> --stages run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `from scripts._X` resolves

from scripts._stages import FileSet, Param, Pipeline, Stage, cli_main

_REPO = Path(__file__).resolve().parents[2]

_PARAMS = (
    Param(
        "dataset",
        str,
        "sb_open",
        help="declared dataset name; the open-data revision is pinned through the job env (decision P6)",
    ),
    Param("target", str, None, required=True, help="the unmigrated trainer module (run as python -m <target>)"),
    Param("target_args", str, "[]", help="JSON list of the target's own CLI arguments (registry-pinned)"),
    Param("fresh_subdir", str, "", help="where the target writes its artifact inside --out ('' = --out itself)"),
)

#: Targets whose `for_each` shard root defaults to an ephemeral /tmp: pinned under the persistent `--out`.
_SHARD_ROOT_TARGETS = frozenset({"scripts.train_match_outcome_dependence"})


def _run_work(key, ctx) -> FileSet:
    """Run an unmigrated trainer's EXISTING CLI unchanged as ONE unit (spec 5.6). Its `--out` (and
    `--shard-root`) live on the persistent store, so the target's own for_each shards keep its resume."""
    import subprocess

    from scripts._thread_pin import deterministic_env

    root = ctx.store.local_path(f"{ctx.generation('run')}/target_out")
    if root is None:
        raise RuntimeError("the adapter runs only on a local (or mounted) store: the target needs a real --out")
    root.mkdir(parents=True, exist_ok=True)
    args = json.loads(ctx.params["target_args"])
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ValueError("--target-args must be a JSON list of strings")
    argv = [sys.executable, "-m", ctx.params["target"], *args, "--out", str(root)]
    if ctx.params["target"] in _SHARD_ROOT_TARGETS:
        argv += ["--shard-root", str(root / "shards")]
    subprocess.run(argv, check=True, env=deterministic_env(), cwd=str(_REPO))  # noqa: S603 -- registry-pinned module + args
    art = root / ctx.params["fresh_subdir"] if ctx.params["fresh_subdir"] else root
    return FileSet({p.name: p.read_bytes() for p in sorted(art.iterdir()) if p.is_file()})


PIPELINE = Pipeline(
    name="adapter",
    module="scripts.hf_jobs.adapter",
    params=_PARAMS,
    stages=(
        Stage(
            name="run",
            units=lambda ctx: ["only"],
            work=_run_work,
            reduce=lambda r, ctx: r["only"],
            params=("target", "target_args", "fresh_subdir"),
            uses_dataset=True,
            # A legacy target reads whatever it reads, in ONE subprocess (one reader): a download job
            # fetches the whole dataset.
            dataset_files=lambda ctx: ("*",),
        ),
    ),
    output="run",
)

if __name__ == "__main__":
    cli_main(PIPELINE)
