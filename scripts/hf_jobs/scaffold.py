"""Scaffold a born-ready compute pipeline (ADR-128 spec 5.8): ``python -m scripts.hf_jobs scaffold
trainer|driver <name> [--root DIR]``.

It writes a two-stage pipeline that passes ``tests/scripts/test_compute_contract.py`` on day one -- an
``ItemStage`` ``extract`` over ``staged_flat_refs`` (one row per match) and a single-unit ``fit``
(trainer) or ``report`` (driver) stage that returns a ``FileSet`` -- plus its W1N1-vs-W2N2 parity test,
its registry entry (``TrainerSpec`` / ``DriverSpec``), its toy input (``TOY_PARAMS`` + ``TOY_BUILDERS``)
and, for a trainer, a commented ``[[artifacts]]`` stub in the artifact manifest. Insertion is text
insertion before the scaffold markers; an existing name is refused.
"""

from __future__ import annotations

import argparse
import pathlib
import re
from collections.abc import Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MARKERS = {
    "trainer": "    # --- scaffold: trainers above ---",
    "driver": "    # --- scaffold: drivers above ---",
    "toy_params": "    # --- scaffold: toys above ---",
    "toy_builders": "    # --- scaffold: toy builders above ---",
}
_NAME = re.compile(r"^[a-z][a-z0-9_]*[a-z0-9]$")

_MODULE = '''"""{title}: a born-ready ADR-128 {kind} (scaffolded by `python -m scripts.hf_jobs scaffold`).

Follow docs/howto/compute-pipelines.md: units are keyed and single-threaded (a stage opts into its vCPU
share only with `thread_safe=True` behind a byte-identity gate, ADR-128 amendment); every reduce combines in
sorted key order; the module reads only staged datasets (the I1 gate). Replace the stage bodies; keep the
contract (`PIPELINE`, the parity test, the registry entry, the toy input).
"""

from __future__ import annotations

import json

import pandas as pd

from scripts._driver_sources import dataset_dir, staged_flat_refs
from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, cli_main

_PARAMS = (Param("dataset", str, required=True, help="the staged flat dataset (scripts/hf_jobs/datasets.toml)"),)


def _refs(ctx):
    return staged_flat_refs(ctx.data_root, ctx.params["dataset"])


def _key(ref):
    return ref.stem


def _load(ref, ctx):
    return pd.read_parquet(dataset_dir(ctx.data_root, ctx.params["dataset"]) / ref.files[0])


def _extract_work(frames, ctx) -> pd.DataFrame:
    """One match -> one row (replace with the real per-match computation): its row and frame counts."""
    n_frames = frames.groupby(["period_id", "frame_id"]).ngroups
    return pd.DataFrame({{"n_rows": [len(frames)], "n_frames": [n_frames]}})


def _extract_reduce(results, ctx) -> pd.DataFrame:
    return pd.concat(list(results.values()), ignore_index=True)


def _one_unit(ctx) -> list[str]:
    return ["only"]


def _{final}_work(key, ctx) -> FileSet:
    rows = ctx.upstream("extract")
    payload = {{"n_matches": len(rows), "n_rows": int(rows["n_rows"].sum())}}
    return FileSet({{"{output_file}": json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")}})


PIPELINE = Pipeline(
    name="{name}",
    module="scripts.{module}",
    params=_PARAMS,
    stages=(
        ItemStage(
            name="extract",
            refs=_refs,
            key=_key,
            load=_load,
            work=_extract_work,
            reduce=_extract_reduce,
            uses_dataset=True,
        ),
        Stage(
            name="{final}",
            units=_one_unit,
            work=_{final}_work,
            needs=("extract",),
            reduce=lambda r, ctx: r["only"],  # a single-unit stage's output is its one result
        ),
    ),
    output="{final}",
)


if __name__ == "__main__":
    cli_main(PIPELINE)
'''

_PARITY = '''"""ADR-128 W1N1-vs-W2N2 parity for `{module}` (scaffolded).

Non-vacuity: the extract really fanned out over 2 jobs (`fanned_jobs`). The scaffold's placeholder unit is
trivial, so one worker can drain a job before the second worker starts (workers launch staggered under RAM
backpressure). When `_extract_work` does real per-match work, add `completion_out_of_order(...)` for the
extract (units finish out of key order inside a job), as every migrated pipeline's parity test does.
"""

from __future__ import annotations

from scripts._toy_corpora import scaffold_flat_toy
from tests.scripts._parity import artifact, fanned_jobs, outputs, plan, run_fanned, run_serial

MODULE = "scripts.{module}"
ARGS = ["--dataset", "toy{name}"]


def test_w1n1_equals_w2n2(tmp_path):
    data = scaffold_flat_toy("toy{name}")(tmp_path / "data")
    run_serial(MODULE, tmp_path / "s1", data, ARGS, tmp_path / "o1")
    run_fanned(MODULE, tmp_path / "s2", data, ARGS, tmp_path / "o2")
    assert outputs(tmp_path / "s1") == outputs(tmp_path / "s2")
    assert artifact(tmp_path / "o1") == artifact(tmp_path / "o2")
    gens = {{s["name"]: s["generation"] for s in plan(MODULE, tmp_path / "s2", data, ARGS)["stages"]}}
    assert fanned_jobs(tmp_path / "s2", gens["extract"]) == 2
'''

_TRAINER_ENTRY = """    "{name}": TrainerSpec(
        name="{name}",
        module="scripts.{module}",
        extras=(),
        base_args=(),
        builds=(VariantBuild("default", (), "", ""),),  # set the committed wheel dir when one ships
        primary_metric="n_rows",  # replace with the metric the head-to-head scorer emits
        metric_direction="higher_better",
        delta=0.01,
        certification="owner",
        pipeline="scripts.{module}",
    ),
"""
_DRIVER_ENTRY = """    "{name}": DriverSpec("{name}", "scripts.{module}", "rebaseline", "docs/research/{name}"),
"""
_TOY_PARAMS = """    "scripts.{module}": ("--dataset", "toy{name}"),
"""
_TOY_BUILDER = """    "scripts.{module}": scaffold_flat_toy("toy{name}"),
"""
_MANIFEST_STUB = """
# Scaffolded `{module}` (ADR-128): declare its artifact before its first campaign, then uncomment.
# [[artifacts]]
# id = "{name}_default"
# destination = "wheel:silly_kicks/<package>/<weights dir>"
# job = "{name}"
# variant = "default"
# tier = "owner"
# certification = "owner"
# dataset = "<datasets.toml entry>"
# training_commit = "<the campaign commit>"
# commit_field = "metrics.json:run_commit"
"""


def _insert(path: pathlib.Path, marker: str, text: str) -> None:
    src = path.read_text(encoding="utf-8")
    if src.count(marker) != 1:
        raise SystemExit(f"{path}: expected exactly one scaffold marker {marker.strip()!r}")
    path.write_text(src.replace(marker, text.rstrip("\n") + "\n" + marker), encoding="utf-8")


def scaffold(kind: str, name: str, *, root: pathlib.Path = REPO_ROOT) -> list[pathlib.Path]:
    """Write the module, its parity test and its registrations under ``root``; returns the files written."""
    if kind not in ("trainer", "driver"):
        raise SystemExit(f"kind must be trainer or driver, got {kind!r}")
    if not _NAME.match(name):
        raise SystemExit(f"{name!r}: use lower_snake_case")
    module = f"train_{name}" if kind == "trainer" else f"build_{name}"
    mod_path = root / "scripts" / f"{module}.py"
    test_path = root / "tests" / "scripts" / f"test_parity_{name}.py"
    registry = root / "scripts" / "hf_jobs" / "registry.py"
    toys = root / "scripts" / "_toy_corpora.py"
    for p in (mod_path, test_path):
        if p.exists():
            raise SystemExit(f"{p} exists: refusing to overwrite (pick another name)")
    if f'"{name}":' in registry.read_text(encoding="utf-8") or f'"scripts.{module}"' in toys.read_text(
        encoding="utf-8"
    ):
        raise SystemExit(f"{name!r} is already registered")
    fields = {
        "name": name,
        "module": module,
        "kind": kind,
        "title": module,
        "final": "fit" if kind == "trainer" else "report",
        "output_file": "metrics.json" if kind == "trainer" else f"{name}.json",
    }
    mod_path.write_text(_MODULE.format(**fields), encoding="utf-8")
    test_path.write_text(_PARITY.format(**fields), encoding="utf-8")
    _insert(registry, MARKERS[kind], (_TRAINER_ENTRY if kind == "trainer" else _DRIVER_ENTRY).format(**fields))
    _insert(toys, MARKERS["toy_params"], _TOY_PARAMS.format(**fields))
    _insert(toys, MARKERS["toy_builders"], _TOY_BUILDER.format(**fields))
    written = [mod_path, test_path, registry, toys]
    if kind == "trainer":
        manifest = root / "docs" / "artifacts" / "manifest.toml"
        with manifest.open("a", encoding="utf-8") as f:
            f.write(_MANIFEST_STUB.format(**fields))
        written.append(manifest)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.hf_jobs scaffold", description=(__doc__ or "scaffold").splitlines()[0]
    )
    ap.add_argument("kind", choices=["trainer", "driver"])
    ap.add_argument("name", help="lower_snake_case; the module becomes scripts/train_<name>.py or build_<name>.py")
    ap.add_argument("--root", type=pathlib.Path, default=REPO_ROOT, help="repo root (tests use a temp copy)")
    a = ap.parse_args(argv)
    for p in scaffold(a.kind, a.name, root=a.root):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
