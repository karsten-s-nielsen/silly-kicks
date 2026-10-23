"""The dataset read-audit gate (ADR-128). Behavioural, not an inventory (ADR-056): every compute-contract
pipeline runs its dataset stages on its toy input, in process, with every file-read entry point watched.

Two rules, both measured on the real reads:

1. NO DIRECT READ. Every read of a dataset file goes through the read gate (`scripts._dataset_reads`):
   on real HF a direct pyarrow read of a mounted parquet file (many small range reads) stalled for
   126-462 s per file and, with 8 workers, failed with EIO (the 2026-10-20 probes).
2. EVERY READ IS DECLARED. Every file a stage reads matches the files that stage declares
   (`datasets.stage_files`: an ItemStage's unit refs plus the dataset metadata; a Stage's
   `dataset_files`), so a `download` job fetches everything it then reads.

The toy datasets resolve through a lock built from the toy tree BEFORE the audit starts, exactly as an
HF job resolves its units from `datasets.lock.json` (enumeration reads no file).
"""

from __future__ import annotations

import builtins
import contextlib
import fnmatch
import importlib
import io
import json
import os
import pathlib
import threading
from collections import defaultdict

import pytest

from tests.scripts.test_compute_contract import _compliant


class _Audit:
    """Records every file read under ``root`` made on the test thread: gated (inside a read slot) or a
    violation (outside one)."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root.resolve()
        self.thread = threading.current_thread()
        self.stage: str | None = None
        self.reads: dict[str | None, set[str]] = defaultdict(set)
        self.violations: list[tuple[str | None, str, str]] = []
        self.gated = 0

    def _rel(self, arg) -> str | None:
        if not isinstance(arg, (str, os.PathLike)):
            return None  # an in-memory buffer (the gated parse), a file descriptor
        try:
            return pathlib.Path(os.fspath(arg)).resolve().relative_to(self.root).as_posix()
        except (ValueError, OSError):
            return None

    def seen(self, entry: str, arg) -> None:
        if threading.current_thread() is not self.thread:
            return  # the toy open-data HTTP server reads its own files on its own thread
        rel = self._rel(arg)
        if rel is None:
            return
        if self.gated:
            self.reads[self.stage].add(rel)
        else:
            self.violations.append((self.stage, entry, rel))


def _install(audit: _Audit, monkeypatch) -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    from scripts import _dataset_reads

    def watch(entry: str, fn):
        def wrapped(target, *a, **k):
            audit.seen(entry, target)
            return fn(target, *a, **k)

        return wrapped

    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", watch("open", real_open))
    monkeypatch.setattr(io, "open", watch("io.open", real_open))
    monkeypatch.setattr(pd, "read_parquet", watch("pandas.read_parquet", pd.read_parquet))
    for name in ("read_table", "read_schema", "read_metadata"):
        monkeypatch.setattr(pq, name, watch(f"pyarrow.parquet.{name}", getattr(pq, name)))
    real_pf = pq.ParquetFile

    class _WatchedParquetFile(real_pf):  # type: ignore[misc,valid-type]
        def __init__(self, source, *a, **k):
            audit.seen("pyarrow.parquet.ParquetFile", source)
            super().__init__(source, *a, **k)

    monkeypatch.setattr(pq, "ParquetFile", _WatchedParquetFile)
    real_slot = _dataset_reads.read_slot

    @contextlib.contextmanager
    def slot():
        with real_slot():
            audit.gated += 1
            try:
                yield
            finally:
                audit.gated -= 1

    monkeypatch.setattr(_dataset_reads, "read_slot", slot)


def _toy_lock(data: pathlib.Path, names: set[str]) -> dict[str, dict]:
    """A lock record per toy dataset, measured from the toy tree like `datasets record` measures a staged
    one: the HF in-job path (units from the lock, no file read to enumerate)."""
    from scripts._driver_sources import staged_flat_refs, staged_pergame_refs

    lock = {}
    for name in names:
        root = data / name
        if not root.is_dir():
            continue  # not a staged dataset (open data arrives over HTTP)
        if (root / "shards").is_dir():
            refs = staged_flat_refs(data, name)
            lock[name] = {
                "generation": refs[0].files[0].split("/")[1],
                "units": [{"stem": r.stem, "source_provider": r.source_provider} for r in refs],
            }
        else:
            refs = staged_pergame_refs(data, name)
            lock[name] = {"units": [{"stem": r.stem, "source_provider": r.source_provider} for r in refs]}
    return lock


def _findings(audit: _Audit, declared: dict[str | None, list[str]]) -> list[str]:
    out = [f"{st}: DIRECT {entry} of {rel} (outside the read gate)" for st, entry, rel in audit.violations]
    for st, rels in sorted(audit.reads.items(), key=lambda kv: str(kv[0])):
        pats = declared.get(st, [])
        out += [
            f"{st}: UNDECLARED read of {rel}"
            for rel in sorted(rels)
            if not any(fnmatch.fnmatchcase(rel, p) for p in pats)
        ]
    return out


def _run_dataset_stages(pipeline, ctx, audit: _Audit) -> dict[str | None, list[str]]:
    """Run every stage up to the last dataset stage (units in process, then the reduce); returns the files
    each stage declares, relative to the data root (the audit's frame)."""
    from scripts._stages import ItemStage, reduce_stage, run_unit, sorted_units
    from scripts.hf_jobs.datasets import stage_files
    from scripts.hf_jobs.executor import InnerPool

    last = max(i for i, st in enumerate(pipeline.stages) if st.uses_dataset)
    declared: dict[str | None, list[str]] = {}
    for st in pipeline.stages[: last + 1]:
        audit.stage = st.name
        keys = sorted_units(ctx, st)
        refs = st.ref_map(ctx) if isinstance(st, ItemStage) else {}
        pool = InnerPool(1) if st.inner_parallel else None
        for k in keys:
            if run_unit(ctx, st, k, worker="audit", pool=pool, ref=refs.get(k))[0] == "failed":
                raw = ctx.store.read_bytes(f"{ctx.generation(st.name)}/{k}.failed.json")
                pytest.fail(f"{st.name}/{k} failed under the audit: {json.loads(raw or b'{}').get('error')}")
        reduce_stage(ctx, st.name)
        if st.uses_dataset:
            name = ctx.params.get(st.dataset_param)
            declared[st.name] = [f"{name}/{p}" if name else p for p in stage_files(ctx, st, keys)]
    audit.stage = None
    return declared


@pytest.mark.parametrize("name", _compliant())
def test_every_dataset_read_is_gated_and_declared(name, tmp_path, monkeypatch):
    from scripts import _driver_sources
    from scripts._stages import LocalStore, build_parser, make_context
    from scripts._toy_corpora import TOY_OPEN_DATA, TOY_PARAMS, toy_session

    mod = importlib.import_module(f"scripts.{name}")
    pipe = mod.PIPELINE
    if pipe.module in TOY_OPEN_DATA:
        # The refs unit reads the served open-data tree through statsbombpy, which only the `hfjobs` group
        # installs: these two cases run in `.venv-hfjobs` and skip in CI, like their parity tests.
        pytest.importorskip("statsbombpy")
    with toy_session(pipe.module, tmp_path / "data") as (data, env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        ns = build_parser(pipe).parse_args([*TOY_PARAMS[pipe.module], "--plan", "--store", str(tmp_path / "s")])
        params = {p.name: getattr(ns, p.name) for p in pipe.params}
        names = {params[st.dataset_param] for st in pipe.stages if st.uses_dataset and params.get(st.dataset_param)}
        lock = _toy_lock(data, names)
        monkeypatch.setattr(_driver_sources, "_lock_units", lambda ds: lock.get(ds) if ds else None)
        ctx = make_context(
            pipe,
            params,
            {"commit": "t", "tree_state": "clean", "code_identity": "t"},
            store=LocalStore(tmp_path / "s"),
            data_root=data,
        )
        audit = _Audit(data)
        _install(audit, monkeypatch)
        declared = _run_dataset_stages(pipe, ctx, audit)
        monkeypatch.undo()  # restore the readers before pytest reports
    assert _findings(audit, declared) == []
    if lock:  # non-vacuity: a pipeline with a STAGED dataset really read it through the gate (open data does not)
        assert any(audit.reads.values()), f"{name}: no gated dataset read was observed"


# ---- the gate's red fixtures (spec 5.8: a gate proves it can fail) ---------------------------------


def _toy_flat(tmp_path) -> pathlib.Path:
    from scripts._toy_corpora import build_flat_corpus

    build_flat_corpus(tmp_path / "data" / "toyflat", n_games=2, providers=("skillcorner",))
    return tmp_path / "data"


def _fixture_ctx(tmp_path, data, stages):
    from scripts._stages import LocalStore, Param, Pipeline, make_context

    pipe = Pipeline(name="audit_fixture", module="tests.fixture", stages=stages, params=(Param("dataset", str),))
    return pipe, make_context(
        pipe,
        {"dataset": "toyflat"},
        {"commit": "t", "tree_state": "clean", "code_identity": "t"},
        store=LocalStore(tmp_path / "s"),
        data_root=data,
    )


def test_the_audit_flags_a_direct_read(tmp_path, monkeypatch):
    import pandas as pd

    from scripts._driver_sources import dataset_dir, staged_flat_refs
    from scripts._stages import ItemStage

    data = _toy_flat(tmp_path)
    st = ItemStage(
        name="extract",
        refs=lambda ctx: staged_flat_refs(ctx.data_root, "toyflat"),
        key=lambda r: r.stem,
        load=lambda r, ctx: pd.read_parquet(dataset_dir(ctx.data_root, "toyflat") / r.files[0]),  # DIRECT
        work=lambda item, ctx: item.head(1)[["game_id"]],
        uses_dataset=True,
    )
    pipe, ctx = _fixture_ctx(tmp_path, data, (st,))
    audit = _Audit(data)
    _install(audit, monkeypatch)
    declared = _run_dataset_stages(pipe, ctx, audit)
    monkeypatch.undo()
    found = _findings(audit, declared)
    assert found and all(f.startswith("extract: DIRECT ") and " of toyflat/shards/" in f for f in found), found
    assert any("DIRECT pandas.read_parquet of toyflat/shards/" in f for f in found), found


def test_the_audit_flags_an_undeclared_read(tmp_path, monkeypatch):
    from scripts._driver_sources import dataset_dir, read_dataset_json
    from scripts._stages import Stage

    data = _toy_flat(tmp_path)
    st = Stage(
        name="meta",
        units=lambda ctx: ["only"],
        work=lambda k, ctx: read_dataset_json(dataset_dir(ctx.data_root, "toyflat") / "home_teams.json"),
        uses_dataset=True,
        dataset_files=lambda ctx: (),  # declares nothing, reads home_teams.json
    )
    pipe, ctx = _fixture_ctx(tmp_path, data, (st,))
    audit = _Audit(data)
    _install(audit, monkeypatch)
    declared = _run_dataset_stages(pipe, ctx, audit)
    monkeypatch.undo()
    assert _findings(audit, declared) == ["meta: UNDECLARED read of toyflat/home_teams.json"]


def test_ghost_gk_validate_declares_every_detection_aware_shard_it_reads(monkeypatch):
    """`validate_corpus_visibility` reads the footer of EVERY detection-aware shard, not one: a download
    job must fetch them all (the toy corpus may hold one provider, so this pins the rule directly)."""
    import scripts._ghost_gk_stages as gk  # ADR-129: `_refs`/`_validate_files` live in the stage module
    from scripts._driver_sources import StagedRef

    refs = [
        StagedRef(
            "gradientsports",
            "1",
            "gradientsports__1",
            "gradientsports",
            ("shards/g/gradientsports__1.parquet", "_actions/gradientsports__1.parquet"),
        ),
        StagedRef(
            "skillcorner",
            "2",
            "skillcorner__2",
            "skillcorner",
            ("shards/g/skillcorner__2.parquet", "_actions/skillcorner__2.parquet"),
        ),
        StagedRef(
            "skillcorner",
            "3",
            "skillcorner__3",
            "unknown",
            ("shards/g/skillcorner__3.parquet", "_actions/skillcorner__3.parquet"),
        ),
        StagedRef(
            "skillcorner",
            "4",
            "skillcorner__4",
            "skillcorner",
            ("shards/g/skillcorner__4.parquet", "_actions/skillcorner__4.parquet"),
        ),
    ]
    monkeypatch.setattr(gk, "_refs", lambda ctx: refs)
    assert gk._validate_files(None) == [
        "home_teams.json",
        "shards/g/gradientsports__1.parquet",  # the schema probe (refs[0])
        "shards/g/skillcorner__2.parquet",
        "shards/g/skillcorner__4.parquet",
    ]
