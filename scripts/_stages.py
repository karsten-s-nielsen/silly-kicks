"""The stage/unit compute contract (ADR-128).

WHY THIS EXISTS. The 4.138.0 re-fit campaign ran 5-9 hour jobs on one of eight vCPUs, from uncommitted
scratchpad orchestration, because the per-trainer phase flags (ADR-123) had untested real-HF paths. This
module is the ONE contract every heavy trainer and corpus driver declares instead: a static DAG of
stages, each a set of independent, keyed, single-threaded units plus an optional reducer. The executor
(`scripts/hf_jobs/executor.py`) places units as W worker processes per job x N jobs, and every output is
byte-identical for every (W, N) (INV-1) because:

1. a unit is single-threaded (the executor starts workers with `_thread_pin.deterministic_env`);
2. a unit writes ONE result file per key, atomically, with a checksum sidecar written first;
3. every reduction receives results in a declared total order of keys, never completion order
   (floating-point addition is not associative, so this rule is load-bearing);
4. a token digests the code identity, the stage's declared parameters, the dataset identity and the
   upstream tokens, so a stale or foreign result is never reused.

Resume (INV-3): a result is trusted only when its checksum matches; a kill loses at most the units in
flight. ADR-052's `for_each` is the implementation of an `ItemStage`, so its shards, exclusion markers,
counters sidecars, conservation and injectivity apply unchanged.

Vocabulary (spec section 3, one meaning each): unit, stage, pipeline, worker, job, store.

Stdlib only at import; pandas is imported inside the functions that need it (the `_driver` rule).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib
import io
import json
import os
import pathlib
import re
import sys
import time
import warnings
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, Protocol

from scripts._driver import _token, _write_bytes_atomically, join_key
from scripts._item_outcome import ItemExcluded

#: The repo root (``scripts/`` parent): the base ``stage_code_identity`` resolves closure paths against.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The job IMAGE's Python version (INV-6), keyed into every clean-tree token. It is a FIXED constant, not the
#: local interpreter's version, so the local controller (``--plan``) and the HF job compute the SAME token
#: (CCTP-IMPL-15). Task 18 sources it from the image lock; until then it is pinned to the image's Python.
_IMAGE_PYTHON = "3.12.14"

#: Numeric key components are zero-padded to this width, so lexicographic key order == numeric order.
KEY_WIDTH = 4
_RESULT_SUFFIXES = (".parquet", ".json", ".fileset.json")
_OUTPUT = "_output"
#: Where a corrupt result's files are moved so its unit is pending again (INV-3, S5.1).
CORRUPT_DIR = ".corrupt"
MISSING = object()
#: A failed unit is retried until it has failed this many times (single-sourced with the campaign
#: controller, `planner.Constraints.unit_attempts`). A memory failure below it is DEFERRED: the unit waits
#: for its escalation job (W=1, the whole box), so it neither fails its stage nor lets it reduce.
UNIT_ATTEMPTS = 2


class StageIncompleteError(RuntimeError):
    """A reduce was requested while units are pending, or failures exceed the stage's tolerance."""


class UpstreamNotReducedError(RuntimeError):
    """A stage read an upstream stage whose output does not exist yet."""


def int_key(i: int) -> str:
    """A zero-padded key component for an integer (folds, replicates): ``int_key(3) == "0003"``."""
    if i < 0:
        raise ValueError(f"key components are non-negative, got {i}")
    return f"{i:0{KEY_WIDTH}d}"


@dataclasses.dataclass(frozen=True)
class FileSet:
    """A result made of named files (e.g. a model artifact, or a persisted study-input directory).

    Names are RELATIVE POSIX paths (``"model.json"``, ``"corpus/X.npy"``): no absolute path, no ``..``,
    no backslash, no component starting with ``.``. A model artifact uses flat names only (ADR-088's
    top-level-files rule is enforced again at publish)."""

    files: Mapping[str, bytes]

    def __post_init__(self) -> None:
        for name in self.files:
            parts = pathlib.PurePosixPath(name).parts
            if (
                not name
                or "\\" in name
                or name.startswith("/")
                or not parts
                or any(p in ("..", "") or p.startswith(".") for p in parts)
            ):
                raise ValueError(f"FileSet names must be relative POSIX paths without dot-components, got {name!r}")


@dataclasses.dataclass(frozen=True)
class Param:
    """One pipeline parameter; its CLI flag is ``--`` + ``name`` with ``_`` -> ``-``."""

    name: str
    type: Callable[[str], Any] = str
    default: Any = None
    required: bool = False
    choices: tuple[Any, ...] | None = None
    flag: bool = False
    help: str = ""

    @property
    def cli(self) -> str:
        return "--" + self.name.replace("_", "-")


@dataclasses.dataclass(frozen=True)
class Stage:
    """A set of keyed units plus an optional reducer (spec 5.1).

    ``units(ctx)`` enumerates the keys deterministically (it may read upstream outputs).
    ``work(key, ctx)`` is pure, deterministic and single-threaded and returns ONE result: a DataFrame,
    a JSON-native value or a ``FileSet``. ``reduce(results, ctx)`` receives ``{key: result}`` in
    ``reduce_order`` order (default: sorted keys) and returns the stage output. An ``inner_parallel``
    unit runs alone at the top level of a job and receives ``ctx.pool`` (an ``InnerPool``); every
    combination of inner results MUST iterate ``ordered_values`` (sorted inner keys).

    ``params`` names the pipeline parameters the stage's numbers depend on; ``declare(params)`` adds
    any further declared inputs (the ADR-052 rule: a token closes silent omission, not mis-declaration).
    Resume across commits is sound by CLOSURE identity (ADR-129 `stage_code_identity`): a clean-tree token
    keys on the stage's locked closure, not the commit, so an unrelated commit reuses the result.
    ``max_failures`` (None = unlimited) is the failed-unit tolerance: the executor aborts when it is exceeded
    and the reduce refuses. ``abort_first_k``: the executor aborts when the first k units that complete in this
    job have all failed (systematic bug).
    """

    name: str
    units: Callable[[RunContext], Sequence[str]]
    work: Callable[[str, RunContext], Any]
    needs: tuple[str, ...] = ()
    reduce: Callable[[dict[str, Any], RunContext], Any] | None = None
    params: tuple[str, ...] = ()
    declare: Callable[[Mapping[str, Any]], Mapping[str, object]] | None = None
    inner_parallel: bool = False
    reduce_order: Callable[[str], Any] | None = None
    max_failures: int | None = 0
    abort_first_k: int = 3
    uses_dataset: bool = False
    #: The parameter naming the dataset this stage reads (``uses_dataset``); its identity keys the token.
    dataset_param: str = "dataset"
    #: An inner_parallel stage's inner key count (e.g. 5 CV folds): the planner never asks for more
    #: inner workers than there are inner keys. Ignored when ``inner_parallel`` is False.
    inner_width: int = 1
    #: A ``uses_dataset`` stage's declared files: ``dataset_files(ctx)`` returns the dataset-relative
    #: paths (or fnmatch patterns) its units read; ``()`` declares that it reads none (open data over
    #: HTTP). A ``download`` job fetches exactly these (`datasets.fetch`); the read-audit gate proves the
    #: declaration covers every read. Required when ``uses_dataset`` (`Pipeline.validate`). An
    #: ItemStage declares through its refs' ``files`` instead.
    dataset_files: Callable[[RunContext], Sequence[str]] | None = None
    #: ADR-129 (S5.9): the repo-tracked PACKAGE-DATA path prefixes this stage opens (e.g. a bundled
    #: ``silly_kicks/tracking/_*_weights/...`` directory, a verdicts JSON). The closure lock expands them with
    #: ``git ls-files`` into the stage's DATA membership; a change to one re-runs only the stages that declare
    #: it. Distinct from ``dataset_files`` (the DATASET mount). Required coverage is gated by the file-audit hook.
    data_files: tuple[str, ...] = ()
    #: ADR-129 (S5.10): Hub weight files this stage ADOPTS, as ``(repo_id, revision, path)`` triples; the token
    #: keys on the adopted content and the attestation records it.
    hub_files: tuple[tuple[str, str, str], ...] = ()


@dataclasses.dataclass(frozen=True)
class ItemStage:
    """A stage whose units are corpus ITEMS, executed through `_driver.for_each` (ADR-052).

    ``refs(ctx)`` returns cheap references; ``key(ref)`` the ADR-052 shard key (keys stay the
    pre-migration keys, `_KEY_EXCEPTIONS`); ``load(ref, ctx)`` the item (resume-before-load);
    ``work(item, ctx)`` one tidy DataFrame or None ("ran, produced nothing"). Exclusions
    (`ItemExcluded`), counters sidecars and conservation are `for_each`'s.
    """

    name: str
    refs: Callable[[RunContext], Sequence[Any]]
    key: Callable[[Any], Any]
    work: Callable[[Any, RunContext], Any]
    load: Callable[[Any, RunContext], Any] | None = None
    counters: Callable[[Any, Any], Mapping[str, object]] | None = None
    needs: tuple[str, ...] = ()
    reduce: Callable[[dict[str, Any], RunContext], Any] | None = None
    params: tuple[str, ...] = ()
    declare: Callable[[Mapping[str, Any]], Mapping[str, object]] | None = None
    reduce_order: Callable[[str], Any] | None = None
    max_failures: int | None = 0
    abort_first_k: int = 3
    uses_dataset: bool = False
    #: The parameter naming the dataset this stage reads (``uses_dataset``); its identity keys the token.
    dataset_param: str = "dataset"
    #: ADR-129 (S5.9/S5.10): see ``Stage.data_files`` / ``Stage.hub_files`` (an ItemStage carries the same two).
    data_files: tuple[str, ...] = ()
    hub_files: tuple[tuple[str, str, str], ...] = ()
    inner_parallel: bool = dataclasses.field(default=False, init=False)

    def units(self, ctx: RunContext) -> list[str]:
        return [join_key(self.key(r)) for r in self.refs(ctx)]

    def ref_map(self, ctx: RunContext) -> dict[str, Any]:
        return {join_key(self.key(r)): r for r in self.refs(ctx)}


AnyStage = Stage | ItemStage


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """A static DAG of stages. Declaration order IS the topological order (validated)."""

    name: str
    module: str
    stages: tuple[AnyStage, ...]
    params: tuple[Param, ...] = ()
    output: str | None = None

    def stage(self, name: str) -> AnyStage:
        for st in self.stages:
            if st.name == name:
                return st
        raise KeyError(f"{self.name}: no stage {name!r} (stages: {[s.name for s in self.stages]})")

    def validate(self) -> None:
        names = [st.name for st in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: duplicate stage names {names}")
        declared = {p.name for p in self.params}
        seen: set[str] = set()
        for st in self.stages:
            if not st.name or st.name.startswith("_") or "/" in st.name or "__" in st.name:
                raise ValueError(f"{self.name}: invalid stage name {st.name!r}")
            for need in st.needs:
                if need not in seen:
                    raise ValueError(f"{self.name}: {st.name} needs {need!r}, which is not declared BEFORE it")
            for p in st.params:
                if p not in declared:
                    raise ValueError(f"{self.name}: stage {st.name} names undeclared param {p!r}")
            if isinstance(st, Stage) and st.uses_dataset and st.dataset_files is None:
                raise ValueError(
                    f"{self.name}: dataset stage {st.name} must declare dataset_files (the files its units read; "
                    "() for none): a download job fetches exactly these"
                )
            seen.add(st.name)
        if self.output is not None and self.output not in seen:
            raise ValueError(f"{self.name}: output stage {self.output!r} is not declared")


class Store(Protocol):
    """POSIX-relative byte store. `LocalStore` executes; `hf_jobs.store.BucketStore` only reads."""

    def read_bytes(self, rel: str) -> bytes | None: ...
    def write_bytes(self, rel: str, data: bytes) -> None: ...
    def names(self, rel_dir: str) -> set[str]: ...
    def delete(self, rel: str) -> None: ...
    def local_path(self, rel: str) -> pathlib.Path | None: ...


@dataclasses.dataclass(frozen=True)
class LocalStore:
    root: pathlib.Path

    def _p(self, rel: str) -> pathlib.Path:
        return pathlib.Path(self.root) / pathlib.PurePosixPath(rel)

    def read_bytes(self, rel: str) -> bytes | None:
        p = self._p(rel)
        return p.read_bytes() if p.is_file() else None

    def write_bytes(self, rel: str, data: bytes) -> None:
        p = self._p(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes_atomically(p, data)

    def names(self, rel_dir: str) -> set[str]:
        p = self._p(rel_dir)
        return {c.name for c in p.iterdir() if c.is_file()} if p.is_dir() else set()

    def delete(self, rel: str) -> None:
        self._p(rel).unlink(missing_ok=True)

    def local_path(self, rel: str) -> pathlib.Path | None:
        return self._p(rel)


def open_store(uri: str | os.PathLike) -> Store:
    """``hf://buckets/<ns>/<bucket>/<prefix>`` -> read-only BucketStore; anything else -> LocalStore."""
    text = str(uri)
    if text.startswith("hf://buckets/"):
        from scripts.hf_jobs.store import BucketStore

        return BucketStore.from_uri(text)
    return LocalStore(pathlib.Path(text))


@dataclasses.dataclass
class RunContext:
    pipeline: Pipeline | PipelineView
    params: Mapping[str, Any]
    prov: Mapping[str, Any]
    store: Store
    data_root: pathlib.Path | None
    token_inputs: Mapping[str, Mapping[str, object]]
    tokens: Mapping[str, str]
    pool: Any = None

    def stage_dir(self, stage: str) -> str:
        return f"{self.pipeline.name}/{stage}"

    def generation(self, stage: str) -> str:
        return f"{self.stage_dir(stage)}/{self.tokens[stage]}"

    def upstream(self, stage: str) -> Any:
        out = read_result(self.store, f"{self.generation(stage)}/{_OUTPUT}")
        if out is MISSING:
            raise UpstreamNotReducedError(f"{self.pipeline.name}: upstream stage {stage!r} has no output yet")
        return out

    def summary(self, stage: str) -> dict:
        raw = self.store.read_bytes(f"{self.generation(stage)}/_summary.json")
        if raw is None:
            raise UpstreamNotReducedError(f"{self.pipeline.name}: stage {stage!r} has no summary yet")
        return json.loads(raw)


def _canon_json(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=True) + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_tokens(
    pipeline: Pipeline,
    params: Mapping[str, Any],
    code_id: str,
    dataset_identity: Mapping[str, object] | None,
    *,
    dataset_identities: Mapping[str, Mapping[str, object] | None] | None = None,
    stage_codes: Mapping[str, str] | None = None,
) -> tuple[dict[str, dict], dict[str, str]]:
    """The declared token inputs and the token (generation name) of every stage, in DAG order.

    ``dataset_identity`` is the ``--dataset`` identity; ``dataset_identities`` maps any OTHER dataset
    parameter a stage reads (`Stage.dataset_param`, e.g. ``probe_dataset``) to its identity. ``stage_codes``
    is the per-stage clean-tree closure identity (ADR-129); a stage absent from it keeps the commit identity."""
    inputs: dict[str, dict] = {}
    tokens: dict[str, str] = {}
    for st in pipeline.stages:
        d: dict[str, object] = {
            "pipeline": pipeline.name,
            "stage": st.name,
            "code": (stage_codes or {}).get(st.name) or code_id,
            "params": {
                p: (params[p].as_posix() if isinstance(params[p], pathlib.PurePath) else params[p]) for p in st.params
            },
            "upstream": {n: tokens[n] for n in sorted(st.needs)},
        }
        if st.declare is not None:
            d["declared"] = dict(st.declare(params))
        if st.uses_dataset:
            ident = (
                dataset_identity if st.dataset_param == "dataset" else (dataset_identities or {}).get(st.dataset_param)
            )
            d["dataset"] = dict(ident) if ident is not None else None
        # JSON-canonical BEFORE digesting: `ruthless.fingerprint` is type-tagged (a tuple and a list
        # digest differently), and the inputs reach workers through JSON (`context_to_json`). Without
        # this, a declared tuple would give the parent and the worker two different generations.
        d = json.loads(json.dumps(d, sort_keys=True))
        inputs[st.name] = d
        tokens[st.name] = _token(d, None)
    return inputs, tokens


@functools.cache
def _closure_file_bytes(path_str: str, _mtime_ns: int, _size: int) -> bytes:
    """The bytes of a closure/data file, cached per (path, mtime, size) so ``--plan`` reads each shared
    closure file once across every stage, yet re-reads a file edited in-process (the gate's token test).

    A TEXT file's CRLF is normalized to LF so the closure token is identical on a CRLF Windows checkout and the
    LF tree a Linux job builds from (ADR-129; otherwise a Windows controller's ``--plan`` token diverges from
    the job's, splitting reuse identity across platforms -- the same class as the image-tag CRLF fix). A BINARY
    file (it contains a NUL byte) is left raw: normalizing it would corrupt the content hash."""
    raw = pathlib.Path(path_str).read_bytes()
    return raw if b"\x00" in raw else raw.replace(b"\r\n", b"\n")


def stage_code_identity(pipeline, st, *, root, env_values, image, python) -> str | None:
    """ADR-129: the digest of the stage's locked closure files, its data members, ``uv.lock``, the image digest,
    the Python version and the output-affecting environment values. A commit that changes none of these keeps
    the identity, so its results are reused (INV-10). None when the stage is absent from the lock (the caller
    keeps the commit identity: no reuse across commits, the safe side)."""
    from scripts.hf_jobs.closures import load_lock

    entry = load_lock().get(f"{pipeline.name}/{st.name}")
    if entry is None:
        return None
    root = pathlib.Path(root)
    h = hashlib.sha256()
    for rel in [*entry["modules"], *entry["data"], "uv.lock"]:
        f = root / rel
        data = _closure_file_bytes(str(f), *(lambda s: (s.st_mtime_ns, s.st_size))(f.stat()))
        h.update(rel.encode("utf-8") + b"\0" + data + b"\0")
    hub: tuple[tuple[str, str, str], ...] = getattr(st, "hub_files", ())
    for repo, rev, path in sorted(hub):
        h.update(f"hub:{repo}@{rev}:{path}\0".encode())
    h.update(
        json.dumps({"image": image, "python": python, "env": dict(sorted(env_values.items()))}, sort_keys=True).encode()
    )
    return "closure-" + h.hexdigest()


def _image_identity() -> str:
    """The job image digest keyed into every clean-tree stage token (ADR-129 S5.12): the in-job
    ``SILLY_KICKS_IMAGE`` (set by `launch.job_env` to the prebuilt-image ref), else the prebuilt-image ref from
    the lock. `image` is stdlib-only, so this import never pulls a visibility concept into the trainer closure."""
    from scripts.hf_jobs import image

    return os.environ.get("SILLY_KICKS_IMAGE") or image.image_ref()


@functools.cache
def _is_compliant(pipeline_name: str) -> bool:
    """Whether ``pipeline_name`` is a compliant pipeline's ``PIPELINE.name`` (`_pipeline_population.compliant`)."""
    from scripts._pipeline_population import compliant

    names = set()
    for stem in compliant():
        try:
            names.add(importlib.import_module(f"scripts.{stem}").PIPELINE.name)
        except Exception:  # noqa: S112 -- a pipeline that fails to import is not this run's pipeline
            continue
    return pipeline_name in names


#: A local corpus file at or below this size enters the dataset identity by CONTENT digest (small
#: metadata such as ``home_teams.json`` can change without changing size); a larger file by name + size.
_CONTENT_DIGEST_MAX_BYTES = 1 << 20


def _listing_row(root: pathlib.Path, p: pathlib.Path) -> list:
    size = p.stat().st_size
    digest = _sha(p.read_bytes()) if size <= _CONTENT_DIGEST_MAX_BYTES else None
    return [p.relative_to(root).as_posix(), size, digest]


def _dataset_identity(
    params: Mapping[str, Any], data_root: pathlib.Path | None, *, param: str = "dataset"
) -> dict | None:
    """``{"name", "repo", "revision"}`` from ``datasets.lock.json`` for a declared dataset; for a local
    dev/test corpus (no lock record), a digest of the corpus file listing: every file's name and size,
    plus the content digest of every file up to ``_CONTENT_DIGEST_MAX_BYTES``."""
    name = params.get(param)
    if not name and param != "dataset":
        return None  # an optional second dataset left unset reads nothing (never the whole data root)
    if name:
        from scripts.hf_jobs.datasets import locked

        try:
            rec = locked(str(name))
            return {"name": str(name), "repo": rec.get("repo"), "revision": rec["revision"]}
        except KeyError:
            if os.environ.get("SILLY_KICKS_IN_HF_JOB"):  # in a job every dataset is recorded; never guess
                raise
    if data_root is None:
        return None
    root = pathlib.Path(data_root) / str(name) if name else pathlib.Path(data_root)
    rows = sorted(_listing_row(root, p) for p in root.rglob("*") if p.is_file())
    return {"name": str(name or "local"), "listing_sha256": _sha(_canon_json(rows))}


def make_context(
    pipeline: Pipeline, params: Mapping[str, Any], prov: Mapping[str, Any], *, store: Store, data_root
) -> RunContext:
    pipeline.validate()
    code_id = str(prov.get("code_identity") or prov["commit"])
    others = sorted({st.dataset_param for st in pipeline.stages if st.uses_dataset and st.dataset_param != "dataset"})
    stage_codes = _clean_tree_stage_codes(pipeline, prov)
    inputs, tokens = compute_tokens(
        pipeline,
        params,
        code_id,
        _dataset_identity(params, data_root),
        dataset_identities={name: _dataset_identity(params, data_root, param=name) for name in others},
        stage_codes=stage_codes,
    )
    return RunContext(pipeline, dict(params), dict(prov), store, data_root, inputs, tokens)


def _clean_tree_stage_codes(pipeline: Pipeline, prov: Mapping[str, Any]) -> dict[str, str] | None:
    """ADR-129: the per-stage closure identity on a CLEAN tree (``None`` on a dirty tree -> every stage keeps the
    ``dirty-<tree hash>`` identity). A compliant pipeline's stage absent from the lock in an HF job refuses the
    run (the lock is stale); outside a job (toy / test pipelines) the stage keeps the commit identity."""
    if prov.get("tree_state") != "clean":
        return None
    from scripts.hf_jobs import env_contract

    # CCTP-IMPL-15/26: the token must be IDENTICAL on the local controller (`--plan`) and the HF job. An
    # output-affecting env value DOES enter the token (S5.9) -- a value that changes a result byte
    # (`GHOST_FORCE_NUMPY` -> ghost-KDE 1e-9; `COORDINATION_REFERENCE_NUMERICS`) must key it. The controller and
    # the job agree because: the Python version is the fixed IMAGE constant (never the local interpreter);
    # `OPEN_DATA_REVISION` is EXCLUDED here (the launcher sets it to the dataset lock revision -- `launch.py`
    # `env[...] = d.revision` -- which is ALREADY the dataset identity's `revision`, so it keys the token
    # through the dataset identity, not twice); and the controller runs `--plan` under
    # `env_contract.controller_plan_env` (the output switches STRIPPED, `campaign.read_plan`), so an operator's
    # shell switch cannot move the token. The controller therefore never keys the token on an output switch, and
    # the job must not set one either (`launch.job_env` refuses it, CCTP-IMPL-27), or the two sides diverge.
    env_values = {
        k: v
        for k, v in env_contract.output_values(os.environ).items()
        if k != "SILLY_KICKS_OPEN_DATA_REVISION"  # keyed via the dataset identity (launch sets it to d.revision)
    }
    image = _image_identity()
    python = _IMAGE_PYTHON
    codes: dict[str, str] = {}
    for st in pipeline.stages:
        sid = stage_code_identity(pipeline, st, root=_REPO_ROOT, env_values=env_values, image=image, python=python)
        if sid is None:
            if os.environ.get("SILLY_KICKS_IN_HF_JOB") and _is_compliant(pipeline.name):
                from scripts.hf_jobs.closures import ClosureViolationError

                raise ClosureViolationError(
                    f"{pipeline.name}/{st.name} is absent from the closure lock (stale lock); "
                    f"run `python -m scripts.hf_jobs closures record`"
                )
            continue
        codes[st.name] = sid
    return codes


def _encode(result: Any) -> tuple[str, bytes, dict[str, bytes]]:
    if isinstance(result, FileSet):
        manifest = {"files": {n: _sha(b) for n, b in sorted(result.files.items())}}
        return ".fileset.json", _canon_json(manifest), dict(result.files)
    if hasattr(result, "to_parquet"):
        assert_storable(result)
        buf = io.BytesIO()
        result.to_parquet(buf, index=False)
        return ".parquet", buf.getvalue(), {}
    if result is None:
        import pandas as pd

        buf = io.BytesIO()
        pd.DataFrame().to_parquet(buf, index=False)
        return ".parquet", buf.getvalue(), {}
    # NOT `_canon_json`: a JSON result keeps its key order (deterministic code builds it in a deterministic
    # order), so a value read back equals the value written -- order included. Sorting here reordered every
    # downstream dict (a model's params, a metrics block) away from the legacy bytes.
    return ".json", (json.dumps(result, separators=(",", ":"), allow_nan=True) + "\n").encode("utf-8"), {}


def write_result(store: Store, base: str, result: Any) -> str:
    """Persist one result at ``base`` (+ suffix). Checksum sidecar FIRST, primary bytes LAST."""
    suffix, data, extra = _encode(result)
    for other in _RESULT_SUFFIXES:
        if other != suffix:
            store.delete(base + other)
    for name, payload in sorted(extra.items()):
        store.write_bytes(f"{base}.files/{name}", payload)
    digest = _sha(data)
    store.write_bytes(base + ".sha256", f"{digest}\n".encode("ascii"))
    store.write_bytes(base + suffix, data)
    return digest


def _present_suffix(names: set[str], stem: str) -> str | None:
    hits = [sfx for sfx in _RESULT_SUFFIXES if f"{stem}{sfx}" in names]
    return hits[0] if hits else None


class UnitStates(dict):
    """``{key: state}`` plus the generation listing it was computed from (``names``), so a caller that
    reads results next reuses ONE listing (INV-11)."""

    names: frozenset[str] = frozenset()


def _read_verified(store: Store, base: str, suffix: str) -> tuple[bytes, dict[str, bytes]] | None:
    """The result bytes and (for a FileSet) its member bytes, each read ONCE and checked against its digest;
    None when any part is absent or fails its digest."""
    data = store.read_bytes(base + suffix)
    side = store.read_bytes(base + ".sha256")
    if data is None or side is None or _sha(data) != side.decode("ascii").strip():
        return None
    members: dict[str, bytes] = {}
    if suffix == ".fileset.json":
        for name, want in json.loads(data)["files"].items():
            payload = store.read_bytes(f"{base}.files/{name}")
            if payload is None or _sha(payload) != want:
                return None
            members[name] = payload
    return data, members


def _verified(store: Store, base: str, suffix: str) -> bool:
    return _read_verified(store, base, suffix) is not None


def _decode(suffix: str, data: bytes, members: Mapping[str, bytes]) -> Any:
    if suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(io.BytesIO(data))
    if suffix == ".json":
        return json.loads(data)
    return FileSet(dict(members))


def read_result(store: Store, base: str, *, names: Collection[str] | None = None) -> Any:
    """The verified result at ``base``, or ``MISSING``. ``names`` is the caller's listing of the result's
    directory; without it the directory is listed here (one listing per call)."""
    rel_dir, _, stem = base.rpartition("/")
    suffix = _present_suffix(set(names) if names is not None else store.names(rel_dir), stem)
    if suffix is None:
        return MISSING
    got = _read_verified(store, base, suffix)
    return MISSING if got is None else _decode(suffix, *got)


def quarantine_corrupt(store: Store, gen: str, key: str, names: Collection[str]) -> None:
    """Move every file of unit ``key`` (result, sidecar, FileSet members) under ``<gen>/.corrupt/<utc>/`` --
    copy, then delete -- so the unit is ``pending`` again (INV-3) and the bytes stay for diagnosis."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = f"{gen}/{CORRUPT_DIR}/{stamp}"
    own = [n for n in names if n == f"{key}.sha256" or any(n == f"{key}{sfx}" for sfx in _RESULT_SUFFIXES)]
    sfx = _present_suffix(set(names), key)
    if sfx == ".fileset.json":
        raw = store.read_bytes(f"{gen}/{key}{sfx}")
        for member in json.loads(raw)["files"] if raw else ():
            rel = f"{key}.files/{member}"
            payload = store.read_bytes(f"{gen}/{rel}")
            if payload is not None:
                store.write_bytes(f"{dest}/{rel}", payload)
                store.delete(f"{gen}/{rel}")
    for name in own:
        payload = store.read_bytes(f"{gen}/{name}")
        if payload is not None:
            store.write_bytes(f"{dest}/{name}", payload)
        store.delete(f"{gen}/{name}")


def read_results(
    store: Store, gen: str, keys: Sequence[str], *, names: Collection[str], cap: int
) -> tuple[dict[str, Any], list[str], int]:
    """Read and verify every result of ``keys`` through a bounded thread pool (``cap`` readers, each under the
    store read gate). Returns the results in ``keys`` order -- never completion order (INV-1) -- the keys
    whose bytes failed their digest, and the TOTAL bytes read (the reduce record's ``bytes_read``, S5.2)."""
    import concurrent.futures

    from scripts._dataset_reads import store_read_slot

    listing = set(names)

    def one(k: str) -> tuple[str, Any, int]:
        sfx = _present_suffix(listing, k)
        if sfx is None:
            return k, MISSING, 0
        with store_read_slot():
            got = _read_verified(store, f"{gen}/{k}", sfx)
        if got is None:
            return k, MISSING, 0
        data, members = got
        return k, _decode(sfx, data, members), len(data) + sum(len(v) for v in members.values())

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(cap))) as ex:
        rows = {k: (res, nbytes) for k, res, nbytes in ex.map(one, keys)}
    corrupt = [k for k in keys if rows[k][0] is MISSING]
    results = {k: rows[k][0] for k in keys if rows[k][0] is not MISSING}
    return results, corrupt, sum(nbytes for _, nbytes in rows.values())


#: A joined key is a file stem: letters, digits, '_' and '-' only. No '.', so `<key>.json` (a result) can
#: never collide with another key's `<key>.counters.json` / `.ledger.json` / `.failed.json` sidecar.
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")


def sorted_units(ctx: RunContext, st: AnyStage) -> list[str]:
    keys = [join_key(k) for k in st.units(ctx)]
    bad = [k for k in keys if k.startswith("_")]
    if bad:
        raise ValueError(f"{st.name}: keys {bad[:3]} start with '_', which is reserved for stage files")
    odd = [k for k in keys if not _KEY_RE.match(k)]
    if odd:
        raise ValueError(f"{st.name}: keys {odd[:3]} contain characters outside [A-Za-z0-9_-]")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{st.name}: units() is not injective")
    return sorted(keys)


def profile_sample(keys: Sequence[str], k: int | None) -> list[str]:
    """The K evenly spaced units a profile job runs (spec 5.5), single-sourced: the executor runs exactly
    these and a ``download`` job fetches exactly these (`datasets.fetch`)."""
    if not keys:
        return []
    k = max(1, min(int(k or 1), len(keys)))
    return sorted({keys[(i * len(keys)) // k] for i in range(k)})


def unit_states(
    ctx: RunContext, st: AnyStage, keys: Sequence[str], *, verify: bool, names: Collection[str] | None = None
) -> UnitStates:
    """``done`` | ``excluded`` | ``failed`` | ``pending`` per key, plus the listing (``.names``). ``verify``
    re-hashes every present result (the in-job resume check, INV-3); a failing one is deleted with a warning
    and is pending. ``names`` is a listing the caller already holds (`plan`, Task 5); without it the
    generation is listed here. Without ``verify`` (the controller's plan) presence of result + sidecar
    counts as done."""
    gen = ctx.generation(st.name)
    listing = set(names) if names is not None else ctx.store.names(gen)
    out = UnitStates()
    deleted = False
    for k in keys:
        suffix = _present_suffix(listing, k)
        if suffix is not None and f"{k}.sha256" in listing:
            if not verify or _verified(ctx.store, f"{gen}/{k}", suffix):
                out[k] = "done"
                continue
            warnings.warn(f"checksum failure for {gen}/{k}{suffix}; deleting it and redoing the unit", stacklevel=2)
            for sfx in (suffix, ".sha256"):
                ctx.store.delete(f"{gen}/{k}{sfx}")
            deleted = True
        if f"{k}.excluded.json" in listing:
            out[k] = "excluded"
        elif f"{k}.failed.json" in listing:
            out[k] = "failed"
        else:
            out[k] = "pending"
    out.names = frozenset(ctx.store.names(gen) if deleted else listing)
    return out


def _reset_peak_rss() -> bool:
    """Reset the kernel's peak-RSS high-water mark for this process (Linux ``clear_refs``)."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


def _peak_rss(reset_ok: bool) -> tuple[int | None, str]:
    """(bytes, scope): ``unit`` after a successful reset, else ``process`` (a monotone upper bound)."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024, ("unit" if reset_ok else "process")
    except OSError:
        pass
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore[attr-defined]
        return int(ru * (1 if sys.platform == "darwin" else 1024)), "process"
    except (ImportError, AttributeError):
        return None, "unavailable"


def _previous_attempts(store: Store, base: str) -> int:
    """The unit's attempts so far: its ledger record, or its failure record when a kill came before any
    ledger write (a worker killed over its memory cap), whichever counts more."""
    counts = [0]
    for suffix in (".ledger.json", ".failed.json"):
        raw = store.read_bytes(base + suffix)
        if raw:
            counts.append(int(json.loads(raw).get("attempt", 0)))
    return max(counts)


def record_memory_failure(store: Store, gen: str, key: str, reason: str) -> int:
    """Record unit ``key`` as FAILED for memory (its worker was killed over its cap, so the unit could
    write nothing itself). The controller escalates such a unit to a job of its own with the whole box.
    Returns the attempt number recorded."""
    base = f"{gen}/{key}"
    attempt = _previous_attempts(store, base) + 1
    store.write_bytes(
        base + ".failed.json", _canon_json({"error": f"MemoryError: {reason}", "attempt": attempt, "memory": True})
    )
    return attempt


def deferred_memory_units(ctx: RunContext, st: AnyStage, keys: Sequence[str]) -> set[str]:
    """The ``keys`` whose last failure was the memory cap and that have failed fewer than
    `UNIT_ATTEMPTS` times: they wait for an ``--escalate`` job (Reviewer B CCHC-IMPL-12)."""
    gen = ctx.generation(st.name)
    out = set()
    for k, v in unit_states(ctx, st, keys, verify=False).items():
        if v != "failed":
            continue
        rec = _failure_record(ctx.store, gen, k)
        if rec.get("memory") and rec["attempt"] < UNIT_ATTEMPTS:
            out.add(k)
    return out


#: `pandas.api.types.infer_dtype` labels of an object column that MAY mix value types (an int-and-float mix,
#: ``mixed-integer-float``, stores as double). "mixed" also covers nested lists / arrays / dicts, which parquet
#: stores fine, so a candidate is confirmed by the conversion parquet itself runs.
_MIXED = frozenset({"mixed", "mixed-integer"})


def _mixes_types(s: Any) -> bool:
    """True for an object column parquet cannot store because its values mix types (e.g. int and str ids).
    Exact: a column `infer_dtype` flags is converted the way `to_parquet` converts it."""
    import pandas as pd

    if s.dtype != object or pd.api.types.infer_dtype(s, skipna=True) not in _MIXED:
        return False
    import pyarrow as pa

    try:
        pa.array(s, from_pandas=True)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
        return True
    return False


def storable_id_column(s: Any) -> Any:
    """An id column a parquet result can store. Unchanged when its values share one type; when they mix
    (gradientsports int ids with idsse / skillcorner str ids in one corpus), the canonical strings of
    `silly_kicks.id_compat.canonical_id_series` (missing stays missing) -- exactly what the legacy
    in-memory combines grouped on (``np.array(ids.tolist())`` and ``astype(str)`` both give ``'10502'``)."""
    if not _mixes_types(s):
        return s
    from silly_kicks.id_compat import canonical_id_series

    return canonical_id_series(s)


def assert_storable(frame: Any) -> None:
    """Refuse a stage result whose columns mix value types, BY NAME, before parquet fails with an opaque
    ``ArrowInvalid`` -- in the reduce, after every unit ran (real HF, Cycle 1)."""
    mixed = [str(c) for c in frame.columns if _mixes_types(frame[c])]
    if mixed:
        raise TypeError(
            f"stage result column(s) {mixed} mix value types (e.g. int and str ids from different providers), "
            "which parquet cannot store: combine unit frames with `_stages.combine_frames(ids=...)` naming the id "
            "columns, and fix any other column at its producer"
        )


def combine_frames(parts: Sequence[Any], *, ids: Sequence[str]) -> Any:
    """THE combine of unit frames into a stage output (ADR-128). Concatenated in the given order; every
    declared id column present is made storable (`storable_id_column`); any other column that mixes types
    refuses by name (`assert_storable`). A multi-provider corpus mixes int and str ids, so every reduce that
    concatenates unit frames goes through here. No part: an empty frame."""
    import pandas as pd

    if not parts:
        return pd.DataFrame()
    out = pd.concat(list(parts), ignore_index=True)
    for c in ids:
        if c in out.columns:
            out[c] = storable_id_column(out[c])
    assert_storable(out)
    return out


def _run_item(
    ctx: RunContext, st: ItemStage, key: str, ref: Any, *, post_work: Callable[[], None] | None = None
) -> tuple[str, str | None]:
    from scripts._driver import for_each

    root = ctx.store.local_path(ctx.stage_dir(st.name))
    if root is None:
        raise RuntimeError("an ItemStage executes only on a local (or mounted) store")
    loader = st.load
    load = (lambda r: loader(r, ctx)) if loader is not None else None

    def _work(item):  # post_work gates the shard write: a non-closure import in work() fails the item (CCTP-IMPL-16)
        out = st.work(item, ctx)
        if post_work is not None:
            post_work()
        return out

    res = for_each(
        [ref],
        key=st.key,
        load=load,
        work=_work,
        counters=st.counters,
        shard_root=root,
        token_inputs=ctx.token_inputs[st.name],
        tag=st.name,
        label=st.name,
        max_consecutive_failures=2,  # one item per call: a failure is recorded, never an abort
    )
    if res.shard_dir.name != ctx.tokens[st.name]:
        raise RuntimeError(f"{st.name}: for_each generation {res.shard_dir.name} != stage token {ctx.tokens[st.name]}")
    if res.failures:
        return "failed", str(next(iter(res.failures.values())))
    if res.exclusions:
        return "excluded", None
    return "done", None


def run_unit(
    ctx: RunContext,
    st: AnyStage,
    key: str,
    *,
    worker: str,
    pool: Any = None,
    ref: Any = None,
    post_work: Callable[[], None] | None = None,
) -> tuple[str, dict]:
    """Run ONE unit, persist its result (or exclusion / failure) and its ledger record. Returns
    ``(outcome, ledger record)`` -- the executor writes the record into its per-job cost aggregate (S5.11).

    ``post_work`` runs AFTER ``work`` returns but BEFORE the result is written (ADR-129 CCTP-IMPL-16: the
    fail-closed module check must gate the write, so a unit that loaded a non-closure module -- cached or not --
    is ``failed`` with NO result file)."""
    from scripts.hf_jobs.phases import publish_first_unit

    gen = ctx.generation(st.name)
    base = f"{gen}/{key}"
    attempt = _previous_attempts(ctx.store, base) + 1
    reset_ok = _reset_peak_rss()
    publish_first_unit(ctx.store)  # once per attempt: the job phase record (S7.2), before the first unit times
    started = time.time_ns()
    outcome, error = "done", None
    unit_ctx = dataclasses.replace(ctx, pool=pool)
    try:
        if isinstance(st, ItemStage):
            outcome, error = _run_item(unit_ctx, st, key, ref, post_work=post_work)
        else:
            result = st.work(key, unit_ctx)
            if post_work is not None:
                post_work()  # gate the write: a non-closure module loaded in work() fails the unit, no result
            write_result(ctx.store, base, result)
    except ItemExcluded as exc:
        from scripts._driver import write_exclusion

        local = ctx.store.local_path(gen)
        if local is None:
            raise
        local.mkdir(parents=True, exist_ok=True)
        write_exclusion(local, key, exc)
        outcome = "excluded"
    except MemoryError:
        outcome, error = "failed", "MemoryError: unit exceeded its memory cap"
    except Exception as exc:  # broad BY DESIGN: recorded per unit, never swallowed (the for_each rule)
        outcome, error = "failed", f"{type(exc).__name__}: {exc}"
    finished = time.time_ns()
    if outcome == "failed":
        ctx.store.write_bytes(base + ".failed.json", _canon_json({"error": error, "attempt": attempt}))
    else:
        ctx.store.delete(base + ".failed.json")
    peak, scope = _peak_rss(reset_ok)
    record = {
        "stage": st.name,
        "key": key,
        "outcome": outcome,
        "error": error,
        "worker": worker,
        "attempt": attempt,
        "job_label": os.environ.get("SILLY_KICKS_JOB_LABEL"),
        "started_ns": started,
        "finished_ns": finished,
        "wall_s": (finished - started) / 1e9,
        "peak_rss_bytes": peak,
        "rss_scope": scope,
        "inner_workers": getattr(pool, "workers", None),
        "inner_completion": list(getattr(pool, "drain_log", lambda: [])()),
    }
    ctx.store.write_bytes(base + ".ledger.json", _canon_json(record))
    return outcome, record


def output_present(ctx: RunContext, name: str, *, verify: bool, names: Collection[str] | None = None) -> bool:
    gen = ctx.generation(name)
    listing = set(names) if names is not None else ctx.store.names(gen)
    suffix = _present_suffix(listing, _OUTPUT)
    if suffix is None:
        return False
    return (not verify) or _verified(ctx.store, f"{gen}/{_OUTPUT}", suffix)


def reduce_stage(ctx: RunContext, name: str) -> None:
    """Write the stage output from every unit result, in ``reduce_order``. Reads and verifies EVERY result
    first (one listing, one read per file, INV-11); writes the summary, inputs and output only when all pass.
    A corrupt result moves to ``.corrupt/`` and its unit is pending again (INV-3). Idempotent and atomic, so
    two jobs that both see a complete stage may both reduce: they write identical bytes."""
    from scripts._dataset_reads import store_read_cap

    st = ctx.pipeline.stage(name)
    keys = sorted_units(ctx, st)
    states = unit_states(ctx, st, keys, verify=False)
    pending = [k for k, v in states.items() if v == "pending"]
    failed = [k for k, v in states.items() if v == "failed"]
    if pending:
        raise StageIncompleteError(f"{ctx.pipeline.name}/{name}: {len(pending)} unit(s) pending, e.g. {pending[:3]}")
    if st.max_failures is not None and len(failed) > st.max_failures:
        raise StageIncompleteError(
            f"{ctx.pipeline.name}/{name}: {len(failed)} unit(s) failed (tolerance {st.max_failures}): {failed[:5]}"
        )
    gen = ctx.generation(name)
    done = sorted((k for k, v in states.items() if v == "done"), key=st.reduce_order or (lambda k: k))
    started = time.time_ns()
    results, corrupt, bytes_read = read_results(ctx.store, gen, done, names=states.names, cap=store_read_cap() or 4)
    if corrupt:
        for k in corrupt:
            warnings.warn(f"checksum failure for {gen}/{k}; moving it to {CORRUPT_DIR}/ (re-run)", stacklevel=2)
            quarantine_corrupt(ctx.store, gen, k, states.names)
        raise StageIncompleteError(f"{ctx.pipeline.name}/{name}: corrupt result(s) {corrupt[:5]} moved; re-run them")
    excluded = {}
    for k in keys:
        if states[k] == "excluded":
            raw = ctx.store.read_bytes(f"{gen}/{k}.excluded.json")
            excluded[k] = json.loads(raw)["reason"] if raw else "unknown"
    summary = {
        "stage": name,
        "token": ctx.tokens[name],
        "n_units": len(keys),
        "n_done": len(done),
        "n_excluded": len(excluded),
        "n_failed": len(failed),
        "failed_keys": failed,
        "excluded": excluded,
    }
    ctx.store.write_bytes(f"{gen}/_summary.json", _canon_json(summary))
    # The job's own record of WHAT it read (dataset name/repo/revision, params, upstream tokens): `adopt`
    # checks the recorded dataset revision against the pin (ADR-128 spec 5.7 step 4). Never a unit key.
    ctx.store.write_bytes(f"{gen}/_inputs.json", _canon_json(ctx.token_inputs[name]))
    output = st.reduce(results, ctx) if st.reduce is not None else summary
    write_result(ctx.store, f"{gen}/{_OUTPUT}", output)
    write_reduce_record(ctx, name, started_ns=started, n_results=len(results), bytes_read=bytes_read)


def write_reduce_record(ctx: RunContext, name: str, *, started_ns: int, n_results: int, bytes_read: int = 0) -> None:
    """The S5.2 reduce ledger record: wall, results read, BYTES read, the read cap, ``in_job`` (False only in
    a separate reduce job, E-4). A no-op outside a job (a laptop, CI): no identity, no record."""
    from scripts._dataset_reads import store_read_cap
    from scripts.hf_jobs import phases

    ident = phases.identity()
    if ident is None:
        return
    finished = time.time_ns()
    rec = {
        "stage": name,
        "token": ctx.tokens[name],
        "started_ns": started_ns,
        "finished_ns": finished,
        "wall_s": (finished - started_ns) / 1e9,
        "n_results": n_results,
        "bytes_read": bytes_read,
        "read_cap": store_read_cap() or 4,
        "in_job": os.environ.get(phases.ACTION_KIND_ENV, "run") != "reduce",
    }
    ctx.store.write_bytes(phases.reduce_rel(ctx.pipeline.name, name, ctx.tokens[name], *ident), _canon_json(rec))


def ordered_values(results: Mapping[str, Any]) -> list:
    """Values in sorted-key order -- the ONLY way an inner combination may iterate pool results."""
    return [results[k] for k in sorted(results)]


def materialize_fileset(ctx: RunContext, stage: str, key: str, dest: pathlib.Path) -> pathlib.Path:
    """Copy a verified FileSet (unit ``key``, or the stage output when ``key == "_output"``) to local
    ``dest`` -- e.g. so a worker can memory-map a persisted design matrix. Returns ``dest``."""
    res = read_result(ctx.store, f"{ctx.generation(stage)}/{key}")
    if not isinstance(res, FileSet):
        raise TypeError(f"{stage}/{key} is not a FileSet result")
    for name, data in res.files.items():
        p = pathlib.Path(dest) / pathlib.PurePosixPath(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return pathlib.Path(dest)


def item_counters(ctx: RunContext, stage: str) -> tuple[dict, int]:
    """An ItemStage's ADR-052 counters merged in sorted key order (the `_driver._merge_counters` rule),
    and the number of done units whose sidecar is missing (`n_counters_unrecorded`)."""
    from scripts._driver import _merge_counters

    st = ctx.pipeline.stage(stage)
    gen = ctx.generation(stage)
    totals: dict = {}
    missing = 0
    for key, state in unit_states(ctx, st, sorted_units(ctx, st), verify=False).items():
        if state != "done":
            continue
        raw = ctx.store.read_bytes(f"{gen}/{key}.counters.json")
        if raw is None:
            missing += 1
            continue
        _merge_counters(totals, json.loads(raw), key)
    return totals, missing


def _history_key(token_inputs: Mapping) -> str:
    """Decision P14 (single-sourced with `hf_jobs.ledger.history_key`, which delegates here): costs are
    keyed by WHAT a stage computes, never by the commit or the upstream tokens."""
    keep = {k: token_inputs.get(k) for k in ("pipeline", "stage", "params", "declared", "dataset")}
    return _sha(json.dumps(keep, sort_keys=True).encode("utf-8"))[:16]


def _failure_record(store: Store, gen: str, key: str) -> dict:
    raw = store.read_bytes(f"{gen}/{key}.failed.json")
    rec = dict(json.loads(raw)) if raw else {}
    rec["attempt"] = int(rec.get("attempt", 1))
    return rec


def _ready(ctx: RunContext, st: AnyStage, names_of: Callable[[str], Collection[str] | None] | None = None) -> bool:
    get = names_of or (lambda _n: None)
    return all(output_present(ctx, n, verify=False, names=get(n)) for n in st.needs)


def plan(ctx: RunContext) -> dict:
    """The JSON the launcher reads (spec 5.1): per stage token, unit counts, done/pending, readiness.
    Lists each generation directory AT MOST once and reads no result bytes (INV-11): the listing is shared
    between the readiness check, the output presence check and the unit states."""
    stages = []
    listings: dict[str, frozenset[str]] = {}

    def names_of(stage_name: str) -> frozenset[str]:
        g = ctx.generation(stage_name)
        if g not in listings:
            listings[g] = frozenset(ctx.store.names(g))
        return listings[g]

    for st in ctx.pipeline.stages:
        gen = ctx.generation(st.name)
        own = names_of(st.name)
        rec: dict[str, Any] = {
            "name": st.name,
            "needs": list(st.needs),
            "token": ctx.tokens[st.name],
            "generation": gen,
            "inner_parallel": st.inner_parallel,
            "inner_width": getattr(st, "inner_width", 1),
            "history_key": _history_key(ctx.token_inputs[st.name]),
            "item_stage": isinstance(st, ItemStage),
            "ready": _ready(ctx, st, names_of),
            "reduced": output_present(ctx, st.name, verify=False, names=own),
            "n_units": None,
            "done": 0,
            "excluded": 0,
            "failed": 0,
            "pending": None,
            "units": None,
            "pending_units": None,
            "failed_units": None,
            "memory_units": None,
            # the stage's declared failure tolerance (None = unlimited): spent failures within it reduce
            "max_failures": st.max_failures,
        }
        try:
            keys = sorted_units(ctx, st)
        except UpstreamNotReducedError:
            keys = None
        if keys is not None:
            states = unit_states(ctx, st, keys, verify=False, names=own)
            failed = {k: _failure_record(ctx.store, gen, k) for k in keys if states[k] == "failed"}
            rec.update(
                n_units=len(keys),
                done=sum(v == "done" for v in states.values()),
                excluded=sum(v == "excluded" for v in states.values()),
                failed=sum(v == "failed" for v in states.values()),
                pending=sum(v == "pending" for v in states.values()),
                units=keys,
                # the controller schedules THESE (canonical order), never the first `pending` of `units`
                pending_units=[k for k in keys if states[k] == "pending"],
                # {key: recorded attempts}: the controller retries a failed unit below its attempt limit
                failed_units={k: rec.get("attempt", 1) for k, rec in failed.items()},
                # failed units whose LAST failure was the memory cap: escalated to a job of their own
                memory_units=[k for k, rec in failed.items() if rec.get("memory")],
            )
        stages.append(rec)
    return {
        "pipeline": ctx.pipeline.name,
        "module": ctx.pipeline.module,
        "output": ctx.pipeline.output,
        "params": {k: (str(v) if isinstance(v, pathlib.PurePath) else v) for k, v in ctx.params.items()},
        "code_identity": ctx.prov.get("code_identity"),
        "stages": stages,
    }


def export_output(ctx: RunContext, dest: pathlib.Path) -> None:
    """Write the pipeline output (a FileSet) into ``dest`` (``--run-all --out``)."""
    if ctx.pipeline.output is None:
        raise ValueError(f"{ctx.pipeline.name} declares no output stage")
    out = ctx.upstream(ctx.pipeline.output)
    if not isinstance(out, FileSet):
        raise TypeError(f"{ctx.pipeline.name}: output stage must return a FileSet, got {type(out).__name__}")
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in out.files.items():
        _write_bytes_atomically(dest / name, data)
    # The stage output is COMMIT-FREE (S5.10), so a model bundle has no provenance keys and no SHA256SUMS, and
    # `load()` refuses it. A LOCAL export is a deliverable, not a shared store entry: stamp the copy with THIS
    # run's provenance (a dirty tree stamps run_tree_dirty=true) through the same core `adopt` uses. Only a
    # bundle (a metadata.json / metrics.json output) is stamped; the store output itself stays commit-free.
    from scripts.hf_jobs import identity

    if any(name in identity.NORMALIZED for name in out.files):
        prov = ctx.prov
        identity.stamp_provenance(
            dest,
            commit=str(prov.get("commit") or "unknown"),
            tree_state=prov.get("tree_state") or "unknown",
            platform=prov.get("platform"),
            training_platform=ctx.params.get("training_platform"),
        )


def _jsonable(params: Mapping[str, Any]) -> dict:
    return {k: (str(v) if isinstance(v, pathlib.PurePath) else v) for k, v in params.items()}


def _read_through_json(store: Any) -> dict | None:
    """The read-through descriptor for a `CanaryReadThroughStore` (ADR-129 S5.10), so a worker rebuilds the same
    store instead of a plain LocalStore that would lose the main-store fallback (CCTP-IMPL-29). None otherwise."""
    main = getattr(store, "main", None)
    allow = getattr(store, "allow", None)
    if main is None or allow is None:
        return None
    main_root = getattr(main, "local_path", lambda _r: None)("")
    if main_root is None:
        return None
    return {"root": str(main_root), "allow": sorted(allow)}


def _wrap_read_through(local: Any, rt: dict | None) -> Any:
    """Rebuild a `CanaryReadThroughStore` over ``local`` from a `_read_through_json` descriptor (worker side)."""
    if not rt:
        return local
    from scripts.hf_jobs.store import CanaryReadThroughStore

    return CanaryReadThroughStore(local, LocalStore(pathlib.Path(rt["root"])), allow=set(rt["allow"]))


def context_to_json(ctx: RunContext) -> str:
    root = ctx.store.local_path("")
    if root is None:
        raise ValueError("only a local store can be handed to workers")
    return json.dumps(
        {
            "module": ctx.pipeline.module,
            "pipeline": ctx.pipeline.name,
            "params": _jsonable(ctx.params),
            "param_types": {p.name: ("path" if p.type is pathlib.Path else "plain") for p in ctx.pipeline.params},
            "stage_modules": {st.name: st.work.__module__ for st in ctx.pipeline.stages},
            "output": getattr(ctx.pipeline, "output", None),
            "prov": ctx.prov,
            "store_root": str(root),
            "read_through": _read_through_json(ctx.store),
            "data_root": None if ctx.data_root is None else str(ctx.data_root),
            "token_inputs": ctx.token_inputs,
            "tokens": ctx.tokens,
        },
        sort_keys=True,
    )


def context_from_json(text: str) -> RunContext:
    d = json.loads(text)
    pipeline = importlib.import_module(d["module"]).PIPELINE
    params = dict(d["params"])
    for p in pipeline.params:  # restore declared types (argparse already converted them once)
        if params.get(p.name) is not None and p.type is pathlib.Path:
            params[p.name] = pathlib.Path(params[p.name])
    data_root = None if d["data_root"] is None else pathlib.Path(d["data_root"])
    store = _wrap_read_through(LocalStore(pathlib.Path(d["store_root"])), d.get("read_through"))
    return RunContext(
        pipeline,
        params,
        d["prov"],
        store,
        data_root,
        d["token_inputs"],
        d["tokens"],
    )


@dataclasses.dataclass(frozen=True)
class PipelineView:
    """What a stage process needs of a pipeline, WITHOUT importing the trainer (S5.9): the name (store paths),
    the module (recorded only), the params and the stages the running stage reaches.

    The view carries the running stage AND the stages its work reads (an upstream ItemStage whose per-unit
    results a reduce-like stage re-reads via ``ctx.pipeline.stage(...)``, e.g. census ``prefit`` ->
    ``fit_prepass``). For a REUSABLE stage those module-mates are all reusable (the stage module's ``STAGES``),
    so the trainer still never loads; for a fit-side stage the trainer IS in the closure, so the full pipeline
    is used. A one-stage view would ``KeyError`` on such a sibling read, so the view is never narrower than the
    stages reached."""

    name: str
    module: str
    params: tuple[Param, ...]
    stages: tuple[AnyStage, ...]
    output: str | None = None

    def stage(self, name: str) -> AnyStage:
        for st in self.stages:
            if st.name == name:
                return st
        raise KeyError(name)


def stage_context_from_json(text: str, stage: str) -> tuple[RunContext, AnyStage]:
    """Build the ``RunContext`` and the ONE running stage from the JSON alone (S5.9). A reusable stage's module
    exports ``STAGES`` (no trainer import); a fit-side stage's module is the trainer (in its own closure)."""
    d = json.loads(text)
    work_mod = None
    try:
        work_mod = importlib.import_module(d["stage_modules"][stage])
    except ModuleNotFoundError:
        # An inline (test) stage has no importable module -> fall through to d["module"]. A real error INSIDE an
        # existing stage module is NOT swallowed (it propagates), so a stage-module bug is never hidden (IMPL-23).
        work_mod = None
    if work_mod is not None and stage in getattr(work_mod, "STAGES", {}):
        stages = tuple(work_mod.STAGES.values())  # the trainer's reusable set, declared together; no trainer import
        st = work_mod.STAGES[stage]
    else:
        # A fit-side stage (its own module IS the trainer, in its closure) or an ad-hoc test pipeline: rebuild
        # from the pipeline's own module. A REUSABLE stage never reaches here, so the trainer never loads for it.
        pipe = importlib.import_module(d["module"]).PIPELINE
        stages = tuple(pipe.stages)
        st = pipe.stage(stage)
    params = dict(d["params"])
    for name, tag in d["param_types"].items():
        if params.get(name) is not None and tag == "path":
            params[name] = pathlib.Path(params[name])
    view = PipelineView(d["pipeline"], d["module"], (), stages, output=d.get("output"))
    data_root = None if d["data_root"] is None else pathlib.Path(d["data_root"])
    store = _wrap_read_through(LocalStore(pathlib.Path(d["store_root"])), d.get("read_through"))
    ctx = RunContext(view, params, d["prov"], store, data_root, d["token_inputs"], d["tokens"])
    return ctx, st


def prefetch_hub_files(stages, *, fetch) -> None:
    """Download every stage's declared ``hub_files`` (ADR-129): the job runs this WITH network before the
    offline run line, so a worker loads the file from the HF cache under ``HF_HUB_OFFLINE=1``."""
    for st in stages:
        for repo, rev, path in getattr(st, "hub_files", ()):
            fetch(repo, rev, path)


def build_parser(pipeline: Pipeline) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"python -m {pipeline.module}",
        description=f"{pipeline.name}: ADR-128 stage-contract pipeline ({', '.join(s.name for s in pipeline.stages)}).",
    )
    for p in pipeline.params:
        if p.flag:
            ap.add_argument(p.cli, dest=p.name, action="store_true", help=p.help)
        else:
            ap.add_argument(
                p.cli, dest=p.name, type=p.type, default=p.default, required=p.required, choices=p.choices, help=p.help
            )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="print the plan JSON (stages, tokens, unit counts) and exit")
    mode.add_argument("--stages", help="comma-separated stages to run in order; only the first may be sharded")
    mode.add_argument("--run-all", action="store_true", help="run the whole DAG locally (the serial parity reference)")
    ap.add_argument(
        "--store", required=True, help="store root: a directory, or hf://buckets/<ns>/<b>/<prefix> (--plan only)"
    )
    ap.add_argument("--data-root", type=pathlib.Path, default=None, help="where the dataset is mounted or unpacked")
    ap.add_argument("--shard", default="0/1", help="I/N: this job's disjoint slice of the first stage's units")
    ap.add_argument(
        "--shard-keys",
        type=pathlib.Path,
        default=None,
        help="JSON list of unit keys: this job runs exactly these units of its ONE stage (planner LPT part)",
    )
    ap.add_argument(
        "--read-through",
        default=None,
        help="a canary: read an allowed upstream generation from this MAIN store (CanaryReadThroughStore, ADR-129)",
    )
    ap.add_argument(
        "--read-through-allow",
        type=pathlib.Path,
        default=None,
        help="JSON list of generations the --read-through store may serve (attested on HEAD/ancestor, not quarantined)",
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--escalate",
        action="store_true",
        help="also run units deferred by a memory failure (the controller's W=1, whole-box escalation job)",
    )
    ap.add_argument("--inner-workers", type=int, default=1, help="pool size for an inner_parallel stage")
    ap.add_argument("--cap-bytes", type=int, default=None, help="per-worker memory cap (planner)")
    ap.add_argument("--peak-rss-bytes", type=int, default=None, help="per-unit peak RSS hint for worker sizing")
    ap.add_argument(
        "--profile", type=int, default=None, help="profile mode: run K evenly spaced units at W=1 then W=max"
    )
    ap.add_argument("--out", type=pathlib.Path, default=None, help="--run-all only: write the output FileSet here")
    ap.add_argument(
        "--allow-dirty", action="store_true", help="permit a dirty tree (dev only; recorded, never laundered)"
    )
    ap.add_argument(
        "--prefetch-hub",
        action="store_true",
        help="fetch the --stages' declared hub_files into the HF cache (WITH network), then exit (ADR-129)",
    )
    return ap


def _hf_hub_download(repo: str, rev: str, path: str) -> None:
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id=repo, revision=rev, filename=path)


def _parse_shard(text: str) -> tuple[int, int]:
    i, _, n = text.partition("/")
    shard = (int(i), int(n))
    if shard[1] < 1 or not (0 <= shard[0] < shard[1]):
        raise ValueError(f"--shard must be I/N with 0 <= I < N, got {text!r}")
    return shard


def main(pipeline: Pipeline, argv: Sequence[str] | None = None) -> int:
    """The uniform CLI every migrated trainer and driver exposes (spec 5.1)."""
    args = build_parser(pipeline).parse_args(argv)
    params = {p.name: getattr(args, p.name) for p in pipeline.params}
    if getattr(args, "prefetch_hub", False):
        names = [n for n in (args.stages or "").split(",") if n] or [st.name for st in pipeline.stages]
        prefetch_hub_files([pipeline.stage(n) for n in names], fetch=_hf_hub_download)
        return 0
    from scripts._provenance import code_identity, env_provenance, git_provenance, require_clean_tree

    git = git_provenance()
    if not args.plan:
        require_clean_tree(git, allow_dirty=args.allow_dirty)
    prov = {**git, "env": env_provenance(), "code_identity": code_identity(git)}
    store = open_store(args.store)
    if args.read_through:  # ADR-129 canary: read an allowed upstream generation from the main store
        from scripts.hf_jobs.store import CanaryReadThroughStore

        allow = (
            set(json.loads(args.read_through_allow.read_text(encoding="utf-8"))) if args.read_through_allow else set()
        )
        store = CanaryReadThroughStore(store, open_store(args.read_through), allow=allow)
    ctx = make_context(pipeline, params, prov, store=store, data_root=args.data_root)
    if args.read_through:  # a canary runs ONE unit: only with --shard-keys or a single-unit stage (S5.10)
        names = [n for n in (args.stages or "").split(",") if n]
        single = len(names) == 1 and len(sorted_units(ctx, pipeline.stage(names[0]))) <= 1
        if not (args.shard_keys or single):
            raise SystemExit("--read-through requires --shard-keys or a single-unit stage (ADR-129 canary)")
    if args.plan:
        print(json.dumps(plan(ctx), sort_keys=True))
        return 0
    from scripts._parallel_launch import RefusalError
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor, StageFailedError

    ex = LocalExecutor(
        ExecutorConfig(
            workers=args.workers,
            inner_workers=args.inner_workers,
            cap_bytes=args.cap_bytes,
            peak_rss_bytes=args.peak_rss_bytes,
            profile_units=args.profile,
            escalate=args.escalate,
        )
    )
    try:
        if args.run_all:
            for st in pipeline.stages:
                ex.run_stage(ctx, st.name, shard=(0, 1))
                ex.require_reduced(ctx, st.name)
            if args.out is not None:
                export_output(ctx, args.out)
            return 0
        names = [n for n in args.stages.split(",") if n]
        shard = _parse_shard(args.shard)
        if len(names) > 1 and shard[1] != 1:
            raise SystemExit("--stages with more than one stage requires --shard 0/1 (only a single stage fans out)")
        keys = None
        if args.shard_keys is not None:
            if len(names) != 1 or shard != (0, 1):
                raise SystemExit("--shard-keys runs exactly ONE stage and excludes --shard (it IS the job's slice)")
            keys = json.loads(args.shard_keys.read_text(encoding="utf-8"))
        for i, name in enumerate(names):
            ex.run_stage(ctx, name, shard=shard if i == 0 else (0, 1), keys=keys if i == 0 else None)
            ex.try_reduce(ctx, name)
        return 0
    except RefusalError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except (StageFailedError, StageIncompleteError) as exc:
        print(f"STAGE FAILED: {exc}", file=sys.stderr)
        return 3
