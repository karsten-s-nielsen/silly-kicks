"""The stage/unit compute contract (ADR-128).

WHY THIS EXISTS. The 4.138.0 re-fit campaign ran 5-9 hour jobs on one of eight vCPUs, from uncommitted
scratchpad orchestration, because the per-trainer phase flags (ADR-123) had untested real-HF paths. This
module is the ONE contract every heavy trainer and corpus driver declares instead: a static DAG of
stages, each a set of independent, keyed, single-threaded units (unless a stage opts into its vCPU share with
`thread_safe=True` behind a byte-identity gate, ADR-128 amendment) plus an optional reducer. The executor
(`scripts/hf_jobs/executor.py`) places units as W worker processes per job x N jobs, and every output is
byte-identical for every (W, N) (INV-1) because:

1. a unit is single-threaded (the executor starts workers with `_thread_pin.deterministic_env`), UNLESS its
   stage sets `thread_safe=True` behind a committed cross-thread-count byte-identity test (ADR-128 amendment);
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
import atexit
import dataclasses
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
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, NoReturn, Protocol

from scripts._driver import _token, _write_bytes_atomically, _write_bytes_exclusively, join_key
from scripts._item_outcome import ItemExcluded
from scripts._thread_pin import RUNTIME_THREADS_VAR

#: Numeric key components are zero-padded to this width, so lexicographic key order == numeric order.
KEY_WIDTH = 4
_RESULT_SUFFIXES = (".parquet", ".json", ".fileset.json")
_OUTPUT = "_output"
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
    ``work(key, ctx)`` is pure, deterministic and single-threaded (unless the stage opts in with
    ``thread_safe=True``, ADR-128 amendment) and returns ONE result: a DataFrame,
    a JSON-native value or a ``FileSet``. ``reduce(results, ctx)`` receives ``{key: result}`` in
    ``reduce_order`` order (default: sorted keys) and returns the stage output. An ``inner_parallel``
    unit runs alone at the top level of a job and receives ``ctx.pool`` (an ``InnerPool``); every
    combination of inner results MUST iterate ``ordered_values`` (sorted inner keys).

    ``params`` names the pipeline parameters the stage's numbers depend on; ``declare(params)`` adds
    any further declared inputs (the ADR-052 rule: a token closes silent omission, not mis-declaration).
    ``cross_commit_version`` opts the stage into resume across commits (decision P9); default None
    means the run commit is part of the token. ``max_failures`` (None = unlimited) is the failed-unit
    tolerance: the executor aborts when it is exceeded and the reduce refuses. ``abort_first_k``: the
    executor aborts when the first k units that complete in this job have all failed (systematic bug).
    """

    name: str
    units: Callable[[RunContext], Sequence[str]]
    work: Callable[[str, RunContext], Any]
    needs: tuple[str, ...] = ()
    reduce: Callable[[dict[str, Any], RunContext], Any] | None = None
    params: tuple[str, ...] = ()
    declare: Callable[[Mapping[str, Any]], Mapping[str, object]] | None = None
    cross_commit_version: str | None = None
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
    #: ADR-128 INV-2 opt-in (Cycle 1): when True the executor launches this stage's worker at the per-job
    #: thread budget instead of pinned to 1. ONLY opt in behind a committed cross-thread-count byte-identity
    #: test (`test_thread_identity`). Default False keeps every unit single-threaded.
    thread_safe: bool = False
    #: The thread lever the budget rides: ``"omp"`` (OpenMP ``prange`` kernels, e.g. HistGB) raises
    #: ``OMP_NUM_THREADS`` and keeps BLAS serial; ``"n_jobs"`` (e.g. xgb) keeps the env serial and exports
    #: ``SILLY_KICKS_RUNTIME_THREADS`` for the trainer (B38). Ignored unless ``thread_safe``.
    thread_lever: Literal["omp", "n_jobs"] = "omp"
    #: HFU T8: the concat id columns, set ONLY on a plain ``combine_frames`` concat reduce, which makes the
    #: reduce SHARDABLE. When a shardable stage's measured reduce wall exceeds the per-job budget the reduce
    #: is range-partitioned: each sub-reduce raw-concats its sorted-key range into a partial (empty-tolerant,
    #: no per-range raise), and the final reduce runs ``reduce`` over the partials so the stage's own empty
    #: policy applies ONCE, byte-identically. None = not shardable (a non-concat reduce stays whole).
    shardable_reduce: tuple[str, ...] | None = None


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
    cross_commit_version: str | None = None
    reduce_order: Callable[[str], Any] | None = None
    max_failures: int | None = 0
    abort_first_k: int = 3
    uses_dataset: bool = False
    #: The parameter naming the dataset this stage reads (``uses_dataset``); its identity keys the token.
    dataset_param: str = "dataset"
    inner_parallel: bool = dataclasses.field(default=False, init=False)
    #: ADR-128 INV-2 opt-in — see ``Stage.thread_safe``/``thread_lever``.
    thread_safe: bool = False
    thread_lever: Literal["omp", "n_jobs"] = "omp"
    #: HFU T8 — see ``Stage.shardable_reduce``. The concat id columns; None = the reduce is not shardable.
    shardable_reduce: tuple[str, ...] | None = None

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
    def write_if_absent(self, rel: str, data: bytes) -> bool: ...
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

    def write_if_absent(self, rel: str, data: bytes) -> bool:
        """Create the file only if absent (O_CREAT|O_EXCL); True when created, False when it exists. Atomic on a
        single POSIX filesystem; across containers on the bucket mount it is best-effort -- see
        ``_write_bytes_exclusively`` for the scope of the guarantee."""
        return _write_bytes_exclusively(self._p(rel), data)

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
    pipeline: Pipeline
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
) -> tuple[dict[str, dict], dict[str, str]]:
    """The declared token inputs and the token (generation name) of every stage, in DAG order.

    ``dataset_identity`` is the ``--dataset`` identity; ``dataset_identities`` maps any OTHER dataset
    parameter a stage reads (`Stage.dataset_param`, e.g. ``probe_dataset``) to its identity."""
    inputs: dict[str, dict] = {}
    tokens: dict[str, str] = {}
    for st in pipeline.stages:
        d: dict[str, object] = {
            "pipeline": pipeline.name,
            "stage": st.name,
            "code": st.cross_commit_version or code_id,
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
    inputs, tokens = compute_tokens(
        pipeline,
        params,
        code_id,
        _dataset_identity(params, data_root),
        dataset_identities={name: _dataset_identity(params, data_root, param=name) for name in others},
    )
    return RunContext(pipeline, dict(params), dict(prov), store, data_root, inputs, tokens)


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


def _verified(store: Store, base: str, suffix: str) -> bool:
    data = store.read_bytes(base + suffix)
    side = store.read_bytes(base + ".sha256")
    if data is None or side is None or _sha(data) != side.decode("ascii").strip():
        return False
    if suffix == ".fileset.json":
        for name, want in json.loads(data)["files"].items():
            payload = store.read_bytes(f"{base}.files/{name}")
            if payload is None or _sha(payload) != want:
                return False
    return True


def read_result(store: Store, base: str) -> Any:
    """The verified result at ``base``, or ``MISSING`` (absent or failing its checksum)."""
    rel_dir, _, stem = base.rpartition("/")
    suffix = _present_suffix(store.names(rel_dir), stem)
    if suffix is None or not _verified(store, base, suffix):
        return MISSING
    data = store.read_bytes(base + suffix)
    assert data is not None  # noqa: S101 -- _verified read it a moment ago
    if suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(io.BytesIO(data))
    if suffix == ".json":
        return json.loads(data)
    names = json.loads(data)["files"]
    return FileSet({n: store.read_bytes(f"{base}.files/{n}") or b"" for n in names})


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


def unit_states(ctx: RunContext, st: AnyStage, keys: Sequence[str], *, verify: bool) -> dict[str, str]:
    """``done`` | ``excluded`` | ``failed`` | ``pending`` per key. ``verify`` re-hashes every present
    result (the in-job resume check, INV-3); a failing one is deleted with a warning and is pending.
    Without ``verify`` (the controller's plan) presence of result + sidecar counts as done."""
    gen = ctx.generation(st.name)
    names = ctx.store.names(gen)
    out: dict[str, str] = {}
    for k in keys:
        suffix = _present_suffix(names, k)
        if suffix is not None and f"{k}.sha256" in names:
            if not verify or _verified(ctx.store, f"{gen}/{k}", suffix):
                out[k] = "done"
                continue
            warnings.warn(f"checksum failure for {gen}/{k}{suffix}; deleting it and redoing the unit", stacklevel=2)
            for sfx in (suffix, ".sha256"):
                ctx.store.delete(f"{gen}/{k}{sfx}")
        if f"{k}.excluded.json" in names:
            out[k] = "excluded"
        elif f"{k}.failed.json" in names:
            out[k] = "failed"
        else:
            out[k] = "pending"
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


def reduce_key_ranges(keys: Sequence[str], n: int) -> list[list[str]]:
    """Partition keys into at most ``n`` CONTIGUOUS ranges, PRESERVING the given order, for a sharded reduce
    (HFU T8). The caller supplies the keys in ``reduce_order`` (`reduce_done_order`), so an ordered concat of
    the per-range sub-concats equals the single reduce: the given order IS the reduce order and ``combine_frames``
    concatenates in it (``storable_id_column`` idempotent). Never more ranges than keys; an empty key set yields
    one empty range. It does NOT sort -- plain-sort boundaries would not align with a custom reduce_order and
    would reorder rows across shards (reviewer B D1-IMPL-08)."""
    ks = list(keys)
    if not ks:
        return [[]]
    n = max(1, min(int(n), len(ks)))
    per = -(-len(ks) // n)  # ceil division, no math import
    return [ks[i : i + per] for i in range(0, len(ks), per)]


def _run_item(ctx: RunContext, st: ItemStage, key: str, ref: Any) -> tuple[str, str | None]:
    from scripts._driver import for_each

    root = ctx.store.local_path(ctx.stage_dir(st.name))
    if root is None:
        raise RuntimeError("an ItemStage executes only on a local (or mounted) store")
    loader = st.load
    load = (lambda r: loader(r, ctx)) if loader is not None else None
    res = for_each(
        [ref],
        key=st.key,
        load=load,
        work=lambda item: st.work(item, ctx),
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


def run_unit(ctx: RunContext, st: AnyStage, key: str, *, worker: str, pool: Any = None, ref: Any = None) -> str:
    """Run ONE unit, persist its result (or exclusion / failure) and its ledger record. Returns the outcome."""
    gen = ctx.generation(st.name)
    base = f"{gen}/{key}"
    attempt = _previous_attempts(ctx.store, base) + 1
    reset_ok = _reset_peak_rss()
    started = time.time_ns()
    outcome, error = "done", None
    unit_ctx = dataclasses.replace(ctx, pool=pool)
    try:
        if isinstance(st, ItemStage):
            outcome, error = _run_item(unit_ctx, st, key, ref)
        else:
            write_result(ctx.store, base, st.work(key, unit_ctx))
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
        "platform": ctx.prov.get("platform"),  # the fit host, in run-history not the artifact (D3)
        "started_ns": started,
        "finished_ns": finished,
        "wall_s": (finished - started) / 1e9,
        "peak_rss_bytes": peak,
        "rss_scope": scope,
        "inner_workers": getattr(pool, "workers", None),
        "inner_child_peak_rss_bytes": getattr(pool, "child_peak_rss_bytes", None),  # HFU T11/B-19: size the pool
        "inner_completion": list(getattr(pool, "drain_log", lambda: [])()),
        # ADR-128 INV-2: the applied per-worker thread budget (1 for a pinned stage; the budget for a
        # thread_safe one). Read from this worker's own env -- either lever resolves to one number.
        "thread_budget": int(os.environ.get(RUNTIME_THREADS_VAR) or os.environ.get("OMP_NUM_THREADS") or 1),
    }
    ctx.store.write_bytes(base + ".ledger.json", _canon_json(record))
    return outcome


def repair_missing_ledgers(store: Store, gen: str, done_keys: Iterable[str], *, stage: str) -> list[str]:
    """D7: a unit whose result is present and verified but whose ``.ledger.json`` was lost to a kill between the
    two writes gets a repair record now -- outcome ``done`` with an UNKNOWN wall (``wall_s`` null), so resume and
    the summary count the unit without inventing a cost. Returns the repaired keys. Idempotent: a unit that
    already has a ledger is left untouched.

    The write is create-only (``write_if_absent``), so a straggler that writes the REAL ledger between the
    ``done_keys`` snapshot and this write keeps it: on a single POSIX filesystem the null-wall repair record
    never clobbers a measured one (reviewer B B14). The production race is cross-container on the bucket mount,
    where that is best-effort, not a proof -- see ``_write_bytes_exclusively`` for the scope. The ``names``
    pre-filter only skips the obvious already-present case cheaply."""
    names = store.names(gen)
    repaired: list[str] = []
    for k in done_keys:
        if f"{k}.ledger.json" in names:
            continue
        record = {
            "stage": stage,
            "key": k,
            "outcome": "done",
            "error": None,
            "worker": None,
            "attempt": 0,
            "job_label": os.environ.get("SILLY_KICKS_JOB_LABEL"),
            "started_ns": None,
            "finished_ns": None,
            "wall_s": None,
            "peak_rss_bytes": None,
            "rss_scope": None,
            "inner_workers": None,
            "inner_completion": [],
            "platform": None,  # the fit host is unknown: the record was lost before this repair
            "repaired": True,
        }
        if store.write_if_absent(f"{gen}/{k}.ledger.json", _canon_json(record)):
            repaired.append(k)
    return repaired


def output_present(ctx: RunContext, name: str, *, verify: bool) -> bool:
    gen = ctx.generation(name)
    suffix = _present_suffix(ctx.store.names(gen), _OUTPUT)
    if suffix is None:
        return False
    return (not verify) or _verified(ctx.store, f"{gen}/{_OUTPUT}", suffix)


def _order_done(states: Mapping[str, str], st: AnyStage) -> list[str]:
    """The DONE unit keys in the stage's ``reduce_order`` (default: sorted keys). The SINGLE source of the reduce
    order, shared by the single reduce and every sharded sub-reduce, so both select the same keys and concatenate
    them in the same order (reviewer B D1-IMPL-08)."""
    return sorted((k for k, v in states.items() if v == "done"), key=st.reduce_order or (lambda k: k))


def reduce_done_order(ctx: RunContext, name: str) -> list[str]:
    """The DONE unit keys of a stage in ``reduce_order``, from the SAME verify=True scan the single reduce uses
    (`_write_reduce_metadata`). A sharded sub-reduce selects and orders its inputs through this, so it skips
    excluded/failed/pending keys (reviewer B D1-IMPL-09) and matches the single reduce byte-for-byte."""
    st = ctx.pipeline.stage(name)
    keys = sorted_units(ctx, st)
    states = unit_states(ctx, st, keys, verify=True)
    return _order_done(states, st)


def _write_reduce_metadata(ctx: RunContext, name: str) -> tuple[dict, list[str]]:
    """Verify the stage is complete and write its ``_summary.json`` + ``_inputs.json`` provenance sidecars;
    return the summary dict and the sorted done keys. SHARED by ``reduce_stage`` and ``reduce_from_partials``,
    so a sharded reduce leaves byte-identical provenance beside its byte-identical output (HFU T8). ``adopt``
    reads ``_inputs.json`` to check the recorded dataset revision against the pin (ADR-128 spec 5.7 step 4), so
    the sharded path must write it too."""
    st = ctx.pipeline.stage(name)
    keys = sorted_units(ctx, st)
    states = unit_states(ctx, st, keys, verify=True)
    # D7: a unit done in a prior job whose ledger was lost to a kill is `done` here (its result verifies) but
    # absent from any later job's shard, so the reduce -- which sees EVERY unit -- is where its record is repaired.
    repair_missing_ledgers(ctx.store, ctx.generation(name), [k for k, v in states.items() if v == "done"], stage=name)
    pending = [k for k, v in states.items() if v == "pending"]
    failed = [k for k, v in states.items() if v == "failed"]
    if pending:
        raise StageIncompleteError(f"{ctx.pipeline.name}/{name}: {len(pending)} unit(s) pending, e.g. {pending[:3]}")
    if st.max_failures is not None and len(failed) > st.max_failures:
        raise StageIncompleteError(
            f"{ctx.pipeline.name}/{name}: {len(failed)} unit(s) failed (tolerance {st.max_failures}): {failed[:5]}"
        )
    gen = ctx.generation(name)
    excluded = {}
    for k in keys:
        if states[k] == "excluded":
            raw = ctx.store.read_bytes(f"{gen}/{k}.excluded.json")
            excluded[k] = json.loads(raw)["reason"] if raw else "unknown"
    done = _order_done(states, st)
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
    return summary, done


def reduce_stage(ctx: RunContext, name: str) -> None:
    """Write the stage output from every unit result, in ``reduce_order``. Idempotent and atomic, so
    two jobs that both see a complete stage may both reduce: they write identical bytes."""
    st = ctx.pipeline.stage(name)
    summary, done = _write_reduce_metadata(ctx, name)
    gen = ctx.generation(name)
    results = {k: read_result(ctx.store, f"{gen}/{k}") for k in done}
    output = st.reduce(results, ctx) if st.reduce is not None else summary
    write_result(ctx.store, f"{gen}/{_OUTPUT}", output)


def reduce_subrange(ctx: RunContext, name: str, shard_id: int, n_shards: int) -> None:
    """HFU T8: sub-reduce shard ``shard_id`` of ``n_shards`` -- raw-concat its contiguous range of the stage's
    DONE units, taken in ``reduce_order`` through `reduce_done_order` (the SAME selection and order the single
    reduce uses), so the sharded path is byte-identical and never reads an excluded/missing unit (reviewer B
    D1-IMPL-08/09). The controller passes only the COUNT; the worker owns the partition. Empty-TOLERANT: a shard
    with no range (``n_shards`` > done count) or an empty range writes an empty partial with NO per-range raise
    (the stage's empty policy applies ONCE at the final reduce). Written to ``{gen}/_subreduce/{shard_id}.parquet``."""
    st = ctx.pipeline.stage(name)
    if st.shardable_reduce is None:  # () is a valid shardable declaration (a plain concat with NO id columns)
        raise StageIncompleteError(f"{ctx.pipeline.name}/{name}: not a shardable reduce (no shardable_reduce ids)")
    ranges = reduce_key_ranges(reduce_done_order(ctx, name), n_shards)
    rng = ranges[shard_id] if shard_id < len(ranges) else []
    gen = ctx.generation(name)
    frames = [read_result(ctx.store, f"{gen}/{k}") for k in rng]
    part = combine_frames([f for f in frames if len(f)], ids=st.shardable_reduce)
    write_result(ctx.store, f"{gen}/_subreduce/{shard_id:04d}", part)
    # HFU T8 / reviewer B D1-IMPL-19: stamp the shard COUNT this partial was computed for. A changed reduce_wall
    # re-partitions the keys, so a partial from an earlier count is stale; the final reduce refuses on a mismatch
    # rather than silently concatenating a subset. The `_meta_` stem is not a digit, so `_subreduce_present` skips it.
    ctx.store.write_bytes(f"{gen}/_subreduce/_meta_{shard_id:04d}.json", _canon_json({"n_shards": int(n_shards)}))


def reduce_from_partials(ctx: RunContext, name: str, n_shards: int) -> None:
    """HFU T8: the FINAL reduce of a sharded stage -- run the stage's OWN reduce over the sub-reduce partials
    (in sorted shard order), so the stage's empty policy applies once and the output is byte-identical to the
    single reduce. Idempotent/atomic like ``reduce_stage``. Refuses (loud) if any partial was computed for a
    DIFFERENT shard count (reviewer B D1-IMPL-19): a changed reduce_wall re-partitions the keys, so reducing a
    stale partial set would drop or duplicate keys. The refusal is a ``StageIncompleteError`` -- the job ends in
    error and, after ``error_resumes`` such jobs, the instance HALTS and surfaces the stale shard count. The
    controller does NOT auto-clear the stale partials (it re-emits only missing shard indices and never reads
    the meta); recovery is manual -- clear ``{gen}/_subreduce/`` and let the sub-reduces re-run. The guard's job
    is to make the mismatch LOUD so no stale subset is ever concatenated silently (reviewer B D1-IMPL-19/23)."""
    st = ctx.pipeline.stage(name)
    summary, _done = _write_reduce_metadata(ctx, name)  # byte-identical provenance beside the sharded output
    gen = ctx.generation(name)
    for i in range(n_shards):
        raw = ctx.store.read_bytes(f"{gen}/_subreduce/_meta_{i:04d}.json")
        meta = json.loads(raw) if raw else None
        if not meta or int(meta.get("n_shards", -1)) != int(n_shards):
            raise StageIncompleteError(
                f"{ctx.pipeline.name}/{name}: sub-reduce partial {i} was computed for a different shard count "
                f"(meta {meta}, expected n_shards={n_shards}); re-run the sub-reduces"
            )
    partials = {f"{i:04d}": read_result(ctx.store, f"{gen}/_subreduce/{i:04d}") for i in range(n_shards)}
    output = st.reduce(partials, ctx) if st.reduce is not None else summary
    write_result(ctx.store, f"{gen}/{_OUTPUT}", output)


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


def _ready(ctx: RunContext, st: AnyStage) -> bool:
    return all(output_present(ctx, n, verify=False) for n in st.needs)


def _subreduce_present(store: Any, gen: str) -> list[int]:
    """HFU T8: the shard indices whose sub-reduce partial is written under ``{gen}/_subreduce/`` -- the
    controller's state for the two-phase sharded reduce (it emits the missing shards, then the final concat
    once all are present)."""
    names = store.names(f"{gen}/_subreduce")
    stems = {nm.split(".", 1)[0] for nm in names}
    return sorted(int(s) for s in stems if s.isdigit() and _present_suffix(names, s))


def plan(ctx: RunContext) -> dict:
    """The JSON the launcher reads (spec 5.1): per stage token, unit counts, done/pending, readiness."""
    stages = []
    for st in ctx.pipeline.stages:
        gen = ctx.generation(st.name)
        sr = getattr(st, "shardable_reduce", None)  # bound so the None-narrowing reaches the list() below
        rec: dict[str, Any] = {
            "name": st.name,
            "needs": list(st.needs),
            "token": ctx.tokens[st.name],
            "generation": gen,
            "inner_parallel": st.inner_parallel,
            "inner_width": getattr(st, "inner_width", 1),
            "thread_safe": st.thread_safe,
            "thread_lever": st.thread_lever,
            "shardable_reduce": None if sr is None else list(sr),
            "subreduce_present": _subreduce_present(ctx.store, gen),  # HFU T8: sub-reduce partials already written
            "history_key": _history_key(ctx.token_inputs[st.name]),
            "item_stage": isinstance(st, ItemStage),
            "ready": _ready(ctx, st),
            "reduced": output_present(ctx, st.name, verify=False),
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
            states = unit_states(ctx, st, keys, verify=False)
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


def _jsonable(params: Mapping[str, Any]) -> dict:
    return {k: (str(v) if isinstance(v, pathlib.PurePath) else v) for k, v in params.items()}


def context_to_json(ctx: RunContext) -> str:
    root = ctx.store.local_path("")
    if root is None:
        raise ValueError("only a local store can be handed to workers")
    return json.dumps(
        {
            "module": ctx.pipeline.module,
            "params": _jsonable(ctx.params),
            "prov": ctx.prov,
            "store_root": str(root),
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
    return RunContext(
        pipeline,
        params,
        d["prov"],
        LocalStore(pathlib.Path(d["store_root"])),
        data_root,
        d["token_inputs"],
        d["tokens"],
    )


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
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--escalate",
        action="store_true",
        help="also run units deferred by a memory failure (the controller's W=1, whole-box escalation job)",
    )
    ap.add_argument("--inner-workers", type=int, default=1, help="pool size for an inner_parallel stage")
    ap.add_argument("--cap-bytes", type=int, default=None, help="per-worker memory cap (planner)")
    ap.add_argument("--inner-cap-bytes", type=int, default=None, help="HFU T11: inner_parallel pool CHILD cap")
    ap.add_argument("--peak-rss-bytes", type=int, default=None, help="per-unit peak RSS hint for worker sizing")
    ap.add_argument(
        "--profile", type=int, default=None, help="profile mode: run K evenly spaced units at W=1 then W=max"
    )
    ap.add_argument("--subreduce", action="store_true", help="HFU T8: sub-reduce this shard's range of DONE units")
    ap.add_argument("--shard-id", type=int, default=None, help="HFU T8: this sub-reduce's shard index")
    ap.add_argument("--n-shards", type=int, default=None, help="HFU T8: this sub-reduce's shard COUNT")
    ap.add_argument("--from-partials", type=int, default=None, help="HFU T8: final reduce over N sub-reduce partials")
    ap.add_argument(
        "--no-inline-reduce", action="store_true", help="HFU T8: a sharded stage's run job skips the inline reduce"
    )
    ap.add_argument("--out", type=pathlib.Path, default=None, help="--run-all only: write the output FileSet here")
    ap.add_argument(
        "--allow-dirty", action="store_true", help="permit a dirty tree (dev only; recorded, never laundered)"
    )
    return ap


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
    from scripts._provenance import code_identity, env_provenance, git_provenance, require_clean_tree

    git = git_provenance()
    if not args.plan:
        require_clean_tree(git, allow_dirty=args.allow_dirty)
    prov = {**git, "env": env_provenance(), "code_identity": code_identity(git)}
    ctx = make_context(pipeline, params, prov, store=open_store(args.store), data_root=args.data_root)
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
            inner_cap_bytes=args.inner_cap_bytes,
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
        if args.from_partials is not None:  # HFU T8: the FINAL reduce over N sub-reduce partials
            reduce_from_partials(ctx, names[0], args.from_partials)
            return 0
        shard = _parse_shard(args.shard)
        if len(names) > 1 and shard[1] != 1:
            raise SystemExit("--stages with more than one stage requires --shard 0/1 (only a single stage fans out)")
        keys = None
        if args.shard_keys is not None:
            if len(names) != 1 or shard != (0, 1):
                raise SystemExit("--shard-keys runs exactly ONE stage and excludes --shard (it IS the job's slice)")
            keys = json.loads(args.shard_keys.read_text(encoding="utf-8"))
        if args.subreduce:  # HFU T8: sub-reduce this shard of DONE units (the worker owns the range; no inline reduce)
            reduce_subrange(ctx, names[0], args.shard_id or 0, args.n_shards or 1)
            return 0
        for i, name in enumerate(names):
            ex.run_stage(ctx, name, shard=shard if i == 0 else (0, 1), keys=keys if i == 0 else None)
            if not args.no_inline_reduce:  # HFU T8: a sharded stage's run job leaves the reduce to the sub-reduce jobs
                ex.try_reduce(ctx, name)
        return 0
    except RefusalError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except (StageFailedError, StageIncompleteError) as exc:
        print(f"STAGE FAILED: {exc}", file=sys.stderr)
        return 3


def hard_exit(rc: int) -> NoReturn:
    """Flush the std streams and hard-exit with ``rc`` after running the Python atexit handlers -- the shared
    exit for every stage-contract ``__main__`` (``cli_main`` and the executor reduce/worker child). A short
    ``python -m`` process that uses Arrow can abort at interpreter shutdown (``terminate called without an
    active exception``, SIGABRT) AFTER it has written its correct output; the HF controller marks ANY nonzero
    rc as a failed job, so that teardown abort would fail a correct job spuriously. ``atexit._run_exitfuncs``
    runs the registered Python handlers (e.g. a worker's temp-dir rmtree); the flush keeps the ``--plan`` JSON
    on stdout; ``os._exit`` then skips interpreter finalization -- the C++ static destructors that raise, and
    ALSO ``threading._shutdown`` (so no non-daemon pool or thread join runs through this exit; an ``InnerPool``
    is joined by ``worker_main``'s own ``finally`` before it returns, so that join completes before this exit
    is reached). Call ONLY from a ``__main__`` -- ``os._exit`` in-process would kill the test runner."""
    atexit._run_exitfuncs()  # the mechanism multiprocessing uses to run Python exit handlers before a hard exit
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


def cli_main(pipeline: Pipeline, argv: Sequence[str] | None = None) -> NoReturn:
    """Process entrypoint for a migrated pipeline module (spec 5.1): run ``main`` then ``hard_exit`` its rc, so
    a native-library C++ teardown abort at interpreter shutdown cannot turn a CORRECT run's exit code nonzero
    (observed on Linux CI: a correct ``--subreduce`` wrote its output, then Arrow aborted at teardown). Results
    are already persisted (atomic writes) before ``main`` returns, so the hard exit loses nothing; see
    ``hard_exit`` for the mechanism. ``main``'s rc reaches the exit unchanged (a raised acceptance gate still
    exits nonzero as before). Called ONLY from a module ``__main__`` (in-process ``os._exit`` kills pytest)."""
    hard_exit(main(pipeline, argv))
