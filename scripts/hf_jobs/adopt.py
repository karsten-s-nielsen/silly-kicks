"""Retrieve, verify, adopt and publish campaign artifacts (ADR-128 spec 5.7, decisions P3, P15, INV-9).

``retrieve`` copies each instance's reduced output FileSet out of the bucket store into a staging dir
OUTSIDE the repo, with a record of what the job read (the token inputs it wrote, the extract counters).
``verify`` loads every artifact in ``.venv-hfjobs`` (SHA256SUMS, chirality, feature contract), checks the
five declared expectations, certifies per the registry policy (`floor.certify`; `identity.compare` for a
rebaseline driver) and runs the ADR-126 integrity gate for xShot/xCross. ``adopt`` turns a verified
staging dir into a commit-ready diff -- wheel bundles, the manifest training commits (of the artifacts with a
``commit_field``), ``observed.json``,
re-rendered card regions, verdict JSONs, a CHANGELOG draft -- under the ADR-037 guards (clean tree at the
campaign commit, except the spec 7.1 expectation edit of the manifest, ADR-132 item 10; the manifest that
``verify`` checked; a diff limited to the allowlist), runs the full local gate and STOPS: it never commits.
``publish`` (an owner go every time) pushes Hub artifacts through the unchanged ADR-088 seam and verifies
what the Hub then holds; ``--sandbox`` targets private ``silly-kicks/sandbox-<name>`` repos.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeVar

from scripts.hf_jobs import manifest as mf
from scripts.hf_jobs.campaign import REPO_ROOT, Campaign, Instance, instances, rate_limited, read_plan, store_uri

HFJOBS_PYTHON = REPO_ROOT / ".venv-hfjobs" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VERDICTS = pathlib.Path("docs/research/trained_artifact_integrity/verdicts")
#: Every model family's loader (``module:Class``; ``Class.load(dir)`` raises on a SHA256SUMS, chirality or
#: feature-contract mismatch).
LOADERS: dict[str, str] = {
    "win_probability": "silly_kicks.win_probability._model:WinProbabilityModel",
    "pass_completion": "silly_kicks.expected_passing._model:PassCompletionModel",
    "xsuccess": "silly_kicks.xsuccess._model:XSuccessModel",
    "match_outcome_dependence": "silly_kicks.match_outcome._dependence:DependenceModel",
    "ghost_gk": "silly_kicks.tracking._ghost_gk:GhostGkModel",
    "ghost_outfield": "silly_kicks.tracking._ghost_outfield:GhostOutfieldModel",
    "xshot_occurrence": "silly_kicks.tracking._xshot_occurrence:XShotOccurrenceModel",
    "xcross_attempt": "silly_kicks.tracking._xcross_attempt:XCrossAttemptModel",
    "receiver_model": "silly_kicks.tracking._receiver:ReceiverModel",
}
#: The families whose ADR-126 integrity verdict is part of certification, with the extract counters
#: that carry the label-domain event count (shots / crosses).
INTEGRITY_FAMILIES: frozenset[str] = frozenset({"xshot_occurrence", "xcross_attempt"})
#: Decision P15: the commit the committed rebaseline output came from (the census: 4.120.0).
REBASELINE_FROM: dict[str, str] = {"build_territory_ranking_census": "dd7ed4f"}
#: Paths `adopt` may change besides the wheel destinations and the cards (the ADR-037 diff-scope guard).
ADOPT_PATHS: tuple[str, ...] = (
    "docs/artifacts/",
    "docs/research/trained_artifact_integrity/",
    "docs/research/hf_campaigns/",
)


class AdoptionRefusedError(RuntimeError):
    """An expectation, a guard or a certification refused the adoption (nothing was written)."""


class AdoptionRollbackError(AdoptionRefusedError):
    """``adopt`` failed in its write phase and could not restore every path (HFU spec 18.3); the message names
    them, and ``__cause__`` is the original error."""


# ---- shared ----------------------------------------------------------------------------------------


def _outside_repo(staging: pathlib.Path, root: pathlib.Path = REPO_ROOT) -> pathlib.Path:
    s = pathlib.Path(staging).resolve()
    if s.is_relative_to(root):
        raise AdoptionRefusedError(f"the staging dir {s} is inside the repo; use a dir outside it (spec 5.7 step 1)")
    return s


def artifacts_for(inst: Instance, man: mf.Manifest) -> list[mf.Artifact]:
    """The manifest artifacts a campaign instance produces (a wheel bundle and/or its Hub repo)."""
    return [a for a in man.artifacts.values() if (a.job, a.variant, a.tier) == (inst.job, inst.variant, inst.tier)]


def _read_json(p: pathlib.Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _record_path(staging: pathlib.Path, inst_id: str) -> pathlib.Path:
    return staging / f"{inst_id}.record.json"


def _artifact_dir(staging: pathlib.Path, inst_id: str) -> pathlib.Path:
    return staging / inst_id


def _final_dir(staging: pathlib.Path, inst_id: str) -> pathlib.Path:
    """``<staging>/_final/<instance>/``: the adopted copy of one instance. :func:`_finalize` writes it;
    :func:`compare_staging` reads it on the baseline side."""
    return staging / "_final" / inst_id


def _open_data_dir(staging: pathlib.Path, inst_id: str) -> pathlib.Path:
    """The open-data cache of one statsbomb_open head-to-head (HFU spec 17.5): one folder per instance, so no two
    concurrent verify workers download into one folder."""
    return staging / "_open_data" / inst_id


def _eval_rows_path(staging: pathlib.Path, inst_id: str) -> pathlib.Path:
    """The run's own extract rows of a staged-corpus public instance (HFU spec 14.4): beside the
    artifact dir, never inside it (adopt's bundle is the artifact dir only)."""
    d = _artifact_dir(staging, inst_id)
    return d.with_name(f"{d.name}.eval_rows.parquet")


def _dataset_kind_of(inst: Instance) -> str | None:
    """The declared kind of the instance's primary dataset (its first mount)."""
    return _declared()[inst.datasets[0].name].kind if inst.datasets else None


def _public_staged(inst: Instance, arm: str | None = None) -> bool:
    """A public-certified instance on a staged ``flat_tc3`` corpus: its head-to-head scores its own extract
    rows (HFU spec 14.4), which retrieve saves. ``arm`` is the entry's shipped arm (ADR-132,
    `manifest.effective_shipped_arm`); None reads the instance tier (retrieve, before any entry is chosen)."""
    from scripts.hf_jobs.registry import TrainerSpec, spec_for

    spec = spec_for(inst.job)
    return (
        isinstance(spec, TrainerSpec)
        and spec.certification_for(arm or inst.tier) == "public"
        and (_dataset_kind_of(inst) == "flat_tc3")
    )


# ---- retrieve --------------------------------------------------------------------------------------


def _counters(
    store: Any, stage: Mapping[str, Any], *, workers: int = 8, progress_s: float = 60.0, label: str = ""
) -> dict:
    """An ItemStage's per-unit counters merged in sorted key order (`_stages.item_counters`'s rule). HFU spec
    14.9: ONE listing of the generation (``store.names``); only the ``<key>.counters.json`` files present are read,
    in batches of ``_READ_BATCH`` (one `_read_many` call each; HFU launch top-up spec §22.2a) over ``workers``
    threads; the merge stays in sorted key order, so the totals are the serial ones.
    ``n_counters_unrecorded`` counts the units with neither a counters nor an excluded file in the listing.
    A long read prints ``[retrieve] <label> counters <k>/<units>`` at most every ``progress_s``."""
    import concurrent.futures

    from scripts._driver import _merge_counters

    gen = stage["generation"]
    keys = sorted(stage.get("units") or [])
    names = store.names(gen)
    present = [k for k in keys if f"{k}.counters.json" in names]
    missing = sum(1 for k in keys if f"{k}.counters.json" not in names and f"{k}.excluded.json" not in names)
    # HFU launch top-up spec §22.2a (F3): the files in batches of `_READ_BATCH`, one `_read_many` call each.
    rels = [f"{gen}/{k}.counters.json" for k in present]
    by_rel: dict[str, bytes | None] = {}
    last = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_read_many, store, rels[i : i + _READ_BATCH]) for i in range(0, len(rels), _READ_BATCH)]
        for fut in concurrent.futures.as_completed(futures):
            by_rel.update(fut.result())
            if time.monotonic() - last >= progress_s:
                print(f"[retrieve] {label} counters {len(by_rel)}/{len(keys)}", file=sys.stderr, flush=True)
                last = time.monotonic()
    raw = {k: by_rel[f"{gen}/{k}.counters.json"] for k in present}
    totals: dict = {}
    for k in present:  # sorted key order: the serial merge
        blob = raw[k]
        if blob is None:  # listed but gone: unrecorded, as the serial read counted it
            missing += f"{k}.excluded.json" not in names
            continue
        _merge_counters(totals, json.loads(blob), k)
    return {"totals": totals, "n_counters_unrecorded": missing, "n_read": len(present)}


#: The files of one `read_many` batch: one `download_bucket_files` call (HFU launch top-up spec §22.2a).
_READ_BATCH = 200


def _read_many(store: Any, rels: Sequence[str]) -> dict[str, bytes | None]:
    """``{rel: bytes or None}``: the store's ``read_many`` (one call for the batch) when it has one, else one
    ``read_bytes`` for each rel. The one place that chooses: `_counters` takes any store with ``read_bytes`` and
    ``names`` (a `LocalStore`, the test fakes)."""
    many = getattr(store, "read_many", None)
    if many is not None:
        return many(rels)
    return {rel: store.read_bytes(rel) for rel in rels}


#: The waits (s) of `retrieve` before each retry of a rate-limited call: 450 s in all, more than the 5-min window of
#: the account limit (2500 requests). HFU launch top-up spec §22.2: run 9b's retrieve crashed twice on a 429 from
#: the xet token that `huggingface_hub` does not retry.
_READ_BACKOFF_S: tuple[float, ...] = (30.0, 60.0, 120.0, 240.0)


_T = TypeVar("_T")


def _rate_limited_call(what: str, fn: Callable[[], _T], *, sleep: Callable[[float], object], bound: Any = None) -> _T:
    """``fn()`` under ``bound`` (when given). A rate-limited error (`campaign.rate_limited`) is retried after each
    ``_READ_BACKOFF_S`` wait, with one ``[retrieve] rate-limited`` line; the wait holds no slot of ``bound``. The
    last error, and any other error, raises."""
    n = len(_READ_BACKOFF_S)
    for i, wait in enumerate((*_READ_BACKOFF_S, None), start=1):
        try:
            if bound is None:
                return fn()
            with bound:
                return fn()
        except Exception as exc:
            if wait is None or not rate_limited(exc):
                raise
            print(f"[retrieve] rate-limited ({what}); retry {i}/{n} in {wait:g} s", file=sys.stderr, flush=True)
            sleep(wait)
    raise AssertionError("unreachable: the last wait is None")


class _BoundedReads:
    """A store whose ``read_bytes``, ``names`` and ``read_many`` (one call per batch) hold one shared semaphore (the
    read bound of `retrieve`) and wait out a rate limit (spec §22.2). Every other attribute is the wrapped store's."""

    def __init__(self, store: Any, bound: Any, *, sleep: Callable[[float], object] = time.sleep) -> None:
        self._store, self._bound, self._sleep = store, bound, sleep

    def read_bytes(self, rel: str) -> bytes | None:
        return _rate_limited_call(
            f"read {rel}", lambda: self._store.read_bytes(rel), sleep=self._sleep, bound=self._bound
        )

    def names(self, rel_dir: str) -> set[str]:
        return _rate_limited_call(
            f"names {rel_dir}", lambda: self._store.names(rel_dir), sleep=self._sleep, bound=self._bound
        )

    def read_many(self, rels: Sequence[str]) -> dict[str, bytes | None]:
        """One slot and one retry for the whole batch (spec §22.2a), around `_read_many` of the wrapped store."""
        return _rate_limited_call(
            f"read_many {len(rels)} files", lambda: _read_many(self._store, rels), sleep=self._sleep, bound=self._bound
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


def retrieve(
    c: Campaign,
    commit: str,
    *,
    staging: pathlib.Path,
    api: Any = None,
    replica: int | None = None,
    plan_fn: Callable[..., dict] = read_plan,
    store_fn: Callable[[str], Any] | None = None,
    workers: int = 4,
    read_workers: int = 8,
    sleep: Callable[[float], object] = time.sleep,
) -> dict[str, pathlib.Path]:
    """Copy every reduced instance output into ``<staging>/<run_key>/<variant>/`` (never into the repo),
    with ``<instance>.record.json`` beside it: the job's recorded token inputs per stage and the extract
    counters. An instance whose output stage is not reduced is skipped (and absent from the result). Up to
    ``workers`` instances at once (HFU spec 17.5): each ``[retrieve]`` line prints when its instance ends; the
    result is in instance order. At most ``read_workers`` store reads or listings run at once over ALL instances
    (HFU fit-units spec C1): 4 instances x 8 counter threads gave 32 concurrent reads and HTTP 429s. An
    instance's ``--plan`` call reads the store too (one request at a time), so it holds one slot of the bound.
    A rate-limited read, listing or plan call waits and retries (``_READ_BACKOFF_S``; HFU launch top-up spec §22.2);
    ``sleep`` is a test seam."""
    import concurrent.futures
    import threading

    from scripts.hf_jobs.store import BucketStore

    staging = _outside_repo(staging)
    make_store = store_fn or (lambda uri: dataclasses.replace(BucketStore.from_uri(uri), api=api))
    bound = threading.BoundedSemaphore(max(1, read_workers))

    def open_store(uri: str) -> Any:
        return _BoundedReads(make_store(uri), bound, sleep=sleep)

    def bounded_plan(inst: Instance, **kwargs: Any) -> dict:
        # a `--plan` subprocess reads the store one request at a time: it holds one slot
        return _rate_limited_call(f"plan {inst.id}", lambda: plan_fn(inst, **kwargs), sleep=sleep, bound=bound)

    insts = instances(c, commit, replica=replica)

    def one(item: tuple[int, Instance]) -> pathlib.Path | None:
        i, inst = item
        return _retrieve_one(
            inst, i, len(insts), c=c, commit=commit, staging=staging, plan_fn=bounded_plan, open_store=open_store
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        dests = list(pool.map(one, enumerate(insts, start=1)))
    return {inst.id: dest for inst, dest in zip(insts, dests, strict=True) if dest is not None}


def _retrieve_one(
    inst: Instance,
    i: int,
    n: int,
    *,
    c: Campaign,
    commit: str,
    staging: pathlib.Path,
    plan_fn: Callable[..., dict],
    open_store: Callable[[str], Any],
) -> pathlib.Path | None:
    """:func:`retrieve` for instance ``i`` of ``n``: its artifact dir, or ``None`` when its output stage is not
    reduced. Every file it writes is this instance's own (the artifact dir, the record, the eval rows)."""
    import pandas as pd

    from scripts._stages import FileSet, read_result

    t0 = time.monotonic()
    plan = plan_fn(inst, bucket=c.bucket)
    stages = {s["name"]: s for s in plan["stages"]}
    rec = stages[plan["output"]]
    if not rec["reduced"]:
        return None
    store = open_store(store_uri(c.bucket, inst.run_key))
    result = read_result(store, f"{rec['generation']}/_output")
    if not isinstance(result, FileSet):
        raise AdoptionRefusedError(f"{inst.id}: output stage {plan['output']!r} is not a FileSet")
    dest = _artifact_dir(staging, inst.id)
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in sorted(result.files.items()):
        if "/" in name or "\\" in name:
            raise AdoptionRefusedError(f"{inst.id}: nested output path {name!r} (an artifact is flat)")
        (dest / name).write_bytes(data)
    inputs = {}
    for name, s in stages.items():
        raw = store.read_bytes(f"{s['generation']}/_inputs.json")
        inputs[name] = json.loads(raw) if raw is not None else None
    record = {
        "instance": inst.id,
        "job": inst.job,
        "tier": inst.tier,
        "variant": inst.variant,
        "module": inst.module,
        "commit": commit,
        "output": plan["output"],
        "code_identity": plan.get("code_identity"),
        "token_inputs": inputs,
        "counters": _counters(store, stages["extract"], label=inst.id) if "extract" in stages else None,
    }
    if _public_staged(inst):  # the head-to-head's eval rows: the run's own extract output (HFU spec 14.4)
        rows = read_result(store, f"{stages['extract']['generation']}/_output")
        if not isinstance(rows, pd.DataFrame):
            raise AdoptionRefusedError(f"{inst.id}: the extract output is not a frame")
        rows.to_parquet(_eval_rows_path(staging, inst.id), index=False)
    p = _record_path(staging, inst.id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ctr = record["counters"] or {}
    units = len(stages["extract"].get("units") or []) if "extract" in stages else 0
    mb = sum(len(v) for v in result.files.values()) / 1e6
    print(  # HFU spec 14.9
        f"[retrieve] {i}/{n} {inst.id}: {len(result.files)} files, {mb:.1f} MB, "
        f"counters {ctr.get('n_read', 0)}/{units} in {time.monotonic() - t0:.1f} s",
        file=sys.stderr,
        flush=True,
    )
    return dest


# ---- verify ----------------------------------------------------------------------------------------


def _dir_reader(d: pathlib.Path) -> Callable[[str], dict]:
    return lambda fname: _read_json(d / fname)


def expectation_violations(a: mf.Artifact, fresh: pathlib.Path, record: Mapping[str, Any], commit: str) -> list[str]:
    """Spec 5.7 step 4, EXACTLY: the training commit is the campaign commit; the recorded dataset revision
    is the pinned one; the corpus digest + n (a public bundle is rehydrated only AFTER this check, step 5)
    equal the staging-time expectation; the shipped tier is the expected tier. Plus the manifest's
    declared ``expect`` values. Empty == compliant."""
    from scripts.hf_jobs.datasets import locked

    bad: list[str] = []
    corpus, naming = ("digest" if a.corpus == "public_ids" else a.corpus), a.naming
    if _dataset_kind(a) == "statsbomb_open":  # a fresh event-only artifact: the lock digest (HFU spec 14.3)
        corpus, naming = "digest", "open"
    fresh_a = dataclasses.replace(a, training_commit=commit, corpus=corpus, naming=naming)
    bad += mf.violations(fresh_a, read=_dir_reader(fresh))
    if a.dataset is not None and a.dataset in _declared():
        pinned = locked(a.dataset)["revision"]
        read = {
            (ti or {}).get("dataset", {}).get("revision")
            for ti in (record.get("token_inputs") or {}).values()
            if (ti or {}).get("dataset") and (ti or {})["dataset"].get("name") == a.dataset
        }
        if read != {pinned}:
            bad.append(f"{a.id}: the job read dataset {a.dataset} at {sorted(map(str, read))}, pinned {pinned}")
        snap = (locked(a.dataset).get("snapshot") or {}).get("revision")
        if snap is not None:  # the staged open-data copy the job read (HFU spec 14.7)
            seen = {
                (ti or {})["dataset"].get("snapshot")
                for ti in (record.get("token_inputs") or {}).values()
                if (ti or {}).get("dataset") and (ti or {})["dataset"].get("name") == a.dataset
            }
            if seen != {snap}:
                bad.append(f"{a.id}: the job read snapshot {sorted(map(str, seen))}, locked {snap}")
    for fname in ("metadata.json", "metrics.json"):
        p = fresh / fname
        if p.is_file():
            sv = _read_json(p).get("shipped_variant")
            want = mf.effective_shipped_arm(a)  # ADR-132: the shipped arm, not the channel
            if sv is not None and sv != want:
                bad.append(f"{a.id}: {fname} ships {sv!r}, the expected shipped arm is {want!r}")
    return bad


def _dataset_kind(a: mf.Artifact) -> str | None:
    d = _declared().get(a.dataset or "")
    return d.kind if d is not None else None


#: The ``expect`` keys that pin a corpus size (HFU spec 14.3): the last segment of the key path.
_COUNT_KEYS = ("n_games", "n_matches", "corpus_n_matches")


def corpus_scope(a: mf.Artifact) -> str | None:
    """How verify checks the corpus a certified artifact trained on (HFU spec 14.3), or ``None``: the
    manifest corpus rule, a count ``expect``, or the lock digest of a ``statsbomb_open`` dataset."""
    if a.corpus in ("public_ids", "digest"):
        return a.corpus
    if any(str(k).rsplit(".", 1)[-1].rsplit(":", 1)[-1] in _COUNT_KEYS for k in a.expect or {}):
        return "count"
    if _dataset_kind(a) == "statsbomb_open":
        return "statsbomb_open"
    return None


def _say_verify(i: int, n: int, inst_id: str, row: Mapping[str, Any], failed: Sequence[str], t0: float) -> None:
    """``[verify] <i>/<n> <instance> ok|FAIL (<failed checks>) in <s> s[ note=<coverage note>]`` (HFU spec 14.9)."""
    outcome = "ok" if not failed else f"FAIL ({', '.join(failed)})"
    note = f" note={row['coverage_note']}" if row.get("coverage_note") else ""
    print(f"[verify] {i}/{n} {inst_id} {outcome} in {time.monotonic() - t0:.1f} s{note}", file=sys.stderr, flush=True)


def coverage_note(*, cv_ok: bool) -> str:
    """The verify note of an ``sb_open`` instance (HFU spec 14.3): the committed event-only artifacts record
    no match ids, so the evidence is the count, the competition set and CV parity."""
    from scripts.hf_jobs.datasets import locked

    rec = locked("sb_open")
    n, k = rec["identity"]["open"]["corpus_n_matches"], len(rec["competitions"])
    return (
        "committed artifact records no match ids; faithful-migration evidence: "
        f"corpus_n_matches {n}, {k} competitions, CV parity {'ok' if cv_ok else 'FAIL'}"
    )


def _declared() -> Mapping[str, Any]:
    from scripts.hf_jobs.datasets import declared

    return declared()


def load_check(job: str, artifact_dir: pathlib.Path, *, run: Callable[..., Any] = subprocess.run) -> str | None:
    """Load the artifact with its family's ``load`` in ``.venv-hfjobs`` (step 2). None == loads clean."""
    mod, _, cls = LOADERS[job].partition(":")
    code = f"import importlib,sys; getattr(importlib.import_module({mod!r}), {cls!r}).load(sys.argv[1])"
    p = run(
        [str(HFJOBS_PYTHON), "-c", code, str(artifact_dir)], capture_output=True, text=True, check=False, cwd=REPO_ROOT
    )
    return None if p.returncode == 0 else (p.stderr or p.stdout or "load failed")[-2000:]


def _committed_metrics(a: mf.Artifact, observed: Mapping[str, Any]) -> dict:
    """The committed side of a CV-parity certification: the wheel bundle's ``metrics.json``, else its
    manifest ``cv_reference`` (HFU spec 14.4), else a refusal naming the artifact; for a Hub artifact the
    observed ``metrics_summary`` (written by `bootstrap_observed` / `adopt`)."""
    if a.kind == "wheel":
        own = REPO_ROOT / a.path / "metrics.json"
        if own.is_file():
            return _read_json(own)
        if a.cv_reference is None:
            raise AdoptionRefusedError(f"{a.id}: no metrics.json and no cv_reference (the committed CV record)")
        path, _, key = a.cv_reference.partition(":")
        doc: Any = _read_json(REPO_ROOT / path)
        for part in key.split(".") if key else ():
            doc = doc[part]
        return dict(doc)
    try:
        return dict(observed[a.id]["metrics_summary"])
    except KeyError as exc:
        raise AdoptionRefusedError(f"{a.id}: no observed metrics_summary (run bootstrap_observed first)") from exc


def _integrity(
    a: mf.Artifact,
    fresh: pathlib.Path,
    record: Mapping[str, Any],
    shipped: Mapping[str, Any],
    staging: pathlib.Path,
    *,
    run: Callable[..., Any],
) -> dict:
    counters = record.get("counters") or {}
    totals = counters.get("totals") or {}
    if counters.get("n_counters_unrecorded") or not totals.get("duration_s"):
        return {"go": False, "reason": "integrity inputs missing (extract counters unrecorded)"}
    fps = float(totals["n_frames"]) / float(totals["duration_s"])
    shipped_path = staging / "_shipped" / f"{a.id}.metrics.json"
    shipped_path.parent.mkdir(parents=True, exist_ok=True)
    shipped_path.write_text(json.dumps(shipped, sort_keys=True), encoding="utf-8")
    out = staging / "verdicts" / f"{a.job}__{a.id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        str(HFJOBS_PYTHON), "-m", "scripts.validate_trained_artifact_integrity",
        "--family", a.job, "--shipped", str(shipped_path), "--fresh", str(fresh / "metrics.json"),
        "--n-events", str(int(totals["n_events"])), "--fps", repr(fps), "--horizon-s", "1.0", "--out", str(out),
    ]  # fmt: skip
    p = run(argv, capture_output=True, text=True, check=False, cwd=REPO_ROOT)
    if p.returncode != 0 and not out.is_file():
        return {"go": False, "reason": (p.stderr or "")[-2000:]}
    v = _read_json(out)
    return {"go": bool(v.get("go")), "verdict": v.get("verdict"), "path": str(out)}


def _rebaseline(spec: Any, fresh: pathlib.Path, commit: str, *, run: Callable[..., Any]) -> dict:
    """A rebaseline driver (decisions P3, P15): its output vs the committed one, provenance removed. A
    difference is classified by the code of the driver's declared input modules at ``dd7ed4f`` and at the
    campaign commit (:func:`_code_delta`, docstrings removed): none changed -> an environment delta
    (reported); else a code delta, which STOPS for an owner ruling -- unless the driver declares a ``tolerance``
    and every difference is within it (:func:`_within_tolerance`, HFU spec 17.4)."""
    from importlib import import_module

    from scripts.hf_jobs.identity import artifact_identity, compare

    committed = REPO_ROOT / spec.output_rel
    reports = tuple(sorted(p.name for p in committed.iterdir() if p.suffix == ".json"))
    diffs = compare(
        artifact_identity(committed, json_files=reports),
        artifact_identity(fresh, json_files=reports),
        same_commit=False,
    )
    if not diffs:
        return {"ok": True, "basis": "rebaseline", "identical": True, "diffs": []}
    contract = import_module(spec.module).input_contract()
    modules = sorted({*contract.get("extractors", []), *contract.get("models", [])})
    paths = [m.replace(".", "/") + ".py" for m in modules]
    changed = _code_delta(paths, REBASELINE_FROM[spec.name], commit, run=run)
    out = {
        "ok": not changed,  # a code delta STOPS adoption of this artifact for an owner ruling (P15)
        "basis": "rebaseline",
        "identical": False,
        "diffs": diffs,
        "delta": "code" if changed else "environment",
        "input_modules": modules,
        "changed_modules": changed,
    }
    if spec.tolerance is not None:  # HFU spec 17.4: within the owner-ratified tolerance passes
        within, notes = _within_tolerance(committed, fresh, spec.tolerance)
        out |= {"ok": out["ok"] or within, "within_tolerance": within, "tolerance_notes": notes}
    return out


def _within_tolerance(committed: pathlib.Path, fresh: pathlib.Path, tol: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """A rebaseline's outputs compared within ``tol`` (HFU spec 17.4): ``(ok, notes)``, the notes being the
    failures, else the measured differences. JSON after the provenance keys, field by field: every number within
    ``tol["rel"]`` relative (zero only equals zero; NaN equals only NaN; an infinity only itself), a dotted
    ``tol["abs"]`` field within its absolute bound, ``tol["uncompared"]`` top-level keys skipped,
    ``tol["fresh_zero"]`` keys 0 in the fresh run, everything else exact. Parquet: the same columns, dtypes, rows
    and order; float columns within ``rel`` (one note per column: the count and the largest relative difference, or
    the first row beyond), the rest exact. Any other file: the same bytes."""
    import math

    import numpy as np
    import pandas as pd

    from scripts.hf_jobs.identity import normalized_json

    rel, bounds = float(tol["rel"]), dict(tol.get("abs", {}))
    bad: list[str] = []
    measured: list[str] = []

    def close(path: str, a: Any, b: Any) -> bool:  # equal values never reach a note (`walk`)
        if not (math.isfinite(a) and math.isfinite(b)):  # NaN equals only NaN; an infinity only itself
            return math.isnan(a) and math.isnan(b)
        if path in bounds:
            return abs(a - b) <= bounds[path]
        if isinstance(a, float) or isinstance(b, float):
            return abs(a - b) <= rel * max(abs(a), abs(b))
        return False

    def walk(path: str, a: Any, b: Any) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                bad.append(f"{path or '<root>'}: keys {sorted(set(a) ^ set(b))} differ")
                return
            for k in sorted(a):
                walk(f"{path}.{k}" if path else k, a[k], b[k])
            return
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                walk(f"{path}[{i}]", x, y)
            return
        numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b))
        if numbers and close(path, a, b):
            if a != b and not (isinstance(a, float) and math.isnan(a)):
                measured.append(f"{path}: {a!r} -> {b!r}")
        elif a != b:
            bad.append(f"{path}: {a!r} != {b!r}")

    names = sorted(p.name for p in committed.iterdir() if p.is_file())
    if names != sorted(p.name for p in fresh.iterdir() if p.is_file()):
        return False, ["the file sets differ"]
    for name in names:
        c, f = committed / name, fresh / name
        if name.endswith(".json"):
            raw = json.loads(f.read_text(encoding="utf-8"))
            zero = tol.get("fresh_zero", ())
            bad += [f"{name}: fresh {k} = {raw.get(k)!r}, expected 0" for k in zero if raw.get(k) != 0]
            skip = set(tol.get("uncompared", ()))
            docs = [json.loads(normalized_json(p.read_bytes())) for p in (c, f)]
            walk("", *({k: v for k, v in d.items() if k not in skip} for d in docs))
        elif name.endswith(".parquet"):
            a, b = pd.read_parquet(c), pd.read_parquet(f)
            if list(a.columns) != list(b.columns) or list(a.dtypes) != list(b.dtypes) or len(a) != len(b):
                bad.append(f"{name}: columns, dtypes or rows differ")
                continue
            for col in a.columns:
                if a[col].dtype.kind != "f":
                    if not (a[col].to_numpy() == b[col].to_numpy()).all():
                        bad.append(f"{name}: column {col} differs")
                    continue
                x, y = a[col].to_numpy(dtype=np.float64), b[col].to_numpy(dtype=np.float64)
                nan = np.isnan(x) & np.isnan(y)  # NaN equals only NaN; an infinity only itself
                finite = np.isfinite(x) & np.isfinite(y)
                scale = np.maximum(np.abs(x), np.abs(y))
                with np.errstate(invalid="ignore"):  # inf - inf
                    within = nan | (x == y) | (finite & (np.abs(x - y) <= rel * scale))
                changed = ~nan & (x != y)
                if not within.all():
                    i = int(np.flatnonzero(~within)[0])
                    bad.append(
                        f"{name}:{col}: {int((~within).sum())} of {len(x)} values beyond {rel:g} relative; "
                        f"first row {i}: {x[i]!r} -> {y[i]!r}"
                    )
                elif changed.any():
                    worst = float((np.abs(x[changed] - y[changed]) / scale[changed]).max())
                    n_changed = int(changed.sum())
                    measured.append(f"{name}:{col}: {n_changed} of {len(x)} values differ, max rel {worst:.3g}")
        elif c.read_bytes() != f.read_bytes():
            bad.append(f"{name}: bytes differ")
    return not bad, bad or measured


def _code_ast(source: str | None) -> str | None:
    """The module's AST dump with every module, class and function docstring removed (``None``: absent)."""
    import ast

    if source is None:
        return None
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def _code_delta(
    module_paths: Sequence[str],
    base: str,
    commit: str,
    *,
    run: Callable[..., Any] = subprocess.run,
    cwd: pathlib.Path = REPO_ROOT,
) -> list[str]:
    """The input modules whose code differs between ``base`` and ``commit`` (HFU spec 14.3, F9): an endpoint
    compare, one ``git show`` per module and endpoint, docstrings removed. A docstring-only commit is not a
    code delta; a module present at only one endpoint is. Empty == no code delta."""

    def _show(rev: str, path: str) -> str | None:
        p = run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True, check=False, cwd=cwd)
        return p.stdout if p.returncode == 0 else None

    return [path for path in module_paths if _code_ast(_show(base, path)) != _code_ast(_show(commit, path))]


def _certify(
    spec: Any,
    inst: Instance,
    ref: mf.Artifact,
    fresh: pathlib.Path,
    fresh_metrics: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    run: Callable[..., Any],
) -> dict:
    """Step 3 (`floor.certify`, the registry policy): ``public`` -> the head-to-head CLI against the
    committed wheel bundle; ``owner`` -> CV parity on the registry ``cv_metric``. Missing inputs FAIL."""
    from scripts.hf_jobs import floor

    arm = mf.effective_shipped_arm(ref)  # ADR-132 (spec 7.1): certify what the entry expects to have shipped
    if spec.certification_for(arm) == "public":
        if ref.kind != "wheel":
            return {"ok": False, "reason": f"{ref.id}: a public head-to-head needs a committed wheel bundle"}
        h2h = fresh.parent / f"{inst.variant}.headtohead.json"
        argv = [
            str(HFJOBS_PYTHON), "-m", "scripts.hf_jobs.headtohead", inst.job,
            "--committed-dir", str(REPO_ROOT / ref.path), "--fresh-dir", str(fresh),
            "--feature-set", "position_only" if "position_only" in inst.variant else "faithful",
            "--certification", "public", "--metrics-json", str(h2h),
        ]  # fmt: skip
        env = None
        rows = fresh.with_name(f"{fresh.name}.eval_rows.parquet")
        if _public_staged(inst, arm):  # the run's own extract rows (HFU spec 14.4)
            argv += ["--eval-rows", str(rows)]
        elif ref.dataset is not None and _dataset_kind(ref) == "statsbomb_open":
            from scripts._sb_open_data import OPEN_DATA_REVISION_ENV
            from scripts.hf_jobs.datasets import locked

            # F14: the public builders read the open data at the LOCKED revision, never live master; with a
            # staged snapshot, from that snapshot into the staging folder (HFU spec 14.7).
            rec = locked(ref.dataset)
            env = {**os.environ, OPEN_DATA_REVISION_ENV: rec["revision"]}
            if rec.get("snapshot"):
                from scripts._sb_open_data import OPEN_DATA_SNAPSHOT_ENV

                repo = _declared()[ref.dataset].snapshot_repo
                env[OPEN_DATA_SNAPSHOT_ENV] = f"{repo}@{rec['snapshot']['revision']}"
                env["SILLY_KICKS_CORPUS_CACHE_DIR"] = str(_open_data_dir(fresh.parents[1], inst.id))
        p = run(argv, capture_output=True, text=True, check=False, cwd=REPO_ROOT, env=env)
        if p.returncode != 0 or not h2h.is_file():
            return {"ok": False, "reason": f"head-to-head failed: {(p.stderr or '')[-2000:]}"}
        try:
            committed = _committed_metrics(ref, observed)
            gate = next((b.in_sample_gate for b in getattr(spec, "builds", ()) if b.variant == inst.variant), True)
            ok, detail = floor.certify(
                spec,
                arm,
                fresh_metrics=fresh_metrics,
                committed_metrics=committed,
                headtohead=_read_json(h2h),
                in_sample_gate=gate,
            )
        except AdoptionRefusedError as exc:
            return {"ok": False, "reason": str(exc)}
        except KeyError as exc:
            return {"ok": False, "reason": f"missing head-to-head pair or cv metric {exc}"}
        return {"ok": ok, **detail}
    try:
        ok, detail = floor.certify(
            spec, arm, fresh_metrics=fresh_metrics, committed_metrics=_committed_metrics(ref, observed)
        )
    except KeyError as exc:
        return {"ok": False, "reason": f"missing cv metric {exc}"}
    return {"ok": ok, **detail}


def verify(
    c: Campaign,
    commit: str,
    *,
    staging: pathlib.Path,
    replica: int | None = None,
    run: Callable[..., Any] = subprocess.run,
    workers: int = 4,
) -> dict:
    """Steps 2-4 for every retrieved instance; writes ``<staging>/verify.json``. ``ok`` per instance. Up to
    ``workers`` instances at once (HFU spec 17.5): each ``[verify]`` line prints when its instance ends; the report
    is in instance order. The report holds ``manifest_sha256`` (:func:`_manifest_sha256`) of the working-tree
    manifest it checked, read once; :func:`adopt` refuses another manifest (ADR-132 item 10)."""
    import concurrent.futures

    staging = _outside_repo(staging)
    text = mf.MANIFEST.read_text(encoding="utf-8")  # read ONCE: the checked manifest and its digest agree
    man = mf.loads(text, base=mf.MANIFEST.resolve().parents[2])
    observed = _read_json(mf.OBSERVED) if mf.OBSERVED.is_file() else {}
    insts = instances(c, commit, replica=replica)

    def one(item: tuple[int, Instance]) -> dict | None:
        i, inst = item
        return _verify_one(
            inst, i, len(insts), commit=commit, staging=staging, man=man, observed=observed, run=run, mode=c.mode
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(one, enumerate(insts, start=1)))
    report: dict[str, Any] = {
        "commit": commit,
        "campaign": c.name,
        "manifest_sha256": _manifest_sha256(text),  # ADR-132 item 10: adopt refuses another manifest
        "instances": {inst.id: row for inst, row in zip(insts, rows, strict=True) if row is not None},
    }
    (staging / "verify.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return report


def _verify_one(
    inst: Instance,
    i: int,
    n: int,
    *,
    commit: str,
    staging: pathlib.Path,
    man: mf.Manifest,
    observed: Mapping[str, Any],
    run: Callable[..., Any],
    mode: str = "rematerialize",
) -> dict | None:
    """:func:`verify` for instance ``i`` of ``n``: its report row, or ``None`` when it was not retrieved. Every
    file it writes is this instance's own (the head-to-head, the open-data cache, the integrity verdict)."""
    from scripts.hf_jobs.registry import DriverSpec, spec_for

    rp = _record_path(staging, inst.id)
    if not rp.is_file():
        return None
    t0 = time.monotonic()
    record = _read_json(rp)
    fresh = _artifact_dir(staging, inst.id)
    spec = spec_for(inst.job)
    row: dict[str, Any] = {"artifacts": {}}
    if isinstance(spec, DriverSpec):
        row["certification"] = _rebaseline(spec, fresh, commit, run=run)
        row["ok"] = bool(row["certification"]["ok"])
        _say_verify(i, n, inst.id, row, [] if row["ok"] else ["certification"], t0)
        return row
    targets = artifacts_for(inst, man)
    if not targets:
        row.update(ok=False, reason="no manifest artifact for this instance")
        _say_verify(i, n, inst.id, row, ["no manifest artifact"], t0)
        return row
    row["load_error"] = load_check(inst.job, fresh, run=run)
    # ONE certification per instance, against its committed reference: the wheel bundle when the
    # instance has one (a Hub mirror / sibling shares it), else the Hub artifact's observed metrics.
    ref = next((a for a in targets if a.kind == "wheel"), targets[0])
    fresh_metrics = _read_json(fresh / "metrics.json") if (fresh / "metrics.json").is_file() else {}
    comparison = _certify(spec, inst, ref, fresh, fresh_metrics, observed, run=run)
    if mode == "refit":  # ADR-130: the arm passes on its own acceptance gates; the comparison is reported
        gates = fresh_metrics.get("acceptance") or {}
        ok = bool(gates) and all(bool(v) for v in gates.values())
        row["certification"] = {
            "ok": ok,
            "basis": "refit_fresh_acceptance",
            "acceptance": gates,
            "comparison": comparison,
        }
    else:
        row["certification"] = comparison
    checks = {"load": row["load_error"] is None, "certification": bool(row["certification"]["ok"])}
    if _dataset_kind_of(inst) == "statsbomb_open":  # the committed side has no ids (HFU spec 14.3)
        row["coverage_note"] = coverage_note(cv_ok=bool(row["certification"].get("cv_parity", {}).get("ok")))
    if inst.job in INTEGRITY_FAMILIES and mode != "refit":  # a refit does not reproduce the committed model
        row["integrity"] = _integrity(ref, fresh, record, _committed_metrics(ref, observed), staging, run=run)
        checks["integrity"] = bool(row["integrity"]["go"])
    for a in targets:
        row["artifacts"][a.id] = {"violations": expectation_violations(a, fresh, record, commit)}
        checks[f"expectations {a.id}"] = not row["artifacts"][a.id]["violations"]
    row["ok"] = all(checks.values())
    _say_verify(i, n, inst.id, row, [k for k, ok in checks.items() if not ok], t0)
    return row


# ---- adopt -----------------------------------------------------------------------------------------


def _git_status(run: Callable[..., Any], root: pathlib.Path) -> set[str]:
    """The changed and untracked paths of the tree at ``root`` (``git status --porcelain``)."""
    p = run(
        ["git", "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True, check=True, cwd=root
    )
    return {line[3:].strip().strip('"') for line in (p.stdout or "").splitlines() if line.strip()}


def _driver_outputs() -> dict[str, str]:
    from scripts.hf_jobs.registry import DRIVERS

    return {name: d.output_rel for name, d in DRIVERS.items()}


def _allowed(path: str, man: mf.Manifest) -> bool:
    """The ADR-037 diff scope: wheel destinations, driver outputs, cards, the manifest dir, verdicts, the
    campaign summaries."""
    dirs = tuple(a.path.rstrip("/") + "/" for a in man.artifacts.values() if a.kind == "wheel")
    dirs += tuple(rel.rstrip("/") + "/" for rel in _driver_outputs().values())
    cards = {a.card for a in man.artifacts.values() if a.card}
    return path.startswith(ADOPT_PATHS + dirs) or path in cards


def _set_manifest_commit(text: str, artifact_id: str, commit: str) -> str:
    """Set ONE artifact's ``training_commit`` line (spec 5.7 step 6: from the campaign record, verified in
    step 4). Every other byte of the human-written manifest is kept."""
    marker = f'\nid = "{artifact_id}"\n'
    if text.count(marker) != 1:
        raise AdoptionRefusedError(f"manifest artifact {artifact_id!r} is declared {text.count(marker)} times")
    start = text.index(marker)
    nxt = text.find("\n[[artifacts]]", start)
    end = len(text) if nxt == -1 else nxt
    lines = text[start:end].split("\n")
    hits = [i for i, line in enumerate(lines) if line.startswith("training_commit = ")]
    if len(hits) != 1:
        raise AdoptionRefusedError(f"manifest artifact {artifact_id!r} has {len(hits)} training_commit lines")
    lines[hits[0]] = f'training_commit = "{commit}"'
    return text[:start] + "\n".join(lines) + text[end:]


def _manifest_sha256(text: str) -> str:
    """ADR-132 item 10, rule 11: the digest of the manifest a ``verify`` checked. The sha256 of the canonical JSON
    of the parsed TOML with every artifact's ``training_commit`` removed: that is the one field ``adopt`` writes
    (INV-9), so a committed adoption keeps the digest its ``verify`` recorded. A comment or a line ending does
    not move it; every value does."""
    from scripts._toml import loads_toml

    raw = loads_toml(text)
    raw["artifacts"] = [{k: v for k, v in a.items() if k != "training_commit"} for a in raw.get("artifacts", [])]
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


#: ADR-132 item 10 (corpus widening spec 7.1): the manifest fields that the operator's edit after a run may change
#: (spec 7.1's table). ``tier`` is the channel and never changes; ``training_commit`` is the one field a tool
#: writes (INV-9).
_EXPECTATION_FIELDS: frozenset[str] = frozenset({"dataset", "shipped_arm", "certification", "corpus", "expect"})
#: The values spec 7.1 sets. ``dataset`` must be a declared, locked dataset (checked apart).
_EXPECTATION_DOMAINS: dict[str, frozenset[str]] = {
    "shipped_arm": frozenset({"public", "sc_extended"}),
    "corpus": frozenset({"public_ids", "digest"}),
    "certification": frozenset({"public", "owner"}),
}
_MANIFEST_REL = "docs/artifacts/manifest.toml"


def _expectation_edit(c: Campaign, commit: str, *, run: Callable[..., Any], root: pathlib.Path) -> bytes | None:
    """ADR-132 item 10: the pre-adopt manifest bytes when the ONLY uncommitted change of ``root`` is an edit of
    the expectation fields of this campaign's entries (spec 7.1 step 2, made after the run and before ``verify``);
    ``None`` when the root has no change or more than the manifest changed, so that today's checks answer it with
    today's messages. Raises :class:`AdoptionRefusedError` for any other manifest edit. Reads only; the HEAD check
    and the ``verify`` binding stay in :func:`adopt`."""
    from scripts._toml import loads_toml
    from scripts.hf_jobs.datasets import locked

    if _git_status(run, root) != {_MANIFEST_REL}:  # rules 1 and 2: before any campaign or HEAD read
        return None
    if not (root / _MANIFEST_REL).is_file():
        raise AdoptionRefusedError("manifest edit: the manifest is deleted; restore it")
    pre = (root / _MANIFEST_REL).read_bytes()
    work = pre.decode("utf-8")
    head = run(["git", "show", f"HEAD:{_MANIFEST_REL}"], capture_output=True, text=True, check=True, cwd=root).stdout
    try:  # rule 4: both sides are valid manifests
        head_man = mf.loads(head, base=root)
        mf.loads(work, base=root)
    except Exception as exc:
        raise AdoptionRefusedError(f"manifest edit: the manifest does not load ({exc})") from exc
    h, w = loads_toml(head), loads_toml(work)
    for key in sorted((set(h) | set(w)) - {"artifacts"}):  # rule 5
        if h.get(key) != w.get(key):
            raise AdoptionRefusedError(f"manifest edit: the top-level table {key!r} changed; only artifacts may")
    ha, wa = h.get("artifacts", []), w.get("artifacts", [])
    if [a["id"] for a in ha] != [a["id"] for a in wa]:  # rule 6
        raise AdoptionRefusedError("manifest edit: the artifact list changed (an entry added, removed or moved)")
    ours = {a.id for inst in instances(c, commit) for a in artifacts_for(inst, head_man)}  # rule 7, from HEAD
    for old, new in zip(ha, wa, strict=True):
        aid = old["id"]
        if aid not in ours:
            if old != new:
                raise AdoptionRefusedError(f"manifest edit: {aid} changed; it is not an entry of this campaign")
            continue
        for key in sorted((set(old) | set(new)) - _EXPECTATION_FIELDS):  # rule 8
            if old.get(key) != new.get(key):
                raise AdoptionRefusedError(
                    f"manifest edit: {aid}.{key} changed; only {sorted(_EXPECTATION_FIELDS)} may change"
                )
        gone = [k for k in sorted(_EXPECTATION_FIELDS) if k in old and k not in new]  # rule 9
        gone += [f"expect key {k!r}" for k in sorted(old.get("expect", {})) if k not in new.get("expect", {})]
        if gone:
            raise AdoptionRefusedError(
                f"manifest edit: {aid} removes {', '.join(gone)}; the edit may set an expectation, never remove one"
            )
        if "dataset" in new:  # rule 10: every present value of a campaign entry
            ds = new["dataset"]
            try:
                usable = ds in _declared() and bool(locked(ds))
            except KeyError:  # declared but never recorded
                usable = False
            if not usable:
                raise AdoptionRefusedError(f"manifest edit: {aid}.dataset {ds!r} is not a declared, locked dataset")
        for key, domain in _EXPECTATION_DOMAINS.items():
            if key in new and new[key] not in domain:
                raise AdoptionRefusedError(f"manifest edit: {aid}.{key} = {new[key]!r} is not one of {sorted(domain)}")
    return pre


def _rollback(run: Callable[..., Any], root: pathlib.Path, *, keep: Mapping[str, bytes] | None = None) -> list[str]:
    """HFU spec 18.3: restore every path ``git status`` of ``root`` reports -- the root was clean at entry, so each
    is adopt's own: a tracked path is checked out from HEAD (a deleted file comes back), a new file is removed.
    ``keep`` (ADR-132 item 10): the paths that were already changed at entry (the operator's manifest edit) and
    their bytes then; each is written back after the restore. Returns the paths it could not restore."""
    unrestored = _restore_to_head(run, root)
    for rel, data in (keep or {}).items():
        try:
            (root / rel).write_bytes(data)
        except Exception:  # named with the other failures
            unrestored.append(rel)
    return unrestored


def _restore_to_head(run: Callable[..., Any], root: pathlib.Path) -> list[str]:
    """Check out every changed tracked path of ``root`` from HEAD and remove every untracked one."""
    try:
        p = run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
        )
    except Exception as exc:  # reported as the unrestored tree
        return [f"<git status failed: {exc!r}>"]
    unrestored: list[str] = []
    for line in (p.stdout or "").splitlines():
        if not line.strip():
            continue
        code, path = line[:2], line[3:].strip().strip('"')
        try:
            if code == "??":
                (root / path).unlink()
            else:
                run(["git", "checkout", "HEAD", "--", path], capture_output=True, text=True, check=True, cwd=root)
        except Exception:  # every path is attempted, the failures are named
            unrestored.append(path)
    return unrestored


def _finalize(staging: pathlib.Path, inst_id: str) -> pathlib.Path:
    """``<staging>/_final/<instance>/``: the instance's staging files with ``metrics.json`` finalized off HF
    (:func:`scripts._corpus_identity.canonical_probe_ids`; HFU spec 18.6), rebuilt on every call. The wheel bundle,
    a Hub artifact's ``observed.json`` entry and ``publish`` all read this copy, so they agree byte for byte."""
    import shutil

    from scripts._corpus_identity import canonical_probe_ids

    src, dst = _artifact_dir(staging, inst_id), _final_dir(staging, inst_id)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if not f.is_file():
            continue
        data = f.read_bytes()
        if f.name == "metrics.json":
            m = json.loads(data)
            fin = canonical_probe_ids(m)
            if fin != m:
                data = (json.dumps(fin, indent=2) + "\n").encode("utf-8")
        (dst / f.name).write_bytes(data)
    return dst


def _bundle(a: mf.Artifact, fresh: pathlib.Path, root: pathlib.Path) -> list[str]:
    """Step 5: the allowlisted files into the wheel destination in canonical shape (an allowlisted file the
    fresh artifact lacks is removed; the wheel MODEL_CARD.md stays and is re-rendered in step 7). A public
    bundle's ``metrics.json`` is rehydrated with the canonical public ids (its digest was verified first)."""
    from scripts._corpus import rehydrate_public_corpus_ids
    from scripts._hub_publish import MODEL_ONLY_ALLOWLIST

    dest = root / a.path
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name in MODEL_ONLY_ALLOWLIST:
        if name == "README.md":
            continue
        src, dst = fresh / name, dest / name
        if src.is_file():
            data = src.read_bytes()
            if name == "metrics.json" and a.corpus == "public_ids":
                from scripts.hf_jobs.datasets import locked

                pairs = locked(a.dataset or "")["public_ids"]  # ADR-132: the dataset's LOCKED public arm
                fin = rehydrate_public_corpus_ids(json.loads(data), pairs=pairs)
                data = (json.dumps(fin, indent=2) + "\n").encode("utf-8")
            if not dst.is_file() or dst.read_bytes() != data:
                dst.write_bytes(data)
                written.append(dst.relative_to(root).as_posix())
        elif dst.is_file():
            dst.unlink()
            written.append(dst.relative_to(root).as_posix())
    return written


def _copy_tree(src: pathlib.Path, dest: pathlib.Path, root: pathlib.Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for f in sorted(src.iterdir()):
        dst = dest / f.name
        if not dst.is_file() or dst.read_bytes() != f.read_bytes():
            dst.write_bytes(f.read_bytes())
            written.append(dst.relative_to(root).as_posix())
    return written


def _observed_entry(d: pathlib.Path, c_name: str, commit: str, now: str) -> dict:
    from scripts.hf_jobs.identity import artifact_identity

    metrics = _read_json(d / "metrics.json") if (d / "metrics.json").is_file() else {}
    meta = _read_json(d / "metadata.json") if (d / "metadata.json").is_file() else {}
    return {
        "sha256": artifact_identity(d)["files"],
        "metrics_summary": metrics,
        "env": metrics.get("env") or meta.get("env") or {},
        "campaign": f"{c_name}@{commit[:7]}",
        "adopted_at": now,
    }


def _write_text_keep_eol(path: pathlib.Path, text: str) -> bool:
    """Write ``text`` (LF) keeping the file's line endings; False when the bytes are unchanged."""
    raw = path.read_bytes() if path.is_file() else b""
    out = (text.replace("\n", "\r\n") if b"\r\n" in raw else text).encode("utf-8")
    if out == raw:
        return False
    path.write_bytes(out)
    return True


def _sibling_cards(man: mf.Manifest, adopted: Iterable[str]) -> set[str]:
    """Cards whose regions render from an adopted WHEEL artifact (a Hub mirror or a Hub-only sibling)."""
    ids = set(adopted)
    return {a.card for a in man.artifacts.values() if a.card and ({a.mirror_of, a.wheel_sibling} & ids)}


def _changelog_draft(man: mf.Manifest, adopted: Sequence[str], c_name: str, commit: str) -> str:
    lines = [f"### Re-fit weights (campaign {c_name}, commit {commit[:7]}) -- DRAFT for a human edit", ""]
    for aid in adopted:
        a = man.artifacts[aid]
        lines.append(
            f"- `{aid}` ({a.destination}): job `{a.job}`, variant `{a.variant}`, tier `{a.tier}`, "
            f"shipped arm `{mf.effective_shipped_arm(a)}`."
        )
    return "\n".join(lines) + "\n"


def _local_gate(run: Callable[..., Any], root: pathlib.Path) -> dict:
    """Step 10: the full tests/ tree + ruff (scoped) + pyright; the report, never a commit."""
    cmds = {
        "pytest": ["python", "-m", "pytest", "tests/", "-m", "not e2e and not slow", "-p", "no:randomly",
                   "--benchmark-skip", "--continue-on-collection-errors", "-q"],
        "ruff_check": ["uvx", "ruff@0.15.7", "check", "silly_kicks/", "tests/", "scripts/"],
        "ruff_format": ["uvx", "ruff@0.15.7", "format", "--check", "silly_kicks/", "tests/", "scripts/"],
        "pyright": ["python", "-m", "pyright"],
    }  # fmt: skip
    out = {}
    for name, argv in cmds.items():
        p = run(argv, capture_output=True, text=True, check=False, cwd=root)
        failed = [line for line in (p.stdout or "").splitlines() if line.startswith(("FAILED", "ERROR"))]
        out[name] = {"returncode": p.returncode, "failed": failed, "tail": (p.stdout or "")[-1500:]}
    return out


def adopt(
    c: Campaign,
    commit: str,
    *,
    staging: pathlib.Path,
    now: str | None = None,
    run: Callable[..., Any] = subprocess.run,
    git: Callable[[], Mapping[str, Any]] | None = None,
    root: pathlib.Path = REPO_ROOT,
    gate: bool = True,
) -> dict:
    """Spec 5.7 steps 5-10 over a verified staging dir; never commits. Refuses (writing nothing) on a dirty
    tree (the provenance check, then the root's own ``git status``), a HEAD other than the campaign commit, a
    ``verify.json`` of another commit or of another manifest (``manifest_sha256``, ADR-132 item 10), or any
    instance that did not verify. The ONE dirty tree it accepts is the spec 7.1 expectation edit: only the
    manifest changed, and only in the expectation fields of this campaign's entries (:func:`_expectation_edit`).
    A failure in the write phase restores the tree (HFU spec 18.3) and keeps the expectation edit; the local gate
    runs after it, outside the rollback. Idempotent: a second run over the same staging dir writes nothing --
    over an uncommitted adoption it refuses at the root status, over a committed one at the campaign commit (the
    adoption commit moved HEAD), and the write phase itself is a fixed point."""
    from scripts._provenance import git_provenance, require_clean_tree

    root = pathlib.Path(root).resolve()
    staging = _outside_repo(staging, root)
    prov = dict((git or git_provenance)())
    pre = _expectation_edit(c, commit, run=run, root=root)  # ADR-132 item 10: None = today's checks, unchanged
    if pre is None:
        try:
            require_clean_tree(prov, allow_dirty=False)
        except SystemExit as exc:
            raise AdoptionRefusedError(f"adopt needs a clean tree (ADR-037): {exc}") from exc
    if prov.get("commit") != commit:
        raise AdoptionRefusedError(f"HEAD {prov.get('commit')} is not the campaign commit {commit} (ADR-037)")
    if pre is None:
        changed = _git_status(run, root)  # the provenance reads the cwd; the root is where adopt writes (HFU 18.3)
        if changed:
            raise AdoptionRefusedError(f"the root tree is not clean (ADR-037): {sorted(changed)[:10]}")
    rep = _read_json(staging / "verify.json")
    if rep.get("commit") != commit:
        raise AdoptionRefusedError(f"verify.json is for commit {rep.get('commit')}, not {commit}")
    checked = rep.get("manifest_sha256")  # ADR-132 item 10, rule 11: the manifest that verify checked
    if not checked:
        raise AdoptionRefusedError("verify.json has no manifest_sha256 (it predates ADR-132 item 10); run verify again")
    if checked != _manifest_sha256((root / _MANIFEST_REL).read_text(encoding="utf-8")):
        raise AdoptionRefusedError("the manifest changed after verify; run verify again")
    bad = sorted(k for k, v in rep["instances"].items() if not v.get("ok"))
    if bad:
        raise AdoptionRefusedError(f"instances that did not verify: {bad}")
    now = now or datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    try:
        adopted, written, draft = _write_phase(c, commit, rep, staging=staging, now=now, run=run, root=root)
    except BaseException as exc:  # HFU spec 18.3: a KeyboardInterrupt too
        unrestored = _rollback(run, root, keep=None if pre is None else {_MANIFEST_REL: pre})
        if unrestored:
            raise AdoptionRollbackError(f"adopt failed and could not restore {unrestored}") from exc
        raise
    out: dict[str, Any] = {"adopted": adopted, "written": sorted(set(written)), "changelog_draft": str(draft)}
    if gate:
        out["gate"] = _local_gate(run, root)
    out["next"] = "commit gate: owner approval required"
    return out


def _write_phase(
    c: Campaign,
    commit: str,
    rep: Mapping[str, Any],
    *,
    staging: pathlib.Path,
    now: str,
    run: Callable[..., Any],
    root: pathlib.Path,
) -> tuple[list[str], list[str], pathlib.Path]:
    """`adopt`'s writes (spec 5.7 steps 5-9) over a clean root: ``(adopted, written, changelog draft)``."""
    manifest_path = root / "docs" / "artifacts" / "manifest.toml"
    observed_path = root / "docs" / "artifacts" / "observed.json"
    man = mf.load(manifest_path)
    written: list[str] = []
    observed = _read_json(observed_path) if observed_path.is_file() else {}
    manifest_text = manifest_path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    adopted: list[str] = []
    drivers = _driver_outputs()
    for inst_id, row in sorted(rep["instances"].items()):
        fresh = _artifact_dir(staging, inst_id)
        cert = row.get("certification") or {}
        if cert.get("basis") == "rebaseline":
            if not cert.get("identical"):  # an environment delta (P15): the fresh output is adopted
                rk = inst_id.split("/")[0]
                names = [n for n in drivers if f"-{n}-" in rk]
                if len(names) != 1:
                    raise AdoptionRefusedError(f"{inst_id}: names {len(names)} registered drivers")
                written += _copy_tree(fresh, root / drivers[names[0]], root)
            continue
        final = _finalize(staging, inst_id) if row.get("artifacts") else fresh
        for aid in sorted(row.get("artifacts", {})):
            a = man.artifacts[aid]
            if a.kind == "wheel":
                written += _bundle(a, final, root)
                observed[aid] = _observed_entry(root / a.path, c.name, commit, now)
            else:
                observed[aid] = _observed_entry(final, c.name, commit, now)
            if a.commit_field:  # an artifact that records no commit keeps its entry (HFU spec 18.2)
                manifest_text = _set_manifest_commit(manifest_text, aid, commit)
            adopted.append(aid)
    if _write_text_keep_eol(manifest_path, manifest_text):
        written.append(manifest_path.relative_to(root).as_posix())
    # adopted_at is the FIRST adoption of these bytes: an unchanged entry keeps its date (idempotence)
    old = _read_json(observed_path) if observed_path.is_file() else {}
    for aid, entry in observed.items():
        prev = old.get(aid)
        if prev and {k: v for k, v in prev.items() if k != "adopted_at"} == {
            k: v for k, v in entry.items() if k != "adopted_at"
        }:
            observed[aid] = prev
    if _write_text_keep_eol(observed_path, json.dumps(observed, indent=2, sort_keys=True) + "\n"):
        written.append(observed_path.relative_to(root).as_posix())
    man = mf.load(manifest_path)  # step 7 renders from the updated training commits
    cards = {man.artifacts[aid].card for aid in adopted if man.artifacts[aid].card} | _sibling_cards(man, adopted)
    for card in sorted(c_ for c_ in cards if c_):
        if _write_text_keep_eol(root / card, mf.render_card(root / card, man, card_id=card)):
            written.append(card)
    vdir = staging / "verdicts"
    if vdir.is_dir():
        (root / VERDICTS).mkdir(parents=True, exist_ok=True)
        for v in sorted(vdir.glob("*.json")):
            dst = root / VERDICTS / v.name
            if not dst.is_file() or dst.read_bytes() != v.read_bytes():
                dst.write_bytes(v.read_bytes())
                written.append((VERDICTS / v.name).as_posix())
    draft = staging / "CHANGELOG-draft.md"
    draft.write_text(_changelog_draft(man, adopted, c.name, commit), encoding="utf-8")
    stray = sorted(p for p in _git_status(run, root) if not _allowed(p, man))
    if stray:
        raise AdoptionRefusedError(f"adopt changed paths outside the allowlist (ADR-037): {stray}")
    return adopted, written, draft


# ---- bootstrap + publish ---------------------------------------------------------------------------


def bootstrap_observed(*, api: Any, now: str | None = None, write: bool = True) -> dict:
    """Observed fields for the artifacts that ship TODAY: wheel bundles hashed in the repo; each Hub repo's
    ``metadata.json``/``metrics.json``/``SHA256SUMS`` downloaded read-only. Each is first checked against
    its manifest expectations (commit, corpus, shipped tier, declared values); a mismatch refuses."""
    from huggingface_hub.errors import EntryNotFoundError

    from scripts.hf_jobs.identity import artifact_identity

    man = mf.load()
    now = now or datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    observed: dict[str, Any] = {}
    problems: dict[str, list[str]] = {}
    for a in sorted(man.artifacts.values(), key=lambda x: x.id):
        if a.kind == "wheel":
            d = REPO_ROOT / a.path
            bad = mf.violations(a) if a.commit_field else []
            metrics = _read_json(d / "metrics.json") if (d / "metrics.json").is_file() else {}
            sha = artifact_identity(d)["files"]
        else:
            files: dict[str, bytes] = {}
            for fname in ("metadata.json", "metrics.json", "SHA256SUMS"):
                try:
                    p = api.hf_hub_download(repo_id=a.repo_id, filename=fname, repo_type="model")
                except EntryNotFoundError:  # e.g. xsuccess-v1 ships no metrics.json: recorded absent, not fatal
                    continue
                files[fname] = pathlib.Path(p).read_bytes()
            docs = {k: json.loads(v) for k, v in files.items() if k.endswith(".json")}
            bad = mf.violations(a, read=docs.__getitem__)
            metrics = docs.get("metrics.json", {})
            sha = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())}
        if bad:
            problems[a.id] = bad
            continue
        observed[a.id] = {
            "sha256": sha,
            "metrics_summary": metrics,
            "env": metrics.get("env") or {},
            "campaign": "bootstrap (shipped before ADR-128)",
            "adopted_at": now,
        }
    if problems:
        raise AdoptionRefusedError(f"shipped artifacts that do not meet their manifest expectations: {problems}")
    if write:
        mf.OBSERVED.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return observed


def publish(
    ids: Sequence[str] | None,
    *,
    sandbox: bool,
    api: Any,
    staging: pathlib.Path,
    approved: Sequence[str] = (),
    run: Callable[..., Any] = subprocess.run,
    publisher: Callable[..., Any] | None = None,
) -> dict:
    """Publish Hub artifacts from ``staging`` (an owner go every time). ``sandbox`` -> private
    ``silly-kicks/sandbox-<name>`` repos, created private BEFORE the ADR-088 seam (whose own
    ``create_repo(exist_ok=True)`` then finds them). A public target needs ``--i-have-owner-approval <repo>``
    for EVERY target repo. Each publish is post-verified: a model-only file set, the remote SHA256SUMS equal
    to the local one, README equal to `card_bytes(card)`, and a load round-trip in ``.venv-hfjobs`` (``from_hub``,
    or the job loader for an artifact without a Hub class; see `post_verify`)."""
    from scripts._hub_publish import MODEL_ONLY_ALLOWLIST, card_bytes, publish_model_with_card

    staging = _outside_repo(staging)
    man = mf.load()
    hub = {a.id: a for a in man.artifacts.values() if a.kind == "hub"}
    targets = [hub[i] for i in (ids or sorted(hub))]
    rep = _read_json(staging / "verify.json")
    src = {aid: k for k, v in rep["instances"].items() for aid in v.get("artifacts", {})}
    plan = []
    for a in targets:  # every check runs BEFORE the first upload
        if a.id not in src:
            raise AdoptionRefusedError(f"{a.id}: not in verify.json (retrieve + verify it first)")
        if not rep["instances"][src[a.id]].get("ok"):
            raise AdoptionRefusedError(f"{a.id}: its instance {src[a.id]} did not verify")
        name = a.repo_id.partition("/")[2]
        plan.append((a, f"silly-kicks/sandbox-{name}" if sandbox else a.repo_id))
    if not sandbox:
        missing = sorted(repo for _a, repo in plan if repo not in approved)
        if missing:
            raise AdoptionRefusedError(f"publishing to public repos needs --i-have-owner-approval for each: {missing}")
    results: dict[str, Any] = {}
    for a, repo in plan:
        d = _finalize(staging, src[a.id])  # the finalized copy adopt also read (HFU spec 18.6)
        if sandbox:
            api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        (publisher or publish_model_with_card)(api, str(d), repo, model_card=str(REPO_ROOT / (a.card or "")))
        results[a.id] = {
            "repo": repo,
            **post_verify(
                api,
                repo,
                a,
                d,
                run=run,
                allowlist=MODEL_ONLY_ALLOWLIST,
                card=card_bytes(REPO_ROOT / (a.card or "")),
                staging=staging,
            ),
        }
    (staging / "publish.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def post_verify(
    api: Any,
    repo: str,
    a: mf.Artifact,
    local: pathlib.Path,
    *,
    run: Callable[..., Any],
    allowlist: Sequence[str],
    card: bytes,
    staging: pathlib.Path,
) -> dict:
    """What the Hub holds after a publish: model-only, the local SHA256SUMS, the rendered card, loadable.

    The load round trip: an artifact that declares a ``hub_class`` loads with that class's ``from_hub``; an
    artifact without one (``hub_class = ""``, e.g. xsuccess: its class has no ``from_hub``) loads the PUBLISHED
    files, downloaded to ``<staging>/_post_verify/<repo name>/`` (emptied first, so a file that an earlier run
    left there cannot pass for a file of the Hub), with the job loader (`load_check`)."""
    files = sorted(s.rfilename for s in (api.model_info(repo).siblings or []))
    problems = []
    nested = [f for f in files if "/" in f]
    extra = [f for f in files if f not in allowlist and f != ".gitattributes"]
    if nested or extra:
        problems.append(f"not model-only: nested={nested} extra={extra}")
    remote_sums = pathlib.Path(
        api.hf_hub_download(repo_id=repo, filename="SHA256SUMS", repo_type="model", force_download=True)
    ).read_bytes()
    if remote_sums.replace(b"\r\n", b"\n") != (local / "SHA256SUMS").read_bytes().replace(b"\r\n", b"\n"):
        problems.append("remote SHA256SUMS differs from the local one")
    readme = pathlib.Path(
        api.hf_hub_download(repo_id=repo, filename="README.md", repo_type="model", force_download=True)
    ).read_bytes()
    if readme.replace(b"\r\n", b"\n") != card:
        problems.append("README.md differs from card_bytes(card)")
    if a.hub_class:
        mod, _, cls = LOADERS[a.job].partition(":")
        code = f"import importlib,sys; getattr(importlib.import_module({mod!r}), {cls!r}).from_hub(sys.argv[1])"
        p = run([str(HFJOBS_PYTHON), "-c", code, repo], capture_output=True, text=True, check=False, cwd=REPO_ROOT)
        if p.returncode != 0:
            problems.append(f"from_hub failed: {(p.stderr or '')[-500:]}")
    else:
        dest = pathlib.Path(staging) / "_post_verify" / repo.partition("/")[2]
        if dest.exists():  # load ONLY what this call downloads: a file of an earlier run is not the Hub's
            shutil.rmtree(dest)
        api.snapshot_download(repo_id=repo, repo_type="model", local_dir=dest)
        err = load_check(a.job, dest, run=run)
        if err is not None:
            problems.append(f"load failed: {err[-500:]}")
    return {"ok": not problems, "problems": problems, "files": files}


# ---- compare (HFU fit-units spec 6 step 3) -----------------------------------------------------------


def _staging_run_key(folder: pathlib.Path, campaign: str | None, job: str, tier: str) -> pathlib.Path | None:
    """The one ``<campaign>-<commit7>-<job>-<tier>`` folder of ``folder``; ``None`` when there is none. Two
    matches are ambiguous and refuse. ``campaign=None`` matches any campaign name (a baseline of `compare`: an
    artifact is named by its job, tier and variant; HFU launch top-up spec §14.2)."""
    import re

    name = re.escape(campaign) if campaign is not None else "[A-Za-z0-9_]+"
    pattern = re.compile(rf"^{name}-[0-9a-f]{{7}}-{re.escape(job)}-{re.escape(tier)}$")
    hits = sorted(p for p in folder.iterdir() if p.is_dir() and pattern.fullmatch(p.name))
    if len(hits) > 1:
        raise AdoptionRefusedError(f"{folder}: {len(hits)} run keys for {job}-{tier}: {[p.name for p in hits]}")
    return hits[0] if hits else None


def _commit_file(job: str, tier: str, variant: str, man: mf.Manifest) -> str | None:
    """The file that the instance's manifest ``commit_field`` names, when it is not one of
    `identity.NORMALIZED` (``model.json`` for match_outcome)."""
    from scripts.hf_jobs.identity import NORMALIZED

    files = {
        a.commit_field.partition(":")[0]
        for a in man.artifacts.values()
        if (a.job, a.variant, a.tier) == (job, variant, tier) and a.commit_field
    } - set(NORMALIZED)
    if len(files) > 1:
        raise AdoptionRefusedError(f"{job}-{tier}/{variant}: more than one commit file {sorted(files)}")
    return next(iter(files), None)


def compare_staging(c: Any, *, staging: pathlib.Path, baselines: Sequence[pathlib.Path]) -> dict[str, dict]:
    """Compare the artifacts of a launch with what shipped, instance by instance (local files only).

    What shipped can come from several launches (HFU launch top-up spec §14.2: launch 7 for 12 artifacts, a refit
    run for 8). The baseline of an instance is the LAST of ``baselines`` that holds a run key of its job and tier,
    under any campaign name, with a folder of its variant; the row names that run key (``baseline_run_key``).

    The NEW side is always its RETRIEVED files (``<staging>/<run_key>/<variant>/``): a job must write the shipped
    form itself. The BASELINE side is its adopted copy (:func:`_final_dir`, the files that `adopt` finalized and
    `publish` uploaded) when the baseline has one for the instance, its retrieved files otherwise: the baseline of
    a parity proof is what shipped (launch 8: the retrieved xCross ``metrics.json`` of launch 7 held staged probe
    names, its adopted copy the canonical ids of HFU spec 18.6, which the trainer writes now).

    One entry per ``<job>-<tier>/<variant>`` of the campaign, in campaign order. A trainer instance:
    ``{"identical", "differences", "baseline"}`` under `identity.cross_commit_differences`, where ``baseline`` is
    ``"adopted"`` or ``"retrieved"`` (the form that was read); an instance whose retrieved folder is absent on a
    side is a difference and names no form. A driver instance (the census): ``{"census_ok", "certification"}`` from
    ``<staging>/verify.json`` -- ok only when the file exists, names the commit of the staging run keys, and the
    instance's row is ``ok``."""
    from scripts.hf_jobs.identity import cross_commit_differences
    from scripts.hf_jobs.registry import DriverSpec, spec_for

    staging = pathlib.Path(staging)
    folders = [pathlib.Path(b) for b in baselines]
    man = mf.load()
    verify_path = staging / "verify.json"
    verify = _read_json(verify_path) if verify_path.is_file() else None
    out: dict[str, dict] = {}
    for j in c.jobs:
        new_rk = _staging_run_key(staging, c.name, j.job, j.tier)
        old_rks = [(b, _staging_run_key(b, None, j.job, j.tier)) for b in folders]
        for v in j.variants:
            key = f"{j.job}-{j.tier}/{v}"
            if isinstance(spec_for(j.job), DriverSpec):
                row = (verify or {}).get("instances", {}).get(f"{new_rk.name}/{v}") if new_rk is not None else None
                commit7 = new_rk.name[len(c.name) + 1 :][:7] if new_rk is not None else ""
                ok = bool(row and row.get("ok") and str((verify or {}).get("commit", "")).startswith(commit7))
                out[key] = {"census_ok": ok, "certification": (row or {}).get("certification")}
                continue
            new_dir = new_rk / v if new_rk is not None else None
            holders = [(b, rk / v) for b, rk in old_rks if rk is not None and (rk / v).is_dir()]
            if new_dir is None or not new_dir.is_dir() or not holders:
                out[key] = {"identical": False, "differences": ["present in only one staging folder"]}
                continue
            baseline, old_dir = holders[-1]  # the last folder that holds the instance
            adopted = _final_dir(baseline, old_dir.relative_to(baseline).as_posix())
            form = "adopted" if adopted.is_dir() else "retrieved"
            base_dir = adopted if form == "adopted" else old_dir
            diffs = cross_commit_differences(new_dir, base_dir, commit_file=_commit_file(j.job, j.tier, v, man))
            out[key] = {
                "identical": not diffs,
                "differences": diffs,
                "baseline": form,
                "baseline_run_key": old_dir.parent.name,
            }
    return out


def compare_ok(result: Mapping[str, Mapping[str, Any]]) -> bool:
    """True when every trainer instance is identical and every driver instance is ``census_ok``."""
    return all(row.get("identical", row.get("census_ok")) is True for row in result.values())
