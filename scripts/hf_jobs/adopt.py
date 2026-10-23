"""Retrieve, verify, adopt and publish campaign artifacts (ADR-128 spec 5.7, decisions P3, P15, INV-9).

``retrieve`` copies each instance's reduced output FileSet out of the bucket store into a staging dir
OUTSIDE the repo, with a record of what the job read (the token inputs it wrote, the extract counters).
``verify`` loads every artifact in ``.venv-hfjobs`` (SHA256SUMS, chirality, feature contract), checks the
five declared expectations, certifies per the registry policy (`floor.certify`; `identity.compare` for a
rebaseline driver) and runs the ADR-126 integrity gate for xShot/xCross. ``adopt`` turns a verified
staging dir into a commit-ready diff -- wheel bundles, the manifest training commits, ``observed.json``,
re-rendered card regions, verdict JSONs, a CHANGELOG draft -- under the ADR-037 guards (clean tree at the
campaign commit; a diff limited to the allowlist), runs the full local gate and STOPS: it never commits.
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
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from scripts.hf_jobs import manifest as mf
from scripts.hf_jobs.campaign import REPO_ROOT, Campaign, Instance, instances, read_plan, store_uri

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


# ---- retrieve --------------------------------------------------------------------------------------


def _counters(store: Any, stage: Mapping[str, Any]) -> dict:
    """An ItemStage's per-unit counters merged in sorted key order (`_stages.item_counters`'s rule),
    read from the store by the unit list the plan reports; ``n_counters_unrecorded`` counts the units
    that are neither excluded nor carry a sidecar."""
    from scripts._driver import _merge_counters

    totals: dict = {}
    missing = 0
    gen = stage["generation"]
    for key in sorted(stage.get("units") or []):
        raw = store.read_bytes(f"{gen}/{key}.counters.json")
        if raw is None:
            if store.read_bytes(f"{gen}/{key}.excluded.json") is None:
                missing += 1
            continue
        _merge_counters(totals, json.loads(raw), key)
    return {"totals": totals, "n_counters_unrecorded": missing}


def retrieve(
    c: Campaign,
    commit: str,
    *,
    staging: pathlib.Path,
    api: Any = None,
    replica: int | None = None,
    plan_fn: Callable[..., dict] = read_plan,
    store_fn: Callable[[str], Any] | None = None,
) -> dict[str, pathlib.Path]:
    """Copy every reduced instance output into ``<staging>/<run_key>/<variant>/`` (never into the repo),
    with ``<instance>.record.json`` beside it: the job's recorded token inputs per stage and the extract
    counters. An instance whose output stage is not reduced is skipped (and absent from the result)."""
    from scripts._stages import FileSet, read_result
    from scripts.hf_jobs.store import BucketStore

    staging = _outside_repo(staging)
    open_store = store_fn or (lambda uri: dataclasses.replace(BucketStore.from_uri(uri), api=api))
    out: dict[str, pathlib.Path] = {}
    for inst in instances(c, commit, replica=replica):
        plan = plan_fn(inst, bucket=c.bucket)
        stages = {s["name"]: s for s in plan["stages"]}
        rec = stages[plan["output"]]
        if not rec["reduced"]:
            continue
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
            "counters": _counters(store, stages["extract"]) if "extract" in stages else None,
        }
        p = _record_path(staging, inst.id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out[inst.id] = dest
    return out


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
    corpus = "digest" if a.corpus == "public_ids" else a.corpus
    # A commit-free output (ADR-128 Task 16) carries no training_commit until `adopt.stamp_provenance` stamps
    # it; the manifest commit check then lands in `adopt` (manifest.py). So drop the commit_field check here for
    # a commit-free fresh (else mf.violations wrongly reports "training_commit missing" for every artifact, B#30).
    expect_a = dataclasses.replace(a, training_commit=commit, corpus=corpus)
    if not _has_training_commit(fresh):
        expect_a = dataclasses.replace(expect_a, commit_field=None)
    bad += mf.violations(expect_a, read=_dir_reader(fresh))
    if a.dataset is not None and a.dataset in _declared():
        pinned = locked(a.dataset)["revision"]
        read = {
            (ti or {}).get("dataset", {}).get("revision")
            for ti in (record.get("token_inputs") or {}).values()
            if (ti or {}).get("dataset") and (ti or {})["dataset"].get("name") == a.dataset
        }
        if read != {pinned}:
            bad.append(f"{a.id}: the job read dataset {a.dataset} at {sorted(map(str, read))}, pinned {pinned}")
    for fname in ("metadata.json", "metrics.json"):
        p = fresh / fname
        if p.is_file():
            sv = _read_json(p).get("shipped_variant")
            if sv is not None and sv != a.tier:
                bad.append(f"{a.id}: {fname} ships {sv!r}, the expected tier is {a.tier!r}")
    return bad


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
    """The committed side of a CV-parity certification: the wheel bundle's ``metrics.json``, or the
    observed ``metrics_summary`` of a Hub artifact (written by `bootstrap_observed` / `adopt`)."""
    if a.kind == "wheel":
        return _read_json(REPO_ROOT / a.path / "metrics.json")
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
    difference is classified by the git log of the driver's declared input modules since ``dd7ed4f``:
    empty -> an environment delta (reported); else a code delta, which STOPS for an owner ruling."""
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
    p = run(
        ["git", "log", "--format=%h", f"{REBASELINE_FROM[spec.name]}..{commit}", "--", *paths],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    commits = (p.stdout or "").split()
    return {
        "ok": not commits,  # a code delta STOPS adoption of this artifact for an owner ruling (P15)
        "basis": "rebaseline",
        "identical": False,
        "diffs": diffs,
        "delta": "code" if commits else "environment",
        "input_modules": modules,
        "commits": commits,
    }


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

    if spec.certification_for(inst.tier) == "public":
        if ref.kind != "wheel":
            return {"ok": False, "reason": f"{ref.id}: a public head-to-head needs a committed wheel bundle"}
        h2h = fresh.parent / f"{inst.variant}.headtohead.json"
        p = run(
            [
                str(HFJOBS_PYTHON),
                "-m",
                "scripts.hf_jobs.headtohead",
                inst.job,
                "--committed-dir",
                str(REPO_ROOT / ref.path),
                "--fresh-dir",
                str(fresh),
                "--feature-set",
                "position_only" if "position_only" in inst.variant else "faithful",
                "--metrics-json",
                str(h2h),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
        )
        if p.returncode != 0 or not h2h.is_file():
            return {"ok": False, "reason": f"head-to-head failed: {(p.stderr or '')[-2000:]}"}
        try:
            ok, detail = floor.certify(
                spec, inst.tier, fresh_metrics=fresh_metrics, committed_metrics={}, headtohead=_read_json(h2h)
            )
        except KeyError as exc:
            return {"ok": False, "reason": f"missing head-to-head pair {exc}"}
        return {"ok": ok, **detail}
    try:
        ok, detail = floor.certify(
            spec, inst.tier, fresh_metrics=fresh_metrics, committed_metrics=_committed_metrics(ref, observed)
        )
    except KeyError as exc:
        return {"ok": False, "reason": f"missing cv metric {exc}"}
    return {"ok": ok, **detail}


def _attestation_pass(c: Campaign, commit: str, *, api: Any, with_replica: int, now: str) -> dict:
    """Main-vs-replica byte compare per stage (S5.10, INV-1): a byte-identical generation gets an attestation
    written to the MAIN store (``attest_rel``), so a later clean HEAD may reuse it without re-materializing.
    Returns ``{main instance id: {stage: {"identical", "diffs"}}}``."""
    from scripts.hf_jobs import launch as launcher
    from scripts.hf_jobs.campaign import _job_phase_records, instances, read_plan
    from scripts.hf_jobs.store import BucketStore

    # This campaign+commit's job labels per run key: the shared content-addressed main store also holds OTHER
    # campaigns' jobs, so the attestation env must come only from THIS campaign's jobs (B#45-B).
    labels_by_rk: dict[str, set[str]] = {}
    for j in launcher.jobs_for(api, campaign=c.name, commit=commit):
        lab = dict(getattr(j, "labels", None) or {})
        labels_by_rk.setdefault(lab.get("run_key", ""), set()).add(launcher.label_of(lab))

    def _scoped(store: Any, run_key: str) -> dict:
        want = labels_by_rk.get(run_key, set())
        return {k: v for k, v in _job_phase_records(store).items() if v.get("label") in want}

    main_by_key = {(i.job, i.tier, i.variant): i for i in instances(c, commit)}
    out: dict[str, Any] = {}
    for ri in instances(c, commit, replica=with_replica):
        mi = main_by_key[(ri.job, ri.tier, ri.variant)]
        main_store = BucketStore(c.bucket, mi.run_key, api=api)
        rep_store = BucketStore(c.bucket, ri.run_key, api=api)
        main_phases = _scoped(main_store, mi.run_key)
        rep_phases = _scoped(rep_store, ri.run_key)
        row: dict[str, Any] = {}
        for st in read_plan(mi, bucket=c.bucket)["stages"]:
            gen, stage, token = st["generation"], st["name"], st["token"]
            # A compare happens only when BOTH sides materialized the generation. An attested stage is SKIPPED
            # in the replica (reduced), so its replica generation is empty -- that is a correct reuse, NOT a
            # determinism mismatch, and must not count as a failed comparison (B#35a). An empty MAIN generation
            # never ran, so never attest it (B#38). Only a both-materialized stage yields a comparison bool.
            if not _gen_tree(main_store, gen):
                row[stage] = {"materialized": False, "reason": "main generation is empty"}
                continue
            if not _gen_tree(rep_store, gen):
                row[stage] = {"materialized": False, "reason": "replica skipped (attested reuse)"}
                continue
            diffs = generation_diff(main_store, rep_store, gen)
            row[stage] = {"materialized": True, "identical": not diffs, "diffs": diffs}
            if not diffs:
                m_env, r_env = _job_env(main_phases), _job_env(rep_phases)
                m_state, r_state = m_env.get("tree_state"), r_env.get("tree_state")
                # A missing tree_state means the env is empty: `phases._env` swallowed `env_provenance` (no repo
                # or lock readable), so the provenance is UNKNOWN. The repo rule treats unknown provenance as
                # untrustworthy, NEVER as clean (_provenance.py:91-93; B#32). Refuse the attestation: no reuse
                # token on undescribable code, else `stamp_provenance` later stamps run_tree_dirty=False on a
                # committed artifact (a false clean claim). The determinism compare still counts (the bytes were
                # identical); only reuse is withheld. A recorded "dirty" IS kept (its token never matches a clean
                # HEAD, so it is never reused cross-commit).
                if not m_state or not r_state:
                    row[stage]["attested"] = False
                    row[stage]["reason"] = "unknown provenance (no tree_state)"
                    continue
                rec = {
                    "commit": commit,
                    "stage": stage,
                    "token": token,
                    "main": {"env": m_env, "tree_state": m_state},
                    "replica": {"env": r_env, "tree_state": r_state},
                    "job_ids": {"main": sorted(main_phases), "replica": sorted(rep_phases)},
                    "time": now,
                }
                main_store.upload_bytes(
                    attest_rel(stage, token, commit), (json.dumps(rec, sort_keys=True) + "\n").encode("utf-8")
                )
                row[stage]["attested"] = True
        out[mi.id] = row
    return out


def _canary_pass(c: Campaign, commit: str, *, api: Any) -> tuple[dict, list[str]]:
    """Compare every unit the canary re-ran to the main store byte for byte (S5.10): a mismatch QUARANTINES the
    main generation (unsound reuse) and adds an ``unsound_reuse`` finding. Returns ``({instance: {stage: ...}},
    findings)``."""
    from scripts.hf_jobs.campaign import canary_verdict, instances, read_plan, run_key
    from scripts.hf_jobs.store import BucketStore

    out: dict[str, Any] = {}
    findings: list[str] = []
    for mi in instances(c, commit):
        main_store = BucketStore(c.bucket, mi.run_key, api=api)
        canary_store = BucketStore(c.bucket, run_key(c, commit, mi.job, mi.tier, canary=True), api=api)
        plan = read_plan(mi, bucket=c.bucket)
        row: dict[str, Any] = {}
        for st in plan["stages"]:
            gen, stage, token = st["generation"], st["name"], st["token"]
            produced = sorted(_gen_tree(canary_store, gen))
            if not produced:
                continue
            bad: list[str] = []
            for n in produced:
                ok = canary_verdict(
                    main_bytes=main_store.read_bytes(f"{gen}/{n}") or b"",
                    canary_bytes=canary_store.read_bytes(f"{gen}/{n}") or b"",
                    quarantine=lambda s=stage, t=token, ms=main_store, pl=plan: ms.quarantine_generation(
                        pl["pipeline"], s, t, {"reason": "canary mismatch", "commit": commit}
                    ),
                )
                if not ok:
                    bad.append(f"{gen}/{n}")
                    break
            row[stage] = {"units": produced, "ok": not bad}
            if bad:
                findings.append(f"unsound_reuse: {mi.id}/{stage} canary differs from main at {bad[0]}")
        if row:
            out[mi.id] = row
    return out, findings


def verify(
    c: Campaign,
    commit: str,
    *,
    staging: pathlib.Path,
    replica: int | None = None,
    api: Any = None,
    with_replica: int | None = None,
    canary: bool = False,
    now: str | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> dict:
    """Steps 2-4 for every retrieved instance; writes ``<staging>/verify.json``. ``ok`` per instance. With
    ``with_replica=N`` (needs ``api``) it also writes a per-stage attestation for every byte-identical
    main/replica generation (S5.10), and with ``canary`` it compares the canary units to main and quarantines
    an unsound reuse."""
    from scripts.hf_jobs.registry import DriverSpec, spec_for

    staging = _outside_repo(staging)
    man = mf.load()
    observed = _read_json(mf.OBSERVED) if mf.OBSERVED.is_file() else {}
    report: dict[str, Any] = {"commit": commit, "campaign": c.name, "instances": {}}
    for inst in instances(c, commit, replica=replica):
        rp = _record_path(staging, inst.id)
        if not rp.is_file():
            continue
        record = _read_json(rp)
        fresh = _artifact_dir(staging, inst.id)
        spec = spec_for(inst.job)
        row: dict[str, Any] = {"artifacts": {}}
        if isinstance(spec, DriverSpec):
            row["certification"] = _rebaseline(spec, fresh, commit, run=run)
            row["ok"] = bool(row["certification"]["ok"])
            report["instances"][inst.id] = row
            continue
        targets = artifacts_for(inst, man)
        if not targets:
            row.update(ok=False, reason="no manifest artifact for this instance")
            report["instances"][inst.id] = row
            continue
        row["load_error"] = load_check(inst.job, fresh, run=run)
        # ONE certification per instance, against its committed reference: the wheel bundle when the
        # instance has one (a Hub mirror / sibling shares it), else the Hub artifact's observed metrics.
        ref = next((a for a in targets if a.kind == "wheel"), targets[0])
        fresh_metrics = _read_json(fresh / "metrics.json") if (fresh / "metrics.json").is_file() else {}
        row["certification"] = _certify(spec, inst, ref, fresh, fresh_metrics, observed, run=run)
        oks = [row["load_error"] is None, bool(row["certification"]["ok"])]
        if inst.job in INTEGRITY_FAMILIES:
            row["integrity"] = _integrity(ref, fresh, record, _committed_metrics(ref, observed), staging, run=run)
            oks.append(bool(row["integrity"]["go"]))
        for a in targets:
            row["artifacts"][a.id] = {"violations": expectation_violations(a, fresh, record, commit)}
            oks.append(not row["artifacts"][a.id]["violations"])
        row["ok"] = all(oks)
        report["instances"][inst.id] = row
    now = now or datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    if with_replica is not None:
        report["attestations"] = _attestation_pass(c, commit, api=api, with_replica=with_replica, now=now)
    if canary:
        report["canary"], report["findings"] = _canary_pass(c, commit, api=api)
    if with_replica is not None or canary:
        _merge_comparisons(c, commit, _comparison_map(report, with_replica), api=api)
    (staging / "verify.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return report


def _comparison_map(report: Mapping[str, Any], with_replica: int | None) -> dict[str, bool]:
    """This verify call's byte-compares, scoped to (campaign, commit), for the S7.2 determinism bar (B#35): a
    main-vs-replica compare (identical True / diffs False) and a canary compare (ok). `cmd_summary` reads these
    instead of counting attestation FILES in the shared store (which span every campaign and omit a mismatch,
    since a failed compare writes no attestation).

    A stage is counted ONLY when it materialized BOTH sides: an attested-reuse skip or an empty generation is
    NOT a comparison (B#35a/B#47). A replica compare carries its replica number in the key (B#48): a matching
    ``--with-replica 2`` must never overwrite a ``--with-replica 1`` mismatch when `_merge_comparisons` merges."""
    comps: dict[str, bool] = {}
    for inst_id, row in report.get("attestations", {}).items():
        for stage, v in row.items():
            if v.get("materialized"):
                comps[f"replica|{with_replica}|{inst_id}|{stage}"] = bool(v.get("identical"))
    for inst_id, row in report.get("canary", {}).items():
        for stage, v in row.items():
            comps[f"canary|{inst_id}|{stage}"] = bool(v.get("ok"))
    return comps


def comparisons_rel(campaign: str, commit: str) -> str:
    """Where `verify` records this campaign's byte-compare results for the S7.2 determinism bar (B#35)."""
    return f"_campaign/{campaign}/{commit[:7]}/comparisons.json"


def _read_comparisons_map(c: Campaign, commit: str, *, api: Any) -> dict[str, bool]:
    from scripts.hf_jobs.store import BucketStore

    raw = BucketStore(c.bucket, "", api=api).read_bytes(comparisons_rel(c.name, commit))
    return dict(json.loads(raw)) if raw else {}


def _merge_comparisons(c: Campaign, commit: str, comps: dict[str, bool], *, api: Any) -> None:
    """MERGE this verify call's byte-compares into the campaign's record, keyed by ``kind|instance|stage`` so a
    later ``verify --canary`` never clobbers an earlier ``verify --with-replica`` mismatch (B#35b)."""
    from scripts.hf_jobs.store import BucketStore

    merged = {**_read_comparisons_map(c, commit, api=api), **comps}
    BucketStore(c.bucket, "", api=api).upload_bytes(
        comparisons_rel(c.name, commit), (json.dumps(merged, sort_keys=True) + "\n").encode("utf-8")
    )


def read_comparisons(c: Campaign, commit: str, *, api: Any) -> list[bool]:
    """The campaign-scoped byte-compare results `verify` recorded, accumulated across verify calls (B#35);
    empty when none were run."""
    return list(_read_comparisons_map(c, commit, api=api).values())


# ---- attestation + adoption rules (S5.10) ----------------------------------------------------------


def attest_rel(stage: str, token: str, commit: str) -> str:
    """The attestation path for a stage generation: ``_attest/{stage}/{token}/{commit}.json`` (S5.10)."""
    return f"_attest/{stage}/{token}/{commit}.json"


def attestations(store: Any, stage: str, token: str) -> list[dict]:
    """Every attestation recorded for ``(stage, token)`` in ``store`` (written by `verify`, `attest_rel`)."""
    prefix = f"_attest/{stage}/{token}"
    out: list[dict] = []
    for name in sorted(store.names(prefix)):
        raw = store.read_bytes(f"{prefix}/{name}")
        if raw:
            out.append(json.loads(raw))
    return out


def is_ancestor(commit: str, head: str, *, run: Callable[..., Any] = subprocess.run) -> bool:
    """True when ``commit`` is an ancestor of ``head`` (controller-side git; S5.10 adoption)."""
    p = run(["git", "merge-base", "--is-ancestor", commit, head], cwd=REPO_ROOT, capture_output=True, check=False)
    return p.returncode == 0


def oldest_ancestor_attester(commits: Iterable[str], *, order: Sequence[str]) -> str | None:
    """The OLDEST attesting commit on HEAD's history: the attester that comes LAST in HEAD's topo order
    (``git rev-list --topo-order HEAD``, newest first). None when no commit is in ``order``."""
    idx = {c: i for i, c in enumerate(order)}
    present = [c for c in commits if c in idx]
    return max(present, key=lambda c: idx[c]) if present else None


#: Run history, never a result: excluded from the main-vs-replica byte compare (r1 B CCTP-PLAN-14).
_ATTEST_EXCLUDE_SUFFIXES: tuple[str, ...] = (".ledger.json", ".failed.json")
_ATTEST_EXCLUDE_NAMES: frozenset[str] = frozenset({"_profile.json"})


def _attest_excluded(rel: str) -> bool:
    """True for a generation file that carries run history, not a result (S5.10): the ledger/failed sidecars,
    the profile, and anything under the ``.corrupt/`` tree. ``rel`` is relative to the generation dir."""
    parts = rel.split("/")
    if ".corrupt" in parts:
        return True
    return parts[-1] in _ATTEST_EXCLUDE_NAMES or rel.endswith(_ATTEST_EXCLUDE_SUFFIXES)


def _gen_tree(store: Any, gen: str) -> set[str]:
    """Every file under ``gen/`` relative to it (recursive via ``list_tree`` when the store has it, else the
    non-recursive ``names``), minus the run-history files the attestation never compares."""
    lister = getattr(store, "list_tree", None)
    names = lister(gen) if lister is not None else store.names(gen)
    return {n for n in names if not _attest_excluded(n)}


def generation_diff(main: Any, replica: Any, gen: str) -> list[str]:
    """The byte-level differences of generation ``gen`` between two stores (S5.10); empty == byte-identical over
    every result, sidecar, FileSet member, ``*.excluded.json``, ``*.counters.json``, ``_summary.json``,
    ``_inputs.json`` and the output. One diff string per differing path."""
    mnames, rnames = _gen_tree(main, gen), _gen_tree(replica, gen)
    diffs = [f"{gen}/{n}: in main only" for n in sorted(mnames - rnames)]
    diffs += [f"{gen}/{n}: in replica only" for n in sorted(rnames - mnames)]
    for n in sorted(mnames & rnames):
        if main.read_bytes(f"{gen}/{n}") != replica.read_bytes(f"{gen}/{n}"):
            diffs.append(f"{gen}/{n}: bytes differ")
    return diffs


def _job_env(phase_records: Mapping[str, Mapping]) -> dict:
    """The environment the run's jobs recorded (S5.10: `phases.publish_first_unit` writes ``env``). Every job of
    one run shares the environment (same image/commit/platform), so the first record carrying an ``env`` is it;
    ``{}`` when none does. Not keyed by stage -- the in-job label is ``job.variant.stage.shard.action``, which
    `parse_stage_label` cannot read, so a stage filter here silently returned ``{}`` for every real label."""
    for _label, rec in sorted(phase_records.items()):
        if rec.get("env"):
            return dict(rec["env"])
    return {}


def adoption_refusals(
    *,
    head_tokens: Mapping[str, str],
    adopted: Mapping[str, str],
    attestations: Mapping[str, Sequence[str]],
    quarantined: set[str],
    canary_ok: Mapping[str, bool],
    is_ancestor: Callable[[str], bool],
) -> list[str]:
    """Every reason an adopted generation must NOT land at a clean HEAD (S5.10), from both sides: its token must
    equal HEAD's token for the stage; some attesting commit must be an ANCESTOR of HEAD; the token must not be
    quarantined; and its canary must have passed. One string per failing (stage, rule)."""
    out: list[str] = []
    for stage, token in sorted(adopted.items()):
        if token != head_tokens.get(stage):
            out.append(f"{stage}: adopted token {token!r} != HEAD token {head_tokens.get(stage)!r}")
            continue
        atts = attestations.get(stage, [])
        if not atts:
            out.append(f"{stage}: no attestation for token {token!r}")
        elif not any(is_ancestor(c) for c in atts):
            out.append(f"{stage}: no attesting commit is an ancestor of HEAD (token {token!r})")
        if token in quarantined:
            out.append(f"{stage}: token {token!r} is quarantined (unsound reuse)")
        if not canary_ok.get(stage, False):
            out.append(f"{stage}: canary did not pass for token {token!r}")
    return out


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
                data = (json.dumps(rehydrate_public_corpus_ids(json.loads(data)), indent=2) + "\n").encode("utf-8")
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


def _observed_entry(d: pathlib.Path, c_name: str, commit: str, now: str, *, head: str | None = None) -> dict:
    """The observed fields of an adopted artifact (INV-9). ``commit`` is the artifact's training commit (the
    attester). On a REUSE adoption (``head`` differs from it) the ADR-129 provenance -- ``adopted_at_commit``
    (the clean HEAD) and ``attested_by`` -- is recorded here, never in the human manifest."""
    from scripts.hf_jobs.identity import artifact_identity

    metrics = _read_json(d / "metrics.json") if (d / "metrics.json").is_file() else {}
    meta = _read_json(d / "metadata.json") if (d / "metadata.json").is_file() else {}
    entry = {
        "sha256": artifact_identity(d)["files"],
        "metrics_summary": metrics,
        "env": metrics.get("env") or meta.get("env") or {},
        "campaign": f"{c_name}@{commit[:7]}",
        "adopted_at": now,
    }
    if head is not None and head != commit:
        entry["adopted_at_commit"] = head
        entry["attested_by"] = commit
    return entry


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
        lines.append(f"- `{aid}` ({a.destination}): job `{a.job}`, variant `{a.variant}`, tier `{a.tier}`.")
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


def stamp_provenance(
    directory: str | pathlib.Path,
    *,
    attestation: Mapping[str, Any],
    params: Mapping[str, Any],
    generation: str | None = None,
) -> None:
    """S5.10: stamp the adopting commit's provenance onto a COMMIT-FREE artifact and rewrite SHA256SUMS. The
    adopting commit is ``attestation["commit"]``; the tree state and platform come from the attesting MAIN job's
    recorded provenance (``attestation["main"]``). ``reproducibility_note`` is rebuilt from the artifact's own
    ``reproducibility`` label and ``providers`` (both commit-free). SHA256SUMS is written LAST. The core is
    `identity.stamp_provenance`, shared with the local ``--run-all --out`` export."""
    from scripts.hf_jobs import identity

    main = attestation.get("main", {})
    identity.stamp_provenance(
        directory,
        commit=str(attestation["commit"]),
        tree_state=main.get("tree_state"),
        platform=(main.get("env") or {}).get("platform"),
        training_platform=params.get("training_platform"),
        generation=generation,
    )


def _topo_order(head: str, run: Callable[..., Any]) -> list[str]:
    """HEAD's history newest first (``git rev-list --topo-order``), for `oldest_ancestor_attester`."""
    p = run(["git", "rev-list", "--topo-order", head], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    out: list[str] = str(p.stdout or "").split()
    return out


def _has_training_commit(fresh: pathlib.Path) -> bool:
    """True when the retrieved artifact already carries ``training_commit`` (a pre-ADR-129 output); a
    commit-free output (Task 16) does not, and `adopt` stamps it from its attestation."""
    for name in ("metadata.json", "metrics.json"):
        p = fresh / name
        if p.is_file() and "training_commit" in _read_json(p):
            return True
    return False


def _assert_adoptable(
    c: Campaign, commit: str, head: str, rep: Mapping[str, Any], *, api: Any, run: Callable[..., Any]
) -> None:
    """Refuse an adoption that ADR-129 S5.10 does not permit. With no ``api`` (the pure-diff unit tests) it
    keeps the ADR-037 rule that HEAD IS the campaign commit. With ``api`` it recomputes each stage's token at
    HEAD (`read_plan`) and runs `adoption_refusals`: every adopted generation's token == HEAD's token, an
    attestation on HEAD or an ancestor, its canary passed (or HEAD attested it this campaign), not quarantined."""
    if api is None:
        if head != commit:
            raise AdoptionRefusedError(f"HEAD {head} is not the campaign commit {commit} (ADR-037)")
        return
    from scripts.hf_jobs.campaign import instances, read_plan
    from scripts.hf_jobs.store import BucketStore

    head_tokens: dict[str, str] = {}
    atts: dict[str, list[str]] = {}
    canary_ok: dict[str, bool] = {}
    quarantined: set[str] = set()
    canary_rep = rep.get("canary", {})
    for inst in instances(c, commit):
        store = BucketStore(c.bucket, inst.run_key, api=api)
        for st in read_plan(inst, bucket=c.bucket)["stages"]:
            stage, token = st["name"], st["token"]
            key = f"{inst.id}|{stage}"  # compound: two instances can share a stage NAME (CCTP-IMPL-31)
            head_tokens[key] = token
            commits = [a.get("commit", "") for a in attestations(store, stage, token)]
            atts[key] = commits
            attested_here = head in commits  # verify wrote an attestation at HEAD this campaign (INV-1)
            canary_ok[key] = attested_here or bool(canary_rep.get(inst.id, {}).get(stage, {}).get("ok"))
            if store.list_tree(f"_quarantine/{stage}/{token}"):  # recursive: quarantine writes {utc}/ subdirs
                quarantined.add(token)
    refusals = adoption_refusals(
        head_tokens=head_tokens,
        adopted=dict(head_tokens),
        attestations=atts,
        quarantined=quarantined,
        canary_ok=canary_ok,
        is_ancestor=lambda a: is_ancestor(a, head, run=run),
    )
    if refusals:
        raise AdoptionRefusedError(f"adoption refused (ADR-129 S5.10): {refusals}")


def _stamp_fresh(
    c: Campaign, commit: str, head: str, inst_id: str, fresh: pathlib.Path, *, api: Any, run: Callable[..., Any]
) -> str:
    """Stamp a COMMIT-FREE retrieved artifact (Task 16) with the provenance of the OLDEST ancestor attester of
    its output generation (S5.10) and return that attester commit (the artifact's ``training_commit``)."""
    from scripts.hf_jobs.campaign import instances, read_plan
    from scripts.hf_jobs.store import BucketStore

    inst = {i.id: i for i in instances(c, commit)}[inst_id]
    store = BucketStore(c.bucket, inst.run_key, api=api)
    plan = read_plan(inst, bucket=c.bucket)
    out = next(s for s in plan["stages"] if s["name"] == plan["output"])
    by_commit = {a.get("commit", ""): a for a in attestations(store, out["name"], out["token"])}
    order = _topo_order(head, run)
    attester = oldest_ancestor_attester([a for a in by_commit if is_ancestor(a, head, run=run)], order=order)
    if attester is None:
        raise AdoptionRefusedError(f"{inst_id}: no ancestor attestation for {out['name']}/{out['token']}")
    stamp_provenance(
        fresh, attestation=by_commit[attester], params={"training_platform": None}, generation=out["token"]
    )
    return attester


def adopt(
    c: Campaign,
    commit: str,
    *,
    staging: pathlib.Path,
    api: Any = None,
    now: str | None = None,
    run: Callable[..., Any] = subprocess.run,
    git: Callable[[], Mapping[str, Any]] | None = None,
    root: pathlib.Path = REPO_ROOT,
    gate: bool = True,
) -> dict:
    """Spec 5.7 steps 5-10 over a verified staging dir; never commits. Refuses (writing nothing) on a dirty
    tree, a ``verify.json`` of another commit, any instance that did not verify, or an adoption the ADR-129
    gate (`_assert_adoptable`) rejects. With ``api`` it reuses an attested generation at a later clean HEAD and
    stamps each commit-free output; without it the ADR-037 rule (HEAD IS the campaign commit) stands.
    Idempotent: a second run over the same staging dir writes nothing."""
    from scripts._provenance import git_provenance, require_clean_tree

    root = pathlib.Path(root).resolve()
    staging = _outside_repo(staging, root)
    prov = dict((git or git_provenance)())
    try:
        require_clean_tree(prov, allow_dirty=False)
    except SystemExit as exc:
        raise AdoptionRefusedError(f"adopt needs a clean tree (ADR-037): {exc}") from exc
    head = str(prov.get("commit"))
    rep = _read_json(staging / "verify.json")
    if rep.get("commit") != commit:
        raise AdoptionRefusedError(f"verify.json is for commit {rep.get('commit')}, not {commit}")
    bad = sorted(k for k, v in rep["instances"].items() if not v.get("ok"))
    if bad:
        raise AdoptionRefusedError(f"instances that did not verify: {bad}")
    _assert_adoptable(c, commit, head, rep, api=api, run=run)
    manifest_path = root / "docs" / "artifacts" / "manifest.toml"
    observed_path = root / "docs" / "artifacts" / "observed.json"
    man = mf.load(manifest_path)
    now = now or datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    before = _git_status(run, root)
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
        # A commit-free output (Task 16) is stamped from its OLDEST ancestor attestation (S5.10); its stamped
        # commit becomes the artifact's training_commit. A pre-ADR-129 output keeps the campaign commit.
        attester = commit
        if api is not None and not _has_training_commit(fresh):
            attester = _stamp_fresh(c, commit, head, inst_id, fresh, api=api, run=run)
        for aid in sorted(row.get("artifacts", {})):
            a = man.artifacts[aid]
            if a.kind == "wheel":
                written += _bundle(a, fresh, root)
                observed[aid] = _observed_entry(root / a.path, c.name, attester, now, head=head)
            else:
                observed[aid] = _observed_entry(fresh, c.name, attester, now, head=head)
            manifest_text = _set_manifest_commit(manifest_text, aid, attester)
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
    stray = sorted(p for p in _git_status(run, root) - before if not _allowed(p, man))
    if stray:
        raise AdoptionRefusedError(f"adopt changed paths outside the allowlist (ADR-037): {stray}")
    out: dict[str, Any] = {"adopted": adopted, "written": sorted(set(written)), "changelog_draft": str(draft)}
    if gate:
        out["gate"] = _local_gate(run, root)
    out["next"] = "commit gate: owner approval required"
    return out


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
    to the local one, README equal to `card_bytes(card)`, and a ``from_hub`` round-trip in ``.venv-hfjobs``."""
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
        d = _artifact_dir(staging, src[a.id])
        if sandbox:
            api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        (publisher or publish_model_with_card)(api, str(d), repo, model_card=str(REPO_ROOT / (a.card or "")))
        results[a.id] = {
            "repo": repo,
            **post_verify(
                api, repo, a, d, run=run, allowlist=MODEL_ONLY_ALLOWLIST, card=card_bytes(REPO_ROOT / (a.card or ""))
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
) -> dict:
    """What the Hub holds after a publish: model-only, the local SHA256SUMS, the rendered card, loadable."""
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
    mod, _, cls = LOADERS[a.job].partition(":")
    code = f"import importlib,sys; getattr(importlib.import_module({mod!r}), {cls!r}).from_hub(sys.argv[1])"
    p = run([str(HFJOBS_PYTHON), "-c", code, repo], capture_output=True, text=True, check=False, cwd=REPO_ROOT)
    if p.returncode != 0:
        problems.append(f"from_hub failed: {(p.stderr or '')[-500:]}")
    return {"ok": not problems, "problems": problems, "files": files}
