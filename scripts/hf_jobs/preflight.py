"""Preflight (spec 5.4, HFU spec 13.2, HFU fit-units spec C3): seven no-spend checks. `launch` refuses without
a passing report for the same commit and the same campaign-file hash. The report lives in the bucket (the
controller is stateless).

Each check prints one ``[preflight]`` line with its elapsed seconds as it finishes; check 6 also prints one line
per instance. No check trains anything: CI's parity tests run every migrated trainer end to end on toy data, and a
local toy run reproduces none of the failures that stop real HF runs (owner decision 2026-11-05)."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from typing import Any

REPORT_REL = "_campaign/{campaign}/{commit7}/preflight.json"
_RUN_JOB_PARAMS = {"image", "command", "env", "secrets", "flavor", "timeout", "attempts", "labels", "volumes", "name"}
#: Concurrent `--plan` subprocesses in check 6. Each is its own Python process: 3.6-19.7 s per instance, 45 s for
#: all 21 at 8 wide (cycle1, measured 2026-11-05).
PLAN_WORKERS = min(8, os.cpu_count() or 1)


@dataclasses.dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _say(line: str) -> None:
    print(line, flush=True)


def _sh(run, argv: list[str]) -> tuple[int, str]:
    p = run(argv, capture_output=True, text=True, check=False)
    return p.returncode, (p.stdout or "")


def check_ref(commit: str, *, run=subprocess.run) -> Check:
    """Check 1: the clone commit is local HEAD, is on a remote branch, and has a green CI run."""
    rc, head = _sh(run, ["git", "rev-parse", "HEAD"])
    if rc or head.strip() != commit:
        return Check("ref", False, f"HEAD {head.strip()[:12]} != campaign commit {commit[:12]}")
    _sh(run, ["git", "fetch", "-q", "origin"])
    rc, remote = _sh(run, ["git", "branch", "-r", "--contains", commit])
    if rc or not remote.strip():
        return Check("ref", False, f"{commit[:12]} is not on any remote branch (push it first)")
    rc, out = _sh(
        run, ["gh", "run", "list", "--commit", commit, "--json", "status,conclusion,workflowName", "--limit", "50"]
    )
    runs = json.loads(out or "[]") if rc == 0 else []
    green = [
        r
        for r in runs
        if r.get("status") == "completed" and r.get("conclusion") == "success" and r.get("workflowName") == "CI"
    ]
    if not green:
        seen = [(r.get("workflowName"), r.get("conclusion")) for r in runs]
        return Check("ref", False, f"no green CI run for {commit[:12]} (saw {seen})")
    return Check("ref", True, f"{commit[:12]} pushed and CI green")


def _image_resolves(image: str) -> bool:
    import urllib.request

    repo, _, digest = image.partition("@")
    name = repo.split(":")[0]
    token_url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/{name}:pull"
    with urllib.request.urlopen(token_url, timeout=30) as resp:  # noqa: S310 -- fixed https Docker Hub URL
        token = json.load(resp)["token"]
    req = urllib.request.Request(
        f"https://registry-1.docker.io/v2/library/{name}/manifests/{digest}",
        method="HEAD",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": (
                "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json"
            ),
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return resp.status == 200


#: The HF job platform (`launch.IMAGE` is python:3.12.14-bookworm on x86_64) for evaluating `uv export` markers.
JOB_MARKER_ENV = {
    "sys_platform": "linux",
    "platform_system": "Linux",
    "platform_machine": "x86_64",
    "os_name": "posix",
    "python_version": "3.12",
    "python_full_version": "3.12.14",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
}
#: The job's install line (`launch.py` `uv sync --frozen ... --extra train ... --group hfjobs`) as an export.
JOB_EXPORT_ARGV = ["uv", "export", "--frozen", "--no-hashes", "--extra", "train", "--group", "hfjobs"]


def check_numba(*, run=subprocess.run) -> Check:
    """Check 3 (HFU spec 13.2): numba is in the job's LOCKED install set on the job platform. A static check of the
    lock, because preflight runs on the controller, where an `import numba` would prove the wrong wheel."""
    from packaging.requirements import InvalidRequirement, Requirement

    rc, out = _sh(run, JOB_EXPORT_ARGV)
    if rc:
        return Check("numba", False, f"`{' '.join(JOB_EXPORT_ARGV)}` failed (rc={rc})")
    for line in out.splitlines():
        text = line.split("#", 1)[0].strip()
        if not text or text.startswith(("-", "\\")):
            continue
        try:
            req = Requirement(text)
        except InvalidRequirement:
            continue
        if req.name.lower() != "numba":
            continue
        if req.marker is None or req.marker.evaluate(JOB_MARKER_ENV):
            return Check("numba", True, f"numba{req.specifier} in the locked job install set (linux cp312)")
    return Check("numba", False, "numba is NOT in the locked job install set for linux cp312 (--extra train)")


def check_lock(*, run=subprocess.run) -> Check:
    """Check 2: `uv lock --check` passes and the pinned image digest resolves on Docker Hub."""
    from scripts.hf_jobs.launch import IMAGE

    rc, _ = _sh(run, ["uv", "lock", "--check"])
    if rc:
        return Check("lock", False, "uv lock --check failed: uv.lock is stale")
    try:
        ok = _image_resolves(IMAGE)
    except OSError as exc:
        return Check("lock", False, f"image digest did not resolve: {exc}")
    return Check("lock", ok, f"uv.lock current; {IMAGE} resolves" if ok else f"{IMAGE} does not resolve")


def check_api(api: Any) -> Check:
    """Check 4: the pinned `run_job`/`list_jobs` expose the typed parameters; cpu-upgrade is 8 vCPU / 32 GB."""
    params = set(inspect.signature(api.run_job).parameters)
    missing = _RUN_JOB_PARAMS - params
    if missing:
        return Check("api", False, f"run_job lacks {sorted(missing)}")
    if "labels" not in inspect.signature(api.list_jobs).parameters:
        return Check("api", False, "list_jobs lacks labels=")
    hw = [h for h in api.list_jobs_hardware() if getattr(h, "name", None) == "cpu-upgrade"]
    if not hw:
        return Check("api", False, "the flavor catalog lacks cpu-upgrade")
    cpu = int(re.findall(r"\d+", str(hw[0].cpu))[0])
    ram = int(re.findall(r"\d+", str(hw[0].ram))[0])
    if (cpu, ram) != (8, 32):
        return Check(
            "api", False, f"the cpu-upgrade flavor is now {cpu} vCPU / {ram} GB, not 8 / 32: re-plan before spending"
        )
    # the catalog prices per `unit_label` (huggingface_hub JobHardwareInfo: "minute"), never per hour
    cost, unit = getattr(hw[0], "unit_cost_usd", "?"), getattr(hw[0], "unit_label", "?")
    return Check("api", True, f"run_job/list_jobs typed surface present; cpu-upgrade 8 vCPU / 32 GB at ${cost}/{unit}")


def check_datasets(names: list[str], *, api: Any) -> Check:
    """Check 5: every staged dataset exists at its pinned revision and matches its lock record."""
    from scripts.hf_jobs.datasets import declared, locked

    bad = []
    for name in sorted(set(names)):
        d = declared()[name]
        try:
            rec = locked(name)
        except KeyError as exc:
            bad.append(str(exc))
            continue
        if d.kind == "statsbomb_open":
            continue  # a commit SHA is immutable; `record` already enumerated it
        try:
            files = set(api.list_repo_files(d.repo, repo_type="dataset", revision=rec["revision"]))
        except Exception as exc:  # reported per dataset, never swallowed
            bad.append(f"{name}: revision {rec['revision'][:12]} unreadable ({exc})")
            continue
        for u in rec["units"]:
            want = (
                f"shards/{rec['generation']}/{u['stem']}.parquet"
                if d.kind == "flat_tc3"
                else f"{u['stem']}/frames.parquet"
            )
            if want not in files:
                bad.append(f"{name}: {want} missing at the pinned revision")
                break
        if d.kind == "flat_tc3" and rec.get("home_teams", True) and "home_teams.json" not in files:
            bad.append(f"{name}: home_teams.json missing")
    return Check("datasets", not bad, "; ".join(bad) or "every dataset matches its lock")


def check_storage(api: Any, *, org: str, limit_bytes: int, headroom_bytes: int) -> Check:
    """Check 7 (HFU fit-units spec C3): the org's private storage leaves ``headroom_bytes`` under ``limit_bytes``.

    Used = the size of every bucket of ``org`` plus ``usedStorage`` of each PRIVATE dataset and model of ``org``
    (launch 7 stopped on the org limit; at the limit HF also refuses the bucket delete, so the cleanup must come
    before the launch). The detail lists the used bytes by repo, largest first. An API error, a repo without a
    size or a listing entry without ``private`` FAILS the check with its reason: a count that cannot be made is
    not a pass. This is a guard computed
    from two API figures; it is not HF's own accounting."""
    used: dict[str, int] = {}
    try:
        for b in api.list_buckets(org):
            used[str(b.id)] = _size(b.id, getattr(b, "size", None))
        for kind, lister, info in (
            ("dataset", api.list_datasets, api.dataset_info),
            ("model", api.list_models, api.model_info),
        ):
            for repo in lister(author=org):
                private = getattr(repo, "private", None)
                if private is None:  # it cannot be classed: counting it as public would fail open
                    raise ValueError(f"{repo.id} ({kind}) has no private field in the API answer")
                if private:
                    used[f"{repo.id} ({kind})"] = _size(
                        repo.id, getattr(info(repo.id, expand=["usedStorage"]), "used_storage", None)
                    )
    except Exception as exc:
        return Check("storage", False, f"the storage count failed: {type(exc).__name__}: {exc}")
    total = sum(used.values())
    by_repo = ", ".join(f"{name} {size / 1e9:.2f} GB" for name, size in sorted(used.items(), key=lambda kv: -kv[1]))
    detail = (
        f"{total / 1e9:.2f} GB used + {headroom_bytes / 1e9:.0f} GB headroom against a {limit_bytes / 1e9:.0f} GB "
        f"limit: {by_repo}"
    )
    return Check("storage", total + headroom_bytes <= limit_bytes, detail)


def _size(name: Any, value: Any) -> int:
    if value is None:
        raise ValueError(f"{name} has no size in the API answer")
    return int(value)


def _plan_one(inst: Any, run) -> tuple[bool, str, float]:
    """(ok, last stderr line, seconds) for one instance's ``--plan`` in a fresh temp store."""
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        root = f"{tmp}/{inst.job}"
        argv = [
            sys.executable,
            "-m",
            inst.module,
            *inst.args,
            "--plan",
            "--store",
            f"{root}/s",
            "--data-root",
            f"{root}/d",
        ]
        p = run(argv, capture_output=True, text=True, check=False)
    try:
        ok = p.returncode == 0 and bool(json.loads(p.stdout or "{}").get("stages"))
    except json.JSONDecodeError:
        ok = False
    err = ([ln for ln in (p.stderr or "").splitlines() if ln.strip()] or ["(no stderr)"])[-1]
    return ok, ("" if ok else err), time.monotonic() - t0


def check_plans(
    instances: list[Any], *, run=subprocess.run, workers: int = PLAN_WORKERS, say: Callable[[str], None] = _say
) -> Check:
    """Check 6: every instance's ``--plan`` passes in this environment, with the campaign's own arguments.

    The instances run in parallel (``workers``), each in its own temp store; each prints one line as it ends.
    ``--plan`` is where the 4.138.0 real-HF defects would have shown up without spend: an empty study
    enumeration, missing arms, a dotdir crash.
    """
    bad = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_plan_one, inst, run): inst for inst in instances}
        for fut in concurrent.futures.as_completed(futures):
            inst = futures[fut]
            ok, err, seconds = fut.result()
            name = f"{inst.job}/{inst.variant}"
            say(f"[preflight] plan {name} {'ok' if ok else 'FAILED'} {seconds:.1f}s" + ("" if ok else f": {err}"))
            if not ok:
                bad.append(f"{name}: --plan failed: {err}")
    return Check("plan", not bad, "; ".join(sorted(bad)) or f"every --plan passed ({len(instances)} instances)")


def run_checks(checks: list[tuple[str, Callable[[], Check]]], *, say: Callable[[str], None] = _say) -> list[Check]:
    """Run each check in order and print one timed ``[preflight] <name> ok|FAIL <s>s: <detail>`` line as it ends."""
    out = []
    for name, check in checks:
        t0 = time.monotonic()
        c = check()
        say(f"[preflight] {name} {'ok' if c.ok else 'FAIL'} {time.monotonic() - t0:.1f}s: {c.detail}")
        out.append(c)
    return out


def assemble_report(
    campaign: str, commit: str, campaign_sha256: str, checks: list[Check], *, now: str | None = None
) -> dict:
    return {
        "campaign": campaign,
        "commit": commit,
        "campaign_sha256": campaign_sha256,
        "created_at": now or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "checks": [dataclasses.asdict(c) for c in checks],
        "ok": all(c.ok for c in checks),
    }


def require_preflight(campaign: str, commit: str, campaign_sha256: str, *, store: Any) -> dict:
    """The launch gate: a passing report for THIS commit and THIS campaign-file hash, or refuse."""
    raw = store.read_bytes(REPORT_REL.format(campaign=campaign, commit7=commit[:7]))
    if raw is None:
        raise SystemExit(f"no preflight report for {campaign}@{commit[:7]}: run `campaign <file> preflight` first")
    rep = json.loads(raw)
    if rep.get("commit") != commit:
        raise SystemExit(f"preflight report is for commit {rep.get('commit', '?')[:12]}, not {commit[:12]}")
    if rep.get("campaign_sha256") != campaign_sha256:
        raise SystemExit("the campaign file changed since preflight: re-run preflight")
    if not rep.get("ok"):
        raise SystemExit(f"preflight failed: {[c for c in rep['checks'] if not c['ok']]}")
    return rep
