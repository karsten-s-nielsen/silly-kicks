"""Preflight (spec 5.4): six no-spend checks. `launch` refuses without a passing report for the same
commit and the same campaign-file hash. The report lives in the bucket (the controller is stateless)."""

from __future__ import annotations

import dataclasses
import datetime
import inspect
import json
import re
import subprocess
import sys
import tempfile
from typing import Any

REPORT_REL = "_campaign/{campaign}/{commit7}/preflight.json"
_RUN_JOB_PARAMS = {"image", "command", "env", "secrets", "flavor", "timeout", "attempts", "labels", "volumes", "name"}


@dataclasses.dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    flags: tuple[str, ...] = ()


def _sh(run, argv: list[str], *, env: dict | None = None) -> tuple[int, str]:
    kw = {} if env is None else {"env": env}
    p = run(argv, capture_output=True, text=True, check=False, **kw)
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
    """Check 3: the pinned `run_job`/`list_jobs` expose the typed parameters; cpu-upgrade is 8 vCPU / 32 GB."""
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
    """Check 4: every staged dataset exists at its pinned revision and matches its lock record."""
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


def check_plans(instances: list[Any], *, run=subprocess.run) -> Check:
    """Check 5: every job's `--plan` and (once per module) a toy `--run-all` pass in this environment.

    This is where the 4.138.0 real-HF defects would have shown up without spend: an empty study
    enumeration, missing arms, a dotdir crash. Open-data modules run against a locally served toy tree.
    """
    import os

    from scripts._toy_corpora import TOY_PARAMS, toy_session

    bad = []
    toyed: set[str] = set()
    for inst in instances:
        with tempfile.TemporaryDirectory() as tmp:
            argv = [sys.executable, "-m", inst.module, *inst.args, "--plan"]
            argv += ["--store", f"{tmp}/s", "--data-root", f"{tmp}/d"]
            rc, out = _sh(run, argv)
            if rc or not json.loads(out or "{}").get("stages"):
                bad.append(f"{inst.job}/{inst.variant}: --plan failed")
                continue
        if inst.module not in TOY_PARAMS or inst.module in toyed:
            continue
        toyed.add(inst.module)
        with tempfile.TemporaryDirectory() as tmp, toy_session(inst.module, f"{tmp}/d") as (data, extra_env):
            argv = [sys.executable, "-m", inst.module, *TOY_PARAMS[inst.module], "--run-all"]
            argv += ["--store", f"{tmp}/s", "--data-root", str(data), "--allow-dirty"]
            rc, _ = _sh(run, argv, env={**os.environ, **extra_env})
            if rc:
                bad.append(f"{inst.module}: toy --run-all failed")
    return Check("plan", not bad, "; ".join(sorted(set(bad))) or "every --plan and toy --run-all passed")


def assemble_report(
    campaign: str, commit: str, campaign_sha256: str, checks: list[Check], *, flags: list[str], now: str | None = None
) -> dict:
    return {
        "campaign": campaign,
        "commit": commit,
        "campaign_sha256": campaign_sha256,
        "created_at": now or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "checks": [dataclasses.asdict(c) for c in checks],
        "flags": list(flags),
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
