"""Preflight (spec 5.4): six no-spend checks. `launch` refuses without a passing report for the same
commit and the same campaign-file hash. The report lives in the bucket (the controller is stateless)."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import inspect
import json
import pathlib
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from typing import Any

REPORT_REL = "_campaign/{campaign}/{commit7}/preflight.json"
PREFLIGHT_CACHE_REL = "_campaign/preflight_cache.json"
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


def _origin_repo(run) -> str | None:
    """The ``owner/repo`` slug from ``origin``, for both ``https://github.com/o/r(.git)`` and
    ``git@github.com:o/r(.git)`` forms. None when ``origin`` is absent or not a GitHub remote."""
    rc, url = _sh(run, ["git", "remote", "get-url", "origin"])
    if rc:
        return None
    m = re.search(r"github\.com[/:]([^/]+/[^/\s]+?)(?:\.git)?/?$", url.strip())
    return m.group(1) if m else None


def _github_ci_runs(repo: str, commit: str) -> list[dict]:
    """The GitHub Actions workflow runs for ``commit`` on a PUBLIC repo, read ANONYMOUSLY over the REST API
    -- the controller needs no ``gh`` CLI and no token (owner: it cannot be brittle to missing GH access). A
    token in ``GH_TOKEN``/``GITHUB_TOKEN`` is used ONLY to raise the anonymous rate limit, never required.
    Each run carries ``name`` (the workflow name), ``status`` and ``conclusion``."""
    import os
    import urllib.request

    url = f"https://api.github.com/repos/{repo}/actions/runs?head_sha={commit}&per_page=50"
    req = urllib.request.Request(  # noqa: S310 -- fixed https host (api.github.com)
        url, headers={"Accept": "application/vnd.github+json", "User-Agent": "silly-kicks-controller"}
    )
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 -- fixed https host
        return json.loads(resp.read().decode("utf-8")).get("workflow_runs", [])


def check_ref(commit: str, *, run=subprocess.run, fetch_ci=None) -> Check:
    """Check 1: the clone commit is local HEAD, is on a remote branch, and has a green CI run. The CI-green
    read is an ANONYMOUS GitHub REST read of the public repo (``fetch_ci``, default ``_github_ci_runs``) --
    no ``gh`` CLI, no token. A read failure degrades to a clear not-ok, never a crash."""
    rc, head = _sh(run, ["git", "rev-parse", "HEAD"])
    if rc or head.strip() != commit:
        return Check("ref", False, f"HEAD {head.strip()[:12]} != campaign commit {commit[:12]}")
    _sh(run, ["git", "fetch", "-q", "origin"])
    rc, remote = _sh(run, ["git", "branch", "-r", "--contains", commit])
    if rc or not remote.strip():
        return Check("ref", False, f"{commit[:12]} is not on any remote branch (push it first)")
    repo = _origin_repo(run)
    if repo is None:
        return Check("ref", False, "cannot derive the GitHub owner/repo from `origin`")
    fetch = fetch_ci or _github_ci_runs
    try:
        runs = fetch(repo, commit)
    except Exception as exc:  # network / rate-limit / parse -- do NOT crash preflight on a read failure
        return Check("ref", False, f"could not read CI status for {commit[:12]} from GitHub ({exc})")
    green = [
        r for r in runs if r.get("status") == "completed" and r.get("conclusion") == "success" and r.get("name") == "CI"
    ]
    if not green:
        seen = [(r.get("name"), r.get("conclusion")) for r in runs]
        return Check("ref", False, f"no green CI run for {commit[:12]} (saw {seen})")
    return Check("ref", True, f"{commit[:12]} pushed and CI green (anonymous read; no gh/token required)")


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


#: The repo root, so the key anchors its paths there and NEVER to the process cwd: running the controller from
#: another directory must not collapse the tree/lock digests to a constant and reuse a stale PASS (reviewer B
#: D1-IMPL-14). `preflight.py` is `scripts/hf_jobs/preflight.py`, so the root is two parents up.
_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _file_digest(path: Any) -> str:
    """The sha256 of one file's bytes, or ``"absent"`` when it is missing (HFU T9 key input). A relative path
    is resolved against the repo root, never the process cwd (reviewer B D1-IMPL-14)."""
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = _ROOT / p
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "absent"


def _tree_digest(globs: Sequence[str]) -> str:
    """A content digest over every file matching ``globs`` UNDER THE REPO ROOT, in sorted path order: each file
    contributes its repo-relative posix path then its bytes, so a rename or an edit moves the digest, the digest
    is independent of the process cwd AND of the absolute checkout location (the three clones agree) -- reviewer
    B D1-IMPL-14 / HFU T9 / B-13."""
    h = hashlib.sha256()
    for pat in globs:
        for p in sorted(_ROOT.glob(pat), key=lambda q: q.as_posix()):
            if p.is_file():
                h.update(p.relative_to(_ROOT).as_posix().encode("utf-8"))
                h.update(b"\0")
                h.update(p.read_bytes())
                h.update(b"\0")
    return h.hexdigest()


def check_plans_key(campaign_sha256: str, *, image: str | None = None) -> str:
    """The cache key of the ~20-30 min ``check_plans`` toy train (spec F / B-13): the campaign hash, the
    ``uv.lock`` digest, the pinned image digest, the ``datasets.lock.json`` digest and a tree digest of
    ``scripts/**/*.py``, ``silly_kicks/**/*.py`` and ``pyproject.toml``. A change to ANY of these re-runs the
    toy train; nothing else does."""
    from scripts.hf_jobs.datasets import LOCK
    from scripts.hf_jobs.launch import IMAGE

    parts = [
        campaign_sha256,
        _file_digest("uv.lock"),
        image or IMAGE,
        _file_digest(LOCK),
        _tree_digest(["scripts/**/*.py", "silly_kicks/**/*.py"]),
        _file_digest("pyproject.toml"),
    ]
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _read_cache(store: Any) -> dict:
    """The per-check cache, or ``{}`` when it is missing or unreadable (a full recompute, never a crash)."""
    raw = store.read_bytes(PREFLIGHT_CACHE_REL)
    if raw is None:
        return {}
    try:
        got = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return got if isinstance(got, dict) else {}


def _write_cache(store: Any, cache: dict) -> None:
    store.upload_bytes(PREFLIGHT_CACHE_REL, json.dumps(cache, indent=2, sort_keys=True).encode("utf-8"))


def cached_check_plans(
    instances: list[Any], *, store: Any, campaign_sha256: str, image: str | None = None, run=subprocess.run
) -> Check:
    """``check_plans`` with its toy train cached by input key (spec F). A stored PASS for the current key is
    reused byte-identically (the ~20-30 min train is skipped); any input change re-runs it. Only a passing
    result is cached. A missing or unreadable cache forces a full recompute. ONLY ``check_plans`` is cached;
    the cheap checks and ``check_ref`` always re-observe, so they are never stored here."""
    key = f"plan:{check_plans_key(campaign_sha256, image=image)}"
    cache = _read_cache(store)
    hit = cache.get(key)
    if isinstance(hit, dict) and hit.get("ok"):
        return Check(hit["name"], hit["ok"], hit["detail"], tuple(hit.get("flags", ())))
    result = check_plans(instances, run=run)
    if result.ok:
        cache[key] = dataclasses.asdict(result)
        _write_cache(store, cache)
    return result


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
