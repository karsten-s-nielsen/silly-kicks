"""The image registry + startup probes (ADR-129 S5.12, E-1/E-6). cpu-upgrade only (pre-approved exploration
spend); every outward publish/commit stays owner-gated.

``probe-registry`` launches one tiny ``cpu-upgrade`` job per candidate reference and reports which pulled.
``probe-startup`` launches the five S5.12 startup jobs (the win_probability ``refs,extract --profile 1`` job
command on the prebuilt image), waits, reads their job phase records and writes ``startup_s`` /
``startup_samples`` / ``startup_jobs`` into ``image.lock.json``. It refuses a dirty tree except for
``image.lock.json`` itself.

This module imports `HfApi` and the launcher, so it is NOT imported at module load by `image` (only inside the
`image probe-*` CLI branch) and never reaches an HF job's import closure (the I1 gate)."""

from __future__ import annotations

import json
import time
from typing import Any

from scripts.hf_jobs import image

TERMINAL = frozenset({"COMPLETED", "CANCELED", "ERROR", "DELETED"})
_STARTUP_JOBS = 5


def _candidates() -> tuple[str, ...]:
    """References to probe, in order: the digest-pinned prebuilt image, then the base image as a fallback
    signal. The digest-pinned ref (not the untagged ``REGISTRY_REF``) is what a job actually pulls."""
    lock = image.load_lock()
    refs = []
    if lock.get("digest"):
        refs.append(image.image_ref())
    refs.append(image.BASE_IMAGE)
    return tuple(refs)


def _api() -> Any:
    from huggingface_hub import HfApi, get_token

    if get_token() is None:
        raise SystemExit("no HF token: run `hf auth login` first")
    return HfApi()


def _wait(api: Any, job_id: str, *, timeout_s: int = 1800, poll_s: int = 20) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = api.inspect_job(job_id=job_id)
        stage = str(getattr(getattr(info, "status", None), "stage", "") or "")
        if stage in TERMINAL:
            return stage
        time.sleep(poll_s)
    return "TIMEOUT"


def probe_registry(api: Any = None) -> dict:
    """Launch a 1-line job on each candidate reference; report which reference the runner pulled (COMPLETED).
    If none pulls, the caller STOPS and reports to the owner (S5.12)."""
    api = api or _api()
    results: dict[str, str] = {}
    for ref in _candidates():
        info = api.run_job(
            image=ref,
            command=["python", "-c", "print(1)"],
            flavor="cpu-upgrade",
            timeout=600,
            labels={"campaign": "image-probe", "run_key": "image-probe-registry"},
        )
        results[ref] = _wait(api, info.id, timeout_s=900)
    pulled = [r for r, s in results.items() if s == "COMPLETED"]
    print(json.dumps({"results": results, "pulled": pulled}, indent=2))
    return {"results": results, "pulled": pulled}


def _startup_from_phases(rec: dict) -> float | None:
    ph = rec.get("phases") or {}
    if "first_unit" not in ph or "script_start" not in ph:
        return None
    s = float(ph["first_unit"]) - float(ph["script_start"])
    if "after_dataset_fetch" in ph and "after_fetch" in ph:
        s -= float(ph["after_dataset_fetch"]) - float(ph["after_fetch"])
    return s


def probe_startup(api: Any = None) -> dict:
    """Launch the five S5.12 startup jobs (win_probability ``refs,extract --profile 1`` on the prebuilt image),
    read each job's phase record and write the median startup into ``image.lock.json``. Refuses a dirty tree
    except for the lock itself."""
    from scripts._provenance import git_provenance
    from scripts.hf_jobs import campaign as campaign_mod
    from scripts.hf_jobs import launch as launcher
    from scripts.hf_jobs.ledger import percentile
    from scripts.hf_jobs.store import BucketStore

    if not image.load_lock().get("digest"):
        raise SystemExit("probe-startup needs a built image: record registry_ref + digest in image.lock.json first")
    prov = git_provenance()
    dirty = {f for f in (prov.get("dirty_files") or []) if f != "scripts/hf_jobs/image.lock.json"}
    if dirty:
        raise SystemExit(f"probe-startup refuses a dirty tree (except image.lock.json): {sorted(dirty)}")
    api = api or _api()

    from huggingface_hub import get_token

    token = get_token()
    if token is None:
        raise SystemExit("no HF token: run `hf auth login` first")
    camp = campaign_mod.load_campaign("scripts/hf_jobs/campaigns/cycle1.toml")
    commit = campaign_mod._head()
    inst = next(i for i in campaign_mod.instances(camp, commit) if i.job == "win_probability")
    tag = image.load_lock()["tag"]
    samples: list[float] = []
    job_ids: list[str] = []
    for i in range(_STARTUP_JOBS):
        rk = f"image-probe-{tag}-{i}-win_probability-public"
        spec = launcher.JobSpec(
            campaign="image-probe",
            run_key=rk,
            job=inst.job,
            variant=inst.variant,
            stages=("refs", "extract"),
            shard=(0, 1),
            module=inst.module,
            args=inst.args,
            commit=commit,
            bucket=camp.bucket,
            workers=1,
            inner_workers=1,
            cap_bytes=None,
            peak_rss_bytes=None,
            timeout_s=1800,
            attempts=1,
            datasets=inst.datasets,
            profile=1,
        )
        job_id = launcher.launch(spec, api=api, token=token)
        job_ids.append(job_id)
        _wait(api, job_id)
        store = BucketStore(camp.bucket, rk, api=api)
        for name in sorted(store.names("_ledger/_jobs")):
            raw = store.read_bytes(f"_ledger/_jobs/{name}")
            if raw:
                s = _startup_from_phases(json.loads(raw))
                if s is not None:
                    samples.append(s)
    startup_s = percentile(samples, 0.5) if samples else None
    lock = image.load_lock()
    lock.update({"startup_s": startup_s, "startup_samples": sorted(samples), "startup_jobs": job_ids})
    image.LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"startup_s": startup_s, "n_samples": len(samples), "job_ids": job_ids}, indent=2))
    return {"startup_s": startup_s, "samples": samples, "job_ids": job_ids}


def run(cmd: str, argv: list[str]) -> int:
    if cmd == "probe-registry":
        probe_registry()
        return 0
    if cmd == "probe-startup":
        probe_startup()
        return 0
    return 2
