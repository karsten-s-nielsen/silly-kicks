"""Per-attempt job identity and the job phase record (ADR-128 Cycle 1b, S5.3, S7.2).

The job script makes an ATTEMPT ID at start (`SILLY_KICKS_ATTEMPT_ID`, `%Y%m%dT%H%M%SZ_<4 hex>`): an HF retry
starts a new script, so it gets a new id. Every per-job file name carries ``<in-job label>.<attempt id>``, so no
per-job file has two writers. The bash job script appends phase timestamps to a local file
(`SILLY_KICKS_PHASE_FILE`); the first unit of the process publishes them, with its own start, to the store."""

from __future__ import annotations

import json
import os
import sys
import threading
import time

ATTEMPT_ENV = "SILLY_KICKS_ATTEMPT_ID"
PHASE_FILE_ENV = "SILLY_KICKS_PHASE_FILE"
ACTION_KIND_ENV = "SILLY_KICKS_ACTION_KIND"
LABEL_ENV = "SILLY_KICKS_JOB_LABEL"
_LOCK = threading.Lock()
_PUBLISHED = False


def job_rel(label: str, attempt: str) -> str:
    return f"_ledger/_jobs/{label}.{attempt}.json"


def cost_rel(pipeline: str, stage: str, token: str, label: str, attempt: str) -> str:
    return f"_ledger/{pipeline}/{stage}/{token}/_costs/{label}.{attempt}.json"


def reduce_rel(pipeline: str, stage: str, token: str, label: str, attempt: str) -> str:
    return f"_ledger/{pipeline}/{stage}/{token}/_reduce/{label}.{attempt}.json"


def identity() -> tuple[str, str] | None:
    """(in-job label, attempt id), or None outside a job (a laptop, CI)."""
    label, attempt = os.environ.get(LABEL_ENV), os.environ.get(ATTEMPT_ENV)
    return (label, attempt) if label and attempt else None


def mark(name: str, t: float | None = None) -> None:
    """Append ``{"phase": name, "t": epoch_s}`` to the local phase file (the bash script calls this). ``t``
    lets the script pass the TRUE phase time: ``script_start`` is captured in bash before the venv exists,
    then marked after it, so its timestamp is the real start, not the mark time (S7.2 startup)."""
    path = os.environ.get(PHASE_FILE_ENV)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"phase": name, "t": time.time() if t is None else float(t)}) + "\n")


def publish_first_unit(store) -> None:
    """At the first unit start of the ATTEMPT: write the job phase record (the local phases plus ``first_unit``)
    to ``job_rel``. ONE writer per attempt: the first process to create the local marker
    ``<phase file>.first`` (``O_CREAT | O_EXCL`` on the job's local disk, which a retry does not share) writes;
    every other worker of the attempt skips (r1 B CCTP-PLAN-12)."""
    global _PUBLISHED
    ident = identity()
    path = os.environ.get(PHASE_FILE_ENV)
    with _LOCK:
        if _PUBLISHED or ident is None or not path:
            return
        _PUBLISHED = True
    try:
        os.close(os.open(path + ".first", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return  # another worker of this attempt is the writer
    rel = job_rel(*ident)
    phases = {}
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                phases.setdefault(row["phase"], row["t"])
    phases["first_unit"] = time.time()
    store.write_bytes(
        rel,
        (
            json.dumps({"label": ident[0], "attempt": ident[1], "phases": phases, "env": _env()}, sort_keys=True) + "\n"
        ).encode(),
    )


def _env() -> dict:
    """The job's environment identity for the attestation (S5.10): ``env_provenance()`` in-job, ``{}`` if it
    cannot be read (a laptop with no repo). ADR-129 verify reads it from the main and replica phase records."""
    try:
        from scripts._provenance import env_provenance

        return env_provenance()
    except Exception:  # a missing repo/lock must never fail the first unit
        return {}


def publish_reduce(store) -> None:
    """At the start of a SEPARATE reduce job (``ACTION_KIND == reduce``): write the job phase record with a
    ``reduce_start`` phase, so the job -- which legitimately runs no unit -- is not misread as an infra
    failure on an ERROR (CCTP-IMPL-11). A no-op in a unit job (whose record carries ``first_unit``)."""
    if os.environ.get(ACTION_KIND_ENV) != "reduce":
        return
    ident = identity()
    path = os.environ.get(PHASE_FILE_ENV)
    if ident is None or not path:
        return
    rows: dict[str, float] = {}
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                rows.setdefault(row["phase"], row["t"])
    rows.setdefault("reduce_start", time.time())
    store.write_bytes(
        job_rel(*ident),
        (
            json.dumps({"label": ident[0], "attempt": ident[1], "phases": rows, "env": _env()}, sort_keys=True) + "\n"
        ).encode(),
    )


if __name__ == "__main__":  # `python -m scripts.hf_jobs.phases mark <name> [<epoch>]`
    if len(sys.argv) in (3, 4) and sys.argv[1] == "mark":
        mark(sys.argv[2], float(sys.argv[3]) if len(sys.argv) == 4 else None)
        raise SystemExit(0)
    raise SystemExit("usage: python -m scripts.hf_jobs.phases mark <phase> [<epoch>]")
