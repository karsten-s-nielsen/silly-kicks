"""The run ledger (spec 5.5). The full per-unit record lives in the store (`<key>.ledger.json`, written by
`_stages.run_unit`); job records come from the HF API (`JobInfo` via labels); a compact per-campaign
summary is committed as `docs/research/hf_campaigns/<campaign>.json` with top-level `run_commit` and
`run_tree_dirty` (it passes `test_artifact_provenance_output`). The planner learns from those summaries."""

from __future__ import annotations

import json
import math
import pathlib
from collections.abc import Iterable, Mapping, Sequence

from scripts.hf_jobs.planner import UnitCost


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (deterministic, no interpolation)."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    s = sorted(values)
    return float(s[max(0, min(len(s) - 1, math.ceil(q * len(s)) - 1))])


def unit_records(store, generation: str) -> list[dict]:
    out = []
    for name in sorted(store.names(generation)):
        if name.endswith(".ledger.json"):
            raw = store.read_bytes(f"{generation}/{name}")
            if raw:
                out.append(json.loads(raw))
    return out


def unit_cost(records: Iterable[Mapping]) -> UnitCost | None:
    done = [r for r in records if r.get("outcome") == "done" and r.get("wall_s") is not None]
    if not done:
        return None
    walls = [float(r["wall_s"]) for r in done]
    peaks = [int(r["peak_rss_bytes"]) for r in done if r.get("peak_rss_bytes")]
    return UnitCost(
        percentile(walls, 0.5), percentile(walls, 0.95), percentile(walls, 0.99), max(peaks) if peaks else 0, len(done)
    )


def history_key(token_inputs: Mapping) -> str:
    """Decision P14: costs are keyed by WHAT is computed, not by which commit computed it. Single-sourced
    in `_stages._history_key` (the plan JSON carries it per stage)."""
    from scripts._stages import _history_key

    return _history_key(token_inputs)


def _iso(dt) -> str | None:
    return None if dt is None else dt.isoformat()


def job_record(info) -> dict:
    started, finished = getattr(info, "started_at", None), getattr(info, "finished_at", None)
    wall = (finished - started).total_seconds() if (started and finished) else None
    status = getattr(info, "status", None)
    return {
        "id": info.id,
        "labels": dict(info.labels or {}),
        "stage": getattr(status, "stage", None),
        "message": getattr(status, "message", None),
        "created_at": _iso(getattr(info, "created_at", None)),
        "started_at": _iso(started),
        "finished_at": _iso(finished),
        "wall_s": wall,
    }


def _epoch_s(iso: str | None) -> float | None:
    if iso is None:
        return None
    import datetime

    dt = datetime.datetime.fromisoformat(iso)
    return (dt if dt.tzinfo else dt.replace(tzinfo=datetime.timezone.utc)).timestamp()


def job_startups(jobs: Iterable[Mapping], records: Iterable[Mapping]) -> list[float]:
    """Per job (a `job_record`): its first unit's start minus `JobInfo.started_at`, in seconds (plan Task 21
    step 8). A unit record belongs to the job whose label it carries and inside whose window
    (started..finished) it started, so a retry of the same shard is measured as its own job, and a record
    that survives from an earlier in-job attempt (it starts before `started_at`) never counts. Pass EVERY
    stage's records of the run: a grouped job's first unit may belong to any of its stages. A job with no
    unit record in its window has no startup."""
    from scripts.hf_jobs.launch import label_of

    recs = [r for r in records if r.get("job_label") and r.get("started_ns") is not None]
    out = []
    for j in jobs:
        start = _epoch_s(j.get("started_at"))
        if start is None:
            continue
        hi = _epoch_s(j.get("finished_at")) or math.inf
        label = label_of(j["labels"])
        firsts = [
            r["started_ns"] / 1e9 for r in recs if r["job_label"] == label and start <= r["started_ns"] / 1e9 <= hi
        ]
        if firsts:
            out.append(min(firsts) - start)
    return out


def stage_workers(records: Iterable[Mapping]) -> tuple[int | None, int | None]:
    """(outer workers of the widest job, widest inner pool) measured from a stage's unit records."""
    by_job: dict[str, set] = {}
    inner: list[int] = []
    for r in records:
        by_job.setdefault(str(r.get("job_label")), set()).add(r.get("worker"))
        inner.append(int(r.get("inner_workers") or 1))
    if not by_job:
        return None, None
    return max(len(w) for w in by_job.values()), max(inner)


def median_startup_s(paths: Iterable[pathlib.Path]) -> float | None:
    """The median measured job startup over every stage of the committed campaign summaries (the planner's
    `Constraints.startup_s`); None before any campaign measured one. A negative or non-finite value is a
    corrupt (hand-edited) summary and fails loud: it would silently under-predict or NaN every wall."""
    values = []
    for p in sorted(paths):
        for st in json.loads(pathlib.Path(p).read_text(encoding="utf-8")).get("stages", []):
            if st.get("startup_s") is None:
                continue
            v = float(st["startup_s"])
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"{p}: stage {st.get('stage')!r} has startup_s={v}; a measured startup is >= 0")
            values.append(v)
    return percentile(values, 0.5) if values else None


def summarize(
    *, campaign: str, campaign_sha256: str, prov: Mapping, stages: list[dict], jobs: list[dict], artifacts: list[dict]
) -> dict:
    return {
        "run_commit": prov["commit"],
        "run_tree_dirty": bool(prov.get("dirty", True)),
        "campaign": campaign,
        "campaign_sha256": campaign_sha256,
        "env": prov.get("env"),
        "stages": stages,
        "jobs": jobs,
        "artifacts": artifacts,
    }


def write_summary(path, summary: Mapping) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_history(paths: Iterable[pathlib.Path]) -> dict[str, dict]:
    """``{history_key: stage summary}`` from committed campaign summaries; later files win."""
    hist: dict[str, dict] = {}
    for p in sorted(paths):
        for st in json.loads(pathlib.Path(p).read_text(encoding="utf-8")).get("stages", []):
            if st.get("history_key"):
                hist[st["history_key"]] = st
    return hist
