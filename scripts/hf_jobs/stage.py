"""Push an owner-tier corpus to a private HF dataset once, so the HF Job consumes a STAGED corpus with
zero pining token/network. ZERO Databricks.

Every staged shape is NEUTRAL parquet the in-job trainer reads through a plain data path (ADR-123
Amendment B -- NO pining token, manifest, or ``_loader_pining`` reaches HF; how the parquet was built,
on DGX or anywhere, is off-HF and the operator's concern):
  * flat TC3 / flat SB360 (``shards/<gen>/*.parquet`` + ``_actions/`` + ``home_teams.json``) -- serves
    ghost_gk / ghost_outfield, the xshot/xcross in-job repackage, receiver's ``--data-dir``, and every
    owner head-to-head eval (build_eval._owner_shards). Materialised by the existing TC3 driver; staged
    here as-is.

Materialization itself is the existing driver, run on DGX/local where the data lives; this module stages
the OUTPUT and records a content manifest so a repeat run is a no-op.

Staging ENDS with ``python -m scripts.hf_jobs datasets record <name>`` (ADR-128): it measures the staged
dataset and writes its pinned revision, unit list and expected corpus identity into
``scripts/hf_jobs/datasets.lock.json``. A campaign reads only recorded datasets, and ``adopt`` verifies
every artifact against that record (INV-9).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time


def corpus_manifest(corpus_dir: pathlib.Path) -> dict:
    corpus_dir = pathlib.Path(corpus_dir)
    files = sorted(p for p in corpus_dir.rglob("*") if p.is_file())
    h = hashlib.sha256()
    nbytes = 0
    for p in files:
        h.update(p.relative_to(corpus_dir).as_posix().encode())
        data = p.read_bytes()
        h.update(data)
        nbytes += len(data)
    return {"sha256": h.hexdigest(), "n_files": len(files), "bytes": nbytes}


def stage_corpus(corpus_dir, dataset_repo, *, uploader, already: dict | None = None) -> dict:
    corpus_dir = pathlib.Path(corpus_dir)
    manifest = corpus_manifest(corpus_dir)
    if already is not None and already.get("sha256") == manifest["sha256"]:
        return manifest  # idempotent: nothing changed, skip the push
    uploader(str(corpus_dir), dataset_repo)
    return manifest


#: Fetch attempts per match in `stage_open_data` (a large raw file can arrive truncated).
_ATTEMPTS = 3


def stage_open_data(revision: str, out_dir, *, workers: int = 32) -> dict:
    """Fill ``out_dir`` with the open-data snapshot of ``revision`` (HFU spec 14.7): the consumer cache, built by
    the SAME fetch functions with the pin set, so a cache hit returns what a fetch returns. Every competition's
    manifest, every match's events and lineups, and the 360 frames where the manifest marks them available.
    The matches are fetched over ``workers`` threads (the fetches are latency-bound; each file is written
    atomically by the cache, so the snapshot does not depend on the order), and a rerun resumes: a file already
    present is a cache hit. Each file is fetched up to ``_ATTEMPTS`` times with backoff. A file that never parses
    is an upstream defect (StatsBomb's ``three-sixty/3845506.json`` at the pin holds NUL bytes): it is recorded in
    ``unreadable`` and staging goes on, since a direct fetch of it fails too. A network failure raises, and no
    snapshot record is written. Writes ``_snapshot.json`` ``{source_revision, n_files, bytes, unreadable}`` (over
    the cache files) and returns it. Off-HF only: the upload and the ``datasets record`` that follow are owner-gated.
    """
    import concurrent.futures
    import threading

    import requests

    import scripts._sb_open_data as sbod

    out = pathlib.Path(out_dir)
    sbod.pin_open_data_revision(revision)
    sbod.assert_statsbomb_open_data_mode()
    comps = sorted({(int(c["competition_id"]), int(c["season_id"])) for c in sbod._open_competitions(out)})
    matches = [
        (str(match_id), match.get("match_status_360") == "available")
        for competition_id, season_id in comps
        for match_id, match in sbod._open_matches(competition_id, season_id, out).items()
    ]
    unreadable: dict[str, str] = {}
    guard = threading.Lock()

    def _file(rel: str, fetch) -> None:
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                fetch()
                return
            except requests.exceptions.JSONDecodeError as exc:  # the payload does not parse
                if attempt == _ATTEMPTS:
                    with guard:
                        unreadable[rel] = str(exc)
                    return
            except requests.exceptions.RequestException:
                if attempt == _ATTEMPTS:
                    raise
            time.sleep(2.0**attempt)

    def _match(item: tuple[str, bool]) -> None:
        match_id, has_360 = item
        _file(f"statsbomb_open/{match_id}.json", lambda: sbod._open_events(match_id, out))
        _file(f"statsbomb_open/lineups/{match_id}.json", lambda: sbod.open_lineups(match_id, out))
        if has_360:
            _file(f"statsbomb_open/{match_id}.360.json", lambda: sbod._open_frames(match_id, out))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for _ in pool.map(_match, matches):  # re-raises the first network failure
            pass
    files = sorted(f for f in out.rglob("*") if f.is_file() and f.name != "_snapshot.json")
    meta = {
        "source_revision": revision,
        "n_files": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "unreadable": dict(sorted(unreadable.items())),
    }
    (out / "_snapshot.json").write_text(json.dumps(meta, sort_keys=True) + "\n", encoding="utf-8")
    return meta
