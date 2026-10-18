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
"""

from __future__ import annotations

import hashlib
import pathlib


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
