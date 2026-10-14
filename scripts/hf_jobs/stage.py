"""Push an owner-tier corpus to a private HF dataset once, so the HF Job consumes a STAGED corpus with
zero pining token/network. ZERO Databricks.

Two staged shapes (see registry source modes):
  * flat TC3 (``shards/<gen>/*.parquet`` + ``_actions/`` + ``home_teams.json``) -- serves ghost_gk /
    ghost_outfield directly, the xshot/xcross in-job repackage, and every owner head-to-head eval
    (build_eval._owner_shards). Materialised by the existing TC3 driver; staged here as-is.
  * SB360 bundle for receiver -- the per-match pining cache (``cache/<provider>/<match_id>/``) + an
    OFFLINE manifest (``manifest/<provider>/matches.json``, written here on the token-bearing host) +
    ``match_ids.json``. The job sets ``SILLY_KICKS_PINING_MANIFEST_DIR`` so enumeration is offline.

Materialization itself is the existing driver, run on DGX/local where the data lives; this module stages
the OUTPUT and records a content manifest so a repeat run is a no-op.
"""

from __future__ import annotations

import hashlib
import json
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


def write_offline_manifest(providers, manifest_dir, *, lister=None, token=None, base_url=None) -> list[pathlib.Path]:
    """Capture each provider's ``/{provider}/matches`` response to ``<manifest_dir>/<provider>/matches.json``.

    Run on the token-bearing host (DGX) at stage time; the staged file is what the air-gapped HF Job reads
    via ``SILLY_KICKS_PINING_MANIFEST_DIR`` (``_loader_pining._list_matches`` offline branch). ``lister`` is
    injected for tests; by default it is the real ``_loader_pining._list_matches`` (a live API call).
    """
    if lister is None:
        from scripts._loader_pining import _base_url, _list_matches, _resolve_token

        tok, base = _resolve_token(token), base_url
        lister = lambda provider: _list_matches(provider, tok, base or _base_url())  # noqa: E731
    manifest_dir = pathlib.Path(manifest_dir)
    written = []
    for provider in providers:
        matches = lister(provider)
        out = manifest_dir / provider / "matches.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"matches": matches}, indent=2))
        written.append(out)
    return written
