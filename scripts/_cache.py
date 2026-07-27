"""Feature-cache validity (spec 3.2).

The trainers gated on `(cache / "features.parquet").exists()`. That predicate cannot see a
schema change -- and the cache's public/owner arm split is now recomputed live every run from a
per-row visibility lookup keyed on `match_ids`. An absent or mismatched cache_meta.json is a
MISS, so a pre-Task-11 cache (written before the schema bump, with no cache_meta.json) can never
be silently reused -- exactly the DGX-populated caches that predate the visibility taxonomy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_SCHEMA_VERSION = 2  # 1 -> 2: requires cache_meta.json + match_ids.npy (pre-Task-11 caches MISS)


def corpus_fingerprint(rows: list[tuple[str, str, str]]) -> str:
    """Stable hash of the (provider, match_id, visibility) triples the cache was built from.

    LIVE as of ADR-050: both the xS and xCross trainers build this from the corpus they request and
    pass it to :func:`write_cache_meta` / :func:`cache_is_valid`, so a cache built from a different
    corpus under the same ``--output-dir`` MISSES. The previous constant-token gate could only
    invalidate a pre-schema cache; "use a fresh ``--output-dir`` per corpus" was a discipline it
    could not enforce.

    Order-insensitive by construction (the rows are sorted before hashing): the same corpus listed
    in a different manifest order is the same corpus.
    """
    payload = json.dumps(sorted(rows), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_cache_meta(cache_dir: Path, *, fingerprint: str) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cache_meta.json").write_text(
        json.dumps({"schema_version": CACHE_SCHEMA_VERSION, "corpus_fingerprint": fingerprint}, indent=2)
    )


def cache_is_valid(cache_dir: Path, *, fingerprint: str) -> bool:
    cache_dir = Path(cache_dir)
    # The payload must exist AND the metadata must match. Keeping the features.parquet check means
    # a half-written cache (meta present, payload missing -- an interrupted extraction) is a MISS,
    # not a crash on load.
    if not (cache_dir / "features.parquet").exists():
        return False
    meta_path = cache_dir / "cache_meta.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    return meta.get("schema_version") == CACHE_SCHEMA_VERSION and meta.get("corpus_fingerprint") == fingerprint
