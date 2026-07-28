"""Shared plumbing for the PARALLELISED corpus producers.

Both TF-19 corpus passes -- GKDV arm values and Layer 2 spells -- have the same shape: an expensive
per-match computation, sharded to disk so a crash resumes instead of restarting, split N ways across
processes by a pinned id list, and reconciled afterwards into ONE corpus manifest.

Extracted rather than duplicated because the reconciliation is the part that has already been wrong
once: N workers writing a single shared manifest let the last writer win, so the artifact reported
one partition's totals (`n_matches: 8`) while describing a 64-match corpus. That is a false
self-description in a provenance-bearing file -- the same class of defect as a false commit SHA --
and it must not be possible to fix it in one producer and leave it broken in the other.

`scripts/_loader_*` is READ-ONLY from here: this module reads the loader's own listing helper so a
partition can never name a match a real run would not fetch, and never edits it.
"""

from __future__ import annotations

import json
import os
import pathlib


def worker_tag(match_ids_json: str | None) -> str:
    """The partition's name, taken from its id-list filename (``all`` for an unpartitioned run)."""
    return pathlib.Path(match_ids_json).stem if match_ids_json else "all"


def list_match_ids(providers: list[str]) -> dict[str, list[str]]:
    """Every available match id per provider, as the JSON a ``--match-ids-json`` split is built from.

    Consumes the loader's own ``_list_matches`` -- the exact call ``load_matches`` makes internally,
    so the id set cannot drift from what a run would actually fetch.
    """
    from scripts._loader_pining import _base_url, _list_matches, _resolve_token

    tok, base = _resolve_token(None), _base_url()
    return {p: [str(m["id"]) for m in _list_matches(p, tok, base)] for p in providers}


def providers_for_slice(providers: list[str], match_ids: dict | None) -> list[str]:
    """Providers this partition actually owns -- those with a NON-EMPTY id list.

    MEASURED trap in the shared loader (`_wanted_for_provider`)::

        wanted = (match_ids.get(provider) if match_ids else None) or list(manifest_ids)

    An empty list is falsy and an absent key is None, so BOTH fall through to the ENTIRE manifest.
    For a partitioned run that inverts the intent exactly: a worker handed nothing for a provider
    would process ALL of it. With a multi-provider driver sliced on one provider, every worker
    loads the other providers in full -- N-times duplicated work AND N processes writing the SAME
    per-match shard paths concurrently.

    The loader's behaviour is right for its own callers (no slice means "everything"); it is this
    partitioning layer that must read "no ids for me" as "nothing for me". Fixing it here also
    keeps `scripts/_loader_*` untouched, which this cycle may not modify.
    """
    if not match_ids:
        return list(providers)
    return [p for p in providers if match_ids.get(p)]


def write_table_atomically(df, path, *, tag: str) -> None:
    """Write a combined table so CONCURRENT workers cannot tear it.

    Every worker rebuilds the combined table from the SHARED shard directory and writes it to the
    same path, so with N workers running there are N writers on one file. A plain `to_parquet` can
    therefore be read -- or left -- half-written. Each worker instead writes a private temp file and
    `os.replace`s it into position, which is atomic: the destination is always some worker's
    COMPLETE table, and the last finisher (the one that has seen the most shards) wins.
    """
    path = pathlib.Path(path)
    tmp = path.with_name(f"{path.stem}.{tag}.tmp{path.suffix}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def aggregate_manifests(dest, *, defaults: tuple[str, ...] = ()) -> dict:
    """Sum every per-worker ``manifest_*.json`` in ``dest`` into corpus-wide totals.

    Integer fields SUM, dict fields merge as counters, and ``partition`` names are collected. The
    per-worker files are the only possible source for counters describing work that produced NO
    output row (drop reasons, exclusions), which is why aggregation cannot simply re-read the shard
    table.

    ``run_commit`` is checked for CONSISTENCY rather than summed: workers are separate processes and
    nothing stops one being launched from a different checkout, which would make the corpus artifact
    a blend of two code versions while looking like a single run. ``run_tree_dirty`` is OR-ed -- one
    dirty worker makes the whole corpus dirty.
    """
    totals: dict[str, int] = {k: 0 for k in defaults}
    counters: dict[str, dict[str, int]] = {}
    partitions: list[str] = []
    commits: set[str] = set()
    dirty = False

    for f in sorted(pathlib.Path(dest).glob("manifest_*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        partitions.append(str(m.get("partition", f.stem)))
        for k, v in m.items():
            if k == "partition":
                continue
            if k == "run_commit":
                commits.add(str(v))
            elif k == "run_tree_dirty":
                dirty = dirty or bool(v)
            elif isinstance(v, bool):
                continue  # bool is an int subclass -- summing flags would be meaningless
            elif isinstance(v, int):
                totals[k] = totals.get(k, 0) + v
            elif isinstance(v, dict):
                c = counters.setdefault(k, {})
                for kk, vv in v.items():
                    c[kk] = c.get(kk, 0) + int(vv)

    return {
        **totals,
        **counters,
        "n_partitions": len(partitions),
        "partitions": sorted(partitions),
        "run_commit": (next(iter(commits)) if len(commits) == 1 else sorted(commits)),
        "commit_consistent": len(commits) <= 1,
        "run_tree_dirty": dirty,
    }
