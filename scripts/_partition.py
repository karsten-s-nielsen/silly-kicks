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

    Integer fields SUM and dict fields merge as counters. ``partition``, ``run_commit``,
    ``run_tree_dirty`` and ``generation`` are handled BY NAME. **Anything else is NOT aggregated**
    -- a bare string matches neither branch, and a stray bool is skipped because ``bool`` is an
    ``int`` subclass and summing flags is meaningless. Such keys are reported in ``dropped_fields``
    rather than vanishing: a field that must reach the corpus artifact needs a named case HERE, not
    merely a place in the per-worker manifest. That rule has caught this cycle twice (``generation``
    and ``run_tree_state``), which is why the report exists.

    The per-worker files are the only possible source for counters describing work that produced NO
    output row (drop reasons, exclusions), which is why aggregation cannot simply re-read the shard
    table.

    ``run_commit`` is checked for CONSISTENCY rather than summed: workers are separate processes and
    nothing stops one being launched from a different checkout, which would make the corpus artifact
    a blend of two code versions while looking like a single run. ``run_tree_dirty`` is OR-ed -- one
    dirty worker makes the whole corpus dirty.

    **Only manifests that CONTRIBUTED data vote on commit consistency.** A pass that built nothing
    -- every shard already present, so every match skipped -- records its own commit but produced no
    row, and letting it vote makes the flag describe the ANALYSIS's lineage instead of the DATA's.
    MEASURED: the §3.3 entanglement artifact reported ``commit_consistent: false`` off eight worker
    manifests unanimously at ``6b242cf`` plus one ``n_matches: 0`` analysis manifest at ``d1fc18d``.
    The corpus was single-commit; the flag said otherwise. A guard that cries wolf is worse than no
    guard, because it teaches readers to skim past the one field built to be un-skippable. The
    analysis commit is not lost -- the driver records it separately as the artifact's top-level
    ``run_commit``. ``commits_seen`` reports every commit encountered including non-contributors, so
    an all-zero-contribution aggregate (a full resume) is visibly vacuous rather than quietly
    ``true``.
    """
    totals: dict[str, int] = {k: 0 for k in defaults}
    counters: dict[str, dict[str, int]] = {}
    partitions: list[str] = []
    commits: set[str] = set()  # contributors only -- these decide `commit_consistent`
    commits_seen: set[str] = set()  # every manifest, contributor or not
    generations: set[str] = set()
    dropped: set[str] = set()  # keys that reached no accumulating branch -- reported, not silent
    dirty = False

    for f in sorted(pathlib.Path(dest).glob("manifest_*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        partitions.append(str(m.get("partition", f.stem)))
        # A manifest loses its vote ONLY by positively declaring that it built nothing. Computed
        # before the field loop so key ordering cannot change the verdict.
        #
        # Fail-SAFE in two directions, both learned from a failing test rather than reasoned:
        #  * a counter-only manifest (e.g. `drop_reasons` with no `n_matches`) DID do real work on
        #    real data -- keying the rule to one field name would silently drop its vote;
        #  * a manifest carrying NO countable field at all cannot prove it built nothing, so it
        #    KEEPS its vote. Only "declared zero" demotes. Anything else and narrowing the vote
        #    would quietly disarm the guard on manifests that simply record less.
        # `generation` joins the meta list so a future widening of `countable` cannot let a
        # staleness token vote on whether this manifest contributed. A no-op today: a `str`
        # already fails both isinstance checks below.
        _meta = ("run_commit", "run_tree_dirty", "partition", "generation")
        countable = [
            v
            for k, v in m.items()
            if k not in _meta and (isinstance(v, dict) or (isinstance(v, int) and not isinstance(v, bool)))
        ]
        contributed = not countable or any((v > 0 if isinstance(v, int) else bool(v)) for v in countable)
        for k, v in m.items():
            if k == "partition":
                continue
            if k == "run_commit":
                commits_seen.add(str(v))
                if contributed:
                    commits.add(str(v))
            elif k == "run_tree_dirty":
                dirty = dirty or bool(v)
            elif k == "generation":
                generations.add(str(v))
            elif isinstance(v, bool):
                # bool is an int subclass -- summing flags would be meaningless. Reported anyway:
                # it is discarded, and a discarded field the reader cannot see is the trap above.
                dropped.add(k)
            elif isinstance(v, int):
                totals[k] = totals.get(k, 0) + v
            elif isinstance(v, dict):
                c = counters.setdefault(k, {})
                for kk, vv in v.items():
                    c[kk] = c.get(kk, 0) + int(vv)
            else:
                # Not meta, not an int to sum, not a dict to merge. Dropping is CORRECT -- a named
                # case carries per-field semantics (`run_commit` is contributor-gated,
                # `run_tree_dirty` is OR-ed, `generation` is a set plus a consistency flag) and one
                # generic collector would give all of them one wrong semantic. Reporting it is what
                # was missing.
                dropped.add(k)

    return {
        **totals,
        **counters,
        "n_partitions": len(partitions),
        "partitions": sorted(partitions),
        "run_commit": (next(iter(commits)) if len(commits) == 1 else sorted(commits)),
        "commit_consistent": len(commits) <= 1,
        # Every commit in the directory, contributors or not. When this is WIDER than the
        # contributing set, a pass ran at a commit that built nothing -- benign, but visible rather
        # than silently absorbed, and the only way an all-resume aggregate's vacuous `true` is
        # distinguishable from a genuinely single-commit one.
        "commits_seen": sorted(commits_seen),
        # The staleness token each worker ran under. The combined table beside the shard root is
        # whichever generation finished LAST -- `write_table_atomically` makes that atomic, not
        # attributable. Surfacing the set buys DETECTION of a mixed-generation corpus, which is
        # what a reader needs before trusting the table. Absent reads as consistent.
        "generations_seen": sorted(generations),
        "generation_consistent": len(generations) <= 1,
        # Manifest keys that reached no accumulating branch. Empty is the healthy case; a name
        # here means a driver writes a field that never reaches the corpus artifact.
        "dropped_fields": sorted(dropped),
        "run_tree_dirty": dirty,
    }
