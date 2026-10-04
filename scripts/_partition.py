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
partition can never name a match a real run would not fetch, and never edits it. That is a property
of THIS module, not a repo-wide fence -- the TF-19 partition cycle that wrote it was declaring its
own scope, and ADR-052 subsequently changed `_loader_databricks.load_matches` (it accepted neither
`tracking_limit` nor `max_per_provider`, so `calibrate_tracking_defaults --source databricks` died
on a `TypeError` before reading a row).
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
    partitioning layer that must read "no ids for me" as "nothing for me". Fixing it here rather
    than in the loader is therefore the correct SEAM, not a scope restriction -- the TF-19
    partition cycle's "may not modify" phrasing described that cycle, and ADR-052 has since
    changed the loader for an unrelated defect.
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

    **Overlapping partitions are REFUSED.** A worker records the shard keys its pass covered in
    ``partition_keys``; a resumed pass REPLAYS each skipped key's counters, so two manifests covering
    one key count it twice. MEASURED (combined-cycle Phase B): two 32-match wave workers plus a later
    64-match full-population pass over one ``--out`` aggregated to ``n_matches: 128`` and doubled frame
    counts. The authoritative combine after a partitioned wave is the producer's ``--reduce-only``, which
    writes no manifest. ``partition_keys`` is an integrity field only: it never reaches the output (it is
    a match-id list, and the aggregate is a cited artifact). A manifest without it is not checked.
    """
    totals: dict[str, int] = {k: 0 for k in defaults}
    counters: dict[str, dict[str, int]] = {}
    key_owner: dict[str, str] = {}  # partition key -> the manifest file that first covered it
    overlaps: dict[tuple[str, str], int] = {}  # (earlier file, later file) -> n shared keys
    partitions: list[str] = []
    commits: set[str] = set()  # contributors only -- these decide `commit_consistent`
    commits_seen: set[str] = set()  # every manifest, contributor or not
    generations: set[str] = set()
    dropped: set[str] = set()  # keys that reached no accumulating branch -- reported, not silent
    dirty = False

    for f in sorted(pathlib.Path(dest).glob("manifest_*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        partitions.append(str(m.get("partition", f.stem)))
        for pk in m.get("partition_keys") or ():
            first = key_owner.setdefault(str(pk), f.name)
            if first != f.name:
                overlaps[(first, f.name)] = overlaps.get((first, f.name), 0) + 1
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
        _meta = ("run_commit", "run_tree_dirty", "partition", "generation", "partition_keys")
        # `n_excluded` is corpus-scoped and REPLAYED on resume (ADR-052 D13): a fully resumed pass
        # reports n_excluded>0 while genuinely building nothing. It is summed below like any int, but
        # it must NOT count as contribution, or a resumed worker regains its commit vote and re-arms
        # the false alarm `test_a_ZERO_CONTRIBUTION_manifest_does_not_vote...` prevents (interp 9).
        _non_contributing = (*_meta, "n_excluded")
        countable = [
            v
            for k, v in m.items()
            if k not in _non_contributing and (isinstance(v, dict) or (isinstance(v, int) and not isinstance(v, bool)))
        ]
        contributed = not countable or any((v > 0 if isinstance(v, int) else bool(v)) for v in countable)
        for k, v in m.items():
            if k in ("partition", "partition_keys"):
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
                # A dict is a COUNTER dict only if every value is numeric. One that is not -- a
                # declared input contract, say -- must be dropped and reported, never raised on:
                # this aggregation runs AFTER the corpus pass, so a raise here destroys the
                # combine of work that already cost hours. Measured: an `input_contract` dict
                # reached this line and `int("build_gkdv_arm_values")` killed a 64-match pass at
                # the final step.
                if all(isinstance(vv, (int, float)) and not isinstance(vv, bool) for vv in v.values()):
                    c = counters.setdefault(k, {})
                    for kk, vv in v.items():
                        c[kk] = c.get(kk, 0) + vv
                else:
                    dropped.add(k)
            else:
                # Not meta, not an int to sum, not a dict to merge. Dropping is CORRECT -- a named
                # case carries per-field semantics (`run_commit` is contributor-gated,
                # `run_tree_dirty` is OR-ed, `generation` is a set plus a consistency flag) and one
                # generic collector would give all of them one wrong semantic. Reporting it is what
                # was missing.
                dropped.add(k)

    if overlaps:
        detail = "; ".join(f"{a} and {b} share {n} key(s)" for (a, b), n in sorted(overlaps.items()))
        raise ValueError(
            f"partition manifests in {dest} overlap ({detail}): every shared key would be counted twice. "
            "Combine a partitioned run with the producer's --reduce-only (it writes no manifest), not with "
            "another counting pass over the same --out."
        )

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


def require_population_shards(shard_root, token_inputs, refs) -> None:
    """Refuse a reduce whose population is not finished: every ref needs a shard or an exclusion marker.

    Checked BEFORE the reduce's own pass, which would otherwise COMPUTE the missing matches in-process
    and present a reduce as a quiet extra worker (the generation is the one the workers wrote).
    """
    from scripts._driver import exclusion_path, generation_dir, shard_path

    gen = generation_dir(shard_root, token_inputs=token_inputs)
    missing = [r.key for r in refs if not shard_path(gen, r.key).is_file() and not exclusion_path(gen, r.key).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(refs)} population matches have no shard; finish the workers first")


def population_table(res):
    """The population's shards (``res.shard_keys``, pass order) concatenated; empty when none hold rows.

    Exactly the population, unlike ``_driver.reconcile``'s whole-generation read: a reduce describes the
    corpus it was asked for, and a shared ``--out`` may hold other runs' matches.
    """
    import pandas as pd

    from scripts._driver import shard_path

    frames = [pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.shard_keys]
    non_empty = [f for f in frames if len(f)]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def worker_lineage(dest, *, prov: dict, generation: str) -> dict:
    """The provenance of the shards a reduce combines, read from the workers' manifests in ``dest``.

    Fail-closed: no worker manifest means the shards' lineage is unknown; a manifest from another commit
    or another generation means the combine would blend runs; overlapping partitions are refused by
    :func:`aggregate_manifests`. The returned fields go on the reduce's corpus artifact.
    """
    agg = aggregate_manifests(dest)
    if not agg["partitions"]:
        raise SystemExit(f"no worker manifest in {dest}: the shards' lineage is unknown, so the reduce refuses")
    foreign = sorted(set(agg["commits_seen"]) - {prov["commit"]})
    if foreign:
        raise SystemExit(f"worker manifest(s) from another commit {foreign}; this reduce runs at {prov['commit']}")
    other_gen = sorted(set(agg["generations_seen"]) - {generation})
    if other_gen:
        raise SystemExit(f"worker manifest(s) from another generation {other_gen}; this reduce reads {generation}")
    return {
        "run_commit": prov["commit"],
        "run_tree_dirty": bool(prov["dirty"]) or bool(agg["run_tree_dirty"]),
        "run_tree_state": prov.get("tree_state"),
        "commit_consistent": True,  # every worker manifest names this commit (checked above)
        "commits_seen": agg["commits_seen"],
        "generations_seen": agg["generations_seen"],
        "n_partitions": agg["n_partitions"],
        "partitions": agg["partitions"],
    }


def partition_keys(res) -> dict:
    """``{"partition_keys": [...]}`` -- the keys whose results a pass's manifest COUNTS, for its PER-WORKER
    manifest only: shards (fresh or resumed, whose counters are replayed) and exclusions.

    :func:`aggregate_manifests` refuses two manifests sharing a key, because a resumed pass replays its
    skipped keys' counters (so an overlap counts each shared key twice). FAILED keys are left out: they
    carry no counters, and ``_parallel_launch`` can deal a failed key to another worker tag on a later
    pass, which completes it without any double count. Never spread this into a cited artifact: it is a
    match-id list. ``tests/scripts/test_partition.py`` derives every worker-manifest writer that
    aggregates and asserts each one records it.
    """
    return {"partition_keys": sorted(k for k in res.keys if k not in res.failures)}
