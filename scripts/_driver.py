"""The shared seam for `scripts/` drivers that walk a corpus.

WHY THIS EXISTS. Twenty-one drivers here do expensive per-item corpus work and only three survive a
crash. Fourteen hold every result in memory and write once at the end, so a failure at hour 13 of a
14-hour pass loses all of it -- measured: a power-analysis pass spent 8.7 hours walking 64 matches,
raised in the cheap analysis step that followed, and lost every one of them.

This was never a missing convention. FOUR partial mechanisms already existed, covering seven drivers
between them, and the split is exactly resume-XOR-staleness: `_partition.py` resumes but has no
staleness token, `_cache.py` has a token but is all-or-nothing, and two driver-local copies each
have one half. **None of the four owns the loop**, which is precisely why resume and progress are
the parts that keep being omitted -- there was nowhere for them to live except in each author's
memory.

This module owns the loop. `for_each` is the default shape; the individual primitives are the
escape hatch for a driver whose loops genuinely cannot invert, and such a driver MUST still call
`assert_conservation` AND `_require_injective` (see the spec, section 4.1).

Relationship to its neighbours: `_partition.py` keeps partitioning and manifest aggregation (it
reads `manifest_*.json` only -- shard reconciliation was always driver-local), and `_cache.py`
supplies the fail-closed cache metadata the cohort cache reuses.
"""

from __future__ import annotations

import collections
import dataclasses
import importlib.util
import pathlib
import re
import time
from collections.abc import Mapping
from pathlib import PurePath, PurePosixPath

# NOTE: `ruthless` and `pandas` are imported INSIDE the functions that need them, not here.
# `ruthless-efficiency` ships only in the [calibration] / [test] / [train] extras, so a
# module-level import would make `shard_path`, `progress` and `already_done` -- pure stdlib
# helpers -- unreachable without one. `reconcile` imports pandas the same way.

#: Joins the components of a composite shard key. Rejected in any component -- see `shard_path`.
KEY_SEPARATOR = "__"


def _normalise(value: object) -> object:
    """Make a declared input's digest platform-independent before it reaches `fingerprint`.

    ruthless guarantees the same LOGICAL value digests identically on every platform; CONSTRUCTING
    that value is the caller's responsibility, and `Path(str)` parses per-platform -- a backslash is
    a separator on Windows and an ordinary character on POSIX. This repo spans a Windows dev box, a
    Linux DGX and both OSes in CI, so an un-normalised path input would orphan a generation on the
    other platform with no version having changed.
    """
    if isinstance(value, PurePath):
        return PurePosixPath(value.as_posix())
    if isinstance(value, Mapping):
        return {k: _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_normalise(v) for v in value)
    return value


def _token(token_inputs: Mapping[str, object], token_reason: str | None) -> str:
    """Digest of the DECLARED inputs, delegated to `ruthless.fingerprint`.

    NOT hand-rolled, and that is a correctness decision rather than a convenience one. An earlier
    draft hashed `sorted(f"{type(v).__name__}:{v!r}")`, which was defective in two measured ways: an
    object with no `__repr__` digested its MEMORY ADDRESS (a different token every process, so the
    driver never matched its own generation and silently full-recomputed), and a numpy major changed
    the digest of an unchanged declared value. `ruthless.fingerprint` is structural, type-tagged in
    both the value and the key position, and FAIL-CLOSED -- it raises on a type it does not
    understand rather than inventing an unstable digest for a cache key.

    ruthless states digest stability as a compatibility contract with no carve-out, backed by a
    golden table of 44 pinned literals (0.4.0). That contract is what makes it safe to key resume on
    these bytes; see the spec, section 4.2.
    """
    from ruthless import fingerprint  # public since 0.4.0; extra-only, so imported here not at module scope

    payload = dict(token_inputs)
    if not payload:
        if not token_reason:
            raise ValueError(
                "token_inputs={} means 'this pass has no staleness risk' and REQUIRES token_reason. "
                "A silent omission and a considered decision must not look identical in the source."
            )
        return fingerprint({"empty_reason": token_reason})
    return fingerprint({k: _normalise(v) for k, v in payload.items()})


def generation_dir(shard_root, *, token_inputs, token_reason: str | None = None) -> pathlib.Path:
    """Resolve (and create) the generation directory for this set of declared inputs.

    The shard path is ``shard_root / token / f"{key}.parquet"`` -- the token names a DIRECTORY, not
    a filename suffix. A changed token therefore yields a different directory, so a stale shard can
    be neither read nor half-overwritten: the failure mode is unrepresentable rather than merely
    guarded, and stale generations stay visible on disk (see `prune_stale_generations`).

    The directory form is load-bearing. Reconciliation is driver-local and every existing site is a
    bare ``glob("*.parquet")``, so a filename suffix would make the combined table concatenate the
    old-token and new-token shard for the SAME item -- with different values -- the first time a
    declared input changed, and `n_shards` (a provenance field) would overcount by the same factor.
    That is exactly the defect class `_partition.py` was extracted to prevent.

    **The token cannot be checked for completeness.** An author who declares the wrong inputs gets a
    digest that never changes when it should. Requiring `token_reason` for an empty declaration
    closes silent OMISSION, not MIS-declaration; the spec states this ceiling in section 7.
    """
    gen = pathlib.Path(shard_root) / _token(token_inputs, token_reason)
    gen.mkdir(parents=True, exist_ok=True)
    return gen


#: A generation directory's name is a `ruthless.fingerprint` digest: 16 lowercase hex characters.
#: Verified against the installed wheel rather than assumed -- `fingerprint(...)` returns e.g.
#: '9fc2b2c66eead2b5', and `test_the_generation_name_is_a_16_hex_digest` pins it.
_GENERATION_NAME = re.compile(r"^[0-9a-f]{16}$")


def prune_stale_generations(shard_root, *, keep) -> list[str]:
    """Delete every generation directory under ``shard_root`` EXCEPT ``keep``. Returns what it removed.

    Explicit operator action only -- ``for_each`` never calls this. A generation directory is the
    sole evidence that a pass at a given set of declared inputs ever ran, so pruning automatically
    would make an accidental token change both unrecoverable and unnoticeable; the whole point of the
    directory form is that stale generations stay VISIBLE.

    Only names matching a fingerprint digest are eligible. ``shard_root`` is caller-supplied and may
    hold other things, and a prune helper that recursed over whatever it found would be one typo away
    from deleting a corpus.
    """
    import shutil

    root, keep = pathlib.Path(shard_root), pathlib.Path(keep)
    if not root.is_dir():
        return []
    removed = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name == keep.name or not _GENERATION_NAME.match(child.name):
            continue
        shutil.rmtree(child)
        removed.append(child.name)
    return removed


def join_key(key) -> str:
    """Validate and join a shard key.

    A bare item id is NOT enough in this corpus: providers demonstrably share ``game_id``s (see
    ``test_validate_xshot_causal_shards.py::test_cluster_key_distinguishes_providers_sharing_a_game_id``),
    so ``match_id`` alone would let two providers overwrite each other's shard while the resume
    check reported a hit. Components are validated rather than trusted -- the failure this prevents
    is silent.

    **The separator check applies to a SEQUENCE key only, and that asymmetry is deliberate.** The
    property being protected is injectivity -- two distinct keys must never resolve to one path. A
    sequence can violate it: ``("a__b", "c")`` and ``("a", "b__c")`` both join to ``"a__b__c"``, so
    one shard would serve two items and the resume check would report a hit for work never done. A
    bare string cannot violate it: the string IS the path stem, one string to one path, injective by
    construction, and callers legitimately pass an already-joined ``"provider__match"``. Rejecting
    the separator there would refuse the codebase's own established shard-naming convention.

    Discovered by execution: the two tests covering these cases contradict each other under a single
    unconditional rule, and only one of them can be satisfied without this distinction.
    """
    if isinstance(key, str):
        if not key:
            raise ValueError("empty key: every item needs a distinct, non-empty shard name")
        return key
    parts = [str(k) for k in key]
    for part in parts:
        if not part:
            raise ValueError(f"empty key component in {key!r}: two distinct keys would share a path")
        if KEY_SEPARATOR in part:
            raise ValueError(
                f"key component {part!r} contains the separator {KEY_SEPARATOR!r} and would "
                f"mis-split on read; rename the component or change the key"
            )
    return KEY_SEPARATOR.join(parts)


def shard_path(generation, key) -> pathlib.Path:
    """The parquet path for ``key`` inside a generation directory."""
    return pathlib.Path(generation) / f"{join_key(key)}.parquet"


def write_shard(path, frame, *, tag: str) -> None:
    """Write one item's result, atomically. A ``None`` or empty frame STILL writes a file.

    An absent shard means "not yet run"; a present empty one means "ran, produced nothing".
    Conflating them makes every barren item recompute on every resume, forever -- which is exactly
    the trap the 14-hour driver this module exists for would fall into, since it has barren items
    (``validate_xs_probe:133`` counts them). The distinction is the resume check's entire input.
    """
    import pandas as pd

    from scripts._partition import write_table_atomically

    write_table_atomically(pd.DataFrame() if frame is None else frame, pathlib.Path(path), tag=tag)


def already_done(generation, key) -> bool:
    """True when this item's shard exists in this generation -- empty shards included."""
    return shard_path(generation, key).is_file()


def progress(label: str, i: int, n: int | None, *, elapsed_s: float, note: str = "") -> None:
    """One line per item, FLUSHED.

    An unflushed detached run is indistinguishable from a hung one, which is how a 14-hour pass
    became unobservable. Flushing per item costs nothing at corpus scale.

    ``n`` is optional because `for_each` STREAMS its corpus and a generator has no length; it passes
    ``None`` at all three call sites and the total renders ``?``. A caller that genuinely knows its
    total -- the primitives path, or a materialised list -- passes it. Counting the corpus just to
    render a denominator would reintroduce exactly the materialisation `for_each` avoids.
    """
    tail = f"  {note}" if note else ""
    total = n if n is not None else "?"
    print(f"  [{i}/{total}] {label}  {elapsed_s:6.1f}s{tail}", flush=True)


def assert_conservation(generation, *, keys, failed: int) -> None:
    """Every item THIS PASS attempted either wrote a shard or is counted as failed.

    Counts only the pass's OWN keys, never a directory-wide glob. N workers share one ``--out`` and
    therefore one generation directory -- the token derives from ``token_inputs``, identical across
    workers, while ``tag`` names the manifest file rather than the directory -- so a glob would
    compare one worker's slice against every worker's shards. That fires non-deterministically,
    after the expensive loop, and before the manifest is written, so the partition vanishes from
    ``aggregate_manifests``: the very "64-match artifact reported ``n_matches: 8``" defect
    ``_partition.py`` was extracted to prevent. It is also unrecoverable, because a resume skips
    everything and reaches the same comparison.

    Race-free because ``providers_for_slice`` guarantees disjoint slices, so only this worker writes
    its own keys.

    WHAT THIS DOES NOT PROVE. It does not prove the driver has no OTHER loop. A driver that calls
    ``for_each`` over something trivial and separately accumulates over the real corpus writes no
    shards for that second loop and lists none of its items here, so this passes. Catching that
    needs a fan-in check at reconcile time -- the union of all manifests' key sets against the
    directory contents -- which is a different property and a recorded follow-up. What this does
    catch: a completed item that silently skipped its write, off-by-one counting, and
    stale-generation contamination.

    It is also NOT sufficient on its own for the escape-hatch path: a non-injective key makes two
    items share one shard, so `present` counts that file once per duplicate key and this relation
    BALANCES on a run that dropped data. `_require_injective` is the other half; the adoption gate
    requires both.

    Getting it exactly right before it ships matters more than shipping it early: an invariant that
    fires on healthy runs is weakened or deleted by the first person it inconveniences.
    """
    # Materialise FIRST: the counting pass and the length would otherwise consume `keys` twice, so
    # a generator -- the natural thing to write at the escape-hatch call site -- would be exhausted
    # by the count, report a length of 0, and raise on a perfectly healthy pass. After the
    # expensive loop, which is precisely the failure this docstring's last line warns about.
    keys = list(keys)
    gen = pathlib.Path(generation)
    present = sum(1 for k in keys if (gen / f"{k}.parquet").is_file())
    expected = len(keys) - failed
    if present != expected:
        raise AssertionError(
            f"conservation violated in {gen}: {present} of this pass's {len(keys)} keys have "
            f"shards, but keys-failed={len(keys)}-{failed}={expected}. A completed item did "
            f"not write its shard, or the failure count is wrong."
        )


def reconcile(generation, combined_path, *, tag: str):
    """Combine this generation's shards and write the table to ``combined_path``.

    ``combined_path`` is deliberately OUTSIDE the generation directory: existing consumers read
    ``dest/<name>.parquet`` and moving it would break a documented CLI contract. The cost is that
    two generations write the same path, so `manifest_fields` records which one produced it.

    Reads the WHOLE generation, deliberately unlike `assert_conservation`, which counts only the
    pass's own keys. Every worker rebuilds the combined table from every shard; scoping this to one
    pass would leave a partitioned run's table holding only the last worker's slice.
    """
    import pandas as pd

    from scripts._partition import write_table_atomically

    shards = sorted(pathlib.Path(generation).glob("*.parquet"))
    frames = [pd.read_parquet(s) for s in shards]
    non_empty = [f for f in frames if len(f)]
    combined = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    if len(combined):
        write_table_atomically(combined, pathlib.Path(combined_path), tag=tag)
    return combined


def manifest_fields(generation, *, attempted: int, failed: int) -> dict:
    """The fields every adopting driver merges into its ``manifest_<tag>.json``.

    ``generation`` is here so a reader can tell whether the combined table beside the shard root
    corresponds to the generation directory beside it.

    ``attempted`` counts TRUE ATTEMPTS and EXCLUDES skips -- deliberately not the conservation
    quantity. `_partition.aggregate_manifests` strips a manifest of its vote on commit consistency
    only when it positively declares it built nothing, so a full-resume pass must be able to declare
    zero. Writing ``attempted + skipped`` would make such a pass claim it built the whole corpus,
    regain its vote, and reproduce the measured `commit_consistent: false` false alarm that
    `_partition.py` documents at length. Two quantities, two names.
    """
    return {
        "generation": pathlib.Path(generation).name,
        "n_attempted": int(attempted),
        "n_failed": int(failed),
    }


def _require_injective(keys) -> None:
    """Refuse a key function that maps two items to the same shard path.

    `join_key` validates that a key does not MIS-SPLIT on read; it cannot see that two different
    items produce the same key. That collision is silent data loss AND it defeats the conservation
    check, which is the worse half: item B finds A's shard, `already_done` returns True, B is
    counted as `skipped` and never processed -- and then `present` counts the same file once per
    duplicate key, so `present == len(keys)` and conservation CERTIFIES the run. A guard that
    reports a run with a dropped item as healthy is worse than no guard.

    It closes a second-order break in the same relation: `failures` is a dict keyed by the shard
    key, so two failures on colliding keys would collapse to one entry and `len(failures)` would
    UNDER-count -- making conservation raise spuriously. Enforcing injectivity up front makes both
    directions unreachable rather than guarding each.

    THIS is the primitives-path form, called BEFORE any work, where the caller has materialised its
    keys into a list anyway and can afford the up-front check. `for_each` cannot use it -- it
    streams -- and carries an equivalent inline check that fires at the colliding item instead.
    """
    dupes = sorted(k for k, n in collections.Counter(keys).items() if n > 1)
    if dupes:
        raise ValueError(
            f"key() is not injective over this corpus: {dupes}. Two items map to the same shard "
            f"path, so the second is skipped as 'already done' and silently lost. Include the "
            f"provider (or another distinguishing component) in the key -- providers in this "
            f"corpus demonstrably share match ids."
        )


@dataclasses.dataclass(frozen=True)
class CorpusPassResult:
    """What a pass did. ``shard_dir`` is the generation directory callers glob and reconcile."""

    shard_dir: pathlib.Path
    attempted: int
    skipped: int
    failed: int
    failures: dict
    counters: dict

    def manifest(self) -> dict:
        return manifest_fields(self.shard_dir, attempted=self.attempted, failed=self.failed)


def for_each(
    items,
    *,
    key,
    work,
    shard_root,
    token_inputs: Mapping[str, object],
    token_reason: str | None = None,
    counters=None,
    tag: str = "all",
    label: str = "item",
    max_consecutive_failures: int = 3,
) -> CorpusPassResult:
    """Walk ``items``, persisting each result so a crash resumes instead of restarting.

    ``work(item)`` returns ONE long-form DataFrame, or ``None`` meaning zero rows -- which still
    writes a shard. Per-item scalars go through ``counters(item, frame)`` into the manifest, where
    ``aggregate_manifests`` already sums ints and merges dict counters; that is why the contract is
    a tidy frame plus counters rather than a dict of frames, which no manifest could absorb.

    A failing item is recorded and skipped, because one bad item must not cost a whole corpus pass.
    ``max_consecutive_failures`` in a row aborts, because that is a systematic bug rather than bad
    luck, and a short clean-looking table is worse than a crash.
    """
    generation = generation_dir(shard_root, token_inputs=token_inputs, token_reason=token_reason)
    # STREAMED, never `list(items)`. `load_matches` is an Iterator that downloads and parses a match
    # -- actions AND a full tracking DataFrame -- inside the loop before yielding, and its own
    # docstring says `max_per_provider` "bounds total memory ... loading all matches at full depth
    # can OOM". Materialising the corpus would hold ~80 matches' frames alive at once, defeat resume
    # (nothing is skipped until everything has been downloaded), and invert this cycle's own thesis:
    # it indicts 14 drivers for holding every RESULT in memory, and inputs are far larger.
    own_keys: list[str] = []  # accumulated as we go -- 80 strings, not 80 tracking frames
    seen: set[str] = set()
    attempted = skipped = 0
    failures: dict = {}
    totals: dict = {}
    run = 0

    for i, item in enumerate(items, start=1):
        k = join_key(key(item))
        if k in seen:
            raise ValueError(
                f"key() is not injective over this corpus: {k!r} appeared twice. Two items map to "
                f"the same shard path, so the second would be skipped as 'already done' and silently "
                f"lost. Include the provider (or another distinguishing component) in the key -- "
                f"providers in this corpus demonstrably share match ids."
            )
        seen.add(k)
        own_keys.append(k)

        if already_done(generation, k):
            skipped += 1
            progress(f"{label} {k}", i, None, elapsed_s=0.0, note="skip (shard exists)")
            continue

        attempted += 1
        t0 = time.perf_counter()
        try:
            frame = work(item)
        except Exception as exc:  # broad BY DESIGN: recorded and counted, never swallowed silently
            failures[k] = f"{type(exc).__name__}: {exc}"
            run += 1
            progress(f"{label} {k}", i, None, elapsed_s=time.perf_counter() - t0, note=f"FAILED {exc}")
            if run >= max_consecutive_failures:
                raise RuntimeError(
                    f"aborting after {run} consecutive failures (last: {k}). This is a systematic "
                    f"problem, not a bad item; fix it rather than resuming past it."
                ) from exc
            continue

        run = 0
        write_shard(shard_path(generation, k), frame, tag=tag)
        if counters is not None:
            import pandas as pd

            for ck, cv in counters(item, pd.DataFrame() if frame is None else frame).items():
                totals[ck] = totals.get(ck, 0) + cv
        progress(
            f"{label} {k}",
            i,
            None,
            elapsed_s=time.perf_counter() - t0,
            note=f"{0 if frame is None else len(frame)} rows",
        )

    assert_conservation(generation, keys=own_keys, failed=len(failures))
    return CorpusPassResult(generation, attempted, skipped, len(failures), failures, totals)


def cohort_cache(path, *, build):
    """Fetch-once / reuse for a whole-cohort query, opt-in via an explicitly named path.

    Deliberately NOT automatic and deliberately NOT inside the loader. A query result has no token
    this module can compute without running the query, and the marts behind these cohorts
    re-materialize regularly -- so an automatic cache would silently serve a stale cohort, which is
    a plausible number from a computation that did not happen. A path the operator names cannot be
    reused by accident.

    ``path=None`` is a pure passthrough, so a caller that does not opt in behaves exactly as before.
    """
    import pandas as pd

    if path is None:
        return build()
    path = pathlib.Path(path)
    if not path.exists() and not (importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet")):
        # Before the multi-minute load, not after it: pandas only raises at to_parquet time.
        raise ValueError("--cohort-cache requires a parquet engine: pip install pyarrow")
    if path.exists():
        return pd.read_parquet(path)
    df = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
