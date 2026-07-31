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
from collections.abc import Mapping, Sized
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
    root = pathlib.Path(shard_root)
    gen = root / _token(token_inputs, token_reason)
    _warn_on_flat_shards(root, gen)
    gen.mkdir(parents=True, exist_ok=True)
    return gen


def _warn_on_flat_shards(root: pathlib.Path, gen: pathlib.Path) -> None:
    """Say so when a pre-generation shard set is sitting in ``shard_root`` and will be IGNORED.

    Every driver that had shards before this seam wrote them FLAT -- ``shard_root/<key>.parquet``.
    The generation directory is a path prefix change, so those files are still perfectly good, and
    the resume check simply never looks at them: the pass reports no skips and recomputes the whole
    corpus. That is silent, it is expensive exactly where this module is meant to save time (the
    measured case is a 64-match, multi-hour arm-values pass), and it looks identical to a healthy
    first run.

    A WARNING, not a raise or an automatic move. A flat file in ``shard_root`` is only PROBABLY a
    stale shard -- the directory is caller-supplied -- and silently relocating data on the strength
    of a guess is worse than the recompute it would avoid. The operator is told the exact
    destination and moves them.
    """
    if not root.is_dir():
        return
    flat = [p.name for p in sorted(root.glob("*.parquet"))]
    if not flat:
        return
    import warnings

    shown = ", ".join(flat[:3]) + (f", ... (+{len(flat) - 3})" if len(flat) > 3 else "")
    warnings.warn(
        f"{len(flat)} shard(s) sit directly in {root} and will be IGNORED: this pass reads "
        f"{gen}. They are pre-generation shards ({shown}); a resume cannot see them, so the corpus "
        f"will be recomputed in full. Move them into {gen} to resume, or delete them.",
        stacklevel=3,
    )


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


def _counters_path(generation, key) -> pathlib.Path:
    """Sidecar holding one item's counters. ``.json``, so it is invisible to every ``*.parquet``
    glob -- `reconcile`, `assert_conservation` and the drivers' own combines all stay unaffected."""
    return pathlib.Path(generation) / f"{join_key(key)}.counters.json"


def _write_counters(generation, key, values: Mapping[str, object]) -> None:
    """Persist an item's counters beside its shard, so a RESUME can still report them.

    MEASURED defect this closes: `counters` is called only for items a pass ATTEMPTS, so a fully
    resumed worker returned ``{}`` and wrote ``{'n_frames_in': 0, 'n_matches': 0}`` into its
    manifest -- while `build_gkdv_arm_values` states in-source that "corpus totals must come from
    summing per-worker manifests -- not from re-reading the table". A partitioned run in which ANY
    worker resumed therefore produced a corpus artifact under-reporting the corpus, silently.

    Deliberately NOT solved by recomputing `counters(item, shard)` on the skip path. The contract
    lets `counters` read per-item metadata the FRAME cannot carry via a closure over `work` -- which
    `build_gkdv_arm_values` does, for its `build_ghost_frames` report -- and on a skipped item that
    closure holds the PREVIOUS item's report. Replaying it would produce confident, wrong numbers,
    which is strictly worse than the zeros it replaced.

    Pass-scoped quantities (`n_attempted`, `n_failed`) are deliberately NOT persisted: a resumed
    pass genuinely attempted nothing, and `_partition.aggregate_manifests` depends on it being able
    to say so. Corpus-scoped counters and pass-scoped counts are different quantities.
    """
    import json

    _counters_path(generation, key).write_text(json.dumps(dict(values), default=str), encoding="utf-8")


def _read_counters(generation, key) -> dict | None:
    """An item's persisted counters, or ``None`` when this generation predates the sidecar."""
    import json

    path = _counters_path(generation, key)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A truncated sidecar (killed mid-write) is UNKNOWN, never zero: the caller counts it as
        # unrecorded so the manifest can say so, rather than quietly under-reporting the corpus.
        return None


def progress(label: str, i: int, n: int | None, *, elapsed_s: float, note: str = "") -> None:
    """One line per item, FLUSHED.

    An unflushed detached run is indistinguishable from a hung one, which is how a 14-hour pass
    became unobservable. Flushing per item costs nothing at corpus scale.

    ``n`` is optional because `for_each` STREAMS its corpus and a generator has no length, so the
    total renders ``?``. When ``items`` is already ``Sized`` -- a list of ids from
    ``select_match_ids``, say -- `for_each` reads that length and the total is real. Counting an
    unsized corpus just to render a denominator would reintroduce exactly the materialisation
    `for_each` avoids, so the ``?`` stays for generators.
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

    **PRECONDITION: the driver has a partition surface** -- `--match-ids-json` plus a worker tag,
    where every run over one ``--out`` is a slice of ONE logical corpus. Without it, this is the
    wrong helper, and wrong in a way that reads as correct. A driver whose corpus SELECTORS
    (``--providers``, ``--max-per-provider``) are not in ``token_inputs`` -- deliberately, so that
    narrowing a corpus reuses shards rather than re-downloading them -- has a generation directory
    that is a SUPERSET of any one run, and this returns the superset. Measured on
    `calibrate_xt_bandwidth` before it stopped using this: a ``--providers skillcorner`` run
    following a two-provider run returned both providers' matches, and its sweep would have run on
    a corpus nobody requested. Such a driver combines from its OWN keys via `shard_path`.
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


def manifest_fields(generation, *, attempted: int, failed: int, counters_unrecorded: int) -> dict:
    """The fields every adopting driver merges into its ``manifest_<tag>.json``.

    ``generation`` is here so a reader can tell whether the combined table beside the shard root
    corresponds to the generation directory beside it.

    **Prefer `CorpusPassResult.manifest()`; this form is for the primitives path.**
    ``counters_unrecorded`` has no default, and losing that default is a bug fix, not tidying.
    Every one of the three drivers migrated before it existed wrote
    ``manifest_fields(shard_dir, attempted=..., failed=...)`` by hand and silently took the 0 --
    so a resumed worker that could replay NONE of its sidecars produced a manifest reading
    ``n_counters_unrecorded: 0`` beside ``n_matches: 0``, i.e. a corpus artifact that reported a
    corpus of nothing and asserted the report was complete. Measured by planting the old call back
    (``test_layer2_manifest_REPORTS_counters_it_could_not_replay`` goes red). A parameter whose
    wrong value is invisible must not have a convenient default.

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
        # Skipped items whose counters could not be replayed (a pre-sidecar generation, or a
        # sidecar truncated by a kill). Non-zero means the corpus-scoped counters beside this field
        # UNDER-report the corpus, and says so in the artifact instead of leaving the reader to
        # infer it from a number that looks complete. `aggregate_manifests` sums it like any int.
        "n_counters_unrecorded": int(counters_unrecorded),
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


def _merge_counters(totals: dict, produced: Mapping[str, object], key: str) -> None:
    """Fold one item's counters into the running totals. Ints SUM, dict counters MERGE.

    Mirrors `_partition.aggregate_manifests` exactly -- these totals are written straight into
    ``manifest_<tag>.json`` for that helper to aggregate, so the two must agree or a per-worker
    manifest cannot round-trip. A dict counter is not hypothetical: `build_gkdv_arm_values` carries
    `drop_reasons` (per-reason frame counts) beside its integer totals, and the int-only version
    raised ``unsupported operand type(s) for +: 'int' and 'dict'`` on it.

    ONE definition, shared by the fresh path and the resume-replay path. Written twice, the two
    would drift and a resumed manifest would aggregate differently from the pass that built it --
    which is precisely the discrepancy the sidecar exists to remove.
    """
    for ck, cv in produced.items():
        if isinstance(cv, dict):
            if not isinstance(totals.setdefault(ck, {}), dict):
                raise TypeError(
                    f"counter {ck!r} was an int on an earlier item and a dict on {key!r}. "
                    f"Coercing would corrupt the manifest a corpus artifact is built from."
                )
            bucket = totals[ck]
            for sub, n in cv.items():
                bucket[sub] = bucket.get(sub, 0) + n
        else:
            if isinstance(totals.setdefault(ck, 0), dict):
                raise TypeError(
                    f"counter {ck!r} was a dict on an earlier item and a scalar on {key!r}. "
                    f"Coercing would corrupt the manifest a corpus artifact is built from."
                )
            totals[ck] += cv


def _known_total(items) -> int | None:
    """``len(items)`` when the caller already holds one, else ``None`` -- never a count of a stream.

    `[37/?]` tells a maintainer watching a detached 14-hour pass that it is alive; `[37/80]` tells
    them when it ends, which is the question they actually have. The `isinstance` respects the
    streaming rule rather than working around it: a generator is not `Sized`, so it keeps the `?`,
    and nothing is consumed or materialised to find out which it is.

    A FUNCTION rather than an inline conditional so the narrowing stays here: written inline, the
    `isinstance` narrows `items` to `Sized` for the remainder of `for_each`, and `Sized` has no
    `__iter__` -- so the very loop it is meant to annotate stops type-checking.
    """
    return len(items) if isinstance(items, Sized) else None


@dataclasses.dataclass(frozen=True)
class CorpusPassResult:
    """What a pass did. ``shard_dir`` is the generation directory callers glob and reconcile.

    ``keys`` is every joined shard key THIS pass covered, in order -- skips and failures included,
    which is what makes it the right input for a combine. It exists because `reconcile` is only
    correct for a driver with a partition surface (see its docstring), so every other adopter has
    to combine from its own keys -- and a STREAMED source cannot be walked a second time to rebuild
    them. Without this, such a driver re-derives the key rule at the read site, and the moment that
    second copy drifts from the ``key`` passed here the combine silently finds nothing.
    """

    shard_dir: pathlib.Path
    attempted: int
    skipped: int
    failed: int
    failures: dict
    counters: dict
    keys: tuple[str, ...] = ()
    counters_unrecorded: int = 0

    def manifest(self) -> dict:
        return manifest_fields(
            self.shard_dir,
            attempted=self.attempted,
            failed=self.failed,
            counters_unrecorded=self.counters_unrecorded,
        )


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
    n_total = _known_total(items)
    own_keys: list[str] = []  # accumulated as we go -- 80 strings, not 80 tracking frames
    seen: set[str] = set()
    attempted = skipped = counters_unrecorded = 0
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
            note = "skip (shard exists)"
            if counters is not None:
                replayed = _read_counters(generation, k)
                if replayed is None:
                    # Pre-sidecar generation, or a truncated sidecar. Counted and NAMED rather than
                    # treated as zero, so a manifest built from a mixed-vintage generation says that
                    # it is incomplete instead of quietly reporting a smaller corpus.
                    counters_unrecorded += 1
                    note = "skip (shard exists; counters unrecorded)"
                else:
                    _merge_counters(totals, replayed, k)
            progress(f"{label} {k}", i, n_total, elapsed_s=0.0, note=note)
            continue

        attempted += 1
        t0 = time.perf_counter()
        try:
            frame = work(item)
        except Exception as exc:  # broad BY DESIGN: recorded and counted, never swallowed silently
            failures[k] = f"{type(exc).__name__}: {exc}"
            run += 1
            progress(f"{label} {k}", i, n_total, elapsed_s=time.perf_counter() - t0, note=f"FAILED {exc}")
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

            produced = dict(counters(item, pd.DataFrame() if frame is None else frame))
            _merge_counters(totals, produced, k)
            # Persisted AFTER the shard: a crash between the two leaves the item looking not-yet-run,
            # which a resume simply redoes. The reverse order would leave counters for work whose
            # shard does not exist, and conservation would then raise on a healthy re-run.
            _write_counters(generation, k, produced)
        progress(
            f"{label} {k}",
            i,
            n_total,
            elapsed_s=time.perf_counter() - t0,
            note=f"{0 if frame is None else len(frame)} rows",
        )

    assert_conservation(generation, keys=own_keys, failed=len(failures))
    return CorpusPassResult(
        generation, attempted, skipped, len(failures), failures, totals, tuple(own_keys), counters_unrecorded
    )


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
