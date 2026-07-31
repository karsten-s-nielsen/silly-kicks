"""Unit tests for the shared corpus-driver seam.

The digest itself is NOT tested here — it is `ruthless.fingerprint`, which carries its own golden
table of 44 pinned literals and a stated stability contract (ruthless 0.4.0). These tests pin what
`_driver` adds on top: that a token names a DIRECTORY, that an empty declaration needs a reason, and
that path inputs are normalised before they reach the digest.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath

import pandas as pd
import pytest

import scripts._driver as mod  # bare import: tests/scripts/ has NO __init__.py


def test_same_inputs_give_the_same_generation(tmp_path):
    a = mod.generation_dir(tmp_path, token_inputs={"box": 40.32, "provider": "gs"})
    b = mod.generation_dir(tmp_path, token_inputs={"box": 40.32, "provider": "gs"})
    assert a == b


def test_a_changed_input_gives_a_DIFFERENT_generation(tmp_path):
    a = mod.generation_dir(tmp_path, token_inputs={"box": 40.32})
    b = mod.generation_dir(tmp_path, token_inputs={"box": 40.30})
    assert a != b


def test_key_ORDER_does_not_change_the_generation(tmp_path):
    """`fingerprint` is order-insensitive over mappings. Pinned because a driver that builds its
    declaration conditionally would otherwise get a different token per branch order."""
    a = mod.generation_dir(tmp_path, token_inputs={"box": 40.32, "provider": "gs"})
    b = mod.generation_dir(tmp_path, token_inputs={"provider": "gs", "box": 40.32})
    assert a == b


def test_the_generation_is_a_DIRECTORY_that_exists(tmp_path):
    """The token names a directory, never a filename suffix. Reconciliation is a bare
    `glob("*.parquet")` at every existing site, so a suffix would let two generations of the same
    item land in one combined table."""
    gen = mod.generation_dir(tmp_path, token_inputs={"box": 40.32})
    assert gen.is_dir()
    assert gen.parent == tmp_path


def test_the_generation_name_is_a_16_hex_digest(tmp_path):
    """Pinned because `prune_stale_generations` gates a `shutil.rmtree` on exactly this shape."""
    gen = mod.generation_dir(tmp_path, token_inputs={"box": 40.32})
    assert len(gen.name) == 16
    assert all(c in "0123456789abcdef" for c in gen.name)


def test_an_EMPTY_declaration_without_a_reason_is_REFUSED(tmp_path):
    """`token_inputs={}` is a legal claim — 'this pass has no staleness risk' — but it must be a
    considered one. A silent omission and a deliberate declaration must not look identical."""
    with pytest.raises(ValueError, match="token_reason"):
        mod.generation_dir(tmp_path, token_inputs={})


def test_an_EMPTY_declaration_WITH_a_reason_is_allowed(tmp_path):
    gen = mod.generation_dir(tmp_path, token_inputs={}, token_reason="pure re-read, no derived state")
    assert gen.is_dir()


def test_different_empty_reasons_give_different_generations(tmp_path):
    """Non-vacuity: the reason must reach the digest, not merely satisfy a guard."""
    a = mod.generation_dir(tmp_path, token_inputs={}, token_reason="reason one")
    b = mod.generation_dir(tmp_path, token_inputs={}, token_reason="reason two")
    assert a != b


def test_equivalent_paths_normalise_to_the_SAME_generation(tmp_path):
    """`fingerprint` guarantees a stable digest for the same LOGICAL value; CONSTRUCTING that value
    is ours. `Path(str)` parses per-platform, so an un-normalised path input would orphan every
    shard when the same declaration is made on the DGX instead of the Windows box."""
    a = mod.generation_dir(tmp_path, token_inputs={"weights": PureWindowsPath("a/b/c.json")})
    b = mod.generation_dir(tmp_path, token_inputs={"weights": PurePosixPath("a/b/c.json")})
    assert a == b


def test_paths_NESTED_in_a_container_are_also_normalised(tmp_path):
    """A declaration is rarely a bare path — it is a list of them, or a dict of them."""
    a = mod.generation_dir(tmp_path, token_inputs={"w": [PureWindowsPath("a/b.json")]})
    b = mod.generation_dir(tmp_path, token_inputs={"w": [PurePosixPath("a/b.json")]})
    assert a == b


def test_a_DIFFERENT_path_still_gives_a_different_generation(tmp_path):
    """The other side of the band: normalisation must not collapse genuinely different paths."""
    a = mod.generation_dir(tmp_path, token_inputs={"w": PurePosixPath("a/b.json")})
    b = mod.generation_dir(tmp_path, token_inputs={"w": PurePosixPath("a/c.json")})
    assert a != b


def test_an_UNDIGESTIBLE_input_is_REFUSED_not_silently_hashed(tmp_path):
    """`fingerprint` is FAIL-CLOSED. The hand-rolled predecessor hashed `repr()`, so an object with
    no `__repr__` digested its MEMORY ADDRESS — a different token every process, so the driver never
    matched its own generation and silently recomputed the whole corpus on every run."""

    class Opaque:
        pass

    # Asserted against the ACTUAL raise, measured: ruthless emits
    #   TypeError: fingerprint: unsupported type 'Opaque'; extend _tag deliberately
    # A broad `pytest.raises(Exception)` with a loose regex would also pass if `generation_dir`
    # blew up for an unrelated reason, which is the failure mode this whole file is about.
    with pytest.raises(TypeError, match=r"fingerprint: unsupported type"):
        mod.generation_dir(tmp_path, token_inputs={"thing": Opaque()})


def test_prune_removes_stale_generations_and_KEEPS_the_current_one(tmp_path):
    root = tmp_path / "shards"
    old = mod.generation_dir(root, token_inputs={"box": 40.30})
    cur = mod.generation_dir(root, token_inputs={"box": 40.32})
    (old / "m1.parquet").write_bytes(b"stale")
    (cur / "m1.parquet").write_bytes(b"current")

    removed = mod.prune_stale_generations(root, keep=cur)

    assert removed == [old.name]
    assert not old.exists()
    assert cur.exists(), "the current generation is never pruned"
    assert (cur / "m1.parquet").is_file(), "and its shards survive"


def test_prune_with_only_the_current_generation_removes_NOTHING(tmp_path):
    """The other side of the band. Without it, `shutil.rmtree(root)` passes the test above."""
    root = tmp_path / "shards"
    cur = mod.generation_dir(root, token_inputs={"box": 40.32})
    assert mod.prune_stale_generations(root, keep=cur) == []
    assert cur.exists()


def test_prune_REFUSES_a_directory_that_is_not_a_generation(tmp_path):
    """`shard_root` is caller-supplied and may hold other things. Only a 16-hex token directory --
    the shape `ruthless.fingerprint` produces -- is eligible for a `shutil.rmtree`."""
    root = tmp_path / "shards"
    cur = mod.generation_dir(root, token_inputs={"box": 40.32})
    (root / "notes").mkdir()
    (root / "notes" / "readme.txt").write_text("keep me", encoding="utf-8")

    removed = mod.prune_stale_generations(root, keep=cur)

    assert removed == []
    assert (root / "notes" / "readme.txt").is_file()


def test_prune_on_a_MISSING_root_is_a_no_op(tmp_path):
    assert mod.prune_stale_generations(tmp_path / "never-ran", keep=tmp_path / "never-ran" / "abc") == []


def test_a_string_key_becomes_one_parquet_file(tmp_path):
    p = mod.shard_path(tmp_path, "gradientsports__10502")
    assert p.name == "gradientsports__10502.parquet"
    assert p.parent == tmp_path


def test_a_tuple_key_is_joined_with_the_separator(tmp_path):
    """Existing shards across this codebase are named `{provider}__{match_id}.parquet`. The tuple
    form preserves that byte-for-byte, so a migration is a path-prefix change and nothing more."""
    assert mod.shard_path(tmp_path, ("gradientsports", "10502")).name == "gradientsports__10502.parquet"


def test_a_component_containing_the_SEPARATOR_is_rejected_LOUDLY(tmp_path):
    """`validate_xshot_causal.py:266-268` already warns that a provider containing `__` "would
    silently mis-split". Silently is the problem: two providers sharing a game_id would overwrite
    each other's shard while the resume check reported a hit."""
    with pytest.raises(ValueError, match="__"):
        mod.shard_path(tmp_path, ("grad__sports", "10502"))


def test_an_EMPTY_component_is_rejected(tmp_path):
    """An empty component collapses two distinct keys onto one path."""
    with pytest.raises(ValueError):
        mod.shard_path(tmp_path, ("gradientsports", ""))


def test_a_NONE_result_STILL_writes_a_shard(tmp_path):
    """The invariant this cycle must not break.

    `build_layer2_spells.py:131-132`: "Written even when EMPTY: an absent shard means 'not yet
    run', a present empty one means 'run, produced no spell'. Conflating them would make a resume
    silently recompute." The same rule is in `validate_xshot_causal.py:230-232` and is pinned by
    `test_validate_xshot_causal_shards.py::test_an_EMPTY_shard_is_distinct_from_an_ABSENT_one`.

    None from `work` means ZERO ROWS, never NO SHARD.
    """
    p = mod.shard_path(tmp_path, "m1")
    mod.write_shard(p, None, tag="all")
    assert p.is_file()
    assert pd.read_parquet(p).empty


def test_an_EMPTY_frame_STILL_writes_a_shard(tmp_path):
    p = mod.shard_path(tmp_path, "m2")
    mod.write_shard(p, pd.DataFrame(), tag="all")
    assert p.is_file()


def test_already_done_distinguishes_ABSENT_from_EMPTY(tmp_path):
    """Absent means "not yet run"; present-and-empty means "ran, produced nothing". If these were
    conflated, every barren item would be recomputed on every resume, forever."""
    assert not mod.already_done(tmp_path, "m3")
    mod.write_shard(mod.shard_path(tmp_path, "m3"), None, tag="all")
    assert mod.already_done(tmp_path, "m3")


def test_write_shard_leaves_no_temp_file(tmp_path):
    """N workers rebuild from a shared directory, so a torn write is reachable."""
    mod.write_shard(mod.shard_path(tmp_path, "m4"), pd.DataFrame({"a": [1]}), tag="w0")
    assert not list(tmp_path.glob("**/*.tmp*"))


def test_an_EMPTY_STRING_key_is_rejected(tmp_path):
    """The string branch skips the separator check by design, so it needs its own empty guard --
    otherwise `shard_path(gen, "")` would silently produce `.parquet`."""
    with pytest.raises(ValueError, match="empty key"):
        mod.shard_path(tmp_path, "")


def test_a_SEQUENCE_key_that_would_COLLIDE_is_rejected(tmp_path):
    """The property the separator check actually protects. `("a__b", "c")` and `("a", "b__c")` both
    join to `a__b__c`: one shard serving two items, with the resume check reporting a hit for work
    that never ran. The bare-string branch cannot express this, which is why it is exempt."""
    with pytest.raises(ValueError, match="__"):
        mod.shard_path(tmp_path, ("a__b", "c"))
    with pytest.raises(ValueError, match="__"):
        mod.shard_path(tmp_path, ("a", "b__c"))


def test_progress_reports_index_total_and_label(capsys):
    mod.progress("match", 2, 64, elapsed_s=12.5, note="1234 rows")
    out = capsys.readouterr().out
    assert "2/64" in out
    assert "match" in out
    assert "1234 rows" in out


def test_progress_renders_an_UNKNOWN_total_as_a_question_mark(capsys):
    """The ONLY branch `for_each` takes. It streams, so it has no total and passes `n=None` at all
    three call sites (skip, failure, success) -- while the `2/64` case above is exercised only by the
    primitives path. Without this, a typo rendering `[3/None]` ships green."""
    mod.progress("match", 3, None, elapsed_s=1.0)
    out = capsys.readouterr().out
    assert "[3/?]" in out, f"unknown total must render as '?', got: {out!r}"
    assert "None" not in out


def test_progress_is_flushed(monkeypatch):
    """A detached DGX run that buffers its output is indistinguishable from a hung one. This is the
    trick `train_ghost_gk.py` already applies locally, with the comment "so background tasks show
    progress immediately" -- centralised here so every adopter inherits it."""
    seen = {}
    monkeypatch.setattr("builtins.print", lambda *a, **kw: seen.update(kw))
    mod.progress("match", 1, 2, elapsed_s=0.0)
    assert seen.get("flush") is True


def test_conservation_holds_when_every_item_wrote_a_shard(tmp_path):
    keys = ["m1", "m2", "m3"]
    for k in keys:
        mod.write_shard(mod.shard_path(tmp_path, k), None, tag="all")
    mod.assert_conservation(tmp_path, keys=keys, failed=0)


def test_conservation_accounts_for_a_FAILED_item(tmp_path):
    """A failed item is the ONLY thing that can be missing, because a completed item always writes
    a shard even when empty (see write_shard). So the relation needs no third category."""
    for k in ("m1", "m2"):
        mod.write_shard(mod.shard_path(tmp_path, k), None, tag="all")
    mod.assert_conservation(tmp_path, keys=["m1", "m2", "m3"], failed=1)


def test_conservation_FAILS_when_an_attempted_item_wrote_NO_shard(tmp_path):
    """What this genuinely proves: every item this pass attempted either wrote a shard or is
    counted as failed. It catches a completed item that silently skipped its write, and off-by-one
    counting. It does NOT prove the driver has no other loop -- see the docstring on the function."""
    mod.write_shard(mod.shard_path(tmp_path, "m1"), None, tag="all")
    with pytest.raises(AssertionError, match="conservation"):
        mod.assert_conservation(tmp_path, keys=["m1", "m2", "m3"], failed=0)


def test_conservation_counts_only_THIS_generation(tmp_path):
    """A stale generation sitting beside this one must not be counted. That is the F1 double-count
    in miniature, and this assertion detects it independently of the directory layout."""
    gen_a = mod.generation_dir(tmp_path, token_inputs={"v": "v1"})
    gen_b = mod.generation_dir(tmp_path, token_inputs={"v": "v2"})
    mod.write_shard(mod.shard_path(gen_a, "m1"), None, tag="all")
    mod.write_shard(mod.shard_path(gen_b, "m1"), None, tag="all")
    mod.assert_conservation(gen_b, keys=["m1"], failed=0)


def test_conservation_IGNORES_shards_outside_this_pass_slice(tmp_path):
    """THE partitioned-run case, and the reason this counts keys rather than globbing.

    N workers share one `--out` and therefore ONE generation directory: the token derives from
    `token_inputs`, which is identical across workers, and `tag` names the manifest file, not the
    directory. A worker owning a 10-match slice would otherwise glob a directory holding every
    other worker's shards too, compare that to 10, and raise -- non-deterministically, AFTER its
    multi-hour loop, and before its manifest was written, so `aggregate_manifests` would never see
    that partition. On re-run every item is skipped, the counts are unchanged, and it fires again:
    the driver could never complete.
    """
    gen = mod.generation_dir(tmp_path, token_inputs={"v": "v1"})
    for k in ("mine1", "mine2"):
        mod.write_shard(mod.shard_path(gen, k), None, tag="w0")
    for k in ("theirs1", "theirs2", "theirs3"):
        mod.write_shard(mod.shard_path(gen, k), None, tag="w1")

    mod.assert_conservation(gen, keys=["mine1", "mine2"], failed=0)


def test_conservation_accepts_a_GENERATOR_of_keys(tmp_path):
    """`keys` is counted and measured, so a naive implementation consumes it twice: the count
    exhausts a generator, the length then reads 0, and a healthy pass raises AFTER its expensive
    loop. `assert_conservation` is THE documented primitive for the escape-hatch path, where
    `keys=(join_key(k) for k in items)` is the natural thing to write."""
    gen = mod.generation_dir(tmp_path, token_inputs={"v": "v1"})
    mod.write_shard(mod.shard_path(gen, "m1"), None, tag="all")

    mod.assert_conservation(gen, keys=(k for k in ["m1"]), failed=0)


def test_reconcile_writes_the_combined_table_at_DEST_not_inside_the_generation(tmp_path):
    """`run_signoff_power --spells/--arm-values` reads `dest/<name>.parquet` today. Moving the
    combined table into the generation directory would be a Hyrum break on a documented CLI."""
    dest = tmp_path / "out"
    dest.mkdir()
    gen = mod.generation_dir(dest / "shards", token_inputs={"v": "v1"})
    mod.write_shard(mod.shard_path(gen, "m1"), pd.DataFrame({"a": [1]}), tag="all")
    mod.write_shard(mod.shard_path(gen, "m2"), pd.DataFrame({"a": [2]}), tag="all")

    out = mod.reconcile(gen, dest / "combined.parquet", tag="all")

    assert (dest / "combined.parquet").is_file()
    assert len(out) == 2


def test_reconcile_skips_EMPTY_shards_without_dropping_them_from_disk(tmp_path):
    """An empty shard contributes no rows but must remain on disk -- it is the resume check's
    record that the item ran."""
    gen = mod.generation_dir(tmp_path, token_inputs={"v": "v1"})
    mod.write_shard(mod.shard_path(gen, "m1"), pd.DataFrame({"a": [1]}), tag="all")
    mod.write_shard(mod.shard_path(gen, "barren"), None, tag="all")

    out = mod.reconcile(gen, tmp_path / "combined.parquet", tag="all")

    assert len(out) == 1
    assert mod.already_done(gen, "barren"), "the empty shard must survive reconciliation"


def test_reconcile_of_an_empty_generation_writes_no_table(tmp_path):
    gen = mod.generation_dir(tmp_path, token_inputs={"v": "v1"})
    out = mod.reconcile(gen, tmp_path / "combined.parquet", tag="all")
    assert out.empty
    assert not (tmp_path / "combined.parquet").exists()


def test_reconcile_still_reads_the_WHOLE_generation(tmp_path):
    """The deliberate asymmetry against Task 5: conservation is PER-PASS, reconciliation is
    CORPUS-WIDE. Every worker rebuilds the combined table from ALL shards -- today's behaviour,
    which must not change. If both were scoped the same way, a partitioned run's combined table
    would hold only the last worker's slice. Pinned so nobody "fixes" the two to match."""
    gen = mod.generation_dir(tmp_path, token_inputs={"v": "v1"})
    mod.write_shard(mod.shard_path(gen, "mine"), pd.DataFrame({"a": [1]}), tag="w0")
    mod.write_shard(mod.shard_path(gen, "theirs"), pd.DataFrame({"a": [2]}), tag="w1")

    out = mod.reconcile(gen, tmp_path / "combined.parquet", tag="w0")

    assert len(out) == 2, "reconcile must see every worker's shards, not just this pass's"


def test_manifest_fields_carry_the_generation_token(tmp_path):
    """Two generations write the same combined-table path, so the file is whichever token ran last.
    write_table_atomically makes that atomic, not ATTRIBUTABLE. Recording the token in the manifest
    makes the ambiguity visible rather than structural -- the way `commits_seen` already surfaces a
    multi-commit corpus."""
    gen = mod.generation_dir(tmp_path, token_inputs={"v": "v1"})
    fields = mod.manifest_fields(gen, attempted=3, failed=1)
    assert fields["generation"] == gen.name
    assert fields["n_attempted"] == 3
    assert fields["n_failed"] == 1


def _items(n):
    return [(f"m{i}", i) for i in range(n)]


def test_for_each_writes_one_shard_per_item_and_returns_counts(tmp_path):
    res = mod.for_each(
        _items(3),
        key=lambda it: it[0],
        work=lambda it: pd.DataFrame({"v": [it[1]]}),
        shard_root=tmp_path,
        token_inputs={"v": "v1"},
    )
    assert res.attempted == 3
    assert res.failed == 0
    assert len(list(res.shard_dir.glob("*.parquet"))) == 3


def test_for_each_SKIPS_an_item_whose_shard_already_exists(tmp_path):
    """Resume. The second run must not re-enter `work` for an item already on disk.

    Both calls spell out their keyword arguments rather than sharing a `**kw` dict: `dict(...)`
    collapses the values to a union type, so a type checker can no longer tell `token_inputs` from
    `max_consecutive_failures` and reports five spurious errors per call site. The repetition is two
    lines and it keeps the two runs legibly identical, which is the property under test.
    """
    mod.for_each(
        _items(2),
        key=lambda it: it[0],
        work=lambda it: pd.DataFrame({"v": [it[1]]}),
        shard_root=tmp_path,
        token_inputs={"v": "v1"},
    )

    entered = []
    res = mod.for_each(
        _items(2),
        key=lambda it: it[0],
        work=lambda it: entered.append(it) or pd.DataFrame(),
        shard_root=tmp_path,
        token_inputs={"v": "v1"},
    )

    assert entered == [], "work was re-entered for an item that already had a shard"
    assert res.skipped == 2


def test_for_each_recomputes_when_a_DECLARED_INPUT_changes(tmp_path):
    """The other side of resume: a changed token is a different generation, so nothing is reused."""
    mod.for_each(
        _items(2),
        key=lambda it: it[0],
        work=lambda it: pd.DataFrame({"v": [it[1]]}),
        shard_root=tmp_path,
        token_inputs={"v": "v1"},
    )
    entered = []
    mod.for_each(
        _items(2),
        key=lambda it: it[0],
        work=lambda it: (entered.append(it), pd.DataFrame({"v": [it[1]]}))[1],
        shard_root=tmp_path,
        token_inputs={"v": "v2"},
    )
    assert len(entered) == 2


def test_one_FAILING_item_does_not_lose_the_others(tmp_path):
    """One bad item must not cost fourteen hours. The failure is recorded and the pass continues."""

    def work(it):
        if it[0] == "m1":
            raise ValueError("bad item")
        return pd.DataFrame({"v": [it[1]]})

    res = mod.for_each(_items(4), key=lambda it: it[0], work=work, shard_root=tmp_path, token_inputs={"v": "v1"})

    assert res.attempted == 4
    assert res.failed == 1
    assert "m1" in res.failures
    assert len(list(res.shard_dir.glob("*.parquet"))) == 3


def test_consecutive_failures_ABORT(tmp_path):
    """Tolerating per-item failure must not turn a systematic bug into a short, clean-looking
    table. A run of consecutive failures is a systematic bug, not bad luck."""
    with pytest.raises(RuntimeError, match="consecutive"):
        mod.for_each(
            _items(10),
            key=lambda it: it[0],
            work=lambda it: (_ for _ in ()).throw(ValueError("always")),
            shard_root=tmp_path,
            token_inputs={"v": "v1"},
            max_consecutive_failures=3,
        )


def test_a_NONE_result_from_work_still_writes_its_shard(tmp_path):
    """End-to-end restatement of the invariant, through the loop rather than the primitive."""
    res = mod.for_each(
        [("barren", 0)], key=lambda it: it[0], work=lambda it: None, shard_root=tmp_path, token_inputs={"v": "v1"}
    )
    assert mod.already_done(res.shard_dir, "barren")
    assert res.failed == 0


def test_a_NON_INJECTIVE_key_is_refused_BEFORE_any_work(tmp_path):
    """Two items mapping to one shard path is silent data loss that the conservation check would
    CERTIFY as healthy: item B finds A's shard, is counted as skipped, and is never processed --
    while `present` counts the same file once per duplicate key, so present == len(keys).

    Measured against the pre-fix implementation: 2 items in, 1 processed, conservation (2, 2), pass.
    """
    entered = []
    items = [("gradientsports", "m1"), ("skillcorner", "m1")]  # same match_id, different providers

    with pytest.raises(ValueError, match="not injective"):
        mod.for_each(
            items,
            key=lambda it: it[1],  # match_id ONLY -- the natural mistake
            work=lambda it: entered.append(it) or pd.DataFrame(),
            shard_root=tmp_path,
            token_inputs={"v": "v1"},
        )

    # STREAMING contract: the collision is caught at the SECOND item, so exactly one item's work has
    # run. Not zero -- `for_each` consumes a generator that loads a match per iteration, so it cannot
    # inspect every key before starting without materialising the corpus (see the body comment).
    # What matters is preserved: item 2 is never silently counted as `skipped` and lost.
    assert len(entered) == 1, "the collision must be caught at the colliding item, not later"


def test_the_SAME_items_are_accepted_with_a_distinguishing_key(tmp_path):
    """Non-vacuity for the guard above: it must reject the key, not the corpus. Without this half,
    a guard that rejected everything would look identical."""
    items = [("gradientsports", "m1"), ("skillcorner", "m1")]
    res = mod.for_each(
        items,
        key=lambda it: (it[0], it[1]),
        work=lambda it: pd.DataFrame({"v": [1]}),
        shard_root=tmp_path,
        token_inputs={"v": "v1"},
    )
    assert res.attempted == 2


def test_for_each_does_NOT_materialise_the_corpus(tmp_path):
    """The blocker this streaming design exists for.

    `load_matches` yields (provider, match_id, actions, frames, home_team_id) and loads each match
    INSIDE the loop; its docstring says `max_per_provider` "bounds total memory ... loading all
    matches at full depth can OOM". A `list(items)` in `for_each` would hold every match's tracking
    frames alive at once and defeat resume, since nothing is skipped until everything is loaded.

    The assertion lives OUTSIDE `work`. `for_each` catches every exception `work` raises and records
    it as a per-item failure, and `AssertionError` is an `Exception` -- so an assertion inside `work`
    would be swallowed, three of them would trip `max_consecutive_failures`, and the test would fail
    with "aborting after 3 consecutive failures" instead of the diagnostic it was written to print.
    Still red against a materialising implementation, but pointing at the wrong cause. So `work`
    only RECORDS how many items had been produced by the time it was called, and the shape of that
    record is asserted afterwards."""
    produced: list[int] = []
    produced_at_each_call: list[int] = []

    def source():
        for i in range(5):
            produced.append(i)
            yield (f"m{i}", i)

    def work(item):
        produced_at_each_call.append(len(produced))
        return pd.DataFrame({"v": [item[1]]})

    res = mod.for_each(source(), key=lambda it: it[0], work=work, shard_root=tmp_path, token_inputs={"v": "v1"})

    # Streaming produces exactly one more item per call: [1, 2, 3, 4, 5].
    # `list(items)` produces all five before the first call: [5, 5, 5, 5, 5].
    assert produced_at_each_call == [1, 2, 3, 4, 5], (
        f"for_each must stream; items produced at each work() call was {produced_at_each_call}"
    )
    assert res.attempted == 5


def test_for_each_collects_per_item_counters(tmp_path):
    res = mod.for_each(
        _items(2),
        key=lambda it: it[0],
        work=lambda it: pd.DataFrame({"v": [it[1]]}),
        counters=lambda it, frame: {"n_rows": len(frame)},
        shard_root=tmp_path,
        token_inputs={"v": "v1"},
    )
    assert res.counters["n_rows"] == 2


def test_cohort_cache_is_a_pure_passthrough_when_no_path_is_given(tmp_path):
    """Absent the flag, behaviour must be byte-identical to today."""
    calls = []
    got = mod.cohort_cache(None, build=lambda: (calls.append(1), pd.DataFrame({"a": [1]}))[1])
    assert len(got) == 1
    assert calls == [1]


def test_cohort_cache_builds_once_then_reuses(tmp_path):
    calls = []
    path = tmp_path / "cohort.parquet"

    def build():
        calls.append(1)
        return pd.DataFrame({"a": [1, 2]})

    a = mod.cohort_cache(path, build=build)
    b = mod.cohort_cache(path, build=build)
    assert calls == [1], "the cohort was re-fetched despite a populated cache"
    assert len(a) == len(b) == 2


def test_cohort_cache_fails_FAST_when_no_parquet_engine_is_available(tmp_path, monkeypatch):
    """Kept from `calibrate_xt_bandwidth.py:225-239`: pandas only surfaces "Unable to find a usable
    engine" at write time, i.e. AFTER the multi-minute cohort load. Check before paying for it."""
    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="parquet engine"):
        mod.cohort_cache(tmp_path / "c.parquet", build=lambda: pd.DataFrame())
