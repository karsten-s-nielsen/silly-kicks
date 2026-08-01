"""Shared partition plumbing for the TF-19 corpus producers (scripts/_partition.py)."""

from __future__ import annotations

import json

import scripts._partition as mod  # bare import: tests/scripts/ has NO __init__.py


def _write(dest, name, payload):
    (dest / f"manifest_{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_integer_totals_SUM_across_partitions(tmp_path):
    """The defect this exists to prevent: N workers writing one shared manifest let the LAST one
    win, so a 64-match corpus reported a single partition's `n_matches: 8`."""
    for i in range(3):
        _write(tmp_path, f"p{i}", {"n_matches": 8, "n_spells": 100, "partition": f"p{i}"})
    got = mod.aggregate_manifests(tmp_path, defaults=("n_matches", "n_spells"))
    assert got["n_matches"] == 24
    assert got["n_spells"] == 300
    assert got["n_partitions"] == 3
    assert got["partitions"] == ["p0", "p1", "p2"]


def test_dict_fields_merge_as_COUNTERS(tmp_path):
    """Counters describe work that produced NO output row, so they can never be recovered by
    re-reading the shard table -- only by summing the per-worker manifests."""
    _write(tmp_path, "a", {"drop_reasons": {"no_possession": 5, "ball_far": 2}})
    _write(tmp_path, "b", {"drop_reasons": {"no_possession": 3, "other": 1}})
    got = mod.aggregate_manifests(tmp_path)
    assert got["drop_reasons"] == {"no_possession": 8, "ball_far": 2, "other": 1}


def test_boolean_flags_are_NOT_summed(tmp_path):
    """`bool` is an `int` subclass: a naive numeric sum would turn two dirty workers into `2`."""
    _write(tmp_path, "a", {"conservation_holds": True, "n_matches": 1})
    _write(tmp_path, "b", {"conservation_holds": True, "n_matches": 1})
    got = mod.aggregate_manifests(tmp_path)
    # Skipped outright rather than summed: a per-worker flag has no corpus-wide numeric meaning,
    # and `True + True == 2` would put a nonsense integer where a claim used to be. The producer
    # that owns the claim recomputes it from the aggregated totals instead.
    assert "conservation_holds" not in got
    assert got["n_matches"] == 2, "real integer totals must still sum alongside the skipped flag"


def test_a_consistent_commit_is_reported_as_ONE_string(tmp_path):
    for i in range(3):
        _write(tmp_path, f"p{i}", {"run_commit": "abc123", "run_tree_dirty": False})
    got = mod.aggregate_manifests(tmp_path)
    assert got["run_commit"] == "abc123"
    assert got["commit_consistent"] is True


def test_workers_from_DIFFERENT_commits_are_reported_not_hidden(tmp_path):
    """Nothing stops one worker being launched from a different checkout. That makes the corpus
    artifact a blend of two code versions while still looking like a single run -- the same class
    of false self-description as a clean SHA stamped on a dirty tree."""
    _write(tmp_path, "a", {"run_commit": "abc123"})
    _write(tmp_path, "b", {"run_commit": "def456"})
    got = mod.aggregate_manifests(tmp_path)
    assert got["commit_consistent"] is False
    assert got["run_commit"] == ["abc123", "def456"]


def test_ONE_dirty_worker_makes_the_whole_corpus_dirty(tmp_path):
    _write(tmp_path, "a", {"run_tree_dirty": False})
    _write(tmp_path, "b", {"run_tree_dirty": True})
    assert mod.aggregate_manifests(tmp_path)["run_tree_dirty"] is True
    # The other side: all-clean must not report dirty, or the flag is decoration.
    other = tmp_path / "clean"
    other.mkdir()
    _write(other, "a", {"run_tree_dirty": False})
    assert mod.aggregate_manifests(other)["run_tree_dirty"] is False


def test_an_empty_dir_yields_declared_defaults_not_a_crash(tmp_path):
    got = mod.aggregate_manifests(tmp_path, defaults=("n_matches", "n_spells"))
    assert got["n_matches"] == 0 and got["n_spells"] == 0
    assert got["n_partitions"] == 0
    assert got["commit_consistent"] is True  # vacuously: no worker disagreed


def test_a_ZERO_CONTRIBUTION_manifest_does_not_vote_on_commit_consistency(tmp_path):
    """MEASURED false positive this fixes. The §3.3 entanglement artifact reported
    `commit_consistent: false` from eight worker manifests unanimously at `6b242cf` PLUS one
    analysis manifest at `d1fc18d` carrying `n_matches: 0` -- it had built nothing, because every
    shard already existed. The DATA was single-commit; the flag said otherwise.

    A guard that cries wolf is worse than no guard: it teaches readers to skim past the one field
    built to be un-skippable.
    """
    for i in range(8):
        _write(tmp_path, f"p{i}", {"n_matches": 22, "run_commit": "6b242cf", "run_tree_dirty": False})
    _write(tmp_path, "all", {"n_matches": 0, "run_commit": "d1fc18d", "run_tree_dirty": False})
    got = mod.aggregate_manifests(tmp_path, defaults=("n_matches",))
    assert got["commit_consistent"] is True
    assert got["run_commit"] == "6b242cf", "the contributing commit is the corpus's commit"
    # ...but the non-contributor is still VISIBLE, not silently absorbed.
    assert got["commits_seen"] == ["6b242cf", "d1fc18d"]


def test_disagreeing_CONTRIBUTORS_are_still_caught(tmp_path):
    """The other side, and the whole reason the flag exists: two workers that BOTH built data at
    different commits must still fail. Narrowing the vote must not disarm it."""
    _write(tmp_path, "p0", {"n_matches": 10, "run_commit": "aaa1111", "run_tree_dirty": False})
    _write(tmp_path, "p1", {"n_matches": 10, "run_commit": "bbb2222", "run_tree_dirty": False})
    got = mod.aggregate_manifests(tmp_path, defaults=("n_matches",))
    assert got["commit_consistent"] is False
    assert got["run_commit"] == ["aaa1111", "bbb2222"]


def test_an_ALL_RESUME_aggregate_is_visibly_vacuous_not_quietly_true(tmp_path):
    """If nothing contributed, no manifest votes and the flag is `true` for lack of evidence rather
    than because of it. `commits_seen` is what makes that case inspectable."""
    _write(tmp_path, "p0", {"n_matches": 0, "run_commit": "aaa1111"})
    _write(tmp_path, "p1", {"n_matches": 0, "run_commit": "bbb2222"})
    got = mod.aggregate_manifests(tmp_path, defaults=("n_matches",))
    assert got["commit_consistent"] is True
    assert got["run_commit"] == []  # no contributor claimed it
    assert got["commits_seen"] == ["aaa1111", "bbb2222"], "two commits ran; the flag alone hides that"


def test_a_manifest_contributing_a_COUNTER_only_still_votes(tmp_path):
    """Contribution is not just `n_matches`: a pass whose only output is a drop-reason counter did
    real work on real data and must vote. Keying the rule to one field name would miss it."""
    _write(tmp_path, "p0", {"drop_reasons": {"no_possession": 5}, "run_commit": "aaa1111"})
    _write(tmp_path, "p1", {"n_matches": 3, "run_commit": "bbb2222"})
    got = mod.aggregate_manifests(tmp_path)
    assert got["commit_consistent"] is False, "a counter-only contributor was wrongly ignored"


def test_a_provider_with_NO_ids_in_this_slice_is_dropped_not_expanded():
    """MEASURED trap in the shared loader::

        wanted = (match_ids.get(provider) if match_ids else None) or list(manifest_ids)

    An empty list is falsy and an absent key is None, so BOTH fall through to the ENTIRE manifest.
    Verified directly against `_wanted_for_provider`: a slice of `{'idsse': []}` returned all seven
    manifest ids. For a partitioned run that inverts the intent -- a worker handed nothing for a
    provider would process ALL of it, N times over, with N processes writing the same shard paths.
    """
    assert mod.providers_for_slice(["a", "b"], {"a": ["1"], "b": []}) == ["a"]
    assert mod.providers_for_slice(["a", "b"], {"a": ["1"]}) == ["a"]  # absent key, not just empty
    assert mod.providers_for_slice(["a", "b"], {"a": ["1"], "b": ["2"]}) == ["a", "b"]


def test_no_slice_at_all_still_means_EVERY_provider():
    """The other side: an unpartitioned run must not be narrowed to nothing. `None` means "no
    partitioning", which is the loader's own reading and the correct one for a single-process run."""
    assert mod.providers_for_slice(["a", "b"], None) == ["a", "b"]
    assert mod.providers_for_slice(["a", "b"], {}) == ["a", "b"]


def test_atomic_write_leaves_no_temp_file_and_lands_the_whole_table(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"a": range(50)})
    dest = tmp_path / "t.parquet"
    mod.write_table_atomically(df, dest, tag="p0")
    assert dest.is_file()
    assert pd.read_parquet(dest).equals(df)
    assert not list(tmp_path.glob("*.tmp*")), "temp file was left behind"


def test_two_workers_writing_CONCURRENTLY_never_collide_on_the_temp_path(tmp_path):
    """The whole point of the per-worker temp name. If both workers used one temp path, one would
    truncate the other's half-written file and `os.replace` would publish the wreck."""
    import pandas as pd

    dest = tmp_path / "t.parquet"
    big, small = pd.DataFrame({"a": range(500)}), pd.DataFrame({"a": range(3)})
    # Interleave by hand: p0 starts (writes its temp), p1 completes entirely, then p0 completes.
    tmp0 = dest.with_name(f"{dest.stem}.p0.tmp{dest.suffix}")
    big.to_parquet(tmp0, index=False)
    mod.write_table_atomically(small, dest, tag="p1")
    assert pd.read_parquet(dest).equals(small), "p1's complete table must be published"
    assert tmp0.is_file(), "p0's temp must be untouched by p1 -- separate names"


def test_worker_tag_names_the_partition_after_its_id_list(tmp_path):
    assert mod.worker_tag(str(tmp_path / "slice_03.json")) == "slice_03"
    assert mod.worker_tag(None) == "all"


def test_a_MIXED_generation_corpus_is_visible_in_the_aggregate(tmp_path):
    """MEASURED before the fix: 'generation' in aggregate -> False. A `str` matches neither the
    int-sum nor the dict-merge branch, so the field was silently dropped and two workers running
    against DIFFERENT staleness tokens produced an artifact that looked single-generation."""
    _write(tmp_path, "w0", {"generation": "aaa", "n_attempted": 4, "run_commit": "c1", "run_tree_dirty": False})
    _write(tmp_path, "w1", {"generation": "bbb", "n_attempted": 4, "run_commit": "c1", "run_tree_dirty": False})
    got = mod.aggregate_manifests(tmp_path, defaults=("n_attempted",))
    assert got["generations_seen"] == ["aaa", "bbb"]
    assert got["generation_consistent"] is False


def test_a_single_generation_corpus_reports_consistent(tmp_path):
    """The other side of the band. Without it, an implementation hard-coding False would pass."""
    _write(tmp_path, "w0", {"generation": "aaa", "n_attempted": 4, "run_commit": "c1", "run_tree_dirty": False})
    _write(tmp_path, "w1", {"generation": "aaa", "n_attempted": 4, "run_commit": "c1", "run_tree_dirty": False})
    got = mod.aggregate_manifests(tmp_path, defaults=("n_attempted",))
    assert got["generations_seen"] == ["aaa"]
    assert got["generation_consistent"] is True


def test_manifests_WITHOUT_a_generation_still_aggregate(tmp_path):
    """Every pre-cycle manifest on disk lacks the field. Absent must not read as inconsistent."""
    _write(tmp_path, "w0", {"n_matches": 4, "run_commit": "c1", "run_tree_dirty": False})
    got = mod.aggregate_manifests(tmp_path, defaults=("n_matches",))
    assert got["generations_seen"] == []
    assert got["generation_consistent"] is True


def test_an_UNNAMED_string_field_is_REPORTED_not_silently_dropped(tmp_path):
    """The trap this seam keeps springing. A `str` matches neither the int-sum nor the dict-merge
    branch, so an unnamed string field vanishes between the per-worker manifest and the corpus
    artifact. It caught this cycle twice -- `generation` and `run_tree_state`.

    Dropping stays the behaviour: a named case carries per-field SEMANTICS (`run_commit` is
    contributor-gated, `run_tree_dirty` is OR-ed, `generation` is a set-plus-consistency-flag), and
    a generic collector would give all of them one wrong semantic. What changes is that the drop is
    now VISIBLE in the output rather than silent."""
    _write(tmp_path, "w0", {"n_matches": 4, "some_new_field": "v1", "run_commit": "c1"})
    got = mod.aggregate_manifests(tmp_path, defaults=("n_matches",))
    assert got["dropped_fields"] == ["some_new_field"]
    assert "some_new_field" not in got, "reported, but still not aggregated -- semantics are per-field"


def test_named_fields_are_NOT_reported_as_dropped(tmp_path):
    """Non-vacuity: `dropped_fields` must name the unhandled, not everything."""
    _write(
        tmp_path,
        "w0",
        {"generation": "aaa", "n_matches": 4, "run_commit": "c1", "run_tree_dirty": False, "partition": "w0"},
    )
    assert mod.aggregate_manifests(tmp_path, defaults=("n_matches",))["dropped_fields"] == []


def test_a_FULL_RESUME_pass_does_not_re_arm_the_commit_false_alarm(tmp_path):
    """B3, and the regression this cycle would otherwise introduce.

    MEASURED: with `n_attempted: 64` on a pass that skipped all 64, `commit_consistent` flips to
    False -- reproducing the section 3.3 entanglement false alarm that `_partition.py` exists to
    prevent. `manifest_fields` is therefore called with `attempted=res.attempted` (true attempts),
    never `res.attempted + res.skipped`. The non-contributor still appears in `commits_seen`, so its
    lineage is recorded rather than erased."""
    _write(
        tmp_path,
        "w0",
        {"generation": "aaa", "n_attempted": 8, "n_failed": 0, "run_commit": "AAA", "run_tree_dirty": False},
    )
    # every item skipped -- this pass built nothing and must not vote
    _write(
        tmp_path,
        "resume",
        {"generation": "aaa", "n_attempted": 0, "n_failed": 0, "run_commit": "BBB", "run_tree_dirty": False},
    )
    got = mod.aggregate_manifests(tmp_path, defaults=("n_attempted",))
    assert got["commit_consistent"] is True, "a pass that built nothing must not vote"
    assert got["commits_seen"] == ["AAA", "BBB"], "but its commit is still recorded"
    assert got["run_commit"] == "AAA"
