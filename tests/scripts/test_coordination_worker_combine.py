"""B-1 (spec 8.3): a partitioned coordination pass combines EVERY worker's share, and refuses anything less.

Two disjoint worker slices are run into one ``--out`` with the real ``for_each``; the combine must see both, in
an order that does not depend on the split, and refuse a missing worker, a failed key, a key handed to two
workers, or a partitioned run with no ``--corpus-json`` to prove the population.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import pytest

from scripts._coordination_corpus import (
    COORD_EXCLUSION_REASONS,
    StageTimer,
    assert_exclusions_declared,
    combine_workers,
    expected_corpus,
    read_declared_exclusions,
    write_worker_partial,
)
from scripts._driver import for_each, join_key
from tests.scripts._script_population import coordination_corpus_drivers

_PROV = {"commit": "abc123", "dirty": False, "tree_state": "clean"}
_CORPUS = {"skillcorner": ["m1", "m2", "m3"], "idsse": ["d1"]}

#: The TF-58 corpus drivers, DECLARED (ADR-056). `test_coordination_driver_population_is_derivable_and_exact` derives
#: the same set independently and asserts equality, so a new or renamed coordination driver fails loudly here rather
#: than silently escaping the token guards below (review A-54). The consumer subset -- the two drivers that also carry
#: the artifact `params_src` provenance (M-5) -- is a real property, not the population, so it stays declared.
_COORDINATION_DRIVERS = (
    "calibrate_coordination",
    "derive_coordination_params",
    "validate_coordination_numerics",
    "validate_team_coordination",
)
_COORDINATION_CONSUMERS = ("validate_coordination_numerics", "validate_team_coordination")


def test_coordination_driver_population_is_derivable_and_exact():
    # ADR-056: the gates below DERIVE their population (tests.scripts._script_population.coordination_corpus_drivers),
    # they do not trust a hand-kept list. The derived set must equal the declared one exactly -- an added driver
    # (appears in derived, not declared) or a renamed one both fail here.
    assert coordination_corpus_drivers() == _COORDINATION_DRIVERS
    assert set(_COORDINATION_CONSUMERS) <= set(_COORDINATION_DRIVERS)


def _work(ref):
    provider, match = ref
    if match == "boom":
        raise RuntimeError("a match that fails")
    return pd.DataFrame({"provider": [provider] * 2, "match_id": [match] * 2, "value": [1.0, 2.0]})


def _run(dest, refs, tag):
    res = for_each(
        list(refs),
        key=lambda r: r,
        work=_work,
        shard_root=dest / "shards",
        token_inputs={"pass": "unit"},
        tag=tag,
        label="match",
    )
    write_worker_partial(dest, "unit", res, _PROV, StageTimer(), tag=tag)
    return res


def _all_keys(providers=("skillcorner", "idsse")):
    return {join_key((p, m)) for p in providers for m in _CORPUS[p]}


def test_two_disjoint_worker_slices_combine_into_the_whole_corpus(tmp_path):
    _run(tmp_path, [("skillcorner", "m3"), ("idsse", "d1")], "w1")
    _run(tmp_path, [("skillcorner", "m1"), ("skillcorner", "m2")], "w0")
    table, summary = combine_workers(tmp_path, "unit", expected=_all_keys())
    assert summary["n_workers"] == 2 and summary["n_listed"] == 4
    assert summary["population_checked_against"] == "corpus_json"
    assert sorted(set(zip(table["provider"], table["match_id"], strict=True))) == sorted(
        (p, m) for p in _CORPUS for m in _CORPUS[p]
    )


def test_the_combined_table_does_not_depend_on_the_split(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    refs = [("skillcorner", "m1"), ("skillcorner", "m2"), ("skillcorner", "m3"), ("idsse", "d1")]
    _run(one, refs, "all")
    _run(two, refs[2:], "w1")
    _run(two, refs[:2], "w0")
    serial, _ = combine_workers(one, "unit", expected=None)
    split, _ = combine_workers(two, "unit", expected=_all_keys())
    pd.testing.assert_frame_equal(serial, split)


def test_combine_categorical_is_byte_identical_and_categorical(tmp_path):
    # reduce-memory (rev 3): categorical=True reads the string columns dictionary-encoded into a shared sorted
    # CategoricalDtype. The combined table's VALUES + row ORDER are identical to the object path; the string
    # columns come back as `category` (the memory win). Byte-identity under the reduce is gated separately (D-5).
    obj_dir, cat_dir = tmp_path / "obj", tmp_path / "cat"
    refs = [("skillcorner", "m1"), ("skillcorner", "m2"), ("skillcorner", "m3"), ("idsse", "d1")]
    for d in (obj_dir, cat_dir):
        _run(d, refs[2:], "w1")
        _run(d, refs[:2], "w0")
    obj, _ = combine_workers(obj_dir, "unit", expected=_all_keys())
    cat, _ = combine_workers(cat_dir, "unit", expected=_all_keys(), categorical=True)
    assert isinstance(cat["provider"].dtype, pd.CategoricalDtype)
    assert isinstance(cat["match_id"].dtype, pd.CategoricalDtype)
    pd.testing.assert_frame_equal(cat, obj, check_dtype=False, check_categorical=False)


def test_the_population_digest_names_the_corpus_not_the_split(tmp_path):
    # D2's objective id (C27) is keyed on it: the same corpus split two ways is one population; a smaller one is not.
    one, two, three = tmp_path / "one", tmp_path / "two", tmp_path / "three"
    refs = [("skillcorner", "m1"), ("skillcorner", "m2"), ("skillcorner", "m3"), ("idsse", "d1")]
    _run(one, refs, "all")
    _run(two, refs[2:], "w1")
    _run(two, refs[:2], "w0")
    _run(three, refs[:3], "all")
    serial = combine_workers(one, "unit", expected=None)[1]["population_digest"]
    split = combine_workers(two, "unit", expected=_all_keys())[1]["population_digest"]
    smaller = combine_workers(three, "unit", expected=None)[1]["population_digest"]
    assert serial == split != smaller


def test_a_worker_that_never_ran_is_refused(tmp_path):
    _run(tmp_path, [("skillcorner", "m1"), ("skillcorner", "m2")], "w0")  # w1 (m3, d1) never ran
    with pytest.raises(SystemExit, match="miss 2 corpus key"):
        combine_workers(tmp_path, "unit", expected=_all_keys())


def test_a_failed_key_is_refused(tmp_path):
    _run(tmp_path, [("skillcorner", "m1"), ("skillcorner", "boom")], "w0")
    expected = {join_key(("skillcorner", "m1")), join_key(("skillcorner", "boom"))}
    with pytest.raises(SystemExit, match="1 key\\(s\\) failed"):
        combine_workers(tmp_path, "unit", expected=expected)


def test_a_key_handed_to_two_workers_is_refused(tmp_path):
    _run(tmp_path, [("skillcorner", "m1"), ("skillcorner", "m2")], "w0")
    _run(tmp_path, [("skillcorner", "m2"), ("skillcorner", "m3"), ("idsse", "d1")], "w1")
    with pytest.raises(SystemExit, match="more than one worker"):
        combine_workers(tmp_path, "unit", expected=_all_keys())


def test_several_shares_without_a_corpus_list_are_refused(tmp_path):
    _run(tmp_path, [("skillcorner", "m1")], "w0")
    _run(tmp_path, [("skillcorner", "m2")], "w1")
    with pytest.raises(SystemExit, match="no --corpus-json"):
        combine_workers(tmp_path, "unit", expected=None)


def test_one_partition_share_without_a_corpus_list_is_refused(tmp_path):
    # review R2-1: ONE share tagged as a partition (w0) is a slice of the corpus, not the corpus -- without a
    # --corpus-json nothing proves otherwise, so a verdict over it would be a one-slice verdict (B-1 reopened)
    _run(tmp_path, [("skillcorner", "m1")], "match_ids_w0")
    with pytest.raises(SystemExit, match="tagged 'match_ids_w0'"):
        combine_workers(tmp_path, "unit", expected=None)


def test_one_unpartitioned_share_without_a_corpus_list_is_the_corpus(tmp_path):
    # the other side: an unpartitioned run (one worker tagged 'all') IS the listed corpus
    _run(tmp_path, [("skillcorner", "m1"), ("skillcorner", "m2")], "all")
    table, summary = combine_workers(tmp_path, "unit", expected=None)
    assert summary["n_listed"] == 2 and summary["n_workers"] == 1 and len(table) > 0


def test_expected_corpus_reads_the_full_list_and_refuses_a_partition_without_it(tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(_CORPUS), encoding="utf-8")
    args = argparse.Namespace(corpus_json=str(corpus), match_ids_json=str(tmp_path / "match_ids_w0.json"))
    assert expected_corpus(args, ("idsse",)) == {join_key(("idsse", "d1"))}  # restricted to the pass's providers
    assert expected_corpus(argparse.Namespace(corpus_json=None, match_ids_json=None), ("idsse",)) is None
    with pytest.raises(SystemExit, match="--corpus-json"):
        expected_corpus(argparse.Namespace(corpus_json=None, match_ids_json="w0.json"), ("idsse",))


def test_the_summary_sums_every_workers_stage_timings(tmp_path):
    _run(tmp_path, [("skillcorner", "m1"), ("skillcorner", "m2")], "w0")
    _run(tmp_path, [("skillcorner", "m3"), ("idsse", "d1")], "w1")
    for tag, seconds in (("w0", 1.5), ("w1", 2.0)):
        path = tmp_path / f"manifest_unit.{tag}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["stage_seconds"] = {"corpus": seconds}
        path.write_text(json.dumps(manifest), encoding="utf-8")
    _, summary = combine_workers(tmp_path, "unit", expected=_all_keys())
    assert summary["stage_seconds"] == {"corpus": 3.5}


def test_workers_must_agree_on_the_consistent_fields(tmp_path):
    # e.g. D1's calibrated occlusion width: one worker computing a different W would split the corpus in two regimes.
    _run(tmp_path, [("skillcorner", "m1"), ("skillcorner", "m2")], "w0")
    _run(tmp_path, [("skillcorner", "m3"), ("idsse", "d1")], "w1")
    for tag, width in (("w0", 40.0), ("w1", 40.0)):
        path = tmp_path / f"manifest_unit.{tag}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["width_m"] = width
        path.write_text(json.dumps(manifest), encoding="utf-8")
    _, summary = combine_workers(tmp_path, "unit", expected=_all_keys(), consistent=("width_m",))
    assert summary["consistent"] == {"width_m": 40.0}
    path = tmp_path / "manifest_unit.w1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["width_m"] = 35.0
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="disagree on 'width_m'"):
        combine_workers(tmp_path, "unit", expected=_all_keys(), consistent=("width_m",))


def test_the_run_params_token_does_not_depend_on_a_workers_provider_subset():
    # every worker of a run must write ONE shard generation (the combine refuses a mix), whatever --providers it got
    from scripts._coordination_corpus import TF58_PROVIDERS, run_params_token

    one = run_params_token(argparse.Namespace(providers=("idsse",)))
    assert one == run_params_token(argparse.Namespace(providers=tuple(TF58_PROVIDERS)))
    assert one == run_params_token(argparse.Namespace(providers=("skillcorner", "gradientsports")))
    assert one != run_params_token(argparse.Namespace(providers=("idsse", "sportec")))  # a non-corpus provider joins


def test_no_driver_tokens_only_its_own_providers():
    # the regression guard: a driver that digests `params_token(args.providers ...)` directly re-splits a run into
    # per-worker generations; every corpus driver goes through run_params_token
    import ast
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    for driver in coordination_corpus_drivers():  # A-54: derived, not hand-listed
        tree = ast.parse((scripts / f"{driver}.py").read_text(encoding="utf-8"))
        direct = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "params_token"
        ]
        assert direct == [], f"{driver} calls params_token directly at lines {direct}"


# --------------------------------------------------------------------------- review A-37 / B minor 2: every refusal
def _edit_manifest(dest, tag, **fields):
    path = dest / f"manifest_unit.{tag}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(fields)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _two_workers(dest):
    _run(dest, [("skillcorner", "m1"), ("skillcorner", "m2")], "w0")
    _run(dest, [("skillcorner", "m3"), ("idsse", "d1")], "w1")


def test_two_shard_generations_are_refused(tmp_path):
    _two_workers(tmp_path)
    _edit_manifest(tmp_path, "w1", generation="0" * 16)
    with pytest.raises(SystemExit, match="different shard generations"):
        combine_workers(tmp_path, "unit", expected=_all_keys())


def test_two_commits_are_refused(tmp_path):
    _two_workers(tmp_path)
    _edit_manifest(tmp_path, "w1", run_commit="f" * 40)
    with pytest.raises(SystemExit, match="different commits"):
        combine_workers(tmp_path, "unit", expected=_all_keys())


def test_a_key_neither_produced_nor_excluded_is_refused(tmp_path):
    _two_workers(tmp_path)
    manifest = json.loads((tmp_path / "manifest_unit.w1.json").read_text(encoding="utf-8"))
    _edit_manifest(tmp_path, "w1", produced=manifest["produced"][1:])  # handed a key, says nothing of it
    with pytest.raises(SystemExit, match="neither produced nor excluded 1 key"):
        combine_workers(tmp_path, "unit", expected=_all_keys())


def test_shares_adding_a_key_beyond_the_corpus_are_refused(tmp_path):
    _two_workers(tmp_path)
    _run(tmp_path, [("idsse", "stray")], "w2")  # a key the corpus list does not hold
    with pytest.raises(SystemExit, match="add 1"):
        combine_workers(tmp_path, "unit", expected=_all_keys())


def test_every_corpus_pass_tokens_the_params_it_consumes():
    # review R2-3: the guard above forbids a bare params_token; this one requires PRESENCE -- every for_each token of
    # every TF-58 driver carries a "params" digest of the values it computes with, and the artifact consumers (D3,
    # the numerics gate) also their provenance (`**params_src`: source + both sha256, owner ruling M-5)
    import ast
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    consumers = set(_COORDINATION_CONSUMERS)
    for driver in coordination_corpus_drivers():  # A-54: derived, not hand-listed
        tree = ast.parse((scripts / f"{driver}.py").read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "for_each"]
        assert calls, driver  # non-vacuity: the driver has corpus passes
        for call in calls:
            token = next(kw.value for kw in call.keywords if kw.arg == "token_inputs")
            assert isinstance(token, ast.Dict), (driver, call.lineno)
            keys = {k.value for k in token.keys if isinstance(k, ast.Constant)}
            assert "params" in keys, (driver, call.lineno)
            if driver in consumers:
                unpacked = {getattr(v, "id", None) for k, v in zip(token.keys, token.values, strict=True) if k is None}
                assert "params_src" in unpacked, (driver, call.lineno)


# --------------------------------------------------------------------------- exclusions nit: declared-allow-list gate
def test_combine_summary_surfaces_the_excluded_keys(tmp_path):
    # the combine summary carries the ACTUAL excluded keys (not just the count), so a reduce can check them against a
    # DECLARED set. A worker that excludes a key it was handed is still "accounted" (B-1 passes).
    from scripts._item_outcome import ItemExcluded

    def work(ref):
        if ref[1] == "m2":
            raise ItemExcluded(reason="no_tracking")
        return pd.DataFrame({"provider": [ref[0]], "match_id": [ref[1]], "value": [1.0]})

    res = for_each(
        [("skillcorner", "m1"), ("skillcorner", "m2")],
        key=lambda r: r,
        work=work,
        shard_root=tmp_path / "shards",
        token_inputs={"pass": "unit"},
        tag="all",
        label="match",
    )
    write_worker_partial(tmp_path, "unit", res, _PROV, StageTimer(), tag="all")
    _table, summary = combine_workers(tmp_path, "unit", expected=None)
    assert summary["n_excluded"] == 1
    assert summary["excluded_keys"] == [join_key(("skillcorner", "m2"))]


def test_read_declared_exclusions_and_the_gate():
    # the declared set is an INPUT with a CLOSED reason vocabulary; the gate refuses an UNDECLARED exclusion and returns
    # the {key: reason} map of what is covered.
    assert read_declared_exclusions(None) == {}
    assert "no_tracking" in COORD_EXCLUSION_REASONS
    declared = {"skillcorner__m2": "no_tracking"}
    assert assert_exclusions_declared(["skillcorner__m2"], declared) == declared  # declared -> covered
    assert assert_exclusions_declared([], declared) == {}  # nothing excluded -> empty map
    with pytest.raises(SystemExit, match="not DECLARED"):
        assert_exclusions_declared(["skillcorner__m9"], declared)  # undeclared -> refuse the subset PASS


def test_read_declared_exclusions_refuses_a_reason_outside_the_vocabulary(tmp_path):
    bad = tmp_path / "decl.json"
    bad.write_text(json.dumps({"skillcorner__m2": "because_i_said_so"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="reason outside"):
        read_declared_exclusions(str(bad))


def test_geometry_rate_gate_is_a_declarable_reason(tmp_path):
    # ADR-115: the SkillCorner spec-4.4 geometry admission gate (ball/player off-pitch RATE) excludes matches at load
    # uniformly across every TF-58 driver; nf_reduce must be able to DECLARE those exclusions with a vocabulary reason.
    assert "geometry_rate_gate" in COORD_EXCLUSION_REASONS
    decl = tmp_path / "decl.json"
    decl.write_text(json.dumps({"skillcorner__m2": "geometry_rate_gate"}), encoding="utf-8")
    declared = read_declared_exclusions(str(decl))  # accepted by the closed vocabulary (RED before the token is added)
    assert declared == {"skillcorner__m2": "geometry_rate_gate"}
    # and the conservation gate is satisfied once the geometry-gate exclusion is declared with that reason.
    assert assert_exclusions_declared(["skillcorner__m2"], declared) == declared
