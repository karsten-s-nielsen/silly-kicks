"""Every driver that walks a corpus must adopt the shared `_driver` seam.

WHY THIS EXISTS. `scripts/validate_xs_probe.py` walked ~80 matches for 14 hours, held every result
in memory, wrote once at the end, and printed nothing. A crash at hour 13 lost the run. That was not
a missing convention: four partial mechanisms already existed, and three fully-resumable drivers
predate that script by weeks. Prose plus exemplars was not enough -- the same finding as
`test_provenance_wiring.py`.

WHAT THIS CHECKS. The population is DERIVED, so a new corpus driver is enrolled the moment it is
written. The verdict is ADOPTION: does the driver call `for_each`, or a registered primitive from
`scripts/_driver.py`?

WHAT IT CANNOT CHECK. Adoption is not correctness -- a driver can call `for_each` for something
trivial and still accumulate over the real corpus in a second loop. **That evasion is NOT caught**,
by this gate or by the runtime invariant: the second loop writes no shards and lists no keys, so
`_driver.assert_conservation` never sees it. An earlier draft claimed otherwise; the claim was
wrong. What `assert_conservation` does prove is narrower and still worth having -- every item a
pass ATTEMPTED either wrote a shard or is counted as failed. Covering the second-loop evasion needs
a fan-in check (the union of all manifests' key sets against the directory contents) and is a
recorded follow-up, not something this gate does today.

A PREVIOUS VERSION OF THIS GATE WAS WRONG IN BOTH DIRECTIONS. It scored capability tokens
(`"shard" in src`, `".is_file()" in src`, `"flush=True" in src`) and certified five drivers, three
of which are accumulate-then-write, while pinning `build_gkdv_arm_values` -- genuinely resumable --
as debt for lacking `flush=True`. Substring and keyword tests over source are not evidence of
behaviour. Do not reintroduce one.

Detection is deliberately BROADER than "calls load_matches" -- that narrow reading is precisely what
let `validate_xs_probe.py` through when the shard idiom was extracted. Any loop over
matches/games/providers qualifies, whatever the source.
"""

from __future__ import annotations

import ast

import pytest

from tests.scripts import _corpus_load_rules as clr
from tests.scripts._script_population import iter_scripts

#: The (refs, load) source factories a MIGRATED driver calls instead of a stream loader (owner-ratified
#: reuse, Task 8.5). They are `_LOADER_NON_CORPUS` (not a stream loader, not a loading loop -- see the
#: gate in `_corpus_load_rules`), so they are absent from `corpus_functions()`; but calling one still
#: means the module pulls a corpus, so detection must count them or a fully-migrated driver drops out of
#: `_population()` and its adoption stops being checked.
_SOURCE_FACTORIES = {"pining_source", "open_data_source", "events_only_loader"}
#: Calling any of these means the driver pulls a corpus. Spec section 5.7: the hand-written set is
#: REPLACED by the population derived from the loader modules (`_corpus_load_rules.corpus_functions`),
#: so a new corpus loader named by the convention is picked up automatically; the source factories are
#: unioned in because they are the post-migration seam (Task 8.5).
_CORPUS_CALLS = set(clr.corpus_functions()) | _SOURCE_FACTORIES
#: A corpus-shaped CLI surface. Paired with a per-item loop it means the same thing.
_CORPUS_ARGS = {"--data-dir", "--match-ids-json", "--max-per-provider", "--providers"}
#: The public surface of `scripts/_driver.py`. Calling any of them is adoption.
_DRIVER_API = {
    "for_each",
    "generation_dir",
    "shard_path",
    "join_key",
    "write_shard",
    "already_done",
    "progress",
    "assert_conservation",
    "reconcile",
    "manifest_fields",
    "cohort_cache",
    "prune_stale_generations",
}
#: The primitive an escape-hatch driver must call. `for_each` calls it internally.
_CONSERVATION = "assert_conservation"
#: The OTHER primitive it must call -- conservation alone is satisfiable by a lossy run.
_INJECTIVE = "_require_injective"
#: Primitives that mean "this driver runs a PER-ITEM pass and writes shards".
#:
#: `cohort_cache` is deliberately NOT among them, and neither is `prune_stale_generations`. A
#: Shape-B driver -- one uncached whole-cohort query, no per-item loop -- adopts the seam purely by
#: routing that query through `cohort_cache`, and demanding `assert_conservation` from it is not a
#: strict reading of the rule but an UNDEFINED one: there are no items to conserve, no keys to be
#: injective over, and no shard the count could be compared against. The only way to satisfy such a
#: demand is to call a conservation check on an empty key list, which asserts nothing and would
#: teach the next contributor that these calls are boilerplate to be appeased rather than invariants
#: to be meant.
#:
#: Found by execution: Task 15 gave the four loop-free xT-GK drivers `--cohort-cache` and the two
#: escape-hatch cases below went red for all four at once. The plan predicted "all pass".
_SHARD_PRIMITIVES = {
    "for_each",
    "generation_dir",
    "shard_path",
    "write_shard",
    "already_done",
    "assert_conservation",
    "reconcile",
}


def _runs_a_per_item_pass(tree: ast.AST) -> bool:
    return bool(_called_names(tree) & _SHARD_PRIMITIVES)


def _called_names(tree: ast.AST) -> set[str]:
    return {
        (getattr(n.func, "id", "") or getattr(n.func, "attr", "")) for n in ast.walk(tree) if isinstance(n, ast.Call)
    }


def _string_literals(tree: ast.AST) -> set[str]:
    return {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("--")
    }


def _accumulates(node: ast.AST) -> bool:
    """Append/extend/update/add, or a subscript assignment.

    The `.append`-only version of this predicate misclassified two drivers as having no per-item
    state -- `.extend` and `out[match_id] = value` are the same defect.
    """
    for n in ast.walk(node):
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Subscript) for t in targets):
                return True
        if isinstance(n, ast.Call) and (getattr(n.func, "id", "") or getattr(n.func, "attr", "")) in {
            "append",
            "extend",
            "update",
            "add",
        }:
            return True
    return False


def _has_per_item_loop(tree: ast.AST) -> bool:
    for n in ast.walk(tree):
        if not isinstance(n, ast.For):
            continue
        if isinstance(n.target, ast.Tuple) and len(n.target.elts) >= 3:
            return True
        if any(k in ast.dump(n.iter) for k in ("match", "game", "provider", "cohort")):
            return True
    return False


def _is_corpus_driver(tree: ast.AST) -> bool:
    if _called_names(tree) & _CORPUS_CALLS:
        return True
    return bool(_string_literals(tree) & _CORPUS_ARGS) and _has_per_item_loop(tree)


def _adopts(tree: ast.AST) -> bool:
    return bool(_called_names(tree) & _DRIVER_API)


def _uses_for_each(tree: ast.AST) -> bool:
    return "for_each" in _called_names(tree)


def _population() -> dict[str, ast.AST]:
    """Corpus drivers, filtered out of the SHARED script universe (Cycle B).

    The glob/parse scaffolding used to live here and again in the artifact-driver gate. Two
    independent walkers over `scripts/*.py` drift with nothing relating them, so the universe is
    single-sourced in `_script_population.iter_scripts` and only the PREDICATE differs. Verified
    byte-equivalent at the refactor: 22 drivers, 73 collected cases, before and after.

    `_is_corpus_driver` and its local `_string_literals` are deliberately UNCHANGED -- the shared
    seam's `string_literals` strips docstrings, and swapping it in here would silently move this
    gate's population. That is a separate decision from single-sourcing the universe.
    """
    return {name: tree for name, tree in iter_scripts().items() if _is_corpus_driver(tree)}


#: Drivers not yet migrated, each with the reason. Asserted EXACTLY, both ways (see the test below):
#: a new offender cannot join silently and a migrated one must be removed. EMPTY as of ADR-052 --
#: every in-population driver adopts the seam. It stays as the mechanism, not as a list: a new
#: unmigrated driver has somewhere to be recorded WITH a reason, and cannot arrive silently.
_NOT_YET_MIGRATED: dict[str, str] = {
    # EMPTY: every in-population corpus driver adopts the seam. `train_match_outcome_dependence` (the
    # last holdout -- 3,961 open-data matches held in memory, no shards) gained `for_each` resume in
    # Task 13, so its entry was removed. The mechanism stays so a new unmigrated driver has somewhere
    # to be recorded WITH a reason and cannot arrive silently.
}


@pytest.mark.parametrize("name", sorted(_population()))
def test_corpus_driver_adopts_the_shared_seam(name):
    if name in _NOT_YET_MIGRATED:
        pytest.skip(f"pending migration: {_NOT_YET_MIGRATED[name]}")
    assert _adopts(_population()[name]), (
        f"{name}.py walks a corpus but calls nothing from scripts/_driver.py. A driver delegated to "
        f"a remote box must persist each item so a crash resumes, skip work already done, and print "
        f"progress. Use `for_each`; if its loops genuinely cannot invert, use the primitives and "
        f"record why."
    )


@pytest.mark.parametrize("name", sorted(_population()))
def test_an_ESCAPE_HATCH_driver_still_asserts_conservation(name):
    """`for_each` calls `assert_conservation` internally. A driver on the primitives path is
    otherwise gated statically only -- and step 4 of the rollout deliberately puts the hardest
    multi-loop driver on exactly that path, so this is where it would go unchecked."""
    if name in _NOT_YET_MIGRATED:
        pytest.skip(f"pending migration: {_NOT_YET_MIGRATED[name]}")
    tree = _population()[name]
    if _uses_for_each(tree) or not _adopts(tree):
        pytest.skip("uses for_each (which asserts internally) or is covered by the adoption test")
    if not _runs_a_per_item_pass(tree):
        pytest.skip("cohort-cache-only adopter: no per-item pass, so conservation is undefined")
    assert _CONSERVATION in _called_names(tree), (
        f"{name}.py uses the primitives directly but never calls {_CONSERVATION}. Then neither the "
        f"static gate nor the runtime invariant covers it."
    )


@pytest.mark.parametrize("name", sorted(_population()))
def test_an_ESCAPE_HATCH_driver_still_checks_key_injectivity(name):
    """`assert_conservation` alone is SATISFIABLE BY A LOSSY RUN, so the conservation gate above is
    not sufficient on this path.

    A colliding key makes `already_done` return True for the duplicate, so the second item is
    skipped and lost -- and `present` counts the single shared shard once per duplicate key, giving
    `present == len(own_keys)`. Conservation then certifies the run as healthy. `for_each` closes
    this with an inline `seen` check it grew when it went streaming; the primitives path has to call
    `_require_injective` itself, and `items` is materialised there so the up-front form is cheap."""
    if name in _NOT_YET_MIGRATED:
        pytest.skip(f"pending migration: {_NOT_YET_MIGRATED[name]}")
    tree = _population()[name]
    if _uses_for_each(tree) or not _adopts(tree):
        pytest.skip("uses for_each (which checks inline) or is covered by the adoption test")
    if not _runs_a_per_item_pass(tree):
        pytest.skip("cohort-cache-only adopter: no per-item keys, so injectivity is undefined")
    assert _INJECTIVE in _called_names(tree), (
        f"{name}.py uses the primitives directly but never calls {_INJECTIVE}. A non-injective key "
        f"would then be silent AND self-certifying: the duplicate is skipped as 'already done' and "
        f"{_CONSERVATION} still passes."
    )


def test_the_pending_list_is_EXACT():
    """Fails BOTH ways -- the only thing that stops a debt list becoming a dumping ground."""
    actual = {n for n, tree in _population().items() if not _adopts(tree)}
    assert actual == set(_NOT_YET_MIGRATED), (
        f"newly unmigrated: {sorted(actual - set(_NOT_YET_MIGRATED))}; "
        f"now migrated, remove from the list: {sorted(set(_NOT_YET_MIGRATED) - actual)}"
    )


def test_the_population_is_not_silently_empty():
    """A derived population that resolved to nothing makes every case above vacuous."""
    pop = _population()
    assert len(pop) >= 20, f"detection collapsed: only {len(pop)} drivers found"
    for expected in ("validate_xs_probe", "build_layer2_spells", "train_ghost_gk"):
        assert expected in pop, f"{expected} no longer detected as a corpus driver"


def test_detection_catches_a_planted_UNMIGRATED_driver():
    """Non-vacuity for the detector: a naive corpus loop must be in-population and NOT adopting."""
    planted = ast.parse(
        "def main():\n"
        "    for provider, match_id, actions, frames, home in load_matches(providers=['x']):\n"
        "        results.append(expensive(frames))\n"
        "    write(results)\n"
    )
    assert _is_corpus_driver(planted)
    assert not _adopts(planted)


def test_detection_catches_a_planted_MIGRATED_driver():
    """The other side: adoption must be recognised, or the gate can never go green."""
    planted = ast.parse(
        "from scripts._driver import for_each\n"
        "def main():\n"
        "    for_each(load_matches(providers=['x']), key=k, work=w, shard_root=d, token_inputs={'v': 1})\n"
    )
    assert _is_corpus_driver(planted)
    assert _adopts(planted)


def test_the_accumulation_predicate_sees_more_than_append():
    """`_accumulates` is the corrected detector: the `.append`-only version misclassified
    `derive_opengoal_range` (`.extend`) and `validate_xtgk_possession_value` (`out[k] = v`) as
    having no per-item state, which would have made them exemption candidates."""
    for src in (
        "for m in matches:\n    out.append(x)\n",
        "for m in matches:\n    out.extend(x)\n",
        "for m in matches:\n    out[m] = x\n",
        "for m in matches:\n    seen.add(x)\n",
    ):
        assert _accumulates(ast.parse(src)), f"missed accumulation in: {src!r}"
    assert not _accumulates(ast.parse("for m in matches:\n    print(m)\n"))


def test_the_cohort_cache_EXEMPTION_does_not_become_a_blanket_escape():
    """Both sides of the exemption, because a one-sided version is how a gate rots into permission.

    A cohort-cache-only driver has no per-item pass, so conservation over "items" is undefined and
    it is exempt. A driver that writes SHARDS is running exactly the pass conservation exists to
    check -- adding `cohort_cache` alongside must not buy it an exemption it has not earned.
    """
    cache_only = ast.parse(
        "from scripts._driver import cohort_cache\n"
        "def main():\n"
        "    df = cohort_cache(args.cohort_cache, build=lambda: load_xtgk_cohort('gs'))\n"
    )
    assert _adopts(cache_only)
    assert not _runs_a_per_item_pass(cache_only)

    shard_writer = ast.parse(
        "from scripts._driver import cohort_cache, generation_dir, write_shard\n"
        "def main():\n"
        "    df = cohort_cache(args.cohort_cache, build=lambda: load_xtgk_cohort('gs'))\n"
        "    gen = generation_dir(d, token_inputs={'v': 1})\n"
        "    for m in matches:\n"
        "        write_shard(gen / f'{m}.parquet', f(m), tag='t')\n"
    )
    assert _runs_a_per_item_pass(shard_writer), "a shard-writing driver must NOT inherit the exemption"


def test_every_exempt_driver_really_has_no_shard_pass():
    """Meta-assertion over the LIVE population, not a plant: the exemption must be justified for
    each driver it actually fires on, or it is silently covering a real escape-hatch adopter."""
    exempt = [
        n for n, t in _population().items() if _adopts(t) and not _uses_for_each(t) and not _runs_a_per_item_pass(t)
    ]
    assert exempt, "no driver exercises the exemption -- the two cases above are then vacuous"
    for name in exempt:
        called = _called_names(_population()[name])
        assert "cohort_cache" in called, f"{name} is exempt but does not even use the cohort cache"
        assert not (called & _SHARD_PRIMITIVES), f"{name} writes shards yet was exempted"


# ==========================================================================================
# Corpus-driver load-seam gate (spec section 5). Rules A-D + derived population + ledgers.
# Landed RED (interpretation 6): the ledgers ARE the recorded 4ac26d0 violation set, so this
# file PASSES with them populated; Tasks 9-15 drain them, Task 17 asserts both empty.
# ==========================================================================================


def _parse(src: str) -> ast.AST:
    return ast.parse(src)


# --- Step 1: derived population, both ways -------------------------------------------------


def test_loader_population_is_classified_exactly_both_ways():
    public = {n for names in clr.public_loader_functions().values() for n in names}
    corpus = set(clr.corpus_functions())
    non_corpus = public - corpus
    assert non_corpus == set(clr._LOADER_NON_CORPUS), (
        f"unclassified public loader functions: {sorted(non_corpus - set(clr._LOADER_NON_CORPUS))}; "
        f"stale _LOADER_NON_CORPUS entries: {sorted(set(clr._LOADER_NON_CORPUS) - non_corpus)}"
    )


def test_corpus_set_contains_the_known_loaders_non_vacuous():
    corpus = clr.corpus_functions()
    for known in (
        "load_matches",
        "load_statsbomb_matches",
        "load_open_data_matches",
        "select_match_ids",
        "list_match_refs",
    ):
        assert known in corpus, f"{known} vanished from the derived corpus set"


# --- Step 2: scan every script, private included ------------------------------------------


def test_all_script_trees_includes_private_scripts():
    trees = clr.all_script_trees()
    assert "_xtgk_comparability" in trees and "_loader_pining" in trees, "private scripts not scanned"
    assert "_xtgk_comparability" not in iter_scripts(), "the public walker must still skip private"


# --- Step 3: Rule A (ledgered, both ways) + plants ----------------------------------------


def test_rule_a_ledger_is_exact_both_ways():
    live = {v.module for v in clr.rule_a()}
    assert live == set(clr._RULE_A_PENDING), (
        f"newly streaming: {sorted(live - set(clr._RULE_A_PENDING))}; "
        f"now migrated, drop from _RULE_A_PENDING: {sorted(set(clr._RULE_A_PENDING) - live)}"
    )


def test_rule_a_plants():
    red = clr.rule_a_tree("plant", _parse("def main():\n    load_matches(providers=['p'])\n"))
    assert red, "Rule A missed a bare stream-loader call"
    green = clr.rule_a_tree(
        "plant",
        _parse(
            "def main():\n"
            "    for_each(list_match_refs(providers=['p']), key=k, work=w,\n"
            "             load=lambda r: load_match(r, events_only=False), shard_root=d, token_inputs={'v': 1})\n"
        ),
    )
    assert not green, "Rule A flagged a migrated for_each driver"


# --- Step 4: Rule B (events_only always keyword) + plants ---------------------------------


def test_rule_b_is_green_every_load_match_passes_events_only():
    assert clr.rule_b() == [], f"load_match without events_only=: {clr.rule_b()}"


def test_rule_b_plants():
    red = clr.rule_b_tree("plant", _parse("def main():\n    load_match(r)\n"))
    assert red, "Rule B missed a load_match without events_only="
    green = clr.rule_b_tree("plant", _parse("def main():\n    load_match(r, events_only=False)\n"))
    assert not green, "Rule B flagged a keyword events_only= call"


# --- Step 5: Rule C (ledgered, both ways) + plants + green sites ---------------------------


def test_rule_c_ledger_is_exact_both_ways():
    live = {f"{v.module}.{v.func}" for v in clr.rule_c()}
    assert live == set(clr._RULE_C_PENDING), (
        f"newly un-sharded: {sorted(live - set(clr._RULE_C_PENDING))}; "
        f"now migrated, drop from _RULE_C_PENDING: {sorted(set(clr._RULE_C_PENDING) - live)}"
    )


def test_rule_c_red_plants():
    reds = {
        "i_direct_stream": "def main():\n    for x in load_matches(providers=['p']):\n        pass\n",
        "i_name_bound": "def main():\n    it = load_matches(providers=['p'])\n    for x in it:\n        pass\n",
        "i_load_param": "def main(load_fn):\n    for x in load_fn():\n        pass\n",
        "ii_body_load_match": "def main():\n    for r in refs:\n        m = load_match(r, events_only=False)\n",
        "ii_map_lambda": "def main():\n    list(map(lambda r: load_match(r, events_only=False), refs))\n",
        "ii_body_load_param": "def main(loader):\n    for r in refs:\n        loader(r)\n",
    }
    for name, src in reds.items():
        assert clr.rule_c_tree("plant", _parse(src)), f"Rule C missed red shape {name}"


def test_rule_c_green_plants():
    rebind = "def main():\n    it = load_matches(providers=['p'])\n    it = other()\n    for x in it:\n        pass\n"
    assert not clr.rule_c_tree("plant", _parse(rebind)), "Rule C flagged a rebound-before-loop iterable"
    id_only = "def main():\n    for mid in select_match_ids(providers=['p']):\n        work(mid)\n"
    assert not clr.rule_c_tree("plant", _parse(id_only)), "Rule C flagged an id-only loop over select_match_ids"


def test_rule_c_named_green_sites_stay_green():
    """Sites that iterate ids WITHOUT loading must not be flagged (spec section 5.5). These four are
    green at 4ac26d0; `_loader_pining_to_cache.main` is flagged until Task 15 moves its load loop."""
    live = {f"{v.module}.{v.func}" for v in clr.rule_c()}
    for site in (
        "train_gk_completion._corpus_taxonomy",
        "train_gk_completion.main",
        "train_xcross_attempt._corpus_fingerprint",
        "train_xshot_occurrence._corpus_fingerprint",
    ):
        assert site not in live, f"{site} was wrongly flagged as a loading loop"


# --- Step 6: Rule D (unadmitted events-only) + plants -------------------------------------


def test_rule_d_is_green_all_events_only_loads_are_admitted():
    assert clr.rule_d() == [], f"unadmitted events_only= load_match: {clr.rule_d()}"


def test_rule_d_red_plants():
    bare = clr.rule_d_tree("driver", _parse("def main():\n    load_match(r, events_only=True)\n"))
    assert bare, "Rule D missed a bare events_only=True in a driver"
    var = clr.rule_d_tree("driver", _parse("def main():\n    load_match(r, events_only=eo)\n"))
    assert var, "Rule D missed a variable-valued events_only="
    second = clr.rule_d_tree("_events_admission", _parse("def other():\n    load_match(r, events_only=True)\n"))
    assert second, "Rule D missed a second un-admitted call in the admission module"


def test_rule_d_green_plants():
    twin = clr.rule_d_tree(
        "_events_admission",
        _parse(
            "def events_only_loader(refs):\n"
            "    def load(ref):\n"
            "        return load_match(ref, events_only=True)\n"
            "    return load\n"
        ),
    )
    assert not twin, "Rule D flagged the admitted events_only_loader closure"
    producer = clr.rule_d_tree(
        "build_skillcorner_s1_event_validity",
        _parse("def _events_pass(refs):\n    for_each(refs, load=lambda r: load_match(r, events_only=True))\n"),
    )
    assert not producer, "Rule D flagged the registered producer's _events_pass"


# --- Step 7: key-pin scaffold + _KEY_EXCEPTIONS -------------------------------------------

#: A migrated driver's `for_each(key=...)` must equal `ref.key` OR its entry here (spec section 5,
#: interpretation: keys stay byte-identical to the pre-migration item key, so finished generations
#: resume). Extended by Tasks 9-15 as they migrate.
_KEY_EXCEPTIONS: dict[str, str] = {
    "build_rq_pass_scores": "ref.match_id",
    "measure_gs_shot_distribution": "f'{ref.provider}_{ref.match_id}'",
    "_xt_corpus": "key",
    # The two trainers key BOTH their sources (pining refs + --data-dir tuples) through a shared
    # module-level `_source_key` (a MatchRef -> its .key, a tuple -> (provider, match_id)).
    "train_xshot_occurrence": "_source_key",
    "train_xcross_attempt": "_source_key",
    "train_receiver_model": "ref.match_id",
    "validate_sb360_licensed_corpus": "_item_key",
    # BOTH sources keyed through a shared module-level `_item_key`: a StatsBomb MatchRef -> its
    # match_id, a Wyscout (match_id, actions) tuple -> its match_id, prefixed by the provider.
    "validate_team_kpi_reliability": "(args.provider, _item_key(item))",
    # Score pass keys by str(ref.match_id) so the single-competition generation's shard filenames stay
    # byte-identical to the pre-migration str(mid) keys (the fit prepass uses ref.key, a new generation).
    "build_territory_ranking_census": "str(ref.match_id)",
    # Keeps its pre-migration f"{comp}_{season}_{mid}" key (NOT ref.key -- which would be
    # "statsbomb__<mid>"), so finished coverage-shard generations resume across the migration.
    "build_sb360_coverage": "f'{ref.competition_id}_{ref.season_id}_{ref.match_id}'",
}


def test_ref_key_joins_exactly_like_the_old_item_key():
    from _driver import join_key
    from _loader_pining import MatchRef

    for p, m in (("skillcorner", "1"), ("gradientsports", "10502"), ("statsbomb", "3986784")):
        assert join_key(MatchRef(p, m).key) == join_key((str(p), str(m)))


def _for_each_load_key_sources() -> dict[str, set[str]]:
    """module -> set of `key=` source snippets for every `for_each(..., load=...)` call."""
    out: dict[str, set[str]] = {}
    for mod, tree in clr.all_script_trees().items():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and clr._call_name(node) == "for_each"):
                continue
            kws = {kw.arg: kw.value for kw in node.keywords}
            if "load" not in kws or kws.get("load") is None:
                continue
            key_node = kws.get("key")
            if key_node is None:
                continue
            out.setdefault(mod, set()).add(ast.unparse(key_node))
    return out


def test_every_migrated_driver_keeps_its_pre_migration_key():
    """A `for_each(..., load=...)` must key by `ref.key` (a lambda returning `<name>.key`) or be a
    driver named in `_KEY_EXCEPTIONS`. Tasks 9-15 extend `_KEY_EXCEPTIONS` as they migrate."""
    for mod, sources in _for_each_load_key_sources().items():
        for src in sources:
            keys_by_ref = src.endswith(".key")  # `lambda ref: ref.key`
            allowed = mod in _KEY_EXCEPTIONS
            assert keys_by_ref or allowed, (
                f"{mod}: for_each(load=...) keys by {src!r}, not `ref.key`. Add a _KEY_EXCEPTIONS entry "
                f"with its pre-migration key if that is intended."
            )


# --- Step 8: anti-rot meta-assertions -----------------------------------------------------


def test_ledgers_are_subsets_of_the_derived_population():
    """A ledger entry that no rule could ever produce is dead weight that hides a real omission."""
    a_universe = set(clr.all_script_trees())
    assert set(clr._RULE_A_PENDING) <= a_universe, sorted(set(clr._RULE_A_PENDING) - a_universe)
    c_modules = {q.split(".")[0] for q in clr._RULE_C_PENDING}
    assert c_modules <= a_universe, sorted(c_modules - a_universe)


def test_exemptions_and_allowlist_name_functions_that_exist():
    trees = clr.all_script_trees()

    def _funcs(mod: str) -> set[str]:
        tree = trees.get(mod)
        return {".".join(s) for s in _all_func_paths(tree)} if tree is not None else set()

    for mod, func in clr._STREAM_LOADER_EXEMPT:
        assert func in {p.split(".")[-1] for p in _funcs(mod)} or mod not in trees, f"{mod}.{func} missing"
    for qual in clr._UNSHARDED_LOOP_EXEMPT:
        mod, _, func = qual.partition(".")
        assert func in {p.split(".")[-1] for p in _funcs(mod)} or mod not in trees, f"{qual} missing"
    for qual in clr._UNADMITTED_EVENTS_ONLY_ALLOWED:
        mod, _, func = qual.rpartition(".")
        assert mod in trees, f"{qual} module missing"


def _all_func_paths(tree: ast.AST) -> set[tuple[str, ...]]:
    out: set[tuple[str, ...]] = set()

    def rec(node, stack):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.add((*stack, child.name))
                rec(child, (*stack, child.name))
            else:
                rec(child, stack)

    rec(tree, ())
    return out


def test_underivable_is_empty():
    """ADR-056 third bucket: nothing is a genuinely-invisible corpus driver -- the rules are complete
    by enumeration over the derived population. The one known residual is pinned as a plant below."""
    assert clr._UNDERIVABLE == {}, f"a driver was parked as underivable: {clr._UNDERIVABLE}"


def test_rule_c_residual_cross_function_indirection_is_uncaught():
    """The stated LIMIT (spec section 4.6), pinned rather than implied: a load hidden behind a
    cross-function call -- neither a load_* corpus function nor a `load*` parameter -- is NOT caught
    by Rule C. If a future rule catches it, this assertion flips and the limit note is updated."""
    residual = (
        "def _load_one(r):\n"
        "    return load_match(r, events_only=False)\n"
        "def main():\n"
        "    for r in refs:\n"
        "        _load_one(r)\n"
    )
    assert not clr.rule_c_tree("plant", _parse(residual)), (
        "Rule C now catches cross-function load indirection -- update the section 4.6 limit note."
    )
