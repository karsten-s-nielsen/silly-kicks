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
import pathlib

import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"

#: Calling any of these means the driver pulls a corpus.
_CORPUS_CALLS = {"load_matches", "select_match_ids", "load_xtgk_cohort", "load_retention_cohort"}
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
    out = {}
    for p in sorted(_SCRIPTS.glob("*.py")):
        if p.name.startswith("_"):
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        if _is_corpus_driver(tree):
            out[p.stem] = tree
    return out


#: Drivers not yet migrated, each with the reason. Asserted EXACTLY, both ways (see the test below):
#: a new offender cannot join silently and a migrated one must be removed. EMPTY as of ADR-052 --
#: every in-population driver adopts the seam. It stays as the mechanism, not as a list: a new
#: unmigrated driver has somewhere to be recorded WITH a reason, and cannot arrive silently.
_NOT_YET_MIGRATED: dict[str, str] = {}


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
