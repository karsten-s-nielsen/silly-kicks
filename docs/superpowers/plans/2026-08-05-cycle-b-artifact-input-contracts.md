# Cycle B — artifact input contracts and registry completeness — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended)
> or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax
> for tracking.

**Goal:** Every research artifact declares which symbols its numbers depend on, every registry derives
its own population instead of guarding it with a floor, and the GS input-convention guard can see the
case it governs.

**Architecture:** Four registry gates (phase 1) land first and independently, each RED before its
repair. They establish the derive-the-population pattern that phase 2's contract mechanism then
depends on. Phase 2 ships the source-side declaration (`scripts/_input_contract.py`) and the
output-side artifact gate together, because designed apart they disagree about what counts as
provenanced. Phase 3 repairs the GS guard, whose measurement step is gated on the clean-tree rule.

**Tech Stack:** Python 3.10–3.14, pandas, numpy, pytest, `ast` (stdlib) for population derivation,
hashlib for contract digests. No new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-08-05-cycle-b-artifact-input-contracts-design.md`

## Global Constraints

* **ADR number: ADR-054.** Version: read off `main` at commit-prep, never assumed. `main` was at
  4.75.0 / ADR-053 when this plan was written, so expect 4.76.0 — confirm, do not assume.
* **Every gate lands RED first — or, where the tree is already correct, PROVES IT CAN FAIL via a
  planted defect that is then reverted.** Either way the failing output is pasted into the commit
  message or the ADR. A gate written after its own fix has never been observed to work.
  **Which applies is per-task and stated in the task**, so a passing step is not evidence the work
  was done wrong: Task 2 and Task 4 Step 4b are planted-defect cases (the DSL count and the `SWEPT`
  registry are both correct today), and K9's `docs/research/**` half in Task 6 is green by
  construction — measured 0 of 7 failing. Tasks 1, 3, 4 and Task 6's bundled-weights half carry a
  live red. Do NOT manufacture a failure to satisfy this bullet, and do not read a green step in a
  planted-defect task as "the predicate was not built as written" — that reconcile-first instruction
  (Task 1 Step 3) governs the live-red tasks only.
* **Lint at CI scope, never `.`**: `python -m ruff check silly_kicks/ tests/ scripts/`,
  `python -m ruff format --check silly_kicks/ tests/ scripts/`, `python -m pyright` (bare).
  Neither tool is on PATH — always `python -m`.
* **`scripts/` is ASCII-only** (ruff `RUF001/2/3` + the cp1252 `--help` gate). No `—`, `·`, `≥`, `→`
  in any file under `scripts/`. Markdown under `docs/` may use them; this plan's code blocks do not.
* **Never invoke a `scripts/*.py` that lacks argparse.** `--help` is ignored and `main()` runs.
* **A driver cannot run in the commit that introduces it** — `scripts/_provenance.py:73-76` counts
  untracked files as dirty. Commit the driver, then run it.
* **Test the failing side too.** Every gate needs a planted-defect test proving it can fail.
* No commits without explicit approval from the owner. Approval to commit includes approval to push.
* Run `python -m pytest tests/ -m "not e2e" -v --tb=short` to a **unique** log path with its own exit
  marker; never share a log file between two runs.

---

## Spec corrections established before writing this plan

Three of the spec's prescriptions do not survive contact with the code. Each was measured, and each
changes what gets built. **Fix the spec as part of Task 9**; build what is below.

**1. §3.3's independent source for K2 is the wrong surface, and its headline finding is an artifact.**
The spec says to pair `dir()` against `__all__` *on the same module*, and cites a live four-way
disagreement (`structural_pass_xfns`, `xcross_attempt_xfns`, `xshot_occurrence_xfns`, `xt_gk_xfns`) as
"exactly the signal an independent source exists to produce". Measured:

    dir(silly_kicks.tracking.features)  _xfns  28
    silly_kicks.tracking.__all__        _xfns  28
    dir(features) - tracking.__all__           []   <- empty

All four are exported at PACKAGE level. The disagreement is an artifact of comparing against
`features.__all__`, which is simply narrower than the package export — not a defect anyone must
answer for. Building against it would manufacture four false findings and force four pointless
`__all__` edits. **The correct independent source is the package `silly_kicks.tracking.__all__`**,
which is also what ADR-033 and ADR-051 pin to. K2's real defects are the remaining three: no atomic
coverage, the tautological meta-assertion, and the `>= 21` floor.

**2. §3.4's pin surface for item 26 would be vacuous on two of three registries.** Measured:
`silly_kicks.spadl.utils` has **no `__all__` at all**, and neither module-level surface exports any
`add_*`:

    spadl.utils.__all__          absent
    spadl.utils      add_* in __all__   0     <- pinning here asserts nothing
    atomic.spadl.utils                  0     <- same
    tracking.features                   25    <- narrower than the package's 33

Pin against the PACKAGE `__all__`, as ADR-033 does. That surface produces a real red (below).

**3. §3.1's reconciliation assertion is tautological once the universe is shared.** "Every artifact
driver that walks a corpus is in ADR-052's population" is true by construction the moment both gates
call the same predicate over the same script set. The reviewer's underlying concern is real — two
independent walkers drift — but the fix is structural, not an assertion: **single-source the universe**
(`iter_scripts()`) and pin that neither gate re-grows its own `glob`. Task 1 builds that.

---

## File Structure

| file | responsibility |
|---|---|
| `tests/scripts/_script_population.py` | **new.** One AST walk over `scripts/*.py`. The shared universe both population gates consume. |
| `tests/scripts/test_provenance_wiring.py` | modify. Replace the `>= 6` floor with the three-bucket completeness assertion. |
| `tests/scripts/test_corpus_driver_resilience.py` | modify. Route `_population()` through the shared seam; drop its local glob. |
| `tests/test_c4_aggregator_count.py` | **new.** Derive 33 from `tracking.__all__`, subtract `_NOT_ACTION_COUPLED`, pin the DSL string. |
| `tests/tracking/test_frame_aware_xfns_dup_action_id.py` | modify. Package-`__all__` meta-assertion, atomic surface, floor removed. |
| `tests/test_enrichment_nan_safety.py` | modify. Add the two-directional package pin beside the floors. |
| `tests/tracking/test_packing_xfns_leakage_guard.py` and the two sibling guards | modify. Same pin. |
| `scripts/_input_contract.py` | **new.** `declare_inputs()` + `contract_digest()`. |
| `tests/scripts/test_input_contracts.py` | **new.** Re-derive each driver's contract, compare to its committed artifacts, warn. |
| `tests/scripts/test_artifact_provenance_output.py` | **new.** K9. Walk committed artifacts, assert they carry provenance. |
| `scripts/measure_gs_shot_distribution.py` | **new.** Item 23 step 2a. Owner-tier GS, summary counts only. |
| `docs/research/gs_input_convention/` | **new.** 2a's provenanced artifact. |
| `tests/datasets/gradientsports/_generate_synthetic_match.py` | modify. Reshape to 2a's numbers. |
| `docs/superpowers/adrs/ADR-054-*.md` | **new.** |

---

## PHASE 1 — the four registry gates

### Task 1: Item 10 — `ARTIFACT_DRIVERS` completeness

**Files:**
- Create: `tests/scripts/_script_population.py`
- Modify: `tests/scripts/test_provenance_wiring.py:124-128` (the floor)
- Modify: `tests/scripts/test_corpus_driver_resilience.py:153-161` (`_population`)

**Interfaces:**
- Produces: `tests.scripts._script_population.iter_scripts() -> dict[str, ast.AST]`,
  `string_literals(tree) -> set[str]`, `SCRIPTS: pathlib.Path`. Consumed by this task's own gate and
  by `test_corpus_driver_resilience`. Tasks 5-8 do **not** consume it — they walk committed
  artifacts, not script sources — but Task 7's new driver must SATISFY this task's gate, and must be
  written so it does so **through code, not prose**: a driver whose docstring cites its own output
  path is harmless only because it is also a real driver, and relying on that is relying on the
  defect Step 4c pins shut.

- [ ] **Step 1: Create the shared universe**

```python
# tests/scripts/_script_population.py
"""One AST walk over `scripts/*.py`, shared by every gate that derives a driver population.

ADR-052's corpus-driver gate and ADR-054's artifact-driver gate need the same scaffolding --
glob, skip private, parse -- and differ only in their predicate. Two independent walkers would
drift with nothing relating them, which is the defect class this cycle exists to remove. The
reconciliation is structural: single-source the UNIVERSE, let the predicates differ.
"""

from __future__ import annotations

import ast
import pathlib

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def iter_scripts() -> dict[str, ast.AST]:
    """Every non-private script in `scripts/`, parsed once, keyed by stem."""
    return {
        p.stem: ast.parse(p.read_text(encoding="utf-8"))
        for p in sorted(SCRIPTS.glob("*.py"))
        if not p.name.startswith("_")
    }


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Identity of every leading Expr(Constant(str)) that IS a docstring."""
    out: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(n, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def string_literals(tree: ast.AST) -> set[str]:
    """Every string literal EXCEPT docstrings.

    Docstrings must be excluded or every content predicate fires on PROSE. Measured: three
    separate source-text scans were fooled during this cycle's review --
    `make_ghost_gk_golden` matched a `_weights` rule solely because its module docstring
    mentions `test_weights_bundle_golden.py`; `render_sb360_matrix` matched a `_provenance`
    scan through the sentence "No provenance guard, deliberately."; and
    `regenerate_gs_et_native_gk` matched a corpus-loader scan through a docstring saying it
    MIRRORS the loader. None of the three carried the literal in code.

    `tests/scripts/test_provenance_wiring.py` already learned this once -- its
    `_shells_out_to_rev_parse` is AST-matched on CALLS with the comment "they cannot tell a
    described defect from a committed one". This is that same lesson, in the shared seam so
    both populations inherit it.
    """
    skip = _docstring_ids(tree)
    return {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip
    }


def called_names(tree: ast.AST) -> set[str]:
    """Every called function/method name -- `f(...)` and `x.f(...)` alike."""
    return {
        (getattr(n.func, "id", "") or getattr(n.func, "attr", ""))
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
```

- [ ] **Step 2: Write the RED completeness gate**

Append to `tests/scripts/test_provenance_wiring.py`:

```python
from tests.scripts._script_population import called_names, iter_scripts, string_literals

#: Matched the derivation rule but is correctly NOT an artifact driver; reason required.
_NOT_A_DRIVER: dict[str, str] = {}

#: Genuinely a driver, enrolled, but invisible to the rule. MUST be empty on landing -- see
#: test_UNDERIVABLE_is_empty. The reason must say WHY it is invisible, so the entry can be
#: retired if the rule improves.
_UNDERIVABLE: dict[str, str] = {}


def _declares_an_out_flag(tree: ast.AST) -> bool:
    """Any `--*out*` flag, not just an `--out` prefix.

    Measured: the prefix rule missed `--report-out` (calibrate_xt_bandwidth) and `--output-dir`.
    """
    return any(s.startswith("--") and "out" in s for s in string_literals(tree))


#: Calls that persist something to disk. Keyed on the CALL, never on a filename literal.
_WRITE_CALLS = frozenset(
    {"write_text", "write_bytes", "to_parquet", "to_csv", "to_json", "savez",
     "savez_compressed", "write_table_atomically", "dump"}
)


def _writes_a_document(tree: ast.AST) -> bool:
    """Detect the WRITE, not the filename.

    An earlier draft matched a `.json`/`.md` suffix literal. Measured, that fails in BOTH
    directions: it MISSES `measure_cover_shadow_argmax_agreement`, which composes its path
    entirely from `args.out` and carries no suffix literal anywhere, and -- worse -- it misses
    `render_sb360_matrix`, the one script the spec names as the counter-example that MUST be a
    candidate so it can be excluded with a reason.
    """
    return bool(called_names(tree) & _WRITE_CALLS)


def _writes_bundled_weights(tree: ast.AST) -> bool:
    """The trainers name a bundled weights path instead of an out-flag. Without this clause the
    three trainers are underivable and the central assertion cannot hold."""
    return any("_weights" in s for s in string_literals(tree))


def _is_artifact_driver(tree: ast.AST) -> bool:
    return (_declares_an_out_flag(tree) and _writes_a_document(tree)) or _writes_bundled_weights(tree)


def _candidates() -> set[str]:
    return {n for n, tree in iter_scripts().items() if _is_artifact_driver(tree)}


def test_the_artifact_driver_population_is_EXACT():
    """Replaces `assert len(ARTIFACT_DRIVERS) >= 6`. A floor cannot detect an omission -- it passed
    at 18 entries while `render_sb360_matrix` and `validate_xcross_causal` were both missing."""
    expected = (set(ARTIFACT_DRIVERS) - set(_UNDERIVABLE)) | set(_NOT_A_DRIVER)
    missing = sorted(_candidates() - expected)
    stale = sorted(expected - _candidates() - set(_UNDERIVABLE))
    assert not missing, (
        f"scripts that look like artifact drivers but are enrolled nowhere: {missing}. Add them to "
        f"ARTIFACT_DRIVERS, or to _NOT_A_DRIVER with a reason."
    )
    assert not stale, f"enrolled but no longer derivable: {stale} -- record in _UNDERIVABLE with a reason"


def test_UNDERIVABLE_is_empty():
    """The blind spot this closes: a script that is neither derivable NOR enrolled is absent from
    every set here and the equality above still holds. That is only unreachable while every enrolled
    driver IS derivable. The day it stops being true, this says so."""
    assert not _UNDERIVABLE, (
        f"_UNDERIVABLE is non-empty: {sorted(_UNDERIVABLE)}. A driver invisible to the rule means a "
        f"NEW driver of the same shape would also be invisible -- broaden the rule instead."
    )


@pytest.mark.parametrize("bucket_name", ["_NOT_A_DRIVER", "_UNDERIVABLE"])
def test_exemptions_name_scripts_that_exist(bucket_name):
    """Self-burning-down, the way _UNMODELLED already is in the C4 gate."""
    bucket = {"_NOT_A_DRIVER": _NOT_A_DRIVER, "_UNDERIVABLE": _UNDERIVABLE}[bucket_name]
    stale = sorted(n for n in bucket if not (_SCRIPTS / f"{n}.py").is_file())
    assert not stale, f"{bucket_name} names scripts that no longer exist: {stale}"
```

- [ ] **Step 3: Run it and RECORD the red**

Run: `python -m pytest tests/scripts/test_provenance_wiring.py -k "population or UNDERIVABLE" -v`

**Expected, measured against `main` @ 5b1a0a1 with this exact predicate — 48 scripts walked, 26
candidates, 18 enrolled:** `test_the_artifact_driver_population_is_EXACT` FAILS listing **8** names,
and `test_UNDERIVABLE_is_empty` PASSES.

    build_worldcup_fixture            make_xcross_directional_fixture   validate_shot_goalmouth_sb
    calibrate_tracking_defaults       regenerate_gs_et_native_gk        validate_xtgk_possession_value
    render_sb360_matrix               stamp_feature_contracts

If your run does not show exactly these 8 with an EMPTY `_UNDERIVABLE`, the predicate was not built
as written — reconcile before continuing, rather than adjusting the registries to match.

Paste the failure into the commit message. This is the acceptance evidence that the gate sees what
the floor could not: it passed at 18 entries while three of these eight were real, un-enrolled,
external-data-consuming artifact drivers.

- [ ] **Step 4a: Enrol the three that are genuinely un-enrolled artifact drivers**

**This is the item finding live defects, and it is the largest unbudgeted piece of work in the
cycle.** Three of the eight consume data from outside the repository, write an artifact, and have
**no provenance guard at all** — the same class as `validate_xcross_causal`, the case that motivated
item 10. Verified individually, not by grep (a `_provenance` grep is itself fooled by prose — see the
seam docstring):

| script | external input | verified at |
|---|---|---|
| `calibrate_tracking_defaults` | pining / DGX corpus | `_loader_pining` import |
| `validate_shot_goalmouth_sb` | StatsBomb open data + pining | `from statsbombpy import sb` :535, `load_matches` :741 |
| `validate_xtgk_possession_value` | Databricks gold marts | `from scripts._loader_databricks import load_xtgk_cohort` :112 |

For each: add to `ARTIFACT_DRIVERS` with a comment naming the external source, then wire it to
satisfy the five existing source assertions — import `scripts._provenance`, add `--allow-dirty`,
call `require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)` from `main()` before any
corpus work, stamp `run_commit` + `run_tree_dirty` into its output, and never shell out to
`git rev-parse`.

Run `python -m pytest tests/scripts/test_provenance_wiring.py -v` after **each** one. Do not batch —
each script's `main()` shape differs.

**Do not exempt one because wiring it looks like scope creep.** An artifact driver with no guard is
the exact defect this item exists to surface, and an exemption reading "not wired yet" converts
`_NOT_A_DRIVER` into the debt list ADR-052's `test_the_pending_list_is_EXACT` docstring warns about:
*"the only thing that stops a debt list becoming a dumping ground."*

- [ ] **Step 4b: Write the five reasoned exemptions**

The other five read committed inputs and consume nothing external, so they are correctly
`_NOT_A_DRIVER`. Three are one class; two are individual judgements:

* **Fixture generators** — `build_worldcup_fixture`, `make_xcross_directional_fixture`,
  `regenerate_gs_et_native_gk`. One shared reason: they regenerate committed test fixtures from
  committed or local inputs. Guarding them would make a fixture unregenerable during the session
  that changes it — the same constraint `render_sb360_matrix` records.
* **`stamp_feature_contracts`** — writes into `_ghost_gk_weights` / `_xcross_weights` /
  `_xshot_weights`, i.e. **the very files Task 6's gate polices**. Decide it once, here, and state it
  so Task 6 inherits the same answer: it is NOT an artifact driver (it consumes nothing external and
  only copies an already-recorded commit), but its OUTPUT is policed by Task 6. That split is
  deliberate — the source-side guard says "do not produce artifacts from a dirty tree", the
  output-side gate says "artifacts must carry provenance", and a stamper needs only the second.
  Guarding it would also make it unusable at exactly the moment it is needed: repairing a provenance
  defect.
* **`render_sb360_matrix`** — the reason already exists verbatim at
  `test_provenance_wiring.py:31-33`, and the script's own docstring line 3 says "No provenance guard,
  deliberately."

Move the `render_sb360_matrix` reason from the comment at `test_provenance_wiring.py:31-33` into the
registry, verbatim, and delete the comment:

```python
_NOT_A_DRIVER: dict[str, str] = {
    "render_sb360_matrix": (
        "reads a COMMITTED registry and writes a document. It does no corpus work and consumes no "
        "external data, so the guard would add nothing and would make the report unrenderable "
        "during the session that produces it -- a guarded driver cannot run on the dirty tree that "
        "produces its own inputs."
    ),
}
```

Then re-run; expected green. The eight above are the complete population as measured at
`5b1a0a1` — any *additional* name means `scripts/` moved since, and it is a genuine finding: enrol
it if it consumes something outside the repository, otherwise exempt it with its own reason. **Do
not add a reason you cannot state**, and classify by reading the script, not by grepping it.

- [ ] **Step 4c: Prove the docstring exclusion is load-bearing**

The seam skips docstrings; without a test, a later "simplification" restores the prose sensitivity
silently. This is the direct analogue of the existing
`test_the_rev_parse_detector_distinguishes_a_CALL_from_PROSE` (`:117`).

```python
def test_the_population_rule_reads_CODE_not_PROSE():
    """Non-vacuity, against the real case.

    `make_ghost_gk_golden` carries ZERO `_weights` string literals in code -- its only match is a
    module docstring mentioning `test_weights_bundle_golden.py` while explaining why an output
    golden exists. A literal scan that includes docstrings enrols it as a candidate on the strength
    of a sentence of prose, and it would then need a bogus exemption.
    """
    tree = iter_scripts()["make_ghost_gk_golden"]
    assert not any("_weights" in s for s in string_literals(tree)), (
        "string_literals() is admitting docstrings again -- source-text heuristics cannot tell a "
        "described path from a written one"
    )
    raw = {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert any("_weights" in s for s in raw), (
        "fixture drifted: this test is only meaningful while the docstring still mentions a "
        "_weights path -- re-point it at another prose-only match or delete it"
    )
```

Note the second assertion: it pins that the *probe still probes something*. Without it, a docstring
edit turns the first assertion vacuously true and the guard silently stops guarding.

- [ ] **Step 5: Replace the floor**

Delete `assert len(ARTIFACT_DRIVERS) >= 6` from `test_the_driver_list_is_not_silently_empty_or_stale`
(line 126). Keep the per-name `is_file()` loop — it is the burn-down half and is not superseded.

- [ ] **Step 6: Prove the rule catches the motivating case**

```python
def test_the_rule_FLAGS_an_unenrolled_driver():
    """Non-vacuity, against the real case. `validate_xcross_causal` (4.74.0) had `--out`, wrote
    metrics.json, was absent from the tuple, and its artifact carried no provenance at all. If the
    rule cannot see it un-enrolled, it would not have prevented the thing it was built for."""
    tree = iter_scripts()["validate_xcross_causal"]
    assert _is_artifact_driver(tree)
    assert "validate_xcross_causal" in _candidates() - (
        (set(ARTIFACT_DRIVERS) - {"validate_xcross_causal"}) | set(_NOT_A_DRIVER)
    )
```

- [ ] **Step 7: Route ADR-052's population through the shared seam**

In `tests/scripts/test_corpus_driver_resilience.py`, replace the body of `_population` (lines
153-161) and delete the now-unused local `_SCRIPTS` glob:

```python
def _population() -> dict[str, ast.AST]:
    return {n: t for n, t in iter_scripts().items() if _is_corpus_driver(t)}
```

- [ ] **Step 8: Verify the refactor is faithful, then pin the universe**

Run: `python -m pytest tests/scripts/test_corpus_driver_resilience.py -v`
Expected: same number of parametrised cases as before the refactor (record both counts). A changed
count means the glob/skip rule differed and must be reconciled before proceeding.

```python
# tests/scripts/test_provenance_wiring.py
def test_both_population_gates_consume_the_SHARED_universe():
    """The reconciliation is structural, not a set relation.

    Asserting "every corpus-walking artifact driver is in ADR-052's population" is TAUTOLOGICAL
    once both gates call the same predicate over the same script set -- it cannot fail. What can
    fail is one gate re-growing its own glob, after which the two universes drift and nothing else
    here would notice.
    """
    import tests.scripts.test_corpus_driver_resilience as adr052

    src = pathlib.Path(adr052.__file__).read_text(encoding="utf-8")
    assert "iter_scripts" in src, "ADR-052's gate no longer consumes the shared universe"
    assert ".glob(" not in src, (
        "the corpus-driver gate re-grew its own glob over scripts/ -- route it through "
        "tests/scripts/_script_population.py or the two populations can drift apart silently"
    )
```

- [ ] **Step 9: Full-file verification**

Run: `python -m pytest tests/scripts/ -v`
Expected: all pass. Then `python -m ruff check tests/ && python -m ruff format --check tests/`.

---

### Task 2: Item 24 — pin the C4 aggregator count

**Files:**
- Create: `tests/test_c4_aggregator_count.py`
- Read-only: `docs/c4/architecture.dsl:23`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: nothing consumed later.

**Note on red:** unlike the other three, the DSL is **correct today** (it says 32; 33 registered
`add_*` minus the jersey helper = 32). There is no live defect, so this gate's red is a PLANTED
drift, observed and recorded. Say so in the commit message rather than implying a live find.

- [ ] **Step 1: Write the gate**

```python
# tests/test_c4_aggregator_count.py
"""The C4 DSL asserts a count of action-coupled aggregators; nothing pinned it to the code.

There are TWO correct numbers and picking the wrong one is the likely failure:

    33  registered add_* in tracking.__all__   the ADR-051 mirror-registry surface
    32  action-coupled aggregators             what the DSL sentence describes

They differ by `add_gradientsports_player_ids`, a jersey-number helper that enriches a roster and
is not coupled to an action. A maintainer who resolves the ambiguity by making the DSL quote 33
turns a true sentence false in a way no test would catch -- which is why this gate names both.
"""

from __future__ import annotations

import pathlib
import re

import silly_kicks.tracking as T

_DSL = pathlib.Path(__file__).resolve().parents[1] / "docs" / "c4" / "architecture.dsl"

#: Registered `add_*` helpers that are NOT action-coupled, each with a stated reason.
_NOT_ACTION_COUPLED: dict[str, str] = {
    "add_gradientsports_player_ids": (
        "jersey-number -> player_id helper. Enriches a ROSTER, takes no actions frame, and emits "
        "no per-action column, so it is not one of the aggregators the DSL sentence counts."
    ),
}


def _registered_add_star() -> set[str]:
    return {n for n in T.__all__ if n.startswith("add_")}


def test_the_dsl_aggregator_count_matches_the_code():
    registered = _registered_add_star()
    assert len(registered) >= 25, f"discovery looks broken, found {sorted(registered)}"

    expected = len(registered) - len(_NOT_ACTION_COUPLED)
    text = _DSL.read_text(encoding="utf-8")
    found = re.search(r"(\d+) action-coupled aggregators", text)
    assert found is not None, "docs/c4/architecture.dsl no longer states an aggregator count"
    assert int(found.group(1)) == expected, (
        f"architecture.dsl says {found.group(1)} action-coupled aggregators; the code registers "
        f"{len(registered)} add_* of which {len(_NOT_ACTION_COUPLED)} are not action-coupled, so "
        f"the sentence should say {expected}. Do NOT resolve this by quoting {len(registered)} -- "
        f"that number is the ADR-051 mirror-registry surface, a different quantity."
    )


def test_not_action_coupled_entries_are_registered_helpers():
    """Self-burning-down: an exemption for a helper that no longer exists is stale scaffolding."""
    stale = sorted(set(_NOT_ACTION_COUPLED) - _registered_add_star())
    assert not stale, f"_NOT_ACTION_COUPLED names helpers that are not registered: {stale}"
```

- [ ] **Step 2: Confirm it passes on today's tree**

Run: `python -m pytest tests/test_c4_aggregator_count.py -v`
Expected: PASS (33 - 1 == 32).

- [ ] **Step 3: Observe the planted red, then revert it**

Temporarily edit `docs/c4/architecture.dsl:23`, changing `32 action-coupled aggregators` to `33`.
Run the gate. Expected: FAIL with the "do NOT resolve this by quoting 33" message. Record the output,
then `git checkout -- docs/c4/architecture.dsl`.

**Warning:** `git checkout --` reverts to HEAD and destroys uncommitted work in that file. That is
safe here **because of task ORDER, not because of anything about the file**: Task 9 Step 4 is the
only other step that touches the DSL and it has not run yet. If Task 9 has already run, or the DSL
carries any other uncommitted edit, revert the one line by hand instead.

- [ ] **Step 4: Lint**

Run: `python -m ruff check tests/ && python -m ruff format --check tests/`

---

### Task 3: K2 — the ADR-020 dup-`action_id` gate

**Files:**
- Modify: `tests/tracking/test_frame_aware_xfns_dup_action_id.py:128-133`

**Interfaces:**
- Consumes: nothing. Produces: nothing.

**Read the spec correction above before starting.** The spec's prescribed comparison
(`dir(features)` vs `features.__all__`) produces four FALSE findings. Build against
`silly_kicks.tracking.__all__`, which agrees exactly today.

- [ ] **Step 1: Measure both surfaces and record the numbers**

Run:

```bash
python -c "
import warnings; warnings.filterwarnings('ignore')
import silly_kicks.tracking as T, silly_kicks.tracking.features as F
import silly_kicks.atomic.tracking.features as AF
d = {n for n in dir(F) if n.endswith('_xfns')}
a = {n for n in T.__all__ if n.endswith('_xfns')}
ad = {n for n in dir(AF) if n.endswith('_xfns')}
aa = {n for n in getattr(AF, '__all__', []) if n.endswith('_xfns')}
print('tracking  dir', len(d), 'pkg __all__', len(a), 'disagree', sorted(d ^ a))
print('atomic    dir', len(ad), 'mod __all__', len(aa), 'disagree', sorted(ad ^ aa))
"
```

Expected for tracking: `28 28 []`. Record the atomic numbers — they determine Step 3's expectation
and the spec's "22" was measured on the module `__all__`, which is the surface that exists there.

- [ ] **Step 2: Write the RED atomic-coverage assertion**

Replace lines 128-133 with:

```python
import silly_kicks.atomic.tracking.features as AF

_XFNS_NAMES = sorted(n for n in dir(F) if n.endswith("_xfns"))
_ATOMIC_XFNS_NAMES = sorted(n for n in dir(AF) if n.endswith("_xfns"))


def test_the_gate_sees_every_registered_xfns_factory():
    """Two INDEPENDENT derivations must agree: the runtime namespace and the declared export.

    The previous version asserted
        set(_XFNS_NAMES) == {n for n in dir(F) if n.endswith("_xfns")}
    -- the same expression on both sides, always true. It also carried
    `assert len(_XFNS_NAMES) >= 21  # bumped for xt_gk_xfns`, a floor inside the very gate that
    exists because floors cannot detect an omission, with a comment recording it had already been
    hand-bumped once.

    The independent source is the PACKAGE export, not `features.__all__`. Measured: all four names
    absent from `features.__all__` ARE in `tracking.__all__`, so pairing against the module surface
    would manufacture four findings that are not defects.
    """
    exported = {n for n in T.__all__ if n.endswith("_xfns")}
    assert set(_XFNS_NAMES) == exported, (
        f"runtime namespace and package export disagree: "
        f"dir-only={sorted(set(_XFNS_NAMES) - exported)}, "
        f"export-only={sorted(exported - set(_XFNS_NAMES))}"
    )


def test_the_gate_covers_the_atomic_mirrors():
    """The gate enumerated `dir(F)` over tracking.features ONLY, so the atomic mirrors have never
    been covered by ADR-020's dup-action_id contract at all."""
    assert _ATOMIC_XFNS_NAMES, "atomic mirror discovery found nothing"
```

- [ ] **Step 3: Run and RECORD the red**

Run: `python -m pytest tests/tracking/test_frame_aware_xfns_dup_action_id.py -k "gate" -v`

The atomic surface is newly enumerated, so its parametrised behavioural cases (Step 4) are what
land red. Record the count of atomic factories that were never exercised — that number is the
finding, and it goes in the ADR.

- [ ] **Step 4: Extend the behavioural parametrisation to the atomic surface**

`_build` (line 100) and `_frame_for` (line 80) hardcode the module `F`. Parameterise both on the
module, then add the sibling parametrisation. Replace lines 80-115 and 137-162 with:

```python
def _frame_for(name):
    fr = _frame()
    if name in _NEEDS_TEAM_IN_POSSESSION:
        fr = fr.copy()
        fr["team_in_possession"] = 1
    return fr


def _build(name, mod=F):
    fac = getattr(mod, name)
    if isinstance(fac, list):
        return fac
    xt = _xt()
    # home_team_id=1 preferred; (xt, home_team_id=1) for xt-takers; bare for factories
    # that take neither (e.g. pitch_control_xfns(method=...), elastic_sync_xfns(*,...)).
    for args, kw in (((), {"home_team_id": 1}), ((xt,), {"home_team_id": 1}), ((), {})):
        try:
            return fac(*args, **kw)
        except TypeError:
            continue
    raise AssertionError(
        f"{name}: no known construction signature -- extend _build (do NOT skip; an "
        f"unprobed family re-opens the hole this gate closes)."
    )


def _run_family(name, mod=F):
    """Run every frame-aware transformer of `name` through a dup-action_id gamestate.
    Discriminates the target bug from a fixture gap so the fix lands on the bug, not the fixture."""
    states = gamestates(_actions(), nb_prev_actions=3)
    assert states[1]["action_id"].duplicated().any()  # precondition: dup exists
    frame = _frame_for(name)
    for t in _build(name, mod):
        if not getattr(t, "_frame_aware", False):
            continue
        try:
            t(states, frame)
        except Exception as exc:
            if _is_dup_symptom(str(exc)):
                raise AssertionError(
                    f"{mod.__name__}.{name}: DUP-ACTION_ID BUG -- retrofit to "
                    f"resolve_frame_ids_by_position / dedup provenance merge."
                ) from exc
            raise AssertionError(
                f"{mod.__name__}.{name}: non-dup error ({type(exc).__name__}: {exc}). This is a "
                f"FIXTURE GAP -- extend _frame(), do NOT alter the family's logic."
            ) from exc


@pytest.mark.parametrize("name", _XFNS_NAMES)
def test_xfns_survives_duplicate_action_id_gamestate(name):
    _run_family(name)  # MUST NOT raise on the non-unique action_id


@pytest.mark.parametrize("name", _ATOMIC_XFNS_NAMES)
def test_atomic_xfns_survives_duplicate_action_id_gamestate(name):
    """The atomic mirrors were never covered by ADR-020's contract at all -- the gate enumerated
    `dir(F)` over tracking.features only."""
    _run_family(name, AF)
```

Where an atomic factory needs a contract column the shared `_frame()` deliberately omits, add its
name to `_NEEDS_TEAM_IN_POSSESSION` (or add a sibling set) rather than adding the column to
`_frame()` — the comment at lines 69-75 explains why: `derive_team_in_possession` merges without
checking for a pre-existing column, so a frame that already carries it comes back with
`team_in_possession_x`/`_y` and every re-deriving consumer dies on `KeyError`.

- [ ] **Step 5: Resolve every atomic failure**

Each failing atomic factory is a real ADR-020 defect: `set_index("action_id").at[aid]` on a
non-unique index. Route it through `_kernels.resolve_frame_ids_by_position`, the single seam. Do not
special-case a factory to make the gate green.

- [ ] **Step 6: Full-family run**

Run: `python -m pytest tests/tracking/test_frame_aware_xfns_dup_action_id.py -v`
Expected: all pass, atomic cases included.

---

### Task 4: Item 26 — the NaN-safety and leakage-guard pins

**Files:**
- Modify: `tests/test_enrichment_nan_safety.py` (after line 94)
- Modify: `tests/tracking/test_packing_xfns_leakage_guard.py` and its two siblings

**Interfaces:**
- Consumes: nothing. Produces: nothing.

**The red is already measured.** Against the PACKAGE `__all__` (see the spec correction — the module
surface is empty for two of three registries):

    spadl          exported=7   decorated=7   UNDECORATED=0
    atomic.spadl   exported=5   decorated=5   UNDECORATED=0
    tracking       exported=33  decorated=29  UNDECORATED=4
                   add_gradientsports_player_ids, add_sync_score,
                   add_xcross_attempt, add_xshot_occurrence

- [ ] **Step 1: Write the RED two-directional pin**

Append to `tests/test_enrichment_nan_safety.py`:

```python
import silly_kicks.atomic.spadl as _ASP
import silly_kicks.spadl as _SP
import silly_kicks.tracking as _TR

#: Public `add_*` helpers deliberately NOT @nan_safe_enrichment, each with a stated reason.
#: An entry is a decision on the record; an omission is a helper whose NaN-safety is never tested.
_NOT_NAN_SAFE: dict[str, str] = {}

_PIN = (
    ("spadl", _SP, STD_ENRICHMENTS),
    ("atomic.spadl", _ASP, ATOMIC_ENRICHMENTS),
    ("tracking", _TR, TRACKING_ENRICHMENTS),
)


@pytest.mark.parametrize("label,pkg,registry", _PIN, ids=[p[0] for p in _PIN])
def test_every_public_add_star_is_enrolled_or_exempted(label, pkg, registry):
    """ADR-003's registry is auto-discovered from the decorator, so it is complete over DECORATED
    helpers -- but decoration is the human-maintained opt-in and nothing tied it to the public
    surface. The three floors below pass identically whether or not a new public `add_*` was
    decorated. ADR-033 and ADR-051 both pin their surface to the public export in BOTH directions;
    this is ADR-003 catching up.

    Pinned to the PACKAGE export, not the module: `silly_kicks.spadl.utils` has no `__all__` at
    all, so a module-level pin would assert nothing on two of the three registries.
    """
    exported = {n for n in pkg.__all__ if n.startswith("add_")}
    decorated = {fn.__name__ for fn in registry}
    unenrolled = sorted(exported - decorated - set(_NOT_NAN_SAFE))
    assert not unenrolled, (
        f"public add_* in {label}.__all__ with no @nan_safe_enrichment and no exemption: "
        f"{unenrolled}. ADR-003 makes NaN-tolerance a contract for the whole public enrichment "
        f"family; an undecorated helper is never exercised against NaN identifiers."
    )


def test_nan_safe_exemptions_are_real_public_helpers():
    """Self-burning-down."""
    public = set().union(*({n for n in pkg.__all__} for _, pkg, _ in _PIN))
    stale = sorted(set(_NOT_NAN_SAFE) - public)
    assert not stale, f"_NOT_NAN_SAFE names helpers that are not public: {stale}"
```

- [ ] **Step 2: Run and RECORD the red**

Run: `python -m pytest tests/test_enrichment_nan_safety.py -k enrolled -v`
Expected: the `tracking` case FAILS naming the four helpers above. Paste it into the commit message.

- [ ] **Step 3: Adjudicate each of the four, one at a time**

For each, decide **decorate** or **exempt**, and test the decision:

* `add_gradientsports_player_ids` — a roster/jersey helper, not an action enricher. Likely exempt;
  the reason must say why NaN identifiers cannot reach it, not merely that it is unusual.
* `add_sync_score` — TF-6, action-coupled. Likely a genuine gap: decorate and let the existing
  auto-discovered parametrisation exercise it.
* `add_xshot_occurrence`, `add_xcross_attempt` — fitted-model aggregators. Decorating them runs the
  NaN fixture through model inference; if that surfaces a real NaN crash, that is a live defect and
  fixing it is in scope for this task.

Do not batch this. Decorate one, run
`python -m pytest tests/test_enrichment_nan_safety.py -v`, resolve, move on.

- [ ] **Step 4: Single-source the swept default lists, and pin them EXACTLY**

All three of `test_packing_xfns_leakage_guard.py`, `test_run_value_xfns_leakage_guard.py`,
`test_shot_goalmouth_no_xfns_guard.py` carry their **own copy** of `_MODULES` + `_default_lists()`
plus `assert len(lists) >= 10`. Three copies of one discovery rule, guarded by a floor — the same
shape as Task 1, so fix it the same way.

**Measured: the floor is `>= 10` against 19 discovered lists.** The heuristic itself is sound (the
only list it skips, `tracking.features.das_xfns`, is a factory surface and not a default list, so
skipping it is correct).

Create `tests/tracking/_xfn_default_lists.py`:

```python
"""The default xfn lists every leakage guard sweeps, discovered once.

Three guards carried three copies of this discovery rule, each pinned only by `>= 10` against a
real population of 19. A floor cannot detect an omission -- and an omission here means a NEW default
list that no leakage guard sweeps, i.e. a leaky factory could be opted into it with nothing looking.
CLAUDE.md calls that a HybridVAEP-class correctness break.
"""

from __future__ import annotations

import importlib

MODULES = (
    "silly_kicks.tracking.features",
    "silly_kicks.atomic.tracking.features",
    "silly_kicks.vaep",
    "silly_kicks.vaep.base",
    "silly_kicks.atomic.vaep",
    "silly_kicks.atomic.vaep.base",
)


def default_lists() -> dict[str, list]:
    found: dict[str, list] = {}
    for modname in MODULES:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        for attr in dir(mod):
            if "default_xfns" in attr or attr.startswith(("xfns_default", "hybrid_xfns_default")):
                obj = getattr(mod, attr)
                if isinstance(obj, list):
                    found[f"{modname}.{attr}"] = obj
    return found


#: Asserted EXACTLY, both ways. A new default list must be registered here or CI fails; a removed
#: one cannot linger. This is the anti-rot property `>= 10` never had.
SWEPT: frozenset[str] = frozenset(
    {
        "silly_kicks.atomic.tracking.features.atomic_actor_pre_window_default_xfns",
        "silly_kicks.atomic.tracking.features.atomic_pitch_control_default_xfns",
        "silly_kicks.atomic.tracking.features.atomic_pre_shot_gk_angle_default_xfns",
        "silly_kicks.atomic.tracking.features.atomic_pre_shot_gk_default_xfns",
        "silly_kicks.atomic.tracking.features.atomic_pre_shot_gk_full_default_xfns",
        "silly_kicks.atomic.tracking.features.atomic_pressure_default_xfns",
        "silly_kicks.atomic.tracking.features.atomic_tracking_default_xfns",
        "silly_kicks.atomic.vaep.base.xfns_default",
        "silly_kicks.tracking.features.actor_pre_window_default_xfns",
        "silly_kicks.tracking.features.pitch_control_default_xfns",
        "silly_kicks.tracking.features.pre_shot_gk_angle_default_xfns",
        "silly_kicks.tracking.features.pre_shot_gk_default_xfns",
        "silly_kicks.tracking.features.pre_shot_gk_full_default_xfns",
        "silly_kicks.tracking.features.pressure_default_xfns",
        "silly_kicks.tracking.features.tracking_default_xfns",
        "silly_kicks.vaep.base.xfns_default",
        "silly_kicks.vaep.base.xfns_default_no_goalscore",
        "silly_kicks.vaep.hybrid_xfns_default_no_goalscore",
        "silly_kicks.vaep.xfns_default_no_goalscore",
    }
)
```

Create `tests/tracking/test_xfn_default_list_registry.py`:

```python
from tests.tracking._xfn_default_lists import SWEPT, default_lists


def test_the_swept_default_lists_are_EXACT():
    """Fails BOTH ways -- the property `assert len(lists) >= 10` never had against 19."""
    found = set(default_lists())
    assert found == set(SWEPT), (
        f"new and unswept: {sorted(found - SWEPT)}; registered but gone: {sorted(SWEPT - found)}"
    )
```

Then in each of the three guards, delete the local `_MODULES` and `_default_lists`, import the
shared ones, and replace `assert len(lists) >= 10` with `assert set(lists) == set(SWEPT)`. Keep each
guard's two named spot-checks (`tracking_default_xfns`, `vaep.base.xfns_default`) — they are cheap
and they document intent.

- [ ] **Step 4b: Verify the registry can fail**

Temporarily delete one entry from `SWEPT` and run
`python -m pytest tests/tracking/test_xfn_default_list_registry.py -v`.
Expected: FAIL naming it under "new and unswept". Restore it.

- [ ] **Step 5: Full phase-1 run**

Run: `python -m pytest tests/ -m "not e2e" -q > /tmp/cycleb_p1.log 2>&1; echo "EXIT=$?" >> /tmp/cycleb_p1.log`
then read the tail. Expected: 0 failed.

- [ ] **Step 6: Lint, then STOP for commit approval**

Run: `python -m ruff check silly_kicks/ tests/ scripts/ && python -m ruff format --check silly_kicks/ tests/ scripts/ && python -m pyright`

Phase 1 is a complete, independently reviewable unit. Ask for approval before committing.

---

## PHASE 2 — the contract mechanism and the output-side gate

### Task 5: Item 9 — `declare_inputs` and the re-derivation gate

**Files:**
- Create: `scripts/_input_contract.py`
- Modify: `scripts/validate_xshot_causal.py`, `scripts/validate_xcross_causal.py`,
  `scripts/measure_covariate_invariance.py`, `scripts/build_gkdv_arm_values.py`
- Create: `tests/scripts/test_input_contracts.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `scripts._input_contract.declare_inputs(**parts) -> dict` with keys
  `{version, digest, ...declared parts}`; `contract_digest(parts) -> str`; `CONTRACT_VERSION: int`.
  Task 7's driver calls `declare_inputs`. Every call must pass `driver="<script stem>"` so the gate
  in Step 5 can key committed artifacts back to their producer.

- [ ] **Step 1: Write the mechanism**

```python
# scripts/_input_contract.py
"""Declared input contracts for research-artifact drivers (ADR-054).

A driver declares WHICH SYMBOLS its numbers depend on, never what those symbols currently contain.
When SHOT_ARM_CONFOUNDERS gains a column or GEOMETRY_VERSION bumps, the digest moves without anyone
editing the driver. That is the difference between this and "a human writes a list": the residual
failure mode is "forgot to reference a symbol at all" -- narrow and visible -- rather than "typed a
list that later went stale".

Deliberately the same shape as ADR-050's feature_contract, because that pattern is already built,
reviewed and trusted here.

KNOWN LIMIT, declared rather than discovered: this catches code drift, not under-declaration. A
driver that never references `theta` digests stably forever. See spec S1.4 for the two alternatives
considered and the stated trigger for escalating to a runtime coverage check.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

CONTRACT_VERSION = 1


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        # `sorted(value)` alone raises TypeError on mixed-type keys -- measured:
        # `{1: "a", "b": "c"}` -> "'<' not supported between instances of 'str' and 'int'".
        # `covariates` is caller-supplied, so that crash would surface at DRIVER-RUN time,
        # not at gate time. `key=repr` matches what the set branch below already does.
        return {str(k): _canonical(value[k]) for k in sorted(value, key=repr)}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_canonical(v) for v in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def contract_digest(parts: dict) -> str:
    """SHA256 over the canonical JSON of everything except the digest itself."""
    body = {k: _canonical(v) for k, v in sorted(parts.items()) if k != "digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def declare_inputs(**parts: Any) -> dict:
    """Build a driver's input contract. Written into its metrics.json beside run_commit."""
    out: dict[str, Any] = {"version": CONTRACT_VERSION}
    out.update({k: _canonical(v) for k, v in parts.items()})
    out["digest"] = contract_digest(out)
    return out
```

- [ ] **Step 2: Unit-test the mechanism**

```python
# tests/scripts/test_input_contracts.py  (first half)
from scripts._input_contract import contract_digest, declare_inputs


def test_the_digest_moves_when_a_declared_symbol_changes():
    """The whole mechanism. If this does not hold, a stale artifact reads as current."""
    before = declare_inputs(covariates={"arm": ("a", "b")}, geometry_version="goal-relative-2")
    after = declare_inputs(covariates={"arm": ("a", "b", "c")}, geometry_version="goal-relative-2")
    assert before["digest"] != after["digest"]


def test_the_digest_is_stable_across_key_order_and_set_iteration():
    """A digest that moves on dict ordering reports every artifact stale on every run, and the
    warning becomes noise nobody reads."""
    a = declare_inputs(covariates={"x": {"p", "q"}}, geometry_version="goal-relative-2")
    b = declare_inputs(geometry_version="goal-relative-2", covariates={"x": {"q", "p"}})
    assert a["digest"] == b["digest"]


def test_canonical_survives_mixed_type_dict_keys():
    """`covariates` is caller-supplied; a naive `sorted(dict)` raises TypeError on mixed keys and
    the crash lands at driver-run time, not gate time."""
    assert declare_inputs(covariates={1: "a", "b": "c"})["digest"]


def test_the_digest_excludes_itself():
    parts = declare_inputs(covariates={"x": ("a",)})
    assert contract_digest(parts) == parts["digest"]
```

Run: `python -m pytest tests/scripts/test_input_contracts.py -v` — expected PASS.

- [ ] **Step 3: Wire the first driver, by symbol reference**

In `scripts/validate_xshot_causal.py`, add:

```python
def input_contract() -> dict:
    # BOTH imports are function-local by necessity, not style. `SHOT_ARM_CONFOUNDERS` and
    # `shot_arm_config` are NOT module-level names in this script -- they are imported inside
    # `analyze()` at :126. A module-scope reference raises NameError at call time.
    from silly_kicks.causal import SHOT_ARM_CONFOUNDERS, shot_arm_config
    from silly_kicks.tracking import _geometry as _geo

    return declare_inputs(
        driver="validate_xshot_causal",
        covariates={
            "shot_arm": SHOT_ARM_CONFOUNDERS,
            "gk_block": shot_arm_config({}).gk_block,
        },
        geometry_version=_geo.GEOMETRY_VERSION,  # a STRING, "goal-relative-2" -- not an int
        extractors=("silly_kicks.tracking._xshot_occurrence",),
    )
```

**Check SCOPE, not spelling.** An earlier draft told you to verify the names "exist with the
spellings used" — the spellings are exactly right, and an executor who checks only that finds nothing
and ships a `NameError`. What is wrong in the naive version is where the names live. Verified:
`validate_xshot_causal.py:126` imports both inside `analyze()`; `_geo.GEOMETRY_VERSION` does exist at
module scope (`silly_kicks/tracking/_geometry.py:33`) and its value is the string
`"goal-relative-2"`.

Then write `input_contract()` into the artifact beside `run_commit`.

- [ ] **Step 4: Wire the remaining three**

Same shape for `validate_xcross_causal`, `measure_covariate_invariance`, `build_gkdv_arm_values`.
Each declares only the symbols it actually reads.

- [ ] **Step 5: Write the re-derivation gate — WARN, never raise, with a detector that is CALLABLE**

**The naive shape is a gate that cannot fail, and its own docstring claims otherwise.** An earlier
draft put the comparison inline in the test body and pointed at a "planted mismatch" test to carry
non-vacuity. Measured, that test was `test_the_digest_moves_when_a_declared_symbol_changes` with
different literals: it constructs no artifact, never calls `_committed_artifacts`, never reaches the
comparison, and never exercises the warning. A reviewer reimplemented the gate with the comparison
INVERTED and `warnings.warn` DELETED — maximally broken — and both tests still passed.

Two independent reasons it could not work: **(a)** zero committed artifacts carry an
`input_contract` key today (measured: 7 of 7 `docs/research/**/metrics.json`, 0 hits), so the
parametrised body executes zero iterations; **(b)** even after Step 7 populates them, a warn-only
test passes whether or not the warning fires, so nothing observes the comparison. (a) is a window;
**(b) is permanent.**

Root cause: the detector was not a function, so no other test could call it. Extract it.

```python
# tests/scripts/test_input_contracts.py  (second half)
import importlib
import json
import pathlib
import warnings

import pytest

from scripts._input_contract import declare_inputs

_DECLARING = ("validate_xshot_causal", "validate_xcross_causal",
              "measure_covariate_invariance", "build_gkdv_arm_values")

_RESEARCH = pathlib.Path(__file__).resolve().parents[2] / "docs" / "research"


class StaleArtifactWarning(UserWarning):
    """An artifact's declared inputs no longer digest to what live code produces."""


def _artifacts_for(driver: str, root: pathlib.Path) -> list[pathlib.Path]:
    return [
        p
        for p in sorted(root.rglob("metrics.json"))
        if json.loads(p.read_text(encoding="utf-8")).get("input_contract", {}).get("driver") == driver
    ]


def stale_artifacts(driver: str, live: dict, root: pathlib.Path) -> list[tuple[pathlib.Path, str, str]]:
    """THE DETECTOR. A separate callable so a test can exercise it on a controlled artifact.

    Returns (path, recorded_digest, live_digest) for every artifact whose declared inputs no longer
    match live code. Empty list means everything is current.
    """
    out = []
    for path in _artifacts_for(driver, root):
        recorded = json.loads(path.read_text(encoding="utf-8"))["input_contract"]
        if recorded.get("digest") != live["digest"]:
            out.append((path, recorded.get("digest", "?"), live["digest"]))
    return out


@pytest.mark.parametrize("driver", _DECLARING)
def test_committed_artifacts_still_match_their_declared_inputs(driver):
    """WARN, do not raise -- spec S1.2. An artifact is not a serving path, so a mismatch is not a
    load failure; it must surface at PR time rather than at read time.

    This test therefore PASSES on the failure it reports. ALL of its evidential weight sits on
    `test_the_detector_FIRES_on_a_planted_stale_artifact` below, which is the only thing that can
    go red if the comparison breaks.
    """
    live = importlib.import_module(f"scripts.{driver}").input_contract()
    for path, recorded, current in stale_artifacts(driver, live, _RESEARCH):
        warnings.warn(
            f"{path} was produced under a different input contract (recorded {recorded[:12]}, "
            f"live {current[:12]}). Its numbers may be stale.",
            StaleArtifactWarning,
            stacklevel=2,
        )


def test_the_detector_FIRES_on_a_planted_stale_artifact(tmp_path):
    """Non-vacuity, against a REAL artifact this test writes.

    Red if the comparison is inverted, if the digest stops being compared, or if `stale_artifacts`
    returns nothing. Deliberately independent of Step 7 having run, so the gate is meaningful in the
    window between the two commits.
    """
    live = declare_inputs(driver="d", covariates={"arm": ("a", "b")})
    stale = declare_inputs(driver="d", covariates={"arm": ("a",)})
    assert stale["digest"] != live["digest"]  # precondition

    art = tmp_path / "planted" / "metrics.json"
    art.parent.mkdir(parents=True)
    art.write_text(json.dumps({"run_commit": "0" * 40, "input_contract": stale}), encoding="utf-8")

    found = stale_artifacts("d", live, tmp_path)
    assert [p for p, _, _ in found] == [art]


def test_the_detector_is_SILENT_on_a_current_artifact(tmp_path):
    """The other side. A detector that flags everything is as useless as one that flags nothing --
    and it would make the warning noise nobody reads."""
    live = declare_inputs(driver="d", covariates={"arm": ("a", "b")})
    art = tmp_path / "current" / "metrics.json"
    art.parent.mkdir(parents=True)
    art.write_text(json.dumps({"run_commit": "0" * 40, "input_contract": live}), encoding="utf-8")

    assert stale_artifacts("d", live, tmp_path) == []


def test_the_warning_actually_reaches_the_caller(tmp_path):
    """Pins the wiring between the detector and `warnings.warn` -- the one seam the two tests above
    do not cross."""
    live = declare_inputs(driver="d", covariates={"arm": ("a", "b")})
    stale = declare_inputs(driver="d", covariates={"arm": ("a",)})
    art = tmp_path / "planted" / "metrics.json"
    art.parent.mkdir(parents=True)
    art.write_text(json.dumps({"input_contract": stale}), encoding="utf-8")

    with pytest.warns(StaleArtifactWarning):
        for path, recorded, current in stale_artifacts("d", live, tmp_path):
            warnings.warn(f"{path} {recorded[:12]} {current[:12]}", StaleArtifactWarning, stacklevel=2)
```

Every `declare_inputs(...)` call in Steps 3 and 4 must pass `driver="<script stem>"` so
`_artifacts_for` can key an artifact back to its producer.

**Note the `root` parameter on `stale_artifacts`.** It exists so the non-vacuity tests can point the
detector at `tmp_path`. Without it the detector could only ever read the real `docs/research` tree,
which is exactly what made the previous version untestable.

on it.

- [ ] **Step 6: Verify the drivers still import and `--help` cleanly**

Run: `python -m scripts.validate_xshot_causal --help` (and the other three). Expected: usage text, no
traceback, no cp1252 encode error. **`scripts/` is ASCII-only** — a `—` in a help string fails here.

- [ ] **Step 7: Regenerate the four artifacts — ALL FIVE RUNS ON ONE CLEAN TREE**

Each driver must re-run to write its `input_contract`, and cannot run on the dirty tree that
introduced it (`scripts/_provenance.py:73-76`) — so these run **after** the phase-2 commit.

**The trap, and it bites on run number two.** `git_provenance` calls `git status --porcelain` and
treats **any** entry as dirty, untracked included (`:73-76`, deliberately). Driver 1's freshly
written `metrics.json` is exactly such an entry, so driver 2 would refuse. Run them naively and you
get either five separate commits, or four artifacts stamped `dirty: true` — and `--allow-dirty` is a
dev hatch, never a way to bundle.

**Recipe: write outside the repository, then copy in.** Verified — all four drivers accept `--out`,
and every shard-writing driver derives its shard root FROM that path
(`validate_xshot_causal.py:271` and `build_gkdv_arm_values.py:270` use `dest / "shards"`,
`validate_xcross_causal.py:172` uses `Path(out) / "shards"`), so nothing lands inside the repo:

```bash
STAGE=/d/Development/_cycleb_artifacts     # OUTSIDE the repo -- keeps `git status` empty
mkdir -p "$STAGE"
python -m scripts.validate_xshot_causal        --out "$STAGE/xshot_causal"
python -m scripts.validate_xcross_causal       --out "$STAGE/xcross_causal"
python -m scripts.measure_covariate_invariance --out "$STAGE/covariate_invariance"
python -m scripts.build_gkdv_arm_values        --out "$STAGE/gkdv_arm_values"
# Task 7 Step 5's GS measurement runs here too, on the same clean tree.
```

Confirm `git status --porcelain` is EMPTY before each run. Then copy the five artifacts into
`docs/research/` and commit them together.

**All five will stamp the same `run_commit`** — the phase-2 commit's SHA. That is correct and
desirable: they were produced from one tree state, and the identical SHA is the evidence.

---

### Task 6: K9 — the output-side artifact gate

**Files:**
- Create: `tests/scripts/test_artifact_provenance_output.py`
- Modify: `silly_kicks/tracking/_xshot_weights/default/metadata.json` + its `SHA256SUMS`
- Modify: `silly_kicks/tracking/_xcross_weights/default/metadata.json` + its `SHA256SUMS`

**Interfaces:**
- Consumes: nothing at build time. The gate asserts `run_commit` / `run_tree_dirty` only; the spec's
  "and -- once declared -- an `input_contract` digest" is deliberately NOT asserted here, because
  Task 5's artifacts are only regenerated in a follow-up commit (Task 5 Step 7) and a gate that
  demands a key nothing has written yet lands red for a reason unrelated to K9.

**Measured constraints — read before editing anything:**

    _xshot_weights/default/SHA256SUMS   covers model.json AND metadata.json
    _xcross_weights/default/SHA256SUMS  covers model.json AND metadata.json
    both loaders VERIFY the sums and raise IntegrityError on mismatch
        (_xshot_occurrence.py:523-534, _xcross_attempt.py:593-604)
    sibling metrics.json already carries run_commit 6e3a132..., run_tree_dirty false
    metrics.json is NOT covered by SHA256SUMS

So the repair is: copy the already-correct commit from the sibling `metrics.json` into
`metadata.json` as `training_commit`, then **regenerate SHA256SUMS** — or every model load breaks.
This is a data edit with a load-time consequence; do not skip Step 5.

- [ ] **Step 0: Widen the glob -- MEASURED, the drafted one is too narrow**

Found by the commit-1 final review, before the gate was written. `docs/research/**/metrics.json`
matches **7** files. Under the same tree there are **11 more JSON artifacts it does not see**:

    gate.json  agreement.json  comparability_report.json  invalidation.json
    prefix_measurement.json  postfix_measurement.json  manifest_skillcorner_full.json
    metrics_rerun_clean_provenance.json  aarch64-linux-py3.12.json
    amd64-windows-py3.14.json

**Three of the drivers commit 1 enrolled write outside the drafted glob**, which is the source/output
asymmetry K9 exists to close, re-created inside K9 itself:

    validate_xtgk_possession_value -> docs/research/xtgk_possession_value/gate.json  (wrong NAME)
    validate_shot_goalmouth_sb     -> shot_goalmouth_sb_report.json                  (outside docs/research)
    calibrate_tracking_defaults    -> calibration_report.json                        (outside docs/research)

Decide explicitly, and record the decision:

* Widen to `docs/research/**/*.json` and add an exemption registry for the files that are inputs
  rather than results (`invalidation.json` is an annotation, the platform probes are fingerprints).
* And/or require enrolled drivers to write a `metrics.json`, making the convention the contract.

Do NOT ship the narrow glob and call the gate complete: it would pass while ignoring 11 of 18
artifacts and all three newly-enrolled drivers.

- [ ] **Step 1: Write the RED gate**

```python
# tests/scripts/test_artifact_provenance_output.py
"""The existing provenance gate reads driver SOURCE; this one reads artifact OUTPUT.

A driver can satisfy every assertion in tests/scripts/test_provenance_wiring.py -- imports the
helper, offers --allow-dirty, calls require_clean_tree from main() -- and still emit an artifact
nobody can trace. K9 exists because only the source half was ever built.

WHICH FIELD carries training provenance had two answers on the tree when this was written:
`_ghost_gk_weights/metadata.json` used `training_commit`; `_xshot_weights` and `_xcross_weights`
used NO commit key at all while their sibling metrics.json carried a correct `run_commit`. An
ABSENT key is worse than a null one -- a null is something a reader can notice. This gate picks
`training_commit` for bundled weights (the incumbent) and `run_commit` for research artifacts, and
applies each uniformly.
"""

from __future__ import annotations

import json
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Committed artifacts deliberately without provenance, each with a stated reason.
_UNPROVENANCED: dict[str, str] = {}


def _research_artifacts() -> list[pathlib.Path]:
    return sorted((_ROOT / "docs" / "research").rglob("metrics.json"))


def _bundled_metadata() -> list[pathlib.Path]:
    return sorted(_ROOT.glob("silly_kicks/**/_*_weights/*/metadata.json"))


@pytest.mark.parametrize("path", _research_artifacts(), ids=lambda p: str(p.relative_to(_ROOT)))
def test_research_artifacts_carry_run_provenance(path):
    key = str(path.relative_to(_ROOT)).replace("\\", "/")
    if key in _UNPROVENANCED:
        pytest.skip(_UNPROVENANCED[key])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("run_commit"), f"{key} carries no run_commit -- its numbers cannot be traced"
    assert data.get("run_tree_dirty") is False, (
        f"{key} was produced from a dirty tree, so its run_commit does not describe the code that ran"
    )


@pytest.mark.parametrize("path", _bundled_metadata(), ids=lambda p: str(p.relative_to(_ROOT)))
def test_bundled_weights_carry_training_provenance(path):
    key = str(path.relative_to(_ROOT)).replace("\\", "/")
    if key in _UNPROVENANCED:
        pytest.skip(_UNPROVENANCED[key])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("training_commit"), (
        f"{key} carries no training_commit. A bundled artifact nobody can trace back to a commit "
        f"cannot be reproduced or audited -- and an ABSENT key is worse than a null one, because a "
        f"null is something a reader can notice."
    )


def test_the_artifact_populations_are_not_silently_empty():
    """Meta-assertion: a parametrised gate over an empty glob passes vacuously."""
    assert len(_research_artifacts()) >= 5
    assert len(_bundled_metadata()) >= 3


def test_unprovenanced_exemptions_name_files_that_exist():
    stale = sorted(k for k in _UNPROVENANCED if not (_ROOT / k).is_file())
    assert not stale, f"_UNPROVENANCED names files that do not exist: {stale}"
```

- [ ] **Step 2: Run and RECORD the red**

Run: `python -m pytest tests/scripts/test_artifact_provenance_output.py -v`

Expected, and this is the acceptance criterion stated in §6: the research half passes **7 of 7**
(green by construction — K3's offender was repaired in 4.74.0), and the bundled half FAILS on
`_xshot_weights/default/metadata.json` and `_xcross_weights/default/metadata.json`. Record both
halves. A green half is a real signal, not RED-first evidence — say so rather than manufacturing a
failure.

- [ ] **Step 3: Read the correct commit from each sibling `metrics.json`**

```bash
python -c "
import json, glob
for p in sorted(glob.glob('silly_kicks/**/_x*_weights/*/metrics.json', recursive=True)):
    d = json.load(open(p))
    print(p, d['run_commit'], d['run_tree_dirty'])
"
```

Expected: both `6e3a132b0e75da425bec7affede02218150f1232`, `False`. **If `run_tree_dirty` is True for
either, STOP** — do not copy a dirty commit into a bundled artifact; that is the exact false-provenance
pattern the rule exists to eliminate. Report it instead.

- [ ] **Step 4: Stamp `training_commit` into both metadata files**

**The byte recipe is measured, not guessed** — verified byte-identical on all three bundled
`metadata.json` files. An earlier draft of this step said "check the existing formatting first and
match it" and then shipped `json.dumps(meta, indent=2, sort_keys=True) + "
"`, which is wrong on
BOTH counts and would silently reformat the whole file, bake that reformat into the regenerated
sums, and bury the one-line change in a whole-file diff.

    trailing newline      NO    (raw.endswith(b"
") is False)
    key-sorted            NO    (insertion order, not alphabetical)
    CRLF                  NONE
    json.dumps(obj, indent=2)                    -> byte-identical  YES
    json.dumps(obj, indent=2) + "
"             -> byte-identical  NO
    json.dumps(obj, indent=2, sort_keys=True)    -> byte-identical  NO

```bash
python -c "
import json, pathlib
for w in ('_xshot_weights', '_xcross_weights'):
    d = pathlib.Path('silly_kicks/tracking') / w / 'default'
    metrics = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
    assert metrics['run_tree_dirty'] is False, d
    meta_path = d / 'metadata.json'
    raw = meta_path.read_bytes()
    meta = json.loads(raw.decode('utf-8'))
    assert 'training_commit' not in meta, meta_path
    # Prove the recipe reproduces the file before trusting it to rewrite the file.
    assert json.dumps(meta, indent=2).encode() == raw, f'recipe does not round-trip {meta_path}'
    meta['training_commit'] = metrics['run_commit']
    meta_path.write_bytes(json.dumps(meta, indent=2).encode())
    print('stamped', meta_path)
"
```

The round-trip assertion is the load-bearing line: it fails BEFORE any write if the file's
formatting ever differs from this recipe, rather than reformatting it and telling you afterwards.

- [ ] **Step 5: Regenerate both `SHA256SUMS`**

Measured format: **LF only** (zero CRLF, on Windows too), **two spaces** between digest and name,
**trailing newline present**, `model.json` first then `metadata.json`. 157 bytes for xshot and
xcross, 164 for ghost.

```bash
python -c "
import hashlib, pathlib
for w in ('_xshot_weights', '_xcross_weights'):
    d = pathlib.Path('silly_kicks/tracking') / w / 'default'
    body = ''.join(
        f'{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}' + chr(10)
        for n in ('model.json', 'metadata.json')
    )
    (d / 'SHA256SUMS').write_bytes(body.encode())
    print('rehashed', d, len(body), 'bytes')
"
```

`write_bytes` rather than `write_text`, so Windows cannot translate LF to CRLF — the loaders write
theirs with `open(..., newline="
")` (`_xshot_occurrence.py:503`), and a CRLF file would fail
every integrity check.

**Do not skip the ghost directory by accident and do not include it here.** Ghost already carries
`training_commit`; its `metadata.json` is untouched by this task, so its sums must not be
regenerated.

**Where this repair belongs long-term:** `scripts/stamp_feature_contracts` already owns
metadata-only rewrites and its docstring already explains why it must not call any model's `save()`.
Task 1 Step 4b classifies it as NOT an artifact driver while its OUTPUT is policed here, which is
consistent. Folding this stamp into it is a reasonable follow-up; doing it inline here is fine for a
two-file one-time repair, and the round-trip assertion above is what makes that safe.

- [ ] **Step 6: Prove the models still load**

Run:

```bash
python -c "
import warnings; warnings.filterwarnings('ignore')
from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel
from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel
XShotOccurrenceModel.load_bundled(); XCrossAttemptModel.load_bundled()
print('both bundled models load OK')
"
```

Use whatever the real bundled-load entry point is called in each module — check before running.
Expected: no `IntegrityError`, no chirality or feature-contract warning that was not there before.

- [ ] **Step 7: Re-run the gate and the model test suites**

Run: `python -m pytest tests/scripts/test_artifact_provenance_output.py tests/tracking/test_xshot_occurrence_integration.py tests/tracking/test_xcross_attempt_integration.py -v`
Expected: all pass.

- [ ] **Step 8: Full suite, lint, STOP for commit approval**

Run the full suite to a unique log with an exit marker, then the three lint commands.

---

## PHASE 3 — item 23, the GS input-convention guard

### Task 7: Step 2a — measure the real GS shot distribution

**Files:**
- Create: `scripts/measure_gs_shot_distribution.py`
- Create (by running it): `docs/research/gs_input_convention/metrics.json`

**Interfaces:**
- Consumes: `scripts/_provenance.py`, `scripts/_driver.py`, `scripts/_input_contract.py` (Task 5).

- [ ] **Step 1: Write the driver**

Requirements, each load-bearing:

* `--out` (not `--report-out`, not `--output-dir`) — §5 of the spec: anything else lands it straight
  in `_UNDERIVABLE` and breaks Task 1's empty-bucket assertion.
* `--allow-dirty`, and `require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)` called
  **from `main()`**, before any corpus work. Task 1's gate checks all three.
* `for_each` from `scripts/_driver.py` with per-match shards (ADR-052).
* `input_contract()` per Task 5.
* **Only summary counts travel, never coordinates.** GS is owner-tier. The output carries
  per-`(match, team, period)` shot COUNTS and the group-reliability tally — nothing reconstructible.
* ASCII-only source.

- [ ] **Step 2: Verify `--help` without running `main()`**

Run: `python -m scripts.measure_gs_shot_distribution --help`
Expected: usage text. A script with no argparse would ignore `--help` and execute — verify argparse
is wired before this command is ever typed.

- [ ] **Step 3: Confirm Task 1's gate now covers it**

Run: `python -m pytest tests/scripts/test_provenance_wiring.py -v`
Expected: the new driver appears in the parametrised cases and passes all five source assertions.
This is the self-test named in §5 — the cycle's own gate catching the cycle's own new driver.

- [ ] **Step 4: STOP — commit before running**

The driver is untracked, so `require_clean_tree` will `SystemExit` before doing any corpus work.
Ask for commit approval. Only after the commit does Step 5 become executable.

- [ ] **Step 5: Run it on owner-tier GS — on the same clean tree as Task 5 Step 7**

This run is subject to the same untracked-file trap: use `--out` pointing OUTSIDE the repository and
batch it with Task 5 Step 7's four runs, before copying any artifact in. See that step for the
recipe and the reason.

Run the driver against the real corpus. Record `n_matches`, and the per-`(match, team, period)`
distribution summary.

- [ ] **Step 6: Verify the artifact passes Task 6's gate**

Run: `python -m pytest tests/scripts/test_artifact_provenance_output.py -v`
Expected: the new `docs/research/gs_input_convention/metrics.json` is picked up and passes.

---

### Task 8: Steps 1, 2b, 3, 4 — reshape, diagnose, plant, exempt

**Files:**
- Modify: `tests/datasets/gradientsports/_generate_synthetic_match.py`
- Modify: `tests/datasets/gradientsports/synthetic_match.json` (regenerated)
- Modify: `silly_kicks/spadl/orientation.py` **only if 2b's diagnosis says so**

- [ ] **Step 1: Reshape the fixture to 2a's numbers**

**The binding constraint is measured and is not per-group shot count.** The committed fixture has 10
shots in `(team 100, period 1)` — AT the `high` threshold — and **only one team has shots at all**,
so it defers on the fewer-than-two-reliable-groups clause (`orientation.py:292`, `:320-321`). Raising
per-group counts would not have made CI see the case. **Give a second team or period a reliable
group** (>= `min_shots_per_group_high` = 10, or >= `min_shots_per_group_medium` = 5 for the medium
tier).

Shape it to 2a's measured distribution. Where 2a's numbers sit near a band boundary and
underdetermine the choice, **state the choice and its reason in the generator** rather than picking
silently (spec Risk 3).

- [ ] **Step 2: Regenerate and confirm CI can now see the case**

Run the generator, then:

```bash
SILLY_KICKS_ASSERT_INVARIANTS=1 python -m pytest tests/spadl/ -k gradientsports -v
```

Expected: the detector now classifies instead of deferring, and the declared-vs-detected disagreement
becomes visible. Record what it says.

- [ ] **Step 3: Diagnose, and let the diagnosis choose the side**

K6 established by field measurement that the OUTPUT is correct: action vs re-projected frame ball,
median `|dy|` 2.75 / 2.79 m across 2,742 linked actions, no period/flip/home-away split, against a
calibrated ~0.2 m (correct) and ~11.8 m (y-inverted). So a detected disagreement is either a detector
false positive or a mis-declared convention — determine which from the fixture, then fix that side.

**Do not repair the symptom by weakening the detector.** That converts a working guard into a
decorative one, which is how a real disagreement would later be lost.

- [ ] **Step 4: Plant a genuinely mis-declared provider and observe the guard fire**

```python
def test_the_guard_FIRES_on_a_genuinely_mis_declared_convention():
    """Non-vacuity. Until 4.76.0 the GS fixture deferred on the fewer-than-two-reliable-groups
    clause, so the guard's raise path was never exercised by CI at all -- `on_mismatch=None`
    resolves to "raise" under SILLY_KICKS_ASSERT_INVARIANTS=1, which ci.yml:58 sets."""
```

Build a fixture whose declared convention is genuinely wrong and assert the guard raises.

- [ ] **Step 5: Any GS exemption goes in a registry with a reason and its own test**

Never `filterwarnings`, never a per-provider `silent`.

- [ ] **Step 6: Full suite with invariants on**

Run both: the normal suite, and `SILLY_KICKS_ASSERT_INVARIANTS=1 python -m pytest tests/ -m "not e2e" -q`.
Expected: 0 failed in both.

---

### Task 9: ADR-054, docs, C4, version

**Files:**
- Create: `docs/superpowers/adrs/ADR-054-artifact-input-contracts-and-registry-completeness.md`
- Modify: `docs/superpowers/specs/2026-08-05-cycle-b-artifact-input-contracts-design.md`
- Modify: `CLAUDE.md`, `CHANGELOG.md`, `TODO.md`, `pyproject.toml`, `docs/c4/architecture.dsl`

- [ ] **Step 1: Fix the spec's three wrong prescriptions**

Apply the three corrections recorded at the top of this plan to §3.1, §3.3 and §3.4, each with the
measurement that overturned it. Per spec Risk 4, a discovery outside §Scope is recorded as a failure
of that document rather than absorbed silently.

- [ ] **Step 2: Write ADR-054**

Include, per gate, the RED output recorded during the build. Name explicitly which gates had a LIVE
red (10, K2, 26, K9-bundled-half) and which had a PLANTED one (24) or landed green by construction
(K9-research-half). Record the four item-26 adjudications and the reasoning for each.

- [ ] **Step 3: Add the durable CLAUDE.md bullet**

One bullet, in the Key-conventions voice: derive the population, justify the exclusions; a floor
cannot detect an omission; the three-bucket shape with `_UNDERIVABLE` asserted empty; and the
structural reconciliation (single-source the universe, do not assert a tautological set relation).
CLAUDE.md was at 83,217 chars against a 150k limit — headroom is fine.

- [ ] **Step 4: Regenerate the C4 diagram**

Follow the `c4` skill: `docs/c4/architecture.dsl` -> `architecture.html`. No new subpackage ships in
this cycle, so the container list should be unchanged; confirm rather than assume, and confirm the
aggregator count still reads what Task 2 pins.

- [ ] **Step 5: Version and CHANGELOG**

Read the version off `main` at this moment — do not assume 4.76.0. Add the CHANGELOG entry keyed by
its ADR and PR number, noting that no retrain is triggered (all changes are gates, plus two
metadata provenance stamps that leave weights untouched).

- [ ] **Step 6: `/final-review`, then STOP for commit approval**

- [ ] **Step 7: PR, watch CI, merge**

`--merge`, **never squash** — this PR's artifacts both stamp and cite commit SHAs.
