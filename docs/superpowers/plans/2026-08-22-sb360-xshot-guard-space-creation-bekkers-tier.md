# SB360 xShot guard + space_creation unlock + bekkers_pi tier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `add_xshot_occurrence` fabricating on velocity-less SB360 freeze-frames, make the ADR-054 velocity-availability contract self-enforcing via a static gate, soften `add_space_creation`'s opponent-perspective raise to per-row NaN with a provenance column, and tier `bekkers_pi` pressing-intensity to honest-NaN on velocity-unavailable frames.

**Architecture:** Four independent changes to `silly_kicks/tracking/`, plus their CI-gate wiring and SB360-audit updates. Three are edge-seam guards keyed on the existing `velocity_unavailable_by_design(frames)` marker (declared-unavailable → honest degrade; undeclared-missing-velocity → loud raise); one is a static AST test that enforces the contract. No fitted model is re-fit; no VAEP retrain. See the design spec: `docs/superpowers/specs/2026-08-21-sb360-xshot-guard-space-creation-unlock-design.md`.

**Tech Stack:** Python 3.10–3.13, pandas (2.x + 3.x span, ADR-057), numpy≥2, pytest. Pure pandas-in/pandas-out; no new runtime dependency.

## Global Constraints

Every task's requirements implicitly include this section. Exact values are copied verbatim from the spec / codebase.

- **No fitted model is re-fit; no VAEP retrain.** All four changes are guards/tiers on the SB360 (velocity-less) path only. Full-tracking output stays byte-identical.
- **Velocity-availability contract (ADR-054/063).** The marker is `silly_kicks.tracking._velocity_availability.velocity_unavailable_by_design(frames)` — `True` iff EVERY row has `speed_source == SPEED_SOURCE_UNAVAILABLE`; `False` when `speed_source` is absent or frames is empty. Two-prong response: DECLARED-unavailable → honest degrade (NaN); UNDECLARED + missing `vx`/`vy` → `raise ValueError` naming `derive_velocities()`. Never swallow the raise into an all-NaN column (ADR-043).
- **Policy at the edge, engine pure.** Guards live at the public compute/serve seam, never inside a `compute_*` kernel or `predict_*`.
- **id comparisons via `silly_kicks.id_compat`** (`ids_match`/`ids_equal`/`same_id`), never raw `==` on identifier columns (ADR-019).
- **`add_*` enrichers are PURE** (ADR-033): return a NEW object, never mutate a caller-supplied DataFrame/Series/ndarray. Any conditionally-added column registers ≥2 purity variants (present + absent branch).
- **`add_*` enrichers are NaN-safe** (ADR-003): `@nan_safe_enrichment`-decorated; NaN identifier inputs route to the documented default, never crash.
- **All `warnings.warn(...)` calls include `stacklevel=2`.**
- **SB360 audit (ADR-053): the machine observation is TRANSCRIBED FROM EXECUTION and CI-locked; only a human writes an adjudication + rationale.** Never assert a verdict you did not observe on regeneration. Regenerating the registry is NOT idempotent — back up `tests/sb360/_entries/` first and diff.
- **A new tracking output column must clear its meta-gates or CI fails:** `feature_glossary` coverage (ADR-048), `TRACKING_CATEGORICAL_DOMAINS` (if categorical), aggregator-column-liveness (`tests/tracking/test_aggregator_column_liveness.py`, non-null on the fixture), ADR-033 purity, and the SB360 registry-surface meta-assertion.
- **Every band/guard test asserts BOTH sides** (a mutation that SHOULD move the value out of the safe state), and every counterfactual asserts it measurably differs from its twin (CLAUDE.md "Every band needs a test from BOTH sides").
- **Lint at CI scope:** `python -m ruff check silly_kicks/ tests/` + `python -m ruff format --check silly_kicks/ tests/`; `python -m pyright` (bare). Test: `python -m pytest tests/ -m "not e2e" -v --tb=short`.
- **NEVER write a commit step.** The user commits once, at the end, on their own explicit approval. Docs live in the first commit of the branch (provenance-untracked-is-dirty trap), but that is a commit-time concern, not a plan step.

---

## File Structure

**Production changes**
- `silly_kicks/tracking/_xshot_occurrence.py` — add the two-prong velocity guard in `compute_xshot_occurrence` (Task 1).
- `silly_kicks/tracking/_space_creation.py` — `_resolve_opponent_team_id` gains `on_unresolvable`; `compute_space_created` marker-gates it and emits `space_opponent_source` (Task 3).
- `silly_kicks/tracking/features.py` — `pressure_on_actor`'s `bekkers_pi` branch: replace the unconditional vx/vy raise with the two-prong seam (Task 4); thread `space_opponent_source` through `add_space_creation`'s per-action assembly if it does not already flow from `compute_space_created` (Task 3).
- `silly_kicks/tracking/schema.py` — add `space_opponent_source` to `TRACKING_CATEGORICAL_DOMAINS` (Task 3).
- `silly_kicks/feature_glossary.py` — `FeatureColumn` entry for `space_opponent_source` IF provenance columns are glossary-covered (Task 3, verify first).

**Test / gate changes**
- `tests/tracking/test_xshot_velocity_guard.py` — NEW, xShot guard TDD (Task 1).
- `tests/tracking/test_velocity_feature_contract.py` — NEW, the static gate (Task 2).
- `tests/tracking/test_space_creation_opponent_softening.py` — NEW, softening TDD (Task 3).
- `tests/tracking/test_bekkers_velocity_tier.py` — NEW, bekkers_pi tier TDD (Task 4).
- `tests/test_add_star_purity.py` — add a purity variant for `add_space_creation` with `space_opponent_source` (Task 3).
- `tests/tracking/test_aggregator_column_liveness.py` — update the `add_space_creation` registration to exercise `include_opponent_perspective=True` (Task 3).
- `tests/sb360/_entries/_gk.py` — xShot rationale cross-reference (Task 1).
- `tests/sb360/_entries/_space.py` — add `space_opponent_source` column verdict (Task 3).
- `tests/sb360/_entries/_context.py` + `scripts/_sb_battery.py` (+ re-export `tests/sb360/_calls.py`) — add `pressure_on_actor__bekkers_pi` column + a methods-passing adapter (Task 4).

**Docs**
- `docs/superpowers/adrs/ADR-054-*.md` (amend) + `docs/superpowers/adrs/ADR-063-*.md` (amend) or a new ADR (number assigned at commit); `CHANGELOG.md`; `CLAUDE.md` (contract bullets); `docs/PRIVATE_CONSUMERS.md`; version bump (sites listed in Task 5) — **no version/PR/ADR number is claimed until commit** (Task 5).

---

## Task 1: xShot velocity-fabrication guard

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py` (imports near top; guard after the `out["xshot_occurrence"] = np.nan` init at `:852`)
- Modify: `tests/sb360/_entries/_gk.py:700-705` (rationale cross-reference only — no observation change)
- Test: `tests/tracking/test_xshot_velocity_guard.py` (create)

**Interfaces:**
- Consumes: `velocity_unavailable_by_design`, `SPEED_SOURCE_UNAVAILABLE` from `silly_kicks.tracking._velocity_availability` / `silly_kicks.tracking.schema` (match the import lines `_xcross_attempt.py` uses).
- Produces: `compute_xshot_occurrence(frames, ...)` returns all-NaN `xshot_occurrence` on declared-velocity-unavailable frames; raises `ValueError` on undeclared missing vx/vy; unchanged on velocity-bearing frames. `add_xshot_occurrence` and `xshot_occurrence_xfns` inherit it (shared seam).

The guard mirrors `_xcross_attempt.py:842-856` verbatim, adapted (`xcross_attempt`→`xshot_occurrence`, `ball_speed`→`speed`, `compute_xcross_attempt`→`compute_xshot_occurrence`).

- [ ] **Step 1: Write the failing tests.**

```python
# tests/tracking/test_xshot_velocity_guard.py
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._xshot_occurrence import compute_xshot_occurrence
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE


def _frames(*, velocity: bool, marker: bool) -> pd.DataFrame:
    """One in-possession frame with two teams + a ball, near the attacked goal.
    velocity=True adds vx/vy; marker=True stamps speed_source=unavailable on every row."""
    rows = [
        # actor (attacking, team 1), a defender (team 2), the ball
        {"game_id": 1, "period_id": 1, "frame_id": 10, "time_seconds": 5.0,
         "player_id": 11, "team_id": 1, "x": 90.0, "y": 34.0, "is_ball": False, "is_goalkeeper": False},
        {"game_id": 1, "period_id": 1, "frame_id": 10, "time_seconds": 5.0,
         "player_id": 21, "team_id": 2, "x": 95.0, "y": 34.0, "is_ball": False, "is_goalkeeper": True},
        {"game_id": 1, "period_id": 1, "frame_id": 10, "time_seconds": 5.0,
         "player_id": pd.NA, "team_id": pd.NA, "x": 90.0, "y": 34.0, "is_ball": True, "is_goalkeeper": False},
    ]
    df = pd.DataFrame(rows)
    df["team_attacking_direction"] = "ltr"
    df["ball_state"] = "alive"
    df["speed"] = 0.0 if velocity else np.nan
    if velocity:
        df["vx"] = 0.0
        df["vy"] = 0.0
    df["speed_source"] = SPEED_SOURCE_UNAVAILABLE if marker else "derived"
    return df


def test_declared_velocity_unavailable_returns_nan_not_fabricated():
    frames = _frames(velocity=False, marker=True)  # SB360 shape: no vx/vy, marker set
    out = compute_xshot_occurrence(frames)  # bundled default via _resolve_model(None)
    assert out["xshot_occurrence"].isna().all()


def test_undeclared_missing_velocity_raises_naming_remedy():
    frames = _frames(velocity=False, marker=False)  # forgot derive_velocities()
    with pytest.raises(ValueError, match="derive_velocities"):
        compute_xshot_occurrence(frames)


def test_velocity_bearing_frame_is_scored_not_nan():
    # NON-VACUITY for the declared-NaN test: prove NaN above is SUPPRESSION, not "nothing scores".
    # Reuse a fixture KNOWN to score in the existing xShot happy-path test rather than a hand-built
    # minimal frame -- a minimal frame can NaN for unrelated reasons (possession/goal resolution),
    # which would look like the guard over-firing and make this assertion vacuous. Import/parametrize
    # off the scoring fixture in tests/tracking/test_xshot_occurrence.py (find the exact builder
    # during implementation; it is the frame set that test asserts produces a finite xshot_occurrence).
    frames = _known_scoring_xshot_frames()
    out = compute_xshot_occurrence(frames)
    assert out["xshot_occurrence"].notna().any()
```

Notes: (1) `compute_xshot_occurrence(frames)` resolves the bundled `XShotOccurrenceModel` default via `_resolve_model(None)`; if loading the bundled artifact is heavy, pass a tiny fitted/stub model or `importorskip("xgboost")`, keeping the three assertions. (2) `_known_scoring_xshot_frames` is NOT hand-built here — locate the existing happy-path scoring fixture during implementation and reuse it, so the non-vacuity assertion cannot fail for a fixture-inadequacy reason (the recurring "a gate is only as good as the rows it scores" trap).

- [ ] **Step 2: Run the tests, verify they fail.**

Run: `python -m pytest tests/tracking/test_xshot_velocity_guard.py -v`
Expected: `test_declared_..._returns_nan` FAILS (currently xShot fabricates a number on the marker frame); `test_undeclared_..._raises` FAILS (currently no raise / different error); `test_velocity_bearing...` may already pass.

- [ ] **Step 3: Add the imports** (match `_xcross_attempt.py`'s import lines — verify their exact form first).

In `silly_kicks/tracking/_xshot_occurrence.py`, near the other `from .schema` / `from ._velocity_availability` imports:

```python
from ._velocity_availability import velocity_unavailable_by_design as _velocity_unavailable_by_design
from .schema import SPEED_SOURCE_UNAVAILABLE
```

- [ ] **Step 4: Insert the guard** immediately after `out["xshot_occurrence"] = np.nan` (`:852`):

```python
    # VELOCITY-AVAILABILITY CONTRACT (ADR-054), at the SHARED seam. All three public entry points
    # reach scoring through here -- `add_xshot_occurrence`, `xshot_occurrence_xfns` and a direct call.
    if _velocity_unavailable_by_design(frames):
        # DECLARED unavailable (the SB360 freeze-frame shape): degrade to NaN. `speed` is a trained
        # feature, so scoring here would have the model impute an input its source structurally
        # cannot carry -- the ADR-053 fabrication shape. NaN is honest, and it is already what this
        # function returns on every other unscoreable path below.
        return out
    if len(frames) and ("vx" not in frames.columns or "vy" not in frames.columns):
        # NOT declared: the "forgot derive_velocities()" case. Fail loud, and name the remedy.
        raise ValueError(
            "compute_xshot_occurrence requires vx/vy on frames (call derive_velocities() first), or "
            f"declare speed_source {SPEED_SOURCE_UNAVAILABLE!r}. See the velocity-availability contract."
        )
```

- [ ] **Step 5: Run the tests, verify they pass.**

Run: `python -m pytest tests/tracking/test_xshot_velocity_guard.py -v`
Expected: all PASS.

- [ ] **Step 6: Add the SB360-audit rationale cross-reference** (no observation change — the fixture has no shot context, so xShot stays `("no_signal","not_exercised")`). In `tests/sb360/_entries/_gk.py:700-705` (and the three `visibility` roster keys `:712-715`, `:723-726`, `:735-737`), append to the existing rationale string a pointer to the static gate, e.g.:

```
"...widening the fixture would move it. [measured cause=velocity+frame_count] "
"NOTE: the velocity-fabrication guard is enforced structurally by the static "
"velocity-feature contract gate (tests/tracking/test_velocity_feature_contract.py), "
"NOT by this fixture -- a not_exercised model is invisible to a runtime audit; fixture "
"widening was deliberately declined in favour of the systemic gate (spec Part 1b)."
```

- [ ] **Step 7: Run the SB360 audit + full non-e2e suite for this area.**

Run: `python -m pytest tests/sb360/ tests/tracking/test_xshot_velocity_guard.py -m "not e2e" -v --tb=short`
Expected: PASS (the rationale edit does not change the machine observation, which the audit re-derives and locks).

---

## Task 2: static velocity-feature contract gate (Part 1b)

**Files:**
- Test: `tests/tracking/test_velocity_feature_contract.py` (create)

**Interfaces:**
- Consumes: the five bundled fitted models' serve entries + feature constants (see the registry below).
- Produces: a CI gate that fails if any frame-consuming model-scoring entry with a velocity feature does not reference `_velocity_unavailable_by_design`, and that catches a NEW frame-served fitted model regardless of feature-naming convention.

This gate is written AFTER Task 1, so it is GREEN at task end (xShot is now guarded). Step 2 documents the one-time red-first observation (revert Task 1's guard, watch the gate go red, restore) so the "would have caught xShot on day one" claim is demonstrated without a committed red state; the permanent non-vacuity proof is the positive control in Step 1.

Population anchor (spec Part 1b): a *frame-consuming model-scoring entry* = a function with a `frames` parameter that scores a fitted model. Curated registry (feature constant lets the velocity scan read the feature names) + an AST completeness check that no OTHER such function exists un-enrolled.

Verified population (subagent-confirmed line numbers):

| serve entry | frames param? | model class | feature constant | velocity feature? |
|---|---|---|---|---|
| `_xshot_occurrence.compute_xshot_occurrence` (`:828`) | yes | `XShotOccurrenceModel` | `XSHOT_FEATURE_NAMES_FAITHFUL` (`:128`) | yes (`speed`) → guard required |
| `_xcross_attempt.compute_xcross_attempt` (`:817`) | yes | `XCrossAttemptModel` | `XCROSS_FEATURE_NAMES_FAITHFUL` (`:68`) | yes (`ball_speed`) → guard required |
| `_ghost_gk._serve_positions_core` (`:2360`) | yes | `GhostGkModel` | `GHOST_GK_FEATURE_NAMES` (`:345`) | yes → guard required |
| `_gk_completion.compute_gk_completion` (`:301`) | yes (`frames: DataFrame\|None`) | `GkCompletionModel` | `GK_COMPLETION_FEATURE_NAMES` (`:44`) | no → `_NO_VELOCITY_FEATURE` |

Out of population by construction (no `frames` param): `GkRetention` (scored by `xtgk._metric.compute_xt_gk_v2`, no `frames` param), the VAEP learners, causal matching.

- [ ] **Step 1: Write the gate with its positive control and completeness/anti-rot checks.**

```python
# tests/tracking/test_velocity_feature_contract.py
"""Static enforcement of the ADR-054 velocity-availability contract (spec Part 1b).

Population anchor = frame-consuming model-scoring entries (a `frames` param + fitted-model scoring),
NOT the *_FEATURE_NAMES* naming convention -- anchoring on a convention would re-commit the exact
sin that let xShot slip. Reads CODE (not docstrings), fails closed.
"""
import ast
import importlib
import inspect
from pathlib import Path

import pytest

import silly_kicks.tracking as T

VELOCITY_KEYWORDS = {"speed", "velocity", "vx", "vy", "accel"}
# maintained allowlist for a non-velocity keyword collision (empty today; add WITH a reason)
_KEYWORD_ALLOWLIST: dict[str, str] = {}

GUARD_NAME = "velocity_unavailable_by_design"

# (module, serve_fn qualname, model class name, feature-constant name)
FRAME_SCORING_ENTRIES = [
    ("silly_kicks.tracking._xshot_occurrence", "compute_xshot_occurrence",
     "XShotOccurrenceModel", "XSHOT_FEATURE_NAMES_FAITHFUL"),
    ("silly_kicks.tracking._xcross_attempt", "compute_xcross_attempt",
     "XCrossAttemptModel", "XCROSS_FEATURE_NAMES_FAITHFUL"),
    ("silly_kicks.tracking._ghost_gk", "_serve_positions_core",
     "GhostGkModel", "GHOST_GK_FEATURE_NAMES"),
    ("silly_kicks.tracking._gk_completion", "compute_gk_completion",
     "GkCompletionModel", "GK_COMPLETION_FEATURE_NAMES"),
]
# fitted models with a `frames`-carrying serve entry but NO velocity feature (frame-served is fine;
# nothing to fabricate). AST-verified against the feature constant by test_no_velocity_exemption_holds.
_NO_VELOCITY_FEATURE = {"compute_gk_completion"}

_TRACKING_DIR = Path(T.__file__).parent
_MODEL_CLASS_NAMES = {"XShotOccurrenceModel", "XCrossAttemptModel", "GhostGkModel",
                      "GkCompletionModel", "GkRetentionModel"}


def _feature_names(module: str, const: str) -> list[str]:
    mod = importlib.import_module(module)
    return list(getattr(mod, const))


def _has_velocity_feature(names) -> bool:
    """Token-match feature names against VELOCITY_KEYWORDS.

    KNOWN LIMIT (documented, not silently ignored): this is convention-based on the SAME axis the
    population deliberately is NOT (the population is anchored structurally on the `frames` param).
    A compound-without-underscore name (`ballspeed`) would be missed, and the blind spot is shared
    with `test_no_velocity_exemption_holds` -- so a mis-detected velocity feature could be wrongly
    exempted with both green. Low probability (all current entries classify correctly), and the real
    backstop is the enrollment meta-assertion (`test_population_is_exact`): a NEW frame-served model
    forces a human to classify it, catching a name the token scan cannot. Do not rely on this scan
    alone to close the class -- it is the cheap first filter, enrollment is the guarantee.
    """
    for n in names:
        toks = set(str(n).lower().replace("__", "_").split("_"))
        if toks & VELOCITY_KEYWORDS and n not in _KEYWORD_ALLOWLIST:
            return True
    return False


def _guard_aliases(tree: ast.Module) -> set[str]:
    """Names the guard is bound to in a module (bare + any `import ... as` alias)."""
    aliases = {GUARD_NAME}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == GUARD_NAME:
                    aliases.add(a.asname or a.name)
    return aliases


def _fn_calls_guard(tree: ast.Module, fn_name: str) -> bool:
    """True iff the FunctionDef `fn_name` CALLS the guard (direct or aliased) in its BODY.

    Stronger than a module-level reference: a module that merely IMPORTS the guard but never calls
    it in the serve function does NOT pass -- exactly the refactor-forgot-to-call bug the gate
    exists to catch. Works on any parsed tree so the positive control can drive it directly.

    CAVEAT (named in the ADR): this requires a DIRECT textual call to the guard in each serve fn,
    and the two-prong block is currently copy-pasted verbatim across the serve modules. If that 4x
    block is ever DRY'd into a shared helper (e.g. `_velocity_contract_gate(frames)`) -- normally the
    right move -- the serve fns would call the helper, not the guard, and this gate goes red. At that
    point teach `_fn_calls_guard` to follow ONE call level (or add the helper name to `_guard_aliases`).
    The strictness deliberately trades refactorability for seam-visibility.
    """
    aliases = _guard_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    f = sub.func
                    if isinstance(f, ast.Name) and f.id in aliases:
                        return True
                    if isinstance(f, ast.Attribute) and f.attr in aliases:
                        return True
            return False
    raise AssertionError(f"{fn_name} not found in the parsed module")


def _serve_fn_calls_guard(module: str, fn_name: str) -> bool:
    return _fn_calls_guard(ast.parse(inspect.getsource(importlib.import_module(module))), fn_name)


@pytest.mark.parametrize("module,fn,model,const", FRAME_SCORING_ENTRIES,
                         ids=[e[1] for e in FRAME_SCORING_ENTRIES])
def test_velocity_feature_entry_references_guard(module, fn, model, const):
    """A frame-served entry with a velocity feature MUST reference the guard; a velocity-free one
    is exempt (fail-closed: an entry we cannot prove velocity-free must guard)."""
    if _has_velocity_feature(_feature_names(module, const)):
        assert _serve_fn_calls_guard(module, fn), (
            f"{fn} scores {model} which has a velocity feature but does not CALL {GUARD_NAME} in its "
            f"body (importing it is not enough). Add the ADR-054 two-prong guard at its serve seam "
            f"(spec Part 1)."
        )
    else:
        assert fn in _NO_VELOCITY_FEATURE, (
            f"{fn}'s {const} scans velocity-free but it is not in _NO_VELOCITY_FEATURE. Enrol it."
        )


def test_no_velocity_exemption_holds():
    """Every _NO_VELOCITY_FEATURE member's feature constant really is velocity-keyword-clean."""
    by_fn = {e[1]: e for e in FRAME_SCORING_ENTRIES}
    for fn in _NO_VELOCITY_FEATURE:
        module, _, _, const = by_fn[fn]
        assert not _has_velocity_feature(_feature_names(module, const)), (
            f"{fn} is exempt as velocity-free but {const} contains a velocity keyword."
        )


def _ast_frame_scoring_fns() -> set[tuple[str, str]]:
    """(module-stem, fn name) pairs for functions that (a) take a `frames` param AND (b) reference a
    bundled model class or call .predict/.predict_proba. The completeness anchor -- catches a NEW
    frame-served fitted model whatever it names its features.

    Keyed on (module-stem, name), NOT bare name: a new frame-scoring function in a DIFFERENT module
    that happens to share a name with an existing registry entry must not be masked -- the anchor is
    the whole gate's guarantee, so it cannot have that hole. SOURCE-TREE-DEPENDENT by construction
    (reads *.py off disk); fine for the documented CI path (`pytest tests/` against the checkout),
    would raise OSError against a source-stripped installed wheel -- inherent to a static AST gate.
    """
    found: set[tuple[str, str]] = set()
    for path in _TRACKING_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            names = [a.arg for a in (args.posonlyargs + args.args + args.kwonlyargs)]
            if "frames" not in names:
                continue
            scores = any(
                (isinstance(s, ast.Name) and s.id in _MODEL_CLASS_NAMES)
                or (isinstance(s, ast.Attribute) and s.attr in {"predict", "predict_proba"})
                for s in ast.walk(node)
            )
            if scores:
                found.add((path.stem, node.name))
    return found


# (module-stem, fn name) pairs: frame-consuming functions that reference a model class / predict but
# are NOT scoring serve entries (training-data builders, aggregator wrappers). Each needs a reason.
_NOT_A_SCORING_ENTRY: dict[tuple[str, str], str] = {
    # POPULATE during implementation from the observed AST set minus the registry, each with a reason,
    # e.g. ("_xshot_occurrence", "prepare_xshot_training_data"): "training-data builder, not a serve path"
}


def _registry_pairs() -> set[tuple[str, str]]:
    # dotted module -> stem, matching _ast_frame_scoring_fns()'s path.stem keys.
    return {(module.rsplit(".", 1)[-1], fn) for module, fn, _model, _const in FRAME_SCORING_ENTRIES}


def test_population_is_exact():
    """The AST-derived frame-scoring set == the registry serve entries UNION the reasoned
    non-scoring bucket. A new frame-served model is caught here even if it invents a feature-decl
    convention (the whole point of anchoring on the frames param, not the *_FEATURE_NAMES* name).
    Keyed on (module-stem, name) so a same-name-different-module function cannot slip through."""
    registry = _registry_pairs()
    observed = _ast_frame_scoring_fns()
    missing = observed - registry - set(_NOT_A_SCORING_ENTRY)
    assert not missing, (
        f"frame-consuming model-scoring functions enrolled nowhere: {sorted(missing)}. "
        f"Add to FRAME_SCORING_ENTRIES (with feature constant), or _NOT_A_SCORING_ENTRY with a reason."
    )
    stale = registry - observed
    assert not stale, f"registry entries the AST walk no longer finds: {sorted(stale)}"


def test_positive_control_call_detection_is_nonvacuous():
    """Non-vacuity, driving the REAL detector (not a reimplementation): a serve fn that IMPORTS the
    guard but never CALLS it fails, while one that calls the aliased guard passes. This is the exact
    refactor-forgot-to-call bug the module-level check would have missed."""
    imports_only = ast.parse(
        "from x import velocity_unavailable_by_design as g\n"
        "def score(frames):\n    return XShotOccurrenceModel().predict_proba(frames)\n"
    )
    calls_it = ast.parse(
        "from x import velocity_unavailable_by_design as g\n"
        "def score(frames):\n    if g(frames):\n        return None\n    return 1\n"
    )
    assert not _fn_calls_guard(imports_only, "score")  # import alone must NOT pass
    assert _fn_calls_guard(calls_it, "score")          # a real (aliased) call passes
```

- [ ] **Step 2: Run the gate; confirm GREEN, then verify red-first manually.**

Run: `python -m pytest tests/tracking/test_velocity_feature_contract.py -v`
Expected: all PASS. Then, to demonstrate red-first (spec claim): temporarily comment out Task 1's guard in `_xshot_occurrence.py`, re-run — `test_velocity_feature_entry_references_guard[compute_xshot_occurrence]` must FAIL — then restore the guard and confirm green again. Do NOT commit the reverted state.

- [ ] **Step 3: Populate `_NOT_A_SCORING_ENTRY`.**

Run: add a temporary `print(sorted(_ast_frame_scoring_fns()))` (or run `test_population_is_exact` and read the `missing` set). For each function in `missing` that is a training-data builder / aggregator wrapper (e.g. `prepare_*_training_data`, `add_*`, `compute_ghost_gk`), add it to `_NOT_A_SCORING_ENTRY` with a one-line reason, OR to `FRAME_SCORING_ENTRIES` if it is a genuine serve path. Re-run until `test_population_is_exact` passes with the buckets fully reasoned.

- [ ] **Step 4: Run the full tracking suite** to confirm no import-time surprises.

Run: `python -m pytest tests/tracking/test_velocity_feature_contract.py -v`
Expected: PASS.

---

## Task 3: space_creation opponent-perspective softening + `space_opponent_source`

**Files:**
- Modify: `silly_kicks/tracking/_space_creation.py` (`_resolve_opponent_team_id` `:56-74`; `compute_space_created` `:174-176`, and the output assembly `:275-313`)
- Modify: `silly_kicks/tracking/features.py` (`add_space_creation` — thread `space_opponent_source` through the per-action assembly, mirroring `space_denied_m2_opponent`)
- Modify: `silly_kicks/tracking/schema.py:119-125` (`TRACKING_CATEGORICAL_DOMAINS`)
- Modify: `silly_kicks/feature_glossary.py:695-713` (IF provenance columns are glossary-covered — verify first)
- Modify: `tests/test_add_star_purity.py:510-517`; `tests/tracking/test_aggregator_column_liveness.py:577`; `tests/sb360/_entries/_space.py:846-950`
- Test: `tests/tracking/test_space_creation_opponent_softening.py` (create)

**Interfaces:**
- Consumes: `velocity_unavailable_by_design` (already imported in `_space_creation.py` via `zero_velocity_if_unavailable`; add the direct import).
- Produces: `_resolve_opponent_team_id(frame, attacking_team_id, *, on_unresolvable="raise")` → `None` in `"nan"` mode on the not-two-teams branch (the `match_mask.sum()!=1` branch STILL raises in both modes). `compute_space_created`/`add_space_creation` emit `space_opponent_source` ∈ `{"resolved","unresolved_one_team"}` whenever `include_opponent_perspective=True`; `space_denied_m2_opponent` is NaN iff `space_opponent_source=="unresolved_one_team"`.

- [ ] **Step 1: Write the failing softening tests.**

```python
# tests/tracking/test_space_creation_opponent_softening.py
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._space_creation import _resolve_opponent_team_id, compute_space_created
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE


def _frame(*, teams, marker: bool) -> pd.DataFrame:
    """teams: list of team_ids to place outfield players for (1 or 2 distinct)."""
    rows = []
    x = 40.0
    for t in teams:
        rows.append({"game_id": 1, "period_id": 1, "frame_id": 1, "time_seconds": 1.0,
                     "player_id": 100 + t, "team_id": t, "x": x, "y": 34.0,
                     "is_ball": False, "is_goalkeeper": False})
        x += 10.0
    rows.append({"game_id": 1, "period_id": 1, "frame_id": 1, "time_seconds": 1.0,
                 "player_id": pd.NA, "team_id": pd.NA, "x": 52.5, "y": 34.0,
                 "is_ball": True, "is_goalkeeper": False})
    df = pd.DataFrame(rows)
    df["team_attacking_direction"] = "ltr"
    df["ball_state"] = "alive"
    df["speed"] = np.nan
    df["speed_source"] = SPEED_SOURCE_UNAVAILABLE if marker else "derived"
    return df


def test_one_team_sb360_frame_softens_not_raises():
    frame = _frame(teams=[1], marker=True)  # one-team FOV + SB360 marker
    out = compute_space_created(frame, attacking_team_id=1, include_opponent_perspective=True)
    assert out["space_created_m2"].notna().any()          # team side still computes
    assert out["space_denied_m2_opponent"].isna().all()   # opponent unresolvable -> NaN
    assert (out["space_opponent_source"] == "unresolved_one_team").all()


def test_one_team_full_tracking_frame_still_raises():
    frame = _frame(teams=[1], marker=False)  # one team, NO marker -> corrupt full-tracking
    with pytest.raises(ValueError):
        compute_space_created(frame, attacking_team_id=1, include_opponent_perspective=True)


def test_two_team_frame_resolves_source_and_computes_opponent():
    frame = _frame(teams=[1, 2], marker=True)
    out = compute_space_created(frame, attacking_team_id=1, include_opponent_perspective=True)
    assert (out["space_opponent_source"] == "resolved").all()
    # space_denied computed on the resolved path (non-vacuity: not all-NaN)
    assert out["space_denied_m2_opponent"].notna().any()


def test_attacking_team_matches_neither_still_raises_both_modes():
    frame = _frame(teams=[1, 2], marker=True)
    with pytest.raises(ValueError):
        _resolve_opponent_team_id(frame, attacking_team_id=999, on_unresolvable="nan")
    with pytest.raises(ValueError):
        _resolve_opponent_team_id(frame, attacking_team_id=999, on_unresolvable="raise")


def test_zero_and_three_team_frames_raise_even_in_nan_mode():
    # 0-team (ball only) and 3-team frames are corrupt, not FOV crops -> raise in BOTH modes.
    # Only len==1 softens; this keeps the "unresolved_one_team" provenance label accurate.
    with pytest.raises(ValueError):
        _resolve_opponent_team_id(_frame(teams=[], marker=True), attacking_team_id=1, on_unresolvable="nan")
    with pytest.raises(ValueError):
        _resolve_opponent_team_id(_frame(teams=[1, 2, 3], marker=True), attacking_team_id=1, on_unresolvable="nan")
```

- [ ] **Step 2: Run the tests, verify they fail.**

Run: `python -m pytest tests/tracking/test_space_creation_opponent_softening.py -v`
Expected: `test_one_team_sb360_frame_softens` FAILS (currently raises); `test_two_team_..._source` FAILS (no `space_opponent_source` column); the two "still raises" tests may pass.

- [ ] **Step 3: Add `on_unresolvable` to `_resolve_opponent_team_id`.**

```python
def _resolve_opponent_team_id(
    frame: pd.DataFrame,
    attacking_team_id: int | str,
    *,
    on_unresolvable: Literal["raise", "nan"] = "raise",
):
    """Resolve the opposing team id from a two-team frame (dtype-robust).

    ``on_unresolvable="raise"`` (default): a frame without exactly two team ids raises -- corrupt
    input fails loud. ``on_unresolvable="nan"``: ONLY the exactly-one-team case returns ``None``
    (a legitimate one-team SB360 FOV crop, marker-gated by the caller). **Zero teams or three-plus
    teams RAISE even in "nan" mode** -- no outfield players, or >2 team ids, is genuinely corrupt,
    not an FOV limit; softening it would be wrong, and restricting the None-return to len==1 keeps
    the caller's ``space_opponent_source == "unresolved_one_team"`` label exactly accurate. The
    ``attacking_team_id`` non-unique-match branch STILL raises in both modes -- a genuine id error.
    """
    uniq = _unique_team_ids(frame)
    if len(uniq) != 2:
        if on_unresolvable == "nan" and len(uniq) == 1:
            return None
        raise ValueError(
            "opponent perspective requires exactly two team ids in the frame "
            f"(excluding ball rows); found {list(uniq)!r}"
        )
    match_mask = ids_match(pd.Series(uniq), attacking_team_id).to_numpy()
    if match_mask.sum() != 1:
        raise ValueError(
            f"attacking_team_id {attacking_team_id!r} does not uniquely match the frame team ids {list(uniq)!r}"
        )
    return uniq[~match_mask][0]
```

- [ ] **Step 4: Marker-gate the call + emit `space_opponent_source` in `compute_space_created`.**

At `:173-176`, replace the unconditional resolve with the marker-gated one, and record the resolution outcome:

```python
    from ._velocity_availability import velocity_unavailable_by_design  # top-of-module import preferred

    opponent_team_id = None
    opponent_source = None  # only meaningful when include_opponent_perspective
    if include_opponent_perspective:
        mode = "nan" if velocity_unavailable_by_design(frame) else "raise"
        opponent_team_id = _resolve_opponent_team_id(frame, attacking_team_id, on_unresolvable=mode)
        opponent_source = "resolved" if opponent_team_id is not None else "unresolved_one_team"
```

When `opponent_team_id is None`, skip the opponent LOO branch (`obso_multiplier_opponent` stays `None`) so `space_denied_m2_opponent` is NaN, and stamp `space_opponent_source` on every returned row. In the output assembly (`:275-313`), after building `results`:
  - the empty-players early return (`:275-279`): add `space_opponent_source` to `base_cols` when `include_opponent_perspective`.
  - the `_analytical_leave_one_out` / `_naive_leave_one_out` results already carry `space_denied_m2_opponent` (NaN when opponent skipped). Add `space_opponent_source = opponent_source` to each result row (or assign the column on the returned DataFrame): `df = pd.DataFrame(results); if include_opponent_perspective: df["space_opponent_source"] = opponent_source; return df`.

Confirm the LOO helpers already produce NaN `space_denied_m2_opponent` when `obso_multiplier_opponent is None` (they receive it as a kwarg); if they instead omit the column, ensure the column exists as NaN so the schema is stable.

- [ ] **Step 5: Thread `space_opponent_source` through `add_space_creation`.**

In `silly_kicks/tracking/features.py`, `add_space_creation` calls `compute_space_created` per action and assembles per-action output. Mirror exactly how `space_denied_m2_opponent` is carried from the per-frame result into the aggregator's output columns, adding `space_opponent_source` alongside it (present only when `include_opponent_perspective=True`). Preserve purity: build a new frame; never mutate `actions`/`frames`.

- [ ] **Step 6: Run the softening tests, verify they pass.**

Run: `python -m pytest tests/tracking/test_space_creation_opponent_softening.py -v`
Expected: all PASS.

- [ ] **Step 7: Wire `space_opponent_source` into `TRACKING_CATEGORICAL_DOMAINS`.**

`silly_kicks/tracking/schema.py:119-125`, add (mirroring `speed_source`, which IS in this dict — `das_source` is NOT; it uses its own VALUES tuple, so this dict is the right home for a simple 2-value set):

```python
    "space_opponent_source": frozenset({"resolved", "unresolved_one_team"}),
```

- [ ] **Step 8: Verify whether provenance columns are glossary-covered; add the entry only if required.**

Run: `python -m pytest tests/test_feature_glossary_coverage.py -v` after Steps 3-7 (the new column now emits). If it FAILS demanding a `space_opponent_source` entry, add to `feature_glossary.py` (near `:695-713`):

```python
    FeatureColumn(
        name="space_opponent_source",
        definition=(
            "Provenance of the opponent-perspective space measurement: 'resolved' (two teams in the "
            "frame) or 'unresolved_one_team' (a one-team SB360 FOV crop -- space_denied_m2_opponent is NaN)."
        ),
        unit="dimensionless",
        emitting_module=_M_SPACE_CREATION,
        attribution=None,
        higher_is_better=None,
    ),
```

If the coverage gate does NOT require it (provenance/`*_source` columns exempt, like `das_source`/`speed_source` which the survey found absent from the glossary), skip this step and record in the task report that provenance columns are glossary-exempt.

- [ ] **Step 9: Register the ADR-033 purity variants (present + absent branch).**

`tests/test_add_star_purity.py:510-517`, extend the `"tracking:add_space_creation"` list with a variant that turns opponent perspective ON (so the `space_opponent_source` column path is purity-checked). The existing 2 variants cover `xt=`; add a third exercising `include_opponent_perspective=True`:

```python
        (
            "opponent_perspective",
            _std_inputs,
            lambda i: F.add_space_creation(i[0], i[1], home_team_id=5, include_opponent_perspective=True),
        ),
```

- [ ] **Step 10: Update the aggregator-column-liveness registration.**

`tests/tracking/test_aggregator_column_liveness.py:577`, change the `add_space_creation` registration so the fixture exercises the `include_opponent_perspective=True` branch (the ENTRIES registry is one-runner-per-name):

```python
    "add_space_creation": _std(F.add_space_creation, home_team_id=5, include_opponent_perspective=True),
```

`space_opponent_source` is object-dtype → the non-constant check is float-gated (exempt); only the non-null (dead-column) check applies, so the liveness fixture must produce it non-null (it will, on a two-team fixture → `"resolved"`).

- [ ] **Step 11: Add the SB360 audit column for `space_opponent_source`.**

`tests/sb360/_entries/_space.py:846-950` (`add_space_creation` entry): add `"space_opponent_source"` to `columns=(...)` and a verdict in the `velocity` dict + all three `visibility` roster keys + `applicability` + `applicability_deltas`. On the two-team audit fixture it resolves on both legs → observation `("identical","works")`. Confirm the audit adapter passes `include_opponent_perspective=True` (if it uses `C.generic`, which does not, add a methods/kwargs-passing adapter or set the aggregator default appropriately for the audit call — verify and, if needed, add a dedicated adapter in `scripts/_sb_battery.py` re-exported via `tests/sb360/_calls.py`). Transcribe from execution — do not assert the verdict; regenerate/hand-edit then run the locked audit test.

- [ ] **Step 12: Run the full affected suite.**

Run: `python -m pytest tests/tracking/test_space_creation_opponent_softening.py tests/test_add_star_purity.py tests/tracking/test_aggregator_column_liveness.py tests/sb360/ tests/test_feature_glossary_coverage.py -m "not e2e" -v --tb=short`
Expected: all PASS.

---

## Task 4: bekkers_pi pressing-intensity honest-NaN tier

**Files:**
- Modify: `silly_kicks/tracking/features.py:1107-1114` (`pressure_on_actor`'s `elif method == "bekkers_pi":` branch)
- Modify: `tests/sb360/_entries/_context.py:301-325`; `scripts/_sb_battery.py` (+ re-export `tests/sb360/_calls.py`)
- Test: `tests/tracking/test_bekkers_velocity_tier.py` (create)

**Interfaces:**
- Consumes: `velocity_unavailable_by_design`.
- Produces: `pressure_on_actor(actions, frames, method="bekkers_pi")` returns all-NaN on declared-velocity-unavailable frames; still raises on undeclared missing vx/vy; unchanged with velocity present. `andrienko_oval`/`link_zones` untouched. `add_pressure_on_actor(..., methods=(...,"bekkers_pi"))` inherits it (delegates to `pressure_on_actor` per method).

Note the API shapes (subagent-confirmed): the per-Series `pressure_on_actor` takes `method=` (singular); the aggregator `add_pressure_on_actor` takes `methods: tuple[Method, ...]` (plural, default `("andrienko_oval",)`). The edge belongs in the per-Series branch; confirm the aggregator delegates (if it inlines, mirror the prong there too).

- [ ] **Step 1: Write the failing tests.**

```python
# tests/tracking/test_bekkers_velocity_tier.py
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.features import pressure_on_actor
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE


def _actions() -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": [1], "period_id": [1], "action_id": [0],
        "start_x": [90.0], "start_y": [34.0], "team_id": [1], "player_id": [11],
        "time_seconds": [5.0],
    })


def _frames(*, velocity: bool, marker: bool) -> pd.DataFrame:
    rows = [
        {"game_id": 1, "period_id": 1, "frame_id": 10, "time_seconds": 5.0,
         "player_id": 11, "team_id": 1, "x": 90.0, "y": 34.0, "is_ball": False, "is_goalkeeper": False},
        {"game_id": 1, "period_id": 1, "frame_id": 10, "time_seconds": 5.0,
         "player_id": 21, "team_id": 2, "x": 92.0, "y": 34.0, "is_ball": False, "is_goalkeeper": False},
        {"game_id": 1, "period_id": 1, "frame_id": 10, "time_seconds": 5.0,
         "player_id": pd.NA, "team_id": pd.NA, "x": 90.0, "y": 34.0, "is_ball": True, "is_goalkeeper": False},
    ]
    df = pd.DataFrame(rows)
    df["team_attacking_direction"] = "ltr"
    df["ball_state"] = "alive"
    df["speed"] = 3.0 if velocity else np.nan
    if velocity:
        df["vx"] = 3.0
        df["vy"] = 0.0
    df["speed_source"] = SPEED_SOURCE_UNAVAILABLE if marker else "derived"
    return df


def test_declared_unavailable_bekkers_is_nan_not_raise():
    s = pressure_on_actor(_actions(), _frames(velocity=False, marker=True), method="bekkers_pi")
    assert s.isna().all()


def test_undeclared_missing_velocity_still_raises():
    with pytest.raises(ValueError, match="derive_velocities"):
        pressure_on_actor(_actions(), _frames(velocity=False, marker=False), method="bekkers_pi")


def test_velocity_bearing_bekkers_is_scored():
    s = pressure_on_actor(_actions(), _frames(velocity=True, marker=False), method="bekkers_pi")
    assert s.notna().any()  # non-vacuity: NaN above is suppression, not "no signal"


def test_andrienko_unaffected_on_declared_frames():
    s = pressure_on_actor(_actions(), _frames(velocity=False, marker=True), method="andrienko_oval")
    assert s.notna().any()  # positional method still computes


def test_artifact_dependence_on_the_real_surface():
    """Justifies the Tier-3 (SUPPRESS) decision by driving the REAL _pressure_bekkers filter
    end-to-end -- NOT a hand-rolled np.where reconstruction (which would prove only that the author's
    copy differs NaN-vs-0, not that the shipped code does). Reuse the existing bekkers happy-path
    fixture (known to link + score -- find it in tests/tracking/test_pressure_*.py; do NOT hand-build
    a minimal frame that may fail linkage) and vary ONLY the defender `speed` column across the
    active-pressing threshold. The discrete `speed_threshold` gate materially changes the real output
    => the zero-velocity form is gate-dependent, not a smooth limit => Tier-3, not Tier-1. If the two
    regimes ever coincide, that premise is false and the tier decision must be revisited."""
    actions, frames_scoring = _known_scoring_bekkers_fixture()
    below = frames_scoring.copy(); below["speed"] = 0.0    # all pressers below threshold -> filtered
    above = frames_scoring.copy(); above["speed"] = 10.0   # all pressers above threshold -> counted
    s_below = pressure_on_actor(actions, below, method="bekkers_pi")
    s_above = pressure_on_actor(actions, above, method="bekkers_pi")
    # the discrete gate MATERIALLY moves the real output (the essential no-smooth-limit property):
    assert not np.allclose(s_below.fillna(0.0).to_numpy(), s_above.fillna(0.0).to_numpy())
    # direction sanity: dropping all pressers (below) never yields MORE pressure than counting them:
    assert (s_above.fillna(0.0) >= s_below.fillna(0.0)).all()
    # (The SB360 manifestation is the NaN-speed vs 0-speed split of this same gate: speed=NaN makes
    #  `NaN < threshold` False so the filter no-ops, speed=0 makes it fire -- which is why a naive
    #  lift is artifact-dependent and we suppress to honest-NaN instead.)
```

- [ ] **Step 2: Run the tests, verify they fail.**

Run: `python -m pytest tests/tracking/test_bekkers_velocity_tier.py -v`
Expected: `test_declared_unavailable_bekkers_is_nan` FAILS (currently raises ValueError at the unconditional `:1109` guard); others may pass.

- [ ] **Step 3: Replace the unconditional guard with the two-prong seam.**

In `silly_kicks/tracking/features.py`, `pressure_on_actor`, the `elif method == "bekkers_pi":` branch (`:1107`). PREPEND the declared-prong before the existing `if "vx" not in ... raise` (`:1109-1114`):

```python
    elif method == "bekkers_pi":
        bp = params if isinstance(params, BekkersParams) else BekkersParams()
        if velocity_unavailable_by_design(frames):
            # DECLARED velocity-unavailable (SB360 freeze-frame): honest-NaN. bekkers_pi's
            # active-pressing speed_threshold filter is velocity-GATED, so its zero-velocity form is
            # artifact-dependent (filter no-ops on speed=NaN; fires to a degenerate 0 on speed=0) --
            # NOT the same model at a limit, so we SUPPRESS rather than lift (ADR-063 amendment).
            return pd.Series(np.nan, index=actions.index, name="pressure_on_actor__bekkers_pi")
        if "vx" not in frames.columns or "vy" not in frames.columns:
            raise ValueError(
                "pressure_on_actor(method='bekkers_pi'): frames missing velocity columns "
                "'vx'/'vy'. Run silly_kicks.tracking.preprocess.derive_velocities(frames) "
                "first, or use a provider that emits velocities natively."
            )
        ...  # unchanged: ctx, ball_xy_v_per_action, _pressure_bekkers, then the shared rename at :1131
```

Add `from ._velocity_availability import velocity_unavailable_by_design` to the module imports if not present.

**HARD PRECONDITION (not a soft "confirm"):** the guard lives in the per-Series `pressure_on_actor`. Before relying on it for the aggregator/audit path (Step 5), verify `add_pressure_on_actor` DELEGATES to `pressure_on_actor` per method. If it inlines the bekkers kernel call instead of delegating, the guard does NOT cover the aggregator and Step 5's audit call would still raise — so in that case mirror the same declared-prong at the aggregator's per-method site BEFORE Step 5. Also mirror it in any `pressure` `*_xfns` transformer for bekkers_pi (bekkers_pi is opt-in, not in the default battery → no default-list/VAEP change).

Also note the test's `_known_scoring_bekkers_fixture` (Step 1) is NOT hand-built: locate the existing bekkers happy-path fixture during implementation (the frame/action set an existing `tests/tracking/test_pressure_*.py` test asserts scores a finite bekkers value) and reuse it, so the artifact-dependence assertion cannot fail for a linkage-inadequacy reason.

- [ ] **Step 4: Run the tests, verify they pass.**

Run: `python -m pytest tests/tracking/test_bekkers_velocity_tier.py -v`
Expected: all PASS.

- [ ] **Step 5: Add the SB360 audit column + a methods-passing adapter.**

`C.generic` does not pass `methods=`, so add a dedicated adapter in `scripts/_sb_battery.py` (re-exported via `tests/sb360/_calls.py`) that calls `add_pressure_on_actor(actions, frames, methods=("andrienko_oval", "bekkers_pi"), links=...)`. In `tests/sb360/_entries/_context.py:301-325`, add `"pressure_on_actor__bekkers_pi"` to `columns=(...)` and a verdict in `velocity` + the three `visibility` roster keys + `applicability` + `applicability_deltas`, using the new adapter. **Transcribe from execution:** regenerate/hand-edit, then run the locked audit test; confirm the bekkers observation is `all_nan`/`partial_nan` → `honest_nan` and NOT `no_signal` → `not_exercised` (if `no_signal`, the fixture lacks a defender in the pressing domain — widen the fixture's pressing context minimally or note the fixture gap, but do not assert `honest_nan` if the machine says `no_signal`).

- [ ] **Step 6: Run the affected suite.**

Run: `python -m pytest tests/tracking/test_bekkers_velocity_tier.py tests/sb360/ -m "not e2e" -v --tb=short`
Expected: all PASS.

---

## Task 5: docs, ADR amendments, CHANGELOG, version bump

> **NO NUMBERS ARE CLAIMED IN THIS PLAN.** Version, PR (`PR-Snnn`), and ADR numbers are assigned ONLY at commit time, against `main` as it stands then — never pre-claimed. A sibling **cover-shadow branch is in flight** and edits the same version-bump files (`pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `silly_kicks/tracking/__init__.py`) but NOT the same production files (`features.py`/`schema.py`/`_space_creation.py`/`_xshot_occurrence.py`), so expect version-file merge conflicts only, no production-line shifts. Coordinate merge order at commit time and take the next free numbers then.

**Files:**
- Amend: `docs/superpowers/adrs/ADR-054-*.md` (xShot joins the contract + the contract becomes self-enforcing via the static gate); `docs/superpowers/adrs/ADR-063-*.md` (bekkers_pi placed in the existing 3-tier taxonomy). Consider a short **new ADR (number assigned at commit)** for the space_creation marker-gated softening if final-review judges it distinct.
- Modify: `CHANGELOG.md` (new version section — version + `PR-Snnn` filled in at commit — with the downstream-consumer note for `space_opponent_source`, marked **additive/non-breaking**).
- Modify: `CLAUDE.md` (durable contract bullets: the velocity-feature static gate; bekkers_pi Tier-3 honest-NaN + the tier-assignment discriminator; space_creation opponent softening + `space_opponent_source` + the marker-as-FOV-proxy coupling).
- Modify: `docs/PRIVATE_CONSUMERS.md` (flag `space_opponent_source` as a new, **additive** `add_space_creation` output column for lakehouse action-context materializers).
- Modify (version sites): `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md` (if it carries the version), `uv.lock` (hand-edit the `silly-kicks` package version). **The number is chosen at commit time against `main`, not now.**

- [ ] **Step 0 (precondition): verify ADR-063's own propagation landed.** Before amending ADR-063, confirm the four velocity-requiring pitch-control aggregators + the `zero_velocity_if_unavailable` edge helper the amendment builds on are present as recorded (grep `zero_velocity_if_unavailable` usages; confirm `add_gk_influence`/`add_cover_shadows`/`add_player_influence`/`add_space_creation` + `pitch_control_at_target` route through it). If any is missing, STOP and surface it — this cycle assumes that foundation.

- [ ] **Step 1: Write the ADR amendments.** ADR-054: record xShot as the third fitted model under the contract (fabrication was LIVE on `pre_shot_gk_full_default_xfns`), and that the contract is now CI-enforced by the frame-consuming-model-scoring static gate (population anchored structurally on "takes a `frames` param AND scores a fitted model", NOT on the `*_FEATURE_NAMES*` naming convention). **Name the gate's one deliberate limitation in the ADR:** it requires a direct in-body call to the guard in each serve fn, so the two-prong block stays intentionally duplicated across the serve modules; if it is ever DRY'd into a shared helper, `_fn_calls_guard` must be taught to follow one call level (or the helper added to its alias set) — the gate trades refactorability for seam-visibility, on purpose. ADR-063: place bekkers_pi in the EXISTING three-tier taxonomy as **Tier-3 (constitutively velocity)** — do NOT coin a parallel LIFT-vs-SUPPRESS binary. Record (a) the **tier-assignment discriminator**: a velocity-derived aggregator whose zero-velocity form is a smooth limit of the same model is Tier-1 (lift); one gated by a velocity-DISCRETE term (a threshold/filter, here `speed_threshold`) is Tier-3, since the discrete gate has no meaningful limit and its zero-velocity value is artifact-dependent; and (b) the **contrast with the existing ball-less degrade** — bekkers already falls back to its base model (pressure-on-player only, `_kernels.py:715-718`) when the ball is missing, a still-meaningful measurement, whereas the velocity-less discrete-gate collapse has no valid reading, which is why velocity-less is honest-NaN and ball-less is a base-model fallback. Supersede the ADR-063 "tier is a separate decision" deferral for bekkers_pi.

- [ ] **Step 2: Write the CHANGELOG entry** (version + `PR-Snnn` assigned at commit): the four changes, "no VAEP retrain", and a **downstream-consumer note** that `add_space_creation` gains a public `space_opponent_source` column — state it is **additive/non-breaking** (consumers ignore unknown columns; lazy adoption, same class as the GS `start_time` handoff).

- [ ] **Step 3: Update `CLAUDE.md`** with the durable contract bullets (velocity-feature static gate + `_NO_VELOCITY_FEATURE` exemption; bekkers_pi Tier-3 + the tier-assignment discriminator; space_creation softening + `space_opponent_source` + the marker-as-FOV-proxy coupling and its migration-to-a-real-FOV-signal note).

- [ ] **Step 4: Flag the downstream consumer** in `docs/PRIVATE_CONSUMERS.md` (additive column).

- [ ] **Step 5: Bump the version in all sites** (number decided at commit against `main`). Verify with a grep that the chosen version string is consistent across the sites and the old one no longer appears (except historical CHANGELOG entries).

- [ ] **Step 6: Run the full non-e2e suite + lint + pyright at CI scope.**

Run:
```
python -m pytest tests/ -m "not e2e" -v --tb=short
python -m ruff check silly_kicks/ tests/ scripts/
python -m ruff format --check silly_kicks/ tests/ scripts/
python -m pyright
```
Expected: all green. Capture EVERY `FAILED` line (never `tail` the output — a truncated failure list has hidden failures twice in this repo).

---

## Self-Review (author checklist — run before handing off)

- **Spec coverage:** Part 1 → Task 1; Part 1b → Task 2; Part 2 → Task 3; Part 4 → Task 4; ADR/Impact/Testing → Tasks 1-5. ✓
- **No numbers claimed.** No version, `PR-Snnn`, or ADR number appears in this plan; all are assigned at commit time against `main` (Task 5 header). Cover-shadow merge-order coordination is flagged (version-file conflicts only). ✓
- **"Locate/verify during implementation" items (each with a concrete command + decision rule, not a vague TODO):** `_known_scoring_xshot_frames` (Task 1, reuse the existing xShot happy-path fixture); `_NOT_A_SCORING_ENTRY` population (Task 2 Step 3, from the AST walk); glossary-coverage branch (Task 3 Step 8, gated by running the coverage test); `_known_scoring_bekkers_fixture` (Task 4, reuse the existing bekkers happy-path fixture); the ADR-063 propagation precondition (Task 5 Step 0). None is a silent placeholder — all are "reuse the known-good existing thing / read the gate output," chosen precisely to avoid the fixture-inadequacy and reimplements-the-code traps the review flagged.
- **Type/name consistency:** `velocity_unavailable_by_design` (guard), `SPEED_SOURCE_UNAVAILABLE` (marker), `on_unresolvable` (kwarg, softens only `len==1`), `space_opponent_source` ∈ `{"resolved","unresolved_one_team"}`, `pressure_on_actor__bekkers_pi` (column), `method=` (per-Series `pressure_on_actor`) vs `methods=(...)` (aggregator `add_pressure_on_actor`), bekkers_pi = ADR-063 **Tier-3** (not a new binary) — used consistently across tasks.
- **Both-sides / non-vacuity tests:** each guard asserts the degrade side AND the raise/compute side; the static gate's guard-check requires a CALL in the serve fn (not a module import) and its positive control drives the REAL detector on import-only vs calls-it; the bekkers artifact-dependence test drives the REAL kernel (below-vs-above threshold), not a hand-rolled filter; space_creation asserts softens-vs-still-raises AND 0/3-team-still-raises. ✓
- **No commit steps.** ✓
