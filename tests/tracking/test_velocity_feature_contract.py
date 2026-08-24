"""Static enforcement of the ADR-054 velocity-availability contract (spec Part 1b).

The population is anchored on the STRUCTURAL property "takes a ``frames`` param AND scores a fitted
model", NOT on the ``*_FEATURE_NAMES*`` naming convention -- anchoring on a naming convention would
re-commit the exact sin that let xShot ship a fabricating serve path. Reads CODE (not docstrings),
fails closed.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

import silly_kicks.tracking as T

VELOCITY_KEYWORDS = {"speed", "velocity", "vx", "vy", "accel"}
# maintained allowlist for a non-velocity keyword collision (empty today; add WITH a reason).
_KEYWORD_ALLOWLIST: dict[str, str] = {}

GUARD_NAME = "velocity_unavailable_by_design"

# (module, serve-fn name, model class name, feature-constant name). The feature constant lets the
# velocity scan read the feature names; the model class name lets the AST completeness walk recognise
# a scoring function.
FRAME_SCORING_ENTRIES = [
    (
        "silly_kicks.tracking._xshot_occurrence",
        "compute_xshot_occurrence",
        "XShotOccurrenceModel",
        "XSHOT_FEATURE_NAMES_FAITHFUL",
    ),
    (
        "silly_kicks.tracking._xcross_attempt",
        "compute_xcross_attempt",
        "XCrossAttemptModel",
        "XCROSS_FEATURE_NAMES_FAITHFUL",
    ),
    ("silly_kicks.tracking._ghost_gk", "_serve_positions_core", "GhostGkModel", "GHOST_GK_FEATURE_NAMES"),
    (
        "silly_kicks.tracking._gk_completion",
        "compute_gk_completion",
        "GkCompletionModel",
        "GK_COMPLETION_FEATURE_NAMES",
    ),
]
# frame-served fitted models with NO velocity feature (frame-served is fine; nothing to fabricate).
# AST-verified against the feature constant by test_no_velocity_exemption_holds.
_NO_VELOCITY_FEATURE = {"compute_gk_completion"}

_TRACKING_DIR = Path(T.__file__).parent
_MODEL_CLASS_NAMES = {
    "XShotOccurrenceModel",
    "XCrossAttemptModel",
    "GhostGkModel",
    "GkCompletionModel",
    "GkRetentionModel",
}


def _feature_names(module: str, const: str) -> list[str]:
    return list(getattr(importlib.import_module(module), const))


def _has_velocity_feature(names) -> bool:
    """Token-match feature names against VELOCITY_KEYWORDS.

    KNOWN LIMIT (documented, not silently ignored): convention-based on the SAME axis the population
    deliberately is NOT. A compound-without-underscore name (``ballspeed``) would be missed, and the
    blind spot is shared with ``test_no_velocity_exemption_holds``. Low probability (all current
    entries classify correctly); the real backstop is ``test_population_is_exact`` -- a NEW
    frame-served model forces a human to classify it, catching a name the scan cannot. This scan is
    the cheap first filter, enrollment is the guarantee.
    """
    for n in names:
        toks = set(str(n).lower().replace("__", "_").split("_"))
        if toks & VELOCITY_KEYWORDS and n not in _KEYWORD_ALLOWLIST:
            return True
    return False


def _guard_aliases(tree: ast.Module) -> set[str]:
    """Names the guard is bound to in a module (bare + any ``import ... as`` alias)."""
    aliases = {GUARD_NAME}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == GUARD_NAME:
                    aliases.add(a.asname or a.name)
    return aliases


def _fn_calls_guard(tree: ast.Module, fn_name: str) -> bool:
    """True iff the FunctionDef ``fn_name`` CALLS the guard (direct or aliased) in its BODY.

    Stronger than a module-level reference: a module that merely IMPORTS the guard but never calls it
    in the serve function does NOT pass -- exactly the refactor-forgot-to-call bug the gate exists to
    catch. Works on any parsed tree so the positive control can drive it directly.

    CAVEAT (named in ADR-054): this requires a DIRECT textual call in each serve fn, and the two-prong
    block is currently copy-pasted across the serve modules. If it is ever DRY'd into a shared helper
    (e.g. ``_velocity_contract_gate(frames)``) -- normally the right move -- the serve fns would call
    the helper, not the guard, and this gate goes red; at that point teach this to follow ONE call
    level (or add the helper to ``_guard_aliases``). The strictness deliberately trades
    refactorability for seam-visibility.
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


@pytest.mark.parametrize("module,fn,model,const", FRAME_SCORING_ENTRIES, ids=[e[1] for e in FRAME_SCORING_ENTRIES])
def test_velocity_feature_entry_references_guard(module, fn, model, const):
    """A frame-served entry with a velocity feature MUST call the guard; a velocity-free one is
    exempt (fail-closed: an entry we cannot prove velocity-free must guard)."""
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
    """(module-stem, fn name) pairs for functions that (a) take a ``frames`` param AND (b) reference
    a bundled model class or call ``.predict``/``.predict_proba``. The completeness anchor -- catches
    a NEW frame-served fitted model whatever it names its features.

    Keyed on (module-stem, name), NOT bare name: a new frame-scoring function in a DIFFERENT module
    that happens to share a name with an existing registry entry must not be masked -- the anchor is
    the whole gate's guarantee. SOURCE-TREE-DEPENDENT by construction (reads ``*.py`` off disk); fine
    for the documented CI path (``pytest tests/`` against the checkout), would raise ``OSError``
    against a source-stripped installed wheel -- inherent to a static AST gate.
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


# (module-stem, fn name) -> reason. Frame-consuming functions the AST walk flags (a `frames` param +
# a model-class/predict reference) that do NOT need their OWN guard call. Four verified categories:
#   (a) wrappers that DELEGATE to a registry serve entry which carries the guard;
#   (b) model-agnostic EVAL probes (offline, not a production serve path);
#   (c) scorers of the velocity-FREE GkCompletionModel (no velocity feature to fabricate);
#   (d) velocity-keyed variant RESOLVERS -- they reference the model CLASS (to call `from_variant`)
#       but never extract features or `.predict`; they return `(model, key)`, and the guard lives in
#       the compute_* / _serve_positions_core registry entry that consumes that return.
# Each reason was verified by reading the function body, not assumed (the delegation/velocity-free/
# resolver claim is the load-bearing fact -- a false one would re-open the hole the gate closes).
_GUARD_EXEMPT: dict[tuple[str, str], str] = {
    ("_xshot_occurrence", "add_xshot_occurrence"): (
        "aggregator wrapper; delegates to compute_xshot_occurrence (registry entry) which guards"
    ),
    ("_xcross_attempt", "add_xcross_attempt"): (
        "aggregator wrapper; delegates to compute_xcross_attempt (registry entry) which guards"
    ),
    ("_ghost_gk", "compute_ghost_gk"): (
        "public ghost serve wrapper; delegates to _serve_positions_core (registry entry) which guards"
    ),
    ("_ghost_gk", "serve_ghost_gk_positions"): (
        "gkdv ghost serve wrapper; delegates to _serve_positions_core (registry entry) which guards"
    ),
    ("_model_eval", "_targets_deltas"): (
        "model-agnostic GK-substitution eval probe (ADR-037), not a production serve path"
    ),
    ("features", "add_gk_completion"): (
        "aggregator over the velocity-FREE GkCompletionModel (no velocity feature to fabricate)"
    ),
    ("_xt_gk", "_completion_p"): ("scores the velocity-FREE GkCompletionModel (no velocity feature to fabricate)"),
    ("_xt_gk", "_resolve_completion_for_frames"): (
        "resolves the velocity-FREE GkCompletionModel (no velocity feature to fabricate)"
    ),
    ("_xt_gk", "compute_xt_gk"): (
        "xt_gk v1 metric; scores the velocity-FREE GkCompletionModel (no velocity feature to fabricate)"
    ),
    ("_xshot_occurrence", "_resolve_xshot_model_for_frames"): (
        "velocity-keyed variant resolver (cat. d): returns (model, key), never extracts/predicts; the "
        "guard lives in compute_xshot_occurrence (registry entry) which calls it"
    ),
    ("_xcross_attempt", "_resolve_xcross_model_for_frames"): (
        "velocity-keyed variant resolver (cat. d): returns (model, key), never extracts/predicts; the "
        "guard lives in compute_xcross_attempt (registry entry) which calls it"
    ),
    ("_ghost_gk", "_resolve_ghost_model_for_frames"): (
        "velocity-keyed variant resolver (cat. d): returns (model, key), never extracts/predicts; the "
        "guard lives in _serve_positions_core (registry entry) which calls it"
    ),
}


def _registry_pairs() -> set[tuple[str, str]]:
    # dotted module -> stem, matching _ast_frame_scoring_fns()'s path.stem keys.
    return {(module.rsplit(".", 1)[-1], fn) for module, fn, _model, _const in FRAME_SCORING_ENTRIES}


def test_population_is_exact():
    """The AST-derived frame-scoring set == the registry serve entries UNION the reasoned
    non-scoring bucket. A new frame-served model is caught here even if it invents a feature-decl
    convention (the whole point of anchoring on the frames param). Keyed on (module-stem, name) so a
    same-name-different-module function cannot slip through."""
    registry = _registry_pairs()
    observed = _ast_frame_scoring_fns()
    missing = observed - registry - set(_GUARD_EXEMPT)
    assert not missing, (
        f"frame-consuming model-scoring functions enrolled nowhere: {sorted(missing)}. "
        f"Add to FRAME_SCORING_ENTRIES (with feature constant), or _GUARD_EXEMPT with a reason."
    )
    stale = registry - observed
    assert not stale, f"registry entries the AST walk no longer finds: {sorted(stale)}"


def test_guard_exempt_entries_still_exist():
    """Self-burning-down (ADR-056 idiom): an exemption that the AST walk no longer sees is stale --
    the function was renamed/removed, so its reason is dead and must be pruned, not left to rot."""
    observed = _ast_frame_scoring_fns()
    stale = sorted(k for k in _GUARD_EXEMPT if k not in observed)
    assert not stale, f"_GUARD_EXEMPT names frame-scoring functions that no longer exist: {stale}"


def test_positive_control_call_detection_is_nonvacuous():
    """Non-vacuity, driving the REAL detector (not a reimplementation): a serve fn that IMPORTS the
    guard but never CALLS it fails, while one that calls the aliased guard passes -- the exact
    refactor-forgot-to-call bug a module-level check would miss."""
    imports_only = ast.parse(
        "from x import velocity_unavailable_by_design as g\n"
        "def score(frames):\n    return XShotOccurrenceModel().predict_proba(frames)\n"
    )
    calls_it = ast.parse(
        "from x import velocity_unavailable_by_design as g\n"
        "def score(frames):\n    if g(frames):\n        return None\n    return 1\n"
    )
    assert not _fn_calls_guard(imports_only, "score")  # import alone must NOT pass
    assert _fn_calls_guard(calls_it, "score")  # a real (aliased) call passes
