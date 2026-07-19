"""ADR-019 id-scalar boundary gate: enumerated, behavioral, complete (ADR-043, 4.53.0).

Replaces ``tests/tracking/test_id_compat_lint.py``. The lint was a NAME heuristic over one
package's glob and could not see the only thing that distinguishes a safe comparison from an
unsafe one -- the PROVENANCE of the scalar. This gate asks the behavioral question instead:

    call the public function twice, once with a dtype-MATCHED id scalar and once with a
    MISMATCHED-but-value-EQUAL one, and require identical output.

A function that compares raw fails; a function routed through ``silly_kicks.id_compat``
passes. No syntax is inspected, so no correct code can ever be flagged.

See ``conftest_id_scalar.py`` for how the public surface is enumerated and why the
enumeration is complete.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from tests.invariants.conftest_id_scalar import (
    COVERED_BY_AGGREGATOR_GATE,
    NOT_EXERCISABLE,
    NOT_INVARIANT,
    PUBLIC_ID_SCALAR_ENTRIES,
    discover_public_id_scalar_functions,
)

# ADR-041 opt-out: this gate sweeps the OBSO/space-creation families on defaults, so their
# synthetic-EPV notice is expected here and unrelated to what the gate asserts.
pytestmark = pytest.mark.filterwarnings("ignore::silly_kicks.tracking.SyntheticEPVWarning")


# --------------------------------------------------------------------------------------
# Output comparison
# --------------------------------------------------------------------------------------


def _is_id_col(name: str) -> bool:
    """Id-valued OUTPUT columns are excluded from the value comparison.

    Some functions RECORD the scalar they were given (``compute_team_shape`` writes
    ``team_id`` into its result, ``_team_shape.py`` L154). A matched run then holds ``5``
    where a mismatched run holds ``"5"`` -- a faithful echo of the input, not a
    mis-resolution. Comparing those would fail every such function for the wrong reason.
    Same rule as the aggregator gate's B1.
    """
    return "team_id" in name or "player_id" in name or name.endswith("_id")


def _normalize(obj):
    """Reduce a return value to a comparable, dtype-agnostic structure."""
    if isinstance(obj, pd.DataFrame):
        cols = [c for c in obj.columns if not _is_id_col(str(c))]
        return ("df", obj[cols].reset_index(drop=True))
    if isinstance(obj, pd.Series):
        return ("series", obj.reset_index(drop=True))
    if isinstance(obj, np.ndarray):
        return ("array", obj)
    if isinstance(obj, dict):
        return ("dict", {k: _normalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))})
    if isinstance(obj, (list, tuple)):
        return ("seq", [_normalize(v) for v in obj])
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj) if not _is_id_col(f.name)}
        return ("dataclass", type(obj).__name__, _normalize(fields))
    return ("scalar", obj)


def _assert_same(matched, mismatched, key: str) -> None:
    a, b = _normalize(matched), _normalize(mismatched)
    _compare(a, b, key, path="")


def _compare(a, b, key: str, path: str) -> None:
    assert type(a) is type(b) and a[0] == b[0], f"{key}{path}: shape differs ({a[0]} vs {b[0]})"
    kind = a[0]
    if kind == "df":
        pd.testing.assert_frame_equal(a[1], b[1], check_dtype=False, check_like=True, obj=f"{key}{path}")
    elif kind == "series":
        pd.testing.assert_series_equal(a[1], b[1], check_dtype=False, check_names=False, obj=f"{key}{path}")
    elif kind == "array":
        np.testing.assert_allclose(a[1], b[1], equal_nan=True, err_msg=f"{key}{path}")
    elif kind == "dict":
        assert a[1].keys() == b[1].keys(), f"{key}{path}: dict keys differ"
        for k in a[1]:
            _compare(a[1][k], b[1][k], key, f"{path}[{k!r}]")
    elif kind == "seq":
        assert len(a[1]) == len(b[1]), f"{key}{path}: length differs"
        for i, (x, y) in enumerate(zip(a[1], b[1], strict=True)):
            _compare(x, y, key, f"{path}[{i}]")
    elif kind == "dataclass":
        assert a[1] == b[1], f"{key}{path}: dataclass type differs"
        _compare(a[2], b[2], key, f"{path}.{a[1]}")
    else:
        if isinstance(a[1], float) and isinstance(b[1], float):
            assert a[1] == pytest.approx(b[1], nan_ok=True), f"{key}{path}: {a[1]} != {b[1]}"
        else:
            assert a[1] == b[1], f"{key}{path}: {a[1]!r} != {b[1]!r}"


def _outputs_differ(a, b) -> bool:
    """Whether two invocations produced DIFFERENT output, by the gate's own comparison rule.

    Reuses ``_assert_same`` rather than re-implementing equality, so "same" means exactly what
    the invariance assertion means by it -- a load-bearing check written against a second,
    looser notion of equality could report a difference the gate itself cannot see.
    """
    try:
        _assert_same(a, b, "<probe>")
    except AssertionError:
        return True
    return False


def _empty_collections(scalar):
    """The entry's scalar with every id COLLECTION slot emptied, other slots untouched.

    Entries express the collection either as the whole scalar (``add_gk_role``'s
    ``goalkeeper_ids``) or as one slot of a tuple (metrica/sportec drive ``home_team_id`` and
    ``goalkeeper_ids`` together), so the walk is recursive.
    """
    if isinstance(scalar, (set, frozenset)):
        return type(scalar)()
    if isinstance(scalar, tuple):
        return tuple(_empty_collections(v) for v in scalar)
    return scalar


def _has_collection(scalar) -> bool:
    if isinstance(scalar, (set, frozenset)):
        return True
    if isinstance(scalar, tuple):
        return any(_has_collection(v) for v in scalar)
    return False


_COLLECTION_ENTRIES = [e for e in PUBLIC_ID_SCALAR_ENTRIES if _has_collection(e.matched)]


@pytest.mark.parametrize("entry", _COLLECTION_ENTRIES, ids=lambda e: e.key)
def test_entity_id_collection_is_load_bearing(entry):
    """An id COLLECTION the fixture never actually resolves makes its entry VACUOUS.

    The dtype-invariance assertion asks "does a matched scalar and a mismatched one give the
    same answer?". If the fixture reaches that answer by some OTHER route -- ``add_gk_role``'s
    ``same_player`` rule already linking the rows, so ``goalkeeper_ids`` never decides anything
    -- then BOTH legs agree for a reason unrelated to id resolution, and the entry passes just
    as happily with a raw ``.isin()`` as with a canonicalized one. It reads as coverage and is
    not.

    Non-vacuity (``_has_live_value``) cannot catch this: the output is fully populated, it is
    simply not RESPONSIVE to the set. So the probe has to be differential -- empty the
    collection and require the answer to CHANGE. An entry that survives that is one where a
    broken ``.isin()`` provably changes the output, which is the only condition under which
    "matched == mismatched" is evidence of anything.

    Measured, not assumed: with the pre-ADR-043 fixture both ``add_gk_role`` entries returned
    an identical ``gk_role`` for ``{"1"}``, ``{1}`` and ``None`` alike.
    """
    resolved = entry.invoke(entry.matched)
    emptied = entry.invoke(_empty_collections(entry.matched))
    assert _outputs_differ(resolved, emptied), (
        f"{entry.key}: emptying the id collection does not change the output, so the set never "
        "RESOLVES against the id column and this entry's dtype-invariance assertion holds for a "
        "broken `.isin()` too. Fix the FIXTURE so the collection decides the answer; do not "
        "weaken the assertion."
    )


def _has_live_value(obj) -> bool:
    """At least one non-null value somewhere in the output.

    NON-VACUITY. An all-NaN result compares equal to another all-NaN result for ANY
    implementation, correct or broken -- which is exactly how ``add_das`` sat in the
    aggregator gate comparing NaN to NaN until ADR-043 gave it teeth. An entry whose output
    is entirely null is not evidence of invariance.
    """
    if isinstance(obj, pd.DataFrame):
        return bool(len(obj)) and bool(obj.notna().to_numpy().any())
    if isinstance(obj, pd.Series):
        return bool(obj.notna().any())
    if isinstance(obj, np.ndarray):
        return bool(obj.size) and bool(np.isfinite(obj).any())
    if isinstance(obj, dict):
        return any(_has_live_value(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_live_value(v) for v in obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return any(_has_live_value(getattr(obj, f.name)) for f in dataclasses.fields(obj))
    if isinstance(obj, float):
        return bool(np.isfinite(obj))
    return obj is not None


# --------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("entry", PUBLIC_ID_SCALAR_ENTRIES, ids=lambda e: e.key)
def test_public_id_scalar_is_dtype_invariant(entry):
    """A value-equal id scalar of a different dtype must not change the answer."""
    matched = entry.invoke(entry.matched)
    mismatched = entry.invoke(entry.mismatched)

    if entry.live_columns:
        # Passthrough-frame returns: judge the COMPUTED columns, not the echoed input.
        for col in entry.live_columns:
            assert isinstance(matched, pd.DataFrame) and col in matched.columns, (
                f"{entry.key}: declared live_column {col!r} is absent from the output"
            )
            assert _has_live_value(matched[col]), (
                f"{entry.key}: computed column {col!r} is entirely null, so 'matched == "
                "mismatched' would hold for a broken implementation too -- fix the fixture "
                "(put the action in the function's domain), do not weaken the assertion."
            )
    else:
        assert _has_live_value(matched), (
            f"{entry.key}: output carries no non-null value, so 'matched == mismatched' would hold "
            "for a broken implementation too -- fix the fixture, do not weaken the assertion."
        )
    _assert_same(matched, mismatched, entry.key)

    # THIRD axis: a FLOAT-valued scalar. The int-vs-str pair above cannot catch a naive
    # `str(column_value) == str(scalar)` compare, because both render identically for
    # integers. A float scalar does: `str(1)` is "1" while `str(1.0)` is "1.0", so the naive
    # form silently matches NOTHING, whereas `canonical_id` collapses both to "1".
    #
    # This is not hypothetical. `_ghost_gk._build_score_lookup` carried exactly that compare
    # against the goal-scoring team, and on a float-backed id EVERY goal fell to the away
    # side: a 3-goal fixture (2 home, 1 away) returned score_diff -3 instead of +1, feeding
    # `score_diff` -- one of the 26 TRAINED ghost-GK features. Renaming the scalar
    # (`home_team_id_norm`) also made it invisible to the AST lint this registry replaced.
    if isinstance(entry.matched, (int, float)) and not isinstance(entry.matched, bool):
        float_out = entry.invoke(float(entry.matched))
        _assert_same(matched, float_out, f"{entry.key} [float scalar]")


# --------------------------------------------------------------------------------------
# Meta-assertions: the registry must track the public surface
# --------------------------------------------------------------------------------------


def test_registry_covers_every_public_id_scalar_function():
    """THE anti-rot assertion.

    The lint's glob silently missed 17 modules because nothing forced it to keep up. This
    enumerates the public surface from ``inspect.signature`` at run time, so a newly-exported
    function taking an id scalar fails CI until it is registered, delegated, or justified.
    """
    discovered = set(discover_public_id_scalar_functions())
    registered = {e.key for e in PUBLIC_ID_SCALAR_ENTRIES}
    accounted = registered | set(COVERED_BY_AGGREGATOR_GATE) | set(NOT_INVARIANT) | set(NOT_EXERCISABLE)

    missing = discovered - accounted
    assert not missing, (
        "public function(s) taking an id-valued scalar are unaccounted for -- add an entry to "
        "PUBLIC_ID_SCALAR_ENTRIES, or (with a written justification) to "
        f"COVERED_BY_AGGREGATOR_GATE / NOT_INVARIANT / NOT_EXERCISABLE: {sorted(missing)}"
    )


def test_discovery_sees_a_module_that_declares_no___all__(monkeypatch):
    """META, red-first: the ``__all__``-less fallback must actually DISCOVER.

    Discovery originally walked ``__all__`` alone. 35 of the walked modules declare none, so
    they contributed nothing and 13 public id-scalar callables -- every provider
    ``convert_to_actions``, every native ``convert_to_frames``, and both ``direction``
    primitives -- sat in NO bucket while the meta-assertion above reported full coverage. A
    discovery rule that silently stops looking is the deleted lint's exact failure mode, so
    the fallback needs its own proof rather than inheriting trust from the entries it found.

    Plants a synthetic ``__all__``-less module carrying an id-scalar function and asserts BOTH
    that discovery returns it and that the anti-rot assertion would flag it as unaccounted --
    the second half matters because discovery that finds a function nothing acts on is inert.
    """
    import types

    from tests.invariants import conftest_id_scalar as C

    planted_mod = types.ModuleType("silly_kicks.tracking._planted_probe")

    def offending_function(frames, *, home_team_id):  # pragma: no cover - never called
        return frames

    offending_function.__module__ = planted_mod.__name__
    # __qualname__ MUST be reset: a function defined inside a test carries
    # "<test name>.<locals>.offending_function", and discovery keys on
    # f"{__module__}.{__qualname__}" -- so without this the plant is discovered under a key
    # that no assertion here names, and the test fails for a reason unrelated to the fallback.
    offending_function.__qualname__ = "offending_function"
    planted_mod.offending_function = offending_function
    assert not hasattr(planted_mod, "__all__"), "the plant must exercise the FALLBACK path"

    real_public_modules = C._public_modules

    def _with_plant(root):
        mods = real_public_modules(root)
        return [*mods, planted_mod] if root == "silly_kicks.tracking" else mods

    monkeypatch.setattr(C, "_public_modules", _with_plant)

    key = "silly_kicks.tracking._planted_probe.offending_function"
    discovered = C.discover_public_id_scalar_functions()
    assert key in discovered, (
        "the planted id-scalar callable in an __all__-less module was NOT discovered -- the "
        "fallback in _public_names is not reaching this shape"
    )
    assert discovered[key] == ("home_team_id",)

    accounted = (
        {e.key for e in C.PUBLIC_ID_SCALAR_ENTRIES}
        | set(C.COVERED_BY_AGGREGATOR_GATE)
        | set(C.NOT_INVARIANT)
        | set(C.NOT_EXERCISABLE)
    )
    assert key not in accounted, "the plant must be UNaccounted, or this proves nothing"
    assert set(discovered) - accounted == {key}, (
        "discovering the plant must make the anti-rot assertion go red on exactly it"
    )


def test_discovery_fallback_ignores_names_merely_imported_into_a_module():
    """The fallback's other half: it must not claim surface a module did not publish.

    Keying on ``obj.__module__`` is what keeps a re-imported helper (or ``pandas``) from being
    filed under the importing module -- which would key entries to a foreign module and make
    the registry unstable across unrelated import churn.
    """
    import types

    from tests.invariants import conftest_id_scalar as C

    mod = types.ModuleType("silly_kicks.tracking._planted_importer")

    def borrowed(frames, *, team_id):  # pragma: no cover - never called
        return frames

    borrowed.__module__ = "silly_kicks.somewhere_else"  # defined ELSEWHERE
    mod.borrowed = borrowed

    assert "borrowed" not in C._public_names(mod), (
        "_public_names claimed a name this module only imported -- entries would be keyed to the wrong module"
    )


def test_registry_has_no_stale_entries():
    """The mirror direction: an entry naming a function that is no longer public (renamed,
    unexported, deleted) is dead weight that reads as coverage. Fail on it rather than let
    the registry drift into fiction."""
    discovered = set(discover_public_id_scalar_functions())
    accounted = (
        {e.key for e in PUBLIC_ID_SCALAR_ENTRIES}
        | set(COVERED_BY_AGGREGATOR_GATE)
        | set(NOT_INVARIANT)
        | set(NOT_EXERCISABLE)
    )

    stale = accounted - discovered
    assert not stale, (
        f"registry entries no longer match a public id-scalar function (renamed or unexported?): {sorted(stale)}"
    )


def test_delegated_entries_are_really_covered():
    """Delegation must be VERIFIED, not promised.

    Every COVERED_BY_AGGREGATOR_GATE key must actually appear in the aggregator gate's own
    registered surface. Without this, a prose 'covered elsewhere' note is indistinguishable
    from an untested exemption -- and would become one the moment that gate dropped an entry.
    """
    from tests.tracking.conftest_id_dtype import AGGREGATORS

    swept = {a.__name__.split("[")[0] for a in AGGREGATORS}
    for key in COVERED_BY_AGGREGATOR_GATE:
        short = key.rsplit(".", 1)[-1]
        assert short in swept, (
            f"{key} is delegated to test_id_dtype_invariance.py, but that gate does not sweep "
            f"{short!r}. Either register it here or restore its coverage there."
        )


def test_every_exemption_carries_a_justification():
    """A bare name in an exemption bucket is an untested function wearing a coverage badge."""
    for bucket, name in (
        (COVERED_BY_AGGREGATOR_GATE, "COVERED_BY_AGGREGATOR_GATE"),
        (NOT_INVARIANT, "NOT_INVARIANT"),
        (NOT_EXERCISABLE, "NOT_EXERCISABLE"),
    ):
        for key, reason in bucket.items():
            assert isinstance(reason, str) and len(reason.strip()) >= 40, (
                f"{name}[{key!r}] needs a real written justification, got {reason!r}"
            )


def test_the_four_play_left_to_right_siblings_are_registered():
    """The known-LIVE shape (see the module docstring) is pinned by name.

    ``play_left_to_right`` is where the defect actually shipped, in four parallel copies. A
    future refactor that drops one from the registry must go red here, not quietly halve the
    coverage of the one seam with a proven failure.
    """
    registered = {e.key for e in PUBLIC_ID_SCALAR_ENTRIES}
    for key in (
        "silly_kicks.spadl.utils.play_left_to_right",
        "silly_kicks.atomic.spadl.utils.play_left_to_right",
        "silly_kicks.vaep.features.core.play_left_to_right",
        "silly_kicks.atomic.vaep.features.play_left_to_right",
    ):
        assert key in registered, f"{key} must be directly registered (known-live defect shape)"


def test_registry_spans_every_adr019_package():
    """ADR-019 names five consuming packages. A registry that only reached ``tracking/``
    would repeat the deleted lint's central mistake (it globbed one package)."""
    prefixes = {".".join(e.key.split(".")[:2]) for e in PUBLIC_ID_SCALAR_ENTRIES}
    for pkg in (
        "silly_kicks.spadl",
        "silly_kicks.atomic",
        "silly_kicks.vaep",
        "silly_kicks.causal",
        "silly_kicks.tracking",
    ):
        assert pkg in prefixes, f"no directly-registered entry from {pkg}"


def test_entries_are_unique():
    keys = [e.key for e in PUBLIC_ID_SCALAR_ENTRIES]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f"duplicate registry keys: {dupes}"


def test_matched_and_mismatched_scalars_differ_in_dtype_not_value():
    """The gate is only meaningful if the two scalars are VALUE-equal and TYPE-different.

    A pair that differs in value would make every entry fail for the wrong reason; a pair
    that differs in neither would make every entry pass vacuously.
    """
    from silly_kicks.id_compat import canonical_id

    def _canon(v):
        # RECURSIVE: an entry that drives two id params at once declares a tuple, and either
        # slot may itself be an id COLLECTION (metrica/sportec pass `(home_team_id,
        # goalkeeper_ids)`). A one-level version canonicalized the inner set by stringifying
        # the container -- "{999}" vs "{'999'}" -- and reported the two scalars as differing in
        # VALUE, which is the one thing this assertion exists to distinguish from a dtype diff.
        if isinstance(v, (set, frozenset)):
            return frozenset(_canon(x) for x in v)
        if isinstance(v, tuple):
            return tuple(_canon(x) for x in v)
        return canonical_id(v)

    for e in PUBLIC_ID_SCALAR_ENTRIES:
        assert _canon(e.matched) == _canon(e.mismatched), f"{e.key}: scalars are not value-equal"
        assert type(e.matched) is not type(e.mismatched) or e.matched != e.mismatched, (
            f"{e.key}: matched and mismatched scalars are identical -- the gate would be vacuous"
        )
