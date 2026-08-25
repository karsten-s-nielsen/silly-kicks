"""Detection-aware provider visibility-usability contract (ADR-XXX; spec 4.3 lineage).

The taxonomy + the shared rule live in the neutral ``tracking/_provider_visibility.py`` (the 4.53.0
``_id_compat`` precedent -- a rule multiple consumers must obey does not belong inside one consumer's
private module). ``keeper_detection_mask`` STAYS in ``_ghost_gk.py`` and delegates.

Covers Task 1 (shared rule + refactor + true clean break) and Task 2 (the materializer build-time
seam). Layer 2 (the ghost-trainer pre-flight) is exercised in
``tests/scripts/test_trainer_cache_and_providers.py`` beside its sibling pre-flight test.
"""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import keeper_detection_mask
from silly_kicks.tracking._provider_visibility import assert_detection_aware_visibility


def test_all_null_detection_aware_raises_with_remedy():
    with pytest.raises(ValueError, match=r"tracking\.skillcorner"):
        assert_detection_aware_visibility(pd.Series([None, None], dtype="object"), provider="skillcorner")


def test_empty_detection_aware_raises():
    # Preserves the original keeper_detection_mask semantics (no len() guard):
    # pd.Series([]).isna().all() is True.
    with pytest.raises(ValueError):
        assert_detection_aware_visibility(pd.Series([], dtype="object"), provider="skillcorner")


def test_non_null_detection_aware_ok():
    assert assert_detection_aware_visibility(pd.Series([True, None, True]), provider="skillcorner") is None


def test_fully_observed_all_null_noop():
    assert assert_detection_aware_visibility(pd.Series([None, None]), provider="gradientsports") is None


def test_keeper_detection_mask_mask_output_unchanged():
    # MASK output preserved by the refactor (the message is unified, the returned array is not).
    out = keeper_detection_mask(pd.Series([True, None, True]), provider="skillcorner")
    np.testing.assert_array_equal(out, np.array([True, False, True]))


def test_keeper_detection_mask_still_raises_on_all_null():
    with pytest.raises(ValueError, match=r"tracking\.skillcorner"):
        keeper_detection_mask(pd.Series([None, None], dtype="object"), provider="skillcorner")


def test_moved_symbols_not_reexported_from_ghost_gk():
    # TRUE clean break (review-2 MEDIUM): the moved names must live ONLY in _provider_visibility.
    # keeper_detection_mask consumes two of them, so a bare `from ._provider_visibility import ...`
    # in _ghost_gk would transitively re-export them -- the shim this move claims to avoid. A module
    # alias (`from . import _provider_visibility as _pv`) keeps them out of _ghost_gk's namespace.
    import silly_kicks.tracking._ghost_gk as gg

    for name in (
        "validate_provider",
        "assert_detection_aware_visibility",
        "_DETECTION_AWARE_PROVIDERS",
        "_FULLY_OBSERVED_PROVIDERS",
    ):
        assert not hasattr(gg, name), (
            f"{name} must live only in _provider_visibility, not re-export via _ghost_gk "
            "(single source; a bare import here re-introduces the transitive re-export)"
        )


# --- Task 2: Layer 1 -- materializer build-time guard ------------------------------------------


def _materializer_module():
    import scripts.materialize_tc3_frames as m

    return m


def test_guard_provider_frames_raises_on_all_null_visibility():
    m = _materializer_module()
    frames = pd.DataFrame({"visibility": pd.Series([None, None], dtype="object"), "x": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"tracking\.skillcorner"):
        m._guard_provider_frames(frames, "skillcorner")


def test_guard_provider_frames_raises_when_visibility_column_absent():
    # M2: a DROPPED column is a discarding regression too, not just a nulled one.
    m = _materializer_module()
    frames = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="visibility"):
        m._guard_provider_frames(frames, "skillcorner")


def test_guard_provider_frames_ok_on_native_visibility():
    m = _materializer_module()
    frames = pd.DataFrame({"visibility": [True, False, True], "x": [1.0, 2.0, 3.0]})
    assert m._guard_provider_frames(frames, "skillcorner") is None


def test_guard_provider_frames_noop_for_fully_observed():
    # Fully-observed provider: no-op even with no visibility column (returns before the column check).
    m = _materializer_module()
    frames = pd.DataFrame({"x": [1.0, 2.0]})
    assert m._guard_provider_frames(frames, "gradientsports") is None


def test_work_calls_guard_provider_frames():
    # Wiring: a module-level guard nobody calls is dead. AST-parse the materializer and require the
    # nested `_work` closure to call `_guard_provider_frames`.
    import ast
    from pathlib import Path

    m = _materializer_module()
    tree = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
    work = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_work")
    calls = {n.func.id for n in ast.walk(work) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_guard_provider_frames" in calls, "_work must call _guard_provider_frames before returning frames"
