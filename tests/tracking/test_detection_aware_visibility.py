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
from silly_kicks.tracking._provider_visibility import assert_detection_aware_visibility, detected_mask


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


# --- detected_mask: the general per-player detection primitive (spec 2026-09-27, ADR-109) -------
# `keeper_detection_mask`'s body is player-agnostic; it is extracted here as the shared seam that
# TF-58 (outfield) + the deferred GK-gate consume. `keeper_detection_mask` delegates, byte-identical.

_FULLY_OBSERVED = ("gradientsports", "sportec", "idsse", "metrica")


def test_detected_mask_general_per_row():
    # Player-agnostic: an arbitrary (e.g. outfield) visibility Series -> per-row mask; per-row null
    # -> False (matches keeper_detection_mask's fillna(False) doctrine).
    out = detected_mask(pd.Series([True, False, True, None]), provider="skillcorner")
    np.testing.assert_array_equal(out, np.array([True, False, True, False]))


def test_detected_mask_all_null_detection_aware_raises():
    with pytest.raises(ValueError, match=r"tracking\.skillcorner"):
        detected_mask(pd.Series([None, None], dtype="object"), provider="skillcorner")


def test_detected_mask_unknown_provider_raises():
    with pytest.raises(ValueError, match="unclassified provider"):
        detected_mask(pd.Series([True, False]), provider="bogus_provider")


@pytest.mark.parametrize("provider", _FULLY_OBSERVED)
def test_detected_mask_fully_observed_all_true(provider):
    # No detection flag exists -> all-True regardless of visibility content.
    out = detected_mask(pd.Series([True, False, None]), provider=provider)
    np.testing.assert_array_equal(out, np.array([True, True, True]))


def test_detected_mask_opt_out_known_provider_all_true():
    # assume_observed=True bypasses the detection verdict -> all-True (the ungated behaviour); this is
    # the ONLY escape, so the default (gate on) must differ.
    vis = pd.Series([True, False, True])
    np.testing.assert_array_equal(
        detected_mask(vis, provider="skillcorner", assume_observed=True), np.array([True, True, True])
    )
    # the default gates:
    np.testing.assert_array_equal(detected_mask(vis, provider="skillcorner"), np.array([True, False, True]))


def test_detected_mask_opt_out_unknown_provider_still_raises():
    # D1-SPEC-04: validate_provider runs FIRST; the opt-out bypasses only the detection verdict, NOT
    # provider classification -> a typo'd/unclassified provider RAISES even under assume_observed.
    with pytest.raises(ValueError, match="unclassified provider"):
        detected_mask(pd.Series([True, False]), provider="skillcornerr", assume_observed=True)


def test_detected_mask_non_vacuous():
    out = detected_mask(pd.Series([True, False, True, False]), provider="skillcorner")
    assert out.any() and not out.all()  # non-constant -- guards the vacuous-fixture trap


def test_keeper_detection_mask_delegates_to_detected_mask():
    # The parity contract: the wrapper == the primitive on the same input.
    vis = pd.Series([True, None, True, False])
    np.testing.assert_array_equal(
        keeper_detection_mask(vis, provider="skillcorner"), detected_mask(vis, provider="skillcorner")
    )


# --- TF-64: keeper-row wrappers (resolve_detected_keeper_xy / keeper_detected) + provider_of_frames
# The public seam the GK detection-gate choke points consume (spec 2026-10-01 §3.1 verdict set).

import math  # noqa: E402

from silly_kicks.tracking._provider_visibility import (  # noqa: E402
    detection_aware_provider,
    keeper_detected,
    resolve_detected_keeper_xy,
)


def _frame(provider, *, gk_vis: object = True, gk_present=True, n_keepers=1, gk_id=7, gk_xy=(5.0, 34.0)):
    """A one-frame fixture with real tracking dtypes: ball + one outfielder (always detected) + the
    defending keeper(s). ``gk_present=False`` drops the keeper row (undetected = MISSING row);
    ``gk_vis`` is the present keeper's visibility (False = extrapolated present row, None = discarded)."""
    # rows: (player_id, team_id, is_goalkeeper, is_ball, x, y, visibility)
    rows: list[tuple] = [
        (pd.NA, "A", False, True, 52.5, 34.0, True),  # ball
        (9, "A", False, False, 40.0, 30.0, True),  # one detected outfielder (visibility not all-null)
    ]
    if gk_present:
        rows.extend((gk_id + k, "B", True, False, gk_xy[0], gk_xy[1], gk_vis) for k in range(n_keepers))
    pid, team, is_gk, is_ball, xs, ys, vis = (list(col) for col in zip(*rows, strict=True))
    return pd.DataFrame(
        {
            "player_id": pd.array(pid, dtype="Int64"),
            "team_id": pd.Series(team, dtype="category"),
            "is_goalkeeper": pd.array(is_gk, dtype="boolean"),
            "is_ball": pd.array(is_ball, dtype="boolean"),
            "x": pd.array(xs, dtype="float32"),
            "y": pd.array(ys, dtype="float32"),
            "visibility": pd.array(vis, dtype="object"),
            "source_provider": pd.Series([provider] * len(pid), dtype="category"),
        }
    )


_KEEPER_GATE_OBSERVED = ("gradientsports", "sportec", "idsse", "metrica", "snapshot")


@pytest.mark.parametrize("provider", _KEEPER_GATE_OBSERVED)
def test_keeper_wrappers_observed_providers_return_position(provider):
    # Fully-observed + SB360 snapshot: never gated. visibility content is ignored; (x,y) returned.
    fr = _frame(provider, gk_vis=False)  # even a False flag is ignored on a non-detection-aware feed
    assert resolve_detected_keeper_xy(fr) == (5.0, 34.0)
    assert keeper_detected(fr, 7) is True


def test_keeper_wrappers_skillcorner_detected():
    fr = _frame("skillcorner", gk_vis=True)
    assert resolve_detected_keeper_xy(fr) == (5.0, 34.0)
    assert keeper_detected(fr, 7) is True


def test_keeper_wrappers_skillcorner_undetected_present_row_is_nan_false():
    # The E7 case: keeper row PRESENT but extrapolated (visibility False) -> honest-NaN / False.
    fr = _frame("skillcorner", gk_vis=False)
    x, y = resolve_detected_keeper_xy(fr)
    assert math.isnan(x) and math.isnan(y)
    assert keeper_detected(fr, 7) is False


def test_keeper_wrappers_absent_keeper_is_nan_none():
    fr = _frame("skillcorner", gk_present=False)
    x, y = resolve_detected_keeper_xy(fr)
    assert math.isnan(x) and math.isnan(y)
    assert keeper_detected(fr, 7) is None  # None != observed (fail-closed)


def test_keeper_wrappers_ambiguous_multi_keeper_is_nan_none():
    fr = _frame("skillcorner", n_keepers=2)
    x, y = resolve_detected_keeper_xy(fr)
    assert math.isnan(x) and math.isnan(y)
    # gk_id 7 is present-and-unique by player_id, but resolve uses is_goalkeeper (2 rows) -> NaN.
    assert keeper_detected(fr, 8) is True  # keeper_detected keys on the id


def test_keeper_wrappers_discarded_flag_present_null_raises():
    # A PRESENT keeper row with null visibility on skillcorner = discarded detection flag -> raise.
    fr = _frame("skillcorner", gk_vis=None)
    with pytest.raises(ValueError, match=r"tracking\.skillcorner"):
        resolve_detected_keeper_xy(fr)
    with pytest.raises(ValueError, match=r"tracking\.skillcorner"):
        keeper_detected(fr, 7)


def test_keeper_wrappers_unclassified_provider_is_observed_not_gated():
    # Unclassified source_provider (a synthetic test fixture, an un-wired feed) -> NOT gated: the
    # gate activates ONLY for a detection-aware provider. No raise at consume time; byte-identical to
    # pre-gate. (The fail-closed-on-unknown doctrine stays in detected_mask + the materializer guard.)
    fr = _frame("skillcorner", gk_vis=False)
    fr["source_provider"] = pd.Series(["synthetic"] * len(fr), dtype="object")
    assert resolve_detected_keeper_xy(fr) == (5.0, 34.0)
    assert keeper_detected(fr, 7) is True


def test_keeper_wrappers_opt_out_reproduces_observed():
    # assume_keeper_observed=True on an undetected skillcorner keeper -> observed (the general opt-out).
    fr = _frame("skillcorner", gk_vis=False)
    assert resolve_detected_keeper_xy(fr, assume_keeper_observed=True) == (5.0, 34.0)
    assert keeper_detected(fr, 7, assume_keeper_observed=True) is True


def test_detection_aware_provider_resolution():
    assert detection_aware_provider(_frame("skillcorner")) == "skillcorner"
    assert detection_aware_provider(_frame("sportec")) is None  # fully observed -> not gated
    assert detection_aware_provider(_frame("snapshot")) is None  # SB360 freeze frame -> not gated
    # mixed snapshot + skillcorner -> the detection-aware provider is surfaced
    mixed = pd.concat([_frame("skillcorner"), _frame("snapshot")], ignore_index=True)
    assert detection_aware_provider(mixed) == "skillcorner"
    # unclassified sentinel -> None (not gated), no raise
    syn = _frame("sportec")
    syn["source_provider"] = pd.Series(["synthetic"] * len(syn), dtype="object")
    assert detection_aware_provider(syn) is None
    # absent column -> None
    assert detection_aware_provider(pd.DataFrame({"x": [1.0]})) is None
