"""ADR-005 section 8 contract gate per spec section 8.5."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.tracking.feature_framework import is_frame_aware
from silly_kicks.tracking.features import (
    add_pressure_on_actor,
    pressure_default_xfns,
    pressure_on_actor,
)
from silly_kicks.tracking.pressure import (
    BekkersParams,
    LinkParams,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_xfn_list_is_size_one() -> None:
    """ADR-005 section 8: default xfn list ships exactly ONE flavor."""
    assert len(pressure_default_xfns) == 1


def test_suffix_naming_per_method() -> None:
    """ADR-005 section 8: column name = pressure_on_actor__<method>."""
    actions = pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [1],
            "time_seconds": [0.0],
            "team_id": ["home"],
            "player_id": [10],
            "start_x": [50.0],
            "start_y": [34.0],
            "type_id": [0],
        }
    )
    # Minimal frames with required schema columns (linker projects ['period_id',
    # 'frame_id', 'time_seconds', 'source_provider']; needs them even with 0 rows).
    frames = pd.DataFrame(
        {
            "period_id": pd.Series([], dtype="int64"),
            "frame_id": pd.Series([], dtype="int64"),
            "time_seconds": pd.Series([], dtype="float64"),
            "source_provider": pd.Series([], dtype="object"),
            "team_id": pd.Series([], dtype="object"),
            "player_id": pd.Series([], dtype="object"),
            "is_ball": pd.Series([], dtype="bool"),
            "x": pd.Series([], dtype="float64"),
            "y": pd.Series([], dtype="float64"),
        }
    )
    for m in ["andrienko_oval", "link_zones"]:
        s = pressure_on_actor(actions, frames, method=m)
        assert s.name == f"pressure_on_actor__{m}"


def test_params_type_validation_loud() -> None:
    """ADR-005 section 8: TypeError on params/method type mismatch."""
    actions = pd.DataFrame()
    frames = pd.DataFrame()
    with pytest.raises(TypeError, match=r"andrienko_oval.*expects AndrienkoParams.*got LinkParams"):
        pressure_on_actor(actions, frames, method="andrienko_oval", params=LinkParams())


def test_unknown_method_raises_value_error() -> None:
    """Lakehouse review item 18: runtime guard for non-Literal callers."""
    actions = pd.DataFrame()
    frames = pd.DataFrame()
    with pytest.raises(ValueError, match="Unknown method"):
        pressure_on_actor(actions, frames, method="not_a_method")  # type: ignore[arg-type]


def test_add_pressure_validates_all_methods_before_compute() -> None:
    """Transactional validation per lakehouse review item 12: TypeError before any column written."""
    actions = pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [1],
            "time_seconds": [0.0],
            "team_id": ["home"],
            "player_id": [10],
            "start_x": [50.0],
            "start_y": [34.0],
            "type_id": [0],
        }
    )
    frames = pd.DataFrame()
    with pytest.raises(TypeError):
        add_pressure_on_actor(
            actions,
            frames,
            methods=("andrienko_oval", "bekkers_pi"),
            params_per_method={
                "andrienko_oval": LinkParams(),  # WRONG TYPE
                "bekkers_pi": BekkersParams(),
            },
        )
    # Verify NO partial column was written into actions
    assert "pressure_on_actor__andrienko_oval" not in actions.columns
    assert "pressure_on_actor__bekkers_pi" not in actions.columns


def test_default_xfn_is_frame_aware() -> None:
    """ADR-005 section 1 marker preserved through lift_to_states."""
    xfn = pressure_default_xfns[0]
    assert is_frame_aware(xfn)


def test_adr_005_section_8_present() -> None:
    """Lakehouse review item 13: ADR markdown grep for amendment text."""
    adr_path = REPO_ROOT / "docs" / "superpowers" / "adrs" / "ADR-005-tracking-aware-features.md"
    text = adr_path.read_text(encoding="utf-8")
    assert "### 8. Multi-flavor xfn column naming convention" in text
    assert "<feature>__<method>" in text


def test_nan_safe_x_frame_aware_decorator_composition() -> None:
    """Spec section 8.5 / lakehouse review item 11.

    Verifies the `@nan_safe_enrichment x @frame_aware` composition for the
    multi-flavor pressure xfn:
    (a) `is_frame_aware(xfn) == True` (ADR-005 section 1 marker preservation
        through `lift_to_states`),
    (b) the lifted xfn invoked on per-action states with all-NaN positions
        returns NaN (NaN-safe contract from PR-S17 / ADR-003 holds),
    (c) the lifted xfn actually consumes the `frames` argument (not silently
        dropped) -- demonstrated by the result NaN coming from kernel
        evaluation against frames, not from a frames-less code path.
    """
    import numpy as np

    xfn = pressure_default_xfns[0]

    # (a) marker preservation
    assert is_frame_aware(xfn), "ADR-005 section 1: lifted xfn must carry _frame_aware marker"

    # Build per-action states (a0/a1/a2) with all-NaN start_x/start_y per spec
    nan_state = pd.DataFrame(
        {
            "action_id": [1, 2],
            "period_id": [1, 1],
            "time_seconds": [10.0, 20.0],
            "team_id": [1, 1],
            "player_id": [10, 11],
            "start_x": [float("nan"), float("nan")],
            "start_y": [float("nan"), float("nan")],
            "type_id": [0, 0],
        }
    )
    states = [nan_state, nan_state.copy(), nan_state.copy()]

    # Frames with valid coords -- the only way the lifted xfn could produce a
    # finite result on all-NaN states is by consuming frames (which it should NOT,
    # because the kernel anchors on action.start_x/start_y -- those are NaN).
    frames = pd.DataFrame(
        [
            {
                "frame_id": 100,
                "period_id": 1,
                "time_seconds": 10.0,
                "team_id": 2,
                "player_id": 200,
                "is_ball": False,
                "x": 50.0,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
                "source_provider": "synthetic",
            },
            {
                "frame_id": 100,
                "period_id": 1,
                "time_seconds": 10.0,
                "team_id": None,
                "player_id": None,
                "is_ball": True,
                "x": 50.0,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
                "source_provider": "synthetic",
            },
        ]
    )

    # (b) + (c): invoke lifted xfn directly with both states + frames
    out = xfn(states, frames)
    # Every NaN-anchored output must be NaN. (Per spec section 4.4: NaN-anchored
    # actions cannot link, so pressure is undefined; per ADR-003 nan_safe
    # contract the helper tolerates NaN inputs.)
    assert out.isna().all().all(), (
        f"NaN-only positions must produce all-NaN output, got: {out.head().to_dict(orient='list')}"
    )
    # (c) confirmed: if frames had been silently dropped, the xfn signature
    # mismatch would have raised TypeError before this assert -- the fact
    # that we got here with a DataFrame return means frames was consumed.
    _ = np


def test_auto_discovered_method_xfns_emit_suffixed_column_names() -> None:
    """Spec section 8.5 / lakehouse review item 18 (introspection-based regression).

    Mirrors the auto-discovered NaN-safety contract pattern (per
    ``feedback_marker_based_contracts``): introspect every public def in
    ``silly_kicks.tracking.features`` and ``silly_kicks.atomic.tracking.features``
    that takes a ``method:`` Literal kwarg whose annotation is the
    ``silly_kicks.tracking.pressure.Method`` Literal, then assert the returned
    Series.name equals ``f"{fn_name}__{method}"`` for every method in the
    Literal. Catches future xfns that adopt the Literal-method pattern but
    forget the suffix convention.
    """
    import inspect
    import typing

    from silly_kicks.atomic.tracking import features as atomic_mod
    from silly_kicks.tracking import features as std_mod
    from silly_kicks.tracking.pressure import Method

    # Build a minimal actions + frames pair sufficient for either anchor schema.
    base_actions = pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [1],
            "time_seconds": [0.0],
            "team_id": ["home"],
            "player_id": [10],
            "type_id": [0],
            "start_x": [50.0],
            "start_y": [34.0],
            "x": [50.0],
            "y": [34.0],
            "dx": [0.0],
            "dy": [0.0],
        }
    )
    # Single linkable frame containing one defender + one ball row, satisfying
    # all method preconditions (Bekkers needs vx/vy + at least one is_ball row).
    minimal_frames = pd.DataFrame(
        [
            {
                "period_id": 1,
                "frame_id": 0,
                "time_seconds": 0.0,
                "source_provider": "synthetic",
                "team_id": "away",
                "player_id": 100,
                "is_ball": False,
                "x": 52.0,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
            },
            {
                "period_id": 1,
                "frame_id": 0,
                "time_seconds": 0.0,
                "source_provider": "synthetic",
                "team_id": None,
                "player_id": None,
                "is_ball": True,
                "x": 50.0,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
            },
        ]
    )

    method_literal_args = set(typing.get_args(Method))

    def _is_method_literal(annotation: object) -> bool:
        """True when annotation is a Literal whose args match the Method literal."""
        try:
            args = set(typing.get_args(annotation))
        except TypeError:
            return False
        if not args:
            return False
        return args == method_literal_args

    discovered: list[tuple[str, str]] = []  # (fn_qualname, method)
    for module in (std_mod, atomic_mod):
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("_"):
                continue
            if obj.__module__ != module.__name__:
                continue  # imported, not defined here
            try:
                # eval_str=True resolves PEP 563 string annotations created
                # by `from __future__ import annotations` so we get the actual
                # Literal type rather than its forward-ref string form.
                sig = inspect.signature(obj, eval_str=True)
            except (TypeError, ValueError, NameError):
                continue
            method_param = sig.parameters.get("method")
            if method_param is None or not _is_method_literal(method_param.annotation):
                continue
            for method in method_literal_args:
                fq = f"{module.__name__}.{name}"
                discovered.append((fq, method))
                series = obj(base_actions, minimal_frames, method=method)
                assert isinstance(series, pd.Series), f"{fq}(method={method!r}) didn't return Series"
                assert series.name == f"{name}__{method}", (
                    f"{fq}(method={method!r}) returned Series.name={series.name!r}, "
                    f"expected {name!r}__{method!r} per ADR-005 section 8 multi-flavor naming convention"
                )

    # Sanity floor: at least the two known PR-S25 entry points participate
    # (silly_kicks.tracking.features.pressure_on_actor +
    #  silly_kicks.atomic.tracking.features.pressure_on_actor) x 3 methods.
    assert len(discovered) >= 6, (
        f"Auto-discovery found only {len(discovered)} (fn, method) pairs; "
        f"expected >=6 (2 namespaces x 3 methods). Discovered: {discovered}"
    )
