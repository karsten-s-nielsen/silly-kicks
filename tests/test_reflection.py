"""Tests for silly_kicks.reflection (ADR-045)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.reflection import (
    ATOMIC_SPADL_REFLECTION_KINDS,
    SPADL_REFLECTION_KINDS,
    TRACKING_REFLECTION_KINDS,
    reflect,
    reflect_columns,
)
from silly_kicks.reflection import GEOMETRIC_NAME as _GEOMETRIC_NAME


# --------------------------------------------------------------------------------------
# Task 1: registries declare the right kinds
# --------------------------------------------------------------------------------------
def test_registries_declare_the_kinds_that_matter():
    assert TRACKING_REFLECTION_KINDS["x"] == "point_x"
    assert TRACKING_REFLECTION_KINDS["y"] == "point_y"
    assert TRACKING_REFLECTION_KINDS["vx"] == "vector_x"
    assert TRACKING_REFLECTION_KINDS["vy"] == "vector_y"
    # speed is a MAGNITUDE -- reflecting it would be the inverse defect.
    assert TRACKING_REFLECTION_KINDS["speed"] == "magnitude"
    # z is invariant: a reflection in the pitch plane does not change height.
    assert TRACKING_REFLECTION_KINDS["z"] == "invariant"
    assert TRACKING_REFLECTION_KINDS["team_attacking_direction"] == "direction_label"
    # The columns the old API structurally could not express:
    assert TRACKING_REFLECTION_KINDS["x_smoothed"] == "point_x"
    assert TRACKING_REFLECTION_KINDS["y_smoothed"] == "point_y"
    # SPADL side, including the ADR-025 enrichment columns.
    assert SPADL_REFLECTION_KINDS["start_x"] == "point_x"
    assert SPADL_REFLECTION_KINDS["end_y"] == "point_y"
    assert SPADL_REFLECTION_KINDS["enriched_start_x"] == "point_x"


# --------------------------------------------------------------------------------------
# Task 2: reflect_columns
# --------------------------------------------------------------------------------------
def _frame():
    return pd.DataFrame(
        {
            "x": [10.0, 90.0],
            "y": [20.0, 60.0],
            "vx": [3.0, -4.0],
            "vy": [-1.0, 2.0],
            "speed": [np.hypot(3.0, 1.0), np.hypot(4.0, 2.0)],
        }
    )


def test_reflect_columns_applies_each_kind_correctly():
    df = _frame()
    mask = np.array([True, False])
    out = reflect_columns(df, mask, point_x=["x"], point_y=["y"], vector_x=["vx"], vector_y=["vy"])
    # masked row: point reflected, vector negated
    assert out.loc[0, "x"] == pytest.approx(95.0)
    assert out.loc[0, "y"] == pytest.approx(48.0)
    assert out.loc[0, "vx"] == pytest.approx(-3.0)
    assert out.loc[0, "vy"] == pytest.approx(1.0)
    # unmasked row: untouched
    assert out.loc[1, "x"] == pytest.approx(90.0)
    assert out.loc[1, "vx"] == pytest.approx(-4.0)
    # speed is a magnitude and was not listed -> unchanged on BOTH rows
    pd.testing.assert_series_equal(out["speed"], df["speed"])


def test_reflect_columns_is_pure():
    df = _frame()
    before = df.copy(deep=True)
    out = reflect_columns(df, np.array([True, True]), point_x=["x"])
    pd.testing.assert_frame_equal(df, before)  # ADR-033: no input mutation
    assert out is not df


# --------------------------------------------------------------------------------------
# Task 3: reflect -- registry-driven, warn default
# --------------------------------------------------------------------------------------
def test_reflect_uses_the_registry_and_covers_the_blind_spot_columns():
    df = pd.DataFrame(
        {
            "x": [10.0],
            "y": [20.0],
            "z": [1.5],
            "vx": [3.0],
            "vy": [-1.0],
            "speed": [np.hypot(3.0, 1.0)],
            "x_smoothed": [10.5],
            "y_smoothed": [20.5],
            "team_attacking_direction": ["ltr"],
            "player_id": ["p1"],
        }
    )
    out = reflect(df, np.array([True]), kinds=TRACKING_REFLECTION_KINDS)
    assert out.loc[0, "x"] == pytest.approx(95.0)
    assert out.loc[0, "vx"] == pytest.approx(-3.0)
    assert out.loc[0, "x_smoothed"] == pytest.approx(94.5)  # the D3b blind spot
    assert out.loc[0, "y_smoothed"] == pytest.approx(47.5)
    assert out.loc[0, "z"] == pytest.approx(1.5)  # height is invariant
    assert out.loc[0, "speed"] == pytest.approx(np.hypot(3.0, 1.0))  # magnitude
    assert out.loc[0, "team_attacking_direction"] == "rtl"
    assert out.loc[0, "player_id"] == "p1"


def test_reflect_warns_on_an_undeclared_GEOMETRIC_column():
    """ADR-045 section 4.5: an undeclared column is treated as invariant -- correct for a
    passenger -- but a GEOMETRY-shaped name is the suspicious case, so it warns."""
    from silly_kicks.reflection import UndeclaredGeometricColumnWarning

    df = pd.DataFrame({"x": [10.0], "mystery_x": [5.0]})
    with pytest.warns(UndeclaredGeometricColumnWarning, match="mystery_x"):
        out = reflect(df, np.array([True]), kinds=TRACKING_REFLECTION_KINDS)
    # Warned, but still treated as invariant -- the library must not guess a kind.
    assert out.loc[0, "mystery_x"] == pytest.approx(5.0)


def test_reflect_is_SILENT_on_an_undeclared_non_geometric_column():
    """The load-bearing half. `preserve_native` surfaces caller-chosen provider fields
    (spadl/utils.py:1651) whose names are unbounded BY CONSTRUCTION, and `invariant` is the
    CORRECT treatment for them. Warning here would be spam on a supported first-party
    feature, which is why the earlier `on_unknown="raise"` default was withdrawn."""
    df = pd.DataFrame({"x": [10.0], "possession": [7]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # ANY warning fails this test
        out = reflect(df, np.array([True]), kinds=TRACKING_REFLECTION_KINDS)
    assert out.loc[0, "possession"] == 7
    assert out.loc[0, "x"] == pytest.approx(95.0)


def test_reflect_escalates_to_an_error_via_the_warning_filter():
    """How a consumer that DOES control its column universe gets fail-closed."""
    from silly_kicks.reflection import UndeclaredGeometricColumnWarning

    df = pd.DataFrame({"x": [10.0], "mystery_x": [5.0]})
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UndeclaredGeometricColumnWarning)
        with pytest.raises(UndeclaredGeometricColumnWarning, match="mystery_x"):
            reflect(df, np.array([True]), kinds=TRACKING_REFLECTION_KINDS)


def test_reflect_on_unknown_raise_is_available_explicitly():
    """Retained as a greppable per-call opt-in; nothing in silly-kicks passes it."""
    df = pd.DataFrame({"x": [10.0], "possession": [7]})
    with pytest.raises(ValueError, match="possession"):
        reflect(df, np.array([True]), kinds=TRACKING_REFLECTION_KINDS, on_unknown="raise")


def test_reflect_extra_kinds_is_the_documented_escape_hatch():
    df = pd.DataFrame({"x": [10.0], "custom_vx": [2.0]})
    out = reflect(
        df,
        np.array([True]),
        kinds=TRACKING_REFLECTION_KINDS,
        extra_kinds={"custom_vx": "vector_x"},
    )
    assert out.loc[0, "custom_vx"] == pytest.approx(-2.0)


def test_extra_kinds_is_add_only_and_may_not_override_the_registry():
    """A call site must not be able to locally redefine a column's semantics."""
    df = pd.DataFrame({"x": [10.0]})
    with pytest.raises(ValueError, match="may not override"):
        reflect(
            df,
            np.array([True]),
            kinds=TRACKING_REFLECTION_KINDS,
            extra_kinds={"x": "invariant"},
        )


# --------------------------------------------------------------------------------------
# Task 4: registry-completeness meta-assertions + kind-plausibility gate
# --------------------------------------------------------------------------------------
def test_meta_every_known_tracking_column_declares_a_kind():
    """A new frame column must declare a reflection kind or this fails. Anti-rot."""
    from silly_kicks.tracking.schema import (
        GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS,
        KLOPPY_TRACKING_FRAMES_COLUMNS,
        TRACKING_FRAMES_COLUMNS,
    )

    known = (
        set(TRACKING_FRAMES_COLUMNS)
        | set(KLOPPY_TRACKING_FRAMES_COLUMNS)
        | set(GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS)
        # preprocess-added -- the columns that caused ADR-045 in the first place
        | {"vx", "vy", "x_smoothed", "y_smoothed", "_preprocessed_with"}
    )
    missing = sorted(known - set(TRACKING_REFLECTION_KINDS))
    assert not missing, (
        f"columns without a declared reflection kind: {missing}. Add them to "
        f"TRACKING_REFLECTION_KINDS (ADR-045) -- an undeclared column is exactly how "
        f"vx/vy went untransformed."
    )


def test_meta_every_known_spadl_column_declares_a_kind():
    """Built by UNION over the real constants, not a hardcoded list.

    This is the PRIMARY fail-closed mechanism in ADR-045 (section 4.5): runtime raising was
    withdrawn, so a column that is not declared here rides through as invariant with at most
    a warning. CI is therefore the only place that catches an undeclared LIBRARY-owned
    column -- and it catches it when the column is ADDED to a schema, not on the first
    production run that happens to reflect it. A hardcoded literal would stay green when a
    fifth provider variant is added -- exactly the rot this guards.
    """
    from silly_kicks.spadl import schema as S

    provider_variants = set().union(*(set(getattr(S, n)) for n in dir(S) if n.endswith("_SPADL_COLUMNS")))
    known = (
        provider_variants  # 14 canonical + 7 provider-specific
        | set(S.SPADL_NAME_COLUMNS)  # type_name / result_name / bodypart_name
        | {  # ADR-025 restart-coordinate enrichment
            "enriched_start_x",
            "enriched_start_y",
            "enriched_end_x",
            "enriched_end_y",
            "start_coord_source",
            "end_coord_source",
            "start_coord_confidence",
            "end_coord_confidence",
        }
    )
    missing = sorted(known - set(SPADL_REFLECTION_KINDS))
    assert not missing, (
        f"columns without a declared reflection kind: {missing}. Under on_unknown='raise' "
        f"these RAISE in production, they do not pass through."
    )
    assert len(known) == 32, f"expected the measured 32-column surface, got {len(known)}"


def test_meta_every_known_atomic_spadl_column_declares_a_kind():
    from silly_kicks.atomic.spadl import schema as A

    known = set(A.ATOMIC_SPADL_COLUMNS) | set(A.ATOMIC_SPADL_NAME_COLUMNS)
    missing = sorted(known - set(ATOMIC_SPADL_REFLECTION_KINDS))
    assert not missing, f"columns without a declared reflection kind: {missing}"
    assert len(known) == 15, f"expected the measured 15-column surface, got {len(known)}"
    # dx/dy are VECTORS -- this is the property the migration must not lose.
    assert ATOMIC_SPADL_REFLECTION_KINDS["dx"] == "vector_x"
    assert ATOMIC_SPADL_REFLECTION_KINDS["dy"] == "vector_y"


# Columns whose NAME reads geometric but whose kind legitimately is not. EMPTY, and measured
# so (2026-07-20): zero columns across the three registries are geometric-named AND declared
# invariant/magnitude, so there is nothing to exempt. The three meta-tests below stay armed:
# they guard against a FUTURE inert or stale entry rather than blessing any present one.
_JUSTIFIED_NON_GEOMETRIC: dict[str, str] = {}


def test_meta_no_geometric_looking_column_is_declared_inert():
    """Guards the failure mode: a geometric column declared invariant.

    Complete-by-enumeration idiom (small maintained dict, visible allowlist with reasons),
    NOT the AST lint ADR-043 deleted.
    """
    for registry in (
        TRACKING_REFLECTION_KINDS,
        SPADL_REFLECTION_KINDS,
        ATOMIC_SPADL_REFLECTION_KINDS,
    ):
        for col, kind in registry.items():
            if _GEOMETRIC_NAME.match(col) and kind in {"invariant", "magnitude"}:
                assert col in _JUSTIFIED_NON_GEOMETRIC, (
                    f"{col!r} is declared {kind!r} but its name reads geometric. Declare the "
                    f"real kind, or add a justification to _JUSTIFIED_NON_GEOMETRIC."
                )


def test_meta_the_plausibility_allowlist_is_not_stale():
    """Every exemption must still correspond to a live registry entry."""
    declared = set(TRACKING_REFLECTION_KINDS) | set(SPADL_REFLECTION_KINDS) | set(ATOMIC_SPADL_REFLECTION_KINDS)
    stale = sorted(set(_JUSTIFIED_NON_GEOMETRIC) - declared)
    assert not stale, f"exemptions for columns that no longer exist: {stale}"


def test_meta_the_plausibility_allowlist_actually_exempts_something():
    """An exemption for a name the pattern never matches is decoration."""
    inert = [c for c in _JUSTIFIED_NON_GEOMETRIC if not _GEOMETRIC_NAME.match(c)]
    assert not inert, (
        f"allowlist entries that the geometric pattern never matches: {inert}. They exempt "
        f"nothing -- delete them, or fix the pattern."
    )


def test_meta_the_plausibility_gate_would_actually_reject_a_bad_declaration():
    """BOTH-SIDES partner. The gate fires on nothing in the current registries (by design --
    nothing is mis-declared), so prove it CAN fire."""
    would_be_caught = {
        "ghost_gk_x": "invariant",  # suffix form
        "receiver_y": "magnitude",  # suffix form
        "x_smoothed": "invariant",  # prefix form -- a REAL registry column
        "dx": "invariant",  # atomic displacement vector
        "defensive_line_x": "invariant",  # derived geometry
    }
    for col, _kind in would_be_caught.items():
        assert _GEOMETRIC_NAME.match(col), f"{col!r} must be recognised as geometric or the gate cannot protect it"
        assert col not in _JUSTIFIED_NON_GEOMETRIC

    # And the inverse: genuinely non-geometric names must NOT trip it, or the gate is noise.
    for col in ("speed", "back_n_count", "lateral_width", "team_id", "_preprocessed_with"):
        assert not _GEOMETRIC_NAME.match(col), f"{col!r} is not geometric; pattern too broad"


# --------------------------------------------------------------------------------------
# Task 9c: D7 -- the two VAEP play_left_to_right helpers (in-place contract preserved)
# --------------------------------------------------------------------------------------
def test_vaep_play_left_to_right_mirrors_enrichment_and_stays_in_place():
    from silly_kicks.vaep.features import play_left_to_right as vaep_ltr

    df = pd.DataFrame(
        [
            {
                "game_id": 1,
                "action_id": 0,
                "period_id": 1,
                "time_seconds": 0.0,
                "team_id": 2,
                "player_id": 7,
                "start_x": 10.0,
                "start_y": 20.0,
                "end_x": 30.0,
                "end_y": 40.0,
                "enriched_start_x": 10.0,
                "enriched_start_y": 20.0,
                "enriched_end_x": 30.0,
                "enriched_end_y": 40.0,
                "type_id": 0,
                "result_id": 1,
                "bodypart_id": 0,
            }
        ]
    )
    states = [df]
    out = vaep_ltr(states, home_team_id=1)  # acting team is AWAY -> mirrored

    assert out[0].loc[0, "start_x"] == pytest.approx(95.0)
    assert out[0].loc[0, "enriched_start_x"] == pytest.approx(95.0)  # the latent trap
    # IN-PLACE contract preserved: the caller's own frame was mutated and returned.
    assert out[0] is df
    assert df.loc[0, "enriched_start_x"] == pytest.approx(95.0)
