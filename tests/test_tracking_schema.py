"""Schema tests for silly_kicks.tracking — column set, dtype variants, dataclasses."""

import dataclasses

from silly_kicks.tracking.schema import (
    GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS,
    KLOPPY_TRACKING_FRAMES_COLUMNS,
    SPORTEC_TRACKING_FRAMES_COLUMNS,
    TRACKING_CATEGORICAL_DOMAINS,
    TRACKING_CONSTRAINTS,
    TRACKING_FRAMES_COLUMNS,
    LinkReport,
    TrackingConversionReport,
)


def test_tracking_frames_columns_is_20_columns():
    assert len(TRACKING_FRAMES_COLUMNS) == 20


def test_tracking_frames_columns_required_keys():
    expected = {
        "game_id",
        "period_id",
        "frame_id",
        "time_seconds",
        "frame_rate",
        "player_id",
        "team_id",
        "is_ball",
        "is_goalkeeper",
        "x",
        "y",
        "z",
        "speed",
        "speed_source",
        "ball_state",
        "team_attacking_direction",
        "confidence",
        "visibility",
        "source_provider",
        "is_goalkeeper_source",
    }
    assert set(TRACKING_FRAMES_COLUMNS) == expected


def test_kloppy_variant_overrides_identifiers_to_object():
    assert KLOPPY_TRACKING_FRAMES_COLUMNS["game_id"] == "object"
    assert KLOPPY_TRACKING_FRAMES_COLUMNS["player_id"] == "object"
    assert KLOPPY_TRACKING_FRAMES_COLUMNS["team_id"] == "object"
    for k, v in TRACKING_FRAMES_COLUMNS.items():
        if k not in {"game_id", "player_id", "team_id"}:
            assert KLOPPY_TRACKING_FRAMES_COLUMNS[k] == v


def test_sportec_variant_matches_kloppy_variant():
    assert SPORTEC_TRACKING_FRAMES_COLUMNS == KLOPPY_TRACKING_FRAMES_COLUMNS


def test_gradientsports_variant_uses_nullable_int64_identifiers():
    assert GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS["player_id"] == "Int64"
    assert GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS["team_id"] == "Int64"
    assert GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS["game_id"] == "int64"


def test_the_base_id_dtypes_can_actually_hold_the_ball_rows_NA():
    """A schema constant no producer can satisfy is a DEFAULT, not a contract.

    Every frame set carries a ball row, which belongs to no team and holds no player, so
    ``player_id``/``team_id`` are NA on it BY CONSTRUCTION. numpy ``int64`` cannot represent NA, so
    the base declaration raised ``IntCastingNaNError`` on every snapshot: ADR-055 measured exactly
    that and dropped its planned dtype pin as *unimplementable*, treating the failure as a property
    of the pin rather than of the declaration it was pinning to.

    It was the declaration. All five provider variants already overrode these two columns -- four
    to ``object``, Gradient Sports to ``Int64`` with a docstring that says "allows NaN on ball
    rows" -- so the base was satisfied by nothing and described one producer's happy path.
    """
    import pandas as pd

    ball_row_is_na = pd.DataFrame({"player_id": [7, pd.NA], "team_id": [1, pd.NA]})
    for col in ("player_id", "team_id"):
        # Must not raise. The assertion IS the cast. Routed through `pandas_dtype` because the
        # schema stores dtypes as `str` and pandas-stubs' `astype` overloads take literals.
        declared = pd.api.types.pandas_dtype(TRACKING_FRAMES_COLUMNS[col])
        cast = ball_row_is_na[col].astype(declared)
        assert cast.isna().iloc[-1], (
            f"{col} declared {TRACKING_FRAMES_COLUMNS[col]!r} silently turned the ball row's "
            f"absent id into a VALUE -- ADR-027 records a non-NA sentinel as a crash source in "
            f"downstream opponent guards, which is worse than the raise it replaced"
        )


def test_every_provider_variant_declares_id_dtypes_that_hold_NA():
    """The same requirement, across the whole declared surface -- complete by enumeration.

    Not a restatement of the test above: that one pins the BASE, this one pins that no variant
    reintroduces a non-nullable id. A future provider added with ``int64`` ids fails here.
    """
    import pandas as pd

    from silly_kicks.tracking import schema as S

    variants = {
        name: value
        for name, value in vars(S).items()
        if name.endswith("_TRACKING_FRAMES_COLUMNS") or name == "TRACKING_FRAMES_COLUMNS"
    }
    assert len(variants) >= 6, f"variant discovery found only {sorted(variants)}"

    for name, columns in sorted(variants.items()):
        for col in ("player_id", "team_id"):
            probe = pd.Series([1, pd.NA])
            try:
                probe.astype(columns[col])
            except (ValueError, TypeError) as exc:
                raise AssertionError(
                    f"{name}[{col!r}] is {columns[col]!r}, which cannot hold the ball row's NA: "
                    f"{type(exc).__name__}. Use a nullable dtype (Int64) or object."
                ) from exc


def test_tracking_constraints_keys_subset_of_columns():
    assert set(TRACKING_CONSTRAINTS) <= set(TRACKING_FRAMES_COLUMNS)


def test_tracking_constraints_x_y_match_spadl_field_dimensions():
    assert TRACKING_CONSTRAINTS["x"] == (0, 105.0)
    assert TRACKING_CONSTRAINTS["y"] == (0, 68.0)


def test_tracking_categorical_domains_keys_subset_of_columns():
    assert set(TRACKING_CATEGORICAL_DOMAINS) <= set(TRACKING_FRAMES_COLUMNS)


def test_tracking_categorical_domains_values():
    assert TRACKING_CATEGORICAL_DOMAINS["ball_state"] == frozenset({"alive", "dead"})
    assert TRACKING_CATEGORICAL_DOMAINS["team_attacking_direction"] == frozenset({"ltr", "rtl"})
    # "unavailable" (ADR-043) declares kinematics STRUCTURALLY absent for the frame source
    # -- distinct from a NULL speed_source, which only means "not derived yet".
    assert TRACKING_CATEGORICAL_DOMAINS["speed_source"] == frozenset({"native", "derived", "unavailable"})
    assert TRACKING_CATEGORICAL_DOMAINS["source_provider"] == frozenset(
        {"gradientsports", "sportec", "metrica", "skillcorner", "snapshot"}
    )


def test_tracking_conversion_report_is_frozen_dataclass():
    r = TrackingConversionReport(
        provider="gradientsports",
        total_input_frames=100,
        total_output_rows=2200,
        n_periods=2,
        frame_coverage_per_period={1: 1.0, 2: 0.99},
        ball_out_seconds_per_period={1: 12.4, 2: 8.7},
        nan_rate_per_column={"z": 0.95, "speed": 0.0},
        derived_speed_rows=0,
        unrecognized_player_ids=set(),
    )
    assert dataclasses.is_dataclass(r) and r.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_tracking_conversion_report_has_unrecognized():
    r1 = TrackingConversionReport("gradientsports", 0, 0, 0, {}, {}, {}, 0, set())
    assert r1.has_unrecognized is False
    r2 = TrackingConversionReport("gradientsports", 0, 0, 0, {}, {}, {}, 0, {123})
    assert r2.has_unrecognized is True


def test_link_report_link_rate_zero_when_empty():
    r = LinkReport(
        n_actions_in=0,
        n_actions_linked=0,
        n_actions_unlinked=0,
        n_actions_multi_candidate=0,
        per_provider_link_rate={},
        max_time_offset_seconds=0.0,
        tolerance_seconds=0.2,
    )
    assert r.link_rate == 0.0


def test_link_report_link_rate_nonzero():
    r = LinkReport(
        n_actions_in=100,
        n_actions_linked=95,
        n_actions_unlinked=5,
        n_actions_multi_candidate=10,
        per_provider_link_rate={"gradientsports": 0.95},
        max_time_offset_seconds=0.18,
        tolerance_seconds=0.2,
    )
    assert r.link_rate == 0.95
