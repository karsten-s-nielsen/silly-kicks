"""T2 -- apply_resolved_gk_geometry: OVERRIDE (not coalesce), stamp semantics, purity.

The GK-distribution domain's canonical SPADL coords are not trustworthy: Gradient Sports leaves
~60% of goal-kick origins NaN, and SkillCorner's native goal-kick origin is the broadcast BALL
detection -- PRESENT, finite, and ~10-20 m wrong. A coalesce fixes the first and silently misses
the second. See ADR-036.
"""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.xtgk import GK_GEOMETRY_SOURCES, apply_resolved_gk_geometry
from silly_kicks.xtgk._resolved_geometry import GK_GEOMETRY_SOURCE_COLUMN


def _frame():
    """Eight rows covering every stamp value reachable with the resolved columns present."""
    return pd.DataFrame(
        {
            "is_gk_distribution": [True, True, True, True, True, False, True, True],
            # row0 native (raw == resolved)     | row1 GS NaN origin rescued
            # row2 SC present-but-WRONG         | row3 dest-only override
            # row4 mixed: resolved origin + NaN dest -> `unresolved` wins (R3)
            # row5 off-domain
            # row6 (S3) GS unresolved-origin/good-dest: raw NaN origin AND resolved-NULL origin
            # row7 (S2) never attested: finite raw coords, ALL resolved coords NULL -> `unattested`
            "start_x": [5.5, np.nan, 25.0, 5.5, np.nan, 60.0, np.nan, 5.5],
            "start_y": [34.0, np.nan, 40.0, 34.0, np.nan, 20.0, np.nan, 34.0],
            "end_x": [40.0, 40.0, 40.0, 33.0, np.nan, 70.0, 40.0, 40.0],
            "end_y": [34.0, 34.0, 34.0, 30.0, np.nan, 20.0, 34.0, 34.0],
            "xt_gk_origin_x": [5.5, 5.5, 4.29, 5.5, 7.08, np.nan, np.nan, np.nan],
            "xt_gk_origin_y": [34.0, 34.0, 34.01, 34.0, 34.44, np.nan, np.nan, np.nan],
            "xt_gk_dest_x": [40.0, 40.0, 40.0, 35.0, np.nan, np.nan, 40.0, np.nan],
            "xt_gk_dest_y": [34.0, 34.0, 34.0, 31.0, np.nan, np.nan, 34.0, np.nan],
        }
    )


def test_every_emitted_stamp_is_a_declared_GK_GEOMETRY_SOURCE():
    """Contract: the emitted stamp vocabulary is EXACTLY `GK_GEOMETRY_SOURCES` -- the public constant
    consumers switch on. Guards an 8th value being emitted without being declared, and a declared
    value silently becoming unreachable."""
    base = set(apply_resolved_gk_geometry(_frame())[GK_GEOMETRY_SOURCE_COLUMN])
    assert base <= set(GK_GEOMETRY_SOURCES), f"undeclared stamp(s): {base - set(GK_GEOMETRY_SOURCES)}"

    # The base fixture reaches 6 of the 7; `resolved_both` needs both coords to change, which
    # test_resolved_both_when_origin_and_dest_change constructs. Together they cover all seven.
    both = _frame()
    both.loc[0, "xt_gk_origin_x"] = 6.0
    both.loc[0, "xt_gk_dest_x"] = 41.0
    covered = base | set(apply_resolved_gk_geometry(both)[GK_GEOMETRY_SOURCE_COLUMN])
    assert covered == set(GK_GEOMETRY_SOURCES), f"unreachable declared stamp(s): {set(GK_GEOMETRY_SOURCES) - covered}"


def test_override_not_coalesce_replaces_present_but_wrong_skillcorner_origin():
    """THE load-bearing case. Row 2's raw origin is PRESENT (25.0) and WRONG (broadcast ball);
    a coalesce would leave it. It must be REPLACED by the resolved keeper origin."""
    out = apply_resolved_gk_geometry(_frame())
    assert out.iloc[2]["start_x"] == pytest.approx(4.29)
    assert out.iloc[2]["start_y"] == pytest.approx(34.01)
    assert out.iloc[2][GK_GEOMETRY_SOURCE_COLUMN] == "resolved_origin"


def test_nan_origin_is_filled_from_resolved():
    out = apply_resolved_gk_geometry(_frame())
    assert out.iloc[1]["start_x"] == pytest.approx(5.5)
    assert out.iloc[1][GK_GEOMETRY_SOURCE_COLUMN] == "resolved_origin"


def test_native_row_unchanged_and_stamped_native():
    out = apply_resolved_gk_geometry(_frame())
    assert out.iloc[0]["start_x"] == pytest.approx(5.5)
    assert out.iloc[0][GK_GEOMETRY_SOURCE_COLUMN] == "native"


def test_dest_override_path_synthetic_real_data_is_a_noop():
    """Real cohorts never exercise this (measured: 0 rows differ) -- so it is tested synthetically."""
    out = apply_resolved_gk_geometry(_frame())
    assert out.iloc[3]["end_x"] == pytest.approx(35.0)
    assert out.iloc[3]["end_y"] == pytest.approx(31.0)
    assert out.iloc[3][GK_GEOMETRY_SOURCE_COLUMN] == "resolved_dest"


def test_unresolved_wins_precedence_on_mixed_row():
    """R3: row 4 has a RESOLVED origin but a still-NaN dest. The stamp answers 'will this row
    score?', so `unresolved` must win over `resolved_origin`."""
    out = apply_resolved_gk_geometry(_frame())
    assert out.iloc[4]["start_x"] == pytest.approx(7.08)  # origin WAS applied
    assert np.isnan(out.iloc[4]["end_x"])  # dest stays NaN
    assert out.iloc[4][GK_GEOMETRY_SOURCE_COLUMN] == "unresolved"


def test_off_domain_row_untouched():
    out = apply_resolved_gk_geometry(_frame())
    assert out.iloc[5]["start_x"] == pytest.approx(60.0)
    assert out.iloc[5][GK_GEOMETRY_SOURCE_COLUMN] == "off_domain"


def test_s3_unresolved_origin_with_good_dest_is_stamped_unresolved():
    """S3: raw NaN origin + resolved-NULL origin + finite dest (the GS `unresolved` shape). Today
    this only lands on `unresolved` via the FINAL precedence np.where -- a reorder would silently
    regress it to `native`/`resolved_dest` with no failing test. This is that test."""
    out = apply_resolved_gk_geometry(_frame())
    assert np.isnan(out.iloc[6]["start_x"])
    assert out.iloc[6][GK_GEOMETRY_SOURCE_COLUMN] == "unresolved"


def test_s2_never_attested_row_is_unattested_not_native():
    """S2: finite raw coords but ALL resolved coords NULL. Nothing attested this row, so stamping
    it `native` ("raw already equalled resolved") would be a lie AND would suppress the metric's
    warn-once."""
    out = apply_resolved_gk_geometry(_frame())
    assert out.iloc[7]["start_x"] == pytest.approx(5.5)  # untouched
    assert out.iloc[7][GK_GEOMETRY_SOURCE_COLUMN] == "unattested"


def test_resolved_both_when_origin_and_dest_change():
    df = _frame()
    df.loc[0, "xt_gk_origin_x"] = 6.0
    df.loc[0, "xt_gk_dest_x"] = 41.0
    out = apply_resolved_gk_geometry(df)
    assert out.iloc[0][GK_GEOMETRY_SOURCE_COLUMN] == "resolved_both"


def test_purity_input_never_mutated_and_new_object_returned():
    df = _frame()
    before = df.copy(deep=True)
    out = apply_resolved_gk_geometry(df)
    pd.testing.assert_frame_equal(df, before)
    assert out is not df


def test_missing_domain_column_raises():
    df = _frame().drop(columns=["is_gk_distribution"])
    with pytest.raises(ValueError, match="is_gk_distribution"):
        apply_resolved_gk_geometry(df)


def test_missing_resolved_columns_warns_noops_and_stamps_unattested_never_native():
    """R2: stamping `native` here would suppress the metric's warn-once while origins are still
    raw -- exactly the SkillCorner present-and-wrong hole the stamp exists to close."""
    df = _frame().drop(columns=["xt_gk_origin_x", "xt_gk_origin_y"])
    with pytest.warns(UserWarning, match="xt_gk_origin_x"):
        out = apply_resolved_gk_geometry(df)
    assert out.iloc[2]["start_x"] == pytest.approx(25.0)  # no-op: raw retained
    in_domain = out["is_gk_distribution"].to_numpy(dtype=bool)
    assert set(out.loc[in_domain, GK_GEOMETRY_SOURCE_COLUMN]) == {"unattested"}
    assert out.iloc[5][GK_GEOMETRY_SOURCE_COLUMN] == "off_domain"
