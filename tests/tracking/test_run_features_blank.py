"""The producer's pointer drop and blank step (action-time repair spec section 5.3).

``run_tracking_features`` takes no frame for a ``"placeholder"`` or ``"order_only"`` row: it drops the row's frame
pointer before the families run, and after they run it writes the missing value of each family-added column's dtype
on such a row (a frame window that slices by the action time, as the off-ball runs do, would otherwise write 0
runners). The other rows are unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import run_tracking_features
from silly_kicks.xthreat import ExpectedThreat
from tests.sb360._fixture import HOME_TEAM_ID, build_leg_a

_POINTER_COLUMNS = ("frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score")
# stamped by the producer's keeper-identity step before the families run (a resolver, not a frame window)
_IDENTITY_COLUMNS = ("defending_gk_player_id", "defending_gk_team_id")
_ROWS = {1: "placeholder", 4: "order_only"}  # a cross and a dribble of the fixture, as restart-like rows


def _run(mark: bool):
    actions, frames, links = build_leg_a()
    actions = actions.assign(xg=np.where(actions["type_name"] == "shot", 0.2, np.nan))
    if mark:
        ts = actions["time_source"].to_numpy(dtype=object).copy()
        for pos, src in _ROWS.items():
            ts[pos] = src
        actions = actions.assign(time_source=ts)
    xt = ExpectedThreat()
    xt.xT = np.full((xt.w, xt.l), 0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, report = run_tracking_features(
            actions, frames, links=links, xt=xt, xg_column="xg", home_team_id=HOME_TEAM_ID
        )
    added = [c for c in out.columns if c not in actions.columns]
    return actions, out, report, added


@pytest.fixture(scope="module")
def runs():
    return _run(mark=False), _run(mark=True)


def test_fixture_runs_the_families_and_has_values_on_the_rows(runs):
    (_a, out, report, added), _ = runs
    assert sum(v == "ran" for v in report.family_status.values()) >= 30  # non-vacuity: the default families ran
    family_cols = [c for c in added if c not in _POINTER_COLUMNS + _IDENTITY_COLUMNS]
    on_rows = out.iloc[list(_ROWS)][family_cols]
    assert on_rows.notna().any(axis=None)  # non-vacuity: the unmarked run has values on these rows


def test_every_family_column_is_missing_on_placeholder_and_order_only_rows(runs):
    _, (_a, out, _r, added) = runs
    family_cols = [c for c in added if c not in _POINTER_COLUMNS + _IDENTITY_COLUMNS]
    still = {c: out[c].iloc[list(_ROWS)].tolist() for c in family_cols if out[c].iloc[list(_ROWS)].notna().any()}
    assert still == {}


def test_off_ball_runner_counts_are_missing_not_zero(runs):
    (_a, plain, _r, added), (_b, out, _s, _) = runs
    runner_cols = [c for c in added if "run" in c and c.startswith("n_")]
    assert runner_cols  # the counts exist (non-vacuity)
    assert plain[runner_cols].iloc[list(_ROWS)].notna().any(axis=None)  # the unmarked run writes counts there
    assert out[runner_cols].iloc[list(_ROWS)].isna().all(axis=None)


def test_the_keeper_identity_keeps_its_value_on_those_rows(runs):
    (_a, plain, _r, _added), (_b, out, _s, _) = runs
    rows = list(_ROWS)
    for col in _IDENTITY_COLUMNS:
        assert plain[col].iloc[rows].notna().all()  # non-vacuity: the resolver stamped a keeper
        assert out[col].iloc[rows].tolist() == plain[col].iloc[rows].tolist()


def test_the_pointer_is_dropped_on_those_rows(runs):
    _, (_a, out, _r, _added) = runs
    rows = out.iloc[list(_ROWS)]
    assert rows["frame_id"].isna().all()
    assert rows["time_offset_seconds"].isna().all() and rows["link_quality_score"].isna().all()
    assert (rows["n_candidate_frames"] == 0).all()  # no candidate frame was offered


def test_other_rows_are_unchanged(runs):
    (_a, plain, _r, added), (_b, out, _s, _) = runs
    others = [i for i in range(len(out)) if i not in _ROWS]
    family_cols = [c for c in added if c not in _POINTER_COLUMNS + _IDENTITY_COLUMNS]
    pd.testing.assert_frame_equal(
        out.iloc[others][family_cols].reset_index(drop=True),
        plain.iloc[others][family_cols].reset_index(drop=True),
        check_dtype=False,  # a blanked numpy-int family column cannot exist (STOP rule); nullable upcasts can
    )
