"""The D2 / D3 reducers on REAL ``match_tables`` output (review A-03, round 2).

``match_tables`` melts every family table -- and the windows table -- into ONE long frame: an outer union in which
each table's rows carry every other table's columns as NaN. The reducers were only ever tested on single-table
stand-ins, so two faults hid there: the hypotheses' windows merge met a second ``attacking_team_id`` (a ``KeyError``
on the real frame, so D2's confirm and D3's reduce could not finish), and ``_metric_samples`` read a pair slice through
its foreign, all-NaN ``team_id`` (every reliability sample came out empty). These tests drive real ``match_tables``
output, from several synthetic matches, through both drivers' table split and their reducers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _fake_corpus import make_loaded  # noqa: E402

import scripts.calibrate_coordination as d2  # noqa: E402
import scripts.validate_team_coordination as d3  # noqa: E402
from scripts._coordination_corpus import match_tables  # noqa: E402
from scripts._coordination_hypotheses import evaluate_hypotheses  # noqa: E402
from silly_kicks.coordination import CoordinationParams  # noqa: E402
from tests.coordination._fixtures import make_coordination_actions, make_coordination_match  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore::silly_kicks.coordination.CoordinationCoverageWarning"),
    pytest.mark.filterwarnings("ignore:vx/vy columns not found"),
]

_SECONDS = 240.0
_POSSESSION_S = 12.0
# planted terminal actions (H3 compares possessions ending in a shot with those ending in a tackle). A tackle ends
# the possession BEFORE it (the regaining action), so each tackle sits two possessions after a shot: it then ends a
# pass possession, not the shot's own (a shot possession keeps the shot as its terminal action).
_SHOTS = (36.0, 84.0, 132.0, 180.0)
_TACKLES = (60.0, 108.0, 156.0, 204.0)


def _match(game: int, team_ids: tuple[int, int], seed: int):
    frames = make_coordination_match(
        seconds=_SECONDS, provider="sportec", periods=2, team_ids=team_ids, seed=seed, dead_intervals=[(100.0, 130.0)]
    )
    frames["game_id"] = game
    parts = []
    for period, fp in frames.groupby("period_id", sort=True):
        acts = make_coordination_actions(
            fp, restarts=[(t, "tackle") for t in _TACKLES], goals=_SHOTS, possession_every_s=_POSSESSION_S
        )
        acts["game_id"] = game
        acts["period_id"] = period
        parts.append(acts)
    actions = pd.concat(parts, ignore_index=True)
    actions["action_id"] = np.arange(len(actions))
    return make_loaded("sportec", str(game), frames=frames, actions=actions)


@pytest.fixture(scope="module")
def real_frame() -> pd.DataFrame:
    """Four matches (team 1 in all, teams 2 and 3 in two each) through the REAL match_tables, switch events in."""
    params = CoordinationParams.for_provider("sportec")
    matches = [_match(1, (1, 2), 11), _match(2, (1, 2), 12), _match(3, (1, 3), 13), _match(4, (1, 3), 14)]
    return pd.concat(
        [match_tables(m, params, n_surrogates=0, include_switch_events=True) for m in matches], ignore_index=True
    )


def test_fixture_preconditions(real_frame):
    # ADR-032: the union really is mixed (several tables + the windows table), and the planted terminal actions reached
    # the possession windows -- otherwise the tests below would pass vacuously.
    tables = set(real_frame["table"])
    assert {"windows", "pair", "pair_phase", "cluster_team", "spectral", "team_sync", "rsi"} <= tables
    assert {"rsi_switch_times", "possession_changes"} <= tables
    windows = real_frame[real_frame["table"] == "windows"]
    assert {"shot", "tackle"} <= set(windows["terminal_action"].dropna())


@pytest.mark.parametrize("split", [d3.split_tables, d2.split_tables], ids=["D3", "D2"])
def test_each_split_table_carries_only_its_own_columns(real_frame, split):
    # the outer union's foreign columns are gone: a table's slice has no column that is all-NaN there and absent from
    # its own schema -- in particular no pair slice carries windows' attacking_team_id or spectral's team_id.
    tables = split(real_frame)
    assert "attacking_team_id" not in tables["pair_phase"].columns
    assert "team_id" not in tables["pair"].columns
    assert "attacking_team_id" in tables["windows"].columns


@pytest.mark.parametrize("split", [d3.split_tables, d2.split_tables], ids=["D3", "D2"])
def test_hypotheses_run_on_real_match_tables(real_frame, split):
    results = evaluate_hypotheses(split(real_frame), seed=CoordinationParams().surrogate_seed)
    assert set(results) == {"H1", "H2", "H3", "H4", "H5", "H6", "H7"}
    assert np.isfinite(results["H3"]["p_anti_phase"]) and np.isfinite(results["H3"]["p_attack_phase"])
    assert np.isfinite(results["H5"]["p_x_gt_y"])


def test_d3_reliability_reads_every_table_with_data(real_frame):
    # A-09: the per-construct reduce (not the old per-family block) emits a construct for every table that has
    # finite metric data, and every construct carries per-(match, entity, half) samples with finite values.
    import _coordination_reliability as rel

    tables = d3.split_tables(real_frame)
    constructs = rel.derive_constructs(tables)
    emitted_tables = {c.table for c, _ in constructs}
    for _c, samples in constructs:
        assert samples["value"].notna().any()
        assert {"game_id", "entity", "period_id", "value"} <= set(samples.columns)
    for table in rel.CONSTRUCT_KEY_COLS:
        df = tables.get(table, pd.DataFrame())
        if len(df) and "window_kind" in df.columns:
            df = df[df["window_kind"] == "period"]  # the reliability unit: one match-half per entity
        scored = rel.reliability_scored_columns(table)
        has_data = any(
            len(df) and col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any() for col in scored
        )
        if has_data:
            assert table in emitted_tables, table


def test_split_tables_refuses_an_unknown_table():
    from scripts._coordination_corpus import split_tables

    frame = pd.DataFrame({"provider": ["sportec"], "match_id": ["1"], "variant": ["base"], "table": ["mystery"]})
    with pytest.raises(ValueError, match="unknown table 'mystery'"):
        split_tables(frame)


def test_switch_event_schemas_match_their_producers(real_frame):
    # drift guard: the declared H7 block schemas are exactly the producers' columns
    from scripts._coordination_corpus import POSSESSION_CHANGE_COLUMNS, RSI_SWITCH_COLUMNS, split_tables

    tables = split_tables(real_frame)
    lead = ["provider", "match_id", "variant", "table"]
    assert list(tables["rsi_switch_times"].columns) == [*lead, *RSI_SWITCH_COLUMNS]
    assert list(tables["possession_changes"].columns) == [*lead, *POSSESSION_CHANGE_COLUMNS]
    switches = real_frame[real_frame["table"] == "rsi_switch_times"]
    assert switches[list(RSI_SWITCH_COLUMNS)].notna().all().all()  # non-vacuity: the declared columns carry data


def _to_sorted_categorical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or isinstance(out[col].dtype, pd.StringDtype):
            cats = sorted(x for x in out[col].dropna().unique())
            out[col] = out[col].astype(pd.CategoricalDtype(categories=cats, ordered=False))
    return out


def test_d2_reliability_over_folds_byte_identical_under_categorical(real_frame):
    # D-5 (D2 leg): the D2 objective's held-out reliability is byte-identical on an object match_tables frame and
    # the same frame with string columns cast to a sorted CategoricalDtype -- exactly what the generalized
    # categorical combine (combine_levels categorical=True) feeds the D2 reduce; observed=True keeps it exact.
    from silly_kicks.calibration import match_cv_splits

    splits = match_cv_splits(d2._join_keys(real_frame))  # keys identical under categorical (astype(str))
    obj_mean, obj_folds = d2.reliability_over_folds(real_frame, d2.COORD_METHOD_FAMILIES, splits)
    cat_mean, cat_folds = d2.reliability_over_folds(
        _to_sorted_categorical(real_frame), d2.COORD_METHOD_FAMILIES, splits
    )

    def _eq(a: float, b: float) -> bool:
        return a == b or (np.isnan(a) and np.isnan(b))

    assert _eq(obj_mean, cat_mean)
    assert len(obj_folds) == len(cat_folds) and all(_eq(a, b) for a, b in zip(obj_folds, cat_folds, strict=True))
