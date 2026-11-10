"""HFU spec 17.4 (F32, owner-ratified 2026-11-09): a census rebaseline passes within a stated tolerance.

The fresh census is the REAL committed census plus the differences launch 6 measured (`tier2.icc` 3.1e-15 relative,
`tier2.lo` +4.52e-6, `n_boot_effective` +1 (launch 6: 399 -> 400), the P4 pass counters of a run, every value of one
ranking column 1.27e-14 relative: launch 6 changed 4,094 of them); each mutation beyond a bound must fail. The real
launch-6 census (staged outside the repo, so not readable in CI) passed the same check: ledger Ruling, A-IMPL-17.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.hf_jobs import adopt as A
from scripts.hf_jobs.registry import DRIVERS

COMMITTED = Path("docs/research/territory_ranking_census")
SPEC = DRIVERS["build_territory_ranking_census"]


def _tolerance() -> Mapping[str, Any]:
    assert SPEC.tolerance is not None, "the census DriverSpec declares its tolerance (HFU spec 17.4)"
    return SPEC.tolerance


TOL = _tolerance()


COL = "territory_xt_prevented_above_expectation"


def _fresh(tmp_path: Path, *, committed: Path = COMMITTED, edit=None, rank_edit=None) -> Path:
    fresh = tmp_path / "fresh"
    shutil.copytree(committed, fresh)
    doc = json.loads((fresh / "census.json").read_text(encoding="utf-8"))
    t2 = doc["census"]["tier2"]
    t2["icc"] *= 1 + 3.1e-15
    t2["lo"] += 4.52e-6
    t2["n_boot_effective"] += 1  # relative to the committed census, adopted or not (HFU spec 18.7)
    doc |= {"n_attempted": 1981, "n_failed": 0, "n_counters_unrecorded": 0, "n_excluded": 0, "generation": "x"}
    if edit is not None:
        edit(doc)
    (fresh / "census.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    rank = pd.read_parquet(fresh / "ranking.parquet")
    rank[COL] = rank[COL].to_numpy(copy=True) * (1 + 1.27e-14)
    if rank_edit is not None:
        rank_edit(rank)
    rank.to_parquet(fresh / "ranking.parquet", index=False)
    return fresh


def test_the_launch6_differences_are_within_tolerance(tmp_path):
    ok, notes = A._within_tolerance(COMMITTED, _fresh(tmp_path), TOL)
    assert ok, notes
    # One summary note per parquet column, not one per value (A-IMPL-18 / B C2).
    ranking = [n for n in notes if n.startswith("ranking.parquet:")]
    assert len(ranking) == 1 and ranking[0].startswith(f"ranking.parquet:{COL}: ") and "max rel" in ranking[0], ranking


def _lo(doc):
    doc["census"]["tier2"]["lo"] += 1e-4


def _boot(doc):
    doc["census"]["tier2"]["n_boot_effective"] += 1  # +2 in all


def _reason(doc):
    doc["verdict"]["reason"] = doc["verdict"]["reason"].replace("lo=0.1211", "lo=0.1212")


def _failed(doc):
    doc["n_failed"] = 1


def _nan(doc):
    doc["census"]["tier2"]["icc"] = float("nan")


def _added(doc):
    doc["census"]["tier2"]["extra"] = 1


def _missing(doc):
    del doc["census"]["tier2"]["power"]


def _row(rank):
    rank.loc[0, "player_id"] = int(rank.loc[0, "player_id"]) + 1


def _value(rank):
    rank.loc[5, COL] = float(rank[COL].to_numpy()[5]) * (1 + 1e-6)


def _value_nan(rank):
    rank.loc[5, COL] = float("nan")


@pytest.mark.parametrize(
    ("edit", "rank_edit"),
    [
        (_lo, None),
        (_boot, None),
        (_reason, None),
        (_failed, None),
        (_nan, None),
        (_added, None),
        (_missing, None),
        (None, _row),
        (None, _value),
        (None, _value_nan),
    ],
    ids=[
        "lo+1e-4",
        "n_boot_effective+2",
        "verdict reason",
        "fresh n_failed 1",
        "json NaN vs number",
        "json key added",
        "json key missing",
        "ranking row",
        "ranking value 1e-6 rel",
        "ranking NaN vs number",
    ],
)
def test_a_difference_beyond_the_bounds_fails(tmp_path, edit, rank_edit):
    ok, notes = A._within_tolerance(COMMITTED, _fresh(tmp_path, edit=edit, rank_edit=rank_edit), TOL)
    assert not ok
    assert len(notes) <= 3, notes  # a failing parquet column is one note, not one per value


def test_a_value_beyond_the_bound_names_its_first_row(tmp_path):
    _ok, notes = A._within_tolerance(COMMITTED, _fresh(tmp_path, rank_edit=_value), TOL)
    assert notes == [notes[0]] and notes[0].startswith(f"ranking.parquet:{COL}: 1 of ") and "first row 5" in notes[0]


def test_nan_equals_nan_at_the_same_field(tmp_path):
    committed = _fresh(tmp_path / "c", edit=_nan, rank_edit=_value_nan)  # NaN in census.json and ranking.parquet
    ok, notes = A._within_tolerance(committed, _fresh(tmp_path, committed=committed), TOL)
    assert ok, notes


def _inf_doc(v: float):
    def edit(doc):
        doc["census"]["tier2"]["icc"] = v

    return edit


def _inf_rank(v: float):
    def edit(rank):
        rank.loc[5, COL] = v

    return edit


def test_equal_infinities_are_equal(tmp_path):
    inf = float("inf")
    committed = _fresh(tmp_path / "c", edit=_inf_doc(inf), rank_edit=_inf_rank(inf))
    ok, notes = A._within_tolerance(committed, _fresh(tmp_path, committed=committed), TOL)
    assert ok, notes


@pytest.mark.parametrize("fresh_value", [-float("inf"), 1.0], ids=["-inf", "a number"])
@pytest.mark.parametrize("where", ["json", "parquet"])
def test_an_infinity_against_another_value_fails(tmp_path, fresh_value, where):
    inf = float("inf")
    committed = _fresh(tmp_path / "c", edit=_inf_doc(inf), rank_edit=_inf_rank(inf))
    edit = {"edit": _inf_doc(fresh_value)} if where == "json" else {"rank_edit": _inf_rank(fresh_value)}
    ok, notes = A._within_tolerance(committed, _fresh(tmp_path, committed=committed, **edit), TOL)
    assert not ok and notes == [notes[0]] and ("tier2.icc" if where == "json" else COL) in notes[0], notes


@pytest.mark.parametrize("where", ["json", "parquet"])
def test_a_fresh_infinity_against_a_finite_committed_value_fails(tmp_path, where):
    """The overflow shape (HFURR-IMPL-24): the committed value is finite, the fresh run's is +inf."""
    inf = float("inf")
    fresh = _fresh(tmp_path, edit=_inf_doc(inf)) if where == "json" else _fresh(tmp_path, rank_edit=_inf_rank(inf))
    ok, notes = A._within_tolerance(COMMITTED, fresh, TOL)
    assert not ok and notes == [notes[0]] and ("tier2.icc" if where == "json" else COL) in notes[0], notes


def test_a_code_delta_rebaseline_within_tolerance_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "_code_delta", lambda *a, **k: ["scripts/_crossed_icc.py"])
    res = A._rebaseline(SPEC, _fresh(tmp_path), "0" * 40, run=lambda *a, **k: None)
    assert res["ok"] and res["within_tolerance"] and res["delta"] == "code" and not res["identical"]
