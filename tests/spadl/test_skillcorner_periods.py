"""The SkillCorner GI period-label repair (TF-46 Arm 1 Part 2 spec section 26; DL ruling 2026-10-25, Q8).

Real data first: all 20 committed opendata fixtures are a no-op pinned against the ``1007ef6`` conversion (golden
digests), and each is mislabelled in both boundary directions and restored byte-identically. Guards G0-G4 are
tested one-sided and two-sided; the OBE rule, the shape, the ``match_periods`` diagnostic, single-sourcing with
``parse_passing_options`` and the provenance registries close the list. Every repair test fails at ``1007ef6``.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.providers.skillcorner import parse_passing_options
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import skillcorner
from silly_kicks.spadl._skillcorner_periods import (
    PERIOD_REPAIR_STATUS_VALUES,
    GiPeriodRepair,
    GiPeriodRepairWarning,
    _clock_seconds,
    repair_gi_periods,
)
from silly_kicks.spadl.schema import SkillCornerConversionReport
from tests.spadl._skillcorner_period_support import (
    GOLDEN,
    _block,
    _clock_ds,
    _fmt_ds,
    canonical_digests,
    implied_starts,
    load_opendata,
    mislabel_p1_to_p2,
    mislabel_p2_to_p1,
    opendata_ids,
)

_IDS = opendata_ids()
_GOLDEN = json.loads(GOLDEN.read_text(encoding="utf-8"))["matches"]
_BASE = "1886347"  # the guard / OBE / shape fixtures' base match (a committed opendata fixture)


def _as_int(v) -> int:
    """pandas Scalar -> int (the untyped arg sidesteps the stub friction)."""
    return int(v)


def _convert(events: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, SkillCornerConversionReport]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", GiPeriodRepairWarning)
        return skillcorner.convert_to_actions(events, meta)


def _no_period_source(actions: pd.DataFrame) -> dict[str, str]:
    return canonical_digests(actions.drop(columns=["period_source"]))


def _identity_repair(events: pd.DataFrame, *, match_periods=None) -> GiPeriodRepair:
    """The repair that changes nothing: the converter's pre-repair behaviour (the "unrepaired conversion")."""
    idx = events.index
    return GiPeriodRepair(
        period=events["period"].copy(),
        clock=_clock_seconds(events),
        period_source=pd.Series("native", index=idx, dtype=object),
        obe_conflict=pd.Series(False, index=idx, dtype=bool),
        status="not_evaluable",
        relabelled={},
        shape=None,
        start_frames={},
        period_start_mismatch=False,
    )


def _unrepaired(events: pd.DataFrame, meta: dict, monkeypatch) -> pd.DataFrame:
    with monkeypatch.context() as mp:
        mp.setattr(skillcorner, "repair_gi_periods", _identity_repair)
        return skillcorner.convert_to_actions(events, meta)[0]


def _frame_rows(actions: pd.DataFrame, planted_pp: set[str]) -> np.ndarray:
    """The output rows whose period comes from a planted possession: a row backed by a GI possession by its
    ``gi_event_id`` (a derived restart's period is its MARKER possession's, while its ``original_event_id`` names the
    receiver), a row without one (inferred defensive / goals rows) by its ``original_event_id``, and a dribble
    preceding such a row."""
    gi = actions["gi_event_id"]
    own = (gi.astype(str).isin(planted_pp) & gi.notna()) | (
        gi.isna() & actions["original_event_id"].astype(str).isin(planted_pp)
    )
    # a keeper save takes the period of the possession after the shot (``infer_keeper_saves``: next_row, at its
    # time - 0.01) and carries no event id
    native = actions[own.to_numpy(dtype=bool) & (actions["action_provenance"] == "native").to_numpy()]
    planted_at = set(
        zip(native["period_id"].astype(int), np.round(native["time_seconds"].astype(float) - 0.01, 6), strict=True)
    )
    save = (actions["type_id"] == spadlconfig.actiontype_id["keeper_save"]) & actions["original_event_id"].isna()
    save_from_planted = [
        bool(s) and (int(p), round(float(t), 6)) in planted_at
        for s, p, t in zip(save, actions["period_id"], actions["time_seconds"], strict=True)
    ]
    own = own.to_numpy(dtype=bool) | np.array(save_from_planted, dtype=bool)
    nxt = np.append(own[1:], False)
    dribble = (actions["type_id"] == spadlconfig.actiontype_id["dribble"]).to_numpy()
    return own | (dribble & nxt)


def _assert_single_sourced(events: pd.DataFrame, actions: pd.DataFrame, game_id: str) -> None:
    options = parse_passing_options(events, game_id=game_id)
    native = actions[actions["action_provenance"] == "native"].drop_duplicates("original_event_id")
    by_eid = native.set_index(native["original_event_id"].astype(str))
    linked = options[options["decision_id"].isin(by_eid.index)]
    assert len(linked) > 0
    assert (
        linked["period_id"].astype(int).to_numpy() == by_eid.loc[linked["decision_id"], "period_id"].to_numpy()
    ).all()
    assert (linked["period_source"].to_numpy() == by_eid.loc[linked["decision_id"], "period_source"].to_numpy()).all()


# ---------------------------------------------------------------------------
# (1) Real-data no-op, pinned against the 1007ef6 conversion (all 20 opendata fixtures)
# ---------------------------------------------------------------------------


def test_golden_covers_every_opendata_fixture():
    assert sorted(_GOLDEN) == _IDS and len(_IDS) == 20


@pytest.mark.parametrize("mid", _IDS)
def test_opendata_conversion_is_a_no_op_against_the_1007ef6_golden(mid):
    events, meta = load_opendata(mid)
    actions, report = _convert(events, meta)  # a GiPeriodRepairWarning would raise here
    assert _no_period_source(actions) == _GOLDEN[mid]["actions"]
    assert list(actions.columns) == [*_GOLDEN[mid]["action_columns"], "period_source"]  # the order too (appended last)
    assert set(actions["period_source"]) == {"native"}
    assert report.period_repair_status == "clean" and report.period_repair_shape is None
    assert report.period_relabelled == {"player_possession": 0, "passing_option": 0, "off_ball_run": 0}
    assert report.obe_period_conflict == 0 and report.period_start_mismatch is False
    options = parse_passing_options(events, game_id=mid)
    assert canonical_digests(options.drop(columns=["period_source"])) == _GOLDEN[mid]["options"]
    assert list(options.columns) == [*_GOLDEN[mid]["option_columns"], "period_source"]
    assert set(options["period_source"]) == {"native"}


# ---------------------------------------------------------------------------
# (2) Mislabel-and-restore, both boundary directions (all 20), + (8) single-sourcing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plant", [mislabel_p2_to_p1, mislabel_p1_to_p2], ids=["p2_to_p1", "p1_to_p2"])
@pytest.mark.parametrize("mid", _IDS)
def test_mislabelled_boundary_block_is_restored_byte_identically(mid, plant):
    events, meta = load_opendata(mid)
    clean, _ = _convert(events, meta)
    bad, rows = plant(events)
    assert not bad.equals(events)  # the plant changed something
    repaired, report = _convert(bad, meta)  # boundary_block: no warning
    assert _no_period_source(repaired) == _no_period_source(clean)
    planted_pp = set(bad.loc[rows][bad.loc[rows, "event_type"] == "player_possession"]["event_id"].astype(str))
    assert (repaired["period_source"].to_numpy() == "frame").tolist() == _frame_rows(repaired, planted_pp).tolist()
    assert report.period_repair_status == "repaired" and report.period_repair_shape == "boundary_block"
    n_linked = int(bad.loc[rows, "event_type"].isin(["passing_option", "off_ball_run"]).sum())
    assert sum(report.period_relabelled.values()) == len(rows) and report.period_relabelled["player_possession"] == 1
    assert report.period_relabelled["passing_option"] + report.period_relabelled["off_ball_run"] == n_linked
    _assert_single_sourced(bad, repaired, mid)
    options = parse_passing_options(bad, game_id=mid)
    assert canonical_digests(options.drop(columns=["period_source"])) == _GOLDEN[mid]["options"]


# ---------------------------------------------------------------------------
# (3) The re-base is pinned: a relabel without it is wrong
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mid", _IDS)
def test_a_relabel_without_the_clock_rebase_misplaces_the_kick_off(mid):
    events, meta = load_opendata(mid)
    clean, _ = _convert(events, meta)
    bad, rows = mislabel_p2_to_p1(events)
    naive = bad.copy()
    naive.loc[rows, "period"] = 2  # the label only: the clock stays on the period-1 basis
    # the converter refuses to repair the naive input: its clock is off the identity under the new label (G2)
    with pytest.warns(GiPeriodRepairWarning, match="guard G2"):
        naive_actions, _ = skillcorner.convert_to_actions(naive, meta)
    s = implied_starts(events)
    eid = str(bad.loc[rows][bad.loc[rows, "event_type"] == "player_possession"]["event_id"].iloc[0])

    def kick_off_time(actions):
        native = actions[(actions["original_event_id"].astype(str) == eid) & (actions["action_provenance"] == "native")]
        return float(native["time_seconds"].iloc[0])

    stoppage = (s[2] - s[1]) / 10.0 - 2700.0  # the period-1 overrun the naive clock carries into period 2
    assert stoppage > 0
    assert kick_off_time(naive_actions) == pytest.approx(kick_off_time(clean) + stoppage)
    assert _no_period_source(naive_actions) != _no_period_source(clean)  # the naive conversion differs
    repaired, report = _convert(bad, meta)
    assert kick_off_time(repaired) == kick_off_time(clean)  # exact
    assert report.restart_counts == skillcorner.convert_to_actions(events, meta)[1].restart_counts


# ---------------------------------------------------------------------------
# (4) Guards G0-G4, one-sided and two-sided
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base():
    events, meta = load_opendata(_BASE)
    bad, rows = mislabel_p2_to_p1(events)
    return events, meta, bad, rows


@pytest.fixture(scope="module")
def base_linked():
    """The first opendata match whose planted boundary block carries a passing option (the G4 tests need one)."""
    for mid in _IDS:
        events, meta = load_opendata(mid)
        bad, rows = mislabel_p2_to_p1(events)
        if (bad.loc[rows, "event_type"] == "passing_option").any():
            return events, meta, bad, rows
    pytest.fail("no opendata boundary block with a passing option")


def _assert_skipped(bad, meta, status, monkeypatch):
    with pytest.warns(GiPeriodRepairWarning, match=f"guard {status[-2:].upper()}"):
        actions, report = skillcorner.convert_to_actions(bad, meta)
    assert report.period_repair_status == status
    assert canonical_digests(actions) == canonical_digests(_unrepaired(bad, meta, monkeypatch))
    assert set(actions["period_source"]) == {"native"}


def test_g0_without_the_gi_frame_columns_is_not_evaluable_without_a_warning(base, recwarn):
    _events, meta, bad, _rows = base
    _actions, report = skillcorner.convert_to_actions(bad.drop(columns=["frame_start"]), meta)
    assert report.period_repair_status == "not_evaluable"
    assert not [w for w in recwarn if issubclass(w.category, GiPeriodRepairWarning)]
    assert _convert(bad, meta)[1].period_repair_status == "repaired"  # the other side


def test_g1_a_labelled_period_without_a_possession_skips(base, monkeypatch):
    _events, meta, bad, _rows = base
    stray = bad[bad["event_type"] == "on_ball_engagement"].head(1).assign(period=3)  # period 3: no possession
    _assert_skipped(pd.concat([bad, stray], ignore_index=True), meta, "skipped_g1", monkeypatch)
    assert _convert(bad, meta)[1].period_repair_status == "repaired"


@pytest.mark.parametrize(("shift_ds", "status"), [(2, "skipped_g2"), (1, "repaired")])
def test_g2_the_clock_identity_tolerance_is_one_decisecond(base, monkeypatch, shift_ds, status):
    events, meta, bad, rows = base
    probe = bad.copy()
    s = implied_starts(events)
    off = {1: 0, 2: 27000}
    po = probe[(probe["event_type"] == "passing_option") & ~probe.index.isin(rows)]
    residual = _clock_ds(po["time_start"]) - (
        po["period"].map(off).to_numpy() + po["frame_start"].astype(int).to_numpy() - po["period"].map(s).to_numpy()
    )
    i = po.index[np.flatnonzero(residual == 0)[0]]  # an option exactly on the identity: the shift is the residual
    probe.loc[i, "time_start"] = _fmt_ds(int(_clock_ds(probe.loc[[i], "time_start"])[0]) + shift_ds)
    if status == "repaired":
        assert _convert(probe, meta)[1].period_repair_status == "repaired"  # within one decisecond: holds
    else:
        _assert_skipped(probe, meta, status, monkeypatch)


def _synthetic(starts: dict[int, int], plant_frame: int | None = None) -> pd.DataFrame:
    """Two periods of possessions on the GI clock identity (the first 50 frames after each start, then every 100);
    optionally one possession at ``plant_frame`` labelled 1 on the period-1 basis (a label-derived clock)."""
    rows = []
    off = {1: 0, 2: 27000}
    for p, s in starts.items():
        for k in range(6):
            f = s + 50 + 100 * k
            rows.append({"event_type": "player_possession", "event_id": f"{p}_{k}", "period": p, "frame_start": f})
            rows[-1]["time_start"] = _fmt_ds(off[p] + f - s)
    if plant_frame is not None:
        rows.append({"event_type": "player_possession", "event_id": "x", "period": 1, "frame_start": plant_frame})
        rows[-1]["time_start"] = _fmt_ds(plant_frame - starts[1])
    out = pd.DataFrame(rows)
    out["frame_end"] = out["frame_start"] + 5
    return out


def test_g3_non_increasing_period_starts_skip():
    bad = _synthetic({1: 5000, 2: 1000}, plant_frame=5010)
    with pytest.warns(GiPeriodRepairWarning, match="guard G3"):
        rep = repair_gi_periods(bad)
    assert rep.status == "skipped_g3" and (rep.period_source == "native").all()
    assert rep.period.equals(bad["period"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", GiPeriodRepairWarning)  # a boundary block: no warning
        good = repair_gi_periods(_synthetic({1: 1000, 2: 5000}, plant_frame=5010))  # the other side
    assert good.status == "repaired" and good.relabelled["player_possession"] == 1
    assert good.shape == "boundary_block" and good.start_frames == {1: 1000, 2: 5000}


def test_g4_a_linked_event_off_its_possession_period_skips(base_linked, monkeypatch):
    events, meta, bad, rows = base_linked
    _assert_skipped(_off_period_option(events, bad, rows), meta, "skipped_g4", monkeypatch)
    assert _convert(bad, meta)[1].period_repair_status == "repaired"


def test_g4_without_the_association_column_skips_only_when_linked_events_move(base_linked, monkeypatch):
    events, meta, bad, rows = base_linked
    no_assoc = bad.drop(columns=["associated_player_possession_event_id"])
    _assert_skipped(no_assoc, meta, "skipped_g4", monkeypatch)
    # the other side: a possession-only plant needs no association -> repaired
    pp_only = events.copy()
    pp_rows = [i for i in rows if bad.loc[i, "event_type"] == "player_possession"]
    pp_only.loc[pp_rows, ["period", "time_start"]] = bad.loc[pp_rows, ["period", "time_start"]].to_numpy()
    _a, report = _convert(pp_only.drop(columns=["associated_player_possession_event_id"]), meta)
    assert report.period_repair_status == "repaired" and report.period_relabelled["player_possession"] == 1


def _integer_ids(events: pd.DataFrame) -> pd.DataFrame:
    """The same GI table with INTEGER event ids: the association column becomes float64 (NaN on every non-linked
    row), the dtype a lakehouse / parquet read gives -- "8.0" next to "8" when joined through str()."""
    code = {e: i for i, e in enumerate(pd.unique(events["event_id"].astype(str)))}
    out = events.copy()
    out["event_id"] = events["event_id"].astype(str).map(code).astype("int64")
    assoc = events["associated_player_possession_event_id"]
    out["associated_player_possession_event_id"] = assoc.map(
        lambda v: np.nan if pd.isna(v) else float(code[str(v)])
    ).astype("float64")
    return out


def _off_period_option(events: pd.DataFrame, bad: pd.DataFrame, rows: pd.Index) -> pd.DataFrame:
    """``bad`` with one planted option moved onto a period-1 frame and clock: it is no conflict, so it keeps period 1
    while its possession is repaired to period 2 -- the G4 post-condition fails."""
    s = implied_starts(events)
    probe = bad.copy()
    i = next(i for i in rows if probe.loc[i, "event_type"] == "passing_option")
    probe.loc[i, "frame_start"] = s[1] + 100
    probe.loc[i, "time_start"] = _fmt_ds(100)
    return probe


def test_g4_joins_ids_through_id_compat_so_integer_ids_cannot_pass_vacuously(base_linked):
    # A C2-DELTA-06: integer event ids with a NaN-promoted association column ("8.0" vs "8") still meet in G4
    events, _meta, bad, rows = base_linked
    probe = _integer_ids(_off_period_option(events, bad, rows))
    assert probe["associated_player_possession_event_id"].dtype == "float64"  # the precondition: float-promoted
    with pytest.warns(GiPeriodRepairWarning, match="guard G4"):
        assert repair_gi_periods(probe).status == "skipped_g4"
    with warnings.catch_warnings():
        warnings.simplefilter("error", GiPeriodRepairWarning)
        assert repair_gi_periods(_integer_ids(bad)).status == "repaired"  # the other side: integer ids, consistent


def test_g4_without_event_ids_skips_when_linked_events_would_move(base_linked):
    # C C2-IMPL-35: the association column present but no event_id (only parse_passing_options can get this input)
    _events, _meta, bad, _rows = base_linked
    with pytest.warns(GiPeriodRepairWarning, match="cannot be read"):
        assert repair_gi_periods(bad.drop(columns=["event_id"])).status == "skipped_g4"


def test_g2_a_conflicting_frame_clock_event_without_a_clock_skips(base_linked):
    # A C2-DELTA-09: an event that would be repaired but has no readable clock cannot be re-based
    _events, _meta, bad, rows = base_linked
    probe = bad.copy()
    i = next(i for i in rows if probe.loc[i, "event_type"] == "passing_option")
    probe.loc[i, "time_start"] = np.nan
    with pytest.warns(GiPeriodRepairWarning, match="without a readable clock"):
        assert repair_gi_periods(probe).status == "skipped_g2"
    # the other side: a clockless option that does NOT conflict leaves the repair alone
    other = bad.copy()
    j = other.index[(other["event_type"] == "passing_option") & ~other.index.isin(rows)][0]
    other.loc[j, "time_start"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error", GiPeriodRepairWarning)
        assert repair_gi_periods(other).status == "repaired"


def test_status_vocabulary_is_closed(base):
    _events, meta, bad, _rows = base
    assert _convert(bad, meta)[1].period_repair_status in PERIOD_REPAIR_STATUS_VALUES


# ---------------------------------------------------------------------------
# (5) Shape: an interior repair warns, a boundary block does not
# ---------------------------------------------------------------------------


def test_an_interior_conflict_is_repaired_with_an_interior_shape_and_warns(base):
    events, meta, _bad, _rows = base
    clean, _ = _convert(events, meta)
    pp2 = events[(events["event_type"] == "player_possession") & (events["period"] == 2)]
    pp2 = pp2.sort_values(["frame_start", "frame_end"])
    eid = pp2["event_id"].iloc[len(pp2) // 2]  # deep inside period 2
    s = implied_starts(events)
    probe = events.copy()
    sel = (probe["event_id"] == eid) & (probe["event_type"] == "player_possession")
    assoc = probe["associated_player_possession_event_id"].astype(str) == str(eid)
    sel = sel | (assoc & probe["event_type"].isin(["passing_option", "off_ball_run"]))
    ds = _clock_ds(probe.loc[sel, "time_start"]) - 27000 + (s[2] - s[1])
    probe.loc[sel, "period"] = 1
    probe.loc[sel, "time_start"] = [_fmt_ds(v) for v in ds]
    with pytest.warns(GiPeriodRepairWarning, match="interior"):
        repaired, report = skillcorner.convert_to_actions(probe, meta)
    assert report.period_repair_status == "repaired" and report.period_repair_shape == "interior"
    assert _no_period_source(repaired) == _no_period_source(clean)


# ---------------------------------------------------------------------------
# (6) OBE: a conflicting OBE is withheld from tackle enrichment (never repaired), both shapes
# ---------------------------------------------------------------------------


_INTERCEPTION = spadlconfig.actiontype_id["interception"]
_TACKLE = spadlconfig.actiontype_id["tackle"]


def _defensive_type(actions: pd.DataFrame, eid: object, period: int) -> set[int]:
    """The type ids of the derived defensive row(s) inferred from possession ``eid`` in ``period``."""
    rows = actions[
        (actions["action_provenance"] == "derived")
        & (actions["original_event_id"].astype(str) == str(eid))
        & (actions["period_id"] == period)
        & actions["type_id"].isin([_INTERCEPTION, _TACKLE])
    ]
    return {int(t) for t in rows["type_id"]}


def _plant_obe(events: pd.DataFrame, pp_i, *, period: int, time_start: str, frame: int) -> pd.DataFrame:
    """``events`` + one ``direct_regain`` OBE by the possession ``pp_i``'s player and team, labelled ``period`` at the
    clock ``time_start`` and the frame ``frame``: a tackle-upgrade candidate for that possession's derived
    interception whenever the OBE is not withheld (same team, same LABELLED period, within the 2 s window)."""
    pp = events.loc[pp_i]
    obe = events[events["event_type"] == "on_ball_engagement"].iloc[[0]].copy()
    obe["event_id"] = "planted_obe"
    for col in ("team_id", "player_id", "x_start", "y_start"):
        obe[col] = pp[col]
    obe["period"] = period
    obe["time_start"] = time_start
    obe["frame_start"] = frame
    obe["frame_end"] = frame + 1
    obe["end_type"] = "direct_regain"
    if "associated_player_possession_event_id" in obe.columns:
        obe["associated_player_possession_event_id"] = np.nan
    return pd.concat([events, obe], ignore_index=True)


def _obe_case(fits_label: bool):
    """``(events, meta, pp row, label period, the OBE clock, the conflicting frame, the non-conflicting frame)``.

    * fits neither (the 2052202 shape): a derived interception of the base match; the OBE carries that possession's
      clock and label, but its frame lies in the other period, so its clock fits neither.
    * fits its label (the 2048846 shape): the first period-2 possession, retyped to ``recovery`` so it infers a derived
      interception at the kick-off; the OBE's frame lies 5 frames before period 2's start and its clock is computed
      from that frame on the period-2 basis (44:59.5), inside the 2 s window."""
    if not fits_label:
        events, meta = load_opendata(_BASE)
        actions, _ = skillcorner.convert_to_actions(events, meta)
        inter = actions[(actions["type_id"] == _INTERCEPTION) & (actions["action_provenance"] == "derived")]
        pp = events[events["event_type"] == "player_possession"]
        pp_i = next(
            i for eid in inter["original_event_id"].astype(str) for i in pp.index[pp["event_id"].astype(str) == eid][:1]
        )
        s = implied_starts(events)
        q = _as_int(events.loc[pp_i, "period"])
        other = 2 if q == 1 else 1
        return (
            events,
            meta,
            pp_i,
            q,
            str(events.loc[pp_i, "time_start"]),
            s[other] + 50,
            _as_int(events.loc[pp_i, "frame_start"]),
        )
    for mid in _IDS:
        events, meta = load_opendata(mid)
        s = implied_starts(events)
        pp2 = events[(events["event_type"] == "player_possession") & (events["period"] == 2)]
        pp2 = pp2.sort_values(["frame_start", "frame_end"])
        pp_i = pp2.index[0]
        if (_as_int(events.loc[pp_i, "frame_start"]) - s[2]) / 10.0 < 1.4:  # the kick-off is inside the window
            events = events.copy()
            events.loc[pp_i, "start_type"] = "recovery"
            return events, meta, pp_i, 2, _fmt_ds(27000 - 5), s[2] - 5, _as_int(events.loc[pp_i, "frame_start"])
    pytest.fail("no opendata kick-off possession inside the 2 s OBE window")


@pytest.mark.parametrize("fits_label", [True, False], ids=["fits_its_label", "fits_neither"])
def test_a_conflicting_obe_is_withheld_from_tackle_enrichment_and_counted(fits_label):
    events, meta, pp_i, q, clock, conflicting, aligned = _obe_case(fits_label)
    eid = events.loc[pp_i, "event_id"]
    bad = _plant_obe(events, pp_i, period=q, time_start=clock, frame=conflicting)
    twin = _plant_obe(events, pp_i, period=q, time_start=clock, frame=aligned)
    bad_actions, bad_report = _convert(bad, meta)
    twin_actions, twin_report = _convert(twin, meta)
    # the twin (frame in the label's period): the OBE is a candidate, so the interception upgrades to a tackle
    assert twin_report.obe_period_conflict == 0 and _defensive_type(twin_actions, eid, q) == {_TACKLE}
    # the conflicting OBE is withheld: the derived row stays an interception, and the conflict is counted
    assert bad_report.obe_period_conflict == 1 and bad_report.period_repair_status == "clean"
    assert _defensive_type(bad_actions, eid, q) == {_INTERCEPTION}
    assert _as_int(bad.loc[bad["event_id"] == "planted_obe", "period"].iloc[0]) == q  # never repaired


# ---------------------------------------------------------------------------
# (7) The match_periods diagnostic (a warning, never a gate)
# ---------------------------------------------------------------------------


def test_a_disagreeing_match_periods_start_warns_and_the_repair_still_runs(base):
    _events, meta, bad, _rows = base
    off = json.loads(json.dumps(meta))
    off["match_periods"][1]["start_frame"] = int(off["match_periods"][1]["start_frame"]) + 5
    with pytest.warns(GiPeriodRepairWarning, match="match_periods"):
        _a, report = skillcorner.convert_to_actions(bad, off)
    assert report.period_start_mismatch is True and report.period_repair_status == "repaired"
    _a, agree = _convert(bad, meta)  # the agreeing fixture: no warning (it would raise), no flag
    assert agree.period_start_mismatch is False


# ---------------------------------------------------------------------------
# (9) Provenance registries
# ---------------------------------------------------------------------------


def test_period_source_provenance_contract(base):
    from silly_kicks.reflection import SPADL_REFLECTION_KINDS
    from silly_kicks.spadl._training_exclusion import provenance_masks, training_exclusion_mask
    from silly_kicks.spadl.schema import SKILLCORNER_SPADL_COLUMNS, SkillCornerConversionReport

    events, meta, bad, _rows = base
    actions, report = _convert(bad, meta)
    clean, _ = _convert(events, meta)  # the same rows, every one "native"
    frame = actions["period_source"] == "frame"
    assert frame.any() and actions["period_source"].notna().all()
    assert set(actions["period_source"]) <= {"native", "frame"}
    # period_source marks nothing unvalidated: the masks equal the clean conversion's row for row (a frame row is
    # excluded / unvalidated exactly when its clean twin is -- for its result or time tier, never for its period)
    assert training_exclusion_mask(actions).tolist() == training_exclusion_mask(clean).tolist()
    clean_masks = provenance_masks(clean)
    for name, mask in provenance_masks(actions).items():
        assert mask.tolist() == clean_masks[name].tolist(), name
    assert "period_source" not in Path(training_exclusion_mask.__code__.co_filename).read_text(encoding="utf-8")
    assert SKILLCORNER_SPADL_COLUMNS["period_source"] == "object"
    assert SPADL_REFLECTION_KINDS["period_source"] == "invariant"
    rt = SkillCornerConversionReport(**dataclasses.asdict(report))
    assert dataclasses.asdict(rt) == dataclasses.asdict(report)


# ---------------------------------------------------------------------------
# (13) The helper's import allowlist (numpy + pandas only; never silly_kicks.tracking)
# ---------------------------------------------------------------------------


def test_period_repair_helper_imports_numpy_pandas_and_id_compat_only():
    src = Path(__file__).parents[2] / "silly_kicks" / "spadl" / "_skillcorner_periods.py"
    found: set[str] = set()
    for node in ast.walk(ast.parse(src.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            found.add("." * node.level + (node.module or ""))
    # id_compat (itself numpy + pandas) carries the AGENTS.md id rule for the G4 join; never silly_kicks.tracking
    assert found <= {"__future__", "dataclasses", "warnings", "collections.abc", "numpy", "pandas", "..id_compat"}, (
        found
    )
    id_compat = Path(__file__).parents[2] / "silly_kicks" / "id_compat.py"
    deps: set[str] = set()
    for node in ast.walk(ast.parse(id_compat.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            deps |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            deps.add("." * node.level + (node.module or ""))
    assert not any(d.startswith("silly_kicks") or d.startswith(".") for d in deps), deps


def test_the_moved_time_helpers_are_the_same_objects():
    from silly_kicks.spadl import _skillcorner_periods as periods

    assert skillcorner._PERIOD_START_SECONDS is periods._PERIOD_START_SECONDS
    assert skillcorner._parse_time_start is periods._parse_time_start
    assert skillcorner.GiPeriodRepairWarning is GiPeriodRepairWarning


def test_the_block_builder_plants_the_possession_and_its_linked_events():
    events, _meta = load_opendata(_BASE)
    rows, eid = _block(events, 2, first=True)
    kinds = set(events.loc[rows, "event_type"])
    assert "player_possession" in kinds and kinds <= {"player_possession", "passing_option", "off_ball_run"}
    assert (events.loc[rows, "period"] == 2).all() and str(eid) in set(events.loc[rows, "event_id"].astype(str))
