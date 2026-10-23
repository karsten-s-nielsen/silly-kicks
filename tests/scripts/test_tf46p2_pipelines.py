"""The TF-46 Arm 1 Part 2 pre-flight pipelines (owner ruling 2026-10-22: born-ready ADR-128 walkers) -- the
scoring core (``scripts._tf46p2_score``), the stage plumbing (``scripts._tf46p2_pipeline``) and the assembly
(``scripts.tf46p2_gate_table``). Offline, synthetic; the end-to-end stage runs are the slow parity fixture
(``tests/scripts/test_parity_tf46p2.py``)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from scripts import _tf46p2_pipeline as P
from scripts import _tf46p2_score as S
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._restart_frame import GiPairing
from silly_kicks.tracking._restart_validation import AGGREGATE_COLUMNS, RATIFIED, DataProfile, derive_gate_table

_SC = DataProfile("skillcorner", "r1", "g1", "cs1")
_IDSSE = DataProfile("sportec", "r1", None, "cs1")
_GS = DataProfile("gradientsports", "r1", None, "wc22")


def _pairing(gid, pairing="paired"):
    return GiPairing(
        gid,
        pairing,
        None if pairing == "paired" else "proximity",
        0.95,
        0,
        0.9,
        1,
        "sc_possession",
        5,
        None,
        None,
        False,
    )


# ---------------------------------------------------------------------------
# GI pairing report (C-11, run 2): the `.pairing` field, is_opendata, and the sensitivity verdicts
# ---------------------------------------------------------------------------


def test_gi_pairing_report_reads_the_pairing_field_and_carries_sensitivity():
    from scripts._tf46p2_common import gi_pairing_report

    rep = gi_pairing_report(
        {"m1": _pairing("m1"), "m2": _pairing("m2", "unpaired")},
        _SC,
        is_opendata=True,
        sensitivity={"pair_prox_m=1.0": {"m1": _pairing("m1", "unpaired"), "m2": _pairing("m2", "unpaired")}},
    )
    assert rep.set_index("match_id")["paired"].to_dict() == {"m1": True, "m2": False}
    assert rep["is_opendata"].all()
    assert rep.set_index("match_id")["paired@pair_prox_m=1.0"].to_dict() == {"m1": False, "m2": False}


def test_assert_known_paired_fires_on_a_real_pairing_report():
    # The end-to-end bug: the report must carry is_opendata + the real paired verdict, so the C-11 calibration
    # assert can actually fire on an unpaired opendata match (it could not before).
    from scripts._tf46p2_common import GiPairingCalibrationError, assert_known_paired, gi_pairing_report

    rep = gi_pairing_report({"m2": _pairing("m2", "unpaired")}, _SC, is_opendata=True)
    with pytest.raises(GiPairingCalibrationError, match="m2"):
        assert_known_paired(rep)
    assert_known_paired(gi_pairing_report({"m2": _pairing("m2", "unpaired")}, _SC, is_opendata=False))


# ---------------------------------------------------------------------------
# OD-6 amendment 2026-10-23: the derived proximity bar, its variants and the per-profile guard
# ---------------------------------------------------------------------------


def test_gi_pairing_report_carries_every_verdict_and_the_period_counts():
    import dataclasses

    from scripts._tf46p2_common import PAIRING_VERDICT_COLUMNS, gi_pairing_report

    gp = dataclasses.replace(
        _pairing("m1", "unpaired"),
        pair_range_ok=True,
        pair_proximity_ok=None,
        pair_lag_ok=True,
        pair_direction_ok=False,
        n_unresolved_periods=1,
        n_insufficient_team_periods=2,
    )
    row = gi_pairing_report({"m1": gp}, _SC).iloc[0]
    assert [row[c] if row[c] is not pd.NA else None for c in PAIRING_VERDICT_COLUMNS] == [True, None, True, False]
    assert row["n_unresolved_periods"] == 1 and row["n_insufficient_team_periods"] == 2


def test_gi_pairing_report_carries_the_range_basis_and_the_violation_classes():
    # spec section 6.1, amended 2026-10-25: the classified Range violations replace the retired n_edge_clipped
    import dataclasses

    from scripts._tf46p2_common import gi_pairing_report

    gp = dataclasses.replace(
        _pairing("m1", "unpaired"),
        range_basis="provider",
        range_violations=((1, 5, 9, "partial_edge"), (2, 1, 2, "outside"), (2, 3, 4, "outside")),
    )
    row = gi_pairing_report({"m1": gp}, _SC).iloc[0]
    assert row["range_basis"] == "provider" and row["n_range_violations"] == 3
    assert row["range_violation_classes"] == "outside,partial_edge"
    clean = gi_pairing_report({"m1": _pairing("m1")}, _SC)
    assert "n_edge_clipped" not in clean.columns and clean.iloc[0]["range_violation_classes"] == ""


def _r2_census() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A period-repair census + pairing report that meet the spec section 26.8 pre-registration exactly."""
    from scripts._tf46p2_common import R2_PERIOD_REPAIRED_MATCHES, R2_SKILLCORNER_MATCHES

    repaired = sorted(R2_PERIOD_REPAIRED_MATCHES)
    others = ["2048846", "2052202"] + [f"c{i}" for i in range(R2_SKILLCORNER_MATCHES - len(repaired) - 2)]
    po = dict(zip(repaired, [6, 6, 6, 6, 6, 6, 12], strict=True))  # 48 options over the 7 matches
    rows = [
        {
            "match_id": m,
            "status": "repaired" if m in po else "clean",
            "relabelled_player_possession": int(m in po),
            "relabelled_passing_option": po.get(m, 0),
            "relabelled_off_ball_run": 0,
            "shape": "boundary_block" if m in po else None,
            "obe_period_conflict": int(m in ("2048846", "2052202")),
            "period_start_mismatch": False,
        }
        for m in repaired + others
    ]
    pairing = pd.DataFrame(
        {"match_id": [r["match_id"] for r in rows], "paired": True, "range_basis": "provider", "n_range_violations": 0}
    )
    return pd.DataFrame(rows), pairing


def test_r2_period_census_holds_on_the_pre_registration_and_names_every_deviation():
    from scripts._tf46p2_common import R2CensusDeviationError, assert_r2_period_census

    pr, pa = _r2_census()
    assert_r2_period_census(pr, pa)  # the exact pre-registration holds
    deviations = [
        (lambda p, q: (p.assign(status=p["status"].where(p["match_id"] != "c0", "skipped_g2")), q), "non-clean"),
        (lambda p, q: (p.assign(relabelled_off_ball_run=p["match_id"].eq("1766992").astype(int)), q), "relabelled"),
        (lambda p, q: (p.assign(shape=p["shape"].where(p["match_id"] != "1766992", "interior")), q), "shapes"),
        (lambda p, q: (p.assign(obe_period_conflict=0), q), "OBE"),
        (lambda p, q: (p.assign(period_start_mismatch=p["match_id"].eq("c3")), q), "match_periods"),
        (lambda p, q: (p.iloc[:-1], q), "census records"),
        (lambda p, q: (p, q.assign(range_basis=q["range_basis"].where(q["match_id"] != "c1", "retained"))), "basis"),
        (lambda p, q: (p, q.assign(n_range_violations=q["match_id"].eq("c2").astype(int))), "Range violations"),
        (lambda p, q: (p, q.assign(paired=q["match_id"].ne("2007448") & q["match_id"].ne("c4"))), "unpaired"),
    ]
    for mutate, needle in deviations:
        bad_pr, bad_pa = mutate(pr.copy(), pa.copy())
        with pytest.raises(R2CensusDeviationError, match=needle):
            assert_r2_period_census(bad_pr, bad_pa)


def test_kick_cells_reports_the_repair_census_and_range_classes_per_data_profile():
    # spec section 26.7 (14): aggregates per data profile (counts and match ids only)
    from scripts.tf46p2_kick_cells import _period_and_range_tables

    pr, _pa = _r2_census()
    pr = pr.assign(data_profile=["a"] * 10 + ["b"] * (len(pr) - 10))
    pairing = pd.DataFrame(
        {
            "data_profile": ["a", "a", "b"],
            "match_id": ["1", "2", "3"],
            "range_basis": ["provider", "retained", "provider"],
            "range_violation_classes": ["", "outside,partial_edge", ""],
        }
    )
    repair, by_profile, ranges = _period_and_range_tables(pr, pairing)
    assert len(repair) == len(pr)
    tot = by_profile.set_index("data_profile")
    assert int(tot["n_repaired"].sum()) == 7 and int(tot["relabelled_passing_option"].sum()) == 48
    assert int(tot["obe_period_conflict"].sum()) == 2 and int(tot["n_matches"].sum()) == len(pr)
    r = ranges.set_index("data_profile")
    assert r.loc["a", "n_retained_basis"] == 1 and r.loc["a", "n_outside"] == 1 and r.loc["a", "n_partial_edge"] == 1
    assert r.loc["b", "n_provider_basis"] == 1 and r.loc["b", "n_other_period"] == 0


def test_r2_census_applies_on_the_full_r2_dataset_only():
    from scripts._tf46p2_common import R2_CENSUS_DATASET, r2_census_applies

    assert r2_census_applies(R2_CENSUS_DATASET, None) is True
    assert r2_census_applies(R2_CENSUS_DATASET, 5) is False  # a capped smoke run
    assert r2_census_applies("toytf46", None) is False  # the toy corpus


def test_a_stale_skillcorner_staging_is_rejected_mechanically():
    # spec section 26.8 (DL ruling Q6): the 6d998a5 staging has no provider range / repair census / period_source
    empty = pd.DataFrame()
    frames = pd.DataFrame({"period_id": [1], "frame_id": [10]})  # a unit WITH tracking frames
    census = {"gi_period_repair": {"status": "clean"}}
    fresh = {"provider_period_frame_ranges": {"1": [10, 20]}, **census}
    acts = pd.DataFrame({"period_source": ["native"]})
    S.assert_staging_current(S.part2_match_from("skillcorner", "7", acts, frames, empty, empty, fresh))
    with pytest.raises(S.StaleStagingError, match="provider_period_frame_ranges"):
        S.assert_staging_current(S.part2_match_from("skillcorner", "7", acts, frames, empty, empty, census))
    with pytest.raises(S.StaleStagingError, match="gi_period_repair"):
        S.assert_staging_current(S.part2_match_from("skillcorner", "7", acts, frames, empty, empty, {}))
    with pytest.raises(S.StaleStagingError, match="period_source"):
        S.assert_staging_current(S.part2_match_from("skillcorner", "7", empty, frames, empty, empty, fresh))
    # a unit staged WITHOUT tracking has no adapter report, so no range: it is not called stale (C2-IMPL-36)
    S.assert_staging_current(S.part2_match_from("skillcorner", "7", acts, empty, empty, empty, census))
    S.assert_staging_current(S.part2_match_from("idsse", "8", empty, empty, empty, empty, {}))  # no GI: passes
    m = S.part2_match_from("skillcorner", "7", acts, frames, empty, empty, fresh)
    assert m.period_ranges == {("7", 1): (10, 20)} and m.period_repair == {"status": "clean"}


def test_load_part2_match_rejects_a_stale_staged_unit(tmp_path):
    # the wiring (C2-IMPL-36): the staged loader itself runs the staging check, before any stage sees the unit
    import json as _json

    from scripts._driver_sources import staged_part2_refs
    from scripts._toy_corpora import build_part2_corpus

    build_part2_corpus(tmp_path)
    refs = {r.stem: r for r in staged_part2_refs(tmp_path, "toytf46")}
    S.load_part2_match(refs["skillcorner__501"], tmp_path / "toytf46")  # the current toy staging loads
    prof = tmp_path / "toytf46" / "_profile" / "skillcorner__501.json"
    meta = _json.loads(prof.read_text(encoding="utf-8"))
    meta.pop("provider_period_frame_ranges")  # the 6d998a5 staging had no provider range
    prof.write_bytes(_json.dumps(meta, sort_keys=True).encode("utf-8"))
    with pytest.raises(S.StaleStagingError, match="skillcorner match 501"):
        S.load_part2_match(refs["skillcorner__501"], tmp_path / "toytf46")


def test_kick_cells_calibrate_runs_the_r2_census_stop_on_the_r2_dataset(monkeypatch):
    # the census call site (B C2-DELTA-07): on the full R.2 dataset a deviating census stops calibrate; on another
    # dataset the census is not asserted (the run goes on to the next guard)
    import scripts.tf46p2_kick_cells as kc
    from scripts._tf46p2_common import R2_CENSUS_DATASET, R2CensusDeviationError

    pr, pa = _r2_census()
    pr = pr.assign(data_profile="p", status=pr["status"].where(pr["match_id"] != "c0", "skipped_g2"))
    pa = pa.assign(data_profile="p", is_opendata=False)
    direction = [{"data_profile": "p", "match_id": "c0", "period_id": p, "agrees": False} for p in (1, 2)]
    direction += [{"data_profile": "p", "match_id": f"m{i}", "period_id": 1, "agrees": True} for i in range(1, 400)]

    class _Ctx:
        def __init__(self, dataset):
            self.params = {"dataset": dataset, "max_matches": None}

    census = P.records_frame({"period_repair": pr.to_dict("records"), "pairing": pa.to_dict("records")})
    monkeypatch.setattr(kc, "done_results", lambda ctx, stage: [census])
    with pytest.raises(R2CensusDeviationError, match="non-clean"):
        kc._calibrate_work("only", ctx=_Ctx(R2_CENSUS_DATASET))
    # another dataset: the census is not asserted, so the run reaches the next guard (the GI direction corpus guard)
    other = P.records_frame({"period_repair": pr.to_dict("records"), "gi_direction": direction})
    monkeypatch.setattr(kc, "done_results", lambda ctx, stage: [other])
    with pytest.raises(AssertionError, match="BOTH periods"):
        kc._calibrate_work("only", ctx=_Ctx("toytf46"))


def test_paired_by_construction_carries_four_true_verdicts():
    # C2 delta C-29: "paired only when all four verdicts are True" holds on transfer-run records
    gp = S.paired_by_construction(pd.DataFrame({"game_id": ["g1"]}))["g1"]
    assert gp.pairing == "paired"
    assert (gp.pair_range_ok, gp.pair_proximity_ok, gp.pair_lag_ok, gp.pair_direction_ok) == (True, True, True, True)


def test_pairing_lag_table_and_guard_both_sides():
    # second R.2 stop ruling 2026-10-23: the offset distribution per profile; a profile median outside +/-2 raises
    from scripts._tf46p2_common import GiPairingLagError, assert_pairing_lag_holds, pairing_lag_table

    ok = pd.DataFrame({"data_profile": "p", "alignment_offset": [-1, -1, -2, 0, None]})
    tab = pairing_lag_table(ok)
    assert tab.set_index("alignment_offset")["n"].to_dict() == {-2: 1, -1: 2, 0: 1}  # the unevaluable row left out
    assert_pairing_lag_holds(ok)
    desync = pd.DataFrame({"data_profile": ["p", "q", "q", "q"], "alignment_offset": [0, -3, -3, 1]})
    with pytest.raises(GiPairingLagError, match="q"):
        assert_pairing_lag_holds(desync)


def test_staging_generation_is_recorded_from_the_unit_refs(tmp_path):
    from scripts._toy_corpora import TOY_GENERATION, build_part2_corpus

    build_part2_corpus(tmp_path)
    ctx = type("Ctx", (), {"data_root": tmp_path, "params": {"dataset": "toytf46", "max_matches": None}})()
    assert P.staging_generation(ctx) == [TOY_GENERATION]


def test_derive_pairing_bar_is_the_quantile_plus_margin_of_finite_negatives():
    from scripts._tf46p2_common import derive_pairing_bar

    neg = [0.05, 0.10, float("nan"), 0.15, None]
    assert derive_pairing_bar(neg, quantile=0.5, margin=0.10) == pytest.approx(0.20)  # median 0.10 + 0.10
    assert np.isnan(derive_pairing_bar([float("nan")], quantile=0.99, margin=0.10))  # no negative -> NaN


def _prox_rows(profile: str, shares: dict[str, float], neg: float, *, opendata: bool) -> list[dict]:
    rows = []
    for mid, s in shares.items():
        for m in (1.0, 2.0, 3.0):
            rows.append(
                {
                    "data_profile": profile,
                    "match_id": mid,
                    "is_opendata": opendata,
                    "prox_m": m,
                    "shift_s": 0.0,
                    "prox_share": s,
                }
            )
            for sh in (-30.0, -10.0, -5.0, 5.0, 10.0, 30.0):
                rows.append(
                    {
                        "data_profile": profile,
                        "match_id": mid,
                        "is_opendata": opendata,
                        "prox_m": m,
                        "shift_s": sh,
                        "prox_share": neg,
                    }
                )
    return rows


def _verdict_report(mids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"match_id": mids, "pair_range_ok": True, "pair_lag_ok": True, "pair_direction_ok": True})


def test_pairing_bar_table_rederives_per_variant_and_counts_the_known_paired():
    from scripts._tf46p2_common import pairing_bar_table

    prox = pd.DataFrame(_prox_rows("od", {"a": 0.85, "b": 0.20}, 0.10, opendata=True))
    tab = pairing_bar_table(prox, _verdict_report(["a", "b"])).set_index("variant")
    assert set(tab.index) == {
        "base",
        "pair_prox_margin=0.05",
        "pair_prox_margin=0.15",
        "pair_prox_m=1.0",
        "pair_prox_m=3.0",
    }
    base = tab.loc["base"].to_dict()
    assert base["bar"] == pytest.approx(0.20) and base["n_negatives"] == 12  # 0.10 + 0.10; 2 matches x 6 shifts
    assert base["n_proximity_pass"] == 2 and base["n_known_paired_pass"] == 2  # 0.20 >= 0.20 passes (closed)
    assert base["min_known_paired_margin"] == pytest.approx(0.0)
    assert tab.loc["pair_prox_margin=0.15"].to_dict()["n_known_paired_pass"] == 1  # 0.20 < 0.25: the margin moved


def test_pairing_bar_guard_both_sides():
    from scripts._tf46p2_common import GiPairingBarError, assert_pairing_bar_holds, pairing_bar_table

    low = pd.DataFrame(_prox_rows("p_low", {"a": 0.85}, 0.10, opendata=False))  # bar 0.20 <= 0.245
    assert_pairing_bar_holds(pairing_bar_table(low, _verdict_report(["a"])))
    high = pd.DataFrame(_prox_rows("p_high", {"a": 0.85}, 0.20, opendata=False))  # bar 0.30 > 0.245
    with pytest.raises(GiPairingBarError, match="p_high"):
        assert_pairing_bar_holds(pairing_bar_table(high, _verdict_report(["a"])))
    none = pd.DataFrame(_prox_rows("p_none", {"a": 0.85}, float("nan"), opendata=False))  # no negative at all
    with pytest.raises(GiPairingBarError, match="p_none"):
        assert_pairing_bar_holds(pairing_bar_table(none, _verdict_report(["a"])))
    # no evaluable GI start at all (no match share, no negative): nothing pairs, nothing to guard
    empty = pd.DataFrame(_prox_rows("p_empty", {"a": float("nan")}, float("nan"), opendata=False))
    assert_pairing_bar_holds(pairing_bar_table(empty, _verdict_report(["a"])))


# ---------------------------------------------------------------------------
# The reduce: calibration, LOMO, the fixed origin bar, GS admission from FINAL verdicts, the subtype baseline
# ---------------------------------------------------------------------------


def _kick_rows(profile, base, errs_by_match, tier="transfer", cell="both_pos"):
    return [
        S.score_row(profile, "corner", "kick_time", cell, base=base, tier=tier, match_id=mid, err=e)
        for mid, errs in errs_by_match.items()
        for e in errs
    ]


def test_calibrated_cell_is_scored_lomo_with_the_tolerance_from_the_other_matches():
    # m0..m4 errors 0.05 s, m5 errors 0.5 s: m5's held-out tolerance (q0.90 of the others) is 0.05 -> it fails.
    errs = {f"m{i}": [0.05] * 8 for i in range(5)} | {"m5": [0.5] * 8}
    agg = S.aggregate_scores(S.scores_frame(_kick_rows(_SC, "idsse", errs)))
    per = agg.set_index("match_id")
    assert per.loc["m5", "k"] == 0 and per.loc["m0", "k"] == 8
    assert per["calibration_ceiling"].iloc[0] == RATIFIED["tau_detector_ceiling_s"]
    assert per["calibration_metric"].iloc[0] >= 0.05  # all-match q0.90 includes the bad match


def test_k_gate_is_centred_on_the_median_offset():
    # A constant decision offset of +0.6 s is not dispersion: the K gate centres on the median (spec 5.2).
    rows = [
        S.score_row(
            _IDSSE,
            "corner",
            "kick_time",
            "k_gate",
            base="idsse",
            tier="reference",
            match_id=f"m{i}",
            err=0.6 + 0.01 * j,
        )
        for i in range(5)
        for j in range(8)
    ]
    agg = S.aggregate_scores(S.scores_frame(rows))
    # Centred, the calibrated tolerance is the DISPERSION (~0.06 s), within the 0.40 s ceiling -> the K gate can
    # ship; uncentred it would be the offset itself (~0.66 s > ceiling -> withheld). The offset is reported.
    assert float(agg["calibration_metric"].iloc[0]) < 0.1 < float(agg["calibration_ceiling"].iloc[0])
    uncentred = S._ok_calibrated(S.scores_frame(rows), 0.90, False)[2]
    assert uncentred > RATIFIED["tau_event_ceiling_s"]


def test_origin_bar_is_median_and_share_within_five_metres():
    rows = [
        S.score_row(
            _IDSSE, "free_kick", "origin", "stationary_window", base="idsse", tier="reference", match_id=f"m{i}", err=e
        )
        for i in range(5)
        for e in (0.5, 1.0, 1.5, 6.0, float("inf"))
    ]
    agg = S.aggregate_scores(S.scores_frame(rows))
    assert int(agg["k"].sum()) == 15 and float(agg["calibration_metric"].iloc[0]) == pytest.approx(1.5)
    assert float(agg["calibration_ceiling"].iloc[0]) == RATIFIED["origin_median_m"]


def test_gs_admission_uses_the_final_calibrated_verdicts():
    # GS reference kick_time rows are calibrated (err, not ok): admission must still be computed from them.
    good = {f"g{i}": [0.05] * 8 for i in range(5)}
    rows = _kick_rows(_GS, "gradientsports", good, tier="reference", cell="tf79_contact")
    rows += _kick_rows(_SC, "gradientsports", good, tier="transfer")
    agg = S.aggregate_scores(S.scores_frame(rows))
    transfer = agg[agg["tier"] == "transfer"]
    assert transfer["admitted"].all()
    table = S.gs_admission_table(S.scores_frame(rows))
    assert table.set_index("signal").loc["kick_time", "gs_admitted"]


def test_gs_never_admitted_for_first_contact():
    rows = [
        S.score_row(
            _GS,
            "corner_crossed",
            "first_contact",
            "uncontested_taker_known",
            base="gradientsports",
            tier="reference",
            match_id=f"g{i}",
            err=0.01,
        )
        for i in range(6)
        for _ in range(8)
    ]
    assert not S.gs_admission_table(S.scores_frame(rows)).set_index("signal").loc["first_contact", "gs_admitted"]


def test_subtype_baseline_margin_both_sides():
    def block(pred_ok, truth):
        return pd.DataFrame(
            {
                "match_id": [f"m{i % 5}" for i in range(len(truth))],
                "cell_state": "crossed",
                "ok": pred_ok,
                "truth": truth,
            }
        )

    # all predictions crossed and correct; the truth is 50/50 -> the rule beats the majority baseline
    beats = block([True] * 20 + [False] * 0, ["crossed"] * 20)
    mixed_truth = block([True] * 10 + [False] * 10, ["crossed"] * 10 + ["short"] * 10)
    assert S._baseline_margin_lb(mixed_truth) < 0.05  # precision 0.5 == baseline 0.5 -> not > 0 with margin
    assert math.isfinite(S._baseline_margin_lb(beats))


# ---------------------------------------------------------------------------
# Plumbing: records round trip, degradation params, the copy exclusion, the cell states, the simulations
# ---------------------------------------------------------------------------


def test_records_frame_round_trip_keeps_nan_inf_and_none():
    tables = {"a": [{"x": 1.0, "y": None}, {"x": float("inf"), "y": "s"}], "b": pd.DataFrame({"z": [np.nan]})}
    out = P.split_records([P.records_frame(tables)])
    assert out["a"]["x"].tolist()[1] == float("inf") and out["a"]["y"].tolist() == [None, "s"]
    assert math.isnan(out["b"]["z"].iloc[0])
    assert P.split_records([P.records_frame({})]) == {}


def test_degradation_params_pool_the_profile_statistics():
    stats = pd.DataFrame(
        [
            {"region": "rest", "statistic": "frame_rate", "value": 10.0, "n": 1, "samples": []},
            {"region": "rest", "statistic": "frame_rate", "value": 10.0, "n": 1, "samples": []},
            {"region": "corner_flag", "statistic": "gap_distribution", "value": 1.0, "n": 2, "samples": [0.4, 1.2]},
            {"region": "corner_flag", "statistic": "gap_distribution", "value": 1.0, "n": 1, "samples": [0.8]},
            {"region": "rest", "statistic": "noise_sd", "value": 0.2, "n": 3, "samples": []},
        ]
    )
    p = P.degradation_params(stats)
    assert p["frame_rate"] == 10.0 and p["gap_lengths"]["corner_flag"] == [0.4, 0.8, 1.2]
    assert p["noise_sd"]["rest"] == pytest.approx(0.2)


def test_copy_excluded_bases_from_the_run_one_location_table():
    from scripts.tf46p2_subtype_merge import copy_excluded_bases

    loc = pd.DataFrame(
        {
            "event_type": ["CornerKick", "FreeKick", "Play"],
            "field": ["position", "source_position", "position"],
            "is_copy": [True, True, True],
        }
    )
    assert copy_excluded_bases(loc) == frozenset({"corner"})  # only the POSITION field decides; Play is no restart
    assert copy_excluded_bases(None) == frozenset()


def test_kick_time_cell_state_is_state_by_sign():
    assert S.kick_time_cell_state("tf79_contact", 0.1) == "tf79_contact_pos"
    assert S.kick_time_cell_state("both", -0.05) == "both_neg"
    assert S.kick_time_cell_state(pd.NA, float("nan")) == "none_zero"


def _actions(rows):
    base = {
        "game_id": "g",
        "period_id": 1,
        "team_id": "t1",
        "player_id": "p1",
        "start_x": 50.0,
        "start_y": 30.0,
        "end_x": 60.0,
        "end_y": 30.0,
        "result_id": 1,
        "bodypart_id": 0,
        "original_event_id": "e",
    }
    return pd.DataFrame([{**base, "action_id": i, **r} for i, r in enumerate(rows)])


def test_merge_simulation_builds_positive_and_negative_pairs():
    a = _actions(
        [
            {"time_seconds": 10.0, "type_id": spadlconfig.actiontype_id["shot_freekick"]},
            {"time_seconds": 50.0, "type_id": spadlconfig.actiontype_id["freekick_short"]},
            {"time_seconds": 51.0, "type_id": spadlconfig.actiontype_id["shot"]},
        ]
    )
    sim, truth = S.merge_simulation(a)
    assert sorted(truth.values()) == [False, True]
    pos = next(k for k, v in truth.items() if v)
    row = sim[sim["action_id"] == pos].iloc[0]
    assert row["action_provenance"] == "derived" and math.isnan(row["start_x"])
    assert (sim["type_id"] == spadlconfig.actiontype_id["shot"]).sum() == 2


def test_simulated_skillcorner_actions_recast_restarts_and_bracket_them():
    a = _actions(
        [
            {"time_seconds": 9.0, "type_id": spadlconfig.actiontype_id["pass"]},
            {"time_seconds": 10.0, "type_id": spadlconfig.actiontype_id["corner_crossed"]},
            {"time_seconds": 11.0, "type_id": spadlconfig.actiontype_id["pass"]},
            {"time_seconds": 30.0, "type_id": spadlconfig.actiontype_id["shot_penalty"]},
        ]
    )
    frames = pd.DataFrame(
        {"period_id": 1, "frame_id": np.arange(100, 131), "time_seconds": np.arange(31) * 1.0, "is_ball": True}
    )
    sim = S.simulated_skillcorner_actions(a, frames)
    corner = sim.iloc[1]
    assert corner["action_provenance"] == "derived" and math.isnan(corner["start_x"]) and pd.isna(corner["player_id"])
    assert (corner["gi_marker_possession_frame_end"], corner["gi_next_possession_frame_start"]) == (109, 111)
    assert sim.iloc[3]["action_provenance"] != "derived"  # a penalty never creates a derived restart row


def test_pass_all_gate_passes_every_key_and_except_signal_withholds_one():
    assert S.PASS_ALL.get(("p", "corner", "kick_time", "x")) is not None
    off = S._ExceptSignal("z_overhead")
    assert off.get(("p", "*", "z_overhead", "-")) is None and off.get(("p", "*", "footprint", "-")) is not None


# ---------------------------------------------------------------------------
# Walker cores in-process on the toy corpus, then assemble -> derive_gate_table (no KeyError)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    from scripts._driver_sources import staged_part2_refs
    from scripts._toy_corpora import build_part2_corpus

    root = tmp_path_factory.mktemp("tf46p2")
    build_part2_corpus(root)
    return {r.stem: S.load_part2_match(r, root / "toytf46") for r in staged_part2_refs(root, "toytf46")}


def test_truth_unit_emits_k_gate_rows_and_characterisation(toy):
    from scripts.tf46p2_truth import characterise, truth_unit

    rec = P.split_records([truth_unit(toy["sportec__301"])])
    assert set(rec["score"]["signal"]) == {"kick_time"} and set(rec["score"]["cell_state"]) == {"k_gate"}
    char = characterise(rec["delta"])
    assert {"a_s", "b_s", "outside_share", "decision_time"} <= set(char.columns)
    assert P.split_records([truth_unit(toy["skillcorner__501"])]) == {}  # not a reference provider


def test_penalty_unit_runs_the_shootout_check_first_and_scores(toy):
    from scripts.tf46p2_penalty import penalty_unit

    rec = P.split_records([penalty_unit(toy["sportec__301"], {})])
    assert rec["shootout"].iloc[0]["n_penalties"] == 1
    assert "detection" in set(rec["score"]["cell_state"])
    assert len(rec["sens"]) == 2 * 12  # every penalty constant at +/-50 %


def test_assemble_then_derive_gate_table_without_key_error(toy, tmp_path):
    from scripts.tf46p2_gate_table import assemble, coverage
    from scripts.tf46p2_origin import origin_unit
    from scripts.tf46p2_truth import truth_unit

    runs = {
        "truth": [truth_unit(toy[k]) for k in sorted(toy)],
        "origin": [origin_unit(toy[k], {}, frozenset()) for k in sorted(toy)],
    }
    for run, parts in runs.items():
        rec = P.split_records(parts)
        agg = S.aggregate_scores(S.scores_frame(rec["score"].to_dict("records")))
        assert list(agg.columns) == list(AGGREGATE_COLUMNS)
        agg.to_parquet(tmp_path / f"{run}.parquet", index=False)
        (tmp_path / f"{run}.json").write_text(json.dumps({"run_commit": "c1"}), encoding="utf-8")
    table, prov = assemble(tmp_path, expect_commit="c1")
    assert set(table["source_run"]) == {"truth", "origin"} and prov["penalty"] is None
    derive_gate_table(table)  # the C3 derivation reads the assembled aggregate with no KeyError
    assert not coverage(table).empty
    with pytest.raises(RuntimeError, match="C-03"):
        assemble(tmp_path, expect_commit="other")


def test_part2_heavy_unit_sorts_first_and_is_heavier():
    # The ADR-128 parity non-vacuity (completion_out_of_order) needs the FIRST-sorted unit to be the heavy one.
    from scripts._toy_corpora import PART2_HEAVY_REPEATS, PART2_HEAVY_UNIT, PART2_TOY_UNITS, part2_toy_match

    assert min(f"{p}__{g}" for p, g in PART2_TOY_UNITS) == "__".join(PART2_HEAVY_UNIT)
    heavy = part2_toy_match(PART2_HEAVY_UNIT[1], PART2_HEAVY_UNIT[0], repeats=PART2_HEAVY_REPEATS)
    light = part2_toy_match(PART2_HEAVY_UNIT[1], PART2_HEAVY_UNIT[0])
    assert len(heavy["frames"]) == PART2_HEAVY_REPEATS * len(light["frames"])
    assert heavy["actions"]["time_seconds"].max() < 2700.0  # inside one 45-minute period


def test_kick_episode_table_reports_multi_episode_share_and_first_vs_last_k():
    # DL ruling 2026-10-22 (C2-IMPL-19 item 6): per profile x type, the share of traces with >= 2 qualifying rest
    # episodes and the first-vs-last K difference on the multi-release ones. A diagnostic, never a tuning input.
    import pandas as pd

    from scripts._tf46p2_score import kick_episode_table

    rows = pd.DataFrame(
        [
            {
                "data_profile": "p",
                "restart_type": "corner",
                "n_qualifying": 1,
                "n_released": 1,
                "k_first": 3.0,
                "k_last": 3.0,
            },
            {
                "data_profile": "p",
                "restart_type": "corner",
                "n_qualifying": 2,
                "n_released": 2,
                "k_first": 1.0,
                "k_last": 3.0,
            },
            {
                "data_profile": "p",
                "restart_type": "corner",
                "n_qualifying": 2,
                "n_released": 1,
                "k_first": 2.0,
                "k_last": 2.0,
            },
        ]
    )
    t = kick_episode_table(rows).iloc[0]
    assert t["n"] == 3 and abs(t["share_multi_episode"] - 2 / 3) < 1e-12
    assert t["n_multi_release"] == 1 and t["median_diff_s"] == 2.0 and t["max_diff_s"] == 2.0
    assert kick_episode_table(pd.DataFrame()).empty


def test_strike_guard_table_flags_both_sides():
    # Spec amendment 2026-10-22: B-41 short-vs-crossed coverage rule and the blocked-vs-not none-rate rule.
    import pandas as pd

    from scripts._tf46p2_score import kick_strike_guard_table

    def rows(spec):
        # spec: (subtype, outcome_class, n_rows, n_none)
        out = []
        for sub, cls, n, n_none in spec:
            for i in range(n):
                out.append(
                    {
                        "data_profile": "p",
                        "restart_type": "free_kick",
                        "subtype": sub,
                        "outcome_class": cls,
                        "reason": "no_strike_ambiguous" if i < n_none else "strike",
                        "retake": False,
                        "retake_first_peak_ms": None,
                    }
                )
        return pd.DataFrame(out)

    balanced = [
        ("freekick_short", "blocked", 5, 1),
        ("freekick_short", "not_blocked", 5, 1),
        ("freekick_crossed", "blocked", 5, 1),
        ("freekick_crossed", "not_blocked", 5, 1),
    ]
    ok = kick_strike_guard_table(rows(balanced)).iloc[0]
    assert not bool(ok["v_kick_returns_to_owner"]) and not bool(ok["rule_returns_to_owner"])
    skewed = [
        ("freekick_short", "blocked", 5, 4),  # short coverage 0.5 vs crossed 1.0; blocked none 0.4 vs 0.0
        ("freekick_short", "not_blocked", 5, 1),
        ("freekick_crossed", "blocked", 5, 0),
        ("freekick_crossed", "not_blocked", 5, 0),
    ]
    bad = kick_strike_guard_table(rows(skewed)).iloc[0]
    assert bool(bad["v_kick_returns_to_owner"]) and bool(bad["rule_returns_to_owner"])
    assert kick_strike_guard_table(pd.DataFrame()).empty


def test_blocked_class_from_native_flag_then_next_opponent_touch_near_the_wall():
    import pandas as pd

    from scripts._tf46p2_score import _blocked_classes

    actions = pd.DataFrame(
        {
            "action_id": [0, 1, 2, 3],
            "period_id": [1, 1, 1, 1],
            "time_seconds": [10.0, 10.8, 30.0, 31.0],
            "team_id": ["H", "A", "H", "H"],
            "start_x": [70.0, 79.0, 70.0, 85.0],
            "start_y": [30.0, 30.0, 30.0, 30.0],
            "shot_blocked": pd.array([pd.NA, pd.NA, pd.NA, pd.NA], dtype="boolean"),
            "cross_blocked": pd.array([pd.NA, pd.NA, pd.NA, pd.NA], dtype="boolean"),
        }
    )
    restarts = pd.DataFrame(
        {
            "restart_id": [0, 2],
            "period_id": [1, 1],
            "t_event": [10.0, 30.0],
            "team_id": ["H", "H"],
            "x": [70.0, 70.0],
            "y": [30.0, 30.0],
        }
    )
    # restart 0: the next action is the OTHER team 9 m away (the wall) -> blocked; restart 2: own team next -> not.
    assert _blocked_classes(actions, restarts) == ["blocked", "not_blocked"]
    flagged = actions.assign(cross_blocked=pd.array([pd.NA, pd.NA, False, pd.NA], dtype="boolean"))
    flagged.loc[0, "cross_blocked"] = False  # a KNOWN native flag wins over the sequence rule
    assert _blocked_classes(flagged, restarts) == ["not_blocked", "not_blocked"]


def test_kick_cells_calibrate_runs_the_gi_direction_corpus_guard(monkeypatch):
    # C2 r2 B-13: the R.2 walker that consumes SkillCorner GI stops on a systematic direction mismatch (spec 4.3)
    # BEFORE any cell is scored -- here both periods of one match disagree with GoalMap.
    import pytest

    import scripts.tf46p2_kick_cells as kc

    rows = [{"data_profile": "p", "match_id": "m0", "period_id": p, "agrees": False} for p in (1, 2)]
    rows += [{"data_profile": "p", "match_id": f"m{i}", "period_id": 1, "agrees": True} for i in range(1, 400)]
    monkeypatch.setattr(kc, "done_results", lambda ctx, stage: [P.records_frame({"gi_direction": rows})])
    with pytest.raises(AssertionError, match="BOTH periods"):
        kc._calibrate_work("only", ctx=None)


def test_aggregate_carries_the_paired_candidate_errors_for_the_agreement_winner():
    # Spec 6.5 (amended 2026-10-22): run (2) kick_time rows carry |t_contact - K| and |t_onset - K| where both were
    # observed; the aggregate stores each match's median of each, the input of the derived agreement winner.
    from scripts._tf46p2_score import aggregate_scores, score_row, scores_frame
    from silly_kicks.tracking._restart_validation import AGGREGATE_COLUMNS, DataProfile, agreement_winner

    prof = DataProfile("skillcorner", "r1", "g1", "2025")
    rows = [
        score_row(
            prof,
            "corner",
            "kick_time",
            "both",
            base="idsse",
            tier="transfer",
            match_id=m,
            err=0.05,
            err_contact=ec,
            err_onset=eo,
        )
        for m, ec, eo in (("m1", 0.10, 0.04), ("m1", 0.20, 0.06), ("m2", 0.12, 0.02))
    ]
    agg = aggregate_scores(scores_frame(rows))
    assert {"agree_err_contact_s", "agree_err_onset_s"} <= set(AGGREGATE_COLUMNS)
    m1 = agg[agg["match_id"] == "m1"].iloc[0]
    assert m1["agree_err_contact_s"] == pytest.approx(0.15) and m1["agree_err_onset_s"] == pytest.approx(0.05)
    assert agreement_winner(agg) == "ball_onset"
