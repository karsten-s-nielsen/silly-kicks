"""Task 2.1 contract: the data profile, ratified constants, statistics, and derived gate table
(`silly_kicks.tracking._restart_validation`, TF-46 Arm 1 Part 2, spec sections 5.3, 15, 16).

Every bar is tested from both sides (spec section 3)."""

from __future__ import annotations

import inspect
import math
from dataclasses import FrozenInstanceError, fields

import pandas as pd
import pytest

from silly_kicks.tracking import _restart_validation as rv
from silly_kicks.tracking._restart_validation import DataProfile, GateEntry

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

_PROFILE_KW = dict(provider="skillcorner", tracking_release="r1", gi_release="g1", competition_season="2022")
PROFILE = DataProfile(**_PROFILE_KW)


def _hits(matches: list[tuple[str, int, int]]) -> pd.DataFrame:
    """A per-row (match_id, ok) frame from per-match (match_id, k, n) counts."""
    mid: list[str] = []
    ok: list[bool] = []
    for m, k, n in matches:
        mid.extend([m] * n)
        ok.extend([True] * k + [False] * (n - k))
    return pd.DataFrame({"match_id": mid, "ok": ok})


def _agg_block(
    signal: str,
    restart_type: str,
    cell_state: str,
    matches: list[tuple[str, int, int]],
    *,
    base: str = "idsse",
    admitted: bool = False,
    stratum: str = "",
    tier: str = "reference",
    parent_cell_state: str = "",
    lomo: float = 0.95,
    calib: float = 0.0,
    ceiling: float = float("nan"),
    footprint: float = float("nan"),
    baseline_lb: float = float("nan"),
) -> pd.DataFrame:
    rows = []
    for m, k, n in matches:
        rows.append(
            {
                **_PROFILE_KW,
                "restart_type": restart_type,
                "signal": signal,
                "cell_state": cell_state,
                "base": base,
                "admitted": admitted,
                "stratum": stratum,
                "parent_cell_state": parent_cell_state,
                "match_id": m,
                "k": k,
                "n": n,
                "tier": tier,
                "lomo_metric": lomo,
                "calibration_metric": calib,
                "calibration_ceiling": ceiling,
                "footprint_inside_share": footprint,
                "baseline_margin_lb": baseline_lb,
            }
        )
    return pd.DataFrame(rows)


_ALL_OK_5x8 = [(f"m{i}", 8, 8) for i in range(5)]  # n=40, m=5, all correct: a passing cell


# ---------------------------------------------------------------------------
# Wilson
# ---------------------------------------------------------------------------


def test_wilson_lower_both_sides_and_empty():
    assert rv.wilson_lower(35, 35) == pytest.approx(0.9011, abs=1e-3)
    assert rv.wilson_lower(35, 35) >= 0.90
    assert rv.wilson_lower(34, 34) < 0.90
    assert math.isnan(rv.wilson_lower(0, 0))


# ---------------------------------------------------------------------------
# cell_passes
# ---------------------------------------------------------------------------


def test_cell_passes_fails_on_too_few_matches():
    # n = 35 all correct, but only m = 4 matches.
    hits = _hits([("m0", 9, 9), ("m1", 9, 9), ("m2", 9, 9), ("m3", 8, 8)])
    assert len(hits) == 35 and hits["match_id"].nunique() == 4
    assert rv.cell_passes(hits) is False


def test_cell_passes_fails_when_errors_cluster_in_one_match():
    # Wilson (independent-row) lower bound clears 0.90, but one bad match carries every error, so the
    # match-cluster bootstrap lower bound falls well below 0.90.
    matches = [(f"good{i}", 200, 200) for i in range(4)] + [("bad", 120, 200)]
    hits = _hits(matches)
    k = int(hits["ok"].sum())
    assert rv.wilson_lower(k, len(hits)) >= 0.90
    assert rv.cluster_bootstrap_lower(hits) < 0.90
    assert rv.cell_passes(hits) is False


def test_cell_passes_accepts_a_clean_large_cell():
    assert rv.cell_passes(_hits(_ALL_OK_5x8)) is True


# ---------------------------------------------------------------------------
# noninferior
# ---------------------------------------------------------------------------


def test_noninferior_both_sides_and_small_n():
    parent = _hits([(f"p{i}", 19, 20) for i in range(5)])  # 0.95 every match
    child_05 = _hits([(f"c{i}", 9, 10) for i in range(5)])  # 0.90 every match -> diff -0.05
    child_15 = _hits([(f"c{i}", 8, 10) for i in range(5)])  # 0.80 every match -> diff -0.15
    assert rv.noninferior(child_05, parent) is True
    assert rv.noninferior(child_15, parent) is False
    assert rv.noninferior(_hits([("c", 18, 19)]), parent) is False  # child n = 19 < 20


# ---------------------------------------------------------------------------
# choose_base
# ---------------------------------------------------------------------------


def test_choose_base_precedence():
    assert rv.choose_base({"idsse": (40, 5)}, frozenset({"idsse"}), same_meaning=False) == "idsse"
    assert (
        rv.choose_base(
            {"idsse": (10, 2), "gradientsports": (40, 5)},
            frozenset({"idsse", "gradientsports"}),
            same_meaning=False,
        )
        == "gradientsports"
    )
    assert (
        rv.choose_base(
            {"idsse": (40, 5), "gradientsports": (60, 6)},
            frozenset({"idsse", "gradientsports"}),
            same_meaning=True,
        )
        == "gradientsports"
    )
    assert (
        rv.choose_base(
            {"idsse": (80, 6), "gradientsports": (40, 5)},
            frozenset({"idsse", "gradientsports"}),
            same_meaning=True,
        )
        == "idsse"
    )
    assert rv.choose_base({"idsse": (10, 2)}, frozenset({"idsse"}), same_meaning=False) is None


def test_choose_base_takes_no_outcome_input():
    assert set(inspect.signature(rv.choose_base).parameters) == {"adequacy", "admitted", "same_meaning"}


# ---------------------------------------------------------------------------
# gate() fail-closed
# ---------------------------------------------------------------------------


def test_shipped_gate_table_equals_the_derivation_of_the_committed_aggregate():
    # Plan Task 2.14 Step 1 (ADR-056): RESTART_GATE_TABLE == derive_gate_table(read_gate_aggregate()) and
    # GATE_TABLE_VERSION == the aggregate's sha256; a missing aggregate (commit 2) derives to {} / "unpinned".
    from scripts._tf46p2_common import gate_table_derivation_mismatches, read_gate_aggregate

    aggregate, version = read_gate_aggregate()
    assert gate_table_derivation_mismatches(rv.RESTART_GATE_TABLE, rv.GATE_TABLE_VERSION, aggregate, version) == []


def test_the_derivation_gate_fails_when_table_and_aggregate_disagree(tmp_path):
    # Anti-rot (ADR-051 / ADR-056): the check is not vacuous -- a passing cell in the aggregate that the shipped
    # table lacks, a stale entry, and a version drift each register.
    from scripts._tf46p2_common import gate_table_derivation_mismatches, read_gate_aggregate

    agg = _agg_block("penalty", "penalty", "detection", _ALL_OK_5x8)
    derived = rv.derive_gate_table(agg)
    assert derived  # precondition: the synthetic aggregate derives a non-empty table
    assert gate_table_derivation_mismatches({}, "unpinned", agg, "unpinned")  # missing derived entry
    assert gate_table_derivation_mismatches(derived, "unpinned", agg, "sha") == [
        "GATE_TABLE_VERSION 'unpinned' != aggregate version 'sha'"
    ]
    assert gate_table_derivation_mismatches(derived, "sha", pd.DataFrame(columns=list(rv.AGGREGATE_COLUMNS)), "sha")
    assert gate_table_derivation_mismatches(derived, "sha", agg, "sha") == []  # consistent -> clean
    agg.to_parquet(tmp_path / "gate_aggregate.parquet", index=False)  # the reader pins the file's sha256
    read, ver = read_gate_aggregate(tmp_path)
    assert len(read) == len(agg) and len(ver) == 64 and ver != "unpinned"


def test_gate_is_fail_closed_on_the_empty_shipped_table():
    before = rv.withheld_lookups
    assert rv.gate(PROFILE, "corner", "kick_time", "both") is None
    assert rv.gate(PROFILE, "*", "footprint", "-") is None
    assert rv.withheld_lookups == before + 2
    assert dict(rv.RESTART_GATE_TABLE) == {}
    assert rv.GATE_TABLE_VERSION == "unpinned"


# ---------------------------------------------------------------------------
# RATIFIED (anti-drift literal) and sensitivity
# ---------------------------------------------------------------------------

_EXPECTED_RATIFIED = {
    "tau_event_ceiling_s": 0.40,
    "tau_event_decision_flag_s": 1.0,
    "tau_detector_ceiling_s": 0.30,
    "tau_fc_ceiling_s": 0.40,
    "bracket_clip_s": 2.0,
    "bracket_outlier_share": 0.02,
    "origin_median_m": 2.0,
    "origin_share_m": 5.0,
    "contested_threshold_m": 0.75,
    "contested_min_m": 0.5,
    "contested_max_m": 1.0,
    "penalty_spot_r_m": 1.0,
    "penalty_rest_s": 1.0,
    "penalty_position_eval_s": 0.5,
    "penalty_d_gk_m": 2.0,
    "penalty_gk_lateral_m": 1.0,
    "penalty_gk_y_tol_m": 4.66,
    "penalty_e_enc_m": 1.0,
    "penalty_enc_disk_m": 9.15,
    "penalty_enc_count_k": 2,
    "penalty_taker_near_m": 3.0,
    "penalty_W_s": 3.0,
    "penalty_W_cross_extra_s": 0.40,
    "penalty_W_rb_s": 8.0,
    "penalty_W_rb_ceiling_s": 10.0,
    "penalty_rb_live_speed_ms": 3.0,
    "penalty_rb_live_min_m": 3.0,
    "retake_rho_penalty_m": 1.0,
    "retake_rho_freekick_m": 1.0,
    "retake_rho_corner_m": 1.5,
    "merge_r_m": 2.0,
    "merge_dt_s": 2.0,
    "m_min": 5,
    "k_rest_m": 0.5,
    "k_rest_s": 0.5,
    "k_release_disp_m": 0.5,
    "k_release_rest_sd_mult": 3.0,
    "k_release_speed_ms": 3.0,
    "k_release_frames": 2,
    "throwin_release_speed_ms": 4.0,
    "throwin_release_dist_m": 0.3,
    "throwin_release_window_s": 0.2,
    "sc_freekick_vkick_ms": 8.0,
    "sc_freekick_ambiguous_share": 0.15,
    "sc_freekick_region_m": 5.0,
    "sc_freekick_short_cover_margin": 0.10,
    "corner_region_m": 10.0,
    "throwin_region_m": 3.0,
    "freekick_region_m": 3.0,
    "corner_rest_m": 1.5,
    "rstar_window_s": 6.0,
    "rstar_flag_m": 10.0,
    "pool_child_min_n": 20,
    "pool_noninferior_margin": 0.10,
    "gi_direction_min_possessions": 5,
    "gi_direction_mismatch_share": 0.01,
    "pair_prox_m": 2.0,
    "pair_prox_share": 0.245,  # derived max per-profile bar (0.24407) rounded up, R.2 rulings 2026-10-23
    "pair_prox_margin": 0.10,
    "pair_neg_quantile": 0.99,
    "pair_lag_tol_frames": 2,  # R.2 second-stop ruling 2026-10-23
    "pair_lag_scan_frames": 20,
    "cam_seg_s": 10.0,
    "cam_stitch_k": 5,
    "cam_hold_report_s": 2.0,
    "footprint_inside_share": 0.99,
    "footprint_tol_m": 1.0,
    "wilson_z": 1.96,
    "bootstrap_b": 2000,
    "bootstrap_q": 0.025,
    "n_min": 35,
}


def test_ratified_is_the_literal_section_16_1_copy():
    assert dict(rv.RATIFIED) == _EXPECTED_RATIFIED


def test_sensitivity_values_round_integer_constants():
    assert rv.sensitivity_values("pair_lag_tol_frames") == (1, 3)
    assert rv.sensitivity_values("pair_lag_scan_frames") == (10, 30)
    assert rv.sensitivity_values("cam_stitch_k") == (2, 8)
    assert rv.sensitivity_values("cam_seg_s") == (5.0, 15.0)


def test_sensitivity_of_a_share_moves_half_the_distance_to_the_nearer_bound():
    # DL ruling 2026-10-23: +/-50 % of min(s, 1 - s); a share near 1 stays a share
    lo, hi = rv.sensitivity_values("footprint_inside_share")  # 0.99 -> 0.985 / 0.995
    assert lo == pytest.approx(0.985) and hi == pytest.approx(0.995)
    lo, hi = rv.sensitivity_values("pair_prox_margin")  # 0.10 -> 0.05 / 0.15
    assert lo == pytest.approx(0.05) and hi == pytest.approx(0.15)
    lo, hi = rv.sensitivity_values("pair_prox_share")  # 0.245 -> 0.1225 / 0.3675
    assert lo == pytest.approx(0.1225) and hi == pytest.approx(0.3675)


@pytest.mark.parametrize("key", sorted(rv.RATIFIED))
def test_every_sensitivity_value_is_valid_for_its_constant(key):
    # shares in [0, 1]; frame / count constants non-negative integers; distances / times / other floats > 0
    v = rv.RATIFIED[key]
    pair = rv.sensitivity_values(key)
    if key in rv.SHARE_CONSTANTS:
        assert 0.0 <= v <= 1.0 and all(0.0 <= x <= 1.0 for x in pair), (key, pair)
    elif isinstance(v, int):
        assert all(isinstance(x, int) and x >= 0 for x in pair), (key, pair)
    else:
        assert v > 0 and all(x > 0 for x in pair), (key, pair)


def test_share_constants_are_ratified_keys():
    assert rv.SHARE_CONSTANTS <= set(rv.RATIFIED)


# ---------------------------------------------------------------------------
# DataProfile has no pairing field; fidelity dependency map (source-path rule)
# ---------------------------------------------------------------------------


def test_data_profile_has_no_pairing_field():
    names = {f.name for f in fields(DataProfile)}
    assert names == {"provider", "tracking_release", "gi_release", "competition_season"}
    assert "pairing" not in names


def test_fidelity_dependencies_put_contact_stats_on_origin_and_penalty():
    contact = {
        "player_visibility_near_ball",
        "nearest_player_visible_at_contact",
        "identity_swap_rate",
        "off_camera_position_error",
    }
    assert contact <= rv.FIDELITY_DEPENDENCIES["origin"]  # B-33 source-path rule
    assert contact <= rv.FIDELITY_DEPENDENCIES["penalty"]  # C-51 split reads a contact
    assert contact.isdisjoint(rv.FIDELITY_DEPENDENCIES["subtype"])  # subtype reads no contact


# ---------------------------------------------------------------------------
# derive_gate_table: footprint gate both sides
# ---------------------------------------------------------------------------


def _footprint_block(share: float) -> pd.DataFrame:
    return _agg_block("footprint", "*", "-", [("m0", 0, 0)], base="transfer", tier="transfer", footprint=share)


def test_derive_footprint_both_sides():
    tbl = rv.derive_gate_table(_footprint_block(0.99))
    assert (PROFILE, "*", "footprint", "-") in tbl
    tbl = rv.derive_gate_table(_footprint_block(0.98))
    assert (PROFILE, "*", "footprint", "-") not in tbl


# ---------------------------------------------------------------------------
# derive_gate_table: penalty cell states and strata pooling
# ---------------------------------------------------------------------------


def test_penalty_cell_states_literal():
    assert rv.PENALTY_CELL_STATES == ("detection", "attribution", "split")


def test_derive_penalty_single_population_passes():
    agg = _agg_block("penalty", "penalty", "detection", _ALL_OK_5x8)
    tbl = rv.derive_gate_table(agg)
    entry = tbl[(PROFILE, "penalty", "penalty", "detection")]
    assert entry.cell_resolution == "cell"


def test_derive_penalty_strata_pool_when_each_is_noninferior():
    in_play = _agg_block("penalty", "penalty", "split", [(f"ip{i}", 5, 5) for i in range(5)], stratum="in_play")
    shoot = _agg_block("penalty", "penalty", "split", [(f"so{i}", 5, 5) for i in range(5)], stratum="shoot_out")
    tbl = rv.derive_gate_table(pd.concat([in_play, shoot], ignore_index=True))
    entry = tbl[(PROFILE, "penalty", "penalty", "split")]
    assert entry.cell_resolution == "pooled_strata"


def test_derive_penalty_withholds_when_a_stratum_is_too_small():
    in_play = _agg_block("penalty", "penalty", "attribution", [(f"ip{i}", 5, 5) for i in range(5)], stratum="in_play")
    # shoot_out n = 19 < 20
    shoot = _agg_block(
        "penalty",
        "penalty",
        "attribution",
        [("so0", 5, 5), ("so1", 5, 5), ("so2", 5, 5), ("so3", 4, 4)],
        stratum="shoot_out",
    )
    tbl = rv.derive_gate_table(pd.concat([in_play, shoot], ignore_index=True))
    assert (PROFILE, "penalty", "penalty", "attribution") not in tbl


# ---------------------------------------------------------------------------
# derive_gate_table: C-03 shipped constant vs gate value
# ---------------------------------------------------------------------------


def test_derive_stores_lomo_as_metric_and_withholds_a_calibration_above_ceiling():
    ok = _agg_block("kick_time", "free_kick", "both", _ALL_OK_5x8, lomo=0.93, calib=0.25, ceiling=0.30)
    entry = rv.derive_gate_table(ok)[(PROFILE, "free_kick", "kick_time", "both")]
    assert entry.metric == 0.93  # the LOMO estimate, not the calibration
    over = _agg_block("kick_time", "free_kick", "both", _ALL_OK_5x8, lomo=0.93, calib=0.35, ceiling=0.30)
    assert (PROFILE, "free_kick", "kick_time", "both") not in rv.derive_gate_table(over)


# ---------------------------------------------------------------------------
# derive_gate_table: pooling ladder
# ---------------------------------------------------------------------------


def test_pooling_ladder_lifts_a_noninferior_child():
    parent = _agg_block("kick_time", "free_kick", "both", _ALL_OK_5x8)
    child = _agg_block(
        "kick_time", "free_kick", "both_pos", [(f"c{i}", 5, 5) for i in range(5)], parent_cell_state="both"
    )
    tbl = rv.derive_gate_table(pd.concat([parent, child], ignore_index=True))
    assert tbl[(PROFILE, "free_kick", "kick_time", "both")].cell_resolution == "cell"
    assert tbl[(PROFILE, "free_kick", "kick_time", "both_pos")].cell_resolution == "pooled_both"


def test_first_contact_contested_never_pools_into_uncontested():
    parent = _agg_block("first_contact", "corner", "uncontested", _ALL_OK_5x8)
    child = _agg_block(
        "first_contact",
        "corner",
        "contested_takerknown",
        [(f"c{i}", 5, 5) for i in range(5)],
        parent_cell_state="uncontested",
    )
    tbl = rv.derive_gate_table(pd.concat([parent, child], ignore_index=True))
    assert (PROFILE, "corner", "first_contact", "uncontested") in tbl
    assert (PROFILE, "corner", "first_contact", "contested_takerknown") not in tbl


@pytest.mark.parametrize(
    "signal,restart_type,cell_state,expected",
    [
        ("subtype", "corner", "left", False),
        ("origin", "free_kick", "native", False),
        ("merge", "free_kick", "events_first", False),
        ("kick_time", "throw_in", "both_pos", False),
        ("kick_time", "free_kick", "retake", False),
        ("kick_time", "free_kick", "conflict_pos", False),
        ("kick_time", "free_kick", "both_pos", True),
        ("first_contact", "corner", "contested_takerknown", True),
    ],
)
def test_may_pool_never_pool_rules(signal, restart_type, cell_state, expected):
    assert rv._may_pool(signal, restart_type, cell_state) is expected


def test_gate_entry_is_frozen():
    e = GateEntry("reference", "idsse", 0.95, 0.91, 0.91, 40, 5, "cell")
    with pytest.raises(FrozenInstanceError):
        e.metric = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# z_overhead (spec section 14.1): profile-level paired gate
# ---------------------------------------------------------------------------


def _z_rows(provider: str, on: list[tuple[str, int, int]], off: list[tuple[str, int, int]]) -> pd.DataFrame:
    a = _agg_block("z_overhead", "*", "gate_on", on)
    b = _agg_block("z_overhead", "*", "gate_off", off)
    out = pd.concat([a, b], ignore_index=True)
    out["provider"] = provider
    return out


def _z_key(provider: str):
    return (DataProfile(**{**_PROFILE_KW, "provider": provider}), "*", "z_overhead", "-")


def test_z_overhead_passes_when_measured_and_the_gate_does_not_hurt():
    on = [(f"m{i}", 9, 10) for i in range(6)]
    off = [(f"m{i}", 8, 10) for i in range(6)]  # the gate helps on every match
    table = rv.derive_gate_table(_z_rows("sportec", on, off))
    entry = table[_z_key("sportec")]
    assert entry.bootstrap_lb >= 0.0 and entry.metric == pytest.approx(0.1) and entry.m == 6


def test_z_overhead_withheld_when_the_gate_hurts_agreement():
    # The GS case (spec 14.1): measured z, but the overhead gate reduces first-contact agreement.
    on = [(f"m{i}", 5, 10) for i in range(6)]
    off = [(f"m{i}", 6, 10) for i in range(6)]
    assert _z_key("gradientsports") not in rv.derive_gate_table(_z_rows("gradientsports", on, off))


def test_z_overhead_withheld_for_an_estimated_z_provider_even_when_it_helps():
    # declared z_source "measured" is NECESSARY: skillcorner z is estimated -> never passes.
    on = [(f"m{i}", 10, 10) for i in range(6)]
    off = [(f"m{i}", 5, 10) for i in range(6)]
    assert _z_key("skillcorner") not in rv.derive_gate_table(_z_rows("skillcorner", on, off))


def test_z_overhead_withheld_when_on_and_off_are_not_paired():
    on = [(f"m{i}", 9, 10) for i in range(6)]
    off = [(f"m{i}", 8, 11) for i in range(6)]  # different contact counts -> not the same contacts
    assert _z_key("sportec") not in rv.derive_gate_table(_z_rows("sportec", on, off))


def test_paired_bootstrap_lower_is_seeded_and_nan_on_empty():
    pm = pd.DataFrame({"k_on": [9, 8, 7], "k_off": [8, 8, 8], "n": [10, 10, 10]})
    assert rv.paired_bootstrap_lower(pm) == rv.paired_bootstrap_lower(pm)
    assert math.isnan(rv.paired_bootstrap_lower(pm.iloc[:0]))


def test_subtype_cell_must_beat_the_majority_baseline_both_sides():
    # Spec 11.2: a passing precision is not enough -- the bootstrap lower bound of (precision - majority
    # baseline) must be > 0. A constant "crossed" on WON corners already reaches Wilson ~0.81 (the reason).
    key = (PROFILE, "corner", "subtype", "crossed")
    beats = _agg_block("subtype", "corner", "crossed", _ALL_OK_5x8, baseline_lb=0.02)
    ties = _agg_block("subtype", "corner", "crossed", _ALL_OK_5x8, baseline_lb=0.0)
    unknown = _agg_block("subtype", "corner", "crossed", _ALL_OK_5x8)
    assert key in rv.derive_gate_table(beats)
    assert key not in rv.derive_gate_table(ties)
    assert key not in rv.derive_gate_table(unknown)


def test_derivation_writes_the_agreement_winner_on_kick_time_entries():
    # Spec 6.5 / 16.1 (amended 2026-10-22): the smaller median absolute error wins; a tie or no paired error gives
    # the documented default tf79_contact; non-kick_time entries carry none.
    def kt(contact, onset):
        agg = _agg_block("kick_time", "corner", "both", _ALL_OK_5x8)
        return agg.assign(agree_err_contact_s=contact, agree_err_onset_s=onset)

    key = (PROFILE, "corner", "kick_time", "both")
    assert rv.derive_gate_table(kt(0.12, 0.06))[key].agreement_winner == "ball_onset"
    assert rv.derive_gate_table(kt(0.06, 0.12))[key].agreement_winner == "tf79_contact"
    assert rv.derive_gate_table(kt(0.08, 0.08))[key].agreement_winner == "tf79_contact"  # exact tie -> default
    assert rv.derive_gate_table(kt(float("nan"), float("nan")))[key].agreement_winner == "tf79_contact"
    pen = rv.derive_gate_table(_agg_block("penalty", "penalty", "detection", _ALL_OK_5x8))
    assert pen[(PROFILE, "penalty", "penalty", "detection")].agreement_winner is None
