"""Wiring guards for scripts/validate_territorial_defense.py (TF-54b, owner-run battery).

The clean-tree / --allow-dirty / no-rev-parse conformance is asserted by
tests/scripts/test_provenance_wiring.py (this driver is enrolled in ARTIFACT_DRIVERS). Here: the
input contract, the schema-token pin (4.77.1), the per-scored-frame probe-battery measure, the
pooled Layer-0/1 reduce, and the locked elite prior.
"""

import pandas as pd

from scripts import validate_territorial_defense as V
from tests.territorial_defense._fixtures import make_fitted_xt, make_per_action_ltr_fixture


def test_input_contract_declares_a_digest_and_the_locked_prior():
    ic = V.input_contract()
    assert ic["driver"] == "validate_territorial_defense"
    assert ic["digest"]  # ADR-056: a stable digest of the declared symbols
    assert ic["params"]["elite_defender_prior"] == V.ELITE_DEFENDER_PRIOR


def test_input_contract_declares_the_dose_constants():
    # ADR-056: the dose battery constants the verdicts depend on move the digest when changed.
    ic = V.input_contract()
    assert ic["params"]["realistic_disp_m"] == V.REALISTIC_DISP_M
    assert ic["params"]["saturating_disp_m"] == V.SATURATING_DISP_M
    assert ic["params"]["n_placebo"] == V.N_PLACEBO


def test_elite_prior_is_locked_before_the_run_and_all_positive():
    # Positive == the defender's positioning SUPPRESSES threat (attacker-value units).
    assert set(V.ELITE_DEFENDER_PRIOR.values()) == {"positive"}
    assert "before the owner run" in V.ELITE_DEFENDER_PRIOR_LOCKED


def test_expected_direction_is_positive_for_both_arms():
    from silly_kicks.territorial_defense._probe import expected_direction_for_arm

    assert expected_direction_for_arm("a_threat_suppressed") == "positive"
    assert expected_direction_for_arm("b_threat_suppressed") == "positive"


def _sb_item():
    """An SB360-shaped 6-tuple (provider, game_id, actions, frames, home, visible_area) that scores.

    Uses the per-action-LTR fixture (the driver runs the SB360 default ``per_action_ltr``): 3 D
    interceptions -> 3 scored Arm-A battery rows.
    """
    actions, frames = make_per_action_ltr_fixture()
    actions = actions.copy()
    actions["player_name"] = ["Virgil van Dijk" if p == 102 else None for p in actions["player_id"]]
    return ("statsbomb", 1, actions, frames, 1, None)


def test_measure_match_emits_a_battery_row_per_scored_frame():
    shard = V._measure_match(_sb_item(), xt=make_fitted_xt())
    assert list(shard.columns) == V._EMITTED_SHARD_COLUMNS
    battery = shard[shard["row_kind"] == "battery"]
    sample = shard[shard["row_kind"] == "sample"]
    # The fixture has 3 scored Arm-A frames for D (#102) -> one per-frame battery row each, plus one
    # "sample" row per library-samples defender (IMPL-04).
    assert len(battery) == 3
    assert len(sample) >= 1
    assert (battery["player_id"] == 102).all()
    assert (sample["player_id"] == 102).all()
    # 4.77.1 (non-vacuous under the reindex): each declared BATTERY column is a REAL key populated on
    # battery rows, and each SAMPLE column is populated on sample rows -- a dropped key would surface as
    # an all-NaN column on its own row kind, not a phantom reindex-NaN that the column-list check misses.
    for col in ("period_id", "frame_id", "a_delta", "realistic_abs", "saturating_abs", "nd_abs", *V._PLACEBO_COLS):
        assert battery[col].notna().all(), col
    for col in ("a_threat_suppressed", "a_frames_scored", "td_source"):
        assert sample[col].notna().all(), col
    # The battery dose magnitudes are finite on the velocity-less fixture (ADR-063 Tier-1 lift).
    for col in ("realistic_abs", "saturating_abs", "nd_abs"):
        assert battery[col].notna().all()


def test_per_defender_covers_an_arm_b_only_defender():
    # IMPL-04: named_defender_signs reads the shard's "sample" rows (the library samples), so a defender
    # with an Arm-B contribution but NO scored Arm-A frame (hence NO battery row) is NOT dropped. The old
    # battery-re-derivation saw only player 1; the sample-row path must surface player 2 too.
    combined = pd.DataFrame(
        [
            {"row_kind": "battery", "game_id": 1, "player_id": 1, "a_delta": 0.5, "realistic_abs": 1.0},
            {"row_kind": "sample", "game_id": 1, "player_id": 1, "a_threat_suppressed": 0.5},
            # Arm-B-only: no battery row, a_threat_suppressed NaN, a real b_threat_suppressed.
            {
                "row_kind": "sample",
                "game_id": 1,
                "player_id": 2,
                "a_threat_suppressed": float("nan"),
                "b_threat_suppressed": 0.3,
                "b_attribution_slippage": 0.0,
            },
        ]
    ).reindex(columns=V._EMITTED_SHARD_COLUMNS)  # a real combined shard carries the full schema
    per_defender = V._per_defender_from_shards(combined)
    assert set(per_defender["player_id"]) == {1, 2}
    p2 = per_defender[per_defender["player_id"] == 2].iloc[0]
    assert p2["b_threat_suppressed"] == 0.3
    assert pd.isna(p2["a_threat_suppressed"])


def test_measure_match_empty_frames_returns_empty_with_columns():
    actions, _frames = make_per_action_ltr_fixture()
    shard = V._measure_match(("statsbomb", 1, actions, pd.DataFrame(), 1, None))
    assert list(shard.columns) == V._EMITTED_SHARD_COLUMNS
    assert shard.empty


def test_elite_prior_report_reads_the_observed_sign():
    shard = V._measure_match(_sb_item(), xt=make_fitted_xt())
    per_defender = V._per_defender_from_shards(shard)
    rep = V._elite_prior_report(per_defender)
    vd = next(c for c in rep["checks"] if c["defender"] == "Van Dijk")
    assert vd["expected"] == "positive"
    assert vd["observed"] in {"positive", "negative", "no_data"}
    assert vd["n_matches"] >= 1  # Van Dijk (#102) is present + scored in the fixture


def test_pool_shards_and_reduce_produce_verdict_strings():
    from silly_kicks.territorial_defense._probe import LAYER0_VERDICTS, LAYER1_VERDICTS

    # A synthetic per-frame frame that clears MIN_DOMAIN_FRAMES with a saturating dose >> the realistic
    # one and a dosed-defender median well above the control band -> a DISCRIMINATING verdict, not just
    # any string (the invariance-test-needs-discriminating-power discipline).
    n = 250
    df = pd.DataFrame(
        {
            "row_kind": ["battery"] * n,
            "game_id": [1] * n,
            "period_id": [1] * n,
            "frame_id": list(range(n)),
            "player_id": [102] * n,
            "player_name": ["D"] * n,
            "a_delta": [0.5] * n,
            "realistic_abs": [1.0] * n,
            "saturating_abs": [10.0] * n,
            "realistic_signed": [1.0] * n,
            "nd_abs": [0.1] * n,
            "b_threat_suppressed": [0.0] * n,
            "b_frames_scored": [0] * n,
            "b_attribution_slippage": [0.0] * n,
        }
    )
    for c in V._PLACEBO_COLS:
        df[c] = [0.1] * n

    verdicts = V.reduce_layer_verdicts(V.pool_shards([df]))
    assert verdicts["layer0"] in LAYER0_VERDICTS
    assert verdicts["layer1"] in LAYER1_VERDICTS
    # sat_med 10 >= SATURATING_MULTIPLE (5) * real_med 1 -> valid; defender_med 1 >= 2 * 0.1 -> responsive.
    assert verdicts["layer0"] == "instrument_valid"
    assert verdicts["layer1"] == "responsive"

    # A tiny frame (< MIN_DOMAIN_FRAMES) is arm_unscoreable on BOTH layers (the other side of the band).
    tiny = V.reduce_layer_verdicts(V.pool_shards([df.head(3)]))
    assert tiny["layer0"] == "arm_unscoreable"
    assert tiny["layer1"] == "arm_unscoreable"


def test_pool_shards_empty_is_empty_and_reduce_is_empty():
    assert V.pool_shards([]) == {}
    assert V.reduce_layer_verdicts({}) == {}
