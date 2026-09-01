"""TF-19 A+2 Task 5: the owner-run instrument-responsiveness driver (map + reduce).

The corpus map (`for_each` over matches, one per-frame shard per match) is owner-run; these tests pin
the NOVEL reduce logic -- the POOL (per-arm pooled-corpus statistics, never per shard), the verdict
reduce (incl. the Layer-0 zero-baseline guard + recorded medians), the DISTINCT nd/placebo quantities,
the per-KEEPER sign table (the "A" deliverable), and the §6.1 `gate_eligible` census (E2: `min_nonzero`
binds on the zero-dominated ΔDAS arm). Full provenance wiring is enforced by
`tests/scripts/test_provenance_wiring.py` (the driver is enrolled in ARTIFACT_DRIVERS).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.build_tf19_instrument_responsiveness import (
    _PLACEBO_COLS,
    _named_keeper_signs,
    pool_shards,
    reduce_layer_verdicts,
)
from silly_kicks.gkdv import aggregate_by_keeper
from silly_kicks.gkdv._probe import MIN_DOMAIN_FRAMES


def _shard(nrows, *, saturating=0.5, realistic=0.05, nd=0.10, placebo=0.10):
    """A per-frame shard as the for_each map emits it: one row per scored frame. `nd_abs` (nearest
    defender) and the R placebo columns are DISTINCT single-player quantities."""
    data = {
        "arm": ["delta_das"] * nrows,
        "realistic_abs": np.full(nrows, realistic),
        "saturating_abs": np.full(nrows, saturating),
        "gk_abs": np.full(nrows, 0.30),  # Regime-I keeper delta
        "nd_abs": np.full(nrows, nd),  # nearest-defender control -> nd_med
    }
    for c in _PLACEBO_COLS:
        data[c] = np.full(nrows, placebo)  # placebo band -> placebo p95
    return pd.DataFrame(data)


def test_two_thin_shards_pool_to_valid_but_one_alone_is_unscoreable():
    half = MIN_DOMAIN_FRAMES // 2 + 1  # each shard < floor; two shards >= it
    pooled = pool_shards([_shard(half), _shard(half)])
    assert pooled["delta_das"]["n_domain"] >= MIN_DOMAIN_FRAMES
    assert reduce_layer_verdicts(pooled)["delta_das"]["layer0"] == "instrument_valid"
    # Non-vacuity: a per-shard implementation would see only `half` (< floor) -> arm_unscoreable.
    one = pool_shards([_shard(half)])
    assert one["delta_das"]["n_domain"] < MIN_DOMAIN_FRAMES
    assert reduce_layer_verdicts(one)["delta_das"]["layer0"] == "arm_unscoreable"


def test_dead_instrument_pools_to_void_not_vacuously_valid():
    # saturating (0.04) clears NEITHER 5x realistic (0.25) NOR placebo p95 (0.10) -> void.
    n = MIN_DOMAIN_FRAMES + 5
    v = reduce_layer_verdicts(pool_shards([_shard(n, saturating=0.04)]))["delta_das"]
    assert v["layer0"] == "instrument_void"


def test_zero_realistic_baseline_pools_to_void_not_vacuously_valid():
    # BLOCKING 2 at driver grain: real_med == 0 must NOT vacuously validate via `sat >= 5*0`. A dead
    # instrument (sat below placebo p95) reads void; a real response (sat above placebo) reads valid.
    n = MIN_DOMAIN_FRAMES + 5
    dead = reduce_layer_verdicts(pool_shards([_shard(n, realistic=0.0, saturating=0.04)]))["delta_das"]
    assert dead["layer0"] == "instrument_void"
    live = reduce_layer_verdicts(pool_shards([_shard(n, realistic=0.0, saturating=0.5)]))["delta_das"]
    assert live["layer0"] == "instrument_valid"  # via the placebo backstop


def test_layer1_responsive_when_gk_beats_ratio_times_controls():
    n = MIN_DOMAIN_FRAMES + 5
    v = reduce_layer_verdicts(pool_shards([_shard(n)]))["delta_das"]
    assert v["layer1"] == "responsive"  # gk 0.30 >= 2.0 * max(nd_med 0.10, placebo p95 0.10)


def test_nd_and_placebo_are_distinct_quantities():
    # The whole point of single-player controls: nd_med (nearest) and placebo_p95 (band) differ, so the
    # Layer-1 `max(nd_med, placebo_p95)` is meaningful (a combined control would collapse them).
    n = MIN_DOMAIN_FRAMES + 5
    s = pool_shards([_shard(n, nd=0.30, placebo=0.05)])["delta_das"]
    assert s["nd_med"] == 0.30 and s["placebo_p95"] == 0.05  # distinct arrays


def test_reduce_records_the_medians_for_auditability():
    n = MIN_DOMAIN_FRAMES + 5
    v = reduce_layer_verdicts(pool_shards([_shard(n)]))["delta_das"]
    assert set(v["medians"]) == {"real_med", "sat_med", "gk_med", "nd_med", "placebo_p95"}
    assert v["medians"]["real_med"] == 0.05 and v["medians"]["gk_med"] == 0.30


def test_pool_is_empty_on_no_shards():
    assert pool_shards([]) == {}
    assert reduce_layer_verdicts({}) == {}


def _keeper_frames(*, keeper, signed, n, depth):
    return pd.DataFrame(
        {
            "arm": ["delta_das"] * n,
            "keeper_key": [keeper] * n,
            "game_id": [1] * (n // 2) + [2] * (n - n // 2),  # 2 distinct games
            "period_id": [1] * n,
            "frame_id": list(range(n)),
            "realistic_signed": [signed] * n,  # nonzero -> n_nonzero == n
            "keeper_gr_depth": [depth] * n,
        }
    )


def test_named_keeper_signs_emits_per_keeper_table_with_signs():
    # THE "A" deliverable: a per-keeper sign table + a face-validity read. keeper 1 is a deterrent
    # (negative), keeper 2 is not (positive); the arm's expected direction is "negative".
    n = 25  # >= min_nonzero(20), 2 games -> both gate_eligible
    combined = pd.concat(
        [
            _keeper_frames(keeper=1, signed=-0.1, n=n, depth=6.0),
            _keeper_frames(keeper=2, signed=+0.1, n=n, depth=9.0),
        ],
        ignore_index=True,
    )
    summary, per_keeper = _named_keeper_signs(combined, min_nonzero=20, min_games=2)
    assert set(per_keeper["player_id"]) == {1, 2}
    k1 = per_keeper[per_keeper["player_id"] == 1].iloc[0]
    assert k1["observed_sign"] == "negative" and bool(k1["sign_matches_expected"])
    k2 = per_keeper[per_keeper["player_id"] == 2].iloc[0]
    assert k2["observed_sign"] == "positive" and not bool(k2["sign_matches_expected"])
    assert summary["n_eligible_sign_matches_expected"] == 1  # only keeper 1 matches
    assert summary["n_keepers"] == 2 and summary["n_gate_eligible"] == 2


def test_named_keeper_signs_counts_unresolved_keeper_frames_not_drops_them():
    # S4.3 dropped-AND-counted: NA keeper_key rows are counted, not silently dropped.
    n = 25
    resolved = _keeper_frames(keeper=1, signed=-0.1, n=n, depth=6.0)
    unresolved = _keeper_frames(keeper=pd.NA, signed=-0.1, n=4, depth=6.0)
    summary, per_keeper = _named_keeper_signs(
        pd.concat([resolved, unresolved], ignore_index=True), min_nonzero=20, min_games=2
    )
    assert summary["n_unresolved_keeper_frames"] == 4
    assert set(per_keeper["player_id"]) == {1}  # the NA keeper is not a fabricated row


def test_gate_eligible_census_binds_on_min_nonzero_for_das():
    # E2: a keeper in >=2 matches but <20 nonzero dDAS obs is NOT gate_eligible (min_nonzero binds).
    obs = pd.DataFrame(
        {
            "player_id": [7] * 25,
            "game_id": [1] * 13 + [2] * 12,  # 2 distinct games -> clears min_games
            "delta_das": [0.0] * 20 + [-0.1] * 5,  # only 5 nonzero -> fails min_nonzero=20
        }
    )
    row = aggregate_by_keeper(obs, value_col="delta_das", min_nonzero=20, min_games=2)
    row = row[row["player_id"] == 7].iloc[0]
    assert int(row["n_games"]) == 2 and int(row["n_nonzero"]) == 5
    assert not bool(row["gate_eligible"])  # min_nonzero binds, not min_games


def test_named_keeper_prior_is_locked_and_confirmatory():
    # S4.4: the LOCKED named prior + the confirmatory face-validity check. Alisson reads negative
    # (matches the deterrent expectation); Neuer here reads positive (does NOT), so the check reports it.
    from scripts.build_tf19_instrument_responsiveness import NAMED_KEEPER_PRIOR, _named_keeper_check

    assert NAMED_KEEPER_PRIOR == {"Alisson": "negative", "Neuer": "negative"}
    per_keeper = pd.DataFrame(
        {
            "player_id": [3512, 9999],
            "mean": [-0.1, 0.1],
            "observed_sign": ["negative", "positive"],
            "gate_eligible": [True, True],
        }
    )
    pk, check = _named_keeper_check(per_keeper, {"3512": "Alisson Becker", "9999": "Manuel Neuer"})
    assert list(pk["keeper_name"]) == ["Alisson Becker", "Manuel Neuer"]  # injected names joined
    assert check["Alisson"]["n_matched_keepers"] == 1 and check["Alisson"]["meets_prior"] is True
    assert check["Neuer"]["meets_prior"] is False  # observed positive != expected negative


def test_named_keeper_check_unresolved_name_does_not_falsely_confirm():
    # Non-vacuity: a named-prior keeper the name map does not resolve must NOT read as confirmed.
    from scripts.build_tf19_instrument_responsiveness import _named_keeper_check

    per_keeper = pd.DataFrame(
        {"player_id": [1], "mean": [-0.1], "observed_sign": ["negative"], "gate_eligible": [True]}
    )
    _pk, check = _named_keeper_check(per_keeper, {"1": "Some Other Keeper"})
    assert check["Alisson"]["n_matched_keepers"] == 0
    assert check["Alisson"]["meets_prior"] is False  # unresolved -> not a fabricated pass


def test_driver_exposes_main_and_reduce_seams():
    import scripts.build_tf19_instrument_responsiveness as d

    assert hasattr(d, "main")
    assert callable(d.pool_shards) and callable(d.reduce_layer_verdicts)
    assert callable(d._named_keeper_signs) and callable(d._provider_support_matrix)
