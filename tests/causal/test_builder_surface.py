"""PR-1 (ADR-037): the builder must express the §3.3 shot arm purely as arguments,
the outcome axis must be result-conditionable, and the placebo must be cluster-aware."""

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.causal.matching import placebo_shift
from silly_kicks.causal.opportunities import (
    EXPOSURE_WINDOW_SECONDS,
    build_opportunities,
    shot_arm_config,
    xcross_config,
)
from tests.causal._fixtures import META, WIDE, actions, frames, frow, simple_actions

SHOT = spadlconfig.actiontype_id["shot"]
SUCCESS = spadlconfig.result_id["success"]
FAIL = spadlconfig.result_id["fail"]


def test_xcross_default_config_reproduces_legacy_byte_identically():
    f = frames({10.0: 5, 10.2: 5}, {10.0: WIDE, 10.2: WIDE})
    a = simple_actions([("cross", 10.1)])
    legacy = build_opportunities(f, a, home_team_id=5, model_metadata=META)
    explicit = build_opportunities(f, a, home_team_id=5, model_metadata=META, config=xcross_config(META))
    pd.testing.assert_frame_equal(legacy, explicit)


def test_shot_arm_outcome_is_the_anchor_inclusive_success_window():
    """P1 second re-registration: own-result-only made control Y ≡ 0 (controls have no
    anchor action) -- the ATT was confounder-invariant and the entanglement gate dead."""
    cfg = shot_arm_config(META)
    assert cfg.outcome_result_ids == (SUCCESS,)
    assert cfg.outcome_window_seconds == 6.0  # == OUTCOME_WINDOW_SECONDS, the registered value
    assert cfg.outcome_window_anchor_inclusive is True
    assert cfg.extractor == "xs"


def test_saved_shot_yields_zero_scored_shot_yields_one():
    f = frames({10.0: 5, 10.2: 5}, {10.0: WIDE, 10.2: WIDE})
    cfg = shot_arm_config(META)
    saved = simple_actions([("shot", 10.1, FAIL)])
    scored = simple_actions([("shot", 10.1, SUCCESS)])
    y_saved = build_opportunities(f, saved, home_team_id=5, model_metadata=META, config=cfg)["Y"]
    y_scored = build_opportunities(f, scored, home_team_id=5, model_metadata=META, config=cfg)["Y"]
    assert int(y_saved.iloc[0]) == 0
    assert int(y_scored.iloc[0]) == 1


def test_cluster_placebo_reassigns_whole_clusters_under_unequal_sizes():
    """P5 property test: cluster-CONSTANT X_gk + UNEQUAL sizes -- every destination
    cluster must receive exactly ONE source cluster's constant (a row-permuting
    implementation stamping 'cluster' FAILS here)."""
    rng = np.random.default_rng(0)
    sizes = [7, 13, 4, 21, 9]  # unequal, contiguous
    clusters = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    n = len(clusters)
    xb = rng.normal(size=(n, 2))
    xg = np.column_stack([clusters.astype(float), clusters.astype(float)])  # cluster-constant
    z = (rng.uniform(size=n) < 0.4).astype(int)
    y = rng.normal(size=n)
    out = placebo_shift(xb, xg, y, z, n_seeds=3, rng_seed=0, cluster_ids=clusters)
    assert out["permutation_unit"] == "cluster"
    # Re-run ONE permutation step exactly as the implementation does and assert the
    # property on the permuted xg it produced (the pure helper `_cluster_reassign`
    # placebo_shift uses per seed, exercised directly):
    from silly_kicks.causal.matching import _cluster_reassign

    permuted = _cluster_reassign(xg, clusters, np.random.default_rng(1))
    for d in np.unique(clusters):
        vals = np.unique(permuted[clusters == d, 0])
        assert len(vals) == 1  # exactly one source cluster's constant


def test_positive_control_ablation_detects_planted_gk_confounding():
    """Instrument validation (spec §3.3): a PLANTED GK->Z,Y confounder must produce a
    gk shift ABOVE the cluster placebo band. Only the null was pinned before."""
    rng = np.random.default_rng(1)
    n = 800
    xb = rng.normal(size=(n, 2))
    gk_signal = rng.normal(size=n)
    xg = np.column_stack([gk_signal, rng.normal(size=n)])
    z = (1 / (1 + np.exp(-(1.5 * gk_signal))) > rng.uniform(size=n)).astype(int)
    y = 0.8 * gk_signal + 0.3 * z + rng.normal(scale=0.5, size=n)
    clusters = np.repeat(np.arange(40), n // 40)
    out = placebo_shift(xb, xg, y, z, n_seeds=30, rng_seed=0, cluster_ids=clusters)
    from silly_kicks.causal.matching import _att_with_block

    real = abs(_att_with_block(xb, xg, y, z, seed=0) - out["base_att"])
    assert real > out["band_p95"], (real, out["band_p95"])


# --- P1(b): builder-level positive control (the wrong-layer lesson) ---

_ENTRY = 10.0  # spell entry (first in-domain frame)
_SPELL_END = 10.5  # last in-domain frame; the ball exits the domain at entry + 1 s
_EXIT_T = 11.0  # out-of-domain frame closing the spell
_TREATED_SHOT_T = 10.4  # in-spell shot -> Z = 1 by construction
_LATE_GOAL_T = 14.0  # entry + 4 s: past spell_end (Z stays 0), inside Y's window


def _shot_arm_corpus(n_spells=200, seed=0):
    """One attacking-third spell per period with a PLANTED GK->(Z,Y) confounder.

    The defending-GK depth ``gk_x`` (== the xS GK_r signal, y pinned to the goal line)
    drives BOTH treatment (in-spell shot) and outcome (a success shot). Control
    conversions use the R9 door: the spell ends early (ball exits the domain at
    entry + 1 s) and the possessing team scores at entry + 4 s -- outside the
    possession-clamped Z window, inside the anchor-inclusive Y window. A naive
    in-spell scoring shot would yield Z = 1 and no control-Y variation.
    """
    rng = np.random.default_rng(seed)
    frame_rows, action_rows, aid = [], [], 0
    for k in range(n_spells):
        per = k + 1
        gk_sig = float(rng.normal())
        gk_x = float(np.clip(8.0 + 4.0 * gk_sig, 1.0, 30.0))  # GK depth = the planted signal
        bx = float(rng.uniform(8.0, 30.0))  # attacking third (goal_x = 0): abs(bx) <= 35
        by = float(rng.uniform(20.0, 48.0))
        z = bool(rng.uniform() < 1.0 / (1.0 + np.exp(-2.0 * gk_sig)))
        p_y = 1.0 / (1.0 + np.exp(-(2.0 * gk_sig + (0.5 if z else 0.0) - 0.3)))
        y = bool(rng.uniform() < p_y)
        for t, (cx, cy) in ((_ENTRY, (bx, by)), (_SPELL_END, (bx, by)), (_EXIT_T, (60.0, 34.0))):
            frame_rows.append(frow(10, 5, False, cx, cy, t, period=per))  # carrier on the ball
            frame_rows += [
                frow(11, 5, False, 18.0, 40.0, t, period=per),
                frow(12, 5, False, 15.0, 30.0, t, period=per),
                frow(21, 6, False, 8.0, 40.0, t, period=per),
                frow(22, 6, False, 10.0, 30.0, t, period=per),
                frow(1, 5, True, 101.0, 34.0, t, period=per),  # team-5 GK high x -> attacks 0
                frow(2, 6, True, gk_x, 34.0, t, period=per),  # PLANTED: defending-GK depth varies
            ]
            frame_rows.append(frow(pd.NA, pd.NA, False, cx, cy, t, is_ball=True, period=per))
        if z:  # treated: in-spell shot; its own result carries Y (anchor-inclusive window)
            action_rows.append([1, aid, per, 5, _TREATED_SHOT_T, SHOT, SUCCESS if y else FAIL, 14, 6, 0, 34])
            aid += 1
        elif y:  # control conversion via the R9 door: spell-ends-early + late goal
            action_rows.append([1, aid, per, 5, _LATE_GOAL_T, SHOT, SUCCESS, 14, 6, 0, 34])
            aid += 1
    f = pd.DataFrame(frame_rows)
    f["player_id"] = f["player_id"].astype("Int64")
    f["team_id"] = f["team_id"].astype("Int64")
    return f, actions(action_rows)


def test_builder_level_positive_control_planted_gk_confounding_clears_band():
    """P1(b): the full build_opportunities(config=shot_arm_config) -> propensity ->
    placebo_shift chain on the BUILDER's output must detect a planted GK->(Z,Y)
    confounder -- the positive control at the layer the guard defends."""
    cfg = shot_arm_config(META)
    # Fixture-validity (R9, load-bearing): the late goal is OUTSIDE the possession-clamped
    # Z window (entry, min(entry + T, spell_end)] but INSIDE the anchor-inclusive Y window.
    assert _LATE_GOAL_T > min(_ENTRY + EXPOSURE_WINDOW_SECONDS, _SPELL_END)
    assert _LATE_GOAL_T <= _ENTRY + cfg.outcome_window_seconds

    f, acts = _shot_arm_corpus()
    opp = build_opportunities(f, acts, home_team_id=5, model_metadata=META, config=cfg)
    assert len(opp) == 200  # one spell per period; the exit frame closed each spell
    assert opp["Z"].nunique() == 2
    # REGISTERED precondition: control Y must VARY or the ATT is confounder-invariant
    # and the instrument is dead (the earlier own-result-only registration failed here).
    assert opp.loc[opp["Z"] == 0, "Y"].var() > 0

    xb = opp[list(cfg.confounders)].to_numpy(dtype=float)
    xg = opp[list(cfg.gk_block)].to_numpy(dtype=float)
    assert np.isfinite(xg).all()  # the GK block resolved from the xS extractor
    y = opp["Y"].to_numpy(dtype=float)
    z = opp["Z"].to_numpy(dtype=int)
    out = placebo_shift(xb, xg, y, z, n_seeds=20, rng_seed=0)
    assert out["band_p95"] > 0  # fixture-validity: a zero band would make the clears-assert vacuous
    from silly_kicks.causal.matching import _att_with_block

    real = abs(_att_with_block(xb, xg, y, z, seed=0) - out["base_att"])
    assert real > out["band_p95"], (real, out["band_p95"])
