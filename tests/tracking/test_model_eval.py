"""PR-1: generic substitution core + registry (ADR-037). xS evaluator tests are added in Task 3."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _model_eval as me
from tests.tracking._probe_fixtures import planted_model, probe_frames


def test_probe_wrappers_registry_lists_both_arms():
    assert set(me.PROBE_WRAPPERS) == {"xcross", "xs", "xs_v2"}
    for name, entry in me.PROBE_WRAPPERS.items():
        assert callable(entry["wrapper"]), name
        assert isinstance(entry["rule_constants"], dict) and entry["rule_constants"], name


def test_registry_meta_every_wrapper_has_a_pinned_rule_test():
    # Meta-assertion (spec §7): each registry key must appear in a PINNED_RULES map here.
    PINNED_RULES = {
        "xcross": {"ratio": 2.0, "abs_floor": 0.01},
        "xs": {"ratio": 2.0, "dose_m": 2.0, "placebo_band_pct": 95.0},
        "xs_v2": {"ratio": 2.0, "dose_m": 2.0, "placebo_band_pct": 95.0, "placebo_pool": "model_relevant_def"},
    }
    assert set(PINNED_RULES) == set(me.PROBE_WRAPPERS)
    for name, pins in PINNED_RULES.items():
        rc = me.PROBE_WRAPPERS[name]["rule_constants"]
        for k, v in pins.items():
            assert rc[k] == v, (name, k)


def test_substitution_deltas_panel_mode_produces_tidy_rows():
    frames = probe_frames()
    out = me.substitution_deltas(planted_model("mixed"), frames, arm="xcross", mode="panel", seed=42)
    assert len(out) > 0  # the fixture frames ARE eligible; empty would be a vacuous pass
    assert set(out.columns) >= {"game_id", "period_id", "frame_id", "actor_role", "displacement_m", "delta_p"}
    assert set(out["actor_role"].unique()) == {"gk", "nearest_def", "placebo_out"}
    assert (out["delta_p"] >= 0).all()


def test_substitution_deltas_target_mode_moves_gk_to_supplied_position():
    frames = probe_frames()
    gk = frames[frames["is_goalkeeper"].astype(bool)]
    targets = (
        gk[["game_id", "period_id", "frame_id"]]
        .drop_duplicates()
        .assign(target_x=90.0, target_y=34.0, ghost_clamped=False, ghost_out_of_box=False)
    )
    out = me.substitution_deltas(
        planted_model("mixed"),
        frames,
        arm="xs",
        mode="targets",
        targets=targets,
        n_placebo_replicates=3,
        seed=42,
    )
    gk_rows = out[out["actor_role"] == "gk"]
    assert len(gk_rows) == len(targets)
    assert (gk_rows["displacement_m"] > 0).all()  # the fixture GK is not at (90,34)
    # paired-vector controls: every control row's displacement equals its frame's GK displacement
    per_frame = out.pivot_table(index="frame_id", columns="actor_role", values="displacement_m", aggfunc="first")
    assert np.allclose(per_frame["nearest_def"], per_frame["gk"])
    reps = out[out["actor_role"] == "placebo_out"]["replicate"].nunique()
    assert reps == 3
    # Production-path False direction of the off-pitch flag: the (90, 34) target keeps
    # every moved actor inside the pitch rectangle, so _moved_off_pitch must report False
    # for ALL rows (kills an always-True mutant; the True direction is pinned in the
    # off-pitch policy test below).
    assert not out["moved_off_pitch"].astype(bool).any()


def test_substitution_deltas_carries_moved_off_pitch_and_scores_off_pitch_controls():
    """Schema + registered off-pitch policy (ADR-037 item 15): a control displaced off-pitch
    by the paired vector is FLAGGED moved_off_pitch=True yet still SCORED (delta_p finite) --
    never clamped/dropped. A large +y paired vector pushes every fixture outfielder past the
    68 m touchline, so every control row is off-pitch."""
    frames = probe_frames()
    gk = frames[frames["is_goalkeeper"].astype(bool)]
    targets = (
        gk[["game_id", "period_id", "frame_id"]]
        .drop_duplicates()
        .assign(target_x=104.0, target_y=99.0, ghost_clamped=False, ghost_out_of_box=False)
    )
    out = me.substitution_deltas(
        planted_model("mixed"),
        frames,
        arm="xs",
        mode="targets",
        targets=targets,
        n_placebo_replicates=3,
        seed=42,
    )
    assert "moved_off_pitch" in out.columns
    placebo = out[out["actor_role"] == "placebo_out"]
    assert len(placebo) > 0
    assert placebo["moved_off_pitch"].astype(bool).all()  # every outfielder crosses the touchline
    assert placebo["delta_p"].notna().all()  # SCORED, never clamped/dropped -> paired-vector intact


def test_targets_zero_overlap_fails_loud_with_key_examples():
    # I1: disjoint key sets must raise (a silent empty result would read as "no signal").
    frames = probe_frames()
    targets = pd.DataFrame(
        [
            {
                "game_id": "OTHER",
                "period_id": 1,
                "frame_id": 999,
                "target_x": 90.0,
                "target_y": 34.0,
                "ghost_clamped": False,
                "ghost_out_of_box": False,
            }
        ]
    )
    with pytest.raises(ValueError, match=r"matched ZERO eligible frames.*OTHER"):
        me.substitution_deltas(planted_model("mixed"), frames, arm="xs", mode="targets", targets=targets)


def test_targets_missing_column_raises():
    frames = probe_frames()
    targets = pd.DataFrame([{"game_id": "g", "period_id": 1, "frame_id": 1, "target_x": 90.0}])
    with pytest.raises(ValueError, match=r"missing required column.*target_y"):
        me.substitution_deltas(planted_model("mixed"), frames, arm="xs", mode="targets", targets=targets)


def test_targets_nan_ghost_clamped_raises():
    # I2: bool(NaN) is True -- a NaN flag would silently shrink the trusted stratum.
    frames = probe_frames()
    targets = pd.DataFrame(
        [
            {
                "game_id": "g",
                "period_id": 1,
                "frame_id": 1,
                "target_x": 90.0,
                "target_y": 34.0,
                "ghost_clamped": np.nan,
                "ghost_out_of_box": False,
            }
        ]
    )
    with pytest.raises(ValueError, match=r"ghost_clamped.*must be non-null"):
        me.substitution_deltas(planted_model("mixed"), frames, arm="xs", mode="targets", targets=targets)


def _deltas(n=300, dose=3.0, gk_scale=0.02, seed=0, zero_frac=0.0, trusted=True):
    """Synthetic tidy deltas: n frames, gk deltas ~ gk_scale, controls ~ gk_scale/4."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        d = dose * (0.5 + rng.random())
        gk_dp = 0.0 if rng.random() < zero_frac else gk_scale * d * (0.5 + rng.random())
        rows.append(
            dict(
                game_id=f"m{i % 10}",
                period_id=1,
                frame_id=i,
                actor_role="gk",
                replicate=0,
                displacement_m=d,
                delta_p=gk_dp,
                ghost_clamped=not trusted,
                ghost_out_of_box=False,
                moved_off_pitch=False,
            )
        )
        rows.append(
            dict(
                game_id=f"m{i % 10}",
                period_id=1,
                frame_id=i,
                actor_role="nearest_def",
                replicate=0,
                displacement_m=d,
                delta_p=gk_scale * d / 4,
                ghost_clamped=False,
                ghost_out_of_box=False,
                moved_off_pitch=False,
            )
        )
        for r in range(me.XS_PROBE_PLACEBO_REPLICATES):
            rows.append(
                dict(
                    game_id=f"m{i % 10}",
                    period_id=1,
                    frame_id=i,
                    actor_role="placebo_out",
                    replicate=r,
                    displacement_m=d,
                    delta_p=gk_scale * d / 8 * rng.random(),
                    ghost_clamped=False,
                    ghost_out_of_box=False,
                    moved_off_pitch=False,
                )
            )
    return pd.DataFrame(rows)


def test_xs_evaluator_passes_on_strong_dose_responsive_signal():
    out = me.evaluate_xs_probe(_deltas(gk_scale=0.02))
    assert out["verdict"] == "pass"
    assert out["gated_band_n"] >= me.XS_PROBE_MIN_BAND_N
    assert out["dose_response_rho"] > 0


def test_xs_evaluator_unmeasurable_when_band_too_small():
    """n=60: the STRATUM floor (50) passes, so the BAND floor (100) is the SOLE trigger
    (P7 - n=20 tripped the stratum floor first and the band floor could regress to 0
    all-green)."""
    out = me.evaluate_xs_probe(_deltas(n=60, dose=1.0))  # dose<2 keeps banded n under 100
    assert out["trusted_stratum"] >= me.XS_PROBE_MIN_STRATUM_N
    assert out["gated_band_n"] < me.XS_PROBE_MIN_BAND_N
    assert out["verdict"] == "unmeasurable_at_dose"


def test_xs_evaluator_unmeasurable_when_trusted_stratum_empty():
    out = me.evaluate_xs_probe(_deltas(trusted=False))
    assert out["verdict"] == "unmeasurable_at_dose"


def test_xs_evaluator_no_valid_placebo_is_fail_closed():
    d = _deltas()
    d.loc[d["actor_role"] == "placebo_out", "delta_p"] = 0.0
    out = me.evaluate_xs_probe(d)
    assert out["verdict"] == "no_valid_placebo"


def test_placebo_zero_concentration_prong_has_its_own_trigger():
    """P7: the MAX_PLACEBO_ZERO_FRACTION prong only decides when zeros CONCENTRATE in
    some replicates while others stay live - the all-zero test above hits p95<=0 first
    and this ceiling could regress unnoticed. 19 dead replicates + replicate-0 zeroed
    in 40% of frames: fraction 0.97 STRICTLY above the 0.95 ceiling (execution found
    the original 19-dead-only fixture lands EXACTLY on 19/20 == 0.95 against the
    strict > rule - a boundary fixture that could never fire), while replicate-0's
    median stays positive so p95 > 0 (the prong's distinguishing condition)."""
    d = _deltas()
    dead = d["actor_role"].eq("placebo_out") & d["replicate"].ne(0)
    d.loc[dead, "delta_p"] = 0.0
    rep0 = d.index[d["actor_role"].eq("placebo_out") & d["replicate"].eq(0)]
    d.loc[rep0[: int(0.4 * len(rep0))], "delta_p"] = 0.0  # 40% of the live replicate
    out = me.evaluate_xs_probe(d)
    assert out["verdict"] == "no_valid_placebo"
    assert out["placebo_p95"] > 0
    assert out["placebo_zero_fraction"] > me.XS_PROBE_MAX_PLACEBO_ZERO_FRACTION


def test_xs_evaluator_flat_dose_response_overrides_band_pass():
    d = _deltas()
    # constant gk delta regardless of dose -> band median can pass, dose-response is flat
    d.loc[d["actor_role"] == "gk", "delta_p"] = 0.05
    out = me.evaluate_xs_probe(d)
    assert out["verdict"] == "band_pass_flat_dose_response"


def test_xs_evaluator_all_zero_gk_band_with_live_controls_is_a_clean_fail():
    """Review B1: zeros + LIVE controls = the keeper does not matter - a publishable
    FAIL, never 'unmeasurable'. The fraction stays reported (it makes the fail
    interpretable)."""
    out = me.evaluate_xs_probe(_deltas(zero_frac=0.9))
    assert out["verdict"] == "fail"
    assert out["gated_band_zero_fraction"] > 0.8
    assert out["placebo_p95"] > 0


def test_dose_response_null_is_centred_on_zero_under_no_signal():
    """Review B4 positive control for the NULL itself: with delta_p carrying no dose
    signal the p-value must not be small - else it is not a p-value."""
    d = _deltas()
    rng = np.random.default_rng(0)
    d.loc[d["actor_role"] == "gk", "delta_p"] = rng.random(int((d["actor_role"] == "gk").sum()))
    out = me.evaluate_xs_probe(d)
    assert out["dose_response_p"] > 0.05
    assert out["dose_response_n_games"] >= me.XS_PROBE_MIN_GAMES  # the test RAN - this is 'flat', not underpowered


def test_underpowered_dose_test_routes_band_pass_to_unmeasurable_never_flat():
    """Review N1(3): low power must not manufacture band_pass_flat_dose_response - but a
    band pass with an unrunnable dose test must not stand alone either. 3 games clears
    the band-n floor while staying under XS_PROBE_MIN_GAMES."""
    rows = _deltas(n=300)
    rows["game_id"] = "m" + (rows["frame_id"] % 3).astype(str)  # 3 games, ~100 gk frames each
    out = me.evaluate_xs_probe(rows)
    assert out["dose_state"] == "underpowered"
    assert out["verdict"] == "unmeasurable_at_dose"
    assert out["verdict"] != "band_pass_flat_dose_response"


def test_xs_evaluator_reports_ladder_and_ood_strata():
    out = me.evaluate_xs_probe(_deltas())
    assert set(out["dose_ladder"]) == set(me.XS_PROBE_DOSE_LADDER)
    assert "ood_stratum" in out and "trusted_stratum" in out


def test_evaluate_xs_probe_reports_off_pitch_control_fraction():
    """ADR-037 item 15: a report-only control off-pitch fraction in [0, 1]. _deltas()'s
    controls are all on-pitch (0.0); flagging every control lifts it to 1.0 -- proving the
    key is a live mean over CONTROL rows (nearest_def + placebo_out)."""
    out = me.evaluate_xs_probe(_deltas())
    assert "off_pitch_control_fraction" in out
    assert 0.0 <= out["off_pitch_control_fraction"] <= 1.0
    assert out["off_pitch_control_fraction"] == 0.0  # _deltas() controls are on-pitch
    d = _deltas()
    d.loc[d["actor_role"].isin(("nearest_def", "placebo_out")), "moved_off_pitch"] = True
    out2 = me.evaluate_xs_probe(d)
    assert out2["off_pitch_control_fraction"] == 1.0


def test_evaluate_xs_probe_off_pitch_fraction_nan_and_verdict_stable_without_column():
    """Backcompat/purity: a pre-fix deltas frame (no moved_off_pitch column) yields NaN for
    the report-only key and the IDENTICAL verdict -- the evaluator stays pure over older frames."""
    d = _deltas()
    out_with = me.evaluate_xs_probe(d)
    out_without = me.evaluate_xs_probe(d.drop(columns=["moved_off_pitch"]))
    assert np.isnan(out_without["off_pitch_control_fraction"])
    assert out_without["verdict"] == out_with["verdict"]


def test_moved_off_pitch_is_verdict_inert():
    """Report-only proof: flipping every moved_off_pitch value moves the reported fraction
    (non-vacuous) but changes NO verdict-determining field."""
    d = _deltas()
    base = me.evaluate_xs_probe(d)
    flipped = d.copy()
    flipped["moved_off_pitch"] = ~flipped["moved_off_pitch"].astype(bool)
    alt = me.evaluate_xs_probe(flipped)
    assert alt["off_pitch_control_fraction"] != base["off_pitch_control_fraction"]  # flip took effect
    assert alt["verdict"] == base["verdict"]
    for k in (
        "gated_band_median",
        "nearest_def_median",
        "placebo_p95",
        "dose_state",
        "dose_response_rho",
        "dose_response_p",
        "gated_band_n",
        "trusted_stratum",
    ):
        assert alt[k] == base[k], k


def test_nan_delta_p_raises_loud():
    """Exec-review fix 2: NaN delta_p must raise, never fail OPEN into the pre-registered
    expected 'fail' (a NaN-poisoned GK band medians to NaN, every >= comparison goes
    False; cf. the 4.18.0 all-NaN canonical-id class)."""
    d = _deltas()
    gk_idx = d.index[d["actor_role"] == "gk"]
    d.loc[gk_idx[0], "delta_p"] = np.nan
    with pytest.raises(ValueError, match=r"NaN delta_p"):
        me.evaluate_xs_probe(d)


def test_panel_shaped_gk_band_raises():
    """Exec-review fix 3: the evaluator consumes TARGETS-mode deltas (one GK row per
    frame key); panel-mode-shaped input (multiple displacement rows per frame) must
    raise, not fan out the banded control merge."""
    d = _deltas()
    gk = d[d["actor_role"] == "gk"]
    with pytest.raises(ValueError, match=r"duplicate.*panel"):
        me.evaluate_xs_probe(pd.concat([d, gk], ignore_index=True))


def test_banding_and_merge_keys_cannot_silently_regress():
    """Exec-review fix 4, adversarial keying/banding canary: frame ids COLLIDE across
    games and out-of-band decoy frames (displacement < dose, majority of the trusted
    stratum) carry huge gk AND control deltas (10.0). Correct behavior needs BOTH (a)
    the banded semi-join on the FULL (game_id, period_id, frame_id) triple - a
    frame_id-only merge drags decoy controls into the banded pool, nd_med explodes and
    the verdict flips to 'fail' - and (b) band-only medians - gk_med over the trusted
    stratum instead is the 10.0 decoy median, tripping the gated_band_median < 1.0
    assertion. The decoys sit at LOW displacement with HUGE deltas, so the trusted
    dose-response is anti-monotone by construction: the correct verdict is the
    band-pass-with-flat-dose one."""
    rng = np.random.default_rng(7)
    rows = []
    for g in range(10):
        for f in range(30):
            decoy = (f + g) % 5 < 3  # every frame_id is a decoy in SOME games, in-band in others
            disp = 1.0 if decoy else 2.0 + (f % 10) * 0.25
            base = dict(
                game_id=f"g{g}",
                period_id=1,
                frame_id=f,
                replicate=0,
                displacement_m=disp,
                ghost_clamped=False,
                ghost_out_of_box=False,
            )
            gk_dp = 10.0 if decoy else 0.02 * disp * (0.5 + rng.random())
            nd_dp = 10.0 if decoy else 0.005 * (0.5 + rng.random())
            rows.append({**base, "actor_role": "gk", "delta_p": gk_dp})
            rows.append({**base, "actor_role": "nearest_def", "delta_p": nd_dp})
            for r in range(me.XS_PROBE_PLACEBO_REPLICATES):
                pl_dp = 10.0 if decoy else 0.002 * (0.5 + rng.random())
                rows.append({**base, "actor_role": "placebo_out", "replicate": r, "delta_p": pl_dp})
    out = me.evaluate_xs_probe(pd.DataFrame(rows))
    assert out["verdict"] == "band_pass_flat_dose_response"
    assert out["gated_band_median"] < 1.0  # the 10.0 decoys stayed OUT of the band


def test_underpowered_dose_with_band_fail_is_a_clean_fail():
    """Exec-review fix 6, the fourth routing cell: underpowered dose test + band FAILURE
    must yield the publishable 'fail' - only a band PASS needs dose-response support."""
    rows = _deltas(n=300)
    rows["game_id"] = "m" + (rows["frame_id"] % 3).astype(str)  # 3 games -> underpowered
    rows.loc[rows["actor_role"] == "gk", "delta_p"] = 0.004  # ~ controls: the band ratio fails
    out = me.evaluate_xs_probe(rows)
    assert out["dose_state"] == "underpowered"
    assert out["verdict"] == "fail"


@pytest.mark.parametrize(
    "arm,probe,entangle,expected",
    [
        ("shot", "pass", "clears", "joins"),
        ("shot", "pass", "inside_band", "joins_with_caveat"),
        ("shot", "band_pass_flat_dose_response", "clears", "gated_flat_dose_response"),
        ("shot", "unmeasurable_at_dose", "clears", "unmeasurable_at_dose"),
        ("shot", "no_valid_placebo", "clears", "unmeasurable_at_dose"),
        ("shot", "pass", "degenerate", "joins_with_caveat"),
        ("shot", "fail", "degenerate", "gated_clean_fail"),
        ("shot", "instrument_invalid", "clears", "verdict_void"),
        ("shot", "fail", "clears", "gated_clean_fail"),
        ("cross", "pass", "clears", "joins"),
        ("cross", "pass", "inside_band", "joins_with_caveat"),
        ("cross", "fail", "inside_band", "gated_clean_fail"),
    ],
)
def test_regate_verdict_table(arm, probe, entangle, expected):
    assert me.regate_verdict(arm=arm, probe_verdict=probe, entanglement=entangle) == expected


def test_regate_verdict_rejects_unknown_inputs():
    with pytest.raises(ValueError):
        me.regate_verdict(arm="shot", probe_verdict="maybe", entanglement="clears")
