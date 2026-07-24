"""TF-19 xS placebo v2 (ADR-037 amendment): the model-relevant defender placebo + the
non-gating attacker diagnostic. Construct guards live here (the 'who is in the pool' errors
CI must catch); the frozen v1 discrimination test in test_probe_discriminating_power.py is
untouched. See docs/superpowers/specs/2026-07-23-tf19-xs-placebo-v2-design.md."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _model_eval as me
from tests.tracking._probe_fixtures import planted_model  # GK-responsive planted models (mixed/gk_blind)


def _pool_frame() -> pd.DataFrame:
    """One frame: ball+carrier A1 at (96,8), attacked goal x=105. 4 defenders at known ball
    distances (D1 nearest the carrier -> the excluded nearest_def), 3 more attackers, a GK, ball."""
    rows = [
        dict(team_id="A", player_id="A1", x=96.0, y=8.0, is_ball=False, is_goalkeeper=False),  # carrier
        dict(team_id="A", player_id="A2", x=99.0, y=34.0, is_ball=False, is_goalkeeper=False),
        dict(team_id="A", player_id="A3", x=92.0, y=26.0, is_ball=False, is_goalkeeper=False),
        dict(team_id="A", player_id="A4", x=90.0, y=40.0, is_ball=False, is_goalkeeper=False),
        dict(team_id="B", player_id="D1", x=97.0, y=9.0, is_ball=False, is_goalkeeper=False),  # nearest carrier
        dict(team_id="B", player_id="D2", x=99.0, y=12.0, is_ball=False, is_goalkeeper=False),
        dict(team_id="B", player_id="D3", x=100.0, y=20.0, is_ball=False, is_goalkeeper=False),
        dict(team_id="B", player_id="D4", x=94.0, y=4.0, is_ball=False, is_goalkeeper=False),
        dict(team_id="B", player_id="Bgk", x=104.0, y=34.0, is_ball=False, is_goalkeeper=True),
        dict(team_id="ball", player_id=None, x=96.0, y=8.0, is_ball=True, is_goalkeeper=False),
    ]
    df = pd.DataFrame(rows)
    df["game_id"] = "g"
    df["period_id"] = 1
    df["frame_id"] = 1
    return df


def test_defender_pool_is_ball_nearest_defenders_minus_nearest_def():
    grp = _pool_frame()
    pool = set(me._model_relevant_def_pool(grp, "B", "A1"))
    # 4 defenders exist (all within nearest-5); nearest_def by carrier is D1 -> excluded.
    assert pool == {"D2", "D3", "D4"}


def test_defender_pool_never_contains_carrier_or_any_attacker():
    grp = _pool_frame()
    pool = set(me._model_relevant_def_pool(grp, "B", "A1"))
    assert "A1" not in pool  # carrier
    assert pool.isdisjoint({"A1", "A2", "A3", "A4"})  # no attacker
    assert "Bgk" not in pool  # def_xy excludes GKs (the 'minus GK' no-op)


def test_attacker_diag_pool_excludes_carrier_and_is_attackers_only():
    grp = _pool_frame()
    diag = set(me._attacker_diag_pool(grp, "B", "A1"))
    assert "A1" not in diag  # carrier/shooter excluded
    assert diag == {"A2", "A3", "A4"}  # the non-carrier attackers (<=5 nearest to ball)
    assert diag.isdisjoint({"D1", "D2", "D3", "D4", "Bgk"})  # no defender, no GK


def _truncation_frame() -> pd.DataFrame:
    """6 defenders (D1..D6) + 6 non-carrier attackers (A1..A6) + carrier A0 + GK + ball, each at a
    STRICTLY-increasing ball-distance so rank 6 is unambiguous. The ONLY fixture with MORE than k=5
    candidates per side -> the sole test that exercises the np.argsort[:k] TRUNCATION (verified
    during planning: nearest_def=D1, def pool={D2..D5}, diag pool={A1..A5})."""
    rows = [
        dict(team_id="A", player_id="A0", x=96.0, y=8.0, is_ball=False, is_goalkeeper=False),  # carrier
        dict(team_id="A", player_id="A1", x=94.0, y=10.0, is_ball=False, is_goalkeeper=False),
        dict(team_id="A", player_id="A2", x=92.0, y=12.0, is_ball=False, is_goalkeeper=False),
        dict(team_id="A", player_id="A3", x=90.0, y=14.0, is_ball=False, is_goalkeeper=False),
        dict(team_id="A", player_id="A4", x=88.0, y=16.0, is_ball=False, is_goalkeeper=False),
        dict(team_id="A", player_id="A5", x=86.0, y=18.0, is_ball=False, is_goalkeeper=False),
        dict(team_id="A", player_id="A6", x=84.0, y=20.0, is_ball=False, is_goalkeeper=False),  # farthest attacker
        dict(team_id="B", player_id="D1", x=97.0, y=9.0, is_ball=False, is_goalkeeper=False),  # nearest carrier
        dict(team_id="B", player_id="D2", x=98.0, y=11.0, is_ball=False, is_goalkeeper=False),
        dict(team_id="B", player_id="D3", x=99.0, y=14.0, is_ball=False, is_goalkeeper=False),
        dict(team_id="B", player_id="D4", x=100.0, y=17.0, is_ball=False, is_goalkeeper=False),
        dict(team_id="B", player_id="D5", x=101.0, y=20.0, is_ball=False, is_goalkeeper=False),
        dict(team_id="B", player_id="D6", x=103.0, y=26.0, is_ball=False, is_goalkeeper=False),  # farthest defender
        dict(team_id="B", player_id="Bgk", x=104.0, y=34.0, is_ball=False, is_goalkeeper=True),
        dict(team_id="ball", player_id=None, x=96.0, y=8.0, is_ball=True, is_goalkeeper=False),
    ]
    df = pd.DataFrame(rows)
    df["game_id"] = "g"
    df["period_id"] = 1
    df["frame_id"] = 1
    return df


def test_pool_truncates_to_k_nearest_and_excludes_the_farthest():
    # The load-bearing anti-vacuity guard: with 6 candidates per side, dropping the [:k] slice
    # (returning ALL defenders/attackers) would re-admit the farthest -> these assertions fail.
    grp = _truncation_frame()
    dpool = set(me._model_relevant_def_pool(grp, "B", "A0"))
    assert dpool == {"D2", "D3", "D4", "D5"}  # 5 nearest ({D1..D5}) minus nearest_def D1
    assert "D6" not in dpool and len(dpool) == 4  # rank-6 truncated; nearest_def removed
    diag = set(me._attacker_diag_pool(grp, "B", "A0"))
    assert diag == {"A1", "A2", "A3", "A4", "A5"}  # 5 nearest non-carrier attackers
    assert "A6" not in diag and len(diag) == 5  # rank-6 truncated; carrier A0 excluded


def _targets_for(frames: pd.DataFrame) -> pd.DataFrame:
    """One ghost target per frame: a DIAGONAL GK->target paired vector (-8, -5). The diagonal is
    load-bearing: an axis-aligned vector (e.g. (-6, 0)) mirrors an x-symmetric defender across the
    ball to the SAME ball-distance -> a false zero delta (verified during planning). Every
    displacement (|v|=9.43) clears the 2 m dose band; ghost not clamped / in box -> trusted."""
    gk = frames[frames["is_goalkeeper"].astype(bool)]
    t = gk[["game_id", "period_id", "frame_id", "x", "y"]].drop_duplicates(subset=["game_id", "period_id", "frame_id"])
    return t.assign(target_x=t["x"] - 8.0, target_y=t["y"] - 5.0, ghost_clamped=False, ghost_out_of_box=False).drop(
        columns=["x", "y"]
    )


class _DistPlanted:
    """Deterministic model: z depends ONLY on the 5 nearest distances of ONE side. Moving a
    player on the OTHER side is an exact no-op -> a two-sided signature for pool provenance."""

    def __init__(self, side: str):  # "def" | "atk"
        assert side in ("def", "atk")
        self.side = side
        self.carrier_params: dict = {}

    def predict_proba(self, feats):
        pfx = "DefDist_" if self.side == "def" else "OffDist_"
        s = np.nansum([(20.0 - feats[f"{pfx}{k}"].to_numpy(float)) / 20.0 for k in range(5)], axis=0)
        return 1.0 / (1.0 + np.exp(-(0.1 + 0.1 * s)))


def _degeneracy_frames(n: int = 24) -> pd.DataFrame:
    """3 near-ball defenders (all in the nearest-5) + a carrier + 8 more attackers (so the v1 random
    pool is ~75% attackers -> EVERY per-replicate median is exactly 0 under the def-only model,
    verified during planning: v1 placebo_p95 == 0.0 at seed 42) + GK + ball, replicated to n
    distinct games. Ball+carrier at (96,8), goal x=105. The nearest 5 non-carrier attackers
    (A2..A6) stay the attacker_diag pool; A7..A9 are far dilution only. COLUMN-COMPLETE
    (vx/vy/ball_state/time_seconds) because it flows through substitution_deltas ->
    _eligible_groups -> infer_ball_carrier, which needs those columns."""
    specs = [
        ("A", "A1", 96.0, 8.0, False),  # carrier (attacker, nearest to ball)
        ("A", "A2", 70.0, 34.0, False),
        ("A", "A3", 60.0, 20.0, False),
        ("A", "A4", 55.0, 44.0, False),
        ("A", "A5", 50.0, 10.0, False),
        ("A", "A6", 45.0, 60.0, False),
        ("A", "A7", 40.0, 30.0, False),
        ("A", "A8", 35.0, 50.0, False),  # far dilution (out of both pools)
        ("A", "A9", 30.0, 18.0, False),
        ("B", "D1", 97.0, 9.0, False),  # nearest defender to carrier -> the excluded nearest_def
        ("B", "D2", 99.0, 13.0, False),
        ("B", "D3", 100.0, 20.0, False),
        ("B", "Bgk", 104.0, 34.0, True),
    ]
    reps = []
    for i in range(n):
        rows: list[dict] = [
            dict(
                game_id=f"m{i}",
                period_id=1,
                frame_id=1,
                time_seconds=40.0 + 10.0 * i,
                team_id=tm,
                player_id=pid,
                x=x,
                y=y,
                vx=0.0,
                vy=0.0,
                is_ball=False,
                is_goalkeeper=gk,
                ball_state="alive",
            )
            for tm, pid, x, y, gk in specs
        ]
        rows.append(
            dict(
                game_id=f"m{i}",
                period_id=1,
                frame_id=1,
                time_seconds=40.0 + 10.0 * i,
                team_id="ball",
                player_id=None,
                x=96.0,
                y=8.0,
                vx=0.0,
                vy=0.0,
                is_ball=True,
                is_goalkeeper=False,
                ball_state="alive",
            )
        )
        reps.append(pd.DataFrame(rows))
    return pd.concat(reps, ignore_index=True)


def _placebo_p95(deltas: pd.DataFrame, role: str = "placebo_out") -> float:
    sub = deltas[deltas["actor_role"] == role]
    rep_med = sub.groupby("replicate")["delta_p"].median()
    return float(np.percentile(rep_med, 95.0)) if len(rep_med) else float("nan")


def _disc_base() -> pd.DataFrame:
    """v2-OWNED discrimination base (decoupled from the frozen probe_frames): carrier A1 +
    attackers A2/A3 + THREE near-ball defenders B1/B2/B3 (so the v2 defender pool is >=2 after
    nearest_def removal) + GK + ball, attacking third, goal x=105. Verified under the existing v1
    path: mixed -> pass, gk_blind -> fail (band_n=150). Decoupling means a future probe_frames
    edit can't perturb this test (and vice-versa)."""
    specs = [
        ("A", "A1", 96.0, 8.0, False),
        ("A", "A2", 99.0, 34.0, False),
        ("A", "A3", 92.0, 26.0, False),
        ("B", "B1", 100.0, 20.0, False),
        ("B", "B2", 98.0, 12.0, False),
        ("B", "B3", 97.0, 14.0, False),
        ("B", "Bgk", 104.0, 34.0, True),
        ("ball", None, 96.0, 8.0, False),
    ]
    rows = [
        dict(
            team_id=tm,
            player_id=pid,
            x=x,
            y=y,
            vx=0.0,
            vy=0.0,
            is_ball=(tm == "ball"),
            is_goalkeeper=gk,
            ball_state="alive",
        )
        for tm, pid, x, y, gk in specs
    ]
    df = pd.DataFrame(rows)
    df["game_id"] = "g"
    df["period_id"] = 1
    df["frame_id"] = 1
    df["time_seconds"] = 40.0
    return df


def _disc_frames(n: int = 150) -> pd.DataFrame:
    """Replicate _disc_base into 12 games x ~12 frames (game_id = i % 12) so the dose-response is
    POWERED (>= MIN_GAMES games, >= MIN_GAME_N frames each) and the band clears MIN_BAND_N."""
    base = _disc_base()
    reps = []
    for i in range(n):
        r = base.copy()
        r["game_id"] = f"m{i % 12}"
        r["frame_id"] = r["frame_id"] + 10 * i
        r["time_seconds"] = r["time_seconds"] + 10.0 * i
        reps.append(r)
    return pd.concat(reps, ignore_index=True)


def _disc_targets_varied(frames: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """VARIED ghost targets (per-frame random spread + y-noise) -> non-constant displacement_m so
    the clustered dose-response has variance to correlate (needed for a 'pass' verdict). Distinct
    from _targets_for (a single fixed vector -> constant displacement -> dose underpowered)."""
    gk = frames[frames["is_goalkeeper"].astype(bool)]
    t = gk[["game_id", "period_id", "frame_id", "x", "y"]].drop_duplicates(subset=["game_id", "period_id", "frame_id"])
    rng = np.random.default_rng(seed)
    return t.assign(
        target_x=t["x"] - 6.0 * (0.5 + rng.random(len(t))),
        target_y=t["y"] + rng.normal(scale=2.0, size=len(t)),
        ghost_clamped=False,
        ghost_out_of_box=False,
    ).drop(columns=["x", "y"])


def test_v2_emits_placebo_out_and_a_distinct_attacker_diag_role():
    frames = _degeneracy_frames()
    targets = _targets_for(frames)
    deltas = me.substitution_deltas(
        _DistPlanted("def"),
        frames,
        arm="xs",
        mode="targets",
        targets=targets,
        seed=42,
        placebo="model_relevant_def",
    )
    roles = set(deltas["actor_role"].unique())
    assert "placebo_out" in roles and "attacker_diag" in roles
    assert {"gk", "nearest_def"} <= roles


def test_v1_default_placebo_emits_no_attacker_diag_and_is_unchanged():
    frames = _degeneracy_frames()
    targets = _targets_for(frames)
    d_default = me.substitution_deltas(_DistPlanted("def"), frames, arm="xs", mode="targets", targets=targets, seed=42)
    d_explicit = me.substitution_deltas(
        _DistPlanted("def"), frames, arm="xs", mode="targets", targets=targets, seed=42, placebo="random"
    )
    assert "attacker_diag" not in set(d_default["actor_role"].unique())
    # placebo="random" is the byte-identical default (frozen v1).
    pd.testing.assert_frame_equal(d_default, d_explicit)


def test_v1_random_path_is_numerically_pinned():
    # ENFORCED byte-identity (the assert_frame_equal above only proves default == explicit-random,
    # both through the NEW code -- a shared perturbation passes it). This pins the v1 random path's
    # numeric outputs to values captured on the PRE-refactor code, so any perturbation of the
    # random draw / base_p / vector fails here. Fixed fixture + fixed targets + seed 42.
    frames = _disc_frames()
    targets = _targets_for(frames)  # fixed diagonal vector -> deterministic
    deltas = me.substitution_deltas(
        planted_model("mixed"), frames, arm="xs", mode="targets", targets=targets, seed=42, placebo="random"
    )
    r = me.evaluate_xs_probe(deltas)
    # Full-precision goldens (rel=1e-6 >> 1 ULP, cross-platform safe): a truncated golden is too
    # coarse for pytest.approx at these small magnitudes.
    assert r["gated_band_median"] == pytest.approx(0.053921843286600435, rel=1e-6)
    assert r["nearest_def_median"] == pytest.approx(0.0003011338421371468, rel=1e-6)
    assert r["placebo_p95"] == pytest.approx(0.0008451766247897785, rel=1e-6)


def _median_delta(deltas: pd.DataFrame, role: str) -> float:
    sub = deltas[deltas["actor_role"] == role]
    return float(sub["delta_p"].median()) if len(sub) else float("nan")


def test_placebo_out_is_defender_sourced_attacker_diag_is_attacker_sourced():
    # def-only model: moving a DEFENDER registers, moving an ATTACKER is an exact no-op.
    frames = _degeneracy_frames()
    targets = _targets_for(frames)
    d = me.substitution_deltas(
        _DistPlanted("def"),
        frames,
        arm="xs",
        mode="targets",
        targets=targets,
        seed=42,
        placebo="model_relevant_def",
    )
    assert _median_delta(d, "placebo_out") > 0  # defenders move the def-only surface
    assert _median_delta(d, "attacker_diag") == 0  # attackers do not -> attacker-sourced


def test_signature_flips_under_an_attacker_only_model():
    # atk-only model: the two-sided converse -> catches the OPPOSITE mis-tagging.
    frames = _degeneracy_frames()
    targets = _targets_for(frames)
    d = me.substitution_deltas(
        _DistPlanted("atk"),
        frames,
        arm="xs",
        mode="targets",
        targets=targets,
        seed=42,
        placebo="model_relevant_def",
    )
    assert _median_delta(d, "placebo_out") == 0  # defenders do not move the atk-only surface
    assert _median_delta(d, "attacker_diag") > 0  # attackers do -> placebo_out is NOT attacker-sourced


def test_v2_defender_placebo_is_live_where_v1_random_placebo_degenerates():
    # Same fixture + def-only model. v1's random pool is 75% attackers -> under a def-only model
    # every attacker draw is an EXACT no-op, so >50% of each replicate's per-frame draws are 0 ->
    # every per-replicate median is exactly 0 -> placebo_p95 == 0.0 (deterministic, verified at
    # seed 42). v2's defender pool {D2,D3} has no no-op members -> every median > 0.
    frames = _degeneracy_frames()
    targets = _targets_for(frames)
    d_v1 = me.substitution_deltas(
        _DistPlanted("def"), frames, arm="xs", mode="targets", targets=targets, seed=42, placebo="random"
    )
    d_v2 = me.substitution_deltas(
        _DistPlanted("def"),
        frames,
        arm="xs",
        mode="targets",
        targets=targets,
        seed=42,
        placebo="model_relevant_def",
    )
    assert _placebo_p95(d_v2) > 0  # the fix works: v2 clears the no_valid_placebo gate
    assert _placebo_p95(d_v1) == 0.0  # ...and the SAME fixture degenerates v1 (principled, not tuned)


def test_xs_v2_registered_alongside_frozen_v1():
    assert "xs_v2" in me.PROBE_WRAPPERS
    rc_v1 = me.PROBE_WRAPPERS["xs"]["rule_constants"]
    rc_v2 = me.PROBE_WRAPPERS["xs_v2"]["rule_constants"]
    assert rc_v2["placebo_pool"] == "model_relevant_def"
    # v2's numeric constants are IDENTICAL to v1 (the only difference is the pool).
    for k, v in rc_v1.items():
        assert rc_v2[k] == v, k


def test_xs_v2_wrapper_relabels_rule_and_keeps_v1_frozen():
    frames = _degeneracy_frames()
    targets = _targets_for(frames)
    out_v2 = me.xs_substitution_probe_v2(_DistPlanted("def"), frames, targets, seed=42)
    assert out_v2["rule"] == "xs-dose-banded-v2"
    assert out_v2["placebo_pool"] == "model_relevant_def"
    out_v1 = me.xs_substitution_probe(_DistPlanted("def"), frames, targets, seed=42)
    assert out_v1["rule"] == "xs-dose-banded-v1"  # frozen evaluator label unchanged


def _run_v2(kind: str, seed: int = 7):
    """v2 discrimination through the v2 wrapper, on the v2-OWNED _disc_frames (NOT the frozen
    probe_frames -> no shared-fixture coupling). The dose-response is pool-independent (gk
    stratum), so v2 shares v1's dose_state (verified v1-valid on this fixture: mixed->pass,
    gk_blind->fail); only the placebo band (>=2 live defenders) differs."""
    frames = _disc_frames()
    targets = _disc_targets_varied(frames, seed=seed)
    out = me.xs_substitution_probe_v2(planted_model(kind), frames, targets, seed=seed)
    assert out["gated_band_n"] >= me.XS_PROBE_MIN_BAND_N, "fixture too small for the registered rule"
    return out


def test_v2_passes_mixed_dependence_and_fails_gk_blind():
    assert _run_v2("mixed")["verdict"] == "pass"
    out_blind = _run_v2("gk_blind")
    assert out_blind["verdict"] == "fail"
    assert out_blind["placebo_p95"] > 0  # defender placebo is live -> a CLEAN fail, not degenerate


def test_evaluate_xs_probe_ignores_attacker_diag_rows():
    # Fail-closed guard (GUARD RULE): even if a future edit tried to band attacker_diag, the
    # evaluator must ignore it. Take a band-clearing v1 deltas frame, inject attacker_diag rows
    # with a HUGE delta_p, and assert the placebo band + verdict + prongs are byte-identical.
    frames = _disc_frames()
    targets = _targets_for(frames)  # fixed vector -> deterministic band-clearing frame
    base = me.substitution_deltas(
        planted_model("mixed"), frames, arm="xs", mode="targets", targets=targets, seed=42, placebo="random"
    )
    ref = me.evaluate_xs_probe(base)
    injected_rows = base[base["actor_role"] == "placebo_out"].head(200).assign(actor_role="attacker_diag", delta_p=10.0)
    injected = pd.concat([base, injected_rows], ignore_index=True)
    got = me.evaluate_xs_probe(injected)
    assert got["placebo_p95"] == ref["placebo_p95"]  # attacker_diag never enters the gated band
    assert got["verdict"] == ref["verdict"]
    assert got["gated_band_median"] == ref["gated_band_median"]
