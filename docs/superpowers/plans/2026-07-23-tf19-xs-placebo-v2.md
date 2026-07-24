# TF-19 xS-probe placebo v2 (relevance-matched) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pre-registered `xs-dose-banded-v2` variant to the TF-19 xS-arm GK-substitution probe whose ONLY difference from the frozen v1 is a relevance-matched (model-relevant defenders) placebo pool, so the clustered dose-response runs for the first time and the arm reaches a real, citeable verdict.

**Architecture:** One private keyword (`placebo=`) threaded through `substitution_deltas`/`_targets_deltas` selects between v1's random-outfielder pool (`"random"`, the untouched default) and v2's ball-nearest defenders (`"model_relevant_def"`); a distinct-role `attacker_diag` population is emitted for reporting but never banded; `evaluate_xs_probe` is reused verbatim; a new `xs_substitution_probe_v2` wrapper + `PROBE_WRAPPERS["xs_v2"]` register the variant; the driver runs both variants side by side and records a lock-commit hash. No production/VAEP surface changes — this is a research instrument in `tracking/_model_eval.py`.

**Tech Stack:** Python, numpy, pandas, scipy; pytest; the existing GKDV probe stack (`silly_kicks/tracking/_model_eval.py`, `scripts/validate_xs_probe.py`).

**Spec:** `docs/superpowers/specs/2026-07-23-tf19-xs-placebo-v2-design.md` (read it for rationale; this plan is the HOW).

**Ground rules (project):** No commit without explicit owner approval; ONE commit per PR; full non-e2e suite green on `.venv312` before proposing. Never claim the release/PR-S/ADR number early. The ~64-match run and its `docs/research/tf19_pr3b_xs_v2/` output are an owner step AFTER the lock commit (blindness discipline) — NOT a TDD task here.

---

## File Structure

- **Modify** `silly_kicks/tracking/_model_eval.py` — add `_model_relevant_def_pool`, `_attacker_diag_pool`; add `placebo=` kwarg to `_targets_deltas` + `substitution_deltas`; emit `attacker_diag` rows in `_targets_deltas`; add `xs_substitution_probe_v2`; register `PROBE_WRAPPERS["xs_v2"]`. (No change to `evaluate_xs_probe`, `_panel_deltas`, `_nearest_def_mask`, `regate_verdict`, or the frozen `xs`/`xcross` registrations.)
- **Create** `tests/tracking/test_xs_placebo_v2.py` — pool-membership unit guards, the two-sided defender/attacker signature guard, the v2-live-vs-v1-degenerate contrast, and v2 discrimination (mixed passes / gk_blind fails).
- **Modify** `tests/tracking/test_model_eval.py` — extend the two registry manifest tests (`:11`, `:18`) to include `xs_v2`.
- **Modify** `scripts/validate_xs_probe.py` — `--variant {v1,v2,both}` (default `both`); per-variant pooled deltas; both verdicts + attacker diagnostic in one `metrics.json`; `--lock-commit`; two-variant `_render`.
- **Modify** `tests/scripts/test_validate_xs_probe.py` — update `_fake_metrics`/`_render`/`run` tests to the new two-variant shape.
- **Modify** `docs/PRIVATE_CONSUMERS.md:25` — add `xs_substitution_probe_v2` to the pinned `_model_eval.py` symbol list (rides with the code PR; no standalone doc commit).

---

### Task 1: v2 placebo pool helpers

**Files:**
- Modify: `silly_kicks/tracking/_model_eval.py` (add two private helpers after `_nearest_def_mask`, ~line 207)
- Test: `tests/tracking/test_xs_placebo_v2.py` (create)

- [ ] **Step 1: Write the failing pool-membership tests**

Create `tests/tracking/test_xs_placebo_v2.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_xs_placebo_v2.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_model_relevant_def_pool'`.

- [ ] **Step 3: Implement the two helpers**

In `silly_kicks/tracking/_model_eval.py`, immediately after `_nearest_def_mask` (ends line 206), add:

```python
def _model_relevant_def_pool(grp: pd.DataFrame, gk_team, cpid, *, k: int = 5) -> np.ndarray:
    """v2 placebo pool: the ball-nearest ``k`` DEFENDERS of ``gk_team``, minus the
    ``nearest_def`` (carrier-nearest, v1's control a) by player_id. Mirrors the xS extractor's
    model reference (5 nearest defenders to the BALL; ``def_xy`` is already GK-free, so the
    'minus GK' is a no-op). Returns an array of 0-5 player_ids (4 when nearest_def is among the
    ball-nearest-k, 5 when it is not, fewer on a sparse-defender frame). See spec §3."""
    from silly_kicks.id_compat import ids_match

    ball = grp[grp["is_ball"].astype(bool)]
    if not len(ball):
        return np.empty(0, dtype=object)
    bx, by = float(ball["x"].iloc[0]), float(ball["y"].iloc[0])
    defenders = grp[
        ids_match(grp["team_id"], gk_team) & ~grp["is_ball"].astype(bool) & ~grp["is_goalkeeper"].astype(bool)
    ]
    if not len(defenders):
        return np.empty(0, dtype=object)
    d2 = (defenders["x"].to_numpy(float) - bx) ** 2 + (defenders["y"].to_numpy(float) - by) ** 2
    order = np.argsort(d2, kind="stable")[:k]
    pool = defenders["player_id"].to_numpy()[order]
    nd_mask = _nearest_def_mask(grp, gk_team, cpid)
    if nd_mask is not None:
        nd_id = grp["player_id"].to_numpy()[nd_mask][0]
        # nd_id and pool are both from grp["player_id"] (same-source column) -> a raw != is
        # dtype-safe here (ADR-019 both-column same-source); no cross-source scalar involved.
        pool = pool[pool != nd_id]
    return pool


def _attacker_diag_pool(grp: pd.DataFrame, gk_team, cpid, *, k: int = 5) -> np.ndarray:
    """Non-gating diagnostic pool: up to ``k`` nearest ATTACKERS (the ~gk_team team, non-GK) to
    the ball, with the carrier (``cpid``) excluded by id. Reported only (actor_role
    'attacker_diag'); NEVER banded by evaluate_xs_probe. See spec §3."""
    from silly_kicks.id_compat import ids_match

    ball = grp[grp["is_ball"].astype(bool)]
    if not len(ball):
        return np.empty(0, dtype=object)
    bx, by = float(ball["x"].iloc[0]), float(ball["y"].iloc[0])
    attackers = grp[
        ~ids_match(grp["team_id"], gk_team) & ~grp["is_ball"].astype(bool) & ~grp["is_goalkeeper"].astype(bool)
    ]
    # carrier id crosses columns (ball_carrier_player_id vs player_id) -> ADR-019 ids_match.
    attackers = attackers[~ids_match(attackers["player_id"], cpid)]
    if not len(attackers):
        return np.empty(0, dtype=object)
    d2 = (attackers["x"].to_numpy(float) - bx) ** 2 + (attackers["y"].to_numpy(float) - by) ** 2
    order = np.argsort(d2, kind="stable")[:k]
    return attackers["player_id"].to_numpy()[order]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_xs_placebo_v2.py -v`
Expected: PASS (4 tests, including the nearest-k truncation guard).

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/tracking/_model_eval.py tests/tracking/test_xs_placebo_v2.py
git commit -m "feat(tracking): TF-19 xS v2 placebo pool helpers (defenders + attacker diag)"
```

---

### Task 2: `placebo=` kwarg + attacker_diag emission in the deltas core

**Files:**
- Modify: `silly_kicks/tracking/_model_eval.py` (`_targets_deltas` ~296-385; `substitution_deltas` ~388-435)
- Test: `tests/tracking/test_xs_placebo_v2.py`

- [ ] **Step 1: Write the failing wiring tests**

Append to `tests/tracking/test_xs_placebo_v2.py`:

```python
def _targets_for(frames: pd.DataFrame) -> pd.DataFrame:
    """One ghost target per frame: a DIAGONAL GK->target paired vector (-8, -5). The diagonal is
    load-bearing: an axis-aligned vector (e.g. (-6, 0)) mirrors an x-symmetric defender across the
    ball to the SAME ball-distance -> a false zero delta (verified during planning). Every
    displacement (|v|=9.43) clears the 2 m dose band; ghost not clamped / in box -> trusted."""
    gk = frames[frames["is_goalkeeper"].astype(bool)]
    t = gk[["game_id", "period_id", "frame_id", "x", "y"]].drop_duplicates(
        subset=["game_id", "period_id", "frame_id"]
    )
    return t.assign(
        target_x=t["x"] - 8.0, target_y=t["y"] - 5.0, ghost_clamped=False, ghost_out_of_box=False
    ).drop(columns=["x", "y"])


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
        ("A", "A2", 70.0, 34.0, False), ("A", "A3", 60.0, 20.0, False),
        ("A", "A4", 55.0, 44.0, False), ("A", "A5", 50.0, 10.0, False),
        ("A", "A6", 45.0, 60.0, False),
        ("A", "A7", 40.0, 30.0, False), ("A", "A8", 35.0, 50.0, False),  # far dilution (out of both pools)
        ("A", "A9", 30.0, 18.0, False),
        ("B", "D1", 97.0, 9.0, False),  # nearest defender to carrier -> the excluded nearest_def
        ("B", "D2", 99.0, 13.0, False), ("B", "D3", 100.0, 20.0, False),
        ("B", "Bgk", 104.0, 34.0, True),
    ]
    reps = []
    for i in range(n):
        rows = [
            dict(game_id=f"m{i}", period_id=1, frame_id=1, time_seconds=40.0 + 10.0 * i,
                 team_id=tm, player_id=pid, x=x, y=y, vx=0.0, vy=0.0,
                 is_ball=False, is_goalkeeper=gk, ball_state="alive")
            for tm, pid, x, y, gk in specs
        ]
        rows.append(dict(game_id=f"m{i}", period_id=1, frame_id=1, time_seconds=40.0 + 10.0 * i,
                         team_id="ball", player_id=None, x=96.0, y=8.0, vx=0.0, vy=0.0,
                         is_ball=True, is_goalkeeper=False, ball_state="alive"))
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
        ("A", "A1", 96.0, 8.0, False), ("A", "A2", 99.0, 34.0, False), ("A", "A3", 92.0, 26.0, False),
        ("B", "B1", 100.0, 20.0, False), ("B", "B2", 98.0, 12.0, False), ("B", "B3", 97.0, 14.0, False),
        ("B", "Bgk", 104.0, 34.0, True), ("ball", None, 96.0, 8.0, False),
    ]
    rows = [
        dict(team_id=tm, player_id=pid, x=x, y=y, vx=0.0, vy=0.0,
             is_ball=(tm == "ball"), is_goalkeeper=gk, ball_state="alive")
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
    t = gk[["game_id", "period_id", "frame_id", "x", "y"]].drop_duplicates(
        subset=["game_id", "period_id", "frame_id"]
    )
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
        _DistPlanted("def"), frames, arm="xs", mode="targets", targets=targets, seed=42,
        placebo="model_relevant_def",
    )
    roles = set(deltas["actor_role"].unique())
    assert "placebo_out" in roles and "attacker_diag" in roles
    assert {"gk", "nearest_def"} <= roles


def test_v1_default_placebo_emits_no_attacker_diag_and_is_unchanged():
    frames = _degeneracy_frames()
    targets = _targets_for(frames)
    d_default = me.substitution_deltas(
        _DistPlanted("def"), frames, arm="xs", mode="targets", targets=targets, seed=42
    )
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
```

> **Golden provenance (load-bearing):** these three values were captured during planning by running the
> **pre-refactor** `_model_eval.py` (which has no `placebo=` kwarg — the current v1 random path). That is
> what makes this a genuine `old == new` guard. **Do NOT recapture them after implementing** `placebo=` —
> recapturing post-refactor would make the pin circular (it would enshrine whatever the refactor produced).
> If the pin fails after Step 3, the refactor perturbed the v1 random path — fix the refactor, not the golden.

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_xs_placebo_v2.py -k "emits or unchanged or pinned" -v`
Expected: FAIL — `substitution_deltas() got an unexpected keyword argument 'placebo'` (the pin test also errors on the missing kwarg until Step 3).

- [ ] **Step 3: Thread `placebo=` through `_targets_deltas` and `substitution_deltas`**

In `_targets_deltas` (line 296), change the signature and the two pool seams. Replace the signature line:

```python
def _targets_deltas(model, frames, *, arm, targets, n_placebo_replicates, seed, advance_m) -> pd.DataFrame:
```

with:

```python
def _targets_deltas(model, frames, *, arm, targets, n_placebo_replicates, seed, advance_m, placebo="random") -> pd.DataFrame:
```

Replace the context-building placebo seam (lines 355-357):

```python
        outs = grp[~grp["is_ball"].astype(bool) & ~grp["is_goalkeeper"].astype(bool)]
        out_ids = outs["player_id"].to_numpy()
        contexts.append((grp, gid, pid, fid, moves, kw, out_ids, ghost_clamped, ghost_oob, base_p))
```

with:

```python
        if placebo == "model_relevant_def":
            out_ids = _model_relevant_def_pool(grp, gk_team, cpid)
            attacker_ids = _attacker_diag_pool(grp, gk_team, cpid)
        else:  # "random" -- the frozen v1 pool: all non-ball, non-GK players of BOTH teams
            out_ids = grp[~grp["is_ball"].astype(bool) & ~grp["is_goalkeeper"].astype(bool)]["player_id"].to_numpy()
            attacker_ids = None
        contexts.append((grp, gid, pid, fid, moves, kw, out_ids, attacker_ids, ghost_clamped, ghost_oob, base_p))
```

Replace the replicate loop (lines 375-384):

```python
    for r in range(n_placebo_replicates):
        rng_r = np.random.default_rng(seed + r)
        for grp, gid, pid, fid, moves, kw, out_ids, ghost_clamped, ghost_oob, base_p in contexts:
            if not len(out_ids):
                continue
            rid = rng_r.choice(out_ids, size=1, replace=False)[0]
            pl_mask = grp["player_id"].to_numpy() == rid
            pl_deltas = _delta_for_move(model, grp, pl_mask, moves, extract_fn, kw, base_p=base_p)
            pl_off = _moved_off_pitch(grp, pl_mask, moves)
            rows += _tidy_rows(gid, pid, fid, "placebo_out", r, moves, pl_deltas, pl_off, ghost_clamped, ghost_oob)
    return pd.DataFrame(rows, columns=_TIDY_COLUMNS)
```

with:

```python
    for r in range(n_placebo_replicates):
        rng_r = np.random.default_rng(seed + r)
        for grp, gid, pid, fid, moves, kw, out_ids, attacker_ids, ghost_clamped, ghost_oob, base_p in contexts:
            if len(out_ids):  # placebo draw FIRST -> v1 rng stream is byte-identical
                rid = rng_r.choice(out_ids, size=1, replace=False)[0]
                pl_mask = grp["player_id"].to_numpy() == rid
                pl_deltas = _delta_for_move(model, grp, pl_mask, moves, extract_fn, kw, base_p=base_p)
                pl_off = _moved_off_pitch(grp, pl_mask, moves)
                rows += _tidy_rows(gid, pid, fid, "placebo_out", r, moves, pl_deltas, pl_off, ghost_clamped, ghost_oob)
            if attacker_ids is not None and len(attacker_ids):  # v2 only: reported, never banded
                aid = rng_r.choice(attacker_ids, size=1, replace=False)[0]
                a_mask = grp["player_id"].to_numpy() == aid
                a_deltas = _delta_for_move(model, grp, a_mask, moves, extract_fn, kw, base_p=base_p)
                a_off = _moved_off_pitch(grp, a_mask, moves)
                rows += _tidy_rows(gid, pid, fid, "attacker_diag", r, moves, a_deltas, a_off, ghost_clamped, ghost_oob)
    return pd.DataFrame(rows, columns=_TIDY_COLUMNS)
```

In `substitution_deltas` (line 388) add the kwarg to the signature (after `advance_m: float = 35.0,`):

```python
    placebo: str = "random",  # "random" (frozen v1 pool) | "model_relevant_def" (v2)
```

Add a validation line next to the existing `mode`/`arm` guards (after line 420):

```python
    if placebo not in ("random", "model_relevant_def"):
        raise ValueError(f"unknown placebo: {placebo!r} (expected 'random' or 'model_relevant_def')")
```

And pass it into the targets-mode delegation (the `_targets_deltas(...)` call, ~426):

```python
        return _targets_deltas(
            model,
            frames,
            arm=arm,
            targets=targets,
            n_placebo_replicates=n_placebo_replicates,
            seed=seed,
            advance_m=advance_m,
            placebo=placebo,
        )
```

(Panel mode ignores `placebo` — it has no targets-mode placebo replicates. Leave `_panel_deltas` untouched.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_xs_placebo_v2.py -v`
Expected: PASS (7 tests: 4 pool + emit + unchanged + numerically-pinned).

- [ ] **Step 5: Run the frozen v1 core tests — must stay green**

Run: `python -m pytest tests/tracking/test_model_eval.py -v`
Expected: PASS unchanged (the two registry manifest tests are updated in Task 4; every other test here must already pass — v1 behaviour is byte-identical).

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/tracking/_model_eval.py tests/tracking/test_xs_placebo_v2.py
git commit -m "feat(tracking): TF-19 xS v2 placebo= kwarg + non-gating attacker_diag rows"
```

---

### Task 3: two-sided pool-provenance signature + v2-live-vs-v1-degenerate contrast

**Files:**
- Test: `tests/tracking/test_xs_placebo_v2.py`

These are the "second-producer seam" construct guards (spec §6): prove `placebo_out` is defender-sourced and `attacker_diag` is attacker-sourced, end-to-end through `substitution_deltas`, from BOTH sides; and prove v2 clears the placebo gate on a fixture where the v1 random pool degenerates.

- [ ] **Step 1: Write the failing signature + contrast tests**

Append to `tests/tracking/test_xs_placebo_v2.py`:

```python
def _median_delta(deltas: pd.DataFrame, role: str) -> float:
    sub = deltas[deltas["actor_role"] == role]
    return float(sub["delta_p"].median()) if len(sub) else float("nan")


def test_placebo_out_is_defender_sourced_attacker_diag_is_attacker_sourced():
    # def-only model: moving a DEFENDER registers, moving an ATTACKER is an exact no-op.
    frames = _degeneracy_frames()
    targets = _targets_for(frames)
    d = me.substitution_deltas(
        _DistPlanted("def"), frames, arm="xs", mode="targets", targets=targets, seed=42,
        placebo="model_relevant_def",
    )
    assert _median_delta(d, "placebo_out") > 0  # defenders move the def-only surface
    assert _median_delta(d, "attacker_diag") == 0  # attackers do not -> attacker-sourced


def test_signature_flips_under_an_attacker_only_model():
    # atk-only model: the two-sided converse -> catches the OPPOSITE mis-tagging.
    frames = _degeneracy_frames()
    targets = _targets_for(frames)
    d = me.substitution_deltas(
        _DistPlanted("atk"), frames, arm="xs", mode="targets", targets=targets, seed=42,
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
        _DistPlanted("def"), frames, arm="xs", mode="targets", targets=targets, seed=42,
        placebo="model_relevant_def",
    )
    assert _placebo_p95(d_v2) > 0  # the fix works: v2 clears the no_valid_placebo gate
    assert _placebo_p95(d_v1) == 0.0  # ...and the SAME fixture degenerates v1 (principled, not tuned)
```

- [ ] **Step 2: Run to verify these fail if the wiring is wrong, pass if right**

Run: `python -m pytest tests/tracking/test_xs_placebo_v2.py -k "sourced or signature or degenerate" -v`
Expected: PASS. The three assertions are all execution-verified at seed 42 (def-only: placebo_out median 0.0033, attacker_diag 0.0; atk-only: placebo_out 0.0, attacker_diag 0.0042; v1 placebo_p95 exactly 0.0, v2 > 0). If any differs, the wiring is wrong — do NOT tune the fixture to pass; find the wiring bug.

- [ ] **Step 3: Commit**

```bash
git add tests/tracking/test_xs_placebo_v2.py
git commit -m "test(tracking): TF-19 xS v2 two-sided pool-provenance + degeneracy-contrast guards"
```

---

### Task 4: `xs_substitution_probe_v2` wrapper + registry + discrimination

**Files:**
- Modify: `silly_kicks/tracking/_model_eval.py` (after `xs_substitution_probe` ~618; registry ~669-689)
- Modify: `tests/tracking/test_model_eval.py` (`:11`, `:18` manifest tests)
- Test: `tests/tracking/test_xs_placebo_v2.py`

- [ ] **Step 1: Write the failing wrapper + registry + discrimination tests**

Append to `tests/tracking/test_xs_placebo_v2.py`:

```python
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
    injected_rows = base[base["actor_role"] == "placebo_out"].head(200).assign(
        actor_role="attacker_diag", delta_p=10.0
    )
    injected = pd.concat([base, injected_rows], ignore_index=True)
    got = me.evaluate_xs_probe(injected)
    assert got["placebo_p95"] == ref["placebo_p95"]  # attacker_diag never enters the gated band
    assert got["verdict"] == ref["verdict"]
    assert got["gated_band_median"] == ref["gated_band_median"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_xs_placebo_v2.py -k "registered or relabels or mixed or ignores" -v`
Expected: FAIL — `module ... has no attribute 'xs_substitution_probe_v2'` (the `ignores` test uses only existing symbols and will PASS immediately — it's a standing guard, red only if the evaluator regresses).

- [ ] **Step 3: Add the wrapper and register it**

In `silly_kicks/tracking/_model_eval.py`, immediately after `xs_substitution_probe` (ends line 617), add:

```python
def xs_substitution_probe_v2(model, frames, targets, *, seed: int = 42) -> dict:
    """The v2 xS probe (ADR-037 amendment): same rule as v1, but the placebo pool is the
    model-relevant defenders (``placebo='model_relevant_def'``) instead of random outfielders.
    Reuses ``evaluate_xs_probe`` verbatim; relabels the report ``rule`` (the pure evaluator is
    unchanged). See docs/superpowers/specs/2026-07-23-tf19-xs-placebo-v2-design.md."""
    deltas = substitution_deltas(
        model, frames, arm="xs", mode="targets", targets=targets, seed=seed, placebo="model_relevant_def"
    )
    out = evaluate_xs_probe(deltas)
    out["rule"] = "xs-dose-banded-v2"  # wrapper-level relabel; evaluator emits v1's constant
    out["placebo_pool"] = "model_relevant_def"
    gk = deltas[deltas["actor_role"] == "gk"]
    out["n_frames_used"] = len(gk[["game_id", "period_id", "frame_id"]].drop_duplicates())
    return out
```

In the registry block (after the `_register_wrapper("xs", ...)` call ends at line 689), add:

```python
_register_wrapper(
    "xs_v2",
    xs_substitution_probe_v2,
    {
        **PROBE_WRAPPERS["xs"]["rule_constants"],  # identical numeric rule
        "placebo_pool": "model_relevant_def",  # the ONE difference, self-documented
    },
)
```

- [ ] **Step 4: Update the two frozen manifest tests**

In `tests/tracking/test_model_eval.py`, line 12:

```python
    assert set(me.PROBE_WRAPPERS) == {"xcross", "xs"}
```

becomes:

```python
    assert set(me.PROBE_WRAPPERS) == {"xcross", "xs", "xs_v2"}
```

and in `PINNED_RULES` (lines 20-23) add the `xs_v2` entry:

```python
    PINNED_RULES = {
        "xcross": {"ratio": 2.0, "abs_floor": 0.01},
        "xs": {"ratio": 2.0, "dose_m": 2.0, "placebo_band_pct": 95.0},
        "xs_v2": {"ratio": 2.0, "dose_m": 2.0, "placebo_band_pct": 95.0, "placebo_pool": "model_relevant_def"},
    }
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_xs_placebo_v2.py tests/tracking/test_model_eval.py -v`
Expected: PASS (all new v2 tests + the updated manifest tests + every frozen v1 test).

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/tracking/_model_eval.py tests/tracking/test_model_eval.py tests/tracking/test_xs_placebo_v2.py
git commit -m "feat(tracking): register xs_v2 probe wrapper (relevance-matched placebo)"
```

---

### Task 5: driver `--variant` + two-variant report + lock-commit hash

**Files:**
- Modify: `scripts/validate_xs_probe.py`
- Modify: `tests/scripts/test_validate_xs_probe.py`

- [ ] **Step 1: Write the failing driver tests (new two-variant shape)**

Rewrite `tests/scripts/test_validate_xs_probe.py`'s `_fake_metrics`, the render assertions, and the `run` test to the new shape. Replace the whole `_fake_metrics` function with:

```python
def _fake_probe(verdict="fail"):
    return {
        "verdict": verdict,
        "n_frames_used": 123,
        "gated_band_median": 0.003,
        "nearest_def_median": 0.0006,
        "placebo_p95": 0.02,
        "gated_band_n": 150,
        "gated_band_zero_fraction": 0.1,
        "off_pitch_control_fraction": 0.0,
        "dose_response_rho": 0.4,
        "dose_response_p": 0.03,
    }


def _fake_metrics(v1_verdict="no_valid_placebo", v2_verdict="pass"):
    return {
        "arm": "xs",
        "variants": {
            "v1": {"probe": _fake_probe(v1_verdict), "regate_verdict": "unmeasurable_at_dose",
                   "rule_constants": {"ratio": 2.0, "min_band_n": 100}},
            "v2": {"probe": {**_fake_probe(v2_verdict), "attacker_diag_p95": 0.05,
                             "rule": "xs-dose-banded-v2", "placebo_pool": "model_relevant_def"},
                   "regate_verdict": "joins_with_caveat",
                   "rule_constants": {"ratio": 2.0, "min_band_n": 100, "placebo_pool": "model_relevant_def"}},
        },
        "entanglement": "inside_band",
        "reconciliation": {
            "total_targets": 200, "n_frames_used": 123, "n_distinct_games": 40,
            "gated_band_n": 150, "targets_to_used_drop_frac": 0.385,
        },
        "corpus": {"n_matches": 5, "match_ids": ["a"] * 5},
        "per_match": [],
        "seed": 42,
        "tracking_limit": None,
        "rng_discipline": "per-match placebo streams",
        "lock_commit": "1abc",
        "run_commit": "deadbeef",
    }
```

Replace the three render/write tests with:

```python
def test_re_gate_maps_fail_inside_band_to_gated_clean_fail():
    assert mod.re_gate("fail", "inside_band") == "gated_clean_fail"


def test_render_shows_both_variants_and_the_lock_commit():
    out = mod._render(_fake_metrics(v1_verdict="no_valid_placebo", v2_verdict="pass"))
    assert "v1" in out and "v2" in out
    assert "no_valid_placebo" in out and "pass" in out
    assert "Re-gate" in out and "reconciliation" in out.lower()
    assert "1abc" in out  # lock commit is auditable in the report


def test_render_shows_na_on_unmeasurable_branch():
    m = _fake_metrics(v2_verdict="unmeasurable_at_dose")
    for k in ("gated_band_median", "nearest_def_median", "placebo_p95", "dose_response_rho", "dose_response_p"):
        m["variants"]["v2"]["probe"][k] = None
    out = mod._render(m)
    assert "n/a (unmeasurable)" in out


def test_write_produces_both_files(tmp_path):
    mod._write(tmp_path, _fake_metrics())
    assert json.loads((tmp_path / "metrics.json").read_text())["arm"] == "xs"
    assert (tmp_path / "report.md").read_text().startswith("# TF-19")
```

Replace `test_run_pools_two_matches_with_DISTINCT_games` with a version that stubs per-variant deltas and asserts the two-variant metrics shape:

```python
def test_run_pools_two_matches_and_scores_both_variants(monkeypatch, tmp_path):
    _gids = iter((100, 101, 200, 201))  # v1 then v2 per match -> 4 substitution_deltas calls

    def fake_load_matches(**kwargs):
        for mid in ("m0", "m1"):
            yield ("gradientsports", mid, pd.DataFrame(), pd.DataFrame(), 1)

    fake_loader = type(sys)("_loader_pining")
    monkeypatch.setitem(sys.modules, "_loader_pining", fake_loader)
    monkeypatch.setattr(fake_loader, "load_matches", fake_load_matches, raising=False)
    monkeypatch.setattr(mod.GhostGkModel, "from_variant", staticmethod(lambda v="default": object()))
    monkeypatch.setattr(mod.XShotOccurrenceModel, "from_variant", staticmethod(lambda v="default": object()))
    monkeypatch.setattr(mod, "build_ghost_frames", lambda frames, **k: (None, pd.DataFrame(), _FakeReport()))
    monkeypatch.setattr(mod, "provenance_to_targets", lambda prov, **k: pd.DataFrame({"x": [0]}))
    monkeypatch.setattr(mod, "substitution_deltas", lambda *a, **k: _fake_deltas(next(_gids)))
    monkeypatch.setattr(mod, "evaluate_xs_probe", lambda d: {"verdict": "unmeasurable_at_dose", "gated_band_n": 0})

    m = mod.run(tmp_path, entanglement="inside_band", seed=7, lock_commit="1abc")
    assert m["corpus"]["n_matches"] == 2
    assert set(m["variants"]) == {"v1", "v2"}
    assert m["variants"]["v1"]["probe"]["verdict"] == "unmeasurable_at_dose"
    assert m["variants"]["v2"]["probe"]["verdict"] == "unmeasurable_at_dose"
    assert m["lock_commit"] == "1abc"
    assert m["seed"] == 7 and (tmp_path / "metrics.json").exists()
```

Keep `test_run_empty_corpus_raises_systemexit` as-is (it does not depend on the shape).

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/scripts/test_validate_xs_probe.py -v`
Expected: FAIL — `_render`/`run` still emit the old single-variant shape (`KeyError: 'variants'`).

- [ ] **Step 3: Refactor the driver to two variants**

In `scripts/validate_xs_probe.py`:

Add a small helper near `_n_unique_gk_frames` (line 45):

```python
def _attacker_diag_p95(deltas: pd.DataFrame) -> float:
    """Non-gating attacker-diagnostic p95 (95th pct of per-replicate medians)."""
    import numpy as np

    sub = deltas[deltas["actor_role"] == "attacker_diag"]
    rep_med = sub.groupby("replicate")["delta_p"].median()
    return float(np.percentile(rep_med, 95.0)) if len(rep_med) else float("nan")


_VARIANT_PLACEBO = {"v1": "random", "v2": "model_relevant_def"}
_VARIANT_WRAPPER = {"v1": "xs", "v2": "xs_v2"}
```

Replace the body of `run` (lines 61-138) with:

```python
def run(out, *, variant="both", match_ids=None, tracking_limit=None, entanglement="inside_band", seed=42,
        token=None, lock_commit=None):
    from _loader_pining import load_matches  # scripts/ on sys.path at runtime (mirrors the trainer)

    variants = ["v1", "v2"] if variant == "both" else [variant]
    ghost_model = GhostGkModel.from_variant("default")
    xs_model = XShotOccurrenceModel.from_variant("default")

    per_variant_deltas = {v: [] for v in variants}
    per_match = []
    for _provider, match_id, _actions, frames, home_team_id in load_matches(
        providers=_PROVIDERS, match_ids=match_ids, token=token, tracking_limit=tracking_limit
    ):
        htid = cast("int | str", home_team_id)
        _cf, prov, report = build_ghost_frames(frames, model=ghost_model, home_team_id=htid)
        targets = provenance_to_targets(prov, frames=frames, home_team_id=htid)
        for v in variants:
            deltas = substitution_deltas(
                xs_model, frames, arm="xs", mode="targets", targets=targets, seed=seed,
                placebo=_VARIANT_PLACEBO[v],
            )
            per_variant_deltas[v].append(deltas)
        per_match.append({
            "match_id": match_id, "n_frames_in": report.n_frames_in,
            "n_frames_scored": report.n_frames_scored, "drop_reasons": report.drop_reasons,
            "n_targets": len(targets),
        })

    if not per_match:
        raise SystemExit("no GS matches loaded — check PINING_FOR_THE_DATA_TOKEN / --match-ids-json")

    results = {}
    for v in variants:
        pooled = pd.concat(per_variant_deltas[v], ignore_index=True)
        res = evaluate_xs_probe(pooled)
        res["n_frames_used"] = _n_unique_gk_frames(pooled)
        if v == "v2":
            res["rule"] = "xs-dose-banded-v2"
            res["placebo_pool"] = "model_relevant_def"
            res["attacker_diag_p95"] = _attacker_diag_p95(pooled)
        results[v] = {
            "probe": res,
            "regate_verdict": re_gate(res["verdict"], entanglement),
            "rule_constants": PROBE_WRAPPERS[_VARIANT_WRAPPER[v]]["rule_constants"],
        }

    ref = results[variants[0]]["probe"]  # gk stratum is pool-independent -> frames_used identical
    total_targets = sum(m["n_targets"] for m in per_match)
    n_games = int(pd.concat(per_variant_deltas[variants[0]], ignore_index=True)["game_id"].nunique())
    used = ref["n_frames_used"]
    reconciliation = {
        "total_targets": total_targets, "n_frames_used": used, "n_distinct_games": n_games,
        "gated_band_n": ref.get("gated_band_n"),
        "targets_to_used_drop_frac": (1.0 - used / total_targets) if total_targets else None,
    }

    metrics = {
        "arm": "xs",
        "variants": results,
        "entanglement": entanglement,
        "reconciliation": reconciliation,
        "corpus": {"providers": _PROVIDERS, "n_matches": len(per_match),
                   "match_ids": [m["match_id"] for m in per_match]},
        "per_match": per_match,
        "seed": seed,
        "tracking_limit": tracking_limit,
        "rng_discipline": "per-match placebo streams (substitution_deltas per match+variant, seed pinned)",
        "lock_commit": lock_commit or _baseline_commit(),
        "run_commit": _baseline_commit(),
    }
    _write(out, metrics)
    return metrics
```

Replace `_render` (lines 159-216) with a two-variant renderer:

```python
def _variant_block(name: str, entry: dict, rc: dict) -> list[str]:
    p, verdict = entry["probe"], entry["probe"]["verdict"]
    prongs_omitted = p.get("gated_band_median") is None
    lines = [
        f"### {name}: `{verdict}`   re-gate: `{entry['regate_verdict']}`"
        + (f"   ({p['placebo_pool']} placebo)" if p.get("placebo_pool") else "   (random placebo)"),
        f"- gated_band_n: {p.get('gated_band_n')} (needs >= {rc.get('min_band_n')})   "
        f"frames_used: {p.get('n_frames_used')}",
        _dose_ladder_line(p),
        f"- nearest_def control: {_fmt(p.get('nearest_def_median'))}   "
        f"placebo_p95: {_fmt(p.get('placebo_p95'))}   "
        f"gated_band_median: {_fmt(p.get('gated_band_median'))}",
        _dose_ratio_line(p),
        f"- dose_response rho / p: {_fmt(p.get('dose_response_rho'))} / {_fmt(p.get('dose_response_p'))}"
        + ("   (prongs omitted — unmeasurable)" if prongs_omitted else ""),
    ]
    if p.get("attacker_diag_p95") is not None:
        lines.append(f"- attacker diagnostic p95 (non-gating): {_fmt(p.get('attacker_diag_p95'))}")
    return lines


def _render(m: dict) -> str:
    rec = m["reconciliation"]
    variants = m["variants"]
    body = [
        "# TF-19 PR-3b xS-arm probe — v1 (random) vs v2 (model-relevant defenders)",
        "",
        f"**Entanglement:** {m['entanglement']}   **seed:** {m['seed']}   "
        f"**Matches:** {m['corpus']['n_matches']}   **Games:** {rec.get('n_distinct_games')}",
        f"**Lock commit:** `{m.get('lock_commit')}`   **Run commit:** `{m.get('run_commit')}`   "
        "(blindness: constants locked before the run; verify any intervening diff is inert)",
        "",
    ]
    if "v2" in variants:  # the honest framing is ABOUT v2 -> only print it on a run that has v2
        body += [
            "## The honest framing",
            "- v2 changes EXACTLY ONE thing vs v1: the placebo pool (random outfielder -> ball-nearest "
            "defenders). The defender placebo is a WEAKER control than nearest_def, so it is INERT in the "
            "ratio (`max()` pins to nearest_def); its job is to clear the no_valid_placebo gate with a "
            "principled null, not to move the bar.",
            "- The ratio prong is therefore a 'beat nearest_def by 2x' test, near-certain to pass. v2's REAL "
            "decider is the clustered dose-response permutation, which v1 never reached.",
            "- The attacker diagnostic is reported (non-gating): the nearest attacker is the shooter, so "
            "gating on attackers would answer a model-sensitivity question, not a deterrence one.",
            "",
        ]
    for name in ("v1", "v2"):
        if name in variants:
            body += ["## " + ("v1 (frozen random placebo)" if name == "v1" else "v2 (relevance-matched)"), ""]
            body += _variant_block(name, variants[name], variants[name]["rule_constants"])
            body += [""]
    body += [
        "## Targets -> used -> band reconciliation",
        f"- total targets: {rec['total_targets']}   n_frames_used: {rec['n_frames_used']}   "
        f"distinct games: {rec.get('n_distinct_games')}   gated_band_n: {rec['gated_band_n']}",
        f"- targets->used drop frac: {rec['targets_to_used_drop_frac']} "
        "(a drop is EXPECTED — ghost vs xs carrier-resolver mismatch; read as 'above that baseline').",
        "",
    ]
    return "\n".join(body) + "\n"
```

Update `main()` to add the CLI flags (after the `--seed` arg, ~line 241):

```python
    ap.add_argument("--variant", choices=["v1", "v2", "both"], default="both",
                    help="which placebo variant(s) to run (default: both, side by side)")
    ap.add_argument("--lock-commit", default=None,
                    help="the commit that froze the v2 pool+constants (auditable blindness; "
                         "defaults to HEAD). Record it so the git DAG shows constants-locked-before-run.")
```

and thread them into the `run(...)` call + the print line:

```python
    m = run(
        args.out,
        variant=args.variant,
        match_ids=match_ids,
        tracking_limit=args.tracking_limit,
        entanglement=args.entanglement,
        seed=args.seed,
        lock_commit=args.lock_commit,
    )
    v2 = m["variants"].get("v2") or next(iter(m["variants"].values()))
    print(
        f"v1={m['variants'].get('v1', {}).get('probe', {}).get('verdict')}  "
        f"v2={v2['probe']['verdict']}  regate_v2={v2['regate_verdict']}  "
        f"matches={m['corpus']['n_matches']}  lock={m.get('lock_commit')}"
    )
```

Also update the module docstring (lines 1-14) so it names the v2 dir and `--variant`; and update `_dose_ladder_line`/`_dose_ratio_line` are already variant-agnostic (they take a probe dict) — no change.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/scripts/test_validate_xs_probe.py -v`
Expected: PASS (all driver tests on the new shape).

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_xs_probe.py tests/scripts/test_validate_xs_probe.py
git commit -m "feat(scripts): validate_xs_probe --variant {v1,v2,both} + lock-commit provenance"
```

---

### Task 6: PRIVATE_CONSUMERS update + full-suite/lint/type + lock-ready handoff

**Files:**
- Modify: `docs/PRIVATE_CONSUMERS.md:25`

- [ ] **Step 1: Add the new symbol to the pinned consumer row**

In `docs/PRIVATE_CONSUMERS.md`, line 25, extend the "probe symbols" list to include `xs_substitution_probe_v2` (the driver now imports it path-wise via `PROBE_WRAPPERS`, and the wiring test references the wrapper). Change:

```
probe symbols `xs_substitution_probe`, `evaluate_xs_probe`, `substitution_deltas`, `regate_verdict`, `PROBE_WRAPPERS`, `_validate_targets`
```

to:

```
probe symbols `xs_substitution_probe`, `xs_substitution_probe_v2`, `evaluate_xs_probe`, `substitution_deltas`, `regate_verdict`, `PROBE_WRAPPERS`, `_validate_targets`
```

(Doc edit rides with the code PR — no standalone doc commit.)

- [ ] **Step 2: Lint + type the full tree (Shift Left; CI lints the whole tree, not just changed files)**

Run: `.venv312\Scripts\ruff.exe check silly_kicks scripts tests`
Run: `.venv312\Scripts\ruff.exe format --check silly_kicks scripts tests`
Run: `.venv312\Scripts\pyright.exe`
Expected: no errors. (Run FULL-tree pyright, not single-file — a test-stub attribute error only surfaces tree-wide; see the CI lesson.)

- [ ] **Step 3: Run the full non-e2e suite on the CI-repro venv**

Run: `.venv312\Scripts\python.exe -m pytest tests/ -m "not e2e" -q`
Expected: all green (the prior baseline was 5582 passed; this adds the v2 tests). If any frozen v1 test regressed, STOP — v1 must be byte-identical; a regression means the `placebo="random"` default path was perturbed.

- [ ] **Step 4: Sanity-run the driver's pure path headless (no data)**

Run: `.venv312\Scripts\python.exe -c "import scripts.validate_xs_probe as m; print(m._render.__name__, m._VARIANT_PLACEBO)"`
Expected: prints `_render {'v1': 'random', 'v2': 'model_relevant_def'}` — imports clean, no runtime edge.

- [ ] **Step 5: Stop — hand to commit-prep (do NOT commit the release without approval)**

The code is lock-ready. Do NOT bump the version, regenerate C4, or create the release commit here. Surface to the owner for the commit-prep decisions:
- **The lock commit** is the single squashed PR commit (code + tests + PRIVATE_CONSUMERS). Its hash is what the owner passes to `--lock-commit` at run time (blindness audit). Confirm the intended commit before running.
- **ADR (spec §9):** amend ADR-037 (register the v2 rule + `placebo=` + the `attacker_diag` role) vs a small new ADR — owner decides.
- **Version bump (5 sites) + `PR-S<NN>` + `vX.Y.Z` tag + PyPI** per the per-PR convention — owner-approved, ONE commit.
- **C4:** unchanged (research instrument, not an aggregator; count stays 30) — `final-review` will confirm, no DSL edit expected.
- **NOTICE:** no new reference (placebo redesign within ADR-037; xS attribution unchanged).

---

## Post-lock owner run (NOT a TDD task — recorded for completeness)

After the lock commit lands, the owner runs the ~64-match GS probe from the lock commit (blindness: the run must happen AFTER the constants are frozen), writing to the v2 research dir:

```bash
# from the lock commit; pining access required; ~1.5-3x the PR-3b runtime (both variants + attacker diag)
python scripts/validate_xs_probe.py --out docs/research/tf19_pr3b_xs_v2 --variant both --lock-commit <lock-sha>
```

Expected deliverable: `docs/research/tf19_pr3b_xs_v2/{metrics.json, report.md}` reporting v1's `no_valid_placebo` and v2's verdict side by side (with the attacker diagnostic), and `lock_commit == <lock-sha>`. The v2 verdict is whatever the locked rule returns — near-certainly the ratio passes; `pass` vs `band_pass_flat_dose_response` turns on the clustered dose-response p-value (the genuine open question). To save runtime the owner may use `--variant v2` and cite the frozen PR-3b v1 verdict by reference; `--variant both` re-proves v1 reproduces PR-3b exactly.

---

## Self-Review

**Spec coverage:**
- §2 D1 (new variant alongside frozen v1) → Task 4 (`xs_v2` registered; v1 untouched, frozen tests green).
- §2 D2 (defender-only pool, attackers non-gating diagnostic) → Task 1 helpers + Task 2 emission + Task 3 two-sided signature.
- §2 D3 (only the pool changes; all constants frozen) → Task 4 registry (`**rc_v1`) + the manifest test asserting numeric identity.
- §2 "inert in the ratio / dose-response is the decider" → surfaced in the driver `_render` honest-framing block (Task 5, guarded to v2-present runs) + report deliverable.
- §3 (pool 0-5, minus nearest_def, empty-pool no fabricated 0, **nearest-k truncation**) → Task 1 helper + the `_truncation_frame` guard + Task 2 `if len(out_ids)` guard.
- §3 attacker_diag distinct role, ignored by evaluator → Task 2 emission + Task 3 signature + the fact evaluate_xs_probe bands only `placebo_out`/`nearest_def`.
- §4 (evaluate_xs_probe verbatim) → no evaluator edit anywhere; wrapper relabels `rule` post-hoc.
- §5 (code structure: helpers, `placebo=`, `xs_v2`, wrapper, driver `--variant`, per-variant frames) → Tasks 1,2,4,5. Per-variant frames = two `substitution_deltas` calls (one placebo population each) → MAJOR-2 collision impossible.
- §6 (CI: non-degeneracy both sides, discrimination, construct guards two levels, v1 frozen, auditable lock) → Task 3 (contrast + signature), Task 4 (discrimination + frozen manifest), Task 5 (`lock_commit` in metrics.json).
- §6/§8 (lock-commit hash auditable) → Task 5 `--lock-commit` + `lock_commit`/`run_commit` fields + report line.
- §10 (C4-free, no retrain, no NOTICE) → Task 6 handoff notes.

**Placeholder scan:** none — every code step shows complete code; every numeric assertion (the truncation pool membership, the two-sided signature medians, `p95_v1 == 0.0`, and the three numeric-pin goldens) was captured by running the real code during planning, not assumed.

**Type consistency:** `_model_relevant_def_pool`/`_attacker_diag_pool` return `np.ndarray` of player_ids, consumed by `rng_r.choice(...)` and `grp["player_id"].to_numpy() == rid` — same idiom as v1's `out_ids`. `placebo` is `str` with the same allowed-set validated in `substitution_deltas` and switched in `_targets_deltas`. `xs_substitution_probe_v2` returns the same `dict` shape as `xs_substitution_probe` plus `rule`/`placebo_pool`. Driver `metrics["variants"][v]["probe"]` is the evaluator dict; `_render`/tests agree on the nested shape.

**Frozen-v1 safety (old==new for THIS refactor, three independent backstops):** (1) `test_v1_random_path_is_numerically_pinned` pins `gated_band_median`/`nearest_def_median`/`placebo_p95` to goldens **captured on the pre-refactor code** (see the golden-provenance note in Task 2) — a genuine `old==new` guard that fails on any perturbation of the random draw / `base_p` / vector this refactor might introduce. (2) By inspection the v1 path is byte-identical by construction: for `placebo="random"`, `attacker_ids is None`, `out_ids` is the same expression, and the `if not len: continue` → `if len:` restructure consumes `rng_r` identically (draw only when non-empty). (3) The post-lock `--variant both` run re-proves "v1 reproduces PR-3b exactly" on real data. `assert_frame_equal(d_default, d_explicit)` is intentionally NOT counted here — it proves default routes to `"random"`, not old==new (both sides run new code). Belt-and-suspenders: the placebo draw stays FIRST in the replicate loop, and `evaluate_xs_probe`, `_panel_deltas`, `_nearest_def_mask`, `regate_verdict`, and the `xs`/`xcross` registrations are not edited (Task 2 Step 5 reruns the frozen `test_model_eval.py` behavioural suite).
