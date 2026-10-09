# ADR-117: TF-65 kinematics smoother — dense-grid gap fix, pre-registered A/B smoother selection, acceleration + uncertainty

| Field | Value |
|---|---|
| **Date** | 2026-10-09 |
| **Status** | Proposed (A/B pre-registered; point-smoother decision filled post-A/B) |
| **Deciders** | Karsten; Claude (Opus 4.8) |

## Context

`tracking.preprocess` smoothed positions and derived velocity on **row order**, ignoring `frame_id`
spacing: a non-detection is a MISSING ROW (not a NaN run), so the pre/post-gap detections sat adjacent
and the Savitzky-Golay window fabricated through-gap velocity — measured **500–850 m/s** ball speed on
broadcast providers (GS 10502: 532 p1 / 367 p2), feeding ghost-GK (`ball_vx/vy/speed`), elastic-sync and
the per-action linked-frame ball state. A ceiling-measurement spike (6 arms, 8 matches, scratch) sized
the prize and showed the smoother choice matters beyond the gap. Value-changing → a batched retrain is
eaten anyway, which makes a downstream A/B affordable. The smoother choice cannot be made by aesthetics;
it must be decided by data, pre-registered to avoid post-hoc metric selection.

## Decision

1. **Dense-grid `frame_id` reindex** in `smooth_frames`/`derive_velocities`: reindex each
   `(game_id, period_id, is_ball, player_id)` group to its contiguous range, bridge ≤ `max_gap_seconds`,
   **segment at > max_gap** (no window spans a big gap), gap-adjacent edge-NaN only (contiguous groups
   byte-identical). `game_id` joins the `derive_velocities` + `interpolate_frames` group key (A-31 parity).
2. **The point smoother is selected by a pre-registered in-cycle A/B** (below) among Butterworth,
   Kalman/RTS, and the SG + dense-grid floor; **ship the floor** if no finalist clears.
3. **Acceleration** (`accel_x`/`accel_y`/`accel`) + an **always-on constant-acceleration Kalman/RTS
   per-frame uncertainty** (`pos_var`/`vel_var`/`accel_var`), decoupled from the point smoother (shipped
   regardless of the A/B winner). Components are `accel_x`/`accel_y`, **not `ax`/`ay`** — those collide
   with `_kernels.py` triangle anchor-coordinate columns in `add_action_context`'s merge.
4. **Derived kinematic columns are NOT declared** in `TRACKING_FRAMES_COLUMNS` / `metric_contracts` /
   `feature_glossary` — they follow the `vx`/`vy` convention (preprocess-added; declared only in
   `reflection.py`). Purity gated by a dedicated `tests/tracking/preprocess/test_preprocess_purity.py`
   (`PURITY_ENTRIES` is `add_*`-exact).
5. **Soft `≤40 m/s` plausibility guard** (`PlausibilityWarning`, NaN + count, never raise). The
   `≤10 m/s²` accel bound is **provisional** (re-tuned at the A/B / TF-50 when accel is consumed).

### Pre-registration of the A/B (recorded 2026-10-09, BEFORE the run)

- **Arms:** SG + dense-grid (floor) · Butterworth (Winter dual-pass) · Kalman/RTS CA. Identical model
  hyperparameters + seeds; the ONLY varying input is the smoother. Split by MATCH (no frame leakage).
  Representative velocity-sensitive model subset: ghost-GK (primary) + xshot/xcross (guardrail).
- **Corpus:** MAE primary on the **full 179-match corpus** (SK 108 / GS 64 / idsse 7). Density-NLL
  confirmatory on a **fixed 10-match subset** = per provider, the lexicographically-first shard
  basenames (5 skillcorner + 4 gradientsports + 1 idsse) — deterministic, by rule, no match-id list.
- **Primary endpoint (RE-AMENDED 2026-10-09, owner-approved, pre-run — HYBRID):**
  - *Decision (full corpus):* ghost-GK out-of-sample **Euclidean MAE** (metres) from `predict_mean`.
  - *Confirmatory (fixed subset):* ghost-GK **density-NLL** `−log p(true keeper cell | predicted
    density)` from `predict_density`, `kde_backend` pinned **`"vectorized"`** (THE exact numpy closed-form
    raw grid — platform-independent, numba-free; `predict_density` has no `"exact"` literal, so the exact
    raw grid IS `"vectorized"`). A strictly-proper score on the consumed density; the MAE winner must not
    regress it vs the floor. Per-match mean NLL from a seeded 150-row eval subsample (predict_density
    ~2 s/row even at the subset's ~88k train fold; full eval infeasible).
  - *Why hybrid (measured in-cycle):* density-NLL is leaf-match KDE, **O(n_train) per eval row**
    (`_leaf_match_weights` = `(n_query, n_train, n_trees)` broadcast); measured 148 ms/row (cpu-numba) at
    n_train 11.7k → ~23 s/row at full-179 train → infeasible full-corpus (cost ≈ L²·K²; tractable only at
    K ≲ 12). MAE (`predict_mean` ≈ 0.2 ms/row) carries full-corpus power; density-NLL confirms calibration
    on a bounded subset. [Supersedes the earlier 2026-10-09 density-NLL-**primary** amendment — infeasible
    at corpus scale. Original "log-loss" was also wrong: ghost-GK is a position/density model.]
- **Superiority test:** (primary) a finalist beats the floor iff the one-sided 95% paired match-level
  bootstrap CI of `[floor − finalist]` per-match mean **MAE excludes 0** (B=2000, fixed seed; lower MAE
  better). (confirmatory) the chosen arm's per-match mean density-NLL on the subset must not be worse than
  the floor's (a regression disqualifies + triggers re-evaluation).
- **Guardrails (any regression vs floor ⇒ disqualify):** implausible-kinematics rate (>12, >40 m/s,
  |accel|>10), velocity coverage (non-NaN not lower by >0.5 pp), accel plausibility (p99 within bound) —
  all FULL corpus, free from materialize, and the direct measure of the fabricated-velocity fix.
  **xshot/xcross log-loss** on a **fixed 30-match subset** (17 SK + 10 GS + 3 idsse, lexicographically-
  first; default params, identical across arms; full-179 would add ~18.6h — measured +124 s/match).
  **DAS→near-term-danger AUC DEFERRED** (owner-approved): full-179 × 3 arms is multi-day; the frame
  guardrails already gate the DAS-facing velocity consumer; a bounded DAS sweep is a follow-up only if a
  finalist is marginal.
- **Decision rule:** largest statistically-distinguishable MAE improvement with no guardrail/density breach;
  tie (overlapping CIs) → DAS validity → Kalman coherent uncertainty → simplicity; else ship the floor.

### A/B outcome (filled post-run)

_TBD — recorded here when the A/B completes; Status → Accepted with the chosen smoother._

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Row-order SG (status quo) | simplest | fabricates through-gap velocity (500–850 m/s) | the bug being fixed |
| B. Pick a smoother by judgment / spike alone | cheap | not data-decided; the spike sizes the prize, doesn't crown | owner: data decides |
| C. Primary = ghost-GK MAE (point) only | trainer emits it; tractable full-corpus | ignores density calibration/sharpness — the consumed artifact | folded into the hybrid as the decision driver, density added as a confirmatory guard |
| D. Primary = density-NLL (full corpus) | proper score on the consumed density | **computationally infeasible** — O(n_train)/row, ~23 s/row at full-179 train (measured) | infeasible at corpus scale |
| E. (chosen) Dense-grid + pre-registered A/B, **hybrid**: MAE primary full-179 + density-NLL confirmatory on a fixed 10-match subset | full-corpus decision power AND a proper-score density guard, both tractable | multi-hour A/B + retrain | — |
| E. GCV spline arm | classical | dominated in the spike (noisy accel) | dropped |

## Consequences

### Positive
- Honest gap handling: no fabricated through-gap velocity; `x_smoothed` fixed too.
- Acceleration + per-frame uncertainty unlocked (gates TF-50; honest-gap confidence).
- The smoother choice is data-decided + pre-registered; the floor is a legitimate outcome.

### Negative
- A batched retrain (6 F1b models + DAS-golden + T10) + a Hyrum notice (velocity values move on broadcast
  providers; new columns).
- Velocity NaN across > `max_gap` gaps + edge-NaN near gaps = some coverage loss (broadcast only).
- `accel`/`*_var` dtype + the `accel_x`/`accel_y` naming are new observable contract (Hyrum; TF-50 + marts).

### Neutral
- Contiguous single-detection-run data is byte-identical (the floor path); full-pitch providers unaffected.
- `das_golden` byte-exact test is aarch64-float-sensitive (last-ULP) — an x86/CI artifact, unrelated.
