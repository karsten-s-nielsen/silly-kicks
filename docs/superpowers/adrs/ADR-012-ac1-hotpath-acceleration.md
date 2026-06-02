# ADR-012: Action-context hot-path acceleration — vectorized ghost-GK KDE + DAS carrier-offside contract

| Field | Value |
|---|---|
| **Date** | 2026-06-01 |
| **Status** | Accepted (amended 4.4.1 — DAS-value-neutral claim corrected, see Amendment) |
| **Deciders** | Karsten Nielsen (maintainer), lakehouse review session |

## Amendment (silly-kicks 4.4.1) — the DAS carrier-forwarding was NOT value-neutral

This ADR (and the 4.2.0 changelog) claimed the ball-carrier offside forwarding was "value-neutral
(zero AS/DAS change) on real data." **That was incorrect.** The validating A/B test
(`test_das_offside.py`) placed the carrier clearly onside, so it never exercised the offside path;
the "real data" A/B likewise did not hit a frame where the carrier crossed the offside line. A
lakehouse golden e2e (IDSSE J03WMX p1) caught it: DAS changed 4.1.1→4.2.0 (`das_diff` maxΔ ≈ 261).

Root cause (verified, accessible-space `core.py:183-204`): with `respect_offside` (the DAS default)
accessible-space **deletes** offside attackers ("treats them like air"). The offside line is
`max(2nd-last-defender_x, ball_x)` and the attacking team comes from `team_in_possession` — both
identical before/after. The *only* delta is that forwarding `player_in_possession_col` **exempts the
passer** from offside removal. On real matches the on-ball carrier is frequently tracked ~0.5 m ahead
of the ball, tipping just over the line; 4.1.1 (no passer forwarded) deleted that on-ball player,
redistributing their central space and distorting AS/DAS (e.g. inflating a nearby defender's
`das_opponent` to an implausible ~367 m²). **So 4.2.0 is a correctness fix, not a regression** — it
is precisely the failure accessible-space's own warning describes ("the ball carrier might be
mis-identified as offside"). The carrier is correctly resolved (its team always equals
`team_in_possession`; on the affected frames it is the nearest player to the ball). The effect is
large but rare (≈1% of frames). **Downstream goldens frozen under ≤4.1.1 must be re-baselined** to
the ≥4.2.0 values. Test correction + the regression-lock (`test_offside_carrier_forwarding_changes_das`)
landed in 4.4.1.

## Context

Full-chain profiling of the lakehouse action-context enrichment pipeline (skillcorner 2011166,
silly-kicks 4.1.1, serverless) found `add_ghost_gk` is **~74 % of the chain wall time**, dominated
by `_ghost_gk.predict_density`'s per-sample `scipy.stats.gaussian_kde` (534 calls, ~1.74 s/call);
DAS — the original re-implementation target — is only **~1 % of the chain** (though it is ~70–74 %
of `get_das` alone). Two adjacent issues surfaced: (1) `get_das`/`get_individual_das` never forwarded
the ball carrier to `accessible-space`, so `respect_offside` (the DAS default) emitted a per-call
`player_in_possession_col` warning that flooded serverless logs and risked mis-flagging the passer
as offside; (2) all-dead-ball link-restricted DAS subsets surfaced a generic `accessible-space`
`ValueError` rather than silly-kicks' clear dead-ball message.

The forcing function is pipeline throughput on a Databricks-serverless `applyInPandas` venue
(1 GB UDF memory cap). The strategy was **cheap-wins-first, GPU gated on re-measurement**.

## Decision

Ship a **vectorized, scipy-faithful weighted-Gaussian KDE** (`_kde_density_vectorized`, reusing
scipy's exact Scott bandwidth + weighted covariance + Cholesky whitening via `scipy.linalg`) as the
**default** `predict_density` path, while **retaining the `scipy.stats.gaussian_kde` path as a
selectable `_reference` oracle** via a `kde_backend="scipy" | "vectorized"` argument. Make
`derive_team_in_possession` **preserve `ball_carrier_player_id`** and have `get_das`/
`get_individual_das` **forward it as `player_in_possession_col`** (default = that column) for correct
offside masking, with silly-kicks owning a one-time no-carrier warning and a clear dead-ball
message. **Defer the GPU multi-backend engine** (numba/JAX/CuPy) to a gated Phase 1, conditional on
the lakehouse serverless re-profile still showing ghost-GK as the bottleneck.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Re-implement DAS (numba/GPU) — original ask | Targets a 70 % share of `get_das` | DAS is ~1 % of the AC-1 chain → Amdahl caps whole-chain at ~1 % | Wrong lever for the pipeline goal (validated by full-chain profiling) |
| B. Replace ghost-GK KDE with `predict_mean` (regressor mean) | Orders of magnitude faster | Consumed features are `mode`/`spread` (density-derived); mean lacks them; unavailable on `load()`-ed models | Changes the model's consumed semantics → a model change, not a perf fix |
| C. Drop scipy `gaussian_kde` entirely | Less code | Loses the validation oracle + runtime fallback | Keep scipy as selectable `_reference` (mirrors accessible-space-as-oracle) |
| D. (chosen) Vectorized scipy-faithful KDE default + scipy `_reference`; carrier forwarding; GPU deferred & gated | No new deps; golden-master faithful; correct offside; honest gate | CPU win is modest (~1.18× on the bundled model — eval-bound on ~36k-point subsets) | — |

## Consequences

### Positive

- ghost-GK `predict_density` keeps a per-sample loop but removes scipy's per-call object overhead and
  vectorizes the leaf-match; the scipy `_reference` enables a model-traveling parity test that
  auto-revalidates on retrain.
- DAS offside is now correct (passer excluded) and the per-call log flood is gone. **(Amended 4.4.1:
  the original "value-neutral / zero AS/DAS change" claim here was wrong — see the Amendment above.
  DAS values DID change in 4.2.0; it is a correctness fix, validated post-hoc, and downstream goldens
  frozen under ≤4.1.1 must be re-baselined.)**
- `ball_carrier_player_id` is now a first-class frame column for **all** downstream consumers, not
  just DAS.
- Behaviour-preserving: golden masters (continuous grid `rtol≈1e-7`+atol+NaN-mask; discrete mode
  exact argmax), the elastic-sync de-`iloc`, and the dead-ball guard change no consumed values.

### Negative

- The pure-CPU KDE win is **~1.18×** on the bundled `"default"` model (each sample's nonzero
  leaf-subset is ~36k points → eval-bound, not overhead-bound). A material win needs the GPU engine
  (Phase 1) — this ADR does not deliver it; it gates it on the lakehouse serverless re-profile.
- A second KDE implementation (`_reference` scipy + vectorized) must be kept in parity — mitigated by
  the model-traveling parity test.
- `derive_team_in_possession` now emits an extra column (additive; low Hyrum risk, but a schema
  surface widening).

### Neutral

- `train_block` default 1024 (≈150 MB/block transient) is conservative for the 1 GB UDF cap; the
  serverless venue runs the small "default" model where chunking rarely binds. Lakehouse owns the
  UDF-memory verification + the consumer-correctness diff (cross-repo acceptance).
- A model-side lever (truncating the ~36k nonzero-weight subset) would give a big CPU win but changes
  ghost_gk values → a separate model change requiring lakehouse sign-off, explicitly out of scope.

## Related

- **Specs:** `docs/superpowers/specs/2026-06-01-action-context-hotpath-acceleration-design.md`
- **Plans:** `docs/superpowers/plans/2026-06-01-action-context-hotpath-acceleration-phase0.md`
- **ADRs:** extends ADR-008 (pitch-control multi-backend precedent); relates to ADR-003 (`nan_safe`
  DAS degradation) and the accessible-space oracle pattern.
- **External references:** Bischofberger & Baca 2026 (DAS / accessible-space); scipy `stats.gaussian_kde`.

## Notes

Phase-0 measurement (local, RTX-box, py3.10/scipy1.15/numpy2.2, bundled "default" model, warm, 16
samples): scipy 4541 ms/sample → vectorized 3832 ms/sample (~1.18×). cProfile: 560 `cho_solve` calls
= 16 samples × ~35 train-blocks → ~36k-point nonzero subsets per sample → eval-bound. Decision gate:
pure-numpy did not collapse the cost → GPU engine (Phase 1, §4 of the spec) justified iff the
serverless re-profile confirms ghost-GK remains dominant. Task 11 (broader `.iloc`/`iterrows` sweep)
deferred — the committed 10-action fixture did not reproduce the lakehouse's ~14 % / 2.1 M-`_ixs`
pathology; attribute on the serverless-scale profile first (YAGNI).
