# ADR-036: xT-GK v2 possession-value surface `V(z,p)` (`silly_kicks/xtgk/`)

| Field | Value |
|---|---|
| **Date** | 2026-07-09 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen; Eyestone (xT-GK v2 collaboration); silly-kicks + analysis sessions |

## Context

xT-GK v1 (`tracking/_xt_gk.py`, ADR-024) is **near-constant across keepers** — a formulation-bound degeneracy confirmed on two cohorts (WC2022/GS + Real-Madrid/SkillCorner). Its value surface is the raw destination-only Singh xT, which is ≈0.0085 and flat in the keeper zone, so `V(s') − V(s)` carries almost no signal. Eyestone delivered xT-GK v2, which replaces the v1 additive composite with a single expectation `ρ·[V(s′)−V(s)] − (1−ρ)·[V(s)+κ·V_opp]` whose highest-leverage change is a new value function `V(z,p)` — the expected xG the possessing team generates over the remainder of the possession, given the ball in 16×12 zone `z` under pressure level `p∈{1,2,3}`. This is v2 **sub-project 1 of 5**; it gates the other four via a make-or-break deep-zone-gradient test on real data.

`V(z,p)`'s reward is expected xG, so it needs a calibrated **per-shot xG on every shot** — which silly-kicks deliberately does not ship (it values threat via VAEP/xthreat). xG therefore enters as an **injected `xg_column`**, sourced from the lakehouse `fct_shot_xg` mart. Both fit cohorts are provider-tier/restricted, so the real-data gate is owner-run, not CI.

## Decision

Ship a new hexagonal `silly_kicks/xtgk/` package: a `PossessionValue` Protocol with two adapters — `MarkovPossessionValue` (production: pressure-stratified value iteration reusing `xthreat`'s tested solver, with an xG-calibrated *first-shot* immediate reward and a goal-kick-inclusive move-set) and `EmpiricalPossessionValue` (a model-free `first_shot` cross-check, not shipped) — plus a pre-registered occupied-cell deep-zone gate. xG is an injected column; silly-kicks ships **no** xG model. Nothing wires into a default VAEP xfn list (opt-in), and no `xthreat` source is modified.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Reuse `scores(window="possession", xg_column=)` as the reward | zero new code | goal-gated (`labels.py:282`) → reproduces the v1 flat deep zone | Factually wrong reward — the degeneracy it must escape |
| B. Widen `xthreat`'s public builders with a `move_actions=` param | DRY | Singh needs all-results, KDE needs success-filtered — one param can't serve both; widens tested API | Two populations; regression surface on classic xT |
| C. (chosen) xtgk-local builders reusing `xthreat` low-level seams; injected `xg_column`; first-shot Markov reward | correct, no `xthreat` edits, parity-gated, decomposable | replicates ~40 LOC of counts | — |

## Consequences

### Positive

- An honest, pressure-conditioned value surface whose deep zone carries a real gradient (validated by the go/no-go gate), unblocking xT-GK v2 sub-projects 2–5.
- `MarkovPossessionValue` reuses `xthreat.value_iteration` verbatim; classic xT stays byte-identical (parity-gated over random cohorts + the frozen oracle).
- `delta_v` exposes a two-factor Shapley split for the metric's decomposition without reaching into estimator internals.
- Pickle-free `save`/`load` (npz + JSON + SHA256) matching the house artifact convention.

### Negative

- `V` is fittable/validatable only where a calibrated per-shot xG exists (the two cohorts) — a deliberate boundary (silly-kicks ships no xG model).
- Two "xt-gk" homes coexist (`tracking/_xt_gk.py` v1 frozen; `xtgk/` v2) until the lakehouse migrates; documented in CLAUDE.md.

### Neutral

- Phase-2 canonical promotion of `V` into the metric, `ρ`, `V_opp`, and the lakehouse migration are separate sub-projects.
- Gate numbers (deep-cell set, effect floor, `N_min`, cross-check tolerance, direction) are owner/Eyestone-locked before fitting; `GateConfig` carries them.

## Amendment (2026-07-09, 4.41.0/PR-S108) — Q3 resolved + G8 frame-aware null-pressure

Two owner-only blockers resolved against the live backend; both refined a design detail (spec rev 4 §5/§6).

- **Q3 — injected `xg_column` = `soccer_analytics.dev_gold.fct_shot_xg.xg`** (calibrated pre-shot, grain `(match_key, action_id)`; a *separate* table from `fct_shot_psxg`, so no post-shot leakage). 100% non-null on both cohorts (WC 1473, RM 2596). **Certification caveat: `ood_flag` = 0 for gradientsports (certified) but 100% for skillcorner (all RM shots OOD).** Per-cohort surfaces (G3) contain it — WC reward clean, RM reward populated-but-uncertified (RM verdict provisional). `MarkovPossessionValue.fit(reward_provenance=)` records a caller-supplied OOD-rate/CI summary (the library never interprets `ood_flag`/CI — no xG model shipped); the owner-run emits `ood_rate_by_source` pre-gate. silly-kicks still ships no xG model.
- **G8 — frame-aware null-pressure rule** (corrects the blanket "fail-loud on missing pressure" in the original §5): distinguish by tracking-frame presence — **frame absent** (genuine gap) → drop/fail-loud (`PressureLevels.apply` backstop); **frame present + `pressure_on_actor` null** (no opponent in the pressure region — a genuinely unpressured restart) → **zero → LOW tercile, keep**. Live: 595/595 GS null-pressure goal-kicks have intact frames; a blanket drop would silently lose 60% of WC goal-kicks (the certified cohort's headline population). Implemented as the pure `coalesce_frame_present_null_pressure(pressure, frame_present)` applied in the owner-run data-prep *before* fit; the unpressured-restart count is reported per cohort (`frame_present_null_pressure_count`) — it is signal, not loss.

No production/xfn change; additive. Phase 11 remains wired-but-not-run, blocked on Q4 (the locked gate numbers) only.

## Amendment (2026-07-09, PR-S109) — v2 completion: gate run + SP2–SP5 in one release

Owner directive: complete v2 in ONE release (build SP2–SP5 **and** wire/run the gate together, rather than gating SP2–5 on the gate result — the components are independently valid). Spec: `docs/superpowers/specs/2026-07-09-xtgk-v2-completion-handoff.md`.

**The metric.** `xT-GK(s,a) = ρ·[V(s′) − V(s)] − (1 − ρ)·[V(s) + κ·V_opp(s,a)]`, assembled by `xtgk._metric.compute_xt_gk_v2` depending only on three ports: `PossessionValue` (SP1), `RetentionModel` (SP3), `TurnoverCost` (SP2). Injection discipline throughout (mirrors `compute_xt_gk`'s `xt=`/`completion=`); V/ρ/V_opp each swappable.

**Gate (Part 1).** `GateConfig` gains a **gate-enforced** `relative_effect_floor` (primary acceptance `|v_lo−v_hi|/mean ≥ 0.25`, alongside the absolute `effect_floor`). **Zone-conditional terciles built for real** (`PressureLevels.mode="zone_conditional"`: per-band cutpoints, deep band = grid columns xi∈{0,1}; `to_meta`/`from_meta` keep the global on-disk form byte-identical, absent `pressure_mode`⇒global back-compat). Pre-registered **three-rung ladder** `run_gate_with_ladder` (global → zone-conditional → STOP; the winning rung is reported, so a rung-2 pass reads as *deep-relative*). Locked Q4 numbers: `effect_floor=0.005`, `relative_effect_floor=0.25`, `n_min=30`, `min_occupied_cells=2`, `expected_direction="decreasing"`. RM (SkillCorner) is INCLUDED as a PROVISIONAL second read (100% OOD), not dropped (owner decision).

**SP2 `V_opp`.** `MirroredTurnoverCost` (production) = `V(mirror_zone(z), policy(p))` on the already-fit V — zero new fitting; `mirror_zone` = 180° point reflection; default pressure-transfer `p_opp = p` (injectable). `EmpiricalTurnoverValue` (cross-check, not shipped) credits the opponent's first post-turnover shot within a bounded window.

**SP3 `ρ`.** `GkRetentionModel` (logistic, pure-numpy serve, JSON+SHA256, per-provider variants) mirrors `GkCompletionModel`. New `retains(actions, *, window_seconds=10.0)` label — retain iff within the window the actor's team keeps the ball (no opponent possession boundary) OR shoots; **truncated-window→NaN** (excluded from training, not falsely-retained). **Marts-native `extract_retention_features`** (8 features: pass geometry + `pressure_on_actor__bekkers_pi`, the pinned pressure measure) sourced from the gold action marts (`fct_action_values` + `fct_action_context`), **NOT tracking frames (deprecated as an active source, owner directive 2026-07-10)** — the frames-only receiver-density feature is dropped and the GK-distribution domain is **goal-kicks** (`gk_was_distributing` unpopulated on GS/SC — lakehouse handoff F1; `gk_role` encodes defensive roles; acting-GK-pass resolution is frames-based → out of scope). **Calibration gate stricter than completion:** every shipped variant must pass `ece ≤ 0.10 AND |reliability_slope − 1| ≤ 0.25` (`silly_kicks/_calibration_metrics.py`, extracted from the completion trainer to an extra-free module so SP3/SP5 don't pull the optuna `[calibration]` extra). **`default` weights BUNDLED** under `xtgk/_retention_weights/default/` via the marts-native `scripts/train_gk_retention.py` + `_loader_databricks.load_retention_cohort`: GS 64-match / 396 goal-kicks, OOF **AUC 0.776, ECE 0.090, slope 1.01** (PASS). **The SkillCorner variant is NOT shipped** — under `bekkers_pi` it is near-chance (AUC 0.54) and fails the calibration gate (slope 0.63, under-calibrated), so `_PROVIDER_VARIANT={}` and every provider falls back to `default` (mirrors SC completion being base-rate-served). **ADR-011 does NOT govern this "trained-light" class** (per ADR-024's completion precedent).

**SP4 decomposition (four coherent additive terms).** `xt_gk_v2 = ρ·ΔV_position + ρ·ΔV_pressure(=PEV) − (1−ρ)·V(s) − (1−ρ)·κ·V_opp`, summing exactly to the metric. Columns namespaced **`xt_gk_v2_*`** (`_position`/`_pev`/`_retention_loss`/`_dzv`/`xt_gk_v2`) — v1's frozen `xt_gk_pev/rav/dzv` (lakehouse-materialized, GK-Analytics-UI-read) must NOT be reused (Hyrum). **PEV is 0 by construction while `p′ = p`** (dormant pending receiver-pressure `q`, Jeff §11). RAV = the total (a label), NOT a separate bar.

**SP5 validation.** `scripts/validate_xtgk_v2.py` (owner-run): out-of-sample construct validity (v2 vs raw completion / destination-only V / v1 composite), cross-competition transfer, ρ calibration; WC2018/Neuer repro stubbed (needs Jeff's old data).

**v1 end-state (M5).** v1 `tracking/_xt_gk.py` is FROZEN alongside v2; removed no earlier than one release after the lakehouse migrates its `xt_gk*` columns. No v2↔v1 imports; `xthreat` + v1 byte-stability regression-gated.

**Open sub-questions flagged to Eyestone (non-blocking):** (1) whether zone-conditional should be the *primary* deep-gate mode (default: fallback rung); (2) confirmation of the PEV/DZV/RAV ↔ four-term acronym mapping. Neither blocks the release.

In **no** default xfn list (opt-in). CI covers the pure/synthetic surfaces; the gate run + weight bundling execute against Databricks gold (owner-run). Additive — no forced VAEP retrain.

## Gate result (owner-run 2026-07-10) — the make-or-break verdict

Ran `scripts/validate_xtgk_possession_value.py` against Databricks gold via the new
`_loader_databricks.load_xtgk_cohort` (`bronze.spadl_actions ⋈ dim_matches ⋈ dev_gold.fct_action_context`
[pressure + frame-present] `⋈ dev_gold.fct_shot_xg` [calibrated xG]; action_id join coord-exact). Report:
`docs/research/xtgk_possession_value/{gate.json, GATE_FINDINGS.md}`.

**Pressure pinned to `bekkers_pi` (§5 Q3).** The initial `andrienko_oval` run STOPped (0 occupied deep
cells) because **52% of actions had pressure exactly 0**, degenerating the terciles. The lakehouse 3-method
audit (handoff F2) showed the zero-mass is a **method artifact, not missing data** — exact-zero rate andrienko
46.9% / link_zones 79.5% / **bekkers_pi 4.7%** — so the gate is pinned to `bekkers_pi` (non-degenerate tail).

**Verdict (bekkers_pi) — GO-leaning, not a STOP:**
- **WC2022 (authorising, certified): fail** — but a **real decreasing monotone deep gradient** (8 cells,
  relative effect **0.86**, V_lo≈0.0045→V_hi≈0.0018). The fails are the *absolute* effect (0.0027 < the
  pre-registered 0.005 floor) and the empirical cross-check divergence.
- **RM (provisional, 100% OOD): fail (cross-check only)** — same real decreasing gradient (17 cells, relative
  **1.05**, effect 0.0089 which *clears* the absolute floor); fails only the cross-check.

The make-or-break signal ("keepers separable by pressure") **is present** on both cohorts. The two fails are
Eyestone-review items, not kills: (1) the 0.005 **absolute floor** looks too high vs the intrinsic deep-zone
first-shot-xG magnitude (≈0.003–0.005) — the scale-free relative effect (0.86–1.05) is the meaningful read;
(2) the model-free **cross-check divergence** (high-variance estimator vs. transition structure) warrants an
audit. **Do NOT lower the pre-registered floor post-hoc.** Per the owner build-ahead directive SP2–SP5 shipped
regardless.

**Two real bugs surfaced + fixed by the run** (synthetic fixtures had missed them): (1) `prepare_cohort` now
DROPS the residual frame-absent tracking-gap nulls (the §5 backstop's intent — `PressureLevels.apply` was
raising on them); (2) a NaN-safe `flat_zones` helper at the four zone-binning seams (real cohorts carry NaN
start/end coords; `xthreat._get_flat_indexes` int-cast would crash).

## Amendment (2026-07-10, 4.43.0/PR-S110) — public `gk_distribution_mask` + ρ loader `is_gk_distribution`

A lakehouse export request: expose the private GK-distribution domain logic (`_gk_distribution_mask`,
`tracking/_xt_gk.py`) as a public, stable, **frame-optional** API so the lakehouse pins one function to
materialize a per-action `is_gk_distribution` column rather than reimplementing it.

- **New public `tracking.gk_distribution_mask(actions, frames=None, *, resolve_gk="robust")`** (in the
  non-frozen `_gk_resolve.py`, beside `acting_gk_from_frames`). `True` for any goal-kick (actor-independent)
  OR a pass/throw-in by the acting GK. Returns a bool `pd.Series` aligned to `actions.index`. `frames=None`
  → goal-kicks-only (the GK open-play-pass term is undetectable without frames).
- **`resolve_gk` lever.** `"native"` = the frozen global-`frames[is_goalkeeper]` (game,team,player)
  set-membership. `"robust"` (default) = per-action time-accurate resolution via `acting_gk_from_frames`
  (linked-frame + roster-identity fallback). **For the GK-pass term `robust ⊆ native`** — it *tightens*
  stale/substituted keepers (native over-includes a substituted-off keeper whose player is still in the
  global set), it never broadens. It is also the resolver the lakehouse pins for its goal-kick-taker
  override, so the domain stays consistent with that override (the reason it is the default). The
  `~40%-undetected-keeper` figure motivates that *taker override*, NOT the mask's GK-pass term.
- **v1 freeze preserved (M5).** The frozen `_gk_distribution_mask` becomes a **byte-identical shim** over
  `gk_distribution_mask(..., resolve_gk="native")` (golden-gated on a fixture containing a native GK
  open-play pass, so the set-membership branch is actually covered). Its three consumers (v1 compute,
  completion, features) are unaffected. No import cycle (`_gk_resolve` never imports `_xt_gk`).
- **Lakehouse contract:** `fct_action_context.is_gk_distribution` (per-action bool) ≡
  `gk_distribution_mask(actions, frames, resolve_gk="robust")`.
- **ρ retention loader/trainer** stop reading the shot-scoped `gk_was_distributing` (a misuse of the
  `add_pre_shot_gk_context` shot feature — that feature is unchanged) and adopt a **self-adapting**,
  **NULL-coalesced** `is_gk_distribution` domain: `goal-kicks ∪ COALESCE(is_gk_distribution, FALSE)`. The
  loader probes `fct_action_context` with a **catalog-qualified** existence check
  (`soccer_analytics.information_schema…`, never a bare `information_schema`) and includes the column only
  when present — absent/NULL → goal-kicks-only (today's behaviour; bundled `default` ρ model unchanged). The
  self-adapting probe is **transitional** (collapse to an unconditional read once the column is permanently
  materialized, alongside the deferred ρ retrain on the broadened domain).
- **Additive** — no `xt_gk`/VAEP value change, no retrain. C4 count stays 28 (a resolver, not an
  action-coupled aggregator).
- **Discovered, out of scope:** `acting_gk_from_frames` compares action-team vs frame-team ids with a raw
  `==` (`_gk_resolve.py`), dtype-fragile if those ever differ (same-provider dtypes match in practice) —
  tracked in `TODO.md`, not fixed here (Chesterton's Fence).

## Amendment (2026-07-11, 4.44.0/PR-S111) — ρ retrain on the broadened domain + loader collapse + resolver dtype fix

With lakehouse F1 live (`fct_action_context.is_gk_distribution` materialized), the two PR-S110 deferred
follow-ups + one deferred hardening land together.

- **ρ retrain (Part A).** The ρ domain broadens from goal-kicks-only to the full GK-distribution set
  (goal-kicks ∪ acting-GK open-play passes — the loader/trainer already OR `is_gk_distribution`). Re-bundled,
  calibration-gated (auditable metrics manifest, ADR-009):
  - `default` (gradientsports): **AUC 0.781 / ECE 0.031 / slope 0.998**, n=2923 (64 matches) — improved from
    goal-kicks-only 0.776 / 0.090 (n=396). The thin-ECE-headroom risk did not materialize.
  - **SkillCorner variant SHIPS** (reversing the PR-S109 no-bundle): the broadened domain (5477 rows, incl.
    GK open-play passes) makes it viable — **AUC 0.650 / ECE 0.020 / slope 0.923, GATE=PASS** (vs old
    near-chance 0.54 on 1189 goal-kicks). `_PROVIDER_VARIANT = {"skillcorner": "skillcorner"}`; other providers
    fall back to `default`. This is data-driven, held to the SAME gate — no lowered bar.
- **F1 CI calibration guard.** `tests/xtgk/test_retention_bundle_calibration.py` certifies every bundled
  variant's recorded `metrics.json` clears the canonical `_ECE_MAX`/`_SLOPE_TOL` (imported, not read from the
  file) + the recorded thresholds match them — a hand-loosened `metrics.json` can't self-certify. The
  bundle-only-if-passes discipline is now a CI-enforced invariant guarding all future re-bundles.
- **Gate scope (unchanged).** The retrained ρ moves the **metric-level construct-validity** (`compute_xt_gk_v2`),
  NOT the deep-zone make-or-break gate (which reads V, not ρ — already GO-leaning, settled). Owner re-runs
  `validate_xtgk_v2.py` with the production ρ (CI uses `_ConstRho`); the deep-zone Q4 numbers stay LOCKED.
- **Loader collapse (Part B).** The transitional self-adapting `is_gk_distribution` probe is retired —
  it's a HARD dependency now (unconditional `SELECT c.is_gk_distribution`); NULLs `fillna(False)` (warning-free).
  MODEL_CARD pressure doc-bug fixed (`andrienko_oval` → the actually-used `bekkers_pi`).
- **Resolver dtype fix (Part C, ADR-019).** Investigation **reframed the PR-S110 premise**: `acting_gk_from_frames`
  is fallback-protected (NOT fragile); the real defect was `defending_gk_from_frames` returning the acting team's
  OWN keeper (not the opponent) on a cross-dtype team mismatch. Fix = per-branch `ids_equal` (acting) /
  `ids_differ` (defending) at the shared `_gk_from_frames_linked` predicate — `ids_differ`'s NA→not-differ
  preserves the unresolved→NaN semantics AND canonicalizes cross-dtype so defending picks the true opponent.
  Byte-identical on matched/NA paths (four resolver gates + a non-vacuous NaN-branch anchor).
- **Version 4.44.0.** `compute_xt_gk_v2` serve output changes → xT-GK v2 retrain trigger (opt-in; not in any
  default xfn list → NOT a forced VAEP retrain). Lakehouse re-materializes xt_gk_v2 on the 4.44.0 pin.
  **Nullable heads-up:** F1 shipped `is_gk_distribution` nullable (899 GS / 557 SC NULLs); silly-kicks is
  defended (`fillna(False)`), relayed for the lakehouse to decide on non-nullable enforcement. C4 count stays 28.

## Amendment (2026-07-11, 4.45.0/PR-S112) — faithful V_opp + the honest construct-validity verdict

The release that closes the v2 validation loop. It replaces the mirror geometric turnover proxy with the
**faithful** observed-post-turnover cost (Jeff §2.3) and runs the full out-of-sample validation. Governed by
the §3 honest-reporting guardrail: the a-priori params were fixed before fitting; the numbers are reported as
they landed and were **NOT retuned to force a pass**.

- **Faithful `EmpiricalTurnoverValue` (`_turnover.py`).** The production `MirroredTurnoverCost` estimated
  `V_opp(z) = V(mirror_zone(z))` — a geometric proxy that, on real data, over-stated deep opponent threat
  ~10–50× at real support (GS zone 96: mirror 0.256 vs faithful 0.005). The faithful adapter estimates the
  opponent's actual first-shot xG after a turnover, indexed origin-zone × pressure, with (a) **possession-bound
  scope** (`window_seconds=None`; scan to the match boundary — a `game_id` fail-loud guard because the scope
  can't be computed without it), and (b) **support-gated hierarchical bin-widening** (native cell → coarse
  `coarsen×coarsen` block → global-per-pressure; `min_support=30` = the gate `n_min`) so a 1–2-turnover deep
  cell is not a noise estimate. `resolution_level(p)` + module `surface_divergence` audit the two adapters.
  `_metric.py` is unchanged — `turnover_cost` stays injected via the port, so the faithful adapter is a better
  recommended injection, not a forced default change.

- **What the faithful V_opp fixed (genuine correction).** Component decomposition on the metric: the deep
  `dzv` `|mean|` share fell from ~87–89% (mirror) to **29%**; `ρ·ΔV` (position) rose from ~8% to **36–42%**.
  The R1 deep-cell disentanglement confirms this is a real mirror over-statement, not a window artifact
  (possession-bound ≪ mirror at real support; 10s ≪ possession-bound = the artifact, not the finding).

- **What it did NOT fix — the honest verdict.** Even with the faithful V_opp, out-of-sample on real Databricks
  gold (`docs/research/xtgk_v2_construct_validity/`):
  - **Outcome-AUC lift** over `max(raw_completion, destination_xt, v1_stored)`: **GS −0.139, SC −0.072** — v2
    does not beat the baselines. (Head-to-head vs v1 on v1-covered rows: GS +0.121, SC −0.072.)
  - **Keeper discrimination** (action-level ICC grouped by resolved `player_key`, R2 — not the degenerate
    CV-on-means): **v2 −0.002 (GS) / 0.011 (SC)** vs **v1 0.019 / 0.018** — both near-zero; v2 is still
    keeper-flat. The R2 ICC vindicated itself: CV had read v2 24% ≫ v1 6%, a near-zero-mean artifact.
  - **Verdict: xT-GK v2 is not construct-validated** by either the outcome-AUC or the keeper-discrimination
    lens. Reported as-is for the Eyestone/Jeff conversation. Open interpretation forks flagged, not patched:
    the V reward uses `E[first-shot xG]` vs Jeff §2.1's *remainder-of-possession* threat, and PEV is dormant
    (`p′=p`; receiver-pressure `q` deferred) — both are candidate explanations for the weak signal, but
    re-implementing V or wiring `q` is out of scope for this release (a separate decision if the forks matter).
    The faithful adapter ships regardless because it is the correct, un-swamped turnover cost.
  - The W6 κ sweep (κ∈{1,1.5,2}, reported for Jeff, κ=1 the a-priori headline) confirmed a larger κ only adds
    more turnover drag (GS AUC 0.484→0.477) — κ=1 was not tuned to this.

- **Data contract (`CLAUDE.md`).** GK-domain consumers use the **resolved `player_key`**, never raw
  `player_id` (NULL for goal-kicks by SPADL design — the reason `acting_gk_from_frames` exists). The xtgk
  cohort loader sources it from the gold `fct_action_context`, guarded by `test_player_key_contract.py`.

- **Version 4.45.0 (MINOR).** Library change is the faithful `EmpiricalTurnoverValue` (opt-in adapter; not in
  any default xfn list → not a forced VAEP retrain). A consumer adopting the faithful injection re-materializes
  `xt_gk_v2_*`. The deep-zone make-or-break gate (reads V, not V_opp) is untouched and stays GO-leaning. C4
  count stays 28.

---

## Amendment (2026-07-12, 4.46.0/PR-S113) — resolved GK-distribution geometry

### The defect

`xtgk._possession_value.flat_zones` maps a NaN coordinate to `(0.0, 0.0)` — i.e. **flat zone 176**,
the own-corner cell. That is a **FIT-PATH contract, not a general one**: it is safe only because no
NaN-coord row ever reaches a fitted surface (`_moves.py`, `_xg_reward.py`, `_markov.py`,
`_empirical.py`, `_turnover.py` drop them *before* calling in; `_markov.py:65`, `_empirical.py:83`
and `_diagnostics.py:123` pass NaN rows *through* to assign pressure terciles and drop them
immediately after).

It was **false at the single SCORING seam**, `_metric.py`, which dropped nothing — so a NaN-origin
goal-kick was scored as a **real number at a location it never had**.

Compounding it, `load_xtgk_cohort` read the **raw** `bronze.spadl_actions.start_x`, while the
**resolved** keeper origins had been materialised into `fct_action_context.xt_gk_origin_x/_y` by
PR-S101 (4.36.0) — in the very table the loader already `LEFT JOIN`s. It never `SELECT`ed them. The
v1 comparator in the 4.45.0 head-to-head (`c.xt_gk`) *did* use resolved origins, so **that
comparison was never apples-to-apples.**

### Measured blast radius (live gold)

| provider | domain | `native` | `resolved_origin` (was fabricated) | `unresolved` (now honest NaN) |
|---|---|---|---|---|
| gradientsports | 3874 | 2928 | **530** | **416** |
| skillcorner | 5487 | 4516 | **971** | 0 |

**946 GS rows (24.4%, incl. 60.2% of its goal-kicks)** were scored at zone 176. SkillCorner's 971
goal-kicks carried a *present-and-wrong* origin — the broadcast **ball** detection, not the keeper
(ADR-024 / PR-S104) — so a `fillna` **coalesce** would have fixed GS and silently missed SkillCorner.
The rule is **OVERRIDE, not coalesce**.

### Decision

1. **`apply_resolved_gk_geometry`** (new public, pure, no I/O): overrides GK-distribution coords from
   `xt_gk_origin_x/_y` + `xt_gk_dest_x/_y` and stamps a 7-value `gk_geometry_source` provenance
   column. `unresolved` **wins** whenever any coordinate is still non-finite (the stamp answers "will
   this row score?"). Resolved columns absent → warn + no-op + **`unattested`**, never `native`
   (stamping `native` would suppress the metric's warn-once while origins were still raw). Missing
   `domain_column` → **`ValueError`** (treating every row as in-domain would overwrite open-play
   coords with keeper geometry).
2. **`compute_xt_gk_v2`** gains: a **NaN guard** (non-finite-coord rows emit NaN across all five
   outputs and never enter the loop — no zone is ever fabricated); ρ is **not scored** on those rows,
   closing `_retention.py:81`'s silent mean-imputation *without* changing `predict_proba`; a
   **coordinate-coherence check** that recomputes the coordinate-derived ρ features from `actions`
   and raises on divergence; and a **warn-once attestation**.
3. **Both Databricks loaders** apply the helper, so all four consumers inherit it.
4. **ρ retrained** on the corrected cohort. Both variants PASS the calibration gate: `default`
   2923→**3451** rows, AUC 0.781→**0.798**; `skillcorner` 5477 rows (identical — only the *geometry*
   changed), AUC 0.650→**0.662**. `_PROVIDER_VARIANT` unchanged.

**The coherence check compares COORDINATES, not provenance** — so it catches resolved-actions +
raw-features, its mirror (raw actions + resolved features), *and* mart-vintage divergence, with no
case table. It must span the **origin**-derived features (`length`/`forwardness`/`dy_abs`): the dest
override is a **measured no-op** on both cohorts, so a `dest_x`-only check would miss every real
divergence. The stamp's remaining job is the one thing coordinates can never reveal — that resolution
was never *attempted* (raw coordinates are perfectly self-consistent).

### ADR-025 interplay (this does NOT breach the never-mutate-canonical fence)

ADR-025's contract is: never mutate canonical `start_x`/`end_x`; emit `enriched_*` side-band columns,
with canonical promotion an explicitly deferred Phase 2. `apply_resolved_gk_geometry` **overrides
canonical columns on a copy**, and that is compatible:

- ADR-025's fence protects the **canonical persisted** coordinates — what converters emit and what the
  lakehouse writes to its marts. This helper produces a **transient scoring-time view**: the overridden
  frame is handed to `compute_xt_gk_v2` and discarded. **`start_x` is never written back to any mart.**
- The side-band idiom was **rejected** because it would force `compute_xt_gk_v2` to grow
  `origin_columns=`/`dest_columns=` parameters, pushing a data-**provenance policy** into the metric
  **engine**. Policy stays at the edge; the engine stays provenance-free and reads exactly
  `start_x`/`end_x` (cf. the geometry tripwire kept at the `add_restart_coordinates` edge).

### The deep-zone gate is NOT re-run

Every fit seam drops NaN coords, so the fitted `V` surface, its support counts,
`EmpiricalPossessionValue` and `EmpiricalTurnoverValue` are all clean. The GO-leaning gate verdict
**stands**. This is asserted by a **regression test**, not prose:
`tests/xtgk/test_deep_zone_gate_nan_invariance.py` — which also carries a **non-vacuity
meta-assertion** (its first draft compared an all-zero surface to an all-zero surface and would have
"passed" while proving nothing).

**Fourth consumer.** `scripts/validate_xtgk_possession_value.py` (the gate script) also imports
`load_xtgk_cohort`, so the in-loader override changes **its** input too. Harmless this cycle — the fit
seams drop NaN rows either way — but **a future gate re-run would fit `V` on different coordinates
than the 4.42.0 run did**, so its numbers would not be directly comparable to the recorded GO-leaning
result.

### Construct-validity re-run — honest and mixed

Two legs (pre-fix ρ / retrained ρ), everything else frozen. Full tables in
`docs/research/xtgk_v2_construct_validity/README.md`.

| lens | provider | 4.45.0 (raw) | 4.46.0 (leg 2) |
|---|---|---|---|
| outcome-AUC lift | gradientsports | −0.1387 | **−0.1474** |
| outcome-AUC lift | skillcorner | −0.0720 | **−0.0268** |
| keeper ICC (v2) | gradientsports | **−0.0020** | **+0.0256** (v1: 0.0193) |
| keeper ICC (v2) | skillcorner | 0.0109 | **0.0147** (v1: 0.0176) |

**The "keeper-flat" leg of the 4.45.0 verdict does not survive.** GS's v2 ICC went from −0.0020
(worse than nothing) to +0.0256 and now **exceeds v1** for the first time — exactly the direction
predicted before the run, since a fabricated origin is **keeper-independent** and therefore compresses
between-keeper variance toward zero.

**The outcome-AUC leg stands.** v2 still loses to simple baselines (on GS, `raw_completion` scores
0.622 while both v1 (0.381) and v2 (0.475) sit below chance on that target). So xT-GK v2 is **still
not construct-validated by outcome-AUC** — but the interpretation-fork decision (V-reward definition
vs dormant PEV) can now be taken on **trustworthy** numbers, and one of its two supporting findings
has been withdrawn.

### Consequences

- **`compute_xt_gk_v2` output changes** (NaN where it previously fabricated; corrected values on the
  resolved rows) and **ρ weights change** → **xT-GK v2 re-materialize trigger**. NOT a forced VAEP
  retrain (v2 is opt-in, in no default xfn list).
- **Cross-repo handoff.** The lakehouse must call `apply_resolved_gk_geometry` before
  `compute_xt_gk_v2`, and **must keep `is_gk_distribution` (or the `gk_geometry_source` stamp) on the
  frame it passes** — the warn-once fires only when a domain column with true rows is present, so a
  pre-filtered slice with the flag dropped would score raw origins in silence. The
  coordinate-coherence check does **not** cover that case (a uniformly-raw pair is self-consistent).
- **`GkRetentionModel.predict_proba` is unchanged** (non-finite → training-mean impute, neutral
  post-standardisation). The metric's upstream mask removes the exposure; documented, not altered.
- **Hyrum.** On pandas 3.0 the `gk_geometry_source` column materialises as `str` dtype, not `object`.
- **Version 4.46.0 (MINOR).** C4 count stays 28 (no new action-coupled aggregator).
