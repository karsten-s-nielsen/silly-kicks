# TF-17: xCrossAttempt — Cross-Attempt Propensity Model (+ causal GK-confounder validation)

**Date:** 2026-06-03
**Status:** Draft — for review
**Layer:** GKDV Layer 2 (TF-17)
**Prereqs (all shipped):** TF-7 pitch control (3.7.0), TF-13 frame GK-ID + TF-14 defensive line (3.4.0), TF-15 GK influence (3.10.0), TF-5 ball carrier (3.5.0), link primitives (PR-S19), TF-16 xShotOccurrence code (4.1.0) + `_geometry` helper.

**Source:** Cao, Y., et al. (2025). *"Framing Causal Questions in Sports Analytics: A Case Study of Crossing in Soccer."* arXiv:2505.11841.

**Review round 3 (resolved 2026-06-03 — PR-S81 session, on the revised spec; all round-2 points verified resolved).** **R2-H1** (HIGH) the surface ablation tests prediction-lift, not TF-19 viability — added a **GK substitution-sensitivity probe** (`|P(cross|actual_GK) − P(cross|shifted_GK)|` distribution = TF-19's actual operation) as a PR-B metric, with a pre-registered **inert-GK contingency** (surface still ships per H-2, but `metrics.json` flags "not TF-19-ready" loudly + gates TF-19 consumption on GK feature-engineering — never shipped silently as "GK novelty") (§1, §1.1, §10). **R2-M1** specified the §9.2 opportunity unit (one continuous wide-area possession-spell, anchored at entry) + dedup rule + `test_opportunity_dedup`. **R2-M2** fixed the stale §9 header (PR-C, private `_causal/`). **R2-L1** the ablation + sensitivity probe run on the **shipped candidate** (paired-test winner), reported in its `metrics.json` (GK signal is provider-variable). **R2-L2** the M-3 xS-label refactor gets a **bit-identical golden-master** test (not just "existing xS tests").

**Review round 2 (resolved 2026-06-03 — PR-S81 session review).** **H-1** the causal harness validates the GK-confounder *hypothesis* (paper-faithful), NOT the shipped surface's GK block — add a **surface-level GK-block ablation** (trained XGBoost held-out PR-AUC/log-loss with vs without the GK group) as the shipped-surface validation (§10 PR-B, §11); de-conflate the two everywhere (§1, §1.1). **H-2** the GK-extended surface **ships regardless** of the causal outcome (a null is valid); **split** the staging — PR-B = weights + surface-ablation + TF-19 wiring; **PR-C** = causal harness + ADR-015 (a one-shot research artifact must not gate a runtime feature's release). **H-3** the GK-beats-placebo-on-real-data check is a **reported metric, not a CI assertion** (a null is a finding, not a regression) — only the known-truth method tests gate CI (§11). **M-1** select domain/horizon by **descriptive** criteria (cross-density threshold; lead-time quantile), not by maximizing the eval metric (selection-on-the-metric bias); any metric-based choice on a held-out fold. **M-2** version collision: PR-S81 also targets 4.10.0 — coordinate, second-merger re-bumps (§14). **M-3** extract a shared `_build_occurrence_labels` consumed by xS + xCross (behaviour-preserving xS refactor, guarded by xS tests). **L-1** start the causal port **private** (`silly_kicks/_causal/`), promote to public `causal/` only when a 2nd consumer lands. **L-2** carrier coverage feeds harness validity (selection-bias caveat), not just a log. **L-3** `home_team_id` unused = inherited xS debt. **L-4** add a test that `faithful` extraction never touches `pitch_control_cache`. **M-4** (PR-S81 code sanity-check — all headline claims verified vs 4.9.0): the §4 wide corridor `y<14|y>54` is `cross_zone`'s dense-zone bound, NOT SkillCorner's converter cross-origin (`_is_cross` = `y<15|y>53`) — reframed as an intentional tighter-zone choice, not "SkillCorner's definition" (§4).

**Review round 1 (resolved 2026-06-03 — cross-session critical review).** The engineering half (propensity surface) was approved as a faithful xS mirror. The research half (causal GK-confounder validation) was under-specified; resolutions now folded in: **C1** placebo/negative-control + pre-registered effect-size + Abadie–Imbens SEs (§9.4); **C2** extract a pure `silly_kicks/causal/matching.py` library port unit-tested against known-truth, script = thin driver (§2, §9, §11, §14); **H1** explicit causal DAG + pre-treatment-measurement argument (§9.1); **H2** propensity surface is **state-anchored**, causal harness is **crosser-anchored** (§4, §9, §13); **H3** drop the §9.2 model-as-propensity cross-check (proxy ≠ paper-faithful confounders); **M1** re-justify the domain from cross-origin evidence + measure by x-band (§4); **M2** pick horizon from data before freezing weights (§Q3); **M3** state ball-vs-carrier anchoring + carrier-coverage log (§5); **M4** default `negative_subsample=None` (§6); nits: drop #7 in the harness, harness reads domain thresholds from metadata, headline estimand = **ATT**, new ADR for the causal-port pattern.

---

## 0. Fact-check ledger (verified against backend 2026-06-03)

| Claim | Status | Evidence |
|---|---|---|
| Paper ATE = 1.6%, ATT = 5.0% | ✅ verified | arXiv:2505.11841 abstract |
| Dataset: Shandong Taishan Luneng (CSL) 2017, 30 matches, **2,225 crossing opportunities / 692 attempted crosses** | ✅ verified | paper HTML §data (TODO said "2,225 crosses" — that is *opportunities*; 692 are actual crosses) |
| Confounder set is **8 engineered variables and contains ZERO goalkeeper variables** | ✅ verified | paper HTML — confounders: score differential, sender→nearest-defender dist, space controlled by sender, sender→nearest-teammate dist, sender→endline dist, off/def player ratio in box, crosser position (FW/MF/DF), ten-minute warning. The GK-confounder gap is the documented literature opening TF-17 fills. |
| Method: 1:1 NN propensity-score matching, **with replacement**, ties allowed, no caliper, logistic propensity, R `Matching` | ✅ verified | paper HTML §methods |
| Paper is **event-level** ("crossing opportunity" rows); silly-kicks is **frame-level** | ✅ noted | deliberate adaptation, mirrors how xS reframed Pipping et al. into a per-frame surface |
| `cross` is a **first-class SPADL actiontype (id 1)**; also `freekick_crossed` (3), `corner_crossed` (5); detected by **every** provider converter | ✅ verified | `spadl/config.py:40`; per-provider cross detection in statsbomb/opta/wyscout/sportec/skillcorner/gradientsports/metrica/kloppy |
| Pining corpus = **81 matches**: 10 skillcorner + 7 idsse (public) + 64 gradientsports (owner) | ✅ verified live | `_list_matches` with owner token (`PINING_FOR_THE_DATA_TOKEN`, len 43, set) |
| HF org `silly-kicks`; only **`ghost-gk-v1` is published**; `xshot-occurrence-v1` bundled-in-wheel but **not on Hub** | ✅ verified live | `HfApi().list_models(author='silly-kicks')` — write access confirmed (member) |
| Cross prevalence per match | ✅ verified live | skillcorner 1886347: **68 open-play crosses** (58 success / 10 fail), 11 freekick_crossed, 2 corner_crossed, vs **23 shots**, over 31,522 frames. **Crosses are ~3× MORE common than shots** (corrects the prior "crosses are rarer" assumption). ≈1,150 open-play crosses in the 17 public matches, ≈5,500 across all 81 → a positive-rich corpus. |

**Correction surfaced (durable):** TF-17 trains its GK confounders on the **actual/resolved** GK frame row (mirror of xS's `_gk_resolve` GK features) — it does **not** consume the ghost-GK *mode*. Therefore the `kde_backend` train/serve-skew guard (ADR-014, amended 4.8.0) binds **TF-19 only** (where the ghost-GK counterfactual substitution happens), **not** TF-17 — exactly as it does not bind TF-16. The TODO/`reference_xs_does_not_consume_ghost_gk_mode` "prospectively TF-17/TF-19" is imprecise: it is TF-19.

---

## 1. Purpose

Predict, from a single tracking-frame snapshot, **the probability that the in-possession team attempts a cross within the next ~k frames (~1 s)**, conditional on the spatial state **including the defending goalkeeper's position**. This is `xCrossAttempt` — the cross analogue of TF-16's `xShotOccurrence` (xS).

xCrossAttempt is **GKDV Layer 2**. TF-19 computes `Δ_cross(action) = P(cross | actual_GK) − P(cross | ghost_GK)` per build-up frame in the final third using this model's per-frame surface as the `P(cross | ...)` term — exactly parallel to how it consumes xS for `P(shot | ...)`.

The **novel research contribution** vs. the source paper: the paper's propensity model deliberately excluded all goalkeeper variables. TF-17 **adds GK-position confounders**, evidenced in three distinct places (§1.1): (1) a **surface ablation** (held-out lift with vs without the GK group — does GK improve prediction?); (2) a **GK substitution-sensitivity probe** (does the shipped surface actually move `P(cross)` when GK position changes? — the TF-19-viability question, R2-H1); (3) a **paper-faithful causal harness** (is GK a real confounder of cross→shot, beyond a placebo? — the scientific gap-closing claim, PR-C). Distinct objects, never conflated.

### 1.1 Scope decomposition (decided)

Two conceptually distinct pieces with **different lifecycles** (a maintained runtime feature vs a one-shot research artifact), staged per ADR-011:

- **Engineering deliverable — the propensity surface** (this spec's core): the per-frame `xCrossAttempt` occurrence model + ADR-005 surfaces, atomic mirror, HPO objective, training CLI, full test suite. State-anchored. This is what TF-19 consumes. It **ships with the GK block regardless of any causal finding** (the block is NaN-tolerant + additive; its ship signal is the **surface-level GK-block ablation**, H-1).
- **Research deliverable — the causal ATT validation harness:** lifts the Cao et al. matching framework on **crosser-anchored** opportunity rows and answers "does adding GK confounders shift the matched ATT vs a placebo?". **It validates the GK-confounder *hypothesis* in a paper-faithful setting (scientific justification for *why* the surface carries a GK block) — it does NOT validate the shipped surface's GK block** (different model, anchoring, unit, and #3 form; H-1). It is a **reported finding, never a ship gate** (a null GK effect is a valid result, not a regression; H-2/H-3).

**Three validations, kept distinct (H-1 / R2-H1):**
- *Prediction-lift* = the **surface GK-block ablation**: trained XGBoost held-out PR-AUC/log-loss **with vs without** the GK feature group (the block is already `test_gk_block_isolatable`). Answers "does the GK block improve prediction?" — a PR-B metric on the **shipped candidate**.
- *TF-19 viability* = the **GK substitution-sensitivity probe** (R2-H1, the actual product question): the distribution of `|P(cross | actual_GK) − P(cross | shifted_GK)|` over the domain — literally TF-19's operation (shift the GK row to a realistic counterfactual / the ghost-GK mode, re-extract, re-predict). **Ablation lift and substitution sensitivity can diverge**: a GK block redundant with defensive-line/box-ratio features can show ~zero held-out lift yet still move `Δ_cross` (carries weight), OR XGBoost can give the GK block ~zero importance → `Δ_cross ≈ 0` → TF-19 emits a degenerate always-~0 feature and xCross's reason-to-exist collapses **silently**. So the ablation does NOT answer the product question; this probe does. PR-B metric on the shipped candidate. **Pre-registered contingency:** if *both* the ablation and this probe show the GK block is inert, TF-17 is **not TF-19-ready** — the surface still ships (H-2: a weak signal isn't a build break), but the inert-GK result is **surfaced loudly in `metrics.json` (not shipped silently as "GK novelty")** and TF-19 consumption is gated on investigating GK feature engineering first.
- *Hypothesis validation* = the **causal harness** (paper-faithful, PR-C). Answers "is GK a real causal confounder of cross→shot?".

**Staging (split per H-2 — TF-19 must not wait on a research study):**
1. **PR-A (code):** complete propensity-model code path, ships **untrained** (`from_variant`/`from_hub` raise `FileNotFoundError`); synthetic CI fixture + 3-trial HPO smoke + fit-on-fixture round-trip + real-provider extraction tests in the regular suite. `xcross_attempt_xfns` **not** wired into any default xfn list yet.
2. **PR-B (weights + surface validation + TF-19 wiring):** maintainer training run on the pining corpus, bundled/Hub weights, pre-registered acceptance gates, the **surface GK-block ablation** metric, two-candidate public/full paired test, xfn-list wiring (GK-union list). No causal harness here.
3. **PR-C (causal research harness + ADR-015):** the `_causal` matching port + thin script driver + crosser-anchored opportunity build + placebo negative-control + Abadie–Imbens SEs + report. Standalone; does not gate PR-B's weights or TF-19.

### 1.2 Out of scope (deliberately)

- The paper's **outcome model for shot creation as a runtime feature** — silly-kicks already values shot/threat via VAEP/xthreat/xS; only the *propensity* (treatment) surface becomes a library feature. The cross→shot causal effect lives in the validation harness, not as an xfn.
- The **`extended` feature variant** (TF-7 pitch-control GK-influence / TF-14 defensive-line / TF-15 GK-influence primitives as confounders) — the `XCrossFeatureSet` Literal + data-driven extractor ship now (they shape the model/metadata API), but only `"faithful"` is implemented; `"extended"` is a documented `NotImplementedError` extension point. Rationale identical to xS: it adds untested-on-real-data primitive coupling + the canonical-vs-counterfactual pitch-control-cache trap (§7).
- **Set-piece crosses** as positives by default — `corner_crossed`/`freekick_crossed` are dead-ball, not open-play GK-deterrence decisions. Excluded by default; exposed as a trainer param (`cross_types`) mirroring xS's `shot_types`.

---

## 2. Module structure (mirror of TF-16)

| Artifact | Path | Role |
|---|---|---|
| Production module | `silly_kicks/tracking/_xcross_attempt.py` | extractor, `XCrossAttemptModel`, `compute_*`/`add_*`/`*_xfns` |
| Shared geometry helper | `silly_kicks/tracking/_geometry.py` (reuse) | already extracted by xS — `to_goal_relative_x`, `GOAL_Y`, `PITCH_LENGTH/WIDTH`, `GEOMETRY_VERSION` |
| Atomic mirror | `silly_kicks/atomic/tracking/features.py` (extend) | re-export `add_xcross_attempt`, `xcross_attempt_xfns` |
| HPO objective | `silly_kicks/tracking/_xcross_attempt_objective.py` | `ruthless` `CachedObjective` (mirror `_xshot_occurrence_objective.py`) |
| Training CLI | `scripts/train_xcross_attempt.py` | I/O + `OptunaStrategy` driver (mirror `train_xshot_occurrence.py`) |
| Publish CLI | `scripts/publish_xcross_attempt.py` | HF upload + round-trip verify (mirror `publish_xshot_occurrence.py`) |
| Shared label helper (PR-A, M-3) | `silly_kicks/tracking/_occurrence_labels.py` (new) | `_build_occurrence_labels(actions, frames, *, types, horizon)` consumed by **both** xS and xCross; `build_xshot_labels`/`build_xcross_labels` become thin wrappers (xS refactor is behaviour-preserving, guarded by xS tests). |
| **Causal matching port (PR-C, C2, private per L-1)** | `silly_kicks/_causal/matching.py` (new **private** module) | **pure** functions: `propensity_match`, `estimate_att`, `estimate_atnt`, `smd_balance`, `placebo_shift`, `abadie_imbens_se`. Provider-agnostic, no I/O. Unit-tested against known truth. Promote to public `silly_kicks/causal/` only when a 2nd consumer (TF-19) actually lands. |
| Causal harness driver (PR-C) | `scripts/validate_xcross_causal.py` + bundled report | **thin I/O driver** over the port: builds crosser-anchored opportunity rows, runs the GK-ablation + placebo, reports metrics + carrier-coverage validity caveat |
| Public exports | `silly_kicks/tracking/__init__.py` (extend) | `compute_xcross_attempt`, `add_xcross_attempt`, `xcross_attempt_xfns`, `XCrossAttemptModel`, `XCrossFeatureSet`, `extract_xcross_features`, `prepare_xcross_training_data`, `subsample_negatives` (reuse xS's) |

**Dependency placement (ADR-011):** inference gates on the existing **`[xgboost]`** extra (lazy import); HPO/training on the existing generic **`[train]`** extra (`ruthless-efficiency[optuna]` + xgboost). The causal harness's matching may use `scikit-learn` NearestNeighbors (already a runtime dep) — **no new dependency, no R**. Reproduce the Cao 1:1-NN-with-replacement matching in numpy/sklearn (documented in the harness; cross-checked against the paper's reported ATE/ATT direction on a sanity fixture).

**No edits** to `silly_kicks/calibration/`, `scripts/calibrate_*`, `scripts/_loader_*` (owned elsewhere). The trainer *calls* `_loader_pining.load_matches` but does not modify it.

---

## 3. Public API (mirror TF-16 signatures verbatim where possible)

**`home_team_id` (L-3):** carried as an accepted-but-unused param across all three surfaces purely for **signature parity with xS** (the goal end is GK-resolved, not derived from `home_team_id`). This is **inherited xS debt**, not new design — documented as such; if xS later drops it, TF-17 follows.

### 3.1 Per-frame primitive (TF-19 consumes)
```python
def compute_xcross_attempt(
    frames: pd.DataFrame, *,
    model: XCrossAttemptModel | XCrossFeatureSet | None = None,
    home_team_id: int | str | None = None,        # unused; goal GK-resolved; kept for symmetry
    pitch_control_cache: PitchControlCache | None = None,   # reserved for "extended"
    link_frame_ids: set[int] | None = None,
) -> pd.DataFrame:  # adds one float column `xcross_attempt` ∈ [0,1] (NaN where state undefined)
```

### 3.2 Action-coupled aggregator
```python
@nan_safe_enrichment
def add_xcross_attempt(actions, frames, *, model=None, links=None,
                       home_team_id=None, pitch_control_cache=None) -> pd.DataFrame:
    # one `xcross_attempt` column at each action's linked frame; provenance-skip guard;
    # `links` pre-link kwarg; pre-populated-column fast path; NaN ids → NaN (ADR-003)
```

### 3.3 VAEP factory (`_frame_aware`)
```python
def xcross_attempt_xfns(*, model=None, home_team_id=None, pitch_control_cache=None) -> list:
    # single FrameAwareTransformer → xcross_attempt_a0/_a1/_a2; frames=None → 3 NaN cols
```

### 3.4 Model class
```python
XCrossFeatureSet = Literal["faithful", "extended"]

class XCrossAttemptModel:
    # pinned-deterministic XGBoost; pickle-free booster JSON + metadata.json + SHA256SUMS
    def __init__(self, *, feature_set="faithful", params=None): ...   # "extended"→NotImplementedError
    def fit(self, features, labels, *, carrier_params=None, horizon_seconds=1.0): ...
    def predict_proba(self, features) -> np.ndarray: ...   # (n,), P(cross)
    def save(self, path): ...   # + metadata (feature_names, feature_set, horizon, cross_types,
                                #   carrier_params, params, version, pitch_length/width,
                                #   geometry_version, xgboost_version, shipped_variant, provider_list)
    @classmethod
    def load(cls, path): ...    # SHA-256 + fail-closed pitch-dim guard + restore carrier_params (R3)
    @classmethod
    def from_variant(cls, variant="default"): ...   # FileNotFoundError until PR-B
    @classmethod
    def from_hub(cls, repo_id="silly-kicks/xcross-attempt-v1"): ...  # inert until PR-B
```

### 3.5 Training-data prep (shared train/serve extractor — the anti-skew guarantee)
```python
def prepare_xcross_training_data(
    frames, actions, *, home_team_id,
    feature_set="faithful", horizon_seconds=1.0,
    wide_area_only=True,                 # domain filter (see §4/§5)
    cross_types=("cross",),              # open-play only by default
    carrier_params=None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:   # (features, labels, groups=game_id)
```

---

## 4. Label definition

**`xcross_attempt` label** (per frame, per in-possession team):

`y = 1` iff an open-play cross by the in-possession team occurs at any frame whose `time_seconds` ∈ `[t, t + horizon_seconds]` **within the same period**, else `0`.

- **Horizon via `time_seconds`, NOT `frame_id` arithmetic** — identical rationale to xS (B1): `frame_id` is not time-contiguous across providers. Per-period `np.searchsorted` on sorted `time_seconds`. **Extract a shared `_build_occurrence_labels(actions, frames, *, types, horizon)` (M-3, `tracking/_occurrence_labels.py`)** consumed by both xS and xCross; `build_xshot_labels` becomes a thin wrapper (behaviour-preserving, guarded by the existing xS label tests — Chesterton/National-Park), `build_xcross_labels` another. Avoids two drifting copies of the time-windowed-occurrence logic.
- **"Cross" = SPADL `type_id == actiontype_id["cross"]`** (open-play) by default; `cross_types` param can add `corner_crossed`/`freekick_crossed` for sensitivity.
- **No linkage for the label (R2)** — actions and frames share the per-period `time_seconds` base; compare the cross action's own `time_seconds` against each frame window for the same `(game_id, period_id, team)`. Avoids the ±0.2 s link smear at a 1 s horizon.
- **Possession at `t`** fixed by `derive_team_in_possession` at `t`; turnover-then-opponent-cross is a NEGATIVE for the team in possession at `t` (mirror xS D5).

### Anchoring (H2) — the surface is STATE-anchored, not sender-anchored
The label is "the in-possession **team** attempts a cross within `[t, t+horizon]`". The carrier at `t` is frequently **not** the eventual crosser (a wide build-up: full-back at `t` → winger crosses 0.8 s later). So the propensity surface answers **"will a cross come from this state?"** — a *state-level* object, which is exactly what GKDV/TF-19 needs. It deliberately does **NOT** replicate the paper's *sender-level* treatment ("will *this sender* cross"); §13 attribution reflects this. Spatial features are **ball-anchored** (like xS, computed from `(bx,by)`); the two carrier-dependent confounders (#3, #7) are flagged in §5 with their anchoring + coverage caveat (M3). The paper-faithful **sender-level** unit is reconstructed only in the causal harness (§9), via **crosser-anchored** opportunity rows.

### Training domain filter — **wide-area frames** (re-justified from cross-origin evidence, M1)
The paper restricts to "crossing opportunities," not all frames. Mirror via a **wide-area + advancement** gate (so the model learns *cross-vs-not given a crossing position*, not *wide-vs-central*):
- `ball_state == "alive"`, possession resolvable, AND
- ball in a **wide corridor** — `cross_zone` width boundary (`vaep/features/specialty.py:54`): SPADL `y < 14 OR y > 54` (outer ~14 m each flank). **Corridor provenance (M-4):** this is `cross_zone`'s **dense-zone** bound, **NOT** SkillCorner's converter cross-origin corridor — `skillcorner._is_cross` (`skillcorner.py:114`) uses the slightly wider `y < 15 OR y > 53`. So a ~1 m band per flank (`y ∈ [14,15] ∪ [53,54]`) is a genuine SkillCorner cross-origin that falls *outside* this domain. This is an **intentional** choice of the tighter dense-crossing zone (consistent with the existing VAEP `cross_zone` feature), not a claim that 14/54 is "SkillCorner's definition." The corridor bound is a param; PR-B reports the boundary-band sensitivity. (Only SkillCorner detects crosses spatially; other providers detect from event metadata, so their cross origins aren't corridor-bounded — a central cross is dropped by the wide-area filter by design.) AND
- ball past an **advancement threshold**. **Re-justification (M1):** the 35 m attacking-third bound is NOT inherited from xS's shot domain — it matches **SkillCorner's own spatial cross-origin definition** (`skillcorner._is_cross` uses `start_x > 70.0` ≡ 35 m from goal). But crossing *density* concentrates at `x ≥ 88` (`cross_zone` zones 2–4, ~17 m). **Selection rule (M-1) — choose the domain by a DESCRIPTIVE criterion, never by maximizing the eval metric** (selecting the domain to maximize PR-AUC and then reporting PR-AUC on the same data is optimistically biased). PR-B measures the **cross-density profile by x-band** (a descriptive statistic) and picks the advancement threshold where cross-density is materially non-trivial (pre-registered rule, e.g. the x-band capturing ≥X% of open-play crosses), reporting `permissive` (`x ≥ 70`) vs `tight` (`x ≥ 88`) as a sensitivity pair — it does **not** pick the domain that maximizes held-out PR-AUC. Default param `wide_area_only=True`; the advancement threshold is recorded in metadata and **read by the harness** (not re-literal'd) so the matched corpus cannot drift from the trained model's domain.

---

## 5. Features — `faithful` (paper 8 confounders + GK extension); `extended` deferred

`XCrossFeatureSet = Literal["faithful", "extended"]`, default `"faithful"`. **This PR implements `"faithful"` only.** The two sets produce different input dimensionality → genuinely different trained models, so `extended` later is additive (new variant + new weights), not a breaking change. `feature_set` recorded in metadata, re-checked at `load()`/inference.

All spatial features in **goal-relative coordinates** via the shared `_geometry` helper (LTR/RTL map identically), exactly as xS.

### 5.1 `faithful` — the paper's 8 confounders, realized with silly-kicks primitives + the novel GK block

| # | Paper confounder | silly-kicks realization (per frame, from the in-possession ball-carrier = "sender") |
|---|---|---|
| 1 | Score differential (team in possession) | from `actions`/match context lookup (reuse xS's `_build_score_lookup` pattern from ghost-GK training prep) |
| 2 | Dist sender→nearest defender | nearest opponent outfielder to the carrier (goal-relative) |
| 3 | Space controlled by sender | **cheap Voronoi-area proxy** at the carrier cell in the runtime surface (keeps `faithful` free of the canonical-vs-counterfactual pitch-control-cache trap that would otherwise bite TF-19 — §7). Full TF-7 pitch control deferred to `extended`. **Carrier-anchored** (M3). The causal harness (§9) may use paper-faithful PC offline; the runtime/harness #3 forms differ by design → the harness's "trained-model-as-alternative-propensity" cross-check is dropped (H3). |
| 4 | Dist sender→nearest teammate | nearest same-team outfielder to the carrier |
| 5 | Dist sender→endline | carrier distance to the attacked goal line (goal-relative `r`-like) |
| 6 | Off/def player ratio in box | count attackers vs defenders inside the penalty area of the attacked goal |
| 7 | Crosser position (FW/MF/DF) | **runtime surface:** proxy by carrier's mean longitudinal role (carrier-anchored). **Causal harness: DROP it** (nit) — a tracking role-proxy correlates with the GK block and the defensive line, contaminating the clean GK ablation; the paper's #7 is categorical event metadata, a genuinely different variable. Document the divergence. |
| 8 | Ten-minute warning (final 10 min of half) | from frame `time_seconds` + period |
| **GK block (novel)** | — | **GK_r, GK_theta** (defending GK dist/bearing to goal centre, from the GK frame row — mirror xS); **GK lateral offset** from goal centre; **GK→near-post / GK→far-post** distances; **carrier-side flag** (is the cross from the GK's near or far side). All from the resolved defending-GK row; cheap, interpretable, isolatable as the headline contribution. |

- **Missing values** (sparse providers, no GK row, <1 defender) → NaN. XGBoost handles NaN natively (no fillna — calibration M3 rule).
- **Carrier-resolution coverage (M3):** confounders #3/#7 are **carrier-anchored** (player properties), and `infer_ball_carrier` reliability is provider-variable (cf. the xS N1 caveat — GK identification was 21–50% on Metrica/SkillCorner pre-fix; carrier inference has similar provider variance). `prepare_xcross_training_data` therefore **logs a carrier-resolution coverage/quality stat over the wide-area domain** per provider (no silent caps); low coverage degrades the novel block and must be visible, not hidden.
- The GK block is implemented as a **contiguous, separately-toggleable feature group** so the causal harness (§9) can ablate "with GK confounders" vs "without" by dropping exactly this block — the clean experimental contrast.

### 5.2 `extended` — deferred extension point (NotImplementedError)
Would add TF-15 GK-influence primitives (threat-weighted PC share, uniquely reachable area, zone closing time), TF-14 defensive-line geometry, richer TF-7 surfaces — through a passed `pitch_control_cache`. Deferred (untested primitive coupling + cache trap). Additive when it ships.

### 5.3 Shared train/serve extractor
`extract_xcross_features(frame_data, *, gk_team_id, goal_x, feature_set, ...) -> 1-row DataFrame` — the single code path for both `prepare_xcross_training_data` and `compute_xcross_attempt`. Anti-skew guarantee (mirror xS's `extract_xshot_features`).

---

## 6. Model & HPO (mirror xS verbatim)

- **Pinned deterministic XGBoost** (`tree_method="hist", n_jobs=1, subsample=1.0, colsample_bytree=1.0, random_state=seed, eval_metric="logloss"`). `base_score = labels.mean()` set before fit (calibrated).
- **`XCrossAttemptObjective`** `CachedObjective`: `prepare()` concats pre-built `(X,y,groups)`; `evaluate_patch` fits + `StratifiedGroupKFold` CV → mean held-out **log-loss** (MINIMIZE) + **PR-AUC**/**Brier** diagnostics; `assert_cache_equivalence` (1e-9). `patch_params = {n_estimators, max_depth, learning_rate, min_child_weight, reg_lambda}`. **No `scale_pos_weight`** (PR-S80 lesson: keep `P(cross)` a calibrated proper-scoring-rule output; the natural base rate is the XGBoost `base_score`).
- **CV:** `StratifiedGroupKFold` (stable per-fold positives at low base rate), `groups = game_id`.
- **R3 carrier-param coupling:** record `tolerance_m/beta/gamma` (from `DEFAULT_CARRIER_PARAMS`) in metadata AND consume them at inference (read from metadata, not live defaults). Identical to xS.
- **Imbalance:** crosses are ~3× more common than shots (live probe: 68 vs 23 per match), so within the wide-area domain the positive rate is *healthier* than xS's — PR-AUC-vs-baseline remains the meaningful gate but the target is less extreme. **Default `negative_subsample=None` (M4)** — the healthy base rate makes subsampling unnecessary. If used, note that the objective's CV folds fit without an explicit `base_score` (xgboost≥2 auto-estimates it per train fold), so under subsampling the HPO log-loss sits on a different intercept than the shipped model's explicit `base_score=labels.mean()` → HPO calibration metrics are not directly comparable to the shipped model. `subsample_negatives` (train-folds only, seeded) is reused from xS verbatim but off by default.

---

## 7. TF-19 integration path
- TF-19 calls `compute_xcross_attempt(frames, model=..., link_frame_ids=...)` or `predict_proba` on a feature matrix it controls. xCrossAttempt stays **counterfactual-agnostic**.
- The counterfactual `P(cross | ghost_GK)` is produced by TF-19 substituting the ghost-GK position into the frame **before** feature extraction. **This is where the `kde_backend` train/serve guard binds** (TF-19, not TF-17) — TF-19 must pin one `kde_backend` for the ghost-GK mode it substitutes and assert metadata match.
  > **Superseded by ADR-016 (4.14.0):** the served `ghost_gk_x/y` is now `predict_mean()` (the deterministic, `kde_backend`-free boosted HGBR position), not the KDE mode — so the `kde_backend` train/serve guard is moot. TF-19 must substitute `predict_mean()` (not the KDE mode/argmax via `predict_density`); then no backend pin / metadata assert is needed. Revisit a pin only if a variant deliberately consumes the density mode or `ghost_gk_density_spread`. See the TF-19 row in `TODO.md`.
- **Canonical-vs-counterfactual cache rule (B2) — RESOLVED via the §5 #3 proxy.** `faithful` #3 uses a cheap Voronoi-area proxy (not full TF-7 PC), so `faithful` is **cache-trap-free**: TF-19 can substitute the ghost GK and re-extract without any `pitch_control_cache` concern. The trap only re-enters with `extended` (which adds TF-7 surfaces); the rule (a counterfactual caller MUST NOT pass `pitch_control_cache`) is documented now so `extended` inherits it.

---

## 8. Serialization (identical to xS)
```
xcross_attempt_v1/
├── model.json        # xgboost native booster
├── metadata.json     # feature_names, feature_set, horizon_seconds, cross_types,
│                     #   carrier_params, params, version, pitch_length/width,
│                     #   geometry_version, xgboost_version, shipped_variant, provider_list
└── SHA256SUMS        # CRLF→LF normalised
```
`load()` SHA-256 verifies, fail-closed pitch-dim guard, warn-only `geometry_version`, restores `carrier_params` (R3). Bundled default at `silly_kicks/tracking/_xcross_weights/default/`.

---

## 9. Causal validation harness (PR-C research deliverable) — private `silly_kicks/_causal/` port + thin driver

### 9.1 Causal model — the explicit DAG (H1)
The estimand is the effect of **treatment** `Z` (a cross is attempted from a crossing opportunity) on **outcome** `Y` (the in-possession team creates a shot shortly after). Assumed DAG:

```
            attacking state S
           /       |        \
          v        v         v
   GK_position    Z (cross)   Y (shot)
       |  \________^          ^
       |           (deterrence: GK_pos → Z)
       \___________________________________> Y   (GK_pos → shot prevention)
```

- `S → Z`, `S → Y`, and `S → GK_position` make `S` a confounder; the paper conditions on (a proxy of) `S` via its 8 covariates.
- **GK_position is a legitimate backdoor confounder, NOT a mediator or collider — by temporal construction.** It is measured at frame `t`; the cross decision occurs in `[t, t+horizon]`, so the measured GK position is **pre-treatment**. A mediator of `Z→Y` would be the GK's *post-cross reaction* (not measured here); a collider would require GK_position to be a common *effect* of two nodes on the `Z–Y` path (it is not — it is caused by `S`). Conditioning on a pre-treatment common cause of `Z` and `Y` (`GK_pos → Z` via deterrence; `GK_pos → Y` via shot prevention) **reduces** confounding bias. The harness states this DAG and the pre-treatment-measurement guarantee explicitly; if a future variant measured GK position *after* the cross it would become a mediator and the analysis would be invalid.

### 9.2 Unit of analysis — crosser-anchored opportunity rows (H2), with an explicit dedup rule (R2-M1)
To match the paper's **sender-level** treatment, the harness builds **one row per crossing opportunity**. **Unit (precise, reproducible):** a *crossing opportunity* = **one continuous wide-area possession-spell** — a maximal run of consecutive in-possession frames (same possessing team, no turnover, ball continuously inside the wide-area/advancement domain), anchored to **the carrier at spell entry**. `Z=1` if a cross by the possessing team occurs **anywhere in that spell within the horizon of entry**, else `Z=0`; `Y` = shot by the possessing team within a short post-window (reuse the xS shot label as outcome). **Dedup rule:** re-entering the corridor after exiting (or a carrier change mid-spell) starts a **new** spell only if the possession broke or the ball left the domain; a carrier hand-off *within* one continuous in-domain spell does **not** create a second opportunity (it stays one row, anchored at entry). This makes treated/control counts — and therefore the matched ATT — well-defined. Domain thresholds + horizon are **read from the trained model's `metadata.json`** (not re-literal'd) so the matched corpus matches the model's domain. Distinct from the runtime surface's state-anchored rows (§4); the divergence is intentional and documented.

### 9.3 Estimator — pure private port `silly_kicks/_causal/matching.py` (C2; private per L-1)
1:1 nearest-neighbor propensity-score matching, **with replacement**, ties allowed, **no caliper** (paper-faithful) — numpy/sklearn `NearestNeighbors`, **no R, no new dependency**. Pure functions: `propensity_match`, `estimate_att`, `estimate_atnt`, `smd_balance`, `placebo_shift`, `abadie_imbens_se`. The propensity is a logistic regression on the paper's 8 confounders ± the GK block (interpretable, matches the paper). **Variance: Abadie–Imbens (2006/2008) matching SEs** — naive and bootstrap SEs are provably biased under matching-with-replacement (C1). `estimate_*` return point estimate + AI SE + the matched balance table.

### 9.4 Headline experiment — GK-confounder ablation WITH a negative control (C1)
Report **ATT** (the headline estimand) with vs without the GK block. **An ATT shift alone is NOT evidence** — adding any equal-width covariate block perturbs a matched ATT via overlap loss, propensity-model variance, and finite-sample matching jitter. So:
- **(a) Placebo null band:** run the same ablation with a **placebo block** of identical dimensionality (permuted-GK / random columns), repeated over seeds, to estimate the null distribution of ATT shifts. The GK shift is "real" **only if it clears the placebo band** (e.g. exceeds the 95th percentile of placebo shifts).
- **(b) Pre-registered effect-size threshold** for "non-trivial," fixed before the run (mirroring §10's pre-registered acceptance gates).
- **(c)** SMD balance must improve post-match (`smd_balance`), and overlap/common-support must hold; **no causal claim is made if overlap or balance fails** (no-silent-caps). Rosenbaum-style sensitivity reported.

**This is a REPORTED finding, never a ship gate or CI assertion (H-2/H-3).** A GK shift that does NOT clear the placebo band is a valid null result, not a regression — it does not block PR-B's weights (the surface ships its GK block regardless; the shipped-surface evidence is the §10 ablation). Only the known-truth *method* tests (§11.3) gate CI.

### 9.5 Validity caveats & sanity check
- **Carrier-resolution coverage (L-2):** the crosser-anchored opportunity rows (§9.2) are selected on frames with a *resolvable* carrier, so low per-provider carrier coverage is a **selection bias on the causal estimate**, not merely weaker features. The harness reports per-provider carrier coverage as a validity caveat and **excludes low-coverage providers from the headline ATT** (threshold pre-registered).
- **Sign sanity:** on the public corpus, ATE/ATT signs match the paper's positive direction (cross → more shots); logged, not a hard gate (different league/era).

Output: a `metrics.json`-style artifact + a short report bundled into the **PR-C** feature commit (not a standalone doc commit).

---

## 10. Shipping scope (staged, collision-safe)

**PR-A (code, untrained):** module, model, objective, training CLI, all three surfaces, atomic mirror, NOTICE, ADR update, full test suite; synthetic CI fixture + 3-trial Optuna smoke + fit-on-fixture round-trip + **real-provider extraction tests in the regular suite**. `from_variant`/`from_hub` raise clear `FileNotFoundError`. `xcross_attempt_xfns` **not** added to any default/union xfn list. Docstring Examples fit-on-fixture (no network/weights).

**PR-B (weights + shipped-surface validation + TF-19 wiring):** maintainer training run on the pining corpus (public + the two-candidate public/full paired test — the 17 public matches already carry ≈1,150 open-play crosses, so the public candidate is *not* positive-starved; whether GS-owner data helps or hurts public generalization is decided by the paired test exactly as xS did it, not assumed either way); bundled/Hub weights; pre-registered acceptance gates (PR-AUC > positive-rate, Brier < base-rate, log-loss < ln2, ≥2 usable folds, artifact < 5 MB); **the surface GK-block ablation metric (H-1)** — trained held-out PR-AUC/log-loss with vs without the GK feature group; **the GK substitution-sensitivity probe (R2-H1)** — the `|P(cross|actual_GK) − P(cross|shifted_GK)|` distribution over the domain (TF-19's actual operation). **Both run on the paired-test winner (the actually-shipped weights) and are reported in its `metrics.json` (R2-L1)** — GK/carrier resolution is provider-variable, so public-only vs full candidates can differ; the metrics describe the shipped model, not an arbitrary candidate. If both come up inert, `metrics.json` flags TF-17 **not TF-19-ready** (loud, not silent). xfn-list wiring (GK-union list only, mirroring PR-S80's xS wiring into `pre_shot_gk_full_default_xfns`); directional CI tripwire fixture (mirror `xshot_directional`). **No causal harness here** — TF-19 can consume the surface as soon as PR-B lands.

**PR-C (causal research harness + ADR-015):** the private `_causal/matching.py` port (unit-tested against known truth, §11.3) + `scripts/validate_xcross_causal.py` thin driver + crosser-anchored opportunity build + GK-vs-placebo ablation + Abadie–Imbens SEs + carrier-coverage validity + report. Standalone research artifact; **does not gate PR-B's weights or TF-19** (a null causal finding is valid). ADR-015 (the causal-port pattern) lands here.

---

## 11. Testing strategy (TDD — tests authored before implementation)

### 11.1 Unit (`tests/tracking/test_xcross_attempt.py`)
- `test_extract_features_faithful_shape` — column count/names/dtypes, goal-relative normalization.
- `test_extended_raises_not_implemented`.
- `test_goal_relative_symmetry` — LTR vs RTL identical (shared `_geometry`).
- `test_gk_block_isolatable` — the GK confounder columns are a contiguous, droppable group (enables both the §10 surface ablation and the §9 causal ablation).
- `test_faithful_never_touches_pitch_control_cache` (L-4) — spy/assert that `faithful` extraction (incl. #3's Voronoi proxy) never reads or memoizes a `pitch_control_cache` surface, locking the counterfactual-substitution guarantee for TF-19.
- `test_box_ratio_*` — off/def-in-box count edge cases (empty box, all attackers, on-line players).
- `test_wide_area_domain_filter` — central-third / central-corridor frames excluded; wide attacking-third frames kept.
- `test_label_horizon_via_time_seconds`, `test_label_robust_to_noncontiguous_frame_id`, `test_label_no_period_bleed`, `test_label_inclusive_of_t`, `test_label_turnover_opponent_cross_is_negative`, `test_label_cross_types` (open-play default; corner/freekick togglable).
- `test_model_fit_predict_proba`, `test_model_deterministic`, `test_model_save_load_roundtrip`, `test_model_sha256_verification`, `test_feature_set_metadata_roundtrip`.
- `test_compute_xcross_no_model`, `test_add_xcross_aggregator` (links + provenance skip), `test_add_xcross_nan_safe` (ADR-003 auto-discovery), `test_xcross_xfns_factory` (`_a0/_a1/_a2`; NaN on frames=None), `test_xfns_frame_aware_marker`.
- `test_inference_uses_metadata_carrier_params` (R3).

### 11.2 Integration (`tests/tracking/test_xcross_attempt_integration.py`)
- dtype-asymmetry (int64 actions + str frames), atomic mirror, `train_xcross_attempt.py` smoke (module/`PYTHONPATH` subprocess), objective cache-equivalence (1e-9), 3-trial Optuna smoke, carrier-params-in-metadata, TF-19 interface stub.
- **`test_build_xshot_labels_bit_identical_after_refactor` (R2-L2)** — the M-3 refactor modifies *shipped* xS code (`build_xshot_labels` → thin wrapper over `_build_occurrence_labels`). A **golden-master**: a fixture's expected xS labels are frozen pre-refactor and the refactored `build_xshot_labels` must reproduce them exactly (`assert_frame_equal`/`assert_array_equal`), so the extraction cannot silently shift xS's labels. Lives in the **xS** test module (it guards xS), runs in PR-A.

### 11.2a Real-provider extraction (regular suite, NOT e2e)
`tests/tracking/test_xcross_attempt_real_data.py` over the slim fixtures + (where present) cross-bearing fixtures: extractor runs per provider (no crash, feature ranges sane), `compute_xcross_attempt` (fixture-fit model) ∈ [0,1]/NaN, goal-relative symmetry on real LTR+RTL, int64/object id asymmetry. (Note: only `sportec_slim.parquet` currently has cross rows — PR-A adds a frozen cross-feature fixture analogous to `xshot_directional`.)

### 11.3 Causal port unit tests (`tests/causal/test_matching.py`, PR-C) — **regular suite, NOT e2e (C2)**
The load-bearing matching estimator is unit-tested against **known truth**, not just a real-data sign-check:
- `test_recovers_known_ate` — synthesize data with a known ATE + known confounding; assert `estimate_att`/`estimate_atnt` recover it within tolerance.
- `test_smd_balance_improves_post_match` — SMD on confounders strictly drops after matching.
- `test_placebo_block_zero_shift` — a permuted/random block produces ~zero ATT shift (the null the GK shift must beat — C1).
- `test_with_replacement_reuses_controls` — a control can match multiple treated (structural check of the with-replacement rule).
- `test_abadie_imbens_se_positive_finite` + a comparison showing the AI SE differs from a naive SE on a constructed case (documents why the naive SE is wrong under replacement).
- `test_no_caliper_keeps_all_treated` — no treated unit dropped (paper-faithful).
- `test_opportunity_dedup` (R2-M1) — one opportunity per continuous wide-area possession-spell anchored at entry; corridor re-entry after a turnover/exit → new spell; mid-spell carrier hand-off → still one row.

### 11.4 E2e (`@pytest.mark.e2e`) — skipped placeholders in PR-A, flipped on in PR-B/PR-C
- **PR-B:** `test_xcross_e2e` (acceptance gates), `test_xcross_cross_provider`, `test_surface_gk_block_ablation_runs` (the with/without-GK ablation **runs and emits both numbers** — asserts the metric is *produced*, not that GK wins), `test_gk_substitution_sensitivity_runs` (R2-H1 — shifts the GK row, re-extracts, re-predicts, emits the `|Δ|` distribution; asserts it is *computed and reported*, not that it exceeds a threshold — an inert result is a reported `not-TF-19-ready` flag, not a CI failure).
- **PR-C:** `test_causal_harness_runs_and_reports` (the harness runs end-to-end and writes ATT + placebo-band + carrier-coverage to the report). **NOT** `gk_beats_placebo` — a null is a valid finding, not a CI failure (H-3); the real-data GK-vs-placebo outcome is reported, never asserted. (Consistent with §9.5's sign-check being logged, not gated.)

### 11.5 Invariants — `xcross_attempt ∈ [0,1]` where non-NaN (fixture spans positives+negatives).
### 11.6 Perf — structural guard (call-count/array-shape spy) on `extract_xcross_features`, NOT a wall-clock ceiling (TF-16 CI-flakiness lesson).

---

## 12. Open questions — RESOLVED (review round 1)

- **Q1 — confounder #3 "space controlled by sender": RESOLVED → cheap Voronoi-area proxy** in the runtime surface (cache-trap-free; §5/§7). Full TF-7 PC deferred to `extended`. The harness's paper-faithful PC may differ → the harness's "trained-model-as-alternative-propensity" cross-check is dropped (H3); the GK-block ablation is form-invariant in #3.
- **Q2 — confounder #7 "crosser position": RESOLVED → proxy in the runtime surface, DROP in the harness.** A tracking role-proxy contaminates the GK ablation; the paper's #7 is categorical event metadata (a different variable). Documented divergence.
- **Q3 — horizon: RESOLVED → pick by a DESCRIPTIVE lead-time quantile (M2 + M-1).** PR-B sets the horizon from the empirical wide-area-entry→cross lead-time distribution (e.g. its median), a descriptive statistic — **NOT** by maximizing held-out PR-AUC (selection-on-the-metric bias). The label builder stays horizon-parameterized; the chosen value is recorded in metadata + read by the harness.
- **Q4 — causal harness placement: RESOLVED → pure PRIVATE port `silly_kicks/_causal/matching.py` + thin script driver (C2; private per L-1).** TF-19 reuse is a normal internal import, not a `scripts/` dependency; promote to public `silly_kicks/causal/` only when a 2nd consumer lands. Ships in **PR-C**, not PR-B (H-2). Gets ADR-015 (§14).

---

## 13. Academic attribution (ADR-005 / ADR-011) — with the H2 faithfulness caveat
New NOTICE entry: xCrossAttempt (`silly_kicks/tracking/_xcross_attempt.py`, TF-17) is **inspired by** Cao et al. (2025), arXiv:2505.11841. **The runtime propensity surface is a per-frame, STATE-anchored occurrence model** ("will the in-possession team cross from this state within the horizon") — it is **not** a faithful reproduction of the paper's *sender-level* treatment, and the attribution must not claim it is (H2). The paper's sender-level causal framework (1:1 NN propensity-score matching with replacement; ATE/ATT) is reproduced **only in the causal validation harness** (`silly_kicks/_causal/matching.py` + `scripts/validate_xcross_causal.py`, PR-C), on crosser-anchored opportunity rows, **extended with goalkeeper-position confounders** (the paper's confounder set excluded all GK variables) and validated against a placebo negative control. `See NOTICE for full bibliographic citations.` in each public docstring.

---

## 14. Version & release (three PRs — H-2)
- **PR-A (code, untrained):** minor bump. Four-file sync (`pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md`, `CHANGELOG.md`).
- **PR-B (weights + surface ablation + TF-19 wiring):** minor bump. CHANGELOG notes the GK-confounder novelty (with the surface-ablation number) + the carrier-param retrain trigger.
- **PR-C (causal harness + ADR-015):** minor bump (or patch — research artifact, no runtime change). CHANGELOG notes the causal finding as a report.
- **Version collision (M-2):** the **PR-S81** session (ghost-GK re-fit + serve-carrier + R3) is also targeting **4.10.0**. TF-17 PR-A is only at planning, so PR-S81 lands first; TF-17 takes the next free number. Whoever merges second re-bumps all four files (version-bump hard gate). Light file overlap — PR-S81 edits `_ghost_gk.py`+`features.py`; TF-17 edits `tracking/__init__.py`+`atomic/tracking/features.py` — expect a small rebase on whichever lands first. **Coordinate the number explicitly at merge.**
- **ADR (resolved):** extend **ADR-011** with a TF-17 note (3rd feature on the lifecycle) AND add a **new ADR-015 "Causal inference port (`silly_kicks/_causal/`, private) — pure matching estimators as a reusable boundary"** in **PR-C** (where the port lands). ADR-015 records: pure-function port + thin script driver; numpy/sklearn matching (no R, no new dep); with-replacement + Abadie–Imbens SEs; the placebo negative-control requirement; the explicit-DAG / pre-treatment-measurement discipline; report-not-gate; private-until-2nd-consumer (L-1).
