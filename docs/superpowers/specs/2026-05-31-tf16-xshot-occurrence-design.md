# TF-16: xShotOccurrence (xS) — Shot-Occurrence Probability Model

**Date:** 2026-05-31
**Status:** Draft — revised after review round 1 (TF-24/sweep session)
**Layer:** GKDV Layer 2 (TF-16)
**PR:** PR-S75
**Prereqs (all shipped):** TF-7 pitch control (3.7.0), TF-13 frame GK-ID + TF-14 defensive line (3.4.0), TF-15 GK influence (3.10.0), TF-5 ball carrier (3.5.0), link primitives (PR-S19).

**Review round 1 (resolved):** B1 label via `time_seconds`-within-period (not `frame_id` arithmetic); B2 cache canonical-frames-only; B3 real-provider extraction tests in the regular suite (not deferred); C1 provisional version; C2 ship `faithful` only; C3 carrier params in metadata; C4 openGoal golden-masters + shadow-union; C5 shared goal-relative helper; C6 PR-AUC/Brier diagnostics; D1–D5 clarifications. ADR is now a standalone trained-model-lifecycle ADR (ADR-011), not an ADR-005 amendment.

**Review round 2 (resolved in this revision):** R1 removed the contradictory `test_extended_uses_pitch_control_cache` (extended is not implemented this PR); R2 label compares the shot action's own `time_seconds` directly (no linkage smear); R3 inference *consumes* the metadata carrier params (not just records them) to prevent train/serve skew when the TF-24 apply-PR changes defaults; R4 e2e gate is PR-AUC-vs-baseline; R5 openGoal golden-masters derived from first-principles geometry; R6 `[train]` name kept + `_geometry` helper xS-only this PR (ghost-gk refactor deferred); R7 typo fix.

---

## 1. Purpose

Predict, from a single tracking-frame snapshot, **the probability that a shot is attempted within the next ~1 second** by the team in possession. This is the metric the source paper calls **xS** — distinct from expected goals (xG), which conditions on a shot already being taken. xS models *shot-taking behaviour*; xG models *shot quality*.

xS is **GKDV Layer 2**: TF-19 will compute `Δ = P(shot | actual_GK) − P(shot | ghost_GK)` per build-up frame in the final third, using this model's per-frame primitive as the `P(shot | ...)` surface. The source paper itself points at this use (Future Work §5.3: *"this framework could be mirrored to quantify shot and goal suppression, with credit assignment for lane-closing and goalkeeper positioning"*).

**Source:** Pipping, Feng & Sabin (2026), *"Beyond Expected Goals: A Probabilistic Framework for Shot Occurrences in Soccer."* arXiv:2512.00203.

### 1.1 Explicit scope boundary

**In scope:** xS only — the per-frame shot-occurrence classifier, its training pipeline, and its action-coupled / VAEP surfaces.

**Out of scope (deliberately):**
- The paper's **xG** re-implementation (P(goal | shot this frame)) — silly-kicks already values goal/threat via VAEP and xthreat (`xthreat.py`); a parallel xG surface would overlap existing functionality.
- The paper's **xG+** (`xS × xG`) and its **possession aggregation** (`max-per-possession` / `1 − ∏(1 − xG+ₜ)`) — these compose xS with xG; the composition shape is also TF-19's territory.
- The **`extended` feature variant** (faithful + pitch-control / defensive-line / GK-influence primitives) — the `XShotFeatureSet` Literal and the data-driven extractor structure ship now (they shape the model/metadata API), but only `"faithful"` is implemented; `"extended"` is a documented extension point (`NotImplementedError`) deferred to the weights/TF-19 follow-up. Rationale: `extended` adds primitive coupling that is untested on real data this PR and carries the canonical-vs-counterfactual pitch-control-cache trap (§7); it belongs where it can be validated.
- **Maintainer training run + bundled/Hub model weights** — deferred to a follow-up PR (see §9). This PR ships code + a synthetic CI fixture + real-provider extraction tests only.

---

## 2. Module structure

| Artifact | Path | Role |
|---|---|---|
| Production module | `silly_kicks/tracking/_xshot_occurrence.py` | feature extractor, `XShotOccurrenceModel`, `compute_*` / `add_*` / `*_xfns` surfaces |
| Shared geometry helper | `silly_kicks/tracking/_geometry.py` (new) | goal-relative coordinate transform extracted as a shared primitive (C5) — see §5.3 |
| Atomic mirror | `silly_kicks/atomic/tracking/features.py` (extend) | `add_xshot_occurrence` re-export per the established atomic-mirror pattern |
| HPO objective | `silly_kicks/tracking/_xshot_occurrence_objective.py` | `ruthless` `CachedObjective` for hyperparameter search |
| Training CLI | `scripts/train_xshot_occurrence.py` | I/O + ruthless `OptunaStrategy` driver |
| Public exports | `silly_kicks/tracking/__init__.py` (extend `__all__` + imports) | `compute_xshot_occurrence`, `add_xshot_occurrence`, `xshot_occurrence_xfns`, `XShotOccurrenceModel`, `XShotFeatureSet`, `prepare_xshot_training_data` |

**Dependency placement (ADR-009 precedent; Q1/Q2 resolved — see §11):**
- **Inference** path (`compute_*` / `add_*` / `*_xfns`) requires **xgboost** — gated behind the **existing `[xgboost]` extra** (`xgboost = ["xgboost>=2.0.0"]`, verified present in `pyproject.toml`), lazily imported (mirrors how ghost-gk gates `huggingface_hub`). Error message: `ImportError("xShotOccurrence inference requires: pip install silly-kicks[xgboost]")`. `import silly_kicks` stays dependency-light; a CI subprocess test guards this (the calibration harness has the same guard).
- **HPO objective + training CLI** require `ruthless-efficiency[optuna]` + xgboost — declared under a **new generic `[train]` extra** (`ruthless-efficiency[optuna]>=0.2.1`, `xgboost>=2.0,<3.0` — the same pins `[calibration]` uses), shared by future trained-model trainers rather than a per-feature `[xshot]` extra (avoids extra-sprawl; D3). The objective module is **not** imported by `silly_kicks/__init__` or by the inference path.

**No edits** to `silly_kicks/calibration/`, `scripts/calibrate_*`, or `scripts/_loader_*` (the live TF-24 sweep owns those). The HPO study writes to its **own fresh SQLite store path**; pointing ruthless at an existing `.db` resumes a study, so the trainer always uses a fresh/temp path or an explicit `--study-db` the caller controls.

---

## 3. Public API

### 3.1 Per-frame primitive (TF-19 consumes this)

```python
def compute_xshot_occurrence(
    frames: pd.DataFrame,
    *,
    model: XShotOccurrenceModel | XShotFeatureSet | None = None,
    home_team_id: int | str,
    pitch_control_cache: PitchControlCache | None = None,   # reserved for "extended" (not impl. this PR)
    link_frame_ids: set[int] | None = None,                 # restrict compute to linked frames (PR-S66)
) -> pd.DataFrame:
    """Add an ``xshot_occurrence`` column to each in-possession non-ball frame row.

    Returns the frames DataFrame with one new float column: P(shot attempted by the
    in-possession team within the next ~1 s), evaluated at each frame. Rows where the
    state is undefined (no resolvable possession, ball dead, NaN ball position) get NaN.

    The ``model`` is counterfactual-agnostic: it scores whatever frame state it is
    given. ``pitch_control_cache`` is accepted for forward-compatibility with the
    ``extended`` variant but is **valid only for canonical (unmodified) frames** —
    a counterfactual caller that has moved a player (e.g. TF-19's ghost-GK
    substitution) MUST omit the cache (the ``faithful`` variant uses no pitch
    control, so this is moot until ``extended`` ships). See §7.
    """
```

### 3.2 Action-coupled aggregator

```python
@nan_safe_enrichment
def add_xshot_occurrence(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    model: XShotOccurrenceModel | XShotFeatureSet | None = None,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
    pitch_control_cache: PitchControlCache | None = None,
) -> pd.DataFrame:
    """Enrich SPADL actions with one ``xshot_occurrence`` column (the xS at each
    action's linked frame). Provenance-skip guard + ``links`` pre-linking kwarg per
    ADR-005/ADR-008 conventions. NaN identifiers route to NaN output (ADR-003)."""
```

### 3.3 VAEP factory (`_frame_aware`)

```python
def xshot_occurrence_xfns(
    *,
    model: XShotOccurrenceModel | XShotFeatureSet | None = None,
    home_team_id: int | str,
    pitch_control_cache: PitchControlCache | None = None,
) -> list:
    """Factory returning a single FrameAwareTransformer. Emits
    ``xshot_occurrence_a0/_a1/_a2`` (3 gamestate slots). Links each slot once and
    restricts the single compute to the union of linked frames (PR-S66 pattern).
    On introspection (frames=None) returns the 3 columns as NaN (ADR-005)."""
```

### 3.4 Model class

```python
XShotFeatureSet = Literal["faithful", "extended"]

class XShotOccurrenceModel:
    """xS classifier: pinned-deterministic XGBoost over snapshot frame features.

    Serialization is pickle-free: xgboost's native booster JSON + a metadata.json
    (feature_names, feature_set, label spec, hyperparams, version) + SHA256SUMS.

    See NOTICE for full bibliographic citations.
    """
    def __init__(self, *, feature_set: XShotFeatureSet = "faithful",
                 params: dict | None = None) -> None:
        # feature_set="extended" raises NotImplementedError this PR (faithful-only);
        # the Literal + branch exist so the API/metadata are stable for the follow-up.
        ...
    def fit(self, features: pd.DataFrame, labels: pd.Series) -> XShotOccurrenceModel: ...
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:   # shape (n,), P(shot)
        ...
    def save(self, path: Path) -> None:                              # booster JSON + metadata + SHA256SUMS
        ...
    @classmethod
    def load(cls, path: Path) -> XShotOccurrenceModel:              # SHA-256 verified; requires xgboost
        ...
    @classmethod
    def from_variant(cls, variant: str = "default") -> XShotOccurrenceModel:
        ...   # raises FileNotFoundError until the follow-up bundles weights (§9)
    @classmethod
    def from_hub(cls, repo_id: str = _HF_REPO_ID) -> XShotOccurrenceModel:
        ...   # wired but inert until weights are published (§9)
```

### 3.5 Training-data prep

```python
def prepare_xshot_training_data(
    frames: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    home_team_id: int | str,
    feature_set: XShotFeatureSet = "faithful",
    horizon_seconds: float = 1.0,
    attacking_third_only: bool = True,
    negative_subsample: float | None = None,   # optional pre-fit row reduction (see §6.3)
    seed: int = 42,                            # seeds negative_subsample (D2 — determinism)
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Return (features, labels, groups) for one match. groups = game_id (for GroupKFold).
    Shared feature extractor guarantees train/serve parity. The carrier params used for
    possession resolution are recorded by the trainer into model metadata (C3)."""
```

---

## 4. Label definition

**`xshot_occurrence` label** (per frame, per in-possession team):

`y = 1` iff a shot by the in-possession team occurs at any frame whose `time_seconds` falls in `[t, t + horizon_seconds]` **within the same period**, else `0`.

- **Horizon via `time_seconds`, NOT `frame_id` arithmetic (B1).** `frame_id` is **not** time-contiguous across providers (SkillCorner is extrapolated; Metrica has heavy gaps; Gradient Sports ships duplicate frames), so `frame_id + n_ahead` would span an inconsistent wall-clock interval and silently mislabel the target. Instead, within each `(game_id, period_id)`, sort distinct frames by `time_seconds` and label a frame positive iff a same-team shot's `time_seconds` lies in `[t, t + horizon_seconds]`. This mirrors the house linkage idiom (`utils.link_actions_to_frames` / `slice_around_event` use `searchsorted` over `time_seconds`, never `frame_id` math). Implementation = per-period `np.searchsorted` on the sorted `time_seconds` array for the window's right edge.
- **"Shot" definition:** SPADL actiontypes `{"shot", "shot_penalty", "shot_freekick"}` (verified in `spadl/config.py`). Penalty/free-kick subtypes included by default; exposed as a trainer parameter so they can be excluded for sensitivity analysis.
- **No linkage for the label (R2).** Actions and frames share the **same per-period `time_seconds`** base (both schemas carry `time_seconds: float64`; TF-24 confirmed action↔frame time alignment, link-rate ≈ 0.98). So the label compares each shot **action's own `time_seconds`** directly against each frame's `[t, t + horizon_seconds]` window for the same `(game_id, period_id, team)` — **no `link_actions_to_frames` step for labelling**. Routing through linkage first would inject the ±tolerance link smear (default 0.2 s) into the label boundary, which is material at a 1 s horizon — the exact boundary effect this design avoids. (Linkage is still used elsewhere — `add_*`/`*_xfns` map actions→frames at serve time — just not for the training label.)
- **Training domain filter** (paper's "clear possession in the attacking third"): keep frames where (a) `ball_state == "alive"`, (b) a possession team is resolvable via `derive_team_in_possession` (TF-5), and (c) the ball is in the in-possession team's attacking third. `attacking_third_only` + the third threshold are trainer params; default = attacking third.

**Label edge semantics (D5):**
- The window is **inclusive of `t`**: a frame at which a shot is already occurring labels positive (xS ≈ 1 there) — the model is asked "is a shot happening now or imminently?".
- Labelling is **relative to the team in possession at `t`**. If possession turns over inside the horizon and the *opponent* shoots, that is a **NEGATIVE** for the team in possession at `t` (their possession produced no shot). Possession at `t` is fixed by `derive_team_in_possession` at `t`.

**Boundary-effect note (paper §5.2):** a 1-second horizon makes labels sensitive to small timestamp misalignment. Labelling on a per-period `time_seconds` window (deterministic given linkage) is the mitigation; a unit test exercises a **non-contiguous `frame_id` gap** to prove the label is gap-robust (§10.1).

---

## 5. Features — `faithful` this PR (`extended` deferred)

The variant axis is the **feature set** (this differs from ghost-gk's data-size axis). `XShotFeatureSet = Literal["faithful", "extended"]`, **default `"faithful"`**. **This PR implements `"faithful"` only**; `"extended"` raises `NotImplementedError` (documented extension point). The Literal + the data-driven extractor structure ship now because they shape the model/metadata API; `feature_set` is recorded in model metadata and re-checked at `load()`/inference. The two sets would produce different input dimensionality → genuinely different trained models, so introducing `extended` later is additive (new variant, new weights), not a breaking change.

### 5.1 `faithful` (paper Appendix A, Table 11 — 27 features)

All in **goal-relative coordinates** (defending goal of the *attacked* end at a canonical origin) via the **shared `_geometry` helper** (C5), so LTR/RTL frames map identically. Note: ghost-gk's `to_gr_x`/`to_gr_vx` are private closures inside `extract_ghost_gk_features` (not importable); this PR extracts the transform into `silly_kicks/tracking/_geometry.py` as a shared primitive that both xS and (in a follow-up refactor) ghost-gk can consume — following the TF-15 precedent of promoting shared primitives (`compute_tti`, `select_back_line_players`). Refactoring ghost-gk to use the shared helper is an opportunistic, behaviour-preserving cleanup guarded by ghost-gk's existing tests (National-Park principle); if it risks scope creep it is deferred and only xS consumes the new helper.

| Group | Features |
|---|---|
| Ball (4) | `r` (dist ball→goal centre), `theta` (angle ball→goal centre), `z` (height), `speed` |
| Goal (1) | `openGoal` ∈ [0,1] — unobstructed goal-mouth share |
| Goalkeeper (2) | `GK_r`, `GK_theta` (defending GK dist/bearing to goal centre) |
| Defenders (10) | `DefDist_{0..4}`, `DefAngle_{0..4}` — 5 nearest **non-GK** defenders, dist+bearing **from the ball** |
| Attackers (10) | `OffDist_{0..4}`, `OffAngle_{0..4}` — 5 nearest attackers **excluding the carrier**, dist+bearing from the ball |

- **`openGoal` construction** (paper Appendix A): model each non-GK defender **between ball and goal** as a 75 cm-diameter circle; compute the ball→defender tangent-line pair; the goal-line segments subtended by tangent pairs are "obstructed"; `openGoal` = unobstructed fraction of the goal mouth. Own small helper (`_open_goal_fraction`), NOTICE-attributed. The GK is **excluded** as an occluder (paper's choice — keeps `openGoal` a defender-structure feature). **Correctness requirements (C4):**
  - Obstructed goal-line intervals from multiple defenders are **UNIONed, not summed** (overlapping shadows must not double-count — the classic occlusion bug). Implementation: collect `[lo, hi]` intervals, merge overlaps, sum merged lengths.
  - Only defenders **between ball and goal** cast a shadow: a defender *behind the ball* (farther from goal than the ball) or *past the goal line* casts none.
  - Validated by **exact-value golden-master tests** on hand-computed configurations (bekkers-golden-master pattern), not only qualitative bounds — see §10.1.
- **Fewer than 5 defenders/attackers** (early frames, sparse providers): missing slots → NaN. XGBoost handles NaN natively (no fillna — matches the calibration M3 "NaN passthrough" rule).

### 5.2 `extended` — deferred extension point (NOT implemented this PR)

`"extended"` would add features from shipped primitives — pitch-control share at the ball (TF-7), defensive-line geometry (TF-14), GK-influence primitives (TF-15) — computed through a passed `pitch_control_cache` for amortised cost (ADR-008). It is **deferred** (C2): it adds primitive coupling untested on real data in this PR and carries the canonical-vs-counterfactual cache trap (§7). The extractor is structured so the feature list is data-driven from `feature_set`, and `"extended"` raises `NotImplementedError` with a pointer to the follow-up. Introducing it later is purely additive (new variant + new weights). This avoids speculative-surface debt while keeping the API shape stable.

### 5.3 Shared train/serve extractor

`extract_xshot_features(frame_data, *, feature_set, ...) -> 1-row DataFrame` is the single code path used by **both** training (`prepare_xshot_training_data`) and inference (`compute_xshot_occurrence`). This is the anti-skew guarantee — there is no second feature implementation that could drift.

---

## 6. Model & hyperparameter optimisation

### 6.1 Model — pinned deterministic XGBoost

Matches the house standard (`calibration/_vaep_brier_objective._xgb_classifier`): `xgb.XGBClassifier(tree_method="hist", n_jobs=1, subsample=1.0, colsample_bytree=1.0, random_state=seed, eval_metric="logloss")`. Deterministic so the HPO cache-equivalence gate holds.

**Not** sklearn HistGB and **not** a hand-rolled numpy traversal — the model is XGBoost both because it is the paper's choice and because it matches the existing ruthless objective; inference uses xgboost natively (decision: §11 resolved).

### 6.2 HPO via ruthless `CachedObjective`

The clean invariant/patch split for this problem:
- **`prepare()` (trial-invariant, expensive):** extract the feature matrix + labels + `game_id` groups once per fold (feature extraction dominates cost; identical across trials).
- **`evaluate_patch(inv, candidate)` (trial-varying, cheap):** fit XGBoost with the candidate hyperparameters and run match-stratified `GroupKFold` CV; return mean held-out **log-loss** (paper's primary metric) as the `Direction.MINIMIZE` objective, **plus PR-AUC and Brier as secondary diagnostics** in the `Metrics` dict (C6). Under the extreme imbalance (~0.02 positive rate) log-loss is minimised by near-constant predictors, so PR-AUC-vs-baseline is the meaningful quality signal and becomes the real acceptance gate in the weights follow-up (§10.3); it is tracked from day one so the diagnostic history exists.
- **`patch_params`** = the searched XGBoost hyperparameters: `frozenset({"n_estimators", "max_depth", "learning_rate", "min_child_weight", "scale_pos_weight", "reg_lambda"})` (exact space in the plan).
- **`assert_cache_equivalence` (1e-9):** an independent monolithic `evaluate` recomputes features+fit so the fast path is proven non-tautological — same guarantee TF-24 ships.

`OptunaStrategy(cfg, seed=42).run(objective, backend=InProcessBackend())` with `sampler="tpe"`, `warm_start` from sensible defaults, `store=StoreConfig(kind="sqlite", path=<fresh>)`.

### 6.3 Class imbalance

xS positives are rare (a shot in the next second is a tiny fraction of in-possession attacking-third frames; the paper's ~0.023 log-loss reflects this). Strategy:
- **Primary: `scale_pos_weight`** ≈ (neg/pos ratio), itself a searched hyperparameter. Deterministic, no row dropping, keeps the full negative distribution.
- **Optional: `negative_subsample`** (trainer param, default off) for wall-clock control on very large corpora. **If used it MUST be seeded** (`seed` param → a local `np.random.default_rng(seed)`), else it breaks reproducibility *and* the `prepare()`-side of cache-equivalence (D2). When used it is logged so coverage reduction is never silent (no-silent-caps rule). Recommend leaving off for the maintainer run.

### 6.4 Cross-validation

Match-stratified `GroupKFold(5)` with `groups = game_id` (never train and test on the same match) — same anti-leakage rule as ghost-gk and the calibration CV.

### 6.5 Carrier-param coupling — record AND consume at inference (C3 + R3)

xS's possession resolution (which frames are in-possession → labelled and scored, and which goal is the "attacked" end for the goal-relative origin) flows through `infer_ball_carrier(frames, tolerance_m=, beta=, gamma=)` → `derive_team_in_possession`. The carrier params live on `infer_ball_carrier` (verified: `derive_team_in_possession` is a parameterless merge of the carrier output). These are **the very defaults the live TF-24 sweep is re-tuning**.

**Record (training):** the trainer writes the exact `tolerance_m`/`beta`/`gamma` used into `metadata.json`.

**Consume (inference) — R3:** `compute_xshot_occurrence` (and `add_*`/`*_xfns`) must resolve possession by calling `infer_ball_carrier` with the carrier params **read from the model's metadata**, NOT the then-current library defaults. Recording alone is insufficient: if the TF-24 apply-PR changes the library default carrier params after this model was trained, inference using live defaults would select a different in-possession frame set and could flip the goal-relative orientation → train/serve skew even with no retrain. Reading the params from metadata makes serve-time possession resolution identical to train-time by construction.

**Retrain trigger:** the model card / CHANGELOG note still state that a carrier-default change (the TF-24 apply-PR) is a retrain trigger to *re-fit* on the new regime; the metadata-consume rule guarantees correctness *until* that retrain. Test: a model carrying non-default carrier params drives `infer_ball_carrier` with those params at compute time (§10.1 `test_inference_uses_metadata_carrier_params`).

---

## 7. TF-19 integration path

TF-19 needs the **per-frame probability surface**, not action-coupled scalars. The clean boundary:
- TF-19 calls `compute_xshot_occurrence(frames, model=..., home_team_id=..., link_frame_ids=...)` and reads the `xshot_occurrence` column — or calls `XShotOccurrenceModel.predict_proba(features)` directly on a feature matrix it controls (object API for spatial-composition consumers; DataFrame API for action-coupled consumers — same split ghost-gk established).
- The counterfactual `P(shot | ghost_GK)` is computed by TF-19 substituting the ghost GK position into the frame state before feature extraction; **xS itself stays counterfactual-agnostic** (it just scores whatever frame it is given). This keeps TF-16 a clean, reusable surface.
- **Canonical-vs-counterfactual cache rule (B2).** ADR-008's `pitch_control_cache` memoizes **canonical** per-frame surfaces only (counterfactual, player-moved surfaces are deliberately never cached — same rule cover_shadows/space_creation follow). So when `extended` eventually ships, a counterfactual caller (TF-19 moving the GK) MUST NOT pass a `pitch_control_cache` to `compute_xshot_occurrence` — a cached canonical surface for a non-canonical frame would yield a wrong xS. The `faithful` variant shipped in this PR uses no pitch control, so the trap cannot fire yet; the rule is documented now so `extended` inherits it. (Additional reason `faithful`-only is the safe scope for this PR.)
- **Honest caveat carried forward (paper §4.2):** `r` (ball→goal distance) dominates xS; `openGoal` and GK features contribute little. A faithfully-replicated xS may therefore be only weakly GK-sensitive, which could make TF-19's xS-arm `Δ` small. This is the empirical motivation for the `extended` variant and for TF-19's complementary DAS / cover-shadow arms — but it is a TF-19-time finding, not a reason to diverge from the paper here. **Recorded as a design risk, not a blocker.**

---

## 8. Serialization

Pickle-free, mirroring ghost-gk's contract:
```
xshot_occurrence_v1/
├── model.json        # xgboost native booster dump (Booster.save_model)
├── metadata.json     # feature_names, feature_set, horizon_seconds, shot_types,
│                     #   hyperparams, silly_kicks + xgboost versions, schema version
└── SHA256SUMS        # per-file integrity hashes (CRLF→LF normalised for .json)
```
`load()` verifies SHA-256 then `xgb.Booster(); booster.load_model(...)`. Tampered artifact → `IntegrityError`. `feature_set` in metadata is authoritative; inference asserts the extractor matches it.

---

## 9. Shipping scope — code + fixture only (collision-safe, staged)

This PR ships the **complete code path** but **no trained weights**:
- ✅ module, model class, objective, training CLI, all three surfaces, atomic mirror, NOTICE, ADR, full test suite.
- ✅ a tiny **synthetic** CI fixture (hand-built frames+actions) exercised by unit + integration tests, including a **3-trial Optuna smoke** and a **fit-on-fixture** round-trip.
- ❌ no maintainer training run; ❌ no bundled or Hub weights.

**Rationale:** training requires the gated multi-provider tracking corpus the live TF-24 sweep is currently pulling; running it now would contend with that sweep (and is heavy). Deferring weights to a follow-up is exactly the staged path TF-18 used (model code 3.19.0 → training pipeline 3.20.0 → bundled weights 3.24.0).

**Consequences:**
- `from_variant("default")` raises a clear `FileNotFoundError` ("xS weights not yet bundled; train via scripts/train_xshot_occurrence.py or await the weights follow-up") until the follow-up. `from_hub` is wired but the repo is empty until then.
- CI-enforced docstring **Examples** therefore **fit-on-a-tiny-fixture** rather than load bundled weights (so they execute without network/weights). Documented as the example idiom.
- **`xshot_occurrence_xfns` is NOT added to any default/union xfn list** (`tracking_default_xfns`, `*_full_default_xfns`, etc.) until weights ship (D1). Shipping it inert/erroring into a real VAEP pipeline would surprise consumers; the factory exists and is tested, but wiring it into the default feature set is part of the weights follow-up.
- CHANGELOG notes the model ships **untrained**; the follow-up PR adds the maintainer run + weights + the empirical acceptance gates from §10 + wiring into the default xfn lists.

---

## 10. Testing strategy (TDD — tests authored before implementation)

### 10.1 Unit (`tests/tracking/test_xshot_occurrence.py`)

| Test | Validates |
|---|---|
| `test_extract_features_faithful_shape` | 27 columns, names, dtypes, goal-relative normalization |
| `test_extended_raises_not_implemented` | `feature_set="extended"` → `NotImplementedError` with follow-up pointer |
| `test_goal_relative_symmetry` | LTR vs RTL frame → identical goal-relative features (via shared `_geometry` helper) |
| `test_open_goal_no_defenders_is_one` | no defenders between ball and goal → `openGoal == 1.0` |
| `test_open_goal_full_wall_is_zero` | defenders fully spanning the mouth → `openGoal == 0.0` |
| `test_open_goal_golden_masters` | **exact** values on 2–3 configs whose reference numbers are **independently derived from first-principles geometry** (single defender's shadow = goal-line segment subtended by the ball→defender tangent pair, computed analytically in the test), NOT copied from the implementation's output — so it is a correctness check, not a change-detector (R5). One worked derivation in the test docstring. |
| `test_open_goal_overlapping_shadows_unioned` | two overlapping defender shadows → length UNIONed, not summed (no double-count) |
| `test_open_goal_defender_behind_ball_no_shadow` | defender farther from goal than ball casts no shadow |
| `test_open_goal_defender_past_goal_line_no_shadow` | defender beyond goal line casts no shadow |
| `test_open_goal_grazing_angle` | ball at extreme angle → finite, in-bounds |
| `test_open_goal_bounds` | `openGoal` ∈ [0,1] across random configs (hypothesis property test) |
| `test_fewer_than_5_players_nan_slots` | < 5 defenders/attackers → trailing slots NaN, no crash |
| `test_label_horizon_via_time_seconds` | shot at `t+0.8s` positive within 1.0s window; `t+1.5s` negative; uses `time_seconds` |
| `test_label_robust_to_noncontiguous_frame_id` | **B1:** inject a `frame_id` gap (dropped frames) with intact `time_seconds` → label unchanged; proves no `frame_id` arithmetic |
| `test_label_no_period_bleed` | window does not cross a `period_id` boundary |
| `test_label_inclusive_of_t` | shot at the frame itself → positive (D5) |
| `test_label_turnover_opponent_shot_is_negative` | possession turns over, opponent shoots in-horizon → NEGATIVE for team-in-possession at t (D5) |
| `test_label_shot_types` | penalty/freekick shots counted; togglable |
| `test_model_fit_predict_proba` | `.fit().predict_proba()` → shape (n,), values ∈ [0,1] |
| `test_model_deterministic` | two fits, same seed/data → identical predictions (pinned XGB) |
| `test_model_save_load_roundtrip` | save→load→predict matches; booster JSON, no pickle |
| `test_model_sha256_verification` | tampered `model.json` → `IntegrityError` |
| `test_feature_set_metadata_roundtrip` | `feature_set` persisted + re-checked on load |
| `test_compute_xshot_no_model` | clear error when no model + Hub inert |
| `test_add_xshot_aggregator` | one `xshot_occurrence` column; `links` kwarg; provenance skip guard |
| `test_add_xshot_nan_safe` | NaN identifier rows → NaN output, no crash (ADR-003 gate also auto-discovers it) |
| `test_xshot_xfns_factory` | `_a0/_a1/_a2` columns; silent NaN on `frames=None` |
| `test_xfns_frame_aware_marker` | transformer has `_frame_aware = True` |
| `test_inference_uses_metadata_carrier_params` | **R3:** a model with non-default `tolerance_m`/`beta`/`gamma` in metadata drives `infer_ball_carrier` with those params at compute time, not library defaults |

### 10.2 Integration (`tests/tracking/test_xshot_occurrence_integration.py`)

| Test | Validates |
|---|---|
| `test_add_xshot_dtype_mismatch` | int64 actions + str frames → no crash (PR-S53 pattern) |
| `test_atomic_mirror` | `atomic.tracking.features.add_xshot_occurrence` same columns |
| `test_train_script_smoke` | `scripts/train_xshot_occurrence.py` on synthetic parquet (3 trials, fresh tmp store) → exits 0, writes `model.json` + `metadata.json` + `SHA256SUMS`. **Invoked as `python -m` / module, OR with `cwd` + `PYTHONPATH` set so the editable install is importable in the subprocess** (avoids the ghost-gk subprocess-import trap seen this session). |
| `test_objective_cache_equivalence` | `assert_cache_equivalence` fast-path == monolithic to 1e-9 |
| `test_optuna_smoke_3_trials` | `OptunaStrategy` runs 3 TPE trials on the fixture; returns finite log-loss + PR-AUC/Brier diagnostics; fresh store |
| `test_carrier_params_in_metadata` | trainer writes the carrier params used into `metadata.json` (C3) |
| `test_tf19_interface_stub` | `predict_proba` on a feature matrix shaped as TF-19 will pass it → finite, in-bounds |

### 10.2a Real-provider extraction tests — **regular suite, NOT deferred (B3)**

The single most important coverage given this codebase's recent history: every TF-24 sweep bug this week (IDSSE `game_id=None`, Gradient Sports 16× duplicate frames, coordinate/ball conventions, dtype-asymmetric `player_id`) surfaced **only on real multi-provider data** — skillcorner-only/synthetic fixtures hid all three. These tests need **no trained weights** (they exercise extraction + a fixture-FIT model), so they ship now using the committed slim real-provider fixtures `tests/datasets/tracking/action_context_slim/{sportec,metrica,skillcorner,pff}_slim.parquet` (already used by `test_action_context_cross_provider.py`). Run in the **regular** suite (not `@pytest.mark.e2e`) so provider-shape pathologies are always exercised.

| Test (`tests/tracking/test_xshot_occurrence_real_data.py`) | Validates |
|---|---|
| `test_extract_features_real_providers[sportec/metrica/skillcorner/pff]` | `extract_xshot_features` runs on each real slim fixture: no crash; `openGoal` ∈ [0,1]; `r`/distances ≥ 0; angle features in radian range; sane feature ranges |
| `test_compute_xshot_real_providers[...]` | `compute_xshot_occurrence` (fixture-fit model) on each provider → `xshot_occurrence` ∈ [0,1] or NaN, no crash |
| `test_goal_relative_symmetry_real_data` | goal-relative symmetry holds on real LTR+RTL frames (both periods) per provider |
| `test_real_provider_dtype_asymmetry` | int64 (gradientsports) + object (kloppy) `player_id`/`team_id` both handled |

### 10.3 E2e (`@pytest.mark.e2e`) — **assertions land in the weights follow-up**

Authored now as **skipped placeholders** (xfail/skip with reason "weights deferred to follow-up") so the contract is visible; the follow-up flips them on with real data:

| Test | Validates (follow-up) |
|---|---|
| `test_xshot_gradientsports_e2e` | full pipeline + acceptance: `log_loss < uniform_baseline`; **PR-AUC > positive-rate baseline** (the meaningful gate under ~0.02 positives — plain `AUC > 0.5` is nearly free, R4); per-fold std bounded |
| `test_xshot_cross_provider` | trains on ≥2 providers; no single-provider degradation |

### 10.4 Invariants (`tests/invariants/`)
- `xshot_occurrence ∈ [0,1]` wherever non-NaN (physical-probability invariant; fixture density must include positive and negative frames).

### 10.5 Performance budget
- A `pytest-benchmark` guard on `extract_xshot_features` per frame (flat budget from worst observed CI timing × 1.5 headroom). The `measure-before-optimize` skill applies if the extractor is later touched.

---

## 11. Open questions — resolved in review round 1

1. **Q1 (training-extra placement) — RESOLVED:** a new **generic `[train]` extra** (`ruthless-efficiency[optuna]>=0.2.1`, `xgboost>=2.0,<3.0` — same pins `[calibration]` uses), shared by calibration/xshot/future trainers, rather than a per-feature `[xshot]` extra (avoids extra-sprawl; D3). See §2.
2. **Q2 (inference-extra) — RESOLVED:** the *inference* path gates on the **existing `[xgboost]` extra** (verified present: `xgboost = ["xgboost>=2.0.0"]`), with `ImportError("pip install silly-kicks[xgboost]")`. See §2.
3. **Q3 (`extended` this PR) — RESOLVED: `faithful` only** (C2). Keep the `XShotFeatureSet` Literal + data-driven extractor; `"extended"` is a `NotImplementedError` extension point. See §1.1, §5.
4. **Q4 (ADR) — RESOLVED: standalone ADR-011** "Trained-model feature lifecycle" (code → training pipeline → bundled/Hub weights staging), documenting the pattern once for ghost-gk (1st), xS (2nd), and future trained features — rather than an ADR-005 amendment. xS also reuses ADR-005 (tracking-aware feature surfaces) + ADR-008 (pitch-control cache, for the future `extended`) + the ADR-009 ruthless pattern. See §13.

### Resolved in review round 2
- **`[train]` extra name:** keep `[train]` (no bikeshed to `[hpo]`). (R6)
- **Shared `_geometry` helper scope:** ship `_geometry.py` consumed by **xS only** in PR-S75; the ghost-gk refactor onto it is a **separate small follow-up PR** (R6). Rationale: rewiring ghost-gk — a shipped, weight-bearing feature with golden/snapshot tests — inside PR-S75 widens the blast radius for no functional gain here. §5.3 already frames the ghost-gk refactor as deferrable; this confirms it is deferred. The National-Park cleanup gets its own clean PR.

---

## 12. Academic attribution (ADR-005 / ADR-011)

NEW NOTICE entry under "Mathematical / Methodological References":

> The xShotOccurrence model (`silly_kicks/tracking/_xshot_occurrence.py`, TF-16) implements the shot-occurrence (xS) component of: Pipping, J., Feng, T., & Sabin, P. (2026). "Beyond Expected Goals: A Probabilistic Framework for Shot Occurrences in Soccer." arXiv:2512.00203. Only the xS sub-model (probability a shot is attempted within ~1 s of a frame) is implemented; the paper's xG and xG+ composition are out of scope (silly-kicks values goals/threat via VAEP and xthreat). The `openGoal` goal-mouth-obstruction feature follows the paper's Appendix A construction.

Plus a `See NOTICE for full bibliographic citations.` line in each public docstring. (Anzer & Bauer 2021 and Lucey 2014 are already cited in NOTICE and remain relevant cross-references for shot-prediction features.)

---

## 13. Version & release

- **Version: 4.1.0 — PROVISIONAL (C1).** New feature → minor bump. **But the TF-24 apply-PR** (updating `infer_ball_carrier`/`k3`/off-ball default constants from the running sweep) is also queued off main (4.0.3) and is a behaviour-change minor that will likely also want 4.1.0. **Reconcile at merge: whichever of {TF-16, TF-24-apply} merges second re-bumps** (to 4.2.0, or 4.1.x if the first took a different number). Do not hard-code 4.1.0 in code until the merge order is known; the version-bump-hard-gate (all four files in sync) still applies at commit time with whatever number is correct then.
- Bump all four in sync: `pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md` ("Current release"), `CHANGELOG.md` (new dated section).
- **CHANGELOG `### Added`:** xShotOccurrence (xS) model (TF-16, GKDV Layer 2) — ships **untrained** (code + synthetic CI fixture + real-provider extraction tests); maintainer training run + bundled/Hub weights + default-xfn-list wiring to follow. Note the xS↔carrier-param coupling (C3): a future TF-24 apply-PR carrier-default change is an xS retrain trigger.
- **TODO.md:** TF-16 row moves from "On Deck" to reflect code-shipped / weights-pending; GKDV program note updated (TF-16 Layer-2 code landed).
- **ADR: new ADR-011** "Trained-model feature lifecycle" (next free number after ADR-010). Documents the code → training-pipeline → bundled/Hub-weights staging used by ghost-gk and xS so future trained features follow one pattern.
```
