# ADR-011: Trained-model feature lifecycle (code → training → bundled/Hub weights)

| Field | Value |
|---|---|
| **Date** | 2026-05-31 |
| **Status** | Accepted (silly-kicks **4.1.0**, provisional — reconcile with the TF-24 apply-PR at merge) |
| **Deciders** | Karsten S. Nielsen, Claude Opus 4.8 (1M); TF-24/sweep session (3-round spec + plan review) |

## Context

silly-kicks now ships more than one *trained-model* feature — a feature whose
runtime behaviour depends on weights fit from data, not just code. Ghost-GK
(TF-18) was the first; xShotOccurrence (TF-16) is the second; the GKDV program
(TF-16/17/18/19) will add more. Each needs the same lifecycle decisions:

- How are weights serialized? (silly-kicks has **zero pickle/joblib** usage and
  must keep it that way.)
- How are weights distributed without inflating the PyPI wheel past its 100 MB
  per-file limit?
- How does inference declare its dependency without making `import silly_kicks`
  pull heavyweight ML libraries?
- How do training-time-only dependencies (Optuna/ruthless) stay out of the
  inference path?
- How is a feature shipped *before* its weights exist, so code review and
  consumer integration can proceed without waiting on a multi-hour maintainer
  training run on gated data?

Deciding these per-feature produced drift (ghost-gk uses ONNX + npz; the
calibration harness uses booster JSON). This ADR fixes one pattern.

## Decision

A trained-model feature follows a **staged lifecycle** across PRs:

1. **Code PR** — ships the feature's pure code: a shared train/serve feature
   extractor (the anti-skew guarantee), a `*Model` class, the ADR-005 surfaces
   (`compute_*` / `add_*` / `*_xfns`), an HPO objective, a training CLI, and a
   full test suite including **real-provider extraction tests in the regular
   suite**. It ships **untrained** — `from_variant`/`from_hub` are wired but
   raise a clear `FileNotFoundError` until weights exist; a synthetic CI fixture
   + a 3-trial HPO smoke + a fit-on-fixture round-trip exercise the whole path.
   The feature's `*_xfns` is **NOT added to any default/union xfn list** until
   weights ship (so an inert/erroring feature is never wired into a real VAEP
   pipeline).
2. **Weights PR (follow-up)** — runs the maintainer training sweep on the gated
   multi-provider corpus, bundles a small default model in the wheel, hosts the
   full model on HuggingFace Hub, adds empirical acceptance gates (e.g.
   PR-AUC-vs-baseline for severely imbalanced targets), and wires the feature
   into the default xfn lists.

**Serialization** is pickle-free, in the model's native non-executable format
plus a JSON metadata sidecar and a `SHA256SUMS` integrity manifest verified on
load (CRLF→LF normalised for `.json`). For an XGBoost model that is the native
booster JSON; for a tree-density model (ghost-gk) it is npz tree-node arrays.

**Dependency placement:**
- **Inference** gates on the model's own runtime extra (e.g. `[xgboost]` for an
  XGBoost model, `[ghost-gk]`/`onnxruntime` for ghost-gk), lazily imported
  inside functions so `import silly_kicks` stays dependency-light (CI-guarded by
  a fresh-subprocess import-isolation test).
- **Training/HPO** dependencies (`ruthless-efficiency[optuna]` + xgboost) live
  under a single generic **`[train]`** extra, shared by all trainers rather than
  a per-feature extra (avoids extra-sprawl). The HPO objective module is never
  imported by `silly_kicks/__init__` or by the inference path.

**Metadata records training-time coupling.** Where a model's feature selection
depends on other tunable defaults (e.g. xS resolves possession via
`infer_ball_carrier`'s `tolerance_m`/`beta`/`gamma`), the exact values used are
written into `metadata.json` AND **consumed at inference** (read from metadata,
not the live library default), so a later change to those defaults cannot create
train/serve skew without an explicit retrain.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| Per-feature ad-hoc lifecycle | Flexible | Drift (ghost-gk ONNX vs calibration booster JSON); each feature re-derives serialization, extras, staging | The drift this ADR exists to stop |
| Ship code + weights in one PR | One PR | Training needs the gated corpus + multi-hour compute; blocks code review on data availability; contends with in-flight sweeps | Staged shipping (TF-18 precedent) decouples the two |
| Per-feature `[xshot]`/`[ghostgk-train]` extras | Precise | Extra-sprawl; every trained feature adds one | A single `[train]` extra suffices |
| pickle/joblib weights | Trivial | Arbitrary-code-execution on load; silly-kicks has zero pickle usage | Security non-starter |

## Consequences

### Positive
- One documented pattern for every trained-model feature (ghost-gk retrofits
  conceptually; xS is the first built to it).
- `import silly_kicks` stays dependency-light; training deps are opt-in.
- A feature lands functionally complete and reviewable before its weights exist.
- Metadata-consumed coupling params make train/serve consistency structural.

### Negative
- A trained-model feature spans two PRs (code, then weights) — more process, but
  it matches how TF-18 actually shipped.
- The code PR ships an inert `from_variant`/`from_hub` until the weights PR.

### Neutral
- Inference runtime extras differ by model kind (`[xgboost]` vs `onnxruntime`);
  that is inherent to the model, not a lifecycle choice.

## Update — TF-16 weights cycle (4.9.0, PR-S80)

The xS weights run executed this lifecycle's "bundled/Hub weights" stage and refined two points:

- **Single bundled variant for a booster, decided by evidence — not a size axis.** Unlike ghost-GK
  (whose KDE artifact scales 9 MB ↔ 91 MB, justifying a `default`/`full` split), an xS XGBoost booster
  is ~1.2 MB regardless of corpus, so the variant axis is **data provenance**, not size. A
  pre-registered two-candidate comparison (`public` vs `full`, common public held-out, shared
  hyperparameters) found owner-tier data *degraded* public-provider generalization, so a **single
  `public` variant shipped**. The lesson: for size-invariant artifacts, justify any multi-variant
  surface with measured generalization, not a presumed size trade-off.
- **Model metadata carries a coordinate/units template** (`pitch_length`/`pitch_width`/
  `geometry_version` + `xgboost_version`/`training_platform` + `shipped_variant`/`provider_list`), and
  `load()` **fails closed** on a pitch-dimension/unit mismatch (warns on a translation-only geometry
  change). This is the template a future ghost-GK R3 refit and any TF-38 coordinate change inherit.

## Update — TF-17 (xCrossAttempt), 4.11.0

xCrossAttempt is the **third** feature built to this lifecycle (after ghost-GK and xS) and follows
the same staged code→weights path. Two refinements it introduces:

- **Three-PR split (vs the usual two).** A trained-model feature whose research novelty needs an
  *offline* validation distinct from the runtime surface splits into: **PR-A** code (untrained) →
  **PR-B** weights + shipped-surface validation (incl. the surface GK-block ablation + the
  GK-substitution-sensitivity probe that determines TF-19 viability) + default-xfn wiring → **PR-C**
  the causal research harness (a private `silly_kicks/_causal/` matching port + a thin script driver)
  + its own ADR-015. Rationale: a one-shot research artifact must not gate (or share a release with)
  a maintained runtime feature, and TF-19 must be able to consume the surface without waiting on a
  research study.
- **Optional match-context inputs.** Where a confounder needs match context not present in a single
  frame (xCross's `score_differential`), the `compute_*`/`add_*` surfaces take an optional `actions=`
  kwarg and degrade gracefully to NaN (XGBoost-tolerant) when it is omitted — reusing ghost-GK's
  `_build_score_lookup`. Faithfulness caveats (here: only 7 of the paper's 8 confounders are
  realized) are recorded in NOTICE rather than overclaimed.

## Note — temporal-leakage discipline (external validation)

A trained-model feature in this lifecycle is **leakage-corrected by construction**: its shared
train/serve extractor uses only information available at the decision moment (xS/xCross use
goal-relative geometry of the frame at time *t* — no post-event descriptors), and CV is
**match-grouped** (`StratifiedGroupKFold` by match; TF-24 `GroupKFold(5)` / leave-one-match-out) so
temporally adjacent frames cannot straddle train/test folds. Peters, Parmar, Davies & James (2026,
"When Timing Matters: Evaluating Temporal Leakage in ML Models of Football Pass Turnovers," *Research
Quarterly for Exercise and Sport*, DOI:10.1080/02701367.2026.2677718) independently quantifies why
both matter (post-execution features inflate ROC-AUC by ~0.14 mean, tree models most affected) and
endorses match-grouped CV as the mitigation. silly-kicks already embodies both; `HybridVAEP`
(`silly_kicks/vaep/hybrid.py`) additionally supplies the result-leakage-free action-value path for
the retrospective-vs-prospective use-case split the paper draws. **No build followed — the paper
validates the existing design** (audit, 2026-06-05).

## Related

- **Specs:** `docs/superpowers/specs/2026-05-31-tf16-xshot-occurrence-design.md`;
  `docs/superpowers/specs/2026-06-02-tf16-weights-run-design.md` (the weights cycle)
- **Plans:** `docs/superpowers/plans/2026-05-31-tf16-xshot-occurrence.md`;
  `docs/superpowers/plans/2026-06-02-pr-s80-tf16-xshot-weights.md`
- **ADRs:** ADR-005 (tracking-aware feature surfaces); ADR-009 (ruthless
  `CachedObjective` calibration harness — the HPO substrate this reuses).
- **Features:** TF-18 ghost-gk (first trained-model feature; pre-dates this ADR);
  TF-16 xShotOccurrence (first built to this lifecycle).
- **External:** Pipping/Feng/Sabin 2026 (arXiv:2512.00203); Peters/Parmar/Davies/James 2026
  (DOI:10.1080/02701367.2026.2677718 — temporal-leakage validation, see note above);
  `ruthless-efficiency` (PyPI); HuggingFace Hub `silly-kicks` org.
