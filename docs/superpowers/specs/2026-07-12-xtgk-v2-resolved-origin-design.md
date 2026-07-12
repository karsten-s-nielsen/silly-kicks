# xT-GK v2 resolved-origin fix — design

- **Date:** 2026-07-12
- **Target release:** silly-kicks 4.46.0
- **Decision record:** ADR-036 amendment (no new ADR)
- **Status:** rev 3 — incorporates the part-deux cross-session reviews (round 1: F1–F8; round 2:
  R1–R6). Round 2 approved the design shape; R1/R2 were contract semantics to settle before
  implementation, and are settled below.

---

## 1. Problem

xT-GK v2 scores a large fraction of its own domain at a **fabricated origin**, and the
resolved coordinates it should have used were already materialised in the gold table the
loader queries.

### 1.1 Root cause (one sentence)

`xtgk._possession_value.flat_zones` maps NaN coordinates to `(0.0, 0.0)` — i.e. **flat zone
176**, the own-corner cell — on the documented assumption that *"Such rows are dropped
downstream"* (`_possession_value.py:47`).

That assumption holds at **every fitting seam**, each of which drops NaN coords itself:

| seam | file:line | drops NaN coords? |
|---|---|---|
| move transitions (Singh) | `_moves.py:70` | yes |
| move transitions (KDE, success-filtered) | `_moves.py:89` | yes |
| xG first-shot reward | `_xg_reward.py:36` | yes |
| Markov support counts | `_markov.py:106` | yes |
| `EmpiricalPossessionValue.fit` | `_empirical.py:87` | yes |
| `EmpiricalTurnoverValue.fit` | `_turnover.py:131` | yes |

The fit seams drop NaN in **two different orders**, and both are safe (R6). Some drop *before*
calling into the grid (`_moves.py`, `_xg_reward.py`, `_markov.py:106`, `_empirical.py:87`,
`_turnover.py:131`); the three that call `flat_zones` **directly** — `_markov.py:65`,
`_empirical.py:83`, `_diagnostics.py:123` — pass NaN rows *through* it to assign pressure terciles
and then drop them before the surface is solved. Either way **no NaN row reaches a fitted
surface**, which is what §1.5 and T4 rest on.

It is **false at the single scoring seam**, `_metric.py:56-57`, which drops nothing and emits
a real number for every row:

```python
zones_o = flat_zones(actions["start_x"], actions["start_y"], l, w)
zones_d = flat_zones(actions["end_x"], actions["end_y"], l, w)
```

A goal-kick whose true origin is the rule point `(5.5, 34)` → zone **80** is instead scored at
zone **176**.

`zones_o` drives three of the metric's four terms — `position` (ΔV), `retention_loss`
(`−(1−ρ)·V(s)`) and `dzv` (`−(1−ρ)·κ·V_opp(s)`); `zones_d` drives ΔV. (`pev` is identically 0
while `p′ = p`.)

### 1.2 Second defect: the loader never reads the resolved coordinates

`scripts/_loader_databricks.py::load_xtgk_cohort` reads the **raw**
`bronze.spadl_actions.start_x`. But PR-S101 (4.36.0) persisted the exact coordinates v1's grid
lookups used into `dev_gold.fct_action_context` as `xt_gk_origin_x/_y` and `xt_gk_dest_x/_y` —
in the **very table `_XTGK_ACTIONS_SQL` already `LEFT JOIN`s**. It simply never `SELECT`s them.

So the v1 comparator in the 4.45.0 head-to-head (`c.xt_gk`, read from `fct_action_context`) was
computed on **resolved keeper origins** via `resolve_gk_geometry`, while v2 was computed on
**raw/NaN** ones. The head-to-head was never apples-to-apples.

### 1.3 Third defect: ρ's silent imputation

`GkRetentionModel.predict_proba` (`_retention.py:81`) replaces non-finite features with the
**training mean** (`Xf = np.where(np.isfinite(X), X, mean[None, :])`).
`extract_retention_features` derives `length` / `dy_abs` / `forwardness` from the coordinates, so
a NaN-origin row yields a **no-information ρ** — silently, with no warning. ρ multiplies every
term of the metric.

The ρ **trainer** does drop these rows (`train_gk_retention.py:45`:
`keep = isfinite(length) & isfinite(dest_x) & isfinite(y)`), so the bundled weights are not fitted
on fabricated geometry — but that creates a **train/serve domain mismatch**: ρ is trained on
finite-geometry rows only and served on everything.

### 1.4 Measured blast radius (live gold, read-only, 2026-07-12)

#### Origin side

**gradientsports** — 3874 `is_gk_distribution` actions, by v1's `xt_gk_origin_source`:

| origin_source | n | raw `start_x` NULL | resolved coord available |
|---|---|---|---|
| `native` | 2928 | 0 | raw == resolved |
| `tracking_gk` | 298 | 298 | **yes** — mean (7.08, 34.44) |
| `goalkick_prior` | 297 | 297 | **yes** — (5.50, 34.00) |
| `unresolved` | 351 | 351 | no — genuinely unresolvable |

→ **946 / 3874 = 24.4%** scored at fabricated zone 176, including **595 of 988 goal-kicks
(60.2%)**. Of those 946, **595 have a resolved origin sitting unused in gold**; 351 must be
*dropped*, not scored.

**skillcorner** — 5487 actions. Raw `start_x` is **never** NULL, but **1181 rows (21.5%, exactly
its goal-kicks)** carry a native origin that v1 **deliberately distrusts and overrides**: per
ADR-024 / PR-S104 (4.37.0), SkillCorner's native goal-kick origin is the broadcast **ball
detection, not the keeper** (`start_x` SD 23.2 m; only 51% inside the own box).

| origin_source | n | raw NULL | resolved |
|---|---|---|---|
| `native` | 4306 | 0 | raw == resolved |
| `tracking_gk` | 778 | 0 | **overrides raw** — mean x 4.29 |
| `goalkick_prior` | 403 | 0 | **overrides raw** — (5.50, 34.00) |

Coverage is **complete** (2928+351+298+297 = 3874; 4306+778+403 = 5487). No unhandled case.

#### Destination side (F3 — measured; the override is a NO-OP, the fabrication is not)

| provider | dest_source | n | raw `end_x` NULL | rescued | overridden (present-but-differ) |
|---|---|---|---|---|---|
| gradientsports | `native` | 3761 | 0 | 0 | **0** |
| gradientsports | `unresolved` | 113 | 113 | 0 | — |
| skillcorner | `native` | 5487 | 0 | 0 | **0** |

Two distinct conclusions, and they must not be conflated:

- The dest **override** has **zero blast radius** on both cohorts — `xt_gk_dest_x` never differs
  from raw `end_x` and never rescues a NULL. We keep it in the helper for correctness and
  symmetry (a future provider may differ), but claim **no** measured effect. Real data does not
  exercise this path, so it is tested **synthetically** (T2).
- The dest **fabrication is real**: **113 GS rows (2.9%)** had `end_x` NaN, so `zones_d` was also
  fabricated to zone 176 — ΔV was computed to a **fabricated destination**. These are *not*
  rescuable (the resolved dest is NULL too) and become NaN under the guard.

Post-fix, the GS rows the metric NaN-outs are the **union** of {351 unresolved origin} and
{113 unresolved dest} (≤ 464; the exact union is reported by the implementation).

#### Consequence for ρ

The **SkillCorner ρ variant (AUC 0.650) was trained on 1181 goal-kicks with the wrong geometry**
— contaminated weights, not merely missing data. GS's ρ saw only 393 of 988 goal-kicks; the fix
grows that arm to 988 (2.5×).

### 1.5 Ruled out (measured, not assumed)

- **No bronze↔gold coordinate skew.** `dev_gold.fct_action_values.start_x` (read by
  `load_retention_cohort`) is byte-identical to `bronze.spadl_actions.start_x` (read by
  `load_xtgk_cohort`): zero rows differ, mean `|Δx| = 0.0`, NULLs match exactly. ρ trained and
  served on the same coordinates. This hypothesis is **dead**.
- **The deep-zone gate is unaffected.** Every fit path drops NaN coords (§1.1), so the fitted `V`
  surface, its support counts, `EmpiricalPossessionValue` and `EmpiricalTurnoverValue` are clean.
  Its GO-leaning verdict **stands and is not re-run** — asserted by regression test T4, not prose.

### 1.6 Why this is urgent, and why the contamination points *toward* the observed failure

The xT-GK v2 program is **blocked** on an owner/Eyestone decision between two *modelling*
interpretation-forks (V's reward = `E[first-shot xG]` vs Jeff §2.1 remainder-of-possession;
dormant PEV pending receiver-pressure `q`). There is a third, **data** explanation nobody listed.

**The contamination is not neutral noise — it is biased toward the verdict that was observed.**
A fabricated origin is **keeper-independent**: every NaN-origin goal-kick is scored at *the same*
zone 176, regardless of which keeper took it. Injecting a keeper-independent constant into ~24% of
each keeper's actions **compresses between-keeper variance while contributing within-keeper
variance** — and ICC is exactly between ÷ (between + within). So the defect biases the
keeper-discrimination ICC **toward zero**, which is precisely the "keeper-flat" finding that
produced the 4.45.0 negative verdict. Likewise, ρ collapses to the training mean on those rows
(§1.3), removing its keeper-varying contribution to every term.

This is **consistent with** the observed v1-vs-v2 ICC asymmetry (v1 0.019 / 0.018 read *resolved*
origins; v2 −0.002 / 0.011 read *raw* ones). It is **not proof** — v1 and v2 are different metrics
and the comparison is confounded — and this spec does not claim the defect fully explains the
verdict; 75% of rows had correct origins.

The claim is narrower and sufficient: **the 4.45.0 verdict is not safe to act on, and the
contamination runs in the direction of the failure it produced.** Correcting it costs one SQL
change plus a guard, versus re-implementing V.

---

## 2. Design

### 2.1 Library — `silly_kicks/xtgk/`

#### `apply_resolved_gk_geometry` — new public pure function

```python
def apply_resolved_gk_geometry(
    actions: pd.DataFrame,
    *,
    domain_column: str = "is_gk_distribution",
    origin_columns: tuple[str, str] = ("xt_gk_origin_x", "xt_gk_origin_y"),
    dest_columns: tuple[str, str] = ("xt_gk_dest_x", "xt_gk_dest_y"),
) -> pd.DataFrame:
```

Pandas in, pandas out. **No I/O.** Returns a **NEW** frame; never mutates the input.

On rows where `domain_column` is true:

- resolved origin **present** → **OVERRIDE** `start_x`/`start_y`;
- resolved destination **present** → **OVERRIDE** `end_x`/`end_y`;
- resolved coordinate **absent** (NULL) → leave the existing value (NaN for `unresolved`);
- rows **off** the domain are untouched.

**Override, not coalesce — load-bearing.** A coalesce (`fillna`) fixes Gradient Sports (NaN →
resolved) and **silently leaves SkillCorner broken**, whose raw value is *present and wrong*.
§1.4 is the evidence.

Absent-column behaviour is **asymmetric by design**:

- **`domain_column` absent → `ValueError`.** The caller is misusing the helper; treating every row
  as in-domain would override open-play coordinates with GK-distribution geometry
  (`feedback_loud_raise_for_required_input_columns`).
- **Resolved-coordinate columns absent → observable no-op + `warnings.warn`.** A provider or mart
  vintage may legitimately not carry `xt_gk_origin_*`; degrade rather than crash, but never
  silently (`feedback_provider_aware_config_fallback`).

#### `gk_geometry_source` — per-row provenance stamp (F2)

The helper emits a per-row `gk_geometry_source` column:

| value | meaning |
|---|---|
| `off_domain` | `domain_column` false — untouched |
| `native` | in domain; all coords finite; no coordinate changed (raw already equalled resolved) |
| `resolved_origin` | origin overridden; all coords finite |
| `resolved_dest` | destination overridden; all coords finite |
| `resolved_both` | both overridden; all coords finite |
| `unresolved` | in domain; **any** coordinate remains non-finite after resolution |
| `unattested` | the resolved-coordinate columns were **absent**; the helper ran but could not resolve |

**Precedence (R3).** Mixed states are real — a GS row can have a `resolved_origin` *and* a
still-NaN destination (the 298 `tracking_gk` origins overlap the 113 unresolved dests; the ≤464
union in §1.4 proves such rows exist). One single-valued column therefore needs a rule:
**`unresolved` wins whenever any coordinate remains non-finite after resolution.** That makes the
stamp answer the question the NaN guard actually pairs with — *"will this row score?"* — and the
per-side override facts stay recoverable from gold's `xt_gk_origin_source` / `xt_gk_dest_source`,
which the loader now SELECTs anyway.

**`unattested` (R2).** When the resolved-coordinate columns are absent, the helper warns and
no-ops — but it must **not** stamp those rows `native`, which would suppress the downstream
warn-once while the origins are still raw (SkillCorner's present-and-wrong case would then score
in silence — the exact hole the stamp exists to close). It stamps **`unattested`**, and
`compute_xt_gk_v2` treats `unattested` as unstamped for warning purposes. This is preferred over
emitting no stamp at all — the safety is identical, but it keeps the warning *truthful*: "the
helper ran and could not resolve" is a different fact from "the helper never ran," and the
operator needs to know which.

**Why this is not optional.** The helper as first specced was loud only when columns were
*missing*. The genuinely dangerous case — **the caller never runs the helper at all** — was
undetectable: SkillCorner's raw coordinates are *present, finite, and wrong*, so
`finite_coord_mask` passes and scoring proceeds in silence. That is the same class of silent
failure this spec exists to kill, and **this bug happened precisely because a prose contract
("v1 resolves ⇒ v2 must too") was never machine-checked.** The stamp makes resolution
*attestable* downstream, mirroring the discipline v1 already established with
`xt_gk_origin_source` in 4.36.0. Note `pandas` `attrs` do **not** survive most operations
(`feedback_pandas_attrs_dont_propagate`), so this must be a **column**, not metadata.

#### `finite_coord_mask` — shared predicate

`finite_coord_mask(actions) -> npt.NDArray[np.bool_]` — true where `start_x`, `start_y`, `end_x`,
`end_y` are all finite. **Lives in `_possession_value.py`, immediately adjacent to `flat_zones`**
(F6), so the corrected docstring and the blessed alternative sit together.

#### `compute_xt_gk_v2` — guard + attestation (`_metric.py`)

1. **NaN guard.** Non-finite-coord rows emit **NaN across all five output columns**, never enter
   the scoring loop (no zone is ever fabricated), are **excluded from the ρ call** (closing the
   §1.3 silent-imputation exposure *without* touching `predict_proba`), and raise one counted
   `warnings.warn`. Finite rows are **byte-identical** to current behaviour.

2. **Attestation warn-once (F2).** If `actions` carries a `domain_column` with any true rows but
   **no** `gk_geometry_source`, warn once: the frame was not passed through
   `apply_resolved_gk_geometry` and its GK-distribution origins may be raw. Scoped this way, CI
   fixtures with no GK domain stay silent. Warn (not raise) — an external caller may legitimately
   resolve by another route.

3. **Coordinate-coherence check — closes F1 and R1 with ONE machine check, not a docstring.**
   `compute_xt_gk_v2` requires caller-built `retention_features` ("never silently defaulted",
   `_metric.py:51-55`). Correctness would otherwise depend *silently* on the caller running the
   helper **before** `extract_retention_features`: a consumer that resolves `actions` for the metric
   but builds ρ features from the **raw** frame gets zones and ρ disagreeing, with every value
   finite and nothing detecting it (F1) — and the **mirror** is equally broken: ρ features from a
   resolved frame with **raw** `actions` passed to the metric (R1).

   **The check compares coordinates, not provenance.** `extract_retention_features` derives
   `length` = `hypot(end − start)`, `forwardness`, `dy_abs`, `dest_x`, `dest_y_off` as **raw
   arithmetic** on the four coordinates (standardisation happens *inside* `GkRetentionModel`).
   So `compute_xt_gk_v2` recomputes exactly those coordinate-derived columns from `actions` and
   compares them to the supplied `retention_features` (NaN-tolerant, `atol=1e-6`). Any mismatch →
   **`ValueError`**.

   This single rule is **strictly stronger than stamp equality** and subsumes it:
   - it catches **F1** (resolved actions, raw features) and **R1** (raw actions, resolved features)
     **symmetrically** — no case table to get wrong;
   - it catches the **mart-vintage** divergence two frames resolved against different vintages
     would slip past an equal-stamp comparison (R4);
   - it depends on **no** provenance column at all.

   It must span the **origin-derived** columns, not just `dest_x`: because the dest override is a
   measured **no-op** (§1.4), `end_x` is identical in the raw and resolved frames, so an
   origin-only divergence would slip past a `dest_x`-only check. `length` / `forwardness` /
   `dy_abs` encode `start_x`/`start_y` and are what actually catch it.

   This is a **contract violation**, not a data condition, so it raises.

   `extract_retention_features` still **passes `gk_geometry_source` through** when present (safe:
   `GkRetentionModel.fit` (`_retention.py:62`) and `predict_proba` (`:80`) both select
   `features[self.feature_names]`, so a non-feature column is inert) — but the stamp's job is now
   **only** the warn-once attestation of (2). That is the one thing coordinates can never reveal:
   raw coordinates are perfectly self-consistent, so no numeric check can tell you resolution was
   never *attempted*.

#### `flat_zones` — behaviour unchanged, docstring corrected

The NaN→0 mapping stays **byte-identical** (Chesterton's Fence: `_markov.py:65`,
`_empirical.py:83` and `_diagnostics.py:123` legitimately call it with NaN rows *before* their own
`dropna`; changing the default would move the fitted surfaces). Its docstring is corrected to state
that NaN→0 is a **fit-path** contract and that **scoring callers MUST mask via
`finite_coord_mask`**.

### 2.2 Loaders — `scripts/_loader_databricks.py`

Both `_XTGK_ACTIONS_SQL` and `_RETENTION_SQL` **already** `LEFT JOIN fct_action_context c`. Add to
each `SELECT`:

```sql
c.xt_gk_origin_x, c.xt_gk_origin_y,
c.xt_gk_dest_x,   c.xt_gk_dest_y,
c.xt_gk_origin_source, c.xt_gk_dest_source
```

Numeric-coerce the four new coordinate columns (gold nullables arrive `object`/`None`), then pipe
**both** loaders through `apply_resolved_gk_geometry`.

**Resolution lands in the loader, upstream of everything.** There are **three** in-repo consumers
of `load_xtgk_cohort` → `compute_xt_gk_v2` — `validate_xtgk_v2.py`,
`xtgk_v2_keeper_discrimination.py`, and `xtgk_v2_kappa_sweep.py` (the review named two; the sweep
is a third). All three follow the same chain (`load_xtgk_cohort` → `prepare_cohort` →
`extract_retention_features` → `compute_xt_gk_v2`), so **all three inherit the fix for free, and
F1's ordering hazard is closed by construction for our pipeline** — ρ features are necessarily
built from the resolved frame. F1's real residual exposure is the **lakehouse**, which builds its
own frames; that is exactly what the §2.1 attestation + coherence check defends.

### 2.3 ADR-025 interplay — why override-on-copy, not a side-band (F4)

ADR-025's enrichment contract is: **never mutate canonical `start_x`/`end_x`** — emit `enriched_*`
side-band columns, with canonical promotion an explicitly deferred Phase 2. This helper overrides
canonical columns (on a copy). These do **not** conflict, and the ADR-036 amendment must say so
plainly or the next reader will read the two ADRs as contradicting:

- ADR-025's fence protects the **canonical persisted** coordinates — what converters emit and what
  the lakehouse writes to its marts. This helper produces a **transient scoring-time view**: the
  overridden frame is passed to `compute_xt_gk_v2` and discarded. **`start_x` is never written
  back to any mart.**
- The side-band idiom was rejected here because it would force `compute_xt_gk_v2` to grow
  `origin_columns=` / `dest_columns=` parameters — pushing a data-**provenance policy** into the
  metric **engine**. The codebase has an explicit convention against that
  (`feedback_policy_at_edge_not_shared_engine` — the same reasoning that kept the geometry tripwire
  at the `add_restart_coordinates` edge). Policy stays at the edge; the engine stays
  provenance-free and reads exactly `start_x`/`end_x`.

### 2.4 ρ retrain (both variants)

`scripts/train_gk_retention.py` needs **no code change** — it consumes `load_retention_cohort`.
Re-run for `gradientsports` → `default` and `skillcorner` → `skillcorner` on the resolved cohort.

Ship whichever variants clear the **existing** calibration gate (`ece ≤ _ECE_MAX`,
`|reliability_slope − 1| ≤ _SLOPE_TOL`, plus the AUC check), certified by the F1 CI guard
`tests/xtgk/test_retention_bundle_calibration.py`. Precedent runs both ways: 4.42.0 **declined** to
ship the SkillCorner variant at slope 0.63 and fell back to `default` via `_PROVIDER_VARIANT = {}`;
4.44.0 shipped it once the domain broadened.

- **SkillCorner fails** → falls back to `default`, `_PROVIDER_VARIANT` updated, failure reported.
  We do not ship weights we know do not calibrate.
- **gradientsports fails** → no fallback exists. That is itself a finding, surfaced, not papered
  over.

Update `metrics.json`, `MODEL_CARD`, `SHA256SUMS`, `_PROVIDER_VARIANT` for whatever ships.

### 2.5 SP5 re-run (owner-run, local, Databricks read-only)

Run on the corrected cohort **twice** — under the **pre-fix ρ** and the **retrained ρ** — so the
delta is attributable. This needs one additive `--retention-weights <dir>` flag, and it must reach
**all three** scripts (F5, extended): `validate_xtgk_v2.py`, `xtgk_v2_keeper_discrimination.py`
(the ICC lens) and `xtgk_v2_kappa_sweep.py`. Otherwise the ICC re-run silently mixes ρ vintages
against a corrected cohort.

**Pre-frozen before the run:** the metrics (outcome-AUC lift over `max(baselines)`; action-level
keeper ICC grouped by `player_key`), the baselines, the κ=1 headline, and every a-priori parameter
are **unchanged**. Only **coordinates** and **ρ** move. No retuning. The run **reports whatever it
shows**, per ADR-036 §3.

**Leg-1 attribution nuance (F8).** Leg 1 (corrected coords + **pre-fix** ρ) also shifts pre-fix ρ's
*input distribution* — its features derive from the now-overridden coordinates. So leg 1 isolates
*"origin effect **including** ρ-input shift"*, **not** pure zone relabeling. The report must state
this so the delta table is read correctly.

**The fix ships regardless of the verdict** — the coordinates are wrong independent of what the
corrected numbers say (the same logic that shipped the faithful `V_opp` in 4.45.0 despite its
negative result). If v2 still does not validate, the fork question goes to owner + Eyestone on
*trustworthy* numbers; if it does, the program is unblocked without touching one modelling
assumption.

Outputs update `docs/research/xtgk_v2_construct_validity/`. The 4.45.0 reports are **retained**
(renamed) as the contaminated-baseline record — the negative result is part of the project history.

---

## 3. Non-goals (explicit)

1. **Do NOT re-run the deep-zone gate.** All fit seams drop NaN coords (§1.1); verdict stands.
   Proven by T4.
2. **Do NOT change `GkRetentionModel.predict_proba`'s mean-imputation.** The upstream mask removes
   the exposure; the imputation is deliberate and neutral post-standardisation (Chesterton's
   Fence). Documented, not altered.
3. **Do NOT touch the interpretation forks** (V's reward; PEV / receiver-pressure `q`). They remain
   the owner/Eyestone decision this PR exists to *inform*.
4. **Do NOT touch v1** (`tracking/_xt_gk.py`) — frozen until the lakehouse migrates.
5. **Do NOT change `flat_zones`' behaviour** — docstring only. **Deferred (F7, TODO line):** a
   `nan_ok: bool = False` parameter (the three NaN-tolerant fit seams pass `True`; default raises)
   would make the pit-of-failure structurally hard to enter rather than merely documented. Deferred
   because it touches three fit seams whose byte-identity is what licenses non-goal #1 in *this*
   cycle; tracked in TODO so it is not lost.

---

## 4. Testing (red-first)

| # | Gate | Asserts |
|---|---|---|
| T1 | `test_metric_nan_coord_guard.py` | NaN-coord row → **NaN** across all 5 outputs, **not** zone 176's value. Counted warning fires. **Finite rows byte-identical** to pre-fix. ρ not called on non-finite rows. |
| T2 | `test_apply_resolved_gk_geometry.py` | **Override, not coalesce**: present-but-different raw origin (SkillCorner case) **is replaced**; NaN raw + resolved **is filled**; genuinely-unresolved **stays NaN**; **dest override path covered synthetically** (real data is a no-op, §1.4); off-domain rows untouched; **purity** (input unmutated, new object); `domain_column` missing → `ValueError`; resolved columns missing → warn + no-op **stamped `unattested`, never `native`** (R2). `gk_geometry_source` correct for all **seven** values, incl. the **`unresolved`-wins precedence** on a mixed row (resolved origin + NaN dest) (R3). |
| T3 | `test_flat_zones_contract.py` | `flat_zones`' NaN→0 → zone-176 behaviour **pinned unchanged**, so the fit seams cannot drift. |
| T4 | `test_deep_zone_gate_nan_invariance.py` | `MarkovPossessionValue.fit` output (surfaces + support) **invariant** to adding/removing NaN-coord rows. **This is the evidence for non-goal #1.** |
| T5 | `test_metric_retention_coherence.py` | **F1 + R1 machine check, both directions.** (a) resolved `actions` + features built from the **raw** frame → `ValueError` (F1). (b) **raw** `actions` + features built from the **resolved** frame → `ValueError` (R1's mirror). (c) A divergence that is **origin-only** (identical `end_x`, since the dest override is a measured no-op) is still caught — proving the check spans `length`/`forwardness`/`dy_abs` and not just `dest_x`. (d) Coherent pair → scores normally. (e) Unstamped (or `unattested`) `actions` with a GK domain → **warn-once**, and it still scores. |
| T6 | `test_resolved_origin_changes_score_e2e.py` | **F6 — the A/B must exercise the path that can change the value.** A SkillCorner-shaped row (present-but-**wrong** raw origin) through helper → metric **must produce a different score** than the unresolved path. Guards against a fix that is inert on the very case it exists for. |
| T7 | `test_retention_bundle_calibration.py` (existing) | Re-certifies whatever ρ variants bundle against canonical `_ECE_MAX` / `_SLOPE_TOL`. |

---

## 5. Consumer impact (Hyrum)

- **`compute_xt_gk_v2` output changes** — NaN where it previously fabricated; corrected values on
  the ~24% (GS) / ~22% (SC) of rows with resolved origins. → **xT-GK v2 re-materialize trigger**.
- **New `gk_geometry_source` column** on the helper's output (additive).
- **ρ bundled weights change** → same trigger.
- **Cross-repo handoff:** the lakehouse must call `apply_resolved_gk_geometry` before
  `compute_xt_gk_v2`. If it does not, it now gets a **warn-once** (and a `ValueError` if its ρ
  features and actions carry divergent coordinates) instead of silently wrong numbers.
- **Handoff coverage caveat — the attestation has one blind spot, and the lakehouse must not walk
  into it (R5).** The warn-once fires only when `domain_column` is **present with true rows**. The
  most likely lakehouse calling shape is a **pre-filtered GK-distribution slice with
  `is_gk_distribution` dropped** — column absent → no warn, no stamp, **silent raw scoring**. The
  handoff therefore states explicitly: **the lakehouse must keep `is_gk_distribution` (or the
  `gk_geometry_source` stamp) on the frame it passes to `compute_xt_gk_v2`**, or the attestation
  cannot protect it. (The coordinate-coherence check of §2.1(3) still fires regardless — but it
  only catches actions↔features *divergence*, not a uniformly-raw pair.)
- **NOT a forced VAEP retrain** — xT-GK v2 is opt-in and in no default xfn list.

---

## 6. Release mechanics

- Version **4.46.0** (minor: new public helper + behaviour change + re-bundled weights).
  Bump `pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md`, `CHANGELOG.md` (+ `uv.lock`).
- **ADR-036 amendment** — must include §2.3 (ADR-025 interplay) verbatim in substance.
- **C4:** no new action-coupled aggregator → count **stays 28**.
- **TODO line (F7):** `flat_zones` `nan_ok` hardening.
- **National Park bundle:** `TODO.md:28` claims TF-19's arms are "**UNBLOCKED**" while the shipped
  weights record `"tf19_ready": false`
  (`silly_kicks/tracking/_xcross_weights/default/metrics.json:134`). One-line correction of a real
  doc-vs-reality defect. *(Cross-session: the part-deux TF-19 spec has ceded this line to this PR.)*

---

## 7. Cross-session coordination

- The part-deux TF-19 cycle intends to lift `icc_one_way` / `keeper_spread` out of
  `scripts/xtgk_v2_keeper_discrimination.py` into a shared module. That lift is **deferred until
  after this cycle's SP5 re-run completes** (user-mediated), so no refactor lands underneath this
  PR.
- The part-deux TF-19 spec now treats the v2 keeper-flatness result as **pending this re-run**
  (see §1.6).

---

## 8. Observations noted, deliberately out of scope

- The **entire `xtgk` public surface (28 exports) sits outside the `Examples` CI gate** —
  `tests/test_public_api_examples.py::_PUBLIC_MODULE_FILES` lists no `xtgk` module. The new public
  helper will carry an `Examples` docstring in house style, but retrofitting all 28 exports and
  extending the gate tuple is its own cycle.
