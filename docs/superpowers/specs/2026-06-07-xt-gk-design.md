# xT-GK — Expected Threat for Goalkeepers (Eyestone)

## Executive summary (for Jeffrey)

*This section is the plain-English overview; the rest of the document is implementation detail for the silly-kicks codebase and can be skimmed.*

**What we're building.** Your **xT-GK** metric, implemented as a first-class, **public, fully-attributed** feature inside the open-source silly-kicks library. Anyone using the library will be able to compute xT-GK on their own data and cite your work. This is exactly the path the library uses for every published method it implements.

**How faithful it is to your deck.** We're including **all of your components** (per your answer): the traditional xT baseline, **Pressure-Escape Value**, **Risk-Adjusted Value**, **Defensive-Zone Value**, plus the **spatial-convolution** and **temporal-sequence** terms from your public app. The metric re-values goalkeeper distributions the way your deck describes — rewarding playing out of pressure, balancing risk vs. reward, and removing the unfair penalty traditional xT gives to back-passes and own-half build-up.

**The few choices we had to make — ✅ confirmed by you 2026-06-08.** Your deck describes the components and their *parameter ranges* conceptually, but not as exact equations, so we wrote down a concrete, faithful set of formulas (§4). You confirmed the one open modeling choice: **the destination's threat is counted once** (owned by the risk-adjusted term), not double-counted by the base — and you approved publishing the **provisional in-range preset values**. Two other specifics (no change needed from you):
- **Pressure** is measured continuously from tracking data (your preference), rather than the deck's three buckets.
- **"Chance the pass succeeds"** (inside Risk-Adjusted Value) reuses the library's existing tracking-based **expected-pass-completion (xC)** model — a real per-pass model, so we don't have to hand-build one.

**One important practical point: xT-GK needs tracking data.** Five of the six components could run on basic event data, but **Pressure-Escape Value needs a pressure signal, and that only exists when you have player-tracking data**. (The basic event feeds simply don't carry pressure once standardized.) Rather than ship a version that quietly drops your pressure component, we make xT-GK a tracking-based metric — which matches your "continuous pressure" preference anyway. It will run on the tracking sources we have (IDSSE, SkillCorner, Gradient Sports).

**Your team-philosophy parameters (γ, δ, φ, η)** ship as the **presets** you defined (possession / counter / direct / high-press / low-block), as you asked. We just need the exact preset values if you have them — the deck gives ranges.

**What the library produces.** For each goalkeeper distribution it outputs the **overall xT-GK score** *and* a **breakdown into its components**, so analysts can see *why* a distribution scored as it did (and so dashboards downstream can recombine them). Dashboards/visuals themselves live in Karsten's platform, not the library — the library produces the numbers.

**Plan of work (one combined program, delivered in stages).**
1. **Now:** the xT-GK metric itself, with your philosophy presets (the metric's internal technical constants get sensible fixed defaults).
2. **Later:** an optional way to fit γ/δ/φ/η to a specific team's actual behaviour (your "team-specific, down the line" idea), without changing what the parameters mean.

**What we'd like from you** (full list in §11): confirm the formulas (§4) capture your intent; the exact preset values for the five philosophies; that estimating pass-success from the data is acceptable; and — later — whether you have a preferred recipe for the team-specific fitting in stage 3.

---

**Date:** 2026-06-07 (updated 2026-06-08) · **Author:** Karsten (with Claude) · **Status:** **Approved for build** — Jeffrey confirmed Q1 (Option B: destination counted once) + Q2 (provisional presets OK) on 2026-06-08; Phase-1 gate cleared.
**Feature family:** GK distribution-value (companion to the future xR-GK; distinct from the GKDV deterrent program — TF-15..19 — which xT-GK does **not** feed)
**Attribution:** Jeffrey Eyestone, *Expected Threat for Goalkeepers (xT-GK)* — winner, Pitch to the Pros 1 (May 2025). Contributed publicly with attribution by Jeffrey's explicit permission (email 2026-06-06: *"Being public with attribution is OK"*). **Formula confirmation 2026-06-08:** Jeffrey confirmed Option B (destination counted once: base origin-only, RAV owns the destination) and approved publishing the provisional in-range presets. NOTICE entry + docstring citation required; both consent points recorded in the ADR.

> **Provenance note.** The deck specifies xT-GK's components and parameter *ranges* conceptually, not as closed-form equations. The functional forms in §4 are **the silly-kicks formulation of Eyestone's xT-GK** — a faithful reconstruction, documented as such. Jeffrey's guidance (email 2026-06-06) was to *"pick the easy answer"* / *"just do whatever is easy."* The interpretation point — the formula forms + preset values (⚠️ **IF-1**) — was a **Phase-1 build gate** (§9) and is now **✅ CLEARED (2026-06-08):** Jeffrey confirmed Option B for the formula (destination counted once) and approved the provisional presets (§11 Q1–Q2). (IF-2, P(success), is also **resolved** — RAV reuses the library's tracking xC model `get_xc`, §4.3.)

---

## 1. Purpose

Traditional (Karun-Singh) xT fails systematically for goalkeepers: the GK zone carries the lowest grid value (≈ 0.001–0.005), identical passes are valued identically regardless of pressure, and back-passes to the GK receive *negative* xT despite tactical value. **xT-GK** is an analytical extension of the xT grid that re-values **GK distribution actions** (goal-kicks, keeper passes, keeper throws) by composing the baseline grid with GK-specific terms, under a frozen, team-tunable parameter set.

xT-GK is a **pure parametric compute feature** — an analytical formula, **not** a trained model. It has no learned weights, so the ADR-011 trained-model lifecycle (code-PR → weights-PR, booster-JSON, `[xgboost]` gate, HPO trainer) does **not** apply. Its closest architectural siblings are OBSO / DAS / pitch-control / cover-shadows: a frame-aware tracking feature with `compute_*` / `add_*` / `*_xfns` surfaces + an atomic mirror.

### 1.1 Scope boundary

**In scope (this combined spec, phased across releases — see §9):**
- **Phase 1:** the xT-GK feature — the composite metric + its decomposed components, `XtGkParams` philosophy presets, frame-aware surfaces, atomic mirror, NOTICE + ADR.
- **Phase 2:** team/dataset calibration of the γ/δ/φ/η philosophy parameters (empirical per-team estimation, opt-in). *(The one structural smoothing constant is fixed by a one-off scan inside Phase 1 — there is no separate calibration phase; §9 option (c).)*

**Out of scope (deliberately):**
- **xR-GK** (Expected Retention) — Jeffrey's companion trained-model metric. Held pending its exact 16-feature list (his Q6: *"not easy to get at the moment"*) and label spec (Q7). It is a *separate* ADR-011 trained-model feature; not in this spec.
- **Dashboards / visualizations** — lakehouse / Power-BI concern (the architecture split). silly-kicks produces numbers only.
- **An events-only xT-GK path** — evaluated and rejected: the PEV (pressure-escape) component needs a pressure signal, and **no pressure survives SPADL conversion** (StatsBomb's `under_pressure` flag is dropped; "Pressure" events are excluded — `statsbomb.py:49`). An events-only metric would silently lack a headline component. xT-GK is therefore **tracking-required**; see §3.1.

---

## 2. Data dependency — tracking-required

xT-GK is **frame-aware and tracking-required.** Rationale (decision 2026-06-07): of the six components, five (base, DZV, RAV, temporal, spatial-convolution) are events-derivable, but **PEV (pressure-escape) requires a pressure signal**, and the only pressure signal in silly-kicks is the tracking-derived `pressure_on_actor`. Because no pressure survives SPADL (§1.1), an events-only path would always drop PEV — a core part of Jeffrey's GK-distribution-value pitch. Shipping a default `xt_gk` silently missing pressure-escape is a Hyrum/honesty trap. Jeffrey's Q3 answer (*"drive it off a continuous pressure measure"*) confirms the faithful metric is the tracking one.

This places xT-GK in the tracking namespace (ADR-005), alongside the rest of the GK suite.

---

## 3. Module structure

| Artifact | Path | Role |
|---|---|---|
| Production module | `silly_kicks/tracking/_xt_gk.py` (new) | components, composite, `XtGkParams`, `compute_*` / `add_*` / `*_xfns` |
| Atomic mirror | `silly_kicks/atomic/tracking/features.py` (extend) | `add_xt_gk` re-export per the atomic-mirror pattern |
| Public exports | `silly_kicks/tracking/__init__.py` (extend `__all__` + imports) | `compute_xt_gk`, `add_xt_gk`, `xt_gk_xfns`, `XtGkParams` |
| Attribution | `NOTICE` (extend) | Eyestone xT-GK bibliographic entry + the NDA-aware email-consent trail (2026-06-06 public-with-attribution permission) for the record |
| Decision record | `docs/superpowers/adrs/ADR-NNN-xt-gk.md` (new) | formula interpretation + tracking-required placement + calibration phasing + consent provenance. **Number:** next free is the **ADR-015 gap** (001–021 exist with 015 missing) or **ADR-022**; ADR-021 is already taken by xthreat (SK-xT-1) — no collision with this spec. Reconcile against `origin/main` at release. |

**Dependency placement.** xT-GK consumes `ExpectedThreat` (from `silly_kicks.xthreat`, lightweight) for the baseline grid and `pressure_on_actor` (from `silly_kicks.tracking.pressure`) for ρ. **RAV's `P(success)` reuses `silly_kicks.tracking.get_xc`** (the existing tracking xC model, §4.3), which wraps `accessible_space.get_expected_pass_completion` — so **the RAV completion term gates on the `accessible-space` (`[das]`) extra**, lazily imported (the same gate DAS uses). This is a deliberate, accepted change to the "pure numpy" story: an empirical zone-grid completion model is far too sparse for the rare GK-distribution corpus (≈ `n_zones²` cells, few GK distributions), so reusing the real per-pass xC model is the correct trade. No `[xgboost]` gate (still not a trained model). Bare `import silly_kicks` stays light; xT-GK pulls `accessible-space` only when actually computing RAV.

**xT injection (no self-fitting — leakage contract).** `ExpectedThreat` is **fit-only** in 4.17.0 (no prefit/`from_grid` constructor; verified `_model.py`). To avoid in-sample leakage, `compute_xt_gk`/`add_xt_gk` take a **required `xt: ExpectedThreat` (already fitted)** kwarg — the caller fits it once on a corpus **disjoint** from the scored matches (the OBSO/frozen pattern). The functions never fit xT internally. A small **tracking-local** frozen-grid convenience helper may be added if needed; we do **not** up-import `calibration.FrozenXt` (calibration-layer-only — verified `calibration/_xt.py`; bad layering, and §3 forbids touching `calibration/`). The same injected `xt` supplies `xT★_counter` (the opponent mirror).

**No edits** to `silly_kicks/calibration/`, `scripts/calibrate_*`, or `scripts/_loader_*` — and with option (c) (§9) xT-GK uses **no** calibration machinery at all (its one smoothing constant is hand-set via a one-off Phase-1 scan), so no cross-session coordination is needed.

---

## 4. The metric

Notation: a GK distribution action *a* moves the ball from origin zone *z* to destination zone *z′* (resolved on a `GridSpec`). Let `xT★(·)` be the spatially-convolved baseline grid value (§4.1), *p* = `P(success)` (§4.3), ρ ∈ [0,1] the normalized pressure on the actor (§4.4).

### 4.1 Base & progress — spatially-convolved values  *(Option B, Jeffrey confirmed 2026-06-08)*
```
progress(a) = xT★(z′) − xT★(z)     # forward move value — feeds PEV only
base(a)     = − xT★(z)              # composite positional term — origin-only
```
**Option B (Jeffrey 2026-06-08, resolves the §11 Q1 double-count):** the destination value `xT★(z′)` enters the composite's main value path **once** — through RAV (completion-weighted, §4.3) — **not** also through a full-weight base term. So the composite `base` is **origin-only** (`−xT★(z)`: the threat given up by leaving the origin). A separate `progress` quantity (`xT★(z′)−xT★(z)`) is retained **solely** for PEV (§4.2), because PEV is a forward-progress signal; folding PEV onto the origin-only base would zero it (`max(0,−xT★(z)) ≡ 0`). (The earlier draft used a single `B = xT★(z′)−xT★(z)` in both roles, which double-counted the destination as `(1+p)·xT★(z′)`; Jeffrey chose to count it once.)

`xT★` is the **caller-injected, already-fitted** `ExpectedThreat` (`singh_counts`) grid (the required `xt=` kwarg, §3), used as a **frozen exogenous artifact** (fit once on a corpus disjoint from the scored matches — the OBSO pattern; never self-fit). It is then smoothed by a **separable Gaussian convolution kernel over neighbouring grid zones** (the public-app *spatial-convolution* term). **This grid-convolution is a new xT-GK-owned primitive** — 4.17.0's xthreat ships KDE-smoothed *transition matrices*, **not** a convolution of the xT *grid* (verified `_transitions.py`), so xT-GK implements it itself. Convolution σ is a genuine smoothing nuisance constant — its default is fixed by a **one-off sensitivity scan in Phase 1** (not a calibration phase; §9 option (c)); σ = 0 / kernel off ⇒ `xT★ ≡ xT` (raw grid).

### 4.2 Components (raw, before parameter weighting)
- **PEV — Pressure-Escape Value:** `PEV(a) = ρ · max(0, progress(a))` — rewards progressing the ball *out of* pressure (uses `progress`, the forward move value, NOT the origin-only `base`; §4.1). (Tracking-required; the reason §2.)
- **RAV — Risk-Adjusted Value:** `RAV(a) = p · xT★(z′) − δ · (1 − p) · xT★_counter(z′)` — completion-weighted destination value minus a risk-aversion-weighted counter-attack threat at the turnover location, with `p` = xC (§4.3). `xT★_counter(z′)` is the opponent's grid value from the loss zone — the **point-reflection of `xT★`**: in LTR-normalized coords (team attacks +x) the opponent attacks −x, so their threat from zone `(x, y)` is `xT★(L − x, W − y)` (**full 180° rotation, both axes**, not an x-only flip). The plan must pin this transform precisely (cheap to get subtly wrong — review #2 minor). δ is carried inside RAV (it is the risk-aversion weight, not an outer scalar). **Modeling caveat (open question for Jeffrey, §11):** this values the counter from the *intended destination* `z′`, but a *failed* distribution is typically intercepted **en route**, not at `z′` — `xT★_counter(z′)` is a simplification of the true (unknown) turnover location.
- **DZV — Defensive-Zone Value:** `DZV(a) = 1[z ∈ defensive third] · (V_def − xT(z))` — restores tactical value to own-third possession and removes the back-pass-to-GK penalty (where traditional xT ≈ 0 or negative, DZV adds a positive baseline `V_def`). **`V_def` and the defensive-third boundary are *normative* constants, NOT smoothing nuisance** — they encode the size of the back-pass-penalty fix and where "defensive third" begins, i.e. the value claim xT-GK exists to make. They are therefore set **by intent (Jeffrey)**, in the same "interpretive, not VAEP-calibrated" bucket as γ/δ/φ/η — **never calibrated** (§9). Tuning them to a prediction loss would distort the correction itself.

### 4.3 P(success) — RAV completion probability  (IF-2 resolved)
Jeffrey's Q2: *"derive it from the library's existing pass-completion machinery."* 4.17.0 ships exactly such machinery: **`silly_kicks.tracking.get_xc`** — a public, tracking-based **expected-pass-completion (xC)** model (per-pass, wrapping `accessible_space.get_expected_pass_completion`; verified `_das.py:420`, exported from `tracking/__init__`). Since xT-GK is **tracking-required anyway**, `get_xc` is the natural and faithful `P(success)` source: a real per-pass model (not a sparse grid), same-layer (tracking→tracking), already validated and used by DAS.

- **`p` = xC from `get_xc`** on the GK-distribution passes + linked frames. This **supersedes** the earlier hand-rolled empirical-grid + OBSO-opt-in proposal (an `n_zones²` zone grid is far too sparse for the rare GK-distribution corpus), and removes the `completion_bandwidth` and `p_success_source` knobs entirely.
- **⚠️ Out-of-distribution risk (review #2.1 — verify before the plan finalizes):** accessible-space's xC is built/validated on **open-play passes**, but the in-scope set is **goal-kicks (long aerial), keeper passes, throws** — goal-kicks especially are a different completion regime. Mitigating nuance: accessible-space is a **physics/geometry-based** reachability model (ball trajectory × player control), not a pure learned event-classifier, so it may extrapolate to GK distributions better than a fitted classifier would — *but* high aerial goal-kick trajectories may still violate its assumptions. **Action:** verify `get_xc` produces sane `xC` on goal-kicks/throws on real fixtures during implementation; if OOD, **document it** and/or special-case goal-kicks (e.g. a geometry-based completion prior for the aerial regime). Do not let "we reuse the real model" hide that the real model may be off-distribution for exactly these actions.
- **Cost + fail-loud (review #2.5):** RAV is **always** part of the composite, so `accessible-space` (`[das]`) is effectively a **hard requirement for `xt_gk`**, not "only when computing RAV." Per §2's no-silent-drop principle, absence of `accessible-space` must raise a **clear `ImportError`** (e.g. `"xT-GK requires: pip install silly-kicks[das]"`), **never** a silent RAV-less composite. Stated here and enforced in §6.

### 4.4 Pressure ρ
Continuous `pressure_on_actor` (default method `andrienko_oval`; `link_zones` / `bekkers_pi` selectable via `XtGkParams`), normalized to [0,1] — this normalization materially shapes PEV (`ρ·max(0,B)`), so the **v1 default is pinned**: a bounded **saturating-exponential squash** `ρ = 1 − exp(−max(0, raw) / s)` (the exponential-CDF form — monotone, smooth, maps [0,∞)→[0,1); **not** a logistic, corrected from the prior draft's misnomer). The `max(0, raw)` clamp guards against any pressure method returning a negative raw value — **confirm during implementation whether `andrienko_oval` / `link_zones` / `bekkers_pi` can go negative**; the clamp makes ρ well-defined regardless. Scale `s` = `pressure_scale` (see §5 note on its borderline-normative status). (Min-max is rejected for v1 — corpus-relative, not train/serve-stable.) Always available (tracking-required).

### 4.5 Temporal-sequence discount  ⚠️ **IF-1 (form)**
```
T(a) = η ** k(a)
```
where `k(a)` is the action's depth within its possession sequence (the public-app *temporal-sequence* term; η ∈ [0.8, 0.9]). v1 uses this positional-discount form (simple, pure).

**Two issues to confirm with Jeffrey (§11), surfaced by review:**
1. **Near-inert for the in-scope domain.** xT-GK scores **GK distributions**, which are overwhelmingly **possession-starters** (`k ≈ 0` ⇒ `T ≈ η⁰ = 1`). So for almost every in-scope action the temporal term does ~nothing — it reads like a *general-action* public-app term applied to a possession-start subset. We keep it for completeness but flag that it may be inert here by construction.
2. **It must not discount the corrective term.** Where `T` does bite (a GK distribution mid-possession), discounting **DZV** would shrink the back-pass-penalty *fix* for late actions — almost certainly unintended. **v1 default: `T` scales only the threat-bearing terms, not DZV** (see §4.6).

*(Factual correction vs. the prior draft: 4.17.0's `value_iteration(p_scoring, p_shot, p_move, transition, *, eps, max_iter)` has **no** discount parameter — verified `_value_iteration.py:13`. So the "fold η into a discounted value_iteration" alternative would first require adding a discount upstream; it is not an available reuse and is dropped from v1.)*

### 4.6 Composite
v1 applies the temporal discount to the **threat-bearing terms only**, leaving the corrective DZV undiscounted (§4.5 issue 2):
```
xT-GK(a) = T(a) · ( base(a) + γ · PEV(a) + RAV(a) ) + φ · DZV(a)
         = T(a) · ( −xT★(z) + γ · PEV(a) + RAV(a) ) + φ · DZV(a)
```
(Pending Jeffrey's confirmation, §11 — if he intends `T` to scale the whole composite including DZV, revert to `T·(base + γ·PEV + RAV + φ·DZV)`.)

**✅ Destination value counted once (Q1 RESOLVED — Jeffrey 2026-06-08: Option B).** The composite `base` is **origin-only** (`−xT★(z)`), and **RAV solely owns the destination value** (`p·xT★(z′)`, completion-weighted). The destination therefore enters the main value path **once**, not as the earlier `(1+p)·xT★(z′)`. PEV uses the separate `progress` term (§4.1/§4.2), so the origin-only base does not neuter it. (Jeffrey's reasoning: avoid double-counting; the risk-adjusted term is the right place for the destination's value.)

### 4.7 Emitted columns (output shape — "Option A": components **and** composite)
`compute_xt_gk` / `add_xt_gk` add, per GK-distribution action:

| Column | Meaning |
|---|---|
| `xt_gk_base` | `base(a) = −xT★(z)` — origin-only positional term (Option B; destination owned by RAV) |
| `xt_gk_pev` | raw PEV term `ρ·max(0, progress)` (progress = `xT★(z′)−xT★(z)`) |
| `xt_gk_rav` | raw RAV term |
| `xt_gk_dzv` | raw DZV term |
| `xt_gk_pressure` | ρ used (transparency) |
| `xt_gk` | the γ/δ/φ/η-weighted composite |

Raw components are emitted **and** the parameterized composite — because the γ/δ/φ/η presets *are* the published method (unlike TF-45, where the composite was an exogenous dataset-fit kept consumer-side). Decomposition supports construct-validity analysis and lets the lakehouse recombine.

**Stability labeling (review #2 / minor) — updated 2026-06-08:** Jeffrey confirmed the §4 **combine form** (Q1 = Option B). The **composite form is now stable**; the preset **point-values** remain provisional-but-approved (Q2 — in-range, no exact table), so `xt_gk` magnitudes may shift if Jeffrey later supplies exact preset numbers. Documented on the column/docstring so a later preset-value revision doesn't silently surprise downstream `xt_gk` consumers.

---

## 5. Parameters & presets

`XtGkParams` — a frozen dataclass (house string-dispatch + frozen-dataclass idiom, e.g. `CoverShadowParams` / `PitchControlParams`; **not** an ABC):

```python
@dataclass(frozen=True)
class XtGkParams:
    # --- interpretive / intent-set (NOT VAEP-calibrated; set by Jeffrey) ---
    gamma: float = ...        # PEV pressure-escape sensitivity      (deck range 0.1–0.4)
    delta: float = ...        # RAV risk-aversion                    (deck range 0.3–0.8)
    phi: float = ...          # DZV defensive-zone weight
    eta: float = ...          # temporal-sequence discount           (deck range 0.8–0.9)
    v_def: float = ...        # NORMATIVE: back-pass-penalty-fix magnitude (review #4)
    defensive_third_boundary: float = ...   # NORMATIVE: where "defensive third" begins (review #4)
    pressure_scale: float = ...             # borderline-normative → intent-set (review #2); ρ squash scale (§4.4)
    # --- structural (single hand-set default; convolution_sigma chosen by a one-off Phase-1 scan, §9) ---
    convolution_sigma: float = ...          # xT★ grid-smoothing
    # --- method selectors ---
    pressure_method: str = "andrienko_oval" # | "link_zones" | "bekkers_pi"

    @classmethod
    def for_philosophy(cls, name: str) -> "XtGkParams": ...
    # "possession" | "counter" | "direct" | "high_press" | "low_block"
```

Dropped vs. the prior draft: `completion_bandwidth` and `p_success_source` — RAV's `p` is now always `get_xc` (§4.3). `v_def` and `defensive_third_boundary` moved into the **interpretive** bucket (review #4). `pressure_scale` is **intent-set** (borderline-normative — it scales PEV's reward, so VAEP-fitting it is a milder version of the #4 category error; hand-pick a sensible scale, document it). That leaves `convolution_sigma` as the **only** genuine smoothing nuisance — and per the §9 decision (option (c)) its default is hand-set via a **one-off sensitivity scan in Phase 1**, not a calibration phase. **xT-GK therefore has no calibration-harness surface.**

`for_philosophy(...)` returns the deck's five team-philosophy presets. ⚠️ **IF-1 (values):** the deck states parameter *ranges*; the exact per-philosophy point values are an open question for Jeffrey (§11). v1 ships defensible values within the stated ranges, clearly marked provisional. Default params (no preset) = a neutral/balanced point. Per Q4, **no calibration in Phase 1** — presets are the answer.

---

## 6. Conventions & frame-aware obligations

- **Domain filter:** GK distribution actions only (goal-kick / keeper pass / keeper throw), GK identity via existing GK-resolution helpers (`_gk_resolve`, derived-GK, frame-based). Non-GK-distribution rows pass through unchanged.
- **NaN-safety (ADR-003):** `add_xt_gk` decorated `@nan_safe_enrichment`; NaN identifiers route to NaN output; auto-discovered by `tests/test_enrichment_nan_safety.py`.
- **Idempotent provenance:** skip the linkage-provenance merge when those columns already exist (the established `if not any(c in out.columns ...)` guard; CI gate `tests/tracking/test_provenance_skip_guard.py`).
- **Signature & pre-linking:** `compute_xt_gk(actions, frames, *, xt: ExpectedThreat, params=None, ...)` and `add_xt_gk(actions, frames, *, xt: ExpectedThreat, params=None, links=None, ...)` — **`xt` is required** (pre-fitted, disjoint corpus; no self-fit — §3). `links` accepts caller-supplied link pointers (skip internal `link_actions_to_frames`). **No `pitch_control_cache` kwarg** — xT-GK consumes no pitch-control surface (the dropped OBSO P(success) path was its only use; `get_xc` doesn't need it).
- **Frame-id resolution (ADR-020):** the `*_xfns` path must resolve frame ids via the shared `_kernels.resolve_frame_ids_by_position` (positional), **not** `set_index("action_id").at[...]` — VAEP shifted gamestate slots repeat boundary actions ⇒ non-unique `action_id`. `xt_gk_xfns` must be enrolled in the auto-enumerating behavioral gate `tests/tracking/test_frame_aware_xfns_dup_action_id.py`.
- **Id-dtype safety (ADR-019):** every action↔frame / `home_team_id` comparison routes through `silly_kicks.tracking._id_compat` (`ids_equal`/`ids_match`/`same_id`/`align_join_keys`); covered by the asymmetric dtype-invariance gate + the AST lint.
- **Atomic mirror:** `add_xt_gk` re-exported from `atomic/tracking/features.py`; atomic end-coordinate synthesis where relevant.
- **`xt_gk_xfns` is a factory carrying the injected `xt` (review #2 — no self-fit leakage):** the parameterless-shaped VAEP xfn must not re-introduce the leakage §3 fixed. It follows the **established `gk_influence_xfns(xt: ExpectedThreat, *, ...)` precedent** (`features.py:2755`): `xt_gk_xfns(xt: ExpectedThreat, *, params=None, ...) -> list[FrameAwareTransformer]` is a **factory that closes over the caller-fitted `xt`** and returns the transformer(s). The returned xfns **never fit xT internally** — a test asserts the xfns path consumes the injected grid and self-fits nothing (mirrors the §7 leakage-contract test for `compute_/add_`). (No calibration-harness compatibility constraint applies — §9 option (c) means xT-GK is never wired into TF-24.)
- **Fail-loud on missing `accessible-space` (review #5):** the composite always needs xC, so `compute_xt_gk`/`add_xt_gk`/`xt_gk_xfns` raise a clear `ImportError("xT-GK requires: pip install silly-kicks[das]")` when `accessible-space` is absent — never a silent RAV-less `xt_gk` (§2 no-silent-drop principle).

---

## 7. Testing

**Construct-invariant gates (PROMOTED to Phase-1 gates — review #8).** For a ground-truth-free *normative* metric, "the code computes the formula" oracles are necessary but circular. The testable substitute is **the corrections xT-GK claims to make**, asserted as executable invariants encoding Jeffrey's intent (they catch formula regressions and need no ground truth):
- a **back-pass to the GK** gets its penalty *corrected upward* — asserted as the **direction**, not an absolute sign (review #2.3): the composite **with DZV is strictly greater than the same composite with DZV disabled** (`φ·DZV > 0` raises it), and `xt_gk_dzv > 0`. Both are **true by construction** given `φ > 0`, `V_def > 0`, regardless of the still-unconfirmed param values. *(Absolute `xt_gk ≥ 0` is kept only as a preset-specific **fixture** check — it holds iff `φ·V_def` outweighs `|B|`, which is param-dependent and would otherwise force φ/V_def or fail on steep back-passes.)*
- a **defensive-third build-up pass** gets a **strictly positive `xt_gk_dzv`** contribution;
- a **higher-pressure forward escape** gets **strictly higher `xt_gk_pev`** than the same pass unpressured (the direction assertion — not just "different").

Then the supporting suite:
- **Per-component oracle tests:** independent closed-form oracles for B / PEV / RAV / DZV / T / composite, `np.isclose` (~1e-9) + exact key-set/dtype (golden-float platform-fragility lesson — oracle mirrors the impl's exact formula).
- **Leakage contract:** a test that `compute_xt_gk`/`add_xt_gk` **require** a pre-fitted `xt` and never fit internally (#3).
- **Preset resolution:** `for_philosophy` returns distinct, in-range param sets for all five philosophies.
- **CI gates inherited:** dup-`action_id` behavioral gate, nan-safety gate, id-dtype gate, provenance-skip gate (all auto-discover the new surface — verify enrollment).
- **Domain filter:** non-GK-distribution actions unchanged; GK actions enriched.
- **Atomic-mirror parity** with the regular surface.
- **No perf budget needed** (pure feature). If a hot primitive emerges, a structural call-count guard (not wall-clock) per `tests/_perf_structural.py`.
- **Construct-validity (note, not a v1 gate):** xT-GK should correlate with xR-GK once xR-GK exists (parked).

**Validation-data caveat (review #8).** xT-GK runs only on the three tracking providers (IDSSE / SkillCorner / Gradient Sports), and some of that substrate has open data-quality issues (e.g. IDSSE pass-typing; GS owngoals / player-ids) — so even eyeball validation will be on few, partly-suspect matches. The construct-invariant gates above are deliberately **synthetic-fixture** based (deterministic, provider-independent) so they don't inherit that fragility; real-provider checks are corroborating, not gating.

---

## 8. C4 / docs

- `tracking` container aggregator **count +1** (`add_xt_gk`) → C4 DSL edit + regen (`mad-scientist-skills:c4`) in the implementation commit. Count invariant: `len([n for n in tracking.__all__ if n.startswith('add_')]) - 1` (minus `add_gradientsports_player_ids`). xT-GK is **not** a trained model and **not** a KDE backend, so no token enumeration change beyond the count.
- CLAUDE.md: one `PR-S## ships … xT-GK …` architecture line in the feature commit.
- NOTICE: Eyestone xT-GK entry; docstring `See NOTICE for full bibliographic citations.`

---

## 9. Phased delivery (combined body of work; subsequent releases)

Versions unassigned — reconciled against `origin/main` at each release per the version-bump checklist (main is currently 4.17.0). ADR number assigned at first release.

**Phase-1 build gate (review #2) — ✅ CLEARED 2026-06-08.** Jeffrey's email approved **publication**; his 2026-06-08 reply then confirmed **Q1** (Option B — destination counted once; `base` origin-only, RAV owns `xT★(z′)`) and **Q2** (provisional in-range presets OK to publish). His confirmation **is** the validation for a ground-truth-free metric (§7). The Phase-1 build is unblocked; the composite form is final (preset point-values provisional-but-approved, §4.7).

**Phase 1 — xT-GK core feature** *(first release)*
The whole of §3–§8: `_xt_gk.py` (components + composite), `XtGkParams` + `for_philosophy` presets, required `xt=` injection, `get_xc` P(success), frame-aware `compute_*`/`add_*`/`xt_gk_xfns`, atomic mirror, NOTICE + ADR, full test suite incl. the construct-invariant gates (§7), C4 regen. **All parameters set by intent/hand:** γ/δ/φ/η + `v_def` + `defensive_third_boundary` from the `for_philosophy` presets; `pressure_scale` a documented intent-set default; and the **one** structural smoothing knob `convolution_sigma` fixed via a **one-off sensitivity scan** baked into this phase (sweep the range once, pick a sensible default, document it). Ships complete and usable.

> **No separate calibration phase (decision: option (c), 2026-06-07).** After the round-2 reviews shrank the calibratable surface to a single smoothing scalar, standing up TF-24 harness wiring + cross-session coordination for one number was not justified (YAGNI). `convolution_sigma`'s default is therefore hand-set via the Phase-1 scan above, **not** a calibration follow-up. Consequently **xT-GK touches no calibration machinery at all** — no edits to `silly_kicks/calibration/` or `scripts/calibrate_*`, and no cross-session coordination needed. (The interpretive params γ/δ/φ/η, `v_def`, `defensive_third_boundary`, `pressure_scale` were never calibration candidates — §4.2, §5.)

**Phase 2 — Team/dataset parameter calibration** *(subsequent release)*
The γ/δ/φ/η *"down the line"* calibration Jeffrey flagged (Q4: team-specific params). This is **empirical per-team estimation** of a team's tactical signature — e.g. δ from observed turnover/risk behaviour, γ from observed pressure-escape behaviour — **not** a VAEP-loss fit, so interpretability is preserved. Surfaces as an opt-in `XtGkParams.for_team(actions)` (or a calibration routine); presets remain the default. The exact estimation method is an open question for Jeffrey (§11).

---

## 10. Effort estimate

Phase 1 ≈ a Tier-5 primitives feature: ~150–250 LOC in `_xt_gk.py` + aggregator + xfns + atomic mirror + `XtGkParams`/presets + the one-off `convolution_sigma` scan + tests + ADR + NOTICE + C4 regen. No multi-PR weights split (no trained model). Phase 2 (team calibration) is a smaller, scoped follow-up.

---

## 11. Open questions for Jeffrey

1. ~~**⚠️ IF-1 (formula confirmation — Phase-1 gate)**~~ **✅ RESOLVED (Jeffrey 2026-06-08): Option B.** The destination value is counted **once** — `base` is origin-only (`−xT★(z)`) and RAV solely owns `p·xT★(z′)` (no `(1+p)` double-count). The remaining §4 forms (PEV = ρ·max(0,progress); RAV = p·xT★(z′) − δ·(1−p)·xT★_counter; DZV = 1[def-third]·(V_def − xT); composite §4.6) stand.
2. ~~**⚠️ IF-1 (preset values — Phase-1 gate)**~~ **✅ RESOLVED (Jeffrey 2026-06-08):** ship the **provisional in-range** γ/δ/φ/η preset values (the deck gives ranges; no exact point table) — approved for publication, labelled provisional.
3. **Temporal-sequence term (review #6):** for GK distributions (which are possession-*starters*, k≈0) the `η^k` discount is near-inert. Is the temporal term meaningful in this domain at all? And do you intend it to discount the corrective **DZV** (v1 says no — it scales only the threat terms, §4.6), or the whole composite?
4. **RAV turnover location (review #7):** RAV values the opponent counter from the *intended destination* `z′`, but a failed distribution is usually intercepted *en route*. Acceptable simplification, or do you have a preferred turnover-location model?
5. **P(success) (IF-2, resolved — confirm):** we use the library's tracking xC model (`get_xc`) for RAV's completion probability (this is the "existing pass-completion machinery" you pointed to). Confirm that's what you intended.
6. **Spatial-convolution:** confirm this is the public-app term and that the §4.1 Gaussian neighbour-smoothing of the xT *grid* (a new xT-GK primitive) is the right reading.
7. **Phase 2 method (team calibration):** for team-specific γ/δ/φ/η, do you have a preferred empirical estimation recipe, or is best-practice ours to design?

---

## 12. Decisions captured (Karsten, 2026-06-07; Jeffrey confirmation 2026-06-08)

- **Formula (Q1) — Jeffrey 2026-06-08: Option B.** Destination value counted **once** (RAV owns `xT★(z′)`); composite `base` is origin-only (`−xT★(z)`). PEV retains a separate `progress = xT★(z′)−xT★(z)` term (origin-only base would otherwise zero PEV). Closes the `(1+p)` double-count.
- **Presets (Q2) — Jeffrey 2026-06-08:** ship provisional in-range γ/δ/φ/η values; approved for publication (no exact point table). Composite form final; preset values provisional.
- End state of the spec session: **spec only**, then review (no code, no commit without explicit go-ahead).
- Output shape: **components + composite** (Option A).
- Data dependency: **tracking-required** (after the SPADL pressure-drop finding ruled out a faithful events-only path).
- Calibration: **option (c) chosen (2026-06-07)** — **no separate calibration phase.** All interpretive params (γ/δ/φ/η, `v_def`, `defensive_third_boundary`, `pressure_scale`) are intent-set, and the lone smoothing knob `convolution_sigma` is hand-set via a one-off sensitivity scan folded into Phase 1. xT-GK touches **no** calibration machinery / no cross-session coordination (§9). Phase 2 is now solely the opt-in **team/dataset γ/δ/φ/η** estimation.
- P(success): **reuse tracking `get_xc`** (not a hand-rolled grid) — RAV gates on the `accessible-space` extra (review round 1, #5).
- xT baseline: **caller-injected pre-fitted `ExpectedThreat`** (required kwarg; no self-fit → no leakage; review round 1, #3).
- Location: this repo (`docs/superpowers/specs/`), Rule #1 lifted for xT-GK by Jeffrey's public-with-attribution permission.

---

## Appendix A — Review history (internal; not for external readers)

*Change log of the internal lakehouse-session review rounds — kept for the audit trail, not part of the design itself.*

**Review round 2 (resolved):** #1 `get_xc` may be **OOD for goal-kicks/throws** → verification-before-plan action + documented fallback (§4.3; physics-model nuance noted). #2 `xt`→`xt_gk_xfns` must not self-fit → **factory carrying the injected `xt`** per the `gk_influence_xfns` precedent + a no-self-fit xfns test (§6). #3 back-pass gate → assert the **correction direction** (DZV strictly raises the composite; `xt_gk_dzv > 0` — construction-true), absolute `≥0` demoted to a preset fixture (§7). #4 **destination double-count** (`B` + RAV both carry `xT★(z′)`) → explicit in Q1 (§11). #5 `[das]` is a **hard composite requirement** → fail-loud `ImportError` (§4.3/§6). Minors: exponential-CDF rename + clamp (§4.4); `pressure_scale` borderline-normative → lean intent-set (§5); **Phase-2 YAGNI** (calibratable surface ≈ one scalar) → **resolved 2026-06-07 as option (c)**: `convolution_sigma` scan folded into Phase 1, separate calibration phase dropped, team-calibration renumbered Phase 2 (§9); `xT★_counter` = full 180° point-reflection (§4.2).

**Review round 1 (resolved):** IP cleared (NDA-aware email trail; cited in NOTICE/ADR). Frozen-xT injection was genuinely missing (`ExpectedThreat` fit-only) → **required pre-fitted `xt=` kwarg**, no self-fit, no `calibration.FrozenXt` up-import. `v_def`/`defensive_third_boundary` are normative → **excluded from calibration** (intent-set). P(success) premise was outdated → **RAV uses tracking `get_xc`** (dropped empirical grid + OBSO + `completion_bandwidth`). Temporal term near-inert for possession-starting GK distributions + **must not discount DZV** (composite restructured). RAV turnover-location surfaced as an explicit Q. **Construct-invariant tests promoted to Phase-1 gates**. Spatial-convolution is xT-GK-owned; **§4.5 `value_iteration`-has-no-discount factual fix**. Minors: v1 pressure normalization pinned; composite labeled provisional; ADR-number reconciled. Publication ≠ formula confirmation → **Phase-1 build gated on Jeffrey confirming Q1+Q2**.

---

### Sources
- Eyestone xT-GK deck (Pitch to the Pros 1 winner) + public app `xotrspur.manus.space` — NDA source material in the external `Jeffrey Eyestone` folder.
- Investigation & fit analysis: `Jeffrey Eyestone/specs and plans/2026-06-05-eyestone-gk-metrics-investigation.md`.
- Karun Singh, *Introducing Expected Threat (xT)* (2018); Silverman (1986) — see `silly_kicks/xthreat/` (ADR-021).
