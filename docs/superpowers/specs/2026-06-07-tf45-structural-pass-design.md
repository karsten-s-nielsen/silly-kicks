# Design: TF-45 `structural_pass` — per-pass structural primitives (LBS / SGM / SDI)

**Date:** 2026-06-07
**Status:** Draft — revised after **two** cross-session spec-review rounds (all findings verified
against the real backend + incorporated; see "Cross-session review resolutions" at end).
Implementation-ready pending maintainer sign-off.
**Author:** Claude Opus 4.8 (1M) + Karsten S. Nielsen
**Reviewers:** cross-session review (mediated by maintainer)
**Tracker:** TODO.md Tier 5, TF-45. **Paper:** arXiv:2603.28916 (Karakuş & Arkadaş, 2026).

---

## Context — what this adds

The paper *"Structural Pass Analysis in Football: Learning Pass Archetypes and Tactical Impact from
Spatio-Temporal Tracking Data"* (Oktay Karakuş & Hasan Arkadaş, submitted 30 Mar 2026 — the same
author pair as the TF-32 line-breaking paper) introduces a *structural* view of passing: how a pass
deforms the opponent's defensive configuration, rather than its outcome value. It derives three
complementary per-pass metrics from synchronized tracking + event data on the 2022 FIFA World Cup
(64 matches, 32 teams, 41,078 open-play successful passes, 29.97 Hz), z-normalizes them into a
composite **Tactical Impact Value (TIV)**, and K-means(4)-clusters passes into four archetypes
(circulatory / destabilising / line-breaking / space-expanding).

This change ships **only the three raw structural primitives** as a pure silly-kicks tracking
feature. The corpus-level layers (TIV z-norm, K-means archetypes, ΔTIV / cumulative-TIV rankings)
are **deliberately out of scope** — they require a population reference (mean/std over a corpus) or a
fitted clustering model, which is stateful. Keeping them consumer-side mirrors the
frozen-exogenous-xT decision (TF-24 / ADR-009): the library stays stateless and pure; downstream
consumers (lakehouse, scouting tools) own the population statistics and may calibrate weights rather
than equal-weight.

### Value & honest positioning

Value is as a **human-interpretable structural / scouting signal**. The paper's own results show TIV
discriminates *territorial progression* (final-third entry probability rises from ~4.5% in the
lowest TIV quantile to >12% in the highest; box entry ~1.5%→~3%) but is **flat against immediate
shots** ("structural pass impact primarily facilitates territorial progression rather than directly
determining the likelihood of immediate shot attempts"). Center-backs dominate cumulative TIV
(Stones, Otamendi, Süle, Dias, Thiago Silva). So the *new* ML signal over existing pitch-control /
cover-shadow / defensive-line features is modest; the strength is interpretability and the
descriptive structural lens. Note: the paper reports **quantile probabilities, not correlation
coefficients** — there is no "correlation with goals" claim to reproduce.

### Empirical fact-check (this design)

Verified against the real backend before writing this spec:

- **Local WC2022 GS event catalog (all 64 matches):** 144,541 total possession events (matches the
  recorded catalog size exactly); 66,715 `PA` passes, 2,412 `CR` crosses. Open-play-successful ≈ 41k
  is fully consistent with the paper's 41,078 (set-pieces + failures ≈ 38% of `PA`).
- **Reference-implementation run through the real silly-kicks GS pipeline** (`scripts/_loader_pining`
  → GS SPADL converter + GS native tracking adapter → `link_actions_to_frames`), 3 matches / 2,466
  open-play successful passes: base rate of final-third entry ≈ 8–10% (paper-consistent); the
  primitives compute cleanly and the σ default below is tuned on this run. This also confirms the
  owner-gated e2e path is feasible exactly as TF-30/TF-32 consume tracking.

---

## Paper math (verbatim) and silly-kicks mapping

The paper uses **y as the attacking axis**; silly-kicks uses **x** (LTR: home attacks +x, pitch
[0,105]×[0,68]; SPADL: the *acting* team always attacks toward x=105). Defenders are the **outfield
players of the team not in possession** — the paper states "in practice, `D_i` corresponds to the
set of outfield players" → **goalkeeper excluded**, n≈10.

Let, for pass `p_i`: passer position `x_s`, receiver position `x_r`, defender set
`D = {d_1..d_n}` (opponent outfield, at the pass frame).

### 1. Line Bypass Score (LBS)
> `b_j = 1 if y_s < y_j ≤ y_r else 0`;  `LBS = Σ_j b_j`

Count of defenders whose attacking-axis coordinate lies strictly above the passer and at-or-below
the receiver. **Forward-only by construction** — a lateral/backward pass (`x_r ≤ x_s`) yields 0.

**silly-kicks:** in the acting-attack-positive (SPADL) frame, `LBS = #{ d ∈ D : start_x < d_x ≤ end_x }`.

### 2. Space Gain Metric (SGM)
> `ρ(x) = Σ_j exp(−‖x − d_j‖² / 2σ²)`;  `S(x) = 1/ρ(x)`;  `SGM = S(x_r) − S(x_s)`

Isotropic Gaussian defender density; "available space" = inverse density; Δ receiver vs passer.
**σ is not disclosed anywhere in the paper** — see the σ decision below. Positive = pass moved into a
lower-pressure region.

### 3. Structural Disruption Index (SDI)
> `c = (1/n) Σ_j d_j`;  `SDI = ‖x_r − c‖ − ‖x_s − c‖`

`c` = the **full-team 2-D centroid of all outfield defenders**. Higher = pass stretches play away
from the defensive centroid.

### Out of scope (consumer-side, corpus-level)
- **TIV** `= w₁·LBS̃ + w₂·SGM̃ + w₃·SDĨ` with z-norm `m̃_k = (m_k − μ_k)/σ_k` and equal weights
  (1/3 each). Requires population μ_k/σ_k.
- **K-means(K=4)** archetypes on `(LBS̃, SGM̃, SDĨ)`. Requires a fitted model.
- **ΔTIV / cumulative-TIV** passer/receiver rankings. Corpus aggregation.

---

## Decisions (locked in brainstorming) and rationale

### D1 — σ = 15 m (frozen default, empirically tuned)

The paper does not disclose σ. Tuned empirically on 2,466 real WC2022 open-play successful passes
(reference impl through the real GS pipeline), σ ∈ {3,5,8,10,12,15,20}, evaluated on three criteria:
SGM standalone discrimination of final-third entry (rank-AUC), numerical conditioning (p99 / max
|SGM|), and K=4 archetype silhouette.

| σ (m) | SGM AUC | TIV rank-AUC | p99 \|SGM\| | max \|SGM\| | silhouette |
|------:|--------:|-------------:|----------:|----------:|-----------:|
| 3  | 0.626 | 0.811 | 1.2e30 | 4.3e51 | 0.408 |
| 5  | 0.629 | 0.810 | 6.3e10 | 2.8e18 | 0.407 |
| 8  | 0.645 | 0.813 | 14112  | 1.07e7 | 0.402 |
| 10 | 0.658 | 0.817 | 350    | 25095  | 0.394 |
| 12 | 0.668 | 0.819 | 46     | 1042   | 0.389 |
| **15** | **0.679** | **0.821** | **7.9** | **68.5** | **0.385** |
| 20 | 0.690 | 0.823 | 1.5    | 6.4    | 0.387 |

**Finding:** both progression signal *and* numerical conditioning improve monotonically with σ — so
the choice is not an interior signal optimum but *where the inverse-density primitive stops being a
numerical footgun*. `S = 1/ρ` explodes when a passer/receiver sits in open space (`ρ → 0`); below
σ≈12 the raw SGM spans 30–51 orders of magnitude, which would destroy any consumer z-normalization
or K-means (precisely the corpus-level steps we hand to consumers). The primitive becomes
**intrinsically bounded by pitch geometry** (no artificial ε-floor needed) at σ≈15 (max |SGM| ≈ 68,
p99 ≈ 8 over 2,466 passes), with near-peak progression discrimination (within 0.002 rank-AUC of
σ=20) and intact archetype structure. σ=20 is marginally cleaner/stronger but means a defender
"influences" a 20 m radius — over-smoothed past any defensible influence scale (literature influence
radii sit ~10 m). **σ = 15 m** is the smallest σ that yields a well-conditioned raw primitive.

Exposed as `StructuralPassParams.sigma = 15.0` (frozen single-field dataclass, no `is_default()` —
matches `CoverShadowParams`/`LineBreakingParams`), tunable by consumers. **No ε-floor / clamping is
added** — σ=15 makes the faithful formula bounded by geometry, so we keep it bit-faithful to the
paper.

### D2 — Receiver location `x_r` = pass destination `end_x/end_y`

SPADL has **no `receiver_player_id`** (confirmed: zero matches repo-wide). x_r = the pass
destination `(end_x, end_y)`. Self-contained (needs only the action + one linked frame at the pass
moment), matches "pass" semantics, no receiver-resolution error. The paper is silent on which frame
it uses, so this is **not less faithful — just simpler**. Documented as a faithfulness caveat in the
NOTICE entry (same reframe discipline as xCross's state-level note).

### D3 — Gating: compute for every `pass` + `cross`, `NA` otherwise

`add_structural_pass` emits raw geometry for **every `pass` (type_id 0) and `cross` (type_id 1)**
row — the geometry is well-defined for failed passes too — and `NA` for non-pass/cross rows.
Open-play + success filtering is left to the consumer (corpus-level, like TIV). Keeps the primitive
pure and stateless and maximally reusable. (The paper used open-play successful only; the e2e test
applies that filter to match the paper, the library does not.)

### D4 — Scope: raw primitives only (see "Out of scope" above)

---

## Public surface (ADR-005 canonical pattern)

New module `silly_kicks/tracking/_structural_pass.py`, re-exported through
`silly_kicks/tracking/features.py` and `silly_kicks/tracking/__init__.py` as the peer modules are.

**Output column contract (Hyrum-locked from commit 1):** the appended columns are
**`structural_lbs` / `structural_sgm` / `structural_sdi`** (namespaced, like every peer:
`defensive_line_x`, `blocking_score`); the xfns emit `structural_lbs_a0 … structural_sdi_a2`. The
oracle test asserts this exact key set. Bare `lbs`/`sgm`/`sdi` are rejected — they would be the least
self-describing columns in a 200-column VAEP frame and become a downstream contract once shipped.

```python
@dataclass(frozen=True)
class StructuralPassParams:
    sigma: float = 15.0          # NO is_default() — peers (CoverShadowParams, LineBreakingParams)
                                 # have none; only PreprocessConfig does, for provider resolution (YAGNI).

# Pure, pandas-free domain core (hexagonal): all the math, the cleanest TDD/oracle target,
# and a clean seam if this ever needs numba (like the other primitives).
def _structural_pass_core(
    defenders_xy: np.ndarray,         # (n, 2) opponent-outfield positions, in acting-attack-positive frame
    passer_xy: tuple[float, float],   # (start_x, start_y) in the same frame
    receiver_xy: tuple[float, float], # (end_x, end_y) in the same frame
    sigma: float,
) -> tuple[float, float, float]:      # (structural_lbs, structural_sgm, structural_sdi)
    """0 defenders -> (nan, nan, nan). >=1 defender -> all three numeric (see Error handling S1)."""

def compute_structural_pass_metrics(
    frame: pd.DataFrame,
    *,
    attacking_team_id: int | str,
    home_team_id: int | str,
    passer_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    params: StructuralPassParams | None = None,
) -> dict[str, float]:
    """Per-frame DataFrame wrapper over _structural_pass_core. Returns
    {'structural_lbs', 'structural_sgm', 'structural_sdi'} for ONE linked frame.
    Schema-agnostic: passer_xy/receiver_xy passed explicitly (NOT read off a row), re-exportable to
    atomic (mirrors lane_control(frame, passer_xy, receiver_xy, *, home_team_id, attacking_team_id),
    _cover_shadows.py:506-514). Defenders = opponent outfield, selected as
    `players = frame[~frame['is_ball'].astype(bool)]` THEN
    `players[~ids_match(players['team_id'], attacking_team_id) & ~players['is_goalkeeper'].astype(bool)]`
    (_cover_shadows.py:566-571 idiom — ~is_ball FIRST: ~ids_match returns False for NA team_id, so a
    ball row would otherwise pass the opponent mask). Defenders mapped tracking->acting-attack-positive
    frame (mirror 105-x, 68-y iff away), then handed to _structural_pass_core."""

# Shared per-action batch kernel (DRY: both add_* and xfns call it, like
# _kernels._defensive_line_at_actions at features.py:1150 & :1201 — preserves the
# "link/select N x per pass, not 9x" call-count budget the VAEP factory depends on).
def _structural_pass_at_actions(
    actions: pd.DataFrame,        # must carry start_x/start_y/end_x/end_y
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    params: StructuralPassParams | None = None,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:                # columns: structural_lbs, structural_sgm, structural_sdi (+ provenance)
    ...  # lives in _kernels.py

@nan_safe_enrichment
def add_structural_pass(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    links: pd.DataFrame | None = None,
    params: StructuralPassParams | None = None,
) -> pd.DataFrame:
    """Append structural_lbs / structural_sgm / structural_sdi (NA for non-pass/cross).
    Idempotent provenance cols. Accepts caller-supplied `links`. Delegates to
    _kernels._structural_pass_at_actions. Body implements the NaN-safety contract;
    @nan_safe_enrichment + the CI gate only VERIFY it."""

def structural_pass_xfns(
    *,
    home_team_id: int | str,
    params: StructuralPassParams | None = None,
) -> list:
    """VAEP factory: ONE FrameAwareTransformer (structural_lbs/sgm/sdi per gamestate slot ->
    structural_lbs_a0 … structural_sdi_a2) that calls the SAME _kernels._structural_pass_at_actions
    per slot, `frames is None` guard, transformer._frame_aware = True."""
```

**Atomic mirror** (`silly_kicks/atomic/tracking/features.py`). structural_pass is a passer→receiver
**displacement** metric, so it consumes the receiver coordinate — unlike `cover_shadows`/
`defensive_line` (passer-only), it **cannot** use a `x→start_x`-only rename or a direct xfns
re-export. Atomic SPADL has no `end_x/end_y` — its schema is `x, y, dx, dy`
(`atomic/spadl/schema.py:11-14`); the receiver is `end_x = x + dx`, `end_y = y + dy`. Therefore:
- `compute_structural_pass_metrics` — **re-export directly** (schema-agnostic: takes explicit
  `passer_xy`/`receiver_xy`).
- `add_structural_pass` — atomic **wrapper** that synthesizes `start_x=x, start_y=y, end_x=x+dx,
  end_y=y+dy` on the atomic actions, delegates to the standard `add_structural_pass`, then renames
  back (NOT the `add_cover_shadows` start-only rename, which would silently yield NaN/garbage
  LBS/SDI on every atomic action).
- `structural_pass_xfns` — atomic **wrapper** (NOT a re-export): the transformer synthesizes
  `start_*/end_*` on each atomic gamestate slot before calling the shared kernel, for the same
  reason.

### Coordinate handling (the one subtlety)

SGM and SDI are Euclidean → frame-invariant; LBS needs a consistent attacking axis. Implement **all
three in the acting-attack-positive (SPADL) frame**: take passer `(start_x, start_y)` and receiver
`(end_x, end_y)` from the action directly, and convert each opponent-outfield defender's tracking
`(x, y)` into that frame — identity if the acting team is home, mirror `(105−x, 68−y)` if away. Then
LBS is the unambiguous `start_x < d_x ≤ end_x` count; the SGM Gaussian and SDI centroid use the same
defender points and the action endpoints.

**Deliberate mirror-direction divergence from the cited peer.** `_line_breaking.py:243–252` mirrors
the *action* coords → tracking frame; we mirror the *defenders* tracking → SPADL frame. Both are
correct (the mirror is an involution). We choose the opposite direction on purpose: LBS is only
clean in attack-positive coords (otherwise the inequality flips sign per team), and SGM/SDI are
isometry-invariant so they don't care which way the mirror goes. One sentence to this effect goes in
the module docstring so future maintainers don't read it as an accident.

**Cross-coordinate-frame invariant (must be documented + asserted).** This feature mixes two
coordinate sources — passer/receiver from the SPADL *action*, defenders from the linked *tracking
frame* — and relies on both sharing the `[0,105]×[0,68]` pitch frame after the per-team mirror. The
hand-placed oracle unit tests (single synthetic coordinate system) **cannot** detect a real
frame/scale divergence; only the e2e exercises real action-vs-frame coords. So: (a) state the
invariant explicitly in the module docstring ("post-normalization SPADL action coords and LTR
tracking coords share the pitch frame; defenders are mirrored into the action's attack-positive
frame"); (b) the e2e asserts a *known forward pass* yields the expected LBS sign/magnitude against
the real linked frame (not just an aggregate AUC) — see Testing §4.

**LBS interpretability caveat (docstring + NOTICE).** LBS is purely 1-D along the attacking axis:
a defender whose `d_x ∈ (start_x, end_x]` is counted as "bypassed" even if he is on the opposite
touchline. This is faithful to the paper's `y_s < y_j ≤ y_r` definition, but a scouting consumer
will over-read LBS without the caveat — flag it next to the D2 receiver caveat.

### Reuse — corrected vs the TODO row

- ❌ The TODO's "reuse defensive centroid from `compute_defensive_line` (`_defensive_line.py:84`)" is
  **wrong**: that function returns `defensive_line_x` = the mean x of the back-line N (3–5 closest)
  players only — a 1-D back-line height. SDI needs a **fresh full-team 2-D centroid over all ~10
  outfield defenders**. Computed inline (~2 lines). (`_team_shape.py` may expose a 2-D centroid; not
  worth a dependency.)
- ❌ The TODO's "inverse-density Gaussian analogous to `_cover_shadows.py`" is **wrong**:
  `_cover_shadows.py` is a sigmoid-interception + Spearman-drag model, not a Gaussian spatial
  density. SGM's ρ is a ~3-line isotropic KDE written directly.
- ✅ **Cite, don't reuse:** LBS ≈ the simple end of TF-32 `detect_line_breaking` (which does
  Ward-clustered line-polyline intersection — heavier, different output); SGM ≈ TF-41
  `compute_space_created` (which is leave-one-out OBSO pitch-control delta — different and
  expensive). Reference both in the docstring/NOTICE as related work.

---

## Error handling / NaN safety (ADR-003)

**The function body implements the NaN-safety contract itself** — `@nan_safe_enrichment` is a
*marker only* (`_nan_safety.py:18-35` sets `fn._nan_safe = True` and catches/converts nothing); the
CI gate (`tests/test_enrichment_nan_safety.py`) *verifies* the claim, it does not grant it. So the
body must handle every degenerate case → `NaN`/`NA`, never crash:
- **0 outfield defenders at the frame → all three NaN** (the *only* mathematically degenerate case:
  ρ=0 divides by zero in SGM; the centroid is undefined). **≥1 defender → all three numeric** — SGM
  (`1/ρ`) and SDI (centroid of ≥1 point) are well-defined. There is **no `<3` threshold** here; that
  belongs to `compute_defensive_line` (3-player back line) and does not apply. The defender-count
  guard runs **before** the `1/ρ` division.
- **`structural_lbs = 0` is reserved for "defenders present, none in the bypass band"** (e.g. a
  forward pass with defenders, none with `start_x < d_x ≤ end_x`) — **distinct from NaN** ("no
  defender data"). The oracle test pins both: 0-defender frame → `structural_lbs` NaN;
  defenders-present-none-in-band → `structural_lbs` 0.
- NaN passer or receiver coords → NaN.
- Unlinked action (no frame within tolerance) → NaN.
- Non-pass/cross row → `NA` (D3).
- Empty `frames` / empty `actions` → empty/NA result with the correct columns.

Faithful (not error) outputs: backward/lateral pass with defenders → `structural_lbs`=0, SGM/SDI may
be negative.

ADR-019 id-dtype safety: opponent defenders are selected by a **series-vs-scalar** comparison
(`attacking_team_id` is a scalar), so use `~ids_match(players["team_id"], attacking_team_id)` —
**not** `ids_differ` (which is series-vs-series, `_id_compat.py:118`, for the merged action-vs-frame
seam). `~is_ball` must be applied **first** (`~ids_match` returns False for NA `team_id`, so a ball
row would otherwise survive the opponent mask); no redundant NA logic — `~is_ball` is the guard.
This is the `_cover_shadows.py:566-571` idiom.

---

## Testing (TDD, hexagonal, e2e)

0. **Pure-core oracle tests (TDD-first, the cleanest target)** —
   `_structural_pass_core(defenders_xy, passer_xy, receiver_xy, sigma)` is pandas-free, so the math is
   asserted with zero DataFrame scaffolding (M4). Hand-placed defenders, exact closed-form values,
   `np.isclose` ~1e-9:
   - LBS integer count, including strict-`<` / `≤`-boundary defenders and the forward-only
     backward-pass = 0 case.
   - SGM = `1/ρ(end) − 1/ρ(start)` against a 2-defender hand computation at σ=15.
   - SDI against a hand-computed 2-D centroid.
   - **`structural_lbs=0` (defenders present, none in band) is DISTINCT from NaN (0 defenders).**
   - **0 defenders → `(nan, nan, nan)`; ≥1 defender → all numeric** (S1; no `<3` threshold).
1. **DataFrame/aggregator oracle tests** — `tests/tracking/test_structural_pass.py`, regular
   (non-e2e) suite. The pandas wrappers over the core; `np.isclose` ~1e-9 **plus the exact appended
   key set `{structural_lbs, structural_sgm, structural_sdi}` and dtypes** (locks the Hyrum contract):
   - **Home vs away** (mirror correctness — same physical geometry, mirrored coords → identical
     metrics).
   - GK excluded from defenders; ball excluded (ball row's NA `team_id` must not survive the opponent
     mask — `~is_ball` first).
   - NaN coords / unlinked / non-pass-cross → NaN/NA.
   - Idempotent provenance (chained `add_*` produces no `_x`/`_y` suffixes — the provenance-skip
     guard).
   - `links` kwarg path == internal-link path (bit-identical).
   - **Atomic mirror parity** — a known atomic action `(x, y, dx, dy)` with **real non-zero dx/dy**
     reproduces the standard `(start_x, start_y, end_x, end_y)` result exactly. (A start-only fixture
     would falsely pass against the broken rename — the fixture MUST exercise dx/dy so it catches a
     missing `end = x + dx` synthesis. Same for the atomic `structural_pass_xfns` wrapper.)
2. **NaN-safety gate** — auto-covers `add_structural_pass` once decorated. (The decorator is a
   marker; this gate is what verifies the body's contract — see Error handling.)
3. **Structural perf guard** — a call-count spy on `_kernels._structural_pass_at_actions` proving
   the VAEP factory invokes it **3× (once per gamestate slot), not 9×**, exactly as the
   `defensive_line` budget does (`features.py:1176-1178`). Deterministic call-count, not a
   wall-clock ceiling (per the 4.15.0 hardening). This is *why* the shared kernel exists.
4. **Owner-gated e2e** — `tests/tracking/test_structural_pass_e2e.py`, `@pytest.mark.e2e`, needs
   `PINING_FOR_THE_DATA_TOKEN`. Loads 1–2 real WC2022 GS matches via the validated loader path.
   **All validation metrics are computed on open-play successful `pass` (type_id 0) rows only** — the
   library also emits crosses, but the e2e must filter to the paper's population or the base-rate
   band drifts. Asserts:
   - base rate of final-third entry ∈ [0.07, 0.13] (observed ≈ 0.08–0.10);
   - LBS rank-AUC for final-third entry ≥ 0.70 (observed ≈ 0.86). **This is a
     correctness/regression guard, NOT an independent reproduction of the paper's progression
     result** (M3): `structural_lbs > 0 ⟺ a forward pass ⟺ mechanically correlated with final-third
     entry`, so the AUC is partly tautological. It trips if LBS plumbing breaks; it does not validate
     the paper's structural claim. The NOTICE/docstring must not cite it as such.
   - **targeted invariant (predicate-selected at runtime, NOT a frozen action_id)** — filter the
     loaded match for a pass with `end_x − start_x > {threshold}` and ≥1 opponent in the x-band, take
     the first, assert `structural_lbs ≥ 1`. This validates the action↔frame coordinate frame against
     *real* linked data. A hardcoded match/action id would bit-rot under GS re-ingestion + the
     16-copy frame-dedup (row identity shifts); M2.
   - SGM conditioning at σ=15: **max |SGM| ≤ 200 and p99 |SGM| ≤ 20** (observed 68 / 8). Concrete
     ceilings, not "a few hundred" — a silent drift to σ=12 (max 1042 / p99 46) must trip the gate.

   Mirrors `test_gradientsports_scoreline_e2e.py`'s owner-gated structure.

5. **VAEP integration note (not a new test, a documented expectation)** — because the metric is `NA`
   for non-pass/cross actions, the lagged gamestate slots (`structural_*_a1`, `structural_*_a2`) are
   `NA` for most rows (most prior actions aren't passes). This is correct — every action-coupled
   feature has this property — and the VAEP feature pipeline's imputation already tolerates
   predominantly-NA columns; stated here so it isn't mistaken for a bug during integration (M5).

### Implementation ordering (TDD, red→green) — for the plan to inherit (M6)

Strict test-first sequence; do **not** implement primitive-first and back-fill:
1. Pure-core oracle tests (Testing §0) — **red**.
2. `_structural_pass_core` (pandas-free math) — **green**.
3. DataFrame primitive `compute_structural_pass_metrics` (defender select + mirror + wrap core).
4. Shared `_kernels._structural_pass_at_actions` (link + per-action batch) + its oracle tests.
5. `add_structural_pass` aggregator (+ NaN-safety gate, provenance-skip).
6. `structural_pass_xfns` (+ 3×-call-count perf spy).
7. Atomic wrappers (`add_*` + `structural_pass_xfns` endpoint synthesis; re-export `compute_*`) +
   atomic-parity test with real dx/dy.
8. Owner-gated e2e **last**.

---

## Docs / metadata touchpoints (all bundled into the single feature commit)

- **NOTICE** — new entry under "Mathematical / Methodological References": arXiv:2603.28916
  (Karakuş & Arkadaş, 2026), with **(a)** the D2 receiver faithfulness caveat, **(b)** the LBS 1-D
  attacking-axis caveat (counts a far-touchline defender as bypassed), and **(c)** the
  cite-not-duplicate note vs TF-32 / TF-41. **Do not** cite the e2e LBS-AUC as a reproduction of the
  paper's progression finding (M3 — it is a tautological regression guard). Per-feature docstrings
  cross-link `See NOTICE for full bibliographic citations.`
- **CLAUDE.md** — one `PR-S## ships TF-45 …` line in the tracking architecture paragraph.
- **TODO.md** — update/close the TF-45 Tier-5 row, including: the **metric-name correction** (Line
  Bypass Score / Space Gain Metric / Structural Disruption Index — not "Line-Breaking / Spatial
  Gain"), the **two corrected reuse claims**, and the **σ=15 empirical result**.
- **CHANGELOG.md** — feature entry + version bump (next free minor after `origin/main` at ship time;
  reconcile per the version-bump checklist — do not pre-reserve).
- **C4** — `add_structural_pass` is a **new tracking aggregator** → the tracking-container
  description's aggregator **count** must bump (DSL edit + regen via the `mad-scientist-skills:c4`
  pipeline). No new KDE backend / trained model / enumerated token otherwise. Confirm the exact
  enumerated count string at implementation time.
- **σ tuning artifact (reproducibility)** — persist the tuning run as
  `scripts/tune_structural_pass_sigma.py` (cleaned-up version of the investigation script; owner-gated
  via the pining loader, like the e2e and the `scripts/train_*.py` family). The σ=15 frozen constant
  **and** the e2e's max|SGM|≤200 / p99≤20 ceiling derive entirely from this run; without a re-runnable
  artifact the magic number and the gate threshold are un-derivable by the next maintainer. The script
  emits the σ-sweep table (the D1 table). Bundles into the feature commit.
- **ADR:** none required *for TF-45 itself* — it is an in-pattern ADR-005 tracking feature (no new
  architectural decision); the σ choice and scope boundary are recorded here + in the tuning script.
  **However**, during plan review a maintainer decision folded a **systemic dup-`action_id` xfns
  fix** into this same commit (a shared `_kernels.resolve_frame_ids_by_position` resolver + a
  behavioral gate over all `*_xfns`, retrofitting ~8 broken families). That cross-cutting invariant
  **does** get its own ADR (frame-aware xfns resolve frame_id by position, never `.at` on a
  possibly-non-unique `action_id`; enforced by `test_frame_aware_xfns_dup_action_id.py`). See the
  implementation plan (`docs/superpowers/plans/2026-06-07-tf45-structural-pass.md`, Tasks 5A–5C +
  Task 10 Step 5) for the full design of the folded-in fix.

---

## Scope boundaries (explicit)

**In:** `_structural_pass.py` with `StructuralPassParams`, `compute_structural_pass_metrics`,
`add_structural_pass`, `structural_pass_xfns`; the shared `_kernels._structural_pass_at_actions`
batch kernel; atomic mirror (re-export `compute_*`; endpoint-synthesizing wrappers for `add_*` and
`structural_pass_xfns`); unit + e2e tests; `scripts/tune_structural_pass_sigma.py`; NOTICE /
CLAUDE.md / TODO / CHANGELOG / C4.

**Out:** TIV composite, z-normalization, K-means archetypes, ΔTIV / cumulative-TIV rankings (all
consumer/corpus-level); any `pitch_control_cache` plumbing (no pitch-control dependency); any new
ADR; any change to `compute_defensive_line` / `_cover_shadows` / TF-32 / TF-41 (cite only).

---

## Cross-session review resolutions (2026-06-07)

### Round 1

Reviewer verdict: architecturally sound and faithful; scope cut correct. All findings verified
against the real backend and incorporated:

- 🔴 **Atomic mirror is not a thin rename** (verified: `atomic/spadl/schema.py:11-14` = `x,y,dx,dy`,
  no `end_*`; `add_cover_shadows` adapter renames `x→start_x` only because it is passer-only). Fixed:
  the atomic `add_structural_pass` **and** `structural_pass_xfns` synthesize `end = x+dx, y+dy`; only
  `compute_*` re-exports directly. Atomic-parity test must use real dx/dy. (The xfns extension beyond
  the reviewer's note is the same root cause — both surfaces consume the receiver.)
- 🟠 **`@nan_safe_enrichment` is a marker only** (verified `_nan_safety.py:18-35`). Fixed: wording
  reframed — body implements, decorator + gate verify.
- 🟠 **Per-action batch belongs in a shared kernel** (verified `add_defensive_line` + xfns both call
  `_kernels._defensive_line_at_actions`, `features.py:1150/1201/1176-1178`). Fixed: added
  `_kernels._structural_pass_at_actions`; both surfaces call it; perf guard spies its call count (3×).
- 🟠 **SPADL-vs-tracking two-frame trap.** Fixed: explicit documented invariant + targeted e2e
  forward-pass LBS assertion (not just AUC).
- 🟡 **Opponent helper** = `~ids_match(series, scalar)` with `~is_ball` first (verified
  `_id_compat.py:118` `ids_differ` is series-vs-series; `_cover_shadows.py:566-571` idiom). Fixed.
- 🟡 **Mirror-direction divergence** from `_line_breaking` — kept (LBS cleaner in attack-positive
  coords; SGM/SDI isometry-invariant), now justified in one sentence.
- 🟡 **LBS 1-D lateral caveat** — added to docstring/NOTICE.
- 🟡 **σ tuning artifact must live in the repo** — added `scripts/tune_structural_pass_sigma.py`.

Reviewer's answers to the original open questions, accepted: σ=15 (ship, no winsorization, persist
artifact); keep crosses in (e2e filters to open-play `pass` for its base-rate/AUC checks); tighten
the e2e SGM ceiling to concrete max≤200 / p99≤20; endorse the explicit `passer_xy`/`receiver_xy`
signature (keep both `attacking_team_id` + `home_team_id` for the mirror).

### Round 2

Verdict: round-1 fixes correct; two should-fix defects + minors, none blocking. All verified against
real code and incorporated:

- 🟠 **S1 — degenerate-defender threshold contradiction** (error-handling said `<1`, testing said
  `<3`; verified the `<3` was inherited from `compute_defensive_line` and is wrong — SGM/SDI are
  defined at ≥1 defender, only ρ=0 divides by zero). Fixed: **0 defenders → all NaN; ≥1 → numeric**;
  and `structural_lbs=0` (defenders present, none in band) is explicitly **distinct** from NaN. Both
  pinned in the oracle test.
- 🟠 **S2 — un-namespaced column contract** (verified peers namespace: `defensive_line_x` etc.).
  Fixed: appended columns are **`structural_lbs` / `structural_sgm` / `structural_sdi`**
  (`…_a0..a2` in xfns), locked in the oracle exact-key-set from commit 1 (Hyrum).
- 🟡 **M1 — `is_default()` cargo-culted** (verified only `PreprocessConfig` has it, for provider
  resolution; `CoverShadowParams`/`LineBreakingParams` do not). **Dropped** (YAGNI).
- 🟡 **M2 — frozen e2e fixture id bit-rots** under GS re-ingest/dedup. Fixed: forward-pass invariant
  **predicate-selected at runtime** (`end_x−start_x > thr` + opponent in band), not a hardcoded id.
- 🟡 **M3 — e2e LBS-AUC is partly tautological** (LBS>0 ⟺ forward pass). Fixed: labelled a
  correctness/regression guard, **not** a paper-reproduction; NOTICE must not cite it as such.
- 🟡 **M4 — hexagonal pure core** added: `_structural_pass_core(defenders_xy, passer_xy, receiver_xy,
  sigma)`, pandas-free; the DataFrame primitive wraps it; it is the primary oracle-test target and a
  clean numba seam.
- 🟡 **M5 — high-NA `*_a1/_a2` slots** documented as expected (VAEP imputation tolerates it).
- 🟡 **M6 — explicit TDD red→green ordering** added (Implementation ordering section) so the plan
  inherits it.
