# TF-54 + TF-55 — Player-Quality Bundle: Territorial Dominance & Duel Ratings

| Field | Value |
|---|---|
| **Date** | 2026-09-03 |
| **Status** | Approved — spec review R2 = APPROVE (2026-09-04); the 3 owner-attributed decisions confirmed by the owner (2026-09-04). See Decision log (§12). |
| **Deciders** | Karsten Nielsen |
| **Bundle** | "Bundle A" — two per-player descriptive metrics grouped by output grain |

---

## 1. Context & goals

silly-kicks has a deep GK/positioning stack and (as of 4.107.0) an event-only GK shot-stopping
metric, but the **coach-facing descriptive cluster** (TF-50/52/54/55/58) is still mostly unbuilt.
This bundle ships the two cleanest per-**player** descriptive quality metrics, grouped by grain so
they share the annoying scaffolding (output schema shape, windowing surface, coverage-report
pattern, frozen-params idiom, glossary/NOTICE/C4 wiring) while keeping independent cores:

- **TF-54 — "Van Dijk" territorial dominance:** how much threat flows *through a defender's own
  territory* — the trimmed convex hull of their own-half defensive-action locations — split into
  threat conceded (opponent passes completed into the hull) vs prevented (failed into the hull).
  Source: Sumpter / Twelve.football "Earpiece" (Soccermatics module 10.2).
- **TF-55 — Glicko-2 duel ratings:** a pairwise skill rating (rating + deviation + volatility) over
  duel win/loss outcomes, updated per match (rating period). Source: Glickman (Glicko-2); StatsBomb
  HOPS pattern (Soccermatics module 10.1).

**Audience:** coach / analyst, defensive-quality lens. Both are descriptive; neither feeds VAEP.

**Why bundle:** same per-player grain → one brainstorm → one spec → one plan → one PR, with the
scaffolding pattern shared. The cores (hull×xT vs Glicko update) and the windowing *semantics*
differ, so the code is **not** shared (see §7 packaging).

## 2. Non-goals / scope boundaries

- **Not** an action-credit metric. Valuing a player's *own* defensive actions by threat is the
  TF-51 defensive-credit family (+ packing, cover shadows) and stays there. TF-54 measures
  opponent-pass outcomes *in a player's territory*, not the player's actions (§6 rejected-alt B).
- **Not** a counterfactual "prevented" in v1. A modeled "threat the defender's presence prevented"
  is the GKDV/ghosting problem in a new setting; it is **reserved behind a typed `method=` door**
  (§5.3) and built later as a construct-validated follow-on, never shipped unvalidated (the xt-gk-v2
  quarantine lesson).
- **Not** aerial duels in v1 (no SPADL aerial-duel type) — ground duels only, coverage stated
  honestly (§5b.2).
- **No** xG/xT/PSxG model shipped — TF-54 injects a fitted `ExpectedThreat` (port pattern), exactly
  as `xt_xfns` / TF-59 do.
- **No** VAEP or tracking retrain; **no** re-materialize (both are additive `compute_*` modules).

## 3. Global constraints

Copied verbatim into every plan task; every task's requirements implicitly include this section.

- **No version/PR-S/ADR number until commit-prep.** The spec/plan reference "the release" and
  "ADR-NNN (assigned at commit-prep after `git fetch && git merge origin/main`)". NOBODY reserves a
  number.
- **Additive — no VAEP/tracking retrain, no re-materialize.** Both are new `compute_*` modules;
  they touch no existing feature values.
- **Event-only.** Both packages import `spadl` / `id_compat` / `xthreat` (TF-54 only, for the
  injected model type) and pandas/numpy/scipy **only** — never `silly_kicks.tracking`. Pinned by a
  per-package AST import-allowlist gate (the `tests/shot_stopping/test_import_allowlist.py` idiom).
- **`compute_*`, not `add_*`.** Neither is an action-coupled aggregator → the C4 aggregator count
  stays **33**; two new containers are added to the C4 model and `architecture.html` re-rendered via
  Graphviz **dot** (never Smetana).
- **ids via `id_compat` (ADR-019).** Any id compare / group / join routes through `id_compat`
  (`ids_equal` / `same_id` / `canonical_id_series`); output ids stay **RAW** (the TF-59 precedent),
  joinable via `id_compat`.
- **Order-insensitive (ADR-065).** Any positional / `.shift()` / trajectory-sequential logic sorts
  chronologically first via the shared robust key.
- **Honest coverage, never fabricate (ADR-042 / ADR-027).** A player/duel/pass that cannot be scored
  is dropped-and-**counted** in a conserving report, never scored as 0 or NaN-silently.
- **Frozen params + `for_provider`** per package (empty override map until an ADR-009 apply-gate).
- **Feature branch, one PR, two commits** — TF-54 then TF-55 (§10); **no worktrees**.
- **Source real e2e data from a public runtime API** (pining, or StatsBomb open data via statsbombpy
  — the sibling restdefense-e2e pattern; importorskip-guarded), never a downloaded/committed data
  folder. (IMPL-10 reconciliation: TF-54's e2e uses statsbombpy; a native-duel provider for TF-55 may
  still come via pining.)

## 4. Architecture overview

Two sibling packages, one shared cycle:

```
silly_kicks/territory/        # TF-54
  __init__.py                 # flat __all__
  _columns.py                 # schema dict + column-name constants
  _config.py                  # TerritoryParams (frozen) + method family params
  _compute.py                 # compute_territorial_dominance(...)
  _report.py                  # TerritoryReport (conserving)
  _hull.py                    # trimmed-hull geometry (scipy ConvexHull + point-in-hull)
silly_kicks/duels/            # TF-55
  __init__.py
  _columns.py
  _config.py                  # DuelRatingParams (frozen; Glicko-2 constants)
  _compute.py                 # update_glicko(...) primitive + compute_duel_ratings(...) orchestrator
  _report.py                  # DuelRatingReport (conserving)
  _extract.py                 # duel extraction (native winner/loser + derived adjacency)
```

Both adopt a **windowed-API-over-atoms** shape — **NEW to this bundle, not an inherited pattern**
(no `window=` aggregation compute exists in silly-kicks today): a per-`(player, match)` atom, plus
an optional window aggregation (`window=` argument) whose semantics differ per metric (§5.5 / §5b.4).

Package/module names are finalizable at plan time (`territory`/`territorial`,
`duels`/`duel_rating`); this spec pins the CONTRACTS, not the exact identifiers.

---

## 5. Feature A — TF-54 Territorial Dominance

### 5.1 Grain & signature

```python
def compute_territorial_dominance(
    actions: pd.DataFrame,
    *,
    xt: ExpectedThreat,                 # injected fitted model (require_fitted_xt), port pattern
    method: str = "completed_failed",   # valuation family (§5.3)
    window: Collection | None = None,    # a collection of game_ids to pool; None -> per-(player, game) atoms (§5.5)
    params: TerritoryParams = _DEFAULT,
) -> tuple[pd.DataFrame, TerritoryReport]:
```

- **Atom grain:** one row per `(game_id, player_id)` — the defender whose own-half defensive-action
  centroid defines the hull.
- `ExpectedThreat` is imported `TYPE_CHECKING`-only and duck-typed at runtime (the ADR-022 idiom);
  `require_fitted_xt(xt, caller="compute_territorial_dominance")` fails closed on `str`/`None`/unfitted.

### 5.2 The trimmed defensive hull (`_hull.py`)

- **Defensive-action set:** `tackle`, `interception`, `clearance` (pinned constant; the SPADL types
  confirmed present). Own-half only (action `start_x < 52.5` in the player's action-LTR frame —
  the metric is about the player's *defensive* territory).
- **Trim:** keep the **70%** of that player's own-half defensive-action locations nearest their
  centroid (`trim_fraction=0.70`, frozen param); convex hull of the survivors (`scipy.spatial.ConvexHull`).
- **Degenerate guard:** < 3 non-collinear survivors → the hull is undefined → the player's row is
  **dropped-and-counted** in `TerritoryReport` (`n_degenerate_hull`), never a fabricated 0/NaN.
- **`hull_source`** provenance column over `{resolved, degenerate, no_actions}` (the `das_source`
  idiom).

### 5.3 The `method=` valuation family (default + reserved door)

Mirrors `ExpectedThreat(method=…)` (ADR-021) — **single-method** string-dispatch + per-method frozen
params, **not** ABCs. (`ExpectedThreat` is the right precedent; `add_pressure_on_actor` uses `methods=`
plural/multi-emit, a different shape.)

- **`method="completed_failed"` (DEFAULT, ships v1):** for each **opponent** pass whose SPADL `end`
  lands inside the hull — for a completed pass the reception; for a failed pass the **death/recovery
  location** (SPADL derives a pass end from the NEXT action's start, `_derive_end_coordinates`, so a
  failed pass's `end` is where the ball died, not the intended target):
  - `xt_conceded += xt(end_zone)` if the pass **completed** — threat that REACHED the territory;
  - `xt_prevented += xt(end_zone)` if the pass **failed** — threat that DIED in the territory (the
    ball entered the defender's zone and was recovered there);
  - `xt_net = xt_conceded − xt_prevented`.
  **Both membership and the xT lookup use the same recorded `end` POINT, for completed and failed
  passes alike** — pinned so the two legs never use different coordinates (never intended-target for
  one, death location for the other). This is a SEPARATE axis from the frame reconciliation below,
  which expresses that same point in different ORIENTATIONS (defender's frame for membership,
  opponent's frame for xT).
  - **SPADL `end` ≠ intended target for a failed pass (SHOULD-FIX 1 — validity, corrected gloss):**
    the default therefore measures *"threat that reached / died in the territory"*, **NOT** *"threat
    that would have been created"* (the earlier gloss was wrong for SPADL). Documented consequence,
    an UNDER-count and never a silent error: a pass AIMED into the zone but intercepted AT/BEFORE the
    boundary has its death location outside the hull → it is not counted as prevented. This is exactly
    the validity gap the reserved `counterfactual` method exists to close with a model; the default is
    the honest SPADL-supported proxy, with the limitation stated in the docstring + surfaced in
    `TerritoryReport`. The Tier-1 exact test (§8) uses a failed pass whose `end` ≠ its intended target
    to pin this behaviour.
  - **Coordinate reconciliation (correctness-critical, ADR-028 class):** the hull is built in the
    *defender's* action-LTR frame (defender's team attacks x=105) while the *opponent's* passes are
    in the opponent's frame — a 180° point reflection apart, so the two purposes must not be conflated:
    - **Hull membership** reflects the opponent pass `end` into the defender's frame
      `(105 − end_x, 68 − end_y)` before the point-in-hull test.
    - **xT value** = `xt.rate` of the pass `end` in the opponent's OWN frame (opponent attacks x=105)
      — the threat at the point the ball reached/died.
    Guarded by a per-row reflection-invariance test: one physical scene scored from either team's
    perspective yields identical conceded/prevented (an opponent pass and its reflected twin score the
    same). The exact defect class ADR-028/ADR-051-D3 exist for.
- **`method="counterfactual"` (RESERVED, typed door, NOT built v1):** modeled threat-prevented.
  Registered in the family so it is a first-class alternative, but raises
  `NotImplementedError("counterfactual prevented is a construct-validated follow-on; see spec §2")`
  until its own validated cycle. Adding it later is purely additive (no default change, no retrain).

### 5.4 Enrichments (within the territorial construct)

- **Forward flag:** a completed forward ball into the zone is worse than a square one. Emit
  `xt_conceded_forward` / `xt_prevented_forward` companions (forward = destination x greater than
  origin x by `forward_threshold_m`, frozen param), alongside the totals.
- **Volume + rate:** `passes_into_hull` (count faced) + `xt_conceded_rate` / `xt_prevented_rate`
  (per pass into hull) — so "territory opponents can't pass into" (low volume) is distinguishable
  from "leaky but rarely tested". A zero-volume hull → rates are honest-NaN (0/0), not 0.
- **Hull geometry:** `hull_area_m2`, `hull_centroid_x`, `hull_centroid_y` (territorial-size context).
- **Descriptive context (not a valuation):** `defensive_actions_in_hull` count. **No** second xT
  valuation of the player's own actions (that is TF-51).

### 5.5 Windowing

`window=None` → per-`(player, game)` atoms. A `window` = a `Collection` of `game_id`s to pool (a
consumer resolves any date-range to ids) → **additive
aggregation**: sum `xt_conceded`/`xt_prevented`/`passes_into_hull` over the window's games, recompute
`xt_net` and the rates; the hull + geometry are **re-derived over the window's pooled actions**
(a season hull, not an average of match hulls).

### 5.6 Columns (glossaried; names finalizable at plan time)

`territory_xt_conceded`, `_xt_prevented`, `_xt_net`, `_xt_conceded_forward`, `_xt_prevented_forward`,
`_passes_into_hull`, `_xt_conceded_rate`, `_xt_prevented_rate`, `_hull_area_m2`, `_hull_centroid_x`,
`_hull_centroid_y`, `_defensive_actions_in_hull`, `_hull_source` (+ `game_id`/`player_id` RAW,
`window_*` when aggregated). Dtypes: ids `object`; counts `Int64`; xT/area/coords `float64`;
`hull_source` `object`.

---

## 5b. Feature B — TF-55 Glicko-2 Duel Ratings

### 5b.1 Primitive + orchestrator

```python
def update_glicko(
    ratings: Mapping[player, GlickoState],
    period_games: Sequence[Game],       # (player_a, player_b, score_a) for one rating period
    *,
    params: DuelRatingParams,
) -> dict[player, GlickoState]:          # pure, stateless — the Glicko-2 math for ONE period

def compute_duel_ratings(
    actions: pd.DataFrame,
    *,
    initial_ratings: Mapping[player, GlickoState] | None = None,  # None -> Glicko-2 defaults; else RESUME
    window: Collection | None = None,
    params: DuelRatingParams = _DEFAULT,
) -> tuple[pd.DataFrame, DuelRatingReport]:   # walks matches chronologically; per-(player, match) snapshots + final
```

- **Primitive** = the Glicko-2 update for one rating period, testable in isolation against
  Glickman's published worked example (the "primitive + assembly" house pattern).
- **Orchestrator** = walks matches in chronological order (ADR-065 sort), threads state, emits the
  per-`(player, match)` rating snapshots (the atoms) + the final ratings. `initial_ratings` lets a
  later batch **resume** (a rating system is long-lived — add matchdays, don't recompute the world).

### 5b.2 Duel extraction (`_extract.py`)

- **Native:** sportec `tackle_winner_*` / `tackle_loser_*` (ADR-001 sportec-specific columns; NaN
  elsewhere) → a clean winner/loser game.
- **Derived:** for providers without native columns, derive from the `tackle` / `take_on` result
  adjacency (result decides winner: a successful tackle beats the dribbler; a successful take_on
  beats the tackler). The strategy is chosen at **frame-set granularity** (native-if-present else
  derive), never per-duel-guess.
- **Ground duels only** (no SPADL aerial type). **Indeterminate duels excluded** (a 50-50 / loose
  ball / result-ambiguous contest with no clear winner) — a Glicko "game" must be a genuine
  win/loss; excluded count surfaced in the report (**owner ruling, TF-54/55 brainstorm 2026-09-03** —
  exclude, not draw=0.5; captured here as the record the plan reads, to be formalized in the cycle's
  ADR at commit-prep).

### 5b.3 Glicko-2 constants (frozen `DuelRatingParams`, `for_provider`-ready)

Initial rating **1500**, RD **350**, volatility σ **0.06**, system constant **τ** exposed
(Glickman default ~0.5); **inactivity RD growth** applied for a player who contests no duel in a
rating period (gold-standard Glicko-2 — uncertainty widens); rating period = **match**; each duel =
one game within the period.

### 5b.4 Windowing

`window=None` → per-`(player, match)` rating snapshots. A `window` (a `Collection` of `game_id`s) → **trajectory slice**:
rating-as-of-window-end (default) or rating-change-over-window (a `window_stat=` param) — NOT an
additive aggregation (ratings are cumulative, not summable). **Scope note:** the per-match rating
trajectory IS the TODO's "rating period = match"; the `window=` slice is a thin convenience over those
atoms (the owner-approved "windowed API over atoms" grain decision), not a new estimator — it extends
the TODO baseline only by that slice helper.

### 5b.5 Columns (glossaried)

`duel_rating`, `duel_rating_deviation`, `duel_volatility`, `duels_contested`, `duels_won`,
`duels_lost`, `duel_winner_source` (`native`/`derived`) (+ `game_id`/`player_id` RAW, `window_*`).
Dtypes: ids/source `object`; counts `Int64`; rating/RD/volatility `float64`.

---

## 6. Rejected alternatives

| Option | Why rejected |
|---|---|
| **B — own-action xT credit inside TF-54** | Already the TF-51 defensive-credit family + packing + cover shadows. Folding it in duplicates TF-51 and blurs "territorial dominance". Offered in error during brainstorm; dropped. At most a descriptive *count* of hull actions, never a second valuation. |
| **C — counterfactual "prevented" built in v1** | An event-only counterfactual is the GKDV/ghosting problem in a new setting; shipping it unvalidated is the xt-gk-v2 quarantine trap. Kept as a **reserved `method=` door** (§5.3), built later as a construct-validated follow-on. |
| **One shared package for both metrics** | The cores (hull×xT vs Glicko) and the windowing *semantics* (additive vs trajectory-slice) differ → no real shared code. Two sibling packages keep separation of concerns and match the `shot_stopping`/`restdefense` idiom. |
| **Per-method ABCs for the TF-54 family** | The house style is string-dispatch + frozen-dataclass params (ADR-021), not ABCs. |
| **Draw=0.5 for indeterminate duels** | "Indeterminate" is not "even"; a draw dilutes the signal. Exclude + count (owner ruling). |

## 7. Packaging & bookkeeping

- **Two sibling packages** (§4), one feature branch, one shared spec, one PR, **two commits**.
- **C4:** two new containers (`territory`, `duels`); aggregator count stays **33**;
  `architecture.html` re-rendered via Graphviz `dot`.
- **`feature_glossary`:** every emitted column documented (a new `compute_*` leg per package) +
  `describe_level` direction metadata (higher-is-better: duel_rating yes; xt_conceded no;
  xt_prevented yes; xt_net context-dependent — pin per column).
- **`NOTICE`:** Sumpter/Twelve (TF-54) + Glickman Glicko-2 (TF-55) entries; per-feature docstrings
  cross-link "See NOTICE …".
- **Additive — no retrain, no re-materialize.**

## 8. Testing & validation (two tiers; the real-data e2e commit gate)

Per the **owner ruling (TF-54/55 brainstorm, 2026-09-03; captured here as the record the plan reads,
to be formalized in the cycle's ADR at commit-prep)**: **each of the two commits is approved only with complete e2e
testing on real data, green, before the commit** — the feature's e2e run, plus the full non-e2e
suite. Interpretation: *complete e2e coverage of the feature being committed on real data*, not a
re-run of every unrelated repo e2e.

**Tier 1 — committed analytic/unit fixtures (CI, every leg):**
- TF-54: exact hull on a hand-placed action set; a known opponent pass into/out of the hull with a
  toy xT → exact conceded/prevented/net; **a FAILED pass whose SPADL `end` (death location) ≠ its
  intended target** — pins death-location membership + xT + the documented under-count (SHOULD-FIX 1);
  forward-flag; degenerate-hull drop-and-count; rate NaN on
  zero volume; RAW-id + mixed-dtype id grouping (ADR-019); `require_fitted_xt` raises on
  `None`/`str`/unfitted; purity (no input mutation, ADR-033); import-allowlist (never `tracking`).
- TF-55: `update_glicko` vs **Glickman's published worked example** (the canonical Glicko-2
  numeric check); resume-equivalence (one batch of N matches == two batches threaded via
  `initial_ratings`); inactivity RD growth; native-vs-derived winner extraction; indeterminate
  exclusion counted; chronological-order-insensitivity (ADR-065); purity; import-allowlist.
- **Red-green** on the load-bearing assertions (perturb the sign / the winner → fail → restore).

**Tier 2 — `@e2e` on real data (pining-sourced; run before each commit):**
- TF-54: a real match + a **fitted `ExpectedThreat`** (fit on a pining corpus via the
  `FrozenXt`/calibration idiom, or load a fitted artifact — the spec pins the reproducible recipe).
  Assert the metric runs end-to-end, columns/dtypes conform, values are finite/plausible, and the
  coverage census conserves.
- TF-55: a **native-duel provider (sportec)** and a **derive-from-adjacency provider**, both pining
  sourced, to exercise both winner paths; assert ratings evolve, the report's native/derived/excluded
  counts conserve, and resume-equivalence holds on the real trajectory.

**Prerequisites flagged now** (so the gate is not blocked at commit time): a reproducible fitted-xt
recipe for TF-54's e2e; sportec + one derive-provider match access via pining for TF-55's e2e.

**Standard gates** (both commits): glossary coverage + NOTICE-linkage, C4 completeness (dot render),
public-api-examples registration, id-dtype-invariance, `_no_e2e` full suite green, pyright 0, ruff
clean.

## 9. Commit plan

- **One feature branch** off the merged default branch; **one PR**.
- **Commit 1 — TF-54** (`silly_kicks/territory/` + tests + glossary/NOTICE/C4 slice): Tier-1 green
  in CI **AND** the TF-54 real-data e2e green → owner-approval gate → commit.
- **Commit 2 — TF-55** (`silly_kicks/duels/` + tests + glossary/NOTICE/C4 slice + commit-prep docs
  bump): Tier-1 green **AND** the TF-55 real-data e2e green → owner-approval gate → commit.
- Non-squash merge so both commits survive; version/PR-S/ADR assigned at commit-prep.
- **No commit without explicit per-commit owner approval** (standing rule) — each gate presents the
  diff + the real-data e2e evidence and waits.

## 10. Open / plan-time details

- Final package + column identifiers (§4/§5.6/§5b.5).
- ~~`WindowSpec` shape~~ **RESOLVED (PLAN-06):** `window` is a plain `Collection` of `game_id`s (no
  formal type; a consumer resolves any date-range to ids); per-metric aggregation (TF-54) / slice
  (TF-55) semantics, tested both sides.
- ~~The exact fitted-xt e2e recipe~~ **RESOLVED (PLAN-07; e2e substrate reconciled per IMPL-10):** the
  TF-54 e2e fits `ExpectedThreat(method="singh_counts")` (the classic deterministic grid) on
  **StatsBomb open data via statsbombpy** (the sibling restdefense-e2e pattern; a public,
  importorskip-guarded runtime API — NOT a downloaded folder), all-but-the-scored-match, with the
  scored match EXCLUDED from the fit; no bundled artifact. Fallback: the `@e2e` skips cleanly if
  statsbombpy / the network is unavailable.
- `for_provider` stays empty (ADR-009) — no per-provider tuning this cycle.

## 11. References (NOTICE)

- **TF-54:** Sumpter, *Soccermatics*; Twelve.football "Earpiece" glossary (module 10.2).
- **TF-55:** Glickman, M.E., *Glicko-2* rating system; StatsBomb HOPS (module 10.1).

## 12. Decision log

- **2026-09-03 (brainstorm):** grain = windowed-API-over-atoms (both); TF-54 `method=` family
  (default `completed_failed`, `counterfactual` reserved), B→TF-51, C→reserved; TF-55 primitive +
  resumable orchestrator; indeterminate duels excluded; two sibling packages.
- **2026-09-03 (owner):** complete real-data e2e before each of the two commits.
- **2026-09-04 (owner):** the three owner-attributed decisions above (indeterminate=exclude;
  e2e-per-commit; windowed-API grain) **confirmed** by the owner in the spec review, closing the
  R2-APPROVE contingency (SPEC-02). Formal durable sink remains the **commit-prep ADR** (§3 forbids
  assigning the ADR number now).
- **2026-09-04 (plan review R1):** `window` simplified from a formal `WindowSpec` to a plain
  `Collection` of `game_id`s (PLAN-06); fitted-xt e2e recipe pinned to `singh_counts` on a pining
  public corpus (PLAN-07).
