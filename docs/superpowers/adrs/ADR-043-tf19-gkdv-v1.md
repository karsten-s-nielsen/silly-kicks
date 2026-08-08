# ADR-043: TF-19 GKDV v1 — ghost-substitution engine + two gate-independent physics arms

| Field | Value |
|---|---|
| **Date** | 2026-07-18 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen (owner) |
| **Supersedes / amends** | Completes **ADR-037**'s PR-3 (the `gkdv → tracking` direction rule) on top of **ADR-040**'s corrected weights; **amends ADR-019** — `_id_compat` is promoted to the public `silly_kicks.id_compat` (breaking, no shim) and its AST lint is DELETED for an enumeration registry, which also retires the ADR-027 note that the lint is the incomplete heuristic; retrofits **ADR-020** onto `das_xfns`; extends **ADR-005** (feature surface). Not engaged: **ADR-033** (`add_*` purity — no `add_*` shipped) |
| **Source plan** | `docs/superpowers/plans/2026-07-18-tf19-gkdv-pr3-implementation.md` |

## Context

TF-19 ("GK Deterrent Value") is the headline metric of the GKDV research arc (TF-15..TF-19): a
per-frame measure of how much the defending goalkeeper's *position* — not their shot-stopping —
depresses what the attacking team can do. The arc's Layer-2 **attempt arms** (xS shot-occurrence,
xCross cross-attempt) were built to carry that signal, and both are **gated**.

ADR-037 (PR-1) and ADR-040 (PR-2) closed the re-gate. The bundled xS/xCross/ghost weights had been
trained y-mirrored and served y-correct since ADR-031, so every prior GK measurement was taken on a
mis-served surface; PR-2 retrained all three and re-ran the frozen probe. The corrected xCross
surface **strengthened to ratio ≈2.21×** (clearing the 2.0× prong) but its `gk_median_abs_delta` of
**0.009697** still misses the pre-registered **0.01** absolute floor — by ~10% *relative*, not the
~10× of the pre-retrain measurement. Verdict: `tf19_ready = false`, classified `gated_clean_fail`,
independently corroborated by the ADR-015 causal harness (`gk_clears_placebo_band = False`). The xS
arm has **never been measured at all** — it needs a ghost-substitution engine that did not exist.

So TF-19 faced a genuine fork: the attempt arms are the specced route to the metric, one is gated
and the other unmeasurable. Two things were nonetheless available immediately. First, the **physics**
arms — ΔDAS and Δthreat-suppression — are *gate-independent*: they read pitch control and accessible
space directly rather than a trained attempt-probability surface, so no probe verdict blocks them.
Second, the xS probe's blocker is precisely the **ghost-substitution engine**, which both physics
arms need anyway. Building the engine therefore ships value now *and* unblocks the measurement that
would settle the attempt arm.

## Decision

Ship `silly_kicks/gkdv/` containing the ghost-substitution engine (`build_ghost_frames`) and the two
**gate-independent physics arms** (`delta_das`, `delta_threat_suppression`), both defined in
attacker-value units as `actual − ghost` so **negative = deterrent** uniformly.

`gkdv/` depends on `silly_kicks.tracking` **public seams**, on the repo-wide public
`silly_kicks.id_compat` (ADR-019 requires every consumer to route id comparisons through it), and
on **exactly one** private tracking symbol: `_das._pin_attacking_direction`, confined to
`_das_port.py`. That one has no public meaning because it encodes what the *optional*
`accessible-space` dependency expects of its input — promoting an optional dependency's input
contract would be wrong. **Never the reverse**; both directions are import-allowlist gated.

An earlier draft of this line read "public seams only", which was false while the allowlist carried
three private modules. Two of the three were genuinely-public seams sitting behind an underscore
and were promoted (`id_compat`, `defended_goal_x`); the exit-condition framing of the allowlist is
what had let them sit there indefinitely, so its review question is now *"is this genuinely
internal and confined?"* rather than *"is the debt documented?"*.

PR-3 ships **without** spec §6.4 Layers 0–3 (the composed headline metric) and **without** the
xS-arm probe, which is PR-3b.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Wait for the attempt arm to clear its gate before building anything | No risk of shipping a metric on a failed arm | The gate is failed on an *absolute floor*, and clearing it needs GK feature-engineering of unknown duration; meanwhile the xS arm stays permanently unmeasurable because its probe needs the engine | Blocks indefinitely on work that the engine itself unblocks |
| B. Ship the full §6.4 composed metric (Layers 0–3) now | One PR, headline deliverable | Composition weights the attempt arms, one of which is `gated_clean_fail` and one unmeasured — the composite would inherit a known-gated component and its validity would be unassessable | Would ship a headline number whose components are not certified |
| C. Put the engine inside `tracking/` next to `_ghost_gk.py` | No new package; direct access to privates | ADR-037 already set the rule that `tracking/` must never import `gkdv/`; co-locating invites the reverse edge and makes the boundary unenforceable | Destroys the dependency rule the allowlist test exists to pin |
| D. **Engine + two gate-independent physics arms in a new `gkdv/` package; composition and the xS probe deferred** | Ships real, ungated value; unblocks the xS measurement; the dependency rule is mechanically enforceable | Two PRs instead of one; the headline metric is still pending | — |

**D1's remaining justification had to be re-stated, because the first one named a dependency that
does not exist.** The draft read *"`gkdv/` is library code and cannot import from `scripts/`"* — but
`gkdv/` does not import `_group_metrics` at all: `grep -rn "_group_metrics" silly_kicks/gkdv/`
returns nothing, and `_validate.py` computes no ICC. The anticipated `gkdv → _group_metrics` edge
never materialised in PR-3. The lift stands on what is actually true: the *shipped* consumers are
`scripts/xtgk_v2_keeper_discrimination.py` and the `tests/gkdv/` + `tests/xtgk/` suites, and before
the lift the statistics lived in an **unversioned script** that the test suite could not import and
the published wheel does not ship (`scripts/` is outside the `silly_kicks/` package). Putting them
in the library makes them the single source and puts them under test; the direction of the
`scripts/` dependency is now inverted, which is the whole point. *(The same superseded sentence
also appears in `silly_kicks/_group_metrics.py`'s module docstring — a source file this
documentation pass may not touch. Flagged for the owner.)*

Two sub-decisions inside D were also contested:

| Sub-decision | Chosen | Why not the alternative |
|---|---|---|
| DAS route (spec §5 left it open) | `get_individual_das(..., attacking_direction_col=…)` summed per team, behind `gkdv/_das_port.py` | The alternative required **editing `silly_kicks/tracking/_das.py`** to thread a direction pin through `get_das`. The chosen route needs **no `_das.py` edit at all** — strictly smaller blast radius on a module with a live downstream consumer history. `get_das` hardcodes `infer_attacking_direction=True` and cannot accept a pin, so it was never viable. |
| `_group_metrics.py` visibility (D1) | **PRIVATE** `silly_kicks/_group_metrics.py` | The original justification was downstream consumption. Confirmed with the lakehouse session 2026-07-18: they have **zero** matches for `icc` / `intraclass` / `keeper_spread` / `group_spread` in code, dbt or docs, and no foreseeable intent — per-keeper aggregates are dbt models and an ICC is a model-validation statistic they consume as a *verdict*, not a computation. Their words: **"don't mint a surface for us."** The downstream justification was therefore factually wrong and is dropped. |

## Consequences

### Positive

- **The xS arm becomes measurable for the first time.** `build_ghost_frames` is exactly the
  substitution engine the xS probe needs; PR-3b can now run it.
- **Two ungated signals ship.** ΔDAS and Δthreat-suppression read physics, not a trained attempt
  surface, so neither inherits the `gated_clean_fail` verdict.
- **The `gkdv → tracking` direction is mechanically enforced**, not merely documented
  (`tests/gkdv/test_import_allowlist.py`), continuing the rule ADR-037 set for `tracking/_model_eval.py`.
- **19 previously-dark DAS tests now run in CI**, and 20 more are shown to be *still* dark for a
  different reason (see the D4 measurement below, which corrects the plan's figure).

### Negative

- **The composed headline metric is still not shipped.** TF-19's deliverable remains open; PR-3b plus
  an owner validation run are required.
- **The two arms model the keeper asymmetrically, and this is inherent, not a bug.**
  `delta_threat_suppression` carries `lambda_gk = 3.0` (Shaw: `lambda_gk = 3 × lambda_outfield`) — a keeper *gain* applied after the influence field,
  so keeper position enters via TTI. `delta_das` is **keeper-blind**: accessible-space's column map
  has no `is_goalkeeper`, so ΔDAS measures the accessible-space consequence of relocating a
  **generic player**. Consumers must not read the two arms as measuring the same construct. Because
  `lambda_gk` is a gain rather than a mechanism, the S9 sensitivity leg is a **gain sweep**, not a
  mechanism probe.
- **We now carry a private-consumer register** (`docs/PRIVATE_CONSUMERS.md`) that will drift and needs
  periodic re-confirmation with the lakehouse.

### Neutral

- **`gkdv` ships NO `add_*` action-coupled aggregator.** Verified:
  `[n for n in silly_kicks.gkdv.__all__ if n.startswith('add_')] == []`. Therefore `PURITY_ENTRIES`
  (ADR-033) and the tracking liveness registry are **untouched** by this cycle, and the C4
  action-coupled aggregator count is **unchanged BY THIS CYCLE**. The absolute number is **30**.
  **The re-derivation is two steps, not one, and an earlier draft stated only the first** —
  `[n for n in tracking.__all__ if n.startswith("add_")]` yields **31**, and the C4 DSL convention
  then excludes `add_gradientsports_player_ids`, which is a jersey-number id helper rather than an
  action-coupled aggregator, giving **30** (matching `docs/c4/architecture.dsl`). Quoting the
  prefix filter alone makes the ADR's own method disagree with its own number. A still earlier
  draft read "unchanged at 29"; that was written pre-merge of PR-S119 — whose
  `add_off_ball_run_values` took the count 29 → 30 — and was stale, which is precisely why the rule
  is to re-derive the count rather than copy it from any written source, this ADR included.
  gkdv's Examples-gate coverage is the **four modules that
  actually define its public surface** (`_arms`, `_engine`, `_metric`, `_validate`), not
  `gkdv/__init__.py`. An earlier draft registered the package `__init__` "following the
  `causal/__init__.py` precedent" — but that precedent WAS the defect: a package `__init__` that
  only re-exports has zero top-level defs, so its parametrized entry asserted nothing while
  reading as coverage, leaving every module that defines the surface unchecked. Registering the
  four real modules took gkdv from **0 to 8 enforced symbols**; the `causal/__init__.py` entry was
  removed the same way (its real modules were already registered, so no coverage was lost), and
  `test_no_registered_entry_is_vacuous` now blocks the shape from recurring.
- Neither arm is in any default xfn list, so nothing here is a VAEP retrain trigger.

## Decisions that would otherwise read as incidental notes

**1. `method="spearman"` is a HARD API constraint, not a recommendation.** It is enforced in
`GkdvParams.__post_init__`, so a GK-blind pitch-control configuration is **unrepresentable** rather
than merely discouraged. The arms deliberately do **not** re-check at call time: a duplicated check
would be a second source of truth that could drift from the dataclass allowlist — the one place a
future GK-aware method would be registered. (`lambda_gk` exists only on `SpearmanParams`.)

**2. The NaN-ghost is dropped-and-COUNTED, never scored as Δ = 0.** A missing or non-finite served
ghost yields a counted `drop_reason`, and a non-finite ghost coordinate **raises** rather than making
the keeper vanish from the frame. Scoring such a frame as zero would read as *"this keeper provided
no deterrence"* and bias per-keeper aggregates toward the null. `GkdvReport` conserves exactly:
`n_frames_scored + Σ drop_reasons == n_frames_in`. Because `counterfactual_frames` is the FULL input
with only the defending keeper substituted, a dropped frame is byte-identical across the two legs —
so consumers **must** restrict to the scored set (`provenance_to_targets`) before differencing, or
they will silently re-admit exactly the Δ = 0 rows the domain exists to exclude.

**3. Neither arm accepts a `pitch_control_cache`, and the reason is CORRECTNESS, not performance.**
`PitchControlCache` keys on `(game_id, period_id, frame_id, team, method, params, ball_position,
decompose)` — which **excludes player positions**. A ghost frame carries the same frame identity as
its factual twin, so a shared cache would serve the counterfactual leg the *factual* leg's surface
and **every delta would collapse to exactly zero with no warning**. This is the same silent-null
shape as ADR-031's y-inversion and ADR-036/PR-S113's fabricated grid origin.

**4. `gkdv/_das_port.py` is the single narrow port onto accessible-space, and the reason is
testability of a live hazard — not tidiness.** accessible-space infers playing direction per period
from a team-x mean that the ghost displacement perturbs, so the two legs could infer **opposite**
directions and the "difference" would not be a counterfactual at all. The port lets the structural
direction-pinning guard run on **every** CI leg rather than only where an optional extra happens to
be installed. Only `_pin_attacking_direction` is a private import; `get_individual_das` is already a
public `silly_kicks.tracking` seam and is consumed as one — keeping the private-exemption surface as
small as it can honestly be.

**5. D4 — `[das]` was installed on ZERO CI legs before this PR.** Measured, not assumed: the TF-28
DAS suites are all `importorskip`-guarded, so absent the extra they **skipped** rather than failed,
and had therefore never run in CI since TF-28 shipped. **The plan's headline figure of 71 does not
reproduce and is corrected here.** Re-measured by AST census over every
`pytest.importorskip("accessible_space")` site (module-level, function-level and fixture-level)
across the five gated files — `tests/tracking/test_das.py`, `test_das_e2e.py`, `test_das_offside.py`,
`tests/invariants/test_das_invariants.py`, `tests/calibration/test_features.py` — at `HEAD`,
i.e. before this cycle's edits:

- **39** test functions were gated on the extra.
- **19** of them carry no `e2e` marker, so they sit inside CI's own `-m "not e2e"` selection and
  were skipping on every leg. Installing the extra activates exactly these.
- **20** are `@pytest.mark.e2e` (all of `test_das_invariants.py` and `test_das_e2e.py`, plus 7 in
  `test_das.py`) and remain excluded by the marker filter. **Installing the extra does not
  activate them** — the dependency was never what was keeping them out, and a reader who takes
  "71 now run" at face value would conclude the DAS suite is fully covered in CI when roughly a
  third of it is still out of reach for an orthogonal reason.

A further 32 tests in `test_das.py` were never gated on the extra at all and had been running
throughout — the count of *dark* tests and the count of *DAS* tests are not the same number, which
is how 71 became quotable. The fix is one token in
`.github/workflows/ci.yml` (`".[kloppy,xgboost,test]"` → `".[kloppy,xgboost,das,test]"`). **Scope
note, recorded because it differs from the plan:** the plan proposed activating them on the ADR-023
primary leg only; the change as shipped installs `das` on **every** leg, because gkdv's DAS arm is a
second consumer of that subsystem and a guard that never runs is not a guard. Any pre-existing
failures the activation surfaces are **reported, not fixed here** — they predate this branch.

**6. A reasoned NO-GATE decision (Task 3, velocity-state defending split).** Two of the three
hardened id-compare sites are mutation-killable; the third is not, **by construction** — its
`gk_team` scalar comes from the same frame it is compared against, so no input can distinguish
`ids_match` from `==`. It is equally not lintable without flagging safe code: the same-source
column-vs-scalar shape is *syntactically identical* to the unsafe cross-source one. The change is
kept for **consistency** — `TestExtractionRestriction`'s golden requires this block's identity rule
to match the extractor's, which Task 3 changed. It therefore ships **deliberately ungated**, recorded
here so a future reader meets the reasoning rather than discovering an untested line and "fixing" it.
Cross-reference **ADR-027**, which established that the name-heuristic id-compat lint is incomplete
and the *behavioural* gate is the real backstop.

**7. `_ghost_gk.py`'s module PATH is pinned downstream.** The luxury-lakehouse ADR-044 executor-env
drift guard (`src/ingestion/exec_visibility.py:467-472`) hardcodes the module paths of
`_ghost_gk.py`, `_xt_gk.py`, `_gk_completion.py` and `_gk_geometry.py` as strings. This PR **modifies**
`_ghost_gk.py` but deliberately does **not** rename or relocate it. A rename would degrade their guard
**silently** — no `ImportError`, just a guard that stops guarding. Any future refactor of these four
modules is cross-repo coordination.

**8. `docs/PRIVATE_CONSUMERS.md` was created in this PR.** The `_ghost_gk` path pin above was found
**by asking a question**, not by any standing mechanism — nothing in either repo would have surfaced
it. The register records what we now know so the next refactor has a known blast radius instead of a
guess. It is **silly-kicks-side bookkeeping the lakehouse can correct**, not a negotiated contract.
One row was verified and **retired** while writing it: the `tracking/_das.py::get_das` coupling carried
the consumer's own stated exit condition *"switch back once `add_das` exposes `chunk_size`"* — and
`add_das` exposes `chunk_size` on `origin/main` (`ec543cc`, 4.51.0,
`silly_kicks/tracking/features.py:2478`), so the condition is met and the row was recorded as retired
rather than logged as a stale live coupling.

**9. `_id_compat` is promoted to the public `silly_kicks/id_compat.py` — a BREAKING import path,
and deliberately without a compatibility shim.** ADR-019 does not merely *permit* consumers to
route id comparisons through this module; it makes doing so **mandatory**. A seam every consumer is
required to use is public API whatever its filename, so the leading underscore was a false signal
about stability, not a true one. It was structurally wrong as well as semantically: measured by AST
over `silly_kicks/`, **39 modules across 6 packages** now import it — `spadl/`, `vaep/`, `atomic/`,
`causal/`, `gkdv/` and `tracking/` — so five packages were reaching into a private *tracking*
submodule, two of them through function-local imports inside `spadl/utils.py` written to dodge a
circular import. **Public naming, not a private relocation, was the requirement.** Moving it to a
private `silly_kicks/_id_compat.py` was considered and rejected: it would have relocated the
problem while leaving the "public seams only" claim in this ADR's own Decision section false, and
`gkdv/` would still have needed a private-import allowlist entry for it — the very exemption the
allowlist review question exists to eliminate. **No shim, because the failure mode is already
loud.** The one known downstream pin is an `import`, which fails at collection with `ImportError`;
the silent-degradation risk `docs/PRIVATE_CONSUMERS.md` exists to catch belongs to the
*path-string* pin in `exec_visibility.py`, which this does not touch. A shim would also have made
the promotion **cosmetic** — nothing would ever migrate, and the private module would remain the
real one indefinitely.

**10. The ADR-019 AST lint was DELETED, not widened, and the reason is that widening is
impossible in principle rather than merely expensive.** `tests/tracking/test_id_compat_lint.py`
is gone. It was a **NAME heuristic**: it missed the ADR-027 `t != action_team` defect because the
operands are not named `*_id`, and it could not see `_ghost_gk`'s `str(t) == home_team_id_norm`
because the scalar had been renamed. Those are fixable by widening. What is not fixable is that
**the safe and the unsafe cases are the identical AST** — a same-source column-vs-scalar compare
and a public-parameter cross-source one are syntactically indistinguishable, and only the scalar's
**provenance** separates them. No syntactic rule can see provenance, so any widening either flags
correct code (and breeds exemptions, which is how `ALLOW_MODULES` grew in the first place) or keeps
missing the real defects. Decision 6 above is a live instance: a hardened site that is
*deliberately* ungated precisely because no lint could tell it from the unsafe shape. The
replacement is **complete by ENUMERATION where the lint was incomplete by HEURISTIC** —
`PUBLIC_ID_SCALAR_ENTRIES` in `tests/invariants/conftest_id_scalar.py`, exercised by
`tests/invariants/test_public_id_scalar_registry.py`, which invokes every public function taking an
id-valued scalar and requires identical output across value-equal scalars of different dtypes
(matched, mismatched-but-value-equal, and float axes), with meta-assertions pinning the registry to
the public surface in both directions. Same idiom as ADR-003's NaN-safety registry and ADR-033's
`PURITY_ENTRIES`. It found two live defects on its first run that the lint structurally could not
have seen, both comparing through `str()` rather than naming an id.

**11. The DAS exception catch is narrowed to a single named `DasUnscoreableError`.** `add_das` /
`das_at_action` / `das_xfns` caught `(ValueError, RuntimeError, ImportError, IndexError,
TypeError)` and degraded to an all-NaN column. That tuple swallowed silly-kicks' **own** bugs — a
missing `vx`/`vy`, the `_check_das_output_alignment` integrity breach, an accessible-space
signature drift — and an all-NaN column is indistinguishable downstream from legitimately-absent
DAS, which is not hypothetical: `calibration/_features.py` carried a private `das_ok` flag plus a
full re-implementation of the DAS lookup purely to work around it. `tracking.DasUnscoreableError`
(a `ValueError` subclass, so consumers catching the broad type keep working) is now the **only**
exception the three entry points degrade on, raised for exactly the conditions the catch existed
for; everything else propagates. **This decision belongs in the ADR because the code cites the ADR
for it** — `_das.py` and `features.py` carry `(ADR-043)` at eleven sites, and a reader following
that pointer must find the reasoning here rather than a feature description. Its provenance
counterpart is `das_source` (decision 12's sibling): a closed vocabulary making "DAS could not be
computed" distinguishable from "DAS is genuinely absent", per row and per cause.

**12. `SPEED_SOURCE_UNAVAILABLE` widens a schema categorical domain, which is a public contract
change and not an additive constant.** `TRACKING_CATEGORICAL_DOMAINS["speed_source"]` goes from
`frozenset({"native", "derived"})` to `frozenset({"native", "derived", "unavailable"})`. A consumer
validating frames against that domain accepted exactly two tokens before this release and must now
accept three; anything keying on the domain — a schema check, a `CASE` expression, an enum load —
sees a value it has never seen. The token is not a snapshot backdoor but a **declaration a
third-party frame builder may make**: this source structurally has no per-player temporal history,
so `speed` and the `vx`/`vy` derived from it can never exist. It is deliberately distinct from a
NULL `speed_source` ("not derived yet"), because without the distinction a velocity consumer cannot
separate "this data structurally has no velocity" from "the caller forgot `derive_velocities()`",
and those demand opposite responses — degrade quietly versus fail loud. `_validate_das_inputs`
reads it accordingly: **all** rows marked degrades to NaN with `das_source="unscoreable_frame"`;
unmarked or **partially** marked frames still raise, so the fail-loud branch wins on a mixed frame
set and the marker never excuses a missing `team_in_possession`.

**13. The public-API Examples gate was redesigned twice over, and both halves are decisions
rather than fixes.** `_PUBLIC_MODULE_FILES` was hand-maintained with nothing tying it to reality,
so a newly-added public module was **silently missed rather than caught** — it simply never entered
the parametrization, the same incomplete-by-heuristic class as the AST lint in decision 10, and
this release proved it live. **(a) The surface is now DERIVED**, as the union of modules that
*define* a symbol some package exports via `__all__` (which is how underscore-named modules like
`tracking/_ghost_gk.py` are public in practice) and modules reachable by an underscore-free dotted
path (`spadl/statsbomb.py`, which re-exports nothing and the first rule alone would miss); measured
at **118 modules**, up from 56 hand-listed. **(b) The debt bucket is keyed per SYMBOL, not per
module** (`"<file>::<qualified_name>"`, currently **225 entries**), because a module-level exemption
cost far more than it excused: when the `+SKIP` tightening demoted 12 filler examples, four whole
modules left enforcement and took their **already-documented** symbols with them — a net coverage
*reduction* hiding inside a change meant to tighten the gate, with `tracking/features.py` alone
excusing 5 gaps at the price of 79 documented symbols. Both halves are **self-burning-down**: a
meta-assertion requires every entry to still have an undocumented symbol, so finishing one turns CI
red with an instruction to promote it, and a new module lands in neither bucket and fails. The
per-symbol key also adds a prong a module-level bucket **structurally could not have** — a symbol
renamed or deleted out from under a still-valid file entry. Bundled with this: `+SKIP` filler and
bare imports no longer count as examples, and `@overload` stubs are skipped (an entry that could
never burn down defeats the bucket's core property).

**14. Twelve symbols are added to the public `tracking.__all__`, and publication is the decision.**
Derived by diffing `tracking.__all__` against `git show HEAD:silly_kicks/tracking/__init__.py`
(224 → 236 names, nothing removed). Grouped by why each is public rather than internal:

- **`serve_ghost_gk_positions`** — the positions-only ghost-GK seam with per-row provenance. This
  is the one `gkdv/_engine.py` consumes, and it exists so the engine reaches a *supported* seam
  instead of `_ghost_gk` internals. Publishing it is what makes decision-section's "public seams
  only" claim true.
- **`compute_threat_pc`** — the threat-weighted pitch-control facade, extracted from
  `_cover_shadows` for `delta_threat_suppression`, for the same reason.
- **`defended_goal_x`** — the spec §4.2 pinned goal map, exported through the same
  `_gk_resolve` → `features` → `__init__` chain its three siblings already use. Consumers must
  call it rather than re-derive the goal-side rule; a fork is exactly what §4.2 forbids.
- **`DasUnscoreableError`** — decision 11's named exception. A narrowed catch is only usable if
  callers can name the type they are allowed to catch.
- **`DAS_SOURCE_VALUES`** plus its five members `DAS_SOURCE_COMPUTED`, `DAS_SOURCE_UNLINKED`,
  `DAS_SOURCE_UNSCOREABLE_FRAME`, `DAS_SOURCE_TEAM_UNRESOLVED`, `DAS_SOURCE_UNSCOREABLE_CALL` — the
  closed vocabulary of the new `das_source` column. Published as constants rather than documented
  as string literals so a consumer's `CASE`/enum can be pinned to the library's own set and break
  loudly if it widens.
- **`SPEED_SOURCE_UNAVAILABLE`** — decision 12's token, public because third-party builders are
  expected to set it.
- **`GhostClampWarning`** — a dedicated category for the ADR-016 pitch clamp. The clamp already
  warned; a named category makes it *filterable and attributable* rather than one anonymous
  `UserWarning` among many, so a consumer can escalate exactly this condition to an error.

## CLAUDE.md Amendment

This ADR adds two **new** conventions rather than carving an exception out of an existing one:

> **Every band needs a test from BOTH sides, and every counterfactual needs a non-vacuity assertion
> that it actually moved something.** Four silent-null defects in this codebase share one shape — a
> y-inversion (ADR-031), a fabricated grid origin (ADR-036/PR-S113), an identity-keyed pitch-control
> cache (this ADR, decision 3), and a mirrored external-provider event frame.

> **Private modules can have downstream consumers.** Before renaming, splitting or relocating any
> `silly_kicks/**/_*.py`, check `docs/PRIVATE_CONSUMERS.md`. Path pins fail **silently**.

Three existing entries were amended rather than added to:

- A `gkdv/` bullet was added to the Architecture list at the density of the `xtgk/` entry.
- The **ADR-019 bullet was re-scoped from "tracking-feature seams" to REPO-WIDE** (decision 9 makes
  the seam public and six packages import it), its file count re-measured, and **its
  CI-gates passage rewritten**: it had documented the AST lint as live, with 4.53.0-specific detail
  about `ALLOW_MODULES` and an `rglob` scan, for a test this release DELETES (decision 10). The
  ADR-027 sub-note is retained as the historical case that killed the lint rather than as a
  standing caveat about it.
- The Tracking bullet gained a `PR-S120 (4.53.0)` paragraph for decisions 11–12 and the twelve new
  public exports of decision 14 — `das_source`, `DasUnscoreableError`, `DAS_SOURCE_VALUES` and
  `SPEED_SOURCE_UNAVAILABLE` were otherwise undocumented public surface.

## Related

- **Specs:** `docs/superpowers/specs/2026-07-12-tf19-gkdv-regate-and-v1-design.md`
- **Plans:** `docs/superpowers/plans/2026-07-18-tf19-gkdv-pr3-implementation.md`
- **ADRs:** builds on `ADR-037` (TF-19 re-gate code, and the `gkdv → tracking` direction rule),
  `ADR-040` (chirality-corrected weights + the frozen probe re-run that produced
  `gated_clean_fail`); consumes `ADR-008` (pitch control / `PitchControlCache`), `ADR-013`/`ADR-016`
  (ghost-GK model + boosted-mean serving), `ADR-023`
  (CI slow-test gating), `ADR-033` (`add_*` purity gate — not engaged, no `add_*` shipped);
  **amends** `ADR-019` (the id-dtype contract — its module is promoted to the public
  `silly_kicks.id_compat` and its AST lint is replaced by an enumeration registry, decisions 9–10)
  and thereby closes `ADR-027`'s standing note (the lint was the name heuristic; there is no longer
  a lint to be heuristic); retrofits `ADR-020` (dup-`action_id`) onto `das_xfns`; cross-references
  `ADR-015` (causal harness corroboration), `ADR-011` (trained-model lifecycle — the
  attempt-arm gate)
- **External references:** Le, H. M., Yue, Y., Carr, P. & Lucey, P. (2017), "Data-Driven Ghosting
  Using Deep Imitation Learning", MIT Sloan SSAC (the ghosting counterfactual); Kim et al. (2025),
  "Better Prevent than Tackle", arXiv:2512.10355 (DEFCON-GNN — **comparator only, NOT implemented**);
  Bischofberger & Baca (2026), `accessible-space` (DAS). Full bibliographic entries live in `NOTICE`.

## Notes

**Follow-up registered, deliberately out of PR-3's scope.** The `importorskip` idiom silently
converts *"optional dependency"* into *"optional testing"*, and nothing in the suite reports it.
That is exactly how the whole `accessible_space`-gated DAS suite — 39 tests, 19 of them inside CI's
own selection — sat dark from TF-28 until this cycle's D4 measurement found them. A cheap standing
guard should be added: either a test asserting that every `importorskip`'d
module is installed on **at least one** CI leg, or simply a CI step that **prints the skip count** so
a silent mass-skip becomes visible. The gap should not go back to being invisible now that it has
been found.

**Remaining TF-19 work after this PR:** PR-3b (the xS-arm substitution probe, now unblocked by the
engine) and the owner validation run — the held-out expected-sign test on known sweeper-keepers
versus line-keepers. Spec §6.4 Layers 0–3 (the composed headline metric) remain unshipped.

> **Superseded in part by ADR-055 (2026-08-08).** `defended_goal_x` is DELETED, not renamed in place: the seam is now `resolve_defended_goals(frames) -> GoalMap`, built once per match and threaded in, with `get` / `attacked_goal` accessors over canonical STRING keys. The reasoning recorded here stands; only the symbol moved.
