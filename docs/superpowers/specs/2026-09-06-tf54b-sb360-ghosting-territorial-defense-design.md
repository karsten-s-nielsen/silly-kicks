# TF-54b (re-scoped) — SB360 counterfactual for territorial defense

**Status:** DRAFT (brainstorm-approved design; pending owner independent review)
**Amendments (2026-09-09, additive — no scope cut):** from eyestone/xT-GK cross-project input — (A) §8/§9
add the team-confound honest-limit + elite-prior collinearity caveat (validated as an INSTRUMENT, NOT
player-attributable; per-defender numbers are not a ranking); (B) §4 relocates the actor bridge to the
shared `keeper_identity.py`. Tracked in the plan's revision log (rev-3).
**Supersedes:** the event-only `method="counterfactual"` shipped in Commit 1 of the TF-54b branch
(`ab9001c`), which validated as weak (`promote: false` — synthetic mechanism ties the naive death
baseline 8.58 vs 8.51; discriminant corr 0.88 with v1 / 0.81 with volume). This design replaces that
method with a tracking-grounded counterfactual on SB360 freeze-frames that extracts the spatial value
those frames provide.
**Vehicle:** amend Commit 1 on the existing branch (owner-directed earlier — but the scale has grown to
a new package + actor bridge + validation battery, so §11.5 asks the owner to *reconfirm* amend-vs-
fresh-commit before the plan). **Keep** the strong reusable `expected_passing.PassCompletionModel`
(AUC 0.81) + the `xthreat.destination_profiles` seam; the event-only cone counterfactual
(`territory/_counterfactual.py`, its columns/dispatch, and `scripts/_synthetic_interception.py`) is
**recommended removed** pending §11.3. Re-run trainer/validator, then present Commit 2.

**Empirical facts established for this spec (real WC2022 SB360, match 3857276):**
- Every freeze-frame carries a uniquely-flagged **actor** (`actor: True`, exactly 1/frame across 2,873
  frames). `shape_snapshots` currently reads `teammate`/`keeper` but **not** `actor`.
- players-per-frame 3–21 — freeze-frames are **heavily FOV-cropped**; the counterfactual is honest-NaN
  where the relevant players are off-frame (ADR-077).
- WC2022 men's (comp 43 / season 106) has SB360 (2,873 frames/match); + the licensed 30-match corpus.

---

## 1. Motivation

v1 (`method="completed_failed"`, 4.108.0) and the event-only counterfactual (this branch) both had to
*guess* a failed pass's intended target — unobservable event-only, so the cone+xT-grid estimator
inherits the death direction and cannot beat "it died here" (measured 8.58 vs 8.51). The battery
correctly refused promotion.

**SB360 freeze-frames remove the guess** — they give the actual attacker/defender positions at each
event. That lets us compute a real spatial counterfactual: how much did *this* defender's positioning
suppress the attacking team's threat, versus that defender not being there. Goal (owner-directed): a
**gold-standard method that extracts the optimal value from SB360**, scope no object, refactor Commit 1
as needed.

## 2. The identity constraint (decides the method's shape)

SB360 rows carry a resolved real `team_id` (~10–11 candidates/team) and an `is_goalkeeper` flag but
**no player identity** (`shape_snapshots` numbers rows `np.arange` at `providers/statsbomb/parse.py:327`;
`snapshot_to_tracking_frames` carries that synthetic id through). The keeper bridge
works only because the `keeper` flag uniquely marks one row/team; there is no outfield equivalent,
cross-frame continuity is structurally infeasible (fresh id/frame, one frame/event), and
`derive_goalkeepers` positional aggregation cannot run.

**The only reliable outfield identity is the actor** — established empirically above: each freeze-frame
has exactly one `actor: True` player, and the acting player's real `player_id` is on the SPADL action.
So the actor's row can inherit the action's real id — the outfield analogue of the keeper bridge,
scoped to the one row that is actually resolvable.

**Consequence:** a defender D is *not* reliably identifiable among the anonymous defender rows of an
*opponent-pass* freeze-frame — ghosting "D's row there" would be a confident wrong guess. D **is**
reliably identified in the freeze-frames of D's *own* defensive actions (D is the actor). That is why
the headline arm is action-anchored.

## 3. Method — two arms, both in attacker-value units (positive = threat suppressed)

Both arms attribute per-`(defender, match)`, group on the canonical player id and emit the raw id
(ADR-019), drop-and-count honestly — never a fabricated 0 (ADR-042) — honest-NaN on unresolvable
geometry / velocity-constitutive quantities (ADR-063) / FOV-cropped frames (ADR-077).
The counterfactual value at a frame is `threat_suppressed = threat_pc(counterfactual) − threat_pc(actual)`
via `compute_threat_pc` (positive = the defender's actual position reduced the attacking team's threat).

### Arm A — action-anchored (IDENTITY-EXACT, the headline)

Domain: freeze-frames of D's own defensive actions (tackle / interception / clearance / block), where
D is the actor (identity known via the re-plumbed actor flag).

1. Resolve D's real freeze-frame row via the actor bridge (§4).
2. Compute the attacking team's threat via `compute_threat_pc` at the actual frame.
3. Form the **counterfactual frame** by removing/relocating D's row (§5), recompute threat.
4. `a_threat_suppressed = threat(counterfactual) − threat(actual)`; sum per `(D, match)`.

**Attribution is exact** — every valued frame is genuinely D's. The elite-defender prior is a clean
face-validity test. Reading: "threat D's positioning neutralized at the moments D actively defended."

**Domain yield (measured, real WC2022 open data — resolves SPEC-01).** The Arm-A domain is D's
`tackle`/`interception`/`clearance` actions (SPADL `type_id` 9/10/18; SB360 gives every event a
freeze-frame). Pooled across the 64-match corpus: **7,051 defensive-action frames** — 35× the
`MIN_DOMAIN_FRAMES=200` Layer-0 instrument-validity floor, so the pooled probe is robust on Arm A
alone. Per elite defender: Gvardiol 77, Otamendi 59, van Dijk 43 (Rüdiger 12 — 3 group games;
Marquinhos name-unresolved). **Fallback policy:** the effective domain is the *scoreable* subset
(FOV-observed, §5/§8), which is smaller; the plan measures the scoreable yield explicitly. A
per-defender count below the ranking `min_volume` returns `not_computed` for that defender (never a
fabricated headline), and **Arm B is the coverage companion** for exactly the defenders Arm A is thin
on (Arm B's per-elite volume was 189–305 on the same corpus). So the headline never ships
unvalidated: the pooled instrument verdict stands on 7,051 frames, and the per-defender elite prior is
reported only where the scoreable count clears `min_volume`.

### Arm B — hull-based (BROADER, attribution-APPROXIMATE, slippage MEASURED)

Domain: opponent passes/attacks whose target reflects into D's trimmed territory hull (v1 territory,
ADR-028 point-reflection for membership).

1. Identify the *contesting* defender by POSITION (defending-team row nearest the target / inside the
   hull) — WITHOUT naming it.
2. Form the counterfactual by removing/relocating that position; recompute the attacking threat.
3. `b_threat_suppressed`, attributed to hull-owner D.

Broader coverage, approximate attribution. **The slippage is measured, not asserted** — a validation
leg quantifies how often the contesting defender actually is D (via the actor anchor, and on full-
tracking providers where identity exists). B always ships with that figure attached.

### Why both
A is rigorous but narrow; B is broad but approximate. Shipping both, each labeled with its attribution
guarantee, avoids silently trading coverage for rigor.

## 4. New building block — the actor identity bridge (independently useful)

1. **Re-plumb the `actor` flag** in `shape_snapshots` — add `"is_actor": bool(row.get("actor"))` to the
   per-player `snap_rows` dict (which today reads only `teammate`/`keeper`). Carry it through
   `snapshot_to_tracking_frames` into a new nullable `is_actor` frame column.
2. **Actor identity bridge** — the outfield analogue of `apply_keeper_identities_to_frames`, scoped to
   the actor row: stamp the action's real `player_id` (from the SPADL action, joined on `action_id` /
   `frame_id`) onto the single `is_actor` row. PURE; ADR-019 id-safe; the non-actor rows keep their
   synthetic ids. This is the only reliable outfield identity SB360 provides and is independently useful
   to any consumer needing the acting player's real position. **Home (added 2026-09-09):** it must live
   in a SHARED module — `silly_kicks/keeper_identity.py`, next to `apply_keeper_identities_to_frames`
   (ADR-084 home) — NOT inside the new `territorial_defense/` package, because the package's own
   import-allowlist forbids any other module importing it, which would make "independently useful" false
   and block the concrete second consumer (the eyestone GK build-up-decision metric). `keeper_identity`
   stays tracking-free (the bridge needs only `id_compat` + pandas/numpy). Placement is owner-ruled.

## 5. Counterfactual mechanism & the ghost-source finding

**Finding (API survey):** the `GhostOutfieldModel` / `serve_ghost_outfield_positions` (ADR-087) is a
**domain mismatch** for this use — it is a possession-conditioned rearguard-*line* model whose public
serve only exposes the *in-possession* slice, with leakage-safe features built for a rearguard line
(opponent-counter-threat geometry), not for an individual defender contesting a ball-in-flight pass.
Using it as-is would be the "confident wrong model" anti-pattern.

**Recommended mechanism — removal / marginal-contribution (no ghost model):** form the counterfactual
by **removing D's row** and recomputing `compute_threat_pc`. Pitch control re-partitions D's controlled
area to the remaining players (both teams), so the threat increase = D's *marginal* spatial
contribution given teammates — a well-established gold-standard player-value method (Fernández-Bornn
pitch-control marginal value). It needs no model, no domain assumption, and no `GhostOutfieldModel`.

**Local-completeness gate (resolves SPEC-02).** Removal re-partitions D's controlled area to the
*remaining* players — but on an FOV-cropped freeze-frame those are the *visible* players only, so an
unobserved teammate/opponent near D would have absorbed some of that area, and the delta is biased
**upward** (D looks more valuable than they are). The principals-present check (D + the attacking
players) is necessary but **not sufficient**. So the frame is scored only when the **local
re-absorbing neighbourhood is observed**: the `visible_area` polygon (ADR-077) must cover at least
`params.min_local_observed_fraction` of a disk of radius `params.local_radius_m` around D's position
(the region pitch control actually re-partitions). A frame failing that gate is **dropped-and-counted**
with `td_source="fov_cropped_local"`, distinct from the whole-frame `fov_cropped` — never scored with
a biased delta. The observed fraction is reported per scored frame so a consumer can see the coverage
regime. (The gate reuses `region_observed_fraction` from `tracking/_visibility.py`, ADR-055/077.)

**Refinement — replacement (above-replacement skill), DEFERRED behind validation:** replacing D with a
*league-average defender position* (rather than removing) measures skill above a replacement-level
defender. That is the fuller "ghosting" ideal but requires a **fit-for-purpose defensive ghost** —
either exposing/validating the ghost-outfield out-of-possession regime (no public serve today) or
training a defense-specific positional ghost (a TF-60-PR5-scale sub-project). Recommendation: ship the
removal counterfactual first (robust, model-free, gold-standard-adjacent); pursue replacement as a
validated follow-on only if the owner wants above-replacement semantics. (Open decision §11.)

**The `PitchControlCache` trap (ADR-043) — a hard correctness constraint, not a perf note.** The cache
keys on `(game_id, period_id, frame_id, team, method, params, ball_position, decompose)` — it EXCLUDES
player positions. A counterfactual frame carries its factual twin's identity, so a shared cache would
serve it the factual surface and every delta would collapse to exactly 0 with no warning. **The arms
must never accept or share a `PitchControlCache` across the factual/counterfactual legs** (verbatim the
gkdv `_arms.py` constraint).

## 6. Threat model
`compute_threat_pc(frame, *, attacking_team_id, xt, goal_map, method="spearman", params=None,
field_weight=None) -> float` (exported from `tracking`). Takes `attacking_team_id`/`xt`/`goal_map`
explicitly, refuses an unfitted `xt` (raises, never silent 0.0), self-degrades velocity via
`zero_velocity_if_unavailable`. Zero-velocity positional model on freeze-frames (ADR-063 Tier-1 lift —
dimensionless model-relative threat is honest at zero velocity; velocity-constitutive m²/s quantities
stay NaN). `goal_map` from `resolve_defended_goals(frames)` (ADR-055, built once).

## 7. Architecture / home
- **New tracking-consuming package** (working name `territorial_defense/`; final name a spec decision),
  mirroring `gkdv/` / `restdefense/`: imports `tracking` public seams only; **nothing imports it** (AST
  import-allowlist gate mirroring `tests/restdefense/test_import_allowlist.py` — the package →
  public-tracking-only test, the `tracking`-never-imports-it reverse test, a planted-violation
  meta-test, and a non-empty-package test. **Note:** restdefense's reverse test is *tracking-scoped*;
  the `nothing-imports-it` sweep over all of `silly_kicks/` must be **authored fresh** (a whole-tree AST
  walk asserting no module imports this package), not copied from the exemplar — the exemplar does not
  contain it. Ships
  `compute_territorial_defense(actions, frames, *, xt, ghost_model=None, links=None, visible_area=None,
  params=_DEFAULT_PARAMS) -> (samples, TerritorialDefenseReport)` with Arms A + B.
- `TerritorialDefenseReport` is a frozen dataclass with the restdefense conservation shape
  (`n_frames_in`, `n_frames_scored`, `drop_reasons`), CI-gated `n_frames_scored + Σ drop_reasons ==
  n_frames_in`.
- **`territory/` stays event-only** — `method="completed_failed"` remains the pure, byte-identical
  default; the event-only `method="counterfactual"` is **recommended removed** (superseded + validated
  weak) pending the owner's §11.3 ruling; docs cross-reference the new package.
- **Kept from Commit 1:** `expected_passing.PassCompletionModel` (reusable; may feed Arm B's receiver/
  lane inference) and `xthreat.destination_profiles`.

## 8. Honest limits (all reported)
- **NOT player-attributable — the team-confound (added 2026-09-09, eyestone/xT-GK cross-project input).**
  The §9 battery validates `a_threat_suppressed` as an INSTRUMENT (responds to D's position, is specific,
  tracks priors); it does NOT establish how much between-defender variance is the DEFENDER vs the
  DEFENSIVE SYSTEM. The removal counterfactual's "marginal contribution GIVEN TEAMMATES" (§5) is
  **team-conditioned by construction**, and this corpus **cannot identify the confound**: WC2022 is
  one-player-one-national-team → **zero cross-team defender observations**, so defender-vs-team is
  structurally unidentifiable and the elite-defender prior is elite-defender/elite-team **collinear**.
  ("Arm A attribution is EXACT" means the FRAME is genuinely D's action — NOT that the VALUE is
  D-net-of-team.) **Per-defender numbers are therefore not a defender ranking.** Ranking, if ever
  pursued, is a future ADR-009 gated on a crossed defender+team variance decomposition (ICC) over a
  MULTI-CLUB TRANSFER corpus, not this one. Precedent: the sister GK-distribution metric passed its own
  face-validity checks then proved ~80% team-confounded ("ranking not licensed").
- Velocity-less → Tier-1 positional pitch control (ADR-063); velocity-constitutive quantities NaN.
- FOV-cropped (players-per-frame 3–21) → honest-NaN not only where D / the attackers are off-frame but
  also where the **local re-absorbing neighbourhood** around D is under-observed (§5 SPEC-02 gate:
  removal biases the delta upward when unobserved players would have absorbed D's area) — the observed
  fraction is reported per scored frame.
- Arm A domain = D's own defensive interventions (narrower than "all zone passes") — the price of exact
  attribution; Arm B is the broader approximate companion.
- Arm B attribution slippage measured and shipped with the number.
- Removal measures *marginal* (present-vs-absent) contribution; above-replacement is the deferred
  refinement.

## 9. Construct validation (owner-run, reported-not-gated; SB360 finally makes it real)
Adapt the gkdv/TF-19 probe battery (`gkdv/_probe.py`) — now on real positions:
- **Dose-response**: impose a positional dose on D, confirm a monotone threat response
  (`impose_*_dose` idiom → `layer0_instrument_verdict` / `layer1_responsiveness_verdict`;
  `MIN_DOMAIN_FRAMES=200`, `SATURATING_MULTIPLE=5.0`, a NEW probe-ratio registration — not reusing
  `PHYSICS_ARM_PROBE_RATIO`/`TF19_PROBE_RATIO`/`XS_PROBE_RATIO`, all model-specific).
- **Paired single-player placebo controls** (`paired_vector_controls`): displace one other defending
  outfielder by D's vector; D's effect must exceed the placebo.
- **Elite-defender prior** (locked, pre-registered — the ADR-089 "Van Dijk" idiom): a FACE-VALIDITY
  check on Arm A, NOT attribution evidence. **Caveat (added 2026-09-09):** in WC2022 an elite defender
  and a strong defensive national side are perfectly collinear, so a clean prior does NOT license
  attribution or ranking (§8 team-confound). Also measure whether the 30 licensed SB360 matches add ANY
  cross-team defender replication (WC2022 alone adds none); if not, record that the corpus has zero
  identifying power for the confound.
- **Arm B slippage leg**: measured contesting-defender-is-NOT-D rate (attribution error; lower = tighter; honest-NaN when identity is un-measurable).
- Default unchanged; promotion is a separate ADR-009 decision after the owner reads the report.
  **Player-attribution / ranking is explicitly out of scope** and, if pursued, is gated on a crossed
  defender+team ICC over a multi-club transfer corpus (the eyestone side has number-gate-verified code
  for the analogous crossed keeper+team ICC).

**Named CI regression tests (methods, not applied results) — the plan enumerates these:**
- **`test_shared_cache_collapses_delta_to_zero` (resolves SPEC-03):** the ADR-043 landmine —
  constructing the arms with a shared `PitchControlCache` (or otherwise serving the counterfactual leg
  the factual surface) must be caught by a test that asserts the delta collapses to exactly 0 *only*
  under the mis-wiring and is non-zero on the correct path (mirrors
  `tests/gkdv/test_arms.py::test_unpinned_implementation_would_measurably_differ`). Two-sided: the
  correct path measurably differs; the mis-wired path is a spurious zero.
- **Local-completeness gate (SPEC-02):** a cropped-neighbourhood fixture is dropped (`fov_cropped_local`)
  while a fully-observed twin is scored, and the scored delta differs from the cropped-then-naively-
  scored value (proving the gate changes the number, not just the count).
- Standard method gates (liveness non-NaN+non-constant, ADR-033 purity, ADR-019 id-dtype, ADR-051 D3
  orientation, ADR-053 SB360 boundary verdict) per the tracking/restdefense suites.

## 10. Corpus & drivers
Licensed 30-match SB360 + open WC2022, via `load_statsbomb_matches(...) -> (provider, match_id, actions,
frames, home_team_id, visible_area)` (6-tuple; velocity-less by construction — no `_preprocess`). The
per-match chain is `flatten_events → convert_to_actions → parse_freeze_frames → shape_snapshots →
snapshot_to_tracking_frames`. Drivers use `scripts/_driver.for_each` (ADR-052 shards) +
`require_clean_tree` (ADR-037). If the replacement refinement (§5) is pursued, its ghost model follows
the ADR-011/040/050 bundle discipline; the removal primary needs no model.

## 11. Open decisions for owner / independent review
1. **Counterfactual mechanism** — removal/marginal (recommended primary, model-free) vs replacement with
   a fit-for-purpose defensive ghost (above-replacement; a sub-project). Ship removal first?
2. **Package name / boundary** (`territorial_defense/` vs folding into an existing package).
3. **Fate of the event-only counterfactual** — remove (recommended) vs keep as a no-tracking fallback.
4. **Arm B contesting-defender rule** — nearest-to-target vs all-defenders-in-hull vs receiver-lane
   (the last would consume the kept `PassCompletionModel` / `ReceiverModel`).
5. **Amend-Commit-1 vs fresh commit** — given the scale (new package, actor bridge, validation battery),
   confirm amend is still the intended vehicle.

## 12. Interfaces (Consumes / Produces) — exact signatures

**Consumes (public `tracking`):**
```python
compute_threat_pc(frame, *, attacking_team_id, xt, goal_map,
                  method="spearman", params=None, field_weight=None) -> float
resolve_defended_goals(frames) -> GoalMap                      # ADR-055, built once
# NB: zero_velocity_if_unavailable(frames, *, method=...) is applied INTERNALLY by compute_threat_pc
#     (ADR-063 edge) — not a separate consume; the arms just pass velocity-less frames through.
region_observed_fraction(polygon, region) -> float             # tracking/_visibility.py (SPEC-02 gate)
# PitchControlSurface.control_in_region(x_min,x_max,y_min,y_max)->float  (region option for Arm B)
```
**Consumes (gkdv probe battery, adapt not import if layering forbids):**
```python
layer0_instrument_verdict(*, realistic_abs, saturating_abs, placebo_p95, n_domain) -> str
layer1_responsiveness_verdict(*, gk_med, nd_med, placebo_p95, n_domain) -> str
paired_vector_controls(frames, targets, *, r, rng) -> dict[str, pd.DataFrame]
```
**Consumes (SB360 port + load):**
```python
shape_snapshots(frames_raw, actions, *, fidelity_version=1) -> (snapshots, visible_area, JoinReport)
#   ^ MODIFY: add "is_actor": bool(row.get("actor")) to the per-player snap_rows dict
snapshot_to_tracking_frames(snapshots, actions) -> (frames, links)   # carry is_actor through
load_statsbomb_matches(*, match_ids=None, token=None, max_matches=None, cache_dir=None)
#   -> Iterator[(provider, match_id, actions, frames, home_team_id, visible_area)]
```
**Produces (new package):**
```python
apply_actor_identities_to_frames(frames, actions) -> pd.DataFrame     # stamp real id onto is_actor row
compute_territorial_defense(actions, frames, *, xt, ghost_model=None,
                            links=None, visible_area=None, params=_DEFAULT_PARAMS)
    -> (samples: pd.DataFrame, report: TerritorialDefenseReport)
# samples columns (per (game_id, player_id)): a_threat_suppressed, a_frames_scored,
#   b_threat_suppressed, b_frames_scored, b_attribution_slippage, td_source (provenance) — final set a
#   spec/plan detail; every metric column glossary-documented (ADR-048) + NOTICE (ADR-005).
```

## 13. Attribution / academic references (NOTICE)
Le et al. 2017 (data-driven ghosting); Fernández & Bornn (pitch control, marginal player value);
Spearman (pitch control); Sumpter/Twelve module 10.2 (territorial dominance / the "Van Dijk" metric).
