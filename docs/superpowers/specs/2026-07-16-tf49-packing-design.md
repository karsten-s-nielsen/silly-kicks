# TF-49 — Packing: Impect-faithful surface over the existing LBS kernel (design)

**Status:** owner-approved design (2026-07-16), pending cross-session spec review.
**Origin:** Soccermatics Pro + Modern Soccer Coach course ingestion (see
`2026-07-16-soccermatics-pro-future-work-plan.md` §W5 and the TF-49 TODO row).
**Sequencing (owner-decided 2026-07-16; fix shape revised per cross-session review F1):** the
GS dribble-end converter fix (§2.3) ships FIRST as its own small PR — **PR-S116**, branch
`pr-s116-gs-dribble-ends`. **The fix is GS-LOCAL, not a global set change:**
`_derive_end_coordinates(actions, *, extra_type_ids=frozenset())` with only
`gradientsports.py` passing `{dribble}` — `_DERIVE_END_TYPE_IDS` itself is untouched because
it is shared by all EIGHT converter paths and statsbomb's ~11% genuine stationary carries
(end==start recorded data) are indistinguishable from placeholders under the guard; a global
addition would rewrite them (Hyrum). Red-first test + GS e2e verification; **GS-only retrain
trigger** — xT/xtgk move-sets include dribbles, VAEP consumes dribble ends; GS golden
re-baselines; zero delta for the other seven providers (byte-identity asserted in tests);
CHANGELOG + ADR-018-amendment-level documentation, no new ADR. Packing then builds on
corrected GS geometry.
**Flags (this spec = the packing PR):** **PR-S117**, branch `pr-s117-packing` off main after
PR-S116 merges; next-free version at its release (4.49.0 is expected to go to PR-S116) +
ADR-039; C4 aggregator count 28 → 29; in NO default xfn list → no retrain trigger; no part-deux
file collisions (new module + `_kernels.py` addition + `features.py` registration only).

## 1. What ships (owner-approved v1 scope)

Canon packing core + goal-threat sub-metric + `secured_reception` + `net_packing`.
**Deferred:** `eligibility="tti"` (MSC's "recovering defender is not packed" rule — needs TTI
wiring + its own real-data validation) — tracked in the plan doc, not in this PR.

New public surface, all in `silly_kicks/tracking/`:

- `_packing.py`: `PackingParams` (frozen dataclass), `compute_packing_metrics` (pure per-frame),
  private helpers.
- `_kernels.py`: `_packing_at_actions` batch loop (sibling of `_structural_pass_at_actions`).
- `features.py`: `@nan_safe_enrichment add_packing(actions, frames, *, home_team_id,
  params=None, links=None)` + `packing_xfns(*, home_team_id, params=None)` factory.
- Atomic mirror in `atomic.tracking.features` (numeric columns only, `end = x+dx` synthesis —
  the structural-pass precedent).

**`_structural_pass.py` is NOT touched.** The ~15 lines of defender-extraction/mirroring are
deliberately duplicated in `_packing.py` (frozen-kernel isolation; Chesterton + Hyrum), with
equivalence pinned by the golden identity gate (§7).

## 2. Verified facts this design rests on (2026-07-16 probes; read-only Databricks
`soccer_analytics.bronze.spadl_actions` + repo source)

1. **Next-touch receiver rule is structurally sound — anchors re-probed under the SHIPPING
   rule (non_action rows skipped; round-2 finding 4):** completed pass-like actions are
   followed by a same-team touch 90.8% (wyscout) / 91.6% (idsse) / 94.8% (metrica) / **95.1%
   (skillcorner)** / 99.6% (statsbomb) / 99.8% (gradientsports); `self_next_rate` ≈ 0 (no
   self-reception ambiguity); period-end truncation ≤ 0.08%. Note: the non_action skip moved
   ONLY skillcorner, and DOWNWARD (97.9 → 95.1% — same-team non_action rows were masking
   following opponent touches; the reviewer's raise-only prediction was directionally wrong,
   which is exactly why the gate anchors to shipping-rule numbers). The owner-gated e2e gates
   against THESE values. In-library resolution additionally requires a non-null next
   `player_id` (bronze's all-null `player_id` for tracking providers is a storage artifact —
   §9).
2. **`player_id` population is provider-split in the lakehouse:** 0% for
   gradientsports/idsse/metrica/skillcorner in `bronze.spadl_actions` (identity in
   `player_id_native`, 99–100%), 100% for statsbomb/wyscout. In-library, converters emit
   `player_id` directly (GS nullable `Int64`, kloppy-family strings) — so the library rule works,
   but the receiver column MUST preserve source dtype (ADR-019) and the lakehouse handoff (§9)
   must name the coalesce.
3. **Gradient Sports dribbles are 100% zero-displacement (850/850 rows, `start == end`) — and
   this is a CONVERTER GAP, not a provider trait** (root-caused 2026-07-16, owner prompt):
   GS maps `OTB`+`BC` ball-carries → SPADL `dribble` (`gradientsports.py:200→218`), initializes
   `end = start` for every event (`:532–535`), derives real ends only for
   `_DERIVE_END_TYPE_IDS` — which excludes `dribble` (`base.py:15–27`) — and is the only event
   converter that never calls `_add_dribbles`. Fix = **GS-local** `extra_type_ids={dribble}`
   passed to `_derive_end_coordinates` by `gradientsports.py` only (review F1: the module-level
   set is shared by all eight converters and the `placeholder_end` guard cannot distinguish
   statsbomb's ~11% genuine stationary carries from placeholders — a global change would
   rewrite recorded data). **GS-only retrain trigger** (xT/xtgk move-sets include dribbles;
   VAEP features consume dribble ends) → **owner-decided 2026-07-16: ships as its own preceding
   fix PR (PR-S116); this packing PR is PR-S117** and builds on corrected GS geometry.
   IDSSE/metrica/skillcorner/wyscout dribbles have real geometry (median 6.2–13.7 m);
   statsbomb 89% distinct endpoints (11% ≈ genuine stationary carries). ⇒ the
   degenerate-geometry policy (§5.6) remains as the residual guard either way (period-last
   carries; unfixed corpora).
4. **45°/135° direction bands are sane and provider-stable:** completed passes split ≈ 24–32%
   forward / 52–58% side / 15–21% back across all six providers. The side-heavy split is a
   property of angle bands (documented, parameterized).
5. Kernel seams (repo): `compute_structural_pass_metrics` is schema-agnostic (explicit
   passer/receiver xy; defenders = opponent outfield, `~is_goalkeeper`; away-mirror via
   `same_id`); `_structural_pass_at_actions` gates on `type_id ∈ (0, 1)` with **no result
   check**; `select_back_line_players(frames, team_id, home_team_id, *, n=4)` selects the N
   outfield players nearest their own goal.

## 3. Definitions

All geometry in the acting team's attack-positive frame (SPADL action coords are already
attack-positive; frame defenders mirrored `(105−x, 68−y)` iff the acting team is AWAY — the
`_structural_pass` invariant, duplicated).

**Domain** (per action): `type_name ∈ params.action_types` AND `result_id == 1` (success).
Off-domain rows → all five columns NaN. Default `action_types` = ("pass", "cross", "throw_in",
"freekick_crossed", "freekick_short", "corner_crossed", "corner_short", "goalkick", "dribble")
— Impect counts every pass; the MSC Ederson worked example requires goalkick; names resolved to
ids once via `spadl.config`.

**Eligible defenders:** opponent players at the linked frame, finite coords, `~is_ball`;
outfield-only by default (`include_gk=False` — matches the kernel, Goes et al. 2019's
"defenders", and MSC's last-4); `include_gk=True` adds the keeper.

- **`packing_made`** — count of eligible defenders with `start_x < d_x ≤ end_x` (forward-only
  by construction; identical inequality to `structural_lbs`). This is the canon Impect count:
  1 point per opponent bypassed by a completed action. Receiver double-credit is realized by
  consumers grouping this value by `packing_receiver_player_id` (no extra column needed).
- **`packing_net`** — direction-multiplied signed count: θ = `atan2(|Δy|, Δx)` (degrees,
  attack-positive); multiplier m = **+1** if θ ≤ `forward_max_deg` (45), **`side_multiplier`**
  (+0.5) if 45 < θ ≤ `back_min_deg` (135), **`back_multiplier`** (−1) if θ > 135 — the
  `football-packing` library's multipliers. Count = eligible defenders with
  `min(sx,ex) < d_x ≤ max(sx,ex)`. Note the deliberate divergence: a forward-diagonal pass
  (θ = 60°) contributes 0.5× to `packing_net` while `packing_made` counts it fully (canon vs
  net view).
- **`packing_goal_threat`** — same forward inequality as `packing_made`, defender set restricted
  to `select_back_line_players(frame, opp_team, home_team_id, n=params.back_line_n)` (default
  4 — MSC "goal threat packing"; outfield by that helper's construction). NaN when the back-line
  selection is empty.
- **`packing_receiver_player_id`** — resolved by a named pure seam
  `resolve_next_touch_receiver(actions)` (public, `spadl/utils.py` — event-only, frames-free,
  reusable by TF-35; house precedent `add_pre_shot_gk_context` lives spadl-side): the next
  **touch** strictly after this action (ordered by `action_id` within `(game_id, period_id)`,
  **skipping `non_action` AND `foul` rows** — neither is a touch (GS emits non-touch
  `non_action` rows; an intervening foul must neither become the receiver — the fouler never
  touched the ball — nor, when advantage is played on an opponent foul, block resolution of the
  genuine reception; execution-review D1, §9.5) with the same `team_id`
  (ADR-019 `same_id` semantics) → its `player_id`. **Dtype rule (review F5):** nullable
  (`Int64`) and object id columns shift natively; a plain `int64` source is pre-converted to
  `Int64` (lossless, NA-safe) before the shift — never forced to object, never allowed to
  float64-upcast (the "366.0" stringification class ADR-019 already paid for). NaN when: next
  touch is by the opponent, period end, or source id is NaN. **The seam is packing-agnostic
  (round-2 finding 5): it resolves a receiver for EVERY action type; the dribble → NaN mask is
  applied in `add_packing`'s assembly** (dribbles have no reception in packing semantics), so
  TF-35 and other consumers get generic next-touch resolution.
- **`packing_secured`** (nullable boolean) — the bounce-pass/front-foot operationalization
  ("ball stays past the line"), shared design with TF-35's reception-quality features. Defined
  only where a receiver resolved AND `packing_made ≥ 1` (else NaN — nothing was bypassed, or
  nothing received). Computed by a named pure seam
  `secured_reception(actions, line_x, receiver_pos=None, *, params)` (tracking-side —
  `receiver_pos=None` computes positions internally so the public export is legally callable;
  `add_packing` passes its precomputed positions; `line_x`
  comes from the pass frame; `receiver_pos` = the positions output of the private
  `spadl.utils._resolve_next_touch_positions`, so the window is genuinely
  RECEPTION-anchored — plan-review blocker 4: t_r is the receiving row's time, located by
  position, and the reception row itself is never a scannable window event; NaN-TEAM rows
  (GS null-actor, ADR-027) are skipped via NA-routed comparisons, never raw `!=` —
  plan-review blocker 5), **built on the `retains()` skeleton
  (round-2 review 2(a)): possession-aware, self-healing `possession_id` via the public
  `spadl.utils.add_possessions` when absent** (verified: `_retention_labels.py:17, 24–25` does
  exactly this) — keeping the label semantically consistent with the shipped ρ-label family.
  Let `line_x` = max bypassed-defender x at the pass frame and `t_r` = the receiving action's
  `time_seconds`. Scan actions after `t_r` within `(game_id, period_id)`, **skipping
  `non_action` AND `foul` rows**; the foul-skip is REQUIRED ON TOP of the skeleton — verified
  2026-07-16: `add_possessions`' rule-4 carve-out suppresses the boundary only at the ensuing
  RESTART row; **the foul row itself still emits a possession boundary** (A-pass P0 → B-foul
  P1 → A-freekick P1), so a bare possession-boundary rule would flip loss at fouls. First
  decisive event wins:
  - same-team `shot` / `shot_penalty` / `shot_freekick` → **True** (round-2 finding 1 — the
    saved-shot case: pass → shot → `keeper_save` must not read as a loss; the set-piece shot
    types compose with the foul-skip: foul skipped, ensuing penalty/free-kick shot decides).
    **Execution-time clarification (PR-S117):** in the literal 3-row shape the shot IS the
    reception (the next same-team touch), and the scan starts after the reception — so a
    reception that is itself a same-team shot decides **True immediately**; otherwise the
    subsequent `keeper_save` possession boundary would read the first-time shot as a loss.
    The reception row's `start_x` is still never tested (blocker 4);
  - opponent action with a possession boundary (`team ≠` AND `possession_id ≠`, the
    `retains()` rule) → **False**;
  - same-team action starting at attack-positive `start_x < line_x` **within the window**
    (`t ≤ t_r + secured_window_seconds`, default 3.0 s) → **False** (the bounce-pass case;
    window-scoped — secured means IMMEDIATE security, so late retreats don't count);
  - **no scannable event inside the window** (round-2 finding 3): the possession-boundary and
    shot tests extend to the FIRST subsequent non-skipped event (out-of-play → opponent
    restart row = possession boundary → False; a late foul is skipped and the next
    possession-implying event decides); the `line_x` test does NOT extend;
  - no decisive event, window fully observed → **True**; window truncated (the `retains()`
    form: `(t_last − t_r) < secured_window_seconds`) with no decisive event → **NaN**.
- **`require_secured=False`** (default) — canon counts gate on completion only; `secured` is
  provenance. When True, the secured gate applies **only to receiver-bearing types** (review
  F3): dribbles keep their raw counts — secured reception is a pass concept (MSC), and gating
  dribbles on a structurally-NaN secured would silently erase all carry packing. For
  receiver-bearing rows with `packing_made ≥ 1`: secured False → 0.0, secured NaN → NaN counts;
  rows with `packing_made == 0` keep their 0 (nothing was bypassed, nothing to un-secure).
  Note on frames: same-team follow-up actions share the acting team's LTR frame, so their
  `start_x` compares directly against `line_x`; opponent actions within the window are used only
  as a boolean turnover trigger, never their coordinates (they live in the mirrored frame).

## 4. `PackingParams` (frozen dataclass, house style; no `is_default()`)

```python
action_types: tuple[str, ...] = (...)   # §3 default
include_gk: bool = False
back_line_n: int = 4
forward_max_deg: float = 45.0
back_min_deg: float = 135.0
side_multiplier: float = 0.5
back_multiplier: float = -1.0
secured_window_seconds: float = 3.0
require_secured: bool = False
```

`__post_init__` validation: `0 < forward_max_deg < back_min_deg < 180`,
`secured_window_seconds > 0`, `back_line_n ≥ 1`, `side_multiplier ≥ 0`, `back_multiplier ≤ 0`
(a negative side or positive back multiplier is nonsense — review minor), non-empty
`action_types` ⊆ known SPADL type names (fail-loud on typos).

## 5. Semantics: NaN / edge policy (ADR-003 discipline)

1. Off-domain (type or result) → all columns NaN.
2. No linked frame / frame missing at `(period_id, frame_id)` → geometry columns NaN
   (receiver/secured still computable — they are event-only; secured needs `line_x` from the
   frame, so secured → NaN too; receiver stays).
3. Zero eligible defenders at the frame → geometry columns NaN (kernel precedent).
4. NaN actor `team_id` (GS null-actor rows, ADR-027) → all NaN.
5. Non-finite action coords → all NaN (never a fabricated count — the PR-S113 lesson).
6. **Degenerate geometry (`start == end`) → NaN for DRIBBLES ONLY** (review R9): a dribble end
   can be a placeholder (pre-PR-S116 GS corpora; post-fix period-last carries), so "didn't
   move" and "not recorded" are indistinguishable → unattested. For pass-class types the end is
   recorded data (native or PR-S116-derived) and `start == end` legitimately yields the honest
   geometric **0** (the bypass interval is empty) — a blanket NaN would convert real zero-count
   actions to missing, provider-quantization-dependently. Both branches unit-tested.
7. Receiver id NaN in source (GS null-actor next action) → receiver NaN, secured NaN.

## 6. Data flow / implementation shape

- **Geometry (frame-coupled):** `_packing_at_actions` mirrors `_structural_pass_at_actions`:
  `resolve_frame_ids_by_position` (dup-action-id safe), per-row `frame_groups.get_group`,
  per-frame `compute_packing_metrics(frame, attacking_team_id=…, home_team_id=…, passer_xy,
  receiver_xy, params)` returning the three numeric metrics + `line_x` (internal, for secured).
  **Goal-threat mirroring composition (review minor):** `select_back_line_players` operates in
  the FRAME convention (home-attacks-right, own-goal proximity); the selected subset's coords
  are then mirrored into action-LTR for away actors exactly like the main defender set —
  select-then-mirror, stated here so the implementer doesn't rediscover the ordering (the
  mirror-invariance gate backstops it).
- **Receiver + secured — named pure seams (review R7):** `resolve_next_touch_receiver(actions)`
  (public, `spadl/utils.py`, Examples-gated) and
  `secured_reception(actions, line_x, receiver_pos=None, *, params)` (`tracking/_packing.py`;
  the reception-anchored window is carried by the positions helper's output)
  are first-class pure functions unit-tested DIRECTLY, not only through
  `add_packing`. Both event-only over `actions` (`groupby(game_id, period_id)`; receiver via
  non_action-skipping shift; secured via the windowed forward scan of §3), consuming `line_x`
  from the geometry pass.
- **Duplication trigger (review R6, recorded in ADR-039):** the ~15-line defender-extraction/
  mirror block is duplicated from `_structural_pass.py` (×2). A THIRD consumer (TF-35 is the
  named candidate) triggers extraction of a shared
  `_eligible_defenders_action_ltr(frame, attacking_team_id, home_team_id, *, include_gk)` with
  a byte-identity gate on structural_pass.
- `add_packing` assembles the five columns, merges provenance idempotently (house
  `if not any(c in out.columns …)` pattern), accepts pre-linked `links`, returns a NEW frame
  (ADR-033 purity).
- `packing_xfns` emits the **three numeric columns only** per gamestate slot
  (receiver id + secured are provenance, excluded — the `_COORD_COLS` precedent); 3×-not-9×
  call-count budget with a structural perf spy (house pattern). **`packing_xfns` REJECTS
  `require_secured=True` (`ValueError`; plan-time finding):** receiver/secured resolution
  needs true next-row relationships, which shifted gamestate slots do not have — secured
  gating is aggregator-only. **Leakage warning (review F4,
  recorded in ADR-039 + the factory docstring):** every packing column gates on the action's
  OWN `result_id` — as an a0-slot feature that is exactly the result-leakage class HybridVAEP
  exists to strip, and the TF-48 auto-discovering guard covers DEFAULT lists only (opt-in
  factories bypass it). `packing_xfns` MUST NOT enter HybridVAEP-class consumers without a0
  exclusion. A result-free a0 variant (completion gate dropped for the a0 slot) is a RECORDED
  FORK in ADR-039, not built (YAGNI until a consumer asks).
- **Signature order** matches `add_structural_pass` exactly (keyword-only; house consistency —
  review minor). **Docstring caveats** (review minors): `take_on` (7) is excluded — a point
  event cannot express bypass under this inequality, so `dribble` does NOT cover 1v1s; the
  completion gate inherits each provider's result semantics (SkillCorner `result_source` tiers,
  sportec eval-allowlist) — cross-provider `packing_made` sums are not provider-comparable
  without that caveat.
- Atomic mirror: numeric columns only; `end = x + dx, y + dy` synthesis; receiver/secured
  omitted (atomic `receival` atoms already carry receiver identity explicitly).

## 7. Testing (red-first TDD; every gate wired)

**Unit (per rule, hand-built fixtures with known counts):** completion gate; each action type
in/out of domain; `include_gk` flag; back-line gate (defender sets differing between full team
and back-4 — pre-verify the multi-domain liveness fixture actually yields differing counts,
else fixture work first); net bands + multipliers (θ at 44°/46°/134°/136° boundary probes + one
x-tie probe for the `min < d_x ≤ max` boundary asymmetry on backward passes — review minor);
receiver (same-team / opponent-next / period-end / dribble / NaN-id / **non_action-skip** /
**dtype trio: int64, Int64, string sources — output dtype asserted, no float64 upcast**);
secured (**fouled-receiver → NOT a loss** — the F2 keystone fixture; **pass → shot →
keeper_save → True** — the round-2 finding-1 keystone; **foul → penalty → shot_penalty → True**
— the skip/shot composition; opponent possession-boundary turnover; behind-line return; empty
window with opponent-restart → False vs **late-foul-then-same-team → True** (round-2 finding
3); truncation → NaN; `packing_made == 0` → NaN; possession self-heal exercised both ways —
`possession_id` present AND absent);
degenerate geometry (**dribble → NaN AND pass-class start==end → 0**, both branches);
non-finite coords NaN; `require_secured` (receiver-bearing gating AND **dribble counts
untouched** — the F3 fixture).

**Golden identity (discriminating):** `packing_made` with
`action_types=("pass","cross")`, defaults otherwise, on completed rows `==` `structural_lbs`
restricted to those rows — AND mutating the completion gate out must turn it red (proves the
gate is the only delta on that slice).

**Mirror invariance:** ADR-028-style both-conventions gate (same physical situation under a
frame mirror → identical outputs) + one ground-truth ASYMMETRIC fixture pinning absolute counts
(symmetry alone is insufficient — house discipline).

**Auto-gate registration:** `PURITY_ENTRIES` (2 variants: defaults + non-default params),
liveness fixture entry (the multi-domain fixture's pass window must yield non-NaN, non-constant
values for the three numeric columns; receiver/secured join the non-float exemption rules as
applicable), id-dtype invariance (auto-enumerated), dup-action-id (xfns, auto), NaN-safety
(decorator, auto), public-API Examples (docstrings — incl. the new public
`resolve_next_touch_receiver`), CI slow-gating rules respected (golden / numeric gates on ALL
legs). **Committed-fixture smoke (review minor): receiver + secured event-only logic runs on the
in-repo statsbomb WC2018 fixture on ALL CI legs** — the owner-gated GS e2e is not the only
guard on the event-only seams. Shared-registry files (`features.py`, `_kernels.py`, atomic
features, `PURITY_ENTRIES`, liveness registry, C4 DSL) are also touched by any parallel
aggregator-shipping session — **serialize the C4 28→29 bump at merge time**.

**Owner-gated e2e (GS WC2022 local; `@e2e`):** hard gates on internal consistency — receiver
resolution rate within ±2 pp of the probe values; GS dribble degenerate-NaN rate consistent
with post-PR-S116 geometry (residual placeholders only — period-last carries — NOT the pre-fix
100%; the e2e doubles as the packing-side verification that PR-S116 landed); per-action
`packing_made` mean in a sane band; secured rate ∈ (0, 1) strict. The MSC practitioner anchors
(≈2 bypassed per packing action; ~8 goal-threat pts/shot; 67.4% of goals involve a packing
action) are **REPORTED, not gated** (league-specific).

## 8. Attribution

NOTICE entries: Impect / Reinartz & Hegeler (packing, defenders-outplayed weighting); Goes,
Kempe, Meerhoff & Lemmink (2019), doi:10.1089/big.2018.0067 (longitudinal outplayed-defender
formalization); `football-packing` (S. K. Varadharajan) for the net-packing multipliers.
Docstring credits: Modern Soccer Coach "Packing Data" lesson + Twelve/Soccermatics course
(practitioner rules: goal-threat last-4, secured reception, anchors). **ADR-039 records:** the
canon-vs-variant decisions (incl. the no-subtraction-rule finding and the deferred TTI
eligibility); the secured design = `retains()` skeleton + foul-skip, with the CORRECTED
rationale (round-2 finding 2: `add_possessions` exists and self-heals — the earlier
"no possession column" premise was false; the foul-skip is load-bearing because heuristic
possessions emit a boundary AT the foul row, verified) + the accepted coupling to the
`add_possessions` heuristic for ρ-family consistency; the `packing_xfns` result-leakage
warning + the recorded result-free-a0 fork (review F4); the receiver-seam placement decision
(`spadl/utils.py`, public, packing-agnostic — dribble mask in assembly) + non_action-skip rule
(review R7 + round-2 finding 5); the duplication consolidation trigger (review R6); the
degenerate-geometry scoping rationale (review R9). **Relay-back observation (recorded, out of
scope):** `retains()` itself, when running on self-healed/heuristic possession ids (the ρ
loader feeds `possession_id_heuristic`), plausibly flips loss at opponent-foul rows — the same
bias class F2 fixed here. Fixing it would be a ρ-label change → ρ retrain; flagged to the
owner + part-deux, not addressed in this PR.

## 9. Lakehouse handoff notes (relay at release)

- Receiver identity: materializing `packing_receiver_player_id` from `bronze.spadl_actions`
  requires `COALESCE(player_id, player_id_native)` — the numeric `player_id` is 0%-populated
  for all four tracking providers (verified) — or better, resolve the gold `player_key`
  (the 4.45.0 keeper-identity convention, generalized).
- GS dribbles are structurally NaN (zero-displacement in the provider feed) — expect NaN, do
  not coalesce to 0.
- `packing_xfns` is opt-in; adopting it into a VAEP feature set is a self-triggered retrain.

## 9.5 Execution-review fixes (2026-07-17, PR-S117 adversarial review — all six confirmed by live reproduction)

- **D1 receiver foul-skip** (§3 amended above): the positions helper skips `foul` alongside
  `non_action`; keeps the seam consistent with secured's `_SKIP_TYPES`.
- **D2 atomic collapsed-atom bridging**: `convert_to_atomic`'s `_simplify` re-types
  `corner_*` → atomic `corner` and `freekick_*`/`shot_freekick` → atomic `freekick`, so the std
  set-piece names carry ZERO atomic rows — the atomic adapter now maps the collapsed atoms into
  the domain when their std names are requested (shot-shaped freekicks stay out via the result
  synthesis: no reception follows a shot).
- **D3 atomic output purity**: atomic `add_packing` assembles its output on a copy of the
  CALLER's frame — the adapter's rewritten `type_id` / synthetic `result_id` never leak.
- **D4 secured scan order**: `secured_reception` scans in `action_id` (canonical play) order —
  the same order the positions helper resolves anchors in — so time-tied, positionally-swapped
  rows resolve identically.
- **D5 atomic keeper receptions**: the result synthesis also accepts a SAME-TEAM
  `keeper_pick_up`/`keeper_claim` next-atom (atomic never inserts `receival` before keeper
  collections; a completed back-pass otherwise synthesized fail).
- **D6 NA possession never decides**: the secured boundary test requires both possession ids
  attested (a caller-supplied `possession_id` with NA rows must not attest a false loss —
  the ADR-027 class, extended to possession). NOTE: `retains()` shared the same patterns
  latently; a 2026-07-17 probe over the LIVE ρ cohorts measured **zero training-label flips**
  (gold-mart possession ids stay continuous through foul rows, no NAs), so the SAME hardening
  was applied to `retains()` in-PR with NO retrain — post-fix gate: shipped == probed variant
  on all 223,718 cohort rows, 0 training-label changes. `retains()` also gained the canonical
  `(time_seconds, action_id)` scan order (the D4-analog, owner-decided after a dedicated order
  probe; bare-`action_id` sort ruled out — GS mart action_id order disagrees with time;
  gate-verified byte-identical on both cohorts). ADR-036 amendment (2026-07-17).

## 10. Explicitly out of scope (v1)

`eligibility="tti"` (recovering-defender rule); per-player season aggregations / packing-score
composite / combination matrices (consumer-side per ADR-009 frozen-exogenous precedent);
SkillCorner dynamic-events cross-validation (part-deux-owned files); any change to
`_structural_pass.py`.
