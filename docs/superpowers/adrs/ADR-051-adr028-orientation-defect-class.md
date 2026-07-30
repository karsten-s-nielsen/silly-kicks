# ADR-051 — ADR-028 orientation defect class: precedence, fail-loud, and the mirror registry

**Status:** Accepted (4.69.0, PR-S137 — PR 1 of 5). RC1 corrected in 4.70.0 (PR-S138 — PR 2 of 5).
RC2 + RC3 corrected in 4.71.0 (PR-S139 — PR 3 of 5; **committed but NOT RELEASED — ships within
4.72.0 alongside PR 4**, since correcting the serving geometry without retraining
`GkCompletionModel` would introduce a train/serve skew that does not exist today).
**Supersedes:** ADR-045 D6 (partially — see D3)
**Amends:** ADR-028 (its repair table names two aggregators in error), ADR-041
**Spec:** `docs/superpowers/specs/2026-07-29-adr028-orientation-defect-class-design.md`

---

## Context

SPADL actions are **action-LTR** (the acting team attacks x=105); `convert_to_frames` output is
**frame-LTR** (the home team attacks x=105 every period). For an away-team action the two are a
180° point reflection apart — `x → 105−x` **and** `y → 68−y`.

Specing TF-30 (b) surfaced a live defect in `add_cover_shadows`, and verifying it surfaced four more
root causes. All were measured on real matches (Gradient Sports 10502, IDSSE DFL-MAT-J03WMX) using a
physical-mirror instrument, chosen over patch-and-diff because it needs no "corrected"
implementation and therefore cannot mistake a bug in the fix for a defect magnitude.

Two measurements frame everything below:

- **`add_cover_shadows` is wrong on 78–85% of away rows**, provider-independent, with
  `n_blocked_receivers` off by a median of one whole receiver.
- **The shared test fixture could not express the defect at all.** `synthesize_actions` stamped
  action coordinates from raw frame positions, so 9/10 actions equalled the raw frame position and
  **0/10** the point reflection. Raw `start_x` was *accidentally correct*, and a correct
  implementation would have been wrong on it.

The second is why this ADR is about detection before correction.

## Decisions

### D1 — Orientation precedence: label → geometry → fail loud. Never identity.

Three sources of attacking direction exist and they are not equally good:

| Source | Mechanism | Fails when |
|---|---|---|
| Identity | `same_id(team, home_team_id)` | frames are not home-attacks-right |
| Label | `acting_team_attacks_rtl` | label absent → silent all-False |
| Geometry | `defended_goal_x` (GK mean x per game/period/team) | needs neither |

**Attacking direction is read from the frames, never inferred from team identity.**
`home_team_id` may answer "which team is home"; it must never answer "which way do they attack".

Geometry is not merely tidiest — it is why `xS`, `xCross` and `ghost_gk` survived a loader bug that
broke everything label-dependent. Those weights are intact today because someone resolved the goal
side from positions rather than trusting a field.

### D2 — The orientation seam fails loud, specified by OUTCOME

`acting_team_attacks_rtl` warns (`OrientationUnresolvedWarning`) whenever it returns an all-False
flip for any reason other than "there were no actions to flip".

**Specified by outcome, not by enumerated condition** — and this is the load-bearing part. The first
specification said "absent or all-null", which missed a join-key branch; the first *implementation*
then missed a fifth branch entirely, the post-merge `.fillna(False)` where the acting team is absent
from the frames or the id spellings differ. Two independent fixture groups walked through that hole
before it was closed. An enumerated rule rots at the next branch; an outcome rule does not.

The signal is **nothing resolved**, not nothing flipped: an all-home action set legitimately yields
an all-False flip, and a *partial* miss is legitimate too (ADR-027 NaN-team rows never resolve).
Period-5 shootouts are exempt — `direction.py`'s `_LTR_KNOWN_PERIODS = (1, 2, 3, 4)` already
excludes them because PSO orientation is undefined.

**Warn, do not raise.** Consumers legitimately hold absolute/unlabelled frames (ADR-029) and a raise
has no reachable remedy inside a converter; fail-closed belongs in CI.

**Correction (4.71.0): CI does NOT currently escalate this category.** `pyproject.toml`
`filterwarnings` escalates `SyntheticEPVWarning` / `IgnoredSurfaceInputsWarning` /
`MissingFeatureContractWarning` only. The PR 1 phrasing "where the opt-out list is zero" stated an
intent as though it were a state; the in-suite arc did reach zero emitters, but nothing enforces it.
This matters now because RC2 makes `resolve_restart_geometry` a genuine emitter on unoriented frames:
escalating the category would turn that into a hard CI failure, so it is a decision to take
deliberately (PR 4/5), not a property to assume.

### D3 — `home_team_id` retires by disuse; it is not removed

**Supersedes ADR-045 D6's "other action-coupled aggregators still take `home_team_id` by design"**,
on measured evidence: identity-keying and direction-keying agree on canonical converter frames *by
construction*, and diverge on the absolute frames the library itself ships.

`home_team_id` stays in signatures — no breaking change mid-cleanup — but stops being **read** for
direction. The re-key targets are byte-identical on converter output (worst 8.53e-14), so this costs
no re-materialize. Once nothing reads it, removal is mechanical, and the Gate B registry is what
proves nothing reads it.

**Honest limit:** geometry resolves the absolute-frames case. It does **not** rescue
`snapshot_to_tracking_frames` output, where each frame is oriented for its own action so a
per-(game, period, team) aggregation mixes orientations. Measured: re-keying `add_defensive_line`
there leaves away rows wrong *and* makes the home row worse (62.0 vs a correct 80.0). That case is
structurally ambiguous under every key, and D2 is its only correct answer.

### D4 — Fixture correctness in the shared helper; coverage as a parameter

`synthesize_actions` emits action-LTR unconditionally. Team balance is an opt-in `balance_teams`
parameter.

A correctness defect in a shared helper belongs in the shared helper; a sampling policy belongs at
the call site. The 9:1 team skew was an artifact of `drop_duplicates(...).head(n)` picking an
arbitrary first-listed player — today's low exposure was sort order, not design, and a parallel
"correct" helper would have made the incorrect one the default.

### D5 — Two gates, because one instrument cannot see both defect classes

**Gate A** — physical mirror; detects convention mixing.
**Gate B** — vary `home_team_id` over `{home, away, nonsense}` on fixed frames; detects
identity-keyed direction.

Gate A is **structurally blind** to identity-keying: swapping `home_team_id` restores the very
invariant identity-keying assumes, so an identity-keyed aggregator is invariant under it whether it
is safe or not. A 5.68e-14 Gate A reading was originally cited as evidence that
`_cover_shadows.py:1030` was safe; that claim is **withdrawn** — it was the expected reading either
way.

Gate B needs no transformed frames, so it never runs an aggregator outside the `convert_to_frames`
contract, and the nonsense id makes it strictly stronger than a two-team swap: it catches
`same_id(x, home) else …` branches a swap leaves looking correct.

Registry-driven with meta-assertions pinning it to `tracking.__all__` in both directions, so a new
aggregator cannot join silently. Per-column mirror classes (`invariant` /
`mirrored_pitch_absolute` / `exempt`) and per-entry tolerances each carry a recorded basis — a
tolerance nobody can revisit on evidence is a number, not a decision.

## Consequences

**PR 1 (4.69.0):** no shipped feature value changes, no retrain, C4 unchanged (32). One new public
warning category.

**PR 2 (4.70.0) — RC1 corrected:** the cover-shadow passer is reprojected into frame coords at both
seams. **Re-materialize trigger, no forced VAEP retrain** (`cover_shadow_xfns` is in no default xfn
list). The two affected columns were measured separately and do NOT share a rate: on away rows
`n_blocked_receivers` changed on 77.8% (GS 10502) / 85.0% (IDSSE DFL-MAT-J03WMX), cheap-path
`max_single_defender_blocking_score` on 90.7% / 100% — one-match point estimates per §2.2, not
corpus rates. The three passer-independent columns and every home row are byte-identical. Gate A's
RC1 marker is deleted; Gate B's stays, because `_cover_shadows.py:1030` is still identity-keyed —
that is D3, a different defect class, not a partial fix of RC1.

**PR 3 (4.71.0) — RC2 + RC3 corrected.** `_gk_geometry`'s two unreprojected samplers and the
space-creation OBSO multiplier. **Re-materialize trigger** for `xt_gk*`, `gk_completion`, `enriched_*`
restart coords and both space-creation columns (away rows only). Three strict xfails deleted
(13 → 10). Two execution findings worth recording:

- **RC2's symptom was a lost tier, not a wrong number.** The goal-area clamp is an action-LTR own-half
  predicate; against a raw away-team frame x it rejects a correctly-placed keeper, so those goal kicks
  silently fell through to the rule-point prior rather than emitting a visibly bad coordinate. Order
  matters: the clamp must follow the reflection.
- **RC3 is fixed at the GRID seam, not the multiplier.** The plan specified reflecting
  `obso_multiplier`; that product also contains a ball-anchored `distance_weight` computed in frame
  coords, which must never be mirrored — the constraint the opponent-perspective branch already
  documented. Reflecting `transition_grid`/`epv_grid` instead is correct and additionally fixes the
  opponent multiplier for free, because it is constructed as a flip of those same artifacts. This is
  the ADR-041 lesson recurring one layer down: the seam you reflect at is itself a correctness choice.

**RC5 — a distinct member of the class, found during PR 3 and FIXED there (4.71.0).**
`_gk_geometry._next_event_start` borrowed the *next action's* `start_x`/`start_y` guarded only on
`game_id`/`period_id` — never on `team_id`. SPADL is per-**acting-team** LTR, so when the next action
belongs to the other team the borrowed coordinate is a 180° point reflection away. Measured: a
shared physical point the opponent records as `(45.0, 20.0)` is `(60.0, 48.0)` in the anchor's own
frame — a 15 m x-error and a 28 m y-error. It feeds `enriched_end_*` and `xt_gk_dest_x/y`, hence RAV.

**Why it belongs in PR 3 rather than PR 4, on ordering grounds:** PR 3 is already release-coupled to
PR 4's `GkCompletionModel` retrain, and a retrain must run against FINAL geometry. Landing this in
PR 4 would risk retraining against geometry that changes within the same PR. Every geometry
correction lands in PR 3; PR 4 then retrains once, against a settled surface.

**It is action-vs-action, not frame-vs-action, so every gate in this ADR is structurally blind to
it** — the mirror registry reflects *frames*, and this defect lives entirely in the action stream.
Dedicated tests are therefore the only guard, and they assert the *wrong* value is not produced as
well as that the right one is. The claim that it was "untestable today" (all fixtures use
`team_id=[1, 1]`) was true of the existing fixtures only; a two-team fixture is four lines.

An **unattested team id never decides**: `ids_differ` is NA-safe-both-present, so an NA on either
side leaves the borrowed coordinate untouched rather than reflecting it — the ADR-027 / `retains()`
rule that "cannot tell" must not silently become "reflect".

**Vindicated by the gates themselves:** Gate B found an **eighth** D3 member the audit missed
(`add_gk_influence`, whose `_gk_influence.py:371-372` applies the correct both-axes reflection keyed
on the wrong thing), and Gate A found a **chiral goal-relative transform** (`_geometry.py` has no
`to_goal_relative_y`) deferred to PR 5 as a retrain trigger.

**Strict xfails encode every known defect — 14 registered at PR 1, 10 remaining after PR 3.**
Strictness is the point: a fix cannot land without deleting its own marker, which is why this count
is stated as a running one rather than an absolute.

## Alternatives rejected

- **Remove `home_team_id` from the geometry layer now.** Stronger — it makes the bug
  unrepresentable rather than tested-against — but it is a breaking change across many public
  signatures, and doing an API migration and a geometry correction in one diff means debugging both
  at once. D3 is the direction of travel; the gate is what makes the eventual removal safe.
- **Raise instead of warn (D2).** No reachable remedy for a consumer holding absolute frames.
- **Land the gate after the fixes.** It would arrive green and never be observed failing — the
  green-by-construction trap TF-30 (a) existed to clean up.
- **A hand-listed gate.** That is what already existed: 4 of 33 aggregators, which is why five root
  causes went unnoticed.
