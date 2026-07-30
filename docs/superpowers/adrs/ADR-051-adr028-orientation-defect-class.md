# ADR-051 — ADR-028 orientation defect class: precedence, fail-loud, and the mirror registry

**Status:** Accepted (4.69.0, PR-S137 — PR 1 of 5)
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
has no reachable remedy inside a converter; fail-closed belongs in CI, where the opt-out list is
**zero**.

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

**This PR:** no shipped feature value changes, no retrain, C4 unchanged (32). One new public warning
category.

**Vindicated by the gates themselves:** Gate B found an **eighth** D3 member the audit missed
(`add_gk_influence`, whose `_gk_influence.py:371-372` applies the correct both-axes reflection keyed
on the wrong thing), and Gate A found a **chiral goal-relative transform** (`_geometry.py` has no
`to_goal_relative_y`) deferred to PR 5 as a retrain trigger.

**14 strict xfails** encode every known defect. Strictness is the point: a fix cannot land without
deleting its own marker.

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
