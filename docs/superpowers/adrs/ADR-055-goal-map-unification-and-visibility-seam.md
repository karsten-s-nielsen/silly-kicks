# ADR-055: Goal-map unification and the observed-region seam

| Field | Value |
|---|---|
| **Date** | 2026-08-08 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

The rule "which goal does this team defend" was implemented **ten times** across five modules,
each spelling a variant of `0.0 if same_id(team, home_team_id) else 105.0`. Three problems
followed from that, and only the third had ever been written down.

**It is identity-keyed, not direction-keyed.** `home_team_id` says who the home team is; it says
nothing about which way anyone attacks. The equivalence holds only while frames are in the
`convert_to_frames` home-attacks-right convention, which is a property of the caller's pipeline
rather than of the function's inputs. This is ADR-051's D3 class, and `add_gk_influence` /
`add_cover_shadows` were its last two live members among the aggregators (measured movement when
`home_team_id` was varied on FIXED frames: `gk_pitch_control_share_weighted` 0.108532,
`gk_closing_time_min_s__six_yard_box` 4.38062 s, `blocking_score` 148.83).

**It has no period term.** A team swaps ends at half time. Every one of the ten sites answered the
same for period 1 and period 2.

**It fails OPEN.** Each site had a total function: some end was always returned. On frames where
the end is genuinely unresolvable — no keeper, an NA team identity, all-NaN coordinates — the
sites returned a confident 105.0 (`nan < 52.5` is False), and the resulting number was
indistinguishable downstream from a measured one.

Separately, StatsBomb 360 ships a `visible_area` polygon per event, and every count feature in
the library treats "nobody there" and "nobody VISIBLE there" as the same observation.

## Decision

One seam, `resolve_defended_goals(frames) -> GoalMap`, built **once per match from the full
frames** and **threaded** into per-frame functions where it *replaces* `home_team_id`; consumers
call `get` (own end) or `attacked_goal` (opponent's end), never a plain dict. An eleventh
implementation fails CI. Plus `tracking/_visibility.py`, shipping `point_observed`,
`region_observed_fraction` and `add_visible_area_coverage` so a consumer can distinguish an
absence from a non-observation.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Keep `home_team_id`, fix each site in place | No API change; smallest diff | Ten copies remain, so an eleventh is still free; the period term has to be added ten times | Fork-by-duplication is the defect, not its symptom |
| B. Build the map per frame inside each function | No threading; no signature change | A different estimator, and unresolvable for a third of team-frames on sparse providers (see *The per-frame cost* below) | A default that re-admits it at every call site that forgets to pass one |
| C. Deprecate `defended_goal_x`, keep it as an alias | No import breaks | A consumer keeps a name whose value changed | A hard rename fails loud at import; an alias fails silently |
| D. **Chosen**: one seam, threaded, `home_team_id` removed | One implementation; period-scoped; refuses rather than guessing | 15 breaking public signature changes across two packages | — |

Two sub-decisions were taken by the owner during implementation, both after the alternatives had
been measured rather than argued:

* **`select_back_line_players` takes `defends_x0: bool`,** not `goal_x: float`. Both remove the
  identity key; only the bool avoids a cascade. `_packing.py` would have had to express a goal end
  as `0.0 if same_id(...) else 105.0` to supply a float, which is the very fork the population gate
  rejects, so `compute_packing_metrics` would have needed a `GoalMap` too — three more breaking
  changes and ~26 files, in a family this cycle's own scope boundary leaves to ADR-051.
* **The four public GK scalar helpers drop `home_team_id`** rather than keeping it unread. The
  repo has both precedents (`space_creation` retains an unread one, "D3 retired by disuse"), and
  the cycle's rule — a dead required parameter is the shape being deleted — was applied
  consistently rather than half.

## Consequences

### Positive

* The goal-end rule exists in exactly ONE place, pinned by a semantic AST gate
  (`tests/tracking/test_goal_map_population.py`) that matches `IfExp`, `if`/`else`-assigning-one-name
  and `np.where` — landed RED and observed failing on all 10 forks across 5 modules.
* Every site gains a period term it structurally lacked.
* Unresolvable ends REFUSE (`GoalEndUnresolvedError`, a `ValueError` subclass) instead of
  returning 105.0. Policy lives at the edge: per-frame functions raise, the `add_*` aggregator
  catches by name and emits a NaN row. Pre-checking at the aggregator was rejected because it
  duplicates the exact lookup the callee is about to do — a second implementation of the decision,
  inside the commit that deletes second implementations.
* `attacked_goal` is a real lookup of the opponent's entry, never `105.0 - get(...)`: the
  arithmetic identity is wrong on a degenerate map, where it would say a team attacks the goal it
  defends.

### Negative / risks

* **15 breaking public signature changes**, across `tracking` and `gkdv`, plus two renames
  (`defended_goal_x` deleted; `visible_fraction` → `observed_pitch_fraction`). All fail loud at
  import or call. Precisely: 14 of the 15 are in an `__all__`; the fifteenth,
  `compute_gk_influence`, is public by documentation rather than by export — CLAUDE.md's TF-15
  row lists it as the family's public surface — so it is counted, and the distinction is
  recorded rather than rounded away.
* **VAEP/tracking retrain trigger.** `add_gk_influence` and `add_cover_shadows` change value
  wherever identity and direction disagreed, and `add_cover_shadows` is now **keeper-dependent on
  freeze-frame input** — on SB360's `gk_absent` roster its five columns collapse from `all_nan` to
  `no_signal`, because the outfield fallback guesses both teams at the same end (measured means
  56.9 and 76.5, both past the 52.5 midline) and `attacked_goal` refuses a degenerate map. That is
  a correction: previously both legs produced numbers the frames could not support.

## The per-frame cost, measured — and a spec figure that does not reproduce

The spec's §15 headline was "78.8% wrong". **It does not reproduce on the four committed slim
fixtures**, and it is cited nowhere in this repo as a result. Measured here instead, comparing a
map built per FRAME against the per-MATCH map, over every `(game, period, team)` the per-match map
resolves:

| provider | team-frames | per-frame map DISAGREES | `attacked_goal` unresolvable |
|---|---|---|---|
| skillcorner | 504 | 36 (**7.1%**) | 180 (**35.7%**) |
| metrica | 1,112 | 24 (2.2%) | 686 (61.7%) |
| gradientsports | 3,600 | 0 (0.0%) | 0 (0.0%) |
| sportec | 1,104 | 0 (0.0%) | 0 (0.0%) |

Two things follow, and both are worth stating plainly rather than rounding into the spec's number.

**The cost is PROVIDER-DEPENDENT, not universal.** On dense tracking (sportec, gradientsports:
every keeper in every frame) a per-frame map is exactly right, and building once per match costs
nothing. The damage is concentrated in sparse broadcast detection — SkillCorner detects a keeper in
~19.6% of frames (ADR-038) — which is precisely the provider class this seam exists to serve.

**The spec's second number DOES reproduce.** Its 34.2% unresolvable rate lands on SkillCorner's
measured 35.7%; only the 78.8% does not, against a measured maximum of 7.1%. The design decision
rests on the reproduced half: a third of team-frames losing direction entirely is decisive on its
own, and the old sites answered those frames with a confident 105.0.

Recorded because shipping the unreproduced figure in the ADR, `CLAUDE.md` and the CHANGELOG would
have put a number nobody could re-derive at the centre of the cycle's rationale — the shape this
repo already names as a plausible result from a computation that did not happen.

## Gate C, and why Gate B was retired for two aggregators

The sharpest review finding of the cycle: re-keying off `home_team_id` makes **Gate B vacuous**,
so the cycle would have deleted the only detector for the class it fixes. Gate B varies
`home_team_id` on fixed frames; once the parameter is gone it skips on `role="unused"`, and the
AST gate cannot see a direction *bool*.

**Gate C** (`test_gate_c_goal_map_is_the_direction_source`) is the same question one variable
further out: hold the frames fixed, swap the MAP, require the declared columns to MOVE. It
reproduces the recorded D3 magnitudes exactly — `share 0.108532`, `closing_min 4.38062`,
`closing_mean 4.02205`, `blocking_score 148.83` — which is the evidence that it detects the class
Gate B used to.

Two honest qualifications, both recorded at the gate:

* **Gate C proves the map is CONSULTED, not that the right accessor was chosen.** `get` and
  `attacked_goal` both move when the map is swapped. The correctness half is
  `tests/tracking/test_goal_map_consumers.py`.
* **Gate C is structurally blind to `_closing_time_per_series`.** The plan asserted that a
  one-column result would mean that path was missed. Executing the defect — patching it back onto
  a self-built map and running Gate C on `add_gk_influence` — leaves Gate C **GREEN**, because
  `add_gk_influence` never calls it; its closing-time columns come from `_gk_influence_at_actions`.
  That path is covered instead by `test_closing_time_helpers_read_the_injected_goal_map`, verified
  red against the same executed defect.

## The pre-registered measurements: M2 and M3

The spec fixed three measurement rules BEFORE measuring, so an answer could not be chosen after
seeing it. M1 (NA-team GK rows in the `GhostGkModel` training corpus) needs the DGX training
corpus and is a recorded follow-up. M2 and M3 are measurable on the committed slim fixtures and
were run:

| provider | M2 — GK-less `(game, period, team)` groups | M3 — end DIFFERS between p1 and p2 |
|---|---|---|
| sportec | 0 of 4 | 0 of 2 |
| gradientsports | 0 of 4 | 0 of 2 |
| skillcorner | **2 of 4** | **1 of 2** |
| metrica | **3 of 4** | **1 of 2** |

**M2 is non-zero**, so the `allow_guess=True` decisions threaded through this cycle are
LOAD-BEARING rather than vacuous: on metrica three of four `(game, period, team)` groups carry no
keeper rows at all, and without the outfield rung those groups would resolve to nothing.

**M3 is non-zero, and it is the finding that matters.** The spec's pre-registered reading:
*non-zero → the three `features.py` sites and `_gk_influence.py:318` are producing wrong values
today for every second-half action on the affected providers.* Measured:
`('skillcorner_1899585', 'away_team')` defends x=0 in period 1 and x=105 in period 2;
`('Sample_Game_3', 'home_team')` the same in reverse.

**It survives `play_left_to_right`.** Re-measured on oriented frames, both flips persist
unchanged — so this is not "the caller forgot to orient", which ADR-029 already covers. The
period term is therefore not a theoretical nicety: on two of four committed providers the
identity-keyed sites, which had no period term at all, answered second-half actions with the
first half's end.

That converts §2.7's "no value effect" into a value change on those providers, which is why the
retrain trigger in this ADR is stated unconditionally rather than hedged. It also means the
sportec/gradientsports byte-identity is a property of THOSE fixtures, not a general claim.

## The D3 unit pin moved

`test_defensive_line_d3_unit_is_enumerated` pinned `{_defensive_line, _packing, _gk_influence}` as
the files reading `home_team_id`. `_gk_influence` left the set, so the pin is now the other two,
with the reason recorded and the departure asserted separately — a member silently dropping out is
exactly how a partial re-key would hide.

## The `_snapshot` dtype pin was DROPPED, and the spec claim corrected

The design (§2.6) recorded that `_snapshot.py`'s `pd.concat` yields `Int64` on pandas 2.3.3 and
`Float64` on 3.0.3, and prescribed casting to `TRACKING_FRAMES_COLUMNS`. Measured on the pinned
resolver (pandas 2.3.3):

* the concat yields **`float64`**, not `Int64`;
* the prescribed cast is **unimplementable** — the schema declares `int64` for `player_id` and
  `team_id`, the ball row is NA in both, and `int64` cannot hold NA (`IntCastingNaNError`), so it
  raises on every snapshot; the declaration is also not what the library's own adapters emit
  (`object` ids for the DFL providers), so forcing it would reject string ids;
* a `restore_id_dtype`-based pin changes **nothing** for numpy-int, nullable-`Int64` or object id
  sources, and with the pin excised **0 of 2** tests written for it went red.

Shipping a change nobody in CI can observe, justified by a measurement that does not reproduce, is
the "plausible result from a computation that did not happen" shape this repo already names. The
residual concern — a frame builder whose id dtype moves with pandas 3 — is real and is a recorded
follow-up, not a silent no-op in this commit.

## The observed-region seam

`point_observed` returns `bool | None`, because `False` is a CLAIM the camera did not see a point
and a missing polygon supports no claim. `region_observed_fraction` takes an `(M, 2)` POLYGON, not
a bounding box: a bbox can only OVER-report coverage for a triangle, i.e. fabricate observation.
`visible_area_fraction` is NaN for every non-`observed` token — never 1.0, never 0.0.

Sutherland–Hodgman requires a convex CLIP, so the **polygon is always the SUBJECT** and the
**region is the clip**: that puts the convexity requirement on the argument the caller constructs
(a triangle, a zone, a pitch-control cell) rather than on broadcast data nobody controls, and a
concave region is REFUSED rather than silently mis-clipped. The primitives live in a neutral
`silly_kicks/_polygon.py` — the `id_compat` / `reflection` position — because `providers/` has no
runtime dependency on `tracking/` and adding one would invert the port layering.

Wiring coverage INTO the count features is deliberately NOT in this cycle: it changes existing
values and decides for the consumer what a partial observation means (ADR-009).

## References

* Spec: `docs/superpowers/specs/2026-08-07-goal-map-unification-and-visibility-seam-design.md`
* Plan: `docs/superpowers/plans/2026-08-07-goal-map-unification-and-visibility-seam.md`
* ADR-051 (the D3 orientation defect class), ADR-054 (SB360 degradation + the StatsBomb port),
  ADR-009 (raw primitives ship; composites stay consumer-side), ADR-042 (a coverage denominator
  must never masquerade as a signal), ADR-019 (dtype-safe id comparisons).
