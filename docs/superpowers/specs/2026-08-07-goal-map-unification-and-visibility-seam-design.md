# Goal-map unification and the visible-area seam — design

**Date:** 2026-08-07
**Revision:** 10 (reviews 1-7 + three executability passes + IMPLEMENTATION findings §17)
**Status:** Approved; core claims MEASURED (§14, §15), not yet implemented
**Successor to:** ADR-053 (SB360 audit), ADR-054 (SB360 repair + StatsBomb port)
**Shape:** ONE commit on one branch, **squash-merged**.

> **Squash unless something cites a branch SHA.** Squash is the default. The recent merge-commit
> exceptions (4.72.0, 4.75.0, 4.76.0) existed solely because a committed research artifact cited a
> branch commit a squash would orphan. This branch runs no artifact driver, so squash applies.
> If §6's measurements force an artifact re-run, that run must happen on an *already-merged*
> commit — a driver cannot stamp a SHA that does not yet exist — so it becomes a follow-up, never
> a second commit here.

---

## Executive summary

`tracking.defended_goal_x` is the pinned public goal map, and its own docstring forbids
re-deriving the rule: *"a second implementation is a fork that can disagree with the first."*
Measured by a **shape-agnostic** AST detector, the rule is derived at **12 sites across 7
modules**.

This cycle replaces every one of them with a single `GoalMap` value object that owns both
construction **and lookup**, gates the result so a thirteenth fails CI, and folds in two SB360 residue
items: the pandas-version-dependent snapshot id dtype, and the `visible_area` consuming seam that
4.76.0 deliberately left for a consumer.

**The discipline each revision added, and the trap each fell into.**

| rev | method | failure |
|---|---|---|
| 1 | enumerated by reading | measured the case where nothing changes, asserted the cases where something does |
| 2 | enumerated by reading | counted the wrong population (2 forks / 4 modules) |
| 3 | executed a detector | the detector was **built from the population already known** — every known site was a ternary, so an `ast.If` statement (`_gk_influence.py:318`) was structurally invisible |
| 4 | executed a **shape-agnostic** detector, and cross-checked against a **behavioural** census that already existed | — |

Rev 3's thesis ("derive the population by executing the detector") was necessary and not
sufficient: a detector built from the sample confirms the sample. §4.2's predicate is now
semantic — *any construct binding one name to `{0.0, pitch-length}`* — and the non-vacuity plant
is written in the spelling that was missing (`if`/`else` statement), so the witness proves the
predicate generalizes past its sample.

**The decisive correction is that a census already existed and is behavioural.** Gate B
(`tests/tracking/_mirror_entries/`) records **8 aggregators** carrying `defect_b="D3 re-key
pending"` — and it found `_gk_influence.py:318` months ago, by running the code rather than
parsing it (§1.8). The AST gate is a fast tripwire; **Gate B is the census.**

**Scope, settled across revs 4-6 (owner): ONE cycle, everything.** Counting, stated once so the
three denominators in this document stop drifting:

| count | what it counts | value |
|---|---|---|
| **census** | goal-end binding EXPRESSIONS the §4.2 detector returns | **12** (1 seam + 1 exempt + 10 to replace) |
| **re-keyed** | sites this cycle edits: the 10, plus 2 direction bools the detector does not see, plus `_gk_influence.py:371` | **13** |
| **D3 aggregators** | Gate B entries carrying `defect_b` (§1.8) — a different population | **8**, of which **2** flip green here |

Plus the `visible_area` seam, the `_snapshot` dtype pin, and **four** breaking public changes (§9).
The size concern was raised and overruled; recorded rather than re-argued.

---

## 1. What is actually wrong

### 1.0 The population, derived by a SHAPE-AGNOSTIC detector

§4.2's detector binds one name to `{0.0, pitch-length}` through **any** construct — ternary,
`if`/`else` statement, `np.where`, dict literal. Run over `silly_kicks/`:

```
_gk_resolve.py:352       defended_goal_x                 IfExp     THE SEAM
_ghost_gk.py:850         _extract_all_ghost_gk_features  IfExp     fork A (map construction)
_ghost_gk.py:876         _extract_all_ghost_gk_features  IfExp     fork A (identity-keyed fallback)
_xcross_attempt.py:291   _build_goal_map                 IfExp     fork B
_shot_goalmouth.py:806   compute_shot_goalmouth          IfExp     PSO fallback -> _EXEMPT
features.py:3108         _gk_influence_at_actions        IfExp     -> add_gk_influence
features.py:3337         _closing_time_per_series        IfExp     -> add_gk_influence
features.py:3525         _get_gi                         IfExp     -> gk_influence_xfns
_gk_influence.py:318     compute_gk_influence            If/else   -> add_gk_influence   <-- rev 3 BLIND
_cover_shadows.py:611    lane_control                    If/else   -> add_cover_shadows  <-- rev 3 BLIND
_cover_shadows.py:910    compute_blocking_score          If/else   -> add_cover_shadows  <-- rev 3 BLIND
_cover_shadows.py:1073   _compute_cover_shadow_dict      If/else   -> add_cover_shadows  <-- rev 3 BLIND
```

**12 sites across 7 modules.** Rev 1 said two forks; rev 2 said four modules; rev 3 said eight
sites across five, having executed a detector that only recognised ternaries. Four sites — every
one of them an `if`/`else` **statement** — were invisible to it.

**Two false positives were found and excluded**, and both are worth recording because they show
where the semantic predicate over-reaches:

| hit | why it is not a goal end |
|---|---|
| `_ghost_gk.py:1935` | a `grid_spec` metadata dict in `save()` carrying `x_min`/`x_max` |
| `vaep/features/temporal.py:47` | `{1: 0, 2: 45, 3: 90, 4: 105, 5: 120}` — period start **minutes**; the `105` is not metres |

The dict-literal clause is therefore **dropped** from the shipped predicate: no real goal-end
derivation uses one, and it was the sole source of both false positives. Final predicate =
`IfExp` ∪ `If`/`else`-assigning-one-name ∪ `np.where`, which returns the 12 above with zero false
positives.

**The generalizable lesson, which is not rev 3's.** Rev 3 concluded "derive the population by
executing the detector". That is necessary and insufficient: a detector built from the sample
confirms the sample and nothing else. The predicate must be justified by the *semantics* of the
thing being detected, and its non-vacuity witness must be spelled in a form absent from the
sample (§4.2).

### 1.1 The forks, and five divergence axes

| axis | `_gk_resolve.defended_goal_x` (pinned) | `_ghost_gk.py:846-850` | `_xcross_attempt._build_goal_map:282-292` |
|---|---|---|---|
| grouped population | all non-ball players | GK rows only | all non-`"ball"` rows |
| no-GK fallback | outfield mean-x | none in the map | outfield mean-x |
| NA-team groups | kept (`dropna=False`) | **dropped** (default) | **dropped** (default) |
| ball exclusion | `~is_ball.astype(bool)` | `~is_ball.astype(bool)` | `team_id != "ball"` |
| `is_*` coercion | `.astype(bool)` ×2 | `.astype(bool)` ×2 | raw truthiness mask |

The last two axes were missing from rev 1 and both are live defects:

**Ball exclusion by string is already known-wrong inside the same call chain.**
`_model_eval.py:143` — a consumer of this very map — documents:

> *"Filter by `is_ball` (not the string 'ball') so a provider/fixture that encodes the ball's
> team_id differently can't be mistaken for a defending team (it would then have no GK row -> the
> frame would be dropped)."*

On any provider whose ball row carries a non-`"ball"` team_id, the xCross map has a different key
set from the pinned one. No committed fixture exercises that encoding, which is precisely why a
naive equivalence oracle would pass.

**`.astype(bool)` is the ADR-019 string trap, and the pinned implementation commits it twice.**
`_shot_goalmouth.py:768` calls `_truthy_bool(frames["is_ball"])` with the comment *"ADR-019: never
`.astype(bool)` a string column"* — and two lines later calls `defended_goal_x(frames)`, which
does exactly that at `_gk_resolve.py:347` and `:350`. `_gk_geometry.py:289-292` states the reason:
`pd.Series(["False"]).astype(bool)` is `True`. The xCross raw mask is a third behaviour again — it
raises on an object column containing NA and on a nullable `boolean` column containing `pd.NA`.

**Consequence for the design:** unification must choose **per axis**, not adopt one implementation
wholesale. §1.5 already establishes that principle for the fallback axis; §2.3 now applies it to
all five.

### 1.2 The ghost defect is not the one recorded

`TODO.md:53-62` says the ghost fork "lacks `defended_goal_x`'s mean-outfield-x fallback when a
`(game, period, team)` carries no GK rows". **Measured: not reachable as an output divergence.**
The ghost loop iterates GK *rows* (`_ghost_gk.py:874`), so a team with no keeper in the frame never
reaches the lookup.

The live divergence is the `dropna` one, landing on an identity-keyed fallback (`_ghost_gk.py:876`):

```python
goal_x = _defending_goal.get((gid, pid, gk_team),
                             0.0 if same_id(gk_team, home_team_id) else _FIELD_LENGTH)
```

Measured:

```
NaN-team keeper at x= 100.0  true goal_x= 105.0  fork fallback= 105.0  coincides
NaN-team keeper at x=   5.0  true goal_x=   0.0  fork fallback= 105.0  *** WRONG ***

same_id(nan, 1) = False  ->  the fallback is a CONSTANT 105.0 for any NA-team keeper
```

Wrong for half the possible cases, and wrong *silently*: the feature vector is computed in the
mirrored goal-relative frame and the model scores it without complaint.

### 1.3 The code documents the hazard six lines above the line that commits it

`_ghost_gk.py:842-844`: *"Using team identity alone to assign goal_x is wrong for the flipped
period."* The fallback at `:876` uses team identity alone — the ADR-028 **D3** shape. The pinned D3
unit does not cover it: `tests/tracking/test_mirror_registry.py:294` opens a hard-coded
`unit = {_defensive_line.py, _packing.py, _gk_influence.py}` and `:308` asserts `reads == unit`.
The gate walks **only those three files**, so it is structurally incapable of seeing a fourth
identity-keyed module — which is exactly how `_ghost_gk.py` escaped, and how the three
`features.py` sites escaped (§1.6). §4.6 records why widening that gate is NOT the answer, and
§1.8 records what DID catch them.

### 1.4 Reachability is NOT established

`_gk_identification.derive_goalkeepers:80` raises on a NA `team_id` in player rows, but SkillCorner
skips derivation (it trusts its native roster), so that guard does not cover every path. **No real
provider has been demonstrated emitting a NA-team GK row.** The divergence is structural, measured
on constructed frames. §6 measures frequency rather than assuming it in either direction.

### 1.5 The seam's guess can be wrong, and its docstring says why it exists

`_gk_resolve.py:327-330` states the fallback's purpose (**N1**):

> *"GK identification quality is provider-variable (Metrica/SkillCorner were 21-50% pre-fix).
> Prefer mean GK x; fall back to the team's mean outfield x when a (game, period, team) has no GK
> rows, **so a mis-/missing-GK does not silently drop the team from the goal map**."*

That intent is load-bearing and rev 1 did not mention it. But the guess can also be wrong: on the
NaN-team fixture the outfield mean returned `0.0` for a team whose keeper defended `105.0`.
Outfield mean-x is a much weaker proxy than keeper mean-x — a team's outfielders cluster toward the
goal they *attack*. The direction of that error is a real property of the heuristic; its frequency
on real data is unmeasured (§6).

So the fix is neither "adopt the seam wholesale" (replaces a gap with a confident error) nor
"delete the fallback" (discards N1's deliberate coverage). It is: **keep the guess, make consuming
it explicit, and let each consumer choose** — §2.2.

### 1.6 Three unrecorded identity-keyed derivations in the public `add_gk_influence` path

The signature (§1.0) surfaced three sites that appear in **no** ADR, no D3 unit and no `TODO.md`
entry. All three are literally:

```python
gk_team = gk_rows.iloc[0]["team_id"]
goal_x = 0.0 if same_id(gk_team, home_team_id) else 105.0
```

| line | enclosing function | reaches |
|---|---|---|
| `features.py:3108` | `_gk_influence_at_actions` | `add_gk_influence` |
| `features.py:3337` | `_closing_time_per_series` | `add_gk_influence` |
| `features.py:3525` | `_get_gi`, nested inside `gk_influence_xfns` | `gk_influence_xfns` |

**The `goal_x` EXPRESSION carries no period term** — directly checkable by reading it, and the
substantive claim. (Rev 3 said "no `period_id` within 15 lines", a line-window heuristic that is
both weaker and borderline false: at `features.py:3525` the enclosing `_get_gi(period_id,
frame_id_int, team_id)` is declared at `:3504` and uses `period_id` at `:3510`, exactly 15 lines
above. The function HAS the period; the derivation ignores it.) They are not fallbacks — they are
the primary derivation, and they structurally cannot express a team defending different ends in
different periods. They are correct today only because `convert_to_frames` emits
home-attacks-right in every period, i.e. correct *by accident of convention* — precisely what
`_ghost_gk.py:842-844` documents as wrong and what CLAUDE.md calls the ADR-028 **D3** shape.

**Why the D3 unit missed them.** `_gk_influence.py` **is** in the pinned unit — but the derivation
lives in `features.py`, which the unit does not walk (§1.3). The same blindness as the ghost fork,
one module over.

**Correction to an earlier draft of this section.** Rev 3 said these sites "appear in no ADR, no
D3 unit and no `TODO.md` entry". That is true of `features.py` as a *location* and misleading about
the *defect*: `tests/tracking/_mirror_entries/influence_family.py:43-50` records it, with a
measurement and an owner, pointing at `_gk_influence.py:318` (§1.8). The finding was made months
ago; only the location was recorded incompletely.

### 1.7 A fourth D3 site of a DIFFERENT shape

`_gk_influence.py:371-372` is already named in CLAUDE.md as a D3 re-key target, and it is **not**
a goal-map fork:

```python
if not same_id(attacking_team_id, home_team_id):
    threat_grid = threat_grid[::-1, ::-1]
```

It gates a grid *reflection* on team identity rather than deriving `goal_x`, so it cannot be fixed
by calling the seam. It shares the diagnosis, not the mechanism, and needs re-keying on
`acting_team_attacks_rtl` (`_action_orientation.py`). Kept in scope (§2.7) because leaving the one
already-named site unfixed while fixing three unnamed ones would be indefensible.

**So `_gk_influence.py` carries TWO identity-keyed sites, not one**: `:318` (a goal-end binding,
in §1.0's census) and `:371-372` (a reflection gate, outside it). The registry records both.

**Two different "eights" appear in this document; the bases differ.** §1.0's census counts
**expressions** that bind a goal end (12, of which 8 are ternaries). §1.8's Gate B census counts
**aggregators** whose behaviour is identity-keyed (8). They are different populations that overlap
in exactly two members — `add_gk_influence` and `add_cover_shadows`.

---

### 1.8 The census that already existed, and is behavioural

`tests/tracking/_mirror_entries/` records **8 aggregators** carrying
`defect_b="D3 re-key pending: identity-keyed direction (spec 4.3)"`:

| file | aggregators |
|---|---|
| `defensive_line_and_breaks.py` | `add_defensive_line`, `add_line_break`, `add_off_ball_context` |
| `influence_family.py` | `add_gk_influence`, `add_player_influence`, `add_cover_shadows` |
| `shape_and_structure.py` | `add_structural_pass`, `add_packing` |

`influence_family.py:74-84` names `_gk_influence.py:318` explicitly, calls `add_gk_influence` the
**"EIGHTH D3 member -- found by Gate B, not by the audit that produced the spec's list of seven"**,
and records the measured movement: `gk_pitch_control_share_weighted` **+0.1085**,
`gk_closing_time_min_s__six_yard_box` **+4.38 s**, identical under `->AWAY` and `->999999`.

**This is the finding of rev 4.** A behavioural gate found, months ago, the site that three
revisions of syntactic reasoning missed — and it found it by *running the aggregator with
`home_team_id` varied*, which no parse can simulate. It is also the argument §4.6 makes for
dropping the syntactic enumeration, arriving from the opposite direction.

> **The AST gate is a fast tripwire. Gate B is the census.**

**A stale comment, corrected here.** `influence_family.py:43-50` says *"GATE B FAILS AND IS NOT
XFAILED -- deliberately… the honest state is RED until the spec's D3 list is corrected or the
re-key lands."* That is no longer true of the code beneath it: the entry at `:82` carries
`defect_b=`, which becomes `pytest.mark.xfail(strict=True)` at `test_mirror_registry.py:224`.
Measured on `12f77f9`:

```
182 passed, 15 skipped, 8 xfailed, 1 warning     (ZERO failures)
```

The marker was added without updating the prose above it. Two consequences:

1. **§5.5 gains this file** — the comment becomes false on merge in a second way.
2. **`strict=True` makes the registry edit mandatory, not advisable.** Fixing a site and leaving
   its `defect_b` in place produces an **XPASS**, which a strict xfail turns into a FAILURE. Rev 3
   called this "likely a hard CI failure"; it is certain, and in both directions — removing
   `defect_b` from an aggregator that does not actually go green fails too. Hence §4.5's criterion:
   **remove `defect_b` iff Gate B passes after the change**, measured per aggregator, never
   predicted.


### 1.9 The cover-shadow family — where "the same edit" is FALSE

`_cover_shadows.py:611`, `:910` and `:1073` are identical:

```python
if same_id(attacking_team_id, home_team_id):
    goal_x_own = 105.0   # defenders' own goal
else:
    goal_x_own = 0.0
```

They differ from every other site in the census in the thing that matters: the key is
`attacking_team_id` and the value wanted is the **defenders'** own end — the *opponent's* defended
goal. `GoalMap.get(gid, pid, attacking_team_id)` returns the attacking team's own end, which is the
**wrong end**, so §2.7's one-line replacement has no counterpart here. Three workarounds were
considered and all three are the disease:

| workaround | why not |
|---|---|
| `105.0 - get(..., attacking_team_id)` | derives an end by arithmetic identity instead of reading the map, and is **wrong in exactly the degenerate case** `_shot_goalmouth.py:787` handles explicitly (both teams mapped to one end). A second implementation of the rule, in the commit that deletes second implementations. |
| derive the opponent id, then `get` | `_model_eval.py:144-147` does this today and documents the `.dropna()` trap (*"boolean value of NA is ambiguous"*). Not a one-liner, and it re-derives opponent identity at a third site. |
| `ends_in_period(...)` then pick the other entry | precisely the raw-tuple picking at `_shot_goalmouth.py:783` that §2.1 exists to remove. |

**Hence `GoalMap.attacked_goal` (§2.1)** — a real lookup of the opponent's entry, returning `None`
when the (game, period) does not resolve to exactly one opponent. It serves `_shot_goalmouth` too,
which does the third workaround by hand today.

**Two of the three are PUBLIC — and, verified, need NO signature change.**

```python
lane_control(frame, passer_xy, receiver_xy, *, home_team_id, attacking_team_id, params=None)
compute_blocking_score(frame, attacking_team_id, xt, *, home_team_id, defenders_to_remove=None, ...)
_compute_cover_shadow_dict(frame_data, passer_xy, attacking_team_id, xt, *, home_team_id, ...)
```

`lane_control` and `compute_blocking_score` are in `tracking.__all__`, and none of the three takes
`game_id` / `period_id`. Review 4 concluded that consulting a `(game_id, period_id, team_id)`-keyed
map therefore requires changing two public signatures — a third and fourth breaking API change.
**Measured, that is not so.** The `frame` argument IS a tracking-frame slice and carries both keys:
`game_id` and `period_id` are in `TRACKING_FRAMES_COLUMNS`, and the call site builds the slice as
`frame_groups.get_group((pid_period, fid))` (`features.py:3703`). These functions already read
`is_ball` and `team_id` off that same frame; they read the map keys the same way.

**CORRECTED at rev 7 — this refutation was right about the KEYS and wrong about the MAP.**
Reading `game_id`/`period_id` off a frame slice does not give you *the period's frames*, and the
seam's contract is the **mean** GK x per `(game, period, team)` — the mean IS the robustness
(`_gk_resolve.py:327-330`). Building the map from one frame is a different estimator, measured at
**78.8% wrong** on broadcast-shaped data (§15). The functions therefore DO change signature — see
§2.8 — but not in the way review 4 predicted, and not merely additively.

One pre-existing assumption is worth recording rather than introducing: the grouping key is
`(period_id, frame_id)` **without** `game_id`, so this path already assumes one call = one match —
consistent with the rest of the tracking layer, and unchanged by this cycle.


## 2. Design

### 2.1 One value object that owns construction AND lookup

Rev 1 returned `tuple[dict, GoalMapReport]`. That was too weak on its own terms: `goal_map, _ =
defended_goal_x(frames)` discards the report exactly as cheaply as today's discard, and neither
`ruff` nor `pyright` flags it. The CLAUDE.md rule rev 1 quoted — *make the observed outcome
unrepresentable, not merely refused* — is satisfied by an object, not by a discardable tuple
element. Worse, rev 1's own migration step for xCross was `{**report.fallback, **goal_map}`: a
provenance-destroying plain dict, the very shape §2.1 argued against.

```python
@dataclass(frozen=True)
class GoalMap:
    """Defended goal end per (game_id, period_id, team_id). Keys are CANONICAL (ADR-019).

    Built only by ``resolve_defended_goals``. The mappings are wrapped in
    ``MappingProxyType`` -- ``frozen=True`` freezes the BINDING, not the mapping.
    """
    resolved: Mapping[tuple, float]       # GK mean-x, finite
    guessed:  Mapping[tuple, float]       # outfield mean-x, finite (N1 coverage)
    unresolved: frozenset[tuple]          # in NEITHER mapping

    def get(self, game_id, period_id, team_id, *, allow_guess: bool = False) -> float | None:
        """The end THIS team defends."""

    def attacked_goal(self, game_id, period_id, team_id, *, allow_guess: bool = False) -> float | None:
        """The end this team ATTACKS -- i.e. the end its OPPONENT defends.

        A real lookup of the opponent's entry, never ``105.0 - get(...)``: the arithmetic
        identity is wrong in exactly the degenerate case ``_shot_goalmouth.py:787`` handles
        explicitly (both teams mapped to the same end), and it would be a second
        implementation of the rule inside the commit that exists to delete them.

        Returns ``None`` when the (game, period) does not resolve to exactly one opponent,
        **OR when that opponent's end equals this team's own end**. The second guard is not
        redundant: in the degenerate case there IS exactly one opponent, so a count-only
        check passes and the accessor reports that a team attacks the goal it defends.
        Measured in the executability pass (§14) -- the accessor as first specced shipped
        with precisely the hole it exists to close.
        """

    def ends_in_period(self, game_id, period_id, *, allow_guess: bool = False) -> Mapping: ...
    @property
    def n_resolved(self) -> int: ...
    @property
    def n_guessed(self) -> int: ...

def resolve_defended_goals(frames: pd.DataFrame) -> GoalMap: ...
```

**The seam owns the lookup, not only the construction.** This is the hexagonal split the cycle is
actually about: the *policy* (may I consume a guess?) belongs at the edge, per CLAUDE.md's
"policy lives at the edge"; the *mechanism* (how a key is canonicalized and matched) belongs in the
port, once. Today the two consumers disagree — `gkdv/_engine.py:179-193` re-keys through
`canonical_id` with a note that a raw lookup *"would present as a plausible pile of
`no_goal_map_entry` drops rather than as the dtype bug it is"*, while `_shot_goalmouth.py:783`
compares raw frames-derived keys against raw `actions`-derived tuples with `==` across two
DataFrames whose id boxing can differ. Closing one fork while leaving that open would be a half
fix.

**Renamed, and deliberately not shimmed.** `defended_goal_x` would name half its return value.
`resolve_defended_goals` is the new public name; the old name is **deleted, not deprecated**.
Review 1 recommended a `DeprecationWarning` shim returning today's merged plain dict, correctly
noting it could not drift (pinned by §4.3's oracle, forbidden internally by §4.2's AST gate).
Weighed and declined: `defended_goal_x` has no entry in `docs/PRIVATE_CONSUMERS.md`, the lakehouse
was verified not to call it, and the owner ruled explicitly that breaking is acceptable. A hard
rename fails loud at import; a shim preserves the plain-dict name this cycle exists to retire.
**Recorded here so review 2 sees it weighed rather than missed.**

**`dropna=False` becomes a stated, tested contract**, because it is what makes `unresolved`
observable instead of silently vanished.

**Keys are canonical, and canonical means STRING.** Measured: `canonical_id(1)` is `'1'`,
`canonical_id(1.0)` is `'1'`, `canonical_id(pd.NA)` is `pd.NA` (never `None`). Two consequences
that are load-bearing rather than trivia:

* **No consumer may ever hold the mappings as a plain dict.** `{**guessed, **resolved}` produces
  string-keyed tuples, and every surviving raw-tuple lookup `goal_map.get((gid, pid, tid))` with
  integer ids then MISSES — silently, returning `None`, which the direction sites treat as
  fail-open. Consumers call `GoalMap.get` / `attacked_goal`, which canonicalize on the way in.
  This is also §2.1's own principle: the merged plain dict is exactly the provenance-destroying
  shape this seam exists to delete, and it must not reappear as a migration convenience.
* **An NA-team key tests `is pd.NA`, never `is None`.**

**Why `unresolved` is a third outcome rather than an absent key.** An NA-keyed dict entry is a
dtype-dependent coin flip. Measured:

```
stored np.nan, looked up np.nan -> HIT   (same singleton, matched by identity)
stored np.nan, looked up pd.NA  -> MISS  (different singleton)
groupby(dropna=False) on Int64 stores pd.NA; lookup with pd.NA -> HIT
```

An NA entry would resolve or not depending on which NA flavour each side holds — ADR-019 class.
Keeping such groups out of both mappings removes the hazard; naming them keeps them countable.

**The three states are mutually exclusive, and the ladder is explicit** (rev 3 contradicted itself
here: §2.1 said an all-NaN-`x` group was "in neither" while §2.4(c) said guess-tolerant consumers
would still reach it through `guessed`):

```
key -> resolved   iff  GK mean-x is FINITE
    -> guessed    iff  GK mean-x is not finite AND outfield mean-x IS finite
    -> unresolved otherwise            (NA team_id, or every x NaN)
```

**Both mappings need the NaN guard, not just the GK one.** If every player in a group has NaN `x`,
`all_mean` is NaN and `nan < 52.5 -> False -> 105.0` fabricates an end **through `guessed`**, by
the identical mechanism §2.3 fixes for the GK mean. The finiteness test is applied to each mapping
as it is built, not once.

### 2.2 Per-consumer policy — stated, not implied

| consumer | `allow_guess` | rationale |
|---|---|---|
| `gkdv/_engine` | **True** | preserves today's values; N1 coverage is what its domain filter assumes. Guess counted in `GkdvReport`. |
| `_shot_goalmouth` | **True** | preserves today's values; a refused shot is a lost measurement, and `shot_crossing_source` already carries provenance. |
| `_xshot_occurrence` | **True** | byte-identical to today (it *is* the extracted pinned implementation). |
| `_xcross_attempt` | **True** | byte-identical to today. |
| `causal/opportunities` | **True** | inherits xCross semantics unchanged. |
| `_model_eval` | **True** | inherits xCross semantics unchanged. |
| `_ghost_gk` | **False** | the only consumer where a wrong end silently corrupts a *model input*. A guessed end mirrors the whole goal-relative feature vector. |
| `features.py` ×3 (gk_influence) | **True** | new consumers (§1.6, §2.7); `True` preserves today's canonical-frame values exactly. |
| `_cover_shadows` ×3 | **True** | new consumers (§1.9); they need `attacked_goal`, not `get`. `True` preserves today's canonical-frame values exactly. |

**`_ghost_gk` passes `work`, not `frames`.** It builds its map from the post-subsample /
post-link-filter frame set (`_ghost_gk.py:826,839,847`), so the call is
`resolve_defended_goals(work)`. Calling it on `frames` would change the map whenever
`subsample_fps` is set. (Rev 2 recorded this in its response log but never put it in the body.)

Rev 1 left this implicit and thereby smuggled a second behavioural change — refusing gkdv and
`_shot_goalmouth` frames that resolve today — in under a correctness fix. Only `_ghost_gk` changes,
and only where it is currently wrong.

### 2.3 Per-axis choices

| axis | choice | reason |
|---|---|---|
| grouped population | all non-ball players | needed for the N1 guess |
| no-GK fallback | keep, as `_guessed` | N1 (§1.5) |
| NA-team groups | `dropna=False` → `unresolved` | §2.1 |
| ball exclusion | **`is_ball`, never the `"ball"` string** | `_model_eval.py:143` |
| `is_*` coercion | **`_truthy_bool`, never `.astype(bool)`** | ADR-019; `pd.Series(["False"]).astype(bool)` is `True` |
| all-NaN-`x` GK group | → **not `resolved`**; falls to `guessed`, or `unresolved` if outfield x is also non-finite | see below, and the §2.1 ladder |

The ball and coercion axes are *bug fixes to the pinned implementation*, not merely unification.
**Their blast radius is bounded and needs no measurement:** `schema.py:18-19` declares both
`is_ball` and `is_goalkeeper` as `"bool"`, and every in-repo converter casts to the declared dtype
(`kloppy.py:216`, `metrica.py:218`, `gradientsports.py:148`), so a string-typed frame set is
**unreachable from any converter path** — it can arrive only from a hand-built frame set or a
third-party builder. That strengthens §9's "no re-materialize" rather than leaving a promise
unkept. (Rev 2 forward-referenced a §6 measurement here that did not exist.)

**The all-NaN-`x` group is a latent fabrication of the same class this cycle fixes.** Today
`float(ref["x"].mean())` is `nan`, and `nan < 52.5` is `False` → **105.0**, silently. Measured:

```
float(nan) < 52.5 -> False  =>  goal_x = 105.0  (FABRICATED)
```

The three-outcome model has the natural home for it, so it routes to `unresolved`. This is decided
explicitly rather than frozen by §3.2's byte-identity requirement, which would otherwise lock the
wart in: the vectorized form is asserted byte-identical to the loop **on the §4.3 oracle's
well-formed frames**, and this shape is characterization-tested separately (§4.4).

### 2.4 Three intended behaviour changes, stated and tested

**(a) `_ghost_gk` refuses an unresolvable end.** A keeper whose group is in `unresolved` yields no
ghost. `ghost_gk_source` gains the token **`goal_end_unresolved`**, exported as
`GHOST_GK_GOAL_END_UNRESOLVED` and added to `GHOST_GK_SOURCE_VALUES` — the vocabulary is closed and
runtime-checked (`features.py:4636-4640` raises on an out-of-vocabulary emission). Emitted by
`add_ghost_gk` and `compute_ghost_gk`; `serve_ghost_gk_positions` has no row to carry a token and
therefore **returns no row**, reusing its existing `len(positions) == 0` branch, per ADR-054
Decision 2 (`gkdv/_engine.py:557-562` raises on a non-finite ghost).

**(b) `_shot_goalmouth` scores some shots it currently refuses.** Its resolution keys on
`len(ends) == 2` (`:788`). Today `dropna=False` puts an NA-team group *into* the map, so
`len(ends) == 3` → `"unresolved"`. Moving NA groups to `unresolved` restores `len(ends) == 2` →
**resolved**. The direction is refuse→resolve, the opposite of what rev 1 implied, and rev 1 also
wrongly claimed `"unresolved"` "becomes reachable" — the docstring at `:783-786` already names two
paths that reach it today.

This is judged **correct**: a shot was being refused because an unrelated NA-team row inflated a
count, not because its own geometry was ambiguous. It is nonetheless a value change on real data,
so it is characterization-tested (§4.4) and `scripts/validate_shot_goalmouth_sb.py` — an ADR-037
artifact driver — must be re-run if §6 finds NA groups in its corpus.

**(c) An all-NaN-`x` GK group stops fabricating an end** (§2.3). Today it silently returns `105.0`.
Under §2.1's ladder it is simply **not `resolved`**: it falls to `guessed` when outfield mean-x is
finite, and to `unresolved` only when it is not. `_ghost_gk` refuses because the key is absent from
`resolved` — not because it is in `unresolved`, which is the rejected model.

### 2.5 The visible-area seam — `tracking/_visibility.py`

**There is already an implementation of "fraction of the pitch observed"**, and shipping a second
in the commit that condemns second implementations would be self-refuting.
`providers/statsbomb/parse.py:108` `visible_fraction(flat)` computes shoelace ÷ (`SB_FIELD_LENGTH` ×
`SB_FIELD_WIDTH`) in native 120×80. Two concrete disagreements with what rev 1 proposed:

* **Empty polygon.** `parse.py:121-122` returns `0.0` — asserting *"we saw nothing"* where the
  truth is *"we don't know"*. That is the ADR-042 shape the section itself invokes. **Fixed to NaN
  in this cycle**, in the port, so there is one rule.
* **Clipping.** `polygon_to_spadl` is deliberately unclipped (`parse.py:136-139`: a broadcast camera
  legitimately sees past the touchline), so a raw shoelace ratio can exceed 1.0.

**Decision (rev 3): ONE quantity — the CLIPPED on-pitch share ∈ [0, 1]. The unclipped ratio is
deleted, not kept under a second name.**

The distinction that resolves it: **clipping the POLYGON destroys information; clipping the AREA
CALCULATION destroys nothing.** ADR-054 Decision 5 keeps `polygon_to_spadl` unclipped so the
observed region is not silently shrunk — that is about the *vertices*, and it stands untouched.
"What share of the pitch is inside this polygon" is a different question whose answer requires
intersecting with the pitch, and the vertices stay unclipped while it is computed.

Rev 2 left this ambiguous ("re-expressed in terms of the one implementation" admits two readings).
Keeping an unclipped variant was considered and rejected: it is not a football quantity, it has no
consumer, and its only actual caller (`build_sb360_coverage.py:255`) names the result
`mean_visible_pitch_fraction` — i.e. means the clipped thing, so today's value is arguably wrong
for its own use. Two near-identically-named quantities separated only by docstrings is exactly the
fork-by-documentation §1 condemns.

**It is RENAMED, not silently redefined.** `visible_fraction` → **`observed_pitch_fraction`**, and
the old name is deleted. §2.1 argued exactly this for `defended_goal_x` (*"a hard rename fails loud
at import"*), and the argument is stronger here: a renamed function fails loud at import, whereas a
public function that keeps its name while changing value on the **common** case is undetectable by
any consumer. Applying §2.1's principle inconsistently would be the fork-by-documentation this
spec condemns, one level up. It also erases the last trace of M18's two-quantities-one-name
problem.

The new name changes value on the common case (broadcast polygons routinely pass the touchline)
and on the empty case. Both are corrections, both recorded in §9, both swept in §5.6.

**The crc non-vacuity witness goes vacuous under clipping, and must be extended.**
`tests/providers/statsbomb/test_parse.py:214-219` witnesses ADR-054 D5's supporting argument —
*"crc is a pure TRANSLATION, so it is invisible to `visible_fraction`"* — using a polygon at
x 10-110, y 10-70, **entirely interior** to the 120×80 pitch. Under clipping that test keeps
passing while the property it witnesses becomes **false**: for any polygon crossing the touchline,
the 0.4375 m translation changes the intersection area. D5's *conclusion* stands (apply crc for
player/polygon alignment); its *supporting argument* does not. A guard that still passes while the
reasoning under it dissolves is worse than one that fails. **Fix:** extend the fixture to a
touchline-crossing polygon, and re-state D5's reason as **alignment**, recording the measured
crc-induced delta on the clipped fraction.

```python
def point_observed(polygon: np.ndarray, x: float, y: float) -> bool | None
def region_observed_fraction(polygon: np.ndarray, region: np.ndarray) -> float
def add_visible_area_coverage(actions, *, visible_area, links=None) -> pd.DataFrame
```

* **`region` is an `(M, 2)` polygon, not a bounding box.** The motivating consumers are
  `defenders_in_triangle_to_goal` (a **triangle**), `receiver_zone_density` /
  `nearest_defender_distance` (radii) and pitch-control cells. A bbox can only *over*-report
  coverage for a triangle — fabricating observation, the exact failure class this seam exists to
  prevent. Implemented via Sutherland–Hodgman polygon clipping (~30 lines, dependency-free,
  matching the existing stdlib shoelace idiom), with a rectangle convenience caller.
* **`point_observed` returns `bool | None`.** `False` means "not observed", which is a claim; a
  missing or degenerate polygon warrants `None`. A bare `bool` would contradict the NaN rule ten
  lines below it.
* **One column, not two.** An earlier sketch also carried `query_region_observed_fraction`, which
  requires deciding *which* region — the triangle to goal? a radius? the pass lane? Each is a
  different feature's question, and choosing here would be the consumer decision this cycle defers.
  The library ships the frame-level fraction (well-defined, no choice needed) plus
  `region_observed_fraction` for the consumer's own region. ADR-009.
* **`visible_area_source` vocabulary** (closed, each token exported, matching `DAS_SOURCE_VALUES` /
  `GHOST_GK_SOURCE_VALUES`): `observed` (polygon present and non-degenerate), `no_polygon` (action
  has no 360 record), `degenerate_polygon` (fewer than 3 vertices), **`unlinked`** (the action has
  no linked frame — the aggregator takes `links`, so link failure is representable and
  `ghost_gk_source` already carries this exact token for the same reason).
  `visible_area_fraction` is NaN for all three non-`observed` tokens — **never 1.0, never 0.0**.

**Placement.** `polygon_to_spadl` already yields SPADL coordinates, so these primitives are
provider-agnostic on arrival — SkillCorner partial visibility is the same shape. `providers/` is a
parse-port layer; an `add_*` there would invert the dependency direction the ports respect.

**Scope boundary.** Wiring coverage *into* the count features is NOT in this cycle: it changes
existing values and decides for the consumer what a partial observation means (ADR-009).

**Live downstream, recorded:** `visible_area` already reaches the lakehouse
(`dbt_project/models/staging/statsbomb/stg_statsbomb__360.sql` and its schema/source YAML, plus
`src/tests/fixtures/statsbomb_bronze_schema_snapshot.json`), so this seam has a real consumer even
though §7 defers lakehouse adoption of the port.

### 2.6 The snapshot dtype pin — a contract test, not a version test

`_snapshot.py:172` (`pd.concat`) yields `Int64` on pandas 2.3.3 and `Float64` on 3.0.3 — the
concat-with-all-NA `FutureWarning` materialising.

Rev 1 said "test the dtype on both majors". **That is not implementable:**
`.github/workflows/ci.yml:34-42` is OS × Python only — there is no pandas axis — and
`pyproject.toml:32` pins `pandas>=2.1.1,!=3.0.4`, leaving the major to resolver output. A
version-shaped test would also silently stop covering both the day the Python floor moves.

Instead: **cast to the declared dtype immediately after the concat, and assert
`frames[c].dtype == TRACKING_FRAMES_COLUMNS[c]` for the id columns.** Version-independent, catches
the next promotion-rule change, needs no CI edit.

### 2.7 The D3 re-key — three mechanisms (see the Executive Summary for the counts)

**Owner decision (rev 3, confirmed revs 4-6): in scope.** The re-key spans **3 modules** in
**three** mechanisms (counts in the Executive Summary table):

| mechanism | sites | edit |
|---|---|---|
| own-end binding | `features.py` ×3, `_gk_influence.py:318` | `goal_map.get(..., allow_guess=True)` |
| **opponent-end** binding | `_cover_shadows.py` ×3 (`:611`, `:910`, `:1073`) | `goal_map.attacked_goal(...)` — §1.9 |
| **direction bool** | `_cover_shadows.py:704`, `:1030`, `_gk_influence.py:371` | `attacked_goal(...) == 105.0` |

**Two corrections from the executability pass (§14), both measured.**

1. **`_gk_influence.py:371` cannot use `acting_team_attacks_rtl`.** That helper is
   `(actions, frames) -> pd.Series`, i.e. **per-action**; `compute_gk_influence` is per-frame and
   has no `actions`. The workable re-key is the seam itself — the attacking team attacks
   `attacked_goal(...)`, so `105.0` means no reflection and `0.0` means reflect. Every site in the
   cycle therefore routes through **one** of two accessors.
2. **`_cover_shadows.py:704` and `:1030` are added to scope (owner, rev 6).** They bind a *bool*
   (`attacking_toward_high_x`), not `{0.0, 105.0}`, so the goal-end detector correctly does not
   see them — and **measured, the three goal-end bindings alone move ZERO of `add_cover_shadows`'
   five Gate B columns.** With these two included all five go to exactly `0.0` and the aggregator
   XPASSes. Sites EDITED by this cycle are therefore **13**: the 10 replaceable census sites,
   these two direction bools, and `_gk_influence.py:371`. See the counting table in the
   Executive Summary -- three denominators are in play and they are not interchangeable.

The `features.py` sites are not adjacent work — they are three more callers of the exact rule this
cycle unifies, and their edit is the same one the map-constructing forks get:

```python
- goal_x = 0.0 if same_id(gk_team, home_team_id) else 105.0
+ goal_x = goal_map.get(gid, pid, gk_team, allow_guess=True)
```

`allow_guess=True` preserves today's values exactly on canonical frames. Routing them through the
seam also gives them a **period term they structurally lack today** (§1.6, §1.9), which is the D3
fix — it falls out of the unification rather than being bolted on.

The cover-shadow three take the same treatment through `attacked_goal` instead of `get`; §1.9
records why that accessor had to exist and why no public signature moves.

`_gk_influence.py:371-372` (§1.7) is a **different mechanism**: an identity-gated grid
reflection, not a `goal_x` derivation. It re-keys onto **`GoalMap.attacked_goal`** — *not*
`acting_team_attacks_rtl`, which is `(actions, frames) -> Series`, i.e. per-ACTION, and cannot
serve a per-frame call (measured, §14.2). Included because leaving the one already-named D3 site
unfixed while fixing the unnamed ones would be indefensible.

**Value effect: none on canonical frames — and the contradiction rev 3 leaned on is resolved.**
Rev 3 quoted `_ghost_gk.py:838-841` as authority for why identity-keying is wrong (*"On
LTR-normalized data with period flips (e.g. SkillCorner), teams swap ends at halftime"*) while also
asserting that every converter emits home-attacks-right. Both cannot hold. The authority is
`utils.play_left_to_right`'s own docstring:

> *"Normalize tracking frames so the home team attacks left-to-right in **every period**."*

So for frames that went through the orientation seam there is **no period flip**, and
identity-keying is correct-by-convention. The ghost comment is **wrong for oriented frames**; this
cycle edits those exact lines, so it is corrected rather than quoted. What identity-keying cannot
survive is an *unoriented* frame set — ADR-029's consumer-built case — which is the real defect.

That reasoning is still an argument, so it is backed by a measurement (§6 M3) rather than trusted:
count `(game, team)` whose GK-derived end differs between period 1 and period 2, per corpus. Zero
proves the invariant; non-zero makes §2.7 a value change with a retrain question.

**Gate B and M3 answer different questions and neither substitutes for the other.** Gate B holds
the frames FIXED and varies `home_team_id`, so it detects that direction is inferred from identity
— and it already fires for this family (`influence_family.py`, §1.8), which is why the RED for this
half of the cycle landed months ago. It cannot tell whether real data actually exercises the
difference, because it varies the *parameter*, not the data. M3 varies the data. The re-key is
justified by Gate B; the *value-effect* claim is justified by M3.

**Coordination:** ADR-028 D3 re-keying has been the other session's ADR-051 territory. Their PR 5
is not on `main` (HEAD is `12f77f9`, this session's merge). Per the owner's 2026-07-31 ruling
their inflight work is not a constraint, but `features.py` and `_gk_influence.py` should be
checked for in-flight conflict before implementation begins.

---

### 2.8 The map's LIFETIME, and the parameter it replaces

**Built once per match from the full frames, then threaded.** The seam's contract is the mean GK x
per `(game, period, team)`; a per-frame map is a different estimator and measured **78.8% wrong**
on broadcast-shaped data (§15). Eight of the fourteen sites live in per-frame functions, so the map
has to arrive from outside.

**`goal_map: GoalMap` REPLACES `home_team_id`. Required, no default.**

| function | public? | `home_team_id` reads today |
|---|---|---|
| `lane_control` | **yes** | `:611` only |
| `compute_blocking_score` | **yes** | `:910`, plus 2 pass-throughs |
| `_voronoi_threat` | no | `:704` only |
| `_compute_cover_shadow_dict` | no | `:1030`, `:1073`, plus 3 pass-throughs |
| `compute_gk_influence` | no | `:318`, `:371`, plus 1 pass-through |

**Verified by AST: every read is a direction derivation or a pass-through to one.** Nothing else
uses it, so replacing rather than adding loses nothing.

**Why replace and not add a keyword.** An additive `goal_map=None` keyword is the repo's `links` /
`pitch_control_cache` pattern and would be non-breaking — but it keeps `home_team_id` in the
signature and keeps the identity-keyed branch alive and reachable, which is the thing this cycle
exists to delete. Replacement makes the D3 defect **unrepresentable** in these signatures rather
than merely unused — the same principle §2.1 applies to the guessed end. CLAUDE.md records the
weaker precedent and its own limitation: `compute_space_created`'s `home_team_id` *"stays in the
signature and stays UNREAD — D3 retires it by disuse, not removal."* Disuse was chosen there
because breaking was not free. Here it is.

**Why not "take full frames".** Handing a per-frame geometry function `frames` plus a key makes it
responsible for slicing — a concern it does not have — and it would rebuild the map per call unless
something threads a cache, which is the keyword option in disguise. The function should receive the
*resolved fact*, not the data to reduce.

**No default is part of the design.** A `goal_map: GoalMap | None = None` fallback would silently
re-admit per-frame construction — the 78.8% path — at exactly the call sites that forget to pass it.

### 2.9 ONE policy for an unresolvable end

Rev 6's plan grew four different handlings of the same condition (raise / `continue` / coerce to
`False` / fail-open), and two of them sat inside `add_*` functions whose contract is NaN-tolerance
(ADR-003). That is a contradiction, not a nuance.

**The policy is: NaN + provenance, never raise, never coerce.** It is the `ghost_gk_source`
precedent this cycle already establishes (§2.4a), applied uniformly:

| seam | on an unresolvable end |
|---|---|
| per-row aggregators (`features.py` ×3, ghost) | NaN output row + a provenance token |
| per-frame functions (`lane_control`, `compute_blocking_score`, `compute_gk_influence`, …) | return their documented empty/NaN result; **never** raise |
| `serve_ghost_gk_positions` | no row (ADR-054 Decision 2 — gkdv raises on a non-finite ghost) |

**Fail-open is specifically forbidden at the direction sites.** `if attacked == 0.0` and
`attacked == 105.0` both silently treat `None` as "attacking rightward". Measured: `attacked_goal`
returns `None` for **34.2%** of team-frames under per-frame construction (§15). Even with the map
built correctly, the guard must be explicit — an unresolved direction produces a NaN result, not a
default one.


## 3. Evidence

### 3.1 Value equivalence on well-formed frames (measured)

Two periods, ends swapped at half-time, every team with a keeper:

```
key            pinned   ghost   xcross   verdict
(1, 1, 1)         0.0     0.0      0.0    agree
(1, 1, 2)       105.0   105.0    105.0    agree
(1, 2, 1)       105.0   105.0    105.0    agree
(1, 2, 2)         0.0     0.0      0.0    agree
=> IDENTICAL
```

**What this does and does not license.** It licenses "no retrain on well-formed frames". It says
**nothing** about the GK-less, NA-team, string-`is_ball` or string-`is_goalkeeper` shapes — which
are precisely where behaviour changes. Rev 1 leaned on this measurement for claims it cannot
support; §4.4 and §6 close that gap.

### 3.2 Performance — a speed-up, not a regression

Map construction on 3,105,000 rows: pinned `0.381 s`, ghost `0.040 s`, xcross `0.348 s`. Rev 1
accepted +0.34 s per match at the ghost site. That regression is avoidable: `_gk_resolve.py:349-352`
is a Python `for key, grp in groupby(...)` with a per-group `.mean()`. Vectorized —

```python
gk_mean  = gk.groupby(keys, dropna=False)["x"].mean()
all_mean = players.groupby(keys, dropna=False)["x"].mean()
# _resolved = gk_mean ; _guessed = all_mean restricted to keys absent from gk_mean
```

— semantics are identical and unification becomes a **speed-up at every construction site**. The vectorized
form must be asserted byte-identical to the loop on the §4.3 oracle.

### 3.3 Aggregator count

`tracking.__all__` holds 33 `add_*`, of which `add_gradientsports_player_ids(jersey_frames, roster)`
is not action-coupled — so the C4 DSL's *"32 action-coupled aggregators"* is **correct, not stale**.
`add_visible_area_coverage` takes it to 33 → DSL edit + regen.

---

## 4. Gates

Per ADR-051, **detection lands before the fix** — every gate below is written and observed RED
before a call site changes.

### 4.1 Behavioural differential gate

A scene where an identity-keyed derivation names the **wrong** end; every goal-consuming aggregator
must match the seam. Fixtures must include, at minimum:

1. ends-swapped two-period scene (the identity-keying trap);
2. a `"False"`-string `is_ball` / `is_goalkeeper` frame set (the ADR-019 trap, §2.3);
3. a nullable-`boolean` frame set containing `pd.NA` (the xCross raw-mask raise);
4. a ball row whose `team_id` is neither `"ball"` nor NA (the `_model_eval.py:143` case).

Both sides asserted: a planted identity-keyed variant must FAIL, or a green reading proves nothing.

### 4.2 AST enumeration gate — with a stated predicate

Rev 1 gave no detection rule. Rev 2 gave one that **matched only 1 of the 3 implementations it had
to catch** — measured: the seam (`_gk_resolve.py:349`) and xCross (`_xcross_attempt.py:288`) use a
loop-then-aggregate shape with the `["x"]` in the *body*, so only `_ghost_gk.py:848`'s chained
`groupby(...)["x"].mean()` matched. A third fork written in the majority style would have passed.
It was also over-inclusive — `_defensive_line.py`, `_line_breaking.py`, `kloppy.py`,
`skillcorner.py`, `_bravery.py` all group on `team_id` — and every one of those would have needed
an `_EXEMPT` judgement call, which is the ADR-043 rot §4.2 claims to escape.

Rev 3 then gave an `IfExp`-only predicate, which was **built from the shapes already found** — all
ternaries — so executing it confirmed those and nothing else. Four `if`/`else` statements were
structurally invisible (§1.0). A detector derived from its own sample is self-confirming.

**The predicate is SEMANTIC, not node-shaped:**

> any construct binding **one name** to `{literal 0.0, pitch-length}`, where pitch-length is
> literal `105.0` or a `Name`/`Attribute` matching `*FIELD_LENGTH*` / `*PITCH_LENGTH*` —
> via `IfExp`, `If`/`else` assigning that name in both branches, or `np.where(c, a, b)`.

Run against the tree: **12 sites across 7 modules, zero false positives** (§1.0 records the two
dict-literal false positives that caused the dict clause to be dropped).

**The non-vacuity plant MUST be an `if`/`else` statement** — the spelling absent from the sample
the predicate was built against. A ternary plant would only prove the gate catches copy-paste,
which is the failure rev 3 shipped.

Post-cycle, population ⊆ `{seam} ∪ _EXEMPT`, where `_EXEMPT` holds:

| exempt site | reason |
|---|---|
| `_shot_goalmouth.py:806` | the documented PSO / degenerate ball-mean fallback (`:781-782`) — a *last-resort* end when the goal map itself is degenerate, so it cannot consult the map |

`tracking/direction.py` is **not** listed: the detector returns zero sites there, so an exempt
entry for it would never match. Per ADR-051's both-directions house rule the gate asserts
**equality**, not subset — which makes a never-matching exempt entry a failure, correctly. Its
exemption belonged to rev 2's `home_team_id` predicate, not to this one.

(The plant requirement is stated once, above: it must be an `if`/`else` statement.)

**Why an AST gate here when ADR-043 deleted the id-dtype lint.** That lint failed because a safe
same-source compare and an unsafe cross-source one are *the identical AST* — only provenance
separated them. That does not hold **for this signature**: it returns 12 sites across 7 modules
with zero false positives, and the one exempt site differs in *purpose* (`_shot_goalmouth.py:806`
is the last-resort fallback for when the map itself is degenerate). This is the ADR-050 shape.

**Note the contrast with §4.6**, which is the same question answered the other way: "reads
`home_team_id`" collects 14 modules of which most are correct, and there the ADR-043 objection
*does* bite. The difference is precision, not principle — which is why one gate ships and the other
is dropped. The behavioural gates remain the backstop for both.

### 4.3 Equivalence oracle — captured from the PRE-change tree

The goldens are generated from the tree **before** any call site changes and committed in the
RED step. **One golden per affected public surface**, which after §2.7 includes
`add_gk_influence` and `gk_influence_xfns` and `add_cover_shadows` — not merely the original six.
This matters because §4.3 is the gate carrying §2.7's "no value effect on canonical frames" claim, and the RED step demonstrates them failing against a deliberately-wrong seam. An oracle
authored after the change is a tautology (ADR-051: *"a gate written after its own repair arrives
green and is never observed failing"*).

### 4.4 Characterization tests — per site × per degenerate shape

The gap rev 1 left: §4.3 covers well-formed frames (nothing changes) and §4.1 covers a wrong-end
scene, so **the only untested cases were the only cases that change**. Required matrix:

| shape | before | `_ghost_gk` | `_xcross` | `gkdv` | `_shot_goalmouth` | `_xshot` | `features.py` ×3 | `_cover_shadows` ×3 |
|---|---|---|---|---|---|---|---|---|
| GK-less `(g,p,team)` | guess in map | n/a (§1.2) | guess used | guess used | guess used | guess used | guess used | guess used |
| NA-team GK row | in map (`dropna=False`) | **no ghost** | unchanged | unchanged | **resolves** (§2.4b) | unchanged | unchanged | **`attacked_goal` -> None** |
| all-NaN-`x` GK group | **`105.0` fabricated** | **no ghost** | guess used | guess used | guess used | guess used | guess used | guess used |
| string `is_ball` | **map is `{}`** | correct map | correct map | correct map | correct map | correct map | correct map | correct map |
| `pd.NA` boolean | xCross **raises** | correct map | correct map | correct map | correct map | correct map | correct map | correct map |

Each cell asserts the **new intended outcome explicitly** — drop / refuse / merge-and-mark — not
merely that it differs from before.

**The string-`is_ball` before-state is worth writing down**, because it is more dramatic than
"wrong": with `is_ball` as `"True"`/`"False"` strings, `~frames["is_ball"].astype(bool)` is `False`
for **every** row (every non-empty string is truthy), so `players` is empty and the map is `{}` —
every consumer refuses everything. The cell asserts `{}` → correct map.

### 4.5a Registry edits — `defect_b`, by measurement

`defect_b` becomes `pytest.mark.xfail(strict=True)` (`test_mirror_registry.py:224`), so the edit is
mandatory and symmetric:

> **Remove an aggregator's `defect_b` iff Gate B passes after the change.** Measured per
> aggregator, never predicted.

Strict xfail fails in both directions — leave the marker on a now-fixed aggregator and the XPASS
fails; remove it from one that did not actually go green and the gate fails. Neither can be
resolved by judgement.

**MEASURED (§14), no longer expected.** Running the proposed diff against the real gate:

| aggregator | result |
|---|---|
| `add_gk_influence` | **XPASS(strict)** -> `defect_b` REMOVED |
| `add_cover_shadows` | **XPASS(strict)** -> `defect_b` REMOVED (requires the two `:704`/`:1030` sites; with only the three goal-end bindings it moves 0 of 5 columns) |
| the other six | untouched, `defect_b` retained — they have no goal-end site at all (§1.8) |

Rev 5 predicted `add_cover_shadows` would NOT flip, reasoning from `influence_family.py:147`'s
recorded `blocking_score` movement of 148.83. That inference was wrong: the movement is real but
its cause is the two direction-bool sites, and once those are re-keyed the column goes to exactly
`0.0`. **This is why §4.5a states a criterion rather than a prediction** — the prediction was made
twice and was wrong twice.

### 4.5 Registrations for the new aggregator

`add_visible_area_coverage` registers in: the feature glossary (ADR-048), the liveness gate
(non-NaN **and** non-constant — the fixture must vary coverage across actions, not supply one
polygon), `PURITY_ENTRIES` (ADR-033; ≥2 variants, since it branches on polygon presence), the
mirror registry (Gate A + Gate B — coverage fraction is **mirror-invariant**, since reflecting the
scene reflects polygon and points alike, so the entries assert a real property), the id-dtype
invariance gate (ADR-019), the SB360 verdict registry (ADR-053), and the C4 DSL.

### 4.6 The mirror-registry enumeration — considered, measured, DROPPED

`tests/tracking/test_mirror_registry.py:294-308` walks only three hard-coded files, so
`assert reads == unit` structurally cannot see a fourth identity-keyed module — the mechanism by
which `_ghost_gk.py` escaped (§1.3), and by which the three `features.py` sites escaped (§1.6). Per
ADR-051's both-directions rule it must enumerate `silly_kicks/tracking/*.py`, collect every module
reading `home_team_id`, and assert against `unit ∪ _EXEMPT_WITH_REASON`.

**Rev 3 ran it, and the result kills the gate as specified.** Measured over
`silly_kicks/tracking/*.py`:

```
modules reading home_team_id : 20
modules KEYING on it         : 14   (via same_id / ids_match / raw ==)
currently in the D3 unit     :  3
NOT in the unit but keying   : _cover_shadows, _ghost_gk, _line_breaking, _off_ball_runs,
                               _player_influence, _structural_pass, _xcross_attempt,
                               direction, features, kloppy, utils
```

**Most of those keyings are correct.** `direction.py:169,252,378` keys on `home_team_id` to compute
attacking direction — that is its entire job. `utils.py:177,287` and `kloppy.py:133` are the
orientation machinery. An enumeration gate would go RED on eleven modules and demand eleven
`_EXEMPT` judgement calls, which is precisely the exemption rot ADR-043 records and which §4.2
escaped only by finding a *precise* signature.

**And it cannot be narrowed the way §4.2 was.** `direction.py` comparing `home_team_id` to decide
direction and `features.py:3108` comparing it to decide `goal_x` are **the identical AST**; only
the provenance of what the comparison *feeds* separates them. That is the case ADR-043 declares
unsolvable by syntax, and the reason it deleted the id-dtype lint rather than widening it.

**Decision: §4.6 is DROPPED from this cycle.** What replaces it:

* §4.2's **semantic** signature catches the goal-map defect class precisely — 12 sites, 7 modules,
  0 false positives after the dict clause is dropped (§1.0) — which is this cycle's actual concern,
  and it does not rot. (Rev 4 left this sentence describing the superseded `IfExp` predicate and its
  8-site count. A subtraction has to be justified under the CURRENT predicate or a reader is right
  to reopen it.)
* Gate B (vary `home_team_id` on fixed frames) is the *behavioural* detector for identity-keying,
  and it already runs over all 33 registered `add_*`. It is the backstop ADR-043 says a lint can
  never be.
* The 14-module measurement above is recorded in `TODO.md` as input to ADR-051's D3 workstream,
  with the note that a *syntactic* D3 gate is not available and behavioural Gate B is the tool.

**Stated honestly, the enumeration WOULD have found something.** `_cover_shadows.py` is in its own
"NOT in the unit but keying" list above, and §1.9 shows those three sites are real and are the
hardest part of this cycle. So the argument is not *"it would have found nothing we need"*. It is:
it would have found **those four plus eleven correct keyings**, at the price of eleven exemption
judgements that rot — and Gate B already carries `add_cover_shadows` with a measurement, without
any of that cost.

This is subtractive, and deliberately so: dropping the gate dissolves eleven exemption decisions,
an unbounded discovery risk, and the scope question of who owns each finding — while the detection
it would have provided is already held, behaviourally, by Gate B.

---

## 5. Caller sweep (CLAUDE.md:146)

The sweep is the FLOOR. Every caller of every changed function, classified with evidence both ways,
plus the four things a symbol sweep structurally cannot see.

### 5.1 Direct callers of `defended_goal_x` → `resolve_defended_goals`

| caller | affected? | evidence |
|---|---|---|
| `gkdv/_engine.py:190` | YES — `allow_guess=True`; drop its local `canonical_id` re-keying (seam owns it) | reads the return value |
| `_shot_goalmouth.py:770` | YES — `allow_guess=True`; §2.4b direction change | reads the return value |
| `_xshot_occurrence.py:744,863` (shim `:667`) | YES — two sites | aliased import |
| `features.py:74,138`, `tracking/__init__.py:166,418` | YES — re-export rename only | no behaviour |
| `scripts/validate_shot_goalmouth_sb.py:663` | YES — **ADR-037 artifact driver**; §5.4 | writes `--out` |
| `tests/scripts/test_validate_shot_goalmouth_sb_shards.py:130` | YES — **monkeypatch returns a bare dict**, breaks on the new type | |
| `tests/tracking/test_gk_resolve_goal_map.py` | YES — incl. `_defended_goal_x is defended_goal_x` | |
| `tests/tracking/_mirror_entries/pre_shot_gk.py:7` | NO — prose reference | no call |
| `tests/gkdv/test_import_allowlist.py:23` | NO — **not a caller**; `:23` is inside the comment block above `ALLOW_PRIVATE` (`:28`) and records the 4.53.0 promotion. Prose that goes stale → §5.5 | verified: the symbol appears nowhere else in the file |

Rev 2 reclassified the allowlist row as an affected caller. That was wrong; it is prose.

### 5.2 Callers of `_build_goal_map` — clause (b), wrapper callers

`_xcross_attempt.py:331,722` (internal), `causal/opportunities.py:39,251` (**builds causal
covariates** — same shape as the ADR-051 PR-5 finding), `tracking/_model_eval.py:129,134` (the
xS/xCross probe path → §5.3).

### 5.3 Second hop — research artifacts (clause e)

`_model_eval` feeds `scripts/validate_xs_probe.py` and `scripts/train_xcross_attempt.py`, which
write `docs/research/tf19_pr3b_xs_v2/` and the xCross weights `metrics.json`.

**Classified unaffected, conditionally:** with `allow_guess=True` the xCross map is byte-identical
to today on every shape except string-`is_ball` (§2.3), where today's map is *wrong*. **Required
check:** confirm the probe corpora contain no string-encoded ball rows and no NA-team groups. If
either exists, these artifacts are affected and must be re-run — as a follow-up on a merged commit,
per the header.

### 5.4 Committed fixtures and artifact drivers (clause c)

* `tests/causal/_fixtures.py:4` encodes `_build_goal_map`'s expected outcome **in a docstring** — it
  calls nothing, so no sweep surfaces it. Must be re-read against the new behaviour.
* `scripts/validate_shot_goalmouth_sb.py` is an ADR-037 driver: any re-run goes through
  `require_clean_tree` and records `run_commit` / `run_tree_dirty`.

### 5.5 Numbers and claims recorded in prose (clause d)

`CLAUDE.md`; `TODO.md:53-62` (the mischaracterised defect, §1.2); `docs/c4/architecture.dsl` (the
count); `gkdv/_engine.py:317` (the note asserting the miss branch is unreachable);
`tests/tracking/_mirror_entries/influence_family.py:43-50` (its *"GATE B FAILS AND IS NOT XFAILED
… the honest state is RED"* is already false — §1.8 — and becomes false a second way on merge);
`docs/superpowers/adrs/ADR-030-shot-goalmouth-trajectory-geometry.md:28` (*"Goal ends come from the
GK map (`defended_goal_x`…)"*); `ADR-043-tf19-gkdv-v1.md:326`; `CHANGELOG.md:1605` (the 4.53.0
public-export entry).

---

### 5.6 `visible_fraction` — the SECOND changed public function

Rev 2 swept only the goal-map functions. `visible_fraction` also changes (§2.5), and CLAUDE.md:146
requires every caller of **every** changed function — the sweep's own named failure mode recurring
inside the spec that quotes the rule.

| caller | affected? | evidence |
|---|---|---|
| `scripts/build_sb360_coverage.py:32,255` | **YES — and it poisons an artifact** | see below |
| `providers/statsbomb/__init__.py:19,30` | YES — public re-export | rename/semantics |

```python
# :255
bucket["sum_area"] += visible_fraction(ff.get("visible_area") or [])
# :279
"mean_visible_pitch_fraction": b["sum_area"] / n if n else float("nan"),
```

The `or []` passes `[]` for every action with no 360 record. Today that adds `0.0`; under §2.5 it
returns NaN, and `sum_area += NaN` poisons the **entire** per-(match, action_type) bucket. With
only 32.6% of goal kicks carrying a freeze-frame, that is most buckets — and the driver writes
`docs/research/sb360_coverage/`, the committed ADR-053 artifact.

**Fix, using the ADR-042 pattern §2.5 already invokes:** accumulate non-NaN only, carry
`n_with_polygon` as the denominator, and report it alongside the rate — the
`n_valued_disruptive_runs` shape, where a coverage denominator must never masquerade as a signal.

### 5.7 Callers of the cover-shadow trio

| caller | affected? | evidence |
|---|---|---|
| `_cover_shadows.py:1058` (`lane_control`), `:1097`, `:1115` (`compute_blocking_score`) | YES — internal, all three sites re-keyed | §1.9 |
| `features.py:3709`, `:3834` (`_compute_cover_shadow_dict`) | YES — the two `add_cover_shadows` / `cover_shadow_xfns` entry points | §1.9 |
| any external caller of the two public functions | **NO** — signatures unchanged (§1.9); values unchanged on canonical frames | verified: `frame` already carries `game_id`/`period_id` |

## 6. Three pre-stated measurements

Both rules are fixed **before** measuring, so an answer cannot be chosen after seeing it.

**M1 — NA-team GK rows in the `GhostGkModel` training corpus.** Those rows trained on features
computed in the mirrored goal-relative frame.
*Zero* → no retrain, record the count. *Non-zero* → retrain trigger, handled as its own weights
cycle (the code-then-weights split the repo already uses).

**M2 — GK-less `(game, period, team)` groups, per corpus, per provider.** This is the N1 rate
(§1.5) and it is nowhere measured; SkillCorner's 19.6% frame-level GK detection makes it a live
question even though a group needs only one keeper frame to resolve.
*Zero* → the `allow_guess=True` decisions in §2.2 are vacuous and can be revisited later.
*Non-zero* → they are load-bearing, and the guess rate is reported in `GkdvReport` and
`shot_crossing_source` so consumers can see it. **Either way the value behaviour is unchanged** —
M2 informs interpretation and future work, it does not gate this cycle.

**M3 — `(game, team)` whose GK-derived defended end DIFFERS between period 1 and period 2**, per
corpus, per provider. This tests §2.7's load-bearing invariant directly, rather than trusting
`play_left_to_right`'s docstring against a contradicting code comment (§2.7).
*Zero* → the canonical-orientation invariant holds and §2.7's "no value effect" is **proven**, not
assumed.
*Non-zero* → the three `features.py` sites and `_gk_influence.py:318` are producing wrong values
**today** for every second-half action on the affected providers, §2.7 becomes a value change, and
`add_gk_influence` consumers acquire a retrain question. Unlike M1/M2 this one **can gate the
cycle**, which is why it is stated before the plan rather than discovered during it.

---

## 7. Deliberately not in this cycle

* **Wiring coverage into the count features** — changes existing values, decides for the consumer.
* **The `GhostGkModel` retrain** — conditional on M1; a weights cycle if triggered.
* **The four unauditable SB360 boundary entry points** — three are a frame-pair shape mismatch
  needing a different harness; `xtgk.compute_xt_gk_v2` is blocked because silly-kicks ships no xG
  model.
* **Lakehouse adoption of `providers/statsbomb`** — not answerable from this repo (but see §2.5's
  note that `visible_area` already has a live dbt consumer there).

---

## 8. Open items

**Three measurements** (§6 M1–M3, each with its decision rule fixed in advance — M3 can gate the
cycle), one **required check** (§5.3) on the probe corpora, and one **coordination check**: §2.7
touches `features.py`, `_gk_influence.py` and `_cover_shadows.py`, which must be checked for
in-flight conflict with the other session's ADR-051 work before implementation begins. No open
*design* questions.

Rev 1 said "None", which contradicted its own §5.3 and §6 and read as closed — discouraging exactly
the review that surfaced S1.

---

## 9. Version and schema impact

* **Breaking:** `defended_goal_x` → `resolve_defended_goals`, returning `GoalMap`. Deleted, not
  shimmed (§2.1). CHANGELOG entry marked BREAKING; version bump per the five canonical sites.
* **Schema (Hyrum):** `ghost_gk_source` gains `goal_end_unresolved`; `add_visible_area_coverage`
  adds two columns.
* **`add_cover_shadows` / `lane_control` / `compute_blocking_score`:** values unchanged on
  canonical frames; the three `goal_x_own` bindings become `GoalMap.attacked_goal` lookups.
  **No signature change** — §1.9 verifies the frame already carries the map keys, so review 4's
  predicted third and fourth breaking API entries do not exist. **`add_cover_shadows`' `defect_b`
  IS removed** — measured `XPASS(strict)` once `:704`/`:1030` are included (§14).
* **NINE breaking public changes across TWO packages** (rev 10; rev 6 said two, rev 7 four, rev 8 six, rev 9 seven -- the count has risen at every revision, which is itself the finding):
  1. `defended_goal_x` → `resolve_defended_goals`, returning `GoalMap`
  2. `visible_fraction` → `observed_pitch_fraction`, clipped
  3. `lane_control`: `home_team_id` → `goal_map: GoalMap`
  4. `compute_blocking_score`: `home_team_id` → `goal_map: GoalMap`
  5. `add_gk_influence`: `home_team_id` **removed**; gains `goal_map: GoalMap | None = None`
  6. `add_cover_shadows`: `home_team_id` **removed**; gains `goal_map: GoalMap | None = None`
  7. `gk_influence_xfns`: `home_team_id` **removed** (`_get_gi` lives inside it, so after the
     re-key the parameter is required-and-unread — the dead-parameter shape this cycle deletes)

  8. `tracking.compute_threat_pc`: `home_team_id` → `goal_map: GoalMap`
  9. `gkdv.delta_threat_suppression`: `home_team_id` → `goal_map: GoalMap`

  The optional aggregator-level `goal_map` is not a convenience: **Gate C cannot exist without
  it**, because a gate that varies the map needs a seam to inject one through.

  **8 and 9 were found by the LINTER during implementation, not by the spec or by seven review
  rounds** (§17.1). They are in scope and they are correct: `home_team_id` in both was only ever
  passed through to derive direction, so keeping it would leave `gkdv` — the one package whose
  contract is *"depends on `silly_kicks.tracking` PUBLIC seams ONLY"* — re-deriving direction from
  team identity, i.e. preserving the D3 defect in the package least able to justify it. `GoalMap`
  is public, so the dependency stays honest.

  Plus the same replacement on three private functions (`compute_gk_influence`,
  `_voronoi_threat`, `_compute_cover_shadow_dict`). 5 and 6 exist because leaving a dead
  `home_team_id` on the two flagship aggregators is the `compute_space_created` pattern
  CLAUDE.md records as a defect class — and because a dead parameter makes Gate B vacuous
  rather than merely idle (§16).
* **`visible_fraction` changes value twice** (§2.5): NaN instead of `0.0` on a degenerate polygon,
  **and** clipped instead of unclipped on the common case. Both are corrections.
* **A committed research artifact changes.** `docs/research/sb360_coverage/`'s
  `mean_visible_pitch_fraction` moves in two directions — **up** because "unknown" stops being
  counted as zero, **down** because off-pitch camera area stops counting as observed pitch. The
  re-run is a **post-merge follow-up** (§header): a driver cannot stamp a `run_commit` that does
  not yet exist, so it can never be a second commit on this branch.
* **No lakehouse re-materialize** — no existing tracking column changes value on well-formed
  frames, and the string-`is_ball` shape is unreachable from any converter path (§2.3).

---

## 10. Review 1 response log

| # | finding | resolution |
|---|---|---|
| S1 | N1 fallback dropped; "no value effect" unevidenced | §2.2 per-consumer `allow_guess`; gkdv + `_shot_goalmouth` **True** (values preserved); §6 M2 |
| S2 | `_shot_goalmouth` changes refuse→resolve; row mis-describes the branch | **Verified.** §2.4b states it as an intended change with a characterization test |
| T1 | table incomplete; ball-string + `.astype(bool)` axes | **Verified.** §1.1 five axes; §2.3 per-axis choice; §4.1 fixtures 2–4 |
| T2 | seam must own the lookup | §2.1 `GoalMap` with `get`/`ends_in_period`; gkdv's local re-keying removed |
| T3 | a tuple is a weak guard | Accepted; the object replaces it |
| T4 | oracle has no capture point | §4.3 pre-change capture, committed RED |
| T5 | the behavioural delta is untested | §4.4 characterization matrix |
| T6 | public break, no versioning statement | §9; rename **without** shim, argument recorded in §2.1 |
| T7 | second implementation of observed-fraction | §2.5 one implementation, clipped ∈[0,1]; port's `0.0` → NaN |
| T8 | bbox over-reports for a triangle | §2.5 `(M,2)` polygon via Sutherland–Hodgman |
| T9 | `bool` cannot express unknown | §2.5 `bool \| None` |
| T10 | no pandas axis in CI | §2.6 contract test + cast, not a version test |
| M1–M4 | report field hygiene | Folded into §2.1's `GoalMap` (no constant `source` field; counts are properties; `frozenset`; typed mappings) |
| M5 | AST gate has no predicate | §4.2 predicate stated; plant must differ from both forks |
| M6 | mirror registry is blind to a 4th module | §4.6, in-cycle |
| M7 | 10× regression is avoidable | §3.2 vectorized → speed-up |
| M8 | ghost must pass `work`, not `frames` | §2.2 note: `resolve_defended_goals(work)` (post-subsample) |
| M9 | token unnamed; `serve_` has no row | §2.4a names `goal_end_unresolved` and splits the three entry points |
| M10 | `visible_area_source` vocabulary missing | §2.5 three tokens, each exported |
| M11 | prose sites missed | §5.5 adds ADR-030:28, ADR-043:326, CHANGELOG:1605 |
| M12 | §8 "None" contradicts §5.3/§6 | §8 rewritten |
| M13 | line nits | `_build_goal_map` **282** and D3 block **294-308** corrected. **`_snapshot.py` `pd.concat` is 172, not 171** — line 171 is the comment; rev 1 and `TODO.md` were right |
| N1 | squash escape hatch conflicts with ONE commit | Header: an artifact re-run is a post-merge follow-up, never a second commit |
| N2 | naming | Adopted: `resolve_defended_goals` |
| N3 | `visible_area` has a live lakehouse consumer | Recorded in §2.5 |

---

## 11. Review 2 response log

| # | finding | resolution |
|---|---|---|
| S3 | `visible_fraction` → NaN poisons a committed artifact; second changed public function unswept | **Verified** (`build_sb360_coverage.py:255,279`). §5.6 sweeps it; driver moves to the ADR-042 non-NaN-denominator pattern; §9 records the artifact change and schedules the post-merge re-run |
| T11 | the §4.2 predicate matched 1 of 3 implementations | **Verified by execution** — only `_ghost_gk.py:848`'s chained form matched. §4.2 now uses the IfExp signature: 8 sites, 5 modules, 0 false positives |
| T12 | population is 8 sites / 5 modules; three unlisted and identity-keyed | **Verified** — `features.py:3108,3337,3525`, zero `period_id` within 15 lines. §1.0 derives the population by execution; §1.6 documents them; §2.7 fixes them (owner: in scope) |
| M14 | §2.3 forward-references a measurement that does not exist | §2.3 now bounds the blast radius structurally instead: `schema.py:18-19` declares both columns `bool` and every converter casts, so the string shape is unreachable from a converter path |
| M15 | §4.4 cells say "fixed" without the outcome | §4.4 gains a `before` column; the string-`is_ball` before-state is `{}` (every non-empty string is truthy → `players` empty) |
| M16 | the M8 response never reached the body | §2.2 now states `resolve_defended_goals(work)` with the subsample reason |
| M17 | vectorization would freeze the all-NaN-`x` → `105.0` wart | **Verified** (`float(nan) < 52.5` is `False`). §2.3 routes it to `unresolved`; §2.4(c) states it; §3.2's byte-identity is scoped to the oracle's well-formed frames |
| M18 | `visible_fraction` semantics ambiguous | §2.5 decides: ONE quantity, clipped ∈[0,1], unclipped deleted. The polygon stays unclipped (ADR-054 D5 untouched) — clipping the *vertices* destroys information, clipping the *area calculation* does not |
| M19 | no `unlinked` token | Added; the aggregator takes `links`, so link failure must be representable |
| M20 | underscore fields; `frozen=True` ≠ immutable mapping | Non-underscore fields; `MappingProxyType`; `resolve_defended_goals` the only builder |
| M21 | `test_import_allowlist:23` is prose, not a caller | **Verified** (`:23` sits above `ALLOW_PRIVATE` at `:28`). Moved to §5.5; rev 2's reclassification was wrong |
| — | **§4.6 dropped** (not a review finding — found by running it) | The enumeration returns 14 keying modules, 11 outside the unit, most **correct**. `direction.py` keying to decide direction and `features.py` keying to decide `goal_x` are the identical AST — the ADR-043-unsolvable case. §4.2's precise signature plus behavioural Gate B cover this cycle; the measurement goes to `TODO.md` for ADR-051 |

**Review 2's summary was accurate:** *rev 1 measured the wrong case; rev 2 counted the wrong
population.* Rev 3's answer to both is the same — **derive the population by executing the
detector**. That method produced the three `features.py` sites (which no amount of reading had
surfaced across two reviews) and it also retired §4.6, whose own enumeration showed it could not
be made precise.

---

## 12. Review 3 response log

| # | finding | resolution |
|---|---|---|
| S4 | a ninth site; §4.2's signature is blind to it; the detector was built from the population already known | **Verified** — `_gk_influence.py:318` is an `ast.If`. §4.2's predicate is now SEMANTIC (any construct binding one name to `{0.0, pitch-length}`); re-run returns **12 sites / 7 modules**, four of which rev 3 could not see. Non-vacuity plant must be an `if`/`else` statement. Two dict-literal false positives found and the clause dropped (§1.0) |
| T13 | the defect is already recorded, already marked, and the registry edit is missing | **Verified, and the reviewer's own citation is stale**: `influence_family.py:43-50` says "FAILS AND IS NOT XFAILED", but `:82`'s `defect_b=` becomes `xfail(strict=True)` at `test_mirror_registry.py:224` — measured `182 passed, 8 xfailed, 0 failed`. This makes the point *stronger*: §4.5a's `defect_b` edit is mandatory and symmetric, not advisable. New §1.8; §1.6 corrected; §5.5 gains the file; §2.7's §4.6 citation fixed |
| T14 | §2.1 and §2.4(c) contradict on `unresolved`; the guess needs the same NaN guard | §2.1 now states the three states as an explicit exclusive ladder (resolved → guessed → unresolved) and applies the finiteness test to **both** mappings |
| T15 | "value effect: none" rests on an invariant the spec elsewhere quotes as false | **Resolved by reading the authority**: `play_left_to_right`'s docstring — *"the home team attacks left-to-right in every period"*. The `_ghost_gk.py:838-841` comment is wrong for oriented frames and is corrected in this cycle rather than cited. Backed by new **§6 M3**, the one measurement that can gate the cycle |
| T16 | `visible_fraction` keeps its name while changing meaning twice | Renamed to `observed_pitch_fraction`, old name deleted — §2.1's own principle applied consistently |
| T17 | clipping makes the crc-invariance witness vacuous | **Verified** (the fixture polygon is entirely interior). §2.5 extends it to a touchline-crossing polygon and re-states ADR-054 D5's reason as *alignment*, not area-invariance |
| M22 | `direction.py` exempt entry never matches | Dropped; the gate asserts **equality** per ADR-051's both-directions rule, so a never-matching exemption is correctly a failure |
| M23 | stale "six sites / six goldens" | §3.2 and §4.3 corrected; goldens are one per affected public surface, now including the gk_influence family and `add_cover_shadows` |
| M24 | "no `period_id` within 15 lines" is a weak proxy and borderline false | **Verified borderline-false** (`_get_gi` declares `period_id` at `:3504`, uses it at `:3510`). §1.6 now claims what is directly checkable: the `goal_x` **expression** carries no period term |
| M25 | §8 misses the in-flight conflict check | §8 now carries three measurements, one required check and one coordination check |
| M26 | §5.5 misses `influence_family.py:43-50` | Added, with both ways it becomes false |
| M27 | two censuses need reconciling; two different "eights" | §1.7 states both bases: §1.0 counts **expressions** (12), §1.8 counts **aggregators** (8); they overlap in exactly two members |

**Scope (owner, rev 4):** all 12 goal-end sites plus `_gk_influence.py:371`, the `visible_area`
seam, the `_snapshot` dtype pin and both renames ship in **one commit**. I recommended un-folding
the SB360 surface given the growth from 6 to 12 sites plus two breaking renames; that was
considered and overruled. Recorded, not re-argued.

**Review 3's summary was right, and its lesson is the one worth keeping:**

> Rev 1 measured the wrong case. Rev 2 counted the wrong population.
> Rev 3 executed a detector built from the population it already knew.

Rev 4's answer is not a better parser. It is that **a behavioural gate already held the census**,
and had held it for months — `add_gk_influence` was found by varying `home_team_id` and running the
aggregator, which no amount of parsing simulates. The AST gate ships as a tripwire against a
*future* fork; it was never going to be the census, and §4.6 reached the same conclusion from the
opposite direction.

---

## 13. Review 4 response log

| # | finding | resolution |
|---|---|---|
| S5(a) | the cover-shadow edit does not transfer — different subject team, opposite end; `GoalMap` has no accessor for it | **Accepted, and it is the real find.** New `GoalMap.attacked_goal(...)` (§2.1) — a real lookup of the opponent's entry, explicitly NOT `105.0 - get(...)`, which is wrong in the degenerate both-teams-one-end case and would be a second implementation inside the commit that deletes them. It also serves `_shot_goalmouth`, which does the workaround by hand today. New §1.9 |
| S5(b) | two public signatures must change — a third and fourth breaking API entry | **REFUTED, measured.** `frame` is a tracking-frame slice; `game_id`/`period_id` are in `TRACKING_FRAMES_COLUMNS`, and the call site builds it as `frame_groups.get_group((pid_period, fid))` (`features.py:3703`). The functions already read `is_ball`/`team_id` off that frame and read the keys the same way. **No signature change; §9 keeps exactly two breaking renames.** Recorded a pre-existing one-call-one-match assumption (the group key omits `game_id`) rather than introducing one |
| S5(c) | propagation gaps — §2.2, §2.7, §4.4, §5, §9 | All five closed: §2.2 policy row, §2.7 rewritten to 8 sites / 3 mechanisms, §4.4 column, new §5.7 sweep, §9 impact line |
| T18 | §4.6's subtraction is justified against the superseded predicate | Re-derived against the semantic predicate (12 sites / 7 modules / 0 FP), **and the honest framing added**: the enumeration WOULD have found `_cover_shadows`; the argument is that it also finds eleven correct keyings and costs eleven rotting exemptions, while Gate B already carries `add_cover_shadows` with a measurement |
| T19 | two stranded edits leaving adjacent contradictions | Both fixed. §2.7's Gate-B sentence is now a statement of what the two checks each prove (Gate B: direction is identity-inferred; M3: whether the data exercises it). §4.2's duplicate plant requirement collapsed to the `if`/`else` one |
| T20 | T14's ladder fixed in §2.1, not propagated | §2.3's axis row and §2.4(c) both rewritten to the ladder: an all-NaN-`x` group is **not `resolved`**, falls to `guessed`, and reaches `unresolved` only if outfield x is also non-finite |
| M28 | "a ninth fails CI" | → thirteenth |
| M29 | §2.7 heading/body pre-cover-shadows | → "eight sites, three mechanisms", with the mechanism table |
| M30 | the four new sites got a census line and no narrative | `_cover_shadows` ×3 now has §1.9 — and M30 names the mechanism by which S5 happened: **a census entry is a claim the design already covers it** |
| M31 | §9 missing cover-shadow lines | Added, including the explicit statement that no further breaking entries exist |

**Review 4's lesson, which is the one to keep:**

> The census and the design must be re-derived together. A census entry is a claim that the design
> already covers it.

Rev 4 grew the census from 8 to 12 and updated §1 and §4.2 behind it, leaving §2 / §4.4 / §5 / §9
describing an 8-site cycle. The three sites stranded that way were exactly the ones where the
design's central claim — *"the edit is the same one the other five get"* — was false. **Rule adopted
for rev 5: when a census line is added, its row in every downstream table is added in the same
edit, or the census line is not added.**

---

## 14. Executability pass — measurements against the tree

Run before plan-writing, on `12f77f9`. The proposed diff was built with
`inspect.getsource` + the exact one-line replacement and **executed** (per
`feedback_execute_the_proposed_diff`: never emulate by pre-transforming inputs — that is a
tautology). Artefacts: a `GoalMap` prototype plus a pytest plugin that patches the real modules at
`pytest_configure` and runs the **real** Gate B.

### 14.1 The four claims

| # | claim | result |
|---|---|---|
| 1 | `add_gk_influence` flips green | **YES** — `XPASS(strict)`, and the mandatory-`defect_b`-removal mechanism (§1.8) is now *observed*, not predicted |
| 2 | `attacked_goal` serves the cover-shadow sites | **Mechanically yes**; the three bindings alone move **zero** columns (§14.3) |
| 3 | `game_id`/`period_id` readable off `frame` | **YES** — every patch applied and ran; §1.9's refutation of review 4's S5(b) holds, no signature changes |
| 4 | vectorized seam byte-identical to the loop | **YES** on well-formed frames |

### 14.2 Four things the spec had wrong

1. **`attacked_goal` shipped with the hole it exists to close.** The specced guard was "not exactly
   one opponent"; in the degenerate case there IS exactly one, so it reported that a team attacks
   the goal it defends. Fixed with an explicit same-end check (§2.1).
2. **`acting_team_attacks_rtl` cannot re-key `_gk_influence.py:371`** — wrong shape entirely
   (per-action Series vs per-frame call). §2.7 corrected.
3. **`features._gk_influence_transformer` does not exist**; the container is `gk_influence_xfns`.
4. **`add_cover_shadows` DOES flip** — the opposite of rev 5's prediction (§4.5a).

### 14.3 The cover-shadow measurement

Gate B max delta on the canonical scene, per column:

```
patched                          n_potential  max_single  n_blocked  blocking   blocked_threat
                                  _receivers   _def_block  _receivers  _score     _fraction
--------------------------------------------------------------------------------------------
(baseline)                            10        2.02238        2       148.83      0.597651
3 goal-end bindings only              10        2.02238        2       148.83      0.597651
+ :704 / :1030 direction bools         0            0          0            0             0
```

Five of five to exactly zero, and the real gate reports `XPASS(strict)`. Hence the two direction
sites join the scope (§2.7) — without them the cycle fixes three bindings and changes nothing
observable.

### 14.4 Two measurement bugs of my own, recorded because the plan must avoid both

Each nearly produced a confident false conclusion, and neither was visible in the output.

1. **A diff-check that iterated only post-change keys.** Three columns went from non-zero to
   *absent* (delta zero) and the check reported "0 changed". Union the key sets — the same
   "assert the COUNT" lesson `feedback_execute_the_proposed_diff` already records.
2. **Patch ordering under a globals snapshot.** `_patch` uses `dict(vars(mod))`, so patching a
   **callee after its caller** leaves the caller closed over the old version. That presented as a
   third, independent, unidentified defect mechanism — a plausible and entirely fictitious
   finding. Callees must be patched first.

Both are properties of the *harness*, not the code under test, which is exactly why an
executability pass must sanity-check its own instrument before trusting a null result.

---

## 15. Second executability pass — the CONSUMERS

Review 5's summary named the gap precisely: *"§14 executed the diff; nobody executed the
consumers."* This pass closes it, and the result changed the design.

### 15.1 Why pass 1 could not have found this

`canonical_scene()` — the fixture Gate B and §14 both use — has **one period** and **exactly one GK
row per team per frame**. Per-frame and per-period construction coincide there by construction. Gate
B passing established identity-independence and **nothing whatever** about the estimator.

### 15.2 The measurement

Fixture: 2 periods (ends swapped), 60 frames each, keeper detected in 19.6% of frames (the
SkillCorner rate, ADR-038:123), keeper occasionally advanced past halfway.

```
period-scoped map (the contract)      : 4/4 correct
per-frame maps                        : 189/240 team-frames WRONG   (78.8%)
   of which came from the GUESS       : 188
attacked_goal returned None           : 82/240                      (34.2%)
```

**The compounding path, exactly as review 5 reasoned it.** A keeper absent from the frame drops its
team to `guessed` = outfield mean-x; §1.5 already establishes outfield mean-x points toward the goal
the team *attacks*, i.e. the opponent's end. `attacked_goal` then compares the opponent's correct end
against this team's wrongly-guessed (and now equal) end, its same-end guard fires, and it returns
`None` — which both direction sites treated as fail-open.

Per-frame construction would therefore have produced **wrong direction in ~79% of frames** on
broadcast-shaped data, silently, inside the two aggregators this cycle exists to correct. Worse than
the defect being fixed.

### 15.3 What it changed

* §2.8 — the map is built once from full frames and threaded; `goal_map: GoalMap` replaces
  `home_team_id` on five functions; no default.
* §2.9 — one uniform unresolvable-end policy, with fail-open explicitly forbidden.
* §1.9 — the S5(b) refutation corrected: right about the keys, wrong about the map.
* §9 — four breaking public changes, not two.

### 15.4 The method lesson

Pass 1 executed the **diff** and pass 2 executed the **consumers**, and only the second found this.
The generalization, which is stronger than "execute the diff":

> **Execute the diff on a fixture that can distinguish the new contract from the old one.** A
> fixture where both behave identically makes a green run evidence of nothing. `canonical_scene`
> was such a fixture for the estimator question, and it looked like a thorough test.

This is the same shape as the repo's `vx=vy=0` fixture defect and the frozen-fixture reflection
finding: the guard ran, and the thing it was guarding could not have shown up in it.

---

## 16. Gate C — replacing the census the re-key would otherwise delete

### 16.1 The problem review 6 found

Gate B holds the frames fixed and varies `home_team_id`. Once direction comes from the map,
`home_team_id` carries nothing — so the gate that varies it stops being a test:

* leave `home_team_id_role="direction_only"` → Gate B **passes vacuously**, and removing `defect_b`
  converts a meaningful RED into a green that is structurally incapable of failing;
* set `role="unused"` → Gate B **skips** (`test_mirror_registry.py:241-242`), which is honest and
  detects nothing.

Meanwhile the AST gate cannot see the three direction-bool sites at all — they bind a bool, not
`{0.0, pitch-length}` (§2.7). So the cycle would leave `_gk_influence.py:371`,
`_cover_shadows.py:704` and `:1030` **with no detector**, having just retired the one that found
them. §1.8's conclusion was *"Gate B is the census"*; this would delete the census for exactly the
two aggregators being re-keyed. Detection-before-the-fix, running in reverse.

### 16.2 Gate C

Same registry idiom, one variable further out: **hold the frames fixed and vary the GOAL MAP.**

1. **Non-vacuity** — call the aggregator with a map whose ends are swapped; the invariant columns
   **must move**. If they do not, the aggregator is not reading the map and the re-key is cosmetic.
2. Gate B's old invariance claim becomes *unrepresentable* rather than merely unasserted, because
   `home_team_id` is gone from the signature (§9 items 5-6) — which is the stronger form.

### 16.3 Proven able to fail, BEFORE being written into the plan

A gate that is the only detector for three sites must not be specced unrun. Measured on
`canonical_scene`, with the map injected and its ends swapped:

```
add_gk_influence    gk_pitch_control_share_weighted     0.108532   (1 invariant column moved)
add_cover_shadows   blocking_score                        148.83   (5 columns moved)
                    n_potential_receivers                     10
                    max_single_defender_blocking_score   2.02238
                    n_blocked_receivers                        2
                    blocked_threat_fraction             0.597651
```

**These are the same magnitudes the registry recorded as the D3 defect** — `influence_family.py`
logs `share +0.1085` and `blocking_score 148.83`. Gate C detects the identical signal through a
different variable, so the census transfers rather than degrading.

**Honest qualification:** `add_gk_influence` moved one column in this probe, not the two the
registry records (`share` **and** `closing_time_min +4.38 s`), because the probe injected the map at
the own-end and reflection sites but not into `_closing_time_per_series`, whose `home_team_id` was
held constant across both runs. With every site re-keyed, Gate C sees the closing-time columns too.
Stated rather than rounded up, because a gate's claimed reach is exactly the kind of thing this
cycle has repeatedly found overstated.

---

## 17. Implementation findings

Recorded during execution, because each is the same shape the review cycle kept surfacing and
none was reachable by reading.

### 17.1 The signature change cascades into a SECOND package

The spec enumerated four cover-shadow functions carrying `home_team_id`. There are **five**:
`compute_threat_pc` also passes it through to `_voronoi_threat`, is **public**, and is consumed by
`gkdv.delta_threat_suppression` via `**kwargs` — which is public too, takes two single frames, and
has no `frames` from which to build a map. That pulls in `scripts/build_gkdv_arm_values.py`.

Found by **ruff `F821`**, after the signature change made the pass-through an undefined name. Not
by the caller sweep, not by seven reviews. The sweep looked for `home_team_id=home_team_id,` **with
a trailing comma**; this one is the last kwarg before a closing paren.

The lesson is narrower and more useful than "sweep harder": a textual sweep for a keyword argument
is sensitive to punctuation the AST does not care about. The type checker and the linter see the
call graph; the grep sees a string.

### 17.2 Both new population-gate guards fired on FIRST use

Review 6 asked for two one-line hardenings on the AST gate (`assert SEAM in found`,
`assert len(found[SEAM]) == 1`). Both caught a real defect immediately:

* the seam named its pitch constant `far`, so the detector could not see the seam's own
  derivation — **the gate would have passed with zero sites detected anywhere**;
* the seam then derived the end **twice** (the `resolved` and `guessed` branches), which is not
  "one implementation". Collapsed into `_end_from_mean_x`, which is better code independently.

A gate asserting only "nothing outside the seam" is satisfied by a scanner that finds nothing.

### 17.3 A prescribed `continue` that could not compile

The plan specified `continue` for all three `features.py` sites. The third is inside `_get_gi`,
which **returns** a cached value rather than looping — `SyntaxError: 'continue' not properly in
loop`. The correct form was `cache[key] = None; return None`, which is how that function already
reports "nothing here".

Three reviews read that step and none could have caught it: the enclosing construct is not visible
in the quoted snippet.
