# Vector-quantity consistency under coordinate reflection — design

**Date:** 2026-07-19
**Target:** silly-kicks 4.55.0 / PR-S122 / ADR-045
**Status:** design, awaiting review

---

## 1. Problem

silly-kicks reflects coordinates in several places. ADR-028 establishes that SPADL actions
(per-acting-team LTR) and tracking frames (home-attacks-right) differ by a **180° point
reflection** — `x -> 105-x` **and** `y -> 68-y`.

A point reflection acts differently on different kinds of quantity:

| kind | example columns | transform |
|---|---|---|
| point | `x`, `y`, `start_x`, `x_smoothed` | `x -> L-x`, `y -> W-y` |
| vector | `vx`, `vy` | negated |
| magnitude | `speed`, distances, areas, counts | unchanged |
| direction label | `team_attacking_direction` | `ltr <-> rtl` |

Every reflection helper in the codebase transforms an **explicitly enumerated column list**.
Any column not on that list rides through untransformed, silently. There is no mechanism that
notices.

An audit (6 independent search lenses, 96 sites examined, 18 adversarially-verified verdicts)
found the pattern instantiated **six** times. Two further sites — `vaep/features/core.py:134`
and `atomic/vaep/features.py:119`, both named `play_left_to_right` — were found in cross-session
review. Two more — `spadl/orientation.py::_mirror_absolute_frame` and `::_mirror_per_period`
(§2 D8) — were found in a fifth review pass. §4.6 gives the precise breakdown: **eleven places
apply a reflection, and two more are defective by omission.** Do not carry a single headline
number; the earlier "eight" did not survive contact with its own table.

**The inventory has been wrong three times, and ADR-045 must not assert exhaustiveness.** All
five `play_left_to_right` definitions appear in a single `grep -rn "def play_left_to_right"`;
three were carried into this design and two were dropped between running that grep and writing
§2. The D8 pair was then missed by *that* correction, even though it is reached by nine
converters and the repo's own test suite already names it as part of the same family
(`tests/invariants/test_play_left_to_right_id_dtype.py`: *"the `play_left_to_right` /
`to_spadl_ltr` family (ADR-019)"*).

Neither miss was a search-coverage failure — the evidence was in hand both times. The operative
lesson is not "search harder": it is that **a count is a claim, and this document has no
mechanism that can establish it.** Every inventory number below should be read as "sites known
as of the fifth pass", and ADR-045 must state the count that way, with the pasted `grep` and the
command that produced it, so the next reader can re-derive rather than trust.

### 1.1 Why it was invisible

`vx` / `vy` are **not in `TRACKING_FRAMES_COLUMNS`** (`silly_kicks/tracking/schema.py:9-32`).
They are added later by `preprocess.derive_velocities`, as are `x_smoothed` / `y_smoothed`
(added by `preprocess.smooth_frames`). An author working from the canonical schema handles
`x`/`y`/`z` and stops — correctly, as far as the schema shows them.

`speed` is a magnitude and is correctly left alone by every site. That is exactly why no
sanity check ever fired: the corruption is invisible to any speed-based or distance-based
check.

---

## 2. Confirmed findings

Anchors below were verified line-by-line against the working tree on 2026-07-19, not taken
from audit output.

### D1 — LIVE. `bekkers_pi` mixes conventions inside one TTI call

`silly_kicks/tracking/utils.py:874` re-projects the context rows:

```python
return reproject_to_action_ltr(rows, row_flip, x_cols=["x"], y_cols=["y"])
```

applied to `actor_rows` / `opposite` / `defending_gk_rows`, all of which carry `vx`/`vy`.
The helper (`_action_orientation.py:196-203`) writes only `x_cols`/`y_cols` — it has no vector
parameter to pass. `_kernels.py:641-648` then reads re-projected positions against
frame-convention velocities, and `_bekkers_tti` uses velocity **directionally**: `d2 = p2 + v2`
(`:546`), `r_reaction = p1 + v1*reaction_time` (`:556`), and a dot product at `:551`. Away
defenders are modelled running backwards.

**Measured** (clean IDSSE match, 1363 actions / 3.36M frames, velocity defect isolated):
away-acting rows only, 97.6% changed; mean `pressure_on_actor__bekkers_pi` 0.2554 shipped vs
0.4181 correct (**-38.9%**); median |error| 0.333, max 0.996 on a [0,1] metric. Home rows
bit-identical (`flip=False` returns early at `_action_orientation.py:194-195`).

Independent corroboration: as-shipped, away pressure sits ~40% below home (0.2897 vs 0.4848) —
a home/away asymmetry with no football justification, which the fix removes (0.4325 vs 0.4848).

### D2 — LIVE. The ball row is never re-projected at all

`silly_kicks/tracking/features.py:921-923`:

```python
ball_rows = frames.loc[frames["is_ball"], ["period_id", "frame_id", "x", "y", "vx", "vy"]]
merged = pointers_with_period.merge(ball_rows, on=["period_id", "frame_id"], how="left")
return merged[["action_id", "x", "y", "vx", "vy"]]
```

`ActionFrameContext` carries no ball rows, so `_reproject_rows` (`utils.py:868-878`) never sees
them. Consumed at `_kernels.py:677-689` against action-LTR `defender_pos`. Gated on
`BekkersParams.use_ball_carrier_max`, which **defaults True**.
`silly_kicks/atomic/tracking/features.py:993` imports the same helper and inherits this verbatim.

This is a **position** defect, not a velocity one — the ball is un-reflected in `x`/`y` too.

**Measured:** actor-to-ball distance at the same linked frame — home/ltr median 6.13 m,
away/rtl median **62.13 m** (= `2*(52.5-x)`, the un-reflected mirror), max 119.41 m.
Ball-position fix alone changes 80.0% of away rows, max |delta| 0.9248.

**Severity shape differs from D1 and this matters for the guard design.** D2's *mean* bias is
only -1.1% (39.3% of rows overstate, 40.7% understate — they nearly cancel), while mean-absolute
error is 0.0657, p99 0.8041, and Spearman rank agreement with the correct value degrades to
0.858. **An aggregate or mean-comparison check passes cleanly on the broken code.**

### D3 / D3b — LATENT everywhere known. Divergent public contract, not a live miscomputation

`silly_kicks/tracking/utils.py:174-176`:

```python
period_flip = out["period_id"].isin(rtl_periods).to_numpy()
out.loc[period_flip, "x"] = 105.0 - out.loc[period_flip, "x"]
out.loc[period_flip, "y"] = 68.0 - out.loc[period_flip, "y"]
```

No `vx`/`vy` negation; no `x_smoothed`/`y_smoothed` reflection. The repo's own sibling
implements the correct contract at `direction.py:284-289`, guarded on `has_vx`/`has_vy`.
**Two public orienters, same declared transform, divergent semantics.**

`play_left_to_right`'s docstring (`utils.py:118-120`) claims "ALL rows in that period ... are
mirrored", so D3b is additionally a documented-contract violation.

**Reachability — resolved empirically, and it contradicts the majority of audit lenses.**
Several verifiers asserted `vaep/base.py:196-199` is a live path (it calls `play_left_to_right`
on caller-supplied frames, and `frames_convention` defaults to `"absolute_frame"` at `:125`).
It is not, because no library producer ever labels home `"rtl"`: measured on the real IDSSE
cohort, `team_attacking_direction` is `ltr` for home in **both** periods (777,788 and 827,849
player rows), so `rtl_periods` is empty and the function is a **measured no-op**. The
real-data pressure deltas attributed to D3 came from a hand-built rtl-labelled input; they
demonstrate the mechanism and magnitude, not an observed production feed.

`orient_frames_to_ltr` (`utils.py:295` -> `play_left_to_right`) *does* compute real per-period
directions and genuinely produces home-`"rtl"`, so it can reach the flip. That is the ADR-029
entry point documented for consumers building frames from non-kloppy bronze.

**The one consumer this design named as the live case does not traverse the path.** Verified from
the lakehouse side during cross-session review: per lakehouse ADR-053 (as amended by ADR-034
TF-23 / ADR-035 TF-23b), the action-context adapters call the silly-kicks
`tracking.{skillcorner,metrica}.convert_to_frames` builders and deliberately omit the orientation
flags, so `orient_frames_to_ltr_by_geometry` — **not** `orient_frames_to_ltr` — is the
orientation authority; the lakehouse-side LTR net was deleted in TF-23 precisely to make that the
single path. The geometric orienter already negates `vx`/`vy` (`direction.py:284-289`). Velocity
is derived through the builders' `preprocess=` seam, inside the same call.

So D3 is **LATENT everywhere currently known**, and the earlier "LIVE for external consumers"
label — which rested on an assumption this design explicitly flagged as unverified (§6) — is
withdrawn. It remains worth fixing: two public orienters with divergent vector semantics is a
real defect, `play_left_to_right`'s docstring actively claims the behaviour it does not implement,
and the divergence guard (§4.7.3) is the durable answer. But ADR-045 must not describe it as a
live miscomputation.

**Mechanism confirmed exactly:** real IDSSE round-trip — positions recovered to 7.1e-15,
`speed` to exactly 0, `vx`/`vy` recovered as the exact negation (`max|vx_out + vx_truth| =
0.000e+00`), median bearing error 180.00° over 1,115,244 moving rows.

D3b has a second-order finding worth recording: `smooth_frames`'s idempotence short-circuit
(`preprocess/_smoothing.py:100-103`) keys on the config tag, so the natural mitigation —
re-running preprocess after orienting — **silently does nothing** (measured: "re-smooth CHANGED
x_smoothed: False", residual 35.0 m).

### D4 — GUARD GAP. `finalize_orientation` flag-flip leg

`direction.py:359-360` reflects positions only, 70 lines above a callee at `:284-289` that
negates vectors. Unreachable today — not because velocities are absent, but because the
**schema projection** at `sportec.py:156` / `gradientsports.py:147` drops any caller-supplied
`vx` (measured: `'vx' in adapter output: False` even when force-injected).

**This finding constrains the fix shape.** The two legs *compose*: on a wrong-flag match (the
ADR-035 scenario, GS WC2022 ET matches 10506/10517) the flag leg flips positions and the
geometric backstop flips them back — net identity — while negating `vx`/`vy` **once**. Adding
negation to only one leg would be wrong in the other direction.

**Do not write "both or neither" — the fifth review pass measured it false.** "Neither" is not
a safe resting state: it leaves an 8 m/s-scale kinematic inconsistency between positions and
velocities on the composed path. Only **both** is zero-error across all four cases
(correct-flag / wrong-flag × geometric-backstop-fires / does-not). The implementation the plan
specifies is right; the slogan justifying it was not, and ADR-045 must not repeat it.

### D5 — LATENT, no consumer, and **needs a decision before it is called a defect**

`_shape_graph.py:919-921` applies `x = -x` for `attacking_direction < 0` (and the same at
`:931-932` for `face_centers_x`), leaving `y` untouched. 3,304 rows differ on real data against
a point-reflected ground truth.

**Do not treat this as settled.** The negation here is `-x`, *not* `105 - x` — it is a
sort-direction trick that reverses level ordering, not a transform into a canonical frame. So
the real question is whether the lateral L/LC/C/RC/R label is meant to be **team-relative**
(the rtl team's "left" is the opposite touchline, so y must mirror) or **pitch-absolute** (y
must not). Only one lens raised it, and it has no consumer either way.

**Required before implementation:** read the TF-39 intent (ADR / `NOTICE` attribution to
Sotudeh 2026) and decide. If team-relative, mirror `y` alongside the `x` negation and add a
guard. If pitch-absolute, the code is correct and this becomes a docstring clarification. This
is the one item in this spec that is a genuine open question rather than a known fix.

### D7 — LATENT. The two VAEP `play_left_to_right` helpers (found in review, not by the audit)

`vaep/features/core.py:189-193` enumerates the same four canonical coordinates as its SPADL
sibling, so ADR-025's `enriched_*` columns ride through unmirrored — identical latent trap to
D3's SPADL analogue. `atomic/vaep/features.py:165-169` is a **fourth correct** copy of the
`dx`/`dy` contract (alongside `direction.py:284-289` and `atomic/spadl/utils.py:1129-1133`),
which strengthens §3's argument rather than weakening it.

Both are **public exported API** (`__all__` at `core.py:35`, `atomic/vaep/features.py:39`, and
`vaep/features/__init__.py:53`) and test-exercised, but neither is called by `compute_features`
any more — ADR-006 / PR-S22 removed that (`vaep/base.py:191`). So they are public-surface latent,
not live.

**They are not drop-in migrations, and this is a real design constraint the rest of §4 does not
face.** Both mutate the caller's frames in place (`actions.loc[away_idx, col] = ...`) and return
the same `gamestates` objects; `reflect()` is pure by contract (ADR-033). A naive swap converts
in-place to pure and silently breaks any caller relying on the mutation. Resolution: compute the
reflection purely, then **assign the resulting columns back** into the caller's frame, preserving
the existing in-place contract while single-sourcing the transform. The migration is therefore
about where the *contract* lives, not about changing what these functions do — and the ADR must
say so, because "we migrated it" would otherwise imply a behaviour change that must not happen.

### D8 — LATENT. The SPADL orienter, missed by the audit *and* by its correction

`silly_kicks/spadl/orientation.py` carries two more hand-enumerated point reflections:

```python
# :222-225, _mirror_absolute_frame
    for col in ("start_x", "end_x"):
        out.loc[away_idx, col] = spadlconfig.field_length - out.loc[away_idx, col].to_numpy()
    for col in ("start_y", "end_y"):
        out.loc[away_idx, col] = spadlconfig.field_width - out.loc[away_idx, col].to_numpy()
```

and byte-analogously at `:272-275` (`_mirror_per_period`, mask `mirror_idx`). Both enumerate
exactly the four canonical coordinates, so ADR-025's `enriched_*` ride through unmirrored —
**the identical trap §4.1 cites as the reason to migrate `spadl/utils.py:1492`.**

Reached through the public `to_spadl_ltr`, which nine converters call (`gradientsports.py:725`,
`kloppy.py:242`, `metrica.py:275`, `opta.py:209`, `skillcorner.py:548`, `sportec.py:650`,
`statsbomb.py:286`, `wyscout.py:321`). That makes it **more reachable than the site this design
already migrates**.

Latent, not live: at the converter seam the frame carries only the canonical coordinates, so
there is no wrong number today. It becomes live for any caller passing enrichment-bearing or
vector-bearing actions through public `to_spadl_ltr`.

**This site is why §4.5 changed.** `to_spadl_ltr` is called *inside* the converter, on a frame
that already carries the caller's `preserve_native` passthrough columns, and its signature has
no `extra_kinds`-shaped parameter. A fail-closed `reflect()` here is not inconvenient — it is
**unimplementable**, because the only person who could satisfy it (the caller of
`convert_to_actions`) has no reachable escape hatch, and giving them one means threading a
reflection-implementation detail through nine public converter signatures.

### D6 — DOCUMENTATION. ADR-042 claim is false

`docs/superpowers/adrs/ADR-042-tf35-off-ball-run-valuation.md:92-94` (mirrored in the CLAUDE.md
ADR-042 bullet) asserts TF-4 "was the last module in the ACTION-COUPLED GEOMETRY layer keyed on
home/away identity". Two action-coupled aggregators still key on identity. Zero values change;
the cost is documentation trust.

---

## 3. Root cause

**The reflection API requires callers to enumerate columns, and silently does the wrong thing
when they miss one.** The API invites the bug.

**This was never missing physics — it was a missing shared seam.** The codebase already
implements the correct contract in *four* places, independently: `direction.py:284-289` negates
`vx`/`vy` guarded on presence; `atomic/spadl/utils.py:1129-1133` and
`atomic/vaep/features.py:165-169` each negate `dx`/`dy` while point-reflecting `x`/`y`; and the
geometric orienter does the same. Four authors got it right; the sites that got it wrong were
not less careful, they were working from a different hand-written enumeration. Eleven hand-maintained
copies of one rule will disagree eventually — which is the argument for a registry rather than
for twelve more careful edits.

Note the three defects differ in *kind*: D1/D3 are vectors, D2/D3b are points, D5 is a
categorical label. **"Negate the velocities" would fix roughly half and leave the trap fully
armed.** Any fix scoped to velocity is a patch, not a solution.

---

## 4. Design

### 4.1 Scope

Tracking frames **and** SPADL actions. Not the numpy grid reflections (ADR-041's
`[::-1, ::-1]` EPV/threat grids) — an ndarray is a different kind of object from a labelled
table, and forcing both through one abstraction would be artificial coupling. The grids get
their own narrow guard (§4.6), not this registry.

The actions side carries the identical latent trap: `spadl/utils.py:1492` mirrors exactly
`start_x`/`start_y`/`end_x`/`end_y`, so ADR-025's `enriched_start_x`/`_y`,
`enriched_end_x`/`_y` would ride through unmirrored today.

### 4.2 New module: `silly_kicks/reflection.py` (public, top-level)

Placement follows the ADR-043 precedent that promoted `_id_compat` to public
`silly_kicks.id_compat`: multiple packages need it, and a seam that is **mandatory** is public
API by definition. Same reasoning applies here — `spadl/`, `tracking/`, `atomic/` and `vaep/`
all reflect coordinates.

### 4.3 Column kinds

```python
ReflectionKind = Literal[
    "point_x",         # x -> L - x
    "point_y",         # y -> W - y
    "vector_x",        # vx -> -vx
    "vector_y",        # vy -> -vy
    "magnitude",       # unchanged (speed, distance, area, count)
    "direction_label", # "ltr" <-> "rtl"
    "invariant",       # ids, timestamps, categoricals, z
]
```

`z` is `invariant`: a 180° reflection in the pitch plane does not touch height.

**No kind expresses θ → θ+π, and that is deliberate.** A bearing is neither invariant, nor a
negation, nor a point reflection. Exactly one such column exists in the repo
(`pre_shot_gk_angle_off_goal_line`), it is goal-referenced, and it is unreachable from any
reflection site — it is an emitted feature, never an input to one. Adding an `angle` kind now
would be speculative API for a case with no instance. If one ever reaches a `reflect()` call it
surfaces as an undeclared column and warns (§4.5), which is the right amount of loudness for a
hypothetical. Worth one sentence in ADR-045; not worth a kind.

### 4.4 Two registries

`TRACKING_REFLECTION_KINDS` and `SPADL_REFLECTION_KINDS`, kept separate because the column
vocabularies are disjoint and a shared dict could silently mis-apply one table's semantics to
the other.

Each registry enumerates **every** known column, including the invariant ones. This is
tractable because they are all ours: `TRACKING_FRAMES_COLUMNS` + the provider variants +
preprocess-added (`vx`, `vy`, `x_smoothed`, `y_smoothed`, `_preprocessed_with`) on the tracking
side; `SPADL_COLUMNS` + provider variants + ADR-025 enrichment columns on the actions side.

### 4.5 One primitive

```python
def reflect(
    df: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    *,
    kinds: Mapping[str, ReflectionKind],
    extra_kinds: Mapping[str, ReflectionKind] | None = None,
    on_unknown: Literal["warn", "raise", "ignore"] = "warn",
) -> pd.DataFrame:
```

Pure — returns a new frame, never mutates (ADR-033). Applies each column's declared transform
on the masked rows.

**A second entry point is required, and the distinction is deliberate.** Not every reflected
table has a declared schema: `_kernels.py:879` re-projects `defensive_line_x` /
`back_line_high_x`, which are *computed feature outputs*, not schema columns, and enumerating
every geometry column emitted across ~30 aggregators would be a registry that rots faster than
it helps. Those callers get:

```python
def reflect_columns(
    df: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    *,
    point_x: Sequence[str] = (),
    point_y: Sequence[str] = (),
    vector_x: Sequence[str] = (),
    vector_y: Sequence[str] = (),
    direction_label: Sequence[str] = (),
) -> pd.DataFrame:
```

Explicit like today's API, but **kind-aware**: the caller must say what each column *is*. The
present `reproject_to_action_ltr(df, mask, x_cols=, y_cols=)` cannot express "this column is a
vector" at all — there is no parameter for it — which is why D1 could not have been fixed at
the call site even by an attentive author. `reflect_columns` closes that gap without demanding
a registry entry for every derived column.

Rule of thumb, stated so reviewers can apply it: **schema-bearing table -> `reflect()`
(registry-driven); derived or pre-canonical columns -> `reflect_columns()` (explicit,
kind-aware).**

This is a rule about *which mechanism knows the kinds*, not about *where it is safe to raise*.
An earlier draft framed it the second way — "fail-closed is safe exactly where the input is
schema-projected" — and that framing did not survive measurement: `spadl/utils.py:1543` is
`ltr_actions = actions.copy()` with no projection, `orient_frames_to_ltr` (`utils.py:295`) is
likewise un-projected, and neither exposes an `extra_kinds` parameter through which a caller
could satisfy a raise. Only the three tracking adapter call sites were ever post-projection.

**Fail-closed lives in the CI meta-assertion, not in the runtime call.** This reverses an
earlier draft of this design, which made `on_unknown="raise"` the default at every migrated
site. Three findings killed that, in ascending order of force.

**1. It is unimplementable at D8.** `to_spadl_ltr` is called *inside* nine converters, on a
frame already carrying the caller's `preserve_native` passthrough columns. `preserve_native`
(`spadl/utils.py:1651`) takes a caller-supplied `list[str]` validated only for presence and
non-collision with the schema — so preserved columns are, by construction, exactly the columns
no registry can enumerate. The string `preserve_native` appeared **zero times** in the first
draft of this design or its plan. There is no reachable `extra_kinds` at that seam, so `raise`
there is not a strict-but-workable choice; it is a break with no remedy.

**2. It would not have caught a single defect in this document's own inventory.** Every
column in D1–D8 is library-owned: `vx`/`vy` (D1, D3, D4), the ball's `x`/`y`/`vx`/`vy` (D2),
`x_smoothed`/`y_smoothed` (D3b), the shape-graph arrays (D5), `enriched_*` (D7, D8). **Zero of
eight** involved a caller-attached column. The registry-completeness meta-assertion catches all
eight; runtime `raise` adds nothing the meta-assertion does not already cover, while breaking
the one case it does reach. The justification "loud-on-upgrade is a one-time cost, silent-wrong
is permanent" is sound for library-owned columns and does not transfer to passengers — for
`possession`, silent-wrong is not the alternative. **Invariant is the correct answer**, and the
design would have been raising on the case it gets right.

**3. Two defaults would recreate D3 one layer up.** D3 is *"two public orienters, same declared
transform, divergent semantics."* Shipping `raise` on `tracking.play_left_to_right` and
`ignore` on `spadl.play_left_to_right` reproduces exactly that shape at the policy layer, in a
family the test suite already treats as one. **One default, everywhere.**

**The design:** `on_unknown="warn"` is the default at every site. An undeclared column is
treated as `invariant` — the correct treatment for a passenger — and warns **only if its name
matches `GEOMETRIC_NAME`**, in a dedicated public category:

```python
class UndeclaredGeometricColumnWarning(UserWarning): ...
```

So `possession` is silent (no warning spam on a supported first-party feature) while
`my_shot_x` is loud. `on_unknown` retains `"raise"` and `"ignore"` for callers who want an
explicit, greppable per-call choice; nothing in silly-kicks passes either.

**Strictness becomes a consumer decision, which is where it belongs.** A consumer that fully
controls its own frames — the lakehouse does — gets hard fail-closed with:

```python
warnings.filterwarnings("error", category=UndeclaredGeometricColumnWarning)
```

An external caller gets a warning and a migration path instead of a broken upgrade. This
follows the ADR-041 precedent (`tracking/_warnings.py`), where three categories are kept
deliberately separate *"so silencing the routine notice doesn't silence genuine misuse."* The
library cannot know whether a given caller's column universe is bounded; the caller can.

**Why this does not relitigate ADR-043.** ADR-043 deleted the id-dtype AST lint for two
reasons, and only the weaker one transfers. The unfixable objection — *"a safe and an unsafe
compare are the IDENTICAL AST; only provenance separates them"* — does not apply: whether
`enriched_start_x` is geometric is a property of the column itself, with no safe-versus-unsafe
variant of the same name. What does transfer is that name heuristics have false negatives.
That is survivable **because the heuristic never decides anything**:

- library-owned columns → registry, complete by enumeration, CI-gated. The heuristic never sees them.
- passenger, non-geometric name → silent `invariant`. Correct.
- passenger, geometric-looking name → warn.

The registry decides; the heuristic reports. A heuristic that raises is a liability; one that
reports is a smoke detector. Its measured blind spots (`team_shape_centroid_x_attacking`,
`defending_centroid_vx`) are derived feature columns that flow through `reflect_columns`, not
`reflect`, so they were never in its domain.

**The residual hole, stated rather than papered over:** a third-party caller who attaches a
geometric column whose name the pattern misses gets silent `invariant` treatment. That is the
honest price of `to_spadl_ltr` having no reachable escape hatch, and ADR-045 must name it as a
scope limit rather than claim a guarantee the code cannot keep.

Filtering the frame to its declared subset and calling `reflect()` on that was considered and
**rejected**: it looks fail-closed but silently skips an undeclared *geometric* column, which is
the original defect shape wearing the fix's clothes.

**Consequence: registry completeness is a CI contract, gated by the §4.7.2 meta-assertion.** A
new column added to any schema constant fails CI until declared — which fires when the column
is *added*, not on the first production run that happens to route it through a reflection site.
The registries still enumerate the measured column universe, not an assumed one:

- **Tracking: 25 columns** — the 20 in `TRACKING_FRAMES_COLUMNS` plus the five preprocess-added
  (`vx`, `vy`, `x_smoothed`, `y_smoothed`, `_preprocessed_with`). Verified against real IDSSE
  frames.
- **SPADL: 32 columns** — the 14 canonical, plus the 7 provider-variant columns across the four
  `*_SPADL_COLUMNS` dicts (`action_provenance`, `is_synthetic`, `result_source`, and the four
  `tackle_*`), plus the 3 `add_names` columns (`type_name`, `result_name`, `bodypart_name`),
  plus the 8 ADR-025 enrichment columns. Computed by union over the real schema module.

**`extra_kinds` is ADD-ONLY.** It declares columns the registry does not know; it may **not**
override an existing declaration. A key collision raises. Overriding was never intended, and an
early draft leaned on it at two call sites to force `team_attacking_direction` to `"invariant"` —
which would have left `direction_label` the one kind no production path exercised. Both uses were
removable and were removed (see §4.6).

### 4.6 Call-site migration

| site | change |
|---|---|
| `_action_orientation.reproject_to_action_ltr` | reimplement over `reflect_columns()`; keep the name and add `vector_x`/`vector_y` passthrough |
| `tracking/utils._resolve_action_frame_context:874` | pass the frame rows' vector columns — **this is the D1 fix** |
| `tracking/utils.play_left_to_right` `:174-176` | route through `reflect()` |
| `tracking/direction.py` `:284-289` **and** `:359-360` | route both legs (§2 D4 — both or neither) |
| `tracking/features._build_ball_xy_v_per_action` | re-project in place via `reflect_columns` (position **and** velocity), reading the flip from `ctx`. Note this is a **position** fix first — the ball is un-reflected in `x`/`y`, not merely in velocity. |
| `tracking/feature_framework.ActionFrameContext` | add a `flip_by_action: pd.Series` field. `_resolve_action_frame_context` already computes it (`utils.py:861-866`, dedupe included); the ball builder must **read** it, not recompute `acting_team_attacks_rtl` independently. Two computations of the same orientation decision in two modules is the exact producer/consumer drift this design exists to remove — and it would add a second full-table groupby per call. Threading the ball *rows* through the context was considered and rejected (YAGNI); threading the *flip* is one field on a dataclass that already reaches this helper, and it makes "ball and players used the same flip" structurally true rather than coincidentally true. |
| `tracking/_shape_graph.py:919-921` | **decide first** (§2 D5): mirror `y` iff the lateral label is team-relative; otherwise docstring only |
| `spadl/utils.play_left_to_right:1492` | route through `reflect()` with the SPADL registry |
| `atomic/spadl/utils.play_left_to_right:1079` | route through `reflect()` with the atomic registry. **Already correct** (`:1129-1133` negates `dx`/`dy`) — migrated, not fixed; any value change means the migration is wrong |
| `vaep/features/core.play_left_to_right:134` | route through `reflect()`, **assigning results back** to preserve the in-place contract (§2 D7). Only non-inert columns are written, so untouched integer columns are not upcast |
| `atomic/vaep/features.play_left_to_right:119` | same, with the atomic registry. **Already correct** on `dx`/`dy` — the 4th independent copy |
| `spadl/orientation._mirror_absolute_frame:222-225` | route through `reflect()` with the SPADL registry (§2 D8). Preserve the documented NA-as-away semantics at `:210-218` — `ids_match` resolves NA to False and `~` sends it away; do **not** add `.notna()` |
| `spadl/orientation._mirror_per_period:272-275` | same, on the `mirror_idx` mask. The two must keep identical NA semantics — `:218` warns explicitly against splitting them |

**Counting, stated precisely, because the earlier "eight sites total" was asserted over an
eleven-row table and did not survive its own paragraph.** Two different things are being
counted and they must not be conflated:

**Eleven places apply a reflection** (they contain the arithmetic, or own it):

1. `_action_orientation.reproject_to_action_ltr` — the shared primitive
2. `tracking/utils.play_left_to_right:175-176`
3. `tracking/direction.py:284-289` — geometric leg
4. `tracking/direction.py:359-360` — flag leg
5. `spadl/utils.play_left_to_right:1545-1548`
6. `atomic/spadl/utils.play_left_to_right:1129-1133`
7. `vaep/features/core.play_left_to_right:189-193`
8. `atomic/vaep/features.play_left_to_right:165-169`
9. `spadl/orientation._mirror_absolute_frame:222-225` — D8
10. `spadl/orientation._mirror_per_period:272-275` — D8
11. `_shape_graph.py:919-921` + `:931-932`

**Two places are defective by omission** — they should reflect and do not: the D1 call site
(`tracking/utils._resolve_action_frame_context:874`, which calls the primitive with an
incomplete column list) and the D2 site (`tracking/features._build_ball_xy_v_per_action`, which
never calls it at all). These are the two LIVE defects, and neither is a "reflection site" in
the arithmetic sense — which is precisely why a count of reflection sites was never going to
describe this defect family.

`feature_framework.ActionFrameContext` is a supporting change (it threads the flip), not a site.

ADR-045 should quote this breakdown rather than a headline number, and paste the command:

```bash
grep -rn "field_length -\|field_width -\|105\.0 -\|68\.0 -\|FIELD_LENGTH -\|FIELD_WIDTH -" silly_kicks/ --include=*.py
```

The grid reflections keep their current form and need **no new guard**. An earlier draft
promised "a shape assertion that both axes are reversed", which is not implementable:
slicing reversal is shape-invariant, so a shape assertion cannot distinguish `[:, ::-1]` from
`[::-1, ::-1]`, and a value assertion on the default grid is vacuous because the synthetic EPV
ramp is y-symmetric — the very property that let the ADR-041 defect survive its first repair.

The error class is already guarded **behaviourally** by
`tests/tracking/test_obso_orientation.py:158` (`TestEpvIsReflectedOnBothAxes`), which injects a
y-ASYMMETRIC EPV grid through the real `add_obso` and asserts the away team's threat is read
from the correct half. ADR-045 should point at that rather than add a second, weaker check.

### 4.7 Guards

**All per-row. Never aggregate.** §2 D2 and D3 both have near-zero mean bias with large
per-row error — a mean-comparison gate passes vacuously on broken code.

1. **Point-reflection invariance**, parametrized over every reflection site: construct a
   physically-complete transform (positions reflected, vectors negated, labels swapped), assert
   output is invariant. Each carries a **non-vacuity partner** asserting a positions-only
   reflection *fails* it — per CLAUDE.md's "every band needs a test from both sides".
2. **Registry completeness meta-assertion**: every column in `TRACKING_FRAMES_COLUMNS` union
   provider variants union preprocess-added union `SPADL_COLUMNS` union ADR-025 enrichment must
   have a declared kind. A new column fails CI until declared. Established idiom
   (`PURITY_ENTRIES`, `PUBLIC_ID_SCALAR_ENTRIES`).
3. **Divergence guard**: the two orienters must produce identical vector semantics on the same
   input — the D3 defect was precisely two siblings disagreeing.
4. **Call-site conformance**: the `reflect_columns` path tolerates a named-but-absent column by
   design (`if col in out.columns`), because frames without `derive_velocities` legitimately have
   no `vx`. That tolerance is necessary — and it means the two LIVE defects are fixed on a path
   whose failure mode is *structurally identical to the original bug*: the caller enumerates, and
   a miss is silent. Guard 2 covers the registries; it covers none of the enumerating sites.
   **`reproject_to_action_ltr` has FOUR call sites, not three** (measured: `utils.py:874`,
   `_kernels.py:879`, `features.py:2026`, `features.py:2433`). An earlier draft gated three and
   did not mention `features.py:2026` (`_reproject_team_shape`, six hand-enumerated columns) at
   all. Gate all four: (a) `_reproject_rows` — build a context on a velocity-bearing fixture and
   assert every geometry-named column in `actor_rows` / `opposite_rows_per_action` /
   `defending_gk_rows` is enumerated; (b) `_kernels.py:879`, where `y_cols=[]` is a live
   assumption that nothing lateral is ever added; (c) `finalize_orientation`, which runs on a
   pre-canonical frame carrying `x_centered` / `y_centered` — benign only because the adapters
   project them away, and nothing asserts they stay dead; (d) `_reproject_team_shape`.

   Three measured caveats for (d). First, its enumeration is **complete today** — under a full
   point reflection only the enumerated centroid/line-height metrics move; hull area, lengths,
   widths, stretch index and inter-line gaps are invariant to ~1e-13 (measured on the real
   `add_team_shape` output) — so this is anti-rot exposure, not a live wrong value. Second,
   `GEOMETRIC_NAME` cannot see its column names (`team_shape_centroid_x_attacking` is infix;
   `team_shape_defensive_line_height_attacking` has no axis token — both `.match() == False`,
   measured), so a name-pattern guard is the wrong instrument: gate it by **behaviour** (mirror
   invariance over auto-discovered columns) rather than by name. **Third, and this one is a live
   test gap, not just exposure:** the pre-existing `test_team_shape_centroids_mirror_invariant`
   is **vacuous on the y-axis**. Measured 2026-07-20 — on its centred `_scenario` the acting-team
   centroid sits ~1 m off the centre line, so `68−y` is a near-identity and disabling
   `_TEAM_SHAPE_Y_COLS` reflection entirely leaves the assertions green. The team-shape
   y-reflection is therefore *effectively untested today*. This is the same shape as ADR-041's
   y-symmetric-grid trap this whole PR exists to close, sitting inside the PR's own subject area.
   The site-(d) gate uses an OFF-centre fixture (`_ghost_scenario`, acting centroid ~17 m off,
   action 1 an away/flip=True row) with a disable-the-reflection both-sides partner that goes red
   (measured ON delta 0, OFF delta ~34). `features.py:2433` is genuinely inert — a two-column
   scratch frame renamed and enumerated on the next line.

   **These tests, not the two-line `vx_cols=["vx"]` change, are what make this a solution
   rather than a pile of patches.**

   The shared pattern is published as `reflection.GEOMETRIC_NAME` so the guards cannot drift
   from private copies. It is **fully anchored** and covers bare (`x`, `vx`, `dx`), prefix
   (`x_centered`, `x_smoothed`) and suffix (`defensive_line_x`) forms. An earlier draft used
   `re.match` against `r"^v?[xy]$|_x$|_y$"`, where the `_x$` branch requires the name to *start*
   with `_x` — measured: `defensive_line_x` and `x_centered` both `False`, making two of the
   three guards unconditionally passing while the third's non-vacuity partner used the four bare
   names that do match, which hid it. **Every guard therefore carries a both-sides partner that
   feeds it a synthetic geometric column and asserts it rejects it** — a guard that fires on
   nothing must prove it can fire.

   **The pattern's limits, measured — do not repeat "zero misses, zero false positives".** It
   is blind to INFIX axis names, and one real column has no axis token at all:

   ```
   team_shape_centroid_x_attacking            False
   defending_centroid_vx                      False
   team_shape_defensive_line_height_attacking False   (an x-position; no pattern can reach it)
   ```

   That is tolerable *only* because of the §4.5 division of labour: the pattern never decides
   anything. Library-owned columns are covered by the registry (complete by enumeration); the
   pattern only reports on passenger columns and only drives the conformance guards below.
   Widening it to catch infix forms would trade false negatives for false positives on names
   like `max_x_velocity`, and ADR-043's lesson is that a name heuristic should not be the
   enforcement mechanism. **State the measured blind spots in ADR-045 rather than a coverage
   claim.**

### 4.8 Out of scope

- The numpy grid reflections — already guarded behaviourally, see §4.6.
- `smooth_frames`'s config-tag idempotence short-circuit (`_smoothing.py:100-103`). Recorded in
  §2 D3b as a defeated mitigation; fixing it is a separate concern.
- TF-51. Its open design question (does flavour C beat flavour A) is decided by a measurement
  that requires corrected `bekkers_pi`, which does not exist until this ships. Speccing it now
  would bake in an ungrounded decision.

---

## 5. Consequences

| consequence | detail |
|---|---|
| `bekkers_pi` values change | away-team actions only; home bit-identical |
| pressure goldens | regenerate (`scripts/regenerate_pressure_snapshot_shas.py`) |
| xT-GK v2 deep-zone gate | **re-run** — the GO-leaning verdict was measured on broken pressure |
| rho retention weights | **retrain** both `default` and `skillcorner` variants |
| lakehouse | re-materialize `fct_action_context.pressure_on_actor__bekkers_pi` |
| bundled xS / xCross / ghost weights | **unaffected** — none consume `bekkers_pi` |
| CLAUDE.md + ADR-042 | correct the D6 claim |

No converter output changes, so no VAEP retrain beyond the `bekkers_pi`-consuming surfaces.

### 5.1 Bundle the recompute with TF-35 — pinned ordering

This PR and the shipped 4.52.0 xT-EPV / TF-35 work **both** force a lakehouse action-context
recompute. On the lakehouse side an AC-1 cold drain is ~5.5 h across 8 workers plus a staging
rebuild and a `rederive_synced_marts --rebuild`; running it twice is a full working day of
compute and two windows of mart churn. The downstream chains also overlap: `bekkers_pi` -> rho
retention retrain (both variants) -> the xT-GK v2 deep-zone gate -> the `xt_gk_v2` family
materialization, which is already queued behind the rho retrain.

**Pinned order — one pass, not two:**

1. Ship both library changes (4.52.0 already out; this PR as 4.55.0)
2. **One** wheel bump
3. **One** AC drain
4. **One** rho retrain (both variants)
5. Re-run the deep-zone gate **once**, on corrected pressure

The three follow-ups in the plan's final task are **not** independent and must not be scheduled
as such. State this in the PR body.

---

## 6. Coverage gaps (what this audit did not establish)

- **Only `bekkers_pi` was measured end-to-end.** Other velocity-directional consumers (DAS,
  pitch control, cover shadows, GK influence) were shown exposed *in principle* via a pitch-control
  probe (8.87% of grid cells changed, max delta 0.500) but their production values were not
  quantified, because in-library frames never reach the D3 flip.
- **Single-match evidence.** All real-data numbers come from one IDSSE match
  (DFL-MAT-J03WMX). Provider generality is argued from code structure, not measured.
- ~~**No lakehouse inspection.**~~ **CLOSED** by cross-session review: the lakehouse AC path uses
  the geometric orienter, not `orient_frames_to_ltr`, so D3 does not reach it. D3 downgraded
  LIVE -> LATENT (§2 D3). The gap is closed for the *named* consumer only — an unknown third-party
  caller of `orient_frames_to_ltr` that derives velocities first would still hit it.
- **The registries are validated against measured schemas, not exhaustively.** The 32-column
  SPADL surface is the union over the four `*_SPADL_COLUMNS` dicts plus `add_names` plus ADR-025
  enrichment. The tracking registry was measured complete against all five
  `*_TRACKING_FRAMES_COLUMNS` (`undeclared: []`); the atomic registry likewise (`missing: []`,
  `extra: []`). The SPADL registry's 32 keys are arithmetically right for that union — but
  `preserve_native`, `add_possessions`, `add_gk_role` and `add_game_state` all attach columns
  outside it, which is why §4.5 does not make runtime completeness a contract.
- **No upgrade break, by design (changed in the fifth pass).** Under the §4.5 warn default a
  caller attaching its own column gets a warning at most, not an exception. The `possession`
  column that `preserve_native` surfaces is silent (non-geometric name). Consumers who want the
  old proposed strictness opt in via `filterwarnings("error", ...)`.
- **Nothing in the fifth review pass was validated on real match data.** Every measurement
  behind D8, the §4.5 reversal, and the Task 6 fixture rebuild is synthetic-fixture or
  source-structural. The claim that the tracking migration is byte-identical on the correct-flag
  path still rests on schema-projection reasoning, not a corpus run — the owner-gated
  two-provider e2e in the plan's Task 14 is what closes it.
