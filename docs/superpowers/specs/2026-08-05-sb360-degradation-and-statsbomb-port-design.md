# SB360 honest degradation and the StatsBomb parse port — design

**Status:** rev 4, 2026-08-05. Branch `sb360-degradation-and-port` off `main` @ `5b1a0a1`
(4.75.0, ADR-053). **Two commits, one branch, merged with `--merge` — never squash.** ADR number and
version are read off `main` at commit-prep, never assumed.

**Successor to** the SB360 coverage audit (ADR-053, `docs/research/sb360_coverage/`), which
deliberately reported rather than repaired.

**Goal.** A consumer running silly-kicks on StatsBomb 360 freeze-frames can (a) tell what the numbers
mean, and (b) get the data in without writing their own parser.

## Corrections established by review round 1

Four findings changed the design; two changed a number. Recorded here rather than silently folded in,
so the next reader sees what moved.

1. **The guard belongs at `_serve_positions_core`, not `add_ghost_gk`.** There are THREE public ghost
   entry points, not one, and two of them bypass the aggregator entirely. Rev 1 fixed one caller in
   four. §1.2.
2. **The velocity-gap number was wrong.** Rev 1 said 17 -> 15. Measured from the committed registry
   it is **8 -> 5**. Rev 1's figure came from a fixed-window regex that overran entry boundaries.
   §1.4 now rests on the principle rather than the magnitude, and states the query. The registry has
   **34** entries, not 33 — `spadl.add_restart_coordinates` is in it.
3. **The port emits a DIFFERENT orientation convention from the one `compute_ghost_gk` documents.**
   New §2.3. The audit could not see this: its fixture shares positions across both legs and its
   header states orientation is not exercised.
4. **"Byte-identical" was asserted while the schema changes.** Split in §1.3.

**Rev 2 wrongly disputed the sweep rule; retracted in rev 3.** Rev 2 claimed the rule was unadopted
on the evidence of `grep -ci "caller sweep" CLAUDE.md` -> 0. That was a search for the *label*, not
the rule. The rule is **`CLAUDE.md:146`, the file's last bullet, verbatim**: "A spec that changes a
public seam enumerates every caller of every CHANGED FUNCTION and classifies each as affected or not
WITH EVIDENCE ON BOTH SIDES -- and the sweep is the FLOOR, not the check." `TODO.md:29` is a task to
MECHANIZE it into a gate, which is evidence it is adopted and ungated, not unadopted. §1.8 is
compliance, not supererogation.

The error is worth recording because it is the one this repo keeps finding: a substring test over
source used as evidence about semantics. It also had a consequence — the rule enumerates four things
a symbol sweep structurally cannot see, and rev 2 missed clause **(c) committed data artifacts, which
call nothing**. Two were absent from §1.8 (§1.8 now carries them).

## Scope

**Commit 1 — honest degradation.** Ghost-GK contract compliance at the shared serving seam; a
`validate_velocity_regime` diagnostic; the SB360 registry re-adjudication that follows.

**Commit 2 — `providers/statsbomb`.** A shape-only parse port that also carries `visible_area`
through as raw data.

**Explicitly NOT in this cycle:**

| item | why not |
|---|---|
| a `visible_area` consuming API | no consumer exists; ADR-009's "ship RAW primitives" applies. Carrying the polygon now is what makes the seam buildable later without touching the port again. |
| `snapshot_to_tracking_frames` dtype pin | "pin to WHAT" is its own Hyrum decision and deserves its own reasoning. |
| the 4 unauditable boundary entry points | blocked, but for **four distinct reasons**, not one. `tests/sb360/test_registry_surface.py::UNAUDITABLE_BOUNDARY` records them: only `compute_xt_gk_v2` is xG-blocked; **three are gkdv entries blocked on SHAPE** (frames-only or frame-pair, so no action-paired comparison exists), which no xG model would unblock. Rev 3 implied a single cause. That gate is `xfail(strict=True)`, so covering any of them forces the marker to be revisited rather than quietly passing — this deferral is mechanically protected, unlike `_defending_goal`'s. |
| fetching | `providers/sportec/parse.py` fetches nothing (measured: zero `requests`/`urllib`/`http`). The caller owns I/O. |
| fixing `compute_ghost_gk`'s identity-keyed `_defending_goal` | Deferred, but **QUEUED, not dropped** — see below. Folding a live orientation defect into a degradation cycle mixes two retrain stories. |

**Rev 3's deferral of `_defending_goal` routed it to a register that does not contain it.** Measured:
the string `_defending_goal` appears **zero** times in `CLAUDE.md`, in `TODO.md`, and across every
file under `docs/superpowers/adrs/`. And D3 is not a loose category —
`tests/tracking/test_mirror_registry.py:294-311` pins the unit as exactly three files
(`_defensive_line.py`, `_packing.py`, `_gk_influence.py`) with an equality assertion whose message
reads *"a partial re-key is the failure mode this test exists to catch"*. `_ghost_gk.py` is not a
member, and `add_ghost_gk` carries no `known_defect` marker in
`tests/tracking/_mirror_entries/gk_models.py`, so it is not among the 8 Gate B / D3 markers
`TODO.md:5` reserves for PR 6. "Declared, not repaired" therefore **dropped** it.

**This cycle must queue it explicitly** — a `TODO.md` row at merge time, or add `_ghost_gk.py` to the
D3 unit and let the membership assertion trip deliberately, which is what that assertion is for.

**And it is more than a D3 orientation defect.** `tracking.defended_goal_x`
(`_gk_resolve.py:323`) is the **pinned public goal map**, and its own docstring says consumers "must
call THIS rather than re-derive the rule, because a second implementation is a fork that can disagree
with the first." `_ghost_gk.py:814-818` is that second implementation, and the two **already
disagree**: `defended_goal_x` falls back to the team's mean outfield x when a `(game, period, team)`
has no GK rows; the inline version has no such fallback and falls through to an identity-keyed default.
So this is a live violation of an existing pinned-seam rule — which makes the eventual fix smaller
than §2.3 implies, and the deferral harder to justify.

---

## 1. Commit 1 — honest degradation

### 1.1 The defect, and why the mechanism matters

CLAUDE.md's `speed_source` bullet already states the rule:

> An UNMARKED or PARTIALLY-marked frame set missing `vx`/`vy` still RAISES: fail-loud wins on a mixed
> frame set.

Measured on the audit's paired fixture at `5b1a0a1`, the ghost path violates it in **both**
directions — marked frames fabricate instead of degrading, unmarked frames fabricate instead of
raising. Neither change is new policy; both restore a written rule that `_das.py:259` and
`_press_commitment.py:100` already obey.

**The mechanism is a learned imputation policy, not a zero-fill.** `extract_ghost_gk_features` yields
NaN for all five velocity features; `predict_mean` reconstructs an sklearn HGBR, which routes NaN
down each split's *learned missing-value direction* — fitted where NaN meant an occasional dropped
measurement, applied where 5 of 26 features are absent on 100% of rows. Measured:
`NaN -> [6.795, 33.522]` versus `zero-fill -> [6.888, 33.362]`. **"Fill the zeros correctly" is
therefore not the fix.**

### 1.2 The guard goes at the SHARED SERVING SEAM

There are three public entry points, all in `tracking.__all__`, all mirrored into
`atomic/tracking/features.py`, all funnelling through `_ghost_gk._serve_positions_core`:

| entry point | reached by | rev 1 covered it? |
|---|---|---|
| `add_ghost_gk` | the aggregator path | yes |
| `compute_ghost_gk` | `ghost_gk_xfns` -> `features.py:4536`, i.e. **the VAEP path** | no |
| `serve_ghost_gk_positions` | `gkdv/_engine.py:537`, i.e. **TF-19** | no |

Worse, `add_ghost_gk` short-circuits at `features.py:4533` —
`if "ghost_gk_x" in frames.columns and frames["ghost_gk_x"].notna().any()` — so a caller who runs
`compute_ghost_gk` first and passes the enriched frames in walks **past** the guard on a documented,
supported path.

**Placement is settled by house precedent, not preference.** `_serve_positions_core`'s own docstring
says it single-sources "the 4.12.1 duplicate-(frame, gk_team) collapse, `predict_mean` and the
4.22.1 physical-pitch clamp" — and that clamp is the exact policy CLAUDE.md cites for *policy lives at
the edge, never in the shared engine*. The serving seam IS that function. The refusal goes there and
all three edges inherit it.

**Reuse, do not restate.** The mixed-regime rule is already encoded in
`velocity_unavailable_by_design` (`tracking/_velocity_availability.py:15`) and the raise at
`_press_commitment.py:103-107`. Call the former; mirror the latter's message. A third divergent
reading of the same rule is the defect this cycle is repairing.

### 1.3 Behaviour after — values identical, schema changes

    frames marked unavailable  -> ghost positions = NaN, ghost_gk_source = "velocity_unavailable"
    frames unmarked, no vx/vy  -> ValueError (mirrors _press_commitment.py:103-107)
    frames with velocity       -> ghost position VALUES byte-identical; schema gains one column

**TWO of three seams emit `ghost_gk_source`; the third signals by returning NO ROWS.** Rev 3 said all
three emit it. That is wrong, and the reason is a hard guard rev 3 did not know about.

`gkdv/_engine.py:557-562` raises when any SCORED row carries a non-finite ghost coordinate, with the
message *"Pitch control silently DROPS NaN-coordinate rows (`_spearman.py` dropna), so a NaN ghost
would make the keeper vanish rather than error."* So a serving seam that returns **NaN rows** on a
marked frame set walks TF-19 straight into that raise — with a message blaming the ghost model rather
than naming the velocity marker, which is a confusing failure for the exact consumer this cycle
exists to make honest.

The decision, and what follows from it:

| seam | writes to | on a marked frame set |
|---|---|---|
| `add_ghost_gk` | **actions** | NaN positions + `ghost_gk_source` |
| `compute_ghost_gk` | **frames** (`_ghost_gk.py:2244-2246`) | NaN positions + `ghost_gk_source` |
| `serve_ghost_gk_positions` | **its own DataFrame**, one row per served frame | **returns no rows** |

Returning no rows routes TF-19 into its existing counted-drop path (`_DROP_NO_GHOST =
"no_ghost_served"`, `_engine.py:49`), which is the clean behaviour. **Rev 3's §1.8 claim that "a NaN
ghost is already dropped-and-COUNTED by ADR-043" was false** — that token fires on frames the seam
returned NOTHING for, never on frames it returned NaN for. Corrected in §1.8.

Two consequences, stated rather than left to plan-time pressure:

* **GKDV deliberately does not carry `ghost_gk_source`.** `_build_provenance` returns
  `out[_PROVENANCE_COLUMNS]` — an explicit 16-column projection (`_engine.py:59-74`) — so the column
  would be filtered out one hop from the consumer anyway. The refusal's REASON is recoverable by the
  caller through `validate_velocity_regime(frames)`, which is what that diagnostic is for. Widening
  `_PROVENANCE_COLUMNS` is a TF-19 decision, not a degradation-cycle one.
* **The naming collision disappears.** `serve_ghost_gk_positions` returns `ghost_gr_x`, `ghost_gr_y`,
  `ghost_clamped`, `ghost_out_of_box` — the `ghost_` prefix, not `ghost_gk_`. Since that seam emits no
  new column, `ghost_gk_source` never has to sit among differently-prefixed siblings, and the closed
  vocabulary stays single-spelled for the enum consumers are told to pin to.

Also carry into the plan: the frame-level column is representable (`ghost_gk_x`/`ghost_gk_y` are
already frame columns), but **verify whether either is declared in the ADR-045 reflection registry** —
a check, not an assumption. And the SB360 registry, liveness gate and glossary all key off
**`add_ghost_gk`**, so §1.6's adjudication covers the aggregator only; the other seams are covered by
direct tests (§4).

Rev 1 asserted "byte-identical" flatly, which was wrong: adding `ghost_gk_source` changes the column
set on the velocity path too. The split matters because **schema-sensitive consumers exist** —
lakehouse materializations, parquet concat across partitions, and any `set(df.columns)` assertion.
The values claim is what keeps this retrain-free; the schema claim is a Hyrum note for the CHANGELOG.

**The column is UNCONDITIONAL.** `_press_commitment.py:46` puts `press_commitment_source` in
`_OUTPUT_COLS` and emits it always; a conditionally-present provenance column is a worse contract
than an always-present one, because a consumer cannot tell "absent because computed" from "absent
because this version predates it". This also settles the ADR-033 question rev 1 left open: the column
is not branch-dependent, so the "≥2 purity variants" contributor contract does not apply to its
presence — but the **NaN vs value** branches still both need purity coverage.

`ghost_gk_source` is a closed vocabulary in the `PRESS_COMMITMENT_SOURCE_VALUES` mould, each token
exported as a module constant so consumer enums pin to the library's set:

    GHOST_GK_COMPUTED             = "computed"
    GHOST_GK_VELOCITY_UNAVAILABLE = "velocity_unavailable"
    GHOST_GK_NO_KEEPER            = "no_keeper"
    GHOST_GK_UNLINKED             = "unlinked"
    GHOST_GK_SOURCE_VALUES        = (...)

**`ghost_gk_xfns` does NOT emit it.** CLAUDE.md is explicit that `das_xfns` deliberately omits
`das_source` because VAEP matrices stay numeric. Same here. The xfns path still inherits the REFUSAL
(it goes through the serving seam), it simply carries no string column.

The raise is **breaking** for a caller feeding unmarked velocity-less frames. What breaks is a
silently fabricated coordinate, so it is a CHANGELOG note, not a migration.

### 1.4 `validate_velocity_regime` — one diagnostic, not N columns

**Measured, with the query stated so it is reproducible.** Loading `tests/sb360/_registry`, iterating
`iter_verdicts(entry)` and filtering the **velocity** axis to adjudications in
`{differs_by_design, silent_degrade}` yields **12** aggregators. Of those, **8** carry a
`[measured cause=velocity]` marker; the other 4 are `cause=frame_count` only (`add_actor_pre_window`,
`add_off_ball_context`, `add_off_ball_runs`, `add_shot_goalmouth`). Subtracting `add_ghost_gk` and the
two that already read the marker (`add_das`, `add_press_commitment`) leaves **5**:

    add_elastic_sync   add_obso   add_pausa   add_pitch_control   add_space_creation

Rev 1 said 15. That figure came from a fixed 9000-character window per registry entry which overran
into neighbouring entries — a measurement bug, and exactly the "number recorded in prose rather than
derived" failure this repo keeps finding.

**5 is a FLOOR over the surface the fixture exercises, not a census.** Two aggregators carry a
`cause=velocity` marker and are excluded for opposite reasons, and only one of those exclusions is a
negative result:

* `add_cover_shadows` — `honest_nan` x5 on the velocity axis. It already declines cleanly.
  Correctly out.
* `add_xshot_occurrence` — `not_exercised` x1. The fixture never reached it on that axis, so it is
  **unknown**, not negative. It may belong in the 5.

The audit's own Limitations already book 26 `not_exercised` verdicts as a known blind spot. Phrase
any downstream use of this number as "at least 5".

**The argument does not rest on the magnitude, and rev 1 was wrong to let it.** Those 5 are not
broken: on freeze-frames they produce honest, usable values — pitch control at zero velocity is a
well-defined positional model. What a consumer cannot tell is that the value is positional-only, and
that is a property of the whole frame set, not of any row. A per-row column would carry a constant,
whether the count is 5 or 15.

So: a third member of an established family — `validate_time_base`/`TimeBaseDiagnosis` (ADR-017) and
`validate_id_dtypes`/`IdDtypeDiagnosis` (ADR-019), the second of which CLAUDE.md already describes as
mirroring the first:

    validate_velocity_regime(
        frames, *, on_mismatch: Literal["warn", "raise", "ignore"] = "raise"
    ) -> VelocityRegimeDiagnosis

    VelocityRegimeDiagnosis:
        regime: str                        # velocity_informed | positional_only | mixed
        speed_source_counts: dict[str, int]
        has_velocity_columns: bool
        message: str

Two deliberate departures from its siblings, each with a reason: it takes **`frames` only** (velocity
regime is a property of frames; an unread `actions` parameter would repeat the dead-`home_team_id`
defect CLAUDE.md records against `space_creation`), and it carries **no list of affected
aggregators** (that would be a hand-maintained registry beside a mechanism that could derive it —
the genus the concurrent Cycle B exists to remove).

### 1.5 The rule this establishes

**A provenance COLUMN where the value changes; a DIAGNOSTIC where only the interpretation changes.**

The ghost path earns a column because a number becomes NaN — a per-row fact. The other 5 produce the
same value either way; only its reading changes, which is frame-set-level. Writing this down is what
stops the next velocity-consuming feature adding a sixth `*_source`.

### 1.6 Registry re-adjudication — four MOVES plus a new COLUMN

Two distinct pieces of work; rev 1 counted only the first.

* The ghost path's four existing verdicts move `silent_degrade` -> `honest_nan`.
* **`ghost_gk_source` is a NEW emitted column.** `_regenerate.py:120` derives columns from actual
  output, so it will appear on all three axes and needs hand-written rationales. Precedent exists and
  de-risks it: `press_commitment_source` is adjudicated across the same axes, including
  `applicability="no_support"` at `_entries/_offball.py:405`.

Regenerated through `_regenerate.py` then `_adjudicate.py` — **never hand-edited**. Do not run the
regenerator concurrently with pytest; it rewrites `_entries/` under a live collection.

### 1.7 Gate surface — Commit 1

Feature glossary (ADR-048), aggregator liveness (non-NaN AND non-constant — note `ghost_gk_source` is
a STRING column, so the non-constant prong is float-gated and does not apply), purity (ADR-033, both
value branches), mirror registry Gates A and B (ADR-051), id-dtype invariance (ADR-019).

`validate_velocity_regime` must be added to `silly_kicks/tracking/__init__.__all__`, and — because
`tests/test_public_api_examples.py` derives its surface — needs a real Examples section, not a
`+SKIP` doctest and not an import-only stub.

### 1.8 Caller sweep

`CLAUDE.md:146` compliance. Every consumer of the three changed seams, classified before landing —
plus the four things that rule says a symbol sweep structurally cannot see.

| consumer | class |
|---|---|
| `silly_kicks/gkdv/_engine.py:537` | inherits the refusal via **no rows**, routing into `_DROP_NO_GHOST` (§1.3). **Rev 3's classification here was wrong**: `_engine.py:557-562` RAISES on a non-finite ghost on a scored row, so returning NaN rows would break TF-19 rather than degrade it. The counted-drop path is only reached by returning nothing. |
| `silly_kicks/atomic/tracking/features.py` | mirrors all three entry points; inherits. |
| `silly_kicks/tracking/features.py:4536, 4628` | the two xfns paths; inherit the refusal, emit no string column. |
| `scripts/make_ghost_gk_golden.py` | captures a same-environment output oracle. Re-capture required if values move; they must not. |
| `scripts/gen_ghost_gk_kde_golden.py` | **second generator, missed by rev 2.** Same classification. |
| `tests/tracking/data/ghost_gk_refactor_golden.npz` (1462 B) | **clause (c): a committed artifact, which calls nothing.** Under the values-identical claim it is unaffected — which is the "classified as unaffected WITH EVIDENCE" case, so the evidence is the byte-identity assertion itself, not silence. |
| `tests/tracking/fixtures/ghost_gk_kde_golden.npz` (5465 B) | same. |
| `docs/huggingface/model-cards/ghost-gk-v1-model-card.md:102` | states serving behaviour; update if it describes the fabricating path. |
| `tests/sb360/_entries/_gk.py` | regenerated, §1.6. |
| `docs/research/sb360_coverage/{README,coverage}.md` | hand-written. The Layer A claim becomes historical — annotate, do not rewrite the measurement. |
| `docs/research/sb360_coverage/behaviour_matrix.md` | **RENDERER-GENERATED, not hand-written** — `render_sb360_matrix.py` "reads a COMMITTED registry and writes a document ... deterministic given the tree". §1.6's re-adjudication forces a RE-RENDER, and its summary counts change by construction (`silent_degrade` 4 -> 0). Grouping it with the two above was wrong. |
| `CLAUDE.md:140` | **clause (d), a number recorded in prose.** Its ADR-053 bullet uses `add_ghost_gk` as the running example of the unfixed fabrication ("a fitted model silently imputing the features it was trained on is not"). This is the file the harness loads as standing instruction, so a stale claim costs more here than in a changelog. NOT deferrable to merge time — unlike `CHANGELOG.md:30` and `TODO.md:5`, which record the same "exactly 4 silent_degrade" figure and ARE deferred by §3. |
| `docs/PRIVATE_CONSUMERS.md` | checked for pins on `_ghost_gk` internals. |

---

## 2. Commit 2 — `providers/statsbomb`

### 2.1 The boundary: shape, never fetch

`providers/sportec/parse.py` fetches nothing (measured). `providers/statsbomb` mirrors it: accepts
already-loaded SB events and 360 payloads, returns the contract `snapshot_to_tracking_frames` expects.
No new runtime dependency.

**`statsbombpy` is a SCRIPT dependency, not a test-only one** (rev 3 said test-only; wrong). It is
imported lazily inside `scripts/build_sb360_coverage.py`'s work functions and is declared **nowhere
in `pyproject.toml`**. The undeclared-dependency situation predates this cycle but is now adjacent to
it; the port itself must not acquire the dependency.

### 2.1a The port is EXTRACTED from `scripts/build_sb360_coverage.py`, not written beside it

Rev 3 treated the port as greenfield. It is not. The audit's own Layer B driver already implements
most of the parse half:

    :57-58   SB_PITCH_LENGTH = 120.0 / SB_PITCH_WIDTH = 80.0
    :137     _defending_gk_visible      -- actor-relative keeper flags
    :146     _acting_side_gk_visible
    :163     _visible_fraction          -- consumes the FLAT visible_area
    :179     _adapt_events              -- events adapter
    :244-262 event_uuid join + zero-overlap handling

Writing a second implementation beside it produces exactly the fork that `defended_goal_x`'s docstring
names as a defect class — in the same cycle that cites that rule against `_ghost_gk`. So: **extract
the shared logic into `providers/statsbomb`, and re-point the script at it.**

Three consequences:

* **`coverage.md`'s published numbers were produced by that script**, including this spec's own
  3-of-22 broken-linkage figure. A divergent port makes the audit's measurement irreproducible from
  the library.
* **§2.6 is therefore not an open choice.** The script already picked: WARN plus an emitted
  `match_join_rate=0.0` — a counted report, not a typed error (`:255-262`). Adopt it, or state why the
  port diverges.
* The script takes `require_clean_tree` under ADR-037, so re-pointing it raises the rule's **clause
  (e) second hop**. **ANSWERED: no re-run needed.** The extraction is an identity MOVE, not a copy --
  verified `mod.defending_gk_visible is defending_gk_visible` (and likewise for the other two), so
  the script calls the very objects the library exposes and `coverage.md`'s numbers cannot drift
  from the port by construction. Declared in the commit message rather than left implicit.

**`providers/__init__.py`'s docstring currently ends "Behind the `[parse-dfl]` extra."** That becomes
false the moment a second, extra-free port lands. Update it in the same commit.

### 2.2 Outputs and their contracts

    snapshots     one row per player per event -- action_id, team_id, player_id,
                  is_goalkeeper, x, y  (SPADL coordinates)
    visible_area  one row per ACTION -- action_id + polygon (SPADL coordinates)
    the event_uuid -> action_id join

Facts about the source that shape this, all measured during the audit:

* **The 360 file carries no event type** — every coverage question is a join against the events file.
* **Player flags are relative to the ACTOR** (`teammate`, `actor`, `keeper`, `location`).
  `is_goalkeeper` comes from `keeper`.
* **There is no player identity, and the synthetic id does NOT recur.** `_snapshot.py:106` assigns
  `np.arange(len(player))` across the whole table, so the same physical player receives a different
  `player_id` in every freeze-frame. **This must be in the port's published contract**, because it
  forecloses per-player aggregation — which for a GK collaboration is exactly what someone will try
  first. State it; do not let it be discovered downstream.
* **`visible_area` is a FLAT `[x1,y1,x2,y2,...]` polygon in StatsBomb 120x80.**

### 2.3 The orientation the port emits, declared

`snapshot_to_tracking_frames` stamps `team_attacking_direction="ltr"` on every row
(`_snapshot.py:131,163`) because snapshot frames are already in SPADL **action-LTR**.
`compute_ghost_gk`'s docstring (`_ghost_gk.py:2199`) requires **home-attacks-right**. Those are
different conventions (CLAUDE.md, ADR-028).

What is actually at risk is narrower than it first appears, and the port must say so:

* **Safe.** The ADR-028 reprojection layer keys on `team_attacking_direction`, reads `"ltr"` and
  correctly declines to flip — so `add_team_shape`, `add_defensive_line` and the shared-context
  kernels are fine.
* **At risk.** Seams keyed on team IDENTITY rather than direction. `compute_ghost_gk`'s
  `_defending_goal` (`_ghost_gk.py:814-818`) takes a GK mean-x per `(game_id, period_id, team_id)`;
  on action-LTR frames one team's keeper appears at BOTH ends within a period, so the mean collapses
  to a single answer that is wrong for whichever share of events the other team took. CLAUDE.md's
  still-open **D3** target `_gk_influence.py:371` has the same shape.
* **Why the audit could not see it.** Its fixture shares positions across both legs, and its header
  states orientation is not exercised. A convention error is invisible to a comparison whose two
  sides share the convention.

After Commit 1 the marked path returns NaN, so this is moot for the ghost path on SB360 — but the
port must **declare the convention it emits** and name which seams are safe under it. That is a
docstring and a spec statement, not code.

### 2.4 The coordinate transform is not trivial, and the clip is wrong for a polygon

`spadl/statsbomb.py:393-427` (`_convert_locations`) does four things beyond scaling: a cell-centre
correction `crc = cell_side/2`, a branch on `xy_fidelity_version` (cell side 0.1 vs 1.0), a special
case for 3-element shot locations (`y_offset = 0.05`), a **y inversion**
(`field_width - ...`), and a **clip** to the pitch.

Three consequences to decide out loud rather than discover:

1. **What gets promoted is the scalar affine, NOT `_convert_locations`.** Rev 2 said "promote it to
   a shared seam both call". Measured, that is the wrong shape and would fail **silently**:
   `spadl/statsbomb.py:415` is `loc[:2] if isinstance(loc, list) and len(loc) >= 2 else [nan, nan]`,
   and a flat `[x1,y1,x2,y2,...]` polygon satisfies `len >= 2`. Run on a 4-vertex polygon it returns
   shape **(1, 2)** — the first vertex only, no error and no NaN.

   So promote the **scalar affine** `(x, y) -> SPADL` (fidelity branch, cell-centre correction,
   y-inversion), leave `_convert_locations` as a thin per-row wrapper over it, and have the polygon
   path reshape to `(N, 2)` **before** calling it. The **3-element special case**
   (`y_offset = 0.05`, a shot's z-height) must NOT apply to polygon vertices.

   The split is precedent-backed, not novel: ADR-038 already separates `_scale_to_spadl` (affine
   only) from `_transform_coords` (scale + clamp) for exactly this reason.
2. **The y-flip reverses polygon winding.** Irrelevant to points, material to any `shapely` or
   `matplotlib.Path` consumer. The port must either normalise winding or document the orientation.
3. **The clip must NOT be applied to `visible_area`.** A camera legitimately sees past the touchline,
   so clipping silently shrinks the observed region — and "observed region" is the entire point of
   the column. This is exactly ADR-038's defect class, where SkillCorner's `_transform_coords` clamp
   is "safe for events, on-pitch by construction, and destructive for tracking". Scale and invert;
   do not clamp.

Rev 1's Risk 3 offered "carry it in native 120x80" as a fallback. **That is withdrawn** — it would
put the polygon in a different coordinate system from the snapshots it describes, which is a trap
rather than a graceful degradation.

**RESOLVED by measurement (implementation): apply `crc`, and the reason is not the one the question
assumed.** The feared conflict with `_visible_fraction` does not exist -- that function returns an
AREA RATIO and `crc` is a pure translation, so it is invisible there (measured: 0.625 either way).
What binds instead is player/polygon alignment: players reach SPADL through the same affine WITH
`crc`, so omitting it on the polygon would offset it **0.4375 m** from the players it bounds, and a
player exactly on the boundary would read as outside. Pinned by
`test_players_and_polygon_share_one_transform` and confirmed on the real committed slice.

The original question, retained because it is what the measurement answered: `crc` exists because SB *event* locations are cell-based — `_convert_locations`' own
docstring says "1,1 is the top-left square 'yard' of the field". `visible_area` is a continuous
polygon, not a cell reference. Applying `crc` where it does not belong is a systematic **~0.44 m**
offset at fidelity 1 (0.044 m at fidelity 2): small, silent, and it would misalign players against
the very polygon meant to bound them. One measurement on the committed slice settles it; it is an
acceptance item, not an assumption carried over from the events path.

**Same-repo prior art to reconcile against, not merely to choose past.**
`scripts/build_sb360_coverage.py:163` (`_visible_fraction`) already consumes this polygon — a shoelace
over `flat[0::2]`/`flat[1::2]`, normalised by `SB_PITCH_LENGTH * SB_PITCH_WIDTH` — applying **no**
cell-centre correction, **no** y-inversion and **no** clip. That is not dispositive (it works in
native 120x80 and never needs SPADL coordinates), but if the port applies `crc` and that function does
not, two readers of the same polygon in the same repo disagree about what it means. Cite it in the
measurement.

### 2.5 Gate surface — Commit 2

* `tests/test_public_api_examples.py` derives its public surface and rule **P2** is "no
  underscore-prefixed path component", so `silly_kicks/providers/statsbomb/*` lands in it: every
  public symbol needs a real Examples section. A module that is neither enforced nor excluded fails
  CI.
* The C4 completeness gate is **unaffected, verified not assumed**: `_shipped_subpackages()` in
  `tests/test_c4_dsl_description_cap.py` calls `_package_root().iterdir()` — top level only, not
  recursive — so it sees `providers` (already modelled) and never descends.
* No new `add_*`, so the documented C4 aggregator count (32 action-coupled) is unchanged.

### 2.6 Broken linkage is a first-class outcome

**Measured: 3 of 22 open-data matches (14%)** ship a 360 file whose `event_uuid`s have zero overlap
with their own events file while claiming the same `match_id` (`3877115`, `3877170`, `3877194`).
Verified against RAW events — upstream, not a join defect.

The port surfaces this as a typed error or a counted report, **never a silently empty join**. The
audit learned this expensively: three one-row `unmapped` shards were indistinguishable from quiet
matches and would have diluted every aggregate they entered.

### 2.7 Fixture

Golden parity against a committed reduced slice with a `SOURCE_SHA`, mirroring
`tests/datasets/sportec/idsse_slice/`.

**The slice already has a home, and the license needs naming.** `tests/datasets/statsbomb/` exists
today (3 event JSONs, `spadl-WorldCup-2018.h5`, `raw/`, a README). **Extend it rather than create a
parallel directory** — it inherits the existing license note, and it needs two things it does not
have: a `SOURCE_SHA` and a `three-sixty/` sibling.

Rev 2 said "open data is redistributable" and dropped the qualifier. The README is more precise: the
**StatsBomb Public Data License (non-commercial)**, with the compliance note that this is
non-commercial use and redistribution is permitted under the same license. That qualifier travels
with the slice, and it is worth stating plainly given this cycle's motivation is a possible
**commercial** collaboration: the committed fixture is non-commercial open data, and nothing about a
commercial feed inherits its license.

**`NOTICE` carries zero StatsBomb entries** today while carrying license attributions for other third
parties (kloppy BSD-3-Clause among them). Landing a StatsBomb parse port plus a committed slice is
the moment that becomes an omission. Add an entry.

**The golden test must not depend on `statsbombpy`.** It is installed in `.venv312` but appears
nowhere in `pyproject.toml`, so an `importorskip`-guarded golden gate is vacuously green wherever it
is absent — the fixture-that-never-runs shape. Read the committed slice with stdlib `json`.

---

## 3. Conflict avoidance

The other session is executing Cycle B concurrently. `TODO.md` and `CHANGELOG.md` are the entire
realistic conflict surface and are deferred to merge time, along with the version bump.

Cycle B touches `tests/test_enrichment_nan_safety.py`,
`tests/tracking/test_frame_aware_xfns_dup_action_id.py`, `docs/c4/architecture.dsl` and SIX files
under `tests/scripts/`.

**CORRECTED at implementation.** An earlier revision claimed this cycle touches none of those. It
does touch `tests/scripts/test_build_sb360_coverage.py` -- Task 7 renames two helpers it asserts
on. The conflict does NOT follow, and the distinction matters: git conflicts are per FILE, and
Cycle B's six files under `tests/scripts/` are `_script_population.py`, `conftest.py`,
`test_artifact_provenance_output.py`, `test_corpus_driver_resilience.py`, `test_input_contracts.py`
and `test_provenance_wiring.py` -- measured, zero occurrences of `test_build_sb360_coverage`. The
directory overlaps; the files do not. So the test is edited directly rather than shimmed behind
underscore aliases, which was the earlier recommendation made before checking.

## 4. Testing

* **The values claim is an assertion, not a hope**: ghost positions on velocity-bearing frames must
  be identical before and after, asserted at the serving seam so all three entry points are covered.
  If it fails, this becomes a retrain trigger and the cycle's premise changes. Measure FIRST.
* **Both sides of every band.** Marked -> NaN *and* unmarked -> raise. A test asserting only the NaN
  passes identically when the computation silently produced nothing.
* **The bypass needs its own test**: run `compute_ghost_gk`, feed the enriched frames to
  `add_ghost_gk`, and assert the refusal still holds — that is the `features.py:4533` short-circuit,
  and it is how rev 1's placement would have leaked.
* **Each of the three entry points is tested directly**, not by assuming the shared seam covers them.
* `validate_velocity_regime` exercises all three regimes, including `mixed`.
* The port needs the broken-linkage case as a fixture, not only the happy path.

## 5. Risks

1. **The values claim fails.** Then ghost output on tracking data has changed, this is a retrain
   trigger, and the scope decision must be revisited rather than absorbed. Measure early.
2. **The raise breaks an unknown consumer.** Mitigated by §1.8's sweep and by what breaks being a
   fabricated number.
3. **Promoting the scalar affine touches the events converter, and there is NO existing gate to keep
   green.** Rev 3 assumed a golden-parity gate; measured, there is none.
   `tests/spadl/test_statsbomb.py` contains exactly two `assert_frame_equal` calls (`:393`, `:398`)
   and both compare conversions to EACH OTHER (option-invariance), pinning no coordinate values.
   `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` is a committed converted-output artifact but it
   is GENERATED by `scripts/build_worldcup_fixture.py` and CONSUMED as an input
   (`tests/conftest.py:18`, `test_enrichment_provider_e2e.py:50`,
   `vaep/test_labels_windowing_e2e.py:12`, `spadl/test_add_possessions.py:878`) — nothing diffs the
   converter against it.

   So the acceptance line requires **BUILDING a characterization test**, not keeping a gate green:
   convert the three committed raw event JSONs, snapshot the coordinates, diff. And that `.h5` is
   itself a **clause-(c) committed artifact derived from the changed function**, so **Commit 2 needs
   its own caller sweep** — §1.8 covers Commit 1 only.
4. **This spec can be incomplete the way its predecessors were.** Rev 1 was, in four ways, within a
   day. Anything discovered outside this Scope is recorded as a failure of this document.

## 6. Acceptance

* Ghost position values byte-identical on velocity-bearing frames, asserted at the serving seam.
* All three entry points refuse; the short-circuit bypass is tested.
* Both contract directions tested, including the planted unmarked case.
* The four SB360 verdicts read `honest_nan` AND `ghost_gk_source` is adjudicated on all three axes —
  regenerated, not hand-edited, round-trip verified.
* `validate_velocity_regime` exercises all three regimes and is in `tracking.__all__` with a real
  Examples section.
* A NEW characterization test pins the events converter's coordinates (none exists today), and the
  scalar-affine promotion is byte-identical against it.
* Commit 2 carries its own caller sweep, including `spadl-WorldCup-2018.h5` as a clause-(c) artifact.
* `_defending_goal` is QUEUED explicitly (TODO row or D3 unit membership), not merely declared.
* The port is extracted from `build_sb360_coverage.py` and the script re-pointed at it; `coverage.md`
  is re-run or explicitly declared still valid.
* **A flat N-vertex polygon round-trips to N vertices** — the measured `(1, 2)` truncation is the
  planted case this test exists to catch.
* `visible_area` is scaled and inverted but NOT clipped, with a test that a beyond-touchline vertex
  survives.
* The `crc` question is MEASURED on the committed slice and the answer recorded, not assumed.
* `add_ghost_gk` and `compute_ghost_gk` emit `ghost_gk_source`; `serve_ghost_gk_positions` returns
  NO ROWS and TF-19's counted drop fires — each tested directly, and the gkdv non-finite RAISE at
  `_engine.py:557` is never reached.
* `tests/datasets/statsbomb/` gains a `three-sixty/` slice with a `SOURCE_SHA`; `NOTICE` gains a
  StatsBomb entry naming the Public Data License (non-commercial).
* The port round-trips a real committed slice with no `statsbombpy` import; broken linkage is counted.
* Full suite 0 failed on `.venv312`; ruff + `ruff format --check` + pyright clean at CI scope
  (`silly_kicks/ tests/ scripts/`, never `.`).
* Version, CHANGELOG and TODO written at merge time from the commit log. `--merge`, never squash.
