# Changelog

All notable changes to silly-kicks will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.83.0] — 2026-08-17

The keeper-box geometry & detection-quality cycle: three independent pining-sourced passes against
one clean commit (`aa34017`), all artifacts stamped `run_commit aa34017`, `run_tree_dirty false`.

### Validated — SkillCorner keeper-origin resolver on the full pining corpus (PR-S152, ADR-024 amendment)

`scripts/validate_skillcorner_keeper_origin.py` confirms the shipped ADR-024 resolver on the **full
108-match SkillCorner pining corpus** (6,865 GK-distribution rows; `docs/research/skillcorner_keeper_origin/`),
closing the rate-gate follow-up ADR-024's 4.37.0 amendment deferred. Two **structural** CI rate-gates
(`tests/scripts/test_skillcorner_rate_gates_structural.py`, all legs): `offpitch_rate` and the gated
`out_of_region_goalkick_rate`, each computed/finite/under-a-loose-ceiling plus a both-sides mutation.
Corpus baseline: gated out-of-region **0.0** (~100 % own-box) vs a raw diagnostic **0.502** (the
broadcast-ball artifact the resolver corrects). The driver computes `gr_x = origin_x` (action-LTR,
defended goal at x=0), **not** via the frame goal map — an orientation bug caught on real data (the
frame-goal-map form scored 28.6 % own-box vs the correct 100 %); the fixture now carries an away-team
goal-kick + a non-vacuity guard.

### Measured — the gr_x behind-line clamp is immaterial; parked (PR-S152, ADR-061)

`scripts/measure_box_constant_delta.py` gains a `training_flip` block (basis A) measuring what a
`gr_x >= 0` clamp does to the actual training examples on the full 179-match corpus
(`docs/research/box_constant_delta/`). Ghost `attackers_in_box` changes **0.213 %**, xcross
`box_off_def_ratio` **0.193 %** — immaterial, so the clamp is **not** shipped (doc-only/parked, ADR-061).
26.8 % of the behind-line box points are > 2 m off-pitch (artifacts) → recorded as a data-quality
(D-data) observation (ADR-061 / `docs/research/box_constant_delta/`), not a geometry clamp. The driver adopts the ADR-052 `for_each` shard seam (resumable per-match shards).

### Recommended — TF-24 Stage-2 tracking defaults, within noise (PR-S152, ADR-009/ADR-060)

`calibrate_tracking_defaults.py --stage 2` over the full 179-match corpus, 60 trials, holding the
ADR-060 Stage-1 carrier params (`docs/research/tf24_stage2_refresh/`). Recommendation
`k3=2.94 / pre_seconds=2.26 / min_displacement_m=4.77` (held-out Brier 0.009553) beats the incumbent
defaults (0.009608) by 0.000055 — **within every per-provider SE** (0.0003–0.0019). Per ADR-009 the
harness recommends, never adopts; this result argues against adoption. **No library default change.**

### Added — xCross training-data meta seam (PR-S152)

`prepare_xcross_training_data` / `extract_xcross_features` gain `return_meta` / `return_box_detail`
(frame-free, additive) so the gr_x measurement sources both decision inputs from a single call. In no
default xfn list; no retrain.

### Doc — C4 model: pining now serves SB360 (owner-tier)

`docs/c4/architecture.{dsl,html}` — `pining-for-the-data` now advertises SB360 (owner-tier),
re-rendered with the pinned Graphviz `dot`; harmonized the redundant "StatsBomb SB360" → "SB360".

**Additive across the board — no library behaviour change, no retrain, no re-materialization, no
public-surface change beyond the two additive `_xcross_attempt` kwargs.**

## [4.82.0] — 2026-08-15

### Hardened — SB360 snapshot direction convention: named, tested, and correctly documented (PR-S151)

`snapshot_to_tracking_frames` labels both teams `team_attacking_direction="ltr"` because a
freeze-frame is already in SPADL action-LTR, so `acting_team_attacks_rtl` returns a resolved
all-`False` (no-flip) mask and SB360 is never re-projected (ADR-028). The value is now a named
constant `_SNAPSHOT_ATTACKING_DIRECTION` with a pointer to the authority in
`validate_period_directions`, pinned by `test_snapshot_actions_are_never_reprojected` (both teams
resolve to no-flip + a non-vacuity mutation that would catch a per-team regression). Corrected six
stale test comments that claimed `validate_period_directions` *rejects* a blanket `"ltr"` (it accepts
uniform `"ltr"`; it raises only on a single team self-contradicting) and two rotted `_snapshot.py:92`
citations (including one inside the `validate_period_directions` docstring the convention is
canonized in). Groomed the SB360 Tech-Debt section: removed the retracted goal-kick-coverage-constraint
row (strikethrough is not used in this repo; the measurement is preserved in
`docs/research/sb360_coverage/`).

**Doc/test only — no behaviour change, no retrain, no re-materialization, no public-surface change.**

## [4.81.0] — 2026-08-15

### Changed (BREAKING) — ghost-GK re-fit onto the canonical penalty-area constant (ADR-050 §6 discharged)

`_ghost_gk` derived its penalty area from a local **40.3 m** box (half-width 20.15) while `spadlconfig`
carries the Law's **40.32** (20.16). `attackers_in_box` is a trained feature, so the divergence was
train/serve skew that ADR-050's feature contract had already turned into a load-time raise. Ghost now
reads the canonical constant through a single vectorized predicate
(`_geometry.in_penalty_area_goal_relative_array`, which also flips the depth boundary `<` → `<=`), and
**both bundled variants are re-fit** on the 179-match corpus (scikit-learn 1.9.0). `attackers_in_box`
shifts, so opted-in ghost/VAEP consumers must re-materialize; the `[train]` extra now pins
`scikit-learn>=1.9`. The xCross box predicate collapses onto the same helper **value-identically**
(its weights are byte-unchanged). The Hub artifact `silly-kicks/ghost-gk-v1` was re-uploaded WITH a
feature contract, discharging `from_variant("full")` / `from_hub`. `CANONICAL_CONTRACT_KEYS` (ADR-050
amendment) splits the enumeration-gate accounting so a migrated constant may disappear from
`DECLARED_CONSTANT_SOURCES` without the gate becoming unsatisfiable.

### Changed (BREAKING) — VAEP labels survive a NULL `team_id` and never invent an opponent

`vaep/labels.py` and its atomic mirror compared team ids with a raw `==`, which ADR-027's Gradient
Sports null-actor rows (NULL `team_id`, nullable `Int64`) broke three ways: a Series compare produced
a `pd.NA` **label** (the calibration harness's `np.unique` then raised), a scalar-in-loop compare
raised on `pd.NA`, and the numpy path silently read `nan != nan` as True and **charged a null-team row
with the opponent's goal**. Every site now routes through `id_compat`; "other team" uses `ids_differ`
(both ids present), never `~ids_equal` (which would promote every unknown-team row to opponent).
Byte-identical on clean `int64`; the retrain trigger is **consumers only** — no bundled trainer imports
`vaep.labels`.

### Added — TF-24 recommendation honesty: the indistinguishable set, and `tolerance_m` as a held constant (ADR-060)

TF-24 Stage 1's `beta`/`gamma` are non-identifiable and its `tolerance_m` is under-determined (the
carrier objective has no loose-ball negatives, so a sweep presses the radius to its upper bound). The
Stage-1 confirmation now emits the **indistinguishable set** under a **prefer-incumbent** rule — the
recommendation stays the shipped default unless a candidate clears **both** a practical effect-size
floor **and** a paired-difference-SE significance test (via a shared `exceeds_noise_floor`, which now
also backs `tf25_gate_fires`) — plus a standing **fold-stability** diagnostic. **`tolerance_m` becomes a
held constant with zero swept, recommended, or consumed representation:** removed from `stage1_config`'s
search space and `CarrierAccuracyObjective`, excluded from the new committed, provenance-stamped
`carrier_selected.json`, and sourced by Stage 2 from `DEFAULT_CARRIER_PARAMS` (Stage 2 now refuses an
unprovenanced or dirty selection artifact). New public `silly_kicks.calibration` surface:
`select_recommended_point`, `PointScore`, `Selection`, `build_selection_artifact`, `exceeds_noise_floor`,
`MIN_EFFECT_SIZE`. Additive — no library default changes (ADR-009 preserved). The DGX confirmation is
done (`docs/research/tf24_stage1_confirmation/`, `run_commit 2cecd2b`, clean tree): the store held
`tolerance_m` at 3.0, `argmax_moved=False`, the fold-stability ratio is ~68,850× (between-fold vs
between-point noise), and the keep-incumbent recommendation is invariant to `δ` across `[0, 0.1]` (the
provisional `0.005` stands). **ADR-060 is Accepted.**

### Changed — provenance and script robustness

`git_provenance()` now records `platform` + `machine` (ADR-056 amendment), so a cross-platform artifact
mismatch is diagnosable from the artifact itself. `tracking_limit` counts records that carry player
data — a SkillCorner `period: null` prefix could otherwise slice to zero rows and read as a corrupt
download (pre-existing since 3.28.0); the `for_each` shard key splits on the first `__` separator only.
An ADR-019 amendment documents the VAEP label seam and the `~ids_equal` NA trap.

## [4.80.0] — 2026-08-11

### Changed (BREAKING) — ADR-051 D3 closed: direction never comes from team identity again

Every site that inferred the defended-goal end from **team identity** now takes **direction**.
`same_id(team, home_team_id)` is correct only while the frames are home-attacks-right and
**silently inverts otherwise** — it produces a confident wrong answer rather than an error, which
is why it survived three releases after ADR-055 named it.

**Scope was SIX sites, not the two the tracker recorded.** The list had ratcheted 2 → 4 → 6 across
three plan revisions, each time because it was *enumerated*. It is now bounded by a **predicate** a
machine can re-run — a site is in scope iff it CALLS `same_id`/`ids_match` with `home_team_id` —
and the pin asserts that predicate finds nothing:

| Site | Serves | Now takes |
|---|---|---|
| `_defensive_line.py` `compute_defensive_line` | both teams | `goal_map` (`get`) |
| `_packing.py` `compute_packing_metrics` | both teams | `goal_map` (`attacked_goal` **and** `get`) |
| `_structural_pass.py` | acting team | `attacks_rtl` bool |
| `_line_breaking.py` (ward path) | acting team | bool, resolved at the edge |
| `_off_ball_runs.py` `_line_break_kernel` | acting team | bool, resolved at the edge |
| `_player_influence.py` | attacking team | `attacks_rtl` bool |

**The mechanism is per-site, not uniform.** Only two of the six serve BOTH teams and need a map;
the other four need one team's direction and take a bool derived from `acting_team_attacks_rtl` —
the repo's single orientation authority (ADR-028/041), which ADR-042 already aligned TF-4 onto.
Threading a `GoalMap` into a one-team site would have reversed that consolidation and, at
`_player_influence`, handed a pitch-x to a function that collapses it to a boolean on its first
line. ADR-055's `goal_map` ruling is packing-specific and does not generalise: it turns on
supplying a float end for the DEFENDING team, which arises only because packing also calls
`select_back_line_players` for the other team.

**Unresolved ends REFUSE on the map path.** `GoalMap.get` returns `float | None` and `None == 0.0`
is `False`, so `get(...) == 0.0` silently means "defends x=105"; per-frame functions raise
`GoalEndUnresolvedError` and the `add_*` edge turns it into NaN rows.

### Changed (BREAKING) — `acting_team_attacks_rtl` returns a NULLABLE boolean

The bool path got the same treatment, because the two halves of one cycle should not disagree about
what "unresolved" means. `acting_team_attacks_rtl` previously returned a bare `bool` and
`.fillna(False)`-ed internally, so **a resolved left-to-right team and a team whose direction was
unknown were the same value**. ADR-028 D2 had already added a warning; the value still said
nothing, and 21 call sites inherited the guess. It now returns `dtype="boolean"` with `<NA>` for
unresolved, and every consumer states its choice explicitly:

* `.fillna(False)` where the metric is symmetric under the flip or the path already NaNs the row —
  written, with a reason, at each site.
* REFUSE where a guess would emit a confident number: `add_player_influence` blanks its three xT
  columns (the grid is reflected) while KEEPING `reachable_area*`, which is **exactly** invariant
  under the flip — measured, max |delta| 0.0 across all 20 players of the canonical scene against
  1.17e3 for `off_ball_xt`. `add_space_creation` refuses the whole row instead, because its two
  columns are EXCHANGED rather than degraded.
* The shared action-context nulls the sampled geometry, so all eight kernels behind it inherit one
  decision rather than eight.

Consequences worth stating plainly. On **unoriented** frames the affected geometry is now NaN where
it used to be a confidently wrong number — `resolve_gk_geometry` falls through to its rule-based
prior with the fallback recorded in `*_coord_source`, and the ADR-029 negative control changed from
asserting the defect (bimodal GK x) to asserting the refusal. Oriented frames are unaffected. Three
committed test fixtures turned out to be unoriented and were only passing because of the old guess;
one of them (`test_atomic_add_pre_shot_gk_context`) also labelled its keeper's team as attacking the
end that keeper was standing in, which nothing read.

The change earned its keep immediately by exposing two live defects. `frames["is_ball"].astype(bool)`
in the orientation resolver was the ADR-019 string-qualifier trap (`pd.Series(["False"]).astype(bool)`
is `True`), so `~` selected NO player rows and the resolver fell through for every provider emitting
an object/string `is_ball` — invisible while the fall-through returned all-`False`, since that is
indistinguishable from a legitimately all-home action set. And `_unresolvable_direction_mask` was a
SECOND hand-rolled answer to "is this direction resolvable" that disagreed with the authority in both
directions: it repeated the same `astype(bool)` trap and tested membership with a raw tuple against a
`(game_id, period_id, team_id)` index, which misses silently across dtypes (ADR-055 rule 2). On
numeric actions against string frames it declared every action unresolvable while the authority
resolved all of them. It is DELETED, not repaired — the `<NA>` contract is what makes a consumer-side
re-derivation unnecessary.

**Breaking surface**, across four packages: `home_team_id` removed from `compute_defensive_line`,
`compute_packing_metrics`, `compute_structural_pass_metrics`, `compute_player_influence`,
`detect_line_breaking`; from `add_defensive_line`, `add_packing`, `add_structural_pass`,
`add_line_break`, `add_off_ball_context`, `add_player_influence` and their `*_xfns` factories; and
from the `atomic.tracking` mirrors — **plus the ELEVEN per-Series helpers in `features.py`**:
`defensive_line_x`, `back_line_high_x`, `compactness_x`, `lateral_width`, `max_lateral_gap` and
`back_n_count` (which take `goal_map=None` instead, mirroring `add_defensive_line`), and
`actor_reachable_area_m2`, `off_ball_xt_team`, `off_ball_xt_opponent`, `reachable_area_team`,
`reachable_area_opponent` (which take no direction argument at all, mirroring
`add_player_influence`).

That last group is worth naming as a process finding, not just a list. The scope predicate bounds
the DEFECT — a site is in scope iff it *calls* `same_id`/`ids_match` with `home_team_id` — and the
per-Series helpers call neither; they merely declared the parameter and forwarded it. So they are
invisible to the predicate **by construction**, and the migration sweep, which counted call sites,
recorded `features.py` as needing none. Five of the eleven shipped briefly forwarding
`home_team_id=` into a kernel that no longer accepted it (a `TypeError` on every call) and six
carried a required parameter nothing read. **The scope of a re-key and the scope of its API
migration are different sets, and the second is strictly larger**: enumerate a removed parameter by
signature diff against the base commit, never by the predicate that found the defect. The four map-consuming aggregators gain `goal_map=None` and
build from their own frames (ADR-055 rule 3). `calibration/_features.py` and
`causal/_confounders.py` migrated — the latter builds causal covariates, so
`docs/research/covariate_invariance/` is downstream of it.

**Values change only where they were WRONG, and this is MEASURED rather than reasoned.** On
home-attacks-right frames identity-keying agreed with the map, so those outputs are unchanged --
verified by running every re-keyed aggregator at the pre-re-key commit and at this one against the
same scene: **15 columns across 4 aggregators, all IDENTICAL** (`defensive_line_x`,
`back_line_high_x`, `compactness_x`, `lateral_width`, `max_lateral_gap`, `back_n_count`,
`packing_made`, `packing_net`, `packing_goal_threat`, `packing_secured`, `structural_lbs`,
`structural_sgm`, `structural_sdi`, `line_break`, `n_attackers_behind_line`). Where the frames are
oriented any other way the away-team geometry moves -- from a wrong value to a correct one; that is
the defect, and the direction-invariance test measured it at `defensive_line_x` 23.25 m before the
fix. **No re-materialization is owed for conventionally-oriented frames.**

**Retrain status, stated explicitly because the answer differs by input.** **NOT** a VAEP/tracking
retrain trigger on conventionally-oriented frames — the 15-column measurement above, plus the
committed goldens (`test_packing_golden_identity`, `test_player_influence_snapshot`, the
`gk_geometry` parquet) reproducing byte-for-byte. **It IS one on UNORIENTED frames, including for
`tracking_default_xfns`**, and that reaches further than the six re-keyed sites: four of the default
list's features — `nearest_defender_distance`, `actor_speed`, `receiver_zone_density`,
`defenders_in_triangle_to_goal` — are served by the shared action-context, which now NULLS the
sampled positions for an action whose direction does not resolve rather than leaving them in the
frame convention. **The remedy is to orient (ADR-029 `orient_frames_to_ltr`), not to retrain**: those
values were mis-projected for roughly half the actions before, so a model retrained on them would be
fitting the defect.

Separately, on the real corpus (4 SkillCorner matches, 16 `(game, period, team)` groups, FULL
frames): **0 unresolved**, so no row becomes NaN through the new refusal either.

### Detection — the fix could not have been made to look like success

* **A behavioural invariance test** (`test_d3_direction_invariance.py`) mirrors the FRAMES and
  holds `home_team_id` CONSTANT. Gate A is structurally blind here (it swaps the id too, restoring
  the very invariant identity-keying assumes) and Gate B goes vacuous the moment the parameter is
  removed. This test saw the defect and survives the fix: the ASSERTION is byte-identical across
  the transition. Observed RED first — `defensive_line_x` 23.25, `packing_made` 6.0,
  `packing_goal_threat` 4.0 — now all 0.
* **Gate C** registered for the four map consumers, with **measured** column sets. The two bool
  sites deliberately get NO Gate C: swapping a map they never receive would move nothing, so such
  an entry would pass because its input is ignored. Their detector is the invariance test.
* **The D3 pin is rewritten onto the call predicate** and renamed
  `test_no_module_infers_direction_from_team_identity`, asserting its population is EXACTLY empty
  over eight modules — with a non-vacuity companion proving the predicate catches a planted
  reintroduction, ignores a goal-map lookup, and is blind to the dead-but-declared parameter at
  `_off_ball_runs.py:98` (whose Gate B green IS the measurement that it is unread).
* **Accessor correctness** is pinned separately, because Gate C cannot see it: `get` and
  `attacked_goal` BOTH move under a map swap, so transposing them is a 105 m error Gate C reports
  as success. `_packing` uses both in one function and is now spied on directly.

### Changed (BREAKING) — every DEAD `home_team_id` in the direction family is gone

Separate from the re-key and larger than it in call sites. EIGHT functions carried a
`home_team_id` that nothing read — residue from EARLIER re-keys (ADR-028/041) that removed the
*use* and left the *parameter*. Pulling that thread found a **forwarding chain**: the obso family
carried the argument solely to hand it to `_precompute_obso_lookup`, which ignored it, so the
cleanup cascaded through `obso_actual` / `obso_peak` / `obso_optimal` / `add_obso` / `obso_xfns`,
then through `_run_values_at_actions` to `add_off_ball_run_values` / `off_ball_run_value_xfns`,
`add_pausa` / `pausa_xfns`, `team_shape_xfns` / `shape_graph_xfns`, `calibration.enrich_invariant`,
and the four `atomic.tracking` mirrors. Driven to a **fixpoint** (three rounds): **25 signatures**
-- the 8 dead at the base commit, plus 17 the cascade KILLED on its way up.
Zero dead `home_team_id` is left in the direction family. Cycle total across the re-key, the
eleven per-Series helpers and this cleanup: **62 signatures** lost the parameter, measured by
AST signature-diff against the base commit rather than counted by hand, across 82 source and test
files (13 in `silly_kicks/`, 68 under `tests/`, 1 in `scripts/`).

**No value moves.** Every removed parameter was AST-verified unread before removal, so this is a
signature change only — no goldens shift, no retrain question, no re-materialization.

**Two Chesterton's Fences were checked rather than assumed, and they went opposite ways.**
`add_xt_gk` / `xt_gk_xfns` documented theirs — *"accepted for GK-feature-family signature
parity"* — and it was **measurably stale**: ADR-055 had re-keyed two of that family off the
parameter, so parity meant matching the minority, and specifically matching `add_ghost_gk`, which
actually READS its copy. A dead parameter that makes itself look live is worse than no parameter,
so it went. By contrast `_off_ball_runs_kernel` KEEPS its unread copy — its Gate B green *is* the
standing measurement that the parameter is unread — as does
`_compute_space_creation_for_action`, the case CLAUDE.md records as *"D3 retires it by disuse, not
removal"*. Those two are the reason `add_off_ball_runs` and `add_space_creation` also keep theirs.

Left alone, and named so the next reader does not re-derive it: four functions outside the
direction family (`causal.join_layer2_confounders`, `_xcross_eval.gk_substitution_probe`,
`_xshot_occurrence.prepare_xshot_training_data` / `compute_xshot_occurrence`) plus two `@overload`
stubs whose `reads=0` is a property of having no body.

### Fixed — the lint gate's verdict was a function of dependency-resolution luck

The lint job exact-pins `ruff`, `pyright` and `pandas-stubs` but let **numpy** float, and numpy is
a typing input to pyright in exactly the way pandas-stubs is -- it ships inline types. Measured on
two runs a day apart: main's lint resolved numpy 2.5.2 and then DOWNGRADED to 2.4.6, while
PR-S150's stayed on 2.5.2, and the seven pyright errors that difference produced sat in three files
the PR's diff could not reach. A gate that goes red with no diff -- and green again on a re-run --
teaches everyone to re-run it.

Pinned on the SECOND install, and that placement is the property: the first install sets the tools,
but `pip install -e ".[test]"` re-resolves and is what actually moves numpy, so a pin on the first
line would be visibly present and bind nothing. `tests/test_ci_lint_pins_wired.py` asserts both the
presence and the placement, and was mutation-tested against all three broken arrangements (pin on
the first install only; pin absent; test matrix pinned too) before being trusted.

**The TEST matrix stays deliberately unpinned, and that asymmetry is asserted too.** There an
unpinned numpy/pandas is COVERAGE -- ADR-057's span, which caught this very release's pandas-3
defect -- so someone "tidying" the two jobs into consistency would silently delete it. Typing
inputs and behavioural inputs want opposite policies.

### Fixed — three defects the re-key introduced

* **The `<NA>` blanking wrote into a READ-ONLY array on pandas 3.** `pitch_control_at_target` does
  `qx = _q["_qx"].to_numpy(dtype="float64")` and then blanks the unresolved rows; under pandas 3's
  copy-on-write `.to_numpy()` hands back a read-only VIEW when no dtype conversion is needed, so
  every affected call raised `ValueError: assignment destination is read-only` — **48 tests on the
  3.11 leg**. pandas 2 returns a writable array, so a local suite on 3.10 passes all 7209 and sees
  none of it. Fixed with an explicit `copy=True`. This is ADR-057's span earning its keep for the
  second time (the first was DAS going silently all-NaN on pandas 3), and the sweep was done by
  predicate rather than by patching the one line: an AST search for "a `to_numpy()` result later
  mutated" finds exactly four sites repo-wide, two of them this one's pair.


* **`add_packing` crashed instead of refusing.** Its `GoalEndUnresolvedError` fallback built the
  three EMITTED columns and not `line_x`, which the event-only assembly reads on the very next
  line, so an unresolvable goal end surfaced as `KeyError: 'line_x'` rather than the NaN row
  ADR-055's edge policy specifies. Found by the SB360 audit's `gk_absent` roster — the one scenario
  with no keeper at either end. The rule the sibling `add_defensive_line` catch already followed: a
  fallback frame must carry every column the code AFTER the `try` reads, not just the ones the
  aggregator emits. The audit's `NOT_EXERCISED_BUDGET` rises 45 → 49 as a result, and that rise is
  an honest loss of comparison rather than a regression: those four cells previously read
  `identical` because identity-keying always answers, i.e. the old reading was a number obtained by
  guessing a side.
* **`add_defensive_line` silently lost its `@nan_safe_enrichment` decorator**, displaced onto the
  private `_nan_frame_for` when `_goal_map_for` was inserted between them — an ADR-003 contract
  break. The decorator COUNT in the file was unchanged at 31, so nothing that counts could see it;
  it was found by diffing the decorated-function SETS against the base commit. Counting is not
  identity.

**A near-miss worth recording.** `packing_goal_threat` was nearly dropped as a dead column — it is
constant `0` on the base leg. But `0` is the CORRECT answer there (the bypassed players are not in
the back line), and flipping the end moves it to `[4, 1, 1, 1]`. It is the ONLY witness for
`_packing.py`'s back-line site, so dropping it would have left that site unwitnessed while Gate C
reported success — a partial re-key reading as green. **A detector's liveness is not "does it vary
across rows" but "does it move when the thing it detects changes".**

## [4.79.0] — 2026-08-11

### Fixed — the build backend was unbounded, so the published artifact was a function of wall-clock time

**No library change; release plumbing only.** The first `v4.79.0` publish attempt failed after a
green build: `InvalidDistribution: Invalid distribution metadata: '2.5' is not a valid metadata
version`. Nothing in this release caused it, and the window was under four hours.

`pyproject.toml` declared `requires = ["hatchling"]` with **no upper bound** while every GitHub
Action is SHA-pinned — so the producer floated and the validator was frozen. hatchling **1.32.0**
landed on PyPI at **05:03Z**; `v4.78.0` had built at **01:39Z** against 1.31.0 and published fine,
and `v4.79.0` built at **12:56Z** against 1.32.0. Measured, not inferred: 1.31.0 emits
`Metadata-Version: 2.4`, 1.32.0 emits `2.5`, and `pypa/gh-action-pypi-publish` **v1.14.1** pins
`packaging==25.0`, whose valid set stops at 2.4 (**v1.14.2** pins `packaging==26.2` + `twine==7.0.0`,
which accept 2.5).

Fixed on **both** sides, because each alone leaves a gap. The action is bumped to **v1.14.2**
(bumping *forward* — metadata 2.5 is legitimate, and pinning the backend below it would only defer
the same failure), and the build backend is **bounded** to `hatchling>=1.27,<2` (not pinned: the
point is that a backend release shows up in a diff rather than in a failed publish). Guarded by
`tests/test_ci_publish_guard_wired.py`, which rejects any unbounded `[build-system]` requirement and
carries a companion test pinning that the predicate rejects the exact `"hatchling"` string that
broke this release — a bound-checking guard passes vacuously if its predicate says yes to everything.

### Fixed — `detect_input_convention` rule 1 concluded from evidence that could not discriminate (ADR-059)

Rule 1 infers `POSSESSION_PERSPECTIVE` from *"every reliable (match, team, period) group attacks
high-x"*. `reliable` is filtered to `n >= min_shots_per_group_medium` (5) — and when that filter
removes the low-side groups, an all-high survivor set is an artifact of the filter rather than a
measurement.

**Measured on real Gradient Sports data, wrong on 2 of 36 matches.** Match 10502: team 51 goes
`P1 13.3 LOW (n=3) -> P2 94.0 HIGH (n=8)`, team 366 goes `P1 95.8 HIGH (n=7) -> P2 9.8 LOW (n=3)`.
Both teams shoot at opposite ends within a period and SWAP between them — textbook
`PER_PERIOD_ABSOLUTE`, which `gradientsports.py` correctly declares. Both LOW groups fall below the
threshold, and rule 1 returned `POSSESSION_PERSPECTIVE` at `confidence="medium"`, contradicting a
correct declaration.

**The fix recorded in `TODO.md` would not have worked.** It diagnosed *"the rule fires on
effectively ONE team's data"* and prescribed *"require >= 2 distinct TEAMS"*. Measured: the
survivors are team 51 (P2) and team 366 (P1) — **two** teams, so that guard permits the misfire
unchanged. It would have reviewed clean against its own rationale and shipped with the defect live.
Reproducing the failure before implementing is the only reason it did not.

Rule 1 now requires a configuration an absolute convention could not have produced — **two distinct
teams reliable in the same period**, or **one team reliable across two periods**. Otherwise it
returns `convention=None` with a diagnostic naming why. Deferral is the safe direction:
`validate_input_convention` reads `None` as "keep the caller's declared convention", so a false
ambiguous leaves output correct and loses only a cross-check, while a false positive contradicts a
correct declaration and **raises** under `on_mismatch="raise"`.

Clause (b) **is** the guard TF-22 added inline to the ABSOLUTE branch in 3.0.1, for the same reason
against the same sparse-asymmetric shape; rule 1 never got it. Both branches now call
`_a_team_spans_periods` — one spelling, not two. Clause (a) is deliberately NOT given to the
ABSOLUTE branch: separating `ABSOLUTE` from `PER_PERIOD` requires observing a team ACROSS periods,
so a single shared "is this discriminating?" helper would have silently loosened TF-22.

**Coverage.** Rule 1 is what validates StatsBomb and SkillCorner as `POSSESSION_PERSPECTIVE`, so
tightening it risks a silent downgrade to ambiguous.
`test_statsbomb_raw_detected_as_possession_perspective` already asserts this on **3 real StatsBomb
matches** and passes — a standing gate, worth more than a one-time corpus run that confirms today
and rots tomorrow. **SkillCorner is guarded too**, by a new
`test_skillcorner_raw_detected_as_possession_perspective` on real public match `1886347` — one of
the ten `PUBLIC_CORPUS` registers as redistributable. That gate was nearly not built: a first
reading took "SkillCorner" to mean owner-tier, which is precisely the provider-name-as-tier-proxy
inference `scripts/_corpus.py` exists to prevent and whose docstring says visibility is keyed
per-match, *"NEVER on the provider name"*. The fixture is deliberately a HARD case — `(team 1805,
period 1)` has 3 shots and is dropped by the same `>= 5` filter that caused the misfire, and the
match still classifies because the survivors discriminate — so it demonstrates the guard does not
over-tighten on real data carrying the defect's own shape. `is_shot` is re-derived from the
converter's own `end_type == "shot"` rule rather than baked into the file, a companion test asserts
the below-threshold group is still present, and the gate is mutation-verified: force both
discriminating predicates false and real SkillCorner data detects as `None`.

**No converted data changes — for ANY provider, including the two affected Gradient Sports
matches.** The detector is a cross-check, not a code path: `to_spadl_ltr` orients on the converter's
DECLARED `input_convention`, and `validate_input_convention` only warns or raises on a mismatch,
never overriding the declaration (`orientation.py`'s own module docstring: *"It never overrides the
declared convention — the converter's declared `input_convention` is the load-bearing contract"*).
So what changes is which inputs get a spurious warning or a spurious `on_mismatch="raise"`, not a
single coordinate. No retrain, no re-materialization.

Beyond that, no verdict change for any provider whose data carries a discriminating configuration;
the ABSOLUTE branch is unchanged.

### Changed — the tracking frames schema declares a NULLABLE id dtype (ADR-058)

`TRACKING_FRAMES_COLUMNS["player_id"]` and `["team_id"]` were `int64`. Every frame set carries a
ball row, which belongs to no team and holds no player, so both are NA on it **by construction** —
and numpy `int64` cannot represent NA. So casting a real frame set to the declared schema, which
4.77.0 attempted, raised `IntCastingNaNError`. Always, on every producer.

4.77.0/ADR-055 met that failure and read it as its planned dtype PIN being *unimplementable*. It was
the DECLARATION. All five provider variants already overrode these two columns — four to `object`,
Gradient Sports to `Int64` with a docstring that literally says *"allows NaN on ball rows"* — so the
base was satisfied by **nothing** and described one producer's happy path. A constant every variant
overrides is a default masquerading as a contract.

Base is now `Int64`. `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS` becomes an **alias** of it rather than
an override — kept as a name, not deleted, because it is exported in `silly_kicks.tracking.__all__`
and aliasing is already this file's idiom (`SPORTEC`, `SKILLCORNER` and `METRICA` all alias
`KLOPPY`). Six declarations collapse to two honest dtypes plus four aliases. Not `object` for the
base: `id_compat`'s both-object path is CONTENT-probed (~15% per side) because boxed floats
raw-compare False against the same id as a string.

Gated by two tests in `tests/test_tracking_schema.py`, both landed RED: one pins the base, and one
is complete by ENUMERATION over every `*_TRACKING_FRAMES_COLUMNS` in the module, so a future
provider added with a non-nullable id dtype fails CI.

**Behaviour surface:** the only site that casts to the BASE at all is `_snapshot._empty_frames()`,
which builds an EMPTY frame — so the broken declaration was reachable only in the one case with no
NA to fail on, while the populated path (`_snapshot.py:172`) selects the 20 columns without applying
dtypes. Empty and populated snapshots therefore disagreed about `player_id`'s dtype and nothing said
so. The empty frame's `player_id`/`team_id` move `int64` → `Int64`; provider adapters cast to their
own variants and are unchanged (Gradient Sports was already `Int64`, the four object variants are
untouched). No retrain, no re-materialization.

### Added — CI's pandas span is DECLARED and asserted, not inherited (ADR-057)

`TODO.md` recorded, of the snapshot dtype question, that *"the concern is only checkable on a
pandas-3 environment, which CI does not have (`ci.yml` is OS x Python only)"*. **That was false.**
`pyproject.toml` pins `pandas>=2.1.1,!=3.0.4` with no upper bound and pandas 3 requires Python
>= 3.11, so pip resolves the newest compatible pandas per interpreter. Measured on run
`31316804815`: `ubuntu-3.10` → **2.3.3**; `ubuntu-3.11`, `ubuntu-3.12`, `windows-3.12` → **3.0.5**.
Three of four test legs already ran pandas 3.

The coverage was real but **ACCIDENTAL**, and that is the actual defect: nothing declared or
asserted it, so it could vanish with no diff and no signal. This repo has one measured instance of
that class already (DAS going silently all-NaN on pandas 3).

Two halves, neither sufficient alone — a test running in one leg cannot observe another:

- `tests/test_ci_pandas_span_wired.py` asserts the **resolved leg set** still straddles the Python
  3.11 boundary. It parses os × python-version **minus `exclude` plus `include`**, never the axis:
  `exclude` is already the pruning mechanism in use, and excluding `ubuntu/3.10` collapses the span
  while leaving `"3.10"` in the axis, which an axis-based assertion passes. Observed RED against
  exactly that mutation.
- A `pandas-span` job (`needs: test`) aggregates each leg's recorded pandas major and fails unless
  the union spans both. Its script was extracted from `ci.yml` and executed against fabricated
  artifact trees: `{2,3}` → pass, `{3}` → fail, `{2}` → fail, none → fail.

The guard asserts the SPAN, not specific versions, so a routine dependency bump does not train a
reader to edit the expectation without thinking.

### Added — `snapshot_to_tracking_frames` id comparability, pinned across both pandas majors

`tests/tracking/test_snapshot_id_dtype_across_pandas.py` asserts the property consumers actually
depend on per ADR-019 — that `id_compat` comparisons keep working — rather than a dtype literal,
which is precisely what left the question unverifiable for two cycles. Run on pandas **2.3.3** and
**3.0.5**. The dtype does diverge (`Int64` source → `Int64` on 2.x, `Float64` on 3.x; numpy-int →
`float64`, object → `object`, both majors), and the behavioural contract holds identically on both,
so no xfail was needed. Also pins that the synthesized ball row's ids stay **NA** rather than
becoming a sentinel (ADR-027), observed RED against a `fillna(0)` mutation.

### Added — the `gk_one_end` SB360 visibility roster reclaims cover-shadow coverage (ADR-053)

`gk_absent` removes BOTH keepers, so `resolve_defended_goals` falls to its outfield rung and guesses
both teams at x=105; a both-teams-same-end map is degenerate, `attacked_goal` refuses it, and
`add_cover_shadows` emits NaN on both legs for a roster-driven reason — no informative row survives.

`gk_one_end` **adds** a third roster rather than widening `gk_absent`, which is a real visibility
axis (keeper outside the observed region) and the only case exercising the both-absent refusal path.
One keeper visible: the visible team resolves to x=0, the other falls to the outfield rung and
guesses x=105, the ends differ, the map is non-degenerate. Measured on the fixture:
`resolved={('7','1','1'): 0.0}`, `guessed={('7','1','2'): 105.0}`, `unresolved=frozenset()` — keys
are `(game, period, team)`. It is also the better-supported case: per the committed coverage report
the DEFENDING keeper is in-frame on **92.2%** of shots, so a freeze-frame with a keeper present is
the common one, and `gk_absent` alone left that majority shape unexercised. (The spec justified this
with a comparative — defending keeper in-frame *"while the acting-side keeper usually is not"* — that
`coverage.md` does not support: its `acting GK` cell for `shot` is `—`, i.e. definitionally not
applicable because the keeper is not the actor on a shot, not a measured lower rate. The report
measures nothing about the far keeper. The roster's justification stands on the 92.2% alone.)

**All five columns reclaimed** (`not_exercised` → `honest_nan`): `n_blocked_receivers`,
`n_potential_receivers`, `blocking_score`, `blocked_threat_fraction`,
`max_single_defender_blocking_score`. Asserted in `test_registry_surface.py`, not left as a task
note. `max_single_defender_player_id` stays unexercised for an unrelated reason (no pressing
sequence in the fixture).

`NOT_EXERCISED_BUDGET` **31 → 41**. It counts `(entry, axis, roster, column)` tuples over 35 entries,
so a third roster can only ADD — a drop was arithmetically impossible, and two earlier drafts of the
success metric were invented to match the claim instead of read off what changes. All 10 new tuples
are enumerated to three causes in the constant's docstring. `gk_absent`'s existing slice is pinned
byte-identical across the change (165 verdicts, floor-asserted before the diff is trusted — a `diff`
of two empty captures succeeds, so a pin that can pass vacuously is not a pin).

`columns_exercised_on_no_roster()` lands as a standing regression pin over columns unexercised under
EVERY roster. It registers **zero change** from this cycle, which is the correct expectation: the
five were already `honest_nan` under `defender_absent`, so they were never in that set.

### Fixed — an unregistered SB360 adapter silently emptied an aggregator's verdicts (4.77.0)

`tests/sb360/_calls.py` defines `visible_area_coverage(fn)` but never registered it in `ADAPTERS`,
so `_regenerate.py` fell back to `C.generic`, raised `TypeError`, and the probe's `except Exception`
swallowed it into `cols = ()` — emptying every roster block for `add_visible_area_coverage`.
Committed verdicts stayed correct, so CI never failed; only a REGENERATION surfaces it. Caught by
the `gk_absent` pin (165 → 163) and fixed by registering the adapter (re-verified 165 == 165).

**The swallowing handler was the mechanism, and it is no longer silent.** Three changes, because a
printed warning scrolls past:

1. Probe failures are **collected and reported** at the end of the run, naming each aggregator, its
   exception, and the likely cause.
2. `_regenerate.py` now **exits non-zero** when any probe failed. The entries are still written —
   you need them to diagnose — but the run is not clean and the exit code says so.
3. A CI gate, `test_every_aggregator_emits_at_least_one_column`, fails on a registry containing an
   entry with `columns=()`. This is the durable protection: the first two only help someone watching
   a hand-run regeneration, while this one fails the build.

The `except` itself stays BROAD deliberately — an aggregator that legitimately refuses this fixture
must not abort a 35-entry regeneration. Narrowing it was considered and rejected as the wrong lever:
it trades one silent-failure mode for a brittle exception allowlist, whereas the gate catches the
*consequence* (an uncovered aggregator) regardless of which exception produced it.

Both defects are repaired, so the gate could not be landed red against a live instance. It was
**mutation-verified**, which ADR-051 permits explicitly: doctoring `add_xcross_attempt` back to
`columns=()` fails the assertion naming it.

**Making it loud immediately found a SECOND instance, live in the committed registry.** Verified by
reintroducing the 4.77.0 state (deleting `visible_area_coverage` from `ADAPTERS` and running the
real regenerator), the report named two aggregators, not one:

```
!! PROBE FAILED for 2 aggregator(s) -- each regenerated with ZERO columns ...
    add_visible_area_coverage: TypeError: add_visible_area_coverage() takes 1 positional argument ...
    add_xcross_attempt: KeyError: 'vx'
```

`add_xcross_attempt` was the only entry in the registry with `columns=()` — its `velocity` block and
all three `visibility` blocks **empty**, so ADR-053's "every `add_*` carries an SB360 freeze-frame
verdict" was not true of it and nothing said so. Cause: it read `frames["vx"]` unguarded and raised a
bare `KeyError` on a frame set that explicitly declares `speed_source="unavailable"` — the marker
ADR-054 introduced precisely so a builder can DECLARE that its source has no temporal history.

### Fixed — `add_xcross_attempt` honours the velocity-availability contract (ADR-054)

It now implements the same two-pronged contract as `_das`, `_ghost_gk` and `_press_commitment`,
because the two shapes are byte-identical at the seam and demand opposite responses:

- **declared unavailable** → degrade to NaN. `ball_speed` is a trained feature, so scoring would
  have the model impute an input its source structurally cannot carry — the ADR-053 fabrication
  shape. NaN is already what this function returns on every other unscoreable path.
- **not declared and `vx`/`vy` absent** → raise a `ValueError` naming the remedy (`derive_velocities()`,
  or declare `speed_source` unavailable). Previously a bare `KeyError: 'vx'`, which names nothing
  actionable and which an upstream handler silently reinterpreted as "this aggregator emits nothing".

The guard sits in `compute_xcross_attempt` — the **shared seam** all three public entry points
(`add_xcross_attempt`, `xcross_attempt_xfns`, a direct call) reach scoring through, the same reason
the ghost guard lives in `_serve_positions_core`. Separately, `extract_xcross_features` — itself
public, and documenting itself "NaN-tolerant" with a NaN pre-fill — now leaves `ball_speed` at that
pre-fill instead of raising, while still computing the positional ball features that do not depend
on velocity. A zero-fill was rejected as worse than either: a fabricated stationary ball fed to a
trained feature.

Four tests landed RED (`tests/tracking/test_xcross_attempt_velocity_contract.py`), including a
non-vacuity control asserting a fully velocity-bearing frame set still scores — without it,
returning all-NaN unconditionally would satisfy every other assertion.

**No provenance column was added, and that is a decision, not an oversight.** `add_ghost_gk`,
`add_das` and `add_press_commitment` each emit one (`ghost_gk_source` and friends), so the house
pattern would suggest an `xcross_attempt_source`. It is declined here because
`compute_xcross_attempt` already returns NaN for four other unscoreable reasons — no possession, no
defending team, an unresolvable defended goal, no linked frame — none of which carries provenance. A
column that named only the velocity cause would imply the other four NaNs were scoreable, which is
worse than silence. Emitting a vocabulary covering all five is a real schema addition (glossary
entry per ADR-048, liveness registration, purity variants, its own SB360 verdicts) and belongs in a
change that decides the whole vocabulary rather than riding along with a crash fix.

**Audit consequences, both recorded rather than absorbed.** The aggregator now probes cleanly and
carries real verdicts, which moves two pins in the *honest* direction:

- `NOT_EXERCISED_BUDGET` **41 → 45**. The 4 new tuples are `xcross_attempt` on all four axes, and
  they are coverage that was **already missing and is now visible** — not a regression. Both legs
  score NaN, and not because of the repair: the velocity-bearing leg is NaN too, because the fixture
  stages no in-possession wide-area cross for the model to score.
- `columns_exercised_on_no_roster` gains its **only** new member this cycle,
  `('add_xcross_attempt', 'xcross_attempt')` — same class as the existing `add_xshot_occurrence`
  entry: a fitted model over a domain this fixture does not produce.

The regenerate → adjudicate round trip was verified to reproduce every other entry **byte-identically**,
so the diff is confined to this one block. No gate forbidding zero-column entries is added: the
condition it would catch is now absent, and a gate written after its own repair arrives green and is
never observed failing (ADR-051).

### Fixed — `add_elastic_sync` scored every action against an EMPTY distance lookup (ADR-019)

Found by making `snapshot_to_tracking_frames` honour its schema: casting the ids broke five SB360
verdicts, and the cause was not the fixture.

`elastic_confidence` was a constant **0.6 on every row, on both legs** — which is exactly
`accel_weight / (accel_weight + proximity_weight)`, the value you get when the proximity term
contributes nothing at all.

The player-ball distance lookup keyed on `merged["player_id"].astype(str)` while the query keyed on
`str(action_row["player_id"])`. Every frame set carries a ball row whose `player_id` is NA, which
upcasts an integer id column to `float64`, so the two sides rendered the same id differently:

```
LOOKUP key:  (7, 1, 0, '10.0')      # frames side
QUERY  key:  (7, 1, 3, '10')        # actions side
```

**Every lookup missed.** `dist` fell to the caller's `inf` default and `proximity_score` to zero. A
miss is indistinguishable from "infinitely far from the ball", so nothing failed loudly — the
scoring loop ran to completion and produced a plausible number from a term that had not been
computed.

**Not snapshot-specific.** A plain `python_int` id column fails identically, because the NA ball row
upcasts it too. Any caller whose frames carried integer player ids was affected; only genuinely
STRING ids worked, which is why no provider test caught it.

**The audit had recorded the broken state as `works`.** Both legs degraded the same way, so the
observation was `identical` — a one-sided check cannot see a defect that breaks both arms equally.
The registry had even recorded the fingerprint: `applicability` was `no_support` with BOTH probe
deltas exactly `0.0`, which is precisely what `applicability_deltas` exists to make visible. It now
reads `support_data_defined` with deltas `{extreme: 0.0004, near: 0.1839}`.

Fixed at the ROOT, not at the symptom: the cast masks it only for the snapshot path, so both sides
now go through `canonical_id_series` / `canonical_id`, which collapse `10` / `10.0` / `Int64(10)` /
`"10"` to one rendering. Four tests landed RED across all four id dtypes.

This is the trap CLAUDE.md already names — *"`str(5.0)` iterrows-upcast player-influence/cover-shadow
mislabel"* — in a module the ADR-043 enumeration registry does not reach, because that registry
covers id-SCALAR arguments of public functions and this is an internal dict key. **The registry's
completeness is over a surface, and this defect lives off it.**

**Consequences, scoped by enumeration rather than asserted.** `elastic_sync_xfns` appears in **no**
default xfn list — checked by enumerating all seven `*_default_xfns` at runtime, not by grepping —
so this is **NOT a VAEP retrain trigger**. It IS a value change for anyone who opted the factory in
explicitly, or who reads `elastic_confidence` / `elastic_frame_id` / `elastic_error_seconds` from
`add_elastic_sync`: those values were previously computed with the proximity term contributing
nothing, so any stored column is wrong and should be re-materialized. `elastic_frame_id` moves too —
the alignment was choosing its best frame on acceleration alone.

### Fixed — the SB360 adjudicator asserted pitch-control semantics about non-pitch-control modules

`_adjudicate.py`'s fallback was `return R["window"] if cause == "frame_count" else R["positional_pc"]`,
so any module that is not pitch-control-derived and whose cause is not exactly `frame_count` had a
rationale about *"pitch control evaluated at zero velocity"* attached to it. Measured on
`add_elastic_sync.elastic_confidence`, which computes no such quantity.

A rationale is the part of an ADR-053 verdict a human is asked to ARGUE WITH, so one describing the
wrong mechanism is worse than a vaguer one that is true. The fallback is now an honest
`mixed_cause` rule that states what the probe actually established — both legs compute from inputs
they hold, the probe could not attribute the change to a single cause, and the two numbers must not
be compared as though they were the same measurement.

### Fixed — the velocity-fixture discriminator was measuring the wrong quantity

`scripts/audit_velocity_fixtures.py` (new) decides which test fixtures declaring
`speed_source="native"` without `vx`/`vy` actually matter. A grep finds 24 candidates; **that is not
a defect count**, and the script exists so nobody treats it as one.

Its first revision contrasted a stationary frame set against a moving one — velocity **MAGNITUDE**.
The defect class is velocity **PRESENCE**: declared available, vector absent, extractor yields NaN,
fitted model routes it down a learned missing-value branch. 4.76.0 measured exactly that
(`NaN → [6.795, 33.522]` vs zero-fill `→ [6.888, 33.362]`).

Under the magnitude contrast `add_ghost_gk` — THE consumer of ADR-053/4.76.0 — scored a 0.0 delta
twice over and was filed **velocity-blind**: the sb360 leg-A fixture declares
`speed_source="unavailable"`, so the by-design marker (an ALL-rows predicate) returned NaN on both
arms, and an all-NaN column contributes no numeric delta. **Measured: the positive control shows the
old instrument CLEARING a reconstruction of the very fixture ADR-053 convicted.**

Corrected to absent-vs-present, with `speed` held CONSTANT across the arms so the contrast isolates
the vector. That second correction mattered: with the PRESENT arm also overwriting `speed`, the
instrument convicted `tests/tracking/test_add_action_context.py` and its atomic mirror — measured,
`add_action_context` is unchanged by `vx`/`vy` at fixed speed and reads only the scalar, so the
delta was entirely the probe moving `speed` underneath them.

Verdicts are now three-way (`sensitive` / `refuses` / `blind`): a consumer that RAISES on
declared-but-absent velocity cannot silently fabricate, and folding it into `blind` is what produced
the false clear. A column flipping between all-NaN and populated is reported as `nan_flipped` rather
than folded into the delta, because `NaN - x` is NaN and never enters a max-abs-difference.

**Result: 0 convicted, 24 cleared.** 7 consumers measure sensitive, 4 refuse; no candidate file calls
any of them. Every one of the 24 is correct as written — *"we checked and it did not matter"* is a
finding, not an absence.

**And it is pinned, not just reported.** `test_no_test_fixture_claims_velocity_and_reaches_a_sensitive_consumer`
re-derives the intersection every CI run, so a fixture added tomorrow with the same shape — declaring
velocity, supplying none, calling one of the seven sensitive aggregators — fails the build instead of
quietly re-creating the 4.76.0 defect. It deliberately pins the **intersection being empty**, not the
candidate count: locking "24" would fail on any unrelated test file mentioning `speed_source` and
train a reader to bump the number without thinking. Mutation-verified by flipping
`add_action_context` from blind to sensitive, which convicts exactly the two files that call it; and
it carries its own non-vacuity guard, because a mis-resolved scan root would otherwise report zero
candidates as a clean bill of health.

`tests/scripts/test_audit_velocity_fixtures.py` is the instrument's control, deliberately SPLIT. The
plan sketched it as *"the reconstructed pre-4.76.0 ghost fixture yields `value_changed is True`"* —
which couples a gate to a defect CONTINUING TO EXIST, the same mistake this repo already shipped
once when `test_at_least_one_column_was_adjudicated_a_fabrication` broke because the fabrication it
asserted had been repaired. Instead: a PLANTED case proves the engine can still convict; a companion
proves the historical fixtures are SURFACED rather than cleared, without asserting which side of the
sensitive/refuses line they fall on; a false-POSITIVE guard pins that the two probe arms differ only
in the vector (observed RED against the contaminated probe — a positive control cannot catch a false
positive, since a contaminated probe surfaces every known positive while convicting the innocent).

## [4.78.0] — 2026-08-10

### Added — artifact input contracts and registry completeness (PR-S147, ADR-056)

**A research artifact recorded `run_commit` — *when* it ran, never *what its numbers depend on*.**
`scripts/_input_contract.py` adds `declare_inputs()`, which records the SYMBOLS a driver's numbers
depend on (covariate tuples, extractor module identity, `GEOMETRY_VERSION`) and digests them. When
`SHOT_ARM_CONFOUNDERS` gains a column the digest moves without anyone editing the driver. Four
drivers declare one and four artifacts now carry it. Warn, never raise — an artifact is not a
serving path. Same shape as ADR-050's `feature_contract`.

**Declared limit, not discovered: this catches code drift, not under-declaration.** The mechanism
names MODULES, so a behavioural change inside one does not move the digest. The concurrent
`sb360-degradation-and-port` ghost-refusal change is a live near-miss for exactly that — it cannot
affect GS, so it is not the escalation trigger, but it is the clearest evidence of where that
trigger will come from.

**Four registry floors replaced by derived populations.** `assert len(ARTIFACT_DRIVERS) >= 6` passed
at 18 entries while `validate_xcross_causal` was absent and its artifact carried no provenance at
all. Each gate now derives its population structurally and asserts it EXACTLY, both directions, with
a three-bucket exemption shape whose `_UNDERIVABLE` bucket is asserted EMPTY — the blind spot the
other two cannot see. Landed at 22 enrolled drivers, 5 reasoned exemptions, `_UNDERIVABLE` empty,
and a `SWEPT` registry of exactly 19 default xfn lists (previously three COPIES of one discovery
rule behind `>= 10`).

**Five live defects found by the new gates:** three artifact drivers with no provenance guard at all
(`calibrate_tracking_defaults`, `validate_shot_goalmouth_sb`, `validate_xtgk_possession_value`), a
real ADR-020 dup-`action_id` crash in the three pressure kernels, and `xshot_occurrence_xfns`
missing from `atomic.tracking.features.__all__`.

**Provenance had FOUR conventions, not two.** Measured across `docs/research/**/*.json`: top-level
`run_commit` (12), nested `_provenance.commit` (2), `training_commit` (the bundled weights, outside
`docs/research/`), and nothing at all (3) — plus one artifact that is not a JSON object. The gate
picks one canonical shape per surface and records divergences WITH the location of their real
provenance, rather than back-stamping artifacts to fit a shape invented after they were produced.

### Fixed — the GS fixture generator could not reproduce its own fixture (PR-S147, ADR-056)

`tests/datasets/gradientsports/_generate_synthetic_match.py` emitted **51** events against a
committed **54**. The three missing ones were the ADR-018 goal-capture events — the RE+G own goal,
the CR+G cross-goal and the `nonEvent` disallowed shot — appended directly to the JSON when goal
capture landed, with the generator never updated. **Any regeneration silently deleted all three**,
including the two that `test_owngoal_crossgoal_captured_disallowed_excluded` asserts. Invisible
because nothing in CI ever runs the generator.

Repaired by reproducing all 54 **verbatim first**, artifacts included (a truncated `ball` dict, a
raw-clock `startTime`, a `startFormattedGameClock` of `"00:12"` at 1000 s, and a trailing newline
`main()` never wrote). Byte-identity against the git blob was the acceptance evidence: a generator
that "improves" the artifact in the same change that restores it cannot show which of the two edits
moved the file.

**Then reshaped so CI can see the case it governs.** The fixture carried shots from ONE team only,
so `detect_input_convention` returned `convention=None, confidence="low"` and the converter's
`validate_input_convention` deferred silently — the guard ran down neither branch. The binding
constraint was never per-group shot count (the fixture already had 10 shots in one group, AT the
`high` threshold). Reshaped to the measured distribution (`docs/research/gs_input_convention/`, 64
matches): both teams on opposite ends within a period, swapping between periods; five shots per
group, because only 6 of 64 real matches reach two `high`-reliable groups while 50 of 64 reach two
at `medium`.

    before   convention=None       confidence=low     (deferred)
    after    PER_PERIOD_ABSOLUTE   confidence=medium  (classifies, and AGREES with the declaration)

Purely additive: all 54 prior events are byte-unchanged and keep their relative order.

### Fixed — a synthesized GS row inherited its parent's derived end coordinates (PR-S147, ADR-056)

**This changes GS conversion values.** Both synthesis sites `.copy()` a pass-class parent AFTER
`_derive_end_coordinates` has rewritten that parent's end to the next action's start, then relabel
the copy `foul` or `shot` — neither of which is in `_DERIVE_END_TYPE_IDS`, i.e. both keep
`end == start` by contract. The copy therefore carried a pass-class destination on a type that must
not have one, reading downstream as a shot or foul that travelled.

Measured on the committed fixture: **all three** synthesized rows were affected. The two synthesized
fouls had been wrong since the row was introduced — their parents are mid-period passes, so a next
action always existed — and were invisible because
`test_shots_tackles_keeper_saves_end_equals_start` does not include `foul` in its type set. The
cross-goal shot read correctly only by accident: its parent was the last period-1 event surviving
exclusion. The input-convention reshape exposed it.

Single-sourced onto `_reset_synthetic_end`. **Real (non-synthetic) rows are byte-identical**, and the
regression test asserts the parent cross still carries its derived end, so derivation is not disabled
wholesale. Small-N but corpus-wide: in real data a cross-goal is essentially never period-last, so
the shot case was wrong every time. A GS consumer persisting `end_x`/`end_y` for synthesized rows
should re-materialize; nothing else changes and no model retrain is triggered.

## [4.77.1] — 2026-08-09

### Fixed — the SB360 coverage driver's shard generation, and two polygon-degeneracy conflations (PR-S146, ADR-055)

**The shard schema moved without its generation token.** 4.77.0 renamed `measure_match`'s
`mean_visible_pitch_fraction` to `mean_observed_pitch_fraction` and added `n_with_polygon` (the
ADR-042 denominator), but left `token_inputs["schema"]` at `"sb360-coverage-2"`. `for_each`
fingerprints `token_inputs` only — never the source — so the un-bumped token resolves to the SAME
generation directory. Measured: 22 stale shards carrying the old column were sitting in it. A
re-run would have skipped all 22 as already-done, reported a clean conserved pass, and combined the
OLD denominator — the ADR-042 fix silently not taking effect. This is the exact silent-dilution
ADR-052 exists to prevent, and the `schema` token is its manual defence.

Nothing in the suite could see it: the driver's own assertion on the renamed column lives in the
`e2e` test, which CI does not run — so CI was green on a broken driver. Fixed by bumping to
`sb360-coverage-3` and by pinning the pair: `_EMITTED_SHARD_COLUMNS` + `_SHARD_SCHEMA_VERSION` in the
driver, three gates in `tests/scripts/test_build_sb360_coverage.py` (the declaration matches the
dict `measure_match` actually builds, `token_inputs["schema"]` references the constant rather than
a literal, and the pair is pinned), plus a run-time assertion that fails at the FIRST shard. All
four defect reintroductions verified RED.

**The artifact was then re-run, and it is NOT stale.** A full 22-match pass on the corrected driver
reproduces every published "visible pitch" value at its published 2-decimal precision (max absolute
change 0.004). Two edits move that column in opposite directions and the decomposition explains why
the net is nil: `n_with_polygon` is **100.0% of `n_events` for every action type** — every SB360
freeze-frame in this corpus carries a `visible_area` — so the ADR-042 denominator change is a
**no-op here**, and the clipping introduced with `observed_pitch_fraction` moves values by at most
0.004 (`shot_penalty` 0.13 → 0.126, `cross` 0.19 → 0.186). The denominator guard is therefore a
correctness guard against a case this corpus does not contain, exactly like M1 below; it is not a
repair of a wrong published number. Recording it the other way round — as a stale artifact — was an
inference from the code change rather than a measurement, and it was wrong.

**The publish workflow now refuses a tag that disagrees with the built version.** `publish.yml`
triggers on `v*`, checks out the **tagged commit** and builds from ITS `pyproject.toml`, so the tag
name is a LABEL and the version published comes from the commit — and nothing compared the two. A
tag on the wrong commit therefore uploaded a version other than the one it named, silently; PyPI
uploads are **irreversible**, so the only recovery is burning another version number. (Live example
of the setup: 4.77.0 is merged but untagged, so `main` carries a `pyproject.toml` at 4.77.0 while a
`v4.77.1` tag placed there would have published 4.77.0.)

The guard compares the tag against the **built wheel**, not `pyproject.toml` — a pyproject check
re-asserts the build INPUT, and the two agree by construction, so it would prove nothing about what
reaches PyPI. It runs in the `build` job **after** the build and **before** the artifact upload, so a
mismatch never reaches `publish` (which `needs: build`). Versions are compared PARSED, not as
strings: a wheel name is PEP 440-normalized while a tag is typed by hand, so a string compare would
fail on a cosmetic difference and pass on none. Executed against seven tag/wheel pairs — matching,
mismatched, `v`-less, PEP 440-equal-but-string-unequal, no wheel, two wheels, and an unparseable tag
— all behaving as specified. Wiring is pinned by `tests/test_ci_publish_guard_wired.py`, which
asserts the step ORDER (a guard after the upload would be useless) and that `publish` still
`needs: build`; each of its four assertions was observed RED against a broken workflow.

**M1 was measured on the full owner corpus, and it is ZERO — no retrain.** ADR-055 left M1 (NA-team
GK rows in the `GhostGkModel` training corpus) as a DGX follow-up under a pre-registered rule:
*zero → record the count; non-zero → retrain trigger.* Measured across all 179 matches with
conservation asserted (every expected match produced exactly one shard, zero failures):
gradientsports 64/64, idsse 7/7, skillcorner 108/108 — **0 NA-team GK rows out of 35,335,209 GK
rows**. The bundled ghost weights are therefore uncontaminated. The retired fallback returned a
constant `goal_x = 105.0` for any NA-team keeper (`same_id(NA, home)` is False), so that path was
reachable in code and never reached by this corpus. The count is a SUPERSET of what enters training
(the extractor further restricts by domain, subsample and link filter), so zero over the superset
bounds the training set at zero.

**"Published and unusable" was being reported as "nothing published".** `shape_snapshots` dropped
any polygon that converted to an empty array, so a published-but-2-vertex `visible_area` produced
no row and read downstream as `no_polygon` — collapsing it into the absent case that
`tracking._visibility` defines a separate `degenerate_polygon` token to keep apart. It now emits a
row whenever something was published, and `polygon_to_spadl` returns an empty `(0, 2)` on an
odd-length flat list instead of raising `ValueError: cannot reshape array of size 7` mid-corpus.
The 4.77.0 note deferring this on fail-loud-vs-degrade grounds was wrong: it assumed the only
alternative to a crash was a silent skip, when the degenerate token was already the honest third
option.

## [4.77.0] — 2026-08-08

### Changed — one goal-map seam replaces ten forks, and the observed-region seam ships (PR-S145, ADR-055)

**VAEP/tracking retrain trigger** for `add_gk_influence` and `add_cover_shadows`: both were still
identity-keyed, and their values change wherever team identity and attacking direction disagreed.
Everything else is additive or a hard rename that fails loud at import.

**The goal-end rule had TEN implementations across 5 modules.** Each was a variant of
`0.0 if same_id(team, home_team_id) else 105.0`, and each had the same three defects: identity-keyed
rather than direction-keyed (ADR-051's D3 class), no period term (teams swap ends at half time), and
fail-OPEN on an unresolvable end (`nan < 52.5` is False, so a keeper-less frame returned a confident
105.0). Replaced by `resolve_defended_goals(frames) -> GoalMap`, built ONCE per match from the full
frames and THREADED in — per-frame construction is a different estimator, and the cost is
PROVIDER-DEPENDENT: measured on the committed slim fixtures, a per-frame map disagrees with the
per-match one on **7.1%** of team-frames (skillcorner) / 2.2% (metrica) / **0.0%** (sportec,
gradientsports), and `attacked_goal` is unresolvable for **35.7%** / 61.7% / 0.0%. (The spec's
78.8% figure comes from its own corpus and does not reproduce on these fixtures; its 34.2%
unresolvable rate does. The decision rests on the reproduced numbers.) An eleventh fork fails CI via a
semantic AST gate that was landed RED and observed failing on all ten.

- **BREAKING — 15 public signatures**, across two packages. `home_team_id` → `goal_map` on
  `compute_gk_influence`, `lane_control`, `compute_blocking_score`, `compute_threat_pc` and
  `gkdv.delta_threat_suppression`; **removed** (replaced by an optional `goal_map`) from
  `add_gk_influence`, `add_cover_shadows`, `gk_influence_xfns`, `cover_shadow_xfns`,
  `gk_pitch_control_share_weighted`, `gk_reachable_area_m2`, `gk_closing_time_min_s`,
  `gk_closing_time_mean_s` and `atomic.tracking.features.add_cover_shadows`; and
  `select_back_line_players` takes `defends_x0: bool` instead. Structurally verified: **zero**
  functions are left declaring `home_team_id` without reading it as a result of this cycle.
- **BREAKING — two renames, both DELETED not aliased.** `tracking.defended_goal_x` →
  `resolve_defended_goals` (returns a `GoalMap`, not a dict) and
  `providers.statsbomb.visible_fraction` → `observed_pitch_fraction`. The second also changes value:
  it now CLIPS to the pitch and returns NaN (not 0.0) for a degenerate polygon. A function keeping
  its name while changing value on the common case is undetectable by any consumer; a deleted name
  raises at import.
- **`GoalMap` keys are canonical STRINGS**, so a raw-tuple lookup misses silently — which had
  already shipped in `scripts/validate_shot_goalmouth_sb.py`, scanning `(k[0], k[1]) == key` and
  returning NaN for every row. Use `get` / `attacked_goal` / `ends_in_period`.
- **`GoalEndUnresolvedError`** (a `ValueError` subclass): per-frame functions REFUSE an unresolvable
  end; the `add_*` edge catches it by name and emits a NaN row. `ghost_gk_source` gains
  `goal_end_unresolved`, distinct from `no_keeper` — a keeper WAS present, and saying otherwise
  states something the frames refute.
- **Gate C** replaces Gate B's DETECTION for the two re-keyed aggregators (Gate B goes vacuous once
  the parameter is gone). It holds frames fixed and swaps the MAP, reproducing the recorded D3
  magnitudes exactly: `share 0.108532`, `closing_min 4.38062 s`, `closing_mean 4.02205 s`,
  `blocking_score 148.83`.
- **`add_cover_shadows` is now keeper-dependent on freeze-frame input.** On SB360's `gk_absent`
  roster its five columns move `all_nan` → `no_signal`, because the outfield fallback guesses both
  teams at the same end (measured means 56.9 and 76.5, both past the 52.5 midline) and
  `attacked_goal` refuses a degenerate map. Previously both legs produced numbers the frames could
  not support.

### Added — the observed-region seam (ADR-055)

`tracking/_visibility.py`: `point_observed` (returns `bool | None` — `False` is a claim, and a
missing polygon supports no claim), `region_observed_fraction` (an `(M, 2)` POLYGON, never a bbox,
which can only OVER-report coverage for a triangle) and `add_visible_area_coverage`, emitting
`visible_area_fraction` + `visible_area_source` over the closed
`{observed, no_polygon, degenerate_polygon, unlinked}`. The fraction is NaN for every non-`observed`
token — never 1.0, never 0.0. Geometry primitives live in the neutral `silly_kicks/_polygon.py`
because `providers/` has no runtime dependency on `tracking/`. `build_sb360_coverage.py` now
accumulates only finite fractions and reports `n_with_polygon` as the denominator (ADR-042).
Wiring coverage INTO the count features is deliberately NOT done (ADR-009). C4 aggregator count
32 → 33.

### Not shipped — the `_snapshot` dtype pin, and why

Spec §2.6 recorded that `_snapshot.py`'s `pd.concat` yields `Int64` on pandas 2.3.3 and `Float64` on
3.0.3, and prescribed a cast to `TRACKING_FRAMES_COLUMNS`. On the pinned resolver the concat yields
**`float64`**; the prescribed cast is **unimplementable** (`int64` cannot hold the ball row's NA, so
it raises on every snapshot, and the declaration is not what the native adapters emit anyway); and a
`restore_id_dtype` pin changes nothing for numpy-int, nullable-`Int64` or object sources — with the
pin excised, **0 of 2** tests written for it went red. Dropped rather than shipped as an
unobservable no-op; the pandas-3 concern is recorded as a follow-up.

## [4.76.0] — 2026-08-06

### Fixed — the ghost-GK path REFUSES on freeze-frames instead of fabricating (PR-S144, ADR-054)

**No retrain, no re-materialize** — ghost positions on velocity-bearing frames are byte-unchanged.
**Schema note (Hyrum):** `add_ghost_gk` and `compute_ghost_gk` gain one column. C4 count unchanged (32).

Repairs the one actionable defect ADR-053's audit found and deliberately left. CLAUDE.md's
`speed_source` bullet already required both directions; the ghost path violated a rule `_das.py`
and `_press_commitment.py` already obeyed — marked frames fabricated instead of degrading, unmarked
frames fabricated instead of raising.

**The mechanism determined the fix.** `extract_ghost_gk_features` yields NaN, and `predict_mean`'s
HGBR reconstruction routes NaN down each split's LEARNED missing-value direction — fitted where NaN
meant an occasional dropped measurement, applied where 5 of 26 features are absent on 100% of rows.
Measured: `NaN -> [6.795, 33.522]` vs `zero-fill -> [6.888, 33.362]`. An imputation POLICY, not a
zero-fill, so "fill the zeros correctly" was never the fix.

**The guard sits at the shared serving seam `_serve_positions_core`**, because there are THREE
public entry points and two bypass the aggregator: `ghost_gk_xfns` reaches `compute_ghost_gk` (the
VAEP path) and `gkdv/_engine.py` calls `serve_ghost_gk_positions` (TF-19). A guard at
`add_ghost_gk` would have fixed one caller in four.

**The seams degrade differently, and the asymmetry is forced.** The two column-emitting seams return
NaN + `ghost_gk_source`; `serve_ghost_gk_positions` returns NO rows, because `gkdv/_engine.py`
RAISES on a non-finite ghost on a scored frame — NaN rows there would break TF-19 rather than
degrade it.

**The audit re-derives to ZERO fabrications, by RULE not hand-edit**: the machine observation
changed `differs` -> `all_nan` and the adjudication followed. `behaviour_matrix.md` now reads 489
verdicts, 0 `silent_degrade`. That is ADR-053's locked-observation / reviewable-adjudication split
working as designed.

**New: `validate_velocity_regime` / `VelocityRegimeDiagnosis`**, a third member of the
`validate_time_base` / `validate_id_dtypes` family. Measured from the registry, **5** aggregators
produce output that moves with velocity but stays honest — pitch control at zero velocity is a
well-defined positional model. That is a frame-set-level fact, so it is a diagnostic rather than
five per-row columns each carrying a constant. Rule: **a provenance COLUMN where the value changes,
a DIAGNOSTIC where only the interpretation changes.**

**Breaking, narrowly:** unmarked velocity-less frames now RAISE. What breaks is a fabricated
coordinate.

### Added — StatsBomb 360 parse port (PR-S144, ADR-054)

`silly_kicks.providers.statsbomb` — freeze-frames in, the `snapshot_to_tracking_frames` contract
out, plus `visible_area` carried as raw per-action polygons. **Shape, never fetch**: no new runtime
dependency, verified by AST over the subpackage.

**EXTRACTED from `scripts/build_sb360_coverage.py`**, not written beside it — verified an identity
move, so the published `coverage.md` numbers cannot drift from the port. The scalar affine is
promoted to `spadl/_sb_coordinates.py` while the clip and the 3-element shot `y_offset` stay behind
as EVENT semantics (ADR-038's split), which is what lets a `visible_area` polygon extend past the
touchline instead of being silently shrunk.

Contracts the source forces, stated rather than discovered downstream: the 360 file carries no event
type (coverage is always a JOIN, and zero overlap is COUNTED via `JoinReport` — measured, 3 of 22
open matches); player flags are ACTOR-relative with no identity; and **`player_id` does not recur
across frames**, which forecloses per-player aggregation.

Committed slice: Women's World Cup 2023 match 3893795, 6 freeze-frames, digests in `SOURCE_SHA`,
read with stdlib `json` so the golden gate cannot skip. `NOTICE` gains a StatsBomb entry — it had
none while the repo already shipped their events.

## [4.75.0] — 2026-08-05

### Added — SB360 coverage audit: a per-column verdict for every `add_*` on freeze-frames (PR-S143, ADR-053)

**Tests, scripts and docs only — no `silly_kicks/` source change, so no retrain trigger and no
re-materialize.** C4 count unchanged (32). Prepares a possible commercial StatsBomb 360 GK
collaboration by answering, with evidence rather than inspection, which tracking features already
work on freeze-frames.

**The question.** SB360 ships a per-event freeze-frame: positions, no velocity, one frame per
action. `snapshot_to_tracking_frames` has bridged it into the tracking schema since PR-S88, but
only ~10 of 33 `add_*` aggregators had a documented compatibility verdict, and those were written
by inspection. An aggregator that degrades *silently* on freeze-frames is indistinguishable
downstream from one that works.

**Two layers, because they answer different questions.** Layer A measures what the CODE does:
each aggregator runs twice on a paired fixture — Leg A built by the real
`snapshot_to_tracking_frames`, Leg B velocity-bearing with identical positions at the linked
frame — and every emitted column gets a verdict. Layer B measures what the DATA contains, over 22
real open-data matches across three competition cells.

**Layer A result: 486 verdicts across 34 entry points — 299 `works`, 97 `honest_nan`, 60
`differs_by_design`, 26 `not_exercised`, and exactly 4 `silent_degrade`, all `add_ghost_gk`.**
Working unchanged on freeze-frames: all 16 `add_xt_gk` columns, `add_gk_completion`,
`add_pre_shot_gk_position`, `add_pre_shot_gk_angle`, `add_team_shape` (20 columns),
`add_defensive_line`, `add_shape_graph`, `add_structural_pass`, `add_packing`, `add_line_break`,
`add_pressure_on_actor`, `add_defensive_credit`, `add_off_ball_run_values`, `add_sync_score` and
`spadl.add_restart_coordinates`.

**Layer B result (19 usable matches).** Shots carry a freeze-frame **97.7%** of the time with the
defending keeper visible **92.2%**; saves **97.3%** / **100%**; crosses **85.0%** / **81.4%**;
goal kicks **32.6%** / **96.4%**. **Shot-facing and save GK work is usable today; the single
constraint is goal-kick FRAME AVAILABILITY**, not the library and not keeper visibility. Quote the
dispersion, never the mean: per-match median 21%, IQR 18–50%, range 8–61% (n=16).

**The observation is locked, the adjudication is not.** CI re-derives each machine observation
(`identical`/`differs`/`all_nan`/`partial_nan`/`no_signal`/`raises_a`) and asserts it; the human
adjudication (`works`/`silent_degrade`/`honest_nan`/…) carries a mandatory rationale and is
deliberately unlocked, because a machine cannot separate *fabricated* from *legitimately
different* — pitch control at zero velocity is a valid positional model; a fitted model silently
imputing the features it was trained on is not. Locking the verdict instead was tried and
rejected: it pins the key set while the content rots. A new `add_*` must register or CI fails, in
both directions.

**A rationale corrected before shipping, at its generating rule.** The `add_ghost_gk` verdict text
first said the model "receives structural zeros". Measured, it does not:
`extract_ghost_gk_features` yields NaN, and `predict_mean`'s HGBR reconstruction routes NaN down
each split's *learned missing-value direction* — a policy fitted where NaN meant an occasional
dropped measurement, applied where 5 of 26 features are absent on 100% of rows, and a **different**
prediction from zero-fill (`NaN → [6.795, 33.522]` vs `zero → [6.888, 33.362]`). The
`silent_degrade` verdict is unaffected — the fabrication is real — but the stated cause was wrong,
which matters for the fix: "fill the zeros correctly" would not address it, whereas refusing on the
`speed_source` marker would. Corrected at the rule in `tests/sb360/_adjudicate.py` and regenerated,
so the 4 call sites, the behaviour matrix, ADR-053, `CLAUDE.md` and the report all move together
rather than drifting apart. **This is the locked-observation/reviewable-adjudication split earning
its keep on its own first example:** a locked machine verdict would have been right and unreviewable,
and the error lived entirely in the human prose the design deliberately left open to correction.

**Excluded and counted: 3 of 22 matches** (`3877115`, `3877170`, `3877194`, MLS 2023) ship a 360
file whose `event_uuid`s have zero overlap with their own events file, verified against the RAW
events. Each would otherwise have contributed a single `unmapped` bucket, visually
indistinguishable from a quiet match while diluting every aggregate it entered. **14% of sampled
matches had unusable 360↔event linkage** — itself a planning fact.

Registry regenerable via `tests/sb360/_regenerate.py` + `_adjudicate.py` (round-trip verified
byte-identical). Report: `docs/research/sb360_coverage/`. Follow-ons in `TODO.md`, chief among
them that `add_ghost_gk` should refuse on freeze-frames rather than emit a plausible coordinate.

## [4.74.0] — 2026-08-05

### Fixed — ADR-028 PR 5 of 5: the goal-relative transform was CHIRAL (PR-S142, ADR-051)

The **final PR of the ADR-051 cycle**. **Retrain trigger + re-materialize.** C4 count unchanged
(32). No new ADR — the design is spec §8b of `docs/superpowers/specs/2026-07-29-adr028-orientation-defect-class-design.md`, and ADR-051 records the outcome under its **"Closure: PR 5"** section.

**The defect.** `silly_kicks/tracking/_geometry.py` had `to_goal_relative_x` and
`to_goal_relative_vx` and **no `to_goal_relative_y`**, so `goal_x=105` mapped `(x, y) -> (105-x, y)`
— an x-only mirror, determinant **-1** — while `goal_x=0` was the identity, **+1**. The two goal
ends used frames of **opposite handedness**. Composed with ADR-028's point reflection that left
every RADIAL feature byte-identical and NEGATED every BEARING, which is why it survived every
obvious check: distances and radii all agree. In production one physical scene scored differently
depending which END the acting team attacked — a systematic home-vs-away split inside a match.

Measured: xS **12 of 27** features flip sign (worst delta 6.123525), xCross **3 of 16** — exactly
the counts ADR-037 records as "sign-inconsistent", reached independently from the ADR-028 side.
After the fix: **0 flips, delta 0.000e+00**, full 27/16 comparison.

Fixed by making the pair the 180-degree **point reflection** `(x, y) -> (105-x, 68-y)`. xCross
converts y at its single seam; xShot could not — `gx` was x-only while y was read at FOUR
independent sites — so its frame is pre-transformed ONCE at the top of `extract_xshot_features`
and `gx` deleted, making "no call site can be missed" a property rather than an assertion.
`GEOMETRY_VERSION` -> `"goal-relative-2"`, mandated by that constant's own contract.

**Verified end-invariant on REAL data, not only the synthetic gate:** across **1,360 scenes x 27
features**, scene S at `goal_x=105` equals mirror(S) at `goal_x=0` with max feature delta and max
prediction delta both **exactly 0**.

**Also fixed:** `_dominant_region_area`'s y grid centred on 34.5 rather than 34.0 (105 divides by
3, 68 does not), so a scene and its left-right mirror at the SAME goal end differed by **5.4%** in
`space_controlled` — xCross model feature #3, on the left-wing/right-wing axis of a cross model.
Now uses a derived symmetric anchor `a = L/2 - (n-1)*res/2`, byte-identical on x. This is NOT an
orientation defect: with the transform fixed and the grid untouched both legs are bit-identical.

**Retrain / re-materialize — the SHAPE, not just the fact.** `xshot_occurrence_xfns` and
`xcross_attempt_xfns` are wired into `pre_shot_gk_full_default_xfns` only, so opted-in VAEP
consumers re-materialize. `to_goal_relative_y(y, goal_x=0)` is the identity, so the transform fix
moves **only rows attacking the high-x goal** — roughly half the corpus, precisely the home-vs-away
split above. The grid change is **two-sided**: on the canonical fixture 0% at one end and -5.4% at
the other, scene-dependent rather than structural.

**Weights.** Both bundled defaults retrained on the corrected geometry and re-stamped:
`shipped_variant public` (SkillCorner + IDSSE, the registered 17), `geometry_version
goal-relative-2`, chirality + feature_contract re-applied on x86, all four acceptance gates
passing. Quality unchanged — xS pr_auc 0.3458 vs 0.3514, **0.37 fold-SD** apart, identical
`positive_rate` and `base_rate_brier`. `_ghost_gk_weights/` verified byte-unchanged.

**TF-19 verdicts re-run on the corrected weights, both UNCHANGED.** xCross GK-substitution probe:
`tf19_ready` **False** before and after; the GK signal strengthened (`gk_median_abs_delta` 0.002417
-> 0.003582, ratio 1.41x -> 1.70x) but still misses both registered prongs (`ratio >= 2.0`,
`abs floor >= 0.01`).

**Liveness gate retired and replaced (ADR-032 idiom).**
`test_bundled_model_is_live_not_degenerate` asserted the model ranks near-goal rows above far ones
at AUC >= 0.9. That premise is **false for the models it guards** — measured `corr(r, p)` is
**+0.89** for the pre-PR-5 model and **+0.94** for this one, both rising with distance (xS predicts
a shot ATTEMPT within ~1 s, and 25-34 m is prime open-play shooting range). It passed only because
two defects compensated: its FAR class sat at r ~= 101 m, ~3x outside the trained domain
(`_ATTACKING_THIRD_M = 35.0`), and the generator zeroed `vx`/`vy`, pinning ball speed to one value
on every row — the same degeneracy PR-S118 fixed in the xCross fixture and left here. Fix either and
the OLD model scores ~0.47, chance. Replaced by three gates — a fixture **precondition** (all rows
in-domain, `speed` non-constant, >= 40 rows), non-constant model output, and a geometry response —
each proven by planting the defect it catches.

**Also:** `cache_dir` threaded through the trainers (a corpus downloaded once instead of per run —
hours per pass at 24-90 s/match), ADR-038 taxonomy wired into `train_gk_completion.py` which had
none, and all 16 pinned `_KNOWN_NON_ASCII_DRIVERS` made ASCII-only so the debt list is now **empty**.

**Measured for PR 7:** the DGX-vs-x86 feature-contract delta is **0.000e+00** across all 27 probe
features, so ADR-050's `atol=1e-6` stands as chosen and platform-scoped fingerprints are not
needed. See `docs/research/pr5_platform_atol/`.

### Added -- two provenanced measurement drivers, and the artifacts they produce

`scripts/measure_covariate_invariance.py` answers which causal covariates a geometry change moves and
**which axis** moved them, using **three** arms -- `parent` (old transform, old grid), `old_grid` (new
transform, old grid) and `current`. Two arms cannot attribute `space_controlled`: the NEW
dominant-region grid is closed under the ADR-028 point reflection (`1.0 -> 67.0` is a grid centre) and
the OLD one is not (`1.5 -> 66.5`), so measuring axis A against current code forces its delta to zero
*by construction* while the baseline, which carries the old grid, does move under it. Measured, that
interaction is real and large: `space_controlled` moves **97.5652** on axis A and **70.9565** on axis B
at `goal_x=105`, versus a clean `0.0 / 70.9565` at the identity end. The two structural invariants the
`tf19_signoff_power` decision rests on are exact -- `GK_r` and `gk_depth_x` are **0.0** on both axes,
because `hypot(a,-b) == hypot(a,b)` and `cos` is even.

`scripts/measure_platform_probe.py` measures the feature-contract fingerprint per platform, one
self-provenanced JSON each, joined by a `--compare` that REFUSES legs disagreeing on `run_commit`,
`geometry_version` or probe identity -- such a delta would confound platform with code. Result across
**all three** bundled contracts (69 features, AMD64/Windows/3.14 vs aarch64/Linux/3.12):
**`max_abs_delta = 0.0`**. So ADR-050's `atol=1e-6` / `rtol=0` stands and fingerprints do not become
platform-scoped. The hand-run predecessor covered 27 of 69 features and carried no provenance at all;
it was removed rather than retro-stamped.

Baseline isolation is by CHECKOUT, not monkeypatch: both extractors bind geometry absolutely, so an
in-process emulation still resolves `_geo` to the CURRENT module -- inert for this diff, and a silent
zero for any future change that alters an existing function's behaviour rather than adding one.

### Fixed -- research artifacts refreshed, and two provenanced for the first time

All verdicts **reproduce unchanged** on the corrected geometry: entanglement `inside_band`; xS probe
v1 `unmeasurable_at_dose` and v2 `joins_with_caveat`; xCross `tf19_ready: false`;
`gk_clears_placebo_band: False`.

* **`xcross_causal`** gains provenance for the first time -- its driver had none, the third recorded
  instance of that class and one the gate could not see, since `ARTIFACT_DRIVERS` is hand-maintained
  behind a floor (`>= 6` against 14 entries). The gate was landed RED and observed failing 3 of 5
  before wiring. **Its corpus also changed** (23,966 -> 52,978 opportunities; 669 -> 4,193 treated),
  so the magnitudes are NOT a before/after of the geometry fix -- 4.66.0 alone changed pooled-arm
  cluster keying. The README says so rather than presenting a comparison that would attribute several
  releases of pipeline change to this PR.
* **`tf19_pr3b`** is rewritten with the v1 leg of the same `--variant both` run whose v2 leg is in
  `tf19_pr3b_xs_v2` -- two VIEWS of one artifact, not two runs, which is why both carry
  `run_commit 08ce9a8`. It previously carried no provenance either.
* **`tf19_signoff_power`** is NOT rebuilt, and the reason is measured rather than argued. Its
  annotation lives in a SIBLING `invalidation.json`, because hand-editing a driver-produced file under
  an unchanged `run_commit` is the mirror image of restamping one. Three classes are recorded, not
  two: invariant now (treatment-derived), stale now (`att` -- `theta` is a build-time spell column, so
  the persisted parquet is stale and re-running only the analysis leg would LAUNDER the decimals), and
  **current now, stale after PR 7** (`icc`). A two-way split would have asserted `icc` is simply
  current and been wrong two PRs later.
* **`tf19_pr2`** now cites the provenanced artifact instead of restating numbers as prose, marks
  Stage B as NOT refreshed, and resolves a shot-arm row that had been `PENDING PR-3` since 4.51.0.

### Fixed -- three defects found by executing, not by reviewing

Seven document review rounds preceded execution; these were found only by running the thing.

* **A leaked monkeypatch.** The `old_grid` arm patched `_grid_centres` and never restored it, so the
  `current` arm would have run on the OLD grid -- axis B reading exactly `0.0` and the artifact
  asserting the grid re-anchor moved nothing, which is the one claim the three arms exist to test.
  Caught by `/final-review`. **The positive control did not catch it**, because axis A still moves; a
  second control now asserts `space_controlled`'s axis-B delta is non-zero.
* **NaN classified as an axis.** `score_differential` is all-NaN by construction, and every NaN
  comparison in the classifier is False, so it fell through to `"B"` -- asserting the grid moved a
  confounder never measured. Both controls were NaN-blind by the same mechanism (`abs(nan) > 1e-6` is
  False), so an entirely-NaN table could have reported `status=ok`.
* **A driver fatal on its own baseline.** `_grid_centres` was ADDED by this cycle, so the baseline
  tree has no such attribute; reading it directly killed the baseline arm. The AttributeError is what
  a correctly-isolated baseline looks like -- had the subprocess resolved to the current tree the
  attribute would have been found and both arms would have silently shared a grid.

### Changed -- the xCross liveness gate is retired and replaced by three

`test_xcross_bundled_model_is_live_not_degenerate` asserted `roc_auc_score(...) >= 0.9` and measured
**1.0000** on a fixture where SEVEN of sixteen features were constant -- the entire GK block among
them, pinned at `(2.0, 34.0)` by the generator. Overwriting that block with `0`, `99` or `NaN` each
left AUC at 1.0000, so a model ignoring keeper position was indistinguishable from the real one.
Ranking is not retained as a substitute: a live-but-GK-blind model scores AUC **1.0000** on the
repaired fixture -- HIGHER than the real model -- while responding **0.00%** to keeper position.

Replaced by a precondition (>= 40 rows, no inert feature, every row inside the model's own
`wide_area_only` domain), a liveness check, a GK-block response gate, and a re-derivation fidelity
pin. The inert clause is NaN-EXPLICIT because neither half suffices alone: a bare `(max - min) <= tol`
is NaN-blind and misses the all-NaN column, while `nunique <= 1` misses two features sitting 3.0 ULP
apart. `_MIN_RANGE` holds frozen literals, not "a fixed fraction of the observed range", which finds
only 7 of 9 and is degenerate when computed at test time.

Probes are pinned in GOAL-RELATIVE coordinates: `gk_x`/`gk_y` are absolute, so a pinned absolute
`(2.0, 34.0)` is a keeper on his line at one goal end and 103 m upfield at the other -- the gate would
still pass, failing quietly on a physically absurd state. `mean(|dp|)`, never `|mean dp|`: across seven
probes the absolute form spans **1.93x** and the signed form **6.83x**.

The fixture is rebuilt to 48 rows / 48 distinct vectors / 48-in-domain / zero inert features, keeper
swept over 12 positions, with negatives redesigned as wide + advanced but not cross-imminent -- they
were previously central and deep, so restricting to the trained domain removed EVERY negative and left
a single-class fixture. It stays SINGLE-ENDED deliberately: a committed table provably cannot carry
chirality evidence, since a reflection pair maps onto the same goal-relative configuration and a
fabricated half is bit-identical to a real extraction on integer coordinates.

## [4.73.0] — 2026-08-01

> **This is the release that publishes the 4.71.0 pairing, and the first tag since `v4.70.0`.**
> 4.71.0 corrected the serving geometry `GkCompletionModel`'s bundled weights were fitted against;
> this release retrains those weights, closing the skew. 4.71.0 and 4.72.0 were both deliberately
> left untagged — on this repo `publish.yml` triggers on `tags: ["v*"]` only, so an untagged version
> publishes nothing — and `v4.73.0` ships 4.71.0 + 4.72.0 + 4.73.0 together.

### Fixed — ADR-028 RC4: the pining loader shipped UNORIENTED SkillCorner frames (PR-S141, ADR-051)

PR 4 of 5. **Research-corpus + bundled-weights change; no library API change.** C4 count unchanged
(32). No new ADR — RC4 is recorded in ADR-051.

**RC4.** `scripts/_loader_pining.py::build_skillcorner_frames` forced
`output_convention="absolute_frame"`, leaving `team_attacking_direction` **NULL on every row**.
`acting_team_attacks_rtl` then resolved nothing, returned an all-False flip, and the **entire ADR-028
per-action re-projection layer silently no-opped** — so every away-team action in the research corpus
carried mixed-convention geometry while looking healthy. The converter's own default is `"ltr"`; the
override was never a decision. Measured on match `1886347` at full frame depth, **both sides**
(`docs/research/adr028_rc4_orientation/`): `unlabelled_fraction` **1.0000 → 0.0000**,
`flip_true_fraction` **0.0000 → 0.4728** (0 → 566 of 1,197 actions), orientation warnings **1 → 0**.

**IDSSE is the control and is deliberately unchanged** — `sportec.py` calls `finalize_orientation`
unconditionally before its convention branch, so those frames are already labelled: **718 of 1,363
actions flipped, `0.5267791636096845` to all 17 digits, identical on both sides.** That also confirms
spec §2.2's independent "718/718" with a second instrument.

**A first measurement of this was CAPPED at `tracking_limit=3000` and recorded it nowhere** — a cap
presented as a corpus, caught in review by arithmetic (66,000 IDSSE player rows is exactly
3,000 frames × 22). A truncated frame set leaves `(game_id, period_id, team_id)` keys out of the
orientation lookup and those actions **default to no-flip silently**, so the capped figures
(SkillCorner 0.2398, IDSSE 0.3155) were **lower bounds**. `unlabelled_fraction` was unaffected — a cap
cannot make labels appear — so the defect itself never depended on it, only its magnitude. Both JSON
artifacts now carry a `_provenance` block recording `tracking_limit`, `max_per_provider` and the
resolved commit.

### Changed — `GkCompletionModel` retrained on corrected geometry (both bundled variants)

Retrained through main's explicit `--mode retrain --feature-space moved --probe-old`, from a **clean
tree** (`run_tree_state: clean`, no `--allow-dirty`), each variant in its own worktree at the RC4
commit so neither run's output could dirty the other's provenance.

**`default` (Gradient Sports, 64 matches).** `N=3491`, `n_native=2953` (85%),
`native_auc` **0.8375 → 0.8549** (CI95 [0.8377, 0.8717]), Brier 0.1195 vs base 0.1756; all three gate
prongs green. Per-type serve modes unchanged: goal-kicks stay **`model`** (AUC 0.835, LCB 0.809 —
GS goal-kick completion *is* geometry-predictable), degenerate throw-ins stay `base_rate`. **Cause is
RC2 + RC5, not RC4** — RC4 is SkillCorner-only and cannot touch a Gradient Sports fit; what moved this
design matrix is `_gk_geometry`'s frame-coordinate reprojection and the cross-team next-event borrow,
both shipped in 4.71.0, reaching the features through
`prepare_gk_completion_training_data → resolve_gk_geometry`. This run **also widens the corpus** from
the committed ~30 matches (`n_rows` 1666) to all 64 now in the manifest, so the weight change is
geometry **and** corpus — stated because the two are not separable after the fact.

**`skillcorner` (10-match public arm).** `N=542` (81 goal-kicks + 461 GK-passes), gate
`bundle_skillcorner`: GK-pass AUC **0.740** (LCB 0.673), ECE 0.045, slope 1.01; goal-kicks stay
`base_rate` (AUC 0.461). **This is a train/serve-consistency fix, not a quality improvement, and the
evidence says so plainly**: the max coefficient delta is **0.0149**, essentially all of it in `dest_defender_density`, the only feature computed DIRECTLY from frame positions at extraction time. The five coordinate features do read frames INDIRECTLY via `resolve_gk_geometry` -- an earlier claim that density was "the only frame-reading feature" was wrong -- but for SkillCorner they barely move, because **461 of 542 rows are open-play GK passes whose origin is NATIVE** (ADR-024/PR-S104: the ball at release IS the keeper, measured 0.4 m), leaving only the 81 goal-kicks frame-resolved. Density is recomputed from frames on every row, and it carries the smallest-magnitude coefficient in the model. It also lands *below* `_CORPUS_IDENTITY_ATOL` (0.05), so a `--mode rebundle`
would have declared "nothing changed" and shipped stale weights against moved features — the exact gap
ADR-052's mode split exists to catch, demonstrated on live data.

**Corpus is capped at the 10 public matches deliberately.** ADR-038 grew the pining SkillCorner
listing to 108, and the 98 additions are non-redistributable; `train_gk_completion.py` has **no
ADR-038 taxonomy enforcement**, so nothing in the trainer would have refused a defaulted run that
pulled restricted matches into a distributed wheel artifact. Wiring the guard in is tracked in
`TODO.md`, not bolted on here.

**Hyrum / re-materialize:** every consumer of `gk_completion` — and `xt_gk`'s RAV term, which consumes
it — sees changed served probabilities for both providers.

### Changed — TF-24 calibration: NO re-sweep trigger (stated, not implied)

RC4's blast-radius analysis names `calibrate_tracking_defaults` as a second consumer, so a **No**
verdict must be recorded rather than left silent. Stage 1's `infer_ball_carrier` params were fitted on
a fold that included the unoriented SkillCorner frames — precisely, `beta` and `gamma` are
Optuna-calibrated at a *held* `tolerance_m=3.0`, itself an engineering default — but carrier inference
is **orientation-invariant**, so the mis-oriented fold returned the same answer. That invariance is now
asserted by `tests/tracking/test_ball_carrier.py::test_carrier_inference_is_orientation_invariant`
(40/40 identical assignments under an exact ADR-045-correct point reflection, distances unchanged to
<1e-9); it had previously been stated as a measurement in four documents with no artifact, script or
test behind it. Stage 2's `k3` and `min_displacement_m` ship as engineering defaults TF-24 never set.
**No shipped constant moves.** A refresh of the harness's *recommendations* on corrected geometry is
tracked in `TODO.md`.

### Fixed — the re-bundle guard accepted NaN drift, and had no shape check (ADR-052 amended)

`_assert_rebundle_reproduces` (extracted this release from two byte-identical
`np.testing.assert_allclose` blocks so they cannot drift apart) initially **accepted a NaN** in
`intercept`, `mean` or `std` where the calls it replaced rejected all four: `max()` is order-dependent
under NaN — every comparison is `False`, so it keeps whichever key it was holding, and only `coef`
aborted, by accident of dict order. Non-finite drift now aborts unconditionally. A changed feature
**count** now aborts with its own message instead of a raw numpy broadcast `ValueError` mid-pass. The
abort reports all four parameters (`assert_allclose` short-circuits on `coef` and never reports the
`mean`/`std` drift that is the signature of a feature-space move) and names the retrain command.
Tolerance semantics are documented: max-absolute drift against `_CORPUS_IDENTITY_ATOL`, dropping
`assert_allclose`'s default `rtol=1e-7`, which contributes ~1e-7 against a 0.05 floor.

**ADR-052 is amended** with this and with the `probe_old` correction: its two-probe design is right,
but its arity was left implicit, and "the design matrix the committed model was fit on" reads as a
historical artifact. `predictions_moved` compares **element-wise**, so the probes must be
**row-aligned** — meaning `probe_old` is the *same corpus* under pre-change geometry, not a
vintage training matrix. A mismatched one usually raises on broadcast — but **not always**: a 1-row
probe broadcasts and answers silently (measured), so the trainer now compares row counts explicitly
rather than relying on numpy. Row
order agrees by construction; row **count** does not (the trainer filters on
`isfinite(length) & isfinite(dest_x)`, both derived from the corrected geometry) and must be observed
by comparing the probe's `n_rows` to the trainer's printed `N=`.

### Added — `metrics.json` records its own corpus bounds (forward-looking)

`max_per_provider`, `tracking_limit`, `feature_space` and the `--probe-old` basename now land in
`metrics.json` on both trainer paths. An unrecorded cap is indistinguishable from a full run and
silently biases every number beside it — the failure this same cycle hit on the RC4 measurement,
where `tracking_limit=3000` went unrecorded and halved a headline figure.

**Honest limit: the two artifacts shipped here do NOT carry these fields.** The retrain runs against
already-committed code, so the change reaches the *next* bundle, not this one. This run's bounds are
recorded in the two `MODEL_CARD.md` files (64 GS matches / 10 SkillCorner public matches, full-match
frames) and in each artifact's `reason` string.

### Added — test coverage the guards lacked

- Per-parameter isolation over all four served parameters (the drift tests moved only `coef`/`mean`,
  so a guard that stopped checking `std` stayed green — verified by planting exactly that mutant).
- Non-finite-drift and changed-feature-count regression tests.
- `_as_weights` against a real bundled `GkCompletionModel` plus a save/load round trip — every prior
  test used plain dicts, so the four *private* attributes it reads were never exercised on a real
  model. Teeth proven by planting a 0.01 coefficient corruption.
- `tests/scripts/test_loader_orientation.py` resolves each builder's `convert_to_frames` argument from
  the AST and pins the whole mapping; an explicit `None` no longer collides with an omitted kwarg.

### Fixed — `.gitignore` silences driver shard roots (provenance, not tidiness)

**Five** drivers resolve a shard root to a relative path on a default invocation, so the run writes
scratch into the repo; untracked files count as dirty by design, so for the three that gate on the
tree the *next* artifact-writing run refuses — caused entirely by its predecessor's scratch, with
`--allow-dirty` the tempting response, which would stamp `run_tree_dirty: true` on a run whose code was
clean. Globs are **root-anchored**, so a `*_shards/` directory nested inside the package or the
committed research tree is not silently masked. `calibrate_xt_bandwidth` needs its own entry: its
shards land under `<--report-out>_corpus/shards/`, which the glob does not match.

## [4.72.0] — 2026-07-31

> **Tag readiness is inherited from `main`, not from anything in this release.** 4.72.0 is
> `scripts/`-only and the wheel is byte-identical to its predecessor, so it adds no release
> hazard of its own. It does sit on top of the committed-but-unreleased 4.71.0, whose RC2 + RC3
> correct the serving geometry `GkCompletionModel`'s bundled weights were fitted against — so the
> first tag cut above 4.71.0, whatever its number, publishes that pairing. See ADR-051; the
> retrain that clears it is tracked there.

### Added / Fixed — the shared corpus-driver seam `scripts/_driver.py` (PR-S140, ADR-052)

Twenty-one `scripts/` drivers walk a corpus; **three survived a crash and fourteen held every
result in memory and wrote once at the end.** Measured cost: a power-analysis driver spent **8.7
hours** walking 64 matches, raised in the cheap analysis step that followed, and lost all of it.
Four partial mechanisms already existed (`_partition.py`, `_cache.py`, `train_ghost_gk`'s own
feature cache, `calibrate_xt_bandwidth`'s `--corpus-cache`) covering seven drivers, split exactly
**resume XOR staleness** — and **none of them owned the loop**, which is why resume and progress
kept being the parts omitted.

`for_each` owns the loop: it streams its corpus, writes one shard per item into a
**generation DIRECTORY** named by a `ruthless.fingerprint` digest of the DECLARED inputs, skips an
item whose shard exists, prints a flushed `[i/n]` line, records failures instead of losing the pass,
and asserts conservation over its own keys. Individual primitives remain as the escape hatch, and a
driver on that path must call `assert_conservation` **and** `_require_injective` (conservation alone
is satisfiable by a lossy run: a colliding key makes two items share one shard, so `present` counts
it twice and the relation balances on a run that dropped an item). Adoption is CI-gated over a
structurally-derived population; the debt list is now **empty**.

**No library behaviour, no weights, no retrain, and the wheel is unchanged from its predecessor**
(`pyproject.toml` packages `silly_kicks` only, and `scripts/` is not shipped). C4-free — the
diagram models `silly_kicks` subpackages, and the action-coupled aggregator count stays **32**.
Corrections shipped alongside:

- **`train_ghost_gk`'s feature cache was stale-blind on three axes.** Its recorded token was the
  penalty-area geometry alone, so a re-run at a different `--subsample-fps` or `--carrier-*`
  silently reused the previous run's feature matrix while `metadata.json` recorded the NEW carrier
  params — the recorded==used invariant PR-S81 exists to hold, broken by the cache beneath it. The
  token is now the shard generation digest, which folds in every declared input. **The cache
  invalidates once; the next run re-extracts.**
- **`train_ghost_gk` stamped false provenance into a SHIPPED artifact.** `training_commit` came
  from a bare `git rev-parse HEAD`, which reads identically on a modified tree. It now refuses a
  dirty tree by default (`--allow-dirty` records `run_tree_dirty` in `metrics.json`) and joins
  `tests/scripts/test_provenance_wiring.py`. The other trainers stamp no commit at all, so they
  make no false claim — whether they should carry provenance is **surfaced, not decided**.
- **`measure_cover_shadow_argmax_agreement` carried a live ADR-028 RC1 defect.** It builds
  `passer_xy` from raw action-LTR `start_x`/`start_y` and passes it beside frame-LTR positions, with
  no home-only filter. 4.70.0 fixed the `features.py` callers; this driver imports
  `_compute_cover_shadow_dict` DIRECTLY, so it was never a registered site. It does **not** cancel
  between the two arms it compares — only the CHEAP path consumes the passer — so
  `docs/research/cover_shadow_identity/`'s numbers were pre-RC1. **The re-measurement shipped in
  this same release** (`ff1948d`), from a clean tree at `7475a27`, same corpus: agreement
  **0.1567 → 0.0443**, i.e. **0.44× the ~0.10 chance rate — WORSE than random**, where it had read
  1.6× better. **The defect had been INFLATING agreement, not suppressing it**; the ceiling argument
  drafted before the re-run ("even if every away row flipped to agreeing, ≤0.657") held, but its
  implied direction was wrong. The `detailed=True` gating verdict is unchanged and considerably
  better supported — the Wilson upper bound is **0.059** against a 0.90 floor. The `max_def`
  distribution is byte-identical across both runs (that column comes from the EXACT path, which
  never consumes the passer), confirming the fix moved the cheap path alone.
- **`calibrate_tracking_defaults --source databricks` could not run at all.** The driver calls
  whichever loader `--source` selects with one kwarg set, and `_loader_databricks.load_matches`
  accepted neither `tracking_limit` nor `max_per_provider`, so every such invocation died on
  `TypeError` before reading a row. The bronze loader now implements both.
- **Provenance is now three-state.** `git_provenance` returns `tree_state` ∈
  `{clean, dirty, unknown}` **beside** the existing boolean, and **14 stamp sites across 13 drivers**
  record `run_tree_state` in their artifacts. `dirty: true` is a positive claim that uncommitted
  modifications EXIST; on a tarball checkout or a box without git that claim is false, and an
  artifact asserting something untrue about its own provenance is the defect this module exists
  to prevent, one level down. **The boolean is unchanged and the two are deliberately not
  merged:** `run_tree_dirty` is already published and is OR-ed across workers, and
  `bool("clean")` is **truthy**, so a tri-state string in the boolean's slot would silently
  invert every aggregate. `run_tree_state` is stamped only where a driver records its OWN run,
  never at the three sites that aggregate — OR-ing `clean`/`dirty`/`unknown` has no defined
  meaning. **Additive**, but `git_provenance`'s key set widened, so a consumer pinning that shape
  sees a new field.
- **`n_counters_unrecorded` was hard-zero in three migrated drivers.** They called
  `manifest_fields(...)` by hand and took its default, so a resumed worker whose counter sidecars
  were missing wrote `n_counters_unrecorded: 0` beside `n_matches: 0` — a corpus artifact reporting
  a corpus of nothing and asserting the report was complete. The parameter now has **no default**
  and drivers use `CorpusPassResult.manifest()`.
- **`train_gk_completion` bundling declares its question.** `--mode {rebundle,retrain}` and
  `--reason` are REQUIRED with no default; `metrics.json` records both plus the superseded
  coefficients. A retrain asserts the **served predictions** moved, keyed on behaviour rather than
  parameter deltas (`mean`/`std` are raw-feature statistics in metres, so a coordinate correction
  moves them while standardisation absorbs it exactly and every served probability is identical).
  The signature takes **two** probes, measured: a single shared probe asks whether two functions
  agree on one input, when the question is whether each model behaves the same on the coordinates
  IT sees. `--feature-space moved` currently always REFUSES, because the weights directory stores
  no design matrix — a loud refusal naming why beats silently answering the wrong question.
  **Investigated and REVERSED, not deferred:** the plan called the re-bundle's corpus-identity
  check a "mirror defect" and specified re-keying it onto served predictions. Measured, the premise
  is inverted — *committed-on-OLD vs fresh-on-NEW* is 1.7e-16 (identical, and irrelevant), but a
  re-bundle **ships the COMMITTED weights** and production serves them on the **NEW** features:
  **0.72** in probability. So the abort is correct, the right action after a feature-space move is
  `--mode retrain`, and **`_CORPUS_IDENTITY_ATOL` is untouched**. Pinned by
  `test_a_rebundle_across_a_MOVED_feature_space_must_still_abort`, which records both numbers.

## [4.71.0] — NOT RELEASED (ships within 4.73.0 alongside PR 4)

> **Do not tag `v4.71.0`.** This version is committed and traceable but deliberately never published
> to PyPI. PR 3 corrects the serving geometry that `GkCompletionModel`'s bundled weights were fitted
> against; releasing it alone would introduce a train/serve skew that does not exist today. PR 4
> retrains those weights, and **the first tag cut above 4.71.0 publishes the pairing**. See ADR-051
> and `docs/superpowers/plans/2026-07-29-adr028-orientation-defect-class.md`.
>
> **Corrected 2026-08-01:** this note originally said PR 4 "bumps to 4.72.0, which is the version that
> ships". That is no longer true — **4.72.0 shipped without the retrain** (ADR-052, corpus-driver
> resilience, a concurrent session), so the skew this note exists to prevent is still open on main and
> PR 4 is **4.73.0**. The number was never the point: what matters is that no tag is cut between
> 4.71.0 and the retrain, and none has been.

### Fixed — ADR-028 RC2 + RC3: two per-action orientation corrections (PR-S139, ADR-051)

PR 3 of 5. **Re-materialize trigger, no forced VAEP retrain.** C4 count unchanged (32). No new ADR —
RC2 and RC3 are both recorded in ADR-051.

**RC2 — `_gk_geometry` wrote frame coordinates into action-LTR quantities.** `_tracking_gk_xy` and
`_tracking_ball_xy` sampled positions from the linked frame and returned them unreprojected, while
their own sibling `_tracking_gk_xy_detected` had always applied the ADR-028 point reflection. Both now
reproject.

**The two halves failed differently, and the distinction is operational.** In `_tracking_gk_xy` the
failure was a *systematic loss of the tracking tier* rather than a wild coordinate: the goal-area
clamp (`gx <= 16.5`) is an own-half predicate in *action-LTR* coords, so applied to a raw away-team
frame x it rejects a correctly-placed keeper (action-LTR x=5 is frame x=100) and the goal kick fell
through to the rule-point fallback. The clamp is now applied strictly *after* the reflection, and the
ordering is commented in place so it is not "tidied" back.

`_tracking_ball_xy` has **no clamp**, so there was nothing to catch a mis-projected ball: restart
origins and destinations moved by up to a full pitch length — spec §2.2 measured a maximum of
**101.24 m** (GS) / **99.58 m** (IDSSE). A consumer reading `enriched_start_x` saw a plausible
on-pitch coordinate at the wrong end.

**RC3 — the space-creation OBSO multiplier was applied unrotated.** `compute_space_created` builds its
multiplier from the attack-LTR `transition_grid`/`epv_grid` and applies it to a frame-LTR
pitch-control surface. For an away action the two conventions are a 180° point reflection apart, and
the measured consequence was that the two emitted columns were **exchanged**:

```
max |base.created - mirrored.denied |  = 4.44e-16   <- the SWAPPED pair agreed to float noise
max |base.created - mirrored.created|  = 1.20688    <- while like-for-like did not
```

The fix point-reflects the two **grids** (both axes), not the finished multiplier — the multiplier
also contains a ball-anchored `distance_weight` computed in frame coords, which must never be
mirrored. That is the rule the opponent-perspective branch had already followed and documented.
Reflecting at the grid seam also corrects the opponent multiplier for free, since it is constructed as
a flip of the same artifacts.

`compute_space_created` and `_compute_space_creation_for_action` gain a keyword-only
`attacks_rtl: bool = False`. The flip is computed once per call from the **frames**
(`acting_team_attacks_rtl`) and threaded in; `home_team_id` remains in both signatures and remains
unread, because it encodes team identity rather than attacking direction (ADR-051 D1). D3 retires that
parameter by disuse, not by removal. `space_creation_xfns` delegates to `add_space_creation` and
inherits the fix — it is not a second seam.

**Downstream — measured away-row change rates** (spec §2.2; one-match point estimates per provider,
not corpus rates):

| Surface | GS 10502 | IDSSE DFL-MAT-J03WMX |
|---|---|---|
| `xt_gk` composite | 19.0% | 0% |
| `gk_completion` | 17.4% | 0% |
| `space_creation` (both columns) | 47.4%, max 0.140 m² | 60.0%, max 0.880 m² |
| restart `enriched_start_x` | 2.55%, max **101.24 m** | 1.11%, max **99.58 m** |

The IDSSE zeroes are not an absence of the defect — that rate is governed entirely by ADR-024
native-origin trust, which keeps IDSSE on the native tier where RC2's imputation ladder is never
reached.

The changed surface is wider than the two `enriched_*` pairs: **all 8 `add_restart_coordinates`
columns** move, including the `*_coord_source` / `*_coord_confidence` provenance, so a consumer
filtering on confidence (as that function's own docstring example does) will see rows appear and
disappear. Home rows are unaffected throughout.

Three strict xfail markers are deleted (13 → 10); their pre-fix magnitudes (0.125, 7.0 m, 1.207) are
retained in the tolerance rationales as the signatures a regression would have to reproduce.

**One documented contract is amended, not silently broken.** `resolve_restart_geometry`'s docstring
promised it was warning-free, so the `resolve_gk_geometry` shim "can never leak a warning onto the
frozen `compute_xt_gk` path". Resolving orientation requires `acting_team_attacks_rtl`, which emits
`OrientationUnresolvedWarning` when nothing resolves — so that promise is now false and the docstring
says so. The warning is intended: unoriented frames mean the re-projection could not happen, which is
the exact condition RC2 exists to make audible. The flag is computed **once per call** and threaded
into all five helper call sites, so a call emits at most one such warning instead of five (and the
redundant per-helper groupby+merge is gone).

**RC5 — the next-event destination proxy ignored `team_id`.** `_next_event_start` borrows the next
action's `start_x`/`start_y` as a destination proxy, guarded only on `game_id`/`period_id`. SPADL is
per-**acting-team** LTR, so when the next action belongs to the other team the borrowed coordinate
describes the same physical point in the opposite convention — a 180° point reflection away.
Measured: a shared point the opponent records as `(45.0, 20.0)` is `(60.0, 48.0)` in the anchor's own
frame (15 m x, 28 m y). It now reflects on a cross-team borrow.

This is **action-vs-action**, not frame-vs-action, so the mirror registry is structurally blind to it
and dedicated tests are the only guard. An **unattested team id never decides** — `ids_differ` is
NA-safe-both-present, so an NA on either side leaves the coordinate untouched rather than reflecting
it (the ADR-027 rule that "cannot tell" must not become "reflect"). Found during this PR's review and
fixed here rather than in PR 4 on ordering grounds: PR 4 retrains `GkCompletionModel`, and a retrain
must run against final geometry.

### Fixed — two carried defects unrelated to any RC

- **`SECURITY.md`** advertised `3.x` as the supported line, stale since 4.0.0 (2026-05-30).
- **`TODO.md`'s TF-19 On-Deck row** carried a 424-character span duplicated verbatim, adding one extra
  table cell via a single unescaped `|`. The row now matches its 16 peers at 6 unescaped pipes, with
  both legitimate `\|` escapes preserved.

## [4.70.0] — 2026-07-29

### Fixed — ADR-028 RC1: the cover-shadow passer was never reprojected (PR-S138, ADR-051)

PR 2 of 5. **Re-materialize trigger, no forced VAEP retrain.**

- **`add_cover_shadows` and `cover_shadow_xfns` compared an ACTION-LTR passer against FRAME-LTR
  positions.** Both seams built `passer_xy = (row["start_x"], row["start_y"])` and differenced it
  against defenders, receivers and the ball — all of which are frame-convention — without ever
  calling `acting_team_attacks_rtl`. For an away-team action the two conventions are a 180° point
  reflection apart (ADR-028), so the passer entered the geometry at the wrong end of the pitch.
  `_cover_shadows.py:1164-1168` is the defect in one screenshot: it reprojects the RECEIVER to
  action-LTR for the xT lookup, then two lines later differences a raw frame-coordinate receiver
  against the action-LTR passer. The module knew the distinction and mixed conventions anyway.
- **The passer is reprojected INTO frame coords, not the frame into action-LTR** — everything
  downstream of that tuple is frame-convention, and the one place that steps out (the xT lookup)
  already reprojects itself. The flip is computed once per call (per *slot* in the xfns path, since
  each gamestate slot is its own action frame).
- **In the xfns path the reprojection must precede `_get_cs`**, which rounds `passer_xy` into its
  cache key. Reprojecting after the key is built would serve a home-oriented surface to an away
  action while looking correct at the call site.

**Scope, measured per column:** `n_blocked_receivers` and — cheap path only —
`max_single_defender_blocking_score` change. `blocking_score`, `blocked_threat_fraction` and
`n_potential_receivers` are passer-independent and are byte-identical; under `detailed=True` the
max-single column comes from the pitch-control counterfactual and is also unaffected. Home-team rows
are byte-identical throughout. The two affected columns were measured **separately**, and they do not
share a rate — on away rows, `n_blocked_receivers` changed on **77.8%** (GS match 10502) / **85.0%**
(IDSSE DFL-MAT-J03WMX), and cheap-path `max_single_defender_blocking_score` on **90.7%** / **100%**.
Per the spec these are one-match point estimates per provider, not corpus rates. The atomic mirror at
`atomic/tracking/features.py:1169` is a thin rename-and-delegate adapter and inherits the fix rather
than being a third site.

**No forced VAEP retrain:** `cover_shadow_xfns` is a factory in no default xfn list. Consumers
persisting the two affected columns re-materialize. C4 count unchanged (32). No new ADR — RC1 is
already recorded in ADR-051, and agreeing with a recorded decision does not need one.

**Gate A's strict xfail for RC1 is deleted, not flipped to xpass.** Gate B's marker *stays*:
`_cover_shadows.py:1030` still keys `attacking_toward_high_x` on identity rather than direction. That
is D3, which the spec keeps out of this cycle (byte-identical on converter output), and it is a
different defect class from RC1 — not a partial fix of the same one.

## [4.69.0] — 2026-07-29

### Added — ADR-028 orientation defect class: DETECTION (PR-S137, ADR-051)

PR 1 of 5. **Test-only plus one public warning category — no shipped feature value changes, no
retrain.** This lands the gate *before* the corrections, so it is observed failing on real defects
rather than arriving green: the anti-rot meta-assertion went red listing all 33 aggregators before
a single entry existed, and every known defect ships as a strict xfail whose fix is forced to delete
its own marker.

- **`silly_kicks.tracking.OrientationUnresolvedWarning`** (new public category) —
  `acting_team_attacks_rtl` returned an all-False re-projection flip **silently** on four distinct
  unresolvable inputs, so away-team geometry mixed coordinate conventions with no signal. Measured
  on one canonical away action, labelled frames vs the same frames unlabelled:
  `nearest_defender_distance` **7.6158 → 19.6977**, `receiver_zone_density` **1 → 0**. Specified by
  OUTCOME, not by enumerated condition — any all-False return that is not "there were no actions to
  flip" warns. That framing matters: an enumerated fix missed the join-key branch, and a *fifth*
  branch (the post-merge "nothing resolved" case, reachable when the acting team is absent from the
  frames or the id spellings differ) was found only after the first implementation shipped. Period-5
  shootouts are exempt (orientation is undefined there by design). Warn rather than raise, because
  consumers legitimately hold absolute frames (ADR-029) and a raise has no reachable remedy inside a
  converter. **CI can escalate it with an opt-out list of zero.**
- **`tests/tracking/test_mirror_registry.py` + `_mirror_entries/`** — a registry-driven pair of gates
  over **all 33** registered tracking `add_*`, with two meta-assertions pinning the registry to
  `tracking.__all__` in both directions. **Gate A** is the ADR-028 physical mirror (detects
  convention mixing). **Gate B** varies `home_team_id` over `{home, away, nonsense}` on FIXED frames
  (detects identity-keyed direction). Two gates because one instrument cannot see both classes:
  swapping `home_team_id` restores the very invariant identity-keying assumes, so Gate A is
  *structurally blind* to that defect class — which is exactly how Gate B found an eighth D3 member
  the audit missed.
- **Fixture repair** — `synthesize_actions` stamped `start_x`/`start_y` from raw FRAME positions, so
  its away actions were in frame convention: an ADR-028 passer defect was **unexpressible** on it and
  a *correct* implementation would have failed. Measured 9/10 actions equal to the raw frame position
  and **0/10** to the point reflection. Now emits action-LTR unconditionally, with team balance as an
  opt-in `balance_teams` parameter (sampling policy belongs at the call site; correctness does not).
  `gradientsports` attacking direction is derived from geometry instead of a hardcoded `"ltr"` scalar
  that labelled both teams the same way.

### Fixed

- **`align_join_keys` treated every object-vs-object key pair as merge-safe** — but a boxed-numeric
  object (`10.0`) against a genuine string (`"10"`) merges happily and matches **nothing**. The module
  contradicted itself: `ids_equal` already content-probes this exact pair via `_raw_comparable`
  (ADR-043). Now probed identically, at cost only on object-vs-object pairs. Found by the new
  orientation seam, whose action→frame merge was silently resolving zero rows on that dtype pair.
- **`_xshot_occurrence` merged with a raw `.astype(str)`** where int64 `1` becomes `"1"` and float64
  `1.0` becomes `"1.0"`, so the join missed every row and `add_xshot_occurrence` returned **all-NaN**
  while `compute_xshot_occurrence` returned real values on the same frames. Routed through
  `align_join_keys`, reconciling it with its structural twin `_xcross_attempt`.
- **`vaep.features.core.feature_column_names`** is a NAME probe — it calls every frame-aware
  transformer against deliberately empty frames purely to read output column names, discarding the
  values. It emitted ~12 orientation warnings per `VAEP.fit`; suppression is scoped to that loop
  only, never package-wide.
- **CLAUDE.md's ADR-028 repair table was wrong about two of the six aggregators it names** —
  `space_creation` was never reprojected (its `home_team_id` is a dead parameter), and
  `cover_shadows` is only half reprojected (the receiver, never the passer). A third,
  `gk_influence`, reprojects correctly but keys on team IDENTITY rather than attacking DIRECTION.

### Known defects, registered as strict xfails (not fixed here)

14 markers: 4 for the RC1/RC2/RC3 value corrections (PRs 2–3), 8 for the D3 identity-keying re-key
targets (the seven predicted by the spec **plus `add_gk_influence`**, found by Gate B), and 2 for a
chiral goal-relative transform deferred to PR 5 — `_geometry.py` has no `to_goal_relative_y`, so the
two ends of the pitch use frames of opposite handedness and every bearing feature negates (xS 12/27
features, xCross 3/16). See `docs/superpowers/specs/2026-07-29-adr028-orientation-defect-class-design.md`.

C4 count unchanged (32): no new action-coupled aggregator.

## [4.68.0] — 2026-07-29

### Added — TF-19 §6.1 / §3.3 corpus-run results (ADR-037)

Research artifacts plus the two registered constants they resolve. **No behaviour change:** the
only code edits are the `_validate.py` docstrings recording what the runs measured. No model,
weights or feature values move.

- **`docs/research/tf19_signoff_power/`** — the §6.1 power curves. **The two legs SPLIT.** The
  **ICC** leg discharges its registered precondition (§6.1 registers the gate only if detection at
  the anchor is >= 0.8): **power 1.0 at all three anchors**, with
  `mean_observed_icc_at_zero = -0.00034` confirming the estimator returns ~zero on no injected
  effect. The **ATT** leg does not: max power **0.055** against a required 0.80, so
  `N_MIN_MATCHED` stays `None` — and its meaning changes from "the run has not happened" to "the
  run happened and no bin reaches 0.80". The degenerate counts make that readable: **0/200 at
  n=4000 and n=8000**, i.e. an estimable design with no power, not a positivity failure wearing its
  clothes. This vindicates ADR-037 **F3**, which split two estimands the spec had conflated; they
  answer oppositely.
- **`docs/research/tf19_entanglement/`** — the §3.3 measurement that closes **F6**. 179 matches,
  98,789 opportunities, GK ablation shift **-0.006999** against a cluster placebo band of
  **0.004690** and a registered floor of 0.01 → **`inside_band`**, giving
  `regate_verdict(shot, pass, inside_band)` = **`joins_with_caveat`**. The measured value **equals
  the registered default** 4.60.0 had assumed, so F6 closes by confirmation rather than reversal:
  the verdict is unchanged but now earned. `commit_consistent: false` is reported honestly (shards
  at `6b242cf`, analysis at `d1fc18d`) and is checkably benign — 4.66.0 touched no shard-building
  code.
- **`docs/research/tf19_pr3b_xs_v2/` gains the clean-provenance re-run** alongside the original,
  which is kept because ADR-037 and TODO cite it. The 4.60.0 artifact stamped a bare
  `git rev-parse HEAD` and carries **no `run_tree_dirty` field** at all; the re-run records
  `lock_commit 78ffc70` (constants frozen 2026-07-23, before any v2 data) with
  `run_commit d1fc18d`, `run_tree_dirty: false`. It **reproduces** the original: v1
  `no_valid_placebo`, v2 `pass` → `joins_with_caveat`, 123,430 of 123,430 targets used.

- **Fixed — `_partition.aggregate_manifests`: only manifests that CONTRIBUTED data vote on
  `commit_consistent`.** The §3.3 artifact reported `commit_consistent: false` off eight worker
  manifests **unanimously at `6b242cf`** (21–23 matches each, 179 total) plus one analysis manifest
  at `d1fc18d` carrying **`n_matches: 0`** — it had built nothing, every shard already existing. A
  pass that contributed no data was voting on the data's lineage, so the flag described the
  *analysis's* commit rather than the *corpus's*. A guard that cries wolf is worse than no guard: it
  teaches readers to skim past the one field built to be un-skippable. The rule is fail-safe in both
  directions — a manifest loses its vote only by **positively declaring zero** work; one with no
  countable field at all, or whose only output is a counter such as `drop_reasons`, keeps it (both
  learned from a failing test, not from reasoning). New `commits_seen` reports every commit
  encountered including non-contributors, so an all-resume aggregate's vacuous `true` stays
  distinguishable from a genuinely single-commit one. The shipped artifact keeps its original
  `false` — a research artifact records what was computed, and editing the number afterwards is the
  falsification this cycle exists to prevent; re-aggregating those same manifests with the fixed
  code yields `commit_consistent: true`, `run_commit: 6b242cf` (verified).

**Net for TF-19:** `joins_with_caveat` now rests on **two measured inputs** instead of one measured
and one defaulted. C4 count unchanged (32).

## [4.67.0] — 2026-07-28

### Fixed / Added — TF-30 cover shadows: invariant repair, clamp verdict, gated per-defender identity (PR-S136)

Test repair and documentation correction, plus ONE additive aggregator-only column. **No change to
any shipped column's values, no API change, no retrain, C4 count stays 32.**

- **The monotonicity invariant could not fail.** `compute_blocking_score` clamps
  `max(threat_unblocked - threat_orig, 0.0)` and the invariant test then asserted `>= -1e-9` on that
  clamped column — green by construction, never once checking the property it named. It now asserts
  on the **unclamped** difference and has been **observed RED** against a planted defect
  (`assert 466.94 - 502.51 >= -1e-09`). The superseded test was deleted, not left beside it.

- **A second clamp is now measured rather than assumed.** `delta = np.maximum(new_recv - old_recv,
  0.0)` makes `max_single_defender_blocking_score` non-negative by a *second* independent mechanism.
  Measured across 600 (receiver, blocker) deltas: minimum **+1.62e-12**, so the clamp has never been
  observed to bind. Three distinct tolerances now exist at three scales (`TOL_INVARIANT` 1e-9 on
  threat, `TOL_RECEPTION` 1e-12 on summed reception probabilities, `TOL_ATTRIB` 1e-12 on
  attribution) — deliberately not one shared constant.

- **`fernandez_bornn` non-negativity settled empirically.** Non-negativity is argued structurally
  only for `spearman`/`voronoi`; for a Gaussian-influence-field model with logistic normalisation
  the proof is a research task. Minimum raw difference over 9 actions: `spearman` +3.79,
  `voronoi` +47.16, `fernandez_bornn` +29.43. All three hold, so the decision to keep both clamps
  stands — with **mixed provenance** (two argued, one measured), which the glossary now says.

- **`test_zero_blocked_implies_low_score` was renamed and repaired — and the repair exposed that it
  had never run.** The zero-blocked population does not exist on provider frames (~10 lane blockers
  per action means some lane is always blocked; measured **0/9** actions under *every* decision
  rule), and the old body `pytest.skip`ped on the empty population. It now asserts the monotone form
  the data can answer: Spearman rho between `n_blocked_receivers` (lane classifier) and
  `blocking_score` (Voronoi threat integral) — two independently computed quantities. Measured
  **rho = 0.935, p = 0.0002**; the asserted floor is 0.5, a gross-breakage catcher deliberately far
  below the observation and honestly **not** pre-registered.

- **NEW aggregator-only column `max_single_defender_player_id`, GATED to `detailed=True`.** The
  identity was computed at both call sites and discarded. It is now emitted — but **only** on the
  exact path. The cheap path can name a defender and deliberately does not: measured against the
  exact path on **970 qualifying actions**, agreement was **0.157** (Wilson 95% [0.135, 0.181])
  versus ~0.10 by chance, against a **pre-registered 0.90 floor**. The disagreements are not
  near-ties — the median names a defender worth 1.6% of the true winner, and at p90 the named
  defender's exact contribution is **exactly zero**. This is not a defect: the cheap path is
  faithful to a lane-based notion of "blocks most" and the exact path to a pitch-control
  counterfactual, and the two rank the top of the list differently (the existing rho >= 0.7
  value-correlation guarantee is near-silent about the argmax). Opting in costs a measured
  **2.3-3.2x**. Evidence: `docs/research/cover_shadow_identity/`.

- **The gate itself is guarded.** `test_cheap_path_never_names_a_defender` fails if the cheap path
  is un-gated — without it, every other test still passes. Its non-vacuity half pins that the same
  actions DO yield identities under `detailed=True`, so the all-NA is a deliberate gate rather than
  an empty fixture. The column is **aggregator-only**, never in `cover_shadow_xfns`
  (`_CS_AGGREGATOR_ONLY_COLS`, the `das_source` precedent) so a player id cannot reach a VAEP
  feature matrix.

- **The gate is re-measurable.** Gating would otherwise leave the measurement script comparing
  `None` against a real id on every row, reporting agreement 0.0 — a number that measures the
  gate, not the cheap path. A private `_ungated_cheap_identity` on `_compute_cover_shadow_dict`
  (single caller: the script; never forwarded by `features.py`) keeps the decision revisitable on
  evidence. Guarded both ways — the public default AND that the hatch still functions, since it
  could otherwise rot silently while the default guard stayed green. Verified end-to-end: the
  script reproduces **0.1992** on match 10502, matching the pre-gating pilot exactly.
  > **SUPERSEDED by 4.72.0.** Every agreement figure in this 4.67.0 entry (0.157, 0.1992, the
  > 1.6×-chance framing) was measured while the producing driver carried the ADR-028 RC1 passer
  > defect. Post-fix the rate is **0.0443**, i.e. **0.44× chance**. Left unedited because a shipped
  > release note records what was measured at the time; see `docs/research/cover_shadow_identity/`
  > for the current numbers. **The GATE decision this entry describes is unchanged.**

- **`scripts/measure_cover_shadow_argmax_agreement.py`** is new, wired to `_provenance`
  (`require_clean_tree` in `main()`, `--allow-dirty`, stamps `run_commit`/`run_tree_dirty`) and
  registered in the artifact-driver gate.

- **Documentation.** The three cover-shadow *scoring* glossary entries now state that non-negativity
  is **by construction** and that this metric — unlike the paper's signed SoccerMap-CNN
  counterfactual — **cannot express a defender whose positioning made things worse**. The TF-30
  design doc gains an appendix recording why RQ3 stays out of scope: its headline 789/822 is
  circular (positions are optimised *toward* the Cone Corridor, then scored as "is it in the
  corridor?"), and the non-circular result reduces threat in only ~75% of snapshots, reported as a
  sign with no magnitude, CI, or placebo.

## [4.66.0] — 2026-07-28

### Fixed — pooled-corpus cluster keys (ADR-037)

Found by RUNNING the §3.3 entanglement pass over its full registered provider set for the first
time: 179 matches across gradientsports + idsse + skillcorner. It died in
`causal.matching._cluster_reassign` with `'<' not supported between instances of 'int' and 'str'`.

- **`_cluster_reassign` falls back to HASH grouping only where sorting raises.** `game_id` is
  `int` for gradientsports and `str` for idsse/skillcorner, and `np.unique` sorts — so a pooled
  corpus has nothing sortable. The sorted path stays **primary**; `pd.factorize(..., sort=False)`
  fires only on the `TypeError`. That split is load-bearing, not caution: factorize orders clusters
  by FIRST APPEARANCE, so `sigma` maps different sources to different destinations, and
  `placebo_shift` documents itself as *deterministic given `rng_seed`*. **Measured over 300 random
  cluster layouts × 4 seeds, switching unconditionally changed the result in 724/1200 cases** —
  statistically the same null, a different **number**, which would have silently stopped every
  recorded placebo band from reproducing. Sortable ids are now pinned byte-identical to the
  pre-4.66.0 implementation by a test carrying that implementation verbatim, with a non-vacuity
  partner proving the reference genuinely dies on the mixed ids the fallback exists for.
- **The shot arm clusters on `(provider, game_id)`, not `game_id` alone — a correctness fix, not
  just a crash fix.** This arm POOLS providers and `game_id` is unique only WITHIN one, so a
  stringifying repair would have silently merged gradientsports `123` with skillcorner `"123"`:
  two unrelated matches fused into one cluster, corrupting the very cluster-exchangeable null the
  placebo band is drawn from. The crash was the lucky failure mode; silent fusion was the other
  one. `_cluster_key` builds the composite only when a `provider` column is present, so
  single-provider callers keep the previous key exactly.

**No retrain.** No model, weights or feature values change; the affected surface is the causal
harness's placebo null, which is reported-not-gated. C4 count unchanged (32).


## [4.65.0] — 2026-07-27

### Fixed / Added — TF-19 power leg: degenerate resamples + a parallel spells pass (PR-S`<NN>`, ADR-037)

Both changes come from one **measured** failure: the first full corpus power run walked all 64
matches over 8.7h, then died in the cheap analysis step that followed and lost every spell it had
built, because nothing had been written to disk.

- **`causal.power.att_power_curve` no longer CRASHES on a single-class resample.** A cluster
  resample that happens to contain no treated unit is a **positivity failure at that size** — the
  ATT is not estimable — but it reached `fit_propensity` and raised out of sklearn
  (`This solver needs samples of at least 2 classes`), killing the whole run. Such a replicate is
  now scored as a **non-detection** and **counted** in a new `n_degenerate_by_size` return key.
  Counting is the load-bearing half: power 0.2 with most replicates inestimable is a completely
  different statement from power 0.2 with none, and dropping them from the **denominator** would
  condition on estimability and inflate the curve exactly where the design is weakest. A size whose
  degenerate count approaches `n_replicates` is reporting an inestimable design at that n, not a
  weak effect. `matched_n_by_size` reports `0` rather than a `nan` for an all-degenerate size.
  Byte-identical on any input that was already estimable — the guard only fires where the previous
  code raised.
- **`scripts/build_layer2_spells.py` (NEW) — the corpus pass is now its own shardable, resumable,
  partitionable driver**, mirroring `build_gkdv_arm_values.py`: per-match shards written on
  completion (an existing shard is skipped, so a crash resumes), `--match-ids-json` +
  `--list-matches` for N-way parallelism against a shared `--out`, fail-closed provenance, and a
  per-worker manifest. `run_signoff_power.py` gains `--spells <parquet>` to consume it, so a crash
  in the analysis step now costs seconds to retry instead of another corpus walk. The manifest
  reports corpus-scope `n_treated` / `treated_prevalence`, because a rare treatment is invisible
  per match yet decides the entire power curve.
- **`scripts/_partition.py` (NEW)** extracts the shard/partition/manifest plumbing now shared by
  both corpus producers. The reconciliation is the part that has already been wrong once — N workers
  writing one shared manifest let the last writer win, so a 64-match artifact reported a single
  partition's `n_matches: 8` — and that defect must not be fixable in one producer while still live
  in the other. Adds a **commit-consistency** check across workers (nothing stops one being launched
  from a different checkout, which would make the corpus artifact a blend of two code versions while
  looking like a single run) and OR-s `run_tree_dirty`.
- **`run_signoff_power.py` records — and refuses — UPSTREAM provenance.** Both its inputs
  (`--spells`, `--arm-values`) are produced by *other* drivers at *other* times, and the rule this
  repo already states is that an artifact whose inputs came from another driver needs provenance on
  **both**, or the clean SHA on the downstream metrics launders the dirty upstream input. It now
  reads each input's sibling manifest into `upstream_provenance` and exits on a dirty upstream, on a
  **missing** manifest (unprovenanced is treated exactly like dirty — a first draft early-returned
  here and silently accepted it, which is why the test asserts the raise), and on an upstream whose
  workers ran at **different commits** — every one of which is individually clean while the table is
  a blend of code versions.
- **Combined tables are written ATOMICALLY (`_partition.write_table_atomically`).** Every worker
  rebuilds the combined table from the shared shard directory and writes the same path, so N workers
  means N concurrent writers on one file, which can be read — or left — half-written. Each worker now
  writes a private temp file and `os.replace`s it into position. The 64-match arm-values pass got
  away with the old code; that was luck, not safety, and both producers are fixed.
- **A partition with NO ids for a provider now drops it, instead of silently expanding to the whole
  corpus (`_partition.providers_for_slice`).** The shared loader resolves a provider's ids as
  `(match_ids.get(provider) if match_ids else None) or list(manifest_ids)` — and an empty list is
  falsy while an absent key is `None`, so **both fall through to the ENTIRE manifest**. Verified
  directly against `_wanted_for_provider`: a slice of `{"idsse": []}` returned all seven manifest
  ids. That reading is correct for the loader's own callers (no slice means "everything") and
  exactly inverted for a partitioned one. Concretely: `validate_xshot_causal.py` defaults to three
  providers, so slicing on `gradientsports` alone would have made **every** worker load the full
  SkillCorner and IDSSE corpora — N-times duplicated work with N processes writing the **same**
  per-match shard paths concurrently. The single-provider arm-values pass escaped it only because
  its slices are never empty. Fixed in the partitioning layer, leaving `scripts/_loader_*`
  untouched; per-match shard writes are additionally routed through `write_table_atomically`, so
  even a same-match collision cannot tear a file.
- **`validate_xshot_causal.py` — the §3.3 entanglement pass — is now shardable, resumable and
  partitionable too.** It was the last driver still shaped like the failure above: ~81 matches
  walked serially, held in memory, writing nothing until the end, so any raise in the analysis
  discarded the entire pass. Now `build_shards` writes a per-match shard on completion (an existing
  shard is skipped; an EMPTY shard is written deliberately, because absent means "not yet run" and
  present-and-empty means "run, produced nothing" — conflating them makes every resume recompute
  the barren matches forever), `analyze_shards` runs the coverage + entanglement analysis over the
  persisted shards, and `--match-ids-json` + `--build-only` let N workers split the corpus against
  one `--out`. The provider is stored as a COLUMN rather than parsed back out of the filename,
  which would mis-split any provider containing the `__` separator. `metrics.json` gains a
  `corpus` block (`n_matches` / `n_opportunities` / `n_partitions` / `n_shards` /
  `commit_consistent`) because a partitioned run can legitimately analyse a subset and an artifact
  that does not state its own scope is the defect this release keeps finding; and a clean analysis
  over shards built from a DIRTY tree reports `run_tree_dirty: true`, so the analysis SHA cannot
  launder its input.
- **Two more drivers wired to the provenance guard, and the wiring is now CI-gated.**
  `validate_xshot_causal.py` had **no provenance at all** — and it produces the §3.3 entanglement
  measurement that corrects F6, i.e. a finding whose entire content is *"a registered default was
  mistaken for a measurement"*, which would have been published with no record of the code that
  produced it. `validate_xs_probe.py` stamped a bare `git rev-parse HEAD` behind a broad `except` —
  the exact pattern `_provenance.py` exists to eliminate — into the cited 4.60.0 xS-v2 artifact.
  Both now refuse a dirty tree from `main()` (enforcement at the entry point; `run()` records the
  truth unconditionally, so it stays directly testable) and stamp `run_commit` + `run_tree_dirty`.
  A hand-run audit found these two; `tests/scripts/test_provenance_wiring.py` is what stops the
  third — every artifact driver must import the helper, offer `--allow-dirty`, never shell out to
  `rev-parse` (matched via AST, so prose describing the defect is not mistaken for it), and call
  `require_clean_tree` from `main()`.
- **The `--help` ASCII gate is now DERIVED from `scripts/*.py`** instead of a hand-listed trio that
  had silently stopped covering most of the tree. It immediately found **18 pre-existing drivers**
  that die with `UnicodeEncodeError` on `--help` from a cp1252 console. They are pinned in an
  **exactly**-asserted debt list (fails both ways: a new offender cannot join silently, a fixed one
  must be removed) rather than repaired here, because two of them are `calibrate_*` — outside what
  this cycle may modify. Narrowing the gate to the files this PR touched would have hidden the
  other sixteen.

## [4.64.0] — 2026-07-27

### Added — trained-model feature contract + canonical penalty-area constant (PR-S135, ADR-050)

Two penalty-area half-widths have always existed in this repo (`_xcross_attempt` / `defensive_credit` use
20.16, `_ghost_gk` uses 20.15) and ADR-047 tracked unifying them. Unifying is **not behaviour-free**:
`_ghost_gk` uses the constant to compute `attackers_in_box`, one of the 26 `GHOST_GK_FEATURE_NAMES` — a real
input to bundled trained weights. Measured on a real WC2022 match, 70 of 175,969 frames (0.0398%) can flip it.
The number is beside the point; the mechanism is the risk — a geometry constant edited far from any model
silently re-defines that model's inputs, and no existing guard could see it (`chirality` fingerprints model
OUTPUT, which a low-weight feature can shift without moving; `geometry_version` covers the transform, not the
constants). So this ships the guard, then the constant.

- **`tracking/_feature_contract.py`** — a sibling of `_chirality.py` recording, per artifact, the feature
  VECTOR its extractor produces on a fixed probe frame **and** the geometry CONSTANTS that extractor consumes.
  A missing contract WARNS (pre-contract artifacts are undeclared, not known-bad); a probe change WARNS and
  skips the fingerprint comparison ONLY; a fingerprint or declared-constant mismatch RAISES with the model's
  own `IntegrityError`. Constants are compared FIRST and ALWAYS — including when the probe changed, since a
  sub-probe-resolution change (20.16 → 20.161) moves no feature and only the declared constant can catch it.
  Tolerance is chosen (`atol=1e-6`, `rtol=0`), not inherited from chirality, whose `rtol=1e-2` would be a
  0.17 m blind spot on a ~17 m feature.
- **Two new public warning categories**, `tracking.MissingFeatureContractWarning` and
  `tracking.UnverifiableFeatureContractWarning`, deliberately independent (neither subclasses the other,
  test-enforced). Declaring a constant REQUIRES extending the probe, which changes the probe hash for every
  saved artifact; one umbrella category would make escalating the missing-contract case silently turn every
  probe extension into a hard failure.
- **All three bundled artifacts stamped** (ghost, xS, xCross) via the committed, re-runnable
  `scripts/stamp_feature_contracts.py` — **metadata-only**: verified 654 ghost arrays bit-identical, both
  boosters byte-identical, all three metadata deltas additive-only. Ghost also gains the `pitch_length` /
  `pitch_width` fail-closed guard xS and xCross always had.
- **CI escalates `MissingFeatureContractWarning`** (`pyproject.toml`), adopting the ADR-041
  `SyntheticEPVWarning` mechanism — with **no** opt-outs, since every bundled artifact is stamped.
- **`tests/tracking/test_geometry_constant_enumeration.py`** — completeness by ENUMERATION (the ADR-043 idiom):
  AST-walks the four extractor modules, finds all 14 geometry constants, and requires each to be declared or
  explicitly exempt with a reason. Reads the BUILT contracts, both directions. This caught a real defect: xS
  had been declaring `penalty_area_half_width`, a constant it does not consume at all, which would have made
  the canonical flip raise on every xS load with xS's features provably unchanged.
- **`spadlconfig.penalty_area_half_width` / `penalty_area_depth`** (FIFA 40.32 / 16.5) + two **frame-explicit**
  predicates `in_penalty_area_absolute(x, y, *, attacked_goal_x)` and `in_penalty_area_goal_relative(gr_x, y)`.
  Two entry points because the call sites differ on FRAME, not just strictness, and a shared `goal_x` parameter
  would mean the *defended* goal in `_geometry` and the *attacked* goal at the xCross site — an 88.5 m error.
  Migrated the two sites already holding 20.16; **byte-identical**, grid-sweep proven, 159 tests unchanged.
- **`_ghost_gk` keeps 40.3** — its weights were fit on it. The contract now records that, so flipping it
  without a re-fit makes `load()` RAISE. (Its depth test is also strict `<` where the canonical helper is `<=`.)
- **Trainer guards:** ghost's feature cache is keyed on the geometry constants (a hand-bumped literal goes
  stale inside the very re-fit cycle it protects); the xS/xCross cache fingerprint is now a LIVE per-corpus hash
  (closing ADR-038's deferral) keyed on the REQUESTED corpus via a selection helper shared with `load_matches`;
  and unclassified providers fail at trainer startup rather than after the full extraction.

**No retrain, no value change.** No weights change and no model output changes. Three `metadata.json` files
gain a `feature_contract` key and their `SHA256SUMS` change — a consumer pinning artifact checksums will see a
diff. C4-free (count stays 32).

## [4.63.0] — 2026-07-26

### Added / Fixed — TF-19 corpus-run tooling (PR-S`<NN>`, ADR-037)

Maintainer-driver changes only; `scripts/` is not packaged, so **the wheel is byte-identical to
4.62.0**. These are the prerequisites for the §6.1/§6.4 corpus runs, landed together so the runs
execute from one immutable commit.

- **`scripts/_provenance.py` (NEW) — fail-closed run provenance.** `git rev-parse HEAD` returns the
  same SHA whether or not the tree is modified, so a driver stamping it records a commit that does
  **not** describe the code that produced the numbers — verifiable-looking and false, which is worse
  than recording nothing. Measured: a corpus pass was launched from a tree with three modified
  drivers while HEAD read clean. `require_clean_tree` now REFUSES an artifact-writing run on a dirty
  tree (naming the dirty files and the SHA that would have been falsely recorded); `--allow-dirty`
  permits a dev run but the artifact still records `dirty: true`. Absent git counts as dirty, never
  clean. Wired into all three drivers: `run_signoff_power.py` (which previously stamped a bare SHA)
  plus `build_gkdv_arm_values.py` and `derive_opengoal_range.py`, whose artifacts previously carried
  **no** provenance at all — a clean SHA on the power metrics would otherwise have laundered a dirty
  input, since the arm-values table is what the ICC number derives from.
- **`build_gkdv_arm_values.py` gains `--match-ids-json` + `--list-matches`**, which is what makes the
  corpus pass parallelisable: split the id list N ways and run N processes against a shared `--out`.
  Without it a second process re-walks the corpus from the start and redoes work, because shards are
  written on COMPLETION rather than claimed up front. Ids are STRINGS in the manifest.
  `--list-matches` consumes the loader's own `_list_matches`, so the id set cannot drift from what a
  run would actually fetch. Serial 64-match cost is ~61 h; partitioned it is ~6–8 h.
- **`--help` no longer crashes on Windows.** All three drivers carried non-ASCII in their module
  docstrings (`Δ`, `—`, `§`); a cp1252 console raises `UnicodeEncodeError` before printing usage —
  on the machine the drivers are invoked from. Now ASCII-clean, with a parametrised regression test.
- `--out` is no longer required for `--list-matches` (listing writes no artifact).

## [4.62.0] — 2026-07-25

### Added / Changed — TF-19 §6.4 sign-off package (PR-S`<NN>`, ADR-037 amendment)

Makes TF-19 spec §6.4 (the GKDV discrimination harness) **signable**: it registers constants the spec
forbade registering bare, builds the power simulator its own gate declares a precondition, and splits
a verdict from the routing that hard-coded one hypothesis. Verifying the three known blockers against
code surfaced three more, all recorded as findings rather than quietly fixed.

- **Plasmode power simulator, two modes.** `silly_kicks/_group_metrics.icc_power_curve` (domain-free:
  values / groups / match blocks) discharges §6.1's registered precondition — `ICC_ANCHORS` shipped in
  PR-3 with a docstring promising "a power curve is reported at all three" that **no code could
  produce**, while §6.1 states the ICC gate "is registered only if detection at the anchor ... is >=
  0.8". New PUBLIC `silly_kicks/causal/power.py` supplies the ATT mode. Both are plasmode, never
  i.i.d.: real values, real clustering, injected known effects. A private numpy-only `_icc_fast`
  serves the permutation loop (`icc_one_way` is shipped and consumer-tested, so it is UNTOUCHED and
  the fast path's equivalence is **gated**, not assumed).
- **THE FIREWALL.** `att_power_curve` accepts **no outcome vector at all** — only an `InjectionSpec`
  recipe it draws from itself, freshly per replicate. Once Layer 2's design exists in code the same
  machinery could *run* it, answering TF-19's open question before the sign-off meant to authorise
  it; the observed outcome is therefore unrepresentable rather than merely refused. A call-count spy
  on `estimate_att` would be VACUOUS (the harness always calls it), so the guard is provenance, and
  its non-vacuity is demonstrated: with the guard removed, a fully duck-typed fake runs to completion.
- **Layer 2's DESIGN lands in the causal builder** (its STUDY does not). `OpportunityConfig` gains a
  covariate-threshold treatment axis alongside its action-occurrence one, an entry-anchor rule (a
  covariate treatment has no anchor action, so both arms anchor on spell entry), `outcome_max_distance_m`,
  and an outcome PARTITION (`Y_attempt` / `Y_close_attempt` / `Y_far_attempt`) computed from ONE
  labelling pass so the three share identical row masks by construction. `layer2_config()` registers
  the Law-defined 16.5 m binarisation. **Every shipped config is byte-identical** (all new fields
  defaulted; guarded per-config, not by the default alone). Treatment depth is `GK_r·cos(GK_theta)`,
  NOT `GK_r` — they agree only on the goal's centre line, and the wide case is explicitly tested.
- **`regate_routing`** splits routing from verdict, against §6.4's own pre-registered disclosure:
  `gated_clean_fail` now routes to `pending_layer2`, not unconditionally to GK feature engineering,
  which had made H2 unreachable by construction. **`regate_verdict` is byte-identical** — every
  recorded verdict stands, pinned by a golden over all input combinations.
- **Registered constants:** `ATT_RELATIVE_ANCHORS = (0.10, 0.15, 0.20)` (row 5's own anchor — the ICC
  anchor is a variance share and does NOT transfer to a spell-level ATT) and
  `LAYER3_HEADROOM_RANGE_FRACTION = 0.02`, committed **before** the measurement it multiplies.
- **Record correction (F6):** 4.60.0's `joins_with_caveat` rested on a **registered default**, not a
  measurement. `regate_verdict` reads `entanglement` only on a `pass`; the driver hard-coded it,
  annotated "inert unless the probe surprises with `pass`" — and v2 surprised, so the parameter
  documented as inert decided the verdict. `scripts/validate_xshot_causal.py` measures it properly and
  had never been run. Nothing shipped was overclaimed (`joins_with_caveat` is the conservative branch)
  but the attribution was false; corrected in ADR-037, TODO.md, CLAUDE.md and the CLI help.
- Also: `causal/_confounders.py` (Layer 2's tracking-confounder join, provenance REGISTERED as
  frames-computed and refusing a `fct_action_context` source, whose `bekkers_pi` is pre-ADR-045 for
  away teams until the lakehouse re-materializes); `scripts/run_signoff_power.py` +
  `scripts/derive_opengoal_range.py`.

- **BEHAVIOUR CHANGE — `compute_threat_pc` now REFUSES an unfitted xT.** Its `xt` parameter is typed
  as a required `ExpectedThreat`, but nothing enforced it: passing `None` did not raise, it returned
  **`0.0`**. A caller persisting a threat column would therefore have persisted structural zeros,
  and an ICC or power curve computed on them is degenerate while looking like a measurement. It now
  routes through the single shipped `require_fitted_xt` guard (ADR-041), so `None` raises
  `ValueError` and a bundled-variant NAME raises `NotImplementedError`. **Hyrum:** a consumer
  relying on the old silent `0.0` now gets an exception — it was never a supported input, and a
  zero threat is indistinguishable downstream from a real one.

- **`scripts/build_gkdv_arm_values.py`** persists per-frame GKDV arm values for the §6.1 ICC leg —
  the expensive pass (accessible-space + Spearman pitch control on every domain frame, twice), run
  ONCE so the power simulator resamples the persisted table. **Per-match shards with resume** (an
  existing shard is skipped; an empty shard is written deliberately so "not run" stays
  distinguishable from "ran, scored nothing") — measured at ~5 GB RSS and 2,224 scored frames from
  175,969 for one WC2022 match, where accumulate-then-write-once would neither fit nor survive a
  crash. Credits **only the DEFENDING keeper**: the serving seam emits a row per team's keeper and
  only the defender is substituted, so a naive pass-through doubles every frame and attributes half
  the deltas to a keeper who never moved — keeper-INDEPENDENT noise that compresses between-keeper
  variance toward zero, the mechanism behind xT-GK v2's "keeper-flat" reading on fabricated origins
  (ADR-036/PR-S113). The threat arm is REFUSED rather than defaulted: `ExpectedThreat` has no
  serialization anywhere (only fit/interpolator/rate), so it needs an in-process fit — a leakage
  decision for its own cycle.

- **C4:** the missing `silly_kicks.causal` container (public since 4.47.0, never modelled) with four
  verified relationships, plus a **completeness gate** that derives the subpackage set from the
  package tree and asserts each has a container. Nothing previously pinned the diagram to the code,
  which is why the gap survived every release; mutate-to-RED verified.

**No VAEP retrain** (no xfns, no aggregator). C4 count unchanged (32). The DGX runs that fill
`N_MIN_MATCHED` and the §6.1 curve are owner actions and are **not** in this release.

## [4.61.0] — 2026-07-25

### Changed / Added — TF-51 v2 defensive-credit refinements (PR-S132, ADR-049)

Four bounded refinements to the shipped v1 defensive-credit family (`silly_kicks/tracking/defensive_credit/`)
plus one bundled v1 bug fix, in one PR. Supersedes the ADR-046/047 **Opta** block-detection status (dropped
from the roadmap, permanent `pd.NA`).

- **B2 fix — `recovery_after_pass` game/period boundary.** The forward opponent scan is now scoped to the
  passer's own `(game_id, period_id)` before the search, so a failed pass near a game/period boundary can no
  longer "recover" into the next match (a foreign team_id read as a real opponent regain). NOT possession-scoped
  (a recovery *is* a possession change). Fewer false cross-game recoveries; lakehouse re-materializes.
- **Item 3 — line-break-gated through-ball.** `rule_failed_marking_through_ball` now fires on a genuine TF-32
  ward `between_lines` line-break (the pass straddling two adjacent same-line defenders), computed
  `home_team_id`-free in action-LTR via the single extracted `_straddle_core` (shared with `detect_line_breaking`)
  and precomputed once on `RuleContext` (candidate-gated so Ward clusters only successful passes). The provisional
  `through_ball_delta_xt_min` ΔxT param is **removed** (frozen dataclass → `TypeError` on the old kwarg). Fires on
  a different set of passes; lakehouse re-materializes.
- **Item 2 — lane-geometry `shot_block` blocker.** `rule_shot_block` credits the defender geometrically in the
  shot→goal corridor (distance-scaled cone, floored) rather than nearest-to-origin; the origin proximity threshold
  is dropped, the goalkeeper is excluded by both the `is_goalkeeper` flag AND a distance-along-lane cap (the GS
  flag can be all-False), with a nearest-to-origin fallback. New `lane_blocker` resolution mode +
  `shot_lane_cone_width_factor` / `shot_lane_max_t` / `shot_lane_min_half_width_m` params. Long-form gains a
  generic **`resolution`** column (11 cols) recording how each credited player was determined
  (`nearest` / `all_within` / `all_within_beyond_nearest` / `lane` / `nearest_fallback` / `anchor_actor`).
  `shot_block` may credit a different player; lakehouse re-materializes.
- **Item 1 — reverse-xT "position won" pressing lens (opt-in).** `DefensiveCreditParams(pressing_lens=True)`
  sizes the four xT-sized turnover rules by `xT(105−x, 68−y)` (rewarding regains near the opponent goal) instead
  of the validated `xT(origin)`; default off → byte-identical. Turnover rows tag `sizing="xt_pressing"`. New
  `SIZING_VALUES` / `ANCHOR_TYPE_VALUES` / `RESOLUTION_VALUES` closed vocabularies. DIVERGES from the validated
  standard and UNDER-VALUES last-ditch defending (documented in NOTICE + docstring).
- **Item 5 — pressure-commitment cue (additive, descriptive).** New `add_press_commitment` aggregator (+ atomic
  mirror) over `tracking/_press_commitment.py`: per action, whether the pressing defender COMMITS (drives in,
  positive) vs CONTAINS (jockeys, negative) — the least-squares slope of the defender's closing-speed over the
  pre-action window, with a closed `press_commitment_source` provenance vocabulary. NOT signed credit, ships
  aggregator-only (no `*_xfns`), no VAEP retrain. A practitioner concept (PSG / Luis Enrique; Sumpter coaching
  literature) attributed in NOTICE.

Also: shared `tracking._opponent_resolution.opponents_within` nearest-opponent core (consumed by
defensive-credit resolution + the press cue); `tracking._velocity_availability.velocity_unavailable_by_design`
extracted from `_das.py`; three `press_commitment*` feature-glossary entries; owner-gated GS e2e (match 10502)
extended with the quantitative acceptance table. **No VAEP retrain from any item** (no xfns). C4 action-coupled
aggregator count 31→32 (`add_press_commitment`).

## [4.60.0] — 2026-07-24

### Added — TF-19 xS-probe placebo v2 run result (PR-S131, ADR-037 amendment)

The post-lock deliverable for the 4.58.0 xS-probe placebo v2 (PR-S129): the ~64-match GradientSports run,
executed **from the lock commit `78ffc70`** (blindness — `metrics.json` records `lock_commit == run_commit ==
78ffc70`) on the DGX. **No `silly_kicks/` code change — the wheel is identical to 4.59.0**; this ships only the
research artifact under `docs/research/tf19_pr3b_xs_v2/`.

- **v1 (frozen random placebo): `no_valid_placebo`** — reproduces the 4.55.4 PR-3b baseline exactly
  (`placebo_p95 = 0.0`, the random-outfielder null is degenerate).
- **v2 (model-relevant-defender placebo): `pass` → re-gate `joins_with_caveat`.** The methodology worked as
  designed: the defender placebo cleared the `no_valid_placebo` gate (`placebo_p95 = 0.00057`, live) yet is
  **inert in the ratio** (weaker than `nearest_def = 0.00503`, so `max()` pins to it), the ratio prong passed
  (`gk_med 0.01548 / nearest_def = 3.08×`), and — the genuine decider — the **clustered dose-response permutation
  is significant: ρ = 0.436, p = 0.001 across all 64 games** (dose-responsive 2 m→4 m: 0.0155→0.0222). The
  non-gating attacker diagnostic p95 = 0.0.
- **`joins_with_caveat`** = the shot arm's GK→shot-occurrence effect is real and dose-responsive (probe passes) so
  it *joins* the TF-19 metric, but with the honest caveat that the banked causal SHOT arm's entanglement was
  `inside_band` (the GK contribution is not cleanly isolable from the xS positional confounders).

Converts the xS arm from PR-3b's `unmeasurable_at_dose` dead-end into a real, citeable `pass`. Research artifact
only — no default xfn, no retrain, C4-free (count unchanged).

## [4.59.0] — 2026-07-24

### Added — Feature-column glossary + `describe_level` (PR-S130, ADR-048)

- **`silly_kicks/feature_glossary.py`** — a pure Python registry of every *derived* feature column
  silly-kicks emits (341 entries: `FeatureColumn(name, definition, unit, emitting_module, attribution,
  higher_is_better)` keyed by base column name). Documents the `add_*` / `*_xfns` / atomic-mirror / spadl-
  enricher / vaep surface; **excludes** base schema columns (`SPADL_COLUMNS` etc.). `unit` is a closed
  `Literal` vocabulary; `emitting_module` names the metric's home/compute module (not the `features.py`
  monolith). Pure `glossary_to_json` (with `GLOSSARY_SCHEMA_VERSION`) + thin `dump_glossary` writer +
  `glossary_entry` / `undocumented_columns` accessors. The 171 combinatorial VAEP one-hots are generated
  from the spadlconfig vocabularies; the ~170 unique features are hand-authored with home modules + NOTICE
  citations.
- **`silly_kicks/reporting.py::describe_level(z, *, higher_is_better=True)`** — generic, NaN-safe,
  direction-aware z-score → verbal band (outstanding/excellent/good/average/below average/poor; NaN →
  unknown), the seed of the reporting/wordalisation layer.
- **CI coverage gate** (`tests/test_feature_glossary_coverage.py`): complete-by-construction — producers
  discovered **by inspection with an `__all__`-less fallback** (not `__all__` alone), every emitted column
  requires an entry (no undocumented, no stale), `emitting_module` importable + not-`.features`. Plus an
  `attribution`↔NOTICE hard linkage gate and a dump→reload→`describe_level` roundtrip e2e (both directions).
- **Additive documentation → no VAEP retrain; C4 count unchanged (31).**

### Changed — TF-7 pitch-control cache threaded through the `*_xfns` path (PR-S130, ADR-008 amendment)

- Every pitch-control-consuming `*_xfns` factory (`pitch_control_xfns`, `obso_xfns`, `space_creation_xfns`,
  `pausa_xfns`, `cover_shadow_xfns`, `gk_influence_xfns`, `player_influence_xfns`, `off_ball_run_value_xfns`)
  and the atomic mirrors gain a keyword-only `pitch_control_cache: PitchControlCache | None = None`,
  threaded into whatever builds the `PitchControlSurface` (caller-injection; `compute_features` untouched).
  A caller builds one cache and passes it to all factories so a multi-family VAEP pass computes each per-
  frame surface once. `None` default is **byte-identical to today** (default xfn lists stay cache-`None`) →
  **no value change, no VAEP retrain.** `xshot_occurrence_xfns`/`xcross_attempt_xfns` are out of scope (their
  `pitch_control_cache` param is reserved for the deferred `extended` variant). Guarded by an all-family
  value-identity test, a cross-family mis-keying test, a cross-family compute-once perf guard, and a wiring
  completeness gate.

## [4.58.0] — 2026-07-23

### Added — TF-19 xS-probe placebo v2 (relevance-matched defender null; PR-S129, ADR-037 amendment)

A pre-registered second variant of the registered xS-arm GK-substitution probe
(`silly_kicks/tracking/_model_eval.py`), whose ONLY difference from the frozen v1 is the placebo pool.
TF-19 PR-3b Part A (4.55.4) returned `no_valid_placebo → unmeasurable_at_dose` — not because there is no
GK effect (it is dose-responsive, ~3.1× the nearest-defender control) but because v1's *random-outfielder*
placebo was degenerate (`placebo_p95 = 0.0`, 66% zero), and that gate short-circuits before the clustered
dose-response ever runs. v2 swaps in the **model-relevant defenders** so the test runs and reaches a real,
citeable verdict.

- **`substitution_deltas(..., placebo=)`** — new keyword: `"random"` (frozen v1 default, byte-identical) or
  `"model_relevant_def"` (the ball-nearest defenders minus the `nearest_def`, mirroring the xS extractor's
  5-nearest-defender reference). A distinct-role `attacker_diag` population (≤5 nearest attackers, carrier
  excluded) is emitted for reporting but NEVER banded by `evaluate_xs_probe`.
- **`xs_substitution_probe_v2`** + **`PROBE_WRAPPERS["xs_v2"]`** — the v2 wrapper (reuses
  `evaluate_xs_probe` verbatim, relabels `rule`) and its registry entry (constants identical to v1; the
  sole difference is `placebo_pool = "model_relevant_def"`, self-documented).
- **Honest framing (baked into the report generator):** the defender placebo is a *weaker* control than
  `nearest_def`, so it is **inert in the ratio** (`max()` pins to `nearest_def`); its job is to clear the
  instrument-validity gate with a principled null and be a reportable fair null — NOT to move the bar. The
  ratio prong is near-certain to pass; **v2's real decider is the clustered dose-response permutation**, run
  for the first time.
- **Driver** `scripts/validate_xs_probe.py` gains `--variant {v1,v2,both}` + `--lock-commit`, writing a
  two-variant `metrics.json`/`report.md` (v1's `no_valid_placebo` and v2's verdict side by side + the
  attacker diagnostic) and recording the **lock-commit hash** for auditable blindness.
- **v1 is byte-identical** (frozen suite + a numeric pin on the pre-refactor random path); `evaluate_xs_probe`
  and the `xs`/`xcross` registrations are untouched.
- **Research instrument** — in no default xfn list, no VAEP consumer: **C4-free (count stays 31), no retrain
  trigger.** The ~64-match GS deliverable run is a post-lock owner step (blindness discipline).

## [4.57.0] — 2026-07-23

### Added — TF-51 per-event defensive credit/debit family (PR-S128, ADR-047)

New `silly_kicks/tracking/defensive_credit/` sub-package: proximity-gated signed defensive credit
attributed to individual defenders, sized by shot **xG** or the attacker's **xT at a turnover**
(`xT(origin)`, the validated Bischofberger/Bauer/Baca arXiv:2606.19931 sizing). Ten named rules
(three shot rules as a mutually-exclusive on-/off-target/blocked partition, four `xT(origin)`-sized
turnover rules, three resulting-shot-xG chained rules) + a per-team **bravery** rollup.

- **`compute_defensive_credits(actions, frames, *, xg_column, xt, ...)`** — long-form, one row per
  (action, credited player, rule); `signed_value` NaN when a rule fired but is unsizable, no row when
  it did not fire (ADR-043 "missing ≠ 0"). On-/off-target is resolved via a tri-state `_on_target`
  (goal → on-target; else an injected `on_target_column`; else the frame-based TF-48
  `shot_on_target_derived` fallback; unknown → the pressure rules abstain, so a **saved** shot is never
  mis-signed as a miss).
- **`add_defensive_credit(...)`** — the per-action aggregate (`defensive_credit_net`/`_plus`/`_minus`,
  `n_defensive_credits`), scoped to the **defending team** (the acting-team `−passer` rows live only in
  the long-form). The C4 +1 action-coupled aggregator (30 → 31).
- **`compute_bravery(actions, *, ...)`** — event-only, per-team: `bravery_shots`,
  `bravery_open_play_crosses`, `bravery_set_piece_crosses = NaN` (v1 column limitation, **exposed** with
  `n_set_piece_crosses_faced`, not silently dropped), and the known-domain headline
  `bravery_pct_known_domain`.
- Ships **no `*_xfns` factory** (F4 result-leakage, ADR-039/042; guarded). No atomic mirror in v1.
- Consumes the shipped `shot_blocked` / `cross_blocked` columns (4.56.0); a cross-provider
  `cross_blocked ⊆ cross-type` invariant is added to the block-detection contract suite.
- **Additive → in no default xfn list → no VAEP retrain.** Owner-gated GS e2e (real WC2022 match 10502)
  validates the family end-to-end.

## [4.56.0] — 2026-07-22

### Added — Block-detection converter columns `shot_blocked` / `cross_blocked` (PR-S127, ADR-046)

Prerequisite for TF-51 (per-event defensive credit): two nullable-boolean (`"boolean"`) columns on
every converter's SPADL output, surfacing the blocked-shot / blocked-cross signal that canonical SPADL
drops (a blocked shot is otherwise `shot`+`fail`). Registered in `SPADL_COLUMNS` (propagates to all
provider schemas via `**SPADL_COLUMNS` spread) + a shared `_blocked_flag` helper with 3-valued
semantics (`True`/`False` on shot/cross rows the provider encodes, `pd.NA` on non-shot/non-cross rows
AND on providers that cannot encode it — a non-applicable row is unknown, never `False`). Declared
`"invariant"` in the ADR-045 reflection registry.

**Per-provider coverage** — `shot_blocked`: Gradient Sports (`shot_outcome_type=="B"`, pining-probed),
StatsBomb (`shot.outcome=="Blocked"`, 12 real blocked shots in fixture 7298), Sportec/DFL
(`shot_outcome_type=="blocked"` minus own-team deflections), Metrica (`subtype.endswith("BLOCKED")`),
kloppy gateway (`ShotResult.BLOCKED`), Wyscout (tag 2101; mechanism-only). `cross_blocked`: Gradient
Sports (`crossOutcomeType=="B"`, open-play `cross` only, pining-probed) + Wyscout (tag 2101,
mechanism-only). `pd.NA` (unknown) elsewhere: Opta (unverified qualifier), SkillCorner (no signal,
real-data verified both tiers), StatsBomb `cross_blocked` (deferred, n=1), Sportec / Metrica / kloppy
`cross_blocked` (infeasible).

**Additive** — no existing column or value changes → **no VAEP/tracking retrain**; atomic-SPADL is
unaffected (it projects to `ATOMIC_SPADL_COLUMNS`, dropping the two SPADL columns). C4-free. Spec +
plan under `docs/superpowers/`.

## [4.55.4] — 2026-07-22

### Added — TF-19 PR-3b Part A: xS-arm GK-substitution probe RUN + recorded verdict (PR-S126)

First end-to-end run of the xS-arm substitution probe on 64 GS matches — `scripts/validate_xs_probe.py`
(reported-not-gated; loads GS matches, `build_ghost_frames` → `provenance_to_targets` → the registered
xS probe, pooling the tidy deltas per match), plus CI-safe seam + orchestration tests and a
`docs/PRIVATE_CONSUMERS.md` entry for the private probe symbols. **No `silly_kicks/` change — the wheel
is byte-identical to 4.55.3.**

**Verdict** (`docs/research/tf19_pr3b/`): `no_valid_placebo` → re-gate `unmeasurable_at_dose`. NOT a null
effect — the GK effect is real and dose-responsive (median |ΔxS| 2 m 0.0154 / 3 m 0.0200 / 4 m 0.0222,
≈3.1× the nearest-defender control, only 5.3% zero-fraction) and would apparently clear both prongs; the
blocker is a **degenerate random-outfielder placebo** (`placebo_p95 = 0.0`, 66.5% of placebo deltas zero —
the aggregate xS features barely respond to a single distant player moving 2 m), so the probe cannot
certify the apparent effect. Next lever = a GK-appropriate placebo / less-aggregate xS features — a
methodology gap, not "no signal". `baseline_commit = ed20ac7` (behaviour-identical to 4.55.3 for this arm).
Spec + plan under `docs/superpowers/{specs,plans}/2026-07-21-tf19-pr3b-xs-arm-probe-run*`.

## [4.55.3] — 2026-07-22

### Added — structural guard that the public-surface doctest CI step stays wired (PR-S125)

`tests/test_ci_doctest_wired.py` parses `.github/workflows/ci.yml` and asserts the semantic wiring
of the 4.55.2 `--doctest-modules` step (mirroring the rigor of `test_ci_slow_gating_wired.py`): the
step exists, targets `silly_kicks/`, runs on **every** leg (not gated on `matrix.primary` — doctest
output is version-sensitive), and its `--ignore-glob` — checked via `fnmatch`, exactly how pytest's
`--ignore-glob` matches — skips single-underscore private modules while KEEPING dunder `__init__.py`
and public modules. Without it, silently dropping the step or drifting the glob would stop enforcing
public examples with no test failing. Test-only; no library change, no new public API, no retrain.

## [4.55.2] — 2026-07-22

### Fixed — every doctest across `silly_kicks/` executes cleanly, and the public surface is CI-enforced (PR-S124)

The `test_public_api_examples` gate checked example SHAPE but never CONTENT, so 96 doctests
(91 failing + 5 that did not even parse) had drifted: illustrative `>>> func(actions, frames, xt)`
fragments referencing match data no docstring can conjure (raising `NameError`), plus malformed
`# ...  # doctest: +SKIP` comment lines in the calibration objectives. Every one is now either a
runnable self-contained doctest or the package's canonical indented RST literal block (the honest
form for anything needing a real match's frames), so the full-package sweep is clean
(141 passed / 0 failed). CI now runs `--doctest-modules` on the **public surface** (non-underscore
modules; dunder `__init__` kept) on every leg, so public examples stay executable; private-module
examples are kept correct but not executed, to bound CI wall-clock. Seven calibration symbols whose
demonstrations lived commented-out behind the malformed `+SKIP` were graduated to real literal
blocks and removed from `_EXAMPLES_DEBT` (the self-burning debt shrinks).

### Changed — `_run_values._safe_index_of` delegates to `id_compat.ids_match` (PR-S124)

The TF-35 off-ball-run valuation carried a local canonical-id resolver that existed only because
`PitchControlSurface.player_share`/`.player_surface` compared ids with a raw `==` (fixed in 4.55.1).
Now that those methods are dtype-safe, `_safe_index_of` delegates its match to the shared
`ids_match` seam instead of re-implementing a `canonical_id` loop — behaviour-identical (first
canonical match, `None` on NA/absent), locked by a dtype-invariance regression test.

No library behaviour change, no new public API, no retrain. C4 count unchanged (30).

## [4.55.1] — 2026-07-21

### Fixed — dtype-safe id resolution in the pitch-control decomposition (ADR-019; PR-S123)

`PitchControlSurface.player_share` / `.player_surface` and `_gk_influence`'s `compute_gk_influence` /
`compute_zone_closing_times` compared a caller-supplied `player_id` / `gk_player_id` scalar against
the frame's id column with a raw `==`, which silently matches nothing across dtypes (an `Int64` frame
id vs a `str`/`int` query) — so on mixed-dtype ids they RAISED `not found` (and `_player_influence`'s
`except ValueError → 0.0` turned that into a silent zero). Both now route through the public
`silly_kicks.id_compat.ids_match`. **Byte-identical on matched dtypes** — the live-cohort case, since
these consumers resolve the keeper from the same frame — so no value change on the shipped path and
no retrain; the same-source `player_team_ids == team_id` compare stays a raw `==` by design (ADR-043
decision 6). Closes the ADR-019 gap the id-scalar registry recorded against these methods.

### Changed — Databricks calibration loader is OAuth-native (`scripts/_loader_databricks.py`; PR-S123)

`_connect` now prefers a `DATABRICKS_TOKEN` PAT (CI / legacy) and otherwise authenticates via OAuth
U2M through a `databricks-sdk` profile (`DATABRICKS_CONFIG_PROFILE`, default `OAUTH`; authenticate
once with `databricks auth login`) — the workspace moved off PATs. Dev-tooling only, no library API
change.

## [4.55.0] — 2026-07-21

### Fixed — vector quantities are now consistent under coordinate reflection (`silly_kicks/reflection.py`; PR-S122, ADR-045)

A 180° point reflection (ADR-028: `x→105−x` **and** `y→68−y` for away-team actions) must
point-reflect **points**, **negate vectors** (`vx`/`vy`/`dx`/`dy`), leave **magnitudes** (`speed`)
alone, and swap **direction labels**. Every reflection helper enumerated an explicit column list,
so any column not on it — crucially `vx`/`vy`/`x_smoothed`/`y_smoothed`, none of which are in
`TRACKING_FRAMES_COLUMNS` — rode through untransformed and silently wrong.

New public module **`silly_kicks.reflection`** provides one seam: `reflect()` (registry-driven,
for schema-bearing tables) and `reflect_columns()` (explicit, kind-aware, for derived/pre-canonical
frames), over `TRACKING`/`SPADL`/`ATOMIC_SPADL_REFLECTION_KINDS`. `reflect()` defaults
`on_unknown="warn"`: an undeclared column is treated as `invariant` and warns
(`UndeclaredGeometricColumnWarning`) only on a geometry-shaped name — fail-closed lives in the CI
registry-completeness meta-assertion, not the runtime call. Eleven reflection sites migrated onto
the seam.

Two **live** defects fixed, both in `pressure_on_actor__bekkers_pi` on away-team actions: the
per-action velocity re-projection (away defenders were modelled running backwards, −38.9%) and the
never-re-projected ball row. SPADL / atomic outputs are byte-identical; no converter output
changes.

**Retrain / re-materialize (bundle with the 4.52.0 TF-35 recompute, one ordered pass):** away-team
`bekkers_pi` values change (home byte-identical) → re-materialize
`fct_action_context.pressure_on_actor__bekkers_pi` → retrain `ρ` (both variants) → re-run the
xT-GK v2 deep-zone gate (its GO-leaning verdict was measured on broken pressure).

## [4.54.0] — 2026-07-20

### Changed — Ghost-GK artifacts are parameters-only (`silly_kicks/tracking/_ghost_gk.py`; PR-S121, ADR-044)

`GhostGkModel.save()` no longer persists the three per-sample training arrays (`training_gk_x`,
`training_gk_y`, `training_leaves` — the raw per-frame goalkeeper positions of every training
sample). A distributed model artifact now carries **learned parameters only** — the two
gradient-boosted tree ensembles and their baselines — not the training corpus. RFCDE density
estimation needs those responses, so `predict_density` (and the whole KDE capability) survives
**only on a locally `fit()` model**, never a loaded one; a loaded parameters-only model raises a
density-specific error. The served position (`predict_mean` / `ghost_gk_x`, `ghost_gk_y`,
`serve_ghost_gk_positions`, the `gkdv/` engine) is **byte-identical** and the chirality fingerprint
is unchanged — so this is **not a VAEP retrain**.

- **Breaking (artifact format):** version `1.2.0` → `1.3.0` (`stores_training_data: false`); no
  released version can read a 1.3.0 artifact (a version-pin consideration for Hub-hosted models).
  The bundled `default` is migrated by a pure `load(old).save(new)` re-save — **no retrain** —
  and shrinks **7,376,181 → 764,418 bytes**.
- **Breaking (column):** `ghost_gk_density_spread` is retired from `compute_ghost_gk`,
  `add_ghost_gk` and `ghost_gk_xfns` (6 VAEP columns, was 9); `kde_backend` is removed from those
  signatures (still accepted by `predict_density`). The column has no numeric consumer; the
  lakehouse re-materializes the passthrough out.
- **`metadata.json` corpus-provenance block** — providers + counts only, never match ids, never a
  public/restricted split. Every trained artifact records it from live data; the migrated bundled
  `default` records what is honestly available at migration.
- **CI name allowlist** over every bundled weights directory — a new array name fails CI until a
  human classifies it as parameter-or-per-sample. This anti-rot control is the generalizable win.
- Retired: three ghost-GK scripts that scored the KDE mode (dead since ADR-016); real-model fft
  fidelity in CI (unmeasurable once artifacts are parameters-only — see ADR-044).

### Fixed — `from_variant("public")` served the restricted `sc_extended` artifact (xS / xCross; PR-S121)

`XShotOccurrenceModel.from_variant("public")` and `XCrossAttemptModel.from_variant("public")`
returned the Hub-hosted, owner-tier-restricted `sc_extended` artifact: no bundled `public/`
directory existed, so the name fell through to `from_hub` and was cached under `"public"`. It was a
stale alias (4.9.0 reserved the name; PR-S118 added `sc_extended` alongside it without re-auditing).
Fixed by an explicit `{"public": "default"}` alias resolved **before** the cache — the bundled
`default` metadata already declares `shipped_variant: "public"`, so the alias is the literal truth.
A serve-time identity gate pins it; ADR-038's gate operates at training time and cannot observe a
loader serving a mislabelled artifact. No caller passes `"public"` today.

## [4.53.0] — 2026-07-19

### Added — TF-19 GKDV v1: ghost-substitution engine + two gate-independent physics arms (`silly_kicks/gkdv/`; PR-S120, ADR-043)

New `silly_kicks.gkdv` package. GKDV values goalkeeper POSITIONING by counterfactual: how much
does the actual keeper's position change the attacking team's accessible space and threat, versus
a league-average "ghost" keeper in the same frame state? Both arms are expressed in
**attacker-value units as `actual - ghost`, so negative = deterrent** uniformly.

- **`build_ghost_frames`** — substitutes the ghost-GK position into a copy of the frames and
  returns `(counterfactual_frames, provenance, GkdvReport)`. Never mutates caller input.
- **`provenance_to_targets`** — turns per-frame provenance into the pre-substituted ghost targets
  the model-agnostic probe core consumes.
- **`delta_das`** — accessible-space arm. Pins ONE attacking direction on the FACTUAL frames and
  passes it to both legs, so the counterfactual cannot silently re-derive a different direction.
- **`delta_threat_suppression`** — threat-weighted pitch-control arm, via an injected fitted
  `ExpectedThreat`.
- **`aggregate_by_keeper`** — per-keeper aggregation keyed on the frames-resolved GK `player_id`.
  The gold-mart `player_key` is deliberately NOT used: it is an actions-grain lakehouse column,
  and a pure library module must not depend on a gold join. (CLAUDE.md's "use `player_key`, never
  raw `player_id`" convention is actions-grain; this aggregation is frames-grain, where every row
  carries a resolved `player_id`.)
- **Validation surface** — `behavioural_anchoring_verdict` + the pre-registered `ICC_ANCHORS`,
  `TERCILE_SEPARATION_M` and `EXPECTED_DIRECTION` constants, re-exported from the package.

**These arms are gate-independent.** The TF-19 *attempt*-arm gate failed cleanly (`tf19_ready:
false`, `gated_clean_fail`) and remains gated; the physics arms do not depend on it, which is why
they ship now. §6.4 Layers 0–3 and the xS-arm substitution probe (PR-3b) are NOT in this release.

`gkdv` depends on **public** `tracking` seams, on the repo-wide public `silly_kicks.id_compat`,
and on exactly ONE private tracking symbol — `_das._pin_attacking_direction`, confined to
`_das_port.py`, which has no public meaning because it encodes what the optional
`accessible-space` dependency expects of its input. Never the reverse: `tracking` must not import
`gkdv`, since the probe consumes ghost targets as data. Both directions — including the single
allowlisted private import — are pinned by `tests/gkdv/test_import_allowlist.py`.

### Added — supporting surfaces

- **`tracking.serve_ghost_gk_positions`** — positions-only ghost-GK seam with per-row provenance.
- **`tracking.GhostClampWarning`** — a dedicated warning category for the ADR-016 pitch clamp. The
  clamp already warned; this makes it *filterable and attributable* rather than one anonymous
  `UserWarning` among many, so a consumer can escalate exactly this condition to an error.
- **`tracking.compute_threat_pc`** — threat-weighted pitch-control facade, extracted from
  `_cover_shadows` and made public so `gkdv.delta_threat_suppression` consumes a supported seam
  rather than reaching into a private module.
- **`silly_kicks/_group_metrics.py`** (private) — `icc_one_way` and `group_spread`, lifted from
  `scripts/` so the between-keeper dispersion statistics live in the library under test rather than
  in an unversioned script. Deliberately private: the lakehouse computes its own statistical gates
  (`src/analytics/xg_calibration.py` precedent) and consumes model-validation results as verdicts,
  not as computations, so there is no downstream consumer to support.
- **`silly_kicks.id_compat.restore_id_dtype`** — the shared "restore the source dtype where that
  dtype can represent the result" rule (see Fixed). It lands in the promoted public module below;
  `tracking/_id_compat.py` is deleted this release, so there is no `tracking._id_compat` path.
- **`docs/PRIVATE_CONSUMERS.md`** — register of downstream code that knowingly imports silly-kicks
  private modules or pins their paths. Underscore modules carry no stability promise; this exists
  so a refactor can see its blast radius. The `_ghost_gk` path pin is the highest-risk entry
  because it degrades **silently** — no `ImportError`, just a weakened downstream guard.

### Changed — `_id_compat` is promoted to the public `silly_kicks.id_compat` (ADR-019)

**BREAKING (import path).** `silly_kicks/tracking/_id_compat.py` → **`silly_kicks/id_compat.py`**,
public-named with no underscore. ADR-019 makes routing every id comparison through this module
*mandatory* for every consumer; a seam consumers are required to use is public API by definition,
and the underscore was a false signal. It was also structurally wrong: **39 files across 6
packages** import it — `spadl/`, `vaep/`, `atomic/`, `causal/`, `gkdv/` and `tracking/` — so five
packages outside `tracking/` were reaching into a private *tracking* submodule, including two
function-local imports inside `spadl/utils.py` written to dodge a circular import. That reach grew
during this release: 3 files outside `tracking/` imported it at 4.52.0, and the ADR-019 fixes below
added the rest, which is the argument for promoting the seam rather than the argument against.

Relocating it to a private `silly_kicks/_id_compat.py` was considered and rejected: it would have
moved the problem rather than fixed it, and `gkdv` would still have needed a private-import
allowlist entry. Public naming is what makes the "public seams" claim true rather than restated.

**No compatibility shim, deliberately.** The one known downstream pin is an `import`, which fails
**loudly** at collection with `ImportError`. The silent-degradation risk `docs/PRIVATE_CONSUMERS.md`
exists to catch belongs to the *path-string* pin (`exec_visibility.py`), which this does not touch.
A shim would also have made the promotion cosmetic — nothing would ever migrate. That register's
`_id_compat` row is retired with its exit condition marked met and the one-line migration recorded.

**`tracking.defended_goal_x` is also now public**, exported through the same
`_gk_resolve` → `features` → `tracking/__init__` chain its three siblings already use. It is the
spec §4.2 pinned goal map; consumers must call it rather than re-derive the goal-side rule, and a
fork is exactly what §4.2 forbids.

### Changed — ADR-019 id-scalar boundary: enumeration replaces heuristic

The AST lint (`tests/tracking/test_id_compat_lint.py`) is **DELETED**, not widened, and replaced
by `PUBLIC_ID_SCALAR_ENTRIES` — a registry that invokes every public function taking an id-valued
scalar and requires identical output across value-equal scalars of different dtypes.

The lint had to go rather than grow. It was a NAME heuristic: it missed the ADR-027
`t != action_team` defect because the operands aren't named `*_id`, and it could not see
`_ghost_gk`'s `str(t) == home_team_id_norm` because the scalar had been *renamed*. Worse, the safe
and unsafe cases are the **identical AST** — only the scalar's *provenance* separates a same-column
compare from a public-parameter compare, and no syntactic rule can see provenance. Widening it
would flag correct code and breed exemptions; its glob had already missed 17 modules. **Complete by
ENUMERATION where the lint was incomplete by HEURISTIC**, the same idiom as ADR-003's NaN-safety
registry and ADR-033's `PURITY_ENTRIES`.

The surface is derived from `inspect.signature` over the `__all__` exports of `spadl/`, `atomic/`,
`vaep/`, `causal/` and `tracking/`, keyed by defining qualname so re-exports collapse: **102
functions — 77 exercised directly** (including the four live-defect `play_left_to_right` siblings,
name-pinned), **22 delegated** to `test_id_dtype_invariance.py` (machine-checked against that
gate's registered surface, not asserted in prose), and **3 justified non-invariant**
(`validate_id_dtypes` IS the diagnostic; `add_gradientsports_player_ids` assigns via `.mask` rather
than comparing; `PitchControlSurface` is a frozen result container). Two meta-assertions pin the
registry to the public surface in **both** directions, so a newly-exported id-scalar function fails
CI until registered, delegated or justified — the anti-rot property the lint's glob lacked.

Each entry is exercised on three axes: a matched scalar, a **mismatched-but-value-equal** one
(`5` vs `"5"`), and a **float** one (`5.0`). The third is not redundant — a naive
`str(value) == str(scalar)` renders identically for integers, so only the float axis exposes it.
Mutation-verified in both directions: re-planting the shipped `team_id != home_team_id` turns the
gate red on `play_left_to_right` alone and reproduces the original signature (on the gate's
three-row fixture `start_x` comes back `[95, 85, 75]` where the correct mirror gives
`[10, 20, 75]` — the HOME rows mirrored, not merely away rows missed), and
planting a naive `str()==str()` is caught by the float axis while the int/str axis passes it.
Non-vacuity is enforced throughout, with a `live_columns` lever for functions that return their
input frame plus a computed column. **Tests only — no library behaviour change.**

### Fixed — cross-source id comparisons in the action↔frame context kernels

`tracking/utils.py`'s `_ids_equal_cols` / `_ids_differ_cols` routed through `_directly_comparable`,
which short-circuits object-vs-object to a raw `==`. But these compare an **action** column against
a **frame** column — cross-source by construction — which is precisely the shape a boxed-numeric
object id column mis-resolves. **This was a live correctness gap in a public seam, not stale prose.**

**Value change / re-materialize trigger.** A boxed-numeric object id column that previously
resolved to *nothing* now resolves correctly, so `_resolve_action_frame_context`'s masks change for
any consumer feeding one — that is every action↔frame context kernel. Rows that were silently
unresolved now carry real values. No provider inside silly-kicks was measured to be in this state
(all eight converter paths are proven dtype-matched), but an external frame builder may be.

`canonical_id_series` also no longer raises on infinities or out-of-int64-range floats: those
routed into an `Int64` cast that cannot represent them (`OverflowError` / `TypeError`) while the
scalar `canonical_id` handled them, so the vectorized path now matches the scalar truth as its
docstring has always claimed. Bounds are exactly-representable float powers of two — **not**
`float(np.iinfo("int64").max)`, which rounds *up* to 2**63 and would re-open the bug.

### Fixed — ADR-019 id-dtype: two stacked defects that silently emptied id joins

A boxed-numeric object id column (an object column holding `2.0`) could not be matched against the
same id carried as a number or a string. The failure mode was an **all-False mask — a silent
all-row join miss, never an error**. `infer_ball_carrier` shipped a live instance of the shape.

- **`canonical_id_series` violated its own contract.** Its docstring promises it "matches
  `canonical_id` element-wise"; its object branch bare-stringified, rendering `2.0` as `"2.0"`
  where the scalar truth gives `"2"`. It now probes object CONTENT (`infer_dtype`) and routes
  boxed-numeric or mixed columns element-wise through the single `_canonical` truth. The
  genuine-string fast path for sportec/kloppy is preserved and asserted non-vacuously.
- **`ids_equal` / `ids_differ` / `ids_match` short-circuited object-vs-object to a raw `==`**, on
  the assumption that two object id columns are both genuine strings. They now content-probe, so
  they no longer contradict the module's own canonicalization. Measured cost is ~15% of the comparison per probe; `_raw_comparable` probes BOTH sides, so the
  guard as actually paid costs ~30% of the raw `==` it guards on a 500k-row column.

### Fixed — `infer_ball_carrier` leaked an object column for 3 of 5 source dtypes

Its dtype restoration was keyed on the single literal `"Int64"`, so `int64`, `float64`, `object`
and `string` sources all fell through and emitted `object` — harmlessly for an `object` source,
which is already its own dtype, and as a boxed-numeric leak for the other three. All three sibling
restoration sites
(`_ball_carrier.py`, `features.py`, `_gk_resolve.py`) now share one `restore_id_dtype` rule.

**Restorability, not blanket casting:** a numpy integer dtype cannot hold NA, so a result with
unmatched rows deliberately stays float. That NA rule is the long-standing behaviour at the two
sibling sites and is preserved exactly; the *dtypes* are not — see below.

**BREAKING (observable), precisely:** `ball_carrier_team_id` / `ball_carrier_player_id`, and the
`player_id` Series returned by `ball_carrier_at_action` / `acting_gk_from_frames` /
`defending_gk_from_frames`, change dtype for **three of the five** source dtypes. Two are
byte-unchanged: nullable `Int64` (the one literal the old code handled) and `object` (object *is*
its source dtype, so there is nothing to restore). The three that move:

- **`int64`** — was `object` at the `_ball_carrier` site and `float64` at the two sibling sites; now
  `int64` when no row is missing, still `float64` when one is, since numpy ints cannot hold NA.
- **`string`** — was `object`, now round-trips to `string`.
- **`float64`** — was `object`, now round-trips to `float64` (with or without a missing row).

### Fixed — ghost-GK scoreline used a naive string id comparison

`_build_score_lookup` classified each goal's scoring team with
`str(t) == str(home_team_id)`. On a float-backed id that renders `"1.0"` against `"1"`, so **every
goal fell to the away side**. Measured on a 3-goal fixture (2 home, 1 away): `score_diff` returned
**−3 instead of +1**, a four-goal swing — and `score_diff` is one of the 26 **trained** ghost-GK
features. Routed through `ids_match`, which is also vectorized and so cheaper than the per-element
Python loop it replaces.

**Latent, so no retrain.** The path is opt-in (`actions=None` leaves `score_diff` at 0.0) and no
shipped provider is float-backed — Gradient Sports emits nullable `Int64` and the kloppy family
emits object strings, both of which stringify correctly — so the bundled ghost-GK weights were fit
on correct values. It is a serve-path repair for any external caller supplying float ids.

### Fixed — ghost-GK feature extractor id comparisons (ADR-019)

`extract_ghost_gk_features` compared player and team ids with raw `==` / `!=` at seven sites. On a
provider whose frame ids differ in dtype from the caller's scalars — the Gradient Sports nullable
`Int64` case ADR-027 documents — those comparisons silently resolve to False, and the extractor
then builds features against the wrong keeper or an empty selection. Routed through the mandated
`_id_compat` seam. VAEP-invariant for matched dtypes, so no retrain.

### Fixed — 180° mirroring on a mismatched `home_team_id` scalar

Six `play_left_to_right`-family sites compared `team_id != home_team_id` with a raw operator
(`spadl/utils.py`, `atomic/spadl/utils.py`, `vaep/features/core.py`, `atomic/vaep/features.py`,
`spadl/orientation.py` ×2). On an object-string `team_id` against an int scalar the comparison is
True for EVERY row, so **home rows were mirrored too**, not merely away rows missed. All six now
route through `ids_match`, and the four `home_team_id: int` annotations widen to `int | str` —
the annotation was the actual trap, since absolute-frame actions come from string-id providers.

Measured as a **latent** fix: all eight converter paths were empirically proven dtype-matched, so
no bundled weights and no current consumer output changes.

### Fixed — DAS output-alignment guard

`_das.py` compared the accessible-space return length against the prepared frame and, on a
mismatch, emitted a `UserWarning` ("output may be misaligned") and then proceeded with the
assignment anyway. Because accessible-space legitimately drops rows whose `team_in_possession` is
NaN — most rows on dead-ball-heavy providers — that warning fired on essentially every real
provider match. The result was persistent warning noise on correct output, which trains callers to
ignore the one signal that would matter if alignment ever genuinely broke.

The check is now an **index-subset assertion**: the returned index must be a subset of
`prepared.index`. A legitimate shrink passes silently, and accessible-space restores caller index
labels, so the assignment was always label-correct — only the check was wrong.

**BREAKING (observable):** this is also a *strictening*. Where the old code only warned, a foreign,
shifted or positionally-reset index — and a length mismatch on an index-less return, which is still
assigned positionally — now raises `ValueError`. A caller relying on the previous
warn-and-continue behaviour will see an exception instead.

### Fixed — DAS failures are narrowly caught and NAMED, not silently NaN-ed (`silly_kicks/tracking/`, `silly_kicks/calibration/`; PR-S120, ADR-043)

`add_das` / `das_at_action` / `das_xfns` caught a broad
`(ValueError, RuntimeError, ImportError, IndexError, TypeError)` and degraded to an all-NaN DAS
column plus a warning. Two defects followed. (1) The tuple swallowed silly-kicks' OWN bugs: a
missing `vx`/`vy` column, the `_check_das_output_alignment` integrity breach, an accessible-space
signature drift, and the `ImportError` for the optional `[das]` extra all became "DAS is NaN
here". (2) An all-NaN column is indistinguishable downstream from legitimately-absent DAS — not
hypothetical: `calibration/_features.py` carried a private `das_ok` flag, plus a full
re-implementation of the DAS lookup, purely to work around it. The catch entered in TF-28 with no
stated rationale and was widened in PR-S60 for degenerate Voronoi + NaN coordinates.

**Narrowed catch.** New `tracking.DasUnscoreableError(ValueError)` is the ONLY exception the three
entry points degrade on, raised for exactly the conditions the catch existed for: the dead-ball
window (all-NaN `team_in_possession`, at both `_pin_attacking_direction` and
`_precompute_das_lookup`) and — converted at the library seam by the new `_das._call_simulation` —
accessible-space's degenerate-Voronoi `IndexError` and NaN-coordinate `TypeError`.
`_call_simulation` BINDS the call before entering its guard, so an accessible-space signature
change raises loudly instead of being mistaken for the NaN-coordinate `TypeError`. Subclassing
`ValueError` keeps consumers that catch the broad `ValueError` working. Everything else PROPAGATES.

**Provenance.** `add_das` emits a new `das_source` column over the closed
`tracking.DAS_SOURCE_VALUES` vocabulary — `computed` / `unlinked` / `unscoreable_frame` /
`team_unresolved` / `unscoreable_call` — so "DAS could not be computed" is distinguishable from
"DAS is genuinely absent for this action", per row and per cause. `das_at_action` returns a bare
Series and carries no provenance — its docstring routes callers to `add_das`. `das_xfns`
deliberately does NOT emit the string column (VAEP feature matrices stay numeric), guarded by a test.

**`das_ok` deleted.** `calibration/_features._compute_das` no longer re-implements the DAS lookup
behind a private try/except: it pre-restricts frames to the action-linked `(period_id, frame_id)`
pairs and calls the public `add_das`. **BREAKING (observable):** `calibration.enrich_invariant` —
public, exported in `calibration/__init__.py`'s `__all__` — returns a **2-tuple**
`(base_actions, links)` instead of the 3-tuple `(base_actions, links, das_ok)`, so a caller
unpacking three values now raises `ValueError`. `das_ok` has no replacement value because it has no
remaining meaning: `_vaep_brier_objective` reads M8 off the per-row `das_source` column instead,
which distinguishes the causes `das_ok` collapsed into one bool. DAS values are unchanged, pinned by a test that
replays the removed inline algorithm verbatim as an oracle; the routing does pick up `add_das`'s
dtype-safe (ADR-019) team match where the deleted code used a raw `dict.get` / `!=`.

**Three CI gates were VACUOUS for DAS and now have teeth.** The broad catch meant `add_das` /
`das_xfns` returned an all-NaN column before ever reaching the behaviour under test, so the
ADR-003 NaN-safety gate, the ADR-019 dtype-invariance gate and the ADR-020 dup-`action_id` gate all
passed without exercising the family. Each now supplies DAS's contract columns
(`vx`/`vy`/`team_in_possession`). Mutation-verified: reverting the ADR-019 dtype-safe team match to
a raw `==` now turns the dtype-invariance gate RED, which was previously impossible.

**Hyrum / consumers.** `add_das` gains a column (additive; the lakehouse re-materializes to pick it
up). A caller passing malformed frames now gets an exception where it previously got a mute NaN
column — that is the point. Direct `get_individual_das` / `get_das` / `get_xc` callers see the
library's degenerate-geometry `IndexError`/`TypeError` arrive as `DasUnscoreableError` instead; a
catch on `ValueError` is unaffected, a catch on `IndexError`/`TypeError` specifically is not. No
DAS *value* changes, so no model retrain.

### Fixed — velocity-availability contract, ADR-020 retrofit, possession idempotency (PR-S120, ADR-043)

Three further defects surfaced by the narrowed catch above.

**(1) `speed_source` gains a third token `"unavailable"`** (`silly_kicks.tracking.SPEED_SOURCE_UNAVAILABLE`,
registered in `TRACKING_CATEGORICAL_DOMAINS`): a frame builder DECLARES that its source has no
per-player temporal history, so `speed` — and the `vx`/`vy` that `derive_velocities` produces from
that same history — can NEVER exist. Deliberately distinct from a NULL `speed_source` ("not derived
YET"): without the distinction a velocity consumer cannot separate "this data structurally has no
velocity" from "the caller forgot `derive_velocities()`", and the two demand opposite responses.
`snapshot_to_tracking_frames` (the StatsBomb-360 freeze-frame bridge, one synthetic frame per
action) stamps it on every player AND ball row; THIRD-PARTY builders may set it deliberately — it is
a public contract, not a snapshot backdoor. `_validate_das_inputs` reads it: ALL rows marked →
`DasUnscoreableError` → `add_das` degrades to NaN with `das_source="unscoreable_frame"` (warned);
UNMARKED or PARTIALLY-marked frames missing `vx`/`vy` still RAISE loud (the fail-loud branch wins on
a mixed frame set), and the marker never excuses a missing `team_in_possession`.
`DasUnscoreableError` carries a validated `das_source` so the raiser names its own provenance.
`DAS_SOURCE_UNSCOREABLE_FRAME` accordingly widens from "the linked frame carries no DAS" to "the
FRAMES, not the computation, are why DAS is absent" (per-action OR structural), leaving
`unscoreable_call` for the re-runnable failures.

**(2) `das_xfns` dup-`action_id` fix (ADR-020).** `_map_das_to_actions` resolved frame ids via
`pointer_lookup.at[aid, "frame_id"]`, which returns a Series on the non-unique `action_id` that
VAEP shifted gamestate slots carry at period boundaries. ADR-020 fixed this exact shape in **8**
other frame-aware families (`pitch_control` / `obso` / `pausa` / `space_creation` / `pressure` /
`cover_shadow` / `gk_influence` / `player_influence`) and `das_xfns` was missed — the
auto-enumerating gate probed it VACUOUSLY, because the pre-ADR-043 broad catch returned all-NaN
before the bug could surface. Now routed through `_kernels.resolve_frame_ids_by_position` like the
other 8 (byte-equivalent on `add_das`'s unique-id path).

**(3) `derive_team_in_possession` is IDEMPOTENT.** A `frames` already carrying
`team_in_possession` / `ball_carrier_player_id` came back with `_x`/`_y` suffixes, and every
consumer that re-derives possession (`_xshot_occurrence`, `_xcross_attempt`) then died on
`KeyError: 'team_in_possession'` — live for exactly the multi-family pipeline the `links` /
`pitch_control_cache` kwargs exist to encourage. Pre-existing columns are now REPLACED (not
preserved): the contract is "possession according to THIS `carrier`", so a retained column from a
different carrier config would silently disagree with the argument just passed. Mirrors the
"linkage-provenance columns are idempotent" `add_*` convention.

### Changed — the public-API Examples gate's module registry is pinned to the real surface

`_PUBLIC_MODULE_FILES` was hand-maintained with nothing tying it to reality, so a newly-added
public module was **silently missed rather than caught** — it simply never entered the
parametrization. Same incomplete-by-heuristic class as the AST lint deleted above and the
non-recursive `gkdv` allowlist glob, and this release proved it live: two public modules were added
and one was registered only because a human noticed.

The surface is now DERIVED as the union of **P1** — modules that *define* a symbol some package
exports via `__all__`, which is how underscore-named modules like `tracking/_ghost_gk.py` become
public in practice — and **P2**, modules reachable by an underscore-free dotted path
(`spadl/statsbomb.py`, which re-exports nothing and P1 alone would miss). Meta-assertions pin the
registry in both directions. `test_registered_modules_are_still_public` additionally guards the
derivation's own health: P1 depends on importing packages, so a package that failed to import would
otherwise take its re-exported modules out of the surface **silently**, and a shrinking surface
always stays a subset of what is accounted for — the anti-rot assertion alone cannot see it.

**Vacuous entries removed.** `gkdv/__init__.py` was registered while having zero top-level defs: it
read as coverage for `gkdv` while all four modules defining its public surface were unchecked. All
four already carried Examples, so registering them directly took gkdv from **0 to 8 enforced
symbols**; the identical `causal/__init__.py` entry went the same way, losing no coverage.
`test_no_registered_entry_is_vacuous` blocks the shape recurring.

Enforcement moves from **56 hand-listed modules carrying 220 public symbols** to a derived
**118 modules / 354 enforced symbols**, measured from the registry rather than quoted. The
derivation pulled in **63 further public modules** carrying 359 public symbols, 204 of them
undocumented; a further **21** undocumented symbols surfaced *inside* the original 56 once the two
tightenings below landed. All **225** are enumerated in an `_EXAMPLES_DEBT` bucket with a written
note each rather than quietly documented or quietly ignored. The bucket is
**self-burning-down**: a meta-assertion requires every entry to still have an undocumented symbol,
so finishing one turns CI red with an instruction to promote it. It shrinks monotonically and
cannot silently absorb a new module — a new module lands in neither bucket and fails.

### Added — Examples sections for three newly-public symbols

`_group_metrics.icc_one_way` / `group_spread` and `tracking.GhostClampWarning`. Every documented
value was verified by execution. The ICC example records two non-obvious properties: ICC(1) is
**not bounded below at zero** (identical group means with noisy members score ≈ −0.26), and
singleton groups are dropped rather than allowed to inflate the estimate. `group_spread`'s shows
that `min_n` filters *before* computing, so `n_keepers` counts survivors, and that a thin cohort
returns the declared shape with NaN rather than raising. `_group_metrics.py` is deliberately **not**
registered in the gate: it is private by name, re-exports nothing, and its own docstring states
that promotion is a deliberate, requested step.

### Fixed — xCross score_differential sign flipped on float-backed ids

`_xcross_attempt` signed `score_differential` with `str(poss_team) == str(home_team_id)` at two
sites. On a float-backed id that compares `"5.0"` against `"5"` — always False — so **every row's
sign inverted** (`[0.0, 1.0]` → `[-0.0, -1.0]`). The module already imported the `id_compat` seam;
these two call sites simply predated it. Both now use `same_id`.

**Found by the new registry gate on its first run**, which is the case for enumeration over
heuristics: the deleted AST lint could not have seen either site, because the scalar is compared
through `str()` rather than named as an id. Latent for shipped providers (Gradient Sports emits
nullable `Int64`, the kloppy family emits object strings), so the bundled xCross weights were fit
on correctly-signed values and **no retrain is required**.

### Changed — `+SKIP` filler no longer counts as an Examples section

`_has_examples_section` accepted any `>>>` line — and a bare `Examples` header with nothing
under it — so `>>> f(x)  # doctest: +SKIP` ticked the box while demonstrating nothing runnable.
**16 of the 354 enforced symbols passed that way**, and the companion escape below accounts for a
further 74. Replaced by `_has_real_example`, which requires
either a doctest that would actually execute (not `+SKIP`, not a `...`/`pass` placeholder) or an
indented illustrative literal block — this package's canonical style, since most entry points need
a real `actions` frame no docstring can conjure. The rule is stated once in `_REAL_EXAMPLE_RULE`
and quoted verbatim by the failure message, so whoever trips it is told what to write rather than
how to silence it.

The illustrative arm is **scoped to the Examples section**, which is load-bearing: a NumPy
`Parameters` block is indented too, and an unscoped check is rescued by essentially every docstring
in the repo — during development that silently shrank the offender set from 16 to 13.
`test_skip_only_rule_is_scoped_to_the_examples_section` pins it.

Fallout split honestly: the **4 offenders in this release's own `gkdv/`** (`delta_threat_suppression`,
`delta_das`, `build_ghost_frames`, `provenance_to_targets`) got real examples, each teaching its
failure mode — differencing all frames instead of the `drop_reason.isna()` ones, letting
accessible-space infer opposite directions per leg, and passing `provenance` straight to the probe
(which selects the *attacking* keeper). The **12 pre-existing offenders** were given real examples
too — enumerated in the debt-granularity section below — rather than having the check weakened
around them.

### Added — `silly_kicks/id_compat.py` ships documented, not in debt

The ADR-019 id-identity module promoted this release carries Examples on all **8** public symbols
and moves out of `_EXAMPLES_DEBT` into the enforced registry. The examples teach the contract
rather than the signature: `ids_match` resolving an `Int64` column against a string scalar beside
the raw `==` that returns an all-False mask **instead of raising**; `ids_equal` shown positional
against the `ValueError` pandas raises on label-aligned Series; `ids_differ` **not** counting an NA
as "differs", so it is not `~ids_equal` and the two masks do not partition the frame (the ADR-027
NaN-actor rows must not read as opponents); `restore_id_dtype` returning `float64` from an `int64`
source when one row is missing — because numpy ints cannot hold NA — while a nullable `Int64`
source round-trips exactly; and `align_join_keys` as the fail-loud seam preventing the
numeric-vs-object merge error. All 48 doctests verified by execution on pandas 2.3.3 **and** 3.0.3.

### Changed — the Examples debt bucket is per-SYMBOL, not per-module

A module-level exemption cost far more than it excused. When the `+SKIP` tightening demoted 12
filler examples, four whole modules left enforcement and took their **already-documented** symbols
with them — a net coverage reduction hiding inside a change meant to tighten the gate. Measured
with the gate's own shipped predicate against the shipped bucket, a module-level key would
un-enforce **129 already-documented symbols** out of the 354 public symbols in those 42 modules;
`tracking/features.py` alone carries 30 gaps and would have taken its **54 documented symbols**
out with it.

`_EXAMPLES_DEBT` is now keyed `"<file>::<qualified_name>"`, every public module is enforced, and an
exemption costs exactly the symbol it names. Every invariant is preserved and re-pointed at the
finer granularity: self-burning-down and per-symbol disjointness both land on
`test_debt_entries_are_really_undocumented` (a documented symbol carrying an exemption is
enforced-and-excused at once, and fails); full accounting gets its own single-assertion
`test_every_public_symbol_is_documented_or_excused`; and `test_debt_entries_name_real_public_symbols`
adds a prong a module-level bucket **structurally could not have** — a symbol renamed or deleted out
from under a still-valid file entry. A module-level key is rejected by construction.

Shipped alongside: the **12 `+SKIP` offenders are documented rather than bucketed** —
`calibration/_gates.py` (2), `calibration/_vaep_brier_objective.py` (3), `calibration/_xt.py` (4)
and `tracking/_shot_goalmouth.py` (3) — seven as runnable doctests with every expected output
verified by execution (the fail-closed xT-corpus exclusion, the `load_xt` sha256 refusal, the H1
gate's two *non-firing* anchors). Docstrings only in `calibration/`; no logic or signature change.

### Fixed — two escapes in the Examples gate itself

**(1) A bare import counted as a demonstration.** `_has_runnable_doctest` asked only whether *some*
doctest line was unskipped, never whether it *showed* anything — so `>>> from x import f` on line
one let every line demonstrating the call stay behind `# doctest: +SKIP`, and a
`>>> # see tests/… for a runnable example` comment did the same job. `_demonstrates_something` now
requires an unskipped statement that is more than an import or a comment, judged by **parsing** the
reconstructed statement rather than matching text, so a multi-line `from x import (a, b)` is still
recognised; an import *followed by* a real call is unaffected. The camouflage hid **74** further
symbols — **4.6× the 16** the `+SKIP` rule caught head-on, which is the measure of how much a rule
inspecting an example's FORM misses about its CONTENT. Two symbols this release touched got real
examples; the other 72, all identical at 4.52.0 and all needing a real match's frames, are tracked
as individual debt entries with written notes.

**(2) An unclearable debt entry.** `_walk_public_definitions` did not skip `@overload`, so the gate
demanded an Examples section on a stub whose body is `...` — an entry that could never burn down,
defeating the bucket's core property. `_is_overload_stub` skips them, keyed on **the decorator each
definition carries rather than on its name**, so an implementation sharing the name stays judged.
`prepare_ghost_gk_training_data`'s entry is retired — and the gate itself drove the removal, going
red to say the entry was now clearable. It is the package's only overloaded function — two
`@overload` stubs on `prepare_ghost_gk_training_data`, the only two in `silly_kicks/`.

**Known gap, measured not guessed:** nothing in CI executes doctests, so the "a doctest that
actually runs" arm accepts examples nothing verifies. Running `doctest.testmod` over every module
in `silly_kicks/` gives **531 attempted, 141 failing across 22 modules** — overwhelmingly
illustrative fragments that reference names bound in a preceding `+SKIP` line, not false claims,
and overwhelmingly predating this release. A further **5 `calibration/` modules cannot be collected
at all**: a `# doctest: +SKIP` directive sitting on a commented-out line makes `doctest` itself
raise `ValueError`, so their examples are neither run nor counted above. Closing this is a body of
work in its own right rather than a gate tweak.

### Changed — CI installs the `das` extra on every matrix leg

`.github/workflows/ci.yml` installs `.[kloppy,xgboost,das,test]` in place of
`.[kloppy,xgboost,test]`. The TF-28 DAS suites are all `importorskip`-guarded, so without the
extra they **skipped rather than failed** — meaning they had never run in CI at all, which is the
same vacuous-gate shape this release keeps surfacing elsewhere. `gkdv`'s DAS arm is a second
consumer of that subsystem and its correctness turns on a direction-inference subtlety inside it.
The arm's own structural direction-pinning guard needs no extra and runs regardless.

### Fixed — the Databricks IDSSE loader fabricated its home team and starting direction

`scripts/_loader_databricks.py` fed **raw** bronze straight into the Sportec converters, took
`home_team_id` from the modal `team_id` in the frames (an explicit placeholder) and hard-coded
`home_team_start_left = True`, then dropped extra time to dodge the ADR-010 ET-without-flag raise.
It now routes through the ADR-031 T3 parse-port shapers (`shape_events_to_native` /
`shape_tracking_to_native`) and derives direction of play — including ET — from the authoritative
DFL `<KickOff>` rows, mirroring `scripts/_loader_pining.py::_build_idsse`.

The identifier domains across the two bronze tables are **asymmetric** and are now fed
accordingly: `bronze.idsse_tracking.team_id` carries the DFL-CLU id, `bronze.idsse_events.team`
carries the literal `"home"`/`"away"`. `convert_to_actions` therefore gets `"home"`, and
`actions.team_id` is remapped onto the CLU ids afterwards so the ADR-028 action↔frame join
resolves — without that remap the join matches nothing and away-team tracking geometry stays 180°
wrong. A missing or all-null native team id now raises with an actionable message instead of a bare
`KeyError` deep inside a converter. Calibration-harness path only; no library behaviour change, but
IDSSE calibration inputs change and prior IDSSE calibration runs are not comparable.

### Changed — `compute_ghost_gk` and `serve_ghost_gk_positions` share one core

The 79-line serving body is extracted to `_serve_positions_core`, so the new positions-only seam
and the feature-emitting aggregator cannot drift apart. No output golden existed anywhere for the
ghost path — the five modules calling `compute_ghost_gk` assert structure and behaviour, not
values, and `test_weights_bundle_golden.py` only import-checks `GhostGkModel` — so a pre-refactor
oracle was captured first (`scripts/make_ghost_gk_golden.py` →
`tests/tracking/data/ghost_gk_refactor_golden.npz`) and the equivalence gate compares against it.
Deliberately a **same-environment** oracle: serving is a sklearn-free numpy reconstruction
(ADR-016), and the npz is not to be repurposed as a cross-platform pin.

### Added — `id_compat.ids_isin`

The id-COLLECTION sibling of `ids_match`, for a caller-supplied id **set** resolved against an id
column. Canonicalises both sides so every integral spelling collapses. Deliberately **not**
`.astype(str).isin(...)`, which renders a float-backed id column as `"999.0"` and matches nothing —
a shape `canonical_id_series`'s own docstring already documented as wrong. Missing ids never match
on either side, replacing a raw `.isin()`'s dtype-*dependent* NA-wildcard behaviour (object +
`{None}` and float64 + `{nan}` matched null rows; `Int64` did not).

### Fixed — `goalkeeper_ids` was resolved with a raw `.isin()` in three places (ADR-019)

**`spadl.wyscout.convert_to_actions`.** The GK aerial-duel reclassification compared a
caller-supplied id set against `player_id` raw, so a caller holding roster ids as strings
(`{"999"}`) against an integer `player_id` column matched nothing: the aerial duel silently stayed a
duel and was **dropped as a `non_action`** instead of becoming `keeper_claim`. No error — the
caller's declaration was simply discarded. Measured: `{999}` → `type_id=15`, `{"999"}` → **0
actions**, identical to passing `None`. Matched-dtype callers are byte-identical, and every in-repo
caller is matched-dtype.

**`spadl.utils.add_gk_role` and its atomic mirror.** The same raw `.isin()` in rule (a)'s known-GK
match, so a dtype-mismatched set produced **no `distribution` rows at all**. This one was found
only because the registry's `add_gk_role` fixtures were de-vacuated first — with the fixture fixed,
both copies immediately failed the dtype-invariance assertion.

**Possible live-data signal, flagged for the lakehouse:**
`docs/research/xtgk_possession_value/LAKEHOUSE_HANDOFF.md:76` records that materialised `gk_role`
carries **only defensive roles — zero `distribution`** across 18,165 wyscout and 54,405 statsbomb
rows. That is precisely the symptom of `goalkeeper_ids` never resolving. If the lakehouse passes a
set whose dtype differs from its `player_id` column, this fix will start emitting `distribution`
rows where there were none — the fix working, not a regression, but a re-materialise decision.

### Testing — a differential non-vacuity guard for id-COLLECTION entries

`test_entity_id_collection_is_load_bearing`: emptying the collection must CHANGE the output, or the
entry's dtype-invariance assertion would hold for a broken `.isin()` too. Both `add_gk_role` entries
were vacuous exactly this way — the fixture's `same_player` rule already produced `distribution`, so
`{"1"}`, `{1}`, `None` **and `set()`** all returned an identical `gk_role`, while the fixture
docstring claimed the opposite. Fixtures corrected (the keeper row is now a different player) rather
than the assertion weakened.

`ids_isin` also carries five unit tests of its own, because **the registry structurally cannot
discriminate a canonicalised `.isin` from a naive stringify**: its float-scalar axis fires only on a
numeric scalar, and a collection entry declares a set. Degrading `ids_isin` to `.astype(str)` passed
all 103 registry tests — the gap is closed by the direct tests, not by the gate.

### Fixed — `GkdvParams.lambda_gk` is now actually forwarded

It is the only term through which the threat arm sees the keeper — which is what the
`pitch_control_method` GK-BLIND construction guard defends — but it was never read:
`delta_threat_suppression` passed no pitch-control params, so `compute_threat_pc` fell back to the
default. Because `GkdvParams.lambda_gk` and `SpearmanParams.lambda_gk` share the default `3.0`,
output is **byte-identical at defaults** (no retrain trigger). The defect was that a caller *raising*
`lambda_gk` silently got the default gain while `GkdvReport` echoed the raised value as though it
had been used — and `GkdvReport`'s own docstring says "registration without traceability is not
registration". Guarded on the CALLS, because an output comparison passes on the unforwarded code.

### Removed — `GkdvParams.seed`

Never read. The spec registered a seed **conditionally** — *"if `accessible_space` is stochastic"* —
and the field took that branch without evaluating the condition. `gkdv/` contains no randomness at
all, so wiring it would mean inventing some; the identity assertion the seed existed to protect is
already enforced behaviourally by `test_das_arm_identical_frames_give_exactly_zero`. `gkdv/` is new
in this release, so no shipped caller can exist.

### Changed — the id-scalar registry discovers modules that declare no `__all__`

Discovery walked `__all__` alone, and **35 walked modules declare none**, so they contributed
nothing: **13 public id-scalar callables sat in no bucket** — every provider `convert_to_actions`,
every native `convert_to_frames`, and both `tracking.direction` primitives — while the anti-rot
meta-assertion reported full coverage. That is the deleted AST lint's defining failure (a discovery
rule that silently stops looking) reproduced inside the gate built to replace it. 11 are now
directly enforced; a new fourth bucket `NOT_EXERCISABLE` holds the two for which no
matched/mismatched pair *exists* (metrica's fixed `'Home'`/`'Away'` literals; kloppy's writer-only
`game_id`), kept distinct from `NOT_INVARIANT` so the reason is not misreported.

`PitchControlSurface`'s `NOT_INVARIANT` justification is narrowed to state it covers the
**constructor only**, and names `.player_surface()` / `.player_share()` as a real, OPEN ADR-019 gap
that this exemption does not close.

### Fixed — DAS was silently all-NaN on pandas 3

`_prepare_frames` casts the columns accessible-space indexes two-dimensionally (`arr[:, None]`) to
numpy `object`, but **omitted the forwarded ball-carrier column** — the one the library receives as
`PASSERS` on the `respect_offside` path, which is the DAS default. On any pandas that infers a
`StringDtype` for it (pandas 3 / py≥3.11, i.e. **every CI leg except `ubuntu-3.10`**), the 2-D index
raised `IndexError`, which `_call_simulation` converted to `DasUnscoreableError` — so **every** DAS
call degraded to NaN with `das_source == "unscoreable_call"`, and nothing detected it.

The cast now names the column the caller actually supplied via the public `player_in_possession_col`
kwarg rather than the `ball_carrier_player_id` literal, so a renamed carrier is covered too;
`get_xc` forwards no carrier and is unchanged.

Measured on the calibration fixture: **py3.12 / pandas 3.0.3 goes 0 → 3 finite DAS rows of 10**;
**py3.10 / pandas 2.3.3 is byte-identical** (verified by a same-process full-vector A/B, not by
inspection). This restores `gkdv.delta_das` (the TF-19 PR-3 physics arm shipped in this release),
`das_xfns` / `add_das`, and the TF-24 calibration DAS features on modern pandas.

**Found by this release's own DAS work.** The old broad exception catch swallowed the `IndexError`
into a mute NaN column; the narrowed `DasUnscoreableError` plus `das_source` provenance is what made
a long-standing silent degradation legible — on its first contact with CI.

**No retrain trigger.** No bundled model weights consume DAS-derived features, verified against the
committed feature lists in every one of the five model families' `metadata.json`/`model.json` (xS,
xCross, Ghost-GK, `GkCompletionModel`, `GkRetentionModel`); grepping every trainer and extractor for
`das` / `accessible_space` / `get_das` / `get_xc` returns zero matches. `das_xfns` is in no default
xfn list and has no atomic mirror.

**pandas-3 consumers must re-materialize:** anyone who opted into `das_xfns` for VAEP; direct
`add_das` / `get_das` / `get_individual_das` / `das_at_action` callers; TF-24 calibration runs (their
Brier-CV scores and `das_degraded` diagnostics were computed against all-NaN DAS); and
`gkdv.delta_das` output. pandas-2 consumers are unaffected — byte-identical.

The new regression guard **pins the carrier dtype explicitly** rather than leaving it to inference,
so the defect now fails on *every* interpreter instead of only where pandas happens to infer
`StringDtype`. The B1 oracle gate (`test_compute_das_values_are_unchanged_by_the_public_routing`)
consequently runs for real on every leg instead of skipping.

### Testing — `delta_das` gains live end-to-end coverage

`test_das_arm_returns_a_LIVE_FINITE_delta_through_real_accessible_space` exercises the arm through
**real** accessible-space with no `_das_port` stubbing. Every other gkdv DAS assertion either stubs
the port or runs on frames carrying **no ball-carrier column** — so `_resolve_player_in_possession_col`
returns `None` and the offside path never runs. Measured: reintroducing the pandas-3 all-NaN defect
left all **163** pre-existing `tests/gkdv/` tests green.

The gap was silent rather than loud because **`team_das` sums `DAS.dropna()`, and an empty sum is
`0.0`** — an all-NaN collapse returns a finite, plausible zero from *both* legs, so a finiteness
assertion alone would have passed. The guard therefore asserts the underlying per-player DAS is
finite and strictly positive **and** that the delta is non-zero. Mutation-verified red on both
interpreters.

### Fixed — provider parse hardening

`providers/sportec/parse.py`: period resolution now raises when neither `period_id` nor `period`
is present instead of degrading silently, and `ball_state` normalization routes numeric values via
`Int64` to avoid the `"0.0"` stringification trap, warning on out-of-domain tokens.

## [4.52.0] — 2026-07-18

### Added — real-xT EPV wiring for the OBSO family (`silly_kicks/xthreat/_physical.py`, `tracking/features.py`, `tracking/_warnings.py`; PR-S119, ADR-041)

`add_obso` / `add_pausa` / `add_space_creation` (and their `*_xfns`) accept `xt=` and will use a
fitted `ExpectedThreat` as the EPV surface. Until now they always multiplied pitch control by a
synthetic placeholder ramp, and nothing said so.

- **`silly_kicks.xthreat.physical_grid`** — resample xT onto a CONSUMER-SUPPLIED grid in ascending-y
  physical orientation. `ExpectedThreat.xT` stores rows y-INVERTED (row 0 = top of pitch); `rate()`
  compensates, `interpolator()` does not. This is the one place that inversion is neutralized.
- **`silly_kicks.xthreat.values_at_points`** — NaN-tolerant per-point xT with exact
  `rate(use_interpolation=False)` semantics.
- **`silly_kicks.xthreat.require_fitted_xt`** — the single fitted-model guard; two divergent copies
  (`vaep/features/expected_threat.py`, `atomic/vaep/features.py`) now delegate to it.
- **New `obso_epv_source` column** (`"xt"` / `"synthetic"` / `"injected"`) and a
  **`SyntheticEPVWarning`** whenever the placeholder is served.
- **Three public warning categories** in `silly_kicks.tracking`: `SyntheticEPVWarning`,
  `IgnoredSurfaceInputsWarning`, `RunValueCoverageWarning`.
- `PitchControlCache.__len__` is now public; `add_pausa` accepts `pitch_control_cache=`.

### Fixed — three per-action orientation defects in the OBSO / influence families (ADR-041, amends ADR-028)

**These change values on every provider, with or without `xt=`.** The OBSO target/EPV reflection is
away-keyed, so `obso_*` / `pausa_*` home rows are byte-identical — but the y-inversion repair
(`interpolator()` -> `physical_grid` in `player_influence` / `gk_influence` / `cover_shadows`, and
`space_creation`'s `axis=(0, 1)` opponent mirror) is a y-MIRROR and moves **home rows too**.
Re-materializing only away rows would leave stale home values for those families.

- **`add_obso` / `add_pausa` / `add_space_creation` never handled orientation at all.** Frames are
  home-attacks-right; actions are per-acting-team LTR. The raw action-LTR target was sampled against
  the frame surface, and the EPV grid always increased toward the HOME team's attacked goal, so away
  actions were sampled at the reflected point AND valued toward their own goal. `home_team_id` was
  accepted and never read. ADR-028 had listed OBSO as "self-reconciling"; it was not.
- **`compute_player_influence` multiplied the raw, y-mirrored `interpolator()` output** instead of a
  physically-oriented grid. **`compute_gk_influence` and `compute_blocking_score` (cover shadows)
  carried byte-identical defective code** and are fixed in the same pass — found by an adversarial
  review of this PR, not by the original sweep.
- **The reflection was applied on ONE axis.** ADR-028's relation is a 180° point reflection
  (`x→105−x` AND `y→68−y`), so the grid transform is `[::-1, ::-1]`, not `[:, ::-1]`. An x-only
  mirror is exact only for a y-symmetric grid — which the synthetic ramp is and a fitted xT nearly
  is, which is why it survived the first round of (x-axis) tests.
- **Grid registration + index rounding**: the EPV sample grid now uses node registration matching the
  OBSO kernel's own indexing, and the kernel's `floor` cell lookup became `round` (±0.505 m at the
  pitch edges).
- **`space_creation`'s opponent mirror** `np.flip(..., axis=1)` → `axis=(0, 1)`.
- **`compute_pass_obso`'s teammate index map truncated** where its target-index sibling rounds —
  a bare `int()` biased every teammate lookup toward the low-index cell by up to a full cell.
  Both maps index the same node-registered grid, so `obso_optimal` shifts slightly.
- **`compute_blocking_score`'s `detailed=False` branch** (the production default) read the raw,
  y-inverted `interpolator()` with no reflection at all, so `max_single_defender_blocking_score`
  matched NEITHER orientation. Its frozen parity oracle read the same raw interpolator and so
  compared the bug to itself; the oracle is corrected too and now has discriminating power.

### Added — TF-35 off-ball run valuation (`silly_kicks/tracking/_run_values.py`; PR-S119, ADR-042)

Values off-ball runs by the threat of the space the runner comes to control, rather than only
counting them (TF-4).

- **`detect_off_ball_runs`** — geometry only; one row per qualifying `(action, runner)`, positions
  emitted in SPADL action-LTR, with a `peak_speed_source` provenance column
  (`"measured"` / `"displacement_rate"`).
- **`value_off_ball_runs`** — `role` (`"target"` / `"disruptive"`), `is_receiver`, `run_value`,
  `enabled_pass_credit`. `run_value` is the MAXIMUM of `pitch_control × threat` over the cells the
  runner controls.
- **`add_off_ball_run_values`** — five wide columns: `run_value_target`, `n_disruptive_runs`,
  `run_value_disruptive_sum`, `n_valued_disruptive_runs`, `run_value_enabled_pass`.
- **`off_ball_run_value_xfns`** — four numeric columns × 3 slots. **Opt-in, in NO default xfn list**:
  the domain gates on the action's own `result_id`, which is the HybridVAEP result-leakage class
  (ADR-039 F4). Enforced by an auto-discovering guard.
- **`RunValuationParams.resolved_region_floor()` fails loud** for any `pitch_control_method` without
  a recorded floor. The 0.1 default is a spec-time starting value, **not calibrated**.
- Atomic mirrors for both the aggregator and the factory.

**Read the aggregator docstring before averaging**: mean disruptive value must divide by
`n_valued_disruptive_runs`, never `n_disruptive_runs` — the sum skips unvalued runs, so the wrong
denominator turns a tracking-coverage gradient into an apparent tactical one.

### Changed — TF-4 `toward_goal` re-keyed onto the direction authority (ADR-042)

`n_off_ball_runners_toward_goal_pre_window` now derives orientation from the frames'
`team_attacking_direction` rather than home/away identity. **This is a behaviour change, not a
refactor**: the two disagree exactly where the acting team has no direction-carrying frame row in
that period (identity-keying always flips the away team; the direction authority conservatively does
not). Not a retrain trigger — `off_ball_context_xfns` is absent from `tracking_default_xfns`.

### Added — import-cycle CI gate (`tests/test_no_import_cycles.py`)

Subprocess-imports each public subpackage standalone. The `xthreat` wiring closed a real cycle twice
during development (`xthreat/_grid → spadl.config → spadl/__init__ → tracking → … → xthreat`), and a
cycle of that shape is invisible to the ordinary suite, which always imports in a friendly order.

### Notes for downstream consumers

- Re-materialize `fct_action_context`: away-team `obso_*` / `pausa_*` / `space_created_m2` /
  `space_denied_m2_opponent`, all `player_influence_*` / `gk_*` (GK influence) / `cover_shadow_*`,
  and TF-4's toward-goal count. Batch this with
  the already-queued 4.49–4.51 triggers.
- Five new candidate columns from TF-35, plus `obso_epv_source` and `peak_speed_source`.
- **`value_off_ball_runs` resolved its frame group with a raw tuple lookup** — dtype-sensitive on
  `game_id` (ADR-019). On the documented lakehouse shape (actions `int64`, frames native string)
  runs were DETECTED but every `run_value` came back NaN, with a coverage warning that
  misattributed the cause to a tracking-visibility gap. Now canonicalized on both sides.
- **Recorded latent gap (not fixed here):** `PitchControlSurface.player_surface` / `.player_share`
  compare player ids with a raw `==`, an ADR-019 gap that also makes `_player_influence`'s
  `except ValueError: → 0.0` silently zero players on dtype-mismatched frames. Tracked in TODO.

## [4.51.0] — 2026-07-17

### Changed — TF-19 PR-2: corrected default weights + fail-closed chirality `load()` enforcement (`silly_kicks/tracking/_chirality.py`, `_xshot_occurrence.py`, `_xcross_attempt.py`, `_ghost_gk.py`; PR-S118, ADR-040)

The bundled default xS / xCross / ghost-GK weights were **chirality-mis-served** — trained y-mirrored
before the ADR-031 kloppy-tracking y-fix but served y-correct — a live correctness bug for VAEP
consumers of `pre_shot_gk_full_default_xfns`. This replaces them with the DGX retrains (trained on
y-correct frames) and completes ADR-037 §9's deferred chirality enforcement:

- **Corrected default weights.** xS/xCross defaults are now the reproducible **public** arm
  (SkillCorner + IDSSE, GS-free); the ghost default is the Stage-B 179-match retrain (§4.3-confirmed
  better generalization). The wheel **shrinks** (ghost 12 MB → 7.2 MB).
- **`load()` fail-closed chirality enforcement** (all three models): re-runs the model's own
  `_chirality_block` on the canonical y-asymmetric probe frame and compares to the stored
  fingerprint. Raises on a **mismatch** and on a **missing** one (every pre-PR-2 artifact = the
  mis-served ones), with an explicit `legacy_override=True` escape hatch (warns). Cross-platform
  tolerance `atol=1e-3 / rtol=1e-2` catches a y-mirror while tolerating float noise. Plus a
  finiteness guard in `chirality_fingerprint`.
- **`base_score` compatibility guard** (`load_xgb_booster_base_score_safe`). xgboost 3.x serializes
  `base_score` as a bracketed string `"[X]"` that xgboost 2.x silently drops to the 0.5 default —
  a mis-served intercept (the enforcement above **caught** this). `load()` now normalizes the
  bracketed form; the bundled weights are re-saved to clean 2.x scalar format. The library supports
  `xgboost>=2.0` across the 2.x/3.x boundary.
- **HF-only `sc_extended` variant** (the Stage-B models trained on the 98 owner SkillCorner matches,
  which beat public on both arms). `from_variant("sc_extended")` routes to `from_hub`; xS's `from_hub`
  is now implemented (mirrors xCross) behind the new `[xshot]` extra. The Hub upload is an owner
  follow-up (`docs/research/tf19_pr2/hf_upload_instructions.md`); the weights themselves are not
  bundled in the wheel.
- Ghost-GK model-card prose fixed (the training filter is a purely geometric goal-relative box, not
  an "active defensive actions" condition); decision-table verdict recorded
  (`docs/research/tf19_pr2/decision_table.md`).
- **Weights-bump CI fixture updates** (the new weights legitimately move model outputs): the frozen
  ghost KDE scipy-oracle golden (`tests/tracking/fixtures/ghost_gk_kde_golden.npz`) is regenerated
  from the Stage-B model via `scripts/gen_ghost_gk_kde_golden.py` (mean_x 10.69→8.68 m; the
  cpu-numba/fft==scipy parity property it locks is unchanged). The xCross directional liveness
  fixture (`tests/datasets/tracking/xcross_directional/frozen_rows.parquet`) was a **degenerate**
  probe — it held `ball_speed=0` (the model's #1 feature) for every row, so both the old and new
  models scored it ≈0 and the AUC gate hinged on a razor-thin ordering that flipped between retrains
  (the new model is provably sound: held-out CV pr_auc 2.85× base rate, all acceptance gates green,
  AUC 1.0 on a realistic probe). New `scripts/make_xcross_directional_fixture.py` regenerates it with
  realistic ball speed + varied geometry (AUC 1.0 on both the 4.18.0 and PR-2 models).
- **Ghost mirror-invariance gate strengthened** (`tests/tracking/test_action_ltr_mirror_invariance.py`).
  The refit shifted the corrected model's inherent lateral asymmetry on the near-goal-**centre** probe
  (old 0.20 m → new 0.59 m), where the y-axis was a *vacuous* guard (a y-reprojection flip moved
  ghost_gk_y by only ~0.1 m). Rather than loosen the tolerance, the probe is now **off-centre** so a
  y-flip moves y by ~7 m (a real guard), with the durable orientation check on x (both mirrors at the
  attacked goal) and an explicit non-vacuity assertion. This strengthens, not weakens, the ADR-028
  construct-validity gate.

**Hyrum / retrain trigger:** the xS/xCross columns of `pre_shot_gk_full_default_xfns` change for
opted-in VAEP consumers (the corrected weights) — re-materialize. The public default arm is GS-free,
so it is unaffected by the 4.49/4.50 Gradient Sports dribble fixes. C4 count unchanged (weights +
a load-guard are not a new model node).

## [4.50.0] — 2026-07-17

### Fixed — Gradient Sports ball-carry results from the native `ballCarryOutcome` (`silly_kicks/spadl/gradientsports.py`; ADR-018 amendment, owner-directed in-PR fix)

The GS result dispatch had NO success condition for `OTB`+`BC` carries — every GS dribble fell
through to the `fail` default (surfaced by the TF-49 packing e2e: 0/12 dribbles in the packing
domain on match 10503; statsbomb dribbles are 100% success — GS was the outlier). The converter
now maps the native `ballCarryOutcome` — live WC2022 vocabulary {R, L}, present on 100% of BC
rows (probed 2026-07-17, 4 matches, 66 carries; the field was already in
`EXPECTED_INPUT_COLUMNS`, flattened but never consulted) — **R (retained) → `success`, L (lost)
→ `fail`**; unknown/absent tokens keep the `fail` default (this converter's exact-token
allowlist style, matching pass/cross `"C"` and shot `"G"`). Empirically cross-checked: L
carries are ~86% opponent-next on the converted stream. **GS-only retrain trigger — folds into
the SAME pending GS re-fit 4.49.0 queued** (GS-fitted xT/xtgk success-filtered move-sets now
include retained carries — previously they excluded ALL GS dribbles; VAEP result features shift
on GS; no additional retrain beyond the queued one). The owner-gated packing e2e now gates the
GS dribble in-domain share strictly interior — a future feed that drops or renames the field
fails loudly instead of mass-failing silently. Lakehouse: GS `spadl_actions.result_id` changes
on dribble rows — re-materialize GS-derived marts on adoption.

### Added — TF-49 packing: Impect-faithful bypass counts + goal-threat + secured reception + net variant (`silly_kicks/tracking/_packing.py`, `silly_kicks/tracking/_kernels.py`, `silly_kicks/tracking/features.py`, `silly_kicks/spadl/utils.py`, atomic mirror; PR-S117, ADR-039)

Coach-facing canon packing over tracking frames, built on the TF-45 bypass inequality
(`start_x < d_x <= end_x`) with the ~15-line defender-extraction/mirror block DELIBERATELY
duplicated from the frozen `_structural_pass.py` (consolidation trigger = a third consumer;
byte-equivalence pinned by a non-vacuity-meta-asserted golden identity gate:
`packing_made == structural_lbs` on completed pass/cross rows, with a failed-row discriminator
proving the completion gate is the only delta).

- **Geometry kernel** (`_kernels._packing_at_actions`, GEOMETRY-ONLY): per in-domain action
  (type ∈ `params.action_types` AND result success) at its linked frame — `packing_made`
  (canon Impect count, outfield-only by default, `include_gk` opt-in), `packing_net`
  (`football-packing` direction multipliers +1 / `side_multiplier` / `back_multiplier` over the
  traversed x-interval; θ = atan2(|Δy|, Δx) with 45°/135° bands), `packing_goal_threat`
  (forward count restricted to `select_back_line_players` back-`back_line_n`; MSC "goal threat
  packing"), + internal `line_x` (max bypassed-defender x; feeds secured, NOT an output column).
  Degenerate `start == end` → NaN for DRIBBLES only (placeholder-indistinguishable: pre-4.49.0
  GS corpora, post-fix period-last carries); pass-class start==end → honest geometric 0.
- **`spadl.utils.resolve_next_touch_receiver`** (NEW public, packing-agnostic, event-only):
  next same-team touch's player_id per action, skipping `non_action` AND `foul` rows (neither
  is a touch: GS emits non-touch rows; the fouler must never resolve as receiver, and an
  advantage-played opponent foul must not block resolution — execution-review D1);
  fully POSITIONAL (ADR-019/PR-S110 lesson; safe for non-RangeIndex/duplicate-index callers).
  Dtype contract (F5): Int64/object pass through; plain int64 AND float64 NaN-coded-int sources
  pre-convert to Int64; the result NEVER float64-upcasts. The private
  `_resolve_next_touch_positions` is a stable internal seam consumed by `secured_reception`'s
  reception-anchored window.
- **`tracking.secured_reception`** (NEW public): nullable-boolean "ball stays past the line" on
  the `retains()` skeleton (possession-aware, `add_possessions` self-heal) + the REQUIRED
  foul-skip (heuristic possessions emit a boundary AT the foul row — verified; a bare boundary
  rule would flip loss at every foul won) + non_action/NaN-team skips (ADR-027). Window is
  anchored at the RECEPTION row; a reception that is itself a same-team shot decides True
  (the literal pass→shot→keeper_save shape); same-team shot → True; opponent possession
  boundary → False; behind-`line_x` same-team action inside the window → False; empty window
  extends the shot/boundary tests to the first non-skipped event (`line_x` does NOT extend);
  truncated window → <NA>. Both keystone protections mutation-probed.
- **`tracking.add_packing`** (@nan_safe_enrichment, C4 aggregator count 28 → 29): five columns —
  `packing_made`/`packing_goal_threat` (Int64), `packing_net` (float64),
  `packing_receiver_player_id` (source-dtype passthrough; <NA> for dribbles/off-domain),
  `packing_secured` (boolean; <NA> unless receiver resolved AND made ≥ 1).
  `require_secured=True` gates the numeric columns for receiver-bearing types only (F3:
  dribbles keep raw counts; made==0 rows keep their 0). Idempotent provenance; `links` accepted.
- **`tracking.packing_xfns`**: ONE FrameAwareTransformer, 3 numeric columns × 3 slots, 3×-not-9×
  perf-spied. REJECTS `require_secured=True` (shifted gamestate slots have no valid next-row
  relationships). **Result-leakage warning (F4, docstring + ADR-039):** every packing column
  gates on the action's OWN result_id — MUST NOT enter HybridVAEP-class consumers without a0
  exclusion; a result-free-a0 variant is a recorded fork, not built. In NO default xfn list —
  enforced by an executable guard (`tests/tracking/test_packing_xfns_leakage_guard.py`,
  auto-discovering + mutation-verified) mirroring shot_goalmouth/xt_xfns; **no retrain trigger**.
- **Atomic mirror** (`atomic.tracking.features`): numeric columns only (receival atoms carry
  receiver identity); `end = x+dx` synthesis + a type-aware synthesized `result_id`
  (SK-xT-2 precedent: dribble intrinsic; pass-class success iff next atom is `receival` OR a
  same-team keeper reception — atomic never inserts receival before keeper collections;
  name-mapped types with collapsed-atom bridging for `_simplify`'s `corner`/`freekick` atoms,
  atomic-only atoms → std `non_action`); output assembled on a copy of the CALLER's frame (the
  adapter's rewritten type_id / synthetic result_id never leak); rejects `require_secured=True`.
- **Adversarial-review hardening (12-agent refute-verified pass over the finished diff; all
  six findings live-reproduced, fixed in-PR — ADR-039 §Execution-review):** receiver foul-skip
  (D1); atomic collapsed-atom domain bridging (D2); atomic output purity (D3);
  `secured_reception` scans in `action_id` order so time-tied positionally-swapped rows resolve
  identically (D4); atomic same-team keeper receptions synthesize success (D5); an unattested
  (NA) caller-supplied `possession_id` never decides the secured boundary (D6, ADR-027 class).
  `retains()` shared the D1/D6 patterns only LATENTLY (live solely on its unused
  `add_possessions` self-heal path): a read-only probe over the live ρ training cohorts measured
  ZERO label flips (gold-mart possession ids stay continuous through foul rows; no NA
  teams/possessions), so **the same hardening was applied to `retains()` in-PR WITHOUT a
  retrain** — a post-fix gate re-verified the shipped function == the probed variant on all
  223,718 cohort rows with 0/3451 (GS) + 0/5483 (SkillCorner) training-label changes; bundled ρ
  weights + recorded metrics untouched. `retains()` additionally gained the canonical
  `(time_seconds, action_id)` scan order (owner-decided after a dedicated order probe: 9,649
  GS time-tie pairs exposed positional-order sensitivity; a bare-`action_id` sort was RULED OUT
  — the GS mart's action_id order disagrees with time_seconds, guard-rejected) — exactly the ρ
  loader's own sort, gate-verified byte-identical on both full live cohorts; labels are now
  input-row-order-insensitive. A PR-S117 delta adversarial review then caught that the NaN-team
  hardening was decider-side only — a NaN-team ANCHOR still got a decisive label (ADR-027
  violated anchor-side); fixed (NaN-team anchor → NaN), also a no-op on the ρ path (both mart
  cohorts carry zero NaN-team rows). ADR-036 amendment (2026-07-17) + ADR-039 relay item 1.
- **Gates:** golden identity, ADR-028 mirror-invariance + asymmetric absolute-count pin,
  liveness (pre-checked on the multi-domain fixture), purity ×2 variants + atomic, id-dtype,
  dup-action-id, NaN-safety, Examples, perf spy, a `packing_xfns`-out-of-default-lists leakage
  guard (auto-discovering + mutation-verified, mirroring shot_goalmouth/xt_xfns), WC2018
  committed-fixture smoke on ALL CI legs (receiver rate ∈ [0.95, 1.0], dtype, secured tri-state,
  synthetic non_action injection), owner-gated GS WC2022 e2e (receiver 0.9976 ± 0.02; degenerate
  dribbles < 10%; GS dribble in-domain share strictly interior; secured rate strictly interior;
  per-action `packing_made` mean gated to [0.5, 3.0] — validated across 4 real WC2022 matches;
  MSC practitioner anchors REPORTED).
- **Recorded, out of scope (ADR-039):** GS `OTB`+`BC` carries all fall through the converter's
  result dispatch to `fail` → GS dribbles are structurally off-domain for packing's completion
  gate (statsbomb dribbles are 100% success — GS is the outlier); fixing it is a GS
  result-semantics change → its own probe/validation + GS-only retrain trigger. Also relayed:
  `retains()` on heuristic possession ids plausibly flips loss at opponent-foul rows (ρ-label
  change → ρ retrain if fixed).

## [4.49.0] — 2026-07-16

### Fixed — Gradient Sports dribbles derive real end coordinates (`silly_kicks/spadl/base.py`, `silly_kicks/spadl/gradientsports.py`)

Every GS dribble shipped with a placeholder `end == start` (verified 850/850 zero-displacement,
0 m, on the live corpus): the converter maps `OTB`+`BC` ball-carries to SPADL `dribble`,
initializes `end = start` for every event, derives real ends only for the shared
`_DERIVE_END_TYPE_IDS` (which excludes `dribble`), and is the only event converter that never
calls `_add_dribbles`. Fix is **GS-local**: `_derive_end_coordinates` gains a keyword-only
`extra_type_ids: frozenset[int] = frozenset()` and ONLY `gradientsports.py` passes
`{dribble}` — the module-level set is untouched because its `placeholder_end` guard cannot
distinguish statsbomb's ~11% genuine stationary carries from placeholders (a global addition
would rewrite recorded data). Default-path byte-identity is regression-locked
(`TestExtraTypeIds::test_default_leaves_dribble_placeholder`); period-last carries honestly
keep the placeholder. Owner-gated e2e (`test_dribble_ends_derived_on_real_wc2022`) asserts
>90% of real WC2022 dribbles now carry a derived end. **GS-only retrain trigger:** xT/xtgk
move-sets include dribbles (GS previously fed zero-displacement transitions into GS-fitted
transition matrices) and VAEP features consume dribble ends — GS-fitted artifacts should be
re-fit on next touch; zero delta for the other seven providers. Lakehouse: re-materialize
GS-derived marts on adoption. ADR-018 amendment; PR-S116. Precursor to TF-49 packing
(PR-S117), whose dribble-packing channel needs real GS carry geometry.

## [4.48.1] — 2026-07-15

### Fixed — native SkillCorner/Metrica builders emit a valid `ball_state` (`silly_kicks/tracking/skillcorner.py`, `silly_kicks/tracking/metrica.py`)

The native `convert_to_frames` builders (ADR-034) set `ball_state = None` for every frame. `None`
is not a valid `ball_state` (the schema value-set is `{"alive", "dead"}`), and it makes the strict
`ball_state == "alive"` domain filter in xShotOccurrence (`_xshot_occurrence.py:644`) and
xCrossAttempt drop **every** frame — so once the pining loader rerouted SkillCorner onto the native
builder (4.48.0/PR-S115), those trainers silently extracted **0 rows** from all SkillCorner matches
(measured: match 1886347 gives 8,071 xS rows once fixed, vs 0 before). The kloppy gateway set real
`ball_state` values, so this only regressed on the native path. The native feed carries no reliable
dead-ball signal, so both builders now default to `"alive"` (in-play): a valid schema value that
makes the alive-filter a no-op for these providers (equivalent to "use all frames"). Surgical —
`== "dead"` / not-dead consumers are unchanged (`None` and `"alive"` are both non-dead). Regression
guards added in `test_skillcorner_builder.py` / `test_metrica_builder.py`. **VAEP/GKDV retrain
trigger** for xS/xCross corpora that include SkillCorner or Metrica (they now contribute rows).

## [4.48.0] — 2026-07-14

### Added / Changed — SkillCorner corpus expansion + visibility surfacing (`silly_kicks/spadl/skillcorner.py`, `silly_kicks/tracking/skillcorner.py`, `scripts/`, PR-S115, ADR-038)

The pining SkillCorner listing grew from 10 to 108 matches. The 98 new ones are **owner-tier**
(`visibility: "private"`, restricted, all Real Madrid LaLiga+UCL), so the **public arm stays 17
matches** (10 SkillCorner + 7 IDSSE) and the prior 4.9.0 / 4.18.0 paired verdicts are unaffected;
the 98 can only expand the owner/full arm 81 → 179. This release makes them reachable and *safely
classified*, surfaces the `is_detected` flag the pipeline had been discarding, fixes two coordinate
defects, and **registers** the expanded-corpus retrain protocol. **Code and tests only — no weights.**
Spec: `docs/superpowers/specs/2026-07-14-skillcorner-corpus-and-visibility-design.md`; evidence:
`docs/research/skillcorner_corpus/`; decision: ADR-038.

- **Native SkillCorner route (ADR-038 §5).** The pining path builds SkillCorner frames through
  `tracking.skillcorner.convert_to_frames` (not the kloppy gateway), surfacing **`visibility`**
  (from the feed's `is_detected`, which the kloppy gateway hard-codes to `None`) and recovering
  **`ball_z`**. One SkillCorner truth instead of two.
- **Clamp split (ADR-038 §3).** `spadl/skillcorner.py::_transform_coords` scales THEN clamps —
  harmless for events (an action is on-pitch by construction), **destructive for tracking**
  (measured: 11.31% of ball rows snapped, up to 9.00 m; a ball nine metres behind the goal becomes a
  ball on the goal line, erasing goal-vs-save). The affine part is extracted to `_scale_to_spadl`
  (no clamp); tracking calls it **directly, never `_transform_coords`**.
- **Pitch-dimension scaling (ADR-038 §4).** The native builder scaled by a fixed 105×68 offset, so
  on a non-standard pitch the goal line landed up to **2.0 m** off (4 of the 10 public matches are
  104/106 m). It now scales via the events converter's own affine transform (single-sourced), keyed
  on SkillCorner's declared `pitch_length`/`pitch_width`; **missing dimensions RAISE** (fail-closed;
  `assume_standard_pitch=True` is the explicit opt-in).
- **Visibility-keyed corpus taxonomy (`scripts/_corpus.py`, ADR-038 §2 — a compliance control).**
  `_PUBLIC_PROVIDERS` is **deleted** (six sites; one set the shipped label, so a restricted
  `sc_extended`-shaped run had shipped labelled `"public"`). Public-vs-owner is now keyed on the
  manifest's `visibility`, fail-closed (unknown/missing ⇒ restricted); the artifact label derives
  from the ship-mask composition; a red-first CI guard forbids a restricted corpus from ever shipping
  a `"public"` label.
- **Registered-protocol machinery.** `scripts/_paired.py` — the fixed-sequence three-candidate
  (`public`/`sc_extended`/`full`) paired test with tuning **nested inside the outer CV** (so `public`
  cannot tune on the 17 matches that are its own evaluation universe). `scripts/_ghost_domain.py` —
  ghost-GK detected-keeper targets, keeper-grouped CV, and a paired sign-consistency admission (the
  interpolator-tell refusal was retired as dead code → a reported diagnostic). `scripts/_cache.py` —
  a feature-cache schema guard so a stale cache is a MISS.
- **S1 recalibration + per-match rate-gate (ADR-038 §6).** `_TOL_BALL` 30 → 15 m (real max ball
  excursion 9.00 m); the deferred rate-gate is implemented (`player_frac(>3 m) > 0.005` or
  `ball_frac(>10 m) > 0.0005` → the match is excluded). Its **pinned limitation**: it cannot detect a
  pitch-dimension error (0.00095 vs a clean-band worst of 0.00086), and neither can action↔frame
  co-location — the only instruments for pitch dimensions are provenance and asking SkillCorner.

**Detection finding.** Goalkeepers are detected in only **19.6% of frames** (~80% interpolated) —
the `is_detected` flag was in the feed all along; the kloppy gateway threw it away. This vindicates
the GS-only GKDV measurement rule and ADR-024/PR-S104's SkillCorner keeper-origin distrust.

**Hyrum events (both flagged):** (1) the **lakehouse re-materializes SkillCorner frames** — geometry
moves up to **2.0 m** on non-standard pitches (a correctness fix; the previous geometry was wrong);
(2) the **research-corpus SkillCorner frames change** (native route) → the owner runs the **Stage A
re-baseline** before any expansion is judged. **No weights ship in this PR** — they land after the
owner Stage A / Stage B runs (PR-B). C4 count stays 28.

## [4.47.0] — 2026-07-12

### Added — TF-19 GKDV attempt-arm re-gate CODE (`silly_kicks/tracking/`, `silly_kicks/causal/`, PR-S114, ADR-037)

PR-1 of the TF-19 GKDV cycle: the code that makes the correctness retrain measurable, the
new xS substitution probe, the public causal port, and the chirality guard's emission
half. The shipped xS/xCross weights are **chirality-mis-served** — trained in a y-mirrored
convention and served on y-correct frames (since ADR-031), so xS reads 12/27 features and
xCross 3/16 sign-inconsistently on every y-correct provider for consumers opted into
`pre_shot_gk_full_default_xfns` — a correctness bug `load()` was structurally blind to. The
retrain lands in PR-2; this release is **code only**. Spec:
`docs/superpowers/specs/2026-07-12-tf19-gkdv-regate-and-v1-design.md`; decision: ADR-037.

- **`silly_kicks.tracking._model_eval`** (PRIVATE to `tracking/`) — the model-agnostic
  GK-substitution probe core, a pure evaluator over pre-substituted ghost TARGETS-as-data
  so `tracking/` never imports the future `gkdv/`. Carries the **registered xS probe
  rule** (`XS_PROBE_RATIO=2.0`, `XS_PROBE_DOSE_M=2.0`, band `n≥100`, trusted-stratum
  `n≥50`, 20 paired-vector placebo replicates, cluster-exact game-level dose-response —
  all locked before any owner run), `evaluate_xs_probe` (dose-banded verdict with the
  ratio prong strengthened to `≥ 2× max(nearest_def, placebo_p95)`; zero-inflation a
  reported diagnostic, not a gate), the `PROBE_WRAPPERS` registry, and `regate_verdict`
  (the §3.5 decision table as a pure, test-parametrized function).
- **Public `silly_kicks.causal`** — promoted from the private `_causal/` port (ADR-015's
  anticipated "one move"). `matching.py` estimators unchanged (fit/match/ATT/ATNT/AI-SE
  byte-identical); `placebo_shift` gains the cluster-aware mode (`cluster_ids` +
  `_cluster_reassign` whole-cluster reassignment-with-recycling, `permutation_unit`
  reported); `opportunities.py` gains the full
  `OpportunityConfig` builder surface incl. a **result-conditioned, anchor-inclusive**
  outcome axis, making the ADR-037 §3.3 shot arm expressible purely as builder arguments
  (`shot_arm_config`). Registered in the `test_public_api_examples.py` gate from day one.
- **`silly_kicks.tracking._chirality`** — a behavioral chirality fingerprint (the model's
  own outputs on a fixed y-asymmetric frame); all three `save()` paths (xS, xCross,
  ghost-GK) **emit** it into `metadata.json`. `load()` fail-closed enforcement + a legacy
  override land in PR-2.
- Probe-sample provenance (provider + match ids) recorded by the xCross train script,
  closing the 4.18.0 provenance gap with an assertion.

### Changed

- **`silly_kicks.tracking._xcross_eval`** keeps its public names as a **byte-equivalent**
  home for the frozen xCross wrapper (internals re-pointed to the `_model_eval` core,
  golden-pinned); the re-run output gains report-only zero-fraction + dose diagnostics so
  the verdict table does not compare a diagnostics-rich xS verdict against a
  diagnostics-blind xCross one.
- **`extract_xshot_features`** ADR-019 canonical-id hardening (raw team-id `==`/`!=`
  compares routed through the canonical-id contract) — **VAEP-invariant for matched
  dtypes**; the ghost-GK extractor's equivalent raw compares are a recorded out-of-scope
  latent gap (`_ghost_gk.py:488-490`).

**No retrain trigger from this PR alone** — it is code only; the correctness retrain of the
xS/xCross/ghost-GK weights (a Hyrum/retrain trigger for `pre_shot_gk_full_default_xfns`
consumers + ghost-GK's `ghost_gk_*`) lands in PR-2. C4 count stays 28.

## [4.46.0] — 2026-07-12

### Fixed — xT-GK v2 scored ~24% of its domain at a fabricated grid zone (`silly_kicks/xtgk/`, PR-S113, ADR-036 amendment)

`flat_zones` maps a NaN coordinate to `(0.0, 0.0)` → **flat zone 176** (the own-corner cell). That is
a **fit-path contract** — safe only because every fitting seam drops NaN-coord rows before a surface
is solved. It was **false at the one scoring seam**, `_metric.py`, which dropped nothing: a NaN-origin
goal-kick was scored as a **real number at a location it never had**.

Compounding it, `load_xtgk_cohort` read the **raw** `bronze.spadl_actions.start_x`, while the
**resolved** keeper origins already existed in `fct_action_context.xt_gk_origin_x/_y` (PR-S101,
4.36.0) — in the very table the loader already `LEFT JOIN`s. It never `SELECT`ed them. The v1
comparator in the 4.45.0 head-to-head **did** use resolved origins, so that comparison was never
apples-to-apples.

**Measured on live gold:** **946 of 3874** Gradient Sports GK-distribution actions (**24.4%**,
including **60.2% of its goal-kicks**) were scored at zone 176 — 530 had a resolved origin sitting
unused in gold, 416 were never resolvable and are now honest NaN. Separately, **971 SkillCorner
goal-kicks** carried a *present-and-wrong* origin: the broadcast **ball** detection, not the keeper
(ADR-024 / PR-S104). A `fillna` **coalesce** would have fixed Gradient Sports and silently missed
SkillCorner — the rule is **OVERRIDE, not coalesce**.

### Added

- **`silly_kicks.xtgk.apply_resolved_gk_geometry`** — pure, no I/O, returns a NEW frame. Overrides
  GK-distribution coordinates from gold's resolved keeper geometry and stamps a 7-value
  **`gk_geometry_source`** provenance column (`off_domain` / `native` / `resolved_origin` /
  `resolved_dest` / `resolved_both` / `unresolved` / `unattested`). `unresolved` **wins** whenever any
  coordinate is still non-finite. Resolved columns absent → warn + no-op + `unattested` (never
  `native`, which would suppress the metric's warn-once while origins were still raw). Missing domain
  column → `ValueError`.
- **`silly_kicks.xtgk.finite_coord_mask`** — the blessed pre-filter for any caller that **scores** on
  the grid, adjacent to `flat_zones` whose docstring now states the fit-path-only contract.
- `--retention-weights` on `validate_xtgk_v2.py`, `xtgk_v2_keeper_discrimination.py` and
  `xtgk_v2_kappa_sweep.py`; every report now names the ρ that produced it.

### Changed

- **`compute_xt_gk_v2`**: non-finite-coordinate rows emit **NaN** across all five outputs and never
  enter the scoring loop — no zone is ever fabricated. ρ is **no longer scored** on those rows, which
  closes `GkRetentionModel.predict_proba`'s silent mean-imputation *without* changing its semantics.
  Adds a **coordinate-coherence check** (recomputes the coordinate-derived ρ features from `actions`
  and raises on divergence — catching resolved-actions/raw-features, its mirror, and mart-vintage
  divergence with one rule) and a **warn-once attestation**.
- **Both Databricks loaders** now `SELECT` `xt_gk_origin_x/_y` + `xt_gk_dest_x/_y` and apply the
  helper, so all four consumers inherit the fix and ρ features are necessarily built from the
  resolved frame.
- **ρ retrained** on the corrected cohort; both variants **PASS** the calibration gate. `default`
  (gradientsports): 2923 → **3451** rows (the goal-kicks whose NaN origins had excluded them from
  training entirely), AUC 0.781 → **0.798**. `skillcorner`: **5477 rows, identical** — nothing added
  or dropped, only the goal-kick *geometry* corrected — AUC 0.650 → **0.662**. `_PROVIDER_VARIANT`
  unchanged.

### Construct-validity re-run — honest and mixed

Two legs (pre-fix ρ / retrained ρ); metrics, baselines, κ=1 headline and a-priori parameters all
frozen. Full tables: `docs/research/xtgk_v2_construct_validity/README.md`.

| lens | provider | 4.45.0 (raw) | 4.46.0 |
|---|---|---|---|
| outcome-AUC lift | gradientsports | −0.1387 | **−0.1474** |
| outcome-AUC lift | skillcorner | −0.0720 | **−0.0268** |
| keeper ICC (v2) | gradientsports | **−0.0020** | **+0.0256** (v1: 0.0193) |
| keeper ICC (v2) | skillcorner | 0.0109 | **0.0147** (v1: 0.0176) |

**The "keeper-flat" leg of the 4.45.0 verdict does not survive.** Gradient Sports' v2 ICC went from
−0.0020 (worse than nothing) to **+0.0256**, now **exceeding v1** for the first time — the direction
predicted before the run, since a fabricated origin is **keeper-independent** and so compresses
between-keeper variance toward zero. **The outcome-AUC leg stands**: v2 still loses to simple
baselines. xT-GK v2 remains **not construct-validated by outcome-AUC**, but the interpretation-fork
decision can now be taken on trustworthy numbers with one of its two supporting findings withdrawn.

### Not re-run

The **deep-zone gate**. Every fit seam drops NaN coords, so the fitted `V` surface and its support are
clean and the GO-leaning verdict stands — asserted by a regression test with a **non-vacuity
meta-assertion**, not by prose (`tests/xtgk/test_deep_zone_gate_nan_invariance.py`).

### Consumer impact

**xT-GK v2 re-materialize trigger** (`compute_xt_gk_v2` output and ρ weights both change). **NOT** a
forced VAEP retrain — v2 is opt-in and in no default xfn list. The lakehouse must call
`apply_resolved_gk_geometry` before `compute_xt_gk_v2`, **and must keep `is_gk_distribution` (or the
stamp) on the frame it passes** — the warn-once fires only when a domain column with true rows is
present, so a pre-filtered slice with the flag dropped would score raw origins in silence. On pandas
3.0 `gk_geometry_source` materialises as `str` dtype, not `object`.

## [4.45.0] — 2026-07-11

### Changed — xT-GK v2 faithful V_opp + full construct-validity + keeper-discrimination validation (`silly_kicks/xtgk/` + `scripts/` + `docs/research/`, PR-S112, ADR-036 amendment)

The release that closes the xT-GK v2 validation loop. It ships the **faithful** turnover-cost adapter
(Jeff §2.3: an observed-post-turnover estimate, not the mirror geometric proxy) and the full
out-of-sample validation instrumentation. Honest-reporting guardrail (§3): the a-priori params were
fixed before fitting and the numbers are reported as they landed — **NOT retuned to force a pass.**

- **Faithful `EmpiricalTurnoverValue` (`silly_kicks/xtgk/_turnover.py`, library).** Rewrote the model-free
  turnover-value adapter into the faithful `V_opp`:
  - **Possession-bound by default** (`window_seconds=None`): the opponent's first-shot xG is scanned from the
    turnover to the *match* boundary (no fixed-time cap), so a real deep-turnover threat is not truncated by a
    window. A finite `window_seconds` is retained as a reported sensitivity.
  - **Support-gated hierarchical bin-widening**: every `(zone, pressure)` cell resolves to the finest estimate
    with `>= min_support` support — native cell → coarse `coarsen×coarsen` block → global-per-pressure — so a
    deep cell with 1–2 native turnovers is not a noise estimate. `min_support` defaults to **30** (= the
    pre-registered deep-zone gate `n_min`). Per-cell `resolution_level(p)` (0 native / 1 block / 2 global /
    −1 unresolved) + a module-level `surface_divergence(a, b, p)` for auditing two adapters.
  - **Fail-loud `game_id` guard**: possession-bound scanning requires a match boundary, so `fit` raises if
    `game_id` is missing/NULL (the possession-bound scope can't be computed without it).
  - The metric assembler `_metric.py` is **unchanged** — `compute_xt_gk_v2` still injects `turnover_cost` via
    the port, so this is a better recommended injection, not a forced default change.
- **Validation harness (`scripts/validate_xtgk_v2.py`, owner-run).** `construct_validity_scores` now
  train-fits the faithful possession-bound `EmpiricalTurnoverValue` on the possession-parity **train** split
  and injects it (V out-of-sample, ρ in-sample); GK-distribution-domain restricted (`is_gk_distribution`);
  reports the component decomposition (position / pev / retention_loss / dzv `|mean|` share) and the R1
  deep-cell disentanglement (possession-bound vs mirror vs 10s, native-n, resolution level) that separates a
  genuine mirror over-statement from a window-shrinkage artifact. A `kappa` passthrough enables the W6 sweep.
- **Keeper-discrimination instrument (`scripts/xtgk_v2_keeper_discrimination.py`, owner-run, NEW).** The real
  SP5 question (Jeff's Bravo/Navas reranking mode): does v2 separate keepers where v1 was flat? Measured by a
  one-way random-effects **ICC on action-level values grouped by the resolved `player_key`** (R2: NOT the
  degenerate CV-on-collapsed-means); CV reported secondary/unstable-near-zero-mean.
- **Secondary faithfulness audit (`scripts/xtgk_v2_kappa_sweep.py`, owner-run, NEW).** κ sweep (reported for
  Jeff, κ=1 the a-priori headline — never tuned) + the V-reward interpretation deferral (we use
  `E[first-shot xG]` vs Jeff §2.1's remainder-of-possession — flagged, not silently changed) + the PEV-dormant
  note (`p′=p`; receiver-pressure `q` deferred).
- **Loader (`scripts/_loader_databricks.py`).** The xtgk cohort SQL now selects `c.is_gk_distribution`,
  `c.xt_gk` (v1 baseline), and `c.player_key` (the resolved keeper — `player_id` is NULL for goal-kicks by
  SPADL design; convention added to `CLAUDE.md`).

### Findings (owner-run, real Databricks gold; `docs/research/xtgk_v2_construct_validity/`)

- **The faithful V_opp is a genuine correction.** It un-swamped the metric: the deep-turnover `dzv` share
  fell from ~87–89% (mirror) to **29%**, and `ρ·ΔV` (position) rose from ~8% to **36–42%**. The mirror
  over-stated deep opponent threat ~10–50× at real support (e.g. GS zone 96: mirror 0.256 vs
  possession-bound 0.005); the R1 disentanglement confirms this is NOT a window artifact (10s → ~0.0000).
- **But v2 still does not beat the baselines.** Outcome-AUC lift over `max(raw_completion, destination_xt,
  v1_stored)`: **GS −0.139, SC −0.072** (v2 AUC 0.484 / 0.513). On v1-covered rows v2 does beat v1
  head-to-head on GS (0.502 vs 0.381) but not on SC (0.513 vs 0.584).
- **And v2 does not discriminate keepers.** Action-level ICC (grouped by `player_key`): **v2 −0.002 (GS) /
  0.011 (SC)** vs **v1 0.019 / 0.018** — both near-zero; v2 is still keeper-flat. (The R2 ICC vindicated
  itself: CV had suggested v2 24% ≫ v1 6%, but that was a near-zero-mean artifact.)
- **Verdict (honest, §3): xT-GK v2 is not construct-validated by the outcome-AUC or keeper-discrimination
  lenses, even with the faithful V_opp.** Reported as-is for the Jeff conversation; open interpretation forks
  (first-shot vs remainder-of-possession V reward; dormant PEV) are flagged, not silently patched. The
  faithful V_opp adapter still ships because it is the correct, un-swamped turnover cost regardless of the
  downstream verdict.

- **Hyrum:** the faithful `EmpiricalTurnoverValue` changes `compute_xt_gk_v2` output vs the mirror proxy; any
  consumer that adopts the faithful injection (recommended) re-materializes `xt_gk_v2_*`. Opt-in (not in any
  default xfn list) → not a forced VAEP retrain. C4 count unchanged (28).

## [4.44.0] — 2026-07-11

### Changed — ρ retrain on the broadened `is_gk_distribution` domain + loader collapse + GK-resolver dtype fix (`silly_kicks/xtgk/` + `tracking/`, PR-S111, ADR-036 amendment)

- **ρ retention retrain (Part A).** With lakehouse F1 live (`fct_action_context.is_gk_distribution`), the ρ
  domain broadened from goal-kicks-only to the full GK-distribution set (goal-kicks + acting-GK open-play
  passes). Re-bundled on the broadened domain, calibration-gated (`ece≤0.10 AND |slope−1|≤0.25`):
  - `default` (gradientsports): **AUC 0.781 / ECE 0.031 / slope 0.998**, n=2923 (64 matches) — improved from
    the goal-kicks-only 0.776 / 0.090 / 1.005 (n=396).
  - **`skillcorner` variant NOW SHIPS** (was base-rate/fallback): the broadened domain makes it viable —
    **AUC 0.650 / ECE 0.020 / slope 0.923**, n=5477 (108 matches), GATE=PASS (vs the old near-chance 0.54 on
    1189 goal-kicks). `_PROVIDER_VARIANT = {"skillcorner": "skillcorner"}`; other providers still fall back
    to `default`.
  - **Serve-output change** for `compute_xt_gk_v2` (new `default` ρ) → xT-GK v2 **retrain trigger** (opt-in;
    not in any default xfn list, so NOT a forced VAEP retrain). Lakehouse re-materializes xt_gk_v2 on re-pin.
- **F1 CI calibration guard** (`tests/xtgk/test_retention_bundle_calibration.py`): every bundled variant's
  recorded `metrics.json` must clear the canonical `_ECE_MAX`/`_SLOPE_TOL` (imported, not read from the file)
  + recorded thresholds must match them (a hand-loosened `metrics.json` can't self-certify). Turns
  "bundle-only-if-passes" into an enforced invariant.
- **Loader collapse (Part B):** the transitional self-adapting `is_gk_distribution` probe is retired —
  `is_gk_distribution` is a HARD dependency (unconditional `SELECT c.is_gk_distribution`); NULLs coalesced to
  False (warning-free). MODEL_CARD pressure doc-bug fixed (`andrienko_oval` → the actually-used `bekkers_pi`).
- **GK-resolver dtype fix (Part C):** the shared `_gk_from_frames_linked` team predicate is now dtype-safe —
  `ids_equal` (acting) / `ids_differ` (defending). **Investigation reframed the target:** `acting_gk_from_frames`
  is fallback-protected (not fragile); the real defect was `defending_gk_from_frames` returning the acting
  team's OWN keeper (not the opponent) on a cross-dtype team mismatch. Byte-identical on matched/NA paths (the
  four resolver gates + a non-vacuous NaN-branch anchor); cross-dtype defending now returns the true opponent.
  ADR-019.

## [4.43.0] — 2026-07-10

### Added — public `gk_distribution_mask` + ρ loader `is_gk_distribution` (`silly_kicks/tracking/`, PR-S110, ADR-036 amendment)

- **feat(tracking): public `gk_distribution_mask`.** Exports the GK-distribution domain logic as a stable,
  frame-optional API. `resolve_gk="robust"` (default) resolves the acting GK per action via
  `acting_gk_from_frames` — **time-accurate**: for the GK-pass term it is a strict **subset** of `"native"`
  (the frozen global-`is_goalkeeper` set-membership), *tightening* stale/substituted keepers, NOT broadening
  (do not switch to `native` "for more rows" — those extra rows are stale-keeper noise). `frames=None` →
  goal-kicks-only. The frozen v1 `_gk_distribution_mask` is now a byte-identical shim over it (golden-gated).
- **ρ retention loader/trainer** drop the shot-scoped `gk_was_distributing` (a misused `add_pre_shot_gk_context`
  column — the shot feature itself is unchanged) for a self-adapting, NULL-coalesced `is_gk_distribution` read
  (lakehouse materializes `fct_action_context.is_gk_distribution = gk_distribution_mask(..., "robust")`); the
  loader's `pressure` column is unchanged (`pressure_on_actor__bekkers_pi`, pinned in PR-S109).
- Additive public API; no `xt_gk`/VAEP value change, no retrain. C4 count stays 28.

## [4.42.0] — 2026-07-10

### Added — xT-GK v2 completion: gate run + SP2–SP5 in one release (`silly_kicks/xtgk/`, PR-S109, ADR-036 amendment)

Completes xT-GK v2 (`xT-GK = ρ·[V(s′)−V(s)] − (1−ρ)·[V(s)+κ·V_opp]`) by assembling the metric on
three injected ports and wiring/running the make-or-break gate together (owner directive; the
components are independently valid).

- **Metric assembler** `xtgk.compute_xt_gk_v2(actions, *, possession_value, retention, turnover_cost,
  kappa=1.0)` depending only on the `PossessionValue`/`RetentionModel`/`TurnoverCost` ports. Four
  coherent additive decomposition columns summing to the metric: `xt_gk_v2_position`, `xt_gk_v2_pev`
  (dormant while `p′=p`, pending receiver-pressure `q`), `xt_gk_v2_retention_loss`, `xt_gk_v2_dzv`,
  and the headline `xt_gk_v2` (= RAV, the total). Namespaced away from v1's frozen `xt_gk_pev/rav/dzv`.
- **SP2 `V_opp`** (`_turnover.py`): `TurnoverCost` port + `MirroredTurnoverCost` = `V(mirror_zone(z),
  policy(p))` on the fitted V (zero new fitting; `mirror_zone` 180° reflection; injectable `p_opp=p`)
  + `EmpiricalTurnoverValue` bounded-window cross-check + `_is_turnover`.
- **SP3 `ρ`** (`_retention.py`): `RetentionModel` port + `GkRetentionModel` (logistic, pure-numpy
  serve, JSON+SHA256, provider variants) mirroring `GkCompletionModel`; new `retains(actions, *,
  window_seconds=10.0)` label (`_retention_labels.py`; truncated-window→NaN excluded); **marts-native
  `extract_retention_features`** (8 features: geometry + `pressure_on_actor__bekkers_pi` from the
  gold action marts — tracking-frames deprecated, so the frames-only receiver-density feature is
  dropped and the domain is goal-kicks); calibration gate `ece≤0.10 AND |reliability_slope−1|≤0.25`
  via the extra-free `silly_kicks/_calibration_metrics.py`; `scripts/train_gk_retention.py` +
  `_loader_databricks.load_retention_cohort`. **`default` weights bundled** (GS AUC 0.776/ECE 0.090/
  slope 1.01, PASS); the SkillCorner variant is not shipped (near-chance + fails calibration under
  bekkers_pi → all providers fall back to `default`).
- **Gate (Part 1)**: gate-enforced `GateConfig.relative_effect_floor` (primary `≥0.25`); real
  zone-conditional terciles (`PressureLevels.mode="zone_conditional"`, per-band cutpoints; global
  on-disk form byte-identical, absent `pressure_mode`⇒global); three-rung ladder
  `run_gate_with_ladder` (global→zone-conditional→STOP) + `run_gate_both_orientations`; locked Q4
  numbers wired into `scripts/validate_xtgk_possession_value.py`; RM included PROVISIONAL (100% OOD).
- **SP5**: owner-run `scripts/validate_xtgk_v2.py` (out-of-sample construct validity vs completion /
  destination-only V / v1 composite, transfer, calibration; WC2018/Neuer stubbed).

v1 `tracking/_xt_gk.py` FROZEN alongside v2 (removed ≥1 release after the lakehouse migrates); no
v2↔v1 imports; `xthreat` + v1 byte-stability regression-gated. In **no** default xfn list (opt-in).
ADR-011 does not govern the trained-light ρ.

**Make-or-break gate — RUN on Databricks gold** (`_loader_databricks.load_xtgk_cohort`; pressure pinned
to `pressure_on_actor__bekkers_pi`): **GO-leaning — a real, decreasing, monotone deep-zone pressure
gradient on both cohorts** (WC2022 relative effect 0.86, 8 cells; RM 1.05, 17 cells). Not a clean PASS:
WC2022's absolute effect (0.0027) is under the pre-registered 0.005 floor (deep-zone xG is intrinsically
tiny) and the empirical cross-check diverges — both Eyestone-review items, not the degenerate STOP. The
pressure measure was pinned to bekkers_pi after the lakehouse 3-method audit (andrienko_oval floors to 0
for ~47% of actions — an artifact; the initial andrienko run STOPped, superseded). Two bugs the run
surfaced+fixed: `prepare_cohort` drops frame-absent tracking-gap nulls; NaN-safe `flat_zones` at the
zone-binning seams. See `docs/research/xtgk_possession_value/{GATE_FINDINGS,LAKEHOUSE_HANDOFF}.md`.

## [4.41.0] — 2026-07-09

### Added — xT-GK v2 SP1: Q3 xG-source wiring + G8 frame-aware null-pressure (`silly_kicks/xtgk/`, PR-S108, ADR-036 amendment)

Increment on the 4.40.0 possession-value surface, resolving the two owner-only blockers against the
live backend (spec rev 4 §5/§6). Additive; no production/xfn change; Phase 11 still wired-but-not-run
(blocked on Q4 gate numbers only).

- **G8 — frame-aware null-pressure rule.** New pure `coalesce_frame_present_null_pressure(pressure,
  frame_present)`: a **frame-present + null-pressure** action (a genuinely *unpressured* restart — e.g.
  a goal-kick with no opponent in the pressure region; 595/595 WC goal-kicks live) coalesces to **0 →
  LOW tercile (kept)**, while a **frame-absent** null is left null so `PressureLevels.apply`'s fail-loud
  stays the backstop for genuine tracking gaps. Corrects the original §5 blanket "fail-loud on missing
  pressure", which would have silently dropped 60% of WC goal-kicks.
- **Q3 — injected reward provenance.** `MarkovPossessionValue.fit(reward_provenance=)` records a
  caller-supplied OOD-rate/CI summary from the lakehouse `fct_shot_xg.xg` mart (the injected
  `xg_column`); silly-kicks never interprets `ood_flag`/CI semantics (ships no xG model). Pre-gate
  input-QC helpers `ood_rate_by_source` + `frame_present_null_pressure_count` for the owner-run.
  **Certification note: `fct_shot_xg.ood_flag` = 0 for gradientsports (certified) but 100% for
  skillcorner (all RM shots OOD) → RM gate verdict is provisional.**
- Owner-run `scripts/validate_xtgk_possession_value.py` wires the frame-aware prep + reports +
  provenance (still not run). ADR-036 amended.

## [4.40.0] — 2026-07-09

### Added — xT-GK v2 sub-project 1: honest possession-value surface `V(z,p)` (`silly_kicks/xtgk/`, PR-S107, ADR-036)

A new hexagonal `silly_kicks/xtgk/` package delivering the value function that replaces xT-GK v1's
flat, destination-only raw-xT surface. `V(z,p)` = expected xG the possessing team generates over the
remainder of the possession, given the ball in 16×12 zone `z` under pressure level `p∈{1,2,3}`.

- **`MarkovPossessionValue`** (production) — pressure-stratified value iteration reusing
  `xthreat.value_iteration` verbatim, with (i) an **xG-calibrated first-shot immediate reward**
  (`E[xG|shot]·P(shot)`; NOT the goal-gated `vaep.labels` surface), (ii) a **goal-kick-inclusive
  move-set** (the metric scores keeper distributions), (iii) pressure terciles. `fit`/`value`/
  `surface`/`delta_v` + pickle-free `save`/`load` (npz + JSON + SHA256).
- **`EmpiricalPossessionValue`** (model-free cross-check, not shipped) — per-action first-shot
  empirical surface; independent of the Markov estimator so disagreement is diagnostic.
- **Pre-registered occupied-cell deep-zone gate** (`run_deep_zone_gate`/`GateConfig`) — the
  make-or-break go/no-go; effect floor, `N_min`, direction, and cross-check tolerance are owner-locked.
- **`delta_v`** two-factor Shapley split (`ΔV_pressure + ΔV_position = ΔV`) for the metric layer.
- **Injected `xg_column`** — silly-kicks ships no xG model; the reward is sourced from the lakehouse
  `fct_shot_xg` mart. `V` is fittable only where a calibrated per-shot xG exists (the fit cohorts).

xtgk-local builders reuse `xthreat`'s low-level seams and modify **no** `xthreat` source — classic xT
stays byte-identical (parity-gated over random cohorts + the frozen oracle). In **no** default xfn
list (opt-in). Owner-run real-data gate wired in `scripts/validate_xtgk_possession_value.py`
(blocked on the locked gate numbers). Attribution: Singh 2018 (xT lineage), Eyestone (xT-GK v2).

## [4.39.0] — 2026-07-01

### Added — `acting_gk_from_frames` (acting-team GK resolver, PR-S106)

A public composable resolver (`silly_kicks.tracking.acting_gk_from_frames`) — the **mirror of
`defending_gk_from_frames`** (TF-13) — returning per action the **acting team's** goalkeeper
`player_id` from tracking frames. Enables a downstream consumer to override a goal-kick's NULL taker
(a goal-kick's taker is unambiguously the acting keeper, but the AC layer's ball-carrier fill credits
whatever outfielder is near the downfield event ball → the value/origin were right, the *credit* was
spread across 29–35 outfielders/match).

- **Identity fallback:** unlike the pure per-frame link (which returns NaN whenever the keeper isn't
  detected in the linked frame — ~40% of goal-kicks on broadcast tracking), it resolves the acting
  GK from the **roster-stable `is_goalkeeper` identity** for that `(game_id, team_id)` even when
  undetected at the event frame (relies on the 4.38.0 roster-trust that keeps `is_goalkeeper` set on
  the keeper throughout the match).
- **GK-sub safety:** a `(game, team)` with >1 keeper identity picks the one **nearest-in-time** to the
  action.
- Shared body factored (`_gk_from_frames_linked`, team predicate parameterized) → `defending_gk_from_frames`
  is **byte-identical** (regression-gated). dtype matches frames' `player_id` (object vs Int64).

Additive — no existing behavior changes; a **pure `player_id` resolver** (never mutates `actions`).
Deciding *when* to apply it (the goal-kick actor override) is the lakehouse's separate synthesis step.

## [4.38.0] — 2026-06-30

### Fixed — SkillCorner GK identification: trust the native roster (batching bug)

`tracking.skillcorner.convert_to_frames` discarded the clean native roster `is_goalkeeper` and
**re-derived it positionally** via `derive_goalkeepers` on every call. Stable on a full match, but on
the lakehouse's 250-frame batches a transiently goal-parked outfielder (a defending CB / camped
forward) gets flagged; across ~164 per-batch builds the **union reached ~15 "keepers"/team** — so
`xt_gk` (a goalkeeper metric) was computed for **19–24 players/match** (≈ both full squads) on the
public SkillCorner gold instead of the ~1–2 keepers. (GS / sportec / idsse were immune — their
converters already trust the native roster.)

- **S1 — trust the roster:** when the input `is_goalkeeper` is a valid native flag (≥1 GK per
  `(game_id, team_id)`), use it as-is (`is_goalkeeper_source = "native"`) and **skip
  `derive_goalkeepers`**; derive only as a fallback for a `(game, team)` whose native flag is absent.
  This makes SkillCorner **batching-immune** (the roster flag is identical in every batch — verified on
  real bronze: per-batch union **15/13 → 1/1 per team**), matching `gradientsports.py`/`sportec.py`.
- **S2 — loud, observable guard:** `warnings.warn` + `TrackingConversionReport.n_implausible_gk_teams`
  when a resolved per-`(game, team)` GK count is implausible (`>2` or `0`). Never fires once the roster
  is trusted, but guards the derive-fallback path so squad-wide contamination can't recur silently.

Regression: GS / sportec / idsse / **metrica** keeper identification unchanged (SkillCorner-only).
This is an `xt_gk` serve-output change for SkillCorner (far fewer scored players) → the lakehouse
re-materializes. **Out of scope (follow-up):** Metrica (no roster) is contaminated the same way and
needs derive-once-per-match (not per batch) — a separable silly-kicks + lakehouse change.

## [4.37.0] — 2026-06-30

### Changed — SkillCorner keeper-origin resolution (broadcast-tracking domain fix, ADR-024 amendment)

SkillCorner broadcast tracking records the GK-distribution origin as the **ball-detection event
location, not the keeper's position** (goal-kick `start_x` SD 23.2; own-box rate 51% vs ~100% for
full-tracking providers). `resolve_gk_geometry` previously trusted that non-NaN native origin,
corrupting `xt_gk` base/DZV and the keeper pressure/PEV. This release makes native-origin trust
**provider-aware** and resolves SkillCorner keeper origins via a detection-aware ladder.

- **Fail-safe provider allowlist** (`native_origin_is_trusted`): unknown / `None` / future providers
  default to **distrust** (route through the ladder); only known full-tracking providers
  (`gradientsports`, `idsse`, `metrica`, `sportec`, `statsbomb`, `wyscout`) trust the native origin.
- **Detection-aware ladder — GOAL-KICKS ONLY** (opt-in `distrust_native_origin` on
  `resolve_restart_geometry` / `resolve_gk_geometry`, default-off → byte-identical): a goal-kick's
  origin resolves by **detected keeper within ±1 s** (`visibility`-gated, nearest-in-time
  ties→at-or-before), **re-projected to action-LTR (ADR-028)** + in-box clamp → `tracking_gk`; else
  rule-point `(5.5, 34)` → `goalkick_prior`. **Open-play GK passes/throws keep their native origin**
  (the ball is at the keeper when they release it → native IS the keeper, validated 0.4 m). Scope
  narrowed from the CR's all-GK-distributions after real-data validation (`unresolved` now rare).
  Destination unchanged (origin-only distrust).
- **S4 out-of-region guard** (`flag_native_goalkick_out_of_region`): a native goal-kick origin
  beyond the penalty area warns + sets a machine-observable per-row flag
  (`xt_gk_native_goalkick_out_of_region`) + `XtGkReport.n_native_goalkick_out_of_region`; never
  reverts/crashes.
- **S1 within-pitch invariant** (`skillcorner.convert_to_frames`): per-row off-pitch → warn +
  `TrackingConversionReport.n_gross_off_pitch`, never clamp. **Layered:** the per-row catastrophic
  hard-fail for player coords stays the pre-existing `derive_goalkeepers` raise (a sign/origin
  transform break trips it; unchanged); S1 adds a thin observability band a fixed margin **inside**
  that shared bound for players, and is the **sole** off-pitch signal for the ball
  (`derive_goalkeepers` is player-only). The deferred CI rate-gate is the systematic backstop.
- **C1 (Hyrum-surface, named explicitly):** removed the mixed-provider `completion=` escape hatch.
  `compute_xt_gk` now enforces one-call-one-match uniformly across the completion AND geometry paths
  (a >1-provider frame set raises even with `completion=`). Verified no caller relied on it.

`xt_gk` is opt-in (in no default xfn list) so this is **not** a forced VAEP retrain, but it changes
the **`xt_gk` serve output for SkillCorner** GK distributions → the lakehouse re-materializes.
Full-tracking providers (GS/idsse/metrica/sportec) are byte-identical (regression-gated).

## [4.36.1] — 2026-06-29

### Added / Docs — xT-GK pre-Jeff verification (handoff Items 2 + 4)

- **Item 4 — golden hand-worked composite test** (`TestGoldenComposite`): a fully-controlled GK
  distribution (known grid σ=0, known coords, pinned pressure ρ via a known `rho_raw`, stubbed
  completion p) with `base`/`pev`/`rav`/`dzv`/`T`/`composite` derived from the **literal formulas**
  (independent of the production helpers) and asserted exact — the first test that proves the assembled
  composite arithmetic end-to-end (unit tests pass without guaranteeing the assembly is right).
- **Item 2 — test↔production parity**: added `_production_amplitude_xt` (defensive third raw xT ≈ 0.0085
  → deep `V_GK` ≈ 0.02) + `TestProductionAmplitude`, reproducing the live WC2022 DZV scale (**+0.02**);
  the cube-ramp `_gk_realistic_xt` understated it (goalkick origin raw xT = 0 → DZV ≈ 0). Parity audit
  written to `docs/research/xtgk_test_production_parity_audit.md` (input-contract table, the proven
  amplitude gap, id-dtype/provider coverage notes, lakehouse-side live-schema confirm flag).
- **Item 5 doc-fix** (magnitude framing): ADR-024 / CLAUDE.md / the [4.35.0] entry corrected — the
  earlier "O(0.01) / deep `V_GK` 0.005–0.01" was the *understating unit fixtures*; production deep
  `V_GK` ≈ 0.02 → DZV ≈ +0.02 (live +0.021); the 2× vs Jeff's ~0.009 La Liga anchor is **grid
  amplitude** (within his sanity band), the PEV/DZV forms are faithful.
- Tests + docs only — **no library code change, no behaviour change** to any `xt_gk_*` value; the new
  fixture is test-only (not a shipped artifact). Released for external visibility. Handoff Item 3's
  guard-test half remains open (separate).

## [4.36.0] — 2026-06-29

### Added — xT-GK resolved-coordinate audit columns (ADR-024 amendment, PR-S101)

- `compute_xt_gk` now emits four resolved-coordinate columns — `xt_gk_origin_x`, `xt_gk_origin_y`,
  `xt_gk_dest_x`, `xt_gk_dest_y` — exposing the **exact origin/destination the grid lookups used**
  for every in-scope GK distribution (NaN off-scope). For goal-kicks ~67% of these are *imputed*
  (the `resolve_gk_geometry` rule-point / tracking-GK origin), not the native `start_x`/`end_x`, so
  every `xt_gk` row is now externally auditable — anyone can see which coordinates produced each
  value. Emitted before the not-scoreable early return, so an unresolvable-destination goal-kick
  still shows its resolved origin (+ NaN dest).
- The coords are a parallel **`_COORD_COLS`** audit set, deliberately **not** in `_OUTPUT_COLS` —
  `xt_gk_xfns` does not surface them as per-slot VAEP features (they are provenance, not a metric).
  They ride through `add_xt_gk` (+ atomic mirror) automatically.
- Tie-to-value test: `xt_gk_base == −xT*(xt_gk_origin_x, xt_gk_origin_y)` on the convolved grid for a
  scored row — pins the persisted coords to the computed value.
- Liveness gate: the four audit coords join the non-constant `provenance` exemption (they are the
  coords the lookups used, legitimately constant across goal-kicks sharing the `(5.5, 34)` rule-point
  origin; the non-null check still applies).
- Additive — **no** behaviour change to any existing `xt_gk_*` value; no retrain trigger. Enables the
  lakehouse persist-coords schema migration (held on this) and external orientation verification.
  C4 count unchanged (28). Atomic mirror inherits.

## [4.35.1] — 2026-06-29

### Fixed — exclude pandas 3.0.4 (C-layer segfault on py3.11+)

- **pandas 3.0.4 segfaults (SIGSEGV / exit 139)** in its C `take_nd` → `maybe_promote` path when a
  whole-DataFrame boolean mask carries a `datetime64` column — reproduced deterministically on
  Python 3.11+ via `spadl.orientation.detect_input_convention` (the sportec actions carry a datetime
  column), which crashed the CI test suite. Bisected in an isolated py3.12 env: **3.0.2 ✓, 3.0.3 ✓,
  3.0.4 ✗** (same numpy 2.4.6 / scipy 1.18.0), so it is purely a pandas-3.0.4 regression.
- Dependency constraint tightened to `pandas>=2.1.1,!=3.0.4` — excludes **only** the broken release
  (pip resolves the safe 3.0.3) so a fixed 3.0.5+ is adopted automatically. `uv.lock` regenerated.
- No library code change; no behaviour change on a non-broken pandas. To be reported upstream
  (pandas-dev/pandas).

## [4.35.0] — 2026-06-27

### Changed — xT-GK PEV/DZV fidelity fix (Eyestone Q1–Q3, ADR-024 amendment, PR-S100)

- **PEV now measures its forward gain on the GK-revalued surface** `V_GK = xT · φ(z,d)`, not raw
  `xT` (CHANGE 1, Eyestone Q1+Q2). On the raw grid the keeper-zone forward gain is ~0 — the measured
  PEV inertia — because keepers live in the flat part of the xT surface; revaluing the surface is the
  point. `progress = V_GK*(z′) − V_GK*(z)`; the pressure-gated rectified form `PEV = ρ·max(0, progress)`
  is **unchanged**. RAV remains the sole owner of the destination value, so Option B is untouched (no
  double-count).
- **DZV is now the published defensive-zone revaluation multiplier** `M(z) = φ(z,d)·[1 − V_GK(z)/max V_GK]`
  applied as the **revaluation increment** on the origin possession value, `(M−1)·V_GK(z)`, gated to
  the defensive third (CHANGE 2, Eyestone Q3; Option A). This replaces the old additive `v_def − xT_raw(z)`
  back-pass floor. The increment (not the revalued total) keeps base — which surrenders the origin's raw
  threat — orthogonal to DZV. Per-action DZV lands O(0.01), not the raw multiplier's O(2.5). (**Magnitude
  clarified 2026-06-29 / PR-S103:** that O(0.01) was the *unit fixtures*; on the corrected production grid
  deep `V_GK` ≈ 0.02 → DZV ≈ **+0.02/action** (live WC2022 +0.021). The 2× vs Jeff's ~0.009 La Liga anchor
  is grid amplitude, within his sanity band — form faithful. See `docs/research/xtgk_test_production_parity_audit.md`.)
- **φ(z,d)** `= α·(1 − d/D_max)^(−β)` for `d < D_threshold`, else 1, with `d` = LTR origin x: `α=2.1`,
  `β=0.8` are **canonical** (Eyestone 2026-06-27); `D_max=105`, `D_threshold=35` (= `defensive_third_boundary`)
  are provisional. `XtGkParams` gains `dzv_alpha`/`dzv_beta`/`dzv_d_max`; the now-dead `v_def` is retired.
  The scalar `phi` param stays the preset-modulated overall DZV weight (the canonical shape lives in the
  φ grid).
- **Invariant (Eyestone constraint):** φ enters value via PEV and DZV **only** — base keeps `−xT*(origin)`
  and RAV keeps `xT*(z′)`/`xT*_counter` on the raw `xT*` surface. Guarded behaviorally
  (`test_phi_shape_changes_only_pev_and_dzv_not_base_or_rav`).
- **Not a forced VAEP retrain** (xt_gk is opt-in, in no default xfn list) — but an `xt_gk` serve-output
  change: the lakehouse re-materializes `fct_action_context` and re-runs the WC2022 cohort/report. C4
  count unchanged (no new aggregator/model/backend; stays 28). Atomic mirror inherits.

## [4.34.0] — 2026-06-19

### Changed — TF-23b geometric frame-LTR backstop on the native tracking adapters (ADR-035, PR-S99)

- The native tracking adapters `tracking.gradientsports.convert_to_frames` and
  `tracking.sportec.convert_to_frames` now **self-correct a wrong/absent extra-time direction
  flag from goalkeeper geometry**, via a shared `direction.finalize_orientation` tail that layers
  the idempotent geometric backstop (`orient_frames_to_ltr_by_geometry`) on top of the per-period
  flag-flip. Byte-identical no-op on the correct-flag path. Closes ADR-031 **Gate D** (IDSSE-ET
  handedness). **VAEP/tracking retrain trigger** for the ≤3 GS WC2022 ET-tracking matches + any
  wrong-flag IDSSE-ET whose ET flag was wrong — see ADR-035 for the exact (G1) changed-match list.
- Public-net change: `orient_frames_to_ltr_by_geometry` gains `on_missing_home` and `copy`
  parameters (both additive, default-preserving — direct/lakehouse callers byte-identical), and
  **no longer orients period-5 / penalty-shootout frames for any caller** (including the TF-23
  SkillCorner/Metrica builders). PSO frames are excluded from geometric analysis (practical impact
  nil); the lakehouse self-assesses any SkillCorner/Metrica PSO re-materialization.
- The backstop's zero-home warning text changed (now emitted by the net via `on_missing_home="warn"`).

## [4.33.0] — 2026-06-18

### Added — TF-23 SkillCorner + Metrica bronze→frame builders (ADR-034, PR-S98)

- Two pure, bronze-consuming converters — `tracking.skillcorner.convert_to_frames` and
  `tracking.metrica.convert_to_frames` — parallel to `tracking.sportec` /
  `tracking.gradientsports`. They single-source the SkillCorner/Metrica coordinate
  rescale, period-relative clock, id-namespacing, GK derivation, speed, and LTR
  orientation that the luxury-lakehouse previously duplicated three ways (the kloppy
  gateway oracle + two lakehouse builders). Emit the kloppy-variant schema
  (`SKILLCORNER_TRACKING_FRAMES_COLUMNS` / `METRICA_TRACKING_FRAMES_COLUMNS`).
- `tracking.orient_frames_to_ltr_by_geometry` — flag-free geometric frame-LTR
  orientation (per-period home-GK-median-x anchor, point-reflect mis-oriented periods,
  idempotent), a schema-adapted port of the luxury-lakehouse ADR-053
  `correct_frames_to_home_ltr`. Retained alongside the flag-based `orient_frames_to_ltr`.
- SkillCorner `ball_z` recovery — the builder maps the (previously discarded) real ball
  height into the `z` column, unblocking SkillCorner post-shot height features (TF-48
  PSxG) that were silently null in production.

### Input contract (Metrica)

- `tracking.metrica.convert_to_frames` requires bronze `y` in **SPADL bottom-to-top**
  convention (the lakehouse bronze landing already provides this). kloppy's metrica NATIVE
  coordinate system is top-to-bottom, so a consumer landing bronze straight from a kloppy
  `TrackingDataset` must flip `y` (`1 − y`) first. Fed contract-honoring bronze, the
  builder matches `tracking.kloppy.convert_to_frames` byte-for-byte (validated dx=dy=0 on
  Metrica open-data game 1, incl. LTR orientation).

### Notes

- Additive; no silly-kicks model retrain (new modules + new public orienter; existing
  converters/gateway untouched; in no default xfn list). The luxury-lakehouse adopts the
  builders + orienter and retires its two builder copies + its orientation net (its
  re-materialize trigger, not silly-kicks'); ADR-031 Gate C closes on the shipping path
  via the event-anchored y-identity gate. GS native-adapter ET orientation is tracked as
  the TF-23b follow-on.

## [4.32.0] — 2026-06-16

### Added — `add_*` input-purity CI gate (ADR-033, PR-S97)

Every public `add_*` enricher must be PURE: it must not mutate any caller-supplied DataFrame/Series/ndarray
and must return a NEW object. New auto-enumerating gate `tests/test_add_star_purity.py` — one canonical
`PURITY_ENTRIES` registry covering the full public surface (`spadl`/`atomic.spadl`/`tracking`/
`atomic.tracking`, including the 15 `atomic.tracking.features` mirrors), build-fresh-owned-inputs-once +
snapshot-every-array-arg + value-equality + `out is not input`. Two meta-assertions pin the surface to the
public export (`__all__` UNION `.features.__all__`), so a new `add_*` cannot land unregistered. A best-effort
AST heuristic nudges toward per-branch coverage; the contributor contract (CLAUDE.md: any column-conditional
`add_*` registers ≥2 variants) is the real backstop. Joins the auto-enumerating-gate family alongside
nan-safety / liveness / dup-`action_id` / id-dtype.

### Fixed — `add_gk_distribution_metrics` mutated the caller's frame when `gk_role` was present (ADR-033, PR-S97)

When `gk_role` was already present, both `add_gk_distribution_metrics` implementations (standard +
atomic-SPADL) assigned their four columns straight onto the caller's input DataFrame and returned it,
contradicting the documented "Sorted copy" contract (the `gk_role`-absent path always copied, so the old
column-list mutation guard never caught it). Now hoists `out = actions.sort_values([...]).reset_index(drop=True)`
to the top and operates on `out`. **Identity + order only, no value miscompute, no recompute:** the sort key
matches `add_gk_role`'s internal sort (so the `require_gk_role` path is value/order-identical) and derivation
is per-row vectorized — the lakehouse need not re-materialize. The repo-wide audit found the mutation class is
otherwise clean (only this helper).

### Changed — `pitch_control_at_action` → `pitch_control_at_target` (BREAKING rename; ADR-033, PR-S97)

The action-coupled function is renamed to match its emitted `pitch_control_at_target__<method>` column base
(unchanged since 4.31.0) — standard + atomic, plus `__all__`, imports, and callers. Window-justified: no
released consumer of the 4.31.0 column rename exists yet. The lakehouse keeps its own DEFCON
`pitch_control_at_action` mart column (different semantics; not silly-kicks').

### Changed — docstring tightening (Part C; ADR-033, PR-S97)

Enumerated emitted columns + dtypes for `add_off_ball_runs` / `add_off_ball_context` / `add_shot_goalmouth`;
added the `gk_pass_length_class` Categorical/Spark-`StringType` note and the `gk_xt_delta`
caller-supplied-`(12,8)`-SPADL-grid (never self-fit) note to `add_gk_distribution_metrics` (standard +
atomic). A doc-accuracy test pins each exhaustive-claim helper's emitted feature set to an explicit
`frozenset` and asserts the docstring names every column.

## [4.31.0] — 2026-06-16

### Changed — pitch control re-aimed to the action destination; dead at-ball column retired (ADR-032, PR-S96)

**BREAKING column rename.** The informationally-dead `pitch_control_at_ball__<method>` (the Spearman PPCF
at the ball is the degenerate ~0.5 reaction-time fallback, so the column was ~0.5 for every well-linked
action) is **retired** and replaced by a live `pitch_control_at_target__<method>` sampled at the action
**destination** `(end_x, end_y)`, where ball-travel-time is positive so players can contest it.

- **Mandatory ADR-028 re-projection (fixes a latent away-team bug the degeneracy had masked).** The old code
  sampled the action-LTR `(start_x, start_y)` against an absolute-frame (home-attacks-right) surface with no
  per-action flip — wrong for away-team actions, harmless only because near-ball is 0.5 in both conventions.
  The new code re-projects the query via `acting_team_attacks_rtl` + `reproject_to_action_ltr` (the cached
  per-frame surface + `PitchControlCache` key are unchanged — only the query point flips). Applies to all
  three methods (`spearman`/`fernandez_bornn`/`voronoi`) and the atomic mirror (synthesizes `end=x+dx,y+dy`).
- **Per-type semantics (kept uniform):** open-play destination control for passes/crosses/carries;
  target-cell contestation (GK/defender-dominated) for shots; ~0.5 for in-place actions (no destination —
  honest). A model conditions on this via `type_id`.
- **Localized:** other PPCF consumers (`obso`/`cover_shadows`/`gk_influence`/`player_influence`/
  `space_creation`) sample their own points and are untouched. The dead column's `STRUCTURAL_CONSTANTS`
  liveness exemption + its near-ball-degeneracy invariant test are removed (the column is now live); a hard
  off-ball-destination precondition guards the liveness gate's teeth.
- **VAEP/tracking + calibration retrain trigger** (dead constant → live signal + away-team correction). The
  silly-kicks calibration feature set + lakehouse consumers re-materialize. **Lakehouse adoption is a
  breaking column-lifecycle migration (AC + DEFCON), atomic with the pin bump — not a currency bump.** C4-free
  (aggregator count stays 28).

## [4.30.0] — 2026-06-16

### Added — DFL / Sportec parse+shape port (ADR-031, PR-S95 / T3)

A new `silly_kicks/providers/sportec/` package (behind a `[parse-dfl]` optional extra) **single-sources
the IDSSE/Sportec DFL parser**, eliminating the dev/prod parser drift and retiring the y-inverting
loader-local kloppy `_kloppy_tracking_to_frames` from the calibration/pining harness in favour of the
native `spadl.sportec` / `tracking.sportec` converters.

- **Public surface:** `parse_dfl_match_info` / `parse_dfl_tracking` / `parse_dfl_events` (DFL XML →
  RAW provider-canonical bronze) + `shape_tracking_to_native` / `shape_events_to_native` (bronze →
  converter input) + `derive_idsse_home_team_start_left{,_extratime}`; typed returns `MatchInfo` /
  `SportecTrackingBronze` / `SportecEventBronze` (silly-kicks' own domain names — a versioned cross-repo
  bronze contract, ADR-031 N1).
- **Verbatim lift.** Parse/shape function bodies are upstreamed byte-for-byte from luxury-lakehouse @
  `0efac60`; the only adaptations are the `logger`-arg defaulting, two inlined cross-module helpers
  (`idsse_native_match_id`, `finalize_bronze_df`), and a materialised 246-column events bronze-column
  set (the lakehouse derives it from a schema module). Pinned by `tests/datasets/sportec/idsse_slice/SOURCE_SHA`.
- **Data-quality is consumer-side.** The port emits RAW bronze (no Savitzky-Golay smoothing / velocity
  derivation); the harness applies `_preprocess` after shaping, and a delete-and-depend lakehouse keeps
  its own smoothing after the parse.
- **Golden parity test** (`tests/providers/sportec/test_parse_port_parity.py`) asserts the port
  reproduces goldens captured by running the **real** lakehouse functions on a reduced real-WC2022 IDSSE
  slice — a genuine "port reproduces production" guard (sensitivity-proven).
- **No new tracking aggregator** → the action-coupled aggregator count is unchanged (28). New C4
  container (`providers.sportec`) feeding both sportec converters.

### Changed — IDSSE harness re-route (N6 retrain trigger)

`scripts/_loader_pining.py::_build_idsse` now parses via the port → native converters. The action↔frame
y-axis now agrees (acting-player frame-y matches the action `start_y` to ~0.2 m after the ADR-028
re-projection, vs ~11.8 m on the retired kloppy path), and the action `team_id` is remapped from the
`"home"/"away"` label to the DFL CLU id so the ADR-028 join aligns with the CLU-keyed frames. **IDSSE
calibration/pining feature values change → those consumers re-materialize.** Documented by
`tests/calibration/test_calibration_invariance_e2e.py`. (Gate D: the native sportec converter was
already y-correct; IDSSE's old misalignment was partial, not the clean SkillCorner-style inversion T1
fixed.)

## [4.29.0] — 2026-06-16

### Fixed — kloppy tracking-gateway y-axis inversion (CS-pin; ADR-031, PR-S94 / T1)

The kloppy **tracking** gateway (`silly_kicks.tracking.kloppy.convert_to_frames`) produced frames with
a y-axis **inverted** relative to the SPADL action y-axis (`action_y == 68 − frame_y`) for every
kloppy-based provider (SkillCorner, Metrica, and the IDSSE dev-harness path). The **event** gateway
pinned the canonical `_SoccerActionCoordinateSystem` (origin `BOTTOM_LEFT`, vertical `BOTTOM_TO_TOP`);
the tracking gateway never did, retaining each provider's kloppy-native vertical. It is a single-axis
y mirror (NOT orientation — orthogonal to ADR-028/029); error `|68 − 2y|` is 0 at centre and ~full
pitch width at the touchlines (why it hid).

- `_SoccerActionCoordinateSystem` extracted to `silly_kicks/spadl/_kloppy_coordinates.py` (with a
  `socceraction_coordinate_system(metadata)` helper); both gateways import it (DRY). The event-path
  output is **byte-identical** (the helper reads the same metadata the inline construction did).
- `tracking/kloppy.py` now pins the coordinate system. **Signature is CS-only** — it drops
  `to_pitch_dimensions` and relies on the CS's own standardized 0–105/0–68 dimensions, matching the
  event gateway. (Keeping `to_pitch_dimensions` while adding the CS silently overrides the CS's
  vertical and leaves y inverted — verified on SkillCorner.) **NOT** a blanket `y = 68 − y` flip
  (which would double-invert an already-canonical provider; guarded by a no-op test).

**Scope (Gate C):** this fixes the **calibration/pining path + external kloppy-gateway consumers** —
the lakehouse builds SkillCorner/Metrica frames via its own bronze builders, not this gateway.
**Retrain trigger:** VAEP + tracking **calibration** consumers for **SkillCorner and Metrica** (both
were inverted; Gate A). The native sportec/IDSSE path is unaffected (Gate D: y-correct). Gradient
Sports native and event-only providers (StatsBomb/Wyscout/Opta) unaffected. Decision: ADR-031.
First of a sequence; the IDSSE/Sportec DFL parse-port single-sourcing (T3) follows in PR-S95.

## [4.28.0] — 2026-06-15

### Added — TF-48 post-shot goalmouth crossing geometry (`add_shot_goalmouth`; ADR-030)

New `silly_kicks.tracking.add_shot_goalmouth(actions, frames, *, links=None, params=None)` derives,
for each shot action (`shot`/`shot_freekick`/`shot_penalty`), the goal-plane crossing from the
post-contact ball trajectory in tracking frames: `shot_crossing_y`/`shot_crossing_z` (SPADL meters,
canonical attacked-goal-at-x=105), `shot_speed` (fitted initial speed at contact — ALWAYS the
contact sub-segment, never a post-bounce refit), `shot_time_to_goal_line`,
`shot_on_target_derived` (posts/bar expanded by the ball-radius tolerance), plus full provenance
(`shot_crossing_source` ∈ {observed, extrapolated, insufficient_frames, no_crossing,
no_ball_frames, unresolved}, `shot_crossing_confidence`, `shot_fit_n_frames`, `shot_fit_rmse`,
`shot_fit_end_reason`, `shot_z_profile` ∈ {airborne, rolling, bounced}). Pure geometry, no model —
the lakehouse scores the output with its existing StatsBomb-trained PSxG model (Goals Prevented for
the tracking providers). Engine (`compute_shot_goalmouth`) is pure + orientation-agnostic (goal
ends from the GK map; `defended_goal_x` extracted byte-identically from xS into `_gk_resolve.py`);
the per-shot kernel is pilot-hardened on real WC2022 data: a sample-and-hold collapse (GS's raw
`balls` channel delivers ~15 Hz positions duplicated at 29.97 Hz stamps — 50% exact
consecutive-duplicate x/y/z, raw-artifact-confirmed; held duplicates are phantom zero-velocity
samples that phase-modulated every speed gate and saw-toothed the fits into phantom
trajectory-breaks), flight-run anchoring for t0 (GS stamps shots up to ~2.6 s before contact) with
a contact anchor (the shooter's own action coordinates — measured exact ball-track points on GS —
split a continuous assist-cross/dribble + shot approach run at the contact; orientation-agnostic
via the goal_x reflection), 0.1 s-baseline velocities (per-frame finite differences amplify
29.97 fps jitter ~30×), LOCAL residual break checks (a segment-anchored linear residual
phantom-breaks any smoothly curving chip/curl ~1 s in; a deflection violates even the local fit),
z-aware flight classification (an airborne decelerating chip is flight; carries/frozen tails are
on the ground), a flight-core trim (slow ground heads/tails removed; away-flying balls stay honest
`no_crossing`; sub-flight balls `insufficient_frames`), an extrapolation-leverage cap
(`max_extrapolation_leverage`: t\* beyond 3× the fitted span is a guess, not a fit — pilot-measured
dy median 6.2 m vs 2.4 m below the cap), and a contact-EXISTENCE bar (a window whose ball never
comes contactably near the stamped shot location — 2-D within 5 m at playable height z ≤ 2.6 m —
provably does not contain the shot → honest `insufficient_frames`; kills the measured worst class,
a 12.6 m "observed" goal crossing fitted from a pre-contact assist arc passing 6 m overhead).
`ShotGoalmouthParams` (pilot-calibrated defaults) + `ShotGoalmouthReport`
QA aggregate + per-Series wrappers + atomic mirror ({shot, shot_penalty}; `shot_freekick` is a
`freekick` atom). **NO VAEP xfns factory** — post-contact outcome descriptors are
HybridVAEP-class result leakage; a guard test auto-discovers every default xfn list and asserts
absence. NOT in any default xfn list → **no retrain trigger**. C4 action-coupled-aggregator count
27 → 28. Owner-gated GS↔StatsBomb WC2022 acceptance harness
(`scripts/validate_shot_goalmouth_sb.py` + held-out e2e with ADR-pre-registered floors; goals/saves
stratified — SB save end_locations are save-points, not plane crossings). **Holdout-validated
accuracy (one-shot protocol, ADR-030 pre-registered floors, 48 held-out WC2022 matches, 999
matched shots): goals |Δy| median 2.17 m (floor ≤ 2.5; p90 5.7 — tail dominated by
observed-straddle GS-vs-SB hand-tag disagreements where GS's own in-net samples corroborate GS),
|Δz| median 0.48 m (floor ≤ 1.25), on-target resolution coverage 0.620 (floor ≥ 0.60), on-target
agreement 0.60 (floor ≥ 0.45) — ALL FLOORS PASS.** The meters→SB y-handedness is settled on GK
GEOMETRY (SB shot freeze-frame defending GK vs the GS-tracked GK: 0.882 flip agreement on 646
voters, pilot-vs-holdout instrument-stable at 0.883/0.882; the round-1 ball-tag gate was measured
too noisy to settle a transform and is demoted to an informational diag). Holdout round 1 aborted
at that ball-tag gate, and its documented failure analysis exposed a harness clock-base bug that
had silently excluded ALL period-2 shots from every pilot metric (GS SPADL `time_seconds` is the
CUMULATIVE match clock — the known lakehouse-guarded GS convention — while the harness converted
SB to period-relative; fixed, matching now covers both halves; full record in ADR-030).
Crossing-z is GS-z-channel-limited (onset lag). **Completion cycle (4.28.0):** two kernel
refinements, re-validated on the FULL 64-match GS corpus (the post-holdout protocol): (1) a
span-gated curve-aware y extrapolation — the constant-velocity fit extrapolated a curling/dipping
flight's crossing LINEARLY (measured 5.4 m on a real chip-curl goal); when the producing segment
supports a curvature estimate AND a quadratic markedly out-fits the line (real curl, not jitter),
the crossing y is taken from the quadratic, capped tighter than the linear leverage; (2) an
earliest-reaching flight-run tie-break — when >1 plane-approach run reaches the goal line, the SHOT
is the EARLIEST (the bare nearest-plane rule had anchored t0 PAST a real in-mouth crossing on a
measured holdout goal). Full-corpus re-validation: goals |Δy| median **2.08 m** (improved from
holdout 2.17), |Δz| 0.49, coverage 0.63, on-target agreement 0.61, GK-handedness 0.882 on 851
voters — ALL FLOORS PASS, no regression. A final-kernel sensitivity sweep (`--sweep`, extended to
the contact/flight module constants on a 10 fps-downsampled copy) confirms the kernel is robust (no
cliffs) and the new constants are inert on resolution. **Provider coverage:** GS is validated;
SkillCorner/IDSSE currently return `insufficient_frames` due to a SEPARATELY-TRACKED upstream bug
(kloppy-derived tracking frames have an inverted y-axis vs SPADL actions —
`docs/research/bug_kloppy_tracking_y_inverted.md`, tracked in TODO.md); TF-48's kernel is
provider-agnostic and resolves SkillCorner to the GS baseline once that input is corrected (proven:
a coordinate-fix smoke-test lifts SC resolution 0.12→0.60). Decision: ADR-030.

### Fixed

- **pining GS loader dropped ball z** (`scripts/_loader_pining.py`): every frame row was hardcoded
  `z=0.0` and the raw ball records' `z` (present on 100% of GS ball records; probe 2026-06-10) was
  never read → all loader-fed GS frames had flat zero ball z. Ball rows now carry the real z
  (players keep 0.0 — no z in GS player records). Affects any loader-fed analysis that consumed GS
  ball z (e.g. xS ball-z features at GS inference saw zeros). **Audited:** re-ran the xS PR-S80
  public-vs-full data-effect test with real GS z (controlled A/B at the shipped public params, GS z
  real vs forced-0) — all 5 folds stay negative (mean Δ -0.058 with real z vs -0.026 with z=0;
  ship_two=false either way), so the shipped xS conclusion (ship the public-only model) is UNCHANGED;
  the loader z bug did not flip it.

## [4.27.1] — 2026-06-15

### Documentation
- **ADR-code reconciliation sweep** — verified all 29 ADRs (ADR-001…029) against the current tree; 25 were clean, no behavioral drift found. Corrected stale prose in 5 living ADRs: ADR-004/ADR-005 (`TRACKING_FRAMES_COLUMNS` is 20 columns, not 19 — the `is_goalkeeper_source` provenance column added in PR-S26 was undocumented; now listed), ADR-004/ADR-006 (the `tracking._direction` module path was renamed to the public `tracking.direction` in 4.0.0 per ADR-010), ADR-010 (Status `pending implementation` → `implemented in 4.0.0`), ADR-017 (de-pinned a drifted `gradientsports.py:416` line-number citation). Historical specs/plans/CHANGELOG were intentionally left untouched (immutable point-in-time records; PR-S19 genuinely shipped 19 columns).
- **TODO.md** — collapsed the bloated multi-paragraph "Last updated" header (which had accreted per-version historical notes) back to a single current-release summary line; relocated the parked `pitch_control_at_ball__spearman`-redesign item from the header into Technical Debt → Blocked or Deferred (and fixed a `>`-at-line-start Markdown blockquote render bug in it).

### Notes
- **Documentation-only — no library/package code, schema, or behavior change; no model retrain.** The `silly_kicks` package is byte-identical to 4.27.0.

## [4.27.0] — 2026-06-13

### Added
- `silly_kicks.tracking.orient_frames_to_ltr(frames, *, home_team_id, home_team_start_left, home_team_start_left_extratime=None)` — orients *unlabeled* absolute tracking frames into the canonical home-attacks-right (LTR) frame, single-sourcing the orientation contract for consumers that build frames from a non-kloppy source (bronze DataFrames). Pure composition of existing primitives (`compute_attacking_direction` + `play_left_to_right`) with fail-loud guards (missing-schema, already-labeled → use `play_left_to_right`, zero home-match, ET-without-flag). Companion to ADR-028: ADR-028's per-action reprojection no-ops on absolute frames (`team_attacking_direction = None`), so consumers must orient first. Decision: ADR-029.

### Notes
- **Additive — no model retrain.** Existing providers (sportec/gradientsports/kloppy) are byte-unchanged; the helper is new and not called internally. **Consumer impact:** adopting `orient_frames_to_ltr` in the lakehouse metrica/skillcorner bronze builders fixes their previously-bimodal tracking action geometry (`pre_shot_gk_x`, `defensive_line_x`, `nearest_defender_distance`, `pressure_on_actor__*`, etc.); those providers must be re-materialized lakehouse-side. The helper is only as correct as the caller-derived `home_team_start_left` — validate it per game.
- Added a positive extra-time orientation regression guard for the native `gradientsports`/`sportec.convert_to_frames` ET path (`tests/tracking/test_adapter_extra_time_orientation.py`), prompted by a live GS-ET flip that was a consumer-side `home_team_start_left_extratime` placeholder bug, not a silly-kicks bug.

## [4.26.0] — 2026-06-12

### Fixed — tracking geometry now emitted in the per-action SPADL LTR frame (systemic orientation bug; ADR-028)

**Breaking value change. VAEP/tracking-retrain trigger — re-materialize all tracking action-context.**

SPADL actions are per-acting-team LTR (the acting team attacks x=105); `convert_to_frames`
output is home-attacks-right (the home team attacks x=105 every period). The two are a 180°
point reflection apart for away-team actions, and the tracking-geometry layer sampled frame
positions **without re-projecting** them into the per-action LTR frame. On ~50% of
tracking-provider action rows (away-team actions) this produced wrong values:

- **Absolute positions at the wrong end** (visibly bimodal): `pre_shot_gk_x/y`,
  `pre_shot_gk_distance_to_goal` (reached 106 m), `defensive_line_x`, `back_line_high_x`,
  `team_shape_centroid_x/y_*`, `team_shape_defensive_line_height_*`.
- **Mixed-frame scalars** (action anchor combined with frame positions → numerically wrong,
  not just mis-oriented, and not visibly bimodal): `nearest_defender_distance`,
  `receiver_zone_density`, `defenders_in_triangle_to_goal`, all `pressure_on_actor__*`,
  `pre_shot_gk_distance_to_shot`, `pre_shot_gk_angle_*`.
- **`ghost_gk_x/y`** were goal-relative (defended goal at x=0) while the actual-GK features
  intended action-LTR → cross-frame "ghost deviation ≈ 90 m" downstream.

Fixed by one canonical re-projection (`tracking/_action_orientation.py`, driven by the
frame's `team_attacking_direction`) applied at three seams: the shared `ActionFrameContext`
(fixes all 8 context kernels at once and makes their hardcoded goal-at-105 correct),
`_defensive_line_at_actions`, and `add_team_shape`/`_team_shape_at_actions`. `add_ghost_gk`
now emits action-LTR (`x → 105 − gr_x`; `y` mirrored for away actions); the model stays
goal-relative. `compute_team_shape` is additionally made orientation-aware so
`defensive_line_height`/`inter_line_gap_*` are each team's *true* defensive line (was the
min-x cluster for everyone → the away team's advanced line). Self-reconciling features
(`structural_pass`, `gk_influence`, `player_influence`, `cover_shadows`, `shape_graph`,
`obso`, `space_creation`, `das`, `pitch_control`, `pausa`, `xt_gk`) are unchanged. A
mirror-symmetry property test (`tests/tracking/test_action_ltr_mirror_invariance.py`) is the
durable guard. Home-team values are byte-identical; only away-team values change.

Also fixed a latent pandas-3.0 compatibility bug surfaced en route: the frame-fallback GK
resolver in `add_pre_shot_gk_context` filled `defending_gk_player_id` via `.fillna()` with an
object Series; pandas 3.0 stopped silently downcasting the result, leaving the column `object`
(float64 on pandas 2.x), which made the downstream float-vs-object GK id match find zero rows →
NaN GK position. The fill now restores the contractual float64 dtype. Affected real data on
pandas 3.0 whenever the GK resolves via the frame fallback (the common path — DFL/Sportec rarely
emit `keeper_save`).

## [4.25.0] — 2026-06-11

### Fixed — GS null-actor duel/foul events emit NaN team_id/player_id (was sentinel 0); nullable Int64 (lakehouse production outage; ADR-001)

The Gradient Sports converter emitted the integer sentinel `0` as `team_id`/`player_id`
on null-actor events. Because `0` is non-NaN, it masqueraded as a real id, bypassed every
downstream `pd.isna` NaN-route, and crashed the strict opponent-resolution guard in
`tracking._space_creation._resolve_opponent_team_id`
(`ValueError: attacking_team_id '0' does not uniquely match the frame team ids [...]`),
taking down every Gradient Sports unit in the lakehouse action-context pipeline (2026-06-11).
Under ≤4.22.1 the same rows produced silent NaN space values; 4.23.0's loud two-team guard
turned the latent corruption into a hard failure. Good guard, bad input — the fix is upstream,
in the converter.

**Root cause.** `spadl/gradientsports.py` did `events["team_id"].astype("Int64").fillna(0)
.astype("int64")` (and the same on `player_id`) because `SPADL_COLUMNS` types both as
non-nullable `int64`, which cannot hold NaN. Gradient Sports is the only int-id provider;
the kloppy-family providers carry object-string ids where the absent actor is naturally
`None` (pd.isna-routable), which is why no other provider hit this.

**Ground truth (canonical PFF WC2022 feed, 64 matches).** The null-team events are the
two-sided duels and dedicated fouls — **594 `OTB`+`CH` challenges + 28 `FOUL`+`FO` fouls** —
and on **every one of them `gameEvents.playerId` is ALSO null** (a challenge is a 50/50 duel
with `homeDuelPlayerId` *and* `awayDuelPlayerId` and no single owning team; a dedicated foul
has no on-the-ball actor). The only team-resolving ids that exist (challenger / winner /
culprit) are possession-event *qualifiers* — synthesizing `team_id` from them is exactly the
ADR-001 violation that silly-kicks 2.0.0 removed (the sportec tackle-winner override the
lakehouse reported in PR-LL2; ADR-001 itself classifies team-less fouls as *legitimate NULL*).
So NaN is the architecturally-correct value, confirmed with the lakehouse, which withdrew its
original "resolve from the acting player's roster" prescription (that acting player does not
exist on the feed).

**Fix.**

- `GRADIENTSPORTS_SPADL_COLUMNS` types `team_id` / `player_id` as nullable **`Int64`** (was
  the inherited `int64`); they mirror the canonical `gameEvents` actor verbatim, **NaN where
  the actor is absent — never a sentinel 0**.
- **ADR-001-legal self-heal** (`_resolve_team_ids`): where a row has a real canonical
  `player_id` but a null `team_id`, derive the team from that player's other same-match rows
  (a player belongs to one team per match). Keys ONLY on the canonical `player_id` column,
  NEVER on a duel/foul qualifier; an ambiguous mapping raises rather than guesses. On the
  canonical feed this resolves nothing (player_id is null wherever team_id is), so all
  null-actor rows are NaN; it self-heals only genuine player-present/team-absent rows.
- Orientation `_mirror_per_period` is NA-safe (`na_value=False`): a null `team_id` keeps the
  EXACT coordinate orientation the pre-fix sentinel 0 produced (`0 != home_team_id` and
  `NA == home_team_id` both collapse to "not home"), so only `team_id`/`player_id` change —
  coordinates are byte-identical.
- `atomic.spadl.convert_to_atomic` preserves the source `team_id`/`player_id` dtype instead
  of force-casting to the atomic schema's `int64` (which crashed on the new GS NaN — and
  would also have crashed on sportec/skillcorner object-string ids; latent bug fixed).
- **`tracking._line_breaking` (Ward line-breaking, `add_line_break(method="ward")`) — two
  fixes a downstream NaN-safety audit surfaced** (the early opponent-resolution crash had been
  masking them): (1) a NaN-team action now NaN-routes instead of raising
  `TypeError: boolean value of NA is ambiguous` at the opponent-set list-comp (`t != <NA>`);
  (2) the opponent set now uses the ADR-019 `same_id` instead of a raw `!=`, which on a
  mixed-dtype pairing (Int64 action team vs object-string frame team — exactly GS actions on
  tracking frames) was always True and silently kept the actor's OWN team as the "opponent",
  mis-computing every GS Ward line-break. The ADR-019 AST lint missed this because the
  operands are named `t`/`action_team`, not `*_id`. **All 14 frame-aware AC consumers
  (space_creation, obso, pitch_control, shape_graph, team_shape, structural_pass,
  line_break[ward+threshold], das, gk_influence, player_influence, cover_shadows, pausa,
  pressure) verified to NaN-route a NaN-team action on a healthy two-team frame** — no crash,
  real rows still compute; the few non-NaN values on the NaN-team row are team-INDEPENDENT
  frame properties (`pitch_control_at_ball`, `pressure_on_actor`), not miscomputes.

**Impact / re-conversion delta (acceptance #5).** ~**622 SPADL actions per WC2022 corpus**
(≈594 tackle + 28 foul + the 1 null-actor touch→bad_touch) flip `team_id`/`player_id` from
the sentinel `0` to NaN. Downstream these now route to the NaN-row default (e.g.
`space_created` returns the NaN row) instead of crashing — they carry NO enrichment, which is
honest for a contested duel / stoppage. **Hyrum / retrain trigger:** GS `team_id`/`player_id`
dtype `int64`→`Int64` is an observable schema change, and the value flip shifts any
team/player-keyed GS feature for these rows — VAEP/tracking consumers re-materialize GS.
Decision: ADR-027 (GS null-actor NaN identifiers), grounded in ADR-001 (no qualifier→identifier
override) + ADR-003 (NaN-safe enrichment) + ADR-019 (id-dtype contract). C4-free (no new
aggregator/model/backend; count stays 27).

## [4.24.0] — 2026-06-11

### Fixed/Changed — opponent OBSO orientation MIRRORED + LEAN 2-column contract (TF-41 round-2; ADR-026 amended; owner-approved breaking)

The lakehouse rejected 4.23.0's opponent triplet: under a complementary pitch-control model
with a SHARED, UNMIRRORED multiplier, `opp_obso = (1−pc)·M` is fully determined by `pc·M`, so
the opponent leave-one-out was the exact pointwise negation of the team LOO
(`opponent_space_destroyed_m2 ≡ space_created_m2` bit-for-bit; reproduced and
algebraically confirmed — informationally empty). The owner additionally directed a contract
reshape in the same release (no consumer has adopted any 4.23.x surface):

- **Semantic fix — the opponent surface is weighed by the opponent's OWN attacking
  geometry**: the same transition/EPV grid ARTIFACTS mirrored along x to the goal the
  opponent attacks; the ball-anchored distance weight is unchanged. Grid resolution, sigmas,
  and PC method stay shared (magnitudes comparable). Both the analytical
  (complement-decomposition) and naive (explicit recompute) paths consume the same mirrored
  multiplier — one metric, two estimators (round-2 acceptance #4 method-consistency test:
  spearman vs voronoi agree in sign and order of magnitude). Anti-mirror gate (round-2
  acceptance #2) red-first then green; a geography pin makes silent un-mirroring untestable.
- **LEAN CONTRACT (breaking, owner decision): `add_space_creation` now emits exactly TWO
  columns** — **`space_created_m2`** (>= 0; the actor's LOO on their own team's OBSO
  surface; attacking value) and **`space_denied_m2_opponent`** (>= 0; the same LOO on the
  mirrored opponent surface; rest-defense value). The structurally-zero columns are RETIRED
  rather than shipped: the LOO is pointwise-MONOTONE — removing a player can only decrease
  his own team's control and increase the opponent's, everywhere, for every shipped PC
  method — so a team-side "destroyed" (zero since TF-41 shipped) and an opponent-side
  "created" are always 0, and net columns are exact redundancies of the live pair.
  `compute_space_created` is leaned identically (per-player `space_created_m2` +
  `space_denied_m2_opponent`); `space_creation_xfns` is 2 features × 3 gamestates = **6 VAEP
  columns**. A retired-columns guard test blocks any resurrection. This answers the round-2
  question "is team destroyed expected to acquire real values?": NO — the column no longer
  exists (round-2 acceptance #1's non-zero-opponent-created clause is mathematically
  unsatisfiable under removal-based LOO; producing it would need a repositioning-counterfactual
  estimand, out of scope).
- **Liveness gate gains the round-2 non-constant check**: every float metric column added by
  any of the 28 aggregators with >= 2 observed values must carry > 1 distinct value, with a
  declared, justified, invariant-tested `STRUCTURAL_CONSTANTS` registry (never silent
  exclusions). The multi-domain fixture gained real per-window variation (velocities,
  y-layout, GK drift, kick power/timing, event-clock jitter off the frame grid, an isolated
  sprinting carrier, receivers ahead of the block, one-man-down windows).
- **New finding flagged by the gate — `pitch_control_at_ball__spearman` is near-ball
  degenerate**: the Spearman PPCF deviates from the 0.5 fallback only ~18 m+ from the ball
  (the ball reaches nearer cells before any player's reaction time), and the column samples
  linked-action START points, which are always near the ball — so it is **~0.5 for every
  well-linked action in production**. Declared + invariant-tested as a structural constant;
  lakehouse should treat the column as informationally dead pending redesign (tracked in
  TODO). This is the third dead-metric instance the gate's bug class covers.

Hyrum: BREAKING schema change vs 4.23.0 (columns dropped + renamed; opponent values change),
accepted by owner/lakehouse agreement — no consumer adopted the 4.23.x surface and the 4.23.x
line is superseded. **Final adoption column list: `space_created_m2`,
`space_denied_m2_opponent`.** Not a VAEP retrain trigger (xfns opt-in, in no default list).

## [4.23.0] — 2026-06-11

### Added — the space-creation `*_opponent` triplet is IMPLEMENTED (TF-41; lakehouse-mandated; ADR-026)

The lakehouse rejected 4.22.2's contract-removal resolution and mandated implementation
(option 1 of the original report). `add_space_creation` now emits a live
`space_created_m2_opponent` / `space_destroyed_m2_opponent` / `net_space_m2_opponent`:

- **Semantics**: the actor's leave-one-out differential OBSO evaluated on the **opposing
  team's OBSO surface** (actor as defender of that surface), per Fernandez & Bornn (2018).
  `*_created_m2` >= 0 is opponent space existing because of the actor's presence;
  `*_destroyed_m2` >= 0 is opponent space the actor's presence denies (the defensive-value
  reading); `net_*` = created − destroyed (signed).
- **Identical inputs by construction**: same linked frame, evaluation grid, OBSO sigmas,
  transition/EPV grids, and pitch-control method as the `_team` triplet — magnitudes are
  directly comparable. Analytical path (Spearman/F&B) derives the opponent surface from the
  complement of the SAME decomposed baseline (zero extra pitch-control computations);
  the Voronoi naive fallback recomputes the opponent surface explicitly per removal.
  Verified by an analytical-vs-naive opponent parity oracle on both decomposable methods.
- **Opponent resolution** is dtype-robust (`ids_match`, ADR-019). A linked frame without
  exactly two team ids **raises `ValueError`** carrying the game/period/frame/action key —
  corrupt input fails loud, never silent NaN. NaN actor identifiers still route to the
  ADR-003 NaN-row default.
- **NaN-mask parity**: the `_opponent` triplet is NaN exactly where the `_team` triplet is
  NaN (single-call design — no new degradation paths). Gated by a coverage-parity test.
- **Contract lockstep**: `_SPACE_CREATION_COLUMNS` (6), both return paths, the docstring,
  and `space_creation_xfns` (now 6 features × 3 gamestates = **18 VAEP columns**, was 9).
- **Meta-gate (recurrence guarantee), repo-wide**: `tests/tracking/test_aggregator_column_liveness.py`
  runs EVERY registered tracking `add_*` (all 28, including the jersey-frames helper) on a
  multi-domain fixture (pass / shot / GK goalkick / attacking-third ball / wide-area cross
  windows with the actor carrying the ball) and asserts every column an aggregator ADDS is
  non-null somewhere — a documented contract column that is 100%-NaN now fails CI for ANY
  aggregator, with NO exception set (conditional columns get domain-exercising fixtures, not
  exclusions) and a meta-assertion pinning the gate surface to `tracking.__all__` so a new
  aggregator cannot land unwired. Plus the space-creation-specific lakehouse acceptance
  tests: coverage parity, symmetry sanity, sign/range oracle, two-team guard.
- `compute_space_created` gains `include_opponent_perspective: bool = False` (additive;
  default output schema unchanged).

Hyrum: `space_creation_xfns` length changes 3 → 6 (opt-in factory, in no default xfn list —
opting in remains a self-triggered VAEP retrain per ADR-005). `add_space_creation` output
gains 3 columns; existing 3 are byte-identical. Lakehouse re-adds the bronze column
(`ADD COLUMNS` adoption PR) and extends its value-audit oracles. Minor bump per the
lakehouse release-mechanics requirement (contract re-expansion must not ship as a patch
on top of 4.22.2's removal).

### Changed — pyright now gates `tests/` + `scripts/` in CI (infra-only, no wheel change)

- **CI type-gate widened from `pyright silly_kicks/` to the full tree** (config-driven: pyproject
  `[tool.pyright] include = ["silly_kicks", "tests", "scripts"]` + `extraPaths = ["scripts"]` so
  the tests' runtime `sys.path` import of scripts-modules resolves statically). 301 pre-existing,
  never-gated diagnostics across 73 files fixed to zero.
- **`scripts/` fixes carry real hardening** (behavior-neutral at every site): explicit
  `RuntimeError("HPO produced no best candidate")` narrowing in `train_xcross_attempt.py` /
  `train_xshot_occurrence.py` (previously a latent end-of-sweep `AttributeError`); post-`fit()`
  Optional-narrowing asserts in `train_gk_completion.py` / `train_ghost_gk.py`; honest return
  annotations on the `_extract` helpers (pyright NoReturn mis-inference).
- **`tests/` fixes are type-only**: trailing `# type: ignore[...]` per the codebase idiom for
  pandas-stubs/numpy-stubs strictness, a handful of precise annotations (e.g. `MockSurface`
  attribute declarations in `test_obso.py`), and `[import-not-found]` suppressions on the two
  importorskip-guarded optional deps (`statsbombpy`, `xarray`). Every edited test file verified
  byte-identical pass/skip outcomes against its pre-edit baseline.
- Known suppressed-not-fixed class: `ruthless` `IntRange`/`Choice`/`FloatRange` `.log` and
  `StoreConfig` Optional annotations are stub gaps in the ruthless package itself
  (runtime-verified present); fix belongs upstream in ruthless, after which the
  `tests/calibration/test_spaces.py` suppressions can drop.

No library code changed (`silly_kicks/` untouched); not a retrain trigger; nothing re-materializes.

## [4.22.2] — 2026-06-11

### Removed — dead `*_opponent` triplet dropped from the `add_space_creation` contract (TF-41)

- **Breaking (column removal): `add_space_creation` no longer emits
  `space_created_m2_opponent`, `space_destroyed_m2_opponent`, `net_space_m2_opponent`.**
  The triplet had been hard-coded `np.nan` on every code path since its introduction
  (3.21.0, PR-S57) — a schema-only dead contract confirmed 100%-NULL across all four
  tracking providers by the lakehouse action-context pipeline (bug report 2026-06-11).
  The TF-41 spec never defined opponent-side semantics: `compute_space_created` is the
  attacking-team leave-one-out differential OBSO (Fernandez & Bornn 2018), and
  `space_creation_xfns` was always deliberately team-side only. An opponent-side metric
  (the actor's leave-one-out effect on a counterfactual opponent-attacking OBSO surface)
  would be a new research feature with its own sign/EPV-mirroring design — not a fill-in
  of these columns. The team triplet (`space_created_m2_team`, `space_destroyed_m2_team`,
  `net_space_m2_team`) is unchanged, byte-identical.
- The contract gate (`tests/tracking/test_space_creation.py`) now asserts the emitted
  columns are exactly the team triplet **and that each populates** (no dead column can
  silently re-enter the contract).

Hyrum note: consumers that mirror the documented column list (lakehouse
`bronze.spadl_action_context`) drop the dead `*_opponent` columns on adoption; no values
change anywhere, so nothing re-materializes. Not a VAEP retrain trigger
(`space_creation_xfns` output is unchanged).

## [4.22.1] — 2026-06-11

### Fixed — lakehouse bug-report 2026-06-11 hardening (ghost-GK clamp, completion-variant alias)

Four small fixes from the lakehouse 4.22.0 production report (items confirmed against source; the
two suspected value bugs — `xt_gk_pev` ≈ 0 and `obso_peak > obso_optimal` — were verified
**by-design** and are documented below rather than changed):

- **Ghost-GK served position clamped to the physical pitch** (`compute_ghost_gk`): garbage input
  (e.g. an upstream mis-flagged `is_goalkeeper`, which can wrong-foot the per-period goal-side flip)
  can push the boosted regressor far outside its trained label domain — a served keeper 5.7 m
  *behind the goal line* is never physically meaningful. Served `ghost_gk_x/y` (goal-relative) are
  now clamped to x ∈ [0, 105], y ∈ [0, 68] with a warning. Clamp target is the **physical pitch,
  not the trained grid domain** — healthy slight extrapolation past the 30 m label filter (sweeper
  rushes) stays **byte-unchanged**, so this only ever fires on corrupt input. The clamp lives at the
  serving seam; `GhostGkModel.predict_mean` keeps its exact-boosted parity contract (ADR-016).
- **`GkCompletionModel.from_variant("gs")` no longer raises `FileNotFoundError`**: variant KEYS
  (the `variant_key_for_provider` vocabulary, where `"gs"` names the GS-construct model) now alias
  onto the bundled weight DIRS (`"gs"` → `"default"`), so the two public APIs compose. Same shared
  cached instance; no behavior change for `compute_xt_gk` (its private resolver already fell back).
- **`tracking.gradientsports.convert_to_frames` `home_team_id` annotation fixed to `int | str`**
  (runtime has been dtype-safe + fail-loud-on-zero-match since 4.15.0/ADR-019; the annotation and
  docstring now say so).
- **`compute_pass_obso` docstring**: `peak_obso` (max over *time* at the fixed target) and
  `optimal_obso` (max over *teammate positions* at the event frame) maximize different axes and are
  **not mutually ordered** — `peak > optimal` is legitimate; both dominate `actual_obso`.

By-design confirmations for the report: `xt_gk_pev = rho × max(0, progress)` is exactly 0 whenever
no opponent is inside the Andrienko pressure oval (~9 m) — structurally true for every goal kick
(law: opponents outside the box) — or the move is non-forward; the emitted `xt_gk_pressure` column
is `rho` and discriminates the two. `LinkReport.per_period_link_rate` (requested as new) has shipped
since 4.12.0/ADR-017.

Hyrum note: the ghost-GK clamp is a serve-output change **only on physically-impossible rows**
(observed: metrica with a corrupted upstream GK flag). No retrain trigger; lakehouse re-materializes
ghost-GK only if it wants the clamped values for already-ingested corrupt matches.

## [4.22.0] — 2026-06-10

### Added — general restart-coordinate enrichment (Phase 1, additive; ADR-025)

New public `silly_kicks.spadl.add_restart_coordinates(actions, *, frames=None, links=None)` imputes
missing coordinates for Law-fixed-spot restart types — goal-kick (6-yard box), penalty (spot), corner
(arc), throw-in (touchline) — and emits them as **new** provenance-tagged columns
(`enriched_start_x/_y`, `enriched_end_x/_y`, `start_coord_source` / `end_coord_source`,
`start_coord_confidence` / `end_coord_confidence`), **never mutating** the canonical
`start_x/start_y/end_x/end_y`. Frames-optional: with `frames` supplied the tracking-ball / in-area
tracking-GK tiers raise confidence; events-only uses native / rule-point / next-event tiers. A
geometry tripwire (à la ADR-018) reverts an imputed origin outside its Law region to
`tripwire_reverted` (warns); native out-of-region coords warn only. Optional aggregate
`silly_kicks.tracking.RestartCoordinateReport` (counts per source + `n_tripwire_reversions`).

This promotes the goal-kick-scoped `resolve_gk_geometry` (ADR-024) into a single general engine
`silly_kicks.tracking.resolve_restart_geometry` (parameterised by `impute_types`); `resolve_gk_geometry`
is now a thin, **byte-identical** shim over it (`impute_types=(goalkick,)`), so xT-GK / completion and
all 4 internal callers are unchanged — **no model retrain**. Scope grounded by a live lakehouse probe:
NaN coordinates are a Gradient Sports set-piece phenomenon (StatsBomb/Wyscout/SkillCorner are 0%), so
the Law-geometry prior is defensible. The canonical-coordinate promotion (which WOULD retrain
VAEP/xT/calibration) is a deferred Phase 2 (separate PR). Additive only — no existing behavior or
output changes.

## [4.21.4] — 2026-06-10

### Changed — xT-GK per-type base-rate serve switch (goal-kick completion honesty)

`compute_xt_gk` now serves the **per-type calibrated base rate** (tagged `xt_gk_completion_source =
"base_rate"`) instead of the geometric model for any completion-variant sub-domain whose held-out AUC
lower-confidence-bound ≤ 0.5 (or degenerate / below a minimum sample) — the gate is a single
`serve_mode_from_lcb(lcb, n)` decision baked into the `GkCompletionModel` artifact (`_type_serve_mode`
\+ `_type_gate_metrics`, version 1.1.0). `load()` **fail-opens**: a pre-gate (4.21.0) artifact serves
all types `"model"` = prior behavior. The switch is **data-driven per variant, not a blanket
goal-kick rule**: the bundled **SkillCorner** gate routes **goal-kicks → `base_rate`** (held-out AUC
0.433, near-chance from tracking geometry) while keeping GK-passes model-scored (AUC 0.737); the
bundled **GS `default`** keeps **goal-kicks `model`-scored** (AUC 0.836, LCB 0.798 — GS goal-kick
completion *is* predictable from geometry). Near-empty throw-in sub-domains (degenerate AUC) base-rate
by construction in both.

Coefficients are **byte-unchanged** — the re-bundle attaches the gate onto the committed model
(corpus-identity-guarded; the guard tolerates the unrecorded-`tracking_limit` density float noise but
aborts on a real retrain). **Not a VAEP retrain** (xt_gk is opt-in, in no default xfn list) — but an
`xt_gk` serve-output change for the flipped types: the lakehouse re-materializes `xt_gk` for the
**SkillCorner goal-kick rows (~15% of its GK-distribution actions) plus degenerate throw-ins (both
variants)**; GS goal-kicks are unaffected. ADR-024 amendment. (4.21.0 §2.3/m3 follow-up.)

## [4.21.3] — 2026-06-09

### Changed — sportec DFL `play_evaluation` success-allowlist (completion robustness)

Native sportec pass/set-piece completion now uses a **success-allowlist** (`fail` iff the DFL
`Evaluation` is non-empty and ∉ `{successfullyCompleted, successful}`) instead of an exact
`== "unsuccessful"` match — so any unseen reason-coded failure token (e.g. `unsuccessfulBecauseOfFoul`)
is failed by construction, and a missing/empty `play_evaluation` still maps to success (no mass-fail
on non-DFL data). Single-sourced across the main and synth-distribution sites (`_extract_play_eval` +
`_play_evaluation_is_fail` + `_warn_unexpected_play_eval`); an unexpected token is warned, not silently
classified. **Aligns the native converter with the kloppy gateway** (same success set) and is
**byte-identical on observed DFL data** — verified on all 7 IDSSE matches, whose only non-success
`play_evaluation` token is `unsuccessful` (robustness hardening, not a re-mapping). Hyrum surface: a
DFL stream carrying failure tokens beyond `unsuccessful` would shift its fail distribution. Adds a
CI-everywhere native-shape distribution regression test and an owner-gated Databricks-bronze e2e over
the 7 IDSSE matches (`fetch_idsse_events`). No shipped-API change. (TODO 4.20.1 follow-up; refines BUG-2.)

## [4.21.2] — 2026-06-09

### Added — owner-gated lakehouse-mart xT held-out-NLL cross-check

A permanent `@pytest.mark.e2e`, owner-gated regression tripwire
(`tests/test_xthreat_nll_lakehouse_e2e.py`) triangulating KDE-vs-Singh held-out transition-NLL on
**passes** against `soccer_analytics.dev_gold.fct_action_values` (the 4.17.0 work ran this as a
non-committed one-off; ~4% relative KDE win on ~8.9M actions). Fits on the full train, scores a
passes-only holdout (parity with the StatsBomb sibling + the published "Held-out NLL (passes)"
3.789→3.748), and **on the full corpus only** hard-asserts at 16×12 that the tuned KDE(4.0) clears a
conservative 1.5% relative-win floor AND the shipped-default KDE(1.0) strictly beats Singh
(no floor — the default's margin erodes with corpus growth); logs 12×8. Adds the
`fetch_action_values` + pure `shape_action_values` mart helpers to `scripts/_loader_databricks.py`
(unit-tested) and pure `nll_relative_win` / `kde_clears_tripwire` verdict helpers (unit-tested).
Skips wherever the owner Databricks credentials + `databricks-sql-connector` are absent (public CI).
**No shipped-library change** — every artifact is in `scripts/` + `tests/`; the `silly_kicks/` wheel
is unchanged except `__version__`. Additive — no behavior change, no retrain trigger. (TODO SK-xT-1
follow-up; ADR-021.)

## [4.21.1] — 2026-06-09

### Changed — ADR-019 AST lint extended to the converter-adapter orientation seam

The boundary lint (`tests/tracking/test_id_compat_lint.py`) no longer blanket-skips the tracking
converter adapters. `ALLOW_MODULES` is narrowed from
`{_id_compat.py, sportec.py, gradientsports.py, kloppy.py}` to **`{_id_compat.py}`** — the helper
module that defines the primitives is now the *sole* exemption, so every tracking module (converter
adapters included) has its id comparisons under the lint. This closes the gap that let **BUG-4** —
the 4.20.1 frame-orientation dtype bug, a raw `team_id == home_team_id` that silently matched zero
players for an int arg vs object-string frames (the `structural_sgm` away-team blow-up root cause) —
reach production: it was a fourth ADR-019 id-dtype instance, and the over-broad file-skip hid it.

- `gradientsports.py` / `sportec.py` `convert_to_frames` already use `ids_match` (the 4.20.1 fix);
  un-skipping them puts the orientation seam under the lint.
- `kloppy.py`'s orientation comparison is routed through `same_id` (it was `str()`-vs-`str()`
  internal — no caller-dtype boundary — so this is **behavior-identical**, chosen for one consistent
  rule and zero per-module exemptions).
- Two guards lock the narrowing: a discriminating proof that the detector actually fires on the BUG-4
  shape (distinguishing a genuinely-clean adapter from a detector that never fires for the shape), and
  an anti-regression assertion pinning `ALLOW_MODULES == {_id_compat.py}`.

ADR-019 amendment. The single library-code change (kloppy's `==` → `same_id`) is behavior-identical
(str-vs-str): no behavior change, no retrain trigger — the BUG-4 *fix* shipped in 4.20.1; this guards
the *class*.

## [4.21.0] — 2026-06-09

### Added — xT-GK (Eyestone): Expected Threat for Goalkeepers (ADR-024)

A new **pure parametric compute feature** (not a trained model) that re-values goalkeeper
distribution actions (goal-kicks, keeper passes/throws), implementing Jeffrey Eyestone's
**xT-GK** (winner, Pitch to the Pros 1) publicly with his attribution. Tracking-required
(the pressure-escape component needs a pressure signal, which no provider preserves through
SPADL). Lives in `silly_kicks/tracking/_xt_gk.py` with the standard ADR-005 surfaces:

- `compute_xt_gk` / `add_xt_gk` (`@nan_safe_enrichment`) / `xt_gk_xfns` (VAEP factory) + atomic mirror.
- `XtGkParams` frozen dataclass + `XtGkParams.for_philosophy(...)` (possession / counter / direct /
  high_press / low_block presets, provisional in-range values).
- Emits raw components `xt_gk_base` / `xt_gk_pev` / `xt_gk_rav` / `xt_gk_dzv` / `xt_gk_pressure`
  plus the composite `xt_gk`, per GK-distribution action.

Design (all confirmed with Jeffrey, 2026-06-08): the destination value is counted **once**
(owned by the risk-adjusted term; the composite base is origin-only — **Option B**); RAV's
pass-completion probability comes from a fitted **`GkCompletionModel`** (see goal-kick coverage
below); the baseline xT grid is a **required caller-injected, pre-fitted `ExpectedThreat`**
(no self-fit, no leakage); the interpretive parameters are intent-set and never calibrated.

In **no** default xfn list — opting xT-GK into a VAEP model is a deliberate, self-triggered
retrain. No change to any existing feature (no retrain trigger). Phase 2 (opt-in team/dataset
parameter estimation) is deferred. Attribution + consent trail in `NOTICE` and ADR-024.

#### Goal-kick coverage — coordinate derivation + RAV completion model (ADR-024 amendment)

The owner-gated OOD smoke escalated: accessible-space's open-play xC resolved for only ~31%
of real goal-kicks (long aerials are out of its validated regime), and ~67% of real GS
goal-kicks carry a NaN origin — together capping real goal-kick coverage at a small fraction.
Both are closed **honestly tagged**, so the composite is defined for ~all in-scope goal-kicks
*with a resolvable destination* and every value carries machine-readable provenance:

- **Coordinate derivation** (`resolve_gk_geometry`, `silly_kicks/tracking/_gk_geometry.py`):
  a **scoped, conditional** origin (native → in-area tracking-GK clamped to `x ≤ 16.5 m` →
  goal-area rule point `(5.5, 34)`) + destination (native → in-period next-event start, guarded
  at `(game_id, period_id)` boundaries) that **feeds the valuation internally and NEVER mutates
  the shared `actions` frame** (a converter-level coordinate change would be a Hyrum/retrain
  trigger for every downstream consumer). Per-row provenance + a continuous confidence are
  emitted: new output columns `xt_gk_origin_source`, `xt_gk_dest_source`,
  `xt_gk_origin_confidence`, plus an optional aggregate `XtGkReport` for pipeline QA.
- **RAV completion model** (`GkCompletionModel`, `silly_kicks/tracking/_gk_completion.py`):
  a **logistic** GK-distribution pass-completion model (sklearn at fit, pure-numpy
  `sigmoid(Xβ)` at serve — **no new runtime dependency**), trained on the observed SPADL
  `result_id == success` label. Bundled GS `default` (30 WC2022 matches, native-origin pooled
  out-of-fold gate: AUC 0.838, CI95 [0.81, 0.86], n_native 1395, Brier 0.122 < base 0.171);
  pickle-free JSON + SHA256 envelope; `from_variant("default")` with a caller `completion=`
  override. Missing-value policy: per-feature density NaN → training-mean impute (neutral after
  standardization); whole-row geometry-unscoreable → per-type base rate (standalone
  `compute_gk_completion` only — the RAV path NaNs unresolvable-destination rows honestly).
- **`[das]` is no longer required** for xT-GK; `compute_xt_gk` / `add_xt_gk` gain a
  `completion: GkCompletionModel | None = None` kwarg. `compute_gk_completion` and
  `add_gk_completion` are exported -- the latter is the lakehouse wide-table aggregator,
  emitting a `gk_completion` column per in-scope GK distribution (NaN out-of-scope) by reusing
  RAV's exact scoring path (geometry on the full action list, then masked), so the column
  equals the P(success) RAV consumes. Train==serve parity is enforced at every producer (shared
  domain predicate, shared geometry resolution on the full action list before masking, shared
  density producer, shared feature extract).

#### SkillCorner completion: native-`result_id` fix + provider-aware variant family (ADR-024 amendment)

Makes SkillCorner `xt_gk` construct-correct and poolable with Gradient Sports.

- **SkillCorner `result_id` → native completion (`silly_kicks/spadl/skillcorner.py`).** The converter
  previously labelled pass/set-piece completion with a `same_team_next` possession proxy, which
  agrees with the native outcome only ~0.72–0.79 and **overstated goal-kick success by ~16 pp**
  (0.86 vs the true 0.70). It now routes `result_id` through the **single native construct** —
  `pass_outcome` (SPADL "reached a teammate") → `received==True` (success-only) → residual
  `same_team_next` — with a new dedicated **`result_source`** column (`native` / `inferred` /
  `stopgap`) recording the per-row label tier. **VAEP-retrain trigger** (SkillCorner scores/concedes
  label distribution shifts; the lakehouse re-materializes SkillCorner VAEP). `received==False` is
  never treated as a failure (it can be a completion to a non-targeted teammate).
- **Provider-aware completion variant** (`GkCompletionModel`): pure `variant_key_for_provider`
  (`skillcorner` → its own weights; everything else → the native-completion `gs` default) + auto-
  selection in `compute_xt_gk`/`add_xt_gk` from `frames["source_provider"]` (caller `completion=`
  override wins; >1 real provider raises; `snapshot` excluded). The GK-completion model trains on the
  **`native` tier only** (`pass_outcome`) — `inferred`/`stopgap` are positive-only / proxy and would
  bias the multiplicatively-consumed calibration.
- **Bundled `skillcorner` variant** (10 SkillCorner matches; GS-transfer re-measured on the corrected
  native label was **0.412** GK-pass AUC, worse than chance → distinct weights required). SkillCorner
  GK-pass **AUC 0.739, ECE 0.036**; goal-kicks are **chance (0.433)** from geometry — model-scored but
  a documented low-discrimination limitation (base-rate-equivalent in practice, on-scale per the
  comparability gate). `from_variant("skillcorner")`.
- **Pooling safety:** new provenance columns `xt_gk_completion_variant` / `xt_gk_completion_source` +
  `XtGkReport.spans_multiple_variants`; a cross-provider comparability gate
  (`scripts/_xtgk_comparability.py`, owner-run) found SC-vs-GS `xt_gk` **within tolerance** on matched
  distance bands → pool directly, no re-scale. The "do not pool across variants without a validated
  comparability" contract is documented (ADR-024).

## [4.20.1] — 2026-06-09

### Fixed — provider data-quality bugs (SkillCorner time-base + goalkick; sportec pass completion; SGM bound + frame-orientation dtype)

Four data-quality defects surfaced while validating GK-distribution completion cross-provider
(corroborated + root-caused with the lakehouse bronze). **The SkillCorner, sportec, and
frame-orientation fixes change VAEP/tracking label/feature distributions for those providers —
retrain triggers.**

- **SkillCorner `time_seconds` is now period-relative (BUG 1, ADR-017).**
  `silly_kicks/spadl/skillcorner.py::_parse_time_start` parsed SkillCorner's `"MM:SS"`
  *continuous broadcast clock* literally, so 2nd-half/ET events landed at ~2700–5800 s while the
  period-relative tracking frames reset to 0 — collapsing action↔frame linkage for the entire
  2nd half + ET (every frame-linked tracking feature silently degraded there). New
  `_to_period_relative` subtracts the period-start offsets `{1:0, 2:2700, 3:5400, 4:6300, 5:7200}`.
  Regression-guarded by a unit test + a strengthened owner-gated e2e (the old check only asserted
  intra-period monotonicity, which a continuous clock also satisfies).
- **SkillCorner goalkick result no longer hard-wired to success (BUG 2).** It was unconditionally
  `success`, bypassing the `same_team_next` possession check used for every other pass; now routed
  through it (lost-to-opponent → `fail`).
- **sportec pass/set-piece completion from native DFL `play_evaluation` (BUG 2).**
  `silly_kicks/spadl/sportec.py` marked *every* pass/cross/freekick/corner/throw-in/goalkick
  `success`, ignoring the `play_evaluation` attribute it already parsed. Now: `unsuccessful` →
  fail; `successfullyCompleted`/`successful`/NULL → success (conservative). Applies to Play,
  set-piece events (which carry it via their nested Play), and the punt-synthesised goalkick
  (inherits its parent Play's evaluation). DFL goalkicks are ~71% complete, not 100%.
- **`metrica.py` left unchanged (measured-correct).** Metrica represents pass loss as a separate
  `BALL LOST` event; a `PASS` is a *completed* pass (98% same-team-next in the fixture, losses
  never attached to a `PASS`), so `result=success` is correct by design.
- **`structural_sgm` numeric blow-up bounded (BUG 3, symptom).**
  `silly_kicks/tracking/_structural_pass.py`: `sgm = 1/rho_r − 1/rho_p` exploded to ~±1e8 when the
  passer/receiver was far from all defenders (the σ=15 "intrinsically bounded, no eps-floor"
  claim was falsified on real byline-cross / fast-break frames). `rho` is now floored at a
  defender's 3σ contribution (`exp(-4.5)≈0.0111`), capping `1/rho`≈90; normal-geometry values are
  unchanged. The falsified docstring is corrected. Defense-in-depth for BUG 4 below.
- **Frame-orientation `home_team_id` dtype bug (BUG 4, ADR-019) — the SGM root cause.**
  `gradientsports.py` and `sportec.py` (tracking adapters) set `team_attacking_direction` via a
  raw `team_id == home_team_id`, which silently matched **zero** players when `home_team_id` was
  passed as `int` and the frame `team_id` was object-string (`"366"`) — every player mislabeled,
  then `play_left_to_right` double-flipped, producing **mis-oriented frames** (the ~4× away-team
  SGM blow-up, and a latent corruption of *every* frame-linked tracking feature whenever the
  caller's `home_team_id` dtype mismatched). Both now use the dtype-safe `_id_compat.ids_match`
  and **fail loud** if `home_team_id` matches no player. Regression-guarded by an int-vs-str
  orientation-invariance test. (The kloppy gateway is unaffected — it derives `home_team_id`
  internally as a string.)

## [4.20.0] — 2026-06-08

### Added — SK-xT-3 calibration-integrated xT bandwidth/resolution sweep (ADR-009, ADR-021)

`silly_kicks.calibration.xt_bandwidth_config` + `XtBandwidthObjective` — a `ruthless`/Optuna sweep
over xT `KDEParams.bandwidth` × `GridSpec` resolution × `adaptive` minimizing K-fold held-out
transition-NLL, with the Singh no-smoothing baseline reported alongside. Recommends a
`KDEParams`+`GridSpec` via an auditable manifest (`scripts/calibrate_xt_bandwidth.py`); **changes no
library default** (ADR-009). The recommendation is scoped to held-out *destination likelihood*
(xT-quality impact reported, not asserted) and a downstream Spearman cross-check vs realised goals
is emitted. The CLI supports download/parse caching for repeated runs: `_loader_pining.load_matches`
gains an opt-in `cache_dir` (persistent, atomic-write artifact cache), and the CLI adds `--cache-dir`,
`--corpus-cache` (assembled-corpus parquet — skips download+parse on re-runs), and `--subsample-games`
(corpus-size contrast off the cache). The corpus is canonicalised to the standard SPADL columns +
string-cast ids so the multi-provider parquet is serialisable.

### Changed — vectorized gaussian xT KDE core (internal; no public-API change)

`kde_smoothed_transition_matrix` now factors a shared, vectorized gaussian seam
(`_gaussian_transition_from_grouped`) — softmax-stabilized, much faster per call, sklearn retained
only for non-gaussian kernels. The gaussian numerics are re-pinned (Chesterton-verified: one caller,
`singh_counts` default) and now stay finite/correct in the small-bandwidth regime where the previous
sklearn-wrapper underflowed to the mean-row fallback.

## [4.19.2] — 2026-06-08

### Changed — CI slow-test gating: invariant heavy tests on a single primary leg (ADR-023)

CI-/test-infra only — **no runtime change** (the wheel is byte-identical; `silly_kicks/` is untouched).
The `test` matrix previously ran the full non-e2e suite on all 4 legs (ubuntu 3.10/3.11/3.12 +
windows 3.12), making the slow Windows runner a ~16–20 min long pole. The expensive
**platform-/interpreter-invariant** tests (train-script smokes, same-run internal-consistency / KDE
parity, calibration cache-equivalence) now carry `@pytest.mark.slow` and run **once on a primary leg**
(`ubuntu-latest` 3.12, identified by a matrix `primary: true` flag); every other leg runs
`-m "not e2e and not slow"`. The `--benchmark-only` step is likewise primary-leg-only.

The `slow` set was chosen from **real Windows-leg CI durations** (local profiling is not a faithful
proxy). **Version-sensitive tests** (golden-hash / snapshot / absolute-numeric) and cheap
behavioral-contract guards (dup-`action_id`, id-dtype-invariance, orientation/roster) are deliberately
**not** marked `slow` — they stay on all legs (OS + interpreter axes). The matrix partition is guarded
structurally by `tests/test_ci_slow_gating_wired.py`; `pyyaml` is now a direct `[test]` dep (the
tripwire's parser). No xdist (it OOM-killed the runners before). Decision: ADR-023.

## [4.19.1] — 2026-06-08

### Added — TF-27 SkillCorner derived-GK Tier-1 roster validation (PR-S86, ADR-007)

Upgrades `_gk_identification.derive_goalkeepers` validation for SkillCorner from Tier-2
(algorithm self-consistency) to **Tier-1** (external ground truth). A new owner-runnable
e2e (`tests/tracking/test_gk_skillcorner_roster_e2e.py`) anchors `derived_gk_picks` against
the pining `match.json` roster GK (`player_role.acronym == "GK"`) per team, with an
exact-set-equality gate (catches over-identification) + a fail-loud join-key guard. Verified
**20/20 team-GKs across all 10 public A-League matches** — no algorithm change required.
A CI-runnable synthetic guard (`tests/tracking/test_gk_skillcorner_roster.py`) shares the same
pure comparator (`tests/_skillcorner_sample.py`), and `scripts/download_skillcorner_sample.py`
populates the sample dir (also unblocks the existing SkillCorner SPADL e2e). Metrica external
verification remains impossible on public anonymized data (no roster) — a documented permanent
limitation (ADR-007).

### Changed

- Refactored `scripts/_loader_pining._build_skillcorner` to delegate frame construction to a
  new `build_skillcorner_frames` seam (single frame path; verbatim relocation, no behaviour
  change — calibration unaffected). Breadcrumb for future calibration work.
- ADR-007 / CLAUDE.md: SkillCorner derived-GK identification recorded as Tier-1 external-roster
  validated.

## [4.19.0] — 2026-06-08

### Added — xT as a VAEP feature (`xt__<method>` xfn factory, ADR-022)

`silly_kicks.vaep.features.xt_xfns(*, model)` (and its atomic mirror
`silly_kicks.atomic.vaep.features.xt_xfns`) wire a fitted `ExpectedThreat` into the VAEP
feature framework as a **frame-free**, opt-in feature transformer. It emits one
`xt__<model.method>` column per gamestate slot (`xt__singh_counts_a0/_a1/_a2`,
`xt__kde_smoothed_*`), following the ADR-005 §8 `<feature>__<method>` naming convention, and
preserves `ExpectedThreat.rate`'s NaN contract for non-move / failed-move actions.

- **Caller-supplies-the-model.** The factory closes over a *fitted* `ExpectedThreat` and fails
  closed otherwise (`None` → `ValueError`, an unfitted model → `NotFittedError`, a `str` →
  `NotImplementedError` — a reserved door for a future bundled-grid variant). Train/serve
  consistency is the caller's responsibility: fit + freeze the grid once and reuse the identical
  object at serve time (mirrors the `FrozenXt` / ADR-009 discipline). `ExpectedThreat` is imported
  only under `TYPE_CHECKING` (duck-typed at runtime) — **no new runtime dependency edge**; bare
  `import silly_kicks` is unaffected.
- **Opt-in — no forced retrain.** `xt_xfns` is in **none** of the default/union xfn lists; opt in
  with `VAEP(xfns=xfns_default + xt_xfns(model=frozen_xt))`. A guard test enforces its absence from
  the defaults. **Opting it into your own xfns is a self-triggered VAEP retrain.**
- **Atomic mirror reuses `model.rate()`** (unchanged) via a synthesized standard-SPADL frame with a
  **type-aware** `result_id` — dribbles are intrinsically successful (never followed by a
  `receival`); pass/cross success iff the next atom is `receival`. A blanket next-atom test would
  NaN every dribble; the type-aware predicate keeps `xt__<method>` column-symmetric across both
  SPADL flavours (verified by a geometry-keyed cross-representation oracle on the committed WC2018
  fixture, plus a dribble keystone gate). Slots map by the composite
  `(game_id, period_id, action_id)` key. A pass/cross that is the last action of a period has no
  following atom and yields NaN (inherent atomic-representation edge; documented).

`ExpectedThreat.rate()` is left **byte-identical** (no `_rate_cells` extraction) — the SK-xT-1
parity gate and golden snapshots are untouched. Decision: ADR-022; attribution Singh (2018).

## [4.18.0] — 2026-06-07

### Added — TF-17 xCrossAttempt (xCross) TRAINED weights + GK validation + TF-19 wiring (PR-S85)

The weights follow-up to PR-A's untrained code (4.11.0). Bundled the **`public`** xCrossAttempt
model (skillcorner + idsse), trained on the clean-4.13.0-GS pining corpus (81 matches, 701,210
wide-area frames / 11,930 cross-positives) against the 4.7.0 carrier defaults, on DGX Spark.
A pre-registered `public`-vs-`full` two-candidate paired test (common public held-out, shared
params) found owner-tier Gradient Sports data **degraded** public generalization in **all 5 folds**
(Δ PR-AUC −0.009…−0.067) → shipped the reproducible public-only model (no Hub repo, mirrors xS).
public CV: PR-AUC 0.0606 > base 0.0177; Brier 0.0172 < 0.0173; log-loss 0.0841 < ln2.
`from_variant("default")` + `from_hub` live; `xcross_attempt_xfns` wired into
`pre_shot_gk_full_default_xfns` (+ atomic mirror) **only**, not the general default.

### Validation (reported in the bundled metrics.json; the GK-extension headline)

- **`tf19_ready = False`** (pre-registered inert-GK contingency): the GK substitution-sensitivity
  probe moves P(cross) by a median **0.00107** on a realistic GK shift — **2.6× the nearest-defender
  control** (0.00041) and ∞× the random-outfielder band (0.0), i.e. GK position carries *relative*
  signal, **but below the pre-registered absolute floor (0.01)** — too small to drive a meaningful
  TF-19 `Δ_cross`. The surface ships regardless (a weak signal is not a build break); TF-19 (GKDV
  Layer 3) consumption is gated on GK feature-engineering first, never shipped silently as novelty.
- GK-block ablation: Δ PR-AUC +0.0011 (≈0 marginal CV lift) — yet `gk_theta` is the #4 feature by
  permutation importance (0.0125): informative-but-collinear (the gate is the probe, not ablation).
- **`score_differential` is the #2 feature by CV-held-out permutation importance (0.0216)** at 1.0
  coverage — *material* for xCross (unlike Ghost-GK). Measured on the clean GS stream: range
  [−5, +6], 0 impossible values (the old ±18 cache would have corrupted this — the clean rebuild was
  load-bearing).

### Added — TF-17 xCross causal validation harness (PR-C, ADR-015)

The paper-faithful causal arm closing TF-17. Private `silly_kicks/_causal/` port (pure numpy/sklearn,
no R, no new dependency): propensity-score matching (ATT/ATNT, 1:1 nearest-neighbor **with
replacement**, no caliper, logistic propensity on standardized covariates, **Abadie–Imbens (2006)
matching SEs**) + a spell-based crosser-anchored opportunity builder. `scripts/validate_xcross_causal.py`
ablates the GK confounder block against a **row-permuted-GK placebo null band**, with a positivity
guard, a PS-overlap + SMD-improvement claim gate, and a GK missing-indicator. The treatment window is
`(entry, min(entry+T, spell_end)]` (fixed-`T` cap → no spell-length confounding; `spell_end` clamp →
no cross-phase misattribution); the outcome is measured strictly post-treatment. The causal finding is
a **reported** research artifact (`docs/research/xcross_causal/`), never a ship/CI gate — only the
known-truth method tests (`tests/causal/`) gate CI. Reconstructs the paper's sender-level unit;
tracking-only-opportunity-detection + league/era divergence reported, not hidden. Decision: ADR-015;
attribution arXiv:2505.11841.

### Causal result (reported in `docs/research/xcross_causal/`; clean all-provider corpus)

Run on the full 3-provider pining corpus (skillcorner + idsse + gradientsports), seed 0:
**23,966 opportunities / 669 treated (base outcome rate 4.3%)**.

- **The cross effect is real and significant.** ATT (with GK block) **+0.0927 (SE 0.0156)**; ATT
  without the GK block +0.0747 (SE 0.0167); ATNT +0.0551 (SE 0.0133) — ≈5σ. Crossing causally raises
  the ~6-second scoring-opportunity outcome by **+7–9 percentage points** over the 4.3% base.
- **The matching is valid:** propensity overlap 1.0 (no density trimming), max SMD 0.51 → **0.078**
  post-match (< 0.1) → `causal_claim_supported = True`.
- **The novel GK-position block does NOT clear the placebo band** (`gk_clears_placebo_band = False`,
  **reported, not a gate**): adding the GK block shifts the ATT by **0.0179**, below the
  row-permuted-GK placebo p95 of **0.0239** — i.e. not distinguishable from a shuffled-GK column on
  this corpus. **This independently corroborates `tf19_ready = False`:** two methods (the PR-B
  predictive substitution probe and this PR-C causal placebo ablation) now agree the GK block carries
  relative-but-not-distinguishable signal → TF-19 stays gated.
- **The GS feature fix was load-bearing.** With the `canonical_id` fix, GK/base NaN fractions are
  ~0 (8.3e-5 / 0.0) and all three providers reach carrier-coverage 1.0 (GS contributes 19,833 of the
  23,966 opportunities). An earlier run on the un-fixed extractor was a **false positive** —
  `gk_clears_placebo_band = True` driven entirely by an 82.8%-NaN GS missingness confound; the fix
  flipped it to the correct negative. (See the GS bug entry below.)

### Fixed — GradientSports xCross feature extraction returned all-NaN (silent)

`extract_xcross_features` matched the ball-carrier / goalkeeper by stringifying the frame's
`player_id` / `team_id` via `.to_numpy().astype(str)`. GradientSports tracking frames carry
**nullable `Int64`** ids, and `Int64.to_numpy()` **upcasts to float64** → `"11094.0"`, which never
equals the clean-int carrier key `"1336"` → the carrier mask matched 0 rows → **every
carrier-anchored confounder and the entire GK block came back NaN for all GradientSports frames**
(≈83% of the real corpus, and the whole shipped GS xCross-inference path). Numeric team comparisons
survived (`366.0 == 366`), so only the string player match broke; kloppy/string-id providers were
unaffected, which is why it stayed latent. Fixed by routing the id match through the ADR-019
`_id_compat.canonical_id` / `canonical_id_series` contract (collapses `366` / `366.0` / `Int64(366)`
/ `"366"` → `"366"`). The existing tests only asserted column *existence*; added
`test_int64_id_frames_resolve_carrier_and_gk_features` to assert feature *values* resolve (notna) on
Int64-id frames. The shipped public model trains on kloppy/string providers so its weights are
unaffected (no retrain); the fix repairs GS xCross *inference* (was silently NaN → xgboost-missing).

- `prepare_xcross_training_data` raised `TypeError: boolean value of NA is ambiguous` on real frames
  whose `team_id` column carries `pd.NA` (ball row / unresolved GS jersey); the defending-team
  computation now filters by `is_ball` + `dropna()` (mirrors `compute_xcross_attempt`). Surfaced by
  the maintainer-run training pilot.

### Note

- A future TF-24 carrier-default change is an xCross retrain trigger (carrier params recorded in
  metadata + consumed at inference).

## [4.17.0] — 2026-06-07

### Added — SK-xT-1: pluggable, evaluatable xT (`silly_kicks.xthreat`)

`silly_kicks/xthreat.py` is now the `silly_kicks/xthreat/` package with a pluggable transition
family in silly-kicks house style (string-dispatch + frozen-dataclass params, no ABCs; ADR-021):

- **`ExpectedThreat(method="singh_counts" | "kde_smoothed", params=..., l=, w=)`** — the
  `singh_counts` default is **byte-identical** to the prior implementation (proven by an
  in-process frozen-oracle parity gate over the WC2018 fixture + `spadl_actions`). KDE-smoothed
  transitions (`kde_smoothed_transition_matrix`, Silverman-1986 bandwidth, optional adaptive
  per-source-zone) are a new flavor; `KDEParams.bandwidth` defaults to 1.0 (pure Silverman — a
  conservative, corpus-agnostic baseline). KDE strictly beats Singh at every scale tested; the
  held-out-NLL-optimal multiplier is corpus-size-dependent (~1 on a 64-match sample, ≥4 on an
  8.9M-action mart) — tune via `compute_holdout_nll`. `singh_transition_matrix` is vectorized
  (`np.add.at`), byte-identical to the legacy per-zone loop (exact-equality parity gate).
- **`GridSpec`** — first-class variable resolution (pitch dims stay in `spadlconfig`; SSOT).
- **Standalone `value_iteration`** (extracted byte-identically from the legacy solver; optional
  `max_iter` guard, default unbounded) + **`singh_transition_matrix`** / `silverman_2d`.
- **Held-out transition-model NLL evaluator** — `holdout_split` (`game_id`-keyed),
  `compute_holdout_nll` (pure: matrix + holdout + grid), `compute_holdout_nll_per_group`. The
  first held-out xT evaluation primitive in silly-kicks. (NOT an xT-quality metric — it scores
  destination likelihood under the transition matrix.)

KNN/conditional xT (pre-publication; tracking-join-dependent) is deferred. The lakehouse `XTGrid`
typed wrapper is NOT adopted (xthreat keeps its raw `.xT` ndarray). **Additive — no behavior
change on the default Singh path, so no retrain trigger for existing consumers** (incl. the TF-24
calibration `FrozenXt`). Promotion proposed by the luxury-lakehouse session; attribution:
Singh (2018), Silverman (1986), Salimi et al. (2026, LISS poster, pre-publication). Decision: ADR-021.

## [4.16.1] — 2026-06-07

### Fixed — Sportec/DFL converter mislabelled ~99% of passes as crosses

`convert_to_actions` flagged a pass as a cross via `_opt("play_flat_cross", False).fillna(False).astype(bool)`.
DFL bronze emits the `play_flat_cross` qualifier as the native string `"true"`/`"false"`, and
`pd.Series(["false"]).astype(bool)` is `True` (any non-empty string is truthy) — so every pass whose
`play_flat_cross` was non-null, **including the literal `"false"`, became a cross**. On real Bundesliga
data this inverted the pass/cross split (e.g. match J03WMX: 875 cross / 7 pass, where cross should be
~2–4% of passes). It was the only `.astype(bool)` on a string qualifier in `sportec.py`; the sibling
qualifiers (`shot_after_free_kick`, the two `*_defensive_clearance`) already parse the string correctly.

Fixed by parsing the string explicitly:
`_opt("play_flat_cross", "").fillna("").astype(str).str.lower().eq("true")`, matching the in-file sibling
convention. Because `str(True).lower() == "true"`, this also handles a native-bool column correctly, so the
existing bool-flag behaviour is preserved. **Hyrum:** Sportec/IDSSE (DFL/Bundesliga) pass-vs-cross labels
change for all event data — a SPADL re-conversion + downstream VAEP retrain trigger for Sportec consumers.

## [4.16.0] — 2026-06-07

### Added — TF-45 structural-pass primitives (LBS / SGM / SDI)

Per-pass structural primitives quantifying how a pass deforms the opponent's defensive structure
(Karakuş & Arkadaş 2026, arXiv:2603.28916): **Line Bypass Score** (`structural_lbs`), **Space Gain
Metric** (`structural_sgm`), **Structural Disruption Index** (`structural_sdi`). New module
`silly_kicks/tracking/_structural_pass.py`: a pure pandas-free core `_structural_pass_core`, the
per-frame `compute_structural_pass_metrics`, the `@nan_safe_enrichment add_structural_pass`
aggregator, and the `structural_pass_xfns` VAEP factory (both via the shared
`_kernels._structural_pass_at_actions` batch kernel — 3×-not-9× call-count budget). Atomic mirror
synthesizes `end = x+dx`. `StructuralPassParams.sigma = 15.0` is empirically tuned on 2,466 real
WC2022 passes (smallest σ at which the inverse-density SGM is intrinsically pitch-bounded; see
`scripts/tune_structural_pass_sigma.py`). **Library ships RAW primitives only** — the TIV z-norm
composite, K-means archetypes, and passer/receiver rankings are corpus-level (consumer-side). Decision:
ADR-005. Owner-gated e2e validates against real WC2022 Gradient Sports tracking.

### Fixed — Systemic dup-`action_id` crash across frame-aware xfns (ADR-020)

The per-slot `pointers.set_index("action_id").at[aid, "frame_id"]` pattern crashed when a `*_xfns`
factory was composed into a VAEP model: shifted gamestate slots repeat the period-boundary action, so
`action_id` is non-unique and `.at` returns a Series (`ValueError: truth value of a Series is
ambiguous`), and provenance merges fan out (`Length mismatch`). Empirically confirmed across **8
families**: `pitch_control`, `obso`, `pausa`, `space_creation`, `pressure`, `cover_shadow`,
`gk_influence`, `player_influence`. Fixed via a shared `_kernels.resolve_frame_ids_by_position`
(positional, dup-safe), a red-first behavioral gate that auto-enumerates every `*_xfns`
(`tests/tracking/test_frame_aware_xfns_dup_action_id.py`), and a per-family retrofit. **Behavior change
(Hyrum):** these `*_xfns` previously raised in the gamestate path and now produce values — a VAEP
feature-matrix change / retrain trigger for any consumer using the xfns path. (The production/lakehouse
`add_*` aggregator path on full action streams was unaffected — unique `action_id`.) Decision: ADR-020.

### Fixed — Ghost-GK public-API export gap

`silly_kicks.tracking` now exports the full ghost-GK feature surface — `add_ghost_gk`, `ghost_gk_xfns`,
`compute_ghost_gk`, `GhostGkModel`, `GhostGkDensity` — from the package root (previously reachable only
via the `silly_kicks.tracking.features` / `._ghost_gk` submodules). `add_ghost_gk` was the only feature
`add_*` aggregator missing from `tracking.__all__`; this aligns ghost-GK with every other tracking
feature (e.g. space-creation, xS, xCross) and corrects the C4 action-coupled-aggregator count.

## [4.15.0] — 2026-06-06

### Added — Dtype-safe id contract at tracking-feature seams (ADR-019)

Tracking-feature consumers compared SPADL-action identifiers against tracking-frame identifiers (and
the scalar `home_team_id` argument), and merged action↔frame frames on id-valued keys, with raw
`==`/`!=`. These silently mis-resolve when the two sides have different dtypes (`Int64(366) == "366"`
→ `False`), or **raise** on a mixed-dtype merge key — so any caller whose id dtype differs from the
library's (e.g. the lakehouse, which persists frame ids as **string** while actions stay **bigint**)
got silently-wrong actor / opponent / defending-GK / defensive-line / possession / attacking-team
resolution. ADR-019 introduces a **dtype-safe id contract** at the consumer seams:

- **New `silly_kicks.tracking._id_compat`** — one definition of "id identity": a single `_canonical`
  truth (scalar `canonical_id` + vectorized `canonical_id_series`, integral-float collapse so
  `366`/`366.0`/`Int64(366)`/`"366"` → `"366"`; genuine strings pass through), comparison helpers
  (`ids_equal`/`ids_differ`/`ids_match`/`same_id`, NA-safe, non-nullable `np.bool_`), and a pre-merge
  `align_join_keys` (numeric-vs-object only; numeric-vs-numeric and object-vs-object merge fine). A
  same-kind/both-object fast path means **zero overhead** for matched-dtype pipelines and
  genuine-string providers (sportec/kloppy).
- **New public `validate_id_dtypes(actions, frames, *, home_team_id=None, on_mismatch="raise")`** +
  `IdDtypeDiagnosis` (exported from `silly_kicks.tracking`) — an opt-in loud pre-flight guard mirroring
  ADR-017's `validate_time_base`. Not threaded through the ~30 aggregators; the seam coercion already
  makes them correct.
- The seams are fixed comprehensively (every registered `add_*` aggregator) and guarded by a red-first
  **asymmetric** dtype-invariance gate (numeric actions × string frames, and the reverse, with
  `home_team_id` an independent axis) + a meta-assertion (gate surface == registered surface) + a
  boundary-focused AST lint + a structural de-dup perf guard.

### Fixed — three latent correctness bugs the contract exposed (Hyrum: feature values change)

The contract corrects pre-existing silently-wrong behavior, so some feature values change for
**numeric (pure-library) callers too**, not only string-id callers. **VAEP models consuming these
features should be re-fit.**

- **`_resolve` opponent mask counted the ball as an opponent** for object-`is_ball` providers
  (kloppy/sportec/metrica/skillcorner): the old `~long["is_ball"]` on an **object**-dtype bool column
  is a no-op (`~True → -2`, truthy), so the ball leaked into opponents. Fixed via `.astype(bool)` +
  `ids_differ`'s both-present rule. Affects any opponent-aggregating feature; notably it inflated
  `bekkers_pi` pressure (the ball was a phantom presser). The
  `test_per_method_cross_provider_median_within_2x` calibration drops `bekkers_pi` (a kinematic model,
  not geometry-comparable across providers; its prior agreement was the ball artifact).
- **`add_player_influence` / `add_cover_shadows` team/opponent mislabel:** `str(action_team) ==
  str(frame_team)` broke because `DataFrame.iterrows()` upcasts a numpy-`int64` action `team_id` to
  `float64` (`str(5.0) != "5"`) while the nullable `Int64` frame side stays `"5"`. Fixed via `same_id`.
- **Object-path opponent join-miss:** an unmatched `how="left"` row satisfied raw `NaN != "5"` → True
  → wrongly "opponent". `ids_differ`'s both-present rule excludes it (the numeric path already did).

### Lakehouse handshake

The lakehouse may drop its string-coercion workaround and rely on the seam coercion, or call
`validate_id_dtypes(..., on_mismatch="raise")` at work-unit entry. ADR-001 (converter identifier
conventions) is preserved — the fix lives entirely at the consumer seams. No new **runtime**
dependencies; `import silly_kicks` stays dependency-light.

### Internal — deterministic perf guards + CI runtime

Library runtime is unaffected (test-infra only). The wall-clock perf budgets (`assert mean_ms <
budget`) flaked on shared CI runners (`compute_team_shape` 6.2ms > 5ms, `compute_gk_influence`
10.4ms > 10ms) — a recurring red-CI source. Every such budget is replaced with a **deterministic
structural guard** that asserts the invariant the budget actually protected, via a call-count spy on
the dominant primitive (`tests/_perf_structural.py`):

- pitch-control consumers (`compute_player_influence` / `compute_gk_influence`) build the per-frame
  surface ONCE (the ADR-008 cache contract), not per player/zone;
- the Ward line decompositions (`compute_team_shape` / `detect_line_breaking`) cluster once per
  frame, not per player/segment;
- `pressure_on_actor` (×3) / `add_actor_pre_window` link actions→frames ONCE per batch, not per
  action;
- the pitch-control kernels (Spearman / Fernandez-Bornn) run one vectorised grid pass per team, not
  per cell;
- the SPADL/atomic throughput converters stay vectorised (zero `apply(axis=1)`/`iterrows`/`itertuples`).

The benchmark *measurements* are retained (no hard timing asserts) and run single-threaded for clean
trend data. The dominant ghost-GK cost is cut at the source: the golden gates run the exact
`cpu-numba` KDE backend (matches `vectorized`/scipy at 1e-9 on the kernel; ~7.8× faster) and the
bundled-model golden slices to 4 frozen samples (`vectorized` ↔ scipy parity stays locked by the
kernel + model-traveling tests). A `pytest-xdist` parallelization was evaluated and reverted — on the
4-core/7GB CI runners it regressed py3.12 from pass to a memory/JIT-pressure kill (the opposite of the
16-core local speedup); the bulk suite stays serial.

## [4.14.0] — 2026-06-06

### Changed — Ghost-GK serves the exact boosted HGBR mean (integrity fix), pickle-free (ADR-016, PR-S83)

`compute_ghost_gk` served the KDE **mode** (~4.65 m held-out MAE) while the model card reported ~1.1 m
for the sklearn `predict_mean` that `save()`/`load()` discarded (it raised after `load()` — **never
served**). 4.14.0 closes the gap: `predict_mean` / `predict()` / `compute_ghost_gk` now serve the
**exact sklearn `HistGradientBoostingRegressor` boosted prediction** — held-out euclidean MAE **1.07 m**
(5-fold, vs the served mode's 4.65 m) — reconstructed **pickle-free** from serialized tree node arrays +
baselines (`baseline + Σ_trees leaf_value`; new `_vectorized_leaf_values` kernel, an
independent-parity-tested sibling of the KDE traversal). Inference stays sklearn-free + numpy-only +
deterministic, and is **sklearn-version-independent** (sklearn couples only at fit/extract).

An earlier attempt to serve the leaf-weighted **conditional mean** (no re-publish) was built and
**empirically rejected** — it measured 7.0 m, *worse* than the 4.65 m mode (the conditional GK-position
density is broad + multimodal, so central tendencies sit in low-density valleys). The boosted mean is a
structurally stronger estimator (squared-error boosting on the full 26-feature interaction). See ADR-016
for the rejection table + the stratified ship gate.

- **`fit()` trains `phase` numerically** (`categorical_features=None`) — removes 24 categorical split
  nodes whose routing bitsets aren't serialized, making the numeric reconstruction match sklearn exactly
  **and** closing a latent KDE categorical-routing capability gap. The density/spread shifts slightly as
  a result (expected; the served value is now the boosted mean, not the mode).
- **Artifact format change (version 1.2.0):** the npz now carries the gk_y tree ensemble + both
  baselines; `metadata.serve_estimator = "boosted_mean"`. **Both bundled `default` (wheel) and Hub
  `full` weights are re-fit + re-published.** `load()` **fails closed** on a conflicting `serve_estimator`
  (R3) and on pre-Option-A artifacts (missing gk_y trees → clear "re-fit required" error).

> **BREAKING — column rename:** the emitted spread column `ghost_gk_spread` is renamed
> **`ghost_gk_density_spread`** (in `compute_ghost_gk`, `add_ghost_gk`, `ghost_gk_xfns`, and the atomic
> mirror). The served position is now the boosted mean while the spread is the conditional-**density**
> dispersion (a different read-out — NOT the standard error of the served point); the rename makes that
> structural. **Lakehouse consumers must rename the column on consume and re-materialize `ghost_gk_*`.**

> **Hyrum's Law / behavior change:** every served `ghost_gk_x/y` value changes (deliberate value change,
> not an API break); `model.predict()` is a public-API **semantic** change (returns the boosted mean, not
> the KDE mode — the mode remains reachable via `predict_density(...).mode_x/mode_y`); old-format weights
> no longer load (re-fit required). The lakehouse must re-materialize the ghost-GK columns.

## [4.13.0] — 2026-06-04

### Added / Fixed — Gradient Sports goal-capture correctness + VAEP own-goal labeling (ADR-018)

Completes the Gradient Sports / PFF FC goal-capture work begun in 4.12.2 (which removed the false
`shot_outcome_type == "O" → owngoal` mapping). Empirically grounded in the full WC2022 catalog (64
matches, 144,541 events).

- **Own goals captured (`silly_kicks.spadl.gradientsports`).** `possession_event_type == "RE"` (rebound)
  with `shot_outcome_type == "G"` is an own goal → `bad_touch` + `owngoal`, attributed to the conceding
  team and the rebounder/scorer (`gameEvents.playerId`), per the StatsBomb/opta/sportec precedent. A
  post-LTR **geometry tripwire** validates each own goal sits in the conceding team's own half
  (`start_x < field_length/2`); a row failing it emits a `UserWarning` and reverts to `keeper_save`/`fail`
  (guards the n=3 rule against rebound-goals/feed anomalies). The 3 real WC2022 own goals
  (Enzo Fernández, Aguerd, Neuer) are captured correctly (owner-gated e2e).
- **Cross-goals captured.** `possession_event_type == "CR"` with `shot_outcome_type == "G"` keeps the
  cross/`freekick_crossed` action and **synthesizes a `shot`/`shot_freekick` + `success`** by the crosser
  (foul-synthesis pattern), so a direct cross-goal registers as a goal (SPADL records goals only as
  shots).
- **Synthesized-row provenance.** A new `is_synthetic` (bool) column on `GRADIENTSPORTS_SPADL_COLUMNS` is
  `True` on converter-injected rows (the cross-goal shot **and** the synthesized foul rows, which share
  their parent's `original_event_id`) and `False` on real 1:1 rows — so a consumer de-duping on
  `original_event_id` can keep the synthesized row instead of silently collapsing/dropping it.
- **Voided events excluded.** `possessionEvents.nonEvent == True` (annulled plays — fouls/advantage called
  back, offside, disallowed goals; 1081 across WC2022, incl. 21 disallowed goals) are now dropped in the
  exclusion stage with a `ConversionReport.excluded_counts["nonEvent"]` tally. The `nonEvent` input column
  is **optional**: absent → an observable no-op (one-time `UserWarning` + the report key omitted, so
  "not checked" ≠ "0 voided"), so existing callers keep working but get a loud nudge to supply it.
- **Own goals counted in VAEP labels (all providers) — ADR-018.** `vaep/labels.py` now detects own goals
  by **result** (`result_id == owngoal`) via a single-source `_is_owngoal` helper, dropping the
  `type_name.str.contains("shot")` gate that silently zeroed out every provider's own goals (they are all
  `bad_touch`). Goal detection uses a sibling `_is_goal` with an explicit `{shot, shot_penalty,
  shot_freekick}` name-set. A guard test forbids the old shot-gated owngoal pattern from reappearing.

> **Hyrum's Law / behavior change:** (1) VAEP `scores`/`concedes`/xG label distributions shift for every
> provider whose data contains own goals (~3–5% of goals previously invisible now count) — VAEP models
> retrained on these labels will shift. (2) Gradient Sports action counts change: voided events dropped,
> own goals now `bad_touch`+`owngoal`, cross-goals gain a synthetic shot row (flagged `is_synthetic=True`,
> sharing the cross's `original_event_id`), and the GS output gains the `is_synthetic` column. (3) The
> `nonEvent` soft input-contract: GS callers must surface `possessionEvents.nonEvent` to exclude voided
> events, else the warning fires. The atomic-SPADL surface inherits all converter changes; the atomic
> label path already counted own goals.

## [4.12.2] — 2026-06-04

### Fixed — Gradient Sports / PFF FC shot `shot_outcome_type == "O"` mis-mapped to `owngoal`

The Gradient Sports converter mapped shot `shot_outcome_type == "O"` to the SPADL `owngoal` result.
`"O"` is in fact the **off-target** shot bucket (alongside `S`=saved, `B`=blocked); the four main
shot outcomes are `G`=goal / `S`=saved / `O`=off-target / `B`=blocked, and only `G` is a success.
The mapping was an unsourced assumption inherited from the original PFF FC converter (2.6.0), never
checked against the PFF FC codebook.

Verified against the full PFF FC / Gradient Sports WC2022 feed (all 64 matches): `"G"` counts
reproduce every final scoreline **and** the exact penalty-shootout arithmetic (e.g. ARG–FRA final
3–3, pens 4–2 → G=12 = 6 regulation/ET + 6 shootout), confirming own goals already surface under
`"G"`; meanwhile MAR–ESP finished **0–0** yet carries `O=10`, and `"O"` recurs 4–17× in *every*
match — impossible for own goals.

- `silly_kicks.spadl.gradientsports`: dropped the `shot_outcome_type == "O" → owngoal` branch. `"O"`
  (and every non-`"G"` shot outcome) now falls through to `fail`, like `S`/`B`.
- The converter now maps **no** shot outcome to `owngoal`. Own goals are encoded as `"G"` and
  `shot_outcome_type` alone cannot distinguish them, so correct own-goal attribution remains an open
  item pending the PFF FC codebook.

> **Hyrum's Law / behavior change:** SPADL stores built from this converter previously contained
> phantom `owngoal` results (~563 across the 64 WC2022 GS matches, up to 17/match); these are now
> `fail`. Consumers that counted or filtered on `owngoal` from Gradient Sports data will see those
> rows reclassified — lakehouse SPADL stores should be re-baselined. The atomic-SPADL surface
> inherits the change via the shared converter.

## [4.12.1] — 2026-06-04

### Fixed — `compute_ghost_gk` crash when a team has ≥2 goalkeepers in one frame

`compute_ghost_gk` (hence `add_ghost_gk` / `ghost_gk_xfns`, and their atomic mirror) raised
`ValueError: Must have equal len keys and value when setting with an iterable` for any frame
containing two or more `is_goalkeeper=True` rows with the same `team_id`. Reported by
luxury-lakehouse: a provider match rostered a backup keeper carried on-pitch alongside the starter
in 100% of frames, so the very first batch of each half crashed and the match produced zero output.
GK-substitution overlap frames trigger the same fault intermittently.

- Root cause: `_extract_all_ghost_gk_features` emits one inference sample **per GK row**, all keyed on
  `(game_id, period_id, frame_id, gk_team_id)`. A second same-team GK in a frame produced duplicate
  keys; the downstream `how="left"` merge onto the GK rows then inflated past `gk_mask.sum()` and the
  positional assignment back into `out.loc[gk_mask, ...]` length-mismatched.
- Fix: `compute_ghost_gk` now collapses duplicate `(frame, gk_team)` inference samples (keeping the
  first) **before** `predict_density`. The features are byte-identical per `(frame, gk_team)` — only
  the per-GK-row label differs, and labels are unused at inference — so both GK rows receive the same
  ghost-GK prediction, and the KDE runs once per `(frame, gk_team)` rather than once per GK row. The
  training-data builder (`prepare_ghost_gk_training_data`) keeps its per-GK-row sampling (distinct
  labels) untouched. Single-GK frames are unaffected (the de-dup is a no-op) — the frame-restriction
  byte-identical golden still holds.

## [4.12.0] — 2026-06-04

### Added — period-relative `time_seconds` contract + loud per-period link-coverage guard (ADR-017)

Documents and enforces silly-kicks' canonical **period-relative** `time_seconds` convention
(seconds since the start of each period, resetting to 0 — NOT absolute match-clock), and makes a
low action↔frame link-coverage outcome loud. Resolves the GradientSports period-2 silent-data-loss
class reported by luxury-lakehouse (a period-relative-vs-absolute time-base mismatch dropped ~81% of
GS period-2 actions with no signal).

- `silly_kicks.tracking.utils.link_actions_to_frames` gains `min_link_rate: float = 0.5` and
  `on_low_coverage: Literal["warn", "raise", "ignore"] = "warn"`. The guard is evaluated **per
  period** (worst period), never the match aggregate — a match-aggregate floor would launder a
  catastrophically-unlinked period behind a healthy one. The warning/error message carries the
  per-period rate, unlinked count, and — when a period's action/frame ranges are near-disjoint — a
  suspected time-base-mismatch hint.
- `LinkReport` gains `per_period_link_rate: dict[int, float]`, computed from the internal per-period
  merge (not the returned pointers, which drop `period_id`).
- New public `silly_kicks.tracking.validate_time_base(actions, frames, *, on_mismatch="raise")` +
  `TimeBaseDiagnosis` — the primary guard for consumers that pre-filter / window / batch actions by
  time before linking (the linker guard cannot see actions a pre-filter already dropped). Call it on
  the **unfiltered** inputs at work-unit entry.
- `MISMATCH_OVERLAP_FLOOR = 0.2` time-base-mismatch diagnostic, decoupled from `min_link_rate` (the
  *cause hypothesis* vs the *symptom*).
- The period-relative convention is documented on the tracking + events converter docstrings,
  `link_actions_to_frames` / `slice_around_event`, and the SPADL + tracking schemas, and pinned by
  convention lock tests for the converters whose `time_seconds` arithmetic the library owns (Opta,
  StatsBomb). GradientSports `time_seconds` is a verbatim pass-through originating upstream and is
  guarded lakehouse-side.

> **Hyrum's Law / behavior change:** `link_actions_to_frames` now emits a `UserWarning` by default
> on low per-period coverage. Consumers running `-W error` / `filterwarnings=error` will start
> failing on genuinely-degraded matches — the intended shift-left. Pass `on_low_coverage="ignore"`
> for a known-partial match, or `"raise"` to escalate. The atomic-SPADL surface inherits the change
> via the shared linker.

## [4.11.0] — 2026-06-03

### Added — xCrossAttempt (xCross) cross-attempt-propensity model (TF-17, GKDV Layer 2)

A per-frame, STATE-anchored surface — `P(the in-possession team attempts a cross within ~1 s of a
frame)` — the cross analogue of xShotOccurrence (TF-16) and the next decision-probability surface in
the GKDV program. Inspired by Cao et al. (2025, arXiv:2505.11841); realizes 7 of the paper's 8
confounders (crosser position #7 omitted — no faithful tracking-only proxy) and **extends the
propensity model with a novel goalkeeper-position confounder block** (the paper's confounder set
excluded all GK variables).

- New `silly_kicks.tracking._xcross_attempt`: `extract_xcross_features`, `build_xcross_labels`,
  `prepare_xcross_training_data`, `XCrossAttemptModel` (pinned-deterministic XGBoost; pickle-free
  booster-JSON + metadata + SHA256SUMS), and the ADR-005 surfaces `compute_xcross_attempt` /
  `add_xcross_attempt` / `xcross_attempt_xfns` (+ atomic mirror).
- Shared `silly_kicks.tracking._occurrence_labels._build_occurrence_labels`, extracted from
  `build_xshot_labels` (now a thin, bit-identical wrapper — xS labels unchanged).
- HPO objective (`_xcross_attempt_objective`) + training CLI (`scripts/train_xcross_attempt.py`),
  gated behind the existing `[train]` extra; inference gates on `[xgboost]` (lazy — `import
  silly_kicks` stays dependency-light).
- **Ships UNTRAINED** (code + synthetic CI fixture + real-provider extraction tests):
  `from_variant`/`from_hub` raise `FileNotFoundError` until the weights follow-up (PR-B), and
  `xcross_attempt_xfns` is NOT wired into any default xfn list yet. The causal ATT/ATNT validation
  harness is a separate follow-up (PR-C).
- `score_differential` (confounder #1) requires match-context `actions`; `compute`/`add` accept an
  optional `actions=` kwarg (NaN-tolerant when omitted). A future `infer_ball_carrier` carrier-default
  change is an xCross retrain trigger (carrier params recorded + consumed from model metadata, R3).
- **Released on top of** the ghost-GK re-fit (PR-S81, 4.10.0); TF-17 ships as 4.11.0.

## [4.10.0] — 2026-06-03

### Fixed — Ghost-GK serve-carrier consistency (PR-S81)

`compute_ghost_gk` now computes the ball-carrier on the **full** frames and threads it into
feature extraction, so the `team_in_possession` feature matches training. Previously the serve
path passed no carrier, leaving `team_in_possession` hardcoded to `0.0` at inference while
training computed the real carrier — a latent train/serve skew (contradicting the TF-18 spec §5).

This **changes served `ghost_gk_x` / `ghost_gk_y`** on the small fraction of frames where the
defending GK's team is in possession. Measured on a real SkillCorner match (3000 GK-samples):
**0.4 % of frames change, max 4.03 m, median 0 m, mean 0.004 m** — a long-tail effect, but a
Hyrum-observable change for consumers (incl. the lakehouse). Driven by the bug fix, so it applies
to every variant, not only the re-fit.

### Changed — Ghost-GK R3 carrier-param record/consume + 4.7.0 re-fit (PR-S81)

- **R3.** `GhostGkModel` now records the ball-carrier scoring params (`tolerance_m`/`beta`/`gamma`)
  it was trained under in `metadata.json` (model `version` 1.0.0 → 1.1.0), plus
  `sklearn_version` / `training_commit` / `training_platform`, and **consumes** them at serve
  (`compute_ghost_gk` resolves possession with `model.carrier_params`, not the live library
  default). Mirrors the xShotOccurrence R3 pattern. Back-compatible: a v1.0.0 artifact without the
  field loads with the library default.
- **Bundled weights re-fit** against the 4.7.0 carrier defaults (`beta=0.0, gamma=0.25`,
  PR-S79) on 81 pining matches (887k samples, DGX Spark): `default` (wheel) + `full` (Hub). The
  re-fit is quality-equivalent to the incumbent (held-out KDE-mode MAE 4.47 m vs 4.41 m;
  `predict_mean` CV 1.12 m vs 1.14 m) and aligns the served carrier regime with the library
  default + adds R3 provenance.
- `compute_ghost_gk` / `add_ghost_gk` / `ghost_gk_xfns` gain an optional `carrier=` passthrough
  (cache convention, mirrors `links`) so pipeline callers compute the carrier once.
- `prepare_ghost_gk_training_data` gains an additive `carrier_params=` kwarg (return type
  unchanged); the shared `_build_occurrence`-style time-windowed extraction is unchanged.

### Internal

- Shared `DEFAULT_CARRIER_PARAMS` consumed by Ghost-GK (anti-drift). New maintainer scripts:
  `validate_ghost_gk_refit.py` (apples-to-apples gate), `measure_ghost_gk_serve_delta.py`,
  `_loader_pining_to_cache.py`. `train_ghost_gk.py` records the carrier params + provenance.

### Packaging

- The `full` Ghost-GK weights (~91 MB) are now **removed from the repository** — they are
  Hub-distributed (`silly-kicks/ghost-gk-v1`) and `from_variant("full")` falls back to
  `from_hub`. A `[tool.hatch.build.targets.sdist]` exclude is added alongside the existing wheel
  exclude (each hatch target has its own include/exclude set): the larger re-fit `default` had
  pushed the sdist — which still bundled `full/` — past PyPI's 100 MB per-file limit.

## [4.9.1] — 2026-06-03

### Fixed — DAS crash on a degenerate (zero-frame) frame subset

`add_das` / `das_at_action` / `get_das` / `get_individual_das` could crash with
`AttributeError: 'NoneType' object has no attribute 'x_grid'` when handed a frame subset in which
**no single frame contains both the ball and players** (after resolving `team_in_possession`).
accessible-space restricts its simulation to frames present in *both* its ball-row set and its
player-row set (`transform_into_arrays`: `frames_to_consider = ball_frames & player_frames`), but its
own emptiness guard runs *before* that intersection — so a non-empty subset whose ball and player
frames are **disjoint** collapses to a zero-frame `PLAYER_POS` (`F == 0`), `simulate_passes_chunked`
returns `None`, and `get_dangerous_accessible_space` dereferences `None.x_grid`. The resulting
`AttributeError` was not in silly-kicks' DAS-degradation `except` tuple, so it propagated as a hard
crash instead of degrading to NaN.

This surfaced in a downstream lakehouse run (Gradient Sports match 10502, one action batch whose
per-action **link-restricted** frames lost their ball or player rows) on silly-kicks 4.9.0 with
accessible-space 2.1.0. The unguarded `None` dereference exists across accessible-space 2.x.

**Fix:** a new `_has_simulatable_frame()` precondition in the silly-kicks DAS boundary detects the
disjoint-frame case *before* calling accessible-space and returns **NaN DAS** (with a `UserWarning`),
consistent with silly-kicks' existing "undefined case → NaN DAS" contract. This makes the whole
`add_das` family robust to the accessible-space fragility for *all* consumers, not just the one that
hit it. Valid frames are unaffected (the guard fires only when `ball_frames ∩ player_frames` is empty).
No public API change; no behavior change for inputs that already produced DAS.

`get_xc` (expected pass completion) shares the same accessible-space boundary and the same degenerate
collapse (`get_expected_pass_completion` runs the identical `transform_into_arrays`, simulating one
frame per pass). When no pass references a frame containing both the ball and players, that path
also reaches `F == 0` — surfacing as `AssertionError: Dimension F is 0` rather than the DAS path's
`AttributeError`, but the same root, and `get_xc` had no NaN degradation of its own. The same
precondition (shared `_frames_with_ball_and_players` helper) now guards `get_xc`, returning **NaN xC**
for the affected passes instead of crashing.

`get_xc` is now also hardened against the accessible-space × pyarrow-strings incompatibility: it used a
lighter frame prep that coerced only `player_id` to numpy object, leaving pyarrow-backed `StringDtype`
team columns in place — and accessible-space's offside path 2-D-indexes the team arrays
(`passer_teams[:, np.newaxis]`), which pyarrow strings reject with `IndexError: too many indices for
array` (the default string dtype on newer pandas / Python 3.11+, so it bit only those CI legs).
`get_xc` now uses the canonical `_prepare_frames` (which coerces `team_id` / `team_in_possession` /
`player_id`) and coerces the pass `team_id` / `player_id` too. This mirrors the DAS path's existing
coercion.

## [4.9.0] — 2026-06-02

### Added — TF-16 xShotOccurrence (xS) trained weights (GKDV Layer 2)

The xS shot-occurrence model now ships **trained** (PR-S75 shipped it untrained). A bundled
`default` variant (~1.2 MB XGBoost booster) loads via `XShotOccurrenceModel.from_variant("default")`
/ `from_hub`, and `model=None` on `compute_xshot_occurrence` / `add_xshot_occurrence` now resolves to
it. `xshot_occurrence_xfns` is wired into `pre_shot_gk_full_default_xfns` (and its atomic mirror) —
**not** the general `tracking_default_xfns`, which stays model-free (adding a frame-time bundled-weights
+ `[xgboost]` dependency to the broad default would be a Hyrum break). New `scripts/publish_xshot_occurrence.py`.

**Training (DGX Spark, 81 matches / 1,194,849 rows / ~18% positive; against the 4.7.0 carrier
defaults).** A pre-registered two-candidate comparison — `public` (skillcorner + idsse) vs `full`
(+ gradientsports), evaluated on a common public held-out set at shared hyperparameters — found that
**adding owner-tier gradientsports data degraded generalization to public-provider matches in all 5
folds** (PR-AUC Δ ≈ −0.037), so the **reproducible `public` model shipped** (CV PR-AUC 0.307 > base
rate 0.202; Brier 0.151 < base-rate 0.161). Model metadata records `shipped_variant` + `provider_list`,
`carrier_params`, `pitch_length`/`pitch_width`/`geometry_version` (TF-38 coordinate-change template),
and `xgboost_version`/`training_platform`. `pyarrow` added to the `[train]` extra (feature-cache).

### Changed

- **xS carrier defaults sourced from a single shared constant**
  `silly_kicks.tracking._ball_carrier.DEFAULT_CARRIER_PARAMS` (the 4.7.0 calibrated `tolerance_m=3.0,
  beta=0.0, gamma=0.25`) — removes the prior stale hardcoded copy and any future drift.
- **xS HPO objective** now uses `StratifiedGroupKFold` (stable per-fold positives under the ~0.02 base
  rate) and **drops `scale_pos_weight`** (xS is a calibrated `P(shot)`; the trainer gates on PR-AUC
  **and** Brier vs base rate and is fail-closed — it refuses to write a sub-bar artifact).
- `home_team_id` is now optional on the xS serve surface (it was unused — goal is resolved GK-based).
- `XShotOccurrenceModel.load` fails closed on a pitch-dimension/unit metadata mismatch (warns on a
  translation-only `geometry_version` change).
- **`prepare_xshot_training_data` no longer subsamples** — the `negative_subsample`/`seed` parameters
  are removed and it always returns the faithful class distribution (it is the train/serve-parity
  entry point; subsampling it pre-split silently contaminated downstream CV eval folds + base-rate
  baselines). Negative subsampling now lives in a standalone **`subsample_negatives(features, labels,
  groups, *, fraction, seed)`** helper with a **train-only** contract, applied by the trainer to
  **train folds only** (HPO + gate CV + paired test + final fit); held-out folds always keep the true
  balance. (Surfaced as review M3.)

## [4.8.0] — 2026-06-02

### Added — opt-in `kde_backend="fft-cic"` ghost-GK KDE backend (CIC / bilinear binning)

A fourth opt-in `kde_backend="fft-cic"` for the ghost-GK KDE, adding **CIC (cloud-in-cell / bilinear)
binning** on the existing FFT-convolution path (`predict_density` / `compute_ghost_gk` /
`add_ghost_gk` / `ghost_gk_xfns`, and the atomic mirror — flat string, no signature change). Binning
is the only seam: `_kde_density_fft` (NGP) and `_kde_density_fft_cic` share the extracted
`_kde_setup` + `_fft_convolve_field` verbatim and differ only in `_bin_ngp` vs `_bin_cic`. On near-tie
**multimodal** grids CIC reduces NGP's spurious mode flips ~76% (real data: NGP shifts the emitted GK
mode up to ~6 m on ~22% of actions → CIC ~5%) and tightens the raw grid (~5.7e-3 vs 1.5e-2 median
rel-err), at ~2× the NGP bin cost (still ~1000×+ over brute force). No new dependency (core scipy).
**Prefer `fft-cic` over `fft` for new FFT consumers** unless you need NGP's extra speed on
known-unimodal data; `vectorized`/`cpu-numba` remain the only exact-raw-grid backends. Decision:
ADR-014 (amended).

### Changed — ADR-014 mode-fidelity correction (`fft` docstring; no runtime change to `fft`)

`"fft"` (NGP) is **unchanged** and stays the fft-default — existing `"fft"` callers are unaffected.
But its documented fidelity contract is corrected: 4.6.0 claimed the emitted scalars (incl. the mode)
are "robust to per-cell binning noise"; that holds for mean/spread always and the mode on *unimodal*
grids, but **on near-tie multimodal grids NGP can flip the emitted mode by several metres** — a claim
4.6.0's *unimodal* parity bench structurally could not surface. The `_kde_density_fft` /
`predict_density` docstrings and ADR-014 are amended accordingly.

**Hyrum heads-up:** any trained-model consumer of the ghost-GK *mode* should pin one `kde_backend` for
train and serve (and persist it in metadata) — under `fft` the GK mode can differ by ~6 m on
multimodal frames. (TF-16 xShotOccurrence is unaffected — it uses the resolved/defending GK, not the
ghost-GK mode.)

## [4.7.0] — 2026-06-02

### Changed — TF-24 apply: Optuna-calibrated `infer_ball_carrier` defaults (`beta` 0.5→0.0, `gamma` 1.0→0.25)

The TF-24 calibration is applied. `infer_ball_carrier` and `ball_carrier_at_action` defaults change:
**`beta` 0.5 → 0.0** (velocity-toward-ball weighting did not help carrier-actor accuracy → selection is
now purely distance-based) and **`gamma` 1.0 → 0.25** (near-stateless hysteresis). These are Optuna-calibrated
at the held `tolerance_m=3.0` against a 3-provider fold (SkillCorner + IDSSE/DFL + Gradient Sports); the
Balanced (25-match) and Gold-max (45-match) folds **independently agreed** (`beta`≈0.0002/0.0009,
`gamma`≈0.221/0.259). Gain is modest — ~+2pp carrier accuracy at the default radius. This closes TF-24.

**`tolerance_m` is deliberately left at 3.0.** The carrier-actor-action calibration objective is
**under-determined on the radius**: its labels are on-ball moments only (no loose-ball negatives), so a wider
radius monotonically improves recall and the objective presses `tolerance_m` to the upper search bound on both
folds — a label-set artifact, not a validated optimum. Calibrating the radius would need loose-ball negatives.
(The earlier `tolerance_m≈1.0` from the pre-4.4.0 sweep was the *opposite* artifact of a since-fixed
precision-only objective; see 4.4.0.)

**Heads-up (Hyrum's Law):** `infer_ball_carrier` is called across the tracking layer (DAS, ghost-GK,
defensive line, team shape, possession), so this shifts carrier attribution for **every** tracking consumer
(including lakehouse) — calibrated and modest, but a behavior change. It is also a retrain input for the
(currently untrained) TF-16 xShotOccurrence model, which records + consumes the carrier params. **TF-25**
(provider-specific pressure-aggregation form) is **not triggered** — the cross-provider dispersion its trigger
requires did not appear (the only dispersion was the `tolerance_m` label-set artifact + carrier data-quality
differences, neither indicating provider-dependent aggregation form).

## [4.6.0] — 2026-06-02

### Added — `kde_backend="fft"` ghost-GK KDE backend (binned-convolution, ~2000× on the full-k regime)

`GhostGkModel.predict_density` / `compute_ghost_gk` / `add_ghost_gk` / `ghost_gk_xfns` accept
`kde_backend="fft"` (default stays `"vectorized"`). The three existing backends are brute-force
point×grid (O(k·m)); `fft` bins the weighted training points onto the fixed grid (nearest-grid-point)
and runs one `scipy.signal.fftconvolve` against the analytic anisotropic Gaussian — **O(k + m·log m)**,
independent of k. On the production regime (`_leaf_match_weights` returns all ~35 816 training points
on every prediction → 137 M Gaussian evals brute-force), measured **~2355×** (4247 → 1.80 ms/prediction).
Reuses the exact `_kde_setup` kernel + `cho_factor` PD-branch, so the singular-covariance uniform
fallback is unchanged. **No new dependency** (scipy is already core). Decision: ADR-014.

**Faithful on the emitted scalars, NOT on the raw grid (opt-in for this reason).** `fft` matches the
scipy oracle on the three values `predict_density` emits — `mode_x`/`mode_y` (39/40 exact, ≤1 grid
cell), `mean_x`/`mean_y` (≤5.5 mm), `spread` (≤0.16% rel) — because those are grid integrals / entropy
/ argmax-peak, robust to per-cell binning noise. It is **NOT bit-faithful on the raw
`GhostGkDensity.probabilities` grid** (NGP binning quantizes per-cell mass: ~1.5% typical, up to ~65%
on near-zero tail cells). **Hyrum's Law:** consumers that read the raw `probabilities` grid (not just
the 3 scalars) should keep `"vectorized"`; consumers that froze a golden on `ghost_gk_x/y` must
re-baseline when adopting `fft` (~2.5% of predictions flip the discrete mode by ≤1 cell — a genuine
flat-ridge near-tie). Default unchanged, so this is non-breaking.

## [4.5.0] — 2026-06-02

### Added — cacheable carrier inference (`pre` / `links` kwargs) for the calibration sweep

`infer_ball_carrier` gains an optional `pre` kwarg (a precomputed `_pre_index_frames(frames)`),
and `ball_carrier_at_action` gains optional `pre` + `links` kwargs (mirroring the `links` convention
on the `add_*` aggregators). The pre-index step (long-form frames → dense per-frame numpy arrays)
**dominates carrier-inference cost — ~99%, measured — yet is a pure function of `frames`**, fully
independent of the swept `tolerance_m`/`beta`/`gamma`; likewise the action→frame linking depends only
on the fixed link tolerance. Callers that re-resolve carriers on the *same* frames with *different*
params can now compute these once and pass them back, skipping the re-marshalling. Both default to
`None` (compute internally) and are **bit-identical** to recomputing — gated by
`tests/tracking/test_ball_carrier.py::TestCachedPreLinks` (`assert_frame_equal` / `assert_series_equal`
across the defaults, the recall-aware optimum region, and a tight radius).

### Changed — TF-24 Stage-1 carrier objective uses an invariant-prepare cache (~50–100× faster sweep)

`CarrierAccuracyObjective` previously re-ran the full per-match pre-index + linking on **every Optuna
trial**, even though both are param-invariant — making the Stage-1 sweep pandas-bound (numba accelerates
only the ~1% kernel, so it gave no real speedup). `_match_accuracy` is now split into a cached
`_prepare_match` (the param-invariant pre-index + link pointers + linked mask + actor ids, computed once
per match and reused across all trials) and a cheap `_accuracy` (kernel + lookup only). Measured **137×**
per-trial speedup on a 20k-frame synthetic match; a full gold-max Stage-1 sweep drops from ~days to
~tens of minutes. The result is bit-identical to the uncached path — `_match_accuracy` is retained as the
one-shot reference oracle, gated by `test_prepare_cached_once_and_matches_uncached` (prepare runs exactly
once per match; cached evaluate == uncached). No public objective API change; zero global state.

## [4.4.1] — 2026-06-01

### Fixed (documentation/test) — correcting the 4.2.0 DAS "value-neutral" claim

The 4.2.0 changelog stated the ball-carrier offside forwarding was *"value-neutral (zero AS/DAS
change) on real data"*. **That was wrong.** Its validating A/B test placed the carrier clearly
onside, so it never exercised the offside path. On real matches the on-ball carrier is frequently
tracked just ahead of the ball/offside line, where accessible-space (with `respect_offside`, the
default) would **delete the carrier as offside** ("treats offside players like air") unless the
passer is exempted. Forwarding `player_in_possession_col` (4.2.0+) exempts the passer, so **DAS
(`das_team`/`das_opponent`/`das_diff`) did change in 4.2.0 — a correctness fix, not a regression**:
the ball carrier is no longer mis-flagged offside. The shift is large but rare (≈1% of frames, only
where the carrier crosses the offside line; tens–hundreds of m² when it hits), because deleting a
central on-ball player materially perturbs the accessible-space tessellation.

No runtime behaviour changes in this release. Changes: (a) the misleading
`test_forwarding_is_value_neutral_and_silences_warning` is renamed/scoped to the onside-only case,
and a new `test_offside_carrier_forwarding_changes_das` locks the correct behaviour (DAS *must*
change when the carrier would be offside); (b) ADR-012 amended to record the corrected finding.

**Downstream (Hyrum's Law):** consumers who froze DAS goldens under ≤4.1.1 must re-baseline — the
≤4.1.1 values encode the pre-fix bug (carrier mis-flagged offside). The ≥4.2.0 values are correct.

## [4.4.0] — 2026-06-01

### Fixed — TF-24 Stage-1 carrier objective was precision-only (no recall term)

`CarrierAccuracyObjective` (`silly_kicks.calibration._carrier_objective`) averaged accuracy **only over
carrier-actor actions where a carrier was inferred** — `matched[valid].mean()` with `valid = inferred.notna()`.
Actions whose actor ended up beyond `tolerance_m` of the ball (→ NaN inference) were dropped from the
denominator instead of counted as misses, so there was **no recall penalty**: accuracy rose monotonically as
the candidacy radius shrank, and the optimum collapsed onto the search lower bound. The objective was
structurally blind to the very parameter it calibrates. `_match_accuracy` now uses the set of carrier-actor
actions that successfully **link** to a frame as the denominator; a linked action with a NaN inferred carrier
is a **miss**, while genuine link failures (independent of the swept params) stay excluded. This makes the
objective sensitive to `tolerance_m` (an over-tight radius is penalized through lost recall). Calibration-only
— no public runtime API change. Regression-gated by `tests/calibration/test_carrier_objective.py`
(`test_unreachable_actor_counts_as_miss` = 0.5, `test_link_failure_excluded_not_penalized` = 1.0).

**Consequence for the TF-24 apply-PR:** the completed maintainer sweep's headline `tolerance_m ≈ 1.0` (both
folds, pressed to the search lower bound) is now understood to be a **degenerate boundary artifact of the old
precision-only objective, not a validated optimum** — the two folds reproduced the same artifact, not an
independent optimum. The `infer_ball_carrier` defaults are therefore **left unchanged** (`tolerance_m=3.0`,
`beta=0.5`, `gamma=1.0`) pending a Stage-1 **re-sweep on the fixed objective**, which will produce a real
interior optimum to apply in a follow-up. Stage-2 (augmented-VAEP Brier; `k3`, `pre_seconds`,
`min_displacement_m`) is a separate held-out-Brier objective and is unaffected by this fix.

**TF-25 (provider-specific defaults) disposition:** not triggered. TF-25 fires only if `tolerance_m`/`k3`
disperse meaningfully across providers; the only Stage-1 signal so far (the boundary collapse) is an artifact,
and Stage-2 was flat. Re-evaluate after the fixed-objective re-sweep.

### Changed — kloppy `convert_to_actions` auto-derives `game_id` from dataset metadata

`silly_kicks.spadl.kloppy.convert_to_actions(dataset, game_id=None)` now falls back to the dataset's own
`metadata.game_id` (stringified to match the tracking gateway `silly_kicks.tracking.kloppy`, which uses
`str(metadata.game_id)`) when the caller omits `game_id`. Previously the column was left unset (`None`).
This is the **library-side fix for the IDSSE/Sportec join failure** that the TF-24 harness worked around at
the loader layer: SPADL actions carried `game_id=None` while the frames carried the real id, so the
`(game_id, period_id, frame_id)` joins in every tracking `add_*` enrichment missed every row. Now any
kloppy-gateway consumer (Sportec/IDSSE, plus Metrica/SkillCorner via kloppy) gets join-compatible event and
frame `game_id`s out of the box. **Heads-up (Hyrum's Law):** a caller that omitted `game_id` and relied on
the column staying `None` will now see the dataset's id; pass `game_id` explicitly to override (caller values
are always respected verbatim, ADR-001). Datasets with no `metadata.game_id` (e.g. the Metrica fixture) keep
the unset/NaN column. Gated by `tests/spadl/test_kloppy.py::TestKloppyGameIdAutoDerive`.

### Build — `numba` added to the `[calibration]` extra

The `[calibration]` extra now installs `numba>=0.59.0`. The Stage-1/Stage-2 objectives call
`infer_ball_carrier` + the pitch-control kernels once per trial; without numba those run as pure-Python loops,
the dominant cost of a full sweep. Calibration venvs now get the compiled fast path by default (the TF-24
maintainer sweep ran without it and was needlessly slow). `import silly_kicks` stays numba-free (lazy `@njit`).

## [4.3.0] — 2026-06-01

### Added — `cpu-numba` ghost-GK KDE backend (~10× the closed-form hot loop, single-thread)

`GhostGkModel.predict_density` / `compute_ghost_gk` / `add_ghost_gk` / `ghost_gk_xfns` accept
`kde_backend="cpu-numba"` (default stays `"vectorized"` = cpu-numpy). It runs a serial `@njit` fully-fused closed-form KDE loop
(no per-block temporaries), validated parity-exact (rtol 1e-9, incl. the production-scale k≈36000 case
and the near-singular zone) against the numpy kernel. The headline **~10× on the hot loop was measured
numba-serial vs numpy with all thread env vars pinned to 1** (`OMP/OPENBLAS/MKL/NUMEXPR/NUMBA_NUM_THREADS=1`)
— single-thread-vs-single-thread, the Spark-`applyInPandas` in-venue reality. The numpy setup keeps
`cho_factor` for the PD/singular branch + `log_det`, so the singular→uniform fallback boundary is
byte-identical to the numpy path. Requires the `[numba]` extra (lazily imported; `import silly_kicks`
stays numba-free). Opt-in — value-equivalent to the numpy default within golden tolerance.

### Changed — default ghost-GK KDE whitening is now closed-form (removes `cho_solve`)

**Heads-up for pinned consumers (Hyrum's Law): this shifts the DEFAULT `vectorized` backend's output, not
just the opt-in `cpu-numba` path.** `_kde_density_vectorized` now computes the 2×2 Mahalanobis energy in
closed form (`0.5/det·(h₂₂·dx² − 2·h₁₂·dx·dy + h₁₁·dy²)`) instead of `cho_solve`, sharing a new `_kde_setup`
with the numba backend. Every consumer's `ghost_gk_x`/`ghost_gk_y`/`ghost_gk_spread` move by `~1e-12..1e-9`
on a plain `4.2.0 → 4.3.0` upgrade, even without selecting a new backend. `cho_factor` is retained for the
PD-branch + `log_det` (singular→uniform boundary unchanged from 4.2.0); value-equivalent within the frozen
golden's `rtol≈1e-7` (golden NOT regenerated). The closed form alone is ~1.0× single-thread (the win is the
numba loop above); it lands as the shared foundation.

## [4.2.0] — 2026-06-01

### Changed — ghost-GK density now uses a vectorized scipy-faithful KDE (default)

`GhostGkModel.predict_density` replaces the per-sample `scipy.stats.gaussian_kde` with a
vectorized weighted-Gaussian KDE kernel that reuses scipy's exact Scott bandwidth +
weighted-covariance + Cholesky whitening (`cho_factor`/`cho_solve`), so outputs match the
scipy reference within float64 tolerance (golden-master gated: continuous grid at
`rtol≈1e-7`+atol+NaN-mask, discrete mode at exact argmax). The scipy path is retained as a
selectable reference via a new `kde_backend="scipy" | "vectorized"` argument (default
`"vectorized"`); the per-sample leaf-match is vectorized and the training set is streamed in
blocks (`train_block`, default 1024) to bound memory under the serverless 1 GB UDF cap.
Motivation: full-chain profiling identified `add_ghost_gk` as the dominant action-context
cost. Output columns (`ghost_gk_x/y/spread`) and the public API are unchanged.

### Added — DAS forwards the ball carrier to accessible-space (correct offside, no log flood)

`derive_team_in_possession` now also preserves `ball_carrier_player_id` on the returned
frames. `get_das` / `get_individual_das` accept `player_in_possession_col`
(default `"ball_carrier_player_id"`): when present it is forwarded to accessible-space so
`respect_offside` (the DAS default) excludes the passer from the offside mask. This silences
accessible-space's per-call `player_in_possession_col` warning that previously flooded logs.
A/B + unit tests confirm the forwarding is value-neutral (zero AS/DAS change) on real data;
any future change would be a documented accuracy improvement. When no carrier column is
available, silly-kicks emits its own one-time guidance instead of the per-call library
warning. An explicitly-named missing column raises `ValueError`.

### Fixed — clearer dead-ball message on link-restricted DAS subsets

When `add_das(..., links=...)` restricts to an all-dead-ball frame subset, silly-kicks now
raises its own clear "dead-ball window" `ValueError` (degraded to NaN as before) instead of
letting accessible-space's generic "empty / no non-NaN team in possession" error surface.

### Changed — elastic-sync distance lookup vectorized

`_build_player_ball_distance_lookup` builds its key/value dict vectorially instead of per-row
`.iloc` access (behaviour-preserving; golden-checked).

## [4.1.1] — 2026-06-01

### Fixed — numba on-disk cache no longer hard-fails import on read-only installs

`@njit(cache=True)` makes numba persist compiled code to disk, which requires a writable
cache *locator* to be resolved **at decoration time** (module import): a writable
`__pycache__` beside the source, a writable user-wide cache dir, or `NUMBA_CACHE_DIR` set.
On read-only / ephemeral installs — e.g. Databricks serverless, where the wheel lands on a
read-only ephemeral NFS path — all three locators fail and numba raises
`RuntimeError: cannot cache function ... no locator available` from *inside* a successful
import. Because the failing decoration runs when `silly_kicks.tracking` is imported, it took
down **all** tracking functionality (`infer_ball_carrier`, pitch control, and everything
that transitively imports them), not just the cached kernel. The existing
`try/except ImportError → _HAS_NUMBA = False` fallbacks did not catch it — the exception is a
`RuntimeError`, not an `ImportError`.

- The four `@njit` kernels (`_carrier_loop_numba` in `tracking/_ball_carrier_numba.py`;
  `tti_numba` / `influence_numba` / `gaussian_influence_numba` in
  `tracking/pitch_control/_numba_kernels.py`) now gate `cache` on a module-level
  `_NUMBA_CACHE` flag that **defaults OFF**, so import never resolves a cache locator and
  cannot hard-fail on a read-only/ephemeral filesystem. `cache=False` keeps full native JIT
  speed; it only drops cross-process cache persistence (a one-time ~1–5 s recompile per fresh
  worker process — which an ephemeral worker discards on teardown anyway).
- Opt back in to on-disk caching in stable environments (persistent cluster, local dev with a
  writable install) via `SILLY_KICKS_NUMBA_CACHE=1`, **or** by pointing numba's own
  `NUMBA_CACHE_DIR` at a writable directory (a consumer that sets it gets caching for free,
  with no second silly-kicks-specific variable to remember).
- Regression coverage: `tests/tracking/test_numba_cache_gating.py` asserts the default env
  disables the cache (the decorated dispatchers keep numba's `NullCache`) and that either
  opt-in env var re-enables it.

Caught 2026-06-01 running tracking enrichment on Databricks serverless.

## [4.1.0] — 2026-05-31

### Added — xShotOccurrence (xS) model (TF-16, GKDV Layer 2)

Per-frame shot-occurrence probability — `xS = P(a shot is attempted by the in-possession
team within ~1 second of a tracking frame)` — implementing the xS sub-model of Pipping,
Feng & Sabin (2026), arXiv:2512.00203 ("Beyond Expected Goals: A Probabilistic Framework
for Shot Occurrences in Soccer"). Distinct from xG: xS models shot *taking*, not shot
*quality*. This is GKDV Layer 2 — TF-19 will decompose `P(shot | actual_GK) −
P(shot | ghost_GK)`. The paper's xG and xG+ composition are deliberately out of scope
(silly-kicks values goals/threat via VAEP and xthreat).

- New `silly_kicks.tracking._xshot_occurrence`: the paper-faithful 27-feature extractor
  (`extract_xshot_features`; ball r/θ/z/speed, `openGoal` goal-mouth obstruction, GK
  distance/bearing, 5 nearest defenders + 5 nearest attackers) in goal-relative
  coordinates via a new shared `silly_kicks.tracking._geometry` helper; a time-windowed
  label builder (`build_xshot_labels`, robust to non-contiguous `frame_id`); the
  `XShotOccurrenceModel` (deterministic XGBoost, pickle-free booster-JSON + SHA256SUMS
  serialization); and the ADR-005 surfaces `compute_xshot_occurrence` /
  `add_xshot_occurrence` (`@nan_safe_enrichment`) / `xshot_occurrence_xfns`.
- `prepare_xshot_training_data` — the shared train/serve feature/label entry point with
  the paper's data-curation domain filter (alive-ball + attacking-third) and an optional
  seeded negative-subsample.
- HPO via the `ruthless` `CachedObjective` substrate (new `silly_kicks.tracking
  ._xshot_occurrence_objective`) + a `scripts/train_xshot_occurrence.py` CLI. New generic
  `[train]` extra (`ruthless-efficiency[optuna]` + xgboost); inference gates on the
  existing `[xgboost]` extra and keeps `import silly_kicks` dependency-light.
- `XShotFeatureSet` Literal with the `"extended"` variant reserved (raises
  `NotImplementedError` this release). Atomic mirror in `atomic.tracking.features`.
- Decision: **ADR-011** (trained-model feature lifecycle: code → training → bundled/Hub
  weights). Attribution: NOTICE entry for arXiv:2512.00203.

**Ships untrained.** This release is code + a synthetic CI fixture + real-provider
extraction tests only; no model weights are bundled (`from_variant`/`from_hub` raise until
the follow-up). The maintainer training run, bundled/Hub weights, the empirical PR-AUC
acceptance gates, and wiring `xshot_occurrence_xfns` into the default xfn lists are
deferred to a follow-up PR (it needs the gated multi-provider corpus the live TF-24 sweep
is using). Note: a future TF-24 apply-PR change to `infer_ball_carrier` defaults is an xS
retrain trigger — the carrier params used are recorded in model metadata and consumed at
inference to keep train/serve consistent until then.

## [4.0.3] — 2026-05-30

### Fixed — TF-24 calibration loader download resilience (maintainer tooling only)

The pining match loader (`scripts/_loader_pining.load_matches`) had no retry: a single transient
download/read blip (an empty/partial S3 fetch surfacing as kloppy `InputNotFoundError`, or a
`urllib`/`OSError` network hiccup) crashed the entire fold load. Across the TF-24 sweep's ~140
match-downloads (two phases × Stage 1 + Stage 2, each re-downloading its matches), a crash during
Stage-2 `prepare()` would discard hours of DAS enrichment.

- New `_build_match_with_retry` wraps each match's download+build in a 3-attempt loop with a fresh
  temp dir and linear backoff, then fails loud if the match is genuinely unfetchable.

**Consumer impact: none.** Confined to `scripts/` + `tests/`; the importable `silly_kicks` package is
byte-identical to 4.0.2 apart from the version string. Released for traceability.

## [4.0.2] — 2026-05-30

### Fixed — TF-24 IDSSE calibration provider exclusion (maintainer tooling only)

The TF-24 calibration loader silently calibrated on two providers instead of three. The Sportec
kloppy-gateway converter (`spadl_kloppy.convert_to_actions`) leaves `game_id` as `None`, while the
loader's frames carry the DFL match id from kloppy metadata. Every tracking-feature join (ball
carrier, DAS, defensive line, team shape) keys on `(game_id, period_id, frame_id)`, so the
`None`-vs-id mismatch dropped every IDSSE row → zero carrier signal → `signal_sanity` excluded IDSSE.

- `scripts/_loader_pining._build_idsse` now stamps `actions.game_id` from the frames' `game_id`
  (verified: 0 → 772/1090 valid carrier inferences on a real IDSSE match).
- `scripts/calibrate_tracking_defaults._load_fold` gains a fail-loud `game_id`-consistency guard
  (`_assert_match_game_id_consistent`) so a silent provider drop can never recur, with unit tests.

**Consumer impact: none.** Changes are confined to `scripts/` + `tests/` (the maintainer calibration
harness, not shipped in the wheel); the importable `silly_kicks` package is byte-identical to 4.0.1
apart from the version string. Released for traceability. The lakehouse stamps `game_id` from its
bronze tables and is unaffected.

## [4.0.1] — 2026-05-30

### Fixed — TF-24 calibration sweep runnable on all three providers

Two latent bugs blocked the TF-24 maintainer calibration sweep. Both lived in code
paths the calibration tests never exercised — the Stage-2 **CLI** wiring and the
**Gradient Sports** Stage-2 path (the e2e + unit fixtures cover SkillCorner only).

- **Stage-2 xT wiring in `scripts/calibrate_tracking_defaults.py`.** `main()` passed the
  `FrozenXt` *artifact* straight into `AugmentedVaepBrierObjective`, but the objective needs
  the inner `ExpectedThreat` (gk-influence / cover-shadows call `xt.interpolator(...)`). Stage 2
  via the CLI crashed at `prepare()` with `AttributeError: 'FrozenXt' object has no attribute
  'interpolator'`. The objective now accepts the `FrozenXt` and unwraps `.xt` internally, so the
  CLI passes one artifact to both the objective and the report manifest. Annotations tightened
  (`Any → FrozenXt` / `ExpectedThreat`) so the type checker rejects the mistake; the e2e + smoke
  tests now drive the same wiring the CLI uses, with a new
  `run_stage(stage=2, xt=<FrozenXt>)` regression guard.

- **`bekkers_pi` pressure crashed on duplicate frame records.** Some Gradient Sports tracking
  exports ship the same `(period, frameNum)` record up to 16× (content-divergent copies).
  `_pressure_bekkers` deduped the actor row but not the ball row, so a multi-row ball context
  built a 3-D `ball_pos` and crashed `_bekkers_tti` with a numpy broadcast error. The ball path
  now dedups keep-first (mirrors the actor path). The calibration loader also dedups the upstream
  duplicate frame records (root cause — restores the ADR-004 one-row-per-`(period, frame, player)`
  contract; otherwise pitch-control / DAS / team-shape silently compute on inflated rows too).

## [4.0.0] — 2026-05-30

### Changed (BREAKING) — symmetric fail-loud extra-time direction

Per-period-absolute converters (Sportec/IDSSE, Metrica, Gradient Sports) flip
coordinates **per period** by the home team's start direction. Extra time
(periods 3/4) requires a separate `home_team_start_left_extratime` flag. The
native converters previously handled a **missing** ET flag inconsistently — some
raised, but **Sportec tracking silently defaulted**, shipping geometrically wrong
ET coordinates with no signal. This release makes the behaviour **symmetric and
fail-loud** across all five converters. Decision: **ADR-010**.

- **`silly_kicks.tracking.sportec.convert_to_frames` now RAISES** on extra-time
  (period 3/4) without `home_team_start_left_extratime` (previously it silently
  defaulted to wrong ET geometry). **This is the breaking behaviour change.**
- **Standardized ET error message across all five converters.** Sportec tracking
  (new), Sportec events, Metrica events, Gradient Sports tracking + events all now
  raise the **same `ValueError` message shape** via the shared guard:
  `"<source>: data contains ET periods (period_id in {3, 4}) but
  home_team_start_left_extratime was not provided. ..."`. Sportec/Metrica **events**
  and Gradient Sports already raised on ET-without-flag (since 3.0.1 / earlier);
  their message **text** is now standardized — the exception **type stays
  `ValueError`** and the trigger condition is unchanged. **Consumers parsing the
  old message text must update** (Hyrum's Law: 4 messages re-worded).
- **New public guard `silly_kicks.tracking.require_et_direction(period_ids,
  home_team_start_left_extratime, *, source)`** — re-exported from
  `silly_kicks.spadl` for the events side. Lets consumers pre-flight-validate a
  batch before converting (and a CI sentinel detect a pin/metadata mismatch).
- **New public helper `silly_kicks.tracking.filter_extratime_frames(frames, *,
  label)`** — drops ET periods for **calibration/sampling only** (with a
  `UserWarning`); production must source the real ET flag, not drop ET.
- **Module rename `silly_kicks.tracking._direction` → `silly_kicks.tracking.direction`**
  (now a public module; single home for the direction helpers + the guard). The
  `home_attacks_right_per_period` function keeps its name.

### Migration

- **Pass `home_team_start_left_extratime`** to `convert_to_frames` /
  `convert_to_actions` for any match with extra time (sourced from provider
  metadata, e.g. DFL `HomeTeamStartLeftSideExtraTime` / Gradient Sports
  `homeTeamStartLeftExtraTime`). Without it, ET matches now raise.
- **Lakehouse / consumers with ET matches: upgrade to the lakehouse Phase-A PR
  first** (adds `MatchMeta.home_team_start_left_extratime` and plumbs it to
  `convert_to_frames`/`convert_to_actions`) **BEFORE** bumping the silly-kicks pin
  to 4.0.0. A pin bump without that plumbing will raise on any in-scope ET match.
- Importers of the old `tracking._direction` module path must update to
  `tracking.direction`.

### Added

- **TF-24 calibration sweep memory bounds.** `scripts/calibrate_tracking_defaults.py`
  gains `--match-ids PROVIDER:id1,id2` (repeatable), `--max-matches-per-provider`,
  and `--tracking-limit`, threaded through `_load_fold` into the loaders;
  `_loader_pining.load_matches` gains `max_per_provider`. Defaults are unchanged
  (load everything); set the flags to bound memory and run the sweep locally
  (previously the fold load hardcoded "all matches at full depth" and could OOM).

## [3.30.0] — 2026-05-30

### Changed
- **`add_das` no longer crashes on all-dead-ball batches.** When `team_in_possession` is
  all-NaN within the frames (a dead-ball window — e.g. the ball is out of play and
  `infer_ball_carrier` found no carrier), `_pin_attacking_direction` now raises the
  canonical `ValueError` that `add_das` already catches and degrades to NaN, instead of
  letting accessible-space's `infer_playing_direction` raise an **uncaught
  `AssertionError`** (which previously escaped `add_das`'s `except` and crashed the caller).
  Attacking direction is genuinely undefined without a possessing team, so DAS is NaN
  there — an honest "not applicable", not a crash. silly-kicks does **not** fabricate
  possession (the PR-S67 invariant: *"DAS is only valid when a team has possession"*);
  supply `attacking_direction_col=...` to bypass inference when the direction is known.
  Happy path (possession present) is bit-identical.
- **`pressure_on_actor(method="bekkers_pi", use_ball_carrier_max=True)` degrades
  per-action on missing ball rows instead of raising / NaN-ing.** When an action's linked
  frame has no ball position (e.g. Metrica windows where kloppy returned no
  `ball_coordinates`), that action falls back to the Bekkers **base model**
  (pressure-on-player only) — a documented variant (Bekkers 2024 §2.4), never NaN, never a
  raise. Actions whose frames *do* have a ball still use the ball-carrier-max improvement.
  Both the whole-batch `ValueError` (no ball rows anywhere) and the pre-3.30.0 per-action
  NaN are removed; genuine data-shape errors (missing `vx`/`vy`) still raise loudly. Happy
  path bit-identical (golden-master + snapshot unchanged). Atomic mirror included.
  Surfaced by the luxury-lakehouse AC-1 (`bronze.spadl_action_context`) production run on
  IDSSE dead-ball batches.

## [3.29.1] — 2026-05-30

### Changed
- **`ruthless-efficiency[optuna]` floor raised to `>=0.2.1`** in the `[calibration]` extra (and
  the dev/test deps). 0.2.1 fixes a `warm_start` off-by-one in `OptunaStrategy`: a fresh
  warm-started study ran `n_trials - 1` trials (at `n_trials=2`, only the warm-start baseline,
  with zero exploration trials). The TF-24 calibration stage configs seed a warm-start (the
  current library defaults), so the maintainer sweep must run against `>=0.2.1` for `n_trials`
  to be honored and the calibration manifest's trial count to be accurate. Calibration-tooling
  only (the `[calibration]` extra is lazy/optional, not imported by `silly_kicks/__init__`);
  no runtime library change.

### Fixed
- **Calibration manifest `silly_kicks_version`** now records `silly_kicks.__version__` (the
  source version that actually ran) instead of `importlib.metadata.version("silly-kicks")`
  (installed-dist metadata, which is stale on an editable install bumped post-install — the
  typical maintainer dev-sweep environment).

## [3.29.0] — 2026-05-29

### Added
- **`attacking_direction_col` passthrough on `add_das` / `_precompute_das_lookup`**
  (`silly_kicks.tracking.features`). When supplied, it names a column on `frames`
  holding a caller-precomputed **per-frame numeric (+1/-1)** attacking direction —
  one value per `(game_id, period_id, frame_id)`, the in-possession team's
  direction. silly-kicks validates it (exists / numeric / fully covered per group,
  restricted to the action-linked frames), **skips `_pin_attacking_direction`**,
  and threads it straight to `get_individual_das` (the 3.25.0 lower-level
  passthrough propagated up one layer). This lets callers bypass per-frame
  direction inference when the direction is already known and inference would
  assert or mis-infer — notably a dead-ball window with no non-NaN
  `team_in_possession`, where `_pin`'s `infer_playing_direction` raises an
  `AssertionError` that escaped `add_das`'s `except`. A misconfigured column fails
  loud (`ValueError`/`TypeError`, e.g. rejecting a raw string `"ltr"`/`"rtl"`
  column); it is **not** degraded to NaN. The contract is purely additive and
  carries no convention coupling: silly-kicks does not interpret
  `team_in_possession`, map string labels, or touch the library's possession gate
  (frames with NaN possession still yield NaN DAS — invariant preserved). The
  per-team→per-frame reduction and possession modeling remain the caller's
  responsibility. `attacking_direction_col=None` is bit-identical to prior
  behavior (direction inferred via `_pin`). Uncovered by the luxury-lakehouse AC-1
  production run on IDSSE dead-ball batches.

## [3.28.0] — 2026-05-29

### Added
- **TF-24 calibration harness** (`silly_kicks.calibration`, optional `[calibration]` extra):
  Optuna-TPE calibration of three tracking defaults — `infer_ball_carrier`
  (`tolerance_m`/`beta`/`gamma`), `LinkParams.k3`, and off-ball-run
  `pre_seconds`/`min_displacement_m` — against real multi-provider tracking data via
  `ruthless-efficiency[optuna]`. Pure, provider-agnostic objectives/CV/gates in the library
  (`CarrierAccuracyObjective`; `AugmentedVaepBrierObjective` as a ruthless `CachedObjective` with
  invariant-prepare + per-trial-patch and a deterministic-XGBoost cache-equivalence guarantee);
  match-stratified CV (GroupKFold-5 / leave-one-match-out); a **frozen exogenous xT artifact**
  (fit on a disjoint corpus, sha256-checksummed, fail-closed exclusion) for train–serve-consistent,
  leak-free feature extraction; H1 degenerate-feature penalty (stateless, default-Brier-anchored);
  per-provider signal-sanity + DAS-degradation surfacing; TF-25 provider-specific-defaults gate.
  Plus a `scripts/calibrate_tracking_defaults.py` CLI with pining-for-the-data + Databricks-bronze
  loaders (SkillCorner/IDSSE public, Gradient Sports owner-tier) and a data + version + xT-identity
  manifest. The harness **recommends** values + produces an auditable report; it does NOT change
  the library default constants (that is a separate "apply" PR after the maintainer's real sweep).

## [3.27.0] — 2026-05-29

### Added
- **`silly_kicks.tracking.gradientsports.add_gradientsports_player_ids`** — resolves Gradient
  Sports tracking jersey numbers to the events SPADL `player_id`/`team_id` int space via the
  roster (`(team_id, jersey_number) → roster player.id`, output `Int64`, unmatched → `pd.NA`
  never `0`), with `is_goalkeeper` from `positionGroupType == "GK"`, `team_id` from a
  caller-supplied home/away split, and a `GradientsportsRosterReport` audit. Run it before
  `convert_to_frames`. Fixes a silent failure where GS tracking carriers (jersey-derived /
  string ids) could not join GS events SPADL (`int64` player_id) — GS ball-carrier /
  DAS / team-in-possession features were silently broken. Order-safe (elementwise map, no
  row explosion); loud `UserWarning`s on a degenerate match rate, duplicate roster keys, a
  missing/zero-GK `positionGroupType`; never raises (ADR-003). Verified end-to-end on real GS
  WC2022 data (carrier accuracy 0.0 → nonzero). (TF-24 PR-A)

## [3.26.0] — 2026-05-29

### Performance
- **Ghost-GK linked-frame restriction (`add_ghost_gk`, `ghost_gk_xfns`, TF-18).**
  `compute_ghost_gk` gains an optional `link_frame_ids` kwarg that restricts both
  the heavy per-frame feature extraction and the per-sample density KDE
  (`predict_density`) to action-linked frames. The extractor still walks every
  frame to maintain the cross-period one-step velocity state and computes the
  per-period defending-goal mean-x over the full frames, so the two cross-frame
  dependencies are preserved exactly and the per-sample KDE has no cross-sample
  coupling — the output is **byte-identical** to the unrestricted compute (golden
  tests cover the goal-flip and velocity edge cases, plus a discrimination test
  proving a naive frame pre-filter would NOT be bit-identical). `add_ghost_gk`
  derives the set from its link pointers (supplied or internally computed);
  `ghost_gk_xfns` restricts to the union of its three gamestate slots. Measured
  with the bundled model: the per-250-frame batch (the lakehouse fan-out unit)
  drops from ~47.5 min to ~27 s (~100×); the dominant residual is the irreducible
  per-linked-frame KDE (~4.4 s/eval), not extraction (~4.7% of the restricted
  cost). No new columns, no API break (additive kwarg). (PR-S66)

## [3.25.1] — 2026-05-28

### Performance
- **cover_shadows `max_single_defender_blocking_score` (`detailed=False`)** is now
  computed via a single vectorized leave-one-out instead of an `O(blockers × receivers)`
  `lane_control` re-run (~4× faster on a dense 10v10 frame). The per-defender man-marking
  re-classification was hoisted out of the loop — it is provably a no-op for lane-blocker
  removals (removing a non-winner cannot change a greedy nearest-first matching; see the
  `TestManMarkerInvariantUnderLaneBlockerRemoval` property test). **Bit-identical within
  `rtol 1e-10`** (validated against an independent frozen oracle) — **no value or API change,
  and no downstream golden/model regeneration required.** The exact `detailed=True` path is
  unchanged. (PR-S65)

## [3.25.0] — 2026-05-28

### Fixed
- **ELASTIC alignment for native-frame-numbered providers** (IDSSE/Sportec):
  `align_events_to_frames` assumed `frame_id == time_seconds * frame_rate`
  (0-based), producing all-NaN alignments for providers whose `frame_id` has a
  non-zero origin (e.g. period 1 from 10000). Now derives a per-`(game_id,
  period_id)` linear `frame_id ↔ time_seconds` fit (`_fit_frame_time_relationship`)
  used for both the candidate-frame window and the `aligned_ts` / `error_seconds`
  conversion; falls back to `time * frame_rate` when frames lack `time_seconds`.
  0-based providers (Metrica/StatsBomb) are unaffected (bit-identical).

### Added
- **Shared per-frame pitch-control surface** (`PitchControlCache`, TF-7): memoizes
  canonical per-frame surfaces keyed on `(game_id, period_id, frame_id, team,
  method, params, ball_position, decompose)`, so the enrichment families that use
  pitch control compute each surface once instead of once per family. Threaded via
  an optional `pitch_control_cache` kwarg (mirrors the `links` pattern) on
  `add_obso`, `add_cover_shadows`, `add_gk_influence`, `add_player_influence`,
  `add_space_creation`, `add_pitch_control` (+ `pitch_control_at_action`). Each
  aggregator uses a fresh local cache by default (within-pass reuse); a
  caller-supplied cache extends reuse across families in one pass. Only
  canonical-frame surfaces are cached — counterfactual (player-removed) surfaces
  stay uncached. Zero global state. Output is bit-identical.
- `attacking_direction_col` passthrough on `get_individual_das` — supply a
  precomputed per-frame direction column instead of inferring it.

### Changed
- **DAS + shape_graph linked-frame restriction** (perf, bit-identical): when
  `links` is supplied, `add_das` / `add_shape_graph` restrict the expensive
  per-frame computation to the action-linked frames. For DAS, attacking direction
  is pinned on the *full* frames first (`_pin_attacking_direction`, reusing
  accessible-space's own `infer_playing_direction`) before restriction, so the
  per-period direction inference cannot flip on the restricted subset — making the
  result provably bit-identical. shape_graph is a pure per-frame snapshot, so its
  restriction is trivially identical.
- **OBSO**: hoisted the per-period `(frame_id, time_seconds)` window table out of
  the per-pass loop (was `O(passes × frames)`); pitch control now flows through the
  shared cache, reusing surfaces across overlapping pass windows. Narrowed the
  per-pass `except Exception` to `(ValueError, KeyError, IndexError)` so unexpected
  errors propagate instead of being masked as NaN (ADR-002 no-silent-swallow).
- **cover_shadows**: hoisted the receiver position / `xT` / baseline lane-control
  out of the per-blocker loop (bit-identical).

## [3.24.0] — 2026-05-28

### Added
- **Bundled Ghost-GK model weights**: `"default"` variant (~9 MB, 36 k training
  samples) ships inside the wheel — zero-config inference out of the box.
  `"full"` variant (~91 MB, 537 k training samples) lazy-downloads from
  HuggingFace Hub on first use (requires `pip install silly-kicks[ghost-gk]`).
- `GhostGkVariant` type alias (`Literal["default", "full"]`) exported from
  `silly_kicks.tracking`.
- `GhostGkModel.from_variant("full")` class method for explicit variant loading.
- `model="default" | "full"` parameter on `compute_ghost_gk` and `add_ghost_gk`
  (backward-compatible: `None` still selects the default model).

### Changed
- `_resolve_model` cascade: caller > env var > bundled variant (for `"default"`)
  or HuggingFace Hub download (for `"full"`).
- Training script round-trip verification compares serialized weights instead
  of running intractable KDE predictions.
- Training script caches extracted features to disk (`_feature_cache/`) and
  uses `predict_mean()` for permutation importance.
- SHA-256 integrity check normalizes CRLF → LF before hashing `.json` files,
  fixing cross-platform (Windows → Linux CI) hash mismatches.

## [3.23.0] — 2026-05-27

### Added
- `snapshot_to_tracking_frames` public API in `silly_kicks.tracking` — converts
  per-event player-position snapshots (e.g. StatsBomb 360 freeze-frames) into
  the 20-column `TRACKING_FRAMES_COLUMNS` schema + pre-built linkage pointers.
  Enables all single-frame `add_*` enrichment functions on freeze-frame data
  without modification. (PR-S61)
- `"snapshot"` added to `TRACKING_CATEGORICAL_DOMAINS["source_provider"]` domain
  set.

### Fixed
- **Ghost-GK goal_x period-flip**: `extract_ghost_gk_features` hardcoded
  `goal_x` by team identity, which is wrong for SkillCorner LTR-normalized
  data where teams swap ends at halftime. Now infers defending goal per
  (game_id, period_id, team_id) from mean GK x position with team-identity
  fallback. Previously dropped ~50% of SkillCorner training data via
  domain filter.

## [3.22.2] — 2026-05-27

### Fixed
- **DAS exception handling**: Widen `add_das()` / `das_at_action()` / VAEP
  transformer exception tuple from `(ValueError, RuntimeError, ImportError)` to
  also include `IndexError` and `TypeError`. Both occur in production on
  degenerate Voronoi tessellations (collinear players) and NaN tracking
  coordinates respectively. Graceful degradation to NaN columns instead of
  pipeline crash.

## [3.22.1] — 2026-05-27

### Added
- **DAS `chunk_size` passthrough**: `add_das()`, `das_at_action()`, and
  `_precompute_das_lookup()` accept optional `chunk_size: int | None` kwarg,
  threaded through to `accessible-space`. Enables memory-constrained
  environments (e.g. Databricks `applyInPandas` with 1 GB group memory cap)
  to process large matches without OOM.

### Fixed
- **Ghost-GK training script**: `pd.NA` boolean ambiguity crash when
  `ball_carrier_team_id` is `pd.NA` (`extract_ghost_gk_features` line 511).
- **Ghost-GK training script**: Glob priority swap — prefer tc3 cache layout
  (`**/frames.parquet`) over flat (`*.parquet`) to avoid stale non-tracking
  parquets in cache root.
- **CI perf budget**: Bump Andrienko pressure budget from 100ms to 120ms to
  accommodate Windows CI runner timing variance.

## [3.22.0] — 2026-05-26

### Added
- **Game state enrichment** (`add_game_state`): Derives running scoreline from
  successful shots and classifies each action as `"winning"`, `"losing"`, or
  `"drawing"` from the acting team's perspective. Pure SPADL enrichment — no
  tracking data required. `@nan_safe_enrichment` decorated; NaN `team_id` rows
  default to `"drawing"` (ADR-003). Exported from `silly_kicks.spadl`.

## [3.21.0] — 2026-05-26

### Added
- **Library extraction — 5 new tracking primitives + 1 enhancement (TF-39..TF-44, PR-S57):**
  - **TF-39 Shape Graph** (`_shape_graph.py`): Sotudeh 2026 iterative
    Delaunay edge-removal + face-center 5×5 position decomposition.
    `compute_shape_graph`, `ShapeGraph`, `add_shape_graph` aggregator,
    `shape_graph_xfns` 36-column VAEP factory.
  - **TF-40 OBSO** (`_obso.py`): Spearman 2018 Off-Ball Scoring Opportunity
    surface. `compute_obso_surface`, `ObsoSurface`/`ObsoParams` frozen
    dataclasses, `add_obso` aggregator with frame-precomputation cache,
    `obso_xfns` 9-column VAEP factory.
  - **TF-41 Space Creation** (`_space_creation.py`): Fernandez & Bornn 2018
    OBSO-weighted leave-one-out counterfactual. `compute_space_created`,
    `SpaceCreationParams`, `add_space_creation` aggregator,
    `space_creation_xfns` 9-column VAEP factory.
  - **TF-42 PAUSA** (`_pausa.py`): Lee 2026 pass utility via temporal-spatial
    OBSO decomposition. `compute_pausa`/`compute_pausa_batch`,
    `add_pausa` aggregator, `pausa_xfns` 9-column VAEP factory.
  - **TF-43 ELASTIC Sync** (`_elastic_sync.py`): Kim et al. 2025 event-tracking
    synchronization via ball acceleration + proximity scoring.
    `extract_ball_features`, `align_events_to_frames`, `ElasticSyncParams`,
    `add_elastic_sync` aggregator, `elastic_sync_xfns` 6-column VAEP factory.
  - **TF-44 Ward inter-line gaps** (`_team_shape.py` enhancement): Ward
    hierarchical clustering for defensive line identification + inter-line
    gap metrics. `n_defensive_lines` parameter; 3 new columns
    (`defensive_line_height`, `inter_line_gap_1`, `inter_line_gap_2`).
- Atomic mirror re-exports for all new VAEP xfn factories.

## [3.20.1] — 2026-05-26

### Fixed
- **Ghost-GK training script OOM prevention:** Replaced bulk `pd.concat` of all
  tracking parquets with per-file on-demand loading following lakehouse TC-3
  pattern. Raw frames are loaded one parquet at a time, features extracted
  per-game, then frames released immediately via explicit `del`. Peak memory
  drops from ~2x total frame data to one parquet file + accumulated feature
  matrix. Schema validation uses zero-data `pyarrow.parquet.read_schema`.

## [3.20.0] — 2026-05-26

### Added
- **Ghost-GK training data assembly + HuggingFace Hub publish pipeline (TF-18):**
  - `prepare_ghost_gk_training_data`: public API for extracting training
    features + labels from tracking frames with match context resolution
    (score state, set-piece phase), label domain filtering, and subsample support
  - `_build_score_lookup`: home-perspective cumulative score from SPADL goal
    actions with own-goal attribution flip
  - `_build_phase_lookup`: set-piece phase with 10s exponential decay
    (throw-in excluded per restart semantics)
  - `_extract_all_ghost_gk_features`: shared batch helper used by both
    `compute_ghost_gk` (inference) and `prepare_ghost_gk_training_data`
    (training), eliminating duplicated iteration logic
  - `compute_ghost_gk` now accepts optional `actions` parameter for
    match context enrichment (score + phase features)
  - `add_ghost_gk` now accepts optional `actions_for_context` parameter,
    threaded through to `compute_ghost_gk`
  - `scripts/train_ghost_gk.py`: full training CLI with StratifiedGroupKFold
    CV, permutation importance, metrics.json acceptance criteria, round-trip
    verification
  - `scripts/publish_ghost_gk.py`: HuggingFace Hub publish CLI with
    `--verify-only` dry-run mode and download round-trip verification

### Fixed
- **`compute_ghost_gk` timestamp key:** Fixed `"timestamp"` → `"time_seconds"`
  key in velocity state tracking, matching the tracking schema column name

## [3.19.0] — 2026-05-25

### Added
- **Ghost-GK positioning model (TF-18, GKDV Layer 2):**
  Per-frame ghost-GK density prediction using RFCDE (leaf co-occurrence
  weighted 2D KDE over HistGradientBoostingRegressor partitions).
  Predicts where a league-average GK would position given game state.
  - `GhostGkModel`: fit/predict/predict_density/save/load/from_hub
  - `GhostGkDensity`: frozen dataclass (60x64 grid, joint 2D mode)
  - `extract_ghost_gk_features`: 26-feature goal-relative extractor
  - `compute_ghost_gk`: batched per-frame primitive
  - `add_ghost_gk`: action-coupled aggregator (no provenance leak)
  - `ghost_gk_xfns`: 9-column VAEP factory (3 cols x 3 states)
  - Vectorized numpy tree traversal (no sklearn at inference)
  - Serialization: npz + metadata.json + SHA256SUMS (no pickle)
  - Lazy download from HuggingFace Hub via `[ghost-gk]` extra
  - New extras: `[ghost-gk]` (huggingface_hub), `[ghost-gk-train]` (skl2onnx)
  - Training script: `scripts/train_ghost_gk.py`
  - Atomic mirror in `silly_kicks.atomic.tracking.features`

## [3.18.2] — 2026-05-25

### Fixed
- **game_id dtype mismatch between actions (int64) and frames (str):**
  Lakehouse SPADL pipelines produce `actions.game_id` as int64 (via
  `hash_native_id_to_bigint`) while `frames.game_id` retains native
  string values. Fixed 5 unguarded merge/lookup sites across
  `_defensive_line_at_actions`, `ball_carrier_at_action`,
  `add_team_shape`, and `_team_shape_at_actions` by casting both sides
  to `str` when dtypes differ. Same pattern as the PR-S44 fix in
  `_off_ball_runs` and `_line_breaking`.

## [3.18.1] — 2026-05-24

### Fixed
- **`slice_around_event` OOM on high-framerate tracking data:** Replaced
  O(A*F) cartesian merge on `period_id` with O(A*log F) per-period
  `np.searchsorted` on sorted frame times. At 25fps (Gradient Sports
  WC2022), the old implementation produced ~1.6 billion intermediate rows
  (12+ GiB allocation) and crashed; the new implementation materializes
  only the windowed subset. Affects `add_actor_pre_window` and
  `add_off_ball_runs` callers.

## [3.18.0] — 2026-05-23

### Added
- `compute_player_influence`: per-frame primitive computing off-ball xT and uniquely reachable area for all outfield players (TF-36 + TF-33)
- `add_player_influence`: action-coupled aggregator emitting 7 columns (`actor_reachable_area_m2`, `off_ball_xt_team`, `off_ball_xt_opponent`, `off_ball_xt_diff`, `reachable_area_team`, `reachable_area_opponent`, `reachable_area_diff`)
- `player_influence_xfns`: VAEP factory (21 columns across 3 gamestate slots)
- 5 per-Series helpers: `actor_reachable_area_m2`, `off_ball_xt_team`, `off_ball_xt_opponent`, `reachable_area_team`, `reachable_area_opponent`
- `PlayerInfluence` frozen dataclass return type

## [3.17.0] — 2026-05-23

### Changed
- **`infer_ball_carrier` ~30-50x faster via numba vectorization:** Replaced
  Python `iterrows()` inner loop with dense numpy pre-indexing + numba `@njit`
  kernel. A full GS WC2022 match (~200K frames) now completes in ~112ms
  (was ~31s). Python fallback when numba unavailable (~10-20x faster than
  iterrows). Public API unchanged; output bit-identical to previous
  implementation.

### Added
- `silly_kicks/tracking/_ball_carrier_numba.py` — optional `@njit(cache=True)`
  kernel for ball-carrier inference.
- Tests: 16 new tests in `test_ball_carrier_numba_parity.py` — Python kernel
  correctness (6), pre-index round-trip (2), numba parity (5), fallback path
  (3), plus 2 e2e tests (benchmark + real-data numba-vs-numpy parity).

## [3.16.2] — 2026-05-23

### Fixed
- **`derive_velocities` crashes on single-frame player-period groups:** `np.gradient`
  requires ≥2 points; a player-period with exactly 1 frame (real-world: GS WC2022
  match 3851, away #10, period 2) triggered `ValueError`. Guard now sets vx/vy/speed
  to NaN for ≤1-frame groups.

### Added
- Tests: `test_single_frame_group_no_crash`, `test_two_frame_group_produces_finite_velocity`,
  `test_mixed_group_sizes_single_and_normal` — 3 edge-case tests for short player-period groups.

## [3.16.1] — 2026-05-22

### Fixed
- **Gradient Sports out-of-bounds coordinates not clipped to pitch:** The GS
  converter was the only provider missing coordinate clipping to SPADL pitch
  bounds [0, 105] x [0, 68]. Lakehouse WC2022 evidence: 1,108/91,931 actions
  (1.2%) had OOB values (max ~5m x, ~8m y from throw-ins, GK overruns,
  tracking noise). Added `.clip()` after LTR normalization, matching all other
  converters.

### Added
- Tests: `TestGradientsportsCoordinateClipping` — 6 tests covering high/low OOB
  start coords, end coords after derive, away-team OOB after LTR flip, in-bounds
  guard, and full synthetic fixture zero-OOB integration test.
- Synthetic fixture: 4 OOB events added (pass high-x, cross high-y, clearance
  low-x/y, away-team pass high-y) with realistic values from lakehouse evidence.

## [3.16.0] — 2026-05-21

### Fixed
- **Gradient Sports NaN `time_seconds` on dedicated FOUL events:** Real GS data
  has NULL `startGameClock` on all 28 dedicated FOUL events (gameEventType=FOUL,
  possessionEventType=FO) across 13/64 WC2022 matches. The converter now imputes
  NaN `time_seconds` via forward-fill + back-fill within each period.

### Added
- Tests: `TestGradientsportsNanTimeSeconds` — 3 tests covering ffill imputation,
  bfill fallback for period-leading NaN, and full synthetic fixture smoke test.
- Synthetic fixture: dedicated FOUL event now has `startGameClock: null` matching
  real GS data; null-actor events (OTB+CH + FOUL+FO) now generated by the script
  rather than manually appended.

## [3.15.4] — 2026-05-21

### Fixed
- **Gradient Sports null-actor `team_id` crash:** Events with null `teamId`
  (OTB+CH challenges and FOUL+FO fouls with no actor, ~17 per WC 2022 match)
  caused `IntCastingNaNError` at `gradientsports.py:420`. Fixed by applying the
  same `Int64 → fillna(0) → int64` pattern already used for `player_id`.

### Added
- Tests: `TestGradientsportsNullActorEvents` — 3 unit tests covering OTB+CH
  and FOUL+FO null-actor events plus mixed-batch conversion.
- Tests: `test_synthetic_match_null_actor_events_convert` — E2E assertion on
  the synthetic fixture with two new null-actor events (gameEventId 46, 47).

## [3.15.3] — 2026-05-18

### Fixed
- **`play_left_to_right` ball-flip bug:** Ball rows were not flipped because
  they have `team_attacking_direction = None` (set by converters). Changed from
  per-team to per-period normalization: identify periods where home team has
  "rtl" direction, then flip ALL rows (players + ball) in those periods. This
  preserves all pairwise Euclidean distances between entities.
- **Downstream `_validate_ltr` validators:** Updated validators in
  `_cover_shadows.py`, `_defensive_line.py`, and `_off_ball_runs.py` to accept
  period-normalized frames (`{"ltr", "rtl"}` after `play_left_to_right`) instead
  of rejecting any "rtl" values. Validators now reject unexpected values or
  all-rtl-only frames.

### Added
- Tests: `test_play_left_to_right_ball_flip.py` — 16 regression tests covering
  ball-player spatial consistency, per-period normalization, edge cases (NaN,
  PSO, ball-only, string team IDs), and downstream validator compatibility.
- Tests: `test_invariant_spatial_consistency.py` — 9 physical-invariant tests
  (3 scenarios × 3 invariants) verifying `play_left_to_right` preserves all
  pairwise distances, normalizes home direction to "ltr", and keeps ball
  direction as None.

## [3.15.2] — 2026-05-17

### Fixed
- **Sportec/IDSSE shot goal detection:** DFL/IDSSE events use
  `shot_outcome_type = "successful"` for goals, but the converter only matched
  `"goal"` (legacy format). Real IDSSE data: all goals had `"successful"`, zero
  had `"goal"`. Now accepts both `"goal"` and `"successful"`.
- **Metrica SHOT compound-subtype goal detection:** SG1 compound subtypes like
  `"ON TARGET-GOAL"` and `"HEAD-ON TARGET-GOAL"` were not matched by
  `sub_raw == "GOAL"`. Replaced with `endswith("GOAL")` pattern (same approach
  as PR-S43's CHALLENGE fix).
- **Ward line-breaking game_id type mismatch:** `detect_line_breaking` dict-based
  frame lookup silently returned empty results when actions carried string
  game_ids and frames carried int game_ids (or vice versa). Now aligns types
  before lookup.
- **Off-ball runs line-break game_id type mismatch:** `_line_break_kernel` had
  the same dict-based lookup vulnerability plus a merge crash on mixed
  `game_id` dtypes. Now aligns types before both the merge and the lookup.

### Added
- Tests: `test_shot_outcome_type_mapping` (6 parametrized cases) in
  `test_sportec.py` covering all real IDSSE `shot_outcome_type` values.
- Tests: `TestMetricaShotCompoundSubtypes` (10 parametrized tests) in
  `test_metrica.py` covering all real SG1 compound SHOT subtypes.
- Tests: `TestGameIdTypeMismatch` (2 tests) in `test_line_breaking.py` covering
  matching and mismatched game_id types.
- Tests: `TestLineBreakKernelGameIdTypeMismatch` (1 test) in
  `test_off_ball_runs.py` covering the off-ball-runs variant.

## [3.15.1] — 2026-05-15

### Fixed
- **`_derive_end_coordinates` NaN guard:** The source-data guard only checked
  `end_x == start_x` (placeholder pattern). When end coordinates are NaN (Metrica
  SG1 set pieces: freekick_short, corner_short, throw_in, goalkick), the guard
  silently skipped derivation because `NaN != NaN` in pandas. Now also triggers on
  `end_x.isna()`.
- **Metrica CHALLENGE compound-subtype parsing:** The old exact-match
  `sub_raw == "WON"` caught 0/233 challenges on SG1 (all real subtypes are compound
  dash-separated: "TACKLE-WON", "GROUND-WON", "AERIAL-WON", etc.). Replaced with
  `endswith("WON")` / `endswith("LOST")` + interior-token decomposition for AERIAL
  and FAULT. Tackles, keeper claims, and fouls now surface correctly on SG1 data.
- **Metrica foul extraction from CHALLENGE-FAULT-LOST:** SG1 has no
  `type == "FAULT"` events; fouls are encoded as CHALLENGE subtypes containing
  "FAULT" + ending in "LOST" (e.g., "TACKLE-FAULT-LOST"). These now map to
  `foul` (fail) with card pairing working via the existing `_apply_card_pairs`.

### Added
- Tests: `TestNaNEndCoordinates` (5 tests) in `test_derive_end_coordinates.py`.
- Tests: `TestMetricaChallengeCompoundSubtypes` (19 parametrized tests) in
  `test_metrica.py` covering compound WON, FAULT-LOST, bare LOST, bare subtypes,
  priority edge cases, GK routing, and card pairing.

## [3.15.0] — 2026-05-15

### Fixed
- **End coordinates for single-position providers (Bug #7):** DFL/Sportec and
  Gradient Sports events carry only one `(x, y)` per event, so all SPADL actions
  had `end_x == start_x`. Replaced `_fix_clearances()` with shared
  `_derive_end_coordinates()` that overwrites `end_x`/`end_y` with the next
  action's `start_x`/`start_y` for 9 pass-class action types (pass, cross,
  throw_in, freekick_crossed, freekick_short, corner_crossed, corner_short,
  clearance, goalkick). Source-data guard preserves providers that already supply
  explicit end coordinates (StatsBomb, Opta, Wyscout, Metrica, SkillCorner).
  Period-boundary safe via `groupby("period_id").shift(-1)`. Eliminates ~90% of
  spurious dribble insertions on single-position providers.
- **GK features NULL for IDSSE/Sportec (Bug #2):**
  `add_pre_shot_gk_context(frames=...)` now uses `defending_gk_from_frames()` as
  a `.fillna()` fallback when events-based lookback finds no `keeper_save` within
  the search window. Shots within tracking coverage now reliably populate
  `defending_gk_player_id` and all downstream GK position/angle features.

### Added
- `_derive_end_coordinates()` in `silly_kicks.spadl.base` — shared end-coordinate
  derivation for all 8 converters (Sportec, StatsBomb, Opta, Wyscout, Metrica,
  SkillCorner, kloppy, Gradient Sports).
- `_DERIVE_END_TYPE_IDS` frozenset — canonical set of 9 action type IDs eligible
  for end-coordinate derivation.
- Unit tests: `tests/spadl/test_derive_end_coordinates.py` (15 tests).
- Integration tests: `tests/spadl/test_end_coord_integration.py` (6 tests across
  Sportec, StatsBomb, and Gradient Sports converters).
- Integration tests: `tests/spadl/test_gk_fallback_integration.py` (2 tests using
  paired IDSSE events + tracking fixture).
- Test fixture: `tests/datasets/idsse/paired_tracking.parquet` — real paired
  tracking data for match J03WMX (2 time windows, 2 GKs, ~37 KB).

### Removed
- `_fix_clearances()` from `silly_kicks.spadl.base` — superseded by
  `_derive_end_coordinates()` which covers all pass-class types, not just
  clearances.

## [3.14.1] — 2026-05-15

### Fixed
- **DAS team symmetry bug:** `_precompute_das_lookup` used `get_das()` which
  returns a single per-frame scalar, producing identical DAS values for both
  teams and `das_diff` always zero. Switched to `get_individual_das()` with
  per-team aggregation — `das_team` and `das_opponent` now correctly differ
  between attacking and defending teams.
- **Cover shadow man-marking over-absorption:** `_classify_man_markers` used
  greedy union — any defender within `man_mark_radius` (3.0m) of *any*
  attacker's behind-point was excluded from lane analysis. In compact
  formations, overlapping exclusion zones from 10 attackers absorbed most/all
  defenders, producing `blocking_score = 0`. Replaced with greedy
  nearest-first 1:1 assignment — each defender marks at most one attacker.

### Added
- `test_precompute_das_lookup_asymmetric` — CI test asserting per-team DAS
  asymmetry with realistic 11v11 spatial setup (was previously untested).
- `test_mutual_exclusion_shared_behind_points` — CI test asserting man-marking
  mutual exclusion with overlapping attacker behind-point zones.
- `test_zero_length_pass_returns_false` — CI test documenting expected Ward
  line-breaking behavior on zero-length trajectories (IDSSE/Sportec root
  cause: single event position produces `start == end`, `pass_len = 0 <
  min_pass_length = 3.0`).

## [3.13.0] — 2026-05-13

### Added
- **Pre-linking optimization (`links` kwarg):** All tracking `add_*` aggregators now accept
  an optional `links: pd.DataFrame | None = None` keyword argument. When provided, the
  function skips its internal `link_actions_to_frames` call and uses the caller-supplied
  pointers. Pipeline callers (e.g. lakehouse) pre-link once and pass `links` to all
  enrichment steps, reducing N × 2-5s to 1 × 2-5s per match (~25-65s saved per match
  at 14 enrichment steps). Fully backwards-compatible — existing callers are unchanged.
  Functions updated: `add_action_context`, `add_pre_shot_gk_position`,
  `add_pre_shot_gk_angle`, `add_actor_pre_window`, `add_pressure_on_actor`,
  `add_defensive_line`, `add_line_break`, `add_off_ball_context`, `add_team_shape`,
  `add_pitch_control`, `pitch_control_at_action`, `add_das`, `add_gk_influence`,
  `add_cover_shadows`, `add_pre_shot_gk_context` (spadl/utils), `pressure_on_actor`.
  Internal helpers also accept `links` for full thread-through.

## [3.12.0] — 2026-05-13

### Changed (BREAKING)
- **PFF → Gradient Sports rename:** All public API symbols, module paths, and runtime
  provider identifiers renamed from `pff` to `gradientsports` to reflect the PFF FC →
  Gradient Sports corporate rebrand.
  - `silly_kicks.spadl.pff` → `silly_kicks.spadl.gradientsports`
  - `silly_kicks.tracking.pff` → `silly_kicks.tracking.gradientsports`
  - `PFF_SPADL_COLUMNS` → `GRADIENTSPORTS_SPADL_COLUMNS`
  - `PFF_TRACKING_FRAMES_COLUMNS` → `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS`
  - `source_provider` column value: `"pff"` → `"gradientsports"`
  - `PreprocessConfig.for_provider("pff")` → `PreprocessConfig.for_provider("gradientsports")`
  - `GRADIENTSPORTS_TRACKING_DIR` env var replaces `PFF_TRACKING_DIR`
  - Example walkthrough: `pff_wc2022_walkthrough.py` → `gradientsports_wc2022_walkthrough.py`

  > **Note:** Historical CHANGELOG entries below this point retain the original "PFF"
  > terminology as they document the state of the codebase at the time of each release.
  > The rename applies from 3.12.0 onwards.

## [3.11.3] — 2026-05-12

### Fixed
- **xT NaN coordinate crash:** `ExpectedThreat.fit()` and `ExpectedThreat.rate()` no longer
  raise `IntCastingNaNError` when move actions (passes, dribbles, crosses) contain NaN
  coordinates. NaN-coordinate actions are silently dropped during transition matrix fitting
  and receive `NaN` ratings, consistent with the existing guard in `_count()`. Affects
  real-world data from Metrica (4 passes + 1148 other actions) and Sportec/IDSSE (160 fouls).

## [3.11.2] — 2026-05-11

### Fixed
- **Provenance column skip guard:** `add_action_context`, `add_pre_shot_gk_position`,
  `add_actor_pre_window`, and `add_pressure_on_actor` now skip merging linkage-provenance
  columns (`frame_id`, `time_offset_seconds`, `link_quality_score`, `n_candidate_frames`)
  when they already exist on the input DataFrame. Aligns with the idempotent pattern
  established by `add_defensive_line`, `add_team_shape`, `add_gk_influence`, and
  `add_cover_shadows` (PR-S27+). Without this guard, chaining multiple `add_*` enrichments
  produced `_x`/`_y` suffixed duplicate columns via `pd.merge`.

## [3.11.1] — 2026-05-11

### Fixed
- **Tracking namespace re-export gap:** `add_actor_pre_window`, `add_pressure_on_actor`,
  `pressure_on_actor`, and 13 related symbols (TF-2 + TF-3 per-Series helpers, xfn lists,
  pressure param types, pre-shot GK per-Series helpers) were exported from
  `silly_kicks.tracking.features` but never re-exported from `silly_kicks.tracking`.
  Oversight from PR-S25 (3.2.0). All 16 symbols now accessible at `silly_kicks.tracking.*`.

## [3.11.0] — 2026-05-11

### Added
- **TF-30: Cover Shadow Features — Lane Control + Blocking Score:**
  - `CoverShadowParams` frozen dataclass with all tunable physics constants
  - `LaneControlResult` frozen dataclass with per-line blocking probabilities + 3 decision flags
  - `ball_drag_time()` — Spearman 2017 quadratic air drag ball travel time
  - `player_tti()` — 3-phase react + accelerate + cruise time-to-intercept
  - `lane_control()` — per-(passer, receiver) corridor-discretized blocking probability
  - `compute_blocking_score()` — grid-based Voronoi counterfactual threat reduction
  - `add_cover_shadows()` — action-coupled aggregator (5 columns: `n_blocked_receivers`, `n_potential_receivers`, `blocking_score`, `blocked_threat_fraction`, `max_single_defender_blocking_score`)
  - `cover_shadow_xfns()` — VAEP factory (15 columns = 5 x 3 game states)
  - Atomic SPADL mirror
  - Ref: Cascioli, Wang, Stradiotti, Van Roy, Robberechts, Wouters, Jaspers & Davis 2025 (Hudl/DTAI, KU Leuven)

## [3.10.1] — 2026-05-10

### Fixed
- **Sportec CornerKick alias:** DFL XML uses `CornerKick` as the event tag but
  the DataFrame converter only accepted `Corner`. Callers passing raw XML tag
  names (e.g., lakehouse ingestion) had corner events silently dropped to
  `unrecognized_counts` (~16% of events in 7-match Bundesliga figshare
  collection). Both `Corner` and `CornerKick` are now accepted.
- **Sportec OtherBallAction handling:** DFL `OtherBallAction` events were
  silently dropped. Now mapped: `DefensiveClearance=true` produces a SPADL
  `clearance` action; other `OtherBallAction` events are mapped internally
  (appear in `mapped_counts`) but filtered as `non_action`.

## [3.10.0] — 2026-05-10

### Added
- **TF-15: GK influence primitives** (GKDV Layer 1):
  - `compute_gk_influence()` per-frame entry point with 3 primitives:
    threat-weighted pitch control share, uniquely reachable area, zone closing time
  - `Zone` dataclass with `six_yard_box()`, `near_post()`, `far_post()` factories
  - `GkInfluence` + `ZoneClosingTime` frozen return dataclasses
  - GK-specific kinematic parameters (`gk_reaction_time`, `gk_max_acceleration`)
  - Action-coupled: `add_gk_influence`, `gk_influence_xfns`, 4 per-Series helpers
  - Atomic SPADL mirror
  - Frame-precomputation cache in xfns factory
- **Prerequisite: `compute_tti`** exported as public API from `pitch_control`
- **Prerequisite: `select_back_line_players`** extracted from `_defensive_line.py`

### Fixed
- **TF-32 H1:** Independent dropna misalignment in `_line_breaking.py` (joint
  dropna prevents silent data corruption when opponent has valid x but NaN y)
- **TF-32 H2:** Extension-poisoning on `line_breaking_type` — `between_lines`
  now correctly dominates when both extension and through-player intersections
  occur in the same cluster
- **TF-32 M4:** Non-pass actions (shots, dribbles, etc.) now correctly produce
  pd.NA instead of being analyzed for line-breaking

## [3.9.0] — 2026-05-09

### Added
- **TF-31 Team Shape Envelope:** `compute_team_shape` per-frame primitive (7 metrics: n_outfield_players, centroid_x, centroid_y, convex_hull_area, team_length, team_width, stretch_index) + `add_team_shape` aggregator (14 action-coupled columns) + `team_shape_xfns` VAEP factory (36 columns). Ref: Clemente et al. 2013.
- **TF-32 Ward Line-Breaking:** `detect_line_breaking` per-action Ward-clustering line-breaking detection (3 columns: line_break__ward, lines_broken__ward, line_breaking_type__ward) + `LineBreakingParams` frozen dataclass + `line_breaking_ward_xfns` VAEP factory (9 columns). Extends `add_line_break` with `method="ward"` dispatch. Ref: Karakus & Arkadas 2025.

### Changed
- `add_line_break` gains `method` kwarg (`"threshold"` default, `"ward"` new) and `params` kwarg for Ward-specific parameters. Default behavior unchanged.
- `synthesize_actions` in test fixtures now gives pass actions a +20m forward trajectory offset (was zero-length).

## [3.8.0] — 2026-05-06

### Added

#### TF-28: DAS adapter — Dangerous Accessible Space

- `silly_kicks.tracking._das` module — thin adapter over `accessible-space` PyPI package (MIT)
- `get_das(frames)` → team-level AS/DAS per frame
- `get_individual_das(frames)` → per-player AS/DAS per frame
- `get_xc(passes, frames)` → expected pass completion per pass
- `derive_team_in_possession(frames, carrier)` → general tracking helper (in `_ball_carrier.py`)
- `das_at_action(actions, frames)` → action-coupled DAS
- `add_das(actions, frames)` → enrichment aggregator (`das_team`, `das_opponent`, `das_diff`)
- `das_xfns` — VAEP-compatible xfn list (single-pass precomputation, 9 columns)
- `[das]` optional extra in pyproject.toml (`accessible-space>=2.0,<3`)

#### TF-29: VAEP design-space variants — windowing + goalscore bias control

- `window` parameter on `scores()` / `concedes()`: `"action"` (default), `"possession"`, `"time"`
- `window_seconds` parameter for time-based windowing (default 15.0s)
- `xfns_default_no_goalscore` in `vaep/base.py`
- `hybrid_xfns_default_no_goalscore` in `vaep/hybrid.py`

#### Academic references (NOTICE)

- Bischofberger & Baca 2026 (Dangerous Accessible Space)
- Cascioli, Robberechts, Van Tente & Davis 2024-2025 (DTAI VAEP design-space blog series)

## [3.7.0] — 2026-05-05

### Added

#### TF-7: Pitch control models (Spearman / Fernandez-Bornn / Voronoi)

- `silly_kicks.tracking.pitch_control` subpackage — three-flavor spatial control computation
- `compute_pitch_control(frame, attacking_team_id, *, method, params, decompose, ball_position)` → `PitchControlSurface`
- `compute_pitch_control_at_points(frame, targets, attacking_team_id, *, method, params, ball_position)` → `np.ndarray`
- `PitchControlSurface` frozen dataclass with `at_point`, `at_points`, `control_in_region`, `player_share`, `player_surface`, `to_xarray` methods
- `SpearmanParams` / `FernandezBornnParams` / `VoronoiParams` frozen parameter dataclasses
- Optional numba acceleration via `_numba_kernels.py` (`@njit(cache=True)` mirrors of numpy kernels; 5-10x speedup)
- `pitch_control_at_action(actions, frames, *, method)` — action-coupled VAEP integration (NaN-safe, introspection-mode compatible)
- `add_pitch_control(actions, frames, *, method)` — enrichment aggregator
- `pitch_control_xfns(method)` / `pitch_control_default_xfns` — VAEP factory + default list
- Atomic-SPADL mirrors: `atomic.tracking.features.pitch_control_at_action`, `add_pitch_control`, `atomic_pitch_control_xfns`

#### Academic references (NOTICE)

- Spearman et al. 2017 (kinematic TTI pitch control)
- Fernandez & Bornn 2018 (bivariate-normal pitch control)
- Shaw & Sudarshan 2020 (ball-travel-time filter)

#### Architecture

- ADR-008: Pitch Control Subpackage Architecture

## [3.6.0] — 2026-05-05

### Added

#### TF-4: Off-ball runs + line-break detection

- `add_off_ball_runs(actions, frames, *, home_team_id)` — 4 off-ball-run columns: `n_off_ball_runners_pre_window`, `max_off_ball_run_displacement_pre_window`, `mean_off_ball_run_speed_pre_window`, `n_off_ball_runners_toward_goal_pre_window`
- `add_line_break(actions, frames, *, home_team_id)` — 2 line-break columns: `line_break` (nullable boolean), `n_attackers_behind_line` (Int64)
- `add_off_ball_context(actions, frames, *, home_team_id)` — umbrella aggregator adding all 6 columns
- `off_ball_context_xfns(home_team_id)` — VAEP factory (6 features x 3 states = 18 columns)

#### Academic references (NOTICE)

- Spearman 2018 (OBSO framework — off-ball-runs and line-break concepts)
- Power et al. 2017 (contextual passing risk/reward; line-breaking passes)

## [3.5.0] — 2026-05-05

### Added

#### TF-5: Per-frame ball-carrier inference

- `silly_kicks.tracking._ball_carrier.infer_ball_carrier(frames, *, tolerance_m, beta, gamma)` — per-frame ball-carrier identification via composite distance + velocity-toward-ball scoring with hysteresis. Returns one row per (game_id, period_id, frame_id) with carrier player_id, distance, and team_id. Distance-only fallback when vx/vy columns absent.
- `silly_kicks.tracking.features.ball_carrier_at_action(actions, frames, ...)` — action-coupled wrapper resolving ball carrier at each linked frame.

#### Consistency: `compute_defensive_line` game_id groupby

- `compute_defensive_line` now includes `game_id` in groupby + return schema, preventing cross-game collisions when processing multi-game batches.

#### Academic references (NOTICE)

- Bauer & Anzer 2021 (Data Mining and Knowledge Discovery) — velocity-toward-ball carrier identification heuristic.
- Vidal-Codina et al. 2022 (Sports Engineering) — hysteresis recommendation for ball-possession algorithms.

## [3.4.0] — 2026-05-05

### Added

#### TF-13: Frame-based defending-GK resolution

- `silly_kicks.tracking.features.defending_gk_from_frames(actions, frames)` — resolves defending GK `player_id` from tracking frames for all actions (not just shots). Fallback for events-based `defending_gk_player_id` NaN rows.

#### TF-14: Defensive-line geometry

- `silly_kicks.tracking._defensive_line.compute_defensive_line(frames, *, home_team_id, n=4)` — per-frame 6-column back-line geometry for both teams. Columns: `defensive_line_x`, `back_line_high_x`, `compactness_x`, `lateral_width`, `max_lateral_gap`, `back_n_count`. Supports fixed N ∈ {3, 4, 5} or `"adaptive"` via x-gap clustering (1.5× dominance rule).
- 6 per-Series action-coupled features: `defensive_line_x`, `back_line_high_x`, `compactness_x`, `lateral_width`, `max_lateral_gap`, `back_n_count`.
- `silly_kicks.tracking.features.add_defensive_line(actions, frames, *, home_team_id, n=4)` — aggregator enriching actions with 6 defensive-line columns + 4 linkage-provenance columns (skip-if-exists).
- `silly_kicks.tracking.features.defensive_line_xfns(home_team_id, *, n=4)` — VAEP xfn factory returning one multi-column transformer (6 cols × 3 states = 18 output columns).

#### NaN-safety CI

- `tests/test_enrichment_nan_safety.py` extended: auto-discovers `@nan_safe_enrichment` helpers in `silly_kicks.tracking.features` (≥6 registry floor); parametrized fuzz tests for all tracking helpers.

#### Academic references (NOTICE)

- Herold et al. 2022 (arXiv:2511.06191) — defensive-line height/compactness as match-outcome discriminators.
- Forcher et al. 2022 (arXiv:2511.00121) — back-line shape for pass-into-box models.
- FIFA EFI 2022 — practitioner 4-back defensive-line metrics.

## [3.3.0] — 2026-05-04

silly-kicks 3.3.0: Kloppy gateway `is_goalkeeper` hardening (PR-S26).

### Added

#### GK identification

- `silly_kicks.tracking._gk_identification.derive_goalkeepers` — B+ filtered algorithm for positional GK identification. Always-run design with agreement-based `is_goalkeeper_source` provenance. Handles: standard GKs (strict criteria: pa_dwell ≥ 0.40 AND dist < 20m), sweeper-keepers (rank-sum fallback), GK substitutions (multi-GK detection), brief outfielders (n_frames filter).
- `is_goalkeeper_source` column added to `TRACKING_FRAMES_COLUMNS` schema — values `"native"` (algorithm agrees with kloppy) or `"derived"` (algorithm overrode kloppy).
- `TrackingConversionReport.n_teams_gk_derived` — count of (game_id, team_id) pairs where source="derived".
- `TrackingConversionReport.derived_gk_picks` — audit trail: `dict[(game_id, team_id), list[player_id]]` of algorithm picks.

#### Kloppy gateway integration

- `silly_kicks.tracking.kloppy.convert_to_frames` now runs the GK identification algorithm on all Metrica/SkillCorner matches, fixing 21-50% → 100% GK detection rate.

#### Native path updates

- `silly_kicks.tracking.sportec.convert_to_frames` emits `is_goalkeeper_source="native"`.
- `silly_kicks.tracking.pff.convert_to_frames` emits `is_goalkeeper_source="native"`.

#### Architectural decision

- ADR-007: GK identification algorithm — documents thresholds, alternatives considered, and agreement-based source resolution design.

#### Test fixtures

- `tests/datasets/tracking/synthetic/gk_substitution.parquet` — multi-GK substitution scenario (2 teams × 2 GKs each).
- `tests/datasets/tracking/synthetic/sweeper_keeper.parquet` — sweeper-keeper fallback case (pa_dwell < 0.40).
- `tests/datasets/tracking/synthetic/brief_outfielder.parquet` — brief substitute exclusion case (n_frames filter).

## [3.2.0] — 2026-05-04

silly-kicks 3.2.0: TF-3 actor pre-window features + TF-2 multi-flavor pressure feature (PR-S25).

### Added

#### Tracking-aware features

- `silly_kicks.tracking.features.actor_arc_length_pre_window` — geometric arc-length of actor's path over the pre-action window (TF-3, default xfn). NOT Bauer & Anzer's filtered/threshold covered-distance feature; pure geometry, no sprint-intensity filtering.
- `silly_kicks.tracking.features.actor_displacement_pre_window` — net Euclidean displacement variant of TF-3 (window-first to window-last valid position).
- `silly_kicks.tracking.features.add_actor_pre_window` — aggregator emitting both columns + 4 provenance columns.
- `silly_kicks.tracking.features.actor_pre_window_default_xfns` — default xfn list (arc-length only).
- `silly_kicks.tracking.features.pressure_on_actor` — multi-flavor pressure feature (TF-2); methods: `andrienko_oval` (default; Andrienko 2017), `link_zones` (Link 2016), `bekkers_pi` (Bekkers 2024).
- `silly_kicks.tracking.features.add_pressure_on_actor` — aggregator emitting one `pressure_on_actor__<method>` per requested method.
- `silly_kicks.tracking.features.pressure_default_xfns` — default xfn list (Andrienko only, single default flavor).
- Atomic-SPADL parallel surface for all of the above (`silly_kicks.atomic.tracking.features.*`).

#### New module

- `silly_kicks.tracking.pressure` — multi-flavor pressure dispatch + per-method parameter dataclasses (`AndrienkoParams`, `LinkParams`, `BekkersParams`, `Method` Literal, `validate_params_for_method`).

#### Architectural decision

- ADR-005 §8 amendment: multi-flavor xfn column-naming convention (`<feature>__<method>` suffixes; default xfn list ships single default-method xfn; per-method params via flavor-specific frozen dataclass).

#### Attribution

- NOTICE entries: Andrienko 2017, Link 2016, Bekkers 2024 + BSD-3-Clause attribution to UnravelSports for the Bekkers TTI port.
- Vendored 30-line BSD-3-Clause excerpt at `tests/_vendored/unravelsports_tti.py` (test-only) so the Bekkers golden-master parity test runs unconditionally on Python 3.10+ without requiring the live `unravelsports` package (which targets Python 3.11+).

#### Test-only optional dependencies

- `unravelsports>=1.2` (extra `golden-master`) — preferred canonical source for the Bekkers golden-master parity test on Python 3.11+; the test falls back to `tests/_vendored/unravelsports_tti.py` when the live package isn't installed (e.g., Python 3.10).

#### Test infrastructure

- `tests/datasets/metrica/sample_match.parquet` regenerated with the 0–1 → 0–105/0–68 SPADL-frame rescale (matches `per_period_match.parquet` and the lakehouse `adapt_metrica_events_for_silly_kicks` adapter); previous fixture leaked Metrica's normalized 0–1 frame into bronze rows. `scripts/extract_provider_fixtures.py --provider metrica` now applies the rescale at extract time.
- Invariant tests (`tests/invariants/test_direction_of_play.py`, `test_gk_position.py`, `test_vaep_geometric_sanity.py`) hardened: `pytest.skip` paths replaced with explicit assertions or parametrize-list exclusions; shot counts now span all SPADL shot variants (`shot` / `shot_penalty` / `shot_freekick`) so converters' set-piece-composition rules don't mask the invariant; GK position invariant now also covers `keeper_pick_up`. Skipping count on the invariant suite went from 11 to 0.

## [3.1.0] — 2026-05-02

### Added

- **TF-6 — `sync_score`** (`silly_kicks.tracking.utils.sync_score`,
  `add_sync_score`, `LinkReport.sync_scores()`): per-action tracking↔events
  sync-quality scores. New columns when used via `add_sync_score`:
  - `sync_score_min`
  - `sync_score_mean`
  - `sync_score_high_quality_frac`
- **TF-8 — smoothing primitives** (`silly_kicks.tracking.preprocess.smooth_frames`,
  `derive_velocities`): Savitzky-Golay (canonical) and EMA smoothing of
  positional columns. Schema-additive output columns:
  - `x_smoothed`, `y_smoothed`
  - `vx`, `vy`, `speed`
  - `_preprocessed_with` (per-row provenance tag — load-bearing because
    `pandas.DataFrame.attrs` does not propagate through merge/concat/applyInPandas)
- **TF-9 — interpolation / gap-filling** (`silly_kicks.tracking.preprocess.interpolate_frames`):
  linear NaN gap-filling up to `max_gap_seconds` (cubic deferred to TF-9-cubic).
  Same schema as input — no new columns, just NaN cells replaced where the
  gap is short enough.
- **TF-12 — `pre_shot_gk_angle_*`** (`silly_kicks.tracking.features.add_pre_shot_gk_angle`,
  `pre_shot_gk_angle_to_shot_trajectory`, `pre_shot_gk_angle_off_goal_line`,
  `pre_shot_gk_angle_default_xfns`, `pre_shot_gk_full_default_xfns` + atomic
  mirror). New columns:
  - `pre_shot_gk_angle_to_shot_trajectory` (float64, radians, signed)
  - `pre_shot_gk_angle_off_goal_line` (float64, radians, signed)
- **`PreprocessConfig`** (`silly_kicks.tracking.preprocess.PreprocessConfig`):
  shared preprocessing config dataclass with `default()` / `for_provider(name)`
  factories and flag-based `is_default()`. Construction-time validator rejects
  `derive_velocity=True` + `smoothing_method=None`.
- **Tracking-converter optional `preprocess` kwarg** on
  `silly_kicks.tracking.sportec.convert_to_frames`,
  `tracking.pff.convert_to_frames`, and `tracking.kloppy.convert_to_frames`.
  Default `None` ⇒ zero behavior change. When set, applies interpolation /
  smoothing / velocity-derivation per the config; auto-promotes
  `PreprocessConfig.default()` to `PreprocessConfig.for_provider(<this_provider>)`,
  with `force_universal=True` + `UserWarning` fallback for unsupported providers.
- **Umbrella facade extension**: `silly_kicks.spadl.utils.add_pre_shot_gk_context`
  (and atomic mirror) now emits 6 GK-tracking columns when called with
  `frames=...` (the existing 4 from PR-S21 plus the 2 new TF-12 angles).
  The `frames=None` path is bit-identical to silly-kicks 2.9.0 — 4 columns.
  Lakehouse boundary tests asserting on the `frames=...` column-set need
  `expected_columns` extended by `pre_shot_gk_angle_to_shot_trajectory` and
  `pre_shot_gk_angle_off_goal_line`.
- **Empirical baselines**: `tests/fixtures/baselines/preprocess_baseline.json`
  + `preprocess_sweep_log.json` (per-provider stats across all 4 supported
  tracking providers including SkillCorner) +
  `scripts/probe_preprocess_baseline.py` +
  `scripts/regenerate_provider_defaults.py` (codegen pipeline replaces
  manual sync hand-edit).

### Changed

- **scipy is now a hard runtime dependency** (`scipy>=1.10.0`) — required by
  `tracking.preprocess` for Savitzky-Golay smoothing + derivative. Previously
  optional for `silly_kicks.xthreat` only.

### Notes

- ADR-005 amendment formalising the multi-flavor convention asymmetry
  (suffixed columns for VAEP xfns; canonical-single columns for preprocessing
  utilities) lands alongside the TF-2 `pressure_on_actor` PR (scheduled
  within 24-48 hours of PR-S24 merge — bounded deferral).
- Lakehouse pin bump: `silly-kicks>=3.1.0,<4`. No 3.0.x → 3.1.0 migration
  needed beyond the boundary-test column-set update above and (when adopting
  preprocessing inside Spark UDFs) declaring `_preprocessed_with` +
  smoothed/velocity fields explicitly in the `applyInPandas` `StructType`
  schema.

## [3.0.1] — 2026-05-02

### Breaking-correctness fix (PR-S23) — Sportec + Metrica per-period direction-of-play

`silly_kicks.spadl.sportec.convert_to_actions` and
`silly_kicks.spadl.metrica.convert_to_actions` now correctly handle
per-period-absolute bronze events (teams switching ends after halftime).
silly-kicks 3.0.0 declared these converters as `ABSOLUTE_FRAME_HOME_RIGHT`,
producing wrong-end SPADL output for half of every match. ADR-006 erratum
documents the corrected per-converter declaration table.

Callers must now pass per-period direction info via one of two paths
(otherwise `ValueError` with migration guidance):

```python
# Path A -- bool pair (preferred; matches PFF events + tracking-Sportec API)
actions, report = sportec.convert_to_actions(
    events,
    home_team_id="DFL-CLU-XXXXX",
    home_team_start_left=True,                     # from DFL MatchInformation.xml
    home_team_start_left_extratime=False,          # only when ET periods present
)

# Path B -- explicit mapping (escape hatch for arbitrary periods)
actions, report = metrica.convert_to_actions(
    events,
    home_team_id="Home",
    home_attacks_right_per_period={1: True, 2: False},
)
```

Trained VAEP / HybridVAEP / xT models on Sportec or Metrica data from
silly-kicks 3.0.0 must be re-trained on 3.0.1 output.

### Test infrastructure

- New per-period orientation fixtures committed at
  `tests/datasets/idsse/per_period_match.parquet` (Bassek et al. CC-BY 4.0)
  and `tests/datasets/metrica/per_period_match.parquet` (CC-BY-NC-4.0;
  same precedent as existing Metrica Sample Game 2 fixture). Both are
  excluded from the published wheel.
- New `test_per_team_per_period_shots_attack_high_x` parametrized over
  both new fixtures in `tests/invariants/test_direction_of_play.py`.
  Closes the invariant-density gap that let PR-S22's bug ship.
- 5 new `TestSportecPerPeriodKwargContract` + 5 new
  `TestMetricaPerPeriodKwargContract` negative-path tests for kwarg
  resolution policy.

### Detector hardening (TF-22)

`silly_kicks.spadl.orientation.detect_input_convention` no longer
false-positives `ABSOLUTE_FRAME_HOME_RIGHT` on sparse-shot
per-period-absolute matches. New guard: when no team has reliable shots
in ≥ 2 distinct periods, returns `convention=None, confidence="low"`.
Validator re-enabled at sportec / metrica / pff converter call sites
declaring `PER_PERIOD_ABSOLUTE`.

### Atomic-SPADL pathway

Smoke test added at `tests/atomic/test_atomic_orientation.py` verifying
the SPADL → atomic-SPADL composition preserves canonical-LTR. No
converter changes (atomic has no native sportec/metrica converter).

### Other

- `silly_kicks/__init__.py` `__version__` bumped from "1.0.2" (stale
  since at least 2.0.0) to "3.0.1" so it now matches `pyproject.toml`.
- `scripts/extract_provider_fixtures.py` gains `--variant {default, per_period}`
  flag for regenerating either fixture variant. Per-period extraction
  pulls from `bronze.idsse_events` / `bronze.metrica_events` on Databricks
  (env-var auth).
- `NOTICE` "Test Data Sources" section attributes the new IDSSE +
  Metrica Sample Game 1 fixtures.

## [3.0.0] — 2026-05-02

### Breaking — Correctness (PR-S22)

**Direction-of-play handling refactor.** The dual-mirror inversion that has
been present since v0.1.0 is fixed. SPADL canonical convention is "all teams
attack left-to-right" -- every team's actions at high-x in their own frame.
Every silly-kicks SPADL converter now produces this convention directly via
the new :func:`silly_kicks.spadl.to_spadl_ltr` dispatcher. Decision: ADR-006.

**Code-side regression window.** The bug was present in the native StatsBomb,
Wyscout, and Opta converters AND in `vaep.base.VAEP.compute_features` since
the v0.1.0 fork (verified `git show 0b29178`). The kloppy gateway acquired
the same code path in 1.7.0 but routed correctly because kloppy's transform
already normalised to absolute-frame-home-right.

**Consumer-artifact impact depends on which converter path each artifact's
data went through.** Categorically affected:

- Cached SPADL action tables derived from native ``silly_kicks.spadl.statsbomb``
  / ``wyscout`` / ``opta`` -- away-team ``(x, y)`` were mirrored to the wrong
  end of the pitch.
- Trained VAEP / HybridVAEP models built on Sportec / Metrica / kloppy-gateway
  / PFF SPADL -- VAEP feature engineering (now correctly free of the second
  mirror) inverted away-team rows in gamestates.
- Trained xG / xT models that consume polar / spatial features.
- Pre-computed xT grids derived from broken SPADL inputs (U-shaped instead of
  goal-monotonic).
- Tracking-aware features: ``add_action_context`` (PR-S20),
  ``add_pre_shot_gk_context`` (PR-S21).
- Any downstream model trained on action-coord features.
- Any test baseline / golden value calibrated on the prior pipeline.
- Any dataset published from silly-kicks output that mirrors SPADL or VAEP.

Per-consumer migration is the consumer's responsibility; this CHANGELOG enumerates
the categorical impact rather than specific consumer artifacts.

### Added

- **`silly_kicks.spadl.orientation`** (NEW module) — canonical direction-of-play
  primitives:
  - ``InputConvention`` enum: ``POSSESSION_PERSPECTIVE`` (StatsBomb, Wyscout),
    ``ABSOLUTE_FRAME_HOME_RIGHT`` (Sportec, Metrica, Opta, kloppy gateway),
    ``PER_PERIOD_ABSOLUTE`` (PFF).
  - ``to_spadl_ltr(actions, *, input_convention, home_team_id, ...)`` —
    single canonical normalizer; each converter calls it exactly once.
  - ``detect_input_convention(events, *, match_col, x_max, ...)`` — heuristic
    detector; tiered confidence (≥10 shots/group = high, 5-9 = medium, <5 =
    ambiguous defer).
  - ``validate_input_convention(events, declared, *, on_mismatch)`` — wired
    into every converter; warn by default, raise under
    ``SILLY_KICKS_ASSERT_INVARIANTS=1``. Surfaces upstream loader regressions.
- **`silly_kicks.vaep.base.VAEP.compute_features(..., frames_convention="absolute_frame")`**
  — explicit kwarg controlling tracking-frame normalisation.
- **`silly_kicks.tracking.{sportec,pff,kloppy}.convert_to_frames(..., output_convention=…)`**
  — opt-in ``"ltr"`` mode for callers wanting SPADL LTR tracking output
  directly. Default behaviour preserved (absolute_frame); ``None`` (legacy
  unspecified) emits ``DeprecationWarning`` recommending callers be explicit.
- **`tests/invariants/`** (NEW directory) — physical-invariant test layer
  parametrised across providers with real fixtures:
  - ``test_direction_of_play.py`` — per-team shots cluster at high-x,
    parametrised × ``xy_fidelity_version ∈ {1, 2}`` for StatsBomb.
  - ``test_vaep_geometric_sanity.py`` — VAEP shot dist < 50m AND xT
    goal-monotonic.
  - ``test_gk_position.py`` — GK actions cluster at defended (low-x) goal.
  - ``test_input_convention_detector.py`` — detector + validator semantics
    against real fixtures.

### Changed

- **`silly_kicks.spadl.statsbomb`, `wyscout`, `opta`, `sportec`, `metrica`,
  `kloppy`, `pff`** — every ``convert_to_actions`` now routes the
  direction-of-play step through ``to_spadl_ltr(input_convention=…)`` and
  emits canonical SPADL LTR. The ``input_convention`` declared by each
  converter is the load-bearing contract; ``validate_input_convention``
  surfaces violations.
- **`silly_kicks.spadl.opta.convert_to_actions`** — docstring contract
  added: the converter expects loader-pre-normalised absolute-frame data
  with NO per-period switching. Raw Opta f24 ships per-period switching;
  callers must pre-normalise upstream.
- **`silly_kicks.vaep.base.VAEP.compute_features`** — removed the inline
  ``play_left_to_right`` call (the dual-mirror that this CHANGELOG fixes).
  Converter output is already canonical SPADL LTR.
- **`silly_kicks.spadl.utils._finalize_output`** — debug-mode invariant
  assertion gated on ``SILLY_KICKS_ASSERT_INVARIANTS=1``: per-team shot mean
  start_x must be > field_length/2.
- **`silly_kicks.spadl.play_left_to_right`** + atomic-SPADL,
  ``silly_kicks.vaep.features.play_left_to_right`` + atomic-VAEP equivalents
  — docstrings updated. Functions are retained as public boundary helpers
  (absolute-frame → SPADL LTR) but no longer called by silly-kicks itself.

### Removed

- **`silly_kicks.spadl.base._fix_direction_of_play`** (private symbol) —
  replaced by ``silly_kicks.spadl.to_spadl_ltr``. Was only ever called by
  the converters themselves; no public API impact.

### Migration

Re-derive any cached artifact whose path went through an affected converter.
Specifically: re-derive SPADL action tables from raw events; re-train VAEP /
HybridVAEP models; re-compute xT grids; re-baseline empirical golden values;
re-publish any silly-kicks-derived datasets. The new validator surfaces input
convention mismatches as warnings; set ``SILLY_KICKS_ASSERT_INVARIANTS=1`` in
CI to promote them to failures.

## [2.9.0] — 2026-05-01

### Added — Pre-shot GK position + baselines backfill (PR-S21)

- **`silly_kicks.tracking.features`** — 4 GK-position helpers: `pre_shot_gk_x`,
  `pre_shot_gk_y`, `pre_shot_gk_distance_to_goal`, `pre_shot_gk_distance_to_shot`.
  Plus aggregator `add_pre_shot_gk_position(actions, frames) -> pd.DataFrame`
  that emits the 4 GK columns + 4 linkage-provenance columns. Decorated with
  `@nan_safe_enrichment` per ADR-003. Plus `pre_shot_gk_default_xfns` (4
  `lift_to_states` wrappers) for HybridVAEP integration.
- **`silly_kicks.atomic.tracking.features`** — atomic-SPADL parity with the
  same public surface (`atomic_pre_shot_gk_default_xfns`). Mirrors the standard
  surface with atomic-shaped column reads (`x, y`) and atomic shot type ids
  (`{shot, shot_penalty}` — atomic does not recognize `shot_freekick`).
- **`silly_kicks.spadl.utils.add_pre_shot_gk_context(*, frames=None)`** — additive
  optional `frames` kwarg. When supplied, emits 4 GK-position columns + 4
  provenance columns by lazy-importing the canonical compute (preserves
  ADR-005 §5 no-cycle invariant). When `frames=None` (default), behavior is
  bit-identical to silly-kicks 2.8.0 — no frames-related columns appear.
  Backward-compat pinned by golden-fixture test.
- **`silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context`** — atomic mirror
  of the same `frames=None` extension.
- **`silly_kicks.tracking._kernels._pre_shot_gk_position`** (private) —
  schema-agnostic compute kernel shared between standard and atomic surfaces.
- **`silly_kicks.tracking.feature_framework.ActionFrameContext`** gains
  `defending_gk_rows: pd.DataFrame` field (default-factory empty DataFrame —
  preserves direct construction backward-compat).
- **`scripts/regenerate_action_context_baselines.py`** — one-shot regenerator
  for `*_expected.parquet` files + `empirical_action_context_baselines.json`.
- **`tests/datasets/tracking/action_context_slim/{provider}_expected.parquet`**
  — per-provider expected output committed for the bit-exact per-row
  regression gate (4 providers).
- **`tests/tracking/_provider_inputs.py`** — shared loader/synthesizer for the
  regenerator and CI gate; keeps both in sync.
- **`tests/tracking/test_action_context_expected_output.py`** — bit-exact
  per-row regression gate (4 providers).
- **`tests/tracking/test_empirical_action_context_baselines.py`** — JSON shape
  gate + JSON-vs-parquet consistency gate.

### Changed

- **`silly_kicks.spadl.utils.add_pre_shot_gk_context`** + atomic mirror —
  bug-fix: `defending_gk_player_id` output column now preserves the input
  `player_id` dtype. Numeric `player_id` (canonical SPADL_COLUMNS:
  PFF / StatsBomb / Opta / Wyscout / Metrica) → `float64` NaN-coded (unchanged).
  Object/string `player_id` (`KLOPPY_SPADL_COLUMNS` / `SPORTEC_SPADL_COLUMNS`
  schema) → `object` dtype with `None` for unidentified rows. Previous
  unconditional `int(gk_id_raw)` cast crashed on string Sportec player_ids;
  surfaced by PR-S21's TF-11 regression-gate exercising real-shot rows on
  Sportec data.
- **`tests/datasets/tracking/empirical_action_context_baselines.json`** —
  all 256 percentile slots backfilled (4 percentiles × 8 features × 4 providers).
  Per-row gate exercises real GK-position computation on at least one shot
  per provider (synthesizer in `tests/tracking/_provider_inputs.py` stamps a
  synthetic keeper_save → shot pair anchored on real frame goalkeeper data
  so the events-side helper populates `defending_gk_player_id` and the
  tracking aggregator emits non-NaN GK position).
- **`NOTICE`** — Anzer & Bauer (2021) entry description expanded to enumerate
  defending-GK-position alongside player_speed and distance-to-defender.
- **`TODO.md`** — TF-1 + TF-11 marked SHIPPED. PR-S21 active-cycle entry.
  Bundled National Park additions: TF-12 (`pre_shot_gk_angle_*`), TF-13
  (frame-based GK identification fallback), TF-14 (defensive-line features).

### Removed

- **4 vestigial `test_placeholder` stubs** (National Park cleanup): the
  `TestKloppyE2E.test_placeholder` (`test_kloppy.py`),
  `TestSpadlConvertorE2E.test_placeholder` (`test_opta.py`, `test_wyscout.py`),
  and `TestSpadlConvertor.test_placeholder` (`test_statsbomb.py`) classes
  were inert `pytest.skip()` calls inherited from the v0.1.0 socceraction
  fork (the original DataLoader classes — `OptaLoader` / `StatsBombLoader` /
  `PublicWyscoutLoader` / `KloppyLoader` — were removed at fork time but the
  e2e test scaffolds were left behind as no-op skip stubs). Plus the
  unreferenced `pytestmark_e2e` module attribute in `test_opta.py`. Net
  effect: `pytest -m e2e` now runs 12 PASSED / 0 SKIPPED instead of
  12 PASSED / 4 SKIPPED — the SKIPPED column is no longer a hiding place
  for genuine missing-fixture failures.

### Notes

- No breaking changes. PR-S21 ships entirely within ADR-005's locked
  architecture; no new ADR.
- Per-Series GK helpers (`pre_shot_gk_x` etc.) silently emit all-NaN when
  `defending_gk_player_id` is absent from `actions` — required by VAEP's
  `feature_column_names` introspection path. The aggregator
  `add_pre_shot_gk_position` raises `ValueError` (user-direct boundary).
  Documented in helper docstrings + `pre_shot_gk_default_xfns`.

## [2.8.0] — 2026-05-01

### Added — Tracking-aware action_context features (PR-S20)

- **`silly_kicks.tracking.features`** --- public per-feature surface for
  standard SPADL: `nearest_defender_distance`, `actor_speed`,
  `receiver_zone_density`, `defenders_in_triangle_to_goal`. Plus aggregator
  `add_action_context(actions, frames, *, receiver_zone_radius=5.0) -> pd.DataFrame`
  that enriches input actions with the 4 features + 4 linkage-provenance
  columns (`frame_id`, `time_offset_seconds`, `link_quality_score`,
  `n_candidate_frames`). Decorated with `@nan_safe_enrichment` per ADR-003.
  Plus `tracking_default_xfns` (4 `lift_to_states` wrappers) for
  HybridVAEP integration.
- **`silly_kicks.atomic.tracking.features`** --- atomic-SPADL parity with
  the same public surface (`atomic_tracking_default_xfns`). Mirrors the
  standard surface with atomic-shaped column reads (`x, y, dx, dy`).
- **`silly_kicks.tracking.feature_framework`** --- `ActionFrameContext`
  frozen dataclass + `lift_to_states` (lifts an `(actions, frames) -> pd.Series`
  helper to a `(states, frames) -> Features` transformer). Re-exports
  `frame_aware`, `is_frame_aware`, `Frames`, `FrameAwareTransformer`.
- **`silly_kicks.tracking._kernels`** (private) --- schema-agnostic compute
  kernels shared between standard and atomic public surfaces. Per
  ADR-005 §3 (kernel-extraction pattern).
- **`silly_kicks.tracking.utils._resolve_action_frame_context`** (private)
  --- builds the linked-context structure (linkage pointers + per-action
  actor row + opposite-team frame rows) once per `add_action_context()` call.
- **`silly_kicks.vaep.feature_framework`** --- extended with `frame_aware`
  decorator, `is_frame_aware` predicate, and `Frames` / `FrameAwareTransformer`
  type aliases. Marker-decorator pattern parallels the existing
  `@nan_safe_enrichment` contract (ADR-003).
- **`silly_kicks.vaep.base.VAEP.compute_features` / `rate`** --- additive
  `frames=None` keyword-only parameter. Frame-aware xfn dispatch via
  `is_frame_aware`. `HybridVAEP` and `AtomicVAEP` inherit the extension
  automatically (no code changes in their files). Symmetric LTR-normalization
  via lazy import of `tracking.utils.play_left_to_right` only when
  `frames is not None` (no module-import-time vaep <-> tracking cycle).
- **`silly_kicks._nan_safety`** --- new `is_nan_safe_enrichment(fn)` peer
  predicate to the existing `nan_safe_enrichment` decorator. Mirrors the
  new `is_frame_aware` introspection API.
- **ADR-005** ([docs/superpowers/adrs/ADR-005-tracking-aware-features.md](docs/superpowers/adrs/ADR-005-tracking-aware-features.md))
  --- tracking-aware feature integration contract. Captures the seven
  cross-cutting decisions PR-S20 introduces so PR-S21+ tracking-aware
  features inherit them without re-litigation.
- **`NOTICE`** --- canonical academic-attribution record at repo root,
  mirroring the lakehouse pattern. Cross-linked from `README.md` and
  `CLAUDE.md`. Cites Lucey et al. (2014), Anzer & Bauer (2021),
  Spearman (2018), Power et al. (2017), Pollard & Reep (1997) for the 4
  PR-S20 features, plus the foundational SPADL / VAEP / Atomic-SPADL / xT
  literature.
- **`TODO.md` restructured** to the lakehouse-style "On Deck" table.
  Eleven follow-up tracking-aware features (TF-1..TF-10) tracked with
  Size / Source / Notes columns and academic citations; TF-11 tracks the
  baselines-JSON backfill.
- **Loop 0 lakehouse probe** --- `scripts/probe_action_context_baselines.py`
  pulls slim-slice action+frame parquets per provider into
  `tests/datasets/tracking/action_context_slim/` (sportec / metrica /
  skillcorner; ~10 actions + linked frames each). Probe + outputs
  committed; real datasets are not. Backbone for the cross-provider
  parity test.
- **Tier-3 cross-provider parity test** ---
  `tests/tracking/test_action_context_cross_provider.py` runs
  `add_action_context` against the lakehouse-derived slim parquets per
  provider; asserts bounds + linkage rate >= 95% + actor_speed populated
  >= 80%.
- **e2e real-data sweep** ---
  `tests/tracking/test_action_context_real_data_sweep.py` (4
  e2e-marked tests, env-gated). Mirrors PR-S19's sweep shape: PFF via
  `PFF_TRACKING_DIR`; IDSSE / Metrica / SkillCorner via Databricks SQL.
  Skips with explicit reason on missing env.

### Backward compatibility

- All existing call sites (`v.compute_features(game, actions)`,
  `v.rate(game, actions)`) work verbatim --- `frames=None` is the
  default and walks the same code path. Regression-tested in
  `test_compute_features_frames_none_is_regression_equivalent`.
- No changes to `xfns_default`, `hybrid_xfns_default`, or atomic
  `xfns_default`. Tracking-aware features must be opted in by appending
  `tracking_default_xfns` (or `atomic_tracking_default_xfns`) to the
  caller's xfns list.

## [2.7.0] — 2026-04-30

### Added

- **`silly_kicks.tracking` namespace** --- first-class tracking-data
  support, parallel to `silly_kicks.spadl`. Hexagonal pure-function
  contract: `convert_to_frames(...) -> tuple[pd.DataFrame,
  TrackingConversionReport]`, zero I/O, zero global-state mutation.
  Nineteen-column long-form canonical schema
  (`TRACKING_FRAMES_COLUMNS`), per-provider dtype variants
  (`KLOPPY_TRACKING_FRAMES_COLUMNS`, `SPORTEC_TRACKING_FRAMES_COLUMNS`,
  `PFF_TRACKING_FRAMES_COLUMNS`), 105 x 68 m SPADL coordinates,
  long-form ball-row encoding (`is_ball=True`), `team_attacking_direction` /
  `ball_state` / `speed_source` provenance columns.
- **Four-provider adapter coverage** --- Sportec/IDSSE
  (`silly_kicks.tracking.sportec`, native), PFF
  (`silly_kicks.tracking.pff`, native), Metrica + SkillCorner
  (`silly_kicks.tracking.kloppy`, gateway via `kloppy.TrackingDataset`).
  PFF native is preferred over kloppy's PFF tracking parser for
  symmetry with `silly_kicks.spadl.pff` (PR-S18) and shared use of the
  `_direction.home_attacks_right_per_period` helper.
- **Linkage primitive**
  (`silly_kicks.tracking.utils.link_actions_to_frames` +
  `slice_around_event`) --- the load-bearing cross-pipeline operation
  that PR-S20+ tracking-aware features will build on. Returns pointer
  DataFrame plus `LinkReport` audit. Default tolerance 0.2 s, pinned
  by an explicit default-stability test.
- **Hybrid speed policy** --- adapters trust native speed where
  provided (PFF, Sportec); derive via `_derive_speed` (per-player
  groupby + diff) where missing (Metrica, SkillCorner). The
  `speed_source` column records provenance.
- **Empirical-probe-driven synthetic fixtures** ---
  `scripts/probe_tracking_baselines.py` measures real-data statistics
  (frame rates, NaN-rate-per-column, off-pitch tail rates,
  ball-visibility rates, distance-to-ball percentiles) from the
  lakehouse mart + local PFF; the committed JSON baseline at
  `tests/datasets/tracking/empirical_probe_baselines.json` parameterizes
  the per-provider synthetic generators. `realistic.parquet` fixtures
  inject baseline-calibrated edge cases (off-pitch tail, ball-out
  interval, ball-x throw-in tail) for CI; deterministic
  `tiny.parquet` / `medium_halftime.parquet` remain available for
  exact-answer unit tests.
- **`tests/test_tracking_real_data_sweep.py`** --- e2e-marked sweep
  exercising all four adapters against real data (local PFF JSONL.bz2 +
  lakehouse-derived Sportec / Metrica / SkillCorner samples). Skipped
  in CI; run locally before each tracking PR's single commit.
- **ADR-004**
  (`docs/superpowers/adrs/ADR-004-tracking-namespace-charter.md`) ---
  silly_kicks.tracking namespace charter; nine invariants locking the
  schema + adapter taxonomy + linkage contract for PR-S20+ to inherit.
- **`pyproject.toml`** --- `kloppy` optional minimum bumped to >= 3.18.0
  (kloppy 3.18 ships Metrica + SkillCorner tracking parsers used by the
  gateway). Pytest `pythonpath` config now includes `["", "tests"]` so
  per-provider synthetic-fixture generators are importable in test code
  via `datasets.tracking.<provider>.generate_synthetic`.

### Changed

- **`silly_kicks/spadl/pff.py`** --- the per-period direction lookup
  (`home_attacks_right_per_period`) is extracted into
  `silly_kicks/tracking/_direction.py` so events PFF, tracking PFF,
  and tracking Sportec adapters share one implementation. Pure
  refactor; the events test suite (127 tests) passes unchanged.

### Deferred

Tracking-aware features deferred to follow-up scoping cycles, in
priority order (per ADR-004 invariant 9): `action_context()` (PR-S20,
target 2.8.0), `pressure_on_carrier()`, `infer_ball_carrier()`,
`sync_score()`, pitch-control models (Spearman / Voronoi), smoothing
primitives (Savitzky-Golay, EMA), multi-frame interpolation /
gap filling, ReSpo.Vision adapter (licensing-gated).

## [2.6.0] — 2026-04-30

### Added

- **`silly_kicks.spadl.pff`** — first-class PFF FC / Gradient Sports
  events-data converter. Hexagonal pure-function contract (events
  DataFrame in, SPADL DataFrame + ConversionReport out, zero I/O).
  Mirrors the sportec / metrica converter shape. Dispatch table covers
  PFF's hierarchical event vocabulary (`gameEvents` × `possessionEvents`
  + `set_piece_type`): pass / cross / shot / clearance / dribble (BC) /
  tackle (CH) / keeper_save+keeper_pick_up (RE) / bad_touch (TC) +
  set-piece compositions (kickoff / open play / corner / free kick /
  throw-in / goal kick / penalty) + foul row synthesis with card
  result mapping. Excludes `OUT` / `SUB` / period-boundary / `OTB+IT`
  rows with full ConversionReport audit trail.
- **`silly_kicks.spadl.PFF_SPADL_COLUMNS`** — extended output schema:
  `SPADL_COLUMNS` + four nullable `Int64` tackle-passthrough columns
  (`tackle_winner_player_id`, `tackle_winner_team_id`,
  `tackle_loser_player_id`, `tackle_loser_team_id`) per ADR-001.
  `Int64` (pandas nullable) is a deliberate dtype departure from
  `SPORTEC_SPADL_COLUMNS`'s `object` dtype: PFF identifiers are integers
  whereas kloppy hands sportec strings.
- **Per-period direction-of-play normalization** — first silly-kicks
  converter requiring perspective-real coordinate handling. Two new
  parameters (`home_team_start_left`, `home_team_start_left_extratime`)
  carry the metadata-derived flip information per period.
- **`tests/datasets/pff/`** — synthetic match fixture
  (`synthetic_match.json`) plus deterministic generator
  (`_generate_synthetic_match.py`). Synthetic-only test policy until
  PFF licensing for redistributable real-data slices is confirmed.
- **`docs/examples/pff_wc2022_walkthrough.py`** — end-to-end pipeline
  demonstration (documentation, not test). Reads from a user-supplied
  PFF directory and walks events → SPADL → Atomic-SPADL → coverage /
  boundary metrics → VAEP labels.
- **`TODO.md` Tracking namespace entry** — captures the deferred
  `silly_kicks.tracking.*` design with verified luxury-lakehouse prior
  art (3 providers / 20 matches / ~38M player-frames in
  `soccer_analytics.dev_gold.fct_tracking_frames` as of 2026-04-30) and
  library-native architectural rules.

### Changed

- **`silly_kicks.spadl._finalize_output`** recognizes pandas extension
  dtypes (`Int64`, `Float64`, `boolean`, `string`, etc.) on schema
  entries — small surface-area generalization, fully backwards-
  compatible with existing object/int64 dtype handling. Required for
  `PFF_SPADL_COLUMNS` `Int64` tackle columns.
- **`tests/spadl/test_cross_provider_parity.py`** — PFF added as a
  parametrize entry; participates in the keeper-action emission gate,
  schema-shape gate, and ADR-001 team_id-mirror gate alongside the five
  pre-existing converters.
- **Pre-release empirical validation** — converter validated against the
  full WC 2022 dataset (64 matches, 144,541 events → 91,931 SPADL actions,
  zero conversion failures, zero unrecognized vocabulary). The sweep
  surfaced 6 vocabulary patterns the hand-authored synthetic-fixture suite
  missed (OFF / ON / G / THIRDKICKOFF / FOURTHKICKOFF game_event_types and
  OTB+empty initialNonEvent markers); all are now in the converter's
  excluded vocabulary, exercised by the synthetic fixture, and asserted by
  test_pff.py. Also surfaced a real-data schema detail: PFF stores
  ``fouls`` as a single dict per event (not a JSON array, contrary to
  initial fixture authoring); fixture + loaders updated. Standalone
  ``FOUL`` gameEventType events with ``possessionEventType="FO"`` now
  convert in-place to the canonical foul SPADL row (no phantom non_action
  parent).

## [2.5.0] — 2026-04-30

### Added

- **`silly_kicks._nan_safety.nan_safe_enrichment`** — marker decorator
  declaring an enrichment helper satisfies the NaN-safety contract
  (ADR-003). Sets `fn._nan_safe = True`; CI gates auto-discover decorated
  helpers via this attribute.
- **`goalkeeper_ids: set | None = None`** keyword-only parameter on
  `silly_kicks.spadl.utils.add_gk_role` and
  `silly_kicks.atomic.spadl.utils.add_gk_role`. When provided,
  distribution-detection extends with two additional matching rules:
  (a) `current player_id ∈ goalkeeper_ids` AND prev keeper-type same-team;
  (b) NaN-team fallback — both player_ids NaN AND same team_id AND prev
  keeper-type. Closes the lakehouse coverage gap on IDSSE/Metrica data
  with sparse player attribution. When `None` (default), behavior is
  byte-for-byte unchanged.
- **`tests/test_enrichment_nan_safety.py`** — auto-discovered NaN-fuzz
  test (15 cases). Parametrizes over every `@nan_safe_enrichment` helper
  × synthetic NaN-laced SPADL fixture; asserts no crash + sensible
  defaults. Includes registry-floor sanity assertions that catch silent
  discovery breakage.
- **`tests/test_enrichment_provider_e2e.py`** — auto-discovered
  cross-provider e2e regression (21 cases). Parametrizes over every
  `@nan_safe_enrichment` standard helper × vendored fixtures from
  StatsBomb / IDSSE / Metrica; atomic helpers run on the
  StatsBomb-derived atomic-SPADL fixture.
- **`tests/test_gk_role_goalkeeper_ids.py`** — feature tests for the new
  `goalkeeper_ids` parameter (8 cases): backward-compat, rule (a)
  known-GK match, rule (b) NaN-team fallback, edge cases (atomic, empty
  set, team-boundary respect).
- **`docs/superpowers/adrs/ADR-003-nan-safety-enrichment-helpers.md`** —
  formalizes the NaN-safety contract for public enrichment helpers,
  alternatives considered, and the registry-floor sanity assertion as
  the bulletproof for the auto-discovery mechanism.
- **CLAUDE.md "Key conventions" amendment** pointing to ADR-003.

### Fixed

- **`silly_kicks.spadl.utils.add_pre_shot_gk_context`** —
  `ValueError: cannot convert float NaN to integer` at line 543 when
  the most-recent defending-keeper-action's `player_id` is NaN
  (e.g. IDSSE bronze data with sparse player attribution). Surfaced
  2026-04-30 by the luxury-lakehouse `compute_spadl_vaep` task. Fix:
  detect NaN before the `int(...)` cast; `continue` to next shot
  (defending_gk_player_id stays NaN per the function's documented
  contract). Symmetric fix at
  `silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context` line 826.
- **`silly_kicks.spadl.utils.add_gk_distribution_metrics`** — latent
  `ValueError: cannot convert float NaN to integer` at lines 374-377
  on `.astype(int)` zone-binning when a distribution-eligible row has
  NaN coordinates. Fix: filter `eligible` mask by `np.isfinite(...)`
  on all four coords. Symmetric fix at
  `silly_kicks.atomic.spadl.utils.add_gk_distribution_metrics`
  lines 665-668.
- **`silly_kicks.spadl.utils.coverage_metrics`** (defensive) — same
  `int(NaN)` crash class on `int(tid)` at line 1074 if input has NaN
  `type_id`. Fix: NaN guard before the cast; NaN type_ids tally as
  "unknown". Symmetric fix at
  `silly_kicks.atomic.spadl.utils.coverage_metrics` line 1036. Not
  under ADR-003 (TypedDict-returning, not enrichment helper) — fixed
  while we're here.

### Changed

- 10 public enrichment helpers (5 standard + 5 atomic) decorated with
  `@nan_safe_enrichment`: `add_possessions`, `add_names`, `add_gk_role`,
  `add_gk_distribution_metrics`, `add_pre_shot_gk_context` × 2 packages.

### Notes

- **Hyrum's Law surface:** `add_gk_role.__signature__` gains the new
  `goalkeeper_ids` keyword-only parameter. Consumers using
  `inspect.signature(add_gk_role)` would see the addition. Documented
  in ADR-003 as accepted exposure.
- **Test count:** 884 → 928 passing, 4 deselected (+44 net delta:
  15 fuzz + 21 e2e + 8 goalkeeper_ids feature tests). Pyright clean
  (0 errors / 0 warnings / 0 informations).
- Future direction: nullable-Int64 dtype migration for `player_id` /
  `team_id` columns is the long-term answer to type-level NaN-safety;
  out of scope for this PR (ADR-003 § Notes / Future direction).

## [2.4.0] — 2026-04-30

### Added

- **`silly_kicks.vaep.feature_framework`** — new public module holding the 7
  framework primitives both standard and atomic VAEP feature stacks build on:
  4 type aliases (`Actions`, `Features`, `FeatureTransfomer`, `GameStates`),
  `gamestates`, `simple`, and the promoted helper
  `actiontype_categorical(actions, spadl_cfg)`. Cross-package framework
  boundary now has a name; atomic-VAEP no longer reaches into
  `vaep.features.core` for framework primitives.
- **`actiontype_categorical(actions, spadl_cfg)`** — promoted from the
  previously-private `_actiontype` helper in `vaep.features.core` to a public,
  SPADL-config-parameterized framework helper. Both standard-VAEP and
  atomic-VAEP wrap it with `@simple` to produce their respective `actiontype`
  feature transformers. Drops the implicit-None config fallback (the function
  is meaningless without a config); positional `spadl_cfg` parameter.
  Examples-section docstring per the public-API discipline.
- **`tests/vaep/test_feature_framework_layout.py`** — 7-case framework-layout
  lock (T-D). Asserts each framework primitive's canonical home is
  `silly_kicks.vaep.feature_framework`.
- **`docs/superpowers/adrs/ADR-002-shared-vaep-feature-framework-boundary.md`** —
  captures the framework-extraction decision, the 4 alternatives considered,
  and the `_actiontype → actiontype_categorical` rename rationale.

### Changed

- **`silly_kicks.vaep.features.core` slimmed** to its standard-SPADL-specific
  helpers (`play_left_to_right`, `feature_column_names`); re-exports the
  framework primitives from `silly_kicks.vaep.feature_framework` so existing
  `from silly_kicks.vaep.features.core import gamestates` paths continue to
  resolve (Hyrum's-Law preservation).
- **`silly_kicks.atomic.vaep.features` imports framework directly from
  `vaep.feature_framework`** (no longer reaches into `vaep.features.core`);
  per-concern feature reuse from `bodypart` / `context` / `temporal` is
  preserved (intentional verbatim code-share, not framework leak).
- **`silly_kicks.vaep.features.actiontype` body updated** to call
  `actiontype_categorical(actions, spadlcfg)` instead of the private
  `_actiontype(actions)` (the latter relied on an implicit-None spadlcfg
  fallback; the new call passes spadlcfg explicitly — same resolved
  behaviour).
- **T-A backcompat (`tests/vaep/test_features_backcompat.py`)** gains one row
  for `actiontype_categorical`. 33 → 34 cases.
- **T-B layout (`tests/vaep/test_features_submodule_layout.py`)** drops the 6
  framework rows now living outside the features package. 33 → 27 cases.
- **T-C atomic-coupling (`tests/atomic/test_features_per_concern_import.py`)
  rewritten** to forbid `vaep.features.core` import for framework primitives
  and require import from `vaep.feature_framework`. Retains the existing
  package-root-import forbid + 3 per-concern-import requirements.
- **Examples-gate file list** (`tests/test_public_api_examples.py`) adds
  `silly_kicks/vaep/feature_framework.py`. 26 → 27 cases.

### Removed

- **`silly_kicks.vaep.features.core._actiontype`** — promoted to public
  `actiontype_categorical(actions, spadl_cfg)` in the new framework module.
  Was a leading-underscore-private symbol; never in `__all__`; never
  documented as public surface.

### Closed

- **TODO A9** — `atomic/vaep/features.py` per-concern coupling — closed via
  framework extraction (the trigger-condition resolution from PR-S15's
  deferral). The `## Architecture` section of `TODO.md` is now empty.
  See ADR-002.

### Notes

- **Hyrum's Law surface:** `gamestates.__module__` (and `simple.__module__`)
  flips from `silly_kicks.vaep.features.core` to
  `silly_kicks.vaep.feature_framework`. Consumers introspecting via
  `inspect.getmodule(gamestates)` would see the new value. Documented in
  ADR-002 as accepted exposure.
- **Test count:** 881 → 884 passing, 4 deselected (+3 net delta: +1 T-A row,
  -6 T-B rows, +7 T-D cases, +1 Examples-gate parametrize). Pyright clean
  (0 errors / 0 warnings / 0 informations).

## [2.3.0] — 2026-04-30

### Changed

- **`silly_kicks.vaep.features` decomposed from a 1170-line monolith into a
  package** of 8 concern-focused submodules (`core`, `actiontype`, `result`,
  `bodypart`, `spatial`, `temporal`, `context`, `specialty`). Hybrid visibility:
  every previously-public symbol remains importable via the package path
  (`from silly_kicks.vaep.features import startlocation` keeps working
  unchanged); submodule paths are also importable for advanced/atomic-internal
  use. Closes the long-standing TODO architecture entry. **Pure structural
  refactor — zero behavior change; every existing test passes through every
  step.**
- **`silly_kicks.atomic.vaep.features` updated to import per-concern.** 12
  symbols imported across 4 grouped statements against
  `vaep.features.{core,bodypart,context,temporal}` (was: single 12-symbol
  monolith import). TODO A9 partially addressed (severity Medium → Low) —
  full decoupling deferred until atomic features need to diverge independently.
  Local type alias duplicates (`Actions = pd.DataFrame` etc.) replaced by a
  single import from `vaep.features.core` (DRY cleanup).

### Added

- **8 new public-API submodule paths** (`silly_kicks.vaep.features.core`,
  `.actiontype`, `.result`, `.bodypart`, `.spatial`, `.temporal`, `.context`,
  `.specialty`). Documented as implementation detail of where each symbol
  lives — the canonical entry point remains the package itself.
- **3 new test files locking the structure:** T-A backcompat (33 parametrized
  cases asserting every public symbol stays importable from the package path),
  T-B submodule layout (33 parametrized cases asserting each symbol's
  `__module__` matches the design contract), T-C atomic-per-concern (1 test
  asserting atomic imports from per-concern submodules, not the package root).
- **CI gate (`tests/test_public_api_examples.py::_PUBLIC_MODULE_FILES`)
  widened from 19 → 26 entries** to cover all 8 new submodule paths. Net +7
  parametrize cases.

### Closed

- **TODO A19** (default hyperparameters scattered across 3 learner functions):
  reviewed and closed without code change. Already centralized as
  `_XGBOOST_DEFAULTS` / `_CATBOOST_DEFAULTS` / `_LIGHTGBM_DEFAULTS` module-level
  constants since 1.9.0; the audit description ("scattered across 3 functions")
  predates that extraction.
- **TODO O-M1** (full `events.copy()` at top of StatsBomb `convert_to_actions`):
  reviewed and closed without code change. The defensive copy is correct by
  design — `_flatten_extra` mutates the DataFrame by adding ~22 underscore
  columns; without the copy, caller's events would be mutated in place.
- **TODO O-M6** (temporary n×3 DataFrame for StatsBomb fidelity version check):
  reviewed and closed without code change. ~50 KB peak per match; could be
  numpy-fied for marginal gain (~25 KB savings); no measurable impact.

No API breakage. 881 tests passing (807 baseline + 33 T-A + 33 T-B + 1 T-C +
7 net gate delta), 4 deselected.

## [2.2.0] — 2026-04-30

### Added

- **`silly_kicks.atomic.spadl.coverage_metrics`** — Atomic-SPADL counterpart to
  the standard `silly_kicks.spadl.coverage_metrics` utility (added in 1.10.0).
  Resolves `type_id` against the atomic 33-type vocabulary
  (`silly_kicks.atomic.spadl.config.actiontypes`) including atomic-only types
  (`receival`, `interception`, `out`, etc.) and post-collapse names (`corner`,
  `freekick`). Reuses the standard `CoverageMetrics` TypedDict from
  `silly_kicks.spadl.utils` as the single source of truth — both standard and
  atomic surfaces import the same type. Closes TODO C-1 (deferred from 1.10.0).
- **Examples sections on 25 previously-uncovered public-API surfaces** across
  `silly_kicks/vaep/labels.py` (5), `silly_kicks/vaep/formula.py` (3),
  `silly_kicks/atomic/vaep/features.py` (9), `silly_kicks/atomic/vaep/labels.py` (5),
  and `silly_kicks/atomic/vaep/formula.py` (3). Closes the PR-S13 documentation
  coverage gap.

### Changed

- **CI guardrail (`tests/test_public_api_examples.py`) widened from 14 → 19
  module files.** The gate now mechanically enforces Examples coverage across
  the entire public API surface; future PRs that add a public function
  without an Example fail CI.

No API breakage. New public symbols (`coverage_metrics`, `CoverageMetrics`
re-export) are additive only.

## [2.1.1] — 2026-04-30

### Added

- **Examples sections on all public API surfaces.** Closes the long-standing D-8
  documentation gap. Every public function / class / method in
  `silly_kicks.spadl`, `silly_kicks.atomic.spadl`, `silly_kicks.vaep`,
  `silly_kicks.atomic.vaep`, and `silly_kicks.xthreat` now has a 3-7 line
  illustrative example showing typical usage. ~50 surfaces newly documented.
- **CI guardrail at `tests/test_public_api_examples.py`.** AST-based parametrized
  test asserts every public symbol has an `Examples` section in its docstring.
  Future PRs that add a public function without an Example fail CI; the failure
  message points to canonical-style references (`add_possessions`,
  `boundary_metrics`).

### Changed

- **D-9 entry removed from `TODO.md`.** Tech-debt entry was stale — all 9
  module-level helpers in `silly_kicks/xthreat.py` are already underscore-
  prefixed; the entry tracked work that was completed prior to silly-kicks 2.0.0.

No API or behavior changes.

## [2.1.0] — 2026-04-29

### ⚠️ Breaking

- **`add_possessions` default for `max_gap_seconds` changed from 5.0 to 7.0**
  in both `silly_kicks.spadl.add_possessions` and
  `silly_kicks.atomic.spadl.add_possessions`. Empirically Pareto-optimal at
  the per-match recall floor on 64 StatsBomb WorldCup-2018 matches (full
  campaign data:
  `docs/superpowers/specs/2026-04-29-add-possessions-precision-improvement-design.md`).
  Same input DataFrame produces different `possession_id` values for any
  pair of actions where the time gap is in `[5, 7)` seconds AND the team
  did not change.

  **Opt-out:** explicit `add_possessions(actions, max_gap_seconds=5.0)`.

  This default change is shipped as a minor bump under pragmatic semver
  (luxury-lakehouse is the only known consumer; one-line opt-out preserves
  prior behavior). Strict semver would call this 3.0.0.

### Added

- **`silly_kicks.spadl.add_possessions` (and atomic counterpart)** new
  opt-in keyword-only parameters for precision-improvement rules:

  - `merge_brief_opposing_actions: int = 0` + `brief_window_seconds: float = 0.0`
    (paired) — brief-opposing-action merge rule. Suppresses team-change
    boundaries when team B has 1..N consecutive actions sandwiched between
    team A actions within the time window. Both must be > 0 to enable;
    both 0 to disable; exactly one > 0 raises `ValueError`.
  - `defensive_transition_types: tuple[str, ...] = ()` — defensive-transition
    rule. Listed action types do not trigger team-change boundaries on
    their own. Recommended: `("interception", "clearance")`.

  All defaults disable the rules, preserving 2.0.x algorithmic behavior
  except for the `max_gap_seconds` default change above.

- **`tests/datasets/statsbomb/spadl-WorldCup-2018.h5`** regenerated with
  `preserve_native=["possession"]` — the 64-match HDF5 fixture is now a
  reusable regression corpus for `add_possessions`. New file size ~6 MB
  (one extra `possession` column on ~128K rows under zlib compression).

- **`tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBomb64Match`**
  64-match parametrized regression gate complementing the existing 3-fixture
  cross-competition gate. Each match independently gated at
  `recall >= 0.83 AND precision >= 0.30`.

### Changed

- **`silly_kicks/spadl/utils.py`** boundary-detection logic refactored
  into a private `_compute_possession_boundaries` helper, mirroring the
  atomic-side `_compute_possessions` factoring. Public API unchanged;
  internal seam for the new opt-in rules.

- **`tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBombNative`**
  per-match recall threshold lowered from 0.85 to 0.83. Absorbs the
  slightly reduced recall margin at the new `max_gap_seconds=7.0` default
  (worst observed across 64 matches: R_min=0.854) plus pandas/numpy
  version-drift safety margin.

### Behavior baselines

`add_possessions` empirical performance at the new default (no opt-in
rules, 64 WC-2018 matches):

| Metric | Mean | sd | Min |
|---|---|---|---|
| Precision | 0.439 | 0.035 | 0.350 |
| Recall | 0.939 | 0.023 | 0.854 |
| F1 | 0.597 | — | — |

(Compare to 2.0.x at `max_gap_seconds=5.0`: P=0.412, R=0.950, F1=0.574.)

Recommended opt-in settings: see `add_possessions` docstring and
`docs/superpowers/specs/2026-04-29-add-possessions-precision-improvement-design.md`.

## [2.0.0] — 2026-04-29

### ⚠️ Breaking

- **`silly_kicks.spadl.sportec.convert_to_actions` no longer overrides
  `team_id` / `player_id` from DFL `tackle_winner` / `tackle_winner_team`
  qualifiers.** Per ADR-001
  (`docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md`),
  the SPADL converter contract is "caller's identifier conventions are
  sacred — never overridden from qualifiers." Caller-supplied `team` /
  `player_id` values mirror verbatim into the output. Pre-2.0.0 behavior
  silently rewrote ~56% of tackle rows on consumers using a
  caller-normalized `team` convention (see luxury-lakehouse PR-LL2
  close-out report).
- **Sportec output schema changes from `KLOPPY_SPADL_COLUMNS` to
  `SPORTEC_SPADL_COLUMNS`** — 14 + 4 = 18 columns. The 4 new columns
  surface DFL qualifier values: `tackle_winner_player_id`,
  `tackle_winner_team_id`, `tackle_loser_player_id`,
  `tackle_loser_team_id`. NaN on non-tackle rows; NaN when the qualifier
  is absent. Sportec consumers asserting against `KLOPPY_SPADL_COLUMNS`
  must switch to `SPORTEC_SPADL_COLUMNS`.

### Migration

If your pre-2.0.0 sportec consumer relied on the tackle-winner override
AND your upstream `team` / `player_id` columns are in the same
identifier convention as DFL's `tackle_winner_team` / `tackle_winner`
qualifiers (raw `DFL-CLU-...` / `DFL-OBJ-...`), call the new helper
post-conversion:

```python
from silly_kicks.spadl import sportec, use_tackle_winner_as_actor
actions, _ = sportec.convert_to_actions(events, home_team_id="DFL-CLU-XXX")
actions = use_tackle_winner_as_actor(actions)
```

If your `team` / `player_id` columns use any other convention, the
post-1.10.0 behavior already preserved your conventions correctly — no
migration needed; the bug fix is automatic on upgrade.

### Added

- **First silly-kicks ADR.** `docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md`
  + `docs/superpowers/adrs/ADR-TEMPLATE.md` (vendored verbatim from
  luxury-lakehouse) establish the silly-kicks ADR pattern. Future
  decisions that add an exception to project-wide conventions, change
  schema ownership, or hardcode a workaround for a platform constraint
  get an ADR.
- **`silly_kicks.spadl.SPORTEC_SPADL_COLUMNS`** schema constant (18-key
  dict) — extends `KLOPPY_SPADL_COLUMNS` with the 4 tackle qualifier
  passthrough columns. Re-exported from `silly_kicks.spadl`.
- **`silly_kicks.spadl.use_tackle_winner_as_actor(actions) -> pd.DataFrame`**
  — pure post-conversion enrichment that restores pre-2.0.0 sportec
  SPADL "actor = winner" semantic for callers whose upstream identifier
  convention matches DFL's qualifier format. Raises `ValueError` early
  on missing required columns. Mirrors the `add_*` helper family pattern.
- **Cross-provider parity regression gate**
  (`tests/spadl/test_cross_provider_parity.py::test_team_id_mirrors_input_team`).
  Parametrized over all 5 DataFrame converters; asserts each output's
  `team_id` values are a subset of the input `team` values. Locks the
  ADR-001 contract per-provider going forward; would have caught the
  1.7.0 sportec bug.
- **e2e on the IDSSE production fixture**
  (`TestSportecAdrContractOnProductionFixture`, 5 tests). Verifies the
  contract works on production-shape data: caller's labels survive
  through the converter; the 4 new columns are populated for qualifier
  rows; the migration helper round-trips correctly; 1.10.0 keeper
  coverage is preserved.

### Changed

- **CLAUDE.md "Key conventions" section** gains one rule citing ADR-001:
  "Converter identifier conventions are sacred. SPADL DataFrame
  converters never override the caller's `team_id` / `player_id`
  columns from provider-specific qualifiers..."
- **Sportec module docstring** documents the 4 tackle qualifier
  passthrough columns + the `SPORTEC_SPADL_COLUMNS` schema + the
  migration helper. References ADR-001.

### Removed

- **`silly_kicks.spadl.sportec` tackle override block** at the previous
  `sportec.py:559-565`. The 6-line override that silently rewrote
  `team_id` / `player_id` from raw DFL qualifier values is gone.
- **`tests/spadl/test_sportec.py::TestSportecActionMappingShotsTacklesFoulsGK::test_tackle_uses_winner_as_actor`**
  — was asserting the now-removed override. Covered by the new
  `TestSportecTackleNoOverride` + `TestSportecTackleWinnerColumns`
  classes.

### Audit findings

Manual cross-converter review (this cycle) confirmed sportec.tackle
was the unique violator of the ADR-001 contract:

| Converter | Override `player_id` / `team_id`? | Notes |
|---|---|---|
| `silly_kicks.spadl.sportec` | YES (removed) | The bug. |
| `silly_kicks.spadl.metrica` | NO | 1.10.0 GK routing only changes `type_id` / `bodypart_id`. |
| `silly_kicks.spadl.wyscout` | NO | 1.0.0 aerial-duel reclassification only changes `type_id` / `subtype_id`. |
| `silly_kicks.spadl.statsbomb` | NO | No qualifier-driven overrides. |
| `silly_kicks.spadl.opta` | NO | No qualifier-driven overrides. |
| `silly_kicks.spadl.kloppy` | NO | Gateway path. |

The 2.0.0 change is surgical (one converter), but the parity gate locks
the contract for all future converter additions.

### Notes

- silly-kicks 2.0.0 is the project's first semver-major release. The
  library is ~3 weeks old (0.1.0 shipped 2026-04-06); major versions
  aren't precious — bumping locks the contract before more downstream
  consumers pin against pre-2.0.0 behavior.
- luxury-lakehouse can bump `silly-kicks>=2.0.0,<3.0` and (optionally)
  drop their `_team_label_to_dfl_id` shim from PR-LL2 close-out, OR
  keep it as a documented winner-attribution post-conversion pattern.

## [1.10.0] — 2026-04-29

### Added
- **Public `silly_kicks.spadl.coverage_metrics(*, actions, expected_action_types)` utility**
  for computing per-action-type coverage on a SPADL action stream. Returns
  a `CoverageMetrics` TypedDict (also re-exported from `silly_kicks.spadl`).
  Keyword-only arguments. Resolves `type_id` to action-type name via
  `spadlconfig.actiontypes_df`; reports any expected action types that
  produced zero rows under `missing`. Out-of-vocab `type_id` values are
  reported as `"unknown"` rather than raising. Mirrors the PR-S8
  `boundary_metrics` shape and discipline.
- **`goalkeeper_ids: set[str] | None = None` parameter on
  `silly_kicks.spadl.sportec.convert_to_actions`** as a supplementary
  signal: when provided, Play events whose `player_id` is in the set
  AND which have NO explicit `play_goal_keeper_action` qualifier are
  routed to the keeper_pick_up + pass 2-action synthesis. The
  qualifier-driven mapping remains the primary contract.
- **`goalkeeper_ids: set[str] | None = None` parameter on
  `silly_kicks.spadl.metrica.convert_to_actions`** as the PRIMARY
  mechanism for surfacing GK actions. Metrica's source format lacks
  native GK markers; with `goalkeeper_ids`, conservative routing applies
  (PASS by GK → synth, RECOVERY by GK → keeper_pick_up, CHALLENGE
  AERIAL-WON by GK → keeper_claim). Without it: 0 keeper_* actions
  (1.9.0 default behaviour preserved — no breaking change).
- **`goalkeeper_ids` no-op acceptance on `statsbomb.convert_to_actions`
  and `opta.convert_to_actions`** for cross-provider API symmetry. Both
  source formats natively mark GK actions; the parameter is silently
  accepted with byte-for-byte identical output.
- **DFL distribution qualifiers `throwOut` and `punt` now produce SPADL
  actions** (sportec converter). Each source row synthesizes TWO
  actions: `keeper_pick_up + pass` (bodypart=other) for `throwOut`,
  `keeper_pick_up + goalkick` (bodypart=foot) for `punt`. Both rows
  inherit the source's `(player_id, team, period, time, x, y)`.
  `preserve_native` columns propagate to both. Action_ids renumbered
  dense after synthesis.
- **Production-shape vendored fixtures** under
  `tests/datasets/idsse/sample_match.parquet` (~166 KB; 308-row subset
  of `soccer_analytics.bronze.idsse_events` match `idsse_J03WMX`,
  includes throwOut + punt rows) and
  `tests/datasets/metrica/sample_match.parquet` (~20 KB; 300-event
  subset of Metrica Sample Game 2). Build script at
  `scripts/extract_provider_fixtures.py` (Databricks pull for IDSSE,
  offline kloppy-fixture subset for Metrica). Attribution READMEs
  alongside.
- **Cross-provider parity meta-test** at
  `tests/spadl/test_cross_provider_parity.py`. Parametrized over all 5
  DataFrame converters (statsbomb, opta, wyscout, sportec, metrica);
  asserts each emits at least one `keeper_*` action when given a
  fixture exercising GK paths. This is the regression gate that would
  have caught Bugs 1-3 in 1.7.0 if it had existed.
- **`pyarrow>=14.0.0` added to `[test]` extras** to back parquet I/O
  for the new fixtures (`pd.read_parquet` / `pd.DataFrame.to_parquet`).

### Fixed
- **Sportec converter no longer drops all DFL `Play` events to
  non_action.** The pre-1.10.0 dispatch checked `et == "Pass"` for
  pass-class events, but DFL bronze never emits `"Pass"` — the actual
  event_type is `"Play"`. Net effect since 1.7.0: all IDSSE matches in
  production lost ~60-80% of their actions (every pass, cross, and head
  pass) to silent non_action drop. Fix restructures the dispatch so
  `Play` events with no GK qualifier route to `pass` / `cross` (with
  optional head bodypart) and `Play` events with a recognized GK
  qualifier route to `keeper_*` actions. Defensive: `Play` events with
  an unrecognized non-empty qualifier still drop to `non_action`.
  ``"Pass"`` is removed from the recognized event-type vocabulary so
  legacy callers (if any) surface in `unrecognized_counts` (loud)
  rather than silently mapping to non_action.
- **Sportec converter no longer drops `throwOut` and `punt` GK
  distribution events to non_action.** These DFL qualifier values
  represent GK distribution actions (throwing or kicking the ball to
  a teammate); pre-1.10.0 they were unmapped. Fix synthesizes 2
  SPADL actions per source event (see Added section).
- **Metrica converter now produces non-zero GK coverage when
  `goalkeeper_ids` is supplied.** Pre-1.10.0 the converter had no
  mechanism to surface GK actions, leaving downstream `add_gk_role` /
  `add_pre_shot_gk_context` enrichments at 100% NULL on every Metrica
  match in production.

### Notes
- This release closes the upstream gap that surfaced during
  luxury-lakehouse PR-LL2 production deploy (2026-04-29): post-deploy
  validation found 100% NULL `gk_role` and `defending_gk_player_id` on
  IDSSE (2,522 rows) and Metrica (5,839 rows) sources. With silly-kicks
  1.10.0, downstream lakehouse can re-run `apply_spadl_enrichments`
  against IDSSE + Metrica with non-NULL GK coverage (handled by
  separate lakehouse PR-LL3).
- Behaviour change for IDSSE consumers: bronze.spadl_actions row count
  per IDSSE match will increase materially (every Play event now
  surfaces as a SPADL pass, plus throwOut/punt rows now produce 2
  actions each). This is the intended fix; downstream aggregation may
  need to re-baseline.
- Wyscout converter unchanged — `goalkeeper_ids` was already present
  from 1.0.0.
- Atomic-SPADL `coverage_metrics` parity is queued as tech debt
  (atomic uses 33 action types vs standard's 23; deferred until a
  consumer asks). Tracked in `TODO.md ## Tech Debt`.

## [1.9.0] — 2026-04-29

### Added
- **Vendored `tests/datasets/statsbomb/spadl-WorldCup-2018.h5`** — committed
  HDF5 fixture for the FIFA World Cup 2018 (64 matches, 128,484 SPADL
  actions, 5.9 MB on disk with zlib compression). All 5 prediction
  pipeline tests in `tests/vaep/`, `tests/test_xthreat.py`, and
  `tests/atomic/` now run on every PR + push. Pre-1.9.0 these tests
  silently skipped in CI and locally because the fixture was never
  committed. Net: ~9 release cycles of zero coverage on the prediction
  pipeline (VAEP fit/rate, xT fit/rate, atomic VAEP fit/rate) is now
  closed.
- **`scripts/build_worldcup_fixture.py`** — reproducible HDF5 generator.
  Downloads StatsBomb open-data WorldCup-2018 raw events (cached at
  `tests/datasets/statsbomb/raw/.cache/`, gitignored), converts each via
  `silly_kicks.spadl.statsbomb.convert_to_actions`, writes the multi-key
  HDFStore. CLI: `--output`, `--cache-dir`, `--no-cache`, `--verbose`,
  `--quiet`. Cold-cache run on broadband: ~30-60 sec. Warm-cache re-run:
  ~5 sec. No new dependencies (stdlib + pandas + already-present
  pytables).
- **`scripts/` is now linted in CI** — `.github/workflows/ci.yml` runs
  `ruff check` and `ruff format --check` on `silly_kicks/`, `tests/`,
  AND `scripts/`. Pyright include stays `silly_kicks/` only — build
  scripts aren't worth full type-checking.

### Changed
- **`tests/conftest.py::sb_worldcup_data` calls `pytest.fail` instead of
  `pytest.skip` when the HDF5 is absent.** Matches the PR-S8 pattern for
  committed fixtures: once a fixture is committed, "missing" is a
  packaging error worth surfacing prominently — not a silent skip that
  lets CI quietly regress. Failure message points at the build script
  for regeneration.
- The 5 `test_predict*` cases (`tests/vaep/test_vaep.py::test_predict`,
  `tests/vaep/test_vaep.py::test_predict_with_missing_features`,
  `tests/test_xthreat.py::test_predict`,
  `tests/test_xthreat.py::test_predict_with_interpolation`,
  `tests/atomic/test_atomic_vaep.py::test_predict`) no longer carry the
  `@pytest.mark.e2e` marker. They run in the regular suite on every CI
  matrix slot (4 slots, ~5-15 sec overhead per slot — negligible).

### Fixed
- **`silly_kicks.xthreat.ExpectedThreat.interpolator()` is no longer
  broken on SciPy 1.14+.** The wrapper used `scipy.interpolate.interp2d`
  which was removed in SciPy 1.14.0 (the import succeeds but the call
  raises `NotImplementedError`). The bug was latent since 1.0.0 because
  `tests/test_xthreat.py::test_predict_with_interpolation` was the only
  consumer and it was `@pytest.mark.e2e`-marked + skipping silently.
  Surfaced precisely when this PR dropped the marker. Replaced with
  `scipy.interpolate.RectBivariateSpline` — the SciPy-recommended
  bug-for-bug compatible replacement for regular grids — wrapped to
  preserve the legacy `interp(xs, ys) -> (W, L)` calling convention so
  callers downstream of `interpolator()` need no changes. Output shape
  and indexing semantics unchanged.
- The `test_interpolate_xt_grid_no_scipy` regression test that mocks
  the missing-scipy path now mocks `RectBivariateSpline` instead of the
  removed `interp2d`.

### Documentation
- **`docs/DEFERRED.md` deleted; live items migrated to a new `## Tech
  Debt` section in `TODO.md`.** Per the National Park Principle —
  bundle the cleanup of the rotting parallel doc into this cycle since
  we're already touching `TODO.md` anyway. Audit history preserved in
  `git log -- docs/DEFERRED.md`. Migrated items: A19 (default
  hyperparameters scattered), D-9 (5 xthreat module-level functions
  naming), O-M1 (StatsBomb `events.copy()`), O-M6 (StatsBomb fidelity
  version check temporary DataFrame). Items judged "by design / accept"
  and not migrated: A15 (kloppy LSP differs by design), A16 (no plugin
  registry — YAGNI for 4 converters), A17 (`_fit_*` coupling — partial
  refactor done, diminishing returns), S5 (optional ML deps no upper
  bounds — librarian convention).
- `CLAUDE.md` no longer references `docs/DEFERRED.md` (file removed).

### Notes
- WorldCup HDF5 file size: 5.9 MB on disk (well under GitHub's 50 MB soft
  warn / 100 MB hard reject thresholds — no Git LFS needed). Total wheel
  size unchanged (test fixtures live under `tests/`, excluded from
  `[tool.hatch.build.targets.wheel] packages = ["silly_kicks"]`).
- The `tests/datasets/statsbomb/raw/.cache/` directory is gitignored —
  raw event JSONs (~192 MB total) are downloaded on demand by the build
  script and never committed.

## [1.8.0] — 2026-04-29

### Added
- **Public `silly_kicks.spadl.boundary_metrics(*, heuristic, native)` utility**
  for computing precision / recall / F1 between two possession-id sequences.
  Returns a `BoundaryMetrics` TypedDict (also re-exported from
  `silly_kicks.spadl`). Keyword-only arguments — the metric is asymmetric
  (precision and recall swap when inputs swap), so positional usage is a
  silent footgun the API surface eliminates. Returns `0.0` for any metric
  whose denominator is zero (empty / single-row / constant sequences).
  Length-mismatched inputs raise `ValueError`.
- 3 vendored StatsBomb open-data fixtures under
  `tests/datasets/statsbomb/raw/events/` (matches 7298, 7584, 3754058 —
  Women's World Cup, Champions League, Premier League; ~9 MB total).
  License attribution in `tests/datasets/statsbomb/README.md`. Used by
  the new parametrized regression gate.

### Changed
- **`add_possessions` docstring is now honest about empirical performance.**
  The previous "boundary-F1 ~0.90" claim was 30+ percentage points above
  the actual measurement on StatsBomb open-data. New text reports
  recall ~0.93, precision ~0.42, F1 ~0.58 (peak ~0.605 at
  `max_gap_seconds=10.0`) and explains why precision is the way it is
  (intrinsic to the team-change-with-carve-outs algorithm class, not a
  defect — StatsBomb's proprietary annotation merges brief opposing-
  team actions back into the containing possession; the heuristic
  cannot replicate that structurally).
- **e2e validation gate replaces F1 ≥ 0.80 with recall ≥ 0.85 AND
  precision ≥ 0.30 per match.** Recall enforces the helper's primary
  contract (catching every real boundary). Precision floor catches the
  "boundary cardinality halved or doubled" regression class that affects
  per-possession aggregation downstream. F1 stays in the assert message
  for diagnostics only — gating on F1 would re-introduce the
  misrepresentation problem this PR is fixing.
- **Test class renamed** `TestBoundaryF1AgainstStatsBombNative` →
  `TestBoundaryAgainstStatsBombNative`. Parametrized over the 3 vendored
  fixtures with per-match independent gates.

### Fixed
- **e2e regression coverage now actually runs in CI.** The previous
  `TestBoundaryF1AgainstStatsBombNative::test_boundary_f1_against_native_possession_id`
  was `@pytest.mark.e2e` and silently skipped on every CI run since
  1.2.0 because the fixture wasn't committed. It was also skipping
  locally (the fixture was never on the user's only development
  machine). Net: ~6 release cycles of zero coverage on this test. PR-S8
  vendors the fixtures and drops the marker so the test runs on every
  PR + push.

### Notes
- Empirical baselines verified locally on the committed fixtures:
  recall {0.9425, 0.9268, 0.9259}, precision {0.4484, 0.4306, 0.3855},
  F1 {0.6077, 0.5880, 0.5443} for matches 7298 / 7584 / 3754058
  respectively. All comfortably above the gate thresholds; tightest
  margin is precision on 3754058 (8.55pp above floor).
- The 5 `test_predict*` cases in `tests/vaep/`, `tests/test_xthreat.py`,
  and `tests/atomic/` continue to skip in CI (and locally) because they
  depend on the un-committed `tests/datasets/statsbomb/spadl-WorldCup-2018.h5`
  fixture. Closing that gap is queued as PR-S9 (generate the HDF5 from
  open-data raw events; commit + drop e2e markers). Tracked in
  `TODO.md`.
- Algorithmic precision improvement for `add_possessions` is queued as
  PR-S10 (look-ahead merge rules for brief opposing-team actions;
  re-measure `max_gap_seconds` defaults using the new
  `boundary_metrics` utility).

## [1.7.0] — 2026-04-29

### Added
- **Dedicated DataFrame SPADL converters for Sportec and Metrica.** New
  modules `silly_kicks.spadl.sportec` and `silly_kicks.spadl.metrica`
  expose `convert_to_actions(events_df, home_team_id, *,
  preserve_native=None) -> tuple[pd.DataFrame, ConversionReport]`,
  matching the established `statsbomb` / `wyscout` / `opta` shape.
  Designed for consumers who already have normalized event data in
  pandas form (lakehouse bronze layers, ETL pipelines, research
  notebooks) and don't want to reconstruct a kloppy `EventDataset` from
  flat rows. Existing kloppy-path consumers continue to use
  `silly_kicks.spadl.kloppy` — both paths produce equivalent SPADL output
  (empirically verified by cross-path consistency tests under
  `tests/spadl/test_sportec.py::TestSportecCrossPathConsistency` and
  `tests/spadl/test_metrica.py::TestMetricaCrossPathConsistency`).
- ~120 recognized DFL qualifier columns surfaced via Sportec converter,
  covering pass / shot / tackle / foul / set-piece / play / cross /
  cards / substitution / penalty / VAR / chance / specialised /
  tracking-derived qualifier groups.
- Metrica set-piece-then-shot composition rule: `SET PIECE` (FREE KICK)
  immediately followed (≤ 5s, same player, same period) by `SHOT`
  upgrades the shot to SPADL `shot_freekick` and drops the SET PIECE
  row.

### Changed
- **`silly_kicks.spadl.kloppy.convert_to_actions` now applies
  `_fix_direction_of_play` automatically** (extracting home team from
  `dataset.metadata.teams[0].team_id`). Pre-1.7.0 the kloppy converter
  was the lone outlier among silly-kicks SPADL converters — it stayed
  in kloppy's `Orientation.HOME_AWAY` (home plays LTR, away plays RTL)
  while StatsBomb / Wyscout / Opta all flipped away-team coords for
  canonical "all-actions-LTR" SPADL convention. 1.7.0 unifies the
  convention across all 6 converters
  (`statsbomb` / `wyscout` / `opta` / `kloppy` / new `sportec` / new
  `metrica`) so all converters emit semantically equivalent SPADL output
  for the same source event stream. Hyrum's Law disclaimer: zero current
  consumers built against 1.6.0's HOME_AWAY-oriented kloppy output (per
  user confirmation during brainstorming).

### Notes
- Cross-path consistency proof: dedicated DataFrame converters and the
  kloppy gateway path produce equivalent SPADL DataFrames when given
  the same source data bridged through test helpers.
- New shared pytest conftest at `tests/spadl/conftest.py` provides
  module-scoped `sportec_dataset` and `metrica_dataset` fixtures
  reusable across `test_kloppy.py`, `test_sportec.py`, and
  `test_metrica.py`.

## [1.6.0] — 2026-04-28

### Added
- **Kloppy converter: Sportec and Metrica support.** `Provider.SPORTEC`
  (Sportec Solutions / IDSSE Bundesliga event format) and `Provider.METRICA`
  (Metrica Sports) are now first-class allowlisted providers in
  `silly_kicks.spadl.kloppy.convert_to_actions`. Empirical verification on
  real fixture data confirms zero new event-type mappings are required —
  both providers' kloppy serializers emit only event types already covered
  by the existing `_MAPPED_EVENT_TYPES` ∪ `_EXCLUDED_EVENT_TYPES` sets.
  `preserve_native` works transparently for both (their `raw_event` is a
  `dict`).
- Real-fixture end-to-end test suites for Sportec and Metrica under
  `tests/spadl/test_kloppy.py`, plus a parametrized coordinate-clamping
  test and a per-provider `ConversionReport` shape test. Test fixtures
  vendored from kloppy's BSD-3-Clause-licensed test files into
  `tests/datasets/kloppy/`.

### Fixed
- **`_SoccerActionCoordinateSystem` was unusable on real datasets.** The
  class definition omitted `__init__`, but `convert_to_actions()`
  instantiated it with `pitch_length=` / `pitch_width=` kwargs. On any
  kloppy version with the current `CoordinateSystem` ABC signature
  (kloppy 3.15+), this raised `TypeError` the moment a real
  `EventDataset` reached `dataset.transform()`. Latent since 1.0.0
  because pre-existing `tests/spadl/test_kloppy.py` was pure mocks
  that never reached the transform call. Affected **all** kloppy-based
  conversion including the previously-allowlisted StatsBomb path.
- 2 pyright errors in `silly_kicks/xthreat.py:402` surfaced by newer
  pandas-stubs / numpy-stubs versions: explicit `dtype=np.float64` added
  to two `np.linspace` calls so the inferred `NDArray[float64]` matches
  the `interp(...)` callable signature.

### Changed
- **Kloppy converter now clamps output coordinates to
  `[0, field_length] × [0, field_width]` (105 × 68 m).** This aligns the
  kloppy converter with the established silly-kicks convention — StatsBomb
  / Wyscout / Opta converters all clamp; kloppy was the lone outlier.
  Empirically Metrica events emit slight off-pitch coords (observed
  `x ∈ [-1.62, 104.63]` on the sample game) within source-recording-noise
  tolerance. Downstream consumers depending on raw off-pitch coordinates
  from the kloppy path specifically should re-verify (no such consumer
  documented).

## [1.5.0] — 2026-04-27

### Added
- **Atomic-SPADL parity for the 1.1.0 → 1.4.0 helper family.** The five
  helpers shipped on standard SPADL (`preserve_native` primitive,
  `add_possessions`, `add_gk_role`, `add_gk_distribution_metrics`,
  `add_pre_shot_gk_context`) plus a new defensive `validate_atomic_spadl`
  helper now have first-class atomic counterparts under
  `silly_kicks.atomic.spadl`:
  - `convert_to_atomic(actions, *, preserve_native=...)` — surfaces
    caller-attached columns from the input SPADL dataframe alongside the
    canonical 13 atomic columns. Synthetic atomic rows generated by the
    conversion (`receival` / `interception` / `out` / `offside` / `goal`
    / `owngoal` / `yellow_card` / `red_card`) receive `NaN` in the
    preserved columns — same behaviour as the standard converters'
    `preserve_native` for synthetic dribble rows.
  - `add_possessions(actions)` — atomic counterpart with two atomic-
    specific adaptations: (a) set-piece restart names match the post-
    collapse atomic types (`corner` / `freekick` / `throw_in` /
    `goalkick`); (b) `yellow_card` / `red_card` synthetic rows are
    transparent to boundary detection — they never trigger a possession
    boundary on their own and inherit the surrounding state via
    forward-fill within `game_id`.
  - `add_gk_role(actions)` — atomic counterpart; reads `x` (NOT
    `start_x`) for the penalty-area threshold check. Same five
    categories.
  - `add_gk_distribution_metrics(actions, xt_grid=None)` — atomic
    counterpart with three atomic-specific adaptations: (a) length is
    `sqrt(dx² + dy²)` from atomic's `(dx, dy)` columns; (b) xT delta is
    from `(x, y)` to `(x + dx, y + dy)`; (c) pass success is detected
    from the FOLLOWING atomic action by row index (`receival` =
    success; `interception` / `out` / `offside` = failure; no following
    action = conservative failure with `gk_xt_delta = NaN`). Atomic
    launch types collapse `{pass, goalkick, freekick_short,
    freekick_crossed}` into `{pass, goalkick, freekick}` (where
    `freekick` is the post-collapse name).
  - `add_pre_shot_gk_context(actions)` — atomic counterpart; recognises
    only `shot` and `shot_penalty` as shot rows. (Standard SPADL's
    `shot_freekick` is collapsed into atomic's `freekick`, mixing
    pass-class and shot-class freekicks; the helper does not attempt to
    disambiguate.)
  - `validate_atomic_spadl(df)` — defensive schema validator. Returns
    input unchanged for chaining; warns on dtype mismatches; raises on
    missing columns.

  All five helpers are vectorised on numpy/pandas; sub-50ms per 1500-
  action match (CI hard bound 200ms; benchmark assertions in
  `tests/test_benchmark.py`). 174 new atomic tests including a
  cross-validation suite asserting algorithmic equivalence between the
  standard and atomic helpers when applied to a SPADL stream and its
  atomic projection.

### Fixed
- Test infra: `tables>=3.9.0` (pytables) added to the `[test]` extras —
  required by `pd.HDFStore` for the `sb_worldcup_data` fixture in
  `tests/conftest.py`. Without it, the 5 `test_predict*` cases (vaep /
  xthreat / atomic vaep) errored at collection time with
  `ImportError("Missing optional dependency 'pytables'")`.
- Test infra: the `sb_worldcup_data` fixture now `pytest.skip(...)`s
  when the `spadl-WorldCup-2018.h5` dataset is not present locally,
  rather than erroring with `FileNotFoundError`. Aligns with the
  `@pytest.mark.e2e` semantics ("requires downloaded datasets") for the
  5 affected tests.

### Notes
- Atomic-SPADL parity TODO is now closed.

## [1.4.0] — 2026-04-27

### Added
- **GK analytics suite v1** — three composable post-conversion enrichments
  for SPADL action streams, mirroring the public-helper shape of
  `add_names()` and `add_possessions()`:
  - `add_gk_role(actions)` — tags each action with the goalkeeper's role
    context: `shot_stopping` / `cross_collection` / `sweeping` / `pick_up` /
    `distribution` (or `None` for non-GK actions). Sweeping is a
    position-based override for `keeper_*` actions taken outside the
    penalty area; in clean event data only `keeper_save` realistically
    appears outside the box (sweeper-style rush-out save). The other
    three keeper types outside the box are illegal handball offences and
    effectively non-existent in regulation play.
  - `add_gk_distribution_metrics(actions, xt_grid=None)` — adds
    `gk_pass_length_m`, `gk_pass_length_class` (short/medium/long),
    `is_launch`, and `gk_xt_delta` to GK distribution actions. Auto-calls
    `add_gk_role` when `gk_role` column is absent. xT delta only computed
    for successful distributions when an xT grid is provided. `is_launch`
    requires both length > `long_threshold` and a deliberate-distribution
    pass type (`pass`, `goalkick`, `freekick_short`, `freekick_crossed`).
  - `add_pre_shot_gk_context(actions)` — for every shot, looks back up to
    `lookback_actions` rows or `lookback_seconds` seconds (smaller wins)
    in the same `(game_id, period_id)` and tags the defending GK's recent
    activity: `gk_was_distributing`, `gk_was_engaged`,
    `gk_actions_in_possession`, `defending_gk_player_id`. Genuinely novel
    — no published OSS / academic equivalent surfaces a goalkeeper's
    pre-shot activity context as explicit per-shot features.

  All three are vectorised on numpy/pandas; sub-50ms per 1500-action match.
  References cited in docstrings: Yam (MIT Sloan), Lamberts GVM (2025),
  Butcher et al. xGOT (2025).

### Notes
- Atomic-SPADL parity for the GK analytics suite is deferred (TODO under
  `## Architecture`). Same disposition as `add_possessions`.

## [1.3.0] — 2026-04-27

### Added
- `pandas-stubs>=2.2.0` pinned in the `[dev]` extras and the CI lint job.
  Without `pandas-stubs`, pyright's bundled pandas typings under-report
  Series / DataFrame types (e.g. arithmetic on ``.values`` collapses to
  the union ``np_1darray | ExtensionArray | Categorical``), masking real
  type issues in CI while spuriously failing locally on certain method
  chains. With `pandas-stubs` in the dev path, pyright reports a
  consistent set of issues across all environments.

### Fixed
- 15 type errors that surfaced once `pandas-stubs` was installed:
  - `vaep/features.py` and `atomic/vaep/features.py` — replaced
    `Series.values` with `Series.to_numpy()` in polar-coordinate
    arithmetic so the return type is `np.ndarray` instead of the
    ``np_1darray | ExtensionArray | Categorical`` union (which doesn't
    support `**` / `/` / `-`).
  - `spadl/opta.py` — same `.values` → `.to_numpy()` swap in
    ``_fix_owngoals`` arithmetic.
  - `spadl/statsbomb.py` — synthetic interception-event `extra` payload
    now built as an explicit ``pd.Series([..], dtype=object)`` instead
    of `[dict] * n`, matching pandas-stubs's accepted setitem value types.
  - `spadl/utils.py` `_finalize_output()` — schema dtype string passed
    through `np.dtype(...)` so it narrows to ``DtypeObj`` for the
    `astype` overload set.
- Removed two `cast(pd.DataFrame, ...)` workarounds in
  `add_possessions` (introduced in 1.2.0). With `pandas-stubs`,
  non-inplace ``sort_values()`` / ``drop()`` correctly return
  `DataFrame`, making the casts redundant.

## [1.2.0] — 2026-04-27

### Added
- `silly_kicks.spadl.utils.add_possessions(actions, *, max_gap_seconds=5.0,
  retain_on_set_pieces=True)` — provider-agnostic possession-sequence
  reconstruction for any SPADL action stream. Adds a `possession_id: int64`
  column via a team-change-with-carve-outs heuristic: boundaries on team
  change, period change (within a game), or time gap >= `max_gap_seconds`,
  with a foul→opposing-team-set-piece carve-out that retains the previous
  possession (the team that won the foul resumes its sequence). Counter
  resets to 0 at each new `game_id`. Mirrors the public-enrichment shape
  of `add_names()` (post-conversion, returns a copy with the new column).
  Vectorised on numpy/pandas; ~1ms per 1500-action match, sub-3ms on 10k.
- Performance benchmarks for `add_possessions` (1500-action and 10k-action
  scenarios) added to `tests/test_benchmark.py` with hard CI bounds
  (200ms / 2s respectively) catching accidental quadratic regressions.
- e2e-marked boundary-F1 validation test against StatsBomb's native
  `possession` field (using `preserve_native=['possession']` from 1.1.0
  to surface the native truth alongside the heuristic). Skips when the
  raw StatsBomb fixture is absent; documents the validation procedure
  for downstream consumers wanting to re-measure the agreement rate
  against their own data.

### Notes
- Atomic-SPADL parity for `add_possessions` is deferred (TODO under
  `## Architecture`). Apply the same passthrough mechanism when there's
  a concrete consumer asking for it.

## [1.1.0] — 2026-04-27

### Added
- `preserve_native` parameter on `convert_to_actions` for all four SPADL
  converters (`statsbomb`, `wyscout`, `opta`, `kloppy`). Surfaces provider-
  native event fields alongside the canonical SPADL output as extra columns
  on the returned DataFrame — useful for surfacing fields that the canonical
  SPADL schema doesn't carry (e.g. StatsBomb's native `possession` sequence
  number, `possession_team`, `play_pattern`; Wyscout bronze passthroughs;
  Opta competition metadata). Each `preserve_native` field must be present
  on the input and must not overlap with the SPADL schema; both conditions
  raise `ValueError` early. Synthetic actions inserted by `_add_dribbles`
  get NaN in preserved columns (no source event to inherit from).
- `extra_columns` parameter on internal `silly_kicks.spadl.utils._finalize_output()`
  that powers the public `preserve_native` feature.
- `_validate_preserve_native()` helper in `silly_kicks.spadl.utils` for
  shared upfront validation across providers (input-column presence +
  schema-overlap check).
- Kloppy `preserve_native` requires kloppy >= 3.15 with raw-event
  preservation. Each preserved field is read from `event.raw_event[field]`.

## [1.0.0] — 2026-04-07

### Added
- DEBUG logging for kloppy silent event drops (aerial duels, unrecognized GK subtypes)
- `.github/CODEOWNERS` for code owner review enforcement

### Fixed
- StatsBomb converter now accepts both `"goalkeeper"` and `"goal_keeper"` keys in the
  extra dict — adapters that snake-case the event type name no longer silently lose all
  keeper actions

### Improved
- `ConversionReport` docstring: full Attributes section, usage example, provider-specific
  key type note
- `add_names()` docstring: explicit guarantee that caller-added columns are preserved
- `_finalize_output()` docstring: guarantee that all SPADL_COLUMNS are present
- `config.py` docstring: `actiontype_id`, `result_id`, `bodypart_id` reverse dicts documented
- Wyscout `convert_to_actions()`: Returns section now documents `ConversionReport`;
  `goalkeeper_ids` notes `None` ≡ empty set equivalence

### Removed
- `docs/plans/` and `docs/specs/` — internal development artifacts with local paths

### Changed
- Version bump: 0.1.0 → 1.0.0 (Production/Stable)
- C4 diagram genericized (removed project-specific references)

## [0.1.0] — 2026-04-06

### Added
- Initial release as maintained successor to socceraction v1.5.3
- SPADL converters: StatsBomb, Opta, Wyscout, Kloppy
- VAEP and Atomic-VAEP frameworks
- HybridVAEP — result-leakage-free action valuation
- xG-targeted labels via `xg_column` parameter
- Expected Saves (xS) label via `save_from_shot()`
- Expected Claims (xC) label via `claim_from_cross()`
- Cross zone feature (Gelade 2017 four-zone classification)
- Assist type feature (through ball, cutback, cross, set piece, progressive pass)
- Wyscout `goalkeeper_ids` parameter for GK aerial duel routing (#37)
- `ConversionReport` audit trail for every conversion
- `validate_spadl()` utility for DataFrame validation
- Input validation with clear error messages per provider
- "Nothing Left Behind" mapping registries (mapped/excluded/unrecognized events)
- Reproducible training via `random_state` parameter

### Changed (from socceraction v1.5.3)
- Dropped pandera dependency — schemas are plain Python constants
- Dropped multimethod dependency
- Removed numpy<2.0 upper bound
- All converters return `tuple[pd.DataFrame, ConversionReport]`
- All `apply(axis=1)` hot paths replaced with `np.select` vectorization
- Wyscout module decomposed into 3 files
- Gamestates uses vectorized shift instead of `groupby().apply()`
- Config DataFrame factories cached with `@functools.cache`
- Labels vectorized (shift-based accumulation replaces 27-column loop)
- `actiontype_result_onehot` uses numpy broadcasting

### Fixed
- Bug #507: Empty game crash in `gamestates()`
- Bug #950: `actiontype` feature wrong for Atomic-SPADL
- Bug #784: Opta converter silently drops card events
- Bug #831: Atomic-SPADL missing "out" for blocked/saved shots
- Bug #37/D44: Wyscout keeper_claim/punch differentiation
- Bug #946: pandas 3.0 `fillna(inplace=True)` deprecation
- pandas 3.0 `groupby().apply(as_index=False)` key column drop
