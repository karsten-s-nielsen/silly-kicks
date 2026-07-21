# ADR-045: Vector-quantity consistency under coordinate reflection

| Field | Value |
|---|---|
| **Date** | 2026-07-21 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen |

## Context

silly-kicks reflects coordinates in many places. ADR-028 establishes that SPADL actions
(per-acting-team LTR) and tracking frames (home-attacks-right) differ by a **180° point
reflection** — `x -> 105-x` **and** `y -> 68-y` — for away-team actions.

A point reflection acts differently on different KINDS of quantity: it point-reflects **points**
(`x`, `start_x`, `x_smoothed`), **negates vectors** (`vx`, `vy`, `dx`, `dy`), leaves
**magnitudes** unchanged (`speed`, distances, areas), and swaps **direction labels**
(`team_attacking_direction`). Every reflection helper in the codebase transformed an **explicitly
enumerated column list**, so any column not on that list rode through untransformed and silently
wrong. Nothing noticed, because `speed` (a magnitude, correctly left alone everywhere) meant no
speed- or distance-based sanity check ever fired.

The root cause is a **missing shared seam, not missing physics.** The correct contract was
already implemented in **four** independent places (`direction.py:284-289`,
`atomic/spadl/utils.py:1129-1133`, `atomic/vaep/features.py:165-169`, and the geometric
orienter). The sites that got it wrong were not less careful — they were working from a
different hand-written enumeration. And the blind-spot columns are invisible from the canonical
schema: `vx`/`vy`/`x_smoothed`/`y_smoothed` are **not in `TRACKING_FRAMES_COLUMNS`** (added later
by `preprocess`), so a schema-driven author handles `x`/`y`/`z` and stops.

Site inventory, stated as a measurement rather than a headline (the audit's original count was
wrong three times over the review cycle): **eleven places apply a reflection, and two more are
defective by omission** (the D1 call site and the D2 site, which should reflect and do not). The
grep that produced it:

```
grep -rn "field_length -\|field_width -\|105\.0 -\|68\.0 -\|FIELD_LENGTH -\|FIELD_WIDTH -" \
    silly_kicks/ --include=*.py
```

## Decision

Introduce one public module `silly_kicks/reflection.py` with two entry points — `reflect()`
(registry-driven, for schema-bearing tables) and `reflect_columns()` (explicit, kind-aware, for
derived/pre-canonical frames) — and migrate every reflection site onto them, so a point
reflection transforms each column by its declared **kind** and can no longer leave velocities,
smoothed positions, or direction labels silently untransformed.

**Fail-closed lives in the CI registry-completeness meta-assertion, not in the runtime call.**
`reflect()`'s `on_unknown` defaults to `"warn"`: an undeclared column is treated as `invariant`
and warns (`UndeclaredGeometricColumnWarning`) only if its name is geometry-shaped.

## Defects (D1–D8)

Only **D1 and D2 are LIVE** (both confined to `pressure_on_actor__bekkers_pi`, away-team
actions). D3/D3b/D4/D5/D7/D8 are LATENT; D6 was a false documentation claim.

- **D1 — LIVE.** `_reproject_rows` (`utils.py:874`) re-projected positions but not velocities, so
  `_bekkers_tti` read action-LTR positions against frame-convention velocity, modelling away
  defenders running backwards. Measured (IDSSE DFL-MAT-J03WMX, 1363 actions / 3.36M frames,
  velocity defect isolated): away rows 97.6% changed, mean 0.2554 vs 0.4181 correct (**−38.9%**),
  median |error| 0.333, max 0.996 on a [0,1] metric. Home rows bit-identical.
- **D2 — LIVE.** `_build_ball_xy_v_per_action` never re-projected the ball at all (position
  included). Away actor-to-ball median 62.13 m vs home 6.13 m; 80% of away rows change. **Its
  mean bias is only −1.1%** (over/under-statements nearly cancel) while MAE is 0.0657 and
  Spearman 0.858 — an aggregate check passes cleanly on the broken code. This forced the
  **per-row-never-aggregate** guard rule below.
- **D3/D3b — LATENT.** `play_left_to_right` mirrored x/y only; `vx`/`vy`, `x_smoothed`/`y_smoothed`
  and the direction label rode through. No library producer labels home `"rtl"`, so it is a
  measured no-op in-library, and the lakehouse reaches orientation through the geometric orienter
  (which already negates velocity). Latent everywhere known; still a real divergence between two
  public orienters.
- **D4 — LATENT.** `finalize_orientation`'s flag leg reflected positions only, 70 lines above a
  geometric leg that negates vectors — unreachable today because the adapter schema projection
  drops `vx`/`vy`. **Both legs must be complete, and "both or neither" is measurably false:**
  "neither" leaves an ~8 m/s-scale kinematic inconsistency on a wrong-flag composed path; only
  "both" is zero-error in all four cases (correct-flag/wrong-flag × backstop-fires/does-not).
- **D5 — LATENT, settled by default.** `_shape_graph.infer_positions` flips `x` (level ordering)
  but not `y`, so the lateral L/LC/C/RC/R label is **pitch-absolute**. It has **no in-library
  consumer**, so no behaviour validates either convention; settled pitch-absolute by default and
  documented, with a both-sides test that goes red if a future consumer makes it team-relative.
- **D6 — DOCUMENTATION.** ADR-042's "TF-4 was the last module keyed on home/away identity" was
  false — other action-coupled aggregators still take `home_team_id` by design. Corrected here
  and in CLAUDE.md.
- **D7 — LATENT.** The two VAEP `play_left_to_right` helpers enumerate the four canonical
  coordinates, so ADR-025 `enriched_*` columns ride through unmirrored. They **mutate in place**
  and return the caller's objects; the migration computes purely via `reflect()` then **assigns
  the non-inert columns back**, preserving the in-place contract without upcasting untouched
  integer columns.
- **D8 — LATENT.** `spadl/orientation.py`'s `_mirror_absolute_frame` / `_mirror_per_period` (the
  two branches of public `to_spadl_ltr`, called by nine converters) enumerate the four canonical
  coordinates — the same ADR-025 trap, on a busier path than the site the audit originally
  carried. Found in the fifth review pass.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. `on_unknown="raise"` at every migrated site (fail-closed at runtime) | loud on an undeclared column | unimplementable at D8; breaks first-party API; caught 0 of the 8 defects | see three reasons below |
| B. Filter each frame to its declared subset, then `reflect()` | looks fail-closed | silently skips an undeclared *geometric* column — the original defect shape wearing the fix's clothes | recreates the bug |
| C. A typed frame value-object | strong typing | violates CLAUDE.md's pandas-in/pandas-out first line | artificial coupling |
| D. Fold the numpy grid reflections into the registry | one abstraction | an ndarray is a different kind of object from a labelled table | grids get a narrow behavioural guard instead |
| E. **`reflect()`/`reflect_columns()`, warn default, CI meta-assertion (chosen)** | single seam; complete-by-enumeration; no upgrade break | passenger columns with a missed geometric name get silent `invariant` | — |

**Why `"raise"` was reversed (Option A → E), all three reasons:**

1. **Unimplementable at D8.** `to_spadl_ltr` is called *inside* nine converters on a frame
   already carrying the caller's `preserve_native` passthrough columns, and its signature has no
   `extra_kinds`. `preserve_native` (`spadl/utils.py:1651`) takes a caller-supplied `list[str]`
   validated only for presence and schema non-collision, so the SPADL column universe is
   **unbounded by construction** and runtime registry completeness is unachievable there. A
   raise has no remedy reachable by the caller.
2. **Zero of the eight catalogued defects involved a caller-owned column.** All eight were
   library-owned (`vx`/`vy`, ball `x`/`y`, `x_smoothed`/`y_smoothed`, shape-graph arrays,
   `enriched_*`). The registry-completeness meta-assertion catches every one; a runtime raise
   adds nothing it does not already cover, while breaking the one case it reaches. For a
   passenger column like `possession`, `invariant` is the *correct* answer — a raise would fire
   on the case it gets right.
3. **A per-site policy split would recreate D3** — two same-named orienters with divergent
   contracts — one layer up.

Strictness is available to consumers who control their column universe (the lakehouse does) via
`warnings.filterwarnings("error", category=UndeclaredGeometricColumnWarning)`.

## Consequences

### Positive

- One seam for the point/vector/magnitude/direction-label contract instead of eleven
  hand-written copies. A new column added to any schema constant fails CI until it declares a
  kind — the anti-rot property, complete by enumeration (the `PURITY_ENTRIES`/`PUBLIC_ID_SCALAR`
  idiom, not the AST lint ADR-043 deleted).
- `bekkers_pi` away-team values are corrected (the D1/D2 home/away asymmetry is removed; measured
  away/home ratio 0.60 → 0.89 on real IDSSE data via an owner-gated e2e).
- The ADR-025 enrichment coordinates now mirror wherever actions are reflected.

### Negative

- **Residual hole, stated plainly:** a third-party caller attaching a geometric column whose name
  `GEOMETRIC_NAME` misses gets silent `invariant` treatment. That is the honest price of
  `to_spadl_ltr` having no reachable escape hatch — a scope limit, not a guarantee.
- `GEOMETRIC_NAME` has **measured blind spots**: infix axis tokens
  (`team_shape_centroid_x_attacking`, `defending_centroid_vx`) and one column with no axis token
  at all (`team_shape_defensive_line_height_attacking`) all fail `.match()`. It never *decides*
  anything (the registries cover library columns; the pattern only reports on passengers and
  drives the conformance guards), which is why widening it — trading false negatives for false
  positives on names like `max_x_velocity` — was rejected per ADR-043's lesson.
- **`_reproject_team_shape` (site 4)** is gated behaviourally, not by name (the pattern cannot see
  its columns). The pre-existing `test_team_shape_centroids_mirror_invariant` was measured
  **vacuous on the y-axis** (centred fixture; disabling the y re-projection left it green); the
  new site-4 gate uses an off-centre fixture with a disable-the-reflection both-sides partner
  (measured ON delta 0, OFF delta 34).

### Neutral

- No `angle` kind. A bearing under a point reflection goes to `θ+π` — neither invariant, nor a
  negation, nor a point reflection. Exactly one such column exists
  (`pre_shot_gk_angle_off_goal_line`), goal-referenced and unreachable from any reflection site;
  adding a kind now would be speculative API. It would surface as an undeclared column and warn.
- The numpy grid reflections are unchanged and guarded **behaviourally** by
  `tests/tracking/test_obso_orientation.py::TestEpvIsReflectedOnBothAxes` (injects a y-asymmetric
  EPV grid through the real `add_obso`). A shape assertion cannot work — slicing reversal is
  shape-invariant, so it cannot distinguish `[:, ::-1]` from `[::-1, ::-1]`.
- No converter output changes; no VAEP retrain beyond the `bekkers_pi`-consuming surfaces (see
  Related → downstream).

## Related

- **Specs:** `docs/superpowers/specs/2026-07-19-vector-reflection-consistency-design.md`
- **Plan:** `docs/superpowers/plans/2026-07-19-vector-reflection-consistency.md`
- **ADRs:** builds on ADR-028 (per-action LTR), ADR-033 (pure `add_*`), ADR-043 (public seams,
  complete-by-enumeration over lint); amends ADR-042 (D6).
- **Downstream (owner-run, NOT in this PR — one ordered pass, not independent follow-ups):**
  re-materialize `fct_action_context.pressure_on_actor__bekkers_pi` → retrain `rho` (both
  variants, must clear `ece<=0.10` / `|slope-1|<=0.25`) → re-run the xT-GK v2 deep-zone gate
  (its GO-leaning verdict was measured on broken pressure). Bundle with the 4.52.0 TF-35
  recompute — one wheel bump, one AC drain, one rho retrain, one gate re-run.

## Notes

Per-row, never aggregate: D2's mean bias is −1.1% and D3's is −0.002 because rows over- and
under-state in near-equal measure, so a mean-comparison gate passes vacuously on broken code.
Every new invariance test carries a non-vacuity partner that proves it can fail (three of this
PR's own guards nearly shipped vacuous — the `re.match`-anchoring bug and the y-symmetric
team-shape probe).

**GS finding (RESOLVED — was briefly tracked as an open item).** An earlier draft scoped the
bekkers real-data e2e to IDSSE because a Gradient Sports match (10502) measured an away/home
`bekkers_pi` ratio of 0.609, and the first hypothesis was a GS velocity-availability gap. That
hypothesis is FALSE and the ratio framing was the wrong invariant. Diagnosis on real data:

- GS carries full per-player velocity (`vx`/`vy` populated, 100% finite) and correct per-period
  `team_attacking_direction` labels; the re-projection fires (791/1414 actions flip). "GS lacks
  velocity" is falsified — `bekkers_pi` in fact *raises* without `vx`/`vy`, yet produced a value.
- `bekkers_pi` is a rigid-motion INVARIANT: fully mirroring the frame (positions **and**
  velocities, via the `reflect` seam) leaves the per-action pressure unchanged to machine
  precision — measured max |Δ| ≈ 6e-14 on **both** GS and IDSSE — while an incomplete
  velocity-unreflected mirror (the D1 defect reconstructed) moves it by ≈ 0.99. The fix is correct
  and the geometry is orientation-independent.
- The away/home RATIO therefore measures GENUINE match pressing asymmetry, not orientation. Across
  three GS matches it is 0.61 / 0.91 / 1.02 and on IDSSE 1.14 — it varies by match. GS 10502 is
  simply a lopsided match, not a defect and not a provider-specific bug.

The bekkers e2e was accordingly rewritten from the fragile away/home-ratio band into the correct,
match-independent MIRROR-INVARIANCE check (`tests/tracking/test_bekkers_home_away_asymmetry_e2e.py`,
run on **both** providers, self-non-vacuous via the incomplete-mirror teeth). Separately, the
4.52.0 obso orientation e2e — which IS orientation-sensitive, so *its* away/home ratio band is a
valid check — was unblocked: it had been crashing on the 4.52.0 `SyntheticEPVWarning` promoted to
an error (token-gated, so it never ran in CI), now filtered, and pinned to match 10502 for
determinism.
