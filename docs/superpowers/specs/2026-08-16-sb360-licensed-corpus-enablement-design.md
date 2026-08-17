# SB360 licensed-corpus enablement + visibility-aware count features — design

- **Date**: 2026-08-16
- **Status**: Draft — **revised after external-review rounds 1 & 2** ("materially plan-ready"; all
  round-1 + round-2 findings folded in: companions live on `add_action_context` only (R2-A/B), a
  feature-level source constant (R2-C), a renamed audit axis (R2-D), the golden test as the de-fork
  guarantee (R2-E), and the internal `_build_match` widened rather than a forked loader (Q5)). Not
  committed.
- **Size**: Wicked/Monstah (a new provider path + a raw-JSON events flattener + a de-fork of
  `build_sb360_coverage.py` + additive companion columns on three features + a licensed-corpus
  validation driver + an ADR-053 audit-axis extension).
- **Scope chosen by owner**: enablement + library features. The keeper-integration **research analysis
  is a follow-up cycle**, out of scope here.
- **Follows / amends**: ADR-053/054 (SB360 audit + port), ADR-055 (`_visibility.py`), ADR-009 (library
  ships raw info; consumer decides policy; calibration data never committed), ADR-038 (fail-closed
  private default), ADR-052/056 (driver seam + provenance registry). New ADR expected for the
  visibility-aware features + the audit visibility-axis.

---

## 1. Executive summary

The licensed StatsBomb 360 corpus **has landed in `pining-for-the-data`** (verified live this cycle: 30
matches under provider `statsbomb` — id `4047626` + 29 more; artifacts `{events, freeze_frames,
metadata, roster}`; **no `tracking` key**). But `scripts/_loader_pining.py` has **no statsbomb
build-branch**, so nothing can load these matches yet.

Five additive components:
1. **A statsbomb builder + a parallel `load_statsbomb_matches` generator** — download the 4 artifacts,
   flatten the raw JSON (scripts-side), and compose the three existing shaping functions into
   `(actions, frames, home_team_id, visible_area, report)`. **`load_matches` is left untouched at its
   5-tuple** (the visible_area travels only on the SB360-specific entry point).
2. **Fidelity from metadata** — thread `xy_fidelity_version`/`shot_fidelity_version` from
   `metadata.json` so the converter never needs to *infer* (it already warns when it does).
3. **Roster → player identity** — `roster.json` + the events' `player_id` give per-match, per-KEEPER
   identity on the **actions** (freeze-frame rows are anonymous by construction). Absent fields via
   `.get()`.
4. **Visibility-aware count features (raw + companion; strictly additive)** — `nearest_defender_distance`,
   `receiver_zone_density`, `defenders_in_triangle_to_goal` keep their **raw visible count/distance
   unchanged, always**. The visibility opt-in lives on **`add_action_context` only** (which already
   bundles the three): passing `visible_area=` there appends companion `<feature>_observed_fraction` +
   `<feature>_observed_source` (consumer applies `where(fraction == 1.0)` — policy at the edge). The
   per-Series functions and `tracking_default_xfns` are untouched, so the calibration path is additive
   by construction.
5. **A licensed-corpus validation artifact** — a `_driver.py` pass that loads all 30 matches, runs the
   ADR-053-`works` aggregator set (per emitted column, not a name list), and reports coverage +
   the honest-degrade distribution. **Owner-run; per-match shards to a gitignored root; only the
   reconciled aggregate + provenance committed; licensed data never committed.**

**No behaviour change for any existing caller or corpus** — a new SB360-only entry point (existing
`load_matches` and its unpack sites untouched) and pure-addition companion columns. The keeper-box
TF-24 Stage-2 corpus is unaffected even though the three features are in `ALL_FEATURES`, because their
primary columns never change and it supplies no `visible_area`.

---

## 2. Context (grounded)

- Live API (verified with the owner token this cycle): `GET {base}/statsbomb/matches` →
  `{provider, matches:[{id, artifacts}]}`, 30 matches, artifact keys `{events, freeze_frames,
  metadata, roster}`, no `tracking`. Tracking corpus separate/unchanged (skc 108 + idsse 7 + gs 64 = 179).
- **The composition spans three modules (M1):** `silly_kicks.spadl.statsbomb.convert_to_actions(events,
  home_team_id, xy_fidelity_version=, shot_fidelity_version=, …)` (statsbomb.py:110); the port's
  `silly_kicks.providers.statsbomb.shape_snapshots(frames_raw, actions, *, fidelity_version=)` →
  `(snapshots, visible_area, JoinReport)` (parse.py:210); `silly_kicks.tracking.snapshot_to_tracking_frames(snapshots,
  actions)` → `(frames, links)` (_snapshot.py:24, **not** exported by the port).
- `convert_to_actions` **warns** (not silent) when it infers fidelity (statsbomb.py:204-209) — so §5's
  value is *avoiding* inference, not silencing it.
- `visible_area` is a **per-action DataFrame** (`action_id → (N,2) polygon`) produced only by
  `shape_snapshots`; it is **not** a frames column. It flows as a separate arg, exactly as
  `add_visible_area_coverage(actions, *, visible_area=…)` already takes it (_visibility.py:151-157).
- The three features' kernels (verified): `_defenders_in_triangle_to_goal` (_kernels.py:184) — a
  triangle to fixed posts `(105, 30.34)/(105, 37.66)`, **convex, already the query region**;
  `_receiver_zone_density` (_kernels.py:140) — a **disk** `dist <= radius` (default 5 m) around
  `(end_x, end_y)`; `_nearest_defender_distance` (_kernels.py:85) — a **min-distance**, NaN when no
  opposite rows (_kernels.py:92,98). All three are in `calibration/_features.py::ALL_FEATURES`.

---

## 3. Goals / non-goals

**Goals** — load any of the 30 licensed matches; correct fidelity + roster identity; the three features
gain an opt-in visibility *companion* (primary unchanged); a reproducible coverage/validation artifact;
**de-fork** the raw-JSON parse so `build_sb360_coverage.py` and the loader share one flattener.

**Non-goals** — the keeper-integration analysis (next cycle); wiring visibility into other features;
SkillCorner/broadcast partial-visibility (no `visible_area` polygon; deferred, see M3b); committing any
licensed data or a licensed fixture.

---

## 4. Component 1 — the statsbomb loader path (B1: non-breaking)

**Files:** `scripts/_loader_pining.py` (download roles + a `load_statsbomb_matches` generator +
`build_statsbomb_match`); a scripts-side flattener `scripts/_sb_raw.py` (H5); tests.

**4.1 Download roles.** In `_download_artifacts`, add a statsbomb branch mapping
`{events, freeze_frames, metadata, roster}` (no `tracking`). Additive dispatch; the other three
branches are byte-identical.

**4.2 `build_statsbomb_match(paths, match_id) -> (actions, frames, home_team_id, visible_area, report)`.**
Composition: parse `metadata.json` (home_team_id, fidelity); flatten `events.json.gz` via `_sb_raw`
(H5) → `convert_to_actions(events, home_team_id, xy_fidelity_version=…, shot_fidelity_version=…)`;
parse `freeze_frames.json.gz` → `list[dict]` → `shape_snapshots(frames_raw, actions,
fidelity_version=xy_fidelity_version)` → `(snapshots, visible_area, join_report)`;
`snapshot_to_tracking_frames(snapshots, actions)` → `(frames, links)`; parse `roster.json` → enrich
`actions` (§6). **Do NOT run `_preprocess`** (freeze-frames have no temporal history;
`speed_source="unavailable"` already set). State that fork loudly.

**4.3 Share the retry/cache machinery; don't fork it (Q5/B1).** A fully parallel `load_statsbomb_matches`
would duplicate `load_matches`'s download/retry/cache/manifest orchestration
(`_build_match_with_retry`, `_download_artifacts`) — itself a fork. Instead:
- **Widen the INTERNAL `_build_match`** (`_loader_pining.py:347`) to return `(actions, frames, home,
  visible_area, report)` — `visible_area = None` for the non-snapshot providers; the statsbomb branch
  produces it. Update its one internal unpack (`:246`) and pass-through in `_build_match_with_retry`
  (`:262`). This is internal only.
- **`load_matches` is NOT modified at its public boundary** — it unpacks the widened internal tuple and
  yields its existing **5-tuple** `(provider, match_id, actions, frames, home)` (drops `visible_area`),
  so `:258` and every unpack site (`build_gkdv_arm_values.py:187`, `tests/causal/test_causal_e2e.py:99`)
  are untouched.
- **`load_statsbomb_matches(...)` is a THIN wrapper** over the same machinery that yields
  `(provider, match_id, actions, frames, home, visible_area)`. The §8 driver and any SB360 consumer use
  it; the calibration path uses `load_matches` and never sees `visible_area` (SB360-only, validation-only).

This is the fix for the round-1 "6th public tuple breaks the additive guarantee" defect, without a
second copy of the download/retry logic.

---

## 5. Component 2 — fidelity from metadata (M2)

The seams already accept it (`convert_to_actions(…, xy_fidelity_version=, shot_fidelity_version=)`;
`shape_snapshots(…, fidelity_version=)`) and the converter **already warns** when it infers
(statsbomb.py:204-209). The loader path reads the value from `metadata.json` and supplies it, so
inference is not needed at all — no silent default. No signature change.

---

## 6. Component 3 — roster → player/keeper identity (L2)

Freeze-frame rows carry no identity (synthetic sequential `player_id`; parse.py:222-226) — identity is
an **action** fact. `build_statsbomb_match` parses `roster.json` into a `player_id → {name, position,
jersey, team}` lookup and adds identity columns to `actions`, keyed by the SPADL `player_id`. **Use
`.get()` throughout** — per the StatsBomb pining contract unset fields are *absent*, not null (mirror
the GS roster parse at `_loader_pining.py:770-779`). Do **not** overload `preserve_native` (that is for
event-native fields; the roster is a separate join). Document the split so nobody retries per-frame-row
aggregation.

---

## 7. Component 4 — visibility-aware count features (raw + companion; H1/H2/H3 + R2-A/B/C/F)

**Decision (H1, owner-approved best-practice):** the primary columns stay the **raw visible
count/distance** — a true "what was seen" observation, never a fabrication, **byte-identical whether or
not `visible_area` is supplied**. Visibility rides in *companion* columns; the consumer applies any
NaN/threshold policy itself (`count.where(fraction == 1.0)`). Strongest additive guarantee; policy at
the edge (ADR-054:25). NaN-on-partial is rejected in §11.

**Companions live on `add_action_context` ONLY (R2-A) — the per-Series functions are untouched.** The
three public functions return a single `pd.Series` (features.py:331/384/412), so emitting extra columns
from them would make the return type unstable and put the primary values one edit away from changing.
`add_action_context` (features.py:442-486) is already the DataFrame enrichment that bundles all three
count features **from one `ctx = _resolve_action_frame_context(actions, frames, links=links)`** plus the
four linkage-provenance columns (`frame_id`, `time_offset_seconds`, `link_quality_score`,
`n_candidate_frames`) — the companions are the same shape. So: add `visible_area: pd.DataFrame | None =
None` to **`add_action_context` only**; when supplied, append the six companion columns
(`<feature>_observed_fraction` + `<feature>_observed_source` × 3). Leave the three per-Series functions
and `tracking_default_xfns` (features.py:489-494, which uses the per-Series functions, NOT
`add_action_context`) untouched. **The calibration/xfns path therefore never computes a companion →
additive by construction**, and the primary columns cannot change; the §8 driver consumes a DataFrame
anyway and calls `add_action_context(actions, frames, visible_area=…)`.

**One enrichment owns the join, `unlinked`, and per-action classification (H2 + R2-B — no fork).**
`region_observed_fraction` returns **NaN** (not a sentinel) for absent/degenerate polygon and zero-area
region, and **raises** on a non-convex region (_visibility.py:120-132). Model the whole thing on
`add_visible_area_coverage`: `add_action_context` already holds `ctx` (pointers + opposite rows), so it
owns the single **ADR-019 `canonical_id` action↔polygon join** (_visibility.py:218-221) and sets
`unlinked` from `ctx.pointers` — exactly as it already sets `frame_id = NaN` for unlinked actions
(features.py:459). This keeps the join and the `unlinked` decision single-sourced instead of forked
three ways.

Factor **one geometry-only** helper `classify_region_observation(polygon, region) -> (fraction, source)`
in `_visibility.py`, called once per (action, region):
- It returns `source` from a **feature-level constant, NOT the shared `VISIBLE_AREA_SOURCE_VALUES`
  (R2-C)** — that public constant is pinned exactly (`test_visibility.py:191`) and exported
  (`tracking/__init__.py:48`), and it means *visible-area-polygon provenance*, a different axis than a
  query-region's own quality. Define a feature-level superset that reuses the four polygon tokens
  (`observed, no_polygon, degenerate_polygon`) **plus `degenerate_region`** (a collapsed region-of-
  interest). **`unlinked` is NOT a geometry property** and is emitted by the enrichment from `ctx`, not
  by this helper (a `(polygon, region)` function structurally can't know it, R2-B).
- `fraction` is the numeric outcome (`1.0` / `(0,1)` / `0.0` / `NaN` unmeasurable) — it carries the
  observed/partial/unobserved distinction, so we do not invent a parallel outcome enum (B2: provenance
  and outcome are orthogonal).

**Per-feature region `R` (H3 — they are NOT three ready polygons):**
- `defenders_in_triangle_to_goal` — `R` = the ball→posts triangle. Convex; it *is* the query region.
- `receiver_zone_density` — a **disk** (radius 5 m, features.py:384). Approximate as an **inscribed**
  convex polygon.
- `nearest_defender_distance` — a **disk of radius = the computed distance**. Also inscribed.
- **Inscribe, never circumscribe (the honesty invariant):** an inscribed disk under-covers → coverage
  is a **lower bound** → we never claim "fully observed" when it wasn't. Pin a fixed vertex count
  (16–24; a resolution/honesty trade-off, not a correctness one — inscription already fixes the
  direction; benchmark where `fraction` stabilises on the open match).
- **Degeneracy guard = zero-area, NOT non-convex (R2-F).** All three regions are convex by construction,
  so `region_observed_fraction`'s non-convex `raise` (_visibility.py:128-132) is unreachable here; the
  real degenerate case is **zero-area** (goal-line anchor → zero-area triangle; radius → 0), which
  returns NaN. The classifier maps that to `source=degenerate_region`, `fraction=NaN` — never a raise
  across the corpus.
- **NaN-distance special case:** `nearest_defender_distance` is NaN when no defender is visible
  (_kernels.py:92,98); its disk is then undefined. Special-case it — propagate the primary NaN and set a
  no-measurement `source`, do **not** call the classifier with a NaN radius.

**Orientation invariant (R2-G).** On the SB360 snapshot path the polygon (SPADL via `sb_xy_to_spadl`)
and each region are both raw-SPADL and share one frame because the count kernels don't re-orient (fixed
goal at x=105, _kernels.py:184-234). Pin this as an explicit invariant: a future re-oriented provider
(SkillCorner-broadcast, deferred in §3) must re-orient the polygon too, or coverage is computed across
mismatched frames.

---

## 8. Component 5 — the licensed-corpus validation artifact (H4/B3/M3)

**Files:** a driver `scripts/validate_sb360_licensed_corpus.py` (adopts `_driver.py`); an artifact under
`docs/research/sb360_licensed_coverage/` (**aggregate only**).

**8.1 What it runs (H4 — battery is net-new, and the degrade IS the finding).**
`build_sb360_coverage.py` runs **none** of the `add_*` battery (only `defending_gk_visible`,
`acting_side_gk_visible`, `observed_pitch_fraction`, `convert_to_actions`, `actiontypes_df` —
build_sb360_coverage.py:281/282/291/205/113). So this driver **adds** an aggregator run; the only thing
it *replaces* is the coverage metrics (the n=16–22 → n=30 swap). Choose the aggregator set **from the
ADR-053 registry, per emitted column** (the `works` verdicts — 299/486 per-column, 4 `silent_degrade`
all `add_ghost_gk`, ADR-053:78), **not** an approximate "~15 aggregators" name list. Snapshot input
honestly degrades many columns (`speed_source=unavailable`, synthetic team/player ids, one frame per
action; `add_cover_shadows` → `no_signal` on `gk_absent`, ADR-055:92-95) — **report the
`honest_nan`/`silent_degrade` distribution as a result**, not a failure. Run the three count features
**with** `visible_area` and report the `observed_source`/`fraction` distributions.

**8.2 Licensed-data discipline (B3 — a real leak surface).** `_driver.py::for_each` writes **one shard
per match** (`_driver.py:596-628`). `build_sb360_coverage.py` deliberately writes shards to a
**gitignored top-level root** (`DEFAULT_SHARD_ROOT="sb360_coverage_shards"`, :87/:390) precisely because
`docs/research/.../_shards` is *not* covered by the gitignore anchor. So: (a) shards go to a **gitignored
root**, only the **reconciled aggregate manifest** lands under `docs/research/`; (b) a CI test asserts
**no per-match row leaves the gitignored root**; (c) the never-commit rule is **ADR-009:11**
("calibration data … must never be committed"), with **ADR-038:58-59** the fail-closed private default —
NOT ADR-038 for the commit policy (round-1 mis-cite); (d) the discipline is stated as: assert via
`is_public_row`/`assert_public_corpus` that **no non-public row enters the committed aggregate** — the
repo has no file-level `restricted:` marker, so do not claim `_corpus.py` "labels the artifact
restricted."

**8.3 Provenance (M3).** `git_provenance()` returns keys `commit`/`dirty`/`tree_state`/… (not
`run_commit`); the driver calls `require_clean_tree(git_provenance(), …)` in `main()` and **stamps**
`run_commit`/`run_tree_dirty` into the artifact itself (caller-invoked, as
build_sb360_coverage.py:399-403 does). Enroll the driver in ADR-056's `ARTIFACT_DRIVERS`
(`test_provenance_wiring.py`) — that gate is real and enforced.

---

## 9. Testing & CI

- **Additive guarantee (now structural, R2-A).** Two gates: (a) the three **per-Series functions and
  `tracking_default_xfns` are unchanged** (they never take `visible_area`), so the calibration path is
  untouched by construction; (b) **`add_action_context`'s three primary columns are byte-identical with
  and without `visible_area`** — only the six companion columns are added. This is the measured evidence
  for §10's "no TF-24 re-run" (ADR-055-style).
- **Open-match E2E (L1).** Extend the loader test beyond a schema check to run the whole chain
  (`load_statsbomb_matches` → snapshots → the three features **with** `visible_area`) on an **OPEN**
  360 match (redistributable), asserting: real primary counts; companion `observed_fraction` in `[0,1]`
  or NaN; `observed_source` in the closed vocabulary; a partially-observed region yields
  `fraction ∈ (0,1)`; no polygon → `fraction=NaN`, `source=no_polygon`, primary unchanged; a degenerate
  region → `source=degenerate_region`, no raise; the NaN-distance case propagates. Licensed matches are
  exercised only in the owner-run driver (§8), never in CI.
- **Flattener de-fork golden (H5).** A golden test cross-checking `_sb_raw`'s flattened output against
  `statsbombpy` on an open match, so the two parses cannot drift.
- **Fidelity threading** — metadata `xy_fidelity_version=2` reaches the converter (scaled coords differ
  from fidelity-1). **Roster identity** — columns attached, keyed by `player_id`, `.get()`-tolerant.
- **ADR-053 re-adjudication (H4/R2-D — a real deliverable, not "register").** ADR-053 mandates a
  CI-re-derived observation + a human adjudication per column on two axes (velocity, and a
  roster-ablation axis ADR-053:61 already calls "**visibility**"). Supplying `visible_area` adds a
  **distinct new axis — name it the *observed-region* axis, NOT "visibility"**, to avoid colliding with
  the existing roster-ablation one. Scope it to the three count features' new companion columns under
  SB360 rather than a global third cross-product over all 486 verdicts, then re-adjudicate those columns.
- **Dual-major** (`.venv` 3.10 / `.venv312` 3.12); lint + pyright at CI scope.

---

## 10. Coordination, risk, provenance

- **No `load_matches` change (B1)** → the TF-24 unpack sites and the calibration path are untouched; the
  three features' **primary columns never change** → **no Stage-2 re-run, no retrain**, tied to the §9
  additive gate as measured evidence.
- **De-fork (H5/R2-E — sized honestly).** Put the events flattener in `scripts/_sb_raw.py` (port stays
  pure-shaping, `statsbombpy`-free — ADR-054:74-77). The **load-bearing de-fork guarantee is the §9
  golden equivalence test** (`_sb_raw` ≡ `statsbombpy` on an open match) — two equivalence-pinned
  implementations cannot silently drift. A *full* re-point of `build_sb360_coverage.py` is **bigger than
  a parser swap** (it acquires events + 360 + metadata via `statsbombpy`'s fetch+flatten —
  `sb.events`/`sb.frames`/`sb.matches`, three acquisitions, not one call), so treat it as
  **optional/sized**: land it if it fits, but the golden pin already prevents the fork without it.
- **C4 / SB360-in-pining** — the keeper-box session is already documenting "pining serves SB360" +
  fixing the "StatsBomb SB360" redundancy (owner-confirmed). This cycle should NOT duplicate that; if it
  documents anything new it is the *silly-kicks-consumes-SB360* relationship (a container/relationship
  edit), coordinated with that session to avoid a C4 collision.
- **New runtime dependency: none** (`statsbombpy` stays a `scripts/`-only dep, ADR-054's per-port rule).
- Both cycles edit `_loader_pining.py` additively (a new branch here); reconcile at merge.

---

## 11. Rejected alternatives

- **NaN-on-partial primary column (round-1 proposal):** rejected — it destroys the true "N seen"
  observation and decides policy for every consumer (the exact "decide for the consumer" ADR-055:255-256
  flagged when deferring feature-wiring). Its one advantage (loud) is **moot here**: calibration never
  supplies `visible_area`, so the only consumers are the validation driver + future SB360 users, who
  read the companion. Raw + companion is strictly more additive and keeps policy at the edge (ADR-054:25);
  a consumer wanting the NaN semantics writes `count.where(fraction == 1.0)`.
- **Weighted count (`count / fraction`):** rejected — fabricates an estimate, hides uncertainty.
- **Non-opt-in / schema-change visibility:** rejected — frames don't carry `visible_area`; the kwarg is
  the only coherent seam and makes the change additive.
- **`load_matches` 6th tuple / widened return:** rejected — breaks the 5-tuple iterator contract (Hyrum;
  every unpack site). A parallel `load_statsbomb_matches` carries the SB360-only `visible_area`.
- **Events flattener inside `providers/statsbomb`:** rejected — contradicts ADR-054 (the port takes
  already-loaded payloads and adds no runtime dep, ADR-054:74-77). Flattener is scripts-side.
- **A parallel `partial/unobserved/no_polygon` source enum:** rejected — mis-cites the real constant,
  drops `degenerate_polygon`/`unlinked`, and fails the runtime post-condition; provenance (source) and
  outcome (fraction) are separate axes.
- **Companion columns on the per-Series functions (R2-A):** rejected — they return a single `pd.Series`,
  so emitting extra columns makes the return type unstable and puts the primary values one edit from
  changing. `add_action_context` (already a DataFrame enrichment bundling the three, from one `ctx`) is
  the home; the calibration/xfns path uses the per-Series functions and never sees a companion.
- **Widening the shared `VISIBLE_AREA_SOURCE_VALUES` for `degenerate_region` (R2-C):** rejected — it is
  pinned + exported (public API + test) and means *polygon* provenance, a different axis than a query
  region's own quality. A feature-level superset constant keeps the shared constant's meaning intact.

---

## 12. Resolved (round 2) — remaining items are plan-time detail

All five round-2 opens are resolved and folded in:
1. **Inscribed-disk vertex count** → pin a fixed N (16–24), benchmark where `fraction` stabilises on the
   open match (§7). Resolution/honesty trade-off, not correctness. *(plan pins the value)*
2. **`degenerate_region`** → a **feature-level** superset constant, NOT a widening of the pinned/exported
   `VISIBLE_AREA_SOURCE_VALUES` (R2-C, §7). *(plan names the constant + members)*
3. **Audit axis** → the new axis is the **observed-region** axis (not "visibility" — collides with
   ADR-053:61's roster-ablation axis), scoped to the three companion columns (R2-D, §9). *(plan designs
   the harness extension)*
4. **Aggregator set** → per-column `works` from the ADR-053 registry; the `visible_area` opt-in stays on
   the three count features only this cycle (§8). Confirmed.
5. **Loader surface** → widen the internal `_build_match` to carry `visible_area` and make
   `load_statsbomb_matches` a thin wrapper over the shared `_build_match_with_retry` — no forked download/
   retry loop, `load_matches` public 5-tuple untouched (Q5, §4).

The genuinely plan-time details (the vertex N, the feature-level constant's membership, the observed-
region harness wiring, and whether the optional `build_sb360_coverage.py` re-point lands) are the
`writing-plans` step's to nail — the design is settled.
