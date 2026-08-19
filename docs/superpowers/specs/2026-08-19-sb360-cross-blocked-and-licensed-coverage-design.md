# SB360 completion — StatsBomb `cross_blocked` enablement + licensed coverage report — design

- **Date**: 2026-08-19
- **Status**: Draft — not committed.
- **Size**: Small/Medium. Part 1 = an ad-hoc corpus probe + a conditional converter change (`_flatten_extra` +
  one masked column, mirroring the existing `shot_blocked` site) + offline verification fixtures + an ADR
  amendment. Part 2 = a deterministic render script reading a committed artifact + the rendered `coverage.md`.
- **Follows / amends**: **amends ADR-046** (block-detection converter columns — un-defers StatsBomb
  `cross_blocked`, which ADR-046 recorded as standing verification debt). Respects ADR-019 (`id_compat`
  dtype-safe id comparisons), ADR-045 (block-detection columns are reflection-`"invariant"`), ADR-037
  (artifact provenance — deliberately N/A here; see §7), ADR-052/056 (driver seam + script-population
  registry — the render enrolls as a non-driver renderer), ADR-063 (velocity-less lift — the 40→31
  fully-NaN result the coverage report renders).
- **Retrain**: **none in silly-kicks** (Part 1 is purely additive — a column that is all-`pd.NA` today
  gains real values for one provider; ADR-045 declares it `"invariant"`; `cross_blocked` is consumed by no
  `vaep/`/`atomic/` feature (verified). Part 2 is a document). **Downstream** consumers see a live-surface
  value change — see §7.3.
- **C4**: unchanged (`cross_blocked` already exists in `SPADL_COLUMNS`; Part 1 populates it, adds no
  action-coupled aggregator/backend/model. Part 2 is a doc + a renderer script).
- **Commits**: **one commit for the whole cycle.** Nothing here is a provenance-guarded artifact driver
  (the probe prints to stdout; the render reads a committed parquet and is unguarded, like
  `render_sb360_matrix.py`), so no clean-tree gate ever forces a mid-branch commit.

---

## 1. Executive summary

Two independent, additive halves of the same SB360 theme, shipped together in one release. **No version
number is claimed here** — it is assigned only when everything is implemented and tested, from `main`
(§7.2). The release *classification* depends on Part 1's outcome (library change vs docs/tooling-only).

**Part 1 — un-defer StatsBomb `cross_blocked` (conditional on a probe).** `spadl/statsbomb.py:282` emits
`actions["cross_blocked"] = _blocked_flag(len(actions))` — **all-`pd.NA`**, tagged `# deferred (BD-2)` —
while its sibling `shot_blocked` (`:277`) carries a real mask. ADR-046 deferred it as an
"n=1-verified, fragile `related_events` join". A direct probe of the three committed real-match fixtures
shows the mechanism is **clean, not fragile**: the `cross → related_events → Block` link is symmetric,
unambiguous, and textbook — the "fragility" was actually the **tiny base rate** of blocked crosses
(1 across 81 open-play crosses ≈ 1.2%), which is exactly what a nullable boolean exists to carry.

We run a **wider open-data probe as step 0** with a **pre-registered decision rule**, and un-deferral is
**conditional on the measurement**. If the join resolves cleanly at corpus scale → ship the mask
(`applicable =` open-play cross, `blocked =` cross links to an opposing-team `Block`). If the probe
surfaces a disqualifying reliability problem → **keep it deferred and record the measured reason** (an
upgrade over ADR-046's `n=1`). Either outcome amends ADR-046 with real numbers.

**Part 2 — render the licensed coverage as `coverage.md`.** `docs/research/sb360_licensed_coverage/`
holds only `coverage.parquet` (7,740 tidy rows, 30 matches) + `manifest_all.json` — no human-readable
narrative, unlike the open-360 `docs/research/sb360_coverage/coverage.md`. A deterministic render script
(mirroring `render_sb360_matrix.py`: reads a committed artifact, writes a doc, no provenance guard,
enrolled in the ADR-056 population registry as a renderer, **not** golden-gated) emits the report:
frame-existence per GK-domain type, the **31-of-230 fully-NaN** battery columns (the ADR-063 velocity-less
result made visible), the ADR-062 companion distributions, real pitch coverage (~27%), roster resolution
(100%), and the `add_space_creation` single-team-frame raises.

**If Part 1's honest outcome is "keep deferred," Part 2 still ships** — it is independent and fully grounded.

---

## 2. Context (grounded)

### 2.1 The `cross_blocked` site and the block-flag contract

- `spadl/statsbomb.py:277-282`:
  ```python
  actions["shot_blocked"] = _blocked_flag(
      len(actions),
      applicable=(events["type_name"] == "Shot").to_numpy(),
      blocked=(events["_shot_outcome"] == "Blocked").to_numpy(),
  )
  actions["cross_blocked"] = _blocked_flag(len(actions))  # deferred (BD-2)
  ```
  Both are computed **before** the non-action filter/sort/dribble insertion (`:284-301`), so the mask is
  length `len(actions)` in the pre-filter, pre-dribble frame and aligns row-for-row with `events`.
- `spadl/utils._blocked_flag(n, *, applicable=None, blocked=None)` (`utils.py:1559`): 3-valued nullable
  `"boolean"`. `applicable=None` → all `pd.NA`. Otherwise `blocked` (NA→`False`, avoiding the
  `astype(bool)` trap) on `applicable` rows, `pd.NA` elsewhere. A non-applicable row is "unknown", never
  `False`.
- `shot_blocked` is trivial because a StatsBomb shot carries `shot.outcome.name == "Blocked"` directly.
  A blocked **cross** carries **no** pass outcome of "Blocked"; the only encoding is a separate `Block`
  event linked via `related_events`.

### 2.2 `related_events` is already reachable (no new parse of the raw stream)

- `scripts/_sb_raw.flatten_events` (`_sb_raw.py:33`) copies the seven `_TOP_LEVEL_KEYS`
  (`id, period, timestamp, team, player, type, location`) to dedicated columns and puts **everything else**
  — including top-level `related_events` — into the `extra` dict verbatim. `statsbombpy.sb.events(fmt="dict")`
  returns the same shape.
- The converter's `_flatten_extra` (`statsbomb.py:73-106`) extracts named nested fields from `extra`
  (e.g. `_pass_cross` at `:86`, `_shot_outcome` at `:91`) but does **not** currently extract
  `related_events`. It is present in `extra`, just unlifted.
- The existing open-play cross predicate already exists inline (`statsbomb.py:442-466`): a `Pass` with
  `pass.cross == True` and `pass.type.name ∉ {Free Kick, Corner, Goal Kick, Throw-in}` → SPADL `cross`
  (`:466, :506`). Set-piece crosses (`corner_crossed`/`freekick_crossed`) are a distinct type and stay
  `pd.NA` (ADR-046 v1 scope).

### 2.3 Probe of the three committed fixtures (measured this cycle)

`tests/datasets/statsbomb/raw/events/{7298,7584,3754058}.json` are committed real matches.

| match | events | open-play crosses | `Block` events | crosses `related`→`Block` | reverse `Block`→open-play-cross |
|---|---|---|---|---|---|
| 7298 | 3793 | 21 | 49 | 0 | 0 |
| 7584 | 3715 | 32 | 35 | **1** | **1** |
| 3754058 | 3576 | 28 | 42 | 0 | 0 |

- **Symmetric**: the forward (cross→Block) and reverse (Block→cross) counts agree exactly (1 vs 1 in 7584,
  0 vs 0 elsewhere) — the link is not direction-dependent.
- **Textbook**: the single 7584 case is cross `e8edd276-8490-456c-b221-240d128f61f1` (min 89, `pass.height`
  = Low Pass, `pass.outcome` = Incomplete, `end_location` [110, 31] near the byline, `related_events` =
  `["a5d2f9c6…" → Block]`). Exactly one related Block, no ambiguity.
- **Base rate ≈ 1.2%** (1 / 81 open-play crosses). This is the whole of ADR-046's "n=1" — a rarity fact,
  not a broken join.

### 2.4 The licensed coverage parquet (what Part 2 renders)

- `docs/research/sb360_licensed_coverage/coverage.parquet`: 7,740 rows × 7 cols
  (`match_id, kind, subject, metric, value, denominator, detail`), 30 matches. `manifest_all.json`:
  `{generation: "100c80a1b37b40a6", n_attempted: 30, n_failed: 0, run_commit: "cf2f155…", run_tree_dirty: false}`.
- `kind` inventory: `battery_column` (6831), `companion_source` (450), `frame_coverage` (166),
  `pitch_coverage_source` (120), `companion_fraction` (90), `pitch_coverage` (30), `roster` (30),
  `battery_raises` (23).
- Measured now from the parquet: **31 of 230** battery columns are fully-NaN across the corpus (mean
  populated fraction 0) — exactly the velocity-derived kinematics, ADR-063 Tier-2 suppressed physical
  quantities, and constitutively-tracking columns (`add_shot_goalmouth.*`, `add_xcross_attempt.xcross_attempt`,
  `add_das.*`, `add_ghost_gk.ghost_gk_x/y`, `add_player_influence.reachable_area_*`, etc.). Roster
  resolution = **100%** on all 30. Pitch coverage mean = **0.273** (min 0.200, max 0.366). The 23
  `battery_raises` are all `add_space_creation` "opponent perspective requires exactly two team ids in the
  linked frame" — a real freeze-frame sometimes carries only one team's players near the action.
- The **40→31** headline: 40 was the pre-ADR-063 fully-NaN count; the committed parquet (at `cf2f155`,
  the 4.85.0 tip) proves **31**. The report renders 31 and cites 40 as clearly-labelled prior state.

### 2.5 The renderer precedent

`scripts/render_sb360_matrix.py` reads a committed registry, writes `behaviour_matrix.md`, and carries
**no** provenance guard by design ("reads a COMMITTED registry and writes a document… Adding the guard
here would be cargo-cult"). No test golden-gates its output (verified: no `tests/` reference to
`render_sb360_matrix` or `behaviour_matrix.md` as a golden). Part 2 mirrors this exactly.

---

## 3. Part 1 — `cross_blocked` enablement

### 3.1 Step 0 — the probe (decisive, ad-hoc, prints to stdout)

A `scripts/` investigation (`scripts/probe_sb_cross_blocked.py`) that measures the join over
**StatsBomb open data** (statsbombpy fetch of a fixed competition/season slice, ~30–50 matches; the three
committed fixtures anchor the offline core). It reuses `_sb_raw.flatten_events` and the converter's
open-play cross predicate so the probe measures the **same** mask the converter would emit.

It is **not** an artifact driver: it prints a summary table to stdout and writes **no committed file**.
The wider open-data numbers become **cited measurements** in the ADR amendment and the research note —
the same evidentiary form ADR-046 itself used for the original GS `crossOutcomeType` probe (WC2022 match
10502). This is what keeps the cycle to a single commit (no provenance guard, runs on a dirty tree).

**Why open data is valid evidence for the licensed corpus.** `related_events` is a standard StatsBomb
event field and the **same** `spadl.statsbomb` converter path serves both open and licensed SB360, so the
open-data measurement is a proxy for the whole StatsBomb provider — it generalizes **by construction**
(identical schema + identical code path), not by direct measurement of licensed rows. The licensed corpus
is un-probeable here by policy (never committed, owner-token-gated), so the open-data probe is the best
available evidence and is the correct instrument regardless.

Per open-play cross the probe records:

1. **Base rate** — fraction with ≥1 `related_events` uuid resolving to a same-match `Block`.
2. **Ambiguity** — count of crosses linked to >1 `Block`; count of `Block`s linked to >1 pass; count of
   crosses with empty/absent `related_events`.
3. **Team side** — of the linked `Block`s, how many are by the **opposing** team vs the same team
   (a same-team link is a data anomaly and must resolve to `False`, mirroring ADR-046's sportec
   own-team-deflection case).
4. **Set-piece leakage** — confirm no `corner_crossed`/`freekick_crossed` row is caught (the mask is
   scoped to open-play `cross`).
5. **Symmetry** — does the linked `Block` also list the cross in *its* `related_events`? (informational.)

### 3.2 Pre-registered decision rule

**Ship the mask iff ALL hold** (thresholds are the pre-registration; the probe reports the actuals):

- **R1 — join present**: `related_events` is populated on the crosses; the fraction of open-play crosses
  with an *empty/absent* `related_events` list is not systematically high (a systematically empty field
  would mean the signal is unencodable on that corpus). Threshold: < 5% of open-play crosses have absent
  `related_events`.
- **R2 — same-team links: a semantic-validity check, NOT an output-correctness one**: the deterministic
  opposing-team rule (§3.3.2) already resolves any same-team `Block` link to `False`, so the emitted mask
  is correct on those rows regardless. R2 does not exist because same-team links corrupt output — they
  don't. It exists because a **high** same-team rate would mean our *model* of `related_events` is wrong
  (the link would not mean "a defender blocked this cross"), which invalidates the join's meaning. Threshold:
  same-team links are absent, or a documented negligible fraction (< 1% of linked crosses). Above that →
  keep-deferred, because the *semantics* are unreliable, not because any row is mis-set.
- **R3 — bounded ambiguity**: multi-`Block` crosses are handled deterministically (§3.3, "≥1 opposing
  Block" is monotone, so a second Block does not change the boolean) and multi-pass `Block`s do not cause
  a cross to be flagged from an unrelated pass's block. Threshold: the rule is well-defined on 100% of
  linked cases (it is, by construction — see §3.3); the probe confirms no case violates the stated
  reading.

**Keep deferred iff** R1 fails (signal unencodable) **or** R2 fails (semantics unreliable — a high
same-team-link rate invalidates our reading of `related_events`). In that case the deliverable is the **measured** deferral
reason in the ADR amendment + research note — strictly better than the current `n=1` sentence.

`shot_blocked`-style thinness (a low base rate) is **not** a disqualifier: a rare-but-correct signal is
the intended use of the nullable boolean. R1–R3 gate *correctness/encodability*, never *frequency*.

### 3.3 Implementation (iff ship)

1. **`_flatten_extra`** (`statsbomb.py:73-106`): add
   `events["_related_events"] = extra.str.get("related_events")` (a list of uuids, or NaN). One line,
   alongside the existing `_pass_*`/`_shot_*` extractions.
2. **The mask**, built at the `cross_blocked` site (`statsbomb.py:282`, replacing the deferred call),
   **before** the non-action filter so all uuids resolve in the full events frame:
   - `applicable` = the open-play cross mask, reusing the **same** predicate the type dispatch uses
     (`is_pass & (_pass_cross == True) & ~_pass_type.isin(["Free Kick","Corner","Goal Kick","Throw-in"])`).
     Factor that predicate into a small local so the type dispatch (`:466`) and the block mask share ONE
     spelling (no second, drifting definition of "open-play cross").
   - `blocked` = per applicable row, `True` iff any uuid in `_related_events` resolves (via a
     `{event_id → (type_name, team_id)}` lookup built once from the full `events`) to a `Block` whose
     `team_id` **differs** from the cross's `team_id`.
   - **Dtype safety (ADR-019)**: the opposing-team comparison uses `id_compat.ids_differ` (numeric team
     ids, NA-safe). The uuid lookup is a plain dict on `event_id` (genuine string uuids — no `id_compat`,
     and no `.astype(str)` on any numeric id, per the ADR-019 dict-key trap).
   - Emit via `_blocked_flag(len(actions), applicable=<mask>, blocked=<mask>)` — identical machinery and
     alignment to `shot_blocked`.
3. **Synthetic rows**: the StatsBomb synthetic paths (`_add_dribbles`, the interception synthesis at
   `:350-364`) are non-cross → `applicable=False` → `pd.NA` automatically, exactly as `shot_blocked`
   already relies on. No special reset needed (verified against the GS pattern, which only resets because
   GS *synthesizes crosses/shots*; StatsBomb does not).

### 3.4 Verification (CI, offline, zero network)

- Extend `tests/spadl/test_statsbomb.py` (and/or `test_block_detection_contract.py`):
  - **Positive**: converting match `7584` yields `cross_blocked == True` on exactly the one action derived
    from cross `e8edd276…`; assert by locating it via `original_event_id`, not row position.
  - **Negative-in-same-match**: 7584's other open-play crosses are `False`; non-cross / set-piece-cross
    rows are `pd.NA`.
  - **No-false-positive**: matches `7298` and `3754058` yield all-`False` on open-play crosses,
    `pd.NA` elsewhere.
- Two existing sites encode the deferred all-`pd.NA` state and must flip to the shipped contract:
  `tests/spadl/test_statsbomb.py:413` (`assert actions["cross_blocked"].isna().all()`, inside
  `test_shot_blocked_true_on_real_blocked_shot` at `:401` — the exact positive-pattern to mirror, using
  the file's `load_statsbomb(7298)` helper) and `test_block_detection_contract.py`'s StatsBomb entry.
  The new positive fixture reuses that `load_statsbomb` helper for `7584`.
- These fixtures make the durable reproducible evidence **offline**; the wider probe is corroborating
  context, not a CI dependency.

### 3.5 Decision record

- **Amend ADR-046** with the probe corpus, the R1–R3 actuals, and the outcome (ship / keep-deferred).
- A short `docs/research/sb360_cross_blocked/README.md` (hand-written, numbers pasted from the stdout
  probe) records the measurement — the "measurement behind a claim" row of the CLAUDE.md history table.
  No committed machine artifact, no provenance guard.
- Update the CLAUDE.md ADR-046 block-detection bullet (StatsBomb `cross_blocked` no longer "deferred at
  n=1").

---

## 4. Part 2 — licensed `coverage.md` render

### 4.1 The script

`scripts/render_sb360_licensed_coverage.py` — reads the committed
`docs/research/sb360_licensed_coverage/coverage.parquet` + `manifest_all.json`, writes `coverage.md`
beside them. No provenance guard (reads a committed artifact; `render_sb360_matrix.py` precedent).
**Enrolled in the ADR-056 script-population registry** (`tests/scripts/_script_population.py`) as a
renderer in the `_NOT_A_DRIVER` bucket with a stated reason + the self-burning `is_file()` check, so the
population-completeness gate stays green. **Not golden-gated** (matches the renderer precedent; avoids
pandas-major golden fragility). Deterministic formatting (fixed decimals, sorted groupings) so output is
byte-stable across pandas 2/3 anyway; **produced/verified on `.venv312`** (the CI-repro env) since the
committed parquet is read the same way on both majors but grouping/formatting is exercised on the newer.

**Staleness guard (the one gap "not golden-gated" leaves).** Unlike the `render_sb360_matrix.py`
precedent — whose source registry changes in the same PR, so drift is caught in review — the licensed
`coverage.parquet` is refreshed **out-of-band** by an owner-run licensed driver. So the parquet can move
without a re-render and nothing would notice. The render already stamps the manifest's `generation` hash
into `coverage.md` (§4.2.1); a single cheap test (§6) asserts that stamped hash **equals** the committed
`manifest_all.json` `generation`. This catches a stale render **without pinning any value** — fully
consistent with not golden-gating the numbers.

### 4.2 Sections (all values computed from the parquet unless marked)

1. **Provenance** — from the manifest: generation hash, `n_attempted=30`, `run_commit=cf2f155`, tree
   clean, the producing driver name (`validate_sb360_licensed_corpus.py`). Provenance **by reference** to
   the committed parquet — the render stamps no SHA of its own.
2. **Frame-existence per GK-domain type** (`kind=frame_coverage`) — subject × `frame_existence_rate` ×
   denominator, the headline table mirroring the open `coverage.md`.
3. **Battery aggregator coverage** (`kind=battery_column`, `metric=non_nan_fraction`) — grouped into the
   **31 fully-NaN** columns (cross-linked to ADR-063: velocity-derived, Tier-2-suppressed, or
   constitutively-tracking) and the populated ones, **under a fixed ADR-042 caveat block**: these are
   structural "did the aggregator run + what fraction populated on real freeze-frames" facts, **not**
   tactical values — the per-column numbers are synthetic-input hybrids from `run_add_star_battery` and a
   `~0.5` half-pitch fraction is a coverage denominator, never a signal. The caveat is repeated **adjacent
   to the numbers**, not only as a one-time section header: every coverage-fraction value in the report (a
   battery non-NaN fraction, a pitch-coverage fraction) carries an inline interpretation label, so a reader
   who lands mid-doc on e.g. `add_visible_area_coverage: 0.5` reads "coverage denominator, not a signal"
   right there (the repo's UX standard — a displayed value must be interpretable in context).
4. **ADR-062 companion distributions** (`kind=companion_source` counts + `companion_fraction` means) —
   `observed / no_polygon / degenerate_polygon / degenerate_region / unlinked` per count feature
   (`nearest_defender_distance`, `receiver_zone_density`, `defenders_in_triangle_to_goal`).
5. **Pitch coverage** (`kind=pitch_coverage`, real `visible_area`) — mean 0.273, dispersion, min/max.
6. **Roster resolution** (`kind=roster`) — 100% on all 30.
7. **`battery_raises`** (`kind=battery_raises`) — the 23 `add_space_creation` single-team-frame raises,
   noted as expected (an honest refusal, not a defect).
8. **The 40→31 fully-NaN lift** — narrative citing the shipped ADR-063 velocity-less cycle; the parquet
   proves 31, the 40 is labelled prior-state context.
9. **Reading limits / reproducing** — licensed data is never committed; the render reads the committed
   parquet; re-run the producing driver (with the owner token) to refresh the parquet, then re-render.

---

## 5. Non-goals

- **Not** regenerating `coverage.parquet` (stays at `cf2f155`; no licensed-data access needed).
- **Not** touching set-piece cross blocking (`corner_crossed`/`freekick_crossed` stay `pd.NA`, ADR-046 v1).
- **Not** extending `cross_blocked` to any other provider (GS already real; Opta/others unchanged).
- **Not** a golden gate on the render, and **not** a committed machine artifact from the probe.
- **Not** an observed-region audit axis (owner-declined in 4.84.0) or any new freeze-frame feature.

---

## 6. Testing

- **Part 1** (offline, all matrix legs — behavioral contract, not version-sensitive): the §3.4 fixtures
  on committed matches `7584`/`7298`/`3754058`; the updated block-detection contract test.
- **Part 2**: the renderer's population-registry enrollment is exercised by the existing ADR-056 gate
  (`tests/scripts/test_*` over `_script_population.py`). A light smoke test that the render runs on the
  committed parquet and emits the expected top-level section headers (structural, not a golden of the
  numbers) — kept cheap and pandas-major-stable.
- **Part 2 staleness guard** (cheap, not a golden): a test asserting the `generation` hash stamped into
  the committed `coverage.md` equals the committed `manifest_all.json` `generation`. Catches a parquet
  refreshed out-of-band without a re-render; pins no value. (§4.1.)
- Full suite green (`python -m pytest tests/ -m "not e2e" -v --tb=short`) before proposing; lint at CI
  scope (`ruff check/format --check silly_kicks/ tests/ scripts/`), `pyright` bare.

---

## 7. Provenance, versioning & cross-session coordination

### 7.1 Provenance / commit strategy

- **No provenance-guarded artifact driver is added or run.** The probe prints to stdout; the render reads
  a committed parquet and is unguarded (`render_sb360_matrix.py` precedent); the parquet is not
  regenerated. Therefore no clean-tree gate fires, and the whole cycle is **one commit**.
- ADR-037/052/056 remain satisfied: the render **enrolls** in the population registry (as a non-driver),
  and no artifact carries a fabricated provenance stamp.

### 7.2 Versioning — assigned at completion, never up front

No version number is claimed in this spec. The version is determined **only when everything is implemented
and tested**, taken from `main` at that point (a number is not reserved by an in-flight spec). The
**release classification depends on Part 1's outcome**:

- **Part 1 ships** (real `cross_blocked`) → a minor library-behavior change + the coverage render → a
  normal minor release.
- **Part 1 keep-defers** → **no importable library code changes at all.** This is **not a release**: it is
  a plain `main` commit (Part 2's coverage render + the ADR-046 amendment + the research note) with **no
  version bump and no tag / PyPI publish**. The word "release" does not apply — only the doc + the ADR land.

### 7.3 Cross-session coordination (named, not resolved here — requester's call)

A parallel lakehouse-adoption spec is in flight and touches the same surface. Two interactions:

- **Version slot (not a real collision once no number is claimed).** Both cycles draw the next number from
  `main`; whichever completes and tests first takes it, the other rebases — numbers are not reserved. This
  spec claiming no number removes the "two specs silently claim 4.86.0" failure mode entirely; the
  requester still owns the ordering.
- **Content interaction (the substantive one).** If Part 1 ships, StatsBomb `cross_blocked` flips from
  all-`pd.NA` to real values. In **silly-kicks** this is a passthrough SPADL column consumed by no
  `vaep/`/`atomic/` feature (verified) → **no silly-kicks retrain**, ADR-045 `"invariant"`. But it is a
  **live-surface value change for downstream consumers** — the lakehouse adds `shot_blocked`/`cross_blocked`
  to its schemas assuming they are stable. Mitigations: an explicit **CHANGELOG Hyrum note** on the ship
  path; a **`docs/PRIVATE_CONSUMERS.md`** entry if the lakehouse pins the deferred all-`pd.NA` state; and a
  flag to the lakehouse session that whether it treats `cross_blocked` as a VAEP feature (retrain-relevant)
  or a passthrough is **its** call. silly-kicks un-deferring is a legitimate library decision and does not
  require downstream sign-off, but the interaction must be surfaced, not discovered.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Probe network / `statsbombpy` unavailable | The offline core (3 committed fixtures) already gives the decisive mechanism check (symmetric, unambiguous) and the CI fixture. The wider fetch is corroborating; if unreachable, the ADR records the offline evidence + the committed positive, and notes the wider probe as re-runnable when network is available. |
| Probe finds same-team / ambiguous contamination (R2/R3 fail) | Pre-registered → **keep deferred**, document the measured reason. The cycle still ships Part 2 and the ADR amendment. |
| "open-play cross" predicate drifts between the type dispatch and the block mask | Factor it into ONE local shared by both sites (§3.3.2). |
| `.astype(str)` on a numeric id in the uuid lookup (ADR-019 dict-key trap) | uuids are genuine strings; the lookup keys on `event_id` (never a numeric id); the only numeric-id compare (`team_id`) uses `id_compat.ids_differ`. |
| pandas-major formatting drift in the render | Deterministic fixed-decimal formatting + sorted groupings; produced/verified on `.venv312`; not golden-gated so no cross-leg flake. |
| Version-site miss | Bump all five sites incl. `uv.lock` per the release checklist. |

---

## 9. Deliverables checklist

- [ ] `scripts/probe_sb_cross_blocked.py` (ad-hoc, stdout, no artifact) + run → recorded numbers.
- [ ] **If ship**: `_flatten_extra` `_related_events` lift; shared open-play-cross predicate; `cross_blocked`
      mask at `statsbomb.py:282`; §3.4 fixtures; `test_block_detection_contract.py` update.
- [ ] **If keep-deferred**: no converter change; the measured reason in the ADR + research note.
- [ ] `docs/research/sb360_cross_blocked/README.md` (probe measurement + decision).
- [ ] ADR-046 amendment.
- [ ] `scripts/render_sb360_licensed_coverage.py` + population-registry enrollment + render smoke test.
- [ ] `docs/research/sb360_licensed_coverage/coverage.md` (rendered).
- [ ] Version bump **at completion** — number from `main`, all five sites incl. `uv.lock`, **only if Part 1
      ships a library change** (keep-defer → docs/tooling-only, §7.2); CHANGELOG entry (with the Part-1
      Hyrum note on the ship path); CLAUDE.md ADR-046 bullet; TODO.md; `docs/PRIVATE_CONSUMERS.md` if the
      lakehouse pins the deferred state (§7.3).
- [ ] Full suite + lint + pyright green.
