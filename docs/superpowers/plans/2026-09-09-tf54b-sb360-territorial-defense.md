# TF-54b (re-scoped) — SB360 Territorial-Defense Counterfactual — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a new tracking-consuming `silly_kicks.territorial_defense` package that measures how much a defender's *positioning* suppresses the attacking team's threat on SB360 freeze-frames, via a model-free removal (marginal-contribution) counterfactual, with two arms (action-anchored identity-exact + hull-based approximate) and an owner-run construct-validation battery.

**Architecture:** A hexagonal sibling of `gkdv/` and `restdefense/` — imports `silly_kicks.tracking` public seams only; nothing imports it. The counterfactual is `threat_suppressed = compute_threat_pc(counterfactual) − compute_threat_pc(actual)` where the counterfactual frame is the factual frame with one defender row **removed** (pitch control re-partitions the vacated space to the remaining players — Fernández–Bornn marginal player value). A new **actor identity bridge** (`is_actor` flag re-plumbed through the SB360 port) gives Arm A exact defender identity; Arm B reuses the v1 `territory` trimmed-hull membership test for broader, attribution-approximate coverage.

**Tech Stack:** Python, pandas, numpy; `silly_kicks.tracking` (pitch control, threat, goal-map, visibility, velocity), `silly_kicks.id_compat` (ADR-019), `scripts/_driver.py` (ADR-052 shards), `scripts/_provenance.py` (ADR-037). No new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-09-06-tf54b-sb360-ghosting-territorial-defense-design.md` (the plan argues from this spec; executors read both).

## Global Constraints

- **Owner rulings locked (this plan implements all five):** (1) mechanism = **removal/marginal**, replacement deferred; (2) package = new **`territorial_defense/`**; (3) event-only counterfactual = **not carried forward** (kept seams retained); (4) Arm-B rule = **nearest-to-target primary** + receiver-lane evaluated; (5) vehicle = **fresh commit off `main`, renumbered, close draft PR #235**.
- **Version / identifiers, re-confirm at commit-prep (never reserve):** next-free at time of writing = **4.112.0 / PR-S183 / ADR-090** (`main` is 4.111.0; PR-S182/ADR-089 are taken by the merged Part Deux). Do NOT hardcode a version anywhere until the release task; re-derive after `git fetch` at commit-prep.
- **Additive only — NO VAEP/tracking retrain, NO re-materialize, C4 count +1 (new container).** Every existing feature column stays byte-identical; the new package is in NO default xfn list; there is no `add_*` aggregator (the 33-count is unchanged — this is a `compute_*`).
- **`pitch_control_method="spearman"` is a HARD constraint** (GK-blind methods are unrepresentable — reject in `__post_init__`, mirroring `GkdvParams`).
- **Sign convention: attacker-value units, `positive = threat suppressed`** (`threat_pc(counterfactual) − threat_pc(actual)`). This is INVERTED from gkdv's "negative = deterrent"; the probe's direction registry must encode `positive`.
- **ADR discipline:** ADR-019 id-safety (`id_compat` everywhere, `canonical_id` for group/dict keys — never `astype(str)`); ADR-042 dropped-and-counted conservation (never a fabricated 0); ADR-027 honest-NaN never a sentinel; ADR-055 `resolve_defended_goals` built once + `GoalMap`, never team identity; ADR-063 velocity tiers (Tier-1 dimensionless threat is lifted at zero velocity, Tier-2 m²/s stays NaN); ADR-077 FOV honest-NaN; ADR-043 the `PitchControlCache` landmine (arms must never accept/share a cache).
- **HONEST LIMIT — validated as an INSTRUMENT, NOT as player-attributable (team-confound):** the §9 battery establishes that `a_threat_suppressed` responds to D's position, is specific, and tracks priors — it does NOT establish how much of the between-defender variance is the DEFENDER vs the DEFENSIVE SYSTEM. The removal counterfactual's "marginal contribution GIVEN TEAMMATES" (vacated space re-partitions to remaining players) is **team-conditioned by construction**. And the validation corpus **cannot identify the confound**: WC2022 is one-player-one-national-team → **zero cross-team defender observations**, so defender-vs-team is structurally unidentifiable and the elite-defender prior (Gvardiol/Otamendi/Van Dijk) is elite-defender/elite-team **collinear** — a clean prior is FACE-VALIDITY, not attribution evidence. **Consequence: per-defender numbers are NOT a defender ranking.** ("Arm A attribution is EXACT" means the FRAME is genuinely D's action — NOT that the VALUE is D-net-of-team.) Ranking is a future ADR-009 gated on a crossed defender+team variance decomposition (ICC) over a MULTI-CLUB TRANSFER corpus, never this one. Precedent: the sister GK-distribution metric (eyestone collaboration) passed its own face-validity checks then proved ~80% team-confounded — "ranking not licensed." This limit must be stated in the CLAUDE.md contract, the research report, and any per-defender output docstring.
- **Lint at CI scope:** `python -m ruff check silly_kicks/ tests/ scripts/`, `python -m ruff format --check silly_kicks/ tests/ scripts/`, `python -m pyright` (bare). Test: `python -m pytest tests/ -m "not e2e" -v --tb=short`. ASCII-only in `scripts/`.
- **Never commit/push without explicit owner approval.** The final task STOPS at the diff for owner review.

## Plan revision log

- **rev-2 (2026-09-09, post `/review-plan`):** closed **PLAN-01** (SHOULD FIX) — the removal counterfactual on a defender-depleted frame is now dropped-and-counted as `removal_undersupported` (Task 5 param `min_defenders_after_removal`, Task 6 guard + test); verified against `_spearman.py:200/217/226` (0-defender surface does not raise, degrades to attacker-controls-all → an upward-biased outlier orthogonal to the SPEC-02 area gate). Applied nits **PLAN-04** (membership `_compute.py:158-159`), **PLAN-06** (deleted the "actual − cf" wrong fragment in Task 7), **PLAN-07** (cleaned the import-allowlist example), **PLAN-08** (SPEC-02 gate split into its own RED step). **Declined PLAN-05** — `NOTICE` is ADR-005 (CLAUDE.md line 132), `feature_glossary` is ADR-048 (line 178); Task 11 already cites both correctly. **PLAN-02 / PLAN-03 / Task 4 column choice / Task 8 hull export** remain owner-ruled (see "Open items flagged for the owner").
- **rev-3 (2026-09-09, cross-project context from the eyestone / xT-GK collaboration):** **Item A** — added the load-bearing honest-limit that the metric is validated as an INSTRUMENT, **NOT as player-attributable**: the removal counterfactual's "marginal contribution given teammates" is team-conditioned by construction, and this corpus (WC2022 + 30 licensed SB360, national-team / single-tournament) **structurally cannot identify the defender-vs-team confound** (one player = one national team → zero cross-team defender observations), so per-defender numbers are NOT a ranking and the elite-defender prior is elite-defender/elite-team **collinear** (Global Constraints; Task 12 Steps 3–4; Task 13 CLAUDE.md bullet). Precedent: the sister GK metric passed face-validity then proved ~80% team-confounded ("ranking not licensed"). **Item B** — hoisted `apply_actor_identities_to_frames` OUT of the new package (where Task 5's gate forbids any importer) into the shared `silly_kicks/keeper_identity.py`, next to `apply_keeper_identities_to_frames` (ADR-084 home), so §4's "independently useful" claim is honest and the concrete second consumer (eyestone GK build-up-decision metric) can reuse it (File Structure; Tasks 4, 5; new flagged item #5 — placement is owner-ruled). **Item C** — §2 SB360 anonymity design independently corroborated (no action).
- **rev-4 (2026-09-09, owner rulings on all five flagged sub-decisions):** (1) Task 3 **REMOVE** the reserved `territory` counterfactual door; (2) Task 4 `is_actor` is a **snapshot-only** extension column; (3) Task 9 **omit** `ghost_model` in v1; (4) Task 8 **export** `build_trimmed_hull`/`Hull` from `territory.__init__` (single-source the hull; new Step 1a); (5) actor bridge home = **`silly_kicks/keeper_identity.py`**. No open items remain; the "Owner rulings" block replaced the "Open items flagged" block.
- **rev-5 (2026-09-09, post re-review R2 — APPROVE):** applied the 3 non-blocking CONSIDERs. **PLAN-10** — wired the not-a-ranking honest-limit into the two consumer-facing places: the `compute_territorial_defense` docstring (Task 9) and each glossary `definition` (Task 11). **PLAN-11** — the whole-tree import sweep example now catches RELATIVE imports too (`n.level>0` / `from . import territorial_defense`), not just absolute. **PLAN-12** — harmonized the keeper-bridge ADR cite: docstring now reads "ADR-078 bridge pattern; ADR-084 module promotion" (spec §4's "ADR-084 home" is the module-placement fact — consistent). R2 confirmed PLAN-01 crux and retracted its own R1 PLAN-05 (NOTICE→ADR-005 decline was correct).

---

## Starting-state facts (verified against `main` = `fa9a9a1` and `ab9001c` on 2026-09-09)

These are the ground truth every task assumes:

- `main` has **none** of the event-only counterfactual. Absent on main: `silly_kicks/expected_passing/`, `silly_kicks/territory/_counterfactual.py`, `silly_kicks/xthreat/_counterfactual_seam.py`, `scripts/train_pass_completion.py`, `scripts/validate_territory_counterfactual.py`, `scripts/_synthetic_interception.py`, `scripts/_sb_open_data.py`.
- `main`'s `territory/_columns.py`: `TERRITORY_METHODS = frozenset({"completed_failed", "counterfactual"})` where `counterfactual` is a **reserved typed door** — `compute_territorial_dominance(method="counterfactual")` raises `NotImplementedError` at `_compute.py:89`. `main`'s `_config.py` has only `TerritoryParams` (no `CounterfactualParams`). `main`'s `territory/__init__.py` exports `TERRITORY_METHODS` but not `columns_for_method`/`CounterfactualParams`.
- The **kept seams** (Decision 3) exist ONLY in the unmerged `ab9001c` and must be carried forward from it: `silly_kicks/expected_passing/{__init__,_features,_model}.py`, `silly_kicks/xthreat/_counterfactual_seam.py` + the `xthreat/__init__.py` export + `silly_kicks/xthreat/_transitions.py` change, `scripts/train_pass_completion.py`, `scripts/_sb_open_data.py`, and their tests (`tests/expected_passing/*`, `tests/xthreat/test_counterfactual_seam.py` + `tests/xthreat/__init__.py`, `tests/scripts/test_sb_open_data.py`), plus the `expected_passing`-relevant lines of `pyproject.toml` and `tests/test_public_api_examples.py`.
- The **not-carried** event-only cone: `territory/_counterfactual.py`, the `territory/{_compute,_columns,_config,_report,__init__}.py` cone modifications, `scripts/_synthetic_interception.py`, `scripts/validate_territory_counterfactual.py`, the cone tests, and the 4 cone feature-glossary columns. (These never existed on `main`, so "removal" = simply not carrying them.)
- `PassCompletionModel.bundled()` raises `FileNotFoundError` until weights are trained+committed (an owner-run step, Phase 5).

### Exact seam signatures (verified — quote these; do not re-derive)

```python
# providers/statsbomb/parse.py:218
def shape_snapshots(frames_raw, actions, *, fidelity_version=1) -> tuple[pd.DataFrame, pd.DataFrame, JoinReport]
#   snap_rows dict built at :305-313 (reads teammate/keeper; NOT actor); row-id np.arange at :324-327

# tracking/_snapshot.py:24
def snapshot_to_tracking_frames(snapshots, actions) -> tuple[pd.DataFrame, pd.DataFrame]
#   player_frames dict :104-145 ; ball_frames dict :148-177 ; schema-select :180 ; _cast_to_declared_schema :203-230

# keeper_identity.py:635  (the bridge PATTERN to mirror)
def apply_keeper_identities_to_frames(frames, keeper_map) -> pd.DataFrame

# tracking/_cover_shadows.py:826  (exported from tracking)
def compute_threat_pc(frame, *, attacking_team_id, xt, goal_map,
                      method="spearman", params=None, field_weight=None) -> float
#   first statement: require_fitted_xt(xt, caller="compute_threat_pc"); raises on unfitted/None

# tracking/_gk_resolve.py:602
def resolve_defended_goals(frames) -> GoalMap        # GoalMap.get(...) / GoalMap.attacked_goal(...)

# tracking/_visibility.py:121
def region_observed_fraction(polygon, region) -> float   # region MUST be convex; nan if polygon absent/degenerate

# tracking/_velocity_availability.py:63
def zero_velocity_if_unavailable(frames, *, method="spearman") -> pd.DataFrame

# tracking/pitch_control/_surface.py:133
PitchControlSurface.control_in_region(x_min, x_max, y_min, y_max) -> float

# territory/_hull.py:34
def build_trimmed_hull(defensive_actions_xy, *, trim_fraction) -> Hull | None   # Hull.contains(points) -> np.ndarray[bool]
#   ADR-028 membership (territory/_compute.py:158-159): refl = column_stack([fl - opp["end_x"], fw - opp["end_y"]]); hull.contains(refl)
#   fl = spadlconfig.field_length (105.0), fw = spadlconfig.field_width (68.0)

# id_compat.py
canonical_id(x); canonical_id_series(s); ids_equal(a,b); ids_match(series,scalar); same_id(a,b); align_join_keys(left,right,keys)

# scripts/_driver.py:529
def for_each(items, *, key, work, shard_root, token_inputs, token_reason=None,
             counters=None, tag="all", label="item", max_consecutive_failures=3) -> CorpusPassResult
# scripts/_provenance.py: git_provenance() -> dict ; require_clean_tree(prov, *, allow_dirty) -> dict
# scripts/_input_contract.py: declare_inputs(**parts) -> dict   # must pass driver="<script stem>"
```

SPADL action `type_id`s for the Arm-A domain (socceraction standard, **verify against `spadl.config.actiontypes_df()` in Task 8**): tackle=9, interception=10, clearance=18.

---

## File Structure

**New library package** `silly_kicks/territorial_defense/` (mirrors `restdefense/`):
- `__init__.py` — public surface (`compute_territorial_defense`, `TerritorialDefenseParams`, `TerritorialDefenseReport`, `TD_SAMPLE_COLUMNS`, `TD_SOURCE_VALUES`). (The actor bridge is NOT here — it lives in the shared `keeper_identity.py`, Item B / rev-3.)
- `_config.py` — `TerritorialDefenseParams` (frozen; `for_provider` empty per ADR-009; `__post_init__` rejects non-spearman).
- `_report.py` — `TerritorialDefenseReport` (conservation dataclass).
- `_columns.py` — sample column names + `TD_SOURCE_VALUES` provenance vocab.
- `_engine.py` — domain filter, removal counterfactual builder, provenance + report, the SPEC-02 local-completeness FOV gate.
- `_arms.py` — Arm A (`arm_a_threat_suppressed[_batch]`), Arm B (`arm_b_threat_suppressed[_batch]`), `_assert_legs_aligned`; **no `pitch_control_cache` parameter anywhere**.
- `_compute.py` — `compute_territorial_defense` orchestrator.
- `_probe.py` — dose imposer, layer-0/1 verdicts (own `TD_PROBE_RATIO`), paired controls, arm→direction registry. Self-contained (adapts the gkdv pattern; does not import gkdv).

**Modified library files:**
- `providers/statsbomb/parse.py` — `shape_snapshots`: add `"is_actor"` to `snap_rows` + columns.
- `tracking/_snapshot.py` — carry `is_actor` through `player_frames`/`ball_frames` and the schema-select/cast.
- `silly_kicks/keeper_identity.py` — add `apply_actor_identities_to_frames` (shared home, next to `apply_keeper_identities_to_frames`; Item B / rev-3). **DECIDED (owner ruled 2026-09-09).**
- `silly_kicks/territory/{_columns,_compute,__init__}.py` — honor Decision 3: drop the reserved `counterfactual` door (flagged sub-decision, Task 3).
- `silly_kicks/feature_glossary.py`, `NOTICE`, `docs/c4/architecture.dsl` (+ re-render `.html`) — governance registrations for the new package.
- `tests/sb360/_registry.py` — SB360 boundary verdicts for the arms.

**Carried-forward-from-`ab9001c` (Task 1, unchanged content):** `silly_kicks/expected_passing/`, `silly_kicks/xthreat/_counterfactual_seam.py` (+ `__init__`/`_transitions` deltas), `scripts/{train_pass_completion,_sb_open_data}.py`, their tests.

**New scripts:** `scripts/validate_territorial_defense.py` (owner-run validation driver).

**New tests:** `tests/territorial_defense/` (mirrors `tests/restdefense/`, `tests/gkdv/`).

---

## Task 1: Fresh branch off `main` + carry forward the kept seams

**Files:**
- Git branch (new): `feat/tf54b-sb360-territorial-defense` off `main` (`fa9a9a1`).
- Carry from `ab9001c`: `silly_kicks/expected_passing/`, `silly_kicks/xthreat/_counterfactual_seam.py`, `silly_kicks/xthreat/__init__.py` (export lines only), `silly_kicks/xthreat/_transitions.py`, `scripts/train_pass_completion.py`, `scripts/_sb_open_data.py`, `tests/expected_passing/`, `tests/xthreat/__init__.py`, `tests/xthreat/test_counterfactual_seam.py`, `tests/scripts/test_sb_open_data.py`.
- Verify byte-identical to main: `silly_kicks/territory/` (all files), everything else.

**Interfaces:**
- Produces: a clean branch = `main` + `expected_passing.PassCompletionModel`, `expected_passing.pass_completion_features`, `xthreat.destination_profiles`, `scripts.train_pass_completion`, `scripts._sb_open_data` — nothing of the cone counterfactual.

- [ ] **Step 1: Create the branch off updated main**

```bash
git fetch origin main
git switch main && git merge --ff-only origin/main   # local main == fa9a9a1
git switch -c feat/tf54b-sb360-territorial-defense
```
(The untracked spec + this plan file travel with the working tree; they are not branch-bound.)

- [ ] **Step 2: Carry ONLY the kept library seams from `ab9001c`**

```bash
git checkout ab9001c -- silly_kicks/expected_passing/ \
  silly_kicks/xthreat/_counterfactual_seam.py \
  silly_kicks/xthreat/_transitions.py \
  scripts/train_pass_completion.py scripts/_sb_open_data.py \
  tests/expected_passing/ tests/xthreat/__init__.py \
  tests/xthreat/test_counterfactual_seam.py tests/scripts/test_sb_open_data.py
```

- [ ] **Step 3: Hand-merge the `xthreat/__init__.py` export (do NOT `git checkout` the whole file — it carries cone-unrelated deltas only in the 3 export lines)**

Open `ab9001c:silly_kicks/xthreat/__init__.py`, copy ONLY the `destination_profiles` / `DestinationProfile` import + `__all__` additions into the current `xthreat/__init__.py`. Diff to confirm no other line moved:
```bash
git diff main -- silly_kicks/xthreat/__init__.py   # expect: only destination_profiles/DestinationProfile lines added
```

- [ ] **Step 4: Verify `territory/` is byte-identical to main (the cone was NOT carried)**

```bash
git diff main --stat -- silly_kicks/territory/   # expect: EMPTY
test -f silly_kicks/territory/_counterfactual.py && echo "FAIL: cone present" || echo "OK: no cone"
```
Expected: empty diff; "OK: no cone".

- [ ] **Step 5: Handle `expected_passing` packaging + public-example lines**

Bring the `expected_passing`-relevant `pyproject.toml` addition (package inclusion / weights data) and the `expected_passing`/`destination_profiles` entries in `tests/test_public_api_examples.py` from `ab9001c` by hand (diff-guided — do NOT carry any `territory._counterfactual` example):
```bash
git show ab9001c -- pyproject.toml
git show ab9001c -- tests/test_public_api_examples.py
```
Apply only the `expected_passing` + `xthreat.destination_profiles` lines.

- [ ] **Step 6: Run the carried tests + territory regression**

Run: `python -m pytest tests/expected_passing/ tests/xthreat/test_counterfactual_seam.py tests/territory/ tests/scripts/test_sb_open_data.py -m "not e2e" -q`
Expected: PASS (carried seams work; territory unchanged from main). If `PassCompletionModel.bundled()` tests fail with `FileNotFoundError`, that is EXPECTED (weights come in Phase 5) — confirm those specific tests already `pytest.importorskip`/`xfail` on missing weights in the carried `ab9001c` versions; if not, mark them `@pytest.mark.skip(reason="weights bundled in Phase 5")` and record it as a Phase-5 un-skip obligation.

- [ ] **Step 7: Confirm clean baseline, no commit**

Run: `git status` (expect: carried files staged/tracked, untracked spec+plan). Do NOT commit yet — Task 1 is a working-tree state, not a release.

---

## Task 2: `is_actor` through the SB360 port (`shape_snapshots`)

**Files:**
- Modify: `silly_kicks/providers/statsbomb/parse.py:305-324` (`shape_snapshots`)
- Test: `tests/providers/statsbomb/test_shape_snapshots_actor.py` (new)

**Interfaces:**
- Consumes: raw SB360 freeze-frame rows (each carries a `actor` boolean, currently dropped).
- Produces: `snapshots` DataFrame gains an `is_actor: bool` column — exactly one `True` per `action_id`, on the acting team.

- [ ] **Step 1: Write the failing test**

```python
# tests/providers/statsbomb/test_shape_snapshots_actor.py
import pandas as pd
from silly_kicks.providers.statsbomb.parse import shape_snapshots

def _raw_frame(action_id, actor_idx):
    players = [
        {"location": [60.0, 40.0], "teammate": True,  "keeper": False, "actor": i == actor_idx}
        for i in range(3)
    ] + [{"location": [30.0, 20.0], "teammate": False, "keeper": True, "actor": False}]
    return {"id": action_id, "freeze_frame": players}

def test_shape_snapshots_carries_is_actor_exactly_one_true_per_action():
    actions = pd.DataFrame({"action_id": [7], "original_event_id": ["e7"], "team_id": [100]})
    snaps, _va, _rep = shape_snapshots([_raw_frame("e7", actor_idx=1)], actions)
    assert "is_actor" in snaps.columns
    assert snaps["is_actor"].dtype == bool
    actor_rows = snaps[snaps["is_actor"]]
    assert len(actor_rows) == 1
    # the actor is on the ACTING team (teammate=True -> acting_team)
    assert actor_rows.iloc[0]["team_id"] == 100
```
(Adjust the `actions` columns to the real join contract — read `shape_snapshots`'s `action_meta` merge to match `action_id`/`original_event_id`/`team_id` exactly before finalizing the fixture.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/providers/statsbomb/test_shape_snapshots_actor.py -v`
Expected: FAIL — `KeyError: 'is_actor'` / column absent.

- [ ] **Step 3: Add `is_actor` to the snap_rows dict + columns list**

In `shape_snapshots`, the per-player append (`parse.py:305-313`) gains one key:
```python
        snap_rows.append(
            {
                "action_id": action_id,
                "team_id": acting_team if bool(row.get("teammate")) else opponent_team,
                "is_goalkeeper": bool(row.get("keeper")),
                "is_actor": bool(row.get("actor")),   # NEW — SB360 marks exactly one actor/frame
                "x": float(x),
                "y": float(y),
            }
        )
```
And the DataFrame column list (`parse.py:324`):
```python
    snapshots = pd.DataFrame(
        snap_rows, columns=["action_id", "team_id", "is_goalkeeper", "is_actor", "x", "y"]
    )
    snapshots["player_id"] = np.arange(len(snapshots))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/providers/statsbomb/test_shape_snapshots_actor.py -v`
Expected: PASS.

- [ ] **Step 5: Guard the empty-frame + missing-`actor`-key path**

Add a test that a freeze-frame row with no `actor` key yields `is_actor=False` (never raises — `row.get("actor")` → `None` → `bool(None)` is `False`), and that a zero-player frame produces zero rows without error. Run the file; expect PASS. Commit boundary: fold into Task 4's commit (the bridge needs all three layers to be testable end-to-end).

---

## Task 3: Honor Decision 3 — drop the `territory` reserved counterfactual door

> **DECIDED (owner ruled 2026-09-09): REMOVE the door.** Context: Decision 3 / spec §11.3 targeted the event-only **cone implementation**, which is **absent on `main`** (it lived only in `ab9001c`, not carried) — already gone by not carrying it. What remains on `main` is a *reserved* `counterfactual` door that only raises `NotImplementedError`; retiring it (motivated by Decision 2 — counterfactual now lives in a separate package) makes `territory` read as purely `completed_failed`. **This shrinks a PUBLIC `frozenset` (`TERRITORY_METHODS`) — a Hyrum surface** (a consumer checking `"counterfactual" in TERRITORY_METHODS` sees a change), so the release note (Task 13) must call it out.

**Files:**
- Modify: `silly_kicks/territory/_columns.py:15` (`TERRITORY_METHODS`), `silly_kicks/territory/_compute.py:87-90` (drop the `NotImplementedError` branch; keep the `unknown method` `ValueError`)
- Test: `tests/territory/test_columns.py`, `tests/territory/test_compute.py`

- [ ] **Step 1: Write/adjust the failing test**

```python
# tests/territory/test_columns.py  (adjust existing test_method_family_and_source_vocab)
def test_method_family_is_completed_failed_only():
    from silly_kicks.territory import TERRITORY_METHODS
    assert TERRITORY_METHODS == frozenset({"completed_failed"})
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/territory/test_columns.py -v`
Expected: FAIL — main still has `{"completed_failed", "counterfactual"}`.

- [ ] **Step 3: Shrink the frozenset + remove the reserved-door raise**

`_columns.py:15`: `TERRITORY_METHODS = frozenset({"completed_failed"})` (and update the `:13-14` comment). `_compute.py`: delete the `if method == "counterfactual": raise NotImplementedError(...)` block (`:89-90`); keep `if method not in TERRITORY_METHODS: raise ValueError(...)` (`:87-88`).

- [ ] **Step 4: Run territory suite — confirm `completed_failed` byte-identical**

Run: `python -m pytest tests/territory/ -m "not e2e" -v`
Expected: PASS, including the `completed_failed_v1.parquet` golden (the default path is untouched). If `tests/territory/test_compute.py::test_method_family` asserts the counterfactual door raises, delete that assertion (the door is gone).

- [ ] **Step 5: Commit boundary** — fold into the Phase-4 governance commit (a public frozenset change belongs with the CLAUDE.md/CHANGELOG note that documents it).

---

## Task 4: `is_actor` through `snapshot_to_tracking_frames` + the actor identity bridge

**Files:**
- Modify: `silly_kicks/tracking/_snapshot.py:104-182` (`snapshot_to_tracking_frames`)
- Modify: `silly_kicks/keeper_identity.py` — add `apply_actor_identities_to_frames` (shared home, next to `apply_keeper_identities_to_frames`; Item B / rev-3)
- Test: `tests/tracking/test_snapshot_is_actor.py`, `tests/keeper_identity/test_actor_bridge.py` (place beside the existing keeper-identity tests)

**Interfaces:**
- Consumes: `shape_snapshots` output (Task 2, `snapshots` has `is_actor`); `snapshot_to_tracking_frames(snapshots, actions) -> (frames, links)`.
- Produces:
  - `frames` gains an `is_actor: boolean` column (ball rows `False`; exactly one `True` per `frame_id`/acting-team). **Design decision:** `is_actor` is a **snapshot-only extension column**, NOT added to base `TRACKING_FRAMES_COLUMNS` — continuous tracking has no per-frame "actor," so widening the base schema would force every native builder + schema/liveness/mirror gate to emit an all-`False` column (broad blast radius, no retrain justification). It is appended after the schema-select and cast to `"boolean"` explicitly. (**DECIDED, owner ruled 2026-09-09**; the base-schema-column alternative was rejected on blast-radius + semantics.)
  - `apply_actor_identities_to_frames(frames: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame` — **exported from `silly_kicks.keeper_identity`** (Item B: a shared home so a second consumer — the eyestone GK build-up-decision metric — can reuse it without violating the new package's import-allowlist; the keeper analogue lives here too). Pure; stamps each action's real `player_id` onto its single `is_actor` frame row (joined `frame_id == action_id`), ADR-019 id-safe; non-actor rows keep their synthetic ids; returns a NEW frame. `keeper_identity` stays tracking-free (imports `id_compat` + pandas/numpy only — the bridge needs no tracking symbol).

- [ ] **Step 1: Write the failing test for the frame column**

```python
# tests/tracking/test_snapshot_is_actor.py
import pandas as pd
from silly_kicks.tracking import snapshot_to_tracking_frames

def _snaps():
    return pd.DataFrame({
        "action_id": [1, 1, 1, 1],
        "team_id":   [100, 100, 100, 200],
        "is_goalkeeper": [False, False, False, True],
        "is_actor":  [False, True, False, False],
        "x": [60.0, 55.0, 50.0, 8.0],
        "y": [40.0, 34.0, 30.0, 34.0],
        "player_id": [0, 1, 2, 3],
    })

def _actions():
    return pd.DataFrame({"action_id":[1],"game_id":[9],"period_id":[1],"time_seconds":[12.0]})

def test_frames_carry_is_actor_ball_false_one_true():
    frames, _links = snapshot_to_tracking_frames(_snaps(), _actions())
    assert "is_actor" in frames.columns
    assert not frames.loc[frames["is_ball"].astype("boolean").fillna(False), "is_actor"].astype("boolean").fillna(False).any()
    assert int(frames["is_actor"].astype("boolean").fillna(False).sum()) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_snapshot_is_actor.py -v`
Expected: FAIL — `is_actor` absent from frames.

- [ ] **Step 3: Thread `is_actor` through both frame dicts + the schema-select**

In `snapshot_to_tracking_frames`, add to `player_frames` (`:116-145`): `"is_actor": player["is_actor"] if "is_actor" in player else False,`. Add to `ball_frames` (`:148-177`): `"is_actor": False,`. After the schema-select/cast (`:180-182`), append + cast:
```python
    frames = frames[list(TRACKING_FRAMES_COLUMNS.keys())]
    frames = _cast_to_declared_schema(frames)
    if "is_actor" in player_frames.columns:                 # snapshot-only extension column
        frames["is_actor"] = pd.concat(
            [player_frames["is_actor"], ball_frames["is_actor"]], ignore_index=True
        ).astype("boolean").reset_index(drop=True)
```
(Match the exact concat/reset order the function already uses when it builds `frames`; if it re-sorts, index `is_actor` by the same key. Read `:178-195` and mirror the ordering rather than assuming `ignore_index`.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_snapshot_is_actor.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test for the actor bridge**

```python
# tests/keeper_identity/test_actor_bridge.py
import pandas as pd
from silly_kicks.keeper_identity import apply_actor_identities_to_frames

def test_actor_row_gets_real_player_id_others_unchanged():
    frames = pd.DataFrame({
        "game_id":[9,9,9], "period_id":[1,1,1], "frame_id":[1,1,1],
        "team_id":[100,100,100], "player_id":[0,1,2],
        "is_ball":[False,False,False], "is_actor":[False,True,False], "x":[1.,2.,3.], "y":[1.,2.,3.],
    })
    actions = pd.DataFrame({"action_id":[1], "player_id":[5551], "team_id":[100]})
    out = apply_actor_identities_to_frames(frames, actions)
    assert out is not frames                                  # pure
    assert frames["player_id"].tolist() == [0,1,2]            # input not mutated
    assert out.loc[out["is_actor"].astype("boolean").fillna(False), "player_id"].iloc[0] == 5551
    assert out.loc[~out["is_actor"].astype("boolean").fillna(False), "player_id"].tolist() == [0,2]
```

- [ ] **Step 6: Run to verify it fails, then implement the bridge**

Run: `python -m pytest tests/keeper_identity/test_actor_bridge.py -v` → FAIL (function missing).

Implement `apply_actor_identities_to_frames` **in `silly_kicks/keeper_identity.py`** (next to `apply_keeper_identities_to_frames`, `:635-713`, whose shape it mirrors) but keyed per-ACTION:
```python
from __future__ import annotations
import numpy as np
import pandas as pd
from silly_kicks.id_compat import canonical_id

def apply_actor_identities_to_frames(frames: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Stamp each action's real ``player_id`` onto its single ``is_actor`` frame row.

    The outfield analogue of ``apply_keeper_identities_to_frames`` (ADR-078 bridge pattern;
    lives in this shared ``keeper_identity`` module per the ADR-084 promotion), scoped to the
    one row SB360 reliably identifies. PURE; ADR-019 id-safe; non-actor rows keep their
    synthetic ids. Keyed on ``frame_id == action_id`` (the snapshot port sets frame_id=action_id).
    """
    out = frames.copy()
    is_actor = out["is_actor"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    if not is_actor.any():
        return out
    by_action = {canonical_id(a): pid for a, pid in zip(actions["action_id"], actions["player_id"], strict=True)}
    new_pid = out["player_id"].copy()
    frame_ids = out["frame_id"].to_numpy()
    for i in np.flatnonzero(is_actor):
        pos = int(i)
        pid = by_action.get(canonical_id(frame_ids[pos]))
        if pid is None or pd.isna(pid):
            continue
        try:
            new_pid.iat[pos] = pid
        except (TypeError, ValueError):
            new_pid = new_pid.astype("object")
            new_pid.iat[pos] = pid
    out["player_id"] = new_pid
    return out
```
Add `apply_actor_identities_to_frames` to `keeper_identity.py`'s `__all__`. (Verify `keeper_identity` still imports no `tracking` symbol after this — the bridge is pure over a `frames` DataFrame + `actions`, so it needs `id_compat` + pandas/numpy only. `tests/providers/test_appearances_import_allowlist.py`-style discipline: `keeper_identity` must stay tracking-free at module import.)

- [ ] **Step 7: Run both tests + a purity check**

Run: `python -m pytest tests/tracking/test_snapshot_is_actor.py tests/keeper_identity/test_actor_bridge.py -v`
Expected: PASS. Add an assertion that a mixed-dtype `player_id` (int frames, string action id) still stamps via the `object` fallback (ADR-019).

- [ ] **Step 8: Commit boundary** — this is the first coherent deliverable (the actor bridge, independently useful). Stage the port + snapshot + bridge + tests. Do NOT commit (owner-approval gate); note it as commit-unit boundary #1 for the eventual single commit.

---

## Task 5: Package skeleton — params, report, columns, import gates

**Files:**
- Create: `silly_kicks/territorial_defense/{__init__,_config,_report,_columns}.py`
- Test: `tests/territorial_defense/{test_import_allowlist,test_config,test_report}.py`

**Interfaces:**
- Produces:
  - `TerritorialDefenseParams` (frozen): `defensive_action_type_ids: tuple[int,...] = (9, 10, 18)`, `pitch_control_method: str = "spearman"`, `lambda_gk: float = 3.0`, `trim_fraction: float = 0.70`, `min_local_observed_fraction: float = 0.7`, `local_radius_m: float = 10.0`, `min_defenders_after_removal: int = 1`, `arm_b_rule: str = "nearest_to_target"`; `for_provider(provider)` empty (ADR-009); `__post_init__` rejects non-`spearman` method AND an unknown `arm_b_rule`.
  - `TerritorialDefenseReport(params, n_frames_in, n_frames_scored, drop_reasons)` (frozen; conservation `n_frames_scored + Σ drop_reasons == n_frames_in`).
  - `TD_SOURCE_VALUES = frozenset({"scored", "no_actor", "no_defenders", "removal_undersupported", "unresolved_geometry", "fov_cropped", "fov_cropped_local", "velocity_unscoreable", "degenerate_hull", "not_in_domain"})`.
  - `TD_SAMPLE_COLUMNS` (per `(game_id, player_id)`): `a_threat_suppressed, a_frames_scored, b_threat_suppressed, b_frames_scored, b_attribution_slippage, td_source`.

- [ ] **Step 1: Write the failing import-allowlist tests (RED-first, per ADR-051)**

Mirror `tests/restdefense/test_import_allowlist.py` (6 functions), then add the **fresh whole-tree sweep** the spec flags as absent from the restdefense exemplar:
```python
# tests/territorial_defense/test_import_allowlist.py  (key additions beyond the restdefense mirror)
import ast, pathlib
ROOT = pathlib.Path("silly_kicks")
PKG = "silly_kicks.territorial_defense"

_PKG_TAIL = "territorial_defense"   # unique package basename

def _imports_pkg(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            # absolute `from silly_kicks.territorial_defense import x`
            # AND relative `from ..territorial_defense import x` (n.level>0, module="territorial_defense")
            if _PKG_TAIL in (n.module or "").split("."):
                return True
            # relative `from . import territorial_defense`
            if n.level > 0 and any(a.name == _PKG_TAIL for a in n.names):
                return True
        if isinstance(n, ast.Import) and any(_PKG_TAIL in a.name.split(".") for a in n.names):
            return True
    return False

def test_nothing_in_silly_kicks_imports_territorial_defense():
    """WHOLE-TREE sweep (authored fresh — restdefense's reverse test is tracking-scoped only)."""
    offenders = [
        str(p) for p in ROOT.rglob("*.py")
        if "territorial_defense" not in p.parts and _imports_pkg(p)
    ]
    assert offenders == [], f"nothing may import {PKG}; offenders: {offenders}"
```
Plus: `test_territorial_defense_imports_only_public_seams` (forward allowlist — the package may import public `silly_kicks.tracking`, `silly_kicks.keeper_identity` (the actor bridge, Item B), and `silly_kicks.territory` (the hull, Task 8) seams; any private `silly_kicks.{tracking,keeper_identity,territory}._*` import fails unless in an explicit empty allowlist set), `test_tracking_never_imports_territorial_defense`, a planted-violation meta-test for each detector, and `test_package_is_non_empty`.

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/territorial_defense/test_import_allowlist.py -v`
Expected: FAIL (package does not exist).

- [ ] **Step 3: Create the skeleton modules**

`_config.py` (mirror `restdefense/_config.py:18-100`; `__post_init__` mirrors `GkdvParams._engine.py:104-125`):
```python
@dataclass(frozen=True)
class TerritorialDefenseParams:
    defensive_action_type_ids: tuple[int, ...] = (9, 10, 18)   # tackle, interception, clearance
    pitch_control_method: str = "spearman"
    lambda_gk: float = 3.0
    trim_fraction: float = 0.70
    min_local_observed_fraction: float = 0.7
    local_radius_m: float = 10.0
    min_defenders_after_removal: int = 1   # PLAN-01: removing D must leave >=1 defender to re-absorb space
    arm_b_rule: str = "nearest_to_target"
    _is_universal_default: bool = field(default=False, compare=False, repr=False)
    _GK_AWARE_METHODS = ("spearman",)
    _ARM_B_RULES = ("nearest_to_target",)   # "receiver_lane" added when validated (Phase 5)

    def __post_init__(self) -> None:
        if self.pitch_control_method not in self._GK_AWARE_METHODS:
            raise ValueError(f"pitch_control_method={self.pitch_control_method!r} is GK-blind; allowed {self._GK_AWARE_METHODS}")
        if self.arm_b_rule not in self._ARM_B_RULES:
            raise ValueError(f"arm_b_rule={self.arm_b_rule!r} not in {self._ARM_B_RULES}")

    @classmethod
    def for_provider(cls, provider: str) -> "TerritorialDefenseParams":
        return cls()   # ADR-009: no per-provider tuning

_DEFAULT_PARAMS = TerritorialDefenseParams()
```
`_report.py` mirrors `restdefense/_report.py:16-33`. `_columns.py` defines `TD_SAMPLE_COLUMNS` + `TD_SOURCE_VALUES`. `__init__.py` exports the public surface (`compute_territorial_defense`, params, report, columns, source vocab) — **NOT** `apply_actor_identities_to_frames`, which is exported from `keeper_identity` (Item B / rev-3).

- [ ] **Step 4: Run to verify the gates pass**

Run: `python -m pytest tests/territorial_defense/test_import_allowlist.py tests/territorial_defense/test_config.py tests/territorial_defense/test_report.py -v`
Expected: PASS (incl. `TerritorialDefenseParams(pitch_control_method="voronoi")` raises; conservation doctest holds).

- [ ] **Step 5: Commit boundary** — fold into the package's first substantive commit-unit with the engine (Task 6).

---

## Task 6: Engine — domain filter + removal counterfactual + FOV & removal-depletion gates + provenance/report

**Files:**
- Create: `silly_kicks/territorial_defense/_engine.py`
- Test: `tests/territorial_defense/test_engine.py`

**Interfaces:**
- Consumes: `resolve_defended_goals` (once), `region_observed_fraction` (SPEC-02), `zero_velocity_if_unavailable`, `compute_threat_pc` (PLAN-01 evidence test only), `id_compat`.
- Produces:
  - `remove_player_row(frame: pd.DataFrame, *, player_pos: int) -> pd.DataFrame` — the counterfactual frame (factual minus one row); PURE.
  - `select_arm_a_domain(actions, frames, *, params) -> pd.DataFrame` — D's defensive-action freeze-frames (type_id ∈ params.defensive_action_type_ids, D is the actor row present).
  - `local_completeness_ok(visible_polygon, center_xy, *, radius_m, min_fraction) -> bool` — SPEC-02: the disk of radius `radius_m` around D must be ≥ `min_fraction` observed (`region_observed_fraction` on a convex disk polygon). Missing polygon → `False`.
  - `removal_leaves_enough_defenders(frame, *, defending_team_id, min_after) -> bool` — **PLAN-01 guard**: the defending-team player count on the factual frame minus 1 (the removed defender) must be `≥ min_after`. Prevents the depleted-defender-side outlier that biases upward on thin FOV frames.
  - Drop-reason constants (`TD_SOURCE_VALUES`) + a `build_report(prov, params) -> TerritorialDefenseReport` (conservation by construction, CI-gated).

- [ ] **Step 1: `remove_player_row` — non-vacuity + purity (RED → green)**

```python
# tests/territorial_defense/test_engine.py
import numpy as np, pandas as pd
from silly_kicks.territorial_defense._engine import remove_player_row

def test_remove_player_row_drops_exactly_one_and_is_pure():
    f = pd.DataFrame({"player_id":[0,1,2,3], "is_ball":[False,False,False,True], "x":[1.,2.,3.,4.]})
    cf = remove_player_row(f, player_pos=1)
    assert cf is not f and len(f) == 4                # pure, input intact
    assert len(cf) == 3 and 1 not in cf["player_id"].tolist()
```

- [ ] **Step 2: `select_arm_a_domain` (RED → green)** — a fixture of actions with mixed type_ids; assert only `params.defensive_action_type_ids` rows with a present actor row survive.

- [ ] **Step 3: SPEC-02 local-completeness gate — its OWN RED step (PLAN-08)**

```python
def test_local_completeness_gate_two_sided():
    disk_center = (60.0, 34.0)
    observed = _polygon_covering(disk_center, radius=12.0)      # fully covers the 10 m disk
    cropped  = _polygon_half_covering(disk_center, radius=12.0) # covers ~40%
    from silly_kicks.territorial_defense._engine import local_completeness_ok
    assert local_completeness_ok(observed, disk_center, radius_m=10.0, min_fraction=0.7) is True
    assert local_completeness_ok(cropped,  disk_center, radius_m=10.0, min_fraction=0.7) is False
    assert local_completeness_ok(None,     disk_center, radius_m=10.0, min_fraction=0.7) is False   # missing polygon
```
Implement: build a convex regular 24-gon disk of radius `radius_m` around `center_xy`; call `region_observed_fraction(visible_polygon, disk)` (the disk is convex — the required `region` shape); return `frac >= min_fraction` (and `False` when `frac` is NaN / polygon absent). This gate prevents the *area*-under-observation upward bias.

- [ ] **Step 4: PLAN-01 removal-depletion guard — its OWN RED step**

```python
def test_removal_depletion_guard_two_sided_with_threat_evidence():
    from silly_kicks.tracking import compute_threat_pc
    from silly_kicks.territorial_defense._engine import removal_leaves_enough_defenders, remove_player_row
    # D is the SOLE defender: 1 defender (team 200) + 2 attackers (team 100) + ball
    thin = _frame_one_defender(def_team=200, atk_team=100)
    assert removal_leaves_enough_defenders(thin, defending_team_id=200, min_after=1) is False
    ok = _frame_three_defenders(def_team=200, atk_team=100)
    assert removal_leaves_enough_defenders(ok, defending_team_id=200, min_after=1) is True
    # EVIDENCE the guard is not cosmetic: removing the sole defender inflates the attacking threat
    xt, gm = _toy_fitted_xt(), _goal_map(thin)
    cf = remove_player_row(thin, player_pos=_sole_defender_pos(thin))     # -> 0 defenders
    assert compute_threat_pc(cf, attacking_team_id=100, xt=xt, goal_map=gm) \
         > compute_threat_pc(thin, attacking_team_id=100, xt=xt, goal_map=gm)
```
Implement: count the defending-team rows (`~is_ball & ids_match(team_id, defending_team_id)`); return `(count - 1) >= min_after`. (Evidence assertion documents *why* the delta on a depleted frame is inadmissible — the `_spearman.py:200/217/226` `def_tti.shape[0] > 0` guard makes the 0-defender surface attacker-controls-all, a finite but upward-biased outlier — so the frame is dropped, never scored.)

- [ ] **Step 5: Conservation test (ADR-042) — RED, both new drop-reasons non-vacuous**

```python
def test_engine_conserves_frames_dropped_and_counted():
    # a mixed domain: one scored + one fov_cropped_local + one removal_undersupported + one no_actor
    report = _run_engine_on_mixed_fixture()
    assert report.n_frames_scored + sum(report.drop_reasons.values()) == report.n_frames_in
    assert report.drop_reasons.get("fov_cropped_local", 0) >= 1        # non-vacuity
    assert report.drop_reasons.get("removal_undersupported", 0) >= 1   # non-vacuity (PLAN-01)
```
Implement the provenance cascade (mirror `gkdv/_engine.py:300-351`: a first-failing-reason `pd.Series(pd.NA)` fill, then a strict `eligible`/`dropped` partition). **Cascade order is deterministic:** `not_in_domain → no_actor → no_defenders → unresolved_geometry → fov_cropped → fov_cropped_local → removal_undersupported → velocity_unscoreable → scored` (a frame that is both cropped and depleted reports the first-failing reason, so the count is unambiguous). Run to green.

- [ ] **Step 6: Commit boundary** — engine + config/report/columns form commit-unit #2.

---

## Task 7: Arm A (action-anchored, identity-exact) + the cache-collapse & pinning guards

**Files:**
- Create: `silly_kicks/territorial_defense/_arms.py` (Arm A portion)
- Test: `tests/territorial_defense/test_arms.py`

**Interfaces:**
- Consumes: `compute_threat_pc`, engine `remove_player_row`, `GoalMap`.
- Produces:
  - `arm_a_threat_suppressed_batch(actual_frames, cf_frames, *, attacking_team_id_by_frame, xt, goal_map, params=_DEFAULT_PARAMS) -> pd.Series` — one `threat_suppressed` per `(game_id, period_id, frame_id)`, `= compute_threat_pc(cf) − compute_threat_pc(actual)` (attacker-value units; **positive = D's presence suppressed the attacking team's threat**). Mirrors `gkdv/_arms.py:285-343` (grouped loop, `_assert_legs_aligned`, NO try/except — threat is possession-independent). **Precondition:** the engine (Task 6) has already dropped `removal_undersupported` frames, so the arm never differences a depleted-defender cf leg.
  - `arm_a_threat_suppressed(actual_frame, cf_frame, *, attacking_team_id, xt, goal_map, params) -> float` (thin wrapper).
  - `_assert_legs_aligned(actual, cf, *, fn)` — verbatim shape of `gkdv/_arms.py:207-225` (row-for-row `(game_id, period_id, frame_id, player_id)`... **note the cf frame has one FEWER row**, so align on `(game_id, period_id, frame_id)` group keys + assert the cf is the actual minus exactly the removed player; adapt the helper accordingly and test it).
  - **No `pitch_control_cache` parameter anywhere in `_arms.py`.**

- [ ] **Step 1: Structural no-cache guard (RED-first)**

```python
import inspect
from silly_kicks.territorial_defense._arms import arm_a_threat_suppressed, arm_b_threat_suppressed
def test_arms_refuse_a_pitch_control_cache():
    for fn in (arm_a_threat_suppressed, arm_b_threat_suppressed):
        assert "pitch_control_cache" not in inspect.signature(fn).parameters
```

- [ ] **Step 2: Two-sided SPEC-03 behavioral guard — the collapse-to-zero landmine (RED-first)**

```python
def test_shared_cache_collapses_delta_to_zero():
    """ADR-043: serving the counterfactual leg the factual surface collapses the delta to 0.
    Two-sided: the correct path (D removed) is non-zero; the mis-wired path (cf == actual) is 0.
    """
    actual, xt, gm = _scored_arm_a_fixture()   # a fixture where D visibly controls space near the attack
    cf = remove_player_row(actual, player_pos=_defender_pos(actual))
    correct = arm_a_threat_suppressed(actual, cf, attacking_team_id=_ATK, xt=xt, goal_map=gm)
    miswired = arm_a_threat_suppressed(actual, actual, attacking_team_id=_ATK, xt=xt, goal_map=gm)
    assert miswired == 0.0                        # identical legs -> exactly 0 (the collapse signature)
    assert correct != 0.0                         # the real path measurably differs
```
(Mirrors `tests/gkdv/test_arms.py::test_unpinned_implementation_would_measurably_differ`; note gkdv has NO committed cache-collapse behavioral test — this is authored fresh per SPEC-03.)

- [ ] **Step 3: Implement Arm A; run Steps 1–2 to green.**

Body core (adapt `gkdv/_arms.py:316-343`):
```python
        atk = att_per_frame[ka]
        a = tracking.compute_threat_pc(a_sub, attacking_team_id=atk, xt=xt, goal_map=goal_map,
                                       method=params.pitch_control_method,
                                       params=SpearmanParams(lambda_gk=params.lambda_gk))
        c = tracking.compute_threat_pc(c_sub, attacking_team_id=atk, xt=xt, goal_map=goal_map,
                                       method=params.pitch_control_method,
                                       params=SpearmanParams(lambda_gk=params.lambda_gk))
        out[ka] = float(c - a)   # positive = D's presence suppressed the attacking team's threat
```

- [ ] **Step 4: Liveness/finiteness guard (non-vacuity, mirror `gkdv` test_arms.py:468-511)**

Assert the underlying `compute_threat_pc` legs are finite and the delta is a real number that is non-zero on the discriminating fixture (a zero delta from both legs is the all-degenerate signature). Run to green.

- [ ] **Step 5: Commit boundary** — Arm A + guards = commit-unit #3 (arms).

---

## Task 8: Arm B (hull-based, nearest-to-target) + slippage measurement

**Files:**
- Create: `silly_kicks/territorial_defense/_arms.py` (Arm B portion)
- Test: `tests/territorial_defense/test_arms_b.py`

**Interfaces:**
- Consumes: `territory.build_trimmed_hull` / `territory.Hull` (the v1 hull). **DECIDED (owner ruled 2026-09-09): export `build_trimmed_hull`/`Hull` from `territory.__init__`** and import via the public surface (single-source the hull definition; NO local re-implementation — so Arm B and v1 territory can never disagree about what "D's territory" is). The ADR-028 membership `refl = column_stack([fl - end_x, fw - end_y]); hull.contains(refl)`.
- Produces:
  - `arm_b_threat_suppressed_batch(...)` — for each opponent pass whose reflected target lands in D's hull: identify the contesting defender by POSITION (`nearest_to_target`: the defending-team row minimizing distance to the pass target), remove it, `threat_pc(cf) − threat_pc(actual)`, attribute to hull-owner D.
  - `arm_b_attribution_slippage(...)` — the measured rate at which the position-chosen contesting defender is NOT D (attribution error, lower = tighter; checked via the Arm-A actor anchor / on identity-bearing providers; honest-NaN when identity is un-measurable). Shipped alongside the number (spec §3).

- [ ] **Step 1a: Export the v1 hull from `territory` (Decision 4)** — add `build_trimmed_hull` and `Hull` to `silly_kicks/territory/__init__.py`'s import block + `__all__`; add a one-line test that `from silly_kicks.territory import build_trimmed_hull, Hull` succeeds and `build_trimmed_hull(xy, trim_fraction=0.70)` returns a `Hull` (or `None` for < 3 points). This is the single source of the hull definition for both v1 territory and Arm B.

- [ ] **Step 1b: Verify SPADL type_ids before writing the domain**

Run:
```python
from silly_kicks.spadl.config import actiontypes_df
print(actiontypes_df().query("type_name in ['tackle','interception','clearance']")[["type_id","type_name"]])
```
Confirm `(9,10,18)` and correct `params.defensive_action_type_ids` if the codebase differs.

- [ ] **Step 2–4: TDD Arm B** — a fixture with a known opponent pass into a known hull; assert the nearest-to-target defender is the one removed, and the suppressed value is positive when that defender controls the lane. Run to green.

- [ ] **Step 5: Slippage test** — on a fixture where identity is known, assert `arm_b_attribution_slippage` returns the true contesting-is-NOT-D rate (non-vacuous: a fixture where it is sometimes NOT D). Run to green.

- [ ] **Step 6: Commit boundary** — fold into commit-unit #3 (arms).

---

## Task 9: `compute_territorial_defense` orchestrator

**Files:**
- Create: `silly_kicks/territorial_defense/_compute.py`
- Test: `tests/territorial_defense/test_compute.py`

**Interfaces:**
- Produces:
  ```python
  def compute_territorial_defense(actions, frames, *, xt, links=None, visible_area=None,
                                  params=_DEFAULT_PARAMS) -> tuple[pd.DataFrame, TerritorialDefenseReport]
  ```
  (**DECIDED, owner ruled 2026-09-09: `ghost_model` is OMITTED in v1** — spec §12 listed `ghost_model=None`, but YAGNI / speculative-surface-debt wins since replacement is deferred; add a keyword-only `ghost_model` only when the replacement refinement is built, which is non-breaking.) `samples` has one row per `(game_id, player_id)` with `TD_SAMPLE_COLUMNS`; grouped on `canonical_id_series(player_id)`, raw id emitted via `.first()` (ADR-019); conserving `TerritorialDefenseReport`.
  - **The `compute_territorial_defense` docstring MUST carry the honest-limit (PLAN-10):** a one-paragraph note that the outputs are an INSTRUMENT-level threat-suppression estimate, **team-conditioned by construction**, and that **per-defender numbers are NOT a defender ranking** (the defender-vs-team confound is unidentifiable on a single-tournament/national-team corpus) — with a `See NOTICE / CLAUDE.md for the full limit.` cross-link. The docstring is one of the two places (with the glossary, Task 11) a code consumer reads the number's meaning.

- [ ] **Step 1: Write the failing e2e-shape test** — a small SB360-shaped fixture (built via `shape_snapshots` → `snapshot_to_tracking_frames` → `apply_actor_identities_to_frames`) with a fitted toy `xt`; assert samples has the right columns, one row per defender, report conserves, and `td_source` uses only `TD_SOURCE_VALUES`.

- [ ] **Step 2–4: Implement the orchestrator** — resolve `goal_map` once; apply the actor bridge; run Arm A over the domain (identity-exact); run Arm B over hull-membership passes; velocity self-degrade via `zero_velocity_if_unavailable`; aggregate per `(game, canonical player)`; assemble the report. Run to green.

- [ ] **Step 5: ADR-051 D3 orientation + ADR-019 id-dtype invariance tests** — mirror the restdefense suite's direction-invariance and id-dtype-invariance guards (numeric actions × string frames, and a mirrored-frames-fixed-`home_team_id` check). Run to green.

- [ ] **Step 6: Commit boundary** — orchestrator = commit-unit #4.

---

## Task 10: Probe module (owner-run battery machinery, CI-tested with fixtures)

**Files:**
- Create: `silly_kicks/territorial_defense/_probe.py`
- Test: `tests/territorial_defense/test_probe.py`

**Interfaces:**
- Produces (self-contained — adapts the gkdv pattern, does NOT import gkdv, so the allowlist stays `tracking`+`territory` public):
  - `MIN_DOMAIN_FRAMES = 200`, `SATURATING_MULTIPLE = 5.0`, **`TD_PROBE_RATIO = 2.0`** (NEW uniquely-named registration — not `PHYSICS_ARM_PROBE_RATIO`/`TF19_PROBE_RATIO`/`XS_PROBE_RATIO`).
  - `EXPECTED_DIRECTION = {"a_threat_suppressed": "positive", "b_threat_suppressed": "positive"}` and `expected_direction_for_arm(col)` raising `KeyError` on an unmapped column (mirror `gkdv/_validate.py:88-103`, sign flipped to `positive`).
  - `impose_defender_dose(frames, targets, *, dose, displacement=None, params)` — a DEFENDER dose imposer (adapt `gkdv/_probe.py:118-182`; the "actor"/contesting defender is the subject, not the keeper).
  - `layer0_instrument_verdict(*, realistic_abs, saturating_abs, placebo_p95, n_domain) -> str` and `layer1_responsiveness_verdict(*, real_med, nd_med, placebo_p95, n_domain) -> str` — verbatim math of `gkdv/_probe.py:195,270` with `TD_PROBE_RATIO`.
  - `paired_vector_controls(frames, targets, *, r, rng)` — one defending outfielder displaced by D's vector: nearest (`nd`) + `r` single-player placebos (adapt `gkdv/_probe.py:303-377`).

- [ ] **Step 1: Write failing verdict tests** — pooled-corpus statistics, non-vacuity (a per-shard-thin domain reads `arm_unscoreable`; the void `and` is load-bearing). Mirror `tests/gkdv/test_probe.py`.

- [ ] **Step 2–4: Implement; run to green.** Assert the direction registry raises on an unknown arm column, and `n_domain < 200 → arm_unscoreable`.

- [ ] **Step 5: Commit boundary** — probe = commit-unit #5.

---

## Task 11: Governance — feature glossary, NOTICE, C4 container, SB360 boundary verdicts

**Files:**
- Modify: `silly_kicks/feature_glossary.py`, `NOTICE`, `docs/c4/architecture.dsl` (+ re-render `.html`), `tests/sb360/_registry.py`
- Test: existing gates (`tests/test_c4_dsl_description_cap.py`, `tests/sb360/*`, glossary coverage gate)

**Interfaces:**
- Produces: the new package fully registered in every governance surface.

- [ ] **Step 1: Feature glossary (ADR-048)** — register the metric columns (`a_threat_suppressed`, `b_threat_suppressed`, `b_attribution_slippage`) with `emitting_module="silly_kicks.territorial_defense._arms"`, `unit="xT"`, `higher_is_better=True`, attribution. (`a_frames_scored`/`b_frames_scored`/`td_source` are counts/provenance — excluded, like territory's provenance columns.) **PLAN-10: each `definition` string MUST include the honest-limit clause** — e.g. "…an instrument-level, team-conditioned threat-suppression estimate; NOT a defender ranking" — because `definition` is the second place (with the docstring, Task 9) a code consumer reads the number's meaning. Run the glossary coverage gate.

- [ ] **Step 2: NOTICE (ADR-005)** — add a "Territorial defense (SB360 counterfactual)" entry citing Fernández & Bornn (pitch control / marginal player value), Spearman, Le et al. 2017 (ghosting comparator). Cross-link from the module docstrings.

- [ ] **Step 3: C4 container (+1 count)** — add `territorial_defense = container "silly_kicks.territorial_defense" "..." "Python" "Library"` to `docs/c4/architecture.dsl` (≤200 chars description) + relationships (`analyst -> territorial_defense`, `territorial_defense -> tracking`, `-> territory`, `-> xthreat`). Re-render via Graphviz `dot` per project convention (NEVER Smetana): `structurizr.war export → c4_assemble.py → plantuml.jar -graphvizdot "C:/Users/Karsten/.claude/tools/graphviz/dot.exe" → c4_assemble.py --svg-dir`. Run `tests/test_c4_dsl_description_cap.py` (derived-completeness — a new shipped subpackage MUST have a container).

- [ ] **Step 4: SB360 boundary verdicts (ADR-053)** — add `territorial_defense.compute_territorial_defense` (and/or the arms) to `BOUNDARY_ENTRY_POINTS` in `tests/sb360/_registry.py` with a per-column `verdict` + `verdict_provenance` (threat arm is a Tier-1 dimensionless lift → `differs_by_design`/`structural` where velocity-invariant; velocity-constitutive quantities → `honest_nan`). Run `tests/sb360/`.

- [ ] **Step 5: Commit boundary** — governance = commit-unit #6 (with Task 3's territory-door change).

---

## Task 12 (OWNER-RUN): Validation driver + battery + PassCompletionModel weights

> These steps are **compute-heavy and owner-run** (full SB360 corpus). The agent builds the DRIVER and its CI wiring (TDD-able with fixtures); the owner RUNS it and commits the artifacts. Reported-not-gated (spec §9); promotion of any default is a separate ADR-009 decision.

**Files:**
- Create: `scripts/validate_territorial_defense.py` (mirror `scripts/validate_territory_counterfactual.py` structure + `scripts/build_tf19_instrument_responsiveness.py`)
- Modify (owner-run output): commit `silly_kicks/expected_passing/weights/*` (PassCompletionModel bundled weights) + `docs/research/territorial_defense_construct_validity/` report.
- Test: `tests/scripts/test_validate_territorial_defense.py` (argparse + `require_clean_tree` FIRST + `for_each`/`declare_inputs` wiring + locked pre-registered constants).

**Interfaces:**
- Consumes: `load_statsbomb_matches`, `scripts/_sb_open_data.py`, `for_each` (ADR-052 shards, `_EMITTED_SHARD_COLUMNS`+`_SHARD_SCHEMA_VERSION`), `require_clean_tree` (ADR-037), `declare_inputs` (ADR-056), the probe module, the arms.

- [ ] **Step 1: Build the driver (agent, TDD with a tiny fixture corpus)** — `main()` calls `require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)` FIRST; `for_each` shards per match into a gitignored root; pre-registered LOCKED constants (`ELITE_DEFENDER_PRIOR` — the "Van Dijk" idiom, `MIN_DOMAIN_FRAMES`, `TD_PROBE_RATIO`, saturating multiple) referenced not inlined; writes `metrics.json` + `named_defender_signs.parquet` with `run_commit`/`run_tree_dirty`/`input_contract`. Wire the `_EMITTED_SHARD_COLUMNS`/`_SHARD_SCHEMA_VERSION` pin (4.77.1). Run `tests/scripts/test_validate_territorial_defense.py` green.

- [ ] **Step 2: (OWNER-RUN) Train + bundle `PassCompletionModel` weights** — `python scripts/train_pass_completion.py` on the public corpus (public-only, `assert_public_corpus`); commit `silly_kicks/expected_passing/weights/*` + `SHA256SUMS`. Un-skip any Task-1 Step-6 skipped `bundled()` tests. (Needed for the receiver-lane Arm-B evaluation leg.)

- [ ] **Step 3: (OWNER-RUN) Run the battery** — dose-response (layer-0/layer-1 pooled verdicts), paired single-player placebo controls, the locked elite-defender prior on Arm A, the Arm-B slippage leg, the receiver-lane-vs-nearest-to-target comparison. Write `docs/research/territorial_defense_construct_validity/findings.md` + `metrics.json` (with provenance). **Report only — no default change.**
  - **The elite-defender prior is reported as FACE-VALIDITY, explicitly caveated with the elite-defender/elite-team collinearity** (Global Constraints honest-limit): in WC2022 an elite defender and a strong defensive national side are perfectly collinear, so a clean prior does NOT license attribution or ranking. State this in the report next to the number.
  - **Measure and report whether the 30 licensed SB360 matches add ANY cross-team defender replication** (a defender observed on ≥2 teams). WC2022 alone adds none (one player = one national team). If the licensed set is also single-team-per-player, record that the corpus has ZERO identifying power for the defender-vs-team confound.
  - Report the team-confound as a first-class honest-limit in `findings.md` (not a footnote).

- [ ] **Step 4: Present the validation report to the owner.** Any promotion (e.g. making `receiver_lane` the Arm-B default, or shipping the metric as a recommended surface) is a SEPARATE ADR-009 decision after the owner reads the report — NOT this cycle. **Player-attribution / defender ranking is explicitly OUT of this cycle** and, if ever pursued, is a future ADR-009 gated on a crossed defender+team variance decomposition (ICC — not CV) over a MULTI-CLUB TRANSFER corpus, NOT WC2022/national-team data. (The eyestone collaboration has number-gate-verified code for exactly this: a Bayesian crossed keeper+team ICC identifiable from a club corpus with ~17 transfer keepers / 49-of-79 multi-keeper clubs — the analogous defender substrate would be the reuse target.)

---

## Task 13: Release prep — STOP for owner approval before commit

**Files:**
- Modify: `silly_kicks/_version.py`, `CHANGELOG.md`, `CLAUDE.md`, `TODO.md`, `docs/superpowers/adrs/ADR-0XX-*.md` (new), the plan/spec files.

- [ ] **Step 1: Re-derive the free identifiers** — `git fetch origin main` then read the latest version/PR/ADR on `main`; compute next-free (expected `4.112.0`/`PR-S183`/`ADR-090` but **re-confirm**, nobody reserves).
- [ ] **Step 2: Bump `silly_kicks/_version.py`** to the re-derived version (single source, ADR-079); `uv lock`.
- [ ] **Step 3: Write the ADR** (`ADR-0XX-tf54b-sb360-territorial-defense.md`) — the five rulings + the removal/marginal mechanism + the actor bridge + the honest-limit reporting + supersession of the event-only `ab9001c` approach.
- [ ] **Step 4: CHANGELOG entry** (keyed by the re-derived `PR-Snnn`) — additive, no retrain, +1 C4 container, kept seams carried, event-only cone not carried.
- [ ] **Step 5: CLAUDE.md** — add the `territorial_defense` durable-contract bullet; the bullet MUST carry the honest-limit verbatim in spirit: **"validated as an INSTRUMENT, NOT player-attributable — the marginal-removal delta is team-conditioned by construction; in a single-tournament/national-team corpus the defender-vs-team confound is unidentifiable, so per-defender numbers are not a defender ranking; ranking is a future ADR-009 gated on a crossed defender+team ICC over a multi-club transfer corpus."** Also rewrite the stale "Territory `method="counterfactual"` + Expected-Passing" bullet (which describes the abandoned cone) to reflect: `territory` event-only `completed_failed` only (door removed per Task 3, if the owner rules remove), `expected_passing`/`destination_profiles` retained as reusable seams, counterfactual now lives in `territorial_defense`.
- [ ] **Step 6: TODO.md** — move TF-54b to shipped; note replacement-refinement (above-replacement ghost) + receiver-lane-promotion as owner-gated follow-ons.
- [ ] **Step 7: `final-review` skill** — run the pre-commit quality gate + re-render C4.
- [ ] **Step 8: Full CI-faithful gate**

Run: `python -m ruff check silly_kicks/ tests/ scripts/` ; `python -m ruff format --check silly_kicks/ tests/ scripts/` ; `python -m pyright` ; `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all green.

- [ ] **Step 9: STOP.** Show the owner the full diff / file list and the single proposed commit message. **Do NOT commit or push.** Wait for explicit approval for this specific commit (commit-discipline hard gate). On approval: one coherent commit (all units above squashed into one fully-tested state — never micro-commits), close draft PR #235, open the new PR from `feat/tf54b-sb360-territorial-defense`.

---

## Self-Review

**1. Spec coverage:**
- §2 identity constraint → Tasks 2, 4 (actor bridge). ✅ (Item C: the "only the actor is reliably identified; positions suffice for the counterfactual" conclusion was independently reached by the eyestone SB360 reconstruction — a second cross-check.)
- §3 Arm A / Arm B / both → Tasks 7, 8. ✅ (sign convention `positive = suppressed` pinned in Global Constraints + Task 7/10.)
- §3 domain yield (SPEC-01, 7,051 frames = **pre-gate**) → the *scoreable* yield is smaller (gated by `fov_cropped_local` AND the new `removal_undersupported`, PLAN-01/PLAN-09); the driver (Task 12) reports the post-gate scoreable yield explicitly and compares it against the `MIN_DOMAIN_FRAMES=200` floor (owner accepted "measure in driver" at spec time). ✅
- §4 actor bridge (re-plumb + `apply_actor_identities_to_frames`) → Tasks 2, 4. ✅
- §5 removal mechanism + SPEC-02 local-completeness gate + PitchControlCache trap → Tasks 6 (gate), 7 (SPEC-03 cache-collapse test). ✅
- §5 replacement DEFERRED → `ghost_model` omitted (Task 9, flagged). ✅
- §6 threat model (`compute_threat_pc`, unfitted-xt raise, velocity self-degrade) → Tasks 7, 9. ✅
- §7 new package + import gates + `territory` stays event-only + kept seams → Tasks 1, 3, 5. ✅ (whole-tree `nothing-imports` sweep authored fresh — Task 5 Step 1.)
- §8 honest limits (velocity tiers, FOV, Arm-A narrowness, Arm-B slippage) → Tasks 6, 8, 9. ✅ **+ the team-confound / not-player-attributable limit (rev-3, Item A)** → Global Constraints + Tasks 12–13. ✅
- §9 validation battery + named CI regression tests (SPEC-03, SPEC-02, standard gates) → Tasks 7, 6, 9, 10, 12. ✅
- §10 corpus & drivers (`for_each`, `require_clean_tree`, `load_statsbomb_matches`) → Task 12. ✅
- §11 decisions → all five in Global Constraints; Decision 3 door-removal in Task 3 (flagged), Decision 5 vehicle in Tasks 1 + 13. ✅
- §12 interfaces → signatures quoted in Starting-state facts + per-task Interfaces. ✅ (`ghost_model` omission flagged.)
- §13 attribution → Task 11 Step 2. ✅

**2. Placeholder scan:** Code steps carry real code or exact `file:line` edits. The two genuinely deferred spots are explicitly owner-run (Task 12 Steps 2–3) or flagged sub-decisions (Task 3 door-removal; Task 9 `ghost_model` omission; Task 4 snapshot-only-column choice) — surfaced for review, not silent.

**3. Type consistency:** `TerritorialDefenseParams`/`TerritorialDefenseReport`/`TD_SAMPLE_COLUMNS`/`TD_SOURCE_VALUES` names consistent across Tasks 5–12. `arm_a_threat_suppressed[_batch]`/`arm_b_threat_suppressed[_batch]` consistent Tasks 7–9. `apply_actor_identities_to_frames` consistent Tasks 4, 9. `compute_threat_pc(... method=..., params=SpearmanParams(lambda_gk=...))` consistent Tasks 7, 9.

**Owner rulings (2026-09-09) — all five resolved, no open items:**
1. **Task 3** — **REMOVE** `territory`'s reserved `counterfactual` door (shrink `TERRITORY_METHODS` to `{"completed_failed"}`, delete the `NotImplementedError` branch).
2. **Task 4** — `is_actor` is a **snapshot-only extension column** (base `TRACKING_FRAMES_COLUMNS` unchanged).
3. **Task 9** — **omit** `ghost_model` from the v1 signature (add a keyword-only param only when the replacement refinement is built).
4. **Task 8** — **export** `build_trimmed_hull`/`Hull` from `territory.__init__` (single-source the hull; no local re-implementation).
5. **Task 4 (Item B)** — the actor bridge lives in **`silly_kicks/keeper_identity.py`** (a future rename to a neutral `identity.py` is out of scope for this cycle).
