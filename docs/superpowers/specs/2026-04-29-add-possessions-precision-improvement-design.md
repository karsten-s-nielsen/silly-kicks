# `add_possessions` algorithmic precision improvement (PR-S12)

**Status:** Approved (design)
**Target release:** silly-kicks 2.1.0
**Author:** Claude Opus 4.7 (1M)
**Date:** 2026-04-29
**Predecessor:** 2.0.0 (`docs/superpowers/specs/2026-04-29-converter-identifier-conventions-design.md`)

---

## 1. Problem

`silly_kicks.spadl.add_possessions` is the post-conversion possession-reconstruction helper for sources without a native possession sequence (Wyscout, Sportec, Metrica, Opta, kloppy). Its team-change-with-carve-outs heuristic was published in 1.2.0 and validated against StatsBomb's native `possession` field in 1.8.0 (PR-S8) with measured per-match boundary recall ≈ 0.95 and precision ≈ 0.41 across 3 fixtures (W-WC, CL, PL).

PR-S8 deferred algorithmic precision improvements to a follow-up. The hypothesis at the time, paraphrased from the brief: brief-opposing-action merge yields +20-30pp precision; defensive-action class yields +15-20pp; spatial continuity yields +5pp. None of these were measured.

Three problems with the status quo:

1. **The hypothetical impacts were never measured.** A PR-S12 baseline campaign across 64 StatsBomb WorldCup-2018 matches (2026-04-29) shows the actual impacts are smaller and come at recall cost. Reality:
   - brief-opposing-action merge: +7-12pp precision at -4-7pp recall.
   - defensive-action class: +5pp precision at -3.5pp recall.
   - spatial continuity: +2-3pp precision but breaches the 0.85 recall floor at every meaningful setting.
   - All meaningful multi-rule combinations breach the per-match recall floor.

2. **`max_gap_seconds=5.0` default is empirically suboptimal.** PR-S8 brainstorming reported that LL2's parameter sweep peaked F1 at `max_gap_seconds=10.0`, but only F1 was reported. The full sweep:

   | `max_gap_seconds` | P_mean | R_mean | F1_mean | R_min |
   |---|---|---|---|---|
   | 3.0 | 0.318 | 0.963 | 0.477 | 0.881 |
   | **5.0 (current default)** | **0.412** | **0.950** | **0.574** | **0.865** |
   | 7.0 | 0.439 | 0.939 | 0.597 | 0.854 |
   | 10.0 | 0.442 | 0.919 | 0.596 | 0.838 |
   | 15.0 | 0.436 | 0.887 | 0.583 | 0.789 |
   | 20.0 | 0.430 | 0.863 | 0.573 | 0.784 |

   The Pareto-optimal default at `R_min ≥ 0.85` is **gap=7.0**: +2.7pp precision over current default, -1pp recall, peak F1, no per-match recall breach. `gap=10.0` is mildly better F1 but breaches the 0.85 floor.

3. **The committed 64-match WorldCup-2018 HDF5 fixture (`tests/datasets/statsbomb/spadl-WorldCup-2018.h5`, 6 MB, shipped 1.9.0) does not preserve the StatsBomb `possession` column.** The build script's adapter drops the field before conversion (`scripts/build_worldcup_fixture.py:145-165`) and the conversion call uses `preserve_native=None` (`:206`). The HDF5 is therefore not a usable regression corpus for `add_possessions`. Every measurement campaign — this one and any future one — must currently re-process the 64 raw open-data JSONs, which costs ~3-5 minutes of CPU and depends on the `.cache/events/` raw files (gitignored, only present after running the build script). This is the second-largest friction point for `add_possessions` regression validation; PR-S8 was the largest and resolved it for the 3-fixture case.

## 2. Goals

1. **Add three opt-in, keyword-only precision-improvement parameters** to `silly_kicks.spadl.add_possessions` and its atomic counterpart `silly_kicks.atomic.spadl.add_possessions`. Default behavior unchanged from a rule-presence perspective; consumers explicitly enable the rules they want.

2. **Change `max_gap_seconds` default from 5.0 to 7.0** in both standard and atomic `add_possessions`. The change is a behavior break under strict semver, but luxury-lakehouse is the only known consumer and pragmatic semver applies. CHANGELOG calls out the change loudly. Opt-out is straightforward: `add_possessions(actions, max_gap_seconds=5.0)`.

3. **Add `possession` to the WorldCup-2018 HDF5 fixture.** Update `scripts/build_worldcup_fixture.py`'s adapter to include `possession`, switch the conversion call to `preserve_native=["possession"]`, regenerate the HDF5. Net file size delta is small (one int column on ~140K rows ≈ +1 MB at zlib compression).

4. **Extend the `add_possessions` regression CI gate** from 3 committed JSON fixtures to *both* the 3 JSONs (cross-competition coverage: W-WC + CL + PL) AND the 64-match HDF5 (within-competition variance: FIFA WC 2018). Both gates run in CI on every PR; both use the same `boundary_metrics` utility.

5. **Lower the per-match recall floor from 0.85 to 0.83.** The new `max_gap_seconds=7.0` default produces R_min=0.854 in the worst-case match (across 64 matches measured). 0.85 leaves only 0.4pp margin and would be flap-prone under pandas/numpy version drift between local and CI; 0.83 leaves a comfortable 2.4pp margin. Precision floor stays at 0.30 (current observation: P_min=0.350 at gap=7.0).

6. **Update docstrings** with the empirical numbers from this campaign and a recommended-settings tradeoff table for the 3 new opt-in rules.

7. **Atomic-SPADL parity.** Same parameters, same `max_gap_seconds=7.0` default, same validation rules. Mirrored in `silly_kicks/atomic/spadl/utils.py::add_possessions._compute_possessions`.

8. **Bump version to 2.1.0.** All API changes are additive parameters; the default change is a documented behavior break under pragmatic semver.

## 3. Non-goals

1. **No spatial continuity rule.** Measured impact (+3pp precision) does not justify the recall cost (breaches 0.85 floor at all useful settings). Park indefinitely; revisit only if a consumer surfaces with a specific use case.

2. **No multi-rule defaults.** All combinations measured in this campaign breach the recall floor; opt-in only, never default-on. Consumers wanting aggressive precision can stack the opt-in flags themselves with explicit awareness of recall tradeoff.

3. **No `add_possessions_strict` variant.** Same justification as PR-S8: keep one function, parameterize. Resist proliferation.

4. **No new `BoundaryMetrics` fields.** `boundary_metrics` (1.8.0 public API) stays as-is. Any future per-match-aggregate metric goes to a separate utility, not a `BoundaryMetrics` extension.

5. **No threshold change for the precision floor.** Stays at 0.30 per match. `gap=7.0` produces P_min=0.350 worst case, comfortable margin.

6. **No touching luxury-lakehouse.** Lakehouse will pin `silly-kicks>=2.1.0,<3.0` as a separate cycle. They are responsible for deciding whether to opt into the new rules and/or revert `max_gap_seconds=5.0` if their pipeline depends on the old default.

7. **No silly-kicks 3.0.0 bump.** Pragmatic semver — known small consumer set, single coordinated lakehouse cycle, CHANGELOG callout. PR-S11 (2.0.0) was a major because the `team_id` / `player_id` contract change had ambiguous downstream implications; this default change is documented and easy to opt out of.

## 4. Architecture

### 4.1 File structure

```
silly_kicks/spadl/utils.py                       MOD  +120 / -10  add 3 opt-in params; gap default 5.0→7.0; refactor boundary computation; docstring
silly_kicks/atomic/spadl/utils.py                MOD  +110 / -10  same params on atomic side; mirror gap default; docstring
tests/spadl/test_add_possessions.py              MOD  +180 / -10  unit tests for new params + lower 3-JSON gate to 0.83 + add HDF5 64-match parametrized test
tests/atomic/test_atomic_add_possessions.py      MOD  +60  / 0    unit tests for new params on atomic side
scripts/build_worldcup_fixture.py                MOD  +3  / -1    adapter includes `possession`; conversion uses preserve_native
tests/datasets/statsbomb/spadl-WorldCup-2018.h5  MOD  ~+1 MB     regenerated with `possession` column preserved
silly_kicks/spadl/__init__.py                    NO CHANGE       boundary_metrics + add_possessions already re-exported
pyproject.toml                                   MOD  +1 / -1     version 2.0.0 → 2.1.0
CHANGELOG.md                                     MOD  +50 / 0     ## [2.1.0] entry with breaking-change callout
TODO.md                                          MOD  +5 / -1     close PR-S12; update PR-Sxx queue
docs/superpowers/specs/2026-04-29-add-...        NEW             this design document
```

### 4.2 New API: opt-in precision-improvement parameters

```python
def add_possessions(
    actions: pd.DataFrame,
    *,
    max_gap_seconds: float = 7.0,                          # CHANGED from 5.0
    retain_on_set_pieces: bool = True,
    merge_brief_opposing_actions: int = 0,                 # NEW
    brief_window_seconds: float = 0.0,                     # NEW
    defensive_transition_types: tuple[str, ...] = (),      # NEW
) -> pd.DataFrame:
```

**Rule 1 — `merge_brief_opposing_actions` + `brief_window_seconds` (paired):**

If team B has between 1 and `merge_brief_opposing_actions` consecutive actions sandwiched between team A actions, AND the time from the first B-action to the next A-action is ≤ `brief_window_seconds`, suppress both team-change boundaries (A→B and B→A). The intuition: a brief deflection / interception that doesn't progress to a new possession sequence (StatsBomb merges these into the containing possession).

Both parameters are required to enable the rule:
- Both `0` → rule disabled (default).
- Both `> 0` → rule enabled with the specified window.
- Exactly one `> 0` → `ValueError` (config error; user probably forgot one).

Recommended setting (per measurement; eligible at the new 0.83 recall floor that PR-S12 establishes):
- `merge_brief_opposing_actions=2, brief_window_seconds=2.0` → P=0.483, R=0.910, F1=0.630 (R_min=0.843, exactly above 0.83 floor with 1.3pp margin).

For consumers willing to accept R_min ≥ 0.80 in their own downstream gate:
- `merge_brief_opposing_actions=3, brief_window_seconds=3.0` → P=0.530, R=0.882, F1=0.662 (R_min=0.811).

Note: silly-kicks's own CI gate runs `add_possessions(actions)` with all rules disabled (default behavior). Consumers who opt into a rule are responsible for choosing thresholds that match their tolerance for recall loss. The recommended-setting table is informational, not a CI guarantee.

**Rule 2 — `defensive_transition_types`:**

Action types listed are not boundary-triggering on their own. A team-change to one of these types does NOT immediately start a new possession; the next non-defensive action's team determines whether the boundary fires. Implementation: AND-out boundary mask where current action's `type_id` is in the configured set.

Allowed values are any subset of `silly_kicks.spadl.config.actiontypes`. Validation raises `ValueError` if an unknown name is provided. Recommended:

- `("interception", "clearance")` → P=0.461, R=0.915, F1=0.612 (R_min=0.854; eligible at 0.85 floor).
- `("interception",)` → P=0.425, R=0.915, F1=0.580 (R_min=0.854).
- `("interception", "clearance", "tackle")` → P=0.476, R=0.868, F1=0.614 (R_min=0.795; below 0.85 floor — opt in only with eyes open).

**Default behavior (both rules disabled, gap=7.0):** P=0.439, R=0.939, F1=0.597. This is silly-kicks 2.1.0's documented baseline.

**Validation:**

```python
if max_gap_seconds < 0:
    raise ValueError(...)
if merge_brief_opposing_actions < 0:
    raise ValueError(...)
if brief_window_seconds < 0:
    raise ValueError(...)
# XOR: both > 0 to enable, both 0 to disable
if (merge_brief_opposing_actions > 0) != (brief_window_seconds > 0):
    raise ValueError(
        "merge_brief_opposing_actions and brief_window_seconds must both be > 0 "
        "to enable the brief-opposing-merge rule, or both 0 to disable. "
        f"Got merge_brief_opposing_actions={merge_brief_opposing_actions}, "
        f"brief_window_seconds={brief_window_seconds}."
    )
# defensive types must be in canonical SPADL vocabulary
invalid = [t for t in defensive_transition_types if t not in spadlconfig.actiontype_id]
if invalid:
    raise ValueError(...)
```

### 4.3 `max_gap_seconds` default change

Default changes from 5.0 to 7.0. This is a behavior break — same input DataFrame produces different `possession_id` values for any pair of actions where the time gap is in `[5, 7)` seconds AND the team did not change.

**Rationale:**
- Empirical Pareto-optimal at the 0.85 floor (now 0.83).
- LL2's earlier sweep flagged the suboptimality (peak F1 at 10.0).
- Lakehouse is the only known consumer.
- Easy opt-out: `add_possessions(actions, max_gap_seconds=5.0)`.

**Coordination:** lakehouse is responsible for deciding whether to opt out or accept. `silly-kicks>=2.1.0,<3.0` constraint is sufficient; the opt-out is one keyword argument.

### 4.4 Atomic-SPADL parity

`silly_kicks/atomic/spadl/utils.py::add_possessions` adds the same three parameters and the same `max_gap_seconds=7.0` default. The boundary-mask computation lives in `_compute_possessions` (already factored). The card-transparency special case (yellow/red cards inheriting possession_id from surrounding context) is unchanged.

Action-type-name validation against `silly_kicks.atomic.spadl.config.actiontypes` (not the standard one — atomic vocabulary is partly different: `corner_short`/`corner_crossed` collapse to `corner`, `freekick_*` collapse to `freekick`, etc.). The 4 candidate defensive types — `interception`, `clearance`, `tackle`, `bad_touch` — are present in both vocabularies (verified at design time).

### 4.5 HDF5 fixture update

Two-line change in `scripts/build_worldcup_fixture.py`:

```python
# _adapt_events_to_silly_kicks_input — add `possession` field
"possession": e.get("possession"),

# _convert_match — switch conversion to preserve native field
actions, _report = statsbomb.convert_to_actions(
    adapted, home_team_id=home_team_id, preserve_native=["possession"]
)
```

Regenerate the HDF5: `uv run python scripts/build_worldcup_fixture.py --verbose`. Cache hits on the 64 raw JSONs already present in `tests/datasets/statsbomb/raw/.cache/events/` make this fast (~30s total).

The new HDF5 size will be ~7 MB (vs current ~6 MB; one extra int column at zlib compression). Stays well under the 50 MB warn threshold.

The 5 existing prediction tests (`test_predict*` in `tests/vaep/`, `tests/test_xthreat.py`, `tests/atomic/test_atomic_vaep.py`) read action DataFrames from `actions/game_<id>` keys. Adding a column does not break them; they project to the columns they consume.

`tests/conftest.py::sb_worldcup_data` fixture stays unchanged.

### 4.6 Test infrastructure

**Existing 3-JSON parametrized test** (`tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBombNative`):
- Lower the recall gate from 0.85 → 0.83 (assertion message updates).
- Precision gate stays at 0.30.
- F1 stays in the assert message only (diagnostic; not gated).
- Test class docstring updated to reflect the new default and gate.

**New 64-match HDF5 parametrized test** (same file, new class `TestBoundaryAgainstStatsBomb64Match`):
- Read from the `sb_worldcup_data` HDFStore session-scoped fixture.
- Iterate `actions/game_<id>` keys; for each match, run `add_possessions` → `boundary_metrics` against the stored `possession` column.
- Per-match independent gates: `recall ≥ 0.83 AND precision ≥ 0.30`.
- Failure assert message includes match_id + recall + precision + F1.
- Total: 64 parametrized cases, ~1-2s additional CI time after the HDF5 cold-load (HDFStore read is the bottleneck; per-match `add_possessions` is fast).
- Atomic-side does NOT add a 64-match equivalent in this PR — atomic regression testing stays on hand-built fixtures + the new unit tests. A 64-match atomic regression test is a future enhancement if a consumer surfaces.

**New unit tests** for the 3 opt-in parameters on hand-constructed mini fixtures:

```
TestBriefOpposingMerge
├── test_aba_within_window_suppresses_boundary       # A, B (1s), A → boundary suppressed
├── test_aba_outside_window_keeps_boundary           # A, B (5s), A → boundary kept (window exceeded)
├── test_abba_within_window_with_n_eq_2_suppresses   # A, B, B, A (N=2, 1s) → suppressed
├── test_abbba_with_n_eq_2_keeps_boundary            # A, B, B, B, A (N=2, 1s) → kept (3 > N=2)
├── test_disabled_when_both_zero                     # both 0 → identical to no-rule baseline
├── test_partial_config_raises                       # one>0, other=0 → ValueError
├── test_game_boundary_blocks_lookahead              # team flips back across game_id change → not merged
└── test_period_boundary_blocks_lookahead            # team flips back across period_id change → not merged

TestDefensiveTransitions
├── test_interception_does_not_trigger_boundary
├── test_pass_after_interception_triggers_if_team_change
├── test_unknown_type_raises_value_error
├── test_empty_tuple_no_op

TestMaxGapDefaultIs7Seconds
├── test_default_value_is_7                          # signature inspection
├── test_5_to_6s_gap_no_boundary_at_default          # behavior at new default
└── test_6_to_8s_gap_boundary_at_default             # behavior break vs 1.x
```

Atomic-side mirrors the same suite in `tests/atomic/test_atomic_add_possessions.py`.

### 4.7 Docstring updates

`silly_kicks/spadl/utils.py::add_possessions` docstring:
- Replace the current Empirical baselines paragraph with the gap=7.0 numbers (P=0.439, R=0.939, F1=0.597) and the 64-match basis.
- Add a "Precision-improvement rules" section listing the 3 opt-in rules with their measured tradeoffs (recommended settings, expected P/R/F1 deltas).
- Document the `max_gap_seconds` default change in a `Versionchanged` directive.

`silly_kicks/atomic/spadl/utils.py::add_possessions` docstring:
- Same updates, with atomic-vocabulary disclaimers (set-piece names collapse, card transparency).

`tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBombNative` class docstring:
- Threshold update (0.85 → 0.83).
- Reference the 64-match HDF5 test as the within-competition complement.

### 4.8 Boundary-mask helper extraction (optional refactor)

`silly_kicks/spadl/utils.py::add_possessions` is currently a single function with vectorized boundary detection inlined. With three new rules added, the function approaches 100 lines. Refactor: extract `_compute_possession_boundaries(sorted_actions, **opts) -> np.ndarray` mirroring the atomic-side `_compute_possessions` pattern. The public function becomes sort + delegation + cumsum + dtype.

Trade-off: marginal complexity reduction (good for readability) vs cross-file churn (CLAUDE.md doesn't mandate this style). Decision: refactor in this PR — the rule additions justify it and the atomic side already has the pattern.

## 5. Empirical baselines (this campaign)

Source: 64 FIFA WorldCup-2018 matches via StatsBomb open-data (the same fixtures vendored in `tests/datasets/statsbomb/raw/.cache/events/` by `scripts/build_worldcup_fixture.py`). Per-match boundary metrics computed via `silly_kicks.spadl.boundary_metrics` after dropping synthetic dribbles (`possession.notna()`).

**Baseline at default `max_gap_seconds=5.0` (silly-kicks 2.0.0 behavior):**

| Metric | Mean | sd | Min |
|---|---|---|---|
| Precision | 0.412 | 0.032 | 0.327 |
| Recall | 0.950 | 0.021 | 0.865 |

**At new default `max_gap_seconds=7.0`:**

| Metric | Mean | sd | Min |
|---|---|---|---|
| Precision | 0.439 | 0.035 | 0.350 |
| Recall | 0.939 | 0.023 | 0.854 |

**Recall-floor 0.85 eligible, ranked by P_mean (single-rule only):**

| Rule | P_mean | R_mean | F1_mean | R_min |
|---|---|---|---|---|
| `defensive_transition_types=("interception", "clearance")` | 0.461 | 0.915 | 0.612 | 0.854 |
| `max_gap_seconds=7.0` (proposed default) | 0.439 | 0.939 | 0.597 | 0.854 |
| `merge_brief_opposing_actions=1, brief_window_seconds=1.0` | 0.428 | 0.940 | 0.588 | 0.865 |
| `defensive_transition_types=("interception",)` | 0.425 | 0.915 | 0.580 | 0.854 |
| baseline (default 5.0) | 0.412 | 0.950 | 0.574 | 0.865 |

**Recall-floor 0.80 eligible, top 5 by F1_mean:**

| Rule | P_mean | R_mean | F1_mean | R_min |
|---|---|---|---|---|
| `merge_brief_opposing_actions=3, brief_window_seconds=3.0` | 0.530 | 0.882 | 0.662 | 0.811 |
| `merge_brief_opposing_actions=2, brief_window_seconds=3.0` | 0.524 | 0.890 | 0.659 | 0.816 |
| `merge_brief_opposing_actions=2, brief_window_seconds=2.0` | 0.483 | 0.910 | 0.630 | 0.843 |
| `merge_brief_opposing_actions=1, brief_window_seconds=2.0` | 0.476 | 0.921 | 0.627 | 0.851 |
| `defensive_transition_types=("interception", "clearance")` | 0.461 | 0.915 | 0.612 | 0.854 |

These tables go into the `add_possessions` docstring (compact version) and `CHANGELOG.md` (linked).

## 6. Pipeline (e2e regression test, end-to-end)

3-JSON test (existing, threshold update only):
```
1. Load tests/datasets/statsbomb/raw/events/<match_id>.json (3 matches)
2. Adapt → statsbomb.convert_to_actions(preserve_native=["possession"])
3. Filter to non-synthetic rows (possession.notna()).copy()
4. add_possessions(filtered)  # default gap=7.0 in 2.1.0
5. boundary_metrics(heuristic=possession_id, native=possession.astype(int64))
6. assert recall >= 0.83 AND precision >= 0.30
```

64-match HDF5 test (new):
```
1. sb_worldcup_data fixture provides HDFStore (session-scoped, 6-7 MB read once)
2. for match_id in 64-match list:
   a. actions = store.get(f"actions/game_{match_id}")
   b. non_synth = actions[actions["possession"].notna()].copy()
   c. add_possessions(non_synth)
   d. boundary_metrics(heuristic, native)
   e. assert recall >= 0.83 AND precision >= 0.30
```

## 7. Test plan + TDD ordering

1. **Unit tests for new parameters first** — write all `TestBriefOpposingMerge` + `TestDefensiveTransitions` + `TestMaxGapDefaultIs7Seconds` cases against not-yet-implemented signatures. All fail at `TypeError: unexpected keyword argument` or assertion. Confirms test wiring + serves as executable spec.

2. **Implement validation logic** — add validation for the 3 new params. Re-run unit tests → validation tests green, behavior tests still red.

3. **Implement boundary-helper extraction + Rule 2 (defensive)** — easiest to vectorize (np.isin + AND-out). Re-run unit tests → defensive tests green.

4. **Implement Rule 1 (brief-opposing-merge)** — vectorized look-ahead via shifted arrays + game/period boundary masks; row-loop fallback if vectorization is intractable. Both produce identical output (cross-checked in unit tests). Re-run unit tests → brief tests green.

5. **Change `max_gap_seconds` default to 7.0** — `add_possessions(actions)` now produces 2.1.0 behavior. Re-run unit tests → default-value tests green; existing tests that depended on the 5.0 default need review (audit during this step).

6. **Update build script** — adapter + conversion call. Run `python scripts/build_worldcup_fixture.py --verbose` to regenerate HDF5. Verify size, validate-output check passes.

7. **Update existing 3-JSON test** — drop recall threshold to 0.83. Tests still green; should pass with healthy margin since baseline measurements show `R_min=0.854` even at gap=7.0.

8. **Add 64-match HDF5 test** — parametrized over `store.get_storer("...")`-discovered keys. All 64 should pass at `recall>=0.83, precision>=0.30`.

9. **Mirror to atomic** — same param additions, same default change. Atomic-side regression coverage stays scoped to the new unit tests in this PR; an atomic StatsBomb-native regression test (3-JSON or HDF5) is a future PR if a consumer surfaces. Existing hand-built atomic fixtures continue to provide algorithm-level coverage.

10. **Docstring updates** in standard + atomic.

11. **CHANGELOG.md** with `## [2.1.0]` entry and Breaking Changes section.

12. **Version bump** + final-review skill + commit cycle.

## 8. Verification gates (before commit)

```bash
# Match exact CI pin (per feedback_ci_cross_version memory)
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113

# Lint + format
uv run ruff check silly_kicks/ tests/ scripts/
uv run ruff format --check silly_kicks/ tests/ scripts/

# Type-check
uv run pyright silly_kicks/

# Regenerate HDF5 (uses the 64 cached raw JSONs)
uv run python scripts/build_worldcup_fixture.py --verbose

# Full pytest suite
uv run pytest tests/ -v --tb=short
```

**Expected baseline:** all existing tests pass + new unit tests pass + 3-JSON parametrized tests pass at 0.83 / 0.30 + 64 new HDF5 parametrized tests pass at 0.83 / 0.30. Zero pyright errors. Zero ruff errors.

**`/final-review`** runs after the verification gates pass — per `feedback_commit_policy` memory it's mandatory before the single commit.

## 9. Commit cycle

Per `feedback_commit_policy` memory: literally one commit per branch, no WIP commits + squash, explicit user approval at every step.

Branch name: `feat/add-possessions-precision-improvement` (or short variant the user prefers).

```
1. All gates green + /final-review pass
2. User approves → git add + git commit -s (one commit)
3. User approves → git push -u origin feat/...
4. User approves → gh pr create
5. User approves → gh pr merge --admin --squash --delete-branch
6. User approves → git tag v2.1.0 + git push origin v2.1.0  # auto-fires PyPI publish
```

## 10. Out of scope

### Parked indefinitely (no follow-up queued)

**Spatial continuity rule.** Measurement showed +3pp precision at substantial recall cost (R_min=0.806 even at the smallest meaningful setting; below the 0.85 floor). This spec is the canonical reference; revisit only if a consumer surfaces with a use case that justifies the recall hit. No `TODO.md` entry.

### Future enhancements (deferred, no concrete consumer ask yet)

**`boundary_metrics` aggregate API.** A future `aggregate_boundary_metrics(matches: list[Series], natives: list[Series]) -> dict` returning mean / sd / min / max / percentiles across multiple matches would simplify downstream measurement campaigns. Not needed for PR-S12's CI gate (per-match independent gates suffice).

**Atomic-SPADL StatsBomb-native regression test.** 3-JSON or 64-match HDF5 regression on the atomic side. Future PR if a consumer surfaces.

### Cross-cutting still queued from PR-S8 era (in `TODO.md`)
- TODO.md D-8 — Docstring `Examples` sections (49 functions, zero examples).
- TODO.md A9 — Atomic VAEP coupling reduction.
- TODO.md unnumbered — `vaep/features.py` decomposition (809 lines).

## 11. Risks + mitigations

| Risk | Mitigation |
|---|---|
| 0.83 recall floor too tight on the 64-match HDF5 — `R_min=0.854` at gap=7.0 leaves 2.4pp margin, but pandas/numpy version drift between local 3.14 and CI 3.10/3.11/3.12 could shift | Pyright + pandas-stubs pin matches CI exactly. CI matrix runs on 3 ubuntu Python versions + 1 windows; if any matrix slot disagrees with the local measurement, that's the canary. Loosen to 0.80 if any flap is observed in the first 2 weeks. |
| Lakehouse pinned to 2.0.0 has implicit dependency on `max_gap_seconds=5.0` | CHANGELOG callout + lakehouse pin to `>=2.1.0,<3.0`. Lakehouse can revert per-call with `add_possessions(..., max_gap_seconds=5.0)` if needed (one-line opt-out). |
| Brief-merge vectorization edge cases at game/period boundaries | Comprehensive unit tests cover game-boundary, period-boundary, end-of-stream cases. Cross-check vectorized vs row-loop reference implementation in TDD step 4. If vectorization proves intractable, use the row-loop directly — n is small enough (~1500 rows per match) that O(n*N) is fine in Python. |
| HDF5 regeneration might fail if StatsBomb open-data URL is down | Cache hits from `tests/datasets/statsbomb/raw/.cache/events/` (64 raw JSONs already present locally). Build script designed for this case. |
| Existing tests depend on `max_gap_seconds=5.0` default — silent recall regression | Audit during TDD step 5: grep for `add_possessions(` call sites; any without explicit `max_gap_seconds` are now 7.0. Test outputs should be reviewed manually for behavior shifts. |
| `defensive_transition_types` validation against atomic vocabulary differs from standard | Mirror the validation logic in atomic helper; reference the atomic `actiontype_id` dict. Unit tests for both vocabularies. |
| Regression CI runtime grows from 3 fixtures (~0.1s) to 64+3 fixtures (~1-2s) | Acceptable. Total test runtime stays well under 30s on fastest CI matrix slot. The HDF5 cold-load is one session-scoped read. |

## 12. Acceptance criteria

1. `silly_kicks.spadl.add_possessions` accepts `merge_brief_opposing_actions`, `brief_window_seconds`, `defensive_transition_types` keyword-only parameters with documented semantics + recommended-settings guidance in the docstring.
2. `silly_kicks.atomic.spadl.add_possessions` accepts the same three parameters and the same default change.
3. `max_gap_seconds` default is 7.0 in both standard and atomic.
4. `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` contains the `possession` column on every `actions/game_<id>` table.
5. `tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBombNative` runs at `recall >= 0.83 AND precision >= 0.30`.
6. New `TestBoundaryAgainstStatsBomb64Match` runs 64 parametrized cases at the same gate (`recall >= 0.83 AND precision >= 0.30`).
7. Unit-test suite for the 3 new parameters (and the default change) is comprehensive: ~12-15 tests across 3 sub-classes, both standard and atomic.
8. `pyproject.toml` version is `2.1.0`. Tag `v2.1.0` pushed → PyPI auto-publish workflow fires.
9. `CHANGELOG.md` has a `## [2.1.0]` entry with a clear "Breaking" subsection citing the `max_gap_seconds` default change AND a "Behavior" section linking to the empirical baselines.
10. Verification gates (ruff, pyright with exact pins, full pytest suite, /final-review) all pass before commit.
11. `add_possessions` (both standard and atomic) docstrings updated with the new gap=7.0 baselines + recommended-setting tradeoff table for the 3 opt-in rules.
12. `TODO.md` PR-S12 entry closed (moved out of "Open PRs" table). Spatial-continuity rule is documented in this spec's §10 ("Out of scope, parked indefinitely") — no separate `TODO.md` Tech-Debt entry; spec is the canonical reference.
