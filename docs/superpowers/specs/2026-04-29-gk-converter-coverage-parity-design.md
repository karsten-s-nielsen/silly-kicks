# GK converter coverage parity — sportec + metrica fixes + coverage_metrics utility

**Status:** Approved (design)
**Target release:** silly-kicks 1.10.0
**Author:** Claude Opus 4.7 (1M)
**Date:** 2026-04-29
**Predecessor:** 1.9.0 (`docs/superpowers/specs/2026-04-29-worldcup-hdf5-e2e-prediction-tests-design.md`)
**Triggered by:** luxury-lakehouse PR-LL2 production deploy (2026-04-29) — post-deploy validation surfaced 100% NULL `gk_role` and `defending_gk_player_id` on IDSSE (2,522 rows) and Metrica (5,839 rows) sources

---

## 1. Problem

PR-LL2's production deploy ran `apply_spadl_enrichments` (calling silly-kicks 1.8.0's `add_gk_role` + `add_pre_shot_gk_context`) on all 4 SPADL sources. Validation discovered:

| Source | rows | gk_role NULL% | defending_gk_player_id NULL% (shots) |
|---|---|---|---|
| StatsBomb | 7,151,518 | non-NULL on GK actions ✓ | non-NULL on shots ✓ |
| Wyscout | 2,465,706 | non-NULL on GK actions ✓ | non-NULL on shots ✓ |
| **IDSSE** | **2,522** | **100% NULL** | **100% NULL on shots** |
| **Metrica** | **5,839** | **100% NULL** | **100% NULL on shots** |

`add_gk_role` and `add_pre_shot_gk_context` operate on `keeper_*` SPADL action types. They correctly emit NULL when no `keeper_*` actions are present in the input stream. So the gap is upstream — at the converter layer.

### 1.1 Empirical investigation (this session, 2026-04-29)

Read-only probes against `bronze.idsse_events`, `bronze.metrica_events`, and `bronze.spadl_actions` revealed **three distinct bugs**, more substantial than the original PR brief's framing.

#### Bug 1 — sportec.py `event_type` vocabulary mismatch (CRITICAL — beyond GK gap)

`silly_kicks/spadl/sportec.py:495` checks `is_pass = (et == "Pass")`. But DFL DataHub's actual event_type per spec and per production bronze is `"Play"`. The lakehouse adapter (`adapt_idsse_events_for_silly_kicks`) is a near-identity passthrough — it doesn't normalize the name.

**Net effect: ALL pass-class events drop to `non_action` and get filtered out.** Direct probe on production match `idsse_J03WMX`:

- Input: 1,715 events including 883 Play events
- Output: 387 SPADL actions, **zero `pass` (type_id 0) or `cross` (type_id 1)** — only tackles, throw-ins, fouls, shots, freekicks, goalkicks, and bad_touches

This isn't merely a GK gap — it's a 60-80% data integrity loss on every IDSSE match in production. The dominant action type in football (passing) has been completely missing from bronze.spadl_actions for IDSSE since silly-kicks 1.7.0 shipped sportec.py.

The bug was masked because:
- The kloppy Sportec test fixture (`tests/datasets/kloppy/sportec_events.xml`) has 29 events — too thin to exercise pass-mapping
- Existing tests assert structural shape, not coverage
- IDSSE wasn't in production until PR-LL2

#### Bug 2 — sportec.py `play_goal_keeper_action` qualifier vocabulary incomplete

`sportec.py:584-595` maps four qualifier values: `"save"` / `"claim"` / `"punch"` / `"pickUp"`. Production IDSSE bronze actually uses:

- `"throwOut"`: 49 occurrences
- `"punt"`: 25 occurrences

These are GK *distribution* actions (GK with ball in hand, throwing or kicking to a teammate). The full DFL vocabulary for `play_goal_keeper_action` includes both shot-stopping (save/claim/punch/pickUp) and distribution (throwOut/punt) values. silly-kicks only handles the former.

**Net effect: 74 IDSSE Play events with GK qualifiers are dropped to `non_action` (one match's data).**

#### Bug 3 — Metrica genuinely lacks keeper-event subtypes (DESIGN)

Production Metrica bronze taxonomy: `PASS / CARRY / RECOVERY-INTERCEPTION / BALL LOST / CHALLENGE / SET PIECE / SHOT` — **no SAVE / BLOCK / CLAIM / PUNCH / PICK-UP-style subtypes anywhere**.

The PR-LL2 brief hypothesized "map Metrica F24-style SAVE/CLAIM/PUNCH/PICK-UP subtypes" but this hypothesis was wrong. Metrica's event format does not natively mark GK actions in any way detectable from event subtype alone.

**Fix requires a different mechanism:** the Wyscout-1.0.0 pattern of `goalkeeper_ids: set | None = None` parameter for player-based routing, paired with conservative routing rules for Metrica event types when a known GK player_id matches.

### 1.2 The underlying test gap that allowed all three

silly-kicks's existing sportec/metrica tests use synthetic mini-fixtures (5-10 hand-crafted rows). The kloppy-derived fixtures (`tests/datasets/kloppy/sportec_events.xml` 29 events, `tests/datasets/kloppy/metrica_events.json` 3,620 events but 0 GK qualifiers) don't exercise:

- Pass-class events with realistic event_type names ("Pass" vs "Play" never tested with realistic input)
- GK qualifier values beyond the documented 4
- Keeper-action emission per converter on production-shape data

Tests assert structural shape (does the converter return a DataFrame with the right columns?) — not coverage (does it emit the expected action-type distribution from realistic input?). These bugs have shipped since silly-kicks 1.7.0 (sportec.py) / pre-1.0.0 (metrica.py) without surfacing.

PR-S10's secondary purpose: institutionalize a per-converter coverage-parity regression gate so this entire bug class can't recur silently.

## 2. Goals

1. **Bug 1 fix** — sportec.py recognizes DFL's `"Play"` event_type for pass-class events.
2. **Bug 2 fix** — sportec.py maps `throwOut` and `punt` qualifier values, synthesizing two SPADL actions per source event (keeper_pick_up + pass/goalkick) to match StatsBomb's per-event shape for the same semantic.
3. **Bug 3 fix** — metrica.py accepts `goalkeeper_ids: set[str] | None = None` parameter; with it, applies conservative routing rules to enable GK coverage despite Metrica's lack of native GK markers.
4. **API symmetry** — all 5 DataFrame converters accept `goalkeeper_ids`. Required-or-no-op semantics depend on whether the source format natively marks GK actions.
5. **Public `coverage_metrics` utility** — new `silly_kicks.spadl.coverage_metrics(*, actions, expected_action_types) -> CoverageMetrics` for downstream consumers and silly-kicks's own regression infrastructure.
6. **Production-shape vendored fixtures** — sportec + metrica regression coverage on real-data shapes.
7. **Cross-provider parity regression gate** — a parametrized test asserts every DataFrame converter emits keeper_* actions when given goalkeeper hints / qualifier-rich source data.
8. **Vocabulary documentation** — explicit DFL event_type + `play_goal_keeper_action` qualifier tables in sportec.py docstring; Metrica goalkeeper_ids contract in metrica.py docstring.

## 3. Non-goals

1. **No changes to `add_gk_role` / `add_pre_shot_gk_context` algorithms** — those work correctly when given a SPADL stream with proper keeper_* actions. After PR-S10's converter fixes, lakehouse will see immediate gk_role coverage. No enrichment-side change needed.
2. **No `add_possessions` algorithmic improvement** — that was the prior PR-S10 (added merge rules, max_gap_seconds tuning). PR-S11 now per re-numbering.
3. **No Wyscout `goalkeeper_ids` changes** — already correct from 1.0.0.
4. **No per-provider atomic-SPADL converter changes** — atomic uses post-conversion `convert_to_atomic(actions)` projection; no per-provider atomic logic exists.
5. **No atomic-SPADL `coverage_metrics` parity** — Atomic-SPADL has its own action vocabulary (33 types vs standard's 23). Add a TODO.md tech-debt entry. Wait for concrete consumer ask before shipping.
6. **No production-shape fixtures for statsbomb / opta / wyscout** — those converters already have CI coverage via PR-S8 (StatsBomb 3-fixture parametrized) and PR-S9 (WorldCup HDF5). Only sportec + metrica need new fixtures.
7. **No solving the "what bodypart for synthesized passes" question separately** — locked here: foot for punt, other (hands) for throwOut.
8. **No lakehouse-side adapter changes** — PR-LL3's territory after silly-kicks 1.10.0 ships.
9. **No touching luxury-lakehouse** — that repo is the other Claude session's responsibility.

## 4. Architecture

### 4.1 File structure

```
silly_kicks/spadl/sportec.py                      MOD  +100 / -10  Replace "Pass" with "Play"; throwOut/punt synthesis (keeper_pick_up + pass); add goalkeeper_ids parameter (supplementary); update docstring with DFL vocabulary tables
silly_kicks/spadl/metrica.py                      MOD  +80 / -5    Add goalkeeper_ids parameter; conservative GK routing (PASS by GK → synth, RECOVERY by GK → keeper_pick_up, AERIAL-WON by GK → keeper_claim); docstring limitation note
silly_kicks/spadl/statsbomb.py                    MOD  +5 / 0      Accept goalkeeper_ids: set[int] | None as no-op (API symmetry); document as no-op
silly_kicks/spadl/opta.py                         MOD  +5 / 0      Accept goalkeeper_ids: set[int] | None as no-op (API symmetry)
silly_kicks/spadl/wyscout.py                      (no change)      Already has goalkeeper_ids from 1.0.0
silly_kicks/spadl/utils.py                        MOD  +60 / 0     New CoverageMetrics TypedDict + coverage_metrics() public utility
silly_kicks/spadl/__init__.py                     MOD  +3 / 0      Re-export coverage_metrics + CoverageMetrics
tests/datasets/idsse/sample_match.parquet         NEW  ~30 KB      Production-shape IDSSE bronze (200-400 rows; includes throwOut/punt rows)
tests/datasets/idsse/README.md                    NEW  ~20 lines   DFL DataHub attribution + provenance
tests/datasets/metrica/sample_match.parquet       NEW  ~30 KB      Production-shape Metrica bronze (200-400 rows)
tests/datasets/metrica/README.md                  NEW  ~20 lines   Metrica open-data attribution + provenance
tests/spadl/test_sportec.py                       MOD  +200 / 0    TestSportecPlayEventRecognition + TestSportecGKQualifierSynthesis + TestSportecGoalkeeperIdsSupplementary + TestSportecCoverageParity
tests/spadl/test_metrica.py                       MOD  +150 / 0    TestMetricaGoalkeeperIdsRouting + TestMetricaCoverageParity
tests/spadl/test_statsbomb.py                     MOD  +20 / 0     TestStatsBombGoalkeeperIdsNoOp
tests/spadl/test_opta.py                          MOD  +20 / 0     TestOptaGoalkeeperIdsNoOp
tests/spadl/test_coverage_metrics.py              NEW  ~120 lines  TestCoverageMetricsContract / Correctness / Degenerate / Errors
tests/spadl/test_cross_provider_parity.py         NEW  ~80 lines   Cross-provider parametrized parity meta-test
scripts/extract_provider_fixtures.py              NEW  ~150 lines  Build script for vendored fixtures (Option B: public-source download from kloppy + DFL DataHub)
pyproject.toml                                    MOD  +1 / -1     Version 1.9.0 → 1.10.0
CHANGELOG.md                                      MOD  +80 lines   ## [1.10.0] entry
TODO.md                                           MOD  ~+5 / -3    Mark PR-S10 shipped; PR-S11 (was PR-S10) bumps in queue; atomic coverage_metrics tech debt
docs/c4/architecture.dsl + .html                  MOD  +1 / -1     Add coverage_metrics to spadl container public-helper enumeration
docs/superpowers/specs/2026-04-29-...-design.md   THIS FILE        Bundled into single commit
docs/superpowers/plans/2026-04-29-....md          NEW              Implementation plan, bundled into single commit
```

### 4.2 `coverage_metrics` public utility

```python
class CoverageMetrics(TypedDict):
    """Per-action-type coverage statistics for a SPADL action stream."""
    counts: dict[str, int]      # action_type_name -> row count
    missing: list[str]           # action types in expected_action_types but absent (sorted)
    total_actions: int


def coverage_metrics(
    *,
    actions: pd.DataFrame,
    expected_action_types: set[str] | None = None,
) -> CoverageMetrics:
    """Compute SPADL action-type coverage for an action DataFrame.

    Resolves ``type_id`` to action-type name via :func:`spadlconfig.actiontypes_df`.
    Counts each action type present; if ``expected_action_types`` is provided,
    reports any of those types with zero rows.

    Use cases:

    1. Test discipline — assert converter X emits action types Y on a fixture.
    2. Downstream validation — consumers calling silly-kicks-converted bronze
       data can verify expected coverage before downstream aggregation.
    3. CI regression gate — per-converter parametrized test asserts coverage
       parity across providers (added in 1.10.0).

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action stream. Must contain ``type_id``.
    expected_action_types : set[str], optional
        Action type names that should be present. Returned as ``missing`` if
        absent. ``None`` (default) skips the expectation check.

    Returns
    -------
    CoverageMetrics
        ``{"counts": {...}, "missing": [...], "total_actions": N}``.

    Raises
    ------
    ValueError
        If ``type_id`` column is missing.

    Examples
    --------
    Validate IDSSE bronze→SPADL output covers all expected action types::

        actions, _ = sportec.convert_to_actions(events, home_team_id="home",
                                                 goalkeeper_ids={"DFL-OBJ-..."})
        m = coverage_metrics(
            actions=actions,
            expected_action_types={"pass", "shot", "tackle", "keeper_pick_up"},
        )
        assert not m["missing"], f"Missing action types: {m['missing']}"
    """
```

**Behavior contract:**

- Empty `actions` → `counts={}`, `missing=sorted(expected_action_types or [])`, `total_actions=0`
- Missing `type_id` column → `ValueError`
- Action type IDs not in spadlconfig → reported under name `"unknown"`; doesn't raise
- `expected_action_types=None` → `missing=[]`

**Test discipline** (mirrors PR-S8 `TestBoundaryMetrics`):

```
TestCoverageMetrics
├── TestCoverageMetricsContract
│   ├── test_returns_dict_with_required_keys
│   ├── test_counts_value_type_is_int
│   ├── test_keyword_only_args_required
│   └── test_missing_type_id_column_raises_value_error
├── TestCoverageMetricsCorrectness
│   ├── test_single_type_dataframe_counts_correctly
│   ├── test_multi_type_dataframe_counts_correctly
│   ├── test_expected_fully_present_missing_empty
│   ├── test_expected_partially_absent_missing_sorted
│   └── test_unknown_type_id_reported_as_unknown
├── TestCoverageMetricsDegenerate
│   ├── test_empty_dataframe_returns_zeros
│   └── test_expected_action_types_none_returns_empty_missing
└── (no Errors sub-class beyond Contract; unified)
```

**Re-export from `silly_kicks.spadl.__init__`**: alphabetically inserted in `__all__` and import block.

### 4.3 sportec.py Bug 1 fix — Pass → Play

Single-line replacement at `silly_kicks/spadl/sportec.py:495`:

```diff
-    is_pass = et == "Pass"
+    is_pass = et == "Play"
```

**No backward-compatibility concern.** Per PR-S7's spec § 5.3, the `"Pass"` string was based on an implementation-time mental model, not the DFL spec. Per `feedback_commit_policy` memory + Hyrum's Law analysis: zero current consumers passed `event_type="Pass"` (lakehouse uses `"Play"`, kloppy gateway uses different code path). Replacing is safe.

### 4.4 sportec.py Bug 2 fix — throwOut / punt synthesis

**Current behavior** (`sportec.py:583-596`):

```python
is_play = et == "Play"  # after Bug 1 fix
play_gk = _opt("play_goal_keeper_action", "").fillna("").astype(str).to_numpy()
play_gk_save = is_play & (play_gk == "save")
play_gk_claim = is_play & (play_gk == "claim")
play_gk_punch = is_play & (play_gk == "punch")
play_gk_pickup = is_play & (play_gk == "pickUp")
type_ids[play_gk_save] = spadlconfig.actiontype_id["keeper_save"]
# ... etc
is_play_no_action = is_play & ~(play_gk_save | play_gk_claim | play_gk_punch | play_gk_pickup)
type_ids[is_play_no_action] = spadlconfig.actiontype_id["non_action"]
```

**New behavior**: extend the qualifier vocabulary AND synthesize two actions for the distribution variants.

```python
# Shot-stopping qualifiers — single keeper_* action per source event
play_gk_save = is_play & (play_gk == "save")
play_gk_claim = is_play & (play_gk == "claim")
play_gk_punch = is_play & (play_gk == "punch")
play_gk_pickup = is_play & (play_gk == "pickUp")

# Distribution qualifiers — synthesize keeper_pick_up + pass/goalkick per source event
play_gk_throwout = is_play & (play_gk == "throwOut")
play_gk_punt = is_play & (play_gk == "punt")
play_gk_distribution = play_gk_throwout | play_gk_punt

# Phase 1: source rows get keeper_pick_up (the "GK had ball" half)
type_ids[play_gk_save] = spadlconfig.actiontype_id["keeper_save"]
type_ids[play_gk_claim] = spadlconfig.actiontype_id["keeper_claim"]
type_ids[play_gk_punch] = spadlconfig.actiontype_id["keeper_punch"]
type_ids[play_gk_pickup] = spadlconfig.actiontype_id["keeper_pick_up"]
type_ids[play_gk_distribution] = spadlconfig.actiontype_id["keeper_pick_up"]
result_ids[is_play] = spadlconfig.result_id["success"]

# Phase 2 (after main DataFrame assembly): for each play_gk_distribution row,
# synthesize an additional pass/goalkick action immediately after.
# Synthesis logic: see _add_synthetic_gk_distribution_actions below.

is_play_no_action = is_play & ~(
    play_gk_save | play_gk_claim | play_gk_punch | play_gk_pickup | play_gk_distribution
)
type_ids[is_play_no_action] = spadlconfig.actiontype_id["non_action"]
```

The synthesis function (new helper, conceptually similar to `_add_dribbles`):

```python
def _add_synthetic_gk_distribution_actions(
    actions: pd.DataFrame,
    is_distribution_mask: np.ndarray,
    qualifier_values: np.ndarray,
    preserve_native: list[str] | None,
) -> pd.DataFrame:
    """For each row matching is_distribution_mask, insert a synthetic pass action
    immediately after representing the GK's distribution to a teammate.

    - Synthetic action type: pass for throwOut, goalkick for punt
    - Synthetic bodypart: other for throwOut (hands), foot for punt
    - Synthetic time/coords: same as the keeper_pick_up action
    - Synthetic player_id: same GK player (the throw/punt is BY the GK)
    - preserve_native columns: copied from the source row (matches _add_dribbles pattern)
    - action_id: renumbered after insertion (existing pattern)
    """
```

**Result**: 1 source `play_goal_keeper_action="throwOut"` event → 2 SPADL actions (keeper_pick_up + pass). Action_id renumbering brings the synthetic pass right after.

**TDD test cases** (TestSportecGKQualifierSynthesis):

- `test_throwout_synthesizes_keeper_pick_up_plus_pass` — fixture has N throwOut rows → output has N keeper_pick_up + N pass (player_id matches)
- `test_punt_synthesizes_keeper_pick_up_plus_goalkick` — fixture has M punt rows → output has M keeper_pick_up + M goalkick
- `test_throwout_pass_has_other_bodypart` — synthesized pass uses bodypart=other (hands)
- `test_punt_goalkick_has_foot_bodypart` — synthesized goalkick uses bodypart=foot
- `test_action_ids_renumbered_after_synthesis` — output action_ids are 0..N-1 with no gaps
- `test_preserve_native_propagates_to_both_synthetic_actions` — both rows get the source's preserved column values

### 4.5 sportec.py — `goalkeeper_ids` supplementary signal

```python
def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: str,
    *,
    preserve_native: list[str] | None = None,
    goalkeeper_ids: set[str] | None = None,  # NEW
) -> tuple[pd.DataFrame, ConversionReport]:
```

**Behavior:** if `goalkeeper_ids` is provided AND a Play event has `player_id` in goalkeeper_ids AND has NO explicit `play_goal_keeper_action` qualifier set → route to keeper_pick_up + pass synthesis (same as throwOut/punt). This catches GK distribution that DFL didn't annotate.

This is a *supplementary* signal because sportec's qualifier-driven mapping is the primary contract.

### 4.6 metrica.py — `goalkeeper_ids` primary mechanism

```python
def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: str,
    *,
    preserve_native: list[str] | None = None,
    goalkeeper_ids: set[str] | None = None,  # NEW
) -> tuple[pd.DataFrame, ConversionReport]:
```

**Routing rules when `goalkeeper_ids` is provided:**

| Source event | Routing |
|---|---|
| `PASS` (any subtype) by GK | Synthesize **keeper_pick_up + pass** (Q3 pattern: GK had ball, then distributed) |
| `RECOVERY` (any subtype: NULL, INTERCEPTION, THEFT) by GK | **keeper_pick_up** (GK gained the ball; source doesn't disambiguate save vs pickup) |
| `CHALLENGE` with subtype exactly `"AERIAL-WON"` by GK | **keeper_claim** (specific aerial-won-by-GK semantics) |
| `CHALLENGE` with subtype `"AERIAL-LOST"` or other AERIAL variants | unchanged (default routing — GK lost the duel, not a claim) |
| All other Metrica event types by GK | unchanged (default routing) |

**Behavior without `goalkeeper_ids`:**

- Metrica output has zero keeper_* actions (current 1.9.0 behavior preserved as default — no breaking change for callers who haven't migrated)
- Documented limitation in converter docstring with explicit "to enable GK coverage, pass goalkeeper_ids from match metadata"

### 4.7 statsbomb.py + opta.py — API symmetry no-op

```python
def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: int,
    *,
    preserve_native: list[str] | None = None,
    goalkeeper_ids: set[int] | None = None,  # NEW — accepted, no-op
    # ... existing params unchanged ...
) -> tuple[pd.DataFrame, ConversionReport]:
```

**Behavior:** parameter silently accepted; no effect on output (StatsBomb/Opta source events natively mark GK actions, so `goalkeeper_ids` is supplementary and currently unused). Tests assert output is byte-for-byte identical with and without the parameter. Documented in the docstring as no-op.

### 4.8 Cross-provider parity meta-test

```python
@pytest.mark.parametrize("provider,fixture_path,goalkeeper_ids,expected_keeper_types", [
    ("statsbomb", "tests/datasets/statsbomb/raw/events/7298.json", None,
     ["keeper_save", "keeper_claim", "keeper_punch", "keeper_pick_up"]),
    ("sportec", "tests/datasets/idsse/sample_match.parquet", {"DFL-OBJ-..."},
     ["keeper_pick_up"]),
    ("metrica", "tests/datasets/metrica/sample_match.parquet", {"P3577"},
     ["keeper_pick_up"]),
])
def test_converter_emits_keeper_actions(provider, fixture_path, goalkeeper_ids, expected_keeper_types):
    """Every DataFrame converter must emit at least one keeper_* action when
    given a fixture exercising GK paths (with goalkeeper_ids where source
    format requires it)."""
    actions = ... # load fixture, run appropriate converter with goalkeeper_ids
    m = coverage_metrics(
        actions=actions,
        expected_action_types=set(expected_keeper_types),
    )
    assert any(m["counts"].get(t, 0) > 0 for t in expected_keeper_types), (
        f"Provider {provider} emits no keeper_* actions on fixture {fixture_path}; "
        f"counts={m['counts']}"
    )
```

This is the regression gate that would have caught Bugs 1-3 in 1.7.0 / earlier had it existed.

## 5. Production-shape fixture sourcing

**Recommended approach: Option B — public-source download script.**

```bash
# scripts/extract_provider_fixtures.py
#   Downloads:
#   - DFL DataHub free-sample match (single XML, ~1500-2000 events with throwOut/punt rows)
#     URL: https://github.com/dfl-eng/datahub-public-data/... (or kloppy's vendored Sportec sample,
#     same source as tests/datasets/kloppy/sportec_events.xml; need a richer match)
#   - Metrica Sample_Game_2 events.json (3,620 events; same source as kloppy fixture)
#   Writes:
#   - tests/datasets/idsse/sample_match.parquet (subset of ~200-400 rows including
#     keeper-relevant rows; converted to bronze-shape via lakehouse adapter logic)
#   - tests/datasets/metrica/sample_match.parquet (similar)
#   Cache: tests/datasets/{idsse,metrica}/.cache/ (gitignored)
```

If Option B can't yield a DFL match with throwOut/punt qualifiers from the public repo, fall back to **Option A** (one-time extraction from production bronze.idsse_events via Databricks SQL with env-var auth). Document the source match_id in the fixture README. This is acceptable because:
- DFL identifiers are public competition IDs (no PII)
- DFL DataHub free-sample license permits non-commercial redistribution
- Precedent: PR-S8's StatsBomb 3-fixture vendoring follows the same pattern

In either case, the fixture file is a SUBSET (~200-400 rows) — not the full bronze table. Small enough to commit (~30 KB compressed parquet); large enough to exercise the relevant code paths.

## 6. Test plan + TDD ordering

(See Section D in brainstorming for the 12-task ordering. Detailed test code per task is in the implementation plan, not this spec.)

Summary:

```
0. Pre-flight (clean baseline at 1.9.0; branch; CI pin install; baseline pytest + lint)
1. coverage_metrics utility — TDD: write tests, implement, re-export, verify
2. Vendor production-shape fixtures (extract via build script, commit)
3. sportec.py Bug 1 — Pass → Play (TDD)
4. sportec.py Bug 2 — throwOut/punt synthesis (TDD)
5. sportec.py — goalkeeper_ids supplementary signal (TDD)
6. metrica.py Bug 3 — goalkeeper_ids routing (TDD)
7. statsbomb.py + opta.py — API symmetry no-op (TDD)
8. Cross-provider parity meta-test
9. Documentation (vocabulary tables, docstrings)
10. Release artifacts (CHANGELOG, TODO, version, C4)
11. Verification gates (ruff/format/pyright/pytest with CI pins)
12. /final-review + 5 user-gated steps (commit/push/PR/merge/tag)
```

## 7. Verification gates

```bash
# CI pin (per feedback_ci_cross_version memory)
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113

uv run ruff check silly_kicks/ tests/ scripts/
uv run ruff format --check silly_kicks/ tests/ scripts/
uv run pyright silly_kicks/
uv run pytest tests/ -v --tb=short
```

**Expected post-PR-S10 baseline:** ~620 passed (560 baseline + ~60 new), ~4 skipped. Pyright + ruff zero errors.

## 8. Commit cycle

Per `feedback_commit_policy` memory + narrowed hook (only `git commit` sentinel-gated):

```
1. /final-review pass + all gates green
2. User approves → sentinel touch → git add + git commit (ONE commit, branch
   feat/gk-converter-coverage-parity)
3. User approves → git push -u origin feat/...
4. User approves → gh pr create
5. CI green → user approves → gh pr merge --admin --squash --delete-branch
6. User approves → git tag v1.10.0 + git push origin v1.10.0  # PyPI auto-publish
```

## 9. Risks + mitigations

| Risk | Mitigation |
|---|---|
| **Bug 1 fix might break the kloppy-routed Sportec path** if anything in `silly_kicks/spadl/kloppy.py` indirectly calls into sportec.py's logic | Verified: kloppy.py has its own complete dispatch (`_parse_pass_event`, `_parse_goalkeeper_event` etc); sportec.py is dedicated DataFrame path only. No shared code. |
| **Action_id renumbering after synthesis might break consumers** that expect dense action_ids | Tested via `_add_dribbles` precedent — same pattern used since 1.0.0. Action_ids are renumbered to 0..N-1 dense after all synthesis. |
| **goalkeeper_ids on statsbomb/opta no-op behavior might drift** as the param gets used elsewhere | Explicit unit tests assert byte-for-byte output identity with and without the parameter. Drift would fail. |
| **Synthesizing 2 actions from 1 source event might inflate row counts unexpectedly** | Documented in CHANGELOG as a behavior change. Lakehouse PR-LL3 needs to be aware that IDSSE's row count will go up. The increase is small (~74 rows per match for typical IDSSE) but real. |
| **Public-source DFL DataHub fixture might not have throwOut/punt qualifiers** | If extraction Option B can't satisfy, fall back to Option A (one-shot Databricks pull). Document in fixture README. |
| **Pyright might complain about the synthesis logic** if action_ids type narrows tightly | Rely on existing `_add_dribbles` pattern; pyright was OK with that since 1.0.0. If issues surface, address inline (typed numpy arrays + explicit casts as needed). |
| **PR-S10 is large** — 60+ new tests, 5 file fixture additions, 4 modified converter files | Same pattern as PR-S9; single-commit ritual handles it. /final-review catches integration issues. The TDD ordering ensures incremental verification. |

## 10. Out of scope (queued follow-ups)

### PR-S11 — `add_possessions` algorithmic precision improvement (re-numbered)

Was PR-S10. Bumps to PR-S11 because PR-S10 is now the GK coverage work. Unchanged design.

### Atomic-SPADL `coverage_metrics` parity

Add `silly_kicks.atomic.spadl.coverage_metrics` with atomic action vocabulary (33 types vs standard 23). Same shape as standard. Wait for concrete consumer ask. Tracked in `TODO.md ## Tech Debt`.

## 11. Acceptance criteria

1. `silly_kicks.spadl.sportec.convert_to_actions` accepts and recognizes DFL `event_type = "Play"` for pass-class events; emits `pass` and `cross` SPADL actions.
2. `silly_kicks.spadl.sportec.convert_to_actions` maps `play_goal_keeper_action` qualifier values `throwOut` and `punt` by synthesizing **two SPADL actions per source event** (keeper_pick_up + pass for throwOut, keeper_pick_up + goalkick for punt). Action_ids renumbered. preserve_native propagates to both.
3. `silly_kicks.spadl.metrica.convert_to_actions` accepts `goalkeeper_ids: set[str] | None = None`. With it: PASS by GK → synth keeper_pick_up + pass; RECOVERY by GK → keeper_pick_up; CHALLENGE-AERIAL-WON by GK → keeper_claim. Without it: 0 keeper_* actions (default behavior preserved).
4. `silly_kicks.spadl.statsbomb.convert_to_actions` and `silly_kicks.spadl.opta.convert_to_actions` accept `goalkeeper_ids` parameter as a no-op (byte-for-byte identical output).
5. `silly_kicks.spadl.coverage_metrics(*, actions, expected_action_types) -> CoverageMetrics` is public, importable from `silly_kicks.spadl`, returns a `CoverageMetrics` TypedDict, raises `ValueError` on missing `type_id` column.
6. `tests/datasets/idsse/sample_match.parquet` and `tests/datasets/metrica/sample_match.parquet` are committed with proper attribution READMEs.
7. The cross-provider parity meta-test (`tests/spadl/test_cross_provider_parity.py`) asserts every DataFrame converter (with appropriate goalkeeper_ids) emits at least one keeper_* action.
8. `pyproject.toml` version is `1.10.0`. Tag `v1.10.0` is pushed → PyPI auto-publish workflow succeeds.
9. `CHANGELOG.md` has a new `## [1.10.0] — 2026-04-29` entry summarizing all 3 bug fixes + new feature.
10. `TODO.md` reflects PR-S10 shipped + PR-S11 in queue.
11. Verification gates (ruff/format/pyright/pytest) all pass before commit. /final-review passes.
12. CI matrix on PR (lint + ubuntu 3.10/3.11/3.12 + windows 3.12) all green before merge.
13. Lakehouse PR-LL3 (separate session, separate cycle) can bump `silly-kicks>=1.10.0,<2.0` and re-run `apply_spadl_enrichments` against IDSSE + Metrica with non-NULL `gk_role` + `defending_gk_player_id` coverage. *(Acceptance is on lakehouse side; silly-kicks side requires only that the new converter behavior support this.)*
