# Recall-based `add_possessions` validation + public `boundary_metrics` utility

**Status:** Approved (design)
**Target release:** silly-kicks 1.8.0
**Author:** Claude Opus 4.7 (1M)
**Date:** 2026-04-29
**Predecessor:** 1.7.0 (`docs/superpowers/specs/2026-04-28-dataframe-converters-sportec-metrica-design.md`)

---

## 1. Problem

`silly_kicks.spadl.add_possessions` shipped in 1.2.0 with an e2e test (`tests/spadl/test_add_possessions.py::TestBoundaryF1AgainstStatsBombNative`) that asserts boundary-F1 ≥ 0.80 against StatsBomb's native `possession` field. Three things are wrong with this:

1. **The fixture is not committed to the repo.** The test is `@pytest.mark.e2e` and has been silently skipping in every CI run since 1.2.0 — and it's also been skipping locally on the user's only development machine because no one ever placed `tests/datasets/statsbomb/raw/events/7584.json` there. Net effect: the existing regression infrastructure has produced zero coverage for ~6 release cycles.

2. **The 0.80 boundary-F1 threshold is misleading.** A measurement campaign during the luxury-lakehouse PR-LL2 cycle (driven by silly-kicks's own `preserve_native=['possession']` API shipped in 1.1.0) produced empirical baselines across 3 diverse open-data matches:

   | Match | Competition | Recall | Precision | F1 |
   |---|---|---|---|---|
   | 7298 | Women's World Cup | ~0.94 | ~0.45 | 0.608 |
   | 7584 | Champions League | ~0.93 | ~0.43 | 0.588 |
   | 3754058 | Premier League | ~0.93 | ~0.39 | 0.544 |

   The `add_possessions` docstring claims "boundary-F1 ~0.90 against StatsBomb's native possession_id" — 30+ percentage points higher than measured. Published "0.85-0.95 F1" baselines exist for related heuristic methods in the literature, but those use looser boundary-matching criteria or different ground-truth annotations than StatsBomb's open-data possession_id.

3. **F1 is the wrong metric to gate on.** F1 conflates recall (~0.93 — the heuristic catches every real possession boundary) and precision (~0.42 — the heuristic emits ~2× more boundaries than StatsBomb because it can't structurally replicate StatsBomb's "merge brief opposing-team actions" rule). The recall is what downstream consumers actually need from a possession-detection helper. The precision gap is intrinsic to the team-change-with-carve-outs algorithm class, not a defect.

## 2. Goals

1. **Honest documentation.** Update `add_possessions` docstring with the empirical baselines (recall ~0.93, precision ~0.42, F1 ~0.58). Replace the "F1 ~0.90" claim entirely.
2. **Real CI regression coverage.** Vendor 3 raw StatsBomb open-data event JSONs (matches 7298, 7584, 3754058) to `tests/datasets/statsbomb/raw/events/`, drop the `@pytest.mark.e2e` marker on the rewritten test, run on every PR.
3. **Recall + precision dual-gate.** Replace the F1-only gate with `recall ≥ 0.85 AND precision ≥ 0.30` per match. Recall gate enforces the helper's primary contract ("catches every real boundary"); precision floor catches the "boundary cardinality halved or doubled" regression class that affects per-possession aggregation downstream.
4. **Promote `boundary_metrics` to public API.** Move the boundary-metric helper from the test file into `silly_kicks/spadl/utils.py`, return a `BoundaryMetrics` TypedDict, enforce keyword-only args (precision/recall are asymmetric — swapping inputs is a silent bug), re-export from `silly_kicks.spadl.__init__`. Downstream consumers can compute the same metrics on their own data.
5. **Save follow-up PR context.** Two larger pieces of work surfaced during brainstorming and were explicitly deferred: PR-S9 (e2e prediction tests in CI via WorldCup HDF5 generation) and PR-S10 (algorithmic precision improvement). Track in `TODO.md` + auto-memory so they survive the session.

## 3. Non-goals

1. **No changes to `add_possessions`'s algorithm.** Documentation + test scaffolding only. The empirical numbers reflect the current implementation; if we change the algorithm later (PR-S10), the numbers shift and so do the gates.
2. **No new dependencies.** `TypedDict` is in stdlib `typing` since Python 3.8; no `typing_extensions` import needed.
3. **No atomic-SPADL parity for `boundary_metrics`.** Atomic's `add_possessions` (shipped 1.5.0) has its own boundary semantics; if a consumer needs an atomic-side regression gate, that's a follow-up PR with its own brainstorming.
4. **No `add_possessions_strict` variant.** Suggested in the brief as a research direction; explicitly out of scope here. Goes with PR-S10's algorithm work.
5. **No prediction-test e2e CI coverage.** The 5 `test_predict*` cases (`tests/vaep/test_vaep.py:72,82`, `tests/test_xthreat.py:219,229`, `tests/atomic/test_atomic_vaep.py:24`) currently silently skip everywhere because `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` doesn't exist on this machine. Closing that gap requires generating the HDF5 from open-data raw events (~1-2 hours of careful work) and is queued as PR-S9. PR-S8 stays focused.
6. **No parameter-tuning of `add_possessions`'s `max_gap_seconds` default.** LL2's parameter sweep showed F1 peaked at 10.0 vs the current default 5.0, but only F1 was reported (not recall/precision split). PR-S10 should re-measure across the full parameter grid using the new `boundary_metrics` utility before changing the default.
7. **No touching luxury-lakehouse.** That repo is the other Claude session's responsibility.

## 4. Architecture

### 4.1 File structure

```
silly_kicks/spadl/utils.py                                        MOD  +60 / -5    add boundary_metrics + BoundaryMetrics TypedDict; rewrite add_possessions docstring
silly_kicks/spadl/__init__.py                                     MOD  +4 / 0     re-export boundary_metrics + BoundaryMetrics
tests/spadl/test_add_possessions.py                               MOD  +120 / -90 remove _boundary_f1; rename + parametrize e2e class; add TestBoundaryMetrics unit-test class
tests/datasets/statsbomb/raw/events/7298.json                     NEW  ~600 KB   vendored open-data fixture
tests/datasets/statsbomb/raw/events/7584.json                     NEW  ~600 KB   vendored open-data fixture
tests/datasets/statsbomb/raw/events/3754058.json                  NEW  ~600 KB   vendored open-data fixture
tests/datasets/statsbomb/README.md                                NEW  ~20 lines  StatsBomb open-data attribution + license
pyproject.toml                                                    MOD  +1 / -1    version 1.7.0 → 1.8.0
CHANGELOG.md                                                      MOD  +35 / 0    ## [1.8.0] entry
TODO.md                                                           MOD  +12 / 0    PR-S9 + PR-S10 entries under new "## Open PRs" section
CLAUDE.md                                                         MOD  +1 / 0     (optional) tighten "Testing" section: committed-fixture tests should not be marked e2e
```

### 4.2 `boundary_metrics` public API

```python
from typing import TypedDict


class BoundaryMetrics(TypedDict):
    precision: float
    recall: float
    f1: float


def boundary_metrics(
    *,
    heuristic: pd.Series,
    native: pd.Series,
) -> BoundaryMetrics:
    ...
```

**Design choices:**

- **Return type is a `TypedDict`, not `dict[str, float]`.** Dict-shaped at runtime (zero overhead, JSON-serializable, dict-style access via `m["recall"]`), but pyright statically validates key names. Caught typos like `m["recal"]` at edit time.
- **Args are keyword-only.** Precision and recall are asymmetric — swapping `heuristic` and `native` swaps `FP` and `FN`, which swaps precision and recall. Both args are `pd.Series`, so static type-checking can't catch this. Keyword-only args force explicit naming at the call site, eliminating the silent footgun.
- **Re-exported from `silly_kicks.spadl.__init__`.** Symmetric with `add_possessions` and the rest of the public post-conversion helper family. Consumers do `from silly_kicks.spadl import add_possessions, boundary_metrics, BoundaryMetrics`.
- **Implementation moves directly from the test file.** The math is unchanged (TP/FP/FN counts on shifted boolean change-detection). Only the wrapping changes: dict instead of float-only return, keyword-only args, length-mismatch input validation.
- **Length-mismatch handling.** Public API should defensively check `len(heuristic) == len(native)` and raise `ValueError` with a clear message. The internal version trusted same-DataFrame-derived inputs; the public version can't.
- **Degenerate-case handling.** When TP+FP=0 (no boundaries in `heuristic`) or TP+FN=0 (no boundaries in `native`) or TP+FP+FN=0 (no boundaries anywhere), the corresponding metric is 0.0. Single-row inputs and empty inputs both return all-zero metrics without erroring.

### 4.3 `add_possessions` docstring update

Find this passage (`silly_kicks/spadl/utils.py:580-582` — three lines):

```rst
The carve-out is approximate (StatsBomb's proprietary possession rules
capture additional context), but matches typical published heuristics
at boundary-F1 ~0.90 against StatsBomb's native possession_id.
```

Replace with:

```rst
The carve-out is approximate (StatsBomb's proprietary possession rules
capture additional context like merging brief opposing-team actions
back into the containing possession). Empirically against StatsBomb
open-data the heuristic achieves:

    - Boundary recall: ~0.93 — every real possession boundary is detected.
    - Boundary precision: ~0.42 — the heuristic emits ~2× more boundaries
      than StatsBomb's native annotation, since it can't replicate the
      "merge brief opposing actions" rule structurally.
    - Boundary F1: ~0.58 (peak ~0.605 at max_gap_seconds=10.0).

Recall is the meaningful metric for downstream consumers — possessions
detected by the heuristic correspond to real possession changes. The
precision gap reflects the algorithm class, not a defect. Consumers
needing strict StatsBomb-equivalent semantics should use the native
possession_id where available; the heuristic is a possession proxy
for sources without one (Wyscout, Sportec, Metrica, etc.).

Published "0.85-0.95 F1" baselines exist for related heuristic methods
in the literature, but use looser boundary-matching criteria or
different ground-truth annotations than StatsBomb's open-data
possession_id.
```

### 4.4 e2e regression test rewrite

Class rename: `TestBoundaryF1AgainstStatsBombNative` → `TestBoundaryAgainstStatsBombNative`. Grep-verified zero other references in the codebase (only `tests/spadl/test_add_possessions.py` mentions any of `_boundary_f1` / `TestBoundaryF1` / `boundary_f1`).

Marker change: drop `@pytest.mark.e2e`. Per CLAUDE.md the marker means "requires uncommitted fixtures"; once the 3 raw-events JSONs are in the repo the marker is incorrect and would silently skip the test in CI.

Test method: parametrized over the 3 fixtures with per-match independent gates.

```python
class TestBoundaryAgainstStatsBombNative:
    """Validate add_possessions against StatsBomb's native possession_id.

    Empirically against StatsBomb open-data, this heuristic achieves
    boundary recall ~0.93 and boundary F1 ~0.58. The precision gap is
    intrinsic to the algorithm class. The CI gate below tests recall
    AND precision because both are observable behaviors that downstream
    consumers can develop dependencies on — F1 conflates two signals
    with very different magnitudes and is recorded for diagnostics only.
    """

    @pytest.mark.parametrize("match_id", [7298, 7584, 3754058])
    def test_boundary_metrics_against_native_possession_id(self, match_id):
        # ... load fixture, adapt to silly-kicks input shape, convert,
        #     drop synthetic dribbles, add_possessions, compute metrics ...
        m = boundary_metrics(
            heuristic=non_synthetic["possession_id"],
            native=non_synthetic["possession"].astype(np.int64),
        )

        assert m["recall"] >= 0.85 and m["precision"] >= 0.30, (
            f"match {match_id}: recall={m['recall']:.4f} "
            f"precision={m['precision']:.4f} f1={m['f1']:.4f}; "
            f"thresholds: recall>=0.85 precision>=0.30"
        )
```

**Threshold choices:**

- Recall ≥ 0.85: 8pp margin below the worst-observed 0.93. Tight enough to catch real regressions, loose enough to absorb match-to-match variance and minor implementation tweaks.
- Precision ≥ 0.30: 9pp margin below the worst-observed 0.39 (match 3754058). Tightest of the three but defensible — a heuristic that claims consistent cross-competition behavior should have a tight cross-competition floor.
- F1 in assert message only: F1 conflates two independent signals; gating on it would re-introduce the misrepresentation problem the docstring rewrite is trying to fix. Recording F1 in the failure message gives a human debugging the failure all three numbers without conflating them in the gate.

### 4.5 `TestBoundaryMetrics` unit-test class

Mirrors the file's existing test discipline (4 nested classes, ~10 small tests):

```
TestBoundaryMetrics
├── TestBoundaryMetricsContract
│   ├── test_returns_typeddict_with_required_keys
│   ├── test_all_metric_values_are_floats
│   └── test_keyword_only_args_required          # boundary_metrics(s1, s2) → TypeError
├── TestBoundaryMetricsCorrectness
│   ├── test_identical_sequences_all_metrics_one
│   ├── test_relabeled_identical_sequences_all_metrics_one  # counter-relabeling invariance
│   ├── test_completely_disjoint_boundaries_all_zero
│   └── test_partial_overlap_hand_computed_values
├── TestBoundaryMetricsDegenerate
│   ├── test_empty_sequences_returns_zeros
│   ├── test_single_row_returns_zeros            # no boundaries possible
│   └── test_constant_sequences_returns_zeros    # no boundaries in either
└── TestBoundaryMetricsErrors
    └── test_length_mismatch_raises_value_error
```

## 5. Pipeline (e2e regression test, end-to-end)

```
1. Load raw event JSON: tests/datasets/statsbomb/raw/events/<match_id>.json
2. Adapt to silly-kicks input shape (top-level keys → typed columns
   game_id / event_id / period_id / timestamp / team_id / player_id /
   type_name / location / extra / possession passthrough)
3. statsbomb.convert_to_actions(adapted, home_team_id=<first_non_null>,
                                preserve_native=["possession"])
4. Keep only non-synthetic rows: actions[actions["possession"].notna()].copy()
5. add_possessions(non_synthetic) → adds possession_id column
6. m = boundary_metrics(
       heuristic=non_synthetic["possession_id"],
       native=non_synthetic["possession"].astype(np.int64),
   )
7. assert m["recall"] >= 0.85 and m["precision"] >= 0.30
```

Step 4 is critical: synthetic dribble actions inserted by `_add_dribbles` have `possession=NaN` (no source event to inherit from). The `.notna()` filter keeps only rows that have a native possession-id to compare against; the trailing `.copy()` ensures the subsequent `add_possessions` call doesn't trigger pandas SettingWithCopyWarning when it adds the new `possession_id` column.

## 6. Fixture vendoring

Three matches from StatsBomb's open-data repo, downloaded once into `tests/datasets/statsbomb/raw/events/`:

```
7298.json     Women's World Cup 2019 — generally diverse possession patterns
7584.json     Champions League 2018-19  — high-tempo elite play
3754058.json  Premier League 2020-21    — wide style range
```

URL pattern: `https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/<match_id>.json`.

Each file is ~600 KB. Total ~1.8 MB committed to the repo, located under `tests/` so excluded from the wheel by `[tool.hatch.build.targets.wheel] packages = ["silly_kicks"]`.

`tests/datasets/statsbomb/README.md` content:

```markdown
# StatsBomb open-data fixtures

Vendored from https://github.com/statsbomb/open-data under the StatsBomb
Public Data License (non-commercial). See
https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf for the
full license text.

Used for offline e2e validation of silly-kicks's SPADL converters and
post-conversion enrichments. The three matches under raw/events/ are
the same ones measured during the luxury-lakehouse PR-LL2 boundary-
metrics campaign that produced the empirical baselines published in
silly_kicks.spadl.add_possessions's docstring.
```

## 7. Test plan + TDD ordering

(Detailed sequencing — each step's outcome verified before moving to the next.)

1. **Write `TestBoundaryMetrics` unit tests against not-yet-existing `boundary_metrics`.** All ~10 tests fail at `ImportError`. Confirms test wiring + serves as executable spec for the new function.
2. **Implement `boundary_metrics` + `BoundaryMetrics` TypedDict** in `silly_kicks/spadl/utils.py`, directly after `add_possessions` (line 693) and before `add_names` (line 696). Re-run unit tests → green. Validates implementation against semantics.
3. **Re-export from `silly_kicks/spadl/__init__.py`.** Verify `from silly_kicks.spadl import boundary_metrics, BoundaryMetrics` works; verify no circular-import regressions.
4. **Vendor the 3 fixture files** + `README.md` attribution. `pytest --collect-only` should pass without errors.
5. **Rewrite the e2e test class.** Drop `@pytest.mark.e2e`. Rename. Parametrize over 3 matches. Use the public `boundary_metrics`. Run against committed fixtures — measure actual recall/precision/F1, confirm above gates. If a match shows tighter-than-expected numbers, surface for design discussion before locking thresholds.
6. **Update `add_possessions` docstring** with empirical baselines. Verify rendering via `python -c "import silly_kicks.spadl.utils as u; help(u.add_possessions)"`.
7. **Update `CHANGELOG.md`, `TODO.md`, optionally `CLAUDE.md`.** Each verified for accuracy against the actual changes.

## 8. Verification gates (before commit)

```bash
# Match exact CI pin to avoid local/CI drift (per feedback_ci_cross_version memory)
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113

# Lint + format (matches CI lint job exactly)
uv run ruff check silly_kicks/ tests/
uv run ruff format --check silly_kicks/ tests/

# Type-check (matches CI lint job exactly)
uv run pyright silly_kicks/

# All tests run (e2e marker for committed-fixture tests is now dropped)
uv run pytest tests/ -v --tb=short
```

**Expected baseline output:** all existing tests pass + 10 new `TestBoundaryMetrics` unit tests pass + 3 parametrized `TestBoundaryAgainstStatsBombNative` tests pass + 5 `test_predict*` tests still skip (PR-S9 territory). Zero pyright errors. Zero ruff errors.

**`/final-review` skill** runs after the verification gates pass — per `feedback_commit_policy` memory it's mandatory before the single commit, not just pre-PR.

**Cross-version risk** (per `feedback_ci_cross_version` memory): this machine runs Python 3.14; CI runs 3.10/3.11/3.12. Pyright + pandas-stubs pin exactly matches above so library-level output is bit-identical. `TypedDict` import is from `typing` (stdlib since 3.8), not `typing_extensions`, so no Python-version branching needed.

## 9. Commit cycle

Per `feedback_commit_policy` memory: literally one commit per branch, no WIP commits + squash, explicit user approval at every step.

```
1. All gates green + /final-review pass
2. User approves → git add + git commit -s (one commit, branch feat/recall-based-add-possessions-validation)
3. User approves → git push -u origin feat/...
4. User approves → gh pr create
5. User approves → gh pr merge --admin --squash --delete-branch
6. User approves → git tag v1.8.0 + git push origin v1.8.0  # auto-fires PyPI publish workflow
```

## 10. Out of scope (queued follow-ups)

Tracked in detail in auto-memory at `project_followup_prs.md`; mirrored in `TODO.md` for in-repo discoverability.

### PR-S9 — e2e prediction tests in CI

Generate `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` from open-data raw events (~64 matches × ~600 KB). Output structure: `games` table + `actions/game_<id>` per match (see `tests/vaep/test_vaep.py:48` for the shape contract). Drop `@pytest.mark.e2e` on the 5 `test_predict*` cases. CI matrix runs the prediction suite on every PR (~30s per matrix slot, 4 slots).

Conversion script committed at `scripts/build_worldcup_fixture.py` for reproducibility. Estimated 1-2 hours of careful work + verification.

### PR-S10 — `add_possessions` algorithmic precision improvement

Close the precision gap from ~42% toward 60-70% via richer rules. Three directions:

1. **Brief-opposing-action merge rule.** Sandwiched team-B actions ≤ N count and ≤ T seconds → suppress B's boundary. Expected +20-30pp precision, ~0pp recall.
2. **Defensive-action class.** Treat `interception`/`clearance`/`tackle`/`blocked_pass` as "defensive transitions" — team change to one doesn't immediately start a new possession. Expected +15-20pp precision, possibly -1-2pp recall.
3. **Spatial continuity check.** `prev.end_(x,y)` ≈ `current.start_(x,y)` AND short time gap → prefer "same possession". +5pp precision, niche.

Plus parameter-tuning: re-measure recall/precision/F1 across `max_gap_seconds` grid using the new `boundary_metrics`, then make an informed default change. (LL2's sweep showed F1 peak at 10.0 vs default 5.0, but recall/precision split was not measured.)

API design questions:
- New parameters: `merge_brief_opposing_actions: bool = True`? `brief_action_window_seconds: float = 2.0`?
- Default behavior shift: opt-in or opt-out? Lakehouse is sole consumer, so flexibility is high — but the convention will be inherited by future consumers.
- `add_possessions_strict` variant vs parameter on the existing function? Keep one function with parameters; resist proliferation.
- Atomic-SPADL parity: if `add_possessions` semantics shift, the atomic counterpart (shipped 1.5.0) must mirror.

Sequencing note: PR-S10 is most productive AFTER PR-S9 — the WorldCup HDF5 provides 64 matches for parameter sweeping, vs PR-S8's 3 matches. Larger fixture → more reliable parameter-tuning measurement.

## 11. Risks + mitigations

| Risk | Mitigation |
|---|---|
| 0.30 precision floor too tight — match 3754058 measures 0.39 (only 9pp margin) | Step 5 of TDD plan re-measures actual values on committed fixtures. If any match comes in below ~0.35 (5pp margin), pause and discuss threshold before locking. |
| Threshold drift from machine to machine — `add_possessions` is deterministic but pandas/numpy version interactions could marginally shift values | Pyright + pandas-stubs pin matches CI exactly; pytest also runs with the pinned versions. CI matrix runs ubuntu-3.10/3.11/3.12 + windows-3.12 — if any matrix slot disagrees with the local measurement, that's a bug worth fixing. |
| TypedDict / `typing` import compatibility | `TypedDict` is in stdlib since 3.8; minimum supported is 3.10. No branching. |
| Precision regression hidden in heuristic refactors that pass unit tests but shift empirical-baseline numbers | Recall + precision floors in CI are precisely this gate. Algorithm-class changes should ship as PR-S10 with explicit before/after measurement. |
| StatsBomb open-data license drift — files vendored today may have license terms that change | License is non-commercial open-data; redistribution is permitted under same terms. README attribution + LICENSE link committed alongside the fixtures. If license terms tighten in the future, removal is a `git rm`. |

## 12. Acceptance criteria

1. `add_possessions` docstring updated with empirical recall / precision / F1 baselines.
2. `silly_kicks.spadl.boundary_metrics` is importable from `silly_kicks.spadl`, returns a `BoundaryMetrics` TypedDict, enforces keyword-only args, and is covered by ~10 unit tests across 4 sub-classes.
3. `tests/datasets/statsbomb/raw/events/{7298,7584,3754058}.json` are committed with the StatsBomb open-data attribution README.
4. `tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBombNative` is parametrized over the 3 fixtures, runs in CI on every PR (no `@pytest.mark.e2e`), gates on `recall ≥ 0.85 AND precision ≥ 0.30` per match, and records F1 + all three metrics in the failure assert message.
5. `CHANGELOG.md` has a new `## [1.8.0] — 2026-04-29` entry summarizing all changes.
6. `TODO.md` has `## Open PRs` section with PR-S9 and PR-S10 entries.
7. `pyproject.toml` version is `1.8.0`. Tag `v1.8.0` is pushed → PyPI auto-publish workflow fires.
8. `add_possessions` algorithm itself is unchanged.
9. Verification gates (ruff, pyright with exact pins, full pytest suite) all pass before commit.
10. `/final-review` skill passes before commit.
