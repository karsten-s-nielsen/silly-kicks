# PR-S12: `add_possessions` precision-improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three opt-in precision-improvement parameters to `add_possessions`, change `max_gap_seconds` default 5.0→7.0, regenerate the WorldCup-2018 HDF5 with native possession preserved, and extend the regression CI gate from 3 fixtures to 64+3 — shipping silly-kicks 2.1.0.

**Architecture:** Single-commit branch (per `feedback_commit_policy`). TDD throughout: failing tests → refactor scaffold → minimal-impl-per-rule → green. Atomic-SPADL mirrors standard with the same parameters and default. Boundary computation factors into a helper to keep the public function focused. Spec: `docs/superpowers/specs/2026-04-29-add-possessions-precision-improvement-design.md`.

**Tech Stack:** Python 3.10+, pandas 2.1+, numpy, pytest, ruff 0.15.7, pyright 1.1.395, pandas-stubs 2.3.3.260113. No new runtime dependencies.

**Per-task commit policy:** Per the user's `feedback_commit_policy`, **NO per-task commits**. Each task ends with verification but does NOT commit. The single commit happens in Task 14 after all gates pass and the user explicitly approves.

---

## File Structure

| Path | Action | Purpose |
|---|---|---|
| `silly_kicks/spadl/utils.py` | Modify | Add 3 keyword params + validation; gap default 5.0→7.0; extract `_compute_possession_boundaries` helper; docstring update |
| `silly_kicks/atomic/spadl/utils.py` | Modify | Same params + same default change in `_compute_possessions`; docstring update |
| `tests/spadl/test_add_possessions.py` | Modify | Add `TestBriefOpposingMerge`, `TestDefensiveTransitions`, `TestMaxGapDefaultIs7Seconds`; lower 3-JSON gate 0.85→0.83; add `TestBoundaryAgainstStatsBomb64Match` |
| `tests/atomic/test_atomic_add_possessions.py` | Modify | Mirror unit tests for atomic side |
| `scripts/build_worldcup_fixture.py` | Modify | Adapter includes `possession`; conversion uses `preserve_native=["possession"]` |
| `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` | Modify (regen) | New file with `possession` column; ~7 MB |
| `pyproject.toml` | Modify | Version 2.0.0→2.1.0 |
| `CHANGELOG.md` | Modify | `## [2.1.0]` entry with Breaking section + Behavior baselines + new params |
| `TODO.md` | Modify | Close PR-S12 entry; clean PR-S queue |
| `silly_kicks/spadl/__init__.py` | NO CHANGE | `boundary_metrics` + `add_possessions` already re-exported |

---

### Task 1: Write failing unit tests for new parameters (standard side)

**Files:**
- Modify: `tests/spadl/test_add_possessions.py`

Adds 3 new test classes covering the new parameters. All will fail with `TypeError: unexpected keyword argument` (params not yet on the function) — confirms test wiring + serves as executable spec.

- [ ] **Step 1: Read existing test file structure to understand the `_make_action` / `_df` fixture pattern**

Run: read `tests/spadl/test_add_possessions.py` lines 1-77 to confirm `_make_action()` fixture builder + `_df()` helper are usable. Confirms `_ACT`, `_RES`, `_BP` short aliases.

- [ ] **Step 2: Append `TestBriefOpposingMerge` class to the test file (just before `TestBoundaryAgainstStatsBombNative`)**

```python
# ---------------------------------------------------------------------------
# Rule 1: brief-opposing-action merge — added in 2.1.0
# ---------------------------------------------------------------------------


class TestBriefOpposingMerge:
    """``merge_brief_opposing_actions`` + ``brief_window_seconds`` pair.

    Suppresses team-change boundaries when team B has 1..N consecutive
    actions sandwiched between team A actions within ``brief_window_seconds``.
    Both kwargs must be > 0 to enable; both 0 to disable.
    """

    def test_aba_within_window_suppresses_boundary(self):
        # A (team 100) at t=0, B (team 200) at t=1.0, A (team 100) at t=2.0
        # Window: 2.0s (covers t=0→t=2.0 from row 0 perspective).
        # With N=1, T=2.0: detect 1 sandwiched B, suppress both boundaries.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=200, time_seconds=1.0),
                _make_action(action_id=2, team_id=100, time_seconds=2.0),
            ]
        )
        result = add_possessions(
            actions, merge_brief_opposing_actions=1, brief_window_seconds=2.0
        )
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 0, "B at t=1 should be merged"
        assert result["possession_id"].iloc[2] == 0, "A at t=2 should be merged"

    def test_aba_outside_window_keeps_boundary(self):
        # Same A B A pattern but B's window exceeded — boundaries stand.
        # T=2.0s, but t=0 to t=3.0 = 3.0s > 2.0s threshold → no merge.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=200, time_seconds=1.0),
                _make_action(action_id=2, team_id=100, time_seconds=3.0),
            ]
        )
        result = add_possessions(
            actions, merge_brief_opposing_actions=1, brief_window_seconds=2.0
        )
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 1
        assert result["possession_id"].iloc[2] == 2

    def test_abba_within_window_with_n_eq_2_suppresses(self):
        # A, B, B, A within window — 2 consecutive B's, N=2 covers it.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=200, time_seconds=0.5),
                _make_action(action_id=2, team_id=200, time_seconds=1.0),
                _make_action(action_id=3, team_id=100, time_seconds=1.5),
            ]
        )
        result = add_possessions(
            actions, merge_brief_opposing_actions=2, brief_window_seconds=2.0
        )
        assert result["possession_id"].nunique() == 1, "all merged into one possession"

    def test_abbba_with_n_eq_2_keeps_boundary(self):
        # A, B, B, B, A — 3 consecutive B's, N=2 doesn't cover; boundaries stand.
        # The N=2 rule means "look ahead at most 2 rows for original team to come back".
        # At i=1 (A→B), look ahead k=1 (still B), k=2 (still B). No match. Boundary stands.
        # At i=4 (B→A), no candidates ahead. Boundary stands.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=200, time_seconds=0.5),
                _make_action(action_id=2, team_id=200, time_seconds=1.0),
                _make_action(action_id=3, team_id=200, time_seconds=1.5),
                _make_action(action_id=4, team_id=100, time_seconds=2.0),
            ]
        )
        result = add_possessions(
            actions, merge_brief_opposing_actions=2, brief_window_seconds=3.0
        )
        # Three distinct possessions: A, B, A
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 1
        assert result["possession_id"].iloc[4] == 2

    def test_disabled_when_both_zero(self):
        # Default values (both 0) → identical to no-rule baseline.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=200, time_seconds=1.0),
                _make_action(action_id=2, team_id=100, time_seconds=2.0),
            ]
        )
        with_rule_off = add_possessions(
            actions, merge_brief_opposing_actions=0, brief_window_seconds=0.0
        )
        baseline = add_possessions(actions)
        assert (with_rule_off["possession_id"] == baseline["possession_id"]).all()

    def test_partial_config_actions_only_raises(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match=r"both"):
            add_possessions(actions, merge_brief_opposing_actions=2, brief_window_seconds=0.0)

    def test_partial_config_seconds_only_raises(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match=r"both"):
            add_possessions(actions, merge_brief_opposing_actions=0, brief_window_seconds=2.0)

    def test_negative_actions_raises(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match=r"merge_brief_opposing_actions"):
            add_possessions(actions, merge_brief_opposing_actions=-1, brief_window_seconds=2.0)

    def test_negative_seconds_raises(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match=r"brief_window_seconds"):
            add_possessions(actions, merge_brief_opposing_actions=1, brief_window_seconds=-0.5)

    def test_game_boundary_blocks_lookahead(self):
        # Brief-merge must not cross game_id boundaries.
        # Game 1: A at t=0, B at t=1
        # Game 2: A at t=0
        # Even though it looks like ABA in row order, the game boundary blocks merge.
        actions = _df(
            [
                _make_action(action_id=0, game_id=1, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, game_id=1, team_id=200, time_seconds=1.0),
                _make_action(action_id=0, game_id=2, team_id=100, time_seconds=0.0),
            ]
        )
        result = add_possessions(
            actions, merge_brief_opposing_actions=1, brief_window_seconds=2.0
        )
        # Game 1 has 2 distinct possessions (A then B). Game 2 has 1 possession.
        assert result["possession_id"].iloc[0] == 0  # game 1 first possession
        assert result["possession_id"].iloc[1] == 1  # game 1 second possession (no merge)
        assert result["possession_id"].iloc[2] == 0  # game 2 starts fresh

    def test_period_boundary_blocks_lookahead(self):
        # Same intent across period boundary.
        actions = _df(
            [
                _make_action(action_id=0, period_id=1, team_id=100, time_seconds=2700.0),
                _make_action(action_id=1, period_id=1, team_id=200, time_seconds=2700.5),
                _make_action(action_id=2, period_id=2, team_id=100, time_seconds=0.0),
            ]
        )
        result = add_possessions(
            actions, merge_brief_opposing_actions=1, brief_window_seconds=2.0
        )
        # Period change forces boundary regardless of brief-merge.
        assert result["possession_id"].iloc[2] == 2
```

- [ ] **Step 3: Append `TestDefensiveTransitions` class right after `TestBriefOpposingMerge`**

```python
class TestDefensiveTransitions:
    """``defensive_transition_types`` rule.

    Action types listed do NOT trigger team-change boundaries on their own.
    Useful action types per measurement: ``interception``, ``clearance``,
    ``tackle``, ``bad_touch``.
    """

    def test_interception_does_not_trigger_boundary(self):
        # A passes, B intercepts, B passes successfully → expected to merge into B's possession.
        # With defensive=("interception",): interception by B doesn't trigger A→B boundary.
        # The next action B-pass is the first non-defensive action; if same team as interception,
        # it's same possession; the team-change is recognized at this point if applicable.
        # In this simple case, B intercepts then B passes → no boundary fires (current behavior
        # would emit boundary at intercept; with rule, intercept is "transitional").
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0, type_name="pass"),
                _make_action(action_id=1, team_id=200, time_seconds=1.0, type_name="interception"),
                _make_action(action_id=2, team_id=200, time_seconds=2.0, type_name="pass"),
            ]
        )
        result = add_possessions(
            actions, defensive_transition_types=("interception",)
        )
        # With rule on: row 1 (interception) doesn't trigger boundary;
        # row 2 (pass by B, team change vs row 1's not-counted-as-team-A; still team 200) → no boundary.
        # Wait — the team_change check still uses prev_team[i]. At row 1, prev_team=100, current=200,
        # team_change=True. With defensive rule, suppress. But at row 2, prev_team=200 (row 1),
        # current=200, no team_change. So overall, no boundary in rows 1 or 2.
        # Possession: 0 0 0 (A's possession continues through the intercept).
        # Empirically this is the StatsBomb-style merge behavior.
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 0
        assert result["possession_id"].iloc[2] == 0

    def test_pass_after_interception_keeps_separate_possession_when_recovered(self):
        # If A's pass, B intercepts, A passes → row 2 is A's pass. Team change at row 1
        # is suppressed (defensive). At row 2, prev_team=200, current=100, team_change=True,
        # current is NOT defensive, so boundary fires. Possession: 0 0 1.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0, type_name="pass"),
                _make_action(action_id=1, team_id=200, time_seconds=1.0, type_name="interception"),
                _make_action(action_id=2, team_id=100, time_seconds=2.0, type_name="pass"),
            ]
        )
        result = add_possessions(
            actions, defensive_transition_types=("interception",)
        )
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 0  # intercept suppressed
        assert result["possession_id"].iloc[2] == 1  # A pass after B intercept = new possession

    def test_unknown_type_raises_value_error(self):
        actions = _df([_make_action(action_id=0)])
        with pytest.raises(ValueError, match=r"unknown action types"):
            add_possessions(actions, defensive_transition_types=("not_a_type",))

    def test_empty_tuple_no_op(self):
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=200, time_seconds=1.0),
            ]
        )
        with_rule = add_possessions(actions, defensive_transition_types=())
        baseline = add_possessions(actions)
        assert (with_rule["possession_id"] == baseline["possession_id"]).all()

    def test_multi_type_tuple(self):
        # With ("interception", "clearance") → both types suppress team-change boundary.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0, type_name="pass"),
                _make_action(action_id=1, team_id=200, time_seconds=1.0, type_name="clearance"),
                _make_action(action_id=2, team_id=200, time_seconds=2.0, type_name="pass"),
            ]
        )
        result = add_possessions(
            actions, defensive_transition_types=("interception", "clearance")
        )
        # Same logic as interception case: clearance suppresses team-change boundary at row 1.
        assert result["possession_id"].iloc[0] == 0
        assert result["possession_id"].iloc[1] == 0
        assert result["possession_id"].iloc[2] == 0
```

- [ ] **Step 4: Append `TestMaxGapDefaultIs7Seconds` class**

```python
class TestMaxGapDefaultIs7Seconds:
    """The ``max_gap_seconds`` default changed from 5.0 to 7.0 in 2.1.0.

    Behavior break: same-team actions with time gap in [5, 7) seconds are
    now in the same possession (previously a new possession at gap >= 5).
    """

    def test_default_value_is_7(self):
        import inspect
        sig = inspect.signature(add_possessions)
        assert sig.parameters["max_gap_seconds"].default == 7.0

    def test_5_to_6s_gap_no_boundary_at_default(self):
        # Same team, gap = 6.0s. Under 5.0 default this would be new possession;
        # under 7.0 default this is same possession.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=100, time_seconds=6.0),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[0] == result["possession_id"].iloc[1]

    def test_7_to_8s_gap_boundary_at_default(self):
        # Same team, gap = 7.5s — new possession under 7.0 default (>= 7.0).
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=100, time_seconds=7.5),
            ]
        )
        result = add_possessions(actions)
        assert result["possession_id"].iloc[0] != result["possession_id"].iloc[1]

    def test_explicit_5_0_still_works(self):
        # Opt-out path: explicitly set max_gap_seconds=5.0.
        actions = _df(
            [
                _make_action(action_id=0, team_id=100, time_seconds=0.0),
                _make_action(action_id=1, team_id=100, time_seconds=6.0),
            ]
        )
        result = add_possessions(actions, max_gap_seconds=5.0)
        assert result["possession_id"].iloc[0] != result["possession_id"].iloc[1]
```

- [ ] **Step 5: Run all new tests; expect all to fail at `TypeError` or assertion**

Run: `uv run pytest tests/spadl/test_add_possessions.py::TestBriefOpposingMerge tests/spadl/test_add_possessions.py::TestDefensiveTransitions tests/spadl/test_add_possessions.py::TestMaxGapDefaultIs7Seconds -v`

Expected: All 3 classes' tests fail with `TypeError: add_possessions() got an unexpected keyword argument` for the params, OR `AssertionError` on the default-value test (default is currently 5.0). Confirms test wiring works.

---

### Task 2: Update existing-test thresholds + add HDF5 64-match skeleton

**Files:**
- Modify: `tests/spadl/test_add_possessions.py:585` (the existing 3-JSON test — change recall threshold)
- Modify: `tests/spadl/test_add_possessions.py` (add `TestBoundaryAgainstStatsBomb64Match` after the 3-JSON class)

- [ ] **Step 1: Lower the 3-JSON test recall gate from 0.85 to 0.83**

Edit `tests/spadl/test_add_possessions.py` find:

```python
        # Per-match independent gates.
        # Recall floor: 0.85 — 8pp below the worst observed (~0.93).
        # Precision floor: 0.30 — 9pp below the worst observed (~0.39 on match 3754058).
        # F1 in message only — F1 conflates two signals; gating on it
        # would re-introduce the misrepresentation problem the docstring
        # rewrite is fixing.
        assert m["recall"] >= 0.85 and m["precision"] >= 0.30, (
            f"match {match_id}: recall={m['recall']:.4f} "
            f"precision={m['precision']:.4f} f1={m['f1']:.4f}; "
            f"thresholds: recall>=0.85 precision>=0.30"
        )
```

Replace with:

```python
        # Per-match independent gates (PR-S12, silly-kicks 2.1.0).
        # Recall floor: 0.83 — 4pp below worst observed (R_min=0.854 at gap=7.0
        # across 64 WC-2018 matches; 3 committed JSON fixtures have higher
        # R_min). Loosened from 0.85 in 2.1.0 alongside the max_gap_seconds
        # default change 5.0→7.0.
        # Precision floor: 0.30 — 5pp below worst observed (P_min=0.350 at gap=7.0).
        # F1 in message only — gating on it conflates two independent signals.
        assert m["recall"] >= 0.83 and m["precision"] >= 0.30, (
            f"match {match_id}: recall={m['recall']:.4f} "
            f"precision={m['precision']:.4f} f1={m['f1']:.4f}; "
            f"thresholds: recall>=0.83 precision>=0.30"
        )
```

- [ ] **Step 2: Update the existing 3-JSON test class docstring**

Find:

```python
class TestBoundaryAgainstStatsBombNative:
    """Validate add_possessions against StatsBomb's native possession_id.

    Empirically against StatsBomb open-data, this heuristic achieves
    boundary recall ~0.93 and boundary F1 ~0.58. The precision gap is
    intrinsic to the team-change-with-carve-outs algorithm class. The
    CI gate below tests recall AND precision because both are observable
    behaviors that downstream consumers can develop dependencies on —
    F1 conflates two signals with very different magnitudes and is
    recorded for diagnostics only.
```

Replace with:

```python
class TestBoundaryAgainstStatsBombNative:
    """Validate add_possessions against StatsBomb's native possession_id.

    PR-S12 (silly-kicks 2.1.0) updated the empirical baselines to the
    new ``max_gap_seconds=7.0`` default: per-match boundary recall ~0.94
    and boundary F1 ~0.60. The precision gap remains intrinsic to the
    team-change-with-carve-outs algorithm class. The CI gate below tests
    recall AND precision because both are observable behaviors that
    downstream consumers can develop dependencies on — F1 conflates two
    signals with very different magnitudes and is recorded for diagnostics
    only.

    Companion test: :class:`TestBoundaryAgainstStatsBomb64Match` runs the
    same gate across 64 FIFA WorldCup-2018 matches via the committed
    HDF5 fixture (`tests/datasets/statsbomb/spadl-WorldCup-2018.h5`).
    Cross-competition coverage here; within-competition variance there.
```

- [ ] **Step 3: Append `TestBoundaryAgainstStatsBomb64Match` class to the test file**

Position: after `TestBoundaryAgainstStatsBombNative`, before the `TestBoundaryMetrics*` classes (line ~592).

```python
class TestBoundaryAgainstStatsBomb64Match:
    """Validate add_possessions against StatsBomb native across 64 WC-2018 matches.

    Reads the committed HDF5 fixture
    ``tests/datasets/statsbomb/spadl-WorldCup-2018.h5`` (regenerated in
    silly-kicks 2.1.0 to preserve the StatsBomb ``possession`` column).
    Each match is gated independently at ``recall >= 0.83 AND
    precision >= 0.30`` per the same per-match contract as
    :class:`TestBoundaryAgainstStatsBombNative`.

    Within-competition variance complement to the cross-competition
    3-fixture test. ~1-2s additional CI runtime after HDFStore cold-load.
    """

    @pytest.fixture(scope="class")
    def match_ids(self, sb_worldcup_data: pd.HDFStore) -> list[int]:
        keys = [k for k in sb_worldcup_data.keys() if k.startswith("/actions/game_")]
        return sorted(int(k.removeprefix("/actions/game_")) for k in keys)

    def test_fixture_has_possession_column(self, sb_worldcup_data: pd.HDFStore, match_ids: list[int]):
        # Sentinel: if the HDF5 was built without preserve_native=["possession"],
        # the entire test class will fail at this guard. Clear failure mode
        # vs 64 cryptic per-match KeyErrors.
        assert match_ids, "no actions/game_<id> keys in HDF5 fixture"
        first = sb_worldcup_data.get(f"actions/game_{match_ids[0]}")
        assert "possession" in first.columns, (
            "HDF5 fixture missing `possession` column. Regenerate with "
            "`uv run python scripts/build_worldcup_fixture.py --verbose`."
        )

    @pytest.mark.parametrize(
        "match_id",
        # Lazy parametrize using the fixture is incompatible with parametrize;
        # we hardcode the 64 WC-2018 match IDs from the manifest. If the manifest
        # changes upstream, this list needs regeneration.
        [
            7525, 7529, 7530, 7531, 7532, 7533, 7534, 7535, 7536, 7537, 7538, 7539,
            7540, 7541, 7542, 7543, 7544, 7545, 7546, 7547, 7548, 7549, 7550, 7551,
            7552, 7553, 7554, 7555, 7556, 7557, 7558, 7559, 7560, 7561, 7562, 7563,
            7564, 7565, 7566, 7567, 7568, 7569, 7570, 7571, 7572, 7576, 7577, 7578,
            7579, 7580, 7581, 7582, 7583, 7584, 7585, 7586, 8649, 8650, 8651, 8652,
            8655, 8656, 8657, 8658,
        ],
    )
    def test_boundary_metrics_against_native_per_match(
        self, sb_worldcup_data: pd.HDFStore, match_id: int
    ):
        actions = sb_worldcup_data.get(f"actions/game_{match_id}")
        non_synth = actions[actions["possession"].notna()].copy()
        non_synth = add_possessions(non_synth)

        m = boundary_metrics(
            heuristic=non_synth["possession_id"],
            native=non_synth["possession"].astype(np.int64),
        )

        assert m["recall"] >= 0.83 and m["precision"] >= 0.30, (
            f"match {match_id}: recall={m['recall']:.4f} "
            f"precision={m['precision']:.4f} f1={m['f1']:.4f}; "
            f"thresholds: recall>=0.83 precision>=0.30"
        )
```

- [ ] **Step 4: Run the existing 3-JSON test to confirm threshold change doesn't break it**

Run: `uv run pytest tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBombNative -v`

Expected: 3 passing tests. Lower threshold should still hold (the 3 fixtures historically have R_min much higher than 0.83 even at gap=5.0).

- [ ] **Step 5: Run the new HDF5 test to confirm it fails at the sentinel**

Run: `uv run pytest tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBomb64Match::test_fixture_has_possession_column -v`

Expected: FAIL with `assert "possession" in first.columns` — fixture currently lacks the column. This confirms the sentinel works and the 64-match test won't run until Task 9.

---

### Task 3: Refactor `add_possessions` — extract `_compute_possession_boundaries` helper

**Files:**
- Modify: `silly_kicks/spadl/utils.py:552-713`

Pure refactor, no behavior change. Extract the boundary-detection logic into a private helper to keep the public function focused and to provide a clean target for the new rules.

- [ ] **Step 1: Read existing `add_possessions` to lock in the current shape**

Run: read `silly_kicks/spadl/utils.py:552-713`. Identify the boundary mask construction at lines 663-703 — this entire block becomes the helper body.

- [ ] **Step 2: Add the new helper `_compute_possession_boundaries` immediately above `add_possessions`**

Insert between line 551 (end of `add_pre_shot_gk_context`) and line 552 (start of `add_possessions`):

```python
def _compute_possession_boundaries(
    sorted_actions: pd.DataFrame,
    *,
    max_gap_seconds: float,
    retain_on_set_pieces: bool,
) -> np.ndarray:
    """Return a boolean mask: True at rows that start a new possession.

    Operates on a pre-sorted ``(game_id, period_id, action_id)`` DataFrame.
    Vectorized; no row-level Python iteration. Shared logic for
    :func:`add_possessions` and any future variants.

    Parameters
    ----------
    sorted_actions : pd.DataFrame
        SPADL action stream, already sorted. Must contain ``game_id``,
        ``period_id``, ``team_id``, ``time_seconds``, ``type_id``.
    max_gap_seconds : float
        Time-gap threshold (seconds) above which a new possession starts
        regardless of team.
    retain_on_set_pieces : bool
        Whether to apply the foul-then-set-piece carve-out.

    Returns
    -------
    np.ndarray
        Shape ``(n,)`` boolean array. ``True`` at rows that begin a new
        possession (game change, period change, gap timeout, or team change
        without carve-out).
    """
    n = len(sorted_actions)
    if n == 0:
        return np.zeros(0, dtype=bool)

    game_id = sorted_actions["game_id"].to_numpy()
    period_id = sorted_actions["period_id"].to_numpy()
    team_id = sorted_actions["team_id"].to_numpy()
    time_seconds = sorted_actions["time_seconds"].to_numpy()
    type_id = sorted_actions["type_id"].to_numpy()

    prev_period = np.empty(n, dtype=period_id.dtype)
    prev_team = np.empty(n, dtype=team_id.dtype)
    prev_time = np.empty(n, dtype=time_seconds.dtype)
    prev_type = np.empty(n, dtype=type_id.dtype)
    prev_period[0] = period_id[0]
    prev_team[0] = team_id[0]
    prev_time[0] = time_seconds[0]
    prev_type[0] = type_id[0]
    prev_period[1:] = period_id[:-1]
    prev_team[1:] = team_id[:-1]
    prev_time[1:] = time_seconds[:-1]
    prev_type[1:] = type_id[:-1]

    game_change = np.empty(n, dtype=bool)
    game_change[0] = True
    game_change[1:] = game_id[1:] != game_id[:-1]

    period_change_within_game = (~game_change) & (period_id != prev_period)
    gap_timeout = (
        (~game_change) & (~period_change_within_game) & ((time_seconds - prev_time) >= max_gap_seconds)
    )
    team_change = (~game_change) & (team_id != prev_team)

    set_piece_ids = {spadlconfig.actiontype_id[name] for name in _SET_PIECE_RESTART_TYPE_NAMES}
    foul_id = spadlconfig.actiontype_id["foul"]
    is_set_piece = np.isin(type_id, list(set_piece_ids))
    prev_is_foul = prev_type == foul_id
    set_piece_carve_out = retain_on_set_pieces & team_change & is_set_piece & prev_is_foul

    return game_change | period_change_within_game | gap_timeout | (team_change & ~set_piece_carve_out)
```

- [ ] **Step 3: Replace the inlined boundary computation in `add_possessions` with a call to the helper**

Find in `add_possessions` (lines 662-703 before refactor):

```python
    # Vectorised boundary detection.
    game_id = sorted_actions["game_id"].to_numpy()
    period_id = sorted_actions["period_id"].to_numpy()
    team_id = sorted_actions["team_id"].to_numpy()
    time_seconds = sorted_actions["time_seconds"].to_numpy()
    type_id = sorted_actions["type_id"].to_numpy()
    # ... (entire boundary block) ...
    new_possession_mask = game_change | period_change_within_game | gap_timeout | (team_change & ~set_piece_carve_out)
```

Replace with:

```python
    new_possession_mask = _compute_possession_boundaries(
        sorted_actions,
        max_gap_seconds=max_gap_seconds,
        retain_on_set_pieces=retain_on_set_pieces,
    )
```

- [ ] **Step 4: Run all existing tests; expect zero regressions**

Run: `uv run pytest tests/spadl/test_add_possessions.py -v --tb=short -k "not TestBriefOpposingMerge and not TestDefensiveTransitions and not TestMaxGapDefaultIs7Seconds and not TestBoundaryAgainstStatsBomb64Match"`

Expected: All existing tests pass. (We exclude the new tests from Task 1+2 since they still fail.)

---

### Task 4: Add new keyword-only parameters with validation (standard side)

**Files:**
- Modify: `silly_kicks/spadl/utils.py` (`add_possessions` signature + validation)

Adds the 3 new parameters to the signature with default values (rules disabled). Adds validation. Behavior unchanged from refactor in Task 3 — rules still need to be implemented in Tasks 5-6.

- [ ] **Step 1: Update the `add_possessions` signature**

Find:

```python
def add_possessions(
    actions: pd.DataFrame,
    *,
    max_gap_seconds: float = 5.0,
    retain_on_set_pieces: bool = True,
) -> pd.DataFrame:
```

Replace with:

```python
def add_possessions(
    actions: pd.DataFrame,
    *,
    max_gap_seconds: float = 7.0,
    retain_on_set_pieces: bool = True,
    merge_brief_opposing_actions: int = 0,
    brief_window_seconds: float = 0.0,
    defensive_transition_types: tuple[str, ...] = (),
) -> pd.DataFrame:
```

Note: this changes `max_gap_seconds` default to 7.0 in the same edit. The `TestMaxGapDefaultIs7Seconds.test_default_value_is_7` test will turn green here.

- [ ] **Step 2: Add the new validation logic after the existing `max_gap_seconds` validation**

Find (around line 651):

```python
    if max_gap_seconds < 0:
        raise ValueError(f"add_possessions: max_gap_seconds must be >= 0, got {max_gap_seconds}")
```

Append after that line (before the `sort_values` call):

```python
    if merge_brief_opposing_actions < 0:
        raise ValueError(
            f"add_possessions: merge_brief_opposing_actions must be >= 0, got {merge_brief_opposing_actions}"
        )
    if brief_window_seconds < 0:
        raise ValueError(
            f"add_possessions: brief_window_seconds must be >= 0, got {brief_window_seconds}"
        )
    if (merge_brief_opposing_actions > 0) != (brief_window_seconds > 0):
        raise ValueError(
            "add_possessions: merge_brief_opposing_actions and brief_window_seconds must "
            "both be > 0 to enable the brief-opposing-merge rule, or both 0 to disable. "
            f"Got merge_brief_opposing_actions={merge_brief_opposing_actions}, "
            f"brief_window_seconds={brief_window_seconds}."
        )
    invalid_defensive = [t for t in defensive_transition_types if t not in spadlconfig.actiontype_id]
    if invalid_defensive:
        raise ValueError(
            f"add_possessions: defensive_transition_types contains unknown action types: "
            f"{sorted(invalid_defensive)}. Valid types: {sorted(spadlconfig.actiontype_id.keys())}"
        )
```

- [ ] **Step 3: Run validation-only tests; expect them green**

Run: `uv run pytest tests/spadl/test_add_possessions.py::TestBriefOpposingMerge::test_partial_config_actions_only_raises tests/spadl/test_add_possessions.py::TestBriefOpposingMerge::test_partial_config_seconds_only_raises tests/spadl/test_add_possessions.py::TestBriefOpposingMerge::test_negative_actions_raises tests/spadl/test_add_possessions.py::TestBriefOpposingMerge::test_negative_seconds_raises tests/spadl/test_add_possessions.py::TestBriefOpposingMerge::test_disabled_when_both_zero tests/spadl/test_add_possessions.py::TestDefensiveTransitions::test_unknown_type_raises_value_error tests/spadl/test_add_possessions.py::TestDefensiveTransitions::test_empty_tuple_no_op tests/spadl/test_add_possessions.py::TestMaxGapDefaultIs7Seconds -v`

Expected: All validation + default tests pass. Behavior tests for the rules still fail.

---

### Task 5: Implement Rule 2 (`defensive_transition_types`)

**Files:**
- Modify: `silly_kicks/spadl/utils.py` (`_compute_possession_boundaries` signature + body, `add_possessions` call site)

Easiest rule. Vectorized via `np.isin` + AND-out of the boundary mask.

- [ ] **Step 1: Extend `_compute_possession_boundaries` signature**

Find:

```python
def _compute_possession_boundaries(
    sorted_actions: pd.DataFrame,
    *,
    max_gap_seconds: float,
    retain_on_set_pieces: bool,
) -> np.ndarray:
```

Replace with:

```python
def _compute_possession_boundaries(
    sorted_actions: pd.DataFrame,
    *,
    max_gap_seconds: float,
    retain_on_set_pieces: bool,
    defensive_transition_types: tuple[str, ...] = (),
) -> np.ndarray:
```

- [ ] **Step 2: Apply the defensive-transition mask just before the final `return` in the helper**

Find the last 4 lines of the helper (the `set_piece_carve_out` and `return`):

```python
    set_piece_ids = {spadlconfig.actiontype_id[name] for name in _SET_PIECE_RESTART_TYPE_NAMES}
    foul_id = spadlconfig.actiontype_id["foul"]
    is_set_piece = np.isin(type_id, list(set_piece_ids))
    prev_is_foul = prev_type == foul_id
    set_piece_carve_out = retain_on_set_pieces & team_change & is_set_piece & prev_is_foul

    return game_change | period_change_within_game | gap_timeout | (team_change & ~set_piece_carve_out)
```

Replace with:

```python
    set_piece_ids = {spadlconfig.actiontype_id[name] for name in _SET_PIECE_RESTART_TYPE_NAMES}
    foul_id = spadlconfig.actiontype_id["foul"]
    is_set_piece = np.isin(type_id, list(set_piece_ids))
    prev_is_foul = prev_type == foul_id
    set_piece_carve_out = retain_on_set_pieces & team_change & is_set_piece & prev_is_foul

    boundary = team_change & ~set_piece_carve_out

    # Rule 2 (PR-S12): defensive_transition_types — listed types do not
    # trigger team-change boundaries on their own.
    if defensive_transition_types:
        defensive_ids = {spadlconfig.actiontype_id[name] for name in defensive_transition_types}
        is_defensive = np.isin(type_id, list(defensive_ids))
        boundary = boundary & ~is_defensive

    return game_change | period_change_within_game | gap_timeout | boundary
```

- [ ] **Step 3: Pass the parameter through from `add_possessions`**

Find the helper call:

```python
    new_possession_mask = _compute_possession_boundaries(
        sorted_actions,
        max_gap_seconds=max_gap_seconds,
        retain_on_set_pieces=retain_on_set_pieces,
    )
```

Replace with:

```python
    new_possession_mask = _compute_possession_boundaries(
        sorted_actions,
        max_gap_seconds=max_gap_seconds,
        retain_on_set_pieces=retain_on_set_pieces,
        defensive_transition_types=defensive_transition_types,
    )
```

- [ ] **Step 4: Run defensive-rule tests; expect green**

Run: `uv run pytest tests/spadl/test_add_possessions.py::TestDefensiveTransitions -v`

Expected: All 5 tests in `TestDefensiveTransitions` pass.

---

### Task 6: Implement Rule 1 (brief-opposing-action merge)

**Files:**
- Modify: `silly_kicks/spadl/utils.py` (`_compute_possession_boundaries`, `add_possessions` call site)

Vectorized look-ahead via shifted aligned arrays + game/period boundary masks. Implementation must be order-of-rule-application aware: brief-merge runs AFTER defensive-rule + set-piece-carve-out (works on the post-rule-2 candidate boundaries).

- [ ] **Step 1: Extend `_compute_possession_boundaries` signature**

Find:

```python
def _compute_possession_boundaries(
    sorted_actions: pd.DataFrame,
    *,
    max_gap_seconds: float,
    retain_on_set_pieces: bool,
    defensive_transition_types: tuple[str, ...] = (),
) -> np.ndarray:
```

Replace with:

```python
def _compute_possession_boundaries(
    sorted_actions: pd.DataFrame,
    *,
    max_gap_seconds: float,
    retain_on_set_pieces: bool,
    defensive_transition_types: tuple[str, ...] = (),
    merge_brief_opposing_actions: int = 0,
    brief_window_seconds: float = 0.0,
) -> np.ndarray:
```

- [ ] **Step 2: Implement vectorized brief-merge suppression in the helper**

Find (after the defensive-rule block, before the final `return`):

```python
    # Rule 2 (PR-S12): defensive_transition_types — listed types do not
    # trigger team-change boundaries on their own.
    if defensive_transition_types:
        defensive_ids = {spadlconfig.actiontype_id[name] for name in defensive_transition_types}
        is_defensive = np.isin(type_id, list(defensive_ids))
        boundary = boundary & ~is_defensive

    return game_change | period_change_within_game | gap_timeout | boundary
```

Replace with:

```python
    # Rule 2 (PR-S12): defensive_transition_types — listed types do not
    # trigger team-change boundaries on their own.
    if defensive_transition_types:
        defensive_ids = {spadlconfig.actiontype_id[name] for name in defensive_transition_types}
        is_defensive = np.isin(type_id, list(defensive_ids))
        boundary = boundary & ~is_defensive

    # Rule 1 (PR-S12): brief-opposing-action merge. For each surviving
    # team-change boundary at row i, look ahead k=1..N rows; if any row
    # i+k has team_id == prev_team[i] (the original team has come back)
    # within the time window AND same game_id/period_id, suppress both
    # the boundary at i AND the boundary at i+k (the team-flip-back).
    if merge_brief_opposing_actions > 0 and brief_window_seconds > 0:
        suppress_at_i = np.zeros(n, dtype=bool)
        suppress_at_k = np.zeros(n, dtype=bool)
        for k in range(1, merge_brief_opposing_actions + 1):
            # Aligned look-ahead: index i sees row i+k; sentinel for last k positions.
            team_at_k = np.empty(n, dtype=team_id.dtype)
            time_at_k = np.empty(n, dtype=time_seconds.dtype)
            game_at_k = np.empty(n, dtype=game_id.dtype)
            period_at_k = np.empty(n, dtype=period_id.dtype)
            if n > k:
                team_at_k[: n - k] = team_id[k:]
                time_at_k[: n - k] = time_seconds[k:]
                game_at_k[: n - k] = game_id[k:]
                period_at_k[: n - k] = period_id[k:]
            # Sentinels for last k positions: time=+inf forces window check to fail;
            # game/period sentinels also fail the same-game-period check.
            team_at_k[n - k :] = team_id[-1]  # arbitrary; never used due to sentinels below
            time_at_k[n - k :] = np.inf
            game_at_k[n - k :] = -1
            period_at_k[n - k :] = -1

            same_game_period = (game_at_k == game_id) & (period_at_k == period_id)
            within_time = (time_at_k - time_seconds) <= brief_window_seconds
            team_back = team_at_k == prev_team
            match = boundary & same_game_period & within_time & team_back

            suppress_at_i |= match
            # Shift match by k positions to mark the team-flip-back boundary for suppression.
            shifted = np.zeros(n, dtype=bool)
            if n > k:
                shifted[k:] = match[: n - k]
            suppress_at_k |= shifted

        boundary = boundary & ~suppress_at_i & ~suppress_at_k

    return game_change | period_change_within_game | gap_timeout | boundary
```

- [ ] **Step 3: Pass the new parameters through from `add_possessions`**

Find:

```python
    new_possession_mask = _compute_possession_boundaries(
        sorted_actions,
        max_gap_seconds=max_gap_seconds,
        retain_on_set_pieces=retain_on_set_pieces,
        defensive_transition_types=defensive_transition_types,
    )
```

Replace with:

```python
    new_possession_mask = _compute_possession_boundaries(
        sorted_actions,
        max_gap_seconds=max_gap_seconds,
        retain_on_set_pieces=retain_on_set_pieces,
        defensive_transition_types=defensive_transition_types,
        merge_brief_opposing_actions=merge_brief_opposing_actions,
        brief_window_seconds=brief_window_seconds,
    )
```

- [ ] **Step 4: Run brief-merge tests; expect green**

Run: `uv run pytest tests/spadl/test_add_possessions.py::TestBriefOpposingMerge -v`

Expected: All 11 tests in `TestBriefOpposingMerge` pass.

- [ ] **Step 5: Run the full standard-side test suite; expect green**

Run: `uv run pytest tests/spadl/test_add_possessions.py -v --tb=short`

Expected: All standard-side tests pass except `TestBoundaryAgainstStatsBomb64Match` (still blocked by HDF5 missing `possession`).

---

### Task 7: Mirror to atomic-SPADL

**Files:**
- Modify: `silly_kicks/atomic/spadl/utils.py` (`add_possessions` signature + validation; `_compute_possessions` body)

Atomic side mirrors the same parameter additions, same default change, same rule semantics — but uses the atomic action-type vocabulary for validation (which differs from standard: `corner`/`freekick` collapse and atomic-only types `receival`, `out`, `dribble_release`, etc.).

- [ ] **Step 1: Read existing atomic `add_possessions` to lock in current shape**

Run: read `silly_kicks/atomic/spadl/utils.py:173-340`. Confirm `_compute_possessions` is the helper (lines 288-340) — that's the target for the rule additions.

- [ ] **Step 2: Update atomic `add_possessions` signature + validation**

Find:

```python
def add_possessions(
    actions: pd.DataFrame,
    *,
    max_gap_seconds: float = 5.0,
    retain_on_set_pieces: bool = True,
) -> pd.DataFrame:
```

Replace with:

```python
def add_possessions(
    actions: pd.DataFrame,
    *,
    max_gap_seconds: float = 7.0,
    retain_on_set_pieces: bool = True,
    merge_brief_opposing_actions: int = 0,
    brief_window_seconds: float = 0.0,
    defensive_transition_types: tuple[str, ...] = (),
) -> pd.DataFrame:
```

Find the validation block:

```python
    missing = [c for c in _ADD_POSSESSIONS_REQUIRED_COLUMNS if c not in actions.columns]
    if missing:
        raise ValueError(
            f"add_possessions: actions missing required columns: {sorted(missing)}. Got: {sorted(actions.columns)}"
        )
    if max_gap_seconds < 0:
        raise ValueError(f"add_possessions: max_gap_seconds must be >= 0, got {max_gap_seconds}")
```

Replace with:

```python
    missing = [c for c in _ADD_POSSESSIONS_REQUIRED_COLUMNS if c not in actions.columns]
    if missing:
        raise ValueError(
            f"add_possessions: actions missing required columns: {sorted(missing)}. Got: {sorted(actions.columns)}"
        )
    if max_gap_seconds < 0:
        raise ValueError(f"add_possessions: max_gap_seconds must be >= 0, got {max_gap_seconds}")
    if merge_brief_opposing_actions < 0:
        raise ValueError(
            f"add_possessions: merge_brief_opposing_actions must be >= 0, got {merge_brief_opposing_actions}"
        )
    if brief_window_seconds < 0:
        raise ValueError(
            f"add_possessions: brief_window_seconds must be >= 0, got {brief_window_seconds}"
        )
    if (merge_brief_opposing_actions > 0) != (brief_window_seconds > 0):
        raise ValueError(
            "add_possessions: merge_brief_opposing_actions and brief_window_seconds must "
            "both be > 0 to enable the brief-opposing-merge rule, or both 0 to disable. "
            f"Got merge_brief_opposing_actions={merge_brief_opposing_actions}, "
            f"brief_window_seconds={brief_window_seconds}."
        )
    invalid_defensive = [t for t in defensive_transition_types if t not in spadlconfig.actiontype_id]
    if invalid_defensive:
        raise ValueError(
            f"add_possessions: defensive_transition_types contains unknown action types: "
            f"{sorted(invalid_defensive)}. Valid types: {sorted(spadlconfig.actiontype_id.keys())}"
        )
```

Note: validation references `spadlconfig.actiontype_id` — but in the atomic file this is `silly_kicks.atomic.spadl.config` (imported at top as `from . import config as spadlconfig`). The atomic vocabulary differs from the standard one. Validation messages will reflect atomic types only.

- [ ] **Step 3: Pass new parameters from `add_possessions` to both `_compute_possessions` call sites**

Find both call sites:

```python
    if not is_card.any():
        # Fast path: no cards → identical algorithm to standard SPADL.
        return _compute_possessions(sorted_actions, max_gap_seconds, retain_on_set_pieces)

    # Slow path: drop cards, compute boundaries on the reduced subset,
    # then forward-fill card rows within game.
    non_card_idx = np.where(~is_card)[0]
    non_card_subset = sorted_actions.iloc[non_card_idx].reset_index(drop=True)
    non_card_with_pids = _compute_possessions(non_card_subset, max_gap_seconds, retain_on_set_pieces)
```

Replace with:

```python
    if not is_card.any():
        # Fast path: no cards → identical algorithm to standard SPADL.
        return _compute_possessions(
            sorted_actions,
            max_gap_seconds,
            retain_on_set_pieces,
            merge_brief_opposing_actions=merge_brief_opposing_actions,
            brief_window_seconds=brief_window_seconds,
            defensive_transition_types=defensive_transition_types,
        )

    # Slow path: drop cards, compute boundaries on the reduced subset,
    # then forward-fill card rows within game.
    non_card_idx = np.where(~is_card)[0]
    non_card_subset = sorted_actions.iloc[non_card_idx].reset_index(drop=True)
    non_card_with_pids = _compute_possessions(
        non_card_subset,
        max_gap_seconds,
        retain_on_set_pieces,
        merge_brief_opposing_actions=merge_brief_opposing_actions,
        brief_window_seconds=brief_window_seconds,
        defensive_transition_types=defensive_transition_types,
    )
```

- [ ] **Step 4: Update `_compute_possessions` signature + body to apply the new rules**

Find:

```python
def _compute_possessions(
    sorted_actions: pd.DataFrame,
    max_gap_seconds: float,
    retain_on_set_pieces: bool,
) -> pd.DataFrame:
    """Compute possession_id on a card-free, pre-sorted Atomic-SPADL frame.

    Mirrors the boundary logic of :func:`silly_kicks.spadl.utils.add_possessions`,
    using atomic-collapsed set-piece names. Mutates and returns *sorted_actions*.
    """
```

Replace with:

```python
def _compute_possessions(
    sorted_actions: pd.DataFrame,
    max_gap_seconds: float,
    retain_on_set_pieces: bool,
    *,
    merge_brief_opposing_actions: int = 0,
    brief_window_seconds: float = 0.0,
    defensive_transition_types: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Compute possession_id on a card-free, pre-sorted Atomic-SPADL frame.

    Mirrors the boundary logic of :func:`silly_kicks.spadl.utils.add_possessions`,
    using atomic-collapsed set-piece names + atomic action vocabulary for the
    PR-S12 opt-in rules. Mutates and returns *sorted_actions*.
    """
```

Find in the body:

```python
    set_piece_ids = {spadlconfig.actiontype_id[name] for name in _ATOMIC_SET_PIECE_RESTART_TYPE_NAMES}
    foul_id = spadlconfig.actiontype_id["foul"]
    is_set_piece = np.isin(type_id, list(set_piece_ids))
    prev_is_foul = prev_type == foul_id
    set_piece_carve_out = retain_on_set_pieces & team_change & is_set_piece & prev_is_foul

    new_possession_mask = game_change | period_change_within_game | gap_timeout | (team_change & ~set_piece_carve_out)
```

Replace with:

```python
    set_piece_ids = {spadlconfig.actiontype_id[name] for name in _ATOMIC_SET_PIECE_RESTART_TYPE_NAMES}
    foul_id = spadlconfig.actiontype_id["foul"]
    is_set_piece = np.isin(type_id, list(set_piece_ids))
    prev_is_foul = prev_type == foul_id
    set_piece_carve_out = retain_on_set_pieces & team_change & is_set_piece & prev_is_foul

    boundary = team_change & ~set_piece_carve_out

    # Rule 2 (PR-S12): defensive_transition_types
    if defensive_transition_types:
        defensive_ids = {spadlconfig.actiontype_id[name] for name in defensive_transition_types}
        is_defensive = np.isin(type_id, list(defensive_ids))
        boundary = boundary & ~is_defensive

    # Rule 1 (PR-S12): brief-opposing-action merge. Vectorized look-ahead.
    if merge_brief_opposing_actions > 0 and brief_window_seconds > 0:
        suppress_at_i = np.zeros(n, dtype=bool)
        suppress_at_k = np.zeros(n, dtype=bool)
        for k in range(1, merge_brief_opposing_actions + 1):
            team_at_k = np.empty(n, dtype=team_id.dtype)
            time_at_k = np.empty(n, dtype=time_seconds.dtype)
            game_at_k = np.empty(n, dtype=game_id.dtype)
            period_at_k = np.empty(n, dtype=period_id.dtype)
            if n > k:
                team_at_k[: n - k] = team_id[k:]
                time_at_k[: n - k] = time_seconds[k:]
                game_at_k[: n - k] = game_id[k:]
                period_at_k[: n - k] = period_id[k:]
            team_at_k[n - k :] = team_id[-1]
            time_at_k[n - k :] = np.inf
            game_at_k[n - k :] = -1
            period_at_k[n - k :] = -1

            same_game_period = (game_at_k == game_id) & (period_at_k == period_id)
            within_time = (time_at_k - time_seconds) <= brief_window_seconds
            team_back = team_at_k == prev_team
            match = boundary & same_game_period & within_time & team_back

            suppress_at_i |= match
            shifted = np.zeros(n, dtype=bool)
            if n > k:
                shifted[k:] = match[: n - k]
            suppress_at_k |= shifted

        boundary = boundary & ~suppress_at_i & ~suppress_at_k

    new_possession_mask = game_change | period_change_within_game | gap_timeout | boundary
```

- [ ] **Step 5: Mirror unit tests in `tests/atomic/test_atomic_add_possessions.py`**

Read the existing atomic test file structure first:

Run: read `tests/atomic/test_atomic_add_possessions.py` lines 1-50 to find the `_make_action` / `_df` equivalents on the atomic side.

Then append three classes mirroring `TestBriefOpposingMerge`, `TestDefensiveTransitions`, `TestMaxGapDefaultIs7Seconds` from the standard-side test file. Reuse the **same test logic** but adapt:
- Atomic action-type names: `pass` → `pass` (same), `interception` → `interception` (same; atomic vocabulary check needed but these 4 types exist in both), `clearance` → `clearance` (same).
- Atomic doesn't have `team_change` set-piece collapse issues (set-piece names are atomic-collapsed but irrelevant for these tests).
- Use the atomic-side `_make_action` builder, which expects atomic columns (`x`, `y`, etc., not `start_x`/`end_x`).

Write the three test classes following the same patterns. Names: `TestAtomicBriefOpposingMerge`, `TestAtomicDefensiveTransitions`, `TestAtomicMaxGapDefaultIs7Seconds`.

- [ ] **Step 6: Run all atomic tests; expect green**

Run: `uv run pytest tests/atomic/test_atomic_add_possessions.py -v --tb=short`

Expected: All atomic tests pass — both existing tests + the 3 new mirror classes.

- [ ] **Step 7: Run the full pytest suite to catch any cross-cutting regressions**

Run: `uv run pytest tests/ -v --tb=short -k "not TestBoundaryAgainstStatsBomb64Match"`

Expected: All tests pass except `TestBoundaryAgainstStatsBomb64Match` (still blocked by HDF5 missing `possession`).

---

### Task 8: Update `scripts/build_worldcup_fixture.py` to preserve `possession`

**Files:**
- Modify: `scripts/build_worldcup_fixture.py`

Two-line change. Adapter includes `possession`; conversion call uses `preserve_native=["possession"]`.

- [ ] **Step 1: Update the adapter to include `possession`**

Find in `_adapt_events_to_silly_kicks_input`:

```python
    return pd.DataFrame(
        [
            {
                "game_id": match_id,
                "event_id": e.get("id"),
                "period_id": e.get("period"),
                "timestamp": e.get("timestamp"),
                "team_id": (e.get("team") or {}).get("id"),
                "player_id": (e.get("player") or {}).get("id"),
                "type_name": (e.get("type") or {}).get("name"),
                "location": e.get("location"),
                "extra": {k: v for k, v in e.items() if k not in _TOP_LEVEL_KEYS},
            }
            for e in events
        ]
    )
```

Replace with:

```python
    return pd.DataFrame(
        [
            {
                "game_id": match_id,
                "event_id": e.get("id"),
                "period_id": e.get("period"),
                "timestamp": e.get("timestamp"),
                "team_id": (e.get("team") or {}).get("id"),
                "player_id": (e.get("player") or {}).get("id"),
                "type_name": (e.get("type") or {}).get("name"),
                "location": e.get("location"),
                "extra": {k: v for k, v in e.items() if k not in _TOP_LEVEL_KEYS},
                # PR-S12 (silly-kicks 2.1.0): preserve native possession sequence
                # for downstream regression validation against add_possessions.
                "possession": e.get("possession"),
            }
            for e in events
        ]
    )
```

- [ ] **Step 2: Update the conversion call to preserve native field**

Find in `_convert_match`:

```python
    actions, _report = statsbomb.convert_to_actions(adapted, home_team_id=home_team_id, preserve_native=None)
    return match_id, actions
```

Replace with:

```python
    actions, _report = statsbomb.convert_to_actions(
        adapted, home_team_id=home_team_id, preserve_native=["possession"]
    )
    return match_id, actions
```

---

### Task 9: Regenerate the WorldCup-2018 HDF5 fixture

**Files:**
- Modify: `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` (regenerated)

- [ ] **Step 1: Run the build script**

Run: `uv run python scripts/build_worldcup_fixture.py --verbose`

Expected: Cache hits on the 64 raw JSONs (already in `tests/datasets/statsbomb/raw/.cache/events/`); ~30s runtime; new HDF5 ~7 MB; validation log shows 64 matches with 100+ actions each. Run in background and poll if the command exceeds the 30s threshold.

- [ ] **Step 2: Verify the HDF5 has the `possession` column**

Run:
```bash
uv run python -c "
import pandas as pd
store = pd.HDFStore('tests/datasets/statsbomb/spadl-WorldCup-2018.h5', mode='r')
keys = [k for k in store.keys() if k.startswith('/actions/game_')]
df = store.get(keys[0])
print(f'first match cols: {list(df.columns)}')
print(f'possession dtype: {df[\"possession\"].dtype}')
print(f'possession non-null: {df[\"possession\"].notna().sum()} / {len(df)}')
store.close()
"
```

Expected: column list includes `possession`, dtype is `Int64` or `int64`, non-null count is high (synthetic dribbles have NaN possession).

- [ ] **Step 3: Run the 64-match HDF5 regression test; expect all green**

Run: `uv run pytest tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBomb64Match -v --tb=short`

Expected: 65 tests (1 sentinel + 64 parametrized). All green at `recall>=0.83 AND precision>=0.30`. Per the probe, R_min across 64 matches at gap=7.0 is 0.854, P_min is 0.350 — comfortable margin above gates.

- [ ] **Step 4: Run the 5 existing prediction tests to confirm no breakage from new column**

Run: `uv run pytest tests/vaep/test_vaep.py::test_predict tests/vaep/test_vaep.py::test_predict_with_missing_features tests/test_xthreat.py::test_predict tests/test_xthreat.py::test_predict_with_interpolation tests/atomic/test_atomic_vaep.py::test_predict -v`

Expected: All 5 tests pass; the new `possession` column is irrelevant to these tests (they project to feature columns).

---

### Task 10: Update `add_possessions` docstrings (standard + atomic)

**Files:**
- Modify: `silly_kicks/spadl/utils.py:558-645` (standard docstring)
- Modify: `silly_kicks/atomic/spadl/utils.py:179-247` (atomic docstring)

- [ ] **Step 1: Replace the standard `add_possessions` docstring's "Empirical baselines" paragraph + add new params section**

Find in `silly_kicks/spadl/utils.py` (currently around lines 580-602):

```python
    The carve-out is approximate (StatsBomb's proprietary possession rules
    capture additional context like merging brief opposing-team actions
    back into the containing possession). Empirically against StatsBomb
    open-data the heuristic achieves:

        - Boundary recall: ~0.93 — every real possession boundary is detected.
        - Boundary precision: ~0.42 — the heuristic emits ~2x more boundaries
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
    possession_id. See :func:`boundary_metrics` for downstream
    measurement.
```

Replace with:

```python
    The carve-out is approximate (StatsBomb's proprietary possession rules
    capture additional context like merging brief opposing-team actions
    back into the containing possession). At the default
    ``max_gap_seconds=7.0`` (PR-S12, 2.1.0) and all opt-in rules disabled,
    the heuristic empirically achieves on 64 StatsBomb WorldCup-2018 matches:

        - Boundary recall: ~0.94 (worst-match 0.85).
        - Boundary precision: ~0.44 (worst-match 0.35).
        - Boundary F1: ~0.60.

    Recall is the meaningful metric for downstream consumers — possessions
    detected by the heuristic correspond to real possession changes. The
    precision gap reflects the algorithm class, not a defect. Consumers
    needing strict StatsBomb-equivalent semantics should use the native
    possession_id where available; the heuristic is a possession proxy
    for sources without one (Wyscout, Sportec, Metrica, etc.).

    Opt-in precision-improvement rules (PR-S12, 2.1.0)
    --------------------------------------------------
    Three opt-in keyword-only parameters trade precision for recall on the
    same algorithm class. Measured on 64 WC-2018 matches:

    +------------------------------------------------+-----------+----------+--------+--------+
    | Setting                                        | P_mean    | R_mean   | F1     | R_min  |
    +================================================+===========+==========+========+========+
    | (default, all rules off)                       | 0.439     | 0.939    | 0.597  | 0.854  |
    +------------------------------------------------+-----------+----------+--------+--------+
    | ``defensive_transition_types=("interception",  | 0.461     | 0.915    | 0.612  | 0.854  |
    |   "clearance")``                               |           |          |        |        |
    +------------------------------------------------+-----------+----------+--------+--------+
    | ``merge_brief_opposing_actions=2,              | 0.483     | 0.910    | 0.630  | 0.843  |
    |   brief_window_seconds=2.0``                   |           |          |        |        |
    +------------------------------------------------+-----------+----------+--------+--------+
    | ``merge_brief_opposing_actions=3,              | 0.530     | 0.882    | 0.662  | 0.811  |
    |   brief_window_seconds=3.0`` (R_min < 0.85!)   |           |          |        |        |
    +------------------------------------------------+-----------+----------+--------+--------+

    See :func:`boundary_metrics` for downstream measurement.

    .. versionchanged:: 2.1.0
        ``max_gap_seconds`` default changed from 5.0 to 7.0 — empirically
        Pareto-optimal at the per-match recall floor R_min >= 0.85 (now
        relaxed to R_min >= 0.83 in CI to absorb pandas/numpy version
        drift). To restore 1.x-2.0.x behavior, pass ``max_gap_seconds=5.0``
        explicitly.

    .. versionadded:: 2.1.0
        ``merge_brief_opposing_actions`` + ``brief_window_seconds`` (paired)
        — opt-in brief-opposing-action merge rule. Both must be > 0 to
        enable, both 0 to disable. ``ValueError`` if exactly one is > 0.

    .. versionadded:: 2.1.0
        ``defensive_transition_types`` — opt-in defensive-transition rule.
        Action types listed do not trigger team-change boundaries on their
        own. Must be a subset of :attr:`silly_kicks.spadl.config.actiontypes`.
```

- [ ] **Step 2: Update the standard `add_possessions` Parameters section**

Find:

```python
    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action stream. Must contain ``game_id``, ``period_id``,
        ``action_id``, ``time_seconds``, ``team_id``, ``type_id``. Other
        columns are preserved unchanged.
    max_gap_seconds : float, default 5.0
        Time-gap threshold (seconds) above which a new possession starts
        even if the team hasn't changed. Set to ``float("inf")`` to disable.
    retain_on_set_pieces : bool, default True
        Whether to apply the foul-then-set-piece carve-out (see Algorithm).
```

Replace with:

```python
    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action stream. Must contain ``game_id``, ``period_id``,
        ``action_id``, ``time_seconds``, ``team_id``, ``type_id``. Other
        columns are preserved unchanged.
    max_gap_seconds : float, default 7.0
        Time-gap threshold (seconds) above which a new possession starts
        even if the team hasn't changed. Set to ``float("inf")`` to disable.
        Default changed from 5.0 in silly-kicks 2.1.0 (see Versionchanged).
    retain_on_set_pieces : bool, default True
        Whether to apply the foul-then-set-piece carve-out (see Algorithm).
    merge_brief_opposing_actions : int, default 0
        Maximum number of consecutive opposing-team actions to merge back
        into the containing possession. Both this AND ``brief_window_seconds``
        must be > 0 to enable; both 0 disables. See "Opt-in precision-
        improvement rules" above.
    brief_window_seconds : float, default 0.0
        Time window (seconds) for the brief-opposing-action merge rule.
        See ``merge_brief_opposing_actions`` for activation pairing.
    defensive_transition_types : tuple[str, ...], default ()
        Action type names that should NOT trigger team-change boundaries
        on their own. Must be a subset of
        :attr:`silly_kicks.spadl.config.actiontypes`. Empty tuple disables.
        Recommended: ``("interception", "clearance")``.
```

- [ ] **Step 3: Mirror the docstring changes in atomic `add_possessions`**

Apply the same structural updates to `silly_kicks/atomic/spadl/utils.py::add_possessions`:
- Same baseline-paragraph rewrite (atomic-vocabulary disclaimers as needed).
- Same opt-in-rules section with the same measurement table (the rules behave identically; the recommendation is portable).
- Same Parameters section additions for the 3 new params + max_gap_seconds default change.
- Same `versionchanged` / `versionadded` directives.

The atomic docstring is shorter than the standard one and references atomic vocabulary; preserve those references. Action-type names in the recommendation table (`interception`, `clearance`) are valid in both vocabularies.

- [ ] **Step 4: Verify docstring rendering**

Run:
```bash
uv run python -c "from silly_kicks.spadl.utils import add_possessions; help(add_possessions)" | head -60
```

Expected: docstring renders cleanly; tables align; no Sphinx-directive syntax errors visible in plain-text help.

---

### Task 11: Update `CHANGELOG.md`, `TODO.md`, `pyproject.toml`

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `TODO.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Read current CHANGELOG.md head to align style**

Run: read `CHANGELOG.md` lines 1-50.

- [ ] **Step 2: Insert new `## [2.1.0]` entry at the top of the changelog (after the "Unreleased" section if any)**

```markdown
## [2.1.0] - 2026-04-29

### Breaking

- **`add_possessions` default for `max_gap_seconds` changed from 5.0 to 7.0** in
  both `silly_kicks.spadl.add_possessions` and
  `silly_kicks.atomic.spadl.add_possessions`. Empirically Pareto-optimal at the
  per-match recall floor on 64 StatsBomb WorldCup-2018 matches (full data:
  `docs/superpowers/specs/2026-04-29-add-possessions-precision-improvement-design.md`).
  Same input DataFrame produces different `possession_id` values for any pair
  of actions where the time gap is in `[5, 7)` seconds AND the team did not
  change.

  **Opt-out:** explicit `add_possessions(actions, max_gap_seconds=5.0)`.

### Added

- **`silly_kicks.spadl.add_possessions` (and atomic counterpart)** new opt-in
  keyword-only parameters for precision-improvement rules:

  - `merge_brief_opposing_actions: int = 0` + `brief_window_seconds: float = 0.0`
    (paired) — brief-opposing-action merge rule. Suppresses team-change
    boundaries when team B has 1..N consecutive actions sandwiched between
    team A actions within the time window.
  - `defensive_transition_types: tuple[str, ...] = ()` — defensive-transition
    rule. Listed action types do not trigger team-change boundaries on their
    own. Recommended: `("interception", "clearance")`.

  All defaults disable the rules, preserving 2.0.x algorithmic behavior except
  for the `max_gap_seconds` default change above.

- **`tests/datasets/statsbomb/spadl-WorldCup-2018.h5`** regenerated with
  `preserve_native=["possession"]` — the 64-match HDF5 fixture is now a
  reusable regression corpus for `add_possessions`. New file size ~7 MB
  (was ~6 MB).

- **`tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBomb64Match`**
  64-match parametrized regression gate complementing the existing 3-fixture
  cross-competition gate.

### Changed

- **`tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBombNative`**
  per-match recall threshold lowered from 0.85 to 0.83. Absorbs the slightly
  reduced recall margin at `max_gap_seconds=7.0` (worst observed across 64
  matches: R_min=0.854) plus pandas/numpy version-drift safety margin.

### Behavior baselines

`add_possessions` empirical performance at the new default (no opt-in rules,
64 WC-2018 matches):

| Metric | Mean | sd | Min |
|---|---|---|---|
| Precision | 0.439 | 0.035 | 0.350 |
| Recall | 0.939 | 0.023 | 0.854 |
| F1 | 0.597 | — | — |

(Compare to 2.0.x at `max_gap_seconds=5.0`: P=0.412, R=0.950, F1=0.574.)

Recommended opt-in settings: see `add_possessions` docstring + spec.
```

- [ ] **Step 3: Update `TODO.md` — close the PR-S12 row**

Find in `TODO.md` (lines 19-23):

```markdown
## Open PRs

| # | Size | Item | Context |
|---|------|------|---------|
| PR-S12 | Medium-Large | `add_possessions` algorithmic precision improvement | Close the precision gap from ~42% toward 60-70% via brief-opposing-action merge rule, defensive-action class, and/or spatial continuity check. Plus re-measure `max_gap_seconds` parameter sweep using the `boundary_metrics` utility before changing the default. New parameters likely: `merge_brief_opposing_actions`, `brief_action_window_seconds`. Atomic-SPADL counterpart must mirror any semantic change. The 64-match WorldCup HDF5 from PR-S9 is available for parameter sweeping. Re-numbered from PR-S11 (which became the converter identifier conventions / sportec tackle override removal work, shipped in silly-kicks 2.0.0). Original numbering: was PR-S10 in PR-S8 era. |
```

Replace with:

```markdown
## Open PRs

(none currently queued — PR-S12 shipped in silly-kicks 2.1.0)
```

- [ ] **Step 4: Bump `pyproject.toml` version**

Find:

```toml
version = "2.0.0"
```

Replace with:

```toml
version = "2.1.0"
```

---

### Task 12: Verification gates (full pre-commit sweep)

- [ ] **Step 1: Pin tooling to CI versions**

Run: `uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113`

Expected: clean install message.

- [ ] **Step 2: Lint check**

Run: `uv run ruff check silly_kicks/ tests/ scripts/`

Expected: zero errors. If errors, fix and re-run.

- [ ] **Step 3: Format check**

Run: `uv run ruff format --check silly_kicks/ tests/ scripts/`

Expected: zero changes needed. If formatting issues, run `uv run ruff format silly_kicks/ tests/ scripts/` and re-check.

- [ ] **Step 4: Type check**

Run: `uv run pyright silly_kicks/`

Expected: zero errors. If errors, address. Pay special attention to: tuple type annotations, `np.ndarray` returns, the new helper signature.

- [ ] **Step 5: Full pytest suite**

Run: `uv run pytest tests/ -v --tb=short`

Expected:
- 0 failures
- 0 errors
- All `e2e`-marked tests still skip (none affected by PR-S12)
- 64 new HDF5 parametrized cases pass
- 3 + 64 + ~24 new unit tests = ~91 new test cases beyond existing suite, all green

- [ ] **Step 6: HDF5 size sanity check**

Run: `uv run python -c "import os; print(f'HDF5 size: {os.path.getsize(\"tests/datasets/statsbomb/spadl-WorldCup-2018.h5\") / 1024 / 1024:.1f} MB')"`

Expected: ~7 MB (was 6 MB; growth from extra column). Stays well under 50 MB warn threshold.

---

### Task 13: Run /final-review skill

Per `feedback_commit_policy` memory: `/final-review` is mandatory before the single commit, not just pre-PR.

- [ ] **Step 1: Invoke /final-review**

Use the Skill tool to launch `mad-scientist-skills:final-review` (or whichever final-review skill is configured for the project).

Expected: skill produces a quality report. Address any actionable findings inline.

- [ ] **Step 2: Address findings**

If `/final-review` flags any issues, fix them. Re-run `/final-review` after each fix until it passes.

---

### Task 14: Single-commit gate (user approval required)

Per `feedback_commit_policy` memory: literally ONE commit per branch; explicit approval before that commit.

- [ ] **Step 1: Confirm branch state**

Run: `git status` and `git log --oneline -5`

Expected: on `feat/add-possessions-precision-improvement` (or user-chosen variant). Working tree shows the expected modifications. No previous WIP commits on this branch (branch points at `main` HEAD).

- [ ] **Step 2: Show user the diff summary**

Run: `git diff --stat main...HEAD` (or `git diff --stat` if branch is uncommitted)

Expected: shows the PR-S12 file changes per the spec's File Structure table, plus the regenerated HDF5 binary.

- [ ] **Step 3: Wait for user approval to commit**

User explicitly approves. Do NOT commit without approval.

- [ ] **Step 4: Single commit**

```bash
git add -A
git commit -s -m "$(cat <<'EOF'
feat(spadl): add_possessions precision-improvement rules + max_gap default 7.0 -- silly-kicks 2.1.0

PR-S12. Add three opt-in precision-improvement parameters to add_possessions
(standard + atomic): merge_brief_opposing_actions / brief_window_seconds
(paired) for brief-opposing-action merge; defensive_transition_types for
defensive-class transitions. All disabled by default.

Behavior break: max_gap_seconds default changed 5.0 -> 7.0 (empirically
Pareto-optimal at per-match recall floor on 64 WC-2018 matches). One-line
opt-out preserves 1.x/2.0.x behavior.

Regenerated tests/datasets/statsbomb/spadl-WorldCup-2018.h5 with
preserve_native=["possession"] -- the 64-match fixture is now a reusable
regression corpus. New TestBoundaryAgainstStatsBomb64Match parametrized gate
complements the existing 3-fixture cross-competition gate. Recall floor
relaxed 0.85 -> 0.83 to absorb the slight margin reduction at gap=7.0.

Spec: docs/superpowers/specs/2026-04-29-add-possessions-precision-improvement-design.md
Plan: docs/superpowers/plans/2026-04-29-add-possessions-precision-improvement.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Verify the commit**

Run: `git log --oneline -2 && git diff HEAD^ HEAD --stat | tail -20`

Expected: one new commit on the branch; diff stats match the File Structure table.

---

### Task 15: Push, PR, merge, tag (user approval required at each step)

- [ ] **Step 1: User approval to push**

Wait for user explicit approval.

- [ ] **Step 2: Push branch**

Run: `git push -u origin feat/add-possessions-precision-improvement`

Expected: clean push.

- [ ] **Step 3: User approval to open PR**

Wait for user approval.

- [ ] **Step 4: Open PR**

```bash
gh pr create --title "feat(spadl): add_possessions precision-improvement rules + max_gap default 7.0 -- silly-kicks 2.1.0" --body "$(cat <<'EOF'
## Summary

PR-S12. Adds three opt-in precision-improvement parameters to `add_possessions` (standard + atomic): brief-opposing-action merge, defensive-action transitions. All disabled by default.

**Behavior break**: `max_gap_seconds` default changed 5.0 → 7.0 (empirically Pareto-optimal at the per-match recall floor on 64 WC-2018 matches). One-line opt-out preserves 1.x/2.0.x behavior.

Regenerated `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` with `preserve_native=["possession"]` — the 64-match fixture is now a reusable regression corpus. New `TestBoundaryAgainstStatsBomb64Match` parametrized gate complements the existing 3-fixture gate. Recall floor relaxed 0.85 → 0.83 to absorb the slight margin reduction at gap=7.0.

## Test plan

- [x] Full local pytest suite (`uv run pytest tests/ -v`)
- [x] ruff check + format check (CI-pinned 0.15.7)
- [x] pyright (CI-pinned 1.1.395 + pandas-stubs 2.3.3.260113)
- [x] HDF5 regenerated with `possession` column; 5 existing prediction tests still pass
- [x] 64 new HDF5 parametrized cases pass at recall>=0.83 / precision>=0.30
- [x] `/final-review` skill clean
- [ ] CI matrix green on all platforms (3.10/3.11/3.12 ubuntu + 3.12 windows)

## Lakehouse coordination

`silly-kicks>=2.1.0,<3.0` upgrade is straightforward; `max_gap_seconds=5.0` opt-out available per-call. No mandatory action.

## Spec

`docs/superpowers/specs/2026-04-29-add-possessions-precision-improvement-design.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: User approval to merge**

Wait for user approval. Confirm CI matrix is green before merging.

- [ ] **Step 6: Merge (squash + delete branch)**

Run: `gh pr merge --admin --squash --delete-branch`

- [ ] **Step 7: User approval to tag**

Wait for user approval.

- [ ] **Step 8: Tag and push**

```bash
git checkout main
git pull
git tag v2.1.0
git push origin v2.1.0
```

Expected: tag push fires the PyPI auto-publish workflow. Verify the publish run completes successfully on GitHub Actions.

- [ ] **Step 9: Verify PyPI release**

Run: `pip index versions silly-kicks` (or check `https://pypi.org/project/silly-kicks/`)

Expected: 2.1.0 appears in the version list.

- [ ] **Step 10: Cleanup throwaway probe**

Run: `rm /c/Users/Karsten/AppData/Local/Temp/probe_pr_s12.py /c/Users/Karsten/AppData/Local/Temp/probe_pr_s12_output.txt`

Throwaway probe per `feedback_databricks_readonly_probe` pattern.

---

## Self-Review Checklist

Spec coverage check (against `docs/superpowers/specs/2026-04-29-add-possessions-precision-improvement-design.md`):

- §2 Goal 1 (3 opt-in params standard side) → Tasks 1, 4, 5, 6
- §2 Goal 2 (max_gap default 5→7) → Task 4 step 1, Task 7 step 2
- §2 Goal 3 (HDF5 with `possession`) → Tasks 8, 9
- §2 Goal 4 (regression CI gate extension) → Task 2 (3-JSON threshold), Task 9 step 3 (64-match)
- §2 Goal 5 (recall floor 0.85→0.83) → Task 2 step 1
- §2 Goal 6 (docstring updates with empirical numbers + recommended-settings table) → Task 10
- §2 Goal 7 (atomic-SPADL parity) → Task 7
- §2 Goal 8 (version 2.1.0) → Task 11 step 4

§3 Non-goals: spec excludes spatial continuity, multi-rule defaults, `add_possessions_strict`, BoundaryMetrics changes, precision threshold change, lakehouse touch, 3.0.0 bump. Plan respects each non-goal.

§4.6 Test infrastructure (3 unit-test classes): Task 1.
§4.6 Test infrastructure (HDF5 64-match class): Task 2 + Task 9.
§4.7 Docstring updates: Task 10.
§4.8 Helper extraction: Task 3.

§7 TDD ordering: closely tracks the plan order Tasks 1-12.

§8 Verification gates: Task 12.

§9 Commit cycle: Tasks 14-15.

Type-consistency check: parameter names match across signatures, validation messages, docstrings, and tests:
- `merge_brief_opposing_actions: int = 0` ✓
- `brief_window_seconds: float = 0.0` ✓
- `defensive_transition_types: tuple[str, ...] = ()` ✓
- `max_gap_seconds: float = 7.0` ✓
- helper `_compute_possession_boundaries` (standard) vs `_compute_possessions` (atomic) — different names per existing convention; both files document why ✓
