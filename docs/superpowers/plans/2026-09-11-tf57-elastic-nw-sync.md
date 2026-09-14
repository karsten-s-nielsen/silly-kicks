# TF-57 — ELASTIC v2 (Extended Needleman–Wunsch) Event↔Tracking Sync — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **This plan is UNCOMMITTED and execution is owner-gated — do not start until the plan is approved and Karsten gives the go-ahead.**

**Goal:** Replace TF-43's greedy MLSA-2025 event↔tracking aligner with a clean-room reimplementation of ELASTIC v2's extended Needleman–Wunsch alignment, exposed through the existing `elastic_*` surface (values move + 3 additive reception columns) and as an alternative strategy behind the canonical `(pointers, LinkReport)` linkage contract.

**Architecture:** One pure NW engine in a rewritten `tracking/_elastic_sync.py` (scipy peak detection + a numpy DP; no neural nets), fed by candidate ball-touch frames + a virtual-termination-event-enriched SPADL sequence, run per possession-episode. Two surfaces over the engine: the upgraded elastic mart producer (`align_events_to_frames`/`add_elastic_sync`/`elastic_sync_xfns` + atomic mirror) and a new `link_actions_to_frames_elastic` returning the exact pointer/`LinkReport` schema (a `links=` drop-in for the whole `add_*` family). The guarded time-based `link_actions_to_frames` is untouched and stays the default.

**Tech Stack:** Python, pandas, numpy, scipy (`scipy.signal.find_peaks`/`argrelextrema`), `silly_kicks.id_compat`, `silly_kicks.spadl` (`add_possessions`, `config`). No new runtime dependency (scipy is already a runtime dep).

**Spec:** `docs/superpowers/specs/2026-09-11-tf57-elastic-nw-sync-design.md` (**rev 3**, independently reviewed → APPROVE WITH FOLLOW-UPS, all addressed; see spec **§16**). The plan argues from the spec; executors read both. Tasks 1–12 remain the as-built baseline (implemented, impl-reviewed, REQUEST CHANGES); **Tasks 13–19 (rev 3) are the fix cycle** — close the exact-frame gap (a scoring transcription divergence, spec §16.2), rebuild the oracle, and fold in the impl-review findings. All formulas are in spec §3.1 (the *intended* maps) with §16.4 the correction.

**Plan review:** **rev 3** (2026-09-12) — after the implementation review (`…-impl.md`, REQUEST CHANGES) + the owner's Goal-5 "gold standard / investigate / do more work" ruling, this plan gains the **rev-3 fix-cycle task block (Tasks 13–19)**: a white-box score-peak harness (landed RED), per-feature scoring re-derivation against the paper, oracle rebuild (real roster + goal span), the folded findings (IMPL-01/02/03/05), the corrected attribution, and DGX 3-match acceptance. The spec-r3 review (`…-spec-r3.md`, APPROVE WITH FOLLOW-UPS) is fully addressed in spec §16 + §16.9 (SO-1 signed off). **rev 2** — addressed the 2026-09-11 independent *plan* review (SHOULD-FIX Task 6 Step 4 rescoped + "intermediate red is expected" Global Constraint; CONSIDER Task 7 test file + docstring-example steps in Tasks 6 & 8).

## Global Constraints

- **ONE commit, no micro-commits (CLAUDE.md — OVERRIDES the writing-plans default of per-task commits).** No task ends in `git commit`. Each task ends by running its tests + affected gates green. The single commit happens only in the final task, after the **full non-e2e suite is green** AND Karsten approves that specific diff. Docs/data/tests/code all land together.
- **Intermediate red is EXPECTED under the one-commit model — scope every Step-4 to its OWN task's tests.** `add_elastic_sync`/`elastic_sync_xfns` in `features.py:6829-6835` still pass the greedy kwargs removed in Task 1, so those functions raise `TypeError` and their tests (`TestAddElasticSync`/`TestElasticSyncXfns`, `test_elastic_sync.py:415/:453`) are **RED from Task 1 until Task 8** fixes `features.py`. A per-task executor must NOT treat these public-surface reds as its own regression (the rubric's exact misattribution trap). The whole `test_elastic_sync.py` file goes green at **Task 8 Step 4**; the whole non-e2e suite at **Task 12 Step 6**. Nothing is committed mid-flight, so intermediate red is harmless.
- **No worktrees.** Work on the existing feature branch `feat/tf57-elastic-nw-sync` (off `main`).
- **Clean-room / licence (hard):** reimplement from the paper (arXiv:2608.30227) only — never lift ELASTIC's MPL-2.0 code, incl. its `compute_sync_accuracy` (reimplement the trivial `|pred−gt|≤N` metric). The DGX MPL clone stays off-repo. Committed oracle data is **CC BY 4.0** (Sportec Open DFL / Bassek et al. 2025 + ELASTIC re-annotation) — **grant evidenced** from the ELASTIC repo README (verified 2026-09-11); attribution in NOTICE + fixture README. **Claude produces the slice on the DGX** (I have box access + context); no owner step.
- **ADR-019 everywhere:** every id comparison/join/dict-key uses `id_compat` (`canonical_id` / `canonical_id_series` / `ids_equal` / `same_id`) — never raw `==`, never `astype(str)`. This is the exact seam of the constant-0.6 bug (`_elastic_sync.py:186-191`).
- **ADR-068:** no rescan-in-loop; group with `silly_kicks._frame_index.group_rows`.
- **ADR-009:** `ElasticSyncParams` frozen, `for_provider` empty; all thresholds are the paper's intent-set constants; validation reported-not-gated except the committed regression floor.
- **TF57-SPEC-08 (from R2 review):** the `_enforce_link_coverage` time-base-hint suppression MUST be an opt-in flag defaulting off, so the default time-based linker's behaviour stays **byte-identical**.
- **Lint at CI scope:** `python -m ruff check silly_kicks/ tests/ scripts/` + `python -m ruff format --check ...`; `python -m pyright` bare. `python -m pytest tests/ -m "not e2e"`. `tests/tracking/` HANGS without `--benchmark-skip`.
- **Numbers (ADR ≈092, PR-S###, release ≈4.113.0) are provisional** — assigned at commit-prep after `git fetch && git merge origin/main` (final task).

---

### Task 1: `ElasticSyncParams` — NW parameters (greedy removed)

**Files:**
- Modify: `silly_kicks/tracking/_elastic_sync.py` (the `ElasticSyncParams` dataclass, ~lines 29-44)
- Test: `tests/tracking/test_elastic_sync.py` (rewrite `TestElasticSyncParams`)

**Interfaces:**
- Produces: `ElasticSyncParams(frame_rate=25, touch_distance_m=3.0, ball_height_max_m=4.0, accel_clip_max=30.0, slope_window_seconds=0.2, slope_clip_mps=7.0, repeat_penalty=-0.1, event_gap_penalty=0.0, candidate_gap_penalty=0.0, min_confidence=0.5)` — frozen dataclass consumed by every engine helper.

- [ ] **Step 1: Write the failing test** (replace the old `TestElasticSyncParams`)

```python
class TestElasticSyncParams:
    def test_frozen(self):
        params = ElasticSyncParams()
        with pytest.raises(AttributeError):
            params.min_confidence = 0.9  # type: ignore[misc]

    def test_defaults_are_paper_constants(self):
        p = ElasticSyncParams()
        assert p.frame_rate == 25
        assert p.touch_distance_m == pytest.approx(3.0)
        assert p.ball_height_max_m == pytest.approx(4.0)
        assert p.accel_clip_max == pytest.approx(30.0)
        assert p.slope_window_seconds == pytest.approx(0.2)
        assert p.slope_clip_mps == pytest.approx(7.0)
        assert p.repeat_penalty == pytest.approx(-0.1)
        assert p.event_gap_penalty == pytest.approx(0.0)
        assert p.candidate_gap_penalty == pytest.approx(0.0)
        assert p.min_confidence == pytest.approx(0.5)

    def test_greedy_fields_removed(self):
        p = ElasticSyncParams()
        assert not hasattr(p, "accel_weight")
        assert not hasattr(p, "proximity_weight")
        assert not hasattr(p, "window_seconds")
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_elastic_sync.py::TestElasticSyncParams -v --benchmark-skip`
Expected: FAIL (old fields still present / new fields missing).

- [ ] **Step 3: Rewrite the dataclass**

```python
@dataclass(frozen=True)
class ElasticSyncParams:
    """Parameters for the ELASTIC v2 (extended Needleman-Wunsch) sync.

    All values are the paper's intent-set constants (arXiv:2608.30227);
    `for_provider` is intentionally absent (ADR-009).
    """
    frame_rate: int = 25
    touch_distance_m: float = 3.0
    ball_height_max_m: float = 4.0
    accel_clip_max: float = 30.0
    slope_window_seconds: float = 0.2
    slope_clip_mps: float = 7.0
    repeat_penalty: float = -0.1
    event_gap_penalty: float = 0.0
    candidate_gap_penalty: float = 0.0
    min_confidence: float = 0.5
```

Update the module docstring header (Kim et al. **2026**, CIKM, arXiv:2608.30227; "See NOTICE for full bibliographic citations.").

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/tracking/test_elastic_sync.py::TestElasticSyncParams -v --benchmark-skip`
Expected: PASS.

---

### Task 2: Ball kinematics + candidate ball-touch frames

**Files:**
- Modify: `silly_kicks/tracking/_elastic_sync.py` (generalize `extract_ball_features`; add `_ball_kinematics`, `_detect_candidate_frames`, a `Candidate` value object)
- Test: `tests/tracking/test_elastic_sync.py` (keep `TestExtractBallFeatures`; add `TestCandidateDetection`)

**Interfaces:**
- Produces:
  - `extract_ball_features(frames, *, params=None) -> pd.DataFrame` — UNCHANGED public columns (`game_id, period_id, frame_id, ball_x, ball_y, ball_speed, ball_accel`); keep the existing tests passing.
  - `_ball_kinematics(frames, *, params) -> pd.DataFrame` — per `(game_id, period_id, frame_id)`: `ball_x/y/z, ball_speed, ball_accel, boundary_dist` (min distance to any pitch line, using `spadlconfig` field length/width). `z` = NaN where absent.
  - `Candidate = namedtuple("Candidate", "game_id period_id frame_id players")` where `players` is a `tuple[str, ...]` of `canonical_id` player ids within `touch_distance_m` of the ball at that frame (∧ ball height ≤ `ball_height_max_m` where `z` present).
  - `_detect_candidate_frames(frames, *, params) -> dict[tuple, list[Candidate]]` keyed by `(canonical game_id, period_id)`, sorted by `frame_id`.

- [ ] **Step 1: Write failing tests**

```python
class TestCandidateDetection:
    def test_accel_spike_is_a_candidate(self):
        # a ball that jumps velocity at frame 10 with a player within 3 m there
        frames = _make_touch_fixture(spike_frame=10, toucher="p1_0", dist_m=1.0)
        params = ElasticSyncParams()
        cands = _detect_candidate_frames(frames, params=params)
        by_frame = {c.frame_id: c for c in cands[("1", 1)]}
        assert 10 in by_frame
        assert "p1_0" in by_frame[10].players

    def test_far_player_excluded_by_feasibility_gate(self):
        frames = _make_touch_fixture(spike_frame=10, toucher="p1_0", dist_m=9.0)
        cands = _detect_candidate_frames(frames, params=ElasticSyncParams())
        by_frame = {c.frame_id: c for c in cands.get(("1", 1), [])}
        # no player within 3 m -> that frame yields no feasible (frame, player) pair
        assert "p1_0" not in by_frame.get(10, _EmptyCand()).players

    def test_high_ball_excluded_when_z_present(self):
        frames = _make_touch_fixture(spike_frame=10, toucher="p1_0", dist_m=1.0, ball_z=6.0)
        cands = _detect_candidate_frames(frames, params=ElasticSyncParams())
        assert 10 not in {c.frame_id for c in cands.get(("1", 1), [])}

    def test_z_absent_does_not_gate(self):
        frames = _make_touch_fixture(spike_frame=10, toucher="p1_0", dist_m=1.0, ball_z=np.nan)
        cands = _detect_candidate_frames(frames, params=ElasticSyncParams())
        assert 10 in {c.frame_id for c in cands.get(("1", 1), [])}
```

Add a `_make_touch_fixture(...)` helper to the test module: a short single-period frame set where the ball is stationary then accelerates at `spike_frame`, with a named player at `dist_m` from the ball at that frame, and a configurable ball `z`.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_elastic_sync.py::TestCandidateDetection -v --benchmark-skip`
Expected: FAIL (`_detect_candidate_frames` undefined).

- [ ] **Step 3: Implement**

- `_ball_kinematics`: reuse the vectorized velocity/accel logic from the current `extract_ball_features` (finite diff per `(game,period)`), add `boundary_dist = min(x, L-x, y, W-y)` from `spadlconfig` and pass `z` through.
- `_detect_candidate_frames`: per `(game,period)`, get candidate frame indices as the **union** of `scipy.signal.argrelextrema` local minima of (a) per-player ball distance and (b) `boundary_dist`, and (c) local maxima of `ball_accel` (`scipy.signal.find_peaks`). For each candidate frame, compute `players` = canonical ids within `touch_distance_m` (reuse a generalized `_player_ball_distance` matrix) AND (`z` NaN OR `z ≤ ball_height_max_m`). Drop frames with an empty `players` set. Group with `group_rows` (ADR-068); keys via `canonical_id`.
- Keep `extract_ball_features` as a thin public wrapper selecting its 7 columns from `_ball_kinematics` (existing `TestExtractBallFeatures` must stay green).

- [ ] **Step 4: Run to verify pass (incl. the unchanged ball-feature tests)**

Run: `python -m pytest tests/tracking/test_elastic_sync.py::TestCandidateDetection tests/tracking/test_elastic_sync.py::TestExtractBallFeatures -v --benchmark-skip`
Expected: PASS.

---

### Task 3: Feature scoring + actor-membership hard gate

**Files:**
- Modify: `silly_kicks/tracking/_elastic_sync.py` (add `_clip_linear`, per-feature scorers, `_score`)
- Test: `tests/tracking/test_elastic_sync.py` (add `TestScoring`)

**Interfaces:**
- Produces:
  - `_clip_linear(x, x0, x1) -> float` — spec §3.1 `f(x; x0, x1)`.
  - `_score(event, candidate, kin_lookup, dist_lookup, opp_dist_lookup, *, params) -> float` — category-dispatched `s(e,c) ∈ [0,1]`; returns **0.0** if `canonical_id(event.player_id)` ∉ `candidate.players` (hard gate).
  - `EventCategory = Literal["outgoing", "incoming", "minor"]` (set-piece uses the outgoing composition).
- Consumes: `Candidate` (Task 2); an `Event` namedtuple defined in Task 4 (for the test, use a lightweight stub with `player_id`, `category`).

- [ ] **Step 1: Write failing tests**

```python
class TestScoring:
    def test_clip_linear_bounds(self):
        assert _clip_linear(-1, 0, 30) == 0.0
        assert _clip_linear(15, 0, 30) == pytest.approx(0.5)
        assert _clip_linear(40, 0, 30) == 1.0

    def test_actor_not_in_candidate_scores_zero(self):
        cand = Candidate("1", 1, 10, players=("p1_0",))
        ev = _StubEvent(player_id="p2_9", category="outgoing")  # not in players
        assert _score(ev, cand, *_touch_lookups(), params=ElasticSyncParams()) == 0.0

    def test_outgoing_high_when_ball_departs_after_touch(self):
        # actor at the ball at frame 10, ball departs after -> high s_KD+, low pre-slope
        ev = _StubEvent(player_id="p1_0", category="outgoing")
        cand = Candidate("1", 1, 10, players=("p1_0",))
        s = _score(ev, cand, *_departing_ball_lookups(), params=ElasticSyncParams())
        assert 0.0 < s <= 1.0
```

- [ ] **Step 2: Run to verify failure** — `... ::TestScoring -v --benchmark-skip` → FAIL.

- [ ] **Step 3: Implement** the per-feature scorers `s_BA/s_PBD/s_KD±/s_PBDS±/s_OD` exactly per spec §3.1 (clip bounds from `params`), the category compositions (outgoing `¼(BA+PBD+KD⁺+PBDS⁻)`, incoming `¼(BA+PBD+KD⁻+PBDS⁺)`, minor `¼(BA+PBD+KD+OD)`), and the actor-membership gate via `canonical_id`. KD⁺/KD⁻/PBDS use small frame windows around `t_c` (`slope_window_seconds`), reading `dist_lookup`.

- [ ] **Step 4: Run to verify pass** → PASS.

---

### Task 4: Event enrichment — virtual termination events

**Files:**
- Modify: `silly_kicks/tracking/_elastic_sync.py` (add `Event`, `_map_category`, `_enrich_events`)
- Test: `tests/tracking/test_elastic_sync.py` (add `TestEventEnrichment`)

**Interfaces:**
- Produces:
  - `Event = namedtuple("Event", "action_id player_id category kind expected_time")` where `kind ∈ {"real","reception","out","goal"}` (virtual events carry `action_id=pd.NA`, `kind!="real"`, and the receiving/expected actor).
  - `_map_category(type_name) -> EventCategory | None` — the spec §7 table; `None` = excluded (`non_action`, `dribble`, cards-as-results never appear).
  - `_enrich_events(actions_period, *, params) -> list[Event]` — per period (already possession-tagged), the excluded types dropped, virtual events inserted between consecutive real events per spec §7 (reception / out / goal). Uses `same_id`/`ids_differ` for the "different player" test and `add_possessions`' `possession_id` for "same episode".
- Consumes: possession-tagged actions (the assembler in Task 6 calls `add_possessions` once, up front).

- [ ] **Step 1: Write failing tests** (the §7 insertion truth table)

```python
class TestEventEnrichment:
    def test_dribble_and_non_action_excluded(self):
        assert _map_category("dribble") is None
        assert _map_category("non_action") is None
        assert _map_category("pass") == "outgoing"
        assert _map_category("throw_in") == "outgoing"
        assert _map_category("interception") == "incoming"
        assert _map_category("tackle") == "minor"

    def test_reception_inserted_for_same_possession_diff_player(self):
        acts = _events_df([("pass","p1_0",1), ("pass","p1_1",1)])  # (type, player, possession)
        evs = _enrich_events(acts, params=ElasticSyncParams())
        kinds = [e.kind for e in evs]
        assert kinds == ["real", "reception", "real"]
        assert evs[1].player_id == "p1_1"  # reception owned by the next actor

    def test_out_inserted_before_restart(self):
        acts = _events_df([("pass","p1_0",1), ("throw_in","p2_0",2)])
        evs = _enrich_events(acts, params=ElasticSyncParams())
        assert [e.kind for e in evs] == ["real", "out", "real"]

    def test_goal_inserted_after_successful_shot_then_kickoff(self):
        acts = _events_df([("shot","p1_0",1,"success"), ("pass","p2_0",2)])
        evs = _enrich_events(acts, params=ElasticSyncParams())
        assert [e.kind for e in evs] == ["real", "goal", "real"]

    def test_no_insertion_same_player_continuation(self):
        acts = _events_df([("pass","p1_0",1), ("take_on","p1_0",1)])
        evs = _enrich_events(acts, params=ElasticSyncParams())
        assert [e.kind for e in evs] == ["real", "real"]
```

Add `_events_df(...)` helper building a minimal actions frame with `type_name`, `player_id`, `possession_id`, `result_name`, `time_seconds`, `action_id`.

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement** `_map_category` (spec §7 table) and `_enrich_events` (drop excluded, iterate consecutive real events applying the reception/out/goal rules; `expected_time` interpolated between the two bounding real-event times). ADR-018 note: "successful shot" = `type_name=="shot" and result_name=="success"` (own goals are `bad_touch`, not shot-gated).

- [ ] **Step 4: Run to verify pass** → PASS.

---

### Task 5: Extended Needleman–Wunsch DP + backtrack

**Files:**
- Modify: `silly_kicks/tracking/_elastic_sync.py` (add `_needleman_wunsch`)
- Test: `tests/tracking/test_elastic_sync.py` (add `TestNeedlemanWunsch`)

**Interfaces:**
- Produces: `_needleman_wunsch(score_matrix: np.ndarray, *, params) -> list[int | None]` — for `score_matrix[m, n]` (events × candidates), returns a length-`m` list mapping each event to a candidate column index (or `None` if event-gapped). Implements spec §3.1 Eq. 14 (diag-match / event-gap `g_e` / candidate-gap `g_c` / down-match with `repeat_penalty`) + backtrack. Order-preserving by construction.

- [ ] **Step 1: Write failing tests** (hand-built matrices with a known optimum)

```python
class TestNeedlemanWunsch:
    def test_monotone_assignment(self):
        # 3 events, 3 candidates, strong diagonal -> identity assignment
        S = np.array([[0.9, 0.1, 0.0],
                      [0.0, 0.9, 0.1],
                      [0.0, 0.1, 0.9]])
        assert _needleman_wunsch(S, params=ElasticSyncParams()) == [0, 1, 2]

    def test_down_match_one_touch(self):
        # events 0 and 1 both best-match candidate 0 (a one-touch); down-match
        # assigns candidate 0 to both at cost repeat_penalty rather than forcing
        # event 1 onto a bad later candidate.
        S = np.array([[0.9, 0.0],
                      [0.8, 0.05]])
        out = _needleman_wunsch(S, params=ElasticSyncParams())
        assert out == [0, 0]

    def test_order_preserving(self):
        S = np.array([[0.2, 0.9],
                      [0.9, 0.2]])  # event 0 prefers cand 1, event 1 prefers cand 0
        out = _needleman_wunsch(S, params=ElasticSyncParams())
        # cannot assign event0->1 and event1->0 (would violate order); monotone result
        assigned = [c for c in out if c is not None]
        assert assigned == sorted(assigned)

    def test_all_low_scores_event_gaps(self):
        S = np.zeros((2, 3))
        out = _needleman_wunsch(S, params=ElasticSyncParams())
        assert all(c is None for c in out)  # g_e=g_c=0, no positive match
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement** the DP table `F[(m+1)×(n+1)]` + backpointer matrix per Eq. 14, then backtrack `(m,n)→(0,0)` recording per-event candidate (or `None`). Numpy arrays; per-episode sizes are small. (Numba is a reserved optimization — NOT baseline; YAGNI until the perf guard/DGX says otherwise.)

- [ ] **Step 4: Run to verify pass** → PASS.

---

### Task 6: `align_events_to_frames` assembly (episode loop, per-period fit, confidence, reception, 7-col output)

**Files:**
- Modify: `silly_kicks/tracking/_elastic_sync.py` (rewrite `align_events_to_frames`; keep `_fit_frame_time_relationship`)
- Test: `tests/tracking/test_elastic_sync.py` (rewrite `TestAlignEventsToFrames` + `TestAlignEventsNonZeroFrameOrigin`)

**Interfaces:**
- Produces: `align_events_to_frames(actions, frames, *, params=None) -> pd.DataFrame` with the **7 columns** (spec §5.2): `action_id, elastic_frame_id (Int64), elastic_confidence (float64), elastic_error_seconds (float64), elastic_receive_frame_id (Int64), elastic_receive_confidence (float64), elastic_receive_error_seconds (float64)`.
- Consumes: Tasks 2–5 helpers; `spadl.add_possessions`.

**Assembly algorithm:**
1. Empty-input guards → typed empty frame.
2. `possessions = add_possessions(actions)`; `kin/dist/candidates` precomputed once.
3. Per `(game_id, period_id)`: `fit = _fit_frame_time_relationship`; `events = _enrich_events(period_actions, params)`; partition events into **episodes** by `possession_id` (a trailing `out`/`goal` virtual event stays with the episode it terminates). For each episode: candidates = those in `[episode_start_time − margin, episode_end_time + margin]`; build `score_matrix` (`_score` over events × candidates); `assign = _needleman_wunsch(...)`.
4. Map assignments back: for each **real** event → `elastic_frame_id = candidate.frame_id`, `elastic_confidence = s`, `elastic_error_seconds = |frame→time(frame) − action_time|` (via `fit`); confidence `< min_confidence` → NaN (unsynced). For the virtual termination event **following** a real event → that real event's `elastic_receive_*`.
5. Assemble the 7-col frame; dtypes per §5.2.

- [ ] **Step 1: Write failing tests** (rewrite existing; update column set to 7; keep the IDSSE-origin regressions but assert on `elastic_frame_id` in native range; add a reception assertion)

```python
class TestAlignEventsToFrames:
    def test_output_columns(self):
        result = align_events_to_frames(_make_actions(), _make_tracking_frames())
        assert set(result.columns) == {
            "action_id", "elastic_frame_id", "elastic_confidence", "elastic_error_seconds",
            "elastic_receive_frame_id", "elastic_receive_confidence", "elastic_receive_error_seconds",
        }

    def test_confidence_in_unit_interval_or_nan(self):
        r = align_events_to_frames(_make_actions(), _make_tracking_frames())
        v = r["elastic_confidence"].dropna()
        assert (v >= 0).all() and (v <= 1).all()

    def test_reception_populated_for_pass_to_teammate(self):
        # a pass then a different-teammate touch -> the pass gets a reception frame
        frames, actions = _make_pass_reception_fixture()
        r = align_events_to_frames(actions, frames)
        pass_row = r[r["action_id"] == 0].iloc[0]
        assert pd.notna(pass_row["elastic_receive_frame_id"])

    def test_empty_actions(self): ...   # keep
    def test_empty_frames(self): ...    # keep
```

Keep `TestAlignEventsNonZeroFrameOrigin` but drop the greedy `window_seconds` references (use `ElasticSyncParams()`); assert alignments land in the native frame range and `elastic_error_seconds` stays sane. Delete `test_custom_weights` (greedy-only). Replace `test_aligned_frame_within_window` with an NW-appropriate "aligned frame is a real candidate frame" assertion.

- [ ] **Step 2: Run to verify failure** → FAIL.
- [ ] **Step 3: Implement** the assembly per the algorithm above (group_rows for the period/episode loops; `id_compat` on every key). **Also update the `align_events_to_frames` docstring Examples block** (currently `_elastic_sync.py:271-277`) to show the **7-column** output — the doctest `--doctest-modules` sweep is on private modules (skipped) but the public-API-examples gate (spec §9) reads it; keep it a literal block, not an executable `>>>`.
- [ ] **Step 4: Run the Task-6 engine classes** (NOT the public-surface classes — see the "Intermediate red" Global Constraint) → `python -m pytest tests/tracking/test_elastic_sync.py -k "ElasticSyncParams or ExtractBallFeatures or CandidateDetection or Scoring or EventEnrichment or NeedlemanWunsch or AlignEvents" -v --benchmark-skip` → PASS. (`TestAddElasticSync`/`TestElasticSyncXfns` remain RED until Task 8 — do not treat as a regression.)

---

### Task 7: `link_actions_to_frames_elastic` + opt-in hint flag

**Files:**
- Modify: `silly_kicks/tracking/utils.py` (add `link_actions_to_frames_elastic`; add opt-in `suppress_time_base_hint` to `_enforce_link_coverage`)
- Modify: `silly_kicks/tracking/__init__.py` (export the new linker)
- Test: `tests/tracking/test_elastic_linker_contract.py` (new); `tests/tracking/test_time_base_contract.py` (where `_enforce_link_coverage` / the low-coverage warnings are tested — add the byte-identical-default assertion here)

**Interfaces:**
- Produces: `link_actions_to_frames_elastic(actions, frames, *, params=None, min_link_rate=0.5, on_low_coverage="warn") -> tuple[pd.DataFrame, LinkReport]` — pointer schema per spec §6; `n_candidate_frames` is the **per-action** count of episode candidates passing the actor-membership gate; `link_quality_score = elastic_confidence`; `LinkReport.tolerance_seconds = float("nan")`.
- Modifies: `_enforce_link_coverage(..., *, suppress_time_base_hint: bool = False)` — when True, skip the `if p in suspected` hint augmentation (per-period floor + `on_low_coverage` still fire). Default False ⇒ time-linker byte-identical (TF57-SPEC-08).

- [ ] **Step 1: Write failing tests**

```python
def test_elastic_linker_pointer_schema_and_dtypes():
    frames, actions = _make_pass_reception_fixture()
    pointers, report = link_actions_to_frames_elastic(actions, frames)
    assert list(pointers.columns) == [
        "action_id", "frame_id", "time_offset_seconds",
        "n_candidate_frames", "link_quality_score",
    ]
    assert str(pointers["frame_id"].dtype) == "Int64"
    # every linked frame_id is a real frame
    real = set(frames["frame_id"].tolist())
    assert set(pointers["frame_id"].dropna().astype(int)).issubset(real)
    assert np.isnan(report.tolerance_seconds)

def test_n_candidate_frames_is_per_action_not_constant():
    # both-sided: a per-episode constant would make this uniform
    frames, actions = _make_multi_episode_fixture()
    pointers, _ = link_actions_to_frames_elastic(actions, frames)
    assert pointers["n_candidate_frames"].nunique() > 1

def test_default_time_linker_unchanged_byte_identical():
    # _enforce_link_coverage default path must be byte-identical
    frames, actions = _make_low_coverage_fixture()
    with pytest.warns(UserWarning, match="suspected period-relative"):
        link_actions_to_frames(actions, frames, on_low_coverage="warn")

def test_elastic_linker_suppresses_time_base_hint():
    frames, actions = _make_low_coverage_fixture()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        link_actions_to_frames_elastic(actions, frames, on_low_coverage="warn")
    msgs = " ".join(str(x.message) for x in w)
    assert "below min_link_rate" in msgs           # floor still fires
    assert "suspected period-relative" not in msgs  # hint suppressed
```

- [ ] **Step 2: Run to verify failure** → FAIL.
- [ ] **Step 3: Implement** the adapter (call `align_events_to_frames`, map to the pointer schema, build `LinkReport`, call `_enforce_link_coverage(..., suppress_time_base_hint=True)`) and add the opt-in flag to `_enforce_link_coverage`. Export in `__init__.py` (`__all__` + the import block).
- [ ] **Step 4: Run to verify pass** (incl. the byte-identical default) → PASS.

---

### Task 8: Public surface — `add_elastic_sync` + `elastic_sync_xfns` + atomic mirror; purity + id-dtype invariance

**Files:**
- Modify: `silly_kicks/tracking/features.py:6798-6910` (`add_elastic_sync`, `elastic_sync_xfns`)
- Modify: `silly_kicks/atomic/tracking/features.py` (re-export unchanged — verify)
- Test: `tests/tracking/test_elastic_sync.py` (`TestAddElasticSync`, `TestElasticSyncXfns`); rewrite `tests/tracking/test_elastic_sync_id_dtype.py`; verify `tests/test_add_star_purity.py:519`
- Delete/rewrite: `tests/tracking/test_elastic_sync_lookup_golden.py` (greedy lookup golden retired → NW golden on a tiny synthetic frame)

**Interfaces:**
- Produces: `add_elastic_sync(actions, frames, *, params: ElasticSyncParams | None = None) -> pd.DataFrame` — adds the 7-column elastic set minus `action_id` (6 new columns) onto a **copy**; drops the greedy kwargs (breaking). `elastic_sync_xfns(*, params=None) -> list` — still lifts exactly `elastic_confidence`, `elastic_error_seconds` (2 xfns; reception columns are pointers, not xfns).

- [ ] **Step 1: Write failing tests** — update `TestAddElasticSync` to expect the 6 added columns (incl. the 3 reception); keep `test_row_count_preserved`, `test_confidence_bounded`; keep `TestElasticSyncXfns` (still 2 xfns, still NaN in introspection). Rewrite `test_elastic_sync_id_dtype.py`: assert NW confidence/frames are byte-identical across numeric-actions × string-frames and reverse (NA ball row present), and never collapse to a constant.

```python
def test_add_elastic_sync_adds_seven_minus_action_id():
    out = add_elastic_sync(_make_spadl_actions(), _make_tracking_frames())
    added = set(out.columns) - set(_make_spadl_actions().columns)
    assert {"elastic_frame_id", "elastic_confidence", "elastic_error_seconds",
            "elastic_receive_frame_id", "elastic_receive_confidence",
            "elastic_receive_error_seconds"}.issubset(added)

def test_id_dtype_invariance_no_constant_collapse():
    frames, actions = _make_pass_reception_fixture()
    a_int, f_str = _cast_ids(actions, "int"), _cast_frame_ids(frames, "str")
    a_str, f_int = _cast_ids(actions, "str"), _cast_frame_ids(frames, "int")
    r1 = align_events_to_frames(a_int, f_str)
    r2 = align_events_to_frames(a_str, f_int)
    pd.testing.assert_frame_equal(r1, r2)
    assert r1["elastic_confidence"].dropna().nunique() > 1  # not the 0.6 collapse
```

- [ ] **Step 2: Run to verify failure** → FAIL.
- [ ] **Step 3: Implement** the signature change (accept `params`; drop greedy kwargs — this is what clears the Task-1→Task-8 intermediate red), add the reception columns to the merge (`_elastic_cols` list grows to 6). **Also update `add_elastic_sync`'s docstring Examples block** (`features.py:6820-6825`) for the new columns (literal block). Confirm atomic mirror import still valid. Verify the purity entry `tests/test_add_star_purity.py:519` passes (`add_elastic_sync` returns a new object, no input mutation).
- [ ] **Step 4: Run the WHOLE module file green** (this is the task where `test_elastic_sync.py` fully passes — the Global-Constraint milestone) → `python -m pytest tests/tracking/test_elastic_sync.py tests/tracking/test_elastic_sync_id_dtype.py tests/test_add_star_purity.py --benchmark-skip -v` → PASS.

---

### Task 9: Liveness coverage for the reception columns + registry verification

**Files:**
- Modify: `tests/tracking/test_aggregator_column_liveness.py` (the shared fixture at ~line 38-66 + the `add_elastic_sync` entry at `:550`)
- Verify: `tests/tracking/_mirror_entries/misc.py` (elastic mirror entry — no oriented geometry; unchanged), `tests/tracking/conftest_id_dtype.py`

**Interfaces:** none new — this task makes existing gates pass under NW.

**Key risk (spec §14.5 / this plan):** NW's `min_confidence=0.5` is stricter than the greedy `0.1`. On the shared synthetic liveness fixture, elastic columns must not go 100 %-null (liveness asserts `out[c].notna().any()`), and at least one **reception** must be produced (else the 3 reception columns are 100 %-null → gate fails).

- [ ] **Step 1: Run the liveness gate as-is** to observe behaviour under NW.

Run: `python -m pytest tests/tracking/test_aggregator_column_liveness.py -k elastic -v --benchmark-skip`
Expected: likely FAIL (elastic columns null / reception null) — diagnose which columns.

- [ ] **Step 2: Enrich the shared fixture's ball trajectory** so each action window has a clear ball-touch (accel spike + actor within 3 m) and includes one **pass→different-teammate** sequence (produces a reception). Change ONLY the ball/actor kinematics needed; keep positions otherwise stable.
- [ ] **Step 3: Run the FULL liveness gate** to confirm no other aggregator regressed.

Run: `python -m pytest tests/tracking/test_aggregator_column_liveness.py -v --benchmark-skip`
Expected: PASS (all aggregators, incl. the 3 new reception columns live).

- [ ] **Step 4: If a reception cannot be produced on the shared fixture without disrupting other aggregators**, STOP and surface to Karsten (a `STRUCTURAL_CONSTANTS`-style justified exception for the reception columns is a scope decision, not a silent choice). Do not silently exclude.

---

### Task 10: CC-BY oracle fixture + from-scratch accuracy metric + CI accuracy gate

**Files:**
- Create (owner-run reduction on the DGX; committed): `tests/datasets/elastic_sync/j03wmx_slice/{frames,actions,gt}.parquet` + `README.md` (provenance + CC BY 4.0 grant + reduction recipe)
- Create: `tests/tracking/test_elastic_sync_oracle.py` (the accuracy metric + the gate)

**Interfaces:**
- Produces: `sync_accuracy(pred: pd.DataFrame, gt: pd.DataFrame, *, tolerances=(0,2,5,25)) -> dict` — reimplemented `|pred_frame − gt_frame| ≤ N` exact/within-N %, per category + pooled. **Never** ELASTIC's metric.

- [ ] **Step 1 (Claude-run on the DGX → committed):** SSH to the box (`karsten@192.168.68.73`), reduce J03WMX to a ≤~3 MB contiguous slice containing all four categories + ≥1 out/goal/reception; write the three parquets + README; scp into the local working tree at `tests/datasets/elastic_sync/j03wmx_slice/`. The CC BY 4.0 grant is **already evidenced** (ELASTIC repo README, 2026-09-11) — carry the attribution; no separate gate. **May be produced ahead of the code** (owner decoupled it: Tasks 1–9,11 don't depend on it).
- [ ] **Step 2: Write the metric + a failing gate**

```python
@pytest.mark.parametrize("tol,floor", [(0, _EXACT_FLOOR), (2, _W2_FLOOR)])
def test_nw_clears_regression_floor(tol, floor):
    frames, actions, gt = _load_oracle_slice()
    pred = align_events_to_frames(actions, frames)
    acc = sync_accuracy(pred, gt, tolerances=(tol,))
    assert acc["pooled"][tol] >= floor

def test_nw_beats_retired_greedy_baseline():
    # a tiny in-test greedy re-impl (argmax over accel in a ±1s window) as the
    # low-water mark; NW must beat it by a large margin on the oracle slice.
    ...

def test_a_mutation_that_should_drop_accuracy_fails_the_floor():
    # both-sided: shuffle candidate assignment -> exact-frame collapses below floor
    ...
```

`_EXACT_FLOOR`/`_W2_FLOOR` are **placeholders in the plan only** — set from the DGX-measured NW accuracy on this exact slice minus a safety margin, during implementation (Global Constraint: regression floor, not the paper's number). Record the measured value + margin in the test module docstring.

- [ ] **Step 3: Run** to verify the gate FAILS before the floor is calibrated / with a broken alignment, PASSES with the real NW. `python -m pytest tests/tracking/test_elastic_sync_oracle.py -v --benchmark-skip`.
- [ ] **Step 4:** Confirm the gate runs in the **regular** suite (fixture committed → NOT `@e2e`) and is fast.

---

### Task 11: ADR-073 sub-quadratic growth guard + SB360 registry re-adjudication

**Files:**
- Create: `tests/tracking/test_elastic_sync_perf_budget.py` (structural growth guard)
- Modify: `tests/sb360/_entries/_context.py` (+ regenerate via `tests/sb360/_regenerate.py` — ⚠ back up `_entries/` first; NOT idempotent)

- [ ] **Step 1: Write the growth guard** — `assert_subquadratic_growth` over the candidate-detection + episode loop, scaling the **episode/group** dimension (per ADR-073: scale the loop-iteration dimension, not within-group). Register a scoped `rows_scanned_counter` proving no rescan-in-loop.
- [ ] **Step 2: Run** → PASS (growth exponent ≤ 1.5).
- [ ] **Step 3: Re-adjudicate the SB360 verdict for `add_elastic_sync`** — elastic is continuous-only; on velocity-less/anonymous SB360 freeze-frames it produces no alignment. Confirm the machine observation + human verdict are unchanged from `main` (still `honest_nan`/`differs_by_design` as recorded), and that the 3 new reception columns are captured by the registry. Back up `_entries/`, regenerate, diff, verify round-trip byte-identical for the unchanged parts.
- [ ] **Step 4: Run** `python -m pytest tests/sb360/ -v --benchmark-skip` → PASS.

---

### Task 12: Bookkeeping + commit-prep (NO commit — await approval)

**Files:**
- Modify: `silly_kicks/feature_glossary.py:1271-1293` (update the 3 elastic definitions to NW; add 3 reception `FeatureColumn` entries)
- Modify: `NOTICE:492-499` (CIKM 2026 / arXiv:2608.30227 + NW description; add the CC BY 4.0 benchmark attribution — Bassek et al. 2025 doi:10.1038/s41597-025-04505-y + ELASTIC paper)
- Create: `docs/superpowers/adrs/ADR-0XX-elastic-nw-sync.md` (per ADR-TEMPLATE)
- Modify: `CHANGELOG.md` (new `PR-S###` entry; Hyrum/re-materialize trigger called out), `TODO.md` (TF-57 On-Deck → shipped + mart re-materialize follow-up), `silly_kicks/_version.py` (bump ≈4.113.0)
- Verify: C4 completeness gate green with no DSL change (no new `add_*` aggregator; count stays 33)

- [ ] **Step 1: Glossary** — update the 3 elastic entries' `definition` to NW; add `elastic_receive_frame_id` / `elastic_receive_confidence` / `elastic_receive_error_seconds` `FeatureColumn`s (`emitting_module=_M_ELASTIC`, `attribution=_A_ELASTIC`, `higher_is_better` per direction). Run the coverage gate.

Run: `python -m pytest -k "feature_glossary or glossary" --benchmark-skip`
Expected: PASS.

- [ ] **Step 2: NOTICE** — rewrite the ELASTIC block; add benchmark attribution. Run the attribution/notice test if present.
- [ ] **Step 3: ADR** — write ADR-0XX (greedy removal; breaking `ElasticSyncParams`; reception columns; two-surface design; clean-room/MPL boundary; CC-BY oracle; mart re-materialize/no-default-retrain trigger).
- [ ] **Step 4: CHANGELOG + TODO + version bump.**
- [ ] **Step 5: C4** — run the C4 completeness gate; confirm no change needed.

Run: `python -m pytest -k "c4 or architecture" --benchmark-skip`
Expected: PASS.

- [ ] **Step 6: FULL suite + lint + types (green before proposing a commit).**

```bash
python -m ruff check silly_kicks/ tests/ scripts/
python -m ruff format --check silly_kicks/ tests/ scripts/
python -m pyright
python -m pytest tests/ -m "not e2e" --benchmark-skip -q
```

Expected: all green.

- [ ] **Step 7: Commit-prep — DO NOT COMMIT.** `git fetch && git merge origin/main` (resolve any drift), assign the real ADR / `PR-S###` / version numbers, re-run the full suite, then **present the exact diff + file list to Karsten and STOP.** The single commit lands only on explicit per-commit approval (CLAUDE.md). The oracle fixture is committed only after the owner confirms the CC BY 4.0 grant (Task 10 Step 1).

---

## Tasks 13–19 (rev 3) — Fix cycle: close the exact-frame gap + fold in the impl-review findings

> **Context (spec §16).** The as-built NW (Tasks 1–12) scores **29.5 % exact** on the committed oracle while the paper reports **88.4 %** on the same data (J03WMX is a paper benchmark match; our candidate density ~16 % matches the paper's ~19 %). Root cause (spec §16.2, code-corroborated by the spec-r3 review): the **scoring does not peak at the true touch** for contested/defensive/shot events — `s(true) < s(chosen)` for 16/16 gross errors; DP / identity vote-map / candidate-detection / density all ruled out. These tasks fix the scoring to the paper, rebuild the oracle, and resolve the impl-review findings. **Nothing new is committed — this still lands in the one uncommitted commit.**
>
> **Clean-room guard (HARD, spec §16.4).** Tasks 14–16/19 debug on the DGX box that hosts the MPL `~/elastic_validation/elastic_repo/` clone. Re-derivation reads **ONLY** the paper (arXiv:2608.30227) + silly-kicks code; the MPL clone stays **black-box run-only**, its scoring source **never opened**; a paper-unrecoverable divergence triggers the §16.6 human-gated **STOP**, never a source peek.

### Task 13: White-box "score peaks at ground-truth" harness (RED first — the fix gate)

**Files:**
- Create: `tests/tracking/test_elastic_score_peaks.py`
- Uses: committed `tests/datasets/elastic_sync/j03wmx_slice/` + `_elastic_sync` internals (`_detect_candidate_frames`, `_build_frame_lookups`, `_enrich_events`, `_score`)

**Interfaces:** Consumes the internal scorer; produces the gate Tasks 14–15 drive to green.

- [ ] **Step 1: Write the harness (lands RED).** For each real event, the candidate **nearest `gt.frame_id`** (the "true" candidate — must contain the actor) must score **≥ the candidate the DP actually chose** (`align_events_to_frames`' `elastic_frame_id`): `s(true) >= s(dp_chosen) − EPS`, `EPS = 1e-9`. **Tolerance (stated, tight — this is THE fix gate):** no slack band; ties (`EPS`) only for genuine score plateaus. *Why "≥ the DP's own pick" and NOT "= the episode/global argmax":* the global best-scoring candidate is often ±thousands of frames away in a **different episode** (spec §16.1 `best@off`), which the order-constrained per-episode DP cannot reach — so a global-argmax assertion would false-fail even a perfect fix. Comparing against the DP's *actual* pick is the robust, can't-false-fail encoding of the defect (`s(true) < s(dp_chosen)` for 16/16 gross errors today) and still forces the true touch to win in-context. This also correctly flags the ±1/±2 near-misses (the DP picked a slightly-higher-scoring neighbour) as scoring mis-peaks. Track, separately and **non-gating**, the pooled exact-frame (current 29.5 %) and the *frame distance* `|nearest_cand − gt|`, so a pure **candidate-granularity** residual (no candidate sits exactly at gt, so `s(true) == s(dp_chosen)` passes but exact-frame is ±1 off) is visible and distinct from a scoring failure.

```python
EPS = 1e-9  # ties only; no slack band

def test_true_touch_scores_at_least_the_dp_pick_per_event():
    # events/lookups/candidates from the committed oracle (mechanism: spec §16.1 diagnostics)
    # for each real event e:
    #   true       = actor-gated candidate nearest gt.frame_id
    #   dp_chosen  = candidate at align_events_to_frames' elastic_frame_id for e
    #   assert _score(e, true) >= _score(e, dp_chosen) - EPS
    # RED now: fails for the 16 gross events (tackle/interception/clearance/shot/some passes)
    ...
```

- [ ] **Step 2: Run — confirm RED.** `python -m pytest tests/tracking/test_elastic_score_peaks.py -v` → FAIL on the gross events (documents the pre-fix state; the defect the fix must clear).
- [ ] **Step 3: Pin fixture validity (ADR-032 idiom).** Assert the harness scores ≥ N in-domain events with non-degenerate actor-gated candidate sets, so a later green is not vacuous.

### Task 14: Per-feature scoring re-derivation against the paper (the core fix)

**Files:**
- Modify: `silly_kicks/tracking/_elastic_sync.py` (`_score` maps `s_BA`/`s_PBD`/`s_KD±`/`s_PBDS±`/`s_OD` + category compositions; possibly add `kick_distance_window_seconds` to `ElasticSyncParams` — within the declared breaking change, spec §5.1/§16.4)
- Test: `tests/tracking/test_elastic_score_peaks.py` (Task 13) + `tests/tracking/test_elastic_sync_oracle.py`

**Interfaces:** Consumes Task 13's gate; produces the fixed `_score` Tasks 16/19 validate on real data.

- [ ] **Step 1: Per-feature audit vs spec §3.1 + the paper.** Confirm each feature's window / sign / normalization bound against arXiv:2608.30227. **Prime suspect (code-verified by spec-r3):** `_score` reuses the 0.2 s slope window (`h_frames=5`) for the `s_KD±` kick-distance term (`_elastic_sync.py:304/309/314`); the paper likely uses a distinct, longer post-touch window, so at the true kick the ball has not yet "departed 3 m" and a later frame wins. Derive the window the paper specifies.
- [ ] **Step 2: Implement only what the paper supports.** If a kick-distance window field is needed, add `kick_distance_window_seconds` to `ElasticSyncParams` at the paper's value (frozen; `for_provider` empty — ADR-009).
- [ ] **Step 3: Run the gate (GREEN).** `python -m pytest tests/tracking/test_elastic_score_peaks.py tests/tracking/test_elastic_sync_oracle.py -v` → the true touch becomes the argmax for the previously-gross events; pooled exact-frame rises materially.
- [ ] **Step 4: Both-sided.** A mutation reverting the window reintroduces the gross errors (harness RED), proving the fix is load-bearing (house "both sides" rule).

### Task 15: Smoothing + candidate-extrema reconciliation (only if Task 14 leaves a residual)

**Files:** Modify `_elastic_sync.py` (extrema params) / document the tc3 smoothing regime.

- [ ] **Step 1:** If exact-frame after Task 14 is still short of the paper, test whether the tc3-cache Savitzky–Golay smoothing shifts `s_BA` accel peaks vs the paper's tracking (align the smoothing or justify the difference), and re-check `argrelextrema`/`find_peaks` params (prominence / `order`) against the §3.1 candidate definition.
- [ ] **Step 2:** Re-run the harness + oracle; record the exact-frame delta each change buys. **If a divergence is unrecoverable from the paper alone → STOP and surface for the owner (spec §16.6 fallback); do NOT read the MPL source.**

### Task 16: Oracle rebuild on the DGX — real roster + goal span (closes IMPL-02)

**Files:** Modify committed `tests/datasets/elastic_sync/j03wmx_slice/` (frames/actions/gt + `README.md`); DGX `~/elastic_validation/make_oracle_slice.py` (off-repo).

- [ ] **Step 1: Licence gate (spec §16.5).** On the DGX, verify the DFL `MatchInformation` roster's CC-BY-4.0 status; record the evidence in the fixture README. If unevidenced → keep the vote-map (conservative floor) + note the identity limit.
- [ ] **Step 2: Rebuild.** Re-extract with a deterministic real-roster identity map (replacing the match-wide vote-map) + a **second short span containing a goal** (spec §14.5 two-span remedy). Re-anchor times per the existing recipe.
- [ ] **Step 3: README — fix the misquote (IMPL-02).** Quote the §8.1 requirement correctly ("all four categories + ≥1 out / **goal** / reception"); document both spans + the roster source + the licence evidence. No claim of compliance the slice does not meet.
- [ ] **Step 4:** Re-run the oracle gate + the Task-13 harness on the rebuilt fixture → green.

### Task 17: Folded test findings — IMPL-01 (NW golden), IMPL-03 (greedy beat), IMPL-05 (dedup)

**Files:** `tests/tracking/test_elastic_sync.py` (align-level golden); `tests/tracking/test_elastic_sync_oracle.py` (greedy beat); `silly_kicks/tracking/utils.py` + `_elastic_sync.py` (dedup).

- [ ] **Step 1 (IMPL-01):** Add `TestAlignEventsToFrames::test_nw_golden_on_synthetic_frame` — a tiny hand-built frames+actions scene whose correct alignment is obvious by construction; assert the exact `elastic_frame_id` (+ reception) output. Distinct from the DP-kernel golden (`TestNeedlemanWunsch`), per spec §9 / plan Task 8's retired→replaced requirement.
- [ ] **Step 2 (IMPL-03):** Replace `test_nw_massively_beats_do_nothing_baseline` with a **tiny in-test greedy re-impl** (the retired `0.6·accel + 0.4·proximity` argmax over a ±1 s window) scored on the oracle; assert NW beats it by a large margin (spec §8.1 — the constant-0.6-catching guard). Both-sided.
- [ ] **Step 3 (IMPL-05):** In `link_actions_to_frames_elastic`, stop calling `_detect_candidate_frames` twice — thread the candidate dict out of `align_events_to_frames` (new optional return/param) or memoize; assert byte-identical pointer output (parity) + a one-detection-call count spy.
- [ ] **Step 4:** Run the elastic module + oracle + linker-contract tests → green.

### Task 18: Corrected attribution + recalibrated floor + bookkeeping (spec §16.3 / §16.8)

**Files:** `CHANGELOG.md`, `TODO.md`, `docs/superpowers/adrs/ADR-093-elastic-nw-sync.md`, fixture `README.md`, `tests/tracking/test_elastic_sync_oracle.py` (floors), `silly_kicks/feature_glossary.py` (if defs change).

- [ ] **Step 1 (attribution — spec §16.3):** Remove the wrong "residual gap = identity vote-map + candidate detection, NOT the algorithm" claim from CHANGELOG / TODO / fixture README / ADR-093; state the corrected root cause (a scoring transcription divergence, now fixed).
- [ ] **Step 2 (floors):** Recalibrate `_START_EXACT_FLOOR` / `_START_W2_FLOOR` / `_RECEIVE_W5_FLOOR` from the **fixed-scoring** achieved-on-box number minus a safety margin (still a regression guard, not the aspirational number); the `test_a_shuffled_alignment_fails_the_floor` mutation still proves teeth.
- [ ] **Step 3:** Update CHANGELOG / ADR-093 / TODO for the scoring fix + oracle rebuild + any new `ElasticSyncParams` field (noted within the declared breaking change). **If a field was added, update the spec §5.1 dataclass listing too** (it ships in the same commit — CONSIDER-5; the field must appear in the spec's `ElasticSyncParams` table, not only the code). Glossary defs unchanged unless a column's meaning moved.

### Task 19: DGX 3-match acceptance + full suite + commit-prep (NO commit)

**Files:** DGX `~/elastic_validation/` (off-repo run); recorded numbers land in CHANGELOG / ADR-093.

- [ ] **Step 1: DGX 3-match run (spec §16.6).** Run the fixed NW on J03WMX / J03WN1 / J03WPY with the real roster; report exact + within-N, per-category, pooled. Compare to the paper (88.4 % / 96.5 %).
- [ ] **Step 2: Acceptance gate (spec §16.6).** Bar met (≈ paper, documented margin) → proceed. A genuine clean-room-limit divergence → **STOP and surface for the owner (SO-2); no silent lower bar, no MPL source peek.**
- [ ] **Step 3: FULL suite + lint + types green** (as Task 12 Step 6): `ruff check` + `ruff format --check` (CI scope) + `pyright` + `pytest -m "not e2e" --benchmark-skip`.
- [ ] **Step 4: Commit-prep — DO NOT COMMIT.** `git fetch && git merge origin/main`; finalize ADR / PR-S / version; re-run the full suite; present the exact diff + the DGX numbers for the §16.9 sign-offs (SO-1 recorded; SO-2 Goal-5 numbers; SO-3 the diff) and **STOP.** Independent impl **re-review** by a fresh owner-started session precedes any commit.

---

## Self-Review

**1. Spec coverage.**
- §2/§3 algorithm → Tasks 2–6. §5 elastic surface → Tasks 1, 6, 8. §6 linkage surface + hint flag → Task 7. §7 mapping → Task 4. §8.1 oracle + gate → Task 10. §8.2 DGX / §8.3 ablation → owner-run (off-repo; noted, not a code task — reported-not-gated per ADR-009). §9 testing → Tasks 1–11. §10 bookkeeping → Task 12. §11 Hyrum → Task 12 ADR/CHANGELOG. §14.5 liveness risk → Task 9. TF57-SPEC-08 → Task 7. **No uncovered requirement.** (§8.2/§8.3 are deliberately owner-run and off-repo — they are validation reports, not commit content.)
- **rev 3 (spec §16) coverage:** §16.1–16.2 root cause → Task 13 (the white-box harness that reproduces it as a RED gate). §16.4 scoring re-derivation → Tasks 14–15. §16.5 oracle rebuild (real roster + goal span, licence gate) → Task 16. §16.3 corrected attribution + §16.8 bookkeeping + recalibrated floor → Task 18. §16.6 acceptance bar + DGX 3-match → Task 19. §16.7 folded findings: IMPL-01/03/05 → Task 17, IMPL-02 → Task 16 (goal span + README misquote), IMPL-04 → spec §16.9 SO-1 (signed off; no code), IMPL-06 → verified (no change). §16.9 sign-offs → Task 19 Step 4 commit-prep. Clean-room guard (§16.4) → block preamble + Tasks 14–16/19. **No uncovered rev-3 requirement.**

**2. Placeholder scan.** The only intentional "to-be-filled" values are `_EXACT_FLOOR`/`_W2_FLOOR` in Task 10 — explicitly specified as DGX-calibrated-during-implementation (a regression floor, Global Constraint), not a plan omission. Test bodies with `...` (`test_nw_beats_retired_greedy_baseline`, mutation test, `_cast_*` helpers) are marked as shapes the executor completes against the concrete fixture; every one has its assertion intent stated. No "TBD/handle edge cases/add validation".

**3. Type consistency.** `ElasticSyncParams` fields (Task 1) are used verbatim in Tasks 2–7. `Candidate` (Task 2) / `Event` (Task 4) namedtuples are consumed by `_score` (Task 3) and `_needleman_wunsch` (Task 5) with consistent fields. `align_events_to_frames` 7-column output (Task 6) matches `add_elastic_sync` (Task 8), the glossary entries (Task 12), and the linker mapping (Task 7). `link_actions_to_frames_elastic` return type matches `link_actions_to_frames` (`tuple[pd.DataFrame, LinkReport]`). No signature drift.

---

## Execution Handoff

Execution is **owner-gated** and comes AFTER (a) this plan is independently reviewed + Karsten approves, and (b) the implementation itself will get its own independent review (I do not run my own — `[[feedback_author_does_not_run_own_review]]`). **Do not begin implementing until Karsten says go.**

When authorized, the recommended approach is **subagent-driven development** (`superpowers:subagent-driven-development`) — a fresh subagent per task with review between tasks — subject to the ONE-commit constraint (subagents do NOT commit; the single commit is the final task after full-suite green + owner approval). Inline execution (`superpowers:executing-plans`) is the alternative. The choice is Karsten's.
