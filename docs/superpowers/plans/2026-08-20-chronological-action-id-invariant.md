# Chronological `action_id` invariant — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** make every SPADL converter **order-insensitive** — a pure function of chronological event content — so `action_id` is a chronological `(game_id, period_id, time_seconds)` index and every `.shift()`-derived field (end coords, dribbles, pass results, set-piece classes, event types) resolves the true time-neighbour; enforce it with an input-permutation CI gate + a runtime raise-guard; harden the `action_id`-alone consumers.

**Architecture:** One shared stable-sort seam every converter uses at the top of its frame (before any positional op). Correctness is **gate-driven, not audit-driven**: a permutation-invariance gate discovers which converters are order-dependent and proves each fixed one is order-insensitive (content-equality, not index-order). A raise-by-default guard at the `_finalize_output` choke point is the runtime net; the `action_id`-alone consumers adopt the robust `(time_seconds, action_id)` key for the mart-reading path.

**Tech Stack:** pandas/numpy; pytest; `silly_kicks/spadl/*` converters; the `[parse-dfl]` extra for the sportec/IDSSE path.

**Design doc:** `docs/superpowers/specs/2026-08-20-chronological-action-id-invariant-design.md` (rev 3, 2 review rounds, converged).

## Global Constraints

- **The invariant is order-insensitivity**, not merely "sort before `action_id`." The sort goes at the **top** of each converter's frame, before ANY `.shift()`-based derivation (`_derive_end_coordinates`/`_add_dribbles` in `base.py`, skillcorner `:331` results/`is_short`, wyscout `_fix_wyscout_events`).
- **Gate-driven, not audit-driven.** Which converters need a sort, where, and with which key columns is decided by making the permutation gate green — never by a hand-list (hand-enumeration missed skillcorner/wyscout in review 1 and mis-prescribed wyscout in review 2). Task 2 builds the gate and RECORDS the observed red set; Task 3 fixes exactly that set.
- **No version number is claimed until the work is complete and tested** (owner rule). CHANGELOG/version is the final task; the number is set then.
- **The runtime guard RAISES by default** — a deliberate deviation from the warn-default `SILLY_KICKS_ASSERT_INVARIANTS` convention, justified because a violation crashes downstream or silently corrupts (unlike the bounded orientation case). NaN-`time_seconds` rows are excluded (not a violation).
- **Value-shift → retrain trigger; ATOMIC migration.** `action_id` is a cross-table join key: renumbering invalidates persisted id-keyed data, and a partial re-conversion misjoins silently. The CHANGELOG must say: re-convert ALL affected-provider data (bronze, goldens, every id-keyed derived table) together, never incrementally.
- **Behavior-preserving where already correct.** kloppy/opta/statsbomb are byte-identical after routing through the shared seam; skillcorner differs only on co-timestamped rows (the stability repair) — asserted, not assumed.
- TDD throughout; **no commit steps** (the user commits once, at the end, on explicit approval). Match existing file style; ASCII-clean in any `scripts/` argparse.
- **Verify per task:** `python -m pytest <targets> -m "not e2e" --benchmark-skip -q`. Final: full `tests/` + `python -m ruff check silly_kicks/ tests/ scripts/` + `ruff format --check` + `python -m pyright` (all CI scope). Long runs (`>30s`) go background-and-poll.

---

### Task 1: The shared chronological-sort seam

**Files:** Create in `silly_kicks/spadl/base.py`; Test `tests/spadl/test_sort_actions_chronologically.py`.

**Interfaces:**
- Produces: `sort_actions_chronologically(frame, *, by=("game_id","period_id","time_seconds"), tiebreak=()) -> pd.DataFrame` — stable (`kind="mergesort"`) sort by `(*by, *tiebreak)`; empty frame returns unchanged; does not mutate input.

- [ ] **Step 1: failing tests**

```python
def test_stable_sort_orders_by_key_and_preserves_ties():
    df = pd.DataFrame({"game_id":[1,1,1], "period_id":[1,1,1], "time_seconds":[2.0,1.0,1.0],
                       "tag":["late","tieA","tieB"]})
    out = sort_actions_chronologically(df)
    assert list(out["tag"]) == ["tieA","tieB","late"]          # time order; ties keep input order (stable)

def test_custom_by_columns_for_raw_event_frames():
    df = pd.DataFrame({"period_id":[1,1], "milliseconds":[500,100], "tag":["b","a"]})
    out = sort_actions_chronologically(df, by=("period_id","milliseconds"))
    assert list(out["tag"]) == ["a","b"]

def test_tiebreak_breaks_equal_time_deterministically():
    df = pd.DataFrame({"game_id":[1,1], "period_id":[1,1], "time_seconds":[1.0,1.0],
                       "__order__":[1.5,1.0], "tag":["synth","parent"]})
    out = sort_actions_chronologically(df, tiebreak=("__order__",))
    assert list(out["tag"]) == ["parent","synth"]

def test_empty_frame_passes_through():
    df = pd.DataFrame({"game_id":[], "period_id":[], "time_seconds":[]})
    assert len(sort_actions_chronologically(df)) == 0

def test_does_not_mutate_input():
    df = pd.DataFrame({"game_id":[1,1], "period_id":[1,1], "time_seconds":[2.0,1.0]})
    before = df.copy()
    sort_actions_chronologically(df)
    pd.testing.assert_frame_equal(df, before)

def test_missing_ordering_key_raises_not_silently_partial_sorts():   # M-D
    df = pd.DataFrame({"period_id":[1,1], "tag":["b","a"]})          # no time_seconds
    with pytest.raises(KeyError, match="absent|ordering key"):
        sort_actions_chronologically(df)                            # default by includes time_seconds
    with pytest.raises(KeyError):
        sort_actions_chronologically(df, by=("period_id","milliseconds"))  # mistyped/absent time col
```

- [ ] **Step 2:** run, verify all fail (`ImportError` / not defined).
- [ ] **Step 3:** implement the helper:

```python
def sort_actions_chronologically(frame, *, by=("game_id", "period_id", "time_seconds"), tiebreak=()):
    if len(frame) == 0:
        return frame
    keys = [*by, *tiebreak]
    missing = [c for c in keys if c not in frame.columns]
    if missing:  # M-D: NEVER silently partial-sort -- the exact bug this seam exists to prevent
        raise KeyError(
            f"sort_actions_chronologically: ordering key(s) {missing} absent. Pass a `by=` matching "
            f"this frame (raw-event frames pass e.g. by=('period_id','milliseconds')); a missing key "
            f"must fail loud, not degrade to a partial sort."
        )
    return frame.sort_values(keys, kind="mergesort").reset_index(drop=True)
```

**M-D:** the seam does **not** filter absent keys — it raises. Callers pass the `by`/`tiebreak` that exist on their frame (SPADL-actions callers use the default incl. `game_id`; raw-event callers pass their own columns without `game_id`). A missing/mistyped ordering key on the one seam whose whole job is to never let ordering rot silently must fail loud.

- [ ] **Step 4:** run, verify green.

---

### Task 2: The permutation-invariance gate + index-chronology assertion (RED-FIRST; discovers the broken set)

**Files:** Create `tests/spadl/test_converter_order_insensitivity.py`; a small `tests/spadl/_converter_cases.py` registry mapping each `convert_to_actions` to a runnable committed fixture + its `convert_to_actions(**kwargs)` call.

**Interfaces:**
- Consumes: each converter's `convert_to_actions` + a committed fixture. Produces: the observed set of order-dependent converters (recorded in the ledger for Task 3).

- [ ] **Step 1: the gate** — for each registered converter:
  1. `out_A = convert(fixture_input)`.
  2. `permuted = reverse_timestamp_blocks(fixture_input)` — **reverse** the order of the timestamp-blocks, preserving within-timestamp order (R3: never disturb genuinely-ambiguous ties). Reverse, not a seeded shuffle (L1: a shuffle can land on the identity permutation and pass vacuously) — deterministic and guaranteed non-identity for >1 block. **Assert `permuted` is not row-equal to `fixture_input`** before converting (non-vacuity backstop; a single-timestamp fixture trips this and must supply the synthetic multi-timestamp input from Step 4). (Deliberate single permutation: reversal maximally disturbs order, so it almost certainly exposes any order-dependence; a second distinct permutation per case is optional belt-and-suspenders against the implausible "invariant to reversal but not to a partial reorder" gap — not needed to ship.)
  3. `out_B = convert(permuted)`.
  4. **Content-equality (R1), NOT event-id join:** drop `action_id`; align by sorting both on the **discrete columns only** (ids/types/int/string fields — L2: never sort the alignment on float columns, whose last-bit differences reorder near-equal rows differently across the two runs and misalign the row-wise compare), then `assert_frame_equal(a, b, atol=1e-9, check_dtype=False)` so the float tolerance operates on already-aligned rows. Content alignment verifies the synthesized rows (dribbles, GS cross-goal/foul) that carry no event id.

```python
@pytest.mark.parametrize("name", sorted(CONVERTER_CASES))
def test_converter_is_order_insensitive(name):
    case = CONVERTER_CASES[name]
    permuted = case.permute(case.input)                 # L1: reverse timestamp-blocks (deterministic)
    assert not _row_equal(permuted, case.input), f"{name}: permutation is a no-op -- gate would be vacuous"
    a = _canonical(case.run(case.input).drop(columns=["action_id"]))   # L2: _canonical sorts DISCRETE cols only
    b = _canonical(case.run(permuted).drop(columns=["action_id"]))
    pd.testing.assert_frame_equal(a, b, atol=1e-9, check_dtype=False)
```

- [ ] **Step 2: index-chronology assertion (§3b, complementary)** — for each converter, `out = convert(fixture)`; assert within every `(game_id, period_id)`, `action_id`-sorted `time_seconds` is non-decreasing (finite rows).
- [ ] **Step 3: meta-assertion** — enumerate every `convert_to_actions` in `silly_kicks/spadl/` (AST or import of `spadl.__all__`/module functions) and assert each has a `CONVERTER_CASES` entry, so a new converter cannot skip the gate. (Same anti-rot shape as the SB360 registry gates.)
- [ ] **Step 4: RUN and RECORD.** Run the gate. It lands **red** for the order-dependent converters. **Observe and record the exact red set** (which converters fail permutation and/or index-chronology) in the ledger — the authoritative input to Task 3 (expected: sportec, gradientsports; TBD: skillcorner, wyscout, metrica). Do NOT fix converters here; the gate must be observed failing first (ADR-051 red-first). If a converter's committed fixture is trivially single-timestamp or too small to permute, add a synthetic multi-timestamp non-chronological input for it (non-vacuity — a gate that only sees pre-sorted input proves nothing).

- [ ] **Step 5: keep CI green between tasks via strict xfail.** After observing the red set, mark those converters' parametrizations `@pytest.mark.xfail(strict=True, reason="order-insensitivity fix pending -- Task 3")`. **L3: a broken converter fails BOTH the permutation gate (Step 1) and the index-chronology assertion (Step 2), so BOTH parametrizations for each broken converter get the marker** (via a shared `_KNOWN_BROKEN` set both tests consult), else Task 2 leaves CI red on the index check. CI is then green (each known-broken case is an expected-fail; correct converters pass normally). Task 3 removes each converter from `_KNOWN_BROKEN` as it fixes it — a strict xfail that starts passing FAILS, so the marker cannot outlive the fix (the SB360 boundary-gate pattern). Record the set in the ledger.

---

### Task 3: Sort-at-top each gate-red converter until green (the value-shifting fix)

**Files:** the converters in the recorded red set (`sportec.py`, `gradientsports.py`, and any of `skillcorner.py`/`wyscout.py`/`metrica.py` the gate flagged); `base.py` positional helpers only if the gate shows they need a pre-sorted contract (they already document it); committed goldens for flagged providers.

**Interfaces:** Consumes `sort_actions_chronologically` (Task 1). Each converter calls it at the TOP of the relevant frame, before any `.shift()`.

- [ ] **Step 1 (per flagged converter, iterate):** insert `frame = sort_actions_chronologically(frame, by=<frame's period/time cols>, tiebreak=<provider tiebreak>)` at the earliest point before any positional derivation:
  - sportec/metrica: the raw actions frame after `_build_raw_actions`, before `_derive_end_coordinates`.
  - gradientsports: `sort_actions_chronologically(events)` at the top (before dispatch/synthesis), default keys. **PLUS the §2b Option-D null-clock-FOUL fix (MEASURED, owner-chosen):** (i) add `start_time` to `EXPECTED_INPUT_COLUMNS` (float; absolute clock); (ii) replace the native-order `groupby("period_id")["time_seconds"].ffill().bfill()` imputation with a **`start_time`-ordered** imputation — within each period, order rows by `start_time`, ffill/bfill `time_seconds` in that order (foul gets its `start_time`-predecessor's game clock); fall back to `event_time` if `start_time` is absent, then to native-order ffill with a `warnings.warn(..., stacklevel=2)` if neither exists; (iii) update `scripts/_loader_pining.py::_gs_flatten_events` to emit `start_time`=`ev["startTime"]` (and `event_time`=`ev["eventTime"]`); (iv) the synthetic generator already emits `startTime`/`eventTime`, but `tests/spadl/test_gradientsports.py::_load_synthetic_events` must read `start_time`; (v) the GS gate case in `_converter_cases.py` now carries a **NaN-time FOUL** (the path is order-insensitive under D — no carve-out). M-C proof: `new(raw)` == `old(startTime-sorted-then-native-ffill)` on the NaN-foul fixture is byte-identical on native-ordered data. Lakehouse GS shaper must supply `start_time` (DOWNSTREAM HANDOFF; CHANGELOG + CLAUDE.md GS contract).
  - skillcorner: before the `:331` results/`is_short` shifts.
  - wyscout: only if the gate is still red despite `_wyscout_events.py:277` — then sort the raw events (`by=("period_id","milliseconds")`) before the pre-`:277` shifts.
  - Choose the tiebreak per §1/R3 (a logical intra-timestamp sequence key; verify the provider's native intra-timestamp order is not a scramble; measure `time_seconds` granularity).
- [ ] **Step 2 (per converter):** re-run Task 2's gate for that converter; iterate the insertion point/key until BOTH the permutation gate AND the index-chronology assertion are green, then **remove that converter's `xfail(strict=True)` marker** (Task 2 Step 5) — the now-passing strict xfail forces its removal, so a fixed converter can't keep a stale marker.
- [ ] **Step 3:** for skillcorner (if flagged), assert it differs from its pre-change output ONLY on co-timestamped rows (the stability repair) — nowhere else.
- [ ] **Step 3b (OPTIONAL cleanup, deferrable — L4):** route the already-correct converters' inline sorts (kloppy/opta/statsbomb) through the shared seam — pure DRY unification, NOT part of the fix (they are already order-insensitive). Guard with a byte-identical assertion vs their pre-refactor output; **if byte-identical FAILS, do NOT force it** — that means their inline sort differs subtly from the helper's default `by`/`tiebreak` (a finding to investigate — e.g. a missing `game_id` or a different tiebreak), not something to paper over. Defer this step entirely rather than risk the byte-identical guarantee.
- [ ] **Step 4: PROVE the value-shift is exactly a sort (M-C), THEN regenerate goldens.** For each flagged converter, establish mechanically — not by eyeballing a large diff — that the ONLY behavioral change is ordering: assert **`new_code(raw_fixture)` == `old_code(chronologically-pre-sorted fixture)`** (drop `action_id`; content-compare per Step 1.4/L2). The old (pre-Task-3) converter does not sort, so feeding it a manually time-sorted input makes it process events chronologically; the new converter sorts then processes — equality proves `new == old ∘ sort`, i.e. **zero unrelated logic drift** smuggled under the diff. Obtain `old_code` from the pre-fix base commit (`git worktree`/`git show` of the converter, or capture its output before editing). This is the crux acceptance check — a regenerated golden merely pins whatever the new code emits (correct or not) and cannot make this argument.

  **The pre-sort MUST mirror THIS converter's exact sort key AND frame (review-2 caveat) — not a uniform input-level pre-sort, or a correct fix false-fails:** (a) **key** — pre-sort with the same `by`+`tiebreak` the new code uses here (sportec/metrica: SPADL `(game_id, period_id, time_seconds)`; wyscout: raw `(period_id, milliseconds)`; GS: + the `__order__` tiebreak); (b) **frame** — apply the pre-sort at the SAME frame the new code sorts. Top-of-pipeline sorters (sportec/metrica/GS) pre-sort the input fixture directly; a MID-pipeline sorter (wyscout sorts inside the event→action path, *after* `_fix_wyscout_events` drops unmatched duels) must either pre-sort at the sort-point frame, or first confirm the input→sort-point path is order-preserving (removal-only ok; any reorder is not). A key/frame mismatch is self-diagnosing (ask: real bug, key mismatch, or frame mismatch?) but avoid it by mirroring exactly. **Only after M-C passes**, regenerate the committed goldens (end coords, dribbles, action_id, skillcorner/wyscout result/type fields); the golden diff is then a readability aid, not the correctness proof. Record the affected-provider list + moved columns (feeds the CHANGELOG + atomic-migration note).
- [ ] **Step 5:** full converter test suites green for all touched providers.

---

### Task 4: Runtime raise-guard at `_finalize_output`

**Files:** `silly_kicks/spadl/utils.py` (`_finalize_output` + a new `_assert_chronological_action_id`); Test `tests/spadl/test_finalize_chronology_guard.py`.

- [ ] **Step 1: failing tests**

```python
def test_guard_raises_on_non_chronological_action_id():
    df = _actions(action_id=[0,1], period_id=[1,1], time_seconds=[5.0,2.0], game_id=[1,1])
    with pytest.raises(ValueError, match="non-decreasing|chronolog"):
        _assert_chronological_action_id(df)

def test_guard_passes_chronological_empty_and_nan_time():
    _assert_chronological_action_id(_actions(action_id=[0,1], time_seconds=[2.0,5.0], period_id=[1,1], game_id=[1,1]))
    _assert_chronological_action_id(_actions(action_id=[], time_seconds=[], period_id=[], game_id=[]))
    _assert_chronological_action_id(_actions(action_id=[0,1], time_seconds=[float("nan"), float("nan")], period_id=[1,1], game_id=[1,1]))
```

- [ ] **Step 2:** run, verify fail.
- [ ] **Step 3:** implement `_assert_chronological_action_id` (finite-time rows only; per `(game_id, period_id)`, `action_id`-sorted `time_seconds` non-decreasing → else `raise ValueError`), and call it from `_finalize_output` after projection. Confirm all 8 converters still pass their suites (they must, post-Task-3).
- [ ] **Step 4:** run the guard tests + a full converter-suite pass, verify green. Add a one-line docstring note that this raises by default (severity rationale per the spec).

---

### Task 5: Harden the `action_id`-alone consumers

**Files:** `silly_kicks/spadl/utils.py` (`add_restart_coordinates` sort), `silly_kicks/tracking/_gk_geometry.py` (its precondition/sort if it sorts), + any consumer found by auditing for `sort_values([... "action_id"])` without `time_seconds`; Tests alongside each.

- [ ] **Step 1: audit** — grep the package for `sort_values(["game_id", "period_id", "action_id"])` / action_id-alone sorts; list them.
- [ ] **Step 2: failing test** (mart-shaped, non-chronological `action_id`): feed `add_restart_coordinates` (and each audited consumer) an actions frame whose `action_id` disagrees with `time_seconds`; assert the geometry matches the time-ordered result (currently it would use the wrong neighbour).
- [ ] **Step 3:** change each action_id-alone sort to `(game_id, period_id, time_seconds, action_id)` where `time_seconds` is present; leave `retains()` as-is (already robust — Chesterton's Fence).
- [ ] **Step 4:** run, verify green; confirm no behavior change on already-chronological input (the added `time_seconds` primary key is a no-op there).

---

### Task 6: Documentation + version

**Files:** `CLAUDE.md`, `CHANGELOG.md`, `TODO.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`.

- [ ] **Step 1: CLAUDE.md** — a durable converter-conventions contract: converters are order-insensitive; sort via `sort_actions_chronologically` at the top of the frame before any `.shift()`; `action_id` is chronological `(game_id, period_id, time_seconds)`; enforced by the permutation gate + the `_finalize_output` raise-guard; consumers sort `(…, time_seconds, action_id)`. **Add the GS §2b contract:** GS `EXPECTED_INPUT_COLUMNS` now requires `start_time` (absolute clock); null-clock FOUL `time_seconds` is imputed by a `start_time`-ordered ffill (`event_time` fallback) — measured order-insensitive + byte-identical on real feeds.
- [ ] **Step 2: CHANGELOG** — a new entry (keyed next `PR-Snnn`): the order-insensitivity fix; name the affected providers (from Task 3's recorded set) and the moved columns; **state the ATOMIC-migration requirement** (re-convert all affected-provider data together — bronze, goldens, id-keyed derived tables — never incrementally) and the **retrain trigger**; **state the GS `start_time` input-contract widening + the lakehouse GS-shaper handoff** (byte-identical on real GS feeds → not itself a retrain trigger, but the shaper MUST supply `start_time`/`event_time` or GS conversion raises the missing-column error).
- [ ] **Step 3: TODO** — update the Release line; remove/expand any now-resolved item.
- [ ] **Step 4: version bump (LAST)** — set the next-available version from `main` across all five sites (`pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG`, `TODO`, `uv.lock` hand-edited — do NOT run `uv lock`; it fails on `main`). Number determined now, not claimed earlier.
- [ ] **Step 5: full verification** — `python -m pytest tests/ -m "not e2e" --benchmark-skip` (background+poll); `ruff check` + `ruff format --check` + `pyright` at CI scope. STOP at commit-ready; the user commits.

---

## Self-review notes (author)
- **Spec coverage:** seam §1 → Task 1; gate §3a/§3b + meta → Task 2; sort-at-top §2 (gate-driven) → Task 3; guard §3c → Task 4; consumer hardening §3d → Task 5; docs/impact/atomic-migration → Task 6.
- **Gate-driven, not audit-driven:** Task 2 discovers + records the red set; Task 3 fixes exactly it and iterates against the gate — no reliance on a hand-list (the failure mode of both prior reviews).
- **Red-first:** Task 2 lands the gate red against the known-broken converters and is observed failing before any converter is fixed.
- **Green at task boundaries (resolved):** Task 2 Step 5 marks the observed-broken converters `xfail(strict=True)`, so CI is green after Task 2; Task 3 Step 2 removes each marker as it fixes that converter (a strict xfail that starts passing forces removal). Same red-first/green-between mechanism the SB360 boundary gate used.
- **No commit steps;** version is the final task, number set then.
