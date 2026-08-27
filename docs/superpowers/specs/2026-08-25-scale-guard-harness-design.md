# Scale-guard harness — closing the O(n²) scale blind-spot — design

**Date:** 2026-08-25
**Status:** Design approved (brainstorming); revised after parallel-critic review #1 (sandbox-verified)
**Cycle:** optimization-audit follow-up (successor to ADR-068 / 4.92.0)
**ADR:** next free number, assigned at commit-prep from merged `origin/main`

## 1. Problem

silly-kicks guards performance with two layers, and **both are blind to a scale-only O(n²) regression**:

1. **Structural guards** — `call_counter` / `row_iteration_counter` (`tests/_perf_structural.py`), ~14 `tests/**/*_perf_budget.py`, the ADR-068 perf tests, and `TestCoverShadowComplexityGuard`. Every one is **scale-invariant**: it fixes the fixture size and asserts a call-count invariant ("`lane_control` runs once per receiver, not per (blocker×receiver)"). A `call_counter == 1` assertion passes whether that one call is internally O(n) or O(n²).
2. **pytest-benchmark** (`tests/test_benchmark.py`, run `--benchmark-only` on the CI primary leg) — pure **measurements**, no assertions. Some are already two-size (`add_possessions` at 1500 *and* 10k), but nothing compares them.

There is **no growth/complexity assertion anywhere**. The blind-spot is exactly the shape of the 4.92.0 turnover bug: one `EmpiricalTurnoverValue.fit` call, internally O(n²), `call_counter`-invisible, only *measured* not *asserted* — so it reached a downstream lakehouse report instead of CI. This cycle closes that gap and pays down two follow-ups the ADR-068 review deferred.

## 2. Goals / non-goals

**Goals**
- A **general, reusable, deterministic** harness that catches a scale-only super-linear regression in any primitive that adopts it — no flaky wall-clock, runs on every CI leg.
- Apply it to the highest-risk primitives this audit touched: all 9 `group_rows` sites, the two real quadratics fixed in 4.92.0 (`EmpiricalTurnoverValue.fit`, `_possession_labels`), and `add_possessions`.
- A derivable **forcing function** for the most likely reintroduction path (new `group_rows` callers) plus a convention + ADR for genuinely new loops.
- The mismatched-dtype characterization tests the ADR-068 review flagged (Batch 1): 2 seam-level + one end-to-end per library `group_rows` consumer (7).

**Non-goals**
- An AST rescan-pattern lint (`df[df[k]==v]` inside a loop). **Considered and declined** — ADR-019 deleted its id-compat AST lint because a safe and an unsafe compare are the identical AST, which bred false-positive exemptions; the rescan pattern is more distinctive but an intentional small-inner-collection filter would still false-positive. Recorded in the ADR as declined, not shipped.
- Auto-detecting "which functions have a dominant loop." Not reliably derivable from the AST; the harness guards primitives someone registers, and the ADR states this limit plainly.
- Catching a regression **milder than ~n^1.5**. By design the harness is a *quadratic-ish* detector (see §4.1); a genuine n^1.3 slowdown is out of scope.
- Any change to production behaviour. This cycle is **tests + docs only** (plus the CLAUDE.md/ADR convention). No library output change, no retrain, C4-free (no new aggregator, backend, model, or subpackage).

## 3. Settled design decisions (with rationale)

1. **General harness, not a targeted retrofit.** The goal is "catch the *next* O(n²)," which a retrofit of already-fixed sites cannot do. A reusable seam makes guarding cheap and uniform.
2. **Operation-count growth ratio, via a SCOPED adopter-supplied counter — not an absolute work-bound, not a generic always-on hook.** Assert on the *growth exponent* of a deterministic operation count across sizes. Integer counts ⇒ exact ⇒ never flakes; fixed overhead only biases the exponent *down* (toward "more linear"), so it never false-positives as quadratic. Rejected: an absolute `work ≤ C·n` bound (each `C` is a hand-tuned magic number that drifts — the flaky-budget failure mode moved from time to a constant; can't tell n·log n from n²). Rejected: a **generic always-on** pandas hook that counts for *all* code (Approach 3) — it over-counts unrelated framework ops, misses pure-Python/numba loops, and is un-scopable. The chosen counters patch a **small set of public pandas method entry points** and are **installed only for the measured call** by the adopter — a scoped instrument, not a blanket one. (This is the defensible line s9 asks for: the objection to Approach 3 was *always-on + generic*, not *touches pandas*.)
3. **Forcing = convention + `group_rows`-caller meta-assertion + per-entry non-degeneracy; no AST lint.** A derivable meta-assertion covers the most likely reintroduction path; a non-degeneracy check makes a registered-but-vacuous guard fail; the convention + ADR cover the rest; the lint is declined.
4. **The counter isolates the super-linear-SUSPECT operation; small fixtures suffice ONLY then.** Because we assert on the *ratio*, a quadratic shows exponent ≈ 2 at n = 256–4096 **iff the counted work is the suspect operation alone**. A quadratic that shares its counter with a *large linear co-term* is masked at small n — **measured in review:** `n²+1000n` → exp **1.344**, `n²+10000n` → exp **1.052** at 256/1024, both would pass. Two independent mitigations, in priority order: (i) **the counter counts only the dominant super-linear-suspect op** (rows scanned by a filter, inner-scan steps) and excludes cheap linear bookkeeping (outer-loop iteration), keeping the co-term coefficient ~1 where the estimator is clean; (ii) a **third, larger size** (default `(256, 1024, 4096)`) rescues co-terms up to ~`1000n` at these sizes (measured: `n²+1000n` 1.34→1.51, now caught) but **NOT** a `≥10⁴·n` co-term (`n²+10000n` 1.05→1.11, still masked — out of scope, §8) — so (i) counter-isolation is the primary fix and (ii) is secondary insurance, not a cure. The rescan proxy (work = `n·m + n`, co-term coefficient 1) gives exp ≈ 2.0 and is safe; adopters whose natural counter mixes a large linear term must isolate the suspect op (documented per adopter). The self-test plants a **mixed-term** quadratic to prove this, not just a textbook `n²` (§4.6).
5. **Compiled/numba primitives are guarded via their pure-Python fallback** (same algorithm ⇒ same complexity).

## 4. Architecture

### §4.1 The harness (`tests/_perf_structural.py`)

```python
def assert_subquadratic_growth(measure_work, *, sizes=(256, 1024, 4096),
                               max_exponent=1.5, work_floor=1, label=""):
    """Assert a primitive's deterministic work-count grows sub-quadratically.

    measure_work(n) -> int: build a size-n input, run the primitive with a work
    counter installed, and return the observed integer op-count.

    Estimates the empirical growth exponent from the extreme size pair:
        exponent = log(work[max] / work[min]) / log(size[max] / size[min])
    and asserts exponent <= max_exponent. Requires work[max] >= work_floor
    (non-degeneracy) unless the caller opts into the degenerate path explicitly.
    """
```
- **Reference exponents (sandbox-measured at these sizes, base-independent):** linear **1.00**, n·log n **1.16**, n^1.5 **1.50** (exactly on the default boundary), quadratic **2.00**. Default `max_exponent = 1.5` therefore admits linear + n·log n, sits on the n^1.5 line, and rejects quadratic. **The harness is a quadratic-ish detector, not a general super-linear detector** — a sub-n^1.5 regression passes by design (§2 non-goals, §8).
- **Extreme-pair exponent** (min vs max size) for maximum leverage; the middle size is reported (monotonicity / non-degeneracy sanity) in the failure message. `sizes` accepts **2 or 3+** points — the exponent needs only `sizes[0]`/`sizes[-1]`, so with two sizes the middle report is simply skipped (r3, for adopters like #11 that reuse existing 2-size fixtures).
- **Non-degeneracy is enforced, not optional.** `work[max] >= work_floor` (default 1) — a guard whose counter never fires is a mis-wired guard, not a passing one. A genuinely zero-work primitive opts into an explicit `degenerate_ok=True` with a stated reason; no adopter in §4.4 uses it (they all do measurable work).
- **`max_exponent` is per-adopter overridable** with a mandatory stated reason (a genuinely n·log n or n^1.5 primitive sets e.g. `1.7`); default `1.5`.
- Runs as an **ordinary test on every CI leg** (deterministic, small) — not benchmark-gated. Three sizes at ≤4096 cost a few ms per adopter.

### §4.2 Work counting

**Design rule (from decision 4):** each `measure_work` counts the *dominant super-linear-suspect operation*, isolated from linear bookkeeping. The harness ships ready-made counters:

- **`rows_scanned_counter()`** (new) — a context manager installed only for the measured call, counting three public-pandas seams, each a proxy for "rows touched":
  1. `DataFrame.__getitem__` **when the key is a boolean mask / boolean ndarray** — add `len(self)`. **It MUST discriminate key type (s7/r2):** a bare `df["col"]`, a label list `df[["a","b"]]`, **and an integer array `df[int_list]` — all label/column selection, NOT a row rescan** — return a length-n Series/frame and are **not** scans; counting them would read benign column access in a loop as quadratic. The positional-int *row* path is `.take` (seam #4), never seam #1. Only **boolean** keys count here.
  2. `pandas.core.indexing._LocIndexer.__getitem__` with a boolean/array key — the `.loc[mask]` path (same discrimination).
  3. `DataFrame.groupby(...)` — add `len(self)` per call (the O(n) index construction at `_frame_index.py:38`). **This is what catches an in-loop `group_rows`/`groupby` rebuild (S4):** the natural regression `for item in items: g = group_rows(df, key); g.get(...)` keeps `.take` total at O(n) but rebuilds the groupby m times ⇒ groupby-scan = O(n·m) ⇒ exp → 2.
  4. `DataFrame.take` — add `len(indices)` (the `group_rows` retrieval path). The fixed group_rows site scans n via groupby once + returns each row once via take ⇒ **O(n)**, exp ≈ 1 (non-degenerate — S3: the group_rows sites do NOT hit the degenerate path).
  A raw `df[df[k]==v]`-in-a-loop regression → boolean-mask `__getitem__` = O(n·m) ⇒ exp → 2.
- **Compiled/numba** (`_opp_first_shot_scan`, backing `EmpiricalTurnoverValue.fit`) → `measure_work` calls the **pure-Python** twin with its numpy inputs wrapped in a **counting array** whose `__getitem__` increments a counter — and counts **inner-`j`-loop element accesses only** (the O(n·k) suspect term), NOT the outer-`i` iteration (decision 4: isolate the suspect op, keep the co-term small). See S2 handling in §4.4.
- **`_possession_labels`** → reuse `call_counter` on `pandas.core.indexing._LocIndexer.__getitem__` (the vectorized path uses ~none in the hot path ⇒ constant; the old O(k²) scalar-`.loc` ⇒ grows). Generalizes the existing `test_loc_count_is_scale_independent` into the harness.
- **`scripts/_loader_databricks.load_matches`** → **NOT routed through the exponent harness** (s8: a constant-query claim is an *equality*, not a growth — an exponent with ~0 tolerance fails on a single additive query). Instead a dedicated **equality guard**: patch the SQL-execute seam (`_query_param`), assert `queries(N_a) == queries(N_b)` constant across match counts (the existing `len(seen) == 2` shape), and register that test in `SCALE_GUARDED` alongside the growth tests. The registry holds "scale-guarded primitives," growth-tested or constant-tested.

### §4.3 Registry + forcing function

- **`tests/_scale_guarded.py`** — a `SCALE_GUARDED` registry mapping each guarded primitive's qualname → the test that guards it (growth or constant).
- **`tests/test_scale_guard_registry.py`** — three meta-assertions:
  1. The registry is a **superset of every function that calls `group_rows`** (AST-derived over `silly_kicks/` + `scripts/`, the mirror/purity-registry discovery style; `_frame_index.py` defines but does not call `group_rows`, so the def site is excluded). A new `group_rows` caller with no guard fails CI.
  2. Every registry entry resolves to a real, collected test (self-burning-down: a stale entry fails).
  3. **Every entry's guard is non-degenerate on its own fixture** (`work[max] >= work_floor`, or the equality guard observes ≥1 query). This is the S5 fix: registration alone does not satisfy the gate — a registered-but-vacuous or mis-wired counter fails. "Gate observed working," not "gate observed green."

**Documented limit (in the ADR):** a brand-new rescan that does *not* route through `group_rows` **and** does not rebuild a `groupby` in the loop is not force-caught — no reliable AST signal for "has a dominant loop." Genuinely new loops rely on review + the convention.

### §4.4 First adopters (all guarded this cycle)

| # | Primitive | Guard | Counter / mechanism | Expected |
|---|---|---|---|---|
| 1 | `causal/_confounders.py::_pressure_at_entry` | growth | `rows_scanned_counter` | O(n) |
| 2 | `causal/opportunities.py::build_opportunities` | growth | `rows_scanned_counter` | O(n) |
| 3 | `tracking/defensive_credit/_orchestration.py::compute_defensive_credits` | growth | `rows_scanned_counter` | O(n) |
| 4 | `spadl/_skillcorner_inference.py::infer_defensive_actions` | growth | `rows_scanned_counter` | O(n) |
| 5 | `tracking/_off_ball_runs.py::_off_ball_runs_kernel` | growth | `rows_scanned_counter` | O(n) |
| 6 | `tracking/_gk_identification.py::derive_goalkeepers` (2 group_rows calls) | growth | `rows_scanned_counter` | O(n) |
| 7 | `tracking/_run_values.py::detect_off_ball_runs` | growth | `rows_scanned_counter` | O(n) |
| 8 | `scripts/_loader_databricks.py::load_matches` (2 group_rows calls) | **constant** | SQL-execute equality | queries constant in #matches |
| 9 | `xtgk/_turnover.py::_opp_first_shot_scan` (pure-Python; backs `EmpiricalTurnoverValue.fit`) | growth | counting-array, **inner-`j` only** | O(n·k), ~linear on a break-binding fixture |
| 10 | `vaep/labels.py::_possession_labels` | growth | `_LocIndexer` `call_counter` | O(n) |
| 11 | `spadl/utils.py::add_possessions` | growth | `rows_scanned_counter` (reuses 1500/10k fixtures) | O(n) |

Rows 1–8 are **8 distinct functions covering every `group_rows` call site** (rows 6, 8 call it twice) — ADR-068's "9 sites" at function granularity. Rows 9–11 are the two real quadratics fixed in 4.92.0 plus the aggregator with existing two-size fixtures.

**Adopter #9 — the fixture is the guard (S2).** `_opp_first_shot_scan` is a nested `for i … for j in range(i+1, n)`; its O(n·k)-vs-O(n²) character is decided entirely by when the inner breaks fire (`game_c[j] != gi`, `(t[j]-ti) > window`). On a single-match, `window=inf` fixture **neither break binds and the CORRECT production code is O(n²)** — a naive fixture would fail the *fixed* code. The plan MUST: (a) build the fixture with a **finite window and multiple matches / possession changes** so the breaks genuinely bind (k bounded ⇒ O(n)); (b) **assert the fixture is discriminating** — a break-removed variant of the same scan must measure `exponent > 1.5` on the same fixture, else a green #9 is evidence of nothing; the break-removed variant runs at **capped sizes (256, 512, 1024)**, NOT the adopter's `4096` (r5: a pure-O(n²) pure-Python scan at 4096 is ~8M steps ≈ 1–2 s; exp>1.5 is already visible at 1024, so the pathological run stays cheap and §4.1's "few ms per adopter" holds for the real guard); (c) document the window/match assumptions in the test docstring.

**National-Park follow-up (n10):** `atomic/spadl/utils.py::add_possessions` is a same-shape sibling of #11 that does **not** call `group_rows` (so the meta-assertion won't force it). Guard it too in this cycle (cheap, one more `rows_scanned_counter` growth test) — resolves the ambiguous qualname and closes the latent sibling rather than leaving it invisible.

### §4.5 Batch 1 — the 6 mismatched-dtype characterization tests

One test per `group_rows` site that replaced a **raw `==`** (ADR-068 review, agent 2): assert the intended **ADR-019 canonical** match on a **mismatched-dtype key** (int column vs `str` lookup key, and the reverse), distinguishing it from the old raw-`==` behaviour (which returned empty). Sites: `_off_ball_runs`, `opportunities`, `_skillcorner_inference`, `_confounders`, `_gk_identification`, `defensive_credit/_resolution`. Small, additive; the seam itself is already dtype-safety-tested in `tests/test_frame_index.py`. These ship first in the cycle (they exercise behaviour that already works, so they land green with no production change).

### §4.6 Harness self-test (non-vacuity — the harness must be proven to catch the REALISTIC shapes)

`tests/test_perf_structural.py` (new/extended), each a planted function:
- **passes:** pure O(n); O(n·log n) (exp ≈ 1.16); a genuinely zero-work primitive via the explicit `degenerate_ok` path.
- **fails (the real value) — gated on ROBUSTLY-caught witnesses (R1):** pure O(n²); **mixed-term `n²+100n`** (measured exp **1.89** at `(256,1024,4096)` — a genuine co-term, `100n` is 39% of `n²` at n=256, so it still proves "catches a quadratic sharing its counter with a linear term" per S1c, but with a **+0.39 margin** so a later `sizes` change cannot silently defang the harness's own catch-proof); an **in-loop `group_rows`/`groupby` rebuild** (S4). The RED assertion requires `exp >= 1.6` (a stated margin), **not merely `> 1.5`** — a self-test whose RED witness clears by 0.005 (which `n²+1000n` does: exp 1.505) is itself the silent-scale-failure this cycle exists to kill.
- **documented reference values (monotonicity / boundary, NOT the pass/fail gate — R1):** `n²+1000n` (exp ≈ **1.51**, the on-boundary demonstration) and `n²+10000n` (≈ **1.11**, the §8 out-of-scope large-co-term) are asserted as *reference numbers* so the boundary behaviour is pinned in a test, without a knife-edge margin being the gate.
- **counter correctness (s7):** a function doing heavy `df["col"]` / `df[["a","b"]]` column selection in a loop measures **~0** rows scanned (label selects are not scans); a function doing `df[boolean_mask]` in a loop measures O(n·m).
- **non-degeneracy (S5):** the harness raises when `work[max] < work_floor` without `degenerate_ok`.

Per the "every band needs a test from both sides, and assert the fixture is discriminating" discipline — this proves the harness discriminates *before* any real adopter relies on it. The self-test module is registry-exempt.

## 5. Testing strategy

- **Harness:** the §4.6 self-test (both directions incl. mixed-term quadratic, in-loop rebuild, key-type discrimination, non-degeneracy).
- **Each adopter:** a growth (or constant) test registered in `SCALE_GUARDED`, with non-degeneracy intrinsic (§4.3 meta-assertion #3). #9 additionally carries the discriminating-fixture assertion (§4.4).
- **Registry:** the three §4.3 meta-assertions.
- **Batch-1 dtype tests:** per §4.5, both directions.
- All deterministic, small-fixture, every-leg. No wall-clock assertions anywhere.

## 6. Constraints

- **Deterministic** (integer op-counts, never timings) — no CI flake.
- **No new runtime dependency** (test-only; numba already optional).
- **No production behaviour change** — tests + docs + convention only; no retrain; C4-free.
- **CI-faithful gate before ready-to-commit:** full `pytest -m "not e2e"` (no `--benchmark-skip`), `ruff check` + `ruff format --check` at CI scope (`silly_kicks/ tests/ scripts/`), bare `pyright`; `/final-review` before the single commit.
- **One commit on one feature branch** (per commit policy). The ordering below is review/execution ordering ONLY — never commit boundaries.

## 7. Execution ordering (review-tractable; NOT commit boundaries)

1. **Batch 1** — the 6 mismatched-dtype tests (green immediately; no production change).
2. **Batch 2** — the harness (`assert_subquadratic_growth` + `rows_scanned_counter` with the 3 seams + key-type discrimination) and its §4.6 self-test. The planted mixed-term quadratic + in-loop-rebuild + non-degeneracy cases must be RED against a deliberately-broken harness before the harness is trusted.
3. **Batch 3** — the 11 adopter tests (rows 1–7, 9–11 growth; row 8 constant; #9's discriminating-fixture assertion; the n10 atomic sibling).
4. **Batch 4** — `SCALE_GUARDED` registry + the three meta-assertions, then the ADR + CLAUDE.md bullet.

The whole cycle is a single commit; batches are ordering for review only.

## 8. Known limits (stated, not discovered)

- **Sub-n^1.5 regressions are out of scope by design** — the harness is a quadratic-ish detector (§4.1); n^1.5 sits on the boundary and a milder super-linear passes.
- **A quadratic sharing its counter with a LARGE linear co-term can be masked at small n** (measured: `n²+10000n` → exp 1.05 at 256/1024). Mitigated by counter-isolation (decision 4) + the third size, and every adopter's counter is chosen to isolate the suspect op — but an adopter that *cannot* isolate it must widen `sizes` and document why. `n²+10000n` is not caught at the default sizes; it requires sizes past where the quadratic dominates the co-term (~n≫10⁴).
- **A brand-new rescan that neither routes through `group_rows` nor rebuilds a `groupby` in the loop is not force-caught** (§4.3) — no reliable AST signal for "has a dominant loop."
- The AST rescan-lint is declined (§2), so write-time detection of a new raw-`==`-in-loop is not provided.
- Adopter #9's linear verdict is a property of its (documented, break-binding) fixture, not of the function in the abstract (§4.4).

## 9. Open questions

None outstanding — scope (all 9 `group_rows` sites; dtype tests in-cycle; the n10 atomic sibling folded in), mechanism (Approach 1, scoped counters, suspect-isolation), sizes (`256/1024/4096`), and forcing function ((a) convention + meta-assertion + non-degeneracy) are all resolved. Review #1 (S1–S5, s6–s9, n10) incorporated.
