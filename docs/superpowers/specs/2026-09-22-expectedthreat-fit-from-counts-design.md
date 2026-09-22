# SK-XT-COUNTS — `ExpectedThreat.fit_from_counts` (distributed counts-based fit) + public zone-binning contract

**Status:** DRAFT (for lakehouse review) · **Proposed version:** `<next-free version>` / PR-Snnn / **ADR-NNN** (re-derive at commit-prep) · **Base:** branch off `origin/main` (silly-kicks **4.121.0**).

**Origin:** luxury-lakehouse sk4118 P2 recompute — handoff `D:\Development\_reviews\2026-09-22-sk-expectedthreat-fit-from-counts-handoff.md` (§Why + §follow-up are consumer context; this spec is written against §"What sk should add" / §Constraints / §Tests). Same collaboration pattern as SK-XT-SER (ADR-100) / SK-EXPORT.

---

## 1. Problem

`ExpectedThreat.fit(actions: pd.DataFrame)` requires a materialised DataFrame. The lakehouse ExT-grid producer must therefore pull each of 28 competitions' actions to the Spark driver via `.toPandas()` and `pd.concat` ~1 GB for the global fit (~9.5 M actions) — driver-bound, single-threaded, timed out at 0/28 grids, and a cost-grows-with-corpus scale cliff.

Every quantity `fit` derives is a **zone-count sum**, computable in one distributed Spark `groupBy` pass and **additive across partitions/competitions** (the global grid's counts are the element-wise sum of the per-competition counts). sk should expose a counts-based fit so the producer never pulls raw rows to the driver.

## 2. Scope (additive; no behaviour change to existing API)

1. **`ExpectedThreat.fit_from_counts(...)`** — a public instance method (mirrors `fit`, mutates `self`, returns `self`) that builds the 4 probability matrices from pre-aggregated integer zone counts, then runs the **identical** `value_iteration` → `self.xT` / `self.heatmaps`.
2. **Single-source the count→matrix cores.** Refactor `_scoring_prob` / `_action_prob` / `singh_transition_matrix` into `(a)` a pure `*_from_counts` core and `(b)` a thin `from-actions` wrapper (count, then call the core). `fit(actions)` calls the wrapper; `fit_from_counts` calls the core. **This makes the two paths provably identical rather than two parallel implementations that can drift** — `fit(actions)` output stays **byte-identical** (the SK-xT-1 frozen-oracle parity + `test_singh_path_byte_identical_to_legacy` are the gate).
3. **Public zone-binning contract** — `zones_of` / `flat_indexes_of` + the action-type membership constants, so the lakehouse replicates sk's exact binning + filters in Spark SQL.

**Out of scope:** KDE counts-fit (KDE is not a pure count aggregate — §5); any change to `fit`/`to_dict`/`from_dict`/`rate` behaviour.

## 3. sk internals this rests on (verified vs `origin/main` 4.121.0)

`fit` (`_model.py` def :112, body :132-147): `scoring_prob_matrix = _scoring_prob(actions,l,w)`; `shot_prob_matrix, move_prob_matrix = _action_prob(actions,l,w)`; `transition_matrix = singh_transition_matrix(actions, grid)` (singh) ; `xT, heatmaps = value_iteration(scoring, shot, move, transition, eps=self.eps)`.

**There is NO smoothing / prior** (correcting the handoff's premise): the matrices are raw counts through `_grid._safe_divide` (`np.divide(a,b, where=b!=0)`, 0 elsewhere):
- `_scoring_prob = _safe_divide(goal_count, shot_count)` (`_grid.py:80-85`).
- `_action_prob = _safe_divide(shot, shot+move), _safe_divide(move, shot+move)` (`_grid.py:154-161`), where **move = ALL** `type_id ∈ {pass,dribble,cross}` (`_get_move_actions`; take_on excluded) regardless of result.
- `singh_transition_matrix` (`_transitions.py:12-49`): it FIRST `dropna(subset=["start_x","start_y","end_x","end_y"])` (`:32`), so its denominator `start_counts[i]` = **all moves with a valid START *and* END** originating in i (success + fail); numerator `counts[i,j]` = **SUCCESSFUL** moves i→j; `T[i,j] = counts[i,j] / start_counts[i]` (nz-guarded) → deliberately **sub-stochastic** (`Σ_j T[i,j] = P(success | move from i) ≤ 1`).

**⚠ The `_action_prob` move-count and the Singh denominator are DIFFERENT populations (the correctness crux).** `_action_prob`'s `movematrix` = `_count(move.start_x, move.start_y)` masks **NaN-start only** (`_grid.py:49-50,154`), so a move with a valid start but a NaN end IS counted. The Singh denominator drops any NaN-start-OR-end first (`_transitions.py:32,36`), so that same move is NOT counted. Therefore a single `move_counts` cannot byte-identically feed both — the contract needs **two** move aggregates (see §4).

**Zone binning** (`_grid._get_cell_indexes`): `xi = clip(int(x/105·l), 0, l-1)`, `yj = clip(int(y/68·w), 0, w-1)` (int-floor, half-open, clamped). **Flat index is y-INVERTED** (`_get_flat_indexes`): `flat = (w-1-yj)·l + xi` (row 0 = pitch top, ADR-041). `_count` returns `(w,l)` where `matrix[row,col]`, `row = w-1-yj`, `col = xi` — i.e. `matrix.ravel()` (C-order) == the flat vector.

## 4. API

```python
def fit_from_counts(
    self,
    *,
    shot_counts: NDArray[np.integer],             # (w, l) shots originating per zone (valid start)
    goal_counts: NDArray[np.integer],             # (w, l) goals from shots originating per zone
    move_counts: NDArray[np.integer],             # (w, l) ALL moves (pass|dribble|cross, any result),
                                                  #        VALID START only — feeds _action_prob
    transition_start_counts: NDArray[np.integer], # (w, l) moves with VALID START *and* END (success+fail)
                                                  #        per start zone — the Singh row denominator
    transition_counts: NDArray[np.integer],       # (w*l, w*l) SUCCESSFUL moves (valid start+end)
                                                  #            flat(from)->flat(to) — the Singh numerator
    params: XtParams | None = None,
) -> "ExpectedThreat":
```

**Five counts, because the move filters differ (§3 crux).** `move_counts` (valid-start) feeds `_action_prob`; `transition_start_counts` (valid start+end) is the Singh denominator; `transition_counts` (successful, valid start+end) is the Singh numerator. `shot_counts`/`goal_counts` are valid-start (matching `_scoring_prob`/`_action_prob`'s shot filter).

Semantics (byte-identical to `fit` on the same aggregates):
- `self.scoring_prob_matrix = _scoring_prob_from_counts(goal_counts, shot_counts)` = `_safe_divide(goal, shot)`.
- `total = move_counts + shot_counts`; `self.shot_prob_matrix, self.move_prob_matrix = _action_prob_from_counts(shot_counts, move_counts)` = `_safe_divide(shot,total), _safe_divide(move,total)`.
- `self.transition_matrix = _singh_from_counts(transition_counts, start_counts=transition_start_counts.ravel())` — row-normalise by the **valid-start-and-end** move count (`transition_start_counts` flattened C-order == the singh flat order), nz-guarded; the same `(w·l, w·l)` sub-stochastic matrix `singh_transition_matrix` produces. **NOT `move_counts` — that is the `_action_prob` population and diverges on valid-start/NaN-end moves.**
- `self.xT, self.heatmaps = value_iteration(scoring, shot, move, transition, eps=self.eps)` — the **same** call as `fit`.
- Returns `self`. `l`/`w`/`eps`/`method`/`grid` come from the constructor (as today); shapes are validated against `(self.w, self.l)` / `(self.w·self.l,)²` (raise on mismatch).

**Public binning helpers** (so the lakehouse bins identically in Spark SQL):
- `ExpectedThreat.zones_of(xs, ys) -> tuple[NDArray[int], NDArray[int]]` — vectorised `_get_cell_indexes` (`(xi, yj)`), instance-aware of `self.l`/`self.w`.
- `ExpectedThreat.flat_indexes_of(xs, ys) -> NDArray[int]` — the y-inverted `(w-1-yj)·l + xi`.
- Exposed membership constants (documented, so the `groupBy` filters agree): `MOVE_TYPE_NAMES = ("pass","dribble","cross")`, `SHOT_TYPE_NAME = "shot"`, goal = shot ∧ `result == "success"`, successful-move = move ∧ `result == "success"`. (Sourced from `spadlconfig`, not re-hardcoded.)

## 5. Constraints (encoded)

1. **Singh (count-based) transition ONLY.** `params` requesting KDE (`method == "kde_smoothed"` or a `KDEParams`) → **raise `ValueError`** ("KDE transition is not a pure count aggregate; use `fit(actions)` for KDE, or `fit_from_counts` with default/singh params"). KDE is a density smooth of raw rows, not a zone-count sum.
2. **Orientation (ADR-041).** Counts must be computed on **LTR-oriented** actions (the orientation `fit` assumes). Documented on `fit_from_counts`; the caller owns orienting before binning. The y-inverted flat index is sk's internal storage convention — `flat_indexes_of` hands the caller the exact ordering so `transition_counts` matches.
3. **Raw counts in, sk normalises.** `fit_from_counts` takes RAW integer counts and applies sk's own `_safe_divide` / row-normalisation — the caller must NOT pre-divide/smooth, so both paths stay identical.
4. **Additivity is the contract.** All inputs are sums, so counts of disjoint action sets sum element-wise to the counts of their union — the property the producer relies on for `global = Σ(per-comp counts)`.

## 6. TDD — red-green (write failing first)

New `tests/xthreat/test_fit_from_counts.py`:
1. **Functional equivalence (byte-identical matrices) — with NAMED boundary fixtures.** For a fixture `A`, aggregate the exact counts; `xt_c = ExpectedThreat(l,w).fit_from_counts(**counts)`; `xt_a = ExpectedThreat(l,w).fit(A)`. Assert `np.array_equal` on `scoring_prob_matrix` / `shot_prob_matrix` / `move_prob_matrix` / `transition_matrix` (same integer counts → same `_safe_divide` → exact), and `np.allclose(xt_c.xT, xt_a.xT)` (value_iteration is iterative → fp-tol). `A` MUST contain, each asserted:
   - **(a) a valid-start / NaN-end move** — the D3 divergence trigger; asserts `array_equal` on BOTH `move_prob_matrix` (counts it) AND `transition_matrix` (drops it). Without this the two-vs-one-count bug passes silently.
   - **(b) a zone with shots but zero goals, and a zone with zero actions** — exercises `_safe_divide`'s `where=b!=0` 0-branch (`_grid.py:60`).
   - **(c) a move whose end is off-pitch** (x>105 or y<0) — exercises the `_get_cell_indexes` clamp to `l-1`/`w-1` (`_grid.py:19-20`) in the transition end-cell.
2. **Additivity:** counts(A) + counts(B) fit ≈ `fit(pd.concat([A, B]))` (the global-from-per-comp property).
3. **Round-trip (ADR-100):** `from_dict(to_dict(fit_from_counts(**counts)))` reconstructs the same model.
4. **Zone-binning parity:** `zones_of` / `flat_indexes_of` reproduce `_get_cell_indexes` / `_get_flat_indexes` on a probe grid incl. the edges (x=0, x=105, y=0, y=68, and the clamp at l-1/w-1).
5. **KDE raises:** `fit_from_counts(**counts, params=KDEParams())` → `ValueError`.
6. **Shape guard:** wrong-shaped counts → `ValueError` (non-vacuity).
7. **`fit` byte-identical after the refactor:** the SK-xT-1 legacy-reference tests (`test_singh_path_byte_identical_to_legacy` + `_on_worldcup`) stay green (the count→matrix core extraction is behaviour-preserving).

## 7. Decisions (for the reviewer)

- **D1 — single-source the count→matrix cores.** Extract `_scoring_prob_from_counts` / `_action_prob_from_counts` / `_singh_from_counts`; `fit` and `fit_from_counts` both call them. Prevents the two fits drifting; keeps `fit` byte-identical (parity-gated). (Alternative — a parallel counts reimplementation — rejected: two sources that a future edit desyncs.)
- **D2 — the handoff's "smoothing/prior" premise is corrected:** sk applies none; `fit_from_counts` is raw counts → `_safe_divide` → value_iteration. Documented so the lakehouse does NOT pre-smooth.
- **D3 — THREE distinct move aggregates, because the filters differ** (corrected after review): `move_counts` (valid-**start**, any end) feeds `_action_prob`; `transition_start_counts` (valid **start+end**, success+fail) is the Singh row denominator; `transition_counts` (**successful**, valid start+end) is the Singh numerator. A single `move_counts` for both `_action_prob` and the denominator would DIVERGE from `fit(actions)` on any valid-start/NaN-end move (`_action_prob` counts it, `singh`'s `dropna(start,end)` drops it) — the exact filter-mismatch trap this decision now closes, with a named §6.1 boundary fixture. The sub-stochastic Singh form is preserved.
- **D4 — expose binning (`zones_of` / `flat_indexes_of`) + membership constants, not a per-row UDF.** The lakehouse bins in pure Spark SQL from the documented edge + y-inverted flat formulas; a parity test pins them. (A Python UDF at 9.5 M rows was the perf problem.) **The edge dims are `spadlconfig.field_length` / `field_width`, NOT literal 105/68** — `zones_of`/`flat_indexes_of` read them, the spec documents them as those symbols, and §6.4's parity test pins that the Spark-SQL replication agrees with `_get_cell_indexes` (so a future `spadlconfig` dim change surfaces as a failing parity test, not a silent grid skew).
- **D5 — `params: XtParams`** (sk's actual union, not the handoff's invented `ExpectedThreatParams`); KDE raises (singh-only).
- **D6 — instance method mutating `self`, returning `self`** (mirrors `fit`).

## 8. Ship checklist

- Spec (this) → lakehouse review → plan → lakehouse review → implement → unbiased impl review (reports to `D:\Development\_reviews\`).
- **Human-approval gate before commit:** implementation stops at the commit boundary (diff/file-list shown) and waits for the owner's explicit approval for that specific commit — no `git commit`/push/tag without it. (Same gate as SK-XT-SER; the plan's final task is this stop.)
- Branch off `origin/main` (4.121.0); `feat/expectedthreat-fit-from-counts`.
- `CHANGELOG.md` `Added`; **ADR-NNN** (records D1–D6; extends ADR-100 / ADR-021); `_version.py` → next-free (single line, at commit-prep); `uv lock` follows.
- `NOTICE`: no new citation (Singh xT already cited).
- Additive — **no VAEP/tracking retrain, no re-materialize, C4-free** (a method + public binning helpers on an existing class); `fit`/`rate`/`to_dict`/`from_dict` + the SK-xT-1 frozen-oracle parity byte-identical.

## 9. Non-goals

- No KDE counts-fit (density smoothing is not a count aggregate).
- No change to `fit`/`rate`/`interpolator`/`values_at_points`/`destination_profiles`/`to_dict`/`from_dict` behaviour or the SK-xT-1 parity oracle.
- No Spark/lakehouse code (handoff §follow-up is consumer context only).
