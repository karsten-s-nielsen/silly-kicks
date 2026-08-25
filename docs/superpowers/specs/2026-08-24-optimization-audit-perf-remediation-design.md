# Optimization-Audit Perf Remediation — Design Spec

**Date:** 2026-08-24
**Status:** APPROVED — design + §7 decisions settled by the owner 2026-08-24 (2 review rounds incorporated); execution PARKED pending PR-S163 landing. Plan is the sibling `docs/superpowers/plans/2026-08-24-optimization-audit-perf-remediation.md`.
**Decision record:** ADR-068 (to assign at commit-prep) — codifies the *rescan-in-loop → prebuilt-group-lookup* pattern + the shared `group_rows` seam + the turnover item-B route.
**Trigger:** Full `/optimization-audit` run 2026-08-24 (5-agent fan-out). The lakehouse-reported `EmpiricalTurnoverValue` O(n²) turned out to be one instance of a repo-wide defect class.

---

## 1. Motivation

The audit surfaced ~32 findings. The load-bearing observation is that **one anti-pattern accounts for every Critical finding and roughly a third of the total**:

> **Rescan-in-loop.** A full-table filter — `df[df["frame_id"] == fid]`, `df[(df["a"]==x) & (df["b"]==y)]`, or a positional forward scan — executed *inside a per-item Python loop*, where the same rows could be located in O(1) from a grouping built once. This is O(n·m) (often O(n²)) where O(n) is available.

The class has **nine confirmed members** across four packages, and the turnover bug the lakehouse hit at ~800 K rows / 505 matches is the tenth (item A, already fixed this cycle). These are invisible at test scale (≤1.5 K actions / ≤2.5 K frames) and only bite at real-match scale (43 K–146 K frames, 0.95 M–3.2 M rows) — which is why a downstream consumer, not CI, reported the first one.

This cycle eliminates that class and the adjacent, output-preserving perf debt the audit found (unvectorized join-backs, loop-invariant recompute, a double XML parse, and script-driver resilience/RAM defaults), and codifies the pattern so it does not re-enter.

### 1.1 Guiding constraints (from the owner + house conventions)

- **Output-preserving.** Every library change reproduces the current output **byte-for-byte**. None is a behavior change; none is a VAEP/model-retrain trigger. (The one algorithmic rewrite, turnover item B, ships behind an exact-equivalence oracle.)
- **TDD, hexagonal, gold-standard.** Core functions stay pure (pandas in, pandas out; zero I/O; policy at the edge). Every fix lands red-first with a parity test **and** a structural call-count guard (the house convention — no wall-clock asserts).
- **No new runtime dependency** for Batches 2–6. Turnover item B's route is chosen to avoid one (pure-numpy); `numba` is a documented fallback only.
- **DRY / long-term.** The recurring rescan fix is consolidated behind one small, tested, dtype-safe seam rather than nine ad-hoc `groupby` calls, so the pattern has a single home and a single guard.
- **ADR-019 dtype-safety.** Any id used as a group/lookup key is canonicalized; a raw `==`/tuple lookup that silently mis-resolves across dtypes is itself a defect (and one such site — `_confounders.py:72` — mixes raw `==` on ids with the rescan).

## 2. Scope

**In scope (owner-selected: Batches 2–6 + turnover item B):**

| Batch | Theme | Sites |
|---|---|---|
| 2 | Rescan-in-loop O(n²) | `causal/opportunities.py:264`, `defensive_credit/_resolution.py:51`, `_gk_identification.py:106,164`, `causal/_confounders.py:72`, `_off_ball_runs.py:127`, `_run_values.py:248`, `spadl/_skillcorner_inference.py:59` |
| 3 | Possession O(k²) label loop | `vaep/labels.py:288-330,333-375` |
| 4 | Loop-invariant / uncached recompute | `pitch_control/_surface.py:79`, grid rebuild in `_spearman/_fernandez_bornn/_voronoi`, `causal/matching.py:227`, `spadl/utils.py:239` (+atomic dup) `add_gk_role`, `_cover_shadows.py:1113` |
| 5 | Parse ports (parity-gated) | `providers/sportec/parse.py:486,527` (double XML parse), `spadl/_skillcorner_inference.py` merge_asof (shared with Batch 2) |
| 6 | Script driver resilience / RAM | `scripts/_loader_pining_to_cache.py`, `scripts/_xtgk_comparability.py`, `scripts/calibrate_tracking_defaults.py::_load_fold`, `scripts/_loader_databricks.py:150` |
| B | Turnover perf rewrite | `xtgk/_turnover.py::_opp_first_shot_after_turnover` |

**Explicitly out of scope (Batch 7 — deferred, separate decision):**
- Ghost-GK default backend (#13) — conflicts with in-flight `_ghost_gk.py` (PR-S163, 154 lines); revisit after that merges.
- DAS all-frame ~394 h cost guardrail.
- The toy-scale benchmark blind spot (the broad test-infra investment). *Note:* the per-fix structural guards below are scale-**independent** by construction, so Batches 2–6 are safe without closing this; the blind spot is about catching the *next, unknown* regression, not these.

**Low-tier deferred (tracked, not this cycle):** `Counter(series)`→`.value_counts()` across 7 converters (diagnostics-only), `coverage_metrics` tally, `features.py:324`/`_gk_resolve.py:196`/`_elastic_sync.py:301` iterrows scatter, metrica/sportec card-pair loops, `_pressure_levels.py:106`, `_transitions.py:174`, `xtgk/_metric.py:150` (self-acknowledged, scope-bounded), `_kernels.py` scatter loops (**deliberate ADR-020 crash-safety — do not "fix"**), `_press_commitment.py:167`. These are byte-identical, low-magnitude cleanups that can ride a later PR; folding a subset into Batch 4 is at the implementer's discretion but not required.

## 3. The shared seam — `group_rows` (Batch 2/5 foundation)

### 3.1 Rationale

Nine sites need the same primitive: *"given a DataFrame and key columns, get the rows for a key in O(1), after paying O(n) once."* The codebase's only frame pre-index (`_pre_index_frames`) is a numba-array packer, not a row lookup, so it is not reusable here. Nine inline `groupby` calls would (a) duplicate a dtype-safety subtlety nine times, (b) leave nine independent guard surfaces, and (c) give the pattern no documented home. One tested seam is the DRY/long-term choice.

### 3.2 Placement and signature

New leaf module `silly_kicks/_frame_index.py` — no intra-package dependencies, importable by `spadl/`, `tracking/`, `vaep/`, `causal/`, `providers/`, and `scripts/` without inverting layering.

**Private (`_`-prefixed), with a YAGNI justification (F7).** The correct precedent is the PRIVATE leaf utils `_geometry.py` / `_polygon.py`, NOT `id_compat.py`: `id_compat`'s underscore was *deliberately removed* (PRIVATE_CONSUMERS.md "Retired entries") precisely because ADR-019 makes it a **mandatory** seam every consumer must route through, and "a mandatory seam is public API by definition." `group_rows` is different in kind — an internal perf primitive with **no cross-repo consumer** and no "must-depend-on-public-only" pressure. So it follows the `_model_eval` / `lane_control` precedent instead: stay private, record in the **in-repo (first-party) consumers** table of `docs/PRIVATE_CONSUMERS.md`, exit condition "promote to `silly_kicks.__all__` only if a cross-package/cross-repo consumer appears." Public promotion is Open Decision §7.1 if the owner prefers it. (Corrects the draft's "position of `id_compat`" framing, which had the public/private precedent backwards.)

```python
# silly_kicks/_frame_index.py
from __future__ import annotations
import pandas as pd
from silly_kicks.id_compat import canonical_id  # ADR-019 dtype-safe key normalization

class RowGroups:
    """O(1) row-group lookup over `df`, grouped by `by` — replaces the rescan-in-loop
    anti-pattern (ADR-068).

    Build ONCE (O(n)); look up per item (O(1)). Backed by `groupby().indices`
    (key -> positional int array) so NO group frames are copied at construction —
    memory is one int array of length n, not a partition of the whole table.

    Keys are canonicalized (ADR-019): an `Int64`-typed group column is still found by
    a Python-`int` or `str` lookup key. A missing key returns an EMPTY frame carrying
    `df`'s columns and dtypes (never `KeyError`) — matching the `df[df[k]==v]`
    semantics it replaces, so downstream `.empty` / column access is unchanged.
    """
    def __init__(self, df: pd.DataFrame, by: str | tuple[str, ...]) -> None:
        self._df = df
        self._by = (by,) if isinstance(by, str) else tuple(by)
        gb = df.groupby(list(self._by), sort=False)
        # canonicalize every group key so lookups are dtype-agnostic (ADR-019)
        self._indices = {self._canon(k): v for k, v in gb.indices.items()}
        # F1(a) COLLISION GUARD: canonical_id collapses 366/366.0/"366" -> "366", so a mixed-dtype
        # key column holding two spellings of the same id would silently overwrite one group and
        # lose its rows (the raw `df[df[k]==v]` rescan kept them separate). Refuse rather than lose.
        if len(self._indices) != len(gb.indices):
            raise ValueError(
                f"group_rows: {len(gb.indices) - len(self._indices)} group key(s) collapsed under "
                f"ADR-019 canonicalization on columns {self._by} -- the key column mixes dtypes "
                f"(e.g. int 366 and str '366'). Clean the key dtype before grouping."
            )

    def _canon(self, key):
        if len(self._by) == 1:
            return canonical_id(key)
        return tuple(canonical_id(k) for k in key)  # multi-key: `key` is a tuple

    def get(self, *key) -> pd.DataFrame:
        k = key[0] if (len(key) == 1 and len(self._by) == 1) else tuple(key)
        pos = self._indices.get(self._canon(k))
        return self._df.take(pos) if pos is not None else self._df.iloc[:0]

    def __contains__(self, key) -> bool:
        # single-key: `key` is the scalar; multi-key: `key` MUST be a tuple (documented).
        return self._canon(key) in self._indices


def group_rows(df: pd.DataFrame, by: str | tuple[str, ...]) -> RowGroups:
    """Convenience constructor. See `RowGroups`."""
    return RowGroups(df, by)
```

Notes:
- `groupby().indices` returns positional arrays and copies no frame data — the memory cost is O(n) ints, negligible vs the 0.95–3.2 M-row tables. `df.take(pos)` materializes only the requested group.
- **`get(*key)` takes positional args (`get(2, 10)` for a 2-key group); `__contains__` takes ONE arg, so a multi-key membership test passes a tuple: `(2, 10) in groups`.** (F8 — spec/plan aligned; `__contains__` has no dead ternary.)
- `sort=False` is a small perf win (skip sorting the group-key dict); it does NOT affect within-group row order — `.indices` preserves each group's original source order regardless of `sort`. (F9 — corrected rationale; the previous note attributed order-preservation to `sort=False`, which was wrong.)
- Canonicalization reuses `id_compat.canonical_id` (already the ADR-019 authority), so this seam does not invent a second id-normalization rule.
- The empty-frame-on-miss semantics are load-bearing: the sites it replaces (`df[df[k]==v]`) yield an empty frame on a missing key, and downstream code branches on `.empty`. A `KeyError` default would change behavior.

### 3.2a Byte-identity is CONDITIONAL — the precondition (F1b)

`group_rows` is byte-identical to `df[df[k]==v]` **only where the key column is already single-dtype-clean.** Canonicalization is a genuine semantic change on a *dirty* (mixed-dtype object) key column: it MERGES `366` and `"366"` that the raw `==` kept apart. Two consequences the consumers must honour:

1. **Where the key column is clean** (single dtype — the normal case for tracking `game_id`/`period_id`/`frame_id`/`team_id`), canonical == raw and the fix is byte-identical. Each consumer's parity fixture must either use a clean key column (proving no change) OR the site must be independently established as clean.
2. **Where the old code's raw `==` MIS-resolved dtypes** (e.g. `_confounders.py:72`, flagged by the audit as mixing raw `==` on ids), routing through `group_rows` CHANGES output — that is an ADR-019 correctness *improvement*, NOT parity. Such a site is NOT a byte-identical task: it is an explicit, owner-approved behavior change, with a fixture that exercises the dirty column and a test proving the new (canonical) resolution is the right one. Do not launder it through a "byte-identical" parity test on a clean fixture. (See F4.)

### 3.3 Guard strategy — two distinct guards, named honestly (F2, F3)

Each fix carries **two** tests with **different jobs**. The draft mislabeled both "red-first"; corrected here.

**(1) Parity test = an INVARIANCE guard, expected GREEN throughout.** It captures current output and asserts the new code equals it. It is green before the change (trivially — code unchanged) and green after (if the refactor preserved output); it goes RED only if the refactor *breaks* output. It is NOT "red-first" — do not instruct an executing agent to "confirm it FAILS first." Its job is to catch a regression at the moment the implementation step lands.

**(2) Structural guard = the ANTI-REGRESSION guard, made genuinely red-CAPABLE per site.** Its job is to fail if the O(n²)/invariant-recompute ever returns. Red-capability differs by fix class:

- **`call_counter`-on-a-NEW symbol the fix introduces (Batch 2 `group_rows`): mutation-proof, NOT red-first.** Pre-fix `group_rows` isn't imported, so "fails before the fix" is an AttributeError = red-for-the-wrong-reason. Demonstrate red-capability with a **mutation test**: move the `group_rows(...)` call back inside the loop, confirm the count rises (`n > 1`). State it as a post-fix invariant + mutation proof.
- **`call_counter`-on-a-PRE-EXISTING library symbol (Batch 4/5): GENUINELY red-first (R2).** `np.unique` (runs `n_seeds=200×` pre-fix in `_cluster_reassign`), `np.meshgrid`, `ids_isin` (K× in `add_gk_role`), `RegularGridInterpolator` (per-`.at_*`-call), `lane_control` (per-receiver), `iterparse` (2× in the sportec parse) all EXIST pre-fix and are called too many times, so `call_counter(...); assert n == 1` genuinely FAILS pre-fix (high count) — a clean red-first. No mutation test needed; the pre-fix run IS the red.
- **`row_iteration_counter` (sound only where an actual `.iterrows()`/`.apply(axis=1)` exists).** Batch 4 join-backs (`add_team_shape`) qualify. **It is VACUOUS for the three pure-Python-loop rewrites (Tasks 8, 9, 20)** — they use `for i in range(n)` / `for j_pos in idx[i+1:]` with `.loc`/mask/`.iloc[0]`, which `row_iteration_counter` cannot see (`_perf_structural.py:56-72` patches only `apply(axis=1)`/`iterrows`/`itertuples`). It reads 0 before AND after → no signal. Do NOT claim it as the guard there.
- **The three genuine algorithmic rewrites (Tasks 8/9/20) get a purpose-built guard OR the parity/exact-equivalence oracle is designated the SOLE guard:**
  - **Task 9 (VAEP possession):** spy `pandas.core.indexing._LocIndexer.__getitem__` via the house helper (`call_counter(monkeypatch, pandas.core.indexing._LocIndexer, "__getitem__")` — its module arg accepts a class). Genuinely red-first (O(k²) `.loc` gets pre-fix). **Assert SCALE-INDEPENDENCE, not a `~0` threshold (R3):** run a k=4 and a k=12 possession and assert the `.loc` count does not scale with k (bounded/constant) — that is the real O(k²)→O(k) invariant and it is robust to incidental `.loc` in the vectorized path.
  - **Task 8 (SkillCorner):** `pd.merge_asof` called once — this IS genuinely red-first (0 calls pre-fix → 1 post), but WEAK as an anti-regression (it proves the new primitive runs, not that the per-row loop is gone), so the committed golden parity fixture remains the primary guard.
  - **Task 20 (turnover):** there is no pandas primitive to spy on a numpy double loop. The **exact-equivalence oracle is the SOLE guard** (spec §Batch B) — do not assert a vacuous structural guard here.

**Fixtures MUST have ≥2 groups** (frames/possessions/games) or a rescan and a lookup cannot disagree — a single-group fixture is a false green (both-sides rule).

## 4. Per-batch design

Every task below is: (1) write a **parity test** = an INVARIANCE guard capturing current output on a fixture that has ≥2 groups (green before AND after — it goes red only if the transform breaks output; NOT expected to "fail first", per §3.3/F3); (2) apply the transform; (3) confirm parity green; (4) add the **structural guard** per §3.3 — genuinely red-first for Batch 4/5 (pre-existing primitive over-called pre-fix), mutation-proof for Batch 2 `group_rows` (new symbol), or the oracle-as-sole-guard for the pure-loop rewrites (Tasks 8/9/20). Fixtures must contain **multiple frames/possessions/games** — a single-group fixture cannot distinguish O(n) from O(n²) and is a false green (the CLAUDE.md both-sides rule).

### Batch 2 — rescan-in-loop

- **`causal/opportunities.py:259-284`** — the per-`(game,period)` loop already sorts `g` by `(time_seconds, frame_id)` and iterates `frame_keys` in that order. Replace `grp = g[g["frame_id"] == fid]` (line 264) with a `group_rows(g, "frame_id")` built once before the `frame_keys` loop; `grp = groups.get(fid)`. Order of iteration is unchanged (still driven by `frame_keys`). Parity: identical spells DataFrame.
- **`defensive_credit/_resolution.py:51`** — `resolve_responsible_defenders` takes `frames` + `frame_id` and does `frames[frames["frame_id"] == frame_id]` per call, called per-action (`_orchestration.py:115`) × per-rule (`:138`). Thread a prebuilt lookup: build the lookup **once** in `compute_defensive_credits`, store it on `RuleContext`, and have `resolve_responsible_defenders` accept an optional `frame_groups: RowGroups | None` (falling back to the current filter when `None`, preserving the unit-test call path). **KEY ON `frame_id` ALONE (F4)** — `group_rows(frames, "frame_id")` — to reproduce the current filter's semantics EXACTLY and stay byte-identical. Do NOT add `period_id` to the key in this task: keying on `(period_id, frame_id)` would be a *behavior change* (it differs from the old output iff `frame_id` is not unique across periods), and folding it into a "byte-identical" refactor laundered through a clean fixture is exactly what F4 forbids. **Separately raise to the owner**: is `frame_id` per-period or globally unique in these frames? If per-period, the old `frame_id`-alone filter is a latent cross-period bug — but fixing it is an explicit, separately-tested behavior change (its own task, with a fixture that has a colliding cross-period `frame_id` and a test proving the `(period,frame)` resolution is correct), NOT part of this refactor. Structural guard: `call_counter` on `group_rows`, assert `n == 1` per `compute_defensive_credits` call (not per action×rule); demonstrate red-capability by the mutation test (§3.3).
- **`_gk_identification.py:105-169`** — `derive_goalkeepers` scans `player_rows` per `(game,team)` (line 106) and `frames_out` per GK (line 164). Replace the team scan with `group_rows(player_rows, ("game_id","team_id"))`; build the GK write via a `group_rows(frames_out, ("game_id","team_id","player_id"))` once and assign through the union of positional indices, instead of a fresh 3-condition boolean mask per GK. **Byte-identity precondition (F1b/F4):** the raw `==` id comparisons this replaces flow through `group_rows`' canonicalization, which is byte-identical ONLY if the `game_id`/`team_id`/`player_id` columns are single-dtype-clean. Establish that first (assert the columns' dtype on the fixture); if any id column is mixed-dtype, routing through canonicalization is a deliberate ADR-019 *correctness change*, not parity — split it out as an owner-approved behavior change with a dirty-column fixture. `group_rows`' construction-time collision guard (§3.2) will RAISE rather than silently lose rows on a mixed-dtype column, so a dirty column fails loud at build, not silently at lookup. Parity on a **2-game × 2-team clean-id** fixture. Severity caveat: production often passes a small `teams` subset, but the function supports `teams=None` (whole batch); the fix helps both.
- **`causal/_confounders.py:61-97`** — `_pressure_at_entry` scans the full `players` table per spell (line 72). Mirror the sibling `_defending_team_id` (which already pre-groups, with a comment naming this exact O(n²)): build `group_rows(players, ("game_id","period_id","frame_id"))` once, `grp = groups.get(gid, per, fid)` per spell. The inner `[same_id(t, team) for t in grp["team_id"]]` (line 73) stays (it filters within a small per-frame group) but can become a vectorized `ids_match(grp["team_id"], team)` mask (ADR-019). Parity on a multi-spell fixture.
- **`_off_ball_runs.py:127` & `_run_values.py:248`** — both do `frames[frames["game_id"] == game_id]` inside `for game_id, game_actions in actions.groupby("game_id")`. Build `group_rows(frames, "game_id")` once before the loop; `game_frames = groups.get(game_id)`. `_run_values.py` already uses `ids_equal` for the compare — the lookup subsumes it. Parity on a **2-game** fixture. (Smaller blast radius: O(n_games), not O(n_actions).)
- **`spadl/_skillcorner_inference.py:59-94`** — `infer_defensive_actions` re-filters `obe_regains` with a 3-condition mask + sort + `.iloc[0]` per defensive-start row. This one is a *windowed nearest-after* join, not a keyed lookup → the right tool is `pd.merge_asof` grouped by `(period, team_id)` (direction/tolerance per the current sort+first-match semantics), not `group_rows`. Parity is critical here (it changes `actions` output shape); gate on a committed SkillCorner fixture and treat any diff as a blocker. (Shared with Batch 5's parity discipline.)

### Batch 3 — possession O(k²) labels

- **`vaep/labels.py:288-375`** — `_scores_possession`/`_concedes_possession`: nested `for i,pos … for j_pos in idx[i+1:]` with scalar `.loc`, and **no `break` on the `xg_column` path** (unbounded). Vectorize per possession group, mirroring `_scores_time`/`_concedes_time`. **F6 — the condition is TEAM-OF-POS-relative** (`_same_team_scalar(team[pos], team[j])`, `labels.py:307,315`), so a single reverse-cumulative-OR/max is correct ONLY if a possession is single-team — and it is NOT: `add_possessions`' `retain_on_set_pieces` / `merge_brief_opposing_actions` carve-outs and native provider `possession_id` can place opposing-team actions inside one possession. So the vectorization must be team-aware: a possession has AT MOST two teams (usually ONE — the carve-outs add a second only sometimes), so precompute one reverse-cumulative aggregate **per team PRESENT** in the group (do NOT hard-code two — a one-team possession is the common case) of the eligible downstream goal (for `_scores`: same-team goal xG / other-team owngoal xG; symmetric for `_concedes`), then index each position by ITS OWN team. Bool path = reverse-cumulative-OR; xG path = reverse-cumulative-**max** (NOT first — a later higher-xG same-team goal must win; the old loop with no break already takes the max via `max(result, xg)`). **Also include the self-scoring second pass** (`labels.py:322-328,367-373`: the goal action itself scores / the owngoal action itself concedes) — it is separate from the pairwise pass. Parity: exhaustive vs current output on `tests/vaep/test_labels_windowing*` for **both** `xg_column=None` and `xg_column=set`, and the oracle fixture MUST contain (i) a possession with BOTH teams' actions (exercises the team-aware split — a single-team fixture is a false green for F6), (ii) the goal/owngoal-scores-itself pass, and (iii) a possession with multiple downstream same-team goals of DECREASING xG (proves max-not-first on the xG path). Structural guard (F2 — `row_iteration_counter` is VACUOUS here, plain `.loc` loop, not iterrows): spy `pandas.core.indexing._LocIndexer.__getitem__` via the house helper (`call_counter(monkeypatch, pandas.core.indexing._LocIndexer, "__getitem__")`). **Assert SCALE-INDEPENDENCE (R3), not `== 0`:** run a k=4 and a k=12 possession and assert the `.loc` count does not scale with k — that is the robust O(k²)→O(k) invariant (a `== 0` threshold is brittle to incidental `.loc` and does not actually prove the win). This is the highest-effort Batch 2–6 item; the possession-window path is public + tested (DTAI-extended training), so it is worth the care.

### Batch 4 — loop-invariant / uncached recompute (all byte-identical)

- **`pitch_control/_surface.py:79-115`** — `PitchControlSurface.at_point`/`.at_points` build a new `RegularGridInterpolator` every call, though the surface is a frozen dataclass shared across xfn families via `PitchControlCache`. Cache the interpolator on the instance: since the dataclass is frozen, build lazily on first use via `object.__setattr__` (or a module-level `functools.lru_cache` keyed on `id(self)` is *not* safe — use instance caching). Parity: identical interpolated values (same interpolator, same grid). Structural guard: `call_counter` on `RegularGridInterpolator`, assert one construction per unique surface across a multi-query pass (e.g. `_obso.py:408` + `:456` querying the same `event_surface` → 1, not 2).
- **`_spearman.py:144`, `_fernandez_bornn.py:96`, `_voronoi.py:42,61`** — `np.linspace`/`np.meshgrid`/`targets` rebuilt per canonical-surface call though they depend only on `params.grid_cells_x/y`. Memoize on `(grid_cells_x, grid_cells_y)` via a module-level `functools.lru_cache`-wrapped helper returning read-only arrays (copy on use if a caller mutates — verify none does). Parity: identical grids. Structural guard: `call_counter` on `np.meshgrid`, assert bounded (1 per unique grid config across a pass).
- **`causal/matching.py:227-266`** — `_cluster_reassign` recomputes `np.unique(ids)` + O(n_clusters) boolean scans per seed inside `placebo_shift`'s `for s in range(n_seeds)` (n_seeds=200). Hoist the invariant grouping (`factorize`/`argsort` of `cluster_ids`) out of the seed loop; per seed apply only the permutation. Parity: identical placebo distribution for a fixed RNG seed. Structural guard: `call_counter` on `np.unique`, assert `n == 1` (not `n_seeds`).
- **`spadl/utils.py:239-270` `add_gk_role` (+ `atomic/spadl/utils.py:199-230` dup)** — inside `for k in range(1, distribution_lookback_actions+1)`, `cur_is_known_gk = ids_isin(cur_player_arr, goalkeeper_ids)` and four array builds are k-invariant. Hoist them above the loop. Latent at default `K=1` but activates for any K>1. Fix once per file (both). Parity: identical `gk_role` for K∈{1,3}. Structural guard: `call_counter` on `ids_isin`, assert `n == 1` (not K), on a K=3 call.
- **`_cover_shadows.py:1113-1131`** — the baseline `n_blocked` loop calls un-batched `lane_control()` per receiver, though the vectorized `_lane_received_batched` (variant-0) computes the identical baseline at line ~1257. Reuse the batched variant-0 output for the baseline instead of the per-receiver `lane_control()` call. **This one carries a documented bit-identical constraint** (spec §5: "kept unchanged so n_blocked_receivers stays provably bit-identical") — the parity test is the whole point; land it only if variant-0 reproduces the loop's `n_blocked` exactly on the committed cover-shadow fixtures, else defer. Structural guard: `call_counter` on `lane_control`, assert it is no longer called per-receiver in the baseline path.

### Batch 5 — parse ports (parity-gated, non-output-changing)

- **`providers/sportec/parse.py:481-540`** — `_parse_positions_xml` runs `ET.iterparse` over the position file **twice** (pass 1 ball, pass 2 players). The docstring already establishes the ordering guarantee (ball FrameSets follow player FrameSets) that makes a single streaming pass viable: collect player rows in one pass with `ball_*` columns blank, accumulate `ball_by_frame` as the stream progresses, then **one vectorized `merge` on `(period, frame)`** — reproducing the identical join the second parse + per-row `dict.get()` produces. This is a merge on identical keys → byte-identical output, not a semantics change. Gate on the existing golden-parity fixture (`tests/providers/sportec/test_parse_port_parity.py`, `idsse_slice/`, `SOURCE_SHA`) — treat any diff as a hard blocker. Structural guard: `call_counter` on `xml.etree.ElementTree.iterparse`, assert `n == 1` (was 2). This is a **retrain-neutral** change (output identical), so no calibration/VAEP trigger.
- **`spadl/_skillcorner_inference.py`** — the `merge_asof` rewrite (shared with Batch 2) belongs to this parity discipline: golden-gate on a committed SkillCorner fixture.

### Batch 6 — script driver resilience / RAM (owner-run drivers)

These are not library code; they run on DGX / HF Jobs (fixed RAM, wall-clock timeout). "Output-preserving" here means the produced artifact/cache is identical; the change is resilience + memory, not results.

- **`scripts/_loader_pining_to_cache.py:44-79`** — no skip guard: a crash re-downloads+re-parses the whole corpus. **First verify it is not deliberately ADR-052-exempt** (it writes a cache, not a registered research artifact — it may legitimately sit outside the artifact-driver population; the fix is still warranted). Add the established skip pattern (mirror `_load_xt_corpus_pining` in `calibrate_tracking_defaults.py`): walk match IDs, skip if the per-match cache dir exists, thread a `--cache-dir`. Prefer adopting `scripts/_driver.py::for_each` if it fits the cache-write shape (per ADR-052). **Sequencing:** this feeds `train_ghost_gk.py`, which PR-S163 is actively reshaping (+12 lines) — rebase on its final loader/CLI shape; no edit conflict, but do this site last of Batch 6.
- **`scripts/_xtgk_comparability.py:79-133`** — no `--cache-dir`; every match's full tracking artifact is downloaded **twice** (grid-fit pass + `_collect` scoring pass), no resume. Add a `--cache-dir` pass-through and reuse the fetched artifact across both passes; adopt `for_each` for resume.
- **`scripts/calibrate_tracking_defaults.py::_load_fold` (110-145)** — materializes the whole corpus in RAM before any Optuna trial; `tracking_limit`/`max_per_provider` default to `None` (load everything). Change the **default** to a safe cap (or fail-loud with a required explicit `--no-cap` opt-out) so an operator does not silently OOM a fixed-RAM container. This is a default-safety change, not an algorithm change; document it in the CLI help + CHANGELOG (Hyrum: an operator relying on the unbounded default must opt in).
- **`scripts/_loader_databricks.py:150-157`** — 2N sequential SQL round-trips (`SELECT * … WHERE match_id=%(mid)s` per match, ×2 tables). Batch with `WHERE match_id IN (…)` + client-side `groupby`. Conditional path (`--source databricks/auto`); medium gain. Parity: identical loaded frames.

### Batch B — turnover perf rewrite

- **`xtgk/_turnover.py::_opp_first_shot_after_turnover`** — pure-Python O(n²) double loop, the fit's dominant cost at ~800 K rows. **Route (recommended): pure-numpy possession pre-aggregation** — no new dependency, matches the possession-bound semantics natively:
  1. Build per-`(game_id, possession_id)`: the possessing `team_id`, the possession start time, the possession's **first shot's xG**, AND **that first shot's `time_seconds`** (F5 — the timestamp is load-bearing; the current loop bounds on `t[shot]-t[turnover] > window_seconds` per action, so a possession-level "has a shot at xg=X" that drops the shot's time cannot honor a finite window and would wrongly credit a beyond-window shot). `first`, not `max` — matches the loop's `break` on first shot. One `groupby` pass, O(n).
  2. Order possessions per game by start time.
  3. Per turnover, walk the game's ordered possession list from the turnover's possession to the **first opponent possession run**, honoring the exact break/continue rules — critically, **the scan spans *consecutive* opponent possessions until a shot / ball-back-to-loser / window/game bound**, not just "the first opponent possession" (the semantics the naive one-liner would get wrong). For a **finite `window_seconds`**, credit the first-shot xG only if `first_shot_time - turnover_time <= window_seconds` (and the run isn't cut by ball-back first); for `window_seconds=None` (possession-bound, the production default) there is no time cap within the run.
  - This composes with item A (already fixed): the canonical sort item A added is a *precondition* that makes the possession-ordered walk correct.
  - **Route decision is measured, not assumed:** before finalizing, benchmark pure-numpy vs the current loop on a realistic synthetic corpus; `numba @njit` of the existing loop verbatim is the documented fallback if pure-numpy underperforms or the vectorization proves semantically fragile (fallback adds `[numba]` coupling to a currently-pure path — a cost, hence second choice).
  - **Exact-equivalence oracle:** a regression test asserting the new `_opp_shot_xg` per row is **identical** to the current loop on a synthetic multi-game corpus with turnovers/shots/possession churn. The fixture MUST include: the consecutive-opponent-possession run; `window_seconds=None`; AND a finite `window_seconds` with a shot placed JUST BEYOND the window (F5 — the plan's `window_seconds=10.0` case is vacuous unless a beyond-window shot exists to be excluded), plus a within-window shot for contrast. Land only when identical.
  - Structural guard: `row_iteration_counter`-style — assert the O(n²) Python double loop is gone (0 per-turnover forward scans; a single `groupby` pass instead).

## 5. Testing strategy (unified)

- **Parity is the primary gate.** Each library fix reproduces current output byte-for-byte. Two acceptable forms: (a) capture the pre-fix output on a committed fixture and assert equality; (b) compute a reference in-test with the old implementation and assert equality. Fixtures MUST have ≥2 groups (frames/possessions/games) so a rescan and a lookup can disagree — a single-group fixture is a false green.
- **Structural guard is the anti-regression.** Per the house convention (`tests/_perf_structural.py`), assert the *invariant the fix establishes*: the grouping/interpolator/`np.unique`/`ids_isin`/`iterparse` runs a bounded number of times (usually 1). `row_iteration_counter == 0` applies ONLY where the fix removes an actual `.iterrows()`/`.apply(axis=1)` (e.g. `add_team_shape`); it is VACUOUS for the pure-Python-loop rewrites (Tasks 8/9/20 use `.loc`/mask/numpy loops it cannot see) — those use the §3.3 per-site guard (`_LocIndexer` scale-independence / `merge_asof`-once / the exact-equivalence oracle). These invariants are scale-independent, so they hold at 22 players and at 3 M rows — this is what makes Batches 2–6 safe without the (deferred) scale-benchmark.
- **Both-sides rule.** Every parity test also asserts the *failing* side: a mutation that should change the output (e.g. a different frame's rows) must move the number, so the test is not vacuously green.
- **No new e2e.** All fixtures committed → tests run in the regular suite (not `@e2e`, not `@slow` — these are behavioral-contract guards that must run on all legs).
- **Full-suite discipline:** run the CI-faithful invocation (`pytest -m "not e2e"`, no `--benchmark-skip`) and lint at CI scope (`ruff check/format silly_kicks/ tests/ scripts/`) + bare `pyright` before declaring any task done.

## 6. Sequencing vs in-flight work

- Other session's `position-only-variants-velocity-autoselect` (PR-S163, ADR-067) touches `_ghost_gk.py`, `_xshot_occurrence.py`, `_xcross_attempt.py`, `_xcross_eval.py`, `_model_eval.py`, `_velocity_availability.py`, `features.py`, `feature_glossary.py`, and `train_*`/`compare_position_only` scripts.
- **No line-level collision with Batches 2–6.** Only shared file is `tracking/features.py`, and the regions are disjoint (their `add_ghost_gk`/`ghost_gk_xfns` ~L4985-5238 vs any Batch-4 features.py touch far above). Re-run the overlap scan if PR-S163 pushes more before this branch is cut.
- **Order:** cut the feature branch off `main` *after* PR-S163 merges (cleanest), or off `main` now and rebase — either is safe since there is no collision. Do the `_loader_pining_to_cache.py` site (Batch 6) last, rebased on PR-S163's final loader/CLI shape.
- Batch 7 #13 (ghost-GK default backend) stays deferred until PR-S163 merges (same `_ghost_gk.py` surface).

## 7. Decisions — RESOLVED (owner, 2026-08-24)

All five settled by the owner as recommended:

1. **`group_rows` seam vs inline `groupby`.** ✅ **RESOLVED: the shared seam** (DRY, one guard, one ADR home).
   - **1a. Private vs public seam.** ✅ **RESOLVED: private `_frame_index.py`**, in-repo PRIVATE_CONSUMERS table, YAGNI (`_geometry`/`_polygon`/`lane_control` precedent). Promote only if a cross-package/cross-repo consumer appears.
2. **Turnover item B route.** ✅ **RESOLVED: pure-numpy possession pre-aggregation, benchmarked before finalizing; `numba @njit` is the documented fallback** if the benchmark shows pure-numpy underperforms or the vectorization proves semantically fragile. Ships behind the exact-equivalence oracle either way.
3. **`calibrate_tracking_defaults` default.** ✅ **RESOLVED: safe cap + explicit `--no-cap` opt-out** (Hyrum note in CHANGELOG — the one operator-facing behavior change in the cycle).
4. **ADR-068.** ✅ **RESOLVED: write it** — one light ADR codifying the rescan→`group_rows` pattern + the seam + the item-B route.

**Not decisions (recorded for the implementer):**
5. **The `defensive_credit` cross-period `frame_id` question (F4).** An INVESTIGATION during Task 3, not a now-decision. The refactor keeps `frame_id`-alone (byte-identical). If `frame_id` turns out per-period, that is a latent cross-period bug whose fix is a SEPARATE owner-approved, separately-tested behavior change — surface it to the owner; do NOT fold it into this cycle.
6. **Execution shape.** One feature branch, ONE commit (owner's standing policy). Batches/phases are execution + review ordering ONLY, never commit boundaries. The owner authors the single commit on explicit approval.

## 8. References

- Audit run: 2026-08-24 (memory `project_optimization_audit_backlog`).
- ADR-019 (id dtype-safety / `id_compat`), ADR-020 (`_kernels` scatter crash-safety — do not "fix"), ADR-052 (corpus-driver resilience / `_driver.py`), ADR-065 (order-insensitive converters / the sort helper item A reused).
- Sibling perf specs: `2026-06-01-action-context-hotpath-acceleration-design.md`, `2026-05-23-ball-carrier-numba-vectorization-design.md`.
- Guard infra: `tests/_perf_structural.py` (`call_counter`, `row_iteration_counter`).
