# NaN-safety contract for enrichment helpers + `goalkeeper_ids` coverage feature (PR-S17)

**Status:** Approved (design)
**Target release:** silly-kicks 2.5.0
**Author:** Claude Opus 4.7 (1M)
**Date:** 2026-04-30
**Predecessor:** 2.4.0 (`docs/superpowers/specs/2026-04-30-A9-feature-framework-extraction-design.md`)
**Triggers:**
- luxury-lakehouse `compute_spadl_vaep` task failing on first IDSSE production data: `ValueError: cannot convert float NaN to integer` at `silly_kicks/spadl/utils.py:543` in `add_pre_shot_gk_context`.
- Lakehouse memo: "gk_role + defending_gk_player_id NULL for IDSSE + Metrica" (1.9.0-era observation; partially mitigated by 1.10.0 converter fixes; distribution-detection coverage gap remains).

---

## 1. Problem

silly-kicks's post-conversion enrichment helpers (`add_possessions`, `add_names`, `add_gk_role`, `add_gk_distribution_metrics`, `add_pre_shot_gk_context`, plus atomic counterparts — 10 functions total) operate on caller-supplied SPADL DataFrames. Caller data shapes vary by provider: StatsBomb has dense, well-attributed `player_id` columns; IDSSE/Sportec, Metrica, and provider data with sparse attribution can have NaN in `player_id` for legitimate reasons (system events, off-camera moments, raw-bronze quality limits). ADR-001 (silly-kicks 2.0.0) commits silly-kicks to caller's identifier conventions: converters never override the caller's `team_id` / `player_id`. That sacred-pass-through implies enrichment helpers must be robust to NaN identifiers.

Two concrete failures observed:

1. **`add_pre_shot_gk_context` crashes** at `silly_kicks/spadl/utils.py:543` (and the symmetric atomic version at `silly_kicks/atomic/spadl/utils.py:826`):
   ```python
   gk_id = int(player_id[window_start + relative_indices[-1]])
   ```
   When the most-recent defending-keeper-action's row has NaN `player_id`, `int(NaN)` raises `ValueError: cannot convert float NaN to integer`. Surfaced 2026-04-30 by the lakehouse daily ingestion job's first end-to-end run on real IDSSE bronze data.

2. **`add_gk_distribution_metrics` has a latent crash risk** at `silly_kicks/spadl/utils.py:374-377` (and atomic counterpart at `silly_kicks/atomic/spadl/utils.py:665-668`):
   ```python
   zone_x_start = np.clip((start_x[eligible] / (_PITCH_LENGTH_M / 12.0)).astype(int), 0, 11)
   ```
   When a distribution row has NaN coordinate values, `.astype(int)` raises the same exception. Standard SPADL coordinates are mandatory and typically populated by all converters, so this bug has not surfaced in production — but it's the same algorithm class as failure #1 and contradicts the helper's documented provider-agnostic contract. We should fix while we're here.

A third concern (lakehouse memo): **`add_gk_role` distribution-detection undercoverage** on data with NaN `player_id`. Not a crash — `add_gk_role` uses `==` comparison on `player_id`, which is NaN-safe (`NaN != NaN` returns False, so the row drops out of the `is_distribution` mask). But the *coverage* gap is real: distribution rows by NaN-player_id GKs go untagged. The lakehouse memo flagged this as the practical reason `gk_role` shows NULL for IDSSE/Metrica's distribution rows.

The deeper problem under all three failures: **silly-kicks does not have an explicit, codified, CI-enforced NaN-safety contract for enrichment helpers.** Each helper makes ad-hoc decisions about NaN handling; some are accidentally NaN-safe, some are accidentally NaN-unsafe. Without a contract + a self-enforcing perimeter (auto-discovered fuzz test), the next helper a contributor adds is just as likely to repeat the bug.

This PR introduces:
1. A formal NaN-safety contract for enrichment helpers, captured in **ADR-003**.
2. A **`@nan_safe_enrichment` decorator** as the explicit opt-in marker for the contract.
3. An **auto-discovered CI gate** (parametrized over decorated helpers × NaN-laced fixtures) that catches future helpers regressing the contract.
4. **Cross-provider e2e tests** running every decorated helper against vendored production-shape fixtures (StatsBomb / IDSSE / Metrica) — catches "real production data crash" class.
5. Surgical fixes: NaN guards in `add_pre_shot_gk_context` (× 2) and `add_gk_distribution_metrics` (× 2). Behavior change: NaN-input rows route to the documented per-row default rather than crashing.
6. A **`goalkeeper_ids: set[player_id_type] | None = None`** opt-in parameter on `add_gk_role` (standard + atomic) that closes the lakehouse coverage gap with a documented coarser-heuristic fallback.
7. Audit + decoration of all 10 public enrichment helpers.

## 2. Goals

1. **Codify the NaN-safety contract.** ADR-003 defines what "NaN-safe enrichment helper" means: tolerates NaN in caller-supplied identifier columns (`player_id`, `team_id`) AND in caller-supplied numeric columns where the helper internally casts to integer (e.g. coordinates flowing into zone-binning). NaN inputs route to the documented per-row default (typically NaN-output / False / 0); helpers never crash on NaN input.

2. **Self-enforcing perimeter.** New private module `silly_kicks/_nan_safety.py` provides a single decorator (`@nan_safe_enrichment`) that sets `fn._nan_safe = True`. Auto-discovered by the CI gate via `inspect.getmembers(module, inspect.isfunction)` filtered on the marker. Adding a new public enrichment helper without writing a NaN-safety test still gets caught — the gate parametrizes over every decorated helper and runs the synthetic NaN-laced fixture against it. ADR-003 is the contract; the decorator is the perimeter; the gate is the enforcement.

3. **Audit + decorate every public enrichment helper.** 10 helpers (5 standard + 5 atomic): `add_possessions`, `add_names`, `add_gk_role`, `add_gk_distribution_metrics`, `add_pre_shot_gk_context`. For each, audit `int(...)` / `.astype(int)` / `float(...)` / similar casts on caller-data values; fix any NaN-unsafety found; decorate. Audit findings (from this session's investigation):
   - `add_pre_shot_gk_context`: NaN-unsafe (line 543 in std, 826 in atomic) → fix + decorate
   - `add_gk_distribution_metrics`: latent NaN risk on coords (lines 374-377 in std, 665-668 in atomic) → fix + decorate
   - `add_gk_role`: NaN-safe (uses `==` comparison) → decorate (locks current behavior)
   - `add_possessions`: audit during implementation; decorate or fix
   - `add_names`: audit during implementation; decorate or fix

4. **Cross-provider e2e regression.** New `tests/test_enrichment_provider_e2e.py` runs every decorated helper against three vendored production-shape fixtures: `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` (1 match), `tests/datasets/idsse/sample_match.parquet`, `tests/datasets/metrica/sample_match.parquet`. Asserts no crash + sensible output shape. Catches future "first time helper X meets provider Y data" bug class — exactly the class that surfaced today.

5. **Synthetic NaN fuzz.** New `tests/test_enrichment_nan_safety.py` runs every decorated helper against a synthetic NaN-laced SPADL fixture (every caller-supplied identifier column has at least one NaN at strategic positions — middle, last, first). Asserts no crash + identifier-column outputs preserve NaN where input was NaN. Catches "adversarial input" class that production-shape fixtures don't necessarily cover.

6. **Registry-floor sanity assertion.** Both new test files include `test_registry_nonempty` cases asserting `len(STD_ENRICHMENTS) >= 5` and `len(ATOMIC_ENRICHMENTS) >= 5`. Catches the silent failure mode where a future refactor accidentally breaks the marker discovery (e.g., decorator renamed) and the parametrized test silently runs zero cases. The fence has bulletproof posts.

7. **`goalkeeper_ids` opt-in coverage feature on `add_gk_role`.** New keyword-only parameter `goalkeeper_ids: set[player_id_type] | None = None` on standard and atomic `add_gk_role`. When `None` (default), behavior is byte-for-byte unchanged. When provided, distribution detection extends with two additional matching rules: (a) known-GK match — when current row's `player_id ∈ goalkeeper_ids` AND preceding (k-step) action was keeper-type AND same `team_id`/`game_id`, tag as distribution; (b) NaN-team fallback — when both current and shifted `player_id` are NaN AND same `team_id`/`game_id` AND shifted is keeper-type, tag as distribution. Caller opting in via the parameter is the explicit signal that they accept the coarser heuristic. Closes the lakehouse coverage gap on IDSSE/Metrica.

8. **Documentation.** ADR-003 captures the contract. CLAUDE.md "Key conventions" gains a one-line rule pointing to ADR-003. Every decorated helper's docstring gains an explicit "NaN-input semantics" sentence. CHANGELOG `[2.5.0]` entry covers everything.

9. **Minor release 2.5.0.** Additive public API (new parameter on `add_gk_role`, new private module, new docstring guarantees that codify previously-implicit intent). Backward compat preserved — every existing call site keeps current behavior unchanged.

## 3. Non-goals

1. **No converter-side changes.** ADR-001 stands: caller's identifiers pass through unchanged. The fix lives at the enrichment-helper layer, not the converter layer. If raw IDSSE bronze events have NaN `player_id`, that propagates through SPADL conversion; enrichment helpers handle it gracefully.

2. **No nullable-Int64 dtype migration.** The cleaner long-term answer to "untyped player_id is sloppy" is migrating `player_id` / `team_id` to pandas nullable `Int64` everywhere — converter outputs, helper inputs, downstream consumer expectations. That's a multi-week migration with major Hyrum's Law surface (every consumer including the lakehouse needs to change). Out of scope here. ADR-003 explicitly notes this as the future direction; PR-S17 tackles the immediate-pain fix without the migration.

3. **No new GK identification API beyond `goalkeeper_ids` on `add_gk_role`.** `add_pre_shot_gk_context` does not gain `goalkeeper_ids` — its documented contract is "NaN when no defending GK is identifiable," and the NaN-safety fix preserves that exactly. Adding `goalkeeper_ids` to it would require team-membership-over-time tracking that isn't a clean fit for a one-line per-row lookup.

4. **No `add_pre_shot_gk_context` semantic change.** The fix is *strictly* "don't crash on NaN; produce the documented NaN default instead." No new columns; no signature change.

5. **No CI-gate AST analyzer.** A complementary mechanism would be an AST gate that flags `int(...)` / `.astype(int)` on Series / array values without a NaN guard. Too high false-positive rate to implement reliably; auto-discovered fuzz is the better fence. Future work if patterns of regressions emerge that the fuzz can't catch.

6. **No `add_gk_role` algorithmic redesign.** The `goalkeeper_ids` parameter is additive; the existing algorithm runs unchanged when `None`. We do not refactor the existing code path, do not change the type-id-based keeper detection, do not change the precedence rules.

7. **No subagent dispatching.** Single-session, single-commit per the user commit policy.

## 4. Architecture

### 4.1 NaN-safety contract (defined by ADR-003)

A **NaN-safe enrichment helper** is a public function `add_*(actions: pd.DataFrame, ...) -> pd.DataFrame` that satisfies *all* of:

1. **No crash on NaN identifiers.** For every column in the input that is a caller-supplied identifier (`player_id`, `team_id`, `game_id`, `period_id`, `action_id`), NaN values do not raise. Internal logic that relies on the value (e.g. `int(player_id[i])`) detects NaN and routes to the per-row default.

2. **No crash on NaN numerics.** For every numeric column the helper internally casts to integer (e.g. coordinate columns flowing into zone-binning), NaN values do not raise. Affected rows are excluded from the cast-dependent computation; their output column receives NaN/default.

3. **NaN preservation in identifier outputs.** When the helper outputs an identifier-derived column (e.g. `defending_gk_player_id`), NaN inputs that prevent identification produce NaN outputs at those rows. Existing rows that have non-NaN inputs produce the same output as before this PR (byte-for-byte for `goalkeeper_ids=None` case).

4. **Documented NaN-input semantics.** The helper's docstring contains an explicit sentence describing what happens when an input row has NaN in an identifier column. Canonical wording (adapt per helper):
   > NaN values in caller-supplied identifier columns (e.g. `player_id`) are treated as "not identifiable" for that row's enrichment lookup; downstream rows behave as if no identifier were present. See ADR-003.

5. **`@nan_safe_enrichment` decorator applied.** Sets `fn._nan_safe = True`. Used by the CI gate for auto-discovery.

### 4.2 Marker decorator: `silly_kicks/_nan_safety.py`

New private module at the package root (cross-package; importable from both `silly_kicks/spadl/utils.py` and `silly_kicks/atomic/spadl/utils.py`). Mirrors the `silly_kicks/vaep/feature_framework.py` placement pattern from PR-S16: when something is genuinely shared across two packages, it lives at the level both can reach without one reaching into the other.

```python
"""NaN-safety contract decorator for enrichment helpers (ADR-003).

Decorated functions claim that they tolerate NaN in caller-supplied
identifier columns: NaN identifiers route to the documented per-row
default rather than crashing.

The CI gate at tests/test_enrichment_nan_safety.py and
tests/test_enrichment_provider_e2e.py auto-discover decorated helpers
via the ``_nan_safe`` attribute set by this decorator.
"""

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=Callable)


def nan_safe_enrichment(fn: T) -> T:
    """Marker decorator declaring fn satisfies the NaN-safety contract.

    See ADR-003 for the contract definition.

    Examples
    --------
    Mark an enrichment helper as NaN-safe::

        from silly_kicks._nan_safety import nan_safe_enrichment

        @nan_safe_enrichment
        def my_enrichment(actions):
            return enriched_actions
    """
    fn._nan_safe = True  # type: ignore[attr-defined]
    return fn
```

The decorator is private (the module is underscore-prefixed; the function name itself is non-underscored only for ergonomic call-site readability). External consumers who maintain their own enrichment helpers can either set `fn._nan_safe = True` directly or — if they want the marker to be a stable API — open a request to make it public in a future minor release.

### 4.3 Audit & decoration of public enrichment helpers

Audit checklist applied to each of the 10 helpers (5 standard + 5 atomic):

1. Read the function body for `int(...)`, `.astype(int)`, `float(...)`, `np.int64(...)` casts on values derived from caller-supplied DataFrame columns.
2. Read for `dict[k]` / `series[idx]` accesses where `k` could be NaN.
3. For each potentially-unsafe cast, determine: (a) is the value guaranteed non-NaN by upstream filtering? (b) if not, what's the desired default?
4. If audit clean → decorate.
5. If audit shows latent unsafety → fix (NaN guard before cast, mask filter before `eligible`, etc.) → decorate.

Audit results:

| Helper | Casts on caller data? | NaN-safe today? | Action |
|---|---|---|---|
| `add_possessions` (std + atomic) | Possession-id arithmetic, no caller-int casts | Yes (audit confirms) | Decorate |
| `add_names` (std + atomic) | Dict lookup `id_to_name.get(int(tid), "unknown")` at line 1074 (std) — protected by `eligible` mask of non-NaN type_ids in practice; safer to add explicit `pd.isna(tid)` guard | Latent risk | Add guard + decorate |
| `add_gk_role` (std + atomic) | `==` comparison on player_id (NaN-safe), `>` on start_x (NaN-safe — NaN > x is False) | Yes | Decorate (no code change needed for NaN-safety; goalkeeper_ids feature is separate) |
| `add_gk_distribution_metrics` (std + atomic) | `.astype(int)` on coord-derived zone indices (lines 374-377 std, 665-668 atomic) — `eligible` mask filters by `is_distribution & success`, but NaN coords on those rows still crash | NO (latent) | Fix + decorate |
| `add_pre_shot_gk_context` (std + atomic) | `int(player_id[idx])` (line 543 std, 826 atomic) — direct hit on NaN player_id | NO (primary bug) | Fix + decorate |

### 4.4 Fix detail: `add_pre_shot_gk_context` (standard + atomic)

Single-line guard before the `int(...)` cast. The skip path matches the existing "no defending keeper in window" path (line 538-539 std, 822-823 atomic), so the function's already-documented contract holds:

```diff
         relative_indices = np.where(defending_keeper_in_window)[0]
-        gk_id = int(player_id[window_start + relative_indices[-1]])
+        gk_id_raw = player_id[window_start + relative_indices[-1]]
+        if pd.isna(gk_id_raw):
+            # Defending keeper action is identified (in window), but its
+            # player_id is NaN — caller's data does not provide enough
+            # information to identify the defending GK. Leave defaults
+            # (gk_was_engaged stays False, defending_gk_player_id stays NaN)
+            # per the function's documented contract.
+            continue
+        gk_id = int(gk_id_raw)
```

Applies symmetrically to `silly_kicks/atomic/spadl/utils.py:826`.

### 4.5 Fix detail: `add_gk_distribution_metrics` (standard + atomic)

Add `np.isfinite(...)` filter to the `eligible` mask before zone-binning. The four zone-binning lines stay unchanged; `eligible` becomes more restrictive:

```diff
     if xt_grid is not None:
         success_id = spadlconfig.result_id["success"]
         result_id_arr = actions["result_id"].to_numpy()
-        eligible = is_distribution & (result_id_arr == success_id)
+        # NaN coordinates would crash the .astype(int) zone-binning below.
+        # Filter to rows where all four coords are finite (guards against
+        # caller data with sparse spatial information).
+        coords_finite = (
+            np.isfinite(start_x) & np.isfinite(start_y)
+            & np.isfinite(end_x) & np.isfinite(end_y)
+        )
+        eligible = is_distribution & (result_id_arr == success_id) & coords_finite
         if eligible.any():
             zone_x_start = np.clip((start_x[eligible] / (_PITCH_LENGTH_M / 12.0)).astype(int), 0, 11)
             ...
```

`xt_delta` stays NaN at non-`eligible` rows (the `np.full(n, np.nan)` initialization handles this). Length classification (lines 348-359) is already NaN-safe via `np.where` and arithmetic NaN-propagation. Atomic counterpart at lines 665-668: same fix shape with atomic SPADL's `x` / `y` / `end_x` / `end_y` columns (atomic uses `x`/`y` not `start_x`/`start_y`, but the algorithm is identical).

### 4.6 Feature: `goalkeeper_ids` parameter on `add_gk_role` (standard + atomic)

New keyword-only parameter; default `None` preserves byte-for-byte behavior.

```python
def add_gk_role(
    actions: pd.DataFrame,
    *,
    penalty_area_x_threshold: float = 16.5,
    distribution_lookback_actions: int = 1,
    goalkeeper_ids: set | None = None,
) -> pd.DataFrame:
    """...
    
    Parameters
    ----------
    ...
    goalkeeper_ids : set, optional
        When provided, distribution-detection extends beyond strict
        ``same_player`` matching to also tag rows where:
        
        - The current row's ``player_id`` is in ``goalkeeper_ids`` AND the
          preceding action (within ``distribution_lookback_actions`` steps,
          same ``team_id`` and ``game_id``) was a keeper-type action.
          (Use case: caller knows the GK player_ids; clean-attribution data.)
        - Both the current row's and the preceding action's ``player_id``
          are NaN AND the ``team_id`` matches AND the preceding action was
          keeper-type. (Use case: caller's data has NaN player_id but the
          team/sequence implies the GK distributed the ball.)
        
        Opting in via this parameter signals that the caller accepts the
        coarser heuristic (the second rule may over-tag if multiple
        NaN-player_id non-keeper actions follow a keeper action by the
        same team within the lookback window).
        
        When ``None`` (default), only strict ``same_player`` matching applies
        — byte-for-byte compatible with pre-2.5.0 behavior.
    """
```

Implementation in the per-`k` loop:

```python
for k in range(1, distribution_lookback_actions + 1):
    shifted_keeper = is_keeper_series.shift(k, fill_value=False).to_numpy(dtype=bool)
    shifted_player = player_id.shift(k).to_numpy()
    shifted_game = game_id.shift(k).to_numpy()
    cur_player_arr = player_id.to_numpy()
    cur_game_arr = game_id.to_numpy()
    same_player = cur_player_arr == shifted_player
    same_game = cur_game_arr == shifted_game

    match = same_player

    if goalkeeper_ids is not None:
        shifted_team = team_id.shift(k).to_numpy()
        cur_team_arr = team_id.to_numpy()
        same_team = cur_team_arr == shifted_team

        # Rule (a) — known-GK match: caller declared a GK player_id set.
        cur_is_known_gk = pd.Series(cur_player_arr).isin(goalkeeper_ids).to_numpy()
        match = match | (cur_is_known_gk & same_team)

        # Rule (b) — NaN-team fallback: both player_ids unidentifiable but
        # same team + prev was keeper. Coarse heuristic; caller opts in.
        cur_player_na = pd.isna(cur_player_arr)
        shifted_player_na = pd.isna(shifted_player)
        match = match | (cur_player_na & shifted_player_na & same_team)

    prev_keeper_within_k |= shifted_keeper & match & same_game
```

Atomic version: identical pattern (atomic `add_gk_role` at line 54 has the same shape).

### 4.7 Test infrastructure

#### 4.7.1 Synthetic NaN fuzz: `tests/test_enrichment_nan_safety.py`

```python
"""NaN-safety contract enforcement (ADR-003).

Auto-discovers every helper decorated with @nan_safe_enrichment and runs
it against a synthetic NaN-laced fixture. Fails fast if a helper crashes
on NaN-input rows in caller-supplied identifier columns.

Catches: future contributor adds a public enrichment helper without writing
a NaN-safety test. Auto-discovery here covers them automatically.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import silly_kicks.atomic.spadl.utils as atomic_utils
import silly_kicks.spadl.utils as std_utils


def _discover(module) -> tuple:
    return tuple(
        fn for _, fn in inspect.getmembers(module, inspect.isfunction)
        if getattr(fn, "_nan_safe", False)
    )


STD_ENRICHMENTS = _discover(std_utils)
ATOMIC_ENRICHMENTS = _discover(atomic_utils)


def test_registry_nonempty_std():
    """Catches: future refactor that breaks marker discovery → silent zero-cases."""
    assert len(STD_ENRICHMENTS) >= 5, (
        f"Expected ≥5 @nan_safe_enrichment helpers in silly_kicks.spadl.utils; "
        f"found {len(STD_ENRICHMENTS)}. Did the marker name change?"
    )


def test_registry_nonempty_atomic():
    assert len(ATOMIC_ENRICHMENTS) >= 5, (
        f"Expected ≥5 @nan_safe_enrichment helpers in silly_kicks.atomic.spadl.utils; "
        f"found {len(ATOMIC_ENRICHMENTS)}. Did the marker name change?"
    )


@pytest.fixture
def std_nan_laced_actions() -> pd.DataFrame:
    """Synthetic standard-SPADL fixture with NaN at strategic positions:
    
    - First row: NaN player_id (boundary case — earliest-position NaN)
    - Middle row: NaN player_id (mid-stream NaN)
    - Last row: NaN player_id (boundary case — latest-position NaN)
    - Two consecutive shot/keeper rows where the keeper action has NaN
      player_id (the exact pattern that crashes add_pre_shot_gk_context).
    """
    # ~15-row synthetic SPADL frame with valid columns + strategic NaNs.
    # Full content in implementation.
    ...


@pytest.fixture
def atomic_nan_laced_actions() -> pd.DataFrame:
    """Symmetric for atomic SPADL schema (x/y/dx/dy instead of start_x/end_x)."""
    ...


@pytest.mark.parametrize("helper", STD_ENRICHMENTS, ids=lambda h: h.__name__)
def test_standard_helper_nan_safe(helper, std_nan_laced_actions):
    """Every @nan_safe_enrichment standard helper survives NaN-laced input
    with default kwargs.
    """
    out = helper(std_nan_laced_actions)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(std_nan_laced_actions), (
        f"{helper.__name__} changed row count on NaN-laced input"
    )


@pytest.mark.parametrize("helper", ATOMIC_ENRICHMENTS, ids=lambda h: h.__name__)
def test_atomic_helper_nan_safe(helper, atomic_nan_laced_actions):
    out = helper(atomic_nan_laced_actions)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(atomic_nan_laced_actions)
```

Per-helper additional assertions where the helper has a known identifier-output column (e.g. `add_pre_shot_gk_context`'s `defending_gk_player_id`):

```python
def test_pre_shot_gk_context_preserves_nan_on_unidentifiable_shot(std_nan_laced_actions):
    """When the defending keeper's player_id is NaN, defending_gk_player_id
    is NaN for that shot — not raises, not 0, not a sentinel int.
    """
    out = std_utils.add_pre_shot_gk_context(std_nan_laced_actions)
    # Specific shot_idx where the fixture has a NaN-keeper preceding a shot.
    # Assert defending_gk_player_id is NaN at that index.
    ...
```

#### 4.7.2 Cross-provider e2e: `tests/test_enrichment_provider_e2e.py`

```python
"""Cross-provider e2e regression (ADR-003 perimeter).

Runs every @nan_safe_enrichment helper against vendored production-shape
fixtures from each supported provider. Catches: helper crashes on real
production data shape from a provider whose data shape differs from
the one used during helper development.

Production-shape fixtures used:
- StatsBomb: tests/datasets/statsbomb/spadl-WorldCup-2018.h5 (1 match)
- IDSSE (DFL Sportec): tests/datasets/idsse/sample_match.parquet
- Metrica: tests/datasets/metrica/sample_match.parquet
"""

import pytest

# Auto-discovery as in 4.7.1.
STD_ENRICHMENTS = _discover(std_utils)


@pytest.fixture(params=["statsbomb", "idsse", "metrica"])
def std_provider_actions(request) -> pd.DataFrame:
    """SPADL DataFrame from one production-shape provider fixture."""
    if request.param == "statsbomb":
        return _load_statsbomb_one_match()
    elif request.param == "idsse":
        return _load_idsse_via_sportec_converter()
    else:
        return _load_metrica_via_metrica_converter()


@pytest.mark.parametrize("helper", STD_ENRICHMENTS, ids=lambda h: h.__name__)
def test_standard_helper_provider_e2e(helper, std_provider_actions):
    """Every @nan_safe_enrichment standard helper produces a DataFrame
    on production-shape input from each supported provider.
    """
    out = helper(std_provider_actions)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(std_provider_actions)
```

Atomic e2e: 5 helpers × 1 fixture (StatsBomb-derived atomic; IDSSE/Metrica atomic conversion is out of scope for this PR — converters need that to be confirmed working first).

#### 4.7.3 `goalkeeper_ids` feature tests

Smoke tests in `tests/test_gk_role_goalkeeper_ids.py` (or extend existing `tests/spadl/test_gk_role.py` if present):

- Backward-compat: `add_gk_role(actions, goalkeeper_ids=None)` produces same output as `add_gk_role(actions)` without the kwarg.
- Rule (a): clean StatsBomb-shape fixture with known GKs in `goalkeeper_ids` — distribution detection covers more rows than the no-`goalkeeper_ids` baseline.
- Rule (b): NaN-player_id fixture with `goalkeeper_ids` provided — distribution rows by NaN-player tagged when team-context implies GK.
- Atomic counterpart: same test shape.

### 4.8 Documentation deltas

#### 4.8.1 ADR-003

`docs/superpowers/adrs/ADR-003-nan-safety-enrichment-helpers.md`. Captures:
- Context (lakehouse production failure 2026-04-30; ADR-001 caller-pass-through implication)
- Decision (NaN-safety contract for public enrichment helpers; `@nan_safe_enrichment` decorator + auto-discovered CI gate)
- Alternatives considered (nullable-Int64 migration; AST-based gate; documentation-only; per-helper hand-maintained tests)
- Consequences (positive: contract is explicit + auto-enforced; negative: opt-in marker can be forgotten on new helpers — mitigated by auto-discovery covering decorated set; neutral: external consumers can extend the marker)
- CLAUDE.md amendment (one-line rule pointing here)
- Notes (`goalkeeper_ids` rationale; future direction toward nullable-Int64)

#### 4.8.2 CLAUDE.md amendment

Under "Key conventions" section, new bullet:

> Public enrichment helpers (post-conversion `add_*` family) tolerate NaN in caller-supplied identifier columns. NaN identifiers route to the documented per-row default (typically NaN-output / False / 0); helpers never crash on NaN input. Decoration with `@nan_safe_enrichment` from `silly_kicks._nan_safety` is the formal opt-in. Decision: ADR-003.

#### 4.8.3 Per-helper docstring deltas

Each decorated helper gains one sentence in its docstring (location: after the main description, before parameter docs):

> NaN values in caller-supplied identifier columns (e.g. `player_id`) are treated as "not identifiable" for that row's enrichment lookup; downstream rows behave as if no identifier were present. See ADR-003.

`add_gk_role` additionally documents the `goalkeeper_ids` parameter (per § 4.6).
`add_gk_distribution_metrics` notes the coordinate-NaN exclusion (per § 4.5).

#### 4.8.4 CHANGELOG `[2.5.0]` entry

Sections: Added (decorator + parameter + tests + ADR), Changed (docstring NaN-input semantics), Fixed (2 crash sites + 2 latent risk sites), Notes (Hyrum's Law + future direction).

## 5. Test plan

### 5.1 Test count delta

| Test file | Before | After | Δ |
|---|---|---|---|
| `tests/test_enrichment_nan_safety.py` (NEW) | 0 | 12 + 2 sanity = ~14 | +14 |
| `tests/test_enrichment_provider_e2e.py` (NEW) | 0 | 15 std × 3 providers + 5 atomic × 1 fixture = ~20 | +20 |
| `tests/test_gk_role_goalkeeper_ids.py` (NEW or extension) | 0 | ~10 (5 std + 5 atomic) | +10 |
| Existing tests | unchanged | unchanged (backward-compat preserved) | 0 |
| **Net test delta** | | | **+~44** |

Baseline 884 + ~44 = **~928 passing, 4 deselected** (target — exact count finalized in implementation).

### 5.2 TDD ordering

Following the structural-refactor pattern (memory: PR-S15 / PR-S16): write failing tests first, then implement to green.

1. **T1 — Create `_nan_safety.py` decorator skeleton.** Empty registry, decorator returns fn unchanged + sets attribute.
2. **T2 — Write `tests/test_enrichment_nan_safety.py` skeleton.** With auto-discovery + registry-floor sanity. Discovery returns empty tuple → registry-floor tests fail → RED.
3. **T3 — Write `tests/test_enrichment_provider_e2e.py` skeleton.** Same auto-discovery; registry-floor tests fail → RED.
4. **T4 — Decorate `add_gk_role` (std + atomic) — known NaN-safe.** Registry-floor tests now pass for those entries. Other helpers still undecorated.
5. **T5 — Decorate `add_possessions` (std + atomic) — audit during decoration; should be NaN-safe (audit confirms).** No code change expected; if audit reveals issue, fix in this step.
6. **T6 — Decorate `add_names` (std + atomic) — audit + likely add `pd.isna(tid)` guard at line 1074.**
7. **T7 — Fix + decorate `add_gk_distribution_metrics` (std + atomic).** Apply § 4.5 patch. Synthetic NaN-fuzz tests for this helper now pass.
8. **T8 — Fix + decorate `add_pre_shot_gk_context` (std + atomic).** Apply § 4.4 patch. Synthetic NaN-fuzz tests for this helper now pass.
9. **T9 — Add `goalkeeper_ids` parameter to `add_gk_role` (std + atomic).** Per § 4.6.
10. **T10 — Write feature tests in `tests/test_gk_role_goalkeeper_ids.py`.** Backward-compat + rule (a) + rule (b). All should pass after T9.
11. **T11 — Build synthetic NaN-laced fixtures.** Concrete DataFrames in test_enrichment_nan_safety.py fixtures. Fuzz tests now run end-to-end.
12. **T12 — Build provider-fixture loaders for e2e.** StatsBomb HDF5 first match; IDSSE parquet → sportec converter; Metrica parquet → metrica converter. e2e tests now run.
13. **T13 — Per-helper specific assertions.** Add the "NaN-input preserves NaN-output" tests (e.g. `test_pre_shot_gk_context_preserves_nan_on_unidentifiable_shot`).
14. **T14 — Verification gates.** Full pytest, ruff, ruff format, pyright. Target ~928 passing.
15. **T15 — Documentation.** ADR-003, CLAUDE.md amendment, per-helper docstring sentences.
16. **T16 — CHANGELOG `[2.5.0]`, version bump.**
17. **T17 — Memory updates.**
18. **T18 — `/final-review`.**
19. **T19 — Single-commit gate.**
20. **T20 — Push, PR, CI watch, squash-merge, tag, PyPI verify.**

## 6. Verification gates (before commit)

```bash
uv run ruff check silly_kicks/ tests/
uv run ruff format --check silly_kicks/ tests/
uv run pyright silly_kicks/

# Smoke — decorator + auto-discovery + cross-package import
uv run python -c "
from silly_kicks._nan_safety import nan_safe_enrichment
import silly_kicks.spadl.utils as su
import silly_kicks.atomic.spadl.utils as au
import inspect
discovered_std = [n for n, fn in inspect.getmembers(su, inspect.isfunction)
                  if getattr(fn, '_nan_safe', False)]
discovered_atomic = [n for n, fn in inspect.getmembers(au, inspect.isfunction)
                     if getattr(fn, '_nan_safe', False)]
assert len(discovered_std) >= 5, discovered_std
assert len(discovered_atomic) >= 5, discovered_atomic
print('discovery OK:', discovered_std + discovered_atomic)
"

# Smoke — failing case from production
uv run python -c "
import pandas as pd
import numpy as np
from silly_kicks.spadl.utils import add_pre_shot_gk_context

# NaN-keeper-player_id pattern (the IDSSE failure shape).
actions = pd.DataFrame({
    'game_id': [1, 1, 1, 1],
    'period_id': [1, 1, 1, 1],
    'action_id': [0, 1, 2, 3],
    'team_id': [10, 20, 20, 10],
    'player_id': [100, np.nan, 200, 100],  # GK action has NaN
    'type_id': [0, 14, 0, 13],  # 14 = keeper_save (defending team), 13 = shot
    'time_seconds': [0.0, 5.0, 8.0, 9.0],
})
out = add_pre_shot_gk_context(actions)
assert pd.isna(out.iloc[3]['defending_gk_player_id']), 'NaN-keeper preserves NaN output'
print('NaN-safe smoke OK')
"

uv run pytest tests/ -m "not e2e" --tb=short -q
```

Expected: all gates green; pytest = **~928 passed, 4 deselected** (884 baseline + ~44 net new). Pyright clean. Ruff clean.

## 7. Risks + Hyrum's Law audit

| Risk | Mitigation |
|---|---|
| `_nan_safe` attribute name collides with a future pandas / numpy attribute | The marker is set on user-defined functions, not on pandas / numpy objects. Collision risk is zero. |
| Adding `goalkeeper_ids=None` parameter changes `add_gk_role.__signature__` — anyone introspecting via `inspect.signature(add_gk_role)` would see the new keyword-only param | Acceptable. Added to the additive Hyrum's Law surface; documented in CHANGELOG. The function's behavior with default `None` is byte-for-byte unchanged, so the parameter addition only affects consumers who pass it explicitly. |
| Marker discovery silently breaks (e.g., `_nan_safe` attribute name typo on a helper) → CI gate runs zero parametrize cases on that helper, contract-violating helper sneaks in | Registry-floor sanity tests (`test_registry_nonempty_std`, `test_registry_nonempty_atomic`) assert `len(REGISTRY) >= 5`. A drop below the floor fails CI explicitly. |
| Auto-discovery picks up functions that should NOT be in the contract (e.g. helpers that intentionally raise on NaN) | Only `@nan_safe_enrichment`-decorated functions are included. Opt-in by design. Helpers that should raise are left undecorated; their CHANGELOG / docstring documents the strict-NaN behavior. `use_tackle_winner_as_actor` (1.10.0 migration helper) is the canonical example — intentionally strict, not decorated. |
| Synthetic NaN-laced fixture shape doesn't match how production data fails (the bug surfaces in a way the fuzz doesn't cover) | The complementary cross-provider e2e (§ 4.7.2) loads real production-shape fixtures from three providers. Combined coverage: synthetic adversarial + production realistic. |
| `goalkeeper_ids` rule (b) NaN-team fallback over-counts distribution rows on data with multiple NaN-player_id non-keeper actions per same team after a keeper action | Documented in the parameter docstring (§ 4.6). Caller's opt-in via `goalkeeper_ids=...` is the explicit signal that they accept the coarser heuristic. Default `goalkeeper_ids=None` preserves strict matching → byte-for-byte backward compat. |
| `pd.isna` on an array with mixed numeric / object dtype could behave inconsistently across pandas versions | All affected columns (`player_id`, coords) are numeric per SPADL schema. `pd.isna` on numeric arrays is stable across pandas 2.x. Pyright + pandas-stubs verify the type contract; pinned versions in CI catch any drift. |
| `inspect.getmembers(module, inspect.isfunction)` includes imported functions, potentially false-positive on a re-imported helper | Only locally-defined functions have their `_nan_safe` attribute set in the module that defines them. A re-imported function carries its `_nan_safe` from origin — but auto-discovery in `tests/test_enrichment_nan_safety.py` walks both `std_utils` and `atomic_utils`. If `add_gk_role` from std is re-imported into atomic for some reason, it would be discovered twice. This is a non-issue (duplicate parametrize) — and the actual atomic helpers are independent definitions (atomic SPADL schema differs). Verified during implementation T6. |
| ADR-003 promises NaN-safety as a contract; future helper that fails the contract slips in if author forgets to decorate AND test author writes their own NaN-safety test instead of leveraging the auto-discovery | The auto-discovery covers the CHECKING side; the decoration covers the OPTING-IN side. A helper that's NaN-unsafe AND undecorated AND has its own dedicated NaN test would satisfy CI but fail ADR-003 in spirit. Mitigation: ADR-003 explicitly says "decoration is the formal opt-in"; CLAUDE.md amendment makes this a project rule reviewers check during PR review. Defense-in-depth not perfect but sufficient. |
| Tests rely on existing vendored fixtures (StatsBomb HDF5, IDSSE parquet, Metrica parquet) — fixture changes downstream could silently break the e2e | Fixtures are committed to the repo; changes go through PR review. The e2e tests reference them by relative path; any move triggers test failure. |

## 8. Acceptance criteria

1. `silly_kicks/_nan_safety.py` exists with the `nan_safe_enrichment` decorator + Examples docstring.
2. 10 public enrichment helpers (5 standard + 5 atomic) are decorated with `@nan_safe_enrichment`. Audit complete; any latent NaN-unsafety found has been fixed.
3. `silly_kicks/spadl/utils.py:543` — `add_pre_shot_gk_context` no longer crashes on NaN keeper-action `player_id`. Symmetric fix at `silly_kicks/atomic/spadl/utils.py:826`.
4. `silly_kicks/spadl/utils.py:374-377` — `add_gk_distribution_metrics` no longer crashes on NaN coords at distribution rows. Symmetric fix at `silly_kicks/atomic/spadl/utils.py:665-668`.
5. `add_gk_role` (std + atomic) accepts new keyword-only `goalkeeper_ids: set | None = None` parameter; default `None` preserves byte-for-byte behavior; non-`None` extends distribution detection per § 4.6.
6. `tests/test_enrichment_nan_safety.py` exists with auto-discovery, registry-floor sanity, parametrized fuzz cases. ~14 cases pass.
7. `tests/test_enrichment_provider_e2e.py` exists with cross-provider e2e. ~20 cases pass.
8. `tests/test_gk_role_goalkeeper_ids.py` exists (or equivalent extension to existing test file). ~10 cases pass covering backward-compat + rules (a) + (b).
9. Total pytest: ~928 passed, 4 deselected (~+44 net).
10. `docs/superpowers/adrs/ADR-003-nan-safety-enrichment-helpers.md` exists.
11. `CLAUDE.md` "Key conventions" section gains the NaN-safety rule.
12. Each decorated helper's docstring contains the canonical NaN-input-semantics sentence.
13. `CHANGELOG.md` `[2.5.0]` entry covers everything.
14. `pyproject.toml` version is `2.5.0`.
15. ruff / ruff format / pyright clean.
16. `/final-review` clean.
17. Memory files refreshed (release_state, MEMORY.md, follow-up_prs).

## 9. Commit cycle

Same pattern as PR-S16. Single commit per project commit policy. Branch: `feat/nan-safety-enrichment-helpers`. Version bump 2.4.0 → 2.5.0 (additive public API: new private module, new decorator, new parameter on `add_gk_role`).
