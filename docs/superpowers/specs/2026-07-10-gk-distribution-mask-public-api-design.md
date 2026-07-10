# Public `gk_distribution_mask` API + ρ loader cleanup — Design

**Status:** DRAFT (for review). NOT committed by this session.
**Date:** 2026-07-10 · **For:** silly-kicks session (implementation).
**Origin:** lakehouse export request (`SILLY_KICKS_EXPORT_REQUEST.md`, in the luxury-lakehouse repo),
following the xT-GK v2 completion (PR-S109, 4.42.0). Governing docs: ADR-036 (xT-GK v2), ADR-007
(GK identification), ADR-019 (dtype-safe id comparisons), M5 (v1 freeze).

## 0. Context — why this exists

The lakehouse wants to materialize a per-action **GK-distribution** flag (`is_gk_distribution`) on
`fct_action_context` so downstream consumers (incl. silly-kicks' ρ retention model) share one
canonical domain marker across all providers. Their handoff surfaced two facts:

1. **The existing `gk_was_distributing` column is *their* shot-scoped `add_pre_shot_gk_context` feature**
   (44 True, all on shots) — NOT the acting-GK-distribution marker. So a **new** column is needed, and
   silly-kicks' ρ loader (which currently references `gk_was_distributing`) must stop reading it.
2. **The canonical domain logic already lives in silly-kicks** as the *private* `_gk_distribution_mask`
   (`tracking/_xt_gk.py:303`). The lakehouse asks silly-kicks to **export it as a public, stable,
   frame-optional API** — using the robust `acting_gk_from_frames` resolver, which is *time-accurate*
   (a strict subset of `native`; it tightens stale/substituted keepers rather than broadening — see §3)
   and is the **same resolver the lakehouse pins for its goal-kick-taker override**, so the mask stays
   consistent with that override — so they can pin one function rather than reimplement it. (The `~40%
   undetected keeper` figure is a goal-kick-frame phenomenon that motivates the *taker override*, not the
   mask's GK-pass term. Their fallback is a thin wrapper over the already-public `acting_gk_from_frames`;
   the export is a cleanliness preference, not a hard block.)

This spec delivers the public API (+ the freeze-preserving shim) and cleans up the ρ loader's
wrong-column reference. It does **not** materialize `is_gk_distribution` (lakehouse-side) or retrain ρ
(data-dependent — deferred).

## 1. Locked design decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Scope** | Export the public API **+** correct the ρ loader's column reference. ρ **retrain deferred**. | The export unblocks the lakehouse; the loader fix removes a known-wrong (dormant) reference. The retrain needs the lakehouse to populate the column first — it can't happen here. |
| **Frozen `_xt_gk.py`** | **Byte-identical shim.** The new public function is canonical; v1's private `_gk_distribution_mask` becomes a golden-gated shim delegating to it (`native` mode). | Single source of truth; matches the `resolve_gk_geometry`→`resolve_restart_geometry` precedent. M5 freeze is about v1 *behaviour* (preserved + golden-gated), not never touching the file. |
| **Robustness lever** | `resolve_gk="robust"` (default) uses `acting_gk_from_frames`; `"native"` reproduces the frozen `frames["is_goalkeeper"]` lookup (shim only). | New consumers get the roster-identity fallback; v1 stays byte-identical. |
| **Loader transition** | **Self-adapting** — probe the `fct_action_context` schema, select `is_gk_distribution` only if present. | Zero future loader change when the column lands; graceful goal-kicks-only until then. Transitional (see §7). |

## 2. Architecture

Three units + one documented external contract:

```
tracking/_gk_resolve.py  (non-frozen; already home of acting_gk_from_frames)
  + gk_distribution_mask(actions, frames=None, *, resolve_gk="robust")   [NEW public]
        resolve_gk="native"  -> frozen is_goalkeeper set-membership (byte-identical to v1)
        resolve_gk="robust"  -> acting_gk_from_frames identity match
        frames=None          -> goal-kicks only (both modes)

tracking/_xt_gk.py  (FROZEN)
  _gk_distribution_mask(actions, frames)  ->  gk_distribution_mask(actions, frames, resolve_gk="native")  [SHIM]

tracking/__init__.py
  export gk_distribution_mask

scripts/_loader_databricks.py + scripts/train_gk_retention.py   (ρ consumer)
  drop gk_was_distributing; self-adapting is_gk_distribution read; domain = goal-kicks ∪ is_gk_distribution

[lakehouse, documented only] fct_action_context.is_gk_distribution = gk_distribution_mask(..., resolve_gk="robust")
```

## 3. The public function

```python
def gk_distribution_mask(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None = None,
    *,
    resolve_gk: Literal["native", "robust"] = "robust",
) -> pd.Series:  # bool, index-aligned to actions.index
```

- **Returns** a **`pd.Series` of bool aligned to `actions.index`** — consistent with its module sibling
  `acting_gk_from_frames` (which returns an index-aligned Series). The v1 shim (§4) calls `.to_numpy()`
  to recover the exact `NDArray[np.bool_]` v1 expects, so byte-identity is preserved while the public
  surface is index-safe.
- **Alignment (avoid the positional↔index hazard):** do **all** boolean logic on **positional numpy
  arrays in `actions` row-order** — `type_id`/`is_open` from `.to_numpy()`; the robust actor test as
  `ids_equal(actions["player_id"], acting_gk).to_numpy()` — then wrap the final mask with `actions.index`
  **exactly once** at the end. This is not stylistic: **`ids_equal` is POSITIONAL and returns a
  RangeIndex `np.bool_` Series** (verified), so `&`-ing it directly against an actions-index Series would
  **silently misalign** when the caller passes a non-default index (our per-match action-context path
  after filtering). Positional-throughout + one final wrap sidesteps it. Validated by the non-`RangeIndex`
  test (§8).
- **Input contract (fail-loud on absence):** `actions` must carry `type_id`, `player_id`, `team_id`,
  `period_id`, `time_seconds` (and `game_id` when present, used for the multi-match key); `frames` (when
  given) must be `TRACKING_FRAMES_COLUMNS`-shaped. State these in the docstring.
- **Semantics:** `True` for any goal-kick (`type_id == goalkick`), OR a **pass OR throw-in**
  (`is_open = type_id ∈ {pass, throw_in}`) whose actor is the acting team's goalkeeper. The goal-kick
  term is **actor-independent** (a goal-kick is in-scope regardless of who took it).
- **`frames is None` → goal-kicks-only.** GK open-play-pass detection is impossible without frames, so
  the open-play term is skipped in both modes. This is the marts-native / no-tracking path. When
  `frames is None` and `resolve_gk="robust"`, the robust request silently degrades to goal-kicks-only —
  documented in the docstring (a one-line note; not an error, since event-only providers rely on it).
- **`resolve_gk="native"`** — reproduces the frozen `_gk_distribution_mask` exactly: build a
  `(game_id, team_id, player_id)` GK-identity set from `frames[is_goalkeeper & ~is_ball]`; a pass/throw-in
  is in-scope iff its `(game, team, player)` is in the set. Dtype-safe via `canonical_id_series` (ADR-019).
- **`resolve_gk="robust"`** — resolve the acting GK per action via `acting_gk_from_frames` (linked-frame
  lookup + roster-identity fallback for the undetected-at-that-frame case); a pass/throw-in is in-scope iff
  `ids_equal(actions.player_id, acting_gk_resolved)` (ADR-019, NA-safe). This is a **time-accurate**
  resolution: on a single detected keeper it **agrees** with `native`; on a **GK substitution** (or a
  momentary mis-flag) `native`'s global set over-includes the *other/stale* keeper, and `robust` correctly
  **excludes** the pass by the keeper who isn't the acting one at that moment.
  - **Direction of the difference (important):** `robust ⊆ native` for the GK-pass term — both hinge on the
    keeper being detected somewhere, so `robust` never *adds* a row `native` lacks; it only **tightens**
    (drops stale/mis-attributed keeper passes). It is also the **same resolver the lakehouse pins for its
    goal-kick-taker override**, so the domain marker stays consistent with that override — the real reason
    it is the default. (The `~40%-undetected-keeper` motivation is about that *override* — goal-kicks are
    actor-independent in the mask — not the GK-pass term, whose keeper is on-screen in open play.)

Pure (never mutates `actions`); NaN-safe (NaN player/team → not-in-scope, per ADR-003 spirit).
**No import cycle:** `_gk_resolve.py` imports only `_id_compat` + `utils` (verified — never `_xt_gk`), so
the frozen `_xt_gk.py` importing `gk_distribution_mask` from it is safe.

## 4. The v1 shim (freeze-preserving)

`_xt_gk.py::_gk_distribution_mask(actions, frames)` becomes a one-liner:
`return gk_distribution_mask(actions, frames, resolve_gk="native")`. Its three consumers — v1
`compute_xt_gk` (`_xt_gk.py:470`), the completion model (`_gk_completion.py:340`), and
`features.py:5376` — are unaffected because `native` is byte-identical to the current body.

**Golden gate:** a snapshot test pins `_gk_distribution_mask` output on the committed WC2018 (or existing
GK fixture) to be byte-identical pre/post refactor; the existing `xthreat`/v1 byte-stability regression
gates stay green. If the golden diverges, the refactor is wrong — fix before shipping.

## 5. ρ loader cleanup (scope B, self-adapting)

- **`load_retention_cohort`** (`_loader_databricks.py`): drop `gk_was_distributing` from the SQL +
  coercion. **Safe because it is a domain-mask *input*, never a model *feature*** — the bundled `default`
  model's `feature_names` are the 8 `RETENTION_FEATURE_NAMES` (verified: `gk_was_distributing ∉` them), so
  removing the column cannot affect inference. Guarded by §8's feature-list assertion (the one place a
  "remove a column" cleanup could silently break serving). Probe `fct_action_context`'s columns and add
  `c.is_gk_distribution` to the SELECT **only if it exists**; otherwise the frame simply lacks the column.
  - **Probe mechanism — catalog-qualify (footgun).** Use a **catalog-qualified** existence check —
    `soccer_analytics.information_schema.columns` (or a catalog-scoped `DESCRIBE
    soccer_analytics.dev_gold.fct_action_context`), never a bare `information_schema` (which resolves to
    the session's default catalog and returned false-negatives twice this session → the loader would
    *silently never* pick up the column even after it lands). Factor the decision as a **pure**
    `should_select_is_gk_distribution(existing_columns: set[str]) -> bool` so it's unit-testable without
    a live connection; the IO (running the catalog-qualified probe) is the thin owner-run seam.
- **`prepare_retention_training_data`** (`train_gk_retention.py`): domain becomes
  `goal-kicks ∪ COALESCE(is_gk_distribution, FALSE)` — the second term applies only if the column is
  present, and its NULLs are **explicitly coalesced to False** (`.fillna(False)` in pandas; the existing
  `gk_was_distributing` OR already does this — preserve it for `is_gk_distribution`).
  - **Rollout NULL semantics (correctness).** During rollout the column *exists but is NULL* for a whole
    population — non-tracking rows (it's tracking-derived) and tracking rows not yet recomputed. A bare
    `False OR NULL` is `NULL` (SQL/nullable-pandas), so those rows would silently fall out of / corrupt
    the domain. The `COALESCE(..., FALSE)` / `.fillna(False)` maps *present-but-NULL* to False (out of
    scope, same as absent). This is tested (§8: the "column present but NULL" case), not just present/absent.
- **Behaviour today:** the lakehouse has not materialized `is_gk_distribution`, so the domain stays
  goal-kicks-only; the bundled `default` ρ model is unaffected. When the column lands (and is non-NULL
  for a row), the domain broadens automatically (retrain still required — §7).

## 6. Lakehouse contract (documented, not built here)

`fct_action_context.is_gk_distribution` (boolean, per action, keyed `(match_key, action_id)`) **==**
`gk_distribution_mask(actions, frames, resolve_gk="robust")`. Coverage per the lakehouse's own note:
full domain on the 4 tracking providers (GS/SC priority), goal-kicks-only on statsbomb/wyscout (no
frames). The lakehouse pins the released public API and materializes the column; silly-kicks reads it.

**Positive alignment (lakehouse-noted):** the frame-optional design *subsumes* the lakehouse's own
goal-kick fallback — for event-only providers they call `gk_distribution_mask(actions, frames=None)`
instead of a bespoke goal-kick OR, so their F1 materialization plan drops that separate branch. One
function, all providers.

## 7. Deferred (paired follow-up, once `is_gk_distribution` is live)

- **Retrain ρ** on the broadened domain (goal-kicks + acting-GK passes, ~3–4× the rows) — likely a
  materially better `default` and possibly a viable SkillCorner variant. Data-dependent.
- **Simplify the loader:** the §5 self-adapting probe is **transitional scaffolding**. Once the column
  is permanently materialized, collapse to an unconditional `SELECT c.is_gk_distribution` and a plain
  `goal-kicks ∪ is_gk_distribution` domain; drop the schema probe + present-check. The self-adapting
  block carries a comment pointing at this cleanup so it isn't forgotten (National Park). The retrain
  follow-up (which already requires the column) is its natural home.
- **Eyestone review** (separate, from PR-S109): the absolute-effect floor + the cross-check divergence.

## 8. Testing

- **`gk_distribution_mask` core:** goal-kicks-only when `frames=None`; full mask with frames; dtype-safe
  ids (numeric-vs-string, ADR-019); NaN team/player → not-in-scope.
- **`native` vs `robust` divergence (proves the lever tightens, `robust ⊆ native`):**
  - **single detected keeper → `robust` == `native`** (the agreement case).
  - **GK substitution fixture → they diverge:** a pass by the *substituted-off* keeper after the sub is
    `True` under `native` (its global set still contains that keeper) but `False` under `robust` (the
    time-resolved acting keeper is the new one). Assert `robust` is the strict subset. This is the honest
    demonstration — NOT "robust finds a pass native misses" (impossible given native's global set).
- **`gk_distribution_mask` predicate edges (lock the exact semantics):**
  - **throw-in by the GK → True** (`is_open` includes `throw_in`, not just `pass` — a distinct `type_id`).
  - **GK shot / GK clearance → False** (only pass/throw-in are open-play distributions).
  - **goal-kick by a non-GK actor → True** (the goal-kick term is actor-independent).
  - **Index contract:** a fixture where `actions` has a **non-`RangeIndex`** (e.g. filtered per-match)
    returns a correctly index-aligned Series with no positional misalignment.
  - **Fail-loud on absence:** `actions` missing a required column (e.g. no `type_id`/`player_id`) raises
    a clear `ValueError` naming the column — pins the §3 contract (otherwise fail-loud is aspirational).
  - **`game_id` key-shape (both modes):** with AND without `game_id`, both `native` (the
    `(game,team,player)` key degrading to `(team,player)`) and `robust` (the `use_game` branch) behave
    correctly. The lakehouse always passes `game_id`; xt_gk callers may not — a quiet key-shape
    divergence between modes would otherwise be easy to miss.
- **Golden shim gate:** v1 `_gk_distribution_mask` byte-identical to a committed snapshot. **The snapshot
  fixture MUST contain ≥1 native-detected GK open-play pass** — assert a non-trivial True count on
  *non-goal-kick* rows — so the golden actually exercises the `is_open & actor_is_gk` set-membership
  logic (the riskiest part of the refactor), not just the goal-kick term. xthreat/v1 byte-stability
  gates stay green.
- **ρ loader / domain:** `is_gk_distribution` **present + non-NULL** → domain broadens; **present + NULL**
  (the rollout population) → coalesced to False → goal-kicks-only (the classic partial-migration case,
  explicitly tested); **absent** → goal-kicks-only. Plus a pure unit test of
  `should_select_is_gk_distribution({...})` for the present/absent decision (no live connection); the
  catalog-qualified DESCRIBE IO itself is the owner-run seam.
- **Loader-drop safety guard:** assert the bundled `default` model's `feature_names` does **not** contain
  `gk_was_distributing` (nor any dropped column) — pins that removing it from the SQL can't break
  inference (it was a domain input, not a feature).
- **Public-surface gates:** the new export auto-registers in the tracking `__all__`; confirm the
  Examples/id-dtype/nan-safety auto-enumerating gates accept it (it's a resolver, not an `add_*`
  aggregator — C4 count unchanged).

## 9. Release

Minor bump **4.42.0 → 4.43.0** (new public API) + tag — the lakehouse needs a released version to pin.
ADR-036 amendment (or an ADR-024 sibling) documenting: the public `gk_distribution_mask` + `resolve_gk`
lever, the `is_gk_distribution` contract, the v1 shim, and the ρ loader's `gk_was_distributing →
is_gk_distribution` correction. Standard version-lockstep (pyproject / `__init__` / CHANGELOG / TODO /
uv.lock) + `/final-review` (C4 count stays 28).

## 10. Open items for the implementation session

1. **`native` shim byte-identity** — the one hard gate. Prove by golden snapshot **on a fixture that
   contains ≥1 native GK open-play pass** (§8), not just goal-kicks, so the set-membership branch is
   actually covered.
2. **Probe mechanism** (resolved in §5): catalog-qualified existence check + a pure, unit-testable
   `should_select_is_gk_distribution` decision. The plan picks the lightest catalog-scoped query (a
   single per-cohort `DESCRIBE`/information_schema call is fine; NOT a per-row cost).
3. **`resolve_gk` default = `robust`** (resolved): only the shim calls with `native` (explicit); new
   callers want robust. Verified no in-repo native-by-default dependency.
4. **No import cycle** (resolved in §3): `_gk_resolve.py` does not import `_xt_gk` (verified) — the
   shim's import is safe.
