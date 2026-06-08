# Design: xT as a VAEP feature — `xt__<method>` xfn factory

**Date:** 2026-06-08
**Status:** Approved (brainstorming) → implementation plan
**Author:** silly-kicks session (Karsten)
**Origin:** Deferred follow-up explicitly carved out of SK-xT-1
(`docs/superpowers/specs/2026-06-07-xthreat-pluggable-xt-promotion-design.md`, "Out of scope":
*"A new VAEP `xt__<method>` xfn factory (xT is a model, not a per-action feature; can be added
later)."*). This is that "later".

## Context

SK-xT-1 (4.17.0, ADR-021) turned `xthreat` into a pluggable package: `ExpectedThreat(method=...)`
with a byte-identical `singh_counts` default and a `kde_smoothed` flavor, plus a held-out
transition-NLL evaluator. xT today is a **standalone model** and an **input** to several tracking
features (`gk_influence`, `player_influence`, `cover_shadows` read its `.xT` / `.interpolator()`).
It is **not** a per-action VAEP feature.

This change wires xT ratings into the VAEP feature framework as a **frame-free** feature-transformer
factory, following the ADR-005 §8 `<feature>__<method>` naming convention
(`xt__singh_counts` / `xt__kde_smoothed`). The xfn is orthogonal to the model itself — it consumes a
fitted `ExpectedThreat` and emits one column per gamestate slot.

`ExpectedThreat.rate(actions)` already returns per-action xT (NaN for non-move actions), so the
standard transform is a thin wrapper. The atomic-SPADL mirror is the only non-mechanical piece
(atomic has no per-action `result` column — see Atomic semantics).

## Decision

Ship a **caller-supplies-the-model** xfn factory for both standard and atomic SPADL:

> `xt_xfns(*, model: ExpectedThreat)` returns a list with one feature-transformer that emits
> `xt__<model.method>` per gamestate slot, NaN-preserving for non-move / failed-move actions.
> Kept out of every default xfn list (opt-in). The atomic mirror reuses the **same `model.rate()`
> path** by synthesizing a standard-SPADL-shaped frame with a **type-aware** `result_id` (dribble
> intrinsic; pass/cross next-atom-`receival`) — `rate()` stays untouched and is the single rating
> path for both flavors.

## Scope

### In scope
1. **Standard factory** `silly_kicks/vaep/features/expected_threat.py::xt_xfns(*, model)`.
   - Closes over a **fitted** `ExpectedThreat`; per slot calls `model.rate(slot)` and emits
     `f"xt__{model.method}"`, lifted to `_a0/_a1/_a2` via the existing `@simple` decorator.
   - Re-exported from `silly_kicks.vaep.features` and `silly_kicks.vaep`.
2. **Atomic mirror** `silly_kicks/atomic/vaep/features.py::xt_xfns(*, model)`.
   - Same column name; move atoms = `pass`/`dribble`/`cross` (via **`atomic.spadl.config`** ids);
     **type-aware** success (dribble = always successful; pass/cross = next atom is `receival`).
   - **Reuses `model.rate()` unchanged** by synthesizing a standard-SPADL-shaped frame (see Atomic
     semantics) — no `_rate_cells`, no orientation/dtype risk, no private-symbol import.
   - **Manual slot loop, NOT `@simple`** (cross-row success ⇒ rate once on `states[0]`, map to slots).
   - Re-exported from `silly_kicks.atomic.vaep`.
3. **No `rate()` refactor / no `_rate_cells` extraction.** An earlier draft extracted the cell-lookup
   into a shared `_rate_cells` helper. Dropped in favor of the synthesized-`result_id` adapter above,
   which makes the **entire** `rate()` path (filter + NaN-drop + y-flip + delta) the single source of
   truth — strictly stronger than sharing only the cell lookup — while leaving `rate()` literally
   untouched (zero parity risk) and avoiding a cross-package private-symbol import
   (`atomic.vaep → xthreat._grid._rate_cells`). See Alternatives.
4. **Fail-closed validation** (see Public API).
5. Tests (TDD), NOTICE is unaffected (no new published method — xT/Singh already cited under
   SK-xT-1), ADR, CHANGELOG, version bump.

### Out of scope (explicit non-goals — YAGNI, not deferred TODOs)
- **Bundled / frozen default xT grid artifact.** The factory takes a live fitted model only. A
  shipped `default` grid (npz+JSON+SHA256, ADR-011 lifecycle: corpus provenance, model card,
  retrain-trigger semantics) is a *separate* deliverable. The signature reserves a typed door
  (`model: ExpectedThreat | str | None`) so it drops in later without an API break, mirroring
  `xshot_occurrence_xfns(model: ... | str | None)`; `str`/`None` raise informatively for now.
- **Adding `xt_xfns` to any default/union xfn list.** Opt-in (see Default list).
- **`use_interpolation` in the xfn.** The xfn uses the discrete grid (`rate(..., use_interpolation=
  False)`); callers wanting the interpolated surface fit/rate xT directly. Keeps the feature cheap
  and deterministic; can widen later non-breakingly.
- **Conditional / per-context xT** (pre-publication; tracking-join-dependent — SK-xT-1 non-goal).

## Architecture

### Where the factory lives, and the dependency direction
Standard VAEP feature-transformers all live in `vaep/features/`; xT-as-feature is a standard,
**non-frame-aware** feature, so it belongs there (alongside `spatial.py`, `temporal.py`, …) — not
co-located with the model the way the *frame-aware* `xshot_occurrence_xfns` lives in `tracking/`.
This introduces a single clean one-way edge **`vaep → xthreat`** (VAEP, the consumer, imports the xT
model). Verified no cycle: `xthreat` imports only `spadl.config` / sklearn / scipy; `vaep` does not
currently import `xthreat`. Using the native `@simple` decorator (already in `vaep`) gives the
`_a0/_a1/_a2` lifting + naming for free.

The atomic mirror lives in `atomic/vaep/features.py` (a single module, not a package) — the same home
as atomic `location`/`movement`/`play_left_to_right`. It imports `ExpectedThreat`'s `_grid` helpers
from `xthreat` (atomic.vaep → xthreat, also clean/one-way).

### Public API

```python
# silly_kicks/vaep/features/expected_threat.py   (standard)
def xt_xfns(*, model: "ExpectedThreat | str | None" = None) -> list[FeatureTransfomer]:
    """Factory: one frame-free transformer emitting xt__<method>_a{0,1,2}.

    NOT added to any default xfn list (opt-in). Caller fits + freezes the grid on their VAEP
    training corpus (or a disjoint exogenous corpus) and reuses the identical object at serve
    time — train/serve consistency is the caller's responsibility (mirrors FrozenXt / ADR-009).
    """
    if isinstance(model, str):
        raise NotImplementedError("bundled xT grid variants are not shipped yet; pass a fitted ExpectedThreat")
    if model is None:
        raise ValueError("xt_xfns requires a fitted ExpectedThreat (model=...)")
    if not np.any(model.xT):          # same fitted-check rate() uses
        raise NotFittedError("xt_xfns requires a fitted ExpectedThreat; call model.fit(actions) first")
    col = f"xt__{model.method}"
    @simple
    def _xt(actions: Actions) -> Features:
        return pd.DataFrame({col: model.rate(actions)}, index=actions.index)
    _xt.__name__ = col
    return [_xt]
```

Atomic signature is identical; only the inner transform differs (see Atomic semantics). The column
name is **derived from `model.method`** — there is no separate `method=` argument, so the column can
never disagree with the model that produced it. The conventional default is therefore whatever
`ExpectedThreat()` defaults to (`singh_counts`).

### Atomic semantics (the only non-mechanical piece)

Standard `rate()` success-filters for free via `result_id == success` (`_grid.py:128`; synthetic
dribbles are always `success` by construction, so standard rates **every** dribble). Atomic SPADL has
**no per-action result column** — pass/cross success is encoded in a *following atom*, while dribbles
carry their success intrinsically. To keep `xt__<method>` **semantically identical** across both
flavors (same name ⇒ same meaning — Hyrum), the atomic success predicate is **type-aware**:

- **Move atoms:** `type_id ∈ {pass, dribble, cross}` (mirror of standard `_get_move_actions`).
- **Dribble ⇒ always successful.** Atomic dribbles are inserted by `_add_dribbles` only as
  *same-team carries* between consecutive same-team actions — their existence *is* the success
  signal, and they are **never** followed by a `receival` atom (the `passlike` list that generates
  receival/interception/out/offside follow-ups, `atomic/spadl/base.py:87-98`, contains `cross` but
  **not** `dribble`). A blanket next-atom-`receival` test would NaN 100% of dribbles while standard
  rates 100% of them — the exact symmetry break this design exists to prevent. So dribbles get a
  finite delta unconditionally.
- **Pass / cross ⇒ next atom is `receival`** (within the same `game_id`+`period_id`, by `action_id`
  order). Explicit fail follow-ups (`interception`/`out`/`offside`) ⇒ NaN. This mirrors atomic's own
  encoding of pass success (`_compute_pass_extras`) — the `gk_xt_delta` precedent
  (`atomic/spadl/utils.py:703-720`) was validated on GK passes/launches only, so it is borrowed for
  pass/cross **only**, not generalized to dribbles.
- **Delta — reuse `model.rate()` via a synthesized frame (single rating path):** build a thin
  standard-SPADL-shaped frame from the move atoms — `start_x=x`, `start_y=y`, `end_x=x+dx`,
  `end_y=y+dy`, `type_id` mapped by **name** to the standard `pass`/`dribble`/`cross` id, and
  `result_id` = `success` iff the type-aware predicate above holds (else a non-success id) — then call
  `model.rate(synthetic_frame)` **once**. `rate()` then does all the filtering, NaN-drop, the
  `rsub(w-1)` y-flip, and the `model.xT` lookup itself, so the atomic path inherits the correct
  orientation for free (no `(12,8)` confusion, no flip to re-implement). NaN coords ⇒ NaN (rate's own
  `dropna`). Non-move atoms are simply absent from the synthesized frame ⇒ NaN.
- **Pitch frame:** atomic SPADL coords are in the same `105×68` frame as standard
  (`spadl.config.field_length/width`) in this codebase; `rate()` bins via `spadl.config`, so feeding
  it atomic `(x, y)`/`(x+dx, y+dy)` is correct. The plan adds an explicit verify step for this.
- **Inherent edge (documented, not a bug):** a pass/cross that is the *last action of a period* has
  no following atom, so atomic cannot observe its success and yields NaN, whereas standard may rate it
  if `result_id==success`. This is an unavoidable property of atomic's representation (it encodes pass
  success only via the follow-up atom), affects ≤1 action per period, and is documented rather than
  papered over.

**Why a manual slot loop, not `@simple`:** next-atom success (for pass/cross) is a *cross-row*
relationship. `@simple` applies the transform to each gamestate slot independently, which is
error-prone for a cross-row predicate on the shifted slots. Instead the factory computes the
per-action xT-delta **once** on the unshifted stream `states[0]` (synthesize the frame + one
`model.rate()` call, as above), then for each slot emits that value mapped to the slot rows. (Standard
*can* use `@simple` because `rate()` reads each row's own `result` — success is self-contained per
row.) The loop mirrors the manual-slot pattern in `xshot_occurrence_xfns`, with `_a{i}` suffixing
applied by hand.

**Map on the composite `(game_id, period_id, action_id)` key — not bare `action_id`.** Atomic
`action_id` is `range(len)` *per converted game*, so bare-`action_id` mapping would collide across
games if a caller ever invokes the transformer on a multi-game concatenation. The standard
`@simple`+`rate()` path is positional and has no such coupling; keying on the composite (all three
columns are present on every slot row, including boundary-filled rows) removes that latent footgun and
keeps the atomic path robust to batching. `states[0]` is the full ordered action stream for the game
(`gamestates()` returns `states[0] = actions`), the correct basis for next-atom detection and the
join key.

**Boundary handling — map by the composite key, do NOT force NaN.** `gamestates()`
(`feature_framework.py:126-138`) fills boundary slot rows with the group's *first* action's values
(not NaN). So under standard `@simple`, a boundary `a1` slot carries a real action's row and `rate()`
emits *that* action's delta. For symmetry, the atomic loop maps each slot row to the precomputed
per-action delta by composite key; a boundary row's key is a real, present key ⇒ map hit ⇒ the same
filled action's delta. There is **no** "boundary ⇒ NaN" clause (an earlier draft had one — it would
have re-introduced an asymmetry vs standard). The only NaN sources are: non-move atom, failed
pass/cross, last-action-of-period pass/cross, and NaN coords.

LTR/away-team handling is already the caller's responsibility upstream (atomic `play_left_to_right`),
same as every other atomic feature — the xfn does not re-mirror.

**Column discovery:** the atomic xfn must run under atomic-VAEP's `feature_column_names` (which builds
atomic-shaped dummy actions: `x`/`y`/`dx`/`dy`/`type_id`/`action_id`/…). The dummy run only needs to
not crash and to surface the column names (derived from `model.method`); the plan verifies the atomic
dummy-action schema covers the columns the loop reads.

### NaN handling
Faithful pass-through of `rate()`'s contract: NaN for non-move / failed-move actions. All three VAEP
learners (xgboost/catboost/lightgbm) split on missingness natively; documented so a caller on a
non-NaN-native learner can `fillna` themselves. No silent 0-fill (would misrepresent "no move" as
"zero threat change").

### Default list & retrain implication
`xt_xfns` is in **none** of `xfns_default` / `xfns_default_no_goalscore` / `hybrid_xfns_default` /
`hybrid_xfns_default_no_goalscore` / any tracking union. Caller opts in:
`VAEP(xfns=fs.xfns_default + xt_xfns(model=frozen_xt))`. Consequences:
- **Additive — zero forced retrain** for existing users (mirrors `xshot_occurrence_xfns` "not in any
  default until weights ship", and SK-xT-1's "additive, no retrain trigger" posture).
- A caller who adds it triggers *their own* retrain by choice; documented in docstring + CHANGELOG.
- A guard test asserts the symbols are absent from all default lists (regression tripwire against an
  accidental future inclusion that would silently re-shape every consumer's feature matrix).

## Alternatives considered

| Option | Why rejected |
|--------|--------------|
| Inline-fit xT inside the xfn | Fits the surface on (a slice of) the data it then rates → leaky + non-reproducible at serve. xfn must consume a frozen exogenous model (FrozenXt/ADR-009 discipline). |
| Bundle a frozen default grid now | Full ADR-011 model-lifecycle weight (provenance, card, SHA256, retrain semantics) — a separate deliverable. Reserved via the typed `str` door. |
| Add to `xfns_default` | Forces a global retrain for every provider and imposes a non-standard methodological choice (xT-as-VAEP-input) on all users. Opt-in instead. |
| Separate `method=` arg on the factory | Could disagree with `model.method` → latent column-mislabel bug. Derive from the model. |
| Atomic raw-positional-delta (no success filter) | `xt__<method>` would then *mean something different* in atomic vs standard under the same name (Hyrum trap). Success-filter (type-aware) for true symmetry. |
| Blanket next-atom-`receival` success for all move atoms | NaNs every dribble (dribbles are never followed by `receival`; `passlike` excludes `dribble`) while standard rates all dribbles → symmetry break on a huge action class. Type-aware: dribble intrinsic, pass/cross next-atom. |
| Reuse `gk_xt_delta` binning for atomic | Wrong grid (`(12,8)` rows=x/cols=y) vs the model's `(12,16)` `[y,x]`. Reuse `model.rate()` (correct grid + flip built in). |
| Extract a shared `_rate_cells` cell-lookup helper | Shares only the cell lookup (standard still runs full `rate()`, atomic a manual loop), refactors shipped `rate()` (parity risk), and needs a cross-package private-symbol import. Synthesize a `result_id` and reuse the whole `rate()` path instead — stronger single-source-of-truth, `rate()` untouched, public-only dependency. |
| Skip atomic mirror | Leaves atomic VAEP with no general xT feature; the work is bounded (reuses `rate()`), so close the question rather than park it. |

## Testing strategy (TDD + hexagonal + e2e)

**Standard unit (red-first):**
- `xt_xfns(model=fitted)` returns one transformer; `feature_column_names([*xt_xfns(model=m)])` yields
  `["xt__singh_counts_a0", "_a1", "_a2"]` (non-frame-aware path: called as `f(gs)`).
- Per-slot values equal `model.rate(slot)` exactly; non-move actions are NaN.
- Column name tracks the model: a `method="kde_smoothed"` model ⇒ `xt__kde_smoothed_*`.
- Fail-closed: `model=None` → `ValueError`; `model="default"` → `NotImplementedError`; unfitted
  `ExpectedThreat()` → `NotFittedError`.

(No `rate()` refactor ⇒ no extraction-parity test needed; the existing SK-xT-1 parity gate + golden
snapshots already pin `rate()` and stay untouched.)

**Atomic — the universal symmetry oracle is the load-bearing test:**
- **Universal cross-representation oracle (red-first, load-bearing):** convert the same committed game
  to both standard and atomic SPADL; assert the atomic `xt__<method>` delta equals the standard
  `rate()` delta for **every move action** (matched across representations), with the **only** allowed
  difference being the documented last-action-of-period pass/cross edge. This single oracle catches:
  the dribble symmetry break, pass/cross success mismatches at `out`/`offside`/`throw_in` follow-ups
  (where the next atom is not `receival`), and any coordinate-frame / orientation drift — failure
  modes the per-case unit tests cannot all surface. *(A blanket next-atom-`receival` predicate fails
  this oracle on dribbles; the type-aware predicate passes.)*
- Targeted unit cases kept as **documentation** (the oracle is the gate): dribble → finite; pass/cross
  + `receival` → finite; pass/cross + `interception`/`out`/`offside` → NaN; pass/cross last-of-period
  → NaN (inherent edge); non-move atom → NaN; NaN coords → NaN; column-name symmetry.
- **Boundary slot-mapping symmetry oracle:** with `nb_prev_actions >= 1`, assert the atomic
  `xt__<method>_a1` value at a group boundary equals the **standard `@simple`+`rate()` `_a1` value**
  for the same converted game (the first-in-group action's delta, since `gamestates()` fills
  boundaries with first-in-group — NOT NaN). Proves the manual loop maps by the composite key and
  stays symmetric at boundaries.
- **Multi-game composite-key guard:** run the atomic transformer on a 2-game concatenation; assert no
  cross-game `action_id` collision corrupts the deltas (bare-`action_id` mapping would; composite key
  must not).

**Default-list guard:** `xt_xfns(...)`-produced fns are not `in` any default/union list (by identity);
meta-style so it can't silently rot.

**VAEP integration (hexagonal):** `VAEP(xfns=fs.xfns_default + xt_xfns(model=m))` →
`compute_features(game, actions)` runs, the `xt__singh_counts_*` columns are present and dtype-float;
a small fit/rate smoke confirms the learner accepts the NaN column. Atomic counterpart via
`AtomicVAEP`.

**e2e (committed fixture, no network):** on the committed WC2018 StatsBomb SPADL fixture
(`tests/datasets/statsbomb/spadl-WorldCup-2018.h5`), fit `ExpectedThreat`, build a VAEP with the xT
feature, compute features for one game, assert finite xT values exist for successful moves and NaN for
shots. **Atomic e2e (the riskier path gets e2e coverage too):** convert the same fixture game to
atomic SPADL and run the cross-representation invariant at integration scale through `AtomicVAEP`
(`compute_features`), asserting the atomic `xt__<method>` column matches standard per move action
(minus the period-last edge) — this is where coordinate-frame and composite-key bugs hide that the
unit oracles on small synthetic frames can miss. Both marked in the regular suite (fixture is
committed → not `@pytest.mark.e2e`).

## Coordination / isolation

1. **Calibration untouched.** `FrozenXt` / `calibration/_xt.py` (the other session's TF-24 area, inside
   my isolation boundary) is **not** modified; this feature only *consumes* `ExpectedThreat`. `rate()`
   itself is unchanged (the atomic path reuses it via a synthesized frame), so there is no xthreat
   internal refactor at all. No edit to `silly_kicks/calibration/`, `scripts/calibrate_*`,
   `scripts/_loader_*`.
2. **C4-free.** The C4 `tracking` container enumerates tracking *backends / trained models /
   aggregator count*; a `vaep` + `xthreat` feature changes none of those tokens or the count → confirm
   unchanged, skip regen.
3. **NOTICE** unchanged — no new published methodology (Singh xT already cited under SK-xT-1).
4. **ADR** (next free at release; reconcile per the version-bump checklist) records the decisions the
   body actually makes: (a) model-provenance / train-serve contract (caller-frozen, fail-closed);
   (b) opt-in, not in any default list; (c) atomic **type-aware** success (dribble intrinsic;
   pass/cross next-atom-`receival`) reusing `model.rate()` via a synthesized `result_id`; (d) the
   documented last-action-of-period NaN edge; (e) boundary map-by-composite-key symmetry.
5. **Versioning** — minor bump to **4.19.0** (next free after `origin/main` 4.18.0; ADR-022).
   One feature branch (`feat/xt-vaep-feature`), single commit (spec + ADR + code + tests + CHANGELOG +
   version sites bundled), PR at the end. No standalone doc commits. No commit/PR without explicit
   approval.

## Success criteria
- `xt_xfns(*, model)` (standard + atomic) ships, fail-closed, opt-in, column `xt__<model.method>`.
- Standard per-slot values equal `model.rate()`; the atomic mirror reuses `model.rate()` via a
  synthesized **type-aware** `result_id` (dribble intrinsic; pass/cross next-atom-`receival`) and maps
  to slots by the composite `(game_id, period_id, action_id)` key; both emit the identical column name.
- `rate()` left unchanged (no extraction); existing SK-xT-1 parity gate + golden snapshots stay green.
- Universal cross-representation symmetry oracle green (atomic == standard per move action, minus the
  documented period-last edge); multi-game composite-key guard green.
- Not present in any default/union list (guard test).
- VAEP + AtomicVAEP integration green; committed-fixture standard **and** atomic e2e green.
- ruff + ruff format + pyright clean; full suite green.
