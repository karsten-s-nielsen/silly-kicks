# ADR-119: gkdv stride-survivor injection for windowed ghost scoring (extends ADR-118/055)

| Field | Value |
|---|---|
| **Date** | 2026-10-12 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

ADR-118 made `gkdv.build_ghost_frames` window-stable by letting a caller inject a whole-unit `GoalMap` + `carrier`. That closed every *per-frame* whole-unit dependency, but the luxury-lakehouse neutrality gate (each window's gkdv output must be byte-identical to whole-unit) still failed on gkdv — a **third** whole-unit dependency the ADR-118 round did not surface.

`GkdvParams.possession_stride = 5` (cost control, spec §5). In `_apply_domain` (`gkdv/_engine.py`) the stride keeps every 5th **eligible** frame, ranked by a `cumcount` over the globally-ordered eligible frames of each `(game_id, period_id)`:

```python
rank = work.loc[surviving].groupby(["game_id", "period_id"], dropna=False).cumcount()
reason.loc[surviving[(rank % stride != 0).to_numpy()]] = _DROP_STRIDE
```

The survivor set is a **whole-unit rank**. Windowing changes the eligible set the rank runs over, so the phase shifts and a **different** set of frames survives. Measured (idsse canary unit, window target 1083): whole-unit survivors `12421, 12426, 12431, …`; windowed survivors `12420, 12425, 12430, …` — the same stride-5, phase off by one. Injecting the ADR-118 `goal_map` + `carrier` does **not** fix it: the stride rank is independent of both.

A full look-ahead audit of the gkdv scoring path (`_apply_domain`, `serve_ghost_gk_positions`, `delta_das`, `delta_threat_suppression`, `_write_back`) confirmed the stride rank is the **sole remaining** whole-unit dependency — every other op is per-frame (and goal-map/carrier-seamed by ADR-118). The **pre-stride** eligible SET is itself window-stable; only the stride SELECTION over it is whole-unit.

## Decision

Expose the whole-unit survivor **selection** and let a windowed caller inject it, so the stride is decided once over the whole unit and the window only *scores* the survivors falling in it:

1. **`select_gkdv_scored_frames(frames, *, carrier=None, goal_map=None, params=_DEFAULT_PARAMS) -> pd.DataFrame`** — a new public function returning the eligible-after-stride survivor `(game_id, period_id, frame_id)`. It runs the §4.1 domain + stride once over the whole unit. Cheap: the domain has no pitch-control / DAS grids, so a positions-only whole-unit pass is bounded. `carrier`/`goal_map` default-`None` resolve internally, mirroring `build_ghost_frames`.
2. **`build_ghost_frames(..., scored_frame_keys: pd.DataFrame | None = None)`** — when provided, `_apply_domain`'s stride step is **replaced** by "keep the eligible frames whose key is in `scored_frame_keys`". The per-frame §4.1 domain still runs (pinning the context the serving/arms need), so a key outside the domain is **not** resurrected; only the global `cumcount` is bypassed. Eligible-pre-stride frames not in the injected set are marked `stride_skipped` — same drop reason, same conservation, decided whole-unit. `None` keeps the internal stride and is **byte-identical** for every existing caller.

The lakehouse computes `carrier`, `goal_map` and `survivors = select_gkdv_scored_frames(...)` once on the whole unit, then calls `build_ghost_frames(window, carrier=…, goal_map=…, scored_frame_keys=survivors)` per window. The union of the windows is byte-identical to the whole-unit gkdv scoring.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Segment the stride per possession so it is window-local | Removes the whole-unit dependency at the source | Changes the stride semantics and every existing gkdv number; a validation + retrain cost nobody asked for | Alters the shipped metric to solve a windowing problem; out of scope |
| B. Drop the stride under windowing (score every eligible frame) | No selection to inject | Not byte-identical (scores 5× the frames); defeats the cost control the stride exists for | Fails the neutrality gate it is meant to pass |
| C. Inject the whole-unit survivor KEYS; per-frame domain still runs; `None` = internal stride (chosen) | One cheap selection pass; byte-identical default; stride value/semantics unchanged | Two public signatures grow (one new fn, one new optional param); threading is a convention, not a gate | — |

## Consequences

### Positive

- gkdv windowed Stage-2 scoring is byte-identical to whole-unit scoring — the last of the three dependencies (goal map, carrier, stride) is closed.
- The stride **value and semantics are unchanged**: the injected survivor set carries the stride already applied whole-unit. No metric change, no retrain, no re-materialize of existing gkdv artifacts.
- The whole-unit selection is reusable and cheap (positions-only), so a caller pays the stride rank once per unit rather than per window.

### Negative

- Two public surfaces grow: a new function and a new optional `build_ghost_frames` parameter.
- No gate enforces that a windowed caller actually injects a *whole-unit* selection: passing a per-window `select_gkdv_scored_frames(window)` would re-introduce the phase drift. The contract is documented here and in both docstrings; the neutrality proof lives in the caller (and in the sk `test_windowed_union_is_byte_identical_to_whole_unit`).
- Downstream windowed callers (the lakehouse) must precompute and inject the survivor set per window, and floor-bump to 4.133.0.

### Neutral

- `scored_frame_keys=None` threads nothing new — the internal stride runs exactly as before, which is what makes the default byte-identical.
- `select_gkdv_scored_frames` is PURE (reads `frames`, mutates nothing) and ships no model: it is a selection helper, not a metric (`add_*`/`compute_*`), so it registers in neither `METRIC_CONTRACTS` nor `PURITY_ENTRIES` (the `add_*` purity gate). It is auto-enforced by the public-Examples gate via `gkdv.__all__`.
- This is the **final** gkdv sk change for lakehouse windowing (confirmed by the look-ahead audit).

## Related

- **ADRs:** extends `ADR-118` (GoalMap/carrier injection for windowed ghost scoring) and `ADR-055` (goal-map unification: build once, thread in); relies on `ADR-019` (canonical id keys for the `scored_frame_keys` membership test).
- **Requesting document:** `D:\Development\_reviews\2026-10-12-silly-kicks-gkdv-stride-seam-handoff.md` (luxury-lakehouse windowing request) — no in-repo spec; handoff-driven.
- **Code:** `silly_kicks/gkdv/_engine.py` (`select_gkdv_scored_frames`, `_apply_domain`, `build_ghost_frames`), `silly_kicks/gkdv/__init__.py`.
- **Tests:** `tests/gkdv/test_engine_scored_frame_keys.py`.
- **Release:** 4.133.0.

## Notes

The stride block is now `surviving = work.index[reason.isna()]` followed by one of two branches: when `scored_frame_keys` is injected, a canonical-key (ADR-019 `_id_key`) set-membership drop of the non-selected survivors; otherwise the original `cumcount` stride. The membership test builds the `(game_id, period_id, frame_id)` keys through `_id_key` on both sides — a raw tuple-set test is a raw `==` in disguise and would silently drop every frame on a dtype mismatch. Verified by TDD: a byte-identical leg (`None` equals the default), a selection-equivalence leg (`select_gkdv_scored_frames` equals the internal stride survivors), a load-bearing leg (an injected subset scores exactly that subset ∩ the per-frame domain; an out-of-domain injected key is not resurrected), and the end-to-end neutrality leg (union of two disjoint windows equals whole-unit, byte-for-byte). Reverting the implementation turns the tests red with `ImportError`/`TypeError: unexpected keyword argument 'scored_frame_keys'`.
