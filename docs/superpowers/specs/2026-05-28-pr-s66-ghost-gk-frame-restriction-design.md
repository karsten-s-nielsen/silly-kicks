# PR-S66 — Ghost-GK linked-frame restriction (perf, bit-identical)

**Date:** 2026-05-28
**Status:** Design approved (pending written-spec review)
**Target version:** 3.26.0 (minor — adds one backward-compatible public kwarg)
**Type:** Performance optimization. No new feature columns. Bit-identical output.

## 1. Problem

After 3.25.0 restricted DAS + shape_graph to action-linked frames, `add_ghost_gk`
is the slow batch step in the lakehouse per-match enrichment pipeline. It calls
`compute_ghost_gk(frames, ...)` which runs `_extract_all_ghost_gk_features` over
**all** frames in the batch (~250/match) and then `predict_density` over **every**
`(frame, GK-team)` sample, before mapping a handful of those to actions via the
link pointers. This is the all-frames-then-map anti-pattern that DAS/shape_graph
already shed.

The dominant cost is `GhostGkModel.predict_density`: a per-sample weighted
`gaussian_kde` evaluated on a 60×64 grid (3 840 points) against the model's
training set. With ~2 GKs × ~250 frames that is ~500 KDE evaluations per match;
only ~`n_actions` of them are ever read. Feature extraction (per-frame pandas +
a small `ConvexHull`) is cheap by comparison — empirically 1–2 orders of
magnitude below the KDE cost. (This ordering will be **empirically confirmed** on
real data before ship; if extraction turns out to be a second bottleneck, that is
a separate follow-up, not this PR.)

## 2. Why a naive restriction is NOT bit-identical

`compute_ghost_gk` cannot simply pre-filter `frames` to the linked frame_ids.
`_extract_all_ghost_gk_features` carries two cross-frame dependencies:

1. **Per-period defending-goal inference** (`_ghost_gk.py`, the
   `groupby(["game_id","period_id","team_id"])["x"].mean()` → `_defending_goal`
   dict). The GK's mean x over the whole period decides which goal (`x=0` vs
   `x=105`) the GK defends, which drives the goal-relative coordinate flip.
   Restricting frames changes the mean and can flip the inferred goal end.

2. **Velocity state across consecutive frames** (the `prev_state` /
   `prev_timestamps` loop producing `defensive_line_speed` and
   `defending_centroid_vx`). Each feature row's velocity is a strict **one-step
   delta** — `(defensive_line_x − prev_defensive_line_x) / dt` — against the
   **previous processed frame** for that `(game_id, gk_team)`; `prev_state` is
   overwritten every iteration (no rolling window / multi-frame smoothing).
   Verified for the §5 variant (N1). Two subtleties the variant must reproduce
   exactly: (a) `state_key = (game_id, gk_team)` does **not** include `period_id`,
   so the predecessor is the previous frame in `(game_id, period_id, frame_id)`
   sort order and **crosses period boundaries** (period-2's first frame's
   predecessor is period-1's last frame); (b) `dt` falls back to
   `_VELOCITY_WINDOW_S` whenever `time_seconds` does not strictly increase vs the
   predecessor (incl. the cross-period reset and the first-ever frame). Dropping
   unlinked frames changes a linked frame's predecessor → changes both the delta
   and `dt`.

Both deps live inside `_extract_all_ghost_gk_features`. (Score/phase context come
from time-keyed callbacks built off `actions`, independent of which frames are
present, so they are not cross-frame.)

## 3. Design — restrict the KDE, keep extraction on full frames

The expensive stage (`predict_density`) is **purely per-sample**: its loop
`for i in range(len(X))` reads only `query_leaves[i]` plus the frozen training
arrays; there is zero cross-sample coupling, and `_vectorized_leaf_indices`
gathers per row. Therefore a restricted sample's density is byte-identical to its
value in the full run.

So the bit-identical optimization is:

> Run `_extract_all_ghost_gk_features` over the **full** frames (preserving both
> cross-frame deps exactly as today), then restrict `batch_features` + `meta` to
> the linked-frame rows **before** calling `predict_density`.

This is structurally simpler than the DAS fix: no direction column to pin and
thread through, because the cross-frame deps are resolved by keeping extraction
full rather than by precomputing-and-passing.

### 3.1 `compute_ghost_gk` — new kwarg

Add `link_frame_ids: set | None = None` (keyword-only, defaults `None` →
unchanged full behaviour; backward compatible, and the VAEP `ghost_gk_xfns` path
keeps calling it with no restriction).

```python
def compute_ghost_gk(
    frames, *, model=None, home_team_id, actions=None,
    link_frame_ids: set | None = None,
) -> pd.DataFrame:
    ...
    batch_features, meta = _extract_all_ghost_gk_features(frames, ...)  # FULL frames
    if len(batch_features) == 0:
        return out
    if link_frame_ids is not None:
        keep = meta["frame_id"].astype(int).isin(link_frame_ids)
        batch_features = batch_features.loc[keep.to_numpy()].reset_index(drop=True)
        meta = meta.loc[keep].reset_index(drop=True)
        if len(batch_features) == 0:
            return out
    densities = resolved.predict_density(batch_features)  # restricted subset only
    ... (merge into GK rows unchanged) ...
```

`meta["frame_id"]` originates from groupby keys (never NaN), so `.astype(int)` is
safe; it mirrors the DAS `int(float(fid))` normalisation. Unlinked GK rows keep
`ghost_gk_* = NaN`, which is harmless — the action mapping only reads linked
frames.

### 3.2 `add_ghost_gk` — always restrict via pointers

`add_ghost_gk` always has `pointers` (the supplied `links`, or
`link_actions_to_frames(...)` computed internally). Derive the restriction set
from whichever it has — **broader win than `add_das`** (which only restricts on
explicitly-supplied `links`), and still bit-identical because the pointers
reference valid frame_ids regardless of source:

```python
link_frame_ids = set(pointers["frame_id"].dropna().astype(int).tolist())
...
ghost_frames = compute_ghost_gk(
    frames, model=resolved_model, home_team_id=home_team_id,
    actions=actions_for_context, link_frame_ids=link_frame_ids,
)
```

The existing short-circuit (`if "ghost_gk_x" in frames.columns and …`) is
untouched — when frames are pre-computed (the xfns path), no compute runs.

### 3.3 `ghost_gk_xfns` (VAEP path) — restrict via union of gamestate slots

In scope. The `_frame_aware` transformer receives the gamestate action slots
(`states`, i.e. a0/a1/a2) alongside `frames`, so it *can* link internally even
though VAEP supplies no `links` kwarg. Link each slot once, take the **union** of
their linked frame_ids, and pass it to the single `compute_ghost_gk` call:

```python
def _ghost_gk_transformer(states, frames):
    ...
    slot_pointers, link_frame_ids = [], set()
    for slot in states[:3]:
        pointers, _ = link_actions_to_frames(slot, frames)
        slot_pointers.append(pointers)
        link_frame_ids |= set(pointers["frame_id"].dropna().astype(int).tolist())
    ghost_frames = compute_ghost_gk(
        frames, model=resolved, home_team_id=home_team_id,
        link_frame_ids=link_frame_ids,
    )
    for i, (slot, pointers) in enumerate(zip(states[:3], slot_pointers)):
        enriched = add_ghost_gk(slot, ghost_frames, model=resolved,
                                home_team_id=home_team_id, links=pointers)
        ...
```

The union guarantees every frame any slot's `add_ghost_gk` mapping reads is
present (others stay NaN, never read). The pre-computed `ghost_frames` short-
circuits each per-slot `add_ghost_gk` compute, and reusing `pointers` as `links`
avoids re-linking. Bit-identical to the current all-frames `ghost_gk_xfns` output.
The `frames is None` column-probing branch is unchanged.

`states[:3]` (N3) reproduces the **original** transformer's loop bound exactly —
the current `_ghost_gk_transformer` iterates `states[:3]` and its `frames is None`
branch fills `range(3)`. The `[:3]` here is intentional parity, not a new cap.

### 3.4 Untouched paths
- `prepare_ghost_gk_training_data`: training, full frames, `link_frame_ids=None`,
  unchanged.
- Atomic mirror (`atomic/tracking/features.py`): re-exports `add_ghost_gk` /
  `ghost_gk_xfns` from the main module — inherits the fix with no separate change.
  (Verify the re-export, no duplicate impl, in the plan.)

## 4. Bit-identical guarantee — golden masters (the crux)

Two tests, mirroring `test_das.py::TestDasLinkedFrameRestriction`.

### 4.1 Regular suite (synthetic, non-e2e)
A small hand-built fixture engineered to exercise **both** cross-frame deps:
- **Velocity dep:** ≥3 frames where the linked frame's true predecessor differs
  from its previous *linked* frame, so a naive frame-level restriction would
  change `defensive_line_speed` / `defending_centroid_vx`.
- **Goal-flip dep:** a period whose full-frame GK mean-x lands on one side while
  the linked-frame subset alone would flip the inferred defending goal (analogous
  to the DAS `J03WMX` adversarial fixture).

Assertion: for every linked-frame GK row,
`compute_ghost_gk(frames, link_frame_ids=None)` (full) equals
`compute_ghost_gk(frames, link_frame_ids={…})` (restricted) **exactly** for
`ghost_gk_x`, `ghost_gk_y`, `ghost_gk_spread` (`rtol=0`/exact, or `np.isclose`
with `rtol=1e-12` to absorb only float-merge noise).

**Discrimination assertion (C2 — the fixture must prove it bites).** Full ==
KDE-restricted passes trivially on a fixture where the cross-frame deps happen not
to matter. Add a third assertion that a **naive extraction pre-filter** —
`compute_ghost_gk(frames[frames.frame_id.isin(linked)])` (i.e. dropping unlinked
frames *before* extraction, no `link_frame_ids`) — produces ghost_gk_* values that
**differ** from the full/KDE-restricted result on this fixture. This proves the
fixture actually triggers the goal-flip / velocity deps, and that the KDE-only
design is *necessary*, not merely sufficient — pinning why the design is shaped
this way against a future "just pre-filter the frames" simplification.

Plus unit tests mirroring DAS:
- restriction actually shrinks the predict set (spy/monkeypatch `predict_density`
  and assert it receives only the linked subset);
- `link_frame_ids=None` → all samples predicted (unchanged);
- `add_ghost_gk` derives `link_frame_ids` from `pointers` (supplied `links` AND
  internally-computed) and passes it to `compute_ghost_gk`.

**VAEP path (§3.3) golden:** assert `ghost_gk_xfns` output is byte-identical
before vs after the union-of-slots restriction (compare the restricted transformer
to a reference that calls `compute_ghost_gk` over full frames), and that the single
`compute_ghost_gk` call receives the union of the slots' linked frame_ids (spy on
`link_frame_ids`).

### 4.2 e2e (realistic scale + perf evidence)
A larger, realistic fixture (a real tracking match if the standard e2e dataset
path is available; otherwise a dense ~250-frame, 2-period, both-GK synthetic
"match" built to realistic scale). Marked `@pytest.mark.e2e`. Asserts:
- restricted action-coupled output bit-identical to the full compute, AND
- the perf win is real — assert the number of `predict_density` samples under
  restriction equals the linked-frame count and is materially below the
  full-frame count (a structural call-count guard, robust to CI timing noise —
  same shape as the cover_shadows perf guard).

## 5. Empirical validation at full-match scale (pre-ship) — gates extraction scope

The "extraction is 1–2 orders below the KDE" claim was reasoned at small scale
where the KDE dominates. Once the KDE is restricted to ~`n_actions` samples,
extraction (per-frame pandas + `ConvexHull`, **linear in total frame-rows**)
becomes the residual cost — and only becomes visible at full scale. A 2-batch
slice would falsely confirm "good enough." (Consumer context: the lakehouse
measured `add_ghost_gk(default)` at ~225 hr/half pre-fix and needs it under a
**~30-min/half** task budget.)

So §5 runs at **full-match scale, not a slice**: ≥1 full half, both GKs,
~70k frame-rows. Before committing, run `add_ghost_gk` on that match both ways and
confirm:
1. byte-identical action-coupled columns (full vs restricted);
2. the KDE-call / wall-clock reduction; **and**
3. the post-KDE-fix wall-clock against budget (below).

**Measurement scale (N2).** The lakehouse does not run ghost_gk as one monolithic
half — it runs it per ~250-frame `applyInPandas` batch under a `for-each` fan-out.
So capture **two** numbers: (i) the per-half-equivalent wall-clock (~70k rows,
single process) for an apples-to-apples KDE-reduction figure, and (ii) the
**per-batch** time (~250-frame group), since `per-batch × batches-per-iteration`
vs the iteration timeout is the consumer's real constraint. State which scale each
number is at. The budget below is expressed per-half (~30 min); convert to the
per-batch figure when reporting (ii).

**Decision gate — numeric, no judgment call at ship time (N2):**
- post-KDE-fix full-half wall-clock **< 50% of the 30-min budget (i.e. < 15 min)**
  → extraction restriction stays a follow-up (per §7); the C3 breadcrumb records
  why.
- post-KDE-fix full-half wall-clock **≥ 15 min** → extraction restriction comes
  **into scope for this PR**.

**Conditional extraction-restriction variant (if the gate fires).** *Not* a
"linked + immediate predecessor" frame subset — the velocity predecessor crosses
period boundaries and has fiddly `dt`/first-frame handling (§2.2), so a predecessor
*set* is error-prone. Instead, use the same "precompute the cross-frame quantity
over full frames, then pass it in" pattern as the DAS direction-pin:
1. **Velocity state:** make one cheap full-frame pass computing only
   `defensive_line_x` + `defending_centroid_x` per processed frame in sort order
   (no `ConvexHull`, no KDE, no 26-feature row), reproducing the exact
   `(game_id, gk_team)` cross-period `prev_state` chain, to build a per-(linked-
   frame, gk_team) lookup of `(prev_defensive_line_x, prev_defending_centroid_x,
   dt)`.
2. **Goal-mean:** compute the per-period `_defending_goal` dict over the full
   frames (cheap GK-only `groupby`) — already full-frame today; just hoist it.
3. Run the **heavy** `extract_ghost_gk_features` (which already accepts
   `prev_*`/`dt`/`goal_x` params) only on the **linked** frames, injecting (1)+(2).
This is the more invasive path the KDE-only design deliberately avoids unless
forced; it requires its own bit-identical golden (full vs extraction-restricted),
distinct from §4's KDE-only golden.

Capture the measured numbers (all of 1–3, both scales) in the PR description
regardless of outcome.

## 6. Housekeeping (hard gates)
- **Version:** 3.26.0 — bump `pyproject.toml`, `silly_kicks/__init__.py`,
  `TODO.md`, `CHANGELOG.md`; all four must match.
- **TODO.md:** delete the PR-S66 candidate row on ship (CHANGELOG is the record).
  **C3 breadcrumb:** if §5 leaves extraction-restriction deferred, add a
  conditional TODO row ("ghost_gk: restrict `_extract_all_ghost_gk_features` via
  full-frame velocity-state/goal-mean precompute + linked-only heavy extraction —
  only if measured a co-bottleneck") AND a one-line
  comment at the extraction call in `compute_ghost_gk` noting it still runs over
  full frames by design (so the cross-frame deps stay intact), so the residual
  all-frames cost is visible to the next maintainer.
- **No new ADR:** ADR-008's `links` pre-linking convention already governs this;
  add a one-line bit-identical note to the CHANGELOG entry.
- **ruff format --check + ruff check + pyright silly_kicks/** clean; full
  non-e2e suite green. `/final-review` before the single commit.

## 7. Scope boundaries
- **Conditionally in-scope (gated by the numeric §5 trigger):** restricting
  `_extract_all_ghost_gk_features` via the precompute-and-inject variant in §5
  (cheap full-frame velocity-state + goal-mean passed into a linked-frames-only
  heavy extraction). Lands in this PR **iff** the §5 full-half post-KDE-fix
  wall-clock is ≥ 15 min (50% of budget); otherwise deferred to a follow-up.
  Either way, leave the C3 breadcrumb.
- **Out of scope:** the parked 4.0.0 per-pass shared-context refactor
  (`project-xfn-context-design`).
