# Spec: `orient_frames_to_ltr` — single-source the frame-LTR contract for consumers

**Date:** 2026-06-13
**Status:** Implemented in 4.27.0
**Decision record:** ADR-029 (to be authored, bundled into the feature commit)
**Origin:** luxury-lakehouse report `tmp/silly_kicks_metrica_skillcorner_ltr_frame_20260613.md` — follow-on to the 4.26.0 / ADR-028 action-LTR geometry fix.

## Executive summary

silly-kicks 4.26.0 (ADR-028) re-projects per-action tracking geometry into the
per-acting-team LTR frame, but only for frames that are already in the canonical
**home-attacks-right** convention. Two of the lakehouse's four tracking providers
— **metrica** and **skillcorner** — have their frames built downstream (from the
lakehouse's own bronze tables) in **absolute** orientation (`team_attacking_direction = None`,
no per-period flip). For those frames, 4.26.0's reprojection silently no-ops, so
~50% of action rows (the team attacking right-to-left in that period) carry
mirror-wrong geometry (`pre_shot_gk_x`, `defensive_line_x`, `nearest_defender_distance`,
`pressure_on_actor__*`, etc.).

The orientation contract is silly-kicks' abstraction but is currently re-implemented
(incompletely) by the consumer for providers that lack a kloppy `TrackingDataset`
input. This spec adds **one public composition helper**, `orient_frames_to_ltr`,
that converts any absolute-orientation frames DataFrame into the canonical
home-attacks-right frame, reusing the exact primitives the native adapters already
use — so the contract is single-sourced.

## Background — what already exists (and what does not)

| Provider | Frame builder | Orientation |
|---|---|---|
| sportec / idsse | `silly_kicks.tracking.sportec.convert_to_frames(output_convention="ltr")` | LTR ✅ |
| gradientsports | `silly_kicks.tracking.gradientsports.convert_to_frames(output_convention="ltr")` | LTR ✅ |
| metrica | kloppy gateway `tracking.kloppy.convert_to_frames` exists, but the **lakehouse builds frames from bronze itself** | absolute ❌ |
| skillcorner | kloppy gateway exists, but the **lakehouse builds frames from bronze itself** | absolute ❌ |

The report's premise that "silly-kicks has no metrica/skillcorner tracking
converter" is **incorrect** — the kloppy gateway covers both. The real gap: the
lakehouse holds **bronze DataFrames**, not kloppy `TrackingDataset` objects, so it
cannot feed the gateway and rolls its own bronze→frame schema mapping. That mapping
(bronze schema, coord scaling) is correctly lakehouse-owned (bounded context:
lakehouse owns ingestion + bronze). The **orientation** is the only part that
belongs upstream — hence option (b), not a native converter (option a).

Two primitives already implement the orientation, and are already shared by the
sportec/GS adapters and the kloppy gateway:

- `tracking.direction.compute_attacking_direction(...)` — per-row pre-flip
  `team_attacking_direction` from `home_team_start_left` + per-period logic. Private
  (stays private — C4).
- `tracking.utils.play_left_to_right(frames, home_team_id)` — per-period flip so the
  home team attacks high x in every period; swaps direction labels. **Public.** This
  is the public entry for frames that are **already labeled** (e.g. kloppy
  `absolute_frame` output).
- `tracking.direction.require_et_direction(...)` — fail-loud ET guard. **Public.**

The gap is the absence of a single composed entry point for **unlabeled** absolute
frames (the lakehouse case: `team_attacking_direction` all-null) — one call that wires
in the schema/ET/zero-match guards and the `compute_attacking_direction` →
`play_left_to_right` composition. `play_left_to_right` already serves the labeled case;
nothing serves the unlabeled case today.

## Empirical confirmation

**Post-4.26.0, isolated (lakehouse local recompute, wheel 0.5.38).** The lakehouse
recomputed AC under 4.26.0; `pre_shot_gk_x` on shot rows shows the exact
converter-vs-builder split the mechanism predicts:

| provider | frame builder | `pre_shot_gk_x` (low / high / avg) | verdict |
|---|---|---|---|
| idsse | sportec `convert_to_frames(ltr)` | 0 / 164 / **102.4** | clean |
| gradientsports | GS `convert_to_frames(ltr)` | 0 / 89 / **101.3** (P1 101.2, P2 101.3) | clean |
| metrica | lakehouse `_bronze_metrica_to_frames` | 27 / 38 / **60.6** | bimodal (per-game inconsistent) |
| skillcorner | lakehouse `_bronze_skillcorner_to_frames` | 111 / 114 / **53.5** | bimodal (≈50/50) |

This supersedes the earlier deployed-table reading (`dev_gold.fct_action_context`,
`updated_at = 2026-06-12`, **pre-4.26.0**), where *all four* providers were bimodal
via the *old* per-team bug 4.26.0 fixes — that stale table confirmed bimodality
existed but could not isolate the metrica/skillcorner-specific cause. The local
4.26.0 recompute makes the isolation empirical, not just deductive.

**Mechanism (code), now empirically corroborated:**
`_action_orientation.acting_team_attacks_rtl` filters frames to
`team_attacking_direction.notna()`; for the all-None metrica/skillcorner frames that
set is empty → returns all-False → no reprojection → absolute coords used as-is →
~50% of actions mirror-wrong. **metrica/skillcorner therefore remain broken even
after the lakehouse re-materializes under 4.26.0** — re-materialization alone cannot
fix them.

## Design

### API

```python
# silly_kicks/tracking/utils.py  (adjacent to play_left_to_right)
def orient_frames_to_ltr(
    frames: pd.DataFrame,
    *,
    home_team_id,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
) -> pd.DataFrame:
    """Orient *unlabeled* absolute-orientation tracking frames into the canonical
    home-attacks-right (LTR) frame, per period.

    For frames that already carry a populated ``team_attacking_direction`` (i.e.
    already labeled, e.g. ``kloppy.convert_to_frames(output_convention="absolute_frame")``
    output), use ``play_left_to_right`` directly — this helper raises on labeled input.

    Returns a new DataFrame; does not mutate the input.
    """
```

### Behaviour (ordered composition; no new orientation math)

All preconditions **raise** — for a load-bearing orientation contract a violated
precondition means the output is *definitely* wrong, not maybe-wrong, so fail loud
(consistent with the ET guard, which already raises unconditionally).

1. **Required-schema guard (C5).** Require columns `x`, `y`, `team_id`, `period_id`,
   `is_ball`, `team_attacking_direction`. Missing → `ValueError` listing the missing
   columns (mirrors sportec's `EXPECTED_INPUT_COLUMNS` pattern). `team_id` dtype is
   *not* hard-asserted — id comparisons route through `ids_match` (ADR-019, which
   tolerates dtype variance); a dtype mismatch that defeats matching is caught by the
   zero-match guard (step 4).
2. **Degenerate input** — empty frame, or no player (non-ball) rows → return a copy
   unchanged.
3. **Already-labeled guard (C2).** If `team_attacking_direction` has **any** non-null
   value on entry → `ValueError` ("frames already carry team_attacking_direction; this
   helper is for unlabeled absolute frames — use play_left_to_right for labeled
   frames"). This makes the not-idempotent edge a loud error rather than a silent
   double-flip, and crisply separates the two public entry points (labeled →
   `play_left_to_right`; unlabeled → `orient_frames_to_ltr`).
4. **Zero-match guard (C1, ADR-019).** Compute `ids_match(team_id, home_team_id)` over
   player rows; if there are player rows but zero match → `ValueError`
   ("home_team_id matched ZERO player rows — id dtype mismatch? orientation would be
   wrong"). RAISES (not warns): zero-match means `play_left_to_right` cannot identify
   flip periods and would return the frames unflipped, i.e. definitely-wrong output.
5. **ET guard** — `require_et_direction(frames["period_id"], home_team_start_left_extratime,
   source="orient_frames_to_ltr")`. Raises `ValueError` if ET periods (3/4) are present
   but the ET flag is unset.
6. **Populate direction** — set `team_attacking_direction` from
   `compute_attacking_direction(team_id=…, period_id=…, is_ball=…, home_team_id=…,
   home_team_start_left=…, home_team_start_left_extratime=…)` (the physical pre-flip
   direction implied by `home_team_start_left`).
7. **Flip to LTR** — return `play_left_to_right(frames, home_team_id)`.

### Contract

- Output convention: **home team attacks x=105 in every period; away attacks x=0**
  — byte-identical to `convert_to_frames(output_convention="ltr")` and exactly what
  ADR-028's reprojection expects.
- **Input must be unlabeled absolute frames** (`team_attacking_direction` all-null) —
  raises otherwise (step 3). Labeled frames use `play_left_to_right`.
- **`home_team_start_left` is the source of truth** for the (now all-null) direction.
  The caller derives the flag (silly-kicks does not infer it — consistent with the
  sportec/GS adapters). **The helper is only as correct as this flag** — see the
  adoption note (C3) on validating it.
- **Not idempotent, and guarded.** A second call raises via step 3 (the first call
  populated the direction column). Test #6 locks this contract.
- Pure: returns a new DataFrame, no global state, no I/O.

### Packaging

- Export **only `orient_frames_to_ltr`** from `silly_kicks/tracking/__init__.py`
  (`__all__` + import). `compute_attacking_direction` stays **private** (C4): the
  composed helper is the single public entry for unlabeled frames, `play_left_to_right`
  (already public) is the entry for labeled frames — no consumer needs the primitive,
  and exposing it would re-invite the orientation-duplication this spec exists to
  prevent.
- **Additive — no retrain trigger for silly-kicks.** Existing providers
  (sportec/GS/kloppy) are byte-unchanged; this is a new helper they do not call.
  Adopting it lakehouse-side fixes metrica/skillcorner → the lakehouse
  re-materializes (its consequence; noted in CHANGELOG, not a bundled-model retrain).
- **ADR-029** (short): frame orientation is single-sourced via `orient_frames_to_ltr`;
  consumers building frames from non-kloppy sources MUST call it before the geometry
  layer. Add to the C4 consumer-contracts clause (matches ADR-017/019/020/028 house
  style). Otherwise C4-free (no new aggregator / backend / model — count stays 27).
- **Version:** next free minor after the tagged release (target **4.27.0**;
  reconcile vs `origin/main` at release). Standard 5-site bump + `uv.lock`.

## Testing (TDD, red-first)

`orient_frames_to_ltr` produces **home-attacks-right frames** — it does NOT do the
per-action reprojection (that is ADR-028's separate downstream job). So the *unit*
tests assert frame-level orientation; one *integration* test chains this helper with
the geometry layer to validate the end-to-end fix the report cares about. All tests
are provider-agnostic on synthetic frames, so they cover metrica/skillcorner without
those providers' fixtures.

Frame convention reminder: "home attacks LTR / high x" means the home team's *own*
goal is at x=0 (defended by the **home GK → x≈0**) and it attacks the x=105 goal; the
away team attacks x=0, so the **away GK → x≈105**. After orient this holds in *every*
period (no per-period flip).

**Unit (frame-level):**

1. **Mirror-invariance guard** (the gap that let this through). Take an unlabeled
   absolute frame `F` and its 180° point reflection `mirror(F)` (`x→105−x, y→68−y` on
   every row) — `mirror(F)` represents the *same logical game kicked off from the
   opposite side*, so `home_team_start_left` flips with it. The precise invariant
   (C6): `orient_frames_to_ltr(F, flag) == orient_frames_to_ltr(mirror(F), not flag)`.
   The flag flips *with* the mirror; this is what makes the assertion meaningful rather
   than degenerate. The durable cross-provider invariant the report asked for.
2. **Per-period orientation.** A 2-period synthetic frame where the home team
   physically attacks right in P1 and left in P2 (absolute). After orient: the home
   GK sits at x≈0 and the away GK at x≈105 in **both** periods; direction labels are
   home="ltr"/away="rtl"; and per-period ball↔player distances are preserved (the flip
   mirrors all rows together).
3. **ET guard.** Frames containing period 3 with `home_team_start_left_extratime=None`
   → `ValueError`.
4. **Zero-match guard (C1).** `home_team_id` that matches no player row → `ValueError`
   (not a warning).
4b. **Required-schema guard (C5).** Frames missing any required column (e.g. `team_id`)
   → `ValueError` naming the missing column(s).
5. **Equivalence to the native adapter.** A raw situation routed through
   `sportec.convert_to_frames(output_convention="ltr")` vs the same situation as
   absolute frames + `orient_frames_to_ltr` → identical orientation. Proves the helper
   is single-sourced against the established adapter behaviour.
6. **Already-labeled / double-call lock (C2).** A test asserts that calling the helper
   on frames that already carry a non-null `team_attacking_direction` (including the
   output of a first call) **raises `ValueError`** pointing to `play_left_to_right`.
   Locks the loud-guard contract so a future move to silent (re)orientation is a
   conscious decision, not drift.

**Integration (end-to-end, matches the report's ask):**

7. **Defending-GK lands at the attacked goal for both home and away shots.** Build
   absolute frames + a home-team shot and an away-team shot (different periods). Chain
   `orient_frames_to_ltr` → `link_actions_to_frames` → the pre-shot-GK geometry
   aggregator (which applies the ADR-028 reprojection). Assert the defending GK's
   `pre_shot_gk_x` clusters near the attacked goal (x≈100+) for **both** shots — i.e.
   no bimodality. A control assertion on the *un-oriented* absolute frames reproduces
   the bimodality (one shot ~100, one ~5), proving the helper is what closes it.

## Out of scope

- Refactoring the sportec/GS/kloppy adapters to route their internal orientation
  through the new helper (decided: helper-only; the primitives are already shared, and
  refactoring risks coordinate goldens / a retrain for zero functional gain —
  Chesterton's Fence).
- Native `metrica.convert_to_frames` / `skillcorner.convert_to_frames` (option a) —
  duplicates the kloppy gateway, cannot consume the lakehouse's bronze DataFrames, and
  TF-23 already retired the metrica native loader.
- The lakehouse-side adoption (deriving `home_team_start_left` for metrica/skillcorner
  and calling the helper in the bronze builders) — consumer work, tracked lakehouse-side.

## Consumer adoption note (required adoption gates)

The lakehouse calls `_bronze_metrica_to_frames` / `_bronze_skillcorner_to_frames` in
`src/ingestion/tracking_context.py`, both of which set `team_attacking_direction = None`
and apply no flip. After this ships, the lakehouse derives `home_team_start_left` per
provider and appends `orient_frames_to_ltr(frames, home_team_id=…, home_team_start_left=…)`
as the final step of each builder.

Two adoption gates are **required**, not optional:

- **Validate the derived flag (C3).** The helper is necessary but *not sufficient*:
  it is only as correct as `home_team_start_left`. metrica's flag is empirically
  inferred from period-1 shot positions (noisy with few shots; the live data shows
  metrica was inconsistent **per game**, not per period — a wrong inferred flag for a
  single game leaves *that game* bimodal even after the helper ships). Adopters must
  **validate the flag per game** post-orient (e.g. assert each game's defending GK
  lands near the attacked goal — the integration-test #7 recipe applied per game),
  not merely call the helper. skillcorner's flag derivation must be equally validated.
- **Cross-provider AC golden (C7).** The test gap that let this through is idsse-only
  AC goldens. A cross-provider golden — for a shot, the defending GK clusters near the
  attacked goal for **every** tracking provider (sportec, GS, metrica, skillcorner) —
  is a **required adoption gate**, lakehouse-side. silly-kicks' integration test #7 is
  the upstream, provider-agnostic equivalent.
