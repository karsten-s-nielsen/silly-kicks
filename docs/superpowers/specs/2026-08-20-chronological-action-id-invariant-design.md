# Chronological `action_id` invariant across all SPADL converters

**Status:** Draft for review, rev 3 (uncommitted). Surfaced by the lakehouse full-adoption effort (second upstream bug after the 4.86.1 line-break guard). Rev 2 incorporated review-1 H1/H2/M1: the fix is **order-insensitivity of the whole converter** (sort at the TOP, before any positional op), verified by an **input-permutation gate** — not "sort before `_derive_end_coordinates`" + an index-order gate, which left skillcorner/wyscout's shift-derived fields corrupt and invisible. Rev 3 incorporates review-2 R1/R2/R3: content-multiset gate alignment (so synthesized rows are verified), a per-frame parameterized sort key, and the inter-timestamp coverage-boundary framing (the tiebreak is load-bearing).

## Executive summary (for a reviewer)

SPADL `action_id` is documented + consumed as **chronological order** (`_gk_geometry.py:412`), but the invariant is enforced by nobody and the converters are inconsistent about producing it. Worse, **the order-dependence is not confined to `action_id` assignment** — several converters run order-dependent *result/type/geometry* derivations via positional `.shift()` on their raw input order, upstream of any sort:

- `sportec.py:656` / `gradientsports.py:764` assign `action_id = range(len)` over raw event order.
- `base.py:59,74` (`_derive_end_coordinates`/`_add_dribbles`) derive end coords + dribbles via positional `.shift(-1)`.
- **`skillcorner.py:331-339`** derives `same_team_next` (pass result) and `is_short` (short corner/freekick) via positional `.shift(-1)` — **before** its `:536` time-sort.
- **`wyscout` `_fix_wyscout_events`** (`_wyscout_events.py`, `shift(-1/-2/1)` at `:181/182/303/335/378`) does the order-dependent event→action conversion at `wyscout.py:314` — **before** its `:326` id assignment.

On non-chronological input this (1) **crashes** `add_packing`→`secured_reception` and VAEP labelling; (2) **silently corrupts** shift-derived fields — end coords + dribbles everywhere, plus skillcorner's pass results / set-piece classes and wyscout's event types/results; (3) **silently mis-serves** consumers that sort by `action_id` alone. **Verified against v4.88.0 and reproduced on committed data** (the 9-event `idsse_slice` has a raw-order time inversion `…28.81, 26.37`). `retains()` (`_retention_labels.py:62`) already documents that "the mart's action_id order genuinely disagrees with time_seconds" in production.

**The fix (owner-directed: best-practice, long-term, scope/breaking not a constraint):** make **"every converter is order-insensitive — a pure function of chronological event content"** a single, enforced invariant. Sort the working frame chronologically at the **top** of each converter (before any positional op), via one shared seam; **verify order-insensitivity with an input-permutation CI gate** (which catches shift-derived corruption anywhere, unlike an index-order check); guard it at runtime; and harden the `action_id`-alone consumers. Value-shifting (action_id renumber + shift-derived fields) → retrain trigger for the currently-non-chronological providers.

## Root cause (two compounding failures)

1. **No single owner of ordering, inconsistently applied.** kloppy/opta/statsbomb sort `(game_id, period_id, time_seconds)` mergesort before `action_id`; skillcorner sorts `(period_id, time_seconds)` **non-stable, no game_id**; sportec/GS/metrica/wyscout don't sort at all.
2. **Order-dependent derivations run on the un-sorted frame — and not only in `base.py`.** The first draft of this spec assumed `_derive_end_coordinates`/`_add_dribbles` were the only order-dependent ops. They are not: skillcorner (`:331`) and wyscout (`_fix_wyscout_events`) compute *results and types* by positional `.shift()` upstream of their sorts. Any fix that sorts "before `_derive_end_coordinates`" leaves those corrupt — and an index-order gate cannot see it. **The only robust framing is order-insensitivity of the whole converter**, achieved by sorting first and verified empirically, not by enumerating positional ops by hand (that enumeration is exactly what missed skillcorner/wyscout).

## Design

### The invariant (two layers)

> **Output invariant:** every converter's SPADL output has `action_id` as a contiguous `0..n-1` index in stable chronological `(game_id, period_id, time_seconds)` order.
> **Process invariant (the real one):** every converter is **order-insensitive** — permuting the input row order (across timestamps) produces identical output (modulo the `action_id` renumber). This is what guarantees every positional derivation resolved the true time-neighbour.

### 1. One shared sort seam (`base.py`), keyed per frame (R2)

```python
def sort_actions_chronologically(
    frame: pd.DataFrame, *, by: tuple[str, ...] = ("game_id", "period_id", "time_seconds"),
    tiebreak: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Stable chronological sort by (*by, *tiebreak). ONE implementation of the ordering invariant;
    `mergesort` = stable; empty frames pass through."""
```

The invariant is one; the **key columns are a parameter** because the sort runs on different frames at different stages (R2): the SPADL actions frame (default `(game_id, period_id, time_seconds)`) for most converters, but the **raw events frame** for converters that derive results/types by `shift()` before SPADL columns exist — e.g. wyscout, whose raw events carry `period_id` + `milliseconds` (NOT `time_seconds`), and which already sorts `("period_id","milliseconds")` at `_wyscout_events.py:277`. So a converter either passes raw key columns (`by=("period_id","milliseconds")`) or maps raw time → `(period_id, time_seconds)` before sorting; which, and whether wyscout needs any *additional* top-of-frame sort at all, is decided by the gate (§2/§3a), not by prescription.

`tiebreak` orders co-timestamped events deterministically. **The tiebreak is load-bearing correctness, not a mere stabilizer (R3):** the gate (§3a) preserves within-timestamp order and so proves only *inter*-timestamp order-insensitivity — the `shift()` relationships *between co-timestamped events* are never exercised and are trusted entirely to the tiebreak. It must therefore be a **logical intra-timestamp sequence key** (a stable content field — provider event id where it is document-sequential; GS keeps its synthetic-row `__order__` offsets adjacent to their parent), chosen with the same rigor as the sort, **per broken provider** — most urgently for any provider with coarse `time_seconds` granularity, where most `shift`-neighbours are intra-timestamp and thus outside the gate's reach.

### 2. Sort at the TOP of each converter (before any positional op)

The sort must precede **every** `.shift()`-based derivation, not just `action_id`. **Correctness is gate-driven, not audit-driven** — the permutation gate (§3a) is the authority on which converters are order-dependent and where; the list below is the *likely* set from a source read, not a prescription, because hand-enumeration has already been wrong twice (it missed skillcorner/wyscout in rev 1, and rev 2's "sort wyscout before `_fix_wyscout_events`" ignored that `_wyscout_events.py:277` already sorts `("period_id","milliseconds")` — so wyscout may already be order-insensitive, or its pre-`:277` `:181/182` shifts may still need a top-of-frame sort; **only the gate settles it**).

- **sportec, metrica** — likely: sort the raw actions frame after `_build_raw_actions`, before `_derive_end_coordinates`.
- **gradientsports** — likely: sort before `_derive_end_coordinates`, `tiebreak=("__order__",)`; keep the synthesis machinery.
- **skillcorner** — likely: sort the possession frame before the `:331` result/`is_short` shifts (not merely replace the `:536` sort).
- **wyscout** — already sorts `("period_id","milliseconds")` at `_wyscout_events.py:277`; the gate decides whether the pre-`:277` shifts need an earlier top-of-frame sort.
- **kloppy, opta, statsbomb** — route their existing pre-`action_id` sort through the shared helper (behavior-preserving).
- `_add_dribbles`'s re-sort by `(game_id, period_id, action_id)` (`base.py:114`) is consistent once `action_id` is chronological — no change.

For each converter the exact top-of-frame insertion point, key columns, and whether a sort is needed at all are settled by making the gate green — never by enumeration.

### 2b. Gradient Sports null-clock FOUL time — startTime-ordered imputation (MEASURED)

GS ships dedicated FOUL events (`gameEventType=FOUL`, `possessionEventType=FO`) with a **NULL `startGameClock`** — 28/28 fouls across 13/64 real WC2022 matches. `time_seconds` derives from `startGameClock`, so a foul has no native time and cannot participate in a `time_seconds` sort; it must be imputed BEFORE the top-of-frame sort. The imputation must itself be **order-insensitive** or GS remains non-invariant (the very defect this cycle removes) and the permutation gate needs a null-row carve-out.

The imputation basis was **measured** over all 64 real GS WC2022 matches (144,541 events; probe `gs_foul_id_probe.py`):

| basis | inversions vs `startGameClock` (per period) | on null-clock fouls | verdict |
|---|---|---|---|
| `gameEventId` | **31,090 / 144,374 ≈ 21.5%**, drops ≤ ~50 min; 3/28 fouls mis-bracketed | — | ❌ order-insensitive but WRONG |
| native array order | **0 / 144,374** | current ffill correct on real feeds | ✅ correct, ✗ not permutation-proof |
| **`startTime` / `eventTime`** | **0 / 144,374**; present on **28/28** fouls; 27/27 bracketed impute correctly, 0 bad | authoritative | ✅ correct AND stable |
| `sequence` | absent in real GS (synthetic-only) | — | n/a |

**Decision: impute the foul's `time_seconds` by an ffill/bfill ordered by `startTime` (per period), with `eventTime` as the documented fallback.** `startTime` travels with the row, so the imputation is a pure function of content → **fully order-insensitive**; it is chronologically faithful (0 inversions) → **correct**; and because real GS feeds ship in chronological order (native == startTime == clock order), it is **byte-identical to the current native-order ffill on real data → no retrain**. This makes GS uniform with the other converters and **removes any null-row carve-out from the permutation gate** — the GS gate case therefore carries a NaN-time FOUL to exercise the now-invariant path.

**Cost — a one-column input-contract widening.** `EXPECTED_INPUT_COLUMNS` gains `start_time` (float, absolute clock; the converter falls back to `event_time` then, if neither is present, to a native-order ffill with a `warnings.warn`). `_gs_flatten_events` (dev loader) and the **lakehouse GS shaper** (downstream handoff) must supply `start_time`/`event_time` from the raw `startTime`/`eventTime`. Breaking input-contract change (owner: scope/breaking not a concern). This subsection concerns only the FOUL time *imputation*; the GS top-of-frame sort key/tiebreak (§1/§2) is a separate, gate-settled question. `gameEventId` is measured unfit as a *chronology* key (21.5% inverted) — it is used nowhere as one here.

**`start_time` is also the sort TIEBREAK.** An imputed FOUL takes its predecessor's (coarse) game clock, so it is CO-TIMESTAMPED with that predecessor; `time_seconds` alone cannot order the pair, and the adjacent pass's `.shift(-1)`-derived end coord depends on which comes first (gate observed RED on `end_x` before the tiebreak). So GS sorts `(game_id, period_id, time_seconds, start_time)` — `start_time` is the intra-timestamp sequence key R3 requires, and the measurement makes it the *authoritative* one (0 inversions vs the game clock).

**MEASURED byte-identity — GS is NOT a retrain trigger.** An OLD-vs-NEW M-C proof over all 64 real WC2022 matches (`scratchpad/gs_mc_realdata.py`) found the NEW imputation + sort produces output **byte-identical** to the OLD native-order behaviour: `foul_time_diff=0` (start_time-ffill == native ffill on every foul) and `order_diff=0` (native feed is already chronological including within-second). So on real feeds this whole change is a **no-op**; its value is purely the order-insensitivity *invariant* (robustness against a permuted / re-persisted feed), not a correction of currently-wrong output.

### 3. Enforcement

**(a) Input-permutation invariance gate (primary, red-first).** For each converter: run it on its fixture (`out_A`); permute the input row order **across timestamps, preserving within-timestamp order** (so intra-timestamp semantics are untouched — R3/Low 1); run again (`out_B`); **drop `action_id` and assert `out_A` and `out_B` are equal as multisets of full SPADL-field rows** (canonical-content sort both, compare with a float tolerance) — **NOT** aligned on event id. Content alignment is load-bearing (R1): the **synthesized rows carry no provider event id** — dribbles (`_add_dribbles`, `prev+0.1`) and GS cross-goal/foul rows — and those are precisely the rows whose *existence and placement* are order-derived; an event-id join would silently drop them (inner) or collide on NaN keys (outer), leaving the highest-risk rows unverified. Content-equality verifies them too (a dribble that appears or moves under permutation is a real order-dependence) and drops any dependence on a converter preserving a native event id in its output. This proves the *whole* converter — every shift-derived field, not just the index — is order-insensitive, catching skillcorner's `same_team_next`/`is_short` and wyscout's event logic. Combined with §3b (post-fix `action_id` is itself permutation-invariant and chronological), the output is fully pinned. Plus a **meta-assertion** enumerating the `convert_to_actions` surface so a new converter must carry a permutation-gate entry. Lands **red** against the order-dependent converters (observed failing), green as each is sorted-at-top; where a fixture is already chronological the permutation forces the sort to actually fire (non-vacuity).

**(b) Output index-chronology assertion (complementary).** `action_id` order == `(game_id, period_id, time_seconds)` order per group — a cheap direct check of the output invariant. Necessary but *not sufficient* (H2: it can't see shift-derived corruption), so it complements (a), never replaces it.

**(c) Runtime guard — RAISES by default.** `_assert_chronological_action_id(actions)` called from `_finalize_output` (the choke point; all 8 converters pass through, empty frames trivially pass). It **raises by default** (finite-`time_seconds` rows only; NaN times can't be ordered and are not violations). This deliberately deviates from the warn-default `SILLY_KICKS_ASSERT_INVARIANTS` convention (`orientation.py:89`): that convention serves orientation, where a wrong guess is a bounded geometry error; here a violation is a hard downstream crash *or* silent corruption, so failing fast at the converter boundary is correct. There is no legitimate non-chronological `action_id`, and `_finalize_output` has no non-converter callers, so raise-by-default is safe. (`SILLY_KICKS_ASSERT_INVARIANTS` can still escalate the *index-chronology* soft checks elsewhere; the `_finalize_output` guard is unconditional.)

**(d) Harden the `action_id`-alone consumers (M1).** Consumers that sort by `(…, action_id)` alone — `add_restart_coordinates` (`utils.py:822`), `_gk_geometry`, and any others found by auditing for `sort_values([... "action_id"])` — adopt `retains()`'s robust `(game_id, period_id, time_seconds, action_id)` key where `time_seconds` is available. Rationale: the converter guard (c) protects fresh conversions, but consumers reading **persisted marts** bypass it, and `retains()`'s live comment is direct evidence marts have carried non-chronological `action_id`. Defense-in-depth for the one path the converter fix cannot reach. `retains()`'s existing robust sort is left as-is (it already does this).

### 4. Documentation
- **CLAUDE.md** converter-conventions contract: order-insensitivity; sort via `sort_actions_chronologically` at the top; enforced by the permutation gate + the `_finalize_output` raise-guard; consumers sort `(…, time_seconds, action_id)`.
- Docstrings on the helper, the guard, and the gate.

## Impact (CHANGELOG — value-shifting + atomic migration)

Renumbers `action_id` and changes **all shift-derived fields** (end coords + dribbles; skillcorner pass results / set-piece classes; wyscout event types/results) for every provider whose **real feed** is non-chronological. **The gate proves order-DEPENDENCE (on synthetic permuted input); the retrain trigger is real-data output CHANGE, which is MEASURED per-provider, not inferred from the gate.** Measured so far: **sportec/IDSSE is a genuine retrain trigger** (real IDSSE events are non-chronological — the committed `idsse_events_native_golden.parquet` has a raw-order `28.806 → 26.370` inversion); **GS is byte-identical on all 64 real WC2022 matches → NOT a retrain trigger** (§2b M-C: native feed already chronological, start_time-ffill == native ffill). metrica / skillcorner / wyscout: retrain status is settled by the same OLD-vs-NEW M-C on their real feeds during Task 3 (a green gate alone does NOT imply a retrain). Hyrum → retrain trigger only for the measured-non-chronological providers; a pure no-op for the already-chronological ones.

**`action_id` is a cross-table join key, so this is an ATOMIC migration (Low 2).** Renumbering invalidates any persisted data keyed on old ids; a partial re-conversion (some tables old-id, some new-id) misjoins **silently**. The CHANGELOG must state: consumers re-convert **all** affected-provider data atomically — bronze, goldens, and every id-keyed derived table together — never incrementally.

## Gate coverage boundary — inter-timestamp only (R3 / Low 1)
The permutation gate (§3a) preserves within-timestamp order, so it proves **inter-timestamp** order-insensitivity only; the correctness of `shift()` *between co-timestamped events* is never exercised and is trusted entirely to the tiebreak (§1). This is a **property of the guarantee, not merely a per-provider checklist item**: the gate is strongest for fine-`time_seconds`-granularity providers and weakest exactly where co-timestamping is common (there most `shift`-neighbours are intra-timestamp, outside the gate's reach, so the tiebreak becomes load-bearing correctness rather than a stabilizer). Consequences: (1) the tiebreak-key choice gets the same rigor as the sort, per broken provider (§1); (2) verify per broken provider that the native intra-timestamp order is a logical sequence (e.g. DFL EventId is document-sequential), not a scramble; (3) measure each affected provider's `time_seconds` granularity — a coarse clock is where residual risk survives even a green gate.

## Consumer inconsistency — resolved at the source AND hardened
The source fix makes `action_id` chronological, so both the action_id-sorting and time-sorting consumers become correct for fresh conversions; §3d additionally hardens the action_id-alone consumers for the mart path. We do not otherwise rewrite consumer logic.

## Open items (settled during implementation, gate-driven)
- Which of metrica/wyscout/skillcorner are *currently* broken vs already-chronological — the permutation gate's first run answers it and sets the CHANGELOG affected-provider + retrain list.
- The exact top-of-frame sort insertion point + time-field name per converter — chosen by making the permutation gate green.
- Whether every provider has a stable event-id tiebreak; if not, the block-preserving permutation (within-timestamp order held) is the fallback so the gate never penalizes genuinely-ambiguous ties.

## Explicitly not in scope
- No change to `_add_dribbles`'s re-sort (consistent post-fix).
- No atomic-SPADL converter change (derives from corrected standard SPADL).
- No broad consumer rewrite beyond the `action_id`-alone hardening (§3d).

## Testing strategy
- Permutation-invariance gate (§3a): per-converter, red-first against the broken ones, green as sorted-at-top; meta-assertion over the converter surface.
- Index-chronology assertion (§3b) per converter.
- Guard unit tests (§3c): non-chronological → raises; chronological / empty / NaN-time → pass.
- Consumer hardening tests (§3d): the hardened consumers produce correct geometry on a deliberately non-chronological-`action_id` (mart-shaped) input.
- Golden regeneration + diff for every provider the gate flags (confirm the change is only the ordering correction + its shift-derived consequences).
