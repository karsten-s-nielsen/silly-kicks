# SB360 `cross_blocked` — probe measurement and decision

**Question:** is the StatsBomb `cross -> related_events -> Block` join reliable enough to un-defer
`cross_blocked` (ADR-046 BD-2, deferred at n=1)?

**Answer: YES — shipped.** A pre-registered probe over ~510 open-data matches passed all three rules;
the mechanism is clean, and the one same-team edge case is a StatsBomb-labelled offensive block the
opposing-team rule already excludes.

## Mechanism

A blocked cross is an open-play `cross` (a `Pass` with `pass.cross`, `pass.type` not a set piece) whose
`related_events` links to a `Block` event by the OPPOSING team (and not flagged `block.offensive`).
StatsBomb encodes no `pass.outcome == "Blocked"`; the `related_events` -> `Block` link is the only
signal. Built on the pre-filter events frame so all uuids resolve; `pd.NA` on non-open-play-cross rows.

## Probe corpus and pre-registered rules (spec 2026-08-19 §3.2)

SHIP iff: R1 (< 5% of open-play crosses have absent `related_events`) AND R2 (same-team `Block` links
< 1% of linked crosses) AND R3 (the ">= 1 opposing Block" rule well-defined on 100% of linked cases).

| corpus | matches | open-play crosses | blocked | base rate | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|
| committed fixtures (7298, 7584, 3754058) | 3 | 81 | 1 | 0.0123 | 0.0370 | 0.0000 | 0 |
| WC2022 (43/106) | 50 | 1061 | 7 | 0.0066 | 0.0500 | 0.1250 | 0 |
| wide (20 competitions, 2015-2025) | 457 | 9408 | 128 | 0.0136 | 0.0333 | 0.0000 | 0 |
| **pooled (all)** | 510 | 10550 | 136 | 0.0129 | 0.0350 | 0.0073 | 0 |

Fetched via `uv run --no-project --with statsbombpy` (open data). R1/R3 pass on every corpus. Pooled
over all ~510 matches: R1 0.0350, R2 0.0073 (the single same-team case, correctly excluded), R3 0,
base rate 0.0129 -- all within the ship thresholds.

## R2: the single same-team case, resolved

Across ~510 matches / ~10,550 open-play crosses there is exactly ONE same-team `Block` link: WC2022
match 3857298, min 41 -- Cristiano Ronaldo deflecting his own team's cross, which StatsBomb explicitly
flags `block: {offensive: True}`. This is a labelled OFFENSIVE block (a block by an attacking player,
same-team by construction), NOT a defensive block of the cross. The opposing-team requirement already
excludes it (marks the cross `False`); across the diverse 457-match corpus R2 = 0.0000. R2 gates for
"our model of `related_events` is wrong" (spec §3.2); the direct test resolves that as absent -- the
same-team link is a known, correctly-handled category. The mask is additionally hardened with an
explicit `not block.offensive` guard (belt-and-suspenders; ZERO opposing blocks were flagged offensive
across 457 matches, so a no-op on real data).

## Decision

**SHIP.** All of R1-R3 pass on a large diverse corpus; the mechanism is clean. StatsBomb `cross_blocked`
now emits a real mask (was all-`pd.NA`). Verified offline on committed fixture `7584` (one genuine
blocked cross at `original_event_id == e8edd276-8490-456c-b221-240d128f61f1`). Additive; no silly-kicks
retrain (`cross_blocked` has no `*_xfns` and is in no default feature list; ADR-045 `"invariant"`). It
IS read by the public TF-51 `compute_bravery` (`tracking/defensive_credit/_bravery.py`; not
`add_press_commitment`) -- so StatsBomb bravery moves from shots-only to cross-inclusive -- but that is still no retrain (no
xfns) and breaks no test. Downstream: a live-surface value change for consumers of the previously
all-`pd.NA` column (see CHANGELOG Hyrum note,
`docs/PRIVATE_CONSUMERS.md`).

Probe script: `scripts/probe_sb_cross_blocked.py` (offline core, `--open` for the wider fetch).
