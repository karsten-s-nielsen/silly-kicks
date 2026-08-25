# SB360 coverage audit

**Question:** have we utilised StatsBomb 360 everywhere it can be utilised, or are there
metrics built for full tracking that could be extended to freeze-frames?

**Motivation:** a possible commercial SB360 collaboration with an NWSL team, scoped to
goalkeeper metrics.

**Method:** every claim below was produced by *executing* the library against a paired
fixture, not by reading it. Spec: `docs/superpowers/specs/2026-08-04-sb360-coverage-audit-design.md`.
Matrix: [`behaviour_matrix.md`](behaviour_matrix.md).

---

## Headline

**SB360 headroom is substantially larger than expected.** Of 486 verdicts across 34 entry
points, **299 are `works`** — the column produces identical values with or without velocity —
and only **4 are `silent_degrade`** -- all `add_ghost_gk`, and **since REPAIRED** (see below), so
immediately after the repair the regenerated `behaviour_matrix.md` reported **489 verdicts and ZERO
`silent_degrade`** (the live matrix has since grown as aggregators were added, and still carries ZERO
`silent_degrade`). The figures in this report are the measurement AS TAKEN at 4.75.0 and are left
standing: rewriting them would erase the finding that motivated the repair.

These work on freeze-frames today, unchanged:

`add_team_shape` (20 columns) · `add_xt_gk` (16) · `add_pre_shot_gk_position` (5) ·
`add_pre_shot_gk_angle` (3) · `add_gk_completion` · `add_defensive_line` · `add_shape_graph` ·
`add_structural_pass` · `add_packing` · `add_line_break` · `add_pressure_on_actor` ·
`add_defensive_credit` · `add_off_ball_run_values` · `add_sync_score` ·
`spadl.add_restart_coordinates`

## The GK picture

Two questions, and they have different answers:

| Entry point | On a freeze-frame | With the keeper **not in frame** |
|---|---|---|
| `add_xt_gk` (16 cols) | works | **works** |
| `add_gk_completion` | works | **works** |
| `add_pre_shot_gk_position` (5) | works | not exercised |
| `add_pre_shot_gk_angle` (3) | works | not exercised |
| `add_gk_influence` (4) | declines cleanly | not exercised |
| `add_ghost_gk` (2) | **fabricates** | not exercised |

**xT-GK v1 — the library's most GK-specific valuation — works on freeze-frames AND does not
require the keeper to be visible**, because it values distribution through resolved
origin/destination geometry rather than the keeper's tracked position. Same for GK completion.

The *pre-shot* GK features do need the keeper in frame. For SB360 that means in the broadcast
camera's view, and **how often that holds is not answerable from code** — it is a property of
the delivered data. That is Layer B (`scripts/build_sb360_coverage.py`).

## Layer B — real open data

**Superseded by the full pass: [`coverage.md`](coverage.md)** (22 matches, 3 excluded and
counted, commit `650ed07`, clean tree). Headline from it: shots and saves carry a freeze-frame
~98% of the time with the relevant keeper visible 92-100%; goal kicks carry one **32.6%** of the
time (per-match median 21%, IQR 18-50%), and when one exists the kicking keeper is in it 96%.
The binding constraint is frame AVAILABILITY for goal kicks, not keeper visibility and not the
library.

The 9-match spot check below is kept for the record; its 23.3% goal-kick figure was low, as its
own caveat predicted.

### Spot check (superseded)

⚠ **Spot measurement only: 9 matches, 3 per cell. Not the full Layer B pass, and not produced by
a provenanced driver run.** Per-match variance is large, so the per-cell contrasts below are not
reliable; the aggregate has the firmer denominator.

Reproducible because the inputs are named — a preliminary number with no way to check it is the
shape this repo's provenance rule exists to prevent:

| Cell | Match IDs | Goal kicks | With a frame |
|---|---|---|---|
| MLS 2023 (44/107) | 3877060, 3877072, 3877090 | 39 | 6 (15%) |
| WWC 2023 (72/107) | 3893787, 3893788, 3893789 | 52 | 20 (38%) |
| WC 2022 (43/106) | 3857254, 3857255, 3857256 | 38 | 4 (11%) |
| **Total** | | **129** | **30 (23.3%)** |

Per-match goal-kick frame existence ranged 0%–50%; matches 3857255 and 3857256 had **zero**.
The full Layer B pass supersedes this table.

**The per-cell rates above carry no causal claim** (ADR-053, spec amendment). The unit of
analysis is the MATCH, not the event: goal kicks within a match share one broadcast, one
camera rig, one crew, so 3 matches is an effective n of ~3 regardless of how many goal kicks
they contain. A credible sex- or tier-contrast needs ~15-20 matches per cell. The deliverable
is the aggregate and its DISPERSION -- and the dispersion is the finding a club actually
needs: coverage that ranges 0-50% by match cannot be planned around as a point estimate.

Two different quantities matter, and only reporting one describes part of the domain as if it
were all of it:

| SPADL type | actions | with a frame | **frame existence** | defending GK visible | acting-side GK visible |
|---|---|---|---|---|---|
| `shot` | 17 | 17 | **100%** | 88.2% | — |
| `keeper_save` | 5 | 5 | **100%** | — | **100%** |
| `shot_freekick` | 2 | 2 | 100% | 100% | — |
| `cross` | 22 | 19 | 86.4% | 73.7% | — |
| `goalkick` | 12 | 1 | **8.3%** | — | 100% |

*(one MLS 2023 match; "—" marks the rate that is definitionally zero for that action type)*

**The shape of the answer: SB360 is strong for shot-facing and save GK work, and weak for
goal-kick distribution.** Shots and saves carry a frame essentially always, with the relevant
keeper visible ~90-100% of the time. Goal kicks — xT-GK's core distribution domain — carry a
frame far less often: **30 of 129 (23.3%)** across the 9 matches, ranging **0% to 50%** per
match, with two WC2022 matches at zero.

So for the xT-GK surface the binding constraint is neither the code (Layer A: it works) nor
keeper visibility (it does not need the keeper in frame) but **whether a freeze-frame exists at
all for the event being valued**.

A prior that did **not** survive contact: this spec argued a World Cup would be the optimistic
upper bound on coverage, being the most lavishly filmed. On goal-kick frame existence WC2022 is
the *worst* of the three cells (11%) and the Women's World Cup the best (38%). At 3 matches per
cell that ordering is not trustworthy — but the tier hypothesis is certainly not supported.

## Two anomalies investigated — both real data, neither a bug

Traced end-to-end (SB events → converter → SPADL actions → `event_uuid` join → frames) on MLS
2023 match 3877060: 3,540 events, 3,175 frames, 2,022 SPADL actions.

1. **"Only one goal kick per match."** StatsBomb ships **12** and provides a 360 frame for **1**.
   The converter handled all 12; the frames do not exist upstream.
2. **"44% of frames are unmapped."** Correct converter behaviour: `Ball Receipt*` (848) and
   `Pressure` (330) are **85%** of them, and both are legitimately non-actions in SPADL — a ball
   receipt is the passive half of a pass. Effective yield for SPADL-action analysis is ~56% of
   delivered frames.

The method generalises: to separate "real data" from "our bug", count the same quantity at every
stage of the chain and find where it drops. Here it never dropped inside our code.

## The one fabrication -- REPAIRED

**Status: fixed.** The ghost path now REFUSES rather than serving an imputed feature vector: the
guard sits at the shared serving seam `_serve_positions_core`, so `add_ghost_gk`,
`compute_ghost_gk` and `serve_ghost_gk_positions` all inherit it. `ghost_gk_x`/`ghost_gk_y`
re-derive to `all_nan` -> `honest_nan` BY RULE (the machine observation changed; the adjudication
followed), and a new `ghost_gk_source` column reports which path produced the value.

The rest of this section is the original finding, retained because it is the reasoning that
produced the fix -- in particular the mechanism, which determined what the fix had to be.

`add_ghost_gk` (`ghost_gk_x`, `ghost_gk_y`) WAS the only column adjudicated `silent_degrade`. A
fitted model silently imputes the five velocity features it was trained on
(`ball_vx`/`ball_vy`/`ball_speed`/`defensive_line_speed`/`defending_centroid_vx`). The output is a
plausible coordinate with no basis, indistinguishable downstream from a velocity-informed
prediction.

The imputation is **not** a zero-fill, and the distinction matters for the fix.
`extract_ghost_gk_features` yields NaN; `predict_mean`'s HGBR reconstruction then routes NaN down
each split's *learned missing-value direction* — a policy fitted where NaN meant an occasional
dropped measurement, applied here where 5 of 26 features are absent on 100% of rows. Measured:
`NaN → [6.795, 33.522]` versus `zero-fill → [6.888, 33.362]`. So the defect is not
out-of-distribution *values* but an out-of-regime imputation policy, and "fill the zeros
correctly" would not address it — refusing on the `speed_source` marker would.

Everything else that differs, differs *coherently*: pitch control evaluated at zero velocity is
a well-defined **positional** model, and a feature needing a temporal window legitimately gives
a single-sample answer on a single frame. Those are `differs_by_design`, and the distinction was
established by isolating cause — Leg A vs an anchor-only leg separates **velocity** from
**frame count**, and without that separation a temporal-window requirement is indistinguishable
from fabricated kinematics.

## Two claims retracted by execution

1. **`add_gk_influence` does not silently degrade.** It was the audit's motivating example
   through five review rounds: it never consults the velocity marker and zero-fills `vx`/`vy`.
   The zero-fill is real and *is* reached (`features.py:5219-5223`) — and the output is still
   NaN. **A reachable code path is not evidence about the value it produces.**

2. **A measurement taken under a different warning policy than the gate is not a measurement.**
   The transcription tool ran under `simplefilter("ignore")` while pytest escalates three
   warning classes, so `add_obso` was recorded `differs` and observed `raises_a`. Root cause:
   three aggregators take a keyword-only `xt=None` and fall back to a synthetic EPV surface.

## Library findings, recorded not fixed

The audit reports; it does not repair.

1. **`snapshot_to_tracking_frames`' id dtype is pandas-version-dependent.** With `Int64`
   snapshot ids the frames come back `Int64` on pandas 2.3.3 and **`Float64`** on 3.0.3 — the
   `FutureWarning` at `_snapshot.py:172` (concat with all-NA entries) materialising. Hyrum's
   law for any consumer pinning it.
2. **Eleven aggregators need bespoke call shapes**, which a first-class `providers/statsbomb`
   producer would have to handle: six require a fitted `ExpectedThreat`; `add_defensive_credit`
   needs an `xg_column` silly-kicks does not ship; `add_pre_shot_gk_angle` takes `frames`
   keyword-only while its sibling takes it positionally; `add_sync_score` takes no frames at all.

## Limitations

- **Synthetic fixture.** It establishes what the code does, not what real freeze-frames contain.
  Layer B answers the second.
- **26 `not_exercised` verdicts** (a locked budget) — columns the fixture does not reach on some
  axis. Most are GK features on the deliberately keeper-ablated roster, where the collapse *is*
  the finding.
- **Four boundary entry points are not audited**, each enumerated with its reason in
  `tests/sb360/test_registry_surface.py::UNAUDITABLE_BOUNDARY`. `xtgk.compute_xt_gk_v2` needs an
  xG-calibrated port and silly-kicks ships no xG model, so any port supplied would audit the
  stub rather than the library.
- **Orientation is not exercised.** Both legs use one convention deliberately, so the ADR-028
  re-projection is a no-op on each; that surface belongs to
  `tests/tracking/test_mirror_registry.py`.
