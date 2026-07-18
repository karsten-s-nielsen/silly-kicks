# ADR-041: real-xT EPV wiring + three per-action orientation repairs

| Field | Value |
|---|---|
| **Date** | 2026-07-18 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen; lakehouse review session (spec rounds 1–2, plan rounds 1–2) |
| **Supersedes / amends** | **Amends ADR-028** (the "self-reconciling families" list was wrong about OBSO); extends ADR-005 (feature surface), ADR-021 (xT), ADR-033 (`add_*` purity) |
| **Source plan** | `docs/superpowers/plans/2026-07-18-xt-epv-wiring-and-tf35-run-valuation.md` |

## Context

The OBSO family (`add_obso`, `add_pausa`, `add_space_creation`) has always multiplied pitch
control by an **EPV** surface, and that surface has always been a synthetic placeholder — a
`linspace(0.01, 0.3)` ramp in x. silly-kicks has shipped a real threat surface since SK-xT-1
(`ExpectedThreat`), and it was wired into VAEP by SK-xT-2, but nothing ever connected it to
the OBSO multiplier. Every `obso_*`, `pausa_*`, and `space_created_m2` value in production
was therefore computed against a demo ramp, and nothing said so.

Wiring a real grid in exposed three latent orientation defects that the synthetic ramp had
been hiding, plus a storage-convention trap in `xthreat` itself.

**The `xT` storage inversion.** `ExpectedThreat.xT` stores rows y-INVERTED (row 0 = the TOP
of the pitch). `rate()` compensates by indexing with the same inversion; `interpolator()`
does not, and hands the caller a y-mirrored surface. Any consumer multiplying raw
`interpolator()` output against an ascending-y pitch-control grid is silently y-mirrored.
This survived because a fitted xT surface is very nearly y-symmetric — the error is real but
small, and no test was asymmetric enough to see it.

**DEFECT A — OBSO never handled orientation at all.** `convert_to_frames` emits
home-attacks-right frames; SPADL actions are per-acting-team LTR. `_precompute_obso_lookup`
passed the raw action-LTR `end_x`/`end_y` as the pitch-control target while sampling
home-attacks-right surfaces, and multiplied by an EPV grid that always increases toward +x
(the HOME team's attacked goal). Away actions were therefore sampled at the reflected point
AND valued toward their own goal. `home_team_id` was accepted by the function and never
read — a dead parameter, which is what ADR-028's classification of OBSO as "self-reconciling"
had rested on. It was not self-reconciling; it simply never handled orientation.

**DEFECT B — `player_influence` multiplied the raw interpolator output.** Same storage
inversion as above, in the one place a real xT grid was already being consumed.

**DEFECT C (found during execution, not in the plan) — the reflection was applied on ONE
axis.** Both repairs above initially flipped the threat/EPV grid as `[:, ::-1]`. ADR-028's
relation is a 180-degree POINT reflection — `x -> 105-x` **and** `y -> 68-y` — so the
correct transform is `[::-1, ::-1]`. An x-only mirror is exact only for a y-symmetric grid,
which the synthetic ramp is and a fitted xT nearly is: the same property that hid the
original inversion also hid the incomplete repair, and the first round of tests (all keyed
on the x axis) passed. Caught by writing a deliberately y-ASYMMETRIC oracle for each site.

## Decision

**1. Fork A — one place neutralizes the inversion.** New `silly_kicks/xthreat/_physical.py`:

- `physical_grid(model, grid_x, grid_y)` returns the xT surface resampled onto a
  CONSUMER-SUPPLIED grid in physical, ascending-y orientation. The `np.flipud` that
  neutralizes the storage inversion happens HERE, once, at xthreat's boundary — not in each
  consumer. Its docstring carries the rule **"Pass the CONSUMER's own grid; do not invent
  one"**, because a helper that invents a grid re-introduces the registration mismatch it
  exists to prevent.
- `values_at_points(model, x, y)` gives exact `rate(use_interpolation=False)` semantics for
  per-point lookups, NaN-tolerant (real provider coordinates carry NaN).
- `require_fitted_xt(model, caller=)` is the SINGLE fitted-model guard. Two divergent copies
  existed (`vaep/features/expected_threat.py` and `atomic/vaep/features.py`); both now
  delegate.

`interpolator()` is left as-is with its docstring corrected to state the storage convention.
Changing its return orientation would be a Hyrum's-Law break for any external caller who has
already compensated.

**2. `xt=` threading with PROVENANCE, not a silent swap.** `add_obso` / `add_pausa` /
`add_space_creation` (and their `*_xfns` factories) accept `xt=`. The resolved surface is
recorded per row in a new `obso_epv_source` column (`"xt"` / `"synthetic"` / `"injected"`),
and serving the placeholder now emits `SyntheticEPVWarning`. Provenance is a COLUMN, not a
DataFrame attr — `attrs` do not survive a merge.

**3. Three public warning categories**, in one module (`tracking/_warnings.py`), so a
consumer's `filterwarnings` line has one stable import path: `SyntheticEPVWarning`,
`IgnoredSurfaceInputsWarning`, `RunValueCoverageWarning`. Deliberately separate, not one
umbrella: silencing the routine synthetic-surface notice must not also silence genuine misuse.

**4. The three orientation repairs**, each RED-verified before the fix:

| Site | Repair |
|---|---|
| `_precompute_obso_lookup` | Point-reflect the action-LTR target into frame coords AND point-reflect the EPV grid (`[::-1, ::-1]`), keyed on `acting_team_attacks_rtl`, not on `home_team_id`. |
| `_player_influence` | Consume `physical_grid(...)` instead of the raw interpolator; away reflection on BOTH axes. |
| `_space_creation` | `np.flip(..., axis=1)` → `axis=(0, 1)`. |
| `_gk_influence` | Same as `_player_influence` (raw interpolator → `physical_grid`; away flip on both axes). Found by in-PR adversarial review, not by the original sweep. |
| `_cover_shadows` | Same again. The review also caught that CLAUDE.md had ALREADY been rewritten to claim this module was repaired — a documentation assertion running ahead of the code. |

**5. Grid registration.** `_resolve_epv_grid` builds the sample grid with NODE registration
(`np.linspace(0, pitch_length, grid_nx)`) to match the OBSO kernel's own indexing, and the
kernel's `floor` index lookup became `round`. Node vs cell-centre registration differ by
±0.505 m at the pitch edges — small, but a systematic bias at exactly the touchlines where
crosses live.

**6. `validate_period_directions` — narrowed to self-contradiction only.** The one
physically impossible frame state is a single team resolving to BOTH `"ltr"` and `"rtl"`
inside one `(game_id, period_id)`. Everything that superficially looks wrong is a legitimate
in-library convention and is ACCEPTED: all-null direction (`output_convention="absolute_frame"`,
the shape `scripts/_loader_pining.py` feeds the training corpora), a uniform label
(`snapshot_to_tracking_frames`, whose frames are ALREADY action-LTR), and period 5 (PSO,
orientation genuinely undefined — `_ORIENTED_PERIODS`).

**Promotion into `acting_team_attacks_rtl` itself is REJECTED ON EVIDENCE, not deferred.**
Even the narrowed rule stays at its single consumer, because all three accepted shapes are
produced by the library itself — so no amount of consumer-side data could establish the
precondition a blanket guard across its 7 call sites would need.

## Consequences

**Downstream value changes (lakehouse re-materialize, batched with the queued 4.49–4.51
triggers):**

- Every AWAY-team `obso_actual` / `obso_peak` / `obso_optimal`, `pausa_*`, and
  `space_created_m2` / `space_denied_m2_opponent` changes (DEFECT A) — this applies **even
  without** `xt=`, because the target reflection is independent of which EPV surface is used.
- All `player_influence_*`, `gk_*` (GK influence) and `cover_shadow_*` values shift (DEFECT B/C).
  The last two were found by adversarial review AFTER the first sweep declared itself done: the
  original grep-style pass keyed on `_player_influence`'s call shape and never asked which OTHER
  modules consumed `xt.interpolator()` directly. The lesson is the sweep, not the fix — enumerate
  every consumer of the raw seam, then fix.
- Home-team rows are byte-identical **only for the away-keyed half** of the repair (the OBSO target reflection + EPV grid flip, which are gated on `acting_team_attacks_rtl`). The **y-inversion half moves EVERY row, home included**: swapping the raw `interpolator()` for `physical_grid` in `player_influence`, `gk_influence` and `cover_shadows`, and `space_creation`'s `axis=1`->`axis=(0, 1)` opponent mirror, are y-MIRRORS, not orientation flips — they are not conditioned on which team is acting. A consumer that re-materializes only away rows will silently keep stale home values for those families.
- New columns: `obso_epv_source`.

**Fixture re-baseline (execution finding, recorded because it is not obvious):** every
synthetic tracking fixture in the repo encoded two teams attacking the SAME way — physically
impossible, and invisible until the guard existed. Four fixtures were corrected
(`test_off_ball_runs.py`, `test_defensive_line.py`, `test_aggregator_column_liveness.py`,
`conftest_id_dtype.py`). The liveness fixture is shared by four auto-enumerating gates, so
correcting it re-baselines all of them at once; the change was verified **value-neutral**
before it was kept. The narrower alternative (guard only the acting team) was considered and
rejected: it would have accepted exactly the fixtures that were wrong.

**A verification lapse worth recording.** The round-1 claim that Fork A was import-cycle-free
was an INFERENCE, not an execution result, and it was wrong twice — `vaep → xthreat` and
`tracking/_player_influence → xthreat` both closed real cycles
(`xthreat/_grid → spadl.config → spadl/__init__ → tracking → ... → xthreat`). Both are now
lazy function-local imports, and the class is permanently gated by
`tests/test_no_import_cycles.py`, which subprocess-imports each public subpackage standalone
— a cycle of this shape is invisible to the ordinary suite, which always imports the package
in a friendly order.

**`_xt_gk._grid_value` was NOT migrated onto `physical_grid`.** Not because it is impossible,
but on cost/benefit: it is pinned to `ExpectedThreat.rate` by a golden test, it reuses
`xthreat._grid._get_cell_indexes` (so it already carries the inversion correctly), and v1
`_xt_gk.py` is frozen pending the lakehouse migration to v2. Migrating it would spend a
golden re-baseline on a module scheduled for removal.

**`physical_grid(...)` / `values_at_points(..., require_fitted=False)` — the orientation
repair must not smuggle in a fail-closed policy.** Sharing `physical_grid` with `_gk_influence` and `_cover_shadows` initially
broke 9 tests, because `require_fitted_xt` raises on an all-zero grid while BOTH modules have a
pinned contract of degrading to NaN there (`test_gk_influence.py::TestXtOrientation::
test_xt_all_zeros_returns_nan`) — and the calibration harness legitimately fits an all-zero grid
from a slim corpus, so this was not a test artifact. The knob relaxes ONLY the all-zero check;
`None` and a variant-name `str` still fail closed, because those are misuse under every contract.
The defect being fixed is ORIENTATION; changing when a module raises is a separate decision and
would have been an unannounced Hyrum break. The knob had to be added to BOTH physical adapters: the
final-review repair of `compute_blocking_score`'s default branch routed through
`values_at_points` and immediately broke the same 8 calibration tests, because the two
adapters disagreed about when they fail closed. They now share one contract, pinned by
`tests/test_xthreat_physical.py::TestValuesAtPointsRequireFittedOptOut`.

**Deferred, carried over from the deleted TODO row: the TRANSITION-factor adapter.** The
On-Deck entry this PR closes asked for a fitted per-source-zone xT row to be adapted onto the
pitch-control grid as the OBSO *transition* factor, and marked that clause "optional, v2". Only the
EPV factor is wired here. The transition factor keeps its synthetic centred-Gaussian default, which
is defensible because it is ball-anchored and orientation-neutral (the reason `_precompute_obso_lookup`
does NOT reflect it), whereas EPV is goal-directed and was the actually-wrong surface. Revisit when a
consumer needs destination-conditioned transition probabilities rather than a distance kernel.

**Fork B (a bundled xT grid variant behind the `str` door reserved in SK-xT-2) is deferred**
— `require_fitted_xt` raises `NotImplementedError` with an explicit message rather than
silently accepting a string and doing something surprising.
