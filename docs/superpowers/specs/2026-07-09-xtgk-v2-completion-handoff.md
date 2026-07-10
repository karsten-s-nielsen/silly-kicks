# xT-GK v2 — Completion Handoff: Gate Run + SP2–SP5 (single release)

**Status:** DRAFT handoff for the silly-kicks session. Design is grounded against the real code
(two research passes, ρ + V_opp). NOT committed by the analysis session.
**Date:** 2026-07-09 · **From:** Bounou/xT-GK collaboration session.
**Predecessors (shipped):** SP1 `V(z,p)` = silly-kicks **4.40.0** (PR-S107) + Q3/G8 increment
**4.41.0** (PR-S108). Spec `docs/superpowers/specs/2026-07-05-xtgk-v2-possession-value-design.md`
(rev 4); ADR-036.

## Owner directive & the build-ahead decision (read first)
The owner wants v2 **completed in one release**, not sub-project-by-sub-project. That means
building SP2–SP5 **and** wiring/running the make-or-break gate together, rather than gating SP2–5
on the gate result. **This overrides ADR-036 §8's "PASS ⇒ authorise SP2–5; FAIL ⇒ do not build
V_opp/ρ."** The owner has accepted the tradeoff explicitly: the downside is bounded — V_opp and ρ
are independently valid components, and if `V(z,p)` turns out degenerate the assembled metric
simply *quantifies* that (a publishable finding), with the components reusable. Do not re-litigate
the sequencing; build it.

**Still out of scope / not blocked-on here:** the lakehouse materialization of v2 (separate repo +
handoff); the optional later swap of Jeffrey's xR-GK weights; the optional WC2018/Neuer
reproduction (needs his old data); Eyestone's confirmation of the gate numbers (owner-set below).

## The v2 metric (target)
```
xT-GK(s,a) = ρ(s,a)·[ V(s′) − V(s) ]  −  (1 − ρ(s,a))·[ V(s) + κ·V_opp(s,a) ]
```
- `V` — honest possession-value surface (SP1, shipped).
- `ρ` — calibrated retention probability P(retain | s,a) — **our own, swappable** (SP3).
- `V_opp` — turnover cost = expected opponent threat after a loss (SP2).
- `κ ≥ 1` — only free scalar; default 1, report sensitivity over [1,2].
- `s` = ball at origin zone under pressure p; `s′` = ball at destination.

---

# Part 1 — Gate run: Q4 pre-registration + loader wiring (finishes SP1's owner-run)

The gate machinery exists (`_diagnostics.py::run_deep_zone_gate`) and the owner-run script exists
(`scripts/validate_xtgk_possession_value.py`) but its loader is `NotImplementedError` and its
`GateConfig` is placeholder. Two things finish it:

## 1a. Q4 pre-registration (owner-set 2026-07-09; Eyestone to confirm later — do NOT alter post-fit)
| Knob | Value | Rationale |
|---|---|---|
| Deep-zone cell set | grid columns `xi ∈ {0,1}` (x < 13.1 m) | keeper distribution zone; gate auto-restricts to *occupied* subset |
| `n_min` (support per occupied cell, **all three terciles**) | **30** | now justified because the owner has chosen to **build zone-conditional terciles for real** (below) — the fallback rung is no longer fictional, so the higher support bar is survivable rather than a guaranteed STOP. (If you'd rather keep the bar at 20 with zone-conditional as pure insurance, that's also fine — say so.) |
| `min_occupied_cells` | 2 | need ≥2 usable deep cells or STOP |
| `effect_floor` (absolute) | **0.005 xG** | smallest xG difference treated as real; below plausible noise once means are stabilised |
| `relative_effect_floor` (primary — **B2: bake into GateConfig, gate-enforced**) | **≥ 0.25** | `|v_lo−v_hi| / mean(v_lo,v_hi)` — a quarter of deep possession value lost low→high; scale-free. NOT an owner eyeball — add the field + enforce it in `run_deep_zone_gate` |
| `expected_direction` | **`decreasing`** | pressure suppresses possession value; opposite-sign gradient does NOT auto-pass — escalate |
| `crosscheck_rel_tol` | 0.5 | shipped default |

```python
GateConfig(effect_floor=0.005, relative_effect_floor=0.25, n_min=30,
           min_occupied_cells=2, crosscheck_rel_tol=0.5, expected_direction="decreasing")
```
`relative_effect_floor` is a **new field** — B2 requires adding it to `GateConfig` and enforcing
`|v_lo−v_hi|/mean(v_lo,v_hi) ≥ relative_effect_floor` inside `run_deep_zone_gate` (report it on
`DeepZoneGateReport`), not leaving it as an owner read.

**Fallback ladder (B1 — RESOLVED by building zone-conditional terciles; owner decision 2026-07-09):**
a real three-rung ladder, pre-registered:
1. **Global terciles** (primary) → if the high-pressure deep tercile can't meet `n_min` in ≥2 deep
   cells,
2. **Zone-conditional terciles** (refit) → if still short,
3. **STOP (inconclusive)** — do NOT lower `n_min`.

## 1c. Build: zone-conditional terciles (was a stub — make it real)
`PressureLevels.mode="zone_conditional"` currently exists in name only: `fit()` computes *global*
quantiles and ignores `self.mode`; `apply()` receives no zone/coordinate context. Make it real:
- **Thread zone context** through `PressureLevels.fit(pressure, *, zones=None)` /
  `apply(pressure, *, zones=None)` / `occupancy(...)`, and through the call sites in
  `MarkovPossessionValue.fit` (`_markov.py:61`) and `EmpiricalPossessionValue.fit` (`_empirical.py:79`)
  (which today call `pl.apply(actions[pressure_column])` with no zone context). `zones` = each action's
  band, derived with the **vectorized `_get_flat_indexes(start_x, start_y, l, w)`** (silly-kicks
  refinement 2 — NOT the scalar per-row `zone_of`, which wraps a one-element Series).
- **Zone-band granularity (decide in the plan; recommended default):** the deep-zone columns
  `xi∈{0,1}` as one band vs. the rest — i.e. compute terciles *within the deep band* so the deep
  high-pressure tercile is populated relative to deep-zone pressure, which is exactly the M3 fix.
  Per-cell terciles are too sparse; a small number of bands (e.g. defensive-third / middle /
  attacking-third, or just deep-vs-rest) is the right grain. Record the choice.
- **Global path must stay byte-identical** when `mode="global"` (regression guard — it's what the
  primary rung and all existing SP1 tests use).
- **Persist `mode` + per-band cutpoints — back-compatibly (silly-kicks refinement 1; this is the
  invasive bit §1c waved past).** Today `PressureLevels.cutpoints` is a single `tuple[float,float]`,
  `from_cutpoints` takes one pair, and `MarkovPossessionValue.save` writes `meta["cutpoints"]=list(cut)`
  (a flat pair). The plan MUST: keep the **global on-disk format byte-identical**; add a `mode` field +
  a per-band cutpoint structure used **only** for zone-conditional; and make `load` **back-compatible
  with existing SP1 4.41.0 artifacts** (absent `mode` ⇒ global). A keeper scored under one mode must
  not be silently re-terciled under another.
- **Record which rung fired, and stamp it on the report (silly-kicks refinement 3).** A PASS under
  rung 1 (global) means "value drops across *absolute* pressure"; a PASS under rung 2
  (zone-conditional) means "value drops across *deep-relative* pressure" (arguably the more defensible
  claim, M3). The gate report must carry the rung so the finding isn't over-read.
- **Tests:** `mode="global"` unchanged; `mode="zone_conditional"` gives each band ~⅓/⅓/⅓
  *within-band* (the G6 mode-conditional assertion already anticipated this); a synthetic where the
  deep band is globally low-pressure shows a populated deep high-tercile only under zone-conditional.
- **Open sub-question for the plan/Eyestone:** whether zone-conditional should be the *primary* mode
  for the deep gate (more faithful to "does pressure matter for deep distribution") rather than a
  fallback rung. Default to fallback-rung for this release; flag it.

## 1b. Loader wiring (`scripts/validate_xtgk_possession_value.py`)
Replace the `NotImplementedError` with the real cohort loader (pining / Databricks). Requirements:
- **Cohorts:** WC2022 (`data_source=gradientsports`) = **authorising** verdict; RM
  (`data_source=skillcorner`, `access_tier=restricted`) = **include-as-provisional** second read
  (owner decision 2026-07-09 — do NOT drop it; report its verdict separately, tagged provisional
  because 100% OOD).
- **Actions:** attack-LTR SPADL, per cohort, fit **per cohort** (G3). Run the honesty/gate in
  **both orientations** (fit + `mirror_y` equivariance; `mirror_x` is the rejection check).
- **Reward join:** `soccer_analytics.dev_gold.fct_shot_xg.xg` (calibrated pre-shot) on
  `(match_key, action_id)`; carry `ood_flag`, `xg_ci_low/high` into provenance.
- **Pressure:** `pressure_on_actor__andrienko_oval` (pin per the §5 gradient check); `frame_present`
  = a frame-derived non-null (e.g. `team_shape_n_outfield_players_defending IS NOT NULL`).
- **Pre-fit prep:** `prepare_cohort()` (already written) — the §5/G8 frame-aware null-pressure
  coalesce (frame-present-null → 0/low tercile).
- **Emit (pre-gate):** `ood_rate_by_source`, `frame_present_null_pressure_count`,
  tercile-occupancy, per-deep-cell support — plus the gate verdict, the per-tercile deep levels
  `v_lo/v_mid/v_hi`, and the **relative effect** (so the ≥0.25 acceptance check is in the artifact).
  Write JSON under `docs/research/xtgk_possession_value/`.
- Keep the `_gate_is_locked` guard and `--force-unlocked` dry-run.

---

# Part 2 — SP2: `V_opp` turnover cost

**Key design (high reuse):** `V(z,p)` is team-agnostic (fit pooled across both teams in attack-LTR).
So the opponent's threat after winning the ball at the mirror of zone z is just `V` at the mirror:
```
V_opp(origin_zone, p) ≈ V( mirror_zone(origin_zone), p_opp )
```
`mirror_zone` = the full 180° point reflection (mirror_x ∘ mirror_y) — the same transform v1's
`_counter_value` (`tracking/_xt_gk.py:218-227`) used on the raw grid, now on the **honest** surface.
Intuition check: a keeper losing it in his deep zone (x≈5, his frame) → opponent holds it at
mirror x≈100 (near the keeper's goal) → high `V_opp`. Correct.

## Build
1. **Promote `mirror_zone(zone: int) -> int`** into `silly_kicks/xtgk/_possession_value.py`
   (alongside `zone_of`). Semantics already proven by the test-only `mirror_x`/`mirror_y`
   (`tests/xtgk/conftest.py`); this is the zone-index-level version (row reversal + column reversal
   on the 16×12 flat index).
2. **`TurnoverCost` port** (in `_possession_value.py` or `_turnover.py`):
   ```python
   @runtime_checkable
   class TurnoverCost(Protocol):
       def value(self, zone: int, p: PressureLevel) -> float: ...          # E[opp threat | turnover at zone,p]
       def surface(self, p: PressureLevel) -> npt.NDArray[np.float64]: ...
       def support(self, p: PressureLevel) -> npt.NDArray[np.int_]: ...    # sparsity is load-bearing — expose it
   ```
3. **`MirroredTurnoverCost`** (production adapter): wraps an already-fit `PossessionValue` +
   `mirror_zone` + a **pressure-transfer policy** (default `p_opp = p`; make it a small injectable so
   an empirical policy can replace it). `value(z,p) = V.value(mirror_zone(z), policy(p))`;
   `support` = the mirrored cell's V-support. **Zero new fitting.**
4. **`_is_turnover(actions)`** helper (mirror `vaep/labels.py::_is_owngoal` one-liner):
   `result_id != success` restricted to `_moves.MOVE_TYPE_IDS`. For provenance + the empirical
   cross-check's event population.
5. **`EmpiricalTurnoverValue`** (cross-check, not shipped): structurally clone
   `_empirical.py::_possession_outcomes`'s reverse scan, retargeted to "first shot by the *opposing*
   team in the possession(s) following a turnover," via `add_possessions` team-change boundary.
   Validates the `p_opp = p` mirror assumption. It is *more* sparse than V — apply the
   `_diagnostics._occupied`/support gate before trusting a cell.
6. **Tests:** `mirror_zone` involution (`mirror(mirror(z))==z`) + row/col-reversal against a known
   cell; `MirroredTurnoverCost.value(deep) == V.value(mirror(deep))`; deep-loss ⇒ high cost on a
   synthetic where the mirror zone is high-value; empirical-vs-mirror agreement on a synthetic with a
   known post-turnover chain.

Note: `p_opp = p` is the pragmatic default; a receiver/turnover-context pressure model is a later
refinement (tie it to the receiver-pressure `q` note in Part 4).

---

# Part 3 — SP3: `ρ` retention classifier (our own, xR-GK-swappable)

Mirror `GkCompletionModel` (`tracking/_gk_completion.py`) — logistic, **pure-numpy serve**,
pickle-free JSON+SHA256 weights, per-provider variants — but for **retention**, and calibrated
(ECE-gated) on every shipped variant.

## Build
1. **`RetentionModel` port** (in `_retention.py`):
   ```python
   @runtime_checkable
   class RetentionModel(Protocol):
       def predict_proba(self, features: pd.DataFrame) -> npt.NDArray[np.float64]: ...
   ```
   The metric takes an **injected** `retention: RetentionModel` (same discipline as `compute_xt_gk`'s
   injected `xt=`/`completion=`). Jeffrey's xR-GK later = a second adapter satisfying this port; zero
   metric changes.
2. **Domain filter:** reuse `tracking/_xt_gk.py::_gk_distribution_mask(actions, frames)` verbatim
   (goal-kick ∪ GK-actor pass/throw-in). Do not re-derive.
3. **New label — `retains(actions, *, window_seconds=10.0)`** (the real gap; `scores`/`concedes`
   are goal-gated and must NOT be reused). This is **genuinely new, not a copy** — the `_scores_time`
   searchsorted time-window over `(game_id, period_id)` (`vaep/labels.py:354-501`) is only the
   *boundary* skeleton; the retain/loss payload is new work layered on `add_possessions` team-change
   boundaries. Label = 1 if within the window either (a) the ball is still with the actor's team at
   the window end (no opponent `possession_id` boundary intervenes), OR (b) the actor's team takes a
   shot; label = 0 if the opponent takes over before either. Budget it as a real subtlety (window
   boundary × possession-boundary interaction), with its own thorough tests (mirror the `_is_goal`/
   `_is_owngoal` single-predicate house pattern).
4. **Features — `extract_retention_features`** (mirror `extract_gk_completion_features`): reuse
   `resolve_gk_geometry` (origin/dest + provenance), `pressure_on_actor` (release, optionally
   destination), `receiver_zone_density`/`nearest_defender_distance`, optionally
   `pitch_control_at_target`. Enforce train==serve parity (one shared extractor, resolve-then-mask).
5. **Train — `scripts/train_gk_retention.py`** (mirror `scripts/train_gk_completion.py`):
   group-K-fold by `game_id`, out-of-fold preds, GS (WC2022) `default` variant + a SkillCorner
   variant via the same GS-transfer-or-bundle decision procedure.
6. **Calibrate + gate — stricter than completion:** apply `_ece(...) ≤ 0.10` **and**
   `_reliability_slope(...)` within `±0.25` to **every** shipped variant (completion only gates its
   default on AUC-CI+Brier; ρ must be explicitly calibrated). **H3 — these live in
   `scripts/train_gk_completion.py:50,63` (script-local), so extract them into a shared calibration
   module first** (e.g. `silly_kicks/calibration/_diagnostics.py`) so both SP3 and SP5 (Part 5.3)
   import them — this is real work, not free reuse. Only if plain logistic fails ECE → reach for
   `sklearn.calibration.CalibratedClassifierCV` (new machinery; nothing in-repo uses it).
7. **Ship weights** under `silly_kicks/xtgk/_retention_weights/{default,skillcorner}/`
   (`model.json` + `SHA256SUMS` + `metrics.json` + `MODEL_CARD.md`), byte-for-byte per
   `GkCompletionModel.save()/load()`.
8. **ADR:** amend ADR-036 (or a sibling) documenting the label construct, the ECE gate, and the
   `RetentionModel` port. (ADR-011 explicitly does NOT govern this "trained-light" class per ADR-024.)

---

# Part 4 — SP4: metric assembly + κ + decompositions (+ v1 retirement)

## 4a. The assembler (`_metric.py`)
A `compute_xt_gk_v2(actions, *, possession_value, retention, turnover_cost, kappa=1.0)` that, per
in-scope GK-distribution action:
- builds `State(zone_of(origin), p)` and `State(zone_of(dest), p′)`,
- `ΔV = V(s′) − V(s)` (reuse `PossessionValue.delta_v` for the ΔV **and** its Shapley split),
- `ρ = retention.predict_proba(features)`,
- `V_opp = turnover_cost.value(zone_of(origin), p)`,
- returns `ρ·ΔV − (1−ρ)·[V(s) + κ·V_opp]` plus the decomposition columns.

Hexagonal: the assembler depends only on the three **ports** (`PossessionValue`, `RetentionModel`,
`TurnoverCost`), never on concrete adapters — so V/ρ/V_opp are each swappable.

**Pressure of `s′` / receiver pressure `q`:** the base metric uses `p′ = p` (the action's tercile)
for `s′`. **Consequence:** the Shapley **pressure** component of ΔV (PEV) is ~0 by construction when
`p′ = p`. That is expected — PEV lights up only once *receiver pressure* `q` (Jeffrey's deferred
§11 extension: pressure at the destination) is supplied as `p′`. Wire the decomposition now;
populate DZV/RAV/ΔV; leave **PEV dormant pending `q`**. (If the owner wants PEV live in this
release, `q` is buildable from tracking — pressure on the receiver at the destination frame — as a
bounded add; flag it, don't assume it.)

## 4b. κ
Default `kappa=1.0`; the owner-run/validation emits the headline results across `κ ∈ [1,2]`
(sensitivity curve). No new information needed.

## 4c. Decompositions (B3 — corrected: a coherent additive partition, NOT "PEV/DZV/RAV four bars")
My earlier "RAV = the whole expectation *and* a fourth bar" was incoherent (a term can't equal the
total and be one of the summed bars). It also contradicted Jeff §4, which says verbatim: *"RAV = the
whole expectation. **Don't carry it as a separate term.**"* So **RAV is the total, not a bar.**

The metric expands exactly into **four additive terms that sum to it**:
```
xT-GK = ρ·ΔV_position      (1) probability-weighted positional value-added
      + ρ·ΔV_pressure      (2) = PEV — pressure slice of value-added  [DORMANT until receiver-pressure q; 0 when p′=p]
      − (1−ρ)·V(s)         (3) expected loss of the value you were holding (retention-weighted)
      − (1−ρ)·κ·V_opp      (4) = DZV — turnover cost (opponent threat conceded)
```
- `ΔV_position` / `ΔV_pressure` come from `delta_v(s,s′)`'s Shapley split (`position_component`
  scaled by ρ, `pressure_component` scaled by ρ). Terms (1)+(2)+(3)+(4) = the metric exactly —
  verify this identity in a test.
- **RAV** = the sum = the metric value (a *label for the total*, per Jeff), emitted as the headline,
  not as an independent bar.
- **PEV** = term (2), dormant until `q` (§4a). **DZV** = term (4).
- **H1 — namespace the columns `xt_gk_v2_*`** (`xt_gk_v2_position`, `xt_gk_v2_pev`,
  `xt_gk_v2_retention_loss`, `xt_gk_v2_dzv`, `xt_gk_v2`). v1's `xt_gk_pev/rav/dzv` are FROZEN, still
  materialized by the lakehouse and read by the GK-Analytics UI (Hyrum's Law) — v2 must NOT reuse
  those names with new semantics.
- This four-term partition is faithful to Jeff §4 (PEV = pressure slice; DZV = V_opp cost; RAV =
  whole). Proceed on it; **flag it to Eyestone for confirmation** of the acronym mapping, but do not
  block the release on his reply.

## 4d. v1 retirement (within silly-kicks — do NOT delete yet)
Add the v2 compute path **alongside** v1 `tracking/_xt_gk.py`. Freeze v1 (no changes); do **not**
remove it — the lakehouse still materializes v1 columns and the GK-Analytics UI reads them
(Hyrum's-Law). Removal happens only after the lakehouse migrates to v2 (separate repo/handoff),
≥1 release later. Record this freeze/deprecation end-state in the ADR + CLAUDE.md (the M5 note).

---

# Part 5 — SP5: validation suite (Jeff §6)

An owner-run `scripts/validate_xtgk_v2.py` (+ synthetic-CI units), emitting a JSON/markdown report:
1. **Construct validity:** action-level xT-GK v2 predicts possession→shot / possession-xG /
   goal-in-possession **better than** (a) raw completion, (b) destination-only xT, (c) the v1
   composite. Report out-of-sample R²/log-loss/AUC with CIs. (Reuse `vaep/labels` outcomes +
   `_ece`/bootstrap patterns.)
2. **Cross-competition transfer:** fit V/ρ on WC2022, test on RM; report the performance drop.
3. **ρ calibration:** reliability diagram + ECE (reuse `_ece`, `_reliability_slope`).
4. **Repeatability:** split-half within the RM seasons; season-to-season where possible.
5. **Motivating finding (optional, flag):** reproduce the WC2018/Neuer mis-ranking and show v2
   corrects it — **needs Jeffrey's old data**; leave a stub + TODO, don't block the release on it.
Run WC2022-authorising + RM-provisional, consistent with the gate.

---

# Part 6 — Architecture, sequencing, testing, DoD

## Ports & package shape after this release
`silly_kicks/xtgk/`: three ports — `PossessionValue` (SP1), `RetentionModel` (SP3),
`TurnoverCost` (SP2) — with their adapters, a `_metric.py` assembler depending only on the ports,
plus the validation harness. Injection discipline throughout (mirror `compute_xt_gk`'s `xt=`/
`completion=`). No new xthreat edits; no v1 edits.

## Sequencing within the single release (parallelizable)
Independent, buildable concurrently: **[gate loader-wiring + §1c zone-conditional] ‖ [SP2 V_opp] ‖
[SP3 ρ]** (V_opp reuses the fitted V surface; ρ is its own model; the loader wires the owner-run).
Then **SP4 metric+κ+decomp** (needs the three ports) → **SP5 validation** (needs the metric).
**Single feature / commit / PR (owner decision 2026-07-09)** — matches the repo's one-commit-per-branch
policy; the order above is the internal task sequence, NOT separate PRs. Everything lands in one
commit after `/final-review`. (silly-kicks had leaned stacked-set for reviewability; owner overrode
to single-PR.)

## Testing (TDD / hexagonal / E2E)
- **Synthetic CI** for every unit + the assembler on **stub** ports (a fake V/ρ/V_opp with known
  values → the metric formula is verified in isolation of the estimators). Both orientations where a
  surface is involved. Negative controls where a gate/threshold exists (mirror SP1's discipline).
- **Owner-run real-data** (not CI): the gate (Part 1) and the validation suite (Part 5),
  `@pytest.mark.e2e` self-skipping without data.
- **Regression:** `xthreat` + v1 `tracking/_xt_gk.py` byte-unchanged (guard test).

## Definition of done
- Gate wired + runnable with the Part-1 `GateConfig`; runs WC2022-authorising + RM-provisional,
  both orientations, emitting the full report (OOD, unpressured-restart, occupancy, support,
  verdict, relative effect).
- `V_opp` (MirroredTurnoverCost + empirical cross-check), `ρ` (calibrated, ECE-gated, shipped
  weights + variants), the `compute_xt_gk_v2` assembler, κ-sweep, and the four-term decomposition
  columns per §4c — `xt_gk_v2_position`, `xt_gk_v2_pev` (dormant-pending-`q`), `xt_gk_v2_retention_loss`,
  `xt_gk_v2_dzv`, and the headline `xt_gk_v2` (= RAV, the sum) — all implemented + tested.
- Validation suite runnable (WC2018/Neuer stubbed pending Jeff's data).
- v1 frozen alongside v2 (not removed); no lakehouse changes; ADR-036 amended (or new ADRs for ρ /
  V_opp / metric); version bumped in lockstep; single `/final-review` + release.

## Explicitly deferred / needs external input
- **Q4 Eyestone confirmation** (numbers owner-set now; run proceeds).
- **Receiver-pressure `q`** (Jeff §11) — lights up PEV; bounded add, flag for a follow-on.
- **xR-GK weight swap** (a second `RetentionModel` adapter when Jeff delivers).
- **WC2018/Neuer** motivating repro (his old data).
- **Lakehouse materialization** of v2 columns — separate repo + handoff.
