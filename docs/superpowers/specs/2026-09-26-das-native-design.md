# Native Dangerous Accessible Space (DAS) — drop `accessible-space` (design)

**Status:** Proposed (awaiting external review)
**Decision ADRs:** ADR-107 (native DAS engine) and ADR-108 (periodic angular quadrature) — to be written in
commit 1. Amendments: ADR-043 (degradation taxonomy), ADR-106 (the `_das` sentinel is removed), ADR-012
(offside carrier resolved through `id_compat`).
**Version / PR:** assigned at commit-prep only (single-sourced `silly_kicks/_version.py`, ADR-079). Base
`__version__` is `4.127.0`; F1b commit-2 will claim the next number first.
**Branch / base:** `feat/das-native` @ `24ef308` (F1b commit-1 as amended 2026-09-26, ADR-106
float32-storage frames; previously `3aa7435` — the amend touched no DAS file). One rebase onto `main` is
expected after F1b merges.
**Model routing (owner rule):** this spec and the plan may be authored by Opus 5.5. **Implementation must
not run on Opus 5.5**; it runs on the owner's chosen implementation model. Every review is independent and
external, coordinated by the owner.
**Owner decisions recorded (2026-09-26):** D1–D5, Q1, Q2, paired-leg kernel, `[das]` extra deletion, and
the plan-review clarifications SC-1 / SC-2 — see §3.

---

## 0. Executive summary (for reviewers)

silly-kicks' DAS (TF-28) is an adapter over the external `accessible-space` 2.0.15 package. That package
is the slowest code path in the repository (measured **~28 ms/frame** single-threaded), materialises 5-D
`F × P × V0 × PHI × T` temporaries (~570 MB per array per 150-frame chunk, the reason the lakehouse runs
`chunk_size=10`), and carries several correctness defects that this investigation proved by probe:

1. **Non-periodic angular quadrature** — rays 0 and 29 receive half-width wedges, so DAS is not
   mirror-invariant: max |DAS(scene) − DAS(point-reflected scene)| = **441** on synthetic scenes, versus
   **4.2e-12** with periodic weights. The missing wedges straddle φ = 0 (+x), which is the home team's
   attacking direction in canonical frames, so production DAS carries a direction-dependent bias.
2. **Frames keyed by `frame_id` alone** — frames conflate across periods and games (probe: period-2 frame
   0 received period-1 DAS; two games in one call doubled team sums). Repo code records that production
   frame ids restart per game/period (`tracking/_model_eval.py:683`).
3. NaN ball position yields AS/DAS **0.0** instead of NaN; fewer than two defenders make the offside line
   an arbitrary masked value; the passer array is aligned by row order, not frame order.

This design replaces the adapter with a **native engine** that reproduces the published physics exactly
(proved by a frozen-reference golden gate and a corpus parity artifact), fixes the defects above as
documented divergences, and is built for speed:

- a **fused numba kernel** (one streaming pass per frame × angle, O(players × radial points) working
  memory, explicit float64 signatures, serial default and opt-in `prange`),
- a **numpy engine** that follows the reference's floating-point order (reference and fallback),
- **ragged per-frame packing** keyed by `(game_id, period_id, frame_id)` (ADR-105 concat-and-slice),
- **link-restricted computation** for every action-coupled entry point (`das_xfns` / `das_at_action`
  simulate ~2k linked frames instead of ~140k per match),
- a **paired-leg kernel** for counterfactual consumers (gkdv ΔDAS, restdefense), bit-identical to two
  independent calls and ~1.4 legs of work instead of 2 (estimate),
- **direction from `GoalMap`** (ADR-055), the repo's single implementation, instead of a re-implemented
  mean-x inference.

`accessible-space` leaves every runtime and CI path; it survives only in a dev-only pinned extra used to
regenerate the golden fixture and to run the owner's corpus parity driver. The `player_id = "ball"` string
sentinel is removed, unblocking a later value-neutral `player_id → category` migration.

Every DAS value changes (quadrature fix), so this is a downstream Hyrum event: consumer models trained on
`das_xfns` retrain, and the lakehouse re-materialises DAS and gkdv `delta_das` once, together with F1b.

---

## 1. Context

### 1.1 Current path (read end to end)

`add_das` / `das_at_action` / `das_xfns` (`tracking/features.py:2905-3362`) call
`_precompute_das_lookup`, which calls `_das.get_individual_das` (`tracking/_das.py:620`), which calls
`accessible_space.get_individual_dangerous_accessible_space` → `get_dangerous_accessible_space` →
`transform_into_arrays` (a pandas pivot over the union of all players in the call) →
`simulate_passes_chunked` → `integrate_surfaces`. When links are supplied, `_pin_attacking_direction`
infers direction on the full frames via the library's `infer_playing_direction` before the frames are
restricted.

Other consumers: `gkdv/_das_port.py` (direction pin + `get_individual_das`, per leg) → `gkdv._arms.delta_das`
/ `delta_das_batch` → `restdefense/_arms.py`; `positioning.DasObjective` (`get_das` on every optimiser
trial frame); `calibration/_features._compute_das` (`add_das(links=, chunk_size=10)`). `get_xc` also runs
on the library (`get_expected_pass_completion`).

### 1.2 Why now

- Owner-set next cycle after F1b (ADR-106): native DAS, drop the unstable dependency, no effort spared.
- ADR-105 added `estimate_das_cost` / `DasCostWarning` precisely because all-frame DAS costs
  ~394 h/season (ADR-014).
- The `_das.py:318` sentinel `out.loc[ball_mask, "player_id"] = "ball"` is the blocker ADR-106 cited for
  keeping `player_id` off `category`.

---

## 2. Investigation findings (evidence)

All probes ran against `accessible-space` 2.0.15 in the repo `.venv` (py3.10, pandas 2.3.3, numpy 2.2.6)
on synthetic frames. They are recorded here so reviewers can reproduce them; the corpus parity artifact
(§7.2) will re-measure every figure on real data.

### 2.1 The reference model

The model is a radial pass simulation, **not** a Voronoi tessellation. Per frame it simulates 30 pass
angles × 15 ball speeds × 46 radial points, computes a two-point time-to-arrive (TTA) per player, turns it
into interception rates through an efficient sigmoid, integrates a cumulative trapezoid, forms per-player
possibility densities, normalises them, weights them with an xG-style danger surface, and sums polar area
elements. The historical "degenerate Voronoi `IndexError`" narrative (memory, ADR-043 decision 11, the
`_call_simulation` docstring) is wrong: that crash class was pandas StringDtype arrays rejecting 2-D
indexing, which `_prepare_frames` already works around.

### 2.2 Measured baseline

300 synthetic frames, 23 objects/frame, `get_individual_das`: **27.7 ms/frame**; `get_das`: 27.9 ms/frame
(single-threaded, workstation). cProfile: `simulate_passes` dominates; `_approximate_sigmoid` 18%,
`nan_to_num` + `nansum` + `isposinf`/`isneginf` passes ~23%, `gc.collect` (called ~5× per chunk) 7%,
row-wise `DataFrame.apply` for per-player values ~1.3% at this size (grows with rows).

### 2.3 Defects proved by probe

| Case | Reference result |
|---|---|
| NaN ball coordinates in frame 1 | frame 1 AS = **0.0**, team DAS sums **0.0** |
| One defender | computed; offside line taken from an arbitrary masked value |
| Zero defenders | `ValueError` from direction inference ("Did not find exactly 2 teams") |
| Same `frame_id` in periods 1 and 2 | one frame simulated; period 2 received period 1's values |
| Same `frame_id` in games 1 and 2 | team sums **doubled** (0.514 = 2 × 0.257) |
| One player with NaN x | handled (player absent) |
| Ball off pitch | handled |
| Point reflection, reference quadrature | max team-DAS gap **441.3** over 5 scenes |
| Point reflection, periodic quadrature | max team-DAS gap **4.2e-12** |
| Periodic vs reference values | relative shift median **1.63%**, p90 **20.1%**, max **65.4%** (synthetic; corpus figure in §7.2) |

The quadrature cause is visible in `accessible_space/core.py:624-632`: `phi_lower_bounds[:, 0] =
phi_grid[:, 0]` and `phi_upper_bounds[:, -1] = phi_grid[:, -1]` copy the radial end-cell rule onto a
periodic angle grid built with `endpoint=False`. The mirror registry already records the symptom
(`tests/tracking/_mirror_entries/trained_and_das.py:40-41`, `_DAS_MIRROR_TOL = 200.0` against a
measured team-attribution swap of 550.6902).

Code-read defects (no probe needed): `PASSERS` is built by `drop_duplicates(frame_col)` in input row
order (`interface.py:908`) while the simulation uses sorted frame order; team and passer identity use raw
`==` on object-cast arrays; a `team_in_possession` that matches no team in the frame silently computes
with zero attackers.

### 2.4 Optimization audit (optimization-audit skill, Both mode, read-only)

| # | Severity | Evidence | Finding |
|---|---|---|---|
| O1 | Critical | `core.py:245-304` | 5-D temporaries: 150 × 23 × 15 × 30 × 46 × 8 B ≈ 571 MB per array, several alive at once → OOM under the 1 GB `applyInPandas` cap |
| O2 | Critical | `features.py:3178`, `:3337` | `das_at_action` / `das_xfns` simulate every frame (~140k/match) although only action-linked frames (~2k) are read |
| O3 | High | `core.py:242,254,293,296,299` | `gc.collect()` in the hot path (7% measured) |
| O4 | High | `core.py:241,250,258-261` | full-array `nan_to_num` / `nansum` / `where` passes (~23% measured) |
| O5 | High | `interface.py:176-204` | every frame padded to the union of players in the call (substitutes, broadcast visibility churn) |
| O6 | High | `interface.py:1088,1091` | row-wise `apply(axis=1)` for per-player DAS (~16 µs/row → ~48 s at 3M rows) |
| O7 | Medium | `_das.py:245,295-318,561,602,688` | 3–4 full-frame copies per call plus a per-column dtype sweep of every column |
| O8 | Medium | `features.py:3111` | `_map_das_to_actions` uses `iterrows` with per-row dict scans |
| O9 | Medium | `features.py:3078` | `_precompute_das_lookup` builds a Python dict through a groupby loop |
| O10 | Medium | `positioning/_objectives.py:184` | `DasObjective` re-infers direction on every trial frame (can flip mid-optimisation) |
| O11 | Low | `_das.py:447-449` | cost constants (`0.02 s/frame`, warn at 5000 frames) are reference-specific; measured 0.028 (1.4× drift, not stale) but meaningless for a native engine |
| O12 | Medium | repo | zero L1 benchmarks and no memory gate on the DAS path |

---

## 3. Owner decisions (2026-09-26)

| ID | Decision |
|---|---|
| D1 | `get_xc` goes native too (required to drop the dependency) |
| D2 | Reference defects are fixed as documented divergences; direction comes from `GoalMap` |
| D3 | Public surface: typed `params=` replaces `**kwargs`; `use_progress_bar` removed; keyword-only `goal_map=` added; `game_id` + `is_goalkeeper` required |
| D4 | Corpus parity bound = full owner corpus, every velocity-bearing provider, all action-linked frames (no shrinking) |
| D5 | Detection-limited-provider visibility (an observability companion for DAS) is **not** part of this cycle |
| Q1 | Periodic angular quadrature is the shipped default, with its own ADR (ADR-108); the reference quadrature survives only as an internal parity-gate mode |
| Q2 | Re-run `tf19_instrument_responsiveness` and `tf19_signoff_power` in-cycle (owner-run DGX; spells rebuilt in the same session); annotate `tf24_stage2_refresh` with `invalidation.json` |
| P | Include the paired-leg kernel for counterfactual consumers |
| SC-1 | Paired-leg moved rows are derived by bitwise kinematic comparison, not caller-declared (§6.6; ratified after plan review DAS-PLAN-01) |
| SC-2 | The carrier column must be constant within a frame, else `ValueError` (§6.7; ratified after plan review DAS-PLAN-02) |
| F1 | Mirror tolerance is the `arccos`-limited floor `rtol=1e-6`, not the unreachable 1e-9 (§6.4; ratified 2026-09-26 after impl measurement) |
| F2 | Paired-leg ships as two-call `compute_das_paired`; the offside-aware sharing kernel is dropped this cycle (SC-1 share-all-unmoved rationale corrected for the offside cross-dependency) (§6.6; ratified 2026-09-26) |
| X | Delete the `[das]` extra; lakehouse moves to `silly-kicks[numba,…]` |

---

## 4. Goals and success criteria

1. **Parity:** in `reference` quadrature the native engines reproduce `accessible-space` 2.0.15 on a
   frozen golden fixture (numpy: `rtol = atol = 1e-12`; numba: `rtol = atol = 1e-10`) with identical
   finite masks, and on the full owner corpus with zero unexplained finiteness mismatches. Values are
   asserted **finite AND close**, never "did not crash".
2. **Speed (acceptance targets, measured in the corpus artifact):** numba serial ≥ 10× reference median
   ms/frame; numpy ≥ 2×; `prange` parallel efficiency ≥ 0.6 at 16 threads; `add_das` per match (links)
   ≥ 10×; `das_xfns` per match ≥ 50×. A missed target is surfaced to the owner, never silently relaxed.
3. **Memory:** engine working memory independent of the number of frames (tracemalloc gate).
4. **Correctness:** periodic quadrature makes DAS mirror-invariant (mirror tolerance 200 → `rtol=1e-6`, the
   `arccos`-limited floor; F1, §6.4); every divergence has a named test.
5. **Surface:** `get_das`, `get_individual_das`, `get_xc`, `add_das`, `das_at_action`, `das_xfns`,
   `estimate_das_cost`, `DasUnscoreableError`, `DAS_SOURCE_*` keep their names and output columns; the
   changes are exactly those in §6.2.
6. **No runtime dependency** on `accessible-space`; no `player_id` write anywhere in DAS.

---

## 5. Scope

### In scope
- Native engine (numpy + numba), packing, public facade, `get_xc`, the paired-leg kernel.
- Migration of every consumer in §6.13, every affected test and registry, the golden fixture and generator.
- Corpus parity + performance driver and artifact; the Q2 downstream artifacts.
- Docs, ADRs, C4, NOTICE, README, AGENTS.md, packaging and CI changes in §10.

### Out of scope (owner-approved)
- **D5:** an observability/visibility companion for DAS on detection-limited providers. DAS on broadcast
  tracking keeps the reference semantics (unseen players are absent).
- **The `player_id → category` migration.** Unblocked by this cycle (§6.11), noted as a consequence, not
  scheduled here.
- **Lakehouse / consumer changes.** The spec supplies the downstream notice (§11.2); the owner relays it.

### Considered and rejected
| Technique | Reason |
|---|---|
| `fastmath` / reassociation | breaks the parity contract |
| float32 compute | ~1e-4–1e-3 relative error through `b1 = −2000` and the compounding integral; destroys both gates; compute-bound kernel gains ≤ ~1.3–1.8×; F1b already captured the at-rest memory win; ADR-106 rejected it explicitly |
| GPU (numba.cuda / CuPy) | FP64 on the DGX GB10 runs at ~1/64 rate; Databricks serverless has no GPU (ADR-076 D) |
| Cross-call frame-keyed DAS cache | the ADR-043 landmine (a counterfactual frame served its factual twin); negligible value once link restriction and the kernel land |
| Shared-TTA-only paired variant | TTA is ~3% of per-leg work; superseded by the shared-interception-rate paired kernel (§6.6) |
| Rebuilding DAS on spearman pitch control / `compute_tti` | a different model: no parity possible, every consumer retrains for a different metric |
| Bug-for-bug numpy port (approach B) | keeps the defects; ~2–4× only |

---

## 6. Design

### 6.1 Architecture (hexagonal)

New private modules under `silly_kicks/tracking/`:

| Module | Role | Responsibility |
|---|---|---|
| `_das_params.py` | domain constants | frozen `PassSimParams` (every physics constant, grid spec, `quadrature: Literal["periodic", "reference"]`); named profiles `DAS_PARAMS`, `XC_PARAMS` (§6.3); `__post_init__` validation. The `reference` mode is reachable only from the parity harness and tests. |
| `_das_pack.py` | input port (pandas → arrays) | contract validation (§6.7); float64 upcast of `x/y/vx/vy` at the read boundary (ADR-106); frame keying by `(game_id, period_id, frame_id)` via `group_rows` + stable lexsort; ball found by the `is_ball` mask; id → integer codes through `id_compat`; per-row attacking mask and passer flag; per-frame direction; per-frame reason code; the paired-leg contract (§6.6). Output: `PackedFrames` (contiguous arrays + ragged offsets + frame-key table). |
| `_das_engine.py` | numpy engine + dispatch | reference-order numpy engine (port and fallback); `compute_das(packed, params, *, chunk_size, n_threads)`; engine selection (numba when importable, `SILLY_KICKS_DAS_FORCE_NUMPY=1` forces numpy); per-call grid precompute via `functools.lru_cache(maxsize=16)` keyed on the frozen params. |
| `_das_numba.py` | fused `@njit` adapter | serial and `prange` kernels, single and paired variants; explicit float64 signatures; lazy import; `cache=_NUMBA_CACHE` convention. **All** DAS `@njit` code lives in this one file. CI mechanism: `tests/test_ci_shard_wiring.py::test_numba_cache_key_covers_all_njit_files` (`:111-136`) detects every `@njit` file dynamically (AST `_defines_njit` over `silly_kicks/**/*.py`, naming explicitly not relied on for detection) and requires each to be covered by the numba cache-key `hashFiles` patterns of both the `test` job (`ci.yml:91`) and the `slow` job (`ci.yml:184`). Those patterns are `silly_kicks/tracking/**/*_numba*.py` plus `silly_kicks/xtgk/_turnover.py`, so a file named `tracking/_das_numba.py` is covered by the existing key with no key edit; a second DAS `@njit` file under another name would fail the gate until the key is extended. |
| `_das.py` | public facade (output port) | `get_das`, `get_individual_das`, `get_xc`, `estimate_das_cost`, `DAS_SOURCE_*`, `DasUnscoreableError`, the confined private `individual_das_paired` used by gkdv; unpacks `DasResult` into DataFrames. |

`DasResult` (internal): per frame — key, AS and DAS for the in-possession team, reason code; per packed
player row — AS and DAS; a row map back to input positions. Data flow: `frames → pack (once) → engine →
DasResult → unpack`.

### 6.2 Public surface

All new parameters are keyword-only.

```python
get_das(frames, *, goal_map=None, attacking_direction_col=None,
        player_in_possession_col="ball_carrier_player_id", params=None,
        chunk_size=None, n_threads=None, warn_cost=True) -> pd.DataFrame        # + AS, DAS (team, per frame)
get_individual_das(frames, *, goal_map=None, attacking_direction_col=None,
        player_in_possession_col="ball_carrier_player_id", params=None,
        chunk_size=None, n_threads=None, warn_cost=True) -> pd.DataFrame        # + AS, DAS (per player)
get_xc(passes, frames, *, params=None, chunk_size=None, n_threads=None) -> pd.DataFrame   # + xC
add_das(actions, frames, *, links=None, goal_map=None, chunk_size=None,
        attacking_direction_col=None, params=None, n_threads=None) -> pd.DataFrame
das_at_action(actions, frames, *, col_name="das_team", links=None, goal_map=None,
        chunk_size=None, attacking_direction_col=None, params=None, n_threads=None) -> pd.Series
das_xfns                                         # xfn protocol unchanged; links + GoalMap built internally
estimate_das_cost(frames, *, n_threads=None) -> float
```

- `goal_map` and `attacking_direction_col` together → `ValueError`. `attacking_direction_col` is new on
  `get_das` (the reference path hardcoded `infer_attacking_direction=True`); `links` is new on
  `das_at_action`.
- `params=None` means `DAS_PARAMS` (or `XC_PARAMS` for `get_xc`); passing a `PassSimParams` with
  `quadrature="reference"` through the public surface raises `ValueError` (parity-only mode).
- `n_threads=None` selects the serial kernel (Spark and `for_each` already saturate cores; ADR-076 D);
  an integer > 1 selects the `prange` kernel under a scoped `numba.set_num_threads` that restores the prior
  value. Ignored (with no warning) on the numpy engine.
- `chunk_size=None` now means the engine's bounded default (chosen by measurement in the plan), not the
  library's former 150.
- **Removed:** `**kwargs` passthrough, `use_progress_bar`.
- `warn_cost` is added to `get_individual_das` (it existed only on `get_das`); `DasCostWarning` now fires
  on an estimated-seconds budget (§6.14).
- `positioning.DasObjective(*, goal_map=None, player_in_possession_col=None)`.
- `estimate_das_cost` becomes engine-aware.

### 6.3 Physics model and constants

Constants are copied verbatim from `accessible-space` 2.0.15 (`core.py` `_DEFAULT_*`,
`interface.py` `_DEFAULT_*_FOR_DAS`, `utility.py` goal geometry, `interface._get_danger`) with a
provenance citation in `_das_params.py`. The golden `metadata.json` records the library's constants at
generation, and `test_das_params.py` asserts equality (drift detection).

**DAS profile (`DAS_PARAMS`):**

| Constant | Value |
|---|---|
| `n_angles`, `phi_offset` | 30, 0 (angles `linspace(0, 2π, 30, endpoint=False)`) |
| `n_v0`, `v0_min`, `v0_max` | 15, 3, 30 (`linspace`) |
| `radial_gridsize`, `pass_start_location_offset`, `time_offset_ball` | 3, 0, 0 |
| radial grid | `arange(offset, L + offset + Δr, Δr)`, `L = hypot(105, 68) + 3Δr` → 46 points |
| `b0`, `b1` | −4.565680899844368, −2000 |
| `player_velocity`, `inertial_seconds`, `tol_distance` | 9, 0.17, 5 |
| `use_max`, `keep_inertial_velocity`, `use_approx_two_point` | False, True, True |
| `v_max`, `a_max` (unused while `use_max=False`) | 19.85563874348074, 10.659091365334193 |
| `factor`, `factor2` | 5.077423030272923, 1.0063028450754512 |
| `use_efficient_sigmoid`, `normalize` | True, True |
| `respect_offside`, `exclude_passer` | True, False |
| `danger_weight` | 1 |
| danger logistic | intercept −0.52156283; coefficients −0.14447723 (distance), 0.40579492 (opening angle) |
| goal geometry | semi-width 7.32/2 + 0.06 = 3.72; `x_goal = 52.5` (centred frame) |
| pitch | [−52.5, 52.5] × [−34, 34] (centred), inclusive bounds |

**xC profile (`XC_PARAMS`):** `b0` −4.565680899844368, `b1` −188.74468208593532,
`pass_start_location_offset` −1.5245340256423476, `time_offset_ball` −0.4384754490159207,
`radial_gridsize` 5.034759576558597, `player_velocity` 34.6836072667285, `inertial_seconds`
1.1043767821571149, `tol_distance` 9.986761680941445, `use_max` True, `v_max` 19.85563874348074, `a_max`
10.659091365334193, `keep_inertial_velocity` True, `factor`/`factor2` as above, `normalize` False,
`respect_offside` False, `exclude_passer` True, `use_poss` True, fixed v0 grid `linspace(8.886015553615485,
42.18118275402132, round(13.751097117532021) = 14)`, one angle per pass `atan2(end − start)`, ball at the
event start coordinates (tracking ball where the event coordinate is NaN), team in possession = event team.

**Model equations (per frame, angle j, speed k, radial index t):**

- `T[k,t] = (D[t] − D[0]) / v[k] + τ`; `dT[k,t] = T[k,t] − T[k,t−1]`; `DT[k] = T[k,1] − T[k,0]`.
- Ball trajectory: `X = bx + cos φ_j · D`, `Y = by + sin φ_j · D`.
- TTA (approximate two-point, `use_max=False`): `mid = pos + v · t_in`; `TTA = t_in + |mid − target| /
  v_player`; where `|pos − target| < tol`: `TTA = hypot(pos − target) / v_player`. With `use_max=True`
  (xC) the second-segment speed is `min(|v| + a_max · t_in, v_max)`. NaN TTA → +∞; offside attackers,
  and the passer when `exclude_passer`, → +∞.
- Interception rate: `a = σ̃(b0 + b1 · (TTA − T[k,t])) / (factor · v[k] ** (−factor2))`,
  `σ̃(x) = 0.5 · (x / (1 + |x|) + 1)`; NaN → 0.
- `S_att[t]`, `S_def[t]` = sequential sums over players in canonical order; `I[t] = I[t−1] + (dT[k,t] ·
  (S[t] + S[t−1])) / 2.0`; `P0 = exp(−I)`.
- `ρ[p,k,t] = ((P0_opp(p) · a) · DT[k]) / Δr`; `ρ[p,t] = max_k ρ[p,k,t]`.
- DAS profile normalisation: `ρ /= max_{p,t}(ρ · Δr)` per (frame, angle), before pitch clipping.
- Team density `ρ_att[t] = max(0, max over attacking players)`.
- Danger on normalised coordinates `(X · dir, Y · dir)`.
- Weights `w[j,t] = dr[t] · dA[j,t]`, `dA = Δφ_j / (2π) · π (r_hi² − r_lo²)`; radial end cells half-width;
  `periodic`: `Δφ_j = 2π / n` for every ray; `reference`: `Δφ_0 = Δφ_{n−1} = π / n`.
- Off-pitch densities zeroed. `AS = Σ ρ · w`, `DAS = Σ danger^(1/danger_weight) · ρ · w`, for the team
  density and for each player.
- xC: `cum = cummax(ρ_att) · Δr`, held at its last on-pitch value, xC = last finite value, clipped to [0, 1].

### 6.4 Numeric contract

- **float64 everywhere.** `_das_pack` upcasts with `np.asarray(..., dtype=np.float64)`; the kernels carry
  explicit float64 signatures, so a float32 array raises `TypeError` instead of compiling a float32
  specialisation (this makes the ADR-106 rule structural for DAS). No `fastmath`.
- **Expression shapes follow the reference exactly**: division where it divides (never multiplication by a
  reciprocal), the same association order, the same `(d · (y1 + y0)) / 2.0` trapezoid.
- **numpy engine:** vectorised per frame chunk, looping over speeds (the max over speeds is exact), padding
  each chunk to its maximum player count with NaN rows (these contribute exactly +0.0 to sums and 0 to
  maxima), calling `scipy.integrate.cumulative_trapezoid` itself and reducing with the same `np.sum` /
  `np.nansum` / `np.nanmax` calls as the reference. Target versus `accessible-space` in `reference` mode:
  Δ = 0; gate `rtol = atol = 1e-12` with identical finite masks; the measured max |Δ| is recorded.
- **numba engine:** identical formulas and player order. Residual differences come only from the final
  area sum (sequential versus numpy's pairwise summation) and libm `exp` versus numpy's SIMD `exp`. Gate
  versus the numpy engine and versus the golden: `rtol = atol = 1e-10`; measured values recorded.
  Bit-identity between the engines is **not** claimed (the ADR-076 KDE `cpu-numba` precedent).
- **Invariance:** serial vs `prange` byte-identical (frames are independent); `chunk_size ∈ {None, 1, 7,
  n_frames}` byte-identical within each engine; input row order has no effect.
- **Mirror (F1, owner-ratified 2026-09-26 — corrects the 1e-9 in this section and §4/§8).** Point-reflection
  DAS is exact in real arithmetic, but the numerical floor is the danger term's `arccos` opening angle,
  whose derivative → ∞ at the goal mouth, so tiny float differences in its argument amplify to **~1e-7
  relative** (measured max mirror residual 1.4e-7 relative on the golden fixture, scene S10). The spec's
  earlier **1e-9 was aspirational and unreachable**; the mirror gate uses a **RELATIVE** tolerance
  `rtol=1e-6` (≈ the measured floor ×~7), matching the engine mirror test. This is ~5e5× below the ~550
  team-attribution swap the gate exists to catch, so the gate keeps teeth (a discrimination test — an
  interior perturbation above the tol still fails the gate — is retained). The `arccos` floor is INHERENT
  to reference parity and is NOT "fixed": a stable `atan2(|u×v|, u·v)` opening angle would drop the floor
  to ~1e-12 but DIVERGE from accessible-space's `arccos`, breaking the numpy-vs-reference 1e-12 parity gate
  (§6.3 constants are verbatim). `add_das`'s `_DAS_MIRROR_TOL` becomes an rtol; its `tolerance_basis`
  records the measured residual + the `arccos` cause + the corpus it was measured on.

### 6.5 numba kernel and numpy engine structure

numba, per frame (the `prange` axis):

1. Load the frame's ragged player block, ball, direction, passer flag, attacking mask.
2. Offside mask (§6.10), then danger, on-pitch mask and weights for this ball position.
3. For each angle: compute `TTA[P, T]`; for each speed stream along t — compute `a` over players (an
   element-wise, vectorisable inner loop), accumulate `S_att`/`S_def` sequentially, advance both
   trapezoids, take the two `exp`, update the running maximum of `ρ[p, t]`; normalise; accumulate team and
   player AS/DAS.

Working memory is O(P · T) per thread (~2k doubles) plus O(n_angles · T) per frame. Operation count per
frame ≈ P · V · Φ · T ≈ 476k interception updates plus ~41k `exp`; the estimated serial cost is
~1–2 ms/frame (to be measured). The numpy engine processes a chunk of frames at once with bounded
temporaries (`chunk × P_max × Φ × T` float64 ≈ 16 MB per array at 64 frames).

### 6.6 Paired-leg kernel (counterfactual consumers)

For an actual leg and a counterfactual leg that differ only in a set of moved rows (the keeper for gkdv; a
defender set for restdefense), the pair is scored through one call site, `compute_das_paired`, which returns
two `DasResult`s **bit-identical to two independent `compute_das` calls** and offside-correct.

**F2 — the SC-1 share-all-unmoved-rows rationale is CORRECTED (owner-ratified 2026-09-26).** SC-1 originally
claimed an unmoved player's interception rate `a[p,k,t]` is "a pure function of its own TTA → identical
across legs", so a shared kernel could compute it once. That is **false for attackers**: the offside line is
`f(all defenders)` (§6.10), so a moved keeper can flip an unmoved attacker's offside status → its TTA (→∞) →
its rate. Only unmoved **defenders** (which carry no offside mask) are safe to share, so the real saving is
**< ~30%**, and only on the already link-restricted (~2k-frame) gkdv/restdefense path.

**Decision (F2): commit-1 ships the two-call `compute_das_paired`** — correct, validated, offside-correct.
The offside-aware sharing kernel (recompute offside per leg; share only status-matching rows) is **dropped
for this cycle** as an owner-delegated scope call: the saving is < 30% on an already-cheap link-restricted
path, it is ADR-043-landmine-adjacent, and there is no evidence the paired path is a bottleneck (the corpus
artifact §7.2 records paired-vs-independent timing, so this can be revisited with data). If a later cycle
builds it, it is its own gated piece (own commit) with an **offside-flip non-vacuity test** — a scene where
the keeper move flips an unmoved attacker's offside must still be bit-identical — not a vague TODO.

**Moved set — derived, not declared (SC-1, owner-ratified 2026-09-26).** A row is "moved" iff any of its
`x`, `y`, `vx`, `vy` differs bitwise (NaN-aware) between the legs. Every other column (`game_id`,
`period_id`, `frame_id`, `player_id`, `team_id`, `is_ball`, `team_in_possession`, the carrier column) and
the whole ball row must agree exactly on every row, and the legs must list the same rows in the same order;
otherwise packing raises `ValueError`. Sharing is therefore applied only to bit-identical rows, which makes
the ADR-043 landmine (a counterfactual leg served factual values) structurally impossible rather than
guarded, and the public `gkdv.delta_das_batch` signature needs no change.

**Guards:** the `ValueError` contract above (no silent sharing); `array_equal` gate versus two independent
calls; non-vacuity gate (a moved keeper produces a non-zero delta). Consumed by
`gkdv/_das_port.team_das_by_frame`; restdefense inherits it through the public `gkdv.delta_das_batch`.

### 6.7 Input contract and degradation taxonomy

Required columns: `game_id, period_id, frame_id, player_id, team_id, is_ball, x, y, vx, vy,
team_in_possession`, plus `is_goalkeeper` whenever neither `goal_map` nor `attacking_direction_col` is
supplied. Optional: `ball_carrier_player_id` (passer), `speed_source` (velocity marker).

| Condition | Response |
|---|---|
| Missing required column; missing `vx`/`vy` without the all-rows `speed_source = unavailable` marker; a non-default `player_in_possession_col` missing | `ValueError` (existing contract) |
| All rows marked velocity-unavailable | `DasUnscoreableError(das_source="unscoreable_frame")` (existing) |
| `team_in_possession` all-NaN in the scored subset | `DasUnscoreableError(das_source="unscoreable_call")` (existing) |
| Duplicate `(game, period, frame, player)` rows; more than one ball row in a frame; `team_in_possession` varying within a frame; the carrier column varying within a frame (NaN-aware; SC-2, owner-ratified 2026-09-26 — the reference silently took the first row in input order, the D-PASSER defect class) | `ValueError` (new) |
| `goal_map` together with `attacking_direction_col`; direction values outside {+1, −1} | `ValueError` (new) |
| Direction column missing / non-numeric / all-NaN group / partially populated | existing `_validate_per_frame_attacking_direction` errors |
| Invalid `PassSimParams` | `ValueError` |
| Default carrier column absent | no passer exclusion; the existing one-time `UserWarning` |

Per-frame degradation (NaN, no exception, internal reason code): no possession, no ball row, no player
rows, NaN ball position, in-possession team absent from the frame, attacked goal unresolved. When no frame
is scoreable the existing `UserWarning` fires (message rewritten without the library reference) and all
values are NaN.

Removed, because the reason they existed is gone: the `_call_simulation` `IndexError`/`TypeError`
conversion (no library seam), `_check_das_output_alignment` (alignment holds by construction; an internal
invariant test replaces it), the `ImportError` for the `[das]` extra (DAS is core).

`DasUnscoreableError` remains the **only** degradable DAS exception and `DAS_SOURCE_VALUES` keeps its five
tokens — nothing is widened. Mapping for `add_das`: no linked frame → `unlinked`; any per-frame reason code
→ `unscoreable_frame`; acting team absent from the frame's teams → `team_unresolved`; otherwise `computed`.
The reason codes stay internal (ADR-054: a provenance column only where the value changes).

### 6.8 Divergences from the reference, and semantics kept deliberately

Each divergence has a named test in `test_das_divergences.py` that asserts both the native behaviour and
the golden-recorded reference defect, and a per-provider count in the corpus artifact; divergent frames are
excluded from value parity with explicit accounting.

| ID | Reference | Native | Basis |
|---|---|---|---|
| D-QUAD | half wedges on rays 0 and n−1 | periodic quadrature | Q1, ADR-108 |
| D-KEY | frames keyed by `frame_id` only | `(game_id, period_id, frame_id)` | §2.3 probe |
| D-BALLNAN | NaN ball → AS/DAS 0.0 | NaN, `unscoreable_frame` | §2.3 probe |
| D-OFF | < 2 finite defenders → arbitrary offside line | offside not applied (the second-last opponent is undeterminable; attackers kept) | §2.3 probe |
| D-PASSER | passer aligned by row order | aligned by frame key | `interface.py:908` |
| D-DIR | mean-x inference over the call's possession frames | `GoalMap` from full frames (ADR-055), `allow_guess=True` | D2 |
| D-IDS | raw `==` on object-cast ids | `id_compat` codes (ADR-019) | convention |
| D-POSSABSENT | possession team absent → zero attackers, AS = 0 | NaN | code read |
| D-DUP | duplicate rows silently dropped (keep first) | `ValueError` | fail loud |
| D-MULTIBALL | several ball rows misalign the ball array | `ValueError` | fail loud |
| D-DIRVAL | direction values scale the coordinates | `ValueError` | fail loud |
| D-POSSVAR | first row's possession after a team sort | `ValueError` | fail loud |
| D-XC-FRAME | missing pass frame → `ValueError` | per-pass NaN + one aggregated `UserWarning` | consistency with DAS |
| D-XC-TEAM | pass team not in tracking teams → `ValueError` | per-pass NaN + aggregated warning | consistency with DAS |

Kept deliberately (documented, counted in the artifact):

- **NaN velocity:** the player counts only inside the tolerance radius (the constant-velocity branch) and is
  absent beyond it. Rare (track starts); fabricating a velocity would be a modelling decision.
- NaN player position → the player is absent.
- An off-pitch ball is simulated normally.
- The normalisation maximum includes off-pitch points (the reference normalises before clipping).
- `AS = Σ ρ · dr · dA` — the package's convention (effectively a constant ×3 radial factor), kept for
  comparability with published DAS values.
- Team DAS in `add_das` = sum of that team's individual player DAS (PR-S41 rule).

### 6.9 Orientation

Per frame, direction = +1 if `goal_map.attacked_goal(game, period, team_in_possession, allow_guess=True)
== 105`, −1 if it is 0, NaN if it is unresolved (ADR-051: an unresolved direction is a value; direction is
never derived from team identity).

- `add_das`, `das_at_action` and `das_xfns` build the map with `resolve_defended_goals` on the **full**
  frames before link restriction; this replaces `_pin_attacking_direction` entirely.
- `get_das` / `get_individual_das` use a supplied `goal_map`; otherwise they build one from the frames
  passed, and the docstring warns that a subset is a different estimator (ADR-055).
- gkdv builds the map once from the **actual** legs and threads it into both legs (the adapted
  `test_das_arm_passes_ONE_pinned_direction_to_BOTH_legs`).
- A caller-supplied `attacking_direction_col` bypasses the map (existing validation plus the ±1 check).
- The DAS entry points join mirror-registry Gate C (`call_with_map` / `gate_c_must_move`): varying the map
  must move the DAS columns.

### 6.10 Offside (DAS profile)

`norm_x = (x − 52.5) · dir`. The second-last defender is the second-largest finite `norm_x` among the
defending team (goalkeeper included). An attacker is offside iff finite, `norm_x > max(second_last,
ball_norm_x)`, `norm_x > 0`, and it is not the passer (resolved through `id_compat`). Offside attackers get
TTA = +∞ (the reference's "treated like air"). Fewer than two finite defenders → offside not applied (D-OFF).

### 6.11 Sentinel removal

`player_id` is never written. The ball is identified by a truthy `is_ball` mask (the nullable-safe house
idiom). Tests: input frames unmodified (purity), `player_id` dtype and values preserved in every output, and
a frame whose `player_id` is `category` runs clean — the evidence that the value-neutral
`player_id → category` migration (ADR-106 option A's blocker) is unblocked. That migration is a consequence
noted here, not scheduled.

### 6.12 `get_xc`

Passes join frames on `(game_id, period_id, frame_id)`. No direction and no offside (xC runs with
`respect_offside=False` and no danger term). A pass whose frame or team is missing gets NaN with one
aggregated `UserWarning` giving counts by reason (D-XC-FRAME, D-XC-TEAM). A passer absent from its frame
means there is nothing to exclude and the pass is computed. The unused per-pass `v0` estimate the reference
computes under `use_fixed_v0=True` is not computed.

### 6.13 Consumers (every caller of every changed symbol, with evidence)

Library and scripts:

| Consumer | Today | Native |
|---|---|---|
| `tracking/features.py` `add_das`, `das_at_action`, `das_xfns`, `_precompute_das_lookup`, `_map_das_to_actions` | `_pin` + `get_individual_das`; `iterrows`; all-frame simulation for `das_at_action`/`das_xfns` | full-frame `GoalMap` → resolve frame ids positionally first (`_kernels.resolve_frame_ids_by_position`, ADR-020; links computed internally when not supplied; for `das_xfns` the union over the three gamestate slots) → pack only those frames → engine → vectorised keyed merge; `_pin` deleted |
| `gkdv/_das_port.py` `pin_direction`, `team_das`, `team_das_by_frame` → `gkdv/_arms.py` `delta_das`, `delta_das_batch` | `pin_direction` imports `_das._pin_attacking_direction` (defined in `_das.py`; imported at `_das_port.py:39`, the one confined private) + `get_individual_das` per leg | `pin_direction` returns the actual-leg `GoalMap`; `team_das_by_frame` uses the paired kernel; confined private swaps `_pin_attacking_direction` → `individual_das_paired` (count stays 1, ADR-037) |
| `restdefense/_arms.py` | via public `gkdv.delta_das_batch` | unchanged call path |
| `positioning/_objectives.py` `DasObjective` | `get_das` per trial, direction re-inferred | `goal_map=` pin |
| `calibration/_features.py` `_compute_das` | `add_das(links=, chunk_size=10)` | unchanged call |
| `tracking/_run_features.py` | `add_das` family | unchanged call |
| `tracking/_snapshot.py`, `tracking/schema.py:19-26`, `tracking/gradientsports.py:363` | comments reference the sentinel / library | comments rewritten |
| `tracking/__init__.py` | exports | unchanged names |
| `scripts/build_gkdv_arm_values.py`, `scripts/build_tf19_instrument_responsiveness.py`, `scripts/run_signoff_power.py` | ΔDAS via gkdv | unchanged calls; in-cycle DGX re-runs |

Tests referencing changed symbols (all migrated; the plan lists each test's disposition):
`tests/tracking/test_das.py`, `test_das_e2e.py`, `test_das_offside.py`, `test_das_cost_guardrail.py`,
`tests/invariants/test_das_invariants.py`, `tests/tracking/_mirror_entries/trained_and_das.py`,
`_mirror_registry.py`, `test_aggregator_column_liveness.py`, `conftest_id_dtype.py`,
`test_position_only_lift.py`, `test_run_tracking_features.py`, `test_snapshot.py`,
`test_frame_aware_xfns_dup_action_id.py`, `_xfn_default_lists.py`, `test_xt_gk.py` (comment-level),
`tests/test_add_star_purity.py`, `tests/test_enrichment_nan_safety.py`,
`tests/invariants/test_public_id_scalar_registry.py`, `tests/sb360/*` (verdicts unchanged),
`tests/gkdv/test_arms.py`, `test_arms_batch.py`, `test_arm_direction_key.py`, `test_import_allowlist.py`,
`tests/positioning/test_objectives.py`, `tests/calibration/test_features.py`, the `test_import_allowlist.py`
files of duels / gk_decision / match_outcome / shot_stopping / team_metrics / territory / restdefense /
`test_fov_registry_import_allowlist.py`. `tests/test_ci_shard_wiring.py` needs **no** edit (its `@njit`
detection is dynamic, §6.1); the stale file count lives in the `ci.yml:82-83` comment ("ALL 4 numba files:
3 @njit-DECORATED … + 1 CALL-form"), which is rewritten to 5 files naming `_das_numba.py`.

Downstream (lakehouse, read-only check): `src/analytics/action_context/enrich.py:374`
`add_das(out, _frames_tip, links=links, chunk_size=10)`; `src/ingestion/gkdv_writer.py` (ΔDAS arm);
`src/ingestion/action_context.py:2152-2155` (profiler stage-name strings `get_dangerous_accessible_space`,
`add_das`, `get_das`, `simulate_passes`); `src/ingestion/exec_visibility.py:402` (probes
`accessible_space` import); `pyproject.toml:71` `silly-kicks[das,ghost-gk,parse-dfl]>=4.123.0,<5`;
`.github/dependabot.yml:62` (`accessible-space` group).

### 6.14 Performance targets and cost model

Targets in §4.2. `estimate_das_cost(frames, *, n_threads=None)` multiplies the scored frame count by a
per-engine measured constant (numba serial, numba parallel scaled by threads, numpy). `DasCostWarning` fires
when the estimate exceeds a seconds budget (replacing the 5000-frame count). Commit 1 sets the constants
from the local L1 benchmark (cited); commit 2 re-derives them from the corpus artifact.

---

## 7. Validation

### 7.1 CI golden-master gate

- **Generator** `tests/tracking/_fixtures/das_golden/_generate.py` (underscore-prefixed, not collected;
  needs only the `das-reference` extra) builds deterministic synthetic scenes covering: normal play in both
  directions; multi-period and multi-game data with non-colliding ids; active offside including a passer
  beyond the line; ball off the pitch; NaN player position; NaN velocity; ragged per-frame player sets with
  substitutes; possession switches; `team_id`/`player_id` as Int64, object and category; a set of xC passes;
  and dedicated divergence scenes whose reference outputs record each defect.
- The reference runs per `(game, period)`, on float64-upcast inputs, with the same direction supplied
  through `attacking_direction_col`, in reference quadrature.
- **Committed:** inputs, reference DAS/xC outputs, `metadata.json` (accessible-space version and constants,
  numpy / scipy / pandas versions, generator commit), `SHA256SUMS`, `.gitattributes binary`. Serialisation
  is deterministic; the generator reproduces the committed content byte-for-byte under the pinned
  dependencies (ADR-056). The checksum test always runs; the reproduction test runs where the extra is
  installed.
- Gates: see §8 (`test_das_engine_parity.py`, `test_das_divergences.py`, `test_das_quadrature.py`,
  `test_xc_native.py`). Discrimination is mandatory: a 1e-9 relative perturbation of `b1`, or switching to
  `periodic`, must make the parity gates fail.

### 7.2 Corpus parity and performance artifact (owner-run, DGX)

- **Driver** `scripts/validate_das_native_parity.py`: adopts `scripts/_driver.py` `for_each` (sharded,
  resumable, resume-before-load over `list_match_refs`), `require_clean_tree(git_provenance())`,
  `declare_inputs`, registered in `ARTIFACT_DRIVERS` (`tests/scripts/test_provenance_wiring.py`); emits
  aggregates only, never owner-tier rows (ADR-038).
- **Population:** the full owner-tier pining corpus, every velocity-bearing provider; counts queried from
  `_loader_pining._list_matches` at run time and recorded; SB360 excluded as structurally unscoreable
  (count recorded). Scored set: every action-linked frame (the production scoring set).
- **Legs per match:** accessible-space 2.0.15 (per `(game, period)`, same direction); native numpy
  (`reference`); native numba (`reference`); native production (`periodic`).
- **Recorded (per provider, per output — team/player × AS/DAS):** max / p99 / p50 of absolute and relative
  |Δ|; finite counts on both sides; finite-mask mismatches (must be 0 outside the divergence classes); counts
  for every D-* and kept-semantics class, including how many production frames D-KEY corrupts today;
  `GoalMap` direction versus reference inference; the quadrature shift median / p90 / max (the figure the
  ADR and CHANGELOG quote); ms/frame for the reference, numpy, numba serial and numba at
  {1, 2, 4, 8, 16, 20} threads; per-match `add_das` and `das_xfns` wall time on both paths; paired versus
  independent legs; peak RSS; platform and software stamp.
- **Artifact gate** `tests/tracking/test_das_parity_artifact.py` (commit 2): parity bounds, zero
  unexplained finiteness mismatches, the §4.2 targets, `run_tree_dirty == false`, `run_commit` present.
- **Cost:** reference ≈ 1.7M frames × ~0.03 s ≈ 14 CPU-hours, ~1 h wall over 20 shards; native legs
  negligible. Any non-zero count for the new fail-loud `ValueError` conditions on the owner corpus is
  surfaced to the owner before merge.

### 7.3 Downstream artifacts (Q2, in-cycle)

1. Regenerate ΔDAS arm values once (`scripts/build_gkdv_arm_values.py`, native + paired kernel).
2. Rebuild the Layer-2 spells (`scripts/build_layer2_spells.py`, ~8.7 h) in the same DGX session.
3. Re-run `scripts/build_tf19_instrument_responsiveness.py` → `docs/research/tf19_instrument_responsiveness/`.
4. Re-run `scripts/run_signoff_power.py` (both legs) → `docs/research/tf19_signoff_power/`; its
   `invalidation.json` is updated or retired to match the fresh run.
5. `docs/research/tf24_stage2_refresh/invalidation.json`: a sibling annotation classifying each DAS-derived
   field (`das_team` / `das_opponent` / `das_diff` as calibration features, `das_degraded`) by measurement and
   magnitude, citing ADR-107/108, the commit and the corpus shift. The TF-24 Stage-2 decision ("within
   noise, no default change", CHANGELOG PR-S152) is historical, so no re-run.

---

## 8. Tests and CI

New modules (flat under `tests/tracking/`; no new `__init__.py`):

| File | Gates |
|---|---|
| `test_das_params.py` | profiles equal the golden-recorded library constants; `__post_init__` validation from both sides |
| `test_das_pack.py` | contract and fail-loud cases; keys; ragged offsets; **float64 for `x`, `y`, `vx`, `vy`** (the AST gate `test_frame_coord_upcast_gate.py` covers only `x/y/z/*_smoothed`); sentinel removal and purity; Int64 / object / category id axes |
| `test_das_engine_parity.py` | golden gates (numpy 1e-12, numba 1e-10), identical finite masks, a finite-count floor, discrimination |
| `test_das_divergences.py` | every D-* case, native behaviour and golden-recorded defect |
| `test_das_quadrature.py` | weight arrays differ only at rays {0, n−1} by exactly the half wedge; ΔDAS = Σ Δw · g via the engine's per-ray integrand hook within 1e-12; a 1e-9 interior perturbation fails |
| `test_das_invariance.py` | `chunk_size`, `n_threads`, serial vs `prange` byte-identical; row shuffle; concatenated games equal per-game calls |
| `test_das_paired.py` | `array_equal` versus independent legs; non-vacuity; leg-contract violation raises |
| `test_das_scale_memory.py` | ADR-073 `assert_subquadratic_growth` (imported from `tests/_perf_structural.py:81`) with a scoped `rows_scanned_counter`; the module is registered in `tests/_scale_guarded.py::SCALE_GUARDED` (`:19`; `tests/test_scale_guard_registry.py` requires every `group_rows` caller to be listed); tracemalloc working memory independent of N |
| `test_das_kernel.py` | float32 → `TypeError`; lazy kernel binding (importing `silly_kicks.tracking._das` / `_das_engine` does not import `_das_numba`; bound on first use, the ADR-076 `test_ghost_gk_does_not_eagerly_import_numba` idiom); `SILLY_KICKS_DAS_FORCE_NUMPY` |
| `test_xc_native.py` | xC golden parity, D-XC-FRAME, D-XC-TEAM |
| `test_das_benchmark.py` | `pytest-benchmark` ms/frame per engine (local; CI keeps `--benchmark-skip`) |
| `test_das_parity_artifact.py` | commit 2, §7.2 |

Migrated suites: library-specific tests in `test_das.py` (`_to_das_coords`, `_call_simulation`, the
alignment guard, the import guard) are deleted with a per-test reason in the plan; contract tests are kept
and retargeted. `test_das_e2e.py` and `tests/invariants/test_das_invariants.py` lose their `e2e` markers
(the library-crash reason is gone), with before/after run counts quoted from the assertion bodies.
`test_das_offside.py` is ported. Registries: mirror tolerance `rtol=1e-6` (F1, §6.4) and Gate C; SB360 verdicts unchanged;
liveness, purity (`PURITY_ENTRIES`), NaN-safety, id-dtype axes updated.

Both engines on every CI leg: tests parametrise over `{numpy, numba}` through the internal engine selector.
Expensive-but-invariant tests carry `@pytest.mark.slow` (ADR-023). `.test_durations` is regenerated
(ADR-074). Public docstrings carry runnable Examples (Examples gate, `--doctest-modules`).

Local verification before any commit proposal (Shift Left): `python -m ruff check silly_kicks/ tests/
scripts/`; `python -m ruff format --check silly_kicks/ tests/ scripts/`; bare `pyright`;
`python -m pytest tests/ -m "not e2e" -v --tb=short` (with `--benchmark-skip` for tracking); the same suite
on a dedicated pandas-3 venv (py3.13 + pandas 3.x) built by the implementing session; full-suite pass
arithmetic reported.

---

## 9. Dependencies and packaging

- `[das]` extra **deleted**. pip and uv report an unknown extra as a warning (the plan verifies this with a
  real `uv lock` dry run against the lakehouse spec).
- `accessible-space` removed from the `test` extra; new dev-only extra `das-reference =
  ["accessible-space==2.0.15"]`, used only by the golden generator and the parity driver, never installed in
  CI.
- `numba` stays an optional accelerator (`[numba]`); the numpy engine is the fallback.
- `.github/workflows/ci.yml`: `das` removed from the three install lines and its comment block rewritten;
  the numba cache-key comment (`ci.yml:82-83`, "ALL 4 numba files") rewritten to five files including
  `tracking/_das_numba.py` (the key patterns themselves already cover it, §6.1).
- DAS depends only on core dependencies (numpy, pandas, scipy).

---

## 10. Documentation and full artifact set

| Artifact | Change | Commit |
|---|---|---|
| `pyproject.toml`, `.github/workflows/ci.yml` | §9 | 1 |
| ADR-107 (native DAS engine), ADR-108 (periodic quadrature) | new | 1 |
| ADR-043, ADR-106, ADR-012 | amendments (retired conversions; sentinel removed; `id_compat` carrier) | 1 |
| `AGENTS.md` | architecture line, DAS bullet, Dependencies line; `tests/test_agents_md_budget.py` must stay green | 1 |
| `docs/context/tracking-features.md`, `docs/context/conventions-core.md` | TF-28 row, DAS narrative, dependencies | 1 |
| `docs/c4/architecture.{dsl,html}` | remove the `accessibleSpace` system and its three relations; DAS native in `tracking`; regenerate with Graphviz `dot` through the pinned pipeline | 1 |
| `NOTICE` | DAS entries state a native reimplementation of Bischofberger & Baca (2026) and carry the accessible-space MIT copyright and permission notice ("Copyright (c) 2024 Jonas Bischofberger") for the ported algorithm and constants | 1 |
| `README.md`, `silly_kicks/feature_glossary.py` descriptions, `docs/PRIVATE_CONSUMERS.md` | library references; new private modules; lakehouse profiler stage names | 1 |
| golden fixture + generator | new | 1 |
| `docs/research/das_native_parity/`, regenerated `tf19_instrument_responsiveness/`, `tf19_signoff_power/`, `tf24_stage2_refresh/invalidation.json` | DGX-produced / annotated | 2 |
| `estimate_das_cost` constants, `silly_kicks/_version.py`, `CHANGELOG.md`, `TODO.md` | re-derived / bumped / groomed | 2 |

---

## 11. Breaking changes and downstream notice

### 11.1 Breaking changes

1. Every DAS value changes (D-QUAD; the corpus median / p90 / max is quoted in the CHANGELOG), plus every
   D-* frame — including the D-KEY frames that are corrupt today. Column names and the `das_source`
   vocabulary are unchanged.
2. Removed: `**kwargs` passthrough and `use_progress_bar` (`TypeError`); `_pin_attacking_direction` and the
   other `_das` privates (`_prepare_frames`, `_to_das_coords`, `_call_simulation`,
   `_check_das_output_alignment`, `_has_simulatable_frame`, `_frames_with_ball_and_players`,
   `_nan_das_result`, `_import_accessible_space`).
3. New requirements: `game_id` and `is_goalkeeper` columns; new `ValueError` conditions (§6.7).
4. Semantics: `chunk_size=None` is the bounded engine default; `DasCostWarning` fires on a seconds budget;
   `get_xc` degrades per pass instead of raising.
5. The `[das]` extra is gone.

### 11.2 Downstream notice (for the owner to relay to the lakehouse)

- Change `pyproject.toml:71` to `silly-kicks[numba,ghost-gk,parse-dfl]…`; numba is currently a dev-only
  dependency there (`pyproject.toml:534`), and it is what makes DAS fast. Widen the `<5` bound if the
  release is a major version.
- Drop `accessible-space` from the dependabot group (`.github/dependabot.yml:62`); update the
  `exec_visibility` probe and the `AC1_PROFILE` stage-name list.
- Re-materialise the `das_*` columns and the gkdv `delta_das` tables **once, together with the F1b
  float32 change** (one downstream event, not two).
- Retrain any consumer model trained on `das_xfns` features.
- `add_das(links=, chunk_size=10)` keeps working; `chunk_size` can be dropped.

---

## 12. Commit structure and sequencing

Owner workflow: spec (external review) → plan (external review) → execute → `/final-review` → external
implementation review → **human commit gate** → CI green. No commit, push, tag or publish without explicit
per-commit owner approval; no micro-commits.

- **Commit 1 (code):** native engine, packing, facade, `get_xc`, paired kernel, every consumer migration,
  tests, golden fixture and generator, packaging and CI, ADRs, docs, C4, NOTICE, README. Fully green on
  pandas 2 and pandas 3.
- **Owner-run DGX at commit 1** (artifact drivers refuse a dirty tree, `_provenance`): parity artifact, ΔDAS
  arm values, spells rebuild, `tf19_instrument_responsiveness`, `tf19_signoff_power`.
- **Commit 2 (artifacts + release):** the artifacts, `test_das_parity_artifact.py`, the tf24
  `invalidation.json`, re-derived `estimate_das_cost` constants, version, CHANGELOG, TODO.

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The numpy engine cannot reach Δ = 0 | the gate still holds at 1e-12; any exceedance is surfaced, never rebased |
| Speed targets missed | surfaced to the owner with the measured curve |
| New fail-loud conditions fire on real data | counted by the corpus driver before merge; surfaced |
| `GoalMap` direction disagrees with the reference inference | engine parity is measured with the same direction on both sides; disagreement counted separately |
| numba threading-layer contention when `n_threads > 1` inside a multi-threaded host | serial default; documented |
| JIT warm-up on serverless (on-disk cache off) | one-time per process; documented; `estimate_das_cost` excludes it |
| D-KEY shows today's production DAS is corrupt for some providers | quantified in the artifact; stated in the CHANGELOG |
| Golden serialisation not byte-stable across library versions | deterministic format chosen in the plan; reproduction pinned to the recorded versions |
| The F1b rebase | one mechanical rebase after F1b merges (expected by the owner) |
| `AGENTS.md` byte budget | condense the DAS bullet to stay under the gate |

---

## 14. Consequences

- DAS becomes core, dependency-free (numpy / pandas / scipy), with numba as an optional accelerator.
- DAS is mirror-invariant and direction-unbiased for the first time.
- The `player_id → category` migration is unblocked (value-neutral, no retrain, mirrors F1b's `team_id`);
  noted, not scheduled.
- `accessible-space` remains a dev-only oracle pinned at 2.0.15.
- gkdv / restdefense counterfactual workloads cost ~1.4 legs instead of 2 (to be measured).

---

## 15. Numbering and provenance

ADR-107 and ADR-108 are the next free numbers on this base (`docs/superpowers/adrs/` ends at ADR-106).
PR-S number and version are claimed at commit prep. Probe scripts used for §2 lived in the session
scratchpad and are not part of the deliverable; the corpus artifact re-measures every figure.
