# TF-60 — Rest-Defense Structure Metrics + GK Integration — Design

- **Status:** APPROVED (owner + independent review, 2 rounds each) — PR1 (Layer 1) implemented; the
  `rd_width` / `rd_depth` shape semantics were ratified as "Option B" by the owner on 2026-08-30 after
  the `/review-impl` gate, and §7.1 is corrected to match. PR2 (Layer 2) shipped (4.103.0). **The arc
  was reshaped 2026-08-30 by the §9 finding (see §17):** PR3 is now the rest-defense GK-ghost re-fit, and
  the Layer-3 arms + ghost-outfield each shift down by one. Inline PR-numbers in §3–§16 use the **new**
  numbering.
- **Date:** 2026-08-30
- **Feature:** TF-60 (On-Deck; top-ranked of the 2026-08 research batch)
- **Decision record:** ADR-080 (written)
- **New package:** `silly_kicks/restdefense/` (a new C4 container) + `silly_kicks/tracking/_ghost_outfield.py` (a new trained-model primitive in the existing `tracking` container)
- **Depends on:** TF-5 (`derive_team_in_possession`), TF-7 (pitch control + `control_in_region` + `compute_threat_pc`), TF-13 (`defending_gk_from_frames`, `resolve_defended_goals`/`GoalMap`), TF-14 (`compute_defensive_line`, `select_back_line_players`), TF-15 (GK influence / reachable area), TF-18 (`GhostGkModel`, `serve_ghost_gk_positions`), TF-28 (`get_individual_das`), TF-31/44 (`compute_team_shape`), TF-49 (`compute_packing_metrics` counting idiom). Reuses `gkdv` public delta seams. ADR-005 (attribution), ADR-011/016/040/044/050 (trained-model discipline), ADR-019 (id_compat), ADR-028/051-D3/055 (orientation + `GoalMap`), ADR-053/054 (SB360 audit + velocity provenance), ADR-062/063/067/077/078 (SB360 FOV / velocity tiers / velocity-keyed variants / keeper identity), ADR-068/073 (no-rescan + sub-quadratic guard), ADR-076 (numba leaf traversal, if the ghost-outfield model reuses the leaf walk).

---

## 1. Executive summary

Rest defense (German *Restverteidigung* / *Absicherung*) is the defensive rearguard structure an
**in-possession** team maintains — while it attacks — to prevent the opponent's counterattack after a
ball loss. TF-60 ships a **descriptive** rest-defense metric family plus a **counterfactual deterrent**
layer, and folds the **goalkeeper** into the rearguard as a first-class agent — a coupling that, per a
targeted literature survey, **no publication currently has**.

The family is three layers, sampled at the **on-ball action grid** (so it works on both continuous
tracking *and* StatsBomb-360 freeze-frames), reduced to per-`(team, possession)` and per-`(team, match)`
rollups plus a distinguished **moment-of-loss** snapshot:

1. **Layer 1 — structure KPIs** (validated anchors): numerical superiority behind the ball (the coaches'
   "+1", with a GK-inclusive variant), rest-defense zone occupancy, rearguard shape (compactness / depth
   / width / relative line height), and GK line-height + GK-to-defensive-line distance.
2. **Layer 2 — danger-behind-the-line valuation**: opponent-controlled dangerous space behind the last
   line (threat-weighted pitch control, with the in-possession keeper included as a `lambda_gk` control
   agent), and attacker space-control in the rest-defense area.
3. **Layer 3 — counterfactual deterrent arms** (the novel valuation): ghost the in-possession team's
   keeper (and, in PR6, its deepest field defenders) to a league-average baseline and price how much the
   *actual* positioning suppresses the opponent's counter-danger. Attacker-value units, so **negative =
   deterrent**, matching the gkdv sign convention.

It ships as a new `restdefense/` package (mirroring `gkdv/` and `xtgk/`: hexagonal, pure functions,
tracking-public-seams-only) reusing `gkdv`'s generic differencing seams, plus one new trained-model
primitive `tracking/_ghost_outfield.py` (a league-average outfield positioning model, mirroring
`GhostGkModel`). It is **additive** — no existing feature changes, **no VAEP retrain** — and it explicitly
ships the *descriptive* KPIs, **not** any of the source papers' weak success-prediction models
(Forcher et al. rest-defense AUC ≈ 0.60; the xBG/EPV transition classifiers).

The honest ceiling: on SB360, the deep rearguard is exactly the region a ball-centred broadcast FOV crops
out when the ball is advanced (the moment rest defense matters most), so SB360 coverage is real but
frequently partial. Every region/count metric therefore carries the ADR-077 FOV-observability companions,
and those columns are load-bearing — a count at `observed_fraction = 0.4` is a *lower bound*, never a
measurement.

---

## 2. Motivation and the gap

### 2.1 What silly-kicks has, and the one thing it lacks

silly-kicks already owns every *ingredient* of rest defense — pitch control (TF-7), DAS (TF-28), team
shape (TF-31/44), defensive line (TF-14), team-in-possession (TF-5), packing (TF-49), and a deep GK stack
(TF-13/15/18, xt-GK, GKDV). What it lacks is a **team-level rest-defense structure metric**. The only
place the codebase frames anything as "rest defence" today is a single per-action feature
(`space_denied_m2_opponent`, TF-41) — there is no structural quantity, and **no public / emitted
rest-defense counting primitive exists**. Prior art exists at the private feature-extractor level:
`GhostGkModel`'s `defenders_behind_ball` (`silly_kicks/tracking/_ghost_gk.py:733`) counts goal-side
defenders via `to_gr_x(x) < ball_x`. The new primitive reuses that *semantics* (§8), but the grep for
`goalside` / `numerical_superiority` / `spare_man` / `restdefen` returns nothing — there is no reusable
public counting surface.

### 2.2 The literature (beyond the two papers on the TODO row)

A survey turned up several papers that sharpen or extend the anchor:

- **Forcher, L., et al. (2023), "The Success Factors of Rest Defense in Soccer" (JSSM; PMC10690503 /
  PubMed 38045740; Bundesliga 2020/21, 153 games).** The anchor. Ten team KPIs, of which the load-bearing
  ones are: **numerical superiority behind the ball** (1.69 ± 1.00 — the coaches' "+1"), **attacker
  space-control in the rest-defense area** (11.51 ± 9.82 %), deep-space control, dangerous-counterattacker
  coverage; #1 success factor = **fast regain after loss**. Its ML success-*prediction* is weak
  (AUC 0.60) → we ship the descriptive KPIs, not the prediction. It **explicitly excludes the goalkeeper
  from the quantitative analysis** — the gap this feature names.
- **Peters, A., et al. (2025), "A rule-based approach to classify counterpressing … relationship with
  rest defence" (Int. J. Perf. Anal. Sport 26(1); DOI 10.1080/24748668.2025.2473799; 380 EPL matches,
  12,460 sequences).** Defines Rest Defence as a **raw headcount in a defined zone at the moment of loss**;
  the count significantly relates to shot concession (p<0.05) and territory concession (p<0.001).
  Simplest, empirically-validated operationalisation.
- **Memmert, D., Biermann, H., et al. (2022), "Towards Expected Counter" (MLSA).** **"Players behind the
  ball" is the single strongest counterattack predictor**; a team concedes a counter ~30 % of the time
  when it loses the ball centrally with **more than two players not behind the ball**. A concrete,
  implementable spare-man-deficit threshold from tracking alone.
- **Novillo, A., et al. (2025), "Offside Control" (Chaos, Solitons & Fractals 197:116445).** The one paper
  that differentiates the GK inside a pitch-control danger model: assigns the keeper a higher control rate
  **λ_GK = 12.9 s⁻¹ (≈ 3× the outfield 4.3 s⁻¹)** and measures pitch control **behind the opponent's
  offside line** (the second-to-last defender). Independent convergence with silly-kicks' own
  **`SpearmanParams.lambda_gk = 3.0`** — a dimensionless GK-only multiplier (Shaw & Sudarshan derive it as
  3× the outfield rate; there is no separate `lambda_outfield` field). Direct anchor for Layer 2's
  GK-included "controlled space behind the line."
- **Ogawa, Y., Fujii, K., et al. (2025), "Space evaluation at the starting point of soccer transitions"
  (OBPV; arXiv 2505.14711).** A pitch-control model built *specifically for the transition zone* because
  OBSO's goal-proximity weighting misbehaves far from goal. Supplies the field-value weighting
  (`w_field(x,y)`, a longitudinal sigmoid × lateral Gaussian) we adopt as an option for the deep-zone
  threat weighting. Framing/weighting reference.
- **Dash, S., et al. (2025), "Prediction-based evaluation of back-four defense with spatial control"
  (arXiv 2511.06191).** Back-four indicators — space control, stretch index, pressure index, **relative
  line height (strongest predictor)** — validated on post-loss defensive sequences. Anchor for Layer 1's
  rearguard-shape "relative line height."
- **Kim, H., et al. (2026), "Better Prevent than Tackle" (DEFCON-GNN; arXiv 2512.10355).** Values the
  defending team's off-ball positioning via counterfactual EPV deltas per defender. **Comparator /
  inspiration** for the Layer-3 outfield deterrent arm (silly-kicks already cites it as the gkdv
  comparator); not reproduced.
- **Le, H. M., et al. (2017), "Data-Driven Ghosting" (MIT Sloan).** The ghosting concept behind both the
  ghost-GK (TF-18) and the new ghost-outfield model.

Practitioner sources for the GK layer, honestly labelled as such: **FIFA EFI** (the only source that
reports GK **line height** and **GK-to-defensive-line distance** as tracked numbers — WC2022 keeper ≈ 13 m
from goal, ≈ 20 m from the defensive line; WWC2023 ≈ 24 m / 30 m; "the distance … remained virtually
consistent" as both lines rose together — but never as a citable formula, never as code); **StatsBomb
"Aggressive Distance"** / **Wyscout "Sweeper Keeper"** (event-based sweeper-distance definitions).

### 2.3 The novel contribution

pitch-control models drop the keeper because an unadjusted model over-credits it near its own goal (the
"Get Goalside" critique) — silly-kicks already fixes this via threat-weighting and `lambda_gk`. Rest-defense
research, from a different (expert-interview) lineage, simply never asked the GK question. **No publication
couples GK depth + defensive-line height + danger-of-space-behind into one quantity.** TF-60's GK-inclusive
structure KPIs (Layer 1), GK-as-control-agent danger surface (Layer 2), and ghost-in-possession-keeper
deterrent (Layer 3) occupy that gap.

---

## 3. Scope and non-goals

### In scope

- The three descriptive/valuation layers above, all providers (tracking-optimal; SB360 best-effort +
  FOV-flagged).
- The Layer-3 counterfactual: **GK** deterrent arms (threat + space) reusing the shipped ghost-GK model,
  and an **outfield-rearguard** deterrent arm requiring a new **ghost-outfield model**.
- The new **numerical-superiority / goal-side counting primitive**.
- A frozen `RestDefenseParams` with `for_provider`, honest degradation, provenance columns, ADR-077 FOV
  companions, and full CI-gate registration.

### Explicit non-goals

- **No predictive success/counter classifier.** We ship the descriptive KPIs, not Forcher's AUC-0.60
  rest-defense-success model nor the xBG/EPV transition classifiers. Those are framing/templates only.
- **No new VAEP feature, no retrain.** Rest-defense metrics are coach-facing descriptive/valuation
  outputs; they enter no default xfn list and change no existing column. (A future opt-in xfn is out of
  scope here and would be its own self-triggered VAEP retrain — the standard retrain-trigger discipline,
  cf. ADR-005 / ADR-024 / ADR-026.)
- **No composites/archetypes/rankings in the library.** TIV-style z-norm composites, archetype clustering,
  and league rankings stay consumer-side — the raw-primitives-ship discipline recorded in `NOTICE` (the
  TF-45 entry) and glossed by ADR-055. The library ships raw per-`(team, window)` quantities.
- **No action-coupled `add_*` on the primary surface.** TF-60's primary output is a `compute_*` table, not
  an `actions`-enriching aggregator. (The ghost-outfield model in `tracking/` may ship an `add_ghost_outfield`
  mirror; that, if included, is the one C4-aggregator-count change and is scoped to PR5.)

---

## 4. Architecture overview

### 4.1 Packages and layering

```
silly_kicks/
  restdefense/                     NEW package (C4 container), hexagonal, tracking-public-seams-only
    __init__.py                    public surface
    _config.py                     RestDefenseParams (frozen, for_provider)
    _geometry.py                   rearguard line + danger-behind-line zone, GoalMap-oriented
    _counting.py                   goal-side / behind-the-ball counting primitive (group_rows-safe)
    _windows.py                    action-grid sampling + possession windows + loss-instant selection
    _structure.py                  Layer 1 KPIs (compose team_shape / defensive_line / GK stack)
    _danger.py                     Layer 2 danger-behind-line valuation (control_in_region / threat)
    _counterfactual.py             Layer 3 ghost-frame builder (in-possession keeper / rearguard)
    _arms.py                       Layer 3 deltas (reuse gkdv delta seams; RestDefenseReport)
    _report.py                     conservation reports
  tracking/
    _ghost_outfield.py             NEW trained model (mirrors _ghost_gk.py); serve_ghost_outfield_positions
```

- `restdefense/` **imports** `silly_kicks.tracking` public seams and `silly_kicks.gkdv` public seams;
  **nothing** imports `restdefense/`. Pinned by `tests/restdefense/test_import_allowlist.py` (mirroring
  `tests/gkdv/test_import_allowlist.py`).
- The **ghost-outfield model lives in `tracking/`**, exactly where `_ghost_gk.py` lives, because it is a
  general positioning primitive reusable by any consumer (restdefense today; potentially gkdv later). It is
  served to `restdefense/` via a `tracking` public seam (`serve_ghost_outfield_positions`), the same way
  `gkdv` consumes `serve_ghost_gk_positions`.

### 4.2 C4 impact (honest)

- **+1 container**: `restdefense`. Required to be modelled by the C4 completeness gate.
- **+0 or +1 tracking aggregator**: only if the ghost-outfield model ships an `add_ghost_outfield`
  action-coupled mirror (decided in PR5). The rest-defense primary surface adds no action-coupled
  aggregator.
- C4 diagram regenerated in the PR that first introduces the container (PR1), via the `mad-scientist-skills:c4`
  pipeline with the pinned Graphviz `dot`.

---

## 5. The measurement model

### 5.1 Why the action grid (SB360-forced, and it happens to be right)

SB360 freeze-frames are single snapshots at each on-ball event — no frames between events, no velocity,
FOV-cropped, anonymous players. A per-*frame* continuous sample is therefore impossible on SB360. The one
grain available on **both** SB360 and continuous tracking is the **in-possession team's on-ball action
grid**: each action carries a freeze-frame (SB360) or a linked tracking frame (Sportec / GS / SkillCorner
/ Metrica). This is also silly-kicks' native action-centric grain.

At each in-possession on-ball action of team A (`derive_team_in_possession`), we resolve the frame at that
action (via `link_actions_to_frames`, whose `(pointers, LinkReport)` output carries the per-period
coverage floor, on continuous tracking; the action's own freeze-frame on SB360) and compute the structure
at that instant.

### 5.2 Windows and the "committed-forward" gate

- **Possession windows** come from `spadl.add_possessions` (`possession_id`). Rest defense is a property of
  A's possession, so the natural window is the possession.
- **Committed-forward gate**: rest defense is only meaningful when A is advanced (the counter risk is real).
  Gate: the ball is at least `RestDefenseParams.min_ball_advance_m` from A's own goal (default: past
  halfway, tunable; un-calibrated at ship, with an empty `for_provider` override map — the
  `CoverShadowParams` pattern, ADR-066 — until a calibration apply-gate clears). Actions failing
  the gate are **dropped-and-counted** (never scored as a null structure), mirroring the gkdv
  drop-conservation discipline.
- **Moment-of-loss snapshot**: the possession's terminal action (the turnover) is flagged
  `is_possession_loss = True`; its structure sample is the loss snapshot. It also carries the linkage to
  the *subsequent* opponent possession's early outcome (a join key), enabling the descriptive
  concession-linkage validation without shipping a predictive model.

### 5.3 Output grains

Three tables, each honest-NaN-reduced (never a fabricated 0 for an unscoreable sample — mirroring the
gkdv `min_count=1` discipline):

1. **`compute_rest_defense(actions, frames, ...)` → per-`(game_id, period_id, team_id, action_id)`
   samples** — one row per in-possession on-ball action that passed the gate, carrying the **Layer-1/2**
   metrics + provenance + FOV companions (Layer-3 arm columns come from the separate deterrent functions,
   §14). This is the richest and most SB360-natural grain.
2. **`summarize_rest_defense(samples, by="possession")` → per-`(team, possession_id)`** — mean / summary
   over a possession's samples, plus the loss snapshot.
3. **`summarize_rest_defense(samples, by="match")` → per-`(team, game_id)`** — the coach-facing per-match
   rollup.

Rollups are pure reductions over the samples table (no re-linking, no re-scoring), so a consumer can
re-aggregate with its own policy.

---

## 6. The rest-defense geometry

All geometry is **oriented via `resolve_defended_goals(frames) → GoalMap`, built once per match and
threaded in** (ADR-055: build once per match, never re-derived per frame, never from team identity). For in-possession
team A in `(game, period)`:

- `G_A = goal_map.get(game, period, A)` — the end **A defends** (∈ {0, 105} in the LTR-normalised frame).
- `G_B = goal_map.attacked_goal(game, period, A)` — the end **A attacks** (a real opponent-entry lookup,
  never `105 − G_A`).
- **Rearguard line** `L_A` = `compute_defensive_line(frames, goal_map=…)[team A].defensive_line_x` (mean x
  of A's deepest `n_rearguard` outfield players; TF-14 `select_back_line_players`, `n` tunable, default 4
  with adaptive fallback).
- **Danger-behind-the-line zone** `Z` = the strip between `L_A` and `G_A`, full pitch width: if `G_A = 0`,
  `Z = [0, L_A] × [0, 68]`; if `G_A = 105`, `Z = [L_A, 105] × [0, 68]`. This is the region a counter runs
  in behind.
- **"Behind the ball"** = between the ball's x and `G_A` (the x-band on A's defensive side of the ball),
  per Memmert's operationalisation.

An unresolvable end (`GoalMap` `unresolved`) → the sample's geometry-dependent metrics are **honest-NaN**
(`GoalEndUnresolvedError` caught at the `compute_*` edge, per ADR-055), never a confident guess.

---

## 7. Metric catalog

Each metric lists its definition, its published anchor, and its provider tier (T1 = all providers incl.
velocity-less SB360; T-vel = velocity-required, tracking-only). Thresholds/weights marked *(calibratable)*
ship with a documented default and an empty `for_provider` override map (the `CoverShadowParams` pattern,
ADR-066) until a calibration apply-gate clears; the harness recommends a value, and applying it is a
separate gated PR (ADR-009).

### 7.1 Layer 1 — descriptive structure KPIs (PR1)

| Column (base) | Definition | Anchor | Tier |
|---|---|---|---|
| `rd_num_superiority` | `(# A players behind the ball) − (# B players behind the ball)`, counted in the x-band `[ball_x → G_A]` via the goal-side counting primitive (§8) | Forcher 2023 (1.69±1.00); Memmert 2022 | T1 |
| `rd_num_superiority_gk` | as above, **including A's keeper** in A's count when the keeper is in the band | Forcher 2023 + coaches' "+1"; novel GK inclusion | T1 |
| `rd_zone_occupancy` | headcount of A's players in the danger zone `Z` | Peters 2025 | T1 |
| `rd_line_height` | `defensive_line_x` of A's rearguard, distance from `G_A` (absolute) | Dash 2025; FIFA EFI | T1 |
| `rd_line_height_relative` | rearguard line x **relative to the ball's x** (vertical coordination with the ball) | Dash 2025 (strongest predictor) | T1 |
| `rd_compactness_x` / `rd_width` / `rd_depth` | **Owner-ratified "Option B" (2026-08-30, ADR-080; supersedes the original "restricted to the rearguard subset" wording after the `/review-impl` gate):** `rd_compactness_x` (rearguard x-range) and `rd_width` (rearguard lateral/y width) both come from `compute_defensive_line` (the back line, GK-excluded) — genuinely rearguard-subset; `rd_depth` is the WHOLE-TEAM front-to-back `team_length` from `compute_team_shape` (a back-line depth would duplicate `rd_compactness_x` since a flat line has ~no independent depth, so the team's vertical stretch is the informative counter-risk signal). | Dash 2025 (arXiv:2511.06191); Zhang 2025 | T1 |
| `rd_shape_2_3_vs_3_2` | rest-defense **stagger** label (2-3 / 3-2) from the Ward line clustering (TF-44) over the **rest-defense unit** — the players *behind the ball* (dynamic, typically 5), NOT the fixed `n_rearguard`=4 back line (a 2-3/3-2 stagger needs 5 players) | practitioner (Coaches' Voice); TF-44 | T1 |
| `rd_gk_line_height` | A's keeper distance from `G_A` (goal-relative x) | FIFA EFI; StatsBomb/Wyscout sweeper distance | T1 |
| `rd_gk_to_line_distance` | A's keeper x minus `defensive_line_x` (the coupled-unit gap FIFA reports descriptively) | FIFA EFI (first citable formalisation); **novel** | T1 |

### 7.2 Layer 2 — danger-behind-the-line valuation (PR2)

| Column (base) | Definition | Anchor | Tier |
|---|---|---|---|
| `rd_attacker_space_control` | team B's pitch-control **share of the zone `Z`** — `1 − PitchControlSurface(attacking=A).control_in_region(x_min, x_max, 0, 68)` over Z's x-bounds, or B's surface directly; positional model on SB360 (ADR-063 Tier-1 lift) | Forcher 2023 (11.51 %) | T1 |
| `rd_danger_behind_line` | threat-weighted counter-danger of `Z` — the zone integral of `B_control(x,y) × threat_toward_G_A(x,y)`. Realised by `compute_threat_pc(frame, attacking_team_id=B, xt, goal_map)` (xT toward `G_A` concentrates it in `Z` for free), optionally re-weighted by OBPV `w_field` *(calibratable)* | Novillo 2025; Ogawa 2025; Spearman 2018 | T1 |
| `rd_danger_behind_line_gk` | as above with A's keeper as a control agent via `SpearmanParams(lambda_gk=…)` — the keeper's contribution is `rd_danger_behind_line` (GK-blind, keeper excluded from A's control) minus this | Novillo 2025 (λ_GK); Shaw & Sudarshan 2020; **novel GK inclusion** | T1 |
| `rd_gk_coverage_behind_line` | fraction of `Z` the keeper controls — pitch-control-share form (universal) | **novel** | T1 |
| `rd_gk_reachable_coverage_m2` | `area(GK reachable region ∩ Z)` — TF-15 reachable-area (m²) form | **novel**; TF-15 | T-vel (honest-NaN on SB360, ADR-063 Tier-2) |

### 7.3 Layer 3 — counterfactual deterrent arms (GK arms = PR4; outfield arm = PR6; GK-ghost re-fit = PR3, see §17)

All arms are `actual − ghost` in **attacker-value units → negative = deterrent** (gkdv convention). Ghost
positions come from a league-average model in the **same frame state** (not a fixed goal-line), so the
counterfactual reacts to the game situation.

| Column (base) | Definition | Ghost source | Anchor | Tier | PR |
|---|---|---|---|---|---|
| `rd_gk_deter_threat` | `threat_pc(B, actual) − threat_pc(B, ghost-A-keeper)` | `serve_ghost_gk_positions` (`rest_defense` variant, PR3) | Le 2017; Novillo 2025; gkdv lineage | T1 (positional) | PR4 |
| `rd_gk_deter_space` | `DAS(B, actual) − DAS(B, ghost-A-keeper)` behind the line | `serve_ghost_gk_positions` (`rest_defense` variant, PR3) | Bischofberger & Baca 2026 | T-vel | PR4 |
| `rd_outfield_deter_threat` | `threat_pc(B, actual) − threat_pc(B, ghost-A-rearguard)` | `serve_ghost_outfield_positions` (new model) | Kim 2026 (DEFCON); Le 2017 | T1 (positional) | PR6 |
| `rd_outfield_deter_space` | `DAS(B, actual) − DAS(B, ghost-A-rearguard)` behind the line | `serve_ghost_outfield_positions` | Bischofberger & Baca 2026 | T-vel | PR6 |

**Reuse, not reimplementation:** the differencing is `gkdv`'s generic `delta_threat_suppression_batch`
(threat) and `delta_das_batch` (space) — read from the code, these take two aligned frame legs +
`attacking_team_id_by_frame` and already carry the identity-cache-trap avoidance, the DAS direction-pin-once,
`min_count=1` honest-NaN, and the sign convention. `restdefense/` imports them (a package may import
`gkdv`; `tracking` may not). Only the **ghost-frame builder** (§9) is new.

---

## 8. The numerical-superiority / goal-side counting primitive (`restdefense/_counting.py`)

No *public* counting surface exists; the only prior art is `GhostGkModel`'s private `defenders_behind_ball`
(`_ghost_gk.py:733`, `to_gr_x(x) < ball_x`). The new primitive is modelled on `compute_packing_metrics`'
bypass-count idiom (`_packing.py:185`: goal-map-mirrored opponent x-coordinates, `np.count_nonzero` over an
x-band), and its orientation is **verified consistent** with that prior art — both count players goal-side
of the ball; the only difference is the orientation *source* (the canonical per-match `GoalMap` here, per
ADR-055, vs. the ghost model's per-frame `to_gr_x`), not the semantics:

```
count_goalside(frame_players, *, team_id, ball_x, goal_x, goal_map) -> int
    # players of team_id whose goal-relative x is between the ball and goal_x (their own goal)
```

Rules (from the ADR-055 / ADR-019 / ADR-068 disciplines):

- **Orientation**: mirror x into the goal-relative frame via `goal_map`, never team identity. Counting is
  1-D along the attacking axis (inherits TF-45/TF-49's documented far-touchline caveat — a wide player in
  the x-band is counted).
- **Ids**: `ids_match(frame["team_id"], team_id)` (ADR-019), never raw `==`.
- **Loops**: any per-frame loop uses `group_rows(frames, ("game_id","period_id","frame_id"))` once, then
  `.get(...)` per sample (ADR-068). A new `group_rows` caller registers a scoped counter in
  `tests/_scale_guarded.SCALE_GUARDED` and passes `assert_subquadratic_growth` with a fixture that scales
  the **sample (loop) dimension** (ADR-073).
- **FOV**: on SB360 the count is over *visible* players only → it under-reports when the rearguard is
  cropped. The primitive returns the raw count; the ADR-077 companion (`_observed_fraction` /
  `_observed_source`) is attached at the metric edge and is load-bearing.

---

## 9. The counterfactual engine (`restdefense/_counterfactual.py`)

A gkdv **sibling**, not a gkdv call — gkdv's `build_ghost_frames` ghosts the *defending* keeper when the
ball is near the *attacked* goal (verified in `gkdv/_engine.py`), the opposite geometry to rest defense.
The rest-defense builder:

```
build_restdefense_ghost_frames(
    frames, *, which: Literal["keeper","rearguard"], model, home_team_id,
    carrier=None, params: RestDefenseParams = _DEFAULT,
) -> tuple[cf_frames, provenance, RestDefenseGhostReport]
```

- **Domain** (drop-and-counted; a CI gate asserts `n_frames_scored + Σ drop_reasons == n_frames_in`, the tested conservation `GkdvReport` uses — fields `n_frames_in` / `n_frames_scored`): alive
  ball; in-possession team A resolved; committed-forward gate (§5.2); A's keeper (or ≥ `n_rearguard` A
  field defenders) present with finite coordinates; `GoalMap` resolvable.
- **Substitution**: `which="keeper"` moves **A's** (in-possession) keeper to
  `serve_ghost_gk_positions(...)`; `which="rearguard"` moves A's deepest-`n_rearguard` field defenders to
  `serve_ghost_outfield_positions(...)`. Pure — never mutates `frames`; writes back only the substituted
  rows (the `_write_back` discipline from gkdv).
- **A NaN/missing ghost is dropped-and-counted, never scored as Δ=0** (the gkdv rule; a zero delta from a
  vanished player biases aggregates toward "no deterrence"). A non-finite served ghost on a scored frame
  **raises** (pitch control silently drops NaN-coordinate rows, so a NaN ghost would make the player
  vanish rather than error — the exact gkdv guard).
- The two legs (`actual`, `cf`) are handed to `_arms.py`, which restricts to the scored set and calls the
  gkdv delta seams with `attacking_team_id_by_frame = B` (the future counter-attacker).

**Validity risk (RESOLVED 2026-08-30 — the gate FIRED):** the shipped `GhostGkModel` was exercised by gkdv
only in the defending-near-attacked-goal domain, and its validity for the **in-possession high-sweeper**
state was measured (`docs/research/tf60_ghost_gk_in_possession_validity/`). Outcome: it is **structurally
inadequate** — `prepare_ghost_gk_training_data` drops every keeper label above `GRID_X_MAX = 30 m` as
"sweeper rushes," so `predict_mean` hard-saturates at ~30 m and cannot place a keeper at the 30–45 m an
in-possession sweeper occupies; the `ghost_out_of_box` flag is blind to it (the model clips its own output
to ≤30 m). Per the owner ruling, the arms therefore do **not** trust an out-of-domain serve — a
rest-defense-appropriate **extended-grid GK-ghost variant** is re-fit in a dedicated cycle *before* the arms
(sub-spec `2026-08-30-tf60-restdefense-gk-ghost-refit-design.md`; §17 arc reshaped). This was a gate, not an
assumption.

---

## 10. The ghost-outfield model (`tracking/_ghost_outfield.py`, PR5 — its own sub-spec)

A new trained positioning model, **mirroring `GhostGkModel`** (verified structure: `fit` / `predict_mean`
(deterministic boosted HGBR) / `predict_density` / `save` / `load` with chirality + feature-contract
guards / `from_variant` / `from_hub`; npz + JSON + `SHA256SUMS`, pickle-free, fail-closed load; velocity-keyed
`position_only` variant for SB360, ADR-067; numba leaf traversal, ADR-076).

- **Target**: the positions of a league-average in-possession team's deepest-`n_rearguard` field defenders,
  conditioned on frame state — per-slot (deepest, 2nd-deepest, …) goal-relative coordinates. Feature set:
  leakage-safe, **state-anchored** goal-relative geometry at decision time (ball position, phase, score
  state, rearguard slot index) — no post-event inputs (ADR-011; Peters/Davies temporal-leakage discipline).
- **Public seam**: `serve_ghost_outfield_positions(frames, *, model, home_team_id, ...)`, mirroring
  `serve_ghost_gk_positions`. Optional `compute_ghost_outfield` / `add_ghost_outfield` mirrors
  (the latter is the only possible C4-aggregator-count change).
- **Data-visibility discipline** (ADR-038 fail-closed corpus): a *bundled public* model trains on a
  public-only corpus; owner-tier variants (`sc_extended`-style) train on owner data and are **not** bundled.
  Same discipline as ghost-GK / xShot / xCross.
- **This section is interface-level.** PR5 gets a dedicated sub-spec (target parametrisation, feature list,
  training corpus, validation harness, HF publishing) exactly as TF-18/16/17 each did. It ships as **one
  cycle** (code + pipeline + bundled weights).

---

## 11. Provider / velocity / FOV handling

### 11.1 Velocity tiers (ADR-063), decided not asked

| Quantity class | SB360 (velocity-less) | Rule |
|---|---|---|
| Counts, pitch-control **share**, threat-weighted control, GK line-height, GK-to-line gap, rearguard shape | **computed** (positional / zero-velocity model) | Tier-1 lift — smooth limit of the same model |
| Reachable-area **m²**, DAS-based space arms | **honest-NaN** | Tier-2/3 — velocity-constitutive; suppressed, never fabricated |

Velocity-less frames route through `tracking._velocity_availability.zero_velocity_if_unavailable(...)`
(ADR-063) at the engine edges; a *forgotten* `derive_velocities()` on a velocity-**declared-absent** SB360
frame is fine, but on an undeclared full-tracking frame it **raises** (the ADR-043/054 fail-loud discipline).

### 11.2 FOV observability (ADR-077), mandatory

Every count/region metric ships `<base>_observed_fraction` + `<base>_observed_source` (over the closed
`{observed, no_polygon, degenerate_polygon, degenerate_region}` vocabulary, plus caller-overlaid `unlinked`),
via the `_fov_registry`-style companion mechanism. The region for a count is the danger zone `Z` (a fixed
action-LTR strip keyed on the metric's role — **never** a `goal_map`, per ADR-077 S1: the polygon + metrics
are action-LTR and a `goal_map` would mis-orient away-possession). The companions are **opt-in on a
`visible_area=` kwarg** — primary columns are byte-identical with and without it, so no VAEP retrain, and
they are glossary/SB360-audit-exempt exactly as ADR-062/077 companions are.

### 11.3 Keeper identity (ADR-078)

`resolve_keeper_identities(actions, frames, *, identity=…, roster=…)` — `native` (delegates to TF-13
`defending_gk_from_frames`) for continuous tracking; `roster` (event→roster→derivation ladder) for
anonymous SB360, with `apply_keeper_identities_to_frames` bridging the resolved id onto the numbered
freeze-frame rows so the GK metrics key on the right player.

---

## 12. Public API surface

```python
from silly_kicks.restdefense import (
    RestDefenseParams,
    compute_rest_defense,          # (actions, frames, *, xt=None, goal_map=None, links=None,
                                   #  pitch_control_cache=None, visible_area=None, params=…) -> samples
    summarize_rest_defense,        # (samples, *, by="possession"|"match") -> rollup
    merge_rest_defense,            # (samples, *arms) -> samples left-joined with the arm tables (§14)
    build_restdefense_ghost_frames,
    rest_defense_gk_deterrent,     # (actions, frames, *, xt, ghost_gk_model, ...) -> arm columns
    rest_defense_outfield_deterrent,  # (actions, frames, *, xt, ghost_outfield_model, ...) [PR6]
    RestDefenseReport, RestDefenseGhostReport,
)
from silly_kicks.tracking import serve_ghost_outfield_positions, GhostOutfieldModel  # PR5
```

- `xt` (fitted `ExpectedThreat`) is a **required, caller-supplied** model for the threat-weighted metrics
  (Layer 2 `rd_danger_behind_line*` and all threat arms) — fail-closed, mirroring xt-GK / xt-xfns
  (`compute_threat_pc` refuses an unfitted xT, 4.62.0). Layer 1 needs no `xt`.
- Frame-consuming functions follow the ADR-078 call convention: `frames` positional-or-keyword next to
  `actions`; every optional parameter keyword-only; accept `links` and `pitch_control_cache` for pipeline
  reuse.
- **No single function emits all three layers.** `compute_rest_defense` emits Layer-1/2 samples; the
  Layer-3 `rest_defense_*_deterrent` functions (which take the ghost model) return **separate keyed
  tables** (key `(game_id, period_id, team_id, action_id)`); `merge_rest_defense(samples, *arms)`
  left-joins them onto the samples table with honest-NaN on arm-dropped rows (§14).

---

## 13. Parameters (`RestDefenseParams`, frozen, `for_provider`)

Combines the `CoverShadowParams` `for_provider` empty-override-map pattern (ADR-066) with the flag-based
`is_default()` of `PreprocessConfig` — note `CoverShadowParams` itself has **no** `is_default`:

```python
@dataclass(frozen=True)
class RestDefenseParams:
    n_rearguard: int = 4                      # back-line size for line geometry (TF-14); NOT the rest-defense unit
    min_ball_advance_m: float = 52.5          # committed-forward gate (past halfway)  (calibratable)
    zone_depth_m: float | None = None         # None => full strip [line, own goal]; else a capped depth
    danger_field_weight: bool = False         # OBPV w_field re-weighting of the deep-zone threat (opt-in)
    possession_stride: int = 1                 # sample every Nth in-possession action (cost control)
    _is_universal_default: bool = field(default=False, compare=False, repr=False)
    # NB: GK control weighting is NOT duplicated here -- SpearmanParams.lambda_gk (=3.0) owns it,
    #     inherited from the caller-supplied / for_provider pitch-control params (one source, no drift).

    @classmethod
    def default(cls, *, force_universal=False) -> "RestDefenseParams": ...
    @classmethod
    def for_provider(cls, provider: str) -> "RestDefenseParams":   # returns base for an unlisted provider
        return dataclasses.replace(cls(), **_PROVIDER_REST_DEFENSE_PARAMS.get(provider, {}))
    def is_default(self) -> bool:              # FLAG-based (from PreprocessConfig; CoverShadowParams has none)
        return self._is_universal_default

_PROVIDER_REST_DEFENSE_PARAMS: dict[str, dict] = {}   # EMPTY until an ADR-066-style apply-gate clears
```

All calibratable defaults ship un-tuned with the empty override map; a per-provider tune is a separate
gated apply PR (ADR-009), never in this cycle.

---

## 14. Output schema

- **Samples table** (`compute_rest_defense`) keys: `(game_id, period_id, team_id, action_id)` +
  `possession_id`, `is_possession_loss` (bool), the **Layer-1 and Layer-2** metric columns, provenance
  columns, and (when `visible_area=` supplied) the FOV companions.
- **Arm tables** (`rest_defense_gk_deterrent`, `rest_defense_outfield_deterrent`) are **separate**, keyed
  on the same `(game_id, period_id, team_id, action_id)`, carrying only the Layer-3 arm columns + each
  arm's `<arm>_source`. The arm drop-domain (§9) is a declared **subset** of the Layer-1 gate (it also
  requires the keeper / ≥`n_rearguard` defenders present + a served finite ghost), so a caller left-joins
  arm columns onto samples and gets **honest-NaN** on rows the arm dropped — never a fabricated 0. A
  convenience `merge_rest_defense(samples, *arms)` performs that left-join, asserts the arm keys are a
  subset of the sample keys, and reconciles drop-conservation across `RestDefenseReport` and
  `RestDefenseGhostReport` (an arm-dropped row is counted once in the ghost report, never double-counted).
- **Provenance / source columns** over closed vocabularies (the `das_source` idiom — a token that *varies*,
  never a constant column): `rd_geometry_source` (`resolved` / `guessed` / `unresolved` — `guessed` is a
  `GoalMap` `allow_guess` outfield-mean fallback whose metrics are computed but whose defended-goal end is
  an inference, which matters on FOV-cropped SB360; PR1 IMPL-02), each counterfactual arm's
  `<arm>_source` (`computed` / `unscoreable_frame` / `ghost_missing` / `unlinked`), and the FOV
  `*_observed_source`. A non-two-team frame set (no resolvable opponent) yields `pd.NA` numerical
  superiority — never a fabricated A-count (the absent B-count would read A's whole rearguard as
  "superiority"; PR1 IMPL-04).
- **Nullable dtypes**: counts `Int64`, fractions/metrics `float64`, flags `boolean` — NA on unscoreable
  rows (never a sentinel 0; ADR-027).
- **Reports**: `RestDefenseReport` (samples: `n_frames_in` / `n_frames_scored` / `drop_reasons`, field
  names mirroring `GkdvReport`) and `RestDefenseGhostReport` (per arm). Conservation
  (`n_frames_scored + Σ drop_reasons == n_frames_in`) is a **CI gate**, not a dataclass property (as in
  gkdv).

---

## 15. Error handling and degradation

- **Unresolvable goal end** → `GoalEndUnresolvedError` raised in the per-frame engine, caught at the
  `compute_*` edge → honest-NaN row (per ADR-055). Never a `nan < 52.5` confident-105 guess.
- **Velocity-required metric on velocity-less-but-declared frames** → honest-NaN (Tier-2/3). On
  velocity-**undeclared** full-tracking frames → **raise** (ADR-063).
- **DAS arm** → `DasUnscoreableError` is the only degradable exception (→ NaN); everything else propagates
  (ADR-043). Reused from `delta_das_batch`.
- **Missing/NaN ghost on a scored frame** → dropped-and-counted (a missing player is not Δ=0); a non-finite
  served ghost → raise (§9).
- **Unfitted / missing `xt`** → refuse (fail-closed), never a structural-zero threat column (4.62.0).
- **FOV** → a cropped region yields a real partial count + a `_observed_fraction < 1` flag; a fully
  unobserved region yields NaN with `_observed_source = degenerate_region`/`no_polygon` (never 1.0, never
  0.0).

---

## 16. Validation

Applied results are **reported, never gated** (repo convention); *methods* are CI-gated.

### 16.1 Face-validity anchors (reported, on the public/open corpora)

- `rd_num_superiority` centres near the Forcher **+1** (≈ 1.69 ± 1.00 on Bundesliga; report our corpus'
  value, not a hard assertion — magnitudes are corpus-dependent).
- `rd_attacker_space_control` order-of-magnitude near Forcher's **≈ 11.5 %**.
- `rd_gk_to_line_distance` order-of-magnitude near FIFA's **≈ 20–30 m** (competition-dependent).
- Counterfactual **sign / expected-keeper test**: known sweeper-keepers (high line, aggressive) should
  score more negative (more deterrent) than deep line-keepers — the gkdv owner-validation pattern.

### 16.2 CI gates (methods)

- **Liveness** (`tests/restdefense/…`): every emitted column non-NaN and (float metrics with ≥2 obs)
  non-constant on a multi-domain fixture; structural constants (if any) registered with a justification.
- **Purity** (ADR-033): `compute_*` / arms never mutate caller inputs; ≥2 variants for any
  present/absent-branch column (the `visible_area` companion path → a 2nd variant).
- **id-dtype invariance** (ADR-019): numeric-actions × string-frames and reverse yield identical results.
- **Orientation / D3** (ADR-051): mirror the frames, hold `home_team_id` constant → action-LTR geometry
  unchanged; the `GoalMap`-consuming metrics move when the map is swapped (Gate C), and direction never
  comes from team identity.
- **FOV completeness** (ADR-077): every FOV-sensitive column companioned or `_OBSERVABILITY_EXEMPT`
  with a reason.
- **Counterfactual non-vacuity** (CLAUDE.md "every counterfactual needs a non-vacuity assertion"): a test
  that the ghost leg *measurably differs* from its factual twin (mirroring
  `tests/gkdv/test_arms.py::test_unpinned_implementation_would_measurably_differ`), and a two-sided band
  test (a mutation that *should* move the number out of band does).
- **SB360 audit** (ADR-053): the boundary entries (`compute_rest_defense`, the arms) carry a per-column
  verdict + `verdict_provenance`; velocity-less-degrading columns adjudicated `honest_nan` /
  `differs_by_design`, never `silent_degrade`.
- **Glossary** (ADR-048): every emitted column documented in `feature_glossary.py` (companions
  glossary-exempt, as ADR-062/077).
- **Import allowlist**: `tracking` never imports `restdefense`.
- **Ghost-outfield model** (PR5): golden / chirality / feature-contract / integrity-on-load gates mirroring
  the ghost-GK suite; `position_only` variant round-trips.
- **Sub-quadratic growth** (ADR-073) for the new counting/looping primitive.
- **End-to-end method gate** (`@e2e`, owner/fixture-gated, following the `worldcup-hdf5-e2e` precedent):
  run `compute_rest_defense` / `summarize_rest_defense` (and, at PR4+, the arms) on **≥1 real linked
  tracking match** (exercises `link_actions_to_frames` coverage + native keeper identity + full-coverage
  FOV) **and ≥1 real SB360 match** (exercises `resolve_keeper_identities(identity="roster")` +
  `apply_keeper_identities_to_frames` + FOV-partial companions on a genuinely cropped advanced-ball
  frame). It asserts the **method** — non-empty output, drop-conservation reconciles, FOV companions
  populated with a `<1.0` observed fraction on the cropped SB360 case — **not** metric values, so it stays
  compatible with the reported-not-gated applied-results convention. Synthetic fixtures cannot exercise
  these integration seams (linking, roster bridging, real FOV crop).

---

## 17. Decomposition into cycles

**Six cycles** (reshaped 2026-08-30 by the §9 finding — see below), each a single fully-tested
squash-merged branch with its own version bump + tag (owner approves each commit / merge / tag
separately). Docs land **in the first commit of each branch** (no standalone doc commit).

**Arc reshaped 2026-08-30:** the §9 in-possession-validity gate FIRED — the shipped `GhostGkModel`
hard-saturates at its trained-label ceiling `GRID_X_MAX = 30 m` and cannot represent the in-possession
high-sweeper regime (`docs/research/tf60_ghost_gk_in_possession_validity/`). Per the owner ruling, a
**rest-defense GK-ghost re-fit model cycle is inserted before the GK arms** (its own sub-spec,
`2026-08-30-tf60-restdefense-gk-ghost-refit-design.md`), mirroring the existing model→arm shape (ghost-
outfield model → outfield arm). Original PR3–PR5 shift down by one.

| Cycle | Content | New model? | C4 |
|---|---|---|---|
| **PR1** | `restdefense/` package skeleton + `RestDefenseParams` + geometry + counting primitive + **Layer 1** KPIs + windows/sampling + all CI-gate registrations + ADR-080 | no | +1 container (regenerate C4) |
| **PR2** | **Layer 2** danger-behind-line valuation (`control_in_region` / `compute_threat_pc` / GK-as-control-agent) | no | — |
| **PR3 (NEW)** | **Rest-defense GK-ghost re-fit** — extended-grid additive `GhostGkModel` variant (grid becomes first-class; label cap lifted; `default`/`position_only`/`full` frozen; **no GKDV/VAEP retrain**) + bundled weights + HF publish — **its own sub-spec** | yes | — |
| **PR4** (was PR3) | **Layer 3 GK arms** (`build_restdefense_ghost_frames(which="keeper")` + threat + space; reuse gkdv delta seams; **consume the PR3 `rest_defense` variant**) | no | — |
| **PR5** (was PR4) | **ghost-outfield model** `tracking/_ghost_outfield.py` (code + training pipeline + bundled weights + fail-closed loader + guards + HF publish) — **its own sub-spec** | yes | +1 tracking aggregator iff `add_ghost_outfield` ships |
| **PR6** (was PR5) | **Layer 3 outfield arm** (`build_restdefense_ghost_frames(which="rearguard")` consuming the PR5 model) | no | — |

Each cycle leaves `main` green and coherent; PR2–PR6 each depend only on the prior cycle's public surface.

---

## 18. Attribution (ADR-005 / NOTICE)

New `NOTICE` "Mathematical / Methodological References" entries (peer-reviewed vs practitioner vs
comparator labelled, per house convention):

- Forcher et al. (2023) JSSM — rest-defense KPI battery (numerical superiority, space control); the
  structural KPIs, not the AUC-0.60 prediction.
- Peters et al. (2025) IJPAS — zone-occupancy operationalisation.
- Memmert/Biermann et al. (2022) MLSA — "players behind the ball" spare-man threshold.
- Novillo et al. (2025) Chaos Solitons & Fractals — λ_GK-included control behind the line (Offside Control).
- Ogawa/Fujii et al. (2025) arXiv 2505.14711 — OBPV transition-zone field-weighting (opt-in weighting only).
- Dash et al. (2025) arXiv 2511.06191 — relative line height / back-four indicators.
- Kim et al. (2026) DEFCON arXiv 2512.10355 — comparator/inspiration for the outfield deterrent arm.
- Le et al. (2017) MIT Sloan — ghosting (shared with TF-18 / gkdv).
- FIFA EFI; StatsBomb "Aggressive Distance"; Wyscout "Sweeper Keeper" — practitioner GK-depth references.

**Pre-existing NOTICE correction to fold into PR1** (verify first, cite by reading): the current `NOTICE`
lists `arXiv:2511.06191` under *Herold et al. (2022)* and `arXiv:2511.00121` under *Forcher et al. (2022)*
— those IDs are Nov-2025 arXiv IDs and cannot belong to 2022 papers (2511.06191 appears to be Dash et al.
2025). Verify the intended IDs for the TF-14 defensive-line citations and correct them while TF-60 touches
the same reference block.

---

## 19. Rejected alternatives

- **Tracking-resident (no new package).** Rejected: would force reimplementing gkdv's DAS
  direction-pin/identity-cache differencing (a documented silent-bug seam), and breaks the established
  gkdv/xtgk precedent that a composite metric subsystem is its own package. (Owner confirmed the new
  package.)
- **A parametric league-average rearguard shape instead of a trained ghost-outfield model.** Rejected: a
  static shape ignores game state, so it is not a realistic counterfactual, and a corpus-percentile shape
  is a frozen-exogenous / consumer-side artifact (the raw-primitives-ship discipline — NOTICE/TF-45,
  ADR-055). The trained model is the gold-standard counterfactual. (Owner chose the trained model.)
- **Per-frame continuous sampling.** Rejected: impossible on SB360 (no inter-event frames). The action
  grid subsumes it and unifies both provider classes.
- **Reusing gkdv `build_ghost_frames` directly.** Rejected: its domain ghosts the *defending* keeper near
  the *attacked* goal — the opposite geometry. Only gkdv's generic *delta* seams are reused.
- **Calling this a VAEP feature.** Rejected: it is coach-facing descriptive/valuation output; opting it
  into a default xfn list would be a self-triggered VAEP retrain (cf. ADR-005/024/026) and is out of scope.
- **The papers' predictive models.** Rejected: weak (AUC ~0.60) and off-charter; we ship the descriptive
  KPIs (Forcher's own recommendation direction).

---

## 20. Open questions and risks

1. **Ghost-GK in-possession validity (gate FIRED, RESOLVED 2026-08-30).** Measured
   (`docs/research/tf60_ghost_gk_in_possession_validity/`): the shipped model hard-saturates at the 30 m
   trained-label ceiling and cannot represent the in-possession high-sweeper. The arc is reshaped — an
   extended-grid rest-defense GK-ghost variant is re-fit in a dedicated cycle before the arms (sub-spec
   `2026-08-30-tf60-restdefense-gk-ghost-refit-design.md`). The **authoritative DGX-corpus fraction of
   committed-forward frames above 30 m** is the one open sub-item, deferred to and measured inside that
   re-fit cycle (owner: "qualitative finding is enough to proceed"). This was a gate, not an assumption.
2. **Ghost-outfield target parametrisation (PR5 sub-spec).** Per-slot deepest-N positions vs a whole-rearguard
   set-prediction; how to handle a variable rearguard size; leakage-safe feature list — deferred to the
   PR5 sub-spec.
3. **Committed-forward gate default.** `min_ball_advance_m = 52.5` (past halfway) is a spec-time default;
   calibration deferred (empty `for_provider` map). Sensitivity to be probed and reported.
4. **Deep-zone threat weighting.** Whether to adopt OBPV's `w_field` (opt-in, off by default) vs relying on
   xT-toward-goal's natural concentration — reported comparison, not a default change.
5. **SB360 coverage floor.** Deep-rearguard cropping when the ball is advanced is intrinsic to broadcast
   FOV; the FOV companions surface it, but the practical yield on SB360 for the advanced-ball rest-defense
   moment must be measured and documented so consumers calibrate expectations.
6. **Counterfactual construct validity.** Like xt-GK v2, the deterrent arms should be reported with an
   honest construct-validity note (expected-sign test, not an outcome-AUC claim) — a deterrent metric is a
   descriptive lens, not a validated predictor.
7. **`rd_shape_2_3_vs_3_2` unit-size edge (PR1 detail).** The 2-3/3-2 stagger label assumes a ~5-player
   behind-the-ball unit; PR1 resolves the 4-/6-player edge (fall back to a count-based `n-m` descriptor
   when the behind-ball unit size ≠ 5) rather than forcing a 5-player split.
