# xT-GK v2 — Lakehouse Handoff & Data-Quality Findings

**From:** silly-kicks (xT-GK v2 completion, PR-S109, 2026-07-10) · **To:** luxury-lakehouse session
**Scope:** `soccer_analytics.dev_gold` action/shot marts + `bronze.spadl_actions`. All numbers are
live probes run this session. Companion: `GATE_FINDINGS.md` (the make-or-break gate verdict).

silly-kicks is a **consumer** — it computes metric surfaces from the marts; the lakehouse owns the
data. Two findings (**F1, F2**) block xT-GK v2 from passing its gate and shipping a full-domain ρ
model. The rest are medium/contract notes. Nothing here needs a silly-kicks change — these are
lakehouse-side.

---

## 0. TL;DR + status (updated after the lakehouse triage, 2026-07-10)

1. **F1 — populate `fct_action_values.gk_was_distributing`** (acting-GK-distribution flag; ~all False
   everywhere). **Lakehouse: CONFIRMED real gap** (the enrichment was never wired) → will write a
   spec+plan to populate it (goal-kicks + acting-GK passes, GS/SC first). Until then, the silly-kicks ρ
   retention domain stays **goal-kicks only**.
2. **F2 — the 52% exactly-zero pressure. RESOLVED (no lakehouse code change needed to unblock).** The
   lakehouse 3-method audit confirmed it's a **method artifact**: exact-zero rate is andrienko 46.9% /
   link_zones 79.5% / **bekkers_pi 4.7%**. silly-kicks **pinned `bekkers_pi`** and re-ran the gate:
   non-degenerate terciles, 8–17 occupied deep cells, a **real decreasing monotone deep gradient**
   (relative effect 0.86 WC2022 / 1.05 RM). Optional lakehouse follow-up: audit why andrienko_oval floors.

The gate is now unblocked and run (see `GATE_FINDINGS.md`). The one remaining substantial lakehouse
item is **F1**.

---

## 1. How silly-kicks consumes the marts (so you can reproduce)

The two loaders are committed in silly-kicks `scripts/_loader_databricks.py`:

- **Gate cohort** (`load_xtgk_cohort`): base SPADL from `bronze.spadl_actions`, bridged to the gold
  surrogate `match_key` via `dim_matches.native_match_id`, LEFT JOIN `dev_gold.fct_action_context`
  (pressure + a frame-present proxy) + `dev_gold.fct_shot_xg` (the calibrated per-shot xG reward). All
  three keyed on `(match_key, action_id)`; the action_id join is **coord-exact** (verified `avg |Δstart_x| = 0.0`).
- **ρ retention cohort** (`load_retention_cohort`): `dev_gold.fct_action_values` (geometry / type /
  result / possession / `gk_was_distributing`) LEFT JOIN `dev_gold.fct_action_context` (pressure).

```sql
-- Gate cohort (per provider), abbreviated:
WITH s AS (SELECT * FROM soccer_analytics.bronze.spadl_actions WHERE data_source = :ds),
     d AS (SELECT match_key, native_match_id FROM soccer_analytics.dev_gold.dim_matches WHERE provider = :ds)
SELECT s.game_id, s.period_id, s.action_id, s.time_seconds, s.type_id, s.result_id,
       s.start_x, s.start_y, s.end_x, s.end_y, s.possession_id_heuristic AS possession_id,
       c.pressure_on_actor__andrienko_oval,
       (c.team_shape_n_outfield_players_defending IS NOT NULL) AS frame_present,
       x.xg, x.ood_flag, x.xg_ci_low, x.xg_ci_high
FROM s
LEFT JOIN d ON s.match_id_native = d.native_match_id
LEFT JOIN soccer_analytics.dev_gold.fct_action_context c ON c.match_key = d.match_key AND c.action_id = s.action_id
LEFT JOIN soccer_analytics.dev_gold.fct_shot_xg      x ON x.match_key = d.match_key AND x.action_id = s.action_id;
```

---

## 2. Findings

### 🔴 F1 — `fct_action_values.gk_was_distributing` is unpopulated across every provider

**Symptom.** The materialized "acting keeper is distributing" flag is ~always False, so it cannot
serve as the GK-distribution domain (goal-kicks ∪ acting-GK passes/throw-ins).

**Evidence.**
```sql
SELECT data_source, COUNT(*) n,
       SUM(CASE WHEN gk_was_distributing THEN 1 ELSE 0 END) gk_dist,
       SUM(CASE WHEN gk_role IS NOT NULL  THEN 1 ELSE 0 END) gk_role_nn
FROM soccer_analytics.dev_gold.fct_action_values GROUP BY data_source;
```
| data_source | n | gk_dist | gk_role_nn |
|---|---|---|---|
| statsbomb | 7,066,329 | 44 | 54,405 |
| wyscout | 2,451,525 | 35 | 18,165 |
| **gradientsports** | 88,958 | **0** | 2,382 |
| **skillcorner** | 134,760 | **0** | 804 |
| idsse | 8,429 | 0 | 74 |
| metrica | 6,154 | 0 | 0 |

`gk_role` only carries **defensive** roles (`sweeping` / `shot_stopping`), so it can't substitute.

**Hypothesis.** The acting-team GK resolution (mirror of silly-kicks `acting_gk_from_frames`, TF-13)
that would flag an acting-GK pass isn't materialized into `gk_was_distributing`, and goal-kicks aren't
being OR-ed in either.

**Fix.** Populate `gk_was_distributing = True` for (a) every goal-kick and (b) any pass/throw-in whose
actor is the acting team's GK (from the acting-GK resolver). GS/SC are the priority (0 today).

**Acceptance.** `gk_was_distributing` true-rate ≈ goal-kicks + acting-GK passes (empirically ~50–70
per match for GS/SC vs ~15 goal-kicks/match today).

**Downstream impact.** silly-kicks' ρ retention model is trained on **goal-kicks only** (396 GS / 1186
SC rows). With F1 fixed it would cover the full GK-distribution population (goal-kicks + GK open-play
passes, ~3–4× the rows), matching the frames-based domain and the metric's intended scope.

---

### 🔴 F2 — `pressure_on_actor__andrienko_oval` is exactly `0` for 52% of WC2022 actions

> **Status: RESOLVED (2026-07-10).** Lakehouse 3-method audit → method artifact; silly-kicks pinned
> `bekkers_pi` (4.7% zero) and the gate now runs (§0). Optional lakehouse follow-up: audit andrienko_oval.

**Symptom.** Half of all actions carry pressure **exactly** 0.0; the deep zone is worse. This drives
the global 1/3-quantile cutpoint to 0 and empties the zone-conditional deep-band middle tercile — the
**direct cause of the xT-GK v2 gate STOP** (0 occupied deep cells).

**Evidence (WC2022 / gradientsports, gate-prepared cohort).**
- `pressure == 0` exactly: **46,956 / 89,909 (52.2%)**; `quantile(1/3, 2/3) = [0.0, 7.08]`.
- Deep-zone (grid cols xi∈{0,1}) support by pressure tercile: L1 = 2,257, **L2 = 0**, L3 = 679
  (under zone-conditional terciles — the middle tercile is empty because >2/3 of deep pressures are 0).
- The gradient itself IS real and decreasing at relaxed support (n_min ≤ 20 global): V_lo 0.0056 →
  V_mid 0.0037 → V_hi 0.0017 (relative ≈ 1.1). So the signal exists; the **stratification** is defeated
  by the pressure-zero mass.

**Investigation (the decisive check).** Compare the exact-zero rate across the three materialized
methods:
```sql
SELECT data_source,
       AVG(CASE WHEN pressure_on_actor__andrienko_oval = 0 THEN 1.0 ELSE 0 END) z_andrienko,
       AVG(CASE WHEN pressure_on_actor__link_zones      = 0 THEN 1.0 ELSE 0 END) z_link,
       AVG(CASE WHEN pressure_on_actor__bekkers_pi       = 0 THEN 1.0 ELSE 0 END) z_bekkers
FROM soccer_analytics.dev_gold.fct_action_context WHERE data_source = 'gradientsports' GROUP BY data_source;
```
- If **all three** ~52% zero → it's defender detection / data (few opponents inside any pressure region).
- If **only andrienko_oval** → the oval parameterization is flooring to 0 (too small / mis-scaled).

Note: some of the zero mass is legitimate (genuinely unpressured restarts — silly-kicks coalesces
frame-present null-pressure goal-kicks to 0 by design, ~4,833 on GS). But 52% *overall* (not just
restarts) is the thing to explain.

**Fix (lakehouse-side, if the pressure feature is the issue).** Either correct the andrienko_oval
computation, or expose a pressure measure with a non-degenerate positive tail. **Independently**,
silly-kicks will adopt an Eyestone-escalated methodology fix (treat `pressure == 0` as its own
"unpressured" stratum and tercile only the positive pressures) — but that only helps if the positive
tail is trustworthy, so the data question above must be answered first.

**Acceptance.** Deep-zone actions have ≥2 grid cells with ≥30 support in **all three** pressure
terciles (currently 0). silly-kicks re-runs `validate_xtgk_possession_value.py` to confirm.

**Downstream impact.** Blocks the xT-GK v2 make-or-break gate (STOP today) and any pressure-stratified
analytics keyed on this column.

---

### 🟠 F3 — `fct_shot_xg.ood_flag = 100%` for skillcorner

`2,596 / 2,596` RM shots are flagged OOD (uncertified). The pre-shot xG model is out-of-distribution
on SkillCorner's tracking-derived shot features. Every RM xG-dependent surface (xT-GK v2 reward, PSxG)
inherits the uncertification — the RM gate verdict is provisional by construction. **Ask:** is SC xG
certification on a roadmap? gradientsports is fully certified (`ood_flag = 0`, clean go/no-go).

### 🟠 F4 — `fct_tracking_frames` gold is sparse / likely stale

Coverage: skillcorner 10 matches, metrica 3, idsse 7, **no gradientsports**, only 10 of 108 SC. Per
the owner directive ("action-values + action-context only; tracking-frames deprecated"), this is
expected — but confirm no consumer reads it as complete, and consider marking it deprecated/frozen.

### 🟡 F5 — `spadl_actions.time_seconds` not non-decreasing vs `action_id`

Within a `(game_id, period_id)`, ordering by `action_id` is not strictly time-non-decreasing (the
synthetic-action 0.5-offset pattern, e.g. cross-goal foul-synth). Time-window consumers must sort by
`(game_id, period_id, time_seconds)`. Contract note; broke silly-kicks' `retains` label until
re-sorted. Consider documenting the ordering contract on `spadl_actions`.

### 🟡 F6 — `gk_role` is defensive-only

Values are `sweeping` / `shot_stopping` — no distribution role. Can't stand in for F1.

### 🟡 F7 — `fct_action_context` join contract

`fct_action_context` lacks base SPADL (`result_id`, `team_id`, `possession_id`) and keys on the
`match_key` surrogate. Consumers must (a) bridge `match_key ↔ native_match_id` via `dim_matches`
(`match_key ≠ spadl_actions.game_id/match_id`, 0 direct overlap), and (b) join
`fct_action_values`/`spadl_actions` for the base action columns. Documented handshake — worth pinning
so it doesn't drift.

---

## 3. What silly-kicks relies on (stability contract)

- **`match_key` surrogate + `dim_matches.native_match_id` bridge** to spadl `match_id_native`.
- **Attack-LTR SPADL** in `spadl_actions`/`fct_action_values` (verified: 99.8% of GS shots, 98.8% of
  SC shots, are in the attacking half — good).
- **Period-relative `time_seconds`** (resets each period).
- **`fct_shot_xg.xg` on `(match_key, action_id)`** as the injected calibrated pre-shot reward
  (100% non-null on both cohorts — good).

## 4. Priority

**F1 and F2 are the blockers.** F1 restores the full ρ domain; F2 unblocks the gate. F3–F7 are
medium/contract. When F1/F2 land, silly-kicks re-runs the gate + retrains ρ (both fully wired) and
reports back.
