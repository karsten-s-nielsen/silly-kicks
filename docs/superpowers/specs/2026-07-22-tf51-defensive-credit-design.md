# Per-event defensive credit/debit family (TF-51) — design

**Date:** 2026-07-22
**Target:** silly-kicks 4.57.0 / PR-S128 / ADR-047
**Status:** design, revised after two review rounds + external prior-art integration (arXiv:2606.19931), awaiting owner review
**Source spec:** `docs/superpowers/specs/2026-07-16-soccermatics-pro-future-work-plan.md` §W4 (module 16.3); TODO.md TF-51 row.
**Prerequisite:** `shot_blocked` + `cross_blocked` converter columns (`docs/superpowers/specs/2026-07-22-block-detection-converter-columns-design.md`) — **✅ SHIPPED as silly-kicks 4.56.0 / PR-S127 / ADR-046 (PR #171, squash `3b4403d`, 2026-07-23).** Both are now live `SPADL_COLUMNS` (dtype `"boolean"`, shared `spadl/utils._blocked_flag`, declared `"invariant"` in the ADR-045 reflection registry); TF-51 consumes the real columns (§2.1).

---

## 1. Problem

silly-kicks values the **on-ball event stream** (VAEP) and **measures** pressure
(`pressure_on_actor`), but nothing assigns **signed defensive credit** — the attribution a coach
reasons in ("that press earned the turnover", "that defender got beaten for the chance"). TF-51
adds a per-event defensive credit/debit family: proximity-gated signed values attributed to
individual defenders, **sized by the danger they removed or conceded** (a shot's **xG**, or the
attacker's **xT** at the point of a turnover).

The design is substantially pre-locked by the source spec (§W4). This document resolves the open
design decisions (some revised by a cross-session review — §3) precisely enough to write a
rule-by-rule, red-first implementation plan.

## 2. Repo state — every primitive already exists

| need | primitive | location |
|---|---|---|
| responsible-defender geometry | nearest-opponent kernels (cover-shadow / xS) | `tracking/_cover_shadows.py`, `tracking/_xshot_occurrence.py` |
| action↔frame link | `link_actions_to_frames`, `links=` kwarg | `tracking/linkage` |
| action-LTR reprojection (ADR-028) | `acting_team_attacks_rtl`, `reproject_to_action_ltr` | `tracking/_action_orientation.py` |
| dtype-safe ids (ADR-019) | `ids_match`, `ids_equal`, `same_id` | `silly_kicks/id_compat.py` |
| injected shot xG | `xg_column` named-column port (fail-loud) | precedent `xtgk/_xg_reward.py`, `vaep/labels.py` |
| injected fitted xT | `require_fitted_xt`, `physical_grid`, `values_at_points` | `xthreat/_physical.py` (4.52.0) |
| possession scoping | `add_possessions` | `spadl/utils.py:966` |
| next-touch / receiver | `resolve_next_touch_receiver`, `_resolve_next_touch_positions` | `spadl/utils.py:1267,1299` |
| possession-boundary logic | `retains`, `derive_team_in_possession` | `xtgk/_retention_labels.py`, `tracking/_ball_carrier.py` |
| penalty-box dims | field geometry | `spadl/config` (`spadlconfig`) |
| **blocked shot/cross signal** | **`shot_blocked` + `cross_blocked` columns (prerequisite, §2.1)** or injected `blocked_column` | `SPADL_COLUMNS` / all 8 converters (shipped 4.56.0) |

No new runtime dependency. silly-kicks ships **no xG model** (injected) and consumes a
**caller-fitted** `ExpectedThreat` (injected, `require_fitted_xt`), exactly as `xtgk`/`obso` do.

### 2.1 Prerequisite — the `shot_blocked` + `cross_blocked` converter columns (✅ SHIPPED, silly-kicks 4.56.0 / ADR-046)

Canonical SPADL has **no "blocked" `result_id`** — a blocked shot/cross flattens to `fail`, and
TF-48 `shot_on_target_derived` has no "blocked" class (and is frame-dependent). Provider audits +
real-data probes (SkillCorner + GS via the pining loader) established that the blocked signal is
present-but-dropped in most providers. The prerequisite PR (**now merged, 4.56.0 / PR-S127 / ADR-046**)
surfaces it as two nullable-boolean (`"boolean"`), cross-provider columns emitted by **all 8
converters** via the shared `spadl/utils._blocked_flag` helper (3-valued True / False / `pd.NA`) —
**`shot_blocked`** (real masks on 6/8: GS / Wyscout / StatsBomb / Sportec / Metrica / kloppy) and
**`cross_blocked`** (real masks on GS + Wyscout; StatsBomb deferred to `pd.NA` at n=1, BD-2;
DFL / Metrica / kloppy infeasible → `pd.NA`), `pd.NA` where a provider cannot encode it
(SkillCorner confirmed absent on both tiers; Opta unverified). **`cross_blocked` is scoped to
open-play `cross` only — set-piece (corner / free-kick) crosses are `pd.NA` in v1**. This is a v1
column limitation, not the bravery definition (which is over all crosses), so v1 bravery **exposes**
the set-piece gap (a `NaN` component + a faced-count) rather than dropping it (§9.3). Where the signal exists, providers separate `Blocked`
from `Saved`, so the columns mean an *outfield-defender block, not a keeper save*. Both are declared
`"invariant"` in the ADR-045 reflection registry and are C4-free (converter columns, not aggregators).
Full per-provider feasibility + implementation in
**`docs/superpowers/specs/2026-07-22-block-detection-converter-columns-design.md`**.

TF-51's `shot_block` rule consumes `shot_blocked`; **bravery consumes both** (numerator = blocked
shots + blocked crosses, with the R2-2 unknown-≠-0 guard applied **per final-action type** — a
provider with `cross_blocked` all-`NA` (e.g. DFL / Metrica) yields a shots-only bravery + an explicit
NaN cross component). Both columns are C4-free (converter columns, not aggregators); TF-51 depends on
them but does not re-specify them here.

## 3. Resolved design decisions

Six original decisions plus four revisions from the cross-session review (D1–D4).

| decision | resolution |
|---|---|
| v1 rule scope | **Full family** — all 8 taught signed rules incl. failed-cross-block + failed-marking-through-ball, **+ synchronized final-third pressure + a `shot_block` rule (the bravery numerator) = 10 named rules (§5) + the bravery rollup.** Lane-blocking + pressure-commitment stay **deferred** (§11). |
| **D1 · xT sizing (`xT(origin)`, validated)** | **Extinguished `xT(origin)`** — the attacker's standing threat at the turnover point, extinguished by the defensive action, for the four turnover rules. **Externally validated:** Paper 2 (Bischofberger/Bauer/Baca — our TF-28 DAS group; arXiv:2606.19931) independently derives `xDT = −ΔxT → +xT(origin)` for failed passes and validates it (fault/contribution metrics correlate with transfermarkt value + FIFA defensive-awareness, ~1 SD > action metrics). Rewards defending under pressure (threat prevented). Sizes by the *known* origin, sidestepping B2 (SPADL failed-pass `end` = interception point, not intent — why *denied ΔxT* was rejected). Shot/marking rules stay xG-sized. Opponent-perspective *reverse-xT* ("position won", rewards high pressing) is a deferred **pressing lens** (§11), NOT the default — it diverges from the validated standard and under-values last-ditch defending. |
| **D2 · blocked signal (revised)** | Injected **`blocked_column`** port (default `"shot_blocked"` — the prerequisite column, §2.1). `shot_block` + bravery consume it; absent → `shot_block` does not fire, bravery numerator is empty (warns). The converter column is the default source; the injection override serves Opta (external qualifier maps), lakehouse-sourced blocked, and testing. |
| **D3 · atomic mirror (revised → deferred)** | **Atomic mirror deferred to v2.** The chained rules (possession-scoped resulting-shot + recovery on **renumbered** atomic action_ids) are materially harder than the SK-xT-2 single-action precedent, and defensive credit is naturally SPADL-action-level (no VAEP xfns → no atomic-VAEP consumer in v1). Justified YAGNI. |
| **D4 · bravery (revised + CS-1/CS-2)** | **Event-only, per-team.** `compute_bravery(actions, *, shot_blocked_column, cross_blocked_column)`; needs no frames/xt. **Source** = shots + **all** crosses (32/40); v1 *knows* shots + **open-play** crosses (shipped `cross_blocked` is `NA` on set-piece → SPADL `corner_crossed`/`freekick_crossed`, a column limitation not the definition). Per **R2-2 (unknown ≠ 0/dropped)** v1 emits a **per-type breakdown** — `bravery_shots`, `bravery_open_play_crosses`, `bravery_set_piece_crosses = NaN` + `n_set_piece_crosses_faced` (gap **exposed**, not dropped) — and a **known-domain headline `bravery_pct_known_domain`** (shots + open-play crosses; not deflated by set-piece). Per-**player** % is ill-defined → per-player block **counts** only (not a rate; **shots-only**, TF-1). §9.3. |
| responsible defender | **Nearest opponent within the box-aware threshold at the triggering action's linked frame** (ADR-028 reprojection, ADR-019 id-compat). v1 = nearest-defender; a marking-assignment model is deferred. |
| output shape | **Long-form per-credit table** (`compute_defensive_credits`) **+ per-action `add_*` aggregate** (`add_defensive_credit`). Per-*player* rollup left to consumers. |
| NaN-safety | ADR-003 `@nan_safe_enrichment`; **rule fired but unsizable → NaN-value long-form row + reason; rule did not fire → no row** (ADR-043 "missing ≠ 0"). The per-action **aggregate is always finite** — a genuine 0 for no-credit actions (§9.2). |

## 4. Architecture & module structure

New sub-package **`silly_kicks/tracking/defensive_credit/`**. It ships a per-action `add_*`
aggregator → it is a **tracking action-coupled aggregator** (C4 count **30 → 31**, confirm the
current count at commit-prep — the parallel session may move it), so it lives under `tracking/`,
not as a top-level package.

```
tracking/defensive_credit/
  __init__.py        # thin declarative re-export ONLY (public names)
  _orchestration.py  # compute_defensive_credits / add_defensive_credit / compute_bravery bodies
  _params.py         # DefensiveCreditParams frozen dataclass; DEFENSIVE_CREDIT_RULES closed vocab
  _resolution.py     # resolve_responsible_defenders(...) — nearest opponent(s), ADR-028 + ADR-019
  _rules.py          # one pure function per rule: (ctx) -> credit rows; the rule registry
  _sizing.py         # xg_of_shot(xg_column) + extinguished_xt(xt) ports
  _chaining.py       # possession-scoped resulting-shot + recovery resolvers (add_possessions)
  _bravery.py        # compute_bravery(actions, *, shot_blocked_column, cross_blocked_column) event-only
```

Public functions re-exported via `tracking/__init__.py`.

### 4.1 Public surface

```python
compute_defensive_credits(
    actions, frames, *, xg_column, xt, blocked_column="shot_blocked",
    links=None, params=DefensiveCreditParams()
) -> pd.DataFrame            # long-form: one row per (action, credited player, rule)

add_defensive_credit(
    actions, frames, *, xg_column, xt, blocked_column="shot_blocked",
    links=None, params=DefensiveCreditParams()
) -> pd.DataFrame            # actions + per-action aggregate columns (the C4 +1 aggregator)

compute_bravery(actions, *, shot_blocked_column="shot_blocked", cross_blocked_column="cross_blocked") -> pd.DataFrame  # per-team, event-only: per-type breakdown (bravery_shots / _open_play_crosses / _set_piece_crosses=NaN) + headline bravery_pct_known_domain + n_set_piece_crosses_faced (§9.3)

DefensiveCreditParams           # frozen dataclass
DEFENSIVE_CREDIT_RULES          # closed tuple[str, ...] vocabulary
```

**No `*_xfns` factory** — defensive credit is per-*defender* and gates on the action's own result +
downstream shot outcome (F4 result-leakage, ADR-039/042). It is not a per-acting-team VAEP feature;
**TF-48 shot-goalmouth is the precedent** for a tracking aggregator that ships no xfns. An
**executable absence guard** (auto-discovering, per ADR-039/042) enforces that no
`*defensive_credit*` transformer appears in any default xfn list.

**No atomic mirror in v1** (D3 — deferred).

### 4.2 `DefensiveCreditParams` (all spec-frozen / intent-set, never calibrated)

| field | default | meaning |
|---|---|---|
| `proximity_outside_box_m` | `4.5` | pressure/marking radius outside the penalty box |
| `proximity_inside_box_m` | `3.0` | pressure/marking radius inside the penalty box |
| `synchronized_zone_boundary_x` | `pitch_length / 3` | synchronized pressure fires when the carrier is in **their own defensive third** (action-LTR `x ≤` this) — the *pressing* team's final third (high press); derived from `spadlconfig` |
| `resulting_shot_max_actions` | `10` | forward cap when scanning a possession for the resulting shot |
| `recovery_max_actions` | `3` | forward cap for the recovery after a failed pass |
| `through_ball_delta_xt_min` | `0.02` | ΔxT floor to call a completed pass a "through ball" (provisional) |
| `beaten_1v1_min_shot_xg` | `0.05` | resulting-shot xG floor for a "quality" chance (provisional) |
| `rules` | all | `frozenset[str]` of enabled rules (⊆ `DEFENSIVE_CREDIT_RULES`) |

**Box-awareness derives box + pitch dims from `spadlconfig`** (never hard-coded — 4.48.0 showed
real pitches vary 104/106/101×67; the "pitch dims live in spadlconfig" convention, ADR-021). The
threshold is `proximity_inside_box_m` when the anchor location is inside the attacked penalty area,
else `proximity_outside_box_m`.

**`rules` gating contract:** a rule name in `params.rules` is emitted; one absent from it emits **no
rows** but **stays reachable** (enabling it re-emits). Tested both ways (a disabled rule produces
zero rows; the same fixture with it enabled produces its expected rows).

## 5. Rule catalog

**Ten named rules** (several emit more than one long-form row — `pressure_pass_fail` and
`recovery_double_credit` each emit a `+`/`−` pair, `failed_cross_block` a `−`/`+` pair, and
`synchronized_final_third_pressure` one `+` per within-threshold defender). Every long-form row is
`(game_id, action_id, player_id, team_id, rule, signed_value, anchor_type, frame_id, sizing)`.

| rule | anchor | party & sign | sizing |
|---|---|---|---|
| `pressure_on_missed_shot` | shot, **off-target** | **+** nearest def ≤thr | shot xG |
| `failed_pressure_shot_on_target` | shot, **on-target/goal** | **−** nearest def ≤thr | shot xG |
| `shot_block` | shot, **blocked** (`blocked_column`) | **+** blocker | shot xG |
| `pressure_pass_fail` | pass, fail | **+** nearest def ≤thr / **−** passer | xT(origin) |
| `recovery_double_credit` | failed pass → own recovery in window | **+** recoverer / **−** passer (again) | xT(origin) |
| `synchronized_final_third_pressure` | failed pass in the carrier's **defensive** third (pressing team's final third) | **+** each within-thr def **beyond the nearest** | xT(origin) |
| `forced_bad_touch` | bad_touch + def ≤thr | **+** presser | xT(origin) |
| `failed_cross_block` | cross → receipt (→ shot) | **−** nearest def at receipt / **+** shot-blocker | resulting-shot xG |
| `failed_marking_through_ball` | high-ΔxT completed pass → shot | **−** responsible def at pass moment | resulting-shot xG |
| `beaten_1v1` | successful take-on → quality shot | **−** beaten def ≤`proximity_outside_box_m` | resulting-shot xG |

**Shot-outcome partition.** The three shot rules are a **mutually-exclusive partition** of shot
outcomes: **blocked** (`blocked_column` true) → `shot_block`; else **on-target or goal** →
`failed_pressure_shot_on_target`; else **off-target** → `pressure_on_missed_shot`. On/off-target
uses the provider result/outcome codes, falling back to TF-48 `shot_on_target_derived` where the
provider does not distinguish (a plan detail — the per-provider outcome vocabulary is resolved in
the plan). Blocked always takes precedence (a blocked shot is neither "on-" nor "off-target").

Notes on the three judgment calls (owner-approved):
1. **Synchronized de-dup.** `synchronized_final_third_pressure` credits within-threshold defenders
   **beyond the nearest** — the nearest already earns `pressure_pass_fail` — so the union is that
   **every** within-4.5 m defender receives exactly one `+` on a final-third failed pass.
2. **Bad-touch sizing.** `xT(origin)` at the bad-touch location — the standing threat the attacker forfeited; `sizing == "xt"`.
3. **Through-ball identification.** A **completed pass with ΔxT ≥ `through_ball_delta_xt_min` that
   leads to a shot in the same possession** — a param, not a hard line-break test (TF-4/TF-32
   line-break gating is a v2 refinement).

**Blocked crosses are bravery-only (TF-1).** `shot_block` credits the individual shot-blocker (sized
by shot xG); there is **no symmetric `cross_block` credit rule** in v1. A blocked cross has no clean
sizing — no xG, a *blocked* cross has no resulting shot, its destination is B2-affected, and the
crosser's origin xT is low. So a cross-blocker earns **team bravery** credit (§9.3) but **no
individual credit** — a deliberate, documented v1 asymmetry (not left implicit). The per-defender
`cross_block` rule + a cross-threat sizing lands in the v2 DPA work (which values crosses by `xDT`).

### 5.1 Sign convention & the passer double-debit

`+` = the defender *removed* danger; `−` = the defender *conceded* it. `pressure_pass_fail` and
`recovery_double_credit` also emit a **`−` on the attacker** (the passer), so the passer is debited
**twice** across the pressure + recovery points — the "double credit" the source spec names. **Doc
note for per-player rollups:** one turnover debits the same passer in two rows; a
`groupby(player_id).sum()` counts both by design — document it so consumers do not read it as a bug.

## 6. Responsible-defender resolution (`_resolution.py`)

`resolve_responsible_defenders(...)` returns the nearest opponent(s) (team_id ≠ acting team,
compared with `ids_differ` — never raw `!=`) within the box-aware threshold at the triggering
action's linked frame:
1. Reference point in **action-LTR** — `start` (pressure/dribble/bad_touch), passer location (pass
   rules), or `end`/receipt (cross); frame player positions reprojected via `reproject_to_action_ltr`
   (ADR-028) so anchor and frame share one reference.
2. Rank opponents by distance; keep those within the threshold. Modes: nearest (single presser /
   beaten defender), all-within (synchronized), all-within-beyond-nearest (synchronized de-dup).
3. **Empty within-threshold set → the rule does not fire (no row).**

**Batching / perf (M3):** the resolver operates over the **whole action set in one pass**, sharing
**one** `link_actions_to_frames` call (or the caller's `links=`) and **one** reprojection pass
across all rules — vectorized `np.select`-style dispatch, never a per-action Python loop. A
**structural call-count budget** (§12) pins `link_actions_to_frames` to one call and the
reprojection to one pass per invocation.

**`shot_block` blocker (v1 limitation):** the blocker is resolved as the nearest opponent to the
shot origin within threshold — a **lane-free approximation**. Kept explicit in the docstring +
fixtures; a lane-intersection blocker model (reusing `_cover_shadows`) is v2.

## 7. Sizing ports (`_sizing.py`)

- **xG** — `xg_of_shot(shot_action, xg_column)`: reads the injected per-shot xG column; **fail-loud
  if `xg_column` is absent** (the `xtgk/_xg_reward` pattern). xG is computed **pre-block**, so the
  injected column carries a value for *blocked* shots (`shot_block` sizing depends on it). A shot
  present but with NaN xG → NaN long-form row (fired, unsizable).
- **Extinguished xT** — `extinguished_xt(points, xt)`: `values_at_points(xt, xs, ys)` — the
  attacker's threat at the turnover origin(s), looked up directly on the injected fitted surface
  (action-LTR, NaN-tolerant); `require_fitted_xt(xt)` guards an unfitted model. **No grid is built**
  — `values_at_points` samples points on the model directly (the earlier `physical_grid(…)` wrapper
  was unnecessary; P-5). Used by `pressure_pass_fail` (passer location), `recovery_double_credit`
  (recovery location; the paired `−passer` at the passer location), `synchronized_final_third_pressure`
  (carrier location), `forced_bad_touch` (bad-touch location).

**Why `xT(origin)` (D1, validated).** Paper 2 (arXiv:2606.19931, our TF-28 DAS group) derives the
identical sizing — `xDT = −ΔxT`, which for a failed pass (`xT_end=0`) is `+xT(origin)` — and
validates it on WC2022 + German leagues. It sizes the credit by the attacker's threat *extinguished*
at the turnover, sidestepping B2 (SPADL failed-pass `end` is the interception point, not the target
— the reason *denied ΔxT* was rejected). **Deferred pressing lens (§11):** opponent-perspective
reverse-xT `xT(L−x, W−y)` ("position won") rewards high pressing but diverges from the validated
standard and under-values last-ditch defending — an optional `sizing` param, not the default.
**Note on `synchronized_final_third_pressure` (TF-3):** under `xT(origin)` a high press deep in the
opponent's half is sized *low* (little threat prevented), so this rule is **attribution-valuable, not
magnitude-valuable** — its contribution is the collective multi-defender credit (who pressed),
consistent with Paper 2's proximity distribution of the small deep `xDT`. It is the **natural first
consumer of the deferred pressing lens (§11)** — a pressing-focused analysis would apply reverse-xT,
which most affects this rule.

## 8. Chaining (`_chaining.py`)

Both chained resolutions are **possession-scoped** via `add_possessions`, boundary-guarded on
`(game_id, period_id, possession_id)`:
- **Resulting shot** (cross-block, through-ball, beaten-1v1): the first shot by the **attacking**
  team in the anchor's possession, within `resulting_shot_max_actions`. None → the rule does not
  fire.
- **Recovery** (double-credit): the first defending-team ball-regain within `recovery_max_actions`
  actions of the failed pass — **typically but not necessarily the next possession's opening
  action** (a loose-ball scramble may intervene; the cap bounds the scan). NaN-team rows skipped
  (ADR-027). None within the cap → only the base `pressure_pass_fail` fires.

## 9. Output shapes

### 9.1 Long-form primitive — `compute_defensive_credits`

One row per *(triggering action, credited player, rule)*. Columns: `game_id, action_id,
player_id, team_id, rule, signed_value, anchor_type, frame_id, sizing` (`sizing` ∈ `{xg, xt}`).
`rule` ∈ `DEFENSIVE_CREDIT_RULES` (closed vocabulary + closed-set guard, the `DAS_SOURCE_VALUES`
pattern). **NaN discipline (ADR-043 "missing ≠ 0"):** rule fired but unsizable (shot lacks xG, no
fitted-xT value) → row with `signed_value = NaN` + the `sizing` reason; rule did not fire (no
linked frame, no defender within threshold, no resulting shot) → **no row**.

### 9.2 Per-action aggregate — `add_defensive_credit`

Returns `actions` + (pure, ADR-033): `defensive_credit_net`, `defensive_credit_plus`,
`defensive_credit_minus` (skipna finite sums), `n_defensive_credits` (Int64 count), rolled up to
the triggering action. **Scoped to the DEFENDING-team rows only (R2-1):** the aggregate answers
"how much defensive credit did the defending team earn on this action", so the attacker-side debits
(the `−passer` rows, which belong to the *acting* team) are **excluded** from the aggregate and
live only in the long-form for per-player rollup. This is essential — the paired rules emit a
defending-team `+` and an acting-team `−` on the *same* `action_id` (`pressure_pass_fail` +presser /
−passer, both sized at the passer origin → equal magnitude; `recovery_double_credit` +recoverer at the
*recovery* location / −passer at the *passer* origin → **different locations, so different
magnitudes** — §7), so an unscoped cross-team net would mix the two teams' signs and wash out the
defending credit; **defending-team scoping (dropping the acting-team −passer rows) keeps `_net` a real
signal regardless of whether the pair happens to be equal-magnitude** (and live without relying on shot
rules). `failed_cross_block` (−def/+blocker, both defending) *can* net ≈0 when both fire —
that is a **correct** neutral team outcome (conceded the cross, blocked the shot); the individual
attributions remain in the long-form. **Always finite** — a genuine 0 for no-credit actions
(`net/plus/minus = 0.0`, `n = 0`); NaN unsizability is confined to the long-form `signed_value`.
The `(net, n)` pair distinguishes **no credit `(0, 0)`** from **fired-but-unsizable `(0, >0)`** —
liveness-safe (non-NaN by construction; non-constant on the multi-domain fixture, which covers all
rules for reachability). Per-*player* rollup = consumers group the long-form by `player_id`.

### 9.3 Bravery — `compute_bravery(actions, *, shot_blocked_column="shot_blocked", cross_blocked_column="cross_blocked")`

**Event-only** (no frames, no xt), **per-team grain only.**

**Source definition vs v1 knowledge.** The source (Sumpter / Tigres Femenil, source spec §W4:
"% of the opposition's **total** final actions — **shots + crosses** — that were blocked"; worked
example 32/40 = 80 %) is over **shots + ALL crosses**. v1 can only *know* the block-status of shots +
**open-play** crosses — the shipped `cross_blocked` is `pd.NA` on set-piece (corner / free-kick →
SPADL `corner_crossed` / `freekick_crossed`) crosses (§2.1). **This is a v1 column limitation, NOT the
definition.** So — per the spec's own **R2-2 discipline (unknown ≠ 0 / dropped)** — v1 **exposes** the
set-piece gap rather than silently excluding it (set-piece box blocks are the *archetype* of "bravery";
dropping them would be the "covered everything" illusion the no-silent-caps rule forbids).

**Per-type breakdown (the R2-2 shape), per (team, game):**
- `bravery_shots` = blocked shots / shots faced (from `shot_blocked_column`).
- `bravery_open_play_crosses` = blocked open-play crosses / open-play (`cross`-type) crosses faced
  (from `cross_blocked_column`).
- `bravery_set_piece_crosses` = **`NaN` in v1** — block-status universally unknown (`cross_blocked` is
  `NA` on `corner_crossed` / `freekick_crossed`); its faced count **`n_set_piece_crosses_faced`** is
  emitted so the gap is **visible, not hidden**. Closing it needs a set-piece `cross_blocked` (v2 —
  folds into the deferred individual `cross_block` rule, §11).
- Headline **`bravery_pct_known_domain`** = blocked(shots + open-play crosses) / (shots + open-play crosses) — the
  **known-domain** rate, **explicitly excluding set-piece crosses** (surfaced separately above). Kept
  over the known domain so it is **honest (not deflated) and usable**: a strict "overall rate = `NaN`
  because a whole type is unknown" would be `NaN` on nearly every match (every match has corners) and
  useless. The exclusion is **documented + paired with `n_set_piece_crosses_faced`**, never silent.
  **The `_known_domain` suffix is deliberate (CS naming call):** v1's headline is a *genuinely
  different number* from the source metric (source = shots + all crosses; v1 = shots + open-play
  crosses), so a bare `bravery_pct` would overclaim. It matters most because the lakehouse
  **materializes columns individually** — the headline travels into a mart *without* its
  `bravery_set_piece_crosses` / `n_set_piece_crosses_faced` siblings, and a bare name read in isolation
  would reintroduce the exact "covered everything" illusion the per-type exposure prevents. The suffix
  also encodes the R2-2 **provider-variability** of the known domain (shots-only for a `cross_blocked`
  all-`NA` provider vs shots + open-play crosses otherwise) and follows the repo's encode-semantics-in-
  the-name convention (`pressure_on_actor__bekkers_pi`, `pitch_control_at_target__spearman`).

**Cross-type identification is from the SPADL action type** (`cross` = open-play;
`corner_crossed` / `freekick_crossed` = set-piece — distinct SPADL types, no separate flag), **NOT
from `cross_blocked` being non-`NA`** (else a genuinely-unblocked open-play cross with
`cross_blocked == False` would be miscounted).

**Unknown ≠ 0 %, per type (R2-2):** if a known-domain blocked column is **absent OR all-`NA` for a
(team, game)**, that type's rate is **`NaN`** — a provider with `cross_blocked` all-`NA`
(DFL / Metrica / kloppy-gateway) yields `bravery_open_play_crosses = NaN` + a shots-only headline; a
provider where *both* shot and cross are all-`NA` (SkillCorner / Opta) yields `bravery_pct_known_domain = NaN` +
a warning — never a fabricated 0 %.

**Per-player counts are separate — two reconciliation gaps (TF-2, CS-2).** Per-**player** block
*counts* are **not** returned here (a player does not face all opponent final-actions → a per-player
rate is undefined); they come from a consumer `groupby(player_id)` over the long-form `shot_block`
rows, **tracking-gated** (blocker attribution needs frames). **Σ per-player blocks ≤ team block
numerator** for **two** reasons, both documented so a consumer is not confused: **(1)** a `shot_block`
row fires only when a blocker resolves within threshold at a linked frame → **frame-unresolvable shot
blocks count for the team but for no player**; **(2)** **every blocked *cross* counts for the team but
for no player** — there is no per-defender `cross_block` rule in v1 (TF-1, §5), so per-player counts
are **shots-only** while the team open-play-cross numerator is not. Gap (2) is typically the larger.

## 10. Safety, purity, atomic mirror

- **NaN-safety (ADR-003):** `@nan_safe_enrichment` on `add_defensive_credit`; NaN identifier
  columns route to no-credit; auto-gated by `tests/test_enrichment_nan_safety.py`.
- **Purity (ADR-033):** `add_defensive_credit` returns a new object, mutates no input; registered
  in `PURITY_ENTRIES`.
- **Atomic mirror — deferred to v2 (D3).** Possession-scoped chaining on renumbered atomic
  action_ids is materially harder than the SK-xT-2 single-action precedent, and there is no
  atomic-VAEP consumer (no xfns). If a consumer needs it, a v2 cycle designs the atomic chaining
  explicitly.

## 11. Deferred (v2)

- **Atomic mirror** (D3) — per-defender credit on the atomic decomposition + chained-rule mapping.
- **`shot_blocked` / `cross_blocked` for Opta** (verify the qualifier + team attribution vs a live
  feed) — currently `pd.NA`. SkillCorner is **confirmed absent on both tiers** (no event-stream path),
  permanently `pd.NA` for both columns.
- **Reverse-xT "position won" pressing lens** (D1 v2) — an optional opponent-perspective
  `xT(L−x, W−y)` sizing alongside the validated `xT(origin)` default, for coaches who want to reward
  high pressing (winning the ball near the opponent's goal) over the danger prevented. Diverges from
  the Paper 2 standard; a selectable `sizing` param, never the default.
- **Passing-lane blocking credit** — geometric lane obstruction without an event (`_cover_shadows`).
- **Pressure-commitment (no-deceleration)** cue — the PSG/Luis Enrique commitment dimension.
- **Role-conditioned responsibility / DPA attribution (v2 — Paper 2 blueprint)** — Bischofberger et
  al. (arXiv:2606.19931; code `github.com/jonas-bischofberger/defensive-network`) is the buildable,
  validated version of the marking/responsibility model: value every opponent pass by `xDT`, define a
  Defensive Pressure Area, distribute value to defenders by proximity (`R=(r−d)/r`, `r=5 m`), and
  average by tactical role (formation template-match + Hungarian, 20-role taxonomy) → "expected
  involvement" = responsibility (blames the out-of-position defender *by role*, even far from the
  ball). Natural v2 for `failed_marking_through_ball` + a **soft multi-defender distribution**
  alternative to v1's nearest-defender attribution (also dissolves R2-1's net-cancellation differently
  — credit spread by proximity, not a paired +/− on one anchor). New primitives it needs:
  **formation/role detection** (related to but distinct from TF-39 `shape_graph` / TF-31 `team_shape`)
  + a **failed-pass expected-receiver** (Power et al. 2017; our `resolve_next_touch_receiver` handles
  completed passes only). **Pass-only + aggregate-KPI grain** — it *complements* TF-51's per-event
  coaching table, does not replace it. Its own v2 spec (scope the port by reading the code first).
- **Lakehouse defensive-KPI mart (roadmap — lakehouse-side, not silly-kicks)** — Paper 2's fault /
  valued-responsibility per-90 / per-pass are *validated player-season KPIs* (correlate with market
  value); a natural lakehouse mart (VAEP/xT-conceded exists; missing = DPA attribution +
  formation-role detection), distinct from TF-51's per-event table.
- **Lane-geometry `shot_block` blocker** — replaces the nearest-to-origin approximation.
- **Line-break-gated through-ball** — TF-4/TF-32 test instead of the ΔxT threshold.
- **Individual `cross_block` credit rule (v2, TF-1)** — a per-defender credit for the cross-blocker
  (v1 has team bravery only), gated on a cross-threat sizing model (no clean v1 sizing — no xG, no
  resulting shot, B2 destination, low crosser-origin xT); folds into the v2 DPA work (crosses valued
  by `xDT`).

## 12. Validation plan (red-first / TDD)

The plan **sequences each rule red-first**: write the near-miss + hit fixture → confirm **red** →
implement the pure rule function → **green**, with the every-rule-reachable meta-assertion as the
backstop. The spec fixes *what* is asserted; the plan writes the tests before the implementations.

- **Rule-by-rule ground-truth fixtures** — each of the 10 rules: a synthetic (action + frame) →
  known signed credit, **from both sides** (fires with the expected value; a discriminating
  near-miss — defender just *outside* the threshold, or no resulting shot in the possession —
  produces **no row**), per the non-vacuity discipline.
- **Mirror-invariance gate (ADR-028)** — the same physical situation as a home and an away action →
  **identical** action-LTR credit; **asymmetric + extreme** fixture (a y-symmetric fixture passes
  vacuously).
- **Closed-vocabulary + conservation** — every emitted `rule` ∈ `DEFENSIVE_CREDIT_RULES`; long-form
  **row-count conservation**; a meta-assertion that **every rule is reachable** (no dead rule).
- **`rules`-gating** — a disabled rule emits zero rows; enabling it re-emits (§4.2).
- **Sizing regression** — a fixture that **demonstrates the `xT(origin)` "threat prevented" sizing**:
  a turnover forced when the attacker was in a dangerous position (high `xT(origin)`, near the
  defending team's goal) scores markedly higher than the same turnover forced deep in the attacker's
  own half (D1, the validated Paper 2 property). An **asymmetric** fixture also pins that the *origin*
  (not a mirrored point) is sampled — a reflection would change the value.
- **Auto-gates via registration** — aggregator-column **liveness** (non-NaN + non-constant),
  **purity**, **id-dtype-invariance** (auto-enumerated). No xfns → no dup-`action_id` xfns gate.
- **Structural perf budget (M3)** — a call-count spy pinning `link_actions_to_frames` to **one call
  per invocation** (`tests/_perf_structural.py` + `*_perf_budget.py`; no wall-clock). No grid build —
  `values_at_points` samples points directly on the fitted model (P-5); batching the per-point `xT`
  lookups into one `values_at_points` call is an optional micro-optimization, not a structural budget.
- **Cross-provider `cross_blocked` ⊆ cross-type invariant (CS-1, the seam the bravery denominator
  rests on)** — a converter-only, cross-provider contract test: for **every** converter, on **every**
  emitted actions frame, `cross_blocked` non-`NA` ⊆ (`type == cross`) — i.e. `cross_blocked` is
  `pd.NA` on every `corner_crossed` / `freekick_crossed` action. Verified true **by construction** on
  the two real emitters today (GS `_open_play_cross` excludes set-piece codes `("F","C")`; Wyscout's
  `cross_blocked` applicable mask is the *identical* `(PASS & CROSS)` predicate as the SPADL `cross`
  assignment), but locked as a **machine-checked invariant** so a future converter (or a change to an
  existing one) cannot silently break the open-play-crosses denominator. **Lives in the block-detection
  contract suite** (`tests/spadl/test_block_detection_contract.py`, the already-shipped prerequisite
  file) — it is a property of the converter columns, needs no frames / xt / TF-51 code, but **ships as
  part of the TF-51 PR** (owner decision: keep it in TF-51 rather than a separate pre-TF-51 follow-up),
  since TF-51's bravery denominator is the consumer that motivates it.
- **Bravery worked-example** — the Tigres Femenil **32/40 = 80 %** per-team fixture (headline
  `bravery_pct_known_domain` over the known domain), **plus set-piece-exposure assertions**: a fixture with faced
  set-piece crosses (`corner_crossed` / `freekick_crossed`, `cross_blocked == NA`) asserts
  `bravery_set_piece_crosses` is **`NaN`** (never 0) and `n_set_piece_crosses_faced` is the true count
  (the gap is exposed, not dropped — R2-2), and that the headline `bravery_pct_known_domain` is **unchanged** by the
  presence of set-piece crosses (they are not in its denominator).
- **Owner-gated e2e** — on a real match (GS + a fitted xT + an injected xG + the `shot_blocked` /
  `cross_blocked` columns) asserting the family runs end-to-end and produces sane sign/magnitude
  distributions (the GS goal-capture / sportec-playeval owner-gated-e2e precedent).
- **SkillCorner-native construct-validity cross-check (owner-run, reported-not-gated)** — a separate
  `scripts/validate_defensive_credit_vs_skillcorner.py` harness compares TF-51's *derived*
  pressure/beaten/recovery credit against SkillCorner Game Intelligence's *native* labels
  (`on_ball_engagement` pressing/pressure, `beaten_by_possession`, `direct/indirect_regain`,
  `pressing_chain`) on RM matches, via a **validation-only** Game-Intelligence reader in `scripts/`.
  The TF-51 **library stays 100% provider-agnostic** (no SkillCorner import, no special-casing) — this
  is the general "validate a derived metric against the best labelled reference" pattern (the
  `validate_xcross_causal.py` / `validate_xtgk_v2.py` precedent), not a SkillCorner accommodation.
  **Honest caveat (recorded in the harness):** SkillCorner's labels are themselves *derived*, so this
  is a **derived-vs-derived** cross-check — agreement is *corroborative*, not ground truth.

## 13. Attribution & C4

- **`NOTICE`** — Sumpter *Soccermatics Pro* module 16.3 (coach-consulted credit rules);
  **Bischofberger/Bauer/Baca, "Blame is easier than praise" (arXiv:2606.19931) — the published,
  validated precedent for the D1 `xT(origin)` sizing (`xDT = −ΔxT`), consistent with the existing
  TF-28 accessible-space/DAS citation;** MSC-corpus additions — Tigres Femenil "bravery" metric, RB
  Salzburg 4-action pressing taxonomy as docstring-grade vocabulary. Per-feature docstrings cross-link
  `See NOTICE …`.
- **C4** — action-coupled aggregator count **30 → 31** (`add_defensive_credit`); **confirm the
  current count at commit-prep** (the parallel session may move it). Update `architecture.html`,
  the liveness / purity / id-dtype registries. The `shot_blocked` + `cross_blocked` prerequisite is **C4-free**.
- **Retrain** — in **no** default xfn list → **no VAEP retrain trigger**. New aggregator only.

## 14. Open questions (carry into the plan, none blocking)

- `through_ball_delta_xt_min` / `beaten_1v1_min_shot_xg` are **provisional** intent-set defaults
  (frozen params, not calibrated).
- Bravery: the **source** defines it over opponent **shots + all crosses** (course worked example
  32/40); v1's headline `bravery_pct_known_domain` is over the **known domain** (shots + open-play crosses) because
  `cross_blocked` is `NA` on set-piece crosses — a v1 column limitation, so the set-piece gap is
  **exposed** (a `NaN` `bravery_set_piece_crosses` + `n_set_piece_crosses_faced`), **not** silently
  dropped (§9.3). Blocked *clearances* stay out (course).
- `xT(origin)` sizing uses the turnover *origin/location* (well-defined), so the failed-pass `end`
  semantics no longer gate it; but the **through-ball ΔxT identification** uses a *completed* pass's
  `end` (valid) and the **recovery location** comes from the regain action — verify both resolve
  sanely per provider in the plan.
