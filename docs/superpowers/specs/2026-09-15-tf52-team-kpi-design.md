# TF-52 — Event-only team-KPI module (`silly_kicks.team_metrics`)

**Date:** 2026-09-15
**Feature:** TF-52 (On-Deck; source: `docs/superpowers/specs/2026-07-16-soccermatics-pro-future-work-plan.md` §T1)
**Status:** Design — awaiting independent review, then implementation plan.
**Scope class:** Architectural (new top-level package, new public API, new C4 container).
**Author:** silly-kicks session (TF-52 cycle).

---

## 1. Motivation & context

silly-kicks ships zero team-level match KPIs today. TF-52 adds an **event-only** team-match KPI
module — the Twelve match-report glossary plus the event-based subset of the Tigres/Clemson/Coventry
practitioner set — as a single new package on the per-`(game_id, team_id)` grain. It is coach-facing,
requires no tracking, retrains nothing, and adds no action-coupled aggregator.

The authoritative KPI definitions and worked values live in the future-work plan (§T1); this spec
pins them into an implementable contract, resolves the definitional ambiguities, and specifies a
**public, reproducible reliability artifact** that characterises which KPIs are stable signal vs
match-to-match noise.

### 1.1 Relationship to TF-53 and the possession seam

TF-52 and TF-53 (match-outcome/xPoints) are **independent** — neither creates a primitive the other
needs. The possession primitive that TF-53's future possession-aware option would consume already
exists as public `spadl.add_possessions`; TF-52 **consumes** it (never forks it), exactly as TF-53
would. TF-52 was sequenced first purely on value grounds, not dependency.

---

## 2. Scope

### 2.1 In scope — v1 (Twelve glossary) + the event-based v2 items

**v1 — Twelve match-report glossary:** PPDA, defensive intensity, field tilt %, pass tempo, the three
line heights (defensive-action / recovery / turnover), time-to-defensive-action, time-to-recovery,
recoveries + within-Ns %, possessions-retained-after-Ns, the conversion chain, final-third entries,
long-ball %, shots + high-opportunity shots (injected xG), and the parameterized "within N s after
recovery" companion family.

**v2 — event-based practitioner items:** counter-press windows (`seconds=` XOR `passes=` with named
presets), post-regain security metrics, the 6-state build-up outcome taxonomy, breakout-by-channel,
switch-of-play-conditioned press success.

### 2.2 Out of scope — excluded with reason (surfaced, not silently dropped)

| Excluded item | Reason |
|---|---|
| Structural interception-height taxonomy (line-relative) | Needs opponent player-line positions → **tracking**. The coordinate-based half is already covered by v1's recovery/turnover line heights, so the plan's "ship both" is satisfied on the event-only side. Reserved as a future tracking item. **Owner-ratified 2026-09-15.** |
| Aerial first→second-ball chaining | **SPADL has no aerial-duel type** (same limit TF-55 documented); not cleanly representable in canonical event SPADL. |
| "Bravery" blocked-final-actions % | **Out of `team_metrics` scope.** Already ships as `compute_bravery` (TF-51, in `silly_kicks.tracking`), so it is neither re-implemented nor imported here — `team_metrics` is event-only and must not import `tracking`. Consumers obtain bravery from the existing surface. |
| Guardiola counter-press gate | Plan marks it **capture-only** (a context-feature candidate), not a computed KPI. |
| Compactness-recovery time | Consumes `compute_team_shape` → **tracking**, not event-only. |
| Block-classification anchors | Consume `defensive_line_height` → **tracking**; also threshold *defaults*, not a computed KPI. |
| Any **per-provider parameter tune** | ADR-009: the reliability study may *recommend* one, but applying it is a separate gated PR — never this cycle. |

---

## 3. Architecture

### 3.1 Package layout (mirrors the sibling `compute_*` packages)

New top-level package `silly_kicks/team_metrics/`, following `territory` / `shot_stopping` / `duels`:

```
silly_kicks/team_metrics/
  __init__.py          # public exports
  _compute.py          # compute_team_kpis orchestrator
  _config.py           # TeamKpiParams (frozen; .default/.for_provider/.is_default; empty override map)
  _columns.py          # TEAM_KPI_COLUMNS + metric-column groups + source/preset constants
  _report.py           # TeamKpiReport (conservation census, CI-asserted)
  _possession.py       # derived possession-foundation layer (over spadl.add_possessions)
  _orientation.py      # reflect the opponent's rows into a team's action-LTR frame (opponent-relative KPIs)
  _pressing.py         # PPDA, defensive intensity, time-to-*, recoveries, counter-press windows
  _progression.py      # field tilt, pass tempo, long-ball, line heights, conversion chain, high-opp shots, breakout-by-channel
  _buildup.py          # build-up outcome taxonomy, post-regain security, switch-conditioned press
```

### 3.2 Public API

```python
compute_team_kpis(
    actions: pd.DataFrame,
    *,
    xg_column: str | None = None,
    params: TeamKpiParams = TeamKpiParams.default(),
) -> tuple[pd.DataFrame, TeamKpiReport]
```

- One row per `(game_id, team_id)` — two rows per match. Opponent-relative KPIs (PPDA, field tilt)
  are computed over the whole match and emitted per team.
- Canonical-id grouping (ADR-019 via `id_compat`), **raw** ids emitted.
- `xg_column` optional — only the high-opportunity-shots KPI reads it; absent → that column is NaN,
  everything else unaffected.
- **Single orchestrator** (chosen over multiple public `compute_*` per family): the deliverable is
  the team-match report row; callers want one shape; internal family helpers keep files focused.
  Rejected alternative in §12.

### 3.3 Possession foundation (the load-bearing seam)

`_possession.py` calls **`spadl.add_possessions` once** (never a forked possession definition —
Chesterton's Fence) and derives the shared layer every family reuses:

- possession **spells** (`possession_id` runs),
- **transition events**: recovery = regain (start of a new own possession), turnover = loss (end of
  an own possession),
- **team-in-possession** per possession,
- **possession-minutes** per team (summed spell durations from `time_seconds`, period-relative per
  ADR-017).

`TeamKpiParams` surfaces the `add_possessions` convention: its fields `possession_max_gap_seconds`
and `possession_retain_on_set_pieces` **map to** the real kwargs `add_possessions(max_gap_seconds=…,
retain_on_set_pieces=…)` (the prefix disambiguates them in the flat params object). The three
precision opt-ins (`merge_brief_opposing_actions`, `brief_window_seconds`,
`defensive_transition_types`) are intentionally **not** surfaced in v1 — they stay at their
`add_possessions` defaults — so the convention is tunable on the two knobs that matter but
single-sourced.

### 3.4 Hexagonal posture

- Event-only; imports `spadl` + `id_compat` + `silly_kicks.reflection` + numpy/pandas **only**, never
  `tracking` (AST import-allowlist gate `tests/team_metrics/test_import_allowlist.py`, like the
  siblings). `reflection` is required by the §6.1 opponent-relative-KPI orientation.
- A `compute_*`, **not** an `add_*`; no `*_xfns`; in **no** default xfn list.
- `feature_glossary` entries for every emitted column; `NOTICE` attribution.
- **+1 C4 container**; action-coupled aggregator count unchanged (33). Nothing imports `team_metrics`.

---

## 4. KPI catalog

All pitch geometry reuses `spadlconfig` constants (final third, penalty area) per ADR-050 — no
hardcoded coordinates. Worked values are the plan's Twelve match-report anchors; they are
**plausibility/order anchors on real data**, not values reproducible without that report's raw match
(see §9.1).

### 4.1 Pressing / defensive — `_pressing.py`

- **PPDA** = opponent passes ÷ our defensive actions, both inside the **opponent's defensive 60%** of
  the pitch. Defensive actions = the pinned SPADL set `{tackle, interception, foul}` (exact set in
  `TeamKpiParams.defensive_action_types`; SPADL has no separate "challenge" type — `tackle` covers
  it). Zone param `ppda_zone_fraction=0.6`. Documented
  variant: the Trainor "exclude the pressing team's own 40%" framing is equivalent; the alternative
  40% cut is noted, not shipped. **Undefined (0 defensive actions) → NaN** (§8). Anchor 4.17.
- **Defensive intensity** = defensive actions per **minute out of possession** (7.16). Reads
  possession-minutes from `_possession.py`.
- **Time to defensive action** / **time to recovery** = mean seconds from a possession loss to the
  first defensive action / to regain (7.30 / 7.69). A loss never regained (period end) → that spell
  is **excluded from the mean** (§8).
- **Recoveries + within-Ns %** = recovery count + fraction within `counterpress_seconds` (default
  5.0) of the loss (14%).
- **Counter-press window (v2)** = generalisation of the above: window is `seconds=` XOR `passes=`,
  with a documented **preset registry** — Barcelona ~6 s, Coventry 5 s, RB Leipzig 10 s, Hammarby
  5 s, **Tigres "The Hunt" = ≤3 passes**. `seconds` XOR `passes` enforced (a `post_init`-style check;
  both-set is an error). Emits regain-within-window count + rate.

### 4.2 Possession / tempo — `_progression.py`

- **Field tilt %** = our final-third open-play touches ÷ (our + opponent final-third open-play
  touches) (63%). `final_third_boundary` from `spadlconfig`. Documented variant: possession-time-based
  field tilt (noted, not shipped).
- **Pass tempo** = passes per **minute of possession** (19.97).
- **Long-ball %** = own-half passes travelling > `long_ball_distance_m` (32.0) ÷ own-half passes.
- **Possessions retained after N s** = fraction of a team's **open-play possessions** whose
  **duration ≥ `retained_after_seconds` (5.0)**, where a possession's duration = `time_seconds` of its
  last action − its first action (period-relative). Numerator = open-play possessions with duration
  ≥ N s; denominator = open-play possessions. Distinct from the §4.5 "within N s after recovery"
  companion (that re-computes offensive output inside the post-recovery window; **this** measures
  whether the ball was *held* ≥ N s). Anchor 62% (worked example ~18 possessions; a plausibility
  anchor per §9.1).

### 4.3 Line heights & progression — `_progression.py`

- **Defensive-action / recovery / turnover line height** = mean x (m) of defensive actions /
  open-play recoveries / open-play losses (41.15 / — / 66.68). SPADL is per-acting-team-LTR, so each
  team's own-action mean x is already in its attacking frame (no reprojection needed for own-action
  heights).
- **Conversion chain** = possessions→final-third %, final-third→box %, box touches, box→shot %. Box =
  `spadlconfig` penalty area.
- **Final-third entries** = raw count of the team's actions carrying the ball **across**
  `x = 2*field_length/3` into the final third (started outside, ended inside), in the team's own
  attacking frame. The per-action count complementing the possession-rate `poss_to_final_third_pct`.
  (Owner-ratified base KPI, commit-gate Option B — §4.5 was under-specified on the offensive-output set.)
- **Shots** = raw count of all shots (open-play + set-piece + penalty). xG-independent; distinct from
  high-opportunity shots. (Owner-ratified base KPI, commit-gate Option B — the module previously carried
  no plain shots count.)
- **High-opportunity shots** = count of **non-penalty** shots with injected xG > `high_opportunity_xg`
  (0.15). `xg_column` absent → NaN; present with none over threshold → real 0.
- **Breakout by channel (v2)** = possession progressed **past halfway in possession**, tallied per
  left/center/right channel (`channel_boundaries`, y-thirds); count + rate per channel.

### 4.4 Build-up & post-regain — `_buildup.py`

- **Build-up outcome taxonomy (v2, 6 states)** — per build-up (a possession starting in the own
  build-up zone `build_up_zone_max_x`): {progressed-to-final-quarter, progressed-to-next-phase,
  opp-interception-own-half, stayed-phase-one, opp-won-ball-own-half, led-to-opp-shot}; counts +
  success rate (anchor 19 build-ups → 7 successful → 36%).
- **Post-regain security (v2)** — 2nd-pass completion rate after a regain; failed-first-pass count
  after regain; forward vs backward/sideways first-option split.
- **Switch-conditioned press success (v2)** — per **short goal kick**: was a switch of play
  prevented × was the ball regained. Needs a switch-of-play definition (a lateral pass exceeding
  `switch_min_lateral_m`) and short-goal-kick identification. **Small-N**: the sample size is emitted
  as a `switch_press_n` column (count of short goal kicks) so the low N is visible.

### 4.5 "Within N s after recovery" companion family

Not a single KPI — a **parameterized post-recovery window** (`post_recovery_window_seconds`, default
10; Hammarby used 5) that re-computes the **transition-output block** inside the window, emitting
`*_post_recovery` companion columns.

**Companion set (owner-ratified, Option B):** the per-action COUNTS `final_third_entries`,
`box_touches`, `shots`, `high_opportunity_shots` — the full textbook counter-attack-output set. A
*breakout* is EXCLUDED on correctness grounds (one channel per possession → a per-possession event has
no per-action-window restriction), as are possession-rates (field tilt, tempo, conversion %); these are
correctness exclusions, not scope cuts. Each companion re-uses its base column's definition, restricted
to the recovering team's actions within `post_recovery_window_seconds` of one of its recoveries;
`high_opportunity_shots_post_recovery` is NaN without `xg_column` (the raw counts are honest 0).

---

## 5. `TeamKpiParams` (frozen)

Mirrors `ShotStoppingParams`: `@dataclass(frozen=True)`, `.default(*, force_universal=False)`,
`.for_provider(provider)`, `.is_default()`, and a `_is_universal_default` compare-excluded flag.

Fields (initial):

- Possession convention (mapped to `add_possessions` kwargs — see §3.3):
  `possession_max_gap_seconds=7.0` → `max_gap_seconds`, `possession_retain_on_set_pieces=True` →
  `retain_on_set_pieces`. The three precision opt-ins are intentionally not surfaced.
- `defensive_action_types: tuple[str, ...]` (pinned SPADL set for PPDA / intensity).
- `ppda_zone_fraction=0.6`.
- `long_ball_distance_m=32.0`.
- `counterpress_seconds=5.0`; `post_recovery_window_seconds=10.0`.
- `retained_after_seconds=5.0` (possessions-retained-after-N-s threshold).
- `high_opportunity_xg=0.15`.
- `switch_min_lateral_m=30.0` (switch-of-play lateral threshold; a chosen default — `for_provider`-tunable, owner may ratify/adjust).
- `channel_boundaries` (default = thirds of `spadlconfig.field_width`) and `build_up_zone_max_x`
  (default = `spadlconfig.field_length / 3`).
- `counterpress_window: CounterpressWindow` (seconds XOR passes; default `seconds=5.0`) — the
  configurable v2 counter-press window; the XOR invariant is enforced in `CounterpressWindow.__post_init__`.
- Counter-press **presets** live as documented module constants (a registry mapping names →
  `CounterpressWindow`), not mutable params.

**`_PROVIDER_TEAM_KPI_PARAMS` ships EMPTY** — every provider resolves to the base default until an
ADR-009 apply-gate clears (§2.2). The seam exists; no per-provider values are populated speculatively.

---

## 6. Correctness pillars

### 6.1 Orientation of opponent-relative KPIs (ADR-028 class)

PPDA and field tilt combine *our* rows and *the opponent's* rows in one pitch zone, but SPADL is
per-acting-team-LTR. Both teams' coordinates are brought into a common frame via the existing `spadl`
reflection seam (`silly_kicks.reflection` / `spadl.orientation`) **before** any zoning. Guarded by a
**per-row** test (ADR-045: reflection guards must be per-row; an aggregate mean is vacuous).

### 6.2 Possession-foundation ground truth (V-f)

Every possession-driven KPI is only as trustworthy as `add_possessions`. StatsBomb open data carries
a **native `possession_id`**, so the reliability artifact validates our segmentation against it
(boundary recall/precision/F1), extending the ~0.94 recall the `add_possessions` docstring records on
WC-2018 to the broader corpus. This proves the foundation on real data.

### 6.3 Purity & determinism

`compute_team_kpis` is pure (no input mutation) and order-insensitive: KPIs are functions of
chronological content; the module sorts on the robust `(game_id, period_id, time_seconds, action_id)`
key `add_possessions` already uses.

---

## 7. `TeamKpiReport` (conservation census)

Small frozen dataclass, CI-asserted conservation (mirrors `ShotStoppingReport`):

- **Match census:** `n_matches_in`, `n_matches_scored` (exactly two team ids present),
  `n_matches_excluded_not_two_teams`. Invariant:
  `n_matches_scored + n_matches_excluded_not_two_teams == n_matches_in`. A ≠2-team game is
  **excluded and counted**, never silently dropped.
- **Shot-xG census** (only when `xg_column` supplied): `n_shots`, `n_shots_with_xg`,
  `n_shots_null_xg`. Invariant: `n_shots_with_xg + n_shots_null_xg == n_shots`. A shot-class row with
  null xG is excluded-and-counted, never treated as xG=0.

---

## 8. Degenerate handling — honest-NaN, never a fabricated zero (ADR-042 / ADR-027)

A KPI whose inputs are absent emits **NaN**; a real `0` is kept distinct from an undefined value:

- PPDA — 0 defensive actions in zone → NaN.
- Defensive intensity — no out-of-possession time → NaN.
- Recoveries-within-5s % — had recoveries, none within 5 s → real **0%**; **no** recoveries → NaN.
- Time-to-recovery — loss never regained → spell excluded from the mean.
- High-opportunity shots — no `xg_column` → NaN; present, none over threshold → real 0.
- Build-up / breakout success rates — zero qualifying possessions → NaN.
- Possessions-retained-after-Ns — zero open-play possessions → NaN, never 0%.
- Conversion chain — each ratio is NaN on an empty denominator, never 0: zero possessions →
  possessions→final-third % NaN; zero final-third entries → final-third→box % NaN; zero box touches →
  box→shot % NaN.

Every degenerate case is tested from **both sides** (§9): the NaN case AND a mutation that should move
the value out of NaN.

---

## 9. Testing

### 9.1 Library module (regular suite unless noted)

- **Analytic fixtures for exact assertions** — hand-built action sequences with hand-computed expected
  KPI values. The plan's worked values (PPDA 4.17, tilt 63%, tempo 19.97, heights 41.15/66.68, times
  7.30/7.69, within-5s 14%, build-up 36%) are **plausibility/order anchors on real data (e2e)**, not
  exact reproductions (their raw match is unavailable). A **dedicated retained-after-Ns fixture**
  (possessions of known durations straddling the 5 s threshold) asserts its exact numerator/denominator.
- **Degenerate fixtures, both sides** — zero-recovery, 100%-possession, ≠2-team, null-xG shots, no
  `xg_column`, possession-never-regained, zero-open-play-possessions (retained-after-Ns +
  conversion-chain NaN) → honest-NaN + a mutation that moves it, plus Report conservation.
- **Orientation per-row test** — known away-team action; PPDA/field-tilt zoning correct post-reflection.
- **Purity + order-insensitivity** — input unmutated; permuted input → identical output modulo
  `action_id`.
- **Repo-wide sibling gates** (several fail only in the full `-m "not e2e"` suite): import-allowlist
  (never `tracking`); `feature_glossary` coverage (+N columns, count updated); C4 completeness (+1
  container, dot-rendered); NOTICE attribution; canonical-id/dtype grouping (ADR-019); `Params`
  default-stability + `is_default`; **scale-guard registration** — `_possession` groups per
  match/team via `group_rows`, so it registers in `SCALE_GUARDED` with a growth fixture that scales
  the **group/loop dimension** (ADR-073 lesson), not a within-group one.
- Doctest examples on `compute_team_kpis` + `TeamKpiParams` (executable or literal block per the
  public-example gate).

### 9.2 Public reliability artifact

`scripts/validate_team_kpi_reliability.py`, adopting the shared driver seams: **ADR-052** `for_each`
sharded/resumable corpus pass; **ADR-037** clean-tree provenance (`require_clean_tree` + `run_commit`
+ `run_tree_dirty`); **ADR-038** fail-closed public-only enforcement (see below); **ADR-056**
input-contract declaration.

- **Corpus:** StatsBomb open (primary; the **FULL open-data manifest by default** — `all_open_competitions()`
  enumerates every public `(competition_id, season_id)` StatsBomb releases: thousands of matches across WC
  2018/2022, the Euros, the Women's World Cup, FA WSL, La Liga, UCL finals, NWSL, …; a single tournament is
  far too thin for a team-discrimination ICC / split-half) + Wyscout public (Pappalardo 2019) for the
  cross-provider comparability leg. Fully redistributable → **reproducible by anyone**. The SPADL
  **converters** exist (`spadl.statsbomb` / `spadl.wyscout`); the driver adds the raw public-data
  **loaders** — StatsBomb open via `statsbombpy` (currently used only via `importorskip` in an e2e test,
  undeclared in `pyproject`; the driver adds it as a **scripts-only** optional dep — **no new runtime
  dependency**), and a script-side reader for the public Wyscout (Pappalardo 2019) dataset.
- **Fail-closed public-only (ADR-038), corpus-appropriate — NOT the pining `assert_public_corpus`.** The
  27-match pining redistribution registry cannot represent an open-data corpus (thousands of matches, none
  registered), so the guard is source-based: the StatsBomb leg REFUSES to run with credentials configured
  (`assert_statsbomb_open_data_mode` — with `SB_USERNAME`/`SB_PASSWORD` set, statsbombpy pulls the PRIVATE
  API; no creds ⇒ the open-data manifest only), and the Wyscout leg allowlists the seven public Pappalardo
  competitions (`_PAPPALARDO_PUBLIC_COMPETITIONS`; a `--wyscout-dir` file naming any other competition
  raises). Both guarantee an artifact can never stamp private data as `public`.
- **Reports:** (1) per-KPI reliability/repeatability — split-half + cross-season ICC + Type-II
  (major-axis) regression + Pearson r (the V-e method) at the team grain; (2) possession-foundation
  ground truth vs StatsBomb native `possession_id` (§6.2); (3) per-provider comparability — StatsBomb
  and Wyscout run separately then compared, pooled only where a comparability check passes (never a
  naive pool). The xG-gated `high_opportunity_shots` KPI is measured on the StatsBomb leg by injecting
  StatsBomb's **own** pre-shot `statsbomb_xg` (joined onto shot rows by `original_event_id`, passed as
  `xg_column`) — else that KPI (and its post-recovery companion) would be all-NaN across the study;
  the Wyscout leg carries no xG, so those two are NaN there (honest).
- **Output:** `docs/research/tf52_team_kpi_reliability/` (findings.md + metrics.json + provenance),
  **corpus bound recorded inside the artifact**, **reported-not-gated** (ADR-009 — changes no library
  default, gates no CI).

---

## 10. Delivery shape

Two commits (the established pattern), each fully-tested and coherent, on one feature branch
(`feat/tf52-team-kpi`):

1. **Module + driver + unit tests** — `silly_kicks/team_metrics/`, the reliability driver + its
   unit tests (on synthetic data with known reliability), glossary/NOTICE/C4 wiring.
2. **Owner-run public-corpus artifact** — `docs/research/tf52_team_kpi_reliability/`, provenance
   stamping commit 1.

- Version / PR-S / ADR numbers are **not** reserved; assign at commit-prep after
  `git fetch && git merge origin/main` (whoever merges first takes the lower number).
- **No commit or push without explicit owner approval for that specific commit.**

---

## 11. Impact

- **Additive** — no existing feature changes; **no VAEP/tracking retrain, no re-materialize.**
- **+1 C4 container**; action-coupled aggregator count unchanged (33); `feature_glossary` grows by the
  emitted-column count.
- New public surface: `compute_team_kpis`, `TeamKpiParams`, `TeamKpiReport`, the column-name
  constants.
- **No new runtime dependency.** `statsbombpy` is not currently declared in `pyproject` (used only via
  `importorskip` in an e2e test); the reliability driver would add it as a **scripts-only** optional
  dep. The Wyscout public dataset is converted via the existing `spadl.wyscout` converter, with a
  script-side reader for its raw format.

---

## 12. Rejected alternatives

- **Multiple public `compute_*` per family** (`compute_pressing_kpis`, …) instead of one orchestrator
  — rejected: multiplies the public surface and the C4/glossary wiring with no caller benefit; the
  deliverable is one team-match row.
- **A forked/new possession definition** for TF-52 — rejected: `spadl.add_possessions` is the single
  validated primitive ("decide once, use everywhere"; Chesterton's Fence). TF-52 tunes it via params,
  never replaces it.
- **Including the structural interception-height / aerial / compactness / block-anchor items** —
  rejected for this cycle: not event-only or not representable in canonical SPADL (§2.2).
- **A per-provider parameter tune in v1** — rejected: ADR-009 forbids baking a tune without an
  evidence-earned, separately-gated apply PR.
- **An owner-tier / single-tournament reliability corpus** — rejected: TF-52 is event-only, so the
  natural corpus is **public** event data (thousands of matches), which is both properly powered and
  reproducible; owner-tier is neither necessary nor desirable for a public artifact.
- **Reaching for Optuna / evolutionary search on the KPIs** — rejected: the KPIs are deterministic
  definitions with nothing to optimise; the gold-standard levers here are definitional rigor, the
  single validated possession foundation, and the reproducible reliability study.

---

## 13. Decisions log

- **Order:** TF-52 before TF-53; both independent (no cross-dependency; both reuse `add_possessions`).
- **Scope:** v1 + event-based v2; four v2 items + two tracking items excluded with reasons (§2.2).
- **Validation:** reproducible **public** artifact on StatsBomb open + Wyscout public; owner-tier
  private run explicitly not part of this deliverable.
- **API:** single `compute_team_kpis` orchestrator; per-`(game, team)` grain; injected `xg_column`.
- **Foundation:** reuse `spadl.add_possessions`.

---

## 14. Attribution (NOTICE)

Twelve.football match-report glossary (Soccermatics module 3); MSC Bootcamp Webinar 3 (Tigres Femenil
/ Clemson / Coventry academy) for the practitioner set; Pappalardo et al. 2019 (public Wyscout data
set) and StatsBomb open data for the reliability corpus. Full citations added to the `NOTICE`
"Mathematical / Methodological References" section per ADR-005.

---

## 15. Revision log

**R1 — independent review** (`D:\Development\_reviews\2026-09-15-tf52-team-kpi-spec.md`):

- **TF52-SPEC-01 (blocking) — FIXED.** "Possessions retained after N s" is now defined (§4.2),
  NaN-handled (§8), and fixtured (§9.1 — a dedicated retained-after-Ns fixture + a degenerate case).
- **TF52-SPEC-03 (should-fix) — FIXED.** The `add_possessions` field→kwarg mapping
  (`possession_max_gap_seconds` → `max_gap_seconds`, etc.) and the three intentionally-unsurfaced
  precision opt-ins are now explicit (§3.3, §5).
- **TF52-SPEC-02 (should-fix) — NOT APPLIED; verified as a false positive.** The reviewer read the ADR
  *file titles* correctly, but the repo *cites these ADR numbers by convention* for these rules, and
  the spec follows that convention:
  - `scripts/validate_gk_decision.py:8` — *"Clean-tree guard runs FIRST (**ADR-037**)"* + *"`for_each`
    (**ADR-052**: per-match shards)"* + input contract (**ADR-056**). Three distinct ADRs, matching
    §9.2. CLAUDE.md repeats this for both sibling battery drivers.
  - CLAUDE.md:135 — *"Academic attribution discipline. … Decision: **ADR-005**"*; gk_decision cites
    *"NOTICE (**ADR-005**/048)"*. Matches §14.
  - Changing the citations would make the spec inconsistent with the codebase. (Separately: ADR-005's
    file title is a pre-existing repo doc-hygiene matter, out of scope for this spec.)
- **CONSIDER items applied:** §2.2 records the interception-height exclusion as owner-ratified
  (2026-09-15) and clarifies bravery is out-of-scope / not-imported; §8 enumerates the conversion-chain
  degenerate; §9.2/§11 state `statsbombpy` accurately (undeclared today, added scripts-only, no new
  runtime dep).
- **COULD NOT VERIFY (reviewer) — no action:** Twelve worked values are external/non-reproducible
  (spec already treats them as anchors); the end-to-end driver run is owner-run, not CI.

**R2 — reconciliations surfaced by the plan review**
(`D:\Development\_reviews\2026-09-15-tf52-team-kpi-plan.md`, CONSIDER items — the plan was found
correct; the spec is aligned to it):

- §3.1 — added `_orientation.py` to the file list (was implied by §6.1 but unlisted).
- §3.4 — added `silly_kicks.reflection` to the import allowlist (§6.1 requires it; the literal
  "spadl + id_compat + np/pd only" was incomplete).
- §5 — pinned `switch_min_lateral_m=30.0` (a chosen default, `for_provider`-tunable), named
  `build_up_zone_max_x` and the `channel_boundaries` defaults concretely, and added the
  `counterpress_window: CounterpressWindow` field (the seconds-XOR-passes v2 window).
- §4.4 — named the `switch_press_n` small-N column.
