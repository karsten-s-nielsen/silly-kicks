# TF-52 team-KPI reliability study

**What this is.** A public, reproducible reliability characterisation of the `silly_kicks.team_metrics`
event-only team KPIs (TF-52, 4.115.0). It measures, for each KPI, **how repeatable / team-discriminating**
it is across a team's matches — *reliability*, not *validity* (a repeatable KPI is not automatically a
correct one). **Reported-not-gated (ADR-009): this changes no library default and gates no CI.**

**Provenance.** All numbers stamp code commit `19a5883` on a clean tree (`run_tree_dirty: false`),
produced by `scripts/validate_team_kpi_reliability.py`. The corpora are fully public, so anyone can
reproduce (see *Reproduce* below). Raw per-`(game, team)` shards are a superset kept out of the repo;
this artifact carries only the aggregate `*_metrics.json` + `comparison.json` + this summary.

## Corpus

| Provider | Source | Matches | Teams |
|---|---|---|---|
| StatsBomb | **full open-data manifest** (`sb.competitions()`, 80 competition-seasons — WC 2018/2022, Euros, Women's WC, FA WSL, La Liga, UCL finals, NWSL, …) | **3961** | 354 |
| Wyscout | **public Pappalardo 2019** (7 competitions: England, Italy, Spain, Germany, France, European Championship, World Cup) | **1941** | 142 |

Fail-closed public-only: the StatsBomb leg refuses to run with credentials configured
(`assert_statsbomb_open_data_mode` — no creds ⇒ open-data only); the Wyscout leg allowlists the 7
public Pappalardo competitions. StatsBomb's own `statsbomb_xg` is injected so `high_opportunity_shots`
is measured on that leg (Wyscout carries no xG).

## Leg 1 — per-KPI reliability

`ICC(1)` = one-way team-discrimination ICC (a team across its matches is the group). `split-half r` =
Pearson r of odd- vs even-match team means (a reliability of the aggregated team profile, so it runs
higher than the single-match ICC). Sorted by StatsBomb ICC.

| KPI | SB ICC | SB split-half | WY ICC | WY split-half |
|---|---|---|---|---|
| long_ball_pct | 0.435 | 0.711 | 0.425 | 0.787 |
| possessions_retained_after_ns_pct | 0.398 | 0.686 | 0.363 | 0.746 |
| post_regain_second_pass_pct | 0.389 | 0.648 | 0.258 | 0.656 |
| field_tilt_pct | 0.363 | 0.706 | 0.249 | 0.462 |
| box_touches | 0.351 | 0.721 | 0.253 | 0.589 |
| poss_to_final_third_pct | 0.334 | 0.644 | 0.260 | 0.580 |
| buildup_success_pct | 0.330 | 0.659 | 0.232 | 0.508 |
| turnover_line_height_m | 0.312 | 0.617 | 0.237 | 0.481 |
| time_to_recovery_s | 0.312 | 0.775 | 0.186 | 0.485 |
| final_third_entries | 0.291 | 0.667 | 0.219 | 0.474 |
| post_regain_failed_first_passes | 0.291 | 0.691 | 0.260 | 0.811 |
| pass_tempo | 0.257 | 0.642 | 0.204 | 0.622 |
| recoveries | 0.252 | 0.733 | 0.191 | 0.525 |
| final_third_entries_post_recovery | 0.251 | 0.625 | 0.255 | 0.598 |
| buildup_final_quarter | 0.245 | 0.529 | 0.159 | 0.429 |
| shots | 0.239 | 0.601 | 0.189 | 0.412 |
| box_touches_post_recovery | 0.219 | 0.554 | 0.105 | 0.279 |
| defensive_action_height_m | 0.212 | 0.422 | 0.107 | 0.121 |
| high_opportunity_shots | 0.190 | 0.465 | — (no xG) | — |
| defensive_intensity | 0.173 | 0.697 | 0.093 | 0.184 |
| ppda | 0.156 | 0.417 | 0.076 | 0.101 |
| shots_post_recovery | 0.140 | 0.437 | 0.077 | 0.106 |
| counterpress_regains | 0.134 | 0.608 | 0.105 | 0.249 |
| recoveries_within_ns_pct / counterpress_regain_pct | 0.120 | 0.453 | 0.114 | 0.221 |
| box_to_shot_pct | 0.033 | 0.128 | 0.031 | 0.125 |
| switch_press_success_pct | 0.024 | 0.006 | 0.007 | −0.077 |

*(Full per-KPI values incl. the build-up / breakout / post-regain families in `statsbomb_metrics.json`
+ `wyscout_metrics.json`.)*

**Reading it.**
- **Most reliable = stable team-style signals:** long-ball %, possessions-retained, field tilt, box
  touches, poss→final-third, pass tempo, build-up success. These discriminate teams consistently and
  are the safest for season-level comparison.
- **The new Option-B counts hold up:** `final_third_entries` (SB ICC 0.291) and `shots` (0.239) sit
  mid-table — solid, comparable to the established counts.
- **Least reliable = small-denominator rates:** `switch_press_success_pct`, `box_to_shot_pct`,
  `breakout_*_pct` — expected for ratios over few qualifying events; use with large samples only.
- **PPDA is moderately noisy** (SB ICC 0.156) — a known property of a game-state-sensitive ratio; the
  count/height pressing KPIs are steadier.
- Every KPI's SB ICC ≥ its Wyscout ICC, consistent with StatsBomb's ~2× larger, denser corpus.

## Leg 2 — possession-foundation ground truth (StatsBomb native `possession_id`)

The KPIs rest on `spadl.add_possessions`; this measures its boundary fidelity vs StatsBomb's native
possession counter over 3961 matches: **recall 0.912 / precision 0.418 / F1 0.572**.

`add_possessions` recovers **91%** of native possession boundaries (high recall) but **over-segments**
(precision 0.42) — its 7 s-gap + set-piece heuristic cuts more possessions than StatsBomb's broader
native counter. This is a transparency measure, not a defect: the KPIs use `add_possessions`
*throughout*, so they are internally consistent; the two segmenters simply draw boundaries differently.

## Leg 3 — cross-provider comparability

A KPI is flagged **poolable** only where both providers report a finite, same-sign ICC within tolerance
(0.20 spread). **42 of 44 KPIs are poolable** across StatsBomb + Wyscout. The **2 not poolable** are
`high_opportunity_shots` and `high_opportunity_shots_post_recovery` — Wyscout carries no xG, so they are
NaN there (honest, not a disagreement). Details in `comparison.json`.

## Honest limits

- **Reliability ≠ validity.** A high ICC means a KPI is repeatable and team-discriminating, not that it
  measures the tactical concept it names.
- **The pooled ICC mixes competitions, genders and eras** (a team's matches span its competitions;
  men's + women's, leagues + tournaments). This is a broad reliability estimate, not per-competition.
- **`high_opportunity_shots` reliability uses StatsBomb's own xG** (the injected model), so it partly
  reflects that model's properties.
- **Possession-boundary precision (0.42)** reflects `add_possessions` over-segmenting vs the native
  counter; see Leg 2.

## Reproduce

```
# StatsBomb (full open-data manifest) — no credentials => public-only
python scripts/validate_team_kpi_reliability.py --provider statsbomb --out <out>/statsbomb
# Wyscout (public Pappalardo 2019; figshare collection 4415000)
python scripts/validate_team_kpi_reliability.py --provider wyscout --wyscout-dir <pappalardo> --out <out>/wyscout
# Cross-provider comparability
python scripts/validate_team_kpi_reliability.py --compare <out>/statsbomb/metrics.json <out>/wyscout/metrics.json --out <out>/comparison
```
