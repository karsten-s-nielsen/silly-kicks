# xT-GK v2 keeper discrimination — gradientsports (FAITHFUL V_opp)
- rho variant: `gs` * GK-distribution actions: **3874** * min 20 dist/keeper * V_opp fit on FULL cohort (descriptive spread)

| metric | ICC (action-level) | CV (means, unstable) | n keepers |
|---|---|---|---|
| **xt_gk_v2** | **-0.0020** | 0.236 | 39 |
| v1 (c.xt_gk) | 0.0193 | 0.060 | 39 |

> ICC = between-keeper variance ÷ total (action-level; NOT per-keeper means). Higher = separates keepers more. CV = std/|mean| of per-keeper means (secondary; unstable when the metric mean ~ 0). **§3: report whatever it shows** — if v2's ICC ~ v1's, v2 is still keeper-flat on the faithful metric.

### xt_gk_v2 top keepers (per-action mean; face validity — the owner's coaching eye)
| # | player_key | v2 mean | n |
|---|---|---|---|
| 1 | `8.323227946508364e+18` | -0.0008 | 33 |
| 2 | `8.263640357118681e+18` | -0.0009 | 117 |
| 3 | `4.917175115125928e+18` | -0.0009 | 117 |
| 4 | `-3.131937957277373e+18` | -0.0010 | 165 |
| 5 | `3.433898586235692e+18` | -0.0010 | 162 |
| 6 | `-6.111100716684954e+18` | -0.0010 | 26 |
| 7 | `7.442119108907284e+18` | -0.0011 | 86 |
| 8 | `2.483016876913307e+18` | -0.0013 | 26 |

---
# xT-GK v2 keeper discrimination — skillcorner (FAITHFUL V_opp)
- rho variant: `skillcorner` * GK-distribution actions: **5487** * min 20 dist/keeper * V_opp fit on FULL cohort (descriptive spread)

| metric | ICC (action-level) | CV (means, unstable) | n keepers |
|---|---|---|---|
| **xt_gk_v2** | **0.0109** | 0.215 | 54 |
| v1 (c.xt_gk) | 0.0176 | 0.096 | 54 |

> ICC = between-keeper variance ÷ total (action-level; NOT per-keeper means). Higher = separates keepers more. CV = std/|mean| of per-keeper means (secondary; unstable when the metric mean ~ 0). **§3: report whatever it shows** — if v2's ICC ~ v1's, v2 is still keeper-flat on the faithful metric.

### xt_gk_v2 top keepers (per-action mean; face validity — the owner's coaching eye)
| # | player_key | v2 mean | n |
|---|---|---|---|
| 1 | `-1.1835133969160976e+17` | -0.0006 | 48 |
| 2 | `-1.1560135370874294e+18` | -0.0039 | 30 |
| 3 | `6.411043353981681e+18` | -0.0040 | 71 |
| 4 | `6.081054062477078e+18` | -0.0041 | 21 |
| 5 | `-3.6944910473613686e+18` | -0.0041 | 39 |
| 6 | `1.3417315090001615e+18` | -0.0043 | 25 |
| 7 | `5.726355527147203e+18` | -0.0043 | 25 |
| 8 | `5.589514683960524e+18` | -0.0044 | 33 |

---
