# xT-GK v2 construct-validity — gradientsports (FAITHFUL V_opp)
- rho variant: `C:/Users/Karsten/AppData/Local/Temp/claude/D--Development-karstenskyt--silly-kicks/f8334e52-8c05-466e-8cc5-3158a9eb1c6a/scratchpad/rho_prefix/default` * GK-distribution test rows: **1902** * V_opp = faithful observed-post-turnover, possession-bound, TRAIN-fit

| metric | AUC | n |
|---|---|---|
| **xt_gk_v2** | 0.4561 | 1902 |
| raw_completion | 0.6223 | 1902 |
| destination_xt | 0.5711 | 1902 |
| v1_stored (c.xt_gk) | 0.3813 | 1710 |
| xt_gk_v2 (on v1-covered rows) | 0.4561 | 1710 |

**LIFT** (v2 - max baseline, full GK-test): **-0.1661**

**v2 vs v1 (matched rows):** v2 0.4561 vs v1 0.3813 (d +0.0749)

### Component decomposition (did the faithful V_opp un-swamp rho*dV?)
| term | \|mean\| share |
|---|---|
| position | 28% |
| pev | 0% |
| retention_loss | 41% |
| dzv | 32% |

AUC (harness target): **rho*dV alone 0.5399** * +retention 0.4595 * full 0.4561

### R1 deep-cell disentanglement (V_opp, train-fit; mean over terciles)
| zone | possession-bound (prod) | mirror (proxy) | 10s (sens.) | native n | level |
|---|---|---|---|---|---|
| 176 | 0.0027 | 0.0133 | 0.0000 | 4 | 1 |
| 177 | 0.0027 | 0.0157 | 0.0000 | 6 | 1 |
| 160 | 0.0027 | 0.0170 | 0.0000 | 1 | 1 |
| 161 | 0.0027 | 0.0278 | 0.0000 | 2 | 1 |
| 144 | 0.0027 | 0.0186 | 0.0000 | 1 | 1 |
| 145 | 0.0027 | 0.0205 | 0.0000 | 3 | 1 |
| 128 | 0.0027 | 0.0243 | 0.0000 | 1 | 1 |
| 129 | 0.0027 | 0.0422 | 0.0000 | 4 | 1 |
| 112 | 0.0050 | 0.1354 | 0.0005 | 2 | 1 |
| 113 | 0.0050 | 0.0724 | 0.0005 | 8 | 1 |
| 96 | 0.0050 | 0.2561 | 0.0005 | 3 | 1 |
| 97 | 0.0050 | 0.1575 | 0.0005 | 4 | 1 |
| 80 | 0.0051 | 0.2864 | 0.0005 | 1 | 1 |
| 81 | 0.0039 | 0.1362 | 0.0005 | 3 | 1 |
| 64 | 0.0050 | 0.1075 | 0.0005 | 1 | 1 |
| 65 | 0.0050 | 0.1089 | 0.0005 | 7 | 1 |
| 48 | 0.0016 | 0.0485 | 0.0000 | 2 | 1 |
| 49 | 0.0016 | 0.0385 | 0.0000 | 4 | 1 |
| 32 | 0.0016 | 0.0426 | 0.0000 | 0 | 1 |
| 33 | 0.0016 | 0.0162 | 0.0000 | 2 | 1 |
| 16 | 0.0016 | 0.0114 | 0.0000 | 0 | 1 |
| 17 | 0.0016 | 0.0303 | 0.0000 | 2 | 1 |
| 0 | 0.0016 | 0.0132 | 0.0000 | 2 | 1 |
| 1 | 0.0016 | 0.0160 | 0.0000 | 1 | 1 |

> Read: **possession-bound << mirror at real support (level 0/1)** = the mirror over-stated deep threat (the genuine finding). **10s << possession-bound** = window shrinkage (an artifact — NOT the finding). Level census over 24 deep cells: native 0 / block 24 / global 0 (a global-fallback deep cell is NOT a real estimate).

> GK-distribution-domain eval (is_gk_distribution); V out-of-sample (possession-parity split), rho IN-SAMPLE (the production model serves its training population); V is ~expected first-shot xG so absolute AUC vs possession-reaches-shot is partly circular -- read LIFT over max(baselines). V_opp = faithful observed-post-turnover (possession-bound), TRAIN-fit; v1_stored from fct_action_context.xt_gk (no frames).
