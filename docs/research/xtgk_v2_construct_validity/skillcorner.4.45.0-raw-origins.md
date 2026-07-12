# xT-GK v2 construct-validity — skillcorner (FAITHFUL V_opp)
- rho variant: `skillcorner` * GK-distribution test rows: **2751** * V_opp = faithful observed-post-turnover, possession-bound, TRAIN-fit

| metric | AUC | n |
|---|---|---|
| **xt_gk_v2** | 0.5125 | 2751 |
| raw_completion | 0.5681 | 2751 |
| destination_xt | 0.5845 | 2751 |
| v1_stored (c.xt_gk) | 0.5840 | 2751 |
| xt_gk_v2 (on v1-covered rows) | 0.5125 | 2751 |

**LIFT** (v2 - max baseline, full GK-test): **-0.0720**

**v2 vs v1 (matched rows):** v2 0.5125 vs v1 0.5840 (d -0.0715)

### Component decomposition (did the faithful V_opp un-swamp rho*dV?)
| term | \|mean\| share |
|---|---|
| position | 36% |
| pev | 0% |
| retention_loss | 35% |
| dzv | 29% |

AUC (harness target): **rho*dV alone 0.5431** * +retention 0.5039 * full 0.5125

### R1 deep-cell disentanglement (V_opp, train-fit; mean over terciles)
| zone | possession-bound (prod) | mirror (proxy) | 10s (sens.) | native n | level |
|---|---|---|---|---|---|
| 176 | 0.0123 | 0.0252 | 0.0001 | 3 | 1 |
| 177 | 0.0123 | 0.0278 | 0.0001 | 2 | 1 |
| 160 | 0.0123 | 0.0347 | 0.0001 | 3 | 1 |
| 161 | 0.0123 | 0.0326 | 0.0001 | 9 | 1 |
| 144 | 0.0123 | 0.0440 | 0.0001 | 3 | 1 |
| 145 | 0.0123 | 0.0378 | 0.0001 | 5 | 1 |
| 128 | 0.0123 | 0.0625 | 0.0001 | 5 | 1 |
| 129 | 0.0123 | 0.0434 | 0.0001 | 5 | 1 |
| 112 | 0.0108 | 0.1277 | 0.0001 | 14 | 1 |
| 113 | 0.0108 | 0.1098 | 0.0001 | 20 | 1 |
| 96 | 0.0134 | 0.5218 | 0.0000 | 21 | 0 |
| 97 | 0.0182 | 0.1681 | 0.0000 | 31 | 0 |
| 80 | 0.0079 | 0.4739 | 0.0000 | 27 | 0 |
| 81 | 0.0082 | 0.1654 | 0.0000 | 45 | 0 |
| 64 | 0.0108 | 0.1823 | 0.0001 | 18 | 1 |
| 65 | 0.0108 | 0.1222 | 0.0001 | 15 | 1 |
| 48 | 0.0105 | 0.0660 | 0.0000 | 4 | 1 |
| 49 | 0.0105 | 0.0636 | 0.0000 | 9 | 1 |
| 32 | 0.0105 | 0.0393 | 0.0000 | 5 | 1 |
| 33 | 0.0105 | 0.0560 | 0.0000 | 5 | 1 |
| 16 | 0.0105 | 0.0312 | 0.0000 | 3 | 1 |
| 17 | 0.0105 | 0.0513 | 0.0000 | 4 | 1 |
| 0 | 0.0105 | 0.0205 | 0.0000 | 0 | 1 |
| 1 | 0.0105 | 0.0379 | 0.0000 | 3 | 1 |

> Read: **possession-bound << mirror at real support (level 0/1)** = the mirror over-stated deep threat (the genuine finding). **10s << possession-bound** = window shrinkage (an artifact — NOT the finding). Level census over 24 deep cells: native 4 / block 20 / global 0 (a global-fallback deep cell is NOT a real estimate).

> GK-distribution-domain eval (is_gk_distribution); V out-of-sample (possession-parity split), rho IN-SAMPLE (the production model serves its training population); V is ~expected first-shot xG so absolute AUC vs possession-reaches-shot is partly circular -- read LIFT over max(baselines). V_opp = faithful observed-post-turnover (possession-bound), TRAIN-fit; v1_stored from fct_action_context.xt_gk (no frames).
