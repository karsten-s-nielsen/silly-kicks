# xT-GK v2 secondary faithfulness audit (W6)

## kappa sweep (faithful V_opp, possession-bound; provider `gradientsports`) — REPORTED, not tuned (§3: kappa=1 is the a-priori headline)
| kappa | xt_gk_v2 AUC | lift |
|---|---|---|
| 1.0 | 0.4836 | -0.1387 |
| 1.5 | 0.4813 | -0.1410 |
| 2.0 | 0.4770 | -0.1453 |

> kappa scales the turnover term `dzv = -(1-rho)*kappa*V_opp`. With the faithful (small) V_opp, raising kappa adds more of a term that (per W4) drags the metric below `rho*dV` alone, so a larger kappa does not help; **kappa=1 stays the headline** (not chosen to optimise this, it is the default). The kappa/turnover-weighting is a genuine question for Jeff, given the faithful V_opp shifts the balance.

## V reward interpretation — DEFERRED (flagged for owner/Jeff, not re-implemented here)
> V uses **`E[first-shot xG]`** (our Singh-spirit reading); Jeff §2.1 says *"expected threat over the remainder of the possession."* First-shot vs cumulative-remainder is a real interpretation fork that may relate to V's weak realized-xG out-of-sample correlation (Spearman 0.03-0.06). **Deferred**: re-implementing V is out of scope for this release (a separate decision if it matters) — surfaced for the Jeff conversation, not silently changed.

## PEV dormant (note)
> PEV is 0 (`p'=p`; receiver-pressure `q` deferred per Jeff §8-step-7), so the metric currently carries no pressure-value-added term — faithful to his sequencing, noted for completeness.
