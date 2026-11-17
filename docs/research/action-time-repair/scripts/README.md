# Action-time repair: measurement and acceptance scripts

The scripts that the spec (`docs/superpowers/specs/2026-11-18-action-time-repair-design.md`) cites, copied from the
DGX working folders so that the acceptance check of Task 16 can run them again on the repaired stage. They are
evidence, not library code: each reads a staged corpus or the raw provider cache in the DGX working folders (read only) and prints to its `.out`. No token or credential is in any of them.

| Script | What it measures | Spec | Origin |
|---|---|---|---|
| `elastic_check.py` | ELASTIC alignment of the idsse event times | 4.3 | the DGX working folder |
| `release_check.py` | the ball's release after a SkillCorner restart | 4.4, 4.6 | the DGX working folder |
| `restart_release_reach.py` | the rest (0.5 m, 0.5 s) and release (2 m within 1 s) rule of the kick | 4.4 | the supervisor's DGX working folder |
| `corner_kick_check.py` | the measured corner kicks (share, place, subtype) | 4.4, 4.6 | the DGX working folder |
| `order_only_corner_check.py` | the rest runs in the gaps of the 690 order-only corners | 4.4 | the DGX working folder |
| `order_only_overlap.py` | the order-only rows and the phase windows | 5.2 | the DGX working folder |
| `spec_r2_checks.py` | points 1, 3, 6, 8 of the supervisor's read (incl. the R44 frame shares) | 4.9, 4.6 | the DGX working folder |
| `sc_score_lag.py` | the running-score effect rule and its lag | 4.15 | the DGX working folder |
| `derived_key_check.py` | the key of a derived row | 4.18 | the DGX working folder |
| `penalty_wait_check.py` | the wait before a penalty kick | 4.17 | the DGX working folder |
| `a23_penalty_check.py` | bar A23: the idsse penalty rows and the goal rows of the 7 stage matches | 4.16 | the DGX working folder |
| `kick_rule_compare.py` | the library kick rule `_find_kick` against `corner_kick_check.rest_release` on the 559 GS corners (agreement, accuracy against the provider time) | 4.4 | the DGX working folder (reviewer B RTR-IMPL-03) |
| `r67_mutations.py` | bar A29: the mutation teeth of the revision-tag gate (a scratch copy, never the repo) | 4.14 | a local scratch copy |
