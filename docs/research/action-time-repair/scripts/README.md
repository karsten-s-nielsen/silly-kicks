# Action-time repair: measurement and acceptance scripts

The scripts that the spec (`docs/superpowers/specs/2026-11-18-action-time-repair-design.md`) cites, and the acceptance
scripts of the restage. They are evidence, not library code. Each one reads a stage, the raw provider cache or the
output of another script in this folder (read only), and prints to stdout or writes to the output that you give it.
No token or credential is in any of them.

Run each one from a clone root with `PYTHONPATH=.` on Linux (the workers fork). The arguments:

- `--stage`: a stage root, as `scripts.materialize_tc3_frames --out` writes it (`_actions/`, `shards/`, `home_teams.json`).
- `--old-stage`: the stage root of the commit before the repair (f5fc2e2b), for the before-and-after compares.
- `--cache-dir`: the raw provider cache root, as `--cache-dir` of `scripts.materialize_tc3_frames` (`<provider>/<match>/`).
- `--out`: the output folder or file. `--workers`: the number of processes.

| Script | What it measures | Spec | Command |
|---|---|---|---|
| `elastic_check.py` | ELASTIC alignment of the idsse event times | 4.3 | `python elastic_check.py --stage S --cache-dir C --out E --workers 8` |
| `release_check.py` | the ball's release after an on-ball action | 4.4, 4.6 | `python release_check.py --stage S --out R --providers skillcorner,gradientsports,idsse --workers 8` |
| `restart_release_reach.py` | the rest (0.5 m, 0.5 s) and release (2 m within 1 s) rule of the kick | 4.4 | `python restart_release_reach.py --cache-dir C` |
| `corner_kick_check.py` | the measured corner kicks (share, place, subtype) | 4.4, 4.6 | `python corner_kick_check.py --stage S --cache-dir C --out K --workers 8` |
| `order_only_corner_check.py` | the rest runs in the gaps of the 690 order-only corners | 4.4 | `python order_only_corner_check.py --cache-dir C --out O.parquet` |
| `order_only_overlap.py` | the order-only rows and the phase windows | 5.2 | `python order_only_overlap.py --cache-dir C` |
| `spec_r2_checks.py` | points 1, 3, 6, 8 of the review of revision 1 (incl. the frame shares of section 4.9) | 4.9, 4.6 | `python spec_r2_checks.py 1 3 6 8 --stage S --cache-dir C --corner-rows K --elastic-rows E` |
| `sc_score_lag.py` | the running-score effect rule and its lag | 4.15 | `python sc_score_lag.py --cache-dir C` |
| `derived_key_check.py` | the key of a derived row | 4.18 | `python derived_key_check.py --stage S --cache-dir C` |
| `sc_penalty_check.py` | the SkillCorner penalties: count, team, kick, mark radius | 4.17 | `python sc_penalty_check.py --cache-dir C --out P.parquet` |
| `penalty_wait_check.py` | the wait before a penalty kick | 4.17 | `python penalty_wait_check.py --table P.parquet` |
| `idsse_penalty_rows_check.py` | the idsse penalty rows and the goal rows of the 7 stage matches | 4.16 | `python idsse_penalty_rows_check.py --cache-dir C` |
| `kick_rule_compare.py` | the library kick rule `_find_kick` against `corner_kick_check.rest_release` on the GS corners | 4.4 | `python kick_rule_compare.py --stage S` |
| `acc_counts.py` | Acceptance: the repaired stage against the f5fc2e2b stage, the census of the raw restart markers, the corner table, the kick shares | 8 | `python acc_counts.py --stage S --old-stage F --cache-dir C --out A` |
| `acc_a6_12.py` | Acceptance: the idsse action times against the CalculatedFrame and the release | 4.3, 8 | `python acc_a6_12.py --stage S --old-stage F --cache-dir C` |
| `acc_zero.py` | Acceptance: the zero counts of the phase, the key and the spans | 8 | `python acc_zero.py --stage S --cache-dir C` |
| `acc_corner_windows.py` | Acceptance: the corner readers on the corners with no kick time | 5.2, 8 | `python acc_corner_windows.py --stage S` |
| `acc_blank.py` | Acceptance: the blank frame features on the unmeasured rows | 8 | `python acc_blank.py --stage S --workers 4` |

In each command, the path of the script is `docs/research/action-time-repair/scripts/<script>`. `acc_zero.py` needs
the pining data token in the environment (`PINING_FOR_THE_DATA_TOKEN`): it lists the matches from the pining
manifest.
