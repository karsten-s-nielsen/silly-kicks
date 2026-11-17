# Cross definition (ADR-133): the start-line measurement

This memo reads `cross_definition.json` next to it. The report comes from `scripts/build_cross_definition.py` at
commit `f5fc2e2b` on a clean tree (`run_commit`, `run_tree_dirty`). The run read two inputs:

- the xshot187 corpus, staged from the same commit into a fresh directory (187 matches: Gradient Sports 62,
  IDSSE 7, SkillCorner 118; stage manifest `run_commit` `f5fc2e2b`, 0 failed);
- the pinned StatsBomb open-data snapshot (revision `4b73468`, 3,961 matches), converted by the same commit.

## The pre-declared rule

ADR-133 decision 5 fixed the rule before the run. Among the candidate lines {52.5, 60, 65, 70, 75, 80} m, select
the line with the highest equal-weight mean F1 of the geometric open-play cross against the provider's own label,
over StatsBomb and Gradient Sports. A tie within 0.005 goes to the larger line. The script re-types the staged
open-play rows at each line, so the result does not depend on the line the stage was typed at.

## Result

| Line (m) | F1 StatsBomb | F1 Gradient Sports | Mean | F1 IDSSE (reported only) |
|---|---|---|---|---|
| 52.5 | 0.596 | 0.435 | 0.516 | 0.124 |
| 60 | 0.606 | 0.440 | 0.523 | 0.132 |
| 65 | 0.614 | 0.444 | 0.529 | 0.133 |
| 70 | 0.627 | 0.449 | 0.538 | 0.137 |
| **75** | **0.631** | **0.448** | **0.540** | 0.146 |
| 80 | 0.602 | 0.432 | 0.517 | 0.163 |

- **Selected line: 75.0 m.** 75 has the highest mean. 70 is within the tie band, and the tie goes to the larger
  line, so the result is 75 either way. 80 is outside the band. `CROSS_START_X` is 75.0, and
  `tests/spadl/test_classify_deliveries.py::test_the_start_line_is_the_measured_value` pins it to this report.
- **The ratio bar.** Median open-play crosses per match at 75.0: Gradient Sports 14, IDSSE 16,
  SkillCorner 17. Largest / smallest is 1.21, inside the bar of 1.5. On `main` the medians were 26.5, 4 and 101.

## What this does not measure

- F1 against a provider's label measures agreement with that provider's convention, not the truth of a cross.
  The rule is the same for every provider; the labels are not. That is why the rule is chosen by two vendors and
  judged by the ratio bar.
- IDSSE has 7 matches and a narrow DFL label (only the passes the annotation calls crosses), so its F1 is low at
  every line. It is reported, not used.
- The ratio bar compares counts per match, not which passes are crosses. The frame-level label rate per provider
  (ADR-133, the label-rate table) is the check that reaches the xCross training label.
