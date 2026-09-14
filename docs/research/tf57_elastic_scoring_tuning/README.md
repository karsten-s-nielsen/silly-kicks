# TF-57 — ELASTIC-NW `_score` OpenEvolve tuning: provenance + held-out validation

Companion provenance/validation artifact for the OpenEvolve-tuned `_score` weights shipped in
commit 1 (feature) of the TF-57 cycle (ADR-093). This is the **second commit** of the 2-phase
single-branch commit: the feature landed first (`e611aa1`), and this validation was then produced
**against that committed, clean tree** so its numbers trace honestly to the code that shipped.

- **Run:** DGX-Spark, 2026-09-14, silly_kicks 4.114.0.
- **Provenance:** `run_commit = e611aa18…` (the feature commit), `run_tree_dirty = false` — see
  `provenance.json` (the machine-readable record; git commit, OpenEvolve config, all numbers).
- **Corpus:** the 3 CC-BY Sportec Open matches (J03WMX / J03WN1 / J03WPY), Bassek et al. 2025 +
  the ELASTIC re-annotation (CC BY 4.0). Cross-source (Kim's events vs our DFL-OBJ tracking,
  vote-map anchored).
- **Clean-room:** only `silly_kicks` + the CC-BY loader were read; never the MPL reference code.

## The tuning (documented, not re-run)

The `_score` discriminative weights (`w_ba, w_pbd, w_kd, w_dyn = 0.6, 1.6, 0.8, 1.0`) + a directional
depart/arrive slope term were derived by an **OpenEvolve** LLM code-search over the `_score`
formulation on the 3-match CC-BY corpus. Config (full record in `provenance.json`): `random_seed=42`,
120 iterations, population 40 / 3 islands, migration interval 20, early-stopping on `combined_score`
(patience 30); proposer models Claude Sonnet-5 (weight 0.8) + Opus-4.8 (weight 0.2); fitness =
**score_peak (primary)** + `0.5·max(0, W2_gain) − 2·max(0, W2_regression) − 2·max(0, coverage_regression)`.
The weights are a **data-fit, not an MPL-source read** — the algorithm, categories and feature set
stay clean-room from the paper. Owner-approved 2026-09-14 (ADR-093 §16.11).

## Headline (committed code, DGX)

Reproduces the CHANGELOG/ADR headline exactly on `e611aa1`:

| metric | value |
|---|---|
| pooled W2 | **0.8624** |
| mean W2 | **0.8622** (per-match 0.8481 / 0.8474 / 0.8912) |
| min-fold W2 | 0.8474 |
| score_peak | 0.7452 |
| reception W5 | mean **0.869** (per-match 0.8961 / 0.8377 / 0.8726) |

The paper's primary metric is W2; our cross-source **0.862 is above the paper's own greedy (0.841 W2)**
and ~10 pts below its same-source NW (0.965). (Corrects the earlier stray "reception W5 0.818", which
was the old oracle-slice number carried into the corpus headline by mistake.)

## Held-out validation (the SO-2 out-of-sample confirmation)

The shipped weights were selected on the **pooled** 3-match `score_peak` — i.e. all three matches are
in-sample. To test whether the ship bar (W2 0.862) is in-sample-inflated, a proper **leave-one-match-out**
refit was run: for each held-out match, the 4 weights were re-selected on the **other two** matches
(maximising pooled `score_peak` over a 27-point grid), and W2 was measured on the held-out match.

| held-out | fit on | selected weights | fit score_peak | **held-out W2** | shipped-weights rank on fit (of 27) |
|---|---|---|---|---|---|
| J03WMX | WN1 + WPY | (0.6, 2.0, 0.8, 1.0) | 0.764 | **0.8488** | 3 |
| J03WN1 | WMX + WPY | (0.6, 1.6, 0.6, 1.0) | 0.749 | **0.8394** | 7 |
| J03WPY | WMX + WN1 | (0.6, 2.0, 0.8, 1.0) | 0.736 | **0.8929** | 8 |

**Mean held-out W2 = 0.8604** — essentially equal to the in-sample mean **0.8622** (gap 0.0018). The
LOO-selected weights all sit in the shipped neighborhood (`w_ba = 0.6` every fold; `w_pbd` 1.6–2.0;
`w_kd` 0.6–0.8), and the shipped vector ranks in the **top 3–8 of 27** on each fit-pair. **Conclusion:
the tuning generalizes out-of-sample; the SO-2 ship bar (W2 0.862) is not in-sample-inflated.**

## Honest limit

There is no strictly independent fourth held-out match — the ELASTIC benchmark re-annotates only these
three matches — so "held-out" here is a leave-one-match-out cross-validation over the three, not an
out-of-corpus test. The grid is a 27-point neighborhood, not a full search. Both are stated so the
0.8604 is read as the LOO-CV number it is.
