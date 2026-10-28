"""The REVIEWED serial reasons (spec r3 D8).

The R1 audit (`campaign.audit_utilization`) passes a stage below the box's vCPU only through one of the spec's two
structural shapes (a final fit at the full thread budget; a sequential study at its fold width) or a reviewed
serial reason. A stage declares its reason as ``Stage.serial_reason`` (or ``ItemStage.serial_reason``); this
registry holds every declaration, keyed ``(pipeline module, stage name)``, with the reason text verbatim.
``tests/scripts/test_serial_reasons.py`` holds the declarations and this registry EXACTLY equal, so a new serial
stage needs a reviewed entry here and a stale entry fails too. Entries are added with the plan's R3-8 per-stage
decision table, which goes to the A/B review.
"""

from __future__ import annotations

#: ``(pipeline module, stage name)`` -> the stage's reviewed ``serial_reason``, verbatim.
SERIAL_REASONS: dict[tuple[str, str], str] = {
    # HFU plan R3-10 (A+B approved 2026-10-30): "a reviewed `serial_reason` because each TPE trial reads every
    # earlier trial".
    ("scripts.train_xsuccess", "study"): (
        "Optuna TPE: each trial reads every earlier trial, so trials cannot run in parallel without changing the "
        "search; with hpo_trials=0 (the campaign default) the unit is a no-op"
    ),
    # HFU plan R3-8 S3 item 12: the SERIAL annex, verbatim (owner ruling 2026-10-30, "too small to pay"; the
    # R1 audit checks each against its measured unit and reduce CPU, `campaign.audit_utilization`).
    ("scripts.train_ghost_gk", "validate"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 1 s of CPU in a 19-22 s wall on "
        "HF; the wall is the visibility check reading each corpus shard's parquet footer in turn through the bucket "
        "mount (about 0.18 s a file)."
    ),
    ("scripts.train_ghost_gk", "select"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 179 column projections, 21-24 s of CPU in "
        "all; a W=8 job is 8.5-9.3 s, mostly worker start-up, and W=16 measured slower (11.7 s)."
    ),
    ("scripts.train_ghost_gk", "cache"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 4 s of CPU, combining the select"
        " outputs into the feature cache."
    ),
    ("scripts.train_ghost_gk", "assemble"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 3 s of CPU, building the metrics"
        " block and the bundle from the cv, fit and importance results."
    ),
    ("scripts.train_ghost_outfield", "assemble"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit of 3.5 s, building the metrics "
        "block and the bundle from the cv and fit results."
    ),
    ("scripts.train_xshot_occurrence", "prep"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 6 s of CPU, the arm-restricted "
        "design matrix from the extract results."
    ),
    ("scripts.train_xshot_occurrence", "fit"): (
        "Too small to pay (under 60 s of unit and reduce CPU, owner ruling 2026-10-30): one xgb fit, 26 s of CPU at one"
        " core with the dead lever (S1), 7.4 s at 5.84 cores with the lever fixed (S1d); a start-up-bound job of under "
        "30 s."
    ),
    ("scripts.train_xshot_occurrence", "select"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 1 s of CPU, choosing the "
        "parameters from the study and cv results."
    ),
    ("scripts.train_xshot_occurrence", "assemble"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 1 s of CPU, the metrics block "
        "and the bundle."
    ),
    ("scripts.train_xcross_attempt", "probe"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): two units, one probe-match extraction "
        "each, 45 s of CPU in all and a 24 s span at W=2."
    ),
    ("scripts.train_xcross_attempt", "probe_comparison"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 4 s of CPU, comparing the two "
        "probe extractions."
    ),
    ("scripts.train_xcross_attempt", "prep"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 4 s of CPU, the arm-restricted "
        "design matrix from the extract results."
    ),
    ("scripts.train_xcross_attempt", "fit"): (
        "Too small to pay (under 60 s of unit and reduce CPU, owner ruling 2026-10-30): one xgb fit, 15 s of CPU at one"
        " core with the dead lever (S1); with the lever fixed the job is 9.3 s at 3.77 cores, half of it start-up "
        "(S1d)."
    ),
    ("scripts.train_xcross_attempt", "select"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 1 s of CPU, choosing the "
        "parameters from the study and cv results."
    ),
    # HFU S3 review S3B-04 (owner rulings 2026-10-31): the new TF-19 and win_probability stages, measured on HF.
    ("scripts.train_xcross_attempt", "tf19_ablation"): (
        "Reviewed floor (owner ruling 2026-10-31, S3B-04): 10 single-thread xgb fits of 8-19 s; the longest unit "
        "bounds the wall. On HF every parallel shape took 30.6-32.3 s (W=8 x 1: 32.3 s, 162 CPU-s, 0.66; W=10 x 1: "
        "0.78-0.81; W=5 x 2: 0.79; job 6ae62901c78863f9c62f3a63), so the stage runs pinned at W=8, the least CPU. "
        "The n_jobs-lever shape (W=1 x 8) took 58.8 s and 327 CPU-s (job 6ae6254b0e0338d1bb7d3760)."
    ),
    ("scripts.train_xcross_attempt", "tf19_probe"): (
        "Reviewed floor (owner ruling 2026-10-31, S3B-04): one single-thread unit, the TF-19 substitution probe, "
        "68-81 s of CPU on HF (job 6ae629e00e0338d1bb7d3825). It cannot split inside one stage: one sequential RNG "
        "draws the frame sample and then each frame's placebo picks, so a split needs a second stage and its job "
        "start-up for about 60 s of gain."
    ),
    ("scripts.train_receiver_model", "extract"): (
        "Reviewed floor (owner ruling 2026-10-31, row 29): 30 per-match units at W=8 measured 0.844 and 0.845 of the "
        "vCPU on HF (S1i), just under the 0.85 bar. A per-(match, period) split (90 units) gave the same bytes but "
        "0.797 and 0.829 (S3 item 16 (c), job 6ae611200e0338d1bb7d327a), so the stage keeps per-match units."
    ),
    ("scripts.train_receiver_model", "pool"): (
        "Too small to pay (under 60 s of unit and reduce CPU, owner ruling 2026-10-30): in the campaign configuration "
        "no pool dataset is passed, so the stage has no units and only an empty reduce; the job is 6.4 s. With a pool "
        "dataset it has units and is measured like `extract` (S3 item 16)."
    ),
    ("scripts.train_receiver_model", "fit"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 2 s of CPU."
    ),
    ("scripts.train_receiver_model", "deploy"): (
        "Too small to pay (under 60 s of unit and reduce CPU, owner ruling 2026-10-30): in the campaign configuration "
        "no deployment dataset is passed, so the stage has no units and only an empty reduce; the job is 5.4 s."
    ),
    ("scripts.train_receiver_model", "assemble"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, under 1 s of CPU."
    ),
    ("scripts.train_win_probability", "refs"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 3 s of CPU in a 6 s wall, "
        "listing the match references."
    ),
    ("scripts.train_win_probability", "cv"): (
        "Reviewed floor (owner ruling 2026-10-31, S3B-04): the 5 GroupKFold folds of the calibration (fixed at 5), "
        "each a single-threaded GLM fit on its training matches and a serve of its held-out matches. On HF the stage "
        "took 85 s, 395 CPU-s and 4.64 cores (0.58) at W=5 (job 6ae624e7c78863f9c62f3947); a finer split re-fits a "
        "fold model per chunk or adds a second stage."
    ),
    ("scripts.train_match_outcome_dependence", "refs"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 3 s of CPU in a 7 s wall, "
        "listing the match references."
    ),
    ("scripts.train_match_outcome_dependence", "fit"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 4 s of CPU."
    ),
    ("scripts.build_territory_ranking_census", "refs"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, under 1 s of CPU."
    ),
    ("scripts.build_territory_ranking_census", "fit_prepass"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 32 units of 0.6-0.9 s, 15 s of CPU in all;"
        " a W=8 job is 9.4 s, mostly worker start-up."
    ),
    ("scripts.build_territory_ranking_census", "prefit"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 1 s of CPU."
    ),
    ("scripts.build_territory_ranking_census", "score"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 32 units of 0.6-0.9 s, 20 s of CPU in all;"
        " a W=8 job is 10.4 s, mostly worker start-up."
    ),
    ("scripts.build_territory_ranking_census", "bootstrap"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 0.1 s of work in a 6 s job; its "
        "idle cores are start-up, not structure."
    ),
    ("scripts.build_territory_ranking_census", "report"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, under 1 s of CPU."
    ),
    ("scripts.train_pass_completion", "refs"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, under 1 s of CPU."
    ),
    ("scripts.train_pass_completion", "extract"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 64 units, 28 s of CPU in all; W=16 "
        "measured slower than W=8 (14.8 s against 13.3 s) and spent 58 % more CPU."
    ),
    ("scripts.train_pass_completion", "cv"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 5 folds of 0.3 s, 1 s of CPU in all."
    ),
    ("scripts.train_pass_completion", "fit"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, under 1 s of CPU."
    ),
    ("scripts.train_xsuccess", "refs"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, under 1 s of CPU."
    ),
    ("scripts.train_xsuccess", "extract"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 64 units, 20 s of CPU in all; a W=8 job is"
        " 9.8 s, mostly worker start-up."
    ),
    ("scripts.train_xsuccess", "cv"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 5 folds, 36 s of CPU in all; at W=5 the "
        "job is 13.4 s."
    ),
    ("scripts.train_xsuccess", "fit"): (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one xgb fit, 9 s of CPU."
    ),
}
