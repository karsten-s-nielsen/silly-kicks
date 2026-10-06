# TF-58 reduce-memory + speed — implementation plan (rev 3, categorical-only)

Design: `docs/superpowers/specs/2026-10-06-tf58-reduce-memory-design.md` (rev 3). Branch `feat/tf58-team-coordination`. Rev 3 pivots to the measured categorical-only architecture (rev-2's projection + liveness schema change dropped). TDD; every value-changing step gated by the Task 1 byte-identity harness (D-5). No DGX run until the fix lands and reviews clear. One owner commit gate.

## Global constraints
- Both pandas legs (pd2 py310 / pd3 py312) -- pd2 is where a missed `observed=True` diverges. Lint CI-scope (`silly_kicks/ tests/ scripts/`), bare pyright. Green before the commit gate.
- No metric-definition / schema change (liveness UNCHANGED). ONE owner-approved reduce-statistic change: the occlusion per-bin CI bootstrap (Task 4b / D-6 / ADR-112 decision 5) -- value-changing (occ `ci_by_bin`), the D1-reduce hotspot fix. Everything else byte-identical.
- Local `.venv` = ruthless 0.6.0 -> `calibrate_coordination` import fails; run calibrate/D2 tests on pd3 + ruthless-0.7.0 (DGX `~/tf58/venv`).

## Task 1 - byte-identity harness (D-5) FIRST
- [ ] `tests/scripts/test_coordination_reduce_byte_identity.py`: synthetic corpus (>=3 matches, >=2 providers, real period + possession windows, dyad + team entities). Capture CURRENT (object) reduce outputs as the reference (`derivation/calibration/metrics/numerics_noflip.json`, strip `*timings*`/`stage_seconds`). Gate for Tasks 2-3.
- [ ] Adversarial fixture precondition (ADR-032, discriminating): **sparse `(level, window_kind)` combos** (the `observed=` empty-group trap), finite values for every family column, every H1-H7 + stoppage + occlusion reached. Assert the reference is non-trivial (finite reliability, H-tests + occlusion curves reachable).

## Task 2 - generalized categorical combine (D-1; A1/A2/A3)
- [ ] RED: `combine_workers` returns `category`-dtype string cols for the match_tables reduces (metrics/stoppage, D2 layer-a/b/confirm, occlusion); combined table values/order == object reference; reduce output == Task-1 reference.
- [ ] Implement: read each share via pyarrow `read_dictionary` on the string cols -> `category`; cast EVERY share to the SAME `CategoricalDtype(categories=sorted(union), ordered=False)` per col BEFORE `pd.concat` so concat stays categorical AND category order is lexicographic; dtype persists through the reduce. **Do NOT rely on bare `union_categoricals`** -- its default `sort_categories=False` leaves append-order categories (sort_values diverges); pin the explicit sorted dtype (A RM-R3-02). Scope the categorical branch to the match_tables pass names (a/b/occlusion_cal/numerics unchanged). Preserve `_ORDER_COLUMNS` sort.
- [ ] Memory assertion: combined-table deep-mem per-row below the object-path size (3x metrics / 7x occlusion).
- [ ] GREEN: D-5 byte-identity on D1/D3 (object leg) -- D2 on the pd3 env.

## Task 2b - observed=True sweep (D-1b; byte-identity under categorical)
- [ ] Enumerate EVERY reduce groupby keyed on a now-categorical column: occlusion (`derive:1217,:1239,:1357,:1488`; `validate:234,:237`), `derive_constructs`/`build_constructs_report`, `reliability_over_folds`, the H1-H7 groupbys, `liveness_block`, stoppage. Pin `observed=True` on each.
- [ ] Grep/AST gate `tests/.../test_categorical_safe_reduce_ops.py`: **every categorical-semantics op on a match_tables categorical column is safe**, not just groupby (A RM-R3-01 / B RM-SPEC-11) -- `groupby`+`observed=True`; and audit/plant-check `merge`/`sort_values`/`set_index`/`astype(str)`/`.isin`/`.map` (`_window_join:93`, `_paired_axis:147`, `_team_key:102`, `_join_keys`) for dtype-safe behaviour. Planted violation fails.
- [ ] Extend the Task-1 adversarial fixture to be **per-categorical-column-consumer**: exercise each non-groupby op above so a merge/sort/index divergence under categorical is caught, not only a groupby empty-group one.
- [ ] GREEN: D-5 byte-identity on BOTH pandas legs (pd2 default `observed=False` / append-order category is where a miss bites). Liveness `n_unique`/`live` identical (computed on the full categorical table; no schema change).

## Task 3 - F2 provider_bootstrap_se index-resample (D-2; D1 speed)
- [ ] RED: `provider_bootstrap_se` (`:354`) output identical to the frame-concat version on a fixture; assert no `pd.concat` in the loop; per-draw row order preserved.
- [ ] Implement: pre-group once; resample integer indices; reduce over views.

## Task 4 - F3 hoist gap-invariant ball-carrier (D-3; pass-b speed)
- [ ] RED: `boundary_f1_by_gap` result identical to current; assert `infer_ball_carrier` called ONCE (not per gap).
- [ ] Hoist `car = infer_ball_carrier(frames)` before the gap loop; pass `carrier=car` into `possession_windows_from_frames` (`_coordination_hypotheses.py:71-72`).
- [ ] Re-measure pass-b per-stage (1 SK + 1 GS, DGX): `_pre_index_frames` 15->1, pass-b ~4x. Record the GS figure for the <=1h/16-worker reconciliation.

## Task 4b - occlusion per-bin CI bootstrap: match-grain + capped (D-6; VALUE-CHANGING, owner-approved 2026-10-07)
Added after the authoritative DGX run profiled the D1 `occ_err` reduce to this bootstrap (> 1 h single-thread). Owner decision 5 (ADR-112); changes `derivation.json` occlusion `ci_by_bin` (NOT `err_by_bin`).
- [x] Rewrite `_bootstrap_weighted_median_ci` (`derive:1191`): resample MATCHES not rows (cluster bootstrap via `pd.factorize(match_id)`; < 2 matches -> NaN) + a seeded per-match-balanced 20k-row subsample cap (`_OCCLUSION_CI_MAX_ROWS`); signature gains `match_ids`. Caller (`:1249`) passes `g["match_id"].to_numpy()`. Point estimate `err_by_bin` untouched.
- [x] 4 unit tests (`test_coordination_driver_modules.py`): match-grain vs row-grain, < 2-match NaN, per-draw cap <= 20k, seeded-deterministic. `occ_frame` fixture -> 2 matches so the CI has >= 2 clusters.
- [x] GREEN local: 4 occ-CI tests + full `test_coordination_reduce_categorical_identity.py` (D-5 object==categorical of the NEW occ-CI statistic) + driver-modules + D2 (ruthless 0.7.0). Speed: 1.5M-row bin ~1 s (was minutes). ruff clean; pyright 0 on changed files.
- [ ] Re-materialize `derivation.json` under D-6 at the authoritative DGX run.

## Task 5 - reduce-memory guard + categorical-op safety (D-4; closes ADR-073 gap)
- [x] `tests/scripts/test_reduce_categorical_safety.py`: `combine_workers(categorical=True)` returns categorical string cols + shrinks vs the object path (`test_categorical_combine_shrinks_and_is_categorical`, the discriminating memory proof).
- [x] Categorical-op safety as a **gold-standard hybrid** (not a dtype-blind AST gate, which can't see a column's dtype): (1) **groupby** `observed=True` static gate (`test_every_coordination_groupby_pivot_sets_observed_true`, derived population, ADR-056 anti-rot); (2) **non-groupby ops**: the ORDER of `sort_values`/`set_index` on a categorical key is made object-identical by the ONE invariant -- categorical cols carry SORTED unordered categories -- asserted structurally on the real combine (`test_combine_output_categoricals_are_sorted_and_cover_the_sort_keys`, which also proves `_ORDER_COLUMNS` come back categorical so D-5 exercises the order-sensitive `:464` sort), proven load-bearing by a plant (`test_sorted_categories_are_load_bearing_for_sort_order`: append-order categories diverge from object). `merge`/`astype(str)`/`isin`/`map` are NOT covered by that order invariant (merge can reorder on categorical keys even when sorted -- pd 2.3.3, TM-IMPL-06; the rest are elementwise) -- they are backstopped by D-5 byte-identity on the ACTUAL ops (object vs sorted-categorical; `_window_join`'s merge feeds order-insensitive aggregates, so its reorder is benign and D-5 green confirms identical output), with a coverage precondition in the D-5 harness (`test_fixture_is_discriminating` asserts provider/match_id/table/level/window_kind are categorical, else D-5 is vacuous).

## Task 6 - full gate + DGX pre-measurement
- [ ] Full `-m "not e2e"` both legs, lint CI-scope, pyright bare. calibrate/D2 on pd3 + ruthless-0.7.
- [ ] DGX pre-measure (clean tree, no `--allow-dirty`): metrics + occlusion + one D2-layer-b combine peak RSS < ceiling (categorical; RSS sampler); pass-b re-time post-hoist (~4x). **The metrics-combine peak RSS is an EXPLICIT GO/NO-GO** (A RM-R3-03): ~58 GB estimated / ~2x headroom; if the measured peak approaches the ceiling, STOP and surface to the owner (projection is the recorded escape) -- do NOT launch the ~2-day authoritative run on an estimate.

## Task 7 - docs + owner commit gate
- [ ] ADR: the categorical reduce-memory architecture (generalized categorical combine + `observed=True` discipline + the guard closing the ADR-073 gap); the F3 budget reconciliation (≤45s/match superseded by corpus <=1h/16-worker; post-hoist GS figure). AGENTS.md convention bullet if a durable rule emerges (reduce-memory = categorical + observed=True).
- [ ] **Record explicit owner decisions** at the gate/ADR: (a) rev-3 pivot (categorical-only, projection/liveness-schema dropped); (b) F3 budget supersession; (c) `_pre_index_frames` de-Arrow deferral; (d) the metrics 3x headroom (projection deferred unless the corpus grows far larger).
- [ ] `/final-review`; fix findings. Independent implementation review (owner-arranged) before commit.
- [ ] **OWNER COMMIT GATE** -- show `git status --short`, `git diff --stat`, proposed message. Amend into commit-1 or a new commit (owner decides). No commit/push without explicit per-commit approval.

## Task 8 - re-run authoritative DGX cycle (after commit + CI green)
- [ ] Wipe `~/tf58/out`; `git fetch`+checkout the fixed commit; verify clean + fixes present; reuse `~/tf58/slices` + `~/tf58/cache`.
- [ ] D1 -> D2 -> D3 -> numerics (homogeneous worker slices). Verify step-7 provenance (`run_commit`, `run_tree_dirty:false`, `derivation_sha256`, `population_checked_against: corpus_json`, `n_failed:0`).
- [ ] Hand the 6 artifacts back for commit-2.

## Citation ledger
`derive_constructs` period filter `:161`; `_joint_source` `:742`; `_confirm` `:648`; numerics `:271` = reduce combine; hyp `:186` = `_has` guard; F2 = `provider_bootstrap_se` (`:354`/`:378`); occlusion groupbys `derive:1217/1239/1357/1488`, `validate:234/237`; guard anti-rot ADR-056 (+ ADR-073). Measured: metrics 38,914 (SK)/65,102 (GS) rows, 126 cols, 86/142 MB; categorical 3.0x (metrics)/7x (occlusion); pass-b 166.6 s/match, `_pre_index_frames` 79/87%.
