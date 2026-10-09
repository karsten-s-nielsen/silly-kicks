# TF-58 D2 layer-a memory — implementation plan (per-variant share split, option C)

Design: `docs/superpowers/specs/2026-10-07-tf58-d2-layer-a-memory-design.md`. Branch `feat/tf58-team-coordination` (continues the reduce-memory cycle; D1 artifacts at the fix-base commit reused, D1 does NOT re-run). TDD; every value step gated by the Task-1 FULL-confirm byte-identity harness. No DGX re-run of D2 until the fix lands + reviews clear. One owner commit gate.

## Global constraints
- Byte-identical D2 outputs: `calibration.json` (incl `hypotheses`/`gate_cleared`/`fallback_reason`/selections) + `_provider_params_generated.py`. The split changes on-disk layout only; the full melt is retained per variant.
- No projection / no column- or row-drop. The rejected-A failure mode (silent gate flip) cannot occur because nothing is dropped — the Task-1 harness nonetheless drives the full confirm to prove it.
- D2 reliability + confirm run on ruthless 0.7.0 (pd3/DGX; local `C:\Python314` user-site now 0.7.0). The split + read-path functions are pure-pandas → unit-testable on the local leg.
- Lint CI-scope (`silly_kicks/ tests/ scripts/`), bare pyright. Green before the commit gate.

## Task 1 — FULL-confirm byte-identity harness (D-5, strengthened) FIRST
- [ ] `tests/scripts/test_d2_layer_a_share_split_identity.py`: synthetic corpus (≥2 matches for CV folds; the H1–H7 tables present — pair/pair_phase/spectral/cluster_team/rsi/windows/rsi_switch_times/possession_changes; ≥2 post-prep variants incl `base`; dyad+team entities; include_switch_events=True). Run the layer-a → combine_levels → **full `_confirm`** path on the CURRENT stacked layout (reference) and assert identical on the per-variant layout: the **WHOLE `calibration.json`** minus the named volatiles only (`*timings*`, `stage_seconds`, `run_commit`/`run_tree_dirty`/`run_tree_state`) — i.e. incl `hypotheses`/`hypotheses_pass`/`gate_cleared`/`fallback_reason`/`moved_multipliers`/selections AND `population` + `objective_ids` (D2-SPEC-05: NOT a named subset, so a population/objective_id drift fails). Assert `derivation.json` (D1) untouched.
- [ ] Precondition (ADR-032, discriminating): assert the reference confirm is non-vacuous — ≥2 CV folds, finite reliability for ≥2 families, ≥1 H-test reaches a finding (not all skipped by `_has`), both `base` and a post-prep variant reachable. Else the identity passes vacuously (this is exactly what hid the A defect).

## Task 2 — per-variant share-write (bounded working set)
- [ ] Split the baseline worker output into one share per `variant`: read the per-match shards (`CorpusPassResult.shard_keys`/`shard_path`), emit per-variant shares one variant at a time (peak ≈ one variant, not the 13× stack). Share name `level_share_name(BASELINE_LEVEL, variant)`; manifest per variant (so `combine_workers` completeness is per-variant, manifest-based — unchanged semantics).
- [ ] RED→GREEN: per-variant shares each carry exactly one variant's full melt (all tables/cols for that variant); their union == the stacked share (no row/col loss); peak materialised frame « the stacked size (memory assertion).

## Task 3 — per-variant combine + read-path
- [ ] `combine_levels`: for the baseline, iterate its variants and `combine_workers(categorical=True)` per variant → `level_combined_path(dest, BASELINE_LEVEL, variant)`. Non-baseline prep levels unchanged (single `base`). **Keep `summaries["baseline"]` a SINGLE entry** (D2-SPEC-05): assert the per-variant combine summaries agree on `generation` + `population_digest` (same shard generation, same match set across variants, else refuse) and fold them to one baseline summary → `generation_tokens`/`objective_id`/`_population` unchanged → whole `calibration.json` byte-identical.
- [ ] Read-path: `level_combined_path(out, level, variant=...)`; update every reader — `CoordinationReliabilityObjective._frame_for` (:281), `_joint_source` (:720, both ≤1-moved branches), `_scored_pairs` (:474) — to read the per-variant file (no more `frame["variant"]==variant` on a stacked file).
- [x] `_layer_joint` (:595): emits one joint variant → one share. **NOT wrapped** (IMPL-02 / ADR-113 decision 3, owner-approved accept-via-test 2026-10-07): single-variant by construction (`variants={"joint": …}`), so `write_worker_partial_by_variant` would write one identical share — a functional no-op, never part of the 56 GiB stack. Schema consistency with the per-variant baseline files is guaranteed BY CONSTRUCTION (all are `match_tables` output) + confirmed by `test_joint_variant_schema_matches_baseline`, not by a spurious wrap. (The earlier "wrap `_layer_joint.work` too" was over-broad; corrected here.)
- [ ] Bump `_SHARD_SCHEMA_VERSION` (layout change) so a stale stacked share never resumes against the per-variant combine.
- [ ] GREEN: Task-1 full-confirm byte-identity on the per-variant layout.

## Task 4 — read-path completeness + schema anti-rot gate (ADR-056)
- [ ] `tests/scripts/test_d2_share_split_readpath.py`: derive the set of `level_combined_path` readers by grep/AST over `calibrate_coordination.py` + the `_coordination_hypotheses` call graph; assert each resolves a variant (no reader left reading a stacked file); assert all per-variant combined files (baseline variants + prep levels + joint) share ONE schema. Plant: a reader left on the stacked path, or a schema-divergent joint file, fails. Register in the anti-rot meta-assertion.

## Task 5 — full gate
- [ ] `tests/scripts/` D2 suite + the new tests, both legs where runnable (full confirm on pd3/0.7.0; split/read-path pure-pandas local). ruff CI-scope, pyright bare. No new skips.
- [ ] Measure per-variant worker-share peak + per-variant baseline-combine peak (1 SK worker + a 1-variant combine, DGX) « the stacked sizes; record for the re-run go/no-go.

## Task 6 — owner commit gate
- [ ] Present the diff + ADR-113 (write at commit-prep, as ADR-112; records C + why A was rejected). No version bump (commit-2 is the artifact commit). OWNER COMMIT GATE → CI.

## Task 7 — authoritative D2→D3→numerics re-run at the fix commit
- [ ] Re-checkout `~/tf58/sk` clean to the fix commit; KEEP `~/tf58/out/d1` (derivation.json reused), wipe `out/{d2,d3,nf}`. Run D2 (11 levels + b + joint + confirm) → D3 → numerics. Per-variant worker-peak + baseline-combine peak-RSS are the explicit go/no-go. → the 6 commit-2 artifacts → Task-28 commit-2 bookkeeping.
