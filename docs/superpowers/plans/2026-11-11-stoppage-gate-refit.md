# Stoppage gate and xShot/xCross refit — implementation plan

> Execute inline with `superpowers:executing-plans` (no subagents, no worktree). Steps use checkboxes.
> r2 (2026-11-11): answers plan reviews A (APPROVE; SGR-PLAN-01 to 06) and B (APPROVE; SG-P01 to 07). Task 1 is
> done (record: `phase0.md`, "Build — Task 1 complete").

**Goal:** dead-ball frames of SkillCorner and Gradient Sports leave xShot/xCross training; the refit campaign file
and `refit` mode exist; the three datasets are re-staged and locked.
**Spec:** `docs/superpowers/specs/2026-11-11-stoppage-gate-refit-design.md` (r4, owner-approved). Read it first.
**Base:** branch `tf-46-arm2-clean` = `origin/main` `2a0cb92b`.

## Global constraints

- Owner yes, per action: every commit, push, PR action, upload, `datasets record`, merge. Merge commit only.
- Minimal and clean: no code beyond the spec. Reuse main's seams. No new file that the spec does not name.
- Tests first (RED, then GREEN). The owner's PC runs edits, `git`, `ruff` and ONE small test file at a time. Every
  other test run, `pyright` and every real-data run use the DGX copy `~/sk-tf46-wt` (owner-approved): stage the
  change, `git diff --binary --cached 2a0cb92b > /d/Temp/tf46-recut/wt.patch`, sync with
  `D:\Temp\tf46-recut\port\dgx_sync.sh`, run with `dgx_run.sh`; record the patch sha256 with each result. Use
  `.venv/bin/python -m pytest` there, never `uv run` (it re-syncs the environment).
- A file is staged when it is written. Temp files under `D:\Temp\tf46-recut\`. Prose in ASD-STE100.
- Records: `D:\Temp\tf46-recut\records\phase0.md` (evidence only). Review summaries go to `reviews_log.md`.

## Tasks

### Task 1: `silly_kicks/tracking/_stoppages.py` (spec 2, 4.1)

DONE. Names: `STOPPAGE_RULE_VERSION = 1`, `BALL_OUT_MARGIN_M = 0.11`, `STOPPAGE_STATES`, `StoppageSpans` (the spans
plus the on-ball event anchors that rules 1 and 2 need; review B SG-P01), `skillcorner_stoppage_spans(events)`,
`event_stoppage_spans(actions)`, `with_stoppage_state(frames, *, spans) -> DataFrame` (pure; private and NOT
exported, so no `add_*` registry takes it in; review A SGR-PLAN-02),
`training_frame_mask(frames, *, horizon_seconds) -> np.ndarray` (per row: `live`, and no `unknown` frame of the same
match and period in `[t, t + horizon]`).
- [ ] Write the tests of spec section 5 for these functions (each table row; unpaired markers; period edges from the
  tracking period with a wrong event label; rule 1 at 0.10 m and 0.12 m beyond each line on a SkillCorner and a
  Gradient Sports frame table; a frame without a ball; precedence `unknown` > `dead` > `live`; the horizon rule; the
  input frames unchanged). Run on the PC: FAIL (no module).
- [ ] Implement. Vectorized (`np.searchsorted` on frame ids or times; no row loop; `group_rows` where a per-game
  split is needed, ADR-068). `spans` is the output of one of the two span functions, or `None` for an observed
  provider (state from `ball_state`). Its purity is tested in its own test file.
- [ ] Run the test file on the DGX copy: PASS; three mutants fail it.

### Task 2: Gradient Sports leaves `_DEAD_BALL_OBSERVED_PROVIDERS` (spec 3, 4.2)

- [ ] Change `tests/tracking/test_dead_ball_taxonomy.py:15` to `{"sportec", "idsse"}`; add to
  `tests/coordination/test_windows.py`: `resolve_stoppages` on a Gradient Sports toy match in `auto` mode returns the
  event evidence; `mode="ball_state"` raises. Run: FAIL.
- [ ] Edit `silly_kicks/tracking/_provider_visibility.py:32` and its comment (owner rulings 2026-10-26, 2026-11-11).
- [ ] Run both files and `tests/coordination` on the DGX copy: PASS.

### Task 3: the training filter (spec 4.4)

- [ ] Tests in `tests/tracking/test_xshot_occurrence.py` and `test_xcross_attempt.py`: with `stoppage_state`, a
  `dead` frame, an `unknown` frame and a `live` frame whose horizon reaches an `unknown` frame are not rows; without
  the column the output equals main's. The oracle (reviews A SGR-PLAN-03, B SG-P05): before the edit, run both
  functions on the DGX copy reset to `2a0cb92b` on the test's frames, and pin the sha256 of the output rounded to
  9 decimals in the test; the existing tests of both functions also stay green unchanged. Run: FAIL.
- [ ] `prepare_xshot_training_data`: where it filters `ball_state == "alive"` (`:863`), filter
  `training_frame_mask(frames, horizon_seconds=horizon_seconds)` when `stoppage_state` is present.
  `prepare_xcross_training_data`: the same rows are skipped in its frame loop (`:455`), keyed by
  `(game_id, period_id, frame_id)`, so its carrier inference and goal map still see the full frames (as main does).
- [ ] Run both test files on the DGX copy: PASS.

### Task 4: the trainers (spec 4.4, 4.5)

- [ ] Tests in `tests/scripts/test_train_xshot_occurrence.py` and the xCross one (find them with `git grep -l`):
  the extract stage raises on a SkillCorner match without `stoppage_state`; the xCross probe stage (`_probe_work`,
  `scripts/train_xcross_attempt.py:620`) raises on a Gradient Sports probe match without it, and drops its
  non-live frames (reviews A SGR-PLAN-01, B SG-P07); a toy run's
  `metrics.json` holds `stoppage` (`rule_version`, and per provider: `source` counts, `dropped_dead`,
  `dropped_unknown`, `dropped_horizon`, `labels_in_dropped`); the saved model's `metadata.json` holds
  `stoppage_provenance` (`rule_version` and the `stoppage_source` counts per provider; spec 4.5; review B SG-P03).
  Run: FAIL.
- [ ] `_extract_work` of both trainers: fail closed (spec 4.4); compute the per-match counts and carry them as
  side columns (broadcast, collision-checked like `_SIDE_COLS`); the assemble stage sums them per provider into
  `metrics["stoppage"]`. Both models: a `stoppage_provenance` attribute (a dict), saved and loaded like
  `training_commit` (absent in an old bundle = `None`). The model cards change with the refit weights in C3 (Part
  Deux; the handover note says so).
- [ ] `scripts/_toy_corpora.py`: the SkillCorner and Gradient Sports toy frames carry `stoppage_state` and
  `stoppage_source` (one dead span each), so the parity tests run.
- [ ] Run the trainer test files and the two trainers' parity tests on the DGX copy: PASS.

### Task 5: staging (spec 4.3)

- [ ] Test in `tests/scripts/test_loader_pining*.py` (find it): `_build_match` returns frames with
  `stoppage_state` for `skillcorner` (spans from the events file), `gradientsports` (from the actions) and `idsse`
  (from `ball_state`), with the provider builders replaced by toys. Run: FAIL.
- [ ] `scripts/_loader_pining.py` `_build_match` (`:614`): call the span function and `with_stoppage_state` per
  provider
  (SkillCorner reads `paths["events"]` for the markers). `scripts/materialize_tc3_frames.py:62`:
  `_SHARD_SCHEMA_VERSION = "tc3-frames-4"`.
- [ ] Run the loader and materializer test files on the DGX copy: PASS.

### Task 6: `refit` mode and the campaign file (spec 4.6)

- [ ] Tests in `tests/scripts/hf_jobs/test_campaign.py` and `test_adopt.py`: `load_campaign` accepts `mode` =
  `rematerialize` (default) or `refit` and refuses any other; in `refit` mode `_verify_one` runs no `_integrity`, an
  arm with all `acceptance` gates true passes, one false or an empty block fails, and the committed-model comparison
  is in the row; `rematerialize` is unchanged. A test that `tf46_refit.toml` holds exactly the 8 instances (xShot,
  xCross x `public`, `sc_extended` x `default`, `position_only`) with `mode = "refit"`. Run: FAIL.
- [ ] `Campaign.mode` in `scripts/hf_jobs/campaign.py`; `verify` passes `c.mode` to `_verify_one`. Write
  `scripts/hf_jobs/campaigns/tf46_refit.toml` from `cycle1.toml`'s xShot/xCross blocks and resource keys, with its
  own `concurrency_cap = 8`. `cycle1.toml` is NOT changed in this cycle (Ruling, Task 6): removing its xShot and
  xCross blocks breaks about a dozen adopt and campaign tests that use them as fixtures (review A SGR-PLAN-04);
  the Part Deux cycle that plans launch 9 removes them.
- [ ] Run `tests/scripts/hf_jobs` on the DGX copy: PASS.

### Task 7: the real-data check before staging (spec 5, last item; read-only, DGX)

- [ ] Script `D:\Temp\tf46-recut\port\stoppage_check.py`: the shipped functions through the loader on 20
  SkillCorner matches and the 2 Gradient Sports probe matches (pining cache on the DGX). Expected: zero `live`
  frames with the ball out by more than 0.11 m; a failure is a code defect: STOP and fix (review B SG-P07). Record
  the shares and the marker miss rate in `phase0.md`; show them to the owner with the staging request.

### Task 8: documents (spec 4.8, section 6)

- [ ] `docs/superpowers/adrs/ADR-130-stoppage-gate.md`; `CHANGELOG.md` entry (Gradient Sports out of the observed
  set; TF-58 behaviour; the schema token; the refit mode); the `TODO.md` row for the stale TF-58 artifacts (owner yes
  given 2026-11-11); `docs/howto/hf-jobs-training.md` (the `mode` key); the `AGENTS.md` TF table line for
  `_stoppages.py` within its budget; the handover note `D:\Temp\tf46-recut\records\handover_partdeux.md`.

### Task 9: C1

- [ ] On the DGX copy: `ruff check`, `ruff format --check`, bare `pyright`, every test file named above, the gates
  (`tests/test_add_star_purity.py`, `tests/scripts/test_compute_contract.py`, `tests/test_agents_md_budget.py`,
  the parity tests of both trainers). The owner sets the version and the PR-S number at commit-prep (the wheel
  changes; review B SG-P07).
- [ ] `/final-review`; the A/B implementation review; the diff to the owner; his yes; commit; push; PR; CI green.

### Task 10: staging and C2 (each step with the owner's yes)

- [ ] Before the uploads, tell the Part Deux session (review A SGR-PLAN-06). On the DGX clone at the pushed C1: the
  three materializer runs with `--match-ids-json` from the lock's units;
  `flat_to_pergame` for the probe; each unit list equals the lock's before upload; upload; `datasets record`.
- [ ] C2 (the lock) with its yes; push; CI green; merge by a merge commit with its yes; message the Part Deux
  session with the handover note.
