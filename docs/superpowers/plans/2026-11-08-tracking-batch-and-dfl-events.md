# Tracking batch performance, public seams, multi-game linking, and DFL event derivation — Implementation Plan

> **For agentic workers:** Execute this plan with superpowers:executing-plans, inline in one session. Do not use
> subagents and do not use worktrees (owner rule, `~/.claude/CLAUDE.md`). Steps use checkbox (`- [ ]`) syntax for
> tracking. **There are no per-task commits.** The whole cycle is one commit (Task 17), made only after an explicit
> owner yes.

**Revision:** r3 (2026-11-08). r2 applied plan review round 1 (A1-PLAN-45…59, TB-PLAN-01…14) and the owner rulings
R10–R12 (spec r6.2 §7). r3 applies round 2 (A2-PLAN-48/50 partials and 60–66; TB-PLAN-05 partial and 15–18). r4 applies round 3
(A3-PLAN-67–70, TB-PLAN-19). r4.1 applies A4-PLAN-71/TB-PLAN-20. r4.2 applies owner ruling R13 and the retirement of MEDIA-PC. r4.3 applies the r5 delta review (A5-PLAN-72–76, TB-PLAN-21–25). r4.4 applies the r6 delta review (A6-PLAN-77–79, TB-PLAN-26–27). r4.5 applies the r7 delta review (A7-PLAN-80, TB-PLAN-28). r4.6 applies the r8 delta review (A8-PLAN-81–82, TB-PLAN-29–30). r4.7 applies the r9 delta review (A9-PLAN-83–85, TB-PLAN-31–32). **r5 (2026-11-08)** adds handoff 3 (shape-graph positions, spec r7.1, owner rulings R14–R16): new Tasks 10A and 11A, and changes to Tasks 1, 2, 10, 11, 15, and 16. r5.1 applies the round-11 delta review (A11-PLAN-91–94, TB-PLAN-33–35). r5.2 applies the round-12 notes (TB-PLAN-36–37, A12-PLAN-95) and the owner's 20-match ruling (spec r7.3). **r5.3 (2026-11-09)** adds Task 3A (owner ruling R17) and applies owner ruling R18 (spec r7.4): the W0 run gets finer units, peak RSS instead of `tracemalloc`, and a run budget (Tasks 2, 3, 15, and 17). **r5.4 (2026-11-09)** applies the plan-r14 findings (A14-PLAN-96–101, TB-PLAN-39–44) and the owner rulings R19 and R20 (spec r7.5): a new Task 3B, and changes to Tasks 2, 3, 3A, 15, 16, and 17. **r5.5 (2026-11-09)** applies the plan-r15 findings (A15-PLAN-102–103, TB-PLAN-45–49) to Tasks 2, 3, and 3B. **r5.6 (2026-11-09)** makes the `ensure_base_tree` takeover exclusive (A16-PLAN-104 = TB-PLAN-50); disposition `…-plan-r16-disposition.md`. **r5.7 (2026-11-09)** bounds the group wait and makes the takeover test able to fail (A17-PLAN-105–106 = TB-PLAN-51–52); disposition `…-plan-r17-disposition.md`. **r5.8 (2026-11-09)** applies the r18 notes (A18-PLAN-107, TB-PLAN-53) after A and B approved r18; disposition `…-plan-r18-disposition.md`. **r5.9 (2026-11-09)** applies the owner rulings R21–R23 (spec r7.10): the member slice, `profile` on the smallest match, and the sample pilot (Task 2); a new Task 3C (the staged HF launch); the W0 legs on the DGX, the Mac Mini, and HF, with no local PC leg (Tasks 3, 15, and 17); and the job-private base tree dropped (Tasks 2 and 17). **r5.10 (2026-11-09)** applies the r20 B findings (TB-PLAN-54–59): the slice rule, one `--matches` format, the budget clock, the CPU stamp, and the upload stop; and the r20 A findings (A20-PLAN-108–113): the CPU recovery path, the parity-test count, the slice stamp, and the seeded in-job index. **r5.11** applies the r21 notes (TB-PLAN-60–61, A21-PLAN-114–116). **r5.12 (2026-11-09)** applies owner ruling R24 (half of the largest match; Task 2, Task 3 step 5, Task 15) and the owner's trim of the W0 toy parity test (two toy matches). **r5.24 (2026-11-10)** applies TB-PLAN-80 (where the launch-id lines go). **r5.23 (2026-11-10)** applies A32-PLAN-127 and TB-PLAN-79 (an empty launch id never matches) and TB-PLAN-78 (reducer markers). **r5.22 (2026-11-10)** applies TB-PLAN-77 and A29-PLAN-126 (a per-launch id; no separate reduce rule). **r5.21 (2026-11-10)** applies TB-PLAN-76 (the probe's own-job rule uses the ledger `job_label`). **r5.20 (2026-11-10)** applies the r29 findings (A29-PLAN-126, TB-PLAN-73–75) to Task 3D. **r5.19 (2026-11-10)** applies the r28 findings (A28-PLAN-122–125, TB-PLAN-66–72) to Task 3D and Task 15. **r5.18 (2026-11-10)** adds Task 3D: owner rulings R25-R27 (spec r7.16) from the failed base legs on `bb0c601`. **r5.17 (2026-11-09)** corrects the R21 block to 10 % per end (20 % in all; spec r7.15). **r5.16 (2026-11-09)** applies TB-PLAN-65 (the after-run writes to its own folder). **r5.15 (2026-11-09)** applies A24-PLAN-121 (one after-run command; `leg_cut` sorts the stem lists). **r5.14 (2026-11-09)** applies the r24 note TB-PLAN-64 (`--parity-tests` in every final leg) and renames `CompareRefusal` to `CompareRefusalError` (ruff N818). **r5.13 (2026-11-09)** applies the r23 findings (TB-PLAN-62-63, A23-PLAN-117-120): the CLI and campaign args match the code, the explicit `--profile-matches`, the cut stamp, `CompareRefusalError`, and the whole-match W3 peak. Dispositions (`D:\Development\_reviews\`): rounds 1–13 `2026-11-08-tracking-batch-dfl-events-plan-disposition.md`; rounds 14–15 `2026-11-09-tracking-batch-dfl-events-plan-r14-disposition.md` and `…-plan-r15-disposition.md`.

**Goal:** Make the silly-kicks tracking compute path fast and correct for the lakehouse, with public entry points
only. The changes are bit-identical except where the owner approved a value change.

**Architecture:** The plan has two parts, and both sit behind public seams that keep the hexagonal purity rule
(pandas in, pandas out). The speed part replaces per-frame pandas work with columnar indexes and batched kernels,
gated by a bit-identity protocol against frozen oracles. The correctness part makes the action→frame linker
game-aware and binds pointers to actions positionally, with a fingerprint check.

**Tech Stack:** Python 3.10+ (the CI matrix is 3.10, 3.11, 3.12), pandas 2/3, numpy ≥ 2, scipy, numba (core after
#299), pytest, ruff, pyright.

**Spec:** `docs/superpowers/specs/2026-11-08-tracking-batch-and-dfl-events-design.md` (r7.3; r6.1 approved by A, B,
and the owner on 2026-11-08; r6.2/r6.3 add the owner rulings R10–R13; r7.1, approved by A and B, adds handoff 3 and
R14–R16). Read the spec section that each task cites. The
plan does not repeat the spec's rationale.

**Amendment r5.18 (2026-11-23): this cycle lands first** (spec section 7, ruling 2026-11-23). This block wins over
r5.17 where they disagree.

- The r5.17 condition "only after main is quiet and the owner says go (C6)" is void. The owner said go.
- **Task 15 step 1 is done:** the branch is on `main` at `6eee84ce`; the work stays uncommitted.
- **Before the freeze,** fix reviewer C's TBC-IMPL-10 (one pass splits the distinct requests over the chunks,
  with a scale guard) and TBC-TEST-11 (a test whose frame has the ball as its first row), by TDD.
- **After the owner's commit yes and the push:** CI green, then one run of the existing parity campaign
  (`scripts/hf_jobs/campaigns/cycle1.toml`) on that commit, compared with the adopted baseline (C's decision 2,
  TBC-GATE-04). The launch needs the owner's yes. Then the PR, the merge and the release, each with its own yes.
- **C5 after the freeze** changes docstrings and `scripts/` only. The memo records `FREEZE_ID` (the runs) and the
  final scoped identity, with the file list between them.
- **C5 as built (2026-11-23).** `tests/oracles/`, `tests/test_oracles.py` and the closure digest are deleted.
  - The spearman batch, `compute_threat_pc_batch` and `compute_shape_graphs` keep an exact guard against the
    per-frame public call in the same test run.
  - The linker's game split and its no-`game_id` rule compare live calls. Its warning and error texts are literal
    pins.
  - The old-against-new comparisons of the rewritten cores are deleted. The structural guards stay: the Kalman
    covariance reuse, the shape-graph angle cache (no angle computed twice in one call), and the candidate-counter
    companion, now on a per-action rescan written in the test.
  - The role gate's baseline arm is `scripts/_shape_graph_v4139_baseline.py`, a standalone copy of the BASE_SHA
    source.
  - Owner ruling 2026-11-23 (spec section 7): value snapshots at `rtol=1e-12` (float32-stored columns: one float32
    ulp; float32 velocity and acceleration: 5e-4 m/s and 1e-2 m/s²; owner rulings of the same day) in
    `tests/regressions/tracking_batch/`
    for the four rewritten cores whose tests pinned no values (the preprocess, the linkers, the shape-graph core,
    `detect_line_breaking`). The DFL parser keeps its lakehouse goldens.

**Amendment r5.17 (2026-11-18): owner rulings on reviewer C's section 8** (spec r7.22, section 7, C1-C6). These
rulings supersede the plan text that they name. Where this block and a task disagree, this block wins.

- **Withdrawn (C2 and the owner's answer on the W0 driver):** Task 2 (the W0 driver) and Task 3 step 5 (the W0
  legs), Tasks 3B (R20), 3C (R23), and 3D (R25-R27), and every W0 campaign, HF launch, `--stage-tree` run, and
  `compare` call in Task 15. Their code and tests are removed from the tree; the ledger keeps their results. The R20
  and R26 defects go to the HF tooling's owner session as findings.
- **Done (C3):** the per-request team match in `_spearman_batch_core` is gone, by TDD
  (`test_ids_match_once_per_part_and_team_unless_mixed`, the id_compat helper tests, W2.4 mixed fixture green).
- **Task 15, new order** (each step runs only after main is quiet and the owner says go, C6):
  1. Merge `main`; run the gate once on the merged tree.
  2. Steps 1b, 3 (freeze), 7 (W9.6 confirming run), 7b and 7c (the W5.7 role gate and its memo) as written. The role
     gate runs before the oracles end.
  3. **C1 record** (replaces Steps 4-6 and the W0 part of Step 8): on the DGX, the merged tree against `main`, one
     input, the lakehouse's own calls: (a) the per-frame pitch-control loop against the batch call with one request
     per frame, (b) the per-team-frame shape-graph loop against `compute_shape_graphs`. Wall clock per frame, and a
     bit compare of the outputs. The memo goes to `docs/research/tracking-compute-perf/`.
  4. **C5:** delete `tests/oracles/`, `FROZEN_ORACLE_CLOSURE_DIGEST` and `tests/test_oracles.py`; add the five
     `_numba_dispatch` examples and remove their debt entries; each batch entry point keeps an exact guard against
     the per-frame public call in the same test run. Step 1 (oracle minimality) is then void.
- **Owner answers (2026-11-18, author's session):**
  - The C1 run also records the `tracemalloc` peak of the batch call on the lakehouse workload (one request per
    frame), which replaces the docstring's measured peak, and the fingerprint cost per match against the 2 % rule
    (Step 7). Step 2 is folded into C1.
  - At C5 the role gate keeps one standalone copy of `shape_graph_v4139` as its baseline arm, in a `scripts/`
    module that only the role gate imports, with no closure digest. Its metrics stop stamping the oracle digest.
  - `scripts/_provenance.git_tree_hash` loses its `source=` modes (used only by the removed W0 driver and HF
    staging). Done.

## Global Constraints

- Branch: one feature branch `feat/tracking-batch-dfl-events`, created from `main` **after PR #299 merges**. Never
  use a worktree.
- One commit for the whole cycle. Before it: `/final-review`, the A/B implementation review, and an explicit owner
  yes (Task 17).
- The version number, the ADR number, and the `PR-Snnn` key are assigned at commit-prep (Task 17 step 3), never
  before. Until then, the docs use the placeholders `X.Y.Z`, `ADR-NNN`, and `PR-Snnn`. ADR-130/131 are claimed
  elsewhere.
- Bit-identity (spec §5.1) uses `scripts/_bitwise.py::assert_bits_equal`, never `np.array_equal` or
  `assert_frame_equal` alone.
- **Approved value changes only:** R1/R6 (action keys), R2 (string id keys and the qualifier `==`), R5/R7/R9
  (multi-game linking and pointer binding), and R14 (the team-relative position labels, by the exact spec W5.6
  relations). Everything else is bit-identical.
- §5.2 rule (owner ruling R13): a candidate that is not bit-identical by construction ships only if it passes on
  every platform we run: AMD x86 AVX2 (HF cpu-upgrade) and arm64 (the Mac Mini and the DGX) (spec R22: the local PC left the rule). No other platform is in the rule.
- §5.3: never loosen a gate. A failed candidate is dropped, and its measured gain is recorded.
- `max_frames_per_chunk` default = `2048`.
- New warning categories: `DflEventTimeWarning`, `DflGameSectionWarning`. Neither subclasses another. All calls use
  `warnings.warn(..., stacklevel=2)`.
- Pointer columns appended in this order after the existing five: `action_row` (int64), `action_fingerprint`
  (uint64). No pointer `game_id` column.
- The fingerprint record is `"{game}|{period}|{action}|{time}"`, hashed with BLAKE2b with an 8-byte digest. The id
  fields come from `id_compat.canonical_id_series`, NA becomes `"<NA>"`, and time is `float(np.float64(t)).hex()`.
  A NaN time also becomes `"<NA>"` (spec W9.2), never `"nan"`.
- **Scoped code identity (spec W0.4, R11):** `CODE_IDENTITY_PATHS = ("silly_kicks", "scripts", "pyproject.toml",
  "uv.lock")`, less `CODE_IDENTITY_EXCLUDE = ("silly_kicks/_version.py",)`. The package identity for the base rule
  (R10) uses `PACKAGE_IDENTITY_PATHS = ("silly_kicks", "pyproject.toml", "uv.lock")`, with the same exclusion.
- **No unnecessary duplication (owner bar, 2026-11-08; R12).** One implementation per rule (spec AG2). Extend an
  existing helper before you add a new one. Reuse these, and never write a second copy:
  - `group_rows` and `id_compat`;
  - `tests._perf_structural.call_counter` and `assert_subquadratic_growth`;
  - the shared AST seam `tests/scripts/_script_population.py`;
  - `scripts._provenance.git_tree_hash` and `env_provenance`;
  - the docstring-example extractor and the public-surface discovery in `tests/test_public_api_examples.py`;
  - the shared bitwise helper `scripts/_bitwise.py`.

  When an old private function loses its last package caller, delete it in the same task.
- Every new public symbol has a real `Examples` section (`tests/test_public_api_examples.py`).
- **No placeholder in scope (A2-PLAN-61, TB-PLAN-18).** No file under `CODE_IDENTITY_PATHS` (less
  `CODE_IDENTITY_EXCLUDE`) holds `X.Y.Z`, `ADR-NNN`, `PR-Snnn`, or the new ADR, `PR-S`, or version number. New code
  cites the spec path (`docs/superpowers/specs/2026-11-08-tracking-batch-and-dfl-events-design.md` §n) and existing
  ADRs only. The placeholders live only in the docs, outside the scope.
- Never pickle (ruff `S`). Use `.npz` or parquet.
- No raw `==`/`!=` across id sources. Use `silly_kicks.id_compat`.
- **Local gate (read `ci.yml` and show the real exit codes):**
  - `python -m ruff check silly_kicks/ tests/ scripts/`
  - `python -m ruff format --check silly_kicks/ tests/ scripts/`
  - `pyright`
  - `python -m pytest tests/ -m "not e2e" -v --tb=short`
  - `pytest --doctest-modules silly_kicks/ --ignore-glob="*/_[!_]*.py" --tb=short` (`ci.yml:138`)
  - `pytest tests/ -m "not e2e" --benchmark-only --tb=short` (`ci.yml:277`)
- Data comes through pining only. MEDIA-PC is permanently gone (2026-11-08). The W0 legs run on the DGX, the Mac Mini, and HF `cpu-upgrade` (staged, Task 3C), never on this PC (spec R22). All local temporary files go under `D:\Temp\tbdfl`. The
  R18 `time` stage runs one unit at a time, and the other W0 stages run within the R19 limits (spec §7). State the expected peak memory to the owner before each heavy run (spec W0.5). The arm64 leg runs
  on the owner's Mac Mini (spec W0.5, R4).

## Review Focus

These are input classes that the spec implies but no workstream's main tests exercise. Each one gets a pinned test
in the task named after it.

1. **Empty inputs to every new public function.** That is empty `frames`, zero requests, an empty `actions`, and
   empty DFL Series. Each one returns an empty, correctly typed output and never raises (Tasks 4, 8, 11, 14).
2. **NA `game_id` on actions or frames.** NA games link only to NA-game frames (canonical `"<NA>"` key), never to a
   real game, and the fingerprint uses `"<NA>"` (Task 4).
3. **`team_id` stored as `category`** (ADR-106) through the W2 index and the W5 batch. Results are bit-identical to
   the object-dtype input (Tasks 7, 11).
4. **Extra time and the shootout (periods 3–5).** DFL `penaltyShootout` → period 5. The linker leaves period-5
   actions with no frames unlinked as `<NA>`, without a cross-game link (Tasks 4, 14).
5. **A team with fewer than 3 outfield players, or a ball-only frame,** in `compute_shape_graphs`.
   `shape_graph_source="too_few_players"`, the summary row is still emitted, and the labels are `<NA>` (Task 11).
6. **A team-frame with mixed or partly null direction values** on its selected outfield rows. It raises
   `ValueError` on the label path only. `add_shape_graph` and `with_positions=False` do not raise (Task 11).
7. **A team that attacks −x with 9 outfield players on the lateral fallback** (a red card). The labels are the
   pinned spec W5.6 binning, not the oracle's (Task 10A).

---

### Task 0: Branch, re-check against merged #299, CPU classes, green baseline

**Files:** none changed. The spec gets an amendment note only if #299 changed something.

- [ ] **Step 1: Confirm #299 has merged.** Run `gh pr view 299 --json state,mergeCommit`. Expected: `"state":"MERGED"`.
  If it is not merged, stop and tell the owner. Do not branch.
- [ ] **Step 2: Branch.** `git checkout main && git pull --ff-only origin main && git checkout -b feat/tracking-batch-dfl-events`.
- [ ] **Step 3: Re-check the spec's #299 assumptions.**
  - Check that `silly_kicks/_numba_dispatch.py` exports `FAMILIES`, `NumbaUnavailableError`, `use_numba`,
    `require_numba`, `import_kernels`, and `numba_version`.
  - Check that `scripts/_provenance.py` has `git_tree_hash`, `env_provenance`, `code_identity`, and
    `require_clean_tree(prov, *, allow_dirty)`.

  If a name or semantics differs from spec §4.0/§4.1, write the difference into the spec under a dated "post-#299
  amendment" line, and tell the owner before going on.
- [ ] **Step 4: CPU-class census (spec §5.2; A1-PLAN-56).** Record the class of each machine that the owner uses for
  these jobs. The known classes (2026-11-08) are:
  - this PC: Intel Core Ultra 9 285H, x86 AVX2 (numpy `__cpu_features__`);
  - DGX Spark (NVIDIA GB10) and the Mac Mini: arm64. Confirm the Mac Mini with `uname -m`;
  - HF `cpu-upgrade`: AMD EPYC 7R13, x86 AVX2. Probed 2026-11-08 with the owner's yes: five jobs
    (`6af0dd38…`, `6af0dd3a…`, `6af0dd3b…`, `6af0dd3d…`, `6af0dd3e…`), all the same CPU.

  These four machines were the platforms of the §5.2 gate (R13). Since spec R22 the platforms are HF, the Mac Mini, and the DGX. Record the result in the session notes. If the
  Mac Mini is not arm64, or a machine is unreachable, tell the owner before you go on.
- [ ] **Step 5: Green baseline.** Run the full local gate (Global Constraints). Expected: exit code 0 for each
  command. Record the commit SHA as `BASE_SHA` in the session notes.

### Task 1: Shared test infrastructure — bitwise helper, AST seam and call graph, oracles, fixtures

**Files:**
- Create: `scripts/_bitwise.py`. The driver's compare (Task 2) and the tests share it, so it lives in `scripts/`.
- Modify: `tests/scripts/_script_population.py`. Add `iter_package_modules()` and `call_graph()`.
- Modify: `tests/_scale_guarded.py`. `group_rows_callers` (`:95-108`) uses `iter_scripts` and
  `iter_package_modules` instead of its own `ast.parse` walk (R12).
- Modify: `tests/coordination/kernels/test_cluster.py:89`, `tests/coordination/test_compute.py:295`, and
  `tests/coordination/test_pair_surrogates_batched.py:181`. Delete the private helpers and use `scripts._bitwise`
  (R12).
- Create: `tests/oracles/__init__.py`, `tests/oracles/README.md`, and the oracle modules listed below.
- Create: `tests/_prod_fixtures.py`
- Test: `tests/scripts/test_bitwise.py`, `tests/scripts/test_script_population_call_graph.py`,
  `tests/test_oracles.py`, `tests/test_prod_fixtures.py`

**Interfaces:**
- Produces, in `scripts/_bitwise.py`:
  - `bits_equal(a, b) -> bool` and `assert_bits_equal(a, b) -> None`. They accept ndarray, Series, DataFrame, scalar
    float, `ShapeGraph`, or a list/tuple of those. Any two NaNs are equal. `-0.0` and `0.0` are different. Dtype and
    index must be equal. `assert_bits_equal` raises `AssertionError` that names the first differing element. These
    are the semantics of the three coordination helpers, so their tests keep their strictness.
- Produces, in `tests/scripts/_script_population.py`:
  - `iter_package_modules() -> dict[str, ast.AST]`, keyed by dotted module name, for every file under `silly_kicks/`;
  - `call_graph(trees: Mapping[str, ast.AST]) -> CallGraph`, with `callers(qualname)`, `callees(qualname)`, and
    `referenced_globals(qualname)`. The edges are:
    - direct calls;
    - "a factory returns a closure that calls X";
    - **dynamic kernel imports** (A2-PLAN-48): a call `import_kernels("<module>")` or
      `importlib.import_module("<module>")` with a string literal is an edge to every function of that module.
      After #299, `_spearman.py` reaches `_numba_kernels` only this way (`_numba_dispatch.py:64-71`).

    Qualified names are `module:function`, with nested closures as `module:factory.<locals>.inner`.
- Produces, in `tests/_prod_fixtures.py`:
  - `make_production_frames(seed: int = 20261108, n_frames: int = 2000, n_periods: int = 2, *, mixed_ids: bool = False) -> pd.DataFrame`,
    with the `TRACKING_FRAMES_COLUMNS` schema, float32 coords, and `team_id` category. It has 22 players plus the
    ball, gaps, NaN detections, and GK flags. With `mixed_ids=True`, half the frames carry int ids and half str ids.
  - `make_two_game_inputs(seed: int = 20261108) -> tuple[pd.DataFrame, pd.DataFrame]` returning `(actions, frames)`.
    The two games share `period_id`, `frame_id`, and `action_id`, and their clocks are offset by 2 s, as in spec §4.9.
    Both games use the same two `team_id`s, with the same home team. So the scalar `home_team_id` that
    `_sb_battery.call_aggregator` takes is valid for the whole call and for each part (A3-PLAN-68).
  - `make_gamestate_slots(actions: pd.DataFrame, nb_prev_actions: int = 3) -> list[pd.DataFrame]`. It calls the
    public `silly_kicks.vaep.feature_framework.gamestates`.
  - `make_single_game_inputs(seed: int = 20261108) -> tuple[pd.DataFrame, pd.DataFrame]`: game 7 of
    `make_two_game_inputs`.
  - `spec_4_9_reproduction_inputs() -> tuple[pd.DataFrame, pd.DataFrame]`. It is the exact spec §4.9 probe: games 7
    and 8, 100 frames each, period 1, game 8 clock +2 s, and action times `[0.40, 1.20, 2.40, 3.20]`.

**Why code oracles and not committed goldens.** A committed golden cannot pin these outputs exactly, for two
reasons:
- Float results from SIMD-dispatched numpy functions (`exp`, `arccos`, `arctan2`) and from small BLAS matmuls change
  with the runner CPU. PR #292 measured this on the pressure snapshot hashes.
- Dtypes differ across the declared pandas-major span (ADR-057). For this reason `test_parse_port_parity.py:60`
  already compares order-agnostic, with `atol=1e-9` and `check_dtype=False`, which is too weak for NG4.

A same-run comparison against frozen code is the only exact check.

**Oracle inventory.** Copy only the functions whose code this cycle changes or deletes. Import every other helper
from the live package. Each oracle module starts with a table of its copies and the task that changes each one:

| Oracle module | Copied functions | Changed by |
|---|---|---|
| `spearman_v4139.py` | `compute_spearman`, `compute_spearman_batch`, `_batch_from_groups`, `_spearman_combine`, `_extract_frame_players` | Task 7 (W2.6 rewire, R2 key, deletions) |
| `cover_shadows_v4139.py` | `compute_threat_pc_batch` (`_cover_shadows.py:908`) | Task 7 (moves to the index adapter) |
| `shape_graph_v4139.py` | the `_shape_graph.py` functions that W5.1 and W5.6 edit (the merge loop, the angle helpers, `compute_shape_graph`, and the whole `infer_positions` closure: `_assign_levels_*`, `_decompose_middle`, `_equal_frequency_assign`, `_face_centroids`, `_bridging_edge_midpoints`, `_is_tree`, the level constants) | Tasks 10, 10A, 11A |
| `shape_graph_features_v4139.py` | `add_shape_graph`, `shape_graph_xfns` | Tasks 5 and 11 |
| `line_breaking_v4139.py` | `detect_line_breaking` and the helpers that W6 edits | Task 12 |
| `preprocess_v4139.py` | `smooth_frames`, `derive_velocities`, `kalman_ca`, and the helpers that W7 edits | Task 13 |
| `sportec_parse_v4139.py` | `parse_dfl_events`, `_build_event_row`, `_scan_kickoff_times`, `_derive_period_from_kickoffs`, `_parse_events_xml`, `_resolve_idsse_team_from_qualifiers` | Task 14 |
| `linker_v4139.py` | `link_actions_to_frames`, `link_actions_to_frames_elastic`, `_count_candidates_within_tolerance`, `_diagnose_time_base` | Task 4 |

Copies that import other copies import them from the oracle package, not from the live package. Example:
`cover_shadows_v4139` calls `spearman_v4139.compute_spearman_batch`. If a later task edits a function that is not
in this table, add the copy then. The closure guard below makes that rule checkable.

- [ ] **Step 1: Write the failing tests.**

```python
def test_bits_equal_rejects_signed_zero():
    assert not bits_equal(np.array([0.0]), np.array([-0.0]))
    with pytest.raises(AssertionError):
        assert_bits_equal(np.array([0.0]), np.array([-0.0]))

def test_bits_equal_any_nan_payload_equal():
    a = np.array([np.nan]); b = np.frombuffer(np.uint64(0x7FF8000000000001).tobytes(), dtype=np.float64)
    assert_bits_equal(a, b)

def test_bits_equal_checks_dtype_and_index():
    with pytest.raises(AssertionError):
        assert_bits_equal(pd.Series([1.0], index=[0]), pd.Series([1.0], index=[1]))

def test_call_graph_sees_factory_closures():
    g = call_graph({"m": ast.parse("def f():\n    def inner():\n        g()\n    return inner\ndef g(): pass\n")})
    assert "m:g" in g.callees("m:f.<locals>.inner") and "m:f" in g.callers("m:g")

def test_group_rows_callers_unchanged():
    assert group_rows_callers() == FROZEN_GROUP_ROWS_CALLERS   # literal captured at BASE_SHA before the migration

def test_call_graph_follows_dynamic_kernel_import():
    trees = {"k": ast.parse("def kern(): pass\n"),
             "m": ast.parse("def f():\n    ks = import_kernels('k')\n    return ks.kern()\n")}
    assert "k:kern" in call_graph(trees).callees("m:f")

def test_no_package_module_imports_oracles():
    assert [n for n, t in iter_package_modules().items() if "tests.oracles" in ast.unparse(t)] == []

def test_oracle_closure_unchanged():
    # sha256 over the source of every live package function, numba kernel, and module-level name that is
    # reachable (call_graph closure + referenced_globals) from each oracle's live imports; frozen at BASE_SHA.
    assert oracle_closure_digest() == FROZEN_ORACLE_CLOSURE_DIGEST

def test_oracle_closure_guard_catches_transitive_edit(tmp_path):
    # copy the package to tmp_path, edit the body of pitch_control/_numba_kernels.py:influence_numba (not imported
    # directly by any oracle; reached only through the dynamic kernel-import edge), and assert the digest changes
    assert oracle_closure_digest(root=planted_copy(tmp_path)) != FROZEN_ORACLE_CLOSURE_DIGEST

def test_two_game_inputs_share_keys():
    actions, frames = make_two_game_inputs()
    g = actions.groupby("game_id")["action_id"].apply(list)
    assert g.iloc[0] == g.iloc[1]
    assert set(frames.loc[frames.game_id == frames.game_id.iloc[0], "frame_id"]) == set(
        frames.loc[frames.game_id != frames.game_id.iloc[0], "frame_id"])
```

- [ ] **Step 2: Run them.** `python -m pytest tests/scripts/test_bitwise.py tests/scripts/test_script_population_call_graph.py tests/test_oracles.py tests/test_prod_fixtures.py -v`. Expected: FAIL with an ImportError.
- [ ] **Step 3: Capture `FROZEN_GROUP_ROWS_CALLERS`** from the unmodified `group_rows_callers()`, then migrate it
  onto the shared seam.
- [ ] **Step 4: Implement.**
  - Copy the oracle sources byte for byte from `git show BASE_SHA:<path>`. Change only the imports: relative imports
    become absolute, and imports of other copied functions point at the oracle package. Put a header comment with
    `BASE_SHA` and "frozen test oracle; never import from silly_kicks/".
  - `oracle_closure_digest(root=REPO)` lives in `tests/oracles/__init__.py`. Compute `FROZEN_ORACLE_CLOSURE_DIGEST`
    at `BASE_SHA` and commit the literal.
  - Move the three coordination helpers onto `scripts._bitwise`. They cast both sides to float64 and check only
    the shape (`test_cluster.py:90-92`). So each migrated call site keeps that cast:
    `assert_bits_equal(np.asarray(got, np.float64), np.asarray(want, np.float64))`, and
    `bits_equal(np.float64(have), np.float64(want))`. Their behaviour does not change (R12; TB-PLAN-17, A2-PLAN-65).
- [ ] **Step 5: Run them again,** plus `tests/coordination/`, `tests/test_scale_guards.py`, and
  `tests/test_scale_guard_registry.py`. Expected: PASS.

### Task 2: W0 measurement driver and scoped code identity (spec §4.0, R10, R11)

**Files:**
- Modify: `scripts/_provenance.py`. Add `CODE_IDENTITY_PATHS`, `CODE_IDENTITY_EXCLUDE`, `PACKAGE_IDENTITY_PATHS`,
  and the `paths`/`source` keywords on the existing `git_tree_hash`. Do not write a second hashing walker.
- Create: `scripts/measure_tracking_compute.py`
- The HF campaign files are Task 3C (spec R23). The uncommitted `tracking-compute-amd-gate.toml` is deleted there.
- Modify: `tests/scripts/test_provenance_wiring.py`, adding the driver to `ARTIFACT_DRIVERS` (`:28`).
- Test: `tests/scripts/test_git_tree_hash_scoped.py`, `tests/scripts/test_measure_tracking_compute.py`

**Interfaces:**
- Produces:
  - `git_tree_hash(*, paths: Sequence[str] | None = None, source: str = "worktree") -> str`.
    - With `paths=None` and `source="worktree"`, the behaviour stays exactly as it is today (whole tree, `"unknown"`
      on a git failure).
    - With `paths`, the throwaway index starts empty, and the content under `paths` (less `CODE_IDENTITY_EXCLUDE`)
      comes from `source`:
      - `"worktree"`: `git add -A -- <paths>` (tracked and untracked, not ignored);
      - `"index"`: the entries of the real index (`git ls-files -s -- <paths>`), that is, what the commit will hold;
      - any other value is a commit-ish: `git ls-tree -r <commit> -- <paths>`.
    - The result is the `git write-tree` hash of that index. With `paths`, a git failure **raises**
      `ProvenanceError`. It never returns `"unknown"`, so two failures can never compare equal (fail-closed,
      A1-PLAN-49).
  - **Leg identity: one rule per context** (R10; A8-PLAN-81, A9-PLAN-83–85, TB-PLAN-29/31). Every leg imports
    `silly_kicks` first, takes `root = Path(silly_kicks.__file__).resolve().parents[1]`, checks its identity, and
    stamps `package_hash`:

    | Leg | Check | Stamp `package_hash` |
    |---|---|---|
    | base, local (Task 3; the checkout at `BASE_SHA` content) | `verify_import_tree(root, BASE_SHA)` | `git_tree_hash(paths=PACKAGE_IDENTITY_PATHS, source=BASE_SHA)` |
    | base, AMD (`dest`): **dropped by spec R23**. The HF base leg is a staged run and uses the "base, local" rule | — | — |
    | final, local (Task 15, uncommitted frozen tree) and HF (the staged frozen tree, spec R23) | `root` is the checkout itself (`root == REPO`) | `git_tree_hash(paths=PACKAGE_IDENTITY_PATHS)`, the working-tree hash, which equals `final_package_hash` in both cases |

    The final rule never uses `HEAD`, because locally `HEAD` is still `BASE_SHA` (A9-PLAN-83). The working-tree hash
    describes the frozen content, uncommitted or committed. `git add -A` applies the repo's clean and EOL rules,
    and `.gitignore` drops `__pycache__`.
  - `verify_import_tree(root: Path, commit: str, *, repo: Path = REPO) -> None` proves that the package files under
    `root` hold exactly `commit`'s content, under the repo's EOL and attribute rules:
    - For every entry of `git -C <repo> ls-tree -r <commit> -- <PACKAGE_IDENTITY_PATHS>`, the blob id of
      `root/<path>`, computed by `git -C <repo> hash-object --path=<path> <root>/<path>`, equals the `ls-tree` blob
      id. `--path` applies the same clean and EOL conversion as `git add`, so a CRLF working copy (`core.autocrlf=true`
      on this PC; `git archive` also writes CRLF there) matches its LF blob (A9-PLAN-84, TB-PLAN-31(1)).
    - Under `root/silly_kicks`, no file exists outside that manifest, except `__pycache__/` folders, `*.pyc`, and
      the `CODE_IDENTITY_EXCLUDE` paths (`silly_kicks/_version.py`, A9-PLAN-85). So an extra module is refused, and
      the bytecode that an import writes is ignored.
    - The excluded paths are not compared.
    - Otherwise it raises `SystemExit` naming the first differing path.
  - The driver CLI:
    - `python -m scripts.measure_tracking_compute run --phase {base,final} --base-sha <sha> --out <dir> [--members]
      [--matches <stem,...>] [--profile-matches <stem,...>] [--half-matches <stem,...>] [--sample]
      [--parity-tests] [--budget-s 7200]`. It runs the `prereq`, `time`, `memory`,
      `profile`, `kernel_diff`, `member`, and (final phase with `--parity-tests`) `parity_tests` stages with their worker counts (spec
      R18, R19), then the reduce stage. `--matches` selects matches by stem (for example `skillcorner__2015213`), the same format as the campaign
      args (TB-PLAN-57). `--sample` cuts every unit's inputs to the
      spec R21 slice (a pilot only). `--budget-s` is the R18 rule 6 watchdog.
    - The engine parameter `matches` (a comma-separated list of stems) replaces `matches_json`, so a campaign
      file can pass it as a job argument (Task 3C). The parameters `repo` and `gate_file` are removed.
    - `python -m scripts.measure_tracking_compute probe --unit <key> ...`: the `time` unit's child process
      (internal; it has a parser).
    - `python -m scripts.measure_tracking_compute compare <base_dir> <final_dir> --base-sha <sha>`
  - The `run` outputs, per `<dir>/<provider>/<match>/`:
    - `timings.json`: the wall-clock time and the cProfile cumulative buckets per step, as in spec W0.2, plus
      `compile_s` per step (spec R18 rule 4);
    - `memory.json`: per step and backend, the `time` unit's peak RSS above its baseline, with its `isolated`
      flag and its RSS method (spec R18 rule 3); and for `compute_pitch_control_batch`, the `memory` unit's
      `tracemalloc` peak, which alone sets and checks the W3 bound (rule 2);
    - `kernel_diff.json`: per kernel (`tti_numba`, `influence_numba`, `influence_rows_numba`, plus any prototype),
      the max abs diff between the numba and numpy backends on this match (W4);
    - `parity/pitch_control_{numba,numpy}.npz`: every surface field of every request;
    - `parity/shape_graphs.npz`: edges, faces, stabilities, points, and position labels. Each team-frame is labelled
      with the direction from its frame label (`ltr` → `+1.0`, `rtl` → `-1.0`). Unresolved team-frames are skipped
      and counted. The final phase also stores two flags per team-frame: `pitch_thirds` (`_pitch_thirds_level` ran
      for the vertical axis) and `lateral_fallback` (`_equal_frequency_assign` ran with the lateral ladder), plus
      `n`, plus `lateral_middle_tie`: the index pair when the lateral `_decompose_middle` call got exactly two middle
      players with equal y (TB-PLAN-36), from the Task 10A `_decompose_middle` wrapper. The flag wiring lands in Task 10A, when `_pitch_thirds_level` exists, and only the `--phase final` path
      reads it (the base-phase API rule).
    - `parity/line_breaking.parquet`, `parity/preprocess_<method>.parquet`, and `parity/pointers.parquet`;
    - with `--members`, `parity/members/<module:function>.parquet` or `.npz`: every W9.7 population member on this
      match, called through the Task 3 adapter;
    - `census.json`: the W8/W9 census counts (spec W0.2), the population size, and the fingerprint cost per match.
  - The `compare` subcommand runs `assert_bits_equal` on every parity file. **A file that exists in one directory
    and not in the other is a failure, never a skip** (TB-PLAN-02). The pointer compare covers the pre-existing five
    columns. The position labels are compared under the spec W5.6 relations (R14):
    - +x team-frames: `vertical` equal, and `horizontal` equal to the base's with L↔R and LC↔RC;
    - −x team-frames: equal, except `vertical` where the final flags `pitch_thirds`, and `horizontal` where it flags
      `lateral_fallback` with n ∉ {5, 10};
    - **the −x lateral exact-tie class** (TB-PLAN-33, tightened by TB-PLAN-36): on a −x team-frame without another
      flag, a `horizontal` difference is explained only if the final flags `lateral_middle_tie`, the differing
      players are exactly that pair, and the base labels of the pair are the final labels swapped (LC↔RC). The cause is the index tie-break: `_decompose_middle` sorts two middle players as `(value, idx)`
      (`:589-593`), so on a tie the lower index gets `label_below`. That is "LC" in the oracle and "RC" in the
      attack frame.

    The compare reports the count per class. Any other label difference is "unexplained": the compare fails, and
    lists each one with its key. With each unexplained difference it records the **margin** (A11-PLAN-94): the
    smallest absolute distance, on that axis, from the player's coordinate to any face-centroid coordinate and to any
    other selected player's coordinate of the team-frame. The compare computes it from the stored points and faces.
    A margin near 0 marks a threshold tie. The compare stays fail-closed, and the owner sees the margin. The graph arrays stay under `assert_bits_equal`.
    It exits non-zero on the first mismatch and names the file and the element. It also **refuses** when
    the two legs' `cpu_model` or `cpu_features` stamps differ, because a cross-CPU compare proves nothing
    (TB-PLAN-27).
  - **Dropped by spec R23:** `archive_argv`, `extract_base_tree`, `materialize_base_tree`, `ensure_base_tree`,
    the `gate` phase, `gate_unit`, the `leg` subcommand, `GATE_FILE`, and `_read_gate_file`. Remove them and their
    tests from the uncommitted driver. `verify_import_tree` stays.
  - **Expected values for `compare`** (A8-PLAN-82, TB-PLAN-29b; amended by spec R23). `compare` takes
    `--base-sha`. It computes `base_package_hash = git_tree_hash(paths=PACKAGE_IDENTITY_PATHS, source=BASE_SHA)`
    and `final_package_hash = git_tree_hash(paths=PACKAGE_IDENTITY_PATHS)`, the frozen working tree. It refuses
    unless the base stamp's `package_hash` equals the first and the final stamp's equals the second. A refusal is
    not a pass. A final leg whose `parity_tests` unit failed is a failure, never a skip.

- [ ] **Step 1: Write the failing tests.**
  - `test_scoped_hash_ignores_docs`: in a tmp git repo, editing `docs/x.md` leaves `git_tree_hash(paths=...)`
    unchanged.
  - `test_scoped_hash_sees_untracked_code`: adding an untracked `scripts/new.py` changes the worktree hash.
  - `test_scoped_hash_excludes_version_file`: editing `silly_kicks/_version.py` leaves the hash unchanged.
  - `test_index_source_ignores_unstaged_edits`: an unstaged edit changes the worktree hash, not the index hash.
  - `test_commit_source_equals_worktree_on_clean_tree`.
  - `test_scoped_hash_git_failure_raises`: with a bad `GIT_DIR`, it raises `ProvenanceError`, so two failures
    cannot compare equal.
  - `test_tree_hash_unscoped_unchanged`: in the tmp repo, `git_tree_hash()` equals the hash from the `BASE_SHA`
    algorithm, run in the test as `git read-tree HEAD`, then `git add -A`, then `git write-tree` (A1-PLAN-59).
  - `test_verify_import_tree_refuses_changed_package`: an edit under `silly_kicks/` makes it raise `SystemExit`.
  - `test_verify_import_tree_refuses_extra_module`: an untracked `silly_kicks/new.py` makes it raise `SystemExit`.
  - `test_verify_import_tree_allows_new_scripts_tests_and_bytecode`: an untracked `scripts/x.py`, a `tests/y.py`, and
    `silly_kicks/__pycache__/z.cpython-312.pyc` pass.
  - `test_verify_import_tree_crlf_copy_passes` (A9-PLAN-84, TB-PLAN-31): in a tmp repo with `core.autocrlf=true`, a
    CRLF working copy of a package file passes, and the same copy with one changed line fails.
  - `test_verify_import_tree_ignores_version_file` (A9-PLAN-85): a differing `silly_kicks/_version.py` passes.
  - `test_final_stamp_uncommitted_equals_committed` (A9-PLAN-83): in a tmp repo, edit a package file. The final
    stamp of the uncommitted tree equals the stamp after committing the same edit, and both equal the value written
    as `final_package_hash`.
  - `test_compare_fails_on_missing_member` and `test_compare_fails_on_one_bit`.
  - `test_driver_registered`: the driver is in `ARTIFACT_DRIVERS`.
  - `test_compare_refuses_wrong_package_hash`: a base leg whose `package_hash` is the final hash is refused.
  - `test_compare_refuses_cpu_mismatch`.
  - `test_compare_label_relations`: on synthetic base and final folders, the +x swap passes, a flagged −x
    difference passes and is counted, and a −x middle-pair tie difference (the flagged pair, swapped) passes in its own class, and a tie
    difference on any other player fails. An unflagged
    difference without a tie fails with its key and its margin.
  - R18 tests:
    - `test_unit_keys_cover_every_step_backend_and_member`: the unit keys of one match are exactly one `prereq` per
      backend, the W0.2 steps × both backends for `time` (and for `profile` on the smallest match only, spec R21 rule 3), one `memory` per backend, one
      `kernel_diff`, one key per public non-exempt member, and in a final leg with `--parity-tests` one `parity_tests` key.
    - `test_time_unit_peak_isolated_after_light_prerequisite`: a planted step that allocates 200 MB reports a peak
      between 200 MB and 260 MB with `isolated: true` on every OS. A step that allocates nothing reports less than
      20 MB.
    - `test_time_unit_flags_heavy_prerequisite` (A14-PLAN-96, TB-PLAN-39): a planted prerequisite with a 500 MB
      transient, freed before the step, then the 200 MB step. On Linux the value is 200–260 MB and `isolated`. On
      Windows and macOS it is `isolated: false` with an upper bound of at least 500 MB, and the reduce never writes
      it as a W3 value.
    - `test_time_unit_compiles_outside_window` (A14-PLAN-97): after the warm-up, the timed call adds no signature
      to any numba dispatcher that the step uses, and `compile_s` and the cache stamp are present.
    - `test_memory_unit_sets_w3_value`: on a planted step, the `memory` value equals the `tracemalloc` peak of the
      call, and it is the only value that the W3 check reads.
    - `test_prereq_key_includes_package_hash` (TB-PLAN-44): a final-leg `prereq` output is not found by a base unit.
    - `test_reduce_reassembles_match_layout`: the reduced folder has the same file names as the old one-unit layout,
      and `compare` passes on two equal reduced folders.
    - `test_time_stage_runs_one_worker`, and `test_budget_watchdog_stops_the_leg` (a planted 0.5 s budget).
    - R21 tests:
      - `test_member_slice_block_size` (TB-PLAN-54): with 23 rows per frame, periods of 1,013, 30, 3, and 2
        distinct frame ids (with id gaps) give k = 102, 3, 1, and the whole period. The kept frames are the
        first k and last k distinct ids, with all their rows.
      - `test_member_slice_closed_bounds`: an action at exactly a block's first or last frame time is kept; an
        action 1 ms outside is dropped.
      - `test_member_slice_pointers_stay_inside`: an action near a block edge whose full-match link is outside
        the slice gets a slice pointer whose frame is inside the slice (or no frame). No pointer names a frame
        outside the slice.
      - `test_member_and_sample_inputs_equal`: member mode and `--sample` give a planted member equal
        frames, actions, and pointers (`assert_bits_equal`), and equal to `member_slice` plus the slice link.
      - `test_member_slice_reads_inputs_only`: two calls on equal inputs give equal outputs, and a planted
        change to a package function that the slice does not call leaves the slice unchanged.
      - `test_profile_units_on_one_match_only` and `test_export_refuses_profile_on_a_larger_match`: `profile` units
        exist only for the `--profile-matches` stems, and the export refuses unless those stems are exactly the
        match with the fewest frame rows (TB-PLAN-62, A23-PLAN-119).
      - `test_sample_cuts_every_kind`: with `--sample`, every unit (`prereq` included) receives the sliced inputs.
    - `test_parity_tests_unit_fails_on_red`: a planted failing test file makes the `parity_tests` unit fail with
      pytest's exit code. A passing file makes it pass. The unit runs each file once per backend.
    - `test_parity_tests_unit_fails_on_all_skipped` (A20-PLAN-109): a planted file that skips at module level
      (`pytest.importorskip("no_such_module")`) exits 0, and the unit still fails. So does a file whose only
      test is deselected.
    - `test_compare_refuses_slice_digest_mismatch` (A20-PLAN-113).
    - `test_slice_digest_sees_helper_edit` (TB-PLAN-60): editing a driver helper that `member_slice` calls changes
      the digest; editing a `silly_kicks` function does not.
  - `test_base_phase_imports_only_base_names`: an AST walk (shared seam) over the driver's base-phase import closure.
    Every `silly_kicks` name it imports must exist in the `git show BASE_SHA:<path>` trees (A6-PLAN-79).
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement the driver on `scripts/_driver.py` `for_each`, with resume before load (ADR-052).**
  - The driver is a corpus walker, so it declares `PIPELINE` (`scripts/_stages.py`, ADR-128) and passes
    `tests/scripts/test_compute_contract.py`. Follow the compute-pipelines skill. The units are:
    - **W0 units (spec R18).** One unit is one (provider, match, kind, step, backend), or one (provider, match,
      member). The kinds are `prereq`, `time`, `memory`, `profile`, `kernel_diff`, and `member`, as spec R18 rule 1
      defines them. The unit keys are strings that sort the same way on every machine.
    - The `time` stage declares `max_workers=1` (Task 3C), so it runs one unit at a time on every machine. The
      other stages run within the spec R19 limits: `--workers 2 --mem-backend none` on the Mac Mini, the `rss`
      backend with `--cap-bytes` and `--peak-rss-bytes` on the DGX, and the campaign's `peak_rss_by_stage` cap
      on HF. The pilots set the values (R19, R21).
    - **The R21 slice** (spec R21 rule 2, TB-PLAN-54). `member_slice(frames, actions) -> tuple[pd.DataFrame,
      pd.DataFrame]`: a frame is a distinct `frame_id`; per period, with n sorted distinct ids,
      k = max(1, ceil(0.10 × n)) (20 % of the frames; corrected from 0.05); keep the first k and the last k ids with all their rows, or the whole period
      when 2k ≥ n; keep the period's actions whose time lies in the closed interval of a block's first and last
      frame times. The slice pointers are a `prereq` item (`pointers_slice`): the leg's own linker on the
      sliced actions and frames. A `member` unit takes the sliced frames, actions, and `pointers_slice`. With
      `--sample`, every unit takes the sliced inputs, and its `pointers` item is the same slice link. A `profile` unit exists only for a stem named
      in the engine parameter `profile_matches` (unit enumeration cannot read row counts: resume before load and
      the dataset read gate). The export checks that the named stem is the match with the fewest frame rows
      (`check_profile_match`), so R21 rule 3 holds fail-closed.
    - **The cut stamp** (TB-PLAN-63, A23-PLAN-117-118). `match_cut(ctx, stem, frames, actions)` is the one place
      that applies `half_matches` and then `--sample`; `MatchInputs` calls it, and `SLICE_SEEDS` includes it and
      `first_half`, so `slice_digest` covers the R24 cut and its call site. Each leg also stamps `cut`
      (`leg_cut`: `matches`, `half_matches`, `sample`, `profile_matches`, `members`). `compare` raises
      `CompareRefusalError` (a `CompareError` subclass, exit 2) for every stamp check (package hashes, CPU,
      `slice_digest`, `cut`, a stamp in one run only, no passing `parity_tests`), and plain `CompareError`
      (exit 1) only for a parity difference or a missing parity file. A refusal is never a gate result, so it
      never starts the revert path (spec §5.3); it goes to the owner. Tests:
      `test_compare_refuses_a_half_base_against_a_full_final`, `test_stamp_refusals_are_not_gate_mismatches`,
      `test_leg_cut_records_the_parameters`, `test_slice_digest_covers_the_cut`.
    - **Half of the largest match** (spec R24). `first_half(frames, actions) -> tuple[pd.DataFrame,
      pd.DataFrame]` keeps, per period, the first ceil(n / 2) sorted distinct frame ids (all rows) and the
      actions in the closed time interval of that block; `member_slice` and `first_half` share one block
      helper. The engine parameter `half_matches` (stems) applies it in `MatchInputs` to every unit of those
      matches, before `--sample`. Tests: `test_first_half_keeps_the_first_half_of_each_period`,
      `test_half_matches_cut_every_unit`.
    - **The `parity_tests` unit** (spec R23). It exists only in the final phase and only with `--parity-tests`; EVERY final
      W0 leg passes the flag (TB-PLAN-64), and `compare` refuses a final leg without a passing unit. It runs `python -m pytest -q -p no:cacheprovider
      <files>` in a child process once per backend (`SILLY_KICKS_<FAMILY>_FORCE_NUMPY` 0 and 1), with
      `--basetemp` in the unit's temporary folder. The files are the gated candidates' parity test files, listed
      in the driver as `PARITY_TEST_FILES`. It runs with `--junitxml` and records the exit codes and the counts.
      It fails the unit on a non-zero exit, and also unless each backend has more than zero passed, zero
      skipped, and zero deselected tests (A20-PLAN-109). A module-level skip (`importorskip`, `skipif`) is
      therefore a failure.
    - **The slice stamp** (A20-PLAN-113, TB-PLAN-60). Each leg stamps `slice_digest`: the SHA-256 over the source
      of `member_slice`, the slice-link function, and every driver-side (`scripts/`) function and module-level
      name in their `call_graph` closure (Task 1 seam, run on the driver's module trees). `silly_kicks` callees
      are left out, because the package is what the gate compares. `compare` refuses when the base and final digests differ.
      `sample` is an engine parameter (`Param("sample", bool, False, flag=True)`), like `matches`.
    - A `time` unit runs its step in a fresh child process (`probe`). The child follows spec R18 rules 3 and 4: it
      loads the inputs and the `prereq` output, warms up on the 200-frame slice (`compile_s`), reads the baseline
      and the pre-step mark, resets the mark on Linux, runs the step once, and reports the wall clock, the peak
      above the baseline, `isolated`, the method, `package_file`, and `package_hash`. The parent refuses a
      stamp that differs from its own (A15-PLAN-103). Read the mark with `scripts/_stages.py` `_peak_rss`, and add a
      Windows branch to it (`GetProcessMemoryInfo`, `PeakWorkingSetSize`). Do not write a second reader.
    - A `memory` unit runs `compute_pitch_control_batch` once per backend under `tracemalloc`, after the same
      warm-up (spec R18 rule 2).
    - A reduce stage puts each match's files together in sorted-key order, with the same names and layout as the
      `run` outputs above. `compare` does not change.
    - `run` arms the `--budget-s` watchdog. When it fires, the driver stops its workers (Task 3B) and prints the
      units left.
  - The HF campaign files and the registry entry are Task 3C.
  - The steps per match are the spec W0.2 list. Each step runs on both backends (`SILLY_KICKS_<FAMILY>_FORCE_NUMPY`
    set to 0 and to 1). The step writes its timing, its memory peak, and its parity output per backend.
  - Every phase checks its identity and stamps `package_hash` by the "Leg identity" table (Interfaces). The
    `final` phase also calls `require_clean_tree(prov, allow_dirty=True)`, and stamps `run_tree_dirty` and
    `git_tree_hash(paths=CODE_IDENTITY_PATHS)`.
  - The machine and the versions come from `env_provenance()`. The new fields are `cpu_features` (the true keys of
    numpy `__cpu_features__`; A1-PLAN-54), `cpu_model` (the `/proc/cpuinfo` model name, or `platform.processor()`),
    `package_file` (`silly_kicks.__file__`), and `package_hash` (see Interfaces).
  - Every unit stamps `cpu_model` and `cpu_features`. The `match` reduce refuses a leg whose units disagree.
    That refusal is "any other failure", and it goes to the owner (TB-PLAN-58). Test:
    `test_reduce_refuses_mixed_cpu`.
  - **The base phase uses the `BASE_SHA` public API only** (A6-PLAN-79, TB-PLAN-27). The driver, and every
    `tests/` and `scripts/` module that it imports in the base phase, imports only `silly_kicks` names that exist at
    `BASE_SHA`. A final-only measurement is guarded by `--phase final`.
  - `--members` imports `tests.invariants._link_population` for the population and the adapter. This repo already
    has scripts that import `tests` (for example `scripts/regenerate_pressure_snapshot_shas.py`).
- [ ] **Step 4: Run the tests.** Expected: PASS. The base run itself happens at the end of Task 3, when the
  population exists.

### Task 3: W9 detection — population, adapter, game-split property, recorder, red at base, W0 base run

**Files:**
- Create: `tests/invariants/_link_population.py`. It derives the population from `iter_package_modules`,
  `iter_scripts`, and `call_graph` (Task 1). It does not parse files itself.
- Create: `tests/invariants/link_population_registry.py`. It holds `LINK_POPULATION: frozenset[str]`,
  `EXEMPT_POOLED: dict[str, str]`, `CANNOT_FAIL_AT_BASE: dict[str, str]`, `COVERED_BY: dict[str, str]`, and
  `ADAPTERS: dict[str, Callable]`. Each is asserted against its derived value.
- Create: `tests/test_link_population.py`, `tests/test_multigame_linking.py`, `tests/test_link_recorder.py`

**Interfaces:**
- Consumes: `make_two_game_inputs`, `make_single_game_inputs`, and `make_gamestate_slots` (Task 1); `call_graph`
  (Task 1).
- Produces, in `_link_population.py`:
  - `derive_population() -> frozenset[str]` (qualified names, `module:function`);
  - `derive_coverage() -> dict[str, str]` (private member → public caller);
  - `accepts_links(member: str) -> bool` (from the signature);
  - `split_by_game(actions, frames) -> tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]`;
  - `resolve_call(member: str) -> Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame | None], object]`. It returns
    a ready call `(actions, frames, links) -> result`, and it never calls the member. The routes are:
    - a registered `add_*` aggregator goes through the existing single copy `scripts._sb_battery.call_aggregator`
      (its `ADAPTER_MAP` already routes `FAMILY_MODEL_REQUIREMENTS` and the call shapes; A2-PLAN-62), with the
      fixture's `home_team_id`;
    - any other member that fits the ADR-078 convention (`frames` positional-or-keyword after `actions`, the
      optional kwargs keyword-only) gets a generic call;
    - the rest go through `ADAPTERS[member]`.
  - `call_member(member, actions, frames, *, links=None) -> object`: `resolve_call(member)(actions, frames, links)`.
  - `nonconforming_members() -> frozenset[str]`: the members that are neither registered `add_*` aggregators nor
    fit the convention.
  - `comparator_for(member: str) -> str` (`per_action`, `per_key`, `linker`, or `pooled`), from the return annotation
    and `EXEMPT_POOLED`.
  - `COMPARATORS: dict[str, GameSplitComparator]`. Each comparator has
    `assert_split_equal(whole, parts, actions) -> None`, which raises only `AssertionError`.
    - `per_action`: rows equal by position.
    - `per_key`: the game-a rows of the whole equal the game-a call, and the same for game b.
    - `linker`: the multiset of the five pre-existing pointer columns per game equals the per-game call. When
      `action_row` is present, it must equal the per-game value offset by game a's length. This comparator works on
      the base output too, so it never raises `KeyError` (A1-PLAN-50).
  - `PUBLIC_NON_EXEMPT: tuple[str, ...]`: the public population members, less `EXEMPT_POOLED`.
  - `BASE_MEMBERS: frozenset[str]` in the registry: the population as derived at the base (this task). The driver's
    `--members` runs exactly these on both legs, so a function that only the cycle adds is never called against the
    base package (TB-PLAN-27). `test_base_members_frozen` asserts it against the Task 3 derivation.

- [ ] **Step 1: Write the population derivation and its tests** (`tests/test_link_population.py`). The rules come
  from spec §4.9 W9.6:
  - **Seed:** the callers of `link_actions_to_frames`, `link_actions_to_frames_elastic`, and
    `_resolve_action_frame_context`; every function with a `links` parameter; every `*_xfns` factory and the
    closures it returns; every function that `lift_to_states` wraps.
  - **Fixpoint:** caller expansion that stops at public `silly_kicks` API functions and at `scripts/` entry points.

  Tests:
  - `test_population_matches_registry`: `derive_population() == LINK_POPULATION`.
  - `test_underivable_empty`.
  - `test_every_private_member_covered`: `derive_coverage()` covers every private member.
  - `test_adapters_exactly_nonconforming`: `set(ADAPTERS) == nonconforming_members()` (A1-PLAN-51).
  - `test_adapter_split` (A2-PLAN-62, A3-PLAN-69): `route(member) -> Literal["battery", "generic", "adapter"]`
    partitions `LINK_POPULATION` (qualified `module:function` names). The rule: `"battery"` when the member's
    public name is in `registered_add_star_aggregators()` (bare names) **and** `getattr(silly_kicks.tracking, name)`
    resolves to the member's own function object (`__module__` + `__qualname__`). A registered `add_*` outside the
    population is ignored. The test asserts that every member has exactly one route, and that the `"adapter"`
    members equal `set(ADAPTERS)`.
  - `test_population_bound`: `len(LINK_POPULATION) <= 150`. If this fails, **stop and ask the owner** (spec stop
    rule).
  - `test_reviewer_a_twenty_present`: the 20 names from spec §4.9 W9.6 are all in the population.
- [ ] **Step 2: Write the game-split property** (`tests/test_multigame_linking.py`), in two variants.

```python
def _whole_and_parts(member, actions, frames, links_mode):
    (a, fa), (b, fb) = split_by_game(actions, frames)
    call = resolve_call(member)                      # scaffolding: outside any try, so its errors are hard failures
    def links_for(acts, frs):
        return None if links_mode == "none" else link_actions_to_frames(acts, frs)[0]   # public linker
    parts = [call(a, fa, links_for(a, fa)), call(b, fb, links_for(b, fb))]   # well-formed-call proof (single game)
    links = links_for(actions, frames)
    try:
        whole = call(actions, frames, links)         # only the two-game member call is inside the try
    except Exception as exc:
        origin = classify_member_raise(traceback.extract_tb(exc.__traceback__))   # pytest.fail if no package frame
        raise AssertionError(f"{member} raised {type(exc).__name__} via {origin.filename}:{origin.lineno}") from exc
    return whole, parts

@pytest.mark.parametrize("member", PUBLIC_NON_EXEMPT)
def test_game_split_links_none(member):
    actions, frames = make_two_game_inputs()
    whole, parts = _whole_and_parts(member, actions, frames, "none")
    COMPARATORS[comparator_for(member)].assert_split_equal(whole, parts, actions)

@pytest.mark.parametrize("member", [m for m in PUBLIC_NON_EXEMPT if accepts_links(m)])
def test_game_split_caller_links(member):     # spec W6 and W9 acceptance: caller-supplied links= on two games
    actions, frames = make_two_game_inputs()
    whole, parts = _whole_and_parts(member, actions, frames, "caller")
    COMPARATORS[comparator_for(member)].assert_split_equal(whole, parts, actions)
```

  Add one parametrised case per frame-aware xfn, using `make_gamestate_slots` input (duplicate ids).

  A "member raised" red counts only when three things hold (A2-PLAN-50, TB-PLAN-05):
  - the same call succeeded on both single-game parts;
  - the exception came from the two-game call itself;
  - its stack enters `silly_kicks/` below the scaffolding frames, even when pandas or numpy raised it (A3-PLAN-67,
    TB-PLAN-19). Examples: a `MergeError`, a merge_asof "keys must be sorted", or a duplicate-label reindex
    (`pandas/core/indexes/base.py:4436`) on the shared `action_id`s.

  `classify_member_raise(frames: list[traceback.FrameSummary]) -> traceback.FrameSummary` lives in
  `_link_population.py`. It returns the deepest frame under `silly_kicks/`. If no frame is under `silly_kicks/`, it
  calls `pytest.fail(...)`. That raises `Failed`, not `AssertionError`, so `xfail(raises=AssertionError)` never
  accepts it.

  "Under `silly_kicks/`" is a resolved-path test, never a substring match (A4-PLAN-71, TB-PLAN-20):
  `pathlib.Path(frame.filename).resolve().is_relative_to(PACKAGE_ROOT)`, where `PACKAGE_ROOT` is
  `pathlib.Path(silly_kicks.__file__).resolve().parent`. Windows paths use `\`, and the base red run happens on this
  Windows PC.

  `test_classify_member_raise` pins three cases with synthetic frame lists built under `PACKAGE_ROOT`:
  - `[test, <PACKAGE_ROOT>/x.py, pandas/…]` → the package frame;
  - the same list with `str(path).replace("/", "\\")` on Windows (native separators) → the package frame;
  - `[test, scripts/_sb_battery.py, pandas/…]` → `Failed`.

  The resolver, the single-game parts, the linker, and the comparators all run outside the `try`. So a wrong
  adapter, a missing model, or a comparator bug is a hard failure, never an expected red.
- [ ] **Step 3: Write the runtime recorder** (`tests/test_link_recorder.py`).
  - It patches both linkers and `_resolve_action_frame_context` with a wrapper that records
    `(f.f_back.f_code.co_filename, f.f_back.f_code.co_firstlineno)`.
  - It maps the record to AST nodes. A decorated node is keyed by its first decorator line, and an undecorated one
    by its `def` line.
  - It runs the fixtures through every public member and every default xfn list, then asserts that the observed set
    is a subset of the population.
  - `test_recorder_catches_planted_xfn` defines an xfn inside the test that calls the linker, and asserts that the
    check fails.
- [ ] **Step 4: Show red at the base, with typed expectations** (A1-PLAN-50, TB-PLAN-05). Run
  `python -m pytest tests/test_link_population.py tests/test_multigame_linking.py tests/test_link_recorder.py -v -rA -p no:cacheprovider`,
  and save the output to the scratchpad. The expected outcome per file:

  | File | Expected at base |
  |---|---|
  | `test_link_population.py` | all PASS |
  | `test_multigame_linking.py`, both variants | every member FAILS with `AssertionError`, except `CANNOT_FAIL_AT_BASE`, each with its reason |
  | `test_link_recorder.py`, including the planted xfn | all PASS |

  - Any other exception type is a scaffolding defect. Fix it before you go on.
  - In the memo, count the comparator mismatches and the "member raised" reds separately, with each raise's type and
    originating frame. Only those two kinds count as reds. List any single-game-part failure as a scaffolding
    defect, and fix it before you go on.
  - Mark the red tests `@pytest.mark.xfail(strict=True, raises=AssertionError, reason="W9 detection; fixed in Task 5")`.
    Remove the mark in Task 5.
  - Run `tests/test_link_recorder.py` on Python 3.10 and on 3.12, the newest CI Python (`ci.yml:56`), with
    `uv run --python 3.10` and `--python 3.12`. Record both results (TB-PLAN-14).
  - The binding fixtures need the Task 4 names, so they are written and shown red in Task 4.
- [ ] **Step 5: Run the W0 base (spec W0.3 as amended by R10; the machines of spec R22).** No package file has
  changed yet, so:
  - **First** finish the R18 and R21 rework of Task 2 (its tests green), Task 3B, and Task 3C.
  - **Projection (spec R18 rule 6, R19, R21).** The full-match pilots on the DGX and the Mac Mini stand (session
    notes). Project them again with the R21 member slice and `profile` on the smallest match. On HF, with the
    owner's yes (billed), run the staged pilot campaign `tracking-compute-w0-pilot.toml` (the R21 sample) and
    project it. If any projection is above 2 hours, **stop and bring the per-kind times to the owner.**
    Otherwise state the expected peak per machine and stage to the owner, then run the legs below with the
    2-hour budget. If a watchdog fires, **stop and bring it to the owner.** A resume needs the owner's yes
    (TB-PLAN-48).
  - **DGX:** with the uncommitted tree shipped byte-identical (`reference_ship_uncommitted_tree_byte_identical`),
    run `python -m scripts.measure_tracking_compute run --phase base --base-sha BASE_SHA --members --out
    <scratch>/base-dgx --budget-s 7200 --matches gradientsports__10517,idsse__DFL-MAT-J03WMX,skillcorner__2015213
    --profile-matches skillcorner__2015213 --half-matches gradientsports__10517` (spec R21, R24). Every local leg,
    base or final, passes `--matches` with these three stems, so its `cut` stamp equals the HF one (A24-PLAN-121).
  - **Mac Mini:** the same with `--out <scratch>/base-arm64`, only when the lakehouse gate is open. First check
    that the Mac clone has the base commit: `git cat-file -e BASE_SHA^{commit}`. If it fails, run
    `git fetch origin main` there and check again.
  - **HF:** with the owner's yes (billed), run `python -m scripts.hf_jobs campaign
    scripts/hf_jobs/campaigns/tracking-compute-w0-base.toml preflight --stage-tree`, then `launch --stage-tree
    --budget-s 7200`, then `retrieve --stage-tree --staging D:\Temp\tbdfl\base-hf`.
  - Each leg uses the "base, local" rule (Task 2). Expected: `verify_import_tree` passes on all three
    machines, and the stamps show `package_hash` equal to the hash built from `BASE_SHA`.
  - Keep the outputs **outside the repo** (locally under `D:\Temp\tbdfl`), and record the paths in the session
    notes.

### Task 3A: R17 — `compute_territorial_defense` with no defensive action (owner ruling R17)

**Files:** Modify `silly_kicks/territorial_defense/_compute.py` (`_assemble_samples`). Test:
`tests/territorial_defense/test_compute.py`.

- [ ] **Step 1: Write the failing test** `test_no_defensive_actions_returns_empty_typed_samples`: the e2e fixture
  without its defensive actions returns an empty table with the columns `["game_id", "player_id",
  *TD_SAMPLE_COLUMNS]`, every column's dtype equal to the dtype of the non-empty call on the same fixture
  (TB-PLAN-41), and a report with `n_frames_in == 0`. Non-vacuity: the fixture had defensive actions to remove. Run it. Expected: FAIL with `KeyError: 'game_id'`.
- [ ] **Step 2: Implement** after the Task 3 base run: build the canonical keys only on non-empty frames, so an empty
  Arm A gives the empty typed table.
- [ ] **Step 3: Run it again,** plus `tests/territorial_defense/`. Expected: PASS.

### Task 3B: R20 — orphaned executor workers (owner ruling R20)

Do this task before the Task 3 step 5 base run. `scripts/` is outside `PACKAGE_IDENTITY_PATHS`, so it does not
change the base package hash.

**Files:** Modify `scripts/hf_jobs/executor.py` (`_dispatch`, `worker_main`). Test:
`tests/scripts/hf_jobs/test_executor.py`.

- [ ] **Step 1: Write the failing tests.**
  - `test_sigterm_controller_ends_workers` (POSIX): a controller with two workers on slow planted units gets
    SIGTERM. Each planted unit also starts a grandchild process (TB-PLAN-46). Five seconds later, no worker and
    no grandchild is alive.
  - `test_killed_controller_ends_workers`: the same with SIGKILL on POSIX, and `TerminateProcess` on Windows.
  - Non-vacuity: before the kill, both workers and both grandchildren are alive, and each worker claimed a unit.
- [ ] **Step 2: Run them.** Expected: FAIL, with the workers or the grandchildren still alive.
- [ ] **Step 3: Implement spec R20:** a process group per worker (`start_new_session=True`), the SIGTERM/SIGHUP
  handler and the group kill in the controller, the `--controller-pid` liveness watchdog in `worker_main` on
  POSIX, and the Job Object (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`) on Windows.
- [ ] **Step 4: Run them, plus `tests/scripts/hf_jobs/` and `tests/scripts/test_compute_contract.py`.** Expected:
  PASS.

### Task 3C: R23 — the staged HF launch (owner rulings R21–R23)

Do this task before the Task 3 step 5 base run. Every file is under `scripts/` or `tests/`, outside
`PACKAGE_IDENTITY_PATHS`, so the base package hash does not change.

**Files:**
- Create: `scripts/hf_jobs/staged.py`; `scripts/hf_jobs/campaigns/tracking-compute-w0-pilot.toml`,
  `…-w0-base.toml`, and `…-w0-final.toml`.
- Delete: the uncommitted `scripts/hf_jobs/campaigns/tracking-compute-amd-gate.toml`.
- Modify: `scripts/hf_jobs/driver.py` (`main`, the controller loop), `scripts/hf_jobs/launch.py` (`JobSpec`,
  `job_command`, `labels_for`), `scripts/hf_jobs/campaign.py` (`CampaignJob`, `load_campaign`, `run_key`,
  `instances`, `cmd_preflight`), `scripts/hf_jobs/preflight.py` (`check_staged`), `scripts/hf_jobs/registry.py` (the
  `measure_tracking_compute` entry), and `scripts/_stages.py` (`Stage.max_workers`).
- Test: `tests/scripts/hf_jobs/test_staged.py`, plus the existing `tests/scripts/hf_jobs/` tests that the changes
  touch.

**Interfaces:**
- Produces, in `scripts/hf_jobs/staged.py`:
  - `StagedTree` (frozen dataclass): `tree: str`, `head: str`, `archive: bytes`, `sha256: str`.
  - `stage_tree(*, repo: Path = REPO_ROOT) -> StagedTree`. `tree` is `git_tree_hash(repo=repo)` (the existing
    whole-tree hash, untracked files included; do not write a second walker). `git_tree_hash` gains a keyword
    `repo: Path | None = None`; `None` keeps today's behaviour (A20-PLAN-111). It raises on `"unknown"`. `archive` is
    `git -c core.autocrlf=false archive --format=tar <tree>`. `head` is `git rev-parse HEAD`.
  - `staged_ident(tree: str) -> str`: `"st" + tree[:10]`. A commit sha never starts with `st` (not hex).
  - `STAGED_PREFIX = "_staged"`, and `upload(st: StagedTree, *, bucket: str, api) -> None`, which puts
    `_staged/<tree>.tar` and `_staged/<tree>.sha256` into the bucket.
- `JobSpec` gains `staged_tree: str | None = None` and `staged_head: str | None = None`. With them, `job_command`
  replaces the fetch-and-checkout lines by: extract `/scratch/_staged/<tree>.tar` into an empty `/work`, `git init`,
  `git fetch -q --depth 1 origin <head>`, `git update-ref HEAD FETCH_HEAD`, and a check that the work-tree hash
  (a throwaway index seeded by `git read-tree HEAD`, then `git -c core.autocrlf=false add -A` and
  `git write-tree`: the same algorithm as the local `git_tree_hash()`, A20-PLAN-111) equals `<tree>`. The other lines do not
  change. `labels_for` adds `staged: "1"`, and the `commit` label is the ident.
- `run_key` keeps a staged ident whole (`{campaign}-st<10 hex>-{job}-{tier}`).
- `CampaignJob.args: tuple[str, ...] = ()` (TOML `args`), appended after a driver's `base_args`. A trainer job with
  `args` is refused at load.
- `preflight.check_staged(tree: str, head: str, *, bucket: str, api, run=subprocess.run) -> Check`: the archive and
  its `.sha256` exist and agree, `tree` equals the current `git_tree_hash()`, and `head` is on a remote branch. With
  `--stage-tree`, `cmd_preflight` runs it in place of `check_ref`, and no green CI run is required.
- The driver CLI: `campaign <file> {preflight,launch,resume,status,cancel,retrieve} --stage-tree [TREE]`.
  Without `TREE`, `preflight` stages and uploads the current tree, and `launch` and `resume` recompute the tree
  and refuse when it differs from the preflight report's. With `TREE` (A20-PLAN-108), the command relaunches a
  retained archive: `check_staged` checks the archive and its SHA-256 but not the current working tree, and
  the in-job check still compares with `TREE`. It refuses when the local campaign file's bytes differ from the
  copy of that file in the archive, because the job args come from the local file (A21-PLAN-116). The `_staged/`
  archives are never deleted before the cycle commit. `adopt`, `verify`, and `publish` with `--stage-tree` exit 2 and write nothing.
- `launch` and `resume` gain `--budget-s <s>`: when `<s>` seconds have passed since the run's first launch, the
  controller cancels every non-terminal job of the run (`cmd_cancel`), prints the units left per stage, and exits
  `4`. The first launch writes its time to `_budget.json` in the run's store, and a resume reads it, so a crash
  does not restart the clock. A resume after a watchdog stop needs the owner's yes and a new `--budget-s` that
  the owner states. Every W0 stage declares `timeout_s` at most 7200, and the controller launches each job with
  `timeout_s = min(stage timeout_s, budget left)`. It launches no job when less than 60 s is left
  (TB-PLAN-56, A21-PLAN-115).
- `preflight --stage-tree` prints the untracked paths that T carries (`git ls-files --others
  --exclude-standard`) and the archive size before it uploads (TB-PLAN-59).
- `Stage.max_workers: int | None = None`: the engine runs the stage with `min(--workers, max_workers)` unit workers,
  the plan JSON carries it, and `_job_spec` caps `workers` the same way.
- The registry entry `measure_tracking_compute` has no gate arguments, and its jobs install the `test` extra (for
  the `parity_tests` unit). The three campaign files each have one job on dataset `tc3`, with the `args`:
  - pilot: `--phase base --base-sha <BASE_SHA> --members --sample --matches skillcorner__2015213
    --profile-matches skillcorner__2015213`;
  - base: `--phase base --base-sha <BASE_SHA> --members --matches gradientsports__10517,idsse__DFL-MAT-J03WMX,
    skillcorner__2015213 --profile-matches skillcorner__2015213 --half-matches gradientsports__10517`;
  - final: the base args with `--phase final` and `--parity-tests` (TB-PLAN-62).

- [ ] **Step 1: Write the failing tests** in `tests/scripts/hf_jobs/test_staged.py`:
  - `test_stage_tree_round_trips` (tmp repo with `core.autocrlf=true`, a CRLF working copy, and an untracked
    file): `stage_tree().tree == git_tree_hash()`. The archive, extracted into an empty folder and hashed there by
    `git init`, `git -c core.autocrlf=false add -A`, and `git write-tree`, gives the same tree.
  - `test_stage_tree_identity_edge_cases` (A20-PLAN-111): a tracked-and-ignored file and a `100755` file
    round-trip to the same tree through the in-job algorithm (seeded index, `autocrlf=false`), or the in-job
    check fails with a non-zero exit. A silent pass with a different tree is never possible.
  - `test_stage_tree_raises_on_unknown`: with a bad `GIT_DIR`, `stage_tree` raises.
  - `test_relaunch_retained_tree`: `--stage-tree <TREE>` with a retained archive passes `check_staged` while
    the working tree differs. A missing or altered archive fails.
  - `test_relaunch_refuses_changed_campaign_file` (A21-PLAN-116): a local campaign file that differs from its
    archived copy makes the relaunch exit non-zero before any launch.
  - `test_staged_job_identity_block_runs` (POSIX): the identity lines of `job_command` for a staged spec, with
    `REPO_URL` patched to a tmp bare repo that holds `head`, run in bash against the extracted archive and exit 0.
    The same lines against an archive with one changed file exit non-zero.
  - `test_staged_job_command_has_no_commit_checkout`: a staged command has no `git checkout` of a commit.
  - `test_staged_ident_labels_and_run_key`: the ident is `st` plus 10 hex characters, every label matches
    `LABEL_PATTERN`, `staged` is `"1"`, and the run key differs from the commit run key of the same campaign.
  - `test_check_staged_refusals`: a missing archive, a SHA-256 mismatch, a tree that differs from the current
    working tree, and a `head` on no remote branch each fail with the cause. The good case passes.
  - `test_staged_refuses_adopt_verify_publish`: each exits 2, and the fake store and hub record no write.
  - `test_campaign_job_args_appended` and `test_campaign_job_args_refused_for_trainer`.
  - `test_budget_clock_survives_resume`: a planted first launch 7,000 s ago in `_budget.json` makes a `resume
    --budget-s 7200` stop after 200 s of planted clock, not 7,200 s.
  - `test_w0_stages_timeout_within_budget`: every W0 stage's `timeout_s` is at most 7200.
  - `test_late_job_timeout_is_budget_left` (A21-PLAN-115): a job launched 6,000 s after the first launch, with
    a 7,200 s budget, gets `timeout_s` 1,200; with 30 s left, no job is launched.
  - `test_budget_cancels_run`: with a fake API that has two running jobs of the run and one of another run, and a
    planted clock past the budget, the controller cancels exactly the two, prints the units left, and exits 4.
  - `test_stage_max_workers`: the local engine with `--workers 4` runs a `max_workers=1` stage one unit at a time
    (planted start and end times do not overlap), and `_job_spec` gives `workers=1`.
- [ ] **Step 2: Run them.** Expected: FAIL (`ModuleNotFoundError: scripts.hf_jobs.staged`, then the missing fields).
- [ ] **Step 3: Implement the Interfaces above.** Then delete the AMD gate file and the dropped driver functions
  (Task 2), and write the three campaign files.
- [ ] **Step 4: Run them, plus `tests/scripts/hf_jobs/`, `tests/scripts/test_compute_contract.py`, and the Task 2
  driver tests.** Expected: PASS.
- [ ] **Step 5: No-spend check.** First show the owner the untracked paths that T carries and the archive size
  (`stage_tree`, `git ls-files --others --exclude-standard`). The upload sends them to the private HF bucket, so
  **stop and wait for the owner's yes** (TB-PLAN-59). After the yes, run `python -m scripts.hf_jobs campaign
  scripts/hf_jobs/campaigns/tracking-compute-w0-pilot.toml preflight --stage-tree` (it uploads the archive to the
  private bucket and starts no job). Expected: every check passes. Then `launch --stage-tree --dry-run`. Expected:
  the launch decision, and no job.

### Task 3D: R25–R27 — the base-leg defects and the HF store read path (owner rulings R25–R27, spec r7.16)

**Files:** `scripts/measure_tracking_compute.py`, `scripts/hf_jobs/datasets.toml`, `scripts/hf_jobs/datasets.lock.json`,
`scripts/hf_jobs/datasets.py`, `scripts/hf_jobs/campaigns/tracking-compute-w0-{base,final}.toml`, `scripts/_stages.py`,
`scripts/hf_jobs/launch.py`, `scripts/hf_jobs/store.py`, `scripts/hf_jobs/campaigns/tracking-compute-w0-pilot.toml`,
the ADR-128 file, `docs/howto/compute-pipelines.md`. Tests: `tests/scripts/test_measure_tracking_compute_r21.py`,
`tests/scripts/hf_jobs/test_launch_typed.py`, `tests/scripts/hf_jobs/test_http_store.py` (new),
`tests/scripts/test_stages_store_env.py` (new).

**Interfaces:**
- R25 (done 2026-11-10): `pitch_control_warm_frames(by_key, requests) -> pd.DataFrame` (each distinct frame once);
  `WARMUP_ERROR_FIELDS = ("warmup_error", "profile_warmup_error")`; `check_warmups(timings: {stem: timings.json})`
  raises `SystemExit` that names each stem, step, backend, and field. `_profile_work` records `warmup_error`;
  `assemble_match` keeps it as `profile_warmup_error`; `_export_work` calls `check_warmups`.
- R26 (done 2026-11-10): the `tc3_download` entry and its lock (`datasets record tc3_download`); the base and
  final campaigns name it; `datasets.main` splits its argv at `--` before it parses.
- R27 (spec r7.17; A28-PLAN-122–125, TB-PLAN-67–72):
  - `scripts/hf_jobs/store.py`: `HTTP_MIN_BYTES = READ_LOG_BYTES`, `HTTP_READ_FLOOR_S = 60`,
    `HTTP_READ_MIN_RATE = 1_000_000` (bytes/s), `HTTP_READ_TRIES = 5`, `HTTP_BACKOFF_CAP_S = 60`.
    `http_read(api, bucket, path, *, info, timeout_s, child_argv=None) -> bytes | None` does ONE read: the
    transfer runs in a child process (`python -m scripts.hf_jobs.store _get <bucket> <path> <dest>`) under
    `timeout_s`. A timeout kills the child and raises `TimeoutError`. The child's entry is
    `_get_main(argv, *, api=None) -> int`: it writes `<dest>.status.json` with `ok`, `status`, `retry_after`, and
    `error` (spec R27 Throttling; `status` and `Retry-After` from an `HfHubHTTPError` response, else `null`). The
    parent reads that record. 429 and 5xx retry as the spec says, with `retry_after` first and
    `[read] retry <status>` lines. No copy is kept. `BucketStore.read_bytes` keeps its memo and calls
    `http_read`.
  - `HttpReadStore(LocalStore)` in `scripts/hf_jobs/store.py`, fields `bucket: str`, `prefix: str`,
    `api: Any = None`. `read_bytes(rel)`: open on the mount (missing: `None`); below `HTTP_MIN_BYTES`, read
    the mount. Else `get_bucket_paths_info` for `<prefix>/<rel>` (after `install_timeouts`).
    `bucket_ms(entry) -> int | None` gives `mtime`, else `uploaded_at`, as floored UTC epoch milliseconds;
    the mount time is `floor(st_mtime_ns / 1e6)`. The branches and their one tag each are the spec's: no entry or
    an older bucket time gives `[read] mount newer`; equal sizes and a bucket time not older give `[read] http`;
    different sizes give `[read] fallback size`; no bucket time gives `[read] fallback no-mtime`. Each line names
    the reading job (`launch=<SILLY_KICKS_LAUNCH_ID>`); `test_large_read_fallback_reasons` asserts it. Writes,
    `names`, `delete`, and `local_path` are `LocalStore`'s.
  - `scripts/_stages.py`: `BUCKET_MOUNT = "/scratch"` (moved from `launch.py`, which imports it),
    `STORE_BUCKET_ENV = "SILLY_KICKS_STORE_BUCKET"` (`<ns>/<bucket>`) and
    `STORE_MOUNT_ENV = "SILLY_KICKS_STORE_MOUNT"`. `open_store(uri, env=os.environ)` lazily builds an
    `HttpReadStore` when both are set and the path is under the mount (the prefix is the path relative to the
    mount). In an HF job (`SILLY_KICKS_JOB_DEADLINE_EPOCH` set), a path under `/scratch` with a missing or
    malformed store variable raises `ValueError`. Otherwise it returns `LocalStore`, as before. The context loader
    (`store_root`, `_stages.py:1250`) opens its store with `open_store`.
  - `scripts/hf_jobs/launch.py`: `job_env(spec)` sets both variables (`spec.bucket`, `BUCKET_MOUNT`).
    In `job_command`, right after `set -euo pipefail` (`launch.py:218`) and before the deadline line, come three
    lines (TB-PLAN-80): `SILLY_KICKS_LAUNCH_ID=$(cat
    /proc/sys/kernel/random/uuid)` (a bare assignment, so a failing `cat` trips `set -e`), then
    `[ -n "$SILLY_KICKS_LAUNCH_ID" ]`, then `export SILLY_KICKS_LAUNCH_ID`. So each launch gets its own id at
    run time, and a job without one never starts (TB-PLAN-77, A32-PLAN-127). `_stages` adds
    `"launch_id": os.environ.get("SILLY_KICKS_LAUNCH_ID")` to the unit ledger record. `reduce_stage` writes
    `<gen>/_reducers/<launch_id>.json` (`{"launch_id", "stage"}`) before `_output` when the variable is set, and
    nothing when it is not (TB-PLAN-78).
  - `scripts/hf_jobs/campaigns/tracking-compute-w0-pilot.toml`: `dataset = "tc3_download"`, and its args drop
    `--sample` (the probe; the pilot measurements are done).
  - Docs: ADR-128 amended in place (the store section: the R27 read rule and the measurements);
    `docs/howto/compute-pipelines.md`; the `Store` docstring (`_stages.py:282`); the `store.py` module docstring.
    Task 16's CHANGELOG entry names R25, R26 (with the #288 fetch fix), and R27.

- [x] **Step 1 (R25, R26): the failing tests, then the fixes.** `test_pitch_control_warm_frames_have_a_unique_index`,
  `test_match_reduce_keeps_the_profile_warmup_error`, `test_export_refuses_a_warmup_error`,
  `test_the_download_fetch_line_parses_in_the_fetch_cli`. Each one was RED for its reason (`AttributeError`,
  `KeyError`, `SystemExit: 2`) and is GREEN.
- [ ] **Step 2 (R27): write the failing tests** in `tests/scripts/hf_jobs/test_http_store.py` (new), with a fake API
  (`get_bucket_paths_info` returns planted size and mtime) and a fake child that writes planted bytes:
  - `test_large_read_goes_over_http_when_the_bucket_is_current`: equal size, bucket mtime equal to the mount's;
    the HTTP bytes come back and the log has `[read] http`.
  - `test_large_read_uses_the_mount_when_the_bucket_is_older`: no bucket entry, then an older bucket time; the
    mount bytes come back and the log has `[read] mount newer`.
  - `test_large_read_fallback_reasons`: different sizes with a current bucket time give `[read] fallback size`;
    `mtime` and `uploaded_at` both `None` give `[read] fallback no-mtime`; `mtime` `None` with `uploaded_at` set
    uses `uploaded_at`.
  - `test_millisecond_truncation_is_not_older`: a mount mtime of `t` ns and a bucket mtime of `t` floored to ms
    give `[read] http`.
  - `test_small_files_never_use_http`: a `.failed.json` with `attempt: 2` on the mount and `attempt: 1` planted
    in the fake API; the read gives `attempt: 2` and the fake API saw no call.
  - `test_missing_on_the_mount_is_none`: no mount file; `None`, and no API call.
  - `test_a_blocked_transfer_times_out`: a child that sleeps; `TimeoutError` within `timeout_s` + 2 s, and the
    child is gone.
  - `test_get_main_records_429_and_retry_after`: the real `_get_main` with a fake API that raises an
    `HfHubHTTPError` with status 429 and `Retry-After: 7` writes `status: 429, retry_after: 7`; a plain
    `RuntimeError` writes `status: null`.
  - `test_429_then_success`: a fake child records 429 with `retry_after: 0`, then succeeds; the bytes come back and
    the log has one `[read] retry 429`. Five 503s raise; a `status: null` failure raises at once.
  - `test_writes_lists_and_deletes_stay_on_the_mount`: `write_bytes`, `names`, `delete`, and `local_path` act on
    the mount; the fake API saw no call.
  - `test_bucket_store_memo_uses_http_read`: `BucketStore.read_bytes` returns the `http_read` bytes once and
    memoizes them.
  - In `tests/scripts/test_stages_store_env.py` (new): `test_open_store_picks_http_only_under_the_mount`,
    `test_open_store_refuses_a_bad_env_in_a_job` (a missing or malformed variable with the job deadline set
    raises; off-job it gives `LocalStore`), and `test_worker_context_keeps_the_http_store` (`context_to_json`
    and its loader under the same env give an `HttpReadStore`).
  - In `test_launch_typed.py`: `test_job_env_sets_the_store_bucket`, and `test_each_launch_gets_its_own_id`
    (POSIX): the script's three launch-id lines, run twice in `bash`, give two different UUIDs, so a reduce
    launch and a one-unit run job with the same `job_label`, and two attempts of one shard, never share an id.
    With the uuid path replaced by a missing file, the lines exit non-zero under `set -euo pipefail`.
  - The probe script treats an empty or missing `launch_id` (on the line or in the ledger) as ANOTHER job, never
    as a match (A32-PLAN-127). Its own check: a planted ledger with `launch_id: ""` and a `mount newer` line
    with `launch=` fail the probe.
  - In the `_stages` tests: `test_ledger_records_the_launch_id` (the env value is in `.ledger.json`);
    `test_reducer_marker_and_output_bytes` (with the variable set, the marker exists and `_output` and
    `_summary.json` are byte-equal to a run without it; two reducers with two ids leave two markers and one
    output; without the variable, no `_reducers/` directory).
- [ ] **Step 3: Run them.** Expected: FAIL (`ImportError` for `HttpReadStore` and `http_read`, then the missing
  env).
- [ ] **Step 4: Implement the R27 Interfaces above, and the docs.**
- [ ] **Step 5: Run them, plus `tests/scripts/hf_jobs/`, `tests/scripts/test_compute_contract.py`, the `_stages`
  tests, the toy parity test, and the lint trio.** Expected: PASS.
- [ ] **Step 6: The probe.** Show the owner the new tree. After the owner's yes, upload it and run the staged pilot
  campaign. Pass (spec R27 probe): the run completes with no `Errno 5`; no `[read] fallback`; every read of 8 MB
  or more of a file that another job wrote logs `[read] http` (at least one), with no separate reduce rule;
  `[read] mount newer` appears only where the reading job's launch id (on the line) equals the
  `launch_id` in the unit ledger of the file's unit, or for a reduced output one of its `_reducers/` markers
  (TB-PLAN-76-78). Any other file of 8 MB or more read as `mount newer` fails the probe. Map each read to its writer with a script under
  `D:\Temp\tbdfl` over the retrieved ledgers and the job logs. From the job logs, report per stage and per reduce the count and total time of
  each tag and the `retry` count. A failed condition stops the work and goes to the owner.
- [ ] **Step 7: The HF base leg.** After the owner's yes, launch the base campaign with `--budget-s 7200`, then
  `retrieve --stage-tree`. The DGX and Mac base legs ran on tree `a962124f1dfa` and need no rerun: R27 and the
  R26 CLI fix change neither the package hash nor `slice_digest`. The memo records each leg's tree.

### Task 4: W9 linker core — game-aware linking, action_row, fingerprint, binding helper, binding fixtures (spec §4.9 W9.1–W9.3, W9.5)

**Files:**
- Create: `silly_kicks/tracking/_link_identity.py`
- Modify: `silly_kicks/tracking/utils.py`:
  - `link_actions_to_frames` (`:324`);
  - `link_actions_to_frames_elastic` (`:507`);
  - `_count_candidates_within_tolerance` (`def` at `:1257`);
  - the floor message count (`:701`);
  - `_diagnose_time_base` (`def` at `:730`; called at `:696` and `:1173`);
  - the public re-export of `action_fingerprints` and `FINGERPRINT_RECORD_FORMAT` (A1-PLAN-58).
- Modify: `silly_kicks/tracking/schema.py:263` (`LinkReport`).
- Modify: `tests/_scale_guarded.py`, registering the rewritten candidate counter.
- Test: `tests/tracking/test_link_identity.py`, `tests/tracking/test_tracking_utils_link_multigame.py`,
  `tests/test_link_binding.py`

**Interfaces:**
- Produces, in `_link_identity.py` (private; the public path is `silly_kicks.tracking.utils`):
  - `fingerprint_records(actions: pd.DataFrame) -> list[str]`: the record strings, built once per column with
    `canonical_id_series` (A2-PLAN-66).
  - `action_fingerprints(actions: pd.DataFrame) -> np.ndarray` (uint64, `len(actions)`), the BLAKE2b-64 of each
    record.
  - `FINGERPRINT_RECORD_FORMAT = "{game}|{period}|{action}|{time}"`.
  - `link_identity_scope()`, a context manager that opens the per-call cache. A cache entry holds
    `(frame_ref, fingerprints)`, and a hit requires `entry.frame_ref is actions`. It is cleared on exit.
  - `resolve_link_positions(actions: pd.DataFrame, links: pd.DataFrame) -> np.ndarray` (int64 position into
    `actions` for each `links` row). It applies the W9.5 rules: a legacy single-game unique-id fallback; otherwise a
    range check and a fingerprint check that raise `ValueError("links must be built from exactly this actions
    frame, in this row order; first mismatch at links row {i}")`.
  - `remap_links(links: pd.DataFrame, permutation: np.ndarray, new_actions: pd.DataFrame) -> pd.DataFrame`, for the
    internal re-sort and filter sites.
- Produces, linker:
  - The pointers keep their existing five columns, then gain `action_row` (int64) and `action_fingerprint` (uint64).
  - Row order is a stable sort on `(period_id, time_seconds, game_rank)`.
  - `LinkReport.per_game_period_link_rate: dict[tuple[str, int], float]`, keyed by
    `(canonical_id(game_id), period_id)`.

- [ ] **Step 1: Write the failing tests.** The frozen vector covers every class in spec W9.2, each in the dtype it
  has in real input (TB-PLAN-06). The expected id fields below were computed on the base `id_compat` on
  2026-11-08, and the NaN time uses the spec W9.2 token `"<NA>"` (A2-PLAN-60). They are the contract, and the
  uint64 literals are their BLAKE2b-64 values.

```python
FROZEN_FRAMES = {
    "object": pd.DataFrame({"game_id": pd.Series([7, "7", "abc", None], dtype=object), "period_id": [1, 1, 1, 2],
                            "action_id": pd.Series([0, "0", "x9", 5], dtype=object),
                            "time_seconds": [12.04, 0.0, -0.0, np.nan]}),
    "typed": pd.DataFrame({"game_id": pd.array([7, pd.NA], dtype="Int64"),
                           "period_id": pd.array([1, pd.NA], dtype="Int64"),
                           "action_id": np.array([0.0, 3.0]),                       # the real slot case (float64)
                           "time_seconds": np.array([12.04, 1234.56], dtype=np.float32)}),
    "nan_period": pd.DataFrame({"game_id": np.array([8]), "period_id": np.array([np.nan]),
                                "action_id": np.array([4]), "time_seconds": [1234.56]}),
}
FROZEN_RECORDS = {
    "object": ["7|1|0|0x1.8147ae147ae14p+3", "7|1|0|0x0.0p+0", "abc|1|x9|-0x0.0p+0", "<NA>|2|5|<NA>"],
    "typed": ["7|1|0|0x1.8147ae0000000p+3", "<NA>|<NA>|3|0x1.34a3d80000000p+10"],
    "nan_period": ["8|<NA>|4|0x1.34a3d70a3d70ap+10"],
}
FROZEN_VECTOR = {...}  # literal uint64 lists per frame, = blake2b(record.encode(), digest_size=8) little-endian

@pytest.mark.parametrize("name", sorted(FROZEN_FRAMES))
def test_fingerprint_records_and_vector(name):
    assert fingerprint_records(FROZEN_FRAMES[name]) == FROZEN_RECORDS[name]      # private record builder
    assert action_fingerprints(FROZEN_FRAMES[name]).tolist() == FROZEN_VECTOR[name]

def test_fingerprint_float32_rounded_time_differs():
    a = pd.DataFrame({"game_id": [7, 7], "period_id": [1, 1], "action_id": [0, 0],
                      "time_seconds": [12.04, float(np.float32(12.04))]})
    fp = action_fingerprints(a)
    assert fp[0] != fp[1]

def test_public_path():
    from silly_kicks.tracking.utils import FINGERPRINT_RECORD_FORMAT, action_fingerprints  # noqa: F401

def test_two_game_reproduction_links_own_game():
    actions, frames = spec_4_9_reproduction_inputs()
    ptr, rep = link_actions_to_frames(actions, frames)
    assert ptr.sort_values("action_row")["frame_id"].tolist() == [10, 30, 10, 30]
    assert list(ptr.columns[-2:]) == ["action_row", "action_fingerprint"]

def test_single_game_pointers_unchanged():
    actions, frames = make_single_game_inputs()
    new, _ = link_actions_to_frames(actions, frames)
    old, _ = linker_v4139.link_actions_to_frames(actions, frames)
    assert_bits_equal(new.iloc[:, :5], old)

def test_single_game_warning_text_pinned(recwarn): ...  # exact strings captured from linker_v4139 on the same input
def test_resolve_link_positions_rejects_reordered_frame(): ...
def test_legacy_links_single_game_unique_ok_multigame_raises(): ...
def test_cache_never_serves_freed_frame(): ...   # create/drop temp frames in a loop inside one scope; fresh fingerprints each time
def test_na_game_links_only_na_frames(): ...     # Review Focus 2
def test_period5_unlinked_no_cross_game(): ...   # Review Focus 4
def test_empty_actions_pointers_have_new_columns(): ...  # Review Focus 1
```

  Compute each `FROZEN_VECTOR` list once with a direct `hashlib.blake2b` over the `FROZEN_RECORDS` strings, not with
  the code under test, then commit the literals.
- [ ] **Step 2: Write the binding fixtures** (`tests/test_link_binding.py`), parametrised over every `links=`
  acceptor in `LINK_POPULATION`.
  - **Caller-mismatch cases** (mart order, reversed games, subset) expect `ValueError` matching
    `"links must be built from exactly this actions frame"`.
  - **Internal re-sort cases** use **single-game** input (A1-PLAN-57). They call `add_pre_shot_gk_context`
    (standard and atomic), `resolve_restart_geometry`, and every re-sort site in the census, with matching `links`.
    They expect values equal to the clean call (`links=None`) under `assert_bits_equal`. A raise is a failure.
  - **Planted mutations** use a fixture `planted(mode)` that monkeypatches `_link_identity`:
    - `"no_verify"` skips the fingerprint check;
    - `"forward_unchanged"` makes `remap_links` a no-op.

    Each case kind goes red under the mutation that removes its own protection (TB-PLAN-15):

    | Case kind | Mutation that must turn it red | Why |
    |---|---|---|
    | caller mismatch (mart, reversed, subset) | `no_verify` | no check raises, so the case gets no `ValueError` |
    | internal re-sort or filter | `forward_unchanged` | the stale pointers then fail the verification, or give wrong values |

    `test_mutation_turns_fixture_red[case]` asserts each pairing in the table. The other pairings are not asserted,
    because correct code can stay green under them. This is the red-first proof for the caller-mismatch cases. The
    reversed-game defect is also red at the base through the Task 3 game-split property (spec W9.6).
- [ ] **Step 3: Run them.** Expected: the Step 1 tests FAIL (missing names). The binding cases cannot run yet
  either. The consumers still join on `action_id`, so the binding cases go green only in Task 5.
- [ ] **Step 4: Implement.**
  - The frame index is built on `(game_key, period_id, frame_id)`, where `game_key` comes from
    `canonical_id_series` with NA mapped to `"<NA>"`. Linking runs per `(game_key, period_id)` group through
    `group_rows`, with `merge_asof` inside each group as today.
  - `_count_candidates_within_tolerance` uses a per-group `searchsorted` and has no `iterrows`. Register it in
    `SCALE_GUARDED` with a call counter.
  - The elastic linker gets the same keys.
  - The fingerprint is vectorised: one `canonical_id_series` call per column, then one Python loop over the joined
    records with `hashlib.blake2b(digest_size=8)`, read with `int.from_bytes(..., "little")`.
- [ ] **Step 5: Run them again.** Expected: the Step 1 tests PASS. The planted-mutation tests PASS for the linker's
  own cases. The existing link tests (`tests/test_tracking_utils_link.py`,
  `tests/tracking/test_elastic_linker_contract.py`) also PASS without edits, apart from asserting the two new
  columns.

### Task 5: W9 consumer migration — positional joins, game-keyed frame joins, re-sort remaps, scripts (spec §4.9 W9.4, R6, R8)

**Files:** every member of `LINK_POPULATION` that joins pointers. The starting census is the spec §4.9 caller table
plus:
- `tracking/utils.py:1317-1322`, `_resolve_action_frame_context`;
- `tracking/features.py:5576-5591` (R6) and `:5727` (`_row_idx`);
- `tracking/_kernels.py:959`;
- `tracking/features.py:1104`;
- `spadl/utils.py:753-754` and `:840`;
- `atomic/spadl/utils.py:900` and `:990-991`;
- `scripts/_rq_corpus.py:58-60`, `scripts/tune_structural_pass_sigma.py:40`,
  `scripts/measure_cover_shadow_argmax_agreement.py:119`.

Test: `tests/tracking/test_shape_graph_rekey.py`, `tests/scripts/test_link_script_joins.py`

**Interfaces:**
- Consumes: `resolve_link_positions`, `remap_links`, and `link_identity_scope` (Task 4).
- Produces: no new public names. Every pointer→action join goes through `resolve_link_positions`, and every
  pointer→frame join keys on `(game_key, period_id, frame_id)`.

- [ ] **Step 1: Extend the spec caller table** (the AGENTS.md "enumerate every caller" rule). Give each row the line
  evidence for **both** sides of the join, and a checkbox. Expected: every population member appears in it, and
  `test_population_matches_registry` stays green.
- [ ] **Step 2: Migrate one row at a time** (TB-PLAN-13). For each row of the extended table:
  1. Apply the change that fits the row:
     - positional rows come from `resolve_link_positions`, and the action's `game_id` is read at those positions;
     - frame merges move from `["period_id", "frame_id"]` to `align_join_keys` on `["game_id", "period_id", "frame_id"]`;
     - `_row_idx` and the other surrogates are replaced by `action_row`;
     - a re-sort site calls `remap_links(links, perm, sorted_actions)`;
     - a public entry point that forwards `links` wraps its body in `link_identity_scope()`.
  2. Run that member's game-split cases (both variants) and its binding cases. Expected: PASS.
  3. Tick the row's checkbox.
- [ ] **Step 3: R2 re-key detection, then the fix.** Write `test_shape_graph_game_id_rekey_detection`: an int-vs-str
  `game_id` between actions and frames. It must equal the result for aligned ids. Run it. Expected: FAIL on the
  current `astype(str)` re-keys (`features.py:5589`, `:5749`). Then remove the re-keys, and run it again. Expected:
  PASS.
- [ ] **Step 4: Script joins, one site at a time (R8; A1-PLAN-51).** For each `scripts/` row:
  1. If the join is inline in `main`, extract it into a named function, with no behaviour change.
  2. Write its test in `tests/scripts/test_link_script_joins.py`, on `make_two_game_inputs`: each action joins only
     its own game's frames. Run it. Expected: FAIL.
  3. Migrate the join, and run the test again. Expected: PASS.
- [ ] **Step 5: Remove the Task 3 `xfail` marks.** Run `python -m pytest tests/test_multigame_linking.py
  tests/test_link_binding.py tests/test_link_recorder.py -v`. Expected: all PASS, including the planted-mutation
  tests.
- [ ] **Step 6: Run the full local gate.** Expected: PASS. A failing pre-existing test is either a real regression,
  which you fix, or a test that asserted the old multi-game, duplicate, or `str`-key behaviour. For the second kind,
  update it **only** if its input belongs to an R1/R2/R5/R6/R9 class. Name it in the session notes for the
  CHANGELOG.

### Task 6: W1 — public backend query (spec §4.1)

**Files:** Create `silly_kicks/backends.py`. Test: `tests/test_backends.py`.

**Interfaces:**
- Produces:
  - `active_backend(family: str) -> Literal["numba", "numpy"]`;
  - `backend_report() -> dict[str, str]`, keyed by `FAMILIES` plus `"numba_version"`, with values `"numba"`,
    `"numpy"`, or `"unavailable"`;
  - the re-exports `FAMILIES` and `NumbaUnavailableError`.

- [ ] **Step 1: Write the failing tests.**
  - `test_active_backend_force_numpy[family]`: with `SILLY_KICKS_<F>_FORCE_NUMPY=1`, it returns `"numpy"`.
  - `test_active_backend_unknown_family_raises`: raises `ValueError`.
  - `test_active_backend_missing_numba_raises`: monkeypatch `_numba_dispatch._import_numba` to raise; it raises
    `NumbaUnavailableError`.
  - `test_backend_report_never_raises_when_numba_missing`: it returns `"unavailable"` for unforced families and
    for `numba_version`.
  - The `Examples` sections of both functions run under the doctest leg.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement it as a thin façade over `_numba_dispatch`.**
- [ ] **Step 4: Run them again,** plus the doctest leg and `tests/test_public_api_examples.py`. Expected: PASS.

### Task 7: W2 — columnar index, per-frame extraction, batched combine, `_influence_rows`, `_cover_shadows`, R2 dispatch key (spec §4.2, R2)

**Files:**
- Create: `silly_kicks/tracking/pitch_control/_index.py`, with `_SpearmanFrameIndex`.
- Modify: `pitch_control/_spearman_batch.py`, `pitch_control/_spearman.py`, `pitch_control/_dispatch.py:212-288`,
  `pitch_control/_cache.py` (`warm`, `:114`), and `pitch_control/_numba_kernels.py`.
- Modify: `silly_kicks/tracking/_cover_shadows.py:908-954` (`compute_threat_pc_batch`).
- Test: `tests/tracking/pitch_control/test_spearman_index.py`, `tests/tracking/pitch_control/test_spearman_batch.py`
  (extended), `tests/tracking/test_cover_shadows.py` (extended).

**Interfaces:**
- Produces (internal):
  - `_SpearmanFrameIndex.from_groups(groups: RowGroups) -> _SpearmanFrameIndex`;
  - `_SpearmanFrameIndex.from_slices(slices: Sequence[pd.DataFrame]) -> _SpearmanFrameIndex`, which never dedups by
    frame id;
  - `_influence_rows(tti_rows: np.ndarray, opp_min_rows: np.ndarray, sigma: float) -> np.ndarray`, with a numpy
    reference and a numba kernel `influence_rows_numba` in the `PITCH_CONTROL` family;
  - `_spearman_batch_core(index, requests, params) -> list[PitchControlSurface]`.
- `compute_spearman_batch(frame_slices, teams, decomposes, ball_positions, *, params)` keeps its exact signature.

- [ ] **Step 1: Write the failing parity tests** against `tests/oracles/spearman_v4139.py` and
  `tests/oracles/cover_shadows_v4139.py`. Each one runs on both backends, with
  `SILLY_KICKS_PITCH_CONTROL_FORCE_NUMPY` set to 0 and to 1.
  - `test_batch_bit_identical_production[decompose,ball]` uses `make_production_frames()` and all requests, and
    checks `assert_bits_equal(new, oracle)` per surface field.
  - `test_batch_edge_frames`: an empty frame, no attackers, no defenders, both-team requests on one frame, duplicate
    requests, and mixed id content (`make_production_frames(mixed_ids=True)`: raw and canonical `ids_match` paths in
    one batch).
  - `test_slice_index_two_slices_one_key`: the cover-shadow pattern of a frame plus its no-GK counterfactual.
  - `test_cover_shadows_threat_pc_batch_bit_identical`: `compute_threat_pc_batch` against the oracle, on the
    cover-shadow fixtures (TB-PLAN-07c).
  - `test_categorical_team_id_identical`: Review Focus 3.
  - `test_dispatch_key_exact_type`: the requests `((k), 1, False)` and `((k), "1", False)` give surfaces whose
    `attacking_team_id` values are `1` and `"1"`. It fails on the current `str()` key (`_dispatch.py:234`). It is
    the R2 detection test.
  - Structural guards, using `tests._perf_structural.call_counter`:
    - `compute_tti` is called once per chunk;
    - `_influence_rows` is called a constant number of times, independent of the frame count, with decompose
      included;
    - extraction runs once per distinct frame;
    - the total `ids_match` elements equal the sum of the request slice lengths.
- [ ] **Step 2: Run them.** Expected: FAIL. `test_dispatch_key_exact_type` fails on the current code, and the new
  names are missing.
- [ ] **Step 3: Implement it as spec §4.2 W2.1–W2.7 sets out.**
  - Sums are per-frame C-order `.sum(axis=0)` over contiguous slices. `reduceat` is banned.
  - The dedup key is `(frame_key, type(team), team, decompose)`.
  - **`_spearman_combine` (spec §8, settled).** After W2.6, `compute_spearman` builds a one-frame index and calls
    `_spearman_batch_core`. No package caller of `_spearman_combine` or `_extract_frame_players` remains, so delete
    both from the package (AG2). They live on only in the oracle. Move the sum-order premise, corrected to
    "sequential C-order axis-0 sum", to the `_spearman_batch_core` docstring.
  - `_cover_shadows.compute_threat_pc_batch` calls `compute_spearman_batch`, which now builds `from_slices`.
  - **Gated candidate (TB-PLAN-04, TB-PLAN-21).** `influence_rows_numba` is not bit-identical by construction.
    Build it always: spec W2 requires both a numpy reference and a numba kernel for `_influence_rows`, with no
    condition. (The Task 10 and Task 13 candidates are conditional because spec W5.1 leaves the technique to the
    builder and spec W7 says "W0 decides what is worth doing".) It is the shipping default: `PITCH_CONTROL` numba is
    on when numba is available. Then the frozen tree, the after-run, and the memo all measure the code that ships.
    A gate mismatch in Task 15 (every platform, HF included; spec R23) reverts it to the numpy reference.
- [ ] **Step 4: Run them again,** plus the existing `tests/tracking/pitch_control/` and cover-shadow tests. Expected:
  PASS.

### Task 8: W3 — `max_frames_per_chunk`, public index, index-path batch (spec §4.3)

**Files:** Modify `pitch_control/_dispatch.py` and `pitch_control/__init__.py`. Test:
`tests/tracking/pitch_control/test_batch_chunking.py`.

**Interfaces:**
- Produces (public, in `silly_kicks.tracking.pitch_control`):
  - `PitchControlFrameIndex`, a frozen dataclass with fields `groups: RowGroups` and `_spearman: _SpearmanFrameIndex`;
  - `build_pitch_control_index(frames: pd.DataFrame) -> PitchControlFrameIndex`;
  - `compute_pitch_control_batch(frames, requests, *, method="spearman", params=None, max_frames_per_chunk: int | None = 2048) -> list[PitchControlSurface]`;
  - `compute_pitch_control_batch_from_index(index: PitchControlFrameIndex, requests, *, method="spearman", params=None, max_frames_per_chunk: int | None = 2048) -> list[PitchControlSurface]`.

- [ ] **Step 1: Write the failing tests.**
  - `test_chunk_sizes_bit_identical[method][size]` covers `method ∈ {spearman, voronoi, fernandez_bornn}` and
    `size ∈ {1, 7, 2048, None}`.
  - `test_index_path_equals_frames_path`.
  - `test_group_rows_called_once_per_call`.
  - `test_invalid_chunk_size_raises[0, -1, 2.5]`.
  - `test_zero_requests_returns_empty_list`: Review Focus 1.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Chunk over distinct frames, with every request on a chunk's frames in that chunk.
  Validate the chunk size. Give each new public name an `Examples` section. The docstring's measured-peak sentence
  keeps a `<PEAK>` placeholder until Task 15 step 2, which is still before the freeze.
- [ ] **Step 4: Run them again,** plus the doctest leg. Expected: PASS.

### Task 9: W4 — backend parity record (spec §4.4)

**Files:** Modify `tests/tracking/pitch_control/test_numba_parity.py`.

- [ ] **Step 1: Add the tests** (TB-PLAN-07a). Each keeps today's **per-element** rule,
  `np.testing.assert_allclose(numba, numpy, rtol=1e-12)` (`test_numba_parity.py:31,41`), and prints the measured
  `max_abs_diff` in a `-s` line for the memo:
  - `test_tti_parity_production`;
  - `test_influence_parity_production` (`influence_numba`);
  - `test_influence_rows_parity_production` (`influence_rows_numba`, only if Task 7 built it);
  - `test_tti_at_target_branch[0.0, 5e-11]`: `distance == 0` and `0 < distance < 1e-10` give `reaction_time` on
    both backends;
  - one production-shape parity test per W5.1/W7.2 prototype kernel that Tasks 10 or 13 build.
- [ ] **Step 2: Run them.** Expected: PASS on the current kernels. A larger measured difference means you **stop and
  report to the owner** (spec §4.4). The corpus max abs diff per kernel and CPU class comes from the driver's
  `kernel_diff.json` (Task 15).

### Task 10: W5.1 — faster shape-graph core, bit-identical (spec §4.5 W5.1)

**Files:** Modify `silly_kicks/tracking/_shape_graph.py`. Test: `tests/tracking/test_shape_graph_parity.py`.

**Interfaces:** the public signatures of `compute_shape_graph` and `infer_positions` are unchanged.

- [ ] **Step 1: Write the failing parity tests** against `tests/oracles/shape_graph_v4139.py`.
  - `test_shape_graph_bit_identical[seed]` runs 200 seeded production-shape team-frames and checks
    `assert_bits_equal` on `edges`, `faces` (in order), `stabilities`, and `points`.
  - `test_shape_graph_degenerate[collinear, n2, duplicates, exact_ties]`, with the `QhullError` warning equal.
  - `test_infer_positions_bit_identical[direction ±1]`. It holds until Task 10A, which replaces it with the W5.6
    relation tests.
  - `test_shape_graph_call_cost_guard`: a `call_counter` on `_angle_at_vertex` shows strictly fewer calls than the
    oracle on the production fixture. This is the structural proof of the speed-up, with no wall-clock assertion.
- [ ] **Step 2: Run them.** Expected: the parity tests PASS (the code is unchanged) and the cost guard FAILS.
- [ ] **Step 3: Implement the by-construction speed-ups only:**
  - an edge→incident-face index maintained across merges;
  - cached opposite angles per (edge, face);
  - `math` scalars for `+ − × ÷ sqrt`.

  Keep `np.arccos`, `np.arctan2`, and `np.degrees` with the same inputs. The tie-break order, the 1e-10 tolerance,
  the face order, the edge sort, and the warning must not change.
- [ ] **Step 4: Run them again.** Expected: all PASS.
- [ ] **Step 5 (conditional, §5.2):** only if W0 shows that the transcendental calls are still material. Then prototype a numba or `math` variant behind a
  new dispatch family, as the shipping default (the Task 7 rule). Add its Task 9 parity test.

### Task 10A: W5.6 — team-relative position labels (spec §4.5 W5.6, R14)

**Files:**
- Modify: `silly_kicks/tracking/_shape_graph.py` (the label path only: `infer_positions` `:863+`,
  `_assign_levels_vertical` `:702`, `_assign_levels_horizontal` `:771`, `_decompose_middle` `:569`,
  `_face_centroids` `:442`, `_HORIZONTAL_LEVELS` `:82`).
- Modify: `tests/tracking/test_shape_graph.py`. Replace `test_lateral_label_is_pitch_absolute_not_team_relative`
  (`:285`) with the mirror and D-lat tests below.
- Modify: `tests/tracking/test_shape_graph_parity.py`. Replace the Task 10 `test_infer_positions_bit_identical` with
  the relation tests below.
- Modify: `tests/tracking/_mirror_entries/shape_and_structure.py:31` (the pitch-absolute note).
- Modify: `scripts/measure_tracking_compute.py`, wiring the final-phase `pitch_thirds`, `lateral_fallback`, and
  `lateral_middle_tie` flags (Task 2).
- Test: `tests/tracking/test_shape_graph_labels.py`.

**Interfaces:**
- Consumes: `to_goal_relative_x_array` and `to_goal_relative_y_array` (`tracking/_geometry.py:204`, `:218`), with
  `goal_x=0.0` for `attacking_direction > 0` and `goal_x=FIELD_LENGTH` for `< 0`; `tests._perf_structural.call_counter`.
- Produces:
  - `infer_positions(shape_graph, positions, attacking_direction) -> list[PositionLabel]`, with the signature
    unchanged. It raises `ValueError` when `not math.isfinite(attacking_direction) or attacking_direction == 0.0`.
  - private `_to_attack_frame(positions: np.ndarray, attacking_direction: float) -> np.ndarray`, the only place
    that reads the direction;
  - private `_pitch_thirds_level(values, middle_indices, levels, *, pitch_dim, label_above, label_below) -> None`,
    the tree branch of `_decompose_middle` moved into its own function, so tests and the W0 driver can count it;
  - `_HORIZONTAL_LEVELS = ("R", "RC", "C", "LC", "L")`, the ascending-y ladder of the attack frame. Its only users
    are `:786` and the horizontal assignment (checked at af03c4b).
  - `_decompose_middle`, `_assign_levels_vertical`, and `_assign_levels_horizontal` lose their `flip` and
    `attacking_direction` parameters. The lateral call uses `label_above="LC"`, `label_below="RC"`.
  - In `_assign_levels_horizontal`, the literal labels swap (TB-PLAN-34): `values > fc_max` gives `"L"` and
    `values < fc_min` gives `"R"`, on the face-centre path (`:796-801`) and the Option D path (`:814-819`).
  - `_face_centroids` sums each face's vertices in ascending vertex-index order.

- [ ] **Step 1: Write the failing tests** in `tests/tracking/test_shape_graph_labels.py`:
  - `test_dlat_plus_x_high_y_full_back_is_L`: on the `positions_442` fixture with `+1.0`, the defender with the
    largest y gets `horizontal == "L"`, and the one with the smallest y gets `"R"`.
  - `test_deqf_fallback_team_relative[+1.0, -1.0]` and its precondition `test_deqf_fixture_reaches_fallback`: the
    fixture has `len(sg.edges) > 0` and an empty `_face_centroids(sg, axis=1)`. For +x, the highest-y player is
    `"L"`. For −x, the lowest-y player is `"L"`. The `[-1.0]` case already passes at the base (pitch-absolute low y is
    "L"), so it is a guard, not a detection test (A11-PLAN-92).
  - `test_dtree_minus_x_deep_middle_is_DM` and its precondition `test_dtree_fixture_reaches_tree_branch`
    (A11-PLAN-93). A wrapper on `_decompose_middle` records its `axis` argument (`axis=0` is vertical), and a spy on
    `_is_tree` inside that call records a `True` return. So the precondition proves the tree branch on the vertical
    axis. Both names exist at the base. The fixture also keeps the horizontal middle group at 2 players or fewer,
    so Option C does not run on that axis (`:595`). The team
    attacks −x, and its middle group is in its defensive third (high x). Expected: `"DM"`. The oracle gives `"AM"`.
  - `test_direction_must_be_finite_nonzero[nan, 0.0, inf, -inf]`: `ValueError`.
  - `test_mirror_invariance[seed]`: 200 seeded float32 team-frames, n from 3 to 10. `R(P)` is computed in float64.
    `infer_positions(compute_shape_graph(P), P, -1.0) == infer_positions(compute_shape_graph(R(P)), R(P), +1.0)`.
  - `test_reflection_preserves_graph_faces[seed]` (the spec premise, TB-SPEC-56): the same edge set and the same face
    vertex sets (as frozensets) for `P` and `R(P)`.
  - `test_minus_x_lateral_fallback_n9`: the pinned labels from the spec rule, rank `r` of `68 − y`,
    `min(5r // 9, 4)` onto `R, RC, C, LC, L`. The expected list is written out in the test.
  - `test_minus_x_two_player_middle_tie` (TB-PLAN-33): a −x team whose horizontal middle group is two players with
    exactly equal y. The new labels are pinned (the lower index gets `"RC"`). The test also asserts that the oracle
    gives the swapped pair, which documents the exact-tie class.

  Add these to `tests/tracking/test_shape_graph_parity.py`, against `tests/oracles/shape_graph_v4139.py`, on the 200
  seeded continuous fixtures:
  - `test_relation_plus_x`: `vertical` equals the oracle's. `horizontal` equals the oracle's with L↔R and LC↔RC.
  - `test_relation_minus_x`: `horizontal` equals the oracle's on every frame where `_equal_frequency_assign` did not
    run with the lateral ladder. `vertical` equals the oracle's on every frame where `_pitch_thirds_level` did not
    run for the vertical axis. A wrapper records the `levels` argument to tell the axes apart. Non-vacuity: at least
    150 of the 200 frames are compared on each axis.
  - The spies degrade (TB-PLAN-35): a spy is attached only when its symbol exists
    (`hasattr(module, "_pitch_thirds_level")`). Otherwise it excludes nothing. So no test fails on a missing name.
- [ ] **Step 2: Run them.** Expected:
  - FAIL: D-lat, D-eqf `[+1.0]`, D-tree, the `ValueError`, the mirror test (lateral), the two-player tie pin, the
    n = 9 pin, and `test_relation_plus_x` (the labels are not yet swapped);
  - PASS: D-eqf `[-1.0]`, `test_relation_minus_x` (the base equals the oracle), the premise, and the two
    preconditions.
- [ ] **Step 3: Implement** spec W5.6: map the positions through `_to_attack_frame` first, compute every label input
  from the attack-frame positions (the Option C sub-graph too), and use the attack-frame ladders on every path.
  Then update the docstring: the convention, a two-team `Examples` that imports from `silly_kicks.tracking`, and
  no ADR-045 D5 text. Update the mirror-entry note.
- [ ] **Step 4: Run them again,** plus `tests/tracking/test_shape_graph.py` and the Task 10 graph parity tests.
  Expected: all PASS. The graph (`compute_shape_graph`) is still bit-identical to the oracle.

### Task 11: W5.2–W5.5 — `compute_shape_graphs`, single source, docstring import rule (spec §4.5)

**Files:**
- Modify: `silly_kicks/tracking/_shape_graph.py` to add the batch API, and `silly_kicks/tracking/__init__.py` to
  export it.
- Modify: `silly_kicks/tracking/features.py` (`add_shape_graph` `:5491+`, `shape_graph_xfns` `:5625+`) and
  `silly_kicks/feature_glossary.py`.
- Modify: `tests/test_public_api_examples.py`. Add the private-import rule there, using its `_examples_section`
  (`:1242`), `_doctest_source` (`:1283`), and `_discover_public_modules` (`:1560`). Do not create a second
  extractor (A1-PLAN-52b).
- Create: `scripts/_frame_reflection.py` (`reflect_frames`, shared with the Task 11A driver). It sits under
  `scripts/`, inside `CODE_IDENTITY_PATHS`, so `FREEZE_ID` covers the mirror arm's code (TB-PLAN-37).
- Test: `tests/tracking/test_shape_graph_batch.py`

**Interfaces:**
- Produces (public, `silly_kicks.tracking`):
  - `compute_shape_graphs(frames: pd.DataFrame, *, stability_threshold: float = 45.0, with_positions: bool = True, retain_graphs: bool = False) -> ShapeGraphBatch`;
  - `ShapeGraphBatch`, a frozen dataclass with `frames_summary: pd.DataFrame`, `players: pd.DataFrame`, and
    `graph(game_id, period_id, frame_id, team_id) -> ShapeGraph`;
  - `SHAPE_GRAPH_SOURCE_VALUES: frozenset[str] = frozenset({"ok", "too_few_players", "degenerate", "direction_unresolved"})`.
- The `frames_summary` columns are `game_id`, `period_id`, `frame_id`, `team_id`, `n_players`, `n_edges`,
  `density`, `mean_stability`, and `shape_graph_source`.
- The `players` columns are `game_id`, `period_id`, `frame_id`, `team_id`, `player_id`, `shape_graph_source`,
  `vertical`, `horizontal`, and `position_label` (spec W5.2, TB-SPEC-55).

- [ ] **Step 1: Write the failing tests.**
  - `test_batch_equals_per_frame_public_calls`.
  - `test_row_order_first_appearance_no_id_sort`, with a mixed-dtype `game_id` that does not raise.
  - `test_direction_from_frame_label`: an `"rtl"` team gets labels equal to `infer_positions(..., -1.0)`, and an
    unknown label gives `<NA>` with `direction_unresolved`.
  - `test_graph_accessor_equals_retained`.
  - `test_mixed_direction_raises[ltr+rtl, ltr+null]`: `ValueError` names the team-frame. The goalkeeper and ball rows
    are not read (TB-SPEC-60). Review Focus 6.
  - `test_direction_checked_only_on_label_path`: the same mixed input does not raise in `add_shape_graph`,
    `shape_graph_xfns`, or `compute_shape_graphs(with_positions=False)`, and there the source is never
    `"direction_unresolved"` (A7-SPEC-90).
  - `test_players_source_filter`: `players["shape_graph_source"]` equals the team-frame's summary value on every row.
    With `with_positions=False`, the source is set and every label is `<NA>`.
  - `test_chunked_calls_concat_equals_whole`: contiguous row slices split on whole frames, concatenated, equal one
    call under `assert_bits_equal` (both DataFrames, row order included) (R16, A7-SPEC-89).
  - `test_all_or_nothing_labels`: in each team-frame, the labels are all set or all `<NA>`.
  - `test_batch_mirror_invariance` (A11-PLAN-91): on float32 frames with both teams and both directions, the
    `players` labels of `compute_shape_graphs(frames)` equal those of
    `compute_shape_graphs(reflect_frames(frames))`, row for row. It uses `reflect_frames` from
    `scripts/_frame_reflection.py`, which this task creates. The Task 11A driver imports the same module, so there
    is one copy.
  - `test_coordinates_read_as_float64`: a float64 `x`/`y` input is used as is. No value is cast to the schema dtype
    (`tracking/schema.py:38-39`).
  - `test_too_few_players_and_ball_only`: Review Focus 5.
  - `test_empty_frames`: Review Focus 1.
  - `test_categorical_team_id`: Review Focus 3.
  - `test_add_shape_graph_bit_identical_single_game` and `test_shape_graph_xfns_bit_identical_single_game`, against
    `tests/oracles/shape_graph_features_v4139.py`.
  - `test_examples_use_public_paths` in `tests/test_public_api_examples.py`: it fails when an `Examples` section
    imports a symbol from a `_private` module although the symbol has a public path.
  - An `assert_subquadratic_growth` guard on `compute_shape_graphs`, registered in `SCALE_GUARDED`.
- [ ] **Step 2: Run them.** Expected: FAIL. `test_examples_use_public_paths` fails on `_shape_graph.py:320`, `:903`
  and `_line_breaking.py:47`, `:97`.
- [ ] **Step 3: Implement.**
  - Use one `group_rows` over (game_id, period_id, frame_id, team_id) on the outfield selection.
  - Read `x` and `y` with `.to_numpy(dtype=np.float64)`. Never cast them to the schema dtype, so a float64 input
    stays exact (A11-PLAN-91).
  - `add_shape_graph` and `shape_graph_xfns` call the same core. They keep the self-referential `team_id` `==`
    exactly (R2). Their action paths are already positional from Task 5.
  - Fix the docstrings that the rule finds.
  - Give each new public name an `Examples` section, and add the glossary entries. The `compute_shape_graphs`
    docstring shows the `"ok"` filter on `players` and the chunked pattern for a caller with a deadline (R16). The
    `vertical`, `horizontal`, and `position_label` glossary entries state the W5.6 convention.
- [ ] **Step 4: Run them again,** plus the doctest leg. Expected: PASS.

### Task 11A: W5.7 — real-data role gate driver (spec §4.5 W5.7, R15)

**Files:**
- Create: `scripts/validate_shape_graph_roles.py`.
- Modify: `tests/scripts/test_provenance_wiring.py`, adding the driver to `ARTIFACT_DRIVERS` (`:28`).
- Test: `tests/scripts/test_validate_shape_graph_roles.py`.

**Interfaces:**
- Consumes: `compute_shape_graphs` (Task 11); `tests/oracles/shape_graph_v4139.py` (the baseline arm);
  `scripts/_loader_pining.build_skillcorner_frames(paths, match_id, tracking_limit, *, with_velocity=False)`
  (`:916`) and `_read_json(paths["metadata"])` (`:660`) for the roster, so no second `match.json` reader is added;
  `scripts/_driver.py` `for_each`; `scripts/_stages.py` `PIPELINE`; `scripts._provenance`.
- Produces (driver-private):
  - no match list of its own: the units are `sorted(PUBLIC_CORPUS["skillcorner"])` from `scripts._corpus`
    (20 matches, spec r7.3), so no second hand-typed list exists;
  - `ROLE_VERTICAL_GROUP: Mapping[str, str]`, which maps each acronym to `"back"`, `"mid"`, or `"fwd"`. This is the
    committed truth table. The side comes from the first letter (`L`/`R`).
  - `scored_population(roster) -> frozenset[str]`, the spec predicate;
  - `modal_vertical(labels) -> str` (mode of the 5-level label, then the group), `modal_side(labels) -> str` (side
    first, then the mode). A tie goes to the label seen first in frame order.
  - `block_permutation_null(scored, *, cell, n_draws=10_000, seed=NULL_SEED) -> dict` with `agreement`, `n`,
    `null_p95`, `p`, `wilson_low`, and `wilson_high`. It uses `rng.permutation` per (match, team) block, inline
    (spec W5.7).
  - `reflect_frames(frames) -> pd.DataFrame`: `x → 105 − x` and `y → 68 − y` in float64, with
    `team_attacking_direction` swapped. The output `x` and `y` columns are float64. It does not use
    `silly_kicks.reflection.reflect`, which stores the result back at the column dtype (`reflection.py:355`). It
    lives in `scripts/_frame_reflection.py` (Task 11; TB-PLAN-37), and the driver imports it (A11-PLAN-91).
  - `gate(metrics) -> list[str]`, the failed rules (spec rules 1–3), empty on a pass;
  - the CLI `python -m scripts.validate_shape_graph_roles run --out <dir>`. It exits 1 when `gate` fails.
- `PIPELINE`: an `ItemStage` with one single-threaded unit per match, which writes the per-player scores, the
  per-team-frame mirror result, and the diagnostics. A reduce in sorted match-key order then runs the null and the
  gate and writes `metrics.json`. Follow the compute-pipelines skill.

- [ ] **Step 1: Role census.** Through pining, fetch only `match.json` for the 20 matches (no tracking). List every
  `player_role.acronym`. Write `ROLE_VERTICAL_GROUP` to cover them, with the handoff's back and forward sets
  (`repro_shape_graph_roles.py:91-92`). Every other outfield acronym maps to `"mid"`, and each one is listed
  explicitly. Record the census in the session notes, and as `ROLE_CENSUS` in the test module.
- [ ] **Step 2: Write the failing tests:**
  - `test_role_table_covers_census` and `test_unknown_acronym_raises` (fail-closed, AG4);
  - `test_population_predicate`: `GK`, `SUB`, a null `start_time`, and a player with no labelled row are excluded.
    A substitute who came on with a real role is scored.
  - `test_modal_vertical_mode_then_group`: B 40 %, DM 30 %, M 30 % gives `"back"`;
  - `test_modal_side_side_then_mode` and `test_tie_first_seen_wins`;
  - `test_block_null_never_crosses_blocks`, `test_block_null_seeded_reproducible`, and `test_p_value_formula`
    (`(1 + #{null ≥ obs}) / (1 + draws)`);
  - `test_wilson_interval_known_values`;
  - `test_gate_each_rule_fails_alone`: three synthetic metrics sets, each of which fails one rule only;
  - `test_reflect_frames_exact`: the output `x`/`y` dtype is float64. Reflecting twice gives float64 values equal
    to the float32 inputs, bitwise. The direction is swapped.
  - `test_mirror_report_names_team_frame`: an injected difference is reported with its key and a face-set flag;
  - `test_baseline_arm_uses_oracle_with_frame_direction`;
  - `test_frame_id_is_raw_frame`: on a SkillCorner sample fixture, `frame_id` equals the raw `frame`, so the sampling
    on `frame_id % 10 == 0` is the handoff's sampling;
  - `test_driver_registered`, and the driver passes `tests/scripts/test_compute_contract.py`, including the W1N1
    vs W2N2 parity.
- [ ] **Step 3: Run them.** Expected: FAIL.
- [ ] **Step 4: Implement** spec W5.7 on these interfaces:
  - load periods 1–2, then sample `frame_id % 10 == 0` right after the load;
  - the final arm is `compute_shape_graphs(...).players` filtered on `shape_graph_source == "ok"`;
  - the baseline arm is the oracle per team-frame, with the direction from the frame label;
  - the mirror arm is `compute_shape_graphs(reflect_frames(frames))`;
  - the diagnostics are the confusion counts, the pitch-thirds counts, the lateral-fallback counts (and those with
    n ∉ {5, 10}), and the tied players per side;
  - the provenance is `require_clean_tree(prov, allow_dirty=True)`, with `run_tree_dirty` and the scoped code
    identity stamped, as in the W0 driver (Task 2). `metrics.json` also stamps `oracle_closure_digest()` (Task 1),
    because the baseline arm imports `tests/oracles/shape_graph_v4139.py`, which is outside `CODE_IDENTITY_PATHS`
    (TB-PLAN-37). The memo checks it against `FROZEN_ORACLE_CLOSURE_DIGEST`.
- [ ] **Step 5: Run them again.** Expected: PASS. The real run happens in Task 15 step 7b, after the freeze.

### Task 12: W6 — `detect_line_breaking` array pass and independence (spec §4.6)

**Files:** Modify `silly_kicks/tracking/_line_breaking.py:66-292`. Test:
`tests/tracking/test_line_breaking_independence.py`.

The W6 two-game detection on `links=None` and on caller-supplied `links=` is the Task 3 game-split property (both
variants), which goes green in Task 5.

- [ ] **Step 1: Write the tests.**
  - `test_bit_identical_single_game_unique` against `tests/oracles/line_breaking_v4139.py`, on seeded fixtures,
    with every NaN route: NaN actor, non-pass, unresolved direction, and no opponents → 0.
  - `test_chunk_property[seed]`: for random chunkings, including game-splitting ones, the concatenated chunks equal
    the whole call in values (`assert_bits_equal` on values; warnings ignored).
  - `test_gamestate_duplicates_each_row_own_result`.
  - An `assert_subquadratic_growth` guard over actions, registered in `SCALE_GUARDED`.
- [ ] **Step 2: Run them.** Expected: the three value tests PASS, because Task 5 already made the joins correct. The
  subquadratic guard FAILS, because `drop_duplicates` (`:147`) and `iterrows` (`:196`) are still there. That guard is
  this task's red.
- [ ] **Step 3: Implement the array pass.**
  - Vectorised masks for type, NaN actor, and direction.
  - One keyed lookup to the opponent group through `group_rows`.
  - A loop only over the rows that need `_straddle_core`.
  - Remove the `drop_duplicates("action_id")`.
- [ ] **Step 4: Run them again.** Expected: PASS.

### Task 13: W7 — `smooth_frames` / `derive_velocities` speed-up (spec §4.7)

**Files:** Modify `tracking/preprocess/_velocity.py`, `_smoothing.py`, and `_kalman.py`. Test:
`tests/tracking/preprocess/test_preprocess_parity.py`.

**Interfaces:** the public signatures are unchanged. Add an internal `_kalman_covariance(mask: np.ndarray, dt:
float, cfg: PreprocessConfig) -> _KalmanCov`, memoised per call by `(dt, cfg-key, mask.tobytes())`.

- [ ] **Step 1: Write the failing tests** against `tests/oracles/preprocess_v4139.py`.
  - `test_derive_velocities_bit_identical[method]`, over `savgol`, `butterworth`, `ema`, and `kalman`. It checks
    all nine columns plus the guard's NaNs and warnings.
  - `test_smooth_frames_bit_identical[method]`, checking `x_smoothed`, `y_smoothed`, and `_preprocessed_with`.
  - `test_kalman_cov_reuse_guard`: a `call_counter` on the covariance recursion shows at most one recursion per
    distinct (dt, mask) on a fixture where many players share a mask.
  - Two-game frames give per-game results (TF-65 §4.1a stays).
- [ ] **Step 2: Run them.** Expected: the parity tests PASS (unchanged code) and the reuse guard FAILS.
- [ ] **Step 3: Implement the by-construction items.**
  - W7.1: covariance memoisation.
  - W7.3: `group_rows` positions instead of `groupby(...).groups`, one sort, and no `list(idx)`.
  - W7.4: only where W0 shows a material share.

  W7.2 (the numba Kalman) is conditional on the base run showing that the data-dependent recursion still
  dominates after W7.1. If built, it is the shipping default, with its Task 9
  parity test.
- [ ] **Step 4: Run them again,** plus `tests/test_preprocess_baseline_integrity.py`. Expected: PASS.

### Task 14: W8 — DFL period, time, actor derivation; parser single-sourced; R2 qualifier fix (spec §4.8)

**Files:**
- Create: `silly_kicks/providers/sportec/_event_derivation.py`
- Modify: `silly_kicks/providers/sportec/parse.py` (`:756-1080`, `:1370-1395`), and
  `silly_kicks/providers/sportec/__init__.py` to export the new names.
- Test: `tests/providers/sportec/test_event_derivation.py`, `tests/providers/sportec/test_parse_bronze_identity.py`,
  `tests/providers/sportec/test_dfl_end_to_end.py`

**Interfaces:**
- Produces (public, `silly_kicks.providers.sportec`, each with an `Examples` section):
  - `derive_dfl_event_periods(event_time: pd.Series, event_type: pd.Series, kickoff_game_section: pd.Series) -> tuple[pd.DataFrame, DflPeriodReport]`;
  - `DflPeriodReport`, a frozen dataclass with `kickoff_times: dict[int, datetime]`, `n_pre_kickoff: int`,
    `n_unparseable_time: int`, `n_incomparable_time: int`, `unrecognized_sections: tuple[str, ...]`, and
    `skipped_kickoffs: tuple[int, ...]`;
  - `DflEventTimeWarning(UserWarning)` and `DflGameSectionWarning(UserWarning)`;
  - `resolve_dfl_event_actors(events: pd.DataFrame, match_info: MatchInfo) -> pd.DataFrame`, with the columns
    `player_id`, `team`, and `team_id_native`;
  - `dfl_event_column(event_type: str, attribute: str, *, nested: str | None = None) -> str`;
  - `dfl_actor_columns(event_type: str) -> tuple[str, ...]`;
  - `DFL_EVENT_TYPE_PREFIX: Mapping[str, str]` and `DFL_NESTED_PREFIX: Mapping[str, str]` (read-only
    `MappingProxyType`);
  - `complete_dfl_event_bronze(events: pd.DataFrame, match_info: MatchInfo) -> tuple[pd.DataFrame, DflPeriodReport]`.
- Internal: `_dfl_period_rule(times: list[datetime | None], kickoffs: list[tuple[int, datetime]]) -> list[tuple[int | None, datetime | None]]`
  and `_dfl_actor_rule(event_type: str, actor_attrs: Mapping[str, str]) -> str`.
- The KickOff contract columns are `kickoff_play_present`, `kickoff_first_play_player`, and `kickoff_player`.

- [ ] **Step 1: Write the failing tests,** with spec §4.8's acceptance list as test names.
  - `test_latest_kickoff_rule_nonmonotone`, `test_equal_kickoff_times_first_wins`, and
    `test_two_pass_first_half_event_after_second_kickoff`.
  - `test_pre_kickoff_counted_not_warned` and `test_unparseable_and_incomparable_warned_once`.
  - `test_unrecognized_section_warns_and_reports` and `test_unparseable_kickoff_skipped_next_wins`.
  - `test_round_half_way_python_semantics`.
  - `test_string_vs_datetime64_bit_equal[utc, +02:00, sub_us]`. The sub-µs case uses the datetime64 path only and
    asserts the `.dt.floor("us")` rule.
  - `test_py310_rejects_Z_and_7_digits_as_unparseable`, which runs only on Python 3.10.
  - The actor cases: every `_PLAYER_ATTR_ORDER` row, an unprefixed type, a KickOff with two Plays, a KickOff
    without a Play, a Play without a Player, a missing `kickoff_play_present` (raises), and an unknown player.
  - `test_qualifier_team_dtype_mismatch_detection`: CLU ids int vs str. It fails on the current `==`
    (`parse.py:1370-1395`). It is the R2 detection test.
  - `test_parse_dfl_events_emits_no_new_warning`.
  - `test_parse_bronze_identity_sample`. On the committed slice `tests/datasets/sportec/idsse_slice/events.xml` and
    `info.xml`, the bronze equals `tests/oracles/sportec_parse_v4139.parse_dfl_events` under `assert_bits_equal`,
    with row order and dtypes. The existing goldens (`idsse_parse_bronze_golden.parquet` and
    `idsse_events_bronze_golden.parquet`, consumed by `test_parse_port_parity.py`) must stay green unchanged.
  - `test_end_to_end_consumer_bronze_equals_parser_path`.
  - `test_shootout_period_5`: Review Focus 4.
  - `test_empty_series`: Review Focus 1.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement as spec §4.8 W8.1–W8.6 sets out.**
  - The parser does one XML pass and calls `_dfl_actor_rule` on the XML attributes and `_dfl_period_rule` on the
    collected times.
  - It drops exactly today's rows and logs the counts.
  - Delete `_scan_kickoff_times`, `_derive_period_from_kickoffs`, and the inline time block.
  - `_resolve_idsse_team_from_qualifiers` uses `id_compat`.
- [ ] **Step 4: Run them again,** plus the existing `tests/providers/sportec/` and the doctest leg. Expected: PASS.

### Task 15: Pre-freeze checks, code freeze, W0 after-run, corpus parity, CPU-class gates, memo (spec §4.0, §5.2, §9 steps 8–9)

**Files:**
- Modify (before the freeze): the `compute_pitch_control_batch` docstring (`<PEAK>`); any oracle copy that the
  Step 1 check deletes.
- Create: `docs/research/tracking-compute-perf/README.md` (the memo) and `docs/research/tracking-compute-perf/*.json`
  (summaries only).
- Create: `docs/research/shape-graph-positions/README.md` (the W5.7 memo) and `metrics.json` (summaries only).

- [ ] **Step 1: Oracle minimality check** (TB-PLAN-10, TB-PLAN-16). For each copied function F in the Task 1
  inventory, a copy may be deleted only when **the whole `call_graph` closure of live F** (functions, kernels, and
  referenced globals) has the same source as at `BASE_SHA`. Then:
  - delete the copy and import live F;
  - recompute the closure digest. It must equal the digest computed over the same closure at `BASE_SHA`. Never
    re-freeze it to a new value;
  - record the deletion in the session notes.

  Then run the local gate. Expected: PASS.
- [ ] **Step 1b: Placeholder check.** `git grep -nE "X\.Y\.Z|ADR-NNN|PR-Snnn" -- <CODE_IDENTITY_PATHS>` gives no
  output (Global Constraints).
- [ ] **Step 2: Measured peak, before the freeze** (spec W3, clarified in r6.2). Run the driver (`--phase final`)
  for the largest match on the DGX (spec R22), on the WHOLE match: no `--half-matches` (A23-PLAN-120). This run
  is not a W0 leg; the docstring and the memo state that the bound comes from the whole gradientsports 10517. Write the rounded-up `memory`-unit (`tracemalloc`) peak per backend from
  `memory.json` into the `<PEAK>` placeholder (spec R18 rule 2). Run the doctest leg. Expected: PASS.
- [ ] **Step 3: Freeze.** (Spec R23: `compare` computes the two package hashes, so no file holds them.) From here on, any edit under `CODE_IDENTITY_PATHS` (less `CODE_IDENTITY_EXCLUDE`) starts
  this task again from Step 1b (the placeholder check), because the edit can add a placeholder or move the
  measured peak. Record `git_tree_hash(paths=CODE_IDENTITY_PATHS)` as `FREEZE_ID`.
- [ ] **Step 4: After-run on HF (spec R22, R23).** With the owner's yes (billed), run the staged final campaign
  (`tracking-compute-w0-final.toml`: `preflight`, then `launch --budget-s 7200`, then `retrieve --staging
  D:\Temp\tbdfl\final-hf`, each with `--stage-tree`), on the same matches as the base, with the same
  `--profile-matches` and `--half-matches` (spec R24). Then run
  `python -m scripts.measure_tracking_compute compare D:\Temp\tbdfl\base-hf D:\Temp\tbdfl\final-hf --base-sha
  BASE_SHA`. If `compare` refuses a CPU difference between the HF legs (A20-PLAN-108), stop and ask the owner
  for the maximum number of relaunches. Then, with **one owner yes per billed relaunch** (TB-PLAN-61), rerun the
  base leg from the retained base archive (`tracking-compute-w0-base.toml launch --stage-tree <T_base>
  --replica <N>`), retrieve it into a fresh folder (`D:\Temp\tbdfl\base-hf-r<N>`), and compare again, until the
  CPUs match or the maximum is reached. Never rerun the final leg from a changed tree.
  Expected:
  - exit code 0, which covers the pitch control, shape graphs (the graph bitwise, and the labels under the W5.6
    relations, Task 2), line breaking, preprocessing, the five pre-existing pointer columns, and every W9.7 member;
  - an "unexplained" label difference is not fixed by hand. **Stop and bring it to the owner** with its keys;
  - every `memory`-unit peak at or below the docstring bound. The RSS values are not checked against it (spec R18
    rule 3);
  - every stamp equal to `FREEZE_ID`.
- [ ] **Step 5: After-run on arm64**, on the Mac Mini and on the DGX, with the frozen tree shipped byte-identical, with
  the base command of Task 3 step 5 with `--phase final` and `--parity-tests` added, and `--out` set to
  `<scratch>/final-dgx` or `<scratch>/final-arm64` (never the base folder, TB-PLAN-65); nothing else changes
  (TB-PLAN-64, A24-PLAN-121). `leg_cut` sorts the stem lists, so their order never causes a refusal.
  Compare `<scratch>/base-arm64` with `<scratch>/final-arm64` (Mac Mini) and `<scratch>/base-dgx` with
  `<scratch>/final-dgx` (DGX), each with `compare … --base-sha BASE_SHA`. Expected: the same as
  Step 4.
- [ ] **Step 6: CPU-class gates** for each gated candidate that Tasks 7, 10, or 13 built. The gate is its parity
  tests plus the Steps 4–5 base-vs-final compare. Run it on HF (AMD x86; the parity tests run as
  the final leg's `parity_tests` unit), the Mac Mini, and the DGX (arm64), all before the commit (spec R22, R23).
  - If it passes on all three machines, it ships, because the frozen tree already has it as the default.
  - Otherwise, revert it to the numpy reference, record its measured gain, and start again from Step 1b.
- [ ] **Step 7: Corpus passes.**
  - W8 bronze identity on every DFL corpus match, and the datetime64-vs-string pass. Expected: equal.
  - The W9.6 game-split property on two real matches combined.
  - Record the census counts and the fingerprint cost per match. If the cost is above 2 % of any measured step,
    **stop and bring it to the owner.**
- [ ] **Step 7b: W5.7 role gate on the DGX** (spec W5.7, R15, R22), with the frozen tree shipped byte-identical. First state the expected peak memory to the owner
  (one SkillCorner match without velocity, sampled after the load; W0.5). Then run
  `python -m scripts.validate_shape_graph_roles run --out <scratch>/roles`. Expected: exit code 0 (all three rules
  pass in all four cells), the stamp equals `FREEZE_ID`, and the stamped `oracle_closure_digest` equals
  `FROZEN_ORACLE_CLOSURE_DIGEST` (TB-PLAN-38). A digest mismatch also stops at the owner. On a failure, **stop and bring it to the owner.**
  Never change a rule or a threshold (spec W5.7, §5.3).
- [ ] **Step 7c: Write the W5.7 memo** in `docs/research/shape-graph-positions/`:
  - the four cells, with agreement, Wilson interval, `n`, `null_p95`, `p`, and the baseline;
  - the mirror result, and the face-set flag of each failure;
  - the diagnostics (spec W5.7), including the tied players and the lateral-fallback counts;
  - the baseline on 1886347, 1899585, and 1925299 against the handoff's variant-B numbers, with every difference
    explained (the frame path and the population);
  - the away vertical gap: the cause if found. A defect goes to the owner before any fix.
- [ ] **Step 8: Write the memo.** It holds:
  - the before/after table per step and provider;
  - the machines, CPU classes (`cpu_features`), and versions;
  - `BASE_SHA`, the base package hash, and `FREEZE_ID`;
  - the red-at-base counts by kind (Task 3);
  - the corpus max abs diff per kernel and CPU class (`kernel_diff.json`);
  - the peak memory per backend;
  - the gates that passed or failed, with the gains of dropped candidates;
  - the census and the fingerprint cost;
  - each leg's tree (the DGX and Mac base legs: `a962124f1dfa`; the HF legs: their staged trees) and the R25-R27
    history of the base legs;
  - the R21 pilot projection: it scales each unit linearly, so it overestimates `derive_velocities` and the kalman
    smoothing (about 700 s projected against about 90 s on HF, TB-PLAN-66). The measured legs replace it.

### Task 16: Docs, ADR, CHANGELOG, handback (spec §6)

No file under `CODE_IDENTITY_PATHS` changes in this task, except `silly_kicks/_version.py` at commit-prep (R11).

**Files:**
- Create: `docs/superpowers/adrs/ADR-NNN-tracking-batch-multigame-linking-dfl-events.md`. The number is assigned at
  commit-prep (Task 17 step 3). Only docs outside the scope cite it. It has an "Approved value changes" section for R1/R6, R2, R5/R7/R9, and R14.
  The R14 part gives D-lat, D-eqf, and D-tree, the W5.6 relations, the −x lateral exact-tie class, and the W5.7
  result. The ADR amends
  ADR-008/105/068/031 and supersedes ADR-045 D5. ADR-128 is amended in place for R27 (Task 3D).
- Modify: `docs/superpowers/adrs/ADR-045-vector-reflection-consistency.md`, adding a "D5 superseded by ADR-NNN"
  pointer at D5 (`:78`).
- Modify:
  - `CHANGELOG.md`: the `PR-Snnn` entry, with every item listed in spec §6, including the R14 behaviour change
    (with the −x lateral exact-tie class, TB-PLAN-33), the new `ValueError` of `infer_positions`, the R17 fix,
    the R20 executor fix (A14-PLAN-101), and R25-R27: the W0 warm-up gate, the `tc3_download` dataset with the
    `datasets fetch` CLI fix (a defect since #288), and the HTTP store read path with its measurements (TB-PLAN-71);
  - `AGENTS.md`: the TF-7 and TF-39 rows, the Providers line, and the backends pointer, within
    `tests/test_agents_md_budget.py`;
  - `docs/context/tracking-features.md`, `providers.md`, and `conventions-core.md`;
  - `docs/PRIVATE_CONSUMERS.md`: the `_frame_index` and `_link_identity` rows;
  - `docs/c4/architecture.dsl` and `.html`, if the C4 pin test requires it;
  - `NOTICE`: no change, because Sotudeh already exists.
- Create (outside the repo): `D:\Development\_handoffs\2026-11-XX-sk-to-lakehouse-tracking-batch-handback.md`, with
  every item in spec §6's handback list.

- [ ] **Step 1: Write the docs.**
- [ ] **Step 2: Run the full local gate.** Expected: exit code 0 for each command. Show the real exit codes.
  Check that `git_tree_hash(paths=CODE_IDENTITY_PATHS)` still equals `FREEZE_ID`.

### Task 17: Final review, A/B implementation review, owner-gated single commit

- [ ] **Step 1: Run `/final-review`** (mad-scientist-skills:final-review). Fix everything it finds. A fix under
  `CODE_IDENTITY_PATHS` restarts Task 15 from Step 1b (all of Steps 1b–8). Then re-run Task 16 step 2.
- [ ] **Step 2: Send the A/B implementation review.** Message `lakehouse (Skunk): Works` (A) and `(D32):Review B-O`
  (B) with `/review-impl` on branch `feat/tracking-batch-dfl-events` (uncommitted tree). Name the spec, this plan,
  the memo, and the spec caller table. The reports go to
  `D:\Development\_reviews\2026-11-XX-tracking-batch-dfl-events-impl-{A,B}.md`.
  - Wait for both. Fix or answer every finding, and take any A/B disagreement to the owner.
  - The Step 1 restart rule applies to every review fix.
  - Run re-review rounds until both approve.
- [ ] **Step 3: Commit prep.**
  - Assign the version number (a minor bump in `silly_kicks/_version.py`, outside the identity by R11), the ADR
    number (rename the file and fill the references), and the `PR-Snnn` key. Replace the docs placeholders. Only
    out-of-scope files change, because of the placeholder rule.
  - Stage the cycle: `git add -A`. The index then holds the whole cycle, which is what the commit will contain.
  - Check that `git_tree_hash(paths=CODE_IDENTITY_PATHS)` equals `FREEZE_ID`. Right after `git add -A`, the working
    tree equals the index, so this working-tree hash is the staged content (review TB-IMPL-87). A mismatch means
    that code changed after the after-run. Restart Task 15 from Step 1b.
  - Run the full local gate once more.
  - Show the owner the complete `git diff --cached --stat`, the file list, and the commit message (Conventional
    Commits, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`).
  - **Stop and wait for an explicit yes.**
- [ ] **Step 4: After the yes, make the commit and push** (one yes covers the commit and the push, per memory). A
  PR, a tag, or a merge each needs its own yes.
- [ ] **Step 5: Ask for the PR.** Show the owner the PR title and body. **Stop and wait for an explicit yes.** After
  the yes, open the PR. `ci.yml:3-7` runs CI only on pull requests to `main` and on pushes to `main`, so the PR is
  what gives the cycle commit its CI run. Wait for CI to pass. Expected: green.
- [ ] **Steps 6–7 are dropped (spec R23).** The HF leg and its revert path run before the commit, in Task 15 steps
  4 and 6. After CI is green, ask the owner for the merge, which needs its own yes.
